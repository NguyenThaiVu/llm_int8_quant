import os 
import torch
import torch.nn as nn
import gemm_cutlass

from utils_quant import quantize_row_int8_symmetric_nd, quantize_tensor

MAX_SEQ_LEN = 2112 # 576 or 1040 or 2112

class Custom_Linear(nn.Module):
    """
    Linear layer with two execution modes:
    1. Calibration mode
        - Input: bf16
        - Weight: bf16
        - Output: bf16
        - Behavior: computes `x @ weight.T` and updates output observer
    
    2. Quantization mode
        - Input: int8
        - Weight: int8
        - Output: int8
        - Behavior: computes quantized matmul with per-row scaling
                    and returns quantized output and its scale
    
    This layer has: per-row scale for activation and 
                    per-tensor scale for weight. 
                    The output scale is per-row
                    
    Input shapes: (M, K) or (B, M, K)
    Weight shapes: (N, K)
    Output shapes: (M, N) or (B, M, N)
    """
    def __init__(self, in_features, out_features, 
                 max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 
                                               dtype=dtype))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        
        self.register_buffer('scale_w', torch.tensor(1.0))
        self.register_buffer('scale_y', torch.ones(out_features))

        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode 
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8,\
                "Expected int8 input in quantization"
            
            seq_len = x.shape[0]
            out_dim = self.weight_q.shape[0]
            scale_y_value = self.scale_y[:seq_len]
            
            row_scale = scale_x / scale_y_value  
            col_scale = self.scale_w.expand(out_dim)
            
            out_q = gemm_cutlass.func_w8a8o8_matmul(x, self.weight_q,\
                row_scale, col_scale)

            return out_q, scale_y_value
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_tensor(self.weight)
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        
        self.is_quantized = True  
        del self.weight
        torch.cuda.empty_cache()
        
        
class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
        self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len))
        
        self.is_quantized = False
        
    def forward(self, x_q, scale_x):
        if self.is_quantized == False: 
            assert x_q.dtype == torch.float32 or \
                    x_q.dtype == torch.bfloat16 or \
                    x_q.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x_q, dim=-1)
            self.out_observer(out)
            return out, 1.0
        else:          
            seq_len = x_q.shape[-1]
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=x_q.device))
            
            scale_x_value = scale_x[:, :seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)

            out_q = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x_q, scale_x_value.view(-1),
                scale_out_value.view(-1), mask
            )
            
            return out_q, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True  
        
        
class Custom_RMSNorm(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=MAX_SEQ_LEN, dim=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # learnable weight of RMSNorm
        self.eps = eps
        self.dim = dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len 
        
        if self.num_heads == 1 or self.num_heads == None:
            self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
            self.register_buffer('scale_out', torch.ones(max_seq_len))
        elif self.num_heads > 1:
            self.out_observer = MinMaxObserverPerLastDim(self.num_heads, max_seq_len=self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len))
        else:
            raise ValueError("num_heads must be >= 1")
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:  
            assert x.dtype == torch.float32 or \
                    x.dtype == torch.bfloat16 or \
                    x.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
                    
            mean_square = x.pow(2).mean(-1, keepdim=True)  
            inv_rms = torch.rsqrt(mean_square + self.eps)   # [seq_len, 1]
            out = x * inv_rms * self.weight  # [seq_len, head_dim]
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expect int8 in quantized mode"
            
            if x.dim() == 2:
                seq_len = x.shape[0]
                scale_x_value = scale_x[:seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
            elif x.dim() == 3:
                seq_len = x.shape[1]
                scale_x_value = scale_x[:, :seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)
        
            y_q = gemm_cutlass.func_rmsnorm_int8(
                x, scale_x_value, self.weight, scale_out_value, self.eps
            )
            return y_q, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True
        
        
class Custom_RoPE(nn.Module):
    def __init__(self, num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=None):
        super(Custom_RoPE, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
        self.register_buffer('scale_out',\
                    torch.ones(self.num_heads, self.max_seq_len)) 
        self.is_quantized = False
        
    def forward(self, x, scale_x, 
                    cos, scale_cos,
                    sin, scale_sin):
        origin_shape = x.shape
        origin_dtype = x.dtype
        
        num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        
        if self.is_quantized == False:
            # 1. Split x into first half and second half
            x1 = x[..., : head_dim // 2]  # First half
            x2 = x[..., head_dim // 2 :]  # Second half

            # 2. Adjust sin and cos shapes
            cos = cos[:seq_len, :].unsqueeze(0)  # Shape:  1, seq_len, head_dim)
            sin = sin[:seq_len, :].unsqueeze(0)

            # 3. Apply the rotary transformation
            rotated = torch.cat((-x2, x1), dim=-1)
            x_rotated = (x * cos) + (rotated * sin)
            
            # 4. Reshape back to original shape and dtype
            out = x_rotated.to(dtype=origin_dtype)
            out = out.view(origin_shape)  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            seq_len = x.shape[1]
            scale_x_value = scale_x[:, :seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)
            
            cos = cos[:seq_len, :]
            sin = sin[:seq_len, :]

            Y_int8 = gemm_cutlass.func_apply_rope_int8(x, scale_x_value, \
                            cos, scale_cos,
                            sin, scale_sin,
                            scale_out_value)
            return Y_int8, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        

class Custom_Matmul(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        if self.num_heads == 1:
            self.out_observer = MinMaxObserverPerLastDim(max_seq_len=self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.max_seq_len)) 
            
        elif self.num_heads > 1:
            self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len)) 
        else:
            raise ValueError(f"num_heads should be >= 1, got {num_heads}")
        self.is_quantized = False
        
    def forward(self, A, scale_A, B, scale_B):
        """
        A: (M, K)
        B: (N, K)
        C = A @ B^T -> (M, N)
        """
        
        if self.is_quantized == False:
            if A.dim() == 2:
                C = torch.matmul(A, B.T)
                self.out_observer(C)
            elif A.dim() == 3:
                C = torch.matmul(A, B.transpose(-1, -2))
                self.out_observer(C)
            return C, 1.0
        else:
            if A.dim() == 2:
                seq_len = A.shape[0]
                
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
                
                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            elif A.dim() == 3:
                batch_size, seq_len, _ = A.shape
                
                scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)

                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale_batched(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            else:
                raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().cuda()
        self.is_quantized = True
        

class MinMaxObserverPerLastDim(nn.Module): 
    def __init__(self, max_batch=1, max_seq_len=1024): 
        super().__init__() 
        self.max_batch = max_batch 
        self.max_seq_len = max_seq_len 
        
        self.register_buffer("max_val",\
                            torch.full((max_batch, max_seq_len), -torch.inf)) 
        self.register_buffer("min_val",\
                            torch.full((max_batch, max_seq_len), torch.inf)) 
        
    @torch.no_grad() 
    def forward(self, x: torch.Tensor): 
        if x.ndim == 2: 
            T, _ = x.shape 
            B = 1 
            xd = x.detach().unsqueeze(0) # (1, T, H) 
        elif x.ndim == 3: 
            B, T, _ = x.shape 
            xd = x.detach() 
        else: raise ValueError(f"Expected 2D or 3D input,\
            got {x.ndim}D with shape {tuple(x.shape)}") 
        
        if B > self.max_batch: 
            raise ValueError(f"B={B} exceeds max_batch={self.max_batch}") 
        if T > self.max_seq_len: 
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}") 
        
        # token-wise over hidden dim -> (B, T) 
        cur_max = xd.amax(dim=-1) 
        cur_min = xd.amin(dim=-1) 
        
        # update only prefix [0:B, 0:T) 
        self.max_val[:B, :T] = torch.maximum(self.max_val[:B, :T], cur_max)
        self.min_val[:B, :T] = torch.minimum(self.min_val[:B, :T], cur_min)
        return x 
    
    def get_scale(self, B=None, T=None): 
        qmax = 127.0
        if T == None: 
            T = self.max_seq_len 
        
        # if input 2D, return (T,) scale; 
        if self.max_batch == None or self.max_batch == 1: 
            max_abs = torch.maximum(\
                self.max_val[0, :T].abs(),\
                self.min_val[0, :T].abs()\
            ) 
            return (max_abs / qmax).clamp(min=1e-8) # (T,) 
        
        # if input 3D, return (B,T) scale; 
        max_abs = torch.maximum(\
                self.max_val[:B, :T].abs(),\
                self.min_val[:B, :T].abs()\
        ) 
        return (max_abs / qmax).clamp(min=1e-8) # (B, T)
    

class PerChannelAbsMaxObserver(nn.Module):
    """Tracks max(abs(x)) per last dimension (channel)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("amax", torch.zeros(num_channels))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # x: [*, C] -> reduce over all dims except last
        c = x.shape[-1]
        assert c == self.amax.numel()
        
        x_abs = x.abs()
        dims = list(range(x_abs.dim() - 1))
        cur = x_abs.amax(dim=dims) # shape: (C,)
        self.amax = torch.maximum(self.amax, cur)

    def get_amax(self):
        return self.amax

def compute_smooth_alpha(input_observer, weight, lambd=0.5):
    """
    Compute SmoothQuant alpha in full FP32 precision.

    Output:
        alpha: torch.float32 tensor of shape (in_features,)
    """

    # Activation stats
    max_a = input_observer.get_amax().to(torch.float32)  # <-- important
    max_a = torch.clamp(max_a, min=1e-6)

    # Weight stats
    max_w = weight.detach().abs().amax(dim=0).to(torch.float32)
    max_w = torch.clamp(max_w, min=1e-6)

    # Compute alpha
    alpha = torch.pow(max_a, lambd) / torch.pow(max_w, (1.0 - lambd))
    alpha = torch.clamp(alpha, min=0.01, max=100.0)
    return alpha  

class Custom_Linear_PerRow(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=MAX_SEQ_LEN):
        super(Custom_Linear_PerRow, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        # Weight quantization
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer('scale_w', torch.ones(out_features))
        
        # Smooth quantization
        self.register_buffer('smooth_alpha', torch.ones(in_features)) 
        self.in_observer = PerChannelAbsMaxObserver(in_features)
        
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.register_buffer('scale_y', torch.ones(max_seq_len, dtype=torch.float32))
        
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if self.is_quantized == False:  
            self.in_observer(x) # Calibrate input statistics for SmoothQuant
             
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expect int8 input in quantization"
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)
            
            out_dim = self.weight_q.shape[0]
            
            row_scale = scale_x / scale_y_value  
            col_scale = self.scale_w.expand(out_dim)
            
            # out_q = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
            #         x, self.weight_q, 
            #         scale_x, self.scale_w, scale_y_value)
            
            out_q = gemm_cutlass.func_w8a8o8_matmul(x, self.weight_q,\
                row_scale, col_scale)
            
            return out_q, scale_y_value
        
    def finish_calibration(self, alpha=None):
        if alpha is None:
            alpha = compute_smooth_alpha(self.in_observer, self.weight)

        self.smooth_alpha.copy_(alpha)  
        w_smooth = self.weight * alpha.unsqueeze(0) 
            
        # Quantize the smoothed weight
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(w_smooth)

        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        
        
class Custom_Silu(nn.Module):
    def __init__(self, emb_dim, dtype=torch.bfloat16):
        super().__init__()
        self.emb_dim = emb_dim
        self.dtype = dtype

        self.is_quantized = False
        self.is_smooth_scale = False
        self.register_buffer("smooth_scale", 
                             torch.ones(emb_dim, dtype=torch.float32))

    def forward(self, x1, scale_x1, x2, scale_x2):
        if not self.is_quantized:
            x_silu = torch.nn.functional.silu(x1)
            x = x_silu * x2
            return x
        else:
            assert x1.dtype == torch.int8 and x2.dtype == torch.int8,\
                "Expected int8 inputs in quantized mode"

            # smooth_scale = self.smooth_scale.to(torch.float32)
            smooth_scale = self.smooth_scale
            x_int8, x_scale = gemm_cutlass.func_silu_mul_int8(\
                x1, scale_x1,\
                x2, scale_x2, smooth_scale)
            
            return x_int8, x_scale

    def finish_calibration(self):
        self.is_quantized = True

    def enable_smooth_scale(self, smooth_scale):
        assert smooth_scale.shape == (self.emb_dim,)
        self.is_smooth_scale = True
        with torch.no_grad():
            smooth_value = smooth_scale.to(torch.float32)
            self.smooth_scale.copy_(smooth_value)
        
        
class Custom_FeedForward(nn.Module):
    """
    In the FNN, we use SmoothQuant for all fc1, fc2, and fc3 layers to 
    minimize the quantization error due to activation outlier.
    
    To speed up computation, we use the same smooth_alpha for 
    both fc1 and fc2, since they have the same input activation X.
    """
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = Custom_Linear_PerRow(cfg["emb_dim"], cfg["hidden_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.fc2 = Custom_Linear_PerRow(cfg["emb_dim"], cfg["hidden_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.fc3 = Custom_Linear_PerRow(cfg["hidden_dim"], cfg["emb_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.silu_layer = Custom_Silu(cfg["hidden_dim"])
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            x_fc1, _ = self.fc1(x, 1.0)
            x_fc2, _ = self.fc2(x, 1.0)
            x = self.silu_layer(x_fc1, 1.0, x_fc2, 1.0)
            out, _ = self.fc3(x, 1.0)
            return out, 1.0
        else:  
            X_smooth_q = x
            scale_x_smooth = scale_x         
            
            x_fc1_int8, scale_fc1 = self.fc1(X_smooth_q, scale_x_smooth)
            
            x_fc2_int8, scale_fc2 = self.fc2(X_smooth_q, scale_x_smooth)
            
            X_smooth_q, scale_x_smooth = self.silu_layer(x_fc1_int8, scale_fc1, 
                                                         x_fc2_int8, scale_fc2)
            
            out = gemm_cutlass.func_int8_matmul(X_smooth_q, self.fc3.weight_q, 1.0)
            out = out * scale_x_smooth.unsqueeze(-1) * self.fc3.scale_w.unsqueeze(0)
            
            return out, 1.0
        
    def finish_calibration(self):
        self.fc1.finish_calibration()
        fc1_smooth_alpha = self.fc1.smooth_alpha

        # fc2 uses the same smooth_alpha as fc1 since they have the same input
        self.fc2.finish_calibration(alpha=fc1_smooth_alpha) 
        
        self.fc3.finish_calibration()
        # Assign fc3's smooth_alpha to silu_layer's 
        fc3_smooth_alpha = self.fc3.smooth_alpha
        self.silu_layer.enable_smooth_scale(fc3_smooth_alpha)
        self.silu_layer.finish_calibration()
        
        self.is_quantized = True