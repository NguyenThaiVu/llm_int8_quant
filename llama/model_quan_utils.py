import os 
import torch
import torch.nn as nn
import gemm_cutlass

from quant_utils import quantize_row_int8_symmetric_nd, quantize_tensor

MAX_SEQ_LEN = 2100 # 640 or 1280 or 2560

class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=MAX_SEQ_LEN):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
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
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len]
            requant_scale = scale_x * self.scale_w / scale_y_value
            requant_scale = requant_scale.to(torch.float32)
            
            if x.dim() == 3:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(
                    x, self.weight_q, requant_scale
                )
            elif x.dim() == 2:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale(
                    x, self.weight_q, requant_scale
                )
            else:
                raise ValueError("Input must be 2D or 3D tensor")
            return out_q, scale_y_value
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_tensor(self.weight)
        
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        
        # Release original weight to save memory
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
        if not self.is_quantized:  # Calibration mode
            assert x_q.dtype == torch.float32 or \
                    x_q.dtype == torch.bfloat16 or \
                    x_q.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x_q, dim=-1)
            self.out_observer(out)
            return out, 1.0
        else:   # Quantized mode            
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
        if not self.is_quantized:  # Calibration mode
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
            # Quantized mode
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
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
                
                # scale_A = scale_A[:seq_len].to(torch.float32)
                # scale_B = scale_B[:seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
                
                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            elif A.dim() == 3:
                batch_size, seq_len, _ = A.shape
                
                # scale_A = scale_A[:, :seq_len].to(torch.float32)
                # scale_B = scale_B[:, :seq_len].to(torch.float32)
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
        
        # Always store as (max_batch, max_seq_len) 
        self.register_buffer("max_val", torch.full((max_batch, max_seq_len), -torch.inf)) 
        self.register_buffer("min_val", torch.full((max_batch, max_seq_len), torch.inf)) 
        
    @torch.no_grad() 
    def forward(self, x: torch.Tensor): 
        """ 
        x can be: - (T, H) -> treated as B=1 - (B, T, H) -> true token-wise per batch 
        """ 
        if x.ndim == 2: 
            T, _ = x.shape 
            B = 1 
            xd = x.detach().unsqueeze(0) # (1, T, H) 
        elif x.ndim == 3: 
            B, T, _ = x.shape 
            xd = x.detach() 
        else: raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D with shape {tuple(x.shape)}") 
        
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
        qmax = 127 # symmetric int8 
        if T == None: 
            T = self.max_seq_len 
        
        # if input 2D, return (T,) scale; 
        if self.max_batch == None or self.max_batch == 1: 
            max_abs = torch.maximum(self.max_val[0, :T].abs(), self.min_val[0, :T].abs()) 
            return (max_abs / qmax).clamp(min=1e-8) # (T,) 
        # if input 3D, return (B,T) scale; 
        max_abs = torch.maximum(self.max_val[:B, :T].abs(), self.min_val[:B, :T].abs()) 
        return (max_abs / qmax).clamp(min=1e-8) # (B, T)
    

class PerChannelAbsMaxObserver(nn.Module):
    """Tracks max(abs(x)) per last dimension (channel)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("amax", torch.zeros(num_channels), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # x: [*, C] -> reduce over all dims except last
        c = x.shape[-1]
        assert c == self.amax.numel()
        x_abs = x.abs()
        # reduce over all dims but last
        dims = list(range(x_abs.dim() - 1))
        cur = x_abs.amax(dim=dims)
        self.amax = torch.maximum(self.amax, cur)

    def get_amax(self):
        return self.amax


class Custom_Linear_PerRow(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=MAX_SEQ_LEN):
        super(Custom_Linear_PerRow, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        # Weight quantization
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer('scale_w', torch.ones(out_features))
        
        # Smooth quantization
        self.lambd = 0.5
        self.register_buffer('smooth_alpha', torch.ones(in_features), persistent=False) 
        self.in_observer = PerChannelAbsMaxObserver(in_features)
        
        # Output quantization
        self.register_buffer('scale_y', torch.ones(max_seq_len))
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        
        self.is_quantized = False
        
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode    
            
            # Calibrate activation statistics for SmoothQuant
            self.in_observer(x)
             
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)  
            
            if x.dim() == 2:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    x, self.weight_q, 
                    scale_x, self.scale_w, scale_y_value
                )
            else:
                raise ValueError("Input must be 2D tensor")
            return out_q, scale_y_value
        
    def finish_calibration(self):
        # activation per-channel max (FP32, correct device)
        # max_a = self.in_observer.get_amax().to(device=self.weight.device, dtype=torch.float32)
        max_a = self.in_observer.get_amax() # shape: (in_features,)
        max_a = torch.clamp(max_a, min=1e-6)

        # weight per-input-channel max (columns)
        max_w = self.weight.detach().abs().amax(dim=0).to(torch.float32)
        max_w = torch.clamp(max_w, min=1e-6)

        alpha = (max_a ** self.lambd) / (max_w ** (1.0 - self.lambd))
        alpha = torch.clamp(alpha, min=0.01, max=100.0)

        self.smooth_alpha.copy_(alpha)  

        # weight smoothing consistent with "x / alpha"
        w_smooth = self.weight * alpha.unsqueeze(0)    
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(w_smooth)

        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        

class Custom_FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = Custom_Linear_PerRow(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=MAX_SEQ_LEN)
        self.fc2 = Custom_Linear_PerRow(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=MAX_SEQ_LEN)
        self.fc3 = Custom_Linear_PerRow(cfg["hidden_dim"], cfg["emb_dim"], max_seq_len=MAX_SEQ_LEN)
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            x_fc1, _ = self.fc1(x, 1.0)
            x_fc2, _ = self.fc2(x, 1.0)
            # x_silu, _ = self.custom_silu(x_fc1, 1.0)
            x_silu = torch.nn.functional.silu(x_fc1)
            
            # x, _ = self.custom_elementwise_mul(x_silu, 1.0, x_fc2, 1.0)
            x = x_silu * x_fc2
            
            out, _ = self.fc3(x, 1.0)
            return out, 1.0
        else:
                                       
            # ===== 1. Compute FC1 =====    
            smooth_factor = self.fc1.smooth_alpha.to(x.device) # shape: (in_dims,)
            X_smooth = x / smooth_factor.unsqueeze(0)
            X_smooth_q, scale_x_smooth = quantize_row_int8_symmetric_nd(X_smooth)
            x_fc1_int8, scale_fc1 = self.fc1(X_smooth_q, scale_x_smooth)
            
            # ===== 2. Compute FC2 =====
            smooth_factor = self.fc2.smooth_alpha.to(x.device) # shape: (in_dims,)
            X_smooth = x / smooth_factor.unsqueeze(0)
            X_smooth_q, scale_x_smooth = quantize_row_int8_symmetric_nd(X_smooth)
            
            x_fc2_int8, scale_fc2 = self.fc2(X_smooth_q, scale_x_smooth)
            
            # === 3. Compute SiLU Quantization (x = silu(x_fc1) * x_fc2) === 
            x_int8, x_scale = gemm_cutlass.func_silu_mul_int8(
                                            x_fc1_int8, scale_fc1,
                                            x_fc2_int8, scale_fc2)
            
            x = x_int8.to(torch.float32) * x_scale.unsqueeze(-1)
            x = x.to(torch.bfloat16)
            
            # ===== 5. Compute FC3 ======            
            smooth_factor = self.fc3.smooth_alpha.to(x.device) # shape: (in_dims,)
            X_smooth = x / smooth_factor.unsqueeze(0)
            X_smooth_q, scale_x_smooth = quantize_row_int8_symmetric_nd(X_smooth)
            
            out = gemm_cutlass.func_int8_matmul(X_smooth_q, self.fc3.weight_q, 1.0)
            out = out * scale_x_smooth.unsqueeze(-1) * self.fc3.scale_w.unsqueeze(0)
            
            return out, 1.0
        
    def finish_calibration(self):
        self.fc1.finish_calibration()
        self.fc2.finish_calibration()
        self.fc3.finish_calibration()
        self.is_quantized = True