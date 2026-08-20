import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import quantize_row_int8_symmetric_nd

MAX_SEQ_LEN = 2112 # 576 or 1040 or 2112
BATCH_SIZE = 1

class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=MAX_SEQ_LEN, is_return_float=False):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight quantization
        self.weight_q = torch.empty(out_features, in_features, dtype=torch.int8, device='cuda')
        self.scale_w = torch.ones(out_features, dtype=torch.float32, device='cuda')
        
        # SmoothQuant scaling factor
        self.smooth_alpha = torch.ones(in_features, dtype=torch.float32, device='cuda')
        self.in_observer = PerChannelAbsMaxObserver(in_features)
        self.input_dims = None
        
        # Output scale quantization
        self.out_observer = MinMaxObserverPerLastDim()
        self.scale_y = torch.ones(max_seq_len, dtype=torch.float32, device='cuda')
        self.is_quantized = False
        self.is_return_float = is_return_float
        
    def forward(self, x, scale_x=1.0):
        if self.is_quantized == False:  
            # Calibrate input for SmoothQuant
            self.in_observer(x) 
            self.input_dims = x.dim()
            
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expect int8 input in quantization"
            # print(f"[DEBUG] Custom_Linear forward: x shape={x.shape}, weight_q shape={self.weight_q.shape}")
            
            if x.dim() == 2:
                seq_len = x.shape[0]
                scale_y_value = self.scale_y[:seq_len]
            elif x.dim() == 3:
                seq_len = x.shape[1]
                scale_y_value = self.scale_y[:, :seq_len]
            else:
                raise ValueError(f"Unsupported input dimensions: {x.dim()}")
            
            row_scale = scale_x / scale_y_value  
            col_scale = self.scale_w
            
            if self.is_return_float:
                if seq_len == 1:
                    out = gemm_cutlass.func_i8_gemv_out_bf16(x, self.weight_q,\
                                        scale_x, self.scale_w, 1.0)
                else:
                    out = gemm_cutlass.func_w8a8_matmul(x, self.weight_q,\
                        row_scale, col_scale)
                return out, 1.0
            
            # Return int8 output (GEMV or GEMM)
            if seq_len == 1:
                out_q = gemm_cutlass.func_i8_gemv_out_i8(\
                    x, self.weight_q, scale_x, self.scale_w, scale_y_value, 1.0)
            else:
                out_q = gemm_cutlass.func_w8a8o8_matmul_fusion(x, self.weight_q,\
                    row_scale, col_scale)
            
            return out_q, scale_y_value
        
    def finish_calibration(self, alpha=None):
        if alpha is None:
            alpha = compute_smooth_alpha(self.in_observer, self.weight)
        self.smooth_alpha = alpha.to(self.weight.device)
        
        # Quantize the smoothed weight
        w_smooth = self.weight * alpha.unsqueeze(0) 
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(w_smooth)

        self.scale_y = self.out_observer.get_scale(input_ndim=self.input_dims)
        self.scale_y = self.scale_y.to(self.weight.device)
        self.is_quantized = True  
        
        # Delete weight and observers to save memory
        del self.weight
        del self.in_observer
        
        
class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.num_heads = num_heads
        
        self.is_quantized = False
        
    def forward(self, x, scale_x, mask):
        if self.is_quantized == False: 
            assert x.dtype == torch.float32 or \
                    x.dtype == torch.bfloat16 or \
                    x.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x, dim=-1)
            return out, 1.0
        else:          
            assert x.dtype == torch.int8, "Expected int8 in quantization mode"
            b, num_heads, current_seq_len, total_kv_len = x.shape
            if current_seq_len > 1:  # Only apply masking on Pre-fill stage
                out_q, scale_out = gemm_cutlass.func_softmax_lastdim_int8_masking(
                    x, scale_x, mask)
            else:  # No masking in decoding stage (seq_len=1)
                out_q, scale_out = gemm_cutlass.func_softmax_int8(
                    x, scale_x)
            return out_q, scale_out
    
    def finish_calibration(self):
        self.is_quantized = True  
        

class Custom_RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.norm_shape = (d_model,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.is_quantized = False
    
    def forward(self, x, scale_x=1.0):
        if self.is_quantized == False:
            y = F.rms_norm(x, self.norm_shape, self.weight, eps=self.eps)
            return y
        else:
            x_int8, scale_x = gemm_cutlass.func_rmsnorm_int8(x, scale_x, self.weight, self.eps)
            return x_int8, scale_x
    
    def finish_calibration(self):
        self.is_quantized = True
        
        
class RMSNorm_Fuse_Quant(nn.Module):
    """
    This module fuse (1) RMSNorm and (2) quantization into single kernel
    This module has two execution modes:
    1. Calibration mode:
    - Input:
        x: (seq_len, emb_dim) in bf16
    - Output:
        y: (seq_len, emb_dim) in bf16
    
    2. Quantization mode:
    - Input: 
        x: (seq_len, emb_dim) in int8
    - Output:
        y: (seq_len, emb_dim) in int8
        scale_y: (seq_len,) in float32
        
    In the quantization mode, we fuse the RMSNorm and quantization into a single int8 kernel, 
    The quantization account for the outlier by using smooth_scale.
    """
    def __init__(self, emb_dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=dtype))
        self.norm_shape = (emb_dim,)
        self.emb_dim = emb_dim
        
        self.is_quantized = False
        self.is_smooth_scale = False
        self.smooth_scale = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))

    def forward(self, x):
        if self.is_quantized == False:
            norm_x = F.rms_norm(x, normalized_shape=self.norm_shape,\
                                weight=self.weight, eps=self.eps)
            return norm_x
        else:
            if self.is_smooth_scale == False:
                fake_smooth_scale = torch.ones(self.emb_dim,\
                                    dtype=torch.float32, device=x.device)
                Y_int8, scale_Y = gemm_cutlass.func_rmsnorm_quant(x, self.weight,\
                                        fake_smooth_scale, self.eps)
            else:
                Y_int8, scale_Y = gemm_cutlass.func_rmsnorm_quant(x, self.weight,\
                                        self.smooth_scale, self.eps)
            return Y_int8, scale_Y
    
    def finish_calibration(self):
        self.is_quantized = True
        
    def enable_smooth_scale(self, smooth_scale):
        """
        This function apply smooth scaling to quantization (for input activations) 
        Note:
        - Normal quantization: Y_int8 = Y / scale_Y
        - Smooth quantization: Y_int8 = (Y / smooth_scale) / scale_Y
        """
        self.is_smooth_scale = True
        self.smooth_scale.data.copy_(smooth_scale)
        
        
class Custom_RoPE(nn.Module):
    def __init__(self, num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=None):
        super(Custom_RoPE, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        self.is_quantized = False

    def forward(self, x, scale_x,
        cos, scale_cos, sin, scale_sin,
        start_offset=0,
    ):
        """
        x can be:
            2D: (seq_len, head_dim)
            3D: (num_heads, seq_len, head_dim)
            4D: (batch_size, num_heads, seq_len, head_dim)

        start_offset:
            The absolute position of the first token in x.

            Without KV cache:
                start_offset = 0

            With KV cache:
                start_offset = past_kv_len
        """

        origin_shape = x.shape
        origin_dtype = x.dtype

        if x.dim() == 2:
            seq_len, head_dim = x.shape
        elif x.dim() == 3:
            num_heads, seq_len, head_dim = x.shape
        elif x.dim() == 4:
            batch_size, num_heads, seq_len, head_dim = x.shape
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}")

        assert head_dim % 2 == 0, "Head dimension must be even"

        assert start_offset >= 0, "start_offset must be non-negative"
        assert start_offset + seq_len <= cos.shape[0], (
            f"RoPE position out of range: "
            f"start_offset={start_offset}, seq_len={seq_len}, "
            f"cos.shape[0]={cos.shape[0]}"
        )

        # Select the correct absolute RoPE positions
        cos = cos[start_offset : start_offset + seq_len, :]
        sin = sin[start_offset : start_offset + seq_len, :]

        if self.is_quantized == False:
            # 1. Split x into first half and second half
            x1 = x[..., : head_dim // 2]
            x2 = x[..., head_dim // 2 :]

            # --------------------------------------------------------
            # 2. Make cos/sin broadcastable to x
            # --------------------------------------------------------
            if x.dim() == 2: # x = (seq_len, head_dim)
                pass

            elif x.dim() == 3:
                # x:   (num_heads, seq_len, head_dim)
                # cos: (1, seq_len, head_dim)
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)

            elif x.dim() == 4:
                # x:   (batch_size, num_heads, seq_len, head_dim)
                # cos: (1, 1, seq_len, head_dim)
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)

            # 3. Apply RoPE
            rotated = torch.cat((-x2, x1), dim=-1)
            x_rotated = (x * cos) + (rotated * sin)

            out = x_rotated.to(dtype=origin_dtype)
            out = out.reshape(origin_shape)

            return out, 1.0

        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            Y_int8, scale_out = gemm_cutlass.func_apply_rope_int8(
                x,
                scale_x,
                cos,
                scale_cos,
                sin,
                scale_sin,
            )

            Y_int8 = Y_int8.reshape(origin_shape)
            scale_out = scale_out.reshape(origin_shape[:-1])

            return Y_int8, scale_out

    def finish_calibration(self):
        self.is_quantized = True
        
        
class Custom_Matmul(nn.Module):
    """
    C = A @ B^T
    This module perform matmul with two execution modes:
    1. Calibration mode:
    - Input:
        A: (M, K) - bf16
        B: (N, K) - bf16
    - Output:
        C: (M, N) - bf16
    
    2. Quantization mode:
    - Input:
        A: (M, K)  - int8
        B: (N, K) - int8
        scale_A: (M,) - float32
        scale_B: (N,) - float32
    - Output:
        C: (M, N) - int8
        scale_C: (M,) - float32
    """
    def __init__(self, num_heads=1, max_seq_len=MAX_SEQ_LEN, is_return_float=False):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.is_return_float = is_return_float
        self.input_dims = None
        
        self.out_observer = MinMaxObserverPerLastDim(max_batch=BATCH_SIZE,\
            max_seq_len=max_seq_len, max_heads=num_heads, device="cuda")
        self.scale_y = None

        self.is_quantized = False
        
    def forward(self, A, scale_A, B, scale_B):
        if self.is_quantized == False:
            self.input_dims = A.dim()
            
            if A.dim() == 2:
                C = torch.matmul(A, B.T)
            elif A.dim() == 3:
                C = torch.matmul(A, B.transpose(-2, -1))
            elif A.dim() == 4:
                C = torch.matmul(A, B.transpose(-2, -1))
            else:
                raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
            self.out_observer(C)
            return C, 1.0
        else:
            if self.is_return_float:
                # print(f"[DEBUG] Custom_Matmul forward (return float): A shape={A.shape}, B shape={B.shape}")
                if A.dim() == 2:
                    seq_len = A.shape[0]
                elif A.dim() == 3:
                    seq_len = A.shape[1]
                elif A.dim() == 4:
                    seq_len = A.shape[2]
                else:
                    raise ValueError(f"Unsupported input dimensions: {A.dim()}")
                
                if seq_len == 1:
                    C = gemm_cutlass.func_i8_gemv_out_bf16(A, B, scale_A, scale_B, 1.0)
                else:
                    C = gemm_cutlass.func_w8a8_matmul(A, B, scale_A, scale_B)
                return C, 1.0
            else:
                # print(f"[DEBUG] Custom_Matmul forward: A shape={A.shape}, B shape={B.shape}")
                if A.dim() == 2:
                    seq_len = A.shape[0]
                    scale_C = self.scale_y[:seq_len]
                elif A.dim() == 3:
                    seq_len = A.shape[1]
                    scale_C = self.scale_y[:, :seq_len]
                elif A.dim() == 4:
                    seq_len = A.shape[2]
                    scale_C = self.scale_y[:, :, :seq_len]
                else:
                    raise ValueError(f"Unsupported input dimensions: {A.dim()}")
                    
                if seq_len == 1:
                    C_i8 = gemm_cutlass.func_i8_gemv_out_i8(A, B,\
                                                scale_A, scale_B, scale_C, 1.0)
                else:
                    row_scale = scale_A / scale_C
                    col_scale = scale_B
                    C_i8 = gemm_cutlass.func_w8a8o8_matmul_fusion(A, B,\
                        row_scale, col_scale)
                    
                return C_i8, scale_C
        
    def finish_calibration(self):
        self.scale_y = self.out_observer.get_scale(input_ndim=self.input_dims)
        self.is_quantized = True
        

# class Custom_Matmul(nn.Module):
#     """
#     C = A @ B^T
#     This module perform matmul with two execution modes:
#     1. Calibration mode:
#     - Input:
#         A: (M, K) - bf16
#         B: (N, K) - bf16
#     - Output:
#         C: (M, N) - bf16
    
#     2. Quantization mode:
#     - Input:
#         A: (M, K)  - int8
#         B: (N, K) - int8
#         scale_A: (M,) - float32
#         scale_B: (N,) - float32
#     - Output:
#         C: (M, N) - int8
#         scale_C: (M,) - float32
#     """
#     def __init__(self, num_heads=1, max_seq_len=MAX_SEQ_LEN, is_return_float=False):
#         super().__init__()
#         self.num_heads = num_heads
#         self.max_seq_len = max_seq_len
#         self.is_return_float = is_return_float

#         self.is_quantized = False
        
#     def forward(self, A, scale_A, B, scale_B):
#         if self.is_quantized == False:
#             if A.dim() == 2:
#                 C = torch.matmul(A, B.T)
#             elif A.dim() == 3:
#                 C = torch.matmul(A, B.transpose(-2, -1))
#             elif A.dim() == 4:
#                 C = torch.matmul(A, B.transpose(-2, -1))
#             else:
#                 raise ValueError(f"Unsupported input dimensions: {A.dim()}")
#             return C, 1.0
#         else:
#             # print(f"[DEBUG] Custom_Matmul forward: A shape={A.shape}, B shape={B.shape}")
            
#             if A.dim() == 2:
#                 if self.is_return_float:
#                     C = gemm_cutlass.func_w8a8_matmul(A, B, scale_A, scale_B)
#                     return C, 1.0
#                 else:
#                     C_int8, scale_C = gemm_cutlass.func_w8a8o8_matmul(
#                         A, B, scale_A, scale_B, 
#                     )
#                     return C_int8, scale_C
#             elif A.dim() == 3:
#                 if self.is_return_float:
#                     C = gemm_cutlass.func_w8a8_matmul(A, B, scale_A, scale_B)
#                     return C, 1.0
#                 else:
#                     C_int8, scale_C = gemm_cutlass.func_w8a8o8_matmul_batched(
#                         A, B, scale_A, scale_B
#                     )
#                     return C_int8, scale_C
#             elif A.dim() == 4:
#                 if self.is_return_float:
#                     C = gemm_cutlass.func_w8a8_matmul(A, B, scale_A, scale_B)
#                     return C, 1.0
#                 else:
#                     C_int8, scale_C = gemm_cutlass.func_w8a8o8_matmul_batched(
#                         A, B, scale_A, scale_B
#                     )
#                     return C_int8, scale_C
#             else:
#                 raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
#     def finish_calibration(self):
#         self.is_quantized = True
        

# class MinMaxObserverPerLastDim(nn.Module): 
#     def __init__(self, max_batch=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN): 
#         super().__init__() 
#         self.max_batch = max_batch 
#         self.max_seq_len = max_seq_len 
    
#         self.max_val = torch.full((max_batch, max_seq_len),\
#                             -torch.inf, device='cuda')
#         self.min_val = torch.full((max_batch, max_seq_len),\
#                             torch.inf, device='cuda')
        
#     @torch.no_grad() 
#     def forward(self, x: torch.Tensor): 
#         if x.ndim == 2: 
#             T, _ = x.shape 
#             B = 1 
#             xd = x.detach().unsqueeze(0) # (1, T, H) 
#         elif x.ndim == 3: 
#             B, T, _ = x.shape 
#             xd = x.detach() 
#         else: raise ValueError(f"Expected 2D or 3D input,\
#             got {x.ndim}D with shape {tuple(x.shape)}") 
        
#         if B > self.max_batch: 
#             raise ValueError(f"B={B} exceeds max_batch={self.max_batch}") 
#         if T > self.max_seq_len: 
#             raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}") 
        
#         # token-wise over hidden dim -> (B, T) 
#         cur_max = xd.amax(dim=-1) 
#         cur_min = xd.amin(dim=-1) 
        
#         # update only prefix [0:B, 0:T) 
#         self.max_val[:B, :T] = torch.maximum(self.max_val[:B, :T], cur_max)
#         self.min_val[:B, :T] = torch.minimum(self.min_val[:B, :T], cur_min)
#         return x 
    
#     def get_scale(self, B=None, T=None): 
#         qmax = 127.0

#         if T is None: 
#             T = self.max_seq_len 

#         max_val = self.max_val[:B, :T].to(torch.float32)
#         min_val = self.min_val[:B, :T].to(torch.float32)
#         max_abs = torch.maximum(
#             max_val.abs(),
#             min_val.abs()
#         )

#         scale = (max_abs / qmax).clamp(min=1e-8)
#         return scale

class MinMaxObserverPerLastDim(nn.Module):
    """
    Tracks min/max over the last dimension `d`.

    Supported input shapes:
        (B, d)
        (B, L, d)
        (B, H, L, d)

    Internally everything is viewed as: (B, H, L, d)

    Stored observer buffers have shape:
        (max_batch, max_heads, max_seq_len)
    """

    def __init__(
        self,
        max_batch: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        max_heads: int = 1,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        self.max_batch = max_batch
        self.max_heads = max_heads
        self.max_seq_len = max_seq_len

        observer_shape = (max_batch, max_heads, max_seq_len)

        self.register_buffer(
            "max_val",
            torch.full(observer_shape, -torch.inf, device=device),
        )

        self.register_buffer(
            "min_val",
            torch.full(observer_shape, torch.inf, device=device),
        )

    def _to_bhld(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """
        Convert supported input shapes to canonical shape (B, H, L, d).
        Returns:
            x4, B, H, L
        """
        x_detached = x.detach()

        if x_detached.ndim == 2:
            # (B, d) -> (B, 1, 1, d)
            B, _ = x_detached.shape
            H = 1
            L = 1
            x4 = x_detached[:, None, None, :]

        elif x_detached.ndim == 3:
            # (B, L, d) -> (B, 1, L, d)
            B, L, _ = x_detached.shape
            H = 1
            x4 = x_detached[:, None, :, :]

        elif x_detached.ndim == 4:
            # (B, H, L, d)
            B, H, L, _ = x_detached.shape
            x4 = x_detached

        else:
            raise ValueError(
                f"Expected input with shape (B, d), (B, L, d), or (B, H, L, d), "
                f"but got {x_detached.ndim}D tensor with shape {tuple(x_detached.shape)}."
            )

        return x4, B, H, L

    def _check_shape_limits(self, B: int, H: int, L: int) -> None:
        if B > self.max_batch:
            raise ValueError(f"B={B} exceeds max_batch={self.max_batch}")

        if H > self.max_heads:
            raise ValueError(f"H={H} exceeds max_heads={self.max_heads}")

        if L > self.max_seq_len:
            raise ValueError(f"L={L} exceeds max_seq_len={self.max_seq_len}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x4, B, H, L = self._to_bhld(x)
        self._check_shape_limits(B, H, L)

        # x4:      (B, H, L, d)
        # cur_max: (B, H, L)
        # cur_min: (B, H, L)
        cur_max = x4.amax(dim=-1)
        cur_min = x4.amin(dim=-1)

        self.max_val[:B, :H, :L] = torch.maximum(
            self.max_val[:B, :H, :L],
            cur_max,
        )

        self.min_val[:B, :H, :L] = torch.minimum(
            self.min_val[:B, :H, :L],
            cur_min,
        )

        return x

    def get_scale(
        self,
        B: int | None = None,
        H: int | None = None,
        L: int | None = None,
        input_ndim: int | None = None,
        qmax: float = 127.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Returns scale.

        If input_ndim is None:
            returns canonical shape (B, H, L)

        If input_ndim == 2:
            input was (B, d), returns (B,)

        If input_ndim == 3:
            input was (B, L, d), returns (B, L)

        If input_ndim == 4:
            input was (B, H, L, d), returns (B, H, L)
        """

        B = self.max_batch if B is None else B
        H = self.max_heads if H is None else H
        L = self.max_seq_len if L is None else L

        self._check_shape_limits(B, H, L)

        max_val = self.max_val[:B, :H, :L].float()
        min_val = self.min_val[:B, :H, :L].float()

        max_abs = torch.maximum(max_val.abs(), min_val.abs())
        scale = (max_abs / qmax).clamp(min=eps)

        if input_ndim is None:  # Canonical observer shape: (B, H, L)
            return scale

        if input_ndim == 2:  # convert scale (B, 1, 1) -> (B,)
            return scale[:, 0, 0]

        if input_ndim == 3: # convert scale (B, 1, L) -> (B, L)
            return scale[:, 0, :]

        if input_ndim == 4: # scale already (B, H, L)
            return scale

        raise ValueError(f"Unsupported input_ndim={input_ndim}")
    
    

class PerChannelAbsMaxObserver(nn.Module):
    """Tracks max(abs(x)) per last dimension (channel)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.amax = torch.zeros(num_channels, device='cuda')

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
    max_a = input_observer.get_amax().to(torch.float32)  
    max_a = torch.clamp(max_a, min=1e-6)

    # Weight stats
    max_w = weight.detach().abs().amax(dim=0).to(torch.float32)
    max_w = torch.clamp(max_w, min=1e-6)

    # Compute alpha
    alpha = torch.pow(max_a, lambd) / torch.pow(max_w, (1.0 - lambd))
    alpha = torch.clamp(alpha, min=0.01, max=100.0)
    return alpha
        
        
class Custom_Silu(nn.Module):
    def __init__(self, emb_dim, dtype=torch.bfloat16):
        super().__init__()
        self.emb_dim = emb_dim
        self.dtype = dtype

        self.is_quantized = False
        self.is_smooth_scale = False
        
        self.smooth_alpha = torch.ones(emb_dim, dtype=torch.float32)

    def forward(self, x1, scale_x1, x2, scale_x2):
        if not self.is_quantized:
            x_silu = torch.nn.functional.silu(x1)
            x = x_silu * x2
            return x
        else:
            assert x1.dtype == torch.int8 and x2.dtype == torch.int8,\
                "Expected int8 inputs in quantized mode"

            x_int8, x_scale = gemm_cutlass.func_silu_mul_int8(\
                x1, scale_x1,\
                x2, scale_x2, self.smooth_alpha, True)
            
            return x_int8, x_scale

    def finish_calibration(self):
        self.is_quantized = True

    def enable_smooth_scale(self, smooth_alpha_value):
        """
        This function apply smooth scaling to quantization (for input activations) 
        Note:
        - Normal quantization: Y_int8 = Y / scale_Y
        - Smooth quantization: Y_int8 = (Y / smooth_scale) / scale_Y
        """
        assert smooth_alpha_value.shape == (self.emb_dim,)
        self.is_smooth_scale = True
        
        smooth_alpha_value = smooth_alpha_value.to(torch.float32)
        self.smooth_alpha = smooth_alpha_value
        
        
class Custom_FeedForward(nn.Module):
    """
    In the FNN, we use SmoothQuant for all fc1, fc2, and fc3 layers to 
    minimize the quantization error due to activation outlier.
    
    We use the same smooth_alpha for both fc1 and fc2, 
    since they have the same input activation X.
    """
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.fc2 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.fc3 = Custom_Linear(cfg["hidden_dim"], cfg["emb_dim"],\
                                        max_seq_len=MAX_SEQ_LEN)
        self.silu_layer = Custom_Silu(cfg["hidden_dim"])
        
        self.is_quantized = False

    def forward(self, x, scale_x=1.0):
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
            
            out = gemm_cutlass.func_w8a8_matmul(X_smooth_q, self.fc3.weight_q,\
                            scale_x_smooth, self.fc3.scale_w)
            
            return out, 1.0
        
    def finish_calibration(self):
        self.fc1.finish_calibration()
        fc1_smooth_alpha = self.fc1.smooth_alpha

        # fc2 uses the same smooth_alpha as fc1 
        self.fc2.finish_calibration(alpha=fc1_smooth_alpha) 
        
        self.fc3.finish_calibration()
        
        # Assign fc3's smooth_alpha to silu_layer's 
        fc3_smooth_alpha = self.fc3.smooth_alpha
        self.silu_layer.enable_smooth_scale(fc3_smooth_alpha)
        self.silu_layer.finish_calibration()
        
        self.is_quantized = True



class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.qk_norm = qk_norm

        self.W_query = Custom_Linear(d_in, self.d_out).to(dtype)
        self.W_key = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.W_value = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.out_proj = Custom_Linear(self.d_out, d_in).to(dtype)
        
        self.query_rope = Custom_RoPE(num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        self.key_rope = Custom_RoPE(num_kv_groups, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        
        self.softmax_layer = Custom_Softmax(num_heads=num_heads).to(dtype)    
        self.qk_score_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.context_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN,\
            is_return_float=True).to(dtype)

        if self.qk_norm:
            self.q_norm = Custom_RMSNorm(head_dim, eps=1e-6).to(dtype)
            self.k_norm = Custom_RMSNorm(head_dim, eps=1e-6).to(dtype)
        else:
            self.q_norm = self.k_norm = None
            
        self.is_quantized = False

    def forward(self, x, scale_x, mask, cos, scale_cos, sin, scale_sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape
        
        if self.is_quantized == False:
            # Q/K/V projections
            queries, _ = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
            keys, _ = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
            values, _ = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

            # Reshape
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

            if self.qk_norm:
                queries = self.q_norm(queries)
                keys_new = self.k_norm(keys_new)

            # Apply RoPE
            queries, _ = self.query_rope(queries, 1.0, cos, 1.0, sin, 1.0, start_offset=start_pos)
            keys_new, _ = self.key_rope(keys_new, 1.0, cos, 1.0, sin, 1.0, start_offset=start_pos)

            # Update KV cache
            if cache is not None:  
                prev_k, prev_v = cache
                keys = torch.cat([prev_k, keys_new], dim=2)
                values = torch.cat([prev_v, values_new], dim=2)
                next_cache = (keys, values)
            else:
                start_pos = 0  # reset RoPE
                keys, values = keys_new, values_new
                next_cache = (keys, values)
            # ---------------------------------------------------------------------

            # Expand K and V to match number of heads
            keys = keys.repeat_interleave(self.group_size, dim=1)
            values = values.repeat_interleave(self.group_size, dim=1)

            # Attention Scores: (b, num_heads, num_tokens, num_tokens)
            attn_scores, _ = self.qk_score_layer(queries, 1.0, keys, 1.0)
            
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_scores = attn_scores / (self.head_dim ** 0.5)
            attn_weights, _ = self.softmax_layer(attn_scores, 1.0, 1.0) 

            # Context: (b, num_heads, num_tokens, head_dim)
            values = values.transpose(2, 3)  # Shape: (b, num_heads, head_dim, num_tokens)
            context, _ = self.context_layer(attn_weights, 1.0, values, 1.0)
            
            # Output projection
            context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out) 
            out, _ = self.out_proj(context)
            return out, next_cache
        else:
            x_int8 = x
            assert x_int8.dtype == torch.int8, "Input must be int8"
            
            # Q, K, V projections
            queries_int8, queries_scale = self.W_query(x_int8, scale_x)
            keys_int8, keys_scale = self.W_key(x_int8, scale_x)
            values_int8, values_scale = self.W_value(x_int8, scale_x)
            
            # Reshape for multi-head 
            queries_int8 = queries_int8.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            queries_scale = queries_scale.unsqueeze(1).expand(-1, self.num_heads, -1)
            keys_int8 = keys_int8.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            keys_scale = keys_scale.unsqueeze(1).expand(-1, self.num_kv_groups, -1)
            values_int8 = values_int8.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            values_scale = values_scale.unsqueeze(1).expand(-1, self.num_kv_groups, -1)
            
            # Normalize Q and K
            if self.qk_norm:
                queries_int8, queries_scale = self.q_norm(queries_int8, queries_scale)
                keys_int8, keys_scale = self.k_norm(keys_int8, keys_scale)
            
            # Apply RoPE 
            queries_int8, queries_scale = self.query_rope(queries_int8, queries_scale,\
                                        cos, scale_cos, sin, scale_sin, start_offset=start_pos)
            
            keys_int8, keys_scale = self.key_rope(keys_int8, keys_scale,\
                                        cos, scale_cos, sin, scale_sin, start_offset=start_pos)
            
            # Update KV cache
            if cache is not None:
                prev_k_int8, prev_k_scale, prev_v_int8, prev_v_scale = cache
                keys_int8 = torch.cat([prev_k_int8, keys_int8], dim=2)  # Shape: (b, n_kv_groups, n_tokens, head_dim)
                keys_scale = torch.cat([prev_k_scale, keys_scale], dim=-1)  # Shape: (b, n_kv_groups, n_tokens)
                values_int8 = torch.cat([prev_v_int8, values_int8], dim=2)
                values_scale = torch.cat([prev_v_scale, values_scale], dim=-1)
                next_cache = (keys_int8, keys_scale, values_int8, values_scale)
            else:
                start_pos = 0  # reset RoPE
                next_cache = (keys_int8, keys_scale, values_int8, values_scale)
            
            # Repeat K and V for grouped attention
            keys_int8 = keys_int8.repeat_interleave(self.group_size, dim=1)
            keys_scale = keys_scale.repeat_interleave(self.group_size, dim=1)
            values_int8 = values_int8.repeat_interleave(self.group_size, dim=1)
            values_scale = values_scale.repeat_interleave(self.group_size, dim=1)
            
            # Attention score 
            attn_scores_int8, attn_scores_scale = self.qk_score_layer(queries_int8,\
                                                        queries_scale,\
                                                        keys_int8,\
                                                        keys_scale)
            
            attn_scores_scale = attn_scores_scale / (self.head_dim ** 0.5)
            attn_weights_int8, attn_weights_scale = self.softmax_layer(attn_scores_int8,\
                                                    attn_scores_scale, mask)
            
            # Compute context with quantization
            values_int8, values_scale = gemm_cutlass.func_dequant_transpose_requant(values_int8,\
                                                        values_scale)
            
            # Context: (b, num_heads, num_tokens, head_dim)
            context, _ = self.context_layer(attn_weights_int8,\
                                            attn_weights_scale,\
                                            values_int8,\
                                            values_scale)
            context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out) 
            
            # Output projection     
            out, _ = self.out_proj(context)
            return out, next_cache
        
    def finish_calibration(self):
        self.W_query.finish_calibration()
        self.W_key.finish_calibration()
        self.W_value.finish_calibration()
        # self.out_proj.finish_calibration()
        if self.qk_norm:
            self.q_norm.finish_calibration()
            self.k_norm.finish_calibration()
        self.query_rope.finish_calibration()
        self.key_rope.finish_calibration()
        self.softmax_layer.finish_calibration()
        self.qk_score_layer.finish_calibration()
        self.context_layer.finish_calibration()
        self.is_quantized = True    
    
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"])
        
        self.ff = Custom_FeedForward(cfg).to(cfg["dtype"])
        
        self.norm1 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-6, dtype=cfg["dtype"])
        self.norm2 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-6, dtype=cfg["dtype"])
        
        self.is_quantized = False

    def forward(self, x, mask, cos, scale_cos, sin, scale_sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        if self.is_quantized == False:
            x = self.norm1(x)
            x, next_cache = self.att(x, 1.0, mask, cos, 1.0, sin, 1.0, start_pos=start_pos, cache=cache)  
        else:
            x_int8, scale_x = self.norm1(x)
            x, next_cache = self.att(x_int8, scale_x, mask,\
                                    cos, scale_cos, sin, scale_sin,\
                                    start_pos=start_pos, cache=cache)
        x = x + shortcut  

        # Shortcut connection for feed-forward block
        shortcut = x
        
        if self.is_quantized == False:
            x = self.norm2(x)
            x, _ = self.ff(x)
        else:
            x_int8, scale_x = self.norm2(x)
            x, _ = self.ff(x_int8, scale_x)
        x = x + shortcut  

        return x, next_cache
    
    def finish_calibration(self):
        self.att.finish_calibration()
        self.norm1.finish_calibration()
        smooth_scale = self.att.W_query.smooth_alpha
        self.norm1.enable_smooth_scale(smooth_scale)
        
        self.ff.finish_calibration()
        self.norm2.finish_calibration()
        smooth_scale = self.ff.fc1.smooth_alpha
        self.norm2.enable_smooth_scale(smooth_scale)
        
        self.is_quantized = True