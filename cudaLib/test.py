import os 
import torch 
from torch import nn
import gemm_cutlass
from utils_quant import *

MAX_SEQ_LEN = 2048
BATCH_SIZE = 1

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
        max_batch: int,
        max_seq_len: int,
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
                    C = gemm_cutlass.func_int8_gemv(A, B, scale_A, scale_B, 1.0)
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
                    C_i8 = gemm_cutlass.func_int8_gemv_out_int8_warp(A, B,\
                                                scale_A, scale_B, scale_C, 1.0)
                else:
                    row_scale = scale_A / scale_C
                    col_scale = scale_B
                    C_i8 = gemm_cutlass.func_w8a8o8_matmul(A, B,\
                        row_scale, col_scale)
                    
                return C_i8, scale_C
        
    def finish_calibration(self):
        self.scale_y = self.out_observer.get_scale(input_ndim=self.input_dims)
        self.is_quantized = True
        
        
if __name__ == "__main__":
    
    dtype = torch.bfloat16
    n_heads = 32
    batch_size = 1
    seq_len = 1
    A = torch.randn((batch_size, n_heads, seq_len, 128), dtype=dtype, device="cuda")
    B = torch.randn((batch_size, n_heads, 1234, 128), dtype=dtype, device="cuda")
    
    matmul_layer = Custom_Matmul(num_heads=n_heads, max_seq_len=MAX_SEQ_LEN,
                                 is_return_float=True).cuda()
    
    # bf16 computation
    C, _ = matmul_layer(A, None, B, None)
    print(f"Input A shape: {A.shape}, B shape: {B.shape}")
    print(f"Output C shape: {C.shape} \n")
    
    bf16_time = measure_time(matmul_layer, A, None, B, None)
    print(f"Latency (bf16 matmul): {bf16_time:.4f} ms")
    
    # int8 computation
    matmul_layer.finish_calibration()
    
    A_i8, scale_A = quantize_row_int8_symmetric_nd(A)
    B_i8, scale_B = quantize_row_int8_symmetric_nd(B)
    
    int8_time = measure_time(matmul_layer, A_i8, scale_A, B_i8, scale_B)
    print(f"Latency (int8 matmul): {int8_time:.4f} ms")
    
    # C_i8, scale_C = matmul_layer(A_i8, scale_A, B_i8, scale_B)
    # C_deq = C_i8.to(torch.float32) * scale_C.unsqueeze(-1)
    # C_deq = C_deq.to(dtype)
    C_deq, _ = matmul_layer(A_i8, scale_A, B_i8, scale_B)
    
    max_diff = (C - C_deq).abs().max()
    print(f"Max difference: {max_diff.item():.6f}")
    mse = torch.mean((C - C_deq) ** 2).item()   
    print(f"MSE: {mse:.6f}")
    
    # print(f"Sample scale_C: {scale_C[:10]}")
    # print(f"Sample scale_y calibration: {matmul_layer.scale_y[:10]}")
    
    