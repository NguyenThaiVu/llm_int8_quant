import os 
import torch 
from torch import nn
import gemm_cutlass
from utils_quant import *

MAX_SEQ_LEN = 2048
BATCH_SIZE = 32

class MinMaxObserverPerLastDim(nn.Module): 
    def __init__(self, max_batch=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN): 
        super().__init__() 
        self.max_batch = max_batch 
        self.max_seq_len = max_seq_len 
    
        self.max_val = torch.full((max_batch, max_seq_len), -torch.inf, device='cuda')
        self.min_val = torch.full((max_batch, max_seq_len), torch.inf, device='cuda')
        
    @torch.no_grad() 
    def forward(self, x: torch.Tensor): 
        if x.ndim == 2: 
            T, _ = x.shape 
            B = 1 
            xd = x.detach().unsqueeze(0) # (1, T, H) 
        elif x.ndim == 3: 
            B, T, _ = x.shape 
            xd = x.detach() # (B, T, H)
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

        if T is None: 
            T = self.max_seq_len 

        max_val = self.max_val[:B, :T].to(torch.float32)
        min_val = self.min_val[:B, :T].to(torch.float32)
        max_abs = torch.maximum(
            max_val.abs(),
            min_val.abs()
        )

        scale = (max_abs / qmax).clamp(min=1e-8)
        return scale
    

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
        
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.scale_y = None

        self.is_quantized = False
        
    def forward(self, A, scale_A, B, scale_B):
        if self.is_quantized == False:
            if A.dim() == 2:
                C = torch.matmul(A, B.T)
            elif A.dim() == 3:
                C = torch.matmul(A, B.transpose(-2, -1))
            elif A.dim() == 4:
                C = torch.matmul(A, B.transpose(-2, -1))
            else:
                raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
            # self.out_observer(C)
            return C, 1.0
        else:
            if self.is_return_float:
                C = gemm_cutlass.func_w8a8_matmul(A, B, scale_A, scale_B)
                return C, 1.0
            else:
                if A.dim() == 2:
                    # C_int8, scale_C = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    #     A, B, scale_A, scale_B, 
                    # )
                    seq_len = A.shape[0]
                    scale_C = self.scale_y[:seq_len]
                    
                    if seq_len == 1:
                        C_i8 = gemm_cutlass.func_int8_gemv_out_int8(A, B,\
                                                    scale_A, scale_B, scale_C, 1.0)
                    else:
                        row_scale = scale_A / scale_C
                        col_scale = scale_B
                        C_i8 = gemm_cutlass.func_w8a8o8_matmul(A, B,\
                            row_scale, col_scale)
                    return C_i8, scale_C
                elif A.dim() == 3:
                    seq_len = A.shape[1]
                    scale_C = self.scale_y[:, :seq_len]
                    if seq_len == 1:
                        C_i8 = gemm_cutlass.func_int8_gemv_out_int8(A, B,\
                                                    scale_A, scale_B, scale_C, 1.0)
                    else:
                        row_scale = scale_A / scale_C
                        col_scale = scale_B
                        C_i8 = gemm_cutlass.func_w8a8o8_matmul(A, B,\
                            row_scale, col_scale)
                    return C_i8, scale_C
                else:
                    raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
    def finish_calibration(self):
        self.scale_y = self.out_observer.get_scale()
        self.is_quantized = True
        
        
if __name__ == "__main__":
    
    dtype = torch.bfloat16
    batch_size = 32
    seq_len = 1
    A = torch.randn((batch_size, seq_len, 128), dtype=dtype, device="cuda")
    B = torch.randn((batch_size, 1024, 128), dtype=dtype, device="cuda")
    
    matmul_layer = Custom_Matmul(num_heads=1, max_seq_len=MAX_SEQ_LEN)
    
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
    
    C_i8, scale_C = matmul_layer(A_i8, scale_A, B_i8, scale_B)
    C_deq = C_i8.to(torch.float32) * scale_C.unsqueeze(-1)
    C_deq = C_deq.to(dtype)
    
    max_diff = (C - C_deq).abs().max()
    print(f"Max difference: {max_diff.item():.6f}")
    mse = torch.mean((C - C_deq) ** 2).item()   
    print(f"MSE: {mse:.6f}")
    
    print(f"Sample scale_C: {scale_C[:10]}")
    print(f"Sample scale_y calibration: {matmul_layer.scale_y[:10]}")
    
    