"""
This file contains the implementation of RMSNorm and its profiling code.
"""

import os 
import torch
import torch.nn as nn
import gemm_cutlass  
from utils import *

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # leanrable weight of RMSNorm
        self.eps = eps

    def forward(self, x):
        # x: [seq_len, head_dim]
        mean_square = x.pow(2).mean(-1, keepdim=True)  # [seq_len, 1]
        inv_rms = torch.rsqrt(mean_square + self.eps)   # [seq_len, 1]
        return x * inv_rms * self.weight  # [seq_len, head_dim]


if __name__ == "__main__":
    seq_length = 1024 * 8
    dim = 1024 * 5 
    dtype = torch.bfloat16
    # dtype = torch.float32
    device = 'cuda'
    
    X = torch.randn(seq_length, dim, dtype=dtype, device=device)
    rms_norm = RMSNorm(dim=dim).to(device).to(dtype)
    Y = rms_norm(X)
    
    # Quantization version
    X_q, scale_X = quantize_tensor(X)
    
    # Assume output scale is obtained via calibration
    _, scale_Y = quantize_tensor(Y)
    
    Y_quant = gemm_cutlass.func_rmsnorm_int8(X_q, scale_X,\
                                            rms_norm.weight, scale_Y, rms_norm.eps)
    
    Y_deq = Y_quant.to(dtype) * scale_Y
    
    print(f"Shape of output: {Y_deq.shape} - dtype: {Y_deq.dtype}")
    
    if torch.allclose(Y, Y_deq, rtol=0.1, atol=0.1):
        print("RMSNorm correct !")
    else:
        print("===== [ERROR] RMSNorm incorrect =====")
    
    max_diff = (Y - Y_deq).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    mse = ((Y - Y_deq) ** 2).mean().item()
    print(f"Mean Squared Error: {mse:.6f}")
    
    print(f"Y_true: {Y[:5, :5]}")
    print(f"Y_deq: {Y_deq[:5, :5]}")
    
    print()
    
    # Measure performance
    time_rmsnorm = measure_time(rms_norm, X, repeat = 10)
    print(f"RMSNorm time: {time_rmsnorm:.2f} ms")
    
    time_rmsnorm_int8 = measure_time(lambda x: gemm_cutlass.func_rmsnorm_int8(x, scale_X,\
                                            rms_norm.weight, scale_Y, rms_norm.eps), X_q, repeat=10)
    print(f"RMSNorm INT8 time: {time_rmsnorm_int8:.2f} ms")

