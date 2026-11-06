"""
In this script, we verify the correctness and latency of the RMSNorm implementation, between:
- PyTorch.
- Our Kernel.
"""

import os
import torch 
from torch import nn

from utils_quant import quantize_row_int8_symmetric_nd, measure_time

import gemm_cutlass



if __name__ == "__main__":
    device = "cuda"
    dtype = torch.bfloat16
    L = 2048
    d = 1024 * 8
    
    X = torch.randn((L, d), device=device, dtype=dtype)
    gamma = torch.randn((d,), device=device, dtype=dtype)
    
    # 1. Torch RMSNorm
    rmsnorm = nn.RMSNorm(d, eps=1e-5).to(device=device, dtype=dtype)
    rmsnorm.weight.data = gamma
    rmsnorm.eval()
    rmsnorm = torch.compile(rmsnorm)
    
    Y = rmsnorm(X)
    time_torch = measure_time(rmsnorm, X)
    print(f"PyTorch RMSNorm time: {time_torch:.6f} seconds")
    
    # 2. Our Kernel RMSNorm
    X_i8, scale_X = quantize_row_int8_symmetric_nd(X)
    
    our_time = measure_time(gemm_cutlass.func_rmsnorm_int8, X_i8, scale_X, gamma, 1e-6)
    print(f"Our Kernel RMSNorm time: {our_time:.6f} seconds")
    
    Y_i8, scale_Y = gemm_cutlass.func_rmsnorm_int8(X_i8, scale_X, gamma, 1e-6)
    Y_deq = Y_i8.to(torch.float32) * scale_Y.unsqueeze(-1)
    
    max_diff = (Y - Y_deq).abs().max()
    print(f"Max difference: {max_diff:.6f}")
    mse = ((Y - Y_deq) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    
    
    