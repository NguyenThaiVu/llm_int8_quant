"""
In this script, we verify the correctness and latency of the Softmax implementation, between:
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
    d = 1024 * 16
    
    X = torch.randn((L, d), device=device, dtype=dtype)
    
    # 1. Torch Softmax
    softmax = nn.Softmax(dim=-1).to(device=device, dtype=dtype)
    softmax.eval()
    softmax = torch.compile(softmax)
    
    Y = softmax(X)
    time_torch = measure_time(softmax, X)
    print(f"PyTorch Softmax time: {time_torch:.6f} seconds")
    
    # 2. Our Kernel Softmax
    X_i8, scale_X = quantize_row_int8_symmetric_nd(X)
    
    our_time = measure_time(gemm_cutlass.func_softmax_int8, X_i8, scale_X)
    print(f"Our Kernel Softmax time: {our_time:.6f} seconds")
    
    Y_i8, scale_Y = gemm_cutlass.func_softmax_int8(X_i8, scale_X)
    Y_deq = Y_i8.to(torch.float32) * scale_Y.unsqueeze(-1)
    
    max_diff = (Y - Y_deq).abs().max()
    print(f"Max difference: {max_diff:.6f}")
    mse = ((Y - Y_deq) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    print(f"Y_torch: {Y[:5, :5]}")
    print(f"Y_deq: {Y_deq[:5, :5]}")
