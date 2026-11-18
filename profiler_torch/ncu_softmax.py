import os 
import torch
from torch import nn
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    L = 2048
    d = 8192
    d_type = torch.bfloat16
    device = "cuda"
    n_iter = 100
    
    X = torch.randn((L, d), dtype=d_type, device=device)
    gamma = torch.randn((d,), device=device, dtype=d_type)
    
    X_i8, scale_X = quantize_row_int8_symmetric_nd(X)
    
    # =======================
    # warmup GPU
    softmax = nn.Softmax(dim=-1).to(device=device, dtype=d_type)
    for _ in range(n_iter):
        softmax(X)

    # 1. Naive Softmax
    for _ in range(n_iter):
        gemm_cutlass.func_softmax_int8_naive(X_i8, scale_X)
        
    # 2. Our Kernel Softmax
    for _ in range(n_iter):
        gemm_cutlass.func_softmax_int8(X_i8, scale_X)