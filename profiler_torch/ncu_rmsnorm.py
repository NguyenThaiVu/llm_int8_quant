""""""
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
    rmsnorm = nn.RMSNorm(d, eps=1e-5).to(device=device, dtype=d_type)
    for _ in range(n_iter):
        rmsnorm(X)

    # 1. Naive RMSNorm
    for _ in range(n_iter):
        gemm_cutlass.func_rmsnorm_naive_int8(X_i8, scale_X, gamma, 1e-6)
        
    # 2. Our Kernel RMSNorm
    for _ in range(n_iter):
        gemm_cutlass.func_rmsnorm_int8(X_i8, scale_X, gamma, 1e-6)
        
    
        
        
    
    