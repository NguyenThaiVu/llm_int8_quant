"""
In this script, we verify the correctness of matmul.
"""

import os
import torch
import torch.nn as nn
import gemm_cutlass
from utils import *

if __name__ == "__main__":
    
    seq_len = 1024
    embed_dim = 8192
    out_dim = 8192
    
    dtype = torch.bfloat16
    device = torch.device("cuda")
    
    X = torch.randn(seq_len, embed_dim, dtype=dtype, device=device)
    W = torch.randn(out_dim, embed_dim, dtype=dtype, device=device)
    W_t = W.t()  
    
    Y = torch.matmul(X, W_t)  # Shape: (seq_len, out_dim)
    
    # Assume output scale via calibration 
    _, Y_scale = quantize_row_int8_symmetric_nd(Y)
    
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X)
    W_int8, W_scale = quantize_row_int8_symmetric_nd(W)
    
    Y_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
        X_int8, W_int8, X_scale, W_scale, Y_scale
    ) 
    
    Y_dequant = Y_int8.to(torch.float32) * Y_scale.unsqueeze(-1) 
    
    max_diff = torch.max(torch.abs(Y - Y_dequant))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_dequant) ** 2).item()
    print(f"MSE: {mse}")
    
    # Measure time 
    
    torch_time = measure_time(torch.matmul, X, W_t)
    print(f"PyTorch time: {torch_time:.4f} seconds")
    
    cutlass_time = measure_time(
        gemm_cutlass.func_int8_matmul_out_int8_three_scale,
        X_int8, W_int8, X_scale, W_scale, Y_scale
    )
    print(f"Cutlass int8 matmul time: {cutlass_time:.4f} seconds")
