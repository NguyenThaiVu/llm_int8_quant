"""
Test quantization
"""

import os 
import torch
from utils import quantize_row_int8_symmetric_nd

if __name__ == "__main__":
    M = 1024
    N = 4096
    dtype = torch.bfloat16
    
    X = torch.randn(M, N, dtype=dtype).to("cuda")
    X_q, scales = quantize_row_int8_symmetric_nd(X, percentile=0.999)
    
    X_deq = X_q.to(dtype) * scales.unsqueeze(-1)
    
    max_diff = (X - X_deq).abs().max()
    print(f"Max diff: {max_diff.item():.6f}")
    mse = ((X - X_deq) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
