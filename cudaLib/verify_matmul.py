"""
In this script, we verify the correctness and latency between:
- Pytorch BF16 matmul
- Our custom INT8 matmul
"""

import math
import numpy as np
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    M = 8192
    N = 4096
    K = 1024 * 6
    dtype = torch.bfloat16
    
    random_seed = np.random.randint(1, 5)
    X = torch.randn((M, K), dtype=dtype).cuda() * random_seed
    
    # init W with kaiming for better quantization results
    W = torch.empty((N, K), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    
    y_torch = X @ W.t()
    time_torch = measure_time(torch.matmul, X, W.t())
    print(f"Pytorch BF16 matmul latency: {time_torch:.2f} ms")
    
    # Quantization
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    W_int8, scale_W = quantize_row_int8_symmetric_nd(W)
    
    # assume scale_Y via calibration
    _, scale_Y = quantize_row_int8_symmetric_nd(y_torch)
    row_scale = scale_X / scale_Y
    print(f"Shape of row_scale: {row_scale.shape}")
    print(f"Shape of col_scale: {scale_W.shape}")
    
    Y_int8 = gemm_cutlass.func_w8a8o8_matmul(X_int8, W_int8, row_scale, scale_W)
    time_int8 = measure_time(gemm_cutlass.func_w8a8o8_matmul, X_int8, W_int8, row_scale, scale_W)
    print(f"Custom INT8 matmul latency: {time_int8:.2f} ms")
    
    # Verify the correctness
    Y_dequant = Y_int8.float() * scale_Y.unsqueeze(-1)
    max_diff = (Y_dequant - y_torch).abs().max()
    print(f"Max absolute diff: {max_diff:.4f}")
    mse = ((Y_dequant - y_torch) ** 2).mean().item()
    print(f"MSE: {mse:.4f}")
    
    # print sample values
    print(f"Y_torch sample: {y_torch[:5, :5]}")
    print(f"Y_dequant sample: {Y_dequant[:5, :5]}")

