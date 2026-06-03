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
    
    M = 2048
    N = 4096
    K = 8192
    dtype = torch.bfloat16
    
    # ======================================================
    # ================= 2D input =======================
    X = torch.randn((M, K), dtype=dtype).cuda()
    
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
    
    Y_int8 = gemm_cutlass.func_w8a8o8_matmul(X_int8, W_int8, row_scale, scale_W)
    time_int8 = measure_time(gemm_cutlass.func_w8a8o8_matmul, X_int8, W_int8, row_scale, scale_W)
    print(f"Custom INT8 matmul latency: {time_int8:.2f} ms")
    
    # Verify the correctness
    Y_dequant = Y_int8.float() * scale_Y.unsqueeze(-1)
    max_diff = (Y_dequant - y_torch).abs().max()
    print(f"Max absolute diff: {max_diff:.4f}")
    mse = ((Y_dequant - y_torch) ** 2).mean().item()
    print(f"MSE: {mse:.4f} \n")
    
    # # print sample values
    # print(f"Y_torch sample: {y_torch[:5, :5]}")
    # print(f"Y_dequant sample: {Y_dequant[:5, :5]}")


    # ======================================================
    # ================= 3D input =======================
    batch_size = 8
    X = torch.randn((batch_size, M, K), dtype=dtype).cuda()
    
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
    col_scale = scale_W
    
    Y_int8 = gemm_cutlass.func_w8a8o8_matmul(X_int8, W_int8, row_scale, col_scale)
    time_int8 = measure_time(gemm_cutlass.func_w8a8o8_matmul, X_int8, W_int8, row_scale, col_scale)
    print(f"Custom INT8 matmul latency: {time_int8:.2f} ms")
    
    # Verify the correctness
    Y_dequant = Y_int8.float() * scale_Y.unsqueeze(-1)
    max_diff = (Y_dequant - y_torch).abs().max()
    print(f"Max absolute diff: {max_diff:.4f}")
    mse = ((Y_dequant - y_torch) ** 2).mean().item()
    print(f"MSE: {mse:.4f} \n")
    
    