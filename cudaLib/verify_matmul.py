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
from utils_power import measure_power



if __name__ == "__main__":
    
    batch = 32
    M = 2048
    N = 4096
    K = 2048
    dtype = torch.bfloat16
    
    # ================= 2D input =======================
    X = torch.randn((batch, M, K), dtype=dtype).cuda()
    W = torch.empty((batch, N, K), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    print(f"Input shape: {X.shape}, Weight shape: {W.shape}")
    
    y_torch = X @ W.transpose(-1, -2)
    time_torch = measure_time(torch.matmul, X, W.transpose(-1, -2))
    print(f"Pytorch BF16 matmul latency: {time_torch:.2f} ms")
    torch_energy = measure_power(torch.matmul, X, W.transpose(-1, -2))
    
    # Quantization
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    W_int8, scale_W = quantize_row_int8_symmetric_nd(W)
    
    # assume scale_Y via calibration
    Y_int8, scale_Y = gemm_cutlass.func_int8_matmul_out_int8_three_scale(X_int8, W_int8, scale_X, scale_W)
    time_int8 = measure_time(gemm_cutlass.func_int8_matmul_out_int8_three_scale, X_int8, W_int8, scale_X, scale_W)
    print(f"Custom INT8 matmul latency: {time_int8:.2f} ms")
    i8_energy = measure_power(gemm_cutlass.func_int8_matmul_out_int8_three_scale, X_int8, W_int8, scale_X, scale_W)
    
    print(f"Time Speedup: {time_torch / time_int8:.2f}x")
    print(f"Energy Improvement: {torch_energy / i8_energy:.2f}x")
    
    # Verify the correctness
    Y_dequant = Y_int8.float() * scale_Y.unsqueeze(-1)
    max_diff = (Y_dequant - y_torch).abs().max()
    print(f"Max absolute diff: {max_diff:.4f}")
    mse = ((Y_dequant - y_torch) ** 2).mean().item()
    print(f"MSE: {mse:.4f} \n")

    