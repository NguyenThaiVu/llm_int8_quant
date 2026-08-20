"""
In this script, we compare the latency between our custom INT8 matmul 
kernel with:
- Pytorch BF16 matmul
- bitsandbytes INT8 matmul 
"""

import os 
import time
import math
import numpy as np
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant

import gemm_cutlass

from utils_quant import *


def bnb_int8_and_dequantize(x_q, w_q, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N) float32
    w_scales: (M,) float32
    Returns:
      y: (B, N, M) float32
    """
    y_int = int8_linear_matmul(x_q, w_q) 
    y = int8_mm_dequant(y_int, x_scales, w_scales)
    return y.to(output_dtype)

if __name__ == "__main__":
    
    # M = 8192
    # N = 4096
    # K = 1024 * 6
    dtype = torch.bfloat16
    list_M = [2048, 2048, 4096, 4096, 8192]
    list_N = [2048, 4096, 4096, 8192, 8192]
    list_K = [2048, 4096, 4096, 8192, 8192]
    
    for (M, N, K) in zip(list_M, list_N, list_K):
        print(f"\n\nTesting matmul with M={M}, N={N}, K={K}")
        X = torch.randn((M, K), dtype=dtype).cuda()
        # init W with kaiming for better quantization results
        W = torch.empty((N, K), dtype=dtype).cuda()
        torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        
        # 1. Pytorch BF16 matmul
        y_torch = X @ W.t()
        time_torch = measure_time(torch.matmul, X, W.t())
        print(f"Pytorch BF16 matmul latency: {time_torch:.2f} ms")
        
        # Quantization
        X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
        W_int8, scale_W = quantize_row_int8_symmetric_nd(W)
        
        # assume scale_Y via calibration
        _, scale_Y = quantize_row_int8_symmetric_nd(y_torch)
        row_scale = scale_X / scale_Y
        
        # 2. bitsandbytes INT8 matmul
        time_bnb = measure_time(bnb_int8_and_dequantize, X_int8, W_int8, scale_X, scale_W)
        print(f"bitsandbytes INT8 matmul (FP16 output) latency: {time_bnb:.2f} ms")
        
        time_bnb_int32 = measure_time(int8_linear_matmul, X_int8, W_int8)
        print(f"bitsandbytes INT8 matmul (INT32 output) latency: {time_bnb_int32:.2f} ms")
        
        # 3. Torch INT8 matmul
        Y_int32_torch = torch._int_mm(X_int8, W_int8.t())
        time_torch_int8 = measure_time(torch._int_mm, X_int8, W_int8.t())
        print(f"Torch INT8 matmul latency: {time_torch_int8:.2f} ms")
        
        # 4. Our custom INT8 matmul
        Y_int8 = gemm_cutlass.func_w8a8o8_matmul_fusion(X_int8, W_int8, row_scale, scale_W)
        time_int8 = measure_time(gemm_cutlass.func_w8a8o8_matmul_fusion, X_int8, W_int8, row_scale, scale_W)
        print(f"Custom INT8 matmul latency: {time_int8:.2f} ms")
        print(f"Speedup over Pytorch BF16: {time_torch / time_int8:.2f}x")
        print(f"Speedup over bitsandbytes INT8 (FP16 output): {time_bnb / time_int8:.2f}x")
        print(f"Speedup over bitsandbytes INT8 (INT32 output): {time_bnb_int32 / time_int8:.2f}x")
        print(f"Speedup over Torch INT8: {time_torch_int8 / time_int8:.2f}x")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1) 
        


        


