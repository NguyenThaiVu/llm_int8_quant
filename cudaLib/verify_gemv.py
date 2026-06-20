"""
Measure latency between int8 and bf16 gemmv
"""

import math
import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":

    batch_size = 16
    in_dims = 4096
    out_dims = 4096 # 1024, 4096, 12288
    # list_in_dims = [2048, 1024 * 5, 4096, 1024 * 6, 8192]
    # list_out_dims = [4096, 1024 * 5, 8192, 1024 * 6, 8192]
    dtype = torch.bfloat16
    n_iter = 100
    
    # for in_dims, out_dims in zip(list_in_dims, list_out_dims):
    print(f"\n\nTesting GEMV with input dim {in_dims} and output dim {out_dims}")
    
    X = torch.randn(batch_size, 1, in_dims, dtype=dtype, device="cuda")

    # init W with kaiming for better quantization results
    W = torch.empty((out_dims, in_dims), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    Y_torch = torch.matmul(X, W.T)
    
    # 1. Measure torch latency
    time_start = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = torch.matmul(X, W.T)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - time_start) / n_iter
    print(f"Latency per inference (torch.matmul): {avg_time * 1000:.2f} ms")
    
    
    X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
    W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
    
    # 2. Measure cutlass latency
    time_start = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = gemm_cutlass.func_w8a8o8_matmul(X_i8, W_i8, scale_x, scale_w)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - time_start) / n_iter
    print(f"Latency per inference (cutlass int8 gemm): {avg_time * 1000:.2f} ms")
    
    # # Measure GEMV latency
    # time_start = time.time()
    # for _ in range(n_iter):
    #     with torch.no_grad():
    #         _ = gemm_cutlass.func_int8_gemv(X_i8, W_i8, scale_x, scale_w, 1.0)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # avg_time = (end_time - time_start) / n_iter
    # print(f"Latency per inference (int8 gemv): {avg_time * 1000:.2f} ms")

    
    # Measure gemv + quantization latency
    
    # Assume output scale via calibration 
    _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
    
    time_start = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = gemm_cutlass.func_int8_gemv_out_int8(X_i8, W_i8,\
                scale_x, scale_w, scale_y, 1.0)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - time_start) / n_iter
    print(f"Latency per inference (int8 gemv + quantization): {avg_time * 1000:.2f} ms")

    y_i8 = gemm_cutlass.func_int8_gemv_out_int8(X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
    Y_deq = y_i8.float() * scale_y.unsqueeze(-1)
    Y_deq = Y_deq.to(dtype)
    
    # Check correctness
    max_diff = torch.max(torch.abs(Y_torch - Y_deq)).item()
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean((Y_torch - Y_deq) ** 2).item()
    print(f"MSE: {mse:.6f}")
    
    print(f"Sample Y_torch: {Y_torch[0, 0, :5]}")
    print(f"Sample Y_deq: {Y_deq[0, 0, :5]}")
    print("\n\n")
    
    