"""
Measure latency between int8 and bf16 gemmv
"""

import math
import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

def compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, latency_sec):
    flops = 2 * seq_len * in_dims * out_dims
    flops_per_sec = flops / latency_sec
    return flops_per_sec
    
    

if __name__ == "__main__":

    batch_size = 32
    seq_len = 1
    in_dims = 1024 # 128, 4096
    out_dims = 128 # 1024, 4096, 12288
    # list_in_dims = [128, 4096, 4096]
    # list_out_dims = [2048, 4096, 12288]
    dtype = torch.bfloat16
    n_iter = 100
    
    # for in_dims, out_dims in zip(list_in_dims, list_out_dims):
    # print(f"\n\nTesting GEMV with input dim {in_dims} and output dim {out_dims}")
    
    X = torch.randn((batch_size, seq_len, in_dims), dtype=dtype, device="cuda")

    # init W with kaiming for better quantization results
    W = torch.empty((batch_size, out_dims, in_dims), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}\n")
    for _ in range(3):
        Y_torch = torch.matmul(X, W.transpose(-2, -1))
    
    # =========================================================================
    # 1. Measure torch latency
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = torch.matmul(X, W.transpose(-2, -1))
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter / 1000
    print(f"Latency per inference (torch.matmul): {avg_time * 1000:.4f} ms")
    torch_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    print(f"Throughput (torch.matmul): {torch_flops_per_sec / 1e9:.2f} GFLOPS\n")
    
    X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
    W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
    
    # =========================================================================
    # 2. Measure cutlass latency
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = gemm_cutlass.func_w8a8o8_matmul(X_i8, W_i8, scale_x, scale_w)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter / 1000
    print(f"Latency per inference (cutlass int8 gemm): {avg_time * 1000:.4f} ms")
    cutlass_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    print(f"Throughput (cutlass int8 gemm): {cutlass_flops_per_sec / 1e9:.2f} GFLOPS \n")

    
    # =========================================================================
    # 3. Measure gemv + quantization latency
    
    # Assume output scale via calibration 
    _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = gemm_cutlass.func_int8_gemv_out_int8(X_i8, W_i8,\
                scale_x, scale_w, scale_y, 1.0)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter / 1000
    print(f"Latency per inference (int8 gemv + quantization): {avg_time * 1000:.4f} ms")
    int8_gemv_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    print(f"Throughput (int8 gemv + quantization): {int8_gemv_flops_per_sec / 1e9:.2f} GFLOPS")

    y_i8 = gemm_cutlass.func_int8_gemv_out_int8(X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
    Y_deq = y_i8.float() * scale_y.unsqueeze(-1)
    Y_deq = Y_deq.to(dtype)
    
    # Check correctness
    max_diff = torch.max(torch.abs(Y_torch - Y_deq)).item()
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean((Y_torch - Y_deq) ** 2).item()
    print(f"MSE: {mse:.6f}\n")
    
    # =========================================================================
    # 4. Measure gemv + quantization latency (WARP version)
    
    # Assume output scale via calibration 
    _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = gemm_cutlass.func_int8_gemv_out_int8_warp(X_i8, W_i8,\
                scale_x, scale_w, scale_y, 1.0)
    end_event.record()
    torch.cuda.synchronize()
    avg_time = start_event.elapsed_time(end_event) / n_iter / 1000
    print(f"Latency per inference (int8 gemv + quantization): {avg_time * 1000:.4f} ms")
    int8_gemv_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    print(f"Throughput (int8 gemv + quantization): {int8_gemv_flops_per_sec / 1e9:.2f} GFLOPS")

    y_i8 = gemm_cutlass.func_int8_gemv_out_int8_warp(X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
    Y_deq = y_i8.float() * scale_y.unsqueeze(-1)
    Y_deq = Y_deq.to(dtype)
    
    # Check correctness
    max_diff = torch.max(torch.abs(Y_torch - Y_deq)).item()
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean((Y_torch - Y_deq) ** 2).item()
    print(f"MSE: {mse:.6f}\n")
    
    # print(f"Sample Y_torch: {Y_torch[0, :5]}")
    # print(f"Sample Y_deq: {Y_deq[0, :5]}")
    # print("\n\n")
    
    