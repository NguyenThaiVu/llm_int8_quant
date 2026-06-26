"""
In this file, we compare the latency between INT8 GEMV and BF16 GEMV.

From our computation, the break-even point is around 7MB data movement.
If the data movement is less than 7MB, BF16 GEMV has similar latency with INT8 GEMV. 
Because, in this case, the latency is dominated by kernel launch overhead.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch
import gemm_cutlass
from utils_quant import *


def init_weights(shape, dtype):
    # init W with kaiming uniform initialization
    W = torch.empty(shape, dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W

def compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, latency_sec):
    flops = 2 * seq_len * in_dims * out_dims
    flops_per_sec = flops / latency_sec
    return flops_per_sec

def compute_flops_batch_matmul(B, M, K, N, latency_sec):
    flops = 2 * B * M * K * N
    flops_per_sec = flops / latency_sec
    return flops_per_sec
    
def compute_total_data_movement(X, W, Y, unit="byte"):
    total_data_move = X.numel() * X.element_size() +\
                W.numel() * W.element_size() + Y.numel() * Y.element_size()
    
    if unit == "byte":
        pass
    elif unit == "KB":
        total_data_move /= 1024
    elif unit == "MB":
        total_data_move /= 1024 * 1024
        total_data_move = round(total_data_move, 2)
    elif unit == "GB":
        total_data_move /= 1024 * 1024 * 1024
        total_data_move = round(total_data_move, 2)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    
    return total_data_move


if __name__ == "__main__":

    # ======================= GEMV BF16 vs INT8 with 2D input ==========================
    # seq_len = 1
    # list_in_dims = [512, 1024, 2048, 4096, 4096, 8192]
    # list_out_dims = [512, 1024, 2048, 4096, 8192, 8192]
    # dtype = torch.bfloat16
    
    # list_total_data_move = []
    # list_latency_bf16 = []
    # list_flops_bf16 = []
    # list_latency_int8 = []
    # list_flops_int8 = []
    
    # for in_dims, out_dims in zip(list_in_dims, list_out_dims):
    #     print(f"\n\nTesting GEMV with input dim {in_dims} and output dim {out_dims}")
    
    #     X = torch.randn((seq_len, in_dims), dtype=dtype, device="cuda")
    #     W = init_weights((out_dims, in_dims), dtype)
        
    #     # print(f"Input shape: {X.shape}")
    #     # print(f"Weight shape: {W.shape}")
    #     for _ in range(3):
    #         Y_torch = torch.matmul(X, W.transpose(-2, -1))
    #     # print(f"Output shape: {Y_torch.shape}")
        
    #     total_data_move = compute_total_data_movement_matmul(X, W, unit='MB')
    #     print(f"Total data movement (BF16 input/output): {total_data_move:.2f} MB \n")
    #     list_total_data_move.append(total_data_move)
        
    #     # =========================================================================
    #     # 1. Measure torch latency (baseline)
    #     # avg_time = measure_time(torch.matmul, X, W.transpose(-2, -1))
    #     # print(f"Latency per inference (torch.matmul): {avg_time:.4f} ms")
    #     # torch_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    #     # print(f"Throughput (torch.matmul): {torch_flops_per_sec / 1e9:.2f} GFLOPS\n")
        
    #     X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
    #     W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
    #     # Assume output scale via calibration 
    #     _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
        
    #     # =========================================================================
    #     # 2. Measure GEMV BF16 latency
    #     avg_time = measure_time(gemm_cutlass.func_bf16_gemv, X, W, 1.0)
    #     print(f"Latency per inference (GEMV BF16 output): {avg_time:.4f} ms")
    #     cutlass_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    #     print(f"Throughput (GEMV BF16 output): {cutlass_flops_per_sec / 1e9:.2f} GFLOPS \n")

    #     list_latency_bf16.append(avg_time)
    #     list_flops_bf16.append(cutlass_flops_per_sec)
        
    #     # =========================================================================
    #     # 3. Measure gemv + quantization latency (WARP version)
    #     avg_time = measure_time(gemm_cutlass.func_int8_gemv_out_int8_warp,\
    #                         X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
    #     print(f"Latency per inference (int8 gemv + quantization): {avg_time:.4f} ms")
    #     int8_gemv_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
    #     print(f"Throughput (int8 gemv + quantization): {int8_gemv_flops_per_sec / 1e9:.2f} GFLOPS")
    
    #     list_latency_int8.append(avg_time)
    #     list_flops_int8.append(int8_gemv_flops_per_sec)
    
    
    # ======================= GEMV BF16 vs INT8 with 3D input ==========================
    seq_len = 1
    L = 1024  # sequence length
    n_heads = 32
    h_dims = 128
    # list_L = [512, 1024, 2048]
    # list_n_heads = [32, 40, 64]
    # list_out_dims = [512, 1024, 2048, 4096, 8192, 8192]
    dtype = torch.bfloat16
    
    list_total_data_move = []
    list_latency_bf16 = []
    list_flops_bf16 = []
    list_latency_int8 = []
    list_flops_int8 = []
    
    # for in_dims, out_dims in zip(list_in_dims, list_out_dims):
        # print(f"\n\nTesting GEMV with input dim {in_dims} and output dim {out_dims}")
    
    X = torch.randn((n_heads, seq_len, h_dims), dtype=dtype, device="cuda")
    W = init_weights((n_heads, L, h_dims), dtype)
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    for _ in range(3):
        Y_torch = torch.matmul(X, W.transpose(-2, -1))
    print(f"Output shape: {Y_torch.shape}")
    
    total_data_move = compute_total_data_movement(X, W, Y_torch, unit='MB')
    print(f"Total data movement (BF16 input/output): {total_data_move:.2f} MB \n")
    list_total_data_move.append(total_data_move)
    
    # =========================================================================
    # 1. Measure torch latency (baseline)
    avg_time = measure_time(torch.matmul, X, W.transpose(-2, -1))
    print(f"Latency per inference (torch.matmul): {avg_time:.4f} ms")
    torch_flops = compute_flops_batch_matmul(n_heads, seq_len, h_dims, L, avg_time)
    print(f"Throughput (torch.matmul): {torch_flops / 1e9:.2f} GFLOPS\n")
    
    X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
    W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
    # Assume output scale via calibration 
    _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
    
    # =========================================================================
    # 2. Measure GEMV BF16 latency
    avg_time = measure_time(gemm_cutlass.func_bf16_gemv, X, W, 1.0)
    print(f"Latency per inference (GEMV BF16 output): {avg_time:.4f} ms")
    bf16_gemv_flops = compute_flops_batch_matmul(n_heads, seq_len, h_dims, L, avg_time)
    print(f"Throughput (GEMV BF16 output): {bf16_gemv_flops / 1e9:.2f} GFLOPS \n")

    list_latency_bf16.append(avg_time)
    list_flops_bf16.append(bf16_gemv_flops)
    
    # =========================================================================
    # 3. Measure gemv + quantization latency 
    avg_time = measure_time(gemm_cutlass.func_i8_gemv_out_i8,\
                        X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
    print(f"Latency per inference (int8 gemv + quantization): {avg_time:.4f} ms")
    int8_gemv_flops = compute_flops_batch_matmul(n_heads, seq_len, h_dims, L, avg_time)
    print(f"Throughput (int8 gemv + quantization): {int8_gemv_flops / 1e9:.2f} GFLOPS\n")

    list_latency_int8.append(avg_time)
    list_flops_int8.append(int8_gemv_flops)
    