"""
Measure latency between int8 and bf16 gemmv
"""

import math
import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

# def compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, latency_sec):
#     flops = 2 * seq_len * in_dims * out_dims
#     flops_per_sec = flops / latency_sec
#     return flops_per_sec

def compute_flops_per_sec_gemv(X, W, latency_ms, unit='GFLOPS'):
    if X.dim() == 2:
        seq_len = X.size(0)
        in_dims = X.size(1)
        out_dims = W.size(0)
    elif X.dim() == 3:
        seq_len = X.size(1)
        in_dims = X.size(2)
        out_dims = W.size(0)
    else:
        raise ValueError(f"Input X must be 2D or 3D, get {X.dim()}D")
    latency_sec = latency_ms / 1000.0
    
    flops = 2 * seq_len * in_dims * out_dims
    flops_per_sec = flops / latency_sec
    
    if unit == 'GFLOPS':
        flops_per_sec /= 1e9
    elif unit == 'TFLOPS':
        flops_per_sec /= 1e12
    
    return flops_per_sec
    
def compute_total_data_movement_matmul(X, W):
    if X.dim() == 2:
        M = X.size(0)
        K = X.size(1)
        N = W.size(0)
        total_data_move = (M * K + K * N + M * N) * X.element_size()
    elif X.dim() == 3:
        B = X.size(0)
        M = X.size(1)
        K = X.size(2)
        N = W.size(0)
        total_data_move = (B * M * K + K * N + B * M * N) * X.element_size()
    else:
        raise ValueError(f"Input X must be 2D or 3D, get {X.dim()}D")
    return total_data_move

def compute_arithmetic_intensity(X, W):
    flops = 2 * X.numel() * W.size(0)  #
    total_data_move = compute_total_data_movement_matmul(X, W)
    return flops / total_data_move


def init_weight_matrix(out_dims, in_dims, dtype=torch.bfloat16):
    W = torch.empty((out_dims, in_dims), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W

if __name__ == "__main__":

    d_type = torch.bfloat16
    # in_dims = 4096
    # out_dims = 8192
    n_iter = 1_000
    
    list_in_dims = [1024, 2048, 4096, 4096, 8192, 8192]
    list_out_dims = [1024, 2048, 4096, 8192, 8192, 16384]
    dtype = torch.bfloat16
    n_iter = 1_000
    
    for in_dims, out_dims in zip(list_in_dims, list_out_dims):
        print("\n\n" + "*" * 60)
        print(f"Testing GEMV with input dim {in_dims} and output dim {out_dims}")
    
        X = torch.randn((1, in_dims), dtype=d_type, device="cuda")
        W = init_weight_matrix(out_dims, in_dims, dtype=d_type)
        
        # Warm up GPU
        for _ in range(100):
            Y_torch = torch.matmul(X, W.T)
            
        # 1. Measure torch latency
        torch_time = measure_time(torch.matmul, X, W.T, repeat=n_iter)
        print(f"torch.matmul latency: {torch_time:.6f} ms")
        
        # 2. Measure GEMV BF16 latency
        gemv_bf16_time = measure_time(gemm_cutlass.func_bf16_gemv, X, W, 1.0, repeat=n_iter)
        print(f"GEMV BF16 latency: {gemv_bf16_time:.6f} ms")
        FLOPS_bf16 = compute_flops_per_sec_gemv(X, W, gemv_bf16_time, unit='GFLOPS')
        print(f"BF16 GEMV: {FLOPS_bf16:.2f} GFLOPS")
        
        X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
        W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
        
        # 3. GEMV W8A8
        gemv_w8a8_time = measure_time(gemm_cutlass.func_i8_gemv_out_bf16, X_i8, W_i8,\
                            scale_x, scale_w, 1.0, repeat=n_iter)
        print(f"GEMV W8A8 latency: {gemv_w8a8_time:.6f} ms")
        FLOPS_w8a8 = compute_flops_per_sec_gemv(X_i8, W_i8, gemv_w8a8_time, unit='GFLOPS')
        print(f"W8A8 GEMV: {FLOPS_w8a8:.2f} GFLOPS")
        
        # 3. Measure GEMV W8A8O8
        _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
        gemv_int8_time = measure_time(gemm_cutlass.func_i8_gemv_out_i8, X_i8, W_i8,\
                            scale_x, scale_w, scale_y, 1.0, repeat=n_iter)
        print(f"GEMV INT8 latency: {gemv_int8_time:.6f} ms")
        
        FLOPS_i8 = compute_flops_per_sec_gemv(X_i8, W_i8, gemv_int8_time, unit='GFLOPS')
        print(f"INT8 GEMV: {FLOPS_i8:.2f} GFLOPS")
        throughput_gain = FLOPS_i8 / FLOPS_bf16
        print(f"Throughput gain (INT8 / BF16): {throughput_gain:.2f}x")
        