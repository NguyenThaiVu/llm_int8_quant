"""
In this file, we compare the latency between INT8 GEMV and BF16 GEMV.

From our computation, the break-even point is around 7MB data movement.
If the data movement is less than 7MB, BF16 GEMV has similar latency with INT8 GEMV. 
Because, in this case, the latency is dominated by kernel launch overhead.
"""

import math
import numpy as np
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


def gemv_arithmetic_intensity(X: torch.Tensor,
                              W: torch.Tensor,
                              Y: torch.Tensor):
    """
    Compute algorithmic arithmetic intensity for GEMV-like operation.

    Assumed math:
        Y = X @ W.T

    Supported shapes:
        X: [K], [R, K], or [B, M, K]
        W: [N, K] or [B, N, K]
        Y: [N], [R, N], or [B, M, N]

    Arithmetic intensity:
        AI = ops / bytes_moved

    where:
        ops = 2 * R * N * K   if count_mac_as_two_ops=True
        ops = R * N * K       if count_mac_as_two_ops=False

    bytes_moved is estimated from tensor sizes:
        bytes = X.numel() * X.element_size()
              + W.numel() * W.element_size()
              + Y.numel() * Y.element_size()
    """

    # ----------------------------
    # Infer X shape
    # ----------------------------
    if X.dim() == 1:
        R = 1
        K = X.shape[0]
    elif X.dim() == 2:
        R, K = X.shape
    elif X.dim() == 3:
        B, M, K = X.shape
        R = B * M
    else:
        raise ValueError(f"Unsupported X shape: {tuple(X.shape)}")

    # ----------------------------
    # Infer W shape
    # ----------------------------
    if W.dim() == 2:
        N, K_w = W.shape
    elif W.dim() == 3:
        B_w, N, K_w = W.shape
    else:
        raise ValueError(f"Unsupported W shape: {tuple(W.shape)}")

    if K_w != K:
        raise ValueError(f"K mismatch: X has K={K}, but W has K={K_w}")

    # ----------------------------
    # Operation count
    # ----------------------------
    ops = 2 * R * N * K

    # ----------------------------
    # Tensor memory traffic
    # ----------------------------
    bytes_x = X.numel() * X.element_size()
    bytes_w = W.numel() * W.element_size()
    bytes_y = Y.numel() * Y.element_size()

    total_bytes = bytes_x + bytes_w + bytes_y

    AI = ops / total_bytes if total_bytes > 0 else 0.0
    return AI


if __name__ == "__main__":

    # ======================= GEMV BF16 vs INT8 with 2D input ==========================
    seq_len = 1
    # list_in_dims = [1024, 2048, 4096, 4096, 8192]
    # list_out_dims = [1024, 2048, 4096, 8192, 8192]
    list_in_dims = [4096]
    list_out_dims = [8192]
    dtype = torch.bfloat16
    
    list_total_data_move = []
    list_latency_bf16 = []
    list_flops_bf16 = []
    list_latency_int8 = []
    list_flops_int8 = []
    
    for in_dims, out_dims in zip(list_in_dims, list_out_dims):
        print("\n\n==================================================")
    
        X = torch.randn((seq_len, in_dims), dtype=dtype, device="cuda")
        W = init_weights((out_dims, in_dims), dtype)
        print(f"Testing GEMV with input: {X.shape}, weight: {W.shape}")
        
        for _ in range(3):
            Y_torch = torch.matmul(X, W.transpose(-2, -1))
        
        total_data_move = compute_total_data_movement(X, W, Y_torch, unit='MB')
        print(f"Total data movement (BF16 input/output): {total_data_move:.2f} MB\n")
        
        # =========================================================================
        #  1. Measure torch latency (baseline)
        avg_time = measure_time(torch.matmul, X, W.transpose(-2, -1), repeat=10)
        print(f"Latency per inference (torch.matmul): {avg_time:.4f} ms")
        torch_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
        print(f"Throughput (torch.matmul): {torch_flops_per_sec / 1e9:.2f} GFLOPS\n")
        
        X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
        W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
        # Assume output scale via calibration 
        _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
        
        # =========================================================================
        # 2. Measure GEMV BF16 latency
        avg_time = measure_time(gemm_cutlass.func_bf16_gemv, X, W, 1.0, repeat=10)
        print(f"Latency per inference (GEMV BF16 output): {avg_time:.4f} ms")
        cutlass_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
        print(f"Throughput (GEMV BF16 output): {cutlass_flops_per_sec / 1e9:.2f} GFLOPS \n")
        ai_bf16 = gemv_arithmetic_intensity(X, W, Y_torch)
        print(f"Arithmetic Intensity (GEMV BF16): {ai_bf16:.4f} (FLOP/Byte)\n")

        list_latency_bf16.append(avg_time)
        list_flops_bf16.append(cutlass_flops_per_sec)
        
        # =========================================================================
        # 2. Measure GEMV INT8 with BF16 output latency
        avg_time = measure_time(gemm_cutlass.func_i8_gemv_out_bf16,\
                                X_i8, W_i8, scale_x, scale_w, 1.0, repeat=10)
        print(f"Latency per inference (int8 gemv + bf16 output): {avg_time:.4f} ms")
        int8_gemv_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
        print(f"Throughput (int8 gemv + bf16 output): {int8_gemv_flops_per_sec / 1e9:.2f} GFLOPS \n")
        total_data_move = compute_total_data_movement(X_i8, W_i8, Y_torch, unit='MB')
        print(f"Total data movement (INT8 input + BF16 output): {total_data_move:.2f} MB\n")
        ai_i8_bf16 = gemv_arithmetic_intensity(X_i8, W_i8, Y_torch)
        print(f"Arithmetic Intensity (GEMV INT8 + BF16 output): {ai_i8_bf16:.4f} (FLOP/Byte)\n")
        
        # =========================================================================
        # 3. Measure gemv + quantization latency
        avg_time = measure_time(gemm_cutlass.func_i8_gemv_out_i8,\
                            X_i8, W_i8, scale_x, scale_w, scale_y, 1.0, repeat=10)
        print(f"Latency per inference (int8 gemv + quantization): {avg_time:.4f} ms")
        int8_gemv_flops_per_sec = compute_flops_per_sec_gemv(seq_len, in_dims, out_dims, avg_time)
        print(f"Throughput (int8 gemv + quantization): {int8_gemv_flops_per_sec / 1e9:.2f} GFLOPS")
        Y_i8 = gemm_cutlass.func_i8_gemv_out_i8(X_i8, W_i8, scale_x, scale_w, scale_y, 1.0)
        total_data_move = compute_total_data_movement(X_i8, W_i8, Y_i8, unit='MB')
        print(f"Total data movement (INT8 input + INT8 output): {total_data_move:.2f} MB\n")
        ai_i8 = gemv_arithmetic_intensity(X_i8, W_i8, Y_i8)
        print(f"Arithmetic Intensity (GEMV INT8 + INT8 output): {ai_i8:.4f} (FLOP/Byte)\n")
    