"""
In this file, we profile the performance of INT8 matrix multiplication.
"""
import os 
import torch
import gemm_cutlass
from utils import *

# B = 4
# M = 4096
# N = 4096
# K = 4096
dtype = torch.bfloat16

list_B = [1, 2, 4, 8]
list_M = [2048, 4096, 4096, 8192]
list_N = [2048, 4096, 8192, 8192]
list_K = [2048, 4096, 8192, 8192]

for B in list_B:
    print(f"================= Batch Size: {B} =================")
    for M, N, K in zip(list_M, list_N, list_K):
        print(f"Profiling B={B}, M={M}, N={N}, K={K}")

        X = torch.rand(B, M, K).to(dtype).cuda()
        W = torch.rand(B, N, K).to(dtype).cuda()
        W_t = W.transpose(1, 2).contiguous()

        torch_time = measure_time(torch.bmm, X, W_t, repeat=100)
        print(f"torch bmm time: {torch_time:.2f} ms")

        X_q, scale_X = quantize_tensor_batched(X)
        W_q, scale_W = quantize_tensor_batched(W)
        scale = scale_X * scale_W
        scale = scale.to(torch.float32)
        # Make scale 1D tensor
        if len(scale.shape) > 1:
            scale = scale.squeeze()
        elif B == 1:
            scale = scale.unsqueeze(0)

        W8A8O8_time = measure_time(gemm_cutlass.func_int8_matmul_output_int8_batched, 
                                X_q, W_q, scale, repeat=100)
        print(f"INT8 bmm time: {W8A8O8_time:.2f} ms")
        print()

