"""
In this script, we verify the correctness of int8 batched matmul
"""
import os 
import torch 
import gemm_cutlass
from utils import *

B = 4
M = 2048
N = 4096
K = 8192

X = torch.randn(B, M, K, device='cuda', dtype=torch.bfloat16)
W = torch.randn(B, N, K, device='cuda', dtype=torch.bfloat16)
W_t = W.transpose(-2, -1).contiguous()

Y_true = torch.matmul(X, W_t)
print(f"Y_true shape: {Y_true.shape}")

X_q, X_scale = quantize_tensor_batched(X)
print(f"X_q shape: {X_q.shape}, X_scale shape: {X_scale.shape}")
W_q, W_scale = quantize_tensor_batched(W)
print(f"W_q shape: {W_q.shape}, W_scale shape: {W_scale.shape}")

# Assume having output scale via calibration
_, Y_true_scale = quantize_tensor_batched(Y_true)

matmul_scale = (X_scale * W_scale) / Y_true_scale
print(f"matmul_scale shape: {matmul_scale.shape}")

Y_q = gemm_cutlass.func_int8_matmul_output_int8_batched(
    X_q, W_q, matmul_scale.to(torch.float32)
)

print(f"Y_q shape: {Y_q.shape}")
Y_deq = Y_q.to(torch.bfloat16) * Y_true_scale.unsqueeze(-1).unsqueeze(-1)
print(f"Y_deq shape: {Y_deq.shape}")
print(f"Y_deq: {Y_deq[0, :5, :5]} \n")
print(f"Y_true: {Y_true[0, :5, :5]} \n")

