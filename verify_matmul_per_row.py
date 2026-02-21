"""
In this script, we verify the correctness of INT8 matmul, 
with per-row output scaling.
"""

import os 
import torch 
import gemm_cutlass
from utils import *

M = 4096
N = 8192
K = 1024 * 5 

dtype = torch.bfloat16
device = 'cuda'

# 1. 2D matrix 
X = torch.randn(M, K, dtype=dtype, device=device)

# W = torch.empty(N, K, dtype=dtype, device=device)
# torch.nn.init.kaiming_normal_(W)
W = torch.randn(N, K, dtype=dtype, device=device)
print("="* 50)
print(f"1. Shapes - X: {X.shape}, W: {W.shape}")

# True computation
W_t = W.t().contiguous()
Y_true = torch.matmul(X, W_t)

# Quantization Computation
X_q, scale_x = quantize_row_wise_tensor(X)
W_q, scale_w = quantize_tensor(W)
scale_x = scale_x.to(torch.float32)
scale_w = scale_w.to(torch.float32)

# Assume output scale via calibration
_, scale_y = quantize_row_wise_tensor(Y_true)
scale_y = scale_y.to(torch.float32)

Y_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale(X_q, W_q, 
                                                           scale_x * scale_w / scale_y)

Y_deq = Y_q.to(torch.float32) * scale_y.view(-1, 1)

max_diff = torch.max(torch.abs(Y_deq - Y_true))
print(f'Max difference: {max_diff.item()}')
mse = torch.mean((Y_deq - Y_true) ** 2).item()
print(f'MSE: {mse}')

print(f"Sample Y_true: {Y_true[:5, :5]}\n")
print(f"Sample Y_deq: {Y_deq[:5, :5]}")

# 2. Batched 3D matrix
print()
print("="* 50)
B = 4
X_b = X.repeat(B, 1, 1)
W_b = W.repeat(B, 1, 1)
W_b_t = W_b.transpose(-1, -2).contiguous()

Y_true_b = torch.matmul(X_b, W_b_t)

X_q_b, scale_x_b = quantize_row_int8_symmetric_nd(X_b)
W_q_b, scale_w_b = quantize_tensor_batched(W_b)
scale_x_b = scale_x_b.to(torch.float32)
scale_w_b = scale_w_b.to(torch.float32)

# Assume output scale via calibration
_, scale_y_b = quantize_row_int8_symmetric_nd(Y_true_b)
scale_y_b = scale_y_b.to(torch.float32)

print(f"Shape of scale_x_b: {scale_x_b.shape}, scale_w_b: {scale_w_b.shape}, scale_y_b: {scale_y_b.shape}")
scale = scale_x_b * scale_w_b.unsqueeze(-1) / scale_y_b
print(f"Shape of scale (broadcasted): {scale.shape}")

Y_q_b = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(X_q_b, W_q_b, 
                                                                      scale)

Y_deq_b = Y_q_b.to(torch.float32) * scale_y_b.unsqueeze(-1)

max_diff = torch.max(torch.abs(Y_deq_b - Y_true_b))
print(f'Max difference (batched): {max_diff.item()}')
mse = torch.mean((Y_deq_b - Y_true_b) ** 2).item()
print(f'MSE (batched): {mse}')

print(f"Sample Y_true_b: {Y_true_b[:1, :5, :5]}\n")
print(f"Sample Y_deq_b: {Y_deq_b[:1, :5, :5]}")


# Measure latency
n_iter = 10
torch_matmul_time = measure_time(torch.matmul, X, W_t, repeat=n_iter)
print(f"torch.matmul time: {torch_matmul_time:.2f} ms")

gemm_cutlass_time = measure_time(gemm_cutlass.func_int8_matmul_out_int8_per_row_scale,
                                X_q, W_q, scale_x * scale_w / scale_y,
                                repeat=n_iter)
print(f"gemm_cutlass int8 matmul with per-row scale time: {gemm_cutlass_time:.2f} ms")
