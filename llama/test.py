import os 
import torch
from utils_quant import measure_time, quantize_row_int8_symmetric_nd
import bitsandbytes as bnb
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant

def bnb_int8_and_dequantize(x_q, w_q, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N) float32
    w_scales: (M,) float32
    Returns:
      y: (B, N, M) float32
    """
    y_int = int8_linear_matmul(x_q, w_q)  # (B, N, M) int32
    y = int8_mm_dequant(y_int, x_scales, w_scales)
    y = torch.ops.bitsandbytes.int8_mm_dequant.default(y_int, x_scales, w_scales)
    return y.to(output_dtype)


input_dims = 2048
hidden_dims = 8192
output_dims = 8192

dtype = torch.bfloat16

X = torch.randn(input_dims, hidden_dims, dtype=dtype).cuda()
W = torch.randn(output_dims, hidden_dims, dtype=dtype).cuda()
X_bf16 = X.to(dtype)
W_bf16 = W.t().to(dtype)

# ==========================
# 1. Measure the correctness
# ==========================
Y_true = torch.matmul(X_bf16, W_bf16)

X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
W_int8, scale_w = quantize_row_int8_symmetric_nd(W)

Y_deq = bnb_int8_and_dequantize(X_int8, W_int8, scale_x, scale_w)

max_diff = torch.max(torch.abs(Y_deq.float() - Y_true.float())).item()
print("Max absolute difference:", max_diff)
mse = torch.mean((Y_deq.float() - Y_true.float()) ** 2).item()
print(f"Mean Squared Error: {mse} \n")

print(f"Sample of Y_true: {Y_true[:5, :5]}")
print(f"Sample of Y_deq: {Y_deq[:5, :5]}")

# ===========================
# 2. Measure the latency
# ===========================
torch_time = measure_time(torch.matmul, X_bf16, W_bf16)
print(f"PyTorch matmul latency: {torch_time:.2f} ms")

int8_time = measure_time(bnb_int8_and_dequantize,
                         X_int8, W_int8, scale_x, scale_w)
print(f"Int8 matmul latency: {int8_time:.2f} ms")