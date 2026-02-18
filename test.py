import torch 
import gemm_cutlass
from utils import *

N = 2048
M = 4096
K = 8192

X = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
W = torch.randn((N, K), dtype=torch.bfloat16, device='cuda')
W_t = W.t().contiguous()

X_q, X_scale = quantize_tensor(X)
W_q, W_scale = quantize_tensor(W)

Y_q = gemm_cutlass.func_pure_int8_matmul_cuda(X_q, W_q, 1.0)

Y_true = torch.matmul(X.float(), W.t().float())

max_diff = torch.max(torch.abs(Y_true - Y_q.float()))
print(f"Max absolute difference between FP32 matmul and quantized matmul: {max_diff.item()}")

mse = torch.mean((Y_true - Y_q.float()) ** 2).item()
print(f"Mean Squared Error between FP32 matmul and quantized matmul: {mse}")

print(f"Sample Y_q: {Y_q[:5, :5]} \n")
print(f"Sample Y_true: {Y_true[:5, :5]} \n")


