import os 
import torch
import gemm_cutlass

input_dims = 1024
hidden_dims = 2048
output_dims = 4096

dtype = torch.int8

X = torch.randint(-128, 127, (input_dims, hidden_dims), dtype=dtype).cuda()
W = torch.randint(-128, 127, (output_dims, hidden_dims), dtype=dtype).cuda()

Y = gemm_cutlass.func_int8_matmul(X, W, 1.0)

true = torch.matmul(X.float(), W.t().float())

print("Max absolute difference:", torch.max(torch.abs(Y.float() - true)))
print(f"Samples of Y: {Y[:5, :5]}")
print(f"Samples of true: {true[:5, :5]}")