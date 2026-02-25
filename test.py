import os 
import torch 
import gemm_cutlass
from utils import *

input_dims = 4096
output_dims = 8192
dtype = torch.bfloat16

X = torch.randn((input_dims, output_dims), dtype=dtype, device='cuda')

X_int8, scale_x = quantize_row_int8_symmetric_nd(X)

X_deq = X_int8.float() * scale_x.unsqueeze(-1)
X_deq = X_deq.to(dtype)

max_diff = (X - X_deq).abs().max().item()
print(f"Max diff: {max_diff:.6f}")
mse = ((X - X_deq) ** 2).mean().item()
print(f"MSE: {mse:.6f}")