"""
In this script, we profile the performance of element-wise operations
- BF16
- INT8 with quantization and dequantization
"""

import os 
import numpy as np  
import torch
import torch.nn as nn
import gemm_cutlass
from utils import *

device = 'cuda'

N = 1024 * 16
d_type = torch.bfloat16
# d_type = torch.float32
acc_dtype = torch.int32

A = torch.randn((N, N), dtype=d_type, device=device) * np.random.randint(1, 5)
B = torch.randn((N, N), dtype=d_type, device=device) * np.random.randint(1, 5)

C_true = torch.mul(A, B)
bf16_time = measure_time(
    torch.mul,
    A,
    B,
    repeat=10
)
print(f"\nElement-wise bf16 time: {bf16_time:.3f} ms")

# Quantize A and B
A_q, scale_A = quantize_tensor(A)
B_q, scale_B = quantize_tensor(B)

# Assume we have scale_C via calibration
scale_C = (C_true.abs().max() / 127.0).to(d_type)

# C_acc = A_q.to(d_type) * B_q.to(d_type)
# C_q = (C_acc * (scale_A * scale_B / scale_C)).round()
# C_q = torch.clamp(C_q, -128, 127).to(torch.int8)

C_q = gemm_cutlass.func_element_wise_mul_int8(
    A_q,
    scale_A.item(),
    B_q,
    scale_B.item(),
    scale_C.item()
)

element_wise_mul_int8_time = measure_time(
    gemm_cutlass.func_element_wise_mul_int8,
    A_q,
    scale_A.item(),
    B_q,
    scale_B.item(),
    scale_C.item(),
    repeat=10
)
print(f"\nElement-wise int8 time: {element_wise_mul_int8_time:.3f} ms")


# Verify correctness
C_deq = C_q.to(d_type) * scale_C

if torch.allclose(C_true, C_deq, atol=1.0):
    print("\nElement-wise int8 quantization correct.\n")
else:
    print("\n===== [ERROR] Element-wise int8 quantization incorrect =====\n")

max_diff = (C_true - C_deq).abs().max()
print(f"Max difference: {max_diff.item()}")
mse = torch.mean((C_true - C_deq).pow(2)).item()
print(f"MSE: {mse}")
    
print(f"C_true: {C_true[0, :10]}")
print()
print(f"C_deq: {C_deq[0, :10]}")

