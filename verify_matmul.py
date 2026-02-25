"""
In this script, we verify the correctness of matmul.
"""

import os
import torch
import torch.nn as nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import *
from utils_layer_int8 import Custom_Matmul
    

if __name__ == "__main__":
    seq_len = 1024
    emd_dim = 4096
    device = 'cuda'
    d_type = torch.bfloat16
    
    # =====================
    # 1. 2D input
    print(f"Testing 2D input, with shape ({seq_len}, {emd_dim})")
    
    A = torch.randn((seq_len, seq_len), dtype=d_type, device=device)
    B = torch.randn((emd_dim, seq_len), dtype=d_type, device=device)
    
    matmul_layer = Custom_Matmul(max_seq_len=seq_len).to(device)
    
    # 1. Calibration
    C, _ = matmul_layer(A, 1.0, B, 1.0)
    
    # 2. Finish calibration
    matmul_layer.finish_calibration()
    
    # 3. Quantized inference
    A_int8, scale_A = quantize_row_int8_symmetric_nd(A, scale_dtype=torch.float32)
    B_int8, scale_B = quantize_row_int8_symmetric_nd(B, scale_dtype=torch.float32)
    
    print(f"Shape of scale_A: {scale_A.shape}, dtype: {scale_A.dtype}")
    print(f"Shape of scale_B: {scale_B.shape}, dtype: {scale_B.dtype}")
    
    C_int8, scale_C = matmul_layer(A_int8, scale_A, B_int8, scale_B)
    
    C_deq = C_int8.float() * scale_C.unsqueeze(-1)
    print(f"Shape of output C_deq: {C_deq.shape}, dtype: {C_deq.dtype}")
    
    max_diff = torch.max(torch.abs(C - C_deq)).item()
    print(f"Max difference: {max_diff}")
    mse = torch.mean((C - C_deq) ** 2).item()
    print(f"MSE: {mse}")
    
    print(f"Output C (float32): {C[:5, :5]}")
    print(f"Output C_deq (float32): {C_deq[:5, :5]}\n")

    # =====================
    # 2. 3D input
    batch_size = 32
    seq_len = 8
    emd_dim = 128
    print(f"Testing 3D input, with shape ({batch_size}, {seq_len}, {emd_dim})")
    
    A = torch.randn((batch_size, seq_len, seq_len), dtype=d_type, device=device)
    B = torch.randn((batch_size, emd_dim, seq_len), dtype=d_type, device=device)
    
    matmul_layer = Custom_Matmul(num_heads=batch_size, max_seq_len=seq_len).to(device)
    
    # 1. Calibration
    C, _ = matmul_layer(A, 1.0, B, 1.0)
    
    # 2. Finish calibration
    matmul_layer.finish_calibration()
    
    # 3. Quantized inference
    A_int8, scale_A = quantize_row_int8_symmetric_nd(A, scale_dtype=torch.float32)
    B_int8, scale_B = quantize_row_int8_symmetric_nd(B, scale_dtype=torch.float32)
    
    print(f"Shape of A_int8: {A_int8.shape}")
    print(f"Shape of B_int8: {B_int8.shape}")
    print(f"Shape of scale_A: {scale_A.shape}")
    print(f"Shape of scale_B: {scale_B.shape}")
    print()
    
    C_int8, scale_C = matmul_layer(A_int8, scale_A, B_int8, scale_B)
    C_deq = C_int8.float() * scale_C.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(C - C_deq)).item()
    print(f"Max difference: {max_diff}")
    mse = torch.mean((C - C_deq) ** 2).item()
    print(f"MSE: {mse}")
    
    print(f"Output C (float32): {C[0, :5, :5]}")
    print(f"Output C_deq (float32): {C_deq[0, :5, :5]}")
    
    