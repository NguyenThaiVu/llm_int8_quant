"""
In this script, we verify the correctness of element-wise multiplication
C = A * B
"""

import os 
import numpy as np
import torch
import gemm_cutlass
from utils import *
from utils_transformer_int8 import MinMaxObserverPerLastDim
from utils_layer_int8 import Custom_Element_Wise


if __name__ == "__main__":
    seq_len = 1024
    emb_dim = 4096
    device = 'cuda'
    dtype = torch.bfloat16

    # 1. 2D input
    print("="*50)
    print(f"Testing 2D input (seq_len={seq_len}, emb_dim={emb_dim})")
    A = torch.randn((seq_len, emb_dim), device=device, dtype=dtype)
    B = torch.randn((seq_len, emb_dim), device=device, dtype=dtype)
    element_wise_layer = Custom_Element_Wise(max_length=seq_len).to(device)
    
    # 1. Calibration
    C, _ = element_wise_layer(A, 1.0, B, 1.0)
    print(f"Shape of C: {C.shape}")
    print(f"Sample values of C: {C[:5, :5]}")
    
    # 2. Finish calibration
    element_wise_layer.finish_calibration()
    print(f"output scale: {element_wise_layer.scale_out.shape}")
    
    # 3. Quantized inference
    A_q, scale_A = quantize_row_int8_symmetric_nd(A, scale_dtype=torch.float32)
    B_q, scale_B = quantize_row_int8_symmetric_nd(B, scale_dtype=torch.float32)
    C_q, scale_C = element_wise_layer(A_q, scale_A, B_q, scale_B)
    
    C_deq = C_q.float() * scale_C.unsqueeze(-1)
    print(f"Dequantized output sample values: {C_deq[:5, :5]}")
    
    max_diff = torch.max(torch.abs(C - C_deq)).item()
    print(f"Max difference: {max_diff}")
    mse = torch.mean((C - C_deq) ** 2).item()
    print(f"MSE: {mse}")
    
    