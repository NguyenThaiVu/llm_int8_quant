"""
In this script, we verify the correctness of matmul.
"""

import os
import torch
import torch.nn as nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import *


class Custom_Matmul(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        if self.num_heads == 1:
            self.out_observer = MinMaxObserverPerLastDim(max_seq_len=self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.max_seq_len)) 
            
        elif self.num_heads > 1:
            self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len)) 
        else:
            raise ValueError(f"num_heads should be >= 1, got {num_heads}")
        self.is_quantized = False
        
    def forward(self, A, scale_A, B, scale_B):
        """
        A: (M, K)
        B: (N, K)
        C = A @ B^T -> (M, N)
        """
        
        if self.is_quantized == False:
            if A.dim() == 2:
                C = torch.matmul(A, B.T)
                self.out_observer(C)
            elif A.dim() == 3:
                C = torch.matmul(A, B.transpose(-1, -2))
                self.out_observer(C)
            return C, 1.0
        else:
            if A.dim() == 2:
                seq_len = A.shape[0]
                
                scale_A = scale_A[:seq_len].to(torch.float32)
                scale_B = scale_B[:seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
                
                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            elif A.dim() == 3:
                batch_size, seq_len, _ = A.shape
                
                scale_A = scale_A[:, :seq_len].to(torch.float32)
                scale_B = scale_B[:, :seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)
                
                print(f"Shape of scale_B: {scale_B.shape}")
                
                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale_batched(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            else:
                raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().cuda()
        self.is_quantized = True
    

if __name__ == "__main__":
    seq_len = 1024
    emd_dim = 8192
    device = 'cuda'
    d_type = torch.bfloat16
    
    # =====================
    # 1. 2D input
    print(f"Testing 2D input, with shape ({seq_len}, {emd_dim})")
    
    A = torch.randn((seq_len, emd_dim), dtype=d_type, device=device)
    B = torch.randn((seq_len, emd_dim), dtype=d_type, device=device)
    
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
    batch_size = 16
    print(f"Testing 3D input, with shape ({batch_size}, {seq_len}, {emd_dim})")
    
    A = torch.randn((batch_size, seq_len, emd_dim), dtype=d_type, device=device)
    B = torch.randn((batch_size, seq_len, emd_dim), dtype=d_type, device=device)
    
    matmul_layer = Custom_Matmul(num_heads=batch_size, max_seq_len=seq_len).to(device)
    
    # 1. Calibration
    C, _ = matmul_layer(A, 1.0, B, 1.0)
    
    # 2. Finish calibration
    matmul_layer.finish_calibration()
    
    # 3. Quantized inference
    A_int8, scale_A = quantize_row_int8_symmetric_nd(A, scale_dtype=torch.float32)
    B_int8, scale_B = quantize_row_int8_symmetric_nd(B, scale_dtype=torch.float32)
    
    C_int8, scale_C = matmul_layer(A_int8, scale_A, B_int8, scale_B)
    C_deq = C_int8.float() * scale_C.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(C - C_deq)).item()
    print(f"Max difference: {max_diff}")
    mse = torch.mean((C - C_deq) ** 2).item()
    print(f"MSE: {mse}")
    
    print(f"Output C (float32): {C[0, :5, :5]}")
    print(f"Output C_deq (float32): {C_deq[0, :5, :5]}")
    
    