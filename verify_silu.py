"""
In this module, 
we verify the correctness of the SiLU (Sigmoid Linear Unit) activation.
"""

import os 
import torch
import gemm_cutlass
from utils import *

class SiLU_Module(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

if __name__ == "__main__":
    
    seq_len = 1024
    emb_dim = 4096
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 1. 2D input 
    print("="* 50)
    print(f"2D SiLU Activation with shape ({seq_len}, {emb_dim})")
    X = torch.randn((seq_len, emb_dim), device=device, dtype=dtype)
    silu_module = SiLU_Module().to(device)
    Y = silu_module(X)
    
    print(f"Shape of Y: {Y.shape}")
    print(f"Sample values of Y: {Y[:5, :5]}")
    
    # Quantization
    # X_int8, scale_X = quantize_tensor(X)
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    scale_X = scale_X.to(torch.float32)
    
    # Assume scale_Y is computed via calibration
    # _, scale_Y = quantize_tensor(Y)
    _, scale_Y = quantize_row_int8_symmetric_nd(Y)
    scale_Y = scale_Y.to(torch.float32)
    
    Y_q = gemm_cutlass.func_apply_silu_int8(X_int8, scale_X, scale_Y)
    Y_deq = Y_q * scale_Y.unsqueeze(-1)
    
    print(f"Shape of Y_deq: {Y_deq.shape}")
    print(f"Sample values of Y_deq: {Y_deq[:5, :5]}")
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max absolute difference: {max_diff.item()}")    
    mse = torch.mean((Y - Y_deq) ** 2)
    print(f"Mean Squared Error: {mse.item()}")
    
    # 2. 3D input 
    print("="* 50)
    batch_size = 4
    print(f"3D SiLU Activation with shape ({batch_size}, {seq_len}, {emb_dim})")
    X = torch.randn((batch_size, seq_len, emb_dim), device=device, dtype=dtype)
    silu_module = SiLU_Module().to(device)
    Y = silu_module(X)
    
    print(f"Shape of Y: {Y.shape}")
    print(f"Sample values of Y: {Y[0, :5, :5]}")
    
    # Quantization
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    scale_X = scale_X.to(torch.float32)
    
    _, scale_Y = quantize_row_int8_symmetric_nd(Y)
    scale_Y = scale_Y.to(torch.float32)
    
    Y_q = gemm_cutlass.func_apply_silu_int8(X_int8, scale_X, scale_Y)
    Y_deq = Y_q * scale_Y.unsqueeze(-1)
    
    print(f"Shape of Y_deq: {Y_deq.shape}")
    print(f"Sample values of Y_deq: {Y_deq[0, :5, :5]}")
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max absolute difference: {max_diff.item()}")    
    mse = torch.mean((Y - Y_deq) ** 2)
    print(f"Mean Squared Error: {mse.item()}")