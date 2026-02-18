"""
Docstring for verify_sigmoid
"""

import os 
import torch 
from utils import *
import gemm_cutlass

class CustomSigmoid(torch.nn.Module):
    def __init__(self):
        super(CustomSigmoid, self).__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))
    
if __name__ == "__main__":
    
    seq_len = 1024
    emb_dim = 4096
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 1. 2D input (seq_len, emb_dim)
    print("=" * 50)
    print(f"Testing 2D input: (seq_len={seq_len}, emb_dim={emb_dim})")
    
    X = torch.randn((seq_len, emb_dim), device=device, dtype=dtype) * 10
    custom_sigmoid = CustomSigmoid().to(device)
    Y = custom_sigmoid(X)
    
    print(f"Shape of Y: {Y.shape}")
    print(f"Sample values of Y: {Y[:5, :5]}")
    
    # Quantization 
    X_q, scale_x = quantize_row_wise_tensor(X)
    scale_x = scale_x.to(torch.float32)
    
    # Assume scale_y via calibration 
    _, scale_y = quantize_row_wise_tensor(Y)
    scale_y = scale_y.to(torch.float32)
    
    Y_q = gemm_cutlass.func_apply_sigmoid_int8(X_q, scale_x, scale_y)
    assert Y_q.dtype == torch.int8, "Output of int8 sigmoid should be int8"
    
    Y_deq = Y_q * scale_y.unsqueeze(-1)
    print(f"Shape of Y_deq: {Y_deq.shape}")
    print(f"Sample values of Y_deq: {Y_deq[:5, :5]}")
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse}")


    # 2. 3D input (batch_size, seq_len, emb_dim)
    print("=" * 50)
    batch_size = 4
    print(f"Testing 3D input: (batch_size={batch_size}, seq_len={seq_len}, emb_dim={emb_dim})")
    
    X = torch.randn((batch_size, seq_len, emb_dim), device=device, dtype=dtype) * 10
    custom_sigmoid = CustomSigmoid().to(device)
    Y = custom_sigmoid(X)
    
    print(f"Shape of Y: {Y.shape}")
    print(f"Sample values of Y: {Y[0, :5, :5]}")
    
    # Quantization
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    scale_x = scale_x.to(torch.float32)
    
    _, scale_y = quantize_row_int8_symmetric_nd(Y)
    scale_y = scale_y.to(torch.float32)
    
    Y_q = gemm_cutlass.func_apply_sigmoid_int8(X_q, scale_x, scale_y)
    assert Y_q.dtype == torch.int8, "Output of int8 sigmoid should be int8"
    
    Y_deq = Y_q * scale_y.unsqueeze(-1)
    print(f"Shape of Y_deq: {Y_deq.shape}")
    print(f"Sample values of Y_deq: {Y_deq[0, :5, :5]}")
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse}")
