"""
In this script, we verify the correctness of the softmax implementation 
in `gemm_cutlass` library
"""

import os 
import torch
from utils_quant import *
import gemm_cutlass

def quantization_func(X_int8, scale_X, dtype_out=torch.bfloat16):
    X_dequant = X_int8.to(torch.float32) * scale_X.unsqueeze(-1)
    X_dequant = X_dequant.to(dtype_out)
    X_dequant = X_dequant.transpose(0, 1).reshape(seq_len, emb_dim) 
    return X_dequant

if __name__ == "__main__":
       
    seq_len = 2000
    emb_dim = 4096
    num_heads = 32
    head_dim = emb_dim // num_heads
    dtype = torch.bfloat16
    
    NUM_HAPPENINGS = 112
    
    X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype).cuda()
    
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    
    quant_time = measure_time(quantization_func, X_int8, scale_X, dtype)
    
    print(f"Quantization time: {quant_time:.4f} ms")
    
    total_quant_time = quant_time * NUM_HAPPENINGS
    print(f"Total quantization time for {NUM_HAPPENINGS} happenings: {total_quant_time:.4f} ms")
    
    
