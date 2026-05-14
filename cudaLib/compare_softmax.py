"""
In this script, we compare the latency of Softmax
- Implicit shared memory reduction.
- Explicit shared memory reduction: using shared memory to store x_fp_s and performing reduction in shared memory.
"""

import os 
import torch
import time
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    # n_heads = 40
    # seq_len = 2048
    d_type = torch.bfloat16
    list_seq_len = [1024, 2048]
    list_n_heads = [24, 32, 64]
    
    for seq_len in list_seq_len:
        for n_heads in list_n_heads:
            print(f"\nTesting Softmax with n_heads={n_heads}, seq_len={seq_len}")
            x = torch.randn((n_heads, seq_len, seq_len), dtype=d_type, device='cuda')
            
            # quantization
            x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
            
            y_int8, scale_y = gemm_cutlass.func_softmax_int8(x_int8, scale_x)
            y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
            y_dequant = y_dequant.to(d_type)    
            
            int8_softmax_latency = measure_time(gemm_cutlass.func_softmax_int8, x_int8, scale_x)
            print(f"Latency of Softmax with implicit global memory reduction: {int8_softmax_latency:.3f} ms")
            
            y_int8_shared, scale_y_shared = gemm_cutlass.func_softmax_int8_shared(x_int8, scale_x)
            y_dequant_shared = y_int8_shared.to(torch.float32) * scale_y_shared.unsqueeze(-1)
            y_dequant_shared = y_dequant_shared.to(d_type)  
            
            int8_softmax_shared_latency = measure_time(gemm_cutlass.func_softmax_int8_shared, x_int8, scale_x)
            print(f"Latency of Softmax with explicit shared memory reduction: {int8_softmax_shared_latency:.3f} ms")
            print(f"Speed up: {int8_softmax_shared_latency/int8_softmax_latency:.4f}x")
            
            max_diff = torch.max(torch.abs(y_dequant - y_dequant_shared)).item()
            print(f"Max absolute difference: {max_diff:.12f}")
            mse = torch.mean((y_dequant - y_dequant_shared) ** 2).item()
            print(f"Mean Squared Error: {mse:.12f}")
