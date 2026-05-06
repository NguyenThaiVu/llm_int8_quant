"""
In this script, we compare the outputs of RMSNorm implemented using 
shared memory reduction and global memory reduction.
"""

import os
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    seq_len = 2048
    embed_dim = 8192
    d_type = torch.bfloat16
    
    print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
    x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
    gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
    
    # quantization
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    
    y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
    y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
    y_dequant = y_dequant.to(d_type)
    
    int8_rmsnorm_latency = measure_time(gemm_cutlass.func_rmsnorm_int8,\
                                    x_int8, scale_x, gamma, 1e-6)
    print(f"Latency of RMSNorm with global memory reduction: {int8_rmsnorm_latency:.3f} ms")
    
    y_int8_shared, scale_y_shared = gemm_cutlass.func_rmsnorm_shared_int8(x_int8, scale_x, gamma, 1e-6)
    y_dequant_shared = y_int8_shared.to(torch.float32) * scale_y_shared.unsqueeze(-1)
    y_dequant_shared = y_dequant_shared.to(d_type)    
    
    int8_rmsnorm_shared_latency = measure_time(gemm_cutlass.func_rmsnorm_shared_int8,\
                                        x_int8, scale_x, gamma, 1e-6)
    print(f"Latency of RMSNorm with shared memory reduction: {int8_rmsnorm_shared_latency:.3f} ms")
    
    max_diff = torch.max(torch.abs(y_dequant - y_dequant_shared)).item()
    print(f"Max absolute difference: {max_diff:.6f}")
    mse = torch.mean((y_dequant - y_dequant_shared) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")
    
    print(f"Sample output from global memory reduction: {y_dequant[:5, :5]}")
    print(f"Sample output from shared memory reduction: {y_dequant_shared[:5, :5]}")
