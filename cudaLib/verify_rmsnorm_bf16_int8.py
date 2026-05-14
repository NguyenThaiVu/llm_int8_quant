"""
In this script, we verify the correctness and latency of RMSNorm between
- BF16
- INT8 
"""

import os 
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    seq_len = 1024 * 2
    embed_dim = 8192
    # list_seq_len = [1024, 2048, 3072]
    # list_embed_dim = [2048, 4096, 5120, 8192]
    d_type = torch.bfloat16
    
    print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
    x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
    gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
    
    # BF16 RMSNorm
    y_bf16 = gemm_cutlass.func_rmsnorm(x, gamma, 1e-6)
    bf16_rmsnorm_time = measure_time(gemm_cutlass.func_rmsnorm, x, gamma, 1e-6)
    print(f"BF16 RMSNorm time: {bf16_rmsnorm_time:.2f} ms")
    
    # INT8 RMSNorm
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
    int8_rmsnorm_time = measure_time(gemm_cutlass.func_rmsnorm_int8, x_int8, scale_x, gamma, 1e-6)
    print(f"INT8 RMSNorm time: {int8_rmsnorm_time:.2f} ms")
    print(f"Speedup: {bf16_rmsnorm_time / int8_rmsnorm_time:.2f}x\n")
