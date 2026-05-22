"""
In this scrip, we also verify the roofline (using NVIDIA Nsight Compute) between:
- BF16 RMSNorm: kernel rmsnorm_kernel
- INT8 RMSNorm (warp reduction version): kernel rmsnorm_int8_kernel
"""

import os 
import time
import torch

import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    seq_len = 2048
    # embed_dim = 8192
    d_type = torch.bfloat16
    list_embed_dim = [4096, 8192, 16384]
    
    for embed_dim in list_embed_dim:
        print(f"\n\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
        
        # 1. BF16 RMSNorm
        for _ in range(5):
            y_bf16 = gemm_cutlass.func_rmsnorm(x, gamma, 1e-6)
            
        # 2. INT8 RMSNorm
        x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
        
        for _ in range(5):
            y_int8, y_scale = gemm_cutlass.func_rmsnorm_int8(x_int8, x_scale, gamma, 1e-6)