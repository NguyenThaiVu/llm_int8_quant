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
    
    # INT8 RMSNorm
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
    y_deq = y_int8.float() * scale_y.unsqueeze(-1)
    
    # Verify correctness
    max_diff = torch.max(torch.abs(y_bf16.float() - y_deq))
    print(f"Max absolute diff: {max_diff.item():.6f}")
    mse = torch.mean((y_bf16.float() - y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")
