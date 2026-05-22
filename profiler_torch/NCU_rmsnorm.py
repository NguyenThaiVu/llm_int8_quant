"""
In this script, we profile (using Nsight Compute) the kernel criteria
(SMEM per block, achieved warps occupancy, Maximum resident blocks per SM), between:
- INT8 RMSNorm - naive implementation.
- INT8 RMSNorm - Warp reduction implementation.
"""

import os
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    seq_len = 2048
    # embed_dim = 8192
    d_type = torch.bfloat16
    
    for embed_dim in [4096, 8192, 16384]:
        print(f"\n\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
        
        # quantization
        x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
        
        for _ in range(5):  # Run multiple iterations to ensure stability
            y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
        y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
        y_dequant = y_dequant.to(d_type)
        
        for _ in range(5):  # Run multiple iterations to ensure stability
            try:
                y_int8_shared, scale_y_shared = gemm_cutlass.func_rmsnorm_shared_int8(x_int8, scale_x, gamma, 1e-6)
            except RuntimeError as e:
                print(f"Error during shared memory reduction: {e}")
                continue
        y_dequant_shared = y_int8_shared.to(torch.float32) * scale_y_shared.unsqueeze(-1)
        y_dequant_shared = y_dequant_shared.to(d_type)    
        
        max_diff = torch.max(torch.abs(y_dequant - y_dequant_shared)).item()
        print(f"Max absolute difference: {max_diff:.6f}")
        mse = torch.mean((y_dequant - y_dequant_shared) ** 2).item()
        print(f"Mean Squared Error: {mse:.6f}")
        
        print(f"Sample output from global memory reduction: {y_dequant[:5, :5]}")
        print(f"Sample output from shared memory reduction: {y_dequant_shared[:5, :5]}")
