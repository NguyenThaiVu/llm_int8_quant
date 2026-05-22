"""
In this script, we compare the latency of RMSNorm layer between:
- Torch implementation (torch.nn.functional.rms_norm)
- BF16 RMSNorm - our implementation.
- INT8 RMSNorm - naive implementation.
- INT8 RMSNorm - Warp reduction implementation.
"""

import os 
import time
import torch
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    seq_len = 2048
    # embed_dim = 8192
    d_type = torch.bfloat16
    list_embed_dim = [4096, 8192, 12288, 16384]
    
    for embed_dim in list_embed_dim:
        print(f"\n\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
        
        # 1. Torch built-in RMSNorm
        with torch.no_grad():
            y_torch = F.rms_norm(x, (embed_dim,), gamma, eps=1e-6)
            time_torch = measure_time(F.rms_norm, x, (embed_dim,), gamma, 1e-6)
            print(f"Torch RMSNorm latency: {time_torch:.3f} ms")
        
        x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
        
        # 2. BF16 RMSNorm - our implementation
        y_bf16 = gemm_cutlass.func_rmsnorm(x, gamma, 1e-6)
        time_bf16 = measure_time(gemm_cutlass.func_rmsnorm, x, gamma, 1e-6)
        print(f"BF16 RMSNorm latency: {time_bf16:.3f} ms")

        # 2. INT8 RMSNorm - naive implementation
        try:
            y_int8, scale_y = gemm_cutlass.func_rmsnorm_shared_int8(x_int8, scale_x, gamma, 1e-6)
            time_int8 = measure_time(gemm_cutlass.func_rmsnorm_shared_int8, x_int8, scale_x, gamma, 1e-6)
            print(f"INT8 RMSNorm (naive) latency: {time_int8:.3f} ms")
        except Exception as e:
            print(f"INT8 RMSNorm (naive) OOM.")
        
        # 3. INT8 RMSNorm - Warp reduction implementation
        y_int8_warp, scale_y_warp = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
        time_int8_warp = measure_time(gemm_cutlass.func_rmsnorm_int8, x_int8, scale_x, gamma, 1e-6)
        print(f"INT8 RMSNorm (warp reduction) latency: {time_int8_warp:.3f} ms")
        
        print(f"Speedup over torch: {time_torch / time_int8_warp:.2f}x")
        print(f"Speedup over BF16: {time_bf16 / time_int8_warp:.2f}x")
        print(f"Speedup over INT8 naive: {time_int8 / time_int8_warp:.2f}x")

        # Clear cache and wait 1 second before next test
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)