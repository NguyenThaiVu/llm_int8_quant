"""
In this script, we compare SiLU with different implementations, including:
- Torch's built-in SiLU
- BF16.
- Int8 with naive reduction
- Int8 with hierarchical warp-block reduction
"""

import os 
import torch 
from torch import nn
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import *


if __name__ == "__main__":
    
    seq_len = 2048
    # hidden_dim = 1024 * 16
    list_hidden_dim = [4096, 8192, 16384]
    d_type = torch.bfloat16
    device = "cuda"

    for hidden_dim in list_hidden_dim:
        print(f"\nHidden dim: {hidden_dim}")
        X1 = torch.randn((seq_len, hidden_dim), dtype=d_type, device=device)
        X2 = torch.randn((seq_len, hidden_dim), dtype=d_type, device=device)
        smooth_alpha = torch.ones(hidden_dim, dtype=torch.float32, device='cuda')
        # print(f"Input shape: {X1.shape}, dtype: {X1.dtype}")
        y_torch = F.silu(X1) * X2
        
        # 1. Torch's built-in SiLU
        with torch.no_grad():
            time_torch = measure_time(lambda: F.silu(X1) * X2 / smooth_alpha)
            print(f"Torch SiLU latency: {time_torch:.3f} ms")
            
        # 2. BF16 SiLU
        time_bf16 = measure_time(gemm_cutlass.func_silu_mul_bf16, X1, X2, smooth_alpha)
        print(f"BF16 SiLU latency: {time_bf16:.3f} ms")
        
        X1_i8, scale_x1 = quantize_row_int8_symmetric_nd(X1)
        X2_i8, scale_x2 = quantize_row_int8_symmetric_nd(X2)

        # # 3. Int8 SiLU with naive reduction
        try:
            time_i8_naive = measure_time(gemm_cutlass.func_silu_mul_int8, X1_i8, scale_x1,\
                                                    X2_i8, scale_x2, smooth_alpha, False, repeat=10)
            print(f"Int8 SiLU (naive reduction) latency: {time_i8_naive:.3f} ms")
        except RuntimeError as e:
            print(f"Int8 SiLU (naive reduction) failed.")
        
        # 4. Int8 SiLU with hierarchical warp-block reduction
        Y_i8, scale_y = gemm_cutlass.func_silu_mul_int8(X1_i8, scale_x1, X2_i8, scale_x2, smooth_alpha, True)
        time_i8_hierarchical = measure_time(gemm_cutlass.func_silu_mul_int8, X1_i8, scale_x1,\
                                                X2_i8, scale_x2, smooth_alpha, True, repeat=10)
        print(f"Int8 SiLU (hierarchical reduction) latency: {time_i8_hierarchical:.3f} ms")
        speed_up_with_bf = time_bf16/time_i8_hierarchical
        speed_up_with_naive = time_i8_naive/time_i8_hierarchical
        print(f"Speed up with BF16: {speed_up_with_bf:.2f}x")
        print(f"Speed up with naive reduction: {speed_up_with_naive:.2f}x")
        
    