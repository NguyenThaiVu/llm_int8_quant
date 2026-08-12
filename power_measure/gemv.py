"""
Measure latency between int8 and bf16 gemmv
"""

import math
import os 
import time
import torch
import gemm_cutlass
from utils_quant import *
from utils_power import PowerSampler, measure_power

if __name__ == "__main__":

    # in_dims = 2048 # 128, 4096
    # out_dims = 4096 # 1024, 4096, 12288
    list_in_dims = [1024, 2048, 4096, 4096, 8192, 8192]
    list_out_dims = [1024, 2048, 4096, 8192, 8192, 16384]
    dtype = torch.bfloat16
    n_iter = 1_000
    
    for in_dims, out_dims in zip(list_in_dims, list_out_dims):
        print("\n\n" + "*" * 60)
        print(f"Testing GEMV with input dim {in_dims} and output dim {out_dims}")
    
        X = torch.randn((1, in_dims), dtype=dtype, device="cuda")

        # init W with kaiming uniform initialization
        W = torch.empty((out_dims, in_dims), dtype=dtype).cuda()
        torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        
        # Warm up GPU
        for _ in range(100):
            Y_torch = torch.matmul(X, W.T)
        
        # =========================================================================
        # 1. Measure torch latency
        # measure_power(torch.matmul, X, W.T, n_iterations=n_iter)
        
        time.sleep(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # =========================================================================
        # 2. Measure GEMV BF16 latency
        print(f"\nBF16 GEMV")
        measure_power(gemm_cutlass.func_bf16_gemv, X, W, 1.0, n_iterations=n_iter)
        
        time.sleep(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # =========================================================================
        # 3. Measure GEMV INT8
        print(f"\nINT8 GEMV")
        X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
        W_i8, scale_w = quantize_row_int8_symmetric_nd(W) 
        # Assume output scale via calibration 
        _, scale_y = quantize_row_int8_symmetric_nd(Y_torch)
        
        measure_power(gemm_cutlass.func_i8_gemv_out_i8, X_i8, W_i8, scale_x,\
                        scale_w, scale_y, 1.0, n_iterations=n_iter)
        
    