"""
In this script, we verify the speed of int8 matmul and scaling.

"""

import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    M = 2048
    K = 1024 * 8
    N = 1024 * 8
    
    dtype = torch.bfloat16
    
    X_bf = torch.randn((M, K), dtype=dtype, device='cuda')
    W_bf = torch.randn((N, K), dtype=dtype, device='cuda')
    
    W_bf_T = W_bf.t().contiguous()
    true_Y = torch.matmul(X_bf, W_bf_T)
    
    _, Y_scale = quantize_row_int8_symmetric_nd(true_Y) # assume quantize via calibration
    
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X_bf)
    W_int8, W_scale = quantize_row_int8_symmetric_nd(W_bf)
    
    # Verify correctness
    # ==========================================================
    Y_custom = gemm_cutlass.func_w8a8_matmul(X_int8, W_int8, X_scale, W_scale)    
    max_diff = torch.max(torch.abs(Y_custom - true_Y))
    print(f"Max diff: {max_diff:.4f}")
    mse = torch.mean((Y_custom - true_Y) ** 2).item()
    print(f"MSE: {mse:.4f}")
    
    # Measure time 
    # ==========================================================
    torch_time = measure_time(torch.matmul, X_bf, W_bf_T)
    print(f"PyTorch fp16 matmul time: {torch_time:.2f} ms")
    time.sleep(1) 
    
    custom_time = measure_time(gemm_cutlass.func_w8a8_matmul, 
                               X_int8, W_int8, X_scale, W_scale)
    print(f"Int8 matmul (fusion scale) time: {custom_time:.2f} ms")
    time.sleep(1)
    
    int8_matmul_time = measure_time(gemm_cutlass.func_int8_matmul, 
                                    X_int8, W_int8, 1.0)
    print(f"Int8 matmul (without dequant) time: {int8_matmul_time:.2f} ms")
    time.sleep(1)
    
    int8_three_scale_time = measure_time(gemm_cutlass.func_int8_matmul_out_int8_three_scale,
                                        X_int8, W_int8, X_scale, W_scale, Y_scale)
    print(f"Int8 matmul (and dequantize kernel) time: {int8_three_scale_time:.2f} ms")
    
    