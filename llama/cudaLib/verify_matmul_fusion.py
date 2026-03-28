"""
In this script, we verify the speed of int8 matmul and scaling.

"""

import os 
import time
import torch
import gemm_cutlass
from utils_quant import *
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant

def bnb_int8_and_dequantize(x_q, w_q, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N) float32
    w_scales: (M,) float32
    Returns:
      y: (B, N, M) float32
    """
    y_int = int8_linear_matmul(x_q, w_q)  # (B, N, M) int32
    y = int8_mm_dequant(y_int, x_scales, w_scales)
    return y.to(output_dtype)


if __name__ == "__main__":
    
    M = 2048
    K = 4096
    N = 8192
    
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
    print(f"Shape of X_scale: {X_scale.shape}")
    print(f"Shape of W_scale: {W_scale.shape}")
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
    
    