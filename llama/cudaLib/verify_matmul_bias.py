"""
In this script, we verify the correctness of the 
func_int8_matmul_bias
"""

import os 
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    M = 2048
    K = 8192
    N = 1024 * 10
    
    dtype = torch.bfloat16
    
    X_bf = torch.randn((M, K), dtype=dtype, device='cuda')
    W_bf = torch.randn((N, K), dtype=dtype, device='cuda')
    
    W_bf_T = W_bf.t().contiguous()
    true_Y = torch.matmul(X_bf, W_bf_T)
    _, Y_scale = quantize_row_int8_symmetric_nd(true_Y)
    
    # Quantization computation
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X_bf)
    W_int8, W_scale = quantize_row_int8_symmetric_nd(W_bf)
    
    Y_custom = gemm_cutlass.func_w8a8_matmul(X_int8, W_int8, X_scale, 
                                              W_scale)
    assert Y_custom.dtype == torch.bfloat16, "Expected output to be in bfloat16"
    
    # Compare results
    max_diff = (Y_custom - true_Y).abs().max().item()
    print(f"Max diff: {max_diff}")
    mse = torch.mean((Y_custom - true_Y) ** 2).item()
    print(f"MSE: {mse}")
    
    # Measure time 
    torch_time = measure_time(torch.matmul, X_bf, W_bf_T)
    print(f"PyTorch matmul time: {torch_time:.2f} ms")
    
    custom_time = measure_time(gemm_cutlass.func_w8a8_matmul, 
                               X_int8, W_int8, X_scale, W_scale)
    print(f"Int8 matmul time: {custom_time:.2f} ms")
    
    int8_matmul_time = measure_time(gemm_cutlass.func_int8_matmul, 
                                    X_int8, W_int8, 1.0)
    print(f"Int8 matmul (without scales) time: {int8_matmul_time:.2f} ms")
    
    bias = torch.randn((N,), dtype=torch.float32, device='cuda')
    int8_matmul_bias_time = measure_time(gemm_cutlass.func_int8_matmul_bias,
                                        X_int8, W_int8, bias)
    print(f"Int8 matmul with bias time: {int8_matmul_bias_time:.2f} ms")
    
    int8_three_scale_time = measure_time(gemm_cutlass.func_int8_matmul_out_int8_three_scale,
                                        X_int8, W_int8, X_scale, W_scale, Y_scale)
    print(f"Int8 matmul with three scales time: {int8_three_scale_time:.2f} ms")