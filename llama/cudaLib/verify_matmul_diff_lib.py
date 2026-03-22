"""
In this script, we verify the speed of int8 matmul and scaling.

"""

import os 
import torch
import gemm_cutlass
from utils_quant import *
from bitsandbytes.functional import int8_linear_matmul, int8_mm_dequant
from torchao.quantization.utils import quant_int8_per_token_matmul

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

def torch_ao_int8_and_dequantize(x_q, w_q_t, x_scales, w_scales, output_dtype=torch.bfloat16):
    """
    x_q: (N, D) int8
    w_q: (M, D) int8
    x_scales: (N,) float32
    w_scales: (M,) float32
    Returns:
      y: (N, M) float32
    """
    return quant_int8_per_token_matmul(x_q, x_scales, w_q_t, w_scales, output_dtype=output_dtype)

if __name__ == "__main__":
    
    M = 2048
    K = 1024 * 8
    N = 1024 * 8
    
    dtype = torch.bfloat16
    
    X_bf = torch.randn((M, K), dtype=dtype, device='cuda')
    W_bf = torch.randn((N, K), dtype=dtype, device='cuda')
    
    W_bf_T = W_bf.t().contiguous()
    true_Y = torch.matmul(X_bf, W_bf_T)
    _, Y_scale = quantize_row_int8_symmetric_nd(true_Y)
    
    # Quantization computation
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X_bf)
    W_int8, W_scale = quantize_row_int8_symmetric_nd(W_bf)
    
    # Measure time 
    torch_time = measure_time(torch.matmul, X_bf, W_bf_T)
    print(f"PyTorch matmul time: {torch_time:.2f} ms")
    
    custom_time = measure_time(gemm_cutlass.func_w8a8_matmul, 
                               X_int8, W_int8, X_scale, W_scale)
    print(f"Int8 matmul (fusion scale) time: {custom_time:.2f} ms")
    
    int8_matmul_time = measure_time(gemm_cutlass.func_int8_matmul, 
                                    X_int8, W_int8, 1.0)
    print(f"Int8 matmul (without dequant) time: {int8_matmul_time:.2f} ms")
    
    int8_three_scale_time = measure_time(gemm_cutlass.func_int8_matmul_out_int8_three_scale,
                                        X_int8, W_int8, X_scale, W_scale, Y_scale)
    print(f"Int8 matmul (and dequantize) time: {int8_three_scale_time:.2f} ms")
    
    print("======= BitsAndBytes kernels =======")
    bnb_2_kernel = measure_time(bnb_int8_and_dequantize, 
                                X_int8, W_int8, X_scale, W_scale)
    print(f"BitsAndBytes (int8 matmul + dequantize kernel) time: {bnb_2_kernel:.2f} ms")
    
    bnb_int8_matmul_time = measure_time(int8_linear_matmul, X_int8, W_int8)
    print(f"BitsAndBytes int8 matmul time: {bnb_int8_matmul_time:.2f} ms")
    
    Y_int32 = int8_linear_matmul(X_int8, W_int8)
    bnb_dequant_time = measure_time(int8_mm_dequant, 
                                    Y_int32, X_scale, W_scale)
    print(f"BitsAndBytes dequantize time: {bnb_dequant_time:.2f} ms")
    
    print("======= TorchAO kernels =======")
    func_torch_ao_int8_and_dequantize = torch.compile(torch_ao_int8_and_dequantize)
    torch_ao_time = measure_time(func_torch_ao_int8_and_dequantize, 
                                X_int8, W_int8.t(), X_scale, W_scale)
    print(f"TorchAO (int8 matmul + dequantize kernel) time: {torch_ao_time:.2f} ms")
    
    