"""
In this script, we compare the latency and numerical error between:
- gemm_cutlass.func_w8a8o8_matmul_fusion (Fusion matmul and quantization)
- gemm_cutlass.func_w8a8o8_matmul (Separate matmul and quantization)
"""

import os 
import time
import math
import torch 
from utils_quant import *
from utils_power import *
import gemm_cutlass

def init_matrix_with_outlier(M, N, outlier_values = 100,\
                        dtype=torch.bfloat16, device='cuda'):
    X = torch.randn((M, N), dtype=dtype).to(device)
    num_outliers = int(0.01 * M * N)  # 1% of the elements are outliers
    for _ in range(num_outliers):
        i = torch.randint(0, M, (1,)).item()
        j = torch.randint(0, N, (1,)).item()
        X[i, j] = outlier_values
    return X.to(device).to(dtype)

if __name__ == "__main__":
    
    dtype = torch.bfloat16
    device = 'cuda'
    M = 1024
    N = 1024
    K = 1024
    list_M = list_N = list_K = [8192]
    
    for (M, N, K) in zip(list_M, list_N, list_K):
        print("\n\n" + "=" * 56)
        
        X = init_matrix_with_outlier(M, K)
        # W = torch.empty((N, K), dtype=dtype).cuda()
        # torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        W = init_matrix_with_outlier(N, K)
        print(f"Input shape: {X.shape}, Weight shape: {W.shape}")
            
        y_torch = X @ W.transpose(-1, -2)
        
        _, scale_Y = quantize_row_int8_symmetric_nd(y_torch) # assume scale_Y via calibration
        
        # Quantization (with new values)
        X = init_matrix_with_outlier(M, K)
        # W = torch.empty((N, K), dtype=dtype).cuda()
        # torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        W = init_matrix_with_outlier(N, K)
        
        y_torch = X @ W.transpose(-1, -2)
        
        X_i8, scale_X = quantize_row_int8_symmetric_nd(X)
        W_i8, scale_W = quantize_row_int8_symmetric_nd(W)
        
        # 1. Fusion matmul and quantization
        row_scale = scale_X / scale_Y
        Y_i8_fusion = gemm_cutlass.func_w8a8o8_matmul_fusion(X_i8, W_i8, row_scale, scale_W)
        Y_deq_fusion = Y_i8_fusion.float() * scale_Y.unsqueeze(-1)
        max_diff = torch.max(torch.abs(Y_deq_fusion - y_torch))
        print(f"Max difference between fusion and torch: {max_diff.item()}")
        rmse = torch.sqrt(torch.mean((Y_deq_fusion - y_torch) ** 2)).item()
        print(f"RMSE between fusion and torch: {rmse}")
        
        measure_power(gemm_cutlass.func_w8a8o8_matmul_fusion, X_i8, W_i8, row_scale, scale_W)
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 2. Separate matmul and quantization
        print("\n" + "─" * 56)
        Y_i8, scale_Y = gemm_cutlass.func_w8a8o8_matmul(X_i8, W_i8, scale_X, scale_W)
        Y_deq_separate = Y_i8.float() * scale_Y.unsqueeze(-1)
        max_diff = torch.max(torch.abs(Y_deq_separate - y_torch))
        print(f"Max difference between separate and torch: {max_diff.item()}")
        rmse = torch.sqrt(torch.mean((Y_deq_separate - y_torch) ** 2)).item()
        print(f"RMSE between separate and torch: {rmse}")

        measure_power(gemm_cutlass.func_w8a8o8_matmul, X_i8, W_i8, scale_X, scale_W)