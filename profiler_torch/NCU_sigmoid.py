import os 
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    seq_len = 2048
    # hidden_dim = 1024 * 12
    list_hidden_dim = [4096, 8192, 16384]
    d_type = torch.bfloat16
    
    for hidden_dim in list_hidden_dim:
        X = torch.randn(seq_len, hidden_dim, dtype=d_type).cuda()
        print(f"\nInput shape: {X.shape}, dtype: {X.dtype}")
        
        # torch sigmoid
        Y_torch = torch.sigmoid(X)
        time_torch = measure_time(torch.sigmoid, X, repeat=10)
        print(f"PyTorch Sigmoid time: {time_torch:.2f} ms")
        
        # BF16 Sigmoid
        Y_bf16 = gemm_cutlass.func_sigmoid_bf16(X)
        time_bf16 = measure_time(gemm_cutlass.func_sigmoid_bf16, X, repeat=10)
        print(f"BF16 Sigmoid time: {time_bf16:.2f} ms")
        
        # Quantized Sigmoid
        X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
        Y_int8, scale_out = gemm_cutlass.func_sigmoid_int8(X_int8, scale_x)
        time_int8 = measure_time(gemm_cutlass.func_sigmoid_int8, X_int8, scale_x, repeat=10)
        print(f"Quantized Sigmoid time: {time_int8:.2f} ms")