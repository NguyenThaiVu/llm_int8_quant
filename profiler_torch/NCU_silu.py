import os 
import torch 
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    dtype = torch.bfloat16
    device = 'cuda'
    seq_len = 2048
    # hidden_dim = 4096
    list_hidden_dim = [4096, 8192, 16384]
    for hidden_dim in list_hidden_dim:
        X = torch.randn(seq_len, hidden_dim, device=device, dtype=dtype)
        print(f"\nInput shape: {X.shape}, dtype: {X.dtype}")

        # 1. Torch SiLU
        torch_time = measure_time(torch.nn.functional.silu, X)
        print(f"Torch SiLU time: {torch_time:.3f} ms")
        
        # 2. BF16 SiLU
        bf16_time = measure_time(gemm_cutlass.func_silu_bf16, X)
        print(f"BF16 SiLU time: {bf16_time:.3f} ms")
        
        # 3. INT8 SiLU
        X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
        int8_time = measure_time(gemm_cutlass.func_silu_int8, X_i8, scale_x)
        print(f"INT8 SiLU time: {int8_time:.3f} ms")