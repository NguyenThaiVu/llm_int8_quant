"""
In this script, we verify the correctness and latency between
- BF16 softmax
- INT8 softmax
"""

import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    # ============================= 2D Input ==============================
    # seq_len = 1024
    # embed_dim = 8192
    dtype = torch.bfloat16
    list_seq_len = [1024, 2048]
    list_embed_dim = [4096, 5120, 8192]
    
    for seq_len in list_seq_len:
        for embed_dim in list_embed_dim:
            print(f"\nSoftmax with seq_len={seq_len}, embed_dim={embed_dim}")
            X = torch.randn((seq_len, embed_dim), dtype=dtype).cuda()
            
            # BF16 softmax
            Y_bf16 = gemm_cutlass.func_bf16_softmax(X)
            bf16_time = measure_time(gemm_cutlass.func_bf16_softmax, X)
            print(f"BF16 softmax latency: {bf16_time:.2f} ms")
            time.sleep(0.5)
            
            # INT8 softmax
            X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
            Y_int8, scale_Y = gemm_cutlass.func_softmax_int8(X_int8, scale_X)
            int8_time = measure_time(gemm_cutlass.func_softmax_int8, X_int8, scale_X)
            
            Y_deq = Y_int8.float() * scale_Y.unsqueeze(-1)
            Y_deq = Y_deq.to(dtype)
            
            print(f"INT8 softmax latency: {int8_time:.2f} ms")
            print(f"Speedup bf16: {bf16_time / int8_time:.2f}x") 
            
            # Verify correctness
            max_diff = torch.max(torch.abs(Y_bf16 - Y_deq))
            print(f"Max diff: {max_diff.item()}")
            mse = torch.mean((Y_bf16 - Y_deq) ** 2).item()
            print(f"MSE: {mse}")
            
    # ============================= 3D Input ==============================
    print("\n\n============================= 3D Input ==============================")
    
    torch.cuda.empty_cache()
    
    # seq_len = 1024
    # embed_dim = 8192
    list_n_heads = [24, 32, 64]
    list_seq_len = [1024, 2048]
    
    for n_heads in list_n_heads:
        for seq_len in list_seq_len:
            print(f"\nSoftmax with n_heads={n_heads}, seq_len={seq_len}")
            X = torch.randn((n_heads, seq_len, seq_len), dtype=dtype).cuda()
            
            # BF16 softmax
            Y_bf16 = gemm_cutlass.func_bf16_softmax(X)
            bf16_time = measure_time(gemm_cutlass.func_bf16_softmax, X)
            print(f"BF16 softmax latency: {bf16_time:.2f} ms")
            
            # INT8 softmax
            X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
            Y_int8, scale_Y = gemm_cutlass.func_softmax_int8(X_int8, scale_X)
            
            Y_deq = Y_int8.float() * scale_Y.unsqueeze(-1)
            Y_deq = Y_deq.to(dtype)
            
            int8_time = measure_time(gemm_cutlass.func_softmax_int8, X_int8, scale_X)
            print(f"INT8 softmax latency: {int8_time:.2f} ms")
            print(f"Speedup: {bf16_time / int8_time:.2f}x")
            
            # Verify correctness
            max_diff = torch.max(torch.abs(Y_bf16 - Y_deq))
            print(f"Max diff: {max_diff.item()}")
            mse = torch.mean((Y_bf16 - Y_deq) ** 2).item()
            print(f"MSE: {mse}")

            
