"""
In this script, we compare the latency of LayerNorm
- Implicit shared memory reduction.
- Explicit shared memory reduction: using shared memory to store x_fp_s and performing reduction in shared memory.
"""

import os
import time
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    # seq_len = 2048
    # embed_dim = 4096
    d_type = torch.bfloat16
    list_seq_len = [1024, 2048]
    list_embed_dim = [4096, 5120, 8192]
    
    for seq_len in list_seq_len:
        for embed_dim in list_embed_dim:
            print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
            x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
            gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
            beta = torch.randn((embed_dim,), dtype=d_type, device='cuda')
            
            # quantization
            x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
            
            y_int8, scale_y = gemm_cutlass.func_layernorm_int8(x_int8, scale_x, gamma, beta, 1e-6)
            y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
            y_dequant = y_dequant.to(d_type)
            
            int8_layernorm_latency = measure_time(gemm_cutlass.func_layernorm_int8,\
                                            x_int8, scale_x, gamma, beta, 1e-6)
            print(f"Latency of LayerNorm with implicit global memory reduction: {int8_layernorm_latency:.3f} ms")
            
            y_int8_shared, scale_y_shared = gemm_cutlass.func_layernorm_int8_shared(x_int8, scale_x, gamma, beta, 1e-6)
            y_dequant_shared = y_int8_shared.to(torch.float32) * scale_y_shared.unsqueeze(-1)
            y_dequant_shared = y_dequant_shared.to(d_type)    
            
            int8_layernorm_shared_latency = measure_time(gemm_cutlass.func_layernorm_int8_shared,\
                                                x_int8, scale_x, gamma, beta, 1e-6)
            print(f"Latency of LayerNorm with explicit shared memory reduction: {int8_layernorm_shared_latency:.3f} ms")
            print(f"Speed up: {int8_layernorm_shared_latency/int8_layernorm_latency:.4f}x")
            
            max_diff = torch.max(torch.abs(y_dequant - y_dequant_shared)).item()
            print(f"Max absolute difference: {max_diff:.12f}")
            mse = torch.mean((y_dequant - y_dequant_shared) ** 2).item()
            print(f"Mean Squared Error: {mse:.12f}")
            
            # print(f"Sample output from global memory reduction: {y_dequant[:5, :5]}")
            # print(f"Sample output from shared memory reduction: {y_dequant_shared[:5, :5]}")

    # # =========================== 3D Input ====================================
    # print(f"\n\nTesting LayerNorm with 3D input\n")
    # torch.cuda.empty_cache()
    # time.sleep(1)
    
    # # n_heads = 24
    # # head_size = 128
    # seq_len = 2048
    # d_type = torch.bfloat16
    # list_head_size = [128, 256, 512]
    # list_n_heads = [24, 32, 64]
    
    # for head_size in list_head_size:
    #     for n_heads in list_n_heads:
    #         print(f"\nTesting LayerNorm with n_heads={n_heads}")
    #         X = torch.randn((n_heads, seq_len, head_size), dtype=d_type, device='cuda')
    #         gamma = torch.randn((head_size,), dtype=d_type, device='cuda')
    #         beta = torch.randn((head_size,), dtype=d_type, device='cuda')
            
    #         # quantization
    #         X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
            
    #         Y_int8, scale_Y = gemm_cutlass.func_layernorm_int8(X_int8, scale_X, gamma, beta, 1e-6)
    #         Y_dequant = Y_int8.to(torch.float32) * scale_Y.unsqueeze(-1)
    #         Y_dequant = Y_dequant.to(d_type)
            
    #         int8_layernorm_latency = measure_time(gemm_cutlass.func_layernorm_int8,\
    #                                         X_int8, scale_X, gamma, beta, 1e-6)
    #         print(f"Latency of LayerNorm with global memory reduction: {int8_layernorm_latency:.3f} ms")
            
    #         Y_int8_shared, scale_Y_shared = gemm_cutlass.func_layernorm_int8_shared(X_int8, scale_X, gamma, beta, 1e-6)
    #         Y_dequant_shared = Y_int8_shared.to(torch.float32) * scale_Y_shared.unsqueeze(-1)
    #         Y_dequant_shared = Y_dequant_shared.to(d_type)  
            
    #         int8_layernorm_shared_latency = measure_time(gemm_cutlass.func_layernorm_int8_shared,\
    #                                             X_int8, scale_X, gamma, beta, 1e-6)
    #         print(f"Latency of LayerNorm with shared memory reduction: {int8_layernorm_shared_latency:.3f} ms")
    #         print(f"Speed up: {int8_layernorm_shared_latency/int8_layernorm_latency:.2f}x")

    #         max_diff = torch.max(torch.abs(Y_dequant - Y_dequant_shared)).item()
    #         print(f"Max absolute difference: {max_diff:.6f}")
    #         mse = torch.mean((Y_dequant - Y_dequant_shared) ** 2).item()
    #         print(f"Mean Squared Error: {mse:.6f}")