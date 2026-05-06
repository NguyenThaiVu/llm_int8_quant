"""
In this script,  
- We verify the correctness of our custom RMSNorm with PyTorch's built-in RMSNorm. 
- We evaluate the performance of our INT8 quantized version against the bfloat16 version.
- We test both 2D and 3D inputs.
"""

import os 
import torch
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    seq_len = 2048
    embed_dim = 8192
    d_type = torch.bfloat16
    
    print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
    x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
    gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
    
    # Baseline bf16 
    y_bf16 = gemm_cutlass.func_rmsnorm(x, gamma, 1e-6)
    
    # quantization
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    
    y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
    y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
    y_dequant = y_dequant.to(d_type)
    
    # compare
    print("Max absolute difference:", (y_dequant - y_bf16).abs().max().item())
    print("Mean absolute difference:", (y_dequant - y_bf16).abs().mean().item(), "\n")        
    
    # # =========================== 3D Input ====================================
    # print(f"\nTesting RMSNorm with 3D input")
    # # seq_len = 1024 * 2
    # # n_heads = 32
    # list_seq_len = [1024, 2048, 3072]
    # list_n_heads = [16, 24, 32, 64]
    # head_dim = 128
    
    # for seq_len in list_seq_len:
    #     for n_heads in list_n_heads:
    #         print(f"\nTesting RMSNorm with seq_len={seq_len}, n_heads={n_heads}")
    #         x = torch.randn((seq_len, n_heads, head_dim), dtype=d_type, device='cuda')
    #         gamma = torch.randn((head_dim,), dtype=d_type, device='cuda')
            
    #         rmsnorm_layer = Custom_RMSNorm(head_dim).cuda()
    #         rmsnorm_layer.weight.data.copy_(gamma)
            
    #         # Baseline with PyTorch
    #         y_torch, _ = rmsnorm_layer(x, 1.0)
    #         torch_rmsnorm_time = measure_time(rmsnorm_layer, x, 1.0)
    #         print(f"PyTorch RMSNorm time (3D): {torch_rmsnorm_time:.2f} ms")
            
    #         # quantization
    #         rmsnorm_layer.finish_quantization()
    #         x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
            
    #         y_int8, scale_y = rmsnorm_layer(x_int8, scale_x)
    #         int8_rmsnorm_time = measure_time(rmsnorm_layer, x_int8, scale_x)
    #         print(f"INT8 RMSNorm time (3D): {int8_rmsnorm_time:.2f} ms\n")
            
    #         y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
    #         y_dequant = y_dequant.to(d_type)
            
    #         # compare
    #         print("Max absolute difference (3D):", (y_dequant - y_torch).abs().max().item())
    #         print("Mean absolute difference (3D):", (y_dequant - y_torch).abs().mean().item(),"\n")            
            
    