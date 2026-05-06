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

class Custom_RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.norm_shape = (d_model,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.is_quantized = False
    
    def forward(self, x, scale_x):
        if self.is_quantized == False:
            print(f"Input to PyTorch RMSNorm: {x.shape}, dtype={x.dtype}")
            y = F.rms_norm(x, self.norm_shape, self.weight, eps=self.eps)
            print(f"Output from PyTorch RMSNorm: {y.shape}, dtype={y.dtype}")
            return y, 1.0
        else:
            x_int8, scale_x = gemm_cutlass.func_rmsnorm_int8(x, scale_x, self.weight, self.eps)
            return x_int8, scale_x
    
    def finish_quantization(self):
        self.is_quantized = True

if __name__ == "__main__":
    
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    # seq_len = 1024 * 2
    # embed_dim = 4096
    list_seq_len = [1024, 2048, 3072]
    list_embed_dim = [2048, 4096, 5120, 8192]
    d_type = torch.bfloat16
    
    for seq_len in list_seq_len:
        for embed_dim in list_embed_dim:
            print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
            x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
            gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
            
            rmsnorm_layer = Custom_RMSNorm(embed_dim).cuda()
            rmsnorm_layer.weight.data.copy_(gamma)
            
            # Baseline with PyTorch
            y_torch, _ = rmsnorm_layer(x, 1.0)
            torch_rmsnorm_time = measure_time(rmsnorm_layer, x, 1.0)
            print(f"PyTorch RMSNorm time: {torch_rmsnorm_time:.2f} ms")
            
            # quantization
            rmsnorm_layer.finish_quantization()
            x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
            
            y_int8, scale_y = rmsnorm_layer(x_int8, scale_x)
            int8_rmsnorm_time = measure_time(rmsnorm_layer, x_int8, scale_x)
            print(f"INT8 RMSNorm time: {int8_rmsnorm_time:.2f} ms\n")

            y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
            y_dequant = y_dequant.to(d_type)
            
            # compare
            print("Max absolute difference:", (y_dequant - y_torch).abs().max().item())
            print("Mean absolute difference:", (y_dequant - y_torch).abs().mean().item(), "\n")        
            # print(f"Sample output (torch): {y_torch[:5, :5]}")
            # print(f"Sample output (dequantized): {y_dequant[:5, :5]}\n")
    
    
    # =========================== 3D Input ====================================
    print(f"\nTesting RMSNorm with 3D input")
    # seq_len = 1024 * 2
    # n_heads = 32
    list_seq_len = [1024, 2048, 3072]
    list_n_heads = [16, 24, 32, 64]
    head_dim = 128
    
    for seq_len in list_seq_len:
        for n_heads in list_n_heads:
            print(f"\nTesting RMSNorm with seq_len={seq_len}, n_heads={n_heads}")
            x = torch.randn((seq_len, n_heads, head_dim), dtype=d_type, device='cuda')
            gamma = torch.randn((head_dim,), dtype=d_type, device='cuda')
            
            rmsnorm_layer = Custom_RMSNorm(head_dim).cuda()
            rmsnorm_layer.weight.data.copy_(gamma)
            
            # Baseline with PyTorch
            y_torch, _ = rmsnorm_layer(x, 1.0)
            torch_rmsnorm_time = measure_time(rmsnorm_layer, x, 1.0)
            print(f"PyTorch RMSNorm time (3D): {torch_rmsnorm_time:.2f} ms")
            
            # quantization
            rmsnorm_layer.finish_quantization()
            x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
            
            y_int8, scale_y = rmsnorm_layer(x_int8, scale_x)
            int8_rmsnorm_time = measure_time(rmsnorm_layer, x_int8, scale_x)
            print(f"INT8 RMSNorm time (3D): {int8_rmsnorm_time:.2f} ms\n")
            
            y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
            y_dequant = y_dequant.to(d_type)
            
            # compare
            print("Max absolute difference (3D):", (y_dequant - y_torch).abs().max().item())
            print("Mean absolute difference (3D):", (y_dequant - y_torch).abs().mean().item(),"\n")            
            # print(f"Sample output (torch, 3D): {y_torch[:5, 0, :5]}")
            # print(f"Sample output (dequantized, 3D): {y_dequant[:5, 0, :5]}")
            
    