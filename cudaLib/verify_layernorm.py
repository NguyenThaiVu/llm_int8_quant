"""
In this script, we verify the correctness of our custom LayerNorm by 
comparing it with PyTorch's built-in LayerNorm. 
"""

import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import *


class Custom_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, gamma, beta, eps=1e-5):
        super(Custom_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if self.is_quantized == False:
            assert x.dtype == torch.bfloat16 or \
                     x.dtype == torch.float16 or \
                     x.dtype == torch.float32, "Input must be float in non-quantized mode"
            y = F.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
            return y
        else:
            assert x.dtype == torch.int8, "Input must be int8 in quantized mode"
            Y_int8, scale_y = gemm_cutlass.func_layernorm_int8(x, scale_x, self.gamma, self.beta, self.eps)
            return Y_int8, scale_y
            

    def finish_calibration(self):
        self.is_quantized = True
        
if __name__ == "__main__":
    # seq_len = 2048
    # embed_dim = 1024 * 5
    dtype = torch.bfloat16
    list_seq_len = [1024, 2048]
    list_embed_dim = [2048, 4096, 5120, 8192, 10240]
    
    for seq_len in list_seq_len:
        for embed_dim in list_embed_dim:
            print(f"\nTesting LayerNorm with seq_len={seq_len}, embed_dim={embed_dim}")
    
            X = torch.randn((seq_len, embed_dim), dtype=dtype, device='cuda') * 2.0
            gamma = torch.randn((embed_dim,), dtype=dtype, device='cuda')
            beta = torch.randn((embed_dim,), dtype=dtype, device='cuda')
            
            # Torch LayerNorm
            custom_layernorm = Custom_LayerNorm(embed_dim, gamma, beta).to('cuda').to(dtype)
            Y_bf16 = custom_layernorm(X, None)
            
            torch_layernorm_latency = measure_time(custom_layernorm, X, None)
            print(f"Latency of PyTorch LayerNorm: {torch_layernorm_latency:.3f} ms")
            time.sleep(0.5) 

            # Quantization
            custom_layernorm.finish_calibration()
            X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
            
            Y_int8, scale_y = custom_layernorm(X_int8, scale_x)
            Y_dequant = Y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
            Y_dequant = Y_dequant.to(dtype)
            
            int8_layernorm_latency = measure_time(custom_layernorm, X_int8, scale_x)
            print(f"Latency of Custom LayerNorm: {int8_layernorm_latency:.3f} ms")
            print(f"Speedup: {torch_layernorm_latency / int8_layernorm_latency:.2f}x")
            
            max_diff = torch.max(torch.abs(Y_bf16 - Y_dequant)).item()
            print(f"Max absolute difference: {max_diff:.6f}")
            mse = torch.mean((Y_bf16 - Y_dequant) ** 2).item()
            print(f"Mean Squared Error: {mse:.6f}")
            
            # print(f"Sample output from PyTorch LayerNorm: {Y_bf16[:5, :5]}")
            # print(f"Sample output from Custom LayerNorm: {Y_dequant[:5, :5]}")
            