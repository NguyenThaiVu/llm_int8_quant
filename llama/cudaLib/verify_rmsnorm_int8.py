"""
In this script, we verify the correctness of 
custom int8 RMSNorm kernel
"""

import os 
import torch
from torch.nn import functional as F
import gemm_cutlass
from utils_quant import *
from torch import nn


class RMSNorm_Fuse_Quant(nn.Module):
    """
    This module fuse RMSNorm and quantization into a single kernel. 
    - Input:
        x: (seq_len, emb_dim) in bf16
    - Output:
        y: (seq_len, emb_dim) in bf16
    OR 
        Y_int8: (seq_len, emb_dim) in int8
        scale_Y: (seq_len,) in float32
    """
    def __init__(self, emb_dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=dtype))
        self.norm_shape = (emb_dim,)
        
        self.is_quantized = False

    def forward(self, x):
        if self.is_quantized == False:
            norm_x = F.rms_norm(x, normalized_shape=self.norm_shape, weight=self.weight, eps=self.eps)
            return norm_x
        else:
            Y_int8, scale_Y = gemm_cutlass.func_rmsnorm_quant(x, 
                                                            self.weight, self.eps)
            return Y_int8, scale_Y
    
    def finish_calibration(self):
        self.is_quantized = True
        
        
if __name__ == "__main__":
    
    M = 2048
    N = 8192
    dtype = torch.bfloat16
    
    X = torch.randn((M, N), dtype=dtype).cuda()
    
    # BF16 RMSNorm
    layer_norm = RMSNorm_Fuse_Quant(N, eps=1e-6, dtype=dtype).cuda()
    Y = layer_norm(X)
    torch_rms_time = measure_time(layer_norm, X)
    print(f"PyTorch RMSNorm time: {torch_rms_time:.2f} ms")
    
    # RMSNorm + quantization
    layer_norm.finish_calibration()
    Y_int8, scale_y = layer_norm(X)
    
    rms_norm_quant_time = measure_time(layer_norm, X)
    print(f"RMSNorm + quantization time: {rms_norm_quant_time:.2f} ms")
    
    Y_dequant = Y_int8.float() * scale_y.unsqueeze(-1)
    Y_dequant = Y_dequant.to(dtype)
    
    # Compute max absolute error
    max_abs_error = torch.max(torch.abs(Y - Y_dequant))
    print(f"Max absolute error: {max_abs_error.item()}")
    mse = torch.mean((Y - Y_dequant) ** 2).item()
    print(f"MSE: {mse}")
    
    