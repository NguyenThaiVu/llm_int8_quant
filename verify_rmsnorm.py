"""
In this script, we verify the correctness of RMSNorm implementation in both floating-point and quantized int8 versions. We define a simple RMSNorm module, generate random input data, and compare the outputs of the floating-point and quantized versions. We also measure the performance of both implementations.
"""

import os 
import torch
import torch.nn as nn
import gemm_cutlass  
from utils import *
from utils_transformer_int8 import MinMaxObserverPerLastDim

class Custom_RMSNorm(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # learnable weight of RMSNorm
        self.eps = eps
        self.dim = dim 
        
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(num_heads * max_seq_len))
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode
            assert x.dtype == torch.float32 or \
                    x.dtype == torch.bfloat16 or \
                    x.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            mean_square = x.pow(2).mean(-1, keepdim=True)  
            inv_rms = torch.rsqrt(mean_square + self.eps)   # [seq_len, 1]
            out = x * inv_rms * self.weight  # [seq_len, head_dim]
            self.out_observer(out)
            return out, 1.0
        else:
            # Quantized mode
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            y_q = gemm_cutlass.func_rmsnorm_int8(
                x, scale_x.to(torch.float32), self.weight, self.scale_out.to(torch.float32), self.eps
            )
            return y_q, self.scale_out
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True
        

if __name__ == "__main__":

    dtype = torch.bfloat16
    device = 'cuda'
    seq_length = 2048
    dim = 1024 * 5 
    
    # 1. Verify 2D input RMSNorm
    print("==== Verifying 2D RMSNorm ====")

    # 1. Calibration
    X = torch.randn(seq_length, dim, dtype=dtype, device=device)
    rms_norm = Custom_RMSNorm(num_heads=1, max_seq_len=seq_length, dim=dim).to(device).to(dtype)
    Y, _ = rms_norm(X, 1.0)
    
    # 2. Finish calibration and switch to quantized mode
    rms_norm.finish_calibration()
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    
    Y_q, scale_Y = rms_norm(X_q, scale_x)

    Y_deq = Y_q.to(torch.float32) * scale_Y.unsqueeze(-1)
    
    max_abs_diff = (Y_deq - Y).abs().max().item()
    print(f"Max diff: {max_abs_diff:.6f}")
    mse = ((Y_deq - Y) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    
    # 2. Verify 3D input RMSNorm (e.g., for multi-head attention)
    print("\n==== Verifying 3D RMSNorm ====")
    batch_size = 4
    
    X = torch.randn(batch_size, seq_length, dim, dtype=dtype, device=device)
    rms_norm_3d = Custom_RMSNorm(num_heads=1, max_seq_len=seq_length, dim=dim).to(device).to(dtype)
    Y_3d, _ = rms_norm_3d(X, 1.0)
    
    rms_norm_3d.finish_calibration()
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    
    Y_q, scale_Y = rms_norm_3d(X_q, scale_x)
    Y_deq = Y_q.to(torch.float32) * scale_Y.unsqueeze(-1)
    
    max_abs_diff = (Y_deq - Y_3d).abs().max().item()
    print(f"Max diff: {max_abs_diff:.6f}")
    mse = ((Y_deq - Y_3d) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    print(f"Sample output (float): {Y_3d[0, :5, :5]}")
    print(f"Sample output (dequantized): {Y_deq[0, :5, :5]}")
    
    
    
    