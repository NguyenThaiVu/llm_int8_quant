import os 
import torch 
from torch import nn
from torch.nn import functional as F
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
            y = F.rms_norm(x, self.norm_shape, self.weight, eps=self.eps)
            return y
        else:
            x_int8, scale_x = gemm_cutlass.func_rmsnorm_int8(x, scale_x, self.weight, self.eps)
            return x_int8, scale_x
    
    def finish_calibration(self):
        self.is_quantized = True
        
if __name__ == "__main__":
    
    seq_len = 2048
    d_model = 8192
    d_type = torch.bfloat16
    
    X = torch.randn((seq_len, d_model), dtype=d_type).cuda()
    gamma = torch.randn((d_model,), dtype=d_type).cuda()
    
    rmsnorm = Custom_RMSNorm(d_model=d_model).cuda()
    rmsnorm.weight.data.copy_(gamma)
    
    rmsnorm_time = measure_time(rmsnorm, X, 1.0)
    print(f"RMSNorm time: {rmsnorm_time:.2f} ms")
    
    # Quantization
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    rmsnorm.finish_calibration()
    
    int8_rmsnorm_time = measure_time(rmsnorm, X_int8, scale_x)
    print(f"Int8 RMSNorm time: {int8_rmsnorm_time:.2f} ms")
    
    # =================================================
    # Test with 3D input
    num_heads = 32
    head_dim = 128
    X = torch.randn((num_heads, seq_len, head_dim), dtype=d_type).cuda()
    gamma = torch.randn((head_dim,), dtype=d_type).cuda()   
    rmsnorm = Custom_RMSNorm(d_model=head_dim).cuda()
    rmsnorm.weight.data.copy_(gamma)
    
    rmsnorm_time = measure_time(rmsnorm, X, 1.0)
    print(f"RMSNorm time (3D input): {rmsnorm_time:.2f} ms")
    
    # Quantization
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    rmsnorm.finish_calibration()
    int8_rmsnorm_time = measure_time(rmsnorm, X_int8, scale_x)
    print(f"Int8 RMSNorm time (3D input): {int8_rmsnorm_time:.2f} ms")