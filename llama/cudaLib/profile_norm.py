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
            y = F.rms_norm(x, self.norm_shape, self.weight, eps=self.eps)
            return y, 1.0
        else:
            x_int8, scale_x = gemm_cutlass.func_rmsnorm_int8(x, scale_x, self.weight, self.eps)
            return x_int8, scale_x
    
    def finish_quantization(self):
        self.is_quantized = True

if __name__ == "__main__":
    
    n_heads = 24
    seq_len = 2048
    d_head = 128
    d_type = torch.bfloat16
    
    x = torch.randn((n_heads, seq_len, d_head), dtype=d_type, device='cuda')
    gamma = torch.randn((d_head,), dtype=d_type, device='cuda')
    
    rmsnorm_layer = Custom_RMSNorm(d_head).cuda()
    rmsnorm_layer.weight.data.copy_(gamma)
    y_torch, _ = rmsnorm_layer(x, 1.0)
    
    # quantization
    rmsnorm_layer.finish_quantization()
    
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    print(f"[DEBUG] Shape of quantized input: {x_int8.shape}, scale_x shape: {scale_x.shape}")
    
    y_int8, scale_y = rmsnorm_layer(x_int8, scale_x)
    print(f"[DEBUG] Shape of quantized output: {y_int8.shape}, scale_y shape: {scale_y.shape}")
    
    y_dequant = y_int8.to(torch.float32) * scale_y.unsqueeze(-1)
    y_dequant = y_dequant.to(d_type)
    
    # compare
    print("Max absolute difference:", (y_dequant - y_torch).abs().max().item())
    print("Mean absolute difference:", (y_dequant - y_torch).abs().mean().item())
    
    print(f"Sample output (torch): {y_torch[:5, :5]}")
    print(f"Sample output (dequantized): {y_dequant[:5, :5]}")
    