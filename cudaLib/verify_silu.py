import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch 
from torch import nn
from utils_quant import *
import custom_ops

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, smooth_scale=None):
        y = torch.nn.functional.silu(x1) * x2
        if smooth_scale is not None:
            y = y / smooth_scale
        return y

if __name__ == "__main__":
    
    d_type = torch.bfloat16
    d = 1024 * 8
    
    X1 = torch.randn((d, d), dtype=d_type, device="cuda")
    X2 = torch.randn((d, d), dtype=d_type, device="cuda")
    smooth_scale = torch.ones((d), dtype=torch.float32, device="cuda")
    
    # 1. Torch SiLU
    silu_func = SiLU().to(device="cuda", dtype=d_type)
    silu_func = torch.compile(silu_func)
    
    Y = silu_func(X1, X2, smooth_scale) 
    
    silu_time = measure_time(silu_func, X1, X2, smooth_scale)
    print(f"PyTorch SiLU time: {silu_time:.6f} ms")
    
    # 2. Our SiLU
    x1_i8, scale_x1 = quantize_row_int8_symmetric_nd(X1)
    x2_i8, scale_x2 = quantize_row_int8_symmetric_nd(X2)
    
    y_i8, scale_y = custom_ops.silu_mul_int8(x1_i8, scale_x1, x2_i8, scale_x2, smooth_scale, True)
        
    our_silu_time_compile = measure_time(custom_ops.silu_mul_int8, x1_i8, scale_x1,\
        x2_i8, scale_x2, smooth_scale, True)
    print(f"Our SiLU with compile time: {our_silu_time_compile:.6f} ms")
    
    # 3. Compare results
    y_deq = y_i8.to(torch.float32) * scale_y.unsqueeze(-1)
    max_diff = torch.max(torch.abs(Y - y_deq))
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean(torch.abs(Y - y_deq) / (torch.abs(Y) + 1e-6))
    print(f"Mean relative error: {mse:.6f}")
    
    print(f"Y: {Y[:5, :5]}")
    print(f"Y_deq: {y_deq[:5, :5]}")
    
    