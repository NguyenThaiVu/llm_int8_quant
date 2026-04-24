import os 
import torch 
from torch import nn
from torch.nn import functional as F
import gemm_cutlass
from utils_quant import *

class Custom_Silu(nn.Module):
    def __init__(self, emb_dim, dtype=torch.bfloat16):
        super().__init__()
        self.emb_dim = emb_dim
        self.dtype = dtype

        self.is_quantized = False
        self.is_smooth_scale = False
        
        self.smooth_alpha = torch.ones(emb_dim, dtype=torch.float32, device='cuda')

    def forward(self, x1, scale_x1, x2, scale_x2):
        if not self.is_quantized:
            x = F.silu(x1) * x2
            return x
        else:
            assert x1.dtype == torch.int8 and x2.dtype == torch.int8,\
                "Expected int8 inputs in quantized mode"

            x_int8, x_scale = gemm_cutlass.func_silu_mul_int8(\
                x1, scale_x1,\
                x2, scale_x2, self.smooth_alpha)
            
            return x_int8, x_scale

    def finish_calibration(self):
        self.is_quantized = True
        
if __name__ == "__main__":
    seq_len = 2048
    emb_dim = 8192
    d_type = torch.bfloat16
    
    # =================================================
    # =============== 2D input silu ================
    X1 = torch.randn(seq_len, emb_dim, dtype=d_type).cuda()
    X2 = torch.randn(seq_len, emb_dim, dtype=d_type).cuda()
    
    silu_layer = Custom_Silu(emb_dim=emb_dim, dtype=d_type).cuda().to(d_type)
    
    bf16_silu_time = measure_time(silu_layer, X1, 1.0, X2, 1.0)
    print(f"Silu time (bf16): {bf16_silu_time:.2f} ms")
    
    silu_layer.finish_calibration()
    
    X1_int8, scale_x1 = quantize_row_int8_symmetric_nd(X1)
    X2_int8, scale_x2 = quantize_row_int8_symmetric_nd(X2)
    
    int8_silu_time = measure_time(silu_layer, X1_int8, scale_x1,\
                                    X2_int8, scale_x2)
    print(f"Silu time (int8): {int8_silu_time:.2f} ms")