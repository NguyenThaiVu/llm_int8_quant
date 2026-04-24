"""
In this script, we profile and compare the performance between bf16 and int8 softmax.
"""

import os 
import torch 
from torch import nn
from torch.nn import functional as F
import gemm_cutlass
from utils_quant import *

class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        self.is_quantized = False
        
    def forward(self, x, scale_x, mask):
        if self.is_quantized == False: 
            x = x.masked_fill(mask, -torch.inf)
            out = torch.softmax(x, dim=-1)
            return out, 1.0
        else:          
            out_q, scale_out = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x, scale_x, mask
            )
            return out_q, scale_out
    
    def finish_calibration(self):
        self.is_quantized = True  
        
        
if __name__ == "__main__":
    
    num_heads = 24
    seq_len = 2048
    d_type = torch.bfloat16
    
    # =================================================
    # =============== 2D input softmax ================
    X = torch.randn(seq_len, seq_len, dtype=d_type).cuda()
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).cuda()
    
    softmax_layer = Custom_Softmax(num_heads=1, max_seq_len=seq_len).cuda().to(d_type)
    
    bf16_softmax_time = measure_time(softmax_layer, X, 1.0, mask)
    print(f"Softmax time (bf16): {bf16_softmax_time:.2f} ms")
    
    softmax_layer.finish_calibration()
    mask = torch.tril(torch.ones((seq_len, seq_len),\
                            dtype=torch.uint8, device="cuda"))
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    
    int8_softmax_time = measure_time(softmax_layer, X_int8, scale_x, mask)
    print(f"Softmax time (int8): {int8_softmax_time:.2f} ms")
    
    # =================================================
    # =============== 3D input softmax ================
    X = torch.randn(num_heads, seq_len, seq_len, dtype=d_type).cuda()
    mask = torch.ones(num_heads, seq_len, seq_len, dtype=torch.bool).cuda()
    
    softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=seq_len).cuda().to(d_type)
    
    bf16_softmax_time = measure_time(softmax_layer, X, 1.0, mask)
    print(f"Softmax time (bf16): {bf16_softmax_time:.2f} ms")
    
    softmax_layer.finish_calibration()
    mask = torch.tril(torch.ones((num_heads, seq_len, seq_len),\
                            dtype=torch.uint8, device="cuda"))
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    
    int8_softmax_time = measure_time(softmax_layer, X_int8, scale_x, mask)
    print(f"Softmax time (int8): {int8_softmax_time:.2f} ms")
