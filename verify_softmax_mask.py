"""
In this script, we verify the correctness of softmax.
"""
import os 
import torch
import torch.nn as nn
import gemm_cutlass

from utils import *
from utils_transformer_int8 import *

        
class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(num_heads * max_seq_len))
        
        self.is_quantized = False
        
    def forward(self, x_q, scale_x):
        if not self.is_quantized:  # Calibration mode
            assert x_q.dtype == torch.float32 or \
                    x_q.dtype == torch.bfloat16 or \
                    x_q.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x_q, dim=-1, dtype=torch.bfloat16)
            self.out_observer(out)
            return out, 1.0
        else:   # Quantized mode            
            seq_len = x_q.shape[-1]
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=x_q.device))
            
            flatten_scale_x = scale_x.view(-1)
            flatten_scale_out = self.scale_out.view(-1)
            
            out_q = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x_q, flatten_scale_x.to(torch.float32), 
                flatten_scale_out.to(torch.float32), mask
            )
            
            return out_q, self.scale_out
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True  
        
        
if __name__ == "__main__":
    seq_len = 1024 * 4
    num_heads = 32
    
    device = 'cuda'
    d_type = torch.bfloat16
    
    X = torch.randn((num_heads, seq_len, seq_len), dtype=d_type, device=device)
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    
    softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=seq_len, dim=-1).to(device)
    
    # Calibration mode
    out_calib, _ = softmax_layer(X, 1.0)
    softmax_layer.finish_calibration()
    
    # Quantized mode
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    out_q, scale_out = softmax_layer(X_q, scale_x)
    
    out_q_dequant = out_q.float() * scale_out.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(out_calib - out_q_dequant))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((out_calib - out_q_dequant) ** 2).item()
    print(f"MSE: {mse}")
    
    print(f"Sample output: {out_calib[:5, :5]}")
    print(f"Sample dequantized: {out_q_dequant[:5, :5]}\n")
    