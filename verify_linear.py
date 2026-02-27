"""
In this script, we verify the correctness of
custom linear int8
"""

import os 
import torch 
import torch.nn as nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import *

class Custom_Linear_PerRow(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=1024):
        super(Custom_Linear_PerRow, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        
        self.register_buffer('scale_w', torch.ones(out_features))
        self.register_buffer('scale_y', torch.ones(max_seq_len))

        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode 
            
            print(f"[DEBUG] Forward pass in calibration mode")
            print(f"[DEBUG] Input shape: {x.shape}")
            print(f"[DEBUG] Weight shape: {self.weight.shape}")
            
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)  
            
            print(f"[DEBUG] Shape scale_x: {scale_x.shape}")
            print(f"[DEBUG] Shape scale_w: {self.scale_w.shape}")
            print(f"[DEBUG] Shape scale_y_value: {scale_y_value.shape}")
            
            if x.dim() == 2:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    x, self.weight_q, 
                    scale_x, self.scale_w, scale_y_value
                )
            else:
                raise ValueError("Input must be 2D tensor")
            return out_q, scale_y_value
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(self.weight)
        
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  


if __name__ == "__main__":
    
    seq_len = 1024
    in_dim = 2560
    out_dim = 9728
    
    device = 'cuda'
    d_type = torch.bfloat16
    
    # ==========================
    # 1. 2D input 
    X = torch.randn((seq_len, in_dim), dtype=d_type, device=device)
    linear = Custom_Linear_PerRow(in_dim, out_dim).to(device).to(d_type)
    
    # Step 1: Calibration
    with torch.no_grad():
        out_float, _ = linear(X, scale_x=1.0)
        linear.finish_calibration()
        
    # Step 2: Quantized inference
    with torch.no_grad():
        X_q, scale_x = quantize_row_int8_symmetric_nd(X)
        out_q, scale_y = linear(X_q, scale_x=scale_x)
    
    # Dequantize output for comparison
    out_q_dequant = out_q.float() * scale_y.unsqueeze(-1)
    
    max_diff = (out_float - out_q_dequant).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    mse = ((out_float - out_q_dequant) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    print(f"Sample output (float): {out_float[:5, :5]}")
    print(f"Sample output (dequantized): {out_q_dequant[:5, :5]} \n")
    