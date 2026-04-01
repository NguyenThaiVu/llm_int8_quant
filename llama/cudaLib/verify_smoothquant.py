"""
In this script, we verify the latency of smoothquant 
quantization.
"""

import os 
import torch
import torch.nn as nn
import gemm_cutlass
from utils_quant import *

MAX_SEQ_LEN = 1024

class Custom_Linear_PerRow(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=MAX_SEQ_LEN):
        super(Custom_Linear_PerRow, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        # Weight quantization
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer('scale_w', torch.ones(out_features))
        
        # Smooth quantization
        self.register_buffer('smooth_alpha', torch.ones(in_features)) 
        self.in_observer = PerChannelAbsMaxObserver(in_features)
        
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.register_buffer('scale_y', torch.ones(max_seq_len, dtype=torch.float32))
        
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if self.is_quantized == False:  
            self.in_observer(x) # Calibrate input statistics for SmoothQuant
             
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expect int8 input in quantization"
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)
            
            out_dim = self.weight_q.shape[0]
            
            row_scale = scale_x / scale_y_value  
            col_scale = self.scale_w.expand(out_dim)
            
            out_q = gemm_cutlass.func_w8a8o8_matmul(x, self.weight_q,\
                row_scale, col_scale)
            
            return out_q, scale_y_value
        
    def finish_calibration(self, alpha=None):
        if alpha is None:
            alpha = compute_smooth_alpha(self.in_observer, self.weight)

        self.smooth_alpha.copy_(alpha)  
        w_smooth = self.weight * alpha.unsqueeze(0) 
            
        # Quantize the smoothed weight
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(w_smooth)

        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        

if __name__ == "__main__":
    
    seq_len = 1024
    emb_dim = 4096
    dtype = torch.bfloat16
    
    X = torch.randn(seq_len, emb_dim, dtype=dtype).cuda()
    linear_layer = Custom_Linear_PerRow(emb_dim, emb_dim, max_seq_len=seq_len)\
        .cuda().to(dtype)
    
    Y, _ = linear_layer(X, 1.0)
    print(f"Output before quantization: {Y.shape}")
    
    linear_layer.finish_calibration()
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    Y_q, scale_y = linear_layer(X_int8, scale_x)
    
    print(f"Output after quantization: {Y_q.shape}, dtype: {Y_q.dtype}")
    Y_deq = Y_q.float() * scale_y.unsqueeze(1)
    Y_deq = Y_deq.to(dtype)
    
    max_diff = (Y - Y_deq).abs().max()
    print(f"Max absolute difference: {max_diff.item()}")
    mse = ((Y - Y_deq) ** 2).mean()
    print(f"Mean Squared Error: {mse.item()}")
    
    