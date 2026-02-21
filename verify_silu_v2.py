"""
In this module, 
we verify the correctness of the SiLU (Sigmoid Linear Unit) activation.
"""

import os 
import torch
from torch import nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import MinMaxObserverPerLastDim

class Custom_Sigmoid(torch.nn.Module):
    def __init__(self, max_seq_len=1024):
        super(Custom_Sigmoid, self).__init__()
        self.max_seq_len = max_seq_len
        self.observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.is_quantized = False
        self.register_buffer('scale_y', torch.ones(max_seq_len))  

    def forward(self, x, scale_x):
        if not self.is_quantized:
            out = torch.sigmoid(x)
            self.observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_x = scale_x[:seq_len].to(torch.float32)
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)
            
            Y_q = gemm_cutlass.func_apply_sigmoid_int8(x, scale_x, scale_y_value)
            return Y_q, scale_y_value
        
    def finish_calibration(self):
        self.scale_y = self.observer.get_scale().to(self.scale_y.device)
        self.is_quantized = True
    
class Custom_Element_Wise(torch.nn.Module):
    def __init__(self, max_length = 1024):
        super().__init__()
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_length)
        self.register_buffer('scale_out', torch.ones(max_length)) 
        self.is_quantized = False
    def forward(self, a, scale_a, b, scale_b):
        if self.is_quantized:
            
            seq_len = a.shape[0]
            scale_a_value = scale_a[:seq_len].to(torch.float32)
            scale_b_value = scale_b[:seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:seq_len].to(torch.float32)
            
            out_int8 = gemm_cutlass.func_element_wise_mul_int8(a, scale_a_value,\
                                b, scale_b_value,\
                                scale_out_value)
            return out_int8, scale_out_value
        else:
            out = a * b
            self.out_observer(out)
        return out, 1.0

    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        

class Custom_SiLU(nn.Module):
    def __init__(self, max_length=1024):
        super(Custom_SiLU, self).__init__()
        
        self.sigmoid = Custom_Sigmoid(max_seq_len=max_length)
        self.element_wise = Custom_Element_Wise(max_length=max_length)
        
        self.observer = MinMaxObserverPerLastDim(max_seq_len=max_length)
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            sigmoid_out, _ = self.sigmoid(x, scale_x)
            out, _ = self.element_wise(x, 1.0, sigmoid_out, 1.0)
            self.observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_x = scale_x[:seq_len].to(torch.float32)
            
            out_sigmoid_q, scale_sigmoid = self.sigmoid(x, scale_x)
            out_q, scale_out = self.element_wise(x, scale_x, out_sigmoid_q, scale_sigmoid)
            return out_q, scale_out

    def finish_calibration(self):
        self.sigmoid.finish_calibration()
        self.element_wise.finish_calibration()
        self.is_quantized = True  


if __name__ == "__main__":
    
    seq_len = 1024
    embedding_dim = 4096
    
    dtype = torch.bfloat16
    device = 'cuda'
    
    X = torch.randn(seq_len, embedding_dim, device=device, dtype=dtype) * 10.0
    silu_layer = Custom_SiLU(max_length=seq_len).to(device)
    Y_true, _ = silu_layer(X, 1.0)
    
    # 1. Calibration
    with torch.no_grad():
        for _ in range(10):  # Run multiple iterations to gather statistics
            x_calib = torch.randn(seq_len, embedding_dim, device=device, dtype=dtype)
            silu_layer(x_calib, 1.0)
    
    # 2. Finish calibration and prepare for quantization
    silu_layer.finish_calibration()  
    
    # 3. Quantized inference
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    
    Y_q, scale_y = silu_layer(X_int8, scale_x)
    Y_deq = Y_q.to(torch.float32) * scale_y.unsqueeze(-1)  
    
    print(f"Dtype of Y_true: {Y_true.dtype}, shape: {Y_true.shape}")
    print(f"Dtype of Y_deq: {Y_deq.dtype}, shape: {Y_deq.shape}")
    
    max_diff = torch.max(torch.abs(Y_true - Y_deq))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((Y_true - Y_deq) ** 2).item()
    print(f"MSE: {mse}")
    
