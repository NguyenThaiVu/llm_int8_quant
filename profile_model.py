"""
In this script, we profile the time of tiny model.
"""

import os
import torch
from torch import nn
from utils import *
import gemm_cutlass

seq_len = 4096
input_dims = 8192
hidden_dims = 8192
output_dims = 8192
dtype = torch.bfloat16

class TinyModel(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(TinyModel, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.softmax = nn.Softmax(dim=-1)
        self.linear2 = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = self.softmax(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Custom_Linear, self).__init__()
        self.weight_q = torch.randint(-128, 128, (out_features, in_features),\
                                        device='cuda', dtype=torch.int8)
        
        self.weight_scale = 0.1  # Dummy scale for weight quantization
        self.out_scale = 0.1 # Dummy scale for output quantization

    def forward(self, x_q, scale_x):
        Y_q = gemm_cutlass.func_int8_matmul_output_int8(
            x_q, 
            self.weight_q, 
            scale_x * self.weight_scale / self.out_scale
        )
        return Y_q, self.out_scale
    
class Custom_Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Custom_Softmax, self).__init__()
        self.dim = dim
        self.out_scale = 0.1  # Dummy scale for output quantization

    def forward(self, x_q, scale_x):
        Y_q = gemm_cutlass.func_softmax_lastdim_int8(x_q, scale_x, self.out_scale)
        return Y_q, self.out_scale
    
class TinyModelQuantized(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(TinyModelQuantized, self).__init__()
        self.linear1 = Custom_Linear(input_dims, hidden_dims)
        self.softmax1 = Custom_Softmax(dim=-1)
        self.linear2 = Custom_Linear(hidden_dims, output_dims)
        self.softmax2 = Custom_Softmax(dim=-1)
        
    def forward(self, x_q, scale_x):
        x_q, scale_x = self.linear1(x_q, scale_x)
        x_q, scale_x = self.softmax1(x_q, torch.full((x_q.size(0),), scale_x, device=x_q.device))
        x_q, scale_x = self.linear2(x_q, scale_x)
        x_q, scale_x = self.softmax2(x_q, torch.full((x_q.size(0),), scale_x, device=x_q.device))
        
        x_deq = x_q.to(torch.bfloat16) * scale_x
        return x_deq


# 1. Normal Model
model = TinyModel(input_dims, hidden_dims, output_dims).cuda().to(dtype)

X = torch.randn((seq_len, input_dims), device='cuda', dtype=dtype)
Y = model(X)

model_time = measure_time(model, X, repeat=100)
print(f"Model time: {model_time:.3f} ms")

# 2. Quantized Model
model_q = TinyModelQuantized(input_dims, hidden_dims, output_dims).cuda()
X_q, scale_x = quantize_tensor(X)
Y_deq = model_q(X_q, scale_x)

model_q_time = measure_time(model_q, X_q, scale_x, repeat=100)
print(f"Quantized Model time: {model_q_time:.3f} ms")

if torch.allclose(Y, Y_deq, rtol=5.0, atol=5.0):
    print("Quantized model correct.")
else:
    print("===== [ERROR] Quantized model incorrect. =====")
    max_diff = (Y - Y_deq).abs().max()
    print(f"Max difference: {max_diff.item()}")