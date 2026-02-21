"""
In this script, we verify the correctness of RoPE
"""

import os 
import torch
from torch import nn
import gemm_cutlass
from utils import *
from utils_transformer import *
from utils_transformer_int8 import *


class Custom_RoPE(nn.Module):
    def __init__(self, num_heads, max_seq_len=1024, head_dim=None):
        super(Custom_RoPE, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
        self.register_buffer('scale_out',\
                    torch.ones(self.num_heads * self.max_seq_len)) 
        self.is_quantized = False
        
    def forward(self, x, scale_x, 
                    cos, scale_cos,
                    sin, scale_sin):
        origin_shape = x.shape
        origin_dtype = x.dtype
        
        num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        
        if self.is_quantized == False:
            # 1. Split x into first half and second half
            x1 = x[..., : head_dim // 2]  # First half
            x2 = x[..., head_dim // 2 :]  # Second half

            # 2. Adjust sin and cos shapes
            cos = cos[:seq_len, :].unsqueeze(0)  # Shape:  1, seq_len, head_dim)
            sin = sin[:seq_len, :].unsqueeze(0)

            # 3. Apply the rotary transformation
            rotated = torch.cat((-x2, x1), dim=-1)
            x_rotated = (x * cos) + (rotated * sin)
            
            # 4. Reshape back to original shape and dtype
            out = x_rotated.to(dtype=origin_dtype)
            out = out.view(origin_shape)  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            seq_len = x.shape[1]
            scale_x_value = scale_x[:seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:seq_len].to(torch.float32)
            
            Y_int8 = gemm_cutlass.func_apply_rope_int8(x, scale_x_value, \
                            cos, scale_cos,
                            sin, scale_sin,
                            scale_out_value)
            return Y_int8, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        
        
if __name__ == "__main__":
    
    seq_len = 1024
    head_dim = 128
    num_heads = 32

    dtype = torch.bfloat16
    device = 'cuda'
    
    X = init_random_tensor((num_heads, seq_len, head_dim), device=device, dtype=dtype)
    print(f"Input sample: {X[0, :5, :5]} \n")
    
    cos, sin = compute_rope_params(head_dim, context_length=seq_len, dtype=dtype)
    cos, sin = cos.to(device), sin.to(device)
    print(f"Input shape: {X.shape}, Cos shape: {cos.shape}, Sin shape: {sin.shape}")
    
    # 1. Calibration
    rope_layer = Custom_RoPE(num_heads, max_seq_len=seq_len, head_dim=head_dim)
    rope_layer = rope_layer.to(device).to(dtype)
    Y, _ = rope_layer(X, 1.0, cos, 1.0, sin, 1.0)
    print(f"Output shape: {Y.shape}")
    print(f"Output sample: {Y[0, :5, :5]} \n")
    
    # 2. Quantization
    rope_layer.finish_calibration()
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    cos_int8, scale_cos = quantize_tensor(cos)
    sin_int8, scale_sin = quantize_tensor(sin)
    print("Calibration finished.")
    
    # 3. Quantized inference
    Y_q, scale_out = rope_layer(X_int8, scale_x,\
                        cos_int8, scale_cos,\
                        sin_int8, scale_sin)
    print(f"Quantized output shape: {Y_q.shape}")
    print(f"Quantized output sample: {Y_q[0, :5, :5]} \n")
    
    Y_deq = Y_q.float() * scale_out.view(num_heads, seq_len, 1)
    print(f"Dequantized output shape: {Y_deq.shape}")
    print(f"Dequantized output sample: {Y_deq[0, :5, :5]} \n")
    
    max_diff = torch.max(torch.abs(Y_deq - Y))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((Y_deq - Y) ** 2).item()
    print(f"MSE: {mse}")