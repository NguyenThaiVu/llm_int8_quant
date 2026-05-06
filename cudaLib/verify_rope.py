"""
In this script, we verify the correctness of our custom RoPE:
- We evaluate the performance of our INT8 quantized version against the bfloat16 version.
- We test both 2D and 3D inputs.
"""


import os 
import torch
import torch.nn as nn
from utils_quant import *
import gemm_cutlass

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

class Custom_RoPE(nn.Module):
    def __init__(self, num_heads, head_dim=None):
        super(Custom_RoPE, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
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
            cos = cos[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, head_dim)
            sin = sin[:seq_len, :].unsqueeze(0)

            # 3. Apply the rotary transformation
            rotated = torch.cat((-x2, x1), dim=-1)
            x_rotated = (x * cos) + (rotated * sin)
            
            # 4. Reshape back to original shape and dtype
            out = x_rotated.to(dtype=origin_dtype)
            out = out.view(origin_shape)  
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            seq_len = x.shape[1]
            
            cos = cos[:seq_len, :]
            sin = sin[:seq_len, :]

            Y_int8, scale_out = gemm_cutlass.func_apply_rope_int8(x, scale_x, \
                            cos, scale_cos,
                            sin, scale_sin)
            return Y_int8, scale_out
    
    def finish_calibration(self):
        self.is_quantized = True
        
if __name__ == "__main__":
    
    # seq_len = 1024
    # num_heads = 32
    head_dim = 128
    list_seq_len = [1024, 2048, 3072, 4096]
    list_num_heads = [16, 24, 32, 64]
    dtype = torch.bfloat16
    
    for seq_len in list_seq_len:
        for num_heads in list_num_heads:
            print(f"\nTesting RoPE with seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
            X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype).cuda()
            
            cos, sin = compute_rope_params(head_dim, context_length=seq_len, dtype=dtype)
            cos = cos.cuda()
            sin = sin.cuda()
            
            # Torch baseline
            rope_layer = Custom_RoPE(num_heads, head_dim).cuda()
            out_torch, _ = rope_layer(X, 1.0, cos, 1.0, sin, 1.0)
            
            torch_rope_time = measure_time(rope_layer, X, 1.0, cos, 1.0, sin, 1.0)
            print(f"PyTorch RoPE time: {torch_rope_time:.2f} ms")
            
            # Quantized version of RoPE
            rope_layer.finish_calibration()  
            X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
            cos_int8, scale_cos = quantize_tensor(cos)
            sin_int8, scale_sin = quantize_tensor(sin)
            
            out_int8, scale_out = rope_layer(X_int8, scale_x, cos_int8, scale_cos, sin_int8, scale_sin)
            int8_rope_time = measure_time(rope_layer, X_int8, scale_x, cos_int8, scale_cos, sin_int8, scale_sin)
            print(f"Quantized RoPE time: {int8_rope_time:.2f} ms")
            
            out_deq = out_int8.float() * scale_out.unsqueeze(-1)
            out_deq = out_deq.to(dtype)

            # Verify correctness
            max_diff = (out_torch - out_deq).abs().max().item()
            print(f"Max difference: {max_diff:.6f}")    
            mse = ((out_torch - out_deq) ** 2).mean().item()
            print(f"MSE: {mse:.6f}\n")
            
        