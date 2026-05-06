import os 
import torch 
from torch import nn
from torch.nn import functional as F
import gemm_cutlass
from utils_quant import *

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
    seq_len = 2048
    num_heads = 16
    head_dim = 128
    
    dtype = torch.bfloat16
    
    X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype).cuda()
    cos = torch.randn(seq_len, head_dim, dtype=dtype).cuda()
    sin = torch.randn(seq_len, head_dim, dtype=dtype).cuda()
    
    rope_layer = Custom_RoPE(num_heads=num_heads, head_dim=head_dim).cuda().to(dtype)
    
    Y_true, _ = rope_layer(X, 1.0, cos, 1.0, sin, 1.0)
    
    bf16_rope_time = measure_time(rope_layer, X, 1.0, cos, 1.0, sin, 1.0)
    print(f"RoPE time (bf16): {bf16_rope_time:.2f} ms")
    
    # Quantization
    rope_layer.finish_calibration()
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    cos_int8, scale_cos = quantize_tensor(cos)
    sin_int8, scale_sin = quantize_tensor(sin)
    
    int8_rope_time = measure_time(rope_layer, X_int8, scale_x, \
                                    cos_int8, scale_cos, \
                                    sin_int8, scale_sin)
    print(f"RoPE time (int8): {int8_rope_time:.2f} ms")
    
    Y_int8, scale_out = rope_layer(X_int8, scale_x, \
                                    cos_int8, scale_cos, \
                                    sin_int8, scale_sin)
    Y_deq = Y_int8.float() * scale_out.unsqueeze(-1)
    Y_deq = Y_deq.to(dtype)
    
    max_diff = (Y_true - Y_deq).abs().max().item()
    mse = ((Y_true - Y_deq).pow(2).mean().item())
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"MSE: {mse:.6f}")