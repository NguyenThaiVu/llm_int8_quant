import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from utils_quant import *

import custom_ops

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


def apply_rope(x, cos, sin):
    num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = "cuda"
    
    n_heads = 40
    head_dim = 128
    context_length = 2048
    
    # Create a input tensor
    cos, sin = compute_rope_params(head_dim, context_length=context_length)
    cos = cos.to(dtype=dtype, device=device)
    sin = sin.to(dtype=dtype, device=device)
    x = torch.randn(n_heads, context_length, head_dim, dtype=dtype, device=device)

    # ============== 1. Torch RoPE ==============
    func_apply_rope = apply_rope
    func_apply_rope = torch.compile(func_apply_rope)  
    x_rotated = func_apply_rope(x, cos, sin)
    print("Input shape:", x.shape)
    print("Rotated shape:", x_rotated.shape)
    
    rope_time = measure_time(func_apply_rope, x, cos, sin)
    print(f"Time (torch) RoPE: {rope_time:.6f} ms")
    
    # ============== 2. INT8 RoPE ==============
    x_i8, scale_x = quantize_row_int8_symmetric_nd(x)
    cos_i8, scale_cos = quantize_tensor(cos)
    sin_i8, scale_sin = quantize_tensor(sin)
    
    y_i8, scale_y = custom_ops.rope_int8(x_i8, scale_x, cos_i8, scale_cos, sin_i8, scale_sin)
    
    i8_rope_time = measure_time(custom_ops.rope_int8, x_i8, scale_x, cos_i8, scale_cos, sin_i8, scale_sin)
    print(f"Time INT8 RoPE: {i8_rope_time:.6f} ms")
    
    # 3. Measure correctness
    y_deq = y_i8.to(torch.float32) * scale_y.unsqueeze(-1)
    max_diff = torch.max(torch.abs(y_deq - x_rotated))
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean((y_deq - x_rotated) ** 2).item()
    print(f"MSE: {mse:.6f}")