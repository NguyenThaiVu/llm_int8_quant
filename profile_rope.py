"""
In this script, we profile the Rotary Positional Embedding (RoPE)
"""

import os 
import math
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from utils import *

def compute_rope_params(head_dim, theta_base=10_000,\
                    context_length=4096, dtype=torch.bfloat16, device=None):
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

    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    
    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply Rotary Positional Embedding (RoPE) to the input tensor x.
    Args:
        x: Input tensor of shape (num_heads, seq_len, head_dim)
        cos: Precomputed cos values of shape (context_length, head_dim)
        sin: Precomputed sin values of shape (context_length, head_dim)
    """
    
    num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half
    rotated = torch.cat((-x2, x1), dim=-1)

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, head_dim)

    # Apply the rotary transformation
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


if __name__ == "__main__":
    seq_len = 1024 * 16
    head_dim = 128
    num_heads = 40

    dtype = torch.bfloat16
    device = 'cuda'
    
    X = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = compute_rope_params(head_dim, context_length=seq_len, dtype=dtype, device=device)
    
    Y = apply_rope(X, cos, sin)
    
    print("Input shape:", X.shape)
    print("Output shape:", Y.shape)
    
    n_iter = 10
    
    rope_time = measure_time(apply_rope, X, cos, sin, repeat=n_iter)
    print(f"RoPE time: {rope_time:.2f} ms")
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     for _ in range(n_iter):
    #         _ = apply_rope(X, cos, sin)
            
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))