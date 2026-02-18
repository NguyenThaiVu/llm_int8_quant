"""
In this script, we profiler time of each layer in Transformer block:
1. Multi-Head Attention
2. Feed-Forward Network (FFN)
3. Layer Normalization

We ignore batch size for simplicity.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function


def compute_rope_params(head_dim, seq_len, base=10000.0, device=None, dtype=torch.float32):
    """
    Returns cos, sin with shape (1, 1, S, D) for broadcasting over (B, H, S, D).
    Half-split convention: rotate first half against second half.
    """
    assert head_dim % 2 == 0

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))  # (D/2,)
    pos = torch.arange(seq_len, device=device, dtype=dtype)  # (S,)
    angles_half = torch.einsum("s,d->sd", pos, inv_freq)     # (S, D/2)

    # Expand to D by repeating the angles for the second half
    angles = torch.cat([angles_half, angles_half], dim=-1)   # (S, D)
    cos = angles.cos()[None, None, :, :]                     # (1,1,S,D)
    sin = angles.sin()[None, None, :, :]                     # (1,1,S,D)
    return cos, sin


def apply_rope(x, cos, sin):
    """
    x:   (B, H, S, D)
    cos: (1, 1, S, D)
    sin: (1, 1, S, D)
    """
    D = x.shape[-1]
    assert D % 2 == 0
    half = D // 2

    x1 = x[..., :half]              # (B,H,S,D/2)
    x2 = x[..., half:]              # (B,H,S,D/2)
    x_rot = torch.cat([-x2, x1], dim=-1)  # (B,H,S,D)

    return (x * cos) + (x_rot * sin)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.cos, self.sin = compute_rope_params(self.head_dim, seq_len=2048, device='cuda', dtype=torch.bfloat16)

    def forward(self, x):
        S, E = x.shape
        scale = self.head_dim ** 0.5
        
        # 1. Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for multi-head attention
        q = q.view(S, self.num_heads, self.head_dim).transpose(0, 1)      # (H,S,D)
        k = k.view(S, self.num_heads, self.head_dim).transpose(0, 1)      # (H,S,D)
        
        # Apply RoPE
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        
        v = v.view(S, self.num_heads, self.head_dim).transpose(0, 1)      # (H,S,D)
        kT = k.transpose(-2, -1).contiguous()                             # (H,D,S)

        # 3. Scaled dot-product attention
        scores = torch.matmul(q, kT) / scale                               # (H,S,S)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                       # (H,S,D)

        # 4. Concatenate heads and final projection
        out = out.transpose(0, 1).contiguous().view(S, E)
        out = self.out_proj(out)
        return out
    
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    
class Transformer_Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x 
        x = self.mha(x)
        x = self.norm1(x + residual) 
        
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return x

def profile_transformer_block(seq_length, embed_dim, num_heads, ffn_dim, dtype, iterations=10):
    device = "cuda"
    model = Transformer_Block(embed_dim, num_heads, ffn_dim).to(device).to(dtype)
    input_tensor = torch.randn(seq_length, embed_dim, device=device, dtype=dtype)

    # Warm-up
    for _ in range(5):
        _ = model(input_tensor)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            _ = model(input_tensor)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))


if __name__ == "__main__":
    seq_length = 2048
    model_name = "gpt2"  # Options: "llama-3.2-3B", "qwen3-8B", "gpt2", "qwen3-14B"
    
    if model_name == "llama-3.2-3B":
        embed_dim = 3072
        num_heads = 24
        ffn_dim = 8192
    elif model_name == "qwen3-8B":
        embed_dim = 4096
        num_heads = 32
        ffn_dim = 12288
    elif model_name == "gpt2":
        embed_dim = 1600
        num_heads = 25
        ffn_dim = 6400
    elif model_name == 'qwen3-14B':
        embed_dim = 5120
        num_heads = 40
        ffn_dim = 17408
    else:
        embed_dim = 768
        num_heads = 12
        ffn_dim = 3072
        
    dtype = torch.bfloat16
    profile_transformer_block(seq_length, embed_dim, num_heads, ffn_dim,\
        dtype=dtype, iterations=10)

