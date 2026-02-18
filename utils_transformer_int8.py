import os 
import torch
import torch.nn as nn

class MinMaxObserver(nn.Module):
    def __init__(self):
        super(MinMaxObserver, self).__init__()
        self.register_buffer('max_val', torch.tensor(float("-inf")))
        self.register_buffer('min_val', torch.tensor(float("inf")))
    
    def forward(self, x):
        x_detached = x.detach()
        current_max = x_detached.max()
        current_min = x_detached.min()
        self.max_val = torch.max(self.max_val, current_max)
        self.min_val = torch.min(self.min_val, current_min)
        return x
    
    def get_scale(self):
        qmax = 127  # int8 symmetric
        max_abs = torch.max(self.max_val.abs(), self.min_val.abs())
        scale = max_abs / qmax
        scale = scale.clamp(min=1e-8)
        return scale    
    
    
class PerHeadMinMaxObserver(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.register_buffer('max_val', torch.full((num_heads,), float("-inf")))
        self.register_buffer('min_val', torch.full((num_heads,), float("inf")))
    
    def forward(self, x):
        # x could be (B, H, L, D) or (H, L, D)
        x_detached = x.detach()
        
        if x.dim() == 4:
            # Reduce Batch (0), Seq (2), and Head_Dim (3)
            reduction_dims = (0, 2, 3)
        elif x.dim() == 3:
            # Reduce Seq (1) and Head_Dim (2)
            reduction_dims = (1, 2)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")

        current_max = x_detached.amax(dim=reduction_dims) 
        current_min = x_detached.amin(dim=reduction_dims)
        
        # Update buffers
        self.max_val = torch.max(self.max_val, current_max)
        self.min_val = torch.min(self.min_val, current_min)
        return x
    
    def get_scale(self):
        qmax = 127
        max_abs = torch.max(self.max_val.abs(), self.min_val.abs())
        scale = max_abs / qmax
        return scale.clamp(min=1e-8)
    
    
class MinMaxObserverPerLastDim(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

        # One value per token position
        self.register_buffer(
            "max_val", torch.full((max_seq_len,), -torch.inf)
        )
        self.register_buffer(
            "min_val", torch.full((max_seq_len,),  torch.inf)
        )

    @torch.no_grad()
    def forward(self, x):
        # x: (T, H)
        T, _ = x.shape

        if T > self.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")

        # token-wise stats over hidden dim
        cur_max = x.detach().amax(dim=-1)  # (T,)
        cur_min = x.detach().amin(dim=-1)  # (T,)

        # update only prefix [0:T)
        self.max_val[:T] = torch.maximum(self.max_val[:T], cur_max)
        self.min_val[:T] = torch.minimum(self.min_val[:T], cur_min)

        return x

    def get_scale(self, T: int):
        qmax = 127  # symmetric int8
        max_abs = torch.maximum(
            self.max_val[:T].abs(),
            self.min_val[:T].abs()
        )
        return (max_abs / qmax).clamp(min=1e-8)  # (T,)

# class MinMaxObserverPerLastDim(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.register_buffer("max_val", None)
#         self.register_buffer("min_val", None)

#     def forward(self, x):
#         x_detached = x.detach()

#         # Reduce over all dims except last
#         reduce_dims = tuple(range(x_detached.ndim - 1))
#         current_max = x_detached.amax(dim=reduce_dims)
#         current_min = x_detached.amin(dim=reduce_dims)

#         if self.max_val is None:
#             self.max_val = current_max.clone()
#             self.min_val = current_min.clone()
#         else:
#             self.max_val = torch.maximum(self.max_val, current_max)
#             self.min_val = torch.minimum(self.min_val, current_min)

#         return x

#     def get_scale(self):
#         qmax = 127
#         max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs())
#         scale = max_abs / qmax
#         return scale.clamp(min=1e-8)


    
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    # This works for (Batch, Heads, Seq, Dim) OR (Heads, Seq, Dim)
    # because we index from the right (-1, -2, -3)
    
    target_seq_len = x.shape[-2]
    head_dim = x.shape[-1]
    
    # Slice cos/sin to match current sequence length
    cos = cos[:target_seq_len, :] # (Seq, Dim)
    sin = sin[:target_seq_len, :] # (Seq, Dim)

    # Unsqueeze enough times to match x's rank
    # If x is 4D: (1, 1, Seq, Dim)
    # If x is 3D: (1, Seq, Dim)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Split and Rotate
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    return (x * cos) + (rotated * sin)

