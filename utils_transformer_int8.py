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
    

# class MinMaxObserverPerLastDim(nn.Module):
#     def __init__(self, max_batch: int = 1, max_seq_len: int = 1024, eps: float = 1e-8):
#         super().__init__()
#         self.max_batch = max_batch
#         self.max_seq_len = max_seq_len
#         self.eps = eps

#         # Always store as float32, independent of model compute dtype
#         self.register_buffer(
#             "max_val",
#             torch.full((max_batch, max_seq_len), -torch.inf, dtype=torch.float32),
#         )
#         self.register_buffer(
#             "min_val",
#             torch.full((max_batch, max_seq_len),  torch.inf, dtype=torch.float32),
#         )

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         """
#         x can be:
#           - (T, H)      -> treated as B=1
#           - (B, T, H)   -> true token-wise per batch
#         We track min/max over the last dim (H).
#         """
#         if x.ndim == 2:
#             T, _ = x.shape
#             B = 1
#             # Detach and upcast to float32 for stable stats
#             xd = x.detach().to(torch.float32).unsqueeze(0)  # (1, T, H)
#         elif x.ndim == 3:
#             B, T, _ = x.shape
#             xd = x.detach().to(torch.float32)               # (B, T, H)
#         else:
#             raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D with shape {tuple(x.shape)}")

#         if B > self.max_batch:
#             raise ValueError(f"B={B} exceeds max_batch={self.max_batch}")
#         if T > self.max_seq_len:
#             raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}")

#         # token-wise over hidden dim -> (B, T), in float32
#         cur_max = xd.amax(dim=-1)  # (B, T)
#         cur_min = xd.amin(dim=-1)  # (B, T)

#         # update only prefix [0:B, 0:T) in float32 buffers
#         self.max_val[:B, :T] = torch.maximum(self.max_val[:B, :T], cur_max)
#         self.min_val[:B, :T] = torch.minimum(self.min_val[:B, :T], cur_min)

#         # Return original x unchanged (keeps original dtype)
#         return x

#     @torch.no_grad()
#     def get_scale(self, B: int | None = None, T: int | None = None, qmax: int = 127) -> torch.Tensor:
#         """
#         Returns:
#           - if max_batch == 1 or B is None:  (T,)      float32 scale
#           - else:                            (B, T)    float32 scale
#         """
#         # Default to full length we've reserved
#         if T is None:
#             T = self.max_seq_len

#         # Always do computations in float32
#         max_val = self.max_val.to(torch.float32)
#         min_val = self.min_val.to(torch.float32)

#         # Per-token max abs
#         if self.max_batch == 1 or B is None:
#             # 2D use-case: we only care about batch 0
#             max_abs = torch.maximum(max_val[0, :T].abs(), min_val[0, :T].abs())  # (T,)
#         else:
#             max_abs = torch.maximum(max_val[:B, :T].abs(), min_val[:B, :T].abs())  # (B, T)

#         scale = (max_abs / float(qmax)).clamp(min=self.eps)  # ensure > 0
#         # Explicitly ensure float32 output
#         return scale.to(torch.float32)    
    
    
class MinMaxObserverPerLastDim(nn.Module): 
    def __init__(self, max_batch=1, max_seq_len=1024): 
        super().__init__() 
        self.max_batch = max_batch 
        self.max_seq_len = max_seq_len 
        
        # Always store as (max_batch, max_seq_len) 
        self.register_buffer("max_val", torch.full((max_batch, max_seq_len), -torch.inf)) 
        self.register_buffer("min_val", torch.full((max_batch, max_seq_len), torch.inf)) 
        
    @torch.no_grad() 
    def forward(self, x: torch.Tensor): 
        """ 
        x can be: - (T, H) -> treated as B=1 - (B, T, H) -> true token-wise per batch 
        """ 
        if x.ndim == 2: 
            T, _ = x.shape 
            B = 1 
            xd = x.detach().unsqueeze(0) # (1, T, H) 
        elif x.ndim == 3: 
            B, T, _ = x.shape 
            xd = x.detach() 
        else: raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D with shape {tuple(x.shape)}") 
        
        if B > self.max_batch: 
            raise ValueError(f"B={B} exceeds max_batch={self.max_batch}") 
        if T > self.max_seq_len: 
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}") 
        
        # token-wise over hidden dim -> (B, T) 
        cur_max = xd.amax(dim=-1) 
        cur_min = xd.amin(dim=-1) 
        
        # update only prefix [0:B, 0:T) 
        self.max_val[:B, :T] = torch.maximum(self.max_val[:B, :T], cur_max) 
        self.min_val[:B, :T] = torch.minimum(self.min_val[:B, :T], cur_min) 
        return x 
    
    def get_scale(self, B=None, T=None): 
        qmax = 127 # symmetric int8 
        if T == None: 
            T = self.max_seq_len 
        
        # if input 2D, return (T,) scale; 
        if self.max_batch == None or self.max_batch == 1: 
            max_abs = torch.maximum(self.max_val[0, :T].abs(), self.min_val[0, :T].abs()) 
            return (max_abs / qmax).clamp(min=1e-8) # (T,) 
        # if input 3D, return (B,T) scale; 
        max_abs = torch.maximum(self.max_val[:B, :T].abs(), self.min_val[:B, :T].abs()) 
        return (max_abs / qmax).clamp(min=1e-8) # (B, T)
    
    
    
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

