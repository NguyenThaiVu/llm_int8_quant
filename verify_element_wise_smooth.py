"""
In this script, we verify the element-wise smoothness
"""

import os 
import torch
from utils import quantize_row_int8_symmetric_nd

def compute_elementwise_smooth_alpha(A: torch.Tensor,
                                     B: torch.Tensor,
                                     lambd: float = 0.5,
                                     eps: float = 1e-6,
                                     alpha_min: float = 0.01,
                                     alpha_max: float = 100.0) -> torch.Tensor:
    """
    Compute per-channel SmoothQuant-style scaling factors for Y = A * B.

    A: (N, d)
    B: (N, d) or (d,)
    lambd: trade-off in [0, 1]; 0.5 is a reasonable default
    Returns:
        alpha: (d,)
    """
    assert A.ndim == 2, "Expected A shape (N, d)"
    N, d = A.shape

    # Activation magnitude per channel
    A_mag = A.abs().amax(dim=0)  # (d,)

    # B magnitude per channel
    if B.ndim == 2:
        assert B.shape[1] == d
        B_mag = B.abs().amax(dim=0)  # (d,)
    elif B.ndim == 1:
        assert B.shape[0] == d
        B_mag = B.abs()
    else:
        raise ValueError("B must be shape (N, d) or (d,)")

    A_mag = torch.clamp(A_mag, min=eps)
    B_mag = torch.clamp(B_mag, min=eps)

    # SmoothQuant-like α
    alpha = (A_mag ** lambd) / (B_mag ** (1.0 - lambd))

    # Clamp to avoid extreme scaling
    alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)
    return alpha  # (d,)

if __name__ == "__main__":
    seq_len = 1024
    emb_dim = 8192
    dtype = torch.bfloat16
    
    A = torch.rand(seq_len, emb_dim, dtype=dtype) * 10.0  
    B = torch.rand(seq_len, emb_dim, dtype=dtype) * 10.0  
    
    C_true = A * B
    
    smooth_factor = compute_elementwise_smooth_alpha(A, B, lambd=0.5)
    
    A_smooth = A / smooth_factor.unsqueeze(0)
    B_smooth = B * smooth_factor.unsqueeze(0)
    
    A_smooth_int8, scale_A_smooth = quantize_row_int8_symmetric_nd(A_smooth)
    B_smooth_int8, scale_B_smooth = quantize_row_int8_symmetric_nd(B_smooth)
    
    C_deq = A_smooth_int8.to(dtype) * B_smooth_int8.to(dtype) 
    C_deq = C_deq * scale_A_smooth.unsqueeze(-1) * scale_B_smooth.unsqueeze(-1)
    
    max_diff = (C_true - C_deq).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    mse = ((C_true - C_deq).float() ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    