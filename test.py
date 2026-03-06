"""
In this script, I measure the time of small kernel
"""

import os 
import torch

def quantize_row_int8_symmetric_nd(
    mat: torch.Tensor,
    scale_dtype=torch.float32,
    percentile = None
):
    """
    Symmetric int8 quantization per row along the last dimension.

    If percentile is None, uses strict max(|x|) per row.
    If percentile is a float in (0,1], uses that percentile of |x| per row
    (e.g. 0.999 for 99.9%) for clipping.
    """

    assert mat.dim() >= 2, "mat must be at least 2D"

    # mat = mat.to(scale_dtype)
    qmin, qmax = -128, 127

    orig_shape = mat.shape  # (..., C)
    last_dim = orig_shape[-1]  # C
    num_vecs = mat.numel() // last_dim

    # Reshape to (num_vecs, C)
    mat_2d = mat.reshape(num_vecs, last_dim)

    max_vals = mat_2d.abs().amax(dim=1, keepdim=True)  # (num_vecs, 1)

    max_vals = max_vals.clamp(min=1e-8)

    # Per-row scale
    scales = (max_vals / qmax).squeeze(1)

    # Quantize
    q_mat_2d = torch.clamp(
        torch.round(mat_2d / scales.unsqueeze(1)),
        qmin,
        qmax
    ).to(torch.int8)

    # Reshape back
    q_mat = q_mat_2d.reshape(orig_shape)
    scales = scales.reshape(orig_shape[:-1])

    return q_mat, scales.to(scale_dtype)


if __name__ == "__main__":
    
    seq_length = 1024
    d_model = 8192
    n_heads = 16
    head_dim = d_model // n_heads

    dtype = torch.bfloat16
    device = "cuda"
    
    X = torch.randn((seq_length, d_model), dtype=dtype, device=device)
    Y = X.view(seq_length, n_heads, head_dim).transpose(0, 1)
    
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X)
    X_int8 = X_int8.view(seq_length, n_heads, head_dim).transpose(0, 1)
    X_scale = X_scale.unsqueeze(0).expand(n_heads, -1)
    print(f"Shape of X_int8: {X_int8.shape}")
    print(f"Shape of X_scale: {X_scale.shape}")
        
    # Dequantize
    X_dequant = (X_int8.float() * X_scale.unsqueeze(-1))
    print(f"Shape of X_dequant: {X_dequant.shape}")
    
    # Measure correctness
    mse = torch.mean((Y.float() - X_dequant) ** 2).item()
    print(f"MSE: {mse}")
    max_diff = torch.max(torch.abs(Y.float() - X_dequant)).item()
    print(f"Max absolute difference: {max_diff}")
    