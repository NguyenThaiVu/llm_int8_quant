"""
In this script, we test the linear layer with smoothness
"""

import os 
import torch
import torch.nn as nn
from utils import quantize_row_int8_symmetric_nd, quantize_tensor
import gemm_cutlass

def dummy_matmul_int8(X_int8, W_int8, dtype=torch.bfloat16):
    assert X_int8.dtype == torch.int8
    assert W_int8.dtype == torch.int8
    
    X = X_int8.to(torch.float32)
    W = W_int8.to(torch.float32)
    Y = X @ W.T
    Y = Y.to(dtype)
    return Y


def compute_smoothquant_alpha(X, W, lambd=0.5, eps=1e-6):
    """
    Compute per-input-channel SmoothQuant scaling factors.

    X: (seq_len, in_dims)
    W: (out_dims, in_dims)
    lambd: tradeoff parameter in [0, 1]
    returns alpha: (in_dims,)
    """
    
    # Make sure X and W match in dimensions
    assert X.shape[1] == W.shape[1], "Input channels must match between X and W"
    
    # activation magnitude per input channel
    A = X.abs().amax(dim=0)  # shape: (in_dims,)

    # weight magnitude per input channel
    W_row = W.abs().amax(dim=0)  # shape: (in_dims,)

    # avoid zeros to prevent NaNs/infs
    A = torch.clamp(A, min=eps)
    W_row = torch.clamp(W_row, min=eps)

    # SmoothQuant formula alpha_j = A_j^lambda / W_j^(1 - lambda)
    alpha = (A ** lambd) / (W_row ** (1.0 - lambd))

    # Optional: clamp alpha to avoid crazy scaling
    alpha = torch.clamp(alpha, min=0.01, max=100.0)

    return alpha


if __name__ == "__main__":
    
    seq_len = 1024
    in_dims = 4096
    out_dims = 8192
    
    dtype = torch.bfloat16
    
    # 1. True linear layer
    X = torch.randn(seq_len, in_dims, dtype=dtype)
    num_outliers = 5_000
    for _ in range(num_outliers):
        idx = torch.randint(0, seq_len, (1,))
        dim = torch.randint(0, in_dims, (1,))
        scale = torch.rand(1) * 100 + 100  # random scale between 100 and 200
        X[idx, dim] *= scale
    X = X.cuda()
        
    percent_outliers = num_outliers / (seq_len * in_dims) * 100
    print(f"X has {percent_outliers:.4f}% outliers.\n")
    
    W = torch.empty(out_dims, in_dims, dtype=dtype).cuda()
    nn.init.kaiming_normal_(W, mode='fan_in', nonlinearity='relu')
    
    Y = X @ W.T
    
    # 2. Quantization on X and W
    X_int8, X_scale = quantize_row_int8_symmetric_nd(X)
    W_int8, W_scale = quantize_tensor(W)
    
    Y_deq = dummy_matmul_int8(X_int8, W_int8, dtype=dtype)
    Y_deq = Y_deq * X_scale.unsqueeze(-1) * W_scale    
    
    print(f"===== Normal quantization =====")
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max diff: {max_diff.item():.4f}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"MSE: {mse:.4f}")
    print()
    
    # 3. Quantization on Smoothness
    alpha = compute_smoothquant_alpha(X, W)
    X_smooth = X / alpha.unsqueeze(0)      
    W_smooth = W * alpha.unsqueeze(0)      
    
    X_int8_smooth, X_scale_smooth = quantize_row_int8_symmetric_nd(X_smooth)
    W_int8_smooth, W_scale_smooth = quantize_row_int8_symmetric_nd(W_smooth)
    Y_deq_smooth = gemm_cutlass.func_int8_matmul(X_int8_smooth, W_int8_smooth, 1.0)
    # Y_deq_smooth = dummy_matmul_int8(X_int8_smooth, W_int8_smooth, dtype=dtype)
    
    Y_deq_smooth = Y_deq_smooth * X_scale_smooth.unsqueeze(-1)\
                                * W_scale_smooth.unsqueeze(0)    
    
    print(f"===== Smooth quantization =====")
    max_diff = torch.max(torch.abs(Y - Y_deq_smooth))
    print(f"Max diff (smoothed): {max_diff.item():.4f}")
    mse = torch.mean((Y - Y_deq_smooth) ** 2).item()
    print(f"MSE (smoothed): {mse:.4f}")
    print()