"""
In this script, we verify the correctness of `func_silu_mul_int8` function
"""
import os 
import torch 
import gemm_cutlass
from utils import quantize_row_int8_symmetric_nd


if __name__ == "__main__":
    seq_len = 1024
    emb_dim = 4096
    d_type = torch.bfloat16
    
    X1 = torch.randn(seq_len, emb_dim, device='cuda', dtype=d_type)
    X2 = torch.randn(seq_len, emb_dim, device='cuda', dtype=d_type)
    
    # Compute reference result using PyTorch
    silu_X1 = X1 * torch.sigmoid(X1)
    ref_result = silu_X1 * X2
    
    # Compute result using our custom CUDA function
    X1_int8, scale_X1 = quantize_row_int8_symmetric_nd(X1)
    X2_int8, scale_X2 = quantize_row_int8_symmetric_nd(X2)
    out_int8, out_scale = gemm_cutlass.func_silu_mul_int8(X1_int8, scale_X1, X2_int8, scale_X2)
    
    # Verify correctness
    custom_result = out_int8.to(torch.float32) * out_scale.unsqueeze(-1)
    custom_result = custom_result.to(d_type)
    
    max_diff = torch.max(torch.abs(ref_result - custom_result)).item()
    print(f"Max difference: {max_diff:.6f}")
    mse = torch.mean((ref_result - custom_result) ** 2).item()
    print(f"MSE: {mse:.6e}")

