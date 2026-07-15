"""
In this script, we verify the correctness and latency of RMSNorm between
- BF16
- INT8 
"""

import os 
import torch
import torch.nn.functional as F
import gemm_cutlass
from utils_quant import *

def compute_total_data_movement(X=None, W=None, Y=None, unit="byte"):
    total_data_move = 0
    if X is not None:
        total_data_move += X.numel() * X.element_size()
    if W is not None:
        total_data_move += W.numel() * W.element_size()
    if Y is not None:
        total_data_move += Y.numel() * Y.element_size()
    if unit == "byte":
        pass
    elif unit == "KB":
        total_data_move /= 1024
    elif unit == "MB":
        total_data_move /= 1024 * 1024
        total_data_move = round(total_data_move, 2)
    elif unit == "GB":
        total_data_move /= 1024 * 1024 * 1024
        total_data_move = round(total_data_move, 2)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    
    return total_data_move

if __name__ == "__main__":
    # =========================== 2D Input ====================================
    print(f"\nTesting RMSNorm with 2D input")
    seq_len = 4096
    embed_dim = 8192
    # list_seq_len = [1024, 2048, 3072]
    # list_embed_dim = [2048, 4096, 5120, 8192]
    d_type = torch.bfloat16
    
    print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
    x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
    gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
    
    # =======================================
    # 1. Pytorch baseline BF16 RMSNorm
    y_torch = F.rms_norm(x, (embed_dim,), weight=gamma, eps=1e-6)
    total_data_move = compute_total_data_movement(X=x, W=gamma, Y=y_torch, unit='MB')
    print(f"Total data movement (PyTorch BF16): {total_data_move:.2f} MB")
    torch_rmsnorm_time = measure_time(F.rms_norm, x, (embed_dim,), gamma, 1e-6)
    print(f"PyTorch RMSNorm time: {torch_rmsnorm_time:.2f} ms")

    # BF16 RMSNorm
    y_bf16 = gemm_cutlass.func_rmsnorm_bf16(x, gamma, 1e-6)
    bf16_rmsnorm_time = measure_time(gemm_cutlass.func_rmsnorm_bf16, x, gamma, 1e-6)
    print(f"BF16 RMSNorm time: {bf16_rmsnorm_time:.2f} ms")
    
    # INT8 RMSNorm
    x_int8, scale_x = quantize_row_int8_symmetric_nd(x)
    y_int8, scale_y = gemm_cutlass.func_rmsnorm_int8(x_int8, scale_x, gamma, 1e-6)
    int8_rmsnorm_time = measure_time(gemm_cutlass.func_rmsnorm_int8, x_int8, scale_x, gamma, 1e-6)
    print(f"INT8 RMSNorm time: {int8_rmsnorm_time:.2f} ms")
    print(f"Speedup over PyTorch: {torch_rmsnorm_time / int8_rmsnorm_time:.2f}x")
    print(f"Speedup over BF16: {bf16_rmsnorm_time / int8_rmsnorm_time:.2f}x\n")
