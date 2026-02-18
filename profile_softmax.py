"""
In this script, we profile the softmax on 3D input along the last dimension:
- Option 1: using PyTorch's built-in softmax function.
- Option 2: using a custom CUDA kernel.
"""

import os 
import torch
from torch.profiler import profile, ProfilerActivity, record_function
import gemm_cutlass  

def profile_function(func, *args, **kwargs):
    num_iters = 10
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(num_iters):
            func(*args, **kwargs)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))

def pytorch_softmax(x):
    return torch.nn.functional.softmax(x, dim=-1)


if __name__ == "__main__":
    
    num_heads = 40
    row = 4096
    col = 4096
    dtype = torch.float32
    X = torch.randn((num_heads, row, col), device="cuda", dtype=dtype)
    
    # 1. Profile Time
    X_q = torch.randint(-128, 127, (num_heads, row, col), device="cuda", dtype=torch.int8)
    scale_x = torch.full((num_heads * row,), 0.1, device="cuda", dtype=dtype)  # per-row scale
    scale_y = 0.007874  # 1/127
    
    print("\nProfiling Custom CUDA Softmax for INT8:")
    profile_function(gemm_cutlass.func_softmax_lastdim_int8, X_q, scale_x, scale_y)
    
    
    # 2. Verify Correctness
    print("\nVerifying correctness for Softmax:")
    Y_torch = pytorch_softmax(X)
    
    Y_int8 = gemm_cutlass.func_softmax_lastdim_int8(X_q, scale_x, scale_y)
    assert Y_int8.dtype == torch.int8
    
    Y_int8_dequant = Y_int8.to(dtype) * scale_y
    
    if torch.allclose(Y_torch, Y_int8_dequant, atol=1.0):
        print("Custom int8 Softmax correct")
    else:
        print("===== [ERROR] Custom int8 Softmax incorrect =====")
        max_diff = torch.max(torch.abs(Y_torch - Y_int8_dequant))
        print(f"Max difference: {max_diff.item()}")