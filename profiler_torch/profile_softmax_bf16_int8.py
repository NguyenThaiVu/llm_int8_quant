"""
In this script, we profile the latency between
- BF16 softmax
- INT8 softmax
"""

import os 
import time
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    

    # ================== 2D input ==================
    dtype = torch.bfloat16
    seq_len = 2048
    embed_dim = 8192
    
    print(f"\nSoftmax with seq_len={seq_len}, embed_dim={embed_dim}")
    X = torch.randn((seq_len, embed_dim), dtype=dtype).cuda()
    
    # BF16 softmax
    Y_bf16 = gemm_cutlass.func_bf16_softmax(X)
    
    # INT8 softmax
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    Y_int8, scale_Y = gemm_cutlass.func_softmax_int8(X_int8, scale_X)
    
    Y_deq = Y_int8.float() * scale_Y.unsqueeze(-1)
    Y_deq = Y_deq.to(dtype)
    
    # Verify correctness
    max_diff = torch.max(torch.abs(Y_bf16 - Y_deq))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((Y_bf16 - Y_deq) ** 2).item()
    print(f"MSE: {mse}")