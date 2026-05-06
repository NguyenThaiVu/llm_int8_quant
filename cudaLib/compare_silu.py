"""
In this script, we compare the latency between SiLU with implicit and explicit
shared memory.
"""

import os 
import torch
import gemm_cutlass
from utils_quant import *

if __name__ == "__main__":
    
    seq_len = 2048
    emb_dim = 8192
    dtype = torch.bfloat16
    
    X1 = torch.randn(seq_len, emb_dim, dtype=dtype).cuda()
    X2 = torch.randn(seq_len, emb_dim, dtype=dtype).cuda()
    X1_int8, scale_X1 = quantize_row_int8_symmetric_nd(X1)
    X2_int8, scale_X2 = quantize_row_int8_symmetric_nd(X2)
    
    dump_smooth_scale = torch.ones(emb_dim, dtype=torch.float32).cuda()

    use_warp_reduction = False
    silu_implicit = gemm_cutlass.func_silu_mul_int8(X1_int8, scale_X1,\
                                                    X2_int8, scale_X2,\
                                                    dump_smooth_scale, use_warp_reduction)
    
    time_implicit_shared_mem = measure_time(gemm_cutlass.func_silu_mul_int8,\
                                            X1_int8, scale_X1,\
                                            X2_int8, scale_X2,\
                                            dump_smooth_scale, use_warp_reduction)
    print(f"Latency of SiLU with implicit shared memory: {time_implicit_shared_mem:.3f} ms")
    
    use_warp_reduction = True
    silu_explicit = gemm_cutlass.func_silu_mul_int8(X1_int8, scale_X1,\
                                                    X2_int8, scale_X2,\
                                                    dump_smooth_scale, use_warp_reduction)
    
    time_explicit_shared_mem = measure_time(gemm_cutlass.func_silu_mul_int8,\
                                            X1_int8, scale_X1,\
                                            X2_int8, scale_X2,\
                                            dump_smooth_scale, use_warp_reduction)
    print(f"Latency of SiLU with explicit shared memory: {time_explicit_shared_mem:.3f} ms")

    
