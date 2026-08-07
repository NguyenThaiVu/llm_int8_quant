import os 
import time
import torch
import torch.nn.functional as F
from utils_quant import *
from utils_power import PowerSampler, measure_power
import gemm_cutlass


if __name__ == "__main__":

    # =========================== 2D Input ====================================
    seq_len = 2048
    # embed_dim = 8192
    list_embed_dim = [4096, 8192, 16384]
    d_type = torch.bfloat16
    
    for embed_dim in list_embed_dim:
        print(f"\nTesting RMSNorm with seq_len={seq_len}, embed_dim={embed_dim}")
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
        
        # 1. PyTorch RMSNorm
        # print(f"\nPyTorch RMSNorm")
        for i in range(10):
            y = F.rms_norm(x, (embed_dim,), gamma, eps=1e-6)
        # torch.cuda.synchronize()
            
        # measure_power(F.rms_norm, x, (embed_dim,), gamma, 1e-6)
        
        time.sleep(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 2. BF16 RMSNorm
        print(f"\nBF16 RMSNorm")
        for i in range(10):
            y_bf16 = gemm_cutlass.func_rmsnorm_bf16(x, gamma, 1e-6)
        measure_power(gemm_cutlass.func_rmsnorm_bf16, x, gamma, 1e-6)
        
        time.sleep(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 3. INT8 RMSNorm (Naive implementation)
        print(f"\nINT8 RMSNorm (Naive implementation)")
        x_i8, scale_x = quantize_row_int8_symmetric_nd(x)
        
        try:
            for i in range(10):
                y_i8, scale_y = gemm_cutlass.func_rmsnorm_naive_int8(x_i8, scale_x, gamma, 1e-6)
            measure_power(gemm_cutlass.func_rmsnorm_naive_int8, x_i8, scale_x, gamma, 1e-6)
        except RuntimeError as e:
            print(f"Error INT8 RMSNorm (Naive), seq_len={seq_len}, embed_dim={embed_dim}")
        time.sleep(1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 4. INT8 RMSNorm (Hierarchical implementation)
        print(f"\nINT8 RMSNorm (Hierarchical implementation)")
        for i in range(10):
            y_i8_hier, scale_y_hier = gemm_cutlass.func_rmsnorm_int8(x_i8, scale_x, gamma, 1e-6)
        measure_power(gemm_cutlass.func_rmsnorm_int8, x_i8, scale_x, gamma, 1e-6)
