import os 
import time
import torch
import torch.nn.functional as F
from utils_quant import quantize_row_int8_symmetric_nd
from utils_power import PowerSampler, measure_power
import gemm_cutlass


if __name__ == "__main__":

    # =========================== 2D Input ====================================
    seq_len = 2048
    embed_dim = 1024 * 8
    print(f"\nTesting SiLU with 2D input, seq_len={seq_len}, embed_dim={embed_dim}")
    
    list_embed_dim = [4096, 8192, 16384]
    d_type = torch.bfloat16
    
    for embed_dim in list_embed_dim:
        print(f"\nTesting SiLU with seq_len={seq_len}, embed_dim={embed_dim}")
        
        # 1. PyTorch SiLU
        print(f"\nTorch SiLU")
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        for i in range(10):
            y = F.silu(x)
        torch.cuda.synchronize()
            
        measure_power(F.silu, x)
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 2. BF16 SiLU
        print(f"\nBF16 SiLU")
        for i in range(10):
            y_bf16 = gemm_cutlass.func_silu_bf16(x)
        
        measure_power(gemm_cutlass.func_silu_bf16, x)
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 3. INT8 SiLU
        print(f"\nINT8 SiLU")
        x_i8, scale_x = quantize_row_int8_symmetric_nd(x)
        
        for i in range(10):
            y_i8, scale_y = gemm_cutlass.func_silu_int8(x_i8, scale_x)
        measure_power(gemm_cutlass.func_silu_int8, x_i8, scale_x)
        