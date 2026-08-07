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
    # embed_dim = 1024 * 8
    
    list_embed_dim = [4096, 8192, 16384]
    d_type = torch.bfloat16
    
    for embed_dim in list_embed_dim:
        x = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
        print(f"\nTesting softmax with seq_len={seq_len}, embed_dim={embed_dim}")
        for i in range(10):
            y = F.softmax(x, dim=-1)
        
        # 1. PyTorch softmax
        # print(f"\nTorch softmax")
        # for i in range(10):
            # y = F.softmax(x, dim=-1)
        # torch.cuda.synchronize()
            
        # measure_power(F.softmax, x, -1)
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 2. BF16 Softmax
        print(f"\nBF16 Softmax")
        for i in range(10):
            y_bf16 = gemm_cutlass.func_bf16_softmax(x)
        measure_power(gemm_cutlass.func_bf16_softmax, x)
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 3. INT8 Softmax Naive
        print(f"\nINT8 Softmax Naive")
        x_i8, scale_x = quantize_row_int8_symmetric_nd(x)
        
        try:
            for i in range(10):
                y_i8, scale_y = gemm_cutlass.func_softmax_int8_shared(x_i8, scale_x)
            measure_power(gemm_cutlass.func_softmax_int8_shared, x_i8, scale_x)
        except Exception as e:
            print(f"INT8 Softmax (naive) OOM.")
        
        time.sleep(1)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 4. INT8 Softmax Hierarchical Reduction
        print(f"\nINT8 Softmax Hierarchical Reduction")
        for i in range(10):
            y_i8_hier, scale_y_hier = gemm_cutlass.func_softmax_int8(x_i8, scale_x)
        measure_power(gemm_cutlass.func_softmax_int8, x_i8, scale_x)
        
        