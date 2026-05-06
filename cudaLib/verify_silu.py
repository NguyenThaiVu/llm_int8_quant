import os 
import torch
import torch.nn as nn
import gemm_cutlass
from utils_quant import *

class Custom_Silu(nn.Module):
    def __init__(self, emb_dim, dtype=torch.bfloat16):
        super().__init__()
        self.emb_dim = emb_dim
        self.dtype = dtype

        self.is_quantized = False
        self.is_smooth_scale = False
        
        self.smooth_alpha = torch.ones(emb_dim, dtype=torch.float32).cuda()  

    def forward(self, x1, scale_x1, x2, scale_x2):
        if not self.is_quantized:
            x = torch.nn.functional.silu(x1) * x2
            return x
        else:
            assert x1.dtype == torch.int8 and x2.dtype == torch.int8,\
                "Expected int8 inputs in quantized mode"

            x_int8, x_scale = gemm_cutlass.func_silu_mul_int8(\
                x1, scale_x1,\
                x2, scale_x2, self.smooth_alpha)
            
            return x_int8, x_scale

    def finish_calibration(self):
        self.is_quantized = True

    def enable_smooth_scale(self, smooth_alpha_value):
        """
        This function apply smooth scaling to quantization (for input activations) 
        Note:
        - Normal quantization: Y_int8 = Y / scale_Y
        - Smooth quantization: Y_int8 = (Y / smooth_scale) / scale_Y
        """
        assert smooth_alpha_value.shape == (self.emb_dim,)
        self.is_smooth_scale = True
        
        smooth_alpha_value = smooth_alpha_value.to(torch.float32)
        self.smooth_alpha = smooth_alpha_value
        
if __name__ == "__main__":
    
    # seq_len = 2048
    # emb_dim = 8192
    list_seq_len = [1024, 2048, 3072]
    list_emb_dim = [4096, 8192]
    dtype = torch.bfloat16
    
    for seq_len in list_seq_len:
        for emb_dim in list_emb_dim:
            print(f"\nTesting SiLU with seq_len={seq_len}, emb_dim={emb_dim}")
            X1 = torch.randn(seq_len, emb_dim, dtype=dtype).cuda()
            X2 = torch.randn(seq_len, emb_dim, dtype=dtype).cuda()
            
            # Baseline: PyTorch SiLU
            silu_layer = Custom_Silu(emb_dim, dtype).cuda()
            Y = silu_layer(X1, None, X2, None)
            bf16_silu_time = measure_time(silu_layer, X1, None, X2, None)
            print(f"PyTorch SiLU time: {bf16_silu_time:.2f} ms")
            
            # Quantized SiLU
            silu_layer.finish_calibration()  
            X1_int8, scale_X1 = quantize_row_int8_symmetric_nd(X1)
            X2_int8, scale_X2 = quantize_row_int8_symmetric_nd(X2)
            
            Y_int8, scale_Y = silu_layer(X1_int8, scale_X1, X2_int8, scale_X2)
            int8_silu_time = measure_time(silu_layer, X1_int8, scale_X1, X2_int8, scale_X2)
            print(f"Quantized SiLU time: {int8_silu_time:.2f} ms")
            
            Y_deq = Y_int8.float() * scale_Y.unsqueeze(-1)
            Y_deq = Y_deq.to(dtype)
            
            # Compare results
            max_diff = torch.max(torch.abs(Y - Y_deq))
            print(f"Max absolute difference: {max_diff.item():.6f}")
            mse = torch.mean((Y - Y_deq) ** 2).item()
            print(f"Mean Squared Error: {mse:.6f}\n")