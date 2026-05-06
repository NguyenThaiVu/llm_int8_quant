import os 
import torch
import torch.nn as nn
import gemm_cutlass
from utils_quant import *

class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        self.is_quantized = False
        
    def forward(self, x, scale_x, mask):
        if self.is_quantized == False: 
            assert x.dtype == torch.float32 or \
                    x.dtype == torch.bfloat16 or \
                    x.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            x = x.masked_fill(mask, -torch.inf)
            out = torch.softmax(x, dim=-1)
            return out, 1.0
        else:          
            x_int8 = x
            
            out_q, scale_out = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x_int8, scale_x, mask
            )
            
            return out_q, scale_out
    
    def finish_calibration(self):
        self.is_quantized = True  
        
if __name__ == "__main__":
    
    # seq_len = 3072
    # n_heads = 32
    dtype = torch.bfloat16
    for seq_len in [1024, 2048, 3072]:
        for n_heads in [16, 24, 32]:
            print(f"\nTesting Softmax with seq_len={seq_len}, n_heads={n_heads}")
            X = torch.randn(n_heads, seq_len, seq_len, dtype=dtype).cuda()
            softmax_layer = Custom_Softmax(num_heads=n_heads, max_seq_len=seq_len).cuda()
            
            # Baseline: PyTorch Softmax
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device='cuda'), diagonal=1)
            Y, _ = softmax_layer(X, None, mask)
            
            bf16_softmax_time = measure_time(softmax_layer, X, None, mask)
            print(f"PyTorch Softmax time: {bf16_softmax_time:.2f} ms")
            
            # Quantized Softmax
            softmax_layer.finish_calibration()
            X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
            
            mask = torch.tril(torch.ones((seq_len, seq_len),dtype=torch.uint8, device='cuda'))
            Y_int8, scale_out = softmax_layer(X_int8, scale_x, mask)
            Y_deq = Y_int8.float() * scale_out.unsqueeze(-1)
            Y_deq = Y_deq.to(dtype)
            
            int8_softmax_time = measure_time(softmax_layer, X_int8, scale_x, mask)
            print(f"Quantized Softmax time: {int8_softmax_time:.2f} ms")
            
            # Compare results
            max_diff = torch.max(torch.abs(Y - Y_deq))
            print(f"Max absolute difference: {max_diff.item():.6f}")    
            mse = torch.mean((Y - Y_deq) ** 2).item()
            print(f"Mean Squared Error: {mse:.6f}\n")
            
            # print(f"Sample output (bf16): {Y[0, :5, :5]}")
            # print(f"Sample output (dequantization): {Y_deq[0, :5, :5]}")
            