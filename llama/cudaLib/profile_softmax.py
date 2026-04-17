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
            x_masked = x.masked_fill(mask, float('-inf'))
            out = torch.softmax(x_masked, dim=-1)
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
    
    num_heads = 24
    seq_len = 2048
    dtype = torch.bfloat16
    
    X = torch.randn(num_heads, seq_len, seq_len, dtype=dtype).cuda()
    softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=seq_len).cuda().eval()
    
    # ========================================================================
    # Float softmax
    mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool), diagonal=1)
    
    out, _ = softmax_layer(X, 1.0, mask)
    
    float_softmax_time = measure_time(softmax_layer, X, 1.0, mask)
    print(f"Float softmax time: {float_softmax_time:.2f} ms")
    
    # ========================================================================
    # Quantization 
    softmax_layer.finish_calibration()
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=X.device))
    
    out_q, scale_out = softmax_layer(X_int8, scale_x, mask)
    out_q_dequant = out_q.to(torch.float32) * scale_out.unsqueeze(-1)
    
    int8_softmax_time = measure_time(softmax_layer, X_int8, scale_x, mask)
    print(f"Int8 softmax time: {int8_softmax_time:.2f} ms\n")
    
    max_diff = torch.max(torch.abs(out - out_q_dequant))
    print(f"Max absolute difference: {max_diff.item():.6f}")
    
    mse = torch.mean((out - out_q_dequant) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")
    
    print(f"Sample output (float32): {out[0, :5, :5]}")
    print(f"Sample output (quantized dequantized): {out_q_dequant[0, :5, :5]}")