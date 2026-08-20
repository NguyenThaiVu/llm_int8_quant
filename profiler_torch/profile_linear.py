import os 
import torch 
from torch import nn
from torch.nn import functional as F
import gemm_cutlass
from utils_quant import *

MAX_SEQ_LEN = 2048

class Custom_Linear(nn.Module):
    """
    Linear layer with two execution modes:
    1. Calibration mode
        - Input: bf16
        - Weight: bf16
        - Output: bf16
        - Behavior: computes `x @ weight.T` and updates output observer
    
    2. Quantization mode
        - Input: int8
        - Weight: int8
        - Output: int8
        - Behavior: computes quantized matmul with per-row scaling
                    and returns quantized output and its scale
    
    This layer has: per-row scale for activation
                    per-tensor scale for weight. 
                    The output scale is per-row
                    
    Input shapes: (M, K) or (B, M, K)
    Weight shapes: (N, K)
    Output shapes: (M, N) or (B, M, N)
    """
    def __init__(self, in_features, out_features, 
                 max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16):
        super(Custom_Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 
                                               dtype=dtype))
        
        # Weight quantization
        self.weight_q = torch.empty(out_features, in_features, dtype=torch.int8)
        self.scale_w = torch.ones((1), dtype=torch.float32, device='cuda')
        
        # Output scale quantization
        self.scale_y = torch.ones((out_features), dtype=torch.float32, device='cuda')  
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  
            out = torch.matmul(x, self.weight.t())  
            return out, 1.0
        else:
            assert x.dtype == torch.int8,\
                "Expected int8 input in quantization"
            
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len]
            
            row_scale = scale_x / scale_y_value  
            col_scale = self.scale_w.expand(self.out_features)
            
            out_q = gemm_cutlass.func_w8a8o8_matmul_fusion(x, self.weight_q,\
                row_scale, col_scale)

            return out_q, scale_y_value
        
    def finish_calibration(self):
        weight_q, scale_w = quantize_tensor(self.weight)
        self.weight_q = weight_q
        self.scale_w = scale_w
        
        self.is_quantized = True  
        del self.weight
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    
    d_model = 8192
    seq_len = 2048
    d_type = torch.bfloat16
    
    # Create random input and weight
    X = torch.randn(seq_len, d_model, dtype=d_type).cuda()
    
    linear_layer = Custom_Linear(d_model, d_model, max_seq_len=seq_len, dtype=d_type).cuda().to(d_type)
    
    # Measure bf16 execution time
    bf16_time = measure_time(linear_layer, X, 1.0)
    print(f"Linear time (bf16): {bf16_time:.2f} ms")
    
    # Quantize the linear layer
    linear_layer.finish_calibration()
    
    X_int8, scale_x = quantize_row_int8_symmetric_nd(X)
    
    int8_time = measure_time(linear_layer, X_int8, scale_x)
    print(f"Linear time (int8): {int8_time:.2f} ms")