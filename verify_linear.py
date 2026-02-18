"""
In this script, we verify the correctness of
custom linear int8
"""

import os 
import torch 
import torch.nn as nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import *

class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        
        self.register_buffer('scale_w', torch.tensor(1.0))
        self.register_buffer('scale_y', torch.ones(out_features))

        self.out_observer = MinMaxObserverPerLastDim()
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode 
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            requant_scale = scale_x * self.scale_w / self.scale_y
            requant_scale = requant_scale.to(torch.float32)
            
            if x.dim() == 3:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(
                    x, self.weight_q, requant_scale
                )
            elif x.dim() == 2:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale(
                    x, self.weight_q, requant_scale
                )
            else:
                raise ValueError("Input must be 2D or 3D tensor")
            return out_q, self.scale_y
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_tensor(self.weight)
        
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  

if __name__ == "__main__":
    
    seq_len = 1024 * 16
    emb_dim = 4096
    
    device = 'cuda'
    d_type = torch.bfloat16
    
    # ==========================
    # 1. 2D input 
    X = torch.randn((seq_len, emb_dim), dtype=d_type, device=device)
    linear = Custom_Linear(emb_dim, emb_dim).to(device).to(d_type)
    
    # Step 1: Calibration
    with torch.no_grad():
        out_float, _ = linear(X, scale_x=1.0)
        linear.finish_calibration()
        
    # Step 2: Quantized inference
    with torch.no_grad():
        X_q, scale_x = quantize_row_int8_symmetric_nd(X)
        out_q, scale_y = linear(X_q, scale_x=scale_x)
    
    # Dequantize output for comparison
    out_q_dequant = out_q.float() * scale_y.unsqueeze(-1)
    
    max_diff = (out_float - out_q_dequant).abs().max().item()
    print(f"Max diff: {max_diff:.6f}")
    mse = ((out_float - out_q_dequant) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    print(f"Sample output (float): {out_float[:5, :5]}")
    print(f"Sample output (dequantized): {out_q_dequant[:5, :5]} \n")
    
    # ==========================
    # 2. 3D input (batched)
    batch_size = 4
    X_batched = torch.randn((batch_size, seq_len, emb_dim), dtype=d_type, device=device)
    linear_batched = Custom_Linear(emb_dim, emb_dim).to(device).to(d_type)
    
    # Step 1: Calibration
    with torch.no_grad():
        out_float_batched, _ = linear_batched(X_batched, scale_x=1.0)
        linear_batched.finish_calibration()
        
    # Step 2: Quantized inference
    with torch.no_grad():
        X_batched_q, scale_x_batched = quantize_row_int8_symmetric_nd(X_batched)
        out_q_batched, scale_y_batched = linear_batched(X_batched_q, scale_x=scale_x_batched)
        
    # Dequantize output for comparison
    out_q_batched_dequant = out_q_batched.float() * scale_y_batched.unsqueeze(-1)
    max_diff_batched = (out_float_batched - out_q_batched_dequant).abs().max().item()
    print(f"Max diff (batched): {max_diff_batched:.6f}")
    mse_batched = ((out_float_batched - out_q_batched_dequant) ** 2).mean().item()
    print(f"MSE (batched): {mse_batched:.6f}")
    
    print(f"Sample output (float, batched): {out_float_batched[0, :5, :5]}")
    print(f"Sample output (dequantized, batched): {out_q_batched_dequant[0, :5, :5]}")