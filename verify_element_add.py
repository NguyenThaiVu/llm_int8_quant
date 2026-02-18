"""
In this script, 
we verify the correctness of the element-wise addition function.
"""
import os 
import torch 
import gemm_cutlass 
from utils import *
from utils_transformer_int8 import *

class Custom_Element_Add(torch.nn.Module):
    def __init__(self, max_length = 1024):
        super().__init__()
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(max_length)) 
        self.is_quantized = False
    def forward(self, a, scale_a, b, scale_b):
        if self.is_quantized:
            out_int8 = gemm_cutlass.func_element_add_int8(a, scale_a,\
                                b, scale_b, self.scale_out.to(torch.float32))
            return out_int8, self.scale_out
        else:
            out = a + b
            self.out_observer(out)
        return out, 1.0

    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        
if __name__ == "__main__":
    seq_len = 4096
    emd_dim = 8192
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    A = torch.randn(seq_len, emd_dim, device=device, dtype=dtype)
    B = torch.randn(seq_len, emd_dim, device=device, dtype=dtype)
    
    element_add_layer = Custom_Element_Add(max_length=seq_len).to(device)
    
    # 1. Calibration
    out, _ = element_add_layer(A, 1.0, B, 1.0)
    print(f"Output: {out[:5, :5]}")
    
    # 2. Finish calibration
    element_add_layer.finish_calibration()
    print(f"output scale: {element_add_layer.scale_out.shape}")
    
    # 3. Quantized inference
    A_int8, scale_A = quantize_row_int8_symmetric_nd(A, scale_dtype=torch.float32)
    B_int8, scale_B = quantize_row_int8_symmetric_nd(B, scale_dtype=torch.float32)
    out_int8, scale_out = element_add_layer(A_int8, scale_A, B_int8, scale_B)
    print(f"Quantized output: {out_int8[:5, :5]}")
    
    out_deq = out_int8.float() * scale_out.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(out - out_deq)).item()
    print(f"Max difference: {max_diff}")
    mse = torch.mean((out - out_deq) ** 2).item()
    print(f"MSE: {mse}")