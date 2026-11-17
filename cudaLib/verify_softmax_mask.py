import os 
import torch
from torch import nn
import gemm_cutlass
from utils_quant import *


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        # Apply the mask to the input tensor
        x_masked = x.masked_fill(mask, -torch.inf)
        # Compute softmax along the specified dimension
        return torch.softmax(x_masked, dim=self.dim)

if __name__ == "__main__":
    
    d_type = torch.bfloat16
    d = 2048
    
    X = torch.randn((d, d), dtype=d_type, device="cuda")
    mask = torch.triu(torch.ones(d, d, device="cuda", dtype=torch.bool), diagonal=1)
    
    # 1. Torch softmax with mask
    masked_softmax = MaskedSoftmax(dim=-1)
    masked_softmax = torch.compile(masked_softmax)
    Y = masked_softmax(X, mask)
    
    torch_time = measure_time(masked_softmax, X, mask)
    print(f"Torch MaskedSoftmax time: {torch_time:.6f} ms")
    
    # 2. Custom MaskedSoftmax
    X_i8, scale_X = gemm_cutlass.func_quantize_i8(X)
    mask = torch.tril(torch.ones((d, d), dtype=torch.uint8, device="cuda"))
    Y_i8, scale_Y = gemm_cutlass.func_softmax_lastdim_int8_masking(X_i8, scale_X, mask)
    
    custom_time = measure_time(gemm_cutlass.func_softmax_lastdim_int8_masking, X_i8, scale_X, mask)
    print(f"Custom MaskedSoftmax time: {custom_time:.6f} ms\n")
    
    Y_deq = Y_i8.to(torch.float32) * scale_Y.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"Mean Squared Error (MSE): {mse}")
    
    print(f"Y (Torch Softmax): {Y[:5, :5]}")
    print(f"Y (Custom MaskedSoftmax): {Y_deq[:5, :5]}")