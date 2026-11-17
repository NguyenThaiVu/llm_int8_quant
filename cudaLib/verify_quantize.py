import os 
import torch 
from utils_quant import *
import gemm_cutlass


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.bfloat16
    L = 2048
    d = 1024 * 8
    
    X = torch.randn((L, d), device=device, dtype=dtype)
    
    # 1. Torch Quantization
    
    torch_quant_time = measure_time(quantize_i8_compiled, X)
    
    print(f"PyTorch quantization time: {torch_quant_time:.6f} ms")
    
    # 2. Our Kernel Quantization
    our_quant_time = measure_time(gemm_cutlass.func_quantize_i8, X)
    print(f"Our Kernel quantization time: {our_quant_time:.6f} ms")