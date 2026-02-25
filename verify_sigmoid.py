"""
Docstring for verify_sigmoid
"""

import os 
import torch 
torch.set_printoptions(sci_mode=False)
from utils import *
import gemm_cutlass
from utils_transformer_int8 import MinMaxObserverPerLastDim
from utils_layer_int8 import Custom_Sigmoid
    
if __name__ == "__main__":
    
    seq_len = 1024
    emb_dim = 4096
    device = 'cuda'
    # dtype = torch.bfloat16
    dtype = torch.float32
    
    # 1. 2D input (seq_len, emb_dim)
    print("=" * 50)
    print(f"Testing 2D input: (seq_len={seq_len}, emb_dim={emb_dim})")
    
    X = torch.randn((seq_len, emb_dim), device=device, dtype=dtype)
    print(f"Sample values of X: {X[:5, :5]} \n")
    
    sigmoid_layer = Custom_Sigmoid(max_seq_len=seq_len).to(device)
    Y, _ = sigmoid_layer(X, 1.0)
    
    print(f"Shape of Y: {Y.shape}")
    print(f"Sample values of Y: {Y[:5, :5]} \n")
    
    # 2. Finish calibration and prepare for quantization
    with torch.no_grad():
        for _ in range(10):  # Run multiple iterations to gather statistics
            x_calib = torch.randn((seq_len, emb_dim), device=device, dtype=dtype) * 10
            sigmoid_layer(x_calib, 1.0)
    
    sigmoid_layer.finish_calibration()
    
    # Quantization
    X_q, scale_x = quantize_row_int8_symmetric_nd(X, scale_dtype=torch.float32)

    Y_q, scale_y = sigmoid_layer(X_q, scale_x) 
    
    Y_deq = Y_q * scale_y.unsqueeze(-1)
    print(f"Sample values of Y_deq: {Y_deq[:5, :5]} \n")
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse}")


    