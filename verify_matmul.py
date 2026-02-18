"""
In this script, I will verify the correctness of matrix multiplication
- bfloat16
- W8A8O8: Weight 8-bit, Activation 8-bit, Output 8-bit
"""

import os 
import numpy as np
import torch
import gemm_cutlass
from utils import *

device = 'cuda'
d_type = torch.bfloat16
tolerance = 5.0

M = 2048
N = 4096
K = 8192

# Read list X, W from file 
MODEL_HUD_FOLDER = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"
data = torch.load(f"{MODEL_HUD_FOLDER}/debug_data.pt", map_location="cpu")
list_X = data['activations']
list_W = data['weights']
list_X = [x.detach().cpu().to(torch.float16).numpy() for x in list_X]
list_W = [w.detach().cpu().to(torch.float16).numpy() for w in list_W]

# Pick an index to test
for _ in range(5):
    idx = np.random.randint(0, len(list_X))
    X_torch = torch.from_numpy(list_X[idx]).to(d_type) 
    W_torch = torch.from_numpy(list_W[idx]).to(d_type)
    X = X_torch.to(device)
    W = W_torch.to(device)
    print(f"Shape X: {X.shape}")
    print(f"Load X: {X[:5, :5]}\n")
    
    print(f"Shape W: {W.shape}")
    print(f"Load W: {W[:5, :5]}\n")

    W_t = W.t().contiguous()

    # 1. True matmul
    Y_true = torch.matmul(X, W_t)

    # 2. Quantization W8A8O8 
    X_q, X_scale = quantize_tensor(X)
    W_q, W_scale = quantize_tensor(W)

    # Assume having output scale via calibration
    _, Y_true_scale = quantize_tensor(Y_true)

    matmul_scale = (X_scale * W_scale) / Y_true_scale
    Y_q = gemm_cutlass.func_int8_matmul_output_int8(
        X_q, W_q, matmul_scale.to(d_type)
    )

    # Verify correctness
    Y_deq = Y_q.to(d_type) * Y_true_scale

    if torch.allclose(Y_deq.float(), Y_true.float(), atol=tolerance):
        print("W8A8O8 MatMul test passed!")
    else:
        print("===== [ERROR] W8A8O8 MatMul test failed ======")

    max_diff = torch.max(torch.abs(Y_deq.float() - Y_true.float()))
    print(f"Max difference: {max_diff}")
    mse = torch.mean((Y_true.float() - Y_deq.float()) ** 2)
    print(f"MSE: {mse:.4f}")
    print()
    print(f"Y_true: {Y_true[:5, :5]} \n")
    print(f"Y_deq: {Y_deq[:5, :5]} \n")
    
    print("\n\n\n")
