"""
In this script, we verify the correctness of
Feed-Forward Network (FFN) with SiLU activation.
"""
import os 
import numpy as np
import torch
from torch import nn
import gemm_cutlass
from utils import *
from utils_transformer_int8 import *
from utils_layer_int8 import Custom_FeedForward

if __name__ == "__main__":
    cfg = {
        "seq_len": 1024,
        "emb_dim": 4096,
        "hidden_dim": 8192,
        "device": 'cuda',
        "dtype": torch.bfloat16
    }
    
    X = torch.randn((cfg["seq_len"], cfg["emb_dim"]), device=cfg["device"], dtype=cfg["dtype"])
    custom_ffn = Custom_FeedForward(cfg).to(device=cfg["device"], dtype=cfg["dtype"])
    
    # 1. Calibrate custom FFN with float input
    for _ in range(10):  # Run multiple iterations to stabilize calibration
        x_calib = torch.randn((cfg["seq_len"], cfg["emb_dim"]), device=cfg["device"], dtype=cfg["dtype"])
        out, _ = custom_ffn(x_calib, 1.0)
    
    out, _ = custom_ffn(X, 1.0)
    print(f"Calibration done. Output shape: {out.shape}")
    print(f"Sample values of out: {out[:5, :5]}")
    
    # 2. Finish calibration to prepare for quantized inference
    custom_ffn.finish_calibration()
    
    # 3. Run quantized inference
    X_int8, scale_X = quantize_row_int8_symmetric_nd(X)
    with torch.no_grad():
        out_quantized, scale_out_quantized = custom_ffn(X_int8, scale_X)
    
    out_deq = out_quantized.float() * scale_out_quantized.unsqueeze(-1)
    print(f"Quantized inference done. Output shape: {out_deq.shape}")
    print(f"Sample values of dequantized output: {out_deq[:5, :5]}")
    
    # Measure error
    max_diff = torch.max(torch.abs(out - out_deq))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((out - out_deq) ** 2)
    print(f"MSE: {mse.item()}")

    
    