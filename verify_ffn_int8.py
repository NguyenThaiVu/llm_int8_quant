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
from utils_layer_int8 import *


class Custom_FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=cfg["seq_len"])
        self.fc2 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=cfg["seq_len"])
        self.fc3 = Custom_Linear(cfg["hidden_dim"], cfg["emb_dim"], max_seq_len=cfg["seq_len"])
        self.custom_silu = Custom_SiLU(max_length=cfg["seq_len"])
        self.custom_elementwise_mul = Custom_Element_Wise(max_length=cfg["seq_len"])
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            x_fc1, _ = self.fc1(x, scale_x)
            x_fc2, _ = self.fc2(x, scale_x)
            x_silu, _ = self.custom_silu(x_fc1, 1.0)
            x, _ = self.custom_elementwise_mul(x_silu, 1.0, x_fc2, 1.0)
            out, _ = self.fc3(x, 1.0)
            return out, 1.0
        else:
            x_fc1, scale_fc1 = self.fc1(x, scale_x)
            x_fc2, scale_fc2 = self.fc2(x, scale_x)
            x_silu, scale_silu = self.custom_silu(x_fc1, scale_fc1)
            x, scale_mul = self.custom_elementwise_mul(x_silu, scale_silu, x_fc2, scale_fc2)
            out, scale_out = self.fc3(x, scale_mul)
        
            return out, scale_out
        
    def finish_calibration(self):
        self.fc1.finish_calibration()
        self.fc2.finish_calibration()
        self.custom_silu.finish_calibration()
        self.custom_elementwise_mul.finish_calibration()
        self.fc3.finish_calibration()
        self.is_quantized = True


if __name__ == "__main__":
    cfg = {
        "seq_len": 1024,
        "emb_dim": 4096,
        "hidden_dim": 4096,
        "device": 'cuda',
        "dtype": torch.bfloat16
    }
    
    X = torch.randn((cfg["seq_len"], cfg["emb_dim"]), device=cfg["device"], dtype=cfg["dtype"])
    
    custom_ffn = Custom_FeedForward(cfg).to(device=cfg["device"], dtype=cfg["dtype"])
    
    # 1. Calibrate custom FFN with float input
    with torch.no_grad():
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

    
    