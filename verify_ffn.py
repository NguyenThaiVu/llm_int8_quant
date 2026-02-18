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
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
    

if __name__ == "__main__":
    cfg = {
        "emb_dim": 4096,
        "hidden_dim": 4096,
        "dtype": torch.float16,
    }
    seq_len = 1024

    # Create random input tensor
    x = torch.randn(seq_len, cfg["emb_dim"], dtype=cfg["dtype"]).cuda()

    # Initialize the FFN model and move it to GPU
    ffn = FeedForward(cfg).cuda()

    # Run the FFN model
    output = ffn(x)

    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)
    print("Sample output values:", output[:5, :5])