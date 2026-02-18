"""
In this script, I profile the normalization layer
"""
import os 
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity


def profile_function(func, *args, **kwargs):
        num_iters = 10
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(num_iters):
                func(*args, **kwargs)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        

class LayerNorm_1D(nn.Module):
    def __init__(self, gamma, beta, eps=1e-6):
        super(LayerNorm_1D, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, x):
        mean = x.mean()
        std = x.std()
        norm_value = (x - mean) / (std + self.eps)
        return self.gamma * norm_value + self.beta
    

class LayerNorm(nn.Module):
    def __init__(self, gamma, beta, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_value = (x - mean) / (std + self.eps)
        return self.gamma * norm_value + self.beta

    
if __name__ == "__main__":
    
    seq_length = 1024 * 8
    dim = 1024 * 5 
    dtype = torch.bfloat16
    device = 'cuda'
    
    X = torch.randn(seq_length, dim, dtype=dtype, device=device)
    layer_norm = LayerNorm(gamma=1.0, beta=1.0)
    Y = layer_norm(X)
    
    print(f"Shape of output: {Y.shape} - dtype: {Y.dtype}")
    
    # Profile LayerNorm
    print("\nProfiling LayerNorm:")
    profile_function(layer_norm, X)
    
        
    