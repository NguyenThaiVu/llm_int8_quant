"""
In this script, we profile the performance of Multi-Head Attention (MHA)
"""
import os
import torch
from torch import nn
from utils import *
import gemm_cutlass
from torch.profiler import profile, record_function, ProfilerActivity

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MultiHeadAttention, self).__init__()
        self.linear_q = nn.Linear(input_dims, hidden_dims)
        self.linear_k = nn.Linear(input_dims, hidden_dims)
        self.linear_v = nn.Linear(input_dims, hidden_dims)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dims ** 0.5)
        attn_weights = self.softmax(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        output = self.linear_out(attn_output)
        return output
    
class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Custom_Linear, self).__init__()
        self.weight_q = torch.randint(-128, 128, (out_features, in_features),\
                                        device='cuda', dtype=torch.int8)
        
        self.weight_scale = 1.0  # Dummy scale for weight quantization
        self.out_scale = 1.0 # Dummy scale for output quantization

    def forward(self, x_q, scale_x):
        Y_q = gemm_cutlass.func_int8_matmul_output_int8(
            x_q, 
            self.weight_q, 
            scale_x * self.weight_scale / self.out_scale
        )
        return Y_q, self.out_scale
    
    
class Custom_Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Custom_Softmax, self).__init__()
        self.dim = dim
        self.out_scale = 1.0  # Dummy scale for output quantization

    def forward(self, x_q, scale_x):
        Y_q = gemm_cutlass.func_softmax_lastdim_int8(x_q, scale_x, self.out_scale)
        return Y_q, self.out_scale
    
    
class MultiHeadAttentionQuantized(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(MultiHeadAttentionQuantized, self).__init__()
        self.linear_q = Custom_Linear(input_dims, hidden_dims)
        self.linear_k = Custom_Linear(input_dims, hidden_dims)
        self.linear_v = Custom_Linear(input_dims, hidden_dims)
        self.linear_out = Custom_Linear(hidden_dims, output_dims)
        self.softmax = Custom_Softmax(dim=-1)
        
    def forward(self, x_q, scale_x):
        
        # 1. Linear Q, K, V
        q_q, scale_q = self.linear_q(x_q, scale_x)
        k_q, scale_k = self.linear_k(x_q, scale_x)
        v_q, scale_v = self.linear_v(x_q, scale_x)
        
        # 2. Attention Weights
        # scale_qk = scale_q * scale_k / (hidden_dims ** 0.5)
        scale_qk = scale_q * scale_k

        attn_weights_q = gemm_cutlass.func_int8_matmul_output_int8(q_q, k_q, scale_qk)
        
        attn_weights_q, scale_attn = self.softmax(attn_weights_q, 
                                torch.full((attn_weights_q.size(0),), scale_qk, device=attn_weights_q.device))
        
        # 3. Attention Output
        attn_output_q = gemm_cutlass.func_int8_matmul_output_int8(
            attn_weights_q, v_q.transpose(-2, -1),
            scale_attn * scale_v
        )
        
        output_q, scale_out = self.linear_out(attn_output_q, scale_attn * scale_v)
        
        x_deq = output_q.to(torch.bfloat16) * scale_out
        return x_deq
    
def profiling_model(model, *inputs):
    # warm-up
    for _ in range(5):
        _ = model(*inputs)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            _ = model(*inputs)
        
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
    
    
if __name__ == "__main__":
    
    seq_len = 8192
    input_dims = hidden_dims = output_dims = 8192
    dtype = torch.bfloat16
 
    # 1. Original MHA Model
    model = MultiHeadAttention(input_dims, hidden_dims, output_dims).cuda().to(dtype)
    
    X = torch.randn((seq_len, input_dims), device='cuda', dtype=dtype) * 10
    
    Y = model(X)
    
    model_time = measure_time(model, X, repeat=100)
    print(f"MHA Model time: {model_time:.3f} ms")
    
    # 2. Quantized Model
    model_q = MultiHeadAttentionQuantized(input_dims, hidden_dims, output_dims).cuda()
    X_q, scale_x = quantize_tensor(X)
    Y_deq = model_q(X_q, scale_x)
    
    model_q_time = measure_time(model_q, X_q, scale_x, repeat=100)
    print(f"Quantized MHA Model time: {model_q_time:.3f} ms")
    
    assert (Y.shape == Y_deq.shape) and (Y.dtype == Y_deq.dtype == torch.bfloat16)
    
    if torch.allclose(Y, Y_deq, rtol=2.0, atol=2.0):
        print("Quantized MHA model correct.")
    else:
        print("===== [ERROR] Quantized MHA model incorrect. =====")
        max_diff = (Y - Y_deq).abs().max()
        print(f"Max difference: {max_diff.item()}")
        
    print("Profiling MHA Model...")
    profiling_model(model, X)
    
    print("Profiling Quantized MHA Model...")
    profiling_model(model_q, X_q, scale_x)
    
    