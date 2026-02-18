"""
In this script, we profile the correctness of Transformer block in INT8.
Here, we use Group Query Attention (GQA).
"""

import os 
import torch
import torch.nn as nn
import gemm_cutlass

from utils import *
from utils_transformer_int8 import *
from utils_layer_int8 import *

    
class Custom_GroupQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,\
        qk_norm=True, max_seq_len=1024, dtype=torch.bfloat16):
        
        super(Custom_GroupQueryAttention, self).__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.qk_norm = qk_norm
        
        self.linear_q = Custom_Linear(d_in, self.d_out)
        self.linear_k = Custom_Linear(d_in, num_kv_groups * head_dim)
        self.linear_v = Custom_Linear(d_in, num_kv_groups * head_dim)
        self.out_proj = Custom_Linear(self.d_out, d_in)
        
        self.query_rope_layer = Custom_RoPE(head_dim, max_seq_len=max_seq_len)
        self.key_rope_layer = Custom_RoPE(head_dim, max_seq_len=max_seq_len)
        
        self.softmax = Custom_Softmax(num_heads, max_seq_len, dim=-1)
        
        if qk_norm:
            self.q_norm = Custom_RMSNorm(num_heads, max_seq_len=max_seq_len, dim=head_dim, eps=1e-6)
            self.k_norm = Custom_RMSNorm(num_kv_groups, max_seq_len=max_seq_len, dim=head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
            
        self.q_reshape_head_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_q_reshape_head', torch.ones(num_heads * max_seq_len))
        
        self.k_reshape_head_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_k_reshape_head', torch.ones(num_kv_groups * max_seq_len))
        
        self.v_reshape_head_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_v_reshape_head', torch.ones(num_kv_groups * max_seq_len))
        
        self.qk_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_qk', torch.ones(num_heads * max_seq_len))
        
        self.atten_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_attn', torch.ones(num_heads * max_seq_len))
        
        self.out_proj_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out_proj', torch.ones(max_seq_len))
        
        self.is_quantized = False
        
    def forward(self, x, scale_x, mask, cos, scale_cos, sin, scale_sin):
        if x.dim() == 3:
            b, num_tokens, _ = x.shape
        elif x.dim() == 2:
            num_tokens, _ = x.shape
        else:
            raise ValueError("Input must be 2D or 3D tensor")

        q, scale_q = self.linear_q(x, scale_x)
        k, scale_k = self.linear_k(x, scale_x)
        v, scale_v = self.linear_v(x, scale_x)
        
        q = q.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
        self.q_reshape_head_observer(q)
        
        k = k.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
        self.k_reshape_head_observer(k)
        
        v = v.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
        self.v_reshape_head_observer(v)
        
        if not self.is_quantized:
            
            if self.qk_norm:
                q, _ = self.q_norm(q, 1.0)
                k, _ = self.k_norm(k, 1.0)
                
            # 1. Apply RoPE to Q and K
            q_rope, _ = self.query_rope_layer(q, 1.0, cos, 1.0, sin, 1.0)
            k_rope, _ = self.key_rope_layer(k, 1.0, cos, 1.0, sin, 1.0)
            
            # Expand K and V to match number of heads
            k_rope = k_rope.repeat_interleave(self.group_size, dim=0)
            v = v.repeat_interleave(self.group_size, dim=0)
            
            # 2. Compute attention weights 
            qk_weights = torch.matmul(q_rope, k_rope.transpose(-2, -1))
            self.qk_observer(qk_weights)
            
            qk_weights = qk_weights / (self.head_dim ** 0.5)
            
            qk_weights = qk_weights.masked_fill(mask.unsqueeze(0).to(torch.bool), float('-inf'))
            softmax_weight, _ = self.softmax(qk_weights, 1.0)
            
            # 3. Compute attention output
            attn_output = torch.matmul(softmax_weight, v)
            self.atten_observer(attn_output)
            
            # 4. Final output projection
            attn_output = attn_output.transpose(0, 1).contiguous().view(num_tokens, self.d_out)
            self.out_proj_observer(attn_output)
            
            output, _ = self.out_proj(attn_output, 1.0)
            return output, 1.0
        else: 
            # ===== Quantized path =====
        
            if self.qk_norm:
                q, scale_query = self.q_norm(q, self.scale_q_reshape_head)
                k, scale_key = self.k_norm(k, self.scale_k_reshape_head)
            else:  # if not apply RMSNorm, we only pass the scale
                scale_query = self.scale_q_reshape_head
                scale_key = self.scale_k_reshape_head
            
            # 1. Apply RoPE to Q and K
            q_rope, scale_q_rope = self.query_rope_layer(q, scale_query,\
                                        cos, scale_cos,\
                                        sin, scale_sin)
            
            k_rope, scale_k_rope = self.key_rope_layer(k, scale_key,\
                                        cos, scale_cos,\
                                        sin, scale_sin)
            
            # Expand K and V to match number of heads
            k_rope = k_rope.repeat_interleave(self.group_size, dim=0)
            scale_k_rope = scale_k_rope.repeat_interleave(self.group_size, dim=0)
            
            v = v.repeat_interleave(self.group_size, dim=0)
            scale_v = self.scale_v_reshape_head.repeat_interleave(self.group_size, dim=0)
            
            # 2. Compute attention weights
            qk_weight_scale = scale_q_rope * scale_k_rope / self.scale_qk
            qk_weight_scale = qk_weight_scale.to(torch.float32)
            qk_weights_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(
                q_rope, k_rope, 
                qk_weight_scale
            )
            
            scale_qk_value = self.scale_qk / (self.head_dim ** 0.5)
            softmax_weight_q, scale_softmax = self.softmax(qk_weights_q,\
                                    scale_x=scale_qk_value)
            
            # 3. Compute attention output
            v = v.transpose(-2, -1).contiguous()
            scale_attn_out = scale_softmax * scale_v / self.scale_attn
            attn_output_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(
                softmax_weight_q, v, scale_attn_out.to(torch.float32)
            )
            
            # 4. Final output projection
            attn_output_q = attn_output_q.transpose(0, 1).contiguous().view(num_tokens, self.d_out)
            output_q, scale_out = self.out_proj(attn_output_q, self.scale_out_proj)
            return output_q, scale_out
        
        
    def finish_calibration(self):
        self.linear_q.finish_calibration()
        self.linear_k.finish_calibration()
        self.linear_v.finish_calibration()
        self.out_proj.finish_calibration()
        self.softmax.finish_calibration()
        self.query_rope_layer.finish_calibration()
        self.key_rope_layer.finish_calibration()
        
        if self.qk_norm:
            self.q_norm.finish_calibration()
            self.k_norm.finish_calibration()
        
        self.scale_q_reshape_head = self.q_reshape_head_observer.get_scale().to(self.scale_q_reshape_head.device)
        self.scale_k_reshape_head = self.k_reshape_head_observer.get_scale().to(self.scale_k_reshape_head.device)
        self.scale_v_reshape_head = self.v_reshape_head_observer.get_scale().to(self.scale_v_reshape_head.device)
        
        self.scale_qk = self.qk_observer.get_scale().to(self.scale_qk.device)
        self.scale_attn = self.atten_observer.get_scale().to(self.scale_attn.device)
        self.scale_out_proj = self.out_proj_observer.get_scale().to(self.scale_out_proj.device)
        
        self.is_quantized = True
        

if __name__ == "__main__":
    device = 'cuda'
    d_type = torch.bfloat16

    seq_len = 8192
    emb_dim = 4096        
    hidden_dims = 4096
    n_heads = 32
    head_dim = 128
    n_kv_groups = 4
    qk_norm = True  
    
    X = torch.randn((seq_len, emb_dim), dtype=d_type, device=device)
    
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    cos, sin = compute_rope_params(head_dim, context_length=seq_len, dtype=d_type)
    cos, sin = cos.to(device), sin.to(device)
    
    Custom_GroupQueryAttention_layer = Custom_GroupQueryAttention(
        d_in=emb_dim, num_heads=n_heads, num_kv_groups=n_kv_groups, head_dim=head_dim, qk_norm=qk_norm, dtype=d_type
    ).to(device=device, dtype=d_type)
    
    # 1. Calibration 
    with torch.no_grad():
        out, _ = Custom_GroupQueryAttention_layer(X, 1.0, mask, cos, 1.0, sin, 1.0)
        
    print(f"\nCalibration done. Some value: {out[:5, :5]} \n")
    
    cos_q, scale_cos = quantize_tensor(cos)
    sin_q, scale_sin = quantize_tensor(sin)
    
    # 2. Finish calibration and switch to quantized mode
    Custom_GroupQueryAttention_layer.finish_calibration()
    
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    with torch.no_grad():
        out_q, scale_out = Custom_GroupQueryAttention_layer(X_q, scale_x, mask,
                                                            cos_q, scale_cos,
                                                            sin_q, scale_sin)
    
    out_deq = out_q.float() * scale_out.unsqueeze(-1)
    
    print(f"Quantized inference done. Some value: {out_deq[:5, :5]}")
    
    max_diff = (out - out_deq).abs().max()
    print(f"Max diff: {max_diff.item()}")
    mse = ((out - out_deq) ** 2).mean()
    print(f"MSE: {mse.item()}")
    
    

