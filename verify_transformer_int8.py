import os 
import torch
import torch.nn as nn
import gemm_cutlass

from utils import *
from utils_transformer_int8 import *


class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight = nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        
        self.register_buffer('scale_w', torch.tensor(1.0))
        self.register_buffer('scale_y', torch.ones(out_features))

        self.out_observer = MinMaxObserverPerLastDim()
        
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Running calibration (x is float)
            out = torch.matmul(x, self.weight.t())  # normal float matmul
            self.out_observer(out)
            return out, torch.ones(out.shape[0], device=out.device)
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            requant_scale = scale_x * self.scale_w / self.scale_y
            requant_scale = requant_scale.to(torch.float32)
            
            if x.dim() == 3:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale_batched(
                    x, self.weight_q, requant_scale
                )
            elif x.dim() == 2:
                out_q = gemm_cutlass.func_int8_matmul_out_int8_per_row_scale(
                    x, self.weight_q, requant_scale
                )
            else:
                raise ValueError("Input must be 2D or 3D tensor")
            return out_q, self.scale_y
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_tensor(self.weight)
        
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        
        
class Custom_Softmax(nn.Module):
    def __init__(self, num_heads, max_seq_len, dim=-1):
        super(Custom_Softmax, self).__init__()
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(num_heads * max_seq_len))
        
        self.is_quantized = False
        
    def forward(self, x_q, scale_x):
        if not self.is_quantized:  # Calibration mode
            assert x_q.dtype == torch.float32 or \
                    x_q.dtype == torch.bfloat16 or \
                    x_q.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x_q, dim=-1, dtype=torch.bfloat16)
            self.out_observer(out)
            return out, 1.0
        else:            
            # Quantized mode
            seq_len = x_q.shape[-1]
            
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=x_q.device))
            
            flatten_scale_x = scale_x.view(-1)
            flatten_scale_out = self.scale_out.view(-1)
            
            out_q = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x_q, flatten_scale_x.to(torch.float32), 
                flatten_scale_out.to(torch.float32), mask
            )
            
            return out_q, self.scale_out
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True  


class Custom_RMSNorm(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # learnable weight of RMSNorm
        self.eps = eps
        self.dim = dim 
        
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(num_heads * max_seq_len))
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode
            assert x.dtype == torch.float32 or \
                    x.dtype == torch.bfloat16 or \
                    x.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            mean_square = x.pow(2).mean(-1, keepdim=True)  
            inv_rms = torch.rsqrt(mean_square + self.eps)   # [seq_len, 1]
            out = x * inv_rms * self.weight  # [seq_len, head_dim]
            self.out_observer(out)
            return out, 1.0
        else:
            # Quantized mode
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            y_q = gemm_cutlass.func_rmsnorm_int8(
                x, scale_x.to(torch.float32), self.weight, self.scale_out.to(torch.float32), self.eps
            )
            return y_q, self.scale_out
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True
        

class Custom_RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=1024):
        super(Custom_RoPE, self).__init__()
        self.head_dim = head_dim
        self.sequence_length = max_seq_len
        
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out',\
                    torch.ones(self.head_dim * self.sequence_length)) 
        self.is_quantized = False
        
    def forward(self, x, scale_x, 
                    cos, scale_cos,
                    sin, scale_sin):
        origin_shape = x.shape
        origin_dtype = x.dtype
        
        num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        
        if self.is_quantized == False:
            # 1. Split x into first half and second half
            x1 = x[..., : head_dim // 2]  # First half
            x2 = x[..., head_dim // 2 :]  # Second half

            # 2. Adjust sin and cos shapes
            cos = cos[:seq_len, :].unsqueeze(0)  # Shape:  1, seq_len, head_dim)
            sin = sin[:seq_len, :].unsqueeze(0)

            # 3. Apply the rotary transformation
            rotated = torch.cat((-x2, x1), dim=-1)
            x_rotated = (x * cos) + (rotated * sin)
            
            # 4. Reshape back to original shape and dtype
            out = x_rotated.to(dtype=origin_dtype)
            out = out.view(origin_shape)  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            Y_int8 = gemm_cutlass.func_apply_rope_int8(x, scale_x.to(torch.float32), \
                            cos, scale_cos,
                            sin, scale_sin,
                            self.scale_out.to(torch.float32))
            return Y_int8, self.scale_out
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True

    
class Custom_GroupQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None,\
        qk_norm=True, dtype=torch.bfloat16):
        
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
        
        self.query_rope_layer = Custom_RoPE(head_dim, max_seq_len=1024)
        self.key_rope_layer = Custom_RoPE(head_dim, max_seq_len=1024)
        
        if qk_norm:
            self.q_norm = Custom_RMSNorm(num_heads, max_seq_len=1024, dim=head_dim, eps=1e-6)
            self.k_norm = Custom_RMSNorm(num_kv_groups, max_seq_len=1024, dim=head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        
        # TODO: assume max_seq_len is 1024 for now
        max_seq_len = 1024
        self.softmax = Custom_Softmax(num_heads, max_seq_len, dim=-1)
            
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
        
        
class Custom_SiLU(nn.Module):
    def __init__(self, max_length=1024):
        super(Custom_SiLU, self).__init__()
        self.observer = MinMaxObserverPerLastDim()
        self.is_quantized = False
        self.register_buffer('scale_y', torch.ones(max_length))  # Assuming max_length = 1024

    def forward(self, x, scale_x):
        if not self.is_quantized:
            out = x * torch.sigmoid(x)
            self.observer(out)
            return out, torch.ones(out.shape[0], device=out.device)
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            scale_x = scale_x.to(torch.float32)
            
            Y_q = gemm_cutlass.func_apply_silu_int8(x, scale_x, self.scale_y.to(torch.float32))
            return Y_q, self.scale_y

    def finish_calibration(self):
        self.scale_y = self.observer.get_scale().to(self.scale_y.device)
        self.is_quantized = True  
        

class Custom_ElementWiseMul(nn.Module):
    def __init__(self, max_length=1024):
        super(Custom_ElementWiseMul, self).__init__()
        self.observer = MinMaxObserverPerLastDim()
        self.is_quantized = False
        self.register_buffer('scale_y', torch.ones(max_length))  # Assuming max_length = 1024

    def forward(self, x1, scale_x1, x2, scale_x2):
        if not self.is_quantized:
            out = x1 * x2
            self.observer(out)
            return out, torch.ones(out.shape[0], device=out.device)
        else:
            assert x1.dtype == torch.int8 and x2.dtype == torch.int8, "Expected int8 inputs in quantized mode"
            out_q = gemm_cutlass.func_element_wise_mul_int8(x1, scale_x1.to(torch.float32),\
                                            x2, scale_x2.to(torch.float32), self.scale_y.to(torch.float32))
            return out_q, self.scale_y

    def finish_calibration(self):
        self.scale_y = self.observer.get_scale().to(self.scale_y.device)
        self.is_quantized = True


class Custom_FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"], )
        self.fc2 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc3 = Custom_Linear(cfg["hidden_dim"], cfg["emb_dim"])
        self.custom_silu = Custom_SiLU()
        self.custom_elementwise_mul = Custom_ElementWiseMul()
        
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
        

class Custom_Element_Add(torch.nn.Module):
    def __init__(self, max_length = 1024):
        super().__init__()
        self.out_observer = MinMaxObserverPerLastDim()
        self.register_buffer('scale_out', torch.ones(max_length)) 
        self.is_quantized = False
    def forward(self, a, scale_a, b, scale_b):
        if self.is_quantized:
            out_int8 = gemm_cutlass.func_element_add_int8(a, scale_a.to(torch.float32),\
                                b, scale_b.to(torch.float32), self.scale_out.to(torch.float32))
            return out_int8, self.scale_out
        else:
            out = a + b
            self.out_observer(out)
        return out, 1.0

    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        

        
class Custom_Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Custom_GroupQueryAttention(cfg["emb_dim"], cfg["num_heads"], cfg["num_kv_groups"], qk_norm=cfg["qk_norm"])
        self.ffn = Custom_FeedForward(cfg)
        self.norm2 = Custom_RMSNorm(cfg["num_heads"], max_seq_len=1024, dim=cfg["emb_dim"], eps=1e-6)
        self.norm1 = Custom_RMSNorm(cfg["num_heads"], max_seq_len=1024, dim=cfg["emb_dim"], eps=1e-6)
        self.add1_layer = Custom_Element_Add(max_length=1024)
        self.add2_layer = Custom_Element_Add(max_length=1024)
        self.is_quantized = False
        
    def forward(self, x, scale_x, mask, cos, scale_cos, sin, scale_sin):
        if not self.is_quantized:
            # 1. Attention block
            shortcut = x
            x, _ = self.norm1(x, 1.0)
            x_attn, _ = self.attn(x, 1.0, mask, cos, 1.0, sin, 1.0)
            x, _ = self.add1_layer(x_attn, 1.0, shortcut, 1.0)
            
            # 2. FFN block
            shortcut = x
            x, _ = self.norm2(x, 1.0)
            x_ffn, _ = self.ffn(x, 1.0)
            x, _ = self.add2_layer(x_ffn, 1.0, shortcut, 1.0)
            return x, 1.0
        else:    
            # 1. Attention block
            shortcut = x
            x, scale_x = self.norm1(x, scale_x)
            x_attn, scale_attn = self.attn(x, scale_x, mask, cos, scale_cos, sin, scale_sin)
            x, scale_add1 = self.add1_layer(x_attn, scale_attn, shortcut, scale_x)
            
            # 2. FFN block
            shortcut = x
            x, scale_norm2 = self.norm2(x, scale_add1)
            x_ffn, scale_ffn = self.ffn(x, scale_norm2)
            x, scale_add2 = self.add2_layer(x_ffn, scale_ffn, shortcut, scale_add1)
            
            return x, scale_add2
        
    
    def finish_calibration(self):
        self.attn.finish_calibration()
        self.ffn.finish_calibration()
        self.add1_layer.finish_calibration()
        self.add2_layer.finish_calibration()
        self.norm1.finish_calibration()
        self.norm2.finish_calibration()
        self.is_quantized = True
        
        
if __name__ == "__main__":
    cfg = {
        "emb_dim": 512,
        "hidden_dim": 2048,
        "num_heads": 8,
        "num_kv_groups": 4,
        "qk_norm": True
    }
    device = 'cuda'
    dtype = torch.bfloat16
    model = Custom_Transformer(cfg).to(device).to(dtype)
    
    seq_len = 1024
    
    X = torch.randn(seq_len, cfg["emb_dim"], device=device, dtype=dtype)
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
    
    cos, sin = compute_rope_params(cfg["emb_dim"] // cfg["num_heads"], context_length=seq_len, dtype=torch.bfloat16)
    cos, sin = cos.to(device), sin.to(device)
    
    # 1. Run calibration
    out, scale_out = model(X, 1.0, mask, cos, 1.0, sin, 1.0)
    print(f"Some out: {out[:5, :5]}")
    
    # 2. Finish calibration and switch to quantized mode
    model.finish_calibration()
    
    cos_q, scale_cos = quantize_tensor(cos)
    sin_q, scale_sin = quantize_tensor(sin)
    
    # 3. Run quantized inference
    X_q, scale_x = quantize_row_int8_symmetric_nd(X)
    out_q, scale_out_q = model(X_q, scale_x, mask, cos_q, scale_cos, sin_q, scale_sin)
    
    out_q_dequant = out_q.float() * scale_out_q.unsqueeze(-1)
    print(f"Some quantized out (dequantized): {out_q_dequant[:5, :5]}")    