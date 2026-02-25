import os 
import torch
import torch.nn as nn
import gemm_cutlass
from utils_transformer_int8 import *
from utils import *

class Custom_Linear(nn.Module):
    def __init__(self, in_features, out_features, max_seq_len=1024):
        super(Custom_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        self.register_buffer(
            "weight_q",
            torch.empty(out_features, in_features, dtype=torch.int8),
            persistent=False,
        )
        
        self.register_buffer('scale_w', torch.tensor(1.0))
        self.register_buffer('scale_y', torch.ones(out_features))

        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.is_quantized = False
        
    def forward(self, x, scale_x):
        if not self.is_quantized:  # Calibration mode 
            out = torch.matmul(x, self.weight.t())  
            self.out_observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_y_value = self.scale_y[:seq_len]
            requant_scale = scale_x * self.scale_w / scale_y_value
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
            return out_q, scale_y_value
        
    def finish_calibration(self):
        self.weight_q, self.scale_w = quantize_tensor(self.weight)
        
        self.scale_y = self.out_observer.get_scale().to(self.scale_w.device)
        self.is_quantized = True  
        
        
class Custom_Softmax(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1, dim=None):
        super(Custom_Softmax, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
        self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len))
        
        self.is_quantized = False
        
    def forward(self, x_q, scale_x):
        if not self.is_quantized:  # Calibration mode
            assert x_q.dtype == torch.float32 or \
                    x_q.dtype == torch.bfloat16 or \
                    x_q.dtype == torch.float16,\
                    "Expected floating point input in calibration mode"
            out = torch.softmax(x_q, dim=-1)
            self.out_observer(out)
            return out, 1.0
        else:   # Quantized mode            
            seq_len = x_q.shape[-1]
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=x_q.device))
            
            scale_x_value = scale_x[:, :seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)

            out_q = gemm_cutlass.func_softmax_lastdim_int8_masking(
                x_q, scale_x_value.view(-1),
                scale_out_value.view(-1), mask
            )
            
            return out_q, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True  
        
        
class Custom_RMSNorm(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1024, dim=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # learnable weight of RMSNorm
        self.eps = eps
        self.dim = dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len 
        
        if self.num_heads == 1 or self.num_heads == None:
            self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
            self.register_buffer('scale_out', torch.ones(max_seq_len))
        elif self.num_heads > 1:
            self.out_observer = MinMaxObserverPerLastDim(self.num_heads, max_seq_len=self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len))
        else:
            raise ValueError("num_heads must be >= 1")
        
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
            
            if x.dim() == 2:
                seq_len = x.shape[0]
                scale_x_value = scale_x[:seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
            elif x.dim() == 3:
                seq_len = x.shape[1]
                scale_x_value = scale_x[:, :seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)
        
            y_q = gemm_cutlass.func_rmsnorm_int8(
                x, scale_x_value, self.weight, scale_out_value, self.eps
            )
            return y_q, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to(self.scale_out.device)
        self.is_quantized = True
        
        
class Custom_RoPE(nn.Module):
    def __init__(self, num_heads, max_seq_len=1024, head_dim=None):
        super(Custom_RoPE, self).__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
        self.register_buffer('scale_out',\
                    torch.ones(self.num_heads, self.max_seq_len)) 
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
            seq_len = x.shape[1]
            scale_x_value = scale_x[:, :seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)
            
            cos = cos[:seq_len, :]
            sin = sin[:seq_len, :]

            Y_int8 = gemm_cutlass.func_apply_rope_int8(x, scale_x_value, \
                            cos, scale_cos,
                            sin, scale_sin,
                            scale_out_value)
            return Y_int8, scale_out_value
    
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        

class Custom_Element_Wise(torch.nn.Module):
    def __init__(self, max_length = 1024):
        super().__init__()
        self.out_observer = MinMaxObserverPerLastDim(max_seq_len=max_length)
        self.register_buffer('scale_out', torch.ones(max_length)) 
        self.is_quantized = False
    def forward(self, a, scale_a, b, scale_b):
        if self.is_quantized:
            
            seq_len = a.shape[0]
            scale_a_value = scale_a[:seq_len].to(torch.float32)
            scale_b_value = scale_b[:seq_len].to(torch.float32)
            scale_out_value = self.scale_out[:seq_len].to(torch.float32)
            
            out_int8 = gemm_cutlass.func_element_wise_mul_int8(a, scale_a_value,\
                                b, scale_b_value,\
                                scale_out_value)
            return out_int8, scale_out_value
        else:
            out = a * b
            self.out_observer(out)
        return out, 1.0

    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().to('cuda')
        self.is_quantized = True
        
class Custom_Sigmoid(torch.nn.Module):
    def __init__(self, max_seq_len=1024):
        super(Custom_Sigmoid, self).__init__()
        self.max_seq_len = max_seq_len
        self.observer = MinMaxObserverPerLastDim(max_seq_len=max_seq_len)
        self.is_quantized = False
        self.register_buffer('scale_y', torch.ones(max_seq_len))  

    def forward(self, x, scale_x):
        if not self.is_quantized:
            out = torch.sigmoid(x)
            self.observer(out)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_x = scale_x[:seq_len].to(torch.float32)
            scale_y_value = self.scale_y[:seq_len].to(torch.float32)
            
            Y_q = gemm_cutlass.func_apply_sigmoid_int8(x, scale_x, scale_y_value)
            return Y_q, scale_y_value
        
    def finish_calibration(self):
        self.scale_y = self.observer.get_scale().to(self.scale_y.device)
        self.is_quantized = True
        
# class Custom_SiLU(nn.Module):
#     def __init__(self, max_length=1024):
#         super(Custom_SiLU, self).__init__()
#         self.observer = MinMaxObserverPerLastDim(max_seq_len=max_length)
#         self.is_quantized = False
#         self.register_buffer('scale_y', torch.ones(max_length)) 

#     def forward(self, x, scale_x):
#         if not self.is_quantized:
#             out = x * torch.sigmoid(x)
#             self.observer(out)
#             return out, 1.0
#         else:
#             assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
#             seq_len = x.shape[0]
#             scale_x = scale_x[:seq_len].to(torch.float32)
#             scale_y_value = self.scale_y[:seq_len].to(torch.float32)
            
#             Y_q = gemm_cutlass.func_apply_silu_int8(x, scale_x, scale_y_value)
#             return Y_q, scale_y_value

#     def finish_calibration(self):
#         self.scale_y = self.observer.get_scale().to(self.scale_y.device)
#         self.is_quantized = True  

class Custom_SiLU(nn.Module):
    def __init__(self, max_length=1024):
        super(Custom_SiLU, self).__init__()
        
        self.sigmoid = Custom_Sigmoid(max_seq_len=max_length)
        self.element_wise = Custom_Element_Wise(max_length=max_length)
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            sigmoid_out, _ = self.sigmoid(x, scale_x)
            out, _ = self.element_wise(x, 1.0, sigmoid_out, 1.0)
            return out, 1.0
        else:
            assert x.dtype == torch.int8, "Expected int8 input in quantized mode"
            
            seq_len = x.shape[0]
            scale_x = scale_x[:seq_len].to(torch.float32)
            
            out_sigmoid_q, scale_sigmoid = self.sigmoid(x, scale_x)
            out_q, scale_out = self.element_wise(x, scale_x, out_sigmoid_q, scale_sigmoid)
            return out_q, scale_out

    def finish_calibration(self):
        self.sigmoid.finish_calibration()
        self.element_wise.finish_calibration()
        self.is_quantized = True


class Custom_FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=1024)
        self.fc2 = Custom_Linear(cfg["emb_dim"], cfg["hidden_dim"], max_seq_len=1024)
        self.fc3 = Custom_Linear(cfg["hidden_dim"], cfg["emb_dim"], max_seq_len=1024)
        self.custom_silu = Custom_SiLU(max_length=1024)
        self.custom_elementwise_mul = Custom_Element_Wise(max_length=1024)
        
        self.is_quantized = False

    def forward(self, x, scale_x):
        if not self.is_quantized:
            x_fc1, _ = self.fc1(x, 1.0)
            x_fc2, _ = self.fc2(x, 1.0)
            x_silu, _ = self.custom_silu(x_fc1, 1.0)
            x, _ = self.custom_elementwise_mul(x_silu, 1.0, x_fc2, 1.0)
            out, _ = self.fc3(x, 1.0)
            return out, 1.0
        else:
            x_fc1, scale_fc1 = self.fc1(x, scale_x)
            x_fc2, scale_fc2 = self.fc2(x, scale_x)
            
            x_silu, scale_silu = self.custom_silu(x_fc1, scale_fc1)
            
            x_mul, scale_mul = self.custom_elementwise_mul(x_silu, scale_silu,
                                                           x_fc2, scale_fc2)
            
            out, scale_out = self.fc3(x_mul, scale_mul)
            return out, scale_out
        
    def finish_calibration(self):
        self.fc1.finish_calibration()
        self.fc2.finish_calibration()
        self.custom_silu.finish_calibration()
        self.custom_elementwise_mul.finish_calibration()
        self.fc3.finish_calibration()
        self.is_quantized = True


class Custom_Matmul(nn.Module):
    def __init__(self, num_heads=1, max_seq_len=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        if self.num_heads == 1:
            self.out_observer = MinMaxObserverPerLastDim(max_seq_len=self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.max_seq_len)) 
            
        elif self.num_heads > 1:
            self.out_observer = MinMaxObserverPerLastDim(self.num_heads, self.max_seq_len)
            self.register_buffer('scale_out', torch.ones(self.num_heads, self.max_seq_len)) 
        else:
            raise ValueError(f"num_heads should be >= 1, got {num_heads}")
        self.is_quantized = False
        
    def forward(self, A, scale_A, B, scale_B):
        """
        A: (M, K)
        B: (N, K)
        C = A @ B^T -> (M, N)
        """
        
        if self.is_quantized == False:
            if A.dim() == 2:
                C = torch.matmul(A, B.T)
                self.out_observer(C)
            elif A.dim() == 3:
                C = torch.matmul(A, B.transpose(-1, -2))
                self.out_observer(C)
            return C, 1.0
        else:
            if A.dim() == 2:
                seq_len = A.shape[0]
                
                # scale_A = scale_A[:seq_len].to(torch.float32)
                # scale_B = scale_B[:seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:seq_len].to(torch.float32)
                
                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            elif A.dim() == 3:
                batch_size, seq_len, _ = A.shape
                
                # scale_A = scale_A[:, :seq_len].to(torch.float32)
                # scale_B = scale_B[:, :seq_len].to(torch.float32)
                scale_out_value = self.scale_out[:, :seq_len].to(torch.float32)

                C_int8 = gemm_cutlass.func_int8_matmul_out_int8_three_scale_batched(
                    A, B, scale_A, scale_B, scale_out_value
                )
                return C_int8, scale_out_value
            else:
                raise ValueError(f"Unsupported input dimensions: {A.dim()}")
        
    def finish_calibration(self):
        self.scale_out = self.out_observer.get_scale().cuda()
        self.is_quantized = True
        
        
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
        