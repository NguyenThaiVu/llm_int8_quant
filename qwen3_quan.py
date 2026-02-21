import os
from pathlib import Path
import zipfile
import math
from datasets import load_dataset
import re
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch.nn import functional as F
from importlib.metadata import version
import time

from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config

from utils_transformer_int8 import *
from utils_layer_int8 import *
from utils import *
from utils_evaluation import load_wikitext2_samples, compute_ppl

pkgs = [
    "huggingface_hub",  # to download pretrained weights
    "tokenizers",       # to implement the tokenizer
    "torch",            # to implement the model
]
for p in pkgs:
    print(f"{p} version: {version(p)}")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "8B"  # Options:, "4B", "8B"


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.weight

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
    

    
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles) 
    sin = torch.sin(angles)

    return cos, sin


MAX_SEQ_LEN = 1024
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = Custom_Linear(d_in, self.d_out, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.W_key = Custom_Linear(d_in, num_kv_groups * head_dim, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.W_value = Custom_Linear(d_in, num_kv_groups * head_dim, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.out_proj = Custom_Linear(self.d_out, d_in, max_seq_len=MAX_SEQ_LEN).to(dtype)

        if qk_norm:
            self.q_norm = Custom_RMSNorm(self.num_heads, max_seq_len=MAX_SEQ_LEN, dim=head_dim).to(dtype)
            self.k_norm = Custom_RMSNorm(self.num_kv_groups, max_seq_len=MAX_SEQ_LEN, dim=head_dim).to(dtype)
        else:
            self.q_norm = self.k_norm = None
            
        self.query_rope = Custom_RoPE(num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        self.key_rope = Custom_RoPE(num_kv_groups, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        
        self.softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)    
        self.qk_score_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
    
        self.is_quantized = False

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # 1. QKV projections
        # queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        # keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        # values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)
        
        # ===== Linear layers with quantization support =====
        original_dtype = x.dtype
        x = x.squeeze(0)  # Remove batch dimension for linear layers
        if self.is_quantized == False:
            queries, _ = self.W_query(x, 1.0)
            keys, _ = self.W_key(x, 1.0)
            values, _ = self.W_value(x, 1.0)
        else:
            x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
            
            queries_int8, queries_scale = self.W_query(x_int8, x_scale)
            keys_int8, keys_scale = self.W_key(x_int8, x_scale)
            values_int8, values_scale = self.W_value(x_int8, x_scale)
            
            queries = queries_int8.to(torch.float32) * queries_scale.unsqueeze(-1)
            keys = keys_int8.to(torch.float32) * keys_scale.unsqueeze(-1)
            values = values_int8.to(torch.float32) * values_scale.unsqueeze(-1)
        
        queries = queries.unsqueeze(0) # Add batch dimension back
        keys = keys.unsqueeze(0)
        values = values.unsqueeze(0)
        queries = queries.to(original_dtype)
        keys = keys.to(original_dtype)
        values = values.to(original_dtype)

        # 2. Reshape and transpose for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            original_dtype = queries.dtype
            queries = queries.squeeze(0)  # Remove batch dimension for normalization
            if self.is_quantized == False:
                queries, _ = self.q_norm(queries, 1.0)
            else:
                queries_int8, queries_scale = quantize_row_int8_symmetric_nd(queries)
                queries_int8, queries_scale = self.q_norm(queries_int8, queries_scale)
                queries = queries_int8.to(torch.float32) * queries_scale.unsqueeze(-1)
            queries = queries.unsqueeze(0) # Add batch dimension back
            queries = queries.to(original_dtype)
            
        if self.k_norm:            
            original_dtype = keys.dtype
            keys = keys.squeeze(0)  # Remove batch dimension for normalization
            if self.is_quantized == False:
                keys, _ = self.k_norm(keys, 1.0)
            else:
                keys_int8, keys_scale = quantize_row_int8_symmetric_nd(keys)
                keys_int8, keys_scale = self.k_norm(keys_int8, keys_scale)
                keys = keys_int8.to(torch.float32) * keys_scale.unsqueeze(-1)
            keys = keys.unsqueeze(0) # Add batch dimension back
            keys = keys.to(original_dtype)
            

        # 3. Apply RoPE        
        # ===== RoPE with quantization support =====
        original_dtype = queries.dtype
        queries = queries.squeeze(0)  # Remove batch dimension 
        keys = keys.squeeze(0)
        if self.is_quantized == False:
            queries, _ = self.query_rope(queries, 1.0, cos, 1.0, sin, 1.0)
            keys, _ = self.key_rope(keys, 1.0, cos, 1.0, sin, 1.0)
        else:
            queries_int8, queries_scale = quantize_row_int8_symmetric_nd(queries)
            keys_int8, keys_scale = quantize_row_int8_symmetric_nd(keys)
            cos_int8, cos_scale = quantize_tensor(cos)
            sin_int8, sin_scale = quantize_tensor(sin)
            
            queries_int8, queries_scale = self.query_rope(queries_int8, queries_scale, cos_int8, cos_scale, sin_int8, sin_scale)
            keys_int8, keys_scale = self.key_rope(keys_int8, keys_scale, cos_int8, cos_scale, sin_int8, sin_scale)
            
            queries = queries_int8.to(torch.float32) * queries_scale.unsqueeze(-1)
            keys = keys_int8.to(torch.float32) * keys_scale.unsqueeze(-1)
        queries = queries.unsqueeze(0) # Add batch dimension back
        keys = keys.unsqueeze(0)
        queries = queries.to(original_dtype)
        keys = keys.to(original_dtype)
        # ========================================
        
        # 4. Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 5. Attention
        # attn_scores = torch.matmul(queries, keys.transpose(2, 3))
        
        # # ===== Attention scores with quantization support =====
        original_dtype = queries.dtype
        queries = queries.squeeze(0)  # Remove batch dimension
        keys = keys.squeeze(0)
        
        if self.is_quantized == False:
            attn_scores, _ = self.qk_score_layer(queries, 1.0, keys, 1.0)
        else:
            queries_int8, queries_scale = quantize_row_int8_symmetric_nd(queries)
            keys_int8, keys_scale = quantize_row_int8_symmetric_nd(keys)
            
            attn_scores_int8, attn_scores_scale = self.qk_score_layer(queries_int8, queries_scale, keys_int8, keys_scale)
            attn_scores = attn_scores_int8.to(torch.float32) * attn_scores_scale.unsqueeze(-1)
        
        attn_scores = attn_scores.unsqueeze(0) # Add batch dimension back
        attn_scores = attn_scores.to(original_dtype)
        # # ========================================
        
        # ===== Softmax with quantization support =====
        original_dtype = attn_scores.dtype
        attn_scores = attn_scores.squeeze(0)  # Remove batch dimension for softmax
        
        if self.is_quantized == False:
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_scores = attn_scores / (self.head_dim ** 0.5)
            attn_weights, _ = self.softmax_layer(attn_scores, 1.0)
        else:
            attn_scores_int8, attn_scores_scale = quantize_row_int8_symmetric_nd(attn_scores)
            attn_scores_scale = attn_scores_scale.to(torch.float32) / (self.head_dim ** 0.5)  # Adjust scale for softmax
            attn_weights_int8, attn_weights_scale = self.softmax_layer(attn_scores_int8, attn_scores_scale)
            attn_weights = attn_weights_int8.to(torch.float32) * attn_weights_scale.unsqueeze(-1)
        
        attn_weights = attn_weights.unsqueeze(0) # Add batch dimension back
        attn_weights = attn_weights.to(original_dtype)   
        # ======================================= 

        # 6. Output
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        
        # ===== Output projection with quantization support =====
        # out = self.out_proj(context)
        original_dtype = context.dtype
        context = context.squeeze(0)  # Remove batch dimension for linear layer
        if self.is_quantized == False:
            out, _ = self.out_proj(context, 1.0)
        else:
            context_int8, context_scale = quantize_row_int8_symmetric_nd(context)
            out_int8, out_scale = self.out_proj(context_int8, context_scale)
            out = out_int8.to(torch.float32) * out_scale.unsqueeze(-1)
        out = out.unsqueeze(0) # Add batch dimension back
        out = out.to(original_dtype)
        # =======================================================
        
        return out
    
    def finish_calibration(self):
        self.W_query.finish_calibration()
        self.W_key.finish_calibration()
        self.W_value.finish_calibration()
        self.query_rope.finish_calibration()
        self.key_rope.finish_calibration()
        self.out_proj.finish_calibration()
        self.q_norm.finish_calibration() if self.q_norm is not None else None
        self.k_norm.finish_calibration() if self.k_norm is not None else None
        self.softmax_layer.finish_calibration()
        self.qk_score_layer.finish_calibration()
        self.is_quantized = True
    
    
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
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        
        self.norm1 = Custom_RMSNorm(max_seq_len = 1024, dim=cfg["emb_dim"]).to(cfg["dtype"])
        
        self.ff = Custom_FeedForward(cfg).to(cfg["dtype"])
        # self.ff = FeedForward(cfg)
        
        # self.norm2 = Custom_RMSNorm(max_seq_len = 1024, dim=cfg["emb_dim"]).to(cfg["dtype"])
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        
        self.is_quantized = False
        

    def forward(self, x, mask, cos, sin):
        # 1. Shortcut for attention block
        shortcut = x
        
        original_dtype = x.dtype
        x = x.squeeze(0)  # Remove batch dimension
        if self.is_quantized == False: # float computation
            x, _ = self.norm1(x, 1.0)
        else:
            x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
            x_int8, x_scale = self.norm1(x_int8, x_scale)
            
            x = x_int8.to(torch.float32) * x_scale.unsqueeze(-1)
        x = x.unsqueeze(0) # Add batch dimension back
        x = x.to(original_dtype)
        
        
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  

        # 2. Shortcut for feed-forward block
        shortcut = x
        x = self.norm2(x)
        
        
        # x = self.ff(x)
        # === Feed-forward with quantization support ===
        original_dtype = x.dtype
        x = x.squeeze(0)  # Remove batch dimension 
        
        if self.is_quantized == False: # float computation
            x, _ = self.ff(x, 1.0)
        else:
            x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
            out_int8, out_scale = self.ff(x_int8, x_scale)
            
            # Dequantization and process
            x = out_int8.to(torch.float32) * out_scale.unsqueeze(-1)
        x = x.to(original_dtype)
        x = x.unsqueeze(0) # Add batch dimension back
        # ========================================
        
        x = x + shortcut  # Add the original input back

        return x

    def finish_calibration(self):
        self.att.finish_calibration()
        self.ff.finish_calibration()
        
        self.norm1.finish_calibration()
        # self.norm2.finish_calibration()
        self.is_quantized = True
        
        
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # self.final_norm = Custom_RMSNorm(max_seq_len = 1024, dim=cfg["emb_dim"]).to(cfg["dtype"])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        
        self.is_quantized = False
        
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        
        x = self.final_norm(x)
        # # ===== Normalization Layer =====
        # original_dtype = x.dtype
        # x = x.squeeze(0)  # Remove batch dimension
        # if self.is_quantized == False: # float computation
        #     x, _ = self.final_norm(x, 1.0)
        # else:
        #     x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
        #     x_int8, x_scale = self.final_norm(x_int8, x_scale)
            
        #     x = x_int8.to(torch.float32) * x_scale.unsqueeze(-1)
        # x = x.unsqueeze(0) # Add batch dimension back
        # x = x.to(original_dtype)
        # # ===============================
        
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def finish_calibration(self):
        for block in self.trf_blocks:
            block.finish_calibration()
        # self.final_norm.finish_calibration()
        self.is_quantized = True
    
    
QWEN3_CONFIG = get_model_config(CHOOSE_MODEL)

model = Qwen3Model(QWEN3_CONFIG)

print(model)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model.to(device);


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
    
        return left 

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.weight = assign(
                att.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.weight = assign(
                att.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.weight = assign(
            block.norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        block.norm2.weight = assign(
            block.norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
        
        
import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
    repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
else:
    repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"

# =================================================================
# IMPORTANT: Change this path to your desired folder to store model weights
# MODEL_HUD_FOLDER = "/scratch/tnguyen10/"
MODEL_HUD_FOLDER = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"


local_dir = Path(repo_id).parts[-1]
local_dir = os.path.join(MODEL_HUD_FOLDER, local_dir)

if CHOOSE_MODEL == "0.6B":
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights_dict = load_file(weights_file)
else:
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)

load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)
del weights_dict


if USE_REASONING_MODEL:
    tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
else:
    tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

tokenizer_file_path = os.path.join(MODEL_HUD_FOLDER, f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json")

hf_hub_download(
    repo_id=repo_id,
    filename="tokenizer.json",
    local_dir=local_dir,
)

if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=USE_REASONING_MODEL
    )
else:
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False
    )
    
    
# Test the tokenizer
prompt = "Give me a short introduction to large language models."

input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
print(text)


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token
            
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
def get_clean_generated_text(generated_text):
    output_text = ""
    for token in generated_text:
        token_id = token.squeeze(0).tolist()
        text = tokenizer.decode(token_id)
        output_text += text
    # Post-processing to remove incomplete special tokens at the end
    incomplete_special_token_pattern = re.compile(r"<\|[^>]*?$")
    output_text = re.sub(incomplete_special_token_pattern, "", output_text)
    output_text = output_text.strip()
    
    return output_text


MAX_NEW_TOKENS = 128

list_prompt = ["What is the capital of VietNam?",\
               "Who is the president of VietNam?",\
               "Who is Son Goku?",\
               "Who is Ho Chi Minh?",\
               "Describe the Iphone 14 Pro Max in detail.",\
               "Which country has a capital city named Paris?",\
               "Describe the Chinese New Year festival.",\
               "Please describe British food in detail.",\
                "Tell me a long story about dragons and knights.",\
                "Explain the Vietnamese food Pho and how to make it at home."]

for idx, prompt in enumerate(list_prompt):
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    generated_text = generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id
    )

    reponse = get_clean_generated_text(generated_text)
    print(f"{idx}. Generated response: {reponse} \n")
    

num_samples = 10

samples = load_wikitext2_samples(num_samples)
print(f"Loaded {len(samples)} samples. Computing perplexity...")

per_text, corpus_ppl = compute_ppl(
    model=model,
    tokenizer=tokenizer,           
    texts=samples,
    context_size=MAX_NEW_TOKENS,
    device=device
)

print("Corpus PPL (before quantization):", corpus_ppl)    



# ========================================================================
# Quantization 
# Loop through all block and call finish_calibration 
model.finish_calibration()
    
print("\n===== Finish calibration. Generated text after quantization: =====\n")
list_prompt = ["What is the capital of VietNam?",\
            "Which country has a capital city named Paris?",\
            "Describe the Chinese New Year festival.",\
            "Explain the Vietnamese food Pho and how to make it at home."]
    
for idx, prompt in enumerate(list_prompt):
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)  

    generated_text_quant = generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id
    )

    response_quant = get_clean_generated_text(generated_text_quant)
    print(f"{idx}. Generated response: {response_quant} \n")



num_samples = 10

samples = load_wikitext2_samples(num_samples)
print(f"Loaded {len(samples)} samples. Computing perplexity...")


per_text, corpus_ppl = compute_ppl(
    model=model,
    tokenizer=tokenizer,           
    texts=samples,
    context_size=MAX_NEW_TOKENS,
    device=device
)

print("Corpus PPL (after quantization):", corpus_ppl)