"""
This script demonstrates how to convert a LLaMA 3 model to LLM.int8(). 
The implementation main rely on the bitsandbytes library.
"""

import os 
from pathlib import Path
from safetensors.torch import load_file

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

from config import get_llama_config
from tokenizer import Tokenizer
from utils_weight import load_weights_into_llama    
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text


LLAMA_SIZE_STR = "3B" # "1B" or "3B"
IS_INSTRUCT = True # True or False

LLAMA32_CONFIG = get_llama_config(LLAMA_SIZE_STR)

if IS_INSTRUCT:
    MODEL_FOLDER = f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    HF_REPO_ID = f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct" 
else:
    MODEL_FOLDER = f"Llama-3.2-{LLAMA_SIZE_STR}"
    HF_REPO_ID = f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}"

MODEL_HUD_FOLDER_1 = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"
MODEL_HUD_FOLDER_2 = "/scratch/tnguyen10/"

if os.path.exists(MODEL_HUD_FOLDER_1):
    MODEL_HUB = MODEL_HUD_FOLDER_1
elif os.path.exists(MODEL_HUD_FOLDER_2):
    MODEL_HUB = MODEL_HUD_FOLDER_2
else:
    raise ValueError("Model hub folder not found. Please check the paths.")

LOCAL_DIR = os.path.join(MODEL_HUB, MODEL_FOLDER)

# ===============================================
# 1. Define Model Architecture
# ===============================================

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
    
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (num_heads, seq_len, head_dim)
    num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0)

    # Apply the rotary transformation    
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads,
            num_kv_groups,
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        # b, num_tokens, d_in = x.shape
        num_tokens, d_in = x.shape

        queries = self.W_query(x)  
        keys = self.W_key(x)  
        values = self.W_value(x) 

        # Reshape queries, keys, and values
        queries = queries.view(num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(0, 1)  # Shape: (num_kv_groups, num_tokens, head_dim)
        values = values.transpose(0, 1)  # Shape: (num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(0, 1)  # Shape: (num_heads, num_tokens, head_dim)

        # Apply RoPE
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # Expand keys and values to match the number of heads
        # Shape: (num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=0)  
        values = values.repeat_interleave(self.group_size, dim=0)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(0, 1).contiguous()

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  
        x = x + shortcut  

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  

        return x
    
    
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg


    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[0]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits



model = Llama3Model(LLAMA32_CONFIG)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device);

# ===============================================
# 2. Load Tokenizer
# ===============================================
tokenizer_file_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="original/tokenizer.model",
    local_dir=LOCAL_DIR
)

tokenizer = Tokenizer(tokenizer_file_path)

        
# ===============================================
# 3. Load Weights into Llama
# ===============================================
print(f"\n[INFO] Loading weights to model...")
if LLAMA_SIZE_STR == "1B":
    weights_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="model.safetensors",
        local_dir=LOCAL_DIR
    )
    combined_weights = load_file(weights_file)
else:
    combined_weights = {}
    for i in range(1, 3):
        weights_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"model-0000{i}-of-00002.safetensors",
            local_dir=LOCAL_DIR
        )
        current_weights = load_file(weights_file)
        combined_weights.update(current_weights)

load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
model.to(device)
del combined_weights  # free up memory
print(f"[INFO] Weights loaded successfully.\n")


# ===============================================
# 4. Generate Text
# ===============================================
MAX_GENERATED_TOKENS = 2048
PPL_CONTEXT_TOKENS = 2048
PPL_STRIDE = PPL_CONTEXT_TOKENS // 2
EVALUATION_DATASET = 'wikitext-103' # "wikitext-2" or "wikitext-103"

list_prompts = ["What is Dragon Ball story?"]

for prompt in list_prompts:
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=MAX_GENERATED_TOKENS,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\nResponse:\n", clean_text(output_text))
    
    
# =================================================
def convert_linear_to_llm_int8(module, threshold=6.0, skip_names=()):
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue

        if isinstance(child, nn.Linear):
            int8_linear = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )

            int8_linear.weight.data = child.weight.data.clone()

            if child.bias is not None:
                int8_linear.bias.data = child.bias.data.clone()

            setattr(module, name, int8_linear)
        else:
            convert_linear_to_llm_int8(child, threshold, skip_names)

    return module

model = model.cpu() # Move model to CPU for quantization
int8_model = convert_linear_to_llm_int8(model, threshold=6.0)
int8_model = int8_model.to(0) # Quantization happens here
print(f"\n[INFO] Model converted to LLM.int8() format successfully.\n")
print(f"Sample Weight after quantization: {int8_model.trf_blocks[0].att.W_query.weight.data[:5, :5]}")

for prompt in list_prompts:
    token_ids = generate(
        model=int8_model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=MAX_GENERATED_TOKENS,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\nResponse:\n", clean_text(output_text))
    
# # ================================================
# # Evaluation
# # ===============================================

samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

ppl = compute_ppl_single_text(int8_model,
                            tokenizer, 
                            samples,
                            context_size=PPL_CONTEXT_TOKENS,
                            stride=PPL_STRIDE)
print(f"\nPPL (LLM.int() technique): {ppl} \n")


# =================================================
# Measure Latency 
# =================================================

samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)
samples = tokenizer.encode(samples)

chunk_tokens = samples[0: PPL_CONTEXT_TOKENS]

input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
print(f"[INFO] Input tokens: {input_ids.shape}")

with torch.no_grad():
    out_ids = int8_model(input_ids)
print(f"[INFO] Output tokens: {out_ids.shape}")
    
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        out_ids = int8_model(input_ids)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
