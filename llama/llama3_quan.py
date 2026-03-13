import os 
from pathlib import Path
from safetensors.torch import load_file

import torch

from utils_evaluation import load_wikitext2_samples
torch.manual_seed(123)
import torch.nn as nn
from huggingface_hub import hf_hub_download

from model_utils import *
from config import get_llama_config
from tokenizer import Tokenizer
from weight_utils import load_weights_into_llama    
from generation_utils import *
from utils_evaluation import *

from quant_utils import quantize_row_int8_symmetric_nd, quantize_tensor
from model_quan_utils import *


LLAMA_SIZE_STR = "3B" # "1B" or "3B"
LLAMA32_CONFIG = get_llama_config(LLAMA_SIZE_STR)


# ===============================================
# 1. Define Model Architecture
# ===============================================

class Custom_GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, dtype=None
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
            
        self.query_rope = Custom_RoPE(num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        self.key_rope = Custom_RoPE(num_kv_groups, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        
        self.softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)    
        self.qk_score_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.context_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
    
        self.is_quantized = False

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        original_dtype = x.dtype
        x = x.squeeze(0)  # Remove batch dimension for linear layers

        # Float computation (calibration)
        if self.is_quantized == False:  
            queries, _ = self.W_query(x, 1.0)
            keys, _ = self.W_key(x, 1.0)
            values, _ = self.W_value(x, 1.0)
            
            # Reshape and transpose for multi-head attention
            queries = queries.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            keys = keys.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            values = values.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            
            # Apply RoPE to Q and K
            queries, _ = self.query_rope(queries, 1.0, cos, 1.0, sin, 1.0)
            keys, _ = self.key_rope(keys, 1.0, cos, 1.0, sin, 1.0)
            
            keys = keys.repeat_interleave(self.group_size, dim=0)
            values = values.repeat_interleave(self.group_size, dim=0)
            
            # Attention score 
            attn_scores, _ = self.qk_score_layer(queries, 1.0, keys, 1.0) 
            
            # Softmax the attention scores
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_scores = attn_scores / (self.head_dim ** 0.5)
            attn_weights, _ = self.softmax_layer(attn_scores, 1.0)         
            
            # Compute context
            values = values.transpose(1, 2)  # Shape: (num_heads, head_dim, num_tokens)
            context, _ = self.context_layer(attn_weights, 1.0, values, 1.0)
            
            # Compute output
            context = context.transpose(0, 1).reshape(num_tokens, self.d_out) 
            out, _ = self.out_proj(context, 1.0)
            
        else: 
            # === Quantized computation ===
            x_int8, x_scale = quantize_row_int8_symmetric_nd(x)
            
            queries_int8, queries_scale = self.W_query(x_int8, x_scale)
            keys_int8, keys_scale = self.W_key(x_int8, x_scale)
            values_int8, values_scale = self.W_value(x_int8, x_scale)
            
            # Reshape for multi-head attention
            queries_int8 = queries_int8.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            keys_int8 = keys_int8.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            values_int8 = values_int8.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)

            queries_scale = queries_scale.unsqueeze(0).expand(self.num_heads, -1)
            keys_scale = keys_scale.unsqueeze(0).expand(self.num_kv_groups, -1)
            values_scale = values_scale.unsqueeze(0).expand(self.num_kv_groups, -1)
            
            # Apply RoPE to quantized Q and K
            cos_int8, cos_scale = quantize_tensor(cos)
            sin_int8, sin_scale = quantize_tensor(sin)
            
            queries_int8, queries_scale = self.query_rope(queries_int8, queries_scale, cos_int8, cos_scale, sin_int8, sin_scale)
            keys_int8, keys_scale = self.key_rope(keys_int8, keys_scale, cos_int8, cos_scale, sin_int8, sin_scale)
            
            # Repeat K and V for grouped attention
            keys_int8 = keys_int8.repeat_interleave(self.group_size, dim=0)
            keys_scale = keys_scale.repeat_interleave(self.group_size, dim=0)
            values_int8 = values_int8.repeat_interleave(self.group_size, dim=0)
            values_scale = values_scale.repeat_interleave(self.group_size, dim=0)
            
            # Attention score with quantization    
            attn_scores_int8, attn_scores_scale = self.qk_score_layer(queries_int8, queries_scale, keys_int8, keys_scale)
            
            # Softmax the attention scores with quantization 
            attn_scores_scale = attn_scores_scale.to(torch.float32) / (self.head_dim ** 0.5)  # Adjust scale for softmax
            attn_weights_int8, attn_weights_scale = self.softmax_layer(attn_scores_int8, attn_scores_scale)
            
            # Compute context with quantization
            values = values_int8.to(torch.float32) * values_scale.unsqueeze(-1)
            values = values.transpose(1, 2)  # Shape: (num_heads, head_dim, num_tokens)
            values_int8, values_scale = quantize_row_int8_symmetric_nd(values)
            
            context_int8, context_scale = self.context_layer(attn_weights_int8, attn_weights_scale,
                                                             values_int8, values_scale)
            
            # Compute output with quantization
            context = context_int8.to(torch.float32) * context_scale.unsqueeze(-1)
            context = context.transpose(0, 1).reshape(num_tokens, self.d_out)
            context = context.to(self.out_proj.weight.dtype)  
            
            out, _ = self.out_proj(context, 1.0)  # Output projection in float for better accuracy
        
        out = out.unsqueeze(0) # Add batch dimension back
        out = out.to(original_dtype)
        
        return out
    
    def finish_calibration(self):
        self.W_query.finish_calibration()
        self.W_key.finish_calibration()
        self.W_value.finish_calibration()
        self.query_rope.finish_calibration()
        self.key_rope.finish_calibration()
        self.softmax_layer.finish_calibration()
        self.qk_score_layer.finish_calibration()
        self.context_layer.finish_calibration()
        # self.out_proj.finish_calibration()
        self.is_quantized = True
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.att = GroupedQueryAttention(
        #     d_in=cfg["emb_dim"],
        #     d_out=cfg["emb_dim"],
        #     num_heads=cfg["n_heads"],
        #     num_kv_groups=cfg["n_kv_groups"],
        #     dtype=cfg["dtype"]
        # )
        self.att = Custom_GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        
        
        # self.ff = FeedForward(cfg)
        self.ff = Custom_FeedForward(cfg).to(cfg["dtype"])
        
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
        
        # x = self.ff(x)
        # # === Feed-forward with quantization support ===
        original_dtype = x.dtype
        x = x.squeeze(0)  # Remove batch dimension
        x, _ = self.ff(x, 1.0)
        x = x.unsqueeze(0) # Add batch dimension back
        x = x.to(original_dtype)
        # # ========================================        
        
        x = x + shortcut  

        return x
    
    def finish_calibration(self):
        self.att.finish_calibration()
        self.ff.finish_calibration()
    

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
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

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def finish_calibration(self):
        for block in self.trf_blocks:
            block.finish_calibration()
            

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
    repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
    filename="original/tokenizer.model",
    local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
)

tokenizer = Tokenizer(tokenizer_file_path)

        
# ===============================================
# 3. Load Weights into Llama
# ===============================================
if LLAMA_SIZE_STR == "1B":
    weights_file = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="model.safetensors",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )
    combined_weights = load_file(weights_file)
else:
    combined_weights = {}
    for i in range(1, 3):
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename=f"model-0000{i}-of-00002.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
        )
        current_weights = load_file(weights_file)
        combined_weights.update(current_weights)


load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
model.to(device)
del combined_weights  # free up memory



# ===============================================
# 4. Generate Text
# ===============================================

MAX_GENERATED_TOKENS = 2048
PPL_CONTEXT_TOKENS = 2048

list_prompt = ["What is the capital of VietNam?",\
                "Who is Son Goku?",\
                "Describe the Chinese New Year festival.",\
                "Which country has a capital city named Paris?"]

for prompt in list_prompt:
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=MAX_GENERATED_TOKENS,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
        temperature=0.
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\nResponse:\n", clean_text(output_text))
    

num_samples = 500

samples = load_wikitext2_samples(num_samples)
print(f"Loaded {len(samples)} samples. Computing perplexity...")

corpus_ppl = compute_ppl(
    model=model,
    tokenizer=tokenizer,           
    texts=samples,
    context_size=PPL_CONTEXT_TOKENS,
    device=device
)
print("Corpus PPL (before quantization):", corpus_ppl)   
    

# ===============================================
# 5. Quantization Utilities
# ===============================================

# Forward pass to collect calibration data for quantization
print("\nCollecting calibration data for quantization...")
calibration_samples = load_wikitext2_samples()
for idx, text in enumerate(calibration_samples):
    input_token_ids = tokenizer.encode(text)
    
    for i in range(0, len(input_token_ids), PPL_CONTEXT_TOKENS):
        chunk_token_ids = input_token_ids[i:i+PPL_CONTEXT_TOKENS]
        if len(chunk_token_ids) < 2:  # Skip short chunks
            continue
    
        input_token_ids_tensor = torch.tensor(chunk_token_ids, device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(input_token_ids_tensor)
        
model.finish_calibration()
print(f"[INFO] Finished calibration {len(calibration_samples)} samples.")


# ========================================================================
# Quantization mode
print("\n===== Generated text after quantization: =====\n")
list_prompt = ["What is the capital of VietNam?",\
            "Describe the Chinese New Year festival.",\
            "Who is Son Goku?"]

for prompt in list_prompt:
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=MAX_GENERATED_TOKENS,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
        temperature=0.
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print("\nResponse:\n", clean_text(output_text)) 
    
corpus_ppl = compute_ppl(
    model=model,
    tokenizer=tokenizer,           
    texts=samples,
    context_size=PPL_CONTEXT_TOKENS,
    device=device
)
print("Corpus PPL (after quantization):", corpus_ppl)