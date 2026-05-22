import os 
import torch
torch.manual_seed(123)
import torch.nn as nn

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from config import get_llama_config
from tokenizer import Tokenizer
from utils_model import *
from utils_weight import load_weights_into_llama    
from utils_generation import *
from utils_quant import *
from utils_model_quan import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text


LLAMA_SIZE_STR = "1B" # "1B" or "3B"
LLAMA32_CONFIG = get_llama_config(LLAMA_SIZE_STR)

MODEL_FOLDER = f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"

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
        self.inv_softmax_scale = 1 / (head_dim ** 0.5) # softmax scale: 1/sqrt(d_k)

        self.W_query = Custom_Linear(d_in, self.d_out).to(dtype)
        self.W_key = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.W_value = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.out_proj = Custom_Linear(self.d_out, d_in).to(dtype)

        self.query_rope = Custom_RoPE(num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        self.key_rope = Custom_RoPE(num_kv_groups, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        
        self.softmax_layer = Custom_Softmax(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)    
        self.qk_score_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.context_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN,\
            is_return_float=True).to(dtype)
    
        self.is_quantized = False

    def forward(self, x, scale_x, mask, cos, scale_cos, sin, scale_sin):
        num_tokens, _ = x.shape

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
            attn_scores = attn_scores * self.inv_softmax_scale
            attn_weights, _ = self.softmax_layer(attn_scores, 1.0, 1.0)         
            
            # Compute context
            values = values.transpose(1, 2)  # Shape: (num_heads, head_dim, num_tokens)
            context, _ = self.context_layer(attn_weights, 1.0, values, 1.0)
            
            # Compute output
            context = context.transpose(0, 1).reshape(num_tokens, self.d_out) 
            out, _ = self.out_proj(context, 1.0)
        else: 
            # === Quantized computation ===
            x_int8 = x
            x_scale = scale_x
            
            queries_int8, queries_scale = self.W_query(x_int8, x_scale)
            keys_int8, keys_scale = self.W_key(x_int8, x_scale)
            values_int8, values_scale = self.W_value(x_int8, x_scale)
            
            # Reshape for multi-head 
            queries_int8 = queries_int8.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            
            keys_int8 = keys_int8.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            
            values_int8 = values_int8.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            values_scale = values_scale.unsqueeze(0).expand(self.num_kv_groups, -1)
            
            
            # Apply RoPE to quantized Q and K
            queries_int8, queries_scale = self.query_rope(queries_int8, queries_scale,\
                                        cos, scale_cos, sin, scale_sin)
            
            keys_int8, keys_scale = self.key_rope(keys_int8, keys_scale,\
                                        cos, scale_cos, sin, scale_sin)
            
            # Repeat K and V for grouped attention
            keys_int8 = keys_int8.repeat_interleave(self.group_size, dim=0)
            keys_scale = keys_scale.repeat_interleave(self.group_size, dim=0)
            values_int8 = values_int8.repeat_interleave(self.group_size, dim=0)
            values_scale = values_scale.repeat_interleave(self.group_size, dim=0)
            
            # Attention score with quantization    
            attn_scores_int8, attn_scores_scale = self.qk_score_layer(queries_int8,\
                                                        queries_scale,\
                                                        keys_int8,\
                                                        keys_scale)
            
            # Softmax the attention scores with quantization 
            attn_scores_scale = attn_scores_scale * self.inv_softmax_scale  # Adjust scale for softmax
            attn_weights_int8, attn_weights_scale = self.softmax_layer(attn_scores_int8,\
                                                    attn_scores_scale, mask)
            
            # Compute context with quantization
            values_int8, values_scale = gemm_cutlass.func_dequant_transpose_requant(values_int8,\
                                                        values_scale)
            
            context, _ = self.context_layer(attn_weights_int8,\
                                            attn_weights_scale,\
                                            values_int8,\
                                            values_scale)

            context = context.transpose(0, 1).reshape(num_tokens, self.d_out) 
            out, _ = self.out_proj(context, 1.0)  # Output float for better accuracy
        
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

        self.att = Custom_GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = Custom_FeedForward(cfg).to(cfg["dtype"])
        
        self.norm1 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        
        self.norm2 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos.to(cfg["dtype"]))
        self.register_buffer("sin", sin.to(cfg["dtype"]))
        
        cos_int8, cos_scale = quantize_tensor(cos)
        sin_int8, sin_scale = quantize_tensor(sin)
        self.register_buffer("cos_int8", cos_int8)
        self.register_buffer("cos_scale", cos_scale)
        self.register_buffer("sin_int8", sin_int8)
        self.register_buffer("sin_scale", sin_scale)

        self.is_quantized = False

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        
        if self.is_quantized == False:
            x = self.norm1(x)
            
            mask = torch.triu(torch.ones(x.shape[0], x.shape[0],\
                                device=x.device, dtype=torch.bool), diagonal=1)
            x = self.att(x, 1.0, mask, self.cos, 1.0, self.sin, 1.0)  
        else:
            x, scale_x = self.norm1(x)
            
            mask = torch.tril(torch.ones((x.shape[0], x.shape[0]),\
                            dtype=torch.uint8, device=x.device))
            x = self.att(x, scale_x, mask,\
                        self.cos_int8, self.cos_scale,\
                        self.sin_int8, self.sin_scale)  
        
        x = x + shortcut 

        # Shortcut connection for feed-forward block
        shortcut = x
        
        if self.is_quantized == False:
            x = self.norm2(x)
            x, _ = self.ff(x, 1.0)
        
        else:
            x, scale_x = self.norm2(x)
            x, _ = self.ff(x, scale_x)
        
        x = x + shortcut  

        return x
    
    def finish_calibration(self):
        self.att.finish_calibration()
        self.norm1.finish_calibration()
        
        self.ff.finish_calibration()
        
        self.norm2.finish_calibration()
        smooth_scale = self.ff.fc1.smooth_alpha
        self.norm2.enable_smooth_scale(smooth_scale)
        
        self.is_quantized = True


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        self.cfg = cfg


    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        
        for block in self.trf_blocks:
            x = block(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def finish_calibration(self):
        for block in self.trf_blocks:
            block.finish_calibration()
            
            
if __name__ == "__main__":

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
        local_dir=LOCAL_DIR
    )

    tokenizer = Tokenizer(tokenizer_file_path)

            
    # ===============================================
    # 3. Load Weights into Llama
    # ===============================================
    if LLAMA_SIZE_STR == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename="model.safetensors",
            local_dir=LOCAL_DIR
        )
        combined_weights = load_file(weights_file)
    else:
        combined_weights = {}
        for i in range(1, 3):
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=LOCAL_DIR
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)


    load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
    model.to(device)
    del combined_weights  # free up memory

    # ===============================================
    # 4. Generate Text
    # ===============================================
    MAX_GENERATED_TOKENS = 1024
    PPL_CONTEXT_TOKENS = 1024
    PPL_STRIDE = PPL_CONTEXT_TOKENS // 2
    EVALUATION_DATASET = 'wikitext-103' # "wikitext-2" or "wikitext-103"

    list_prompts = ["What is Dragon Ball story?"]

    for prompt in list_prompts:
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=MAX_GENERATED_TOKENS,
            context_size=LLAMA32_CONFIG["context_length"],
            top_k=1)

        output_text = token_ids_to_text(token_ids, tokenizer)
        print("\nResponse:\n", clean_text(output_text))
        

    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

    ppl = compute_ppl_single_text(model,
                                tokenizer, 
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print("PPL:", ppl)   
        
    # ===============================================
    # 5. Quantization
    # ===============================================

    print("\nCollecting calibration for quantization...")
    calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                    split="train", n=100_000)
    calibrate_tokens = tokenizer.encode(calibrate_samples)
    print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")
            
    for i in range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE):
        chunk_tokens = calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]

        input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)
            
    model.finish_calibration()
    print(f"[INFO] Finished calibration.")


    # ========================================================================
    # Quantization mode
    print("\n===== Generated text after quantization: =====\n")
    list_prompts = ["What is Dragon Ball story?"]

    for prompt in list_prompts:
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=MAX_GENERATED_TOKENS,
            context_size=LLAMA32_CONFIG["context_length"],
            top_k=1)

        output_text = token_ids_to_text(token_ids, tokenizer)
        print("\nResponse:\n", clean_text(output_text)) 
        

    ppl = compute_ppl_single_text(model,
                                tokenizer, 
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print("PPL:", ppl)   
        