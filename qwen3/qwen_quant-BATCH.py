import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select GPU "0", "1", "2",...

import time
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch.nn import functional as F

import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config, load_weights_into_qwen
from utils_model import *
from utils_generation import *
from utils_evaluation import load_wikitext_single_text
from utils_model_quan import *

import gemm_cutlass
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "4B"  # Options: "4B", "8B", "14B"

class Custom_GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, dtype=torch.bfloat16
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.dtype = dtype

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = Custom_Linear(d_in, self.d_out).to(dtype)
        self.W_key = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.W_value = Custom_Linear(d_in, num_kv_groups * head_dim).to(dtype)
        self.out_proj = Custom_Linear(self.d_out, d_in).to(dtype)

        self.query_rope = Custom_RoPE(num_heads, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        self.key_rope = Custom_RoPE(num_kv_groups, max_seq_len=MAX_SEQ_LEN, head_dim=head_dim).to(dtype)
        
        self.softmax_layer = Custom_Softmax(num_heads=num_heads).to(dtype)    
        self.qk_score_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN).to(dtype)
        self.context_layer = Custom_Matmul(num_heads=num_heads, max_seq_len=MAX_SEQ_LEN,\
            is_return_float=True).to(dtype)
        
        self.q_norm = Custom_RMSNorm(head_dim, eps=1e-6).to(dtype)
        self.k_norm = Custom_RMSNorm(head_dim, eps=1e-6).to(dtype)
    
        self.is_quantized = False

    def forward(self, x, scale_x, mask, cos, scale_cos, sin, scale_sin):
        if x.dim() == 2:
            num_tokens, _ = x.shape
        elif x.dim() == 3:
            batch, num_tokens, _ = x.shape
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}")

        if self.is_quantized == False:  
            queries, _ = self.W_query(x, 1.0)
            keys, _ = self.W_key(x, 1.0)
            values, _ = self.W_value(x, 1.0)
            
            # Reshape and transpose for multi-head attention
            queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            keys = keys.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            values = values.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            
            # Normalize Q and K
            queries = self.q_norm(queries, 1.0)
            keys = self.k_norm(keys, 1.0)
            
            # Apply RoPE to Q and K
            queries, _ = self.query_rope(queries, 1.0, cos, 1.0, sin, 1.0)
            keys, _ = self.key_rope(keys, 1.0, cos, 1.0, sin, 1.0)
            
            keys = keys.repeat_interleave(self.group_size, dim=1) 
            values = values.repeat_interleave(self.group_size, dim=1) 
            
            # Attention score 
            attn_scores, _ = self.qk_score_layer(queries, 1.0, keys, 1.0) 
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_scores = attn_scores / (self.head_dim ** 0.5)
            attn_weights, _ = self.softmax_layer(attn_scores, 1.0, 1.0)         
            
            # Compute context
            values = values.transpose(2, 3)  # Shape: (num_heads, head_dim, num_tokens)
            context, _ = self.context_layer(attn_weights, 1.0, values, 1.0)
            
            # Compute output
            context = context.transpose(1, 2).reshape(batch, num_tokens, self.d_out) 
            out, _ = self.out_proj(context, 1.0)
        else: 
            # === Quantized computation ===
            x_int8 = x
            x_scale = scale_x
            
            queries_int8, queries_scale = self.W_query(x_int8, x_scale)
            keys_int8, keys_scale = self.W_key(x_int8, x_scale)
            values_int8, values_scale = self.W_value(x_int8, x_scale)
            
            # Reshape for multi-head 
            queries_int8 = queries_int8.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            queries_scale = queries_scale.unsqueeze(1).expand(-1, self.num_heads, -1)
            
            keys_int8 = keys_int8.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            keys_scale = keys_scale.unsqueeze(1).expand(-1, self.num_kv_groups, -1)
            values_int8 = values_int8.view(batch, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            values_scale = values_scale.unsqueeze(1).expand(-1, self.num_kv_groups, -1)
            
            # Normalize Q and K
            queries_int8, queries_scale = self.q_norm(queries_int8, queries_scale)
            keys_int8, keys_scale = self.k_norm(keys_int8, keys_scale)
            
            # Apply RoPE to quantized Q and K
            queries_int8, queries_scale = self.query_rope(queries_int8, queries_scale,\
                                        cos, scale_cos, sin, scale_sin)
            
            keys_int8, keys_scale = self.key_rope(keys_int8, keys_scale,\
                                        cos, scale_cos, sin, scale_sin)
            
            # Repeat K and V for grouped attention
            keys_int8 = keys_int8.repeat_interleave(self.group_size, dim=1)
            keys_scale = keys_scale.repeat_interleave(self.group_size, dim=1)
            values_int8 = values_int8.repeat_interleave(self.group_size, dim=1)
            values_scale = values_scale.repeat_interleave(self.group_size, dim=1)
            
            # Attention score 
            attn_scores_int8, attn_scores_scale = self.qk_score_layer(queries_int8,\
                                                        queries_scale,\
                                                        keys_int8,\
                                                        keys_scale)
            
            attn_scores_scale = attn_scores_scale / (self.head_dim ** 0.5)
            attn_weights_int8, attn_weights_scale = self.softmax_layer(attn_scores_int8,\
                                                    attn_scores_scale, mask)
            
            # Compute context with quantization
            values_int8, values_scale = gemm_cutlass.func_dequant_transpose_requant(values_int8,\
                                                        values_scale)
            
            context, _ = self.context_layer(attn_weights_int8,\
                                            attn_weights_scale,\
                                            values_int8,\
                                            values_scale)

            context = context.transpose(1, 2).reshape(batch, num_tokens, self.d_out) 
            out, _ = self.out_proj(context, 1.0)  # Output float for better accuracy
        
        return out
    
    def finish_calibration(self):
        self.W_query.finish_calibration()
        # Key and Value use the same smooth alpha as Query
        smooth_query = self.W_query.smooth_alpha
        self.W_key.finish_calibration(alpha=smooth_query) 
        self.W_value.finish_calibration(alpha=smooth_query) 
        
        self.q_norm.finish_calibration()
        self.k_norm.finish_calibration()
        self.query_rope.finish_calibration()
        self.key_rope.finish_calibration()
        self.softmax_layer.finish_calibration()
        self.qk_score_layer.finish_calibration()
        self.context_layer.finish_calibration()
        # self.out_proj.finish_calibration()
        self.is_quantized = True


class TransformerBlock_Quant(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = Custom_GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        
        self.norm1 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-6, dtype=cfg["dtype"])
        
        self.norm2 = RMSNorm_Fuse_Quant(cfg["emb_dim"], eps=1e-6, dtype=cfg["dtype"])
        
        self.ff = Custom_FeedForward(cfg).to(cfg["dtype"])
        
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            # context_length=cfg["context_length"]
            context_length=MAX_SEQ_LEN
        )
        self.register_buffer("cos", cos.to(cfg["dtype"]))
        self.register_buffer("sin", sin.to(cfg["dtype"]))
        
        cos_i8, cos_scale = quantize_tensor(cos)
        sin_i8, sin_scale = quantize_tensor(sin)
        self.register_buffer("cos_i8", cos_i8)
        self.register_buffer("cos_scale", cos_scale)
        self.register_buffer("sin_i8", sin_i8)
        self.register_buffer("sin_scale", sin_scale)

        self.is_quantized = False

    def forward(self, x):
        
        if x.dim() == 2:
            n_tokens, _ = x.shape
        elif x.dim() == 3:
            b, n_tokens, _ = x.shape
        
        # 1. Shortcut for attention block
        shortcut = x
        if self.is_quantized == False:
            x = self.norm1(x) 
            mask = torch.triu(torch.ones(n_tokens, n_tokens, device=x.device, dtype=torch.bool), diagonal=1)
            x = self.att(x, 1.0, mask, self.cos, 1.0, self.sin, 1.0)  
        else:
            x, x_scale = self.norm1(x)
            mask = torch.tril(torch.ones((n_tokens, n_tokens), dtype=torch.uint8, device=x.device))
            x = self.att(x, x_scale, mask, self.cos_i8, self.cos_scale, self.sin_i8, self.sin_scale)
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
        smooth_scale = self.att.W_query.smooth_alpha
        self.norm1.enable_smooth_scale(smooth_scale)
        
        self.ff.finish_calibration()
        self.norm2.finish_calibration()
        smooth_scale = self.ff.fc1.smooth_alpha
        self.norm2.enable_smooth_scale(smooth_scale)
        
        self.is_quantized = True


class Qwen3Model_Quant(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock_Quant(cfg) for i in range(cfg["n_layers"])]
        )
        
        self.final_norm = RMSNorm(cfg["emb_dim"])
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
    
    QWEN3_CONFIG = get_model_config(CHOOSE_MODEL)
    model = Qwen3Model_Quant(QWEN3_CONFIG)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model.to(device);

    # =================================================================
    # 1. Load model weights 
    # =================================================================
    # IMPORTANT: Change this path to your desired folder to store model weights
    MODEL_HUB_FOLDER_1 = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"
    MODEL_HUB_FOLDER_2 = "/scratch/tnguyen10/"
    if os.path.exists(MODEL_HUB_FOLDER_1):
        MODEL_HUB_FOLDER = MODEL_HUB_FOLDER_1
    elif os.path.exists(MODEL_HUB_FOLDER_2):
        MODEL_HUB_FOLDER = MODEL_HUB_FOLDER_2
    else:
        raise ValueError("Please update the MODEL_HUB_FOLDER.")
    
    if USE_REASONING_MODEL:
        hf_repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Thinking-2507"
        local_model_name = f"Qwen3-{CHOOSE_MODEL}-Thinking-2507"
    elif USE_INSTRUCT_MODEL:
        hf_repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Instruct-2507"
        local_model_name = f"Qwen3-{CHOOSE_MODEL}-Instruct-2507"
    else:
        hf_repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"
        local_model_name = f"Qwen3-{CHOOSE_MODEL}-Base"

    local_dir = os.path.join(MODEL_HUB_FOLDER, local_model_name)
    print(f"[INFO] Local model disk: {local_dir} \n")

    if CHOOSE_MODEL == "0.6B":
        weights_file = os.path.join(local_dir, "model.safetensors")
        weights_dict = load_file(weights_file)
    else:
        weights_dict = {}
        
        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))
        for filename in shard_files:
            shard_path = os.path.join(local_dir, filename)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")

            shard = load_file(shard_path)
            weights_dict.update(shard)

    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model.to(device)
    print(f"[INFO] Model weights loaded successfully. \n")
    del weights_dict

    # ================================================================
    # 2. Load tokenizer
    # ================================================================
    tokenizer_file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename="tokenizer.json",
        local_dir=local_dir
    )
    print(f"[INFO] Tokenizer path: {tokenizer_file_path}")

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=hf_repo_id,
        apply_chat_template=USE_REASONING_MODEL or USE_INSTRUCT_MODEL,
        add_generation_prompt=USE_REASONING_MODEL or USE_INSTRUCT_MODEL,
        add_thinking=USE_REASONING_MODEL
    )

    # ================================================================
    # 3. Text generation
    # ================================================================
    MAX_NEW_TOKENS = 2048
    PPL_CONTEXT_TOKENS = 2048
    EVALUATION_DATASET = "wikitext-2"  # Options: "wikitext-2", "wikitext-103"
    PPL_STRIDE = PPL_CONTEXT_TOKENS // 2

    prompt = "What is Dragon Ball story?"
    list_prompts = []
    for _ in range(BATCH_SIZE):
        list_prompts.append(prompt)
    
    batch_input_ids = torch.stack([
        text_to_token_ids(prompt, tokenizer, batch_dim=True).squeeze(0)
        for prompt in list_prompts
    ], dim=0).to(device)

    token_ids_batch = generate_batch(
        model=model,
        idx=batch_input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        context_size=PPL_CONTEXT_TOKENS,
        temperature=0.0,
        top_k=1,
        eos_id=None
    )

    for i in range(token_ids_batch.shape[0]):
        output_text = token_ids_to_text(token_ids_batch[i].unsqueeze(0), tokenizer, batch_dim=True)
        print('-' * 50)
        print(f"\nResponse {i}:\n", clean_text(output_text))


    # ================================================================
    # Quantization
    # ================================================================

    print("\nCollecting calibration for quantization...")
    calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                    split="train", n=1_000)
    calibrate_tokens = tokenizer.encode(calibrate_samples)
    print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")

    # Construct calibration batches
    CALIBRATION_BATCH_SIZE = BATCH_SIZE

    calibration_chunks = [
        calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]
        for i in range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE)
    ]

    model.eval()
    
    input_ids = torch.tensor(
        calibration_chunks[:CALIBRATION_BATCH_SIZE],
        dtype=torch.long,
        device=device
    )  # [B, T]
    print(f"[INFO] Input shape: {input_ids.shape}")
            
    
    # Measure Latency before quantization
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)
            
    n_iter = 10
    start_time = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = model(input_ids)
    end_time = time.time()
    avg_latency = (end_time - start_time) / n_iter
    print(f"\n[INFO] Average latency (before quantization): {avg_latency:.4f} seconds (batch = {BATCH_SIZE}).")
    print("Model information:")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Batch size: {BATCH_SIZE}\n\n")        

    model.finish_calibration()
    print(f"[INFO] Finished calibration.")


    # ===============================================
    # Measure Latency after quantization
    # ===============================================
    # Warm-up runs
    for _ in range(5):
        with torch.no_grad():
            out_ids = model(input_ids)
    
    n_iter = 10
    start_time = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            out_ids = model(input_ids)
    end_time = time.time()
    avg_latency = (end_time - start_time) / n_iter
    print(f"\n[INFO] Average latency (after quantization): {avg_latency:.4f} seconds (batch = {BATCH_SIZE}).")
    print("Model information:")
    print(f"Model: Qwen3-{CHOOSE_MODEL} with QUANTIZATION")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Batch size: {CALIBRATION_BATCH_SIZE}\n")
        
    
    print("\nProfiling the quantized model...")
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
            out_ids = model(input_ids)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))