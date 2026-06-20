"""
This script demonstrates how to convert a Qwen3 model to SmoothQuant technique. 
"""
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from pathlib import Path
from tqdm import tqdm
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch.nn import functional as F

import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config, load_weights_into_qwen
from utils_model import *
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text
from utils_quant import *
from utils_model_quan import compute_smooth_alpha, PerChannelAbsMaxObserver

import gemm_cutlass
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "14B"  # Options: "4B", "8B", "14B"


class SmoothQuant_Linear(nn.Module):
    """
    Linear layer with two execution modes:
    1. Calibration mode
        - Input: bf16
        - Weight: bf16
        - Output: bf16
        - Behavior: computes `x @ weight.T` and updates output observer
    
    2. Quantization mode
        - Input: int8
        - Weight: int8
        - Output: int8
        - Behavior: computes quantized matmul with per-row scaling
                    and returns quantized output and its scale
    
    This layer has: per-row scale for activation
                    per-tensor scale for weight. 
                    The output scale is per-row
                    
    Input shapes: (M, K) or (B, M, K)
    Weight shapes: (N, K)
    Output shapes: (M, N) or (B, M, N)
    """
    def __init__(self, in_features, out_features, dtype=torch.bfloat16):
        super(SmoothQuant_Linear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, 
                                               dtype=dtype))
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight quantization
        self.weight_q = torch.empty(out_features, in_features, dtype=torch.int8)
        self.scale_w = torch.ones(out_features, dtype=torch.float32, device='cuda')
        
        # SmoothQuant scaling factor
        self.smooth_alpha = torch.ones(in_features, dtype=torch.float32)
        self.in_observer = PerChannelAbsMaxObserver(in_features)
        
        # Output scale quantization
        self.is_quantized = False
        
    def forward(self, x, scale_x=1.0):
        if self.is_quantized == False:  
            
            if self.is_quantized == False:
                self.in_observer(x) # Calibrate input for SmoothQuant
             
            out = torch.matmul(x, self.weight.t())  
            
            return out
        else:
            assert x.dtype == torch.int8, "Expect int8 input in quantization"
            
            row_scale = scale_x 
            col_scale = self.scale_w.expand(self.out_features)
            
            out = gemm_cutlass.func_w8a8_matmul(x, self.weight_q,\
                row_scale, col_scale)
            
            return out
        
    def finish_calibration(self, alpha=None):
        if alpha is None:
            alpha = compute_smooth_alpha(self.in_observer, self.weight)
        self.smooth_alpha = alpha
        
        # # Quantize the smoothed weight
        w_smooth = self.weight * alpha.unsqueeze(0) 
        self.weight_q, self.scale_w = quantize_row_int8_symmetric_nd(w_smooth)

        self.is_quantized = True  
        
        # Delete bf16 weight and observer to save memory
        del self.weight
        del self.in_observer
        self.weight = None
        self.in_observer = None
        
        
class SmoothQuant_RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super(SmoothQuant_RMSNorm, self).__init__()
        self.normalized_shape = (emb_dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))
        
        self.is_quantized = False
        self.is_smooth_scale = False
        self.smooth_scale = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))
        
    def forward(self, x):
        if self.is_quantized == False:
            y = F.rms_norm(x, normalized_shape=self.normalized_shape,\
                                weight=self.weight, eps=self.eps)
            return y.to(x.dtype)
        else:
            if self.is_smooth_scale == False:
                dummy_scale = torch.ones(self.emb_dim, dtype=torch.float32, device=x.device)
                y_int8, scale_y = gemm_cutlass.func_rmsnorm_bf16_to_int8(x, self.weight,\
                                                dummy_scale, self.eps)
            else:
                y_int8, scale_y = gemm_cutlass.func_rmsnorm_bf16_to_int8(x, self.weight,\
                                                self.smooth_scale, self.eps)
            return y_int8, scale_y
    
    
    def finish_calibration(self):
        self.is_quantized = True
        
    def enable_smooth_scale(self, smooth_scale):
        """
        This function apply smooth scaling to quantization (for input activations) 
        Note:
        - Normal quantization: Y_int8 = Y / scale_Y
        - Smooth quantization: Y_int8 = (Y / smooth_scale) / scale_Y
        """
        self.is_smooth_scale = True
        self.smooth_scale.data.copy_(smooth_scale)


class SmoothQuant_Silu_Mul(nn.Module):
    def __init__(self, embed_dim):
        super(SmoothQuant_Silu_Mul, self).__init__()
        self.embed_dim = embed_dim
        self.is_quantized = False
        self.smooth_scale = nn.Parameter(torch.ones(embed_dim, dtype=torch.float32))
        self.is_smooth_scale = False

    def forward(self, x1, x2):
        if self.is_quantized == False:
            y = F.silu(x1) * x2
            return y
        else:
            if self.is_smooth_scale == False:
                dummy_smooth_scale = torch.ones(self.embed_dim, dtype=torch.float32, device=x1.device)
                y_i8, scale_y = gemm_cutlass.func_silu_mul_quant(x1, x2, dummy_smooth_scale)
            else:
                y_i8, scale_y = gemm_cutlass.func_silu_mul_quant(x1, x2, self.smooth_scale)
            return y_i8, scale_y
    
    def finish_calibration(self):
        self.is_quantized = True
        
    def enable_smooth_scale(self, smooth_scale):
        self.is_smooth_scale = True
        self.smooth_scale.data.copy_(smooth_scale)
            

class SmoothQuant_FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = SmoothQuant_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"])
        self.fc2 = SmoothQuant_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"])
        self.fc3 = SmoothQuant_Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.silu_mul_layer = SmoothQuant_Silu_Mul(cfg["hidden_dim"])
        
        self.is_quantized = False

    def forward(self, x, scale_x=1.0):
        if self.is_quantized == False:
            x_fc1 = self.fc1(x)
            x_fc2 = self.fc2(x)
            x = nn.functional.silu(x_fc1) * x_fc2
            x = self.fc3(x)
            return x
        else:
            x_i8 = x
            scale_x = scale_x
            assert x_i8.dtype == torch.int8, "Expect int8 input in quantization"
            
            x_fc1 = self.fc1(x_i8, scale_x)
            x_fc2 = self.fc2(x_i8, scale_x)
            
            x_i8, scale_x = self.silu_mul_layer(x_fc1, x_fc2)

            x = self.fc3(x_i8, scale_x)
            return x
    
    def finish_calibration(self):
        self.fc1.finish_calibration()
        fc1_smooth_alpha = self.fc1.smooth_alpha

        # fc2 uses the same smooth_alpha as fc1 
        self.fc2.finish_calibration(alpha=fc1_smooth_alpha) 
        
        self.fc3.finish_calibration()
        fc3_smooth_alpha = self.fc3.smooth_alpha
        
        self.silu_mul_layer.finish_calibration()
        self.silu_mul_layer.enable_smooth_scale(fc3_smooth_alpha)
        
        self.is_quantized = True
        
        
class SmoothQuant_GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.d_in = d_in
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        
        self.W_query = SmoothQuant_Linear(self.d_in, self.d_out, dtype=dtype)
        self.W_key = SmoothQuant_Linear(self.d_in, num_kv_groups * head_dim, dtype=dtype)
        self.W_value = SmoothQuant_Linear(self.d_in, num_kv_groups * head_dim, dtype=dtype)
        self.out_proj = SmoothQuant_Linear(self.d_out, self.d_in, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
            
        self.is_quantized = False

    def forward(self, x, scale_x, mask, cos, sin):
        
        if self.is_quantized == False:
            num_tokens, _ = x.shape

            # 1. QKV projections
            queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
            keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
            values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

            # 2. Reshape and transpose for multi-head attention
            queries = queries.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            keys = keys.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            values = values.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)

            if self.q_norm:
                queries = self.q_norm(queries)
            if self.k_norm:
                keys = self.k_norm(keys)

            # 3. Apply RoPE
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

            # 4. Expand K and V to match number of heads
            keys = keys.repeat_interleave(self.group_size, dim=0)
            values = values.repeat_interleave(self.group_size, dim=0)

            # 5. Attention
            attn_scores = queries @ keys.transpose(1, 2)
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

            # 6. Output
            context = attn_weights @ values  # Shape: (num_heads, num_tokens, head_dim)
            context = context.transpose(0, 1).reshape(num_tokens, self.d_out)
            out = self.out_proj(context)
            return out
        else:
            num_tokens, _ = x.shape
            
            x_int8 = x
            scale_x = scale_x

            # 1. QKV projections
            queries = self.W_query(x_int8, scale_x)  # (b, num_tokens, num_heads * head_dim)
            keys = self.W_key(x_int8, scale_x)       # (b, num_tokens, num_kv_groups * head_dim)
            values = self.W_value(x_int8, scale_x)   # (b, num_tokens, num_kv_groups * head_dim)

            # 2. Reshape and transpose for multi-head attention
            queries = queries.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
            keys = keys.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
            values = values.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)

            if self.q_norm:
                queries = self.q_norm(queries)
            if self.k_norm:
                keys = self.k_norm(keys)

            # 3. Apply RoPE
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

            # 4. Expand K and V to match number of heads
            keys = keys.repeat_interleave(self.group_size, dim=0)
            values = values.repeat_interleave(self.group_size, dim=0)

            # 5. Attention
            attn_scores = queries @ keys.transpose(1, 2)
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)
            attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

            # 6. Context computation
            context = attn_weights @ values  # Shape: (num_heads, num_tokens, head_dim)
            context = context.transpose(0, 1).reshape(num_tokens, self.d_out)
            
            # Smooth quantization for output activation of attention
            context_i8, scale_context = gemm_cutlass.func_quantize_row_int8_with_smooth_cuda(context, self.out_proj.smooth_alpha)
            
            out = self.out_proj(context_i8, scale_context)
            return out
    
    def finish_calibration(self):
        self.W_query.finish_calibration()
        smooth_query = self.W_query.smooth_alpha
        self.W_key.finish_calibration(alpha=smooth_query)
        self.W_value.finish_calibration(alpha=smooth_query)
         
        self.out_proj.finish_calibration()
        self.is_quantized = True
        

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = SmoothQuant_GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = SmoothQuant_FeedForward(cfg)
        self.norm1 = SmoothQuant_RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = SmoothQuant_RMSNorm(cfg["emb_dim"], eps=1e-6)
        
        self.is_quantized = False

    def forward(self, x, mask, cos, sin):
        # 1. Shortcut for attention block
        shortcut = x
        if self.is_quantized == False:
            x = self.norm1(x)
            x = self.att(x, 1.0, mask, cos, sin)  
        else:
            x_i8, scale_x = self.norm1(x)
            x = self.att(x_i8, scale_x, mask, cos, sin)
        x = x + shortcut  

        # 2. Shortcut for feed-forward block
        shortcut = x
        if self.is_quantized == False:
            x = self.norm2(x)
            x = self.ff(x)
        else:
            x_i8, scale_x = self.norm2(x)
            x = self.ff(x_i8, scale_x)
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
        
    
        
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for i in range(cfg["n_layers"])]
        )
        
        self.final_norm = RMSNorm(cfg["emb_dim"])
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

        num_tokens = x.shape[0]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def finish_calibration(self):
        for block in self.trf_blocks:
            block.finish_calibration()


if __name__ == "__main__":
    
    QWEN3_CONFIG = get_model_config(CHOOSE_MODEL)
    model = Qwen3Model(QWEN3_CONFIG)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device);

    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"

    # =================================================================
    # 1. Load model weights 
    # =================================================================
    # IMPORTANT: Change this path to your desired folder to store model weights
    MODEL_HUD_FOLDER_1 = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"
    MODEL_HUD_FOLDER_2 = "/scratch/tnguyen10/"

    if os.path.exists(MODEL_HUD_FOLDER_1):
        MODEL_HUD_FOLDER = MODEL_HUD_FOLDER_1
    elif os.path.exists(MODEL_HUD_FOLDER_2):
        MODEL_HUD_FOLDER = MODEL_HUD_FOLDER_2
    else:
        raise ValueError("Please update the MODEL_HUD_FOLDER.")

    local_dir = Path(repo_id).parts[-1]
    local_dir = os.path.join(MODEL_HUD_FOLDER, local_dir)

    if CHOOSE_MODEL == "0.6B":
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir)
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

    # ================================================================
    # 2. Load tokenizer
    # ================================================================
    if USE_REASONING_MODEL:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
    else:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

    tokenizer_file_path = os.path.join(MODEL_HUD_FOLDER, f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json")

    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)

    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=USE_REASONING_MODEL)
    else:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=False,
            add_generation_prompt=False,
            add_thinking=False)

    # ================================================================
    # 3. Text generation
    # ================================================================
    MAX_NEW_TOKENS = 2048
    PPL_CONTEXT_TOKENS = 2048
    EVALUATION_DATASET = "wikitext-103"  # Options: "wikitext-2", "wikitext-103"
    PPL_STRIDE = PPL_CONTEXT_TOKENS // 2

    list_prompt = ["What is the capital of VietNam?",\
                    "What is the Dragon Ball story?"]

    for idx, prompt in enumerate(list_prompt):
        input_token_ids = tokenizer.encode(prompt)
        input_token_ids_tensor = torch.tensor(input_token_ids, device=device)

        generated_text = func_generate_text(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id
        )

        response = get_clean_generated_text(generated_text, tokenizer)
        print(f"{idx}. Generated response: {response} \n")
        
    # # =================================================
    # Quantization
    # # =================================================
    print("\nCollecting calibration for quantization...")
    calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                    split="train", n=1_000)
    calibrate_tokens = tokenizer.encode(calibrate_samples)
    print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")
            
    # for i in range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE):
    for i in tqdm(range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE)):
        chunk_tokens = calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]

        input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)
            
    model.finish_calibration()
    print(f"[INFO] Finished SmoothQuant calibration.")
        
    
    # # ================================================
    # # Evaluation
    # # ===============================================

    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

    ppl = compute_ppl_single_text(model,
                                tokenizer, 
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print(f"\nPPL (SmoothQuant technique): {ppl} \n")
    
    # ================================================================
    # 3. Measure latency
    # ================================================================
    model.eval()

    samples = tokenizer.encode(samples)
    chunk_tokens = samples[0:PPL_CONTEXT_TOKENS]
    input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)

    print(f"[INFO] Input tokens: {input_ids.shape}")

    # Warm-up
    with torch.no_grad():
        out_ids = model(input_ids)
    torch.cuda.synchronize()

    print(f"[INFO] Output tokens: {out_ids.shape}")

    # Clear previous peak stats after warm-up
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Actual measured run
    with torch.no_grad():
        out_ids = model(input_ids)
    torch.cuda.synchronize()

    def calc_gpu_gb(x):
        return f"{x / 1024 / 1024 / 1024:.2f} GB"
    print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")
    
    
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
    print()
    print("Model Information (SmoothQuant technique):")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Data used for PPL evaluation: {EVALUATION_DATASET}")
