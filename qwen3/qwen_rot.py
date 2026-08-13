import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select GPU "0", "1", "2",...
from pathlib import Path
import zipfile
from tqdm import tqdm
import math
from datasets import load_dataset
import re
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch.nn import functional as F

import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config, load_weights_into_qwen
from utils_model import compute_rope_params, apply_rope
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text,\
                            evaluate_arc, evaluate_piqa
from utils_quant import *
import gemm_cutlass
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "14B"  # Options: "4B", "8B", "14B"


class Rot_Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.bfloat16, bias=False):
        super(Rot_Linear, self).__init__()
            
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        
        self.weight_q = torch.empty(out_features, in_features, dtype=torch.int8)
        self.scale_w = torch.ones(out_features, dtype=torch.float32, device='cuda')
        
        self.is_quantized = False
    
    def forward(self, x):
        if self.is_quantized == False:
            y = x @ self.weight.T
        else:
            x_rot = gemm_cutlass.func_apply_hadamard(x)
            x_rot_q, scale_x = gemm_cutlass.func_quantize_i8(x_rot)
            
            y = gemm_cutlass.func_w8a8_matmul(x_rot_q, self.weight_q, scale_x, self.scale_w)
            
        return y.to(dtype=self.dtype)
    
    def finish_calibration(self):
        """
        In this function, we: 
        - Apply Hadamard rotation to the weight matrix.
        - Quantize the rotated weight matrix to int8.
        """
        
        # w_rot = block_hadamard_rotate(self.weight)
        # w_rot_q, scale_w = quantize_row_int8_symmetric_nd(w_rot)
        w_rot = gemm_cutlass.func_apply_hadamard(self.weight)
        w_rot_q, scale_w = gemm_cutlass.func_quantize_i8(w_rot)
        self.weight_q = w_rot_q
        self.scale_w = scale_w
        
        self.is_quantized = True


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Rot_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = Rot_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = Rot_Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
    def finish_calibration(self):
        self.fc1.finish_calibration()
        self.fc2.finish_calibration()
        self.fc3.finish_calibration()
        print(f"[INFO] Finish calibration for FeedForward module.")
        
    
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class Rot_GroupedQueryAttention(nn.Module):
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

        self.W_query = Rot_Linear(d_in, self.d_out, dtype=dtype, bias=False)
        self.W_key = Rot_Linear(d_in, num_kv_groups * head_dim, dtype=dtype, bias=False)
        self.W_value = Rot_Linear(d_in, num_kv_groups * head_dim, dtype=dtype, bias=False)  
        self.out_proj = Rot_Linear(self.d_out, d_in, dtype=dtype, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        
        self.is_quantized = False

    def forward(self, x, mask, cos, sin):
        num_tokens, _ = x.shape

        # 1. QKV projections
        queries = self.W_query(x)  # (num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (num_tokens, num_kv_groups * head_dim)

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
        values = values.repeat_interleave(self.group_size, dim=0) # Shape: (num_heads, num_tokens, head_dim)

        # 5. Attention
        keys = keys.transpose(1, 2) 
        attn_scores = queries @ keys
            
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1) # Shape: (num_heads, num_tokens, num_tokens)
        
        context = attn_weights @ values  # Shape: (num_heads, num_tokens, head_dim)

        # 6. Output
        context = context.transpose(0, 1).reshape(num_tokens, self.d_out)
        output = self.out_proj(context)
        return output

    def finish_calibration(self):
        self.W_query.finish_calibration()
        self.W_key.finish_calibration()
        self.W_value.finish_calibration()
        self.out_proj.finish_calibration()
        self.is_quantized = True
        print(f"[INFO] Finish calibration for Rot_GroupedQueryAttention module.")
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = Rot_GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # 1. Shortcut for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # 2. Shortcut for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x
    
    def finish_calibration(self):
        self.att.finish_calibration()
        self.ff.finish_calibration()
    
class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
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
            # context_length=cfg["context_length"]
            context_length=1024 * 10
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
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Model folder does not exist: {local_dir}")
    print(f"[INFO] Loading model weights from disk: {local_dir} \n")

    if CHOOSE_MODEL == "0.6B":
        weights_file = os.path.join(local_dir, "model.safetensors")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Missing weights file: {weights_file}")
        weights_dict = load_file(weights_file)
    else:
        repo_dir = local_dir
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing index file: {index_path}")

        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}

        shard_files = sorted(set(index["weight_map"].values()))

        for filename in shard_files:
            shard_path = os.path.join(repo_dir, filename)

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
    EVALUATION_DATASET = "wikitext-2"  # Options: "wikitext-2", "wikitext-103"
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
    
    # Evaluate PPL before quantization
    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)
    ppl = compute_ppl_single_text(model,
                                tokenizer, 
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print(f"\nPPL (Before Quantization): {ppl} \n")
    print(f"Model Information: ")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Data used for PPL evaluation: {EVALUATION_DATASET}")
    
    # # =================================================
    # 4. Quantization
    # # =================================================            
    model.finish_calibration()
    model.eval()
    print(f"\n[INFO] Finished QuantRot.\n")
    
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
    
    print(f"\n[INFO] Start Evaluation...")

    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

    ppl = compute_ppl_single_text(model,
                                tokenizer, 
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print(f"PPL (Qwen - Rotation Quantization): {ppl}")
    print(f"Model Information: ")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Data used for PPL evaluation: {EVALUATION_DATASET}")
    
    # ================================================================
    # 5. Measure Latency
    # ================================================================
    samples = tokenizer.encode(samples)
    chunk_tokens = samples[0:PPL_CONTEXT_TOKENS]
    input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
    print(f"[INFO] Input tokens: {input_ids.shape}")

    # Warm-up
    with torch.no_grad():
        out_ids = model(input_ids)
    torch.cuda.synchronize()
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
            out_ids = model(input_ids)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        
        
    # # ================================================================
    # # 5. ARC-Easy evaluation
    # # ================================================================
    NUM_ARC_SAMPLES = None  # Use None for the complete test set
    list_data_set_arc = ["ARC-Easy", "ARC-Challenge"]
    for DATASET_ARC in list_data_set_arc:
        print(f"[INFO] Start {DATASET_ARC} Evaluation... \n")
        
        arc_result = evaluate_arc(
            model=model,
            tokenizer=tokenizer,
            device=device,
            subset=DATASET_ARC,
            max_samples=NUM_ARC_SAMPLES,  # Use None for the complete test set
        )
        print(f"\n{DATASET_ARC} results")
        print(f"Model: Qwen3-{CHOOSE_MODEL}")
        print(f"Number of questions: {arc_result['num_samples']}")
        print(f"Accuracy:            {arc_result['acc'] * 100:.2f}%")
        print(f"Normalized accuracy: {arc_result['acc_norm'] * 100:.2f}%")
    
    # ================================================================
    # 7. PIQA evaluation
    # ================================================================
    NUM_PIQA_SAMPLES = None  # None uses the complete validation set
    print("[INFO] Start PIQA Evaluation...\n")

    piqa_result = evaluate_piqa(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_samples=NUM_PIQA_SAMPLES,
    )

    print("\nPIQA results")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Number of questions: {piqa_result['num_samples']}")
    print(f"Accuracy:            {piqa_result['acc'] * 100:.2f}%")
    print(f"Normalized accuracy: {piqa_result['acc_norm'] * 100:.2f}%")