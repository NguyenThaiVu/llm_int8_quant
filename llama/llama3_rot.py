import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import time
from safetensors.torch import load_file
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

from config import get_llama_config
from tokenizer import Tokenizer
from utils_weight import load_weights_into_llama    
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text

from utils_quant import *
from utils_model_quan import compute_smooth_alpha, PerChannelAbsMaxObserver
from utils_model import compute_rope_params, apply_rope
import gemm_cutlass

from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text, evaluate_arc, evaluate_piqa
from utils_power import measure_power

import gemm_cutlass
    
    
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
    
    def forward(self, x, is_apply_rotation=True):
        if self.is_quantized == False:
            y = x @ self.weight.T
        else:
            if is_apply_rotation:
                x_rot = gemm_cutlass.func_apply_hadamard(x)
                x_rot_q, scale_x = gemm_cutlass.func_quantize_i8(x_rot)
                y = gemm_cutlass.func_w8a8_matmul(x_rot_q, self.weight_q, scale_x, self.scale_w)
            else:
                (x_q, scale_x) = x 
                assert x_q.dtype == torch.int8, "Input tensor must be quantized to int8."
                y = gemm_cutlass.func_w8a8_matmul(x_q, self.weight_q, scale_x, self.scale_w)
            
        return y.to(dtype=self.dtype)
    
    def finish_calibration(self):
        """
        In this function, we: 
        - Apply Hadamard rotation to the weight matrix.
        - Quantize the rotated weight matrix to int8.
        """
        w_rot = gemm_cutlass.func_apply_hadamard(self.weight)
        w_rot_q, scale_w = gemm_cutlass.func_quantize_i8(w_rot)
        self.weight_q = w_rot_q
        self.scale_w = scale_w
        
        self.is_quantized = True
        
        # Delete bf16 weight and observer to save memory
        del self.weight


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Rot_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = Rot_Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = Rot_Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.is_quantized = False

    def forward(self, x):
        if self.is_quantized == False:
            x_fc1 = self.fc1(x)
            x_fc2 = self.fc2(x)
        else:
            x_rot = gemm_cutlass.func_apply_hadamard(x)
            x_rot_q, scale_x = gemm_cutlass.func_quantize_i8(x_rot)
            x_fc1 = self.fc1((x_rot_q, scale_x), is_apply_rotation=False)
            x_fc2 = self.fc2((x_rot_q, scale_x), is_apply_rotation=False)
            
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
    def finish_calibration(self):
        self.fc1.finish_calibration()
        self.fc2.finish_calibration()
        self.fc3.finish_calibration()
        self.is_quantized = True
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

        self.W_query = Rot_Linear(d_in, self.d_out, dtype=dtype, bias=False)
        self.W_key = Rot_Linear(d_in, num_kv_groups * head_dim, dtype=dtype, bias=False)
        self.W_value = Rot_Linear(d_in, num_kv_groups * head_dim, dtype=dtype, bias=False)  
        self.out_proj = Rot_Linear(self.d_out, d_in, dtype=dtype, bias=False)
        
        self.is_quantized = False

    def forward(self, x, mask, cos, sin):
        num_tokens, _ = x.shape

        # 1. QKV projections
        if self.is_quantized == False:
            queries = self.W_query(x)  # (num_tokens, num_heads * head_dim)
            keys = self.W_key(x)       # (num_tokens, num_kv_groups * head_dim)
            values = self.W_value(x)   # (num_tokens, num_kv_groups * head_dim)
        else:
            x_rot = gemm_cutlass.func_apply_hadamard(x)
            x_rot_q, scale_x = gemm_cutlass.func_quantize_i8(x_rot)
            queries = self.W_query((x_rot_q, scale_x), is_apply_rotation=False) 
            keys = self.W_key((x_rot_q, scale_x), is_apply_rotation=False) 
            values = self.W_value((x_rot_q, scale_x), is_apply_rotation=False) 

        # 2. Reshape and transpose for multi-head attention
        queries = queries.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
        keys = keys.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)
        values = values.view(num_tokens, self.num_kv_groups, self.head_dim).transpose(0, 1)

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
            num_kv_groups=cfg["n_kv_groups"],
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
    

# # =================================================
# Quantization
# # =================================================
print("\nCollecting calibration for quantization...")
calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                split="train", n=10_000)
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
model.eval()
    
# # ================================================
# # Evaluation
# # ===============================================

samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

ppl = compute_ppl_single_text(model,
                            tokenizer, 
                            samples,
                            context_size=PPL_CONTEXT_TOKENS,
                            stride=PPL_STRIDE)

print()
print(f"\nPPL (SmoothQuant technique): {ppl} \n")
print(f"Evaluation Information:")
print(f"Model: LLaMA 3.2 {LLAMA_SIZE_STR} {'Instruct' if IS_INSTRUCT else 'Base'}")
print(f"Context size for evaluation: {PPL_CONTEXT_TOKENS}")
print(f"Dataset: {EVALUATION_DATASET}")


# ===============================================
# Measure Latency
# ===============================================
samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)
samples = tokenizer.encode(samples)
chunk_tokens = samples[0:PPL_CONTEXT_TOKENS]
input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
print(f"[INFO] Input tokens: {input_ids.shape}")

# Warm-up 
for _ in range(5):
    with torch.no_grad():
        _ = model(input_ids)
torch.cuda.synchronize()    

# Measure latency 
n_iter = 10
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
with torch.no_grad():
    for _ in range(n_iter):
        out_ids = model(input_ids)
end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
avg_latency_ms = elapsed_time_ms / n_iter
print(f"[INFO] Average model latency over {n_iter} iterations: {avg_latency_ms:.4f} ms.")

# Profiler to measure latency and memory usage
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
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

# Measure power consumption
measure_power(model, input_ids, n_iterations=10)

# # ===============================================
# # 5. ARC-Easy evaluation
# # ================================================================
# NUM_ARC_SAMPLES = None  # Use None for the complete test set
# list_data_set_arc = ["ARC-Easy", "ARC-Challenge"]
# for DATASET_ARC in list_data_set_arc:
#     print(f"[INFO] Start {DATASET_ARC} Evaluation... \n")
    
#     arc_result = evaluate_arc(
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         subset=DATASET_ARC,
#         max_samples=NUM_ARC_SAMPLES,  # Use None for the complete test set
#     )
#     print(f"\n{DATASET_ARC} results")
#     print(f"Model: Llama-3.2-{LLAMA_SIZE_STR}")
#     print(f"Number of questions: {arc_result['num_samples']}")
#     print(f"Accuracy:            {arc_result['acc'] * 100:.2f}%")
#     print(f"Normalized accuracy: {arc_result['acc_norm'] * 100:.2f}%")

# # ================================================================
# # 7. PIQA evaluation
# # ================================================================
# NUM_PIQA_SAMPLES = None  # None uses the complete validation set
# print("[INFO] Start PIQA Evaluation...\n")

# piqa_result = evaluate_piqa(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     max_samples=NUM_PIQA_SAMPLES,
# )

# print("\nPIQA results")
# print(f"Model: Llama-3.2-{LLAMA_SIZE_STR}")
# print(f"Number of questions: {piqa_result['num_samples']}")
# print(f"Accuracy:            {piqa_result['acc'] * 100:.2f}%")
# print(f"Normalized accuracy: {piqa_result['acc_norm'] * 100:.2f}%")
