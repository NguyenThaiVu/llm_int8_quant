import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
from pathlib import Path
from tqdm import tqdm

import torch
torch.manual_seed(123)
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from config import *
from utils_tokenizer import Qwen3Tokenizer
from utils_model import compute_rope_params, RMSNorm
from utils_model_quan import TransformerBlock
from utils_quant import *
from utils_evaluation import load_wikitext_single_text
from utils_generation import generate_text_autoregressive, benchmark_llm_decode


CHOOSE_MODEL = "14B" # Options: "4B", "8B", "14B"

# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
           
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        cos_int8, scale_cos = quantize_tensor(cos)
        sin_int8, scale_sin = quantize_tensor(sin)
        self.register_buffer("cos_int8", cos_int8, persistent=False)
        self.register_buffer("sin_int8", sin_int8, persistent=False)    
        self.register_buffer("scale_cos", scale_cos, persistent=False)
        self.register_buffer("scale_sin", scale_sin, persistent=False)
        
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache
        
        self.is_quantized = False

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        
        if cache is not None:   # Cache mode
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            if self.is_quantized == False:
                mask = torch.triu(torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1)
                mask = mask[pos_start:pos_end, :pos_end]
                mask = mask[None, None, :, :] # Broadcast to batch and heads
            else:
                mask = torch.tril(torch.ones((pos_end, pos_end), dtype=torch.uint8, device=x.device))
                mask = mask[pos_start:pos_end, :pos_end]
        else:  
            # No-cache mode
            pos_start = 0
            if self.is_quantized == False:
                mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),\
                                diagonal=1)
                mask = mask[None, None, :, :] # Broadcast to batch and heads
            else:
                mask = torch.tril(torch.ones((num_tokens, num_tokens),\
                                dtype=torch.uint8, device=x.device))
        
        
        for i, block in enumerate(self.trf_blocks):
            if cache is not None:
                blk_cache = cache.get(i)
            else:
                blk_cache = None

            if self.is_quantized == False:
                x, new_blk_cache = block(x, mask, self.cos, 1.0, self.sin, 1.0, start_pos=pos_start, cache=blk_cache)
            else:
                x, new_blk_cache = block(x, mask, self.cos_int8, self.scale_cos, self.sin_int8, self.scale_sin,
                                        start_pos=pos_start,
                                        cache=blk_cache)
                
            if cache is not None:
                cache.update(i, new_blk_cache)


        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0
        
    def finish_calibration(self):
        for block in self.trf_blocks:
            block.finish_calibration()
        self.is_quantized = True
            

# ========================================================
# 1. Define model configuration 
# ========================================================            
QWEN3_CONFIG = get_model_config(CHOOSE_MODEL)

model = Qwen3Model(QWEN3_CONFIG)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device);
        

# =================================================================
# 2. Load model weights 
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
# 3. Load tokenizer
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
    
# ========================================================
# 4. Text generation with KV cache
# ========================================================
INPUT_PROMPT_LENGTH = 128
MAX_NEW_TOKENS = 128
print("[INFO] INPUT PROMPT LENGTH:", INPUT_PROMPT_LENGTH)
print("[INFO] MAX NEW TOKENS:", MAX_NEW_TOKENS)

prompt = "What is the Dragon Ball story?"
if INPUT_PROMPT_LENGTH is None:
    input_token_ids = tokenizer.encode(prompt)
    input_token_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)
else:
    prompt = prompt * 300
    input_token_ids = tokenizer.encode(prompt)[-INPUT_PROMPT_LENGTH:]
    input_token_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

# Duplicate the input tokens to create a batch (if needed)
from utils_model_quan import BATCH_SIZE
if BATCH_SIZE > 1:
    input_token_tensor = input_token_tensor.expand(BATCH_SIZE, -1)
    print(f"[INFO] Expanded input_token_tensor to batch size {BATCH_SIZE}. New shape: {input_token_tensor.shape} \n")

print(f"[INFO] Shape of input_token_tensor: {input_token_tensor.shape} \n")


END_OFF_TOKEN_ID = None # Option: tokenizer.eos_token_id or None 
# If END_OFF_TOKEN_ID is set to None, the generation will not stop until reaching MAX_NEW_TOKENS. 
# This is useful for benchmarking the maximum generation speed.

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
time.sleep(1)  
generated_tokens = 0

# warm up 
for _ in range(3):
    with torch.no_grad():
        _ = model(input_token_tensor)

result = benchmark_llm_decode(
    model=model,
    token_ids=input_token_tensor,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=END_OFF_TOKEN_ID,          # disable EOS 
    warmup_decode_tokens=5,
)

print(f"Prefill / TTFT: {result['prefill_ms']:.4f} ms")
print(f"Decode mean: {result['decode_mean_ms']:.4f} ms/token")
print(f"Decode throughput: {result['decode_tokens_per_sec']:.2f} tokens/sec")
print(f"Total time: {result['total_time']:.4f} ms\n")
    
# ========================================================
# 4. Calibration for quantization
# ========================================================
PPL_CONTEXT_TOKENS = 128
EVALUATION_DATASET = "wikitext-2"  # Options: "wikitext-2", "wikitext-103"
PPL_STRIDE = PPL_CONTEXT_TOKENS // 2 
N_SAMPLES = 1_000

print("\nCollecting calibration for quantization...")
calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                split="train", n=N_SAMPLES)
calibrate_tokens = tokenizer.encode(calibrate_samples)
print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")
        
for i in tqdm(range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE)):
    chunk_tokens = calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]

    input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = model(input_ids)
        
model.finish_calibration()
print(f"\n[INFO] Finished calibration for quantization.")

model.reset_kv_cache()
model.eval()
model.to(device)

# ========================================================
# 5. Text generation with quantized model
# ========================================================
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
time.sleep(1)

# warm up 
for _ in range(3):
    with torch.no_grad():
        _ = model(input_token_tensor)

result = benchmark_llm_decode(
    model=model,
    token_ids=input_token_tensor,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=END_OFF_TOKEN_ID,          # disable EOS 
    warmup_decode_tokens=5,
)

print(f"Prefill / TTFT: {result['prefill_ms']:.4f} ms")
print(f"Decode mean: {result['decode_mean_ms']:.4f} ms/token")
print(f"Decode throughput: {result['decode_tokens_per_sec']:.2f} tokens/sec")
print(f"Total time: {result['total_time']:.4f} ms\n")
