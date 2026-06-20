import os 
import time
from pathlib import Path
from safetensors.torch import load_file

import torch
torch.manual_seed(123)
import torch.nn as nn
from huggingface_hub import hf_hub_download

from utils_model import Llama3Model
from config import get_llama_config
from tokenizer import Tokenizer
from utils_weight import load_weights_into_llama    
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text
from utils_model import print_tensor_memory_all_attrs


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
print(f"[INFO] Using model hub directory: {LOCAL_DIR}")

# ===============================================
# 1. Define Model Architecture
# ===============================================
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
    local_dir=LOCAL_DIR,
    local_files_only=True
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
EVALUATION_DATASET = 'wikitext-2' # "wikitext-2" or "wikitext-103"

list_prompts = ["What is Dragon Ball story?"]

# for prompt in list_prompts:
#     token_ids = generate(
#         model=model,
#         idx=text_to_token_ids(prompt, tokenizer).to(device),
#         max_new_tokens=MAX_GENERATED_TOKENS,
#         context_size=LLAMA32_CONFIG["context_length"],
#         top_k=1,
#     )

#     output_text = token_ids_to_text(token_ids, tokenizer)
#     print("\nResponse:\n", clean_text(output_text))
    
# # ================================================
# # 5. Evaluation
# # ===============================================

# samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)

# ppl = compute_ppl_single_text(model,
#                             tokenizer, 
#                             samples,
#                             context_size=PPL_CONTEXT_TOKENS,
#                             stride=PPL_STRIDE)
# print(f"\nPPL: {ppl} \n")
# print("Model information:")
# print(f"Model: Llama-3.2-{LLAMA_SIZE_STR}")
# print(f"Context size: {PPL_CONTEXT_TOKENS}")

# ===============================================
# Measure Memory usage
# ===============================================
samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)
samples = tokenizer.encode(samples)
chunk_tokens = samples[0:PPL_CONTEXT_TOKENS]
input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
print(f"[INFO] Input tokens: {input_ids.shape}")


time.sleep(1)  
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# Actual measured run
with torch.no_grad():
    out_ids = model(input_ids)
torch.cuda.synchronize()
print(f"[INFO] Output tokens: {out_ids.shape}")

def calc_gpu_gb(x):
    return f"{x / 1024 / 1024 / 1024:.2f} GB"
print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}\n")

total_model_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Total model size: {calc_gpu_gb(total_model_size)}\n")

embed_size = model.tok_emb.weight.numel() * model.tok_emb.weight.element_size()
logit_layer_size = model.out_head.weight.numel() * model.out_head.weight.element_size()
print(f"Embedding layer size: {calc_gpu_gb(embed_size)}")
print(f"Logit layer size: {calc_gpu_gb(logit_layer_size)}\n")

# ===============================================
# Measure Latency
# ===============================================
# Warm-up 
for _ in range(5):
    with torch.no_grad():
        _ = model(input_ids)
torch.cuda.synchronize()    

# Measured iterations
n_iter = 10
start_time = time.time()
for _ in range(n_iter):
    with torch.no_grad():
        _ = model(input_ids)
torch.cuda.synchronize()
end_time = time.time()
avg_latency = (end_time - start_time) / n_iter
print(f"Average latency per run: {avg_latency:.4f} seconds")
