import os 
from pathlib import Path
from safetensors.torch import load_file

import torch
torch.manual_seed(123)
import torch.nn as nn
from huggingface_hub import hf_hub_download

from model_utils import Llama3Model
from config import get_llama_config
from tokenizer import Tokenizer
from weight_utils import load_weights_into_llama    
from generation_utils import *


LLAMA_SIZE_STR = "3B" # "1B" or "3B"
LLAMA32_CONFIG = get_llama_config(LLAMA_SIZE_STR)

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

PROMPT = "What is the capital of Vietnam?"

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=150,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=1,
    temperature=0.
)

output_text = token_ids_to_text(token_ids, tokenizer)


print("\n\nOutput text:\n\n", clean_text(output_text))