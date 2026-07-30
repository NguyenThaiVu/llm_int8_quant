"""
This script demonstrates how to convert a Qwen3 model to quantization model 
using LLM.int8() technique. 
"""
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import torch
torch.manual_seed(123)
import torch.nn as nn
from torch.nn import functional as F

import json
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config, load_weights_into_qwen
from utils_model import *
from utils_generation import *
from utils_evaluation import load_wikitext_single_text, compute_ppl_single_text
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "4B"  # Options: "4B", "8B", "14B"

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
    EVALUATION_DATASET = "wikitext-2"  # Options: "wikitext-2", "wikitext-103"
    PPL_STRIDE = PPL_CONTEXT_TOKENS // 2

    # list_prompt = ["What is the capital of VietNam?",\
    #                 "What is the Dragon Ball story?"]
    list_prompt = ["What is the capital of VietNam?"]

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
        
    # ================================================================
    # 4. Quantization to LLM.int8() format and evaluation
    # ================================================================
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
    print(f"[INFO] Converting model to LLM.int8() format...")
    int8_model = convert_linear_to_llm_int8(model, threshold=6.0)
    print(f"[INFO] Model converted to LLM.int8() format successfully.")
    
    # Delete the original model to free up memory
    del model
    torch.cuda.empty_cache()
    
    int8_model = int8_model.to(0) # Quantization happens here
    print(f"\n[INFO] Model converted to LLM.int8() format successfully.\n")
    print(f"Sample Weight after quantization: {int8_model.trf_blocks[0].att.W_query.weight.data[:5, :5]}")

    for idx, prompt in enumerate(list_prompt):
        input_token_ids = tokenizer.encode(prompt)
        input_token_ids_tensor = torch.tensor(input_token_ids, device=device)

        generated_text = func_generate_text(
            model=int8_model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id
        )

        response = get_clean_generated_text(generated_text, tokenizer)
        print(f"{idx}. Generated response: {response} \n")
        
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
    print(f"Model Information: ")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Data used for PPL evaluation: {EVALUATION_DATASET}")


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
    
    
    