import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select GPU "0", "1", "2",...
from tqdm import tqdm
import json

import torch
torch.manual_seed(123)
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from utils_tokenizer import Qwen3Tokenizer
from config import get_model_config, load_weights_into_qwen
from utils_model import *
from utils_generation import *
from utils_evaluation import evaluate_arc, load_wikitext_single_text, compute_ppl_single_text
from utils_model_quan import TransformerBlock_Quant
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

CHOOSE_MODEL = "14B"  # Options: "4B", "8B", "14B"


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
    # Change this path to your desired folder to store model weights
    MODEL_HUD_FOLDER_1 = "/sciclone/home/tnguyen10/Desktop/LLM_Quantization/model/"
    MODEL_HUD_FOLDER_2 = "/scratch/tnguyen10/"
    if os.path.exists(MODEL_HUD_FOLDER_1):
        MODEL_HUD_FOLDER = MODEL_HUD_FOLDER_1
    elif os.path.exists(MODEL_HUD_FOLDER_2):
        MODEL_HUD_FOLDER = MODEL_HUD_FOLDER_2
    else:
        raise ValueError("Please update the MODEL_HUD_FOLDER.")
    
    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        repo_id = f"Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen3-{CHOOSE_MODEL}-Base"
    local_dir = os.path.join(MODEL_HUD_FOLDER, repo_id)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Model folder does not exist: {local_dir}")
    else:
        print(f"[INFO] Loading model weights from disk: {local_dir} \n")

    if CHOOSE_MODEL == "0.6B":
        weights_file = os.path.join(local_dir, "model.safetensors")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Missing weights file: {weights_file}")
        else:
            weights_dict = load_file(weights_file)
    else:
        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Missing index file: {index_path}")

        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}

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
    if USE_REASONING_MODEL:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
    else:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

    tokenizer_file_path = os.path.join(MODEL_HUD_FOLDER, tokenizer_file_path)

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
    MAX_NEW_TOKENS = 1024
    PPL_CONTEXT_TOKENS = 1024
    EVALUATION_DATASET = "wikitext-103"  # Options: "wikitext-2", "wikitext-103"
    PPL_STRIDE = PPL_CONTEXT_TOKENS // 2

    list_prompts = ["What is the capital of VietNam?",\
                    "What is the Dragon Ball story?"]

    for idx, prompt in enumerate(list_prompts):
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
    # 4. Quantization
    # ================================================================
    print("\nCollecting calibration for quantization...")
    calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET,
                                                    split="train", n=200_000)
    calibrate_tokens = tokenizer.encode(calibrate_samples)
    print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")
            
    for i in tqdm(range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE)):
        chunk_tokens = calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]

        input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)
            
    model.finish_calibration()
    print(f"[INFO] Finished calibration.")


    # ========================================================================
    # Quantization mode
    print("\n===== Generated text after quantization: =====\n")
    list_prompts = ["What is the capital of VietNam?",\
                    "What is the Dragon Ball story?"]

    for idx, prompt in enumerate(list_prompts):
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
    # 3. Measure latency
    # ================================================================
    model.eval()

    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET)
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
    print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}\n")

    # # ================================================================
    # # 5. PPL evaluation
    # # ================================================================
    # print(f"[INFO] ...Start Evaluation...")

    # ppl = compute_ppl_single_text(model,
    #                             tokenizer,
    #                             samples,
    #                             context_size=PPL_CONTEXT_TOKENS,
    #                             stride=PPL_STRIDE)
    # print(f"PPL: {ppl} \n")
    # print(f"Model Information: ")
    # print(f"Model: Qwen3-{CHOOSE_MODEL}")
    # print(f"Context size: {PPL_CONTEXT_TOKENS}")
    # print(f"Dataset evaluation: {EVALUATION_DATASET}")
    
    # ================================================================
    # 6. ARC-Easy evaluation
    # ================================================================
    NUM_ARC_SAMPLES = None  # Use None for the complete test set
    DATASET_ARC = "ARC-Challenge"  # Options: "ARC-Easy", "ARC-Challenge"
    
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
