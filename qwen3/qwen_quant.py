import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # select GPU "0", "1", "2",...
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
from utils_evaluation import evaluate_arc, load_wikitext_single_text, compute_ppl_single_text, evaluate_piqa
from utils_model_quan import TransformerBlock_Quant
from utils_power import *
    
# Select which model to use via the following flag; only one can be True
USE_BASE_MODEL = False
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = True
if (USE_BASE_MODEL + USE_REASONING_MODEL + USE_INSTRUCT_MODEL) != 1:
    raise AttributeError("Only one of the options above can be True.")

CHOOSE_MODEL = "8B"  # Options: "4B", "8B", "14B"

MAX_NEW_TOKENS = PPL_CONTEXT_TOKENS = 2048  # Options: 2048, 4096, 8192
EVALUATION_DATASET = "wikitext-103"  # Options: "wikitext-2", "wikitext-103"
PPL_STRIDE = PPL_CONTEXT_TOKENS // 2


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
    
    if (USE_REASONING_MODEL == True) or (USE_INSTRUCT_MODEL == True):
        repo_id = f"Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen3-{CHOOSE_MODEL}-Base"
    print(f"[INFO] Loading model weights from repo: {repo_id} \n")
        
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
    if (USE_REASONING_MODEL == True) or (USE_INSTRUCT_MODEL == True):
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
    else:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

    tokenizer_file_path = os.path.join(MODEL_HUD_FOLDER, tokenizer_file_path)

    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)

    if (USE_REASONING_MODEL == True) or (USE_INSTRUCT_MODEL == True):
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=USE_REASONING_MODEL)
        
        evaluation_tokenizer = Qwen3Tokenizer(
                tokenizer_file_path=tokenizer_file_path,
                repo_id=repo_id,
                apply_chat_template=False,
                add_generation_prompt=False,
                add_thinking=False
            )
        print(f"[INFO] To avoid chat template, we have a separate tokenizer for evaluation PPL, ARC, PIQA.\n")
    else:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=False,
            add_generation_prompt=False,
            add_thinking=False)
        evaluation_tokenizer = tokenizer

    # ================================================================
    # 3. Quantization Calibration
    # ================================================================
    print("\nCollecting calibration for quantization...")
    calibrate_samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET, split="train", n=10_000)
    
    # calibrate_tokens = tokenizer.encode(calibrate_samples)
    calibrate_tokens = evaluation_tokenizer.encode(calibrate_samples)
    
    print(f"[INFO] Load calibration with {len(calibrate_tokens)} tokens.")
            
    for i in tqdm(range(0, len(calibrate_tokens) - PPL_CONTEXT_TOKENS + 1, PPL_STRIDE)):
        chunk_tokens = calibrate_tokens[i:i + PPL_CONTEXT_TOKENS]

        input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)
            
    model.finish_calibration()
    print(f"[INFO] Finished calibration.")
    
    model.eval()
    model = torch.compile(model)
    print(f"\n[INFO] Model compiled successfully. Start evaluation... \n")

    # ========================================================================
    # 4. Quantization Generation
    # ========================================================================
    print("\n===== Generated text after quantization: =====\n")
    
    en = """
    Last summer, my family visited a small village near the mountains. 
    We spent three days exploring the area, trying local food, and meeting friendly people. 
    Although the weather was sometimes rainy, we really enjoyed our trip.
    """
    
    input_article = """
    In 2025, the city of Paris introduced 50 kilometers of new bicycle lanes to encourage sustainable transportation.
    Following the expansion, bicycle usage increased by 30%, while car traffic in the city center decreased by 12%.
    However, some residents expressed concerns about reduced parking spaces.
    """
    
    list_prompts = [f"Translate the following English text to French: {en}", \
            "Which country has a capital city named Paris?",\
            f"Summarize the article {input_article} in 1 sentence and extract important information into JSON format.",\
            "Describe the Vietnames culture in detail."]
    # list_prompts = ["Describe the Vietnames culture in detail."]

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
    # 5. Measure latency
    # ================================================================
    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET, split="test")
    samples = evaluation_tokenizer.encode(samples)
    chunk_tokens = samples[0:PPL_CONTEXT_TOKENS]
    input_ids = torch.tensor(chunk_tokens, dtype=torch.long, device=device)
    print(f"[INFO] Input tokens: {input_ids.shape}")

    # Warm-up
    with torch.no_grad():
        out_ids = model(input_ids)
    torch.cuda.synchronize()
    print(f"[INFO] Output tokens: {out_ids.shape}")
    
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

    # # ================================================================
    # # 5. PPL evaluation
    # # ================================================================
    samples = load_wikitext_single_text(dataset_name=EVALUATION_DATASET, split="test")
    ppl = compute_ppl_single_text(model,
                                evaluation_tokenizer,
                                samples,
                                context_size=PPL_CONTEXT_TOKENS,
                                stride=PPL_STRIDE)
    print(f"\nPPL: {ppl}")
    print(f"Model Information: ")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Context size: {PPL_CONTEXT_TOKENS}")
    print(f"Dataset evaluation: {EVALUATION_DATASET}")
    
    # ================================================================
    # 6. ARC-Easy evaluation
    # ================================================================
    NUM_ARC_SAMPLES = None  # Use None for the complete test set
    list_data_set_arc = ["ARC-Easy", "ARC-Challenge"]
    for DATASET_ARC in list_data_set_arc:
        print(f"\n[INFO] Start {DATASET_ARC} Evaluation...")
        
        arc_result = evaluate_arc(
            model=model,
            tokenizer=evaluation_tokenizer,
            device=device,
            subset=DATASET_ARC,
            max_samples=NUM_ARC_SAMPLES,  # Use None for the complete test set
        )
        print(f"{DATASET_ARC} results")
        print(f"Model: Qwen3-{CHOOSE_MODEL}")
        print(f"Number of questions: {arc_result['num_samples']}")
        print(f"Accuracy:            {arc_result['acc'] * 100:.2f}%")
        print(f"Normalized accuracy: {arc_result['acc_norm'] * 100:.2f}% \n")
    
    # ================================================================
    # 7. PIQA evaluation
    # ================================================================
    NUM_PIQA_SAMPLES = None  # None uses the complete validation set
    print("[INFO] Start PIQA Evaluation...\n")

    piqa_result = evaluate_piqa(
        model=model,
        tokenizer=evaluation_tokenizer,
        device=device,
        max_samples=NUM_PIQA_SAMPLES,
    )

    print("\nPIQA results")
    print(f"Model: Qwen3-{CHOOSE_MODEL}")
    print(f"Number of questions: {piqa_result['num_samples']}")
    print(f"Accuracy:            {piqa_result['acc'] * 100:.2f}%")
    print(f"Normalized accuracy: {piqa_result['acc_norm'] * 100:.2f}%")