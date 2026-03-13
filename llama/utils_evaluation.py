import os 
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, context_size, device="cuda"):

    model.eval()  
    total_nll = 0.0
    total_tok = 0

    for txt in texts:
        ids = tokenizer.encode(txt)
        if len(ids) < 2:
            continue

        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        L = ids_t.size(0)

        for t in range(1, L):            
            start = max(0, t - context_size)
            inp = ids_t[start:t].unsqueeze(0)   # [1, w]
            logits = model(inp)                 # [1, w, V]
            last_logits = logits[:, -1, :]    # [1, V]
            target = ids_t[t].view(1)

            loss = F.cross_entropy(last_logits, target, reduction="sum")
            total_nll += float(loss.item())
            total_tok += 1

    corpus_ppl = math.exp(total_nll / max(total_tok, 1))

    return corpus_ppl



def load_wikitext2_samples(n=1_000, min_length=10):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    samples = [x["text"] for x in dataset if len(x["text"].strip()) > min_length]
    if n is not None:
        samples = samples[:n]
    return samples