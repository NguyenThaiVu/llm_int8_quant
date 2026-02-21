import os 
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, context_size, device='cuda'):
    """
    Faster PPL: slide windows of length <= context_size and
    score only the last token of each window (which has full left context).
    """
    model_was_training = model.training
    model.eval()

    results = []
    for txt in texts:
        ids = tokenizer.encode(txt)
        if len(ids) < 2:
            results.append({"num_tokens": 0, "nll_sum": 0.0, "ppl": float("nan")})
            continue

        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        nll_sum = 0.0
        tok_cnt = 0

        # We will take windows ending at positions end=1..L-1
        L = ids_t.size(0)
        end = 1
        while end < L:
            start = max(0, end - context_size)        # include up to token end-1
            inp = ids_t[start:end].unsqueeze(0)       # [1, w] (predict token at 'end')
            logits = model(inp)                        # [1, w, V]
            last_logits = logits[:, -1, :]            # prediction for token at 'end'
            target = ids_t[end].view(1)               # [1]
            loss = F.cross_entropy(last_logits, target, reduction="sum")
            nll_sum += float(loss.item())
            tok_cnt += 1

            # Jump ahead by a stride: score roughly one token per window
            # (Tune stride for speed/accuracy trade-off; 1 is exact; larger is faster.)
            stride = max(1, context_size - 1)
            end += stride

        # If we skipped some tail tokens due to large stride, optionally finish them:
        if end - (context_size - 1) < L - 1:
            # exact tail sweep to ensure full coverage
            for t in range(max(1, L - context_size + 1), L):
                start = max(0, t - context_size)
                inp = ids_t[start:t].unsqueeze(0)
                logits = model(inp)
                last_logits = logits[:, -1, :]
                target = ids_t[t].view(1)
                loss = F.cross_entropy(last_logits, target, reduction="sum")
                nll_sum += float(loss.item())
                tok_cnt += 1

        ppl = math.exp(nll_sum / max(tok_cnt, 1))
        results.append({"num_tokens": tok_cnt, "nll_sum": nll_sum, "ppl": ppl})

    total_nll = sum(r["nll_sum"] for r in results)
    total_tok = sum(r["num_tokens"] for r in results) or 1
    corpus_ppl = math.exp(total_nll / total_tok)

    if model_was_training: model.train()
    return results, corpus_ppl


def load_wikitext2_samples(n=1000, min_length=10):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    # Filter out empty or too-short lines
    samples = [x["text"] for x in dataset if len(x["text"].strip()) > min_length]
    return samples[:n]