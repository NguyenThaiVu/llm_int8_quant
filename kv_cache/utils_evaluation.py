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
            inp = ids_t[start:t]
            logits = model(inp)            
            last_logits = logits[-1, :]    
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


def load_wikitext_single_text(dataset_name="wikitext-2", split="validation", n=None):
    """
    dataset_name: "wikitext-2" or "wikitext-103"
    """

    config = f"{dataset_name}-raw-v1"
    dataset = load_dataset("wikitext", config, split=split)
    
    if n is not None:
        if n < len(dataset):
            dataset = dataset.select(range(n))

    text = "\n".join(x["text"] for x in dataset if x["text"].strip() != "")
    return text


@torch.no_grad()
def compute_ppl_single_text(model, tokenizer, text, context_size, device="cuda", stride=None):
    """
    Compute perplexity on one long text string.

    Args:
        model: causal LM
        tokenizer: tokenizer
        text: single long string
        context_size: max number of input tokens the model can use
        device: "cuda" or "cpu"
        stride: how many new tokens to score per step.
                If None, defaults to context_size.

    Returns:
        float: corpus perplexity
    """
    model.eval()

    if stride is None:
        stride = context_size

    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("inf")

    ids = torch.tensor(ids, dtype=torch.long, device=device)
    seq_len = ids.size(0)

    total_nll = 0.0
    total_tok = 0

    # We score tokens in [start_idx, end_idx), using up to context_size tokens of left context
    for end_idx in range(1, seq_len, stride):
        target_end = min(end_idx + stride, seq_len)

        input_start = max(0, target_end - context_size - 1)
        input_ids = ids[input_start:target_end - 1]
        target_ids = ids[input_start + 1:target_end]

        # Only score the last `trg_len` positions in this window
        trg_len = target_end - end_idx
        mask = torch.full_like(target_ids, -100)
        mask[-trg_len:] = target_ids[-trg_len:]

        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            mask.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        total_nll += loss.item()
        total_tok += trg_len

    return math.exp(total_nll / total_tok)