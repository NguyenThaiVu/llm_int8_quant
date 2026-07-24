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


import torch
import torch.nn.functional as F
from datasets import load_dataset


def arc_prompt(example):
    lines = [f"Question: {example['question']}"]

    for label, text in zip(
        example["choices"]["label"],
        example["choices"]["text"],
    ):
        lines.append(f"{label}. {text}")

    lines.append("Answer:")
    return "\n".join(lines)


@torch.inference_mode()
def score_choice(model, tokenizer, prompt, choice, device):
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + " " + choice)

    answer_start = len(prompt_ids)

    input_ids = torch.tensor(
        full_ids,
        dtype=torch.long,
        device=device,
    )

    # logits shape: [sequence_length, vocab_size]
    logits = model(input_ids)

    # logits[t] predicts input_ids[t + 1]
    log_probs = F.log_softmax(
        logits[:-1].float(),
        dim=-1,
    )

    # Targets have shape [sequence_length - 1]
    targets = input_ids[1:]

    # Select the log-probability assigned to each true next token.
    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # The first answer token is predicted at position answer_start - 1.
    answer_log_probs = token_log_probs[answer_start - 1:]

    total_score = answer_log_probs.sum().item()
    normalized_score = answer_log_probs.mean().item()

    return total_score, normalized_score


@torch.inference_mode()
def evaluate_arc(
    model,
    tokenizer,
    device,
    subset="ARC-Easy",
    split="test",
    max_samples=None,
):
    if subset not in {"ARC-Easy", "ARC-Challenge"}:
        raise ValueError(
            "subset must be 'ARC-Easy' or 'ARC-Challenge'"
        )

    dataset = load_dataset(
        "allenai/ai2_arc",
        subset,
        split=split,
    )

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    model.eval()

    correct = 0
    correct_norm = 0

    for i, example in enumerate(dataset):
        prompt = arc_prompt(example)

        labels = list(example["choices"]["label"])
        choices = list(example["choices"]["text"])
        answer_key = str(example["answerKey"])

        scores = [
            score_choice(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice=choice,
                device=device,
            )
            for choice in choices
        ]

        raw_scores = [score[0] for score in scores]
        norm_scores = [score[1] for score in scores]

        raw_index = max(
            range(len(raw_scores)),
            key=raw_scores.__getitem__,
        )
        norm_index = max(
            range(len(norm_scores)),
            key=norm_scores.__getitem__,
        )

        predicted = str(labels[raw_index])
        predicted_norm = str(labels[norm_index])

        correct += int(predicted == answer_key)
        correct_norm += int(predicted_norm == answer_key)
        
    return {
        "task": subset,
        "acc": correct / len(dataset),
        "acc_norm": correct_norm / len(dataset),
        "num_samples": len(dataset),
    }
    
    
# =============================================================================
# PIQA evaluation
# =============================================================================
def piqa_prompt(example):
    return f"Question: {example['goal']}\nAnswer:"

@torch.inference_mode()
def evaluate_piqa(
    model,
    tokenizer,
    device,
    split="validation",
    max_samples=None,
):
    """
    Evaluate a causal language model on PIQA.

    PIQA fields:
        goal:  physical-reasoning question
        sol1:  first candidate solution
        sol2:  second candidate solution
        label: 0 if sol1 is correct, 1 if sol2 is correct

    Returns:
        acc:
            Accuracy using total candidate log-likelihood.

        acc_norm:
            Accuracy using average log-likelihood per candidate token.
    """

    # This is the dataset path currently used by lm-evaluation-harness.
    dataset = load_dataset(
        "baber/piqa",
        split=split,
    )

    if max_samples is not None:
        dataset = dataset.select(
            range(min(max_samples, len(dataset)))
        )

    model.eval()

    correct = 0
    correct_norm = 0

    for i, example in enumerate(dataset):
        prompt = piqa_prompt(example)

        choices = [
            example["sol1"],
            example["sol2"],
        ]

        answer_index = int(example["label"])

        scores = [
            score_choice(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                choice=choice,
                device=device,
            )
            for choice in choices
        ]

        raw_scores = [
            total_score
            for total_score, normalized_score in scores
        ]

        norm_scores = [
            normalized_score
            for total_score, normalized_score in scores
        ]

        predicted_index = max(
            range(len(raw_scores)),
            key=raw_scores.__getitem__,
        )

        predicted_norm_index = max(
            range(len(norm_scores)),
            key=norm_scores.__getitem__,
        )

        correct += int(predicted_index == answer_index)
        correct_norm += int(
            predicted_norm_index == answer_index
        )

    num_samples = len(dataset)

    return {
        "task": "PIQA",
        "acc": correct / num_samples,
        "acc_norm": correct_norm / num_samples,
        "num_samples": num_samples,
    }