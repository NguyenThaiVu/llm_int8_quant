import os 
import re
import torch 

def func_generate_text(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)
            out = out[-1, :]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            token_ids = torch.cat([token_ids, next_token], dim=0)
    return token_ids
            

def token_ids_to_text(token_ids, tokenizer, batch_dim=False):
    if batch_dim:
        flat = token_ids.squeeze(0)  # remove batch dimension
    else:
        flat = token_ids
    return tokenizer.decode(flat.tolist())

            
            
def get_clean_generated_text(generated_text, tokenizer):
    output_text = token_ids_to_text(generated_text, tokenizer)
    
    # Post-processing to remove incomplete special tokens at the end
    incomplete_special_token_pattern = re.compile(r"<\|[^>]*?$")
    output_text = re.sub(incomplete_special_token_pattern, "", output_text)
    output_text = output_text.strip()
    
    return output_text


def text_to_token_ids(text, tokenizer, batch_dim=False):
    encoded = tokenizer.encode(text)
    if batch_dim:
        # add batch dimension
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  
    else:
        encoded_tensor = torch.tensor(encoded)
    return encoded_tensor




def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[-1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        idx = torch.cat((idx, idx_next), dim=0)  # (num_tokens+1)

    return idx


def generate_batch(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
):
    """
    Batched generation.

    idx:
        [B, T]

    Returns:
        [B, T + max_new_tokens]
    """

    assert idx.dim() == 2, f"Expected idx shape [B, T], got {idx.shape}"

    model.eval()

    B = idx.shape[0]

    # Track which sequences have already produced EOS
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

    for _ in range(max_new_tokens):
        # Correct batch-aware context slicing
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # logits: [B, T, vocab_size]
        # Take only the last token position for every batch item
        logits = logits[:, -1, :]

        # Optional top-k filtering
        if top_k is not None:
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)

            if temperature == 0.0:
                selected = torch.argmax(top_logits, dim=-1, keepdim=True)
                idx_next = top_indices.gather(dim=-1, index=selected)
            else:
                top_logits = top_logits / temperature
                probs = torch.softmax(top_logits, dim=-1)
                selected = torch.multinomial(probs, num_samples=1)
                idx_next = top_indices.gather(dim=-1, index=selected)

        else:
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

        # Handle EOS per batch item
        if eos_id is not None:
            idx_next = torch.where(
                finished.unsqueeze(1),
                torch.full_like(idx_next, eos_id),
                idx_next
            )

            finished = finished | (idx_next.squeeze(1) == eos_id)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id is not None and finished.all():
            break

    return idx

def clean_text(text, start_token="<|start_of_text|>", end_token="<|end_of_text|>"):
    if text is None:
        return text

    start_idx = text.find(start_token)
    end_idx = text.find(end_token)

    if start_idx != -1:
        text = text[start_idx + len(start_token):]

    if end_idx != -1:
        text = text[:text.find(end_token)]
        
    # remove all special tokens like <|eot_id|>
    text = re.sub(r"<\|.*?\|>", "", text)

    return text.strip()