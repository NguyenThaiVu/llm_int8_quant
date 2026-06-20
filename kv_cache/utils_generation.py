import os 
import numpy as np
import torch

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
            
    def get_total_cache_size(self):
        total_size = 0
        for layer_cache in self.cache:
            if layer_cache is not None:
                if len(layer_cache) == 2:  # Assuming (key, value) tuple
                    total_size += layer_cache[0].element_size() * layer_cache[0].nelement()  # Key size
                    total_size += layer_cache[1].element_size() * layer_cache[1].nelement()  # Value size
                elif len(layer_cache) == 4: 
                    total_size += layer_cache[0].element_size() * layer_cache[0].nelement()  # Key int8
                    total_size += layer_cache[1].element_size() * layer_cache[1].nelement()  # Key scale
                    total_size += layer_cache[2].element_size() * layer_cache[2].nelement()  # Value int8
                    total_size += layer_cache[3].element_size() * layer_cache[3].nelement()  # Value scale
                else:
                    raise ValueError(f"Unexpected cache format. Got {len(layer_cache)} elements.")
                
        return total_size
            

def generate_text_autoregressive(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    """
    This function generates text autoregressively using KV cache.

    * Args:
    - model: It should return logits with shape:
                (batch_size, sequence_length, vocab_size)

    - token_ids: Input token. Shape: (batch_size, prompt_length)

    - max_new_tokens.

    - eos_token_id.

    - context_size: Optional maximum context length. If provided, the input prompt is
            truncated from the left to keep only the most recent `context_size` tokens.

    * Yields:
    - next_token: generated token at each decoding step. Shape: (batch_size, 1)
    """
    model.eval()

    if context_size is not None:
        token_ids = token_ids[:, -context_size:]

    with torch.no_grad():
        cache = KVCache(n_layers=model.cfg["n_layers"])
        print(f"[INFO] KV cache initialized with {model.cfg['n_layers']} layers.")
        model.reset_kv_cache()

        # Prefill stage
        logits = model(token_ids, cache=cache)

        # Decoding stage
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            # Decode stage: feed only the newest token
            logits = model(next_token, cache=cache)
            
    total_cache_size = cache.get_total_cache_size()
    print(f"\n[INFO] Total KV cache size: {total_cache_size / 1024 / 1024:.2f} MB")
    
    
@torch.inference_mode()
def benchmark_llm_decode(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    context_size=None,
    warmup_decode_tokens=5,
):
    model.eval()

    if context_size is not None:
        token_ids = token_ids[:, -context_size:]

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # ============================================================
    # Prefill / TTFT
    # ============================================================
    torch.cuda.synchronize()

    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
    logits = model(token_ids, cache=cache)
    prefill_end.record()

    torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    decode_times_ms = []
    generated = []

    # First token from prefill logits
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_token)

    if eos_token_id is not None and torch.all(next_token == eos_token_id):
        return {
            "prefill_ms": prefill_ms,
            "decode_times_ms": [],
            "generated_tokens": torch.cat(generated, dim=1),
        }

    # ============================================================
    # Decode loop
    # ============================================================
    for step in range(1, max_new_tokens):
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        logits = model(next_token, cache=cache)

        end.record()
        torch.cuda.synchronize()

        decode_ms = start.elapsed_time(end)

        if step > warmup_decode_tokens:
            decode_times_ms.append(decode_ms)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    generated_tokens = torch.cat(generated, dim=1)

    if len(decode_times_ms) > 0:
        decode_arr = np.array(decode_times_ms)
        decode_mean_ms = decode_arr.mean()
        decode_tokens_per_sec = 1000.0 / decode_mean_ms
        total_time = prefill_ms + sum(decode_times_ms)
    else:
        decode_mean_ms = None
        decode_median_ms = None
        decode_p90_ms = None
        decode_p99_ms = None
        decode_tokens_per_sec = None
        total_time = prefill_ms

    return {
        "prefill_ms": prefill_ms,
        "decode_mean_ms": decode_mean_ms,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "decode_times_ms": decode_times_ms,
        "generated_tokens": generated_tokens,
        "total_time": total_time,
    }