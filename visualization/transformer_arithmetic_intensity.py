import torch
import pandas as pd


def dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def memory_bytes(num_elements, dtype):
    return num_elements * dtype_size(dtype)


def add_result(results, name, operations, memory_access):
    results.append({
        "Layer Name": name,
        "Operations": operations,
        "Memory Access": memory_access,
        "Arithmetic Intensity": operations / memory_access,
    })


def add_matmul(
    results,
    name,
    m,
    k,
    n,
    dtype,
):
    """
    Y[M, N] = X[M, K] @ W[K, N]
    """

    operations = 2 * m * k * n

    memory_access = (
        memory_bytes(m * k, dtype)       # input
        + memory_bytes(k * n, dtype)     # weight
        + memory_bytes(m * n, dtype)     # output
    )

    add_result(
        results,
        name,
        operations,
        memory_access,
    )


def analyze_gqa_transformer_block(
    batch_size=1,
    query_len=2048,
    kv_len=2048,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=11008,
    dtype=torch.bfloat16,
):
    """
    Analytical arithmetic-intensity estimate for one
    LLaMA/Qwen-style Transformer block using GQA.

    Prefill:
        query_len = kv_len = prompt length

    Decode:
        query_len = 1
        kv_len = current KV-cache length
    """

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads"
        )

    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by "
            "num_key_value_heads"
        )

    results = []

    head_dim = hidden_size // num_attention_heads
    group_size = num_attention_heads // num_key_value_heads

    # K/V projection output dimension under GQA.
    kv_hidden_size = num_key_value_heads * head_dim

    query_tokens = batch_size * query_len

    # =========================================================
    # 1. Attention RMSNorm
    # =========================================================

    norm_elements = query_tokens * hidden_size

    # Approximate convention:
    # square, reduction, scaling, epsilon add,
    # sqrt/rsqrt, normalization, gamma multiplication.
    norm_operations = 7 * norm_elements

    norm_memory = (
        memory_bytes(norm_elements, dtype)
        + memory_bytes(hidden_size, dtype)
        + memory_bytes(norm_elements, dtype)
    )

    add_result(
        results,
        "attention_norm",
        norm_operations,
        norm_memory,
    )

    # =========================================================
    # 2. Q, K and V projections
    # =========================================================

    # Q: hidden_size -> num_attention_heads * head_dim
    add_matmul(
        results,
        "q_proj",
        m=query_tokens,
        k=hidden_size,
        n=hidden_size,
        dtype=dtype,
    )

    # K: hidden_size -> num_key_value_heads * head_dim
    add_matmul(
        results,
        "k_proj",
        m=query_tokens,
        k=hidden_size,
        n=kv_hidden_size,
        dtype=dtype,
    )

    # V: hidden_size -> num_key_value_heads * head_dim
    add_matmul(
        results,
        "v_proj",
        m=query_tokens,
        k=hidden_size,
        n=kv_hidden_size,
        dtype=dtype,
    )

    # =========================================================
    # 3. QK^T attention
    # =========================================================

    q_elements = (
        batch_size
        * num_attention_heads
        * query_len
        * head_dim
    )

    # Physical K-cache size: only H_kv heads are stored.
    k_cache_elements = (
        batch_size
        * num_key_value_heads
        * kv_len
        * head_dim
    )

    score_elements = (
        batch_size
        * num_attention_heads
        * query_len
        * kv_len
    )

    # Every query head still computes against one associated KV head.
    qk_operations = (
        2
        * batch_size
        * num_attention_heads
        * query_len
        * kv_len
        * head_dim
    )

    # Assume a true GQA kernel:
    # K is stored/read as H_kv heads rather than physically duplicated.
    qk_memory = (
        memory_bytes(q_elements, dtype)
        + memory_bytes(k_cache_elements, dtype)
        + memory_bytes(score_elements, dtype)
    )

    add_result(
        results,
        "qk_matmul_gqa",
        qk_operations,
        qk_memory,
    )

    # =========================================================
    # 4. Attention scaling
    # =========================================================

    attention_scale_operations = score_elements

    attention_scale_memory = (
        memory_bytes(score_elements, dtype)
        + memory_bytes(score_elements, dtype)
    )

    add_result(
        results,
        "attention_scale",
        attention_scale_operations,
        attention_scale_memory,
    )

    # =========================================================
    # 5. Softmax
    # =========================================================

    # max, subtract, exp, sum, divide
    softmax_operations = 5 * score_elements

    softmax_memory = (
        memory_bytes(score_elements, dtype)
        + memory_bytes(score_elements, dtype)
    )

    add_result(
        results,
        "softmax",
        softmax_operations,
        softmax_memory,
    )

    # =========================================================
    # 6. Attention probabilities times V
    # =========================================================

    v_cache_elements = (
        batch_size
        * num_key_value_heads
        * kv_len
        * head_dim
    )

    attention_output_elements = (
        batch_size
        * num_attention_heads
        * query_len
        * head_dim
    )

    sv_operations = (
        2
        * batch_size
        * num_attention_heads
        * query_len
        * kv_len
        * head_dim
    )

    sv_memory = (
        memory_bytes(score_elements, dtype)
        + memory_bytes(v_cache_elements, dtype)
        + memory_bytes(attention_output_elements, dtype)
    )

    add_result(
        results,
        "sv_matmul_gqa",
        sv_operations,
        sv_memory,
    )

    # =========================================================
    # 7. Attention output projection
    # =========================================================

    add_matmul(
        results,
        "o_proj",
        m=query_tokens,
        k=hidden_size,
        n=hidden_size,
        dtype=dtype,
    )

    # =========================================================
    # 8. Attention residual addition
    # =========================================================

    residual_elements = query_tokens * hidden_size

    residual_operations = residual_elements

    # Read two operands and write one result.
    residual_memory = 3 * memory_bytes(
        residual_elements,
        dtype,
    )

    add_result(
        results,
        "attention_residual_add",
        residual_operations,
        residual_memory,
    )

    # =========================================================
    # 9. MLP RMSNorm
    # =========================================================

    add_result(
        results,
        "mlp_norm",
        norm_operations,
        norm_memory,
    )

    # =========================================================
    # 10. Gate and up projections
    # =========================================================

    add_matmul(
        results,
        "gate_proj",
        m=query_tokens,
        k=hidden_size,
        n=intermediate_size,
        dtype=dtype,
    )

    add_matmul(
        results,
        "up_proj",
        m=query_tokens,
        k=hidden_size,
        n=intermediate_size,
        dtype=dtype,
    )

    # =========================================================
    # 11. SiLU
    # =========================================================

    mlp_elements = query_tokens * intermediate_size

    # Approximate:
    # exp, add, reciprocal/divide, multiplication.
    silu_operations = 4 * mlp_elements

    silu_memory = (
        memory_bytes(mlp_elements, dtype)
        + memory_bytes(mlp_elements, dtype)
    )

    add_result(
        results,
        "silu",
        silu_operations,
        silu_memory,
    )

    # =========================================================
    # 12. Gated element-wise multiplication
    # =========================================================

    gate_mul_operations = mlp_elements

    # Read gate, read up, write result.
    gate_mul_memory = 3 * memory_bytes(
        mlp_elements,
        dtype,
    )

    add_result(
        results,
        "gate_mul",
        gate_mul_operations,
        gate_mul_memory,
    )

    # =========================================================
    # 13. Down projection
    # =========================================================

    add_matmul(
        results,
        "down_proj",
        m=query_tokens,
        k=intermediate_size,
        n=hidden_size,
        dtype=dtype,
    )

    # =========================================================
    # 14. MLP residual addition
    # =========================================================

    add_result(
        results,
        "mlp_residual_add",
        residual_operations,
        residual_memory,
    )

    metadata = {
        "head_dim": head_dim,
        "group_size": group_size,
        "kv_hidden_size": kv_hidden_size,
    }

    return results, metadata


def create_table(results):
    df = pd.DataFrame(results)

    df["Operations (GOP)"] = df["Operations"] / 1e9
    df["Memory Access (MiB)"] = (
        df["Memory Access"] / (1024 ** 2)
    )

    return df[
        [
            "Layer Name",
            "Operations (GOP)",
            "Memory Access (MiB)",
            "Arithmetic Intensity",
        ]
    ]


if __name__ == "__main__":
    dtype = torch.bfloat16

    # Example GQA configuration:
    #
    # 32 query heads
    # 8 KV heads
    # 4 query heads share each KV head
    hidden_size = 5120
    num_attention_heads = 40
    num_key_value_heads = 8
    intermediate_size = 17408

    # =========================================================
    # Prefill
    # =========================================================

    prefill_results, metadata = analyze_gqa_transformer_block(
        batch_size=1,
        query_len=2048,
        kv_len=2048,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        dtype=dtype,
    )

    print("GQA configuration")
    print(f"  Head dimension:       {metadata['head_dim']}")
    print(f"  Query heads / KV head:{metadata['group_size']}")
    print(f"  KV hidden size:       {metadata['kv_hidden_size']}")

    print("\n================ PREFILL ================\n")

    prefill_table = create_table(prefill_results)

    print(
        prefill_table.to_string(
            index=False,
            formatters={
                "Operations (GOP)": "{:.3f}".format,
                "Memory Access (MiB)": "{:.3f}".format,
                "Arithmetic Intensity": "{:.3f}".format,
            },
        )
    )

    # =========================================================
    # Decode: one new query token with a 2048-token KV cache
    # =========================================================

    decode_results, _ = analyze_gqa_transformer_block(
        batch_size=1,
        query_len=1,
        kv_len=2048,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        dtype=dtype,
    )

    print("\n================ DECODE ================\n")

    decode_table = create_table(decode_results)

    print(
        decode_table.to_string(
            index=False,
            formatters={
                "Operations (GOP)": "{:.6f}".format,
                "Memory Access (MiB)": "{:.3f}".format,
                "Arithmetic Intensity": "{:.3f}".format,
            },
        )
    )