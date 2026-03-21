import torch
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    w8a8_block_int8_matmul,
)

# Example shapes:
# A: [M, K]
# B: [N, K]   <-- note: vLLM expects B shaped (N, K), not (K, N)
M, K, N = 2048, 4096, 8192
block_n, block_k = 128, 128

# INT8 activations and weights
A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")

# Scales:
# As is per-token-group for A, shape [M, ceil(K / block_k)]
# Bs is per-block for B, shape [ceil(N / block_n), ceil(K / block_k)]
As = torch.ones((M, (K + block_k - 1) // block_k), dtype=torch.float32, device="cuda")
Bs = torch.ones(((N + block_n - 1) // block_n, (K + block_k - 1) // block_k),
                dtype=torch.float32, device="cuda")

# Output is usually fp16/bf16, even though inputs are int8
C = w8a8_block_int8_matmul(
    A=A,
    B=B,
    As=As,
    Bs=Bs,
    block_size=[block_n, block_k],
    output_dtype=torch.float16,
)

print(C.shape)   # [M, N]
print(C.dtype)   # torch.float16