import torch
import torch.nn.functional as F


def bench_cuda(fn, iters=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        y = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        y = fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    assert torch.cuda.is_available()

    device = "cuda"
    batch = 2048
    hidden = 4096
    shape = (batch, hidden)
    normalized_shape = (hidden,)

    dtype = torch.bfloat16
    # dtype = torch.float32

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.ones(hidden, device=device, dtype=dtype)
    bias = torch.zeros(hidden, device=device, dtype=dtype)

    eps = 1e-5

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name())
    print("dtype:", dtype)
    print("shape:", shape)

    # Optional: reduce timing noise
    torch.cuda.empty_cache()

    ln_ms = bench_cuda(
        lambda: F.layer_norm(
            x,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    )

    rms_ms = bench_cuda(
        lambda: F.rms_norm(
            x,
            normalized_shape,
            weight=weight,
            eps=eps,
        )
    )

    # Also test LayerNorm without bias, for a fairer comparison with RMSNorm
    ln_no_bias_ms = bench_cuda(
        lambda: F.layer_norm(
            x,
            normalized_shape,
            weight=weight,
            bias=None,
            eps=eps,
        )
    )

    print(f"LayerNorm with bias:    {ln_ms:.6f} ms")
    print(f"LayerNorm without bias: {ln_no_bias_ms:.6f} ms")
    print(f"RMSNorm:                {rms_ms:.6f} ms")

    print()
    print(f"RMSNorm / LayerNorm with bias:    {rms_ms / ln_ms:.3f}x")
    print(f"RMSNorm / LayerNorm without bias: {rms_ms / ln_no_bias_ms:.3f}x")


if __name__ == "__main__":
    main()