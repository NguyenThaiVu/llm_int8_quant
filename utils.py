import os 
import numpy as np
import time
import torch

def quantize_tensor(x, dtype=torch.int8, scale_dtype=torch.float32):
    """
    Quantize a tensor to a specified integer type using a scaling factor.

    Parameters:
    x (torch.Tensor): Input tensor to be quantized.
    dtype (torch.dtype): Target data type for quantization.

    Returns:
    torch.Tensor: Quantized tensor.
    scale (float): Scaling factor for quantization.
    """
    q_min, q_max = -128, 127  # for int8
    scale = x.abs().max() / q_max
    x_q = torch.clamp((x / scale).round(), q_min, q_max).to(dtype)
    return x_q, scale.to(scale_dtype)


def quantize_tensor_batched(x, scale_dtype=torch.float32):
    """
    Quantize a batched tensor and return per-batch scaling factors.
    """
    q_min, q_max = -128, 127  # for int8
    scales = x.abs().amax(dim=[1,2], keepdim=True) / q_max  # (B, 1, 1)
    x_q = torch.clamp((x / scales).round(), q_min, q_max).to(torch.int8)
    return x_q, scales.squeeze().to(scale_dtype)
    
    
def quantize_row_wise_tensor(mat: torch.Tensor, scale_dtype=torch.bfloat16):
    """
    Symmetric int8 quantization per row.
    mat: (N, M) float tensor
    Returns:
      q_mat: (N, M) int8
      scales: (N,) bfloat16
    """
    qmin, qmax = -128, 127
    
    max_vals = mat.abs().amax(dim=1, keepdim=True)  # (N, 1)
    max_vals = max_vals.clamp(min=1e-8)

    scales = (max_vals / qmax).squeeze(1)          # (N,)
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(1)), qmin, qmax).to(torch.int8)

    return q_mat, scales.to(scale_dtype)


def quantize_row_int8_symmetric_nd(
    mat: torch.Tensor,
    scale_dtype=torch.float32,
    percentile = None
):
    """
    Symmetric int8 quantization per row along the last dimension.

    If percentile is None, uses strict max(|x|) per row.
    If percentile is a float in (0,1], uses that percentile of |x| per row
    (e.g. 0.999 for 99.9%) for clipping.
    """

    assert mat.dim() >= 2, "mat must be at least 2D"

    # mat = mat.to(scale_dtype)
    
    qmin, qmax = -128, 127

    orig_shape = mat.shape  # (..., C)
    last_dim = orig_shape[-1]  # C
    num_vecs = mat.numel() // last_dim

    # Reshape to (num_vecs, C)
    mat_2d = mat.reshape(num_vecs, last_dim)

    if percentile is None:
        max_vals = mat_2d.abs().amax(dim=1, keepdim=True)  # (num_vecs, 1)
    else:
        assert 0.0 < percentile <= 1.0, "percentile must be in (0.0, 1.0]"
        abs_mat = mat_2d.abs()

        # 99.9% percentile per row
        max_vals = torch.quantile(
            abs_mat,
            q=percentile,
            dim=1,
            keepdim=True
        )

    max_vals = max_vals.clamp(min=1e-8)

    # Per-row scale
    scales = (max_vals / qmax).squeeze(1)

    # Quantize
    q_mat_2d = torch.clamp(
        torch.round(mat_2d / scales.unsqueeze(1)),
        qmin,
        qmax
    ).to(torch.int8)

    # Reshape back
    q_mat = q_mat_2d.reshape(orig_shape)
    scales = scales.reshape(orig_shape[:-1])

    return q_mat, scales.to(scale_dtype)


@torch.no_grad()
def quantize_row_int8_symmetric_nd_chunked(W, alpha=None, chunk_rows=1024, eps=1e-8):
    """
    Row-wise symmetric int8 quantization for W (out_features, in_features).
    Optionally applies per-column alpha scaling on the fly: W_smooth = W * alpha.
    Returns (W_q int8, scale_w float32 [out_features]).
    """
    device = W.device
    out_features, in_features = W.shape

    W_q = torch.empty((out_features, in_features), dtype=torch.int8, device=device)
    scale_w = torch.empty((out_features,), dtype=torch.float32, device=device)

    # Keep alpha on device in float32/float16
    if alpha is not None:
        alpha = alpha.to(device=device, dtype=W.dtype)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        W_chunk = W[start:end, :]  # view

        # Apply smoothing lazily (creates only chunk-sized tensor)
        if alpha is not None:
            W_chunk = W_chunk * alpha.unsqueeze(0)

        # Compute per-row scale: maxabs / 127
        maxabs = W_chunk.abs().amax(dim=1).to(torch.float32).clamp_min(eps)
        s = maxabs / 127.0
        scale_w[start:end] = s

        # Quantize: round(W / s)
        # Use broadcasting with chunk-sized tensor only
        q = torch.round(W_chunk / s.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        W_q[start:end, :] = q

    return W_q, scale_w



def measure_time(func, *args, repeat=100):
    """
    Measure the average execution time of a function over a number of repetitions.

    Parameters:
    func (callable): The function to measure.
    *args: Arguments to pass to the function.
    repeat (int): Number of times to repeat the function call.

    Returns:
    float: Average execution time in milliseconds.
    """
    # warm-up
    for _ in range(10):
        func(*args)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / repeat
    return elapsed_time


def print_memory_layout(tensor, name):
    print(f"Memory layout of {name}:")
    print(f"  Size: {tensor.size()}")
    print(f"  Stride: {tensor.stride()}")
    print(f"  Is contiguous: {tensor.is_contiguous()}")
    print()
    
def get_address_of_tensor(tensor):
    return tensor.data_ptr()


def get_address_element_2d(tensor, index):
    row, col = index
    stride_row, stride_col = tensor.stride()
    base_address = tensor.data_ptr()
    element_address = base_address + (row * stride_row + col * stride_col) * tensor.element_size()
    return element_address

    
def init_random_tensor(shape, dtype=torch.bfloat16, device='cuda'):
    X_1 = torch.randn(shape, dtype=dtype, device=device)
    seed_1 = np.random.randint(0, 5)
    X_2 = torch.randn(shape, dtype=dtype, device=device)
    seed_2 = np.random.randint(0, 5)
    X = (X_1 * seed_1 + X_2 * seed_2) 
    return X
    