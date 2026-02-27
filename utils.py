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

    mat = mat.to(scale_dtype)
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


def quantize_row_int8_asymmetric_nd(
    mat: torch.Tensor,
    scale_dtype=torch.float32,
    percentile: float | None = None,
):

    assert mat.dim() >= 2, "mat must be at least 2D"

    mat = mat.to(scale_dtype)

    qmin, qmax = -128, 127
    dtype = torch.int8
    q_range = qmax - qmin  # 255

    orig_shape = mat.shape  # (..., C)
    last_dim = orig_shape[-1]
    num_vecs = mat.numel() // last_dim

    # Reshape to (num_vecs, C)
    mat_2d = mat.reshape(num_vecs, last_dim)

    # 1) Determine per-row range [min_vals, max_vals]
    if percentile is None:
        min_vals, _ = mat_2d.min(dim=1, keepdim=True)  # (N, 1)
        max_vals, _ = mat_2d.max(dim=1, keepdim=True)  # (N, 1)
    else:
        assert 0.0 < percentile <= 1.0, "percentile must be in (0.0, 1.0]"
        # For percentile=0.999, lower_q=0.001, upper_q=0.999
        upper_q = percentile
        lower_q = 1.0 - percentile

        # Handle degenerate case where lower_q <= 0
        lower_q = max(lower_q, 0.0)

        # Per-row lower/upper quantiles
        min_vals = torch.quantile(mat_2d, q=lower_q, dim=1, keepdim=True)
        max_vals = torch.quantile(mat_2d, q=upper_q, dim=1, keepdim=True)

    # Avoid zero/negative ranges
    ranges = (max_vals - min_vals).clamp(min=1e-8)

    # 2) Per-row scale
    scales = (ranges / q_range).squeeze(1)  # (N,)

    # 3) Per-row zero-point (integer)
    zero_points = qmin - (min_vals.squeeze(1) / scales)  # (N,)
    zero_points = torch.round(zero_points).clamp(qmin, qmax).to(torch.int32)

    # 4) Optional: clip in float space to [min_vals, max_vals]
    mat_2d_clipped = mat_2d.clamp(min_vals, max_vals)

    # 5) Quantize: q = round(x / scale + zero_point)
    q_mat_2d = torch.round(
        mat_2d_clipped / scales.unsqueeze(1) + zero_points.unsqueeze(1)
    )
    q_mat_2d = torch.clamp(q_mat_2d, qmin, qmax).to(dtype)

    # 6) Reshape back
    q_mat = q_mat_2d.reshape(orig_shape)
    scales = scales.reshape(orig_shape[:-1])
    zero_points = zero_points.reshape(orig_shape[:-1])

    return q_mat, scales.to(scale_dtype), zero_points


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
    