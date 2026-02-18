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


def quantize_tensor_batched(x, dtype=torch.int8):
    """
    Quantize a batched tensor and return per-batch scaling factors.
    """
    q_min, q_max = -128, 127  # for int8
    scales = x.abs().amax(dim=[1,2], keepdim=True) / q_max  # (B, 1, 1)
    x_q = torch.clamp((x / scales).round(), q_min, q_max).to(dtype)
    return x_q, scales.squeeze()
    
    
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


def quantize_row_int8_symmetric_nd(mat: torch.Tensor, scale_dtype=torch.float32):
    """
    Symmetric int8 quantization per row along the last dimension.

    mat: float tensor of shape (..., C), with dim >= 2
         e.g. (N, M) or (B, T, C) or (D0, D1, ..., C)

    Returns:
      q_mat: same shape as mat, int8
      scales: shape (...,), same leading dims as mat without the last dim
              e.g. (N,) for (N, M)
                    (B, T) for (B, T, C)
    """
    assert mat.dim() >= 2, "mat must be at least 2D"

    qmin, qmax = -128, 127

    orig_shape = mat.shape          # (..., C)
    last_dim = orig_shape[-1]       # C
    num_vecs = mat.numel() // last_dim

    # Reshape to (num_vecs, C)
    mat_2d = mat.reshape(num_vecs, last_dim)

    # Per-row max abs
    max_vals = mat_2d.abs().amax(dim=1, keepdim=True)   # (num_vecs, 1)
    max_vals = max_vals.clamp(min=1e-8)

    # Per-row scale
    scales = (max_vals / qmax).squeeze(1)               # (num_vecs,)

    # Quantize
    q_mat_2d = torch.clamp(
        torch.round(mat_2d / scales.unsqueeze(1)),
        qmin,
        qmax,
    ).to(torch.int8)                                    # (num_vecs, C)

    # Reshape back
    q_mat = q_mat_2d.reshape(orig_shape)                # (..., C)
    scales = scales.reshape(orig_shape[:-1])            # (...)

    return q_mat, scales.to(scale_dtype)


def dequantize_row_int8_symmetric(q_mat: torch.Tensor, scales: torch.Tensor,\
                            out_dtype=torch.bfloat16):
    """
    Dequantize a row-wise symmetric int8 quantized matrix.
    q_mat: (N, M) int8
    scales: (N,) float32
    Returns:
      mat: (N, M) float32
    """
    mat = q_mat.float() * scales.unsqueeze(1)
    return mat.to(out_dtype)


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
    