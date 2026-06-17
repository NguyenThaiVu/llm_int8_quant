"""
In this script, we implement quantization functions.
"""

import os 
import torch


def quantize_tensor(x, dtype=torch.int8, scale_dtype=torch.float32):
    """
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


@torch.inference_mode()
def quantize_row_int8_symmetric_nd(
    mat: torch.Tensor,
    scale_dtype=torch.float32
):
    """
    Symmetric int8 quantization per row along the last dimension.
    """

    assert mat.dim() >= 2, "mat must be at least 2D"
    
    qmin, qmax = -128, 127

    orig_shape = mat.shape  # (..., C)
    last_dim = orig_shape[-1]  # C
    num_vecs = mat.numel() // last_dim

    # Reshape to (num_vecs, C)
    mat_2d = mat.reshape(num_vecs, last_dim)
    max_vals = mat_2d.abs().amax(dim=1, keepdim=True)  # (num_vecs, 1)
    max_vals = max_vals.clamp(min=1e-8)

    # Per-row scale
    scales = (max_vals / qmax).squeeze(1)

    # Quantize
    q_mat_2d = mat_2d / scales.unsqueeze(1)
    q_mat_2d.round_()
    q_mat_2d.clamp_(qmin, qmax)
    q_mat_2d = q_mat_2d.to(torch.int8)

    # Reshape back
    q_mat = q_mat_2d.reshape(orig_shape)
    scales = scales.reshape(orig_shape[:-1])

    return q_mat, scales.to(scale_dtype)



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
    for _ in range(5):
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