"""
In this script, we implement quantization functions.
"""

import os 
import torch
import torch.nn as nn


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
    # Set to evaluation mode if it's a nn.Module
    if isinstance(func, nn.Module):
        func.eval()
    
    # warm-up
    with torch.no_grad():
        for _ in range(5):
            func(*args)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(repeat):
            func(*args)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / repeat 
    return elapsed_time


class MinMaxObserverPerLastDim(nn.Module): 
    def __init__(self, max_batch=1, max_seq_len=1024): 
        super().__init__() 
        self.max_batch = max_batch 
        self.max_seq_len = max_seq_len 
        
        self.register_buffer("max_val",\
                            torch.full((max_batch, max_seq_len), -torch.inf)) 
        self.register_buffer("min_val",\
                            torch.full((max_batch, max_seq_len), torch.inf)) 
        
    @torch.no_grad() 
    def forward(self, x: torch.Tensor): 
        if x.ndim == 2: 
            T, _ = x.shape 
            B = 1 
            xd = x.detach().unsqueeze(0) # (1, T, H) 
        elif x.ndim == 3: 
            B, T, _ = x.shape 
            xd = x.detach() 
        else: raise ValueError(f"Expected 2D or 3D input,\
            got {x.ndim}D with shape {tuple(x.shape)}") 
        
        if B > self.max_batch: 
            raise ValueError(f"B={B} exceeds max_batch={self.max_batch}") 
        if T > self.max_seq_len: 
            raise ValueError(f"T={T} exceeds max_seq_len={self.max_seq_len}") 
        
        # token-wise over hidden dim -> (B, T) 
        cur_max = xd.amax(dim=-1) 
        cur_min = xd.amin(dim=-1) 
        
        # update only prefix [0:B, 0:T) 
        self.max_val[:B, :T] = torch.maximum(self.max_val[:B, :T], cur_max)
        self.min_val[:B, :T] = torch.minimum(self.min_val[:B, :T], cur_min)
        return x 
    
    def get_scale(self, B=None, T=None): 
        qmax = 127.0
        if T == None: 
            T = self.max_seq_len 
        
        # if input 2D, return (T,) scale; 
        if self.max_batch == None or self.max_batch == 1: 
            max_abs = torch.maximum(\
                self.max_val[0, :T].abs(),\
                self.min_val[0, :T].abs()\
            ) 
            return (max_abs / qmax).clamp(min=1e-8) # (T,) 
        
        # if input 3D, return (B,T) scale; 
        max_abs = torch.maximum(\
                self.max_val[:B, :T].abs(),\
                self.min_val[:B, :T].abs()\
        ) 
        return (max_abs / qmax).clamp(min=1e-8) # (B, T)
    

class PerChannelAbsMaxObserver(nn.Module):
    """Tracks max(abs(x)) per last dimension (channel)."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("amax", torch.zeros(num_channels))

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # x: [*, C] -> reduce over all dims except last
        c = x.shape[-1]
        assert c == self.amax.numel()
        
        x_abs = x.abs()
        dims = list(range(x_abs.dim() - 1))
        cur = x_abs.amax(dim=dims) # shape: (C,)
        self.amax = torch.maximum(self.amax, cur)

    def get_amax(self):
        return self.amax

def compute_smooth_alpha(input_observer, weight, lambd=0.5):
    """
    Compute SmoothQuant alpha in full FP32 precision.

    Output:
        alpha: torch.float32 tensor of shape (in_features,)
    """

    # Activation stats
    max_a = input_observer.get_amax().to(torch.float32)  # <-- important
    max_a = torch.clamp(max_a, min=1e-6)

    # Weight stats
    max_w = weight.detach().abs().amax(dim=0).to(torch.float32)
    max_w = torch.clamp(max_w, min=1e-6)

    # Compute alpha
    alpha = torch.pow(max_a, lambd) / torch.pow(max_w, (1.0 - lambd))
    alpha = torch.clamp(alpha, min=0.01, max=100.0)
    return alpha  


