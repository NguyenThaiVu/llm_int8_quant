import os
import math
import torch 
from utils_quant import *
import gemm_cutlass

def init_X_with_outliers(M, K, device="cuda", dtype=torch.bfloat16):
    """
    Initialize a matrix X of size M x K with random values and introduce outliers.
    """
    X = torch.randn((M, K), dtype=dtype, device=device)
    
    # Introduce outliers 
    num_outliers = 10
    outlier_value = 100.0  # A large value to simulate outliers
    
    for _ in range(num_outliers):
        row_idx = torch.randint(0, M, (1,)).item()
        col_idx = torch.randint(0, K, (1,)).item()
        X[row_idx, col_idx] = outlier_value
    
    return X

def create_hadamard_normalized_matrix(n, device="cuda", dtype=torch.bfloat16):
    assert n > 0 and (n & (n - 1)) == 0

    H = torch.tensor([[1.0]], dtype=torch.float32)

    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)

    H /= math.sqrt(n)

    return H.to(device=device, dtype=dtype)



def apply_block_hadamard_rotate(x, H, block_size=256):
    """
    x: [..., K]
    Apply Hadamard rotation independently to blocks
    along the last dimension.
    """

    K = x.shape[-1]
    assert K % block_size == 0
    assert block_size > 0 and (block_size & (block_size - 1)) == 0
    assert H.shape[0] == block_size and H.shape[1] == block_size

    # [..., K] -> [..., num_blocks, block_size]
    x_block = x.reshape(*x.shape[:-1], K // block_size, block_size)

    x_rot = x_block @ H
    return x_rot.reshape_as(x)



if __name__ == "__main__":
    d_type = torch.bfloat16
    M = 2048
    N = 4096
    K = 8192
    
    X = init_X_with_outliers(M, K, device="cuda", dtype=d_type)
    # W = torch.empty((N, K), dtype=d_type).cuda()
    # torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    W = init_X_with_outliers(N, K, device="cuda", dtype=d_type)
    
    # ============================================================
    # 1. Float computation
    Y = X @ W.T
    torch_time = measure_time(torch.matmul, X, W.T)
    print(f"Time for torch.matmul: {torch_time:.6f} ms")
    
    # ============================================================
    # 2. Quantization without rotation
    X_i8, scale_x = quantize_row_int8_symmetric_nd(X)
    W_i8, scale_w = quantize_row_int8_symmetric_nd(W)
    Y_deq = gemm_cutlass.func_w8a8_matmul(X_i8, W_i8, scale_x, scale_w)
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2)
    print(f"MSE: {mse.item()}\n")
    
    # ============================================================
    # 2. Quantization with rotation
    # R = create_hadamard_normalized_matrix(K)
    
    H = create_hadamard_normalized_matrix(256)
    
    # # X_rot = apply_block_hadamard_rotate(X, H, block_size=256)
    # # W_rot = apply_block_hadamard_rotate(W, H, block_size=256)
    # X_rot = gemm_cutlass.func_apply_hadamard(X)
    # W_rot = gemm_cutlass.func_apply_hadamard(W)
    # # Y_hadamard = X_rotated @ W_rotated
    
    # # Quantize the rotated matrices
    # X_rot_i8, scale_x_rot = quantize_row_int8_symmetric_nd(X_rot)
    # W_rot_i8, scale_w_rot = quantize_row_int8_symmetric_nd(W_rot)
    X_rot = apply_block_hadamard_rotate(X, H)
    X_rot_i8, scale_x_rot = quantize_row_int8_symmetric_nd(X_rot)
    
    W_rot = apply_block_hadamard_rotate(W, H)
    W_rot_i8, scale_w_rot = quantize_row_int8_symmetric_nd(W_rot)
    
    Y_rot_deq = gemm_cutlass.func_w8a8_matmul(X_rot_i8, W_rot_i8, scale_x_rot, scale_w_rot)
    
    max_diff_rotated = torch.max(torch.abs(Y - Y_rot_deq))
    print(f"Max difference: {max_diff_rotated.item()}")
    mse_rotated = torch.mean((Y - Y_rot_deq) ** 2)
    print(f"MSE: {mse_rotated.item()}\n")
    
    # ============================================================
    # Measure time
    print(f"Measure time for Hadamard rotation + quantization + i8 matmul")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    n_iter = 1_000
    # warm up
    for _ in range(10):
        _ = gemm_cutlass.func_apply_hadamard(X)
    
    start_event.record()
    for _ in range(n_iter):
        X_rot = gemm_cutlass.func_apply_hadamard(X)
        X_rot_i8, scale_x_rot = gemm_cutlass.func_quantize_i8(X_rot)
        Y_rot_deq = gemm_cutlass.func_w8a8_matmul(X_rot_i8, W_rot_i8, scale_x_rot, scale_w_rot)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / n_iter
    print(f"Time for Hadamard rotation + quantization + i8 matmul: {elapsed_time_ms:.6f} ms")
    
    speed_up = torch_time / elapsed_time_ms
    print(f"Speed up (Rotated i8 vs torch): {speed_up:.2f}x")
    
    
    
    
