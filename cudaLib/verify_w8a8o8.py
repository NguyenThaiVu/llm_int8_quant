import math
import torch
from utils_quant import *
import custom_ops


if __name__ == "__main__":

    d_type = torch.bfloat16
    device = torch.device("cuda:0")
    # M, N, K = 2048, 2048, 2048
    M, N, K = 2048, 5120, 8192

    # ========================================================
    # 1. Initialize weights (fixed for calibration and inference)
    W = torch.randn((N, K), dtype=d_type, device=device)
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    W_i8, scale_W = quantize_row_int8_symmetric_nd(W)

    # ========================================================
    # 2. Calibration
    X_calib = torch.randn((M, K), dtype=d_type, device=device)
    Y_calib = torch.matmul(X_calib, W.T)

    _, scale_Y_calib = quantize_row_int8_symmetric_nd(Y_calib)

    # ========================================================
    # 3. New inference inputs
    X = torch.randn((M, K), dtype=d_type, device=device)

    Y_ref = torch.matmul(X, W.T).float()
    
    torch_time = measure_time(torch.matmul, X, W.T)
    print(f"PyTorch matmul time: {torch_time:.6f} seconds")

    # ========================================================
    # 5. INT8 GEMM
    X_i8, scale_X = quantize_row_int8_symmetric_nd(X)
    
    row_scale = scale_X / scale_Y_calib
    col_scale = scale_W

    Y_i8_fused = custom_ops.w8a8o8_matmul_fusion(X_i8, W_i8, row_scale, col_scale)
    Y_fused = Y_i8_fused.float() * scale_Y_calib.unsqueeze(-1)
    
    our_time = measure_time(custom_ops.w8a8o8_matmul_fusion, X_i8, W_i8, row_scale, col_scale)
    print(f"Our Kernel fused W8A8O8 GEMM time: {our_time:.6f} seconds")
    print(f"Speedup: {torch_time / our_time:.2f}x")

    # ========================================================
    # 6. Compare results
    max_diff = (Y_ref - Y_fused).abs().max()
    print(f"Max difference: {max_diff:.6f}")
    mse = ((Y_ref - Y_fused) ** 2).mean().item()
    print(f"MSE: {mse:.6f}")
    
    print(f"Y_torch: {Y_ref[:5, :5]}")
    print(f"Y_fused: {Y_fused[:5, :5]}")
    
    
    # ========================================================
    # 7. W8A8 latency
    w8a8_time = measure_time(custom_ops.w8a8_matmul, X_i8, W_i8, scale_X, scale_W)
    print(f"Our Kernel W8A8 GEMM time: {w8a8_time:.6f} seconds")