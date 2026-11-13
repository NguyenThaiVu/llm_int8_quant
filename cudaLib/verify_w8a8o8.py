import math
import torch
import gemm_cutlass
from utils_quant import *


if __name__ == "__main__":

    d_type = torch.bfloat16
    device = torch.device("cuda:0")
    # M, N, K = 2048, 2048, 2048
    M, N, K = 1024, 1020, 1024

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

    X_i8, scale_X = quantize_row_int8_symmetric_nd(X)

    # ========================================================
    # 4. Separate INT8 GEMM
    Y_i8_sep, scale_Y_sep = gemm_cutlass.func_w8a8o8_matmul(
        X_i8, W_i8, scale_X, scale_W
    )

    Y_sep = (
        Y_i8_sep.float() * scale_Y_sep.unsqueeze(-1)
    )

    # ========================================================
    # 5. Fused INT8 GEMM
    row_scale = scale_X / scale_Y_calib
    col_scale = scale_W

    Y_i8_fused = gemm_cutlass.func_w8a8o8_matmul_fusion(
        X_i8, W_i8, row_scale, col_scale
    )

    Y_fused = (
        Y_i8_fused.float() * scale_Y_calib.unsqueeze(-1)
    )

    # ========================================================
    # 6. Accuracy evaluation
    def evaluate(name, Y_pred):
        error = Y_ref - Y_pred

        max_abs_err = error.abs().max().item()
        mse = error.square().mean().item()
        rmse = math.sqrt(mse)

        print(f"\n===== {name} =====")
        print(f"Max absolute error: {max_abs_err:.6f}")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")

    evaluate("Separate Version", Y_sep)
    evaluate("Fused Version", Y_fused)

    # ========================================================
    # 7. Compare outputs
    print("\n===== Separate vs Fused =====")

    diff = (Y_sep - Y_fused).abs()

    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")

    # Saturation ratios
    for name, y in [
        ("Separate", Y_i8_sep),
        ("Fused", Y_i8_fused),
    ]:
        saturation = (
            (y == 127) | (y == -127)
        ).float().mean().item()

        print(f"{name} saturation: {saturation * 100:.4f}%")