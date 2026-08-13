import os 
import math
import torch 
import gemm_cutlass


if __name__ == "__main__":
    dtype = torch.bfloat16
    X = torch.randn(2048, 4096, dtype=dtype, device="cuda")
    W = torch.empty((8192, 4096), dtype=dtype).cuda()
    torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    
    Y = torch.matmul(X, W.T)
    
    X_i8, X_scale = gemm_cutlass.func_quantize_i8(X)
    W_i8, W_scale = gemm_cutlass.func_quantize_i8(W)
    
    Y_i8, Y_scale = gemm_cutlass.func_int8_matmul_out_int8_three_scale(X_i8, W_i8, X_scale, W_scale)
    
    Y_deq = Y_i8.to(torch.float32) * Y_scale.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(Y - Y_deq))
    print(f"Max diff: {max_diff.item()}")
    mse = torch.mean((Y - Y_deq) ** 2).item()
    print(f"MSE: {mse}")