import os 
import torch 
import gemm_cutlass

def dummy_int8_matmul(A, B):
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    return A @ B

def dummy_int8_elementwise_mul(A, B, out_dtype=torch.bfloat16):
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    return (A * B).to(out_dtype)


def tensor_quant(
    x: torch.Tensor,
    scale_type: torch.dtype = torch.float32,
    eps: float = 1e-8
):
    x_fp32 = x.to(torch.float32)
    q_min, q_max = -128, 127

    max_val = x_fp32.abs().amax()
    scale = max_val / q_max
    scale = torch.clamp(scale, min=eps).to(device=x.device, dtype=scale_type)

    qx = torch.round(x_fp32 / scale).clamp_(q_min, q_max).to(torch.int8)
    return qx, scale

def row_wise_quant(
    x: torch.Tensor,
    scale_type=torch.float32,
    eps: float = 1e-8
):
    x_fp32 = x.to(torch.float32)
    q_min, q_max = -128, 127
    
    max_val = x_fp32.abs().amax(dim=1, keepdim=True)
    scale = max_val / q_max
    scale = torch.clamp(scale, min=eps).to(device=x.device, dtype=scale_type)
    qx = torch.round(x_fp32 / scale).clamp_(q_min, q_max).to(torch.int8)
    return qx, scale.squeeze(1)
    

def col_wise_quant(
    x: torch.Tensor,
    scale_type=torch.float32,
    eps: float = 1e-8
):
    x_fp32 = x.to(torch.float32)
    q_min, q_max = -128, 127
    
    max_val = x_fp32.abs().amax(dim=0, keepdim=True)
    scale = max_val / q_max
    scale = torch.clamp(scale, min=eps).to(device=x.device, dtype=scale_type)
    qx = torch.round(x_fp32 / scale).clamp_(q_min, q_max).to(torch.int8)
    return qx, scale.squeeze(0)


if __name__ == "__main__":
    
    input_dim = 2048
    output_dim = 8192
    dtype = torch.bfloat16
    # dtype = torch.float32
    
    X = torch.randn(input_dim, output_dim, dtype=dtype).cuda()
    W = torch.randn(input_dim, output_dim, dtype=dtype).cuda()
    
    Y_true = X * W
    print(f"Y_true: {Y_true[:5, :5]}")
    
    # 1. Tensor quantization
    X_int8, X_scale = tensor_quant(X)
    W_int8, W_scale = tensor_quant(W)
    
    Y_deq = dummy_int8_elementwise_mul(X_int8, W_int8, out_dtype=dtype)
    Y_deq = Y_deq * X_scale * W_scale
    print(f"Y_deq (tensor quant): {Y_deq[:5, :5]}")
    
    max_diff = torch.max(torch.abs(Y_true - Y_deq))
    print(f"Max difference (tensor quant): {max_diff.item()}")
    mse = torch.mean((Y_true - Y_deq) ** 2)
    print(f"MSE (tensor quant): {mse.item()}")
    print("-" * 50 + "\n\n")
    
    
    # 2. Row-wise quantization
    X_int8, X_scale = row_wise_quant(X)
    # W_int8, W_scale = col_wise_quant(W)
    W_int8, W_scale = tensor_quant(W)
    
    # Y_deq = gemm_cutlass.func_int8_matmul(X_int8, W_int8, 1.0)
    Y_deq = dummy_int8_elementwise_mul(X_int8, W_int8, out_dtype=dtype)
    Y_deq = Y_deq * X_scale.unsqueeze(-1) * W_scale.unsqueeze(0)
    print(f"Y_deq: {Y_deq[:5, :5]}")
    
    max_diff = torch.max(torch.abs(Y_true - Y_deq))
    print(f"Max difference: {max_diff.item()}")
    mse = torch.mean((Y_true - Y_deq) ** 2)
    print(f"MSE: {mse.item()}")

