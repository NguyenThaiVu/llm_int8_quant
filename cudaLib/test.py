import os 
import torch 
import gemm_cutlass

if __name__ == "__main__":
    
    seq_len = 2048
    embed_dim = 4096
    d_type = torch.bfloat16
    
    X = torch.randn((seq_len, embed_dim), dtype=d_type, device='cuda')
    gamma = torch.randn((embed_dim,), dtype=d_type, device='cuda')
    eps = 1e-6
    
    # Baseline with PyTorch
    y_torch = torch.nn.functional.rms_norm(X, (embed_dim,), gamma, eps=eps)
    
    # Custom RMSNorm 
    y_i8, scale_y = gemm_cutlass.func_rmsnorm_bf16_to_int8(X, gamma, eps)
    y_deq = y_i8.float() * scale_y.unsqueeze(-1)
    
    max_diff = torch.max(torch.abs(y_torch - y_deq))
    print(f"Max absolute difference: {max_diff.item():.6f}")
    mse = torch.mean((y_torch - y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")