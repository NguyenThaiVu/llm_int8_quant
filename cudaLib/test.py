import os 
import torch 
import gemm_cutlass

if __name__ == "__main__":
    
    seq_len = 2048
    embed_dim = 4096
    d_type = torch.bfloat16
    
    X1 = torch.randn(seq_len, embed_dim, dtype=d_type, device='cuda')
    X2 = torch.randn(seq_len, embed_dim, dtype=d_type, device='cuda')
    
    # Torch silu
    y_torch = torch.nn.functional.silu(X1) * X2
    
    # Custom 
    y_i8, scale_y = gemm_cutlass.func_silu_mul_quant(X1, X2)
    y_deq = y_i8.float() * scale_y.unsqueeze(1)  
    
    max_diff = torch.max(torch.abs(y_torch - y_deq))
    print(f"Max absolute difference: {max_diff.item():.6f}")
    mse = torch.mean((y_torch - y_deq) ** 2).item()
    print(f"Mean Squared Error: {mse:.6f}")