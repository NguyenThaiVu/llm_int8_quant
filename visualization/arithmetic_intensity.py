import os 
import torch 
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        total_flops = 0
        
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        total_flops += x.numel()  # for max operation
        
        x = x - x_max
        total_flops += x.numel()  # for subtraction
        
        exp_x = torch.exp(x)
        total_flops += x.numel()  # for exponentiation
        
        sum_exp_x = torch.sum(exp_x, dim=self.dim, keepdim=True)
        total_flops += exp_x.numel()  # for sum operation
        
        output = exp_x / sum_exp_x
        total_flops += exp_x.numel()  # for division
        
        # Compute arithmetic intensity
        total_data_movement = x.numel() * x.element_size() + output.numel() * output.element_size()
        AI = total_flops / total_data_movement

        return output, total_flops, total_data_movement, AI
    
class Matmul(nn.Module):
    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, A, B):
        output = torch.matmul(A, B)
        
        # Compute AI
        total_flops = 2 * A.size(0) * A.size(1) * B.size(1)  # 2 * M * K * N
        total_data_movement = A.numel() * A.element_size() + B.numel() * B.element_size() + output.numel() * output.element_size()
        AI = total_flops / total_data_movement
        
        return output, total_flops, total_data_movement, AI

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        total_flops = 0
        
        x_square = x ** 2
        total_flops += x.numel()  # for square operation
        
        mean_square = torch.mean(x_square, dim=-1, keepdim=True)
        total_flops += mean_square.numel()  # for mean operation
        
        rms = torch.sqrt(mean_square + self.eps)
        total_flops += rms.numel()  # for sqrt operation
        
        output = x / rms
        total_flops += x.numel()  # for division operation
        
        # Compute AI
        total_data_movement = x.numel() * x.element_size() + output.numel() * output.element_size()
        AI = total_flops / total_data_movement
        
        return output, total_flops, total_data_movement, AI

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        total_flops = 0
        
        x_exp = torch.exp(x)
        total_flops += x.numel()  # for exponentiation operation
        
        sigmoid_x = x_exp / (1 + x_exp)
        total_flops += x.numel()  # for division operation
        
        output = x * sigmoid_x
        total_flops += x.numel()  # for multiplication operation
        
        # Compute AI
        total_data_movement = x.numel() * x.element_size() + output.numel() * output.element_size()
        AI = total_flops / total_data_movement
        return output, total_flops, total_data_movement, AI    

    
if __name__ == "__main__":
    
    device = 'cuda'
    dtype = torch.bfloat16
    
    seq_len = 2048
    dims = 4096
    
    # 1. Softmax Layer
    X = torch.randn(seq_len, dims, device=device, dtype=dtype)
    print(f"Input tensor shape: {X.shape}, dtype: {X.dtype}")
    
    softmax_layer = Softmax(dim=-1).to(device=device)    
    Y, total_flops, total_data_movement, AI = softmax_layer(X)
    total_flops = total_flops / 1e9  # Convert to GFLOPs
    memory_access = total_data_movement / (1024 ** 2)  # Convert to MB
    print(f"Output tensor shape: {Y.shape}, dtype: {Y.dtype}")
    print(f"Total FLOPs: {total_flops} GFLOPs")
    print(f"Memory Access: {memory_access} MB")
    print(f"Arithmetic Intensity (FLOPs/Byte): {AI}\n")
    
    # 2. Matmul Layer
    A = torch.randn(seq_len, dims, device=device, dtype=dtype)
    B = torch.randn(dims, dims, device=device, dtype=dtype)
    print(f"\nInput tensor A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input tensor B shape: {B.shape}, dtype: {B.dtype}")
    
    matmul_layer = Matmul().to(device=device)
    C, total_flops, total_data_movement, AI = matmul_layer(A, B)
    total_flops = total_flops / 1e9  # Convert to GFLOPs
    memory_access = total_data_movement / (1024 ** 2)  # Convert to MB
    print(f"Output tensor C shape: {C.shape}, dtype: {C.dtype}")
    print(f"Total FLOPs: {total_flops} GFLOPs")
    print(f"Memory Access: {memory_access} MB")
    print(f"Arithmetic Intensity (FLOPs/Byte): {AI}\n")
    
    # 3. RMSNorm Layer
    X = torch.randn(seq_len, dims, device=device, dtype=dtype)
    print(f"\nInput tensor shape: {X.shape}, dtype: {X.dtype}")
    
    rmsnorm_layer = RMSNorm().to(device=device)
    Y, total_flops, total_data_movement, AI = rmsnorm_layer(X)
    total_flops = total_flops / 1e9  # Convert to GFLOPs
    memory_access = total_data_movement / (1024 ** 2)  # Convert to MB
    print(f"Output tensor shape: {Y.shape}, dtype: {Y.dtype}")
    print(f"Total FLOPs: {total_flops} GFLOPs")
    print(f"Memory Access: {memory_access} MB")
    print(f"Arithmetic Intensity (FLOPs/Byte): {AI}\n")
    
    # 4. SiLU Layer
    X = torch.randn(seq_len, dims, device=device, dtype=dtype)
    print(f"\nInput tensor shape: {X.shape}, dtype: {X.dtype}")
    
    silu_layer = SiLU().to(device=device)
    Y, total_flops, total_data_movement, AI = silu_layer(X)
    total_flops = total_flops / 1e9  # Convert to GFLOPs
    memory_access = total_data_movement / (1024 ** 2)  # Convert to MB
    print(f"Output tensor shape: {Y.shape}, dtype: {Y.dtype}")
    print(f"Total FLOPs: {total_flops} GFLOPs")
    print(f"Memory Access: {memory_access} MB")
    print(f"Arithmetic Intensity (FLOPs/Byte): {AI}\n")
    
    # 5. GEMV
    A = torch.randn(1, dims, device=device, dtype=dtype)
    B = torch.randn(dims, dims, device=device, dtype=dtype)
    
    print(f"\nInput tensor A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input tensor B shape: {B.shape}, dtype: {B.dtype}")
    
    matmul_layer = Matmul().to(device=device)
    C, total_flops, total_data_movement, AI = matmul_layer(A, B)
    total_flops = total_flops / 1e9  # Convert to GFLOPs
    memory_access = total_data_movement / (1024 ** 2)  # Convert to MB
    print(f"Output tensor C shape: {C.shape}, dtype: {C.dtype}")
    print(f"Total FLOPs: {total_flops} GFLOPs")
    print(f"Memory Access: {memory_access} MB")
    print(f"Arithmetic Intensity (FLOPs/Byte): {AI}\n")