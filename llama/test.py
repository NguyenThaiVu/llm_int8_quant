import os 
import torch

from llama3_quan import *

M = 512
N = 4096
dtype = torch.bfloat16

LLAMA32_CONFIG = get_llama_config(LLAMA_SIZE_STR)

X = torch.randn((1, M, N), dtype=dtype).cuda()
cos = torch.randn((1, M, N), dtype=dtype).cuda()
sin = torch.randn((1, M, N), dtype=dtype).cuda()
mask = torch.ones((1, M, M), dtype=torch.bool).cuda()

gqa = Custom_GroupedQueryAttention(d_in=4096, num_heads=16,\
	num_kv_groups=4, dtype=dtype).cuda()

Y = gqa(X, 1.0, mask, cos, sin)
print(Y.shape)
