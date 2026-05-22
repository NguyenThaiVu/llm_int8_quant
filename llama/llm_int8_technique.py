import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt


def convert_linear_to_llm_int8(module, threshold=6.0, skip_names=()):
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue

        if isinstance(child, nn.Linear):
            int8_linear = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )

            int8_linear.weight.data = child.weight.data.clone()

            if child.bias is not None:
                int8_linear.bias.data = child.bias.data.clone()

            setattr(module, name, int8_linear)
        else:
            convert_linear_to_llm_int8(child, threshold, skip_names)

    return module

if __name__ == "__main__":
    
    seq_len = 1024
    in_dims = 2048
    hidden_dims = 4096
    out_dims = 8192
    dtype = torch.bfloat16
    
    input_tensor = torch.randn(seq_len, in_dims, dtype=dtype).cuda()
    
    model = nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.ReLU(),
        nn.Linear(hidden_dims, out_dims)
    ).to(dtype).cuda()
    
    out_torch_tensor = model(input_tensor)

    model = model.cpu() # Move model to CPU for quantization
    int8_model = convert_linear_to_llm_int8(model, 6.0)

    print(f"Weight before quantization: {int8_model[0].weight.data[:5, :5]}")
    # int8_model = int8_model.to(0) # Quantization happens here
    int8_model = int8_model.to('cuda')
    print(f"Weight after quantization: {int8_model[0].weight.data[:5, :5]}")
    
    out_int8_tensor = int8_model(input_tensor)
    
    print(f"Output from torch model: {out_torch_tensor[:5, :5]}")
    print(f"Output from int8 model: {out_int8_tensor[:5, :5]}")
    
