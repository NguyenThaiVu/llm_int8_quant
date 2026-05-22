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

fp16_model = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024)
)

# int8_model = nn.Sequential(
#     Linear8bitLt(64, 64, has_fp16_weights=False),
#     Linear8bitLt(64, 64, has_fp16_weights=False)
# )

# int8_model.load_state_dict(fp16_model.state_dict())

int8_model = convert_linear_to_llm_int8(fp16_model, 6.0)

print(f"Weight before quantization: {int8_model[0].weight.data[:5, :5]}")

int8_model = int8_model.to(0) # Quantization happens here

print(f"Weight after quantization: {int8_model[0].weight.data[:5, :5]}")