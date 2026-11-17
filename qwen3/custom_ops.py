
import torch
from torch import Tensor
import gemm_cutlass

@torch.library.custom_op(
    "int8_ops::rmsnorm_int8",
    mutates_args=(),
    device_types="cuda",
)
def rmsnorm_int8(
    x: Tensor,
    scale_x: Tensor,
    weight: Tensor,
    eps: float
) -> tuple[Tensor, Tensor]:
    return gemm_cutlass.func_rmsnorm_int8(
        x, scale_x, weight, eps
    )


@rmsnorm_int8.register_fake
def _(
    x: Tensor,
    scale_x: Tensor,
    weight: Tensor,
    eps: float
) -> tuple[Tensor, Tensor]:
    # Assuming the CUDA kernel returns:
    # output: same shape as x, INT8
    # scale: one FP32 scale per row
    y = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale_y = torch.empty(
        x.shape[:-1], device=x.device, dtype=torch.float32
    )
    return y, scale_y




@torch.library.custom_op(
    "int8_ops::silu_mul_int8",
    mutates_args=(),
    device_types="cuda",
)
def silu_mul_int8(
    x1: Tensor,
    scale_x1: Tensor,
    x2: Tensor,
    scale_x2: Tensor,
    smooth_scale: Tensor,
    use_warp_reduction: bool = True,
) -> tuple[Tensor, Tensor]:

    return gemm_cutlass.func_silu_mul_int8(
        x1,
        scale_x1,
        x2,
        scale_x2,
        smooth_scale,
        use_warp_reduction,
    )


@silu_mul_int8.register_fake
def _(
    x1: Tensor,
    scale_x1: Tensor,
    x2: Tensor,
    scale_x2: Tensor,
    smooth_scale: Tensor,
    use_warp_reduction: bool = True,
) -> tuple[Tensor, Tensor]:

    # Output INT8: same shape as x1
    y = torch.empty(
        x1.shape,
        device=x1.device,
        dtype=torch.int8,
    )

    # Output FP32 scale: one scale per row
    scale_y = torch.empty(
        x1.shape[:-1],
        device=x1.device,
        dtype=torch.float32,
    )

    return y, scale_y


# ============================================================
# Fused W8A8O8 GEMM
# ============================================================

@torch.library.custom_op(
    "int8_ops::w8a8o8_matmul_fusion",
    mutates_args=(),
    device_types="cuda",
)
def w8a8o8_matmul_fusion(
    x: Tensor,
    weight: Tensor,
    row_scale: Tensor,
    col_scale: Tensor,
) -> Tensor:

    return gemm_cutlass.func_w8a8o8_matmul_fusion(
        x,
        weight,
        row_scale,
        col_scale,
    )

@w8a8o8_matmul_fusion.register_fake
def _(x, weight, row_scale, col_scale):
    assert x.ndim in (2, 3, 4)
    assert weight.ndim in (2, 3, 4)
    assert x.shape[-1] == weight.shape[-1]

    batch_shape = torch.broadcast_shapes(
        x.shape[:-2],
        weight.shape[:-2]
    )

    return torch.empty(
        (*batch_shape, x.shape[-2], weight.shape[-2]),
        device=x.device,
        dtype=torch.int8,
    )




@torch.library.custom_op(
    "int8_ops::rmsnorm_quant",
    mutates_args=(),
    device_types="cuda",
)
def rmsnorm_quant(
    x: Tensor,
    weight: Tensor,
    smooth_scale: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:

    return gemm_cutlass.func_rmsnorm_quant(
        x,
        weight,
        smooth_scale,
        eps,
    )


@rmsnorm_quant.register_fake
def _(
    x: Tensor,
    weight: Tensor,
    smooth_scale: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:

    # Output INT8: same shape as x
    y = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.int8,
    )

    # Output scale FP32: one scale per row
    scale_y = torch.empty(
        x.shape[:-1],
        device=x.device,
        dtype=torch.float32,
    )

    return y, scale_y


@torch.library.custom_op(
    "int8_ops::w8a8_matmul",
    mutates_args=(),
    device_types="cuda",
)
def w8a8_matmul(
    A: Tensor,
    B: Tensor,
    scale_A: Tensor,
    scale_B: Tensor,
) -> Tensor:

    return gemm_cutlass.func_w8a8_matmul(
        A,
        B,
        scale_A,
        scale_B,
    )


@w8a8_matmul.register_fake
def _(x, weight, scale_x, scale_w):
    assert x.ndim in (2, 3, 4)
    assert x.shape[-1] == weight.shape[-1]

    batch_shape = torch.broadcast_shapes(
        x.shape[:-2],
        weight.shape[:-2]
    )

    return torch.empty(
        (*batch_shape, x.shape[-2], weight.shape[-2]),
        device=x.device,
        dtype=torch.bfloat16  # Match actual CUDA output
    )



import torch
from torch import Tensor
import gemm_cutlass


@torch.library.custom_op(
    "int8_ops::quantize_i8_smooth",
    mutates_args=(),
    device_types="cuda",
)
def quantize_i8_smooth(
    x: Tensor,
    smooth: Tensor,
) -> tuple[Tensor, Tensor]:

    return gemm_cutlass.func_quantize_row_int8_with_smooth_cuda(
        x,
        smooth,
    )


@quantize_i8_smooth.register_fake
def _(
    x: Tensor,
    smooth: Tensor,
) -> tuple[Tensor, Tensor]:

    # x: [M, K], BF16
    # smooth: [K], FP32

    y = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.int8,
    )

    scale_y = torch.empty(
        x.shape[:-1],
        device=x.device,
        dtype=torch.float32,
    )

    return y, scale_y



@torch.library.custom_op(
    "int8_ops::softmax_lastdim_int8_masking",
    mutates_args=(),
    device_types="cuda",
)
def softmax_lastdim_int8_masking(
    x: Tensor,
    scale_x: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:

    return gemm_cutlass.func_softmax_lastdim_int8_masking(
        x, scale_x, mask
    )


@softmax_lastdim_int8_masking.register_fake
def _(
    x: Tensor,
    scale_x: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:

    # x:       [..., S] INT8
    # scale_x: [...] FP32
    # mask:    broadcastable masking tensor
    #
    # y:       [..., S] INT8
    # scale_y: [...] FP32

    y = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.int8,
    )

    scale_y = torch.empty(
        x.shape[:-1],
        device=x.device,
        dtype=torch.float32,
    )

    return y, scale_y



@torch.library.custom_op(
    "int8_ops::dequant_transpose_requant",
    mutates_args=(),
    device_types="cuda",
)
def dequant_transpose_requant(
    x: Tensor,
    scale_x: Tensor,
) -> tuple[Tensor, Tensor]:

    return gemm_cutlass.func_dequant_transpose_requant(
        x, scale_x
    )


@dequant_transpose_requant.register_fake
def _(
    x: Tensor,
    scale_x: Tensor,
) -> tuple[Tensor, Tensor]:

    # x:       [H, T, D] INT8
    # scale_x: [H, T] FP32
    #
    # y:       [H, D, T] INT8
    # scale_y: [H, D] FP32

    assert x.ndim == 3

    H, T, D = x.shape

    y = torch.empty(
        (H, D, T),
        device=x.device,
        dtype=torch.int8,
    )

    scale_y = torch.empty(
        (H, D),
        device=x.device,
        dtype=torch.float32,
    )

    return y, scale_y



@torch.library.custom_op(
    "int8_ops::rope_int8",
    mutates_args=(),
    device_types="cuda",
)
def rope_int8(
    x: Tensor,
    scale_x: Tensor,
    cos: Tensor,
    scale_cos: Tensor,
    sin: Tensor,
    scale_sin: Tensor,
    return_bf16: bool = False,
) -> tuple[Tensor, Tensor]:
    return gemm_cutlass.func_apply_rope_int8(
        x, scale_x, cos, scale_cos, sin, scale_sin, return_bf16
    )


@rope_int8.register_fake
def _(
    x: Tensor,
    scale_x: Tensor,
    cos: Tensor,
    scale_cos: Tensor,
    sin: Tensor,
    scale_sin: Tensor,
    return_bf16: bool = False,
) -> tuple[Tensor, Tensor]:
    y = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.bfloat16 if return_bf16 else torch.int8,
    )
    scale_y = torch.empty(
        x.shape[:-1],
        device=x.device,
        dtype=torch.float32,
    )
    # With return_bf16=True the real CUDA kernel writes scale_y == 1.
    return y, scale_y
