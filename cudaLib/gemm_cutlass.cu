#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/float8.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/gemm/device/gemm.h"

// #include "epilogue/thread/linear_combination.h" // my custom epilogue

#include "gemm_softmax.cu"
#include "gemm_rope.cu"
#include "gemm_matmul.cu"
#include "gemm_rmsnorm.cu"
#include "gemm_layernorm.cu"
#include "gemm_activation.cu"
#include "gemm_matrix_utils.cu"

using namespace torch::indexing;


// ================================================================
// PyBind entry point
// ================================================================

torch::Tensor func_int8_matmul(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    float alpha            // FP32
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_host(input, weight, alpha);
}

torch::Tensor func_bf16_softmax(
    torch::Tensor x          // BF16
) {
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  return softmax_bf16_cuda(x);
}

std::tuple<torch::Tensor, torch::Tensor> func_softmax_int8(
    torch::Tensor x_q,          // int8
    torch::Tensor scale_x      // float32, length dim0*dim1
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_int8_cuda(x_q, scale_x);
}

std::tuple<torch::Tensor, torch::Tensor> func_softmax_int8_shared(
    torch::Tensor x_q,          // int8, shape (dim0, dim1, dim2)
    torch::Tensor scale_x      // float32, shape (dim0, dim1) 
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_int8_explicit_shared_cuda(x_q, scale_x);
}

std::tuple<torch::Tensor, torch::Tensor> func_softmax_lastdim_int8_masking(
    torch::Tensor x_q,          // int8, shape (dim0, dim1, dim2)
    torch::Tensor scale_x,      // float32, shape (dim0, dim1) 
    torch::Tensor mask      // bool, shape (dim0, dim1, dim2)   
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_lastdim_int8_masking_cuda(x_q, scale_x, mask);
}

torch::Tensor func_rmsnorm_bf16(
    torch::Tensor x,      // BF16, shape (tokens, d_model)
    torch::Tensor gamma,  // BF16, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_bf16_cuda(x, gamma, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_rmsnorm_bf16_to_int8(
    torch::Tensor x,      // FP32, shape (tokens, d_model)
    torch::Tensor gamma,  // FP32, shape (d_model,)
    torch::Tensor smooth_scale, // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_bf16_to_int8_cuda(x, gamma, smooth_scale, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_rmsnorm_quant(
    torch::Tensor x,      // BF16, shape (tokens, d_model)
    torch::Tensor gamma,  // BF16, shape (d_model,)
    torch::Tensor smooth_scale, // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_quant_cuda(x, gamma, smooth_scale, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_rmsnorm_int8(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_int8_cuda(x, scale_x, gamma, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_rmsnorm_shared_int8(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_int8_shared_cuda(x, scale_x, gamma, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_layernorm_int8(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    torch::Tensor beta,   // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return layernorm_int8_cuda(x, scale_x, gamma, beta, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_layernorm_int8_shared(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    torch::Tensor beta,   // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return layernorm_int8_shared_cuda(x, scale_x, gamma, beta, eps);
}

std::tuple<torch::Tensor, torch::Tensor> func_apply_rope_int8(
    torch::Tensor x,          // int8, shape [batch_size, num_heads, seq_len, head_dim]
    torch::Tensor scale_x,
    torch::Tensor cos,       // int8, shape [seq_len, head_dim]
    float scale_cos,
    torch::Tensor sin,       // int8, shape [seq_len, head_dim]
    float scale_sin) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rope_int8_host(x, scale_x, cos, scale_cos, sin, scale_sin);
}

torch::Tensor func_apply_silu_int8(
    torch::Tensor input,      // int8
    torch::Tensor scale_in,   // float32 scalar
    torch::Tensor scale_out   // float32 scalar
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return silu_int8_cuda_rowwise(input, scale_in, scale_out);
}

torch::Tensor func_silu_mul(
    torch::Tensor fc1, 
    torch::Tensor fc2
) {
    const at::cuda::OptionalCUDAGuard device_guard(fc1.device());
    return silu_mul_cuda(fc1, fc2);
}

std::tuple<torch::Tensor, torch::Tensor> func_silu_mul_int8(
    torch::Tensor fc1,  // int8 - shape (M, D)
    torch::Tensor scale_fc1, // float32 - shape (M, 1)
    torch::Tensor fc2,  // int8 - shape (M, D)
    torch::Tensor scale_fc2,  // float32 - shape (M, 1)
    torch::Tensor smooth_scale,  // float32 - shape (M, 1)
    bool use_warp_reduction
) {
    const at::cuda::OptionalCUDAGuard device_guard(fc1.device());
    return silu_mul_int8_cuda(fc1, scale_fc1, fc2, scale_fc2, smooth_scale, use_warp_reduction);
}

std::tuple<torch::Tensor, torch::Tensor> func_int8_matmul_out_int8_three_scale(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    torch::Tensor row_scale, // FP32 - shape (M, 1)
    torch::Tensor col_scale  // FP32 - shape (N, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_three_scale_host(input, weight, row_scale, col_scale);
}

std::tuple<torch::Tensor, torch::Tensor> func_int8_matmul_out_int8_three_scale_batched(
    torch::Tensor input,   // INT8 - shape (B, M, K) 
    torch::Tensor weight,  // INT8 - shape (B, N, K) or (N, K)
    torch::Tensor row_scale, // FP32 - shape (B, M, 1) or (M, 1)
    torch::Tensor col_scale // FP32 - shape (B, N, 1) or (N, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_three_scale_batched_host(input, weight, row_scale, col_scale);
}

torch::Tensor func_w8a8_matmul(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    torch::Tensor alphaRow,  // FP32 - shape (M, 1)
    torch::Tensor alphaCol // FP32 - shape (N, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    if (input.dim() == 2) {
        return matmul_w8a8_2D_host(input, weight, alphaRow, alphaCol);
    } else if (input.dim() == 3) {
        return matmul_w8a8_batched_host(input, weight, alphaRow, alphaCol);
    } else if (input.dim() == 4) {
        return matmul_w8a8_batched_host(input, weight, alphaRow, alphaCol);
    } else {
        throw std::invalid_argument("Input tensor must be 2D or 3D or 4D");
    }
}

torch::Tensor func_w8a8o8_matmul(
    torch::Tensor input,   // INT8 - shape (M, K) or (B, M, K)
    torch::Tensor weight,  // INT8 - shape (N, K) or (B, N, K)
    torch::Tensor alphaRow,  // FP32 - shape (M) or (B, M)
    torch::Tensor alphaCol // FP32 - shape (N) or (B, N)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());

    if (input.dim() == 2) {
        return matmul_w8a8o8_2D_host(input, weight, alphaRow, alphaCol);
    } else if (input.dim() == 3) {
        return matmul_w8a8o8_3D_host(input, weight, alphaRow, alphaCol);
    } else if (input.dim() == 4) {
        return matmul_w8a8o8_4D_host(input, weight, alphaRow, alphaCol);
    } else {
        throw std::invalid_argument("Input tensor must be 2D or 3D");
    }
}

std::vector<torch::Tensor> func_dequant_transpose_requant(
    torch::Tensor input_int8,   // int8, [H, L, D]
    torch::Tensor input_scale   // float, [H, L]
) {
    const at::cuda::OptionalCUDAGuard device_guard(input_int8.device());

    if (input_int8.dim() == 3) {
        return dequant_transpose_requant_3d_host(input_int8, input_scale);
    } else if (input_int8.dim() == 4) {
        return dequant_transpose_requant_4d_host(input_int8, input_scale);
    } else {
        throw std::invalid_argument("Input tensor must be 3D or 4D");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("func_int8_matmul",
        &func_int8_matmul,
        "Int8 MatMul using CUTLASS (INT8 input/weight, BFloat16 output)");

    m.def("func_bf16_softmax",
        &func_bf16_softmax,
        "Softmax along last dimension for BF16 tensor");

    m.def("func_softmax_int8",
        &func_softmax_int8,
        "Softmax along last dimension for int8 with per-row scale");

    m.def("func_softmax_int8_shared",
        &func_softmax_int8_shared,
        "Softmax along last dimension for int8 with per-row scale, EXPLICITLY using shared memory for reduction");

    m.def("func_softmax_lastdim_int8_masking",
        &func_softmax_lastdim_int8_masking,
        "Softmax along last dimension for 3D int8 with per-row scale and masking");

    m.def("func_rmsnorm_bf16",
        &func_rmsnorm_bf16,
        "RMSNorm for 2D bf16 input and return bf16 output");

    m.def("func_rmsnorm_bf16_to_int8",
        &func_rmsnorm_bf16_to_int8,
        "RMSNorm for 2D bf16 input and return quantized int8 output with per-row scale");

    m.def("func_rmsnorm_quant",
        &func_rmsnorm_quant,
        "RMSNorm for 2D float32 input with float32 gamma, return quantized int8 output and per-row scale");

    m.def("func_rmsnorm_int8",
        &func_rmsnorm_int8,
        "RMSNorm for int8 input with float32 gamma and input scale");

    m.def("func_rmsnorm_shared_int8",
        &func_rmsnorm_shared_int8,
        "RMSNorm for int8 input with float32 gamma and input scale, EXPLICITLY using shared memory for reduction");

    m.def("func_layernorm_int8",
        &func_layernorm_int8,
        "LayerNorm for int8 input with float32 gamma and beta, using global memory for reduction");

    m.def("func_layernorm_int8_shared",
        &func_layernorm_int8_shared,
        "LayerNorm for int8 input with float32 gamma and beta, EXPLICITLY using shared memory for reduction");

    m.def("func_apply_rope_int8",
        &func_apply_rope_int8,
        "Apply RoPE to int8 tensor with given cos/sin tables and scales");

    m.def("func_apply_silu_int8",
        &func_apply_silu_int8,
        "Apply SiLU activation function to int8 input tensor with given input/output scales");

    m.def("func_silu_mul",
        &func_silu_mul,
        "Apply SiLU to fc1 and multiply with fc2 (both int8) with proper scaling");

    m.def("func_silu_mul_int8",
        &func_silu_mul_int8,
        "Apply SiLU to fc1 and multiply with fc2 (both int8) with proper scaling, return int8 output with scale");

    m.def("func_int8_matmul_out_int8_three_scale",
        &func_int8_matmul_out_int8_three_scale,
        "Int8 MatMul with three scales using CUTLASS (INT8 input/weight, BFloat16 per-element scale, INT8 output)");
    
    m.def("func_int8_matmul_out_int8_three_scale_batched",
        &func_int8_matmul_out_int8_three_scale_batched,
        "Batched Int8 MatMul with three scales using CUTLASS (INT8 input/weight, BFloat16 per-element scale, INT8 output)");

    m.def("func_w8a8_matmul",
        &func_w8a8_matmul,
        "MatMul for INT8 input and INT8 weight with per-row and per-column scales using CUTLASS (INT8 input/weight, FP32 per-row and per-column scales, FP32 output)");

    m.def("func_w8a8o8_matmul",
        &func_w8a8o8_matmul,
        "MatMul for INT8 input and INT8 weight with per-row and per-column scales using CUTLASS (INT8 input/weight, FP32 per-row and per-column scales, INT8 output)");

    m.def("func_dequant_transpose_requant",
        &func_dequant_transpose_requant,
        "Dequantize, transpose, and requantize an int8 matrix.");
}