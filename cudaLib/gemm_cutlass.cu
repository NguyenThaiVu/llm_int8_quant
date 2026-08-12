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

#include "gemm_softmax.cu"
#include "gemm_rope.cu"
#include "gemm_matmul.cu"
#include "gemm_rmsnorm.cu"
#include "gemm_activation.cu"
#include "gemm_matrix_utils.cu"
#include "gemv.cu"
#include "gemm_sigmoid.cu"
#include "hadamard.cu"

using namespace torch::indexing;


// ================================================================
// PyBind entry point
// ================================================================

torch::Tensor func_bf16_gemv(
    torch::Tensor input,   // BF16 - shape (1, K) or (B, M, K)
    torch::Tensor weight,  // BF16 - shape (N, K)
    float alpha            // FP32
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return bf16_gemv_out_bf16(input, weight, alpha);
}

torch::Tensor func_i8_gemv_out_bf16(
    torch::Tensor input,   
    torch::Tensor weight,  
    torch::Tensor x_scale, 
    torch::Tensor w_scale, 
    float alpha            
) {
    // Get shape 
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    if (input.dim() <= 3) {
        return i8_gemv_out_bf16(input, weight, x_scale, w_scale, alpha);
    } else if (input.dim() == 4) {
        return i8_gemv_out_bf16_4d(input, weight, x_scale, w_scale, alpha);
    } else {
        throw std::invalid_argument("Input tensor must be 2D or 3D or 4D");
    }
}

torch::Tensor func_i8_gemv_out_i8(
    torch::Tensor input,   // INT8 - shape (1, K) or (B, M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    torch::Tensor x_scale, // FP32 - shape (1, 1) or (B, M, 1)
    torch::Tensor w_scale, // FP32 - shape (N, 1)
    torch::Tensor y_scale, // FP32 - shape (1, 1) or (B, M, 1)
    float alpha            // FP32
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    if (input.dim() <= 3) {
        return int8_gemv_out_i8(input, weight, x_scale, w_scale, y_scale, alpha);
    } else if (input.dim() == 4) {
        return int8_gemv_out_i8_4d(input, weight, x_scale, w_scale, y_scale, alpha);
    } else {
        throw std::invalid_argument("Input tensor must be 2D or 3D or 4D");
    }
}

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

std::tuple<torch::Tensor, torch::Tensor> func_softmax_int8_naive(
    torch::Tensor x_q,          // int8, shape (dim0, dim1, dim2)
    torch::Tensor scale_x      // float32, shape (dim0, dim1) 
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_int8_naive_cuda(x_q, scale_x);
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

std::tuple<torch::Tensor, torch::Tensor> func_rmsnorm_naive_int8(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_int8_naive_cuda(x, scale_x, gamma, eps);
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

std::tuple<torch::Tensor, torch::Tensor> func_silu_mul_quant(
    torch::Tensor fc1, 
    torch::Tensor fc2,
    torch::Tensor smooth_scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(fc1.device());
    return silu_mul_quant_cuda(fc1, fc2, smooth_scale);
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

torch::Tensor func_silu_mul_bf16(
    torch::Tensor fc1, 
    torch::Tensor fc2,
    torch::Tensor smooth_scale
) {
    const at::cuda::OptionalCUDAGuard device_guard(fc1.device());
    return silu_mul_bf16_cuda(fc1, fc2, smooth_scale);
}

std::tuple<torch::Tensor, torch::Tensor> func_silu_int8(
    torch::Tensor x_int8,  // int8 - shape (M, D)
    torch::Tensor scale_x // float32 - shape (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(x_int8.device());
    return silu_int8_cuda(x_int8, scale_x);
}

torch::Tensor func_silu_bf16(
    torch::Tensor x_bf16 // bf16 - shape (M, D)
) {
    const at::cuda::OptionalCUDAGuard device_guard(x_bf16.device());
    return silu_bf16_cuda(x_bf16);
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

std::tuple<torch::Tensor, torch::Tensor> func_quantize_row_int8_with_smooth_cuda(
    torch::Tensor x,  // BF16, shape (M, K)
    torch::Tensor smooth    // FP32, shape (K,)
) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return quantize_row_int8_with_smooth_cuda(x, smooth);
}

torch::Tensor func_sigmoid_bf16(
    torch::Tensor x_bf16 // bf16 - shape (M, D)
) {
    const at::cuda::OptionalCUDAGuard device_guard(x_bf16.device());
    return sigmoid_bf16_cuda(x_bf16);
}

std::tuple<torch::Tensor, torch::Tensor> func_sigmoid_int8(
    torch::Tensor x_int8,  // int8 - shape (M, D)
    torch::Tensor scale_x // float32 - shape (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(x_int8.device());
    return sigmoid_int8_cuda(x_int8, scale_x);
}

torch::Tensor func_apply_hadamard(
    torch::Tensor input  // BF16, shape (M, K)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return apply_hadamard_cuda(input);
}

std::tuple<torch::Tensor, torch::Tensor> func_fusion_hadamard_quant(
    torch::Tensor input  // BF16, shape (M, K)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return fusion_hadamard_quant_cuda(input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("func_bf16_gemv",
        &func_bf16_gemv,
        "BF16 GEMV using CUDA- DP4A (BF16 input/weight, BF16 output)");

    m.def("func_i8_gemv_out_bf16",
        &func_i8_gemv_out_bf16,
        "Int8 GEMV using DP4A (INT8 input/weight, BFloat16 output)");

    m.def("func_i8_gemv_out_i8",
        &func_i8_gemv_out_i8,
        "Int8 GEMV with output quantization using DP4A (INT8 input/weight, INT8 output)");

    m.def("func_int8_matmul",
        &func_int8_matmul,
        "Int8 MatMul using CUTLASS (INT8 input/weight, BFloat16 output)");

    m.def("func_bf16_softmax",
        &func_bf16_softmax,
        "Softmax along last dimension for BF16 tensor");

    m.def("func_softmax_int8",
        &func_softmax_int8,
        "Softmax along last dimension for int8 with per-row scale");

    m.def("func_softmax_int8_naive",
        &func_softmax_int8_naive,
        "Softmax along last dimension for int8 with per-row scale, naive implementation.");

    m.def("func_softmax_lastdim_int8_masking",
        &func_softmax_lastdim_int8_masking,
        "Softmax along last dimension for 3D int8 with per-row scale and masking");

    m.def("func_rmsnorm_bf16",
        &func_rmsnorm_bf16,
        "RMSNorm for 2D bf16 input and return bf16 output");

    m.def("func_rmsnorm_int8",
        &func_rmsnorm_int8,
        "RMSNorm for int8 input with float32 gamma and input scale");

    m.def("func_rmsnorm_naive_int8",
        &func_rmsnorm_naive_int8,
        "RMSNorm for int8 input with float32 gamma and input scale, EXPLICITLY using shared memory for reduction");

    m.def("func_rmsnorm_bf16_to_int8",
        &func_rmsnorm_bf16_to_int8,
        "RMSNorm for 2D bf16 input and return quantized int8 output with per-row scale");

    m.def("func_rmsnorm_quant",
        &func_rmsnorm_quant,
        "RMSNorm for 2D float32 input with float32 gamma, return quantized int8 output and per-row scale");

    m.def("func_apply_rope_int8",
        &func_apply_rope_int8,
        "Apply RoPE to int8 tensor with given cos/sin tables and scales");

    m.def("func_silu_mul_quant",
        &func_silu_mul_quant,
        "Apply SiLU to fc1 and multiply with fc2 (bf16), return quantized int8 output with per-row scale");

    m.def("func_silu_mul_bf16",
        &func_silu_mul_bf16,
        "Apply SiLU to fc1 and multiply with fc2 (bf16), return bf16 output");

    m.def("func_silu_mul_int8",
        &func_silu_mul_int8,
        "Apply SiLU to fc1 and multiply with fc2 (both int8) with proper scaling, return int8 output with scale");
    
    m.def("func_silu_int8",
        &func_silu_int8,
        "Apply SiLU to int8 input with proper scaling, return int8 output with scale");

    m.def("func_silu_bf16",
        &func_silu_bf16,
        "Apply SiLU to bf16 input, return bf16 output");

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

    m.def("func_quantize_row_int8_with_smooth_cuda",
        &func_quantize_row_int8_with_smooth_cuda,
        "Quantize a BF16 matrix to INT8 with per-row smooth quantization");

    m.def("func_sigmoid_bf16",
        &func_sigmoid_bf16,
        "Apply Sigmoid to bf16 input, return bf16 output");

    m.def("func_sigmoid_int8",
        &func_sigmoid_int8,
        "Apply Sigmoid to int8 input with proper scaling, return int8 output with scale");

    m.def("func_apply_hadamard",
        &func_apply_hadamard,
        "Apply Hadamard transform to bf16 input, return bf16 output");

    m.def("func_fusion_hadamard_quant",
        &func_fusion_hadamard_quant,
        "Apply Hadamard transform to bf16 input, return quantized int8 output with per-row scale");
}