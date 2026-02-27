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

#include "epilogue/thread/linear_combination.h" // my custom epilogue

#include "gemm_softmax.cu"
#include "gemm_rope.cu"
#include "gemm_matmul.cu"
#include "gemm_rmsnorm.cu"
#include "gemm_activation.cu"
#include "gemm_element_wise.cu"

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

torch::Tensor func_int8_matmul_output_int8(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    float scale            // BFloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_output_int8_host(input, weight, scale);
}

torch::Tensor func_int8_matmul_output_int8_batched(
    torch::Tensor input,   // INT8 - shape (B, M, K) or (M, K)
    torch::Tensor weight,  // INT8 - shape (B, N, K) or (N, K)
    torch::Tensor scales   // 1D float tensor, length = batch_size
) {
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  return int8_matmul_output_int8_batched_host(input, weight, scales);
}

// torch::Tensor func_int8_gemm_per_output_scale_int8(
//     torch::Tensor input,   // INT8 - shape (M, K)
//     torch::Tensor weight,  // INT8 - shape (N, K)
//     torch::Tensor scale            // BFloat16 (M, N)
// ) {
//   const at::cuda::OptionalCUDAGuard device_guard(input.device());
//   return int8_gemm_per_output_scale_int8_host(input, weight, scale);
// }   

// torch::Tensor func_int8_gemm_per_output_scale_int8_batched(
//     torch::Tensor input,   // INT8 - shape (B, M, K) or (M, K)
//     torch::Tensor weight,  // INT8 - shape (B, N, K) or (N, K)
//     torch::Tensor scales   // BFloat16 tensor, shape (B, M, N) or (M, N)
// ) {
//   const at::cuda::OptionalCUDAGuard device_guard(input.device());
//   return int8_gemm_per_output_scale_int8_batched_host(input, weight, scales);
// }


torch::Tensor func_softmax_lastdim_int8(
    torch::Tensor x_q,          // int8
    torch::Tensor scale_x,      // float32, length dim0*dim1
    torch::Tensor scale_y       // float32, length dim0*dim1
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_lastdim_int8_cuda(x_q, scale_x, scale_y);
}

torch::Tensor func_softmax_lastdim_int8_masking(
    torch::Tensor x_q,          // int8
    torch::Tensor scale_x,      // float32, length dim0*dim1
    torch::Tensor scale_y,      // float32, length dim0*dim1
    torch::Tensor mask      // bool, shape (dim0, dim1, dim2)   
) {
  const at::cuda::OptionalCUDAGuard device_guard(x_q.device());
  return softmax_lastdim_int8_masking_cuda(x_q, scale_x, scale_y, mask);
}

torch::Tensor func_rmsnorm(
    torch::Tensor x,      // FP32, shape (tokens, d_model)
    torch::Tensor gamma,  // FP32, shape (d_model,)
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_cuda(x, gamma, eps);
}

torch::Tensor func_rmsnorm_int8(
    torch::Tensor x,      // INT8, shape (tokens, d_model)
    torch::Tensor scale_x,
    torch::Tensor gamma,  // FP32, shape (d_model,)
    torch::Tensor scale_y,
    float eps) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return rmsnorm_int8_cuda(x, scale_x, gamma, scale_y, eps);
}

torch::Tensor func_element_wise_mul_int8(
    torch::Tensor a,
    torch::Tensor scale_a,
    torch::Tensor b,
    torch::Tensor scale_b,
    torch::Tensor scale_c) 
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    return element_wise_mul_int8_rowwise_cuda(a, scale_a, b, scale_b, scale_c);
}

torch::Tensor func_apply_rope_int8(
    torch::Tensor x,          // int8, shape [batch_size, num_heads, seq_len, head_dim]
    torch::Tensor scale_x,
    torch::Tensor cos,       // int8, shape [seq_len, head_dim]
    float scale_cos,
    torch::Tensor sin,       // int8, shape [seq_len, head_dim]
    float scale_sin,
    torch::Tensor scale_out) 
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    return apply_rope_int8_host(x, scale_x, cos, scale_cos, sin, scale_sin, scale_out);
}

torch::Tensor func_apply_sigmoid_int8(
    torch::Tensor input,      // int8
    torch::Tensor scale_in,   // float32 scalar
    torch::Tensor scale_out   // float32 scalar
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return sigmoid_int8_cuda(input, scale_in, scale_out);
}

torch::Tensor func_apply_silu_int8(
    torch::Tensor input,      // int8
    torch::Tensor scale_in,   // float32 scalar
    torch::Tensor scale_out   // float32 scalar
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return silu_int8_cuda_rowwise(input, scale_in, scale_out);
}

torch::Tensor func_element_add_int8(
    torch::Tensor a,          // int8
    torch::Tensor scale_a,    // float32 scalar
    torch::Tensor b,          // int8
    torch::Tensor scale_b,    // float32 scalar
    torch::Tensor scale_out   // float32 scalar
) {
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    return element_add_int8_cuda(a, scale_a, b, scale_b, scale_out);
}

torch::Tensor func_int8_matmul_out_int8_per_row_scale(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    torch::Tensor scale    // FP32 - shape (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_per_row_scale_host(input, weight, scale);
}

torch::Tensor func_int8_matmul_out_int8_per_row_scale_batched(
    torch::Tensor input,   // INT8 - shape (B, M, K) 
    torch::Tensor weight,  // INT8 - shape (B, N, K) or (N, K)
    torch::Tensor scale    // FP32 - shape (B, M, 1) or (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_per_row_scale_batched_host(input, weight, scale);
}

// ================================================================
// Still Testing and verifying
// ================================================================
torch::Tensor func_int8_matmul_out_int8_three_scale(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    torch::Tensor row_scale, // FP32 - shape (M, 1)
    torch::Tensor col_scale,  // FP32 - shape (N, 1)
    torch::Tensor out_scale // FP32 - shape (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_three_scale_host(input, weight, row_scale, col_scale, out_scale);
}

torch::Tensor func_int8_matmul_out_int8_three_scale_batched(
    torch::Tensor input,   // INT8 - shape (B, M, K) 
    torch::Tensor weight,  // INT8 - shape (B, N, K) or (N, K)
    torch::Tensor row_scale, // FP32 - shape (B, M, 1) or (M, 1)
    torch::Tensor col_scale, // FP32 - shape (B, N, 1) or (N, 1)
    torch::Tensor out_scale  // FP32 - shape (B, M, 1) or (M, 1)
) {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    return int8_matmul_out_int8_three_scale_batched_host(input, weight, row_scale, col_scale, out_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("func_int8_matmul",
        &func_int8_matmul,
        "Int8 MatMul using CUTLASS (INT8 input/weight, BFloat16 output)");

    m.def("func_int8_matmul_output_int8",
        &func_int8_matmul_output_int8,
        "Int8 MatMul using CUTLASS (INT8 input/weight/output, BFloat16 scale)");
    
    m.def("func_int8_matmul_output_int8_batched",
        &func_int8_matmul_output_int8_batched,
        "Batched Int8 MatMul using CUTLASS (INT8 input/weight/output, BFloat16 scale)");

    // m.def("func_int8_gemm_per_output_scale_int8",
    //     &func_int8_gemm_per_output_scale_int8,
    //     "Int8 MatMul using CUTLASS (INT8 input/weight/output, BFloat16 per-element scale)");

    // m.def("func_int8_gemm_per_output_scale_int8_batched",
    //     &func_int8_gemm_per_output_scale_int8_batched,
    //     "Batched Int8 MatMul using CUTLASS (INT8 input/weight/output, BFloat16 per-element scale)");

    m.def("func_softmax_lastdim_int8",
        &func_softmax_lastdim_int8,
        "Softmax along last dimension for 3D int8 with per-row scale");

    m.def("func_softmax_lastdim_int8_masking",
        &func_softmax_lastdim_int8_masking,
        "Softmax along last dimension for 3D int8 with per-row scale and masking");

    m.def("func_rmsnorm",
        &func_rmsnorm,
        "RMSNorm for 2D float32 input with float32 gamma");

    m.def("func_rmsnorm_int8",
        &func_rmsnorm_int8,
        "RMSNorm for 2D int8 input with float32 gamma and input scale");
        
    m.def("func_element_wise_mul_int8",
        &func_element_wise_mul_int8,
        "Element-wise multiply for int8 tensors with scales");
    
    m.def("func_apply_rope_int8",
        &func_apply_rope_int8,
        "Apply RoPE to int8 tensor with given cos/sin tables and scales");

    m.def("func_apply_sigmoid_int8",
        &func_apply_sigmoid_int8,
        "Apply Sigmoid activation function to int8 input tensor with given input/output scales");
    
    m.def("func_apply_silu_int8",
        &func_apply_silu_int8,
        "Apply SiLU activation function to int8 input tensor with given input/output scales");

    m.def("func_element_add_int8",
        &func_element_add_int8,
        "Element-wise add for int8 tensors with scales");

    m.def("func_int8_matmul_out_int8_per_row_scale",
        &func_int8_matmul_out_int8_per_row_scale,
        "Int8 MatMul with per-row output scale using CUTLASS (INT8 input/weight, FP32 per-row scale, INT8 output)");
    
    m.def("func_int8_matmul_out_int8_per_row_scale_batched",
        &func_int8_matmul_out_int8_per_row_scale_batched,
        "Batched Int8 MatMul with per-row output scale using CUTLASS (INT8 input/weight, FP32 per-row scale, INT8 output)");

    m.def("func_int8_matmul_out_int8_three_scale",
        &func_int8_matmul_out_int8_three_scale,
        "Int8 MatMul with three scales using CUTLASS (INT8 input/weight, BFloat16 per-element scale, INT8 output)");
    
    m.def("func_int8_matmul_out_int8_three_scale_batched",
        &func_int8_matmul_out_int8_three_scale_batched,
        "Batched Int8 MatMul with three scales using CUTLASS (INT8 input/weight, BFloat16 per-element scale, INT8 output)");
}