#pragma once

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <limits>
#include <tuple>

#include "gemm_utils.cu"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// ============================================================
// BF16 Sigmoid with __nv_bfloat162 vectorized loads/stores.
//
// Input:
//   x_bf16   [..., cols]
//
// Output:
//   out_bf16 [..., cols]
//
// Requirements:
//   - cols % 2 == 0
//   - input/output contiguous
// ============================================================
__global__ void sigmoid_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols
) {
    const int row = static_cast<int>(blockIdx.x);

    if (row >= rows) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int vec_cols = cols >> 1;
    const int vec_row_offset = row * vec_cols;

    const __nv_bfloat162* __restrict__ x_vec = reinterpret_cast<const __nv_bfloat162*>(x_bf16);
    __nv_bfloat162* __restrict__ out_vec = reinterpret_cast<__nv_bfloat162*>(out_bf16);

    for (int vec_col = tid; vec_col < vec_cols; vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;

        const __nv_bfloat162 x_pair = x_vec[vec_index];
        const float2 x_float = __bfloat1622float2(x_pair);

        const float y0 = sigmoid_fast(x_float.x);
        const float y1 = sigmoid_fast(x_float.y);

        out_vec[vec_index] = __floats2bfloat162_rn(y0, y1);
    }
}

torch::Tensor sigmoid_bf16_cuda(
    torch::Tensor x_bf16
) {
    TORCH_CHECK(x_bf16.is_cuda(), "x_bf16 must be a CUDA tensor");

    TORCH_CHECK(
        x_bf16.scalar_type() == torch::kBFloat16,
        "x_bf16 must have dtype torch.bfloat16"
    );

    TORCH_CHECK(
        x_bf16.dim() >= 1,
        "x_bf16 must have at least one dimension"
    );

    TORCH_CHECK(
        x_bf16.numel() > 0,
        "x_bf16 must not be empty"
    );

    const c10::cuda::CUDAGuard device_guard(x_bf16.device());

    x_bf16 = x_bf16.contiguous();

    const int64_t cols64 = x_bf16.size(-1);
    const int64_t rows64 = x_bf16.numel() / cols64;

    TORCH_CHECK(
        cols64 > 0,
        "The last dimension must be greater than zero"
    );

    TORCH_CHECK(
        cols64 % 2 == 0,
        "The last dimension must be divisible by 2 "
        "for __nv_bfloat162 vectorization"
    );

    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max(),
        "rows exceeds the supported int range"
    );

    TORCH_CHECK(
        cols64 <= std::numeric_limits<int>::max(),
        "cols exceeds the supported int range"
    );

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);

    auto out_bf16 = torch::empty_like(x_bf16);

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(rows);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            x_bf16.get_device()
        ).stream();

    sigmoid_bf16_vec2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x_bf16.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<__nv_bfloat16*>(
            out_bf16.data_ptr<at::BFloat16>()
        ),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_bf16;
}

// ============================================================
// INT8 Sigmoid with char4 vectorized loads/stores.
//
// Input:
//   x_int8       [rows, cols]
//   input_scales [rows]
//
// Output:
//   out_int8     [rows, cols]
//   out_scales   [rows]
//
// Requirements:
//   - cols % 4 == 0
//   - blockDim.x is a multiple of 32
//   - block_reduce_max returns the result to every thread
// ============================================================
__global__ void sigmoid_int8_vec4_kernel(
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ input_scales,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scales,
    int rows,
    int cols
) {
    const int row = static_cast<int>(blockIdx.x);

    if (row >= rows) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int vec_cols = cols >> 2;
    const int vec_row_offset = row * vec_cols;

    const float input_scale = input_scales[row];

    const char4* __restrict__ x_vec =
        reinterpret_cast<const char4*>(x_int8);

    char4* __restrict__ out_vec =
        reinterpret_cast<char4*>(out_int8);

    // --------------------------------------------------------
    // Pass 1: compute maximum sigmoid output in this row.
    // --------------------------------------------------------
    float local_max = 0.0f;

    for (int vec_col = tid;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;
        const char4 qx = x_vec[vec_index];

        const float x0 = static_cast<float>(qx.x) * input_scale;
        const float x1 = static_cast<float>(qx.y) * input_scale;
        const float x2 = static_cast<float>(qx.z) * input_scale;
        const float x3 = static_cast<float>(qx.w) * input_scale;

        const float y0 = sigmoid_fast(x0);
        const float y1 = sigmoid_fast(x1);
        const float y2 = sigmoid_fast(x2);
        const float y3 = sigmoid_fast(x3);

        local_max = fmaxf(local_max, y0);
        local_max = fmaxf(local_max, y1);
        local_max = fmaxf(local_max, y2);
        local_max = fmaxf(local_max, y3);
    }

    const float row_max = block_reduce_max(local_max);

    __shared__ float shared_inv_output_scale;

    if (tid == 0) {
        const float output_scale =
            row_max > 0.0f
                ? row_max * (1.0f / 127.0f)
                : 1.0f;

        out_scales[row] = output_scale;
        shared_inv_output_scale = 1.0f / output_scale;
    }

    __syncthreads();

    const float inv_output_scale = shared_inv_output_scale;

    // --------------------------------------------------------
    // Pass 2: recompute sigmoid and quantize.
    // --------------------------------------------------------
    for (int vec_col = tid;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;
        const char4 qx = x_vec[vec_index];

        const float x0 = static_cast<float>(qx.x) * input_scale;
        const float x1 = static_cast<float>(qx.y) * input_scale;
        const float x2 = static_cast<float>(qx.z) * input_scale;
        const float x3 = static_cast<float>(qx.w) * input_scale;

        const float y0 = sigmoid_fast(x0) * inv_output_scale;
        const float y1 = sigmoid_fast(x1) * inv_output_scale;
        const float y2 = sigmoid_fast(x2) * inv_output_scale;
        const float y3 = sigmoid_fast(x3) * inv_output_scale;

        const char4 qout = make_char4(
            static_cast<char>(quantize_int8(y0)),
            static_cast<char>(quantize_int8(y1)),
            static_cast<char>(quantize_int8(y2)),
            static_cast<char>(quantize_int8(y3))
        );

        out_vec[vec_index] = qout;
    }
}

std::tuple<torch::Tensor, torch::Tensor> sigmoid_int8_cuda(
    torch::Tensor x_int8,
    torch::Tensor scale_x
) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");

    TORCH_CHECK(
        x_int8.scalar_type() == torch::kInt8,
        "x_int8 must have dtype torch.int8"
    );

    TORCH_CHECK(
        scale_x.scalar_type() == torch::kFloat32,
        "scale_x must have dtype torch.float32"
    );

    TORCH_CHECK(
        x_int8.device() == scale_x.device(),
        "x_int8 and scale_x must be on the same CUDA device"
    );

    TORCH_CHECK(
        x_int8.dim() >= 1,
        "x_int8 must have at least one dimension"
    );

    TORCH_CHECK(
        x_int8.numel() > 0,
        "x_int8 must not be empty"
    );

    const c10::cuda::CUDAGuard device_guard(x_int8.device());

    x_int8 = x_int8.contiguous();
    scale_x = scale_x.contiguous();

    const int64_t cols64 = x_int8.size(-1);
    const int64_t rows64 = x_int8.numel() / cols64;

    TORCH_CHECK(
        cols64 > 0,
        "The last dimension must be greater than zero"
    );

    TORCH_CHECK(
        cols64 % 4 == 0,
        "The last dimension must be divisible by 4 "
        "for char4 vectorization"
    );

    TORCH_CHECK(
        scale_x.numel() == rows64,
        "scale_x must contain one scale per flattened row. ",
        "Expected ",
        rows64,
        ", but received ",
        scale_x.numel()
    );

    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max(),
        "rows exceeds the supported int range"
    );

    TORCH_CHECK(
        cols64 <= std::numeric_limits<int>::max(),
        "cols exceeds the supported int range"
    );

    TORCH_CHECK(
        BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024,
        "BLOCK_SIZE must be between 1 and 1024"
    );

    TORCH_CHECK(
        BLOCK_SIZE % 32 == 0,
        "BLOCK_SIZE must be a multiple of 32"
    );

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);

    auto input_scales = scale_x.view({rows64});
    auto out_int8 = torch::empty_like(x_int8);

    auto scale_shape = x_int8.sizes().vec();
    scale_shape.pop_back();

    if (scale_shape.empty()) {
        scale_shape.push_back(1);
    }

    auto out_scales = torch::empty(
        scale_shape,
        x_int8.options().dtype(torch::kFloat32)
    );

    const dim3 block(BLOCK_SIZE);
    const dim3 grid(rows);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            x_int8.get_device()
        ).stream();

    sigmoid_int8_vec4_kernel<<<grid, block, 0, stream>>>(
        x_int8.data_ptr<int8_t>(),
        input_scales.data_ptr<float>(),
        out_int8.data_ptr<int8_t>(),
        out_scales.data_ptr<float>(),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out_int8, out_scales};
}





