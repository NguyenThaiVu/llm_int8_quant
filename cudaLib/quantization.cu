#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include "gemm_utils.cu"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

template <typename scalar_t, int VEC>
struct Vec {
    scalar_t x[VEC];
};

template <int VEC>
struct Int8Vec {
    int8_t x[VEC];
};

template <typename scalar_t, int VEC>
__global__ void row_quant_int8_vec_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int num_vecs = K / VEC;

    float local_max = 0.0f;

    const Vec<scalar_t, VEC>* input_vec =
        reinterpret_cast<const Vec<scalar_t, VEC>*>(input + row_offset);

    for (int vec_id = tid; vec_id < num_vecs; vec_id += blockDim.x) {
        const Vec<scalar_t, VEC> v = input_vec[vec_id];

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float x = static_cast<float>(v.x[j]);
            local_max = fmaxf(local_max, fabsf(x));
        }
    }

    float max_val = block_reduce_max(local_max);
    max_val = fmaxf(max_val, 1e-8f);

    const float scale = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;

    if (tid == 0) {
        scales[row] = scale;
    }

    Int8Vec<VEC>* output_vec =
        reinterpret_cast<Int8Vec<VEC>*>(output + row_offset);

    for (int vec_id = tid; vec_id < num_vecs; vec_id += blockDim.x) {
        const Vec<scalar_t, VEC> v = input_vec[vec_id];

        Int8Vec<VEC> q;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float x = static_cast<float>(v.x[j]);
            q.x[j] = quantize_int8(x * inv_scale);
        }

        output_vec[vec_id] = q;
    }
}


template <typename scalar_t>
__global__ void row_quant_int8_scalar_kernel(
    const scalar_t* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * K;

    // 1. Row absmax
    float local_max = 0.0f;

    for (int i = tid; i < K; i += blockDim.x) {
        const float x = static_cast<float>(input[row_offset + i]);
        local_max = fmaxf(local_max, fabsf(x));
    }

    float max_val = block_reduce_max(local_max);
    max_val = fmaxf(max_val, 1e-8f);

    // ========================================================
    // 2. Scale
    // ========================================================
    const float scale = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;
    if (tid == 0) {
        scales[row] = scale;
    }

    // 3. Quantize
    for (int i = tid; i < K; i += blockDim.x) {
        const float x = static_cast<float>(input[row_offset + i]);
        output[row_offset + i] = quantize_int8(x * inv_scale);
    }
}

std::tuple<torch::Tensor, torch::Tensor> quantize_row_int8_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");

    input = input.contiguous();

    const int64_t K = input.size(-1);
    const int64_t rows = input.numel() / K;

    // Output
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
    std::vector<int64_t> scale_shape(input.sizes().begin(), input.sizes().end() - 1);
    auto scales = torch::empty(scale_shape, input.options().dtype(torch::kFloat32));

    // ========================================================
    // Launch
    // ========================================================

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "row_quant_int8",
        [&] {
            if (K % 8 == 0) {
                row_quant_int8_vec_kernel<scalar_t, 8>
                <<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            } else if (K % 4 == 0) {
                row_quant_int8_vec_kernel<scalar_t, 4>
                <<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            } else {
                row_quant_int8_scalar_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(output, scales);
}

template <typename scalar_t, int VEC>
__global__ void row_smooth_quant_int8_vec_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int num_vecs = K / VEC;

    const Vec<scalar_t, VEC>* input_vec =
        reinterpret_cast<const Vec<scalar_t, VEC>*>(input + row_offset);

    const Vec<float, VEC>* smooth_vec =
        reinterpret_cast<const Vec<float, VEC>*>(smooth_scale);

    // ========================================================
    // 1. Smooth + row absmax
    // ========================================================

    float local_max = 0.0f;

    for (int vec_id = tid; vec_id < num_vecs; vec_id += blockDim.x) {
        const Vec<scalar_t, VEC> x_vec = input_vec[vec_id];
        const Vec<float, VEC> s_vec = smooth_vec[vec_id];

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float x = static_cast<float>(x_vec.x[j]);
            const float smooth_x = x / s_vec.x[j];

            local_max = fmaxf(local_max, fabsf(smooth_x));
        }
    }

    float max_val = block_reduce_max(local_max);
    max_val = fmaxf(max_val, 1e-8f);

    // ========================================================
    // 2. Quantization scale
    // ========================================================

    const float scale = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;

    if (tid == 0) {
        scales[row] = scale;
    }

    // ========================================================
    // 3. Smooth + quantize
    // ========================================================

    Int8Vec<VEC>* output_vec =
        reinterpret_cast<Int8Vec<VEC>*>(output + row_offset);

    for (int vec_id = tid; vec_id < num_vecs; vec_id += blockDim.x) {
        const Vec<scalar_t, VEC> x_vec = input_vec[vec_id];
        const Vec<float, VEC> s_vec = smooth_vec[vec_id];

        Int8Vec<VEC> q;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float x = static_cast<float>(x_vec.x[j]);
            const float smooth_x = x / s_vec.x[j];

            q.x[j] = quantize_int8(smooth_x * inv_scale);
        }

        output_vec[vec_id] = q;
    }
}

template <typename scalar_t>
__global__ void row_smooth_quant_int8_scalar_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ smooth_scale,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * K;

    float local_max = 0.0f;

    for (int i = tid; i < K; i += blockDim.x) {
        const float x = static_cast<float>(input[row_offset + i]);
        const float smooth_x = x / smooth_scale[i];

        local_max = fmaxf(local_max, fabsf(smooth_x));
    }

    float max_val = block_reduce_max(local_max);
    max_val = fmaxf(max_val, 1e-8f);

    const float scale = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;

    if (tid == 0) {
        scales[row] = scale;
    }

    for (int i = tid; i < K; i += blockDim.x) {
        const float x = static_cast<float>(input[row_offset + i]);
        const float smooth_x = x / smooth_scale[i];

        output[row_offset + i] = quantize_int8(smooth_x * inv_scale);
    }
}

std::tuple<torch::Tensor, torch::Tensor> quantize_row_int8_smooth_cuda(
    torch::Tensor input,
    torch::Tensor smooth_scale)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(smooth_scale.is_cuda(), "smooth_scale must be CUDA");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");

    input = input.contiguous();
    smooth_scale = smooth_scale.contiguous();

    const int64_t K = input.size(-1);
    const int64_t rows = input.numel() / K;

    TORCH_CHECK(smooth_scale.numel() == K, "smooth_scale must have K elements");
    TORCH_CHECK(smooth_scale.scalar_type() == torch::kFloat32, "smooth_scale must be float32");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));

    std::vector<int64_t> scale_shape(input.sizes().begin(), input.sizes().end() - 1);
    auto scales = torch::empty(scale_shape, input.options().dtype(torch::kFloat32));

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "row_smooth_quant_int8",
        [&] {
            if (K % 8 == 0) {
                row_smooth_quant_int8_vec_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    smooth_scale.data_ptr<float>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            } else if (K % 4 == 0) {
                row_smooth_quant_int8_vec_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    smooth_scale.data_ptr<float>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            } else {
                row_smooth_quant_int8_scalar_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    smooth_scale.data_ptr<float>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output, scales);
}