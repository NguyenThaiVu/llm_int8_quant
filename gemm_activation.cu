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

using namespace torch::indexing;


__device__ __forceinline__ float to_float(float x) { return x; }

__device__ __forceinline__ int8_t clamp_int8(int v) {
    v = v > 127 ? 127 : v;
    v = v < -128 ? -128 : v;
    return static_cast<int8_t>(v);
}

__global__ void sigmoid_int8_rowwise_scale2d_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scale_in_row,   // (M,)
    int8_t* __restrict__ output,
    const float* scale_out_row,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = M * N;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / N;

        float s_in = scale_in_row[row];
        float s_out = scale_out_row[row];
        float inv_s_out = 1.0f / s_out;
        float x = (float)input[i] * s_in;

        float y = 1.0f / (1.0f + __expf(-x));
        int q = __float2int_rn(y * inv_s_out);
        output[i] = clamp_int8(q);
    }
}

torch::Tensor sigmoid_int8_rowwise_2d_cuda(
    torch::Tensor input,        // int8 (M,N)
    torch::Tensor scale_in_row, // float (M,)
    torch::Tensor scale_out_row // float (M,)
) {
    TORCH_CHECK(input.is_cuda() && scale_in_row.is_cuda() && scale_out_row.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(scale_in_row.is_contiguous(), "scale_in_row must be contiguous");
    TORCH_CHECK(scale_out_row.is_contiguous(), "scale_out_row must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kChar, "input must be int8");
    TORCH_CHECK(scale_in_row.dtype() == torch::kFloat32, "scale_in_row must be float32");
    TORCH_CHECK(scale_out_row.dtype() == torch::kFloat32, "scale_out_row must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (M,N)");
    TORCH_CHECK(scale_in_row.dim() == 1, "scale_in_row must be 1D (M,)");
    TORCH_CHECK(scale_out_row.dim() == 1, "scale_out_row must be 1D (M,)");

    int64_t M = input.size(0);
    int64_t N = input.size(1);
    TORCH_CHECK(scale_in_row.size(0) == M, "scale_in_row must have size M");
    TORCH_CHECK(scale_out_row.size(0) == M, "scale_out_row must have size M");
    auto output = torch::empty_like(input);
    
    int threads = 256;

    int64_t total = M * N;
    int blocks = (int)((total + threads - 1) / threads);

    // optional cap like you did
    int device = input.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;
    blocks = std::min(blocks, max_blocks);

    auto stream = at::cuda::getCurrentCUDAStream();
    sigmoid_int8_rowwise_scale2d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<int8_t>(),
        scale_in_row.data_ptr<float>(),
        output.data_ptr<int8_t>(),
        scale_out_row.data_ptr<float>(),
        M, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void sigmoid_int8_rowwise_scale3d_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scale_in_bm,  // (B*M) flattened (B,M)
    int8_t* __restrict__ output,
    const float* scale_out_row,
    int64_t B,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = B * M * N;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i / N;        // tmp in [0, B*M)
        int64_t row = tmp % M;      // row in [0, M)
        int64_t b   = tmp / M;      // batch in [0, B)

        float s_in = scale_in_bm[b * M + row];
        float s_out = scale_out_row[b * M + row];
        float inv_s_out = 1.0f / s_out;
        float x = (float)input[i] * s_in;

        float y = 1.0f / (1.0f + __expf(-x));
        int q = __float2int_rn(y * inv_s_out);
        output[i] = clamp_int8(q);
    }
}

torch::Tensor sigmoid_int8_rowwise_3d_cuda(
    torch::Tensor input,          // int8 (B, M, N)
    torch::Tensor scale_in_bm,    // float (B, M)
    torch::Tensor scale_out_row   // float (B, M)
) {
    TORCH_CHECK(input.is_cuda() && scale_in_bm.is_cuda() && scale_out_row.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(scale_in_bm.is_contiguous(), "scale_in_bm must be contiguous");
    TORCH_CHECK(scale_out_row.is_contiguous(), "scale_out_row must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kChar, "input must be int8");
    TORCH_CHECK(scale_in_bm.dtype() == torch::kFloat32, "scale_in_bm must be float32");
    TORCH_CHECK(scale_out_row.dtype() == torch::kFloat32, "scale_out_row must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B,M,N)");
    TORCH_CHECK(scale_in_bm.dim() == 2, "scale_in_bm must be 2D (B,M)");
    TORCH_CHECK(scale_out_row.dim() == 2, "scale_out_row must be 2D (B,M)");

    int64_t B = input.size(0);
    int64_t M = input.size(1);
    int64_t N = input.size(2);

    TORCH_CHECK(scale_in_bm.size(0) == B && scale_in_bm.size(1) == M,
                "scale_in_bm must be (B,M)");
    TORCH_CHECK(scale_out_row.size(0) == B && scale_out_row.size(1) == M,
                "scale_out_row must be (B,M)");

    auto output = torch::empty_like(input);
    int threads = 256;

    int64_t total = B * M * N;
    int blocks = (int)((total + threads - 1) / threads);

    int device = input.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;
    blocks = std::min(blocks, max_blocks);

    auto stream = at::cuda::getCurrentCUDAStream();
    sigmoid_int8_rowwise_scale3d_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<int8_t>(),
        scale_in_bm.data_ptr<float>(), // flattened is OK since contiguous
        output.data_ptr<int8_t>(),
        scale_out_row.data_ptr<float>(),
        B, M, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor sigmoid_int8_cuda(torch::Tensor input,
    torch::Tensor scale_in,
    torch::Tensor scale_out_row
) {
    if (input.dim() == 2) { // 2D input
        return sigmoid_int8_rowwise_2d_cuda(input, scale_in, scale_out_row);
    }
    else if (input.dim() == 3) { // 3D input
        return sigmoid_int8_rowwise_3d_cuda(input, scale_in, scale_out_row);
    }
    else {
        TORCH_CHECK(false, "Unsupported input dimension: ", input.dim());   
    }
}
    


__global__ void silu_int8_rowwise_2d_kernel(
    const int8_t* __restrict__ input,       // (M,N) flattened
    const float* __restrict__ scale_in_row, // (M,)
    int8_t* __restrict__ output,            // (M,N) flattened
    const float* scale_out,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / N;
        float s_in = scale_in_row[row];
        float s_out = scale_out[row];

        float x = (float)input[i] * s_in;
        float sigmoid_x = 1.0f / (1.0f + __expf(-x));
        float silu_x = x * sigmoid_x;

        int q = __float2int_rn(silu_x / s_out);
        output[i] = clamp_int8(q);
    }
}


__global__ void silu_int8_rowwise_3d_kernel(
    const int8_t* __restrict__ input,      // (B,M,N) flattened
    const float* __restrict__ scale_in_bm, // (B*M) flattened view of (B,M)
    int8_t* __restrict__ output,
    const float* __restrict__ scale_out,
    int64_t B,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = B * M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i / N;        // in [0, B*M)
        int64_t row = tmp % M;      // in [0, M)
        int64_t b   = tmp / M;      // in [0, B)

        float s_in = scale_in_bm[b * M + row];
        float s_out = scale_out[b * M + row];

        float x = (float)input[i] * s_in;
        float sigmoid_x = 1.0f / (1.0f + __expf(-x));
        float silu_x = x * sigmoid_x;

        int q = __float2int_rn(silu_x / s_out);
        output[i] = clamp_int8(q);
    }
}


torch::Tensor silu_int8_cuda_rowwise(
    torch::Tensor input,      // int8, (M,N) or (B,M,N)
    torch::Tensor scale_in,   // float32, (M,) or (B,M)
    torch::Tensor scale_out
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(scale_in.is_cuda(), "scale_in must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(scale_in.is_contiguous(), "scale_in must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kChar, "input must be int8");
    TORCH_CHECK(scale_in.dtype() == torch::kFloat32, "scale_in must be float32");
    TORCH_CHECK(scale_out.dtype() == torch::kFloat32, "scale_out must be float32");
    TORCH_CHECK(scale_out.sum().item<float>() != 0.0f, "scale_out must be non-zero");

    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "input must be 2D or 3D");

    auto output = torch::empty_like(input);

    int threads = 256;

    // block cap like your sigmoid version (recommended)
    int device = input.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;

    auto stream = at::cuda::getCurrentCUDAStream();

    if (input.dim() == 2) {
        int64_t M = input.size(0);
        int64_t N = input.size(1);

        TORCH_CHECK(scale_in.dim() == 1 && scale_in.size(0) == M,
                    "for 2D input, scale_in must be shape (M,)");
        TORCH_CHECK(scale_out.dim() == 0 || (scale_out.dim() == 1 && scale_out.size(0) == M),
                    "scale_out must be either a scalar or shape (M,)");

        int64_t total = M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        silu_int8_rowwise_2d_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<int8_t>(),
            scale_in.data_ptr<float>(),
            output.data_ptr<int8_t>(),
            scale_out.data_ptr<float>(), 
            M, N
        );
    } else {
        int64_t B = input.size(0);
        int64_t M = input.size(1);
        int64_t N = input.size(2);

        TORCH_CHECK(scale_in.dim() == 2 && scale_in.size(0) == B && scale_in.size(1) == M,
                    "for 3D input, scale_in must be shape (B,M)");
        TORCH_CHECK(scale_out.dim() == 0 || (scale_out.dim() == 2 && scale_out.size(0) == B && scale_out.size(1) == M),
                    "scale_out must be either a scalar or shape (B,M)");

        int64_t total = B * M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        silu_int8_rowwise_3d_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<int8_t>(),
            scale_in.data_ptr<float>(),   // contiguous (B,M) is fine as flat
            output.data_ptr<int8_t>(),
            scale_out.data_ptr<float>(), 
            B, M, N
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
