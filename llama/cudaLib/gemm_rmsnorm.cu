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

// ================================================================
// Custom RMSNorm kernel for FP32 input and gamma, FP32 output
// - Input: FLOAT32 - (tokens, d_model)
// - gamma: FLOAT32 - (d_model)
// - Output: FLOAT32 - (tokens, d_model)
// ================================================================
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int d_model,
    float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_ptr = x + token_idx * d_model;
    float* y_ptr = y + token_idx * d_model;

    // Step 1: local sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = x_ptr[i];
        sum_sq += v * v;
    }

    // Step 2: block-wide reduction in shared memory
    extern __shared__ float sdata[];
    sdata[tid] = sum_sq;
    __syncthreads();

    // reduce in shared mem
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    float rms_inv;
    if (tid == 0) {
        float mean_sq = sdata[0] / d_model;
        rms_inv = rsqrtf(mean_sq + eps);
        sdata[0] = rms_inv;  // broadcast via shared mem
    }
    __syncthreads();
    rms_inv = sdata[0];

    // Step 3: normalize and scale
    for (int i = tid; i < d_model; i += blockDim.x) {
        y_ptr[i] = x_ptr[i] * rms_inv * gamma[i];
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor gamma, float eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (tokens, d_model)");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma size must match x's last dim");

    auto x_contig = x.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto y = torch::empty_like(x);

    int64_t num_tokens = x.size(0);
    int64_t d_model = x.size(1);

    int threads = (int)std::min<int64_t>(d_model, 1024);
    // make threads a power of two for reduction (optional but nice)
    if (threads & (threads - 1)) {
        // round down to nearest power of two
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)num_tokens);
    size_t shmem_bytes = threads * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_kernel<<<grid, block, shmem_bytes, stream>>>(
        x_contig.data_ptr<float>(),
        gamma_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        (int)d_model,
        eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ================================================================
// RMSNorm for INT8 input
// - Input: INT8 - (tokens, d_model)
// - gamma: FLOAT32 - (d_model)
// - Output: INT8 - (tokens, d_model)
// ================================================================
// template <typename T>
// __global__ void rmsnorm_int8_kernel(
//     const int8_t* __restrict__ x,
//     float scale_x,
//     const T* __restrict__ gamma, // Templated gamma
//     int8_t* __restrict__ y,
//     float scale_y,
//     int d_model,
//     float eps
// ) {
//     int token_idx = blockIdx.x;
//     int tid = threadIdx.x;

//     const int8_t* x_ptr = x + token_idx * d_model;
//     int8_t* y_ptr = y + token_idx * d_model;

//     float sum_sq = 0.0f;
//     for (int i = tid; i < d_model; i += blockDim.x) {
//         float v = scale_x * static_cast<float>(x_ptr[i]);
//         sum_sq += v * v;
//     }

//     extern __shared__ float sdata[];
//     sdata[tid] = sum_sq;
//     __syncthreads();

//     for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
//         if (tid < offset) {
//             sdata[tid] += sdata[tid + offset];
//         }
//         __syncthreads();
//     }

//     float rms_inv;
//     if (tid == 0) {
//         float mean_sq = sdata[0] / d_model;
//         rms_inv = rsqrtf(mean_sq + eps);
//         sdata[0] = rms_inv;
//     }
//     __syncthreads();
//     rms_inv = sdata[0];

//     // Compute pre-scaled factor to save operations in the loop
//     float combined_scale = (scale_x * rms_inv) / scale_y;

//     for (int i = tid; i < d_model; i += blockDim.x) {
//         // cast gamma[i] to float regardless of input type (T)
//         float g = static_cast<float>(gamma[i]); 
//         int q = __float2int_rn(static_cast<float>(x_ptr[i]) * combined_scale * g);
        
//         // Clamp to int8 range
//         q = max(-128, min(127, q));
//         y_ptr[i] = static_cast<int8_t>(q);
//     }
// }

// torch::Tensor rmsnorm_int8_cuda(
//     torch::Tensor x, // INT8 - (tokens, d_model)
//     float scale_x,  // Float32 scale for INT8 input
//     torch::Tensor gamma,
//     float scale_y,  // Float32 scale for INT8 output
//     float eps) 
// {
//     TORCH_CHECK(gamma.dtype() == torch::kFloat32 || gamma.dtype() == torch::kBFloat16, 
//                 "gamma must be float32 or bfloat16");

//     auto x_contig = x.contiguous();
//     auto gamma_contig = gamma.contiguous();
//     auto y = torch::empty({x.size(0), x.size(1)}, x.options().dtype(torch::kChar));

//     int64_t num_tokens = x.size(0);
//     int64_t d_model = x.size(1);

//     // Threading logic (Keep your power-of-two logic here)
//     int threads = (int)std::min<int64_t>(d_model, 512);
//     if (threads & (threads - 1)) {
//         int p = 1;
//         while ((p << 1) <= threads) p <<= 1;
//         threads = p;
//     }
//     threads = std::max(threads, 32);

//     dim3 block(threads);
//     dim3 grid((unsigned)num_tokens);
//     size_t shmem_bytes = threads * sizeof(float);
//     auto stream = at::cuda::getCurrentCUDAStream();

//     // The Magic Dispatcher
//     AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gamma.scalar_type(), "rmsnorm_int8_cuda", ([&] {
//         rmsnorm_int8_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
//             x_contig.data_ptr<int8_t>(),
//             scale_x,
//             gamma_contig.data_ptr<scalar_t>(), // scalar_t is float or bfloat16
//             y.data_ptr<int8_t>(),
//             scale_y,
//             (int)d_model,
//             eps
//         );
//     }));

//     return y;
// }

template <typename T>
__global__ void rmsnorm_int8_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ scale_x,
    const T* __restrict__ gamma,
    int8_t* __restrict__ y,
    const float* __restrict__ scale_y,
    int d_model,
    float eps
) {
    int row_idx = blockIdx.x;   // now: row over all leading dims
    int tid = threadIdx.x;

    const int8_t* x_ptr = x + row_idx * d_model;
    int8_t* y_ptr = y + row_idx * d_model;
    float scale_x_value = scale_x[row_idx];  
    float scale_y_value = scale_y[row_idx];

    float sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = scale_x_value * static_cast<float>(x_ptr[i]);
        sum_sq += v * v;
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq = sdata[0] / d_model;
        sdata[0] = rsqrtf(mean_sq + eps);   // store rms_inv in sdata[0]
    }
    __syncthreads();
    float rms_inv = sdata[0];

    float combined_scale = (scale_x_value * rms_inv) / scale_y_value;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float g = static_cast<float>(gamma[i]);
        int q = __float2int_rn(static_cast<float>(x_ptr[i]) * combined_scale * g);
        q = max(-128, min(127, q));
        y_ptr[i] = static_cast<int8_t>(q);
    }
}

torch::Tensor rmsnorm_int8_cuda(
    torch::Tensor x,      // INT8 - (..., d_model)
    torch::Tensor scale_x,  // Float32 scalar scale for INT8 input
    torch::Tensor gamma,  // (d_model)
    torch::Tensor scale_y,  // Float32 scalar scale for INT8 output
    float eps
) {
    TORCH_CHECK(x.scalar_type() == torch::kChar, "x must be int8");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32 || gamma.scalar_type() == torch::kBFloat16,
                "gamma must be float32 or bfloat16");

    TORCH_CHECK(x.numel() > 0, "x must be non-empty");
    TORCH_CHECK(x.size(-1) == gamma.size(0), "gamma size must match x.size(-1)");

    auto x_contig = x.contiguous();
    auto gamma_contig = gamma.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");
    const int64_t n_rows = x_contig.numel() / d_model;
    TORCH_CHECK(n_rows * d_model == x_contig.numel(), "x.numel must be divisible by d_model");

    TORCH_CHECK(scale_x.is_cuda() && scale_x.scalar_type() == torch::kFloat32,
            "scale_x must be CUDA float32");
    TORCH_CHECK(scale_y.is_cuda() && scale_y.scalar_type() == torch::kFloat32,
            "scale_y must be CUDA float32");
    TORCH_CHECK(scale_x.numel() == n_rows, "scale_x must have n_rows elements");
    TORCH_CHECK(scale_y.numel() == n_rows, "scale_y must have n_rows elements");

    // output same shape as input
    auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar));

    int threads = (int)std::min<int64_t>(d_model, 512);
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)n_rows);
    size_t shmem_bytes = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        gamma_contig.scalar_type(),
        "rmsnorm_int8_cuda",
        ([&] {
            rmsnorm_int8_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x.data_ptr<float>(),
                gamma_contig.data_ptr<scalar_t>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                (int)d_model,
                eps
            );
        })
    );

    return y.view(x.sizes());  
}


