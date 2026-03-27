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
#include <math.h>
#include <stdint.h>

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/float8.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/gemm/device/gemm.h"

using namespace torch::indexing;

// ================================================================
// Custom RMSNorm kernel for BF16 input and gamma, BF16 output
// - Input: BF16 - (tokens, d_model)
// - gamma: BF16 - (d_model)
// - Output: BF16 - (tokens, d_model)
// ================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // up to 1024 threads = 32 warps

    const int lane = threadIdx.x & 31;   // index within warp
    const int warp = threadIdx.x >> 5;   // warp index within block
    const int num_warps = (blockDim.x + 31) >> 5;

    // Reduce inside each warp
    val = warp_reduce_sum(val);

    // One thread per warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    // First warp reduces the warp sums
    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    return block_sum;
}

__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,       // [num_tokens, d_model]
    const __nv_bfloat16* __restrict__ gamma,   // [d_model]
    __nv_bfloat16* __restrict__ y,             // [num_tokens, d_model]
    int d_model,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_ptr = x + static_cast<size_t>(token_idx) * d_model;
    __nv_bfloat16* y_ptr = y + static_cast<size_t>(token_idx) * d_model;

    // ------------------------------------------------------------
    // Step 1: each thread computes a partial sum of squares in FP32
    // ------------------------------------------------------------
    float local_sum_sq = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float v = __bfloat162float(x_ptr[i]);
        local_sum_sq += v * v;
    }

    // ------------------------------------------------------------
    // Step 2: block-wide reduction using warp reduction
    // ------------------------------------------------------------
    float block_sum_sq = block_reduce_sum(local_sum_sq);

    // Broadcast rms_inv through shared memory
    __shared__ float shared_rms_inv;

    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(d_model);
        shared_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float rms_inv = shared_rms_inv;

    // ------------------------------------------------------------
    // Step 3: normalize and scale, then write BF16 output
    // y[i] = x[i] * rms_inv * gamma[i]
    // ------------------------------------------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_val = __bfloat162float(x_ptr[i]);
        float g_val = __bfloat162float(gamma[i]);
        float y_val = x_val * rms_inv * g_val;
        y_ptr[i] = __float2bfloat16(y_val);
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor gamma, float eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be BF16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (tokens, d_model)");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
    TORCH_CHECK(x.size(1) == gamma.size(0), "gamma size must match x's last dim");

    auto x_contig = x.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto y = torch::empty_like(x);

    int64_t num_tokens = x.size(0);
    int64_t d_model = x.size(1);

    int threads = (int)std::min<int64_t>(d_model, 512);
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)num_tokens);

    auto stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma_contig.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        (int)d_model,
        eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}



// ============================================================
// Helper function to do block-wide max reduction for floats.
// ============================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_maxs[32];   // max 1024 threads => 32 warps

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) {
        warp_maxs[warp] = val;
    }
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        out = (lane < num_warps) ? warp_maxs[lane] : 0.0f;
        out = warp_reduce_max(out);
    }
    return out;
}

// ============================================================
// Fused RMSNorm + symmetric INT8 quantization
//
// Input:
//   x        : BF16, [num_tokens, d_model]
//   gamma    : BF16, [d_model]
//
// Output:
//   y_int8   : INT8, [num_tokens, d_model]
//   scale_y  : FP32, [num_tokens]   (per-row quant scale)
//
// Quantization:
//   scale_y[row] = max(abs(y_fp)) / 127
//   y_int8 = round(y_fp / scale_y)
// ============================================================

__global__ void rmsnorm_quant_int8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    int8_t* __restrict__ y_int8,
    float* __restrict__ scale_y,
    int d_model,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_ptr =
        x + static_cast<size_t>(token_idx) * d_model;
    int8_t* y_ptr =
        y_int8 + static_cast<size_t>(token_idx) * d_model;

    // --------------------------------------------------------
    // Step 1. Compute sum of squares of x in FP32
    // --------------------------------------------------------
    float local_sum_sq = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_val = __bfloat162float(x_ptr[i]);
        local_sum_sq += x_val * x_val;
    }

    float block_sum_sq = block_reduce_sum(local_sum_sq);

    __shared__ float shared_rms_inv;
    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(d_model);
        shared_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float rms_inv = shared_rms_inv;

    // --------------------------------------------------------
    // Step 2. Find row-wise max absolute value after RMSNorm
    //         y_fp = x * rms_inv * gamma
    // --------------------------------------------------------
    float local_amax = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_val = __bfloat162float(x_ptr[i]);
        float g_val = __bfloat162float(gamma[i]);
        float y_val = x_val * rms_inv * g_val;
        local_amax = fmaxf(local_amax, fabsf(y_val));
    }

    float block_amax = block_reduce_max(local_amax);

    __shared__ float shared_scale_y;
    if (tid == 0) {
        // Symmetric per-row scale.
        // Guard against all-zero row to avoid divide-by-zero.
        float s = block_amax / 127.0f;
        if (s == 0.0f) {
            s = 1.0f;
        }
        shared_scale_y = s;
        scale_y[token_idx] = s;
    }
    __syncthreads();

    const float row_scale_y = shared_scale_y;
    const float inv_scale_y = 1.0f / row_scale_y;

    // --------------------------------------------------------
    // Step 3. Quantize normalized output to INT8
    // --------------------------------------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_val = __bfloat162float(x_ptr[i]);
        float g_val = __bfloat162float(gamma[i]);
        float y_val = x_val * rms_inv * g_val;

        int q = __float2int_rn(y_val * inv_scale_y);
        q = max(-128, min(127, q));
        y_ptr[i] = static_cast<int8_t>(q);
    }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <vector>
#include <tuple>

// Kernel declaration
__global__ void rmsnorm_quant_int8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    int8_t* __restrict__ y_int8,
    float* __restrict__ scale_y,
    int d_model,
    float eps
);

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_quant_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");

    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be BF16");

    TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                "x must be 2D [tokens, d_model] or 3D [batch, tokens, d_model]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [d_model]");
    TORCH_CHECK(x.size(-1) == gamma.size(0),
                "gamma size must match x's last dimension");

    auto x_contig = x.contiguous();
    auto gamma_contig = gamma.contiguous();

    const int64_t d_model = x_contig.size(-1);

    // Flatten all leading dimensions into rows.
    const int64_t num_rows = x_contig.numel() / d_model;

    // Output y has same shape as x, but int8 dtype.
    auto y = torch::empty(
        x_contig.sizes(),
        x_contig.options().dtype(torch::kChar)
    );

    // scale_y has shape equal to x.shape[:-1]
    std::vector<int64_t> scale_shape;
    for (int i = 0; i < x_contig.dim() - 1; ++i) {
        scale_shape.push_back(x_contig.size(i));
    }

    auto scale_y = torch::empty(
        scale_shape,
        x_contig.options().dtype(torch::kFloat32)
    );

    int threads = static_cast<int>(std::min<int64_t>(d_model, 512));

    // round down to power of 2
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) {
            p <<= 1;
        }
        threads = p;
    }

    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(num_rows));

    auto stream = at::cuda::getCurrentCUDAStream();

    rmsnorm_quant_int8_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma_contig.data_ptr()),
        y.data_ptr<int8_t>(),
        scale_y.data_ptr<float>(),
        static_cast<int>(d_model),
        eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y, scale_y);
}

// std::tuple<torch::Tensor, torch::Tensor> rmsnorm_quant_cuda(torch::Tensor x, 
//     torch::Tensor gamma, float eps
// ) {
//     TORCH_CHECK(x.is_cuda(), "x must be CUDA");
//     TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
//     TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
//     TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be BF16");
//     TORCH_CHECK(x.dim() == 2, "x must be 2D (tokens, d_model)");
//     TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
//     TORCH_CHECK(x.size(1) == gamma.size(0), "gamma size must match x's last dim");

//     auto x_contig = x.contiguous();
//     auto gamma_contig = gamma.contiguous();

//     auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar)); 
//     auto scale_y = torch::empty({x_contig.size(0)}, x_contig.options().dtype(torch::kFloat32));

//     int64_t num_tokens = x.size(0);
//     int64_t d_model = x.size(1);

//     int threads = (int)std::min<int64_t>(d_model, 512);
//     if (threads & (threads - 1)) {
//         int p = 1;
//         while ((p << 1) <= threads) p <<= 1;
//         threads = p;
//     }
//     threads = std::max(threads, 32);

//     dim3 block(threads);
//     dim3 grid((unsigned)num_tokens);

//     auto stream = at::cuda::getCurrentCUDAStream();

//     rmsnorm_quant_int8_kernel<<<grid, block, 0, stream>>>(
//         reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr()),
//         reinterpret_cast<const __nv_bfloat16*>(gamma_contig.data_ptr()),
//         y.data_ptr<int8_t>(),
//         scale_y.data_ptr<float>(),
//         (int)d_model,
//         eps
//     );

//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     return std::make_tuple(y, scale_y);
// }



// ================================================================
/*
Custom RMSNorm kernel for INT8 input and INT8 output
- Input: x -  INT8, shape (..., d_model) 
    scale_x - FLOAT32, shape (...) - per-row scale for x
    gamma: FLOAT32 or BF16 - (d_model)
    scale_y - FLOAT32, shape (...) - per-row scale for y
- Output: 
    y - INT8, shape (..., d_model)
*/
template <typename T>
__global__ void rmsnorm_int8_kernel(
    const int8_t* __restrict__ x,         // quantized input: [num_rows, d_model]
    const float* __restrict__ scale_x,    // input dequant scale per row: [num_rows]
    const T* __restrict__ gamma,          // RMSNorm weight: [d_model]
    int8_t* __restrict__ y,               // quantized output: [num_rows, d_model]
    const float* __restrict__ scale_y,    // output quant scale per row: [num_rows]
    int d_model,
    float eps
) {
    // One CUDA block handles one row.
    const int row = blockIdx.x;
    const int thread_id = threadIdx.x;

    // Start of this row in the flattened input/output tensors.
    const int8_t* row_x = x + row * d_model;
    int8_t* row_y = y + row * d_model;

    // Per-row quantization scales.
    const float x_scale = scale_x[row];
    const float y_scale = scale_y[row];

    // ------------------------------------------------------------
    // Part 1. Compute sum of squares of the dequantized row.
    // Real value:  x_real[i] = x_scale * row_x[i]
    // We need:  sum_sq = sum_i (x_real[i]^2)
    // ------------------------------------------------------------
    float local_sum_sq = 0.0f;

    for (int col = thread_id; col < d_model; col += blockDim.x) {
        float x_real = x_scale * static_cast<float>(row_x[col]);
        local_sum_sq += x_real * x_real;
    }

    // ------------------------------------------------------------
    // Part 2. Reduce all thread-local sums into one block-wide sum.
    // shared[tid] will temporarily store each thread's partial sum.
    // ------------------------------------------------------------
    extern __shared__ float shared[];
    shared[thread_id] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared[thread_id] += shared[thread_id + stride];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------
    // Part 3. Compute inverse RMS for this row.
    // mean_sq = sum_sq / d_model
    // rms_inv = 1 / sqrt(mean_sq + eps)
    //
    // Store it in shared[0] so all threads can read it.
    // ------------------------------------------------------------
    if (thread_id == 0) {
        float sum_sq = shared[0];
        float mean_sq = sum_sq / static_cast<float>(d_model);
        float rms_inv = rsqrtf(mean_sq + eps);
        shared[0] = rms_inv;
    }
    __syncthreads();

    const float rms_inv = shared[0];

    // ------------------------------------------------------------
    // Part 4. Combine scales for the final quantized output.
    const float input_to_output_scale = (x_scale * rms_inv) / y_scale;

    // ------------------------------------------------------------
    // Part 5. Apply RMSNorm, then quantize to int8.
    // ------------------------------------------------------------
    for (int col = thread_id; col < d_model; col += blockDim.x) {
        float gamma_value = static_cast<float>(gamma[col]);

        float y_fp =
            static_cast<float>(row_x[col]) *
            input_to_output_scale *
            gamma_value;

        int y_int = __float2int_rn(y_fp);
        y_int = max(-128, min(127, y_int));
        row_y[col] = static_cast<int8_t>(y_int);
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


// ================================================================

template <typename T>
__global__ void rmsnorm_int8_kernel_optimize(
    const int8_t* __restrict__ x,         // quantized input: [num_rows, d_model]
    const float* __restrict__ scale_x,    // input dequant scale per row: [num_rows]
    const T* __restrict__ gamma,          // RMSNorm weight: [d_model]
    int8_t* __restrict__ y,               // quantized output: [num_rows, d_model]
    const float* __restrict__ scale_y,    // output quant scale per row: [num_rows]
    int d_model,
    float eps
) {
    // One block handles one row.
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Row pointers
    const int8_t* row_x = x + static_cast<size_t>(row) * d_model;
    int8_t* row_y = y + static_cast<size_t>(row) * d_model;

    // Per-row scales
    const float x_scale = scale_x[row];
    const float y_scale = scale_y[row];

    // ------------------------------------------------------------
    // Part 1. Compute per-thread partial sum of dequantized values.
    // x_real[col] = x_scale * row_x[col]
    // local_sum_sq = sum over this thread's columns of x_real^2
    // ------------------------------------------------------------
    float local_sum_sq = 0.0f;
    for (int col = tid; col < d_model; col += blockDim.x) {
        float x_real = x_scale * static_cast<float>(row_x[col]);
        local_sum_sq += x_real * x_real;
    }

    // ------------------------------------------------------------
    // Part 2. Reduce local sums across the block.
    // ------------------------------------------------------------
    float block_sum_sq = block_reduce_sum(local_sum_sq);

    // Shared scalar for broadcasting rms_inv to all threads
    __shared__ float shared_rms_inv;

    // ------------------------------------------------------------
    // Part 3. Compute inverse RMS for the row.
    // mean_sq = sum_sq / d_model
    // rms_inv = 1 / sqrt(mean_sq + eps)
    // ------------------------------------------------------------
    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(d_model);
        shared_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float rms_inv = shared_rms_inv;

    // ------------------------------------------------------------
    // Part 4. Precompute combined scale.
    //
    // y_int8 = round( x_int8 * (x_scale * rms_inv / y_scale) * gamma )
    // ------------------------------------------------------------
    const float input_to_output_scale = (x_scale * rms_inv) / y_scale;

    // ------------------------------------------------------------
    // Part 5. Apply RMSNorm and quantize to int8.
    // ------------------------------------------------------------
    for (int col = tid; col < d_model; col += blockDim.x) {
        float gamma_value = static_cast<float>(gamma[col]);

        float y_fp =
            static_cast<float>(row_x[col]) *
            input_to_output_scale *
            gamma_value;

        int y_int = __float2int_rn(y_fp);
        y_int = max(-128, min(127, y_int));
        row_y[col] = static_cast<int8_t>(y_int);
    }
}

torch::Tensor rmsnorm_optimized_int8_cuda(
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
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        gamma_contig.scalar_type(),
        "rmsnorm_optimized_int8_cuda",
        ([&] {
            rmsnorm_int8_kernel_optimize<scalar_t><<<grid, block, 0, stream>>>(
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