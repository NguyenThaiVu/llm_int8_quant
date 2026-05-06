#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <tuple>

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

#include "gemm_utils.cu"

using namespace torch::indexing;

// ================================================================
// Custom RMSNorm kernel for BF16 input and gamma, BF16 output
// - Input: BF16 - (tokens, d_model)
// - gamma: BF16 - (d_model)
// - Output: BF16 - (tokens, d_model)
// ================================================================

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

    // all threads can read shared_rms_inv
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


// ================================================================
/*
RMSNorm kernel for INT8 input and INT8 output. And we incorporate the "smooth quantization" 
to the output scale.

- Input: x -  INT8, shape (..., d_model) 
    scale_x - FLOAT32, shape (...) - per-row scale for x
    gamma: FLOAT32 or BF16 - (d_model)
- Output: 
    y - INT8, shape (..., d_model)
    scale_y - FLOAT32, shape (...) - per-row scale for y
*/
// ================================================================
__global__ void rmsnorm_quant_int8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    int8_t* __restrict__ y_int8,
    float* __restrict__ scale_y,
    int d_model,
    const float* __restrict__ smooth_scale,
    float eps
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_ptr = x + static_cast<size_t>(token_idx) * d_model;
    int8_t* y_ptr = y_int8 + static_cast<size_t>(token_idx) * d_model;

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
    //         y_fp = x * rms_inv * gamma * smooth_scale
    // --------------------------------------------------------
    float local_amax = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_val = __bfloat162float(x_ptr[i]);
        float g_val = __bfloat162float(gamma[i]);
        float s_val = smooth_scale[i];

        float y_val = x_val * rms_inv * g_val / s_val;
        local_amax = fmaxf(local_amax, fabsf(y_val));
    }

    float block_amax = block_reduce_max(local_amax);

    __shared__ float shared_scale_y;
    if (tid == 0) {
        // Symmetric per-row scale.
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
        float s_val = smooth_scale[i];
        float y_val = x_val * rms_inv * g_val / s_val;

        int q = __float2int_rn(y_val * inv_scale_y);
        q = max(-128, min(127, q));
        y_ptr[i] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_quant_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor smooth_scale,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(smooth_scale.is_cuda(), "smooth_scale must be CUDA");

    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be BF16");
    TORCH_CHECK(smooth_scale.scalar_type() == torch::kFloat32,
                    "smooth_scale must be FP32");

    TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
                "x must be 2D [tokens, d_model] or 3D [batch, tokens, d_model]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D [d_model]");
    TORCH_CHECK(x.size(-1) == gamma.size(0),
                "gamma size must match x's last dimension");
    TORCH_CHECK(smooth_scale.dim() == 1, "smooth_scale must be 1D [d_model]");
    TORCH_CHECK(smooth_scale.size(0) == gamma.size(0),
                "smooth_scale size must match gamma size");

    auto x_contig = x.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto smooth_scale_contig = smooth_scale.contiguous();
    const int64_t d_model = x_contig.size(-1);

    // Flatten all leading dimensions into rows.
    const int64_t num_rows = x_contig.numel() / d_model;

    // Output y has same shape as x, but int8 dtype.
    auto y = torch::empty(x_contig.sizes(), x_contig.options().dtype(torch::kChar));

    // scale_y has shape equal to x.shape[:-1]
    std::vector<int64_t> scale_shape;
    for (int i = 0; i < x_contig.dim() - 1; ++i) {
        scale_shape.push_back(x_contig.size(i));
    }
    auto scale_y = torch::empty(
        scale_shape,
        x_contig.options().dtype(torch::kFloat32)
    );

    // Determine block and grid sizes
    int threads = static_cast<int>(std::min<int64_t>(d_model, 512));
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
        smooth_scale_contig.data_ptr<float>(),
        eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y, scale_y);
}



// ================================================================
/*
RMSNorm kernel for INT8 input and INT8 output
- Input: x -  INT8, shape (..., d_model) 
    scale_x - FLOAT32, shape (...) - per-row scale for x
    gamma: FLOAT32 or BF16 - (d_model)
- Output: 
    y - INT8, shape (..., d_model)
    scale_y - FLOAT32, shape (...) - per-row scale for y
*/
// ================================================================
template <typename T>
__global__ void rmsnorm_int8_kernel(
    const int8_t* __restrict__ x,         // [num_rows, d_model]
    const float* __restrict__ scale_x,    // shape [num_rows]
    const T* __restrict__ gamma,          // RMSNorm weight: shape [d_model]
    int8_t* __restrict__ y,               // Shape [num_rows, d_model]
    float* __restrict__ scale_y,          // shape [num_rows]
    int d_model,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int8_t* row_x = x + row * d_model;
    int8_t* row_y = y + row * d_model;

    const float x_scale = scale_x[row];

    // ------------------------------------------------------------
    // Pass 1a: Compute sum of squares 
    // x_real = x_q * x_scale
    // sum_sq = sum(x_real^2)
    // ------------------------------------------------------------
    float local_sum_sq = 0.0f;
    for (int col = tid; col < d_model; col += blockDim.x) {
        float x_real = x_scale * static_cast<float>(row_x[col]);
        local_sum_sq += x_real * x_real;
    }

    float block_sum_sq = block_reduce_sum(local_sum_sq);

    // Share block_sum_sq for all threads to compute rms_inv
    __shared__ float shared_block_sum_sq;
    if (tid == 0) {
        shared_block_sum_sq = block_sum_sq;
    }
    __syncthreads();

    float mean_sq = shared_block_sum_sq / static_cast<float>(d_model);
    float rms_inv = rsqrtf(mean_sq + eps);

    // ------------------------------------------------------------
    // Pass 1b: Compute max absolute floating-point output
    // y_fp = (x_q * x_scale) * rms_inv * gamma[col]
    // ------------------------------------------------------------
    float local_max_abs = 0.0f;
    const float norm_scale = x_scale * rms_inv;

    for (int col = tid; col < d_model; col += blockDim.x) {
        float gamma_val = static_cast<float>(gamma[col]);
        float y_fp = static_cast<float>(row_x[col]) * norm_scale * gamma_val;
        float a = fabsf(y_fp);
        local_max_abs = fmaxf(local_max_abs, a);
    }

    float block_amax = block_reduce_max(local_max_abs);

    // Share block_amax for all threads to compute quantization scale for output
    __shared__ float shared_scale_y;
    if (tid == 0) {
        float s = block_amax / 127.0f;
        if (s == 0.0f) {
            s = 1.0f;
        }
        shared_scale_y = s;
        scale_y[row] = s;
    }
    __syncthreads();

    const float inv_scale_y = 1.0f / shared_scale_y;

    // ------------------------------------------------------------
    // Pass 2: Quantize output
    // ------------------------------------------------------------
    for (int col = tid; col < d_model; col += blockDim.x) {
        float gamma_val = static_cast<float>(gamma[col]);
        float y_fp = static_cast<float>(row_x[col]) * norm_scale * gamma_val;

        int q = __float2int_rn(y_fp * inv_scale_y);
        q = max(-127, min(127, q));

        row_y[col] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_int8_cuda(
    torch::Tensor x,      // INT8 - (..., d_model)
    torch::Tensor scale_x,  // Float32 scalar scale for INT8 input
    torch::Tensor gamma,  // (d_model)
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kChar, "x must be int8");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == torch::kFloat32, "scale_x must be float32");
    
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32 || gamma.scalar_type() == torch::kBFloat16,
                "gamma must be float32 or bfloat16");

    TORCH_CHECK(x.numel() > 0, "x must be non-empty");
    TORCH_CHECK(x.size(-1) == gamma.size(0), "gamma size must match x.size(-1)");

    auto x_contig = x.contiguous();
    auto scale_x_contig = scale_x.contiguous();
    auto gamma_contig = gamma.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");
    const int64_t n_rows = x_contig.numel() / d_model;
    TORCH_CHECK(n_rows * d_model == x_contig.numel(), "x.numel must be divisible by d_model");

    TORCH_CHECK(scale_x.is_cuda() && scale_x.scalar_type() == torch::kFloat32,
            "scale_x must be CUDA float32");
    TORCH_CHECK(scale_x.numel() == n_rows, "scale_x must have n_rows elements");

    // output same shape as input
    auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar));
    auto scale_y = torch::empty({n_rows}, x_contig.options().dtype(torch::kFloat32));

    // Determine block and grid sizes
    int threads = (int)std::min<int64_t>(d_model, 512);
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)n_rows);
    // size_t shmem_bytes = 2 * threads * sizeof(float);
    size_t shmem_bytes = 0; 
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(
        gamma_contig.scalar_type(),
        "rmsnorm_int8_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            rmsnorm_int8_kernel<float><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<float>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            rmsnorm_int8_kernel<at::BFloat16><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<at::BFloat16>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y.view(x.sizes()), scale_y.view(scale_x.sizes()));  
}


/*
This function has the same computation as rmsnorm_int8_kernel, 
except that it uses shared memory to store intermediate results.
*/
template <typename T>
__global__ void rmsnorm_int8_shared_kernel(
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ scale_x,
    const T* __restrict__ gamma,
    int8_t* __restrict__ y_int8,
    float* __restrict__ scale_y,
    int d_model,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    extern __shared__ float x_fp_s[];

    float sx = scale_x[row];

    // -----------------------------
    // 1. Dequantize X_int8 -> X_fp
    // -----------------------------
    float local_sum_sq = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        x_fp_s[i] = x_fp;
        local_sum_sq += x_fp * x_fp;
    }

    __syncthreads();

    // -----------------------------
    // 2. Compute RMSNorm scale
    // -----------------------------
    float sum_sq = block_reduce_sum(local_sum_sq);
    __shared__ float shared_sum_sq;
    if (tid == 0) {
        shared_sum_sq = sum_sq;
    }
    __syncthreads();

    float mean_sq = shared_sum_sq / static_cast<float>(d_model);
    float rms_inv = rsqrtf(mean_sq + eps);

    // -----------------------------
    // 3. Compute Y_fp and find absmax
    // -----------------------------
    float local_absmax = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float y_fp = x_fp_s[i] * rms_inv * gamma[i];
        x_fp_s[i] = y_fp;  // reuse shared memory to store Y_fp
        local_absmax = fmaxf(local_absmax, fabsf(y_fp));
    }
    __syncthreads();

    float absmax = block_reduce_max(local_absmax);

    __shared__ float shared_scale_y;

    if (tid == 0) {
        float sy = (absmax > 0.0f && isfinite(absmax)) ? absmax / 127.0f : 1.0f;
        shared_scale_y = sy;
        scale_y[row] = sy;
    }
    __syncthreads();

    float inv_sy = 1.0f / shared_scale_y;

    // -----------------------------
    // 4. Quantize Y_fp -> Y_int8
    // -----------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float y_fp = x_fp_s[i];

        float qf = y_fp * inv_sy;
        if (!isfinite(qf)) qf = 0.0f;

        int q = __float2int_rn(qf);
        q = max(-127, min(127, q));

        y_int8[row_offset + i] = static_cast<int8_t>(q);
    }
}


std::tuple<torch::Tensor, torch::Tensor> rmsnorm_int8_shared_cuda(
    torch::Tensor x,      // INT8 - (..., d_model)
    torch::Tensor scale_x,  // Float32 scalar scale for INT8 input
    torch::Tensor gamma,  // (d_model)
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kChar, "x must be int8");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == torch::kFloat32, "scale_x must be float32");
    
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D (d_model)");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32 || gamma.scalar_type() == torch::kBFloat16,
                "gamma must be float32 or bfloat16");

    TORCH_CHECK(x.numel() > 0, "x must be non-empty");
    TORCH_CHECK(x.size(-1) == gamma.size(0), "gamma size must match x.size(-1)");

    auto x_contig = x.contiguous();
    auto scale_x_contig = scale_x.contiguous();
    auto gamma_contig = gamma.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");
    const int64_t n_rows = x_contig.numel() / d_model;
    TORCH_CHECK(n_rows * d_model == x_contig.numel(), "x.numel must be divisible by d_model");

    TORCH_CHECK(scale_x.is_cuda() && scale_x.scalar_type() == torch::kFloat32,
            "scale_x must be CUDA float32");
    TORCH_CHECK(scale_x.numel() == n_rows, "scale_x must have n_rows elements");

    // output same shape as input
    auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar));
    auto scale_y = torch::empty({n_rows}, x_contig.options().dtype(torch::kFloat32));

    // Determine block and grid sizes
    int threads = (int)std::min<int64_t>(d_model, 512);
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)n_rows);
    size_t shmem_bytes = d_model * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(
        gamma_contig.scalar_type(),
        "rmsnorm_int8_shared_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            rmsnorm_int8_shared_kernel<float><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<float>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            rmsnorm_int8_shared_kernel<at::BFloat16><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<at::BFloat16>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y.view(x.sizes()), scale_y.view(scale_x.sizes()));  
}

