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


// ============================================================
// INT8 LayerNorm with implicit shared-memory reductions
//
// x_int8  : [num_rows, d_model]
// scale_x : [num_rows]
// gamma   : [d_model]
// beta    : [d_model]
// y_int8  : [num_rows, d_model]
// scale_y : [num_rows]
//
// One block computes one row.
// ============================================================

template <typename T>
__global__ void layernorm_int8_kernel(
    const int8_t* __restrict__ x_int8,
    const float*  __restrict__ scale_x,
    const T*      __restrict__ gamma,
    const T*      __restrict__ beta,
    int8_t*       __restrict__ y_int8,
    float*        __restrict__ scale_y,
    int d_model,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    __shared__ float shared_scale_y;

    float sx = scale_x[row];

    // --------------------------------------------------
    // 1. First pass: compute sum(x)
    // --------------------------------------------------
    float local_sum = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        local_sum += x_fp;
    }

    float sum = block_reduce_sum(local_sum);

    if (tid == 0) {
        shared_mean = sum / static_cast<float>(d_model);
    }

    __syncthreads();

    float mean = shared_mean;

    // --------------------------------------------------
    // 2. Second pass: compute variance
    // --------------------------------------------------
    float local_var_sum = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        float diff = x_fp - mean;
        local_var_sum += diff * diff;
    }

    float var_sum = block_reduce_sum(local_var_sum);

    if (tid == 0) {
        float var = var_sum / static_cast<float>(d_model);
        shared_inv_std = rsqrtf(var + eps);
    }

    __syncthreads();

    float inv_std = shared_inv_std;

    // --------------------------------------------------
    // 3. Third pass: compute y_fp and absmax(y)
    //
    // y = (x - mean) * inv_std * gamma + beta
    // --------------------------------------------------
    float local_absmax = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;

        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);

        float y_fp = (x_fp - mean) * inv_std * g + b;

        local_absmax = fmaxf(local_absmax, fabsf(y_fp));
    }

    float absmax = block_reduce_max(local_absmax);

    if (tid == 0) {
        float sy = (absmax > 0.0f && isfinite(absmax))
                 ? absmax / 127.0f
                 : 1.0f;

        shared_scale_y = sy;
        scale_y[row] = sy;
    }

    __syncthreads();

    float inv_sy = 1.0f / shared_scale_y;

    // --------------------------------------------------
    // 4. Fourth pass: recompute y_fp and quantize
    // --------------------------------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;

        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);

        float y_fp = (x_fp - mean) * inv_std * g + b;

        float qf = y_fp * inv_sy;

        if (!isfinite(qf)) {
            qf = 0.0f;
        }

        int q = __float2int_rn(qf);
        q = max(-127, min(127, q));

        y_int8[row_offset + i] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> layernorm_int8_cuda(
    torch::Tensor x,        // int8, shape (..., d_model)
    torch::Tensor scale_x,  // float32, shape [num_rows]
    torch::Tensor gamma,    // float32 or bfloat16, shape [d_model]
    torch::Tensor beta,     // float32 or bfloat16, shape [d_model]
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kChar, "x must be int8");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == torch::kFloat32, "scale_x must be float32");

    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA");

    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");

    TORCH_CHECK(
        gamma.scalar_type() == torch::kFloat32 ||
        gamma.scalar_type() == torch::kBFloat16,
        "gamma must be float32 or bfloat16"
    );

    TORCH_CHECK(
        beta.scalar_type() == gamma.scalar_type(),
        "beta must have same dtype as gamma"
    );

    auto x_contig = x.contiguous();
    auto scale_x_contig = scale_x.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto beta_contig = beta.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");

    const int64_t n_rows = x_contig.numel() / d_model;

    TORCH_CHECK(
        n_rows * d_model == x_contig.numel(),
        "x.numel must be divisible by d_model"
    );

    TORCH_CHECK(
        gamma_contig.size(0) == d_model,
        "gamma size must match d_model"
    );

    TORCH_CHECK(
        beta_contig.size(0) == d_model,
        "beta size must match d_model"
    );

    TORCH_CHECK(
        scale_x_contig.numel() == n_rows,
        "scale_x must have n_rows elements"
    );

    auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar));
    auto scale_y = torch::empty({n_rows}, x_contig.options().dtype(torch::kFloat32));


    int threads = static_cast<int>(std::min<int64_t>(d_model, 512));
    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(n_rows));

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(
        gamma_contig.scalar_type(),
        "layernorm_int8_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            layernorm_int8_kernel<float><<<grid, block, 0, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<float>(),
                beta_contig.data_ptr<float>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            layernorm_int8_kernel<at::BFloat16><<<
                grid,
                block,
                0,
                stream
            >>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<at::BFloat16>(),
                beta_contig.data_ptr<at::BFloat16>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        y.view(x.sizes()),
        scale_y.view(scale_x.sizes())
    );
}


template <typename T>
__global__ void layernorm_int8_shared_kernel(
    const int8_t* __restrict__ x_int8,
    const float*  __restrict__ scale_x,
    const T*      __restrict__ gamma,
    const T*      __restrict__ beta,
    int8_t*       __restrict__ y_int8,
    float*        __restrict__ scale_y,
    int d_model,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    extern __shared__ float smem[];    // Dynamic shared memory, shape = [d_model + blockDim.x]
    float* x_fp_s = smem;              // [d_model]
    float* red_s  = smem + d_model;    // [blockDim.x]

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    __shared__ float shared_scale_y;

    float sx = scale_x[row];

    // --------------------------------------------------
    // 1. Dequantize x_int8 -> x_fp_s
    //    Also compute local sum
    // --------------------------------------------------
    float local_sum = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;

        x_fp_s[i] = x_fp;

        local_sum += x_fp;
    }

    __syncthreads();

    // --------------------------------------------------
    // 2. Explicit shared-memory reduction for sum(x)
    // --------------------------------------------------
    red_s[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_s[tid] += red_s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = red_s[0];
        shared_mean = sum / static_cast<float>(d_model);
    }

    __syncthreads();

    float mean = shared_mean;

    // --------------------------------------------------
    // 3. Compute local variance sum
    //    var = mean((x - mean)^2)
    // --------------------------------------------------
    float local_var_sum = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float diff = x_fp_s[i] - mean;
        local_var_sum += diff * diff;
    }

    __syncthreads();

    // --------------------------------------------------
    // 4. Explicit shared-memory reduction for variance sum
    // --------------------------------------------------
    red_s[tid] = local_var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_s[tid] += red_s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float var = red_s[0] / static_cast<float>(d_model);
        shared_inv_std = rsqrtf(var + eps);
    }

    __syncthreads();

    float inv_std = shared_inv_std;

    // --------------------------------------------------
    // 5. Compute y_fp and local absmax
    //
    // y = (x - mean) * inv_std * gamma + beta
    // --------------------------------------------------
    float local_absmax = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);

        float y_fp = (x_fp_s[i] - mean) * inv_std * g + b;

        // Reuse x_fp_s to store y_fp.
        // This is safe because we no longer need raw x_fp after this.
        x_fp_s[i] = y_fp;

        local_absmax = fmaxf(local_absmax, fabsf(y_fp));
    }

    __syncthreads();

    // --------------------------------------------------
    // 6. Explicit shared-memory reduction for absmax(y)
    // --------------------------------------------------
    red_s[tid] = local_absmax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_s[tid] = fmaxf(red_s[tid], red_s[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float absmax = red_s[0];

        float sy = (absmax > 0.0f && isfinite(absmax))
                 ? absmax / 127.0f
                 : 1.0f;

        shared_scale_y = sy;
        scale_y[row] = sy;
    }

    __syncthreads();

    float inv_sy = 1.0f / shared_scale_y;

    // --------------------------------------------------
    // 7. Quantize y_fp -> y_int8
    // --------------------------------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float y_fp = x_fp_s[i];

        float qf = y_fp * inv_sy;

        if (!isfinite(qf)) {
            qf = 0.0f;
        }

        int q = __float2int_rn(qf);
        q = max(-127, min(127, q));

        y_int8[row_offset + i] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> layernorm_int8_shared_cuda(
    torch::Tensor x,        // int8, shape (..., d_model)
    torch::Tensor scale_x,  // float32, shape same as rows
    torch::Tensor gamma,    // float32 or bfloat16, shape (d_model)
    torch::Tensor beta,     // float32 or bfloat16, shape (d_model)
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kChar, "x must be int8");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == torch::kFloat32, "scale_x must be float32");

    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA");

    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.dim() == 1, "beta must be 1D");

    TORCH_CHECK(
        gamma.scalar_type() == torch::kFloat32 ||
        gamma.scalar_type() == torch::kBFloat16,
        "gamma must be float32 or bfloat16"
    );

    TORCH_CHECK(
        beta.scalar_type() == gamma.scalar_type(),
        "beta must have same dtype as gamma"
    );

    auto x_contig = x.contiguous();
    auto scale_x_contig = scale_x.contiguous();
    auto gamma_contig = gamma.contiguous();
    auto beta_contig = beta.contiguous();

    const int64_t d_model = x_contig.size(-1);
    const int64_t n_rows = x_contig.numel() / d_model;

    TORCH_CHECK(d_model > 0, "d_model must be > 0");
    TORCH_CHECK(x_contig.numel() % d_model == 0, "x.numel must be divisible by d_model");

    TORCH_CHECK(gamma_contig.size(0) == d_model, "gamma size must match d_model");
    TORCH_CHECK(beta_contig.size(0) == d_model, "beta size must match d_model");

    TORCH_CHECK(scale_x_contig.numel() == n_rows, "scale_x must have n_rows elements");

    auto y = torch::empty_like(x_contig, x_contig.options().dtype(torch::kChar));
    auto scale_y = torch::empty({n_rows}, x_contig.options().dtype(torch::kFloat32));

    int threads = static_cast<int>(std::min<int64_t>(d_model, 512));

    // Make threads power-of-two for explicit reduction.
    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) {
            p <<= 1;
        }
        threads = p;
    }

    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(n_rows));

    // Important:
    // x_fp_s needs d_model floats.
    // red_s needs threads floats.
    size_t shmem_bytes = (d_model + threads) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_SWITCH(
        gamma_contig.scalar_type(),
        "layernorm_int8_shared_cuda",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            layernorm_int8_shared_kernel<float><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<float>(),
                beta_contig.data_ptr<float>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            layernorm_int8_shared_kernel<at::BFloat16><<<grid, block, shmem_bytes, stream>>>(
                x_contig.data_ptr<int8_t>(),
                scale_x_contig.data_ptr<float>(),
                gamma_contig.data_ptr<at::BFloat16>(),
                beta_contig.data_ptr<at::BFloat16>(),
                y.data_ptr<int8_t>(),
                scale_y.data_ptr<float>(),
                static_cast<int>(d_model),
                eps
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        y.view(x.sizes()),
        scale_y.view(scale_x.sizes())
    );
}