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

// In this script, we define the utility functions for matrix operations 
// used in our custom GEMM kernels.
// ===============================================================
// ===============================================================

/*
Description: 
This kernel performs the following operations on an input quantized matrix:
1. Dequantizes input int8 -> float
2. Transposes matrix from [H, L, D] to [H, D, L]
3. Requantizes the transposed matrix back to int8 with a new scale.

Inputs:
- in_q: [H, L, D] - int8
- in_scale: per-row scale [H, L] - float32
- H: number of heads
- L: sequence length
- D: head dimension

Outputs:
- out_q: [H, D, L] - int8
- out_scale: per-row scale [H, D] - float32
*/

__global__ void dequant_transpose_requant_kernel(
    const int8_t* __restrict__ in_q,        // [H, L, D]
    const float* __restrict__ in_scale,   // [H, L] or [L]
    int8_t* __restrict__ out_q,             // [H, D, L]
    float* __restrict__ out_scale,          // [H, D]
    int H, int L, int D
) {
    int h = blockIdx.y;
    int d = blockIdx.x;

    if (h >= H || d >= D) return;

    extern __shared__ float smem[];
    float* vals = smem;                     // [blockDim.x] temp values for this tile
    __shared__ float row_amax;

    float local_amax = 0.0f;

    // Pass 1: scan row and find max abs
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        int in_idx = (h * L + l) * D + d;   // [H, L, D]
        float s = (float)in_scale[h * L + l];
        float x = (float)in_q[in_idx] * s;
        local_amax = fmaxf(local_amax, fabsf(x));
    }

    // block reduction for max
    // simple shared-memory reduction version
    vals[threadIdx.x] = local_amax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_amax = vals[0];
        float new_s = (row_amax == 0.0f) ? 1.0f : (row_amax / 127.0f);
        out_scale[h * D + d] = new_s;
    }
    __syncthreads();

    float inv_new_s = (row_amax == 0.0f) ? 0.0f : (127.0f / row_amax);

    // Pass 2: requantize and write transposed output
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        int in_idx  = (h * L + l) * D + d;   // [H, L, D]
        int out_idx = (h * D + d) * L + l;   // [H, D, L]

        float s = (float)in_scale[h * L + l];
        float x = (float)in_q[in_idx] * s;

        int q = __float2int_rn(x * inv_new_s);
        q = max(-127, min(127, q));
        out_q[out_idx] = (int8_t)q;
    }
}

std::vector<torch::Tensor> dequant_transpose_requant_host(
    torch::Tensor values_int8,   // int8, [H, L, D]
    torch::Tensor values_scale   // float, [H, L]
) {
    TORCH_CHECK(values_int8.is_cuda(), "values_int8 must be CUDA");
    TORCH_CHECK(values_scale.is_cuda(), "values_scale must be CUDA");
    TORCH_CHECK(values_int8.dtype() == torch::kInt8, "values_int8 must be int8");
    TORCH_CHECK(values_int8.dim() == 3, "values_int8 must be [H, L, D]");

    int H = values_int8.size(0);
    int L = values_int8.size(1);
    int D = values_int8.size(2);

    auto out_q = torch::empty({H, D, L}, values_int8.options());
    auto out_s = torch::empty({H, D}, values_scale.options().dtype(torch::kFloat));

    dim3 grid(D, H);
    int threads = 256;
    size_t shm = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values_scale.scalar_type(), "fused_v_transform_cuda", [&] {
        dequant_transpose_requant_kernel<<<grid, threads, shm>>>(
            values_int8.data_ptr<int8_t>(),
            values_scale.data_ptr<float>(),
            out_q.data_ptr<int8_t>(),
            out_s.data_ptr<float>(),
            H, L, D
        );
    });

    return {out_q, out_s};
}