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

__global__ void element_wise_mul_int8_rowwise_2d_kernel(
    const int8_t* __restrict__ a,
    const float* __restrict__ scale_a_row,  // (M,)
    const int8_t* __restrict__ b,
    const float* __restrict__ scale_b_row,  // (M,)
    int8_t* __restrict__ c,
    const float* __restrict__ scale_c_row,  // (M,)
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / N;

        float sa = scale_a_row[row];
        float sb = scale_b_row[row];
        float sc = scale_c_row[row];
        float mul = (sa * sb) / sc;   // sc must be non-zero

        int32_t prod = (int32_t)a[i] * (int32_t)b[i];
        float cf = (float)prod * mul;

        int q = __float2int_rn(cf);
        q = max(-128, min(127, q));
        c[i] = (int8_t)q;
    }
}

__global__ void element_wise_mul_int8_rowwise_3d_kernel(
    const int8_t* __restrict__ a,
    const float* __restrict__ scale_a_bm,  // (B*M) flatten of (B,M)
    const int8_t* __restrict__ b,
    const float* __restrict__ scale_b_bm,  // (B*M)
    int8_t* __restrict__ c,
    const float* __restrict__ scale_c_bm,  // (B*M)
    int64_t B,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = B * M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i / N;      // in [0, B*M)
        int64_t row = tmp % M;
        int64_t bch = tmp / M;

        int64_t s_idx = bch * M + row;

        float sa = scale_a_bm[s_idx];
        float sb = scale_b_bm[s_idx];
        float sc = scale_c_bm[s_idx];
        float mul = (sa * sb) / sc;

        int32_t prod = (int32_t)a[i] * (int32_t)b[i];
        float cf = (float)prod * mul;

        int q = __float2int_rn(cf);
        q = max(-128, min(127, q));
        c[i] = (int8_t)q;
    }
}

torch::Tensor element_wise_mul_int8_rowwise_cuda(
    torch::Tensor a,              // int8, (M,N) or (B,M,N)
    torch::Tensor scale_a,        // float32, (M,) or (B,M)
    torch::Tensor b,              // int8, same shape
    torch::Tensor scale_b,        // float32, (M,) or (B,M)
    torch::Tensor scale_c         // float32, (M,) or (B,M)
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda() && scale_c.is_cuda(),
                "scales must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kChar && b.dtype() == torch::kChar, "a,b must be int8");
    TORCH_CHECK(scale_a.dtype() == torch::kFloat32 &&
                scale_b.dtype() == torch::kFloat32 &&
                scale_c.dtype() == torch::kFloat32, "scales must be float32");

    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a,b must be contiguous");
    TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous() && scale_c.is_contiguous(),
                "scales must be contiguous");

    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    TORCH_CHECK(a.dim() == 2 || a.dim() == 3, "a must be 2D or 3D");

    auto c = torch::empty_like(a);

    int threads = 256;

    // cap blocks (like your sigmoid style)
    int device = a.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;

    auto stream = at::cuda::getCurrentCUDAStream();

    if (a.dim() == 2) {
        int64_t M = a.size(0);
        int64_t N = a.size(1);

        TORCH_CHECK(scale_a.dim() == 1 && scale_a.size(0) == M, "scale_a must be (M,)");
        TORCH_CHECK(scale_b.dim() == 1 && scale_b.size(0) == M, "scale_b must be (M,)");
        TORCH_CHECK(scale_c.dim() == 1 && scale_c.size(0) == M, "scale_c must be (M,)");

        int64_t total = M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        element_wise_mul_int8_rowwise_2d_kernel<<<blocks, threads, 0, stream>>>(
            a.data_ptr<int8_t>(),
            scale_a.data_ptr<float>(),
            b.data_ptr<int8_t>(),
            scale_b.data_ptr<float>(),
            c.data_ptr<int8_t>(),
            scale_c.data_ptr<float>(),
            M, N
        );
    } else {
        int64_t B = a.size(0);
        int64_t M = a.size(1);
        int64_t N = a.size(2);

        TORCH_CHECK(scale_a.dim() == 2 && scale_a.size(0) == B && scale_a.size(1) == M,
                    "scale_a must be (B,M)");
        TORCH_CHECK(scale_b.dim() == 2 && scale_b.size(0) == B && scale_b.size(1) == M,
                    "scale_b must be (B,M)");
        TORCH_CHECK(scale_c.dim() == 2 && scale_c.size(0) == B && scale_c.size(1) == M,
                    "scale_c must be (B,M)");

        int64_t total = B * M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        element_wise_mul_int8_rowwise_3d_kernel<<<blocks, threads, 0, stream>>>(
            a.data_ptr<int8_t>(),
            scale_a.data_ptr<float>(),
            b.data_ptr<int8_t>(),
            scale_b.data_ptr<float>(),
            c.data_ptr<int8_t>(),
            scale_c.data_ptr<float>(),
            B, M, N
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

__global__ void element_add_int8_rowwise_kernel(
    const int8_t* __restrict__ a,
    const float* __restrict__ scale_a,   // (M,) if B==1 else (B*M)
    const int8_t* __restrict__ b,
    const float* __restrict__ scale_b,   // same
    int8_t* __restrict__ c,
    const float* __restrict__ scale_c,   // same
    int64_t B,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = B * M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i / N;       // in [0, B*M)
        int64_t row = tmp % M;
        int64_t bch = tmp / M;

        int64_t s_idx = (B == 1) ? row : (bch * M + row);

        float sa = scale_a[s_idx];
        float sb = scale_b[s_idx];
        float sc = scale_c[s_idx];         // must be non-zero
        float inv_sc = 1.0f / sc;

        float cf = ((float)a[i] * sa + (float)b[i] * sb) * inv_sc;

        int q = __float2int_rn(cf);
        q = max(-128, min(127, q));
        c[i] = (int8_t)q;
    }
}

torch::Tensor element_add_int8_cuda(
    torch::Tensor a,              // int8 (M,N) or (B,M,N)
    torch::Tensor scale_a,        // float32 (M,) or (B,M)
    torch::Tensor b,              // int8 same shape
    torch::Tensor scale_b,        // float32 (M,) or (B,M)
    torch::Tensor scale_c         // float32 (M,) or (B,M)
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda() && scale_c.is_cuda(), "scales must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kChar && b.dtype() == torch::kChar, "a,b must be int8");
    TORCH_CHECK(scale_a.dtype() == torch::kFloat32 &&
                scale_b.dtype() == torch::kFloat32 &&
                scale_c.dtype() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a,b must be contiguous");
    TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous() && scale_c.is_contiguous(),
                "scales must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    TORCH_CHECK(a.dim() == 2 || a.dim() == 3, "a must be 2D or 3D");

    int64_t B = 1, M, N;
    if (a.dim() == 2) {
        M = a.size(0);
        N = a.size(1);
        TORCH_CHECK(scale_a.dim() == 1 && scale_a.size(0) == M, "scale_a must be (M,)");
        TORCH_CHECK(scale_b.dim() == 1 && scale_b.size(0) == M, "scale_b must be (M,)");
        TORCH_CHECK(scale_c.dim() == 1 && scale_c.size(0) == M, "scale_c must be (M,)");
    } else {
        B = a.size(0);
        M = a.size(1);
        N = a.size(2);
        TORCH_CHECK(scale_a.dim() == 2 && scale_a.size(0) == B && scale_a.size(1) == M, "scale_a must be (B,M)");
        TORCH_CHECK(scale_b.dim() == 2 && scale_b.size(0) == B && scale_b.size(1) == M, "scale_b must be (B,M)");
        TORCH_CHECK(scale_c.dim() == 2 && scale_c.size(0) == B && scale_c.size(1) == M, "scale_c must be (B,M)");
    }

    auto c = torch::empty_like(a);

    int threads = 256;
    int64_t total = B * M * N;

    int device = a.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;

    int blocks = (int)((total + threads - 1) / threads);
    blocks = std::min(blocks, max_blocks);

    auto stream = at::cuda::getCurrentCUDAStream();
    element_add_int8_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<int8_t>(),
        scale_a.data_ptr<float>(),
        b.data_ptr<int8_t>(),
        scale_b.data_ptr<float>(),
        c.data_ptr<int8_t>(),
        scale_c.data_ptr<float>(),
        B, M, N
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

// ================================================================
// BACK UP
// ================================================================

// __global__ void element_wise_mul_int8_kernel(
//     const int8_t* __restrict__ a,
//     float scale_a,
//     const int8_t* __restrict__ b,
//     float scale_b,
//     int8_t* __restrict__ c,
//     float scale_c,
//     int64_t size)
// {
//     int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int64_t stride = blockDim.x * gridDim.x;

//     for (int64_t i = idx; i < size; i += stride) {
//         int32_t prod = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
//         float c_float = static_cast<float>(prod) * (scale_a * scale_b / scale_c);
//         int tmp = max(-128, min(127, __float2int_rn(c_float)));
//         c[i] = static_cast<int8_t>(tmp);
//     }
// }

// torch::Tensor element_wise_mul_int8_host(
//     torch::Tensor a,
//     float scale_a,
//     torch::Tensor b,
//     float scale_b,
//     float scale_c) 
// {
//     TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
//     TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
//     TORCH_CHECK(a.dtype() == torch::kChar, "a must be int8");
//     TORCH_CHECK(b.dtype() == torch::kChar, "b must be int8");
//     TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
//     TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
//     TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

//     auto size = a.numel();
//     auto c = torch::empty_like(a);

//     int threads = 256;
//     int blocks = (size + threads - 1) / threads;

//     auto stream = at::cuda::getCurrentCUDAStream();

//     element_wise_mul_int8_kernel<<<blocks, threads, 0, stream>>>(
//         a.data_ptr<int8_t>(),
//         scale_a,
//         b.data_ptr<int8_t>(),
//         scale_b,
//         c.data_ptr<int8_t>(),
//         scale_c,
//         size);

//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     return c;
// }