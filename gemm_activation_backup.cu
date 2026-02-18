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

template <typename ScaleT>
__global__ void sigmoid_int8_kernel(
    const int8_t* __restrict__ input, 
    ScaleT scale_in,
    int8_t* __restrict__ output,
    ScaleT scale_out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float s_in  = to_float(scale_in);
    float s_out = to_float(scale_out);
    float inv_s_out = 1.0f / s_out;   // scale_out must be non-zero

    for (int64_t i = idx; i < n; i += stride) {
        float x = (float)input[i] * s_in;
        float y = 1.0f / (1.0f + __expf(-x));
        int q = __float2int_rn(y * inv_s_out);
        output[i] = clamp_int8(q);
    }
}

torch::Tensor sigmoid_int8_cuda(torch::Tensor input,
    float scale_in,
    float scale_out
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kChar, "Input tensor must be int8");
    TORCH_CHECK(scale_out != 0.0f, "scale_out must be non-zero");

    auto output = torch::empty_like(input);
    int64_t n = input.numel();

    const int threads = 256;

    // cap blocks: grid-stride loop handles the rest
    int device = input.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20; // tweak 10–30
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > max_blocks ? max_blocks : blocks;

    auto stream = at::cuda::getCurrentCUDAStream();
    sigmoid_int8_kernel<float><<<blocks, threads, 0, stream>>>(
        input.data_ptr<int8_t>(),
        scale_in,
        output.data_ptr<int8_t>(),
        scale_out,
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename scalar_t>
__global__ void silu_int8_kernel(
    const int8_t* __restrict__ input,
    scalar_t scale_in,
    int8_t* __restrict__ output,
    scalar_t scale_out,
    int64_t num_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    float inv_scale_out = 1.0f / to_float(scale_out);

    for (int64_t i = idx; i < num_elements; i += stride) {
        float x = static_cast<float>(input[i]) * scale_in;
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float silu_x = x * sigmoid_x;
        int q = __float2int_rn(silu_x * inv_scale_out);
        output[i] = clamp_int8(q);
    }
}

torch::Tensor silu_int8_cuda(
    torch::Tensor input,
    float scale_in,
    float scale_out) 
{
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kChar, "Input tensor must be int8");

    auto output = torch::empty_like(input);
    int64_t n = input.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream();

    silu_int8_kernel<<<blocks, threads, 0, stream>>>(
        input.data_ptr<int8_t>(),
        scale_in,
        output.data_ptr<int8_t>(),
        scale_out,
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}