#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include "gemm_utils.cu"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

template <typename scalar_t>
__global__ void row_quant_int8_kernel(
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

    const int64_t row_offset =
        static_cast<int64_t>(row) * K;

    // ========================================================
    // 1. Row absmax
    // ========================================================

    float local_max = 0.0f;

    for (int i = tid; i < K; i += blockDim.x) {
        const float x =
            static_cast<float>(input[row_offset + i]);

        local_max = fmaxf(local_max, fabsf(x));
    }

    float max_val =
        block_reduce_max(local_max);

    max_val = fmaxf(max_val, 1e-8f);

    // ========================================================
    // 2. Scale
    // ========================================================

    const float scale = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;

    if (tid == 0) {
        scales[row] = scale;
    }

    // ========================================================
    // 3. Quantize
    // ========================================================

    for (int i = tid; i < K; i += blockDim.x) {
        const float x =
            static_cast<float>(input[row_offset + i]);

        output[row_offset + i] =
            quantize_int8(x * inv_scale);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
quantize_row_int8_cuda(torch::Tensor input)
{
    TORCH_CHECK(
        input.is_cuda(),
        "input must be CUDA"
    );

    TORCH_CHECK(
        input.dim() >= 2,
        "input must be at least 2D"
    );

    input = input.contiguous();

    const int64_t K = input.size(-1);
    const int64_t rows = input.numel() / K;

    // ========================================================
    // Output
    // ========================================================

    auto output = torch::empty(
        input.sizes(),
        input.options().dtype(torch::kInt8)
    );

    std::vector<int64_t> scale_shape(
        input.sizes().begin(),
        input.sizes().end() - 1
    );

    auto scales = torch::empty(
        scale_shape,
        input.options().dtype(torch::kFloat32)
    );

    // ========================================================
    // Launch
    // ========================================================

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "row_quant_int8",
        [&] {
            row_quant_int8_kernel<scalar_t>
                <<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int8_t>(),
                    scales.data_ptr<float>(),
                    static_cast<int>(rows),
                    static_cast<int>(K)
                );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output, scales);
}