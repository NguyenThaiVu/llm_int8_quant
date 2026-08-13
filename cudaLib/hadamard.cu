#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include "gemm_utils.cu"

constexpr int HADAMARD_BLOCK_SIZE = 256;
// constexpr int HADAMARD_THREADS = 256;

constexpr int HADAMARD_THREADS = 32;
constexpr int VALUES_PER_THREAD = 8;


__device__ __forceinline__
void hadamard_8(float x[8])
{
    #pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;

            const float a = x[idx];
            const float b = x[idx + stride];

            x[idx] = a + b;
            x[idx + stride] = a - b;
        }
    }
}


template <typename scalar_t>
__global__ void hadamard_rotation_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int block_id = blockIdx.y;
    const int lane = threadIdx.x;

    if (row >= rows) {
        return;
    }

    constexpr int BLOCK_H = 256;
    constexpr float SCALE = 1.0f / 16.0f;

    const int base_k = block_id * BLOCK_H;

    const int64_t offset =
        static_cast<int64_t>(row) * K + base_k;

    // ========================================================
    // 1. Each thread loads 8 contiguous values
    // ========================================================

    const Vec8<scalar_t> input_vec =
        reinterpret_cast<const Vec8<scalar_t>*>(
            input + offset
        )[lane];

    float x[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = static_cast<float>(input_vec.x[i]);
    }

    // ========================================================
    // 2. H8 inside each thread
    //
    // Register-only Hadamard.
    // ========================================================

    hadamard_8(x);

    // ========================================================
    // 3. H32 across the warp
    //
    // 5 stages:
    // 1, 2, 4, 8, 16
    // ========================================================

    #pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {

        const float sign =
            (lane & stride) ? -1.0f : 1.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {

            const float other =
                __shfl_xor_sync(
                    0xffffffff,
                    x[i],
                    stride
                );

            x[i] = sign * x[i] + other;
        }
    }

    // ========================================================
    // 4. Normalize + convert back to input type
    // ========================================================

    Vec8<scalar_t> output_vec;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        output_vec.x[i] =
            static_cast<scalar_t>(x[i] * SCALE);
    }

    // ========================================================
    // 5. Vectorized store
    // ========================================================

    reinterpret_cast<Vec8<scalar_t>*>(
        output + offset
    )[lane] = output_vec;
}

torch::Tensor apply_hadamard_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");

    input = input.contiguous();

    const int64_t K = input.size(-1);

    TORCH_CHECK(
        K % HADAMARD_BLOCK_SIZE == 0,
        "last dimension must be divisible by ",
        HADAMARD_BLOCK_SIZE,
        ", got K = ",
        K
    );

    const int64_t rows = input.numel() / K;

    auto output = torch::empty_like(input);

    dim3 grid(rows, K / HADAMARD_BLOCK_SIZE);

    dim3 block(HADAMARD_THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "hadamard_rotation",
        [&] {
            hadamard_rotation_kernel<scalar_t><<<grid, block, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows,
                K
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

