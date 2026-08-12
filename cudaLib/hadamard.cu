#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

constexpr int HADAMARD_BLOCK_SIZE = 256;

template <typename scalar_t, int BLOCK_H>
__global__ void block_fwht_rotate_rows_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int block_id = blockIdx.y;

    if (row >= rows)
        return;

    const int base_k = block_id * BLOCK_H;

    __shared__ float smem[BLOCK_H];

    const scalar_t* row_in = input + (int64_t)row * K;
    scalar_t* row_out = output + (int64_t)row * K;


    // Load one Hadamard block
    for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {
        smem[i] = static_cast<float>(row_in[base_k + i]);
    }

    __syncthreads();


    // Fast Hadamard Transform
    for (int stride = 1; stride < BLOCK_H; stride <<= 1) {

        for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {

            if ((i & stride) == 0) {
                const int j = i ^ stride;

                float a = smem[i];
                float b = smem[j];

                smem[i] = a + b;
                smem[j] = a - b;
            }
        }

        __syncthreads();
    }


    // Normalize + store
    const float scale = rsqrtf((float)BLOCK_H);

    for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {
        row_out[base_k + i] =
            static_cast<scalar_t>(smem[i] * scale);
    }
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

    dim3 block(HADAMARD_BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "block_fwht_rotate_rows",
        [&] {
            block_fwht_rotate_rows_kernel<scalar_t, HADAMARD_BLOCK_SIZE><<<grid, block, 0, stream>>>(
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

template <typename scalar_t, int BLOCK_H>
__global__ void hadamard_block_max_kernel(
    const scalar_t* __restrict__ input,
    float* __restrict__ block_max,
    int rows,
    int K)
{
    const int row = blockIdx.x;
    const int block_id = blockIdx.y;

    if (row >= rows) {
        return;
    }

    const int base_k = block_id * BLOCK_H;
    const int64_t row_offset = static_cast<int64_t>(row) * K;

    __shared__ float smem[BLOCK_H];

    // ========================================================
    // 1. Load one Hadamard block
    // ========================================================

    for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {
        smem[i] = static_cast<float>(
            input[row_offset + base_k + i]
        );
    }

    __syncthreads();

    // ========================================================
    // 2. Fast Hadamard Transform
    // ========================================================

    for (int stride = 1; stride < BLOCK_H; stride <<= 1) {
        for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {
            if ((i & stride) == 0) {
                const int j = i ^ stride;

                const float a = smem[i];
                const float b = smem[j];

                smem[i] = a + b;
                smem[j] = a - b;
            }
        }

        __syncthreads();
    }

    // ========================================================
    // 3. Compute block absmax
    // ========================================================

    const float hadamard_scale = rsqrtf(static_cast<float>(BLOCK_H));

    float local_max = 0.0f;

    for (int i = threadIdx.x; i < BLOCK_H; i += blockDim.x) {
        const float v = smem[i] * hadamard_scale;
        local_max = fmaxf(local_max, fabsf(v));
    }

    const float max_val = block_reduce_max(local_max);

    // ========================================================
    // 4. Store one maximum per Hadamard block
    // ========================================================

    if (threadIdx.x == 0) {
        const int blocks_per_row = K / BLOCK_H;

        block_max[
            static_cast<int64_t>(row) * blocks_per_row + block_id
        ] = max_val;
    }
}

template <typename scalar_t, int BLOCK_H>
__global__ void hadamard_row_quant_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ block_max,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int K,
    int blocks_per_row)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int64_t row_offset = static_cast<int64_t>(row) * K;

    __shared__ float smem[BLOCK_H];
    __shared__ float row_inv_scale;

    // ========================================================
    // 1. Reduce block maxima to row maximum
    // ========================================================

    float local_max = 0.0f;

    for (int i = tid; i < blocks_per_row; i += blockDim.x) {
        const float v =
            block_max[
                static_cast<int64_t>(row) * blocks_per_row + i
            ];

        local_max = fmaxf(local_max, v);
    }

    float max_val = block_reduce_max(local_max);

    // ========================================================
    // 2. Compute row-wise quantization scale
    // ========================================================

    if (tid == 0) {
        max_val = fmaxf(max_val, 1e-8f);

        const float scale = max_val / 127.0f;

        scales[row] = scale;
        row_inv_scale = 1.0f / scale;
    }

    __syncthreads();

    const float inv_scale = row_inv_scale;
    const float hadamard_scale = rsqrtf(static_cast<float>(BLOCK_H));

    // ========================================================
    // 3. Recompute each Hadamard block
    // ========================================================

    for (int block_id = 0; block_id < blocks_per_row; ++block_id) {
        const int base_k = block_id * BLOCK_H;

        // ----------------------------------------------------
        // Load original input
        // ----------------------------------------------------

        for (int i = tid; i < BLOCK_H; i += blockDim.x) {
            smem[i] = static_cast<float>(
                input[row_offset + base_k + i]
            );
        }

        __syncthreads();

        // ----------------------------------------------------
        // Hadamard transform
        // ----------------------------------------------------

        for (int stride = 1; stride < BLOCK_H; stride <<= 1) {
            for (int i = tid; i < BLOCK_H; i += blockDim.x) {
                if ((i & stride) == 0) {
                    const int j = i ^ stride;

                    const float a = smem[i];
                    const float b = smem[j];

                    smem[i] = a + b;
                    smem[j] = a - b;
                }
            }

            __syncthreads();
        }

        // ----------------------------------------------------
        // Normalize + quantize directly to INT8
        // ----------------------------------------------------

        for (int i = tid; i < BLOCK_H; i += blockDim.x) {
            const float v = smem[i] * hadamard_scale;
            const float q = v * inv_scale;

            output[row_offset + base_k + i] =
                quantize_int8(q);
        }

        __syncthreads();
    }
}

std::tuple<torch::Tensor, torch::Tensor>fusion_hadamard_quant_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");

    input = input.contiguous();

    const int64_t K = input.size(-1);

    TORCH_CHECK(
        K % HADAMARD_BLOCK_SIZE == 0,
        "K must be divisible by ",
        HADAMARD_BLOCK_SIZE,
        ", got K = ",
        K
    );

    const int64_t rows = input.numel() / K;
    const int blocks_per_row = K / HADAMARD_BLOCK_SIZE;

    // ========================================================
    // Outputs
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
    // Small temporary buffer
    // ========================================================

    auto block_max = torch::empty(
        {rows, blocks_per_row},
        input.options().dtype(torch::kFloat32)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ========================================================
    // Kernel 1:
    // Hadamard + block maximum
    // ========================================================

    dim3 grid_hadamard(rows, blocks_per_row);
    dim3 block_hadamard(HADAMARD_BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "hadamard_block_max",
        [&] {
            hadamard_block_max_kernel<
                scalar_t,
                HADAMARD_BLOCK_SIZE
            ><<<grid_hadamard, block_hadamard, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                block_max.data_ptr<float>(),
                rows,
                K
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // ========================================================
    // Kernel 2:
    // Row max + recompute Hadamard + INT8 quantization
    // ========================================================

    constexpr int QUANT_THREADS = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "hadamard_row_quant",
        [&] {
            hadamard_row_quant_kernel<
                scalar_t,
                HADAMARD_BLOCK_SIZE
            ><<<rows, QUANT_THREADS, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                block_max.data_ptr<float>(),
                output.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                K,
                blocks_per_row
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output, scales);
}