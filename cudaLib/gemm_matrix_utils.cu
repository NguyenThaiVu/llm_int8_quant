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

std::vector<torch::Tensor> dequant_transpose_requant_3d_host(
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


std::vector<torch::Tensor> dequant_transpose_requant_4d_host(
    torch::Tensor values_int8,   // int8,  [B, H, L, D]
    torch::Tensor values_scale   // float, [B, H, L]
) {
    TORCH_CHECK(values_int8.is_cuda(), "values_int8 must be CUDA");
    TORCH_CHECK(values_scale.is_cuda(), "values_scale must be CUDA");

    TORCH_CHECK(values_int8.dtype() == torch::kInt8,
                "values_int8 must be int8");

    TORCH_CHECK(values_scale.dtype() == torch::kFloat32 ||
                values_scale.dtype() == torch::kFloat16 ||
                values_scale.dtype() == torch::kBFloat16,
                "values_scale must be float, half, or bfloat16");

    TORCH_CHECK(values_int8.dim() == 4,
                "values_int8 must be [B, H, L, D]");

    TORCH_CHECK(values_scale.dim() == 3,
                "values_scale must be [B, H, L]");

    int64_t B = values_int8.size(0);
    int64_t H = values_int8.size(1);
    int64_t L = values_int8.size(2);
    int64_t D = values_int8.size(3);

    TORCH_CHECK(values_scale.size(0) == B &&
                values_scale.size(1) == H &&
                values_scale.size(2) == L,
                "values_scale shape must be [B, H, L]");

    int64_t BH = B * H;

    auto values_int8_contig = values_int8.contiguous();
    auto values_scale_contig = values_scale.contiguous();

    // Flatten batch and head:
    // [B, H, L, D] -> [B*H, L, D]
    // [B, H, L]    -> [B*H, L]
    auto values_int8_3d = values_int8_contig.reshape({BH, L, D});
    auto values_scale_2d = values_scale_contig.reshape({BH, L});

    auto out_q_3d = torch::empty(
        {BH, D, L},
        values_int8.options()
    );

    auto out_s_2d = torch::empty(
        {BH, D},
        values_scale.options().dtype(torch::kFloat)
    );

    dim3 grid((unsigned)D, (unsigned)BH);
    int threads = 256;
    size_t shm = threads * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    dequant_transpose_requant_kernel<<<grid, threads, shm, stream>>>(
        values_int8_3d.data_ptr<int8_t>(),
        values_scale_2d.data_ptr<float>(),
        out_q_3d.data_ptr<int8_t>(),
        out_s_2d.data_ptr<float>(),
        (int)BH,
        (int)L,
        (int)D
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Restore 4D shape:
    // [B*H, D, L] -> [B, H, D, L]
    // [B*H, D]    -> [B, H, D]
    auto out_q = out_q_3d.reshape({B, H, D, L});
    auto out_s = out_s_2d.reshape({B, H, D});

    return {out_q, out_s};
}



// ================================================================
// context_smooth[m, n] = context[m, n] / smooth[n]
//
// scale[m] = max(abs(context_smooth[m, :])) / 127
//
// y_i8[m, n] = round(context_smooth[m, n] / scale[m])
// ================================================================
__global__ void quantize_row_int8_with_smooth_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ smooth,
    int8_t* __restrict__ y_i8,
    float* __restrict__ y_scale,
    int M,
    int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    const __nv_bfloat16* x_row =
        x + static_cast<size_t>(row) * K;

    int8_t* y_row =
        y_i8 + static_cast<size_t>(row) * K;

    __shared__ float smem[BLOCK_SIZE];

    // ------------------------------------------------------------
    // Pass 1: absmax of x[row, col] / smooth[col]
    // ------------------------------------------------------------
    float local_absmax = 0.0f;

    for (int col = tid; col < K; col += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[col]);

        // SmoothQuant division
        float s = smooth[col];
        v = v / s;

        local_absmax = fmaxf(local_absmax, fabsf(v));
    }

    smem[tid] = local_absmax;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    float absmax = smem[0];
    float scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;

    if (tid == 0) {
        y_scale[row] = scale;
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Pass 2: quantize x[row, col] / smooth[col]
    // ------------------------------------------------------------
    for (int col = tid; col < K; col += BLOCK_SIZE) {
        float v = __bfloat162float(x_row[col]);

        float s = smooth[col];
        v = v / s;

        int q = static_cast<int>(nearbyintf(v / scale));
        q = max(-128, min(127, q));

        y_row[col] = static_cast<int8_t>(q);
    }
}


std::tuple<torch::Tensor, torch::Tensor> quantize_row_int8_with_smooth_cuda(
    torch::Tensor x,
    torch::Tensor smooth
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(smooth.is_cuda(), "smooth must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be BFloat16");
    TORCH_CHECK(smooth.dtype() == torch::kFloat32, "smooth must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(smooth.dim() == 1, "smooth must be [K]");
    TORCH_CHECK(
        smooth.size(0) == x.size(1),
        "smooth.size(0) must match x.size(1)"
    );

    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));

    auto y_i8 = torch::empty(
        {M, K},
        torch::TensorOptions()
            .device(x.device())
            .dtype(torch::kInt8)
    );

    auto y_scale = torch::empty(
        {M},
        torch::TensorOptions()
            .device(x.device())
            .dtype(torch::kFloat32)
    );

    constexpr int threads = BLOCK_SIZE;
    dim3 grid(M);
    dim3 block(threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    quantize_row_int8_with_smooth_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        smooth.data_ptr<float>(),
        y_i8.data_ptr<int8_t>(),
        y_scale.data_ptr<float>(),
        M,
        K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y_i8, y_scale);
}