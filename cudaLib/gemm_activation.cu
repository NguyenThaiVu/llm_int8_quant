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


__device__ __forceinline__ float to_float(float x) { return x; }

__device__ __forceinline__ int8_t clamp_int8(int v) {
    v = v > 127 ? 127 : v;
    v = v < -128 ? -128 : v;
    return static_cast<int8_t>(v);
}

/*
This kernel computes SiLU(x) for int8 input with per-row quantization.
- Input:
    input: int8 tensor of shape (M,N)
    scale_in_row: float tensor of shape (M,) 
- Output:
    output: int8 tensor of shape (M,N)
*/
__global__ void silu_int8_rowwise_2d_kernel(
    const int8_t* __restrict__ input,       // shape (M, N)
    const float* __restrict__ scale_in_row, // (M,)
    int8_t* __restrict__ output,            // shape (M, N) 
    const float* scale_out,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t row = i / N;
        float s_in = scale_in_row[row];
        float s_out = scale_out[row];

        float x = (float)input[i] * s_in;
        float sigmoid_x = 1.0f / (1.0f + __expf(-x));
        float silu_x = x * sigmoid_x;

        int q = __float2int_rn(silu_x / s_out);
        output[i] = clamp_int8(q);
    }
}

/*
This kernel computes SiLU(x) for 3D int8 input with per-row quantization.
- Input:
    input: int8 tensor of shape (B,M,N)
    scale_in_bm: float tensor of shape (B,M)
- Output:
    output: int8 tensor of shape (B,M,N)

*/
__global__ void silu_int8_rowwise_3d_kernel(
    const int8_t* __restrict__ input,      // (B,M,N) flattened
    const float* __restrict__ scale_in_bm, // (B*M) flattened view of (B,M)
    int8_t* __restrict__ output,
    const float* __restrict__ scale_out,
    int64_t B,
    int64_t M,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t total = B * M * N;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i / N;        // in [0, B*M)
        int64_t row = tmp % M;      // in [0, M)
        int64_t b   = tmp / M;      // in [0, B)

        float s_in = scale_in_bm[b * M + row];
        float s_out = scale_out[b * M + row];

        float x = (float)input[i] * s_in;
        float sigmoid_x = 1.0f / (1.0f + __expf(-x));
        float silu_x = x * sigmoid_x;

        int q = __float2int_rn(silu_x / s_out);
        output[i] = clamp_int8(q);
    }
}


torch::Tensor silu_int8_cuda_rowwise(
    torch::Tensor input,      // int8, (M,N) or (B,M,N)
    torch::Tensor scale_in,   // float32, (M,) or (B,M)
    torch::Tensor scale_out   // float32, (M,) or (B,M)
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(scale_in.is_cuda(), "scale_in must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(scale_in.is_contiguous(), "scale_in must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kChar, "input must be int8");
    TORCH_CHECK(scale_in.dtype() == torch::kFloat32, "scale_in must be float32");
    TORCH_CHECK(scale_out.dtype() == torch::kFloat32, "scale_out must be float32");
    TORCH_CHECK(scale_out.sum().item<float>() != 0.0f, "scale_out must be non-zero");

    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "input must be 2D or 3D");

    auto output = torch::empty_like(input);

    int threads = 256;

    // block cap like your sigmoid version (recommended)
    int device = input.get_device();
    auto props = at::cuda::getDeviceProperties(device);
    int max_blocks = props->multiProcessorCount * 20;

    auto stream = at::cuda::getCurrentCUDAStream();

    if (input.dim() == 2) {
        int64_t M = input.size(0);
        int64_t N = input.size(1);

        TORCH_CHECK(scale_in.dim() == 1 && scale_in.size(0) == M,
                    "for 2D input, scale_in must be shape (M,)");
        TORCH_CHECK(scale_out.dim() == 0 || (scale_out.dim() == 1 && scale_out.size(0) == M),
                    "scale_out must be either a scalar or shape (M,)");

        int64_t total = M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        silu_int8_rowwise_2d_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<int8_t>(),
            scale_in.data_ptr<float>(),
            output.data_ptr<int8_t>(),
            scale_out.data_ptr<float>(), 
            M, N
        );
    } else {
        int64_t B = input.size(0);
        int64_t M = input.size(1);
        int64_t N = input.size(2);

        TORCH_CHECK(scale_in.dim() == 2 && scale_in.size(0) == B && scale_in.size(1) == M,
                    "for 3D input, scale_in must be shape (B,M)");
        TORCH_CHECK(scale_out.dim() == 0 || (scale_out.dim() == 2 && scale_out.size(0) == B && scale_out.size(1) == M),
                    "scale_out must be either a scalar or shape (B,M)");

        int64_t total = B * M * N;
        int blocks = (int)((total + threads - 1) / threads);
        blocks = std::min(blocks, max_blocks);

        silu_int8_rowwise_3d_kernel<<<blocks, threads, 0, stream>>>(
            input.data_ptr<int8_t>(),
            scale_in.data_ptr<float>(),   // contiguous (B,M) is fine as flat
            output.data_ptr<int8_t>(),
            scale_out.data_ptr<float>(), 
            B, M, N
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


/*
This kernel compute 
- Input: fc1 (float, M,N), fc2 (float, M,N)
- Output: out (float, M,N)
- Operation: out[i,j] = SiLU(fc1[i,j]) * fc2[i,j]
*/
__global__ void silu_mul_kernel(
    const float* __restrict__ fc1,  // [rows, cols]
    const float* __restrict__ fc2,  // [rows, cols]
    float* __restrict__ out,        // [rows, cols]
    int rows,
    int cols
) {
    int row = blockIdx.x;      // one block per row
    if (row >= rows) return;

    int tid = threadIdx.x;

    // base pointer for this row
    int row_offset = row * cols;

    // each thread processes a strided subset of the columns
    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row_offset + col;

        float v = fc1[idx];
        float silu = v / (1.0f + expf(-v));
        out[idx] = silu * fc2[idx];
    }
}

torch::Tensor silu_mul_cuda(torch::Tensor fc1, torch::Tensor fc2) {
    TORCH_CHECK(fc1.is_cuda() && fc2.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(fc1.dtype() == torch::kFloat32 && fc2.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(fc1.dim() == 2 && fc2.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(fc1.sizes() == fc2.sizes(), "Input sizes must match");

    int rows = fc1.size(0);
    int cols = fc1.size(1);

    auto out = torch::empty_like(fc1);

    int threads = 256;
    int blocks = std::min(rows, (int)at::cuda::getDeviceProperties(fc1.get_device())->multiProcessorCount * 20);

    auto stream = at::cuda::getCurrentCUDAStream();
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(
        fc1.data_ptr<float>(),
        fc2.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ============================================================
// This kernel computes SiLU(x1) * x2 with per-row quantization for input/output.
//
// Input:
//   x1_int8  : INT8, [num_tokens, d_model]
//   scale_x1 : FP32, [num_tokens]   (per-row quant scale for x1)
//   x2_int8  : INT8, [num_tokens, d_model]
//   scale_x2 : FP32, [num_tokens]   (per-row quant scale for x2)
//   smooth_scale: FP32, [num_tokens] (per-row smooth for quantization, optional)
//
// Output:
//   y_int8   : INT8, [num_tokens, d_model]
//   out_scales  : FP32, [num_tokens]   (per-row quant scale)
// ============================================================

__global__ void silu_mul_int8_kernel(
    const int8_t* __restrict__ x1_int8,   // [rows, cols]
    const int8_t* __restrict__ x2_int8,   // [rows, cols]
    float* __restrict__ scale_x1,  // [rows]
    float* __restrict__ scale_x2,  // [rows]
    int8_t* __restrict__ out_int8,        // [rows, cols]
    float* __restrict__ out_scales,       // [rows]
    const float* __restrict__ smooth_scale, // [rows] 
    int rows,
    int cols
) {
    const int row = blockIdx.x;   // one block per row
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = row * cols;

    const float s_x1 = scale_x1[row];
    const float s_x2 = scale_x2[row];

    extern __shared__ float sdata[];

    // -------- Pass 1: compute max | y / s[j] | in this row --------
    float local_max = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row_offset + col;

        // dequantize inputs
        const float x1 = static_cast<float>(x1_int8[idx]) * s_x1;
        const float x2 = static_cast<float>(x2_int8[idx]) * s_x2;

        // exact float computation remains unchanged
        const float silu = x1 / (1.0f + expf(-x1));
        const float y = silu * x2;

        // apply SmoothQuant-style output smoothing only before quantization
        const float s = smooth_scale[col];
        const float y_smooth = y / s;

        const float a = fabsf(y_smooth);
        if (a > local_max) local_max = a;
    }

    sdata[tid] = local_max;
    __syncthreads();

    // block reduction for row max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    const float row_max = sdata[0];

    __shared__ float shared_scale_out;
    if (tid == 0) {
        float scale_out = (row_max > 0.0f) ? (row_max / 127.0f) : 1.0f;
        shared_scale_out = scale_out;
        out_scales[row] = scale_out;
    }
    __syncthreads();

    const float scale_out = shared_scale_out;
    const float inv_scale_out = 1.0f / scale_out;

    // -------- Pass 2: recompute y, smooth, quantize --------
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row_offset + col;

        const float x1 = static_cast<float>(x1_int8[idx]) * s_x1;
        const float x2 = static_cast<float>(x2_int8[idx]) * s_x2;

        const float silu = x1 / (1.0f + expf(-x1));
        const float y = silu * x2;

        const float s = smooth_scale[col];
        const float y_smooth = y / s;
        
        int q = __float2int_rn(y_smooth * inv_scale_out);

        q = max(-127, min(127, q));

        out_int8[idx] = static_cast<int8_t>(q);
    }
}

// ============================================================
// This kernel computes SiLU(x1) * x2 with per-row quantization for input/output.
// This kernel uses warp-level reduction for the max computation.
//
// Input:
//   x1_int8  : INT8, [num_tokens, d_model]
//   scale_x1 : FP32, [num_tokens]   (per-row quant scale for x1)
//   x2_int8  : INT8, [num_tokens, d_model]
//   scale_x2 : FP32, [num_tokens]   (per-row quant scale for x2)
//   smooth_scale: FP32, [num_tokens] (per-row smooth for quantization, optional)
//
// Output:
//   y_int8   : INT8, [num_tokens, d_model]
//   out_scales  : FP32, [num_tokens]   (per-row quant scale)
// ============================================================
__global__ void silu_mul_int8_kernel_warp_reduction(
    const int8_t* __restrict__ x1_int8,
    const int8_t* __restrict__ x2_int8,
    const float* __restrict__ scale_x1,
    const float* __restrict__ scale_x2,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scales,
    const float* __restrict__ smooth_scale,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = row * cols;

    const float s_x1 = scale_x1[row];
    const float s_x2 = scale_x2[row];

    // -------- Pass 1: compute local max --------
    float local_max = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row_offset + col;

        const float x1 = static_cast<float>(x1_int8[idx]) * s_x1;
        const float x2 = static_cast<float>(x2_int8[idx]) * s_x2;

        const float silu = x1 / (1.0f + expf(-x1));
        const float y = silu * x2;

        const float s = smooth_scale[col];
        const float y_smooth = y / s;

        local_max = fmaxf(local_max, fabsf(y_smooth));
    }

    // -------- Warp + block reduction --------
    const float row_max = block_reduce_max(local_max);

    __shared__ float shared_scale_out;

    if (tid == 0) {
        float scale_out = (row_max > 0.0f && isfinite(row_max))
                              ? row_max / 127.0f
                              : 1.0f;

        shared_scale_out = scale_out;
        out_scales[row] = scale_out;
    }
    __syncthreads();

    const float inv_scale_out = 1.0f / shared_scale_out;

    // -------- Pass 2: recompute and quantize --------
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row_offset + col;

        const float x1 = static_cast<float>(x1_int8[idx]) * s_x1;
        const float x2 = static_cast<float>(x2_int8[idx]) * s_x2;

        const float silu = x1 / (1.0f + expf(-x1));
        const float y = silu * x2;

        const float s = smooth_scale[col];
        const float y_smooth = y / s;

        float qf = y_smooth * inv_scale_out;
        if (!isfinite(qf)) qf = 0.0f;

        int q = __float2int_rn(qf);
        q = max(-127, min(127, q));

        out_int8[idx] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> silu_mul_int8_cuda(
    torch::Tensor x1_int8, 
    torch::Tensor scale_x1,
    torch::Tensor x2_int8, 
    torch::Tensor scale_x2,
    torch::Tensor smooth_scale, 
    bool use_warp_reduction
) {
    TORCH_CHECK(x1_int8.is_cuda() && x2_int8.is_cuda(), 
                "Input int8 tensors must be CUDA");
    TORCH_CHECK(scale_x1.is_cuda() && scale_x2.is_cuda(), 
                "Scale tensors must be CUDA");
    TORCH_CHECK(smooth_scale.is_cuda(), 
                "Smooth scale tensor must be CUDA");

    TORCH_CHECK(x1_int8.dtype() == torch::kChar && 
                x2_int8.dtype() == torch::kChar, 
                "Input tensors must be int8");

    TORCH_CHECK(scale_x1.dtype() == torch::kFloat32 && 
                scale_x2.dtype() == torch::kFloat32, 
                "Scale tensors must be float32");

    TORCH_CHECK(smooth_scale.dtype() == torch::kFloat32, 
                "Smooth scale tensor must be float32");

    TORCH_CHECK(x1_int8.sizes() == x2_int8.sizes(), 
                "Input tensor sizes must match");

    TORCH_CHECK(x1_int8.dim() == 2 || x1_int8.dim() == 3,
                "x1_int8 must be 2D [T, D] or 3D [B, T, D]");
    TORCH_CHECK(x2_int8.dim() == x1_int8.dim(),
                "x2_int8 must have same number of dims as x1_int8");

    TORCH_CHECK(smooth_scale.dim() == 1, 
                "smooth_scale must be 1D");

    const bool is_batched = x1_int8.dim() == 3;

    int64_t B = 1;
    int64_t T;
    int64_t D;
    int64_t rows;
    int64_t cols;

    std::vector<int64_t> out_shape;
    std::vector<int64_t> scale_out_shape;

    torch::Tensor x1_2d;
    torch::Tensor x2_2d;
    torch::Tensor scale_x1_1d;
    torch::Tensor scale_x2_1d;

    if (!is_batched) {
        // ------------------------------------------------------------
        // 2D case:
        // x1_int8:  [T, D]
        // scale_x1: [T]
        // ------------------------------------------------------------
        T = x1_int8.size(0);
        D = x1_int8.size(1);

        TORCH_CHECK(scale_x1.dim() == 1 && scale_x2.dim() == 1,
                    "For 2D input, scale_x1 and scale_x2 must be 1D [T]");

        TORCH_CHECK(scale_x1.size(0) == T &&
                    scale_x2.size(0) == T,
                    "For 2D input, scale size must match T");

        x1_2d = x1_int8.contiguous();
        x2_2d = x2_int8.contiguous();

        scale_x1_1d = scale_x1.contiguous();
        scale_x2_1d = scale_x2.contiguous();

        rows = T;
        cols = D;

        out_shape = {T, D};
        scale_out_shape = {T};
    } else {
        // ------------------------------------------------------------
        // 3D case:
        // x1_int8:  [B, T, D]
        // scale_x1: [B, T]
        // ------------------------------------------------------------
        B = x1_int8.size(0);
        T = x1_int8.size(1);
        D = x1_int8.size(2);

        TORCH_CHECK(scale_x1.dim() == 2 && scale_x2.dim() == 2,
                    "For 3D input, scale_x1 and scale_x2 must be 2D [B, T]");

        TORCH_CHECK(scale_x1.size(0) == B &&
                    scale_x1.size(1) == T,
                    "scale_x1 must have shape [B, T]");

        TORCH_CHECK(scale_x2.size(0) == B &&
                    scale_x2.size(1) == T,
                    "scale_x2 must have shape [B, T]");

        x1_2d = x1_int8.contiguous().reshape({B * T, D});
        x2_2d = x2_int8.contiguous().reshape({B * T, D});

        scale_x1_1d = scale_x1.contiguous().reshape({B * T});
        scale_x2_1d = scale_x2.contiguous().reshape({B * T});

        rows = B * T;
        cols = D;

        out_shape = {B, T, D};
        scale_out_shape = {B, T};
    }

    TORCH_CHECK(smooth_scale.size(0) == D,
                "smooth_scale size must match last dimension D");

    auto smooth_scale_contig = smooth_scale.contiguous();

    auto out_int8_2d = torch::empty(
        {rows, cols},
        x1_int8.options()
    );

    auto out_scales_1d = torch::empty(
        {rows},
        torch::dtype(torch::kFloat32).device(x1_int8.device())
    );

    int threads = static_cast<int>(std::min<int64_t>(cols, 512));

    if (threads & (threads - 1)) {
        int p = 1;
        while ((p << 1) <= threads) p <<= 1;
        threads = p;
    }

    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(rows);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (use_warp_reduction == false) {
        size_t shared_mem_size = threads * sizeof(float);

        silu_mul_int8_kernel<<<grid, block, shared_mem_size, stream>>>(
            x1_2d.data_ptr<int8_t>(),
            x2_2d.data_ptr<int8_t>(),
            scale_x1_1d.data_ptr<float>(),
            scale_x2_1d.data_ptr<float>(),
            out_int8_2d.data_ptr<int8_t>(),
            out_scales_1d.data_ptr<float>(),
            smooth_scale_contig.data_ptr<float>(),
            static_cast<int>(rows),
            static_cast<int>(cols)
        );
    } else {
        silu_mul_int8_kernel_warp_reduction<<<grid, block, 0, stream>>>(
            x1_2d.data_ptr<int8_t>(),
            x2_2d.data_ptr<int8_t>(),
            scale_x1_1d.data_ptr<float>(),
            scale_x2_1d.data_ptr<float>(),
            out_int8_2d.data_ptr<int8_t>(),
            out_scales_1d.data_ptr<float>(),
            smooth_scale_contig.data_ptr<float>(),
            static_cast<int>(rows),
            static_cast<int>(cols)
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto out_int8 = out_int8_2d.reshape(out_shape);
    auto out_scales = out_scales_1d.reshape(scale_out_shape);

    return std::make_tuple(out_int8, out_scales);
}