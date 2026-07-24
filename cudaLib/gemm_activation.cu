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
#include "gemm_utils.cu"


#define BLOCK_SIZE 256

inline int heuristic_block_size(int cols) {
    return cols <= 4096 ? 256 : 512;
}

// ============================================================
// Naive shared-memory SiLU-multiply INT8 kernel, vectorized by 4
//
// Computes:
//   y = SiLU(x1) * x2 / smooth_scale
//
// Vectorization:
//   x1_int8, x2_int8 : char4 loads
//   smooth_scale     : float4 loads
//   shared values    : float4 accesses
//   out_int8         : char4 stores
//
// Shared-memory requirement:
//   cols * sizeof(float)
//
// Requirements:
//   - cols must be divisible by 4
//   - input/output pointers must be suitably aligned
// ============================================================

__device__ __forceinline__ int8_t quantize_int8_value(float value) {
    if (!isfinite(value)) {
        value = 0.0f;
    }

    int q = __float2int_rn(value);
    q = max(-127, min(127, q));

    return static_cast<int8_t>(q);
}

__device__ __forceinline__ float silu_float(float x) {
    return x / (1.0f + expf(-x));
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
// This kernel uses hierarchical reduction for the max computation.
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
__global__ void silu_mul_int8_kernel_hierarchical_reduction(
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

    int threads = 256;
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
        silu_mul_int8_kernel_hierarchical_reduction<<<grid, block, 0, stream>>>(
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


// ================================================================
// Kernel: BF16 x1, x2 -> INT8 y + FP32 row scale
//
// x1, x2 : BF16 [M, K]
// y_i8   : INT8 [M, K]
// y_scale: FP32 [M]
//
// Computation
// Y = SiLU(x1) * x2
// Y = Y / smooth_scale 
// Y_i8, scale = quantize(Y) 
// ================================================================
__global__ void silu_mul_quant_kernel(
const __nv_bfloat16* __restrict__ x1,
    const __nv_bfloat16* __restrict__ x2,
    const float* __restrict__ smooth_scale,
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

    const __nv_bfloat16* x1_row = x1 + static_cast<size_t>(row) * K;
    const __nv_bfloat16* x2_row = x2 + static_cast<size_t>(row) * K;
    int8_t* y_row = y_i8 + static_cast<size_t>(row) * K;

    __shared__ float sdata[BLOCK_SIZE];

    // ------------------------------------------------------------
    // Pass 1: compute absmax after SmoothQuant scaling
    // ------------------------------------------------------------
    float local_absmax = 0.0f;

    for (int col = tid; col < K; col += BLOCK_SIZE) {
        float a = __bfloat162float(x1_row[col]);
        float b = __bfloat162float(x2_row[col]);

        float sigmoid = 1.0f / (1.0f + expf(-a));
        float v = a * sigmoid * b;

        // SmoothQuant correction:
        float s = smooth_scale[col];
        v = v / s;

        local_absmax = fmaxf(local_absmax, fabsf(v));
    }

    sdata[tid] = local_absmax;
    __syncthreads();

    // Block reduction: max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float absmax = sdata[0];
    float scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;

    if (tid == 0) {
        y_scale[row] = scale;
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Pass 2: quantize after SmoothQuant scaling
    // ------------------------------------------------------------
    for (int col = tid; col < K; col += BLOCK_SIZE) {
        float a = __bfloat162float(x1_row[col]);
        float b = __bfloat162float(x2_row[col]);

        float sigmoid = 1.0f / (1.0f + expf(-a));
        float v = a * sigmoid * b;

        float s = smooth_scale[col];
        v = v / s;

        int q = static_cast<int>(nearbyintf(v / scale));
        q = max(-128, min(127, q));

        y_row[col] = static_cast<int8_t>(q);
    }
}


std::tuple<torch::Tensor, torch::Tensor> silu_mul_quant_cuda(torch::Tensor x1,
                                                            torch::Tensor x2,
                                                            torch::Tensor smooth_scale) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda() && smooth_scale.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(x1.dtype() == torch::kBFloat16 && x2.dtype() == torch::kBFloat16, "Inputs must be BFloat16");
    TORCH_CHECK(smooth_scale.dtype() == torch::kFloat32, "smooth_scale must be float32");

    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");

    const auto M = static_cast<int>(x1.size(0));
    const auto K = static_cast<int>(x1.size(1));

    TORCH_CHECK(smooth_scale.dim() == 1 && smooth_scale.size(0) == K,
                "smooth_scale must be 1D and match the last dimension of x1/x2");

    auto y_i8 = torch::empty_like(x1, torch::TensorOptions().dtype(torch::kInt8));
    auto y_scale = torch::empty({M}, x1.options().dtype(torch::kFloat32));

    constexpr int threads = BLOCK_SIZE;
    dim3 grid(M);
    dim3 block(threads);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    silu_mul_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
        smooth_scale.data_ptr<float>(),
        y_i8.data_ptr<int8_t>(),
        y_scale.data_ptr<float>(),
        M,
        K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y_i8, y_scale);
}


// ============================================================
// BF16 SiLU-multiply kernel with per-channel smooth scaling
//
// Computes:
//   out[row, col] = SiLU(x1[row, col]) * x2[row, col] / smooth_scale[col]
//
// Input:
//   x1_bf16     : BF16 [rows, cols]
//   x2_bf16     : BF16 [rows, cols]
//   smooth_scale: FP32 [cols]
//
// Output:
//   out_bf16    : BF16 [rows, cols]
//
// Mapping:
//   - One CUDA block processes one logical row.
//   - Computation is performed in FP32.
// ============================================================

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ x1_bf16,
    const __nv_bfloat16* __restrict__ x2_bf16,
    const float* __restrict__ smooth_scale,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int row_offset = row * cols;  // pointer to X[row, cols]

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {

        const int index = row_offset + col;

        const float x1 = __bfloat162float(x1_bf16[index]);
        const float x2 = __bfloat162float(x2_bf16[index]);

        const float silu = x1 / (1.0f + expf(-x1));
        const float y = silu * x2;

        // Per-channel smoothing
        const float output = y / smooth_scale[col];

        out_bf16[index] = __float2bfloat16_rn(output);
    }
}

// ============================================================
// Vectorized BF16 SiLU-multiply kernel
//
// Computes:
//   out[row, col]
//       = SiLU(x1[row, col]) * x2[row, col]
//         / smooth_scale[col]
//
// Vectorization:
//   - Each iteration processes four BF16 elements.
//   - x1 and x2 use two __nv_bfloat162 loads.
//   - smooth_scale uses one float4 load.
//   - output uses two __nv_bfloat162 stores.
//
// Fast-path requirement:
//   cols % 4 == 0
// ============================================================

__global__ void silu_mul_bf16_vec4_kernel(
    const __nv_bfloat16* __restrict__ x1_bf16,
    const __nv_bfloat16* __restrict__ x2_bf16,
    const float* __restrict__ smooth_scale,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols
) {
    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    // Four scalar elements are processed per vector iteration.
    const int vec_cols = cols / 4;

    // Each __nv_bfloat162 contains two BF16 values.
    const int bf16x2_row_offset = row * (cols / 2);

    const __nv_bfloat162* __restrict__ x1_vec =
        reinterpret_cast<const __nv_bfloat162*>(x1_bf16);

    const __nv_bfloat162* __restrict__ x2_vec =
        reinterpret_cast<const __nv_bfloat162*>(x2_bf16);

    __nv_bfloat162* __restrict__ out_vec =
        reinterpret_cast<__nv_bfloat162*>(out_bf16);

    const float4* __restrict__ smooth_vec =
        reinterpret_cast<const float4*>(smooth_scale);

    for (int vec_col = threadIdx.x;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        // One vec_col corresponds to four BF16 elements,
        // or two __nv_bfloat162 values.
        const int pair_index =
            bf16x2_row_offset + vec_col * 2;

        // ----------------------------------------------------
        // Load four BF16 values from each input.
        // ----------------------------------------------------

        const __nv_bfloat162 x1_pair_0 =
            x1_vec[pair_index];

        const __nv_bfloat162 x1_pair_1 =
            x1_vec[pair_index + 1];

        const __nv_bfloat162 x2_pair_0 =
            x2_vec[pair_index];

        const __nv_bfloat162 x2_pair_1 =
            x2_vec[pair_index + 1];

        // Convert BF16 pairs to float2.
        const float2 x1_01 =
            __bfloat1622float2(x1_pair_0);

        const float2 x1_23 =
            __bfloat1622float2(x1_pair_1);

        const float2 x2_01 =
            __bfloat1622float2(x2_pair_0);

        const float2 x2_23 =
            __bfloat1622float2(x2_pair_1);

        // Load four FP32 smoothing values.
        const float4 smooth =
            smooth_vec[vec_col];

        // ----------------------------------------------------
        // Compute four SiLU-multiply outputs in FP32.
        // ----------------------------------------------------

        const float silu_0 = x1_01.x / (1.0f + expf(-x1_01.x));

        const float silu_1 = x1_01.y / (1.0f + expf(-x1_01.y));

        const float silu_2 = x1_23.x / (1.0f + expf(-x1_23.x));

        const float silu_3 = x1_23.y / (1.0f + expf(-x1_23.y));

        const float out_0 =
            silu_0 * x2_01.x / smooth.x;

        const float out_1 =
            silu_1 * x2_01.y / smooth.y;

        const float out_2 =
            silu_2 * x2_23.x / smooth.z;

        const float out_3 =
            silu_3 * x2_23.y / smooth.w;

        // ----------------------------------------------------
        // Convert four FP32 outputs to BF16 and store them as
        // two __nv_bfloat162 vectors.
        // ----------------------------------------------------

        out_vec[pair_index] =
            __floats2bfloat162_rn(out_0, out_1);

        out_vec[pair_index + 1] =
            __floats2bfloat162_rn(out_2, out_3);
    }
}

torch::Tensor silu_mul_bf16_cuda(
    torch::Tensor x1_bf16,
    torch::Tensor x2_bf16,
    torch::Tensor smooth_scale
) {
    TORCH_CHECK(
        x1_bf16.is_cuda() &&
        x2_bf16.is_cuda() &&
        smooth_scale.is_cuda(),
        "x1_bf16, x2_bf16, and smooth_scale must be CUDA tensors"
    );

    TORCH_CHECK(
        x1_bf16.device() == x2_bf16.device() &&
        x1_bf16.device() == smooth_scale.device(),
        "All input tensors must be on the same CUDA device"
    );

    // --------------------------------------------------------
    // Data-type checks
    // --------------------------------------------------------

    TORCH_CHECK(
        x1_bf16.scalar_type() == torch::kBFloat16 &&
        x2_bf16.scalar_type() == torch::kBFloat16,
        "x1_bf16 and x2_bf16 must have dtype torch.bfloat16"
    );

    TORCH_CHECK(
        smooth_scale.scalar_type() == torch::kFloat32,
        "smooth_scale must have dtype torch.float32"
    );

    // --------------------------------------------------------
    // Shape checks
    // --------------------------------------------------------

    TORCH_CHECK(
        x1_bf16.dim() == 2 || x1_bf16.dim() == 3,
        "Inputs must have shape [T, D] or [B, T, D]"
    );

    TORCH_CHECK(
        x1_bf16.sizes() == x2_bf16.sizes(),
        "x1_bf16 and x2_bf16 must have identical shapes"
    );

    TORCH_CHECK(
        x1_bf16.numel() > 0,
        "Input tensors must not be empty"
    );

    const int cols = static_cast<int>(x1_bf16.size(-1));
    const int rows = static_cast<int>(x1_bf16.numel() / cols);

    TORCH_CHECK(
        smooth_scale.dim() == 1, "smooth_scale must be one-dimensional with shape [D]"
    );

    TORCH_CHECK(
        smooth_scale.numel() == cols, "smooth_scale must have shape [D], where D = ",
        cols, ", but received ", smooth_scale.numel(), " elements"
    );

    // --------------------------------------------------------
    // Contiguous memory layout
    // --------------------------------------------------------
    x1_bf16 = x1_bf16.contiguous();
    x2_bf16 = x2_bf16.contiguous();
    smooth_scale = smooth_scale.contiguous();

    auto out_bf16 = torch::empty_like(x1_bf16);

    const int threads = BLOCK_SIZE;
    const dim3 block(threads);
    const dim3 grid(rows);

    const c10::cuda::CUDAGuard device_guard(x1_bf16.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x1_bf16.get_device()).stream();

    silu_mul_bf16_vec4_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x1_bf16.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<const __nv_bfloat16*>(
            x2_bf16.data_ptr<at::BFloat16>()
        ),
        smooth_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            out_bf16.data_ptr<at::BFloat16>()
        ),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_bf16;
}



// ============================================================
// INT8 SiLU with char4 vectorized loads and stores.
//
// Computes:
//   x[j]     = x_int8[j] * input_scale[row]
//   y[j]     = SiLU(x[j])
//   out_scale[row] = max_j(abs(y[j])) / 127
//   out_int8[j]    = round(y[j] / out_scale[row])
//
// Layout:
//   x_int8:    [rows, cols], contiguous row-major
//   out_int8:  [rows, cols], contiguous row-major
//   in_scales: [rows]
//   out_scales:[rows]
//
// Mapping:
//   - One CUDA block processes one row.
//   - One thread processes multiple char4 vectors.
//
// Requirements:
//   - cols % 4 == 0
//   - blockDim.x is a multiple of 32
//   - input and output pointers are at least 4-byte aligned
// ============================================================
__global__ void hierarchical_silu_int8_vec4_kernel(
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ input_scales,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scales,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;

    // Each char4 represents four consecutive INT8 elements.
    const int vec_cols = cols >> 2;
    const int vec_row_offset = row * vec_cols;

    const float input_scale = input_scales[row];

    const char4* __restrict__ x_vec =
        reinterpret_cast<const char4*>(x_int8);

    char4* __restrict__ out_vec =
        reinterpret_cast<char4*>(out_int8);

    // --------------------------------------------------------
    // Pass 1:
    // Dequantize, compute SiLU, and find the maximum absolute
    // output value for this row.
    // --------------------------------------------------------
    float local_max = 0.0f;

    for (int vec_col = tid;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;
        const char4 qx = x_vec[vec_index];

        const float x0 = static_cast<float>(qx.x) * input_scale;
        const float x1 = static_cast<float>(qx.y) * input_scale;
        const float x2 = static_cast<float>(qx.z) * input_scale;
        const float x3 = static_cast<float>(qx.w) * input_scale;

        const float y0 = silu_float(x0);
        const float y1 = silu_float(x1);
        const float y2 = silu_float(x2);
        const float y3 = silu_float(x3);

        local_max = fmaxf(local_max, fabsf(y0));
        local_max = fmaxf(local_max, fabsf(y1));
        local_max = fmaxf(local_max, fabsf(y2));
        local_max = fmaxf(local_max, fabsf(y3));
    }

    const float row_max = block_reduce_max(local_max);

    // Every thread needs the reciprocal output scale during the second pass.
    __shared__ float shared_inv_output_scale;

    if (tid == 0) {
        const float output_scale = row_max > 0.0f ? row_max * (1.0f / 127.0f) : 1.0f;
        out_scales[row] = output_scale;
        shared_inv_output_scale = 1.0f / output_scale;
    }

    __syncthreads();

    const float inv_output_scale =
        shared_inv_output_scale;

    // --------------------------------------------------------
    // Pass 2:
    // Recompute SiLU and quantize four elements at a time.
    //
    // Recomputing avoids storing FP32 intermediate outputs in
    // shared or global memory.
    // --------------------------------------------------------
    for (int vec_col = tid;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;
        const char4 qx = x_vec[vec_index];

        const float x0 = static_cast<float>(qx.x) * input_scale;
        const float x1 = static_cast<float>(qx.y) * input_scale;
        const float x2 = static_cast<float>(qx.z) * input_scale;
        const float x3 =static_cast<float>(qx.w) * input_scale;

        const float y0 = silu_float(x0) * inv_output_scale;
        const float y1 = silu_float(x1) * inv_output_scale;
        const float y2 = silu_float(x2) * inv_output_scale;
        const float y3 = silu_float(x3) * inv_output_scale;

        const char4 qout = make_char4(
            static_cast<char>(quantize_int8(y0)),
            static_cast<char>(quantize_int8(y1)),
            static_cast<char>(quantize_int8(y2)),
            static_cast<char>(quantize_int8(y3))
        );

        out_vec[vec_index] = qout;
    }
}

std::tuple<torch::Tensor, torch::Tensor> silu_int8_cuda(
    torch::Tensor x_int8,
    torch::Tensor scale_x
) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");

    TORCH_CHECK(x_int8.scalar_type() == torch::kInt8, "x_int8 must be torch.int8");

    TORCH_CHECK(scale_x.scalar_type() == torch::kFloat32, "scale_x must be torch.float32");

    TORCH_CHECK(scale_x.device() == x_int8.device(), "x_int8 and scale must on the same device");


    // [T, D], [B, T, D], and higher-dimensional inputs are treated
    // as a flattened collection of rows with the last dimension D.
    x_int8 = x_int8.contiguous();
    scale_x = scale_x.contiguous();

    const int64_t cols64 = x_int8.size(-1);
    const int64_t rows64 = x_int8.numel() / cols64;

    TORCH_CHECK(
        cols64 % 4 == 0, "The last dimension must be divisible by 4 for char4 vectorization"
    );

    TORCH_CHECK(
        scale_x.numel() == rows64,
        "scale_x must contain one scale per flattened input row. ",
        "Expected ",
        rows64,
        " scales, but received ",
        scale_x.numel()
    );

    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max(),
        "The number of rows exceeds the supported int range"
    );

    TORCH_CHECK(
        cols64 <= std::numeric_limits<int>::max(),
        "The number of columns exceeds the supported int range"
    );

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);

    // Preserve the original input shape for the INT8 output.
    auto out_int8 = torch::empty_like(x_int8);
    auto out_scales = torch::empty_like(scale_x);

    const int threads = BLOCK_SIZE;
    const dim3 block(threads);
    const dim3 grid(rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_int8.get_device()).stream();
    const c10::cuda::CUDAGuard device_guard(x_int8.device());

    hierarchical_silu_int8_vec4_kernel<<<grid, block, 0, stream>>>(
            x_int8.data_ptr<int8_t>(),
            scale_x.data_ptr<float>(),
            out_int8.data_ptr<int8_t>(),
            out_scales.data_ptr<float>(),
            rows,
            cols
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        out_int8,
        out_scales
    };
}


// ============================================================
// BF16 SiLU with __nv_bfloat162 vectorized loads/stores.
//
// Computes:
//   y[j] = SiLU(x[j])
// ============================================================
__global__ void silu_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int cols
) {
    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;

    // Each __nv_bfloat162 contains two BF16 elements.
    const int vec_cols = cols >> 1;
    const int vec_row_offset = row * vec_cols;

    const __nv_bfloat162* __restrict__ x_vec =
        reinterpret_cast<const __nv_bfloat162*>(x_bf16);

    __nv_bfloat162* __restrict__ out_vec =
        reinterpret_cast<__nv_bfloat162*>(out_bf16);

    for (int vec_col = tid;
         vec_col < vec_cols;
         vec_col += blockDim.x) {

        const int vec_index = vec_row_offset + vec_col;

        // Vectorized 32-bit load containing two BF16 elements.
        const __nv_bfloat162 x_pair = x_vec[vec_index];

        // Convert both BF16 values to FP32.
        const float2 x_float = __bfloat1622float2(x_pair);

        const float y0 = silu_float(x_float.x);
        const float y1 = silu_float(x_float.y);

        // Convert the two FP32 results back to BF16.
        const __nv_bfloat162 y_pair =
            __floats2bfloat162_rn(y0, y1);

        // Vectorized 32-bit store.
        out_vec[vec_index] = y_pair;
    }
}

torch::Tensor silu_bf16_cuda(
    torch::Tensor x_bf16
) {
    TORCH_CHECK(
        x_bf16.is_cuda(),
        "x_bf16 must be a CUDA tensor"
    );

    TORCH_CHECK(
        x_bf16.scalar_type() == torch::kBFloat16,
        "x_bf16 must have dtype torch.bfloat16"
    );

    TORCH_CHECK(
        x_bf16.dim() >= 1,
        "x_bf16 must have at least one dimension"
    );

    TORCH_CHECK(
        x_bf16.numel() > 0,
        "x_bf16 must not be empty"
    );

    const c10::cuda::CUDAGuard device_guard(
        x_bf16.device()
    );

    // [D], [T, D], [B, T, D], and higher-dimensional tensors
    // are interpreted as a flattened collection of rows whose
    // last dimension is D.
    x_bf16 = x_bf16.contiguous();

    const int64_t cols64 = x_bf16.size(-1);
    const int64_t rows64 = x_bf16.numel() / cols64;

    TORCH_CHECK(
        cols64 > 0,
        "The last dimension of x_bf16 must be greater than zero"
    );

    TORCH_CHECK(
        cols64 % 2 == 0,
        "The last dimension must be divisible by 2 for "
        "__nv_bfloat162 vectorization"
    );

    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max(),
        "The number of rows exceeds the supported int range"
    );

    TORCH_CHECK(
        cols64 <= std::numeric_limits<int>::max(),
        "The number of columns exceeds the supported int range"
    );

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);

    // Preserve the input shape and BF16 dtype.
    auto out_bf16 = torch::empty_like(x_bf16);

    const int threads = BLOCK_SIZE;

    TORCH_CHECK(
        threads > 0 && threads <= 1024,
        "BLOCK_SIZE must be between 1 and 1024"
    );

    const dim3 block(threads);
    const dim3 grid(rows);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            x_bf16.get_device()
        ).stream();

    silu_bf16_vec2_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x_bf16.data_ptr<at::BFloat16>()
        ),
        reinterpret_cast<__nv_bfloat16*>(
            out_bf16.data_ptr<at::BFloat16>()
        ),
        rows,
        cols
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_bf16;
}