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

#include "gemm_utils.cu"


// ============================================================
// INT8 RoPE with char4 vectorized loads and stores.
//
// Logical input:
//   x         [outer_size, seq_len, head_dim]
//   scale_x   [outer_size, seq_len]
//   cos       [seq_len, head_dim]
//   sin       [seq_len, head_dim]
//   out       [outer_size, seq_len, head_dim]
//   scale_out [outer_size, seq_len]
// ============================================================
__global__ void rope_int8_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ scale_x,
    const int8_t* __restrict__ cos,
    float scale_cos,
    const int8_t* __restrict__ sin,
    float scale_sin,
    int outer_size,
    int head_dim,
    int seq_len,
    int8_t* __restrict__ out,
    float* __restrict__ scale_out
) {
    const int outer = static_cast<int>(blockIdx.z);
    const int pos   = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);

    if (outer >= outer_size || pos >= seq_len) {
        return;
    }

    const int half = head_dim >> 1;

    // Four elements are processed from each half per iteration.
    const int half_vec_cols = half >> 2;

    // Flatten [outer_size, seq_len] to one row index.
    const int row_idx = outer * seq_len + pos;

    // Scalar INT8 offset of this [head_dim] row.
    const int row_base = row_idx * head_dim;
    const int first_half_base = row_base;
    const int second_half_base = row_base + half;

    // cos/sin are shared across outer dimensions.
    const int table_base = pos * head_dim;

    const float sx = scale_x[row_idx];  

    // Interpret the scalar buffers as groups of four INT8 values.
    const char4* __restrict__ x_vec = reinterpret_cast<const char4*>(x);
    const char4* __restrict__ cos_vec = reinterpret_cast<const char4*>(cos);
    const char4* __restrict__ sin_vec = reinterpret_cast<const char4*>(sin);
    char4* __restrict__ out_vec = reinterpret_cast<char4*>(out);

    // Convert scalar offsets into char4 offsets.
    const int first_vec_base = first_half_base >> 2;
    const int second_vec_base = second_half_base >> 2;
    const int table_vec_base = table_base >> 2;

    // --------------------------------------------------------
    // Pass 1: compute the maximum absolute RoPE output.
    // --------------------------------------------------------
    float local_max = 0.0f;

    for (int vec_col = tid; vec_col < half_vec_cols; vec_col += blockDim.x) {

        // Load four values from each half.
        const char4 qx1 = x_vec[first_vec_base + vec_col];
        const char4 qx2 = x_vec[second_vec_base + vec_col];

        // Because cos/sin are duplicated across the two halves,
        // only the first-half table values are required.
        const char4 qc = cos_vec[table_vec_base + vec_col];

        const char4 qs = sin_vec[table_vec_base + vec_col];

        // Dequantize four first-half values.
        const float x1_0 = static_cast<float>(qx1.x) * sx;
        const float x1_1 = static_cast<float>(qx1.y) * sx;
        const float x1_2 = static_cast<float>(qx1.z) * sx;
        const float x1_3 = static_cast<float>(qx1.w) * sx;

        // Dequantize four second-half values.
        const float x2_0 = static_cast<float>(qx2.x) * sx;
        const float x2_1 = static_cast<float>(qx2.y) * sx;
        const float x2_2 = static_cast<float>(qx2.z) * sx;
        const float x2_3 = static_cast<float>(qx2.w) * sx;

        // Dequantize cosine values.
        const float c0 = static_cast<float>(qc.x) * scale_cos;
        const float c1 = static_cast<float>(qc.y) * scale_cos;
        const float c2 = static_cast<float>(qc.z) * scale_cos;
        const float c3 = static_cast<float>(qc.w) * scale_cos;

        // Dequantize sine values.
        const float s0 = static_cast<float>(qs.x) * scale_sin;
        const float s1 = static_cast<float>(qs.y) * scale_sin;
        const float s2 = static_cast<float>(qs.z) * scale_sin;
        const float s3 = static_cast<float>(qs.w) * scale_sin;

        // Four first-half outputs.
        const float y1_0 = x1_0 * c0 - x2_0 * s0;
        const float y1_1 = x1_1 * c1 - x2_1 * s1;
        const float y1_2 = x1_2 * c2 - x2_2 * s2;
        const float y1_3 = x1_3 * c3 - x2_3 * s3;

        // Four second-half outputs.
        const float y2_0 = x1_0 * s0 + x2_0 * c0;
        const float y2_1 = x1_1 * s1 + x2_1 * c1;
        const float y2_2 = x1_2 * s2 + x2_2 * c2;
        const float y2_3 = x1_3 * s3 + x2_3 * c3;

        local_max = fmaxf(local_max, fabsf(y1_0));
        local_max = fmaxf(local_max, fabsf(y1_1));
        local_max = fmaxf(local_max, fabsf(y1_2));
        local_max = fmaxf(local_max, fabsf(y1_3));

        local_max = fmaxf(local_max, fabsf(y2_0));
        local_max = fmaxf(local_max, fabsf(y2_1));
        local_max = fmaxf(local_max, fabsf(y2_2));
        local_max = fmaxf(local_max, fabsf(y2_3));
    }

    const float max_abs = block_reduce_max(local_max);

    const float output_scale = max_abs > 0.0f ? max_abs * (1.0f / 127.0f) : 1.0f;
    const float inv_output_scale = 1.0f / output_scale;
    if (tid == 0) {
        scale_out[row_idx] = output_scale;
    }

    // --------------------------------------------------------
    // Pass 2: recompute, quantize, and issue char4 stores.
    // --------------------------------------------------------
    for (int vec_col = tid; vec_col < half_vec_cols; vec_col += blockDim.x) {

        const char4 qx1 = x_vec[first_vec_base + vec_col];
        const char4 qx2 = x_vec[second_vec_base + vec_col];
        const char4 qc = cos_vec[table_vec_base + vec_col];
        const char4 qs = sin_vec[table_vec_base + vec_col];

        const float x1_0 = static_cast<float>(qx1.x) * sx;
        const float x1_1 = static_cast<float>(qx1.y) * sx;
        const float x1_2 = static_cast<float>(qx1.z) * sx;
        const float x1_3 = static_cast<float>(qx1.w) * sx;

        const float x2_0 = static_cast<float>(qx2.x) * sx;
        const float x2_1 = static_cast<float>(qx2.y) * sx;
        const float x2_2 = static_cast<float>(qx2.z) * sx;
        const float x2_3 = static_cast<float>(qx2.w) * sx;

        const float c0 = static_cast<float>(qc.x) * scale_cos;
        const float c1 = static_cast<float>(qc.y) * scale_cos;
        const float c2 = static_cast<float>(qc.z) * scale_cos;
        const float c3 = static_cast<float>(qc.w) * scale_cos;

        const float s0 = static_cast<float>(qs.x) * scale_sin;
        const float s1 = static_cast<float>(qs.y) * scale_sin;
        const float s2 = static_cast<float>(qs.z) * scale_sin;
        const float s3 = static_cast<float>(qs.w) * scale_sin;

        // Scale the FP32 outputs directly into INT8 units.
        const float y1_0 = (x1_0 * c0 - x2_0 * s0) * inv_output_scale;
        const float y1_1 = (x1_1 * c1 - x2_1 * s1) * inv_output_scale;
        const float y1_2 = (x1_2 * c2 - x2_2 * s2) * inv_output_scale;
        const float y1_3 = (x1_3 * c3 - x2_3 * s3) * inv_output_scale;

        const float y2_0 = (x1_0 * s0 + x2_0 * c0) * inv_output_scale;
        const float y2_1 = (x1_1 * s1 + x2_1 * c1) * inv_output_scale;
        const float y2_2 = (x1_2 * s2 + x2_2 * c2) * inv_output_scale;
        const float y2_3 = (x1_3 * s3 + x2_3 * c3) * inv_output_scale;

        const char4 qout_first = make_char4(
            static_cast<char>(quantize_int8(y1_0)),
            static_cast<char>(quantize_int8(y1_1)),
            static_cast<char>(quantize_int8(y1_2)),
            static_cast<char>(quantize_int8(y1_3))
        );

        const char4 qout_second = make_char4(
            static_cast<char>(quantize_int8(y2_0)),
            static_cast<char>(quantize_int8(y2_1)),
            static_cast<char>(quantize_int8(y2_2)),
            static_cast<char>(quantize_int8(y2_3))
        );

        // Two vectorized stores:
        //   - four outputs in the first half
        //   - four outputs in the second half
        out_vec[first_vec_base + vec_col] = qout_first;
        out_vec[second_vec_base + vec_col] = qout_second;
    }
}


std::tuple<torch::Tensor, torch::Tensor> rope_int8_host(
    torch::Tensor x,          // int8 [H,T,D] or [B,H,T,D]
    torch::Tensor scale_x,    // float32 [H,T], [B,H,T], [B,T], or [T]
    torch::Tensor cos,        // int8 [T,D]
    float scale_cos,
    torch::Tensor sin,        // int8 [T,D]
    float scale_sin
) {
    // ========================================================
    // Device and dtype validation
    // ========================================================
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");
    TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
    TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");

    TORCH_CHECK(
        x.scalar_type() == torch::kInt8,
        "x must have dtype torch.int8"
    );

    TORCH_CHECK(
        scale_x.scalar_type() == torch::kFloat32,
        "scale_x must have dtype torch.float32"
    );

    TORCH_CHECK(
        cos.scalar_type() == torch::kInt8,
        "cos must have dtype torch.int8"
    );

    TORCH_CHECK(
        sin.scalar_type() == torch::kInt8,
        "sin must have dtype torch.int8"
    );

    TORCH_CHECK(
        x.device() == scale_x.device() &&
        x.device() == cos.device() &&
        x.device() == sin.device(),
        "x, scale_x, cos, and sin must be on the same CUDA device"
    );

    TORCH_CHECK(
        x.dim() == 3 || x.dim() == 4,
        "x must have shape [H,T,D] or [B,H,T,D]"
    );

    TORCH_CHECK(
        cos.dim() == 2,
        "cos must have shape [T,D]"
    );

    TORCH_CHECK(
        sin.dim() == 2,
        "sin must have shape [T,D]"
    );

    const c10::cuda::CUDAGuard device_guard(x.device());

    // ========================================================
    // Infer dimensions from the final two axes.
    //
    // [H,T,D]   -> outer_size = H
    // [B,H,T,D] -> outer_size = B * H
    // ========================================================
    const int64_t seq_len64 =
        x.size(-2);

    const int64_t head_dim64 =
        x.size(-1);

    TORCH_CHECK(
        seq_len64 > 0,
        "seq_len must be greater than zero"
    );

    TORCH_CHECK(
        head_dim64 > 0,
        "head_dim must be greater than zero"
    );

    TORCH_CHECK(
        head_dim64 % 2 == 0,
        "head_dim must be even"
    );

    const int64_t vectors_per_sequence =
        seq_len64 * head_dim64;

    const int64_t outer_size64 =
        x.numel() / vectors_per_sequence;

    TORCH_CHECK(
        x.numel() ==
            outer_size64 * seq_len64 * head_dim64,
        "x cannot be interpreted as [outer_size,T,D]"
    );

    TORCH_CHECK(
        cos.size(0) == seq_len64 &&
        cos.size(1) == head_dim64,
        "cos must have shape [",
        seq_len64,
        ", ",
        head_dim64,
        "]"
    );

    TORCH_CHECK(
        sin.size(0) == seq_len64 &&
        sin.size(1) == head_dim64,
        "sin must have shape [",
        seq_len64,
        ", ",
        head_dim64,
        "]"
    );

    TORCH_CHECK(
        outer_size64 <= std::numeric_limits<int>::max(),
        "outer_size exceeds the supported int range"
    );

    TORCH_CHECK(
        seq_len64 <= std::numeric_limits<int>::max(),
        "seq_len exceeds the supported int range"
    );

    TORCH_CHECK(
        head_dim64 <= std::numeric_limits<int>::max(),
        "head_dim exceeds the supported int range"
    );

    const int outer_size =
        static_cast<int>(outer_size64);

    const int seq_len =
        static_cast<int>(seq_len64);

    const int head_dim =
        static_cast<int>(head_dim64);

    // ========================================================
    // Make data contiguous and flatten x logically to [O,T,D].
    //
    // view() does not copy after contiguous().
    // ========================================================
    auto xq =
        x.contiguous().view({
            outer_size64,
            seq_len64,
            head_dim64
        });

    auto cosq =
        cos.contiguous();

    auto sinq =
        sin.contiguous();

    // ========================================================
    // Normalize scale_x to [outer_size,T].
    //
    // Supported layouts:
    //
    // x [H,T,D]:
    //   scale_x [H,T]
    //   scale_x [T]
    //
    // x [B,H,T,D]:
    //   scale_x [B,H,T]
    //   scale_x [B,T]
    //   scale_x [T]
    //
    // After this section, the kernel always sees:
    //   sx [outer_size,T]
    // ========================================================
    torch::Tensor sx;

    if (scale_x.dim() == 1) {
        // Shared by all batches and heads: [T].
        TORCH_CHECK(
            scale_x.size(0) == seq_len64,
            "1D scale_x must have shape [",
            seq_len64,
            "]"
        );

        sx = scale_x
            .view({1, seq_len64})
            .expand({outer_size64, seq_len64})
            .contiguous();

    } else if (
        scale_x.numel() ==
        outer_size64 * seq_len64
    ) {
        // Covers:
        //   [H,T]   for x [H,T,D]
        //   [B,H,T] for x [B,H,T,D]
        //
        // The physical flattened ordering matches x.
        TORCH_CHECK(
            scale_x.size(-1) == seq_len64,
            "The final dimension of scale_x must equal seq_len"
        );

        sx = scale_x
            .contiguous()
            .view({outer_size64, seq_len64});

    } else if (
        x.dim() == 4 &&
        scale_x.dim() == 2
    ) {
        // Batch-wise scales shared across heads:
        //   x       [B,H,T,D]
        //   scale_x [B,T]
        const int64_t batch_size64 =
            x.size(0);

        const int64_t num_heads64 =
            x.size(1);

        TORCH_CHECK(
            scale_x.size(0) == batch_size64 &&
            scale_x.size(1) == seq_len64,
            "For x [B,H,T,D], a batch-wise scale_x must have "
            "shape [B,T]"
        );

        sx = scale_x
            .view({
                batch_size64,
                1,
                seq_len64
            })
            .expand({
                batch_size64,
                num_heads64,
                seq_len64
            })
            .contiguous()
            .view({
                outer_size64,
                seq_len64
            });

    } else {
        TORCH_CHECK(
            false,
            "Unsupported scale_x shape. Expected [T], [H,T], "
            "[B,H,T], or [B,T], depending on the shape of x"
        );
    }

    // ========================================================
    // Outputs
    // ========================================================
    auto out_flat =
        torch::empty_like(xq);

    // The kernel always produces one scale for every
    // flattened [outer, token] row.
    auto scale_out_flat = torch::empty(
        {outer_size64, seq_len64},
        x.options().dtype(torch::kFloat32)
    );

    constexpr int threads = 256;

    const dim3 block(threads);
    const dim3 grid(
        1,
        static_cast<unsigned int>(seq_len),
        static_cast<unsigned int>(outer_size)
    );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            x.get_device()
        ).stream();

    rope_int8_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        xq.data_ptr<int8_t>(),
        sx.data_ptr<float>(),
        cosq.data_ptr<int8_t>(),
        scale_cos,
        sinq.data_ptr<int8_t>(),
        scale_sin,
        outer_size,
        head_dim,
        seq_len,
        out_flat.data_ptr<int8_t>(),
        scale_out_flat.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Restore the original x shape.
    auto out =
        out_flat.view(x.sizes());

    // Output scale shape is x.shape[:-1]:
    //
    // x [H,T,D]   -> scale_out [H,T]
    // x [B,H,T,D] -> scale_out [B,H,T]
    auto scale_shape =
        x.sizes().vec();

    scale_shape.pop_back();

    auto scale_out =
        scale_out_flat.view(scale_shape);

    return {
        out,
        scale_out
    };
}

// ============================================================
// BF16 RoPE using the rotate-half convention.
//
// Input:
//   x_bf16:  [num_heads, seq_len, head_dim]
//   cos:      [seq_len, head_dim], FP32
//   sin:      [seq_len, head_dim], FP32
//
// Python equivalent:
//
//   x1 = x[..., :head_dim / 2]
//   x2 = x[..., head_dim / 2:]
//
//   rotate_half(x) = cat([-x2, x1], dim=-1)
//
//   y = x * cos + rotate_half(x) * sin
//
// For pair index i:
//
//   first  = x[i]
//   second = x[i + head_dim / 2]
//
//   y[i]              = first * cos[i]
//                       - second * sin[i]
//
//   y[i + head_dim/2] = second * cos[i + head_dim/2]
//                       + first  * sin[i + head_dim/2]
//
// Since the Python code duplicates the angles:
//
//   cos[i] == cos[i + head_dim/2]
//   sin[i] == sin[i + head_dim/2]
//
// One block handles one [head_dim] vector.
// ============================================================
__global__ void rope_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ out_bf16,
    int rows,
    int seq_len,
    int head_dim
) {
    const int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int half_dim = head_dim >> 1;

    // Input layout:
    //   [num_heads, seq_len, head_dim]
    //
    // Flattened row:
    //   row = head_idx * seq_len + token_idx
    const int token_idx = row % seq_len;

    const int row_offset =
        row * head_dim;

    const int table_offset =
        token_idx * head_dim;

    for (int i = tid;
         i < half_dim;
         i += blockDim.x) {

        const int first_index =
            row_offset + i;

        const int second_index =
            row_offset + half_dim + i;

        const float first =
            __bfloat162float(x_bf16[first_index]);

        const float second =
            __bfloat162float(x_bf16[second_index]);

        const float cos_first =
            cos_table[table_offset + i];

        const float sin_first =
            sin_table[table_offset + i];

        const float cos_second =
            cos_table[table_offset + half_dim + i];

        const float sin_second =
            sin_table[table_offset + half_dim + i];

        const float out_first =
            first * cos_first -
            second * sin_first;

        const float out_second =
            second * cos_second +
            first * sin_second;

        out_bf16[first_index] =
            __float2bfloat16_rn(out_first);

        out_bf16[second_index] =
            __float2bfloat16_rn(out_second);
    }
}

torch::Tensor rope_bf16_cuda(
    torch::Tensor x_bf16,
    torch::Tensor cos_table,
    torch::Tensor sin_table
) {
    TORCH_CHECK(
        x_bf16.is_cuda(),
        "x_bf16 must be a CUDA tensor"
    );

    TORCH_CHECK(
        cos_table.is_cuda(),
        "cos_table must be a CUDA tensor"
    );

    TORCH_CHECK(
        sin_table.is_cuda(),
        "sin_table must be a CUDA tensor"
    );

    TORCH_CHECK(
        x_bf16.scalar_type() == torch::kBFloat16,
        "x_bf16 must have dtype torch.bfloat16"
    );

    TORCH_CHECK(
        cos_table.scalar_type() == torch::kFloat32,
        "cos_table must have dtype torch.float32"
    );

    TORCH_CHECK(
        sin_table.scalar_type() == torch::kFloat32,
        "sin_table must have dtype torch.float32"
    );

    TORCH_CHECK(
        x_bf16.device() == cos_table.device() &&
        x_bf16.device() == sin_table.device(),
        "All tensors must be on the same CUDA device"
    );

    TORCH_CHECK(
        x_bf16.dim() == 3,
        "x_bf16 must have shape "
        "[num_heads, seq_len, head_dim]"
    );

    TORCH_CHECK(
        cos_table.dim() == 2,
        "cos_table must have shape "
        "[seq_len, head_dim]"
    );

    TORCH_CHECK(
        sin_table.dim() == 2,
        "sin_table must have shape "
        "[seq_len, head_dim]"
    );

    const c10::cuda::CUDAGuard device_guard(
        x_bf16.device()
    );

    x_bf16 = x_bf16.contiguous();
    cos_table = cos_table.contiguous();
    sin_table = sin_table.contiguous();

    const int64_t num_heads64 =
        x_bf16.size(0);

    const int64_t seq_len64 =
        x_bf16.size(1);

    const int64_t head_dim64 =
        x_bf16.size(2);

    TORCH_CHECK(
        head_dim64 > 0 && head_dim64 % 2 == 0,
        "head_dim must be a positive even number"
    );

    TORCH_CHECK(
        cos_table.size(0) == seq_len64 &&
        cos_table.size(1) == head_dim64,
        "cos_table must have shape [",
        seq_len64,
        ", ",
        head_dim64,
        "]"
    );

    TORCH_CHECK(
        sin_table.size(0) == seq_len64 &&
        sin_table.size(1) == head_dim64,
        "sin_table must have shape [",
        seq_len64,
        ", ",
        head_dim64,
        "]"
    );

    const int64_t rows64 =
        num_heads64 * seq_len64;

    TORCH_CHECK(
        rows64 <= std::numeric_limits<int>::max(),
        "rows exceeds the supported int range"
    );

    TORCH_CHECK(
        seq_len64 <= std::numeric_limits<int>::max(),
        "seq_len exceeds the supported int range"
    );

    TORCH_CHECK(
        head_dim64 <= std::numeric_limits<int>::max(),
        "head_dim exceeds the supported int range"
    );

    const int rows =
        static_cast<int>(rows64);

    const int seq_len =
        static_cast<int>(seq_len64);

    const int head_dim =
        static_cast<int>(head_dim64);

    auto out_bf16 =
        torch::empty_like(x_bf16);

    constexpr int threads = 256;

    const dim3 block(threads);
    const dim3 grid(rows);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            x_bf16.get_device()
        ).stream();

    rope_bf16_vec2_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        reinterpret_cast<const __nv_bfloat16*>(
            x_bf16.data_ptr<at::BFloat16>()
        ),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            out_bf16.data_ptr<at::BFloat16>()
        ),
        rows,
        seq_len,
        head_dim
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_bf16;
}