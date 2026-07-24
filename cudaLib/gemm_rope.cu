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


__global__ void rope_int8_kernel(
    const int8_t* __restrict__ x,  // [outer_size, seq_len, D]
    const float* __restrict__ scale_x, // [outer_size, seq_len] or [seq_len]
    const int8_t* __restrict__ cos,  // [seq_len, D]
    float scale_cos,  
    const int8_t* __restrict__ sin,  // [seq_len, D]
    float scale_sin,
    int outer_size,     // H for 3D, B*H for 4D
    int num_heads,      // H
    int head_dim,
    int seq_len,    
    int8_t* __restrict__ out,  // [outer_size, seq_len, D]
    float* __restrict__ scale_out,  // [outer_size, seq_len]
    int scale_mode  
) {
    int outer = (int)blockIdx.z;
    int pos   = (int)blockIdx.y;

    int half = head_dim >> 1;

    if (outer >= outer_size || pos >= seq_len || (head_dim & 1)) {
        return;
    }

    // x/out layout is logically:
    // 3D: [H, T, D]
    // 4D: [B, H, T, D]
    //
    // After contiguous flattening, both can be viewed as:
    // [outer_size, T, D]
    int row_idx  = outer * seq_len + pos;
    int row_base = row_idx * head_dim;

    float sx;

    if (scale_mode == 0) { // scale_x shape is [outer_size, seq_len]
        sx = scale_x[outer * seq_len + pos];
    } else if (scale_mode == 1) {  // scale_x shape is [batch_size, seq_len]
        int batch = outer / num_heads;
        sx = scale_x[batch * seq_len + pos];
    } else { // scale_mode == 2, scale_x shape is [seq_len]
        sx = scale_x[pos];
    }

    float local_max = 0.0f;

    // pass 1: compute max abs value
    for (int col = threadIdx.x; col < half; col += blockDim.x) {
        float x1 = (float)x[row_base + col] * sx;
        float x2 = (float)x[row_base + col + half] * sx;

        // cos/sin are shared across batch and head.
        float c = (float)cos[pos * head_dim + col] * scale_cos;
        float s = (float)sin[pos * head_dim + col] * scale_sin;

        float y1 = x1 * c - x2 * s;
        float y2 = x1 * s + x2 * c;

        local_max = fmaxf(local_max, fabsf(y1));
        local_max = fmaxf(local_max, fabsf(y2));
    }

    float max_abs = block_reduce_max(local_max);

    float s_out   = fmaxf(max_abs / 127.0f, 1e-8f);
    float inv_out = 1.0f / s_out;

    if (threadIdx.x == 0) {
        scale_out[row_idx] = s_out;
    }

    __syncthreads();

    // pass 2: apply RoPE and quantize
    for (int col = threadIdx.x; col < half; col += blockDim.x) {
        float x1 = (float)x[row_base + col] * sx;
        float x2 = (float)x[row_base + col + half] * sx;

        float c = (float)cos[pos * head_dim + col] * scale_cos;
        float s = (float)sin[pos * head_dim + col] * scale_sin;

        float y1 = x1 * c - x2 * s;
        float y2 = x1 * s + x2 * c;

        int q1 = __float2int_rn(y1 * inv_out);
        int q2 = __float2int_rn(y2 * inv_out);

        q1 = max(-128, min(127, q1));
        q2 = max(-128, min(127, q2));

        out[row_base + col]        = (int8_t)q1;
        out[row_base + col + half] = (int8_t)q2;
    }
}


std::tuple<torch::Tensor, torch::Tensor> rope_int8_host(
    torch::Tensor x,          // int8 [H, seq_len, D] or [B, H, seq_len, D]
    torch::Tensor scale_x,    // float32 [H, seq_len], [B, H, seq_len], or [seq_len]
    torch::Tensor cos,        // int8 [seq_len, D]
    float scale_cos,
    torch::Tensor sin,        // int8 [seq_len, D]
    float scale_sin
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
    TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kChar, "x must be int8");
    TORCH_CHECK(cos.dtype() == torch::kChar, "cos must be int8");
    TORCH_CHECK(sin.dtype() == torch::kChar, "sin must be int8");
    TORCH_CHECK(scale_x.dtype() == torch::kFloat, "scale_x must be float32");

    TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
                "x must be [num_heads, seq_len, head_dim] or [batch_size, num_heads, seq_len, head_dim]");

    TORCH_CHECK(cos.dim() == 2, "cos must be 2D [seq_len, head_dim]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D [seq_len, head_dim]");

    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    int outer_size;

    bool is_batched = x.dim() == 4;

    if (is_batched) {
        batch_size = x.size(0);
        num_heads  = x.size(1);
        seq_len    = x.size(2);
        head_dim   = x.size(3);
        outer_size = batch_size * num_heads;
    } else {
        batch_size = 1;
        num_heads  = x.size(0);
        seq_len    = x.size(1);
        head_dim   = x.size(2);
        outer_size = num_heads;
    }

    TORCH_CHECK((head_dim % 2) == 0, "head_dim must be even");

    TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim,
                "cos shape must be [seq_len, head_dim]");

    TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim,
                "sin shape must be [seq_len, head_dim]");

    int scale_mode;

    if (!is_batched) {
        // x: [H, T, D]

        if (scale_x.dim() == 2) {
            TORCH_CHECK(scale_x.size(0) == num_heads &&
                        scale_x.size(1) == seq_len,
                        "For 3D x, 2D scale_x must be [num_heads, seq_len]");
            scale_mode = 0;
        } else if (scale_x.dim() == 1) {
            TORCH_CHECK(scale_x.size(0) == seq_len,
                        "For 3D x, 1D scale_x must be [seq_len]");
            scale_mode = 2;
        } else {
            TORCH_CHECK(false,
                        "For 3D x, scale_x must be [num_heads, seq_len] or [seq_len]");
        }

    } else {
        // x: [B, H, T, D]

        if (scale_x.dim() == 3) {
            TORCH_CHECK(scale_x.size(0) == batch_size &&
                        scale_x.size(1) == num_heads &&
                        scale_x.size(2) == seq_len,
                        "For 4D x, 3D scale_x must be [batch_size, num_heads, seq_len]");
            scale_mode = 0;
        } else if (scale_x.dim() == 2) {
            TORCH_CHECK(scale_x.size(0) == batch_size &&
                        scale_x.size(1) == seq_len,
                        "For 4D x, 2D scale_x must be [batch_size, seq_len]");
            scale_mode = 1;
        } else if (scale_x.dim() == 1) {
            TORCH_CHECK(scale_x.size(0) == seq_len,
                        "For 4D x, 1D scale_x must be [seq_len]");
            scale_mode = 2;
        } else {
            TORCH_CHECK(false,
                        "For 4D x, scale_x must be [B,H,T], [B,T], or [T]");
        }
    }

    auto xq   = x.contiguous();
    auto sx   = scale_x.contiguous();
    auto cosq = cos.contiguous();
    auto sinq = sin.contiguous();

    auto out = torch::empty_like(xq);

    torch::Tensor scale_out;

    if (is_batched) {
        scale_out = torch::empty(
            {batch_size, num_heads, seq_len},
            x.options().dtype(torch::kFloat)
        );
    } else {
        scale_out = torch::empty(
            {num_heads, seq_len},
            x.options().dtype(torch::kFloat)
        );
    }

    dim3 block(256);
    dim3 grid(1, (unsigned)seq_len, (unsigned)outer_size);

    auto stream = at::cuda::getCurrentCUDAStream();
    size_t shared_mem_size = 0;

    rope_int8_kernel<<<grid, block, shared_mem_size, stream>>>(
        xq.data_ptr<int8_t>(),
        sx.data_ptr<float>(),
        cosq.data_ptr<int8_t>(),
        scale_cos,
        sinq.data_ptr<int8_t>(),
        scale_sin,
        outer_size,
        num_heads,
        head_dim,
        seq_len,
        out.data_ptr<int8_t>(),
        scale_out.data_ptr<float>(),
        scale_mode
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(out, scale_out);
}
