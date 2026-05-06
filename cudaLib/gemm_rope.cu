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

/*
This function. take input tensor in int8, apply RoPE, and output int8 tensor. 

* Inputs:
- x: int8 [num_heads, seq_len, head_dim]
- scale_x: float32 [num_heads, seq_len] or [seq_len] (if shared across heads)
- cos: int8 [seq_len, head_dim]
- sin: int8 [seq_len, head_dim]
- scale_cos: float32 scalar
- scale_sin: float32 scalar

* Output:
- out: int8 [num_heads, seq_len, head_dim]
- scale_out: float32 [num_heads, seq_len]
*/

__global__ void rope_int8_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ scale_x,
    const int8_t* __restrict__ cos,
    float scale_cos,
    const int8_t* __restrict__ sin,
    float scale_sin,
    int head_dim,
    int seq_len,
    int8_t* __restrict__ out,
    float* __restrict__ scale_out, 
    int scale_mode // 0 = per-token per-head, 1 = per-token shared across heads
) {
    int head = (int)blockIdx.z;
    int pos  = (int)blockIdx.y;

    int half = head_dim >> 1;
    if (pos >= seq_len || (head_dim & 1)) return;

    int row_idx  = head * seq_len + pos;
    int row_base = row_idx * head_dim;

    float sx;
    if (scale_mode == 0) {
        sx = (float)scale_x[row_idx];
    } else {
        sx = (float)scale_x[pos];
    }

    float local_max = 0.0f;

    // pass 1: compute max abs value for scaling
    for (int col = threadIdx.x; col < half; col += blockDim.x) {
        float x1 = (float)x[row_base + col]        * sx;
        float x2 = (float)x[row_base + col + half] * sx;

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
        float x1 = (float)x[row_base + col]        * sx;
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
    torch::Tensor x,          // int8 [num_heads, seq_len, head_dim]
    torch::Tensor scale_x,    // float32 [num_heads, seq_len] or [seq_len] (if shared across heads)
    torch::Tensor cos,        // int8 [seq_len, head_dim]
    float scale_cos,  
    torch::Tensor sin,        // int8 [seq_len, head_dim]
    float scale_sin)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
    TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kChar, "x must be int8");
    TORCH_CHECK(cos.dtype() == torch::kChar, "cos must be int8");
    TORCH_CHECK(sin.dtype() == torch::kChar, "sin must be int8");
    TORCH_CHECK(scale_x.dtype() == torch::kFloat, "scale_x must be float32");

    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [num_heads, seq_len, head_dim]");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D [seq_len, head_dim]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D [seq_len, head_dim]");

    int num_heads = x.size(0);
    int seq_len   = x.size(1);
    int head_dim  = x.size(2);

    TORCH_CHECK((head_dim % 2) == 0, "head_dim must be even");
    TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim,
                "cos shape must be [seq_len, head_dim]");
    TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim,
                "sin shape must be [seq_len, head_dim]");

    auto xq   = x.contiguous();
    auto sx   = scale_x.contiguous();
    auto cosq = cos.contiguous();
    auto sinq = sin.contiguous();

    auto out = torch::empty_like(xq);
    auto scale_out = torch::empty({num_heads, seq_len},
        x.options().dtype(torch::kFloat));

    int scale_mode;
    if (scale_x.dim() == 2) {
        if (scale_x.size(0) != num_heads || scale_x.size(1) != seq_len) {
            TORCH_CHECK(false, "If scale_x is 2D, it must have shape [num_heads, seq_len]");
        }
        scale_mode = 0; // per-token per-head
    } else if (scale_x.dim() == 1) {
        if (scale_x.size(0) != seq_len) {
            TORCH_CHECK(false, "If scale_x is 1D, it must have shape [seq_len]");
        }
        scale_mode = 1; // per-token shared across heads
    } else {
        TORCH_CHECK(false, "Invalid shape for scale_x");
    }

    dim3 block(512);
    dim3 grid(1, (unsigned)seq_len, (unsigned)num_heads);

    auto stream = at::cuda::getCurrentCUDAStream();

    rope_int8_kernel<<<grid, block, 0, stream>>>(
        xq.data_ptr<int8_t>(),
        sx.data_ptr<float>(),
        cosq.data_ptr<int8_t>(),
        scale_cos,
        sinq.data_ptr<int8_t>(),
        scale_sin,
        head_dim,
        seq_len,
        out.data_ptr<int8_t>(),
        scale_out.data_ptr<float>(),
        scale_mode);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(out, scale_out);
}


