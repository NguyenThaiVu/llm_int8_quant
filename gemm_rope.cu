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


template <typename scalar_t>
__global__ void apply_rope_int8_kernel(
    const int8_t* __restrict__ x,          // [num_heads, seq_len, head_dim]
    const scalar_t* __restrict__ scale_x,  // [num_heads, seq_len]
    const int8_t* __restrict__ cos,        // [seq_len, head_dim]
    float scale_cos,
    const int8_t* __restrict__ sin,        // [seq_len, head_dim]
    float scale_sin,
    int head_dim,
    int seq_len,
    int8_t* __restrict__ out,
    const scalar_t* __restrict__ scale_out // [num_heads, seq_len]
) {
    int head = (int)blockIdx.z;
    int pos  = (int)blockIdx.y;
    int j    = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

    int half = head_dim / 2;
    if (pos >= seq_len || j >= half) return;

    // Query position at [head, pos]
    float sx = static_cast<float>(scale_x[head * seq_len + pos]);
    float s_out_h = static_cast<float>(scale_out[head * seq_len + pos]);

    int row_base = (head * seq_len + pos) * head_dim;

    float x1 = (float)x[row_base + j]        * sx;
    float x2 = (float)x[row_base + j + half] * sx;

    float c = (float)cos[pos * head_dim + j] * scale_cos;
    float s = (float)sin[pos * head_dim + j] * scale_sin;

    float out1 = x1 * c - x2 * s;
    float out2 = x1 * s + x2 * c;

    float inv_sout = 1.0f / s_out_h;
    int q1 = __float2int_rn(out1 * inv_sout);
    int q2 = __float2int_rn(out2 * inv_sout);

    q1 = max(-128, min(127, q1));
    q2 = max(-128, min(127, q2));

    out[row_base + j]        = (int8_t)q1;
    out[row_base + j + half] = (int8_t)q2;
}



torch::Tensor apply_rope_int8_host(
    torch::Tensor x,          // int8, shape [num_heads, seq_len, head_dim]
    torch::Tensor scale_x,    // float32 or bfloat16, shape [num_heads, seq_len]
    torch::Tensor cos,        // int8, shape [seq_len, head_dim]
    float scale_cos,
    torch::Tensor sin,        // int8, shape [seq_len, head_dim]
    float scale_sin,
    torch::Tensor scale_out)  // float32 or bfloat16, shape [num_heads, seq_len]
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
    TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be a CUDA tensor");
    TORCH_CHECK(scale_out.is_cuda(), "scale_out must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kChar, "x must be int8");
    TORCH_CHECK(cos.dtype() == torch::kChar, "cos must be int8");
    TORCH_CHECK(sin.dtype() == torch::kChar, "sin must be int8");

    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [num_heads, seq_len, head_dim]");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D tensor [seq_len, head_dim]");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D tensor [seq_len, head_dim]");

    int num_heads = x.size(0);
    int seq_len   = x.size(1);
    int head_dim  = x.size(2);

    TORCH_CHECK(scale_x.dim() == 2, "scale_x must be 2D");
    TORCH_CHECK(scale_x.size(0) == num_heads && scale_x.size(1) == seq_len,
            "scale_x shape must be (num_heads, seq_len)");
    TORCH_CHECK(scale_out.dim() == 2, "scale_out must be 2D");
    TORCH_CHECK(scale_out.size(0) == num_heads && scale_out.size(1) == seq_len,
            "scale_out shape must be (num_heads, seq_len)");
    TORCH_CHECK(cos.size(0) == seq_len && cos.size(1) == head_dim,
                "cos shape must be (seq_len, head_dim)");
    TORCH_CHECK(sin.size(0) == seq_len && sin.size(1) == head_dim,
                "sin shape must be (seq_len, head_dim)");

    // Allow float32 or bfloat16, and require they match
    auto sx_dtype = scale_x.scalar_type();
    auto so_dtype = scale_out.scalar_type();
    TORCH_CHECK(
        (sx_dtype == torch::kFloat || sx_dtype == torch::kBFloat16),
        "scale_x must be float32 or bfloat16");
    TORCH_CHECK(
        sx_dtype == so_dtype,
        "scale_x and scale_out must have the same dtype (both float32 or both bfloat16)");

    auto xq   = x.contiguous();
    auto sx   = scale_x.contiguous();
    auto cosq = cos.contiguous();
    auto sinq = sin.contiguous();
    auto sout = scale_out.contiguous();
    auto out  = torch::empty_like(xq);

    dim3 block(256); // threads per block over half-dim
    dim3 grid(
        (head_dim / 2 + block.x - 1) / block.x,  // over pairs in D
        (unsigned)seq_len,                       // over positions
        (unsigned)num_heads                      // over heads
    );

    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch over scale dtype: float or bfloat16
    if (sx_dtype == torch::kFloat) {
        using scalar_t = float;
        apply_rope_int8_kernel<scalar_t><<<grid, block, 0, stream>>>(
            xq.data_ptr<int8_t>(),
            sx.data_ptr<scalar_t>(),
            cosq.data_ptr<int8_t>(),
            scale_cos,
            sinq.data_ptr<int8_t>(),
            scale_sin,
            head_dim,
            seq_len,
            out.data_ptr<int8_t>(),
            sout.data_ptr<scalar_t>());
    } else if (sx_dtype == torch::kBFloat16) {
        using scalar_t = at::BFloat16;
        apply_rope_int8_kernel<scalar_t><<<grid, block, 0, stream>>>(
            xq.data_ptr<int8_t>(),
            sx.data_ptr<scalar_t>(),
            cosq.data_ptr<int8_t>(),
            scale_cos,
            sinq.data_ptr<int8_t>(),
            scale_sin,
            head_dim,
            seq_len,
            out.data_ptr<int8_t>(),
            sout.data_ptr<scalar_t>());
    } else {
        TORCH_CHECK(false, "Unhandled dtype for scale_x / scale_out");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



