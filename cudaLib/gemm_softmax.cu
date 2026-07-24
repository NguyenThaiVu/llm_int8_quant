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


// --------------------------------------------------
// BF16 softmax kernel
// x: BF16 [..., d_model]
// y: BF16 [..., d_model]
// --------------------------------------------------

__global__ void softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ y,
    int d_model
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    __shared__ float shared_row_max;
    __shared__ float shared_sum_exp;

    // --------------------------------------------------
    // 1. First pass: compute row max
    // --------------------------------------------------

    float local_max = -FLT_MAX;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = __bfloat162float(x[row_offset + i]);
        local_max = fmaxf(local_max, x_fp);
    }

    float row_max_reduced = block_reduce_max(local_max);

    if (tid == 0) {
        shared_row_max = row_max_reduced;
    }

    __syncthreads();

    float row_max = shared_row_max;

    // --------------------------------------------------
    // 2. Second pass: compute sum_exp
    // --------------------------------------------------

    float local_sum_exp = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = __bfloat162float(x[row_offset + i]);
        float e = expf(x_fp - row_max);
        local_sum_exp += e;
    }

    float sum_exp_reduced = block_reduce_sum(local_sum_exp);

    if (tid == 0) {
        shared_sum_exp = sum_exp_reduced;
    }

    __syncthreads();

    float sum_exp = shared_sum_exp;
    bool valid = sum_exp > 0.0f && isfinite(sum_exp);

    // --------------------------------------------------
    // 3. Third pass: compute softmax and store BF16
    // --------------------------------------------------

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = __bfloat162float(x[row_offset + i]);
        float e = expf(x_fp - row_max);

        float out = valid ? e / sum_exp : 0.0f;

        if (!isfinite(out)) {
            out = 0.0f;
        }

        y[row_offset + i] = __float2bfloat16(out);
    }
}

torch::Tensor softmax_bf16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    auto x_contig = x.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");

    const int64_t num_rows = x_contig.numel() / d_model;

    TORCH_CHECK(
        num_rows * d_model == x_contig.numel(),
        "x.numel() must be divisible by last dimension"
    );

    auto y = torch::empty_like(x_contig);

    int threads = static_cast<int>(std::min<int64_t>(d_model, 256));
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(num_rows));

    auto stream = at::cuda::getCurrentCUDAStream();

    softmax_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        static_cast<int>(d_model)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y.view(x.sizes());
}


__global__ void softmax_int8_kernel(
    const int8_t* __restrict__ x_int8,
    const float*  __restrict__ scale_x,
    int8_t*       __restrict__ y_int8,
    float*        __restrict__ scale_y,
    int d_model
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    __shared__ float shared_row_max;
    __shared__ float shared_sum_exp;
    __shared__ float shared_scale_y;

    float sx = scale_x[row];

    // --------------------------------------------------
    // 1. First pass: compute row max
    // --------------------------------------------------
    float local_max = -FLT_MAX;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        local_max = fmaxf(local_max, x_fp);
    }

    float row_max_reduced = block_reduce_max(local_max);

    if (tid == 0) {
        shared_row_max = row_max_reduced;
    }

    __syncthreads();

    float row_max = shared_row_max;

    // --------------------------------------------------
    // 2. Second pass: compute sum_exp
    // --------------------------------------------------
    float local_sum_exp = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        float e = expf(x_fp - row_max);
        local_sum_exp += e;
    }

    float sum_exp_reduced = block_reduce_sum(local_sum_exp);

    if (tid == 0) {
        float sum_exp = sum_exp_reduced;

        shared_sum_exp = sum_exp;

        // max softmax value = 1 / sum_exp
        float sy = (sum_exp > 0.0f && isfinite(sum_exp))
                 ? 1.0f / (sum_exp * 127.0f)
                 : 1.0f;

        shared_scale_y = sy;
        scale_y[row] = sy;
    }

    __syncthreads();

    float sum_exp = shared_sum_exp;
    float sy = shared_scale_y;

    bool valid = (sum_exp > 0.0f && isfinite(sum_exp) && sy > 0.0f);

    // --------------------------------------------------
    // 3. Third pass: recompute exp and quantize
    // --------------------------------------------------
    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;
        float e = expf(x_fp - row_max);

        // Because scale_y = 1 / (sum_exp * 127),
        // q = softmax / scale_y = exp(x - max) * 127.
        float qf = valid ? e * 127.0f : 0.0f;

        if (!isfinite(qf)) {
            qf = 0.0f;
        }

        int q = __float2int_rn(qf);

        // Softmax output is non-negative.
        q = max(0, min(127, q));

        y_int8[row_offset + i] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> softmax_int8_cuda(
    torch::Tensor x_int8,   // int8, shape [..., d_model]
    torch::Tensor scale_x   // float, shape [...]
) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be CUDA");
    TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8");
    TORCH_CHECK(x_int8.dim() >= 2, "x_int8 must have at least 2 dimensions");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == at::kFloat, "scale_x must be float32");

    auto x_contig = x_int8.contiguous();
    auto scale_contig = scale_x.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");

    const int64_t num_rows = x_contig.numel() / d_model;

    TORCH_CHECK(
        num_rows * d_model == x_contig.numel(),
        "x_int8.numel() must be divisible by last dimension"
    );

    TORCH_CHECK(
        scale_contig.numel() == num_rows,
        "scale_x must have one scale per row, i.e. shape x_int8.sizes()[:-1]"
    );

    auto y_int8 = torch::empty_like(x_contig);
    auto scale_y = torch::empty_like(scale_contig);

    int threads = static_cast<int>(std::min<int64_t>(d_model, 256));
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(num_rows));

    // For implicit-shared-memory, no dynamic shared memory.
    size_t shared_mem_size = 0;

    auto stream = at::cuda::getCurrentCUDAStream();

    softmax_int8_kernel<<<grid, block, shared_mem_size, stream>>>(
        x_contig.data_ptr<int8_t>(),
        scale_contig.data_ptr<float>(),
        y_int8.data_ptr<int8_t>(),
        scale_y.data_ptr<float>(),
        static_cast<int>(d_model)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        y_int8.view(x_int8.sizes()),
        scale_y.view(scale_x.sizes())
    );
}


// ============================================================
// Explicit shared-memory INT8 softmax
//
// x_int8  : [num_rows, d_model]
// scale_x : [num_rows]
// y_int8  : [num_rows, d_model]
// scale_y : [num_rows]
//
// One CUDA block handles one row.
//
// Dynamic shared memory layout:
//
// smem:
// | x_s[0 ... d_model-1] | red_s[0 ... blockDim.x-1] |
//
// x_s   stores dequantized logits, then exp(logit - max)
// red_s is used for explicit reductions
// ============================================================

__global__ void softmax_int8_explicit_shared_kernel(
    const int8_t* __restrict__ x_int8,
    const float*  __restrict__ scale_x,
    int8_t*       __restrict__ y_int8,
    float*        __restrict__ scale_y,
    int d_model
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int row_offset = row * d_model;

    extern __shared__ float smem[];
    float* x_s   = smem;             // [d_model]
    float* red_s = smem + d_model;   // [blockDim.x]

    __shared__ float shared_row_max;
    __shared__ float shared_sum_exp;
    __shared__ float shared_scale_y;

    float sx = scale_x[row];

    // --------------------------------------------------
    // 1. Dequantize and compute local max
    // --------------------------------------------------
    float local_max = -FLT_MAX;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float x_fp = static_cast<float>(x_int8[row_offset + i]) * sx;

        x_s[i] = x_fp;

        local_max = fmaxf(local_max, x_fp);
    }

    __syncthreads();

    // --------------------------------------------------
    // 2. Explicit shared-memory reduction for row max
    // --------------------------------------------------
    red_s[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_s[tid] = fmaxf(red_s[tid], red_s[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_row_max = red_s[0];
    }

    __syncthreads();

    float row_max = shared_row_max;

    // --------------------------------------------------
    // 3. Compute exp(x - max) and local sum
    // --------------------------------------------------
    float local_sum_exp = 0.0f;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float e = expf(x_s[i] - row_max);

        // Reuse x_s to store exp(x - max)
        x_s[i] = e;

        local_sum_exp += e;
    }

    __syncthreads();

    // --------------------------------------------------
    // 4. Explicit shared-memory reduction for sum_exp
    // --------------------------------------------------
    red_s[tid] = local_sum_exp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_s[tid] += red_s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum_exp = red_s[0];

        shared_sum_exp = sum_exp;

        // Softmax max value is 1 / sum_exp because max shifted exp is 1.
        //
        // scale_y = max(softmax) / 127
        //         = (1 / sum_exp) / 127
        //         = 1 / (sum_exp * 127)
        float sy = (sum_exp > 0.0f && isfinite(sum_exp))
                 ? 1.0f / (sum_exp * 127.0f)
                 : 1.0f;

        shared_scale_y = sy;
        scale_y[row] = sy;
    }

    __syncthreads();

    float sum_exp = shared_sum_exp;
    float sy = shared_scale_y;

    // --------------------------------------------------
    // 5. Quantize softmax output
    //
    // softmax_i = exp_i / sum_exp
    // y_q       = round(softmax_i / sy)
    //
    // Since sy = 1 / (sum_exp * 127),
    //
    // y_q = round((exp_i / sum_exp) * (sum_exp * 127))
    //     = round(exp_i * 127)
    //
    // So we do not need to explicitly divide by sum_exp here.
    // The output scale_y stores the normalization factor.
    // --------------------------------------------------
    bool valid = (sum_exp > 0.0f && isfinite(sum_exp) && sy > 0.0f);

    for (int i = tid; i < d_model; i += blockDim.x) {
        float e = x_s[i];

        float qf = valid ? e * 127.0f : 0.0f;

        if (!isfinite(qf)) {
            qf = 0.0f;
        }

        int q = __float2int_rn(qf);

        // Softmax is non-negative, so clamp to [0, 127]
        q = max(0, min(127, q));

        y_int8[row_offset + i] = static_cast<int8_t>(q);
    }
}

std::tuple<torch::Tensor, torch::Tensor> softmax_int8_explicit_shared_cuda(
    torch::Tensor x_int8,   // int8, shape [..., d_model]
    torch::Tensor scale_x   // float, shape [...]
) {
    TORCH_CHECK(x_int8.is_cuda(), "x_int8 must be CUDA");
    TORCH_CHECK(x_int8.scalar_type() == at::kChar, "x_int8 must be int8");
    TORCH_CHECK(x_int8.dim() >= 2, "x_int8 must have at least 2 dimensions");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == at::kFloat, "scale_x must be float32");

    auto x_contig = x_int8.contiguous();
    auto scale_contig = scale_x.contiguous();

    const int64_t d_model = x_contig.size(-1);
    TORCH_CHECK(d_model > 0, "d_model must be > 0");

    const int64_t num_rows = x_contig.numel() / d_model;

    TORCH_CHECK(
        num_rows * d_model == x_contig.numel(),
        "x_int8.numel() must be divisible by d_model"
    );

    TORCH_CHECK(
        scale_contig.numel() == num_rows,
        "scale_x must have one scale per row, i.e. shape x_int8.sizes()[:-1]"
    );

    auto y_int8 = torch::empty_like(x_contig);
    auto scale_y = torch::empty_like(scale_contig);

    int threads = static_cast<int>(std::min<int64_t>(d_model, 256));
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid(static_cast<unsigned int>(num_rows));

    // Explicit shared memory:
    // x_s[d_model] + red_s[threads]
    size_t shared_mem_size =
        static_cast<size_t>(d_model + threads) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    softmax_int8_explicit_shared_kernel<<<
        grid,
        block,
        shared_mem_size,
        stream
    >>>(
        x_contig.data_ptr<int8_t>(),
        scale_contig.data_ptr<float>(),
        y_int8.data_ptr<int8_t>(),
        scale_y.data_ptr<float>(),
        static_cast<int>(d_model)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        y_int8.view(x_int8.sizes()),
        scale_y.view(scale_x.sizes())
    );
}


bool check_broadcastable(const at::Tensor& x, const at::Tensor& mask) {
    // Check if mask can be broadcast to x's shape

    // 2D input
    if (mask.sizes() == x.sizes()) return true;

    // 3D input with broadcasting 
    if (mask.dim() + 1 == x.dim()) {
        auto x_last2 = x.sizes().slice(x.dim() - 2);
        auto m_last2 = mask.sizes().slice(mask.dim() - 2);
        return (x_last2 == m_last2) && (mask.size(0) == x_last2[0]);
    }

    // 4D input with broadcasting
    if (mask.dim() + 2 == x.dim()) {
        auto x_last2 = x.sizes().slice(x.dim() - 2);
        auto m_last2 = mask.sizes().slice(mask.dim() - 2);
        return (x_last2 == m_last2) && (mask.size(0) == x_last2[0]) && (mask.size(1) == x_last2[0]);
    }

    return false;
}

/*
This kernel compute int8 softmax with masking. 
The mask can be either the same shape as x_q (no broadcasting) or broadcastable to x_q with broadcasting on the first dimension.

Input
- x_q: int8 tensor of shape [..., C]
- scale_x: float tensor of shape [...]
- mask: uint8 tensor of shape [..., C] or [C, C] (broadcastable to x_q)

Output
- y_q: int8 tensor of shape [..., C]
- scale_y: float tensor of shape [...],
*/
__global__ void softmax_lastdim_int8_masking_kernel(
    const int8_t* __restrict__ x_q,
    int8_t* __restrict__ y_q,
    const float* __restrict__ scale_x,  // [num_vecs]
    float* __restrict__ scale_y,  // [num_vecs]
    const uint8_t* __restrict__ mask,      // [num_vecs * dim2]
    int64_t num_vecs,
    int64_t dim2,
    bool broadcast_first_dim)
{
    int64_t vec = (int64_t)blockIdx.x;
    if (vec >= num_vecs) return;

    float sx = (float)scale_x[vec];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ int8_t xq_s[];
    const int8_t* xq = x_q + vec * dim2;
    int8_t* yq = y_q + vec * dim2;

    const uint8_t* mask_vec;
    if (broadcast_first_dim) {
        int64_t row = vec % dim2;
        mask_vec = mask + row * dim2;
    } else {
        mask_vec = mask + vec * dim2;
    }

    __shared__ float row_scale_out;

    for (int64_t j = tid; j < dim2; j += stride) {
        xq_s[j] = xq[j];
    }
    __syncthreads();

    // ============================================================
    // 1) max over unmasked positions
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) continue;
        float v = sx * (float)xq_s[j];
        local_max = fmaxf(local_max, v);
    }

    float max_val = block_reduce_max_neg_inf(local_max);

    // ============================================================
    // 2) sum over unmasked positions
    float local_sum = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) continue;
        float v = sx * (float)xq_s[j];
        local_sum += __expf(v - max_val);
    }

    float sum_val = block_reduce_sum(local_sum);
    sum_val = fmaxf(sum_val, 1e-20f); // avoid div-by-zero

    // Share sum_vale for all threads
    __shared__ float sum_value_shared;
    if (tid == 0) {
        sum_value_shared = sum_val;
    }
    __syncthreads();

    float inv_sum = 1.0f / sum_value_shared;  // all threads read sum_value_shared

    // 3) compute max-abs of output values
    float local_absmax = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) continue;
        float v = sx * (float)xq_s[j];
        float p = __expf(v - max_val) * inv_sum;
        local_absmax = fmaxf(local_absmax, fabsf(p));
    }

    float absmax_val = block_reduce_max(local_absmax);

    if (tid == 0) {
        float s = (absmax_val > 0.0f) ? (absmax_val / 127.0f) : 1.0f;
        row_scale_out = s;
        scale_y[vec] = s;
    }
    __syncthreads();

    float inv_scale_out = 1.0f / row_scale_out;

    // 4) quantize and write
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) {
            yq[j] = (int8_t)0;
            continue;
        }

        float v = sx * (float)xq_s[j];
        float p = __expf(v - max_val) * inv_sum;

        int q = __float2int_rn(p * inv_scale_out);
        q = max(0, min(127, q));
        yq[j] = (int8_t)q;
    }
}

std::tuple<torch::Tensor, torch::Tensor> softmax_lastdim_int8_masking_cuda(
    torch::Tensor x_q,          // int8, shape [..., C]
    torch::Tensor scale_x,      // float shape [...]
    torch::Tensor mask          // uint8, shape [..., C] or [C, C] (broadcast)
) {
    TORCH_CHECK(x_q.is_cuda(), "x_q must be CUDA");
    TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
    TORCH_CHECK(x_q.dim() >= 1, "x_q must be at least 1D");

    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.scalar_type() == at::kFloat, "scale_x must be float32");

    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
    TORCH_CHECK(mask.scalar_type() == at::kByte, "mask must be uint8");

    auto xq = x_q.contiguous();

    int64_t dim2 = xq.size(-1);
    TORCH_CHECK(dim2 > 0, "last dimension must be > 0");

    int64_t num_vecs = xq.numel() / dim2;

    // Expected scale shape = leading dims of x_q
    std::vector<int64_t> expected_scale_shape;
    expected_scale_shape.reserve(xq.dim() - 1);
    for (int i = 0; i < xq.dim() - 1; ++i) {
        expected_scale_shape.push_back(xq.size(i));
    }

    TORCH_CHECK(
        scale_x.dim() == xq.dim() - 1,
        "scale_x must have shape equal to x_q.shape[:-1]"
    );

    for (int i = 0; i < xq.dim() - 1; ++i) {
        TORCH_CHECK(
            scale_x.size(i) == xq.size(i),
            "scale_x shape must equal x_q.shape[:-1]"
        );
    }

    TORCH_CHECK(
        scale_x.numel() == num_vecs,
        "scale_x numel must equal product of leading dims of x_q"
    );

    auto sx = scale_x.contiguous().view({num_vecs});

    bool broadcast_first_dim = false;
    torch::Tensor masking;

    if (mask.sizes() == xq.sizes()) {
        masking = mask.contiguous();
    } else if (check_broadcastable(xq, mask)) {
        broadcast_first_dim = true;
        masking = mask.contiguous();
    } else {
        TORCH_CHECK(false, "mask shape must be either same as x_q or broadcastable to x_q");
    }

    auto y_q = torch::empty_like(xq);
    auto sy = torch::empty({num_vecs}, sx.options().dtype(torch::kFloat));

    int threads = (int)std::min<int64_t>(dim2, 512);
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)num_vecs);

    size_t shared_mem_size = (size_t)dim2 * sizeof(int8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    softmax_lastdim_int8_masking_kernel<<<grid, block, shared_mem_size, stream>>>(
            xq.data_ptr<int8_t>(),
            y_q.data_ptr<int8_t>(),
            sx.data_ptr<float>(),
            sy.data_ptr<float>(),
            masking.data_ptr<uint8_t>(),
            num_vecs,
            dim2,
            broadcast_first_dim
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(y_q, sy.view(expected_scale_shape));
}
