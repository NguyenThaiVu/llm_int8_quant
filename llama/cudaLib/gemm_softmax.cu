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

/*
================================================================
SOFTMAX KERNELS
================================================================
*/
__inline__ __device__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__inline__ __device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}


template <typename scalar_t>
__global__ void softmax_lastdim_int8_kernel(
    const int8_t* __restrict__ x_q,
    int8_t* __restrict__ y_q,
    const scalar_t* __restrict__ scale_x,  // length num_vecs
    const scalar_t* __restrict__ scale_y,  // length num_vecs
    int64_t num_vecs,
    int64_t dim2)
{
    int64_t vec = (int64_t)blockIdx.x;
    if (vec >= num_vecs) return;

    float sx = (float)scale_x[vec];
    float scaleOut = (float)scale_y[vec];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Optional: stage x into shared memory
    extern __shared__ int8_t xq_s[];
    const int8_t* xq = x_q + vec * dim2;
    int8_t* yq       = y_q + vec * dim2;

    for (int64_t j = tid; j < dim2; j += stride) xq_s[j] = xq[j];
    __syncthreads();

    __shared__ float warp_buf[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (stride + 31) >> 5;

    // 1) max
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        local_max = fmaxf(local_max, v);
    }
    float warp_m = warp_max(local_max);
    if (lane == 0) warp_buf[warp] = warp_m;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float max_val = warp_buf[0];

    // 2) sum
    float local_sum = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        local_sum += __expf(v - max_val);
    }
    float warp_s = warp_sum(local_sum);
    if (lane == 0) warp_buf[warp] = warp_s;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float sum_val = warp_buf[0];

    // Avoid div-by-zero
    sum_val = fmaxf(sum_val, 1e-20f);

    // 3) write (quantize)
    float inv_sum = 1.0f / sum_val;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        float p = __expf(v - max_val) * inv_sum;     // in [0,1]
        
        int q = __float2int_rn(p / scaleOut);
        q = max(0, min(127, q));
        yq[j] = (int8_t)q;
    }
}

torch::Tensor softmax_lastdim_int8_cuda(
    torch::Tensor x_q,          // int8, shape [..., C]
    torch::Tensor scale_x,      // float/half/bf16, shape [num_vecs]
    torch::Tensor scale_y       // float/half/bf16, shape [num_vecs]
) {
    TORCH_CHECK(x_q.is_cuda(), "x_q must be CUDA");
    TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
    TORCH_CHECK(x_q.dim() >= 1, "x_q must be at least 1D");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.dim() == 1, "scale_x must be 1D");
    TORCH_CHECK(scale_y.is_cuda(), "scale_y must be CUDA");
    TORCH_CHECK(scale_y.dim() == 1, "scale_y must be 1D");

    auto xq = x_q.contiguous();
    auto sx = scale_x.contiguous();
    auto sy = scale_y.contiguous();
    int64_t dim2 = xq.size(-1);
    TORCH_CHECK(dim2 > 0, "last dimension must be > 0");

    int64_t num_vecs = xq.numel() / dim2;  // product of leading dims
    TORCH_CHECK(sx.numel() == num_vecs,
                "scale_x length must equal product of leading dims (num_vecs)");

    auto y_q = torch::empty_like(xq);

    // Thread config: 256 often better than 512 for softmax
    int threads = (int)std::min<int64_t>(dim2, 256);
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)num_vecs);

    // Shared memory for staging x vector (int8)
    // If dim2 can be huge, this can exceed shared memory — see note below.
    size_t shared_mem_size = (size_t)dim2 * sizeof(int8_t);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                    sx.scalar_type(), "softmax_lastdim_int8_cuda", [&] {
        softmax_lastdim_int8_kernel<scalar_t>
            <<<grid, block, shared_mem_size, stream>>>(
                xq.data_ptr<int8_t>(),
                y_q.data_ptr<int8_t>(),
                sx.data_ptr<scalar_t>(),
                scale_y.data_ptr<scalar_t>(),
                num_vecs,
                dim2);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y_q;
}

bool check_broadcastable(const at::Tensor& x, const at::Tensor& mask) {
    // Check if mask can be broadcast to x's shape
    if (mask.sizes() == x.sizes()) return true;

    if (mask.dim() + 1 == x.dim()) {
        auto x_last2 = x.sizes().slice(x.dim() - 2);
        auto m_last2 = mask.sizes().slice(mask.dim() - 2);
        return (x_last2 == m_last2) && (mask.size(0) == x_last2[0]);
    }
    return false;
}

template <typename scalar_t>
__global__ void softmax_lastdim_int8_masking_kernel(
    const int8_t* __restrict__ x_q,
    int8_t* __restrict__ y_q,
    const scalar_t* __restrict__ scale_x,  // [num_vecs]
    const scalar_t* __restrict__ scale_y,  // [num_vecs]
    const uint8_t* __restrict__ mask,      // either [num_vecs * dim2] or [dim2 * dim2]
    int64_t num_vecs,
    int64_t dim2,
    bool broadcast_first_dim)
{
    int64_t vec = (int64_t)blockIdx.x;
    if (vec >= num_vecs) return;

    float sx       = (float)scale_x[vec];
    float scaleOut = (float)scale_y[vec];

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ int8_t xq_s[];
    const int8_t* xq = x_q + vec * dim2;
    int8_t* yq       = y_q + vec * dim2;

    const uint8_t* mask_vec;
    if (broadcast_first_dim) {
        // x_q is (B, T, T), mask is (T, T)
        // num_vecs = B * T, dim2 = T
        int64_t row = vec % dim2;          // t_q
        mask_vec = mask + row * dim2;      // mask[row, :]
    } else {
        // mask has same flattened layout as x_q
        mask_vec = mask + vec * dim2;
    }

    __shared__ float warp_buf[32];
    int lane      = tid & 31;
    int warp      = tid >> 5;
    int num_warps = (stride + 31) >> 5;

    // load
    for (int64_t j = tid; j < dim2; j += stride) {
        xq_s[j] = xq[j];
    }
    __syncthreads();

    // 1) max over unmasked positions
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) continue;
        float v = sx * (float)xq_s[j];
        local_max = fmaxf(local_max, v);
    }
    float warp_m = warp_max(local_max);
    if (lane == 0) warp_buf[warp] = warp_m;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float max_val = warp_buf[0];

    // 2) sum over unmasked positions
    float local_sum = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) continue;
        float v = sx * (float)xq_s[j];
        local_sum += __expf(v - max_val);
    }
    float warp_s = warp_sum(local_sum);
    if (lane == 0) warp_buf[warp] = warp_s;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float sum_val = warp_buf[0];

    sum_val = fmaxf(sum_val, 1e-20f);
    float inv_sum = 1.0f / sum_val;

    // 3) write
    for (int64_t j = tid; j < dim2; j += stride) {
        if (!mask_vec[j]) {
            yq[j] = (int8_t)0;
            continue;
        }
        float v = sx * (float)xq_s[j];
        float p = __expf(v - max_val) * inv_sum;

        int q = __float2int_rn(p / scaleOut);
        q = max(0, min(127, q));
        yq[j] = (int8_t)q;
    }
}

torch::Tensor softmax_lastdim_int8_masking_cuda(
    torch::Tensor x_q,          // int8, shape [..., C]
    torch::Tensor scale_x,      // float/half/bf16, shape [num_vecs]
    torch::Tensor scale_y,      // float/half/bf16, shape [num_vecs]
    torch::Tensor mask          // uint8, shape [..., C] or [C, C] (broadcast)
) {
    TORCH_CHECK(x_q.is_cuda(), "x_q must be CUDA");
    TORCH_CHECK(x_q.scalar_type() == at::kChar, "x_q must be int8");
    TORCH_CHECK(x_q.dim() >= 1, "x_q must be at least 1D");
    TORCH_CHECK(scale_x.is_cuda(), "scale_x must be CUDA");
    TORCH_CHECK(scale_x.dim() == 1, "scale_x must be 1D");
    TORCH_CHECK(scale_y.is_cuda(), "scale_y must be CUDA");
    TORCH_CHECK(scale_y.dim() == 1, "scale_y must be 1D");
    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
    TORCH_CHECK(mask.scalar_type() == at::kByte, "mask must be uint8");

    auto xq = x_q.contiguous();
    auto sx = scale_x.contiguous();
    auto sy = scale_y.contiguous();

    int64_t dim2 = xq.size(-1);      // C or T
    TORCH_CHECK(dim2 > 0, "last dimension must be > 0");

    int64_t num_vecs = xq.numel() / dim2;  // product of leading dims
    TORCH_CHECK(sx.numel() == num_vecs,
                "scale_x length must equal product of leading dims (num_vecs)");
    TORCH_CHECK(sy.numel() == num_vecs,
                "scale_y length must equal product of leading dims (num_vecs)");

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

    int threads = (int)std::min<int64_t>(dim2, 256);
    threads = std::max(threads, 32);

    dim3 block(threads);
    dim3 grid((unsigned)num_vecs);

    size_t shared_mem_size = (size_t)dim2 * sizeof(int8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
                                    sx.scalar_type(), "softmax_lastdim_int8_masking_cuda", [&] {
        softmax_lastdim_int8_masking_kernel<scalar_t>
            <<<grid, block, shared_mem_size, stream>>>(
                xq.data_ptr<int8_t>(),
                y_q.data_ptr<int8_t>(),
                sx.data_ptr<scalar_t>(),
                sy.data_ptr<scalar_t>(),
                masking.data_ptr<uint8_t>(),
                num_vecs,
                dim2,
                broadcast_first_dim);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y_q;
}
