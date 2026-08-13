#pragma once
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

template <typename scalar_t>
struct alignas(16) Vec8 {
    scalar_t x[8];
};

struct alignas(8) Int8Vec8 {
    int8_t x[8];
};

__device__ __forceinline__ int8_t quantize_int8(float value) {
    int q = __float2int_rn(value);
    q = max(-127, min(127, q));
    return static_cast<int8_t>(q);
}


inline __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return v;
}

// inline __device__ float block_reduce_max(float v) {
//     __shared__ float smem[32];
//     int lane = threadIdx.x & 31; // index within the warp
//     int warp = threadIdx.x >> 5; // warp index within the block

//     v = warp_reduce_max(v);

//     if (lane == 0) smem[warp] = v;
//     __syncthreads();

//     // Only the first warp will read the results from shared memory
//     float out = 0.0f;
//     if (warp == 0) {
//         int nw = (blockDim.x + 31) >> 5;  // number of warps in the block
//         out = (lane < nw) ? smem[lane] : 0.0f;
//         out = warp_reduce_max(out);
//     }
//     __syncthreads();

//     if (threadIdx.x == 0) smem[0] = out;
//     __syncthreads();
//     return smem[0];
// }


__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float smem[32];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    v = warp_reduce_max(v);

    if (lane == 0) {
        smem[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        float out = (lane < num_warps) ? smem[lane] : -FLT_MAX;
        out = warp_reduce_max(out);

        if (lane == 0) {
            smem[0] = out;
        }
    }
    __syncthreads();

    return smem[0];
}


/*
The function block_reduce_max_neg_inf is similar to block_reduce_max,
but initializes the reduction with -FLT_MAX instead of 0. 
This is important for max reductions to ensure that any valid value will be greater than the initial value.
*/
inline __device__ float block_reduce_max_neg_inf(float v) {
    __shared__ float smem[32];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_max(v);

    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = -FLT_MAX;

    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        out = (lane < nw) ? smem[lane] : -FLT_MAX;
        out = warp_reduce_max(out);
    }

    if (threadIdx.x == 0) smem[0] = out;
    __syncthreads();

    return smem[0];
}


inline __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

inline __device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // up to 1024 threads = 32 warps

    const int lane = threadIdx.x & 31;   // index within warp
    const int warp = threadIdx.x >> 5;   // warp index within block
    const int num_warps = (blockDim.x + 31) >> 5;

    // Reduce inside each warp
    val = warp_reduce_sum(val);

    // One thread per warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    // First warp reduces the warp sums
    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }

    return block_sum;
}

__inline__ __device__ int32_t warp_reduce_sum_int32(int32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ int32_t block_reduce_sum_int32(int32_t val) {
    static __shared__ int32_t shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum_int32(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;

    val = 0;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0;
        val = warp_reduce_sum_int32(val);
    }

    return val;
}


__inline__ __device__ float warp_reduce_max_float(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__inline__ __device__ float block_reduce_max_float(float val) {
    static __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max_float(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;

    val = 0.0f;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_max_float(val);
    }

    return val;
}
