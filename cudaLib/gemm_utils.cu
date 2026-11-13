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


template <int THREADS>
__device__ __forceinline__
float block_reduce_max_fast(float v) {

    static_assert(THREADS >= 32 &&
                  THREADS <= 1024 &&
                  THREADS % 32 == 0);

    constexpr int NUM_WARPS = THREADS / 32;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_max(v);

    if constexpr (NUM_WARPS == 1) {
        return __shfl_sync(0xffffffffu, v, 0);
    }
    else {
        __shared__ float smem[NUM_WARPS];

        if (lane == 0)
            smem[warp] = v;

        __syncthreads();

        if (warp == 0) {
            float out = (lane < NUM_WARPS)
                      ? smem[lane]
                      : -FLT_MAX;

            out = warp_reduce_max(out);

            if (lane == 0)
                smem[0] = out;
        }

        __syncthreads();

        return smem[0];
    }
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

struct SumMax {
    float sum;
    float amax;
};

template <int THREADS> 
__device__ __forceinline__ SumMax block_reduce_sum_max(SumMax v) {
    static_assert(THREADS >= 32 &&
                  THREADS <= 1024 &&
                  THREADS % 32 == 0);

    constexpr int NUM_WARPS = THREADS / 32;
    constexpr unsigned MASK = 0xffffffffu;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Step 1: Reduce both statistics within each WARP
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.sum += __shfl_down_sync(MASK, v.sum, offset);
        v.amax = fmaxf(v.amax,__shfl_down_sync(MASK, v.amax, offset));
    }

    // Fast path: one warp requires no shared memory
    if constexpr (NUM_WARPS == 1) {
        v.sum  = __shfl_sync(MASK, v.sum, 0);
        v.amax = __shfl_sync(MASK, v.amax, 0);
        return v;
    } else {
        __shared__ SumMax partial[NUM_WARPS];

        // Step 2: Each warp stores its partial results
        if (lane == 0) {
            partial[warp] = v;
        }
        __syncthreads();

        // Step 3: Warp 0 reduces the warp results
        if (warp == 0) {
            SumMax total;
            total.sum = (lane < NUM_WARPS) ? partial[lane].sum : 0.0f;

            total.amax = (lane < NUM_WARPS) ? partial[lane].amax : 0.0f;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total.sum += __shfl_down_sync(MASK, total.sum, offset);
                total.amax = fmaxf(total.amax, __shfl_down_sync( MASK, total.amax, offset));
            }

            // Assign the final result to shared memory at index 0
            if (lane == 0) {
                partial[0] = total;
            }
        }
        __syncthreads();
        return partial[0];
    }
}



