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

inline __device__ float warp_reduce_max(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return v;
}

inline __device__ float block_reduce_max(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    v = warp_reduce_max(v);

    if (lane == 0) smem[warp] = v;
    __syncthreads();

    float out = 0.0f;
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        out = (lane < nw) ? smem[lane] : 0.0f;
        out = warp_reduce_max(out);
    }
    __syncthreads();

    if (threadIdx.x == 0) smem[0] = out;
    __syncthreads();
    return smem[0];
}