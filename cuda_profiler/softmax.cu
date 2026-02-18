#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "/sciclone/home/tnguyen10/Desktop/GPU_learn/helper.cpp"


# define BLOCK 512

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

__global__ void softmax_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    // shared memory for per-warp partials (max or sum)
    extern __shared__ float warp_partials[];  // size >= num_warps

    int tid   = threadIdx.x;
    int lane  = tid & 31;          // lane id in warp
    int warp  = tid >> 5;          // warp id in block
    int stride = blockDim.x;

    int num_warps = (blockDim.x + 31) >> 5;

    // -----------------------------
    // 1) Block max via warp max
    // -----------------------------
    float local_max = -FLT_MAX;
    for (int64_t i = tid; i < n; i += stride) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Reduce within warp
    float wmax = warp_max(local_max);

    // Write warp result
    if (lane == 0) warp_partials[warp] = wmax;
    __syncthreads();

    // First warp reduces warp maxima
    float block_max = -FLT_MAX;
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    block_max = warp_partials[0];

    // -----------------------------
    // 2) Block sum exp(x - max)
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        local_sum += expf(x[i] - block_max);
    }

    float wsum = warp_sum(local_sum);
    if (lane == 0) warp_partials[warp] = wsum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    block_sum = warp_partials[0];

    // -----------------------------
    // 3) Write normalized output
    // -----------------------------
    // Optional: guard against divide-by-zero if n==0 (usually n>0)
    float inv_sum = 1.0f / block_sum;
    for (int64_t i = tid; i < n; i += stride) {
        float ex = expf(x[i] - block_max) * inv_sum;
        y[i] = ex;
    }
}


void softmax_1d_host(const float* x, float* y, int64_t n)
{
   float *d_x;
   float *d_y;
   size_t size = n * sizeof(float);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int threads = BLOCK;
    // int shared_mem_size = n * sizeof(float);
    int shared_mem_size = (threads / 32) * sizeof(float); // per-warp partials

    for (int i = 0; i < 10; ++i) {
        softmax_1d_kernel<<<1, threads, shared_mem_size>>>(d_x, d_y, n);
    }

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void softmax_lastdim_2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t rows,
    int64_t cols)
{
    // one block computes one row
    int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float warp_partials[];  // size >= num_warps

    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int warp  = tid >> 5;
    int stride = blockDim.x;

    int num_warps = (blockDim.x + 31) >> 5;

    const float* xrow = x + row * cols;
    float* yrow       = y + row * cols;

    // -----------------------------
    // 1) Row max
    // -----------------------------
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < cols; j += stride) {
        local_max = fmaxf(local_max, xrow[j]);
    }

    float wmax = warp_max(local_max);
    if (lane == 0) warp_partials[warp] = wmax;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    float row_max = warp_partials[0];

    // -----------------------------
    // 2) Row sum exp(x - max)
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t j = tid; j < cols; j += stride) {
        local_sum += expf(xrow[j] - row_max);
    }

    float wsum = warp_sum(local_sum);
    if (lane == 0) warp_partials[warp] = wsum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    float row_sum = warp_partials[0];

    // -----------------------------
    // 3) Write normalized output
    // -----------------------------
    float inv_sum = 1.0f / row_sum;
    for (int64_t j = tid; j < cols; j += stride) {
        float ex = expf(xrow[j] - row_max) * inv_sum;
        yrow[j] = ex;
    }
}

// Host function to launch softmax on last dimension of 2D array
void softmax_lastdim_2d_host(const float* x, float* y, int64_t rows, int64_t cols)
{
    float *d_x = nullptr, *d_y = nullptr;
    size_t size = (size_t)(rows * cols) * sizeof(float);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int threads = BLOCK;                 
    int blocks  = (int)rows;          // one block per row
    int shared_mem_size = ((threads + 31) / 32) * sizeof(float);

    // warmup / timing loop
    for (int i = 0; i < 10; ++i) {
        softmax_lastdim_2d_kernel<<<blocks, threads, shared_mem_size>>>(d_x, d_y, rows, cols);
    }

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}


__global__ void softmax_lastdim_3d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t dim0,   // A
    int64_t dim1,   // B
    int64_t dim2)   // C (softmax over this)
{
    // Each block handles one (i0, i1) vector of length dim2
    int64_t vec = (int64_t)blockIdx.x;          // 0 .. dim0*dim1-1
    int64_t num_vecs = dim0 * dim1;
    if (vec >= num_vecs) return;

    extern __shared__ float warp_partials[];    // size >= num_warps

    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warp   = tid >> 5;
    int stride = blockDim.x;
    int num_warps = (blockDim.x + 31) >> 5;

    // Map vec -> (i0, i1)
    // i0 = vec / dim1, i1 = vec % dim1
    int64_t base = vec * dim2;

    const float* xvec = x + base;
    float* yvec       = y + base;

    // -----------------------------
    // 1) Vector max
    // -----------------------------
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < dim2; j += stride) {
        local_max = fmaxf(local_max, xvec[j]);
    }

    float wmax = warp_max(local_max);
    if (lane == 0) warp_partials[warp] = wmax;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    float vmax = warp_partials[0];

    // -----------------------------
    // 2) Vector sum exp(x - max)
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        local_sum += expf(xvec[j] - vmax);
    }

    float wsum = warp_sum(local_sum);
    if (lane == 0) warp_partials[warp] = wsum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_partials[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_partials[0] = v;
    }
    __syncthreads();
    float vsum = warp_partials[0];

    // -----------------------------
    // 3) Write normalized output
    // -----------------------------
    float inv = 1.0f / vsum;
    for (int64_t j = tid; j < dim2; j += stride) {
        yvec[j] = expf(xvec[j] - vmax) * inv;
    }
}

void softmax_lastdim_3d_host(const float* x, float* y,
                            int64_t dim0, int64_t dim1, int64_t dim2)
{
    float *d_x = nullptr, *d_y = nullptr;
    size_t size = (size_t)(dim0 * dim1 * dim2) * sizeof(float);

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    int threads = BLOCK; // e.g. 128/256
    int blocks  = (int)(dim0 * dim1); // one block per (dim0, dim1) vector
    int shared_mem_size = ((threads + 31) / 32) * sizeof(float);

    for (int i = 0; i < 5; ++i) {
        softmax_lastdim_3d_kernel<<<blocks, threads, shared_mem_size>>>(
            d_x, d_y, dim0, dim1, dim2);
    }

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

void init_1d_vector(float* x, int64_t n, float value) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] = value;
    }
}


int main() {
    const int64_t n = 40 * 4096 * 4096;
    float h_x[n], h_y[n];
    
    // init_1d_vector(h_x, n, 5.0f);
    // softmax_1d_host(h_x, h_y, n);

    init_1d_vector(h_x, n, 5.0f);
    softmax_lastdim_3d_host(h_x, h_y, 40, 4096, 4096);

    float true_y[n];
    softmax_lastdim_3d_cpu(h_x, true_y, 40, 4096, 4096);
    
    bool correct = allclose(h_y, true_y, n, 1.0f);
    if (correct) {
        printf("passed\n");
    } else {
        printf("===== failed ===== \n");
    }


    // int8_t h_xq[n];
    // init_1d_vector_int8(h_xq, n, 5);
    // float scale = 0.1f;
    // softmax_1d_int8_host(h_xq, h_y, scale, n);

    return 0;
}

