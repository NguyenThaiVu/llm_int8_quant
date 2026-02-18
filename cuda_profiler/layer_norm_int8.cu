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

__global__ void LayerNorm_int8_kernel(
    const int8_t* __restrict__ x,
    float scale_x,
    float* __restrict__ y,
    const float gamma,
    const float beta,
    int64_t n,
    float eps)
{
    // shared memory for per-warp partials (max or sum)
    extern __shared__ float shared_mem[];  // size >= num_warps * 2

    int tid   = threadIdx.x;
    int lane  = tid & 31;          // lane id in warp
    int warp  = tid >> 5;          // warp id in block
    int stride = blockDim.x;

    int num_warps = (blockDim.x + 31) >> 5;

    // -----------------------------
    // 1) Block mean via warp sum
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        local_sum += static_cast<float>(x[i]) * scale_x;
    }

    // Reduce within warp
    float wsum = warp_sum(local_sum);

    // Write warp result
    if (lane == 0) shared_mem[warp] = wsum;
    __syncthreads();

    // First warp reduces warp sums
    float block_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        v = warp_sum(v);
        block_sum = v;
        shared_mem[0] = block_sum;  // store for later
    }
    __syncthreads();

    float mean = shared_mem[0] / n;

    // -----------------------------
    // 2) Block variance via warp sum
    // -----------------------------
    float local_var_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        float diff = static_cast<float>(x[i]) * scale_x - mean;
        local_var_sum += diff * diff;
    }

    // Reduce within warp
    float wvar_sum = warp_sum(local_var_sum);

    // Write warp result
    if (lane == 0) shared_mem[warp] = wvar_sum;
    __syncthreads();

    // First warp reduces warp sums
    float block_var_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        v = warp_sum(v);
        block_var_sum = v;
        shared_mem[0] = block_var_sum;  // store for later
    }
    __syncthreads();    

    float var = shared_mem[0] / n;
    float inv_std = rsqrtf(var + eps);

    // -----------------------------
    // 3) Normalize and write output
    // -----------------------------
    for (int64_t i = tid; i < n; i += stride) {
        float norm_val = (static_cast<float>(x[i]) * scale_x - mean) * inv_std;
        y[i] = norm_val * gamma + beta;
    }
}

void LayerNorm_int8_host(
    const int8_t* x,
    float scale_x,
    float* y,
    const float gamma,
    const float beta,
    int64_t n,
    float eps)
{
    int8_t *d_x;
    float *d_y;
    size_t size = n * sizeof(int8_t);

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = BLOCK;
    int shared_mem_size = (threads / 32) * sizeof(float) * 2;  // for warp partials (max and sum)

    LayerNorm_int8_kernel<<<1, threads, shared_mem_size>>>(
        d_x, scale_x, d_y, gamma, beta, n, eps);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_x);
    cudaFree(d_y);
}

__global__ void LayerNorm_int8_2D_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ scale_x,
    float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int64_t n,
    int64_t rows,
    float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;  // safety

    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int warp  = tid >> 5;
    int stride = blockDim.x;
    int num_warps = (blockDim.x + 31) >> 5;

    // Single dynamic shared memory buffer
    extern __shared__ unsigned char smem[];

    // Layout:
    // [0 .. num_warps*sizeof(float)-1]  -> float warp partials
    // [num_warps*sizeof(float) .. ]     -> int8_t x_shared[n]

    float* shared_mem = reinterpret_cast<float*>(smem);  // size >= num_warps
    int8_t* x_shared = reinterpret_cast<int8_t*>(
        smem + num_warps * sizeof(float)
    );  // size >= n

    const int8_t* x_row = x + row * n;
    float* y_row = y + row * n;
    float scale_val = scale_x[row];

    // -----------------------------
    // 0) Copy x_row into shared
    // -----------------------------
    for (int64_t i = tid; i < n; i += stride) {
        x_shared[i] = x_row[i];
    }
    __syncthreads();

    // -----------------------------
    // 1) Mean
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        local_sum += static_cast<float>(x_shared[i]) * scale_val;
    }

    float wsum = warp_sum(local_sum);
    if (lane == 0) shared_mem[warp] = wsum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        v = warp_sum(v);
        block_sum = v;
        shared_mem[0] = block_sum;
    }
    __syncthreads();

    float mean = shared_mem[0] / static_cast<float>(n);

    // -----------------------------
    // 2) Variance
    // -----------------------------
    float local_var_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        float diff = static_cast<float>(x_shared[i]) * scale_val - mean;
        local_var_sum += diff * diff;
    }

    float wvar_sum = warp_sum(local_var_sum);
    if (lane == 0) shared_mem[warp] = wvar_sum;
    __syncthreads();

    float block_var_sum = 0.0f;
    if (warp == 0) {
        float v = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        v = warp_sum(v);
        block_var_sum = v;
        shared_mem[0] = block_var_sum;
    }
    __syncthreads();

    float var = shared_mem[0] / static_cast<float>(n);
    float inv_std = rsqrtf(var + eps);

    // -----------------------------
    // 3) Normalize + write
    // -----------------------------
    for (int64_t i = tid; i < n; i += stride) {
        float norm_val =
            (static_cast<float>(x_shared[i]) * scale_val - mean) * inv_std;
        y_row[i] = norm_val * gamma[i] + beta[i];
    }
}


void LayerNorm_int8_2D_host(
    const int8_t* x,
    float* scale_x,
    float* y,
    const float* gamma,
    const float* beta,
    int64_t n,   // number of feature dimensions (d)
    int64_t rows,  // number of tokens (L)
    float eps)
{
    int8_t *d_x;
    float *d_scale_x;
    float *d_y;
    float *d_gamma;
    float *d_beta;

    cudaMalloc((void**)&d_x, n * rows * sizeof(int8_t));
    cudaMalloc((void**)&d_scale_x, rows * sizeof(float));
    cudaMalloc((void**)&d_y, n * rows * sizeof(float));
    cudaMalloc((void**)&d_gamma, n * sizeof(float));
    cudaMalloc((void**)&d_beta, n * sizeof(float));

    cudaMemcpy(d_x, x, n * rows * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_x, scale_x, rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = BLOCK;
    // int shared_mem_size = (threads / 32) * sizeof(float) * 2;  // for warp partials (max and sum)
    int num_warps = (threads + 31) / 32;
    int shared_mem_size = num_warps * sizeof(float) + n * sizeof(int8_t);  // for warp partials + x_shared

    LayerNorm_int8_2D_kernel<<<rows, threads, shared_mem_size>>>(
        d_x, d_scale_x, d_y, d_gamma, d_beta, n, rows, eps);

    cudaMemcpy(y, d_y, n * rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_x);
    cudaFree(d_scale_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}


int main()
{
    int seq_length = 1024 * 16; 
    int embed_dim = 8192;

    float* gamma = new float[embed_dim];
    float* beta  = new float[embed_dim];
    for (int i = 0; i < embed_dim; ++i) {
        gamma[i] = 1.0f;
        beta[i]  = 0.0f;
    }

    int8_t* h_x = new int8_t[seq_length * embed_dim];
    float *scale_x = new float[seq_length]; // Each token have different scale
    for (int i = 0; i < seq_length; ++i) {
        scale_x[i] = 1.0f;
    }
    float* h_y = new float[seq_length * embed_dim];

    init_1d_vector_int8(h_x, seq_length * embed_dim, 2);  // all elements are 2

    LayerNorm_int8_2D_host(h_x, scale_x, h_y, gamma, beta,
                    embed_dim, seq_length, 1e-5);
    
    delete[] h_x;
    delete[] h_y;
    delete[] scale_x;
    delete[] gamma;
    delete[] beta;
    return 0;
}