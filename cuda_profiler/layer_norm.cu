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

__global__ void LayerNorm_kernel(
    const float* __restrict__ x,
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
        local_sum += x[i];
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
        float diff = x[i] - mean;
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
        float norm_val = (x[i] - mean) * inv_std;
        y[i] = norm_val * gamma + beta;
    }
}

void LayerNorm_host(
    const float* x,
    float* y,
    const float gamma,
    const float beta,
    int64_t n,
    float eps)
{
    float *d_x, *d_y;
    size_t size = n * sizeof(float);

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int threads = BLOCK;
    int shared_mem_size = (threads / 32) * sizeof(float);  // for warp partials

    LayerNorm_kernel<<<1, threads, shared_mem_size>>>(
        d_x, d_y, gamma, beta, n, eps);

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

// This function compute LayerNorm for 2D input (rows x n)
__global__ void LayerNorm_2D_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int64_t n,        // number of features per row (d)
    int64_t rows,     // number of rows (L)
    float eps)
{
    // shared memory for per-warp partials
    extern __shared__ float shared_mem[];

    int tid      = threadIdx.x;
    int lane     = tid & 31;        // lane id in warp
    int warp     = tid >> 5;        // warp id in block
    int stride   = blockDim.x;
    int num_warps = (blockDim.x + 31) >> 5;

    // Each block handles one row
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    // Pointer to the start of this row
    const float* x_row = x + row * n;
    float*       y_row = y + row * n;

    // -----------------------------
    // 1) Block mean via warp sum
    // -----------------------------
    float local_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        local_sum += x_row[i];
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

    float mean = shared_mem[0] / static_cast<float>(n);

    // -----------------------------
    // 2) Block variance via warp sum
    // -----------------------------
    float local_var_sum = 0.0f;
    for (int64_t i = tid; i < n; i += stride) {
        float diff = x_row[i] - mean;
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

    float var     = shared_mem[0] / static_cast<float>(n);
    float inv_std = rsqrtf(var + eps);

    // -----------------------------
    // 3) Normalize and write output
    // -----------------------------
    for (int64_t i = tid; i < n; i += stride) {
        float norm_val = (x_row[i] - mean) * inv_std;
        y_row[i] = norm_val * gamma[i] + beta[i];
    }
}

void LayerNorm_host_2D(
    const float* x,
    float* y,
    const float* gamma,
    const float* beta,
    int64_t rows,   // L
    int64_t n,      // d
    float eps)
{
    float *d_x, *d_y;
    size_t size = static_cast<size_t>(rows) * static_cast<size_t>(n) * sizeof(float);
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    float *d_gamma, *d_beta;
    size_t param_size = n * sizeof(float);
    cudaMalloc((void**)&d_gamma, param_size);
    cudaMalloc((void**)&d_beta, param_size);
    cudaMemcpy(d_gamma, gamma, param_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, param_size, cudaMemcpyHostToDevice);

    int threads = BLOCK;  // same as before (must be multiple of 32 ideally)
    int num_warps = (threads + 31) / 32;
    int shared_mem_size = num_warps * sizeof(float);  // for warp partials

    dim3 grid(rows);     // one block per row
    dim3 block(threads);

    LayerNorm_2D_kernel<<<grid, block, shared_mem_size>>>(
        d_x, d_y, d_gamma, d_beta, n, rows, eps);

    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}


void init_1d_vector(float* x, int64_t n, float value) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] = value;
    }
}

int main()
{
    int seq_length = 2048;
    int embed_dim = 2048;

    float* gamma = new float[embed_dim];
    float* beta  = new float[embed_dim];
    for (int i = 0; i < embed_dim; ++i) {
        gamma[i] = 1.0f;
        beta[i]  = 0.0f;
    }
    
    float eps = 1e-5;
    
    float *h_x, *h_y;
    h_x = new float[seq_length * embed_dim];
    h_y = new float[seq_length * embed_dim];

    init_1d_vector(h_x, seq_length * embed_dim, 5.0f);
    LayerNorm_host_2D(h_x, h_y, gamma, beta, seq_length, embed_dim, eps);
    
    delete[] h_x;
    delete[] h_y;
    delete[] gamma;
    delete[] beta;
    return 0;
}

