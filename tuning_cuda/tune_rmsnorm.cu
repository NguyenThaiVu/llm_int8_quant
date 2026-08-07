#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================================
// Cooperative Warp/Block Reduction Utilities
// ============================================================================
template <typename T>
__device__ inline T block_reduce_sum(T val) {
    __shared__ T shared[32]; // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction over warp results
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : static_cast<T>(0);
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template <typename T>
__device__ inline T block_reduce_max(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Final reduction over warp results
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -1e20f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

// ============================================================================
// Your RMSNorm INT8 Kernel
// ============================================================================
template <typename T>
__global__ void rmsnorm_int8_kernel(
    const int8_t* __restrict__ x,         // [num_rows, d_model]
    const float* __restrict__ scale_x,    // shape [num_rows]
    const T* __restrict__ gamma,          // RMSNorm weight: shape [d_model]
    int8_t* __restrict__ y,               // Shape [num_rows, d_model]
    float* __restrict__ scale_y,          // shape [num_rows]
    int d_model,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const int8_t* row_x = x + row * d_model;
    int8_t* row_y = y + row * d_model;

    const float x_scale = scale_x[row];

    // Pass 1a: Compute sum of squares 
    float local_sum_sq = 0.0f;
    for (int col = tid; col < d_model; col += blockDim.x) {
        float x_real = x_scale * static_cast<float>(row_x[col]);
        local_sum_sq += x_real * x_real;
    }

    float block_sum_sq = block_reduce_sum(local_sum_sq);

    __shared__ float shared_block_sum_sq;
    if (tid == 0) {
        shared_block_sum_sq = block_sum_sq;
    }
    __syncthreads();

    float mean_sq = shared_block_sum_sq / static_cast<float>(d_model);
    float rms_inv = rsqrtf(mean_sq + eps);

    // Pass 1b: Compute max absolute floating-point output
    float local_max_abs = 0.0f;
    const float norm_scale = x_scale * rms_inv;

    for (int col = tid; col < d_model; col += blockDim.x) {
        float gamma_val = static_cast<float>(gamma[col]);
        float y_fp = static_cast<float>(row_x[col]) * norm_scale * gamma_val;
        float a = fabsf(y_fp);
        local_max_abs = fmaxf(local_max_abs, a);
    }

    float block_amax = block_reduce_max(local_max_abs);

    __shared__ float shared_scale_y;
    if (tid == 0) {
        float s = block_amax / 127.0f;
        if (s == 0.0f) {
            s = 1.0f;
        }
        shared_scale_y = s;
        scale_y[row] = s;
    }
    __syncthreads();

    const float inv_scale_y = 1.0f / shared_scale_y;

    // Pass 2: Quantize output
    for (int col = tid; col < d_model; col += blockDim.x) {
        float gamma_val = static_cast<float>(gamma[col]);
        float y_fp = static_cast<float>(row_x[col]) * norm_scale * gamma_val;

        int q = __float2int_rn(y_fp * inv_scale_y);
        q = max(-127, min(127, q));

        row_y[col] = static_cast<int8_t>(q);
    }
}

// ============================================================================
// Benchmarking Orchestrator
// ============================================================================
int main() {
    // LLM Activation Tensors Dimensions (e.g., Llama/Mistral layers)
    const int n_rows = 8192; 
    const int d_model = 1024 * 6; 
    const float eps = 1e-5f;

    size_t x_size = n_rows * d_model * sizeof(int8_t);
    size_t scale_size = n_rows * sizeof(float);
    size_t gamma_size = d_model * sizeof(float);

    // Allocate Host Memory
    std::vector<int8_t> h_x(n_rows * d_model, 12);
    std::vector<float> h_scale_x(n_rows, 0.05f);
    std::vector<float> h_gamma(d_model, 1.0f);

    // Allocate Device Memory
    int8_t *d_x, *d_y;
    float *d_scale_x, *d_scale_y, *d_gamma;

    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_y, x_size);
    cudaMalloc(&d_scale_x, scale_size);
    cudaMalloc(&d_scale_y, scale_size);
    cudaMalloc(&d_gamma, gamma_size);

    // Copy vectors to GPU
    cudaMemcpy(d_x, h_x.data(), x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_x, h_scale_x.data(), scale_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), gamma_size, cudaMemcpyHostToDevice);

    // Configurations to test
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    
    int best_block_size = 0;
    float min_latency = 1e20f;

    std::cout << "========================================\n";
    std::cout << "  Empirical CUDA Auto-Tuning Summary    \n";
    std::cout << "  Matrix Shape: [" << n_rows << " x " << d_model << "]\n";
    std::cout << "========================================\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int threads : block_sizes) {
        dim3 block(threads);
        dim3 grid(n_rows);

        // 1. Warm-up to skip driver overheads
        for (int i = 0; i < 5; ++i) {
            rmsnorm_int8_kernel<float><<<grid, block>>>(d_x, d_scale_x, d_gamma, d_y, d_scale_y, d_model, eps);
        }
        cudaDeviceSynchronize();

        // 2. Exact Timing Iterations
        const int iterations = 50;
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            rmsnorm_int8_kernel<float><<<grid, block>>>(d_x, d_scale_x, d_gamma, d_y, d_scale_y, d_model, eps);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_latency = milliseconds / iterations;

        std::cout << "Threads per Block: " << threads << " \t-> Avg Latency: " << avg_latency << " ms\n";

        if (avg_latency < min_latency) {
            min_latency = avg_latency;
            best_block_size = threads;
        }
    }

    std::cout << "----------------------------------------\n";
    std::cout << "Optimal Selection: THREADS_PER_BLOCK = " << best_block_size << " (" << min_latency << " ms)\n";
    std::cout << "========================================\n";

    // Resource Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_scale_x); cudaFree(d_scale_y);
    cudaFree(d_gamma);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}