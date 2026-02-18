#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "/sciclone/home/tnguyen10/Desktop/GPU_learn/helper.cpp"

// For convenience, alias float16 type
using float16_t = __half;

__global__ void element_wise_mul_kernel(
    const float16_t* __restrict__ A,
    const float16_t* __restrict__ B,
    float16_t* __restrict__ C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] * B[i];
    }
}

void element_wise_mul_host(
    const float16_t* A,
    const float16_t* B,
    float16_t* C,
    int N
) {
    // init device memory
    float16_t *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float16_t);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < 5; i++) {
        element_wise_mul_kernel<<<numBlocks, blockSize>>>(
        d_A,
        d_B,
        d_C,
        N);
    }
    cudaDeviceSynchronize();

    // copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Check cuda errors
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int N = 8192 * 8192;
    size_t size = N * sizeof(float16_t);

    // Allocate host memory
    float16_t *h_A = (float16_t*)malloc(size);
    float16_t *h_B = (float16_t*)malloc(size);
    float16_t *h_C = (float16_t*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float16_t>(static_cast<float>(rand()) / RAND_MAX);
        h_B[i] = static_cast<float16_t>(static_cast<float>(rand()) / RAND_MAX);
    }

    // Perform element-wise multiplication on GPU
    element_wise_mul_host(h_A, h_B, h_C, N);

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, __half2float(h_C[i]));
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}