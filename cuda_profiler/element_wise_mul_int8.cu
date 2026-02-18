#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "/sciclone/home/tnguyen10/Desktop/GPU_learn/helper.cpp"

__global__ void element_wise_mul_int8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    const float scale_A,
    const float scale_B,
    const float scale_C,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        int32_t prod = static_cast<int32_t>(A[i]) * static_cast<int32_t>(B[i]);
        float c_float = static_cast<float>(prod) * (scale_A * scale_B / scale_C);
        int tmp = max(-128, min(127, __float2int_rn(c_float)));
        C[i] = static_cast<int8_t>(tmp);
    }
}

void element_wise_mul_int8_host(
    const int8_t* A,
    const float scale_A,
    const int8_t* B,
    const float scale_B,
    const float scale_C,
    int8_t* C,
    int N
) {
    // init device memory
    int8_t *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int8_t);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < 5; i++) {
        element_wise_mul_int8_kernel<<<numBlocks, blockSize>>>(
        d_A,
        d_B,
        d_C,
        scale_A,
        scale_B,
        scale_C,
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
    size_t size = N * sizeof(int8_t);

    // Allocate host memory
    int8_t *h_A = (int8_t*)malloc(size);
    int8_t *h_B = (int8_t*)malloc(size);
    int8_t *h_C = (int8_t*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<int8_t>(rand() % 256 - 128);
        h_B[i] = static_cast<int8_t>(rand() % 256 - 128);
    }

    float scale_A = 0.1f;
    float scale_B = 0.2f;
    float scale_C = 0.05f;

    // Perform element-wise multiplication
    element_wise_mul_int8_host(h_A, scale_A, h_B, scale_B, scale_C, h_C, N);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}