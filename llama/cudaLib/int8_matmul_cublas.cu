#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>

void int8_gemm_lt(int m, int n, int k,
                  const int8_t* A, const int8_t* B, int32_t* C) {
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    int lda = k;
    int ldb = n;
    int ldc = n;

    int32_t alpha = 1;
    int32_t beta = 0;

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    // Create operation descriptor
    cublasLtMatmulDescCreate(&operationDesc,
                            CUBLAS_COMPUTE_32I,
                            CUDA_R_32I);

    cublasLtMatmulDescSetAttribute(operationDesc,
                                   CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc,
                                   CUBLASLT_MATMUL_DESC_TRANSB,
                                   &transb, sizeof(transb));

    // Matrix layouts
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc);

    // Execute
    cublasLtMatmul(ltHandle,
                   operationDesc,
                   &alpha,
                   A, Adesc,
                   B, Bdesc,
                   &beta,
                   C, Cdesc,
                   C, Cdesc,
                   nullptr,
                   nullptr, 0,
                   0);

    // Cleanup
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int m = 1024;
    int n = 1024;
    int k = 1024;

    // Input value of A on host
    int8_t* h_A = new int8_t[m * k];
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = 1; // Example values
    }

    int8_t* h_B = new int8_t[k * n];
    for (int i = 0; i < k * n; ++i) {
        h_B[i] = 1; // Example values
    }

    int32_t* h_C = new int32_t[m * n];
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = 0; // Initialize to zero
    }

    int8_t* d_A;
    int8_t* d_B;
    int32_t* d_C;

    checkCudaError(cudaMalloc(&d_A, m * k * sizeof(int8_t)), "Allocating A");
    checkCudaError(cudaMalloc(&d_B, k * n * sizeof(int8_t)), "Allocating B");
    checkCudaError(cudaMalloc(&d_C, m * n * sizeof(int32_t)), "Allocating C");

    checkCudaError(cudaMemcpy(d_A, h_A, m * k * sizeof(int8_t), cudaMemcpyHostToDevice), "Copying A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, k * n * sizeof(int8_t), cudaMemcpyHostToDevice), "Copying B to device");
    checkCudaError(cudaMemcpy(d_C, h_C, m * n * sizeof(int32_t), cudaMemcpyHostToDevice), "Copying C to device");

    int8_gemm_lt(m, n, k, d_A, d_B, d_C);

    // Print some values from C to verify correctness
    checkCudaError(cudaMemcpy(h_C, d_C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost), "Copying C to host");
    std::cout << "C[0]: " << h_C[0] << std::endl;
    std::cout << "C[1]: " << h_C[1] << std::endl;

    delete[] h_C;
    delete[] h_B;
    delete[] h_A;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}