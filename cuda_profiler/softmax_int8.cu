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

__global__ void softmax_1d_int8_kernel(
    const int8_t* __restrict__ x_q, 
    int8_t* __restrict__ y,
    float scale_x, float scale_y, int n)
{
    extern __shared__ unsigned char smem[]; // dynamic
    int8_t* xq_s = reinterpret_cast<int8_t*>(smem);

    int tid = threadIdx.x;

    // 0) Cache x_q in shared (global read ONCE)
    for (int i = tid; i < n; i += BLOCK) {
        xq_s[i] = x_q[i];
    }
    __syncthreads();

    // 1) max reduction (compute from shared)
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += BLOCK) {
        float v = scale_x * (float)xq_s[i];  // dequantize back from int8 to float
        local_max = fmaxf(local_max, v);
    }

    // reduce within warp
    float warp_m = warp_max(local_max);

    // reduce across warps
    __shared__ float warp_buf[32]; // enough for up to 1024 threads
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_buf[warp] = warp_m;
    __syncthreads();

    float max_val = -FLT_MAX;
    if (warp == 0) {
        max_val = (lane < (BLOCK + 31) / 32) ? warp_buf[lane] : -FLT_MAX;
        max_val = warp_max(max_val);
        if (lane == 0) warp_buf[0] = max_val;
    }
    __syncthreads();
    max_val = warp_buf[0];

    // 2) sum reduction
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += BLOCK) {
        float v = scale_x * (float)xq_s[i];
        local_sum += __expf(v - max_val);
    }

    float warp_s = warp_sum(local_sum);
    if (lane == 0) warp_buf[warp] = warp_s;
    __syncthreads();

    float sum_val = 0.0f;
    if (warp == 0) {
        sum_val = (lane < (BLOCK + 31) / 32) ? warp_buf[lane] : 0.0f;
        sum_val = warp_sum(sum_val);
        if (lane == 0) warp_buf[0] = sum_val;
    }
    __syncthreads();
    sum_val = warp_buf[0];

    // 3) write output
    for (int i = tid; i < n; i += BLOCK) {
        float v = scale_x * (float)xq_s[i];
        float ex = __expf(v - max_val) / sum_val;
        y[i] = (int8_t)(ex / scale_y);
    }
}

void softmax_1d_int8_host(const int8_t* x_q, int8_t* y, float scale_x, 
    float scale_y, int n)
{
   int8_t *d_xq;
   int8_t *d_y;
   size_t size = n * sizeof(int8_t);
   size_t size_f = n * sizeof(int8_t);

    cudaMalloc(&d_xq, size);
    cudaMalloc(&d_y, size_f);

    cudaMemcpy(d_xq, x_q, size, cudaMemcpyHostToDevice);

    int threads = BLOCK;
    int shared_mem_size = n * sizeof(int8_t);

    for (int i = 0; i < 5; ++i) {
        softmax_1d_int8_kernel<<<1, threads, shared_mem_size>>>(d_xq, d_y, scale_x, scale_y, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(y, d_y, size_f, cudaMemcpyDeviceToHost);
    cudaFree(d_xq);
    cudaFree(d_y);
}

// x_q: [dim0, dim1, dim2] int8, contiguous row-major
// y_q : [dim0, dim1, dim2] int8, contiguous row-major
// scale_x: length dim0*dim1
// scale_y: scalar
// softmax along dim2
__global__ void softmax_lastdim_3d_int8_kernel(
    const int8_t* __restrict__ x_q,
    int8_t* __restrict__ y_q,
    const float* __restrict__ scale_x,  // length dim0*dim1
    float scale_y,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2)
{
    int64_t vec = (int64_t)blockIdx.x;   // 0 .. dim0*dim1-1
    int64_t num_vecs = dim0 * dim1;
    if (vec >= num_vecs) return;

    float sx = scale_x[vec];            // per-row scale

    int tid = threadIdx.x;
    int stride = blockDim.x;

    extern __shared__ unsigned char smem[];
    int8_t* xq_s = reinterpret_cast<int8_t*>(smem);

    int64_t base = vec * dim2;
    const int8_t* xq = x_q + base;
    int8_t* yq       = y_q + base;

    for (int64_t j = tid; j < dim2; j += stride) xq_s[j] = xq[j];
    __syncthreads();

    __shared__ float warp_buf[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = (stride + 31) >> 5;

    // 1) max
    float local_max = -FLT_MAX;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        local_max = fmaxf(local_max, v);
    }
    float warp_m = warp_max(local_max);
    if (lane == 0) warp_buf[warp] = warp_m;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float max_val = warp_buf[0];

    // 2) sum
    float local_sum = 0.0f;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        local_sum += __expf(v - max_val);
    }
    float warp_s = warp_sum(local_sum);
    if (lane == 0) warp_buf[warp] = warp_s;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < num_warps) ? warp_buf[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) warp_buf[0] = v;
    }
    __syncthreads();
    float sum_val = warp_buf[0];

    // 3) write (quantize)
    float inv_sum = 1.0f / sum_val;
    for (int64_t j = tid; j < dim2; j += stride) {
        float v = sx * (float)xq_s[j];
        float p = __expf(v - max_val) * inv_sum;   // [0,1] ideally
        int q = __float2int_rn(p / scale_y);
        q = max(0, min(127, q));                   // softmax is non-negative
        yq[j] = (int8_t)q;
    }
}

void softmax_lastdim_3d_int8_host(
    const int8_t* x_q, 
    int8_t* y_q,
    const float* h_scale_x,  // length dim0*dim1
    float scale_y,
    int64_t dim0, int64_t dim1, int64_t dim2)
{
    int8_t *d_xq=nullptr, *d_yq=nullptr;
    float *d_scale_x=nullptr;

    size_t n_elem = (size_t)(dim0 * dim1 * dim2);
    size_t size_q = n_elem * sizeof(int8_t);
    size_t size_s = (size_t)(dim0 * dim1) * sizeof(float);

    cudaMalloc(&d_xq, size_q);
    cudaMalloc(&d_yq, size_q);
    cudaMalloc(&d_scale_x, size_s);

    cudaMemcpy(d_xq, x_q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_x, h_scale_x, size_s, cudaMemcpyHostToDevice);

    int threads = BLOCK;
    int blocks  = (int)(dim0 * dim1);
    size_t shared_mem_size = (size_t)dim2 * sizeof(int8_t);

    for (int i = 0; i < 5; ++i) {
        softmax_lastdim_3d_int8_kernel<<<blocks, threads, shared_mem_size>>>(
            d_xq, d_yq, d_scale_x, scale_y, dim0, dim1, dim2);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(y_q, d_yq, size_q, cudaMemcpyDeviceToHost);

    cudaFree(d_scale_x);
    cudaFree(d_xq);
    cudaFree(d_yq);
}



int main() {
    const int64_t n = 40 * 4096 * 4096;

    int8_t* h_xq = new int8_t[n];
    int8_t* h_yq  = new int8_t[n];
    float* h_scale_x = new float[40 * 4096];

    init_1d_vector_int8(h_xq, n, 5);
    init_1d_vector_float(h_scale_x, 40 * 4096, 0.1f);
    
    float scale_y = 0.007874f; // 1/127

    softmax_lastdim_3d_int8_host(h_xq, h_yq, h_scale_x, scale_y, 40, 4096, 4096);

    delete[] h_xq;
    delete[] h_yq;
    delete[] h_scale_x;

    return 0;
}

