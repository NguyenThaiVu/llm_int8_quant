#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "/sciclone/home/tnguyen10/Desktop/GPU_learn/helper.cpp"


# define BLOCK 128

__global__ void compute_rope_params_kernel(
    float* cos_out,
    float* sin_out,
    int head_dim,
    int seq_len,
    float theta_base)
{
    int pos = blockIdx.y * blockDim.y + threadIdx.y;  // position index
    int j   = blockIdx.x * blockDim.x + threadIdx.x;  // index in [0, head_dim/2)

    int half = head_dim / 2;
    if (pos >= seq_len || j >= half) return;

    // This matches:
    // inv_freq[j] = 1.0 / (theta_base ** ((2*j)/head_dim))
    float exponent = (2.0f * j) / float(head_dim);
    float inv_freq = powf(theta_base, -exponent);  // == 1.0f / powf(theta_base, exponent)

    float angle = float(pos) * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int row_base = pos * head_dim;

    // Duplicate across the two halves (as in your PyTorch code)
    cos_out[row_base + j]        = c;
    cos_out[row_base + j + half] = c;

    sin_out[row_base + j]        = s;
    sin_out[row_base + j + half] = s;
}

// Simple host wrapper
void compute_rope_params_host(
    float* cos_dev,
    float* sin_dev,
    int head_dim,
    int seq_len,
    float theta_base)
{
    int half = head_dim / 2;

    dim3 block(16, 16);  
    dim3 grid(
        (half + block.x - 1) / block.x,
        (seq_len + block.y - 1) / block.y
    );

    compute_rope_params_kernel<<<grid, block>>>(
        cos_dev, sin_dev, head_dim, seq_len, theta_base
    );
}

__global__ void apply_rope_int8_kernel(
    const int8_t* __restrict__ x,
    float scale_x,
    const int8_t* __restrict__ cos,
    float scale_cos,
    const int8_t* __restrict__ sin,
    float scale_sin,
    int head_dim,
    int seq_len,
    int8_t* __restrict__ out,
    float scale_out)
{
    int head = blockIdx.z;                     // head index
    int pos  = blockIdx.y;                     // position index
    int j    = blockIdx.x * blockDim.x + threadIdx.x;  // index in [0, head_dim/2)

    int half = head_dim / 2;
    if (pos >= seq_len || j >= half) return;

    // Compute base index of input for this head and position
    int row_base = (head * seq_len + pos) * head_dim;

    // // // Load and dequantize input
    float x1 = static_cast<float>(x[row_base + j]) * scale_x;
    float x2 = static_cast<float>(x[row_base + j + half]) * scale_x;

    // // Load and dequantize cos and sin
    float c = static_cast<float>(cos[pos * head_dim + j]) * scale_cos;
    float s = static_cast<float>(sin[pos * head_dim + j]) * scale_sin;

    // // Apply RoPE
    float out1 = x1 * c - x2 * s;
    float out2 = x1 * s + x2 * c;

    // Quantization output and write back
    out[row_base + j]        = static_cast<int8_t>(fmaxf(fminf(roundf(out1 / scale_out), 127.0f), -128.0f));
    out[row_base + j + half] = static_cast<int8_t>(fmaxf(fminf(roundf(out2 / scale_out), 127.0f), -128.0f));

    // out[row_base + j]        = out1;
    // out[row_base + j + half] = out2;
}


void apply_rope_int8_host(
    const int8_t* __restrict__ x_dev,
    float scale_x,
    const int8_t* __restrict__ cos_dev,
    float scale_cos,
    const int8_t* __restrict__ sin_dev,
    float scale_sin,
    int8_t* __restrict__ out_dev,
    float scale_out,
    int num_heads,
    int seq_len,
    int head_dim)
{
    int half = head_dim / 2;

    int threads = BLOCK;
    dim3 block(threads);
    dim3 grid(
        (half + threads - 1) / threads,  // over dim-half
        seq_len,                         // over positions
        num_heads                        // over heads
    );

    apply_rope_int8_kernel<<<grid, block>>>(
        x_dev, scale_x,
        cos_dev, scale_cos,
        sin_dev, scale_sin,
        head_dim, seq_len,
        out_dev, scale_out
    );
}

int main()
{
    int num_heads = 40;
    int head_dim  = 128;
    int seq_len   = 1024 * 16;
    float theta_base = 10000.0f;
    float scale_x = 1.0f;
    float scale_cos = 1.0f;
    float scale_sin = 1.0f;
    float scale_out = 1.0f;

    // Allocate device memory
    int8_t* x;          // Input in int8
    int8_t* cos;        // RoPE cos parameters in int8
    int8_t* sin;        // RoPE sin parameters in int8
    int8_t* out;        // Output in int8

    x = new int8_t[num_heads * seq_len * head_dim];
    cos = new int8_t[seq_len * head_dim];
    sin = new int8_t[seq_len * head_dim];
    out = new int8_t[num_heads * seq_len * head_dim];

    init_1d_vector_int8(x, num_heads * seq_len * head_dim, 2);
    init_1d_vector_int8(cos, seq_len * head_dim, 3);
    init_1d_vector_int8(sin, seq_len * head_dim, 4);

    // Allocate device memory
    int8_t* x_dev;
    int8_t* cos_dev;
    int8_t* sin_dev;
    int8_t* out_dev;

    cudaMalloc(&x_dev, num_heads * seq_len * head_dim * sizeof(int8_t));
    cudaMalloc(&cos_dev, seq_len * head_dim * sizeof(int8_t));
    cudaMalloc(&sin_dev, seq_len * head_dim * sizeof(int8_t));
    cudaMalloc(&out_dev, num_heads * seq_len * head_dim * sizeof(int8_t));

    // Copy data to device
    cudaMemcpy(x_dev, x, num_heads * seq_len * head_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cos_dev, cos, seq_len * head_dim * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(sin_dev, sin, seq_len * head_dim * sizeof(int8_t), cudaMemcpyHostToDevice);

    int n_iter = 5;
    for (int i = 0; i < n_iter; ++i) {
        apply_rope_int8_host(x_dev, scale_x, 
                            cos_dev, scale_cos, 
                            sin_dev, scale_sin, out_dev, scale_out,
                            num_heads, seq_len, head_dim);
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Cleanup and free device memory
    cudaFree(x_dev);
    cudaFree(cos_dev);
    cudaFree(sin_dev);
    cudaFree(out_dev);

    delete[] x;
    delete[] cos;
    delete[] sin;
    delete[] out;

    return 0;
}