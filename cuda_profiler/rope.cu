/*
In this file, we implement Rotary Positional Embeddings (RoPE)
*/

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

__global__ void apply_rope_kernel(
    const float* x,      // (num_heads, seq_len, head_dim)
    const float* cos,    // (seq_len, head_dim)
    const float* sin,    // (seq_len, head_dim)
    float* out,          // (num_heads, seq_len, head_dim)
    int num_heads,
    int seq_len,
    int head_dim)
{
    int head = blockIdx.z;  // [0, num_heads)
    int pos  = blockIdx.y;  // [0, seq_len)
    int j    = blockIdx.x * blockDim.x + threadIdx.x;  // [0, head_dim/2)

    int half = head_dim / 2;
    if (head >= num_heads || pos >= seq_len || j >= half)
        return;

    // Index base for (head, pos, :)
    int x_base   = (head * seq_len + pos) * head_dim;
    int cos_base = pos * head_dim;

    // Load x1, x2
    float x1 = x[x_base + j];
    float x2 = x[x_base + j + half];

    // Load cos/sin for this position & dim pair
    float c = cos[cos_base + j];   // cos[..., j] == cos[..., j+half]
    float s = sin[cos_base + j];

    // RoPE rotation
    float out1 = x1 * c - x2 * s;
    float out2 = x1 * s + x2 * c;

    // Store back
    out[x_base + j]       = out1;
    out[x_base + j + half] = out2;
}


// Host wrapper
void apply_rope_host(
    const float* x_dev,
    const float* cos_dev,
    const float* sin_dev,
    float* out_dev,
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

    apply_rope_kernel<<<grid, block>>>(
        x_dev, cos_dev, sin_dev, out_dev,
        num_heads, seq_len, head_dim
    );
}


int main()
{
    int num_heads = 40;
    int head_dim  = 128;
    int seq_len   = 1024 * 16;
    float theta_base = 10000.0f;

    size_t x_size   = num_heads * seq_len * head_dim * sizeof(float);
    size_t rope_size = seq_len * head_dim * sizeof(float);

    float *x_dev, *cos_dev, *sin_dev, *out_dev;

    cudaMalloc(&x_dev, x_size);
    cudaMalloc(&cos_dev, rope_size);
    cudaMalloc(&sin_dev, rope_size);
    cudaMalloc(&out_dev, x_size);

    compute_rope_params_host(cos_dev, sin_dev, head_dim, seq_len, theta_base);

    int n_iter = 5;
    for (int i = 0; i < n_iter; ++i) {
        apply_rope_host(x_dev, cos_dev, sin_dev, out_dev, num_heads, seq_len, head_dim);
    }

    cudaFree(x_dev);
    cudaFree(cos_dev);
    cudaFree(sin_dev);
    cudaFree(out_dev);

    return 0;
}

