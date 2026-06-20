#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT8(x) TORCH_CHECK(x.scalar_type() == torch::kChar, #x " must be torch.int8")

// ============================================================
// Warp reduction
// ============================================================
__inline__ __device__ int32_t warp_reduce_sum_int32(int32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================
// Block reduction
// For blockDim.x <= 1024
// ============================================================
__inline__ __device__ int32_t block_reduce_sum_int32(int32_t val) {
    static __shared__ int32_t shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum_int32(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;

    val = 0;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0;
        val = warp_reduce_sum_int32(val);
    }

    return val;
}

// ============================================================
// Helper: allocate output
// ============================================================
torch::Tensor make_output(torch::Tensor x, int64_t N) {
    auto out_options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(x.device());

    if (x.dim() == 2) {
        return torch::empty({1, N}, out_options);
    } else {
        return torch::empty({N}, out_options);
    }
}

// ============================================================
// Helper: common checks
// ============================================================
void check_gemv_inputs(torch::Tensor x, torch::Tensor weight) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_INT8(x);
    CHECK_INT8(weight);

    TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must have shape [K] or [1, K]");

    TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");

    int64_t K;

    if (x.dim() == 1) {
        K = x.size(0);
    } else {
        TORCH_CHECK(x.size(0) == 1, "if x is 2D, it must have shape [1, K]");
        K = x.size(1);
    }

    TORCH_CHECK(weight.size(1) == K,
                "weight must have shape [N, K], where K matches x");

    TORCH_CHECK(K > 0, "K must be > 0");
    TORCH_CHECK(weight.size(0) > 0, "N must be > 0");

    TORCH_CHECK(K % 4 == 0,
                "K must be divisible by 4 for dp4a");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0,
                "x pointer must be 4-byte aligned");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
                "weight pointer must be 4-byte aligned");
}


// ============================================================
// One CUDA block computes one output element out[n].
//
// Quantized GEMV:
//   acc[n] = sum_k int8(x[k]) * int8(weight[n, k])
//
// Dequantization:
//   out[n] = bf16(acc[n] * x_scale[0] * w_scale[n] * alpha)
//
// x       : int8  [K]
// weight  : int8  [N, K], row-major
// x_scale : float [1]
// w_scale : float [N]
// out     : bf16  [N]
// ============================================================
__global__ void int8_gemv_bf16_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ out,
    int N,
    int K,
    float alpha
) {
    int n = blockIdx.x;

    if (n >= N) {
        return;
    }

    int K4 = K >> 2;

    const int32_t* __restrict__ x4 =
        reinterpret_cast<const int32_t*>(x);

    const int32_t* __restrict__ w4 =
        reinterpret_cast<const int32_t*>(
            weight + static_cast<int64_t>(n) * K
        );

    int32_t acc = 0;

    for (int i = threadIdx.x; i < K4; i += blockDim.x) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];

        acc = __dp4a(x_pack, w_pack, acc);
    }

    acc = block_reduce_sum_int32(acc);

    if (threadIdx.x == 0) {
        float scale = x_scale[0] * w_scale[n] * alpha;
        float y = static_cast<float>(acc) * scale;
        out[n] = __float2bfloat16(y);
    }
}

torch::Tensor int8_gemv_bf16(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    double alpha
) {
    check_gemv_inputs(x, weight);

    x = x.contiguous();
    weight = weight.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t K = (x.dim() == 1) ? x.size(0) : x.size(1);
    int64_t N = weight.size(0);

    torch::Tensor out = make_output(x, N);

    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();

    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

    int threads = 256;
    dim3 block(static_cast<unsigned int>(threads));
    dim3 grid(static_cast<unsigned int>(N));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int8_gemv_bf16_kernel<<<grid, block, 0, stream>>>(
        x_ptr,
        w_ptr,
        x_scale.data_ptr<float>(),
        w_scale.data_ptr<float>(),
        out_ptr,
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<float>(alpha)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}




// ============================================================
// Warp max reduction
// ============================================================
__inline__ __device__ float warp_reduce_max_float(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ============================================================
// Block max reduction
// ============================================================
__inline__ __device__ float block_reduce_max_float(float val) {
    static __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max_float(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;

    val = 0.0f;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_max_float(val);
    }

    return val;
}


// ============================================================
// Batched one-kernel GEMV + output quantization
// with precomputed y_scale
//
// x       : int8  [R, K] logically
// weight  : int8  [N, K], row-major
// x_scale : float [1] or [R]
// w_scale : float [N]
// y_scale : float [1] or [R]
// y_i8    : int8  [R, N]
//
// Math:
//   acc[row, n] = sum_k int8(x[row, k]) * int8(weight[n, k])
//
//   y_float = acc[row, n]
//             * x_scale[row or 0]
//             * w_scale[n]
//             * alpha
//
//   y_i8[row, n] = round(y_float / y_scale[row or 0])
// ============================================================
__global__ void int8_gemv_out_i8_batched_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const float* __restrict__ y_scale,
    int8_t* __restrict__ y_i8,
    int R,
    int N,
    int K,
    int x_scale_numel,
    int y_scale_numel,
    float alpha
) {
    int n = blockIdx.x;      // output channel
    int row = blockIdx.y;    // batch row

    if (row >= R || n >= N) {
        return;
    }

    int K4 = K >> 2;

    const int8_t* x_row =
        x + static_cast<int64_t>(row) * K;

    const int8_t* w_row =
        weight + static_cast<int64_t>(n) * K;

    const int32_t* __restrict__ x4 =
        reinterpret_cast<const int32_t*>(x_row);

    const int32_t* __restrict__ w4 =
        reinterpret_cast<const int32_t*>(w_row);

    int32_t acc = 0;

    for (int i = threadIdx.x; i < K4; i += blockDim.x) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];

        acc = __dp4a(x_pack, w_pack, acc);
    }

    acc = block_reduce_sum_int32(acc);

    if (threadIdx.x == 0) {
        float sx = (x_scale_numel == 1) ? x_scale[0] : x_scale[row];
        float sy = (y_scale_numel == 1) ? y_scale[0] : y_scale[row];

        float deq_scale = sx * w_scale[n] * alpha;
        float y = static_cast<float>(acc) * deq_scale;

        float inv_y_scale = 1.0f / sy;
        float q = nearbyintf(y * inv_y_scale);

        q = fminf(127.0f, fmaxf(-128.0f, q));

        y_i8[static_cast<int64_t>(row) * N + n] =
            static_cast<int8_t>(q);
    }
}

torch::Tensor int8_gemv_out_i8(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    torch::Tensor y_scale,
    double alpha
) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CUDA(x_scale);
    CHECK_CUDA(w_scale);
    CHECK_CUDA(y_scale);

    CHECK_INT8(x);
    CHECK_INT8(weight);

    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat32,
                "x_scale must be float32");

    TORCH_CHECK(w_scale.scalar_type() == torch::kFloat32,
                "w_scale must be float32");

    TORCH_CHECK(y_scale.scalar_type() == torch::kFloat32,
                "y_scale must be float32");

    TORCH_CHECK(weight.dim() == 2,
                "weight must have shape [N, K]");

    TORCH_CHECK(x.dim() == 1 || x.dim() == 2 || x.dim() == 3,
                "x must have shape [K], [B, K], or [B, M, K]");

    x = x.contiguous();
    weight = weight.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();
    y_scale = y_scale.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t N = weight.size(0);
    int64_t K = weight.size(1);

    int64_t R = 1;

    bool x_is_1d = (x.dim() == 1);
    bool x_is_2d = (x.dim() == 2);
    bool x_is_3d = (x.dim() == 3);

    int64_t B = 1;
    int64_t M = 1;

    if (x_is_1d) {
        TORCH_CHECK(x.size(0) == K,
                    "x shape [K] must match weight shape [N, K]");
        R = 1;
    } else if (x_is_2d) {
        TORCH_CHECK(x.size(1) == K,
                    "x shape [B, K] must match weight shape [N, K]");
        B = x.size(0);
        R = B;
    } else {
        TORCH_CHECK(x.size(2) == K,
                    "x shape [B, M, K] must match weight shape [N, K]");
        B = x.size(0);
        M = x.size(1);
        R = B * M;
    }

    TORCH_CHECK(K % 4 == 0,
                "K must be divisible by 4 for dp4a");

    TORCH_CHECK(w_scale.dim() == 1 && w_scale.size(0) == N,
                "w_scale must have shape [N]");

    TORCH_CHECK(x_scale.numel() == 1 || x_scale.numel() == R,
                "x_scale must contain either 1 element or R elements");

    TORCH_CHECK(y_scale.numel() == 1 || y_scale.numel() == R,
                "y_scale must contain either 1 element or R elements");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0,
                "x pointer must be 4-byte aligned");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
                "weight pointer must be 4-byte aligned");

    auto options_int8 = torch::TensorOptions()
                            .dtype(torch::kChar)
                            .device(x.device());

    torch::Tensor y_i8_flat = torch::empty({R, N}, options_int8);

    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();

    const float* x_scale_ptr = x_scale.data_ptr<float>();
    const float* w_scale_ptr = w_scale.data_ptr<float>();
    const float* y_scale_ptr = y_scale.data_ptr<float>();

    int8_t* y_i8_ptr = y_i8_flat.data_ptr<int8_t>();

    int threads = 256;

    dim3 block(static_cast<unsigned int>(threads));
    dim3 grid(
        static_cast<unsigned int>(N),
        static_cast<unsigned int>(R)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int8_gemv_out_i8_batched_kernel<<<grid, block, 0, stream>>>(
        x_ptr,
        w_ptr,
        x_scale_ptr,
        w_scale_ptr,
        y_scale_ptr,
        y_i8_ptr,
        static_cast<int>(R),
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(x_scale.numel()),
        static_cast<int>(y_scale.numel()),
        static_cast<float>(alpha)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (x_is_1d) {
        return y_i8_flat.view({N});
    } else if (x_is_2d) {
        return y_i8_flat.view({B, N});
    } else {
        return y_i8_flat.view({B, M, N});
    }
}