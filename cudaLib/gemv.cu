#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "gemm_utils.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INT8(x) TORCH_CHECK(x.scalar_type() == torch::kChar, #x " must be torch.int8")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == torch::kBFloat16, #x " must be torch.bfloat16")
#define CHECK_FP32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be torch.float32")

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;


// ============================================================
// INT8 GEMV with BF16 output
// One WARP computes one output element y[row, n].
//
// x       : int8  [R, K] logically
// weight  : int8  [N, K] or [B, N, K],
// x_scale : float [1] or [R]
// w_scale : float [N] or [B, N]
// y_bf16  : bf16  [R, N]
//
// Input x has shape [R, K] logically, meaning that if real input has
// 3D input shape [B, M, K], then R = B * M.
// ============================================================
__global__ void i8_gemv_out_bf16_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ y_bf16,
    int R,
    int B,
    int M,
    int N,
    int K,
    int x_scale_numel,
    int w_scale_numel,
    bool weight_is_batched,
    float alpha
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;  // thread index within the warp

    // Each warp computes one output channel n.
    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Flattened [B, M] row.
    int row = blockIdx.y;
    if (row >= R || n >= N) {
        return;
    }
    const int8_t* x_row = x + static_cast<int64_t>(row) * K;

    int b = row / M;

    int K4 = K >> 2;

    const int8_t* w_row;
    if (weight_is_batched) { // weight shape: [B, N, K]
        w_row = weight + (static_cast<int64_t>(b) * N + n) * K;
    } else {  // weight shape: [N, K]
        w_row = weight + static_cast<int64_t>(n) * K;
    }

    // Compute the dot product using dp4a
    const int32_t* __restrict__ x4 = reinterpret_cast<const int32_t*>(x_row);
    const int32_t* __restrict__ w4 = reinterpret_cast<const int32_t*>(w_row);

    int32_t acc = 0;
    for (int i = lane; i < K4; i += WARP_SIZE) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];
        acc = __dp4a(x_pack, w_pack, acc);
    }

    acc = warp_reduce_sum_int32(acc);

    /*
    Because each warp computes one output element,
    we only need one thread (first thread of warp or lane = 0) to write output.
    */
    if (lane == 0) {
        float sx = (x_scale_numel == 1) ? x_scale[0] : x_scale[row];

        float sw;
        if (w_scale_numel == N) {  // shared scale: [N]
            sw = w_scale[n];
        } else {  // batched scale: [B, N]
            sw = w_scale[static_cast<int64_t>(b) * N + n];
        }

        // Compute and write output as BF16
        float y = static_cast<float>(acc) * sx * sw * alpha;
        y_bf16[static_cast<int64_t>(row) * N + n] = __float2bfloat16(y);
    }
}

torch::Tensor i8_gemv_out_bf16(
    torch::Tensor x, // shape [K] or [B, K] or [B, M, K]
    torch::Tensor weight, // shape [N, K] or [B, N, K]
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    double alpha
) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CUDA(x_scale);
    CHECK_CUDA(w_scale);
    CHECK_INT8(x);
    CHECK_INT8(weight);
    CHECK_FP32(x_scale);
    CHECK_FP32(w_scale);

    TORCH_CHECK(x.dim() == 1 || x.dim() == 2 || x.dim() == 3,
                "x must have shape [K], [B, K], or [B, M, K]");

    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3,
                "weight must have shape [N, K] or [B, N, K]");

    x = x.contiguous();
    weight = weight.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t B = 1;
    int64_t M = 1;
    int64_t R = 1;
    int64_t N = 0;
    int64_t K = 0;

    bool x_is_1d = x.dim() == 1;
    bool x_is_2d = x.dim() == 2;
    bool x_is_3d = x.dim() == 3;

    bool weight_is_batched = weight.dim() == 3;
    if (weight_is_batched) {
        B = weight.size(0);
        N = weight.size(1);
        K = weight.size(2);
    } else {
        N = weight.size(0);
        K = weight.size(1);
    }

    if (x_is_1d) {
        TORCH_CHECK(!weight_is_batched,
                    "x shape [K] cannot be used with batched weight [B, N, K]");

        TORCH_CHECK(x.size(0) == K, "x shape [K] must match weight K");
        B = 1;
        M = 1;
        R = 1;
    } else if (x_is_2d) {
        TORCH_CHECK(x.size(1) == K, "x shape [B, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B, "x batch size must match weight batch size");
        } else {
            B = x.size(0);
        }

        M = 1;
        R = B;
    } else {
        TORCH_CHECK(x.size(2) == K, "x shape [B, M, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B, "x batch size must match weight batch size");
        } else {
            B = x.size(0);
        }

        M = x.size(1);
        R = B * M;
    }

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for dp4a");

    TORCH_CHECK(x_scale.numel() == 1 || x_scale.numel() == R,
                "x_scale must contain either 1 element or R elements");

    if (weight_is_batched) {
        TORCH_CHECK(
            w_scale.numel() == N || w_scale.numel() == B * N,
            "for batched weight [B, N, K], w_scale must have shape [N] or [B, N]"
        );
    } else {
        TORCH_CHECK(
            w_scale.numel() == N,
            "for weight [N, K], w_scale must have shape [N]"
        );
    }

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0,
                "x pointer must be 4-byte aligned");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
                "weight pointer must be 4-byte aligned");

    // Construct output tensor
    auto options_bf16 = torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(x.device());
    torch::Tensor y_bf16_flat = torch::empty({R, N}, options_bf16);
    __nv_bfloat16* y_bf16_ptr =
        reinterpret_cast<__nv_bfloat16*>(y_bf16_flat.data_ptr<at::BFloat16>());


    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();
    const float* x_scale_ptr = x_scale.data_ptr<float>();
    const float* w_scale_ptr = w_scale.data_ptr<float>();


    dim3 block(static_cast<unsigned int>(BLOCK_SIZE));
    dim3 grid(
        static_cast<unsigned int>((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
        static_cast<unsigned int>(R)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    i8_gemv_out_bf16_kernel<<<grid, block, 0, stream>>>(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            y_bf16_ptr,
            static_cast<int>(R),
            static_cast<int>(B),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(x_scale.numel()),
            static_cast<int>(w_scale.numel()),
            weight_is_batched,
            static_cast<float>(alpha)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (x_is_1d) {
        return y_bf16_flat.view({N});
    } else if (x_is_2d) {
        return y_bf16_flat.view({B, N});
    } else {
        return y_bf16_flat.view({B, M, N});
    }
}


// ============================================================
// Pad x from [R, K] logical to [R, K_pad]
// R = G * B * M
// ============================================================
__global__ void pad_x_k_kernel(
    const int8_t* __restrict__ x,
    int8_t* __restrict__ x_pad,
    int R,
    int K,
    int K_pad
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(R) * K_pad;

    if (idx >= total) {
        return;
    }

    int k_pad = idx % K_pad;
    int row = idx / K_pad;

    if (k_pad < K) {
        x_pad[idx] = x[static_cast<int64_t>(row) * K + k_pad];
    } else {
        x_pad[idx] = 0;
    }
}


// ============================================================
// Pad weight from [GB, N, K] logical to [GB, N, K_pad]
// where GB = G * B
// weight rows = G * B * N
// ============================================================
__global__ void pad_weight_k_kernel(
    const int8_t* __restrict__ weight,
    int8_t* __restrict__ weight_pad,
    int weight_rows,
    int K,
    int K_pad
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(weight_rows) * K_pad;

    if (idx >= total) {
        return;
    }

    int k_pad = idx % K_pad;
    int row = idx / K_pad;

    if (k_pad < K) {
        weight_pad[idx] = weight[static_cast<int64_t>(row) * K + k_pad];
    } else {
        weight_pad[idx] = 0;
    }
}


// ============================================================
// 4D Input INT8 GEMV with BF16 output
// One WARP computes one output element: y[g, b, m, n]
//
// x       : int8  [G, B, M, K]
// weight  : int8  [G, B, N, K]
// x_scale : float [1] or [G * B * M]
// w_scale : float [N] or [G * B * N]
// y_bf16  : bf16  [G, B, M, N]

// ============================================================
__global__ void i8_gemv_out_bf16_4d_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    __nv_bfloat16* __restrict__ y_bf16,
    int G,
    int B,
    int M,
    int N,
    int K,       // this is now K_pad
    int R,
    int x_scale_numel,
    int w_scale_numel,
    float alpha
) {

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    int out_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int row = blockIdx.y;

    if (row >= R || out_n >= N) {
        return;
    }

    int BM = B * M;

    int g = row / BM;
    int rem = row - g * BM;
    int b = rem / M;
    int m = rem - b * M;

    int K4 = K >> 2;

    const int8_t* x_row =
        x + static_cast<int64_t>(((g * B + b) * M + m)) * K;

    const int8_t* w_row =
        weight + static_cast<int64_t>(((g * B + b) * N + out_n)) * K;

    const int32_t* __restrict__ x4 =
        reinterpret_cast<const int32_t*>(x_row);

    const int32_t* __restrict__ w4 =
        reinterpret_cast<const int32_t*>(w_row);

    int32_t acc = 0;

    for (int i = lane; i < K4; i += WARP_SIZE) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];

        acc = __dp4a(x_pack, w_pack, acc);
    }

    acc = warp_reduce_sum_int32(acc);

    if (lane == 0) {
        float sx = (x_scale_numel == 1)
                     ? x_scale[0]
                     : x_scale[row];

        float sw;

        if (w_scale_numel == N) {
            sw = w_scale[out_n];
        } else {
            sw = w_scale[static_cast<int64_t>((g * B + b) * N + out_n)];
        }

        float y = static_cast<float>(acc) * sx * sw * alpha;

        y_bf16[static_cast<int64_t>(row) * N + out_n] =
            __float2bfloat16(y);
    }
}


torch::Tensor i8_gemv_out_bf16_4d(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor x_scale,
    torch::Tensor w_scale,
    double alpha
) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CUDA(x_scale);
    CHECK_CUDA(w_scale);
    CHECK_INT8(x);
    CHECK_INT8(weight);
    CHECK_FP32(x_scale);
    CHECK_FP32(w_scale);

    TORCH_CHECK(x.dim() == 4, "x must have shape [G, B, M, K]");

    TORCH_CHECK(weight.dim() == 4, "weight must have shape [G, B, N, K]");

    x = x.contiguous();
    weight = weight.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t G = x.size(0);
    int64_t B = x.size(1);
    int64_t M = x.size(2);
    int64_t K = x.size(3);
    int64_t N = weight.size(2);
    int64_t R = G * B * M; // flattened row from [G, B, M, K] to [R, K]

    TORCH_CHECK(weight.size(0) == G, "weight G dimension must match x G dimension");
    TORCH_CHECK(weight.size(1) == B, "weight B dimension must match x B dimension");
    TORCH_CHECK(weight.size(3) == K, "weight K dimension must match x K dimension");

    int64_t K_pad = ((K + 3) / 4) * 4;

    TORCH_CHECK(x_scale.numel() == 1 || x_scale.numel() == R,
                "x_scale must contain either 1 element or G * B * M elements");

    TORCH_CHECK(
        w_scale.numel() == N || w_scale.numel() == G * B * N,
        "w_scale must have shape [N] or [G, B, N]"
    );

    auto options_int8 = torch::TensorOptions()
                            .dtype(torch::kChar)
                            .device(x.device());

    torch::Tensor x_compute = x;
    torch::Tensor weight_compute = weight;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (K_pad != K) {
        x_compute = torch::empty({G, B, M, K_pad}, options_int8);
        weight_compute = torch::empty({G, B, N, K_pad}, options_int8);

        int pad_threads = 256;

        int64_t x_total = R * K_pad;
        int64_t w_rows = G * B * N;
        int64_t w_total = w_rows * K_pad;

        dim3 x_pad_grid(
            static_cast<unsigned int>((x_total + pad_threads - 1) / pad_threads)
        );

        dim3 w_pad_grid(
            static_cast<unsigned int>((w_total + pad_threads - 1) / pad_threads)
        );

        pad_x_k_kernel<<<x_pad_grid, pad_threads, 0, stream>>>(
            x.data_ptr<int8_t>(),
            x_compute.data_ptr<int8_t>(),
            static_cast<int>(R),
            static_cast<int>(K),
            static_cast<int>(K_pad)
        );

        pad_weight_k_kernel<<<w_pad_grid, pad_threads, 0, stream>>>(
            weight.data_ptr<int8_t>(),
            weight_compute.data_ptr<int8_t>(),
            static_cast<int>(w_rows),
            static_cast<int>(K),
            static_cast<int>(K_pad)
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Construct output tensor
    auto options_bf16 = torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(x.device());
    torch::Tensor y_bf16_flat = torch::empty({R, N}, options_bf16);
    __nv_bfloat16* y_bf16_ptr =
        reinterpret_cast<__nv_bfloat16*>(y_bf16_flat.data_ptr<at::BFloat16>());

    const int8_t* x_ptr = x_compute.data_ptr<int8_t>();
    const int8_t* w_ptr = weight_compute.data_ptr<int8_t>();

    const float* x_scale_ptr = x_scale.data_ptr<float>();
    const float* w_scale_ptr = w_scale.data_ptr<float>();

    dim3 block(static_cast<unsigned int>(BLOCK_SIZE));
    dim3 grid(
        static_cast<unsigned int>((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
        static_cast<unsigned int>(R)
    );

    i8_gemv_out_bf16_4d_kernel<<<grid, block, 0, stream>>>(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            y_bf16_ptr,
            static_cast<int>(G),
            static_cast<int>(B),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K_pad),   
            static_cast<int>(R),
            static_cast<int>(x_scale.numel()),
            static_cast<int>(w_scale.numel()),
            static_cast<float>(alpha)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y_bf16_flat.view({G, B, M, N});
}



// ============================================================
// One-kernel GEMV + output quantization (int8 output)
//
// One WARP computes one output element y[row, n].
//
// x       : int8  [R, K] logically
// weight  : int8  [N, K] or [B, N, K]
// x_scale : float [1] or [R]
// w_scale : float [N] or [B, N]
// y_scale : float [1] or [R]
// y_i8    : int8  [R, N]
//
// Math:
//   acc[row, n] = sum_k int8(x[row, k]) * int8(weight[b, n, k])
//
//   y_float = acc[row, n]
//             * x_scale[row or 0]
//             * w_scale[n or b,n]
//             * alpha
//
//   y_i8[row, n] = round(y_float / y_scale[row or 0])
// ============================================================
__global__ void i8_gemv_out_i8_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const float* __restrict__ y_scale,
    int8_t* __restrict__ y_i8,
    int R,
    int B,
    int M,
    int N,
    int K,
    int x_scale_numel,
    int w_scale_numel,
    int y_scale_numel,
    bool weight_is_batched,
    float alpha
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Each warp computes one output channel n.
    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // One block row corresponds to one flattened input row.
    int row = blockIdx.y;
    if (row >= R || n >= N) {
        return;
    }

    // Recover original batch index from flattened row.
    int b = row / M;

    const int8_t* x_row = x + static_cast<int64_t>(row) * K;

    const int8_t* w_row;
    if (weight_is_batched) { // weight shape: [B, N, K]
        w_row = weight + (static_cast<int64_t>(b) * N + n) * K;
    } else {  // weight shape: [N, K]
        w_row = weight + static_cast<int64_t>(n) * K;
    }

    const int32_t* __restrict__ x4 = reinterpret_cast<const int32_t*>(x_row);

    const int32_t* __restrict__ w4 = reinterpret_cast<const int32_t*>(w_row);

    int32_t acc = 0;

    // Each lane handles part of K.
    // For K = 128, K4 = 32, so each lane does exactly one dp4a.
    int K4 = K >> 2;
    for (int i = lane; i < K4; i += WARP_SIZE) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];

        acc = __dp4a(x_pack, w_pack, acc);
    }

    // Reduce inside the warp only.
    acc = warp_reduce_sum_int32(acc);

    // Lane 0 writes the final output.
    if (lane == 0) {
        float sx = (x_scale_numel == 1) ? x_scale[0] : x_scale[row];
        float sy = (y_scale_numel == 1) ? y_scale[0] : y_scale[row];

        float sw;
        if (w_scale_numel == N) {  // shared scale: [N]
            sw = w_scale[n];
        } else { // batched scale: [B, N]
            sw = w_scale[static_cast<int64_t>(b) * N + n];
        }

        // Compute and write output as int8
        float y = static_cast<float>(acc) * sx * sw * alpha;
        float q = nearbyintf(y / sy);
        q = fminf(127.0f, fmaxf(-128.0f, q));
        y_i8[static_cast<int64_t>(row) * N + n] = static_cast<int8_t>(q);
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
    CHECK_FP32(x_scale);
    CHECK_FP32(w_scale);
    CHECK_FP32(y_scale);

    TORCH_CHECK(x.dim() == 1 || x.dim() == 2 || x.dim() == 3,
                "x must have shape [K], [B, K], or [B, M, K]");

    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3,
                "weight must have shape [N, K] or [B, N, K]");

    x = x.contiguous();
    weight = weight.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();
    y_scale = y_scale.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    bool weight_is_batched = weight.dim() == 3;

    int64_t B = 1;
    int64_t M = 1;
    int64_t R = 1;
    int64_t N = 0;
    int64_t K = 0;

    bool x_is_1d = x.dim() == 1;
    bool x_is_2d = x.dim() == 2;
    bool x_is_3d = x.dim() == 3;

    if (weight_is_batched) {
        B = weight.size(0);
        N = weight.size(1);
        K = weight.size(2);
    } else {
        N = weight.size(0);
        K = weight.size(1);
    }

    if (x_is_1d) {
        TORCH_CHECK(!weight_is_batched,
                    "x shape [K] cannot be used with batched weight [B, N, K]");

        TORCH_CHECK(x.size(0) == K, "x shape [K] must match weight K");

        B = 1;
        M = 1;
        R = 1;
    } else if (x_is_2d) {
        TORCH_CHECK(x.size(1) == K, "x shape [B, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B, "x batch must match weight batch");
        } else {
            B = x.size(0);
        }

        M = 1;
        R = B;
    } else {
        TORCH_CHECK(x.size(2) == K, "x shape [B, M, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B, "x batch must match weight batch");
        } else {
            B = x.size(0);
        }

        M = x.size(1);
        R = B * M;
    }

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for dp4a");

    TORCH_CHECK(x_scale.numel() == 1 || x_scale.numel() == R,
                "x_scale must contain either 1 element or R elements");

    TORCH_CHECK(y_scale.numel() == 1 || y_scale.numel() == R,
                "y_scale must contain either 1 element or R elements");

    if (weight_is_batched) {
        TORCH_CHECK(
            w_scale.numel() == N || w_scale.numel() == B * N,
            "for batched weight [B, N, K], w_scale must have shape [N] or [B, N]"
        );
    } else {
        TORCH_CHECK(
            w_scale.numel() == N,
            "for weight [N, K], w_scale must have shape [N]"
        );
    }

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0,
                "x pointer must be 4-byte aligned");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
                "weight pointer must be 4-byte aligned");

    auto options_int8 = torch::TensorOptions()
                            .dtype(torch::kChar)
                            .device(x.device());
    torch::Tensor y_i8_flat = torch::empty({R, N}, options_int8);
    int8_t* y_i8_ptr = y_i8_flat.data_ptr<int8_t>();

    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();
    const float* x_scale_ptr = x_scale.data_ptr<float>();
    const float* w_scale_ptr = w_scale.data_ptr<float>();
    const float* y_scale_ptr = y_scale.data_ptr<float>();

    dim3 block(static_cast<unsigned int>(BLOCK_SIZE));

    dim3 grid(
        static_cast<unsigned int>((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
        static_cast<unsigned int>(R)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    i8_gemv_out_i8_kernel<<<grid, block, 0, stream>>>(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            y_scale_ptr,
            y_i8_ptr,
            static_cast<int>(R),
            static_cast<int>(B),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(x_scale.numel()),
            static_cast<int>(w_scale.numel()),
            static_cast<int>(y_scale.numel()),
            weight_is_batched,
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


// ============================================================
// 4D input GEMV + output quantization
//
// One WARP computes one output element: y[g, b, m, out_n]
//
// x       : int8  [G, B, M, K]
// weight  : int8  [G, B, N, K]
// x_scale : float [1] or [G * B * M]
// w_scale : float [N] or [G * B * N]
// y_scale : float [1] or [G * B * M]
// y_i8    : int8  [G, B, M, N]
// ============================================================
__global__ void i8_gemv_out_i8_4d_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weight,
    const float* __restrict__ x_scale,
    const float* __restrict__ w_scale,
    const float* __restrict__ y_scale,
    int8_t* __restrict__ y_i8,
    int G,
    int B,
    int M,
    int N,
    int K,
    int R,
    int x_scale_numel,
    int w_scale_numel,
    int y_scale_numel,
    float alpha
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Output channel.
    int out_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Flattened [G, B, M] row.
    int row = blockIdx.y;

    if (row >= R || out_n >= N) {
        return;
    }

    // Decode row -> g, b, m.
    int BM = B * M;

    int g = row / BM;
    int rem = row - g * BM;
    int b = rem / M;
    int m = rem - b * M;

    int K4 = K >> 2;

    // x[g, b, m, :]
    const int8_t* x_row = x + static_cast<int64_t>(((g * B + b) * M + m)) * K;

    // weight[g, b, out_n, :]
    const int8_t* w_row =
        weight + static_cast<int64_t>(((g * B + b) * N + out_n)) * K;

    const int32_t* __restrict__ x4 =
        reinterpret_cast<const int32_t*>(x_row);

    const int32_t* __restrict__ w4 =
        reinterpret_cast<const int32_t*>(w_row);

    int32_t acc = 0;

    for (int i = lane; i < K4; i += WARP_SIZE) {
        int32_t x_pack = x4[i];
        int32_t w_pack = w4[i];

        acc = __dp4a(x_pack, w_pack, acc);
    }

    acc = warp_reduce_sum_int32(acc);

    if (lane == 0) {
        float sx = (x_scale_numel == 1) ? x_scale[0] : x_scale[row];
        float sy = (y_scale_numel == 1) ? y_scale[0] : y_scale[row];

        float sw;

        if (w_scale_numel == N) { // Shared weight scale: [N]
            sw = w_scale[out_n];
        } else { // 4D batched weight scale: [G, B, N]
            sw = w_scale[static_cast<int64_t>((g * B + b) * N + out_n)];
        }

        // Compute and write output as int8
        float y = static_cast<float>(acc) * sx * sw * alpha;

        float q = nearbyintf(y / sy);
        q = fminf(127.0f, fmaxf(-128.0f, q));

        y_i8[static_cast<int64_t>(row) * N + out_n] = static_cast<int8_t>(q);
    }
}

torch::Tensor int8_gemv_out_i8_4d(
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
    CHECK_FP32(x_scale);
    CHECK_FP32(w_scale);
    CHECK_FP32(y_scale);
    TORCH_CHECK(x.dim() == 4, "x must have shape [G, B, M, K]");
    TORCH_CHECK(weight.dim() == 4, "weight must have shape [G, B, N, K]");

    x = x.contiguous();
    weight = weight.contiguous();
    x_scale = x_scale.contiguous();
    w_scale = w_scale.contiguous();
    y_scale = y_scale.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t G = x.size(0);
    int64_t B = x.size(1);
    int64_t M = x.size(2);
    int64_t K = x.size(3);
    int64_t N = weight.size(2);
    int64_t R = G * B * M;  // flattened row from [G, B, M, K] to [R, K]

    TORCH_CHECK(weight.size(0) == G, "First dimension of W must match X dimension");
    TORCH_CHECK(weight.size(1) == B, "Second dimension of W must match X dimension");
    TORCH_CHECK(weight.size(3) == K, "Fourth dimension of W must match X dimension");

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for dp4a");

    TORCH_CHECK(x_scale.numel() == 1 || x_scale.numel() == R,
                "x_scale must contain either 1 element or G * B * M elements");

    TORCH_CHECK(y_scale.numel() == 1 || y_scale.numel() == R,
                "y_scale must contain either 1 element or G * B * M elements");

    TORCH_CHECK(
        w_scale.numel() == N || w_scale.numel() == G * B * N,
        "w_scale must have shape [N] or [G, B, N]"
    );

    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<int8_t>()) % 4 == 0,
                "x pointer must be 4-byte aligned");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight.data_ptr<int8_t>()) % 4 == 0,
                "weight pointer must be 4-byte aligned");

    // Construct output tensor 
    auto options_int8 = torch::TensorOptions()
                            .dtype(torch::kChar)
                            .device(x.device());
    torch::Tensor y_i8_flat = torch::empty({R, N}, options_int8);
    int8_t* y_i8_ptr = y_i8_flat.data_ptr<int8_t>();


    const int8_t* x_ptr = x.data_ptr<int8_t>();
    const int8_t* w_ptr = weight.data_ptr<int8_t>();
    const float* x_scale_ptr = x_scale.data_ptr<float>();
    const float* w_scale_ptr = w_scale.data_ptr<float>();
    const float* y_scale_ptr = y_scale.data_ptr<float>();

    dim3 block(static_cast<unsigned int>(BLOCK_SIZE));
    dim3 grid(
        static_cast<unsigned int>((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
        static_cast<unsigned int>(R)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    i8_gemv_out_i8_4d_kernel
        <<<grid, block, 0, stream>>>(
            x_ptr,
            w_ptr,
            x_scale_ptr,
            w_scale_ptr,
            y_scale_ptr,
            y_i8_ptr,
            static_cast<int>(G),
            static_cast<int>(B),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(R),
            static_cast<int>(x_scale.numel()),
            static_cast<int>(w_scale.numel()),
            static_cast<int>(y_scale.numel()),
            static_cast<float>(alpha)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y_i8_flat.view({G, B, M, N});
}



// ============================================================
// GEMV with BF16 input, BF16 weight, BF16 output
// One WARP computes one output element y[row, n].
//
// x       : bf16 [R, K] logically
// weight  : bf16 [N, K] or [B, N, K]
// y_bf16  : bf16 [R, N]
//
// Input x has shape [R, K] logically, meaning:
//   if real input is [B, M, K], then R = B * M.
//
// Math:
//   acc[row, n] = sum_k float(x[row, k]) * float(weight[b, n, k])
//
//   y_float = acc[row, n] * alpha
//
//   y_bf16[row, n] = bf16(y_float)
// ============================================================
__global__ void bf16_gemv_out_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y_bf16,
    int R,
    int B,
    int M,
    int N,
    int K,
    bool weight_is_batched,
    float alpha
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;  // thread index within the warp

    // Each warp computes one output channel n.
    int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Flattened row from [B, M].
    int row = blockIdx.y;
    if (row >= R || n >= N) {
        return;
    }

    const __nv_bfloat16* x_row = x + static_cast<int64_t>(row) * K;

    // Recover original batch index from flattened row.
    int b = row / M;

    // weight[n, :] or weight[b, n, :]
    const __nv_bfloat16* w_row;

    if (weight_is_batched) {  // shape: [B, N, K]
        w_row = weight + (static_cast<int64_t>(b) * N + n) * K;
    } else {  // shape: [N, K]
        w_row = weight + static_cast<int64_t>(n) * K;
    }

    // Each lane computes a partial dot product.
    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP_SIZE) {
        float xv = __bfloat162float(x_row[k]);
        float wv = __bfloat162float(w_row[k]);
        acc += xv * wv;
    }

    // Compute the sum across all thread within the warp
    acc = warp_reduce_sum(acc);

    // Only lane 0 writes the final output.
    if (lane == 0) {
        float y = acc * alpha;
        y_bf16[static_cast<int64_t>(row) * N + n] = __float2bfloat16(y);
    }
}

torch::Tensor bf16_gemv_out_bf16(
    torch::Tensor x,       // shape [K] or [B, K] or [B, M, K]
    torch::Tensor weight,  // shape [N, K] or [B, N, K]
    double alpha
) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_BF16(x);
    CHECK_BF16(weight);

    TORCH_CHECK(x.dim() == 1 || x.dim() == 2 || x.dim() == 3,
                "x must have shape [K], [B, K], or [B, M, K]");

    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3,
                "weight must have shape [N, K] or [B, N, K]");

    x = x.contiguous();
    weight = weight.contiguous();

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int64_t B = 1;
    int64_t M = 1;
    int64_t R = 1;
    int64_t N = 0;
    int64_t K = 0;

    bool x_is_1d = x.dim() == 1;
    bool x_is_2d = x.dim() == 2;
    bool x_is_3d = x.dim() == 3;

    bool weight_is_batched = weight.dim() == 3;

    if (weight_is_batched) {
        B = weight.size(0);
        N = weight.size(1);
        K = weight.size(2);
    } else {
        N = weight.size(0);
        K = weight.size(1);
    }

    if (x_is_1d) {
        TORCH_CHECK(!weight_is_batched,
                    "x shape [K] cannot be used with batched weight [B, N, K]");

        TORCH_CHECK(x.size(0) == K,
                    "x shape [K] must match weight K");

        B = 1;
        M = 1;
        R = 1;
    } else if (x_is_2d) {
        TORCH_CHECK(x.size(1) == K,
                    "x shape [B, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B,
                        "x batch size must match weight batch size");
        } else {
            B = x.size(0);
        }

        M = 1;
        R = B;
    } else {
        TORCH_CHECK(x.size(2) == K,
                    "x shape [B, M, K] must match weight K");

        if (weight_is_batched) {
            TORCH_CHECK(x.size(0) == B,
                        "x batch size must match weight batch size");
        } else {
            B = x.size(0);
        }

        M = x.size(1);
        R = B * M;
    }

    auto options_bf16 = torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(x.device());
    torch::Tensor y_bf16_flat = torch::empty({R, N}, options_bf16);

    const __nv_bfloat16* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());

    const __nv_bfloat16* w_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());

    __nv_bfloat16* y_ptr =
        reinterpret_cast<__nv_bfloat16*>(y_bf16_flat.data_ptr<at::BFloat16>());

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        static_cast<unsigned int>((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
        static_cast<unsigned int>(R)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bf16_gemv_out_bf16_kernel<<<grid, block, 0, stream>>>(
            x_ptr,
            w_ptr,
            y_ptr,
            static_cast<int>(R),
            static_cast<int>(B),
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            weight_is_batched,
            static_cast<float>(alpha)
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (x_is_1d) {
        return y_bf16_flat.view({N});
    } else if (x_is_2d) {
        return y_bf16_flat.view({B, N});
    } else {
        return y_bf16_flat.view({B, M, N});
    }
}