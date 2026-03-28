#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/float8.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/gemm/device/gemm.h"


using namespace torch::indexing;

/*
Description: A custom int8 matrix multiplication using CUTLASS.

Input: 
- input: INT8 tensor of shape (M, K)
- weight: INT8 tensor of shape (N, K),
        weight is interpreted as column-major for GEMM

Output: BF16 tensor of shape (M, N)
*/
template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul(
    torch::Tensor input,   // INT8 - shape (M, K)
    torch::Tensor weight,  // INT8 - shape (N, K)
    float alpha            // FP32
) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  TORCH_CHECK(input.dtype() == torch::kChar,
              "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar,
              "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // weight is (N, K)

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  // We will pad K up to a multiple of 32 for int8 Tensor Cores (Sm80, mma 16x8x32)
  TORCH_CHECK(K > 0, "K must be > 0");
  int64_t K_gemm = ((K + 31) / 32) * 32;  // padded K used for GEMM

  input = input.contiguous();
  weight = weight.contiguous();

  // ---- Align N for epilogue (128-bit BF16 stores ⇒ 8 elements) ----
  int64_t N_gemm = ((N + 7) / 8) * 8;     // padded N for GEMM / epilogue
  bool padN = (N_gemm != N);
  bool padK = (K_gemm != K);

  // Prepare (possibly padded) input, weight, and output tensors
  torch::Tensor input_used;
  torch::Tensor weight_used;
  torch::Tensor out_full;

  auto out_options = torch::TensorOptions()
                         .dtype(torch::kBFloat16)
                         .device(input.device());

  // ---- Pad input along K if needed: (M, K_gemm) ----
  if (padK) {
    input_used = torch::zeros({M, K_gemm}, input.options());
    // Copy original data into first K columns
    input_used.index_put_({Slice(), Slice(0, K)}, input);
  } else {
    input_used = input;
  }

  // ---- Pad weight along N and/or K: (N_gemm, K_gemm) row-major ----
  if (padN || padK) {
    weight_used = torch::zeros({N_gemm, K_gemm}, weight.options());
    // Copy original weight into the top-left (N x K) block
    weight_used.index_put_({Slice(0, N), Slice(0, K)}, weight);
  } else {
    weight_used = weight;
  }

  // Output: (M, N_gemm), will slice back to N if we padded N
  out_full = torch::empty({M, N_gemm}, out_options);

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using LayoutInputA  = cutlass::layout::RowMajor;
  using LayoutInputB  = cutlass::layout::ColumnMajor;
  using LayoutOutput  = cutlass::layout::RowMajor;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // 8 BF16 per access
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,  // int8 Tensor Core MMA
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  // Use padded M x N_gemm x K_gemm for the actual GEMM
  cutlass::gemm::GemmCoord problem_size(M, N_gemm, K_gemm);

  cutlass::MatrixCoord input_size (M,      K_gemm);
  cutlass::MatrixCoord weight_size(K_gemm, N_gemm);
  cutlass::MatrixCoord output_size(M,      N_gemm);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input_used.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight_used.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out_full.data_ptr<torch::BFloat16>()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {alpha, 0.0f},
      1  
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM configuration not supported");

  status = gemm_op.initialize(arguments, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM initialization failed");

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm_op(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM execution failed");

  // Slice back to (M, N) if we padded N
  if (padN) {
    auto out = out_full.index({Slice(), Slice(0, N)}).contiguous();
    return out;
  } else {
    return out_full;
  }
}

torch::Tensor int8_matmul_host(
    torch::Tensor input,   // INT8
    torch::Tensor weight,  // INT8
    float alpha            // FP32
) {
  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M == 512 && N == 4096 && K == 14336) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 4;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (K == 4096 && N == 4096) {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else if (M == 1024 && N == 14336 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul<TileShape, WarpShape, kStages>(input, weight, alpha);
  }
}


// ================================================================
// The custom int8 matmul with per-row scale
// - Input: INT8 - (M, K)
// - Weight: INT8 - (N, K)
// - Scale: BFloat16 - scalar
// - Output: INT8 - (M, N)
// ================================================================
template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul_out_int8_per_row_scale(
    torch::Tensor input,   // INT8 - (M, K)
    torch::Tensor weight,  // INT8 - (N, K)
    float scale            // scalar scale, applied in epilogue
) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  TORCH_CHECK(input.dtype() == torch::kChar,
              "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar,
              "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // weight is (N, K)

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  // For int8 Tensor Cores (Sm80, mma shape 16x8x32) K should be multiple of 32
  TORCH_CHECK(K % 32 == 0,
              "K must be a multiple of 32 for int8 Tensor Core GEMM on SM80");

  input = input.contiguous();
  weight = weight.contiguous();

  // ---- Align N for epilogue (128-bit BF16 stores ⇒ 8 elements) ----
  int64_t N_aligned = ((N + 7) / 8) * 8;
  bool padN = (N_aligned != N);

  // Prepare (possibly padded) weight and output tensors
  torch::Tensor weight_used;
  torch::Tensor out_full;

  auto out_options = torch::TensorOptions()
                         .dtype(torch::kBFloat16)
                         .device(input.device());

  if (padN) {
    // weight_padded: (N_aligned, K), int8
    weight_used = torch::zeros({N_aligned, K}, weight.options());
    // Copy original weights into first N rows
    weight_used.index_put_({Slice(0, N), Slice()}, weight);
    // Output: (M, N_aligned)
    out_full = torch::empty({M, N_aligned}, out_options);
  } else {
    weight_used = weight;
    out_full = torch::empty({M, N}, out_options);
  }

  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using LayoutInputA  = cutlass::layout::RowMajor;
  using LayoutInputB  = cutlass::layout::ColumnMajor;
  using LayoutOutput  = cutlass::layout::RowMajor;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // 8 BF16 per access
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,  // int8 Tensor Core MMA
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  // Use aligned N for the actual GEMM
  int64_t N_gemm = N_aligned;

  cutlass::gemm::GemmCoord problem_size(M, N_gemm, K);

  cutlass::MatrixCoord input_size (M, K);
  cutlass::MatrixCoord weight_size(K, N_gemm);
  cutlass::MatrixCoord output_size(M, N_gemm);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  // weight_used is (N_gemm, K) row-major, interpreted as (K, N_gemm) col-major
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight_used.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out_full.data_ptr<torch::BFloat16>()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      out_ref,
      out_ref,
      {1.0f, 0.0f},
      1  // batch count
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM configuration not supported");

  status = gemm_op.initialize(arguments, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM initialization failed");

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm_op(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS GEMM execution failed");

  // Slice back to (M, N) if we padded
  if (padN) {
    auto out = out_full.index({Slice(), Slice(0, N)}).contiguous();
    return out;
  } else {
    return out_full;
  }
}

__global__ void rowwise_quantize_kernel(
    const cutlass::bfloat16_t* __restrict__ input,  // (M,N)
    const float* __restrict__ row_scale,           // (M,)
    int8_t* __restrict__ output,                   // (M,N)
    int M,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int row = idx / N;

    float x = static_cast<float>(input[idx]);

    float scaled = x * row_scale[row];

    // Round to nearest int and clamp to int8 range
    int q = __float2int_rn(scaled);
    q = max(-128, min(127, q));

    output[idx] = static_cast<int8_t>(q);
}

torch::Tensor int8_matmul_out_int8_per_row_scale_host(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    torch::Tensor row_scale // float (M,)
) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;

    auto bf16_out = int8_matmul_out_int8_per_row_scale<TileShape, WarpShape, kStages>(
        input, weight, 1.0f);

    int M_out = bf16_out.size(0);
    int N_out = bf16_out.size(1);

    auto output_int8 = torch::empty({M_out, N_out}, torch::dtype(torch::kChar).device(input.device()));

    // Launch the rowwise quantization kernel
    int threads = 256;
    int blocks = (M_out * N_out + threads - 1) / threads;
    rowwise_quantize_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(bf16_out.data_ptr<torch::BFloat16>()),
        row_scale.data_ptr<float>(),
        output_int8.data_ptr<int8_t>(),
        M_out, N_out
    );

    cudaDeviceSynchronize(); // ensure kernel completes

    return output_int8;
}

torch::Tensor int8_matmul_out_int8_per_row_scale_batched_host(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor row_scales  // (batch_size, M)
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kChar,
                "A must be torch.int8 (kChar)");
    TORCH_CHECK(B.dtype() == torch::kChar,
                "B must be torch.int8 (kChar)");
    
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor (batched)");
    const int64_t batch_size = A.size(0);
    const int64_t M = A.size(1);
    const int64_t K = A.size(2);

    // row_scales: [batch_size, M] float32
    TORCH_CHECK(row_scales.dim() == 2, "row_scales must be 2D tensor");
    TORCH_CHECK(row_scales.size(0) == batch_size &&
                row_scales.size(1) == M,
                "row_scales shape must match batch size and M of A");
    TORCH_CHECK(row_scales.dtype() == torch::kFloat32,
                "row_scales must be float32");

    bool shared_B = false;
    int64_t N;

    if (B.dim() == 2) {
        shared_B = true;
        TORCH_CHECK(B.size(0) > 0 && B.size(1) == K,
                    "B shape must be (N, K) with same K as A");
        N = B.size(0);
    } else {
        TORCH_CHECK(B.dim() == 3, "B must be 2D or 3D tensor");
        TORCH_CHECK(B.size(0) == batch_size &&
                    B.size(2) == K,
                    "B shape must be (batch_size, N, K) with same K as A");
        N = B.size(1);
    }

    auto out = torch::empty({batch_size, M, N}, A.options().dtype(torch::kChar));

    for (int64_t b = 0; b < batch_size; ++b) {
        auto A_b = A.select(0, b).contiguous();  // (M, K)
        torch::Tensor B_b;
        if (shared_B) {
            B_b = B;  // shared weight
        } else {            B_b = B.select(0, b).contiguous();  // (N, K)
        }
        auto row_scales_b = row_scales.select(0, b).contiguous();  // (M,)

        auto out_b_result = int8_matmul_out_int8_per_row_scale_host(
            A_b, B_b, row_scales_b);

        out.select(0, b).copy_(out_b_result);

    }
    return out;
}


// ================================================================
// The custom int8 matmul with three scales (row, col, output)
// - Input: INT8 - (M, K)
// - Weight: INT8 - (N, K)
// - Row Scale: float - (M,)
// - Col Scale: float - (N,)
// - Output Scale: float - (M,)
// - Output: INT8 - (M, N)
// ================================================================

template<int Vec>
__global__ void three_scale_quantize_kernel(
    const cutlass::bfloat16_t* __restrict__ input,   // (M, N)
    const float* __restrict__ row_factor,            // (M,) = row_scale / out_scale
    const float* __restrict__ col_scale,             // (N,)
    int8_t* __restrict__ output,                     // (M, N)
    int M,
    int N)
{
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col0 = (blockIdx.x * blockDim.x + threadIdx.x) * Vec;

    if (row >= M) return;

    float rf = row_factor[row];
    int base = row * N + col0;

    #pragma unroll
    for (int i = 0; i < Vec; ++i) {
        int col = col0 + i;
        if (col < N) {
            float x = static_cast<float>(input[base + i]);
            float scaled = x * rf * col_scale[col];

            int q = __float2int_rn(scaled);
            q = q < -128 ? -128 : (q > 127 ? 127 : q);

            output[base + i] = static_cast<int8_t>(q);
        }
    }
}


torch::Tensor int8_matmul_out_int8_three_scale_host(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    torch::Tensor out_scale)
{
    auto bf16_out = int8_matmul_host(input, weight, 1.0f);

    int M_out = bf16_out.size(0);
    int N_out = bf16_out.size(1);

    auto output_int8 = torch::empty(
        {M_out, N_out},
        torch::dtype(torch::kChar).device(input.device()));

    // Precompute row_factor = row_scale / out_scale
    auto row_factor = row_scale / out_scale;

    constexpr int Vec = 4;
    dim3 threads(128, 2);
    dim3 blocks((N_out + threads.x * Vec - 1) / (threads.x * Vec),
                (M_out + threads.y - 1) / threads.y);

    three_scale_quantize_kernel<Vec><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(bf16_out.data_ptr<torch::BFloat16>()),
        row_factor.data_ptr<float>(),
        col_scale.data_ptr<float>(),
        output_int8.data_ptr<int8_t>(),
        M_out, N_out
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_int8;
}

torch::Tensor int8_matmul_out_int8_three_scale_batched_host(
    torch::Tensor A,  // (batch_size, M, K)
    torch::Tensor B,  // (batch_size, N, K) 
    torch::Tensor row_scales,  // (batch_size, M)
    torch::Tensor col_scales,  // (batch_size, N)
    torch::Tensor out_scales   // (batch_size, M)
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kChar,
                "A must be torch.int8 (kChar)");
    TORCH_CHECK(B.dtype() == torch::kChar,
                "B must be torch.int8 (kChar)");
    
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor (batched)");
    const int64_t batch_size = A.size(0);
    const int64_t M = A.size(1);
    const int64_t K = A.size(2);

    bool shared_B = false;
    int64_t N;  
    N = B.size(1);

    // row_scales: [batch_size, M] float32
    TORCH_CHECK(row_scales.dim() == 2, "row_scales must be 2D tensor");
    TORCH_CHECK(row_scales.size(0) == batch_size &&
                row_scales.size(1) == M,
                "row_scales shape must match batch size and M of A");
    TORCH_CHECK(row_scales.dtype() == torch::kFloat32,
                "row_scales must be float32");

    // col_scales: [batch_size, N] float32
    TORCH_CHECK(col_scales.dim() == 2, "col_scales must be 2D tensor");
    TORCH_CHECK(col_scales.size(0) == batch_size &&
                col_scales.size(1) == N,
                "col_scales shape must match batch size and N of B");
    TORCH_CHECK(col_scales.dtype() == torch::kFloat32,
                "col_scales must be float32");

    // out_scales: [batch_size, M] float32
    TORCH_CHECK(out_scales.dim() == 2, "out_scales must be 2D tensor");
    TORCH_CHECK(out_scales.size(0) == batch_size &&
                out_scales.size(1) == M,
                "out_scales shape must match batch size and M of A");
    TORCH_CHECK(out_scales.dtype() == torch::kFloat32,
                "out_scales must be float32");

    auto out = torch::empty({batch_size, M, N}, A.options().dtype(torch::kChar));

    for (int64_t b = 0; b < batch_size; ++b) {
        auto A_b = A.select(0, b).contiguous();  // (M, K)
        torch::Tensor B_b;
        if (shared_B) {
            B_b = B;  // shared weight
        } else {
            B_b = B.select(0, b).contiguous();  // (N, K)
        }
        auto row_scales_b = row_scales.select(0, b).contiguous();  // (M,)
        auto col_scales_b = col_scales.select(0, b).contiguous();  // (N,)
        auto out_scales_b = out_scales.select(0, b).contiguous();  // (M,)

        auto out_b_result = int8_matmul_out_int8_three_scale_host(
            A_b, B_b, row_scales_b, col_scales_b, out_scales_b);

        out.select(0, b).copy_(out_b_result);
    }
    return out;
}




#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/gemm/device/gemm_universal_streamk_with_broadcast.h>

#include <cutlass/util/reference/host/error_metrics.h>
#include <cutlass/util/reference/host/tensor_foreach.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>


/*
Description: fusion of int8 matmul with 
            per-row and per-column scale in the epilogue, output bf16

Input:
- A: int8, (M, K), row-major
- B: int8, (N, K), column-major
- alphaRow: float, (M,), row scale
- alphaCol: float, (N,), column scale

Output:
- C: bf16, (M, N), row-major
*/

template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor matmul_w8a8(
    const torch::Tensor &A,         // int8 [M, K]
    const torch::Tensor &B,         // int8 [N, K]
    const torch::Tensor &alphaRow,  // float [M, 1]
    const torch::Tensor &alphaCol   // float [1, N]
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() 
                && alphaRow.is_cuda() && alphaCol.is_cuda());
    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32, 
                "alphaCol must be float32");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32, 
                "alphaRow must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(alphaRow.numel() == A.size(0), "alphaRow must have M elements");
    TORCH_CHECK(alphaCol.numel() == B.size(0), "alphaCol must have N elements");

    int32_t M = A.size(0);
    int32_t N = B.size(0);
    int32_t K = A.size(1);

    TORCH_CHECK(B.size(1) == K, "B must have shape (N, K)");
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");

    auto tensor_a  = A.contiguous();
    auto tensor_b  = B.contiguous();
    auto tensor_v1 = alphaRow.contiguous();
    auto tensor_v2 = alphaCol.contiguous();

    auto D = torch::empty({M, N},
        torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16));

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementScale = float;
    using ElementC = cutlass::bfloat16_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    constexpr int EVTEpilogueStages = 1;

    using namespace cute;

    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            TileShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    using V1Broadcast =
        cutlass::epilogue::threadblock::VisitorColBroadcast<
            OutputTileThreadMap, ElementScale,
            cute::Stride<_1, _0, int32_t>>;

    using V2Broadcast =
        cutlass::epilogue::threadblock::VisitorRowBroadcast<
            OutputTileThreadMap, ElementScale,
            cute::Stride<_0, _1, int32_t>>;

    using Compute0 =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies, ElementCompute, ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 =
        cutlass::epilogue::threadblock::Sm80EVT<
            Compute0, Accum, V1Broadcast>;

    using Compute1 =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies, ElementCompute, ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 =
        cutlass::epilogue::threadblock::Sm80EVT<
            Compute1, EVTCompute0, V2Broadcast>;

    using StoreD =
        cutlass::epilogue::threadblock::VisitorAuxStore<
            OutputTileThreadMap, ElementOutput,
            cutlass::FloatRoundStyle::round_to_nearest,
            cute::Stride<int64_t, _1, int64_t>>;

    using EVTD =
        cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVTCompute1>;

    using Kernel =
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
            ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
            ElementC, LayoutC, AlignmentC,
            ElementAccumulator,
            ElementCompute,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            TileShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 32>,
            EVTD,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            kStages,
            cutlass::arch::OpMultiplyAddSaturate,
            EVTEpilogueStages
        >::GemmKernel;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename EVTD::Arguments callback_args{
        {
            {
                {},
                {tensor_v1.data_ptr<float>(), ElementScale(0), {_1{}, _0{}, int32_t(M)}},
                {}
            },
            {tensor_v2.data_ptr<float>(), ElementScale(0), {_0{}, _1{}, int32_t(N)}},
            {}
        },
        {
            reinterpret_cast<ElementOutput*>(D.data_ptr<at::BFloat16>()),
            {int64_t{N}, _1{}, int64_t{M * N}}
        }
    };

    typename DeviceGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        callback_args,
        tensor_a.data_ptr<ElementA>(),
        tensor_b.data_ptr<ElementB>(),
        nullptr,
        nullptr,
        int64_t(M) * K,
        int64_t(N) * K,
        0,
        0,
        tensor_a.stride(0),
        tensor_b.stride(0),
        0,
        0
    );

    DeviceGemm gemm_op;
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

    size_t workspace_size = DeviceGemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed");

    status = gemm_op.initialize(arguments, workspace.get(), stream.stream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed");

    status = gemm_op(stream.stream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed");

    return D;
}

torch::Tensor matmul_w8a8_host(
    const torch::Tensor &A,          // int8 [M, K]
    const torch::Tensor &B,          // int8 [N, K]
    const torch::Tensor &alphaRow,    // float [M, 1]
    const torch::Tensor &alphaCol   // float [1, N]
) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);

    if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return matmul_w8a8<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
  } else if (M == 512 && N == 4096 && K == 14336) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 4;
    return matmul_w8a8<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
  } else if (K == 4096 && N == 4096) {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return matmul_w8a8<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
  } else if (M == 1024 && N == 14336 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return matmul_w8a8<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
  } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return matmul_w8a8<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
  }
}


/*
Description: fusion of int8 matmul with
             per-row and per-column scale in the epilogue, output int8

Input:
- A: int8, (M, K), row-major
- B: int8, (N, K), column-major
- alphaRow: float, (M,), row scale
- alphaCol: float, (N,), column scale

Output:
- C: int8, (M, N), row-major
*/

template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor matmul_w8a8o8_kernel(
    const torch::Tensor &A,         // int8 [M, K]
    const torch::Tensor &B,         // int8 [N, K]
    const torch::Tensor &alphaRow,  // float [M]
    const torch::Tensor &alphaCol   // float [N]
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() &&
                alphaRow.is_cuda() && alphaCol.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32, "alphaRow must be float32");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32, "alphaCol must be float32");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(alphaRow.dim() == 1 || (alphaRow.dim() == 2 && alphaRow.size(1) == 1),
                "alphaRow must have shape [M] or [M,1]");
    TORCH_CHECK(alphaCol.dim() == 1 || (alphaCol.dim() == 2 && alphaCol.size(0) == 1),
                "alphaCol must have shape [N] or [1,N]");

    int32_t M = static_cast<int32_t>(A.size(0));
    int32_t K = static_cast<int32_t>(A.size(1));
    int32_t N = static_cast<int32_t>(B.size(0));

    TORCH_CHECK(B.size(1) == K, "B must have shape (N, K)");
    TORCH_CHECK(alphaRow.numel() == M, "alphaRow must have M elements");
    TORCH_CHECK(alphaCol.numel() == N, "alphaCol must have N elements");
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");

    auto tensor_a  = A.contiguous();
    auto tensor_b  = B.contiguous();
    auto tensor_v1 = alphaRow.contiguous().view({M});
    auto tensor_v2 = alphaCol.contiguous().view({N});

    auto D = torch::empty(
        {M, N},
        torch::TensorOptions().device(A.device()).dtype(torch::kChar));

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementScale = float;
    using ElementC = int8_t;
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    constexpr int EVTEpilogueStages = 1;

    using namespace cute;

    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            TileShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

    using Accum =
        cutlass::epilogue::threadblock::VisitorAccFetch;

    using V1Broadcast =
        cutlass::epilogue::threadblock::VisitorColBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_1, _0, int32_t>>;

    using V2Broadcast =
        cutlass::epilogue::threadblock::VisitorRowBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_0, _1, int32_t>>;

    using Compute0 =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 =
        cutlass::epilogue::threadblock::Sm80EVT<
            Compute0,
            Accum,
            V1Broadcast>;

    using Compute1 =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 =
        cutlass::epilogue::threadblock::Sm80EVT<
            Compute1,
            EVTCompute0,
            V2Broadcast>;

    using StoreD =
        cutlass::epilogue::threadblock::VisitorAuxStore<
            OutputTileThreadMap,
            ElementOutput,
            cutlass::FloatRoundStyle::round_to_nearest,
            cute::Stride<int64_t, _1, int64_t>>;

    using EVTD =
        cutlass::epilogue::threadblock::Sm80EVT<
            StoreD,
            EVTCompute1>;

    using Kernel =
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            ElementA,
            LayoutA,
            cutlass::ComplexTransform::kNone,
            AlignmentA,
            ElementB,
            LayoutB,
            cutlass::ComplexTransform::kNone,
            AlignmentB,
            ElementC,
            LayoutC,
            AlignmentC,
            ElementAccumulator,
            ElementCompute,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            TileShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 32>,
            EVTD,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            kStages,
            cutlass::arch::OpMultiplyAddSaturate,
            EVTEpilogueStages>::GemmKernel;

    using DeviceGemm =
        cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename EVTD::Arguments callback_args{
        {
            {
                {},
                {
                    tensor_v1.data_ptr<float>(),
                    ElementScale(0),
                    {_1{}, _0{}, int32_t(M)}
                },
                {}
            },
            {
                tensor_v2.data_ptr<float>(),
                ElementScale(0),
                {_0{}, _1{}, int32_t(N)}
            },
            {}
        },
        {
            reinterpret_cast<ElementOutput*>(D.data_ptr<int8_t>()),
            {int64_t{N}, _1{}, int64_t{M * N}}
        }
    };

    typename DeviceGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,                          // batch count
        callback_args,
        tensor_a.data_ptr<ElementA>(),
        tensor_b.data_ptr<ElementB>(),
        nullptr,                    // C
        nullptr,                    // D (not used directly; stored via EVT)
        int64_t(M) * K,             // batch stride A
        int64_t(N) * K,             // batch stride B
        0,                          // batch stride C
        0,                          // batch stride D
        tensor_a.stride(0),         // lda
        tensor_b.stride(0),         // ldb
        0,                          // ldc
        0                           // ldd
    );

    DeviceGemm gemm_op;
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

    size_t workspace_size = DeviceGemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed");

    status = gemm_op.initialize(arguments, workspace.get(), stream.stream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed");

    status = gemm_op(stream.stream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed");

    return D;
}

torch::Tensor matmul_w8a8o8_2D_host(
    const torch::Tensor &A,          // int8 [M, K]
    const torch::Tensor &B,          // int8 [N, K]
    const torch::Tensor &alphaRow,    // float [M]
    const torch::Tensor &alphaCol   // float [N]
) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);

    if (M == 512 && N == 4096 && K == 4096) {
        using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } else if (M == 512 && N == 4096 && K == 14336) {
        using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
        constexpr int kStages = 4;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } else if (K == 4096 && N == 4096) {
        using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } else if (M == 1024 && N == 14336 && K == 4096) {
        using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } else {
        using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    }
}

torch::Tensor matmul_w8a8o8_3D_host(
    const torch::Tensor &A,          // int8 [batch_size, M, K]
    const torch::Tensor &B,          // int8 [N, K] or [batch_size, N, K]
    const torch::Tensor &alphaRow,    // float [M] or [batch_size, M] 
    const torch::Tensor &alphaCol   // float [N] or [batch_size, N]
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() &&
                alphaRow.is_cuda() && alphaCol.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32, 
                "alphaRow must be float32");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32, 
                "alphaCol must be float32");

    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor (batched)");
    TORCH_CHECK(B.dim() == 2 || B.dim() == 3, "B must be 2D or 3D tensor");

    if (alphaRow.dim() == 1) {
        TORCH_CHECK(alphaRow.size(0) == A.size(1), "alphaRow must have M elements");
    } else if (alphaRow.dim() == 2) {
        TORCH_CHECK(alphaRow.dim() == 2 && alphaRow.size(0) == A.size(0) &&
                    alphaRow.size(1) == A.size(1),
                    "alphaRow must have shape (batch_size, M)");
    } else {
        TORCH_CHECK(false, "alphaRow must be 1D or 2D tensor");
    }

    if (alphaCol.dim() == 1) {
        TORCH_CHECK(alphaCol.size(0) == B.size(0), "alphaCol must have N elements");
    } else if (alphaCol.dim() == 2) {
        TORCH_CHECK(alphaCol.dim() == 2 && alphaCol.size(0) == A.size(0) &&
                    alphaCol.size(1) == B.size(0),
                    "alphaCol must have shape (batch_size, N)");
    } else {
        TORCH_CHECK(false, "alphaCol must be 1D or 2D tensor");
    }

    const int64_t batch_size = A.size(0);
    const int64_t M = A.size(1);
    const int64_t K = A.size(2);
    int64_t N;
    bool shared_B = false;

    if (B.dim() == 2) {
        shared_B = true;
        TORCH_CHECK(B.size(0) > 0 && B.size(1) == K,
                    "B shape must be (N, K) with same K as A");
        N = B.size(0);
    } else {
        TORCH_CHECK(B.size(0) == batch_size &&
                    B.size(2) == K,
                    "B shape must be (batch_size, N, K) with same K as A");
        N = B.size(1);
    }

    auto out = torch::empty({batch_size, M, N},
                        A.options().dtype(torch::kChar));
    for (int64_t b = 0; b < batch_size; ++b) {
        auto A_b = A.select(0, b).contiguous();  // (M, K)
        torch::Tensor B_b;
        torch::Tensor alphaRow_b;
        torch::Tensor alphaCol_b;
        if (shared_B) {
            B_b = B;  // shared weight
            alphaRow_b = alphaRow.contiguous();  // (M,)
            alphaCol_b = alphaCol.contiguous();  // (N,)
        } else {
            B_b = B.select(0, b).contiguous();  // (N, K)
            alphaRow_b = alphaRow.select(0, b).contiguous();  // (M,)
            alphaCol_b = alphaCol.select(0, b).contiguous();  // (N,)
        }
        
        auto out_b_result = matmul_w8a8o8_2D_host(A_b, B_b,
                                     alphaRow_b, alphaCol_b);
        out.select(0, b).copy_(out_b_result);
    }
    return out;
}
