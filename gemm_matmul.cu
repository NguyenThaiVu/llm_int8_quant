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
#include "cutlass/epilogue/thread/linear_combination_relu.h"

#include "epilogue/thread/linear_combination.h" // my custom epilogue


using namespace torch::indexing;
// template <typename TileShape, typename WarpShape, int kStages>
// torch::Tensor int8_matmul(
//     torch::Tensor input,   // INT8 - shape (M, K)
//     torch::Tensor weight,  // INT8 - shape (N, K)
//     float alpha            // FP32
// ) {
//   TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
//   TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

//   TORCH_CHECK(input.dtype() == torch::kChar,
//               "input must be torch.int8 (kChar)");
//   TORCH_CHECK(weight.dtype() == torch::kChar,
//               "weight must be torch.int8 (kChar)");

//   TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
//               "input and weight must be 2D tensors");

//   auto M = input.size(0);
//   auto K = input.size(1);
//   auto N = weight.size(0);  // weight is (N, K)

//   TORCH_CHECK(weight.size(1) == K,
//               "weight shape must be (N, K) with same K as input");

//   // For int8 Tensor Cores (Sm80, mma shape 16x8x32) K should be multiple of 32
//   TORCH_CHECK(K % 32 == 0,
//               "K must be a multiple of 32 for int8 Tensor Core GEMM on SM80");

//   input = input.contiguous();
//   weight = weight.contiguous();

//   // ---- Align N for epilogue (128-bit BF16 stores ⇒ 8 elements) ----
//   int64_t N_aligned = ((N + 7) / 8) * 8;
//   bool padN = (N_aligned != N);

//   // Prepare (possibly padded) weight and output tensors
//   torch::Tensor weight_used;
//   torch::Tensor out_full;

//   auto out_options = torch::TensorOptions()
//                          .dtype(torch::kBFloat16)
//                          .device(input.device());

//   if (padN) {
//     // weight_padded: (N_aligned, K), int8
//     weight_used = torch::zeros({N_aligned, K}, weight.options());
//     // Copy original weights into first N rows
//     weight_used.index_put_({Slice(0, N), Slice()}, weight);
//     // Output: (M, N_aligned)
//     out_full = torch::empty({M, N_aligned}, out_options);
//   } else {
//     weight_used = weight;
//     out_full = torch::empty({M, N}, out_options);
//   }

//   using ElementOutput = cutlass::bfloat16_t;
//   using ElementAccumulator = int32_t;
//   using ElementComputeEpilogue = float;
//   using ElementInputA = int8_t;
//   using ElementInputB = int8_t;

//   using LayoutInputA  = cutlass::layout::RowMajor;
//   using LayoutInputB  = cutlass::layout::ColumnMajor;
//   using LayoutOutput  = cutlass::layout::RowMajor;

//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput,
//       128 / cutlass::sizeof_bits<ElementOutput>::value,  // 8 BF16 per access
//       ElementAccumulator,
//       ElementComputeEpilogue>;

//   using Gemm = cutlass::gemm::device::Gemm<
//       ElementInputA,
//       LayoutInputA,
//       ElementInputB,
//       LayoutInputB,
//       ElementOutput,
//       LayoutOutput,
//       ElementAccumulator,
//       cutlass::arch::OpClassTensorOp,
//       cutlass::arch::Sm80,
//       TileShape,
//       WarpShape,
//       cutlass::gemm::GemmShape<16, 8, 32>,  // int8 Tensor Core MMA
//       EpilogueOp,
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//       kStages>;

//   // Use aligned N for the actual GEMM
//   int64_t N_gemm = N_aligned;

//   cutlass::gemm::GemmCoord problem_size(M, N_gemm, K);

//   cutlass::MatrixCoord input_size (M, K);
//   cutlass::MatrixCoord weight_size(K, N_gemm);
//   cutlass::MatrixCoord output_size(M, N_gemm);

//   cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
//       reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
//       LayoutInputA::packed(input_size));

//   // weight_used is (N_gemm, K) row-major, interpreted as (K, N_gemm) col-major
//   cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
//       reinterpret_cast<ElementInputB*>(weight_used.data_ptr<int8_t>()),
//       LayoutInputB::packed(weight_size));

//   cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
//       reinterpret_cast<ElementOutput*>(out_full.data_ptr<torch::BFloat16>()),
//       LayoutOutput::packed(output_size));

//   typename Gemm::Arguments arguments{
//       problem_size,
//       input_ref,
//       weight_ref,
//       out_ref,
//       out_ref,
//       {alpha, 0.0f},
//       1  // batch count
//   };

//   Gemm gemm_op;

//   size_t workspace_size = Gemm::get_workspace_size(arguments);
//   cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//   cutlass::Status status = gemm_op.can_implement(arguments);
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM configuration not supported");

//   status = gemm_op.initialize(arguments, workspace.get());
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM initialization failed");

//   auto stream = at::cuda::getCurrentCUDAStream();
//   status = gemm_op(stream.stream());
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM execution failed");

//   // Slice back to (M, N) if we padded
//   if (padN) {
//     auto out = out_full.index({Slice(), Slice(0, N)}).contiguous();
//     return out;
//   } else {
//     return out_full;
//   }
// }

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

  // weight_used is (N_gemm, K_gemm) row-major, interpreted as (K_gemm, N_gemm) col-major
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
// The custom int8 matmul
// - Input: INT8 - (M, K)
// - Weight: INT8 - (N, K)
// - Scale: BFloat16 - scalar
// - Output: INT8 - (M, N)
// TODO: figure out shape of scale
// ================================================================
template <typename TileShape, typename WarpShape, int kStages>
torch::Tensor int8_matmul_output_int8(
    torch::Tensor input,   // INT8 - (M, K)
    torch::Tensor weight,  // INT8 - (N, K)
    float scale            // scalar scale, applied in epilogue
) {
  TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  TORCH_CHECK(input.dtype()  == torch::kChar, "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar, "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  // weight: (N, K)

  TORCH_CHECK(weight.size(1) == K,
              "weight shape must be (N, K) with same K as input");

  // For int8 Tensor Cores on SM80 (mma 16x8x32), K should be multiple of 32
  TORCH_CHECK(K % 32 == 0,
              "K must be a multiple of 32 for int8 Tensor Core GEMM on SM80");

  // Make sure we have contiguous memory
  input  = input.contiguous();
  weight = weight.contiguous();

  // ---- Align N for 128-bit epilogue on int8 (16 elements per access) ----
  constexpr int kElementsPerAccess =
      128 / cutlass::sizeof_bits<int8_t>::value;  // 128 / 8 = 16

  int64_t N_aligned = ((N + kElementsPerAccess - 1) / kElementsPerAccess) * kElementsPerAccess;
  bool padN = (N_aligned != N);

  // Prepare (possibly padded) weight and output tensors
  torch::Tensor weight_used;
  torch::Tensor out_full;

  auto out_options = torch::TensorOptions()
                         .dtype(torch::kChar)      // int8 output
                         .device(input.device());

  if (padN) {
    // weight_padded: (N_aligned, K), int8
    weight_used = torch::zeros({N_aligned, K}, weight.options());
    // Copy original weights into first N rows
    weight_used.index_put_({Slice(0, N), Slice()}, weight);

    // Output: (M, N_aligned), int8
    out_full = torch::empty({M, N_aligned}, out_options);
  } else {
    weight_used = weight;
    out_full    = torch::empty({M, N}, out_options);
  }

  using ElementOutput          = int8_t;   // int8 output
  using ElementAccumulator     = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA          = int8_t;
  using ElementInputB          = int8_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      kElementsPerAccess,           // 16 int8 per access (128-bit)
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

  int64_t N_gemm = N_aligned;

  cutlass::gemm::GemmCoord problem_size(M, N_gemm, K);

  cutlass::MatrixCoord input_size (M, K);
  cutlass::MatrixCoord weight_size(K, N_gemm);
  cutlass::MatrixCoord output_size(M, N_gemm);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
      LayoutInputA::packed(input_size));

  // weight_used: (N_gemm, K) row-major, interpreted as (K, N_gemm) column-major
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      reinterpret_cast<ElementInputB*>(weight_used.data_ptr<int8_t>()),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      reinterpret_cast<ElementOutput*>(out_full.data_ptr<int8_t>()),
      LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,   // A
      weight_ref,  // B
      out_ref,     // C
      out_ref,     // D
      {scale, 0.0f},  // epilogue: D = scale * accum + 0
      1               // batch_count
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

torch::Tensor int8_matmul_output_int8_host(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    float scale // BFloat16
) {
  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);

  if (M == 512 && N == 4096 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
    constexpr int kStages = 3;
    return int8_matmul_output_int8<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (M == 512 && N == 4096 && K == 14336) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 4;
    return int8_matmul_output_int8<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (K == 4096 && N == 4096) {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_output_int8<TileShape, WarpShape, kStages>(input, weight, scale);
  } else if (M == 1024 && N == 14336 && K == 4096) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_output_int8<TileShape, WarpShape, kStages>(input, weight, scale);
    } else {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;
    return int8_matmul_output_int8<TileShape, WarpShape, kStages>(input, weight, scale);
    }
}

// ===============================================================
torch::Tensor int8_matmul_output_int8_batched_host(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor scales   // 1D float tensor, length = batch_size
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

    // scales: [batch_size] float32 (on CPU or CUDA depending on how you use it)
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D tensor");
    TORCH_CHECK(scales.size(0) == batch_size,
                "scales length must match batch size");
    TORCH_CHECK(scales.dtype() == torch::kFloat32,
                "scales must be float32");

    // Get pointer to scales data
		torch::Tensor scales_cpu = scales;
		if (scales.is_cuda()) {
				scales_cpu = scales.to(torch::kCPU);
		}
		scales_cpu = scales_cpu.contiguous();
		const float* scales_ptr = scales_cpu.data_ptr<float>();

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

    // Loop through batches
		for (int64_t b = 0; b < batch_size; ++b) {
				auto A_b = A.select(0, b).contiguous();  // (M, K)
				torch::Tensor B_b;
				if (shared_B) {
						B_b = B;  // shared weight
				} else {
						B_b = B.select(0, b).contiguous();  // (N, K)
				}
				float scale_b = scales_ptr[b];

				// Call the single matmul function
				auto out_b_result = int8_matmul_output_int8_host(
						A_b, B_b, scale_b);

				// Copy result to the appropriate slice of out
				out.select(0, b).copy_(out_b_result);
		}

    return out;
}


// // ================================================================
// // The custom int8 matmul
// // - Input: INT8 - (M, K)
// // - Weight: INT8 - (N, K)
// // - Scale: BFloat16 - (M, N)
// // - Output: INT8 - (M, N)
// // TODO: figure out shape of scale
// // ================================================================
// template <typename TileShape, typename WarpShape, int kStages>
// torch::Tensor int8_gemm_per_output_scale_int8(
//     torch::Tensor input,   // INT8 - (M, K)
//     torch::Tensor weight,  // INT8 - (N, K)
//     torch::Tensor scale            // BFloat16 - (M, N)
// ) {
//   TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
//   TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

//   TORCH_CHECK(input.dtype()  == torch::kChar, "input must be torch.int8 (kChar)");
//   TORCH_CHECK(weight.dtype() == torch::kChar, "weight must be torch.int8 (kChar)");

//   TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
//               "input and weight must be 2D tensors");

//   auto M = input.size(0);
//   auto K = input.size(1);
//   auto N = weight.size(0);  // weight: (N, K)

//   TORCH_CHECK(weight.size(1) == K,
//               "weight shape must be (N, K) with same K as input");

//   // For int8 Tensor Cores on SM80 (mma 16x8x32), K should be multiple of 32
//   TORCH_CHECK(K % 32 == 0,
//               "K must be a multiple of 32 for int8 Tensor Core GEMM on SM80");

//   TORCH_CHECK(scale.size(0) == M && scale.size(1) == N, "scale shape must be (M, N)");
//   TORCH_CHECK(scale.dtype() == torch::kBFloat16, "scale must be torch.bfloat16");

//   input  = input.contiguous();
//   weight = weight.contiguous();
//   scale = scale.contiguous();

//   // ---- Align N for 128-bit epilogue on int8 (16 elements per access) ----
//   constexpr int kElementsPerAccess =
//       128 / cutlass::sizeof_bits<int8_t>::value;  // 128 / 8 = 16

//   int64_t N_aligned = ((N + kElementsPerAccess - 1) / kElementsPerAccess) * kElementsPerAccess;
//   bool padN = (N_aligned != N);

//   // Prepare (possibly padded) weight and output tensors
//   torch::Tensor weight_used;
//   torch::Tensor out_full;

//   auto out_options = torch::TensorOptions()
//                          .dtype(torch::kChar)      // int8 output
//                          .device(input.device());

//   if (padN) {
//     // weight_padded: (N_aligned, K), int8
//     weight_used = torch::zeros({N_aligned, K}, weight.options());
//     // Copy original weights into first N rows
//     weight_used.index_put_({Slice(0, N), Slice()}, weight);

//     // Output: (M, N_aligned), int8
//     out_full = torch::empty({M, N_aligned}, out_options);
//   } else {
//     weight_used = weight;
//     out_full    = torch::empty({M, N}, out_options);
//   }

//   using ElementOutput          = int8_t;   // int8 output
//   using ElementAccumulator     = int32_t;
//   using ElementComputeEpilogue = float;
//   using ElementInputA          = int8_t;
//   using ElementInputB          = int8_t;

//   using LayoutInputA = cutlass::layout::RowMajor;
//   using LayoutInputB = cutlass::layout::ColumnMajor;
//   using LayoutOutput = cutlass::layout::RowMajor;

// //   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
// //       ElementOutput,
// //       kElementsPerAccess,           // 16 int8 per access (128-bit)
// //       ElementAccumulator,
// //       ElementComputeEpilogue>;
  
//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination_Dequant<
//       ElementOutput,
//       kElementsPerAccess,
//       ElementAccumulator,
//       ElementComputeEpilogue>;

//   using Gemm = cutlass::gemm::device::Gemm<
//       ElementInputA,
//       LayoutInputA,
//       ElementInputB,
//       LayoutInputB,
//       ElementOutput,
//       LayoutOutput,
//       ElementAccumulator,
//       cutlass::arch::OpClassTensorOp,
//       cutlass::arch::Sm80,
//       TileShape,
//       WarpShape,
//       cutlass::gemm::GemmShape<16, 8, 32>,  // int8 Tensor Core MMA
//       EpilogueOp,
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//       kStages>;

//   int64_t N_gemm = N_aligned;

//   cutlass::gemm::GemmCoord problem_size(M, N_gemm, K);

//   cutlass::MatrixCoord input_size (M, K);
//   cutlass::MatrixCoord weight_size(K, N_gemm);
//   cutlass::MatrixCoord output_size(M, N_gemm);

//   cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
//       reinterpret_cast<ElementInputA*>(input.data_ptr<int8_t>()),
//       LayoutInputA::packed(input_size));

//   cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
//       reinterpret_cast<ElementInputB*>(weight_used.data_ptr<int8_t>()),
//       LayoutInputB::packed(weight_size));

//   cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
//       reinterpret_cast<ElementOutput*>(out_full.data_ptr<int8_t>()),
//       LayoutOutput::packed(output_size));
  
//   cutlass::TensorRef<ElementOutput, LayoutOutput> scale_ref(
//     reinterpret_cast<ElementOutput*>(scale.data_ptr<torch::BFloat16>()),
//     LayoutOutput::packed(output_size));

//   typename Gemm::Arguments arguments{
//       problem_size,
//       input_ref,   // A
//       weight_ref,  // B
//       scale_ref,     // C
//       out_ref,     // D
//       {1.0, 1.0f},  // epilogue: D = scale * accum + 0
//       1               // batch_count
//   };

//   Gemm gemm_op;

//   size_t workspace_size = Gemm::get_workspace_size(arguments);
//   cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//   cutlass::Status status = gemm_op.can_implement(arguments);
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM configuration not supported");

//   status = gemm_op.initialize(arguments, workspace.get());
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM initialization failed");

//   auto stream = at::cuda::getCurrentCUDAStream();
//   status = gemm_op(stream.stream());
//   TORCH_CHECK(status == cutlass::Status::kSuccess,
//               "CUTLASS GEMM execution failed");

//   // Slice back to (M, N) if we padded N
//   if (padN) {
//     auto out = out_full.index({Slice(), Slice(0, N)}).contiguous();
//     return out;
//   } else {
//     return out_full;
//   }
// }

// torch::Tensor int8_gemm_per_output_scale_int8_host(
//     torch::Tensor input,    // INT8
//     torch::Tensor weight,   // INT8
//     torch::Tensor scale   // BFloat16 (M, N)
// ) {
//   auto M = input.size(0);
//   auto K = input.size(1);
//   auto N = weight.size(0);

//   if (M == 512 && N == 4096 && K == 4096) {
//     using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 64, 128>;
//     constexpr int kStages = 3;
//     return int8_gemm_per_output_scale_int8<TileShape, WarpShape, kStages>(input, weight, scale);
//   } else if (M == 512 && N == 4096 && K == 14336) {
//     using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
//     constexpr int kStages = 4;
//     return int8_gemm_per_output_scale_int8<TileShape, WarpShape, kStages>(input, weight, scale);
//   } else if (K == 4096 && N == 4096) {
//     using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
//     constexpr int kStages = 3;
//     return int8_gemm_per_output_scale_int8<TileShape, WarpShape, kStages>(input, weight, scale);
//   } else if (M == 1024 && N == 14336 && K == 4096) {
//     using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
//     constexpr int kStages = 3;
//     return int8_gemm_per_output_scale_int8<TileShape, WarpShape, kStages>(input, weight, scale);
//     } else {
//     using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
//     constexpr int kStages = 3;
//     return int8_gemm_per_output_scale_int8<TileShape, WarpShape, kStages>(input, weight, scale);
//     }
// }

// torch::Tensor int8_gemm_per_output_scale_int8_batched_host(
//     torch::Tensor A,
//     torch::Tensor B,
//     torch::Tensor scales   // 2D BFloat16 tensor, shape = (batch_size, M, N)
// ) {
//     TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
//     TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
//     TORCH_CHECK(A.dtype() == torch::kChar,
//                 "A must be torch.int8 (kChar)");
//     TORCH_CHECK(B.dtype() == torch::kChar,
//                 "B must be torch.int8 (kChar)");
    
//     TORCH_CHECK(A.dim() == 3, "A must be 3D tensor (batched)");
//     const int64_t batch_size = A.size(0);
//     const int64_t M = A.size(1);
//     const int64_t K = A.size(2);

//     // scales: [batch_size, M, N] BFloat16 (on CPU or CUDA depending on how you use it)
//     TORCH_CHECK(scales.dim() == 3, "scales must be 3D tensor");
//     TORCH_CHECK(scales.size(0) == batch_size &&
//                 scales.size(1) == M,
//                 "scales shape must match batch size and M of A");
//     TORCH_CHECK(scales.dtype() == torch::kBFloat16,
//                 "scales must be bfloat16");

//     bool shared_B = false;
//     int64_t N;

//     if (B.dim() == 2) {
//         shared_B = true;
//         TORCH_CHECK(B.size(0) > 0 && B.size(1) == K,
//                     "B shape must be (N, K) with same K as A");
//         N = B.size(0);
//     } else {
//         TORCH_CHECK(B.dim() == 3, "B must be 2D or 3D tensor");
//         TORCH_CHECK(B.size(0) == batch_size &&
//                     B.size(2) == K,
//                     "B shape must be (batch_size, N, K) with same K as A");
//         N = B.size(1);
//     }

//     auto out = torch::empty({batch_size, M, N}, A.options().dtype(torch::kChar));

//     // Loop through batches
//     for (int64_t b = 0; b < batch_size; ++b) {
//         auto A_b = A.select(0, b).contiguous();  // (M, K)
//         torch::Tensor B_b;
//         if (shared_B) {
//             B_b = B;  // shared weight
//         } else {
//             B_b = B.select(0, b).contiguous();  // (N, K)
//         }
//         auto scale_b = scales.select(0, b).contiguous();  // (M, N)

//         // Call the single matmul function
//         auto out_b_result = int8_gemm_per_output_scale_int8_host(
//             A_b, B_b, scale_b);

//         // Copy result to the appropriate slice of out
//         out.select(0, b).copy_(out_b_result);
//     }

//     return out;
// }



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




__global__ void three_scale_quantize_kernel(
    const cutlass::bfloat16_t* __restrict__ input,  // (M,N)
    const float* __restrict__ row_scale,           // (M,)
    const float* __restrict__ col_scale,           // (N,)
    const float* __restrict__ out_scale,           // (M,) 
    int8_t* __restrict__ output,                   // (M,N)
    int M,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    float x = static_cast<float>(input[idx]);

    float scaled = x * row_scale[row] * col_scale[col] / out_scale[row];

    // Round to nearest int and clamp to int8 range
    int q = __float2int_rn(scaled);
    q = max(-128, min(127, q));

    output[idx] = static_cast<int8_t>(q);
}

torch::Tensor int8_matmul_out_int8_three_scale_host(
    torch::Tensor input,    // INT8
    torch::Tensor weight,   // INT8
    torch::Tensor row_scale, // float (M,)
    torch::Tensor col_scale, // float (N,)
    torch::Tensor out_scale  // float (M,)
) {
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;

    auto bf16_out = int8_matmul<TileShape, WarpShape, kStages>(
        input, weight, 1.0f);

    int M_out = bf16_out.size(0);
    int N_out = bf16_out.size(1);

    auto output_int8 = torch::empty({M_out, N_out}, torch::dtype(torch::kChar).device(input.device()));

    // Launch the rowwise quantization kernel
    int threads = 256;
    int blocks = (M_out * N_out + threads - 1) / threads;
    three_scale_quantize_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(bf16_out.data_ptr<torch::BFloat16>()),
        row_scale.data_ptr<float>(),
        col_scale.data_ptr<float>(),
        out_scale.data_ptr<float>(),
        output_int8.data_ptr<int8_t>(),
        M_out, N_out
    );

    cudaDeviceSynchronize(); // ensure kernel completes

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
