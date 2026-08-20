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

#include "gemm_utils.cu"
#include "quantization.cu"


using namespace torch::indexing;

/*
This function performs int8 matrix multiplication using CUTLASS.

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
  TORCH_CHECK(input.dtype() == torch::kChar, "input must be torch.int8 (kChar)");
  TORCH_CHECK(weight.dtype() == torch::kChar, "weight must be torch.int8 (kChar)");

  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input and weight must be 2D tensors");

  auto M = input.size(0);
  auto K = input.size(1);
  auto N = weight.size(0);  
  TORCH_CHECK(weight.size(1) == K, "weight shape must be (N, K) with same K as input");

  // We will pad K up to a multiple of 32 for int8 Tensor Cores
  TORCH_CHECK(K > 0, "K must be > 0");
  int64_t K_gemm = ((K + 31) / 32) * 32;  

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
    // Copy original data to first K columns
    input_used = torch::zeros({M, K_gemm}, input.options());
    input_used.index_put_({Slice(), Slice(0, K)}, input);  
  } else {
    input_used = input;
  }

  // ---- Pad weight along N and/or K: (N_gemm, K_gemm) row-major ----
  if (padN || padK) {
    // Copy original weight into the top-left (N x K) block
    weight_used = torch::zeros({N_gemm, K_gemm}, weight.options());
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
    const torch::Tensor &alphaRow,  // float [M] or [M, 1]
    const torch::Tensor &alphaCol   // float [N] or [1, N]
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() &&
                alphaRow.is_cuda() && alphaCol.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32, "alphaRow must be float32");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32, "alphaCol must be float32");

    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(B.size(1) == A.size(1), "B must have shape (N, K) with same K as A");
    TORCH_CHECK(alphaRow.numel() == A.size(0), "alphaRow must have M elements");
    TORCH_CHECK(alphaCol.numel() == B.size(0), "alphaCol must have N elements");

    int32_t M = static_cast<int32_t>(A.size(0));
    int32_t N = static_cast<int32_t>(B.size(0));
    int32_t K = static_cast<int32_t>(A.size(1));

    TORCH_CHECK(M > 0 && N > 0 && K > 0, "M, N, K must be > 0");

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

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16 int8
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16 int8
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8 bf16
    constexpr int EVTEpilogueStages = 1;

    // For SM80 int8 Tensor Core MMA, K=32 is the natural instruction granularity.
    // For output/epilogue, padding N to AlignmentC (=8) is usually the safe choice.
    int32_t K_gemm = static_cast<int32_t>(((K + 31) / 32) * 32);
    int32_t N_gemm = static_cast<int32_t>(((N + AlignmentC - 1) / AlignmentC) * AlignmentC);

    bool padK = (K_gemm != K);
    bool padN = (N_gemm != N);

    auto A_used = A.contiguous();
    auto B_used = B.contiguous();
    auto alphaRow_used = alphaRow.contiguous().view({M});
    auto alphaCol_used = alphaCol.contiguous().view({N});

    // Pad A: [M, K] -> [M, K_gemm]
    if (padK) {
        auto A_pad = torch::zeros({M, K_gemm}, A.options());
        A_pad.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, K)}, A_used);
        A_used = A_pad;
    }

    // Pad B: [N, K] -> [N_gemm, K_gemm]
    if (padN || padK) {
        auto B_pad = torch::zeros({N_gemm, K_gemm}, B.options());
        B_pad.index_put_(
            {torch::indexing::Slice(0, N), torch::indexing::Slice(0, K)},
            B_used
        );
        B_used = B_pad;
    }

    // alphaRow is indexed by M only, so it does not need padding.
    // alphaCol is indexed by N, so it must match padded output width.
    if (padN) {
        auto alphaCol_pad = torch::zeros({N_gemm}, alphaCol.options());
        alphaCol_pad.index_put_({torch::indexing::Slice(0, N)}, alphaCol_used);
        alphaCol_used = alphaCol_pad;
    }

    auto D_full = torch::empty(
        {M, N_gemm},
        torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16)
    );

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
                {alphaRow_used.data_ptr<float>(), ElementScale(0), {_1{}, _0{}, int32_t(M)}},
                {}
            },
            {alphaCol_used.data_ptr<float>(), ElementScale(0), {_0{}, _1{}, int32_t(N_gemm)}},
            {}
        },
        {
            reinterpret_cast<ElementOutput*>(D_full.data_ptr<at::BFloat16>()),
            {int64_t{N_gemm}, _1{}, int64_t{M * N_gemm}}
        }
    };

    typename DeviceGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N_gemm, K_gemm},
        1,
        callback_args,
        A_used.data_ptr<ElementA>(),
        B_used.data_ptr<ElementB>(),
        nullptr,
        nullptr,
        int64_t(M) * K_gemm,
        int64_t(N_gemm) * K_gemm,
        0,
        0,
        A_used.stride(0),
        B_used.stride(0),
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

    // Slice back to original shape [M, N]
    if (padN) {
        return D_full.index({torch::indexing::Slice(), torch::indexing::Slice(0, N)}).contiguous();
    }
    return D_full;
}

torch::Tensor matmul_w8a8_2D_host(
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

torch::Tensor matmul_w8a8_batched_host(
    const torch::Tensor &A,          // int8 [B, M, K] or [B0, B1, M, K]
    const torch::Tensor &B,          // int8 [B, N, K], [B0, B1, N, K], or shared [N, K]
    const torch::Tensor &alphaRow,   // float [B, M] or [B0, B1, M]
    const torch::Tensor &alphaCol    // float [B, N], [B0, B1, N], or shared [N]
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

    TORCH_CHECK(A.dim() == 3 || A.dim() == 4,
                "A must be 3D [B,M,K] or 4D [B0,B1,M,K]");

    const bool is_4d = A.dim() == 4;
    const bool shared_B = B.dim() == 2;

    TORCH_CHECK(shared_B || B.dim() == A.dim(),
                "B must be shared 2D [N,K] or have same number of dims as A");

    int64_t outer_batch;
    int64_t M;
    int64_t K;
    int64_t N;

    torch::Tensor A_3d;
    torch::Tensor B_3d;
    torch::Tensor B_shared_2d;

    torch::Tensor alphaRow_2d;
    torch::Tensor alphaCol_2d;
    torch::Tensor alphaCol_shared_1d;

    std::vector<int64_t> output_shape;

    // ============================================================
    // Case 1: A is 3D
    // A: [Bsz, M, K]
    // B: [Bsz, N, K] or shared [N, K]
    // ============================================================
    if (!is_4d) {
        const int64_t Bsz = A.size(0);
        M = A.size(1);
        K = A.size(2);
        outer_batch = Bsz;

        if (shared_B) {
            TORCH_CHECK(B.size(1) == K,
                        "For shared B [N,K], B.size(1) must match A.size(2)");
            N = B.size(0);
            B_shared_2d = B.contiguous();
        } else {
            TORCH_CHECK(B.size(0) == Bsz,
                        "For 3D B, B.size(0) must match A.size(0)");
            TORCH_CHECK(B.size(2) == K,
                        "For 3D B, B.size(2) must match A.size(2)");
            N = B.size(1);
            B_3d = B.contiguous();
        }

        TORCH_CHECK(alphaRow.dim() == 2, "For 3D A, alphaRow must be [B, M]");
        TORCH_CHECK(alphaRow.size(0) == Bsz &&
                    alphaRow.size(1) == M, "For 3D A, alphaRow must be [B, M]");

        if (shared_B) {
            TORCH_CHECK(alphaCol.dim() == 1, "For shared B, alphaCol must be shared [N]");
            TORCH_CHECK(alphaCol.size(0) == N, "For shared B, alphaCol must have shape [N]");
            alphaCol_shared_1d = alphaCol.contiguous();
        } else {
            TORCH_CHECK(alphaCol.dim() == 2, "For batched B, alphaCol must be [B, N]");
            TORCH_CHECK(alphaCol.size(0) == Bsz &&
                        alphaCol.size(1) == N, "For batched B, alphaCol must be [B, N]");
            alphaCol_2d = alphaCol.contiguous();
        }

        A_3d = A.contiguous();
        alphaRow_2d = alphaRow.contiguous();

        output_shape = {Bsz, M, N};
    }

    // ============================================================
    // Case 2: A is 4D
    // A: [B0, B1, M, K]
    // B: [B0, B1, N, K] or shared [N, K]
    // ============================================================
    else {
        const int64_t B0 = A.size(0);
        const int64_t B1 = A.size(1);

        M = A.size(2);
        K = A.size(3);
        outer_batch = B0 * B1;

        if (shared_B) {
            TORCH_CHECK(B.size(1) == K, "For shared B [N,K], B.size(1) must match A.size(3)");
            N = B.size(0);
            B_shared_2d = B.contiguous();
        } else {
            TORCH_CHECK(B.size(0) == B0, "For 4D B, B.size(0) must match A.size(0)");
            TORCH_CHECK(B.size(1) == B1, "For 4D B, B.size(1) must match A.size(1)");
            TORCH_CHECK(B.size(3) == K, "For 4D B, B.size(3) must match A.size(3)");
            N = B.size(2);
            B_3d = B.contiguous().reshape({outer_batch, N, K});
        }

        TORCH_CHECK(alphaRow.dim() == 3, "For 4D A, alphaRow must be [B0, B1, M]");
        TORCH_CHECK(alphaRow.size(0) == B0 &&
                    alphaRow.size(1) == B1 &&
                    alphaRow.size(2) == M,
                    "For 4D A, alphaRow must be [B0, B1, M]");

        if (shared_B) {
            TORCH_CHECK(alphaCol.dim() == 1,
                        "For shared B, alphaCol must be shared [N]");
            TORCH_CHECK(alphaCol.size(0) == N,
                        "For shared B, alphaCol must have shape [N]");
            alphaCol_shared_1d = alphaCol.contiguous();
        } else {
            TORCH_CHECK(alphaCol.dim() == 3,
                        "For batched 4D B, alphaCol must be [B0, B1, N]");
            TORCH_CHECK(alphaCol.size(0) == B0 &&
                        alphaCol.size(1) == B1 &&
                        alphaCol.size(2) == N,
                        "For batched 4D B, alphaCol must be [B0, B1, N]");
            alphaCol_2d = alphaCol.contiguous().reshape({outer_batch, N});
        }

        A_3d = A.contiguous().reshape({outer_batch, M, K});
        alphaRow_2d = alphaRow.contiguous().reshape({outer_batch, M});

        output_shape = {B0, B1, M, N};
    }

    auto D_3d = torch::empty(
        {outer_batch, M, N},
        torch::TensorOptions()
            .device(A.device())
            .dtype(torch::kBFloat16)
    );

    // ============================================================
    // Reuse function matmul_w8a8_2D_host for each flattened batch item
    // ============================================================
    for (int64_t b = 0; b < outer_batch; ++b) {
        auto A_b = A_3d.select(0, b);                // [M, K]
        auto alphaRow_b = alphaRow_2d.select(0, b);  // [M]

        torch::Tensor B_b;
        torch::Tensor alphaCol_b;

        if (shared_B) {
            B_b = B_shared_2d;                       // [N, K]
            alphaCol_b = alphaCol_shared_1d;         // [N]
        } else {
            B_b = B_3d.select(0, b);                 // [N, K]
            alphaCol_b = alphaCol_2d.select(0, b);   // [N]
        }

        auto D_b = matmul_w8a8_2D_host(A_b, B_b, alphaRow_b, alphaCol_b);
        D_3d.select(0, b).copy_(D_b);
    }

    return D_3d.reshape(output_shape);
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

    // ------------------------------------------------------------
    // Small-M cases
    // ------------------------------------------------------------
    if (M <= 16) {
        using TileShape = cutlass::gemm::GemmShape<16, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } 
    else if (M <= 32) {
        using TileShape = cutlass::gemm::GemmShape<32, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    } 
    else if (M <= 64) {
        using TileShape = cutlass::gemm::GemmShape<64, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
        constexpr int kStages = 3;
        return matmul_w8a8o8_kernel<TileShape, WarpShape, kStages>(A, B, alphaRow, alphaCol);
    }

    // ------------------------------------------------------------
    // Large-M cases
    // ------------------------------------------------------------
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
    const torch::Tensor &A,          // int8 [B, M, K]
    const torch::Tensor &Bmat,       // int8 [N, K] or [B, N, K]
    const torch::Tensor &alphaRow,   // float [M] or [B, M]
    const torch::Tensor &alphaCol    // float [N] or [B, N]
) {
    TORCH_CHECK(A.is_cuda() && Bmat.is_cuda() &&
                alphaRow.is_cuda() && alphaCol.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(Bmat.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32,
                "alphaRow must be float32");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32,
                "alphaCol must be float32");

    TORCH_CHECK(A.dim() == 3,
                "A must be 3D tensor with shape [B, M, K]");
    TORCH_CHECK(Bmat.dim() == 2 || Bmat.dim() == 3,
                "B must be [N, K] or [B, N, K]");

    const int64_t batch_size = A.size(0);
    const int64_t M = A.size(1);
    const int64_t K = A.size(2);

    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");

    bool shared_B = false;
    int64_t N = 0;

    if (Bmat.dim() == 2) {
        // Bmat: [N, K]
        shared_B = true;
        TORCH_CHECK(Bmat.size(1) == K, "B [N, K] must have same K as A");
        N = Bmat.size(0);
    } else {
        // Bmat: [B, N, K]
        shared_B = false;
        TORCH_CHECK(Bmat.size(0) == batch_size &&
                    Bmat.size(2) == K, "B [B, N, K] must match A batch size and K");
        N = Bmat.size(1);
    }

    auto A_contig = A.contiguous();
    auto B_contig = Bmat.contiguous();
    auto alphaRow_contig = alphaRow.contiguous();
    auto alphaCol_contig = alphaCol.contiguous();

    auto out = torch::empty({batch_size, M, N},
                        A.options().dtype(torch::kChar));
    for (int64_t b = 0; b < batch_size; ++b) {
        auto A_b = A_contig.select(0, b);  // (M, K)
        torch::Tensor B_b;
        torch::Tensor alphaRow_b = alphaRow_contig.select(0, b).view({M});  // (M,)
        torch::Tensor alphaCol_b;
        if (shared_B) {
            B_b = B_contig;  // shared weight
            alphaCol_b = alphaCol_contig.view({N});  // (N,)
        } else {
            B_b = B_contig.select(0, b);  // (N, K)
            alphaCol_b = alphaCol_contig.select(0, b).view({N});  // (N,)
        }
        
        auto out_b_result = matmul_w8a8o8_2D_host(A_b, B_b, alphaRow_b, alphaCol_b);
        out.select(0, b).copy_(out_b_result);
    }
    return out;
}

torch::Tensor matmul_w8a8o8_4D_host(
    const torch::Tensor &A,          // int8 [B0, B1, M, K]
    const torch::Tensor &Bmat,       // int8 [N, K], [B0*B1, N, K], or [B0, B1, N, K]
    const torch::Tensor &alphaRow,   // float [M], [B0*B1, M], or [B0, B1, M]
    const torch::Tensor &alphaCol    // float [N], [B0*B1, N], or [B0, B1, N]
) {
    TORCH_CHECK(A.is_cuda() && Bmat.is_cuda() &&
                alphaRow.is_cuda() && alphaCol.is_cuda(),
                "All tensors must be CUDA tensors");

    TORCH_CHECK(A.scalar_type() == torch::kInt8, "A must be int8");
    TORCH_CHECK(Bmat.scalar_type() == torch::kInt8, "B must be int8");
    TORCH_CHECK(alphaRow.scalar_type() == torch::kFloat32,
                "alphaRow must be float32");
    TORCH_CHECK(alphaCol.scalar_type() == torch::kFloat32,
                "alphaCol must be float32");

    TORCH_CHECK(A.dim() == 4,
                "A must be 4D tensor with shape [B0, B1, M, K]");

    const int64_t B0 = A.size(0);
    const int64_t B1 = A.size(1);
    const int64_t M  = A.size(2);
    const int64_t K  = A.size(3);

    const int64_t flat_batch = B0 * B1;

    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");

    auto A_3d = A.contiguous().view({flat_batch, M, K});

    torch::Tensor B_3d_or_2d;

    if (Bmat.dim() == 2) {
        // Bmat: [N, K], shared across all B0 * B1 batches
        TORCH_CHECK(Bmat.size(1) == K,
                    "B [N, K] must have same K as A");
        B_3d_or_2d = Bmat.contiguous();

    } else if (Bmat.dim() == 3) {
        // Bmat: [B0*B1, N, K]
        TORCH_CHECK(Bmat.size(0) == flat_batch &&
                    Bmat.size(2) == K,
                    "B [B0*B1, N, K] must match flattened batch and K");
        B_3d_or_2d = Bmat.contiguous();

    } else if (Bmat.dim() == 4) {
        // Bmat: [B0, B1, N, K]
        TORCH_CHECK(Bmat.size(0) == B0 &&
                    Bmat.size(1) == B1 &&
                    Bmat.size(3) == K,
                    "B [B0, B1, N, K] must match A batch dims and K");

        const int64_t N = Bmat.size(2);
        B_3d_or_2d = Bmat.contiguous().view({flat_batch, N, K});

    } else {
        TORCH_CHECK(false,
                    "B must be [N, K], [B0*B1, N, K], or [B0, B1, N, K]");
    }

    torch::Tensor alphaRow_3d_or_1d;

    if (alphaRow.dim() == 1) {
        // alphaRow: [M], shared
        TORCH_CHECK(alphaRow.size(0) == M,
                    "alphaRow [M] must have M elements");
        alphaRow_3d_or_1d = alphaRow.contiguous();

    } else if (alphaRow.dim() == 2) {
        // alphaRow: [B0*B1, M]
        TORCH_CHECK(alphaRow.size(0) == flat_batch &&
                    alphaRow.size(1) == M,
                    "alphaRow [B0*B1, M] must match flattened batch and M");
        alphaRow_3d_or_1d = alphaRow.contiguous();

    } else if (alphaRow.dim() == 3) {
        // alphaRow: [B0, B1, M]
        TORCH_CHECK(alphaRow.size(0) == B0 &&
                    alphaRow.size(1) == B1 &&
                    alphaRow.size(2) == M,
                    "alphaRow [B0, B1, M] must match A batch dims and M");

        alphaRow_3d_or_1d = alphaRow.contiguous().view({flat_batch, M});

    } else {
        TORCH_CHECK(false,
                    "alphaRow must be [M], [B0*B1, M], or [B0, B1, M]");
    }

    torch::Tensor alphaCol_3d_or_1d;

    if (alphaCol.dim() == 1) {
        // alphaCol: [N], shared
        alphaCol_3d_or_1d = alphaCol.contiguous();

    } else if (alphaCol.dim() == 2) {
        // alphaCol: [B0*B1, N]
        TORCH_CHECK(alphaCol.size(0) == flat_batch,
                    "alphaCol [B0*B1, N] must match flattened batch");
        alphaCol_3d_or_1d = alphaCol.contiguous();

    } else if (alphaCol.dim() == 3) {
        // alphaCol: [B0, B1, N]
        TORCH_CHECK(alphaCol.size(0) == B0 &&
                    alphaCol.size(1) == B1,
                    "alphaCol [B0, B1, N] must match A batch dims");

        const int64_t N = alphaCol.size(2);
        alphaCol_3d_or_1d = alphaCol.contiguous().view({flat_batch, N});

    } else {
        TORCH_CHECK(false,
                    "alphaCol must be [N], [B0*B1, N], or [B0, B1, N]");
    }

    auto out_3d = matmul_w8a8o8_3D_host(
        A_3d,
        B_3d_or_2d,
        alphaRow_3d_or_1d,
        alphaCol_3d_or_1d
    );

    const int64_t N = out_3d.size(2);

    return out_3d.view({B0, B1, M, N});
}



/*
Description: This function perform 2 steps:
1. INT8 GEMM -> BF16
2. BF16 -> row-wise INT8

Input: 
- input: int8, (..., M, K)
- weight: int8, (N, K) 
- row_scale: float, (..., M)
- col_scale: float, (N,)

Output:
- output: int8, (..., M, N), row-major
*/
std::tuple<torch::Tensor, torch::Tensor> matmul_w8a8_quantize_row_host(
    torch::Tensor input,       // (..., M, K) INT8
    torch::Tensor weight,      // (N, K) INT8
    torch::Tensor row_scale,
    torch::Tensor col_scale)
{
    torch::Tensor bf16_out;

    // 1. INT8 GEMM -> BF16
    if (input.dim() == 2) {
        bf16_out = matmul_w8a8_2D_host(input, weight, row_scale, col_scale);
    } else if (input.dim() == 3 || input.dim() == 4) {
        bf16_out = matmul_w8a8_batched_host(input, weight, row_scale, col_scale );
    } else {
        TORCH_CHECK(false, "Input tensor must be 2D, 3D, or 4D");
    }

    TORCH_CHECK(bf16_out.defined(), "Output tensor is not defined, error in matmul_w8a8");

    // 2. BF16 -> row-wise INT8
    return quantize_row_int8_cuda(bf16_out);
}


std::tuple<torch::Tensor, torch::Tensor> matmul_w8a8_quantize_row_host_batched_host(
    torch::Tensor A,           // int8 [B, M, K] or [B0, B1, M, K]
    torch::Tensor B,           // int8 [B, N, K] or [B0, B1, N, K]
    torch::Tensor row_scales,  // fp32 [B, M] or [B0, B1, M]
    torch::Tensor col_scales   // fp32 [B, N] or [B0, B1, N]
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(row_scales.is_cuda(), "row_scales must be a CUDA tensor");
    TORCH_CHECK(col_scales.is_cuda(), "col_scales must be a CUDA tensor");

    TORCH_CHECK(A.dtype() == torch::kChar, "A must be torch.int8");
    TORCH_CHECK(B.dtype() == torch::kChar, "B must be torch.int8");
    TORCH_CHECK(row_scales.dtype() == torch::kFloat32, "row_scales must be float32");
    TORCH_CHECK(col_scales.dtype() == torch::kFloat32, "col_scales must be float32");

    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be 3D or 4D");
    TORCH_CHECK(B.dim() == A.dim(), "B must have the same number of dimensions as A");

    const bool is_4d = (A.dim() == 4);

    int64_t outer_batch;
    int64_t M;
    int64_t K;
    int64_t N;

    std::vector<int64_t> output_shape;
    std::vector<int64_t> scale_output_shape;

    torch::Tensor A_3d;
    torch::Tensor B_3d;
    torch::Tensor row_scales_2d;
    torch::Tensor col_scales_2d;

    // ============================================================
    // Case 1: 3D input
    // A: [B, M, K]
    // B: [B, N, K]
    // ============================================================
    if (!is_4d) {
        const int64_t Bsz = A.size(0);
        M = A.size(1);
        K = A.size(2);

        TORCH_CHECK(B.size(0) == Bsz, "For 3D input, B.size(0) must match A.size(0)");
        TORCH_CHECK(B.size(2) == K, "For 3D input, B.size(2) must match A.size(2)");

        N = B.size(1);
        outer_batch = Bsz;

        TORCH_CHECK(row_scales.dim() == 2, "For 3D A, row_scales must be [B, M]");
        TORCH_CHECK(row_scales.size(0) == Bsz &&
                    row_scales.size(1) == M, "For 3D A, row_scales shape must be [B, M]");

        TORCH_CHECK(col_scales.dim() == 2, "For 3D A, col_scales must be [B, N]");
        TORCH_CHECK(col_scales.size(0) == Bsz &&
                    col_scales.size(1) == N, "For 3D A, col_scales shape must be [B, N]");

        A_3d = A.contiguous();
        B_3d = B.contiguous();
        row_scales_2d = row_scales.contiguous();
        col_scales_2d = col_scales.contiguous();

        output_shape = {Bsz, M, N};
        scale_output_shape = {Bsz, M};
    }

    // ============================================================
    // Case 2: 4D input
    // A: [B0, B1, M, K]
    // B: [B0, B1, N, K]
    // ============================================================
    else {
        const int64_t B0 = A.size(0);
        const int64_t B1 = A.size(1);

        M = A.size(2);
        K = A.size(3);

        TORCH_CHECK(B.size(0) == B0, "For 4D input, B.size(0) must match A.size(0)");
        TORCH_CHECK(B.size(1) == B1, "For 4D input, B.size(1) must match A.size(1)");
        TORCH_CHECK(B.size(3) == K, "For 4D input, B.size(3) must match A.size(3)");

        N = B.size(2);
        outer_batch = B0 * B1;

        TORCH_CHECK(row_scales.dim() == 3, "For 4D A, row_scales must be [B0, B1, M]");
        TORCH_CHECK(row_scales.size(0) == B0 &&
                    row_scales.size(1) == B1 &&
                    row_scales.size(2) == M,
                    "For 4D A, row_scales shape must be [B0, B1, M]");

        TORCH_CHECK(col_scales.dim() == 3,
                    "For 4D A, col_scales must be [B0, B1, N]");
        TORCH_CHECK(col_scales.size(0) == B0 &&
                    col_scales.size(1) == B1 &&
                    col_scales.size(2) == N,
                    "For 4D A, col_scales shape must be [B0, B1, N]");

        A_3d = A.contiguous().reshape({outer_batch, M, K});
        B_3d = B.contiguous().reshape({outer_batch, N, K});

        row_scales_2d = row_scales.contiguous().reshape({outer_batch, M});
        col_scales_2d = col_scales.contiguous().reshape({outer_batch, N});

        output_shape = {B0, B1, M, N};
        scale_output_shape = {B0, B1, M};
    }

    auto out_3d = torch::empty(
        {outer_batch, M, N},
        A.options().dtype(torch::kChar)
    );

    auto out_scales_2d = torch::empty(
        {outer_batch, M},
        torch::dtype(torch::kFloat32).device(A.device())
    );

    // ============================================================
    // Reuse existing 2D function per flattened item
    // ============================================================
    for (int64_t b = 0; b < outer_batch; ++b) {
        auto A_b = A_3d.select(0, b).contiguous();              // [M, K]
        auto B_b = B_3d.select(0, b).contiguous();              // [N, K]
        auto row_scales_b = row_scales_2d.select(0, b).contiguous(); // [M]
        auto col_scales_b = col_scales_2d.select(0, b).contiguous(); // [N]

        auto result = matmul_w8a8_quantize_row_host(
            A_b,
            B_b,
            row_scales_b,
            col_scales_b
        );

        auto out_b_result = std::get<0>(result);        // [M, N]
        auto scale_b_result = std::get<1>(result);      // [M]

        out_3d.select(0, b).copy_(out_b_result);
        out_scales_2d.select(0, b).copy_(scale_b_result);
    }

    auto out = out_3d.reshape(output_shape);
    auto out_scales = out_scales_2d.reshape(scale_output_shape);

    return {out, out_scales};
}
