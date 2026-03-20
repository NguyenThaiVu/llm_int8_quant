#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/half.h"

#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"

#define CUDA_CHECK(x)                                                     \
  do {                                                                    \
    cudaError_t err = (x);                                                \
    if (err != cudaSuccess) {                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";     \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define CUTLASS_CHECK(x)                                                  \
  do {                                                                    \
    cutlass::Status st = (x);                                             \
    if (st != cutlass::Status::kSuccess) {                                \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(st)        \
                << "\n";                                                  \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int main() {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombinationBiasElementwise<
          ElementC,
          ElementAccumulator,
          ElementCompute,
          ElementC,
          ElementC,
          8,
          cutlass::epilogue::thread::Identity<ElementCompute>,
          cutlass::multiplies<ElementCompute>
      >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      8,
      8
  >;

  int M = 128;
  int N = 128;
  int K = 64;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  std::vector<ElementA> hA(M * K, ElementA(1));
  std::vector<ElementB> hB(K * N, ElementB(1));
  std::vector<ElementC> hZ(M * N, ElementC(0));
  std::vector<ElementC> hBroadcast(M, ElementC(1));  // matches issue repro style

  ElementA* dA = nullptr;
  ElementB* dB = nullptr;
  ElementC* dZ = nullptr;
  ElementC* dBroadcast = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(ElementA) * hA.size()));
  CUDA_CHECK(cudaMalloc((void**)&dB, sizeof(ElementB) * hB.size()));
  CUDA_CHECK(cudaMalloc((void**)&dZ, sizeof(ElementC) * hZ.size()));
  CUDA_CHECK(cudaMalloc((void**)&dBroadcast, sizeof(ElementC) * hBroadcast.size()));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(ElementA) * hA.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(ElementB) * hB.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dZ, hZ.data(), sizeof(ElementC) * hZ.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBroadcast, hBroadcast.data(), sizeof(ElementC) * hBroadcast.size(), cudaMemcpyHostToDevice));

  int batch_count = 1;

  int64_t batch_stride_A = int64_t(M) * K;
  int64_t batch_stride_B = int64_t(K) * N;
  int64_t batch_stride_C = int64_t(M) * N;
  int64_t batch_stride_Z = int64_t(M) * N;
  int64_t batch_stride_Broadcast = int64_t(M);
  int64_t batch_stride_T = int64_t(M) * N;

  int stride_A = K;   // RowMajor A [M, K]
  int stride_B = K;   // ColumnMajor B [K, N]
  int stride_C = N;   // RowMajor C/Z [M, N]
  int stride_Z = N;
  int stride_Broadcast = 0;  // must be zero for broadcast
  int stride_T = N;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      batch_count,
      {1.0f, 0.0f},
      dA,
      dB,
      nullptr,      // C
      dZ,           // Z / output
      dBroadcast,   // broadcast tensor
      nullptr,      // T
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_Z,
      batch_stride_Broadcast,
      batch_stride_T,
      stride_A,
      stride_B,
      stride_C,
      stride_Z,
      stride_Broadcast,
      stride_T
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace));
  CUTLASS_CHECK(gemm_op());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hZ.data(), dZ, sizeof(ElementC) * hZ.size(), cudaMemcpyDeviceToHost));

  std::cout << "Z[0] = " << float(hZ[0]) << "\n";

  if (workspace) cudaFree(workspace);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dZ);
  cudaFree(dBroadcast);

  return 0;
}