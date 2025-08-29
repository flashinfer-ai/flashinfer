/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
#define FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_

#include <cassert>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <typeinfo>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include <flashinfer/allocator.h>

namespace flashinfer {
namespace gemm {

// SM120 implementation with fixed scale granularity following CUTLASS examples
// For SM120, only ScaleGranularityM = 1, ScaleGranularityN = 128, ScaleGranularityK = 128 are supported
template<int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK, 
         int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120(
    void* float_buffer, size_t float_buffer_size_in_bytes,
    DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr, float* SFB_ptr, DTypeOut* D_ptr,
    int m, int n, int k, int l, cudaStream_t stream) {
  // SM120 only supports these specific scale granularities (per CUTLASS examples)
  static_assert(ScaleGranularityM == 1, "SM120 only supports ScaleGranularityM = 1");
  static_assert(ScaleGranularityN == 128 || ScaleGranularityN == 64 || ScaleGranularityN == 32 || ScaleGranularityN == 16, 
                "SM120 only supports ScaleGranularityN = 128, 64, 32 or 16");
  static_assert(ScaleGranularityK == 128, "SM120 only supports ScaleGranularityK = 128");
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;
  
  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = DTypeOut;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // Use different tile shapes based on MmaSM like CUTLASS examples
  // MmaSM=1 uses 128x128x128, MmaSM=2 uses smaller tile to avoid register spilling
  using MmaTileShape_MNK = std::conditional_t<MmaSM == 1,
                                                Shape<_128, _128, _128>,
                                                Shape<_64, _128, _128>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  // Define blockwise scale configuration following CUTLASS pattern
  // SM120 needs explicit ScaleMajorK handling like SM100
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                                                  cute::UMMA::Major::K, cute::UMMA::Major::K>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN>>;

  // Use decltype like SM100 does for consistency
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // SM120 blockwise scaling uses auto stage count with epilogue carveout
  // This matches the CUTLASS example configuration
  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  // For beta=0 case, C and D can be the same buffer
  // This avoids allocating extra memory from workspace
  DTypeOut* C_ptr = D_ptr;  // Use D as both input and output when beta=0
  
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
      {{}, C_ptr, stride_C, D_ptr, stride_D}};  // C and D point to same buffer when beta=0
  
  // Set alpha and beta for the epilogue
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  // Debug print arguments - match CUTLASS example format
  printf("\n=== FlashInfer SM120 Kernel Configuration ===\n");
  printf("Problem shape: m=%d, n=%d, k=%d, l=%d\n", m, n, k, l);
  printf("Data types:\n");
  printf("  ElementA: %s (size=%d bits)\n", typeid(ElementA).name(), cutlass::sizeof_bits<ElementA>::value);
  printf("  ElementB: %s (size=%d bits)\n", typeid(ElementB).name(), cutlass::sizeof_bits<ElementB>::value);
  printf("  ElementC: %s (size=%d bits)\n", typeid(ElementC).name(), cutlass::sizeof_bits<ElementC>::value);
  printf("  ElementD: %s (size=%d bits)\n", typeid(ElementD).name(), cutlass::sizeof_bits<ElementD>::value);
  printf("  ElementAccumulator: %s\n", typeid(ElementAccumulator).name());
  printf("Scale Configuration:\n");
  printf("  ScaleGranularity: M=%d, N=%d, K=%d\n", ScaleGranularityM, ScaleGranularityN, ScaleGranularityK);
  printf("  ScaleMajorK: %s\n", ScaleMajorK ? "true" : "false");
  printf("  SFA dimensions: %dx%d\n", m/ScaleGranularityM, k/ScaleGranularityK);
  printf("  SFB dimensions: %dx%d\n", k/ScaleGranularityK, n/ScaleGranularityN);
  printf("Kernel Configuration:\n");
  printf("  MmaSM: %d\n", MmaSM);
  printf("  TileShape: %dx%dx%d\n", MmaSM == 1 ? 128 : 64, 128, 128);
  printf("  ClusterShape: 1x1x1\n");
  printf("Pointers:\n");
  printf("  A=%p, B=%p\n", A_ptr, B_ptr);
  printf("  C=%p, D=%p\n", C_ptr, D_ptr);
  printf("  SFA=%p, SFB=%p\n", SFA_ptr, SFB_ptr);
  printf("Layouts and Strides:\n");
  std::cout << "  stride_A: " << stride_A << std::endl;
  std::cout << "  stride_B: " << stride_B << std::endl;
  std::cout << "  stride_C: " << stride_C << std::endl;
  std::cout << "  stride_D: " << stride_D << std::endl;
  std::cout << "  layout_SFA: " << layout_SFA << std::endl;
  std::cout << "  layout_SFB: " << layout_SFB << std::endl;
  printf("Epilogue Configuration:\n");
  printf("  alpha=%.6f, beta=%.6f\n", arguments.epilogue.thread.alpha, arguments.epilogue.thread.beta);
  printf("  Mode: %s\n", arguments.mode == cutlass::gemm::GemmUniversalMode::kGemm ? "kGemm" : "Other");
  
  // Print first few values to verify data
  float h_sfa[4], h_sfb[4];
  cutlass::float_e4m3_t h_a[4];
  cutlass::float_e4m3_t h_b[4];
  int sfa_count = std::min(4, (m/ScaleGranularityM)*(k/ScaleGranularityK));
  int sfb_count = std::min(4, (k/ScaleGranularityK)*(n/ScaleGranularityN));
  cudaMemcpy(h_sfa, SFA_ptr, sizeof(float) * sfa_count, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sfb, SFB_ptr, sizeof(float) * sfb_count, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a, A_ptr, sizeof(cutlass::float_e4m3_t) * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, B_ptr, sizeof(cutlass::float_e4m3_t) * 4, cudaMemcpyDeviceToHost);
  
  printf("Tensor Values (first 4 elements):\n");
  printf("  A: [%.6f, %.6f, %.6f, %.6f]\n", 
         float(h_a[0]), float(h_a[1]), float(h_a[2]), float(h_a[3]));
  printf("  B: [%.6f, %.6f, %.6f, %.6f]\n",
         float(h_b[0]), float(h_b[1]), float(h_b[2]), float(h_b[3]));
  printf("  SFA: [");
  for (int i = 0; i < sfa_count; i++) printf("%.6f%s", h_sfa[i], i < sfa_count-1 ? ", " : "");
  printf("]\n");
  printf("  SFB: [");
  for (int i = 0; i < sfb_count; i++) printf("%.6f%s", h_sfb[i], i < sfb_count-1 ? ", " : "");
  printf("]\n");

  printf("===================================\n");

  // Check device compute capability first
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);
  printf("Device: %s (SM %d.%d)\n", props.name, props.major, props.minor);
  
  Gemm gemm;
  
  // Print kernel type info for debugging
  printf("Kernel info: ElementA=%s, ElementB=%s, ElementC=%s\n",
         typeid(typename Gemm::ElementA).name(),
         typeid(typename Gemm::ElementB).name(), 
         typeid(typename Gemm::ElementC).name());
  // MmaTileShape depends on MmaSM value
  if constexpr (MmaSM == 1) {
    printf("MmaTileShape: M=128, N=128, K=128 (MmaSM=1)\n");
  } else {
    printf("MmaTileShape: M=64, N=128, K=128 (MmaSM=2)\n");
  }
  printf("CollectiveMainloop TileShape: (may not print correctly due to template instantiation)\n");
  
  cutlass::Status status = gemm.can_implement(arguments);
  printf("gemm.can_implement status: %d (0=success)\n", (int)status);
  if (status != cutlass::Status::kSuccess) {
    printf("ERROR: Kernel cannot implement these arguments with status %d!\n", (int)status);
    return cudaErrorNotSupported;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  
  printf("Workspace check: kernel needs %zu bytes, available=%zu\n", 
         workspace_size, float_buffer_size_in_bytes);
  
  if (workspace_size > float_buffer_size_in_bytes) {
    printf("ERROR: Insufficient workspace. Need %zu bytes, have %zu bytes\n", 
           workspace_size, float_buffer_size_in_bytes);
    return cudaErrorInsufficientDriver;
  }

  // Pass workspace pointer only if needed
  void* kernel_workspace = nullptr;
  if (workspace_size > 0) {  // Only provide a pointer if workspace is actually needed
    kernel_workspace = float_buffer;
  }
  
  // Debug: print ALL arguments right before kernel initialization
  printf("\n=== RIGHT BEFORE gemm.initialize() ===\n");
  printf("Arguments structure size: %zu bytes\n", sizeof(arguments));
  printf("Gemm::Arguments type: %s\n", typeid(typename Gemm::Arguments).name());
  printf("Arguments structure:\n");
  printf("  mode: %s\n", arguments.mode == cutlass::gemm::GemmUniversalMode::kGemm ? "kGemm" : "Other");
  printf("  problem_shape: {%d, %d, %d, %d}\n", 
         cute::get<0>(arguments.problem_shape), cute::get<1>(arguments.problem_shape), 
         cute::get<2>(arguments.problem_shape), cute::get<3>(arguments.problem_shape));
  printf("  mainloop.ptr_A: %p\n", arguments.mainloop.ptr_A);
  printf("  mainloop.ptr_B: %p\n", arguments.mainloop.ptr_B);
  printf("  mainloop.ptr_SFA: %p\n", arguments.mainloop.ptr_SFA);
  printf("  mainloop.ptr_SFB: %p\n", arguments.mainloop.ptr_SFB);
  printf("  epilogue.ptr_C: %p\n", arguments.epilogue.ptr_C);
  printf("  epilogue.ptr_D: %p\n", arguments.epilogue.ptr_D);
  printf("  epilogue.thread.alpha: %.6f\n", arguments.epilogue.thread.alpha);
  printf("  epilogue.thread.beta: %.6f\n", arguments.epilogue.thread.beta);
  printf("  kernel_workspace: %p\n", kernel_workspace);
  printf("  workspace_size: %zu bytes\n", workspace_size);
  printf("  float_buffer: %p (total size=%zu)\n", float_buffer, float_buffer_size_in_bytes);
  // Note: Strides and layouts already printed earlier
  
  status = gemm.initialize(arguments, kernel_workspace);
  printf("gemm.initialize status: %d (0=success)\n", (int)status);
  if (status != cutlass::Status::kSuccess) {
    printf("ERROR: Kernel initialization failed with status %d!\n", (int)status);
    const char* error_str = "Unknown error";
    switch(status) {
      case cutlass::Status::kErrorMisalignedOperand: error_str = "Misaligned Operand"; break;
      case cutlass::Status::kErrorInvalidDataType: error_str = "Invalid Data Type"; break;
      case cutlass::Status::kErrorInvalidLayout: error_str = "Invalid Layout"; break;
      case cutlass::Status::kErrorInvalidProblem: error_str = "Invalid Problem"; break;
      case cutlass::Status::kErrorNotSupported: error_str = "Not Supported"; break;
      case cutlass::Status::kErrorInternal: error_str = "Internal Error"; break;
      case cutlass::Status::kErrorArchMismatch: error_str = "Architecture Mismatch"; break;
      case cutlass::Status::kErrorInsufficientDriver: error_str = "Insufficient Driver"; break;
      case cutlass::Status::kErrorMemoryAllocation: error_str = "Memory Allocation Failed"; break;
    }
    printf("Error details: %s\n", error_str);
    
    // Don't continue if initialization failed
    return cudaErrorNotSupported;
  }

  printf("\n=== RIGHT BEFORE gemm.run() ===\n");
  printf("Stream: %p\n", stream);
  printf("Gemm object address: %p\n", &gemm);
  // Re-check key pointers to ensure nothing changed
  printf("Verifying arguments still intact:\n");
  printf("  mainloop.ptr_A: %p\n", arguments.mainloop.ptr_A);
  printf("  mainloop.ptr_B: %p\n", arguments.mainloop.ptr_B);
  printf("  mainloop.ptr_SFA: %p\n", arguments.mainloop.ptr_SFA);
  printf("  mainloop.ptr_SFB: %p\n", arguments.mainloop.ptr_SFB);
  printf("  epilogue.ptr_C: %p\n", arguments.epilogue.ptr_C);
  printf("  epilogue.ptr_D: %p\n", arguments.epilogue.ptr_D);
  
  status = gemm.run(stream);
  printf("gemm.run status: %d (0=success)\n", (int)status);
  if (status != cutlass::Status::kSuccess) {
    printf("ERROR: Kernel execution failed with status %d!\n", (int)status);
    return cudaErrorUnknown;
  }
  
  // Sync to ensure kernel completes
  cudaError_t cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    printf("ERROR: CUDA sync failed: %s\n", cudaGetErrorString(cuda_err));
    return cuda_err;
  }
  printf("Kernel execution completed successfully\n");

  return cudaSuccess;
#else
  return cudaErrorNotSupported;
#endif
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_