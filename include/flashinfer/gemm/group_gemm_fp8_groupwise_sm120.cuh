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
#ifndef FLASHINFER_GROUP_GEMM_FP8_GROUPWISE_SM120_CUH_
#define FLASHINFER_GROUP_GEMM_FP8_GROUPWISE_SM120_CUH_

#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include <cassert>
#include <cute/tensor.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <iterator>
#include <type_traits>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace group_gemm {

// Kernel for argument preparation with blockwise scaling
template <typename ScaleConfig, typename ElementA, typename ElementSF, typename ElementD,
          typename ProblemShape, typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, bool ScaleMajorK>
__global__ void compute_sm120_cutlass_group_gemm_args(
    ElementA* A, ElementA* B, ElementSF* SFA, ElementSF* SFB, ElementD* D, int* m_indptr, int max_m,
    int n, int k, int num_groups, int scale_granularity_m, int scale_granularity_n,
    int scale_granularity_k, ProblemShape* problem_sizes, ElementA const** A_ptr,
    ElementA const** B_ptr, ElementSF const** SFA_ptr, ElementSF const** SFB_ptr, ElementD** D_ptr,
    StrideA* stride_A, StrideB* stride_B, StrideD* stride_D, LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_groups) return;

  const int sf_n = (n + scale_granularity_n - 1) / scale_granularity_n;
  const int sf_k = (k + scale_granularity_k - 1) / scale_granularity_k;

  int m_offset = m_indptr[i];
  int m_offset_next = m_indptr[i + 1];
  int m = m_offset_next - m_offset;
  int sf_m_offset = m_offset / scale_granularity_m;

  problem_sizes[i] = ProblemShape(m, n, k);
  stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  A_ptr[i] = A + int64_t(m_offset) * int64_t(k);
  B_ptr[i] = B + int64_t(i) * int64_t(n) * int64_t(k);
  D_ptr[i] = D + int64_t(m_offset) * int64_t(n);

  if constexpr (ScaleMajorK) {
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    SFA_ptr[i] = SFA + int64_t(sf_m_offset) * int64_t(sf_k);
  } else {
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m, n, k, 1));
    SFA_ptr[i] = SFA + int64_t(sf_m_offset);
  }
  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  SFB_ptr[i] = SFB + int64_t(i) * int64_t(sf_n) * int64_t(sf_k);
}

using namespace cute;

// SM120 uses Cooperative schedule with 128x128x128 tile shape
template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          typename DTypeIn, typename DTypeOut>
cudaError_t CutlassFP8GroupwiseScaledGroupGEMMSM120(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeIn* A, DTypeIn* B, float* SFA, float* SFB, DTypeOut* D,
    int* m_indptr, int max_m, int n, int k, int num_groups, cudaStream_t stream) {
  // SM120 only supports these specific scale granularities
  static_assert(ScaleGranularityM == 1 || ScaleGranularityM == 128,
                "SM120 only supports ScaleGranularityM = 1 or 128");
  static_assert(ScaleGranularityN == 128, "SM120 only supports ScaleGranularityN = 128");
  static_assert(ScaleGranularityK == 128, "SM120 only supports ScaleGranularityK = 128");
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutD = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;  // Memory access granularity/alignment of D
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentC = 0;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // SM120 uses fixed 128x128x128 tile shape (Cooperative schedule)
  using MmaTileShape_MNK = Shape<_128, _128, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  // Define blockwise scale configuration
  // SM120's Sm120BlockwiseScaleConfig takes UMMA::Major parameters based on ScaleMajorK
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  // Use decltype like SM100 does for consistency
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // For SM120 grouped GEMM, use auto schedule for epilogue
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC*, AlignmentC, ElementD, LayoutD*, AlignmentD, EpilogueSchedule>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB, cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK, StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);

  auto problem_sizes = allocator.aligned_alloc<typename ProblemShape::UnderlyingProblemShape>(
      num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape), 16,
      "sm120_groupwise_group_gemm_problem_sizes");
  auto A_ptr = allocator.aligned_alloc<const ElementA*>(num_groups * sizeof(const ElementA*), 16,
                                                        "sm120_groupwise_group_gemm_A_ptr");
  auto B_ptr = allocator.aligned_alloc<const ElementB*>(num_groups * sizeof(const ElementB*), 16,
                                                        "sm120_groupwise_group_gemm_B_ptr");
  auto D_ptr = allocator.aligned_alloc<ElementD*>(num_groups * sizeof(ElementD*), 16,
                                                  "sm120_groupwise_group_gemm_D_ptr");
  auto SFA_ptr = allocator.aligned_alloc<const ElementAccumulator*>(
      num_groups * sizeof(const ElementAccumulator*), 16, "sm120_groupwise_group_gemm_SFA_ptr");
  auto SFB_ptr = allocator.aligned_alloc<const ElementAccumulator*>(
      num_groups * sizeof(const ElementAccumulator*), 16, "sm120_groupwise_group_gemm_SFB_ptr");

  auto stride_A = allocator.aligned_alloc<StrideA>(num_groups * sizeof(StrideA), 16,
                                                   "sm120_groupwise_group_gemm_stride_A");
  auto stride_B = allocator.aligned_alloc<StrideB>(num_groups * sizeof(StrideB), 16,
                                                   "sm120_groupwise_group_gemm_stride_B");
  auto stride_D = allocator.aligned_alloc<StrideD>(num_groups * sizeof(StrideD), 16,
                                                   "sm120_groupwise_group_gemm_stride_D");
  auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(num_groups * sizeof(LayoutSFA), 16,
                                                       "sm120_groupwise_group_gemm_layout_SFA");
  auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(num_groups * sizeof(LayoutSFB), 16,
                                                       "sm120_groupwise_group_gemm_layout_SFB");

  int num_threads = std::min(num_groups, 1024);
  int num_blocks = (num_groups + num_threads - 1) / num_threads;
  cudaLaunchConfig_t config;
  config.gridDim = num_blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;

  // Argument preparation with blockwise scaling
  auto prepare_args_kernel =
      compute_sm120_cutlass_group_gemm_args<ScaleConfig, ElementA, float, ElementD,
                                            ProblemShape::UnderlyingProblemShape, StrideA, StrideB,
                                            StrideD, LayoutSFA, LayoutSFB, ScaleMajorK>;

  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
      &config, prepare_args_kernel, A, B, SFA, SFB, D, m_indptr, max_m, n, k, num_groups,
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, problem_sizes, A_ptr, B_ptr, SFA_ptr,
      SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA, layout_SFB));

  thread_local int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {num_groups, problem_sizes, /*problem_sizes_host=*/nullptr},
                                     {
                                         A_ptr,
                                         stride_A,
                                         B_ptr,
                                         stride_B,
                                         SFA_ptr,
                                         layout_SFA,
                                         SFB_ptr,
                                         layout_SFB,
                                     },
                                     {
                                         {},       // epilogue.thread
                                         nullptr,  // C_ptr
                                         nullptr,  // stride_C
                                         D_ptr,
                                         stride_D,
                                     },
                                     hw_info};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 16, "sm120_groupwise_group_gemm_float_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true));
  return cudaSuccess;
#else
  return cudaErrorNotSupported;
#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
}

}  // namespace group_gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_FP8_GROUPWISE_SM120_CUH_
