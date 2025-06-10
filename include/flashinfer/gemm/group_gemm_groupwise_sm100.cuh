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
#ifndef FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_
#define FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_

#include <cassert>
#include <iterator>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../status.h"
#include "../utils.cuh"

namespace flashinfer {

namespace gemm {

using namespace cute;

template <typename ScaleConfig, typename DTypeIn, typename DTypeSF, typename DTypeOut,
          typename ProblemShape, typename StrideA, typename StrideB, typename StrideC,
          typename LayoutSFA, typename LayoutSFB>
__global__ void compute_sm100_cutlass_group_gemm_args(
    DTypeIn* A, DTypeIn* B, DTypeSF* SFA, DTypeSF* SFB, DTypeOut* C, int* m_indptr, int cum_m,
    int n, int k, int batch_size, int scale_granularity_m, int scale_granularity_n,
    int scale_granularity_k, ProblemShape* problem_sizes, const DTypeIn** A_ptr,
    const DTypeIn** B_ptr, const DTypeSF** SFA_ptr, const DTypeSF** SFB_ptr, const DTypeOut** C_ptr,
    DTypeOut** D_ptr, StrideA* stride_A, StrideB* stride_B, StrideC* stride_C,
    LayoutSFA* layout_SFA, LayoutSFB* layout_SFB) {
  int i = blockIdx.x;
  int m = m_indptr[i + 1] - m_indptr[i];
  problem_sizes[i] = ProblemShape(m, n, k);
  stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  stride_C[i] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(cum_m, n, k, 1));
  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  A_ptr[i] = A + m_indptr[i] * k;
  B_ptr[i] = B + i * k * n;
  C_ptr[i] = C + m_indptr[i] * n;
  D_ptr[i] = C + m_indptr[i] * n;
  SFA_ptr[i] = SFA + m_indptr[i] / scale_granularity_m;
  SFB_ptr[i] = SFB + i * k * n / scale_granularity_n / scale_granularity_k;
}

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, typename DTypeIn,
          typename DTypeOut>
Status CutlassGroupwiseScaledGroupGEMMSM100(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A, DTypeIn* B, float* SFA, float* SFB,
                                            DTypeOut* C, int* m_indptr, int cum_m, int n, int k,
                                            int batch_size, cudaStream_t stream) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementC = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<_256, _128, _128>;
  using ClusterShape_MNK = Shape<_2, _1, _1>;

  using ScaleConfig =
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB, cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  static_assert(
      cute::is_same_v<typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA, LayoutSFA>);
  static_assert(
      cute::is_same_v<typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB, LayoutSFB>);

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes(
      batch_size);

  cutlass::DeviceAllocation<const typename Gemm::ElementA*> A_ptr(batch_size);
  cutlass::DeviceAllocation<const typename Gemm::ElementB*> B_ptr(batch_size);
  cutlass::DeviceAllocation<const typename Gemm::ElementC*> C_ptr(batch_size);
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput*> D_ptr(batch_size);
  cutlass::DeviceAllocation<const ElementAccumulator*> SFA_ptr(batch_size);
  cutlass::DeviceAllocation<const ElementAccumulator*> SFB_ptr(batch_size);

  cutlass::DeviceAllocation<StrideA> stride_A(batch_size);
  cutlass::DeviceAllocation<StrideB> stride_B(batch_size);
  cutlass::DeviceAllocation<StrideC> stride_C(batch_size);
  cutlass::DeviceAllocation<LayoutSFA> layout_SFA(batch_size);
  cutlass::DeviceAllocation<LayoutSFB> layout_SFB(batch_size);

  compute_sm100_cutlass_group_gemm_args<ScaleConfig><<<batch_size, 1, 0, stream>>>(
      A, B, SFA, SFB, C, m_indptr, cum_m, n, k, batch_size, ScaleGranularityM, ScaleGranularityN,
      ScaleGranularityK, problem_sizes.get(), A_ptr.get(), B_ptr.get(), SFA_ptr.get(),
      SFB_ptr.get(), C_ptr.get(), D_ptr.get(), stride_A.get(), stride_B.get(), stride_C.get(),
      layout_SFA.get(), layout_SFB.get());

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host(batch_size);
  problem_sizes.copy_to_host(problem_sizes_host.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {batch_size, problem_sizes.get(), problem_sizes_host.data()},
                                     {
                                         A_ptr.get(),
                                         stride_A.get(),
                                         B_ptr.get(),
                                         stride_B.get(),
                                         SFA_ptr.get(),
                                         layout_SFA.get(),
                                         SFB_ptr.get(),
                                         layout_SFB.get(),
                                     },
                                     {
                                         {},  // epilogue.thread
                                         C_ptr.get(),
                                         stride_C.get(),
                                         D_ptr.get(),
                                         stride_C.get(),
                                     },
                                     hw_info};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 32 * 1024 * 1024, "sm100_groupwise_group_gemm_float_workspace");

  FLASHINFER_CALL(gemm.can_implement(arguments));
  FLASHINFER_CALL(gemm.initialize(arguments, workspace_ptr));
  FLASHINFER_CALL(gemm.run(stream));
  return Status::Success();
}

}  // namespace gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_
