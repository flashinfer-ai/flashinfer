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

#include <flashinfer/allocator.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <typeinfo>

#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {

template <typename Gemm, typename ScaleConfig, typename DTypeIn, typename DTypeOut>
cudaError_t RunSm120GroupwiseGemm(void* float_buffer, size_t float_buffer_size_in_bytes,
                                  DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr, float* SFB_ptr,
                                  DTypeOut* D_ptr, int m, int n, int k, int l,
                                  cudaStream_t stream) {
  using namespace cute;

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

  DTypeOut* C_ptr = D_ptr;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
      {{}, C_ptr, stride_C, D_ptr, stride_D}};

  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > float_buffer_size_in_bytes) {
    return cudaErrorInsufficientDriver;
  }

  void* kernel_workspace = nullptr;
  if (workspace_size > 0) {
    kernel_workspace = float_buffer;
  }

  cutlass::Status status = gemm.initialize(arguments, kernel_workspace);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

template <typename PingpongGemm, typename CooperativeGemm, typename ScaleConfig, typename DTypeIn,
          typename DTypeOut>
cudaError_t TrySm120PingpongThenCooperative(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
  auto status = RunSm120GroupwiseGemm<PingpongGemm, ScaleConfig>(
      float_buffer, float_buffer_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, D_ptr, m, n, k, l,
      stream);
  if (status == cudaErrorNotSupported || status == cudaErrorInsufficientDriver) {
    return RunSm120GroupwiseGemm<CooperativeGemm, ScaleConfig>(
        float_buffer, float_buffer_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, D_ptr, m, n, k, l,
        stream);
  }

  return status;
}

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
  // SM120/SM121 only supports these specific scale granularities
  static_assert(ScaleGranularityM == 1 || ScaleGranularityM == 128,
                "SM120/SM121 only supports ScaleGranularityM = 1 or 128");
  static_assert(ScaleGranularityN == 128, "SM120/SM121 only supports ScaleGranularityN = 128");
  static_assert(ScaleGranularityK == 128, "SM120/SM121 only supports ScaleGranularityK = 128");
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

  using CooperativeMmaTileShape_MNK = Shape<_128, _128, _128>;
  using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  constexpr bool kCanUsePingpong = (64 % ScaleGranularityM == 0);

  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CooperativeCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CooperativeMmaTileShape_MNK,
      ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
      ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CooperativeStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CooperativeCollectiveEpilogue::SharedStorage))>;

  using CooperativeCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, CooperativeMmaTileShape_MNK, ClusterShape_MNK,
      CooperativeStageCount, cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using CooperativeGemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CooperativeCollectiveMainloop,
                                           CooperativeCollectiveEpilogue, void>;

  using CooperativeGemm = cutlass::gemm::device::GemmUniversalAdapter<CooperativeGemmKernel>;

  if constexpr (kCanUsePingpong) {
    using PingpongCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, PingpongMmaTileShape_MNK,
        ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using PingpongStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename PingpongCollectiveEpilogue::SharedStorage))>;

    using PingpongCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
        cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
        AlignmentB, ElementAccumulator, PingpongMmaTileShape_MNK, ClusterShape_MNK,
        PingpongStageCount,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::CollectiveOp;

    using PingpongGemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, PingpongCollectiveMainloop,
                                             PingpongCollectiveEpilogue, void>;

    using PingpongGemm = cutlass::gemm::device::GemmUniversalAdapter<PingpongGemmKernel>;

    return TrySm120PingpongThenCooperative<PingpongGemm, CooperativeGemm, ScaleConfig>(
        float_buffer, float_buffer_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, D_ptr, m, n, k, l,
        stream);
  } else {
    return RunSm120GroupwiseGemm<CooperativeGemm, ScaleConfig>(
        float_buffer, float_buffer_size_in_bytes, A_ptr, B_ptr, SFA_ptr, SFB_ptr, D_ptr, m, n, k, l,
        stream);
  }
#else
  return cudaErrorNotSupported;
#endif
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
