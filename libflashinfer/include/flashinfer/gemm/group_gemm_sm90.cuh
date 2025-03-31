/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_GROUP_GEMM_SM90_CUH_
#define FLASHINFER_GEMM_GROUP_GEMM_SM90_CUH_

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace group_gemm {

using namespace cute;

#define DISPATCH_WEIGHT_LAYOUT(is_column_major, WEIGHT_LAYOUT, ...) \
  if (is_column_major) {                                            \
    using WEIGHT_LAYOUT = cutlass::layout::ColumnMajor;             \
    __VA_ARGS__                                                     \
  } else {                                                          \
    using WEIGHT_LAYOUT = cutlass::layout::RowMajor;                \
    __VA_ARGS__                                                     \
  }

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template <typename DTypeIn, typename DTypeOut>
cudaError_t CutlassSegmentGEMMSM90Run(void* float_buffer, size_t float_buffer_size_in_bytes,
                                      void* int_buffer, size_t int_buffer_size_in_bytes,
                                      void* all_problems, int64_t batch_size, void* x, void* w,
                                      void* y, void* x_stride, void* w_stride, void* y_stride,
                                      bool weight_column_major, cudaStream_t stream) {
  auto compute_capacity = GetCudaComputeCapability();
  if (compute_capacity.first < 9) {
    std::cerr << "CutlassSegmentGEMMSM90Run requires compute capability of at least 9.0"
              << std::endl;
    return cudaErrorNotSupported;
  }

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using ElementA = DTypeIn;
  using ElementB = DTypeIn;
  using ElementC = DTypeOut;

  DISPATCH_WEIGHT_LAYOUT(weight_column_major, WEIGHT_LAYOUT, {
    if constexpr (std::is_same_v<WEIGHT_LAYOUT, cutlass::layout::RowMajor> &&
                  sizeof(DTypeIn) == 1) {
      std::ostringstream err_msg;
      err_msg << "Row-major layout is not supported for fp8 data type";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      using LayoutA = cutlass::layout::RowMajor;
      constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

      using LayoutB = WEIGHT_LAYOUT;
      constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

      using LayoutC = cutlass::layout::RowMajor;
      constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

      constexpr bool is_fp8 = sizeof(DTypeIn) == 1;

      using ElementAccumulator = float;
      using ArchTag = cutlass::arch::Sm90;
      using OperatorClass = cutlass::arch::OpClassTensorOp;
      using TileShape =
          typename std::conditional<is_fp8, Shape<_256, _128, _128>, Shape<_128, _128, _128>>::type;
      using ClusterShape =
          typename std::conditional<is_fp8, Shape<_2, _2, _1>, Shape<_2, _1, _1>>::type;
      using StageCountType = cutlass::gemm::collective::StageCountAuto;
      using KernelSchedule = typename std::conditional<
          is_fp8, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::type;
      using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

      using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
          ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC,
          EpilogueSchedule>::CollectiveOp;

      using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

      using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                              CollectiveEpilogue>;
      using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

      using StrideA = typename Gemm::GemmKernel::InternalStrideA;
      using StrideB = typename Gemm::GemmKernel::InternalStrideB;
      using StrideC = typename Gemm::GemmKernel::InternalStrideC;
      using StrideD = typename Gemm::GemmKernel::InternalStrideD;

      cutlass::KernelHardwareInfo hw_info;
      cudaGetDevice(&hw_info.device_id);
      hw_info.sm_count =
          cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

      typename Gemm::EpilogueOutputOp::Params params;
      params =
          typename Gemm::EpilogueOutputOp::Params(ElementAccumulator(1.f), ElementAccumulator(0.f));

      typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGrouped,
          {int(batch_size), reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(all_problems),
           nullptr},
          {static_cast<const DTypeIn**>(x), reinterpret_cast<StrideA*>(x_stride),
           static_cast<const DTypeIn**>(w), reinterpret_cast<StrideB*>(w_stride)},
          {params, static_cast<const DTypeOut**>(y), reinterpret_cast<StrideC*>(y_stride),
           static_cast<DTypeOut**>(y), reinterpret_cast<StrideD*>(y_stride)},
          hw_info};

      Gemm gemm;

      size_t workspace_size = Gemm::get_workspace_size(arguments);
      AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
      auto workspace_ptr = float_allocator.aligned_alloc<void>(workspace_size, 64,
                                                               "sm90_group_gemm_float_workspace");

      CUTLASS_CHECK(gemm.can_implement(arguments));
      CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
      CUTLASS_CHECK(gemm.run(stream));
    }
  });

  return cudaSuccess;
}

}  // namespace group_gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUP_GEMM_SM90_CUH_
