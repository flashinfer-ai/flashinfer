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

#include <sstream>

#include "../allocator.h"
#include "../utils.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "group_gemm_cutlass.cuh"

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
                                      void* int_buffer, size_t int_buffer_size_in_bytes, DTypeIn* x,
                                      DTypeIn* w, DTypeOut* y, int64_t* xy_indptr_d,
                                      int64_t* w_indices_d, unsigned int batch_size,
                                      unsigned int d_in, unsigned int d_out,
                                      bool weight_column_major, cudaStream_t stream) {
  auto compute_capacity = GetCudaComputeCapability();
  if (compute_capacity.first < 9) {
    std::cerr << "CutlassSegmentGEMMSM90Run requires compute capability of at least 9.0"
              << std::endl;
    return cudaErrorNotSupported;
  } else {
    // Compute capability >= 9.0
    // Reference implementation
    // -
    // https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
    using ProblemShape =
        cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
    using ElementA = DTypeIn;   // Element type for A matrix operand
    using ElementB = DTypeIn;   // Element type for B matrix operand
    using ElementC = DTypeOut;  // Element type for C and D matrix operands

    DISPATCH_WEIGHT_LAYOUT(weight_column_major, WEIGHT_LAYOUT, {
      if constexpr (std::is_same_v<WEIGHT_LAYOUT, cutlass::layout::RowMajor> &&
                    sizeof(DTypeIn) == 1) {
        std::ostringstream err_msg;
        err_msg << "Row-major layout is not supported for fp8 data type";
        throw std::runtime_error(err_msg.str());
      } else {
        using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
        constexpr int AlignmentA =
            128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of
                                                          // elements (up to 16 bytes)

        // B matrix configuration
        using LayoutB = WEIGHT_LAYOUT;  // Layout type for B matrix operand
        constexpr int AlignmentB =
            128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of
                                                          // elements (up to 16 bytes)

        // C/D matrix configuration
        using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
        constexpr int AlignmentC =
            128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of
                                                          // elements (up to 16 bytes)

        constexpr bool is_fp8 = sizeof(DTypeIn) == 1;
        // Core kernel configurations
        using ElementAccumulator = float;     // Element type for internal accumulation
        using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the
                                              // intended feature
        using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
        using TileShape =
            typename std::conditional<is_fp8, Shape<_128, _128, _64>,
                                      Shape<_128, _64, _64>>::type;  // Threadblock-level tile size
        using ClusterShape =
            typename std::conditional<is_fp8, Shape<_2, _2, _1>, Shape<_2, _1, _1>>::
                type;  // Shape of the threadblocks in a cluster
        using StageCountType = cutlass::gemm::collective::StageCountAuto;  // Stage count maximized
                                                                           // based on the tile size
        using KernelSchedule = typename std::conditional<
            is_fp8, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::type;  // Kernel to launch
        using EpilogueSchedule =
            cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;  // Epilogue to launch

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

        // Reference device GEMM implementation type
        using DeviceGemmReference =
            cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
                                             LayoutC, ElementAccumulator, ElementAccumulator>;

        using StrideA = typename Gemm::GemmKernel::InternalStrideA;
        using StrideB = typename Gemm::GemmKernel::InternalStrideB;
        using StrideC = typename Gemm::GemmKernel::InternalStrideC;
        using StrideD = typename Gemm::GemmKernel::InternalStrideD;

        AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);
        ProblemShape::UnderlyingProblemShape* problem_sizes_device =
            allocator.aligned_alloc<ProblemShape::UnderlyingProblemShape>(
                batch_size * sizeof(ProblemShape::UnderlyingProblemShape), 16,
                "problem_sizes_device");
        DTypeIn** x_data =
            allocator.aligned_alloc<DTypeIn*>(batch_size * sizeof(DTypeIn*), 16, "x_data");
        DTypeIn** w_data =
            allocator.aligned_alloc<DTypeIn*>(batch_size * sizeof(DTypeIn*), 16, "w_data");
        DTypeOut** y_data =
            allocator.aligned_alloc<DTypeOut*>(batch_size * sizeof(DTypeOut*), 16, "y_data");
        StrideA* x_stride =
            allocator.aligned_alloc<StrideA>(batch_size * sizeof(StrideA), 16, "x_stride");
        StrideB* w_stride =
            allocator.aligned_alloc<StrideB>(batch_size * sizeof(StrideB), 16, "w_stride");
        StrideC* y_stride =
            allocator.aligned_alloc<StrideC>(batch_size * sizeof(StrideC), 16, "y_stride");

        cutlass::KernelHardwareInfo hw_info;
        cudaGetDevice(&hw_info.device_id);
        hw_info.sm_count =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

        typename Gemm::EpilogueOutputOp::Params params;
        // TODO(Zihao): support block alpha and beta
        params = typename Gemm::EpilogueOutputOp::Params(/*alpha=*/ElementAccumulator(1.f),
                                                         /*beta=*/ElementAccumulator(0.f));

        typename Gemm::Arguments arguments;

        arguments = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {int(batch_size), problem_sizes_device, nullptr},
            {const_cast<const DTypeIn**>(x_data), x_stride, const_cast<const DTypeIn**>(w_data),
             w_stride},
            {params, const_cast<const DTypeIn**>(y_data), y_stride, y_data, y_stride},
            hw_info};

        compute_sm90_cutlass_group_gemm_args<<<batch_size, 1, 0, stream>>>(
            problem_sizes_device, x_data, w_data, y_data, x_stride, w_stride, y_stride, (DTypeIn*)x,
            (DTypeIn*)w, (DTypeOut*)y, xy_indptr_d, w_indices_d, d_in, d_out, weight_column_major);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cerr << "Failed to launch compute_sm90_cutlass_group_gemm_args kernel: "
                    << cudaGetErrorString(err) << std::endl;
          return err;
        }

        // Initialize the gemm kernel
        Gemm gemm;

        // Using the arguments, query for extra workspace required for matrix multiplication
        // computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
        auto workspace_ptr = float_allocator.aligned_alloc<void>(workspace_size, 64,
                                                                 "sm90_group_gemm_float_workspace");

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));

        // Correctness / Warmup iteration
        CUTLASS_CHECK(gemm.run());  // Warmup
      }
    });
  }

  return cudaSuccess;
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUP_GEMM_SM90_CUH_
