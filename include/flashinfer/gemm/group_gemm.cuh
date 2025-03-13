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
#ifndef FLASHINFER_GEMM_GROUP_GEMM_CUH_
#define FLASHINFER_GEMM_GROUP_GEMM_CUH_

#include <sstream>

#include "../allocator.h"
#include "../cutlass_utils.cuh"

namespace flashinfer {

namespace group_gemm {

#define DISPATCH_WEIGHT_LAYOUT(is_column_major, WEIGHT_LAYOUT, ...) \
  if (is_column_major) {                                            \
    using WEIGHT_LAYOUT = cutlass::layout::ColumnMajor;             \
    __VA_ARGS__                                                     \
  } else {                                                          \
    using WEIGHT_LAYOUT = cutlass::layout::RowMajor;                \
    __VA_ARGS__                                                     \
  }

#define DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, ...) \
  if (smem_limit_per_sm >= 147968) {                             \
    constexpr uint32_t NUM_STAGES = 4;                           \
    __VA_ARGS__                                                  \
  } else {                                                       \
    constexpr uint32_t NUM_STAGES = 2;                           \
    __VA_ARGS__                                                  \
  }

template <typename DType>
cudaError_t CutlassSegmentGEMMRun(void* workspace_buffer, size_t workspace_buffer_size_in_bytes,
                                  void* all_problems, int64_t batch_size, void* x, void* w, void* y,
                                  void* x_ld, void* w_ld, void* y_ld, bool weight_column_major,
                                  cudaStream_t stream) {
  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  int device;
  int smem_limit_per_sm;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  DISPATCH_WEIGHT_LAYOUT(weight_column_major, WEIGHT_LAYOUT, {
    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, {
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          DType,                                   // Element A
          cutlass::layout::RowMajor,               // Layout A
          cutlass::ComplexTransform::kNone,        //
          8,                                       // Granularity A
          DType,                                   // Element B
          WEIGHT_LAYOUT,                           // Layout B
          cutlass::ComplexTransform::kNone,        //
          8,                                       // Granularity B
          DType,                                   // Element C&D
          cutlass::layout::RowMajor,               // Layout C&D
          float,                                   // Element Accumulator
          cutlass::arch::OpClassTensorOp,          // Operator Class Tag
          cutlass::arch::Sm80,                     // Architecture
          cutlass::gemm::GemmShape<128, 128, 32>,  // Thread Block Shape
          cutlass::gemm::GemmShape<64, 64, 32>,    // Warp Shape
          cutlass::gemm::GemmShape<16, 8, 16>,     // Instruction Shape
          cutlass::epilogue::thread::LinearCombination<DType, 8, float, float>,  // Epilogue
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,  // Swizzling Operator
          NUM_STAGES                                                          // Stages
          >::GemmKernel;

      using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
      typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);
      using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
      typename GemmGrouped::Arguments args(
          reinterpret_cast<cutlass::gemm::GemmCoord*>(all_problems), (int)batch_size,
          /*threadblock_count=*/4, epilogue_op, static_cast<DType**>(x), static_cast<DType**>(w),
          static_cast<DType**>(y), static_cast<DType**>(y), reinterpret_cast<int64_t*>(x_ld),
          reinterpret_cast<int64_t*>(w_ld), reinterpret_cast<int64_t*>(y_ld),
          reinterpret_cast<int64_t*>(y_ld));

      GemmGrouped gemm;
      auto status = gemm.initialize(args, nullptr, stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "cutlass group_gemm.initialize failed: " << cutlassGetStatusString(status);
        FLASHINFER_ERROR(err_msg.str());
      }
      status = gemm.run(stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "cutlass group_gemm.run failed: " << cutlassGetStatusString(status);
        FLASHINFER_ERROR(err_msg.str());
      }
    });
  });

  return cudaSuccess;
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUP_GEMM_CUH_
