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
#ifndef FLASHINFER_GROUP_GEMM_WRAPPER_CUH_
#define FLASHINFER_GROUP_GEMM_WRAPPER_CUH_

#include <sstream>

#include "../allocator.h"
#include "handler.cuh"

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

template <typename DType>
cudaError_t CutlassSegmentGEMMWrapper(CutlassSegmentGEMMHandler* handler, DType* x, DType* w,
                                      DType* y, int64_t* xy_indptr_d, int64_t* w_indices_d,
                                      unsigned int batch_size, unsigned int d_in,
                                      unsigned int d_out, bool weight_column_major,
                                      cudaStream_t stream) {
  AlignedAllocator allocator(handler->GetWorkspace(), handler->GetWorkspaceSizeInBytes());
  cutlass::gemm::GemmCoord* problem_sizes_device =
      allocator.aligned_alloc<cutlass::gemm::GemmCoord>(
          batch_size * sizeof(cutlass::gemm::GemmCoord), 16, "problem_sizes_device");
  DType** x_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "x_data");
  DType** w_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "w_data");
  DType** y_data = allocator.aligned_alloc<DType*>(batch_size * sizeof(DType*), 16, "y_data");
  int64_t* ld_x = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_x");
  int64_t* ld_w = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_w");
  int64_t* ld_y = allocator.aligned_alloc<int64_t>(batch_size * sizeof(int64_t), 16, "ld_y");

  // NOTE(Zihao): I didn't successfully launch the kernel with cudaLaunchKernel API,
  // so I just use the kernel function directly, need to investigate more.
  auto compute_args_kernel = compute_cutlass_group_gemm_args<DType>;
  compute_args_kernel<<<batch_size, 1, 0, stream>>>(
      problem_sizes_device, x_data, w_data, y_data, ld_x, ld_w, ld_y, (DType*)x, (DType*)w,
      (DType*)y, xy_indptr_d, w_indices_d, d_in, d_out, weight_column_major);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
    return err;
  }

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  DISPATCH_WEIGHT_LAYOUT(weight_column_major, WEIGHT_LAYOUT, {
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
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,     // Swizzling Operator
        8                                                                      // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(problem_sizes_device, batch_size, 4, epilogue_op, x_data,
                                         w_data, y_data, y_data, ld_x, ld_w, ld_y, ld_y);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      std::ostringstream err_msg;
      err_msg << "cutlass group_gemm.initialize failed: " << cutlassGetStatusString(status);
      throw std::runtime_error(err_msg.str());
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      std::ostringstream err_msg;
      err_msg << "cutlass group_gemm.run failed: " << cutlassGetStatusString(status);
      throw std::runtime_error(err_msg.str());
    }
  });

  return cudaSuccess;
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_WRAPPER_CUH_
