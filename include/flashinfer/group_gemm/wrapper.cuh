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

#include "handler.cuh"

namespace flashinfer {

namespace group_gemm {

template <typename DType>
cudaError_t CutlassSegmentGEMMWrapper(CutlassSegmentGEMMHandler* handler, DType* x, DType* w,
                                      DType* y, cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  if (handler->IsProblemRegistered()) {
    using cutlass::epilogue::thread::LinearCombination;
    using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
    if (!handler->IsWeightColumnMajor()) {
      // TODO(Zihao): investigate the difference between GroupScheduleMode::kDeviceOnly and
      // GroupScheduleMode::kHostPrecompute
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          cutlass_t,                              // Element A
          cutlass::layout::RowMajor,              // Layout A
          cutlass::ComplexTransform::kNone,       //
          8,                                      // Granularity A
          cutlass_t,                              // Element B
          cutlass::layout::RowMajor,              // Layout B
          cutlass::ComplexTransform::kNone,       //
          8,                                      // Granularity B
          cutlass_t,                              // Element C&D
          cutlass::layout::RowMajor,              // Layout C&D
          float,                                  // Element Accumulator
          cutlass::arch::OpClassTensorOp,         // Operator Class Tag
          cutlass::arch::Sm80,                    // Architecture
          cutlass::gemm::GemmShape<32, 128, 16>,  // Thread Block Shape
          cutlass::gemm::GemmShape<32, 64, 16>,   // Warp Shape
          cutlass::gemm::GemmShape<16, 8, 8>,     // Instruction Shape
          cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
          GemmIdentityThreadblockSwizzle<1>,                     // Swizzling Operator
          2,                                                     // Stages
          cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly  // Group Schedule Mode
          >::GemmKernel;

      using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
      typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0);

      using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
      typename GemmGrouped::Arguments args(
          handler->GetProblemSizes(), handler->GetBatchSize(), 512, epilogue_op,
          handler->GetXPtr<cutlass_t>(), handler->GetWPtr<cutlass_t>(),
          handler->GetYPtr<cutlass_t>(), handler->GetYPtr<cutlass_t>(), handler->GetLdX(),
          handler->GetLdW(), handler->GetLdY(), handler->GetLdY());

      GemmGrouped gemm;
      auto status = gemm.initialize(args, nullptr, stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "sgmv_cutlass gemm.initialize failed: " << cutlassGetStatusString(status);
        throw std::runtime_error(err_msg.str());
      }
      status = gemm.run(stream);
      if (status != cutlass::Status::kSuccess) {
        std::ostringstream err_msg;
        err_msg << "sgmv_cutlass gemm.run failed: " << cutlassGetStatusString(status);
        throw std::runtime_error(err_msg.str());
      }
    } else {
      std::ostringstream err_msg;
      // TODO: support column-major weight matrix
      err_msg << "CutlassSegmentGEMMWrapper only supports row-major weight matrix";
      throw std::runtime_error(err_msg.str());
    }
  } else {
    std::ostringstream err_msg;
    err_msg << "Please call CutlassSegmentGEMMHandler's RegisterProblem() before calling "
               "BatchDecodeWithPagedKVCacheWrapper()";
    throw std::runtime_error(err_msg.str());
  }
  return cudaSuccess;
}

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_WRAPPER_CUH_
