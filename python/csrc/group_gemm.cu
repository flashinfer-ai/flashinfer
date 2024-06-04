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
#include <flashinfer/group_gemm/wrapper.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer::group_gemm;

void CutlassSegmentGEMMPyTorchWrapper::RegisterProblem(torch::Tensor workspace_buffer,
                                                       unsigned int batch_size, unsigned int d_in,
                                                       unsigned int d_out, bool weight_column_major,
                                                       torch::Tensor seg_indptr,
                                                       torch::Tensor weight_indices,
                                                       torch::Tensor empty_data, ) {
  CHECK_CUDA(workspace_buffer);
  // TODO(Zihao): add more checks here
  size_t workspace_size_in_bytes = workspace_buffer.size(0) * workspace_buffer.element_size();
  // cast seg_indptr to int64
  seg_indptr = seg_indptr.to(torch::kInt64).to(workspace_buffer.device());
  bool weight_indices_defined = weight_indices.numel() > 0;
  if (weight_indices_defined) {
    weight_indices = weight_indices.to(torch::kInt64).to(workspace_buffer.device());
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(empty_data.scalar_type(), c_type, [&] {
    cudaError_t status = handler_->RegisterProblem<c_type>(
        static_cast<void*>(workspace_buffer.data_ptr()), workspace_size_in_bytes,
        static_cast<int64_t*>(seg_indptr.data_ptr()),
        weight_indices_defined ? static_cast<int64_t*>(weight_indices.data_ptr()) : nullptr,
        batch_size, d_in, d_out, weight_column_major);
    TORCH_CHECK(status == cudaSuccess, "Failed to register problem: ", cudaGetErrorString(status));
  });
}

torch::Tensor CutlassSegmentGEMMPyTorchWrapper::Forward(torch::Tensor x, torch::Tensor weight) {
  // TODO(Zihao): Add more checks here
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_DIM(2, x);       // x: [sum(m_i), d_in]
  CHECK_DIM(2, weight);  // weight: [d_out, d_in] if weight_column_major, [d_in, d_out] otherwise
  size_t cumulative_batch_size = x.size(0);
  size_t d_out = handler_->IsWeightColumnMajor() ? weight.size(0) : weight.size(1);
  size_t d_in = handler_->IsWeightColumnMajor() ? weight.size(1) : weight.size(0);
  CHECK_EQ(x.size(1), d_in);
  auto y = torch::empty({cumulative_batch_size, d_out}, x.options());
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(x.scalar_type(), c_type, [&] {
    cudaError_t status = CutlassSegmentGEMMWrapper<c_type>(
        handler_, static_cast<c_type*>(x.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
        static_cast<c_type*>(y.data_ptr()), torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
  });
}