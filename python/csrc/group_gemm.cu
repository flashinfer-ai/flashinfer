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

void CutlassSegmentGEMMPyTorchWrapper::RegisterWorkspaceBuffer(torch::Tensor workspace_buffer) {
  handler_->RegisterWorkspace(static_cast<void*>(workspace_buffer.data_ptr()),
                              workspace_buffer.size(0) * workspace_buffer.element_size());
}

torch::Tensor CutlassSegmentGEMMPyTorchWrapper::Forward(torch::Tensor seg_indptr,
                                                        torch::Tensor weight_indices,
                                                        torch::Tensor x, torch::Tensor weight,
                                                        unsigned int batch_size,
                                                        bool weight_column_major) {
  // TODO(Zihao): Add more checks here
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  auto device = x.device();
  CHECK_EQ(seg_indptr.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, x);       // x: [sum(m_i), d_in]
  CHECK_DIM(3, weight);  // weight: [num_weights, d_out, d_in] if weight_column_major, [num_weights,
                         // d_in, d_out] otherwise
  int64_t cumulative_batch_size = x.size(0);
  int64_t d_out = weight_column_major ? weight.size(1) : weight.size(2);
  int64_t d_in = weight_column_major ? weight.size(2) : weight.size(1);
  CHECK_EQ(x.size(1), d_in);
  auto y = torch::zeros({cumulative_batch_size, d_out}, x.options());
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  seg_indptr = seg_indptr.to(torch::kInt64);

  bool weight_indices_defined = weight_indices.numel() > 0;
  if (weight_indices_defined) {
    CHECK_INPUT(weight_indices);
    CHECK_EQ(weight_indices.device(), device);
    weight_indices = weight_indices.to(torch::kInt64);
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(x.scalar_type(), c_type, [&] {
    using cutlass_t = typename cutlass_dtype<c_type>::type;
    auto status = CutlassSegmentGEMMWrapper<cutlass_t>(
        handler_.get(), static_cast<cutlass_t*>(x.data_ptr()),
        static_cast<cutlass_t*>(weight.data_ptr()), static_cast<cutlass_t*>(y.data_ptr()),
        static_cast<int64_t*>(seg_indptr.data_ptr()),
        weight_indices_defined ? static_cast<int64_t*>(weight_indices.data_ptr()) : nullptr,
        batch_size, d_in, d_out, weight_column_major, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassSegmentGEMM: ", cudaGetErrorString(status));
    return true;
  });

  return y;
}