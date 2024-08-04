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
#include <flashinfer/norm.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor rmsnorm(torch::Tensor input, torch::Tensor weight, double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto output = torch::empty_like(input);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::RMSNorm(static_cast<c_type*>(input.data_ptr()),
                                       static_cast<c_type*>(weight.data_ptr()),
                                       static_cast<c_type*>(output.data_ptr()), batch_size,
                                       hidden_size, eps, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "RMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
  return output;
}

void fused_add_rmsnorm(torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
                       double eps) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(static_cast<c_type*>(input.data_ptr()),
                                               static_cast<c_type*>(residual.data_ptr()),
                                               static_cast<c_type*>(weight.data_ptr()), batch_size,
                                               hidden_size, eps, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess, "FusedAddRMSNorm failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}
