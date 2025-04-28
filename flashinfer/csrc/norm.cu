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
#include <cstdint>
#include <flashinfer/norm.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
             bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  CHECK_EQ(output.size(0), batch_size);
  CHECK_EQ(output.size(1), hidden_size);

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::RMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
        static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
        output.stride(0), eps, enable_pdl, stream);
    TORCH_CHECK(status == cudaSuccess,
                "RMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}

void fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps,
                       bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
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

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()), batch_size, hidden_size, input.stride(0),
        residual.stride(0), eps, enable_pdl, stream);
    TORCH_CHECK(status == cudaSuccess, "FusedAddRMSNorm failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}

void gemma_rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, double eps,
                   bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);
  CHECK_EQ(output.size(0), batch_size);
  CHECK_EQ(output.size(1), hidden_size);

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::GemmaRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
        static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
        output.stride(0), eps, enable_pdl, stream);
    TORCH_CHECK(status == cudaSuccess,
                "GemmaRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}

void gemma_fused_add_rmsnorm(at::Tensor& input, at::Tensor& residual, at::Tensor& weight,
                             double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
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

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::GemmaFusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()), batch_size, hidden_size, input.stride(0),
        residual.stride(0), eps, enable_pdl, stream);
    TORCH_CHECK(status == cudaSuccess, "GemmaFusedAddRMSNorm failed with error code " +
                                           std::string(cudaGetErrorString(status)));
    return true;
  });
}
