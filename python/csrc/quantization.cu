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
#include <flashinfer/quantization.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

torch::Tensor packbits(torch::Tensor x, const std::string& bitorder) {
  CHECK_INPUT(x);
  auto device = x.device();
  TORCH_CHECK(bitorder == "big" || bitorder == "little", "bitorder must be 'big' or 'little'");
  x = x.to(torch::kBool);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());

  int64_t num_elements = x.numel();
  int64_t num_output_elements = (num_elements + 7) / 8;

  auto y = torch::empty({num_output_elements}, x.options().dtype(torch::kUInt8));

  cudaError_t status = quantization::PackBits(
      static_cast<bool*>(x.data_ptr()), static_cast<uint8_t*>(y.data_ptr()), num_elements,
      bitorder == "big" ? quantization::BitOrder::kBig : quantization::BitOrder::kLittle,
      torch_current_stream);

  TORCH_CHECK(status == cudaSuccess,
              "PackBits failed with error code " + std::string(cudaGetErrorString(status)));
  return y;
}

torch::Tensor segment_packbits(torch::Tensor x, torch::Tensor input_indptr,
                               torch::Tensor output_indptr, const std::string& bitorder) {
  CHECK_INPUT(x);
  CHECK_INPUT(input_indptr);
  CHECK_INPUT(output_indptr);
  auto device = x.device();
  CHECK_EQ(input_indptr.device(), device);
  CHECK_EQ(output_indptr.device(), device);
  TORCH_CHECK(bitorder == "big" || bitorder == "little", "bitorder must be 'big' or 'little'");
  unsigned int batch_size = input_indptr.size(0) - 1;
  CHECK_EQ(output_indptr.size(0), batch_size + 1);
  input_indptr = input_indptr.to(torch::kInt32);
  output_indptr = output_indptr.to(torch::kInt32);
  int64_t output_nnz = output_indptr[batch_size].item<int64_t>();
  auto y = torch::empty({output_nnz}, x.options().dtype(torch::kUInt8));

  cudaError_t status = quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()), static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()), batch_size,
      bitorder == "big" ? quantization::BitOrder::kBig : quantization::BitOrder::kLittle,
      c10::cuda::getCurrentCUDAStream(device.index()));
  return y;
}
