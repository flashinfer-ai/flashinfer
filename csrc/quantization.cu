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

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void packbits(TensorView x, const std::string& bitorder, TensorView y) {
  CHECK_INPUT(x);
  auto device = x.device();
  TVM_FFI_ICHECK(bitorder == "big" || bitorder == "little") << "bitorder must be 'big' or 'little'";

  int64_t num_elements = x.numel();
  auto stream = get_stream(x.device());
  cudaError_t status = quantization::PackBits(
      static_cast<bool*>(x.data_ptr()), static_cast<uint8_t*>(y.data_ptr()), num_elements,
      bitorder == "big" ? quantization::BitOrder::kBig : quantization::BitOrder::kLittle, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "PackBits failed with error code " << cudaGetErrorString(status);
}

void segment_packbits(TensorView x, TensorView input_indptr, TensorView output_indptr,
                      const std::string& bitorder, TensorView y) {
  CHECK_INPUT(x);
  CHECK_INPUT(input_indptr);
  CHECK_INPUT(output_indptr);
  CHECK_DEVICE(input_indptr, x);
  CHECK_DEVICE(output_indptr, x);
  TVM_FFI_ICHECK(bitorder == "big" || bitorder == "little") << "bitorder must be 'big' or 'little'";
  unsigned int batch_size = input_indptr.size(0) - 1;
  TVM_FFI_ICHECK_EQ(output_indptr.size(0), batch_size + 1)
      << "output_indptr must be on the same device as x";

  auto stream = get_stream(x.device());
  cudaError_t status = quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()), static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()), batch_size,
      bitorder == "big" ? quantization::BitOrder::kBig : quantization::BitOrder::kLittle, stream);
}
