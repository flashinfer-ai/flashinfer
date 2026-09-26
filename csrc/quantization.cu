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

void per_token_group_quant_8bit(TensorView input, TensorView output_q, TensorView output_s,
                                double eps, int64_t group_size, int64_t groups_per_row,
                                int64_t scale_stride, bool column_major, bool scale_ue8m0) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  CHECK_DEVICE(output_q, input);
  CHECK_DEVICE(output_s, input);
  CHECK_INPUT_TYPE(output_s, dl_float32);
  TVM_FFI_ICHECK(input.dtype() == dl_float16 || input.dtype() == dl_bfloat16)
      << "input must have dtype float16 or bfloat16";
  TVM_FFI_ICHECK_EQ(group_size, 128) << "CUDA register backend currently requires group_size=128";
  TVM_FFI_ICHECK_EQ(input.numel() % group_size, 0)
      << "input element count must be divisible by group_size";
  TVM_FFI_ICHECK_EQ(output_q.numel(), input.numel())
      << "quantized output must have the same number of elements as input";
  TVM_FFI_ICHECK(!scale_ue8m0 || column_major) << "scale_ue8m0 requires column_major_scales=True";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(output_q.data_ptr()) % alignof(uint2), 0)
      << "quantized output must be 8-byte aligned";

  float min_8bit;
  float max_8bit;
  if (output_q.dtype() == dl_float8_e4m3fn) {
    min_8bit = -448.0f;
    max_8bit = 448.0f;
  } else if (output_q.dtype() == dl_float8_e5m2) {
    min_8bit = -57344.0f;
    max_8bit = 57344.0f;
  } else if (output_q.dtype() == dl_int8) {
    min_8bit = -128.0f;
    max_8bit = 127.0f;
  } else {
    TVM_FFI_THROW(TypeError) << "output_q must have dtype float8_e4m3fn, float8_e5m2, or int8";
  }

  const int64_t num_groups = input.numel() / group_size;
  if (num_groups == 0) {
    return;
  }
  TVM_FFI_ICHECK(groups_per_row > 0) << "groups_per_row must be positive";
  TVM_FFI_ICHECK(scale_stride > 0) << "scale_stride must be positive";
  if (column_major) {
    TVM_FFI_ICHECK_EQ(num_groups % groups_per_row, 0)
        << "num_groups must be divisible by groups_per_row";
    const int64_t num_rows = num_groups / groups_per_row;
    TVM_FFI_ICHECK_GE(scale_stride, num_rows) << "scale_stride must cover every input row";
    TVM_FFI_ICHECK_GE(output_s.numel() / groups_per_row, scale_stride)
        << "scale output storage is too small";
  } else {
    TVM_FFI_ICHECK_GE(output_s.numel(), num_groups) << "scale output storage is too small";
  }

  auto stream = get_stream(input.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), DTypeIn, [&] {
    cudaError_t status;
    if (output_q.dtype() == dl_float8_e4m3fn) {
      status = quantization::PerTokenGroupQuant8BitRegister<DTypeIn, __nv_fp8_e4m3>(
          static_cast<DTypeIn*>(input.data_ptr()), static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()), num_groups, static_cast<float>(eps), min_8bit,
          max_8bit, groups_per_row, scale_stride, column_major, scale_ue8m0, stream);
    } else if (output_q.dtype() == dl_float8_e5m2) {
      status = quantization::PerTokenGroupQuant8BitRegister<DTypeIn, __nv_fp8_e5m2>(
          static_cast<DTypeIn*>(input.data_ptr()), static_cast<__nv_fp8_e5m2*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()), num_groups, static_cast<float>(eps), min_8bit,
          max_8bit, groups_per_row, scale_stride, column_major, scale_ue8m0, stream);
    } else {
      status = quantization::PerTokenGroupQuant8BitRegister<DTypeIn, int8_t>(
          static_cast<DTypeIn*>(input.data_ptr()), static_cast<int8_t*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()), num_groups, static_cast<float>(eps), min_8bit,
          max_8bit, groups_per_row, scale_stride, column_major, scale_ue8m0, stream);
    }
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "PerTokenGroupQuant8BitRegister failed with error " << cudaGetErrorString(status);
    return true;
  });
}
