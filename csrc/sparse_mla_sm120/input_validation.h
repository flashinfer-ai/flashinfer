// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <flashinfer/attention/sparse_mla_sm120/compute/mega.cuh>

#include "attention_validation.h"
namespace flashinfer::sparse_mla_sm120 {
inline SparseMlaMegaParams check_attention_rope_fp8_inputs(
    TensorView q, TensorView positions, TensorView cos_sin, TensorView fp8_output,
    TensorView output_scales) {
  for (const auto& tensor : {positions, cos_sin, fp8_output, output_scales}) {
    check_attention_device(q, tensor, "mega tensor");
    TVM_FFI_ICHECK(tensor.IsContiguous()) << "mega tensors must be contiguous";
  }
  CHECK_INPUT_TYPE(positions, dl_int32);
  CHECK_INPUT_TYPE(cos_sin, dl_float32);
  CHECK_INPUT_TYPE(fp8_output, dl_uint8);
  CHECK_INPUT_TYPE(output_scales, dl_uint8);
  TVM_FFI_ICHECK(positions.ndim() == 1 && positions.size(0) == q.size(0));
  TVM_FFI_ICHECK(cos_sin.ndim() == 2 && cos_sin.size(0) > 0 && cos_sin.size(1) == 64);
  TVM_FFI_ICHECK(fp8_output.ndim() == 3 && fp8_output.size(0) == q.size(0) &&
                 fp8_output.size(1) == q.size(1) && fp8_output.size(2) == 512);
  TVM_FFI_ICHECK(output_scales.ndim() == 3 && output_scales.size(0) == q.size(0) &&
                 output_scales.size(1) == q.size(1) && output_scales.size(2) == 16);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(fp8_output.data_ptr()) % 8 == 0);
  SparseMlaMegaParams mega{static_cast<const int32_t*>(positions.data_ptr()),
                           static_cast<const float*>(cos_sin.data_ptr()),
                           static_cast<uint8_t*>(fp8_output.data_ptr()),
                           static_cast<uint8_t*>(output_scales.data_ptr())};
  return mega;
}
}  // namespace flashinfer::sparse_mla_sm120
