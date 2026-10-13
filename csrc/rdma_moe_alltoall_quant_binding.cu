/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 * Numerical kernels are derived from DeepEP under LICENSE.deepep.txt.
 */
#include <cmath>

#include "flashinfer/comm/rdma_moe_alltoall/kernels/extension_api.cuh"
#include "rdma_moe_alltoall_runtime_utils.h"

namespace deep_ep {
namespace {
int hidden_and_tokens(TensorView input, DLDataType dtype, int hidden_multiplier, int& tokens) {
  const int device = input.device().device_id;
  LL_CHECK(input.ndim() == 2 || input.ndim() == 3,
           "quantization input must have rank two or three");
  check_tensor(input, "input", dtype, input.ndim(), device);
  check_vector_alignment(input, "input");
  const auto hidden = input.size(input.ndim() - 1) * hidden_multiplier;
  LL_CHECK(hidden == 4096 || hidden == 6144 || hidden == 7168,
           "quantization supports hidden 4096, 6144, or 7168");
  const auto rows = input.numel() / input.size(input.ndim() - 1);
  LL_CHECK(rows <= std::numeric_limits<int>::max(), "too many quantization rows");
  tokens = static_cast<int>(rows);
  return static_cast<int>(hidden);
}

void check_row_shape(TensorView tensor, TensorView input, const char* name, DLDataType dtype,
                     int64_t width) {
  check_tensor(tensor, name, dtype, input.ndim(), input.device().device_id);
  for (int i = 0; i < input.ndim() - 1; ++i)
    LL_CHECK(tensor.size(i) == input.size(i), name, " row shape mismatch");
  LL_CHECK(tensor.size(input.ndim() - 1) == width, name, " width mismatch");
}
}  // namespace

void quantize_nvfp4(TensorView input, TensorView global_scale, TensorView packed,
                    TensorView scales) {
  int tokens;
  const int hidden = hidden_and_tokens(input, dl_bfloat16, 1, tokens);
  check_row_shape(global_scale, input, "global_scale", dl_float32, 1);
  check_row_shape(packed, input, "packed", dl_uint8, hidden / 2);
  check_vector_alignment(packed, "packed");
  check_row_shape(scales, input, "scales", dl_uint8, hidden / 16);
  check_dispatch_overlap({packed, scales}, {input, global_scale});
  ffi::CUDADeviceGuard guard(input.device().device_id);
  if (tokens)
    extensions::quantize_bf16_to_nvfp4(input.data_ptr(), pointer<float>(global_scale),
                                       packed.data_ptr(), scales.data_ptr(), tokens, hidden,
                                       get_stream(input.device()));
  LL_CUDA(cudaGetLastError());
}

void quantize_nvfp4_uniform(TensorView input, float global_scale, TensorView packed,
                            TensorView scales) {
  LL_CHECK(std::isfinite(global_scale) && global_scale > 0.f,
           "global_scale must be a positive finite float32 scalar");
  int tokens;
  const int hidden = hidden_and_tokens(input, dl_bfloat16, 1, tokens);
  check_row_shape(packed, input, "packed", dl_uint8, hidden / 2);
  check_vector_alignment(packed, "packed");
  check_row_shape(scales, input, "scales", dl_uint8, hidden / 16);
  check_dispatch_overlap({packed, scales}, {input});
  ffi::CUDADeviceGuard guard(input.device().device_id);
  if (tokens)
    extensions::quantize_bf16_to_nvfp4_uniform(input.data_ptr(), global_scale, packed.data_ptr(),
                                               scales.data_ptr(), tokens, hidden,
                                               get_stream(input.device()));
  LL_CUDA(cudaGetLastError());
}

void dequantize_nvfp4(TensorView packed, TensorView global_scale, TensorView scales,
                      TensorView output) {
  int tokens;
  const int hidden = hidden_and_tokens(packed, dl_uint8, 2, tokens);
  check_row_shape(global_scale, packed, "global_scale", dl_float32, 1);
  check_row_shape(scales, packed, "scales", dl_uint8, hidden / 16);
  check_row_shape(output, packed, "output", dl_bfloat16, hidden);
  check_vector_alignment(output, "output");
  check_dispatch_overlap({output}, {packed, global_scale, scales});
  ffi::CUDADeviceGuard guard(packed.device().device_id);
  if (tokens)
    extensions::dequantize_nvfp4_to_bf16(packed.data_ptr(), pointer<float>(global_scale),
                                         scales.data_ptr(), output.data_ptr(), tokens, hidden,
                                         get_stream(packed.device()));
  LL_CUDA(cudaGetLastError());
}

void quantize_fp8(TensorView input, TensorView packed, TensorView scales) {
  int tokens;
  const int hidden = hidden_and_tokens(input, dl_bfloat16, 1, tokens);
  check_row_shape(packed, input, "packed", dl_uint8, hidden);
  check_vector_alignment(packed, "packed");
  check_row_shape(scales, input, "scales", dl_float32, hidden / 128);
  check_dispatch_overlap({packed, scales}, {input});
  ffi::CUDADeviceGuard guard(input.device().device_id);
  extensions::quantize_bf16_to_fp8(input.data_ptr(), packed.data_ptr(), scales.data_ptr(), tokens,
                                   hidden, get_stream(input.device()));
  LL_CUDA(cudaGetLastError());
}

void dequantize_fp8(TensorView packed, TensorView scales, TensorView output) {
  int tokens;
  const int hidden = hidden_and_tokens(packed, dl_uint8, 1, tokens);
  check_row_shape(scales, packed, "scales", dl_float32, hidden / 128);
  check_row_shape(output, packed, "output", dl_bfloat16, hidden);
  check_vector_alignment(output, "output");
  check_dispatch_overlap({output}, {packed, scales});
  ffi::CUDADeviceGuard guard(packed.device().device_id);
  extensions::dequantize_fp8_to_bf16(packed.data_ptr(), scales.data_ptr(), output.data_ptr(),
                                     tokens, hidden, get_stream(packed.device()));
  LL_CUDA(cudaGetLastError());
}
}  // namespace deep_ep

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_nvfp4, deep_ep::quantize_nvfp4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_nvfp4_uniform, deep_ep::quantize_nvfp4_uniform);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequantize_nvfp4, deep_ep::dequantize_nvfp4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_fp8, deep_ep::quantize_fp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequantize_fp8, deep_ep::dequantize_fp8);
