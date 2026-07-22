/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"
#include "tvm_ffi_utils.h"

namespace torch_ext {

namespace {

int32_t getMultiProcessorCount(int32_t device_index) {
  static thread_local int32_t cached_multi_processor_count = -1;
  static thread_local int32_t cached_device_index = -1;

  if (device_index == cached_device_index && cached_multi_processor_count != -1) {
    return cached_multi_processor_count;
  }
  int32_t count;
  cudaError_t cudaStatus =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
  TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
      << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
  cached_multi_processor_count = count;
  cached_device_index = device_index;
  return count;
}

}  // namespace

// (xq, sf) = NVFP4-quantize(x * pqs) in one pass: the per-input-channel pre_quant_scale smoothing
// is fused into the quantize -- byte-identical to a separate x_hat = x*s elementwise pass followed
// by fp4_quantize. x [m, n] bf16, pqs [n] bf16, global_scale f32 (numel >= 1, the per-tensor
// scale). Outputs are caller-allocated: xq [m, n/2] uint8 (packed e2m1), sf uint8 with numel >=
// ceil(m/128)*128 * ceil((n/16)/4)*4 (swizzled 128x4 UE4M3 block scales, vec size 16). SM100+.
void nvfp4_quantize_smooth(TensorView x, TensorView pqs, TensorView global_scale, TensorView xq,
                           TensorView sf, bool enable_pdl) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(pqs, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(global_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(xq, dl_uint8);
  CHECK_INPUT_AND_TYPE(sf, dl_uint8);
  CHECK_DEVICE(pqs, x);
  CHECK_DEVICE(global_scale, x);
  CHECK_DEVICE(xq, x);
  CHECK_DEVICE(sf, x);

  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [m, n]";
  int const m = static_cast<int>(x.size(0));
  int const n = static_cast<int>(x.size(1));
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be divisible by 16 (NVFP4 SF vector size)";
  TVM_FFI_ICHECK_EQ(pqs.numel(), n) << "pqs must have n elements";
  TVM_FFI_ICHECK_GE(global_scale.numel(), 1) << "global_scale must contain at least one element";

  TVM_FFI_ICHECK_EQ(xq.ndim(), 2) << "xq must be [m, n/2]";
  TVM_FFI_ICHECK_EQ(xq.size(0), m) << "xq must be [m, n/2]";
  TVM_FFI_ICHECK_EQ(xq.size(1), n / 2) << "xq must be [m, n/2]";
  // Swizzled 128x4 SF layout: rows padded to a multiple of 128, SF columns (n/16) to a multiple
  // of 4.
  int64_t const sfSize =
      static_cast<int64_t>((m + 128 - 1) / 128 * 128) * ((n / 16 + 4 - 1) / 4 * 4);
  TVM_FFI_ICHECK_GE(sf.numel(), sfSize)
      << "sf is smaller than the required swizzled scale layout (" << sfSize << " bytes)";

  int const smCount = getMultiProcessorCount(x.device().device_id);
  auto stream = get_stream(x.device());
  flashinfer::gemm::nvfp4_smooth_quantize(
      xq.data_ptr(), sf.data_ptr(), x.data_ptr(), pqs.data_ptr(),
      static_cast<float const*>(global_scale.data_ptr()), m, n, smCount, stream, enable_pdl);
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_quantize_smooth, torch_ext::nvfp4_quantize_smooth);
