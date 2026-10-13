/*
 * Copyright (c) 2026 by FlashInfer team.
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
#include <flashinfer/mla_kv_pack_fp8.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

/*!
 * \brief Fused MLA context K/V pack with saturating fp8 e4m3 cast.
 *
 * Input:
 *   - kv_nope: [num_tokens, num_heads, 256] bf16 ([k_nope(128) | v(128)] per head)
 *   - k_pe:    [num_tokens, 64] bf16 (shared across heads)
 * Output (pre-allocated, contiguous):
 *   - key:   [num_tokens, num_heads, 192] fp8 e4m3 = [k_nope | k_pe broadcast]
 *   - value: [num_tokens, num_heads, 128] fp8 e4m3 = v
 */
void concat_mla_kv_quant_fp8(TensorView kv_nope, TensorView k_pe, TensorView key,
                             TensorView value) {
  using namespace mla_kv_pack;
  CHECK_INPUT(kv_nope);
  CHECK_INPUT(k_pe);
  CHECK_INPUT(key);
  CHECK_INPUT(value);
  CHECK_DEVICE(kv_nope, k_pe);
  CHECK_DEVICE(kv_nope, key);
  CHECK_DEVICE(kv_nope, value);
  CHECK_DIM(3, kv_nope);  // [num_tokens, num_heads, 256]
  CHECK_DIM(2, k_pe);     // [num_tokens, 64]
  CHECK_DIM(3, key);      // [num_tokens, num_heads, 192]
  CHECK_DIM(3, value);    // [num_tokens, num_heads, 128]
  CHECK_INPUT_TYPE(kv_nope, dl_bfloat16);
  CHECK_INPUT_TYPE(k_pe, dl_bfloat16);
  CHECK_INPUT_TYPE(key, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(value, dl_float8_e4m3fn);

  const int64_t num_tokens = kv_nope.size(0);
  const int64_t num_heads = kv_nope.size(1);
  TVM_FFI_ICHECK_EQ(kv_nope.size(2), kKVDim) << "kv_nope last dim must be 256 (nope 128 | v 128)";
  TVM_FFI_ICHECK_EQ(k_pe.size(0), num_tokens) << "k_pe and kv_nope must have the same num_tokens";
  TVM_FFI_ICHECK_EQ(k_pe.size(1), kRopeDim) << "k_pe last dim must be 64";
  TVM_FFI_ICHECK_EQ(key.size(0), num_tokens) << "key and kv_nope must have the same num_tokens";
  TVM_FFI_ICHECK_EQ(key.size(1), num_heads) << "key and kv_nope must have the same num_heads";
  TVM_FFI_ICHECK_EQ(key.size(2), kQKDim) << "key last dim must be 192 (nope 128 | rope 64)";
  TVM_FFI_ICHECK_EQ(value.size(0), num_tokens) << "value and kv_nope must have the same num_tokens";
  TVM_FFI_ICHECK_EQ(value.size(1), num_heads) << "value and kv_nope must have the same num_heads";
  TVM_FFI_ICHECK_EQ(value.size(2), kVDim) << "value last dim must be 128";
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= 1024) << "num_heads out of range";
  // 16 B vectorized loads/stores: every row start must be 16 B aligned.
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(kv_nope.data_ptr()) % 16, 0)
      << "kv_nope must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(k_pe.data_ptr()) % 16, 0)
      << "k_pe must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(key.data_ptr()) % 16, 0)
      << "key must be 16-byte aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(value.data_ptr()) % 16, 0)
      << "value must be 16-byte aligned";

  ffi::CUDADeviceGuard device_guard(kv_nope.device().device_id);
  const cudaStream_t stream = get_stream(kv_nope.device());
  cudaError_t status =
      MLAKVPackFP8(kv_nope.data_ptr(), k_pe.data_ptr(), key.data_ptr(), value.data_ptr(),
                   num_tokens, static_cast<int>(num_heads), stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "MLAKVPackFP8 failed with error: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(concat_mla_kv_quant_fp8, concat_mla_kv_quant_fp8);
