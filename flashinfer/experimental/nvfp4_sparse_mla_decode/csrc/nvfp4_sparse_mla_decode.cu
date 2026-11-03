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
#include <cstdint>

#include "nvfp4_sparse_mla_decode.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::TensorView;
namespace kernel = flashinfer::nvfp4_sparse_mla_decode;

namespace {

void check_aligned(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16, 0)
      << name << " must be 16-byte aligned";
}

}  // namespace

// kv_cache: uint8 [..., 352] contiguous nvfp4_ds_mla rows; query: float8_e4m3fn [T, 16, 576];
// indices: int32 [T, topk_width] flat row ids (-1 = no key); out: bfloat16 [T, 16, 512].
void nvfp4_sparse_mla_decode_run(TensorView kv_cache, TensorView query, TensorView indices,
                                 TensorView out, int64_t num_ctas_per_token, double sm_scale_log2,
                                 double output_scale) {
  CHECK_INPUT_AND_TYPE(kv_cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(query, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(indices, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_DIM(3, query);
  CHECK_DIM(2, indices);
  CHECK_DIM(3, out);
  CHECK_DEVICE(query, kv_cache);
  CHECK_DEVICE(query, indices);
  CHECK_DEVICE(query, out);
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2) << "kv_cache must be [..., " << kernel::ROWB << "]";
  TVM_FFI_ICHECK_EQ(kv_cache.size(kv_cache.ndim() - 1), kernel::ROWB)
      << "kv_cache rows must be " << kernel::ROWB << " bytes (nvfp4_ds_mla)";
  const int64_t num_tokens = query.size(0);
  TVM_FFI_ICHECK_EQ(query.size(1), kernel::H) << "query must have " << kernel::H << " heads";
  TVM_FFI_ICHECK_EQ(query.size(2), kernel::D) << "query head dim must be " << kernel::D;
  TVM_FFI_ICHECK_EQ(indices.size(0), num_tokens) << "indices must have one row per query token";
  TVM_FFI_ICHECK(out.size(0) == num_tokens && out.size(1) == kernel::H && out.size(2) == kernel::DV)
      << "out must be [num_tokens, " << kernel::H << ", " << kernel::DV << "]";
  check_aligned(kv_cache, "kv_cache");
  check_aligned(query, "query");
  if (num_tokens == 0) return;
  const int64_t topk_width = indices.size(1);
  TVM_FFI_ICHECK(
      topk_width <= INT32_MAX &&
      kernel::is_valid_config(static_cast<int>(topk_width), static_cast<int>(num_ctas_per_token)))
      << "unsupported configuration: topk_width=" << topk_width
      << ", num_ctas_per_token=" << num_ctas_per_token;
  TVM_FFI_ICHECK_LE(num_tokens * num_ctas_per_token, INT32_MAX) << "too many query tokens";

  // The shared-memory attribute is set on, and the kernel launched from, the current device.
  tvm::ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(out.device());
  cudaError_t status = kernel::launch(
      static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const uint8_t*>(query.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int>(num_tokens), static_cast<int>(topk_width), static_cast<float>(sm_scale_log2),
      static_cast<float>(output_scale), static_cast<int>(num_ctas_per_token), stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "NVFP4 sparse MLA decode launch failed: " << cudaGetErrorString(status);
}

// Clusters of `num_ctas_per_token` CTAs that fit on the current device in one wave.
int64_t nvfp4_sparse_mla_decode_max_active_clusters(int64_t num_ctas_per_token) {
  int count = 0;
  cudaError_t status = kernel::max_active_clusters(static_cast<int>(num_ctas_per_token), &count);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaOccupancyMaxActiveClusters failed: " << cudaGetErrorString(status);
  return count;
}
