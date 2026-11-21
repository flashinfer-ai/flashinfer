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
#include <climits>
#include <flashinfer/top_p_renorm_specialized.cuh>

#include "tvm_ffi_utils.h"

namespace topp = flashinfer::sampling::top_p_renorm_specialized;

// probs: [B, V] float32, top_p: [B] float32, renorm_probs: [B, V] float32 (uninitialised is
// fine: every element is written). Launches on the caller's current stream; no host sync.
void top_p_renorm_probs_specialized(TensorView probs, TensorView top_p, TensorView renorm_probs) {
  CHECK_INPUT_AND_TYPE(probs, dl_float32);
  CHECK_INPUT_AND_TYPE(top_p, dl_float32);
  CHECK_INPUT_AND_TYPE(renorm_probs, dl_float32);
  CHECK_DIM(2, probs);
  CHECK_DIM(1, top_p);
  CHECK_DIM(2, renorm_probs);
  CHECK_DEVICE(top_p, probs);
  CHECK_DEVICE(renorm_probs, probs);
  const int64_t batch_size = probs.size(0);
  const int64_t vocab_size = probs.size(1);
  TVM_FFI_ICHECK_EQ(top_p.size(0), batch_size) << "top_p must hold one value per row";
  TVM_FFI_ICHECK_EQ(renorm_probs.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(renorm_probs.size(1), vocab_size);
  // One cluster of up to 16 CTAs per row on grid.x; element offsets within a row are int.
  TVM_FFI_ICHECK_LE(batch_size, INT_MAX / 16) << "batch_size too large for the specialized kernel";
  TVM_FFI_ICHECK_LE(vocab_size, INT_MAX / 2) << "vocab_size too large for the specialized kernel";

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  const cudaStream_t stream = get_stream(probs.device());
  cudaError_t status = topp::topp_renorm_launch(
      static_cast<const float*>(probs.data_ptr()), static_cast<const float*>(top_p.data_ptr()),
      static_cast<float*>(renorm_probs.data_ptr()), static_cast<int>(batch_size),
      static_cast<int>(vocab_size), stream);
  if (status != cudaSuccess) {
    // A rejected launch configuration is not sticky; clear it so the caller's fallback path and
    // later kernels do not observe it.
    (void)cudaGetLastError();
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "top_p_renorm_probs_specialized failed with error code " << cudaGetErrorString(status);
}

// Introspection: the launch configuration the host plan picks for (batch_size, vocab_size),
// packed as cluster_size * 1e8 + num_threads * 1e4 + min_blocks * 1e3 + unroll * 10 + vectorized
// (16-byte aligned rows assumed).
int64_t top_p_renorm_probs_specialized_plan(int64_t batch_size, int64_t vocab_size) {
  const topp::LaunchConfig& k = topp::kLaunchConfigs[topp::plan_launch(
      static_cast<int>(batch_size), static_cast<int>(vocab_size))];
  return int64_t(k.cluster_size) * 100000000 + int64_t(k.num_threads) * 10000 +
         int64_t(k.min_blocks) * 1000 + int64_t(k.unroll) * 10 + int64_t(vocab_size % 4 == 0);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_p_renorm_probs_specialized, top_p_renorm_probs_specialized);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_p_renorm_probs_specialized_plan,
                              top_p_renorm_probs_specialized_plan);
