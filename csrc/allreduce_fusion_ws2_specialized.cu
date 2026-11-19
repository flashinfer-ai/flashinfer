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
#include <flashinfer/comm/allreduce_fusion_ws2_specialized.cuh>

#include "tvm_ffi_utils.h"

namespace ws2 = flashinfer::allreduce_fusion_ws2;

namespace {

void check_aligned_16(const TensorView& t, const char* name) {
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16, 0)
      << name << " must be 16-byte aligned";
}

}  // namespace

// One call of the specialized AR + residual add + RMSNorm (pattern kARResidualRMSNorm) on a
// two-rank workspace. input / residual_in / residual_out / norm_out: [num_tokens, 6144] bf16,
// rms_gamma: [6144] bf16, all contiguous on one device; residual_out and norm_out may be the same
// tensor as input or residual_in. own_region / peer_region: this rank's and the peer's mapping of
// the peer-mapped region (ws2::region_bytes(capacity) bytes each); epoch: rank-local int32 call
// counter of ws2::epoch_words(grid) words. Launches on the caller's current stream; no host sync,
// no allocation (CUDA-graph capturable). Every failure raises: the call is collective, so a rank
// must never silently take another path than its peer.
void allreduce_fusion_ws2_ar_rmsnorm(TensorView input, TensorView residual_in, TensorView rms_gamma,
                                     TensorView residual_out, TensorView norm_out,
                                     int64_t own_region, int64_t peer_region, TensorView epoch,
                                     int64_t capacity, int64_t grid, double rms_eps,
                                     double weight_bias, bool launch_with_pdl) {
  CHECK_INPUT_AND_TYPE(input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual_in, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(rms_gamma, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual_out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(norm_out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(epoch, dl_int32);
  CHECK_DIM(2, input);
  CHECK_DIM(2, residual_in);
  CHECK_DIM(2, residual_out);
  CHECK_DIM(2, norm_out);
  CHECK_DIM(1, rms_gamma);
  CHECK_DIM(1, epoch);
  CHECK_DEVICE(residual_in, input);
  CHECK_DEVICE(rms_gamma, input);
  CHECK_DEVICE(residual_out, input);
  CHECK_DEVICE(norm_out, input);
  CHECK_DEVICE(epoch, input);
  const int64_t num_tokens = input.size(0);
  TVM_FFI_ICHECK_EQ(input.size(1), ws2::kHidden) << "hidden size must be " << ws2::kHidden;
  TVM_FFI_ICHECK(residual_in.size(0) == num_tokens && residual_in.size(1) == ws2::kHidden)
      << "residual_in must have the shape of input";
  TVM_FFI_ICHECK(residual_out.size(0) == num_tokens && residual_out.size(1) == ws2::kHidden)
      << "residual_out must have the shape of input";
  TVM_FFI_ICHECK(norm_out.size(0) == num_tokens && norm_out.size(1) == ws2::kHidden)
      << "norm_out must have the shape of input";
  TVM_FFI_ICHECK_EQ(rms_gamma.size(0), ws2::kHidden) << "rms_gamma must hold the hidden size";
  TVM_FFI_ICHECK(num_tokens >= 1 && num_tokens <= capacity)
      << "num_tokens " << num_tokens << " outside [1, " << capacity << "]";
  TVM_FFI_ICHECK_GE(grid, 1) << "grid must be positive";
  TVM_FFI_ICHECK_GE(epoch.size(0), ws2::epoch_words(grid)) << "epoch counter too small";
  TVM_FFI_ICHECK(own_region != 0 && peer_region != 0) << "region is not mapped";
  check_aligned_16(input, "input");
  check_aligned_16(residual_in, "residual_in");
  check_aligned_16(rms_gamma, "rms_gamma");
  check_aligned_16(residual_out, "residual_out");
  check_aligned_16(norm_out, "norm_out");

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());
  const cudaError_t status = ws2::launch(
      input.data_ptr(), residual_in.data_ptr(), rms_gamma.data_ptr(), residual_out.data_ptr(),
      norm_out.data_ptr(), reinterpret_cast<void*>(own_region),
      reinterpret_cast<void*>(peer_region), static_cast<uint32_t*>(epoch.data_ptr()), capacity,
      static_cast<int>(num_tokens), static_cast<int>(grid), static_cast<float>(rms_eps),
      static_cast<float>(weight_bias), launch_with_pdl, stream);
  if (status != cudaSuccess) {
    // Clear a non-sticky launch error so later kernels do not observe it; the call still raises.
    (void)cudaGetLastError();
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "allreduce_fusion_ws2_ar_rmsnorm failed with error code " << cudaGetErrorString(status);
}

// Puts this rank's own region (ws2::region_bytes(capacity) bytes) and its call counter into the
// idle state, on the current stream of epoch's device. The caller synchronises, and both ranks
// must have done this before either launches.
void allreduce_fusion_ws2_init(TensorView epoch, int64_t own_region, int64_t capacity) {
  CHECK_INPUT_AND_TYPE(epoch, dl_int32);
  CHECK_DIM(1, epoch);
  TVM_FFI_ICHECK(own_region != 0) << "region is not mapped";
  TVM_FFI_ICHECK_GE(capacity, 1);
  ffi::CUDADeviceGuard device_guard(epoch.device().device_id);
  const cudaStream_t stream = get_stream(epoch.device());
  const cudaError_t status =
      ws2::init(reinterpret_cast<void*>(own_region), capacity,
                static_cast<uint32_t*>(epoch.data_ptr()), epoch.size(0), stream);
  if (status != cudaSuccess) (void)cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "allreduce_fusion_ws2_init failed with error code " << cudaGetErrorString(status);
}

int64_t allreduce_fusion_ws2_region_bytes(int64_t capacity) {
  return static_cast<int64_t>(ws2::region_bytes(capacity));
}

int64_t allreduce_fusion_ws2_epoch_words(int64_t grid) { return ws2::epoch_words(grid); }

// CTAs per launch on `device` (all co-resident); <= 0 when the occupancy query fails.
int64_t allreduce_fusion_ws2_max_grid(int64_t device) {
  return static_cast<int64_t>(ws2::max_grid(static_cast<int>(device)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_fusion_ws2_ar_rmsnorm, allreduce_fusion_ws2_ar_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_fusion_ws2_init, allreduce_fusion_ws2_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_fusion_ws2_region_bytes, allreduce_fusion_ws2_region_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_fusion_ws2_epoch_words, allreduce_fusion_ws2_epoch_words);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(allreduce_fusion_ws2_max_grid, allreduce_fusion_ws2_max_grid);
