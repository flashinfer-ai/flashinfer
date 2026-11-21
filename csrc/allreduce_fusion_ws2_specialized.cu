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
#include <mutex>

#include "tvm_ffi_utils.h"

namespace ws2 = flashinfer::allreduce_fusion_ws2;

namespace {

bool is_aligned_16(const TensorView& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// Eager launches of the kernel on one device run one grid at a time, also across streams, in
// the order the host issued them (the same order on every rank). Each grid holds every
// co-resident CTA slot of its device and waits on its peer, so two grids in flight on different
// streams can deadlock across the ranks: rank 0 runs grid A first and waits for the peer's A,
// while the peer runs grid B first (its A queued behind other work) and A cannot become resident
// behind B. A launch on another stream than the device's previous eager launch therefore waits
// for the work enqueued on that previous stream so far (an event recorded there now). A launch
// on the same stream as the previous one (the usual case) costs a comparison. Launches during a
// stream capture are not ordered here: a graph's launches follow the stream it is replayed on.
// The previous stream is used again only to record that event; PyTorch's streams are never
// destroyed (an external stream must outlive the next launch on another stream). Ordering is
// best effort: a CUDA error here is cleared and the launch proceeds unordered, since raising on
// one rank only would leave its peer waiting in the kernel.
constexpr int kMaxDevices = 64;

struct LaunchOrder {
  std::mutex mu;
  bool has_last = false;
  cudaStream_t last = nullptr;
  cudaEvent_t done = nullptr;
};

void order_after_previous_launch(int device, cudaStream_t stream) {
  static LaunchOrder orders[kMaxDevices];
  if (device < 0 || device >= kMaxDevices) return;
  LaunchOrder& order = orders[device];
  std::lock_guard<std::mutex> lock(order.mu);
  if (order.has_last && order.last == stream) return;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }
  if (capture != cudaStreamCaptureStatusNone) return;
  if (order.has_last) {
    cudaStreamCaptureStatus last_capture = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(order.last, &last_capture);
    if (err == cudaSuccess && last_capture == cudaStreamCaptureStatusNone) {
      if (order.done == nullptr) {
        err = cudaEventCreateWithFlags(&order.done, cudaEventDisableTiming);
      }
      if (err == cudaSuccess) err = cudaEventRecord(order.done, order.last);
      if (err == cudaSuccess) err = cudaStreamWaitEvent(stream, order.done, 0);
    }
    if (err != cudaSuccess) (void)cudaGetLastError();
  }
  order.last = stream;
  order.has_last = true;
}

}  // namespace

// One call of the specialized AR + residual add + RMSNorm (pattern kARResidualRMSNorm) on a
// two-rank workspace. input / residual_in / residual_out / norm_out: [num_tokens, 6144] bf16,
// rms_gamma: [6144] bf16, all contiguous on one device; residual_out and norm_out may be the same
// tensor as input or residual_in. own_region / peer_region: this rank's and the peer's mapping of
// the peer-mapped region (ws2::region_bytes(capacity) bytes each); epoch: rank-local int32 call
// counter of ws2::epoch_words(grid) words. Launches on the caller's current stream, after the
// device's previous eager launch of this kernel (see order_after_previous_launch); no host sync,
// no allocation (CUDA-graph capturable). Returns 0 after the launch, or 1 without launching when
// a tensor is not 16-byte aligned (the caller stages aligned copies and calls again, so a rank
// with a misaligned view still runs the kernel its peer runs). Every other failure raises: the
// call is collective, so a rank must never silently take another path than its peer.
int64_t allreduce_fusion_ws2_ar_rmsnorm(TensorView input, TensorView residual_in,
                                        TensorView rms_gamma, TensorView residual_out,
                                        TensorView norm_out, int64_t own_region,
                                        int64_t peer_region, TensorView epoch, int64_t capacity,
                                        int64_t grid, double rms_eps, double weight_bias,
                                        bool launch_with_pdl) {
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
  if (!(is_aligned_16(input) && is_aligned_16(residual_in) && is_aligned_16(rms_gamma) &&
        is_aligned_16(residual_out) && is_aligned_16(norm_out))) {
    return 1;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());
  order_after_previous_launch(input.device().device_id, stream);
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
  return 0;
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
