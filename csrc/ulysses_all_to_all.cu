/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Adapted from ThunderKittens' NVLink all-to-all kernel:
 * https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/parallel/all_to_all/all_to_all.cu
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

// TVM-FFI host bindings for the fused-transpose Ulysses NVLink-P2P all-to-all.
// Kernel logic lives in include/flashinfer/comm/ulysses_all_to_all.cuh.

#include <tvm/ffi/container/array.h>

#include <algorithm>
#include <cstdint>

#include "flashinfer/comm/ulysses_all_to_all.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;

// Fake pointer type, matches fptr_t used by the vLLM custom all-reduce bindings.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace fi = flashinfer::comm::ulysses;

fptr_t init_ulysses_a2a(Array<fptr_t> out_ipc_ptrs, Array<fptr_t> signal_ipc_ptrs, int64_t rank,
                        int64_t world_size, bool full_nvlink) {
  TVM_FFI_ICHECK_LE(world_size, 8) << "ulysses a2a world size > 8 is not supported";
  TVM_FFI_ICHECK(world_size == 2 || world_size == 4 || world_size == 6 || world_size == 8)
      << "ulysses a2a only supports world size in (2, 4, 6, 8)";
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size) << "invalid rank passed in";
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(out_ipc_ptrs.size()), world_size)
      << "out_ipc_ptrs size must equal world_size";
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(signal_ipc_ptrs.size()), world_size)
      << "signal_ipc_ptrs size must equal world_size";

  fi::Signal* signals[8];
  void* out_bufs[8];
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<fi::Signal*>(signal_ipc_ptrs[i]);
    out_bufs[i] = reinterpret_cast<void*>(out_ipc_ptrs[i]);
  }
  // The multi_gpu_barrier counters must start at zero, and cudaMalloc does not
  // zero memory. Each rank owns signals[rank] (the others are IPC-mapped peer
  // buffers), so every rank zeroes its own signal here. cudaMemset is
  // asynchronous with respect to the host (the missing "Async" suffix does
  // NOT make it a host fence), so callers MUST (1) synchronize this device
  // after init returns and (2) issue a process-group barrier before the first
  // all-to-all, so the zeroing is complete and globally visible.
  // UlyssesCommunicator does both inside its init transaction.
  auto st = cudaMemset(signals[rank], 0, sizeof(fi::Signal));
  TVM_FFI_ICHECK(st == cudaSuccess) << "failed to zero the ulysses a2a signal buffer";
  return (fptr_t) new fi::UlyssesA2A(signals, out_bufs, static_cast<int>(rank),
                                     static_cast<int>(world_size), full_nvlink);
}

void dispose_ulysses_a2a(fptr_t _fa) { delete reinterpret_cast<fi::UlyssesA2A*>(_fa); }

// Fused-transpose Ulysses all-to-all.
//   mode == 0: inp [B, S_local, H, D]        -> out [B, S_global, H_local, D]
//   mode == 1: inp [B, S_global, H_local, D] -> out [B, S_local, H, D]
// where H is the *global* head count and H_local = H / world_size.
void ulysses_a2a(fptr_t _fa, TensorView inp, TensorView out, int64_t B, int64_t S_local, int64_t H,
                 int64_t D, int64_t mode) {
  auto fa = reinterpret_cast<fi::UlyssesA2A*>(_fa);
  ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  auto stream = get_stream(inp.device());

  TVM_FFI_ICHECK_EQ(inp.dtype(), out.dtype());
  TVM_FFI_ICHECK_EQ(inp.numel(), out.numel());
  TVM_FFI_ICHECK(mode == 0 || mode == 1) << "ulysses_a2a mode must be 0 or 1";
  const int W = fa->world_size_;
  TVM_FFI_ICHECK_EQ(H % W, 0) << "global head count must be divisible by world size";
  const int H_local = static_cast<int>(H / W);

  const int64_t num_rows = B * static_cast<int64_t>(W) * S_local;
  const int blocks =
      static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(fi::kMaxBlocks, num_rows)));
  const int threads = fi::kUlyssesThreads;
  const size_t out_bytes = out.numel() * get_element_size(out);

#define LAUNCH_ULYSSES_A2A(T, NG, MD)                                                              \
  fi::ulysses_a2a_kernel<T, NG, MD><<<blocks, threads, 0, stream>>>(                               \
      reinterpret_cast<const T*>(inp.data_ptr()), fa->out_ptrs_, fa->sg_, fa->self_sg_, fa->rank_, \
      static_cast<int>(B), static_cast<int>(S_local), H_local, static_cast<int>(D))

#define DISPATCH_NGPUS(T, MD)                                                       \
  switch (W) {                                                                      \
    case 2:                                                                         \
      LAUNCH_ULYSSES_A2A(T, 2, MD);                                                 \
      break;                                                                        \
    case 4:                                                                         \
      LAUNCH_ULYSSES_A2A(T, 4, MD);                                                 \
      break;                                                                        \
    case 6:                                                                         \
      LAUNCH_ULYSSES_A2A(T, 6, MD);                                                 \
      break;                                                                        \
    case 8:                                                                         \
      LAUNCH_ULYSSES_A2A(T, 8, MD);                                                 \
      break;                                                                        \
    default:                                                                        \
      TVM_FFI_ICHECK(false) << "ulysses_a2a only supports world size in (2,4,6,8)"; \
  }

#define DISPATCH_DTYPE(MD)                                                                \
  switch (encode_dlpack_dtype(out.dtype())) {                                             \
    case float32_code: {                                                                  \
      DISPATCH_NGPUS(float, MD);                                                          \
      break;                                                                              \
    }                                                                                     \
    case float16_code: {                                                                  \
      DISPATCH_NGPUS(half, MD);                                                           \
      break;                                                                              \
    }                                                                                     \
    case bfloat16_code: {                                                                 \
      DISPATCH_NGPUS(nv_bfloat16, MD);                                                    \
      break;                                                                              \
    }                                                                                     \
    default:                                                                              \
      TVM_FFI_ICHECK(false) << "ulysses_a2a only supports float32, float16 and bfloat16"; \
  }

  if (mode == 0) {
    DISPATCH_DTYPE(0);
  } else {
    DISPATCH_DTYPE(1);
  }

#undef DISPATCH_DTYPE
#undef DISPATCH_NGPUS
#undef LAUNCH_ULYSSES_A2A

  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
  // Copy this rank's completed result out of the staging buffer.
  auto status = cudaMemcpyAsync(out.data_ptr(), fa->local_out_buf_, out_bytes,
                                cudaMemcpyDeviceToDevice, stream);
  TVM_FFI_ICHECK(status == cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_ulysses_a2a, init_ulysses_a2a);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispose_ulysses_a2a, dispose_ulysses_a2a);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ulysses_a2a, ulysses_a2a);
