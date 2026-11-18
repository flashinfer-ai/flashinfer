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

// The SM count of a CUDA stream's context, read by every small-M decode launch of
// flashinfer.cake_fmha to plan its split count on the SMs the stream holds (a green
// context's partition).  One FFI call per launch: the same queries through the
// Python driver bindings cost several microseconds of host time per launch.
//
// The legacy and per-thread default streams run on the thread's current context
// (``cuCtxGetCurrent``: a green context made current runs them on its partition).
// Any other stream's context is ``cuStreamGetCtx``; where that does not answer
// (before CUDA 13 it returns CUDA_ERROR_NOT_SUPPORTED for a stream of
// ``cuGreenCtxStreamCreate``) the stream's green context (``cuStreamGetCtx_v2``,
// ``cuCtxFromGreenCtx``), else the current context.  A device's primary context
// is retained once and known by its handle, so a launch on one of its streams
// makes one driver call; any other context's count is cached per ``cuCtxGetId``
// (IDs are never reused, unlike handles).  No ``cuStreamGetId``: on a stream
// being captured into a CUDA graph it returns CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
// and invalidates the capture, while every query here is legal during a capture.

#include <cuda.h>
#include <tvm/ffi/function.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

using StreamGetCtx = CUresult(CUDAAPI*)(CUstream, CUcontext*);
using StreamGetCtxV2 = CUresult(CUDAAPI*)(CUstream, CUcontext*, CUgreenCtx*);
using CtxFromGreenCtx = CUresult(CUDAAPI*)(CUcontext*, CUgreenCtx);
using CtxGetId = CUresult(CUDAAPI*)(CUcontext, unsigned long long*);
using CtxGetDevResource = CUresult(CUDAAPI*)(CUcontext, CUdevResource*, CUdevResourceType);

// The entry point of ``name`` for driver ABI ``version`` (``nullptr`` when the
// driver does not have it).  Entry points newer than the driver are never asked
// for: an older driver would answer with the older ABI.
template <typename Proc>
Proc LoadDriverProc(char const* name, int version) {
  void* address = nullptr;
  CUdriverProcAddressQueryResult status{};
  if (cuGetProcAddress(name, &address, version, CU_GET_PROC_ADDRESS_DEFAULT, &status) !=
          CUDA_SUCCESS ||
      status != CU_GET_PROC_ADDRESS_SUCCESS) {
    return nullptr;
  }
  return reinterpret_cast<Proc>(address);
}

struct DriverApi {
  StreamGetCtx stream_ctx = nullptr;
  StreamGetCtxV2 stream_ctx_v2 = nullptr;
  CtxFromGreenCtx ctx_from_green_ctx = nullptr;
  CtxGetId ctx_id = nullptr;
  CtxGetDevResource ctx_dev_resource = nullptr;

  DriverApi() {
    int driver = 0;
    if (cuDriverGetVersion(&driver) != CUDA_SUCCESS) return;
    stream_ctx = LoadDriverProc<StreamGetCtx>("cuStreamGetCtx", 9020);
    if (driver >= 12000) ctx_id = LoadDriverProc<CtxGetId>("cuCtxGetId", 12000);
    if (driver >= 12040) {
      ctx_from_green_ctx = LoadDriverProc<CtxFromGreenCtx>("cuCtxFromGreenCtx", 12040);
      ctx_dev_resource = LoadDriverProc<CtxGetDevResource>("cuCtxGetDevResource", 12040);
    }
    if (driver >= 12050) stream_ctx_v2 = LoadDriverProc<StreamGetCtxV2>("cuStreamGetCtx", 12050);
  }
};

DriverApi const& Driver() {
  static DriverApi const api;
  return api;
}

constexpr int64_t kMaxDevices = 64;

struct PrimaryContext {
  std::once_flag retained;
  CUcontext handle = nullptr;
  std::atomic<int> sm_count{0};
};

std::array<PrimaryContext, kMaxDevices> g_primary_contexts;
std::mutex g_mutex;
std::unordered_map<unsigned long long, int> g_sm_counts;  // context ID -> SM count

// The SM count of ``context``, read once per context ID (-1 when unknown).
int ContextSmCount(DriverApi const& api, CUcontext context) {
  if (api.ctx_dev_resource == nullptr) return -1;
  unsigned long long id = 0;
  bool const has_id = api.ctx_id != nullptr && api.ctx_id(context, &id) == CUDA_SUCCESS;
  if (has_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto const it = g_sm_counts.find(id);
    if (it != g_sm_counts.end()) return it->second;
  }
  CUdevResource resource{};
  if (api.ctx_dev_resource(context, &resource, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS) {
    return -1;
  }
  int const count = static_cast<int>(resource.sm.smCount);
  if (has_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sm_counts.emplace(id, count);
  }
  return count;
}

// The context of ``stream``'s green context (``nullptr`` without one).
CUcontext GreenContextOf(DriverApi const& api, CUstream stream) {
  if (api.stream_ctx_v2 == nullptr || api.ctx_from_green_ctx == nullptr) return nullptr;
  CUcontext context = nullptr;
  CUgreenCtx green = nullptr;
  if (api.stream_ctx_v2(stream, &context, &green) != CUDA_SUCCESS || green == nullptr) {
    return nullptr;
  }
  context = nullptr;
  return api.ctx_from_green_ctx(&context, green) == CUDA_SUCCESS ? context : nullptr;
}

int64_t SmCount(int64_t stream_handle, int64_t device, bool ask_stream) {
  DriverApi const& api = Driver();
  CUstream const stream = reinterpret_cast<CUstream>(static_cast<uintptr_t>(stream_handle));
  CUcontext context = nullptr;
  // The legacy and per-thread default streams (``0``, ``CU_STREAM_LEGACY``,
  // ``CU_STREAM_PER_THREAD``) run on the thread's current context, which
  // ``cuCtxGetCurrent`` reads without looking the stream up.
  if (stream_handle >= 0 && stream_handle <= 2) {
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) return -1;
  } else if (!ask_stream || api.stream_ctx == nullptr ||
             api.stream_ctx(stream, &context) != CUDA_SUCCESS) {
    context = nullptr;
  }
  if (context == nullptr) {
    context = GreenContextOf(api, stream);
    if (context == nullptr && (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)) {
      return -1;
    }
  }
  if (device >= 0 && device < kMaxDevices) {
    PrimaryContext& primary = g_primary_contexts[device];
    // A retained context is never destroyed, so no other context gets its handle.
    std::call_once(primary.retained, [&] {
      CUdevice cu_device = 0;
      CUcontext handle = nullptr;
      if (cuDeviceGet(&cu_device, static_cast<int>(device)) == CUDA_SUCCESS &&
          cuDevicePrimaryCtxRetain(&handle, cu_device) == CUDA_SUCCESS) {
        primary.handle = handle;
      }
    });
    if (primary.handle != nullptr && context == primary.handle) {
      int count = primary.sm_count.load(std::memory_order_acquire);
      if (count <= 0) {
        count = ContextSmCount(api, context);
        if (count > 0) primary.sm_count.store(count, std::memory_order_release);
      }
      return count;
    }
  }
  return ContextSmCount(api, context);
}

// The SM count of the context of ``stream`` (a raw ``CUstream`` handle) on CUDA
// device ``device``; -1 when the driver cannot tell.
int64_t StreamContextSmCount(int64_t stream, int64_t device) {
  return SmCount(stream, device, true);
}

// The same without asking ``cuStreamGetCtx``, as a pre-CUDA-13 driver answers
// for a green context's stream (tests only).
int64_t StreamContextSmCountUnanswered(int64_t stream, int64_t device) {
  return SmCount(stream, device, false);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stream_context_sm_count, StreamContextSmCount);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stream_context_sm_count_unanswered, StreamContextSmCountUnanswered);
