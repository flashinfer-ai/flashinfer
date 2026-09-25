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
// Generated source; do not edit manually.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

TVM_FFI_EMBED_CUBIN(mla_decode_value_split_paged_bf16_ee41d3ceea);

namespace mla_host_shim {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

inline int64_t MlaDeviceMultiprocessorCount(int device_id) {
  constexpr int kMaxCachedCudaDevices = 64;
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxCachedCudaDevices, RuntimeError)
      << "physical-SM-count cache does not cover cuda:" << device_id;
  static std::atomic<int> count_by_device[kMaxCachedCudaDevices]{};
  int cached = count_by_device[device_id].load(std::memory_order_acquire);
  if (cached > 0) return cached;

  int count = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device_id);
  TVM_FFI_CHECK(error == cudaSuccess && count > 0, RuntimeError)
      << "querying multiProcessorCount failed for cuda:" << device_id
      << ": cudaError=" << static_cast<int>(error) << ", count=" << count;
  // Concurrent first launches may repeat the immutable device query, but all
  // publication is atomic and every later hot-path lookup is one acquire load.
  count_by_device[device_id].store(count, std::memory_order_release);
  return count;
}

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckCurrentCudaDevice(
    const TensorView& reference,
    const char* reference_name) {
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed while validating " << reference_name
      << ": cudaError=" << static_cast<int>(error);
  TVM_FFI_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match " << reference_name
      << ": current=cuda:" << current_device
      << ", tensor=cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

#include <dlfcn.h>

// CUDA 13.4 adds an oversized shared-memory mode (function/launch attribute
// SHARED_MEMORY_MODE = 3) that lets a kernel exceed the standard
// MaxSharedMemoryPerBlockOptin ceiling up to device attribute 150
// (OversizedSharedMemoryPerBlock), at the cost of an 8 KiB L1 carveout.
#if (TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDA_VERSION) && CUDA_VERSION >= 13040) || \
    (!TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API && defined(CUDART_VERSION) && CUDART_VERSION >= 13040)
#define MLA_HAS_OVERSIZED_SMEM 1
#else
#define MLA_HAS_OVERSIZED_SMEM 0
#endif

// Per-device dynamic-SMEM opt-in. Returns true when launches on this device
// must carry the ALLOW_OVERSIZED shared-memory-mode launch attribute: the
// request exceeds the device's standard opt-in ceiling
// (MaxSharedMemoryPerBlockOptin, attribute 97) but fits the oversized ceiling
// (OversizedSharedMemoryPerBlock, attribute 150). Within the standard ceiling
// it sets MAX_DYNAMIC_SHARED_SIZE_BYTES for the device, mirroring
// CubinKernel::SetMaxDynamicSharedMemory. cache: 0 = unresolved,
// 1 = standard opt-in done, 2 = oversized mode required.
inline bool MlaConfigureDynamicSmem(tvm::ffi::CubinKernel& kernel, int device_id,
                                     int smem_bytes, signed char* cache, int cache_len) {
  namespace cuda_api = tvm::ffi::cuda_api;
  TVM_FFI_CHECK(device_id >= 0 && device_id < cache_len, RuntimeError)
      << "dynamic-SMEM opt-in cache does not cover cuda:" << device_id;
  if (cache[device_id] != 0) {
    return cache[device_id] == 2;
  }
  auto device = cuda_api::GetDeviceHandle(device_id);
  int optin_max = 0;
  cuda_api::ResultType err = cuda_api::GetDeviceAttribute(
      &optin_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN /
         cudaDevAttrMaxSharedMemoryPerBlockOptin */
      cuda_api::DeviceAttrType(97), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "querying MaxSharedMemoryPerBlockOptin failed for cuda:" << device_id;
  if (smem_bytes <= optin_max) {
    err = cuda_api::SetKernelMaxDynamicSharedMem(kernel.GetHandle(), smem_bytes, device);
    TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
        << "MAX_DYNAMIC_SHARED_SIZE_BYTES=" << smem_bytes << " rejected for cuda:" << device_id;
    cache[device_id] = 1;
    return false;
  }
#if MLA_HAS_OVERSIZED_SMEM
  int oversized_max = 0;
  err = cuda_api::GetDeviceAttribute(
      &oversized_max,
      /* CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK /
         cudaDevAttrOversizedSharedMemoryPerBlock */
      cuda_api::DeviceAttrType(150), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess && oversized_max >= smem_bytes, RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id << " and the oversized ceiling is "
      << oversized_max << " B";
  // Also set the function-level shared-memory mode: profiler-instrumented
  // launches (ncu) honor only the function-level attribute for oversized
  // dynamic SMEM and fail with LaunchFailed on the launch attribute alone.
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  err = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE,
                             CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY,
                             kernel.GetHandle(), device);
  TVM_FFI_CHECK(err == cuda_api::kSuccess, RuntimeError)
      << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
      << " (error " << static_cast<int>(err) << ")";
#else
  // The process cudart may predate 13.4 (e.g. torch's pip cudart 13.0) and
  // reject cudaFuncAttributeSharedMemoryMode with cudaErrorInvalidValue even
  // though the driver supports the mode, so call the driver entry point
  // directly: cudaKernel_t is interchangeable with CUkernel, the runtime
  // device ordinal is the CUdevice, and libcuda.so.1 is already loaded.
  {
    using MlaCuKernelSetAttributeFn = int (*)(int attrib, int val, void* kernel, int dev);
    static const auto mla_cu_kernel_set_attribute =
        reinterpret_cast<MlaCuKernelSetAttributeFn>(dlsym(RTLD_DEFAULT, "cuKernelSetAttribute"));
    TVM_FFI_CHECK(mla_cu_kernel_set_attribute != nullptr, RuntimeError)
        << "cuKernelSetAttribute not resolvable while enabling the oversized shared-memory mode";
    int drv_err = mla_cu_kernel_set_attribute(
        /* CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE */ 17,
        /* CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY */ 3,
        reinterpret_cast<void*>(kernel.GetHandle()), device_id);
    TVM_FFI_CHECK(drv_err == 0, RuntimeError)
        << "setting SHARED_MEMORY_MODE=ALLOW_OVERSIZED failed for cuda:" << device_id
        << " (driver error " << drv_err << ")";
  }
#endif
  cache[device_id] = 2;
  return true;
#else
  TVM_FFI_THROW(RuntimeError)
      << "dynamic smem " << smem_bytes << " B exceeds the standard opt-in ceiling ("
      << optin_max << " B) on cuda:" << device_id
      << " and this CUDA toolkit predates the 13.4 oversized shared-memory mode";
#endif
}

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  size_t used = 0;
};

// Immutable, process-lifetime device tensor-map slots for the pointer ABI.
// A slot is never rewritten: different descriptor bytes always get a new
// address, so concurrent streams cannot observe a partially updated map. The
// chunked arena caps storage at 512 KiB per CUDA context in this host module.
static inline void* TmaDeviceSlot(
    const CUtensorMap& tm,
    int device_id,
    cudaStream_t stream) {
  static std::mutex mu;
  static auto* slots = new std::unordered_map<std::string, void*>();
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();

  // Device allocations are context-owned. Resolve and validate the active
  // context before cache lookup so a warm entry can never bypass the same
  // checks as a cold entry or leak a pointer across contexts on one device.
  CUcontext current_context = nullptr;
  CUresult result = cuCtxGetCurrent(&current_context);
  TVM_FFI_CHECK(result == CUDA_SUCCESS && current_context != nullptr, RuntimeError)
      << "pointer TMA ABI requires an active CUDA context: CUresult="
      << static_cast<int>(result);
  CUdevice current_device = -1;
  result = cuCtxGetDevice(&current_device);
  TVM_FFI_CHECK(result == CUDA_SUCCESS && current_device == device_id, RuntimeError)
      << "TMA descriptor device mismatch: current=" << current_device
      << ", tensor=" << device_id;

  std::string key =
      std::to_string(reinterpret_cast<uintptr_t>(current_context));
  key.push_back(':');
  key.append(reinterpret_cast<const char*>(&tm), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(mu);

  // CUcontext is an opaque, recyclable handle.  Before trusting a warm arena,
  // prove that its first allocation still belongs to this live context.  A
  // destroyed context makes the pointer query fail; a subsequently reused
  // handle must therefore discard the stale pointer and descriptor entries.
  auto arena_it = arenas->find(current_context);
  if (arena_it != arenas->end() && !arena_it->second.chunks.empty()) {
    CUcontext allocation_context = nullptr;
    result = cuPointerGetAttribute(
        &allocation_context,
        CU_POINTER_ATTRIBUTE_CONTEXT,
        arena_it->second.chunks.front());
    if (result != CUDA_SUCCESS || allocation_context != current_context) {
      const std::string prefix =
          std::to_string(reinterpret_cast<uintptr_t>(current_context)) + ":";
      for (auto slot_it = slots->begin(); slot_it != slots->end();) {
        if (slot_it->first.compare(0, prefix.size(), prefix) == 0) {
          slot_it = slots->erase(slot_it);
        } else {
          ++slot_it;
        }
      }
      arenas->erase(arena_it);
    }
  }

  auto it = slots->find(key);
  if (it != slots->end()) return it->second;

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(
      reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuStreamIsCapturing for TMA descriptor slot failed: CUresult="
      << static_cast<int>(result);
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "pointer TMA ABI cannot create a new device descriptor slot inside "
         "CUDA Graph capture; prewarm this exact tensor/layout binding or "
         "compile with tma_abi='grid_constant'";

  TmaDeviceArena& arena = (*arenas)[current_context];
  TVM_FFI_CHECK(arena.used < TmaDeviceArena::kMaxSlots, RuntimeError)
      << "pointer TMA ABI exhausted its immutable descriptor arena in CUDA "
         "context " << current_context << " on device " << device_id
      << " (capacity=" << TmaDeviceArena::kMaxSlots
      << "); reuse tensor/layout bindings or compile with tma_abi='grid_constant'";
  if (arena.used % TmaDeviceArena::kSlotsPerChunk == 0) {
    CUdeviceptr chunk = 0;
    result = cuMemAlloc(
        &chunk,
        TmaDeviceArena::kSlotsPerChunk * sizeof(CUtensorMap));
    TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
        << "cuMemAlloc for TMA descriptor arena failed: CUresult="
        << static_cast<int>(result);
    arena.chunks.push_back(chunk);
  }
  size_t chunk_index = arena.used / TmaDeviceArena::kSlotsPerChunk;
  size_t slot_index = arena.used % TmaDeviceArena::kSlotsPerChunk;
  CUdeviceptr dev = arena.chunks[chunk_index] +
                    slot_index * sizeof(CUtensorMap);
  result = cuMemcpyHtoD(dev, &tm, sizeof(CUtensorMap));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuMemcpyHtoD for TMA descriptor slot failed: CUresult="
      << static_cast<int>(result);
  ++arena.used;
  void* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(dev));
  (*slots)[key] = pointer;
  return pointer;
}

namespace variant_mla_decode_value_split_paged_bf16_ee41d3ceea_3d9d1dc71fbc {

// 3D TMA descriptor for buffer 'tmap_q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'tmap_q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "tmap_q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'tmap_q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'tmap_q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (64, 64, 1) exceeds resolved global dims for 'tmap_q'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'tmap_q') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'tmap_k_nope' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_k_nope(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_k_nope' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_k_nope' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'tmap_k_nope' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'tmap_k_nope' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 32u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'tmap_k_nope'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_nope' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'tmap_k_nope') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'tmap_k_rope' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_k_rope(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_k_rope' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_k_rope' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'tmap_k_rope' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'tmap_k_rope' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 32u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'tmap_k_rope'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_k_rope' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'tmap_k_rope') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'tmap_v' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_v(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_v' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_v' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'tmap_v' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'tmap_v' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 32u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'tmap_v'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = 64;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'tmap_v' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'tmap_v') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_tmap_q, TensorView arg_tmap_k_nope, TensorView arg_tmap_k_rope, TensorView arg_tmap_v, TensorView arg_O, TensorView arg_seq_lens_kv, TensorView arg_page_table, TensorView arg_sinks, double arg_softmax_scale_log2, double arg_bmm2_scale, int64_t arg_total_work_items, int64_t arg_value_split_count, int64_t arg_max_pages_per_seq, int64_t arg_enable_sink, int64_t grid_x, int64_t grid_y, int64_t grid_z, cudaStream_t stream) {
  DLDevice dev = arg_tmap_q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_tmap_q, "tmap_q");
  CheckDtype(arg_tmap_q, "tmap_q", 4, 16, 1);
  CheckCudaTensor(arg_tmap_k_nope, "tmap_k_nope");
  CheckDtype(arg_tmap_k_nope, "tmap_k_nope", 4, 16, 1);
  CheckContiguous(arg_tmap_k_nope, "tmap_k_nope");
  CheckCudaTensor(arg_tmap_k_rope, "tmap_k_rope");
  CheckDtype(arg_tmap_k_rope, "tmap_k_rope", 4, 16, 1);
  CheckContiguous(arg_tmap_k_rope, "tmap_k_rope");
  CheckCudaTensor(arg_tmap_v, "tmap_v");
  CheckDtype(arg_tmap_v, "tmap_v", 4, 16, 1);
  CheckContiguous(arg_tmap_v, "tmap_v");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_seq_lens_kv, "seq_lens_kv");
  CheckDtype(arg_seq_lens_kv, "seq_lens_kv", 0, 32, 1);
  CheckContiguous(arg_seq_lens_kv, "seq_lens_kv");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckDtype(arg_page_table, "page_table", 0, 32, 1);
  CheckContiguous(arg_page_table, "page_table");
  CheckCudaTensor(arg_sinks, "sinks");
  CheckDtype(arg_sinks, "sinks", 2, 32, 1);
  CheckContiguous(arg_sinks, "sinks");
  TVM_FFI_CHECK(arg_total_work_items >= -2147483648LL && arg_total_work_items <= 2147483647LL, ValueError)
      << "scalar 'total_work_items' value " << arg_total_work_items
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_value_split_count >= -2147483648LL && arg_value_split_count <= 2147483647LL, ValueError)
      << "scalar 'value_split_count' value " << arg_value_split_count
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_pages_per_seq >= -2147483648LL && arg_max_pages_per_seq <= 2147483647LL, ValueError)
      << "scalar 'max_pages_per_seq' value " << arg_max_pages_per_seq
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_enable_sink >= -2147483648LL && arg_enable_sink <= 2147483647LL, ValueError)
      << "scalar 'enable_sink' value " << arg_enable_sink
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_tmap_k_nope, arg_tmap_q, "tmap_k_nope", "tmap_q");
  CheckSameCudaDevice(arg_tmap_k_rope, arg_tmap_q, "tmap_k_rope", "tmap_q");
  CheckSameCudaDevice(arg_tmap_v, arg_tmap_q, "tmap_v", "tmap_q");
  CheckSameCudaDevice(arg_O, arg_tmap_q, "O", "tmap_q");
  CheckSameCudaDevice(arg_seq_lens_kv, arg_tmap_q, "seq_lens_kv", "tmap_q");
  CheckSameCudaDevice(arg_page_table, arg_tmap_q, "page_table", "tmap_q");
  CheckSameCudaDevice(arg_sinks, arg_tmap_q, "sinks", "tmap_q");
  CheckCurrentCudaDevice(arg_tmap_q, "tmap_q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  TVM_FFI_CHECK(grid_x % 2 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";


  CUtensorMap h_tmap_q = EncodeTma_tmap_q(arg_tmap_q);
  void* p_tmap_q = TmaDeviceSlot(h_tmap_q, arg_tmap_q.device().device_id, stream);
  CUtensorMap h_tmap_k_nope = EncodeTma_tmap_k_nope(arg_tmap_k_nope);
  void* p_tmap_k_nope = TmaDeviceSlot(h_tmap_k_nope, arg_tmap_k_nope.device().device_id, stream);
  CUtensorMap h_tmap_k_rope = EncodeTma_tmap_k_rope(arg_tmap_k_rope);
  void* p_tmap_k_rope = TmaDeviceSlot(h_tmap_k_rope, arg_tmap_k_rope.device().device_id, stream);
  CUtensorMap h_tmap_v = EncodeTma_tmap_v(arg_tmap_v);
  void* p_tmap_v = TmaDeviceSlot(h_tmap_v, arg_tmap_v.device().device_id, stream);
  void* p_O = arg_O.data_ptr();
  void* p_seq_lens_kv = arg_seq_lens_kv.data_ptr();
  void* p_page_table = arg_page_table.data_ptr();
  void* p_sinks = arg_sinks.data_ptr();
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  float v_bmm2_scale = (float)arg_bmm2_scale;
  int32_t v_total_work_items = (int32_t)arg_total_work_items;
  int32_t v_value_split_count = (int32_t)arg_value_split_count;
  int32_t v_max_pages_per_seq = (int32_t)arg_max_pages_per_seq;
  int32_t v_enable_sink = (int32_t)arg_enable_sink;
  void* kargs[] = {&p_tmap_q, &p_tmap_k_nope, &p_tmap_k_rope, &p_tmap_v, &p_O, &p_seq_lens_kv, &p_page_table, &p_sinks, &v_softmax_scale_log2, &v_bmm2_scale, &v_total_work_items, &v_value_split_count, &v_max_pages_per_seq, &v_enable_sink};

  static auto kernel = EmbedCubinModule_mla_decode_value_split_paged_bf16_ee41d3ceea::Global()->mod.GetKernel("kernel_mla_decode_value_split_paged_bf16");
  static signed char mla_smem_mode_cache[64] = {0};
  const bool use_oversized_smem = MlaConfigureDynamicSmem(
      kernel, (int)arg_tmap_q.device().device_id, 231424,
      mla_smem_mode_cache, 64);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(384u, 1u, 1u);

  // Extended launch — mirrors CUDAKernel.prepare_launch_cluster attributes.
  tvm::ffi::cuda_api::LaunchConfig config;
  int n = 0;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attrs[2];
  attrs[n].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attrs[n].value.clusterDim.x = 2u;
  attrs[n].value.clusterDim.y = 1u;
  attrs[n].value.clusterDim.z = 1u;
  ++n;
#if MLA_HAS_OVERSIZED_SMEM
  if (use_oversized_smem) {
    attrs[n].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
    attrs[n].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
    ++n;
  }
#endif
  config.gridDimX = grid.x;
  config.gridDimY = grid.y;
  config.gridDimZ = grid.z;
  config.blockDimX = block.x;
  config.blockDimY = block.y;
  config.blockDimZ = block.z;
  config.sharedMemBytes = 231424u;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#else
  cudaLaunchAttribute attrs[2];
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = 2u;
  attrs[n].val.clusterDim.y = 1u;
  attrs[n].val.clusterDim.z = 1u;
  ++n;
#if MLA_HAS_OVERSIZED_SMEM
  if (use_oversized_smem) {
    attrs[n].id = cudaLaunchAttributeSharedMemoryMode;
    attrs[n].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
    ++n;
  }
#endif
  config.gridDim = {grid.x, grid.y, grid.z};
  config.blockDim = {block.x, block.y, block.z};
  config.dynamicSmemBytes = 231424u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(kargs, config));
}
}  // namespace variant_mla_decode_value_split_paged_bf16_ee41d3ceea_3d9d1dc71fbc

int SelectVariant(float softmax_scale_log2, float bmm2_scale, int32_t total_work_items, int32_t value_split_count, int32_t max_pages_per_seq, int32_t enable_sink, int32_t grid_x, int32_t grid_z) {
  if ((((total_work_items > 0) && (grid_x > 0)) && (grid_z > 0))) return 0;
  return -1;
}

void Dispatch(tvm::ffi::TensorView arg_q_rows, tvm::ffi::TensorView arg_kv_pages, tvm::ffi::TensorView arg_output, tvm::ffi::TensorView arg_seq_lens, tvm::ffi::TensorView arg_page_table, tvm::ffi::TensorView arg_sinks, float softmax_scale_log2, float bmm2_scale, int32_t total_work_items, int32_t value_split_count, int32_t max_pages_per_seq, int32_t enable_sink, int32_t grid_x, int32_t grid_z, int64_t cuda_stream_ptr) {
  CheckCudaTensor(arg_q_rows, "q_rows");
  CheckCudaTensor(arg_kv_pages, "kv_pages");
  CheckCudaTensor(arg_output, "output");
  CheckCudaTensor(arg_seq_lens, "seq_lens");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckCudaTensor(arg_sinks, "sinks");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream_ptr));
  int v = SelectVariant(softmax_scale_log2, bmm2_scale, total_work_items, value_split_count, max_pages_per_seq, enable_sink, grid_x, grid_z);
  switch (v) {
    case 0: {
      {
        int64_t gx = grid_x;
        int64_t gy = 1;
        int64_t gz = grid_z;
        mla_host_shim::variant_mla_decode_value_split_paged_bf16_ee41d3ceea_3d9d1dc71fbc::Run(arg_q_rows, arg_kv_pages, arg_kv_pages, arg_kv_pages, arg_output, arg_seq_lens, arg_page_table, arg_sinks, softmax_scale_log2, bmm2_scale, total_work_items, value_split_count, max_pages_per_seq, enable_sink, gx, gy, gz, stream);
      }
      return;
    }
    default:
      TVM_FFI_CHECK(false, ValueError) << "no dispatch route for mla_bf16_vquarter shape (softmax_scale_log2=" << softmax_scale_log2 << ", bmm2_scale=" << bmm2_scale << ", total_work_items=" << total_work_items << ", value_split_count=" << value_split_count << ", max_pages_per_seq=" << max_pages_per_seq << ", enable_sink=" << enable_sink << ", grid_x=" << grid_x << ", grid_z=" << grid_z << ")";
  }
}

int64_t RouteOf(float softmax_scale_log2, float bmm2_scale, int32_t total_work_items, int32_t value_split_count, int32_t max_pages_per_seq, int32_t enable_sink, int32_t grid_x, int32_t grid_z) { return SelectVariant(softmax_scale_log2, bmm2_scale, total_work_items, value_split_count, max_pages_per_seq, enable_sink, grid_x, grid_z); }

}  // namespace mla_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mla_host_shim::Dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_of, mla_host_shim::RouteOf);
