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

TVM_FFI_EMBED_CUBIN(fp8_p32_exact_producer_c3bae9716e);

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

namespace variant_fp8_p32_exact_producer_c3bae9716e_ad932b8f296a {

// 2D TMA descriptor for buffer 'Q_map' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q_map(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q_map' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q_map' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q_map' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "Q_map");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q_map' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'Q_map' resolved a non-positive global dim";
  TVM_FFI_CHECK(16u <= global_dim[1], ValueError)
      << "TMA box (128, 16) exceeds resolved global dims for 'Q_map'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'Q_map' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'Q_map' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'Q_map' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 16u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'Q_map') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'K_map' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K_map(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'K_map' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K_map' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'K_map' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "K_map");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K_map' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'K_map' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[1], ValueError)
      << "TMA box (128, 32) exceeds resolved global dims for 'K_map'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'K_map' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'K_map' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'K_map' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'K_map') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'V_map' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_V_map(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'V_map' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'V_map' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'V_map' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "V_map");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'V_map' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'V_map' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[1], ValueError)
      << "TMA box (128, 32) exceeds resolved global dims for 'V_map'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'V_map' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'V_map' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 8) % 8 == 0, ValueError)
      << "TMA descriptor for 'V_map' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'V_map') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Q_map, TensorView arg_K_map, TensorView arg_V_map, TensorView arg_page_table, TensorView arg_seq_lens, TensorView arg_partial_O, TensorView arg_partial_stats, TensorView arg_O, TensorView arg_completion, int64_t arg_q_len, int64_t arg_page_table_stride, int64_t arg_max_num_ctas_q, int64_t arg_max_num_ctas_kv, double arg_bmm1_scale_log2, double arg_bmm2_scale, int64_t grid_x, int64_t grid_y, int64_t grid_z, cudaStream_t stream) {
  DLDevice dev = arg_Q_map.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_Q_map, "Q_map");
  CheckDtype(arg_Q_map, "Q_map", 1, 8, 1);
  CheckCudaTensor(arg_K_map, "K_map");
  CheckDtype(arg_K_map, "K_map", 1, 8, 1);
  CheckCudaTensor(arg_V_map, "V_map");
  CheckDtype(arg_V_map, "V_map", 1, 8, 1);
  CheckCudaTensor(arg_page_table, "page_table");
  CheckDtype(arg_page_table, "page_table", 0, 32, 1);
  CheckContiguous(arg_page_table, "page_table");
  CheckCudaTensor(arg_seq_lens, "seq_lens");
  CheckDtype(arg_seq_lens, "seq_lens", 0, 32, 1);
  CheckContiguous(arg_seq_lens, "seq_lens");
  CheckCudaTensor(arg_partial_O, "partial_O");
  CheckDtype(arg_partial_O, "partial_O", 4, 16, 1);
  CheckContiguous(arg_partial_O, "partial_O");
  CheckCudaTensor(arg_partial_stats, "partial_stats");
  CheckDtype(arg_partial_stats, "partial_stats", 2, 32, 1);
  CheckContiguous(arg_partial_stats, "partial_stats");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_completion, "completion");
  CheckDtype(arg_completion, "completion", 1, 32, 1);
  CheckContiguous(arg_completion, "completion");
  TVM_FFI_CHECK(arg_q_len >= -2147483648LL && arg_q_len <= 2147483647LL, ValueError)
      << "scalar 'q_len' value " << arg_q_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_page_table_stride >= -2147483648LL && arg_page_table_stride <= 2147483647LL, ValueError)
      << "scalar 'page_table_stride' value " << arg_page_table_stride
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_num_ctas_q >= -2147483648LL && arg_max_num_ctas_q <= 2147483647LL, ValueError)
      << "scalar 'max_num_ctas_q' value " << arg_max_num_ctas_q
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_num_ctas_kv >= -2147483648LL && arg_max_num_ctas_kv <= 2147483647LL, ValueError)
      << "scalar 'max_num_ctas_kv' value " << arg_max_num_ctas_kv
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_K_map, arg_Q_map, "K_map", "Q_map");
  CheckSameCudaDevice(arg_V_map, arg_Q_map, "V_map", "Q_map");
  CheckSameCudaDevice(arg_page_table, arg_Q_map, "page_table", "Q_map");
  CheckSameCudaDevice(arg_seq_lens, arg_Q_map, "seq_lens", "Q_map");
  CheckSameCudaDevice(arg_partial_O, arg_Q_map, "partial_O", "Q_map");
  CheckSameCudaDevice(arg_partial_stats, arg_Q_map, "partial_stats", "Q_map");
  CheckSameCudaDevice(arg_O, arg_Q_map, "O", "Q_map");
  CheckSameCudaDevice(arg_completion, arg_Q_map, "completion", "Q_map");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";


  CUtensorMap p_Q_map = EncodeTma_Q_map(arg_Q_map);
  CUtensorMap p_K_map = EncodeTma_K_map(arg_K_map);
  CUtensorMap p_V_map = EncodeTma_V_map(arg_V_map);
  void* p_page_table = arg_page_table.data_ptr();
  void* p_seq_lens = arg_seq_lens.data_ptr();
  void* p_partial_O = arg_partial_O.data_ptr();
  void* p_partial_stats = arg_partial_stats.data_ptr();
  void* p_O = arg_O.data_ptr();
  void* p_completion = arg_completion.data_ptr();
  int32_t v_q_len = (int32_t)arg_q_len;
  int32_t v_page_table_stride = (int32_t)arg_page_table_stride;
  int32_t v_max_num_ctas_q = (int32_t)arg_max_num_ctas_q;
  int32_t v_max_num_ctas_kv = (int32_t)arg_max_num_ctas_kv;
  float v_bmm1_scale_log2 = (float)arg_bmm1_scale_log2;
  float v_bmm2_scale = (float)arg_bmm2_scale;
  void* kargs[] = {&p_Q_map, &p_K_map, &p_V_map, &p_page_table, &p_seq_lens, &p_partial_O, &p_partial_stats, &p_O, &p_completion, &v_q_len, &v_page_table_stride, &v_max_num_ctas_q, &v_max_num_ctas_kv, &v_bmm1_scale_log2, &v_bmm2_scale};

  static auto kernel = EmbedCubinModule_fp8_p32_exact_producer_c3bae9716e::Global()->mod.GetKernel("kernel_fp8_p32_exact_producer");
  static signed char mla_smem_mode_cache[64] = {0};
  const bool use_oversized_smem = MlaConfigureDynamicSmem(
      kernel, (int)arg_Q_map.device().device_id, 178560,
      mla_smem_mode_cache, 64);
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(512u, 1u, 1u);

  if (use_oversized_smem) {

  // Extended launch — mirrors CUDAKernel.prepare_launch_cluster attributes.
  tvm::ffi::cuda_api::LaunchConfig config;
  int n = 0;
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attrs[1];
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
  config.sharedMemBytes = 178560u;
  config.hStream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#else
  cudaLaunchAttribute attrs[1];
#if MLA_HAS_OVERSIZED_SMEM
  if (use_oversized_smem) {
    attrs[n].id = cudaLaunchAttributeSharedMemoryMode;
    attrs[n].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
    ++n;
  }
#endif
  config.gridDim = {grid.x, grid.y, grid.z};
  config.blockDim = {block.x, block.y, block.z};
  config.dynamicSmemBytes = 178560u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = n;
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(kargs, config));

  } else {
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 178560u));
  }
}
}  // namespace variant_fp8_p32_exact_producer_c3bae9716e_ad932b8f296a

int SelectVariant(int32_t batch, int32_t q_len, int32_t page_table_stride, int32_t max_num_ctas_q, int32_t max_num_ctas_kv, float bmm1_scale_log2, float bmm2_scale) {
  if (((batch > 0) && (q_len > 0))) return 0;
  return -1;
}

void Dispatch(tvm::ffi::TensorView arg_query, tvm::ffi::TensorView arg_kv, tvm::ffi::TensorView arg_page_table, tvm::ffi::TensorView arg_seq_lens, tvm::ffi::TensorView arg_output, tvm::ffi::TensorView arg_completion, tvm::ffi::TensorView arg_ws_partial_output, tvm::ffi::TensorView arg_ws_partial_stats, int32_t batch, int32_t q_len, int32_t page_table_stride, int32_t max_num_ctas_q, int32_t max_num_ctas_kv, float bmm1_scale_log2, float bmm2_scale, int64_t cuda_stream_ptr) {
  CheckCudaTensor(arg_query, "query");
  CheckCudaTensor(arg_kv, "kv");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckCudaTensor(arg_seq_lens, "seq_lens");
  CheckCudaTensor(arg_output, "output");
  CheckCudaTensor(arg_completion, "completion");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream_ptr));
  int v = SelectVariant(batch, q_len, page_table_stride, max_num_ctas_q, max_num_ctas_kv, bmm1_scale_log2, bmm2_scale);
  switch (v) {
    case 0: {
      CheckCudaTensor(arg_ws_partial_output, "partial_output");
      CheckDtype(arg_ws_partial_output, "partial_output", 4, 16, 1);
      TVM_FFI_CHECK(arg_ws_partial_output.numel() == (int64_t)((((batch * 32) * q_len)) * (2) * (16) * (128)), ValueError)
          << "workspace 'partial_output' numel " << arg_ws_partial_output.numel()
          << " != expected " << (int64_t)((((batch * 32) * q_len)) * (2) * (16) * (128));
      CheckCudaTensor(arg_ws_partial_stats, "partial_stats");
      CheckDtype(arg_ws_partial_stats, "partial_stats", 2, 32, 1);
      TVM_FFI_CHECK(arg_ws_partial_stats.numel() == (int64_t)((((batch * 32) * q_len)) * (2) * (16) * (2)), ValueError)
          << "workspace 'partial_stats' numel " << arg_ws_partial_stats.numel()
          << " != expected " << (int64_t)((((batch * 32) * q_len)) * (2) * (16) * (2));
      {
        int64_t gx = (q_len * 2);
        int64_t gy = 32;
        int64_t gz = batch;
        mla_host_shim::variant_fp8_p32_exact_producer_c3bae9716e_ad932b8f296a::Run(arg_query, arg_kv, arg_kv, arg_page_table, arg_seq_lens, arg_ws_partial_output, arg_ws_partial_stats, arg_output, arg_completion, q_len, page_table_stride, max_num_ctas_q, max_num_ctas_kv, bmm1_scale_log2, bmm2_scale, gx, gy, gz, stream);
      }
      return;
    }
    default:
      TVM_FFI_CHECK(false, ValueError) << "no dispatch route for mla_fp8_p32_qk_l2 shape (batch=" << batch << ", q_len=" << q_len << ", page_table_stride=" << page_table_stride << ", max_num_ctas_q=" << max_num_ctas_q << ", max_num_ctas_kv=" << max_num_ctas_kv << ", bmm1_scale_log2=" << bmm1_scale_log2 << ", bmm2_scale=" << bmm2_scale << ")";
  }
}

int64_t RouteOf(int32_t batch, int32_t q_len, int32_t page_table_stride, int32_t max_num_ctas_q, int32_t max_num_ctas_kv, float bmm1_scale_log2, float bmm2_scale) { return SelectVariant(batch, q_len, page_table_stride, max_num_ctas_q, max_num_ctas_kv, bmm1_scale_log2, bmm2_scale); }

}  // namespace mla_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mla_host_shim::Dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_of, mla_host_shim::RouteOf);
