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

TVM_FFI_EMBED_CUBIN(full_abi_tail_bf16_83451b7317);

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

namespace variant_full_abi_tail_bf16_83451b7317_d1d6ab272ce6 {

void Run(TensorView arg_Q, TensorView arg_KV, TensorView arg_fp8_lut, TensorView arg_page_table, TensorView arg_sparse_indices, TensorView arg_row_batches, TensorView arg_row_seq_lens, TensorView arg_O, TensorView arg_LSE, TensorView arg_sinks, int64_t arg_num_heads, int64_t arg_qk_dim, int64_t arg_value_dim, int64_t arg_kv_stride, int64_t arg_page_size, int64_t arg_page_table_width, int64_t arg_sparse_width, int64_t arg_use_sparse, double arg_softmax_scale, double arg_bmm2_scale, int64_t arg_enable_sink, int64_t arg_write_lse, int64_t grid_x, int64_t grid_y, int64_t grid_z, cudaStream_t stream) {
  DLDevice dev = arg_Q.device();
  ScopedCudaDevice device_guard(dev.device_id);
  CheckCudaTensor(arg_Q, "Q");
  CheckDtype(arg_Q, "Q", 4, 16, 1);
  CheckContiguous(arg_Q, "Q");
  CheckCudaTensor(arg_KV, "KV");
  CheckDtype(arg_KV, "KV", 4, 16, 1);
  CheckContiguous(arg_KV, "KV");
  CheckCudaTensor(arg_fp8_lut, "fp8_lut");
  CheckDtype(arg_fp8_lut, "fp8_lut", 2, 32, 1);
  CheckContiguous(arg_fp8_lut, "fp8_lut");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckDtype(arg_page_table, "page_table", 0, 32, 1);
  CheckContiguous(arg_page_table, "page_table");
  CheckCudaTensor(arg_sparse_indices, "sparse_indices");
  CheckDtype(arg_sparse_indices, "sparse_indices", 0, 32, 1);
  CheckContiguous(arg_sparse_indices, "sparse_indices");
  CheckCudaTensor(arg_row_batches, "row_batches");
  CheckDtype(arg_row_batches, "row_batches", 0, 32, 1);
  CheckContiguous(arg_row_batches, "row_batches");
  CheckCudaTensor(arg_row_seq_lens, "row_seq_lens");
  CheckDtype(arg_row_seq_lens, "row_seq_lens", 0, 32, 1);
  CheckContiguous(arg_row_seq_lens, "row_seq_lens");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_LSE, "LSE");
  CheckDtype(arg_LSE, "LSE", 2, 32, 1);
  CheckContiguous(arg_LSE, "LSE");
  CheckCudaTensor(arg_sinks, "sinks");
  CheckDtype(arg_sinks, "sinks", 2, 32, 1);
  CheckContiguous(arg_sinks, "sinks");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_qk_dim >= -2147483648LL && arg_qk_dim <= 2147483647LL, ValueError)
      << "scalar 'qk_dim' value " << arg_qk_dim
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_value_dim >= -2147483648LL && arg_value_dim <= 2147483647LL, ValueError)
      << "scalar 'value_dim' value " << arg_value_dim
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_kv_stride >= -2147483648LL && arg_kv_stride <= 2147483647LL, ValueError)
      << "scalar 'kv_stride' value " << arg_kv_stride
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_page_size >= -2147483648LL && arg_page_size <= 2147483647LL, ValueError)
      << "scalar 'page_size' value " << arg_page_size
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_page_table_width >= -2147483648LL && arg_page_table_width <= 2147483647LL, ValueError)
      << "scalar 'page_table_width' value " << arg_page_table_width
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sparse_width >= -2147483648LL && arg_sparse_width <= 2147483647LL, ValueError)
      << "scalar 'sparse_width' value " << arg_sparse_width
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_use_sparse >= -2147483648LL && arg_use_sparse <= 2147483647LL, ValueError)
      << "scalar 'use_sparse' value " << arg_use_sparse
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_enable_sink >= -2147483648LL && arg_enable_sink <= 2147483647LL, ValueError)
      << "scalar 'enable_sink' value " << arg_enable_sink
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_write_lse >= -2147483648LL && arg_write_lse <= 2147483647LL, ValueError)
      << "scalar 'write_lse' value " << arg_write_lse
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_KV, arg_Q, "KV", "Q");
  CheckSameCudaDevice(arg_fp8_lut, arg_Q, "fp8_lut", "Q");
  CheckSameCudaDevice(arg_page_table, arg_Q, "page_table", "Q");
  CheckSameCudaDevice(arg_sparse_indices, arg_Q, "sparse_indices", "Q");
  CheckSameCudaDevice(arg_row_batches, arg_Q, "row_batches", "Q");
  CheckSameCudaDevice(arg_row_seq_lens, arg_Q, "row_seq_lens", "Q");
  CheckSameCudaDevice(arg_O, arg_Q, "O", "Q");
  CheckSameCudaDevice(arg_LSE, arg_Q, "LSE", "Q");
  CheckSameCudaDevice(arg_sinks, arg_Q, "sinks", "Q");
  CheckCurrentCudaDevice(arg_Q, "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";


  void* p_Q = arg_Q.data_ptr();
  void* p_KV = arg_KV.data_ptr();
  void* p_fp8_lut = arg_fp8_lut.data_ptr();
  void* p_page_table = arg_page_table.data_ptr();
  void* p_sparse_indices = arg_sparse_indices.data_ptr();
  void* p_row_batches = arg_row_batches.data_ptr();
  void* p_row_seq_lens = arg_row_seq_lens.data_ptr();
  void* p_O = arg_O.data_ptr();
  void* p_LSE = arg_LSE.data_ptr();
  void* p_sinks = arg_sinks.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_qk_dim = (int32_t)arg_qk_dim;
  int32_t v_value_dim = (int32_t)arg_value_dim;
  int32_t v_kv_stride = (int32_t)arg_kv_stride;
  int32_t v_page_size = (int32_t)arg_page_size;
  int32_t v_page_table_width = (int32_t)arg_page_table_width;
  int32_t v_sparse_width = (int32_t)arg_sparse_width;
  int32_t v_use_sparse = (int32_t)arg_use_sparse;
  float v_softmax_scale = (float)arg_softmax_scale;
  float v_bmm2_scale = (float)arg_bmm2_scale;
  int32_t v_enable_sink = (int32_t)arg_enable_sink;
  int32_t v_write_lse = (int32_t)arg_write_lse;
  void* kargs[] = {&p_Q, &p_KV, &p_fp8_lut, &p_page_table, &p_sparse_indices, &p_row_batches, &p_row_seq_lens, &p_O, &p_LSE, &p_sinks, &v_num_heads, &v_qk_dim, &v_value_dim, &v_kv_stride, &v_page_size, &v_page_table_width, &v_sparse_width, &v_use_sparse, &v_softmax_scale, &v_bmm2_scale, &v_enable_sink, &v_write_lse};

  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(full_abi_tail_bf16_83451b7317, "kernel_full_abi_tail_bf16");
  tvm::ffi::dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  tvm::ffi::dim3 block(32u, 1u, 1u);

  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(kargs, grid, block, stream, 16384u));
}
}  // namespace variant_full_abi_tail_bf16_83451b7317_d1d6ab272ce6

int SelectVariant(int32_t num_heads, int32_t qk_dim, int32_t value_dim, int32_t kv_stride, int32_t page_size, int32_t page_table_width, int32_t sparse_width, int32_t use_sparse, float softmax_scale, float bmm2_scale, int32_t enable_sink, int32_t write_lse, int32_t num_rows) {
  if ((num_rows > 0)) return 0;
  return -1;
}

void Dispatch(tvm::ffi::TensorView arg_query, tvm::ffi::TensorView arg_kv, tvm::ffi::TensorView arg_fp8_lut, tvm::ffi::TensorView arg_page_table, tvm::ffi::TensorView arg_sparse_indices, tvm::ffi::TensorView arg_row_batches, tvm::ffi::TensorView arg_row_seq_lens, tvm::ffi::TensorView arg_output, tvm::ffi::TensorView arg_lse, tvm::ffi::TensorView arg_sinks, int32_t num_heads, int32_t qk_dim, int32_t value_dim, int32_t kv_stride, int32_t page_size, int32_t page_table_width, int32_t sparse_width, int32_t use_sparse, float softmax_scale, float bmm2_scale, int32_t enable_sink, int32_t write_lse, int32_t num_rows, int64_t cuda_stream_ptr) {
  CheckCudaTensor(arg_query, "query");
  CheckCudaTensor(arg_kv, "kv");
  CheckCudaTensor(arg_fp8_lut, "fp8_lut");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckCudaTensor(arg_sparse_indices, "sparse_indices");
  CheckCudaTensor(arg_row_batches, "row_batches");
  CheckCudaTensor(arg_row_seq_lens, "row_seq_lens");
  CheckCudaTensor(arg_output, "output");
  CheckCudaTensor(arg_lse, "lse");
  CheckCudaTensor(arg_sinks, "sinks");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream_ptr));
  int v = SelectVariant(num_heads, qk_dim, value_dim, kv_stride, page_size, page_table_width, sparse_width, use_sparse, softmax_scale, bmm2_scale, enable_sink, write_lse, num_rows);
  switch (v) {
    case 0: {
      {
        int64_t gx = num_rows;
        int64_t gy = num_heads;
        int64_t gz = 1;
        mla_host_shim::variant_full_abi_tail_bf16_83451b7317_d1d6ab272ce6::Run(arg_query, arg_kv, arg_fp8_lut, arg_page_table, arg_sparse_indices, arg_row_batches, arg_row_seq_lens, arg_output, arg_lse, arg_sinks, num_heads, qk_dim, value_dim, kv_stride, page_size, page_table_width, sparse_width, use_sparse, softmax_scale, bmm2_scale, enable_sink, write_lse, gx, gy, gz, stream);
      }
      return;
    }
    default:
      TVM_FFI_CHECK(false, ValueError) << "no dispatch route for mla_bf16_tail shape (num_heads=" << num_heads << ", qk_dim=" << qk_dim << ", value_dim=" << value_dim << ", kv_stride=" << kv_stride << ", page_size=" << page_size << ", page_table_width=" << page_table_width << ", sparse_width=" << sparse_width << ", use_sparse=" << use_sparse << ", softmax_scale=" << softmax_scale << ", bmm2_scale=" << bmm2_scale << ", enable_sink=" << enable_sink << ", write_lse=" << write_lse << ", num_rows=" << num_rows << ")";
  }
}

int64_t RouteOf(int32_t num_heads, int32_t qk_dim, int32_t value_dim, int32_t kv_stride, int32_t page_size, int32_t page_table_width, int32_t sparse_width, int32_t use_sparse, float softmax_scale, float bmm2_scale, int32_t enable_sink, int32_t write_lse, int32_t num_rows) { return SelectVariant(num_heads, qk_dim, value_dim, kv_stride, page_size, page_table_width, sparse_width, use_sparse, softmax_scale, bmm2_scale, enable_sink, write_lse, num_rows); }

}  // namespace mla_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mla_host_shim::Dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_of, mla_host_shim::RouteOf);
