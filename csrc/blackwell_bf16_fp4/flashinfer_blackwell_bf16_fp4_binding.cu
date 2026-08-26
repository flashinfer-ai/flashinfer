/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

TVM_FFI_EMBED_CUBIN(FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT);

namespace flashinfer::blackwell_bf16_fp4 {

using tvm::ffi::TensorView;

constexpr int32_t kTileM = 16;
constexpr int32_t kTileN = 64;
constexpr int32_t kNativeTmaK = 256;
constexpr int32_t kSplitReduceThreads = 128;

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap ABI size must remain 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap ABI alignment must remain 128 bytes");

struct Problem {
  int32_t m;
  int32_t n;
  int32_t k;
  bool tiled;
  bool output_bf16;
  bool has_alpha;
};

inline void CheckCudaResult(CUresult result, const char* operation) {
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << operation << " failed: CUresult=" << static_cast<int>(result);
}

inline void CheckCudaTensor(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor";
}

inline void CheckSameDevice(const TensorView& tensor, const TensorView& reference,
                            const char* name) {
  TVM_FFI_CHECK(tensor.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on cuda:" << reference.device().device_id << ", got cuda:"
      << tensor.device().device_id;
}

inline void CheckDtype(const TensorView& tensor, const char* name, int code, int bits) {
  const DLDataType dtype = tensor.dtype();
  TVM_FFI_CHECK(static_cast<int>(dtype.code) == code && static_cast<int>(dtype.bits) == bits &&
                    static_cast<int>(dtype.lanes) == 1,
                TypeError)
      << name << " has the wrong dtype: got DLDataType(code=" << static_cast<int>(dtype.code)
      << ", bits=" << static_cast<int>(dtype.bits)
      << ", lanes=" << static_cast<int>(dtype.lanes) << ")";
}

inline void CheckMatrix(const TensorView& tensor, const char* name) {
  CheckCudaTensor(tensor, name);
  TVM_FFI_CHECK(tensor.ndim() == 2, ValueError)
      << name << " must be a rank-2 tensor, got rank " << tensor.ndim();
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckTarget(int32_t device_id) {
#if !defined(FLASHINFER_BLACKWELL_BF16_FP4_TARGET_MINOR)
#error "the exact Blackwell BF16 x FP4 target minor must be defined"
#endif
  int major = 0;
  int minor = 0;
  cudaError_t status =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(major) failed: " << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(minor) failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(major == 10 && minor == FLASHINFER_BLACKWELL_BF16_FP4_TARGET_MINOR,
                RuntimeError)
      << "this module requires compute capability 10."
      << FLASHINFER_BLACKWELL_BF16_FP4_TARGET_MINOR << ", got " << major << "." << minor;
}

inline void CheckCurrentDevice(const TensorView& reference) {
  int current_device = -1;
  const cudaError_t status = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match a: current=cuda:" << current_device
      << ", a=cuda:" << reference.device().device_id;
}

inline Problem CheckInputs(const TensorView& a, const TensorView& b,
                           const TensorView& b_descale, const TensorView& alpha,
                           const TensorView& out, int64_t layout_code) {
  CheckMatrix(a, "a");
  CheckMatrix(b, "b");
  CheckMatrix(b_descale, "b_descale");
  CheckMatrix(out, "out");
  CheckCudaTensor(alpha, "alpha");
  TVM_FFI_CHECK(alpha.IsContiguous(), ValueError) << "alpha must be contiguous";
  TVM_FFI_CHECK(alpha.ndim() == 1 && alpha.numel() == 1, ValueError)
      << "alpha must be a rank-1 float32 tensor with one element";

  CheckSameDevice(b, a, "b");
  CheckSameDevice(b_descale, a, "b_descale");
  CheckSameDevice(alpha, a, "alpha");
  CheckSameDevice(out, a, "out");
  CheckCurrentDevice(a);
  CheckTarget(a.device().device_id);

  CheckDtype(a, "a", kDLBfloat, 16);
  CheckDtype(alpha, "alpha", kDLFloat, 32);
  TVM_FFI_CHECK(layout_code == 0 || layout_code == 1, ValueError)
      << "layout_code must be 0 (native) or 1 (tiled), got " << layout_code;

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  TVM_FFI_CHECK(m64 > 0 && k64 > 0 && k64 % 16 == 0, ValueError)
      << "a must have positive shape (M, K) with K divisible by 16, got (" << m64 << ", "
      << k64 << ")";

  const bool tiled = layout_code == 1;
  int64_t n64 = 0;
  CheckDtype(b_descale, "b_descale", kDLUInt, 8);
  if (tiled) {
    CheckDtype(b, "b", kDLInt, 32);
    TVM_FFI_CHECK(b.size(0) == k64 / 16 && b.size(1) % 2 == 0, ValueError)
        << "tiled b must have shape (K / 16, N * 2)";
    n64 = b.size(1) / 2;
    TVM_FFI_CHECK(n64 > 0 && n64 % 64 == 0, ValueError)
        << "tiled N must be a positive multiple of 64, got " << n64;
    TVM_FFI_CHECK(b_descale.size(0) == k64 / 16 && b_descale.size(1) == n64, ValueError)
        << "tiled b_descale must have shape (K / 16, N)";
    CheckDtype(out, "out", kDLBfloat, 16);
  } else {
    CheckDtype(b, "b", kDLUInt, 8);
    n64 = b.size(0);
    TVM_FFI_CHECK(n64 > 0 && b.size(1) == k64 / 2, ValueError)
        << "native b must have shape (N, K / 2)";
    TVM_FFI_CHECK(b_descale.size(0) == n64 && b_descale.size(1) == k64 / 16, ValueError)
        << "native b_descale must have shape (N, K / 16)";
    const DLDataType output_dtype = out.dtype();
    const bool output_bf16 = output_dtype.code == kDLBfloat && output_dtype.bits == 16 &&
                             output_dtype.lanes == 1;
    const bool output_f16 = output_dtype.code == kDLFloat && output_dtype.bits == 16 &&
                            output_dtype.lanes == 1;
    TVM_FFI_CHECK(output_bf16 || output_f16, TypeError)
        << "native out must be bfloat16 or float16";
  }
  TVM_FFI_CHECK(out.size(0) == m64 && out.size(1) == n64, ValueError)
      << "out must have shape (M, N)";
  TVM_FFI_CHECK(m64 <= INT_MAX && n64 <= INT_MAX && k64 <= INT_MAX, ValueError)
      << "M, N, and K must fit the kernel's int32 parameters";

  const bool output_bf16 = out.dtype().code == kDLBfloat;
  // The fixed seven-argument FFI uses pointer identity as its has-alpha tag.
  // Python represents alpha=None with a float32 one-element view beginning at
  // a.data_ptr(); every explicit alpha is an independent contiguous tensor.
  const bool has_alpha = alpha.data_ptr() != a.data_ptr();
  return Problem{static_cast<int32_t>(m64), static_cast<int32_t>(n64),
                 static_cast<int32_t>(k64), tiled, output_bf16, has_alpha};
}

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  size_t used = 0;
};

// The generated bundle uses the pointer TMA ABI. Descriptor slots are
// immutable for process lifetime, so concurrent streams never observe a
// descriptor being rewritten while a prior launch is still in flight.
inline void* TmaDeviceSlot(const CUtensorMap& tensor_map, int32_t device_id,
                           cudaStream_t stream) {
  static std::mutex mutex;
  static auto* slots = new std::unordered_map<std::string, void*>();
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();

  CUcontext context = nullptr;
  CheckCudaResult(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  TVM_FFI_CHECK(context != nullptr, RuntimeError)
      << "pointer TMA ABI requires an active CUDA context";
  CUdevice current_device = -1;
  CheckCudaResult(cuCtxGetDevice(&current_device), "cuCtxGetDevice");
  TVM_FFI_CHECK(current_device == device_id, RuntimeError)
      << "TMA descriptor device mismatch";

  std::string key = std::to_string(reinterpret_cast<uintptr_t>(context));
  key.push_back(':');
  key.append(reinterpret_cast<const char*>(&tensor_map), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = slots->find(key);
  if (found != slots->end()) {
    return found->second;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CheckCudaResult(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status),
                  "cuStreamIsCapturing");
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "a new TMA descriptor cannot be created during CUDA Graph capture; prewarm this "
         "tensor/layout binding";

  TmaDeviceArena& arena = (*arenas)[context];
  TVM_FFI_CHECK(arena.used < TmaDeviceArena::kMaxSlots, RuntimeError)
      << "the immutable TMA descriptor arena is exhausted";
  if (arena.used % TmaDeviceArena::kSlotsPerChunk == 0) {
    CUdeviceptr chunk = 0;
    CheckCudaResult(cuMemAlloc(&chunk, TmaDeviceArena::kSlotsPerChunk * sizeof(CUtensorMap)),
                    "cuMemAlloc(TMA descriptor arena)");
    arena.chunks.push_back(chunk);
  }
  const size_t chunk_index = arena.used / TmaDeviceArena::kSlotsPerChunk;
  const size_t slot_index = arena.used % TmaDeviceArena::kSlotsPerChunk;
  const CUdeviceptr slot =
      arena.chunks[chunk_index] + slot_index * sizeof(CUtensorMap);
  CheckCudaResult(cuMemcpyHtoD(slot, &tensor_map, sizeof(CUtensorMap)),
                  "cuMemcpyHtoD(TMA descriptor)");
  ++arena.used;
  void* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(slot));
  slots->emplace(std::move(key), pointer);
  return pointer;
}

inline CUtensorMap EncodeTma2D(const TensorView& tensor, CUtensorMapDataType data_type,
                               uint32_t element_bytes, uint32_t box_x, uint32_t box_y,
                               CUtensorMapSwizzle swizzle, bool allow_oob_x,
                               bool allow_oob_y, const char* name) {
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.stride(1) == 1, ValueError)
      << name << " must be a contiguous rank-2 TMA source";
  const uint64_t global_dim[2] = {static_cast<uint64_t>(tensor.size(1)),
                                  static_cast<uint64_t>(tensor.size(0))};
  const uint64_t global_strides[1] = {
      static_cast<uint64_t>(tensor.stride(0)) * element_bytes};
  TVM_FFI_CHECK((allow_oob_x || box_x <= global_dim[0]) &&
                    (allow_oob_y || box_y <= global_dim[1]),
                ValueError)
      << "TMA box exceeds " << name << " without an out-of-bounds-enabled axis";
  const uint32_t box_dim[2] = {box_x, box_y};
  const uint32_t element_strides[2] = {1u, 1u};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, data_type, 2, tensor.data_ptr(), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled failed for " << name
      << ": CUresult=" << static_cast<int>(result);
  return tensor_map;
}

inline CUtensorMap EncodeWarpA3D(const TensorView& a, uint32_t tile_m) {
  TVM_FFI_CHECK(a.size(1) % 64 == 0, ValueError)
      << "warp A descriptor requires K divisible by 64";
  const uint64_t global_dim[3] = {64u, static_cast<uint64_t>(a.size(0)),
                                  static_cast<uint64_t>(a.size(1) / 64)};
  const uint64_t global_strides[2] = {
      static_cast<uint64_t>(a.stride(0)) * 2u, 64u * 2u};
  const uint32_t box_dim[3] = {64u, tile_m, 2u};
  const uint32_t element_strides[3] = {1u, 1u, 1u};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, a.data_ptr(), global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled failed for warp A: CUresult="
      << static_cast<int>(result);
  return tensor_map;
}

enum class Component : uint8_t {
  kNativeTmaBf16,
  kNativeTmaF16,
  kNativeCpAsyncBf16,
  kNativeCpAsyncF16,
  kTiledBaseBf16,
  kNativeGroupM128Bf16,
  kTiledWarpM16Bf16,
  kTiledWarpM32Bf16,
  kTiledWarpM64Bf16,
  kTiledWarpM16K16Bf16,
  kTiledWarpM16K32Bf16,
  kTiledWarpM16K48Bf16,
  kNativeSplitK2PartialF32,
  kNativeSplitK2ReduceBf16,
};

struct KernelSpec {
  Component component;
  bool has_alpha;
  bool enable_pdl;
  bool flat_grid;
  const char* symbol;
  uint32_t threads;
  uint32_t smem_bytes;
};

#define FLASHINFER_BF16_FP4_FOUR(COMPONENT, PREFIX, THREADS, SMEM, FLAT)                  \
  KernelSpec{COMPONENT, false, false, FLAT, PREFIX "_a0_pdl0", THREADS, SMEM},          \
      KernelSpec{COMPONENT, false, true, FLAT, PREFIX "_a0_pdl1", THREADS, SMEM},       \
      KernelSpec{COMPONENT, true, false, FLAT, PREFIX "_a1_pdl0", THREADS, SMEM},       \
      KernelSpec{COMPONENT, true, true, FLAT, PREFIX "_a1_pdl1", THREADS, SMEM}

constexpr std::array<KernelSpec, 74> kKernelSpecs = {{
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeTmaBf16,
        "kernel_flashinfer_bf16_fp4_cudnn_tma_bf16", 512u, 107520u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeTmaF16,
        "kernel_flashinfer_bf16_fp4_cudnn_tma_f16", 512u, 107520u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeCpAsyncBf16,
        "kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16", 512u, 107520u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeCpAsyncF16,
        "kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16", 512u, 107520u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledBaseBf16,
        "kernel_flashinfer_bf16_fp4_cute_bf16", 512u, 107520u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeTmaBf16,
        "kernel_flashinfer_bf16_fp4_cudnn_tma_bf16_flat", 512u, 107520u, true),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeTmaF16,
        "kernel_flashinfer_bf16_fp4_cudnn_tma_f16_flat", 512u, 107520u, true),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeCpAsyncBf16,
        "kernel_flashinfer_bf16_fp4_cudnn_cp_async_bf16_flat", 512u, 107520u, true),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeCpAsyncF16,
        "kernel_flashinfer_bf16_fp4_cudnn_cp_async_f16_flat", 512u, 107520u, true),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledBaseBf16,
        "kernel_flashinfer_bf16_fp4_cute_bf16_flat", 512u, 107520u, true),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeGroupM128Bf16,
        "kernel_flashinfer_bf16_fp4_cudnn_group_m128_bf16", 512u, 139264u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM16Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m16_bf16", 96u, 150528u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM32Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m32_bf16", 160u, 218112u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM64Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m64_bf16", 160u, 73728u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM16K16Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m16_k16_bf16", 96u, 150528u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM16K32Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m16_k32_bf16", 96u, 150528u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kTiledWarpM16K48Bf16,
        "kernel_flashinfer_bf16_fp4_cute_warp_mma_m16_k48_bf16", 96u, 150528u, false),
    FLASHINFER_BF16_FP4_FOUR(
        Component::kNativeSplitK2PartialF32,
        "kernel_flashinfer_bf16_fp4_cudnn_split_k2_partial_f32", 512u, 107520u, false),
    KernelSpec{Component::kNativeSplitK2ReduceBf16, false, false, false,
               "kernel_flashinfer_bf16_fp4_cudnn_split_k2_reduce_bf16_pdl0", 128u, 0u},
    KernelSpec{Component::kNativeSplitK2ReduceBf16, false, true, false,
               "kernel_flashinfer_bf16_fp4_cudnn_split_k2_reduce_bf16_pdl1", 128u, 0u},
}};

#undef FLASHINFER_BF16_FP4_FOUR

static_assert(kKernelSpecs.size() == 74, "the standalone bundle must expose 74 kernels");

inline const KernelSpec& FindKernelSpec(Component component, bool has_alpha, bool enable_pdl,
                                        bool flat_grid = false) {
  for (const KernelSpec& spec : kKernelSpecs) {
    const bool alpha_match = component == Component::kNativeSplitK2ReduceBf16 ||
                             spec.has_alpha == has_alpha;
    if (spec.component == component && alpha_match && spec.enable_pdl == enable_pdl &&
        spec.flat_grid == flat_grid) {
      return spec;
    }
  }
  TVM_FFI_THROW(RuntimeError) << "no physical kernel matches the selected GEMM route";
}

inline tvm::ffi::CubinKernel& GetKernel(const KernelSpec& spec) {
  static std::mutex mutex;
  static auto* kernels =
      new std::unordered_map<std::string, std::unique_ptr<tvm::ffi::CubinKernel>>();
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = kernels->find(spec.symbol);
  if (found != kernels->end()) {
    return *found->second;
  }
  auto kernel = std::make_unique<tvm::ffi::CubinKernel>(
      EmbedCubinModule_FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT::Global()
          ->mod.GetKernelWithMaxDynamicSharedMemory(spec.symbol, spec.smem_bytes));
  tvm::ffi::CubinKernel& reference = *kernel;
  kernels->emplace(spec.symbol, std::move(kernel));
  return reference;
}

inline void LaunchKernel(const KernelSpec& spec, void** args, uint32_t grid_x,
                         uint32_t grid_y, uint32_t grid_z, cudaStream_t stream) {
  tvm::ffi::CubinKernel& kernel = GetKernel(spec);
  tvm::ffi::cuda_api::LaunchConfig config{};
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attribute{};
  config.gridDimX = grid_x;
  config.gridDimY = grid_y;
  config.gridDimZ = grid_z;
  config.blockDimX = spec.threads;
  config.blockDimY = 1u;
  config.blockDimZ = 1u;
  config.sharedMemBytes = spec.smem_bytes;
  config.hStream = reinterpret_cast<CUstream>(stream);
  config.attrs = nullptr;
  config.numAttrs = 0;
  if (spec.enable_pdl) {
    attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attribute.value.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
#else
  cudaLaunchAttribute attribute{};
  config.gridDim = {grid_x, grid_y, grid_z};
  config.blockDim = {spec.threads, 1u, 1u};
  config.dynamicSmemBytes = spec.smem_bytes;
  config.stream = stream;
  config.attrs = nullptr;
  config.numAttrs = 0;
  if (spec.enable_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
#endif
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.LaunchEx(args, config));
}

inline uint32_t CheckedGrid(int64_t value, const char* name) {
  TVM_FFI_CHECK(value > 0 &&
                    static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max(),
                ValueError)
      << name << " does not fit a CUDA grid dimension: " << value;
  return static_cast<uint32_t>(value);
}

inline int64_t CeilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

inline int32_t MultiProcessorCount(int32_t device_id) {
  int count = 0;
  const cudaError_t status =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id);
  TVM_FFI_CHECK(status == cudaSuccess && count > 0, RuntimeError)
      << "cudaDeviceGetAttribute(multiProcessorCount) failed: "
      << cudaGetErrorString(status);
  return count;
}

inline void LaunchBase(const Problem& problem, const TensorView& a, const TensorView& b,
                       const TensorView& b_descale, const TensorView& alpha,
                       const TensorView& out, bool enable_pdl, cudaStream_t stream) {
  const bool cp_async = !problem.tiled && problem.k % kNativeTmaK != 0;
  const int64_t grid_m = CeilDiv(problem.m, kTileM);
  const int64_t grid_n = CeilDiv(problem.n, kTileN);
  const bool flat_grid = grid_m > 65535;

  CUtensorMap a_map =
      EncodeTma2D(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, 64u, 16u,
                  CU_TENSOR_MAP_SWIZZLE_128B, true, true, "a");
  void* p_a = TmaDeviceSlot(a_map, a.device().device_id, stream);
  void* p_b = b.data_ptr();
  void* p_b_descale = b_descale.data_ptr();
  CUtensorMap b_map{};
  CUtensorMap b_descale_map{};
  if (!cp_async) {
    if (problem.tiled) {
      b_map = EncodeTma2D(b, CU_TENSOR_MAP_DATA_TYPE_INT32, 4u, 128u, 4u,
                         CU_TENSOR_MAP_SWIZZLE_NONE, false, true, "b");
      b_descale_map =
          EncodeTma2D(b_descale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 64u, 4u,
                      CU_TENSOR_MAP_SWIZZLE_NONE, false, true, "b_descale");
    } else {
      b_map = EncodeTma2D(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 32u, 64u,
                         CU_TENSOR_MAP_SWIZZLE_NONE, true, true, "b");
      b_descale_map =
          EncodeTma2D(b_descale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 16u, 64u,
                      CU_TENSOR_MAP_SWIZZLE_NONE, true, true, "b_descale");
    }
    p_b = TmaDeviceSlot(b_map, b.device().device_id, stream);
    p_b_descale =
        TmaDeviceSlot(b_descale_map, b_descale.device().device_id, stream);
  }

  void* p_alpha = alpha.data_ptr();
  void* p_out = out.data_ptr();
  int32_t m = problem.m;
  int32_t n = problem.n;
  int32_t k = problem.k;
  void* args[] = {&p_a, &p_b, &p_b_descale, &p_alpha, &p_out, &m, &n, &k};

  Component component;
  if (problem.tiled) {
    component = Component::kTiledBaseBf16;
  } else if (cp_async) {
    component = problem.output_bf16 ? Component::kNativeCpAsyncBf16
                                    : Component::kNativeCpAsyncF16;
  } else {
    component = problem.output_bf16 ? Component::kNativeTmaBf16
                                    : Component::kNativeTmaF16;
  }
  const KernelSpec& spec =
      FindKernelSpec(component, problem.has_alpha, enable_pdl, flat_grid);
  if (flat_grid) {
    LaunchKernel(spec, args, CheckedGrid(grid_m * grid_n, "flat grid.x"), 1u, 1u,
                 stream);
  } else {
    LaunchKernel(spec, args, CheckedGrid(grid_n, "grid.x"), CheckedGrid(grid_m, "grid.y"),
                 1u, stream);
  }
}

inline void LaunchGroupM128(const Problem& problem, const TensorView& a,
                            const TensorView& b, const TensorView& b_descale,
                            const TensorView& alpha, const TensorView& out,
                            bool enable_pdl, cudaStream_t stream) {
  CUtensorMap a_map =
      EncodeTma2D(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, 64u, 128u,
                  CU_TENSOR_MAP_SWIZZLE_128B, false, false, "a");
  CUtensorMap b_map =
      EncodeTma2D(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 32u, 64u,
                  CU_TENSOR_MAP_SWIZZLE_NONE, false, false, "b");
  CUtensorMap b_descale_map =
      EncodeTma2D(b_descale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 16u, 64u,
                  CU_TENSOR_MAP_SWIZZLE_NONE, false, false, "b_descale");
  void* p_a = TmaDeviceSlot(a_map, a.device().device_id, stream);
  void* p_b = TmaDeviceSlot(b_map, b.device().device_id, stream);
  void* p_b_descale =
      TmaDeviceSlot(b_descale_map, b_descale.device().device_id, stream);
  void* p_alpha = alpha.data_ptr();
  void* p_out = out.data_ptr();
  int32_t m = problem.m;
  int32_t n = problem.n;
  int32_t k = problem.k;
  void* args[] = {&p_a, &p_b, &p_b_descale, &p_alpha, &p_out, &m, &n, &k};
  const KernelSpec& spec = FindKernelSpec(Component::kNativeGroupM128Bf16,
                                          problem.has_alpha, enable_pdl);
  LaunchKernel(spec, args, CheckedGrid(CeilDiv(problem.n, 64), "group grid.x"),
               CheckedGrid(problem.m / 128, "group grid.y"), 1u, stream);
}

inline void LaunchWarp(const Problem& problem, const TensorView& a, const TensorView& b,
                       const TensorView& b_descale, const TensorView& alpha,
                       const TensorView& out, Component component, uint32_t tile_m,
                       uint32_t tile_k, bool enable_pdl, cudaStream_t stream) {
  const bool short_k = tile_k != 128u;
  CUtensorMap a_map =
      short_k ? EncodeTma2D(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, tile_k, 16u,
                            CU_TENSOR_MAP_SWIZZLE_NONE, false, true, "a")
              : EncodeWarpA3D(a, tile_m);
  CUtensorMap b_map =
      EncodeTma2D(b, CU_TENSOR_MAP_DATA_TYPE_INT32, 4u, 128u, tile_k / 16u,
                  CU_TENSOR_MAP_SWIZZLE_NONE, false, !short_k, "b");
  CUtensorMap out_map =
      EncodeTma2D(out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, 64u, tile_m,
                  CU_TENSOR_MAP_SWIZZLE_128B, true, true, "out");
  void* p_a = TmaDeviceSlot(a_map, a.device().device_id, stream);
  void* p_b = TmaDeviceSlot(b_map, b.device().device_id, stream);
  void* p_b_descale = b_descale.data_ptr();
  CUtensorMap b_descale_map{};
  if (!short_k) {
    b_descale_map =
        EncodeTma2D(b_descale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 64u, 8u,
                    CU_TENSOR_MAP_SWIZZLE_NONE, false, true, "b_descale");
    p_b_descale =
        TmaDeviceSlot(b_descale_map, b_descale.device().device_id, stream);
  }
  void* p_alpha = alpha.data_ptr();
  void* p_out = TmaDeviceSlot(out_map, out.device().device_id, stream);
  int32_t m = problem.m;
  int32_t n = problem.n;
  int32_t k = problem.k;
  void* args[] = {&p_a, &p_b, &p_b_descale, &p_alpha, &p_out, &m, &n, &k};

  const int64_t total_tiles = CeilDiv(problem.m, tile_m) * CeilDiv(problem.n, 64);
  const int64_t persistent_grid =
      std::min<int64_t>(total_tiles, MultiProcessorCount(a.device().device_id));
  const KernelSpec& spec =
      FindKernelSpec(component, problem.has_alpha, enable_pdl);
  LaunchKernel(spec, args, CheckedGrid(persistent_grid, "persistent grid.x"), 1u, 1u,
               stream);
}

struct WorkspaceKey {
  CUcontext context;
  uintptr_t stream;
  int32_t m;
  int32_t n;

  bool operator==(const WorkspaceKey& other) const {
    return context == other.context && stream == other.stream && m == other.m && n == other.n;
  }
};

struct WorkspaceKeyHash {
  size_t operator()(const WorkspaceKey& key) const {
    size_t value = std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.context));
    value ^= std::hash<uintptr_t>{}(key.stream) + 0x9e3779b9u + (value << 6) + (value >> 2);
    value ^= std::hash<int32_t>{}(key.m) + 0x9e3779b9u + (value << 6) + (value >> 2);
    value ^= std::hash<int32_t>{}(key.n) + 0x9e3779b9u + (value << 6) + (value >> 2);
    return value;
  }
};

inline void* SplitWorkspace(const Problem& problem, cudaStream_t stream) {
  static std::mutex mutex;
  static auto* workspaces =
      new std::unordered_map<WorkspaceKey, CUdeviceptr, WorkspaceKeyHash>();
  CUcontext context = nullptr;
  CheckCudaResult(cuCtxGetCurrent(&context), "cuCtxGetCurrent(split workspace)");
  TVM_FFI_CHECK(context != nullptr, RuntimeError)
      << "split workspace requires an active CUDA context";
  const WorkspaceKey key{context, reinterpret_cast<uintptr_t>(stream), problem.m, problem.n};
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = workspaces->find(key);
  if (found != workspaces->end()) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(found->second));
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CheckCudaResult(cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status),
                  "cuStreamIsCapturing(split workspace)");
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "the split-K workspace must be warmed before CUDA Graph capture";
  const uint64_t elements = static_cast<uint64_t>(problem.m) * problem.n;
  TVM_FFI_CHECK(elements <= std::numeric_limits<size_t>::max() / (2u * sizeof(float)),
                ValueError)
      << "split-K workspace size overflows size_t";
  CUdeviceptr allocation = 0;
  CheckCudaResult(cuMemAlloc(&allocation, elements * 2u * sizeof(float)),
                  "cuMemAlloc(split workspace)");
  workspaces->emplace(key, allocation);
  return reinterpret_cast<void*>(static_cast<uintptr_t>(allocation));
}

inline void LaunchSplitK2(const Problem& problem, const TensorView& a,
                          const TensorView& b, const TensorView& b_descale,
                          const TensorView& alpha, const TensorView& out,
                          bool enable_pdl, cudaStream_t stream) {
  CUtensorMap a_map =
      EncodeTma2D(a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2u, 64u, 16u,
                  CU_TENSOR_MAP_SWIZZLE_128B, true, true, "a");
  CUtensorMap b_map =
      EncodeTma2D(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 32u, 64u,
                  CU_TENSOR_MAP_SWIZZLE_NONE, false, false, "b");
  CUtensorMap b_descale_map =
      EncodeTma2D(b_descale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1u, 16u, 64u,
                  CU_TENSOR_MAP_SWIZZLE_NONE, false, false, "b_descale");
  void* p_a = TmaDeviceSlot(a_map, a.device().device_id, stream);
  void* p_b = TmaDeviceSlot(b_map, b.device().device_id, stream);
  void* p_b_descale =
      TmaDeviceSlot(b_descale_map, b_descale.device().device_id, stream);
  void* p_alpha = alpha.data_ptr();
  void* p_partials = SplitWorkspace(problem, stream);
  int32_t m = problem.m;
  int32_t n = problem.n;
  int32_t k = problem.k;
  void* partial_args[] = {
      &p_a, &p_b, &p_b_descale, &p_alpha, &p_partials, &m, &n, &k};
  const KernelSpec& partial_spec = FindKernelSpec(
      Component::kNativeSplitK2PartialF32, problem.has_alpha, enable_pdl);
  LaunchKernel(partial_spec, partial_args,
               CheckedGrid(CeilDiv(problem.n, 64), "split partial grid.x"), 1u, 2u,
               stream);

  void* p_out = out.data_ptr();
  int32_t elements = problem.m * problem.n;
  void* reduce_args[] = {&p_partials, &p_out, &elements};
  const KernelSpec& reduce_spec =
      FindKernelSpec(Component::kNativeSplitK2ReduceBf16, false, enable_pdl);
  LaunchKernel(reduce_spec, reduce_args,
               CheckedGrid(CeilDiv(elements, kSplitReduceThreads), "split reducer grid.x"),
               1u, 1u, stream);
}

void Run(TensorView a, TensorView b, TensorView b_descale, TensorView alpha,
         TensorView out, int64_t layout_code, bool enable_pdl) {
  const Problem problem = CheckInputs(a, b, b_descale, alpha, out, layout_code);
  const DLDevice device = a.device();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  if (!problem.tiled && problem.output_bf16 && problem.k % kNativeTmaK == 0 &&
      problem.m == 768 && problem.n == 2112 && problem.k == 2048) {
    LaunchGroupM128(problem, a, b, b_descale, alpha, out, enable_pdl, stream);
    return;
  }
  if (!problem.tiled && problem.output_bf16 && problem.k % kNativeTmaK == 0 &&
      problem.m == 1 && problem.n == 4096 && problem.k == 4096) {
    LaunchSplitK2(problem, a, b, b_descale, alpha, out, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.k == 16) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM16K16Bf16, 16u, 16u, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.k == 32) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM16K32Bf16, 16u, 32u, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.k == 48) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM16K48Bf16, 16u, 48u, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.m <= 16 && problem.k >= 128 && problem.k % 64 == 0) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM16Bf16, 16u, 128u, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.m >= 17 && problem.m <= 32 && problem.k >= 128 &&
      problem.k % 64 == 0) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM32Bf16, 32u, 128u, enable_pdl, stream);
    return;
  }
  if (problem.tiled && problem.m >= 33 && problem.k % 128 == 0) {
    LaunchWarp(problem, a, b, b_descale, alpha, out,
               Component::kTiledWarpM64Bf16, 64u, 128u, enable_pdl, stream);
    return;
  }
  LaunchBase(problem, a, b, b_descale, alpha, out, enable_pdl, stream);
}

}  // namespace flashinfer::blackwell_bf16_fp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_bf16_fp4::Run);
