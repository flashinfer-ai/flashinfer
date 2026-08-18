#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

#define uint8_t cake_dsv4_generated_uint8_t
#define uint16_t cake_dsv4_generated_uint16_t
#define uint32_t cake_dsv4_generated_uint32_t
#define uint64_t cake_dsv4_generated_uint64_t
#define int32_t cake_dsv4_generated_int32_t
#define int16_t cake_dsv4_generated_int16_t
#define CakeTensorMap cake_dsv4_generated_CakeTensorMap
#define CUtensorMap cake_dsv4_generated_CUtensorMap
#include "cake_dsv4_fp8_lowhead_decode.cu"
#undef CUtensorMap
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer::cake_dsv4 {

namespace {




using tvm::ffi::TensorView;

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

// 4D TMA descriptor for buffer 'tmap_q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'tmap_q' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'tmap_q' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 128 == 0, ValueError)
      << "TMA source 'tmap_q' extent " << d1
      << " must divide exactly by " << 128;
  uint64_t global_dim[4] = {(uint64_t)(128), (uint64_t)(d2), (uint64_t)((d1 / 128)), (uint64_t)(d3)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'tmap_q' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 64, 1, 1) exceeds resolved global dims for 'tmap_q'";
  uint64_t global_strides[3] = {
      (uint64_t)((d1 * 8) / 8),
      (uint64_t)((128 * 8) / 8),
      (uint64_t)(((d2 * d1) * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'tmap_q') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'tmap_swa_kv' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_swa_kv(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_swa_kv' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_swa_kv' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'tmap_swa_kv' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "tmap_swa_kv");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'tmap_swa_kv' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'tmap_swa_kv' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1) exceeds resolved global dims for 'tmap_swa_kv'";
  uint64_t global_strides[1] = {
      (uint64_t)((s2 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'tmap_swa_kv') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'tmap_compressed_kv' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_tmap_compressed_kv(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'tmap_compressed_kv' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'tmap_compressed_kv' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'tmap_compressed_kv' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "tmap_compressed_kv");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'tmap_compressed_kv' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'tmap_compressed_kv' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1) exceeds resolved global dims for 'tmap_compressed_kv'";
  uint64_t global_strides[1] = {
      (uint64_t)((s2 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'tmap_compressed_kv') failed: CUresult=" << (int)r;
  return tm;
}

}  // namespace


void Run_fp8_lowhead_decode(TensorView arg_tmap_q, TensorView arg_tmap_swa_kv, TensorView arg_tmap_compressed_kv, TensorView arg_O, TensorView arg_partial_lse, TensorView arg_sparse_indices, TensorView arg_sparse_topk_lens, TensorView arg_sinks, TensorView arg_bmm1_scale, TensorView arg_bmm2_scale, int64_t arg_num_heads, int64_t arg_num_query_tokens, int64_t arg_sparse_topk, int64_t arg_has_sinks, int64_t arg_total_work_items, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError)
      << "cuda_stream must be non-negative";
  ffi::CUDADeviceGuard device_guard(arg_tmap_q.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      static_cast<uintptr_t>(cuda_stream));
  CheckCudaTensor(arg_tmap_q, "tmap_q");
  CheckDtype(arg_tmap_q, "tmap_q", 1, 8, 1);
  CheckContiguous(arg_tmap_q, "tmap_q");
  CheckCudaTensor(arg_tmap_swa_kv, "tmap_swa_kv");
  CheckDtype(arg_tmap_swa_kv, "tmap_swa_kv", 1, 8, 1);
  CheckCudaTensor(arg_tmap_compressed_kv, "tmap_compressed_kv");
  CheckDtype(arg_tmap_compressed_kv, "tmap_compressed_kv", 1, 8, 1);
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_partial_lse, "partial_lse");
  CheckDtype(arg_partial_lse, "partial_lse", 2, 32, 1);
  CheckContiguous(arg_partial_lse, "partial_lse");
  CheckCudaTensor(arg_sparse_indices, "sparse_indices");
  CheckDtype(arg_sparse_indices, "sparse_indices", 0, 32, 1);
  CheckContiguous(arg_sparse_indices, "sparse_indices");
  CheckCudaTensor(arg_sparse_topk_lens, "sparse_topk_lens");
  CheckDtype(arg_sparse_topk_lens, "sparse_topk_lens", 0, 32, 1);
  CheckContiguous(arg_sparse_topk_lens, "sparse_topk_lens");
  CheckCudaTensor(arg_sinks, "sinks");
  CheckDtype(arg_sinks, "sinks", 2, 32, 1);
  CheckContiguous(arg_sinks, "sinks");
  CheckCudaTensor(arg_bmm1_scale, "bmm1_scale");
  CheckDtype(arg_bmm1_scale, "bmm1_scale", 2, 32, 1);
  CheckContiguous(arg_bmm1_scale, "bmm1_scale");
  CheckCudaTensor(arg_bmm2_scale, "bmm2_scale");
  CheckDtype(arg_bmm2_scale, "bmm2_scale", 2, 32, 1);
  CheckContiguous(arg_bmm2_scale, "bmm2_scale");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_query_tokens >= -2147483648LL && arg_num_query_tokens <= 2147483647LL, ValueError)
      << "scalar 'num_query_tokens' value " << arg_num_query_tokens
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sparse_topk >= -2147483648LL && arg_sparse_topk <= 2147483647LL, ValueError)
      << "scalar 'sparse_topk' value " << arg_sparse_topk
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_has_sinks >= -2147483648LL && arg_has_sinks <= 2147483647LL, ValueError)
      << "scalar 'has_sinks' value " << arg_has_sinks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_work_items >= -2147483648LL && arg_total_work_items <= 2147483647LL, ValueError)
      << "scalar 'total_work_items' value " << arg_total_work_items
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_tmap_swa_kv, arg_tmap_q, "tmap_swa_kv", "tmap_q");
  CheckSameCudaDevice(arg_tmap_compressed_kv, arg_tmap_q, "tmap_compressed_kv", "tmap_q");
  CheckSameCudaDevice(arg_O, arg_tmap_q, "O", "tmap_q");
  CheckSameCudaDevice(arg_partial_lse, arg_tmap_q, "partial_lse", "tmap_q");
  CheckSameCudaDevice(arg_sparse_indices, arg_tmap_q, "sparse_indices", "tmap_q");
  CheckSameCudaDevice(arg_sparse_topk_lens, arg_tmap_q, "sparse_topk_lens", "tmap_q");
  CheckSameCudaDevice(arg_sinks, arg_tmap_q, "sinks", "tmap_q");
  CheckSameCudaDevice(arg_bmm1_scale, arg_tmap_q, "bmm1_scale", "tmap_q");
  CheckSameCudaDevice(arg_bmm2_scale, arg_tmap_q, "bmm2_scale", "tmap_q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  TVM_FFI_CHECK(grid_x % 2 == 0 && grid_y % 1 == 0 && grid_z % 1 == 0, ValueError)
      << "launch grid (" << grid_x << ", " << grid_y << ", " << grid_z
      << ") must be divisible by cluster dims (2, 1, 1)";


  CUtensorMap h_tmap_q = EncodeTma_tmap_q(arg_tmap_q);
  void* p_tmap_q = TmaDeviceSlot(h_tmap_q, arg_tmap_q.device().device_id, stream);
  CUtensorMap h_tmap_swa_kv = EncodeTma_tmap_swa_kv(arg_tmap_swa_kv);
  void* p_tmap_swa_kv = TmaDeviceSlot(h_tmap_swa_kv, arg_tmap_swa_kv.device().device_id, stream);
  CUtensorMap h_tmap_compressed_kv = EncodeTma_tmap_compressed_kv(arg_tmap_compressed_kv);
  void* p_tmap_compressed_kv = TmaDeviceSlot(h_tmap_compressed_kv, arg_tmap_compressed_kv.device().device_id, stream);
  void* p_O = arg_O.data_ptr();
  void* p_partial_lse = arg_partial_lse.data_ptr();
  void* p_sparse_indices = arg_sparse_indices.data_ptr();
  void* p_sparse_topk_lens = arg_sparse_topk_lens.data_ptr();
  void* p_sinks = arg_sinks.data_ptr();
  void* p_bmm1_scale = arg_bmm1_scale.data_ptr();
  void* p_bmm2_scale = arg_bmm2_scale.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_num_query_tokens = (int32_t)arg_num_query_tokens;
  int32_t v_sparse_topk = (int32_t)arg_sparse_topk;
  int32_t v_has_sinks = (int32_t)arg_has_sinks;
  int32_t v_total_work_items = (int32_t)arg_total_work_items;
  void* kargs[] = {&p_tmap_q, &p_tmap_swa_kv, &p_tmap_compressed_kv, &p_O, &p_partial_lse, &p_sparse_indices, &p_sparse_topk_lens, &p_sinks, &p_bmm1_scale, &p_bmm2_scale, &v_num_heads, &v_num_query_tokens, &v_sparse_topk, &v_has_sinks, &v_total_work_items};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);
  cudaError_t status = cudaSuccess;
  status = cudaFuncSetAttribute(kernel_cake_dsv4_fp8_lowhead_decode,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      173056);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_cake_dsv4_fp8_lowhead_decode) failed: "
      << cudaGetErrorString(status);
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 173056u;
  config.stream = stream;
  cudaLaunchAttribute attrs[1] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2u;
  attrs[0].val.clusterDim.y = 1u;
  attrs[0].val.clusterDim.z = 1u;
  config.attrs = attrs;
  config.numAttrs = 1;
  status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_dsv4_fp8_lowhead_decode), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_cake_dsv4_fp8_lowhead_decode launch failed: " << cudaGetErrorString(status);
}


}  // namespace flashinfer::cake_dsv4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    run_fp8_lowhead_decode, flashinfer::cake_dsv4::Run_fp8_lowhead_decode);
