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

// Generated for Cake FMHA — do not edit.
// tvm-ffi direct-source launcher for Cake FMHA kernel 'kernel_cake_fmha_dcp_spec_bf16_v4'.
#include <cuda.h>
#include <cuda_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" __global__ void kernel_cake_fmha_dcp_spec_bf16_v4();


namespace cake_fmha_host_shim {

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

struct TmaDeviceSlotState {
  CUdeviceptr pointer = 0;
  cudaEvent_t completion = nullptr;
  cudaStream_t last_stream = nullptr;
  std::string key;
  bool has_completion = false;
  bool reserved = false;
  bool pinned = false;
};

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxReusableSlots = 4096;
  static constexpr size_t kMaxPinnedSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  std::vector<TmaDeviceSlotState> slots;
  std::unordered_map<std::string, size_t> pinned_slots;
  size_t reusable_slots = 0;
  size_t pinned_count = 0;
  size_t cursor = 0;
};

struct TmaDeviceSlotLease {
  void* pointer;
  CUcontext context;
  size_t slot_index;
  bool track_completion;
};

static inline bool TmaDeviceSlotReady(const TmaDeviceSlotState& slot) {
  if (!slot.has_completion) return true;
  cudaError_t status = cudaEventQuery(slot.completion);
  if (status == cudaSuccess) return true;
  TVM_FFI_CHECK(status == cudaErrorNotReady, RuntimeError)
      << "failed to query pointer TMA descriptor completion: "
      << cudaGetErrorString(status);
  return false;
}

static inline void AddTmaDeviceSlotChunk(TmaDeviceArena& arena) {
  size_t count = std::min(
      TmaDeviceArena::kSlotsPerChunk,
      TmaDeviceArena::kMaxReusableSlots - arena.reusable_slots);
  TVM_FFI_CHECK(count > 0, RuntimeError)
      << "pointer TMA descriptor reusable arena is exhausted";
  CUdeviceptr chunk = 0;
  CUresult result = cuMemAlloc(&chunk, count * sizeof(CUtensorMap));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuMemAlloc for pointer TMA descriptor arena failed: CUresult="
      << static_cast<int>(result);
  arena.chunks.push_back(chunk);
  for (size_t index = 0; index < count; ++index) {
    cudaEvent_t completion = nullptr;
    cudaError_t status = cudaEventCreateWithFlags(
        &completion, cudaEventDisableTiming);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to create pointer TMA descriptor completion event: "
        << cudaGetErrorString(status);
    arena.slots.push_back({
        chunk + index * sizeof(CUtensorMap), completion, nullptr, "",
        false, false, false});
  }
  arena.reusable_slots += count;
}

static inline std::mutex& TmaDeviceSlotMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

static inline std::unordered_map<CUcontext, TmaDeviceArena>&
TmaDeviceArenas() {
  static auto* arenas =
      new std::unordered_map<CUcontext, TmaDeviceArena>();
  return *arenas;
}

// Eager descriptors live in a bounded, completion-tracked pool. An exact
// prewarmed descriptor is removed from that pool when capture first observes
// it, keeping its device address immutable for every replay of that graph.
static inline TmaDeviceSlotLease TmaDeviceSlot(
    const CUtensorMap& tm,
    int device_id,
    cudaStream_t stream) {
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

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(
      reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuStreamIsCapturing for TMA descriptor slot failed: CUresult="
      << static_cast<int>(result);

  std::string key(reinterpret_cast<const char*>(&tm), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  TmaDeviceArena& arena = TmaDeviceArenas()[current_context];
  auto pinned = arena.pinned_slots.find(key);
  if (pinned != arena.pinned_slots.end()) {
    CUdeviceptr pointer = arena.slots[pinned->second].pointer;
    return {reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)),
            current_context, pinned->second, false};
  }

  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    for (size_t index = 0; index < arena.slots.size(); ++index) {
      auto& slot = arena.slots[index];
      if (!slot.pinned && slot.key == key) {
        TVM_FFI_CHECK(
            arena.pinned_count < TmaDeviceArena::kMaxPinnedSlots,
            RuntimeError)
            << "captured pointer TMA descriptor arena is exhausted";
        slot.pinned = true;
        --arena.reusable_slots;
        ++arena.pinned_count;
        arena.pinned_slots.emplace(key, index);
        return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)),
                current_context, index, false};
      }
    }
    TVM_FFI_CHECK(false, RuntimeError)
        << "prewarm each pointer TMA tensor/layout binding before CUDA Graph "
           "capture";
  }

  for (size_t index = 0; index < arena.slots.size(); ++index) {
    auto& slot = arena.slots[index];
    if (!slot.pinned && !slot.reserved && slot.key == key &&
        (!slot.has_completion || slot.last_stream == stream ||
         TmaDeviceSlotReady(slot))) {
      slot.reserved = true;
      slot.last_stream = stream;
      return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)),
              current_context, index, true};
    }
  }

  size_t selected = arena.slots.size();
  for (size_t offset = 0; offset < arena.slots.size(); ++offset) {
    size_t index = (arena.cursor + offset) % arena.slots.size();
    auto& slot = arena.slots[index];
    if (!slot.pinned && !slot.reserved && TmaDeviceSlotReady(slot)) {
      selected = index;
      break;
    }
  }
  if (selected == arena.slots.size() &&
      arena.reusable_slots < TmaDeviceArena::kMaxReusableSlots) {
    size_t first_new_slot = arena.slots.size();
    AddTmaDeviceSlotChunk(arena);
    selected = first_new_slot;
  }
  if (selected == arena.slots.size()) {
    for (size_t offset = 0; offset < arena.slots.size(); ++offset) {
      size_t index = (arena.cursor + offset) % arena.slots.size();
      auto& slot = arena.slots[index];
      if (!slot.pinned && !slot.reserved) {
        cudaError_t status = cudaEventSynchronize(slot.completion);
        TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
            << "failed to wait for a reusable pointer TMA descriptor: "
            << cudaGetErrorString(status);
        selected = index;
        break;
      }
    }
  }
  TVM_FFI_CHECK(selected < arena.slots.size(), RuntimeError)
      << "too many concurrent pointer TMA descriptor leases";

  auto& slot = arena.slots[selected];
  result = cuMemcpyHtoD(slot.pointer, &tm, sizeof(CUtensorMap));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "cuMemcpyHtoD for TMA descriptor slot failed: CUresult="
      << static_cast<int>(result);
  slot.key = key;
  slot.last_stream = stream;
  slot.reserved = true;
  arena.cursor = (selected + 1) % arena.slots.size();
  return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)),
          current_context, selected, true};
}

static inline void RecordTmaDeviceSlotUses(
    std::initializer_list<TmaDeviceSlotLease> leases,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  for (const auto& lease : leases) {
    if (!lease.track_completion) continue;
    auto arena_it = TmaDeviceArenas().find(lease.context);
    TVM_FFI_CHECK(arena_it != TmaDeviceArenas().end(), RuntimeError)
        << "pointer TMA descriptor arena disappeared before completion";
    auto& arena = arena_it->second;
    TVM_FFI_CHECK(lease.slot_index < arena.slots.size(), RuntimeError)
        << "pointer TMA descriptor lease index is out of range";
    auto& slot = arena.slots[lease.slot_index];
    if (slot.pinned) continue;
    cudaError_t status = cudaEventRecord(slot.completion, stream);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to record pointer TMA descriptor completion: "
        << cudaGetErrorString(status);
    slot.has_completion = true;
    slot.last_stream = stream;
    slot.reserved = false;
  }
}

// 3D TMA descriptor for buffer 'Qt' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Qt(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Qt' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Qt' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Qt' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "Qt");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Qt' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Qt' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Qt' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 8u <= global_dim[1] && 2u <= global_dim[2], ValueError)
      << "TMA box (64, 8, 2) exceeds resolved global dims for 'Qt'";
  uint64_t global_strides[2] = {
      (uint64_t)((s2 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 8u, 2u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Qt') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'K' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  int64_t s4 = t.stride(t.ndim() - 4) * 1;
  TVM_FFI_CHECK(s4 > 0, ValueError)
      << "TMA source 'K' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'K' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[5] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 16u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (64, 16, 1, 1, 1) exceeds resolved global dims for 'K'";
  uint64_t global_strides[4] = {
      (uint64_t)((s2 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s4 * 16) / 8),
  };
  uint32_t box_dim[5] = {64u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'K') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'V' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_V(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'V' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'V' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'V' trailing dims must be positive";
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  int64_t s4 = t.stride(t.ndim() - 4) * 1;
  TVM_FFI_CHECK(s4 > 0, ValueError)
      << "TMA source 'V' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'V' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[5] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'V' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 16u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (64, 16, 1, 1, 1) exceeds resolved global dims for 'V'";
  uint64_t global_strides[4] = {
      (uint64_t)((s2 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)((s3 * 16) / 8),
      (uint64_t)((s4 * 16) / 8),
  };
  uint32_t box_dim[5] = {64u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'V') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Qt, TensorView arg_K, TensorView arg_V, TensorView arg_partial_O_ptr, TensorView arg_partial_LSE_ptr, TensorView arg_O_ptr, TensorView arg_LSE_ptr, TensorView arg_split_completion, TensorView arg_page_table, TensorView arg_causal_seqlens_kv_global, int64_t arg_max_pages_per_seq, int64_t arg_max_local_seq_len, double arg_softmax_scale_log2, int64_t arg_cp_rank, int64_t arg_num_q_heads, int64_t arg_num_kv_heads, int64_t arg_batch_size, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  CheckCudaTensor(arg_Qt, "Qt");
  CheckDtype(arg_Qt, "Qt", 4, 16, 1);
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 4, 16, 1);
  CheckCudaTensor(arg_V, "V");
  CheckDtype(arg_V, "V", 4, 16, 1);
  CheckCudaTensor(arg_partial_O_ptr, "partial_O_ptr");
  CheckDtype(arg_partial_O_ptr, "partial_O_ptr", 4, 16, 1);
  CheckContiguous(arg_partial_O_ptr, "partial_O_ptr");
  CheckCudaTensor(arg_partial_LSE_ptr, "partial_LSE_ptr");
  CheckDtype(arg_partial_LSE_ptr, "partial_LSE_ptr", 2, 32, 1);
  CheckContiguous(arg_partial_LSE_ptr, "partial_LSE_ptr");
  CheckCudaTensor(arg_O_ptr, "O_ptr");
  CheckDtype(arg_O_ptr, "O_ptr", 4, 16, 1);
  CheckContiguous(arg_O_ptr, "O_ptr");
  CheckCudaTensor(arg_LSE_ptr, "LSE_ptr");
  CheckDtype(arg_LSE_ptr, "LSE_ptr", 2, 32, 1);
  CheckContiguous(arg_LSE_ptr, "LSE_ptr");
  CheckCudaTensor(arg_split_completion, "split_completion");
  CheckDtype(arg_split_completion, "split_completion", 0, 32, 1);
  CheckContiguous(arg_split_completion, "split_completion");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckDtype(arg_page_table, "page_table", 0, 32, 1);
  CheckContiguous(arg_page_table, "page_table");
  CheckCudaTensor(arg_causal_seqlens_kv_global, "causal_seqlens_kv_global");
  CheckDtype(arg_causal_seqlens_kv_global, "causal_seqlens_kv_global", 0, 32, 1);
  CheckContiguous(arg_causal_seqlens_kv_global, "causal_seqlens_kv_global");
  TVM_FFI_CHECK(arg_max_pages_per_seq >= -2147483648LL && arg_max_pages_per_seq <= 2147483647LL, ValueError)
      << "scalar 'max_pages_per_seq' value " << arg_max_pages_per_seq
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_local_seq_len >= -2147483648LL && arg_max_local_seq_len <= 2147483647LL, ValueError)
      << "scalar 'max_local_seq_len' value " << arg_max_local_seq_len
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_cp_rank >= -2147483648LL && arg_cp_rank <= 2147483647LL, ValueError)
      << "scalar 'cp_rank' value " << arg_cp_rank
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_kv_heads >= -2147483648LL && arg_num_kv_heads <= 2147483647LL, ValueError)
      << "scalar 'num_kv_heads' value " << arg_num_kv_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_batch_size >= -2147483648LL && arg_batch_size <= 2147483647LL, ValueError)
      << "scalar 'batch_size' value " << arg_batch_size
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_K, arg_Qt, "K", "Qt");
  CheckSameCudaDevice(arg_V, arg_Qt, "V", "Qt");
  CheckSameCudaDevice(arg_partial_O_ptr, arg_Qt, "partial_O_ptr", "Qt");
  CheckSameCudaDevice(arg_partial_LSE_ptr, arg_Qt, "partial_LSE_ptr", "Qt");
  CheckSameCudaDevice(arg_O_ptr, arg_Qt, "O_ptr", "Qt");
  CheckSameCudaDevice(arg_LSE_ptr, arg_Qt, "LSE_ptr", "Qt");
  CheckSameCudaDevice(arg_split_completion, arg_Qt, "split_completion", "Qt");
  CheckSameCudaDevice(arg_page_table, arg_Qt, "page_table", "Qt");
  CheckSameCudaDevice(arg_causal_seqlens_kv_global, arg_Qt, "causal_seqlens_kv_global", "Qt");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  DLDevice dev = arg_Qt.device();
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap h_Qt = EncodeTma_Qt(arg_Qt);
  auto slot_Qt = TmaDeviceSlot(h_Qt, arg_Qt.device().device_id, stream);
  void* p_Qt = slot_Qt.pointer;
  CUtensorMap h_K = EncodeTma_K(arg_K);
  auto slot_K = TmaDeviceSlot(h_K, arg_K.device().device_id, stream);
  void* p_K = slot_K.pointer;
  CUtensorMap h_V = EncodeTma_V(arg_V);
  auto slot_V = TmaDeviceSlot(h_V, arg_V.device().device_id, stream);
  void* p_V = slot_V.pointer;
  void* p_partial_O_ptr = arg_partial_O_ptr.data_ptr();
  void* p_partial_LSE_ptr = arg_partial_LSE_ptr.data_ptr();
  void* p_O_ptr = arg_O_ptr.data_ptr();
  void* p_LSE_ptr = arg_LSE_ptr.data_ptr();
  void* p_split_completion = arg_split_completion.data_ptr();
  void* p_page_table = arg_page_table.data_ptr();
  void* p_causal_seqlens_kv_global = arg_causal_seqlens_kv_global.data_ptr();
  int32_t v_max_pages_per_seq = (int32_t)arg_max_pages_per_seq;
  int32_t v_max_local_seq_len = (int32_t)arg_max_local_seq_len;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  int32_t v_cp_rank = (int32_t)arg_cp_rank;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_kv_heads = (int32_t)arg_num_kv_heads;
  int32_t v_batch_size = (int32_t)arg_batch_size;
  void* kargs[] = {&p_Qt, &p_K, &p_V, &p_partial_O_ptr, &p_partial_LSE_ptr, &p_O_ptr, &p_LSE_ptr, &p_split_completion, &p_page_table, &p_causal_seqlens_kv_global, &v_max_pages_per_seq, &v_max_local_seq_len, &v_softmax_scale_log2, &v_cp_rank, &v_num_q_heads, &v_num_kv_heads, &v_batch_size};

  static std::once_flag smem_once;
  static cudaError_t smem_status = cudaSuccess;
  std::call_once(smem_once, [] {
    smem_status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel_cake_fmha_dcp_spec_bf16_v4),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        145408);
  });
  TVM_FFI_CHECK(smem_status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute for kernel_cake_fmha_dcp_spec_bf16_v4 failed: "
      << cudaGetErrorString(smem_status);
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);

  cudaError_t launch_status = cudaLaunchKernel(
      reinterpret_cast<const void*>(kernel_cake_fmha_dcp_spec_bf16_v4), grid, block, kargs, 145408u, stream);
  RecordTmaDeviceSlotUses({slot_Qt, slot_K, slot_V}, stream);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernel for kernel_cake_fmha_dcp_spec_bf16_v4 failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_fmha_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_fmha_host_shim::Run);
