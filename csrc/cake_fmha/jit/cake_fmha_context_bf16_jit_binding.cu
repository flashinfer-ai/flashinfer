/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/variant.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

#ifndef NUM_M_BLOCKS
#error "NUM_M_BLOCKS must be supplied by the route-specific JIT"
#endif
#ifndef NUM_Q_HEADS
#error "NUM_Q_HEADS must be supplied by the route-specific JIT"
#endif
#ifndef HEADS_PER_GROUP
#error "HEADS_PER_GROUP must be supplied by the route-specific JIT"
#endif
#ifndef PACK_G
#error "PACK_G must be supplied by the route-specific JIT"
#endif
#ifndef TOK_PER_STAGE
#error "TOK_PER_STAGE must be supplied by the route-specific JIT"
#endif
#ifndef PAGE_SIZE
#error "PAGE_SIZE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_IS_CAUSAL
#error "CAKE_FMHA_CONTEXT_IS_CAUSAL must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_RETURN_LSE
#error "CAKE_FMHA_CONTEXT_RETURN_LSE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_CONTEXT_ENABLE_SINK
#error "CAKE_FMHA_CONTEXT_ENABLE_SINK must be supplied by the route-specific JIT"
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}

double ScalarScale(Variant<double, ffi::Tensor> scale, const char* name) {
  auto scalar = scale.as<double>();
  TVM_FFI_ICHECK(scalar.has_value()) << name << " must be a host scalar on this specialization";
  return scalar.value();
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
  std::vector<cudaEvent_t> events;
  std::vector<TmaDeviceSlotState> slots;
  std::unordered_map<std::string, size_t> pinned_slots;
  unsigned long long context_id = 0;
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

bool TmaDeviceSlotReady(const TmaDeviceSlotState& slot) {
  if (!slot.has_completion) return true;
  cudaError_t status = cudaEventQuery(slot.completion);
  if (status == cudaSuccess) return true;
  TVM_FFI_ICHECK_EQ(status, cudaErrorNotReady)
      << "failed to query Cake FMHA TMA descriptor completion: "
      << cudaGetErrorString(status);
  return false;
}

void AddTmaDeviceSlotChunk(TmaDeviceArena& arena) {
  size_t count = std::min(TmaDeviceArena::kSlotsPerChunk,
                          TmaDeviceArena::kMaxReusableSlots - arena.reusable_slots);
  TVM_FFI_ICHECK_GT(count, 0);
  CUdeviceptr chunk = 0;
  CUresult result = cuMemAlloc(&chunk, count * sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to allocate Cake FMHA TMA descriptor chunk";
  arena.chunks.push_back(chunk);
  for (size_t index = 0; index < count; ++index) {
    cudaEvent_t completion = nullptr;
    cudaError_t status = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to create Cake FMHA TMA descriptor completion event: "
        << cudaGetErrorString(status);
    arena.events.push_back(completion);
    arena.slots.push_back(
        {chunk + index * sizeof(CUtensorMap), completion, nullptr, "", false, false, false});
  }
  arena.reusable_slots += count;
}

std::mutex& TmaDeviceSlotMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<CUcontext, TmaDeviceArena>& TmaDeviceArenas() {
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();
  return *arenas;
}

// Eager descriptors live in a bounded, completion-tracked pool. An exact
// prewarmed descriptor is removed from that pool when capture first observes
// it, keeping its device address immutable for every replay of that graph.
TmaDeviceSlotLease TmaDeviceSlot(const CUtensorMap& tm, int device_id, cudaStream_t stream) {
  CUcontext current_context = nullptr;
  CUresult result = cuCtxGetCurrent(&current_context);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_context != nullptr)
      << "Cake FMHA TMA launch requires an active CUDA context";
  CUdevice current_device = -1;
  result = cuCtxGetDevice(&current_device);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_device == device_id)
      << "Cake FMHA TMA descriptor device mismatch";
  unsigned long long current_context_id = 0;
  result = cuCtxGetId(current_context, &current_context_id);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to resolve Cake FMHA CUDA context identity";

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);

  std::string key(reinterpret_cast<const char*>(&tm), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  auto& arenas = TmaDeviceArenas();
  auto arena_it = arenas.find(current_context);
  if (arena_it != arenas.end() &&
      arena_it->second.context_id != current_context_id) {
    arenas.erase(arena_it);
  }
  TmaDeviceArena& arena = arenas[current_context];
  arena.context_id = current_context_id;
  auto pinned = arena.pinned_slots.find(key);
  if (pinned != arena.pinned_slots.end()) {
    CUdeviceptr pointer = arena.slots[pinned->second].pointer;
    return {reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)), current_context,
            pinned->second, false};
  }

  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    for (size_t index = 0; index < arena.slots.size(); ++index) {
      auto& slot = arena.slots[index];
      if (!slot.pinned && slot.key == key) {
        TVM_FFI_ICHECK_LT(arena.pinned_count, TmaDeviceArena::kMaxPinnedSlots)
            << "Cake FMHA captured TMA descriptor arena is exhausted";
        slot.pinned = true;
        --arena.reusable_slots;
        ++arena.pinned_count;
        arena.pinned_slots.emplace(key, index);
        return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
                index, false};
      }
    }
    TVM_FFI_ICHECK(false)
        << "prewarm each Cake FMHA tensor/layout binding before CUDA Graph capture";
  }

  for (size_t index = 0; index < arena.slots.size(); ++index) {
    auto& slot = arena.slots[index];
    if (!slot.pinned && !slot.reserved && slot.key == key &&
        (!slot.has_completion || slot.last_stream == stream || TmaDeviceSlotReady(slot))) {
      slot.reserved = true;
      slot.last_stream = stream;
      return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
              index, true};
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
        TVM_FFI_ICHECK_EQ(status, cudaSuccess)
            << "failed to wait for a reusable Cake FMHA TMA descriptor: "
            << cudaGetErrorString(status);
        selected = index;
        break;
      }
    }
  }
  TVM_FFI_ICHECK_LT(selected, arena.slots.size())
      << "too many concurrent Cake FMHA TMA descriptor leases";

  auto& slot = arena.slots[selected];
  result = cuMemcpyHtoD(slot.pointer, &tm, sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);
  slot.key = key;
  slot.last_stream = stream;
  slot.reserved = true;
  arena.cursor = (selected + 1) % arena.slots.size();
  return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
          selected, true};
}

void RecordTmaDeviceSlotUses(std::initializer_list<TmaDeviceSlotLease> leases,
                             cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  for (const auto& lease : leases) {
    if (!lease.track_completion) continue;
    auto arena_it = TmaDeviceArenas().find(lease.context);
    TVM_FFI_ICHECK(arena_it != TmaDeviceArenas().end())
        << "Cake FMHA TMA descriptor arena disappeared before completion";
    auto& arena = arena_it->second;
    TVM_FFI_ICHECK_LT(lease.slot_index, arena.slots.size())
        << "Cake FMHA TMA descriptor lease index is out of range";
    auto& slot = arena.slots[lease.slot_index];
    if (slot.pinned) {
      slot.reserved = false;
      continue;
    }
    cudaError_t status = cudaEventRecord(slot.completion, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to record Cake FMHA TMA descriptor completion: "
        << cudaGetErrorString(status);
    slot.has_completion = true;
    slot.last_stream = stream;
    slot.reserved = false;
  }
}

CUtensorMap EncodeTmaQ(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(tensor.size(2), 128);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), 1);
  uint64_t global_dim[5] = {64u, PACK_G, static_cast<uint64_t>(tensor.size(0)), 2u,
                            NUM_Q_HEADS / PACK_G};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * PACK_G * 2)};
  uint32_t box_dim[5] = {64u, PACK_G, TOK_PER_STAGE, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA context query map";
  return tm;
}

CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS / HEADS_PER_GROUP);
  TVM_FFI_ICHECK_EQ(tensor.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(3), 128);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  uint64_t global_dim[5] = {64u, PAGE_SIZE, 2u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2) * 2), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2)};
  uint32_t box_dim[5] = {64u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA context " << name << " map";
  return tm;
}

__global__ void PrepareContextMetadata(const int* cum_seq_lens_q, const int* seq_lens_kv,
                                       int* expanded_q, int* expanded_kv, int* expanded_cu_q,
                                       int total_bh, int units_per_batch) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < total_bh) {
    int batch = index / units_per_batch;
    expanded_q[index] = cum_seq_lens_q[batch + 1] - cum_seq_lens_q[batch];
    expanded_kv[index] = seq_lens_kv[batch];
    expanded_cu_q[index] = cum_seq_lens_q[batch];
  }
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

void cake_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> key_block_scales, Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor, Optional<bool> uses_shared_paged_kv_idx,
    Optional<bool> use_fp16_softmax, Optional<bool> uses_spcompress, bool is_causal,
    Optional<TensorView> lse, int64_t lse_stride_tokens, int64_t lse_stride_heads) {
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() && !value_block_scales.has_value());
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));
  TVM_FFI_ICHECK(!uses_spcompress.value_or(false));
  TVM_FFI_ICHECK_EQ(window_left, -1);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  TVM_FFI_ICHECK_EQ(ScalarScale(bmm2_scale, "bmm2_scale"), 1.0);
  TVM_FFI_ICHECK_EQ(is_causal ? 1 : 0, CAKE_FMHA_CONTEXT_IS_CAUSAL);
  TVM_FFI_ICHECK_EQ(attention_sinks.has_value() ? 1 : 0, CAKE_FMHA_CONTEXT_ENABLE_SINK);
  TVM_FFI_ICHECK_EQ(lse.has_value() ? 1 : 0, CAKE_FMHA_CONTEXT_RETURN_LSE);
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(max_q_len, 0);
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);

  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(query.size(2), 128);
  TVM_FFI_ICHECK_EQ(query.stride(2), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), NUM_Q_HEADS / HEADS_PER_GROUP);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), key_cache.size(1));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(value_cache.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(key_cache.size(3), 128);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), 128);
  TVM_FFI_ICHECK_EQ(key_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(value_cache.stride(3), 1);

  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_q.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_kv.IsContiguous());
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 || block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK(multi_ctas_kv_counter_buffer.IsContiguous());

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, multi_ctas_kv_counter_buffer, "multi_ctas_kv_counter_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  CheckSameDevice(query, cum_seq_lens_q, "cum_seq_lens_q");
  CheckSameDevice(query, cum_seq_lens_kv, "cum_seq_lens_kv");
  if (attention_sinks.has_value()) {
    CheckSameDevice(query, attention_sinks.value(), "attention_sinks");
  }
  if (lse.has_value()) CheckSameDevice(query, lse.value(), "lse");

  int* table_base = static_cast<int*>(block_tables.data_ptr());
  int* table_k = table_base;
  int* table_v = table_base;
  int64_t page_row_stride = block_tables.stride(0);
  if (uses_shared_paged_kv_idx.value_or(true)) {
    TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
    TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(block_tables.stride(1), 1);
  } else {
    TVM_FFI_ICHECK_EQ(block_tables.ndim(), 3);
    TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(block_tables.size(1), 2);
    TVM_FFI_ICHECK_EQ(block_tables.stride(2), 1);
    table_v = table_base + block_tables.stride(1);
  }

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  CUtensorMap h_q = EncodeTmaQ(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");

  constexpr int units_per_batch = NUM_Q_HEADS / PACK_G;
  int64_t total_bh_64 = batch_size * units_per_batch;
  TVM_FFI_ICHECK_LE(total_bh_64, static_cast<int64_t>(INT32_MAX));
  int total_bh = static_cast<int>(total_bh_64);
  int64_t seq_q_offset = 0;
  int64_t seq_kv_offset = AlignUp(total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  int64_t cu_q_offset = seq_kv_offset + total_bh_64 * static_cast<int64_t>(sizeof(int));
  int64_t cursor = AlignUp(cu_q_offset + total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  int64_t actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA context BF16 workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  int64_t counter_bytes =
      multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(counter_bytes, 2 * static_cast<int64_t>(sizeof(uint32_t)));

  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* seq_q_expanded = reinterpret_cast<int*>(workspace + seq_q_offset);
  auto* seq_kv_expanded = reinterpret_cast<int*>(workspace + seq_kv_offset);
  auto* cu_q_expanded = reinterpret_cast<int*>(workspace + cu_q_offset);
  int const threads = 256;
  int const blocks = (total_bh + threads - 1) / threads;
  PrepareContextMetadata<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(cum_seq_lens_q.data_ptr()),
      static_cast<const int*>(seq_lens.data_ptr()), seq_q_expanded, seq_kv_expanded, cu_q_expanded,
      total_bh, units_per_batch);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "failed to prepare Cake FMHA context metadata";

  float* lse_ptr = reinterpret_cast<float*>(seq_q_expanded);
  if (lse.has_value()) {
    auto const& lse_tensor = lse.value();
    TVM_FFI_ICHECK_EQ(lse_tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(lse_tensor.ndim(), 2);
    TVM_FFI_ICHECK_EQ(lse_tensor.size(0), query.size(0));
    TVM_FFI_ICHECK_EQ(lse_tensor.size(1), NUM_Q_HEADS);
    TVM_FFI_ICHECK_EQ(lse_stride_tokens, NUM_Q_HEADS);
    TVM_FFI_ICHECK_EQ(lse_stride_heads, 1);
    lse_ptr = static_cast<float*>(lse_tensor.data_ptr());
  }
  float* sinks_ptr = reinterpret_cast<float*>(seq_q_expanded);
  if (attention_sinks.has_value()) {
    auto const& sinks = attention_sinks.value();
    TVM_FFI_ICHECK_EQ(sinks.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(sinks.numel(), NUM_Q_HEADS);
    TVM_FFI_ICHECK(sinks.IsContiguous());
    sinks_ptr = static_cast<float*>(sinks.data_ptr());
  }

  float softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") * 1.4426950408889634);
  unsigned int total_tiles = static_cast<unsigned int>(NUM_M_BLOCKS * total_bh);
  unsigned int grid_x = std::min<unsigned int>(static_cast<unsigned int>(sm_count), total_tiles);
  auto* dynamic_counter = static_cast<uint32_t*>(multi_ctas_kv_counter_buffer.data_ptr());
  auto q_slot = TmaDeviceSlot(h_q, query.device().device_id, stream);
  auto k_slot = TmaDeviceSlot(h_k, query.device().device_id, stream);
  auto v_slot = TmaDeviceSlot(h_v, query.device().device_id, stream);
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(q_slot.pointer);
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(k_slot.pointer);
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(v_slot.pointer);
  cudaError_t status = cake_fmha_launch_context_bf16(
      p_q, p_k, p_v, static_cast<__nv_bfloat16*>(out.data_ptr()), lse_ptr, sinks_ptr, table_k,
      table_v, seq_q_expanded, seq_kv_expanded, cu_q_expanded, softmax_scale_log2, total_bh,
      static_cast<int>(page_row_stride), static_cast<int>(grid_x), dynamic_counter, grid_x, 1, 1,
      stream);
  RecordTmaDeviceSlotUses({q_slot, k_slot, v_slot}, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA context BF16 launch failed: " << cudaGetErrorString(status);

  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_context,
                              flashinfer::cake_fmha::cake_paged_attention_context);
