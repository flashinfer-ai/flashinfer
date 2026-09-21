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

#ifndef Q_LEN
#error "Q_LEN must be supplied by the route-specific JIT"
#endif
#ifndef GROUP
#error "GROUP must be supplied by the route-specific JIT"
#endif
#ifndef NUM_SPLIT
#error "NUM_SPLIT must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_SMALLM_N_ROWS
#error "CAKE_FMHA_SMALLM_N_ROWS must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_SMALLM_PAGE_SIZE
#error "CAKE_FMHA_SMALLM_PAGE_SIZE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_SMALLM_LAUNCH
#error "CAKE_FMHA_SMALLM_LAUNCH must name the component launch binding"
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
TmaDeviceSlotLease TmaDeviceSlot(const CUtensorMap& tm, int device_id,
                                 cudaStream_t stream) {
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
        return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)),
                current_context, index, false};
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

CUtensorMap EncodeTmaPackedQ(TensorView tensor) {
  // Q [total_q, num_q_heads, 256] viewed as (64 dims, heads, rows, 4 k-groups);
  // one box covers GROUP heads x Q_LEN rows x two k-groups (one 128-wide half).
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), 256);
  uint64_t global_dim[4] = {64u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0)), 4u};
  uint64_t global_strides[3] = {static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2), 128u};
  uint32_t box_dim[4] = {64u, static_cast<uint32_t>(GROUP), static_cast<uint32_t>(Q_LEN), 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA packed query tensor map";
  return tm;
}

CUtensorMap EncodeTmaPagedKvHd256(TensorView tensor, const char* name) {
  // HND [pages, num_kv_heads, PAGE_SIZE, 256] viewed as (64, page tokens, 4 k-groups, heads, pages).
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(tensor.size(2), CAKE_FMHA_SMALLM_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(3), 256);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(2), 0);
  uint64_t global_dim[5] = {64u, static_cast<uint64_t>(CAKE_FMHA_SMALLM_PAGE_SIZE), 4u,
                            static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2) * 2), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2)};
  uint32_t box_dim[5] = {64u, static_cast<uint32_t>(CAKE_FMHA_SMALLM_PAGE_SIZE), 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA " << name << " tensor map";
  return tm;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

// Small-M speculative decode: BF16 Q/KV/O, head_dim 256, HND paged KV, packed
// (Q_LEN x GROUP) rows per KV head, one resident wave of NUM_SPLIT Split-KV CTAs
// per (batch, kv_head) tile and a fused in-launch merge.  The signature is the
// shared Cake FMHA decode ABI; unsupported options fail closed.
void cake_paged_attention_decode(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl,
    int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> cum_seq_lens_q, Optional<TensorView> key_block_scales,
    Optional<TensorView> value_block_scales, Optional<float> skip_softmax_threshold_scale_factor,
    Optional<bool> uses_shared_paged_kv_idx, Optional<TensorView> lse, int64_t lse_stride_tokens,
    int64_t lse_stride_heads, bool enable_block_sparse_attention,
    Optional<TensorView> sparse_mla_top_k_lens) {
  constexpr int kHeadDim = 256;
  constexpr int kNumRows = CAKE_FMHA_SMALLM_N_ROWS;
  static_assert(Q_LEN * GROUP == kNumRows, "Q_LEN * GROUP must equal the packed row count");
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(query.IsContiguous());
  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(max_q_len, Q_LEN);
  TVM_FFI_ICHECK_EQ(query.size(0), batch_size * Q_LEN);
  int64_t num_q_heads = query.size(1);
  int64_t num_kv_heads = key_cache.size(1);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_EQ(num_q_heads, num_kv_heads * GROUP);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), value_cache.size(2));
  TVM_FFI_ICHECK_EQ(key_cache.size(3), value_cache.size(3));
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!cum_seq_lens_q.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() && !value_block_scales.has_value());
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(uses_shared_paged_kv_idx.value_or(true));
  TVM_FFI_ICHECK(!enable_block_sparse_attention && !sparse_mla_top_k_lens.has_value());
  TVM_FFI_ICHECK_EQ(sparse_mla_top_k, 0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK(!attention_sinks.has_value()) << "small-M hd256 decode has no attention sinks";
  TVM_FFI_ICHECK_LT(window_left, 0) << "small-M hd256 decode has no sliding window";
  TVM_FFI_ICHECK_EQ(ScalarScale(bmm2_scale, "bmm2_scale"), 1.0);
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 || block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK(block_tables.IsContiguous());
  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK(seq_lens.dtype() == dl_int32 || seq_lens.dtype() == dl_uint32);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);
  int64_t tiles = batch_size * num_kv_heads;
  TVM_FFI_ICHECK_LE(tiles * NUM_SPLIT, sm_count)
      << "small-M hd256 decode requires one resident wave: batch * kv_heads * NUM_SPLIT <= SMs";

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  if (lse.has_value()) CheckSameDevice(query, lse.value(), "lse");

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  CUtensorMap h_q = EncodeTmaPackedQ(query);
  CUtensorMap h_k = EncodeTmaPagedKvHd256(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKvHd256(value_cache, "value_cache");
  auto q_slot = TmaDeviceSlot(h_q, query.device().device_id, stream);
  auto k_slot = TmaDeviceSlot(h_k, query.device().device_id, stream);
  auto v_slot = TmaDeviceSlot(h_v, query.device().device_id, stream);
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(q_slot.pointer);
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(k_slot.pointer);
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(v_slot.pointer);

  // Workspace: BF16 partial O slots, FP32 partial LSE slots, two u32 counters per
  // tile (zeroed before every launch; the kernel resets them as well), and the
  // LSE destination when the caller does not provide one.
  int64_t slots = tiles * NUM_SPLIT;
  int64_t partial_o_offset = 0;
  int64_t cursor = AlignUp(slots * kNumRows * kHeadDim * static_cast<int64_t>(sizeof(__nv_bfloat16)), 256);
  int64_t partial_lse_offset = cursor;
  cursor = AlignUp(cursor + slots * kNumRows * static_cast<int64_t>(sizeof(float)), 256);
  int64_t counters_offset = cursor;
  cursor = AlignUp(cursor + tiles * 2 * static_cast<int64_t>(sizeof(uint32_t)), 256);
  int64_t lse_offset = cursor;
  if (!lse.has_value()) {
    cursor += query.size(0) * query.size(1) * static_cast<int64_t>(sizeof(float));
  }
  int64_t actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA small-M hd256 decode workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* partial_o = reinterpret_cast<__nv_bfloat16*>(workspace + partial_o_offset);
  auto* partial_lse = reinterpret_cast<float*>(workspace + partial_lse_offset);
  auto* counters = reinterpret_cast<uint32_t*>(workspace + counters_offset);
  TVM_FFI_ICHECK_EQ(cudaMemsetAsync(counters, 0, tiles * 2 * sizeof(uint32_t), stream), cudaSuccess)
      << "failed to reset Cake FMHA small-M merge counters";

  float* lse_ptr = nullptr;
  if (lse.has_value()) {
    auto const& lse_tensor = lse.value();
    TVM_FFI_ICHECK_EQ(lse_tensor.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(lse_tensor.ndim(), 2);
    TVM_FFI_ICHECK_EQ(lse_tensor.size(0), query.size(0));
    TVM_FFI_ICHECK_EQ(lse_tensor.size(1), query.size(1));
    TVM_FFI_ICHECK_EQ(lse_stride_tokens, query.size(1));
    TVM_FFI_ICHECK_EQ(lse_stride_heads, 1);
    lse_ptr = static_cast<float*>(lse_tensor.data_ptr());
  } else {
    lse_ptr = reinterpret_cast<float*>(workspace + lse_offset);
  }

  float softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") * 1.4426950408889634);

  cudaError_t status = CAKE_FMHA_SMALLM_LAUNCH(
      p_q, p_k, p_v, partial_o, partial_lse, static_cast<__nv_bfloat16*>(out.data_ptr()), lse_ptr,
      counters, static_cast<int*>(block_tables.data_ptr()), static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int>(block_tables.size(1)), softmax_scale_log2, static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads), static_cast<unsigned int>(slots), 1u, 1u, stream);
  RecordTmaDeviceSlotUses({q_slot, k_slot, v_slot}, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA small-M hd256 decode launch failed: " << cudaGetErrorString(status);

  (void)multi_ctas_kv_counter_buffer;
  (void)enable_pdl;
  (void)max_kv_len;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
