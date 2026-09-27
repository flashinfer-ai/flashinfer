/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// On-device load-balanced BF16 paged decode (Cake route ``decode_balanced_bf16_v1``).
//
// One persistent CTA per SM plans the split-KV schedule on the device from the
// ``seq_lens`` buffer: a scheduler warp derives the chunk length, buckets the
// (request, KV head) work tiles and hands out tickets; the last chunk of a split
// tile merges the partials in place.  Batch, heads and KV lengths are runtime
// kernel arguments, so one module per ``Q_LEN`` serves every shape, and a
// prepared launch replays under CUDA Graph capture for any length vector.
//
// Two generated programs share this adapter:
//   * ``CAKE_FMHA_BALANCED_N_ROWS == 0``  — the eight-row kernel (``Q_LEN == 1``,
//     one GQA-8 query row per item).
//   * ``CAKE_FMHA_BALANCED_N_ROWS == 32 | 64`` — the packed-row MTP kernel
//     (``Q_LEN`` 3..8: one ``8 * Q_LEN``-row tile per (request, KV head) so each
//     KV chunk streams once per request instead of once per draft row).
//
// The self-resetting split counters live in the zero-initialized
// ``multi_ctas_kv_counter_buffer`` (the trtllm-gen contract: zeroed once at
// allocation, reset by the kernel before it exits); FP32 partial outputs and
// statistics are carved from the caller-owned ``workspace_buffer``, which may be
// uninitialized.

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
#ifndef CAKE_FMHA_BALANCED_N_ROWS
#error "CAKE_FMHA_BALANCED_N_ROWS must be supplied by the route-specific JIT (0, 32 or 64)"
#endif
#ifndef CAKE_FMHA_BALANCED_LAUNCH
#error "CAKE_FMHA_BALANCED_LAUNCH must name the exported launch binding"
#endif

#if CAKE_FMHA_BALANCED_N_ROWS == 0
static_assert(Q_LEN == 1, "the eight-row balanced kernel serves q_len == 1");
#elif CAKE_FMHA_BALANCED_N_ROWS == 32
static_assert(Q_LEN >= 3 && Q_LEN <= 4, "the 32-row packed MTP kernel serves q_len 3..4");
#elif CAKE_FMHA_BALANCED_N_ROWS == 64
static_assert(Q_LEN >= 5 && Q_LEN <= 8, "the 64-row packed MTP kernel serves q_len 5..8");
#else
#error "CAKE_FMHA_BALANCED_N_ROWS must be 0, 32 or 64"
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

constexpr int64_t kGroup = 8;              // query heads per KV head (ForGen GQA-8 layout)
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 16;
constexpr int64_t kMaxRequests = 1024;     // MAX_REQUEST_GROUPS * REQUEST_GROUP
constexpr int64_t kMaxBalanceFactor = 8;   // chunk length >= total work / (k * CTAs)
constexpr int64_t kQueueCounters = 4;      // ticket, done CTAs, L, total_items
#if CAKE_FMHA_BALANCED_N_ROWS == 0
constexpr int64_t kPartialOPerSlot = 8 * kHeadDim;   // FP32 O[8, 128] per split item
constexpr int64_t kStatsPerSlot = 16;                // m[8] then l[8]
constexpr int64_t kCountersPerTile = 1;              // chunk arrivals
#else
constexpr int64_t kPartialOPerSlot = 64 * kHeadDim;  // FP32 O^T[64, 128] per split item
constexpr int64_t kStatsPerSlot = 128;               // max[64] then sum[64]
constexpr int64_t kCountersPerTile = 2;              // chunk arrivals, merge slices done
constexpr uint32_t kQBoxRows = CAKE_FMHA_BALANCED_N_ROWS / kGroup;  // query tokens per Q box
#endif

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

#if CAKE_FMHA_BALANCED_N_ROWS == 0
// Eight-row kernel: the query [batch, Hq, 128] is one dense row matrix
// [batch * Hq, 128]; a box of eight consecutive rows is one GQA group.
CUtensorMap EncodeTmaQuery(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), kHeadDim);
  uint64_t global_dim[3] = {64u, static_cast<uint64_t>(tensor.size(0) * tensor.size(1)), 2u};
  uint64_t global_strides[2] = {256u, 128u};
  uint32_t box_dim[3] = {64u, 8u, 2u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA query tensor map";
  return tm;
}
#else
// Packed MTP kernel: the query [batch * q_len, Hq, 128] is read as a 4-D map
// (64 dims, head, token, k-group) whose box (64, GROUP, Q_BOX_ROWS, 2) lands in
// SMEM as packed row r = token * GROUP + head.  Tokens past the tensor are TMA
// zero fill (padding rows whose results are never stored), so the token box may
// exceed the token extent.
CUtensorMap EncodeTmaQuery(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), kHeadDim);
  TVM_FFI_ICHECK_GE(tensor.size(1), kGroup);
  uint64_t global_dim[4] = {64u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0)), 2u};
  uint64_t global_strides[3] = {static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2), 128u};
  uint32_t box_dim[4] = {64u, static_cast<uint32_t>(kGroup), kQBoxRows, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA packed query tensor map";
  return tm;
}
#endif

// K/V pools [pages, Hkv, 16, 128] as a 5-D map (64 dims, token, k-group, head,
// page) whose token/head/page steps are the source's physical strides, so the
// production stacked ``kv_cache[:, side]`` views load in place.
CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(tensor.size(2), kPageSize);
  TVM_FFI_ICHECK_EQ(tensor.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), kHeadDim);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  uint64_t global_dim[5] = {64u, 16u, 2u, static_cast<uint64_t>(tensor.size(1)),
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
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA " << name << " tensor map";
  return tm;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Shape-independent planner bounds: split items < 2 * k * CTAs and every split
// tile has at least two items, with k <= kMaxBalanceFactor.
int64_t MaxSplitItems(int64_t num_ctas) { return 2 * kMaxBalanceFactor * num_ctas; }
int64_t MaxSplitTiles(int64_t num_ctas) { return kMaxBalanceFactor * num_ctas; }

// Host ticket-loop bound: whole tiles, every possible split chunk and (packed
// kernel) every merge ticket.  Mirrors ``max_items_bound`` of the Cake modules.
int64_t MaxItems(int64_t batch_size, int64_t num_kv_heads, int64_t num_ctas) {
#if CAKE_FMHA_BALANCED_N_ROWS == 0
  return batch_size * Q_LEN * num_kv_heads + MaxSplitItems(num_ctas);
#else
  return batch_size * num_kv_heads + MaxSplitItems(num_ctas) +
         MaxSplitTiles(num_ctas) * (2 * Q_LEN);
#endif
}

}  // namespace

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
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_LE(batch_size, kMaxRequests)
      << "the balanced scheduler plans at most " << kMaxRequests << " requests per launch";
  TVM_FFI_ICHECK_EQ(query.size(0), batch_size * Q_LEN);
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  int64_t const num_q_heads = query.size(1);
  int64_t const num_kv_heads = key_cache.size(1);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(num_q_heads, kGroup * num_kv_heads)
      << "the balanced BF16 decode route serves exactly eight query heads per KV head";
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
  TVM_FFI_ICHECK(!attention_sinks.has_value()) << "the balanced route has no sink specialization";
  TVM_FFI_ICHECK_LT(window_left, 0) << "the balanced route serves full attention only";
  TVM_FFI_ICHECK(!lse.has_value()) << "the balanced route omits LSE";
  TVM_FFI_ICHECK_EQ(ScalarScale(bmm2_scale, "bmm2_scale"), 1.0);
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
  TVM_FFI_ICHECK_EQ(block_tables.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK(block_tables.IsContiguous());
  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK(multi_ctas_kv_counter_buffer.IsContiguous());
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, multi_ctas_kv_counter_buffer, "multi_ctas_kv_counter_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");

  // One persistent CTA per SM; the device scheduler plans against this count.
  int64_t const num_ctas = sm_count;
  int64_t const max_split_items = MaxSplitItems(num_ctas);
  int64_t const max_split_tiles = MaxSplitTiles(num_ctas);

  // Self-resetting counters: tile counters first, then the 16-byte-aligned
  // queue counters.  The buffer is zero at allocation and every counter the
  // kernel touched reads zero again when it exits.
  int64_t const tile_counter_bytes = max_split_tiles * kCountersPerTile * static_cast<int64_t>(sizeof(uint32_t));
  int64_t const queue_counter_offset = AlignUp(tile_counter_bytes, 16);
  int64_t const counter_bytes = queue_counter_offset + kQueueCounters * static_cast<int64_t>(sizeof(uint32_t));
  int64_t const actual_counter_bytes =
      multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(actual_counter_bytes, counter_bytes)
      << "Cake FMHA balanced decode needs " << counter_bytes
      << " zero-initialized counter bytes for " << num_ctas << " CTAs";
  auto* counters = static_cast<uint8_t*>(multi_ctas_kv_counter_buffer.data_ptr());
  auto* tile_counters = reinterpret_cast<unsigned int*>(counters);
  auto* queue_counters = reinterpret_cast<unsigned int*>(counters + queue_counter_offset);

  // FP32 partial outputs and statistics of split items; uninitialized is fine.
  int64_t const partial_o_bytes = max_split_items * kPartialOPerSlot * static_cast<int64_t>(sizeof(float));
  int64_t const partial_stats_offset = AlignUp(partial_o_bytes, 256);
  int64_t const workspace_bytes =
      partial_stats_offset +
      (max_split_items + 1) * kStatsPerSlot * static_cast<int64_t>(sizeof(float));  // +1: plan-facts slot
  int64_t const actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, workspace_bytes)
      << "Cake FMHA balanced decode workspace requires " << workspace_bytes << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, workspace_bytes);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* partial_o = reinterpret_cast<float*>(workspace);
  auto* partial_stats = reinterpret_cast<float*>(workspace + partial_stats_offset);

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  CUtensorMap h_q = EncodeTmaQuery(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  auto q_slot = TmaDeviceSlot(h_q, query.device().device_id, stream);
  auto k_slot = TmaDeviceSlot(h_k, query.device().device_id, stream);
  auto v_slot = TmaDeviceSlot(h_v, query.device().device_id, stream);
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(q_slot.pointer);
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(k_slot.pointer);
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(v_slot.pointer);

  // The kernel clamps page indices to each request's last page, so the caller's
  // block table is used in place (no padded copy, no metadata kernel).
  int const max_pages_per_seq = static_cast<int>(block_tables.size(1));
  double const softmax_scale = ScalarScale(bmm1_scale, "bmm1_scale");
  unsigned int const max_items = static_cast<unsigned int>(MaxItems(batch_size, num_kv_heads, num_ctas));
  unsigned int const grid_x = static_cast<unsigned int>(num_ctas);

  cudaError_t status = CAKE_FMHA_BALANCED_LAUNCH(
      p_q, p_k, p_v, static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<int*>(block_tables.data_ptr()), static_cast<int*>(seq_lens.data_ptr()),
      partial_o, partial_stats, tile_counters, queue_counters, max_pages_per_seq,
#if CAKE_FMHA_BALANCED_N_ROWS == 0
      static_cast<float>(softmax_scale), static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(kGroup), static_cast<int>(batch_size),
      Q_LEN, max_items,
#else
      static_cast<float>(softmax_scale * 1.4426950408889634), static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(batch_size), Q_LEN, max_items,
#endif
      grid_x, 1u, 1u, stream);
  RecordTmaDeviceSlotUses({q_slot, k_slot, v_slot}, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA balanced BF16 decode launch failed: " << cudaGetErrorString(status);

  (void)lse_stride_tokens;
  (void)lse_stride_heads;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
