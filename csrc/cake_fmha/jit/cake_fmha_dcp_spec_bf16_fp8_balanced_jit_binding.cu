/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// On-device load-balanced BF16-Q / E4M3-KV page-64 DCP speculative decode
// (Cake add-on family ``dcp_spec_bf16_fp8_balanced``, programs ``n32`` / ``n64``;
// the JIT names the exported launch binding as ``CAKE_FMHA_DCP_BALANCED_LAUNCH``).
//
// The BF16 balanced DCP engine on the round-2 E4M3 operand path: E4M3 page-64
// K/V rings, the BF16 query staged by TMA and split into E4M3 hi / lo images
// on the device, E4M3 P in TMEM.  One persistent CTA per SM plans the split-KV
// schedule from the requests' global prefixes, rank and world, so batch,
// heads, lengths, rank and world are runtime kernel arguments, one module per
// packed-row tile (32 live rows for q_len 3..4, 64 for 5..8) serves every
// shape, and a prepared launch replays under CUDA Graph capture for any
// length vector.  Scales are host scalars fixed at launch:
// ``softmax_scale_log2 = bmm1_scale * log2(e)`` with ``bmm1_scale = sm_scale *
// k_scale``, and ``output_scale = bmm2_scale = v_scale``, applied after the
// normalisation (the static route's order).  The kernel writes the rank-local
// BF16 output and the FP32 base-2 LSE (``O = 0`` / ``LSE = -inf`` for rows
// without a visible local key).
//
// The self-resetting split counters live in the zero-initialized
// ``multi_ctas_kv_counter_buffer``; FP32 partial outputs (64 x 128 per split
// item) and statistics (max[64], sum[64]) are carved from the caller-owned
// ``workspace_buffer``, which may be uninitialized.  K/V arrive as the
// ``uint8`` views of the E4M3 caches.

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/variant.h>

#include <cstdint>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

#ifndef CAKE_FMHA_DCP_BALANCED_LAUNCH
#error "CAKE_FMHA_DCP_BALANCED_LAUNCH must name the exported launch binding"
#endif

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

constexpr DLDataType kQueryDtype = dl_bfloat16;
constexpr DLDataType kKvDtype = dl_uint8;  // E4M3 cache viewed as bytes
constexpr CUtensorMapDataType kKvTmaDtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
constexpr char const* kRouteName = "BF16-Q / E4M3-KV page-64";
constexpr int64_t kKvElementBytes = 1;
constexpr int64_t kKvInner = 128;          // E4M3 elements per 128-byte swizzle atom
constexpr int64_t kKvBoxTokens = 64;       // tokens per K/V box (one page)

constexpr int64_t kGroup = 8;              // query heads per KV head (ForGen GQA-8 layout)
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 64;
constexpr int64_t kMaxQTiles = 1;          // one row tile per request (q_len 1..8)
constexpr int64_t kQBoxTokens = 8;         // speculative rows per Q box (the full M64 tile)
constexpr int64_t kMaxNRows = 64;          // physical packed tile (rows = speculative rows x group)
constexpr int64_t kQBoxRows = kMaxNRows / kGroup;      // speculative rows per row tile
constexpr int64_t kMaxQLen = kMaxQTiles * kQBoxRows;   // 8 on every family
constexpr int64_t kQBoxKGroups = kHeadDim / 64;        // 64-element column groups of one Q row
constexpr int64_t kMaxRequests = 1024;     // MAX_REQUEST_GROUPS * REQUEST_GROUP
constexpr int64_t kMaxBalanceFactor = 8;   // chunk length >= total work / (k * CTAs)
constexpr int64_t kQueueCounters = 4;      // ticket, done CTAs; [2]-[3] unused, read zero
constexpr int64_t kPartialOPerSlot = kMaxNRows * kHeadDim;  // FP32 O[64, kHeadDim] per split item
constexpr int64_t kStatsPerSlot = 2 * kMaxNRows;            // max[64] then sum[64]
constexpr int64_t kCountersPerTile = 4;    // arrivals, two reduce-queue words, published flag
constexpr int64_t kReduceSlicesMax = 8;    // reduce tickets per split tile at most

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}

// Query [batch * q_len, Hq, kHeadDim] read in place as a 4-D map (64 dims,
// head, token, k-group) whose box (64, kGroup, kQBoxTokens, kQBoxKGroups) is one
// packed row tile: SMEM row r = token * kGroup + head.  Tokens past the tensor
// end are TMA zero fill (padding rows whose results are never stored), so the
// token box may exceed the token extent of a small batch.
CUtensorMap EncodeTmaQuery(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), kQueryDtype);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), kHeadDim);
  TVM_FFI_ICHECK_GE(tensor.size(1), kGroup);
  uint64_t global_dim[4] = {64u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0)),
                            static_cast<uint64_t>(kQBoxKGroups)};
  uint64_t global_strides[3] = {static_cast<uint64_t>(tensor.stride(1) * 2),
                                static_cast<uint64_t>(tensor.stride(0) * 2), 128u};
  uint32_t box_dim[4] = {64u, static_cast<uint32_t>(kGroup), static_cast<uint32_t>(kQBoxTokens),
                         static_cast<uint32_t>(kQBoxKGroups)};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA packed query tensor map";
  return tm;
}

// K/V pools [pages, Hkv, kPageSize, kHeadDim] as a 5-D map (kKvInner elements,
// token, k-group, head, page) whose token/head/page steps are the source's
// physical strides, so the production stacked ``kv_cache[:, side]`` views load
// in place; kKvBoxTokens tokens per box.
CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(tensor.size(2), kPageSize);
  TVM_FFI_ICHECK_EQ(tensor.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), kHeadDim);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(1) * kKvElementBytes % 16, 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(0) * kKvElementBytes % 16, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16, 0);
  uint64_t global_dim[5] = {static_cast<uint64_t>(kKvInner), static_cast<uint64_t>(kPageSize),
                            static_cast<uint64_t>(kHeadDim / kKvInner),
                            static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2) * kKvElementBytes), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * kKvElementBytes),
                                static_cast<uint64_t>(tensor.stride(0) * kKvElementBytes)};
  uint32_t box_dim[5] = {static_cast<uint32_t>(kKvInner), static_cast<uint32_t>(kKvBoxTokens), 1u,
                         1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, kKvTmaDtype, 5, tensor.data_ptr(), global_dim, global_strides,
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

// Row tiles per request: ceil(q_len / kQBoxRows) (one on the D128 families).
int64_t QTiles(int64_t q_len) { return (q_len + kQBoxRows - 1) / kQBoxRows; }

// Host ticket-loop bound: whole row tiles, every possible split chunk and the
// most reduce tickets per split tile (``max_items_bound`` of the Cake modules).
int64_t MaxItems(int64_t batch_size, int64_t q_len, int64_t num_kv_heads, int64_t num_ctas) {
  return batch_size * num_kv_heads * QTiles(q_len) + MaxSplitItems(num_ctas) +
         MaxSplitTiles(num_ctas) * kReduceSlicesMax;
}

// The kernel takes W = 2 ** cp_world_log2; the production contract admits 1, 2, 4 and 8.
int CpWorldLog2(int64_t cp_world) {
  TVM_FFI_ICHECK(cp_world == 1 || cp_world == 2 || cp_world == 4 || cp_world == 8)
      << "cp_world must be 1, 2, 4 or 8, got " << cp_world;
  int log2 = 0;
  while ((int64_t{1} << log2) < cp_world) ++log2;
  return log2;
}

}  // namespace

void Run(TensorView query, TensorView key_cache, TensorView value_cache, TensorView out,
         TensorView lse, TensorView block_tables, TensorView causal_seqlens_kv_global,
         TensorView workspace_buffer, TensorView counter_buffer, double softmax_scale_log2,
         double output_scale,
         int64_t cp_rank, int64_t cp_world, int64_t num_q_heads, int64_t num_kv_heads,
         int64_t batch_size, int64_t q_len, int64_t num_ctas) {
  TVM_FFI_ICHECK_EQ(query.dtype(), kQueryDtype);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(out.dtype(), kQueryDtype);
  TVM_FFI_ICHECK_EQ(lse.dtype(), dl_float32);
  TVM_FFI_ICHECK(query.IsContiguous());
  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(query.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK(q_len >= 1 && q_len <= kMaxQLen)
      << "the balanced " << kRouteName << " DCP route serves q_len 1.." << kMaxQLen;
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_LE(batch_size, kMaxRequests)
      << "the balanced scheduler plans at most " << kMaxRequests << " requests per launch";
  TVM_FFI_ICHECK_EQ(query.size(0), batch_size * q_len);
  TVM_FFI_ICHECK_EQ(query.size(1), num_q_heads);
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_EQ(num_q_heads, kGroup * num_kv_heads)
      << "the balanced " << kRouteName << " DCP route serves exactly " << kGroup
      << " query heads per KV head";
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_cache.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), value_cache.size(2));
  TVM_FFI_ICHECK_EQ(key_cache.size(3), value_cache.size(3));
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(lse.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(lse.size(1), num_q_heads);
  TVM_FFI_ICHECK(lse.IsContiguous());
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
  TVM_FFI_ICHECK_EQ(block_tables.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK_GT(block_tables.size(1), 0);
  TVM_FFI_ICHECK(block_tables.IsContiguous());
  TVM_FFI_ICHECK_EQ(causal_seqlens_kv_global.ndim(), 1);
  TVM_FFI_ICHECK_EQ(causal_seqlens_kv_global.dtype(), dl_int32);
  TVM_FFI_ICHECK_EQ(causal_seqlens_kv_global.size(0), batch_size);
  TVM_FFI_ICHECK(causal_seqlens_kv_global.IsContiguous());
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK(counter_buffer.IsContiguous());
  TVM_FFI_ICHECK(cp_rank >= 0 && cp_rank < cp_world)
      << "cp_rank must be in [0, cp_world), got " << cp_rank << " of " << cp_world;
  int const cp_world_log2 = CpWorldLog2(cp_world);
  TVM_FFI_ICHECK_GT(num_ctas, 0);
  TVM_FFI_ICHECK(softmax_scale_log2 > 0.0 && output_scale > 0.0)
      << "the balanced " << kRouteName
      << " DCP route requires positive host softmax (bmm1 * log2 e) and output (bmm2) scales";
  float const softmax_scale_log2_f = static_cast<float>(softmax_scale_log2);
  float const output_scale_f = static_cast<float>(output_scale);

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, lse, "lse");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, causal_seqlens_kv_global, "causal_seqlens_kv_global");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, counter_buffer, "multi_ctas_kv_counter_buffer");

  // One persistent CTA per SM; the device scheduler plans against this count.
  int64_t const max_split_items = MaxSplitItems(num_ctas);
  int64_t const max_split_tiles = MaxSplitTiles(num_ctas);

  // Self-resetting counters: four words per split tile first, then the
  // 16-byte-aligned queue counters.  The buffer is zero at allocation and every
  // counter the kernel touched reads zero again when it exits.
  int64_t const tile_counter_bytes =
      max_split_tiles * kCountersPerTile * static_cast<int64_t>(sizeof(uint32_t));
  int64_t const queue_counter_offset = AlignUp(tile_counter_bytes, 16);
  int64_t const counter_bytes =
      queue_counter_offset + kQueueCounters * static_cast<int64_t>(sizeof(uint32_t));
  int64_t const actual_counter_bytes = counter_buffer.numel() * get_element_size(counter_buffer);
  TVM_FFI_ICHECK_GE(actual_counter_bytes, counter_bytes)
      << "Cake FMHA balanced " << kRouteName << " DCP decode needs " << counter_bytes
      << " zero-initialized counter bytes for " << num_ctas << " CTAs";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(counter_buffer.data_ptr()) % 16, 0);
  auto* counters = static_cast<uint8_t*>(counter_buffer.data_ptr());
  auto* tile_counters = reinterpret_cast<unsigned int*>(counters);
  auto* queue_counters = reinterpret_cast<unsigned int*>(counters + queue_counter_offset);

  // FP32 partial outputs and statistics of split items (plus the plan-facts
  // slot); uninitialized is fine.
  int64_t const partial_o_bytes =
      max_split_items * kPartialOPerSlot * static_cast<int64_t>(sizeof(float));
  int64_t const partial_stats_offset = AlignUp(partial_o_bytes, 256);
  int64_t const workspace_bytes =
      partial_stats_offset + (max_split_items + 1) * kStatsPerSlot * static_cast<int64_t>(sizeof(float));
  int64_t const actual_workspace_bytes =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, workspace_bytes)
      << "Cake FMHA balanced " << kRouteName << " DCP decode workspace requires "
      << workspace_bytes << " bytes";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace_buffer.data_ptr()) % 256, 0);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* partial_o = reinterpret_cast<float*>(workspace);
  auto* partial_stats = reinterpret_cast<float*>(workspace + partial_stats_offset);

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  CUtensorMap h_q = EncodeTmaQuery(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  CUtensorMap const& p_q = h_q;
  CUtensorMap const& p_k = h_k;
  CUtensorMap const& p_v = h_v;

  // The kernel derives every request's rank-local length from its global prefix,
  // rank and world and clamps page indices to the request's pages, so the
  // caller's block table and prefixes are used in place (no metadata kernel).
  int const max_pages_per_seq = static_cast<int>(block_tables.size(1));
  unsigned int const max_items =
      static_cast<unsigned int>(MaxItems(batch_size, q_len, num_kv_heads, num_ctas));
  unsigned int const grid_x = static_cast<unsigned int>(num_ctas);

  cudaError_t status = CAKE_FMHA_DCP_BALANCED_LAUNCH(
      p_q, p_k, p_v, static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()), static_cast<int*>(block_tables.data_ptr()),
      static_cast<int*>(causal_seqlens_kv_global.data_ptr()), partial_o, partial_stats,
      tile_counters, queue_counters, max_pages_per_seq,
      softmax_scale_log2_f, output_scale_f,
      static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(batch_size), static_cast<int>(q_len), static_cast<int>(cp_rank),
      cp_world_log2, max_items, grid_x, 1u, 1u, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA balanced " << kRouteName << " DCP decode launch failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_fmha::Run);
