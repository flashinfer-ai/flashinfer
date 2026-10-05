/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

// On-device load-balanced FP8-KV paged decode (Cake routes ``decode_balanced_fp8_v1``,
// ``decode_balanced_bf16q_v1`` and ``decode_balanced_fp16q_v1``;
// ``CAKE_FMHA_BALANCED_Q_DTYPE`` selects the query / output element type).
//
// The trtllm-gen ``Q8Kv128 SwapsAb`` FP8 ForGen decode body (E4M3 K/V pages, P
// scaled into E4M3, FP32 statistics) driven by the same scheduler warp,
// work-token ring and last-CTA merge as the BF16 balanced kernel: one
// persistent CTA per SM plans the split-KV schedule on the device from the
// ``seq_lens`` buffer, so batch, heads and KV lengths are runtime kernel
// arguments, one module per query dtype serves every shape, and a prepared
// launch replays under CUDA Graph capture for any length vector.
//
// Three generated programs share this adapter:
//   * ``CAKE_FMHA_BALANCED_Q_DTYPE == 0`` -- E4M3 query and output.  The query
//     [batch, Hq, 128] is one dense byte matrix read through a u8 TMA map whose
//     box (128 bytes, 8 rows) is one GQA-8 group.
//   * ``CAKE_FMHA_BALANCED_Q_DTYPE == 1 | 2`` -- BF16 / FP16 query and output over
//     the E4M3 cache.  The loader warp reads the query rows in place as 32-bit
//     words (two elements per word, 64 words per row) and splits every element
//     into an E4M3 hi/lo pair, so ``Qt`` is a plain word pointer.
//
// Scales are host scalars baked into the launch (a device scale would need a
// per-launch host read that breaks graph-replay safety): trtllm-gen's
// ``scaleSoftmaxLog2 = bmm1_scale * log2(e)`` with ``bmm1_scale = q_scale *
// k_scale / sqrt(128)``, and ``ptrOutputScale = bmm2_scale = v_scale / o_scale``,
// applied by the kernel after the normalisation.  Only ``q_len == 1`` rows exist
// (no packed MTP tile in FP8).
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

#include <cstdint>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

#ifndef CAKE_FMHA_BALANCED_Q_DTYPE
#error "CAKE_FMHA_BALANCED_Q_DTYPE must be supplied by the route-specific JIT (0 = E4M3, 1 = BF16, 2 = FP16 query)"
#endif
#ifndef CAKE_FMHA_BALANCED_LAUNCH
#error "CAKE_FMHA_BALANCED_LAUNCH must name the exported launch binding"
#endif
#if CAKE_FMHA_BALANCED_Q_DTYPE < 0 || CAKE_FMHA_BALANCED_Q_DTYPE > 2
#error "CAKE_FMHA_BALANCED_Q_DTYPE must be 0, 1 or 2"
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

// Query / output element type of this instance; the K/V cache is always E4M3.
#if CAKE_FMHA_BALANCED_Q_DTYPE == 0
using Output = uint8_t;  // E4M3 bytes
constexpr DLDataType kQueryDtype = dl_float8_e4m3fn;
constexpr char const* kRouteName = "FP8";
#elif CAKE_FMHA_BALANCED_Q_DTYPE == 1
using Output = __nv_bfloat16;
constexpr DLDataType kQueryDtype = dl_bfloat16;
constexpr char const* kRouteName = "BF16-Q / FP8-KV";
#else
using Output = __half;
constexpr DLDataType kQueryDtype = dl_float16;
constexpr char const* kRouteName = "FP16-Q / FP8-KV";
#endif
constexpr DLDataType kKvDtype = dl_float8_e4m3fn;
constexpr CUtensorMapDataType kKvTmaDtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
constexpr int64_t kElementBytes = 1;       // E4M3 K/V element
constexpr int64_t kKvInner = 128;          // E4M3 elements per 128-byte swizzle atom

constexpr int64_t kGroup = 8;              // query heads per KV head (ForGen GQA-8 layout)
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 16;
constexpr int64_t kMaxRequests = 1024;     // MAX_REQUEST_GROUPS * REQUEST_GROUP
constexpr int64_t kMaxBalanceFactor = 8;   // chunk length >= total work / (k * CTAs)
constexpr int64_t kQueueCounters = 4;      // ticket, done CTAs, L, total_items
constexpr int64_t kPartialOPerSlot = 8 * kHeadDim;   // FP32 O[8, 128] per split item
constexpr int64_t kStatsPerSlot = 16;                // m[8] then l[8]
constexpr int64_t kCountersPerTile = 1;              // chunk arrivals

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

#if CAKE_FMHA_BALANCED_Q_DTYPE == 0
// E4M3 query [batch, Hq, 128] as the byte matrix [batch * Hq, 128]: a 3-D map
// (128 bytes, row, one k-group) whose box of eight consecutive rows is one
// GQA-8 group.
CUtensorMap EncodeTmaQuery(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), kQueryDtype);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), kHeadDim);
  uint64_t global_dim[3] = {128u, static_cast<uint64_t>(tensor.size(0) * tensor.size(1)), 1u};
  uint64_t global_strides[2] = {128u, 128u};
  uint32_t box_dim[3] = {128u, 8u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, tensor.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA query tensor map";
  return tm;
}
#endif

// K/V pools [pages, Hkv, kPageSize, kHeadDim] as a 5-D map (kKvInner elements,
// token, k-group, head, page) whose token/head/page steps are the source's
// physical strides, so the production stacked ``kv_cache[:, side]`` views load
// in place.  Boxes are 16 tokens regardless of the page size.
CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(tensor.size(2), kPageSize);
  TVM_FFI_ICHECK_EQ(tensor.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), kHeadDim);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(1) * kElementBytes % 16, 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(0) * kElementBytes % 16, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16, 0);
  uint64_t global_dim[5] = {static_cast<uint64_t>(kKvInner), static_cast<uint64_t>(kPageSize),
                            static_cast<uint64_t>(kHeadDim / kKvInner),
                            static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2) * kElementBytes), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * kElementBytes),
                                static_cast<uint64_t>(tensor.stride(0) * kElementBytes)};
  uint32_t box_dim[5] = {static_cast<uint32_t>(kKvInner), 16u, 1u, 1u, 1u};
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

// Host ticket-loop bound: whole (request row, KV head) tiles plus every possible
// split chunk (``max_items_bound`` of the Cake modules).
int64_t MaxItems(int64_t batch_size, int64_t q_len, int64_t num_kv_heads, int64_t num_ctas) {
  return batch_size * q_len * num_kv_heads + MaxSplitItems(num_ctas);
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
  TVM_FFI_ICHECK_EQ(query.dtype(), kQueryDtype);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), kKvDtype);
  TVM_FFI_ICHECK_EQ(out.dtype(), kQueryDtype);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK(query.IsContiguous());
  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(query.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(max_q_len, 1) << "the balanced " << kRouteName << " decode route serves q_len == 1";
  int64_t const q_len = 1;
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_LE(batch_size, kMaxRequests)
      << "the balanced scheduler plans at most " << kMaxRequests << " requests per launch";
  TVM_FFI_ICHECK_EQ(query.size(0), batch_size * q_len);
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  int64_t const num_q_heads = query.size(1);
  int64_t const num_kv_heads = key_cache.size(1);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), num_kv_heads);
  TVM_FFI_ICHECK_EQ(num_q_heads, kGroup * num_kv_heads)
      << "the balanced " << kRouteName << " decode route serves exactly eight query heads per KV head";
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
  double const bmm1 = ScalarScale(bmm1_scale, "bmm1_scale");
  double const bmm2 = ScalarScale(bmm2_scale, "bmm2_scale");
  TVM_FFI_ICHECK(bmm1 > 0.0 && bmm2 > 0.0)
      << "the balanced " << kRouteName << " decode route requires positive host softmax and output scales";
  float const softmax_scale_log2 = static_cast<float>(bmm1 * 1.4426950408889634);
  float const output_scale = static_cast<float>(bmm2);
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
  TVM_FFI_ICHECK_GE(max_kv_len, q_len);
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

  // Self-resetting counters: one word per split tile first, then the
  // 16-byte-aligned queue counters.  The buffer is zero at allocation and every
  // counter the kernel touched reads zero again when it exits.
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
#if CAKE_FMHA_BALANCED_Q_DTYPE == 0
  CUtensorMap h_q = EncodeTmaQuery(query);
  CUtensorMap const& p_q = h_q;
#else
  // BF16 / FP16 query rows read in place as 32-bit words (two elements per word).
  auto* p_q = reinterpret_cast<uint32_t*>(query.data_ptr());
#endif
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  CUtensorMap const& p_k = h_k;
  CUtensorMap const& p_v = h_v;

  // The kernel clamps page indices to each request's last page, so the caller's
  // block table is used in place (no padded copy, no metadata kernel).
  int const max_pages_per_seq = static_cast<int>(block_tables.size(1));
  unsigned int const max_items =
      static_cast<unsigned int>(MaxItems(batch_size, q_len, num_kv_heads, num_ctas));
  unsigned int const grid_x = static_cast<unsigned int>(num_ctas);

  cudaError_t status = CAKE_FMHA_BALANCED_LAUNCH(
      p_q, p_k, p_v, static_cast<Output*>(out.data_ptr()),
      static_cast<int*>(block_tables.data_ptr()), static_cast<int*>(seq_lens.data_ptr()),
      partial_o, partial_stats, tile_counters, queue_counters, max_pages_per_seq,
      softmax_scale_log2, output_scale, static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(kGroup), static_cast<int>(batch_size),
      static_cast<int>(q_len), max_items,
      grid_x, 1u, 1u, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA balanced " << kRouteName << " decode launch failed: " << cudaGetErrorString(status);

  (void)lse_stride_tokens;
  (void)lse_stride_heads;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
