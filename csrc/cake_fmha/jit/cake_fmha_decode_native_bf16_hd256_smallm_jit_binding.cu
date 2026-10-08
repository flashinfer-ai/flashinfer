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
#include <cstring>
#include <string>
#include <vector>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

#ifndef Q_LEN
#error "Q_LEN must be supplied by the route-specific JIT"
#endif
#ifndef GROUP
#error "GROUP must be supplied by the route-specific JIT"
#endif
#ifndef Q_BOX_ROWS
#error "Q_BOX_ROWS must be supplied by the route-specific JIT"
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
  // The box always fills the whole packed-row tile: Q_BOX_ROWS = rows / GROUP tokens.
  uint32_t box_dim[4] = {64u, static_cast<uint32_t>(GROUP), static_cast<uint32_t>(Q_BOX_ROWS), 2u};
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
  // HND [pages, num_kv_heads, PAGE_SIZE, 256] viewed as (64, page tokens, 4 k-groups, heads,
  // pages).
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
  static_assert(Q_LEN * GROUP <= kNumRows && Q_LEN * GROUP > kNumRows / 2,
                "Q_LEN * GROUP must fit the smallest 32/64-row tile (rows above it are padding)");
  static_assert(kNumRows % GROUP == 0 && Q_BOX_ROWS * GROUP == kNumRows,
                "Q_BOX_ROWS * GROUP must cover the whole packed-row tile");
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
  // The kernel takes the descriptors as __grid_constant__ parameters, so they are
  // passed by value (and captured by value in CUDA graphs): no device slots, no
  // prewarm, and query/out need no stable addresses across capture and replay.
  CakeFmhaTensorMap tm_q, tm_k, tm_v;
  static_assert(sizeof(CakeFmhaTensorMap) == sizeof(CUtensorMap));
  std::memcpy(&tm_q, &h_q, sizeof(tm_q));
  std::memcpy(&tm_k, &h_k, sizeof(tm_k));
  std::memcpy(&tm_v, &h_v, sizeof(tm_v));
  auto const* p_q = &tm_q;
  auto const* p_k = &tm_k;
  auto const* p_v = &tm_v;

  // Workspace: BF16 partial O slots, FP32 partial LSE slots, two u32 counters per
  // tile (zeroed before every launch; the kernel resets them as well), and the
  // LSE destination when the caller does not provide one.
  int64_t slots = tiles * NUM_SPLIT;
  int64_t partial_o_offset = 0;
  int64_t cursor =
      AlignUp(slots * kNumRows * kHeadDim * static_cast<int64_t>(sizeof(__nv_bfloat16)), 256);
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
  // Merge tickets: the kernel's last CTA resets both counters, so the caller's
  // persistent zero-initialized multi_ctas_kv_counter_buffer (same contract as
  // trtllm-gen) needs no per-launch memset; a memset graph node adds ~15us of
  // launch latency per layer in CUDA-graph decode. Fall back to the workspace.
  uint32_t* counters = nullptr;
  int64_t counter_bytes = tiles * 2 * static_cast<int64_t>(sizeof(uint32_t));
  if (multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer) >=
      counter_bytes) {
    CheckSameDevice(query, multi_ctas_kv_counter_buffer, "multi_ctas_kv_counter_buffer");
    TVM_FFI_ICHECK(multi_ctas_kv_counter_buffer.IsContiguous());
    counters = static_cast<uint32_t*>(multi_ctas_kv_counter_buffer.data_ptr());
  } else {
    counters = reinterpret_cast<uint32_t*>(workspace + counters_offset);
    TVM_FFI_ICHECK_EQ(cudaMemsetAsync(counters, 0, counter_bytes, stream), cudaSuccess)
        << "failed to reset Cake FMHA small-M merge counters";
  }

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
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA small-M hd256 decode launch failed: " << cudaGetErrorString(status);

  (void)enable_pdl;
  (void)max_kv_len;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
