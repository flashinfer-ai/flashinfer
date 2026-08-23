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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

#ifndef CAKE_FMHA_HAS_SINK
#error "CAKE_FMHA_HAS_SINK must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_HAS_WINDOW
#error "CAKE_FMHA_HAS_WINDOW must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_USE_SCALE_PTR
#error "CAKE_FMHA_USE_SCALE_PTR must be supplied by the route-specific JIT"
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

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  size_t used = 0;
};

// Tensor-map pointers are part of the kernel ABI. Keep immutable, context-local
// copies so warm bindings remain CUDA-Graph-capture safe.
void* TmaDeviceSlot(const CUtensorMap& tm, int device_id, cudaStream_t stream) {
  static std::mutex mu;
  static auto* slots = new std::unordered_map<std::string, void*>();
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();

  CUcontext current_context = nullptr;
  CUresult result = cuCtxGetCurrent(&current_context);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_context != nullptr)
      << "Cake FMHA TMA launch requires an active CUDA context";
  CUdevice current_device = -1;
  result = cuCtxGetDevice(&current_device);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_device == device_id)
      << "Cake FMHA TMA descriptor device mismatch";

  std::string key = std::to_string(reinterpret_cast<uintptr_t>(current_context));
  key.push_back(':');
  key.append(reinterpret_cast<const char*>(&tm), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(mu);
  auto it = slots->find(key);
  if (it != slots->end()) return it->second;

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);
  TVM_FFI_ICHECK_EQ(capture_status, CU_STREAM_CAPTURE_STATUS_NONE)
      << "prewarm each Cake FMHA tensor/layout binding before CUDA Graph capture";

  TmaDeviceArena& arena = (*arenas)[current_context];
  TVM_FFI_ICHECK_LT(arena.used, TmaDeviceArena::kMaxSlots)
      << "Cake FMHA immutable TMA descriptor arena is exhausted";
  if (arena.used % TmaDeviceArena::kSlotsPerChunk == 0) {
    CUdeviceptr chunk = 0;
    result = cuMemAlloc(&chunk, TmaDeviceArena::kSlotsPerChunk * sizeof(CUtensorMap));
    TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);
    arena.chunks.push_back(chunk);
  }
  size_t chunk_index = arena.used / TmaDeviceArena::kSlotsPerChunk;
  size_t slot_index = arena.used % TmaDeviceArena::kSlotsPerChunk;
  CUdeviceptr dev = arena.chunks[chunk_index] + slot_index * sizeof(CUtensorMap);
  result = cuMemcpyHtoD(dev, &tm, sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);
  ++arena.used;
  void* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(dev));
  (*slots)[key] = pointer;
  return pointer;
}

CUtensorMap EncodeTmaQt(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(2), 128);
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

CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(tensor.size(2), 16);
  TVM_FFI_ICHECK_EQ(tensor.size(3), 128);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(2), 0);
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

__global__ void PrepareDecodeMetadata(const int* seq_lens, int* causal_prefix,
                                      const int* page_table, int* padded_page_table, int batch_size,
                                      int source_pages, int padded_pages) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < batch_size) {
    causal_prefix[index] = max(seq_lens[index] - Q_LEN, 0);
  }
  if (padded_page_table != nullptr && index < batch_size * padded_pages) {
    int column = index % padded_pages;
    padded_page_table[index] =
        column < source_pages ? page_table[index / padded_pages * source_pages + column] : 0;
  }
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
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
  TVM_FFI_ICHECK_EQ(query.size(2), 128);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(batch_size, BATCH_SIZE);
  TVM_FFI_ICHECK_EQ(max_q_len, Q_LEN);
  TVM_FFI_ICHECK_EQ(query.size(0), BATCH_SIZE * Q_LEN);
  TVM_FFI_ICHECK_EQ(query.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(key_cache.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), NUM_KV_HEADS);
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
  TVM_FFI_ICHECK_EQ(CAKE_FMHA_HAS_SINK, attention_sinks.has_value() ? 1 : 0);
  TVM_FFI_ICHECK_EQ(CAKE_FMHA_HAS_WINDOW, window_left >= 0 ? 1 : 0);

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  if (attention_sinks.has_value()) {
    CheckSameDevice(query, attention_sinks.value(), "attention_sinks");
  }
  if (lse.has_value()) CheckSameDevice(query, lse.value(), "lse");

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  CUtensorMap h_q = EncodeTmaQt(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_q, query.device().device_id, stream));
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_k, query.device().device_id, stream));
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_v, query.device().device_id, stream));

  int64_t even_kv_blocks = (max_kv_len + 127) / 128;
  even_kv_blocks += even_kv_blocks % 2;
  even_kv_blocks = std::max<int64_t>(4, even_kv_blocks);
  int64_t required_pages = even_kv_blocks * 8;
  int64_t source_pages = block_tables.size(1);
  int64_t prefix_offset = 0;
  int64_t cursor = AlignUp(batch_size * static_cast<int64_t>(sizeof(int)), 16);
  bool needs_page_padding = source_pages < required_pages;
  int64_t page_offset = cursor;
  if (needs_page_padding) {
    cursor += batch_size * required_pages * static_cast<int64_t>(sizeof(int));
    cursor = AlignUp(cursor, 16);
  }
  int64_t lse_offset = cursor;
  if (!lse.has_value()) {
    cursor += query.size(0) * query.size(1) * static_cast<int64_t>(sizeof(float));
  }
  int64_t actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA decode-native BF16 workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* causal_prefix = reinterpret_cast<int*>(workspace + prefix_offset);
  auto* padded_page_table =
      needs_page_padding ? reinterpret_cast<int*>(workspace + page_offset) : nullptr;
  int64_t metadata_items =
      std::max<int64_t>(batch_size, needs_page_padding ? batch_size * required_pages : 0);
  int const threads = 256;
  int const blocks = static_cast<int>((metadata_items + threads - 1) / threads);
  PrepareDecodeMetadata<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(seq_lens.data_ptr()), causal_prefix,
      static_cast<const int*>(block_tables.data_ptr()), padded_page_table,
      static_cast<int>(batch_size), static_cast<int>(source_pages),
      static_cast<int>(required_pages));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "failed to prepare Cake FMHA decode metadata";

  int* page_table =
      needs_page_padding ? padded_page_table : static_cast<int*>(block_tables.data_ptr());
  int max_pages_per_seq = static_cast<int>(needs_page_padding ? required_pages : source_pages);
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

  float* scale_ptr = reinterpret_cast<float*>(seq_lens.data_ptr());
  float softmax_scale_log2 = 0.0f;
#if CAKE_FMHA_USE_SCALE_PTR
  auto scale_tensor = bmm1_scale.as<ffi::Tensor>();
  TVM_FFI_ICHECK(scale_tensor.has_value())
      << "decode-native BF16 scale-pointer specialization requires a tensor";
  auto const& scale = scale_tensor.value();
  TVM_FFI_ICHECK_EQ(scale.dtype(), dl_float32);
  TVM_FFI_ICHECK_EQ(scale.numel(), 1);
  TVM_FFI_ICHECK_EQ(scale.device().device_type, query.device().device_type);
  TVM_FFI_ICHECK_EQ(scale.device().device_id, query.device().device_id);
  scale_ptr = static_cast<float*>(scale.data_ptr());
#else
  softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") * 1.4426950408889634);
#endif
  float* sinks_ptr = scale_ptr;
  if (attention_sinks.has_value()) {
    auto const& sinks = attention_sinks.value();
    TVM_FFI_ICHECK_EQ(sinks.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(sinks.numel(), NUM_Q_HEADS);
    TVM_FFI_ICHECK(sinks.IsContiguous());
    sinks_ptr = static_cast<float*>(sinks.data_ptr());
  }

  unsigned int grid_x;
  unsigned int grid_y = 1;
  unsigned int grid_z = 1;
  // The exact sink kernel maps one CTA per (query, KV head, batch) tile.
  // Other native-BF16 members use their existing persistent 1-D launch.
#if BATCH_SIZE == 256 && Q_LEN == 1 && NUM_Q_HEADS == 32 && NUM_KV_HEADS == 4 && \
    CAKE_FMHA_HAS_SINK == 1 && CAKE_FMHA_HAS_WINDOW == 0 && CAKE_FMHA_USE_SCALE_PTR == 0
  grid_x = Q_LEN;
  grid_y = NUM_KV_HEADS;
  grid_z = BATCH_SIZE;
#else
  unsigned int total_tiles = BATCH_SIZE * Q_LEN * NUM_KV_HEADS;
  grid_x = std::min<unsigned int>(static_cast<unsigned int>(sm_count), total_tiles);
#endif
  cudaError_t status = cake_fmha_launch_decode_native_bf16(
      p_q, p_k, p_v, static_cast<__nv_bfloat16*>(out.data_ptr()), lse_ptr, page_table,
      causal_prefix, scale_ptr, sinks_ptr, max_pages_per_seq, static_cast<int>(max_kv_len),
      softmax_scale_log2, static_cast<int>(window_left), NUM_Q_HEADS, NUM_KV_HEADS, BATCH_SIZE,
      grid_x, grid_y, grid_z, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA decode-native BF16 launch failed: " << cudaGetErrorString(status);

  (void)multi_ctas_kv_counter_buffer;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
