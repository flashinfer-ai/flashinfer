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
#ifndef CAKE_FMHA_CONTEXT_NVFP4
#define CAKE_FMHA_CONTEXT_NVFP4 0
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
// copies so warmed bindings remain CUDA-Graph-capture safe.
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

CUtensorMap EncodeTmaQ(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(tensor.size(2), 128);
  TVM_FFI_ICHECK_EQ(tensor.stride(2), 1);
  uint64_t global_dim[5] = {128u, PACK_G, static_cast<uint64_t>(tensor.size(0)), 1u,
                            NUM_Q_HEADS / PACK_G};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(1)),
                                static_cast<uint64_t>(tensor.stride(0)), 128u,
                                static_cast<uint64_t>(tensor.stride(1) * PACK_G)};
  uint32_t box_dim[5] = {128u, PACK_G, TOK_PER_STAGE, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, tensor.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS) << "failed to encode Cake FMHA context query map";
  return tm;
}

CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4) << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS / HEADS_PER_GROUP);
  TVM_FFI_ICHECK_EQ(tensor.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(3), 128);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  uint64_t global_dim[5] = {128u, PAGE_SIZE, 1u, static_cast<uint64_t>(tensor.size(1)),
                            static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {static_cast<uint64_t>(tensor.stride(2)), 128u,
                                static_cast<uint64_t>(tensor.stride(1)),
                                static_cast<uint64_t>(tensor.stride(0))};
  uint32_t box_dim[5] = {128u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, tensor.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA context " << name << " map";
  return tm;
}

CUtensorMap EncodeTmaContiguousPagedKv(void* data, int64_t pages, int64_t page_stride,
                                       const char* name) {
  uint64_t global_dim[5] = {128u, PAGE_SIZE, 1u, NUM_Q_HEADS / HEADS_PER_GROUP,
                            static_cast<uint64_t>(pages)};
  uint64_t global_strides[4] = {128u, 128u, PAGE_SIZE * 128u,
                                static_cast<uint64_t>(page_stride)};
  uint32_t box_dim[5] = {128u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, data, global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA dequantized context " << name << " map";
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
    bool is_causal, Optional<TensorView> lse, int64_t lse_stride_tokens, int64_t lse_stride_heads) {
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(),
#if CAKE_FMHA_CONTEXT_NVFP4
                    dl_uint8
#else
                    dl_float8_e4m3fn
#endif
  );
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), key_cache.dtype());
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
#if CAKE_FMHA_CONTEXT_NVFP4
  TVM_FFI_ICHECK(key_block_scales.has_value() && value_block_scales.has_value());
#else
  TVM_FFI_ICHECK(!key_block_scales.has_value() && !value_block_scales.has_value());
#endif
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK_EQ(window_left, -1);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  float output_scale = static_cast<float>(ScalarScale(bmm2_scale, "bmm2_scale"));
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
#if CAKE_FMHA_CONTEXT_NVFP4
  TVM_FFI_ICHECK_EQ(PAGE_SIZE, 16);
  TVM_FFI_ICHECK_EQ(key_cache.size(3), 64);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), 64);
  TVM_FFI_ICHECK(key_cache.IsContiguous());
  TVM_FFI_ICHECK(value_cache.IsContiguous());
  auto const& key_scales = key_block_scales.value();
  auto const& value_scales = value_block_scales.value();
  TVM_FFI_ICHECK_EQ(key_scales.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_scales.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_scales.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_scales.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_scales.size(0), key_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_scales.size(1), key_cache.size(1));
  TVM_FFI_ICHECK_EQ(key_scales.size(2), PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(key_scales.size(3), 8);
  TVM_FFI_ICHECK_EQ(value_scales.size(0), key_scales.size(0));
  TVM_FFI_ICHECK_EQ(value_scales.size(1), key_scales.size(1));
  TVM_FFI_ICHECK_EQ(value_scales.size(2), key_scales.size(2));
  TVM_FFI_ICHECK_EQ(value_scales.size(3), key_scales.size(3));
  TVM_FFI_ICHECK(key_scales.IsContiguous());
  TVM_FFI_ICHECK(value_scales.IsContiguous());
#else
  TVM_FFI_ICHECK_EQ(key_cache.size(3), 128);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), 128);
  TVM_FFI_ICHECK_EQ(key_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(value_cache.stride(3), 1);
#endif

  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK(seq_lens.dtype() == dl_int32 || seq_lens.dtype() == dl_uint32);
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
#if CAKE_FMHA_CONTEXT_NVFP4
  CheckSameDevice(query, key_scales, "key_block_scales");
  CheckSameDevice(query, value_scales, "value_block_scales");
#endif
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
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  int64_t actual_workspace_bytes = workspace_buffer.numel() * get_element_size(workspace_buffer);
  int64_t workspace_prefix = 0;
  CUtensorMap h_q = EncodeTmaQ(query);
#if CAKE_FMHA_CONTEXT_NVFP4
  int64_t pages = key_cache.size(0);
  TVM_FFI_ICHECK_GT(pages, 0);
  int64_t output_page_stride =
      static_cast<int64_t>(NUM_Q_HEADS / HEADS_PER_GROUP) * PAGE_SIZE * 128;
  TVM_FFI_ICHECK_LE(output_page_stride, static_cast<int64_t>(INT32_MAX));
  int64_t kv_bytes = pages * output_page_stride;
  int64_t value_offset = AlignUp(kv_bytes, 16);
  workspace_prefix = AlignUp(value_offset + kv_bytes, 16);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, workspace_prefix)
      << "Cake FMHA context NVFP4 dequantization requires " << workspace_prefix << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, workspace_prefix);
  auto* dequantized_k = workspace;
  auto* dequantized_v = workspace + value_offset;
  int64_t total_groups_64 = pages * (NUM_Q_HEADS / HEADS_PER_GROUP) * PAGE_SIZE * 8;
  TVM_FFI_ICHECK_LE(total_groups_64, static_cast<int64_t>(INT32_MAX));
  int total_groups = static_cast<int>(total_groups_64);
  unsigned int dequant_grid = static_cast<unsigned int>((total_groups_64 + 1023) / 1024);
  cudaError_t dequant_status = cake_fmha_launch_context_nvfp4_dequant(
      static_cast<uint8_t*>(key_cache.data_ptr()), static_cast<uint8_t*>(value_cache.data_ptr()),
      static_cast<uint8_t*>(key_scales.data_ptr()), static_cast<uint8_t*>(value_scales.data_ptr()),
      dequantized_k, dequantized_v, total_groups, static_cast<int>(output_page_stride),
      dequant_grid, 1, 1, stream);
  TVM_FFI_ICHECK_EQ(dequant_status, cudaSuccess)
      << "Cake FMHA context NVFP4 dequantization failed: "
      << cudaGetErrorString(dequant_status);
  CUtensorMap h_k =
      EncodeTmaContiguousPagedKv(dequantized_k, pages, output_page_stride, "key_cache");
  CUtensorMap h_v =
      EncodeTmaContiguousPagedKv(dequantized_v, pages, output_page_stride, "value_cache");
#else
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
#endif
  void* p_q = TmaDeviceSlot(h_q, query.device().device_id, stream);
  void* p_k = TmaDeviceSlot(h_k, query.device().device_id, stream);
  void* p_v = TmaDeviceSlot(h_v, query.device().device_id, stream);

  constexpr int units_per_batch = NUM_Q_HEADS / PACK_G;
  int64_t total_bh_64 = batch_size * units_per_batch;
  TVM_FFI_ICHECK_LE(total_bh_64, static_cast<int64_t>(INT32_MAX));
  int total_bh = static_cast<int>(total_bh_64);
  int64_t seq_q_offset = workspace_prefix;
  int64_t seq_kv_offset =
      AlignUp(seq_q_offset + total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  int64_t cu_q_offset = seq_kv_offset + total_bh_64 * static_cast<int64_t>(sizeof(int));
  int64_t cursor = AlignUp(cu_q_offset + total_bh_64 * static_cast<int64_t>(sizeof(int)), 16);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA context FP8 workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  int64_t counter_bytes =
      multi_ctas_kv_counter_buffer.numel() * get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(counter_bytes, 2 * static_cast<int64_t>(sizeof(uint32_t)));

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
  cudaError_t status = cake_fmha_launch_context_fp8(
      p_q, p_k, p_v, static_cast<uint8_t*>(out.data_ptr()), lse_ptr, sinks_ptr, table_k, table_v,
      seq_q_expanded, seq_kv_expanded, cu_q_expanded, softmax_scale_log2, output_scale, total_bh,
      static_cast<int>(page_row_stride), static_cast<int>(grid_x), dynamic_counter, grid_x, 1, 1,
      stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA context FP8 launch failed: " << cudaGetErrorString(status);

  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_context,
                              flashinfer::cake_fmha::cake_paged_attention_context);
