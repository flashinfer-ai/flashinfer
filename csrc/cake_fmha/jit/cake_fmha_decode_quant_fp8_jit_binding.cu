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

#ifndef CAKE_FMHA_PAGE_SIZE
#error "CAKE_FMHA_PAGE_SIZE must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_FULL_BLOCKS
#error "CAKE_FMHA_FULL_BLOCKS must be supplied by the route-specific JIT"
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

constexpr int kHeadDim = 128;
constexpr int kTileQ = 8;

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  size_t used = 0;
};

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

CUtensorMap EncodeTmaQuery(TensorView tensor) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 3);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK(tensor.IsContiguous());
  TVM_FFI_ICHECK_EQ(tensor.size(0), BATCH_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(tensor.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16, 0);
  uint64_t global_dim[3] = {
      kHeadDim, static_cast<uint64_t>(BATCH_SIZE * NUM_Q_HEADS), 1u};
  uint64_t global_strides[2] = {kHeadDim, kHeadDim};
  uint32_t box_dim[3] = {kHeadDim, kTileQ, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA FP8 query tensor map";
  return tm;
}

CUtensorMap EncodeTmaPagedKv(TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 4)
      << name << " must be rank-4 HND paged KV";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(tensor.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(tensor.size(2), CAKE_FMHA_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(tensor.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(tensor.stride(3), 1);
  TVM_FFI_ICHECK_GT(tensor.stride(0), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(1), 0);
  TVM_FFI_ICHECK_GT(tensor.stride(2), 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(0) % 16, 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(1) % 16, 0);
  TVM_FFI_ICHECK_EQ(tensor.stride(2) % 16, 0);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16, 0);

  uint64_t global_dim[5] = {
      kHeadDim, CAKE_FMHA_PAGE_SIZE, 1u,
      static_cast<uint64_t>(tensor.size(1)),
      static_cast<uint64_t>(tensor.size(0))};
  uint64_t global_strides[4] = {
      static_cast<uint64_t>(tensor.stride(2)), kHeadDim,
      static_cast<uint64_t>(tensor.stride(1)),
      static_cast<uint64_t>(tensor.stride(0))};
  uint32_t box_dim[5] = {kHeadDim, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA " << name << " tensor map";
  return tm;
}

__global__ void PrepareQuantDecode(
    const int* page_table, int* padded_page_table, int page_items,
    int source_pages, int padded_pages, float* bmm1_scalar_ptr,
    float bmm1_scalar, float* bmm2_scalar_ptr, float bmm2_scalar) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (padded_page_table != nullptr && index < page_items) {
    int column = index % padded_pages;
    int batch = index / padded_pages;
    padded_page_table[index] =
        column < source_pages
            ? page_table[batch * source_pages + column]
            : 0;
  }
  if (index == 0) {
    if (bmm1_scalar_ptr != nullptr) *bmm1_scalar_ptr = bmm1_scalar;
    if (bmm2_scalar_ptr != nullptr) *bmm2_scalar_ptr = bmm2_scalar;
  }
}

struct SplitPlan {
  int even_blocks;
  int num_splits;
  int blocks_per_split;
};

SplitPlan ResolveSplitPlan(TensorView seq_lens, int64_t max_kv_len,
                           int sm_count, cudaStream_t stream) {
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUresult capture_result =
      cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_ICHECK_EQ(capture_result, CUDA_SUCCESS);
  TVM_FFI_ICHECK_EQ(capture_status, CU_STREAM_CAPTURE_STATUS_NONE)
      << "FP8 optimized routing must resolve seq_lens before CUDA Graph capture";

  std::vector<int> host_lengths(BATCH_SIZE);
  cudaError_t status = cudaMemcpyAsync(
      host_lengths.data(), seq_lens.data_ptr(), BATCH_SIZE * sizeof(int),
      cudaMemcpyDeviceToHost, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess);
  status = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess);

  int max_seen = 0;
  int even_bucket = -1;
  bool split_safe = true;
  for (int length : host_lengths) {
    TVM_FFI_ICHECK_GE(length, 512)
        << "optimized FP8 decode requires at least four KV blocks";
#if CAKE_FMHA_FULL_BLOCKS
    TVM_FFI_ICHECK_EQ(length % 128, 0)
        << "optimized FP8 decode requires full 128-token blocks";
#endif
    int blocks = (length + 127) / 128;
    blocks += blocks % 2;
    if (even_bucket < 0) even_bucket = blocks;
    TVM_FFI_ICHECK_EQ(blocks, even_bucket)
        << "optimized FP8 decode requires one evened-block bucket";
    max_seen = std::max(max_seen, length);
    split_safe = split_safe && length % 256 == 0;
  }
  TVM_FFI_ICHECK_EQ(max_seen, max_kv_len);

  int total_tiles = BATCH_SIZE * NUM_KV_HEADS;
  int num_splits = 1;
  int blocks_per_split = even_bucket;
  if (split_safe && total_tiles < sm_count && even_bucket >= 8) {
    int target = std::max(1, sm_count / total_tiles);
    blocks_per_split = (even_bucket + target - 1) / target;
    blocks_per_split = std::max(4, blocks_per_split + blocks_per_split % 2);
    while (blocks_per_split < even_bucket &&
           even_bucket % blocks_per_split == 2) {
      blocks_per_split += 2;
    }
    num_splits = (even_bucket + blocks_per_split - 1) / blocks_per_split;
  }
  return {even_bucket, num_splits, blocks_per_split};
}

}  // namespace

void cake_paged_attention_decode(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query,
    TensorView key_cache, TensorView value_cache, TensorView workspace_buffer,
    TensorView multi_ctas_kv_counter_buffer, TensorView block_tables,
    TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale,
    Variant<double, ffi::Tensor> bmm2_scale, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size,
    Optional<TensorView> attention_sinks,
    Optional<TensorView> cum_seq_lens_q,
    Optional<TensorView> key_block_scales,
    Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor,
    Optional<bool> uses_shared_paged_kv_idx, Optional<TensorView> lse,
    int64_t lse_stride_tokens, int64_t lse_stride_heads,
    bool enable_block_sparse_attention,
    Optional<TensorView> sparse_mla_top_k_lens) {
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK(query.IsContiguous());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK_EQ(batch_size, BATCH_SIZE);
  TVM_FFI_ICHECK_EQ(max_q_len, 1);
  TVM_FFI_ICHECK_EQ(NUM_Q_HEADS, 8 * NUM_KV_HEADS);
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!attention_sinks.has_value());
  TVM_FFI_ICHECK(!cum_seq_lens_q.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() &&
                 !value_block_scales.has_value());
  TVM_FFI_ICHECK(!lse.has_value());
  TVM_FFI_ICHECK_EQ(lse_stride_tokens, 0);
  TVM_FFI_ICHECK_EQ(lse_stride_heads, 0);
  TVM_FFI_ICHECK_EQ(
      skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(!enable_block_sparse_attention &&
                 !sparse_mla_top_k_lens.has_value());
  TVM_FFI_ICHECK_EQ(sparse_mla_top_k, 0);
  TVM_FFI_ICHECK_EQ(window_left, -1);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);
  TVM_FFI_ICHECK(uses_shared_paged_kv_idx.value_or(true));
  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(workspace_buffer.data_ptr()) % 16, 0);
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 2);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 ||
                 block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK(block_tables.IsContiguous());
  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK(seq_lens.dtype() == dl_int32 ||
                 seq_lens.dtype() == dl_uint32);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");

  auto bmm1_tensor = bmm1_scale.as<ffi::Tensor>();
  auto bmm2_tensor = bmm2_scale.as<ffi::Tensor>();
  float bmm1_scalar = 0.0f;
  float bmm2_scalar = 0.0f;
  if (bmm1_tensor.has_value()) {
    auto const& scale = bmm1_tensor.value();
    TVM_FFI_ICHECK_EQ(scale.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(scale.numel(), 1);
    TVM_FFI_ICHECK(scale.IsContiguous());
    TVM_FFI_ICHECK_EQ(scale.device().device_type,
                      query.device().device_type);
    TVM_FFI_ICHECK_EQ(scale.device().device_id, query.device().device_id);
  } else {
    auto scalar = bmm1_scale.as<double>();
    TVM_FFI_ICHECK(scalar.has_value());
    bmm1_scalar = static_cast<float>(scalar.value());
  }
  if (bmm2_tensor.has_value()) {
    auto const& scale = bmm2_tensor.value();
    TVM_FFI_ICHECK_EQ(scale.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(scale.numel(), 1);
    TVM_FFI_ICHECK(scale.IsContiguous());
    TVM_FFI_ICHECK_EQ(scale.device().device_type,
                      query.device().device_type);
    TVM_FFI_ICHECK_EQ(scale.device().device_id, query.device().device_id);
  } else {
    auto scalar = bmm2_scale.as<double>();
    TVM_FFI_ICHECK(scalar.has_value());
    bmm2_scalar = static_cast<float>(scalar.value());
  }

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  SplitPlan plan = ResolveSplitPlan(seq_lens, max_kv_len,
                                    static_cast<int>(sm_count), stream);
  int64_t source_pages = block_tables.size(1);
  int64_t required_pages =
      plan.even_blocks * (128 / CAKE_FMHA_PAGE_SIZE);
  int64_t padded_pages = std::max(source_pages, required_pages);
  bool needs_page_padding = source_pages < padded_pages;

  int64_t cursor = 0;
  int64_t bmm1_offset = cursor;
  if (!bmm1_tensor.has_value()) cursor += sizeof(float);
  int64_t bmm2_offset = cursor;
  if (!bmm2_tensor.has_value()) cursor += sizeof(float);
  cursor = AlignUp(cursor, 16);
  int64_t page_offset = cursor;
  if (needs_page_padding) {
    cursor += BATCH_SIZE * padded_pages * static_cast<int64_t>(sizeof(int));
    cursor = AlignUp(cursor, 16);
  }
  int64_t partial_o_offset = cursor;
  int64_t partial_rows =
      BATCH_SIZE * NUM_Q_HEADS * static_cast<int64_t>(plan.num_splits);
  cursor += partial_rows * kHeadDim * static_cast<int64_t>(sizeof(float));
  int64_t partial_max_offset = cursor;
  cursor += partial_rows * static_cast<int64_t>(sizeof(float));
  int64_t partial_sum_offset = cursor;
  cursor += partial_rows * static_cast<int64_t>(sizeof(float));

  int64_t actual_workspace_bytes =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA FP8 decode workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* padded_page_table =
      needs_page_padding ? reinterpret_cast<int*>(workspace + page_offset)
                         : nullptr;
  float* bmm1_ptr =
      bmm1_tensor.has_value()
          ? static_cast<float*>(bmm1_tensor.value().data_ptr())
          : reinterpret_cast<float*>(workspace + bmm1_offset);
  float* bmm2_ptr =
      bmm2_tensor.has_value()
          ? static_cast<float*>(bmm2_tensor.value().data_ptr())
          : reinterpret_cast<float*>(workspace + bmm2_offset);

  CUtensorMap h_q = EncodeTmaQuery(query);
  CUtensorMap h_k = EncodeTmaPagedKv(key_cache, "key_cache");
  CUtensorMap h_v = EncodeTmaPagedKv(value_cache, "value_cache");
  void* p_q = TmaDeviceSlot(h_q, query.device().device_id, stream);
  void* p_k = TmaDeviceSlot(h_k, query.device().device_id, stream);
  void* p_v = TmaDeviceSlot(h_v, query.device().device_id, stream);

  int64_t page_items =
      needs_page_padding ? BATCH_SIZE * padded_pages : 0;
  int64_t prep_items = std::max<int64_t>(page_items, 1);
  int const prep_threads = 256;
  int const prep_blocks =
      static_cast<int>((prep_items + prep_threads - 1) / prep_threads);
  PrepareQuantDecode<<<prep_blocks, prep_threads, 0, stream>>>(
      static_cast<const int*>(block_tables.data_ptr()), padded_page_table,
      static_cast<int>(page_items), static_cast<int>(source_pages),
      static_cast<int>(padded_pages),
      bmm1_tensor.has_value() ? nullptr : bmm1_ptr, bmm1_scalar,
      bmm2_tensor.has_value() ? nullptr : bmm2_ptr, bmm2_scalar);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
      << "failed to prepare Cake FMHA FP8 decode inputs";

  int* page_table_ptr =
      needs_page_padding ? padded_page_table
                         : static_cast<int*>(block_tables.data_ptr());
  auto* partial_o = reinterpret_cast<float*>(workspace + partial_o_offset);
  auto* partial_max =
      reinterpret_cast<float*>(workspace + partial_max_offset);
  auto* partial_sum =
      reinterpret_cast<float*>(workspace + partial_sum_offset);
  int bmm1_is_log2 = bmm1_tensor.has_value() ? 1 : 0;
  unsigned int grid_x = static_cast<unsigned int>(
      BATCH_SIZE * NUM_KV_HEADS * plan.num_splits);
  cudaError_t status = cake_fmha_launch_decode_quant_fp8(
      p_q, p_k, p_v, static_cast<uint8_t*>(out.data_ptr()), page_table_ptr,
      static_cast<int*>(seq_lens.data_ptr()), bmm1_ptr, bmm2_ptr, partial_o,
      partial_max, partial_sum, static_cast<int>(padded_pages), 0,
      bmm1_is_log2, plan.num_splits, plan.blocks_per_split, grid_x, 1, 1,
      stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA FP8 decode launch failed: " << cudaGetErrorString(status);

  if (plan.num_splits > 1) {
    status = cake_fmha_launch_decode_quant_fp8_reduce(
        partial_o, partial_max, partial_sum,
        static_cast<uint8_t*>(out.data_ptr()), bmm2_ptr, plan.num_splits,
        BATCH_SIZE * NUM_Q_HEADS, 1, 1, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "Cake FMHA FP8 reduce launch failed: "
        << cudaGetErrorString(status);
  }

  (void)multi_ctas_kv_counter_buffer;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    cake_paged_attention_decode,
    flashinfer::cake_fmha::cake_paged_attention_decode);
