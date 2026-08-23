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

#ifndef CAKE_FMHA_HD256_FP8
#error "CAKE_FMHA_HD256_FP8 must be supplied by the route-specific JIT"
#endif
#ifndef CAKE_FMHA_SOURCE_PAGE_SIZE
#error "CAKE_FMHA_SOURCE_PAGE_SIZE must be supplied by the route-specific JIT"
#endif

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {
namespace {

using tvm::ffi::TensorView;

constexpr int kHeadDim = 256;
constexpr int kBlockM = 128;
constexpr int kMicroPage = 16;
constexpr int kSupportThreads = 256;

int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

double ScalarScale(Variant<double, ffi::Tensor> scale, const char* name) {
  auto scalar = scale.as<double>();
  TVM_FFI_ICHECK(scalar.has_value()) << name << " must be a host scalar";
  return scalar.value();
}

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
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

CUtensorMap EncodeStagedQ(void* pointer, int64_t total_rows) {
#if CAKE_FMHA_HD256_FP8
  uint64_t global_dim[3] = {128u, static_cast<uint64_t>(total_rows), 2u};
  uint64_t global_strides[2] = {256u, 128u};
  uint32_t box_dim[3] = {128u, 128u, 1u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
#else
  uint64_t global_dim[3] = {64u, static_cast<uint64_t>(total_rows), 4u};
  uint64_t global_strides[2] = {512u, 128u};
  uint32_t box_dim[3] = {64u, 128u, 2u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#endif
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, dtype, 3, pointer, global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA HD256 staged query map";
  return tm;
}

CUtensorMap EncodeStagedKv(void* pointer, int64_t total_micro_pages) {
#if CAKE_FMHA_HD256_FP8
  uint64_t global_dim[4] = {
      128u, kMicroPage, 2u, static_cast<uint64_t>(total_micro_pages)};
  uint64_t global_strides[3] = {256u, 128u, 4096u};
  uint32_t box_dim[4] = {128u, kMicroPage, 1u, 1u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
#else
  uint64_t global_dim[4] = {
      64u, kMicroPage, 4u, static_cast<uint64_t>(total_micro_pages)};
  uint64_t global_strides[3] = {512u, 128u, 8192u};
  uint32_t box_dim[4] = {64u, kMicroPage, 1u, 1u};
  CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
#endif
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult result = cuTensorMapEncodeTiled(
      &tm, dtype, 4, pointer, global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to encode Cake FMHA HD256 staged KV map";
  return tm;
}

unsigned int SupportGrid(int64_t items) {
  return static_cast<unsigned int>((items + kSupportThreads - 1) /
                                   kSupportThreads);
}

}  // namespace

void cake_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query,
    TensorView key_cache, TensorView value_cache, TensorView workspace_buffer,
    TensorView multi_ctas_kv_counter_buffer, TensorView block_tables,
    TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale,
    Variant<double, ffi::Tensor> bmm2_scale, double o_sf_scale,
    int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q,
    TensorView cum_seq_lens_kv, int64_t sm_count, bool enable_pdl,
    int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> key_block_scales,
    Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor,
    Optional<bool> uses_shared_paged_kv_idx,
    Optional<bool> use_fp16_softmax, Optional<bool> uses_spcompress,
    bool is_causal, Optional<TensorView> lse, int64_t lse_stride_tokens,
    int64_t lse_stride_heads) {
#if CAKE_FMHA_HD256_FP8
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK(is_causal);
  constexpr int kInputBytes = kHeadDim;
  constexpr int kOutputBytes = 2 * kHeadDim;
#else
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float16);
  TVM_FFI_ICHECK_EQ(out.dtype(), dl_float16);
  TVM_FFI_ICHECK(!is_causal);
  constexpr int kInputBytes = 2 * kHeadDim;
  constexpr int kOutputBytes = 2 * kHeadDim;
  TVM_FFI_ICHECK_EQ(ScalarScale(bmm2_scale, "bmm2_scale"), 1.0);
#endif
  TVM_FFI_ICHECK(!out_scale_factor.has_value());
  TVM_FFI_ICHECK(!attention_sinks.has_value());
  TVM_FFI_ICHECK(!key_block_scales.has_value() &&
                 !value_block_scales.has_value());
  TVM_FFI_ICHECK(!lse.has_value());
  TVM_FFI_ICHECK_EQ(lse_stride_tokens, 0);
  TVM_FFI_ICHECK_EQ(lse_stride_heads, 0);
  TVM_FFI_ICHECK_EQ(
      skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f);
  TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));
  TVM_FFI_ICHECK(!uses_spcompress.value_or(false));
  TVM_FFI_ICHECK_EQ(window_left, -1);
  TVM_FFI_ICHECK_EQ(o_sf_scale, -1.0);
  TVM_FFI_ICHECK_EQ(o_sf_vec_size, -1);
  TVM_FFI_ICHECK_EQ(o_sf_start_index, 0);
  TVM_FFI_ICHECK(!uses_shared_paged_kv_idx.value_or(true));
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(max_q_len, 0);
  TVM_FFI_ICHECK_GT(max_kv_len, 0);
  TVM_FFI_ICHECK_GT(sm_count, 0);
  TVM_FFI_ICHECK_EQ((max_q_len + kBlockM - 1) / kBlockM,
                    NUM_M_BLOCKS);

  TVM_FFI_ICHECK_EQ(query.ndim(), 3);
  TVM_FFI_ICHECK_EQ(query.size(1), NUM_Q_HEADS);
  TVM_FFI_ICHECK_EQ(query.size(2), kHeadDim);
  TVM_FFI_ICHECK_EQ(query.stride(2), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  TVM_FFI_ICHECK_EQ(out.size(0), query.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), query.size(1));
  TVM_FFI_ICHECK_EQ(out.size(2), query.size(2));
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(value_cache.ndim(), 4);
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(value_cache.size(1), NUM_KV_HEADS);
  TVM_FFI_ICHECK_EQ(key_cache.size(2), CAKE_FMHA_SOURCE_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(value_cache.size(2), CAKE_FMHA_SOURCE_PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(key_cache.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(value_cache.size(3), kHeadDim);
  TVM_FFI_ICHECK_EQ(key_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(value_cache.stride(3), 1);
  TVM_FFI_ICHECK_EQ(block_tables.ndim(), 3);
  TVM_FFI_ICHECK_EQ(block_tables.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(block_tables.size(1), 2);
  TVM_FFI_ICHECK(block_tables.dtype() == dl_int32 ||
                 block_tables.dtype() == dl_uint32);
  TVM_FFI_ICHECK_EQ(block_tables.stride(2), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), batch_size);
  TVM_FFI_ICHECK(seq_lens.dtype() == dl_int32 ||
                 seq_lens.dtype() == dl_uint32);
  TVM_FFI_ICHECK(seq_lens.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_q.IsContiguous());
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.ndim(), 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_kv.dtype(), dl_int32);
  TVM_FFI_ICHECK(cum_seq_lens_kv.IsContiguous());

  CheckSameDevice(query, key_cache, "key_cache");
  CheckSameDevice(query, value_cache, "value_cache");
  CheckSameDevice(query, out, "out");
  CheckSameDevice(query, workspace_buffer, "workspace_buffer");
  CheckSameDevice(query, multi_ctas_kv_counter_buffer,
                  "multi_ctas_kv_counter_buffer");
  CheckSameDevice(query, block_tables, "block_tables");
  CheckSameDevice(query, seq_lens, "seq_lens");
  CheckSameDevice(query, cum_seq_lens_q, "cum_seq_lens_q");
  CheckSameDevice(query, cum_seq_lens_kv, "cum_seq_lens_kv");

  int64_t padded_q = NUM_M_BLOCKS * kBlockM;
  int64_t max_micro_pages = (max_kv_len + kMicroPage - 1) / kMicroPage;
  int64_t total_q_rows = batch_size * NUM_Q_HEADS * padded_q;
  int64_t total_micro_pages =
      batch_size * NUM_KV_HEADS * max_micro_pages;
  int64_t cursor = 0;
  int64_t q_offset = cursor;
  cursor += total_q_rows * kInputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t k_offset = cursor;
  cursor += total_micro_pages * kMicroPage * kInputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t v_offset = cursor;
  cursor += total_micro_pages * kMicroPage * kInputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t o_offset = cursor;
  cursor += total_q_rows * kOutputBytes;
  cursor = AlignUp(cursor, 16);
  int64_t seq_q_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t seq_kv_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t cu_q_offset = cursor;
  cursor += batch_size * NUM_Q_HEADS * static_cast<int64_t>(sizeof(int));
  int64_t page_offset = cursor;
  cursor += total_micro_pages * static_cast<int64_t>(sizeof(int));

  TVM_FFI_ICHECK(workspace_buffer.IsContiguous());
  TVM_FFI_ICHECK_EQ(
      reinterpret_cast<uintptr_t>(workspace_buffer.data_ptr()) % 16, 0);
  int64_t actual_workspace_bytes =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  TVM_FFI_ICHECK_GE(actual_workspace_bytes, cursor)
      << "Cake FMHA HD256 context workspace requires " << cursor << " bytes";
  TVM_FFI_ICHECK_GE(workspace_size, cursor);
  int64_t counter_bytes = multi_ctas_kv_counter_buffer.numel() *
                          get_element_size(multi_ctas_kv_counter_buffer);
  TVM_FFI_ICHECK_GE(counter_bytes, static_cast<int64_t>(sizeof(uint32_t)));

  auto* workspace = static_cast<uint8_t*>(workspace_buffer.data_ptr());
  auto* q_packed = workspace + q_offset;
  auto* k_packed = workspace + k_offset;
  auto* v_packed = workspace + v_offset;
  auto* o_packed = workspace + o_offset;
  auto* seq_lens_q = reinterpret_cast<int*>(workspace + seq_q_offset);
  auto* seq_lens_kv = reinterpret_cast<int*>(workspace + seq_kv_offset);
  auto* cu_seq_lens_q = reinterpret_cast<int*>(workspace + cu_q_offset);
  auto* kernel_page_table = reinterpret_cast<int*>(workspace + page_offset);
  auto* dynamic_counter =
      static_cast<uint32_t*>(multi_ctas_kv_counter_buffer.data_ptr());

  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  cudaStream_t stream = get_stream(query.device());
  unsigned int stage_q_grid = SupportGrid(total_q_rows * kInputBytes);
  unsigned int stage_kv_grid = SupportGrid(
      total_micro_pages * kMicroPage * static_cast<int64_t>(kInputBytes));
  unsigned int metadata_grid = SupportGrid(std::max<int64_t>(
      batch_size * NUM_Q_HEADS, total_micro_pages));
  unsigned int scatter_grid = SupportGrid(total_q_rows * kOutputBytes);

  cudaError_t status = cake_fmha_launch_hd256_stage_q(
      static_cast<const uint8_t*>(query.data_ptr()), q_packed,
      static_cast<const int*>(cum_seq_lens_q.data_ptr()),
      static_cast<int>(batch_size), NUM_Q_HEADS, static_cast<int>(padded_q),
      kInputBytes, query.stride(0) * get_element_size(query),
      query.stride(1) * get_element_size(query), stage_q_grid, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 query staging failed: " << cudaGetErrorString(status);
  status = cake_fmha_launch_hd256_stage_kv(
      static_cast<const uint8_t*>(key_cache.data_ptr()),
      static_cast<const uint8_t*>(value_cache.data_ptr()), k_packed, v_packed,
      static_cast<const int*>(block_tables.data_ptr()),
      static_cast<const int*>(seq_lens.data_ptr()), static_cast<int>(batch_size),
      NUM_KV_HEADS, CAKE_FMHA_SOURCE_PAGE_SIZE,
      static_cast<int>(max_micro_pages), kInputBytes,
      key_cache.stride(0) * get_element_size(key_cache),
      key_cache.stride(2) * get_element_size(key_cache),
      key_cache.stride(1) * get_element_size(key_cache), block_tables.stride(0),
      block_tables.stride(1), stage_kv_grid, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 KV staging failed: " << cudaGetErrorString(status);
  status = cake_fmha_launch_hd256_prepare_metadata(
      static_cast<const int*>(cum_seq_lens_q.data_ptr()),
      static_cast<const int*>(seq_lens.data_ptr()), seq_lens_q, seq_lens_kv,
      cu_seq_lens_q, kernel_page_table, dynamic_counter,
      static_cast<int>(batch_size), NUM_Q_HEADS, NUM_KV_HEADS,
      static_cast<int>(padded_q), static_cast<int>(max_micro_pages),
      metadata_grid, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 metadata staging failed: "
      << cudaGetErrorString(status);

  CUtensorMap h_q = EncodeStagedQ(q_packed, total_q_rows);
  CUtensorMap h_k = EncodeStagedKv(k_packed, total_micro_pages);
  CUtensorMap h_v = EncodeStagedKv(v_packed, total_micro_pages);
  auto const* p_q = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_q, query.device().device_id, stream));
  auto const* p_k = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_k, query.device().device_id, stream));
  auto const* p_v = reinterpret_cast<CakeFmhaTensorMap const*>(
      TmaDeviceSlot(h_v, query.device().device_id, stream));
  float softmax_scale_log2 =
      static_cast<float>(ScalarScale(bmm1_scale, "bmm1_scale") *
                         1.4426950408889634);
  unsigned int total_tiles = static_cast<unsigned int>(
      NUM_M_BLOCKS * batch_size * NUM_Q_HEADS);
  unsigned int grid_x =
      std::min<unsigned int>(static_cast<unsigned int>(sm_count), total_tiles);
#if CAKE_FMHA_HD256_FP8
  status = cake_fmha_launch_context_fp8_hd256(
      p_q, p_k, p_v, o_packed, kernel_page_table, seq_lens_q, seq_lens_kv,
      cu_seq_lens_q, softmax_scale_log2,
      static_cast<float>(ScalarScale(bmm2_scale, "bmm2_scale")),
      static_cast<int>(batch_size * NUM_Q_HEADS),
      static_cast<int>(max_micro_pages), dynamic_counter, grid_x, 1, 1,
      stream);
#else
  status = cake_fmha_launch_context_fp16_hd256(
      p_q, p_k, p_v, reinterpret_cast<__half*>(o_packed), kernel_page_table,
      seq_lens_q, seq_lens_kv, cu_seq_lens_q, softmax_scale_log2,
      static_cast<int>(batch_size * NUM_Q_HEADS),
      static_cast<int>(max_micro_pages), dynamic_counter, grid_x, 1, 1,
      stream);
#endif
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 context launch failed: "
      << cudaGetErrorString(status);
  status = cake_fmha_launch_hd256_scatter_o(
      o_packed, static_cast<uint8_t*>(out.data_ptr()),
      static_cast<const int*>(cum_seq_lens_q.data_ptr()),
      static_cast<int>(batch_size), NUM_Q_HEADS, static_cast<int>(padded_q),
      kOutputBytes, out.stride(0) * get_element_size(out),
      out.stride(1) * get_element_size(out), scatter_grid, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA HD256 output scatter failed: "
      << cudaGetErrorString(status);

  (void)cum_seq_lens_kv;
  (void)enable_pdl;
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    cake_paged_attention_context,
    flashinfer::cake_fmha::cake_paged_attention_context);
