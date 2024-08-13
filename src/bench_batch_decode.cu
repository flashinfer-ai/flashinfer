/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_vector.h>

#include <cstddef>
#include <cstdint>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "flashinfer_ops.cuh"
#include "utils.h"

using utils::vec_bytes;
using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void bench_flashinfer_batch_decode(nvbench::state& state) {
  constexpr size_t head_dim = 128;
  constexpr auto pos_encoding_mode = PosEncodingMode::kNone;
  size_t seqlen = state.get_int64("seqlen");
  size_t batch_size = state.get_int64("batch_size");
  size_t page_size = state.get_int64("page_size");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  bool cooperative = state.get_int64("cooperative");

  // KV cache:
  auto pages_per_seq = (seqlen + page_size - 1) / page_size;
  auto num_pages = pages_per_seq * batch_size;
  std::vector<int32_t> kv_indptr_host{0};
  std::vector<int32_t> kv_indicies_host;
  std::vector<int32_t> kv_last_page_len_host;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t p = 0; p < pages_per_seq; ++p) {
      kv_indicies_host.push_back(i * pages_per_seq + p);
    }
    kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
    kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
  }
  thrust::device_vector<T> kv_data(num_pages * 2 * num_kv_heads * page_size * head_dim);
  thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
  thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
  thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);
  paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data.data()), thrust::raw_pointer_cast(kv_indices.data()),
      thrust::raw_pointer_cast(kv_indptr.data()),
      thrust::raw_pointer_cast(kv_last_page_len.data()));
  // Allocate input data:
  thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim);
  thrust::device_vector<T> o(batch_size * num_qo_heads * head_dim);
  state.add_global_memory_reads<uint8_t>(
      vec_bytes(q) + (num_pages * 2 * num_kv_heads * page_size * head_dim) * sizeof(T) +
          vec_bytes(kv_indptr) + vec_bytes(kv_indices) + vec_bytes(kv_last_page_len),
      "Read");
  state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");
  BatchDecodeHandler handler;

  if (cooperative) {
    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
    // begin forward
    BatchDecodeHandlerBeginForward<PageStorage::kIndices, T, T, T, int32_t>(
        &handler, (void*)thrust::raw_pointer_cast(float_buffer.data()),
        float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(int_buffer.data()),
        int_workspace_size_in_bytes, kv_indptr_host.data(), kv_last_page_len_host.data(),
        batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, pos_encoding_mode);
    state.exec([&](nvbench::launch&) {
      cudaError_t status =
          BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, T, T, T, int32_t>(
              &handler, thrust::raw_pointer_cast(q.data()), /*q_offset=*/nullptr, paged_kv,
              thrust::raw_pointer_cast(o.data()), /*lse=*/nullptr, num_qo_heads, pos_encoding_mode);
      if (status != cudaSuccess) {
        state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
      }
    });
  } else {
    state.exec([&](nvbench::launch&) {
      cudaError_t status =
          BatchDecodeWithPagedKVCacheNoSplitKV<PageStorage::kIndices, T, T, T, int32_t>(
              thrust::raw_pointer_cast(q.data()), /*q_offset=*/nullptr, paged_kv,
              kv_partition_info_t<int32_t>(), thrust::raw_pointer_cast(o.data()),
              /*lse=*/nullptr, num_qo_heads, pos_encoding_mode);
      if (status != cudaSuccess) {
        state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
      }
    });
  }
}

template <typename T>
void bench_flashinfer_batch_decode_with_prefill(nvbench::state& state) {
  constexpr size_t head_dim = 128;
  constexpr auto pos_encoding_mode = PosEncodingMode::kNone;
  size_t seqlen = state.get_int64("seqlen");
  size_t batch_size = state.get_int64("batch_size");
  size_t page_size = state.get_int64("page_size");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");

  // KV cache:
  auto pages_per_seq = (seqlen + page_size - 1) / page_size;
  auto num_pages = pages_per_seq * batch_size;
  std::vector<int32_t> kv_indptr_host{0};
  std::vector<int32_t> kv_indicies_host;
  std::vector<int32_t> kv_last_page_len_host;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t p = 0; p < pages_per_seq; ++p) {
      kv_indicies_host.push_back(i * pages_per_seq + p);
    }
    kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
    kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
  }
  thrust::device_vector<T> kv_data(num_pages * 2 * num_kv_heads * page_size * head_dim);
  thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
  thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
  thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);
  paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data.data()), thrust::raw_pointer_cast(kv_indices.data()),
      thrust::raw_pointer_cast(kv_indptr.data()),
      thrust::raw_pointer_cast(kv_last_page_len.data()));

  // Allocate input data:
  thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim);
  thrust::device_vector<T> o(batch_size * num_qo_heads * head_dim);
  std::vector<int32_t> qo_indptr_h{0};
  for (uint32_t i = 0; i < batch_size; ++i) {
    qo_indptr_h.push_back(qo_indptr_h.back() + 1);
  }
  thrust::device_vector<int32_t> qo_indptr_d(qo_indptr_h);
  state.add_global_memory_reads<uint8_t>(
      vec_bytes(q) + (num_pages * 2 * num_kv_heads * page_size * head_dim) * sizeof(T) +
          vec_bytes(kv_indptr) + vec_bytes(kv_indices) + vec_bytes(kv_last_page_len),
      "Read");
  state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");
  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  handler.BeginForward<T, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      qo_indptr_h.data(), kv_indptr_host.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      page_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudaError_t status =
        BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, T, T, T, int32_t>(
            &handler, thrust::raw_pointer_cast(q.data()),
            thrust::raw_pointer_cast(qo_indptr_d.data()),
            /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o.data()),
            /*lse=*/nullptr, num_qo_heads,
            /*causal=*/false, pos_encoding_mode);
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_DECODE(dtype)                                                 \
  auto bench_flashinfer_batch_decode_##dtype##_ = bench_flashinfer_batch_decode<dtype>;      \
  NVBENCH_BENCH(bench_flashinfer_batch_decode_##dtype##_)                                    \
      .set_name("bench_flashinfer_batch_decode_" STR(dtype))                                 \
      .add_int64_axis("seqlen",                                                              \
                      {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536})  \
      .add_int64_axis("batch_size",                                                          \
                      {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  \
                       15,  16,  20,  24,  28,  32,  40,  48,  56,  64,  80,  96,  112, 128, \
                       160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024})         \
      .add_int64_axis("page_size", {4, 8, 16, 32, 64})                                       \
      .add_int64_axis("num_qo_heads", {32})                                                  \
      .add_int64_axis("num_kv_heads", {32, 4})                                               \
      .add_int64_axis("cooperative", {0, 1})

#define BENCH_FLASHINFER_BATCH_DECODE_WITH_PREFILL(dtype)                                   \
  auto bench_flashinfer_batch_decode_with_prefill_##dtype##_ =                              \
      bench_flashinfer_batch_decode_with_prefill<dtype>;                                    \
  NVBENCH_BENCH(bench_flashinfer_batch_decode_with_prefill_##dtype##_)                      \
      .set_name("bench_flashinfer_batch_decode_with_prefill_" STR(dtype))                   \
      .add_int64_axis("seqlen",                                                             \
                      {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) \
      .add_int64_axis("batch_size", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024})          \
      .add_int64_axis("page_size", {16})                                                    \
      .add_int64_axis("num_qo_heads", {32})                                                 \
      .add_int64_axis("num_kv_heads", {32, 4})

BENCH_FLASHINFER_BATCH_DECODE(half);
BENCH_FLASHINFER_BATCH_DECODE_WITH_PREFILL(half);
