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
#include <unordered_set>
#include <vector>

#include "flashinfer_ops.cuh"
#include "utils.h"

using utils::vec_bytes;
using namespace flashinfer;

std::unordered_set<int> dev_to_bench{0};

template <typename T>
void bench_flashinfer_batch_decode_mla(nvbench::state& state) {
  int dev_id = state.get_device().value().get_id();
  if (dev_to_bench.count(dev_id) == 0) return;

  cudaSetDevice(dev_id);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

  constexpr size_t head_dim_ckv = 512;
  constexpr size_t head_dim_kpe = head_dim_ckv / 8;
  const size_t num_qo_heads = state.get_int64("num_qo_heads");
  ;

  size_t batch_size = state.get_int64("batch_size");
  size_t seqlen = state.get_int64("seqlen");
  size_t page_size = state.get_int64("page_size");

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
  thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
  thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
  thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);

  thrust::device_vector<T> q_nope(batch_size * num_qo_heads * head_dim_ckv);
  thrust::device_vector<T> q_pe(batch_size * num_qo_heads * head_dim_kpe);
  thrust::device_vector<T> ckv_data(num_pages * page_size * head_dim_ckv);
  thrust::device_vector<T> kpe_data(num_pages * page_size * head_dim_kpe);
  thrust::device_vector<T> o(q_nope.size());

  flashinfer::paged_kv_mla_t<T, int32_t> paged_kv_mla(
      page_size, head_dim_ckv, head_dim_kpe, batch_size, thrust::raw_pointer_cast(ckv_data.data()),
      thrust::raw_pointer_cast(kpe_data.data()), thrust::raw_pointer_cast(kv_indices.data()),
      thrust::raw_pointer_cast(kv_indptr.data()),
      thrust::raw_pointer_cast(kv_last_page_len.data()));

  state.add_global_memory_reads<uint8_t>(vec_bytes(q_nope) + vec_bytes(q_pe) + vec_bytes(ckv_data) +
                                             vec_bytes(kpe_data) + vec_bytes(kv_indptr) +
                                             vec_bytes(kv_indices) + vec_bytes(kv_last_page_len),
                                         "Read");
  state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");

  flashinfer::BatchDecodeHandler handler;
  handler.SetCUDAStream(stream);
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
  flashinfer::BatchDecodeHandlerPlanMLA<T, T, T, int32_t>(
      &handler, (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      kv_indptr_host.data(), kv_last_page_len_host.data(), batch_size, num_qo_heads, head_dim_ckv,
      page_size);

  state.exec([&](nvbench::launch&) {
    cudaError_t status = flashinfer::BatchDecodeWithPagedKVCacheWrapperMLA<T, T, T, int32_t>(
        &handler, thrust::raw_pointer_cast(q_nope.data()), thrust::raw_pointer_cast(q_pe.data()),
        /*q_rope_offset=*/nullptr, paged_kv_mla, thrust::raw_pointer_cast(o.data()),
        /*lse=*/nullptr, num_qo_heads, std::sqrt(192.0));
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });

  cudaStreamDestroy(stream);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define BENCH_FLASHINFER_BATCH_DECODE(dtype)                                                    \
  auto bench_flashinfer_batch_decode_mla_##dtype##_ = bench_flashinfer_batch_decode_mla<dtype>; \
  NVBENCH_BENCH(bench_flashinfer_batch_decode_mla_##dtype##_)                                   \
      .set_name("bench_flashinfer_batch_decode_mla_" STR(dtype))                                \
      .add_int64_axis("page_size", {64})                                                        \
      .add_int64_axis("batch_size", {16, 256})                                                  \
      .add_int64_axis("seqlen", {1024, 16384})                                                  \
      .add_int64_axis("num_qo_heads", {8, 16, 32, 40, 64, 128})

BENCH_FLASHINFER_BATCH_DECODE(half);
