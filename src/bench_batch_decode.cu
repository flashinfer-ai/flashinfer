#include <thrust/device_vector.h>

#include <cstdint>
#include <flashinfer/decode.cuh>
#include <nvbench/nvbench.cuh>
#include <vector>

#include "utils.h"

using utils::vec_bytes;

template <typename T>
void bench_flashinfer_batch_decode(nvbench::state& state) {
  constexpr size_t head_dim = 128;
  constexpr size_t num_layers = 3;
  constexpr size_t layer_idx = 1;
  constexpr auto rotary_mode = flashinfer::RotaryMode::kNone;
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
  std::vector<int32_t> kv_last_page_offset_host;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t p = 0; p < pages_per_seq; ++p) {
      kv_indicies_host.push_back(i * pages_per_seq + p);
    }
    kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
    kv_last_page_offset_host.push_back((seqlen - 1) % page_size + 1);
  }
  thrust::device_vector<T> kv_data(num_pages * num_layers * 2 * num_kv_heads * page_size *
                                   head_dim);
  thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
  thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
  thrust::device_vector<int32_t> kv_last_page_offset(kv_last_page_offset_host);
  thrust::device_vector<float> tmp(4 * 1024 * 1024);
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      thrust::raw_pointer_cast(kv_data.data()), thrust::raw_pointer_cast(kv_indptr.data()),
      thrust::raw_pointer_cast(kv_indices.data()),
      thrust::raw_pointer_cast(kv_last_page_offset.data()));

  // Allocate input data:
  thrust::device_vector<T> q(batch_size * num_qo_heads * head_dim);
  thrust::device_vector<T> o(batch_size * num_qo_heads * head_dim);
  state.add_global_memory_reads<uint8_t>(
      vec_bytes(q) + (num_pages * 2 * num_kv_heads * page_size * head_dim) * sizeof(T) +
          vec_bytes(kv_indptr) + vec_bytes(kv_indices) + vec_bytes(kv_last_page_offset),
      "Read");
  state.add_global_memory_writes<uint8_t>(vec_bytes(o), "Write");

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudaError_t status = flashinfer::BatchDecodeWithPagedKVCache<T, T>(
        thrust::raw_pointer_cast(q.data()), paged_kv, thrust::raw_pointer_cast(o.data()),
        thrust::raw_pointer_cast(tmp.data()),
        num_qo_heads, rotary_mode);
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_DECODE(dtype)                                                   \
  auto bench_flashinfer_batch_decode_##dtype##_ = bench_flashinfer_batch_decode<dtype>;        \
  NVBENCH_BENCH(bench_flashinfer_batch_decode_##dtype##_)                                      \
      .set_name("bench_flashinfer_batch_decode_" STR(dtype))                                   \
      .add_int64_axis("seqlen", {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) \
      .add_int64_axis("batch_size",                                                            \
                      {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,    \
                       15,  16,  20,  24,  28,  32,  40,  48,  56,  64,  80,  96,  112, 128,   \
                       160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024})           \
      .add_int64_axis("page_size", {4, 8, 16, 32, 64})                                         \
      .add_int64_axis("num_qo_heads", {32})                                                    \
      .add_int64_axis("num_kv_heads", {32, 4})

BENCH_FLASHINFER_BATCH_DECODE(half);
