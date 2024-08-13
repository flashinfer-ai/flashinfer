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
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cstddef>
#include <cstdint>
#include <nvbench/nvbench.cuh>
#include <optional>

#include "flashinfer/attention/handler.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer_ops.cuh"

using namespace flashinfer;

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <typename dtype_in, typename dtype_out, bool append>
void bench_flashinfer_batch_prefill_with_ragged_kv(nvbench::state& state) {
  size_t kv_len = state.get_int64("kv_len");
  size_t qo_len = kv_len;
  size_t batch_size = state.get_int64("batch_size");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  size_t pos_encoding_mode = state.get_int64("pos_encoding_mode");
  size_t kv_layout = state.get_int64("kv_layout");
  bool causal = state.get_int64("causal");
  bool cooperative = state.get_int64("cooperative");
  bool allow_fp16_qk_reduction = state.get_int64("allow_fp16_qk_reduction");

  // Allocate input data:
  thrust::device_vector<dtype_in> Q(batch_size * qo_len * num_qo_heads * head_dim);
  thrust::device_vector<dtype_in> K(batch_size * kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_in> V(batch_size * kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_out> O(batch_size * qo_len * num_qo_heads * head_dim);
  size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
  thrust::device_vector<uint8_t> float_workspace(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<uint8_t> int_workspace(int_workspace_size_in_bytes);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(
      (batch_size * qo_len * num_qo_heads + 2 * batch_size * kv_len * num_kv_heads) * head_dim,
      "Read");
  state.add_global_memory_writes<dtype_out>(qo_len * batch_size * num_qo_heads * head_dim, "Write");

  std::vector<int32_t> qo_indptr_h(batch_size + 1);
  std::vector<int32_t> kv_indptr_h(batch_size + 1);

  for (uint32_t i = 0; i <= batch_size; ++i) {
    qo_indptr_h[i] = i * qo_len;
    kv_indptr_h[i] = i * kv_len;
  }

  thrust::device_vector<int32_t> qo_indptr_d(qo_indptr_h);
  thrust::device_vector<int32_t> kv_indptr_d(kv_indptr_h);

  BatchPrefillHandler handler;

  handler.BeginForward<dtype_out>(
      thrust::raw_pointer_cast(float_workspace.data()), float_workspace_size_in_bytes,
      thrust::raw_pointer_cast(int_workspace.data()), int_workspace_size_in_bytes,
      qo_indptr_h.data(), kv_indptr_h.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      /*page_size=*/1);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status;
    status = BatchPrefillWithRaggedKVCacheWrapper<dtype_in, dtype_in, dtype_out, int32_t>(
        &handler, thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(qo_indptr_d.data()),
        thrust::raw_pointer_cast(K.data()), thrust::raw_pointer_cast(V.data()),
        thrust::raw_pointer_cast(kv_indptr_d.data()),
        /*q_offset=*/nullptr, /*k_rope_pos_offset=*/nullptr, thrust::raw_pointer_cast(O.data()),
        /*lse=*/nullptr, batch_size, num_qo_heads, num_kv_heads, head_dim, causal,
        QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
    timer.stop();
  });
  const auto measured_mean = static_cast<nvbench::float32_t>(
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
  auto& summ = state.add_summary("nv/tflops");
  summ.set_string("description", "Achieved TFlops/s");
  summ.set_string("name", "TFlops/s");
  float tflops;
  if (causal) {
    tflops = (batch_size * (qo_len * (2 * kv_len - qo_len) * 2 * num_qo_heads * head_dim)) /
             measured_mean / 1e12;
  } else {
    tflops = (batch_size * qo_len * kv_len * 4 * num_qo_heads * head_dim) / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_BATCH_PREFILL_WITH_RAGGED_KV(dtype_in, dtype_out)                     \
  auto bench_flashinfer_batch_prefill_with_ragged_kv_##dtype_in##_##dtype_out##_ =             \
      bench_flashinfer_batch_prefill_with_ragged_kv<dtype_in, dtype_out, false>;               \
  NVBENCH_BENCH(bench_flashinfer_batch_prefill_with_ragged_kv_##dtype_in##_##dtype_out##_)     \
      .set_name(                                                                               \
          ("bench_flashinfer_batch_prefill_with_ragged_kv_" STR(dtype_in) "_" STR(dtype_out))) \
      .add_int64_axis("kv_len", {32, 64, 128, 256, 512, 1024, 2048, 4096})                     \
      .add_int64_axis("batch_size", {4, 8, 32})                                                \
      .add_int64_axis("num_qo_heads", {32})                                                    \
      .add_int64_axis("num_kv_heads", {32})                                                    \
      .add_int64_axis("head_dim", {128})                                                       \
      .add_int64_axis("causal", {0, 1})                                                        \
      .add_int64_axis("kv_layout", {0})                                                        \
      .add_int64_axis("pos_encoding_mode", {0})                                                \
      .add_int64_axis("allow_fp16_qk_reduction", {0})                                          \
      .add_int64_axis("cooperative", {1})

BENCH_FLASHINFER_BATCH_PREFILL_WITH_RAGGED_KV(half, half);
