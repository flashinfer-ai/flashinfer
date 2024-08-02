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

#include <nvbench/nvbench.cuh>

#include "flashinfer_ops.cuh"

using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

inline uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <bool append>
void bench_flashinfer_single_prefill_fp8(nvbench::state& state) {
  size_t kv_len = state.get_int64("kv_len");
  size_t qo_len = kv_len;
  if (append) {
    qo_len = state.get_int64("qo_len");
    if (qo_len > kv_len) {
      state.skip("qo_len > kv_len");
    }
  }
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  size_t pos_encoding_mode = state.get_int64("pos_encoding_mode");
  size_t kv_layout = state.get_int64("kv_layout");
  bool causal = state.get_int64("causal");
  bool cooperative = state.get_int64("cooperative");
  bool allow_fp16_qk_reduction = state.get_int64("allow_fp16_qk_reduction");
  // Allocate input data:
  thrust::device_vector<half> Q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<__nv_fp8_e4m3> K(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<__nv_fp8_e4m3> V(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<half> O(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<half> tmp(16 * 1024 * 1024);

  // Provide throughput information:
  state.add_global_memory_reads<uint8_t>(
      (qo_len * num_qo_heads * sizeof(half) + 2 * kv_len * num_kv_heads) * head_dim, "Read");
  state.add_global_memory_writes<half>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status;
    status = flashinfer::SinglePrefillWithKVCache<half, __nv_fp8_e4m3, half>(
        thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
        thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
        /*tmp=*/cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr,
        /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal,
        QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction,
        /*maybe_sm_scale=*/std::nullopt,
        /*rope_scale=*/1.f,
        /*rope_theta=*/1e4, launch.get_stream());
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
    tflops = qo_len * (2 * kv_len - qo_len) * 2 * num_qo_heads * head_dim / measured_mean / 1e12;
  } else {
    tflops = qo_len * kv_len * 4 * num_qo_heads * head_dim / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

template <typename dtype_in, typename dtype_out, bool append>
void bench_flashinfer_single_prefill(nvbench::state& state) {
  size_t kv_len = state.get_int64("kv_len");
  size_t qo_len = kv_len;
  if (append) {
    qo_len = state.get_int64("qo_len");
    if (qo_len > kv_len) {
      state.skip("qo_len > kv_len");
    }
  }
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  size_t pos_encoding_mode = state.get_int64("pos_encoding_mode");
  size_t kv_layout = state.get_int64("kv_layout");
  bool causal = state.get_int64("causal");
  bool cooperative = state.get_int64("cooperative");
  bool custom_mask = state.get_int64("custom_mask");
  bool allow_fp16_qk_reduction = state.get_int64("allow_fp16_qk_reduction");
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<dtype_in> K(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_in> V(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<uint8_t> mask(ceil_div(qo_len * kv_len, 8));
  thrust::device_vector<dtype_out> O(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<dtype_out> tmp(16 * 1024 * 1024);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(
      (qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * head_dim, "Read");
  state.add_global_memory_writes<dtype_out>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status;
    if (custom_mask) {
      status = flashinfer::SinglePrefillWithKVCacheCustomMask<dtype_in, dtype_out>(
          thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
          thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(mask.data()),
          thrust::raw_pointer_cast(O.data()),
          /*tmp=*/cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr,
          /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim,
          QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction,
          /*maybe_sm_scale=*/std::nullopt,
          /*rope_scale=*/1.f,
          /*rope_theta=*/1e4, launch.get_stream());
    } else {
      status = flashinfer::SinglePrefillWithKVCache<dtype_in, dtype_in, dtype_out>(
          thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
          thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
          /*tmp=*/cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr,
          /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal,
          QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction,
          /*maybe_sm_scale=*/std::nullopt,
          /*rope_scale=*/1.f,
          /*rope_theta=*/1e4, launch.get_stream());
    }
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
    tflops = qo_len * (2 * kv_len - qo_len) * 2 * num_qo_heads * head_dim / measured_mean / 1e12;
  } else {
    tflops = qo_len * kv_len * 4 * num_qo_heads * head_dim / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_PREFILL(dtype_in, dtype_out)                                       \
  auto bench_flashinfer_single_prefill_##dtype_in##_##dtype_out##_ =                        \
      bench_flashinfer_single_prefill<dtype_in, dtype_out, false>;                          \
  NVBENCH_BENCH(bench_flashinfer_single_prefill_##dtype_in##_##dtype_out##_)                \
      .set_name(("bench_flashinfer_single_prefill_" STR(dtype_in) "_" STR(dtype_out)))      \
      .add_int64_axis("kv_len",                                                             \
                      {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) \
      .add_int64_axis("num_qo_heads", {32})                                                 \
      .add_int64_axis("num_kv_heads", {32})                                                 \
      .add_int64_axis("head_dim", {128})                                                    \
      .add_int64_axis("causal", {0, 1})                                                     \
      .add_int64_axis("kv_layout", {0, 1})                                                  \
      .add_int64_axis("pos_encoding_mode", {0, 1})                                          \
      .add_int64_axis("allow_fp16_qk_reduction", {0, 1})                                    \
      .add_int64_axis("custom_mask", {0})                                                   \
      .add_int64_axis("cooperative", {1})

auto bench_flashinfer_single_prefill_fp8_kv = bench_flashinfer_single_prefill_fp8<false>;
NVBENCH_BENCH(bench_flashinfer_single_prefill_fp8_kv)
    .set_name(("bench_flashinfer_single_prefill_fp8_kv"))
    .add_int64_axis("kv_len", {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536})
    .add_int64_axis("num_qo_heads", {32})
    .add_int64_axis("num_kv_heads", {32})
    .add_int64_axis("head_dim", {128})
    .add_int64_axis("causal", {0, 1})
    .add_int64_axis("kv_layout", {0, 1})
    .add_int64_axis("pos_encoding_mode", {0, 1})
    .add_int64_axis("allow_fp16_qk_reduction", {0, 1})
    .add_int64_axis("custom_mask", {0})
    .add_int64_axis("cooperative", {1});

#define BENCH_FLASHINFER_APPEND_PREFILL(dtype_in, dtype_out)                                  \
  auto bench_flashinfer_single_append_prefill_##dtype_in##_##dtype_out##_ =                   \
      bench_flashinfer_single_prefill<dtype_in, dtype_out, true>;                             \
  NVBENCH_BENCH(bench_flashinfer_single_append_prefill_##dtype_in##_##dtype_out##_)           \
      .set_name(("bench_flashinfer_single_append_prefill_" STR(dtype_in) "_" STR(dtype_out))) \
      .add_int64_axis("qo_len", {128})                                                        \
      .add_int64_axis("kv_len", {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) \
      .add_int64_axis("num_qo_heads", {32})                                                   \
      .add_int64_axis("num_kv_heads", {32})                                                   \
      .add_int64_axis("head_dim", {128})                                                      \
      .add_int64_axis("causal", {0, 1})                                                       \
      .add_int64_axis("kv_layout", {0, 1})                                                    \
      .add_int64_axis("pos_encoding_mode", {0, 1})                                            \
      .add_int64_axis("allow_fp16_qk_reduction", {0, 1})                                      \
      .add_int64_axis("custom_mask", {0})                                                     \
      .add_int64_axis("cooperative", {0, 1})

BENCH_FLASHINFER_PREFILL(half, half);
BENCH_FLASHINFER_APPEND_PREFILL(half, half);
