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
#include <flashinfer/attention/cascade.cuh>
#include <nvbench/nvbench.cuh>

#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void bench_merge_states(nvbench::state& state) {
  const auto num_index_sets = state.get_int64("num_index_sets");
  const auto seq_len = state.get_int64("seq_len");
  const auto num_heads = state.get_int64("num_heads");
  const auto head_dim = state.get_int64("head_dim");

  std::vector<T> V_host(seq_len * num_index_sets * num_heads * head_dim);
  std::vector<float> S_host(seq_len * num_index_sets * num_heads);

  utils::vec_normal_(V_host);
  utils::vec_uniform_(S_host, 5, 10);

  thrust::device_vector<T> V_device(V_host);
  thrust::device_vector<float> S_device(S_host);
  thrust::device_vector<T> V_merged(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged(seq_len * num_heads);

  state.add_global_memory_reads<T>(V_host.size(), "Read");
  state.add_global_memory_writes<T>(V_merged.size(), "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = MergeStates(
        thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
        thrust::raw_pointer_cast(V_merged.data()), thrust::raw_pointer_cast(S_merged.data()),
        num_index_sets, seq_len, num_heads, head_dim);
    timer.stop();
  });
}

template <typename T>
void bench_two_level_single_prefix_cascade_decode(nvbench::state& state) {
  const auto batch_size = state.get_int64("batch_size");
  const auto shared_prefix_length = state.get_int64("shared_prefix_length");
  const auto unique_kv_length = state.get_int64("unique_kv_length");
  const auto num_kv_heads = state.get_int64("num_kv_heads");
  const auto num_qo_heads = state.get_int64("num_qo_heads");
  const auto use_cascade = state.get_int64("use_cascade");
  const auto head_dim = state.get_int64("head_dim");

  constexpr uint32_t page_size = 16;

  auto [testcase_float_data, testcase_int_data] = utils::create_shared_prefix_testcase_data<T>(
      batch_size, shared_prefix_length, unique_kv_length,
      /*qo_append_length=*/1, num_qo_heads, num_kv_heads, head_dim, page_size);

  std::vector<T> q_h = std::move(testcase_float_data[0]),
                 shared_k_h = std::move(testcase_float_data[1]),
                 shared_v_h = std::move(testcase_float_data[2]),
                 kv_data_h = std::move(testcase_float_data[3]);

  std::vector<int32_t> kv_indices_combined_h = std::move(testcase_int_data[1]),
                       kv_indices_unique_h = std::move(testcase_int_data[2]),
                       kv_indptr_combined_h = std::move(testcase_int_data[3]),
                       kv_indptr_unique_h = std::move(testcase_int_data[4]),
                       kv_last_page_len_combined_h = std::move(testcase_int_data[5]),
                       kv_last_page_len_unique_h = std::move(testcase_int_data[6]);

  thrust::device_vector<T> kv_data_d(kv_data_h);
  thrust::device_vector<T> q_d(q_h);
  constexpr PageStorage page_storage = PageStorage::kIndices;

  state.add_global_memory_reads<T>(kv_data_h.size() + q_h.size(), "Read");
  state.add_global_memory_writes<T>(q_h.size(), "Write");

  if (use_cascade) {
    thrust::device_vector<T> shared_k_d(shared_k_h), shared_v_d(shared_v_h),
        o_cascade_0_d(q_h.size()), o_cascade_1_d(q_h.size());
    thrust::device_vector<T> tmp_0_d(16 * 1024 * 1024);
    thrust::device_vector<float> lse_cascade_0_d(batch_size * num_qo_heads),
        lse_cascade_1_d(batch_size * num_qo_heads);
    thrust::device_vector<int32_t> kv_indptr_unique_d(kv_indptr_unique_h),
        kv_indices_unique_d(kv_indices_unique_h),
        kv_last_page_len_unique_d(kv_last_page_len_unique_h);
    paged_kv_t<page_storage, T, int32_t> paged_kv_casacde_d(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data_d.data()),
        thrust::raw_pointer_cast(kv_indices_unique_d.data()),
        thrust::raw_pointer_cast(kv_indptr_unique_d.data()),
        thrust::raw_pointer_cast(kv_last_page_len_unique_d.data()));
    BatchDecodeHandler cascade_handler;
    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
    BatchDecodeHandlerBeginForward<page_storage, T, T, T, int32_t>(
        &cascade_handler, (void*)thrust::raw_pointer_cast(float_buffer.data()),
        float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(int_buffer.data()),
        int_workspace_size_in_bytes, kv_indptr_unique_h.data(), kv_last_page_len_unique_h.data(),
        batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, PosEncodingMode::kNone);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      cudaError_t status = SinglePrefillWithKVCache(
          thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(shared_k_d.data()),
          thrust::raw_pointer_cast(shared_v_d.data()),
          thrust::raw_pointer_cast(o_cascade_0_d.data()), thrust::raw_pointer_cast(tmp_0_d.data()),
          thrust::raw_pointer_cast(lse_cascade_0_d.data()), num_qo_heads, num_kv_heads,
          /*qo_len=*/batch_size, /*kv_len=*/shared_prefix_length, head_dim,
          /*causal=*/false, /*kv_layout=*/QKVLayout::kNHD,
          /*pos_encoding_mode=*/PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

      if (status != cudaSuccess) {
        state.skip("Cascade implementation prefill failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }

      status = BatchDecodeWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
          &cascade_handler, thrust::raw_pointer_cast(q_d.data()),
          /*q_offset=*/nullptr, paged_kv_casacde_d, thrust::raw_pointer_cast(o_cascade_1_d.data()),
          /*lse=*/thrust::raw_pointer_cast(lse_cascade_1_d.data()), num_qo_heads,
          PosEncodingMode::kNone);

      if (status != cudaSuccess) {
        state.skip("Cascade implementation decode failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }

      status = MergeStateInPlace(thrust::raw_pointer_cast(o_cascade_0_d.data()),
                                 thrust::raw_pointer_cast(lse_cascade_0_d.data()),
                                 thrust::raw_pointer_cast(o_cascade_1_d.data()),
                                 thrust::raw_pointer_cast(lse_cascade_1_d.data()), batch_size,
                                 num_qo_heads, head_dim);

      if (status != cudaSuccess) {
        state.skip("Cascade implementation merge failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }
      timer.stop();
    });
  } else {
    thrust::device_vector<T> o_baseline_d(q_h.size());
    thrust::device_vector<int32_t> kv_indptr_combined_d(kv_indptr_combined_h),
        kv_indices_combined_d(kv_indices_combined_h),
        kv_last_page_len_combined_d(kv_last_page_len_combined_h);
    paged_kv_t<page_storage, T, int32_t> paged_kv_baseline_d(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data_d.data()),
        thrust::raw_pointer_cast(kv_indices_combined_d.data()),
        thrust::raw_pointer_cast(kv_indptr_combined_d.data()),
        thrust::raw_pointer_cast(kv_last_page_len_combined_d.data()));
    BatchDecodeHandler baseline_handler;
    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
    BatchDecodeHandlerBeginForward<page_storage, T, T, T, int32_t>(
        &baseline_handler, (void*)thrust::raw_pointer_cast(float_buffer.data()),
        float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(int_buffer.data()),
        int_workspace_size_in_bytes, kv_indptr_combined_h.data(),
        kv_last_page_len_combined_h.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
        page_size, PosEncodingMode::kNone);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      cudaError_t status = BatchDecodeWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
          &baseline_handler, thrust::raw_pointer_cast(q_d.data()),
          /*q_offset=*/nullptr, paged_kv_baseline_d, thrust::raw_pointer_cast(o_baseline_d.data()),
          /*lse=*/nullptr, num_qo_heads, PosEncodingMode::kNone);
      if (status != cudaSuccess) {
        state.skip("Cascade implementation decode failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }
      timer.stop();
    });
  }
}

template <typename T>
void bench_two_level_single_prefix_cascade_append(nvbench::state& state) {
  const auto batch_size = state.get_int64("batch_size");
  const auto shared_prefix_length = state.get_int64("shared_prefix_length");
  const auto unique_kv_length = state.get_int64("unique_kv_length");
  const auto qo_append_length = state.get_int64("qo_append_length");
  const auto num_kv_heads = state.get_int64("num_kv_heads");
  const auto num_qo_heads = state.get_int64("num_qo_heads");
  const auto use_cascade = state.get_int64("use_cascade");
  const auto head_dim = state.get_int64("head_dim");

  constexpr uint32_t page_size = 16;

  auto [testcase_float_data, testcase_int_data] = utils::create_shared_prefix_testcase_data<T>(
      batch_size, shared_prefix_length, unique_kv_length, qo_append_length, num_qo_heads,
      num_kv_heads, head_dim, page_size);

  std::vector<T> q_h = std::move(testcase_float_data[0]),
                 shared_k_h = std::move(testcase_float_data[1]),
                 shared_v_h = std::move(testcase_float_data[2]),
                 kv_data_h = std::move(testcase_float_data[3]);

  std::vector<int32_t> qo_indptr_h = std::move(testcase_int_data[0]),
                       kv_indices_combined_h = std::move(testcase_int_data[1]),
                       kv_indices_unique_h = std::move(testcase_int_data[2]),
                       kv_indptr_combined_h = std::move(testcase_int_data[3]),
                       kv_indptr_unique_h = std::move(testcase_int_data[4]),
                       kv_last_page_len_combined_h = std::move(testcase_int_data[5]),
                       kv_last_page_len_unique_h = std::move(testcase_int_data[6]);

  thrust::device_vector<T> kv_data_d(kv_data_h);
  thrust::device_vector<T> q_d(q_h);
  thrust::device_vector<int32_t> qo_indptr_d(qo_indptr_h);
  constexpr PageStorage page_storage = PageStorage::kIndices;

  state.add_global_memory_reads<T>(kv_data_h.size() + q_h.size(), "Read");
  state.add_global_memory_writes<T>(q_h.size(), "Write");

  if (use_cascade) {
    thrust::device_vector<T> shared_k_d(shared_k_h), shared_v_d(shared_v_h),
        o_cascade_0_d(q_h.size()), o_cascade_1_d(q_h.size());
    thrust::device_vector<T> tmp_0_d(8 * 1024 * 1024);
    thrust::device_vector<float> lse_cascade_0_d((batch_size * qo_append_length) * num_qo_heads),
        lse_cascade_1_d((batch_size * qo_append_length) * num_qo_heads);
    thrust::device_vector<int32_t> kv_indptr_unique_d(kv_indptr_unique_h),
        kv_indices_unique_d(kv_indices_unique_h),
        kv_last_page_len_unique_d(kv_last_page_len_unique_h);
    paged_kv_t<page_storage, T, int32_t> paged_kv_casacde_d(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data_d.data()),
        thrust::raw_pointer_cast(kv_indices_unique_d.data()),
        thrust::raw_pointer_cast(kv_indptr_unique_d.data()),
        thrust::raw_pointer_cast(kv_last_page_len_unique_d.data()));
    BatchPrefillHandler cascade_handler;
    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
    cascade_handler.BeginForward<T, int32_t>(
        (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
        (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
        qo_indptr_h.data(), kv_indptr_unique_h.data(), batch_size, num_qo_heads, num_kv_heads,
        head_dim, page_size);
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      cudaError_t status = SinglePrefillWithKVCache(
          thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(shared_k_d.data()),
          thrust::raw_pointer_cast(shared_v_d.data()),
          thrust::raw_pointer_cast(o_cascade_0_d.data()), thrust::raw_pointer_cast(tmp_0_d.data()),
          thrust::raw_pointer_cast(lse_cascade_0_d.data()), num_qo_heads, num_kv_heads,
          /*qo_len=*/batch_size * qo_append_length,
          /*kv_len=*/shared_prefix_length, head_dim,
          /*causal=*/false, /*kv_layout=*/QKVLayout::kNHD,
          /*pos_encoding_mode=*/PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

      if (status != cudaSuccess) {
        state.skip("Cascade implementation prefill failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }

      status = BatchPrefillWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
          &cascade_handler, thrust::raw_pointer_cast(q_d.data()),
          thrust::raw_pointer_cast(qo_indptr_d.data()),
          /*q_offset=*/nullptr, paged_kv_casacde_d, thrust::raw_pointer_cast(o_cascade_1_d.data()),
          thrust::raw_pointer_cast(lse_cascade_1_d.data()), num_qo_heads, /*causal=*/true,
          PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

      if (status != cudaSuccess) {
        state.skip("Cascade implementation unique kv prefill failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }

      status = MergeStateInPlace(thrust::raw_pointer_cast(o_cascade_0_d.data()),
                                 thrust::raw_pointer_cast(lse_cascade_0_d.data()),
                                 thrust::raw_pointer_cast(o_cascade_1_d.data()),
                                 thrust::raw_pointer_cast(lse_cascade_1_d.data()),
                                 batch_size * qo_append_length, num_qo_heads, head_dim);
      if (status != cudaSuccess) {
        state.skip("Cascade implementation merge failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }
      timer.stop();
    });
  } else {
    thrust::device_vector<T> o_baseline_d(q_h.size());
    thrust::device_vector<int32_t> kv_indptr_combined_d(kv_indptr_combined_h),
        kv_indices_combined_d(kv_indices_combined_h),
        kv_last_page_len_combined_d(kv_last_page_len_combined_h);
    paged_kv_t<page_storage, T, int32_t> paged_kv_baseline_d(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data_d.data()),
        thrust::raw_pointer_cast(kv_indices_combined_d.data()),
        thrust::raw_pointer_cast(kv_indptr_combined_d.data()),
        thrust::raw_pointer_cast(kv_last_page_len_combined_d.data()));
    BatchPrefillHandler baseline_handler;
    size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
    thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
    size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
    thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
    baseline_handler.BeginForward<T, int32_t>(
        (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
        (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
        qo_indptr_h.data(), kv_indptr_combined_h.data(), batch_size, num_qo_heads, num_kv_heads,
        head_dim, page_size);
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      timer.start();
      cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
          &baseline_handler, thrust::raw_pointer_cast(q_d.data()),
          thrust::raw_pointer_cast(qo_indptr_d.data()),
          /*q_offset=*/nullptr, paged_kv_baseline_d, thrust::raw_pointer_cast(o_baseline_d.data()),
          /*lse=*/nullptr, num_qo_heads, /*causal=*/true, PosEncodingMode::kNone,
          /*allow_fp16_qk_reduction=*/false);

      if (status != cudaSuccess) {
        state.skip("Baseline implementation failed with error: " +
                   std::string(cudaGetErrorString(status)));
      }
      timer.stop();
    });
  }
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_MERGE_KERNELS(T)                            \
  auto bench_flashinfer_merge_states_##T##_ = bench_merge_states<T>; \
  NVBENCH_BENCH(bench_flashinfer_merge_states_##T##_)                \
      .set_name("flashinfer_merge_states_" STR(T))                   \
      .add_int64_axis("num_index_sets", {2, 16, 64, 128, 256})       \
      .add_int64_axis("seq_len", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
      .add_int64_axis("num_heads", {32})                             \
      .add_int64_axis("head_dim", {128})

#define BENCH_FLASHINFER_TWO_LEVEL_SINGLE_PREFIX_CASCADE_DECODE_KERNELS(T)      \
  auto bench_flashinfer_two_level_single_prefix_cascade_decode_##T##_ =         \
      bench_two_level_single_prefix_cascade_decode<T>;                          \
  NVBENCH_BENCH(bench_flashinfer_two_level_single_prefix_cascade_decode_##T##_) \
      .set_name("flashinfer_two_level_single_prefix_cascade_decode_" STR(T))    \
      .add_int64_axis("batch_size", {1, 8, 16, 64, 128, 256})                   \
      .add_int64_axis("shared_prefix_length", {1024, 2048, 8192, 32768})        \
      .add_int64_axis("unique_kv_length", {128, 256, 512, 1024, 2048})          \
      .add_int64_axis("num_kv_heads", {32})                                     \
      .add_int64_axis("num_qo_heads", {32})                                     \
      .add_int64_axis("use_cascade", {1, 0})                                    \
      .add_int64_axis("head_dim", {128})

#define BENCH_FLASHINFER_TWO_LEVEL_SINGLE_PREFIX_CASCADE_APPEND_KERNELS(T)      \
  auto bench_flashinfer_two_level_single_prefix_cascade_append_##T##_ =         \
      bench_two_level_single_prefix_cascade_append<T>;                          \
  NVBENCH_BENCH(bench_flashinfer_two_level_single_prefix_cascade_append_##T##_) \
      .set_name("flashinfer_two_level_single_prefix_cascade_append_" STR(T))    \
      .add_int64_axis("batch_size", {1, 8, 16, 64, 128, 256})                   \
      .add_int64_axis("shared_prefix_length", {1024, 2048, 8192, 32768})        \
      .add_int64_axis("unique_kv_length", {128, 256, 512, 1024, 2048})          \
      .add_int64_axis("qo_append_length", {128})                                \
      .add_int64_axis("num_kv_heads", {32})                                     \
      .add_int64_axis("num_qo_heads", {32})                                     \
      .add_int64_axis("use_cascade", {1, 0})                                    \
      .add_int64_axis("head_dim", {128})

BENCH_FLASHINFER_MERGE_KERNELS(half);
BENCH_FLASHINFER_TWO_LEVEL_SINGLE_PREFIX_CASCADE_DECODE_KERNELS(half);
BENCH_FLASHINFER_TWO_LEVEL_SINGLE_PREFIX_CASCADE_APPEND_KERNELS(half);
