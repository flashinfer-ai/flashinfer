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
#include <gtest/gtest.h>

#include <flashinfer/attention/cascade.cuh>

#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;
constexpr QKVLayout kv_layout = QKVLayout::kHND;

bool is_prime(int x) {
  for (int i = 2; i < int(std::sqrt(x)); ++i) {
    if (x % i == 0) return false;
  }
  return true;
}

template <typename T>
void _TestVariableLengthMergeKernelCorrectness(size_t seq_len, size_t num_heads, size_t head_dim,
                                               bool sparse_s) {
  const uint32_t max_num_index_sets = 512;
  std::vector<int32_t> lengths(seq_len);
  utils::vec_randint_(lengths, 1, max_num_index_sets);
  std::vector<int32_t> indptr{0};
  for (size_t i = 0; i < seq_len; ++i) {
    indptr.push_back(indptr.back() + lengths[i]);
  }
  std::vector<T> V_padded_host(seq_len * max_num_index_sets * num_heads * head_dim);
  std::vector<T> V_ragged_host(indptr.back() * num_heads * head_dim);
  std::vector<float> S_padded_host(seq_len * max_num_index_sets * num_heads);
  std::vector<float> S_ragged_host(indptr.back() * num_heads);

  utils::vec_normal_(V_ragged_host);
  for (uint32_t j = 0; j < seq_len; ++j) {
    std::copy(V_ragged_host.begin() + indptr[j] * num_heads * head_dim,
              V_ragged_host.begin() + indptr[j + 1] * num_heads * head_dim,
              V_padded_host.begin() + j * max_num_index_sets * num_heads * head_dim);
  }
  if (sparse_s) {
    for (uint32_t i = 0; i < max_num_index_sets; ++i) {
      float fill_val = is_prime(i) ? 10 : -10;
      for (uint32_t j = 0; j < seq_len; ++j) {
        if (i < lengths[j]) {
          std::fill(S_ragged_host.begin() + (indptr[j] + i) * num_heads,
                    S_ragged_host.begin() + (indptr[j] + i + 1) * num_heads, fill_val);
          std::fill(S_padded_host.begin() + (j * max_num_index_sets + i) * num_heads,
                    S_padded_host.begin() + (j * max_num_index_sets + i + 1) * num_heads, fill_val);
        } else {
          std::fill(S_padded_host.begin() + (j * max_num_index_sets + i) * num_heads,
                    S_padded_host.begin() + (j * max_num_index_sets + i + 1) * num_heads, -5e4);
        }
      }
    }
  } else {
    utils::vec_uniform_(S_ragged_host, -10, 10);
    for (uint32_t j = 0; j < seq_len; ++j) {
      std::copy(S_ragged_host.begin() + indptr[j] * num_heads,
                S_ragged_host.begin() + indptr[j + 1] * num_heads,
                S_padded_host.begin() + (j * max_num_index_sets) * num_heads);
      std::fill(
          S_padded_host.begin() + (j * max_num_index_sets + indptr[j + 1] - indptr[j]) * num_heads,
          S_padded_host.begin() + (j + 1) * max_num_index_sets * num_heads, -5e4);
    }
  }

  thrust::device_vector<T> V_padded_device(V_padded_host);
  thrust::device_vector<T> V_ragged_device(V_ragged_host);
  thrust::device_vector<float> S_padded_device(S_padded_host);
  thrust::device_vector<float> S_ragged_device(S_ragged_host);
  thrust::device_vector<int32_t> indptr_device(indptr);
  thrust::device_vector<T> V_merged_0_device(seq_len * num_heads * head_dim);
  thrust::device_vector<T> V_merged_1_device(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged_0_device(seq_len * num_heads);
  thrust::device_vector<float> S_merged_1_device(seq_len * num_heads);

  // Method 0: use MergeStates on padded data
  MergeStates(thrust::raw_pointer_cast(V_padded_device.data()),
              thrust::raw_pointer_cast(S_padded_device.data()),
              thrust::raw_pointer_cast(V_merged_0_device.data()),
              thrust::raw_pointer_cast(S_merged_0_device.data()), max_num_index_sets, seq_len,
              num_heads, head_dim);

  // Method 1: use VariableLengthMergeStates on ragged data
  VariableLengthMergeStates(thrust::raw_pointer_cast(V_ragged_device.data()),
                            thrust::raw_pointer_cast(S_ragged_device.data()),
                            thrust::raw_pointer_cast(indptr_device.data()),
                            thrust::raw_pointer_cast(V_merged_1_device.data()),
                            thrust::raw_pointer_cast(S_merged_1_device.data()), seq_len, num_heads,
                            head_dim);

  thrust::host_vector<T> V_merged_0_host(V_merged_0_device), V_merged_1_host(V_merged_1_device);
  thrust::host_vector<float> S_merged_0_host(S_merged_0_device), S_merged_1_host(S_merged_1_device);

  // Compare results
  size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0, num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (size_t i = 0; i < seq_len * num_heads * head_dim; ++i) {
    EXPECT_FALSE(std::isnan(float(V_merged_0_host[i]))) << "V_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(V_merged_1_host[i]))) << "V_merged_1_host[" << i << "] is nan";
    num_V_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(V_merged_0_host[i]), float(V_merged_1_host[i]), 1e-3, 1e-3));
  }
  for (size_t i = 0; i < seq_len * num_heads; ++i) {
    EXPECT_FALSE(std::isnan(float(S_merged_0_host[i]))) << "S_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(S_merged_1_host[i]))) << "S_merged_1_host[" << i << "] is nan";
    num_S_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(S_merged_0_host[i]), float(S_merged_1_host[i]), 1e-3, 1e-3));
  }
  float V_result_accuracy =
      1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads * head_dim);
  float S_result_accuracy =
      1.0 - float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads);
  std::cout << "seq_len=" << seq_len << ", num_heads=" << num_heads << ", head_dim=" << head_dim
            << ", sparse_s=" << sparse_s
            << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
            << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy << std::endl;

  EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
  EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";
}

template <typename T>
void _TestMergeKernelCorrectness(size_t num_index_sets, size_t seq_len, size_t num_heads,
                                 size_t head_dim, bool sparse_s) {
  std::vector<T> V_host(seq_len * num_index_sets * num_heads * head_dim);
  std::vector<float> V_host_trans_f32(num_index_sets * seq_len * num_heads * head_dim);
  std::vector<float> S_host(seq_len * num_index_sets * num_heads);
  std::vector<float> S_host_trans(num_index_sets * seq_len * num_heads);

  utils::vec_normal_(V_host);
  if (sparse_s) {
    for (uint32_t i = 0; i < num_index_sets; ++i) {
      float fill_val = is_prime(i) ? 10 : -10;
      for (uint32_t j = 0; j < seq_len; ++j) {
        for (uint32_t k = 0; k < num_heads; ++k) {
          S_host[(j * num_index_sets + i) * num_heads + k] = fill_val;
        }
      }
    }
  } else {
    utils::vec_uniform_(S_host, -10, 10);
  }

  for (uint32_t i = 0; i < num_index_sets; ++i) {
    for (uint32_t j = 0; j < seq_len; ++j) {
      std::transform(V_host.begin() + (j * num_index_sets + i) * num_heads * head_dim,
                     V_host.begin() + (j * num_index_sets + i + 1) * num_heads * head_dim,
                     V_host_trans_f32.begin() + (i * seq_len + j) * num_heads * head_dim,
                     [](T x) { return static_cast<float>(x); });
      std::copy(S_host.begin() + (j * num_index_sets + i) * num_heads,
                S_host.begin() + (j * num_index_sets + i + 1) * num_heads,
                S_host_trans.begin() + (i * seq_len + j) * num_heads);
    }
  }

  thrust::device_vector<T> V_device(V_host);
  thrust::device_vector<float> V_device_trans_f32(V_host_trans_f32);
  thrust::device_vector<float> S_device(S_host);
  thrust::device_vector<float> S_device_trans(S_host_trans);

  thrust::device_vector<float> V_merged_0_device(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged_0_device(seq_len * num_heads);
  thrust::device_vector<T> V_merged_1_device(seq_len * num_heads * head_dim);
  thrust::device_vector<float> S_merged_1_device(seq_len * num_heads);

  if (num_index_sets > 1) {
    // Method 0: use MergeState
    MergeState(thrust::raw_pointer_cast(V_device_trans_f32.data()),
               thrust::raw_pointer_cast(S_device_trans.data()),
               thrust::raw_pointer_cast(V_device_trans_f32.data() + seq_len * num_heads * head_dim),
               thrust::raw_pointer_cast(S_device_trans.data() + seq_len * num_heads),
               thrust::raw_pointer_cast(V_merged_0_device.data()),
               thrust::raw_pointer_cast(S_merged_0_device.data()), seq_len, num_heads, head_dim);
    for (uint i = 2; i < num_index_sets; ++i) {
      MergeStateInPlace(
          thrust::raw_pointer_cast(V_merged_0_device.data()),
          thrust::raw_pointer_cast(S_merged_0_device.data()),
          thrust::raw_pointer_cast(V_device_trans_f32.data() + i * seq_len * num_heads * head_dim),
          thrust::raw_pointer_cast(S_device_trans.data() + i * seq_len * num_heads), seq_len,
          num_heads, head_dim);
    }
  } else {
    V_merged_0_device = V_device;
    S_merged_0_device = S_device;
  }

  // Method 1: use MergeStates
  MergeStates(thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
              thrust::raw_pointer_cast(V_merged_1_device.data()),
              thrust::raw_pointer_cast(S_merged_1_device.data()), num_index_sets, seq_len,
              num_heads, head_dim);

  thrust::host_vector<float> V_merged_0_host(V_merged_0_device);
  thrust::host_vector<T> V_merged_1_host(V_merged_1_device);
  thrust::host_vector<float> S_merged_0_host(S_merged_0_device), S_merged_1_host(S_merged_1_device);
  size_t num_V_result_errors_atol_1e_3_rtol_1e_3 = 0, num_S_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (size_t i = 0; i < seq_len * num_heads * head_dim; ++i) {
    EXPECT_FALSE(std::isnan(float(V_merged_0_host[i]))) << "V_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(V_merged_1_host[i]))) << "V_merged_1_host[" << i << "] is nan";
    num_V_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(V_merged_0_host[i]), float(V_merged_1_host[i]), 1e-3, 1e-3));
  }
  for (size_t i = 0; i < seq_len * num_heads; ++i) {
    EXPECT_FALSE(std::isnan(float(S_merged_0_host[i]))) << "S_merged_0_host[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(S_merged_1_host[i]))) << "S_merged_1_host[" << i << "] is nan";
    num_S_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(S_merged_0_host[i]), float(S_merged_1_host[i]), 1e-3, 1e-3));
  }
  float V_result_accuracy =
      1.0 - float(num_V_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads * head_dim);
  float S_result_accuracy =
      1.0 - float(num_S_result_errors_atol_1e_3_rtol_1e_3) / (seq_len * num_heads);
  std::cout << "num_index_sets=" << num_index_sets << ", seq_len=" << seq_len
            << ", num_heads=" << num_heads << ", head_dim=" << head_dim << ", sparse_s=" << sparse_s
            << ", V accuracy (atol=1e-3, rtol=1e-3)=" << V_result_accuracy
            << ", S accuracy (atol=1e-3, rtol=1e-3)=" << S_result_accuracy << std::endl;
  EXPECT_GT(V_result_accuracy, 0.99) << "V result correctness test failed.";
  EXPECT_GT(S_result_accuracy, 0.99) << "S result correctness test failed.";
}

template <typename T>
void _TestTwoLevelSinglePrefixCascadeDecodeCorrectness(size_t batch_size,
                                                       size_t shared_prefix_length,
                                                       size_t unique_kv_length, size_t num_qo_heads,
                                                       size_t num_kv_heads, size_t head_dim) {
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

  thrust::device_vector<T> shared_k_d(shared_k_h), shared_v_d(shared_v_h), kv_data_d(kv_data_h),
      q_d(q_h), o_baseline_d(q_h.size()), o_cascade_0_d(q_h.size()), o_cascade_1_d(q_h.size());
  thrust::device_vector<T> tmp_0_d(16 * 1024 * 1024);
  thrust::device_vector<float> lse_cascade_0_d(batch_size * num_qo_heads),
      lse_cascade_1_d(batch_size * num_qo_heads);

  thrust::device_vector<int32_t> kv_indptr_combined_d(kv_indptr_combined_h),
      kv_indptr_unique_d(kv_indptr_unique_h), kv_indices_combined_d(kv_indices_combined_h),
      kv_indices_unique_d(kv_indices_unique_h),
      kv_last_page_len_combined_d(kv_last_page_len_combined_h),
      kv_last_page_len_unique_d(kv_last_page_len_unique_h);

  constexpr PageStorage page_storage = PageStorage::kIndices;

  paged_kv_t<page_storage, T, int32_t> paged_kv_baseline_d(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data_d.data()),
      thrust::raw_pointer_cast(kv_indices_combined_d.data()),
      thrust::raw_pointer_cast(kv_indptr_combined_d.data()),
      thrust::raw_pointer_cast(kv_last_page_len_combined_d.data()));

  paged_kv_t<page_storage, T, int32_t> paged_kv_casacde_d(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data_d.data()),
      thrust::raw_pointer_cast(kv_indices_unique_d.data()),
      thrust::raw_pointer_cast(kv_indptr_unique_d.data()),
      thrust::raw_pointer_cast(kv_last_page_len_unique_d.data()));

  BatchDecodeHandler baseline_handler, cascade_handler;

  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  BatchDecodeHandlerBeginForward<page_storage, T, T, T, int32_t>(
      &baseline_handler, (void*)thrust::raw_pointer_cast(float_buffer.data()),
      float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(int_buffer.data()),
      int_workspace_size_in_bytes, kv_indptr_combined_h.data(), kv_last_page_len_combined_h.data(),
      batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, PosEncodingMode::kNone);

  BatchDecodeHandlerBeginForward<page_storage, T, T, T, int32_t>(
      &cascade_handler, (void*)thrust::raw_pointer_cast(float_buffer.data()),
      float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(int_buffer.data()),
      int_workspace_size_in_bytes, kv_indptr_unique_h.data(), kv_last_page_len_unique_h.data(),
      batch_size, num_qo_heads, num_kv_heads, head_dim, page_size, PosEncodingMode::kNone);

  // Compute result using baseline implementation
  cudaError_t status = BatchDecodeWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
      &baseline_handler, thrust::raw_pointer_cast(q_d.data()),
      /*q_offset=*/nullptr, paged_kv_baseline_d, thrust::raw_pointer_cast(o_baseline_d.data()),
      /*lse=*/nullptr, num_qo_heads, PosEncodingMode::kNone);

  EXPECT_EQ(status, cudaSuccess) << "Baseline implementation failed with error: "
                                 << cudaGetErrorString(status);

  // Compute result using cascade implementation
  status = SinglePrefillWithKVCache(
      thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(shared_k_d.data()),
      thrust::raw_pointer_cast(shared_v_d.data()), thrust::raw_pointer_cast(o_cascade_0_d.data()),
      thrust::raw_pointer_cast(tmp_0_d.data()), thrust::raw_pointer_cast(lse_cascade_0_d.data()),
      num_qo_heads, num_kv_heads, /*qo_len=*/batch_size, /*kv_len=*/shared_prefix_length, head_dim,
      /*causal=*/false, /*kv_layout=*/QKVLayout::kNHD,
      /*pos_encoding_mode=*/PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

  EXPECT_EQ(status, cudaSuccess) << "Cascade implementation prefill failed with error: "
                                 << cudaGetErrorString(status);

  status = BatchDecodeWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
      &cascade_handler, thrust::raw_pointer_cast(q_d.data()),
      /*q_offset=*/nullptr, paged_kv_casacde_d, thrust::raw_pointer_cast(o_cascade_1_d.data()),
      /*lse=*/thrust::raw_pointer_cast(lse_cascade_1_d.data()), num_qo_heads,
      PosEncodingMode::kNone);

  EXPECT_EQ(status, cudaSuccess) << "Cascade implementation decode failed with error: "
                                 << cudaGetErrorString(status);

  status = MergeStateInPlace(thrust::raw_pointer_cast(o_cascade_0_d.data()),
                             thrust::raw_pointer_cast(lse_cascade_0_d.data()),
                             thrust::raw_pointer_cast(o_cascade_1_d.data()),
                             thrust::raw_pointer_cast(lse_cascade_1_d.data()), batch_size,
                             num_qo_heads, head_dim);

  EXPECT_EQ(status, cudaSuccess) << "Cascade implementation merge failed with error: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<T> o_baseline_h(o_baseline_d), o_cascade_h(o_cascade_0_d);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (size_t i = 0; i < o_baseline_h.size(); ++i) {
    EXPECT_FALSE(std::isnan(float(o_baseline_h[i]))) << "o_baseline_h[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(o_cascade_h[i]))) << "o_cascade_h[" << i << "] is nan";
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_baseline_h[i]), float(o_cascade_h[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(o_baseline_h.size());
  std::cout << "batch_size=" << batch_size << ", shared_prefix_length=" << shared_prefix_length
            << ", unique_kv_length=" << unique_kv_length << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", head_dim=" << head_dim
            << ", result_accuracy (atol=1e-3, rtol=1e-3)=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
}

template <typename T>
void _TestTwoLevelSinglePrefixCascadeAppendCorrectness(size_t batch_size,
                                                       size_t shared_prefix_length,
                                                       size_t unique_kv_length,
                                                       size_t qo_append_length, size_t num_qo_heads,
                                                       size_t num_kv_heads, size_t head_dim) {
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

  thrust::device_vector<T> shared_k_d(shared_k_h), shared_v_d(shared_v_h), kv_data_d(kv_data_h),
      q_d(q_h), o_baseline_d(q_h.size()), o_cascade_0_d(q_h.size()), o_cascade_1_d(q_h.size());
  thrust::device_vector<T> tmp_0_d(16 * 1024 * 1024);
  thrust::device_vector<float> lse_cascade_0_d((batch_size * qo_append_length) * num_qo_heads),
      lse_cascade_1_d((batch_size * qo_append_length) * num_qo_heads);

  thrust::device_vector<int32_t> qo_indptr_d(qo_indptr_h),
      kv_indptr_combined_d(kv_indptr_combined_h), kv_indptr_unique_d(kv_indptr_unique_h),
      kv_indices_combined_d(kv_indices_combined_h), kv_indices_unique_d(kv_indices_unique_h),
      kv_last_page_len_combined_d(kv_last_page_len_combined_h),
      kv_last_page_len_unique_d(kv_last_page_len_unique_h);

  constexpr PageStorage page_storage = PageStorage::kIndices;

  paged_kv_t<page_storage, T, int32_t> paged_kv_baseline_d(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data_d.data()),
      thrust::raw_pointer_cast(kv_indices_combined_d.data()),
      thrust::raw_pointer_cast(kv_indptr_combined_d.data()),
      thrust::raw_pointer_cast(kv_last_page_len_combined_d.data()));

  paged_kv_t<page_storage, T, int32_t> paged_kv_casacde_d(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data_d.data()),
      thrust::raw_pointer_cast(kv_indices_unique_d.data()),
      thrust::raw_pointer_cast(kv_indptr_unique_d.data()),
      thrust::raw_pointer_cast(kv_last_page_len_unique_d.data()));

  BatchPrefillHandler baseline_handler, cascade_handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  baseline_handler.BeginForward<T, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      qo_indptr_h.data(), kv_indptr_combined_h.data(), batch_size, num_qo_heads, num_kv_heads,
      head_dim, page_size);
  cascade_handler.BeginForward<T, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      qo_indptr_h.data(), kv_indptr_unique_h.data(), batch_size, num_qo_heads, num_kv_heads,
      head_dim, page_size);

  cudaError_t status = BatchPrefillWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
      &baseline_handler, thrust::raw_pointer_cast(q_d.data()),
      thrust::raw_pointer_cast(qo_indptr_d.data()),
      /*q_offset=*/nullptr, paged_kv_baseline_d, thrust::raw_pointer_cast(o_baseline_d.data()),
      /*lse=*/nullptr, num_qo_heads, /*causal=*/true, PosEncodingMode::kNone,
      /*allow_fp16_qk_reduction=*/false);

  EXPECT_EQ(status, cudaSuccess) << "Baseline implementation failed with error: "
                                 << cudaGetErrorString(status);

  status = SinglePrefillWithKVCache(
      thrust::raw_pointer_cast(q_d.data()), thrust::raw_pointer_cast(shared_k_d.data()),
      thrust::raw_pointer_cast(shared_v_d.data()), thrust::raw_pointer_cast(o_cascade_0_d.data()),
      thrust::raw_pointer_cast(tmp_0_d.data()), thrust::raw_pointer_cast(lse_cascade_0_d.data()),
      num_qo_heads, num_kv_heads, /*qo_len=*/batch_size * qo_append_length,
      /*kv_len=*/shared_prefix_length, head_dim,
      /*causal=*/false, /*kv_layout=*/QKVLayout::kNHD,
      /*pos_encoding_mode=*/PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

  EXPECT_EQ(status, cudaSuccess)
      << "Cascade implementation shared prefix prefill failed with error: "
      << cudaGetErrorString(status);

  status = BatchPrefillWithPagedKVCacheWrapper<page_storage, T, T, T, int32_t>(
      &cascade_handler, thrust::raw_pointer_cast(q_d.data()),
      thrust::raw_pointer_cast(qo_indptr_d.data()),
      /*r_rope_position=*/nullptr, paged_kv_casacde_d,
      thrust::raw_pointer_cast(o_cascade_1_d.data()),
      thrust::raw_pointer_cast(lse_cascade_1_d.data()), num_qo_heads, /*causal=*/true,
      PosEncodingMode::kNone, /*allow_fp16_qk_reduction=*/false);

  EXPECT_EQ(status, cudaSuccess) << "Cascade implementation unique kv prefill failed with error: "
                                 << cudaGetErrorString(status);

  status = MergeStateInPlace(thrust::raw_pointer_cast(o_cascade_0_d.data()),
                             thrust::raw_pointer_cast(lse_cascade_0_d.data()),
                             thrust::raw_pointer_cast(o_cascade_1_d.data()),
                             thrust::raw_pointer_cast(lse_cascade_1_d.data()),
                             batch_size * qo_append_length, num_qo_heads, head_dim);
  EXPECT_EQ(status, cudaSuccess) << "Cascade implementation merge failed with error: "
                                 << cudaGetErrorString(status);

  thrust::host_vector<T> o_baseline_h(o_baseline_d), o_cascade_h(o_cascade_0_d);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  for (size_t i = 0; i < o_baseline_h.size(); ++i) {
    EXPECT_FALSE(std::isnan(float(o_baseline_h[i]))) << "o_baseline_h[" << i << "] is nan";
    EXPECT_FALSE(std::isnan(float(o_cascade_h[i]))) << "o_cascade_h[" << i << "] is nan";
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_baseline_h[i]), float(o_cascade_h[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(o_baseline_h.size());
  std::cout << "batch_size=" << batch_size << ", shared_prefix_length=" << shared_prefix_length
            << ", unique_kv_length=" << unique_kv_length
            << ", qo_append_length=" << qo_append_length << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", head_dim=" << head_dim
            << ", result_accuracy (atol=1e-3, rtol=1e-3)=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
}

template <typename T>
void TestMergeKernelCorrectness() {
  for (size_t num_index_sets : {1, 2, 9, 81, 513}) {
    for (size_t seq_len : {4, 16, 77}) {
      for (size_t num_heads : {1, 21, 32}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (bool sparse_s : {false, true}) {
            _TestMergeKernelCorrectness<T>(num_index_sets, seq_len, num_heads, head_dim, sparse_s);
          }
        }
      }
    }
  }
}

template <typename T>
void TestVariableLengthMergeKernelCorrectness() {
  for (size_t seq_len : {1, 3, 77, 191}) {
    for (size_t num_heads : {1, 4, 32}) {
      for (size_t head_dim : {64, 128, 256}) {
        for (bool sparse_s : {false, true}) {
          _TestVariableLengthMergeKernelCorrectness<T>(seq_len, num_heads, head_dim, sparse_s);
        }
      }
    }
  }
}

template <typename T>
void TestTwoLevelSinglePrefixCascadeDecodeCorrectness() {
  for (size_t batch_size : {1, 8, 16, 64, 128}) {
    for (size_t shared_prefix_length : {1024, 2048, 8192, 32768}) {
      for (size_t unique_kv_length : {128, 256, 512, 1024}) {
        for (size_t num_qo_heads : {32}) {
          for (size_t num_kv_heads : {32}) {
            for (size_t head_dim : {128}) {
              _TestTwoLevelSinglePrefixCascadeDecodeCorrectness<T>(batch_size, shared_prefix_length,
                                                                   unique_kv_length, num_qo_heads,
                                                                   num_kv_heads, head_dim);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void TestTwoLevelSinglePrefixCascadeAppendCorrectness() {
  for (size_t batch_size : {1, 8, 16, 64, 128}) {
    for (size_t shared_prefix_length : {1024, 2048, 8192, 32768}) {
      for (size_t unique_kv_length : {128, 256, 512, 1024}) {
        for (size_t qo_append_length : {128}) {
          for (size_t num_qo_heads : {32}) {
            for (size_t num_kv_heads : {32}) {
              for (size_t head_dim : {128}) {
                _TestTwoLevelSinglePrefixCascadeAppendCorrectness<T>(
                    batch_size, shared_prefix_length, unique_kv_length, qo_append_length,
                    num_qo_heads, num_kv_heads, head_dim);
              }
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, MergeKernelCorrectnessTestFP16) {
  TestMergeKernelCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, VariableLengthMergeKernelCorrectnessTestFP16) {
  TestVariableLengthMergeKernelCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, TwoLevelSinglePrefixCascadeDecodeTestFP16) {
  TestTwoLevelSinglePrefixCascadeDecodeCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, TwoLevelSinglePrefixCascadeAppendTestFP16) {
  TestTwoLevelSinglePrefixCascadeAppendCorrectness<half>();
}
