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

#include <cstdint>

#include "cpu_reference.h"
#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;
constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename T>
void _TestBatchPagedPrefillKernelOneHotCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                                   size_t page_size, size_t head_dim, bool causal,
                                                   PosEncodingMode pos_encoding_mode,
                                                   bool allow_fp16_qk_reduction) {
  uint32_t batch_size = 9;
  std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
  utils::vec_randint_(q_lens, 1, 15);
  utils::vec_randint_(kv_lens, 15, 257);
  std::vector<int32_t> append_indptr{0};
  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<T> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  std::vector<std::vector<T>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = (kv_len - 1) % page_size + 1;
    std::vector<T> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    key.push_back(k);
    value.push_back(v);
    kv_last_page_len.push_back(last_page_len);
    kv_indptr.push_back(kv_indptr.back() + num_pages);
    for (size_t j = 0; j < num_pages; ++j) {
      kv_indices.push_back(page_counter++);
    }
  }

  kv_data.resize(page_counter * 2 * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<kv_layout, T, int32_t>(paged_kv_cpu, key, value,
                                                              append_indptr);

  // copy data to device
  thrust::device_vector<T> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  BatchPrefillHandler handler;
  size_t workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> buffer(workspace_size_in_bytes);

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    // create one-hot queries
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<int32_t> q_indptr{0};
    for (uint32_t i = 0; i < batch_size; ++i) {
      q_indptr.push_back(i >= request_idx ? q_len : 0);
    }
    std::vector<T> q(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(q);

    std::vector<T> o_ref = cpu_reference::single_mha<T, T>(
        q, key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads, num_kv_heads,
        head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);

    thrust::device_vector<int32_t> q_indptr_device(q_indptr);
    thrust::device_vector<T> q_device(q);
    thrust::device_vector<T> o_device(q_len * num_qo_heads * head_dim);

    handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()), workspace_size_in_bytes,
                         thrust::raw_pointer_cast(append_indptr.data()), batch_size, num_qo_heads,
                         num_kv_heads, head_dim);

    for (uint32_t num_runs = 0; num_runs < 10; ++num_runs) {
      auto status = flashinfer::BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices,
                                                                    kv_layout, T, T, int32_t>(
          &handler, thrust::raw_pointer_cast(q_device.data()),
          thrust::raw_pointer_cast(q_indptr_device.data()), /*q_offset=*/nullptr, paged_kv,
          thrust::raw_pointer_cast(o_device.data()),
          /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
      EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
    }

    thrust::host_vector<T> o_host(o_device);
    size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
    bool nan_detected = false;
    for (size_t i = 0; i < q_len * num_qo_heads * head_dim; ++i) {
      if (std::isnan(float(o_host[i]))) {
        nan_detected = true;
      }
      num_result_errors_atol_1e_3_rtol_1e_3 +=
          (!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
    }
    float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                     max(float(q_len * num_qo_heads * head_dim), 1.f);
    std::cout << "request_idx=" << request_idx << ", page_size=" << page_size
              << ", num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
              << ", q_len=" << q_len << ", kv_len=" << kv_len << ", head_dim=" << head_dim
              << ", causal=" << causal
              << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
              << ", result_accuracy=" << result_accuracy << std::endl;
    EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
    EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
  }
}

template <typename T>
void _TestBatchRaggedPrefillKernelCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                              size_t head_dim, bool causal,
                                              PosEncodingMode pos_encoding_mode,
                                              bool allow_fp16_qk_reduction) {
  uint32_t batch_size = 9;
  std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
  utils::vec_randint_(q_lens, 1, 15);
  utils::vec_randint_(kv_lens, 15, 257);
  std::vector<int32_t> append_indptr{0}, kv_indptr{0};

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + q_lens[request_idx]);
    kv_indptr.push_back(kv_indptr.back() + kv_lens[request_idx]);
  }

  std::vector<T> queries;
  std::vector<T> keys;
  std::vector<T> values;
  std::vector<T> output_refs;

  BatchPrefillHandler handler;
  size_t workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> buffer(workspace_size_in_bytes);

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    std::vector<T> q(q_lens[request_idx] * num_qo_heads * head_dim);
    std::vector<T> k(kv_lens[request_idx] * num_kv_heads * head_dim),
        v(kv_lens[request_idx] * num_kv_heads * head_dim);
    uint32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    utils::vec_normal_(q);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    std::vector<T> o_ref =
        cpu_reference::single_mha<T, T>(q, k, v, q_len, kv_len, num_qo_heads, num_kv_heads,
                                        head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);
    // NOTE(Zihao): The following code is only compatible with kv_layout = QKVLayout::kNHD
    std::copy(q.begin(), q.end(), std::back_inserter(queries));
    std::copy(k.begin(), k.end(), std::back_inserter(keys));
    std::copy(v.begin(), v.end(), std::back_inserter(values));
    std::copy(o_ref.begin(), o_ref.end(), std::back_inserter(output_refs));
  }

  thrust::device_vector<T> queries_device(queries);
  thrust::device_vector<T> keys_device(keys);
  thrust::device_vector<T> values_device(values);
  thrust::device_vector<T> output_device(queries.size());
  thrust::device_vector<int32_t> append_indptr_device(append_indptr);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);

  handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()), workspace_size_in_bytes,
                       thrust::raw_pointer_cast(append_indptr_device.data()), batch_size,
                       num_qo_heads, num_kv_heads, head_dim);

  auto status = BatchPrefillWithRaggedKVCacheWrapper<T, T, int32_t>(
      &handler, thrust::raw_pointer_cast(queries_device.data()),
      thrust::raw_pointer_cast(append_indptr_device.data()),
      thrust::raw_pointer_cast(keys_device.data()), thrust::raw_pointer_cast(values_device.data()),
      thrust::raw_pointer_cast(kv_indptr_device.data()),
      /*q_offset=*/nullptr,
      /*k_rope_pos_offset=*/nullptr, thrust::raw_pointer_cast(output_device.data()),
      /*lse=*/nullptr, batch_size, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout,
      pos_encoding_mode, allow_fp16_qk_reduction);

  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<T> output_host(output_device);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < output_refs.size(); ++i) {
    if (std::isnan(float(output_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(output_host[i]), float(output_refs[i]), 1e-3, 1e-3));
  }

  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / max(float(output_refs.size()), 1.f);
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", head_dim=" << head_dim << ", causal=" << causal
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;

  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
}

template <typename T>
void _TestBatchPagedPrefillKernelShortContextCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                                         size_t page_size, size_t head_dim,
                                                         bool causal,
                                                         PosEncodingMode pos_encoding_mode,
                                                         bool allow_fp16_qk_reduction) {
  uint32_t batch_size = 7;
  std::vector<int32_t> q_lens(batch_size);
  utils::vec_randint_(q_lens, 1, 64);
  std::vector<int32_t> kv_lens(q_lens);
  std::vector<int32_t> q_indptr{0};
  for (uint32_t i = 0; i < batch_size; ++i) {
    q_indptr.push_back(q_indptr.back() + q_lens[i]);
  }
  std::vector<int32_t> append_indptr{0};
  for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<T> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;
  std::vector<std::vector<T>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = (kv_len - 1) % page_size + 1;
    std::vector<T> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    key.push_back(k);
    value.push_back(v);
    kv_last_page_len.push_back(last_page_len);
    kv_indptr.push_back(kv_indptr.back() + num_pages);
    for (size_t j = 0; j < num_pages; ++j) {
      kv_indices.push_back(page_counter++);
    }
  }

  kv_data.resize(page_counter * 2 * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<kv_layout, T, int32_t>(paged_kv_cpu, key, value,
                                                              append_indptr);

  // copy data to device
  thrust::device_vector<T> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  std::vector<std::vector<T>> q, o_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx];
    std::vector<T> qi(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(qi);
    q.push_back(qi);
  }
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    // create one-hot queries
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<T> o_ref_i = cpu_reference::single_mha<T, T>(
        q[request_idx], key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);
    o_ref.push_back(o_ref_i);
  }

  std::vector<T> q_concat, o_concat_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_concat.insert(q_concat.end(), q[request_idx].begin(), q[request_idx].end());
    o_concat_ref.insert(o_concat_ref.end(), o_ref[request_idx].begin(), o_ref[request_idx].end());
  }
  thrust::device_vector<T> q_device(q_concat);

  thrust::device_vector<int32_t> q_indptr_device(q_indptr);
  thrust::device_vector<T> o_device(o_concat_ref.size());

  BatchPrefillHandler handler;
  size_t workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> buffer(workspace_size_in_bytes);

  handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()), workspace_size_in_bytes,
                       thrust::raw_pointer_cast(append_indptr.data()), batch_size, num_qo_heads,
                       num_kv_heads, head_dim);

  auto status =
      BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, kv_layout, T, T, int32_t>(
          &handler, thrust::raw_pointer_cast(q_device.data()),
          thrust::raw_pointer_cast(q_indptr_device.data()),
          /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o_device.data()),
          /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<T> o_host(o_device);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < o_concat_ref.size(); ++i) {
    if (std::isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_concat_ref[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / max(float(o_concat_ref.size()), 1.f);
  std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", head_dim=" << head_dim
            << ", causal=" << causal
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
}

template <typename T>
void _TestBatchPagedPrefillKernelLongContextCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                                        size_t page_size, size_t head_dim,
                                                        bool causal,
                                                        PosEncodingMode pos_encoding_mode,
                                                        bool allow_fp16_qk_reduction) {
  std::vector<std::vector<std::vector<T>>> keys, values;
  std::vector<int32_t> q_lens{63}, kv_lens{2047};
  std::vector<int32_t> q_indptr{0, 63};
  std::vector<int32_t> append_indptr{0, 2047};
  std::vector<T> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  size_t num_pages = (kv_lens[0] + page_size - 1) / page_size;
  size_t last_page_len = (kv_lens[0] - 1) % page_size + 1;
  std::vector<T> k(kv_lens[0] * num_kv_heads * head_dim), v(kv_lens[0] * num_kv_heads * head_dim);
  utils::vec_normal_(k);
  utils::vec_normal_(v);
  kv_last_page_len.push_back(last_page_len);
  kv_indptr.push_back(kv_indptr.back() + num_pages);
  for (size_t j = 0; j < num_pages; ++j) {
    kv_indices.push_back(page_counter++);
  }

  kv_data.resize(page_counter * 1 * 2 * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, 1, kv_data.data(), kv_indices.data(), kv_indptr.data(),
      kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<kv_layout, T, int32_t>(paged_kv_cpu, {k}, {v},
                                                              append_indptr);

  // copy data to device
  thrust::device_vector<T> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  // create one-hot queries
  std::vector<T> q(q_lens[0] * num_qo_heads * head_dim);
  utils::vec_normal_(q);

  std::vector<T> o_ref =
      cpu_reference::single_mha<T, T>(q, k, v, q_lens[0], kv_lens[0], num_qo_heads, num_kv_heads,
                                      head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);

  thrust::device_vector<int32_t> q_indptr_device(q_indptr);
  thrust::device_vector<T> q_device(q);
  thrust::device_vector<T> o_device(q_lens[0] * num_qo_heads * head_dim);

  BatchPrefillHandler handler;
  size_t workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> buffer(workspace_size_in_bytes);

  handler.BeginForward((void*)thrust::raw_pointer_cast(buffer.data()), workspace_size_in_bytes,
                       thrust::raw_pointer_cast(append_indptr.data()),
                       /*batch_size=*/1, num_qo_heads, num_kv_heads, head_dim);

  auto status =
      BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, kv_layout, T, T, int32_t>(
          &handler, thrust::raw_pointer_cast(q_device.data()),
          thrust::raw_pointer_cast(q_indptr_device.data()),
          /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o_device.data()),
          /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<T> o_host(o_device);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < q_lens[0] * num_qo_heads * head_dim; ++i) {
    if (std::isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
  }
  float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                   max(float(q_lens[0] * num_qo_heads * head_dim), 1.f);
  std::cout << ", page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", q_len=" << q_lens[0]
            << ", kv_len=" << kv_lens[0] << ", head_dim=" << head_dim << ", causal=" << causal
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
}

template <typename T>
void TestBatchPagedPrefillKernelOneHotCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t page_size : {1, 8, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelOneHotCorrectness<T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void TestBatchPagedPrefillKernelShortContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t page_size : {1, 8, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelShortContextCorrectness<T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void TestBatchPagedPrefillKernelLongContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {1, 2, 8}) {
    for (size_t num_qo_heads : {8}) {
      for (size_t page_size : {1, 8, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelLongContextCorrectness<T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void TestBatchRaggedPrefillKernelCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t head_dim : {64, 128, 256}) {
        for (size_t causal : {false, true}) {
          for (size_t pos_encoding_mode : {0, 1}) {
            _TestBatchRaggedPrefillKernelCorrectness<T>(num_kv_heads, num_qo_heads, head_dim,
                                                        causal, PosEncodingMode(pos_encoding_mode),
                                                        allow_fp16_qk_reduction);
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestFP16) {
  TestBatchPagedPrefillKernelShortContextCorrectness<half>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestFP16QKHalfAccum) {
  TestBatchPagedPrefillKernelShortContextCorrectness<half>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestFP16QKHalfAccum) {
  TestBatchPagedPrefillKernelLongContextCorrectness<half>(true);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillKernelCorrectnessTestOneHotFP16) {
  TestBatchPagedPrefillKernelOneHotCorrectness<half>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillKernelCorrectnessTestOneHotFP16QKHalfAccum) {
  TestBatchPagedPrefillKernelOneHotCorrectness<half>(true);
}

TEST(FlashInferCorrectnessTest, BatchRaggedPrefillTestFP16) {
  TestBatchRaggedPrefillKernelCorrectness<half>(false);
}

TEST(FlashInferCorrectnessTest, BatchRaggedPrefillTestFP16QKHalfAccum) {
  TestBatchRaggedPrefillKernelCorrectness<half>(true);
}
