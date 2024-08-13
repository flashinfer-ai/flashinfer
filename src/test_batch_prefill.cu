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
#include "flashinfer/pos_enc.cuh"
#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;
constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename DTypeQO, typename DTypeKV>
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
  std::vector<DTypeKV> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  std::vector<std::vector<DTypeKV>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = (kv_len - 1) % page_size + 1;
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
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
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value, append_indptr);

  // copy data to device
  thrust::device_vector<DTypeKV> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.k_data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.v_data = paged_kv.k_data + paged_kv_cpu.kv_ptr_delta();
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    // create one-hot queries
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<int32_t> q_indptr{0};
    for (uint32_t i = 0; i < batch_size; ++i) {
      q_indptr.push_back(i >= request_idx ? q_len : 0);
    }
    std::vector<DTypeQO> q(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(q);

    std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        q, key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads, num_kv_heads,
        head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);

    thrust::device_vector<int32_t> q_indptr_device(q_indptr);
    thrust::device_vector<DTypeQO> q_device(q);
    thrust::device_vector<DTypeQO> o_device(q_len * num_qo_heads * head_dim);

    handler.BeginForward<DTypeQO, int32_t>(
        (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
        (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
        q_indptr.data(), kv_indptr.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
        page_size);

    for (uint32_t num_runs = 0; num_runs < 10; ++num_runs) {
      auto status = flashinfer::BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, DTypeQO,
                                                                    DTypeKV, DTypeQO, int32_t>(
          &handler, thrust::raw_pointer_cast(q_device.data()),
          thrust::raw_pointer_cast(q_indptr_device.data()), /*q_offset=*/nullptr, paged_kv,
          thrust::raw_pointer_cast(o_device.data()),
          /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
      EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
    }

    thrust::host_vector<DTypeQO> o_host(o_device);
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

template <typename DTypeQO, typename DTypeKV>
void _TestBatchRaggedPrefillKernelCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                              size_t head_dim, bool causal,
                                              PosEncodingMode pos_encoding_mode,
                                              bool allow_fp16_qk_reduction) {
  uint32_t batch_size = 9;
  std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
  utils::vec_randint_(q_lens, 10, 15);
  utils::vec_randint_(kv_lens, 128, 2048);
  std::vector<int32_t> append_indptr{0}, kv_indptr{0};

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + q_lens[request_idx]);
    kv_indptr.push_back(kv_indptr.back() + kv_lens[request_idx]);
  }

  std::vector<DTypeQO> queries;
  std::vector<DTypeKV> keys;
  std::vector<DTypeKV> values;
  std::vector<DTypeKV> output_refs;

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    std::vector<DTypeQO> q(q_lens[request_idx] * num_qo_heads * head_dim);
    std::vector<DTypeKV> k(kv_lens[request_idx] * num_kv_heads * head_dim),
        v(kv_lens[request_idx] * num_kv_heads * head_dim);
    uint32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    utils::vec_normal_(q);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        q, k, v, q_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, QKVLayout::kNHD,
        pos_encoding_mode);
    // NOTE(Zihao): The following code is only compatible with kv_layout = QKVLayout::kNHD
    std::copy(q.begin(), q.end(), std::back_inserter(queries));
    std::copy(k.begin(), k.end(), std::back_inserter(keys));
    std::copy(v.begin(), v.end(), std::back_inserter(values));
    std::copy(o_ref.begin(), o_ref.end(), std::back_inserter(output_refs));
  }

  thrust::device_vector<DTypeQO> queries_device(queries);
  thrust::device_vector<DTypeKV> keys_device(keys);
  thrust::device_vector<DTypeKV> values_device(values);
  thrust::device_vector<DTypeQO> output_device(queries.size());
  thrust::device_vector<int32_t> append_indptr_device(append_indptr);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);

  handler.BeginForward<DTypeQO, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      append_indptr.data(), kv_indptr.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      /*page_size=*/1);

  auto status = BatchPrefillWithRaggedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
      &handler, thrust::raw_pointer_cast(queries_device.data()),
      thrust::raw_pointer_cast(append_indptr_device.data()),
      thrust::raw_pointer_cast(keys_device.data()), thrust::raw_pointer_cast(values_device.data()),
      thrust::raw_pointer_cast(kv_indptr_device.data()),
      /*q_offset=*/nullptr,
      /*k_rope_pos_offset=*/nullptr, thrust::raw_pointer_cast(output_device.data()),
      /*lse=*/nullptr, batch_size, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout,
      pos_encoding_mode, allow_fp16_qk_reduction);

  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<DTypeQO> output_host(output_device);
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

template <typename DTypeQO, typename DTypeKV>
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
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_indptr.push_back(q_indptr.back() + q_lens[request_idx]);
  }
  std::vector<int32_t> append_indptr{0};
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<DTypeKV> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;
  std::vector<std::vector<DTypeKV>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = (kv_len - 1) % page_size + 1;
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
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
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value, append_indptr);

  // copy data to device
  thrust::device_vector<DTypeKV> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.k_data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.v_data = paged_kv.k_data + paged_kv_cpu.kv_ptr_delta();
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  std::vector<std::vector<DTypeQO>> q, o_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx];
    std::vector<DTypeQO> qi(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(qi);
    q.push_back(qi);
  }
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<DTypeQO> o_ref_i = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        q[request_idx], key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);
    o_ref.push_back(o_ref_i);
  }

  std::vector<DTypeQO> q_concat, o_concat_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_concat.insert(q_concat.end(), q[request_idx].begin(), q[request_idx].end());
    o_concat_ref.insert(o_concat_ref.end(), o_ref[request_idx].begin(), o_ref[request_idx].end());
  }
  thrust::device_vector<DTypeQO> q_device(q_concat);

  thrust::device_vector<int32_t> q_indptr_device(q_indptr);
  thrust::device_vector<DTypeQO> o_device(o_concat_ref.size());

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  handler.BeginForward<DTypeQO, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      q_indptr.data(), kv_indptr.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      page_size);

  auto status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, DTypeQO, DTypeKV,
                                                    DTypeQO, int32_t>(
      &handler, thrust::raw_pointer_cast(q_device.data()),
      thrust::raw_pointer_cast(q_indptr_device.data()),
      /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o_device.data()),
      /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<DTypeQO> o_host(o_device);
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

template <typename DTypeQO, typename DTypeKV>
void _TestBatchPagedPrefillKernelQMinMaxKVMinMaxCorrectness(
    size_t batch_size, size_t num_kv_heads, size_t num_qo_heads, size_t page_size, size_t head_dim,
    bool allow_fp16_qk_reduction, uint32_t q_len_min, uint32_t q_len_max, uint32_t kv_len_min,
    uint32_t kv_len_max) {
  std::vector<int32_t> q_lens(batch_size);
  utils::vec_randint_(q_lens, q_len_min, q_len_max);
  std::vector<int32_t> kv_lens(batch_size);
  utils::vec_randint_(kv_lens, kv_len_min, kv_len_max);

  std::vector<int32_t> q_indptr{0};
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_indptr.push_back(q_indptr.back() + q_lens[request_idx]);
  }
  std::vector<int32_t> append_indptr{0};
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<DTypeKV> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;
  std::vector<std::vector<DTypeKV>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = num_pages == 0 ? 0 : (kv_len - 1) % page_size + 1;
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
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
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value, append_indptr);

  // copy data to device
  thrust::device_vector<DTypeKV> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.k_data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.v_data = paged_kv.k_data + paged_kv_cpu.kv_ptr_delta();
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  std::vector<std::vector<DTypeQO>> q, o_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx];
    std::vector<DTypeQO> qi(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(qi);
    q.push_back(qi);
  }
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<DTypeQO> o_ref_i = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        q[request_idx], key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, /*causal=*/false, QKVLayout::kNHD,
        /*pos_encoding_mode*/ PosEncodingMode::kNone);
    o_ref.push_back(o_ref_i);
  }

  std::vector<DTypeQO> q_concat, o_concat_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_concat.insert(q_concat.end(), q[request_idx].begin(), q[request_idx].end());
    o_concat_ref.insert(o_concat_ref.end(), o_ref[request_idx].begin(), o_ref[request_idx].end());
  }
  thrust::device_vector<DTypeQO> q_device(q_concat);

  thrust::device_vector<int32_t> q_indptr_device(q_indptr);
  thrust::device_vector<DTypeQO> o_device(o_concat_ref.size());

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  handler.BeginForward<DTypeQO, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      q_indptr.data(), kv_indptr.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      page_size);

  auto status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, DTypeQO, DTypeKV,
                                                    DTypeQO, int32_t>(
      &handler, thrust::raw_pointer_cast(q_device.data()),
      thrust::raw_pointer_cast(q_indptr_device.data()),
      /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o_device.data()),
      /*lse=*/nullptr, num_qo_heads, /*causal=*/false,
      /*pos_encoding_mode*/ PosEncodingMode::kNone);
  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<DTypeQO> o_host(o_device);
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
  std::cout << "batch_size=" << batch_size << ", page_size=" << page_size
            << ", num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", head_dim=" << head_dim << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected in output.";
}

template <typename DTypeQO, typename DTypeKV>
void _TestBatchPagedPrefillKernelLongContextCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                                        size_t page_size, size_t head_dim,
                                                        bool causal,
                                                        PosEncodingMode pos_encoding_mode,
                                                        bool allow_fp16_qk_reduction) {
  std::vector<std::vector<std::vector<DTypeKV>>> keys, values;
  std::vector<int32_t> q_lens{33}, kv_lens{32768};
  std::vector<int32_t> q_indptr{0, 33};
  std::vector<int32_t> append_indptr{0, 32768};
  std::vector<DTypeKV> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  size_t num_pages = (kv_lens[0] + page_size - 1) / page_size;
  size_t last_page_len = (kv_lens[0] - 1) % page_size + 1;
  std::vector<DTypeKV> k(kv_lens[0] * num_kv_heads * head_dim),
      v(kv_lens[0] * num_kv_heads * head_dim);
  utils::vec_normal_(k);
  utils::vec_normal_(v);
  kv_last_page_len.push_back(last_page_len);
  kv_indptr.push_back(kv_indptr.back() + num_pages);
  for (size_t j = 0; j < num_pages; ++j) {
    kv_indices.push_back(page_counter++);
  }

  kv_data.resize(page_counter * 1 * 2 * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, 1, kv_layout, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, {k}, {v}, append_indptr);

  // copy data to device
  thrust::device_vector<DTypeKV> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.k_data = thrust::raw_pointer_cast(kv_data_device.data());
  paged_kv.v_data = paged_kv.k_data + paged_kv_cpu.kv_ptr_delta();
  paged_kv.indices = thrust::raw_pointer_cast(kv_indices_device.data());
  paged_kv.indptr = thrust::raw_pointer_cast(kv_indptr_device.data());
  paged_kv.last_page_len = thrust::raw_pointer_cast(kv_last_page_len_device.data());

  // create one-hot queries
  std::vector<DTypeQO> q(q_lens[0] * num_qo_heads * head_dim);
  utils::vec_normal_(q);

  std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
      q, k, v, q_lens[0], kv_lens[0], num_qo_heads, num_kv_heads, head_dim, causal, QKVLayout::kNHD,
      pos_encoding_mode);

  thrust::device_vector<int32_t> q_indptr_device(q_indptr);
  thrust::device_vector<DTypeQO> q_device(q);
  thrust::device_vector<DTypeQO> o_device(q_lens[0] * num_qo_heads * head_dim);

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);

  handler.BeginForward<DTypeQO, int32_t>(
      (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      append_indptr.data(), kv_indptr.data(),
      /*batch_size=*/1, num_qo_heads, num_kv_heads, head_dim, page_size);

  auto status = BatchPrefillWithPagedKVCacheWrapper<PageStorage::kIndices, DTypeQO, DTypeKV,
                                                    DTypeQO, int32_t>(
      &handler, thrust::raw_pointer_cast(q_device.data()),
      thrust::raw_pointer_cast(q_indptr_device.data()),
      /*q_offset=*/nullptr, paged_kv, thrust::raw_pointer_cast(o_device.data()),
      /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, allow_fp16_qk_reduction);
  EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));

  thrust::host_vector<DTypeQO> o_host(o_device);
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
  std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
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
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelOneHotCorrectness<T, T>(
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
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelShortContextCorrectness<T, T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestBatchPagedPrefillFP8KernelShortContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0}) {
              _TestBatchPagedPrefillKernelShortContextCorrectness<half, DTypeKV>(
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
    for (size_t group_size : {1, 3, 4, 5, 6, 7, 8}) {
      size_t num_qo_heads = num_kv_heads * group_size;
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelLongContextCorrectness<T, T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), allow_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestBatchPagedPrefillFP8KernelLongContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {1, 2, 8}) {
    for (size_t group_size : {1, 3, 4, 5, 6, 7, 8}) {
      size_t num_qo_heads = num_kv_heads * group_size;
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0}) {
              _TestBatchPagedPrefillKernelLongContextCorrectness<half, DTypeKV>(
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
void TestBatchPagedPrefillKernelZeroContextCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t batch_size : {1, 4, 7, 11, 19, 37, 99}) {
    for (size_t num_kv_heads : {1, 4}) {
      for (size_t group_size : {1, 8}) {
        size_t num_qo_heads = num_kv_heads * group_size;
        for (size_t page_size : {1, 16}) {
          for (size_t head_dim : {64, 128, 256}) {
            for (size_t kv_len_max : {0, 3}) {
              _TestBatchPagedPrefillKernelQMinMaxKVMinMaxCorrectness<T, T>(
                  batch_size, num_kv_heads, num_qo_heads, page_size, head_dim,
                  allow_fp16_qk_reduction,
                  /*q_len_min=*/1, /*q_len_max=*/3, /*kv_len_min=*/0, kv_len_max);
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
            _TestBatchRaggedPrefillKernelCorrectness<T, T>(
                num_kv_heads, num_qo_heads, head_dim, causal, PosEncodingMode(pos_encoding_mode),
                allow_fp16_qk_reduction);
          }
        }
      }
    }
  }
}

template <typename DTypeKV>
void TestBatchRaggedPrefillFP8KernelCorrectness(bool allow_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t head_dim : {64, 128, 256}) {
        for (size_t causal : {false, true}) {
          for (size_t pos_encoding_mode : {0}) {
            _TestBatchRaggedPrefillKernelCorrectness<half, DTypeKV>(
                num_kv_heads, num_qo_heads, head_dim, causal, PosEncodingMode(pos_encoding_mode),
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

TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestFP16) {
  TestBatchPagedPrefillKernelLongContextCorrectness<half>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillZeroContextTestFP16) {
  TestBatchPagedPrefillKernelZeroContextCorrectness<half>(false);
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

#ifdef FLASHINFER_ENABLE_FP8

TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestE4M3) {
  TestBatchPagedPrefillFP8KernelShortContextCorrectness<__nv_fp8_e4m3>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestE5M2) {
  TestBatchPagedPrefillFP8KernelShortContextCorrectness<__nv_fp8_e5m2>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestE4M3) {
  TestBatchPagedPrefillFP8KernelLongContextCorrectness<__nv_fp8_e4m3>(false);
}

TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestE5M2) {
  TestBatchPagedPrefillFP8KernelLongContextCorrectness<__nv_fp8_e5m2>(false);
}
#endif