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

#include <flashinfer/page.cuh>
#include <type_traits>

#include "cpu_reference.h"
#include "utils.h"

using namespace flashinfer;

template <QKVLayout kv_layout, typename T>
void _TestAppendPagedKVKernelCorrectness(size_t page_size, size_t batch_size, size_t num_heads,
                                         size_t head_dim) {
  // number of conversation rounds
  size_t num_conv_rounds = 3;
  size_t max_decode_len = 1;
  size_t max_prefill_len = 128;
  size_t max_num_pages =
      num_conv_rounds * batch_size * ((max_decode_len + max_prefill_len) / page_size + 1);
  std::vector<T> kv_data_cpu(2 * max_num_pages * page_size * num_heads * head_dim);
  utils::vec_zero_(kv_data_cpu);
  thrust::device_vector<T> kv_data_gpu(kv_data_cpu);
  std::vector<int32_t> seq_len(batch_size);
  utils::vec_fill_(seq_len, 0);
  std::vector<std::vector<int32_t>> page_indices(batch_size);
  std::vector<int32_t> last_page_len(batch_size);
  utils::vec_fill_(last_page_len, 0);
  size_t page_counter = 0;

  for (size_t round = 0; round < 2 * num_conv_rounds; ++round) {
    std::vector<int32_t> append_len(batch_size);
    std::vector<int32_t> append_indptr{0};
    std::vector<std::vector<T>> keys;
    std::vector<std::vector<T>> values;
    if (round % 2 == 0) {
      utils::vec_randint_(append_len, 1, max_prefill_len + 1);
    } else {
      utils::vec_fill_<int32_t>(append_len, max_decode_len);
    }
    for (size_t i = 0; i < batch_size; ++i) {
      append_indptr.push_back(append_indptr.back() + append_len[i]);
      seq_len[i] += append_len[i];
      for (size_t j = 0; j < append_len[i]; ++j) {
        if (last_page_len[i] % page_size == 0) {
          page_indices[i].push_back(page_counter++);
          last_page_len[i] = 1;
        } else {
          last_page_len[i] += 1;
        }
      }
      std::vector<T> ki(append_len[i] * num_heads * head_dim),
          vi(append_len[i] * num_heads * head_dim);
      utils::vec_normal_(ki);
      utils::vec_normal_(vi);
      keys.push_back(ki);
      values.push_back(vi);
    }

    std::vector<int32_t> indptr_cpu{0};
    std::vector<int32_t> indices_cpu;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < page_indices[i].size(); ++j) {
        indices_cpu.push_back(page_indices[i][j]);
      }
      indptr_cpu.push_back(indptr_cpu.back() + page_indices[i].size());
    }
    paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
        num_heads, page_size, head_dim, batch_size, kv_data_cpu.data(), indices_cpu.data(),
        indptr_cpu.data(), last_page_len.data());
    cpu_reference::append_paged_kv_cache<kv_layout>(paged_kv_cpu, keys, values, append_indptr);

    thrust::device_vector<int32_t> indptr_gpu(indptr_cpu);
    thrust::device_vector<int32_t> indices_gpu(indices_cpu);
    thrust::device_vector<int32_t> last_page_len_gpu(last_page_len);
    paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_gpu(
        num_heads, page_size, head_dim, batch_size, thrust::raw_pointer_cast(kv_data_gpu.data()),
        thrust::raw_pointer_cast(indices_gpu.data()), thrust::raw_pointer_cast(indptr_gpu.data()),
        thrust::raw_pointer_cast(last_page_len_gpu.data()));

    thrust::device_vector<int32_t> append_indptr_gpu(append_indptr);
    thrust::device_vector<T> keys_gpu(append_indptr.back() * num_heads * head_dim);
    thrust::device_vector<T> values_gpu(append_indptr.back() * num_heads * head_dim);
    for (size_t i = 0; i < batch_size; ++i) {
      thrust::device_vector<T> ki(keys[i]);
      thrust::device_vector<T> vi(values[i]);
      thrust::copy(ki.begin(), ki.end(),
                   keys_gpu.begin() + append_indptr[i] * num_heads * head_dim);
      thrust::copy(vi.begin(), vi.end(),
                   values_gpu.begin() + append_indptr[i] * num_heads * head_dim);
    }
    if (round % 2 == 0) {
      // call prefill kernel
      cudaError_t status =
          AppendPagedKVCache(paged_kv_gpu, thrust::raw_pointer_cast(keys_gpu.data()),
                             thrust::raw_pointer_cast(values_gpu.data()),
                             thrust::raw_pointer_cast(append_indptr_gpu.data()));
      EXPECT_EQ(status, cudaSuccess) << "AppendPagedKVCache kernel launch failed, error message: "
                                     << cudaGetErrorString(status);
    } else {
      // call decode kernel
      cudaError_t status =
          AppendPagedKVCacheDecode(paged_kv_gpu, thrust::raw_pointer_cast(keys_gpu.data()),
                                   thrust::raw_pointer_cast(values_gpu.data()));
      EXPECT_EQ(status, cudaSuccess)
          << "AppendPagedKVCacheDecode kernel launch failed, error message: "
          << cudaGetErrorString(status);
    }
  }

  thrust::host_vector<T> kv_data_gpu_h(kv_data_gpu);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < kv_data_cpu.size(); ++i) {
    if (std::isnan(float(kv_data_gpu_h[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(kv_data_cpu[i]), float(kv_data_gpu_h[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(kv_data_cpu.size());
  std::cout << "kv_layout=" << QKVLayoutToString(kv_layout) << ", page_size=" << page_size
            << ", batch_size=" << batch_size << ", num_heads=" << num_heads
            << ", head_dim=" << head_dim << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "Nan detected in the result.";
}

template <QKVLayout kv_layout, typename T>
void _TestPagedKVCacheToRaggedTensorCorrectness(size_t page_size, size_t batch_size,
                                                size_t num_heads, size_t head_dim) {
  size_t num_pages_per_request = 10;
  size_t max_num_pages = batch_size * num_pages_per_request;
  std::vector<T> kv_data_cpu(max_num_pages * 2 * page_size * num_heads * head_dim);
  utils::vec_normal_(kv_data_cpu);

  std::vector<int32_t> paged_kv_indptr_host(batch_size + 1);
  std::vector<int32_t> paged_kv_indices_host(batch_size * num_pages_per_request);
  std::vector<int32_t> paged_kv_last_page_len_host(batch_size);
  std::vector<int32_t> kv_indptr_ref(batch_size + 1);
  std::vector<T> key_ref(batch_size * num_pages_per_request * page_size * num_heads * head_dim);
  std::vector<T> value_ref(batch_size * num_pages_per_request * page_size * num_heads * head_dim);
  for (size_t i = 0; i < batch_size; ++i) {
    paged_kv_indptr_host[i] = i * num_pages_per_request;
    for (size_t j = 0; j < num_pages_per_request; ++j) {
      paged_kv_indices_host[i * num_pages_per_request + j] = i + j * batch_size;
    }
    paged_kv_last_page_len_host[i] = page_size;
    kv_indptr_ref[i] = page_size * num_pages_per_request * i;
  }
  paged_kv_indptr_host[batch_size] = batch_size * num_pages_per_request;
  kv_indptr_ref[batch_size] = page_size * num_pages_per_request * batch_size;

  paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_cpu(
      num_heads, page_size, head_dim, batch_size, kv_data_cpu.data(), paged_kv_indices_host.data(),
      paged_kv_indptr_host.data(), paged_kv_last_page_len_host.data());

  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < num_pages_per_request; ++j) {
      if constexpr (kv_layout == QKVLayout::kHND) {
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t entry_idx = 0; entry_idx < page_size; ++entry_idx) {
            std::copy(
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_k_elem_offset(i + j * batch_size, h, entry_idx, 0),
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_k_elem_offset(i + j * batch_size, h, entry_idx + 1, 0),
                key_ref.begin() +
                    (((i * num_pages_per_request + j) * page_size + entry_idx) * num_heads + h) *
                        head_dim);
            std::copy(
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_v_elem_offset(i + j * batch_size, h, entry_idx, 0),
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_v_elem_offset(i + j * batch_size, h, entry_idx + 1, 0),
                value_ref.begin() +
                    (((i * num_pages_per_request + j) * page_size + entry_idx) * num_heads + h) *
                        head_dim);
          }
        }
      } else {
        for (size_t entry_idx = 0; entry_idx < page_size; ++entry_idx) {
          for (size_t h = 0; h < num_heads; ++h) {
            std::copy(
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_k_elem_offset(i + j * batch_size, h, entry_idx, 0),
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_k_elem_offset(i + j * batch_size, h + 1, entry_idx, 0),
                key_ref.begin() +
                    (((i * num_pages_per_request + j) * page_size + entry_idx) * num_heads + h) *
                        head_dim);
            std::copy(
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_v_elem_offset(i + j * batch_size, h, entry_idx, 0),
                kv_data_cpu.begin() +
                    paged_kv_cpu.get_v_elem_offset(i + j * batch_size, h + 1, entry_idx, 0),
                value_ref.begin() +
                    (((i * num_pages_per_request + j) * page_size + entry_idx) * num_heads + h) *
                        head_dim);
          }
        }
      }
    }
  }

  thrust::device_vector<T> kv_data_gpu(kv_data_cpu);
  thrust::device_vector<int32_t> paged_kv_indptr_gpu(paged_kv_indptr_host);
  thrust::device_vector<int32_t> paged_kv_indices_gpu(paged_kv_indices_host);
  thrust::device_vector<int32_t> paged_kv_last_page_len_gpu(paged_kv_last_page_len_host);
  paged_kv_t<PageStorage::kIndices, kv_layout, T, int32_t> paged_kv_gpu(
      num_heads, page_size, head_dim, batch_size, thrust::raw_pointer_cast(kv_data_gpu.data()),
      thrust::raw_pointer_cast(paged_kv_indices_gpu.data()),
      thrust::raw_pointer_cast(paged_kv_indptr_gpu.data()),
      thrust::raw_pointer_cast(paged_kv_last_page_len_gpu.data()));

  std::vector<int32_t> kv_indptr_h(batch_size + 1);
  cudaError_t status =
      PagedKVCacheToRaggedTensorComputeIndptr<PageStorage::kIndices, kv_layout, T, int32_t>(
          paged_kv_gpu, kv_indptr_h);
  EXPECT_EQ(status, cudaSuccess) << "PagedKVCacheToRaggedTensorComputeIndptr "
                                    "kernel launch failed, error message: "
                                 << cudaGetErrorString(status);

  for (size_t i = 0; i < kv_indptr_h.size(); ++i) {
    EXPECT_EQ(kv_indptr_h[i], kv_indptr_ref[i]);
  }

  thrust::device_vector<T> key(batch_size * num_pages_per_request * page_size * num_heads *
                               head_dim);
  thrust::device_vector<T> value(batch_size * num_pages_per_request * page_size * num_heads *
                                 head_dim);
  thrust::device_vector<int32_t> kv_indptr = kv_indptr_h;
  status = PagedKVCacheToRaggedTensor<PageStorage::kIndices, kv_layout, T, int32_t>(
      paged_kv_gpu, thrust::raw_pointer_cast(key.data()), thrust::raw_pointer_cast(value.data()),
      thrust::raw_pointer_cast(kv_indptr.data()));
  EXPECT_EQ(status, cudaSuccess)
      << "PagedKVCacheToRaggedTensor kernel launch failed, error message: "
      << cudaGetErrorString(status);

  thrust::host_vector<T> key_h(key);
  thrust::host_vector<T> value_h(value);
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < key_h.size(); ++i) {
    if (std::isnan(float(key_h[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(key_ref[i]), float(key_h[i]), 1e-3, 1e-3));
  }
  for (size_t i = 0; i < value_h.size(); ++i) {
    if (std::isnan(float(value_h[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(value_ref[i]), float(value_h[i]), 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / float(key_h.size() + value_h.size());
  std::cout << "kv_layout=" << QKVLayoutToString(kv_layout) << ", page_size=" << page_size
            << ", batch_size=" << batch_size << ", num_heads=" << num_heads
            << ", head_dim=" << head_dim << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
}

template <typename T>
void TestAppendPagedKVKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 17}) {
    for (size_t batch_size : {1, 2, 3, 5, 7, 23, 79, 91}) {
      for (size_t num_heads : {32}) {
        for (QKVLayout kv_layout : {QKVLayout::kNHD, QKVLayout::kHND}) {
          for (size_t head_dim : {64, 128, 256}) {
            DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
              _TestAppendPagedKVKernelCorrectness<KV_LAYOUT, T>(page_size, batch_size, num_heads,
                                                                head_dim);
            });
          }
        }
      }
    }
  }
}

template <typename T>
void TestPagedKVCacheToRaggedTensorCorrectness() {
  for (size_t page_size : {1, 3, 7, 17}) {
    for (size_t batch_size : {1, 2, 3, 5, 7, 23, 79, 91}) {
      for (size_t num_heads : {32}) {
        for (QKVLayout kv_layout : {QKVLayout::kNHD, QKVLayout::kHND}) {
          for (size_t head_dim : {64, 128, 256}) {
            DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
              _TestPagedKVCacheToRaggedTensorCorrectness<KV_LAYOUT, T>(page_size, batch_size,
                                                                       num_heads, head_dim);
            });
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestFP16) {
  TestAppendPagedKVKernelCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, PagedKVCacheToRaggedTensorCorrectnessTestFP16) {
  TestPagedKVCacheToRaggedTensorCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestFP32) {
  TestAppendPagedKVKernelCorrectness<float>();
}

TEST(FlashInferCorrectnessTest, PagedKVCacheToRaggedTensorCorrectnessTestFP32) {
  TestPagedKVCacheToRaggedTensorCorrectness<float>();
}

#ifdef FLASHINFER_ENABLE_BF16
TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestBF16) {
  TestAppendPagedKVKernelCorrectness<__nv_bfloat16>();
}
TEST(FlashInferCorrectnessTest, PagedKVCacheToRaggedTensorCorrectnessTestBF16) {
  TestPagedKVCacheToRaggedTensorCorrectness<__nv_bfloat16>();
}
#endif

#ifdef FLASHINFER_ENABLE_FP8
TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestE4M3) {
  TestAppendPagedKVKernelCorrectness<__nv_fp8_e4m3>();
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestE5M2) {
  TestAppendPagedKVKernelCorrectness<__nv_fp8_e5m2>();
}

TEST(FlashInferCorrectnessTest, PagedKVCacheToRaggedTensorCorrectnessTestE4M3) {
  TestPagedKVCacheToRaggedTensorCorrectness<__nv_fp8_e4m3>();
}

TEST(FlashInferCorrectnessTest, PagedKVCacheToRaggedTensorCorrectnessTestE5M2) {
  TestPagedKVCacheToRaggedTensorCorrectness<__nv_fp8_e5m2>();
}
#endif
