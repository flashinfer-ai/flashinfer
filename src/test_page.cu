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

template <typename T>
void _TestAppendPagedKVKernelCorrectness(size_t page_size, size_t batch_size, size_t num_heads,
                                         size_t head_dim, QKVLayout kv_layout) {
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
    paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv_cpu(
        num_heads, page_size, head_dim, batch_size, kv_layout, kv_data_cpu.data(),
        indices_cpu.data(), indptr_cpu.data(), last_page_len.data());
    cpu_reference::append_paged_kv_cache(paged_kv_cpu, keys, values, append_indptr);

    thrust::device_vector<int32_t> indptr_gpu(indptr_cpu);
    thrust::device_vector<int32_t> indices_gpu(indices_cpu);
    thrust::device_vector<int32_t> last_page_len_gpu(last_page_len);
    paged_kv_t<PageStorage::kIndices, T, int32_t> paged_kv_gpu(
        num_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data_gpu.data()), thrust::raw_pointer_cast(indices_gpu.data()),
        thrust::raw_pointer_cast(indptr_gpu.data()),
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

template <typename T>
void TestAppendPagedKVKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 17}) {
    for (size_t batch_size : {1, 2, 3, 5, 7, 23, 79, 91}) {
      for (size_t num_heads : {32}) {
        for (QKVLayout kv_layout : {QKVLayout::kNHD, QKVLayout::kHND}) {
          for (size_t head_dim : {64, 128, 256}) {
            _TestAppendPagedKVKernelCorrectness<T>(page_size, batch_size, num_heads, head_dim,
                                                   kv_layout);
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestFP16) {
  TestAppendPagedKVKernelCorrectness<half>();
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestFP32) {
  TestAppendPagedKVKernelCorrectness<float>();
}

#ifdef FLASHINFER_ENABLE_BF16
TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestBF16) {
  TestAppendPagedKVKernelCorrectness<__nv_bfloat16>();
}
#endif

#ifdef FLASHINFER_ENABLE_FP8
TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestE4M3) {
  TestAppendPagedKVKernelCorrectness<__nv_fp8_e4m3>();
}

TEST(FlashInferCorrectnessTest, AppendPagedKVKernelCorrectnessTestE5M2) {
  TestAppendPagedKVKernelCorrectness<__nv_fp8_e5m2>();
}
#endif
