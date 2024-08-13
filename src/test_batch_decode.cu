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

#include <type_traits>

#include "cpu_reference.h"
#include "flashinfer_ops.cuh"
#include "utils.h"

using namespace flashinfer;

constexpr QKVLayout kv_layout = QKVLayout::kNHD;

template <typename DTypeQO, typename DTypeKV>
void _TestBatchDecodingKernelCorrectness(size_t page_size, size_t batch_size, size_t num_qo_heads,
                                         size_t num_kv_heads, size_t head_dim,
                                         flashinfer::PosEncodingMode pos_encoding_mode,
                                         bool cooperative) {
  std::vector<int32_t> seq_lens(batch_size);
  utils::vec_randint_(seq_lens, 1, 1024);
  std::vector<int32_t> append_indptr{0};
  for (size_t i = 0; i < batch_size; ++i) {
    append_indptr.push_back(append_indptr.back() + seq_lens[i]);
  }
  std::vector<DTypeQO> q;
  std::vector<DTypeQO> o_ref;
  std::vector<DTypeKV> kv_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;

  std::vector<std::vector<DTypeKV>> keys, values;
  for (size_t i = 0; i < batch_size; ++i) {
    size_t seq_len = seq_lens[i];
    size_t num_pages = (seq_len + page_size - 1) / page_size;
    size_t last_page_len = (seq_len - 1) % page_size + 1;
    std::vector<DTypeQO> qi(num_qo_heads * head_dim);
    std::vector<DTypeKV> ki(seq_len * num_kv_heads * head_dim),
        vi(seq_len * num_kv_heads * head_dim);
    utils::vec_normal_(qi);
    utils::vec_normal_(ki);
    utils::vec_normal_(vi);

    // compute reference output
    std::vector<DTypeQO> o_ref_i = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        qi, ki, vi, 1, seq_len, num_qo_heads, num_kv_heads, head_dim, false, QKVLayout::kNHD,
        pos_encoding_mode);
    keys.push_back(ki);
    values.push_back(vi);
    // append new q and o_ref
    q.insert(q.end(), qi.begin(), qi.end());
    o_ref.insert(o_ref.end(), o_ref_i.begin(), o_ref_i.end());
    // append new kv_indptr, kv_indices and kv_last_page_len
    kv_last_page_len.push_back(last_page_len);
    kv_indptr.push_back(kv_indptr.back() + num_pages);
    for (size_t j = 0; j < num_pages; ++j) {
      kv_indices.push_back(page_counter++);
    }
  }
  kv_data.resize(page_counter * 2 * num_kv_heads * page_size * head_dim);
  utils::vec_zero_(kv_data);
  assert(q.size() == batch_size * num_qo_heads * head_dim);
  assert(o_ref.size() == batch_size * num_qo_heads * head_dim);

  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout, kv_data.data(), kv_indices.data(),
      kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, keys, values, append_indptr);

  // copy data to device
  thrust::device_vector<DTypeKV> kv_data_device(kv_data);
  thrust::device_vector<int32_t> kv_indptr_device(kv_indptr);
  thrust::device_vector<int32_t> kv_indices_device(kv_indices);
  thrust::device_vector<int32_t> kv_last_page_len_device(kv_last_page_len);
  thrust::device_vector<DTypeQO> q_device(q);
  thrust::device_vector<DTypeQO> o_device(o_ref.size());

  // create paged_kv object
  flashinfer::paged_kv_t<PageStorage::kIndices, DTypeKV, int32_t> paged_kv(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout,
      thrust::raw_pointer_cast(kv_data_device.data()),
      thrust::raw_pointer_cast(kv_indices_device.data()),
      thrust::raw_pointer_cast(kv_indptr_device.data()),
      thrust::raw_pointer_cast(kv_last_page_len_device.data()));
  flashinfer::BatchDecodeHandler handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char> float_buffer(float_workspace_size_in_bytes);
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char> int_buffer(int_workspace_size_in_bytes);
  BatchDecodeHandlerBeginForward<PageStorage::kIndices, DTypeQO, DTypeKV, DTypeQO, int32_t>(
      &handler, (void*)thrust::raw_pointer_cast(float_buffer.data()), float_workspace_size_in_bytes,
      (void*)thrust::raw_pointer_cast(int_buffer.data()), int_workspace_size_in_bytes,
      kv_indptr.data(), kv_last_page_len.data(), batch_size, num_qo_heads, num_kv_heads, head_dim,
      page_size, pos_encoding_mode);

  if (!cooperative) {
    // use non-cooperative kernel
    cudaError_t status =
        flashinfer::BatchDecodeWithPagedKVCacheNoSplitKV<PageStorage::kIndices, DTypeQO, DTypeKV,
                                                         DTypeQO, int32_t>(
            thrust::raw_pointer_cast(q_device.data()), /*q_offset=*/nullptr, paged_kv,
            kv_partition_info_t<int32_t>(), thrust::raw_pointer_cast(o_device.data()),
            /*lse=*/nullptr, num_qo_heads, pos_encoding_mode);
    EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
  } else {
    cudaError_t status =
        flashinfer::BatchDecodeWithPagedKVCacheWrapper<PageStorage::kIndices, DTypeQO, DTypeKV,
                                                       DTypeQO, int32_t>(
            &handler, thrust::raw_pointer_cast(q_device.data()), /*q_offset=*/nullptr, paged_kv,
            thrust::raw_pointer_cast(o_device.data()), /*lse=*/nullptr, num_qo_heads,
            pos_encoding_mode);
    EXPECT_EQ(status, cudaSuccess) << "CUDA error: " + std::string(cudaGetErrorString(status));
  }
  // compare result
  thrust::host_vector<DTypeQO> o_host = o_device;
  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < batch_size * num_qo_heads * head_dim; ++i) {
    if (std::isnan(float(o_host[i]))) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 +=
        (!utils::isclose(float(o_host[i]), float(o_ref[i]), 1e-3, 1e-3));
  }
  float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
                                   float(batch_size * num_qo_heads * head_dim);
  std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", batch_size=" << batch_size
            << ", head_dim=" << head_dim
            << ", pos_encoding_mode=" << flashinfer::PosEncodingModeToString(pos_encoding_mode)
            << ", result accuracy (atol=1e-3, rtol=1e-3): " << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected.";
}

template <typename DTypeQO, typename DTypeKV>
void TestBatchDecodeKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 16}) {
    for (size_t batch_size : {1, 7, 37, 61}) {
      for (size_t num_qo_heads : {32}) {
        for (size_t num_kv_heads : {32, 8, 4}) {
          for (size_t head_dim : {64, 128, 256}) {
            for (size_t pos_encoding_mode : {0U, 1U}) {
              _TestBatchDecodingKernelCorrectness<DTypeQO, DTypeKV>(
                  page_size, batch_size, num_qo_heads, num_kv_heads, head_dim,
                  flashinfer::PosEncodingMode(pos_encoding_mode), false);
            }
          }
        }
      }
    }
  }
}

template <typename DTypeQO, typename DTypeKV>
void TestCooperativeBatchDecodeKernelCorrectness() {
  for (size_t page_size : {1, 3, 7, 16}) {
    for (size_t batch_size : {1, 2, 4, 8}) {
      for (size_t num_qo_heads : {32}) {
        for (size_t num_kv_heads : {32, 8, 4}) {
          for (size_t head_dim : {64, 128, 256}) {
            for (size_t pos_encoding_mode : {0U, 1U}) {
              _TestBatchDecodingKernelCorrectness<DTypeQO, DTypeKV>(
                  page_size, batch_size, num_qo_heads, num_kv_heads, head_dim,
                  flashinfer::PosEncodingMode(pos_encoding_mode), true);
            }
          }
        }
      }
    }
  }
}

TEST(FlashInferCorrectnessTest, BatchDecodeKernelCorrectnessTestFP16) {
  TestBatchDecodeKernelCorrectness<half, half>();
}

#ifdef FLASHINFER_ENABLE_BF16
TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessBF16) {
  TestBatchDecodeKernelCorrectness<__nv_bfloat16, __nv_bfloat16>();
}
#endif

#ifdef FLASHINFER_ENABLE_FP8
TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE4M3) {
  TestBatchDecodeKernelCorrectness<half, __nv_fp8_e4m3>();
}

TEST(FlashInferCorrectnessTest, TestBatchDecodeKernelCorrectnessE5M2) {
  TestBatchDecodeKernelCorrectness<half, __nv_fp8_e5m2>();
}
#endif

TEST(FlashInferCorrectnessTest, TestCooperativeBatchDecodeKernelCorrectnessTestFP16) {
  TestCooperativeBatchDecodeKernelCorrectness<half, half>();
}
