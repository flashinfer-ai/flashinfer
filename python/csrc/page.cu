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
#include <flashinfer/page.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void append_paged_kv_cache(torch::Tensor append_key, torch::Tensor append_value,
                           torch::Tensor batch_indices, torch::Tensor positions,
                           torch::Tensor paged_k_cache, torch::Tensor paged_v_cache,
                           torch::Tensor kv_indices, torch::Tensor kv_indptr,
                           torch::Tensor kv_last_page_len, unsigned int layout) {
  CHECK_INPUT(append_key);
  CHECK_INPUT(append_value);
  CHECK_INPUT(batch_indices);
  CHECK_INPUT(positions);
  // NOTE(Zihao): doesn't have to be contiguous
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_v_cache);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_last_page_len);
  CHECK_DIM(3, append_key);
  CHECK_DIM(3, append_value);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(1, positions);
  CHECK_DIM(4, paged_k_cache);
  CHECK_DIM(4, paged_v_cache);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_last_page_len);
  unsigned int nnz = append_key.size(0);
  unsigned int batch_size = kv_last_page_len.size(0);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(batch_indices.size(0), nnz);
  CHECK_EQ(positions.size(0), nnz);
  CHECK_EQ(batch_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(positions.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_indptr.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_indices.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_last_page_len.scalar_type(), torch::kInt32);
  auto device = append_key.device();
  CHECK_EQ(append_key.device(), device);
  CHECK_EQ(append_value.device(), device);
  CHECK_EQ(paged_k_cache.device(), device);
  CHECK_EQ(paged_v_cache.device(), device);
  CHECK_EQ(kv_indices.device(), device);
  CHECK_EQ(kv_indptr.device(), device);
  CHECK_EQ(kv_last_page_len.device(), device);

  QKVLayout kv_layout = QKVLayout(layout);

  unsigned int num_heads, page_size, head_dim;
  head_dim = paged_k_cache.size(3);
  if (kv_layout == QKVLayout::kHND) {
    num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_heads = paged_k_cache.size(2);
  }

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  auto append_k_strides = append_key.strides();
  auto append_k_stride_n = append_k_strides[0];
  auto append_k_stride_h = append_k_strides[1];
  auto append_v_strides = append_value.strides();
  auto append_v_stride_n = append_v_strides[0];
  auto append_v_stride_h = append_v_strides[1];

  CHECK_EQ(append_key.size(1), num_heads);
  CHECK_EQ(append_key.size(2), head_dim);
  CHECK_EQ(append_value.size(1), num_heads);
  CHECK_EQ(append_value.size(2), head_dim);

  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());

  auto kv_scalar_dtype = paged_k_cache.scalar_type();

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(kv_scalar_dtype, c_type, [&] {
    paged_kv_t<c_type, int32_t> paged_kv(
        num_heads, page_size, head_dim, batch_size, kv_layout,
        static_cast<c_type*>(paged_k_cache.data_ptr()),
        static_cast<c_type*>(paged_v_cache.data_ptr()), kv_cache_strides,
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status = AppendPagedKVCache(paged_kv, static_cast<c_type*>(append_key.data_ptr()),
                                            static_cast<c_type*>(append_value.data_ptr()),
                                            static_cast<int32_t*>(batch_indices.data_ptr()),
                                            static_cast<int32_t*>(positions.data_ptr()), nnz,
                                            append_k_stride_n, append_k_stride_h, append_v_stride_n,
                                            append_v_stride_h, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "AppendPagedKVCache failed with error: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
}
