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

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           unsigned int layout, int64_t cuda_stream) {
  CHECK_LAST_DIM_CONTIGUOUS(append_key);
  CHECK_LAST_DIM_CONTIGUOUS(append_value);
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

  auto kv_scalar_dtype = paged_k_cache.scalar_type();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(kv_scalar_dtype, c_type, [&] {
    paged_kv_t<c_type, int32_t> paged_kv(
        num_heads, page_size, head_dim, batch_size, kv_layout,
        static_cast<c_type*>(paged_k_cache.data_ptr()),
        static_cast<c_type*>(paged_v_cache.data_ptr()), kv_cache_strides,
        static_cast<int32_t*>(kv_indices.data_ptr()), static_cast<int32_t*>(kv_indptr.data_ptr()),
        static_cast<int32_t*>(kv_last_page_len.data_ptr()));
    cudaError_t status =
        AppendPagedKVCache(paged_kv, static_cast<c_type*>(append_key.data_ptr()),
                           static_cast<c_type*>(append_value.data_ptr()),
                           static_cast<int32_t*>(batch_indices.data_ptr()),
                           static_cast<int32_t*>(positions.data_ptr()), nnz, append_k_stride_n,
                           append_k_stride_h, append_v_stride_n, append_v_stride_h, stream);
    TORCH_CHECK(status == cudaSuccess,
                "AppendPagedKVCache failed with error: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "AppendPagedKVCache failed to dispatch with dtype ", kv_scalar_dtype);
}

void block_sparse_indices_to_vector_sparse_offsets(at::Tensor block_sparse_indices,
                                                   at::Tensor block_sparse_indptr,
                                                   at::Tensor vector_sparse_offsets,
                                                   at::Tensor vector_sparse_indptr,
                                                   at::Tensor kv_len_arr, unsigned int stride_block,
                                                   unsigned int stride_n, unsigned int batch_size,
                                                   unsigned int block_size, int64_t cuda_stream) {
  CHECK_INPUT(block_sparse_indices);
  CHECK_INPUT(block_sparse_indptr);
  CHECK_INPUT(vector_sparse_offsets);
  CHECK_INPUT(vector_sparse_indptr);
  CHECK_INPUT(kv_len_arr);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  cudaError_t status = BlockSparseIndicesToVectorSparseOffset(
      static_cast<int32_t*>(block_sparse_indices.data_ptr()),
      static_cast<int32_t*>(block_sparse_indptr.data_ptr()),
      static_cast<int32_t*>(vector_sparse_offsets.data_ptr()),
      static_cast<int32_t*>(vector_sparse_indptr.data_ptr()),
      static_cast<int32_t*>(kv_len_arr.data_ptr()), stride_block, stride_n, batch_size, block_size,
      stream);

  TORCH_CHECK(status == cudaSuccess, "BlockSparseIndicesToVectorSparseOffset failed with error: ",
              cudaGetErrorString(status));
}
