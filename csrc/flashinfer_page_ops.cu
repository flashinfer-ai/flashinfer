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
#include "pytorch_extension_utils.h"

void append_paged_kv_cache(at::Tensor append_key, at::Tensor append_value, at::Tensor batch_indices,
                           at::Tensor positions, at::Tensor paged_k_cache, at::Tensor paged_v_cache,
                           at::Tensor kv_indices, at::Tensor kv_indptr, at::Tensor kv_last_page_len,
                           int64_t layout);

void append_paged_mla_kv_cache(at::Tensor append_ckv, at::Tensor append_kpe,
                               at::Tensor batch_indices, at::Tensor positions, at::Tensor ckv_cache,
                               at::Tensor kpe_cache, at::Tensor kv_indices, at::Tensor kv_indptr,
                               at::Tensor kv_last_page_len);

void block_sparse_indices_to_vector_sparse_offsets(
    at::Tensor block_sparse_indices, at::Tensor block_sparse_indptr,
    at::Tensor vector_sparse_offsets, at::Tensor vector_sparse_indptr, at::Tensor kv_len_arr,
    int64_t stride_block, int64_t stride_n, int64_t batch_size, int64_t block_size);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // "Append paged KV-Cache operator"
  m.def("append_paged_kv_cache", append_paged_kv_cache);
  // "Append paged MLA KV-Cache operator"
  m.def("append_paged_mla_kv_cache", append_paged_mla_kv_cache);
  // "Precompute block sparse offsets"
  m.def("block_sparse_indices_to_vector_sparse_offsets",
        block_sparse_indices_to_vector_sparse_offsets);
}
