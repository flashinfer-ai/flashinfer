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
#include "tvm_ffi_utils.h"

using tvm::ffi::Tensor;

void append_paged_kv_cache(Tensor append_key, Tensor append_value, Tensor batch_indices,
                           Tensor positions, Tensor paged_k_cache, Tensor paged_v_cache,
                           Tensor kv_indices, Tensor kv_indptr, Tensor kv_last_page_len,
                           int64_t layout);

void append_paged_mla_kv_cache(Tensor append_ckv, Tensor append_kpe, Tensor batch_indices,
                               Tensor positions, Tensor ckv_cache, Tensor kpe_cache,
                               Tensor kv_indices, Tensor kv_indptr, Tensor kv_last_page_len);

void block_sparse_indices_to_vector_sparse_offsets(Tensor block_sparse_indices,
                                                   Tensor block_sparse_indptr,
                                                   Tensor vector_sparse_offsets,
                                                   Tensor vector_sparse_indptr, Tensor kv_len_arr,
                                                   int64_t stride_block, int64_t stride_n,
                                                   int64_t batch_size, int64_t block_size);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(append_paged_kv_cache, append_paged_kv_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(append_paged_mla_kv_cache, append_paged_mla_kv_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_sparse_indices_to_vector_sparse_offsets,
                              block_sparse_indices_to_vector_sparse_offsets);
