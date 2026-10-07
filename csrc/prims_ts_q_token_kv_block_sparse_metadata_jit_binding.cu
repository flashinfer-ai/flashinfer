/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tvm_ffi_utils.h"

void PrimsTSQTokenKvBlockSparseMetadataRunFixed(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv, bool release_pdl);

void PrimsTSQTokenKvBlockSparseMetadataRunPacked(
    TensorView block_indices, TensorView block_table, TensorView token_to_request,
    TensorView query_positions, TensorView qo_indptr,
    TensorView q_token_kv_block_sparse_page_indices,
    TensorView q_token_kv_block_sparse_page_memberships, TensorView seq_lens, int64_t group_size,
    int64_t storage_page_size, int64_t sparse_block_size, int64_t max_seq_len_kv, bool release_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_fixed, PrimsTSQTokenKvBlockSparseMetadataRunFixed);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_packed, PrimsTSQTokenKvBlockSparseMetadataRunPacked);
