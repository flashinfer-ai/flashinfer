/*
 * Copyright (c) 2024 by FlashInfer team.
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

using tvm::ffi::Optional;

void radix_topk(TensorView input, TensorView output_indices,
                Optional<TensorView> maybe_output_values,
                Optional<TensorView> maybe_row_states_buffer, int64_t top_k, bool sorted_output,
                bool deterministic, int64_t tie_break, bool dsa_graph_safe);

void radix_topk_page_table_transform(TensorView input, TensorView output_page_table,
                                     TensorView src_page_table,
                                     Optional<TensorView> maybe_row_to_batch, TensorView lengths,
                                     Optional<TensorView> maybe_row_states_buffer, int64_t top_k,
                                     bool deterministic, int64_t tie_break, int64_t page_size,
                                     bool dsa_graph_safe, Optional<TensorView> maybe_row_starts,
                                     Optional<TensorView> maybe_page_table_row_starts,
                                     Optional<TensorView> maybe_output_raw_indices);

void radix_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                                 TensorView lengths, Optional<TensorView> maybe_row_states_buffer,
                                 int64_t top_k, bool deterministic, int64_t tie_break,
                                 bool dsa_graph_safe, Optional<TensorView> maybe_row_starts);

bool can_implement_filtered_topk();

void cub_topk_page_table_transform(
    TensorView input, TensorView output_page_table, TensorView src_page_table, TensorView lengths,
    Optional<TensorView> maybe_output_raw_indices, Optional<TensorView> maybe_workspace_buffer,
    int64_t top_k, int64_t tie_break, int64_t page_size, Optional<TensorView> maybe_row_to_batch,
    Optional<TensorView> maybe_row_starts, Optional<TensorView> maybe_page_table_row_starts);

int64_t cub_topk_page_table_transform_workspace_size(TensorView input, TensorView lengths,
                                                     int64_t top_k, int64_t tie_break,
                                                     bool with_raw_indices);

void cub_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                               TensorView lengths, Optional<TensorView> maybe_workspace_buffer,
                               int64_t top_k, int64_t tie_break,
                               Optional<TensorView> maybe_row_starts);

int64_t cub_topk_ragged_transform_workspace_size(TensorView input, TensorView lengths,
                                                 int64_t top_k, int64_t tie_break);

// Radix-based Top-K selection
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk, radix_topk);

// Fused Top-K + Page Table Transform for sparse attention
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk_page_table_transform, radix_topk_page_table_transform);

// Fused Top-K + Ragged Index Transform for sparse attention
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk_ragged_transform, radix_topk_ragged_transform);

// Check if GPU supports FilteredTopK algorithm
TVM_FFI_DLL_EXPORT_TYPED_FUNC(can_implement_filtered_topk, can_implement_filtered_topk);

// CUB DeviceBatchedTopK-backed Top-K and its workspace size query
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cub_topk_page_table_transform, cub_topk_page_table_transform);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cub_topk_page_table_transform_workspace_size,
                              cub_topk_page_table_transform_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cub_topk_ragged_transform, cub_topk_ragged_transform);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cub_topk_ragged_transform_workspace_size,
                              cub_topk_ragged_transform_workspace_size);
