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

void radix_topk(TensorView input, TensorView output_indices, TensorView output_values,
                Optional<TensorView> maybe_row_states_buffer, int64_t top_k,
                int64_t deterministic_mode);

void radix_topk_page_table_transform(TensorView input, TensorView output_page_table,
                                     TensorView src_page_table,
                                     Optional<TensorView> maybe_row_to_batch, TensorView lengths,
                                     Optional<TensorView> maybe_row_states_buffer, int64_t top_k,
                                     int64_t deterministic_mode);

void radix_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                                 TensorView lengths, Optional<TensorView> maybe_row_states_buffer,
                                 int64_t top_k, int64_t deterministic_mode);

bool can_implement_filtered_topk();

// Radix-based Top-K selection
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk, radix_topk);

// Fused Top-K + Page Table Transform for sparse attention
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk_page_table_transform, radix_topk_page_table_transform);

// Fused Top-K + Ragged Index Transform for sparse attention
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radix_topk_ragged_transform, radix_topk_ragged_transform);

// Check if GPU supports FilteredTopK algorithm
TVM_FFI_DLL_EXPORT_TYPED_FUNC(can_implement_filtered_topk, can_implement_filtered_topk);
