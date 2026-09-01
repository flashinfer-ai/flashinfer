/*
 * Copyright (c) 2026 by FlashInfer team.
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

void expand_block_route(TensorView block_indices, TensorView query_positions,
                        TensorView sequence_lengths, TensorView token_to_req, TensorView out,
                        int64_t compress_ratio);

void qsa_route_from_blocks(TensorView block_indices, TensorView query_positions,
                           TensorView sequence_lengths, TensorView token_to_req,
                           TensorView block_table, TensorView out_logical, TensorView out_route,
                           TensorView out_mask, int64_t compress_ratio, int64_t page_size,
                           int64_t num_slots);

void qsa_route_from_logical(TensorView logical, TensorView token_to_req, TensorView block_table,
                            TensorView out_route, TensorView out_mask, int64_t valid_rows,
                            int64_t page_size, int64_t num_slots);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(expand_block_route, expand_block_route);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qsa_route_from_logical, qsa_route_from_logical);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(qsa_route_from_blocks, qsa_route_from_blocks);
