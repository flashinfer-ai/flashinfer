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

void qsa_pre_indexer(TensorView q, TensorView k, TensorView positions, TensorView cos_sin_cache,
                     TensorView q_norm_weight, TensorView k_norm_weight, double eps,
                     TensorView q_out, TensorView state_cache, TensorView state_slots,
                     TensorView state_block_table, TensorView query_start_loc,
                     TensorView logical_positions, TensorView compressed_cache,
                     TensorView compressed_slots, TensorView work_metadata, int64_t compress_ratio,
                     int64_t mrope_h, int64_t mrope_w, bool is_k_mrope, bool cache_has_rope_pos);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(qsa_pre_indexer, qsa_pre_indexer);
