/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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

void FMHACutlassSM100Run(TensorView workspace_buffer, TensorView q, TensorView k, TensorView v,
                         TensorView qo_segment_offsets, TensorView kv_segment_offsets,
                         TensorView work_indptr, TensorView qo_tile_indices,
                         TensorView qo_head_indices, TensorView batch_indices, TensorView o,
                         Optional<TensorView> maybe_lse, int64_t mask_mode_code, double sm_scale,
                         double scale_q, double scale_k, double scale_v, double o_scale,
                         int64_t max_qo_len);

void blackwell_fmha_plan(TensorView qo_segment_offsets, TensorView kv_segment_offsets,
                         TensorView work_indptr, TensorView qo_tile_indices,
                         TensorView head_indices, TensorView batch_indices, int64_t qo_tile_size,
                         int64_t num_heads, int64_t num_buckets, bool causal);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, FMHACutlassSM100Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, blackwell_fmha_plan);
