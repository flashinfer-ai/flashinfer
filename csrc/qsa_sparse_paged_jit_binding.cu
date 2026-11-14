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

using tvm::ffi::Optional;

void QsaSparsePagedAttentionRun(TensorView q, TensorView k_data, TensorView v_data,
                                Optional<TensorView> k_sf, Optional<TensorView> v_sf,
                                TensorView indices, TensorView block_table, TensorView token_to_req,
                                TensorView gate, TensorView out, double score_scale,
                                double output_scale, int64_t kv_format, int64_t num_splits);

int64_t QsaSparsePagedAttentionCtasPerSm(int64_t kv_format);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, QsaSparsePagedAttentionRun);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctas_per_sm, QsaSparsePagedAttentionCtasPerSm);
