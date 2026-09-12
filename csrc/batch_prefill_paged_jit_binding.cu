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
#include "batch_prefill_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

void BatchPrefillWithPagedKVCacheRun(TensorView float_workspace_buffer,
                                     TensorView int_workspace_buffer, Array<int64_t> plan_info_vec,
                                     TensorView q, TensorView paged_k_cache,
                                     TensorView paged_v_cache, TensorView qo_indptr,
                                     TensorView paged_kv_indptr, TensorView paged_kv_indices,
                                     TensorView paged_kv_last_page_len, TensorView o,
                                     Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                                     int64_t layout, int64_t window_left,
                                     bool enable_pdl ADDITIONAL_FUNC_PARAMS);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, BatchPrefillWithPagedKVCacheRun);
