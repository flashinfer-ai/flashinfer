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

void append_paged_kv_cache(TensorView append_key, TensorView append_value, TensorView batch_indices,
                           TensorView positions, TensorView paged_k_cache, TensorView paged_v_cache,
                           TensorView kv_indices, TensorView kv_indptr, TensorView kv_last_page_len,
                           int64_t layout);

void append_paged_mla_kv_cache(TensorView append_ckv, TensorView append_kpe,
                               TensorView batch_indices, TensorView positions, TensorView ckv_cache,
                               TensorView kpe_cache, TensorView kv_indices, TensorView kv_indptr,
                               TensorView kv_last_page_len);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(append_paged_kv_cache, append_paged_kv_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(append_paged_mla_kv_cache, append_paged_mla_kv_cache);
