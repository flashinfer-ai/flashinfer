/*
 * Copyright (c) 2023-2026 by FlashInfer team.
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

// FMHAv2 paged-KV prepare JIT binding.
// Exposes prepare_paged() which converts paged KV indptr/indices/last_page_len
// into kv_lens, seq_lens_q, and a dense block_table — entirely on GPU.

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

void fmha_v2_prepare_paged(
    ffi::TensorView qo_indptr,
    ffi::TensorView paged_kv_indptr,
    ffi::TensorView paged_kv_last_page_len,
    ffi::TensorView paged_kv_indices,
    ffi::TensorView seq_lens_q_out,
    ffi::TensorView kv_lens_out,
    ffi::TensorView block_tables_out,
    int page_size, int batch_size, int max_blocks_per_seq);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_paged, fmha_v2_prepare_paged);
