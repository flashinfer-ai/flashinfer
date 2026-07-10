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

// FMHAv2 prepare JIT bindings.
// Exposes prepare() (ragged) and prepare_paged() (paged) as siblings of run()
// in the FMHAv2 JIT module so a single loaded module hosts both the prep
// launch (called once per plan) and the FMHA launch (called once per run).
// Both are instantiations of the same fused prep kernel in fmha_v2_prepare.cu;
// the paged variant additionally derives kv_lens and the dense block_tables
// from the paged indptr/indices in the same single launch.

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;
using tvm::ffi::Optional;

void fmha_v2_prepare(ffi::TensorView seq_lens_q, ffi::TensorView seq_lens_kv, int batch_size,
                     int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, double scale_bmm1,
                     double scale_bmm2, Optional<ffi::TensorView> cum_seq_lens_q,
                     Optional<ffi::TensorView> cum_seq_lens_kv,
                     Optional<ffi::TensorView> tile_id_counter,
                     Optional<ffi::TensorView> scale_bmm1_d,
                     Optional<ffi::TensorView> scale_bmm2_d);

void fmha_v2_prepare_paged(ffi::TensorView qo_indptr, ffi::TensorView paged_kv_indptr,
                           ffi::TensorView paged_kv_last_page_len, ffi::TensorView paged_kv_indices,
                           ffi::TensorView kv_lens_out, ffi::TensorView block_tables_out,
                           int page_size, int max_blocks_per_seq, int batch_size,
                           int scale_bmm1_dtype_code, int scale_bmm2_dtype_code, double scale_bmm1,
                           double scale_bmm2, Optional<ffi::TensorView> cum_seq_lens_q,
                           Optional<ffi::TensorView> cum_seq_lens_kv,
                           Optional<ffi::TensorView> tile_id_counter,
                           Optional<ffi::TensorView> scale_bmm1_d,
                           Optional<ffi::TensorView> scale_bmm2_d);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, fmha_v2_prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_paged, fmha_v2_prepare_paged);
