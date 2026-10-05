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

using tvm::ffi::TensorView;

void nvfp4_sparse_mla_decode_run(TensorView kv_cache, TensorView query, TensorView indices,
                                 TensorView out, int64_t num_ctas_per_token, double sm_scale_log2,
                                 double output_scale);

int64_t nvfp4_sparse_mla_decode_max_active_clusters(int64_t num_ctas_per_token);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, nvfp4_sparse_mla_decode_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_active_clusters, nvfp4_sparse_mla_decode_max_active_clusters);
