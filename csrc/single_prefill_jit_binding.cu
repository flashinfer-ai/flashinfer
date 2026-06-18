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
#include "single_prefill_config.inc"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void single_prefill_with_kv_cache(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v,
                                  ffi::TensorView tmp, ffi::TensorView o,
                                  Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code,
                                  int64_t layout, int64_t window_left ADDITIONAL_FUNC_PARAMS);

// Single-request prefill attention with KV-Cache operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, single_prefill_with_kv_cache);
