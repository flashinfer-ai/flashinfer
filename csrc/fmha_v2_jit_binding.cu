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

// FMHAv2 JIT Binding
// This file exports the fmha_v2_run function via TVM FFI

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void fmha_v2_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView o,
                 Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code, float scale_softmax,
                 float scale_bmm1, float scale_bmm2, float softcapping_scale);

// FMHAv2 attention operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fmha_v2_run);
