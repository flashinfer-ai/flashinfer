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

void vibecuda_ssd_combined_fwd(TensorView x, TensorView dt, Optional<TensorView> dt_bias,
                               TensorView a, TensorView b, TensorView c, Optional<TensorView> d,
                               Optional<TensorView> z, Optional<TensorView> initial,
                               Optional<TensorView> seq_idx, TensorView state_in, TensorView out,
                               TensorView final_states, int64_t softplus, double dt_lo,
                               double dt_hi, int64_t d_has_hdim, int64_t varlen,
                               int64_t all_single_host);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vibecuda_ssd_combined_fwd, vibecuda_ssd_combined_fwd);
