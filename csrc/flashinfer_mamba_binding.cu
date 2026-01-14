/*
 * Copyright (c) 2025 by FlashInfer team.
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

// Declare the function(s) implemented in selective_state_update.cu
using tvm::ffi::Optional;

namespace flashinfer::mamba {

void selective_state_update(TensorView state, TensorView x, TensorView dt, TensorView output,
                            TensorView A, TensorView B, TensorView C, TensorView D,
                            Optional<TensorView> z, Optional<TensorView> dt_bias, bool dt_softplus,
                            Optional<TensorView> state_batch_indices, int64_t pad_slot_id);

}  // namespace flashinfer::mamba

// Export the function(s) via TVM-FFI
// This enables cross-language bindings (not just PyTorch)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(selective_state_update, flashinfer::mamba::selective_state_update);

// Add more mamba operations here as they are implemented
