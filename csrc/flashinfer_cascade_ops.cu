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

using tvm::ffi::Optional;

void merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor v_merged, Tensor s_merged);

void merge_state_in_place(Tensor v, Tensor s, Tensor v_other, Tensor s_other,
                          Optional<Tensor> mask);

void merge_states(Tensor v, Tensor s, Tensor v_merged, Tensor s_merged);

// Merge two self-attention states
TVM_FFI_DLL_EXPORT_TYPED_FUNC(merge_state, merge_state);
// Merge another self-attention state in-place.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(merge_state_in_place, merge_state_in_place);
// "Merge multiple self-attention states"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(merge_states, merge_states);
