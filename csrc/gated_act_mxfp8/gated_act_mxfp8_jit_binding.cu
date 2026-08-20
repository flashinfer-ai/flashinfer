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

namespace flashinfer::gated_act_mxfp8 {

void Forward(TensorView gated_input, TensorView row_output, TensorView col_output,
             TensorView row_scales, TensorView col_scales, bool rowwise, bool colwise);

void Backward(TensorView gated_input, TensorView grad_output, TensorView row_output,
              TensorView col_output, TensorView row_scales, TensorView col_scales,
              bool rowwise, bool colwise);

}  // namespace flashinfer::gated_act_mxfp8

TVM_FFI_DLL_EXPORT_TYPED_FUNC(forward, flashinfer::gated_act_mxfp8::Forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(backward, flashinfer::gated_act_mxfp8::Backward);
