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
// #include "pytorch_extension_utils.h"
#include "tvm_ffi_utils.h"

namespace flashinfer_norm {

using tvm::ffi::Tensor;

void rmsnorm(Tensor out, Tensor input, Tensor weight, double eps, bool enable_pdl);

// void fused_add_rmsnorm(ffi::Tensor& input, ffi::Tensor& residual, ffi::Tensor& weight, double
// eps,
//                        bool enable_pdl);

// void gemma_rmsnorm(ffi::Tensor& out, ffi::Tensor& input, ffi::Tensor& weight, double eps,
//                    bool enable_pdl);

// void gemma_fused_add_rmsnorm(ffi::Tensor& input, ffi::Tensor& residual, ffi::Tensor& weight,
//                              double eps, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm, flashinfer_norm::rmsnorm);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm, flashinfer_norm::fused_add_rmsnorm);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma_rmsnorm, flashinfer_norm::gemma_rmsnorm);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma_fused_add_rmsnorm, flashinfer_norm::gemma_fused_add_rmsnorm);
}  // namespace flashinfer_norm
