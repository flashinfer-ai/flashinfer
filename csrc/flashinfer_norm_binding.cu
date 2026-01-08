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

void rmsnorm(TensorView out, TensorView input, TensorView weight, double eps, bool enable_pdl);

void rmsnorm_quant(TensorView out, TensorView input, TensorView weight, double scale, double eps,
                   bool enable_pdl);

void fused_add_rmsnorm(TensorView input, TensorView residual, TensorView weight, double eps,
                       bool enable_pdl);

void fused_add_rmsnorm_quant(TensorView output, TensorView input, TensorView residual,
                             TensorView weight, double scale, double eps, bool enable_pdl);

void gemma_rmsnorm(TensorView out, TensorView input, TensorView weight, double eps,
                   bool enable_pdl);

void gemma_fused_add_rmsnorm(TensorView input, TensorView residual, TensorView weight, double eps,
                             bool enable_pdl);

void layernorm(Tensor out, Tensor input, Tensor gamma, Tensor beta, double eps);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_quant, rmsnorm_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm, fused_add_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_add_rmsnorm_quant, fused_add_rmsnorm_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma_rmsnorm, gemma_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemma_fused_add_rmsnorm, gemma_fused_add_rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm, layernorm);
