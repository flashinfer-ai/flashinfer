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

using tvm::ffi::Tensor;

void fused_qk_norm_rope(TensorView q, TensorView k, TensorView q_weight, TensorView k_weight,
                        TensorView pos_ids, int64_t rotary_dim, double eps, double rope_theta,
                        bool interleave, double yarn_factor, double yarn_low, double yarn_high,
                        double yarn_attention_factor, bool is_qk_norm);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qk_norm_rope, fused_qk_norm_rope);
