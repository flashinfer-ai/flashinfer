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

using tvm::ffi::Tensor;

void apply_rope(Tensor q, Tensor k, Tensor q_rope, Tensor k_rope, Tensor indptr, Tensor offsets,
                int64_t rotary_dim, bool interleave, double rope_scale, double rope_theta);

void apply_llama31_rope(Tensor q, Tensor k, Tensor q_rope, Tensor k_rope, Tensor indptr,
                        Tensor offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta, double low_freq_factor, double high_freq_factor,
                        double old_context_length);

void apply_rope_pos_ids(Tensor q, Tensor k, Tensor q_rope, Tensor k_rope, Tensor pos_ids,
                        int64_t rotary_dim, bool interleave, double rope_scale, double rope_theta);

void apply_llama31_rope_pos_ids(Tensor q, Tensor k, Tensor q_rope, Tensor k_rope, Tensor pos_ids,
                                int64_t rotary_dim, bool interleave, double rope_scale,
                                double rope_theta, double low_freq_factor, double high_freq_factor,
                                double old_context_length);

void apply_rope_pos_ids_cos_sin_cache(Tensor q, Tensor k, Tensor q_rope, Tensor k_rope,
                                      Tensor cos_sin_cache, Tensor pos_ids, bool interleave);

void mla_rope_quantize(Tensor q_rope_in, Tensor k_rope_in, Tensor q_nope_in, Tensor k_nope_in,
                       Tensor q_rope_out, Tensor k_rope_out, Tensor q_nope_out, Tensor k_nope_out,
                       Tensor cos_sin_cache, Tensor pos_ids, double quant_scale_q,
                       double quant_scale_kv, bool interleave);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope, apply_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_llama31_rope, apply_llama31_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope_pos_ids, apply_rope_pos_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_llama31_rope_pos_ids, apply_llama31_rope_pos_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope_pos_ids_cos_sin_cache, apply_rope_pos_ids_cos_sin_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mla_rope_quantize, mla_rope_quantize);