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

void apply_rope(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope, TensorView indptr,
                TensorView offsets, int64_t rotary_dim, bool interleave, double rope_scale,
                double rope_theta);

void apply_llama31_rope(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                        TensorView indptr, TensorView offsets, int64_t rotary_dim, bool interleave,
                        double rope_scale, double rope_theta, double low_freq_factor,
                        double high_freq_factor, double old_context_length);

void apply_rope_pos_ids(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                        TensorView pos_ids, int64_t rotary_dim, bool interleave, double rope_scale,
                        double rope_theta);

void apply_llama31_rope_pos_ids(TensorView q, TensorView k, TensorView q_rope, TensorView k_rope,
                                TensorView pos_ids, int64_t rotary_dim, bool interleave,
                                double rope_scale, double rope_theta, double low_freq_factor,
                                double high_freq_factor, double old_context_length);

void apply_rope_pos_ids_cos_sin_cache(TensorView q, TensorView k, TensorView q_rope,
                                      TensorView k_rope, TensorView cos_sin_cache,
                                      TensorView pos_ids, bool interleave);

void rope_quantize(TensorView q_rope_in, TensorView k_rope_in, TensorView q_nope_in,
                   TensorView k_nope_in, TensorView q_rope_out, TensorView k_rope_out,
                   TensorView q_nope_out, TensorView k_nope_out, TensorView cos_sin_cache,
                   TensorView pos_ids, double quant_scale_q, double quant_scale_kv, bool interleave,
                   bool enable_pdl);

void rope_quantize_append_paged_kv_cache(
    TensorView q_rope_in, TensorView k_rope_in, TensorView q_nope_in, TensorView k_nope_in,
    TensorView v_in, TensorView q_rope_out, TensorView q_nope_out, TensorView cos_sin_cache,
    TensorView pos_ids, TensorView k_cache, TensorView v_cache, TensorView ckv_cache,
    TensorView kpe_cache, TensorView kv_indices, TensorView kv_indptr, TensorView batch_indices,
    TensorView positions, int64_t kv_layout_code, int64_t page_size, double quant_scale_q,
    double quant_scale_kv, bool interleave, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope, apply_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_llama31_rope, apply_llama31_rope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope_pos_ids, apply_rope_pos_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_llama31_rope_pos_ids, apply_llama31_rope_pos_ids);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_rope_pos_ids_cos_sin_cache, apply_rope_pos_ids_cos_sin_cache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rope_quantize, rope_quantize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rope_quantize_append_paged_kv_cache,
                              rope_quantize_append_paged_kv_cache);
