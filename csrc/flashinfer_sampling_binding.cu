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

void softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
             Optional<TensorView> maybe_temperature_arr, double temperature_val, bool enable_pdl);

void sampling_from_probs(TensorView probs, TensorView output, Optional<TensorView> maybe_indices,
                         bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void sampling_from_logits(TensorView logits, TensorView output, Optional<TensorView> maybe_indices,
                          bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void top_p_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void top_k_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void min_p_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_min_p_arr, double min_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);

void top_k_top_p_sampling_from_probs(TensorView probs, TensorView output,
                                     Optional<TensorView> maybe_indices,
                                     Optional<TensorView> maybe_top_k_arr, double top_k_val,
                                     Optional<TensorView> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, uint64_t philox_seed,
                                     uint64_t philox_offset);

void top_p_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_p_arr, double top_p_val);

void top_k_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                        TensorView row_states_buffer);

void top_k_mask_logits(TensorView logits, TensorView mask_logits,
                       Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                       TensorView row_states_buffer);

void chain_speculative_sampling(TensorView draft_probs, TensorView draft_token_ids,
                                TensorView target_probs, TensorView output_token_ids,
                                TensorView output_accepted_token_num,
                                TensorView output_emitted_draft_token_num, bool deterministic,
                                uint64_t philox_seed, uint64_t philox_offset);

// Softmax
TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, softmax);
// Sample from probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sampling_from_probs, sampling_from_probs);
// Sample from logits
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sampling_from_logits, sampling_from_logits);
// Top-k sampling from probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_k_sampling_from_probs, top_k_sampling_from_probs);
// Min-p sampling from probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(min_p_sampling_from_probs, min_p_sampling_from_probs);
// Top-p sampling from probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_p_sampling_from_probs, top_p_sampling_from_probs);
// Top-k and top-p sampling from probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_k_top_p_sampling_from_probs, top_k_top_p_sampling_from_probs);
// Renormalize probabilities by top-k mask
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_k_renorm_probs, top_k_renorm_probs);
// Renormalize probabilities by top-p mask
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_p_renorm_probs, top_p_renorm_probs);
// Mask logits by top-k mask
TVM_FFI_DLL_EXPORT_TYPED_FUNC(top_k_mask_logits, top_k_mask_logits);
// Speculative sampling from sequence of probabilities
TVM_FFI_DLL_EXPORT_TYPED_FUNC(chain_speculative_sampling, chain_speculative_sampling);
