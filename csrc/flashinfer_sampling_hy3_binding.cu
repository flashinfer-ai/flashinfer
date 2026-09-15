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

void fused_sampling_from_logits_hy3(
    TensorView workspace_buffer, TensorView logits, TensorView output,
    Optional<TensorView> maybe_penalty_mask, Optional<TensorView> maybe_slot_id,
    Optional<TensorView> maybe_repetition_penalty, double repetition_penalty_val,
    Optional<TensorView> maybe_temperature, double temperature_val, int64_t softmax_policy,
    Optional<TensorView> maybe_top_k, int64_t top_k_val, Optional<TensorView> maybe_top_p,
    double top_p_val, int64_t max_top_k, Optional<TensorView> maybe_gumbel_noise,
    Optional<TensorView> maybe_draft_token_ids, int64_t sm_count, uint64_t seed, uint64_t offset,
    bool temperature_only);

// HY3 fused repetition penalty, temperature, top-k/top-p and sampling,
// optimized for NVIDIA SM100/B200.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_sampling_from_logits_hy3, fused_sampling_from_logits_hy3);
