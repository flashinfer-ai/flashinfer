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

void selective_state_update(
    TensorView state,              // (batch, dim, dstate) or (batch, nheads, dim, dstate)
    TensorView x,                  // (batch, dim) or (batch, nheads, dim) for single-token
                                   // or (batch, T, nheads, dim) for multi-token
                                   // or (total_tokens, nheads, dim) for varlen multi-token
    TensorView dt,                 // same layout as x
    TensorView A,                  // (dim, dstate) or (nheads, dim, dstate)
    TensorView B,                  // (batch, dstate) or (batch, ngroups, dstate) for single-token
                                   // or (batch, T, ngroups, dstate) for multi-token
                                   // or (total_tokens, ngroups, dstate) for varlen multi-token
    TensorView C,                  // same layout as B
    TensorView D,                  // (dim,) or (nheads, dim)
    Optional<TensorView> z,        // same layout as x
    Optional<TensorView> dt_bias,  // (dim,) or (nheads, dim)
    bool dt_softplus,
    Optional<TensorView> state_batch_indices,      // (batch,) or (N, max_seqlen)
    Optional<TensorView> dst_state_batch_indices,  // (batch,) or (N, max_seqlen)
    int64_t pad_slot_id,
    Optional<TensorView> state_scale,  // float32: (state_cache_size, nheads, dim)
    TensorView output,                 // same as x
    bool disable_state_update,
    Optional<TensorView> intermediate_states_buffer,  // (batch, cache_steps, nheads, dim, dstate)
    Optional<TensorView> intermediate_state_indices,  // (batch,)
    Optional<TensorView> intermediate_state_scales,   // float32: (batch, cache_steps, nheads, dim)
    Optional<TensorView> rand_seed,  // device-side int64 tensor for Philox rounding
    int64_t cache_steps,
    Optional<TensorView> cu_seqlens,           // (N + 1,)
    Optional<TensorView> num_accepted_tokens,  // (N,)
    Optional<TensorView> prev_tokens,          // (N,) int64: skip output for first prev_k steps
    int64_t algorithm);  // SSUAlgorithm: 0=auto, 1=simple, 2=vertical, 3=horizontal

}  // namespace flashinfer::mamba

// Export the function(s) via TVM-FFI
// This enables cross-language bindings (not just PyTorch)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(selective_state_update, flashinfer::mamba::selective_state_update);

// Add more mamba operations here as they are implemented
