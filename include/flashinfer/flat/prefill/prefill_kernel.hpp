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
#pragma once

#include <cstdint>

#include "cuda_runtime_api.h"

// Forward declarations to avoid including full cutlass headers
namespace cutlass::arch {
struct Sm90;
}  // namespace cutlass::arch

namespace flat {

template <typename ArchTag,  // TODO: hide this
          typename TO, typename TQKV, typename TState>
void launch_delta_rule_prefill_kernel(cudaStream_t stream, TO* output, TState* output_state,
                                      TQKV const* q, TQKV const* k, TQKV const* v,
                                      TState const* input_state, float const* alpha,
                                      float const* beta, int64_t const* cu_seqlens,
                                      uint8_t* workspace_buffer, int32_t num_seqs,
                                      int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
                                      int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
                                      float scale, int32_t sm_count = 0);

}  // namespace flat
