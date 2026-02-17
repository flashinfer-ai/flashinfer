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
#include <cuda_bf16.h>

#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

// Extern template declarations prevent implicit instantiation here.
// Explicit instantiations are in separate generated files for parallel compilation.
#include "flat_prefill_kernel_delta_rule_sm90_extern.inc"

namespace flat {

using namespace cute;

template <typename ArchTag,  // FIXME: hide this
          typename TO, typename TQKV, typename TState>
void launch_delta_rule_prefill_kernel(cudaStream_t stream, TO* output, TState* output_state,
                                      TQKV const* q, TQKV const* k, TQKV const* v,
                                      TState const* input_state, float const* alpha,
                                      float const* beta, int64_t const* cu_seqlens,
                                      uint8_t* workspace_buffer, int32_t num_seqs,
                                      int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads,
                                      int32_t num_o_heads, int32_t head_size, int64_t total_seqlen,
                                      float scale, int32_t sm_count) {
  bool is_gva = num_v_heads > num_q_heads;
  bool needs_beta = beta != nullptr;
  bool needs_alpha = alpha != nullptr;
  bool init_state = input_state != nullptr;

#define LAUNCH(is_gva, needs_beta, needs_alpha, init_state)                                      \
  launch_delta_rule_prefill_kernel_gbai<is_gva, needs_beta, needs_alpha, init_state, ArchTag>(   \
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens,               \
      workspace_buffer, num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, \
      total_seqlen, scale, sm_count);

  if (init_state) {
    if (is_gva && needs_beta && needs_alpha) {
      LAUNCH(true, true, true, true);
    } else if (is_gva && needs_beta && !needs_alpha) {
      LAUNCH(true, true, false, true);
    } else if (is_gva && !needs_beta && needs_alpha) {
      LAUNCH(true, false, true, true);
    } else if (is_gva && !needs_beta && !needs_alpha) {
      LAUNCH(true, false, false, true);
    } else if (!is_gva && needs_beta && needs_alpha) {
      LAUNCH(false, true, true, true);
    } else if (!is_gva && needs_beta && !needs_alpha) {
      LAUNCH(false, true, false, true);
    } else if (!is_gva && !needs_beta && needs_alpha) {
      LAUNCH(false, false, true, true);
    } else if (!is_gva && !needs_beta && !needs_alpha) {
      LAUNCH(false, false, false, true);
    } else {
      throw std::runtime_error("unreachable");
    }
  } else {
    if (is_gva && needs_beta && needs_alpha) {
      LAUNCH(true, true, true, false);
    } else if (is_gva && needs_beta && !needs_alpha) {
      LAUNCH(true, true, false, false);
    } else if (is_gva && !needs_beta && needs_alpha) {
      LAUNCH(true, false, true, false);
    } else if (is_gva && !needs_beta && !needs_alpha) {
      LAUNCH(true, false, false, false);
    } else if (!is_gva && needs_beta && needs_alpha) {
      LAUNCH(false, true, true, false);
    } else if (!is_gva && needs_beta && !needs_alpha) {
      LAUNCH(false, true, false, false);
    } else if (!is_gva && !needs_beta && needs_alpha) {
      LAUNCH(false, false, true, false);
    } else if (!is_gva && !needs_beta && !needs_alpha) {
      LAUNCH(false, false, false, false);
    } else {
      throw std::runtime_error("unreachable");
    }
  }

#undef LAUNCH
}

// Explicit instantiations for the outer dispatch function only.
// The inner launch_delta_rule_prefill_kernel_gbai instantiations are in separate files.
template void launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, half, half, float>(
    cudaStream_t stream, half* output, float* state, half const* q, half const* k, half const* v,
    float const* input_state, float const* alpha, float const* beta, int64_t const* cu_seqlens,
    uint8_t* workspace_buffer, int32_t num_seqs, int32_t num_q_heads, int32_t num_k_heads,
    int32_t num_v_heads, int32_t num_o_heads, int32_t head_size, int64_t total_seqlen, float scale,
    int32_t sm_count);

template void
launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, nv_bfloat16, nv_bfloat16, float>(
    cudaStream_t stream, nv_bfloat16* output, float* state, nv_bfloat16 const* q,
    nv_bfloat16 const* k, nv_bfloat16 const* v, float const* input_state, float const* alpha,
    float const* beta, int64_t const* cu_seqlens, uint8_t* workspace_buffer, int32_t num_seqs,
    int32_t num_q_heads, int32_t num_k_heads, int32_t num_v_heads, int32_t num_o_heads,
    int32_t head_size, int64_t total_seqlen, float scale, int32_t sm_count);

}  // namespace flat
