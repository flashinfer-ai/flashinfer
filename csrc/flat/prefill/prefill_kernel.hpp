#pragma once

#include <cstdint>
#include "cuda_runtime_api.h"

namespace flat {

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState>
void launch_linear_attention_prefill_kernel(
    cudaStream_t   stream,
    TO*            output,
    TState*        output_state,
    TQKV const*    q,
    TQKV const*    k,
    TQKV const*    v,
    TState const*  input_state,
    int64_t const* cu_seqlens,
    int32_t        num_seqs,
    int32_t        num_qo_heads,
    int32_t        num_kv_heads,
    int32_t        head_size,
    int64_t        total_seqlen,
    float          scale,
    float          decay,
    float const*   per_head_deacy,
    int32_t        decay_exponent_offset,
    int32_t        sm_count = 0
);

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState>
void launch_delta_rule_prefill_kernel(
    cudaStream_t   stream,
    TO*            output,
    TState*        output_state,
    TQKV const*    q,
    TQKV const*    k,
    TQKV const*    v,
    TState const*  input_state,
    float const*   alpha,
    float const*   beta,
    int64_t const* cu_seqlens,
    int32_t        num_seqs,
    int32_t        num_q_heads,
    int32_t        num_k_heads,
    int32_t        num_v_heads,
    int32_t        num_o_heads,
    int32_t        head_size,
    int64_t        total_seqlen,
    float          scale,
    int32_t        sm_count = 0
);

}  // namespace flat
