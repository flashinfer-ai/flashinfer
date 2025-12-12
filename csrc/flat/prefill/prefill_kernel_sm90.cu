#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

#include "flat/common.hpp"

namespace flat {

using namespace cute;

template <
    bool NeedsScale,
    bool NeedsDecay,
    bool InitStateFromInput,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState>
void launch_linear_attention_prefill_kernel_sdi(
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
    float const*   per_head_decay,
    int32_t        decay_exponent_offset,
    int32_t        sm_count
);

template <
    typename ArchTag,  // FIXME: hide this
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
    float const*   per_head_decay,
    int32_t        decay_exponent_offset,
    int32_t        sm_count
) {
  bool needs_scale = scale != 1.0f;
  bool needs_decay = (decay != 1.0f) || (per_head_decay != nullptr);
  bool init_state  = input_state != nullptr;

  if (init_state) {
    if (needs_scale && needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<true, true, true, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, scale, decay, per_head_decay, decay_exponent_offset, sm_count
      );
    } else if (needs_scale && !needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<true, false, true, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, scale, 0.0f, nullptr, 0, sm_count
      );
    } else if (!needs_scale && needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<false, true, true, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, 0.0f, decay, per_head_decay, decay_exponent_offset, sm_count
      );
    } else {
      launch_linear_attention_prefill_kernel_sdi<false, false, true, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, 0.0f, 0.0f, nullptr, 0, sm_count
      );
    }
  } else {
    if (needs_scale && needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<true, true, false, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, scale, decay, per_head_decay, decay_exponent_offset, sm_count
      );
    } else if (needs_scale && !needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<true, false, false, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, scale, 0.0f, nullptr, 0, sm_count
      );
    } else if (!needs_scale && needs_decay) {
      launch_linear_attention_prefill_kernel_sdi<false, true, false, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, 0.0f, decay, per_head_decay, decay_exponent_offset, sm_count
      );
    } else {
      launch_linear_attention_prefill_kernel_sdi<false, false, false, ArchTag>(
          stream, output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, head_size, total_seqlen, 0.0f, 0.0f, nullptr, 0, sm_count
      );
    }
  }
}

template void launch_linear_attention_prefill_kernel<cutlass::arch::Sm90, half, half, float>(
    cudaStream_t   stream,
    half*          output,
    float*         output_state,
    half const*    q,
    half const*    k,
    half const*    v,
    float const*   input_state,
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
    int32_t        sm_count
);

using bf16 = cute::bfloat16_t;

template void launch_linear_attention_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t   stream,
    bf16*          output,
    float*         output_state,
    bf16 const*    q,
    bf16 const*    k,
    bf16 const*    v,
    float const*   input_state,
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
    int32_t        sm_count
);


}  // namespace flat
