#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/arch.h"

#include "flat/common.hpp"

namespace flat {

using namespace cute;

template <
    bool IsGVA,
    bool NeedsBeta,
    bool NeedsAlpha,
    bool InitStateFromInput,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState>
void launch_delta_rule_prefill_kernel_gbai(
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
    int32_t        sm_count
);

template <
    typename ArchTag,  // FIXME: hide this
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
    int32_t        sm_count
) {
  bool is_gva      = num_v_heads > num_q_heads;
  bool needs_beta  = beta != nullptr;
  bool needs_alpha = alpha != nullptr;
  bool init_state  = input_state != nullptr;

#define LAUNCH(is_gva, needs_beta, needs_alpha, init_state)                                                  \
  launch_delta_rule_prefill_kernel_gbai<is_gva, needs_beta, needs_alpha, init_state, ArchTag>(               \
      stream, output, output_state, q, k, v, input_state, alpha, beta, cu_seqlens,                           \
      num_seqs, num_q_heads, num_k_heads, num_v_heads, num_o_heads, head_size, total_seqlen, scale, sm_count \
  );

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

template void launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, half, half, float>(
    cudaStream_t   stream,
    half*          output,
    float*         state,
    half const*    q,
    half const*    k,
    half const*    v,
    float const*   input_state,
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
    int32_t        sm_count
);

using bf16 = cute::bfloat16_t;

template void launch_delta_rule_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t   stream,
    bf16*          output,
    float*         state,
    bf16 const*    q,
    bf16 const*    k,
    bf16 const*    v,
    float const*   input_state,
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
    int32_t        sm_count
);

}  // namespace flat
