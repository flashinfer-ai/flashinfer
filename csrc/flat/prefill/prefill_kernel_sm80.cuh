#pragma once

#include "cute/numeric/numeric_types.hpp"

#include "flat/common.hpp"
#include "cutlass/arch/arch.h"
#include "flat/prefill/pipeline_sm80.cuh"

namespace flat {

template <
    typename ArchTag,
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
    int32_t        sm_count
) {
  FLAT_UNUSED_PARAMETER(total_seqlen);
  FLAT_UNUSED_PARAMETER(sm_count);
  constexpr int NumThreads = 256;
  constexpr int BlockSize  = 32;

#define LAUNCH(head_size)                                                                                            \
  {                                                                                                                  \
    auto  bytes = linear_attention_prefill_kernel_smem_size<NumThreads, (head_size), BlockSize, TO, TQKV, TState>(); \
    void* ptr   = (void*)linear_attention_prefill_kernel<NumThreads, (head_size), BlockSize, TO, TQKV, TState>;      \
    CUDA_CHECK(cudaFuncSetAttribute(ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));                       \
                                                                                                                     \
    int32_t num_ctas = num_qo_heads * num_seqs;                                                                      \
    linear_attention_prefill_kernel<NumThreads, (head_size), BlockSize><<<num_ctas, NumThreads, bytes, stream>>>(    \
        output, output_state, q, k, v, input_state, cu_seqlens, num_seqs, num_qo_heads, num_kv_heads, scale,         \
        decay, per_head_deacy, decay_exponent_offset                                                                 \
    );                                                                                                               \
  }

  if (head_size == 128) {
    LAUNCH(128);
    // } else if (head_size == 64) {
    //   LAUNCH(64);
  } else {
    throw std::runtime_error("unsupported head size " + std::to_string(head_size));
  }
}

}  // namespace flat
