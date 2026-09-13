// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#include <flashinfer/attention/sparse_mla_sm120/execution/dsv4_nvfp4_launch.cuh>

#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

cudaError_t dispatch_attention(const Dsv4Nvfp4AttentionParams& p, const execution::ExecutionPlan& plan,
                               cudaStream_t stream) {
  const bool dual = p.extra_cache != nullptr;
#define DISPATCH(H, K)                                                                      \
  if (plan.specialized_heads == H && plan.specialized_topk == K) {                          \
    if (plan.implementation == execution::Implementation::Dsv4Nvfp4Prefill)                    \
      return dual ? launch_prefill<H, K, execution::FixedPageSize, true>(p, plan, stream)   \
                  : launch_prefill<H, K, execution::FixedPageSize, false>(p, plan, stream); \
    return dual ? launch_decode<H, K, execution::FixedPageSize, true>(p, plan, stream)      \
                : launch_decode<H, K, execution::FixedPageSize, false>(p, plan, stream);    \
  }
  SPARSE_MLA_DSV4_NVFP4_INSTANCES(DISPATCH)
#undef DISPATCH
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
