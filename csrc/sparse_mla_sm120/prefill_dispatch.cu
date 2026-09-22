// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <flashinfer/attention/sparse_mla_sm120/execution/prefill_launch.cuh>

#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120 {

PrefillLaunchResult dispatch_prefill(const execution::AttentionParams& p,
                                     const execution::ExecutionPlan& plan, cudaStream_t stream) {
  const auto& m = plan.metadata;
  const auto mt = static_cast<ModelType>(m.model);
  if (plan.implementation == execution::Implementation::FullBF16) {
    return execution::visit_decode_heads<ModelType::DSV4_1>(m.heads, [&](auto head) {
      constexpr int H = decltype(head)::value;
      return m.extra_fp4 ? launch_dsv41_bf16_prefill<H, true>(p, stream)
                         : launch_dsv41_bf16_prefill<H, false>(p, stream);
    });
  }
  PrefillColdParams cold{p.sm_scale,          m.tokens,
                         m.page_stride_bytes, m.extra_page_stride_bytes,
                         m.lse_stride,        m.topk,
                         m.extra_topk,        p.attn_sink,
                         p.topk_length,       p.extra_topk_length,
                         p.extra_kv,          p.extra_indices,
                         m.extra_page_size,   m.page_size};
  cold.lse_scale = p.lse_scale;
  if (cache_format_info(mt).inline_scale) cold.kv_stride_bytes = m.row_stride_bytes;
  if (mt == ModelType::DSV4 || mt == ModelType::DOTS3_SWA) {
    cold.main_div = flashinfer::uint_fastdiv(uint32_t(m.page_size));
    if (m.extra_topk > 0) cold.extra_div = flashinfer::uint_fastdiv(uint32_t(m.extra_page_size));
  }
#define SINGLE_ARGS m.heads, p.q, p.kv, p.indices, p.attn_sink, p.output, p.out_lse, stream
#define V32(FN)                                            \
  switch (mt) {                                            \
    case ModelType::DSV3_2:                                \
      return FN<ModelType::DSV3_2>(SINGLE_ARGS, cold);     \
    case ModelType::GLM_NSA:                               \
      return FN<ModelType::GLM_NSA>(SINGLE_ARGS, cold);    \
    case ModelType::GLM53_NOPE:                            \
      return FN<ModelType::GLM53_NOPE>(SINGLE_ARGS, cold); \
    default:                                               \
      return false;                                        \
  }
  switch (plan.implementation) {
    case execution::Implementation::SwapAB: {
      V32(dispatch_v32_swapab);
    }
    case execution::Implementation::SG:
    case execution::Implementation::MixedCache: {
      if (mt == ModelType::GLM53_NOPE && plan.numeric == execution::NumericRoute::QkBF16PvFP8) {
        // The resolver limits the strict BF16-QK route to this single-group
        // specialization, including short queries that normally use split-K.
        return launch_prefill_sg<ModelType::GLM53_NOPE, QkComputeMode::BF16, 16, 64>(
            p.q, p.kv, p.indices, p.attn_sink, p.output, p.out_lse, stream, cold);
      }
      if (mt == ModelType::DSV4_1) {
        if (plan.implementation == execution::Implementation::MixedCache)
          return dispatch_dsv41_sg<Dsv41MixedCachePrefillSchedule>(SINGLE_ARGS, cold);
        return dispatch_dsv41_sg<Dsv41PrefillGatherSchedule>(SINGLE_ARGS, cold);
      }
      if (mt == ModelType::DOTS3_SWA) return dispatch_dots3_swa_sg(SINGLE_ARGS, cold);
      V32(dispatch_v32_sg);
    }
    case execution::Implementation::MG:
    case execution::Implementation::FullTile: {
      if (m.extra_topk > 0)
        return dispatch_dsv4_dual(plan, m.heads, p.q, p.kv, p.indices, p.extra_kv, p.extra_indices,
                                  p.attn_sink, p.output, p.out_lse, stream, cold);
      if (mt == ModelType::DSV4) return dispatch_dsv4_single(plan, SINGLE_ARGS, cold);
      V32(dispatch_v32_mg);
    }
    default:
      return false;
  }
#undef V32
#undef SINGLE_ARGS
}

}  // namespace flashinfer::sparse_mla_sm120
