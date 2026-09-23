// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include <tvm/ffi/container/array.h>

#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv4_nvfp4/resources.cuh>

#include "../tvm_ffi_utils.h"
#include "attention_descriptor.h"
#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120::execution {

ExecutionPlan resolve_dsv4_nvfp4(int tokens, int heads, int topk, int extra_topk, int page_size,
                                 int extra_page_size, size_t page_stride_bytes,
                                 size_t extra_page_stride_bytes, int cpb, int sm_count,
                                 size_t max_shared_bytes, bool prefill, bool stage1_only) {
  using namespace nvfp4;
  TVM_FFI_ICHECK(tokens > 0 && has_instance(heads, topk)) << "unsupported NVFP4 tokens/heads/topk";
  TVM_FFI_ICHECK(page_size == FixedPageSize &&
                 page_stride_bytes >= size_t(page_size) * Dsv4Nvfp4Layout::BYTES_PER_TOKEN &&
                 page_stride_bytes % 16 == 0)
      << "unsupported NVFP4 page layout";
  TVM_FFI_ICHECK(
      extra_topk >= 0 &&
      (extra_topk == 0 ||
       (visit_extra_page(extra_page_size, [](auto) { return true; }).supported &&
        extra_page_stride_bytes >= size_t(extra_page_size) * Dsv4Nvfp4Layout::BYTES_PER_TOKEN &&
        extra_page_stride_bytes % 16 == 0)))
      << "unsupported NVFP4 extra page layout";
  TVM_FFI_ICHECK_LE(extra_topk, MaxExtraTopK) << "NVFP4 extra_topk exceeds integer chunk range";
  TVM_FFI_ICHECK_GT(sm_count, 0) << "SM count must be positive";
  TVM_FFI_ICHECK(!prefill || !stage1_only) << "stage1_only is decode-only";
  const int chunks = (topk + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW +
                     (extra_topk + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW;
  const int grouped_blocks = (heads + STREAMING_HEADS_PER_CTA - 1) / STREAMING_HEADS_PER_CTA;
  const bool grouped = prefill || (heads >= STREAMING_HEADS_PER_CTA && chunks >= 8 &&
                                   int64_t{tokens} * grouped_blocks >= 8);
  const int h_blocks = grouped ? grouped_blocks : (heads + HPB - 1) / HPB;
  if (prefill) cpb = chunks;
  if (cpb < 1 || cpb > chunks) {
    const int64_t per_token_head = int64_t{tokens} * h_blocks;
    const int64_t target_waves = (per_token_head + sm_count - 1) / sm_count;
    cpb = 1;
    float best_gap = static_cast<float>(target_waves) + 1.f;
    for (int candidate = 1; candidate <= chunks; ++candidate) {
      const int splits = (chunks + candidate - 1) / candidate;
      const int64_t active = per_token_head * splits;
      const int64_t waves = (active + sm_count - 1) / sm_count;
      if (waves != target_waves) continue;
      const float gap = static_cast<float>(waves) - static_cast<float>(active) / sm_count;
      if (gap < best_gap - 1e-6f || (gap < best_gap + 1e-6f && candidate > cpb)) {
        best_gap = gap;
        cpb = candidate;
      }
    }
  }
  const int active = (chunks + cpb - 1) / cpb;
  const Merge merge = stage1_only   ? Merge::Stage1
                      : active == 1 ? Merge::Direct
                      : active == 2 ? Merge::Merge2
                                    : Merge::General;
  const size_t shared = grouped ? StreamingNVFP4Smem::SIZE : DecodeNVFP4Smem<ModelType::DSV4>::SIZE;
  TVM_FFI_ICHECK_LE(shared, max_shared_bytes) << "NVFP4 shared memory exceeds device capacity";
  const size_t slots = merge == Merge::Direct ? 0 : size_t(tokens) * heads * active;
  return {
      NumericRoute::NVFP4,
      prefill   ? Implementation::Dsv4Nvfp4Prefill
      : grouped ? Implementation::Dsv4Nvfp4GroupedDecode
                : Implementation::Dsv4Nvfp4Decode,
      cpb,
      chunks,
      active,
      active,
      heads,
      merge,
      h_blocks,
      grouped ? STREAMING_BLOCK_THREADS : DECODE_BLOCK_THREADS,
      shared,
      slots * Dsv4Nvfp4Layout::D_V * sizeof(bf16),
      slots * sizeof(float),
      16,
      heads,
      topk,
      {int(ModelType::DSV4), tokens, heads, topk, extra_topk, page_size, extra_page_size,
       page_stride_bytes, extra_page_stride_bytes, Dsv4Nvfp4Layout::BYTES_PER_TOKEN, size_t(topk),
       size_t(extra_topk), size_t(heads), false, false, false, false, int(prefill)}};
}

ffi::Module resolve_dsv4_nvfp4_descriptor(ffi::Array<int64_t> values, int64_t numeric, int64_t cpb,
                                          int64_t sm_count, int64_t max_shared_bytes,
                                          bool stage1_only) {
  const auto m = unpack_metadata(values, 0);
  TVM_FFI_ICHECK(numeric == int64_t(NumericRoute::NVFP4) && cpb >= 0 && cpb <= INT_MAX &&
                 sm_count > 0 && sm_count <= INT_MAX && max_shared_bytes > 0)
      << "invalid DSV4 NVFP4 resolver capabilities/route";
  TVM_FFI_ICHECK(m.model == int(ModelType::DSV4) &&
                 m.row_stride_bytes == Dsv4Nvfp4Layout::BYTES_PER_TOKEN && !m.extra_fp4 &&
                 (m.variant == 0 || m.variant == 1) && (!m.has_extra_lengths || m.extra_topk > 0) &&
                 m.indices_stride == size_t(m.topk) &&
                 m.extra_indices_stride == size_t(m.extra_topk) && m.lse_stride == size_t(m.heads))
      << "DSV4 NVFP4 attention metadata mismatch";
  auto p = resolve_dsv4_nvfp4(m.tokens, m.heads, m.topk, m.extra_topk, m.page_size,
                              m.extra_page_size, m.page_stride_bytes, m.extra_page_stride_bytes,
                              cpb, sm_count, max_shared_bytes, m.variant == 1, stage1_only);
  p.metadata = m;
  return pack_plan(p);
}

ffi::Module resolve_dsv4_nvfp4_query(int64_t tokens, int64_t heads, int64_t topk,
                                     int64_t extra_topk, int64_t page_size, int64_t extra_page_size,
                                     int64_t page_stride, int64_t extra_page_stride, int64_t cpb,
                                     int64_t sm_count, int64_t shared, bool prefill, bool stage1,
                                     bool lengths, bool extra_lengths, bool sink) {
  return resolve_dsv4_nvfp4_descriptor(
      {int(ModelType::DSV4), tokens, heads, topk, extra_topk, page_size, extra_page_size,
       page_stride, extra_page_stride, Dsv4Nvfp4Layout::BYTES_PER_TOKEN, topk, extra_topk, heads,
       lengths, extra_lengths, sink, false, prefill},
      int(NumericRoute::NVFP4), cpb, sm_count, shared, stage1);
}

bool supports_attention(int64_t heads, int64_t topk, int64_t page, int64_t extra_topk,
                        int64_t extra_page) {
  if (heads < 1 || heads > INT_MAX || topk < 1 || topk > INT_MAX || page < 1 || page > INT_MAX ||
      extra_topk < 0 || extra_topk > nvfp4::MaxExtraTopK || extra_page < 0 ||
      extra_page > INT_MAX || (!extra_topk && extra_page))
    return false;
  try {
    resolve_dsv4_nvfp4(1, heads, topk, extra_topk, page, extra_page,
                       size_t(page) * Dsv4Nvfp4Layout::BYTES_PER_TOKEN,
                       size_t(extra_page) * Dsv4Nvfp4Layout::BYTES_PER_TOKEN, 1, 1, SIZE_MAX, false,
                       false);
    return true;
  } catch (const ffi::Error&) {
    return false;
  }
}

ffi::Map<ffi::String, ffi::Any> dsv4_nvfp4_format_info() {
  ffi::Array<int64_t> heads, topks, pages;
#define PAGE(P) pages.push_back(P);
  SPARSE_MLA_EXTRA_PAGES(PAGE)
#undef PAGE
#define INSTANCE(H, K) \
  heads.push_back(H);  \
  topks.push_back(K);
  SPARSE_MLA_DSV4_NVFP4_INSTANCES(INSTANCE)
#undef INSTANCE
  return {{"query_dim", Dsv4Nvfp4Layout::D_QK},
          {"value_dim", Dsv4Nvfp4Layout::D_V},
          {"bytes_per_token", Dsv4Nvfp4Layout::BYTES_PER_TOKEN},
          {"chunk_width", nvfp4::DECODE_CAND_WINDOW},
          {"page_size", FixedPageSize},
          {"extra_page_sizes", pages},
          {"heads", heads},
          {"topks", topks}};
}

}  // namespace flashinfer::sparse_mla_sm120::execution

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv4_nvfp4_format_info,
                              flashinfer::sparse_mla_sm120::execution::dsv4_nvfp4_format_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(supports_attention,
                              flashinfer::sparse_mla_sm120::execution::supports_attention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(resolve_dsv4_nvfp4,
                              flashinfer::sparse_mla_sm120::execution::resolve_dsv4_nvfp4_query);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    dsv4_nvfp4_resolve_attention,
    flashinfer::sparse_mla_sm120::execution::resolve_dsv4_nvfp4_descriptor);
