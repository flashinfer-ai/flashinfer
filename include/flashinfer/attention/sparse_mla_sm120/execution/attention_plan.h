// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../model/model_type.h"
#include "prefill_result.h"

namespace flashinfer::sparse_mla_sm120::execution {

constexpr int DecodeMaxHeads = 128;
constexpr int FixedPageSize = 64;
constexpr bool runtime_page(ModelType model) { return model == ModelType::DSV4_1; }

// DSV4 main-cache page sizes with compiled instantiations: 64 (default) and
// 32 (vLLM's DeepSeek page). The page size feeds only index arithmetic
// (page stride, footer offset), never tile shapes, so both are the same
// kernel template with full constant folding.
#define SPARSE_MLA_DSV4_MAIN_PAGES(F) F(64) F(32)

constexpr bool main_page_supported(ModelType model, int page) {
  if (runtime_page(model)) return page > 0;
  if (model == ModelType::DSV4) return page == FixedPageSize || page == 32;
  return page == FixedPageSize;
}

template <typename F>
PrefillLaunchResult visit_dsv4_main_page(int page, F&& call) {
#define PAGE(P) \
  if (page == P) return call(std::integral_constant<int, P>{});
  SPARSE_MLA_DSV4_MAIN_PAGES(PAGE)
#undef PAGE
  return false;
}

template <ModelType MT, int Variant, typename F>
PrefillLaunchResult visit_prefill_heads(int heads, F&& call) {
  constexpr bool inline_scale =
      MT == ModelType::DSV3_2 || MT == ModelType::GLM_NSA || MT == ModelType::GLM53_NOPE;
  constexpr bool sg = Variant == 1;
  constexpr bool mg = Variant == 2 || Variant == 3;
  constexpr bool swap = Variant == 4;
#define HEAD(H) \
  if (heads == H) return call(std::integral_constant<int, H>{});
  if constexpr ((sg && (inline_scale || MT == ModelType::DSV4_1 || MT == ModelType::DOTS3_SWA)) ||
                (mg && MT == ModelType::DSV4)) {
    HEAD(8) HEAD(16)
  }
  if constexpr ((sg && (MT == ModelType::DSV4_1 || MT == ModelType::DOTS3_SWA)) ||
                (mg && (inline_scale || MT == ModelType::DSV4))) {
    HEAD(32) HEAD(64)
  }
  if constexpr (swap && inline_scale) {
    HEAD(64) HEAD(128)
  }
  if constexpr (mg && (inline_scale || MT == ModelType::DSV4)) {
    HEAD(128)
  }
#undef HEAD
  return false;
}

inline bool has_prefill_heads(ModelType model, int variant, int heads) {
  auto present = [](auto) { return true; };
#define MODEL(M)                                                                             \
  case ModelType::M:                                                                         \
    if (variant == 1) return visit_prefill_heads<ModelType::M, 1>(heads, present).supported; \
    if (variant == 4) return visit_prefill_heads<ModelType::M, 4>(heads, present).supported; \
    return visit_prefill_heads<ModelType::M, 2>(heads, present).supported
  switch (model) {
    MODEL(DSV3_2);
    MODEL(DSV4);
    MODEL(GLM_NSA);
    MODEL(GLM53_NOPE);
    MODEL(DOTS3_SWA);
    MODEL(DSV4_1);
  }
#undef MODEL
  return false;
}

#define SPARSE_MLA_EXTRA_PAGES(F) F(64) F(2)

template <typename F>
PrefillLaunchResult visit_extra_page(int page, F&& call) {
#define PAGE(P) \
  if (page == P) return call(std::integral_constant<int, P>{});
  SPARSE_MLA_EXTRA_PAGES(PAGE)
#undef PAGE
  return false;
}

template <ModelType MT, typename F>
auto visit_decode_heads(int heads, F&& call) {
#define HEAD(H) \
  if (heads == H) return call(std::integral_constant<int, H>{});
  HEAD(8);
  HEAD(16);
  HEAD(32);
  HEAD(64);
  if constexpr (MT != ModelType::GLM53_NOPE && MT != ModelType::DOTS3_SWA) {
    HEAD(128);
  }
#undef HEAD
  return call(std::integral_constant<int, 0>{});
}

constexpr int decode_scratch_heads(int heads) { return heads == 8 ? 8 : (heads + 15) / 16 * 16; }

enum class NumericRoute : int64_t { FP8, QkBF16PvFP8, FullBF16, NVFP4 };
enum class Implementation : int64_t {
  Ordinary,
  MixedCache,
  FullBF16,
  Dsv4Nvfp4Decode,
  Dsv4Nvfp4GroupedDecode,
  SG,
  MG,
  FullTile,
  SwapAB,
  Dsv4Nvfp4Prefill
};
enum class Merge : int64_t { Direct, Merge2, General, Stage1 };

struct AttentionMetadata {
  int model;
  int tokens;
  int heads;
  int topk;
  int extra_topk;
  int page_size;
  int extra_page_size;
  size_t page_stride_bytes;
  size_t extra_page_stride_bytes;
  int row_stride_bytes;
  size_t indices_stride;
  size_t extra_indices_stride;
  size_t lse_stride;
  bool has_lengths;
  bool has_extra_lengths;
  bool has_sink;
  bool extra_fp4;
  int variant;
};

struct DeviceCaps {
  int sm_count;
  size_t max_shared_bytes;
};

int resolve_wave_cpb(int tokens, int head_blocks, int chunks, int requested_cpb, int sm_count);

struct ExecutionPlan {
  NumericRoute numeric;
  Implementation implementation;
  int cpb;
  int chunk_capacity;
  int active_splits;
  int scratch_split_stride;
  int scratch_heads;
  Merge merge;
  int head_blocks;
  int block_threads;
  size_t shared_bytes;
  size_t partial_bytes;
  size_t lse_bytes;
  int alignment;
  int specialized_heads;
  int specialized_topk;
  AttentionMetadata metadata{};
};

// QkBF16PvFP8 as a request permits that route; the resolver may still select FP8.
ExecutionPlan resolve_attention(const AttentionMetadata& metadata, NumericRoute requested,
                                int requested_cpb, DeviceCaps caps);

ExecutionPlan resolve_dsv4_nvfp4(int tokens, int heads, int topk, int extra_topk, int page_size,
                                 int extra_page_size, size_t page_stride_bytes,
                                 size_t extra_page_stride_bytes, int cpb, int sm_count,
                                 size_t max_shared_bytes, bool prefill, bool stage1_only);

}  // namespace flashinfer::sparse_mla_sm120::execution
