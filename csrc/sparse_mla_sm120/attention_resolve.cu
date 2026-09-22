// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include <tvm/ffi/container/array.h>

#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv32_fp8/resources.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_bf16/resources.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_fp8/resources.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_decode/resources.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/resources.cuh>

#include "../tvm_ffi_utils.h"
#include "attention_descriptor.h"

namespace flashinfer::sparse_mla_sm120::execution {

using kernels::dsv41_bf16::Dsv41Bf16Resources;
using kernels::dsv41_bf16::Dsv41Bf16Smem;

int resolve_wave_cpb(int tokens, int head_blocks, int chunks, int requested_cpb, int sm_count) {
  if (requested_cpb >= 1 && requested_cpb <= chunks) return requested_cpb;
  TVM_FFI_ICHECK_GT(sm_count, 0) << "SM count must be positive";
  int cpb = 1;
  float best_gap = 2.0f;
  const int64_t per_token_head = int64_t{tokens} * head_blocks;
  for (int candidate = 1; candidate <= chunks; ++candidate) {
    const int eff = (chunks + candidate - 1) / candidate;
    const int64_t active = per_token_head * eff;
    const int64_t ceil_w = (active + sm_count - 1) / sm_count;
    if (ceil_w > 3) continue;
    const float waves = (float)active / (float)sm_count;
    const float gap = (float)ceil_w - waves;
    if (gap < best_gap - 1e-6f || (gap < best_gap + 1e-6f && candidate > cpb)) {
      best_gap = gap;
      cpb = candidate;
    }
  }
  return cpb;
}

template <ModelType MT>
void decode_resources(ExecutionPlan& p) {
  p.specialized_heads =
      visit_decode_heads<MT>(p.metadata.heads, [](auto head) { return int(head); });
  using KV = KVCacheTraits<MT>;
  if constexpr (MT == ModelType::DSV3_2 || MT == ModelType::GLM_NSA ||
                MT == ModelType::GLM53_NOPE) {
    p.block_threads = DSV32_BLOCK_THREADS;
    p.shared_bytes = Dsv32DecodeSmem<MT>::LAUNCH_BYTES;
  } else {
    p.block_threads = DecodeTileCfg<MT>::BASE_BLOCK_THREADS;
    p.shared_bytes = Fp8DecodeSharedLayout<MT>::TOTAL_BYTES;
    if constexpr (MT == ModelType::DSV4_1) {
      if (p.implementation == Implementation::MixedCache) {
        p.block_threads = kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources::BLOCK_THREADS;
        p.shared_bytes = Fp8DecodeSharedLayout<MT, true>::TOTAL_BYTES;
      } else if (p.implementation == Implementation::FullBF16) {
        p.block_threads = Dsv41Bf16Resources::BLOCK_THREADS;
        p.shared_bytes = sizeof(Dsv41Bf16Smem);
      }
    }
  }
}

template <ModelType MT>
void prefill_resources(ExecutionPlan& p) {
  if constexpr (MT == ModelType::GLM53_NOPE) {
    if (p.numeric == NumericRoute::QkBF16PvFP8) {
      using Resources = Fp8PrefillResources<MT, QkComputeMode::BF16, void>;
      p.block_threads = Resources::BLOCK_THREADS;
      p.shared_bytes = Resources::SHARED_BYTES;
      return;
    }
  }
  if constexpr (MT == ModelType::DSV4) {
    p.block_threads = BLOCK_THREADS;
    p.shared_bytes = p.numeric == NumericRoute::QkBF16PvFP8
                         ? SmemLayoutMG<MT, QkComputeMode::BF16>::TOTAL
                         : SmemLayoutMG<MT, QkComputeMode::FP8>::TOTAL;
  } else {
    using Resources = Fp8PrefillResources<MT, QkComputeMode::FP8, void>;
    p.block_threads = Resources::BLOCK_THREADS;
    p.shared_bytes = Resources::SHARED_BYTES;
    if constexpr (MT == ModelType::DSV4_1) {
      if (p.metadata.extra_fp4) {
        using MixedResources =
            Fp8PrefillResources<MT, QkComputeMode::FP8,
                                kernels::dsv41_fp8::Dsv41MixedCachePrefillResources>;
        p.block_threads = MixedResources::BLOCK_THREADS;
        p.shared_bytes = MixedResources::SHARED_BYTES;
      }
    } else if constexpr (MT != ModelType::DOTS3_SWA) {
      if (p.implementation == Implementation::SwapAB) {
        p.shared_bytes = SmemLayoutSwapAB<MT>::TOTAL;
        p.block_threads = BLOCK_THREADS;
      } else if (p.metadata.variant == 2) {
        p.shared_bytes = SmemLayoutMG<MT, QkComputeMode::FP8>::TOTAL;
        p.block_threads = BLOCK_THREADS;
      }
    }
  }
}

ExecutionPlan resolve_attention(const AttentionMetadata& m, NumericRoute requested,
                                int requested_cpb, DeviceCaps caps) {
  TVM_FFI_ICHECK(m.model >= 0 && m.model <= 5 && m.tokens > 0 && m.heads > 0 &&
                 m.heads <= DecodeMaxHeads && m.topk > 0)
      << "unsupported sparse-MLA attention metadata";
  TVM_FFI_ICHECK(m.extra_topk >= 0 && m.topk <= INT_MAX - 64 - m.extra_topk &&
                 m.tokens <= INT_MAX / 8 && (!m.has_extra_lengths || m.extra_topk > 0))
      << "invalid attention capacity/length metadata";
  TVM_FFI_ICHECK_GT(caps.sm_count, 0) << "SM count must be positive";
  const auto mt = static_cast<ModelType>(m.model);
  const auto format = cache_format_info(mt);
  const bool inline_scale = format.inline_scale;
  const bool dual = m.extra_topk > 0;
  const int bpt = bytes_per_token(mt);
  TVM_FFI_ICHECK(main_page_supported(mt, m.page_size)) << "unsupported sparse-MLA main page size";
  TVM_FFI_ICHECK(m.row_stride_bytes >= bpt &&
                 (inline_scale ? m.row_stride_bytes % 16 == 0 : m.row_stride_bytes == bpt))
      << "invalid row stride";
  TVM_FFI_ICHECK(m.page_stride_bytes >= size_t(m.page_size) * m.row_stride_bytes &&
                 m.page_stride_bytes % 16 == 0)
      << "invalid main page stride";
  TVM_FFI_ICHECK(m.indices_stride >= size_t(m.topk) && m.lse_stride >= size_t(m.heads))
      << "invalid indices/LSE stride";
  TVM_FFI_ICHECK(!m.extra_fp4 || (mt == ModelType::DSV4_1 && dual))
      << "extra FP4 requires DSV41 dual cache";
  TVM_FFI_ICHECK(!dual || mt == ModelType::DSV4 || mt == ModelType::DSV4_1)
      << "sparse-MLA has no dual-cache form";
  TVM_FFI_ICHECK(!dual || (main_page_supported(mt, m.extra_page_size) &&
                           m.extra_indices_stride >= size_t(m.extra_topk) &&
                           m.extra_page_stride_bytes >=
                               size_t(m.extra_page_size) *
                                   (m.extra_fp4 ? Dsv41Fp4Layout::BYTES_PER_TOKEN : bpt) &&
                           m.extra_page_stride_bytes % 16 == 0))
      << "invalid extra page layout";
  TVM_FFI_ICHECK(mt != ModelType::DOTS3_SWA ||
                 m.topk >= DecodeTileCfg<ModelType::DOTS3_SWA>::WINDOW)
      << "sparse-MLA DOTS3 requires topk >= 513";
  TVM_FFI_ICHECK(requested != NumericRoute::NVFP4) << "wrong module for NVFP4";
  const bool require_bf16_qk = requested == NumericRoute::RequireQkBF16PvFP8;
  TVM_FFI_ICHECK(!require_bf16_qk ||
                 (mt == ModelType::GLM53_NOPE && m.heads == 16 && m.page_size == 64 &&
                  (m.topk == 2112 || m.topk == 2176) && !dual && m.variant == 1))
      << "bf16_qk requires GLM53 NoPE, H=16, PBS=64, topk=2112/2176, single-cache SG";
  TVM_FFI_ICHECK(
      requested != NumericRoute::FullBF16 ||
      (mt == ModelType::DSV4_1 && ((m.variant == 0 && m.tokens <= 64) || m.variant == 1)))
      << "full BF16 requires DSV41 decode or SG prefill";
  ExecutionPlan p{};
  p.metadata = m;
  p.alignment = 16;
  p.specialized_topk = 0;
  p.numeric = require_bf16_qk                       ? NumericRoute::QkBF16PvFP8
              : requested == NumericRoute::FullBF16 ? requested
                                                    : NumericRoute::FP8;
  if (m.variant == 0) {
    const int tile = mt == ModelType::DOTS3_SWA ? DecodeTileCfg<ModelType::DOTS3_SWA>::BI
                                                : DecodeTileCfg<ModelType::DSV4>::BI;
    p.chunk_capacity = (m.topk + tile - 1) / tile + (m.extra_topk + tile - 1) / tile;
    p.head_blocks = (m.heads + 15) / 16;
    p.cpb =
        resolve_wave_cpb(m.tokens, p.head_blocks, p.chunk_capacity, requested_cpb, caps.sm_count);
    p.active_splits = (p.chunk_capacity + p.cpb - 1) / p.cpb;
    p.scratch_split_stride = p.chunk_capacity;
    p.scratch_heads = decode_scratch_heads(m.heads);
    p.merge = Merge::General;
    p.implementation = requested == NumericRoute::FullBF16 ? Implementation::FullBF16
                       : m.extra_fp4                       ? Implementation::MixedCache
                                                           : Implementation::Ordinary;
    const size_t slots = size_t(m.tokens) * p.scratch_heads * p.chunk_capacity;
    p.partial_bytes = slots * format.value_dim * sizeof(__nv_bfloat16);
    p.lse_bytes = slots * sizeof(float);
#define RES(M)                         \
  case ModelType::M:                   \
    decode_resources<ModelType::M>(p); \
    break
    switch (mt) {
      RES(DSV3_2);
      RES(GLM_NSA);
      RES(GLM53_NOPE);
      RES(DSV4);
      RES(DOTS3_SWA);
      RES(DSV4_1);
    }
#undef RES
  } else if (requested == NumericRoute::FullBF16) {
    p.implementation = Implementation::FullBF16;
    p.head_blocks = (m.heads + 15) / 16;
    p.cpb = 1;
    p.chunk_capacity = (m.topk + 63) / 64 + (m.extra_topk + 63) / 64;
    p.active_splits = 1;
    p.merge = Merge::Direct;
    p.specialized_heads =
        visit_decode_heads<ModelType::DSV4_1>(m.heads, [](auto h) { return int(h); });
    p.block_threads = Dsv41Bf16Resources::BLOCK_THREADS;
    p.shared_bytes = sizeof(Dsv41Bf16Smem);
  } else {
    TVM_FFI_ICHECK(m.variant >= 1 && m.variant <= 4 && m.topk % 64 == 0 &&
                   m.indices_stride == size_t(m.topk))
        << "invalid sparse-MLA prefill variant/topk/indices";
    TVM_FFI_ICHECK(!dual || m.extra_indices_stride == size_t(m.extra_topk))
        << "prefill extra indices must be dense";
    TVM_FFI_ICHECK(!inline_scale || m.page_stride_bytes == size_t(m.page_size) * m.row_stride_bytes)
        << "inline prefill requires contiguous pages";
    TVM_FFI_ICHECK(has_prefill_heads(mt, m.variant, m.heads))
        << "unsupported sparse-MLA prefill heads";
    if (m.variant == 1) {
      TVM_FFI_ICHECK(!dual || mt == ModelType::DSV4_1) << "unsupported SG dual cache";
      p.implementation = m.extra_fp4 ? Implementation::MixedCache : Implementation::SG;
      p.head_blocks = (m.heads + 15) / 16;
    } else if (m.variant == 4) {
      TVM_FFI_ICHECK(!dual) << "unsupported sparse-MLA swapAB metadata";
      p.implementation = Implementation::SwapAB;
      p.head_blocks = m.heads / 64;
    } else {
      TVM_FFI_ICHECK(m.variant == 3 ? (mt == ModelType::DSV4 && dual) : !dual)
          << "sparse-MLA MG dual metadata mismatch";
      p.implementation = dual && !m.has_lengths && !m.has_extra_lengths && m.extra_topk % 64 == 0
                             ? Implementation::FullTile
                             : Implementation::MG;
      if (mt == ModelType::DSV4 && (dual || m.topk <= 256)) p.numeric = NumericRoute::QkBF16PvFP8;
      TVM_FFI_ICHECK(requested != NumericRoute::FP8 || p.numeric == NumericRoute::FP8)
          << "explicit FP8 forbids hybrid numerical route";
      p.head_blocks = m.heads <= 16 ? 1 : m.heads / 32;
    }
    p.cpb = 1;
    p.chunk_capacity = (m.topk + 63) / 64 + (m.extra_topk + 63) / 64;
    p.active_splits = 1;
    p.merge = Merge::Direct;
    p.specialized_heads = m.heads;
#define RES(M)                          \
  case ModelType::M:                    \
    prefill_resources<ModelType::M>(p); \
    break
    switch (mt) {
      RES(DSV3_2);
      RES(GLM_NSA);
      RES(GLM53_NOPE);
      RES(DSV4);
      RES(DOTS3_SWA);
      RES(DSV4_1);
    }
#undef RES
  }
  TVM_FFI_ICHECK_LE(p.shared_bytes, caps.max_shared_bytes)
      << "shared memory exceeds device capacity";
  return p;
}

ffi::Module resolve_descriptor(ffi::Array<int64_t> values, int64_t numeric, int64_t cpb,
                               int64_t sm_count, int64_t max_shared) {
  const auto m = unpack_metadata(values, 0);
  TVM_FFI_ICHECK(
      ((numeric >= 0 && numeric <= 2) || numeric == int64_t(NumericRoute::RequireQkBF16PvFP8)) &&
      cpb >= 0 && cpb <= INT_MAX && sm_count > 0 && sm_count <= INT_MAX && max_shared > 0)
      << "invalid resolver capabilities/route";
  return pack_plan(resolve_attention(m, static_cast<NumericRoute>(numeric), cpb,
                                     {int(sm_count), size_t(max_shared)}));
}
ffi::Array<int64_t> candidates(int64_t model, int64_t heads, int64_t topk, int64_t page, bool dual,
                               int64_t extra_page, ffi::String precision) {
  ffi::Array<int64_t> result;
  if (model < 0 || model > int(ModelType::DSV4_1) || heads < 1 || heads > DecodeMaxHeads ||
      topk < 1 || topk > INT_MAX - 128 || page < 1 || page > INT_MAX || extra_page < 0 ||
      extra_page > INT_MAX)
    return result;
  if (precision == "bf16_qk") {
    if (model != int(ModelType::GLM53_NOPE)) return result;
  } else if (precision != "default" && model != int(ModelType::DSV4_1)) {
    return result;
  }
  const auto format = cache_format_info(static_cast<ModelType>(model));
  const auto numeric = precision == "bf16_qk" ? NumericRoute::RequireQkBF16PvFP8
                       : precision == "bf16"  ? NumericRoute::FullBF16
                       : precision == "fp8"   ? NumericRoute::FP8
                                              : NumericRoute::QkBF16PvFP8;
  AttentionMetadata m{int(model),
                      1,
                      int(heads),
                      int(topk),
                      dual ? 64 : 0,
                      int(page),
                      dual ? int(extra_page) : 0,
                      (size_t(page) * format.bytes_per_token + 15) / 16 * 16,
                      dual ? (size_t(extra_page) * format.bytes_per_token + 15) / 16 * 16 : 0,
                      format.bytes_per_token,
                      size_t(topk),
                      dual ? 64u : 0u,
                      size_t(heads),
                      false,
                      false,
                      false,
                      false,
                      0};
  for (int variant = 0; variant <= int(PrefillVariant::SWAPAB); ++variant) {
    m.variant = variant;
    try {
      resolve_attention(m, numeric, 1, {1, SIZE_MAX});
      result.push_back(variant);
    } catch (const ffi::Error&) {
      // Unsupported candidates are the same failures reported by explicit
      // resolution.
    }
  }
  return result;
}

ffi::Map<int64_t, int64_t> metadata_candidates(ffi::Array<int64_t> values, int64_t numeric,
                                               int64_t sm_count, int64_t max_shared) {
  TVM_FFI_ICHECK(
      ((numeric >= 0 && numeric <= 2) || numeric == int64_t(NumericRoute::RequireQkBF16PvFP8)) &&
      sm_count > 0 && sm_count <= INT_MAX && max_shared > 0)
      << "invalid candidate capabilities/route";
  auto m = unpack_metadata(values, 0);
  ffi::Map<int64_t, int64_t> result;
  for (int variant = 0; variant <= int(PrefillVariant::SWAPAB); ++variant) {
    if (variant == 0 && m.tokens > 64) continue;
    m.variant = variant;
    try {
      const auto p = resolve_attention(m, static_cast<NumericRoute>(numeric), 1,
                                       {int(sm_count), size_t(max_shared)});
      result.Set(variant, p.chunk_capacity);
    } catch (const ffi::Error&) {
    }
  }
  return result;
}

ffi::Map<ffi::String, int64_t> format_info(int64_t model) {
  TVM_FFI_ICHECK(model >= 0 && model <= int(ModelType::DSV4_1));
  const auto f = cache_format_info(static_cast<ModelType>(model));
  return {
      {"query_dim", f.query_dim},
      {"value_dim", f.value_dim},
      {"bytes_per_token", f.bytes_per_token},
      {"inline_scale", f.inline_scale},
      {"nope_dim", f.nope_dim},
      {"rope_dim", f.rope_dim},
      {"num_scales", f.num_scales},
      {"scale_bytes", f.scale_bytes},
      {"data_bytes", f.data_bytes},
      {"rope_offset", f.rope_offset},
      {"fp4_bytes_per_token", Dsv41Fp4Layout::BYTES_PER_TOKEN},
      {"min_topk",
       model == int(ModelType::DOTS3_SWA) ? DecodeTileCfg<ModelType::DOTS3_SWA>::WINDOW : 1},
      {"runtime_page", runtime_page(static_cast<ModelType>(model))},
      {"max_heads", DecodeMaxHeads},
      {"page_size", FixedPageSize},
      {"chunk_width", model == int(ModelType::DOTS3_SWA) ? DecodeTileCfg<ModelType::DOTS3_SWA>::BI
                                                         : DecodeTileCfg<ModelType::DSV4>::BI}};
}

ffi::Array<int64_t> decode_head_counts(int64_t model) {
  TVM_FFI_ICHECK(model >= 0 && model <= int(ModelType::DSV4_1));
  ffi::Array<int64_t> result;
  for (int heads = 1; heads <= DecodeMaxHeads; ++heads) {
    int specialized = 0;
#define MODEL(M)                                                                          \
  case ModelType::M:                                                                      \
    specialized = visit_decode_heads<ModelType::M>(heads, [](auto h) { return int(h); }); \
    break
    switch (static_cast<ModelType>(model)) {
      MODEL(DSV3_2);
      MODEL(DSV4);
      MODEL(GLM_NSA);
      MODEL(GLM53_NOPE);
      MODEL(DOTS3_SWA);
      MODEL(DSV4_1);
    }
#undef MODEL
    if (specialized != 0) result.push_back(specialized);
  }
  return result;
}

ffi::Array<int64_t> main_page_sizes(int64_t model) {
  TVM_FFI_ICHECK(model >= 0 && model <= int(ModelType::DSV4_1));
  ffi::Array<int64_t> result;
  const auto mt = static_cast<ModelType>(model);
  if (runtime_page(mt)) return result;  // any positive page size
  result.push_back(FixedPageSize);
  return result;
}

int64_t resolve_format(int64_t query_dim, ffi::String scale) {
  for (int model = 0; model <= int(ModelType::DSV4_1); ++model) {
    const auto mt = static_cast<ModelType>(model);
    const auto f = cache_format_info(mt);
    const bool selected = mt == ModelType::DSV3_2 ? (scale == "auto" || scale == "pow2_fp32")
                          : mt == ModelType::GLM_NSA || mt == ModelType::GLM53_NOPE
                              ? scale == "arbitrary_fp32"
                          : mt == ModelType::DSV4_1 ? scale == "ue8m0_g32"
                                                    : scale == "auto";
    if (f.query_dim == query_dim && selected) return model;
  }
  TVM_FFI_THROW(ValueError) << "unsupported d_qk=" << query_dim
                            << " with kv_scale_format=" << scale;
}

}  // namespace flashinfer::sparse_mla_sm120::execution

TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_scratch_heads,
                              flashinfer::sparse_mla_sm120::execution::decode_scratch_heads);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(resolve_format,
                              flashinfer::sparse_mla_sm120::execution::resolve_format);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(candidates, flashinfer::sparse_mla_sm120::execution::candidates);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(format_info, flashinfer::sparse_mla_sm120::execution::format_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_head_counts,
                              flashinfer::sparse_mla_sm120::execution::decode_head_counts);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(main_page_sizes,
                              flashinfer::sparse_mla_sm120::execution::main_page_sizes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(metadata_candidates,
                              flashinfer::sparse_mla_sm120::execution::metadata_candidates);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(resolve_attention,
                              flashinfer::sparse_mla_sm120::execution::resolve_descriptor);
