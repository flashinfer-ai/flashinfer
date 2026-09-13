// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// TVM-FFI binding for sparse-MLA SM120 paged attention.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>

#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include <flashinfer/attention/sparse_mla_sm120/model/dsv41_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include "attention_descriptor.h"
#include "attention_dispatch.h"
#include "attention_validation.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

using bf16 = __nv_bfloat16;

void SparseMlaSm120PagedAttention(TensorView q, TensorView kv_cache, TensorView indices,
                                  TensorView output, TensorView out_lse, double sm_scale,
                                  int64_t model_type, int64_t variant,
                                  Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                  Optional<TensorView> extra_kv_cache,
                                  Optional<TensorView> extra_indices,
                                  Optional<TensorView> extra_topk_length, bool extra_fp4);

// Thin TVM-FFI wrapper for the decode-dsv4 standalone path. The caller passes
// already-sized scratch tensors mid_out + mid_lse plus the output and lse.
// Handles DSV4 decode with page_block_size=64 and the supported head/top-k
// instantiation grid in sparse_mla_sm120_decode.cu.
template <bool BF16 = false>
void SparseMlaSm120DecodeDsv4(TensorView q, TensorView kv_cache, TensorView indices,
                              TensorView mid_out, TensorView mid_lse, TensorView output,
                              TensorView out_lse, int64_t num_splits, double sm_scale,
                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                              Optional<TensorView> extra_kv_cache,
                              Optional<TensorView> extra_indices,
                              Optional<TensorView> extra_topk_length, int64_t model_type,
                              int64_t chunks_per_block_override, bool extra_fp4) {
  check_decode_tensors(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, num_splits);
  check_attention_optional(q, extra_kv_cache, "extra_kv_cache");
  check_attention_optional(q, extra_indices, "extra_indices");
  check_attention_vector(q, extra_topk_length, q.size(0), dl_int32, "extra_topk_length");
  TVM_FFI_ICHECK(!extra_indices.has_value() || extra_kv_cache.has_value())
      << "extra_indices requires extra_kv_cache";
  TVM_FFI_ICHECK(!extra_topk_length.has_value() || extra_kv_cache.has_value())
      << "extra_topk_length requires extra_kv_cache";
  check_attention_vector(q, topk_length, q.size(0), dl_int32, "topk_length");
  check_attention_vector(q, attn_sink, q.size(1), dl_float32, "attn_sink");
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  // indices may be 2D [T, topk] or 3D [T, s_q=1, topk] (some callers keep
  // the s_q singleton dim through the call stack). Rows may be strided views
  // into a wider persistent buffer; the kernel walks each row with
  // stride(0). The last dim must stay contiguous.
  TVM_FFI_ICHECK_GE(indices.ndim(), 2)
      << "indices must have at least 2 dims; got ndim=" << indices.ndim();
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }
  CHECK_INPUT_TYPE(indices, dl_int32);
  TVM_FFI_ICHECK_EQ(indices.stride(-1), 1) << "indices last dimension must be contiguous";
  const size_t stride_indices_token = static_cast<size_t>(indices.stride(0));
  // out_lse rows may be strided views into a wider buffer, same slicing
  // pattern as indices; the head dim must stay contiguous.
  CHECK_CUDA(out_lse);
  CHECK_INPUT_TYPE(out_lse, dl_float32);
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2) << "out_lse must be [T, H]";
  TVM_FFI_ICHECK_EQ(out_lse.stride(-1), 1) << "out_lse last dimension must be contiguous";
  const size_t stride_out_lse = static_cast<size_t>(out_lse.stride(0));

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  TVM_FFI_ICHECK_EQ(indices.size(0), num_tokens)
      << "indices leading dimension must match num_tokens";
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  // This kernel serves the footer-scale model types. model_type is the
  // explicit selector from the Python planner; -1 keeps the legacy width
  // inference (512 -> DSV4, 1088 -> DOTS3_SWA). Width alone cannot separate
  // DSV4 from DSV4_1 (both are d_qk=512), so DSV4_1 is only reachable
  // explicitly.
  TVM_FFI_ICHECK(d_qk == 512 || d_qk == 1088)
      << "decode-dsv4 supports d_qk 512 (DSV4/DSV4_1) or 1088 (DOTS3_SWA); got " << d_qk;
  const ModelType mt = model_type == -1 ? ((d_qk == 512) ? ModelType::DSV4 : ModelType::DOTS3_SWA)
                                        : static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK((d_qk == 512 && (mt == ModelType::DSV4 || mt == ModelType::DSV4_1)) ||
                 (d_qk == 1088 && mt == ModelType::DOTS3_SWA))
      << "decode-dsv4 model_type mismatch: d_qk=" << d_qk << " model_type=" << model_type;
  // DOTS3_SWA's sliding window (513 candidates, DecodeTileCfg::WINDOW) needs an
  // indices buffer at least that wide; a narrower one can never name the full
  // window. Report it here so the message names the actual constraint.
  TVM_FFI_ICHECK(mt != ModelType::DOTS3_SWA || topk >= KVCacheTraits<ModelType::DOTS3_SWA>::WINDOW)
      << "decode-dsv4 (dots3_swa) requires topk >= 513 to hold the 513-wide "
         "sliding window; got indices width topk="
      << topk;
  TVM_FFI_ICHECK(mt != ModelType::DOTS3_SWA || !extra_kv_cache.has_value())
      << "decode-dsv4 (dots3_swa) has no dual-cache form; extra_kv_cache is "
         "DSV4/DSV4_1-only";
  // V41_FP4 is defined only as the extra (compressed) cache of a V4.1 FP8
  // main cache (FlashMLA tests/quant.py); the gather path upconverts it to
  // the canonical DSV4_1 FP8 smem row.
  TVM_FFI_ICHECK(!extra_fp4 || mt == ModelType::DSV4_1)
      << "decode-dsv4 extra_fp4 requires a DSV4_1 main cache; got model_type="
      << static_cast<int>(mt);
  TVM_FFI_ICHECK(!extra_fp4 || extra_kv_cache.has_value())
      << "decode-dsv4 extra_fp4 requires extra_kv_cache";

  // topk_length is optional for DOTS3_SWA: DecodeTileCfg<DOTS3_SWA>::WINDOW caps
  // the per-token candidate count inside the kernel, so omitting it costs
  // nothing beyond the window itself. Unused slots must still carry -1, which
  // the QK mask turns into -inf.
  const int bpt = bytes_per_token(mt);
  const PagedKVLayout kv_layout =
      parse_paged_kv_layout(kv_cache, bpt, /*inline_scale=*/false, "kv_cache");
  // Footer-scale kernels (DSV4, DOTS3_SWA) gather with a tightly packed row
  // advance; only the decode-v32 path honors stride_kv_row. Reject padded rows
  // loudly instead of reading the wrong bytes.
  TVM_FFI_ICHECK_EQ(kv_layout.stride_kv_row, bpt)
      << "decode-dsv4 (footer-scale layout) requires tightly packed KV rows "
      << "(stride_kv_row == bytes_per_token=" << bpt
      << "); padded-row KV caches are supported only on the decode-v32 path";
  const int page_block_size = kv_layout.page_block_size;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_indices_ptr =
      extra_indices.has_value() ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                                : nullptr;
  const int* extra_topk_len_ptr =
      extra_topk_length.has_value() ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                    : nullptr;
  // extra_topk and stride_extra_kv_block are derived from the optional tensors.
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  size_t stride_extra_indices_token = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    const auto& eidx = extra_indices.value();
    CHECK_INPUT_TYPE(eidx, dl_int32);
    TVM_FFI_ICHECK(eidx.ndim() == 2 || (eidx.ndim() == 3 && eidx.size(1) == 1))
        << "extra_indices must be [T, K] or [T, 1, K]";
    TVM_FFI_ICHECK_GE(eidx.stride(0), eidx.size(-1)) << "extra_indices rows must not overlap";
    TVM_FFI_ICHECK_GT(eidx.size(-1), 0) << "extra_indices width must be positive";
    TVM_FFI_ICHECK_EQ(eidx.size(0), num_tokens)
        << "extra_indices leading dimension must match num_tokens";
    TVM_FFI_ICHECK_EQ(eidx.stride(-1), 1) << "extra_indices last dimension must be contiguous";
    extra_topk_arg = static_cast<int>(eidx.size(-1));
    stride_extra_indices_token = static_cast<size_t>(eidx.stride(0));
    // The extra (dual) cache carries the same per-token layout as the main
    // one, unless extra_fp4 selects the V41_FP4 row format (288 B/token).
    const int bpt_extra = extra_fp4 ? Dsv41Fp4Layout::BYTES_PER_TOKEN : bpt;
    const PagedKVLayout extra_layout =
        parse_paged_kv_layout(ekv, bpt_extra, /*inline_scale=*/false, "extra_kv_cache");
    TVM_FFI_ICHECK_EQ(extra_layout.stride_kv_row, static_cast<size_t>(bpt_extra))
        << "decode-dsv4 extra_kv_cache requires tightly packed KV rows "
        << "(stride_kv_row == bytes_per_token=" << bpt_extra
        << "); padded-row KV caches are supported only on the decode-v32 path";
    TVM_FFI_ICHECK(!extra_fp4 || extra_layout.stride_kv_block % 16 == 0)
        << "decode-dsv4 extra_fp4 requires a 16B-aligned extra block stride; got "
        << extra_layout.stride_kv_block;
    pbs_extra_arg = extra_layout.page_block_size;
    stride_extra_kv_block = extra_layout.stride_kv_block;
  }

  check_decode_scratch_capacity(q, mid_out, mid_lse, num_splits);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t stream = get_stream(q.device());
  int sm_count = 0, max_shared = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           q.device().device_id),
                    cudaSuccess);
  const execution::AttentionMetadata metadata{int(mt),
                                              num_tokens,
                                              num_heads,
                                              topk,
                                              extra_topk_arg,
                                              page_block_size,
                                              pbs_extra_arg,
                                              kv_layout.stride_kv_block,
                                              stride_extra_kv_block,
                                              bpt,
                                              stride_indices_token,
                                              stride_extra_indices_token,
                                              stride_out_lse,
                                              topk_length.has_value(),
                                              extra_topk_length.has_value(),
                                              attn_sink.has_value(),
                                              extra_fp4,
                                              0};
  const auto plan = execution::resolve_attention(
      metadata, BF16 ? execution::NumericRoute::FullBF16 : execution::NumericRoute::FP8,
      chunks_per_block_override, {sm_count, size_t(max_shared)});
  const execution::AttentionParams params{num_heads,
                                       topk,
                                       static_cast<const bf16*>(q.data_ptr()),
                                       static_cast<const uint8_t*>(kv_cache.data_ptr()),
                                       static_cast<const int32_t*>(indices.data_ptr()),
                                       static_cast<bf16*>(mid_out.data_ptr()),
                                       static_cast<float*>(mid_lse.data_ptr()),
                                       topk_len_ptr,
                                       static_cast<bf16*>(output.data_ptr()),
                                       static_cast<float*>(out_lse.data_ptr()),
                                       attn_sink_ptr,
                                       extra_kv_ptr,
                                       extra_indices_ptr,
                                       extra_topk_len_ptr,
                                       extra_topk_arg,
                                       pbs_extra_arg,
                                       stride_extra_kv_block,
                                       num_tokens,
                                       int(num_splits),
                                       plan.cpb,
                                       float(sm_scale),
                                       kv_layout.stride_kv_block,
                                       stride_indices_token,
                                       stride_extra_indices_token,
                                       stride_out_lse,
                                       page_block_size,
                                       extra_fp4,
                                       BF16};
  const auto status = dispatch_decode(params, plan, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "decode-dsv4: " << cudaGetErrorString(status);
}

// Thin TVM-FFI wrapper for the decode-dsv3_2 standalone path (V32 family,
// no dual cache). Mirrors SparseMlaSm120DecodeDsv4: pre-allocated mid_out +
// mid_lse scratch, static (num_tokens × H_BLOCKS × num_splits) grid, V4-style
// warp-spec + per-buffer mbarrier pipeline.
void SparseMlaSm120DecodeDsv3_2(TensorView q, TensorView kv_cache, TensorView indices,
                                TensorView mid_out, TensorView mid_lse, TensorView output,
                                TensorView out_lse, int64_t num_splits, double sm_scale,
                                Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                int64_t model_type, int64_t chunks_per_block_override) {
  check_decode_tensors(q, kv_cache, indices, mid_out, mid_lse, output, out_lse, num_splits);
  check_attention_vector(q, topk_length, q.size(0), dl_int32, "topk_length");
  check_attention_vector(q, attn_sink, q.size(1), dl_float32, "attn_sink");
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  TVM_FFI_ICHECK_GE(indices.ndim(), 2);
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }
  CHECK_INPUT_TYPE(indices, dl_int32);
  TVM_FFI_ICHECK_EQ(indices.stride(-1), 1) << "indices last dimension must be contiguous";
  const size_t stride_indices_token = static_cast<size_t>(indices.stride(0));
  // out_lse rows may be strided views into a wider buffer, same slicing
  // pattern as indices; the head dim must stay contiguous.
  CHECK_CUDA(out_lse);
  CHECK_INPUT_TYPE(out_lse, dl_float32);
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2) << "out_lse must be [T, H]";
  TVM_FFI_ICHECK_EQ(out_lse.stride(-1), 1) << "out_lse last dimension must be contiguous";
  const size_t stride_out_lse = static_cast<size_t>(out_lse.stride(0));

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  TVM_FFI_ICHECK_EQ(indices.size(0), num_tokens)
      << "indices leading dimension must match num_tokens";
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK((d_qk == 576 && (mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)) ||
                 (d_qk == 512 && mt == ModelType::GLM53_NOPE))
      << "decode-v32 expects DSV3_2/GLM_NSA d_qk=576 or GLM53_NOPE d_qk=512; got d_qk=" << d_qk
      << " model_type=" << model_type;

  const PagedKVLayout kv_layout =
      parse_paged_kv_layout(kv_cache, bytes_per_token(mt), /*inline_scale=*/true, "kv_cache");
  TVM_FFI_ICHECK_EQ(kv_layout.page_block_size, 64) << "decode-v32 requires page_block_size=64";

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;

  check_decode_scratch_capacity(q, mid_out, mid_lse, num_splits);
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t stream = get_stream(q.device());
  int sm_count = 0, max_shared = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           q.device().device_id),
                    cudaSuccess);
  const execution::AttentionMetadata metadata{int(mt),
                                              num_tokens,
                                              num_heads,
                                              topk,
                                              0,
                                              64,
                                              0,
                                              kv_layout.stride_kv_block,
                                              0,
                                              kv_layout.stride_kv_row,
                                              stride_indices_token,
                                              0,
                                              stride_out_lse,
                                              topk_length.has_value(),
                                              false,
                                              attn_sink.has_value(),
                                              false,
                                              0};
  const auto resolved =
      execution::resolve_attention(metadata, execution::NumericRoute::FP8,
                                   chunks_per_block_override, {sm_count, size_t(max_shared)});
  execution::AttentionParams params{};
  params.num_heads = num_heads;
  params.topk = topk;
  params.num_tokens = num_tokens;
  params.allocated_splits = num_splits;
  params.chunks_per_block = resolved.cpb;
  params.q = static_cast<const bf16*>(q.data_ptr());
  params.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
  params.indices = static_cast<const int32_t*>(indices.data_ptr());
  params.mid_out = static_cast<bf16*>(mid_out.data_ptr());
  params.mid_lse = static_cast<float*>(mid_lse.data_ptr());
  params.output = static_cast<bf16*>(output.data_ptr());
  params.out_lse = static_cast<float*>(out_lse.data_ptr());
  params.topk_length = topk_len_ptr;
  params.attn_sink = attn_sink_ptr;
  params.sm_scale = sm_scale;
  params.page_stride_bytes = kv_layout.stride_kv_block;
  params.indices_stride_elems = stride_indices_token;
  params.out_lse_stride_elems = stride_out_lse;
  const auto status = dispatch_dsv32(params, resolved, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "decode-dsv3_2: " << cudaGetErrorString(status);
}

void ExecuteAttentionPlan(ffi::Module descriptor, TensorView q, TensorView cache,
                          TensorView indices, TensorView mid, TensorView mlse, TensorView output,
                          TensorView lse, double scale, Optional<TensorView> lengths,
                          Optional<TensorView> sink, Optional<TensorView> extra_cache,
                          Optional<TensorView> extra_indices, Optional<TensorView> extra_lengths) {
  const auto& plan = execution::unpack_plan(descriptor, false);
  const auto& m = plan.metadata;
  const auto format = cache_format_info(static_cast<ModelType>(m.model));
  const bool inline_scale = format.inline_scale;
  const size_t q_alignment = plan.numeric == execution::NumericRoute::QkBF16PvFP8            ? 2
                             : plan.implementation == execution::Implementation::SwapAB ? 8
                                                                                        : 16;
  check_decode_tensors(q, cache, indices, mid, mlse, output, lse, plan.chunk_capacity, q_alignment);
  TVM_FFI_ICHECK(q.size(0) == m.tokens && q.size(1) == m.heads && indices.size(-1) == m.topk &&
                 indices.stride(0) == int64_t(m.indices_stride) &&
                 lse.stride(0) == int64_t(m.lse_stride))
      << "execution plan tensor metadata mismatch";
  const auto mt = static_cast<ModelType>(m.model);
  TVM_FFI_ICHECK_EQ(q.size(2), format.query_dim) << "execution plan query width mismatch";
  const auto layout = parse_paged_kv_layout(cache, bytes_per_token(mt), inline_scale, "kv_cache");
  TVM_FFI_ICHECK(layout.page_block_size == m.page_size &&
                 layout.stride_kv_block == m.page_stride_bytes &&
                 layout.stride_kv_row == size_t(m.row_stride_bytes))
      << "execution plan cache metadata mismatch";
  TVM_FFI_ICHECK(m.variant == 0 || !inline_scale ||
                 m.page_stride_bytes == size_t(m.page_size) * m.row_stride_bytes)
      << "execution plan inline prefill stride mismatch";
  TVM_FFI_ICHECK(lengths.has_value() == m.has_lengths && sink.has_value() == m.has_sink &&
                 extra_lengths.has_value() == m.has_extra_lengths &&
                 extra_cache.has_value() == (m.extra_topk > 0) &&
                 extra_indices.has_value() == extra_cache.has_value())
      << "execution plan optional metadata mismatch";
  check_attention_vector(q, lengths, m.tokens, dl_int32, "topk_length");
  check_attention_vector(q, sink, m.heads, dl_float32, "attn_sink");
  check_attention_vector(q, extra_lengths, m.tokens, dl_int32, "extra_topk_length");
  execution::AttentionParams p{};
  p.num_heads = m.heads;
  p.topk = m.topk;
  p.num_tokens = m.tokens;
  p.allocated_splits = plan.chunk_capacity;
  p.chunks_per_block = plan.cpb;
  p.sm_scale = scale;
  p.q = static_cast<const bf16*>(q.data_ptr());
  p.kv = static_cast<const uint8_t*>(cache.data_ptr());
  p.indices = static_cast<const int32_t*>(indices.data_ptr());
  p.mid_out = static_cast<bf16*>(mid.data_ptr());
  p.mid_lse = static_cast<float*>(mlse.data_ptr());
  p.output = static_cast<bf16*>(output.data_ptr());
  p.out_lse = static_cast<float*>(lse.data_ptr());
  p.page_stride_bytes = m.page_stride_bytes;
  p.page_size = m.page_size;
  p.indices_stride_elems = m.indices_stride;
  p.out_lse_stride_elems = m.lse_stride;
  p.extra_fp4 = m.extra_fp4;
  p.use_full_bf16 = plan.numeric == execution::NumericRoute::FullBF16;
  p.topk_length =
      lengths.has_value() ? static_cast<const int*>(lengths.value().data_ptr()) : nullptr;
  p.attn_sink = sink.has_value() ? static_cast<const float*>(sink.value().data_ptr()) : nullptr;
  if (extra_cache.has_value()) {
    const auto& ex = extra_cache.value();
    const auto& ix = extra_indices.value();
    check_attention_device(q, ex, "extra_kv_cache");
    check_attention_device(q, ix, "extra_indices");
    CHECK_INPUT_TYPE(ix, dl_int32);
    TVM_FFI_ICHECK(ix.ndim() == 2 && ix.size(0) == m.tokens && ix.size(1) == m.extra_topk &&
                   ix.stride(0) == int64_t(m.extra_indices_stride) && ix.stride(1) == 1)
        << "execution plan extra indices mismatch";
    const auto el = parse_paged_kv_layout(
        ex, m.extra_fp4 ? Dsv41Fp4Layout::BYTES_PER_TOKEN : bytes_per_token(mt), false,
        "extra_kv_cache");
    TVM_FFI_ICHECK(el.page_block_size == m.extra_page_size &&
                   el.stride_kv_block == m.extra_page_stride_bytes)
        << "execution plan extra cache mismatch";
    p.extra_kv = static_cast<const uint8_t*>(ex.data_ptr());
    p.extra_indices = static_cast<const int32_t*>(ix.data_ptr());
    p.extra_topk = m.extra_topk;
    p.extra_page_size = m.extra_page_size;
    p.extra_page_stride_bytes = m.extra_page_stride_bytes;
    p.extra_indices_stride_elems = m.extra_indices_stride;
    p.extra_topk_length = extra_lengths.has_value()
                              ? static_cast<const int*>(extra_lengths.value().data_ptr())
                              : nullptr;
  }
  if (m.variant == 0) check_decode_scratch_capacity(q, mid, mlse, plan.chunk_capacity);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  if (m.variant != 0) {
    TVM_FFI_ICHECK(plan.numeric == execution::NumericRoute::FullBF16 ||
                   (indices.IsContiguous() &&
                    (!extra_indices.has_value() || extra_indices.value().IsContiguous())))
        << "prefill execution indices must be contiguous";
    const auto result = dispatch_prefill(p, plan, get_stream(q.device()));
    TVM_FFI_ICHECK(result.supported) << "prefill execution plan route mismatch";
    TVM_FFI_ICHECK_EQ(result.error, cudaSuccess)
        << "execute_attention " << result.operation << ": " << cudaGetErrorString(result.error);
    return;
  }
  if (inline_scale) {
    TVM_FFI_ICHECK_EQ(layout.stride_kv_row, m.row_stride_bytes)
        << "execution plan row stride mismatch";
    const auto status = dispatch_dsv32(p, plan, get_stream(q.device()));
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "execute_attention V32: " << cudaGetErrorString(status);
    return;
  }
  const auto status = dispatch_decode(p, plan, get_stream(q.device()));
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "execute_attention: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(inspect_metadata,
                              flashinfer::sparse_mla_sm120::inspect_attention_metadata<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(execute_attention,
                              flashinfer::sparse_mla_sm120::ExecuteAttentionPlan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_paged_attention,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120PagedAttention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv4,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv4<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv41_bf16,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv4<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv3_2,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv3_2);
