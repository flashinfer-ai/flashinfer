// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// TVM-FFI binding for sparse-MLA SM120 paged attention.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

using bf16 = __nv_bfloat16;

void SparseMlaSm120PagedAttention(TensorView q, TensorView kv_cache, TensorView indices,
                                  TensorView output, TensorView out_lse, double sm_scale,
                                  int64_t model_type, int64_t variant,
                                  Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                  Optional<TensorView> extra_kv_cache,
                                  Optional<TensorView> extra_indices,
                                  Optional<TensorView> extra_topk_length);

bool launch_sparse_mla_decode_dsv4(
    ModelType mt, int num_heads, int topk, int page_block_size, int num_tokens, int num_splits,
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out, float* mid_lse,
    bf16* output, float* out_lse, const int* topk_length, const float* attn_sink,
    const uint8_t* extra_KV_cache, const int32_t* extra_indices, const int* extra_topk_length,
    int extra_topk, int pbs_extra, size_t stride_extra_kv_block, int chunks_per_block_override,
    float sm_scale, size_t stride_kv_block, size_t stride_indices_token,
    size_t stride_extra_indices_token, cudaStream_t stream);

bool launch_sparse_mla_decode_dsv3_2(ModelType mt, int num_heads, int topk, int num_tokens,
                                     int num_splits, const bf16* Q, const uint8_t* KV_cache,
                                     const int32_t* indices, bf16* mid_out, float* mid_lse,
                                     bf16* output, float* out_lse, const int* topk_length,
                                     const float* attn_sink, int chunks_per_block_override,
                                     float sm_scale, size_t stride_kv_block,
                                     size_t stride_indices_token, int stride_kv_row,
                                     cudaStream_t stream);

namespace {

struct PagedKVLayout {
  int page_block_size;
  size_t stride_kv_block;
  // Per-token advance in bytes. Equals bytes_per_token for a packed cache, but
  // may be larger when the caller pads rows so several layer types with
  // different geometries can share one KV cache group. The packed payload
  // always sits at the row start, so only the advance changes.
  int stride_kv_row;
};

inline PagedKVLayout parse_paged_kv_layout(const TensorView& kv, int bpt, const char* name) {
  const size_t elem_bytes = static_cast<size_t>(kv.dtype().bits / 8);
  if (kv.ndim() == 2) {
    const size_t block_bytes = static_cast<size_t>(kv.size(1)) * elem_bytes;
    TVM_FFI_ICHECK_EQ(block_bytes % static_cast<size_t>(bpt), 0)
        << name << " 2D block width " << block_bytes
        << " is not divisible by bytes_per_token=" << bpt;
    // A flat 2D block carries no row padding to infer, so the row advance is
    // exactly bytes_per_token.
    return {static_cast<int>(block_bytes / static_cast<size_t>(bpt)), block_bytes, bpt};
  }
  auto row_advance = [&](int64_t token_axis) {
    TVM_FFI_ICHECK_EQ(kv.stride(-1), 1) << name << " last dim must be contiguous";
    const size_t bytes = static_cast<size_t>(kv.size(-1)) * elem_bytes;
    TVM_FFI_ICHECK_GE(bytes, static_cast<size_t>(bpt))
        << name << " row width " << bytes << " is smaller than bytes_per_token=" << bpt;
    // The per-token advance is the token axis's real stride: a caller slicing
    // the last dim of a wider buffer (padded rows) changes the size but not
    // the stride, and the kernel must step by the stride.
    const size_t advance = static_cast<size_t>(kv.stride(token_axis)) * elem_bytes;
    TVM_FFI_ICHECK_GE(advance, bytes)
        << name << " token-axis stride " << advance << " is smaller than the row width " << bytes;
    return static_cast<int>(advance);
  };
  if (kv.ndim() == 3) {
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes,
            row_advance(1)};
  }
  TVM_FFI_ICHECK_EQ(kv.ndim(), 4) << name << " must be 2D [num_pages, page_bytes], 3D "
                                  << "[num_pages, page_block_size, bytes_per_token], HND "
                                  << "[num_pages, 1, page_block_size, bytes_per_token], or NHD "
                                  << "[num_pages, page_block_size, 1, bytes_per_token]";
  if (kv.size(1) == 1) {
    return {static_cast<int>(kv.size(2)), static_cast<size_t>(kv.stride(0)) * elem_bytes,
            row_advance(2)};
  }
  if (kv.size(2) == 1) {
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes,
            row_advance(1)};
  }
  TVM_FFI_ICHECK(false) << name << " 4D form must have singleton KV-head axis at dim 1 "
                        << "(HND) or dim 2 (NHD)";
  return {0, 0, 0};
}

}  // namespace

// Thin TVM-FFI wrapper for the decode-dsv4 standalone path. The caller passes
// already-sized scratch tensors mid_out + mid_lse plus the output and lse.
// Handles DSV4 decode with page_block_size=64 and the supported head/top-k
// instantiation grid in sparse_mla_sm120_decode_dsv4.cu.
void SparseMlaSm120DecodeDsv4(TensorView q, TensorView kv_cache, TensorView indices,
                              TensorView mid_out, TensorView mid_lse, TensorView output,
                              TensorView out_lse, int64_t num_splits, double sm_scale,
                              Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                              Optional<TensorView> extra_kv_cache,
                              Optional<TensorView> extra_indices,
                              Optional<TensorView> extra_topk_length,
                              int64_t chunks_per_block_override) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
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

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  // This kernel serves the footer-scale model types. d_qk selects between them:
  // 512 -> DSV4, 1088 -> DOTS3_SWA (sliding-window family, d_v 1024).
  TVM_FFI_ICHECK(d_qk == 512 || d_qk == 1088)
      << "decode-dsv4 supports d_qk 512 (DSV4) or 1088 (DOTS3_SWA); got " << d_qk;
  const ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DOTS3_SWA;
  // DOTS3_SWA's sliding window (513 candidates, DecodeTileCfg::WINDOW) needs an
  // indices buffer at least that wide; a narrower one can never name the full
  // window. Report it here so the message names the actual constraint.
  TVM_FFI_ICHECK(mt != ModelType::DOTS3_SWA || topk >= 513)
      << "decode-dsv4 (dots3_swa) requires topk >= 513 to hold the 513-wide "
         "sliding window; got indices width topk="
      << topk;

  // topk_length is optional for DOTS3_SWA: DecodeTileCfg<DOTS3_SWA>::WINDOW caps
  // the per-token candidate count inside the kernel, so omitting it costs
  // nothing beyond the window itself. Unused slots must still carry -1, which
  // the QK mask turns into -inf.
  const int bpt = bytes_per_token(mt);
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, bpt, "kv_cache");
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
    TVM_FFI_ICHECK_EQ(eidx.stride(-1), 1) << "extra_indices last dimension must be contiguous";
    extra_topk_arg = static_cast<int>(eidx.size(-1));
    stride_extra_indices_token = static_cast<size_t>(eidx.stride(0));
    // The extra (dual) cache carries the same per-token layout as the main one.
    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, bpt, "extra_kv_cache");
    TVM_FFI_ICHECK_EQ(extra_layout.stride_kv_row, bpt)
        << "decode-dsv4 extra_kv_cache requires tightly packed KV rows "
        << "(stride_kv_row == bytes_per_token=" << bpt
        << "); padded-row KV caches are supported only on the decode-v32 path";
    pbs_extra_arg = extra_layout.page_block_size;
    stride_extra_kv_block = extra_layout.stride_kv_block;
  }

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_dsv4(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr, extra_kv_ptr,
      extra_indices_ptr, extra_topk_len_ptr, extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),
      kv_layout.stride_kv_block, stride_indices_token, stride_extra_indices_token, stream);
  TVM_FFI_ICHECK(ok) << "decode-dsv4 launch failed (unsupported shape or kernel error)";
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
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2);
  TVM_FFI_ICHECK_GE(indices.ndim(), 2);
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }
  CHECK_INPUT_TYPE(indices, dl_int32);
  TVM_FFI_ICHECK_EQ(indices.stride(-1), 1) << "indices last dimension must be contiguous";
  const size_t stride_indices_token = static_cast<size_t>(indices.stride(0));

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK((d_qk == 576 && (mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)) ||
                 (d_qk == 512 && mt == ModelType::GLM53_NOPE))
      << "decode-v32 expects DSV3_2/GLM_NSA d_qk=576 or GLM53_NOPE d_qk=512; got d_qk=" << d_qk
      << " model_type=" << model_type;

  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, bytes_per_token(mt), "kv_cache");

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_dsv3_2(
      mt, num_heads, topk, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale),
      kv_layout.stride_kv_block, stride_indices_token, kv_layout.stride_kv_row, stream);
  TVM_FFI_ICHECK(ok) << "decode-dsv3_2 launch failed (unsupported shape or kernel error)";
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_paged_attention,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120PagedAttention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv4,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv3_2,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv3_2);
