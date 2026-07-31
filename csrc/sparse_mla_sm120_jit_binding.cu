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
                                  int64_t model_type, Optional<TensorView> topk_length,
                                  Optional<TensorView> attn_sink,
                                  Optional<TensorView> extra_kv_cache,
                                  Optional<TensorView> extra_indices,
                                  Optional<TensorView> extra_topk_length);

bool launch_sparse_mla_decode_dsv4(ModelType mt, int num_heads, int topk, int page_block_size,
                                   int num_tokens, int num_splits, const bf16* Q,
                                   const uint8_t* KV_cache, const int32_t* indices, bf16* mid_out,
                                   float* mid_lse, bf16* output, float* out_lse,
                                   const int* topk_length, const float* attn_sink,
                                   const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                   const int* extra_topk_length, int extra_topk, int pbs_extra,
                                   size_t stride_extra_kv_block, int chunks_per_block_override,
                                   float sm_scale, size_t stride_kv_block, cudaStream_t stream);

bool launch_sparse_mla_decode_dsv3_2(ModelType mt, int num_heads, int topk, int num_tokens,
                                     int num_splits, const bf16* Q, const uint8_t* KV_cache,
                                     const int32_t* indices, bf16* mid_out, float* mid_lse,
                                     bf16* output, float* out_lse, const int* topk_length,
                                     const float* attn_sink, int chunks_per_block_override,
                                     float sm_scale, size_t stride_kv_block, cudaStream_t stream);

namespace {

struct PagedKVLayout {
  int page_block_size;
  size_t stride_kv_block;
};

inline PagedKVLayout parse_paged_kv_layout(const TensorView& kv, int bpt, const char* name) {
  const size_t elem_bytes = static_cast<size_t>(kv.dtype().bits / 8);
  if (kv.ndim() == 2) {
    const size_t block_bytes = static_cast<size_t>(kv.size(1)) * elem_bytes;
    TVM_FFI_ICHECK_EQ(block_bytes % static_cast<size_t>(bpt), 0)
        << name << " 2D block width " << block_bytes
        << " is not divisible by bytes_per_token=" << bpt;
    return {static_cast<int>(block_bytes / static_cast<size_t>(bpt)), block_bytes};
  }
  if (kv.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
        << name << " 3D form must be [num_pages, page_block_size, " << bpt << "]";
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK_EQ(kv.ndim(), 4) << name << " must be 2D [num_pages, page_bytes], 3D "
                                  << "[num_pages, page_block_size, bytes_per_token], HND "
                                  << "[num_pages, 1, page_block_size, bytes_per_token], or NHD "
                                  << "[num_pages, page_block_size, 1, bytes_per_token]";
  TVM_FFI_ICHECK_EQ(kv.size(-1), bpt)
      << name << " last dim must be bytes_per_token=" << bpt << ", got " << kv.size(-1);
  if (kv.size(1) == 1) {
    return {static_cast<int>(kv.size(2)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  if (kv.size(2) == 1) {
    return {static_cast<int>(kv.size(1)), static_cast<size_t>(kv.stride(0)) * elem_bytes};
  }
  TVM_FFI_ICHECK(false) << name << " 4D form must have singleton KV-head axis at dim 1 "
                        << "(HND) or dim 2 (NHD)";
  return {0, 0};
}

}  // namespace

// Thin TVM-FFI wrapper for the decode-dsv4 standalone path. The caller passes
// already-sized scratch tensors mid_out + mid_lse plus the output and lse.
// Currently only handles DSV4 h=128 topk=512 pbs=64.
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
  // the s_q singleton dim through the call stack). The kernel walks
  // `indices + t_idx * stride_per_t`, where stride is captured below from
  // .stride(0).
  TVM_FFI_ICHECK_GE(indices.ndim(), 2)
      << "indices must have at least 2 dims; got ndim=" << indices.ndim();
  if (indices.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(indices.size(1), 1)
        << "indices 3D form requires size(1) == 1; got " << indices.size(1);
  }

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::DSV4 : ModelType::DSV3_2;
  // Currently the kernel only supports DSV4.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::DSV4))
      << "decode-dsv4 currently DSV4-only";

  constexpr int BPT_DSV4 = 584;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV4, "kv_cache");
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
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(-1));
    const PagedKVLayout extra_layout = parse_paged_kv_layout(ekv, BPT_DSV4, "extra_kv_cache");
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
      kv_layout.stride_kv_block, stream);
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

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const int d_qk = static_cast<int>(q.size(2));
  TVM_FFI_ICHECK_EQ(d_qk, 576) << "decode-dsv3_2 expects DSV3_2 layout (d_qk=576); got " << d_qk;
  const auto mt = static_cast<ModelType>(model_type);
  TVM_FFI_ICHECK(mt == ModelType::DSV3_2 || mt == ModelType::GLM_NSA)
      << "decode-dsv3_2 expects model_type DSV3_2 or GLM_NSA; got " << model_type;

  constexpr int BPT_DSV3_2 = 656;
  const PagedKVLayout kv_layout = parse_paged_kv_layout(kv_cache, BPT_DSV3_2, "kv_cache");

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
      kv_layout.stride_kv_block, stream);
  TVM_FFI_ICHECK(ok) << "decode-dsv3_2 launch failed (unsupported shape or kernel error)";
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_paged_attention,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120PagedAttention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv4,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv3_2,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv3_2);
