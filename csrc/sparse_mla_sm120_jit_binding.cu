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
                                  Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
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

bool launch_sparse_mla_decode_dsv3_2(int num_heads, int topk, int num_tokens, int num_splits,
                                     const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                     bf16* mid_out, float* mid_lse, bf16* output, float* out_lse,
                                     const int* topk_length, const float* attn_sink,
                                     int chunks_per_block_override, float sm_scale,
                                     size_t stride_kv_block, cudaStream_t stream);

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
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2) << "kv_cache must be 2D [num_blocks, page_bytes] or 4D "
                                           "[num_blocks, page_block_size, 1, bpt]";
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

  // kv_cache may be:
  //   2D: [num_blocks, page_bytes]              — flat layout
  //   4D: [num_blocks, page_block_size, 1, bpt] — paged layout (may pad
  //       the block stride for alignment)
  // For 4D the block stride in bytes = product of dims[1..ndim-1] when
  // strictly contiguous; with padding it may exceed the natural row size.
  // Use stride(0) to capture the true block stride.
  size_t stride_kv_block;
  if (kv_cache.ndim() == 2) {
    stride_kv_block = static_cast<size_t>(kv_cache.size(1));
  } else {
    // stride(0) reports elements; convert to bytes via the dtype size.
    // For uint8 (1B element) this is identical numerically.
    stride_kv_block = static_cast<size_t>(kv_cache.stride(0));
  }
  // decode-dsv4 only supports page_block_size=64 (DSv4 layout). The
  // dispatcher rejects any other value.
  constexpr int page_block_size = 64;

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
  // extra_topk and stride_extra_kv_block derived from the optional tensors.
  // pbs_extra: 4D form gives it directly from dim -3; 2D form infers it
  // from the total row width / BPT_DSV4 (= 584).
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(-1));
    if (ekv.ndim() >= 3) {
      pbs_extra_arg = static_cast<int>(ekv.size(-3));
      // Use stride(0) to capture true block stride including any padding.
      stride_extra_kv_block = static_cast<size_t>(ekv.stride(0));
    } else {
      // 2D fallback: assume DSV4 bpt = 584. Infer pbs from row width.
      constexpr int BPT_DSV4 = 584;
      stride_extra_kv_block = static_cast<size_t>(ekv.size(1));
      pbs_extra_arg = static_cast<int>(stride_extra_kv_block / BPT_DSV4);
    }
  }

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_dsv4(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr, extra_kv_ptr,
      extra_indices_ptr, extra_topk_len_ptr, extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale), stride_kv_block,
      stream);
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
                                int64_t chunks_per_block_override) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_GE(kv_cache.ndim(), 2) << "kv_cache must be 2D [num_blocks, page_bytes] or 4D "
                                           "[num_blocks, page_block_size, 1, bpt]";
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

  // kv_cache: 2D [num_blocks, page_bytes] or 4D [num_blocks, pbs, 1, bpt].
  size_t stride_kv_block;
  if (kv_cache.ndim() == 2) {
    stride_kv_block = static_cast<size_t>(kv_cache.size(1));
  } else {
    stride_kv_block = static_cast<size_t>(kv_cache.stride(0));
  }

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_dsv3_2(
      num_heads, topk, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()), static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(mid_out.data_ptr()),
      static_cast<float*>(mid_lse.data_ptr()), static_cast<bf16*>(output.data_ptr()),
      static_cast<float*>(out_lse.data_ptr()), topk_len_ptr, attn_sink_ptr,
      static_cast<int>(chunks_per_block_override), static_cast<float>(sm_scale), stride_kv_block,
      stream);
  TVM_FFI_ICHECK(ok) << "decode-dsv3_2 launch failed (unsupported shape or kernel error)";
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_paged_attention,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120PagedAttention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv4,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_dsv3_2,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeDsv3_2);
