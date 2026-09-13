// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <flashinfer/attention/sparse_mla_sm120/model/dsv41_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/dsv4_nvfp4_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

#include "../tvm_ffi_utils.h"
#include "attention_descriptor.h"

namespace flashinfer::sparse_mla_sm120 {

struct PagedKVLayout {
  int page_block_size;
  size_t stride_kv_block;
  int stride_kv_row;
};

inline PagedKVLayout parse_paged_kv_layout(const TensorView& kv, int bpt, bool inline_scale,
                                           const char* name) {
  CHECK_CUDA(kv);
  CHECK_INPUT_TYPE(kv, dl_uint8);
  TVM_FFI_ICHECK(kv.ndim() >= 2 && kv.ndim() <= 4) << name << " must be 2D, 3D or 4D";
  TVM_FFI_ICHECK_EQ(kv.stride(-1), 1) << name << " last dim must be contiguous";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16, 0)
      << name << " data pointer must be 16B-aligned";
  const int64_t page_stride = kv.stride(0);
  TVM_FFI_ICHECK_EQ(page_stride % 16, 0) << name << " block stride must be 16B-aligned";
  if (kv.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(kv.size(1) % bpt, 0)
        << name << " block width must be divisible by bytes_per_token";
    TVM_FFI_ICHECK_GT(kv.size(1), 0) << name << " page capacity must be positive";
    TVM_FFI_ICHECK_GE(page_stride, kv.size(1))
        << name << " block stride is smaller than packed block width";
    return {static_cast<int>(kv.size(1) / bpt), static_cast<size_t>(page_stride), bpt};
  }
  int axis = 1;
  if (kv.ndim() == 4) {
    TVM_FFI_ICHECK(kv.size(1) == 1 || kv.size(2) == 1)
        << name << " requires singleton KV-head axis";
    axis = kv.size(1) == 1 ? 2 : 1;
  }
  const int64_t rows = kv.size(axis), width = kv.size(-1), advance = kv.stride(axis);
  TVM_FFI_ICHECK_GT(rows, 0) << name << " page capacity must be positive";
  TVM_FFI_ICHECK_GE(width, bpt) << name << " row width is smaller than bytes_per_token";
  TVM_FFI_ICHECK_GE(advance, width) << name << " token-axis stride is smaller than row width";
  if (inline_scale) {
    TVM_FFI_ICHECK_EQ(advance % 16, 0) << name << " token-axis stride must be 16B-aligned";
  } else {
    TVM_FFI_ICHECK_EQ(advance, bpt) << name << " footer-scale rows must stay tightly packed";
  }
  TVM_FFI_ICHECK_GE(page_stride, rows * advance)
      << name << " block stride is smaller than page capacity";
  return {static_cast<int>(rows), static_cast<size_t>(page_stride), static_cast<int>(advance)};
}

inline void check_attention_device(const TensorView& q, const TensorView& tensor,
                                   const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA &&
                 tensor.device().device_id == q.device().device_id)
      << name << " must be on the same CUDA device as q";
}

inline void check_attention_optional(const TensorView& q, const ffi::Optional<TensorView>& tensor,
                                     const char* name) {
  if (tensor.has_value()) check_attention_device(q, tensor.value(), name);
}

inline void check_attention_vector(const TensorView& q, const ffi::Optional<TensorView>& tensor,
                                   int64_t count, DLDataType dtype, const char* name) {
  if (!tensor.has_value()) return;
  const auto& value = tensor.value();
  check_attention_device(q, value, name);
  TVM_FFI_ICHECK_EQ(value.dtype(), dtype) << name << " has incorrect dtype";
  TVM_FFI_ICHECK(value.ndim() == 1 && value.size(0) == count && value.IsContiguous())
      << name << " must be a contiguous vector of length " << count;
}

inline void check_attention_alignment(const TensorView& tensor, size_t alignment,
                                      const char* name) {
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment, 0)
      << name << " must be " << alignment << "B-aligned";
}

inline void check_decode_tensors(const TensorView& q, const TensorView& cache,
                                 const TensorView& indices, const TensorView& mid_out,
                                 const TensorView& mid_lse, const TensorView& output,
                                 const TensorView& lse, int64_t splits, size_t q_alignment = 16) {
  check_attention_alignment(q, q_alignment, "q");
  check_attention_alignment(output, 16, "output");
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_DIM(3, q);
  const int64_t tokens = q.size(0), heads = q.size(1);
  TVM_FFI_ICHECK(tokens > 0 && heads > 0 && heads <= execution::DecodeMaxHeads && splits > 0)
      << "decode requires positive tokens/splits and heads <= 128";
  check_attention_device(q, cache, "kv_cache");
  check_attention_device(q, indices, "indices");
  check_attention_device(q, mid_out, "mid_out");
  check_attention_device(q, mid_lse, "mid_lse");
  check_attention_device(q, output, "output");
  check_attention_device(q, lse, "out_lse");
  CHECK_INPUT_TYPE(cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(mid_out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(mid_lse, dl_float32);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(mid_out.data_ptr()) % 16, 0)
      << "mid_out must be 16B-aligned";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(mid_lse.data_ptr()) % 16, 0)
      << "mid_lse must be 16B-aligned";
  CHECK_INPUT_TYPE(indices, dl_int32);
  CHECK_INPUT_TYPE(lse, dl_float32);
  const int64_t dim = q.size(2) == KVCacheTraits<ModelType::DOTS3_SWA>::D_QK
                          ? KVCacheTraits<ModelType::DOTS3_SWA>::D_V
                          : KVCacheTraits<ModelType::DSV4>::D_V;
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == tokens && output.size(1) == heads &&
                 output.size(2) == dim)
      << "output shape mismatch";
  TVM_FFI_ICHECK(lse.ndim() == 2 && lse.size(0) == tokens && lse.size(1) == heads &&
                 lse.stride(1) == 1 && lse.stride(0) >= heads)
      << "out_lse shape/stride mismatch";
  TVM_FFI_ICHECK(indices.ndim() == 2 || (indices.ndim() == 3 && indices.size(1) == 1))
      << "indices must be [T, K] or [T, 1, K]";
  TVM_FFI_ICHECK_EQ(indices.size(0), tokens) << "indices leading dimension must match num_tokens";
  TVM_FFI_ICHECK(indices.size(-1) > 0 && indices.stride(-1) == 1 &&
                 indices.stride(0) >= indices.size(-1))
      << "indices shape/stride mismatch";
}

inline void check_decode_scratch_capacity(const TensorView& q, const TensorView& mid_out,
                                          const TensorView& mid_lse, int64_t splits) {
  const int64_t tokens = q.size(0), heads = q.size(1);
  const int64_t dim = q.size(2) == KVCacheTraits<ModelType::DOTS3_SWA>::D_QK
                          ? KVCacheTraits<ModelType::DOTS3_SWA>::D_V
                          : KVCacheTraits<ModelType::DSV4>::D_V;
  const int64_t scratch_heads = execution::decode_scratch_heads(heads);
  int64_t out_capacity = 1, lse_capacity = 1;
  for (int i = 0; i < mid_out.ndim(); ++i) out_capacity *= mid_out.size(i);
  for (int i = 0; i < mid_lse.ndim(); ++i) lse_capacity *= mid_lse.size(i);
  TVM_FFI_ICHECK_GE(out_capacity, tokens * scratch_heads * splits * dim)
      << "mid_out scratch capacity must cover padded heads and allocated splits";
  TVM_FFI_ICHECK_GE(lse_capacity, tokens * scratch_heads * splits)
      << "mid_lse scratch capacity must cover padded heads and allocated splits";
}

template <bool IsDsv4Nvfp4>
ffi::Array<int64_t> inspect_attention_metadata(
    TensorView q, TensorView cache, TensorView indices, TensorView output,
    ffi::Optional<TensorView> lengths, ffi::Optional<TensorView> sink,
    ffi::Optional<TensorView> extra, ffi::Optional<TensorView> extra_indices,
    ffi::Optional<TensorView> extra_lengths, ffi::Optional<TensorView> lse,
    ffi::Optional<TensorView> mid, ffi::Optional<TensorView> mlse, int64_t model, bool extra_fp4,
    int64_t value_dim) {
  TVM_FFI_ICHECK(model >= 0 && model <= int(ModelType::DSV4_1)) << "unknown cache format";
  const auto format = cache_format_info(static_cast<ModelType>(model));
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_DIM(3, q);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  const int t = q.size(0), h = q.size(1);
  TVM_FFI_ICHECK(t > 0 && h > 0 && h <= execution::DecodeMaxHeads) << "unsupported tokens/heads";
  TVM_FFI_ICHECK(q.size(2) == format.query_dim && (value_dim == 0 || value_dim == format.value_dim))
      << "query/d_v geometry mismatch";
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == t && output.size(1) == h &&
                 output.size(2) == format.value_dim)
      << "output shape mismatch";
  TVM_FFI_ICHECK(extra.has_value() == extra_indices.has_value())
      << "extra cache and indices must be provided together";
  TVM_FFI_ICHECK(!extra_lengths.has_value() || extra.has_value()) << "extra lengths require cache";
  TVM_FFI_ICHECK(mid.has_value() == mlse.has_value()) << "scratch must be passed together";
  check_attention_device(q, output, "output");
  check_attention_vector(q, lengths, t, dl_int32, "topk_length");
  check_attention_vector(q, extra_lengths, t, dl_int32, "extra_topk_length");
  check_attention_vector(q, sink, h, dl_float32, "attn_sink");
  auto index_check = [&](TensorView ix) {
    check_attention_device(q, ix, "indices");
    CHECK_INPUT_TYPE(ix, dl_int32);
    TVM_FFI_ICHECK((ix.ndim() == 2 || (ix.ndim() == 3 && ix.size(1) == 1)) && ix.size(0) == t &&
                   ix.size(-1) > 0 && ix.stride(-1) == 1 && ix.stride(0) >= ix.size(-1))
        << "indices shape/stride mismatch";
    if constexpr (IsDsv4Nvfp4) TVM_FFI_ICHECK(ix.IsContiguous()) << "NVFP4 indices must be contiguous";
  };
  index_check(indices);
  size_t lse_stride = h;
  if (lse.has_value()) {
    const auto& value = lse.value();
    check_attention_device(q, value, "out_lse");
    CHECK_INPUT_TYPE(value, dl_float32);
    TVM_FFI_ICHECK(value.ndim() == 2 && value.size(0) >= t && value.size(1) >= h &&
                   value.stride(1) == 1 && value.stride(0) >= h)
        << "out_lse capacity/stride mismatch";
    lse_stride = value.stride(0);
  }
  for (const auto& item : {std::make_pair(mid, dl_bfloat16), std::make_pair(mlse, dl_float32)}) {
    if (!item.first.has_value()) continue;
    const auto& value = item.first.value();
    check_attention_device(q, value, "scratch");
    TVM_FFI_ICHECK(value.dtype() == item.second && value.IsContiguous())
        << "scratch layout mismatch";
    check_attention_alignment(value, 16, "scratch");
  }
  auto parse = [&](TensorView value, bool compressed) {
    check_attention_device(q, value, "kv_cache");
    const int bpt = IsDsv4Nvfp4       ? Dsv4Nvfp4Layout::BYTES_PER_TOKEN
                    : compressed ? Dsv41Fp4Layout::BYTES_PER_TOKEN
                                 : format.bytes_per_token;
    if constexpr (IsDsv4Nvfp4) {
      TVM_FFI_ICHECK(value.ndim() == 2 || value.size(-1) == bpt) << "NVFP4 cache width mismatch";
    }
    return parse_paged_kv_layout(value, bpt, !IsDsv4Nvfp4 && format.inline_scale, "kv_cache");
  };
  const auto layout = parse(cache, false);
  PagedKVLayout ex{};
  if (extra.has_value()) {
    index_check(extra_indices.value());
    ex = parse(extra.value(), extra_fp4);
  }
  if constexpr (IsDsv4Nvfp4) TVM_FFI_ICHECK(lse_stride == size_t(h)) << "NVFP4 LSE must be contiguous";
  return execution::pack_metadata(
      {int(model), t, h, int(indices.size(-1)),
       extra_indices.has_value() ? int(extra_indices.value().size(-1)) : 0, layout.page_block_size,
       ex.page_block_size, layout.stride_kv_block, ex.stride_kv_block, layout.stride_kv_row,
       size_t(indices.stride(0)),
       extra_indices.has_value() ? size_t(extra_indices.value().stride(0)) : 0, lse_stride,
       lengths.has_value(), extra_lengths.has_value(), sink.has_value(), extra_fp4, 0});
}

}  // namespace flashinfer::sparse_mla_sm120
