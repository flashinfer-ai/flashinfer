/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <flashinfer/hpc_rope.h>

#include <cmath>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

void check_common(TensorView key_cache, TensorView value_cache, TensorView qkv, TensorView cos_sin,
                  TensorView seq_lens, TensorView q_indptr, TensorView page_indices,
                  int64_t qk_norm_policy) {
  CHECK_INPUT(key_cache);
  CHECK_INPUT(value_cache);
  CHECK_INPUT_AND_TYPE(qkv, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(cos_sin, dl_float32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(q_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(page_indices, dl_int32);
  CHECK_DIM(4, key_cache);
  CHECK_DIM(4, value_cache);
  CHECK_DIM(2, qkv);
  CHECK_DIM(2, cos_sin);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, q_indptr);
  CHECK_DIM(2, page_indices);
  CHECK_DEVICE(value_cache, qkv);
  CHECK_DEVICE(key_cache, qkv);
  CHECK_DEVICE(cos_sin, qkv);
  CHECK_DEVICE(seq_lens, qkv);
  CHECK_DEVICE(q_indptr, qkv);
  CHECK_DEVICE(page_indices, qkv);
  TVM_FFI_ICHECK_GE(qk_norm_policy, 0);
  TVM_FFI_ICHECK_LE(qk_norm_policy, 2);
  TVM_FFI_ICHECK_EQ(q_indptr.size(0), seq_lens.size(0) + 1);
  TVM_FFI_ICHECK_EQ(page_indices.size(0), seq_lens.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0));
  TVM_FFI_ICHECK_EQ(key_cache.size(1), value_cache.size(1));
  TVM_FFI_ICHECK_EQ(key_cache.size(2), value_cache.size(2));
}

void check_optional_weight(Optional<TensorView> value, TensorView like, int64_t head_dim,
                           const char* name) {
  if (!value.has_value()) return;
  TensorView tensor = value.value();
  CHECK_INPUT_AND_TYPE(tensor, dl_float32);
  CHECK_DEVICE(tensor, like);
  CHECK_DIM(1, tensor);
  TVM_FFI_ICHECK_EQ(tensor.size(0), head_dim) << name;
}

template <typename T>
T* optional_ptr(Optional<TensorView> value) {
  return value.has_value() ? reinterpret_cast<T*>(value.value().data_ptr()) : nullptr;
}

void check_supported_shape(TensorView key_cache, TensorView value_cache, TensorView qkv) {
  const int64_t kv_heads = key_cache.size(2);
  const int64_t qk_dim = key_cache.size(3);
  const int64_t v_dim = value_cache.size(3);
  const int64_t q_heads = (qkv.size(1) - kv_heads * qk_dim - kv_heads * v_dim) / qk_dim;
  TVM_FFI_ICHECK((q_heads == 8 && kv_heads == 1) || (q_heads == 64 && kv_heads == 8))
      << "hpc fused RoPE supports (q_heads, kv_heads)=(8,1) or (64,8)";
  TVM_FFI_ICHECK_EQ(qk_dim, 128);
  TVM_FFI_ICHECK_EQ(v_dim, 128);
  TVM_FFI_ICHECK_EQ(qkv.size(1), q_heads * qk_dim + kv_heads * qk_dim + kv_heads * v_dim);
}

void check_shape_3d(TensorView value, int64_t dim0, int64_t dim1, int64_t dim2, const char* name) {
  TVM_FFI_ICHECK_EQ(value.ndim(), 3) << name << " must be 3D";
  TVM_FFI_ICHECK_EQ(value.size(0), dim0) << name << " dimension 0 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(1), dim1) << name << " dimension 1 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(2), dim2) << name << " dimension 2 mismatch";
}

void check_shape_2d(TensorView value, int64_t dim0, int64_t dim1, const char* name) {
  TVM_FFI_ICHECK_EQ(value.ndim(), 2) << name << " must be 2D";
  TVM_FFI_ICHECK_EQ(value.size(0), dim0) << name << " dimension 0 mismatch";
  TVM_FFI_ICHECK_EQ(value.size(1), dim1) << name << " dimension 1 mismatch";
}

}  // namespace

void hpc_rope_norm_store_kv(TensorView out_q, TensorView key_cache, TensorView value_cache,
                            TensorView qkv, TensorView cos_sin, TensorView seq_lens,
                            TensorView q_indptr, TensorView page_indices, bool is_prefill,
                            Optional<TensorView> q_norm_weight, Optional<TensorView> k_norm_weight,
                            Optional<TensorView> out_k, Optional<TensorView> out_v,
                            int64_t qk_norm_policy) {
  check_common(key_cache, value_cache, qkv, cos_sin, seq_lens, q_indptr, page_indices,
               qk_norm_policy);
  CHECK_INPUT_AND_TYPE(out_q, dl_bfloat16);
  CHECK_DEVICE(out_q, qkv);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_bfloat16);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_bfloat16);
  check_supported_shape(key_cache, value_cache, qkv);
  const int64_t num_rows = qkv.size(0);
  const int64_t num_kv_heads = key_cache.size(2);
  const int64_t head_dim = key_cache.size(3);
  const int64_t num_q_heads =
      (qkv.size(1) - num_kv_heads * (head_dim + value_cache.size(3))) / head_dim;
  check_shape_3d(out_q, num_rows, num_q_heads, head_dim, "out_q");
  TVM_FFI_ICHECK_EQ(cos_sin.size(1), head_dim)
      << "cos_sin second dimension must equal the Q/K head dimension";
  if (qk_norm_policy != 0) {
    TVM_FFI_ICHECK(q_norm_weight.has_value() && k_norm_weight.has_value())
        << "q_norm_weight and k_norm_weight are required when qk_norm_policy != 0";
  }
  check_optional_weight(q_norm_weight, qkv, key_cache.size(3), "q_norm_weight shape mismatch");
  check_optional_weight(k_norm_weight, qkv, key_cache.size(3), "k_norm_weight shape mismatch");
  if (out_k.has_value()) {
    CHECK_INPUT_AND_TYPE(out_k.value(), dl_bfloat16);
    CHECK_DEVICE(out_k.value(), qkv);
    check_shape_3d(out_k.value(), num_rows, num_kv_heads, head_dim, "out_k");
  }
  if (out_v.has_value()) {
    CHECK_INPUT_AND_TYPE(out_v.value(), dl_bfloat16);
    CHECK_DEVICE(out_v.value(), qkv);
    check_shape_3d(out_v.value(), num_rows, num_kv_heads, value_cache.size(3), "out_v");
  }

  ffi::CUDADeviceGuard device_guard(qkv.device().device_id);
  hpc::rope::rope_norm_store_kv_async(
      reinterpret_cast<__nv_bfloat16*>(out_q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr()), optional_ptr<__nv_bfloat16>(out_k),
      optional_ptr<__nv_bfloat16>(out_v), reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      static_cast<const float*>(cos_sin.data_ptr()), static_cast<const int*>(seq_lens.data_ptr()),
      static_cast<const int*>(q_indptr.data_ptr()),
      static_cast<const int*>(page_indices.data_ptr()), optional_ptr<float>(q_norm_weight),
      optional_ptr<float>(k_norm_weight), static_cast<int>(key_cache.stride(0)),
      static_cast<int>(value_cache.stride(0)), static_cast<int>(seq_lens.size(0)),
      static_cast<int>(page_indices.size(1)), static_cast<int>(key_cache.size(1)),
      static_cast<int>(qkv.size(0)), static_cast<int>(out_q.size(1)),
      static_cast<int>(key_cache.size(2)), static_cast<int>(key_cache.size(3)),
      static_cast<int>(value_cache.size(3)), is_prefill, static_cast<int>(qk_norm_policy),
      get_stream(qkv.device()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void hpc_rope_norm_store_kv_fp8(TensorView out_q, TensorView q_scale, TensorView split_k_flag,
                                TensorView key_cache, TensorView value_cache, TensorView qkv,
                                TensorView cos_sin, TensorView seq_lens, TensorView q_indptr,
                                TensorView page_indices, bool is_prefill, TensorView k_scale,
                                TensorView v_scale, int64_t quant_policy, int64_t max_seqlen,
                                double upper_max, Optional<TensorView> q_scale_inv,
                                Optional<TensorView> q_norm_weight,
                                Optional<TensorView> k_norm_weight, Optional<TensorView> out_k,
                                Optional<TensorView> out_v, int64_t qk_norm_policy) {
  check_common(key_cache, value_cache, qkv, cos_sin, seq_lens, q_indptr, page_indices,
               qk_norm_policy);
  CHECK_INPUT_AND_TYPE(out_q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(q_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(split_k_flag, dl_int32);
  CHECK_INPUT_AND_TYPE(k_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(v_scale, dl_float32);
  CHECK_DEVICE(out_q, qkv);
  CHECK_DEVICE(q_scale, qkv);
  CHECK_DEVICE(split_k_flag, qkv);
  CHECK_DEVICE(k_scale, qkv);
  CHECK_DEVICE(v_scale, qkv);
  TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK_EQ(value_cache.dtype(), dl_float8_e4m3fn);
  TVM_FFI_ICHECK(quant_policy == 1 || quant_policy == 2);
  TVM_FFI_ICHECK_EQ(k_scale.numel(), 1);
  TVM_FFI_ICHECK_EQ(v_scale.numel(), 1);
  check_supported_shape(key_cache, value_cache, qkv);
  TVM_FFI_ICHECK(std::isfinite(upper_max) && upper_max > 0.0 && upper_max <= 448.0)
      << "upper_max must be finite and in the interval (0, 448]";
  const int64_t num_rows = qkv.size(0);
  const int64_t num_requests = seq_lens.size(0);
  const int64_t num_kv_heads = key_cache.size(2);
  const int64_t head_dim = key_cache.size(3);
  const int64_t num_q_heads =
      (qkv.size(1) - num_kv_heads * (head_dim + value_cache.size(3))) / head_dim;
  check_shape_3d(out_q, num_rows, num_q_heads, head_dim, "out_q");
  check_shape_2d(split_k_flag, num_requests, num_kv_heads, "split_k_flag");
  TVM_FFI_ICHECK_EQ(cos_sin.size(1), head_dim)
      << "cos_sin second dimension must equal the Q/K head dimension";
  if (quant_policy == 1) {
    if (is_prefill) {
      TVM_FFI_ICHECK_GT(max_seqlen, 0)
          << "max_seqlen must be positive for dynamic prefill quantization";
      TVM_FFI_ICHECK_GE(num_requests * max_seqlen, num_rows)
          << "max_seqlen is too small for the packed Q row count";
      const int64_t aligned_max_seqlen = (max_seqlen + 127) / 128 * 128;
      check_shape_3d(q_scale, num_requests, num_q_heads, aligned_max_seqlen, "q_scale");
    } else {
      check_shape_2d(q_scale, num_rows, num_q_heads, "q_scale");
    }
  } else {
    TVM_FFI_ICHECK_EQ(q_scale.numel(), 0) << "q_scale must be empty for static Q quantization";
  }
  if (qk_norm_policy != 0) {
    TVM_FFI_ICHECK(q_norm_weight.has_value() && k_norm_weight.has_value());
  }
  if (quant_policy == 2) {
    TVM_FFI_ICHECK(q_scale_inv.has_value()) << "q_scale_inv is required for static Q quantization";
  }
  check_optional_weight(q_norm_weight, qkv, key_cache.size(3), "q_norm_weight shape mismatch");
  check_optional_weight(k_norm_weight, qkv, key_cache.size(3), "k_norm_weight shape mismatch");
  if (q_scale_inv.has_value()) {
    CHECK_INPUT_AND_TYPE(q_scale_inv.value(), dl_float32);
    CHECK_DEVICE(q_scale_inv.value(), qkv);
    TVM_FFI_ICHECK_EQ(q_scale_inv.value().numel(), 1);
  }
  if (out_k.has_value()) {
    CHECK_INPUT_AND_TYPE(out_k.value(), dl_float8_e4m3fn);
    CHECK_DEVICE(out_k.value(), qkv);
    check_shape_3d(out_k.value(), num_rows, num_kv_heads, head_dim, "out_k");
  }
  if (out_v.has_value()) {
    CHECK_INPUT_AND_TYPE(out_v.value(), dl_float8_e4m3fn);
    CHECK_DEVICE(out_v.value(), qkv);
    check_shape_3d(out_v.value(), num_rows, num_kv_heads, value_cache.size(3), "out_v");
  }

  ffi::CUDADeviceGuard device_guard(qkv.device().device_id);
  hpc::rope::rope_norm_store_kv_fp8_async(
      reinterpret_cast<__nv_fp8_e4m3*>(out_q.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(key_cache.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(value_cache.data_ptr()), optional_ptr<__nv_fp8_e4m3>(out_k),
      optional_ptr<__nv_fp8_e4m3>(out_v), static_cast<int32_t*>(split_k_flag.data_ptr()),
      q_scale.numel() ? static_cast<float*>(q_scale.data_ptr()) : nullptr,
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      static_cast<const float*>(cos_sin.data_ptr()), static_cast<const int*>(seq_lens.data_ptr()),
      static_cast<const int*>(q_indptr.data_ptr()),
      static_cast<const int*>(page_indices.data_ptr()), optional_ptr<float>(q_norm_weight),
      optional_ptr<float>(k_norm_weight), static_cast<const float*>(k_scale.data_ptr()),
      static_cast<const float*>(v_scale.data_ptr()), optional_ptr<float>(q_scale_inv),
      static_cast<float>(upper_max), static_cast<int>(max_seqlen),
      static_cast<int>(key_cache.stride(0)), static_cast<int>(value_cache.stride(0)),
      static_cast<int>(seq_lens.size(0)), static_cast<int>(page_indices.size(1)),
      static_cast<int>(key_cache.size(1)), static_cast<int>(qkv.size(0)),
      static_cast<int>(out_q.size(1)), static_cast<int>(key_cache.size(2)),
      static_cast<int>(key_cache.size(3)), static_cast<int>(value_cache.size(3)), is_prefill,
      static_cast<int>(qk_norm_policy), static_cast<int>(quant_policy), get_stream(qkv.device()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(hpc_rope_norm_store_kv, hpc_rope_norm_store_kv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hpc_rope_norm_store_kv_fp8, hpc_rope_norm_store_kv_fp8);
