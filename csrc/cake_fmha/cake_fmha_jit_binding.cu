/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include <tvm/ffi/container/variant.h>

#include <cstdint>

#include "include/cake_fmha.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::Variant;

namespace flashinfer {
namespace cake_fmha {

namespace {

constexpr int kHndLayout = 0;
constexpr int kBFloat16 = 0;
constexpr int kFloat16 = 1;
constexpr int kFloat8E4M3 = 2;
constexpr int kNvFp4 = 3;

__global__ void FillUniformQIndptr(int* q_indptr, int batch_size, int q_len) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index <= batch_size) {
    q_indptr[index] = index * q_len;
  }
}

int CakeDType(DLDataType dtype, bool allow_nvfp4, const char* tensor_name) {
  if (dtype == dl_bfloat16) {
    return kBFloat16;
  }
  if (dtype == dl_float16) {
    return kFloat16;
  }
  if (dtype == dl_float8_e4m3fn) {
    return kFloat8E4M3;
  }
  if (allow_nvfp4 && dtype == dl_uint8) {
    return kNvFp4;
  }
  TVM_FFI_THROW(TypeError) << tensor_name
                           << " must use bfloat16, float16, float8_e4m3fn"
                           << (allow_nvfp4 ? ", or packed NVFP4 uint8" : "");
  return -1;
}

double ScalarScale(Variant<double, ffi::Tensor> scale, const char* name) {
  auto scalar = scale.as<double>();
  if (scalar.has_value()) {
    return scalar.value();
  }
  TVM_FFI_THROW(ValueError)
      << "Cake FMHA requires " << name
      << " to be materialized as a host scalar before the JIT binding";
  return 0.0;
}

void CheckSameDevice(TensorView query, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(query.device().device_type, tensor.device().device_type)
      << name << " must be on the query device";
  TVM_FFI_ICHECK_EQ(query.device().device_id, tensor.device().device_id)
      << name << " must be on the query device";
}

struct CommonArgs {
  TensorView out;
  Optional<TensorView> out_scale_factor;
  TensorView query;
  TensorView key_cache;
  TensorView value_cache;
  TensorView workspace_buffer;
  TensorView block_tables;
  TensorView seq_lens;
  Optional<TensorView> q_indptr;
  Optional<TensorView> attention_sinks;
  Optional<TensorView> key_block_scales;
  Optional<TensorView> value_block_scales;
  Optional<TensorView> lse;
  int64_t batch_size;
  int64_t uniform_q_len;
  int64_t window_left;
  bool is_causal;
  bool uses_shared_paged_kv_idx;
  double bmm1_scale;
  double bmm2_scale;
  double o_sf_scale;
  int64_t o_sf_start_index;
  int64_t lse_stride_tokens;
  int64_t lse_stride_heads;
};

void LaunchCompat(const CommonArgs& args) {
  TVM_FFI_ICHECK_EQ(args.query.ndim(), 3) << "query must be rank 3 [tokens, heads, dim]";
  TVM_FFI_ICHECK_EQ(args.key_cache.ndim(), 4) << "key cache must be rank 4 HND";
  TVM_FFI_ICHECK_EQ(args.value_cache.ndim(), 4) << "value cache must be rank 4 HND";
  TVM_FFI_ICHECK_EQ(args.out.ndim(), 3) << "output must be rank 3";
  TVM_FFI_ICHECK_EQ(args.query.stride(2), 1) << "query head dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(args.key_cache.stride(3), 1)
      << "key cache head dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(args.value_cache.stride(3), 1)
      << "value cache head dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(args.out.stride(2), 1) << "output head dimension must be contiguous";
  TVM_FFI_ICHECK_GT(args.batch_size, 0) << "batch size must be positive";
  TVM_FFI_ICHECK_EQ(args.seq_lens.ndim(), 1) << "seq_lens must be rank 1";
  TVM_FFI_ICHECK_EQ(args.seq_lens.size(0), args.batch_size)
      << "seq_lens must contain one value per request";
  TVM_FFI_ICHECK(args.seq_lens.dtype() == dl_int32 || args.seq_lens.dtype() == dl_uint32)
      << "seq_lens must be int32 or uint32";
  CheckSameDevice(args.query, args.key_cache, "key_cache");
  CheckSameDevice(args.query, args.value_cache, "value_cache");
  CheckSameDevice(args.query, args.out, "out");
  CheckSameDevice(args.query, args.block_tables, "block_tables");
  CheckSameDevice(args.query, args.seq_lens, "seq_lens");

  int const q_dtype = CakeDType(args.query.dtype(), false, "query");
  int const kv_dtype = CakeDType(args.key_cache.dtype(), true, "KV cache");
  TVM_FFI_ICHECK_EQ(args.key_cache.dtype(), args.value_cache.dtype())
      << "key and value cache dtypes must match";
  int const o_dtype = CakeDType(args.out.dtype(), true, "output");
  int const num_q_heads = static_cast<int>(args.query.size(1));
  int const num_kv_heads = static_cast<int>(args.key_cache.size(1));
  int const head_dim = static_cast<int>(args.query.size(2));
  int const kv_head_dim =
      static_cast<int>(args.key_cache.size(3)) * (kv_dtype == kNvFp4 ? 2 : 1);
  int const out_head_dim = static_cast<int>(args.out.size(2)) * (o_dtype == kNvFp4 ? 2 : 1);
  int const page_size = static_cast<int>(args.key_cache.size(2));
  TVM_FFI_ICHECK(head_dim == 128 || head_dim == 256)
      << "Cake FMHA compatibility route supports head_dim 128 or 256";
  TVM_FFI_ICHECK_EQ(kv_head_dim, head_dim) << "query and KV head dimensions must match";
  TVM_FFI_ICHECK_EQ(out_head_dim, head_dim) << "query and output head dimensions must match";
  TVM_FFI_ICHECK_GT(num_kv_heads, 0);
  TVM_FFI_ICHECK_EQ(num_q_heads % num_kv_heads, 0)
      << "num_q_heads must be divisible by num_kv_heads";
  TVM_FFI_ICHECK(!(!args.is_causal && args.window_left >= 0))
      << "Cake FMHA does not support non-causal sliding-window attention";
  TVM_FFI_ICHECK(args.block_tables.dtype() == dl_int32 ||
                 args.block_tables.dtype() == dl_uint32)
      << "block_tables must be int32 or uint32";

  void const* k_scales_ptr = nullptr;
  void const* v_scales_ptr = nullptr;
  int64_t ksf_s0 = 0, ksf_s1 = 0, ksf_s2 = 0, ksf_s3 = 0;
  int64_t vsf_s0 = 0, vsf_s1 = 0, vsf_s2 = 0, vsf_s3 = 0;
  if (kv_dtype == kNvFp4) {
    TVM_FFI_ICHECK(args.key_block_scales.has_value() && args.value_block_scales.has_value())
        << "packed NVFP4 KV requires key and value scale tensors";
    auto const& ksf = args.key_block_scales.value();
    auto const& vsf = args.value_block_scales.value();
    TVM_FFI_ICHECK_EQ(ksf.ndim(), 4);
    TVM_FFI_ICHECK_EQ(vsf.ndim(), 4);
    TVM_FFI_ICHECK_EQ(ksf.dtype(), dl_float8_e4m3fn);
    TVM_FFI_ICHECK_EQ(vsf.dtype(), dl_float8_e4m3fn);
    k_scales_ptr = ksf.data_ptr();
    v_scales_ptr = vsf.data_ptr();
    ksf_s0 = ksf.stride(0);
    ksf_s1 = ksf.stride(1);
    ksf_s2 = ksf.stride(2);
    ksf_s3 = ksf.stride(3);
    vsf_s0 = vsf.stride(0);
    vsf_s1 = vsf.stride(1);
    vsf_s2 = vsf.stride(2);
    vsf_s3 = vsf.stride(3);
  }

  void* output_scales_ptr = nullptr;
  int output_scale_columns = 0;
  if (o_dtype == kNvFp4) {
    TVM_FFI_ICHECK_EQ(q_dtype, kFloat8E4M3);
    TVM_FFI_ICHECK_EQ(kv_dtype, kFloat8E4M3);
    TVM_FFI_ICHECK(args.out_scale_factor.has_value())
        << "packed NVFP4 output requires an output scale tensor";
    auto const& output_scales = args.out_scale_factor.value();
    TVM_FFI_ICHECK_EQ(output_scales.ndim(), 2);
    TVM_FFI_ICHECK_EQ(output_scales.dtype(), dl_float8_e4m3fn);
    TVM_FFI_ICHECK(output_scales.IsContiguous());
    TVM_FFI_ICHECK_GT(args.o_sf_scale, 0.0);
    output_scales_ptr = output_scales.data_ptr();
    output_scale_columns = static_cast<int>(output_scales.size(1));
  }

  float* sinks_ptr = nullptr;
  if (args.attention_sinks.has_value()) {
    auto const& sinks = args.attention_sinks.value();
    TVM_FFI_ICHECK_EQ(sinks.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(sinks.numel(), num_q_heads);
    sinks_ptr = static_cast<float*>(sinks.data_ptr());
  }

  float* lse_ptr = nullptr;
  if (args.lse.has_value()) {
    auto const& lse = args.lse.value();
    TVM_FFI_ICHECK_EQ(lse.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(lse.ndim(), 2);
    TVM_FFI_ICHECK_EQ(lse.size(0), args.query.size(0));
    TVM_FFI_ICHECK_EQ(lse.size(1), num_q_heads);
    TVM_FFI_ICHECK_EQ(args.lse_stride_tokens, num_q_heads)
        << "Cake FMHA currently requires contiguous LSE";
    TVM_FFI_ICHECK_EQ(args.lse_stride_heads, 1)
        << "Cake FMHA currently requires contiguous LSE";
    lse_ptr = static_cast<float*>(lse.data_ptr());
  }

  int* table_base = static_cast<int*>(args.block_tables.data_ptr());
  int* table_k = table_base;
  int* table_v = table_base;
  int64_t table_k_s0 = args.block_tables.stride(0);
  int64_t table_v_s0 = table_k_s0;
  if (!args.uses_shared_paged_kv_idx) {
    TVM_FFI_ICHECK_EQ(args.block_tables.ndim(), 3)
        << "separate K/V page tables must have shape [batch, 2, pages]";
    TVM_FFI_ICHECK_EQ(args.block_tables.size(1), 2);
    table_v = table_base + args.block_tables.stride(1);
  } else {
    TVM_FFI_ICHECK_EQ(args.block_tables.ndim(), 2)
        << "shared page tables must have shape [batch, pages]";
  }

  ffi::CUDADeviceGuard device_guard(args.query.device().device_id);
  cudaStream_t stream = get_stream(args.query.device());
  int* q_indptr_ptr = nullptr;
  if (args.q_indptr.has_value()) {
    auto const& q_indptr = args.q_indptr.value();
    TVM_FFI_ICHECK_EQ(q_indptr.dtype(), dl_int32);
    TVM_FFI_ICHECK_EQ(q_indptr.ndim(), 1);
    TVM_FFI_ICHECK_EQ(q_indptr.size(0), args.batch_size + 1);
    TVM_FFI_ICHECK(q_indptr.IsContiguous());
    q_indptr_ptr = static_cast<int*>(q_indptr.data_ptr());
  } else {
    TVM_FFI_ICHECK_GT(args.uniform_q_len, 0);
    TVM_FFI_ICHECK_EQ(args.query.size(0), args.batch_size * args.uniform_q_len)
        << "uniform decode query rows must equal batch_size * q_len";
    int64_t required_bytes = (args.batch_size + 1) * static_cast<int64_t>(sizeof(int));
    int64_t workspace_bytes = args.workspace_buffer.numel() * get_element_size(args.workspace_buffer);
    TVM_FFI_ICHECK_GE(workspace_bytes, required_bytes)
        << "workspace is too small for Cake FMHA query indptr";
    q_indptr_ptr = static_cast<int*>(args.workspace_buffer.data_ptr());
    int const threads = 128;
    int const blocks = static_cast<int>((args.batch_size + 1 + threads - 1) / threads);
    FillUniformQIndptr<<<blocks, threads, 0, stream>>>(
        q_indptr_ptr, static_cast<int>(args.batch_size), static_cast<int>(args.uniform_q_len));
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess)
        << "failed to initialize Cake FMHA uniform query indptr";
  }

  cudaError_t status = cake_fmha_launch_compat_v1(
      args.query.data_ptr(), args.key_cache.data_ptr(), args.value_cache.data_ptr(), k_scales_ptr,
      v_scales_ptr, args.out.data_ptr(), output_scales_ptr, lse_ptr, table_k, table_v,
      q_indptr_ptr, static_cast<int*>(args.seq_lens.data_ptr()), sinks_ptr,
      static_cast<int>(args.batch_size), num_q_heads, num_kv_heads, head_dim, page_size, kHndLayout,
      q_dtype, kv_dtype, o_dtype, static_cast<int>(args.is_causal),
      static_cast<int>(args.window_left), static_cast<int>(sinks_ptr != nullptr),
      static_cast<int>(lse_ptr != nullptr), 1.0f, 1.0f, static_cast<float>(args.bmm2_scale), 1.0f,
      static_cast<float>(args.bmm1_scale),
      o_dtype == kNvFp4 ? static_cast<float>(args.o_sf_scale) : 1.0f,
      static_cast<int>(args.o_sf_start_index), output_scale_columns, args.query.stride(0),
      args.query.stride(1), args.key_cache.stride(0), args.key_cache.stride(1),
      args.key_cache.stride(2), args.key_cache.stride(3), args.value_cache.stride(0),
      args.value_cache.stride(1), args.value_cache.stride(2), args.value_cache.stride(3), ksf_s0,
      ksf_s1, ksf_s2, ksf_s3, vsf_s0, vsf_s1, vsf_s2, vsf_s3, table_k_s0, table_v_s0,
      args.out.stride(0), args.out.stride(1), static_cast<unsigned int>(args.query.size(0)),
      static_cast<unsigned int>(num_q_heads), 1, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "Cake FMHA compatibility launch failed: " << cudaGetErrorString(status);
}

}  // namespace

void cake_paged_attention_decode(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl,
    int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> cum_seq_lens_q, Optional<TensorView> key_block_scales,
    Optional<TensorView> value_block_scales, Optional<float> skip_softmax_threshold_scale_factor,
    Optional<bool> uses_shared_paged_kv_idx, Optional<TensorView> lse, int64_t lse_stride_tokens,
    int64_t lse_stride_heads, bool enable_block_sparse_attention,
    Optional<TensorView> sparse_mla_top_k_lens) {
  TVM_FFI_ICHECK_EQ(sparse_mla_top_k, 0) << "Cake FMHA does not implement sparse MLA";
  TVM_FFI_ICHECK(!enable_block_sparse_attention)
      << "Cake FMHA does not implement block-sparse attention";
  TVM_FFI_ICHECK(!sparse_mla_top_k_lens.has_value())
      << "Cake FMHA does not implement sparse MLA lengths";
  TVM_FFI_ICHECK(o_sf_vec_size == -1 || o_sf_vec_size == 16)
      << "Cake FMHA NVFP4 output requires o_sf_vec_size=16";
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f)
      << "Cake FMHA does not implement skip-softmax approximation";
  (void)max_kv_len;
  (void)sm_count;
  (void)enable_pdl;
  (void)workspace_size;
  (void)multi_ctas_kv_counter_buffer;
  LaunchCompat(CommonArgs{
      out,
      out_scale_factor,
      query,
      key_cache,
      value_cache,
      workspace_buffer,
      block_tables,
      seq_lens,
      cum_seq_lens_q,
      attention_sinks,
      key_block_scales,
      value_block_scales,
      lse,
      batch_size,
      max_q_len,
      window_left,
      true,
      uses_shared_paged_kv_idx.value_or(true),
      ScalarScale(bmm1_scale, "bmm1_scale"),
      ScalarScale(bmm2_scale, "bmm2_scale"),
      o_sf_scale,
      o_sf_start_index,
      lse_stride_tokens,
      lse_stride_heads,
  });
}

void cake_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
    TensorView block_tables, TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, Optional<TensorView> attention_sinks,
    Optional<TensorView> key_block_scales, Optional<TensorView> value_block_scales,
    Optional<float> skip_softmax_threshold_scale_factor, Optional<bool> uses_shared_paged_kv_idx,
    Optional<bool> use_fp16_softmax, Optional<bool> uses_spcompress, bool is_causal,
    Optional<TensorView> lse, int64_t lse_stride_tokens, int64_t lse_stride_heads) {
  TVM_FFI_ICHECK(o_sf_vec_size == -1 || o_sf_vec_size == 16)
      << "Cake FMHA NVFP4 output requires o_sf_vec_size=16";
  TVM_FFI_ICHECK_EQ(skip_softmax_threshold_scale_factor.value_or(0.0f), 0.0f)
      << "Cake FMHA does not implement skip-softmax approximation";
  TVM_FFI_ICHECK(!use_fp16_softmax.value_or(false));
  TVM_FFI_ICHECK(!uses_spcompress.value_or(false));
  (void)max_q_len;
  (void)max_kv_len;
  (void)cum_seq_lens_kv;
  (void)sm_count;
  (void)enable_pdl;
  (void)workspace_size;
  (void)multi_ctas_kv_counter_buffer;
  LaunchCompat(CommonArgs{
      out,
      out_scale_factor,
      query,
      key_cache,
      value_cache,
      workspace_buffer,
      block_tables,
      seq_lens,
      cum_seq_lens_q,
      attention_sinks,
      key_block_scales,
      value_block_scales,
      lse,
      batch_size,
      0,
      window_left,
      is_causal,
      uses_shared_paged_kv_idx.value_or(true),
      ScalarScale(bmm1_scale, "bmm1_scale"),
      ScalarScale(bmm2_scale, "bmm2_scale"),
      o_sf_scale,
      o_sf_start_index,
      lse_stride_tokens,
      lse_stride_heads,
  });
}

}  // namespace cake_fmha
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_decode,
                              flashinfer::cake_fmha::cake_paged_attention_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_paged_attention_context,
                              flashinfer::cake_fmha::cake_paged_attention_context);
