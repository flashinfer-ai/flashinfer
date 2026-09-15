// Copyright (C) 2026 Tencent.
// SPDX-License-Identifier: MIT

#include <flashinfer/rope_norm_store_kv_hy3.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Tensor;

// This binding promotes the model-agnostic behavior of Tencent/HPC-Ops' fused
// Q/K RMSNorm + NeoX RoPE + paged-KV-store operator into FlashInfer. The
// source-faithful specialization is the fallback for every supported shape;
// the narrower SM100 uniform-decode path fuses final-page tail clearing, maps
// one row directly to one request, and reuses RoPE coefficients from registers.
void qk_rmsnorm_rope_append_paged_kv_cache_hy3(
    TensorView packed_qkv, TensorView cos_sin_cache, TensorView sequence_lengths,
    TensorView q_indptr, TensorView block_table, TensorView q_norm_weight, TensorView k_norm_weight,
    TensorView k_scale, TensorView v_scale, TensorView q_scale_inverse, TensorView output_q,
    TensorView output_q_scale, TensorView split_k_flag, TensorView output_k, TensorView output_v,
    TensorView key_cache, TensorView value_cache, bool is_prefill, int64_t norm_policy,
    int64_t quant_policy, int64_t max_sequence_length, double fp8_upper_bound, bool use_output_k,
    bool use_output_v, bool enable_sm100_uniform_decode) {
  CHECK_INPUT(packed_qkv);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(sequence_lengths);
  CHECK_INPUT(q_indptr);
  CHECK_INPUT(block_table);
  CHECK_INPUT(q_norm_weight);
  CHECK_INPUT(k_norm_weight);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);
  CHECK_INPUT(q_scale_inverse);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_q_scale);
  CHECK_INPUT(split_k_flag);
  CHECK_INPUT(output_k);
  CHECK_INPUT(output_v);
  CHECK_INPUT(key_cache);
  CHECK_INPUT(value_cache);

  CHECK_DIM(2, packed_qkv);
  CHECK_DIM(2, cos_sin_cache);
  CHECK_DIM(1, sequence_lengths);
  CHECK_DIM(1, q_indptr);
  CHECK_DIM(2, block_table);
  CHECK_DIM(3, output_q);
  CHECK_DIM(4, key_cache);
  CHECK_DIM(4, value_cache);

  const auto check_same_device = [&](const TensorView& tensor) {
    TVM_FFI_ICHECK_EQ(packed_qkv.device().device_type, tensor.device().device_type);
    TVM_FFI_ICHECK_EQ(packed_qkv.device().device_id, tensor.device().device_id);
  };
  for (const TensorView* tensor :
       {&cos_sin_cache, &sequence_lengths, &q_indptr, &block_table, &q_norm_weight, &k_norm_weight,
        &k_scale, &v_scale, &q_scale_inverse, &output_q, &output_q_scale, &split_k_flag, &output_k,
        &output_v, &key_cache, &value_cache}) {
    check_same_device(*tensor);
  }

  TVM_FFI_ICHECK_EQ(packed_qkv.dtype(), dl_bfloat16) << "packed_qkv must have bfloat16 dtype";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.dtype(), dl_float32) << "cos_sin_cache must have float32 dtype";
  TVM_FFI_ICHECK_EQ(sequence_lengths.dtype(), dl_int32) << "sequence_lengths must have int32 dtype";
  TVM_FFI_ICHECK_EQ(q_indptr.dtype(), dl_int32) << "q_indptr must have int32 dtype";
  TVM_FFI_ICHECK_EQ(block_table.dtype(), dl_int32) << "block_table must have int32 dtype";

  const int64_t batch_size = sequence_lengths.size(0);
  const int64_t num_rows = packed_qkv.size(0);
  const int64_t num_kv_heads = key_cache.size(2);
  const int64_t qk_head_dim = key_cache.size(3);
  const int64_t v_head_dim = value_cache.size(3);
  const int64_t page_size = key_cache.size(1);
  const int64_t hidden_size = packed_qkv.size(1);
  TVM_FFI_ICHECK_GT(batch_size, 0) << "batch_size must be positive";
  TVM_FFI_ICHECK_GT(num_rows, 0) << "packed_qkv must contain at least one row";
  TVM_FFI_ICHECK_GT(page_size, 0) << "page_size must be positive";
  TVM_FFI_ICHECK_EQ(q_indptr.size(0), batch_size + 1)
      << "q_indptr must have shape [batch_size + 1]";
  TVM_FFI_ICHECK_EQ(block_table.size(0), batch_size)
      << "block_table must have shape [batch_size, max_pages_per_request]";
  TVM_FFI_ICHECK_EQ(key_cache.size(0), value_cache.size(0))
      << "key/value cache page counts must match";
  TVM_FFI_ICHECK_EQ(key_cache.size(1), value_cache.size(1))
      << "key/value cache page sizes must match";
  TVM_FFI_ICHECK_EQ(key_cache.size(2), value_cache.size(2))
      << "key/value cache KV-head counts must match";
  TVM_FFI_ICHECK_EQ(qk_head_dim, 128) << "qk_head_dim must be 128";
  TVM_FFI_ICHECK_EQ(v_head_dim, 128) << "v_head_dim must be 128";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), qk_head_dim)
      << "cos_sin_cache must have shape [max_position, 128]";

  const int64_t q_width = hidden_size - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim;
  TVM_FFI_ICHECK_GE(q_width, 0) << "packed_qkv hidden dimension is too small";
  TVM_FFI_ICHECK_EQ(q_width % qk_head_dim, 0)
      << "packed_qkv hidden dimension is incompatible with cache shapes";
  const int64_t num_q_heads = q_width / qk_head_dim;
  TVM_FFI_ICHECK((num_q_heads == 8 && num_kv_heads == 1) ||
                 (num_q_heads == 64 && num_kv_heads == 8))
      << "supported (num_q_heads, num_kv_heads) pairs are (8, 1) and (64, 8)";
  TVM_FFI_ICHECK_GE(norm_policy, 0);
  TVM_FFI_ICHECK_LE(norm_policy, 2)
      << "norm_policy must be 0 (none), 1 (RoPE then norm), or 2 (norm then RoPE)";

  TVM_FFI_ICHECK_EQ(output_q.size(0), num_rows);
  TVM_FFI_ICHECK_EQ(output_q.size(1), num_q_heads);
  TVM_FFI_ICHECK_EQ(output_q.size(2), qk_head_dim);

  if (norm_policy > 0) {
    // The upstream launcher passes const float* weights; keeping FP32 here is
    // intentional rather than an omitted BF16 dispatch.
    CHECK_DIM(1, q_norm_weight);
    CHECK_DIM(1, k_norm_weight);
    TVM_FFI_ICHECK_EQ(q_norm_weight.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(k_norm_weight.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(q_norm_weight.numel(), qk_head_dim);
    TVM_FFI_ICHECK_EQ(k_norm_weight.numel(), qk_head_dim);
  } else {
    TVM_FFI_ICHECK_EQ(q_norm_weight.numel(), 0)
        << "q_norm_weight must be empty when norm_policy is 0";
    TVM_FFI_ICHECK_EQ(k_norm_weight.numel(), 0)
        << "k_norm_weight must be empty when norm_policy is 0";
  }

  const bool is_fp8 = key_cache.dtype() == dl_float8_e4m3fn;
  TVM_FFI_ICHECK(value_cache.dtype() == key_cache.dtype()) << "key/value cache dtypes must match";
  TVM_FFI_ICHECK(output_q.dtype() == key_cache.dtype())
      << "output_q dtype must match the cache dtype";
  if (is_fp8) {
    TVM_FFI_ICHECK(quant_policy == 1 || quant_policy == 2)
        << "FP8 quant_policy must be 1 (dynamic Q) or 2 (static Q)";
    TVM_FFI_ICHECK_GT(fp8_upper_bound, 0.0);
    TVM_FFI_ICHECK_LE(fp8_upper_bound, 448.0) << "fp8_upper_bound must be in (0, 448] for E4M3FN";
    TVM_FFI_ICHECK_EQ(k_scale.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(v_scale.dtype(), dl_float32);
    TVM_FFI_ICHECK_EQ(k_scale.numel(), 1);
    TVM_FFI_ICHECK_EQ(v_scale.numel(), 1);
    if (quant_policy == 2) {
      TVM_FFI_ICHECK_EQ(q_scale_inverse.dtype(), dl_float32);
      TVM_FFI_ICHECK_EQ(q_scale_inverse.numel(), 1);
      TVM_FFI_ICHECK_EQ(output_q_scale.numel(), 0);
    } else {
      TVM_FFI_ICHECK_EQ(q_scale_inverse.numel(), 0)
          << "q_scale_inverse must be empty for dynamic-Q quantization";
      TVM_FFI_ICHECK_GE(max_sequence_length, 0);
      const int64_t max_sequence_length_aligned = ((max_sequence_length + 127) / 128) * 128;
      TVM_FFI_ICHECK_EQ(output_q_scale.dtype(), dl_float32);
      if (is_prefill) {
        TVM_FFI_ICHECK_GT(max_sequence_length, 0)
            << "dynamic-Q prefill requires max_sequence_length > 0";
        CHECK_DIM(3, output_q_scale);
        TVM_FFI_ICHECK_EQ(output_q_scale.size(0), batch_size);
        TVM_FFI_ICHECK_EQ(output_q_scale.size(1), num_q_heads);
        TVM_FFI_ICHECK_EQ(output_q_scale.size(2), max_sequence_length_aligned);
      } else {
        CHECK_DIM(2, output_q_scale);
        TVM_FFI_ICHECK_EQ(output_q_scale.size(0), num_rows);
        TVM_FFI_ICHECK_EQ(output_q_scale.size(1), num_q_heads);
      }
    }
    CHECK_DIM(2, split_k_flag);
    TVM_FFI_ICHECK_EQ(split_k_flag.dtype(), dl_int32);
    TVM_FFI_ICHECK_EQ(split_k_flag.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(split_k_flag.size(1), num_kv_heads);
  } else {
    TVM_FFI_ICHECK_EQ(key_cache.dtype(), dl_bfloat16)
        << "cache dtype must be bfloat16 or float8_e4m3fn";
    TVM_FFI_ICHECK_EQ(quant_policy, 0) << "BF16 path requires quant_policy=0";
    TVM_FFI_ICHECK_EQ(k_scale.numel(), 0);
    TVM_FFI_ICHECK_EQ(v_scale.numel(), 0);
    TVM_FFI_ICHECK_EQ(q_scale_inverse.numel(), 0);
    TVM_FFI_ICHECK_EQ(output_q_scale.numel(), 0);
    TVM_FFI_ICHECK_EQ(split_k_flag.numel(), 0);
  }

  const auto check_optional_output = [&](TensorView output, bool enabled, int64_t head_count,
                                         int64_t head_dimension, const char* name) {
    if (!enabled) {
      TVM_FFI_ICHECK_EQ(output.numel(), 0) << name << " must be empty when disabled";
      return;
    }
    TVM_FFI_ICHECK_EQ(output.ndim(), 3) << name << " must be rank 3";
    TVM_FFI_ICHECK_EQ(output.dtype(), key_cache.dtype())
        << name << " dtype must match the cache dtype";
    TVM_FFI_ICHECK_EQ(output.size(0), num_rows);
    TVM_FFI_ICHECK_EQ(output.size(1), head_count);
    TVM_FFI_ICHECK_EQ(output.size(2), head_dimension);
  };
  check_optional_output(output_k, use_output_k, num_kv_heads, qk_head_dim, "output_k");
  check_optional_output(output_v, use_output_v, num_kv_heads, v_head_dim, "output_v");

  // The optimized kernel omits all clear-only CTAs. Its contract therefore
  // requires a trusted host-side guarantee that q_indptr is exactly
  // [0, 1, ..., batch_size]. Shape equality alone is not sufficient for
  // ragged or CUDA-graph-padded batches, so this remains an explicit flag.
  const bool use_sm100_fast_path = enable_sm100_uniform_decode && !is_prefill && is_fp8 &&
                                   quant_policy == 1 && norm_policy == 2 && num_q_heads == 64 &&
                                   num_kv_heads == 8 && batch_size >= 256 && num_rows == batch_size;

  ffi::CUDADeviceGuard device_guard(packed_qkv.device().device_id);
  const cudaStream_t stream = get_stream(packed_qkv.device());
  const int max_sequence_length_aligned =
      quant_policy == 1 ? static_cast<int>(((max_sequence_length + 127) / 128) * 128) : 0;

  namespace fused_rope = flashinfer::rope_norm_store_kv_hy3;
#define FLASHINFER_LAUNCH_QK_NORM_ROPE(CACHE_TYPE, QUANT_POLICY)                                  \
  fused_rope::dispatch_shape<CACHE_TYPE, QUANT_POLICY>(                                           \
      static_cast<CACHE_TYPE*>(output_q.data_ptr()),                                              \
      static_cast<CACHE_TYPE*>(key_cache.data_ptr()),                                             \
      static_cast<CACHE_TYPE*>(value_cache.data_ptr()),                                           \
      use_output_k ? static_cast<CACHE_TYPE*>(output_k.data_ptr()) : nullptr,                     \
      use_output_v ? static_cast<CACHE_TYPE*>(output_v.data_ptr()) : nullptr,                     \
      is_fp8 ? static_cast<int32_t*>(split_k_flag.data_ptr()) : nullptr,                          \
      quant_policy == 1 ? static_cast<float*>(output_q_scale.data_ptr()) : nullptr,               \
      static_cast<const __nv_bfloat16*>(packed_qkv.data_ptr()),                                   \
      static_cast<const float*>(cos_sin_cache.data_ptr()),                                        \
      static_cast<const int32_t*>(sequence_lengths.data_ptr()),                                   \
      static_cast<const int32_t*>(q_indptr.data_ptr()),                                           \
      static_cast<const int32_t*>(block_table.data_ptr()),                                        \
      norm_policy > 0 ? static_cast<const float*>(q_norm_weight.data_ptr()) : nullptr,            \
      norm_policy > 0 ? static_cast<const float*>(k_norm_weight.data_ptr()) : nullptr,            \
      is_fp8 ? static_cast<const float*>(k_scale.data_ptr()) : nullptr,                           \
      is_fp8 ? static_cast<const float*>(v_scale.data_ptr()) : nullptr,                           \
      quant_policy == 2 ? static_cast<const float*>(q_scale_inverse.data_ptr()) : nullptr,        \
      static_cast<float>(fp8_upper_bound), max_sequence_length_aligned, key_cache.stride(0),      \
      value_cache.stride(0), static_cast<int>(batch_size), static_cast<int>(block_table.size(1)), \
      static_cast<int>(page_size), static_cast<int>(num_rows), static_cast<int>(num_q_heads),     \
      static_cast<int>(num_kv_heads), is_prefill, static_cast<int>(norm_policy), stream)

  cudaError_t status;
  if (use_sm100_fast_path) {
    // Instantiate exactly the one measured fast specialization. Keeping the
    // runtime-impossible BF16/static-Q/head/norm combinations out of this
    // branch materially reduces JIT time and the generated fatbin.
    status = fused_rope::launch_specialized<__nv_fp8_e4m3, 1, 64, 8, 2, true>(
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(key_cache.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(value_cache.data_ptr()),
        use_output_k ? static_cast<__nv_fp8_e4m3*>(output_k.data_ptr()) : nullptr,
        use_output_v ? static_cast<__nv_fp8_e4m3*>(output_v.data_ptr()) : nullptr,
        static_cast<int32_t*>(split_k_flag.data_ptr()),
        static_cast<float*>(output_q_scale.data_ptr()),
        static_cast<const __nv_bfloat16*>(packed_qkv.data_ptr()),
        static_cast<const float*>(cos_sin_cache.data_ptr()),
        static_cast<const int32_t*>(sequence_lengths.data_ptr()),
        static_cast<const int32_t*>(q_indptr.data_ptr()),
        static_cast<const int32_t*>(block_table.data_ptr()),
        static_cast<const float*>(q_norm_weight.data_ptr()),
        static_cast<const float*>(k_norm_weight.data_ptr()),
        static_cast<const float*>(k_scale.data_ptr()),
        static_cast<const float*>(v_scale.data_ptr()), nullptr, static_cast<float>(fp8_upper_bound),
        max_sequence_length_aligned, key_cache.stride(0), value_cache.stride(0),
        static_cast<int>(batch_size), static_cast<int>(block_table.size(1)),
        static_cast<int>(page_size), static_cast<int>(num_rows), false, stream);
  } else if (is_fp8) {
    status = quant_policy == 1 ? FLASHINFER_LAUNCH_QK_NORM_ROPE(__nv_fp8_e4m3, 1)
                               : FLASHINFER_LAUNCH_QK_NORM_ROPE(__nv_fp8_e4m3, 2);
  } else {
    status = FLASHINFER_LAUNCH_QK_NORM_ROPE(__nv_bfloat16, 0);
  }
#undef FLASHINFER_LAUNCH_QK_NORM_ROPE

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "qk_rmsnorm_rope_append_paged_kv_cache_hy3 failed with error code "
      << cudaGetErrorString(status);
}
