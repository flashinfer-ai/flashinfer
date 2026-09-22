// Copyright (C) 2026 Tencent.
// SPDX-License-Identifier: MIT

#include "tvm_ffi_utils.h"

using tvm::ffi::Tensor;

void qk_rmsnorm_rope_append_paged_kv_cache_hy3(
    TensorView packed_qkv, TensorView cos_sin_cache, TensorView sequence_lengths,
    TensorView q_indptr, TensorView block_table, TensorView q_norm_weight, TensorView k_norm_weight,
    TensorView k_scale, TensorView v_scale, TensorView q_scale_inverse, TensorView output_q,
    TensorView output_q_scale, TensorView split_k_flag, TensorView output_k, TensorView output_v,
    TensorView key_cache, TensorView value_cache, bool is_prefill, int64_t norm_policy,
    int64_t quant_policy, int64_t max_sequence_length, double fp8_upper_bound, bool use_output_k,
    bool use_output_v, bool enable_sm100_uniform_decode);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(qk_rmsnorm_rope_append_paged_kv_cache_hy3,
                              qk_rmsnorm_rope_append_paged_kv_cache_hy3);
