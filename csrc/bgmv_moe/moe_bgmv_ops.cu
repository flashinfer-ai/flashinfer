/*
 * Dispatch logic for BGMV MoE kernels.
 * Routes to the correct template instantiation based on tensor dtypes and dimensions.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "moe_bgmv_config.h"
#include "tvm_ffi_utils.h"

// ====== Utils ======

inline constexpr uint64_t pack_u32(uint32_t a, uint32_t b) {
  return (uint64_t(a) << 32) | uint64_t(b);
}

// ====== MoE BGMV Shrink Launcher ======

template <typename T>
inline bool launch_moe_shrink_sliced_kernel(T* Y, const T* X, T** w_ptr,
                                            const int64_t* sorted_token_ids,
                                            const int64_t* expert_ids, const int64_t* lora_indices,
                                            uint32_t feat_in, uint32_t feat_out, int64_t num_pairs,
                                            int64_t num_slices, int64_t num_experts,
                                            int64_t num_tokens, int64_t lora_stride) {
  switch (pack_u32(feat_in, feat_out)) {
#define CASE_MOE_SHRINK(in_T, out_T, W_T, narrow, wide)                                 \
  case pack_u32(wide, narrow):                                                          \
    moe_bgmv_shrink_sliced<wide, narrow, in_T, out_T, W_T>(                             \
        Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, num_pairs, num_slices, \
        num_experts, num_tokens, lora_stride, 1.0f);                                    \
    return true;
    FOR_MOE_ALL_WIDE_NARROW(CASE_MOE_SHRINK, T, T, T)
#undef CASE_MOE_SHRINK
    default:
      return false;
  }
}

// ====== MoE BGMV Expand Launcher ======

template <typename T>
inline bool launch_moe_expand_sliced_kernel(
    float* Y, const T* X, T** w_ptr, const int64_t* sorted_token_ids, const int64_t* expert_ids,
    const int64_t* lora_indices, const float* topk_weights, const int64_t* slice_start_loc,
    uint32_t feat_in, uint32_t feat_out, int64_t num_pairs, int64_t num_slices, int64_t num_experts,
    int64_t total_feat_out, int64_t num_tokens, int64_t lora_stride) {
  switch (pack_u32(feat_in, feat_out)) {
#define CASE_MOE_EXPAND(in_T, out_T, W_T, narrow, wide)                                           \
  case pack_u32(narrow, wide):                                                                    \
    moe_bgmv_expand_sliced<narrow, wide, in_T, W_T>(                                              \
        Y, X, w_ptr, sorted_token_ids, expert_ids, lora_indices, topk_weights, slice_start_loc,   \
        num_pairs, num_slices, num_experts, total_feat_out, wide, num_tokens, lora_stride, 1.0f); \
    return true;
    FOR_MOE_ALL_WIDE_NARROW(CASE_MOE_EXPAND, T, T, T)
#undef CASE_MOE_EXPAND
    default:
      return false;
  }
}

// ====== TVM-FFI dispatch: MoE Shrink ======

void bgmv_moe_shrink(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView lora_indices, int64_t lora_stride) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(sorted_token_ids);
  CHECK_INPUT(expert_ids);
  CHECK_INPUT(lora_indices);
  CHECK_DIM(3, y);
  CHECK_DIM(2, x);
  CHECK_DIM(2, w_ptr);
  CHECK_DIM(1, sorted_token_ids);
  CHECK_DIM(1, expert_ids);
  CHECK_DIM(1, lora_indices);

  int64_t num_slices = y.size(0);
  int64_t num_pairs = sorted_token_ids.size(0);
  int64_t num_tokens = lora_indices.size(0);
  int64_t feat_in = x.size(1);
  int64_t feat_out = y.size(2);
  int64_t num_experts = w_ptr.size(1);

  TVM_FFI_ICHECK_EQ(w_ptr.size(0), num_slices) << "w_ptr slice dim mismatch";
  TVM_FFI_ICHECK(sorted_token_ids.dtype() == dl_int64) << "sorted_token_ids must be int64";
  TVM_FFI_ICHECK(expert_ids.dtype() == dl_int64) << "expert_ids must be int64";
  TVM_FFI_ICHECK(w_ptr.dtype() == dl_int64) << "w_ptr must be int64";
  TVM_FFI_ICHECK(lora_indices.dtype() == dl_int64) << "lora_indices must be int64";

  ffi::CUDADeviceGuard guard(x.device().device_id);
  bool ok = false;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), DType, [&] {
    ok = launch_moe_shrink_sliced_kernel(
        static_cast<DType*>(y.data_ptr()), static_cast<DType*>(x.data_ptr()),
        reinterpret_cast<DType**>(static_cast<int64_t*>(w_ptr.data_ptr())),
        static_cast<int64_t*>(sorted_token_ids.data_ptr()),
        static_cast<int64_t*>(expert_ids.data_ptr()),
        static_cast<int64_t*>(lora_indices.data_ptr()), feat_in, feat_out, num_pairs, num_slices,
        num_experts, num_tokens, lora_stride);
    return true;
  });

  TVM_FFI_ICHECK(ok) << "BGMV MoE shrink failed. feat_in=" << feat_in << " feat_out=" << feat_out
                     << ". Dimension pair not compiled.";
}

// ====== TVM-FFI dispatch: MoE Expand ======

void bgmv_moe_expand(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView topk_weights, TensorView lora_indices,
                     TensorView slice_start_loc, int64_t first_feat_out, int64_t lora_stride) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(sorted_token_ids);
  CHECK_INPUT(expert_ids);
  CHECK_INPUT(topk_weights);
  CHECK_INPUT(lora_indices);
  CHECK_INPUT(slice_start_loc);
  CHECK_DIM(2, y);
  CHECK_DIM(3, x);
  CHECK_DIM(2, w_ptr);
  CHECK_DIM(1, sorted_token_ids);
  CHECK_DIM(1, expert_ids);
  CHECK_DIM(1, topk_weights);
  CHECK_DIM(1, lora_indices);
  CHECK_DIM(1, slice_start_loc);

  int64_t num_slices = x.size(0);
  int64_t num_pairs = sorted_token_ids.size(0);
  int64_t num_tokens = lora_indices.size(0);
  int64_t feat_in = x.size(2);
  int64_t total_feat_out = y.size(1);
  int64_t num_experts = w_ptr.size(1);

  TVM_FFI_ICHECK_EQ(w_ptr.size(0), num_slices) << "w_ptr slice dim mismatch";
  TVM_FFI_ICHECK(sorted_token_ids.dtype() == dl_int64) << "sorted_token_ids must be int64";
  TVM_FFI_ICHECK(expert_ids.dtype() == dl_int64) << "expert_ids must be int64";
  TVM_FFI_ICHECK(w_ptr.dtype() == dl_int64) << "w_ptr must be int64";
  TVM_FFI_ICHECK(lora_indices.dtype() == dl_int64) << "lora_indices must be int64";
  TVM_FFI_ICHECK(slice_start_loc.dtype() == dl_int64) << "slice_start_loc must be int64";
  TVM_FFI_ICHECK(topk_weights.dtype() == dl_float32) << "topk_weights must be float32";
  TVM_FFI_ICHECK(y.dtype() == dl_float32) << "y must be float32 accumulation buffer";
  TVM_FFI_ICHECK(first_feat_out > 0) << "first_feat_out must be positive";

  ffi::CUDADeviceGuard guard(x.device().device_id);
  bool ok = false;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), DType, [&] {
    ok = launch_moe_expand_sliced_kernel(
        static_cast<float*>(y.data_ptr()), static_cast<DType*>(x.data_ptr()),
        reinterpret_cast<DType**>(static_cast<int64_t*>(w_ptr.data_ptr())),
        static_cast<int64_t*>(sorted_token_ids.data_ptr()),
        static_cast<int64_t*>(expert_ids.data_ptr()),
        static_cast<int64_t*>(lora_indices.data_ptr()),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int64_t*>(slice_start_loc.data_ptr()), feat_in,
        static_cast<int32_t>(first_feat_out), num_pairs, num_slices, num_experts, total_feat_out,
        num_tokens, lora_stride);
    return true;
  });

  TVM_FFI_ICHECK(ok) << "BGMV MoE expand failed. feat_in=" << feat_in
                     << " feat_out=" << first_feat_out << ". Dimension pair not compiled.";
}
