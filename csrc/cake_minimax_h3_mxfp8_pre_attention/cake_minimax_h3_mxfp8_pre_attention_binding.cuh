/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
// Thin tvm-ffi launcher shared by every MiniMax-H3 MXFP8 pre-attention module.
//
// The JIT loader writes one short translation unit per exact token count:
//
//   #define CAKE_MINIMAX_H3_MXFP8_STAGE 1      (RMSNorm + AdaLN + MXFP8 quantize)
//   #define CAKE_MINIMAX_H3_MXFP8_M 4824
//   #include "cake_minimax_h3_mxfp8_pre_attention_binding.cuh"
//
//   #define CAKE_MINIMAX_H3_MXFP8_STAGE 2      (Q/K RMSNorm + RoPE + destination pack)
//   #define CAKE_MINIMAX_H3_MXFP8_M 4824
//   #define CAKE_MINIMAX_H3_MXFP8_P 8
//   #define CAKE_MINIMAX_H3_MXFP8_HEADS_PER_DESTINATION 7
//   #define CAKE_MINIMAX_H3_MXFP8_ROWS_PER_DESTINATION 101304
//   #define CAKE_MINIMAX_H3_MXFP8_SCALE_STRIDE 405504
//   #include "cake_minimax_h3_mxfp8_pre_attention_binding.cuh"
//
// The generated device source is included unchanged. The short schedule macros
// it reads (M, P, HEADS_PER_DESTINATION, ROWS_PER_DESTINATION, SCALE_STRIDE)
// are bound to the values above only around that include, so the token stream
// the compiler sees is exactly the `#define` block the per-token-count copies
// used to carry.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#if !defined(CAKE_MINIMAX_H3_MXFP8_STAGE) || !defined(CAKE_MINIMAX_H3_MXFP8_M)
#error "CAKE_MINIMAX_H3_MXFP8_STAGE (1 or 2) and CAKE_MINIMAX_H3_MXFP8_M must be defined"
#endif

#define M CAKE_MINIMAX_H3_MXFP8_M
#if CAKE_MINIMAX_H3_MXFP8_STAGE == 1
#include "cake_minimax_h3_mxfp8_norm_adaln_quantize_device.cu"
#elif CAKE_MINIMAX_H3_MXFP8_STAGE == 2
#if !defined(CAKE_MINIMAX_H3_MXFP8_P) || !defined(CAKE_MINIMAX_H3_MXFP8_HEADS_PER_DESTINATION) || \
    !defined(CAKE_MINIMAX_H3_MXFP8_ROWS_PER_DESTINATION) ||                                     \
    !defined(CAKE_MINIMAX_H3_MXFP8_SCALE_STRIDE)
#error "stage 2 needs P, HEADS_PER_DESTINATION, ROWS_PER_DESTINATION and SCALE_STRIDE"
#endif
#define P CAKE_MINIMAX_H3_MXFP8_P
#define HEADS_PER_DESTINATION CAKE_MINIMAX_H3_MXFP8_HEADS_PER_DESTINATION
#define ROWS_PER_DESTINATION CAKE_MINIMAX_H3_MXFP8_ROWS_PER_DESTINATION
#define SCALE_STRIDE CAKE_MINIMAX_H3_MXFP8_SCALE_STRIDE
#include "cake_minimax_h3_mxfp8_qk_rope_destination_pack_device.cu"
#else
#error "CAKE_MINIMAX_H3_MXFP8_STAGE must be 1 or 2"
#endif

namespace flashinfer::cake_minimax_h3_mxfp8_pre_attention {

constexpr int64_t kM = M;
constexpr int64_t kHidden = 5376;
constexpr int64_t kHeads = 56;
constexpr int64_t kKinds = 3;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kRopeWidth = 96;
constexpr int64_t kAdalnRows = 9;
constexpr int64_t kScaleBlock = 32;
constexpr int64_t kScaleTileRows = 128;
constexpr uint32_t kThreads = THREADS;
constexpr int64_t kPaddedM = (kM + kScaleTileRows - 1) / kScaleTileRows * kScaleTileRows;
#if CAKE_MINIMAX_H3_MXFP8_STAGE == 1
constexpr uint32_t kDynamicSmemBytes = SMEM_TOTAL;
constexpr int64_t kGridX = kM;
constexpr int64_t kActivationScaleBytes = kPaddedM * (kHidden / kScaleBlock);
#else
constexpr int64_t kP = P;
constexpr int64_t kHeadsPerDestination = HEADS_PER_DESTINATION;
constexpr int64_t kRowsPerDestination = ROWS_PER_DESTINATION;
constexpr int64_t kScaleStride = SCALE_STRIDE;
constexpr int64_t kRowsPerBlock = 8;
constexpr uint32_t kDynamicSmemBytes = 0;
constexpr int64_t kGridX = (kP * kRowsPerDestination + kRowsPerBlock - 1) / kRowsPerBlock;
static_assert(kHeadsPerDestination * kP == kHeads, "P must divide the head count");
static_assert(kRowsPerDestination == kM * kHeadsPerDestination * kKinds,
              "ROWS_PER_DESTINATION must equal M * (56 / P) * 3");
static_assert(kScaleStride == (kRowsPerDestination + kScaleTileRows - 1) / kScaleTileRows *
                                  kScaleTileRows * (kHeadDim / kScaleBlock),
              "SCALE_STRIDE must equal round_up(ROWS_PER_DESTINATION, 128) * 4");
#endif
static_assert(kM > 0, "M must be positive");

}  // namespace flashinfer::cake_minimax_h3_mxfp8_pre_attention

// The generated source's private macros end with the device body.
#undef M
#undef P
#undef HEADS_PER_DESTINATION
#undef ROWS_PER_DESTINATION
#undef SCALE_STRIDE
#undef THREADS
#undef CAKE_INF
#undef HIDDEN
#undef SCALE_COLS
#undef NUM_MAIN_STAGES
#undef SMEM_TOTAL
#undef SMEM_SMEM_RSTD_OFF
#undef SMEM_SMEM_RSTD_STAGE_BYTES
#undef SMEM_SMEM_RSTD_STRIDE
#undef SMEM_SMEM_INDEX_OFF
#undef SMEM_SMEM_INDEX_STAGE_BYTES
#undef SMEM_SMEM_INDEX_STRIDE
#undef SMEM_SMEM_PARTIAL_OFF
#undef SMEM_SMEM_PARTIAL_STAGE_BYTES
#undef SMEM_SMEM_PARTIAL_STRIDE

// Exact extent check on top of the target's CHECK_DIM.
#define CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(t, axis, expected) \
  TVM_FFI_ICHECK_EQ(t.size(axis), expected) << #t " dim " #axis " must be " << expected;

namespace flashinfer::cake_minimax_h3_mxfp8_pre_attention {

inline void CheckLaunch(cudaError_t status) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "MiniMax-H3 MXFP8 pre-attention launch failed: " << cudaGetErrorString(status);
}

#if CAKE_MINIMAX_H3_MXFP8_STAGE == 1

void Run(TensorView x, TensorView x_norm_weight, TensorView adaln_scale, TensorView adaln_shift,
         TensorView adaln_index, TensorView activation_q, TensorView activation_sf, double eps) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(x_norm_weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(adaln_scale, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(adaln_shift, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(adaln_index, dl_int32);
  CHECK_INPUT_AND_TYPE(activation_q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(activation_sf, dl_uint8);
  CHECK_DEVICE(x, x_norm_weight);
  CHECK_DEVICE(x, adaln_scale);
  CHECK_DEVICE(x, adaln_shift);
  CHECK_DEVICE(x, adaln_index);
  CHECK_DEVICE(x, activation_q);
  CHECK_DEVICE(x, activation_sf);
  CHECK_DIM(2, x);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(x, 0, kM);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(x, 1, kHidden);
  CHECK_DIM(1, x_norm_weight);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(x_norm_weight, 0, kHidden);
  CHECK_DIM(2, adaln_scale);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(adaln_scale, 0, kAdalnRows);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(adaln_scale, 1, kHidden);
  CHECK_DIM(2, adaln_shift);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(adaln_shift, 0, kAdalnRows);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(adaln_shift, 1, kHidden);
  CHECK_DIM(1, adaln_index);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(adaln_index, 0, kM);
  CHECK_DIM(2, activation_q);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(activation_q, 0, kM);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(activation_q, 1, kHidden);
  CHECK_DIM(1, activation_sf);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(activation_sf, 0, kActivationScaleBytes);

  ffi::CUDADeviceGuard device_guard(x.device().device_id);
  cudaStream_t stream = get_stream(x.device());
  kernel_cake_minimax_h3_mxfp8_norm_adaln_quantize<<<dim3(static_cast<uint32_t>(kGridX)),
                                                     dim3(kThreads), kDynamicSmemBytes, stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()), static_cast<int*>(adaln_index.data_ptr()),
      static_cast<uint8_t*>(activation_q.data_ptr()), static_cast<uint8_t*>(activation_sf.data_ptr()),
      static_cast<float>(eps));
  CheckLaunch(cudaGetLastError());
}

#else

void Run(TensorView qkv_bf16, TensorView q_norm_weight, TensorView k_norm_weight,
         TensorView rope_cos_sin, TensorView out_q, TensorView out_sf, TensorView debug_q_bf16,
         TensorView debug_k_bf16, int64_t write_debug, double eps) {
  CHECK_INPUT_AND_TYPE(qkv_bf16, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(q_norm_weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(k_norm_weight, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(rope_cos_sin, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out_q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out_sf, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_q_bf16, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(debug_k_bf16, dl_bfloat16);
  CHECK_DEVICE(qkv_bf16, q_norm_weight);
  CHECK_DEVICE(qkv_bf16, k_norm_weight);
  CHECK_DEVICE(qkv_bf16, rope_cos_sin);
  CHECK_DEVICE(qkv_bf16, out_q);
  CHECK_DEVICE(qkv_bf16, out_sf);
  CHECK_DEVICE(qkv_bf16, debug_q_bf16);
  CHECK_DEVICE(qkv_bf16, debug_k_bf16);
  CHECK_DIM(2, qkv_bf16);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(qkv_bf16, 0, kM);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(qkv_bf16, 1, kHeads * kKinds * kHeadDim);
  CHECK_DIM(1, q_norm_weight);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(q_norm_weight, 0, kHeadDim);
  CHECK_DIM(1, k_norm_weight);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(k_norm_weight, 0, kHeadDim);
  CHECK_DIM(2, rope_cos_sin);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(rope_cos_sin, 0, kM);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(rope_cos_sin, 1, kRopeWidth);
  CHECK_DIM(5, out_q);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_q, 0, kP);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_q, 1, kM);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_q, 2, kHeadsPerDestination);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_q, 3, kKinds);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_q, 4, kHeadDim);
  CHECK_DIM(2, out_sf);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_sf, 0, kP);
  CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(out_sf, 1, kScaleStride);
  if (write_debug != 0) {
    CHECK_DIM(3, debug_q_bf16);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_q_bf16, 0, kM);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_q_bf16, 1, kHeads);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_q_bf16, 2, kHeadDim);
    CHECK_DIM(3, debug_k_bf16);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_k_bf16, 0, kM);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_k_bf16, 1, kHeads);
    CAKE_MINIMAX_H3_MXFP8_CHECK_SIZE(debug_k_bf16, 2, kHeadDim);
  }

  ffi::CUDADeviceGuard device_guard(qkv_bf16.device().device_id);
  cudaStream_t stream = get_stream(qkv_bf16.device());
  kernel_cake_minimax_h3_mxfp8_qk_rope_destination_pack<<<dim3(static_cast<uint32_t>(kGridX)),
                                                          dim3(kThreads), kDynamicSmemBytes,
                                                          stream>>>(
      static_cast<__nv_bfloat16*>(qkv_bf16.data_ptr()),
      static_cast<__nv_bfloat16*>(q_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(k_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(rope_cos_sin.data_ptr()), static_cast<uint8_t*>(out_q.data_ptr()),
      static_cast<uint8_t*>(out_sf.data_ptr()), static_cast<__nv_bfloat16*>(debug_q_bf16.data_ptr()),
      static_cast<__nv_bfloat16*>(debug_k_bf16.data_ptr()), static_cast<int>(write_debug),
      static_cast<float>(eps));
  CheckLaunch(cudaGetLastError());
}

#endif

}  // namespace flashinfer::cake_minimax_h3_mxfp8_pre_attention

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_minimax_h3_mxfp8_pre_attention::Run);
