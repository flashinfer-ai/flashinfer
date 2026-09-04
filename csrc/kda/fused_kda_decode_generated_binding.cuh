/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef FLASHINFER_FUSED_KDA_DECODE_BODY_FILE
#error "FLASHINFER_FUSED_KDA_DECODE_BODY_FILE must name one frozen CUDA body"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_KERNEL
#error "FLASHINFER_FUSED_KDA_DECODE_KERNEL must name the frozen CUDA kernel"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_THREADS
#error "FLASHINFER_FUSED_KDA_DECODE_THREADS must match the frozen launch contract"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES
#error "FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES must match the frozen launch contract"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS
#error "FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS must match the frozen kernel ABI"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16
#error "FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 must match the frozen state dtype"
#endif
#ifndef FLASHINFER_FUSED_KDA_DECODE_ARG_PLAN_SHA256
#error "FLASHINFER_FUSED_KDA_DECODE_ARG_PLAN_SHA256 must seal the kernel ABI"
#endif

static_assert(FLASHINFER_FUSED_KDA_DECODE_THREADS > 0 &&
              FLASHINFER_FUSED_KDA_DECODE_THREADS <= 1024 &&
              FLASHINFER_FUSED_KDA_DECODE_THREADS % 32 == 0);
static_assert(FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES > 0);
static_assert(FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS == 0 ||
              FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS == 1);
static_assert(FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 == 0 ||
              FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 == 1);
static_assert(sizeof(FLASHINFER_FUSED_KDA_DECODE_ARG_PLAN_SHA256) == 65,
              "fused KDA decode argument-plan identity must be a full SHA-256");

#include <cstdint>

// Frozen bodies own private fixed-width aliases. Keep those declarations
// separate from the CUDA and TVM-FFI headers used by this translation unit.
#define int8_t flashinfer_fused_kda_generated_int8_t
#define uint8_t flashinfer_fused_kda_generated_uint8_t
#define uint16_t flashinfer_fused_kda_generated_uint16_t
#define uint32_t flashinfer_fused_kda_generated_uint32_t
#define uint64_t flashinfer_fused_kda_generated_uint64_t
#define int32_t flashinfer_fused_kda_generated_int32_t
#define int16_t flashinfer_fused_kda_generated_int16_t
#define CUtensorMap flashinfer_fused_kda_generated_CUtensorMap
#include FLASHINFER_FUSED_KDA_DECODE_BODY_FILE
#undef CUtensorMap
#undef int16_t
#undef int32_t
#undef uint64_t
#undef uint32_t
#undef uint16_t
#undef uint8_t
#undef int8_t

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace fused_kda_decode_generated {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kQKV = 3;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckDtype(const TensorView& tensor, const char* name, DLDataType expected) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " has the wrong dtype";
}

inline void CheckCudaTensor(const TensorView& tensor, const char* name, int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id;
}

inline int32_t CheckedInt32(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= 0 && value <= std::numeric_limits<int32_t>::max())
      << name << " is outside the non-negative int32 range";
  return static_cast<int32_t>(value);
}

inline void CheckContiguous(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void CheckAlignment(const TensorView& tensor, const char* name, uintptr_t alignment) {
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0)
      << name << " data pointer must be aligned to " << alignment << " bytes";
}

inline void CheckTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == FLASHINFER_FUSED_KDA_DECODE_TARGET_MINOR)
      << "this fused KDA decode module requires compute capability 10."
      << FLASHINFER_FUSED_KDA_DECODE_TARGET_MINOR << ", got " << major << "." << minor;
}

template <size_t Expected, size_t Actual>
constexpr void CheckArgumentCount(void* (&)[Actual]) {
  static_assert(Expected == Actual, "fused KDA decode generated kernel ABI changed");
}

void Run(TensorView x, TensorView weight, TensorView conv_state, TensorView raw_gate,
         TensorView raw_beta, TensorView A_log, TensorView dt_bias, TensorView state_indices,
         TensorView state, TensorView output_gate, TensorView norm_weight, TensorView output,
         int64_t use_lower_bound, double lower_bound, double norm_eps) {
  TVM_FFI_ICHECK(x.device().device_type == kDLCUDA) << "x must be a CUDA tensor";
  const int32_t device_id = x.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id);

  for (const auto& named :
       {std::pair<const TensorView*, const char*>(&x, "x"),
        std::pair<const TensorView*, const char*>(&weight, "weight"),
        std::pair<const TensorView*, const char*>(&conv_state, "conv_state"),
        std::pair<const TensorView*, const char*>(&raw_gate, "raw_gate"),
        std::pair<const TensorView*, const char*>(&raw_beta, "raw_beta"),
        std::pair<const TensorView*, const char*>(&A_log, "A_log"),
        std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
        std::pair<const TensorView*, const char*>(&state_indices, "state_indices"),
        std::pair<const TensorView*, const char*>(&state, "state"),
        std::pair<const TensorView*, const char*>(&output_gate, "output_gate"),
        std::pair<const TensorView*, const char*>(&norm_weight, "norm_weight"),
        std::pair<const TensorView*, const char*>(&output, "output")}) {
    CheckCudaTensor(*named.first, named.second, device_id);
  }

  for (const auto& named : {std::pair<const TensorView*, const char*>(&x, "x"),
                            std::pair<const TensorView*, const char*>(&conv_state, "conv_state"),
                            std::pair<const TensorView*, const char*>(&raw_gate, "raw_gate"),
                            std::pair<const TensorView*, const char*>(&raw_beta, "raw_beta"),
                            std::pair<const TensorView*, const char*>(&output_gate, "output_gate"),
                            std::pair<const TensorView*, const char*>(&output, "output")}) {
    CheckDtype(*named.first, named.second, dl_bfloat16);
  }
  for (const auto& named :
       {std::pair<const TensorView*, const char*>(&weight, "weight"),
        std::pair<const TensorView*, const char*>(&A_log, "A_log"),
        std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
        std::pair<const TensorView*, const char*>(&norm_weight, "norm_weight")}) {
    CheckDtype(*named.first, named.second, dl_float32);
  }
  CheckDtype(state_indices, "state_indices", dl_int32);
#if FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16
  CheckDtype(state, "state", dl_bfloat16);
#else
  CheckDtype(state, "state", dl_float32);
#endif
  TVM_FFI_ICHECK(use_lower_bound == 0 || use_lower_bound == 1)
      << "use_lower_bound must be zero or one";
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && ((use_lower_bound == 1 && lower_bound < 0.0) ||
                                                (use_lower_bound == 0 && lower_bound == 0.0)))
      << "lower_bound must be finite and negative when enabled, or zero when disabled";
  TVM_FFI_ICHECK(std::isfinite(norm_eps) && norm_eps >= 0.0)
      << "norm_eps must be finite and non-negative";

  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(0) > 0 && x.size(1) % (kQKV * kHeadDim) == 0)
      << "x must have shape [rows, 3 * H * 128]";
  const int64_t rows = x.size(0);
  const int64_t num_heads = x.size(1) / (kQKV * kHeadDim);
  TVM_FFI_ICHECK(num_heads == 12 || num_heads == 24 || num_heads == 32 || num_heads == 48 ||
                 num_heads == 96)
      << "H must be one of 12, 24, 32, 48, or 96";
  const int64_t hidden = num_heads * kHeadDim;
  const int64_t qkv_size = kQKV * hidden;
  TVM_FFI_ICHECK(x.stride(1) == 1) << "x must have unit channel stride";

  TVM_FFI_ICHECK(weight.ndim() == 3 && weight.size(0) == kQKV && weight.size(1) == 4 &&
                 weight.size(2) == hidden)
      << "weight must have shape [3, 4, H * 128]";
  CheckContiguous(weight, "weight");
  TVM_FFI_ICHECK(conv_state.ndim() == 3 && conv_state.size(0) > 0 &&
                 conv_state.size(1) == qkv_size && conv_state.size(2) == 3 &&
                 conv_state.stride(1) == 1 && conv_state.stride(2) == qkv_size &&
                 conv_state.stride(0) >= kQKV * qkv_size)
      << "conv_state must use the fused decode cache layout";
  CheckAlignment(conv_state, "conv_state", 8);
  TVM_FFI_ICHECK(conv_state.stride(0) % 4 == 0)
      << "conv_state slot stride must preserve eight-byte alignment";

  TVM_FFI_ICHECK(raw_gate.ndim() == 4 && raw_gate.size(0) == 1 && raw_gate.size(1) == rows &&
                 raw_gate.size(2) == num_heads && raw_gate.size(3) == kHeadDim)
      << "raw_gate must have shape [1, rows, H, 128]";
  CheckContiguous(raw_gate, "raw_gate");
  TVM_FFI_ICHECK(raw_beta.ndim() == 3 && raw_beta.size(0) == 1 && raw_beta.size(1) == rows &&
                 raw_beta.size(2) == num_heads && raw_beta.stride(2) == 1)
      << "raw_beta must have shape [1, rows, H] with unit head stride";
  TVM_FFI_ICHECK(A_log.ndim() == 1 && A_log.size(0) == num_heads) << "A_log must have shape [H]";
  CheckContiguous(A_log, "A_log");
  TVM_FFI_ICHECK(dt_bias.ndim() == 1 && dt_bias.size(0) == hidden)
      << "dt_bias must have shape [H * 128]";
  CheckContiguous(dt_bias, "dt_bias");
  TVM_FFI_ICHECK(state_indices.ndim() == 1 && state_indices.size(0) == rows)
      << "state_indices must have shape [rows]";
  CheckContiguous(state_indices, "state_indices");

  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) == conv_state.size(0) &&
                 state.size(1) == num_heads && state.size(2) == kHeadDim &&
                 state.size(3) == kHeadDim && state.stride(3) == 1 && state.stride(2) == kHeadDim &&
                 state.stride(1) == kHeadDim * kHeadDim &&
                 state.stride(0) >= num_heads * kHeadDim * kHeadDim)
      << "state must have shape [slots, H, 128, 128] with contiguous slot contents";
#if FLASHINFER_FUSED_KDA_DECODE_STATE_IS_BFLOAT16
  CheckAlignment(state, "state", 16);
#else
  CheckAlignment(state, "state", 32);
#endif
  TVM_FFI_ICHECK(state.stride(0) % 8 == 0) << "state slot stride must preserve 32-byte alignment";

  TVM_FFI_ICHECK(output_gate.ndim() == 3 && output_gate.size(0) == rows &&
                 output_gate.size(1) == num_heads && output_gate.size(2) == kHeadDim &&
                 output_gate.stride(2) == 1 && output_gate.stride(1) == kHeadDim)
      << "output_gate must have shape [rows, H, 128] with contiguous head rows";
  TVM_FFI_ICHECK(norm_weight.ndim() == 1 && norm_weight.size(0) == kHeadDim)
      << "norm_weight must have shape [128]";
  CheckContiguous(norm_weight, "norm_weight");
  TVM_FFI_ICHECK(output.ndim() == 4 && output.size(0) == 1 && output.size(1) == rows &&
                 output.size(2) == num_heads && output.size(3) == kHeadDim)
      << "output must have shape [1, rows, H, 128]";
  CheckContiguous(output, "output");

  int32_t x_row_stride = CheckedInt32(x.stride(0), "x row stride");
  int32_t conv_slot_stride = CheckedInt32(conv_state.stride(0), "conv_state slot stride");
  int32_t beta_row_stride = CheckedInt32(raw_beta.stride(1), "raw_beta row stride");
  int32_t state_slot_stride = CheckedInt32(state.stride(0), "state slot stride");
  int32_t output_gate_row_stride = CheckedInt32(output_gate.stride(0), "output_gate row stride");
  int32_t H = CheckedInt32(num_heads, "H");
  int32_t rows_i32 = CheckedInt32(rows, "rows");
  int32_t use_lower_bound_i32 = static_cast<int32_t>(use_lower_bound);
  float lower_bound_log2 = static_cast<float>(lower_bound * 1.4426950408889634);
  float norm_eps_f32 = static_cast<float>(norm_eps);

  void* x_ptr = x.data_ptr();
  void* weight_ptr = weight.data_ptr();
  void* conv_state_ptr = conv_state.data_ptr();
  void* raw_gate_ptr = raw_gate.data_ptr();
  void* raw_beta_ptr = raw_beta.data_ptr();
  void* A_log_ptr = A_log.data_ptr();
  void* dt_bias_ptr = dt_bias.data_ptr();
  void* state_indices_ptr = state_indices.data_ptr();
  void* state_ptr = state.data_ptr();
  void* output_gate_ptr = output_gate.data_ptr();
  void* norm_weight_ptr = norm_weight.data_ptr();
  void* output_ptr = output.data_ptr();
  void* args[] = {&x_ptr,
                  &weight_ptr,
                  &conv_state_ptr,
                  &raw_gate_ptr,
                  &raw_beta_ptr,
                  &A_log_ptr,
                  &dt_bias_ptr,
                  &state_indices_ptr,
                  &state_ptr,
                  &output_gate_ptr,
                  &norm_weight_ptr,
                  &output_ptr,
                  &x_row_stride,
                  &conv_slot_stride,
                  &beta_row_stride,
                  &state_slot_stride,
                  &output_gate_row_stride,
                  &H,
#if FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS
                  &rows_i32,
#endif
                  &use_lower_bound_i32,
                  &lower_bound_log2,
                  &norm_eps_f32};
#if FLASHINFER_FUSED_KDA_DECODE_HAS_ROWS
  CheckArgumentCount<22>(args);
#else
  CheckArgumentCount<21>(args);
#endif

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  const dim3 grid(static_cast<uint32_t>(num_heads), static_cast<uint32_t>(rows), 1);
  const dim3 block(FLASHINFER_FUSED_KDA_DECODE_THREADS, 1, 1);
  const void* kernel = reinterpret_cast<const void*>(FLASHINFER_FUSED_KDA_DECODE_KERNEL);

#if FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES > 49152
  int max_dynamic_smem = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
      "cudaDeviceGetAttribute(max dynamic shared memory)");
  TVM_FFI_ICHECK(FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES <= max_dynamic_smem)
      << "fused KDA decode dynamic shared memory exceeds device capacity";
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES),
            "cudaFuncSetAttribute(fused KDA decode)");
#endif

  CheckCuda(
      cudaLaunchKernel(kernel, grid, block, args, FLASHINFER_FUSED_KDA_DECODE_SMEM_BYTES, stream),
      "fused KDA decode launch");
}

}  // namespace fused_kda_decode_generated
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::fused_kda_decode_generated::Run);
