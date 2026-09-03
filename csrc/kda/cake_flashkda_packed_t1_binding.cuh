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

#ifndef FLASHKDA_PACKED_T1_BODY_FILE
#error "FLASHKDA_PACKED_T1_BODY_FILE must name one frozen generated body"
#endif
#ifndef FLASHKDA_PACKED_T1_KERNEL
#error "FLASHKDA_PACKED_T1_KERNEL must name the frozen kernel symbol"
#endif
#ifndef FLASHKDA_PACKED_T1_VALUE_SPLITS
#error "FLASHKDA_PACKED_T1_VALUE_SPLITS must describe the frozen value tiling"
#endif
#ifndef FLASHINFER_FLASH_KDA_PACKED_T1_TARGET_KIND
#error "FLASHINFER_FLASH_KDA_PACKED_T1_TARGET_KIND must identify the target"
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

// The frozen kernel bodies provide private fixed-width aliases and a private
// CUtensorMap stand-in. Rename those declarations at the include boundary so
// they cannot collide with CUDA and libc headers pulled in by TVM-FFI. The
// generated body remains byte-for-byte frozen in its own translation unit.
#define uint8_t flashkda_packed_generated_uint8_t
#define uint16_t flashkda_packed_generated_uint16_t
#define uint32_t flashkda_packed_generated_uint32_t
#define uint64_t flashkda_packed_generated_uint64_t
#define int32_t flashkda_packed_generated_int32_t
#define int16_t flashkda_packed_generated_int16_t
#define FlashKDATensorMap flashkda_packed_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_packed_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_packed_generated_CUtensorMap
#include FLASHKDA_PACKED_T1_BODY_FILE
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#undef FlashKDATensorMap
#undef FlashKDATensorMapPack
#undef CUtensorMap

namespace flashinfer {
namespace flash_kda_packed_t1 {

constexpr int32_t kHeads = 12;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kMixedWidth = 3 * kHeads * kHeadDim;
constexpr int32_t kGateWidth = kHeads * kHeadDim;
constexpr int32_t kTargetFamily = 100;
constexpr int32_t kTargetSM100a = 1000;
constexpr int32_t kTargetKind = FLASHINFER_FLASH_KDA_PACKED_T1_TARGET_KIND;

static_assert(kTargetKind == kTargetFamily || kTargetKind == kTargetSM100a,
              "packed KDA T=1 must be compiled for SM100f or legacy exact SM100a");
static_assert(FLASHKDA_PACKED_T1_VALUE_SPLITS == 8 || FLASHKDA_PACKED_T1_VALUE_SPLITS == 16,
              "packed KDA T=1 exports only tile16 or tile8");

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  if (kTargetKind == kTargetFamily) {
    TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3 || minor == 7))
        << "this packed KDA T=1 module requires the SM100 family "
           "(compute capability 10.0, 10.3 or 10.7), got "
        << major << "." << minor;
  } else {
    TVM_FFI_ICHECK(major == 10 && minor == 0)
        << "this packed KDA T=1 module requires exact compute capability 10.0, got " << major << "."
        << minor;
  }
}

inline std::pair<uintptr_t, uintptr_t> TensorByteRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " has a non-byte dtype";

  uint64_t last_element = 0;
  for (int32_t i = 0; i < tensor.ndim(); ++i) {
    TVM_FFI_ICHECK(tensor.size(i) >= 0 && tensor.stride(i) >= 0)
        << name << " must not have negative shapes or strides";
    if (tensor.size(i) > 0) {
      const uint64_t extent = static_cast<uint64_t>(tensor.size(i) - 1);
      const uint64_t stride = static_cast<uint64_t>(tensor.stride(i));
      TVM_FFI_ICHECK(stride == 0 || extent <= std::numeric_limits<uint64_t>::max() / stride)
          << name << " byte range overflows uint64";
      const uint64_t contribution = extent * stride;
      TVM_FFI_ICHECK(last_element <= std::numeric_limits<uint64_t>::max() - contribution)
          << name << " byte range overflows uint64";
      last_element += contribution;
    }
  }

  const uint64_t elements = tensor.numel() == 0 ? 0 : last_element + 1;
  TVM_FFI_ICHECK(elements <= std::numeric_limits<uint64_t>::max() / (bits / 8))
      << name << " byte range overflows uint64";
  const uint64_t bytes = elements * (bits / 8);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                           const char* rhs_name) {
  const auto lhs_range = TensorByteRange(lhs, lhs_name);
  const auto rhs_range = TensorByteRange(rhs, rhs_name);
  TVM_FFI_ICHECK(lhs_range.first >= rhs_range.second || rhs_range.first >= lhs_range.second)
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

void Run(TensorView mixed_qkv, TensorView raw_gate, TensorView raw_beta, TensorView A_log,
         TensorView dt_bias, TensorView state, TensorView state_indices, TensorView out,
         int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(mixed_qkv);
  const int32_t device_id = mixed_qkv.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id);

  CHECK_CUDA(raw_gate);
  CHECK_CUDA(raw_beta);
  CHECK_CUDA(A_log);
  CHECK_CUDA(dt_bias);
  CHECK_CUDA(state);
  CHECK_CUDA(state_indices);
  CHECK_CUDA(out);
  CHECK_DEVICE(mixed_qkv, raw_gate);
  CHECK_DEVICE(mixed_qkv, raw_beta);
  CHECK_DEVICE(mixed_qkv, A_log);
  CHECK_DEVICE(mixed_qkv, dt_bias);
  CHECK_DEVICE(mixed_qkv, state);
  CHECK_DEVICE(mixed_qkv, state_indices);
  CHECK_DEVICE(mixed_qkv, out);

  CHECK_INPUT_TYPE(mixed_qkv, dl_bfloat16);
  CHECK_INPUT_TYPE(raw_gate, dl_bfloat16);
  CHECK_INPUT_TYPE(raw_beta, dl_bfloat16);
  CHECK_INPUT_TYPE(A_log, dl_float32);
  CHECK_INPUT_TYPE(dt_bias, dl_float32);
  CHECK_INPUT_TYPE(state, dl_bfloat16);
  CHECK_INPUT_TYPE(state_indices, dl_int32);
  CHECK_INPUT_TYPE(out, dl_bfloat16);

  TVM_FFI_ICHECK(mixed_qkv.ndim() == 2 && mixed_qkv.size(0) > 0 && mixed_qkv.size(1) == kMixedWidth)
      << "mixed_qkv must have shape [B, " << kMixedWidth << "]";
  const int64_t batch = mixed_qkv.size(0);
  TVM_FFI_ICHECK(batch <= 65535) << "batch exceeds the CUDA grid.y limit";
  CHECK_LAST_DIM_CONTIGUOUS(mixed_qkv);
  TVM_FFI_ICHECK(mixed_qkv.stride(0) >= kMixedWidth)
      << "mixed_qkv must have a compact last dimension and disjoint rows";

  TVM_FFI_ICHECK(raw_gate.ndim() == 2 && raw_gate.size(0) == batch &&
                 raw_gate.size(1) == kGateWidth && raw_gate.stride(0) >= kGateWidth)
      << "raw_gate must have shape [B, " << kGateWidth
      << "] with a compact last dimension and disjoint rows";
  CHECK_LAST_DIM_CONTIGUOUS(raw_gate);
  TVM_FFI_ICHECK(raw_beta.ndim() == 2 && raw_beta.size(0) == batch && raw_beta.size(1) == kHeads &&
                 raw_beta.stride(0) >= kHeads)
      << "raw_beta must have shape [B, " << kHeads
      << "] with a compact last dimension and disjoint rows";
  CHECK_LAST_DIM_CONTIGUOUS(raw_beta);

  TVM_FFI_ICHECK(A_log.ndim() == 1 && A_log.numel() == kHeads)
      << "A_log must be one-dimensional with " << kHeads << " elements";
  TVM_FFI_ICHECK(dt_bias.ndim() == 1 && dt_bias.numel() == kGateWidth)
      << "dt_bias must be one-dimensional with " << kGateWidth << " elements";
  CHECK_CONTIGUOUS(A_log);
  CHECK_CONTIGUOUS(dt_bias);

  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) > 0 && state.size(1) == kHeads &&
                 state.size(2) == kHeadDim && state.size(3) == kHeadDim)
      << "state must have shape [N, " << kHeads << ", " << kHeadDim << ", " << kHeadDim << "]";
  CHECK_LAST_DIM_CONTIGUOUS(state);
  TVM_FFI_ICHECK(state.stride(2) == kHeadDim && state.stride(1) == kHeadDim * kHeadDim &&
                 state.stride(0) >= kHeads * kHeadDim * kHeadDim)
      << "state must have compact [H,V,K] blocks and a positive, disjoint "
         "outer slot stride";
  TVM_FFI_ICHECK(state.stride(0) > 0) << "state outer slot stride must be positive";
  TVM_FFI_ICHECK(state.size(0) <= std::numeric_limits<int64_t>::max() / state.stride(0))
      << "state indexed extent overflows int64";

  TVM_FFI_ICHECK(state_indices.ndim() == 1 && state_indices.numel() == batch)
      << "state_indices must have shape [B]";
  CHECK_CONTIGUOUS(state_indices);
  TVM_FFI_ICHECK(out.ndim() == 3 && out.size(0) == batch && out.size(1) == kHeads &&
                 out.size(2) == kHeadDim)
      << "output must have shape [B, " << kHeads << ", " << kHeadDim << "]";
  CHECK_CONTIGUOUS(out);

  const std::pair<const TensorView*, const char*> read_tensors[] = {
      {&mixed_qkv, "mixed_qkv"}, {&raw_gate, "raw_gate"}, {&raw_beta, "raw_beta"},
      {&A_log, "A_log"},         {&dt_bias, "dt_bias"},   {&state_indices, "state_indices"},
  };
  CheckNoOverlap(state, "state", out, "output");
  for (const auto& named : read_tensors) {
    CheckNoOverlap(state, "state", *named.first, named.second);
    CheckNoOverlap(out, "output", *named.first, named.second);
  }

  const dim3 grid(kHeads * FLASHKDA_PACKED_T1_VALUE_SPLITS, static_cast<uint32_t>(batch), 1);
  const dim3 block(32, 1, 1);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int32_t state_base_mod8 = static_cast<int32_t>(
      (reinterpret_cast<uintptr_t>(state.data_ptr()) / sizeof(__nv_bfloat16)) & 7);
  FLASHKDA_PACKED_T1_KERNEL<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(mixed_qkv.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(raw_gate.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(raw_beta.data_ptr()),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),
      reinterpret_cast<int*>(state_indices.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), mixed_qkv.stride(0), raw_gate.stride(0),
      raw_beta.stride(0), state.stride(0), state_base_mod8);
  CheckCuda(cudaGetLastError(), "frozen packed KDA T=1 launch");
}

}  // namespace flash_kda_packed_t1
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda_packed_t1::Run);
