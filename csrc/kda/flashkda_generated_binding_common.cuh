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

#include <cstddef>
#include <cstdint>

#include "flashkda_binding_common.cuh"
#include "flashkda_generated_bt16_descriptor_common.cuh"

#if defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#ifndef FLASHKDA_GENERATED_CUBIN_IDENT
#error "FLASHKDA_GENERATED_CUBIN_IDENT must name the embedded cubin"
#endif
#define FLASHKDA_GENERATED_EMBED_CUBIN_IMPL(name) TVM_FFI_EMBED_CUBIN(name)
#define FLASHKDA_GENERATED_EMBED_CUBIN(name) \
  FLASHKDA_GENERATED_EMBED_CUBIN_IMPL(name)
FLASHKDA_GENERATED_EMBED_CUBIN(FLASHKDA_GENERATED_CUBIN_IDENT);
#define FLASHKDA_GENERATED_GET_KERNEL_IMPL(name, kernel_name) \
  TVM_FFI_EMBED_CUBIN_GET_KERNEL(name, kernel_name)
#define FLASHKDA_GENERATED_GET_KERNEL(name, kernel_name) \
  FLASHKDA_GENERATED_GET_KERNEL_IMPL(name, kernel_name)
#define FLASHKDA_GENERATED_STRINGIFY_IMPL(value) #value
#define FLASHKDA_GENERATED_STRINGIFY(value) \
  FLASHKDA_GENERATED_STRINGIFY_IMPL(value)
#define FLASHKDA_GENERATED_KERNEL_ARGUMENT nullptr
#else
#define FLASHKDA_GENERATED_KERNEL_ARGUMENT \
  reinterpret_cast<const void*>(FLASHKDA_GENERATED_KERNEL)
#endif

#ifndef FLASHKDA_GENERATED_BODY_FILE
#error "FLASHKDA_GENERATED_BODY_FILE must name one audited generated body"
#endif
#ifndef FLASHKDA_GENERATED_KERNEL
#error "FLASHKDA_GENERATED_KERNEL must name the audited kernel symbol"
#endif
#ifndef FLASHKDA_GENERATED_THREADS
#error "FLASHKDA_GENERATED_THREADS must match the audited launch bounds"
#endif
#ifndef FLASHKDA_GENERATED_SMEM_BYTES
#error "FLASHKDA_GENERATED_SMEM_BYTES must match the audited dynamic shared memory"
#endif
#ifndef FLASHKDA_GENERATED_USE_PDL
#error "FLASHKDA_GENERATED_USE_PDL must be 0 or 1"
#endif
#ifndef FLASHKDA_GENERATED_STATE_MODE
#error "FLASHKDA_GENERATED_STATE_MODE must select the generated state pointer slots"
#endif

#define FLASHKDA_GENERATED_STATE_NONE 0
#define FLASHKDA_GENERATED_STATE_BF16 1
#define FLASHKDA_GENERATED_STATE_FP32 2
#define FLASHKDA_GENERATED_STATE_BF16_F32_DEPENDENCY 3

#define FLASHKDA_GENERATED_VARIANT_DEFAULT 1
#define FLASHKDA_GENERATED_VARIANT_SERVING 2
#define FLASHKDA_GENERATED_VARIANT_VTILE 3

#define FLASHKDA_GENERATED_ROUTE_DIRECT_M128_MAIN 1
#define FLASHKDA_GENERATED_ROUTE_DIRECT_M128_N16_MAIN 2
#define FLASHKDA_GENERATED_ROUTE_SOURCE599_VTILE_M128_MAIN 3
#define FLASHKDA_GENERATED_ROUTE_INDEPENDENT_DVSPLIT_M64_MAIN 4
#define FLASHKDA_GENERATED_ROUTE_SCALAR_CHUNK_LPT_M128_MAIN 5
#define FLASHKDA_GENERATED_ROUTE_PIECE_PERSISTENT_M128_MAIN 6
#define FLASHKDA_GENERATED_ROUTE_TASKIZED_PERSISTENT_M128_MAIN 7
#define FLASHKDA_GENERATED_ROUTE_SMALL_BH_OWNER_HELPER_M128_MAIN 8
#define FLASHKDA_GENERATED_ROUTE_BT16_PREPARE_CHAIN_M64_PREPARE 9
#define FLASHKDA_GENERATED_ROUTE_BT16_PREPARE_CHAIN_M64_MAIN 10
#define FLASHKDA_GENERATED_ROUTE_AFFINE_SPLIT_M128_AFFINE_MAIN 11
#define FLASHKDA_GENERATED_ROUTE_AFFINE_SPLIT_M128_AFFINE_MAP 12
#define FLASHKDA_GENERATED_ROUTE_AFFINE_SPLIT_M128_AFFINE_SCAN 13
#define FLASHKDA_GENERATED_ROUTE_AFFINE_SPLIT_M128_AFFINE_CORRECTION 14

#define FLASHKDA_GENERATED_AFFINE_NONE 0
#define FLASHKDA_GENERATED_AFFINE_FP32_SPLIT_STATE 1
#define FLASHKDA_GENERATED_AFFINE_BF16_STATE_WITH_FP32_SPLIT_DEPENDENCY 2
#define FLASHKDA_GENERATED_AFFINE_FP32_CARRY_DEPENDENCY 3
#define FLASHKDA_GENERATED_AFFINE_BF16_INDEXED_INITIAL_FP32_FINAL 4

static_assert(FLASHKDA_GENERATED_THREADS > 0);
static_assert(FLASHKDA_GENERATED_SMEM_BYTES > 0);
static_assert(FLASHKDA_GENERATED_USE_PDL == 0 || FLASHKDA_GENERATED_USE_PDL == 1);
static_assert(FLASHKDA_GENERATED_STATE_MODE >= FLASHKDA_GENERATED_STATE_NONE &&
              FLASHKDA_GENERATED_STATE_MODE <= FLASHKDA_GENERATED_STATE_BF16_F32_DEPENDENCY);

#if !defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
// The generated source is standalone and owns private fixed-width aliases.
// Isolate them from CUDA and TVM-FFI declarations in this translation unit.
#define int8_t flashkda_generated_private_int8_t
#define uint8_t flashkda_generated_private_uint8_t
#define uint16_t flashkda_generated_private_uint16_t
#define uint32_t flashkda_generated_private_uint32_t
#define uint64_t flashkda_generated_private_uint64_t
#define int32_t flashkda_generated_private_int32_t
#define int16_t flashkda_generated_private_int16_t
#define FlashKDATensorMap flashkda_generated_private_TensorMap
#define FlashKDATensorMapPack flashkda_generated_private_TensorMapPack
#define CUtensorMap flashkda_generated_private_CUtensorMap
#include FLASHKDA_GENERATED_BODY_FILE
#undef CUtensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef int8_t
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

#ifdef THREADS
static_assert(THREADS == FLASHKDA_GENERATED_THREADS,
              "generated body and selector thread counts disagree");
#endif
static_assert(SMEM_TOTAL == FLASHKDA_GENERATED_SMEM_BYTES,
              "generated body and selector shared-memory sizes disagree");
#endif

namespace flashinfer {
namespace flash_kda_generated {

using flash_kda::CheckCuda;
using flash_kda::CheckCudaTensor;
using flash_kda::CheckCudaTensorDevice;
using flash_kda::CheckDtype;
using flash_kda::CheckDynamicSmemCapacity;
using flash_kda::CheckFlashKDATarget;
using flash_kda::EncodeTmaPointers;
using flash_kda::PackBetaForTmaIfNeeded;
using flash_kda::ResolveAndCheckServingStatePoolForDtype;
using flash_kda::TmaPointers;

struct StatePointerSlots {
  void* initial_state;
  void* final_state;
  void* initial_state_f32;
  void* final_state_f32;
  DLDataType dtype;
  int64_t pool_slots;
};

inline DLDataType GeneratedStateDtype() {
#if FLASHKDA_GENERATED_STATE_MODE == FLASHKDA_GENERATED_STATE_FP32
  return dl_float32;
#else
  return dl_bfloat16;
#endif
}

inline dim3 CheckedGrid(int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  for (const auto& named : {std::pair<int64_t, const char*>{grid_x, "grid_x"},
                            std::pair<int64_t, const char*>{grid_y, "grid_y"},
                            std::pair<int64_t, const char*>{grid_z, "grid_z"}}) {
    TVM_FFI_ICHECK(named.first > 0 &&
                   named.first <= std::numeric_limits<uint32_t>::max())
        << named.second << " must be in the positive uint32 range";
  }
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
              static_cast<uint32_t>(grid_z));
}

inline int32_t CheckedInt32(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= std::numeric_limits<int32_t>::min() &&
                 value <= std::numeric_limits<int32_t>::max())
      << name << " must fit int32";
  return static_cast<int32_t>(value);
}

inline void* CheckedBufferPointer(const TensorView& tensor, const char* name,
                                  int32_t device_id, DLDataType dtype,
                                  bool allow_empty = false) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
  TVM_FFI_ICHECK(allow_empty || tensor.numel() > 0)
      << name << " must not be empty";
  return tensor.numel() == 0 ? nullptr : tensor.data_ptr();
}

inline void* CheckedDescriptorPointer(const TensorView& tensor, const char* name,
                                      int32_t device_id, void* empty_fallback = nullptr) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dl_uint8);
  if (tensor.numel() == 0) {
    TVM_FFI_ICHECK(empty_fallback != nullptr)
        << name << " may be empty only when an audited fallback descriptor exists";
    return empty_fallback;
  }
  TVM_FFI_ICHECK(tensor.numel() >= static_cast<int64_t>(sizeof(CUtensorMap)))
      << name << " must hold at least one CUtensorMap";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) %
                     flash_kda::kTensorMapAlignment ==
                 0)
      << name << " must be CUtensorMap aligned";
  return tensor.data_ptr();
}

struct PreparedCommonInputs {
  int32_t device_id;
  int64_t num_sequences;
  cudaStream_t stream;
  StatePointerSlots state;
  TmaPointers tma;
};

inline StatePointerSlots ResolveStatePointerSlots(
    const TensorView& state_indices, const TensorView& initial_state,
    const TensorView& final_state, int32_t device_id, int64_t num_seqs,
    int64_t num_heads, int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state);
inline cudaStream_t CheckedStream(int64_t cuda_stream);

template <int ValueRows, int ChunkTokens, bool PairPackedBeta,
          bool QkStyleValueTma>
inline PreparedCommonInputs PrepareCommonInputs(
    const TensorView& q, const TensorView& k, const TensorView& v,
    const TensorView& g, const TensorView& beta, const TensorView& beta_tma,
    const TensorView& a_log, const TensorView& dt_bias,
    const TensorView& cu_seqlens, const TensorView& seq_order,
    const TensorView& state_indices, const TensorView& initial_state,
    const TensorView& out, const TensorView& final_state,
    const TensorView& descriptor_storage, int64_t prepare_descriptors,
    int64_t num_heads, int64_t beta_token_stride, int64_t state_slot_stride,
    int64_t use_state_indices, int64_t use_initial_state,
    int64_t store_final_state, double scale, double lower_bound,
    int64_t cuda_stream) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA)
      << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  CheckFlashKDATarget(device_id);
  const int64_t unchecked_num_sequences = cu_seqlens.numel() - 1;
  StatePointerSlots state = ResolveStatePointerSlots(
      state_indices, initial_state, final_state, device_id,
      unchecked_num_sequences, num_heads, state_slot_stride, use_state_indices,
      use_initial_state, store_final_state);
  const int64_t num_sequences = flash_kda::CheckCommonInputs(
      q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
      initial_state, out, final_state, descriptor_storage, prepare_descriptors,
      num_heads, use_initial_state, store_final_state, scale, lower_bound, true,
      state.pool_slots, PairPackedBeta, state.dtype);
  TVM_FFI_ICHECK(beta_token_stride == beta.stride(beta.ndim() - 2))
      << "beta_token_stride must match beta's physical token stride";
  const cudaStream_t stream = CheckedStream(cuda_stream);
  if constexpr (!PairPackedBeta) {
    PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);
  }
  TmaPointers tma = EncodeTmaPointers<ValueRows, ChunkTokens, PairPackedBeta,
                                      ValueRows, QkStyleValueTma>(
      q, k, v, g, beta_tma, out, descriptor_storage, prepare_descriptors,
      stream);
  return {device_id, num_sequences, stream, state, tma};
}

template <int ValueRows, int ChunkTokens, bool PairPackedBeta,
          bool QkStyleValueTma, bool FinalStateIsFP32 = false>
inline PreparedCommonInputs PrepareCommonInputsWithRawState(
    const TensorView& q, const TensorView& k, const TensorView& v,
    const TensorView& g, const TensorView& beta, const TensorView& beta_tma,
    const TensorView& a_log, const TensorView& dt_bias,
    const TensorView& cu_seqlens, const TensorView& seq_order,
    const TensorView& initial_state, const TensorView& out,
    const TensorView& final_state, const TensorView& descriptor_storage,
    int64_t prepare_descriptors, int64_t num_heads,
    int64_t beta_token_stride, double scale, double lower_bound,
    int64_t cuda_stream, StatePointerSlots state) {
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA)
      << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  const int64_t num_sequences = flash_kda::CheckCommonInputs(
      q, k, v, g, beta, beta_tma, a_log, dt_bias, cu_seqlens, seq_order,
      initial_state, out, final_state, descriptor_storage, prepare_descriptors,
      num_heads, 0, 0, scale, lower_bound, true, 0, PairPackedBeta,
      state.dtype, FinalStateIsFP32);
  TVM_FFI_ICHECK(beta_token_stride == beta.stride(beta.ndim() - 2))
      << "beta_token_stride must match beta's physical token stride";
  const cudaStream_t stream = CheckedStream(cuda_stream);
  if constexpr (!PairPackedBeta) {
    PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);
  }
  TmaPointers tma = EncodeTmaPointers<ValueRows, ChunkTokens, PairPackedBeta,
                                      ValueRows, QkStyleValueTma>(
      q, k, v, g, beta_tma, out, descriptor_storage, prepare_descriptors,
      stream);
  return {device_id, num_sequences, stream, state, tma};
}

inline StatePointerSlots ResolveStatePointerSlots(
    const TensorView& state_indices, const TensorView& initial_state,
    const TensorView& final_state, int32_t device_id, int64_t num_seqs,
    int64_t num_heads, int64_t state_slot_stride, int64_t use_state_indices,
    int64_t use_initial_state, int64_t store_final_state) {
#if FLASHKDA_GENERATED_STATE_MODE != FLASHKDA_GENERATED_STATE_BF16 && \
    FLASHKDA_GENERATED_STATE_MODE != FLASHKDA_GENERATED_STATE_FP32
  TVM_FFI_ICHECK(false)
      << "this generated module does not expose serving-state pointer slots";
  return {nullptr, nullptr, nullptr, nullptr, dl_bfloat16, 0};
#else
  const DLDataType dtype = GeneratedStateDtype();
  CheckCudaTensorDevice(initial_state, "initial_state", device_id);
  CheckCudaTensorDevice(final_state, "final_state", device_id);
  CheckDtype(initial_state, "initial_state", dtype);
  CheckDtype(final_state, "final_state", dtype);
  const int64_t pool_slots = ResolveAndCheckServingStatePoolForDtype(
      state_indices, initial_state, final_state, device_id, num_seqs, num_heads,
      state_slot_stride, use_state_indices, use_initial_state, store_final_state, dtype);
  StatePointerSlots slots{nullptr, nullptr, nullptr, nullptr, dtype, pool_slots};
#if FLASHKDA_GENERATED_STATE_MODE == FLASHKDA_GENERATED_STATE_FP32
  slots.initial_state_f32 = initial_state.data_ptr();
  slots.final_state_f32 = final_state.data_ptr();
#else
  slots.initial_state = initial_state.data_ptr();
  slots.final_state = final_state.data_ptr();
#endif
  return slots;
#endif
}

template <size_t Expected, size_t Actual>
constexpr void CheckArgumentCount(void* (&)[Actual]) {
  static_assert(Expected == Actual, "generated kernel ABI argument count changed");
}

static inline void ConfigureAndLaunch(const void* kernel, dim3 grid,
                                      cudaStream_t stream, void** args,
                                      const char* launch_name) {
  TVM_FFI_ICHECK(grid.x > 0 && grid.y > 0 && grid.z > 0)
      << "generated kernel grid dimensions must be positive";
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  CheckDynamicSmemCapacity(device_id, FLASHKDA_GENERATED_SMEM_BYTES);
#if defined(FLASHKDA_GENERATED_EMBEDDED_CUBIN)
  (void)kernel;
  static auto embedded_kernel = FLASHKDA_GENERATED_GET_KERNEL(
      FLASHKDA_GENERATED_CUBIN_IDENT,
      FLASHKDA_GENERATED_STRINGIFY(FLASHKDA_GENERATED_KERNEL));
  namespace cuda_api = tvm::ffi::cuda_api;
  auto device = cuda_api::GetDeviceHandle(device_id);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuda_api::SetKernelMaxDynamicSharedMem(
      embedded_kernel.GetHandle(), FLASHKDA_GENERATED_SMEM_BYTES, device));
#if FLASHKDA_GENERATED_USE_PDL
  cuda_api::LaunchConfig config{};
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute.value.programmaticStreamSerializationAllowed = 1;
  config.gridDimX = grid.x;
  config.gridDimY = grid.y;
  config.gridDimZ = grid.z;
  config.blockDimX = FLASHKDA_GENERATED_THREADS;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = FLASHKDA_GENERATED_SMEM_BYTES;
  config.hStream = stream;
#else
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.gridDim = grid;
  config.blockDim = dim3(FLASHKDA_GENERATED_THREADS, 1, 1);
  config.dynamicSmemBytes = FLASHKDA_GENERATED_SMEM_BYTES;
  config.stream = stream;
#endif
  config.attrs = &attribute;
  config.numAttrs = 1;
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
      embedded_kernel.LaunchEx(args, config));
#else
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(embedded_kernel.Launch(
      args, tvm::ffi::dim3(grid.x, grid.y, grid.z),
      tvm::ffi::dim3(FLASHKDA_GENERATED_THREADS, 1, 1), stream,
      FLASHKDA_GENERATED_SMEM_BYTES));
#endif
#else
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FLASHKDA_GENERATED_SMEM_BYTES),
            "cudaFuncSetAttribute(generated FlashKDA kernel)");
#if FLASHKDA_GENERATED_USE_PDL
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(FLASHKDA_GENERATED_THREADS, 1, 1);
  config.dynamicSmemBytes = FLASHKDA_GENERATED_SMEM_BYTES;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  CheckCuda(cudaLaunchKernelExC(&config, kernel, args), launch_name);
#else
  CheckCuda(cudaLaunchKernel(kernel, grid, dim3(FLASHKDA_GENERATED_THREADS, 1, 1), args,
                             FLASHKDA_GENERATED_SMEM_BYTES, stream),
            launch_name);
#endif
#endif
}

inline cudaStream_t CheckedStream(int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

}  // namespace flash_kda_generated
}  // namespace flashinfer
