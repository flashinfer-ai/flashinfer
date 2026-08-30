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
#include <limits>
#include <utility>

#include "cake_kda_binding_common.cuh"

#define FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN 1
#define FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAP 2
#define FLASHINFER_CAKE_KDA_AFFINE_ROLE_CORRECTION 3
#define FLASHINFER_CAKE_KDA_AFFINE_ROLE_SCAN 4

#ifndef FLASHINFER_CAKE_KDA_AFFINE_BODY_FILE
#error "FLASHINFER_CAKE_KDA_AFFINE_BODY_FILE must name one sealed Cake KDA body"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_KERNEL
#error "FLASHINFER_CAKE_KDA_AFFINE_KERNEL must name the sealed Cake KDA kernel"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_THREADS
#error "FLASHINFER_CAKE_KDA_AFFINE_THREADS must match the sealed launch contract"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES
#error "FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES must match the sealed launch contract"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_USE_PDL
#error "FLASHINFER_CAKE_KDA_AFFINE_USE_PDL must be zero or one"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_ROLE
#error "FLASHINFER_CAKE_KDA_AFFINE_ROLE must select one sealed affine role"
#endif
#ifndef FLASHINFER_CAKE_KDA_AFFINE_ARG_PLAN_SHA256
#error "FLASHINFER_CAKE_KDA_AFFINE_ARG_PLAN_SHA256 must seal the kernel ABI"
#endif

static_assert(FLASHINFER_CAKE_KDA_AFFINE_THREADS > 0);
static_assert(FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES > 0);
static_assert(FLASHINFER_CAKE_KDA_AFFINE_USE_PDL == 0 ||
              FLASHINFER_CAKE_KDA_AFFINE_USE_PDL == 1);
static_assert(FLASHINFER_CAKE_KDA_AFFINE_ROLE >=
                  FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN &&
              FLASHINFER_CAKE_KDA_AFFINE_ROLE <=
                  FLASHINFER_CAKE_KDA_AFFINE_ROLE_SCAN);
static_assert(sizeof(FLASHINFER_CAKE_KDA_AFFINE_ARG_PLAN_SHA256) == 65,
              "Cake KDA affine arg-plan identity must be a full SHA-256");

// Frozen bodies own private fixed-width aliases. Keep them isolated from the
// CUDA and TVM-FFI declarations in this translation unit.
#define int8_t cake_kda_affine_generated_int8_t
#define uint8_t cake_kda_affine_generated_uint8_t
#define uint16_t cake_kda_affine_generated_uint16_t
#define uint32_t cake_kda_affine_generated_uint32_t
#define uint64_t cake_kda_affine_generated_uint64_t
#define int32_t cake_kda_affine_generated_int32_t
#define int16_t cake_kda_affine_generated_int16_t
#define CakeTensorMap cake_kda_affine_generated_CakeTensorMap
#define CakeTensorMapPack cake_kda_affine_generated_CakeTensorMapPack
#define CUtensorMap cake_kda_affine_generated_CUtensorMap
#include FLASHINFER_CAKE_KDA_AFFINE_BODY_FILE
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef int8_t
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

#ifdef THREADS
static_assert(THREADS == FLASHINFER_CAKE_KDA_AFFINE_THREADS,
              "Cake KDA body and selector thread counts disagree");
#endif
static_assert(SMEM_TOTAL == FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES,
              "Cake KDA body and selector shared-memory sizes disagree");

namespace flashinfer {
namespace cake_kda {

inline dim3 CakeKDAAffineCheckedGrid(int64_t grid_x, int64_t grid_y,
                                    int64_t grid_z) {
  for (const auto& named :
       {std::pair<int64_t, const char*>{grid_x, "grid_x"},
        std::pair<int64_t, const char*>{grid_y, "grid_y"},
        std::pair<int64_t, const char*>{grid_z, "grid_z"}}) {
    TVM_FFI_ICHECK(named.first > 0 &&
                   named.first <= std::numeric_limits<uint32_t>::max())
        << named.second << " must be in the positive uint32 range";
  }
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y),
              static_cast<uint32_t>(grid_z));
}

inline int32_t CakeKDAAffineCheckedInt32(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= std::numeric_limits<int32_t>::min() &&
                 value <= std::numeric_limits<int32_t>::max())
      << name << " must fit int32";
  return static_cast<int32_t>(value);
}

inline cudaStream_t CakeKDAAffineCheckedStream(int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0)
      << "cuda_stream must be a non-negative stream handle";
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

inline void CakeKDAAffineCheckInactiveTensor(const TensorView& tensor,
                                             const char* name,
                                             int32_t device_id,
                                             DLDataType dtype) {
  CheckCudaTensor(tensor, name, device_id);
  CheckDtype(tensor, name, dtype);
  TVM_FFI_ICHECK(tensor.numel() == 0)
      << name << " must be empty for this Cake KDA affine role";
}

inline void CakeKDAAffineCheckCompactState(const TensorView& state,
                                           const char* name,
                                           int32_t device_id,
                                           DLDataType dtype,
                                           int64_t slots,
                                           int64_t num_heads) {
  CheckCudaTensor(state, name, device_id);
  CheckDtype(state, name, dtype);
  TVM_FFI_ICHECK(slots > 0 && num_heads > 0);
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) == slots &&
                 state.size(1) == num_heads && state.size(2) == kHeadDim &&
                 state.size(3) == kHeadDim)
      << name << " must have shape [" << slots << ", " << num_heads
      << ", 128, 128]";
  TVM_FFI_ICHECK(
      state.stride(0) == num_heads * kHeadDim * kHeadDim &&
      state.stride(1) == kHeadDim * kHeadDim &&
      state.stride(2) == kHeadDim && state.stride(3) == 1)
      << name << " must use compact contiguous state slots";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0)
      << name << " must be 16-byte aligned";
}

inline void CakeKDAAffineCheckBFloat16StatePool(
    const TensorView& state, const char* name, int32_t device_id,
    int64_t num_heads, int64_t state_slot_stride) {
  CheckCudaTensorDevice(state, name, device_id);
  CheckDtype(state, name, dl_bfloat16);
  const int64_t compact_slot_stride = num_heads * kHeadDim * kHeadDim;
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) > 0 &&
                 state.size(1) == num_heads && state.size(2) == kHeadDim &&
                 state.size(3) == kHeadDim)
      << name << " must have shape [N_pool, " << num_heads
      << ", 128, 128]";
  TVM_FFI_ICHECK(state_slot_stride >= compact_slot_stride &&
                 state.stride(0) == state_slot_stride &&
                 state.stride(1) == kHeadDim * kHeadDim &&
                 state.stride(2) == kHeadDim && state.stride(3) == 1)
      << name
      << " must be contiguous inside each slot and match state_slot_stride";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                 state_slot_stride * sizeof(__nv_bfloat16) % 16 == 0)
      << name << " pool base and slot stride must be 16-byte aligned";
}

template <size_t Expected, size_t Actual>
constexpr void CakeKDAAffineCheckArgumentCount(void* (&)[Actual]) {
  static_assert(Expected == Actual,
                "Cake KDA affine kernel ABI argument count changed");
}

inline void CakeKDAAffineConfigureAndLaunch(const void* kernel, dim3 grid,
                                            int32_t device_id,
                                            cudaStream_t stream, void** args,
                                            const char* launch_name) {
  TVM_FFI_ICHECK(grid.x > 0 && grid.y > 0 && grid.z > 0)
      << "Cake KDA affine grid dimensions must be positive";
  CheckDynamicSmemCapacity(device_id,
                           FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES);
  CheckCuda(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES),
            "cudaFuncSetAttribute(Cake KDA affine kernel)");
#if FLASHINFER_CAKE_KDA_AFFINE_USE_PDL
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim =
      dim3(FLASHINFER_CAKE_KDA_AFFINE_THREADS, 1, 1);
  config.dynamicSmemBytes = FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  CheckCuda(cudaLaunchKernelExC(&config, kernel, args), launch_name);
#else
  CheckCuda(cudaLaunchKernel(
                kernel, grid,
                dim3(FLASHINFER_CAKE_KDA_AFFINE_THREADS, 1, 1), args,
                FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES, stream),
            launch_name);
#endif
}

}  // namespace cake_kda
}  // namespace flashinfer
