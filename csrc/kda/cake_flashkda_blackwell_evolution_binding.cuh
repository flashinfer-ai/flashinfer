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

#ifndef FLASHKDA_BLACKWELL_EVOLUTION_BODY_FILE
#error "FLASHKDA_BLACKWELL_EVOLUTION_BODY_FILE must name one frozen generated body"
#endif
#ifndef FLASHKDA_BLACKWELL_EVOLUTION_KERNEL
#error "FLASHKDA_BLACKWELL_EVOLUTION_KERNEL must name the frozen kernel symbol"
#endif
#ifndef FLASHKDA_BLACKWELL_EVOLUTION_VALUE_ROWS
#error "FLASHKDA_BLACKWELL_EVOLUTION_VALUE_ROWS must be 64 or 128"
#endif
#ifndef FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE
#error "FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE must be 0 or 1"
#endif

#include "flashkda_binding_common.cuh"

// The standalone body owns private fixed-width aliases and a CUtensorMap
// stand-in. Isolate those declarations from CUDA and TVM-FFI headers while
// keeping the generated body byte-for-byte frozen below its license prefix.
#define int8_t flashkda_evolution_generated_int8_t
#define uint8_t flashkda_evolution_generated_uint8_t
#define uint16_t flashkda_evolution_generated_uint16_t
#define uint32_t flashkda_evolution_generated_uint32_t
#define uint64_t flashkda_evolution_generated_uint64_t
#define int32_t flashkda_evolution_generated_int32_t
#define int16_t flashkda_evolution_generated_int16_t
#define CakeTensorMap flashkda_evolution_generated_CakeTensorMap
#define CakeTensorMapPack flashkda_evolution_generated_CakeTensorMapPack
#define CUtensorMap flashkda_evolution_generated_CUtensorMap
#include FLASHKDA_BLACKWELL_EVOLUTION_BODY_FILE
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

namespace flashinfer {
namespace flash_kda_evolution {

constexpr int32_t kThreads = 1024;
constexpr int32_t kValueRows = FLASHKDA_BLACKWELL_EVOLUTION_VALUE_ROWS;
constexpr bool kHasTileSchedule = FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE != 0;
static_assert(kValueRows == 64 || kValueRows == 128);
static_assert(FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE == 0 ||
              FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE == 1);

void Run(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
         TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
         TensorView seq_order, TensorView tile_schedule, TensorView tile_schedule_counts,
         TensorView initial_state, TensorView out, TensorView final_state,
         TensorView descriptor_storage, int64_t prepare_descriptors, int64_t grid_x,
         int64_t num_heads, int64_t use_initial_state, int64_t store_final_state, double scale,
         double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  flash_kda::CheckFlashKDATarget(device_id);

  const int64_t num_seqs = flash_kda::CheckCommonInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound);
  TVM_FFI_ICHECK(num_seqs > 0);
  TVM_FFI_ICHECK(grid_x > 0 && grid_x <= std::numeric_limits<uint32_t>::max())
      << "evolution grid.x is out of range: " << grid_x;
#if FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE
  {
    flash_kda::CheckCudaTensor(tile_schedule, "tile_schedule", device_id);
    flash_kda::CheckCudaTensor(tile_schedule_counts, "tile_schedule_counts", device_id);
    flash_kda::CheckDtype(tile_schedule, "tile_schedule", dl_int32);
    flash_kda::CheckDtype(tile_schedule_counts, "tile_schedule_counts", dl_int32);
  }
#endif

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  flash_kda::CheckDynamicSmemCapacity(device_id, kSmemBytes);
  flash_kda::CheckCuda(
      cudaFuncSetAttribute(FLASHKDA_BLACKWELL_EVOLUTION_KERNEL,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      "cudaFuncSetAttribute(FlashKDA Blackwell evolution)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const flash_kda::TmaPointers tma =
      flash_kda::EncodeTmaPointers<kValueRows, 32, false, kValueRows, kValueRows == 128>(
          q, k, v, g, beta_tma, out, descriptor_storage, prepare_descriptors, stream);
  flash_kda::PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta.stride(beta.ndim() - 2),
                                    stream);
  const dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
  const dim3 block(kThreads, 1, 1);

#if FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE
  {
    FLASHKDA_BLACKWELL_EVOLUTION_KERNEL<<<grid, block, kSmemBytes, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.q),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.k),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.v),
        reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.g),
        reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.beta),
        reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
        reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
        reinterpret_cast<int*>(seq_order.data_ptr()),
        reinterpret_cast<int*>(tile_schedule.data_ptr()),
        reinterpret_cast<int*>(tile_schedule_counts.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.out),
        reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
        static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
        static_cast<float>(scale), static_cast<float>(lower_bound));
  }
#else
  {
    FLASHKDA_BLACKWELL_EVOLUTION_KERNEL<<<grid, block, kSmemBytes, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.q),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.k),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.v),
        reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.g),
        reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.beta),
        reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
        reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
        reinterpret_cast<int*>(seq_order.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<flashkda_evolution_generated_CakeTensorMap const*>(tma.out),
        reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
        static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
        static_cast<float>(scale), static_cast<float>(lower_bound));
  }
#endif
  flash_kda::CheckCuda(cudaGetLastError(), "FlashKDA Blackwell evolution launch");
}

}  // namespace flash_kda_evolution
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda_evolution::Run);
