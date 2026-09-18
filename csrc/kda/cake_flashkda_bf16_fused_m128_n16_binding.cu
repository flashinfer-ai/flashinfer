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

#include "flashkda_binding_common.cuh"

// Keep standalone generated typedefs and tensor-map wrappers out of the
// binding translation unit's CUDA/standard-library namespaces.
#define int8_t flashkda_generated_int8_t
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define FlashKDATensorMap flashkda_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_generated_FlashKDATensorMapPack
#define CakeTensorMap flashkda_generated_CakeTensorMap
#define CakeTensorMapPack flashkda_generated_CakeTensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#if defined(FLASHINFER_FLASH_KDA_N16_SHORT)
#include "cake_flashkda_bf16_fused_m128_n16_short.cu"
#else
#include "cake_flashkda_bf16_fused_m128_n16.cu"
#endif
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef int8_t
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda {

#if defined(FLASHINFER_FLASH_KDA_N16_SHORT)
using GeneratedTensorMap = flashkda_generated_CakeTensorMap;
#else
using GeneratedTensorMap = flashkda_generated_FlashKDATensorMap;
#endif

#if defined(FLASHINFER_FLASH_KDA_N16_SHORT)
constexpr int kThreads = 512;
#else
// The frozen source declares __launch_bounds__(1024) but no longer emits a
// duplicate THREADS macro for its host binding.
constexpr int kThreads = 1024;
#endif
static_assert(STORE_BACKWARD_TAPE == 0);
static_assert(SPLIT_WORK_ITEMS == 0);
#if defined(FLASHINFER_FLASH_KDA_N16_SHORT)
static_assert(SMEM_TOTAL == 112256);
#else
static_assert(SMEM_TOTAL == 117376);
#endif

void RunM128N16(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
                TensorView seq_order, TensorView state_indices, TensorView initial_state,
                TensorView out, TensorView final_state, TensorView state_checkpoints,
                TensorView checkpoint_cu_starts, TensorView descriptor_storage,
                int64_t prepare_descriptors, int64_t num_heads, int64_t beta_token_stride,
                int64_t state_slot_stride, int64_t use_state_indices, int64_t use_initial_state,
                int64_t store_final_state, int64_t checkpoint_every_n_tokens, double scale,
                double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t unchecked_num_seqs = cu_seqlens.numel() - 1;
  const int64_t state_pool_slots = ResolveAndCheckServingStatePool(
      state_indices, initial_state, final_state, device_id, unchecked_num_seqs, num_heads,
      state_slot_stride, use_state_indices, use_initial_state, store_final_state);
  const int64_t num_seqs = CheckCommonInputs(
      q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order, initial_state, out,
      final_state, descriptor_storage, prepare_descriptors, num_heads, use_initial_state,
      store_final_state, scale, lower_bound, true, state_pool_slots);
  TVM_FFI_ICHECK(beta_token_stride == beta.stride(beta.ndim() - 2))
      << "beta_token_stride must match beta's physical token stride";
  CheckServingCheckpointInputs(state_checkpoints, checkpoint_cu_starts, device_id, num_seqs,
                               num_heads, checkpoint_every_n_tokens);
  CheckServingAuxiliaryNoOverlap(state_indices, state_checkpoints, checkpoint_cu_starts, q, k, v, g,
                                 beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                                 initial_state, out, final_state, descriptor_storage,
                                 use_state_indices, checkpoint_every_n_tokens);

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);

  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128 N16)");

  const int64_t grid_x_i64 = num_seqs * num_heads;
  TVM_FFI_ICHECK(grid_x_i64 > 0 && grid_x_i64 <= std::numeric_limits<uint32_t>::max())
      << "M128 N16 FlashKDA grid.x is out of range: " << grid_x_i64;
  const dim3 grid(static_cast<uint32_t>(grid_x_i64), 1, 1);
  const dim3 block(kThreads, 1, 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128, 16>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                     prepare_descriptors, stream);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta_token_stride, stream);

  kernel_flashkda_bf16_fused_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.v),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.g),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.beta),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<GeneratedTensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(state_indices.data_ptr())),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(state_checkpoints.data_ptr())),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(checkpoint_cu_starts.data_ptr())),
      static_cast<int64_t>(beta_token_stride), static_cast<int64_t>(state_slot_stride),
      static_cast<int32_t>(use_state_indices), static_cast<int32_t>(checkpoint_every_n_tokens),
      // The frozen forward/training source retains a private training-only ABI tail even though
      // STORE_BACKWARD_TAPE and SPLIT_WORK_ITEMS are both fixed to zero in this serving export.
      // Keep those data pointers null so an accidental use fails validation instead of silently
      // corrupting a public tensor. TensorMap acquire is emitted for every descriptor argument,
      // so the final disabled slot deliberately aliases the already valid q descriptor.
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, static_cast<int32_t>(0),
      static_cast<int32_t>(0), reinterpret_cast<GeneratedTensorMap const*>(tma.q), nullptr);
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m128 N16 launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunM128N16);
