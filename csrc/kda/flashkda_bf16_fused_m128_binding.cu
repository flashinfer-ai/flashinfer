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

// See the M64 binding for why the frozen standalone typedefs are isolated.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#include "flashkda_bf16_fused_m128.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda {

static_assert(THREADS == 1024);
static_assert(SMEM_TOTAL == 227328);

void RunM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
             TensorView beta_tma, TensorView A_log, TensorView dt_bias, TensorView cu_seqlens,
             TensorView seq_order, TensorView initial_state, TensorView out, TensorView final_state,
             TensorView descriptor_storage, int64_t prepare_descriptors, int64_t num_heads,
             int64_t use_initial_state, int64_t store_final_state, double scale, double lower_bound,
             int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t num_seqs =
      CheckCommonInputs(q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                        initial_state, out, final_state, descriptor_storage, prepare_descriptors,
                        num_heads, use_initial_state, store_final_state, scale, lower_bound);

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);

  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128)");

  const int64_t grid_x_i64 = num_seqs * num_heads;
  TVM_FFI_ICHECK(grid_x_i64 > 0 && grid_x_i64 <= std::numeric_limits<uint32_t>::max())
      << "M128 FlashKDA grid.x is out of range: " << grid_x_i64;
  const dim3 grid(static_cast<uint32_t>(grid_x_i64), 1, 1);
  const dim3 block(THREADS, 1, 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                 prepare_descriptors, stream);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, stream);

  kernel_flashkda_bf16_fused_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()), tma.q,
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), tma.k,
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), tma.v,
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), tma.g,
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), tma.beta,
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), tma.out,
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m128 launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunM128);
