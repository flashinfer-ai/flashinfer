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

#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define FlashKDATensorMap flashkda_generated_FlashKDATensorMap
#define FlashKDATensorMapPack flashkda_generated_FlashKDATensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "cake_flashkda_bf16_persistent_m128.cu"
#undef CUtensorMap
#undef FlashKDATensorMapPack
#undef FlashKDATensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda {

static_assert(THREADS == 1024);
static_assert(SMEM_TOTAL == 221696);

void RunPersistentM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                       TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                       TensorView cu_seqlens, TensorView seq_order, TensorView task_ids,
                       TensorView task_offsets, TensorView initial_state, TensorView out,
                       TensorView final_state, TensorView descriptor_storage,
                       int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                       int64_t store_final_state, double scale, double lower_bound,
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
  CheckCudaTensor(task_ids, "task_ids", device_id);
  CheckCudaTensor(task_offsets, "task_offsets", device_id);
  CheckDtype(task_ids, "task_ids", dl_int32);
  CheckDtype(task_offsets, "task_offsets", dl_int32);
  const int64_t total_tasks = num_seqs * num_heads;
  TVM_FFI_ICHECK(task_ids.ndim() == 1 && task_ids.numel() == total_tasks)
      << "task_ids must contain exactly N * H task indices";
  TVM_FFI_ICHECK(task_offsets.ndim() == 1 && task_offsets.numel() >= 2)
      << "task_offsets must contain one entry per worker plus its terminal offset";
  const int64_t worker_count = task_offsets.numel() - 1;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t sm_count = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  const bool validated_device =
      major == 10 && (minor == 0 || minor == 3) && (sm_count == 148 || sm_count == 152);
  TVM_FFI_ICHECK(validated_device)
      << "persistent FlashKDA is validated only on CC10.0/CC10.3 with 148/152 SMs; got CC" << major
      << "." << minor << " with " << sm_count << " SMs";
  TVM_FFI_ICHECK(total_tasks > sm_count && worker_count > 0 && worker_count <= sm_count)
      << "persistent FlashKDA requires N * H > physical SM count and at most one worker per SM";
  TVM_FFI_ICHECK(use_initial_state == 1 && store_final_state == 1)
      << "persistent FlashKDA requires initial and final state tensors";

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_persistent_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(kernel_flashkda_bf16_persistent_m128)");

  const dim3 grid(static_cast<uint32_t>(worker_count), 1, 1);
  const dim3 block(THREADS, 1, 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128, 32>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                     prepare_descriptors, stream);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, beta.stride(beta.ndim() - 2), stream);

  kernel_flashkda_bf16_persistent_m128<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.v),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.g),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.beta),
      reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(seq_order.data_ptr()), reinterpret_cast<int*>(task_ids.data_ptr()),
      reinterpret_cast<int*>(task_offsets.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<flashkda_generated_FlashKDATensorMap const*>(tma.out),
      reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), static_cast<int32_t>(num_heads),
      static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_persistent_m128 launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunPersistentM128);
