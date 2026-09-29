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
#include "cake_flashkda_bf16_piece_persistent_m128.cu"
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

void RunPiecePersistentM128(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                            TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                            TensorView cu_seqlens, TensorView seq_order, TensorView task_ids,
                            TensorView task_offsets, TensorView task_token_starts,
                            TensorView task_token_counts, TensorView task_state_sources,
                            TensorView task_state_destinations, TensorView mid_state,
                            TensorView mid_state_ready, TensorView initial_state, TensorView out,
                            TensorView final_state, TensorView descriptor_storage,
                            int64_t prepare_descriptors, int64_t num_heads,
                            int64_t use_initial_state, int64_t store_final_state, double scale,
                            double lower_bound, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  const int64_t num_seqs =
      CheckCommonInputs(q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                        initial_state, out, final_state, descriptor_storage, prepare_descriptors,
                        num_heads, use_initial_state, store_final_state, scale, lower_bound);
  for (const auto& named : {std::pair<const TensorView*, const char*>{&task_ids, "task_ids"},
                            {&task_offsets, "task_offsets"},
                            {&task_token_starts, "task_token_starts"},
                            {&task_token_counts, "task_token_counts"},
                            {&task_state_sources, "task_state_sources"},
                            {&task_state_destinations, "task_state_destinations"},
                            {&mid_state, "mid_state"},
                            {&mid_state_ready, "mid_state_ready"}}) {
    CheckCudaTensor(*named.first, named.second, device_id);
  }
  for (const auto& named : {std::pair<const TensorView*, const char*>{&task_ids, "task_ids"},
                            {&task_offsets, "task_offsets"},
                            {&task_token_starts, "task_token_starts"},
                            {&task_token_counts, "task_token_counts"},
                            {&task_state_sources, "task_state_sources"},
                            {&task_state_destinations, "task_state_destinations"}}) {
    CheckDtype(*named.first, named.second, dl_int32);
  }
  CheckDtype(mid_state, "mid_state", dl_bfloat16);
  CheckDtype(mid_state_ready, "mid_state_ready", dl_uint32);

  const int64_t total_tasks = num_seqs * num_heads;
  const int64_t entry_count = task_ids.numel();
  TVM_FFI_ICHECK(task_ids.ndim() == 1 && entry_count > total_tasks)
      << "task_ids must contain more than N * H entries for split recurrence chains";
  for (const auto& named :
       {std::pair<const TensorView*, const char*>{&task_token_starts, "task_token_starts"},
        {&task_token_counts, "task_token_counts"},
        {&task_state_sources, "task_state_sources"},
        {&task_state_destinations, "task_state_destinations"}}) {
    TVM_FFI_ICHECK(named.first->ndim() == 1 && named.first->numel() == entry_count)
        << named.second << " must contain one value per task entry";
  }
  TVM_FFI_ICHECK(task_offsets.ndim() == 1 && task_offsets.numel() >= 2)
      << "task_offsets must contain one entry per worker plus its terminal offset";
  const int64_t worker_count = task_offsets.numel() - 1;
  int32_t sm_count = 0;
  CheckCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
            "cudaDeviceGetAttribute(multiProcessorCount)");
  TVM_FFI_ICHECK(sm_count == 148 || sm_count == 152)
      << "piece-persistent FlashKDA is validated only on 148-SM or 152-SM Blackwell; got "
      << sm_count << " SMs";
  TVM_FFI_ICHECK(worker_count == sm_count)
      << "piece-persistent FlashKDA requires one resident worker per physical SM";
  TVM_FFI_ICHECK(use_initial_state == 1 && store_final_state == 1 &&
                 initial_state.data_ptr() == final_state.data_ptr())
      << "piece-persistent FlashKDA requires one caller-owned in-place state tensor";

  const int64_t handoff_count = mid_state_ready.numel();
  TVM_FFI_ICHECK(handoff_count > 0 && mid_state_ready.ndim() == 1)
      << "mid_state_ready must contain at least one handoff counter";
  TVM_FFI_ICHECK(mid_state.ndim() == 3 && mid_state.size(0) == handoff_count &&
                 mid_state.size(1) == kHeadDim && mid_state.size(2) == kHeadDim)
      << "mid_state must have shape [handoff_count, 128, 128]";

  for (const auto& metadata : {std::pair<const TensorView*, const char*>{&task_ids, "task_ids"},
                               {&task_offsets, "task_offsets"},
                               {&task_token_starts, "task_token_starts"},
                               {&task_token_counts, "task_token_counts"},
                               {&task_state_sources, "task_state_sources"},
                               {&task_state_destinations, "task_state_destinations"}}) {
    CheckNoOverlap(*metadata.first, metadata.second, mid_state, "mid_state");
    CheckNoOverlap(*metadata.first, metadata.second, mid_state_ready, "mid_state_ready");
    CheckNoOverlap(*metadata.first, metadata.second, initial_state, "initial_state");
    CheckNoOverlap(*metadata.first, metadata.second, out, "out");
    CheckNoOverlap(*metadata.first, metadata.second, descriptor_storage, "descriptor_storage");
  }
  CheckNoOverlap(mid_state, "mid_state", mid_state_ready, "mid_state_ready");
  CheckNoOverlap(mid_state, "mid_state", initial_state, "initial_state");
  CheckNoOverlap(mid_state, "mid_state", out, "out");
  CheckNoOverlap(mid_state, "mid_state", descriptor_storage, "descriptor_storage");
  CheckNoOverlap(mid_state_ready, "mid_state_ready", initial_state, "initial_state");
  CheckNoOverlap(mid_state_ready, "mid_state_ready", out, "out");
  CheckNoOverlap(mid_state_ready, "mid_state_ready", descriptor_storage, "descriptor_storage");

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
      reinterpret_cast<int*>(task_token_starts.data_ptr()),
      reinterpret_cast<int*>(task_token_counts.data_ptr()),
      reinterpret_cast<int*>(task_state_sources.data_ptr()),
      reinterpret_cast<int*>(task_state_destinations.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(mid_state.data_ptr()),
      reinterpret_cast<flashkda_generated_uint32_t*>(mid_state_ready.data_ptr()),
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunPiecePersistentM128);
