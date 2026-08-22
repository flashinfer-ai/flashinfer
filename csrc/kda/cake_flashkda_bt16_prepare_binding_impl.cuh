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

#ifndef FLASHINFER_BT16_PREPARE_KERNEL
#error "FLASHINFER_BT16_PREPARE_KERNEL must name the frozen prepare kernel"
#endif

namespace flashinfer {
namespace flash_kda {

static_assert(THREADS == 128);
static_assert(SMEM_TOTAL == 44032);

void RunBt16Prepare(
    TensorView q, TensorView k, TensorView raw_gate, TensorView beta_logits,
    TensorView a_log, TensorView dt_bias, TensorView cu_seqlens, TensorView cu_chunks,
    TensorView chunk_to_seq, TensorView ws_qd, TensorView ws_kd,
    TensorView ws_w, TensorView ws_qk_t, TensorView ws_diag, TensorView descriptor_storage,
    int64_t prepare_descriptors, int64_t total_chunks, int64_t num_heads,
    double gate_lower_bound, int64_t prepare_total_ctas, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckBt16PrepareInputs(q, k, raw_gate, beta_logits, a_log, dt_bias, cu_seqlens, cu_chunks,
                         chunk_to_seq, ws_qd, ws_kd, ws_w, ws_qk_t, ws_diag,
                         descriptor_storage, prepare_descriptors, prepare_total_ctas, total_chunks,
                         num_heads, gate_lower_bound);

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(FLASHINFER_BT16_PREPARE_KERNEL,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(BT16 prepare)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const Bt16PrepareTmaPointers tma =
      EncodeBt16PrepareTma(q, k, raw_gate, beta_logits, ws_qd, ws_kd, ws_w,
                           descriptor_storage, prepare_descriptors, stream);
  const dim3 grid(static_cast<uint32_t>(prepare_total_ctas), 1, 1);
  const dim3 block(THREADS, 1, 1);
  FLASHINFER_BT16_PREPARE_KERNEL<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.q),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.k),
      reinterpret_cast<__nv_bfloat16*>(raw_gate.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.raw_gate),
      reinterpret_cast<__nv_bfloat16*>(beta_logits.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.beta_logits),
      reinterpret_cast<float*>(a_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(cu_chunks.data_ptr()),
      reinterpret_cast<int*>(chunk_to_seq.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(ws_qd.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_qd),
      reinterpret_cast<__nv_bfloat16*>(ws_kd.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_kd),
      reinterpret_cast<__nv_bfloat16*>(ws_w.data_ptr()),
      reinterpret_cast<flashkda_generated_CakeTensorMap const*>(tma.ws_w),
      reinterpret_cast<__nv_bfloat16*>(ws_qk_t.data_ptr()),
      reinterpret_cast<float*>(ws_diag.data_ptr()), static_cast<int32_t>(total_chunks),
      static_cast<int32_t>(num_heads), static_cast<float>(gate_lower_bound));
  CheckCuda(cudaGetLastError(), "BT16 prepare launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunBt16Prepare);

#undef FLASHINFER_BT16_PREPARE_KERNEL
