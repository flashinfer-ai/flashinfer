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

#include "cake_flashkda_bt16_binding_common.cuh"

namespace flashinfer {
namespace flash_kda {

void RunBt16Prepare(TensorView q, TensorView k, TensorView raw_gate, TensorView beta_logits,
                    TensorView a_log, TensorView dt_bias, TensorView cu_seqlens,
                    TensorView cu_chunks, TensorView chunk_to_seq, TensorView ws_qd,
                    TensorView ws_kd, TensorView ws_w, TensorView ws_qk_t, TensorView ws_diag,
                    TensorView descriptor_storage, int64_t prepare_descriptors,
                    int64_t total_chunks, int64_t num_heads, double gate_lower_bound,
                    int64_t prepare_total_ctas, int64_t cuda_stream);

void RunBt16Chain(TensorView ws_qd, TensorView ws_kd, TensorView ws_w, TensorView ws_qk,
                  TensorView ws_diag, TensorView v, TensorView cu_seqlens, TensorView cu_chunks,
                  TensorView seq_order, TensorView initial_state, TensorView out,
                  TensorView final_state, TensorView descriptor_storage,
                  int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                  int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream);

void RunBt16PrepareChainM64(
    TensorView q, TensorView k, TensorView raw_gate, TensorView beta_logits, TensorView a_log,
    TensorView dt_bias, TensorView cu_seqlens, TensorView cu_chunks, TensorView chunk_to_seq,
    TensorView ws_qd, TensorView ws_kd, TensorView ws_w, TensorView ws_qk, TensorView ws_diag,
    TensorView v, TensorView seq_order, TensorView initial_state, TensorView out,
    TensorView final_state, TensorView prepare_descriptor_storage,
    TensorView chain_descriptor_storage, int64_t prepare_prepare_descriptors,
    int64_t chain_prepare_descriptors, int64_t total_chunks, int64_t num_heads,
    double gate_lower_bound, int64_t prepare_total_ctas, int64_t use_initial_state,
    int64_t store_final_state, double scale, int64_t chain_grid_x, int64_t cuda_stream) {
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  const Bt16ChainLaunchPlan chain_plan = PrepareBt16ChainLaunch(
      ws_qd, ws_kd, ws_w, ws_qk, ws_diag, v, cu_seqlens, cu_chunks, seq_order, initial_state, out,
      final_state, chain_descriptor_storage, chain_prepare_descriptors, num_heads,
      use_initial_state, store_final_state, scale, chain_grid_x, cuda_stream);
  RunBt16Prepare(q, k, raw_gate, beta_logits, a_log, dt_bias, cu_seqlens, cu_chunks, chunk_to_seq,
                 ws_qd, ws_kd, ws_w, ws_qk, ws_diag, prepare_descriptor_storage,
                 prepare_prepare_descriptors, total_chunks, num_heads, gate_lower_bound,
                 prepare_total_ctas, cuda_stream);
  LaunchBt16Chain(chain_plan);
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunBt16PrepareChainM64);
