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

#ifndef FLASHKDA_DECODE_VALUE_SPLIT
#error "FLASHKDA_DECODE_VALUE_SPLIT must be defined by the binding translation unit"
#endif
#ifndef FLASHKDA_DECODE_LAUNCH_THREADS
#error "FLASHKDA_DECODE_LAUNCH_THREADS must be defined by the binding translation unit"
#endif
#ifndef FLASHKDA_DECODE_EXPECTED_SMEM
#error "FLASHKDA_DECODE_EXPECTED_SMEM must be defined by the binding translation unit"
#endif

namespace flashinfer {
namespace flash_kda_decode {

static_assert(THREADS == 256);
static_assert(SMEM_TOTAL == FLASHKDA_DECODE_EXPECTED_SMEM);
static_assert(GATE_KIND == 0);
static_assert(DIRECT_PREFIX_CHECKPOINT == 0);
static_assert(BLOCK_CHECKPOINT_MMA == 0);
static_assert(FLASHKDA_DECODE_VALUE_SPLIT == 1 ||
              FLASHKDA_DECODE_VALUE_SPLIT == 2 ||
              FLASHKDA_DECODE_VALUE_SPLIT == 4 ||
              FLASHKDA_DECODE_VALUE_SPLIT == 8);
static_assert(kHeadDim % (16 * FLASHKDA_DECODE_VALUE_SPLIT) == 0);
static_assert(FLASHKDA_DECODE_LAUNCH_THREADS % 32 == 0);
static_assert(FLASHKDA_DECODE_LAUNCH_THREADS <= THREADS);

void Run(TensorView q, TensorView k, TensorView v, TensorView g,
         TensorView beta, TensorView A_log, TensorView dt_bias,
         TensorView state, TensorView out, TensorView cu_seqlens,
         TensorView ssm_state_indices, TensorView num_accepted_tokens,
         double scale, double lower_bound, int64_t cuda_stream) {
  const LaunchContext ctx =
      CheckInputs<FLASHKDA_DECODE_VALUE_SPLIT>(
          q, k, v, g, beta, A_log, dt_bias, state, out, cu_seqlens,
          ssm_state_indices, num_accepted_tokens, scale, lower_bound,
          cuda_stream);
  ffi::CUDADeviceGuard device_guard(ctx.device_id);
  CheckDynamicSmemCapacity(ctx.device_id, SMEM_TOTAL);
  CheckCuda(
      cudaFuncSetAttribute(
          kernel_flashinfer_recurrent_kda_wy_vtile_short,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL),
      "cudaFuncSetAttribute(flashkda decode)");

  const dim3 grid(
      ctx.num_value_heads * FLASHKDA_DECODE_VALUE_SPLIT,
      ctx.num_sequences, 1);
  const dim3 block(FLASHKDA_DECODE_LAUNCH_THREADS, 1, 1);
  kernel_flashinfer_recurrent_kda_wy_vtile_short
      <<<grid, block, SMEM_TOTAL, ctx.stream>>>(
          reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
          reinterpret_cast<float*>(A_log.data_ptr()),
          reinterpret_cast<float*>(dt_bias.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          reinterpret_cast<int*>(cu_seqlens.data_ptr()),
          reinterpret_cast<int*>(ssm_state_indices.data_ptr()),
          reinterpret_cast<int*>(num_accepted_tokens.data_ptr()),
          static_cast<float>(scale), static_cast<float>(lower_bound),
          ctx.gate_token_stride, ctx.state_slot_stride, ctx.num_heads,
          ctx.num_value_heads, ctx.head_ratio);
  CheckCuda(cudaGetLastError(), "frozen FlashKDA decode launch");
}

}  // namespace flash_kda_decode
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda_decode::Run);
