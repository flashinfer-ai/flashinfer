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

#ifndef CAKE_KDA_DECODE_VALUE_SPLIT
#error "CAKE_KDA_DECODE_VALUE_SPLIT must be defined by the binding translation unit"
#endif
#ifndef CAKE_KDA_DECODE_HEAD_DIM
#error "CAKE_KDA_DECODE_HEAD_DIM must be defined by the binding translation unit"
#endif
#ifndef CAKE_KDA_DECODE_TOKENS
#error "CAKE_KDA_DECODE_TOKENS must be defined by the binding translation unit"
#endif
#ifndef CAKE_KDA_DECODE_GATE_KIND
#error "CAKE_KDA_DECODE_GATE_KIND must be defined by the binding translation unit"
#endif
#ifndef CAKE_KDA_DECODE_LAUNCH_THREADS
#error "CAKE_KDA_DECODE_LAUNCH_THREADS must be defined by the binding translation unit"
#endif

namespace flashinfer {
namespace cake_kda_decode {

using ActiveVariant = VariantTraits<CAKE_KDA_DECODE_HEAD_DIM, CAKE_KDA_DECODE_TOKENS,
                                    CAKE_KDA_DECODE_GATE_KIND, CAKE_KDA_DECODE_VALUE_SPLIT>;

static_assert(THREADS == 32);
static_assert(CAKE_KDA_DECODE_HEAD_DIM == 128);
static_assert(CAKE_KDA_DECODE_TOKENS == 1);
static_assert(CAKE_KDA_DECODE_GATE_KIND == 0 || CAKE_KDA_DECODE_GATE_KIND == 2);
static_assert(CAKE_KDA_DECODE_VALUE_SPLIT == 8 || CAKE_KDA_DECODE_VALUE_SPLIT == 16);
static_assert(CAKE_KDA_DECODE_LAUNCH_THREADS == THREADS);

void Run(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta, TensorView A_log,
         TensorView dt_bias, TensorView state, TensorView out, TensorView cu_seqlens,
         TensorView ssm_state_indices, TensorView num_accepted_tokens, double scale,
         double lower_bound, int64_t beta_is_logit, int64_t cuda_stream) {
  const LaunchContext ctx = CheckInputs<ActiveVariant>(
      q, k, v, g, beta, A_log, dt_bias, state, out, cu_seqlens, ssm_state_indices,
      num_accepted_tokens, scale, lower_bound, beta_is_logit, cuda_stream);
  ffi::CUDADeviceGuard device_guard(ctx.device_id);

  const dim3 grid(ctx.num_value_heads * CAKE_KDA_DECODE_VALUE_SPLIT, ctx.num_sequences, 1);
  const dim3 block(CAKE_KDA_DECODE_LAUNCH_THREADS, 1, 1);
#if CAKE_KDA_DECODE_GATE_KIND == 2
  kernel_flashinfer_recurrent_kda_t1_unbounded_softplus<<<grid, block, 0, ctx.stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<float*>(A_log.data_ptr()),
      reinterpret_cast<float*>(dt_bias.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<int*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(ssm_state_indices.data_ptr()),
      reinterpret_cast<int*>(num_accepted_tokens.data_ptr()), static_cast<float>(scale),
      ctx.q_token_stride, ctx.k_token_stride, ctx.v_token_stride, ctx.gate_token_stride,
      ctx.beta_token_stride, ctx.state_slot_stride, static_cast<int64_t>(ctx.q_token_stride),
      static_cast<int64_t>(ctx.k_token_stride), static_cast<int64_t>(ctx.v_token_stride),
      static_cast<int64_t>(ctx.gate_token_stride), static_cast<int64_t>(ctx.beta_token_stride),
      static_cast<int64_t>(ctx.state_slot_stride), ctx.beta_is_logit, 0, ctx.num_heads,
      ctx.num_value_heads, ctx.head_ratio);
#else
  kernel_flashinfer_recurrent_kda_t1_direct<<<grid, block, 0, ctx.stream>>>(
      reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<int*>(cu_seqlens.data_ptr()),
      reinterpret_cast<int*>(ssm_state_indices.data_ptr()),
      reinterpret_cast<int*>(num_accepted_tokens.data_ptr()), static_cast<float>(scale),
      ctx.gate_token_stride, ctx.state_slot_stride, ctx.num_heads, ctx.num_value_heads,
      ctx.head_ratio);
#endif
  CheckCuda(cudaGetLastError(), "frozen CakeKDA direct decode launch");
}

}  // namespace cake_kda_decode
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_kda_decode::Run);
