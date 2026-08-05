/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include <cuda_runtime.h>

// The three original generated payloads came from Cake checkpoint
// 924403cd62ea6dc160ffcea63c944d1e828d393a and MR474 cleanup
// d77b50ee1e56fb3701a512f1dd2765e6067b1be2. Their pre-integration Weave
// sm_100a and sm_103a output was byte-identical:
//   bootstrap: sha256:46049370dbd905ff7234d414745d197f883157c8edc01d83230562b4dff5f862
//   rowwise:   sha256:e47f82923112e143173f32dbb145986f8c9a1503fd2bb71f22f5c065769162e3
//   warp:      sha256:8bc9df2b57d4e6021d0add03110a11f0cbb3bab1218e94d00d92f4bbf597ee9e
// Integration adds standard integer types, sm_103a-only PDL device control,
// and a CUDA <12.9 2x128-bit fallback for bootstrap's 256-bit inline PTX.
// The additive sm_103a 32K specialization comes from Cake f120b38798:
//   mr515 exp2 t512/vec4 raw generated payload:
//   sha256:8ca2513545faf9cb960c6393fb48f4c9f4598a781ad9c929e8d88ebda738a89b

extern "C" {

__global__ void kernel_flashinfer_blackwell_softmax_bootstrap_seed(
    float* logits, float* parameter, float* output, float* partial_max, float* partial_sum,
    int rows, int vocab_size, int splits, int parameter_kind, float scalar_temperature);

__global__ void kernel_flashinfer_blackwell_softmax_followup_rowwise(
    float* logits, float* parameter, float* output, int rows, int vocab_size, int parameter_kind,
    float scalar_temperature);

__global__ void kernel_flashinfer_blackwell_softmax_followup_warp(float* logits, float* parameter,
                                                                  float* output, int rows,
                                                                  int vocab_size,
                                                                  int parameter_kind,
                                                                  float scalar_temperature);

__global__ void kernel_mr474_manual_softmax_exp2_t512_vec4(float* logits, float* temperature,
                                                           float* output,
                                                           float scalar_temperature);

}  // extern "C"
