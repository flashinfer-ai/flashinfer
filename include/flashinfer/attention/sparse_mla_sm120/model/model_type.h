// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cstdint>

// ModelType determines KV cache layout, dimensions, and scale format.
//   DSV3_2:  d_nope=512, power-of-2 FP32 scale inline, 656B/token
//   DSV4:    d_nope=448, UE8M0 scale footer, 584B/token
//   GLM_NSA: d_nope=512, d_rope=64, arbitrary FP32 scale inline, 656B/token
//   GLM53_NOPE: d_nope=512, d_rope=0, arbitrary FP32 scale inline,
//               656B/token (the final 128 bytes are reserved cache padding)
//   DOTS3_SWA: d_nope=1024, d_rope=64, UE8M0 scale footer, 1160B/token
//
// DOTS3_SWA is the sliding-window family: its candidate list is a 513-token
// positional window rather than a genuine top-k. It is the first model whose
// d_v diverges from 512 (it is 1024), so it opts out of the shared D_V assert
// in kv_cache_traits.cuh.
enum class ModelType { DSV3_2, DSV4, GLM_NSA, GLM53_NOPE, DOTS3_SWA };

// Bytes per packed KV cache token row, per model type.
constexpr int bytes_per_token(ModelType mt) {
  switch (mt) {
    case ModelType::DSV3_2:
    case ModelType::GLM_NSA:
    case ModelType::GLM53_NOPE:
      return 656;
    case ModelType::DSV4:
      return 584;
    case ModelType::DOTS3_SWA:
      return 1160;
  }
  return 0;  // unreachable for a valid ModelType
}

// Prefill kernel variants selected by the Python dispatch planner
// (flashinfer/mla/_sparse_mla_sm120_plan.py, KernelVariant; the values must
// match). The C++ dispatch is policy-free: it launches the named variant and
// re-checks the variant's envelope defensively. DECODE_SPLITK=0 never crosses
// this boundary (decode goes through the standalone decode entry points).
enum class PrefillVariant : int64_t { SG = 1, MG = 2, MG_DUAL = 3, SWAPAB = 4 };

enum class ScaleFormat { POW2_FP32, UE8M0_BYTE, ARBITRARY_FP32 };

// ComputeMode selects the MMA precision path.
//   FP8:  QK and XV use UE8M0 block-scaled FP8 MMA; Q is quantized to FP8
//         on the fly; KV stays FP8 in smem. Highest throughput.
//   BF16: QK and XV use BF16 MMA; FP8 KV is dequantized to BF16 in smem.
//         Lower throughput but higher accuracy. Current dispatch sites use
//         ComputeMode::BF16 for the DSV4 prefill path when topk == 128,
//         including dual-cache variants.
enum class ComputeMode { FP8, BF16 };
