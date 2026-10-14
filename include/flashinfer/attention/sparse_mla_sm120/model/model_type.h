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
//   GLM53_NOPE: d_nope=512, d_rope=0, arbitrary FP32 scale inline, 528B/token.
//               The vLLM fp8_ds_mla ABI pads the row to 656B; the kernels take
//               the gmem row advance as a runtime stride, so a legacy 656B pool
//               and a compact 528B pool are the same kernel (the payload prefix
//               is identical). A flat 2D cache must be packed at 528B.
//   DOTS3_SWA: d_nope=1024, d_rope=64, UE8M0 scale footer, 1160B/token
//   DSV4_1:  d_nope=512, d_rope=0, UE8M0 scale footer (32-wide groups),
//            528B/token. DeepSeek-V4.1 quantizes the full 512-wide K (rope
//            lanes included) to FP8, so there is no BF16 rope segment: the
//            geometry matches GLM53_NOPE while the scale placement matches
//            DSV4. The 528B payload collides with GLM53_NOPE's, so this type
//            can only be selected explicitly, never inferred from widths.
//
// DOTS3_SWA is the sliding-window family: its candidate list is a 513-token
// positional window rather than a genuine top-k. It is the first model whose
// d_v diverges from 512 (it is 1024), so it opts out of the shared D_V assert
// in kv_cache_traits.cuh.
enum class ModelType { DSV3_2, DSV4, GLM_NSA, GLM53_NOPE, DOTS3_SWA, DSV4_1 };

// Prefill kernel variants selected by the Python dispatch planner
// (flashinfer/mla/_sparse_mla_sm120/_policy.py, KernelVariant; the values must
// match). The C++ dispatch is policy-free: it launches the named variant and
// re-checks the variant's envelope defensively. DECODE_SPLITK=0 never crosses
// this boundary (decode goes through the standalone decode entry points).
enum class PrefillVariant : int64_t { SG = 1, MG = 2, MG_DUAL = 3, SWAPAB = 4 };

enum class ScaleFormat { POW2_FP32, UE8M0_BYTE, ARBITRARY_FP32 };

// Selects NoPE QK operands, not the complete attention numerical route.
// Prefill PV remains FP8 for both modes; BF16 QK dequantizes FP8 K operands.
// DSV4 single-cache prefill selects BF16 QK for topk <= 256; dual uses BF16 QK.
// The explicit full-BF16 attention route is separate.
enum class QkComputeMode { FP8, BF16 };
