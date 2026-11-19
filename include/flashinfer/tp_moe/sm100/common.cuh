// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// Numerics shared by the SM100 fused TP MoE kernels.
#pragma once

#include <cstdint>

namespace flashinfer::tp_moe::sm100 {

// Scale exponent of one 32-value block of the routed FC1 output h when it is quantized to MXFP8 for
// FC2: 2^ceil(log2(amax / 448)) (amax 0 -> the smallest normal scale), the same rule as the x
// quantization. Every routed-h quantization site calls this helper; the scale rule is defined here
// only.
__device__ __forceinline__ int h_scale_exp(float amax) {
  const uint32_t b = __float_as_uint(fmaxf(amax * (1.0f / 448.0f), 1.17549435e-38f));
  return static_cast<int>((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}

}  // namespace flashinfer::tp_moe::sm100
