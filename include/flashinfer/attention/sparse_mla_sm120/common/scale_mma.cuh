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

#include "../arch/common.cuh"
#include "../model/kv_cache_traits.cuh"
#include "../model/scale_convert.cuh"

struct Fp8WeightQuad {
  uint8_t h0_e0;
  uint8_t h0_e1;
  uint8_t h1_e0;
  uint8_t h1_e1;
};

template <ScaleFormat SF>
struct WeightFp8PassTraits {
  static constexpr int PASSES = (SF == ScaleFormat::ARBITRARY_FP32) ? 2 : 1;
};

template <typename KV>
__device__ __forceinline__ uint8_t qk_k_scale_selector(const uint8_t* scale_base, int blk) {
  if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
    return 0x7F;
  } else if constexpr (KV::SCALE_IN_KV_SMEM) {
    return KV::scale_to_ue8m0(reinterpret_cast<const float*>(scale_base)[blk]);
  } else {
    return scale_base[blk];
  }
}

template <typename KV>
__device__ __forceinline__ float kv_scale_fp32(const uint8_t* scale_base, int blk) {
  if constexpr (KV::SCALE_IN_KV_SMEM) {
    return reinterpret_cast<const float*>(scale_base)[blk];
  } else {
    return ue8m0_to_fp32(scale_base[blk]);
  }
}

template <ScaleFormat SF>
__device__ __forceinline__ void init_qk_acc(const float qk[4], float& acc0, float& acc1,
                                            float& acc2, float& acc3) {
  if constexpr (SF == ScaleFormat::ARBITRARY_FP32) {
    acc0 = 0.f;
    acc1 = 0.f;
    acc2 = 0.f;
    acc3 = 0.f;
  } else {
    acc0 = qk[0];
    acc1 = qk[1];
    acc2 = qk[2];
    acc3 = qk[3];
  }
}

template <typename KV>
__device__ __forceinline__ void commit_qk_acc(float qk[4], float acc0, float acc1, float acc2,
                                              float acc3, const uint8_t* e0_scale_base,
                                              const uint8_t* e1_scale_base, int blk) {
  if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
    const float ks0 = kv_scale_fp32<KV>(e0_scale_base, blk);
    const float ks1 = kv_scale_fp32<KV>(e1_scale_base, blk);
    qk[0] += acc0 * ks0;
    qk[1] += acc1 * ks1;
    qk[2] += acc2 * ks0;
    qk[3] += acc3 * ks1;
  } else {
    qk[0] = acc0;
    qk[1] = acc1;
    qk[2] = acc2;
    qk[3] = acc3;
  }
}

template <ScaleFormat SF>
__device__ __forceinline__ uint8_t quantize_weight_e4m3_for_pass(float x, int pass) {
  if constexpr (SF == ScaleFormat::ARBITRARY_FP32) {
    return (pass == 0) ? quantize_e4m3_byte(x) : quantize_e4m3_residual_byte(x);
  } else {
    return quantize_e4m3_byte(x);
  }
}

template <ScaleFormat SF>
__device__ __forceinline__ Fp8WeightQuad quantize_weight_quad_for_pass(float h0_e0, float h0_e1,
                                                                       float h1_e0, float h1_e1,
                                                                       int pass) {
  return Fp8WeightQuad{quantize_weight_e4m3_for_pass<SF>(h0_e0, pass),
                       quantize_weight_e4m3_for_pass<SF>(h0_e1, pass),
                       quantize_weight_e4m3_for_pass<SF>(h1_e0, pass),
                       quantize_weight_e4m3_for_pass<SF>(h1_e1, pass)};
}
