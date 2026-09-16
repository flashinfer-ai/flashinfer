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

#include <cuda_bf16.h>

#include "../arch/barrier.cuh"
#include "../model/nvfp4_cache_traits.cuh"
#include "nvfp4_quantization.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int DSV4_NVFP4_NUM_SCALES = NVFP4CacheTraits<ModelType::DSV4>::NUM_SCALES;
constexpr int DSV4_NVFP4_Q_PACKED_STRIDE = NVFP4CacheTraits<ModelType::DSV4>::Q_PACKED_STRIDE;
constexpr int DSV4_NVFP4_SCALE_STRIDE = NVFP4CacheTraits<ModelType::DSV4>::Q_SCALE_STRIDE;
static_assert(NVFP4CacheTraits<ModelType::DSV4>::SCALE_GROUP_SIZE == SF_VEC_SIZE);

template <int MATH_THREADS, int PACKED_STRIDE = DSV4_NVFP4_Q_PACKED_STRIDE,
          int SCALE_STRIDE = DSV4_NVFP4_SCALE_STRIDE>
__device__ __forceinline__ void quantize_q_nvfp4_to_smem(uint8_t* q_nope_fp4,
                                                         uint8_t* q_nope_scales, bf16* q_rope,
                                                         const bf16* q_base, int valid_hpb = HPB) {
  using KV = KVCacheTraits<ModelType::DSV4>;
  constexpr int NUM_GROUPS = HPB * DSV4_NVFP4_NUM_SCALES;

  for (int group = threadIdx.x; group < NUM_GROUPS; group += MATH_THREADS) {
    const int head = group / DSV4_NVFP4_NUM_SCALES;
    const int group_in_head = group % DSV4_NVFP4_NUM_SCALES;
    uint8_t* packed_dst = q_nope_fp4 + head * PACKED_STRIDE + group_in_head * FP4_PACKED_PER_GROUP;
    uint8_t* scale_dst = q_nope_scales + head * SCALE_STRIDE + group_in_head;
    if (head < valid_hpb) {
      const bf16* src = q_base + head * KV::D_QK + group_in_head * SF_VEC_SIZE;
      quantize_group16_to_nvfp4(src, packed_dst, scale_dst);
    } else {
      *reinterpret_cast<uint2*>(packed_dst) = make_uint2(0, 0);
      *scale_dst = 0;
    }
  }

  for (int i = threadIdx.x; i < HPB * KV::D_ROPE; i += MATH_THREADS) {
    const int head = i / KV::D_ROPE;
    const int dim = i % KV::D_ROPE;
    q_rope[i] =
        (head < valid_hpb) ? q_base[head * KV::D_QK + KV::D_NOPE + dim] : __float2bfloat16(0.f);
  }
  bar_sync_t<2, MATH_THREADS>();
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
