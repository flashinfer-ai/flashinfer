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
#include <cute/swizzle_layout.hpp>

#include <cutlass/arch/barrier.h>
#include "../../compute/mega.cuh"
#include "../../compute/nvfp4_quantization.cuh"
#include "../dsv4_nvfp4/resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

// (head=16, RoPE feature=64), in BF16 elements. Preserve each 16-byte
// LDSM vector while distributing its eight row addresses over the banks.
using CutePrefillQRopeLayout = decltype(cute::composition(cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_16, cute::_64>, cute::Stride<cute::_64, cute::_1>>{}));

static_assert(Dsv4Nvfp4Layout::SCALE_GROUP_SIZE == SF_VEC_SIZE);

// Both Q groups and the padded P rows are 16-byte aligned.
__device__ __forceinline__ void cute_quantize_prefill_group16(const bf16* input,
                                                        uint8_t* output, uint8_t* scale) {
  float values[16];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    uint4 packed = *reinterpret_cast<const uint4*>(input + i * 8);
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 v = __bfloat1622float2(pairs[j]);
      values[i * 8 + j * 2] = v.x;
      values[i * 8 + j * 2 + 1] = v.y;
    }
  }
  quantize_fp32_group16_to_nvfp4(values, output, scale);
}

template <int MATH_THREADS, int PACKED_STRIDE = DSV4_NVFP4_Q_PACKED_STRIDE,
          int SCALE_STRIDE = DSV4_NVFP4_SCALE_STRIDE>
__device__ __forceinline__ void cute_quantize_prefill_q_nvfp4_to_smem(uint8_t* q_nope_fp4,
                                                         uint8_t* q_nope_scales, bf16* q_rope,
                                                         const bf16* q_base, int valid_hpb = HPB,
                                                         const float* mega_cs = nullptr) {
  using KV = KVCacheTraits<ModelType::DSV4>;
  constexpr int NUM_GROUPS = HPB * DSV4_NVFP4_NUM_SCALES;

  for (int group = threadIdx.x; group < NUM_GROUPS; group += MATH_THREADS) {
    const int head = group / DSV4_NVFP4_NUM_SCALES;
    const int group_in_head = group % DSV4_NVFP4_NUM_SCALES;
    uint8_t* packed_dst = q_nope_fp4 + head * PACKED_STRIDE + group_in_head * FP4_PACKED_PER_GROUP;
    uint8_t* scale_dst = q_nope_scales + head * SCALE_STRIDE + group_in_head;
    if (head < valid_hpb) {
      const bf16* src = q_base + head * KV::D_QK + group_in_head * SF_VEC_SIZE;
      cute_quantize_prefill_group16(src, packed_dst, scale_dst);
    } else {
      *reinterpret_cast<uint2*>(packed_dst) = make_uint2(0, 0);
      *scale_dst = 0;
    }
  }

  for (int i = threadIdx.x * 4; i < HPB * KV::D_ROPE; i += MATH_THREADS * 4) {
    const int head = i / KV::D_ROPE;
    const int dim = i % KV::D_ROPE;
    uint2 packed = head < valid_hpb
        ? *reinterpret_cast<const uint2*>(q_base + head * KV::D_QK + KV::D_NOPE + dim)
        : make_uint2(0, 0);
    bf16 result[4];
    const bf16* values = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      result[j] = head < valid_hpb ? __float2bfloat16_rn(mega_rotate(
          __bfloat162float(values[j]), __bfloat162float(values[j ^ 1]),
          KV::D_NOPE + dim + j, mega_cs)) : __float2bfloat16(0.f);
    }
    *reinterpret_cast<uint2*>(q_rope + CutePrefillQRopeLayout{}(head, dim)) =
        *reinterpret_cast<const uint2*>(result);
  }
  cutlass::arch::NamedBarrier::sync(MATH_THREADS,
      static_cast<cutlass::arch::ReservedNamedBarriers>(Dsv4Nvfp4Sync::Q_STAGE));
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
