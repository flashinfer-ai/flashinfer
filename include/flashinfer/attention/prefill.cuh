/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#ifdef FP16_QK_REDUCTION_SUPPORTED
#include "../fp16.h"
#endif
#include "../frag_layout_swizzle.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"
namespace flashinfer {

DEFINE_HAS_MEMBER(maybe_q_rope_offset)
DEFINE_HAS_MEMBER(maybe_k_rope_offset)
DEFINE_HAS_MEMBER(maybe_prefix_len_ptr)
DEFINE_HAS_MEMBER(maybe_token_pos_in_items_ptr)
DEFINE_HAS_MEMBER(token_pos_in_items_len)
DEFINE_HAS_MEMBER(maybe_max_item_len_ptr)

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t WARP_SIZE = 32;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}

template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];
      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
    };
    struct {  // NOTE(Zihao): synchronize attention states across warps
      alignas(
          16) std::conditional_t<NUM_WARPS_KV == 1, float[1],
                                 float[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO]> cta_sync_o_smem;
      alignas(16) std::conditional_t<NUM_WARPS_KV == 1, float2[1],
                                     float2[NUM_WARPS_KV * CTA_TILE_Q]> cta_sync_md_smem;
    };
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];
  };
};

template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_, uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_, uint32_t NUM_WARPS_Q_,
          uint32_t NUM_WARPS_KV_, PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_>
struct KernelTraits {
  static constexpr uint32_t NUM_STAGES = 1;  // used for BatchAttention Template
  static constexpr MaskMode MASK_MODE = MASK_MODE_;
  static constexpr uint32_t NUM_MMA_Q = NUM_MMA_Q_;
  static constexpr uint32_t NUM_MMA_KV = NUM_MMA_KV_;
  static constexpr uint32_t NUM_MMA_D_QK = NUM_MMA_D_QK_;
  static constexpr uint32_t NUM_MMA_D_VO = NUM_MMA_D_VO_;
  static constexpr uint32_t NUM_WARPS_Q = NUM_WARPS_Q_;
  static constexpr uint32_t NUM_WARPS_KV = NUM_WARPS_KV_;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  static constexpr uint32_t HEAD_DIM_QK = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO = NUM_MMA_D_VO * 16;
  static constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM_QK / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_K = HEAD_DIM_QK / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_V = HEAD_DIM_VO / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_O = HEAD_DIM_VO / upcast_size<DTypeO_>();
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16;

  static constexpr SwizzleMode SWIZZLE_MODE_Q = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KV =
      (sizeof(DTypeKV_) == 1 && HEAD_DIM_VO == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  static constexpr uint32_t KV_THR_LAYOUT_ROW = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 8;
  static constexpr uint32_t KV_THR_LAYOUT_COL = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 4;
  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using DTypeQKAccum = DTypeQKAccum_;
  using IdType = IdType_;
  using AttentionVariant = AttentionVariant_;

  static constexpr bool IsInvalid() {
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && NUM_MMA_D_VO > 4 &&
             NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
            (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
  }

  using SharedStorage = SharedStorageQKVO<NUM_WARPS_KV, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_QK,
                                          HEAD_DIM_VO, DTypeQ, DTypeKV, DTypeO>;
#ifdef FP16_QK_REDUCTION_SUPPORTED
  template <typename DT>
  static constexpr DT getNegInf() {
    if constexpr (std::is_same<DT, __half>::value) {
      return std::bit_cast<half>(fp16_ieee_from_fp32_value(-math::inf));
    } else {
      return static_cast<DTypeQKAccum>(-math::inf);
    }
  }

  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? getNegInf<DTypeQKAccum>() : DTypeQKAccum(0.f);
#else
  static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                "then recompile to support fp16 reduction");
  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? DTypeQKAccum(-math::inf) : DTypeQKAccum(0.f);
#endif
};

namespace {

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q(const uint32_t tid_y = threadIdx.y) {
  if constexpr (KTraits::NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return tid_y;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_kv(const uint32_t tid_z = threadIdx.z) {
  if constexpr (KTraits::NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return tid_z;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx(const uint32_t tid_y = threadIdx.y,
                                                 const uint32_t tid_z = threadIdx.z) {
  return get_warp_idx_kv<KTraits>(tid_z) * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid_y);
}

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_rope_offset) {
  float pos[2] = {static_cast<float>(q_rope_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_rope_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <bool produce_v, SharedMemFillMode fill_mode, typename KTraits>
__device__ __forceinline__ void produce_kv(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                           uint32_t* smem_offset, typename KTraits::DTypeKV** gptr,
                                           const uint32_t stride_n, const uint32_t kv_idx_base,
                                           const uint32_t kv_len, const dim3 tid = threadIdx) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), lane_idx = tid.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        smem.template load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        *gptr += 8 * upcast_size<DTypeKV>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DTypeKV) * NUM_MMA_D;
      *gptr += NUM_WARPS * 4 * stride_n - sizeof(DTypeKV) * NUM_MMA_D * upcast_size<DTypeKV>();
    }
    *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset);
      kv_idx += NUM_WARPS * 8;
      *gptr += NUM_WARPS * 8 * stride_n;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  }
}

template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(typename KTraits::SharedStorage* smem_storage,
                                                uint32_t* smem_offset,
                                                typename KTraits::DTypeKV* kv_ptr,
                                                const uint32_t kv_idx_base,
                                                const size_t* thr_local_kv_offset,
                                                const uint32_t kv_len, const uint32_t warp_idx,
                                                const uint32_t lane_idx) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  smem_t<KTraits::SWIZZLE_MODE_KV> smem(produce_v ? smem_storage->v_smem : smem_storage->k_smem);
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
      DType* gptr = kv_ptr + thr_local_kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = kv_ptr + thr_local_kv_offset[i];
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset);
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4], const float rope_rcp_scale,
                                               const float rope_rcp_theta,
                                               const uint32_t tid_x = threadIdx.x) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  const uint32_t lane_idx = tid_x;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[mma_d][j] =
          rope_rcp_scale *
          __powf(rope_rcp_theta,
                 float(2 * ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                            (HEAD_DIM / 2))) /
                     float(HEAD_DIM));
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,
                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][8],
                                            typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2]) {
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m[mma_q][j] = typename KTraits::DTypeQKAccum(-math::inf);
        d[mma_q][j] = 1.f;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const dim3 tid = threadIdx) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  const uint32_t lane_idx = tid.x, warp_idx_x = get_warp_idx_q<KTraits>(tid.y);

  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    uint32_t q_smem_offset_w = q_smem->template get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2 * 2; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr =
            q_ptr_base + q * q_stride_n + r * q_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          // load q fragment from gmem to smem
          q_smem->template load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                                       q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
          q_ptr += 8 * upcast_size<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, UPCAST_STRIDE_Q>(q_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_QK;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = tid.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<typename KTraits::DTypeQ>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di],
            q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4,
            group_size);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const typename KTraits::IdType* q_rope_offset,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const uint_fastdiv group_size,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const dim3 tid = threadIdx) {
  if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = tid.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<typename KTraits::DTypeQ, typename KTraits::IdType>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di], q_packed_idx_base + mma_q * 16 + lane_idx / 4, group_size,
            q_rope_offset);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    float (*rope_freq)[4], const dim3 tid = threadIdx) {
  using DTypeKV = typename KTraits::DTypeKV;
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = tid.x;
  if constexpr (KTraits::NUM_MMA_D_QK == 4 && KTraits::NUM_WARPS_Q == 4) {
    static_assert(KTraits::NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<KTraits>(tid.y);
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(KTraits::NUM_MMA_KV % 2 == 0,
                  "when NUM_MMA_D_QK == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * UPCAST_STRIDE_K;
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV / 2; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[mma_di], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * UPCAST_STRIDE_K;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + KTraits::NUM_MMA_KV) * 16 * UPCAST_STRIDE_K;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y),
                   warp_idx_z = get_warp_idx_kv<KTraits>(tid.z);
    static_assert(KTraits::NUM_MMA_D_QK % (2 * KTraits::NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*NUM_MMA_KV                | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*NUM_MMA_KV-32*NUM_MMA_KV    | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * KTraits::NUM_MMA_KV * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_MMA_D_QK / (2 * KTraits::NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * KTraits::NUM_WARPS_Q;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[mma_di], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->template advance_offset_by_column<2 * KTraits::NUM_WARPS_Q>(
                k_smem_offset_r_first_half, mma_di);
      }
      *k_smem_offset_r += 16 * UPCAST_STRIDE_K;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8]) {
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t a_frag[KTraits::NUM_MMA_Q][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);
      *q_smem_offset_r =
          q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, mma_d) -
                       KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
      } else {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(*k_smem_offset_r);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          if (mma_d == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          }
        } else if (std::is_same_v<typename KTraits::DTypeQKAccum, half>) {
          if (mma_d == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>(
                (uint32_t*)s_frag[mma_q][mma_kv], a_frag[mma_q], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[mma_q][mma_kv],
                                                      a_frag[mma_q], b_frag);
          }
        }
      }
    }
    if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
      if (mma_d % 2 == 1) {
        *k_smem_offset_r =
            k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d / 2);
      }
      *k_smem_offset_r -= KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, mma_d) -
                         KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    }
  }
  *q_smem_offset_r -= KTraits::NUM_MMA_D_QK * 2;
  *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * sizeof(typename KTraits::DTypeKV);
}

template <typename KTraits, typename Params, typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint_fastdiv group_size,
    DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8], const dim3 tid = threadIdx,
    const uint32_t kv_head_idx = blockIdx.z) {
  const uint32_t lane_idx = tid.x;
  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];
  float logits = 0., logitsTransformed = 0.;

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];

#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          logits = std::bit_cast<float>(fp16_ieee_to_fp32_value(s_frag[mma_q][mma_kv][reg_id]));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          logits = s_frag[mma_q][mma_kv][reg_id];
        }
#else
        static_assert(!std::is_same<DTypeQKAccum, __half>::value,
                      "Set -DFP16_QK_REDUCTION_SUPPORTED and install boost_math "
                      "then recompile to support fp16 reduction");
        logits = s_frag[mma_q][mma_kv][reg_id];
#endif
        logitsTransformed = variant.LogitsTransform(params, logits, batch_idx, q_idx, kv_idx,
                                                    qo_head_idx, kv_head_idx);
#ifdef FP16_QK_REDUCTION_SUPPORTED
        if constexpr (std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] =
              std::bit_cast<half>(fp16_ieee_from_fp32_value(logitsTransformed));
        } else if constexpr (!std::is_same<DTypeQKAccum, __half>::value) {
          s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
        }
#else
        s_frag[mma_q][mma_kv][reg_id] = logitsTransformed;
#endif
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t chunk_end, const uint_fastdiv group_size,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8], const dim3 tid = threadIdx,
    const uint32_t kv_head_idx = blockIdx.z) {
  const uint32_t lane_idx = tid.x;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  uint32_t q[NUM_MMA_Q][2], r[NUM_MMA_Q][2];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal || MASK_MODE == MaskMode::kMultiItemScoring
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask_multi_item_scoring(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t window_left, const uint32_t chunk_end,
    const uint_fastdiv group_size, typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    // new arguments for compact description of mask
    const uint32_t prefix_len, uint16_t* token_pos_in_items, const uint32_t lane_idx = threadIdx.x,
    const uint32_t kv_head_idx = blockIdx.z) {
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  uint32_t q[NUM_MMA_Q][2], r[NUM_MMA_Q][2];

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
    }
  }
  // prefetching global memory to registers
  uint16_t token_pos_in_items_regs[NUM_MMA_Q][(4 / 2)];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t eff_reg_id = 0; eff_reg_id < (4 / 2); ++eff_reg_id) {
      const uint32_t q_idx = q[mma_q][eff_reg_id];
      // use __ldca to hint compiler to cache in L1 for further reuse by other tiles
      const int idx_in_original_seq = q_idx + kv_len - qo_len;
      if (idx_in_original_seq >= prefix_len & idx_in_original_seq < kv_len) {
        token_pos_in_items_regs[mma_q][eff_reg_id] =
            __ldca(token_pos_in_items + idx_in_original_seq - prefix_len);
      }
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[mma_q][(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                                    2 * (lane_idx % 4) +
                                                                    8 * (reg_id / 4) + reg_id % 2;
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        const uint32_t idx_in_original_seq = q_idx + kv_len - qo_len;
        const bool out_of_boundary = kv_idx > idx_in_original_seq || (kv_idx >= chunk_end) ||
                                     kv_idx + window_left < idx_in_original_seq;
        const bool is_prefix = idx_in_original_seq < prefix_len;
        if (out_of_boundary || is_prefix) {
          s_frag[mma_q][mma_kv][reg_id] =
              out_of_boundary ? (KTraits::MaskFillValue) : s_frag[mma_q][mma_kv][reg_id];
        } else {
          s_frag[mma_q][mma_kv][reg_id] =
              (kv_idx < prefix_len |
               (idx_in_original_seq < kv_idx + token_pos_in_items_regs[mma_q][((reg_id % 4) / 2)]))
                  ? s_frag[mma_q][mma_kv][reg_id]
                  : (KTraits::MaskFillValue);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states(
    typename KTraits::AttentionVariant variant,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],
    float (*d)[2]) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr bool use_softmax = AttentionVariant::use_softmax;

  if constexpr (use_softmax) {
    const float sm_scale = variant.sm_scale_log2;
    if constexpr (std::is_same_v<DTypeQKAccum, float>) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_prev = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            float m_local =
                max(max(s_frag[mma_q][mma_kv][j * 2 + 0], s_frag[mma_q][mma_kv][j * 2 + 1]),
                    max(s_frag[mma_q][mma_kv][j * 2 + 4], s_frag[mma_q][mma_kv][j * 2 + 5]));
            m[mma_q][j] = max(m[mma_q][j], m_local);
          }
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x2));
          m[mma_q][j] = max(m[mma_q][j], math::shfl_xor_sync(m[mma_q][j], 0x1));

          float o_scale = math::ptx_exp2(m_prev * sm_scale - m[mma_q][j] * sm_scale);
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            s_frag[mma_q][mma_kv][j * 2 + 0] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 0] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 1] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 1] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m[mma_q][j] * sm_scale);
            s_frag[mma_q][mma_kv][j * 2 + 5] = math::ptx_exp2(
                s_frag[mma_q][mma_kv][j * 2 + 5] * sm_scale - m[mma_q][j] * sm_scale);
          }
        }
      }
    } else if constexpr (std::is_same_v<DTypeQKAccum, half>) {
      const half2 sm_scale = __float2half2_rn(variant.sm_scale_log2);
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        half m_prev[2];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          m_prev[j] = m[mma_q][j];
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            half2 m_local = __hmax2(*(half2*)&s_frag[mma_q][mma_kv][j * 2],
                                    *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4]);
            m[mma_q][j] = __hmax(m[mma_q][j], __hmax(m_local.x, m_local.y));
          }
        }
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x2));
        *(half2*)&m[mma_q] =
            __hmax2(*(half2*)&m[mma_q], math::shfl_xor_sync(*(half2*)&m[mma_q], 0x1));
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float o_scale = math::ptx_exp2(float(m_prev[j] * sm_scale.x - m[mma_q][j] * sm_scale.x));
          d[mma_q][j] *= o_scale;
#pragma unroll
          for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
            o_frag[mma_q][mma_d][j * 2 + 0] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 1] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 4] *= o_scale;
            o_frag[mma_q][mma_d][j * 2 + 5] *= o_scale;
          }
          half2 m2 = make_half2(m[mma_q][j], m[mma_q][j]);
#pragma unroll
          for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
            *(half2*)&s_frag[mma_q][mma_kv][j * 2] =
                math::ptx_exp2(*(half2*)&s_frag[mma_q][mma_kv][j * 2] * sm_scale - m2 * sm_scale);
            *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] = math::ptx_exp2(
                *(half2*)&s_frag[mma_q][mma_kv][j * 2 + 4] * sm_scale - m2 * sm_scale);
          }
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][8],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], float (*d)[2]) {
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];
  if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],
                                                           s_frag[mma_q][mma_kv]);
      }
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);
        } else {
          mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag[mma_q][mma_kv]);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
      uint32_t b_frag[4];
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (mma_d % 2 == 0) {
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
        } else {
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        vec_cast<typename KTraits::DTypeQ, typename KTraits::DTypeKV>::cast<8>(
            (typename KTraits::DTypeQ*)b_frag, (typename KTraits::DTypeKV*)b_frag_f8);
        swap(b_frag[1], b_frag[2]);
      } else {
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
      }
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        if constexpr (std::is_same_v<typename KTraits::DTypeQKAccum, float>) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d], (uint32_t*)s_frag[mma_q][mma_kv], b_frag);
        }
      }
      if constexpr (sizeof(typename KTraits::DTypeKV) == 1) {
        if (mma_d % 2 == 1) {
          *v_smem_offset_r =
              v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d / 2);
        }
      } else {
        *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, mma_d);
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V>(*v_smem_offset_r) -
        sizeof(typename KTraits::DTypeKV) * KTraits::NUM_MMA_D_VO;
  }
  *v_smem_offset_r -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V;
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m(typename KTraits::AttentionVariant variant,
                                           typename KTraits::DTypeQKAccum (*m)[2]) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        if (m[mma_q][j] != typename KTraits::DTypeQKAccum(-math::inf)) {
          m[mma_q][j] *= variant.sm_scale_log2;
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void transform_output(
    const Params& params, typename KTraits::AttentionVariant variant,
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::DTypeQKAccum (*m)[2],
    float (*d)[2], const uint32_t batch_idx, const uint32_t kv_tile_idx,
    const uint32_t qo_packed_idx_base, const uint32_t warp_idx, const uint32_t lane_idx,
    uint32_t kv_head_idx, const uint_fastdiv group_size) {
  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];
  float scale[KTraits::NUM_MMA_Q][2];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx / 4 + 8 * j, q[mma_q][j],
                        r[mma_q][j]);
      uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][j];
      // Update the m and d when attention sinks are used.
      variant.update_m_d(params, kv_tile_idx, qo_head_idx, m[mma_q][j], d[mma_q][j],
                         scale[mma_q][j]);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t qo_idx = q[mma_q][(reg_id % 4) / 2];
        const uint32_t qo_head_idx = kv_head_idx * group_size + r[mma_q][(reg_id % 4) / 2];
        o_frag[mma_q][mma_d][reg_id] = variant.OutputTransform(
            params, o_frag[mma_q][mma_d][reg_id], batch_idx, qo_idx, qo_head_idx,
            m[mma_q][(reg_id % 4) / 2], d[mma_q][(reg_id % 4) / 2], scale[mma_q][(reg_id % 4) / 2]);
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along threadIdx.z.
 */
template <typename KTraits>
__device__ __forceinline__ void threadblock_sync_mdo_states(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], typename KTraits::SharedStorage* smem_storage,
    typename KTraits::DTypeQKAccum (*m)[2], float (*d)[2], const uint32_t warp_idx,
    const uint32_t lane_idx, const dim3 tid = threadIdx) {
  // only necessary when blockDim.z > 1
  if constexpr (KTraits::NUM_WARPS_KV > 1) {
    float* smem_o = smem_storage->cta_sync_o_smem;
    float2* smem_md = smem_storage->cta_sync_md_smem;
    // o: [num_warps, NUM_MMA_Q, NUM_MMA_D_VO, WARP_SIZE(32), 8]
    // md: [num_warps, NUM_MMA_Q, 16, 2 (m/d)]
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        vec_t<float, 8>::memcpy(
            smem_o + (((warp_idx * KTraits::NUM_MMA_Q + mma_q) * KTraits::NUM_MMA_D_VO + mma_d) *
                          WARP_SIZE +
                      lane_idx) *
                         8,
            o_frag[mma_q][mma_d]);
      }
    }

    if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          smem_md[((warp_idx * KTraits::NUM_MMA_Q + mma_q) * 2 + j) * 8 + lane_idx / 4] =
              make_float2(float(m[mma_q][j]), d[mma_q][j]);
        }
      }

      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        float o_scale[2][KTraits::NUM_WARPS_KV];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_new = -math::inf, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            float2 md = smem_md[(((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                      KTraits::NUM_MMA_Q +
                                  mma_q) *
                                     2 +
                                 j) *
                                    8 +
                                lane_idx / 4];
            float mi = md.x;
            o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
          }
          m[mma_q][j] = typename KTraits::DTypeQKAccum(m_new);
          d[mma_q][j] = d_new;
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o + ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                    KTraits::NUM_MMA_Q +
                                mma_q) *
                                   KTraits::NUM_MMA_D_VO +
                               mma_d) *
                                  WARP_SIZE +
                              lane_idx) *
                                 8);

#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    } else {
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < KTraits::NUM_WARPS_KV; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_o + ((((i * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) *
                                    KTraits::NUM_MMA_Q +
                                mma_q) *
                                   KTraits::NUM_MMA_D_VO +
                               mma_d) *
                                  WARP_SIZE +
                              lane_idx) *
                                 8);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
              o_new[reg_id] += oi[reg_id];
            }
          }
          o_new.store(o_frag[mma_q][mma_d]);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][8], smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size, const dim3 tid = threadIdx) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(tid.y);
  const uint32_t lane_idx = tid.x;

  if constexpr (sizeof(DTypeO) == 4) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / 4 + mma_q * 16 + j * 8, q, r);
        const uint32_t o_idx = q;
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          if (o_idx < qo_upper_bound) {
            *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                       (lane_idx % 4) * 2) =
                *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][j * 2]);
            *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d * 16 +
                                       8 + (lane_idx % 4) * 2) =
                *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][4 + j * 2]);
          }
        }
      }
    }
  } else {
    if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
          uint32_t o_frag_f16[8 / 2];
          vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);

#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
          uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
              (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx % 16,
              mma_d * 2 + lane_idx / 16);
          o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
          uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
              (warp_idx_x * KTraits::NUM_MMA_Q + mma_q) * 16 + lane_idx / 4, mma_d * 2);
          ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
          ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * UPCAST_STRIDE_O))[lane_idx % 4] =
              o_frag_f16[1];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                       8 * UPCAST_STRIDE_O))[lane_idx % 4] = o_frag_f16[3];
#endif
        }
      }

      uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O>(
          warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 2 * 2; ++j) {
          uint32_t q, r;
          group_size.divmod(o_packed_idx_base + lane_idx / 8 + mma_q * 16 + j * 4, q, r);
          const uint32_t o_idx = q;
          DTypeO* o_ptr =
              o_ptr_base + q * o_stride_n + r * o_stride_h + (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
          for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_VO / 4; ++mma_do) {
            if (o_idx < qo_upper_bound) {
              o_smem->store_128b(o_smem_offset_w, o_ptr);
            }
            o_ptr += 8 * upcast_size<DTypeO>();
            o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, mma_do);
          }
          o_smem_offset_w =
              o_smem->template advance_offset_by_row<4, UPCAST_STRIDE_O>(o_smem_offset_w) -
              2 * KTraits::NUM_MMA_D_VO;
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
 * \tparam POS_ENCODING_MODE The positional encoding mode.
 * \tparam NUM_MMA_Q The number of fragments in x dimension.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
 * \tparam DTypeO The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param rope_rcp_scale 1/(rope_scale), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param rope_rcp_theta 1/(rope_theta), where rope_theta is the theta
 *   used in RoPE.
 */
template <typename KTraits, typename Params>
__device__ __forceinline__ void SinglePrefillWithKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t chunk_idx = blockIdx.y,
    const uint32_t kv_head_idx = blockIdx.z, const uint32_t num_chunks = gridDim.y,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    DTypeQ* q = params.q;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    DTypeO* o = params.o;
    float* lse = params.lse;
    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;
    const bool partition_kv = params.partition_kv;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t v_stride_h = params.v_stride_h;
    const uint_fastdiv& group_size = params.group_size;

    static_assert(sizeof(DTypeQ) == 2);
    const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
    const uint32_t num_qo_heads = num_kv_heads * group_size;

    const uint32_t max_chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
    const uint32_t chunk_start = partition_kv ? chunk_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((chunk_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    auto block = cg::this_thread_block();
    auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
    AttentionVariant variant(params, /*batch_idx=*/0, smem);
    const uint32_t window_left = variant.window_left;

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];
    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    // cooperative fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base =
        (bx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
    DTypeQ* q_ptr_base = q + (kv_head_idx * group_size) * q_stride_h;
    DTypeO* o_ptr_base = partition_kv
                             ? o + chunk_idx * o_stride_n + (kv_head_idx * group_size) * o_stride_h
                             : o + (kv_head_idx * group_size) * o_stride_h;

    uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);
    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_ptr_base, q_stride_n, q_stride_h,
                                group_size, &qo_smem, tid);

    cp_async::commit_group();
    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq, tid);
      block.sync();
    }

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

    const uint32_t num_iterations = ceil_div(
        MASK_MODE == MaskMode::kCausal
            ? min(chunk_size,
                  sub_if_greater_or_zero(
                      kv_len - qo_len + ceil_div(((bx + 1) * CTA_TILE_Q), group_size), chunk_start))
            : chunk_size,
        CTA_TILE_KV);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + ceil_div((bx + 1) * CTA_TILE_Q, group_size),
                                        qo_len + window_left + chunk_start),
                 CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + ceil_div((bx * CTA_TILE_Q), group_size) - qo_len,
                                          chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

    DTypeKV* k_ptr =
        k +
        (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * k_stride_n +
        kv_head_idx * k_stride_h + (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
    DTypeKV* v_ptr =
        v +
        (chunk_start + warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * v_stride_n +
        kv_head_idx * v_stride_h + (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);
    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                           k_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                            v_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<KTraits>(chunk_start + iter * CTA_TILE_KV, &k_smem,
                                             &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
      uint32_t kv_idx_base =
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;
      logits_transform<KTraits>(params, variant, /*batch_idx=*/0, qo_packed_idx_base, kv_idx_base,
                                qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(params, variant, /*batch_idx=*/0, qo_packed_idx_base, kv_idx_base,
                             qo_len, kv_len, chunk_end, group_size, s_frag, tid, kv_head_idx);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
          k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
          v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    // transform output
    transform_output<KTraits, Params>(params, variant, o_frag, m, d, /*batch_idx=*/0, chunk_idx,
                                      qo_packed_idx_base, warp_idx, lane_idx, kv_head_idx,
                                      group_size);

    // write back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr || partition_kv) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_len) {
                if (partition_kv) {
                  lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[qo_idx * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void SinglePrefillWithKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  SinglePrefillWithKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t qo_len = params.qo_len;
  const uint32_t kv_len = params.kv_len;
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    FLASHINFER_ERROR(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  int64_t packed_qo_len = qo_len * group_size;
  uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO);

  DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
    constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
    constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
    constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

    using DTypeQKAccum =
        typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                  float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    const int num_ctas_per_sm =
        max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                                (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
            ? 2
            : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    const uint32_t max_num_mma_kv_reg =
        (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
         !USE_FP16_QK_REDUCTION)
            ? 2
            : (8 / NUM_MMA_Q);
    const uint32_t max_num_mma_kv_smem =
        (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
        ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

    // control NUM_MMA_KV for maximum warp occupancy
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
      using KTraits =
          KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                       NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                       DTypeQKAccum, typename Params::IdType, AttentionVariant>;
      if constexpr (KTraits::IsInvalid()) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
                << " NUM_WARPS_KV=" << NUM_WARPS_KV
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
        auto kernel = SinglePrefillWithKVCacheKernel<KTraits, Params>;
        size_t smem_size = sizeof(typename KTraits::SharedStorage);
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_num_kv_chunks = (num_blocks_per_sm * num_sm) /
                                     (num_kv_heads * ceil_div(qo_len * group_size, CTA_TILE_Q));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          params.partition_kv = false;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          params.partition_kv = true;
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM_VO);
          auto o = params.o;
          auto lse = params.lse;
          params.o = tmp;
          params.lse = tmp_lse;
          void* args[] = {(void*)&params};
          dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), num_chunks, num_kv_heads);
          dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                             HEAD_DIM_VO, stream));
          } else {
            FLASHINFER_CUDA_CALL(
                AttentionSum(tmp, o, num_chunks, qo_len, num_qo_heads, HEAD_DIM_VO, stream));
          }
        }
      }
    })
  });
  return cudaSuccess;
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithRaggedKVCacheKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    DTypeQ* q = params.q;
    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    IdType* q_indptr = params.q_indptr;
    IdType* kv_indptr = params.kv_indptr;
    DTypeKV* k = params.k;
    DTypeKV* v = params.v;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const bool partition_kv = params.partition_kv;
    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint32_t v_stride_n = params.v_stride_n;
    const uint32_t v_stride_h = params.v_stride_h;
    const uint_fastdiv& group_size = params.group_size;

    static_assert(sizeof(DTypeQ) == 2);
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
    const dim3& tid = threadIdx;

    auto block = cg::this_thread_block();
    const uint32_t bx = blockIdx.x, lane_idx = tid.x,
                   warp_idx = get_warp_idx<KTraits>(tid.y, tid.z), kv_head_idx = blockIdx.z;
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = group_size * num_kv_heads;
    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    extern __shared__ uint8_t smem[];
    auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

    // skip out-of-window kv tile by add non-zero kv_start_idx offset
    const uint32_t kv_start_idx = sub_if_greater_or_zero(
        kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size, qo_len + window_left);
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len - kv_start_idx;
    const uint32_t chunk_start =
        partition_kv ? min(kv_tile_idx * max_chunk_size + kv_start_idx, kv_len) : kv_start_idx;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size + kv_start_idx, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

    DTypeQ* q_ptr_base =
        q + q_indptr[request_idx] * q_stride_n + kv_head_idx * group_size * q_stride_h;

    DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h
                                      : o + o_indptr[request_idx] * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h;

    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                q_stride_h, group_size, &qo_smem, tid);

    cp_async::commit_group();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      IdType* q_rope_offset = nullptr;

      if constexpr (has_maybe_q_rope_offset_v<Params>) {
        q_rope_offset = params.maybe_q_rope_offset;
      }
      if (!q_rope_offset) {
        q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                             &qo_smem, &q_smem_offset_r, rope_freq, tid);
      } else {
        q_smem_inplace_apply_rotary_with_pos<KTraits>(
            qo_packed_idx_base, q_rope_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, tid);
      }
      block.sync();
    }

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len - qo_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size),
                       chunk_start))
             : chunk_size),
        CTA_TILE_KV);

    const uint32_t window_iteration = ceil_div(
        sub_if_greater_or_zero(kv_len + ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size),
                               qo_len + window_left + chunk_start),
        CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len + ceil_div((qo_tile_idx * CTA_TILE_Q), group_size) - qo_len,
                       chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);

    DTypeKV* k_ptr = k +
                     (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                      lane_idx / KV_THR_LAYOUT_COL) *
                         k_stride_n +
                     kv_head_idx * k_stride_h +
                     (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
    DTypeKV* v_ptr = v +
                     (kv_indptr[request_idx] + chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                      lane_idx / KV_THR_LAYOUT_COL) *
                         v_stride_n +
                     kv_head_idx * v_stride_h +
                     (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>();

    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(k_smem, &k_smem_offset_w, &k_ptr,
                                                           k_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();
    produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(v_smem, &v_smem_offset_w, &v_ptr,
                                                            v_stride_n, 0, chunk_size, tid);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        IdType* k_rope_offset = nullptr;
        if constexpr (has_maybe_k_rope_offset_v<Params>) {
          k_rope_offset = params.maybe_k_rope_offset;
        }
        k_smem_inplace_apply_rotary<KTraits>(
            (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
                iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
      uint32_t kv_idx_base =
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;
      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                                kv_idx_base, qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                             kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                             kv_head_idx);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
          k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<true, SharedMemFillMode::kFillZero, KTraits>(
          v_smem, &v_smem_offset_w, &v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, tid);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    const uint32_t num_kv_chunks =
        ceil_div(min(kv_len_safe, window_left + CTA_TILE_Q), kv_chunk_size);
    // transform output
    transform_output<KTraits, Params>(params, variant, o_frag, m, d, /*batch_idx=*/request_idx,
                                      kv_tile_idx, qo_packed_idx_base, warp_idx, lane_idx,
                                      kv_head_idx, group_size);

    // write back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (AttentionVariant::use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_len) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice(
    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,
    const uint32_t bx = blockIdx.x, const uint32_t kv_head_idx = blockIdx.z,
    const uint32_t num_kv_heads = gridDim.z) {
  using DTypeQ = typename Params::DTypeQ;
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    using IdType = typename Params::IdType;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;
    using AttentionVariant = typename KTraits::AttentionVariant;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW = KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL = KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

    IdType* request_indices = params.request_indices;
    IdType* qo_tile_indices = params.qo_tile_indices;
    IdType* kv_tile_indices = params.kv_tile_indices;
    DTypeQ* q = params.q;
    IdType* q_indptr = params.q_indptr;
    IdType* o_indptr = params.o_indptr;
    DTypeO* o = params.o;
    float* lse = params.lse;
    bool* block_valid_mask = params.block_valid_mask;
    const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
    const bool partition_kv = params.partition_kv;
    const uint_fastdiv& group_size = params.group_size;

    uint32_t* maybe_prefix_len_ptr = nullptr;
    if constexpr (has_maybe_prefix_len_ptr_v<Params>) {
      maybe_prefix_len_ptr = params.maybe_prefix_len_ptr;
    }
    uint16_t* maybe_token_pos_in_items_ptr = nullptr;
    if constexpr (has_maybe_token_pos_in_items_ptr_v<Params>) {
      maybe_token_pos_in_items_ptr = params.maybe_token_pos_in_items_ptr;
    }
    uint32_t token_pos_in_items_len = 0;
    if constexpr (has_token_pos_in_items_len_v<Params>) {
      token_pos_in_items_len = params.token_pos_in_items_len;
    }
    uint16_t* maybe_max_item_len_ptr = nullptr;
    if constexpr (has_maybe_max_item_len_ptr_v<Params>) {
      maybe_max_item_len_ptr = params.maybe_max_item_len_ptr;
    }

    static_assert(sizeof(DTypeQ) == 2);
    auto block = cg::this_thread_block();
    const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

    const uint32_t lane_idx = tid.x, warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
    if (block_valid_mask && !block_valid_mask[bx]) {
      return;
    }
    const uint32_t num_qo_heads = num_kv_heads * group_size;

    const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                   kv_tile_idx = kv_tile_indices[bx];
    auto smem = reinterpret_cast<uint8_t*>(&smem_storage);
    AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
    const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                   window_left = variant.window_left;
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

    const uint32_t kv_start_idx = sub_if_greater_or_zero(
        kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size, qo_len + window_left);
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len - kv_start_idx;
    const uint32_t chunk_start =
        partition_kv ? min(kv_tile_idx * max_chunk_size + kv_start_idx, kv_len) : kv_start_idx;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size + kv_start_idx, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][8];
    alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][8];
    DTypeQKAccum m[NUM_MMA_Q][2];
    float d[NUM_MMA_Q][2];
    float rope_freq[NUM_MMA_D_QK / 2][4];

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      const float rope_rcp_scale = params.rope_rcp_scale;
      const float rope_rcp_theta = params.rope_rcp_theta;
      init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, tid.x);
    }
    init_states<KTraits>(variant, o_frag, m, d);

    const uint32_t qo_packed_idx_base =
        (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
    smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
    const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

    DTypeQ* q_ptr_base =
        q + q_indptr[request_idx] * q_stride_n + (kv_head_idx * group_size) * q_stride_h;
    DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h
                                      : o + o_indptr[request_idx] * o_stride_n +
                                            (kv_head_idx * group_size) * o_stride_h;
    uint32_t q_smem_offset_r = qo_smem.get_permuted_offset<UPCAST_STRIDE_Q>(
        get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                q_stride_h, group_size, &qo_smem, tid);

    cp_async::commit_group();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      cp_async::wait_group<0>();
      block.sync();
      IdType* q_rope_offset = nullptr;
      if constexpr (has_maybe_q_rope_offset_v<Params>) {
        q_rope_offset = params.maybe_q_rope_offset;
      }
      if (q_rope_offset == nullptr) {
        q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size,
                                             &qo_smem, &q_smem_offset_r, rope_freq, tid);
      } else {
        q_smem_inplace_apply_rotary_with_pos<KTraits>(
            qo_packed_idx_base, q_rope_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, tid);
      }
      block.sync();
    }

    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
    size_t thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_COL / 2 / NUM_WARPS_Q];

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + 8 * (lane_idx / 16) +
                     lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
             k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL),
             v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
                 warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
                 lane_idx % KV_THR_LAYOUT_COL);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t packed_page_iter_base =
        paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
    for (uint32_t i = 0;
         i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                    lane_idx / KV_THR_LAYOUT_COL +
                                    KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
    }
    page_produce_kv<false, KTraits>(&smem_storage, &k_smem_offset_w, paged_kv.k_data, 0,
                                    thr_local_kv_offset, chunk_size, warp_idx, lane_idx);
    cp_async::commit_group();
    page_produce_kv<true, KTraits>(&smem_storage, &v_smem_offset_w, paged_kv.v_data, 0,
                                   thr_local_kv_offset, chunk_size, warp_idx, lane_idx);
    cp_async::commit_group();

    uint32_t num_iterations_prefix;
    uint32_t num_iterations_mask;
    uint32_t num_iterations = 0;

    if constexpr (MASK_MODE != MaskMode::kMultiItemScoring) {
      num_iterations = ceil_div(
          (MASK_MODE == MaskMode::kCausal
               ? min(chunk_size,
                     sub_if_greater_or_zero(
                         kv_len - qo_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size),
                         chunk_start))
               : chunk_size),
          CTA_TILE_KV);
    } else if constexpr (MASK_MODE == MaskMode::kMultiItemScoring) {
      num_iterations_prefix = ceil_div(
          min(min(chunk_size,
                  sub_if_greater_or_zero(
                      kv_len - qo_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size),
                      chunk_start)),
              sub_if_greater_or_zero(__ldg(maybe_prefix_len_ptr + request_idx), chunk_start)),
          CTA_TILE_KV);
      num_iterations_mask =
          max(min(chunk_size,
                  sub_if_greater_or_zero(
                      sub_if_greater_or_zero(
                          kv_len - qo_len + ceil_div((qo_tile_idx * CTA_TILE_Q), group_size),
                          __ldg(maybe_max_item_len_ptr + request_idx)),
                      chunk_start)) /
                  (CTA_TILE_KV),
              num_iterations_prefix);

      num_iterations = max(
          num_iterations_mask,
          ceil_div(min(chunk_size,
                       sub_if_greater_or_zero(
                           kv_len - qo_len + ceil_div(((qo_tile_idx + 1) * CTA_TILE_Q), group_size),
                           chunk_start)),
                   CTA_TILE_KV));
    }

    const uint32_t window_iteration = ceil_div(
        sub_if_greater_or_zero(kv_len + ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size),
                               qo_len + window_left + chunk_start),
        CTA_TILE_KV);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal || MASK_MODE == MaskMode::kMultiItemScoring
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len + ceil_div((qo_tile_idx * CTA_TILE_Q), group_size) - qo_len,
                       chunk_start))
             : chunk_size) /
        CTA_TILE_KV;

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations;
         iter = (MASK_MODE == MaskMode::kMultiItemScoring)
                    ? ((iter + 1 == num_iterations_prefix) ? num_iterations_mask : (iter + 1))
                    : (iter + 1)) {
      const uint32_t prefetch_skip_step =
          (MASK_MODE == MaskMode::kMultiItemScoring)
              ? ((iter + 1 == num_iterations_prefix) ? (num_iterations_mask - num_iterations_prefix)
                                                     : 0)
              : 0;
      packed_page_iter_base += (1 + prefetch_skip_step) * CTA_TILE_KV;
#pragma unroll
      for (uint32_t i = 0;
           i < NUM_MMA_KV * (SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 2) / NUM_WARPS_Q; ++i) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * KV_THR_LAYOUT_ROW +
                                      lane_idx / KV_THR_LAYOUT_COL +
                                      KV_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                  page_iter, entry_idx);
        thr_local_kv_offset[i] = paged_kv.protective_get_kv_offset(
            page_iter, kv_head_idx, entry_idx,
            (lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV>(), last_indptr);
      }
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<KTraits>(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                chunk_start + iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq, tid);
        block.sync();
      }

      // compute attention score
      compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);
      uint32_t kv_idx_base =
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>(tid.z)) * NUM_MMA_KV * 16;
      logits_transform<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                                kv_idx_base, qo_len, kv_len, group_size, s_frag, tid, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom) {
        logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                             kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                             kv_head_idx);
      } else {
        if constexpr (MASK_MODE != MaskMode::kMultiItemScoring) {
          if (iter >= mask_iteration || iter < window_iteration) {
            logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                                 kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                                 kv_head_idx);
          }
        } else if constexpr (MASK_MODE == MaskMode::kMultiItemScoring) {
          if (iter + 1 >= num_iterations_prefix) {
            logits_mask_multi_item_scoring<KTraits>(
                params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base, kv_idx_base, qo_len,
                kv_len, window_left, chunk_end, group_size, s_frag,
                __ldg(maybe_prefix_len_ptr + request_idx),
                maybe_token_pos_in_items_ptr + request_idx * token_pos_in_items_len, tid.x,
                kv_head_idx);
          } else {
            if (iter >= mask_iteration || iter < window_iteration) {
              logits_mask<KTraits>(params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
                                   kv_idx_base, qo_len, kv_len, chunk_end, group_size, s_frag, tid,
                                   kv_head_idx);
            }
          }
        }
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      block.sync();
      page_produce_kv<false, KTraits>(&smem_storage, &k_smem_offset_w, paged_kv.k_data,
                                      (iter + 1) * CTA_TILE_KV, thr_local_kv_offset, chunk_size,
                                      warp_idx, lane_idx);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, KTraits>(&smem_storage, &v_smem_offset_w, paged_kv.v_data,
                                     (iter + 1) * CTA_TILE_KV, thr_local_kv_offset, chunk_size,
                                     warp_idx, lane_idx);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);

    // threadblock synchronization
    threadblock_sync_mdo_states<KTraits>(o_frag, &smem_storage, m, d, warp_idx, lane_idx, tid);

    const uint32_t num_kv_chunks =
        ceil_div(min(kv_len_safe, window_left + CTA_TILE_Q), kv_chunk_size);

    // transform output
    transform_output<KTraits, Params>(params, variant, o_frag, m, d, /*batch_idx=*/request_idx,
                                      kv_tile_idx, qo_packed_idx_base, warp_idx, lane_idx,
                                      kv_head_idx, group_size);

    // write_back
    write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                              /*o_stride_n=*/
                              partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                              /*o_stride_h=*/o_stride_h, group_size, tid);

    // write lse
    if constexpr (variant.use_softmax) {
      if (lse != nullptr) {
        if (get_warp_idx_kv<KTraits>(tid.z) == 0) {
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              uint32_t q, r;
              group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + mma_q * 16, q, r);
              const uint32_t qo_head_idx = kv_head_idx * group_size + r;
              const uint32_t qo_idx = q;
              if (qo_idx < qo_upper_bound) {
                if (partition_kv) {
                  lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) *
                          num_qo_heads +
                      qo_head_idx] = math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                } else {
                  lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                      math::ptx_log2(d[mma_q][j]) + float(m[mma_q][j]);
                }
              }
            }
          }
        }
      }
    }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif

#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, bool enable_pdl,
                                                    cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithRaggedKVCacheKernel<KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }

      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        void* args[] = {(void*)&params};
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // partition kv
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        void* args[] = {(void*)&params};
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
              num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        }
      }
    }
  });
  return cudaSuccess;
}

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int num_ctas_per_sm =
      max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_mma_kv_reg =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q);
  const uint32_t max_num_mma_kv_smem =
      (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits =
        KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                     NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
                     DTypeQKAccum, typename Params::IdType, AttentionVariant>;
    if constexpr (KTraits::IsInvalid()) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=" << NUM_MMA_Q
              << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
              << " NUM_MMA_KV=" << NUM_MMA_KV << " NUM_WARPS_Q=" << NUM_WARPS_Q
              << " NUM_WARPS_KV=" << NUM_WARPS_KV
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      FLASHINFER_ERROR(err_msg.str());
    } else {
      size_t smem_size = sizeof(typename KTraits::SharedStorage);
      auto kernel = BatchPrefillWithPagedKVCacheKernel<KTraits, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }

      if (tmp_v == nullptr) {
        // do not partition kv
        params.partition_kv = false;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows,
              params.total_num_rows, num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.merge_indptr, o, params.max_total_num_rows, params.total_num_rows,
              num_qo_heads, HEAD_DIM_VO, enable_pdl, stream));
        }
      }
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
