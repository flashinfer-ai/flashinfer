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
#ifndef FLASHINFER_MLA_HOPPER_CUH_
#define FLASHINFER_MLA_HOPPER_CUH_
#include <cooperative_groups.h>

#include <cstdint>
#include <sstream>

#include "hopper.cuh"
#include "mla.cuh"
#include "mla_params.cuh"
#include "prefill.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

namespace mla {

namespace hopper {

enum class NamedBarriers { kBarrierQ = 1, kBarrierO = 2, kConsumerSync = 3 };

__device__ __forceinline__ void barrier_arrive(int num_threads, NamedBarriers barrier) {
  cutlass::arch::NamedBarrier::arrive(num_threads, static_cast<int>(barrier));
}

__device__ __forceinline__ void barrier_sync(int num_threads, NamedBarriers barrier) {
  cutlass::arch::NamedBarrier::sync(num_threads, static_cast<int>(barrier));
}

template <typename MainloopPipeline, uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV,
          uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeQ, typename DTypeKV,
          typename DTypeO>
struct HopperSharedStorageQKVO {
  struct {
    struct {
      alignas(16) DTypeQ nope[CTA_TILE_Q * HEAD_DIM_CKV];
      alignas(16) DTypeQ pe[CTA_TILE_Q * HEAD_DIM_KPE];
    } q_smem;
    union {
      struct {
        alignas(16) DTypeKV ckv[CTA_TILE_KV * HEAD_DIM_CKV];
        union {
          alignas(16) DTypeKV kpe[CTA_TILE_KV * HEAD_DIM_KPE];
          alignas(16) DTypeKV p[CTA_TILE_Q * CTA_TILE_KV];
        };
      };
      alignas(16) DTypeO o[(CTA_TILE_Q / 2) * HEAD_DIM_CKV];
    } kv_o_smem[NUM_STAGES];
    union {
      alignas(16) float m_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
      alignas(16) float d_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
    };
  };

  typename MainloopPipeline::SharedStorage pipeline_q, pipeline_kv;
};

template <bool CAUSAL_, uint32_t NUM_STAGES_, uint32_t HEAD_DIM_CKV_, uint32_t HEAD_DIM_KPE_,
          uint32_t CTA_TILE_Q_, uint32_t CTA_TILE_KV_, typename DTypeQ_, typename DTypeKV_,
          typename DTypeO_, typename IdType_>
struct HopperKernelTraits
    : KernelTraits<CAUSAL_, NUM_STAGES_, /*QK_SHARD_=*/false, HEAD_DIM_CKV_, HEAD_DIM_KPE_,
                   CTA_TILE_Q_, CTA_TILE_KV_, DTypeQ_, DTypeKV_, DTypeO_, IdType_> {
  static constexpr uint32_t NUM_THREADS = 384;
  static constexpr uint32_t NUM_COPY_THREADS = 128;
  static constexpr uint32_t NUM_MMA_THREADS = 256;
  static constexpr uint32_t NUM_REGS_S_FRAG = CTA_TILE_KV_ / 4;
  static constexpr uint32_t NUM_REGS_O_FRAG = HEAD_DIM_CKV_ / 4;
  static constexpr uint32_t NUM_REGS_P_FRAG = CTA_TILE_KV_ / 8;
  using MainloopPipeline = cutlass::PipelineAsync<NUM_STAGES_>;
  using SharedStorage =
      HopperSharedStorageQKVO<MainloopPipeline, NUM_STAGES_, CTA_TILE_Q_, CTA_TILE_KV_,
                              HEAD_DIM_CKV_, HEAD_DIM_KPE_, DTypeQ_, DTypeKV_, DTypeO_>;
};

template <typename KTraits>
__device__ __forceinline__ void init_states_(float* o_frag, float* m, float* d, float* o_scale) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] = 0.f;
  }

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m[j] = -math::inf;
    d[j] = 1.f;
    o_scale[j] = 1.f;
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeQ* q_nope,
    typename KTraits::DTypeQ* q_pe, const uint32_t q_nope_stride_n, const uint32_t q_nope_stride_h,
    const uint32_t q_pe_stride_n, const uint32_t q_pe_stride_h, const uint32_t q_len,
    const uint32_t packed_offset, const uint_fastdiv& num_heads) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem.nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem.pe);

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < 2; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t q, r;
      num_heads.divmod(packed_offset + lane_idx / 8 + (j + mma_q * 2) * 16 + warp_idx_in_wg * 4, q,
                       r);
      DTypeQ* q_nope_ptr = q_nope + q * q_nope_stride_n + r * q_nope_stride_h +
                           (lane_idx % 8) * upcast_size<DTypeQ>();
      DTypeQ* q_pe_ptr =
          q_pe + q * q_pe_stride_n + r * q_pe_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
      uint32_t q_smem_nope_offset_w =
          get_swizzle_offset<KTraits::SWIZZLE_MODE_Q_NOPE, UPCAST_STRIDE_Q_NOPE>(
              32 * mma_q + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);
      uint32_t q_smem_pe_offset_w =
          get_swizzle_offset<KTraits::SWIZZLE_MODE_Q_PE, UPCAST_STRIDE_Q_PE>(
              32 * mma_q + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        q_smem_nope.load_128b_async<SharedMemFillMode::kFillZero>(q_smem_nope_offset_w, q_nope_ptr,
                                                                  q < q_len);
        q_smem_nope_offset_w += 64;
        q_nope_ptr += 8 * upcast_size<DTypeQ>();
      }
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        q_smem_pe.load_128b_async<SharedMemFillMode::kFillZero>(q_smem_pe_offset_w, q_pe_ptr,
                                                                q < q_len);
        q_smem_pe_offset_w += 64;
        q_pe_ptr += 8 * upcast_size<DTypeQ>();
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_kv(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeKV* ckv,
    typename KTraits::DTypeKV* kpe, typename KTraits::IdType* indices, const uint32_t ckv_stride_n,
    const uint32_t ckv_stride_page, const uint32_t kpe_stride_n, const uint32_t kpe_stride_page,
    const uint32_t kv_bound, const uint32_t packed_block_iter_base, const uint_fastdiv& block_size,
    const uint32_t stage_idx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->kv_o_smem[stage_idx].ckv);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kv_o_smem[stage_idx].kpe);

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t q, r;

      block_size.divmod(
          packed_block_iter_base + lane_idx / 8 + (j + mma_kv * 2) * 16 + warp_idx_in_wg * 4, q, r);

      DTypeKV* ckv_ptr = ckv + (q < kv_bound ? indices[q] : 0) * ckv_stride_page +
                         r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr = kpe + (q < kv_bound ? indices[q] : 0) * kpe_stride_page +
                         r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      uint32_t ckv_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV>(
          32 * mma_kv + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);
      uint32_t kpe_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_KPE, UPCAST_STRIDE_KPE>(
          32 * mma_kv + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr,
                                                               q < kv_bound);
        ckv_smem_offset_w += 64;
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr,
                                                               q < kv_bound);
        kpe_smem_offset_w += 64;
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx, float* s_frag) {
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  auto desc_q_pe =
      make_smem_desc<KTraits::SWIZZLE_MODE_Q_PE, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_KPE * 16, typename KTraits::DTypeQ>(
          smem_storage->q_smem.pe);
  auto desc_k_pe =
      make_smem_desc<KTraits::SWIZZLE_MODE_KPE, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_KPE * 16, typename KTraits::DTypeKV>(
          smem_storage->kv_o_smem[stage_idx].kpe +
          (warp_group_idx - 1) * (KTraits::CTA_TILE_KV / 2) * KTraits::HEAD_DIM_KPE);
  using wgmma = WGMMA_ASYNC_SS<typename KTraits::DTypeKV, float, 64, KTraits::CTA_TILE_KV / 2, 16,
                               Major::K, Major::K, ScaleIn::One, ScaleIn::One>;

  warpgroup_fence_frag<KTraits::NUM_REGS_S_FRAG>(s_frag);
  warpgroup_arrive();
#pragma unroll
  for (uint32_t mma_d_pe = 0; mma_d_pe < KTraits::NUM_MMA_D_KPE; ++mma_d_pe) {
    if (mma_d_pe == 0) {
      wgmma::op</*init=*/true>(desc_q_pe, desc_k_pe, s_frag);
    } else {
      wgmma::op</*init=*/false>(desc_q_pe, desc_k_pe, s_frag);
    }
    if ((mma_d_pe + 1) % 4 == 0) {
      desc_q_pe += 64 - 6;
      desc_k_pe += 64 - 6;
    } else {
      desc_q_pe += 2;
      desc_k_pe += 2;
    }
  }
  auto desc_q_nope =
      make_smem_desc<KTraits::SWIZZLE_MODE_Q_NOPE, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_CKV * 16, typename KTraits::DTypeQ>(
          smem_storage->q_smem.nope);
  auto desc_ckv =
      make_smem_desc<KTraits::SWIZZLE_MODE_CKV, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_CKV * 16, typename KTraits::DTypeKV>(
          smem_storage->kv_o_smem[stage_idx].ckv +
          (warp_group_idx - 1) * (KTraits::CTA_TILE_KV / 2) * KTraits::HEAD_DIM_CKV);

#pragma unroll
  for (uint32_t mma_d_ckv = 0; mma_d_ckv < KTraits::NUM_MMA_D_CKV; ++mma_d_ckv) {
    wgmma::op</*init=*/false>(desc_q_nope, desc_ckv, s_frag);
    if ((mma_d_ckv + 1) % 4 == 0) {
      desc_q_nope += 64 - 6;
      desc_ckv += 64 - 6;
    } else {
      desc_q_nope += 2;
      desc_ckv += 2;
    }
  }

  warpgroup_commit_batch();
  warpgroup_fence_frag<KTraits::NUM_REGS_S_FRAG>(s_frag);
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx, float* o_frag) {
  barrier_sync(KTraits::NUM_MMA_THREADS, NamedBarriers::kConsumerSync);
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  auto desc_p = make_smem_desc<KTraits::SWIZZLE_MODE_P, /*leading_byte_offset=*/16,
                               /*stride_byte_offset=*/KTraits::CTA_TILE_KV * 16, KTraits::DTypeKV>(
      smem_storage->kv_o_smem[stage_idx].p);
  auto desc_ckv =
      make_smem_desc<KTraits::SWIZZLE_MODE_CKV, /*leading_byte_offset=*/KTraits::CTA_TILE_KV * 32,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_CKV * 16, KTraits::DTypeKV>(
          smem_storage->kv_o_smem[stage_idx].ckv +
          (warp_group_idx - 1) * 8 * (KTraits::HEAD_DIM_CKV / 2));
  warpgroup_fence_frag<KTraits::NUM_REGS_O_FRAG>(o_frag);
  warpgroup_arrive();
  using wgmma = WGMMA_ASYNC_SS<typename KTraits::DTypeKV, float, 64, KTraits::HEAD_DIM_CKV / 2, 16,
                               Major::K, Major::MN, ScaleIn::One, ScaleIn::One>;

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
    wgmma::op</*init=*/false>(desc_p, desc_ckv, o_frag);
    desc_p += 2;
    desc_ckv += 1024;
  }
  warpgroup_commit_batch();
  warpgroup_fence_frag<KTraits::NUM_REGS_O_FRAG>(o_frag);
}

template <typename KTraits>
__device__ __forceinline__ void logits_mask_(const uint32_t qo_packed_idx_base,
                                             const uint32_t kv_idx_base, const uint32_t qo_len,
                                             const uint32_t kv_len, const uint32_t kv_end,
                                             const uint_fastdiv num_heads, float* s_frag) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  uint32_t q[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    q[j] = (qo_packed_idx_base + warp_idx_in_wg * 16 + lane_idx / 4 + 8 * j) / num_heads;
  }

#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_S_FRAG; ++reg_id) {
    const uint32_t q_idx = q[(reg_id % 4) / 2],
                   kv_idx = kv_idx_base + (warp_group_idx - 1) * (NUM_MMA_KV / 2) * 16 +
                            2 * (lane_idx % 4) + 8 * (reg_id / 4) + reg_id % 2;
    const bool mask = (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                                         : kv_idx >= kv_end));
    s_frag[reg_id] = (mask) ? s_frag[reg_id] : (KTraits::MaskFillValue);
  }
}

template <typename KTraits>
__device__ __forceinline__ void rescale_o_(float* o_scale, float* o_frag) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] *= o_scale[(reg_id % 4) / 2];
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_md_(typename KTraits::SharedStorage* smem_storage,
                                           typename KTraits::AttentionVariant variant,
                                           float* s_frag, float* m, float* d, float* o_scale) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  float m_prev[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m_prev[j] = m[j];
#pragma unroll
    for (uint32_t k = 0; k < KTraits::CTA_TILE_KV / 16; ++k) {
      float m_local = max(s_frag[k * 4 + j * 2 + 0], s_frag[k * 4 + j * 2 + 1]);
      m[j] = max(m[j], m_local);
    }
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
    if (lane_idx % 4 == 0) {
      smem_storage->m_wg[warp_group_idx - 1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = m[j];
    }
  }

  barrier_sync(KTraits::NUM_MMA_THREADS, NamedBarriers::kConsumerSync);

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m[j] = max(smem_storage->m_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4],
               smem_storage->m_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4]);
    o_scale[j] = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
    float d_local = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < KTraits::CTA_TILE_KV / 16; ++k) {
      s_frag[k * 4 + j * 2 + 0] =
          math::ptx_exp2(s_frag[k * 4 + j * 2 + 0] * sm_scale - m[j] * sm_scale);
      s_frag[k * 4 + j * 2 + 1] =
          math::ptx_exp2(s_frag[k * 4 + j * 2 + 1] * sm_scale - m[j] * sm_scale);

      d_local += s_frag[k * 4 + j * 2 + 0] + s_frag[k * 4 + j * 2 + 1];
    }
    d[j] = d[j] * o_scale[j] + d_local;
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_p_rmem_smem(typename KTraits::SharedStorage* smem_storage,
                                                  const uint32_t stage_idx, uint32_t* p_frag) {
  static constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kv_o_smem[stage_idx].p);
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
    uint32_t p_smem_offset_w =
        get_swizzle_offset<KTraits::SWIZZLE_MODE_P, KTraits::UPCAST_STRIDE_P>(
            warp_idx_in_wg * 16 + lane_idx % 16,
            (warp_group_idx - 1) * NUM_MMA_KV + mma_kv * 2 + lane_idx / 16);
    p_smem.stmatrix_m8n8x4(p_smem_offset_w, p_frag + mma_kv * 4);
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d_(typename KTraits::SharedStorage* smem_storage,
                                             float* o_frag, float* m, float* d) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d[j] += __shfl_xor_sync(0x11111111, d[j], 0x2);
    d[j] += __shfl_xor_sync(0x11111111, d[j], 0x1);
    if (lane_idx % 4 == 0) {
      smem_storage->d_wg[warp_group_idx - 1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = d[j];
    }
  }

  barrier_sync(KTraits::NUM_MMA_THREADS, NamedBarriers::kConsumerSync);

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d[j] = smem_storage->d_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] +
           smem_storage->d_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];
  }

  float d_rcp[2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d_rcp[j] = (m[j] != -math::inf) ? math::ptx_rcp(d[j]) : 0.f;
  }

#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] = o_frag[reg_id] * d_rcp[(reg_id % 4) / 2];
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o(typename KTraits::SharedStorage* smem_storage,
                                        const uint32_t stage_counter,
                                        typename KTraits::DTypeO* final_o, float* final_lse,
                                        typename KTraits::DTypeO* partial_o, float* partial_lse,
                                        float(*o_frag), float* m, float* d,
                                        const uint32_t o_stride_n, const uint32_t o_stride_h,
                                        const uint32_t q_len, const uint32_t packed_offset,
                                        const uint_fastdiv& num_heads) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O = KTraits::UPCAST_STRIDE_FINAL_O;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  smem_t<KTraits::SWIZZLE_MODE_O> o_smem[2];
  o_smem[0] = smem_storage->kv_o_smem[stage_counter % KTraits::NUM_STAGES].o;
  o_smem[1] = smem_storage->kv_o_smem[(stage_counter + 1) % KTraits::NUM_STAGES].o;

  // step 0. rmem to smem
#pragma unroll
  for (uint32_t k = 0; k < HEAD_DIM_CKV / 32; ++k) {
    uint32_t o_frag_f16[8 / 2];
    vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, &o_frag[k * 8]);
    uint32_t o_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
        (warp_idx_in_wg % 2) * 16 + lane_idx % 16,
        (warp_group_idx - 1) * NUM_MMA_D_CKV + k * 2 + lane_idx / 16);
    o_smem[warp_idx_in_wg / 2].template stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
  }

  if (partial_o != nullptr) {
    // NOTE(Zihao): o_smem is not used if write to partial_o, and we can avoid the barrier
    // write to partial_o

#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 8) / num_heads;
      DTypeO* o_partial_ptr =
          partial_o +
          ((blockIdx.x * 4 + warp_idx_in_wg) * 16 + 4 * j + lane_idx / 8) * HEAD_DIM_CKV +
          (warp_group_idx - 1) * (HEAD_DIM_CKV / 2) + (lane_idx % 8) * upcast_size<DTypeO>();
      uint32_t o_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
          (warp_idx_in_wg % 2) * 16 + 4 * j + lane_idx / 8,
          (warp_group_idx - 1) * NUM_MMA_D_CKV + lane_idx % 8);
#pragma unroll
      for (uint32_t k = 0; k < HEAD_DIM_CKV / 128; ++k) {
        if (q_idx < q_len) {
          o_smem[warp_idx_in_wg / 2].template store_128b(o_smem_offset_w, o_partial_ptr);
        }
        o_partial_ptr += 8 * upcast_size<DTypeO>();
        o_smem_offset_w += 64;
      }
    }
    barrier_arrive(KTraits::NUM_COPY_THREADS + KTraits::NUM_MMA_THREADS, NamedBarriers::kBarrierO);

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 4) / num_heads;
      if (lane_idx % 4 == 0 && q_idx < q_len) {
        partial_lse[(blockIdx.x * 4 + warp_idx_in_wg) * 16 + 8 * j + lane_idx / 4] =
            math::ptx_log2(d[j]) + float(m[j]);
      }
    }
  } else {
    // write to final_o

// step 1. smem to gmem
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t q, r;
      num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 8, q, r);
      DTypeO* o_final_ptr = final_o + q * o_stride_n + r * o_stride_h +
                            (warp_group_idx - 1) * (HEAD_DIM_CKV / 2) +
                            (lane_idx % 8) * upcast_size<DTypeO>();
      uint32_t o_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
          (warp_idx_in_wg % 2) * 16 + 4 * j + lane_idx / 8,
          (warp_group_idx - 1) * NUM_MMA_D_CKV + lane_idx % 8);
#pragma unroll
      for (uint32_t k = 0; k < HEAD_DIM_CKV / 128; ++k) {
        if (q < q_len) {
          o_smem[warp_idx_in_wg / 2].template store_128b(o_smem_offset_w, o_final_ptr);
        }
        o_final_ptr += 8 * upcast_size<DTypeO>();
        o_smem_offset_w += 64;
      }
    }
    barrier_arrive(KTraits::NUM_COPY_THREADS + KTraits::NUM_MMA_THREADS, NamedBarriers::kBarrierO);

    if (final_lse) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 4, q, r);
        if (lane_idx % 4 == 0 && q < q_len) {
          final_lse[q * num_heads + r] = math::ptx_log2(d[j]) + float(m[j]);
        }
      }
    }
  }
}

template <typename Params>
__device__ __forceinline__ auto get_block_coord(const Params& params, const uint32_t work_idx) {
  return std::tuple(params.q_indptr[work_idx], params.kv_indptr[work_idx],
                    params.partial_indptr[work_idx], params.q_len[work_idx],
                    params.kv_len[work_idx], params.q_start[work_idx], params.kv_start[work_idx],
                    params.kv_end[work_idx]);
}

template <typename KTraits>
__device__ __forceinline__ convert_s_to_p(float* s_frag, uint32_t* p_frag) {
#pragma unroll
  for (uint32_t i = 0; i < KTraits::NUM_REGS_S_FRAG / 8; ++i) {
    vec_cast<typename KTraits::DTypeKV, float>::cast<8>(
        ((typename KTraits::DTypeKV*)p_frag) + i * 8, s_frag + i * 8);
  }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPageAttentionHopperKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);

  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = KTraits::SWIZZLE_MODE_Q_NOPE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_PE = KTraits::SWIZZLE_MODE_Q_PE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_CKV = KTraits::SWIZZLE_MODE_CKV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KPE = KTraits::SWIZZLE_MODE_KPE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  [[maybe_unused]] constexpr uint32_t NUM_COPY_THREADS = KTraits::NUM_COPY_THREADS;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_THREADS = KTraits::NUM_MMA_THREADS;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;

  DTypeQ* q_nope = params.q_nope;
  DTypeQ* q_pe = params.q_pe;
  DTypeKV* ckv = params.ckv;
  DTypeKV* kpe = params.kpe;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_nope_stride_n = params.q_nope_stride_n;
  const uint32_t q_nope_stride_h = params.q_nope_stride_h;
  const uint32_t q_pe_stride_n = params.q_pe_stride_n;
  const uint32_t q_pe_stride_h = params.q_pe_stride_h;
  const uint32_t ckv_stride_page = params.ckv_stride_page;
  const uint32_t ckv_stride_n = params.ckv_stride_n;
  const uint32_t kpe_stride_page = params.kpe_stride_page;
  const uint32_t kpe_stride_n = params.kpe_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;

  const uint32_t lane_predicate = cute::elect_one_sync();
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx = cutlass::canonical_warp_idx();

  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  PipelineParams pipeline_params;
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  MainloopPipeline pipeline_q(smem_storage.pipeline_q, pipeline_params);
  MainloopPipeline pipeline_kv(smem_storage.pipeline_kv, pipeline_params);

  __syncthreads();

  if (warp_group_idx == 0) {
    // producer
    cutlass::arch::warpgroup_reg_dealloc<56>();
    const uint32_t warp_idx_in_warpgroup = __shfl_sync(0xffffffff, warp_idx % 4, 0);
    PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_write_kv = cutlass::make_producer_start_state<MainloopPipeline>();

    bool is_first_tile = true;

#pragma unroll 1
    for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
         ++work_idx) {
      auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end] =
          get_block_coord(params, work_idx);
      const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
      const uint32_t qo_upperbound =
          min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

      uint32_t kv_bound = kv_indptr + (kv_len + block_size - 1) / block_size;
      int kv_tile_idx =
          ceil_div(
              (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                      : kv_end),
              CTA_TILE_KV) -
          1 - (kv_start / CTA_TILE_KV);

      const uint32_t block_iter_base = kv_indptr * block_size + kv_start;

      pipeline_kv.producer_acquire(smem_pipe_write_kv);
      load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                       kpe_stride_n, kpe_stride_page, kv_bound,
                       block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                       smem_pipe_write_kv.index());
      pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
      kv_tile_idx -= 1;
      ++smem_pipe_write_kv;

      if (!is_first_tile) {
        barrier_sync(KTraits::NUM_COPY_THREADS + KTraits::NUM_MMA_THREADS,
                     NamedBarriers::kBarrierQ);
      }

      pipeline_q.producer_acquire(smem_pipe_write_q);
      load_q<KTraits>(&smem_storage, q_nope + q_indptr * q_nope_stride_n,
                      q_pe + q_indptr * q_pe_stride_n, q_nope_stride_n, q_nope_stride_h,
                      q_pe_stride_n, q_pe_stride_h, qo_upperbound, qo_packed_idx_base,
                      params.num_heads);
      pipeline_q.producer_commit(smem_pipe_write_q, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_q;

      if (kv_tile_idx >= 1) {
        pipeline_kv.producer_acquire(smem_pipe_write_kv);
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, kv_bound,
                         block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                         smem_pipe_write_kv.index());
        pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_kv;
        --kv_tile_idx;
      }

      if (!is_first_tile) {
        barrier_sync(KTraits::NUM_COPY_THREADS + KTraits::NUM_MMA_THREADS,
                     NamedBarriers::kBarrierO);
      }

#pragma unroll 1
      for (; kv_tile_idx >= 0; --kv_tile_idx) {
        pipeline_kv.producer_acquire(smem_pipe_write_kv);
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, kv_bound,
                         block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                         smem_pipe_write_kv.index());
        pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_kv;
      }

      is_first_tile = false;
    }
  } else {
    // consumer
    cutlass::arch::warpgroup_reg_alloc<224>();
    const uint32_t warp_idx_in_warpgroup = __shfl_sync(0xffffffff, warp_idx % 4, 0);

    PipelineState smem_pipe_read_q;
    PipelineState smem_pipe_read_kv;
    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    float s_frag[KTraits::NUM_REGS_S_FRAG];
    uint32_t p_frag[KTraits::NUM_REGS_P_FRAG];
    alignas(16) float o_frag[KTraits::NUM_REGS_O_FRAG];
    float m[2];
    float d[2];
    float o_scale[2];

#pragma unroll 1
    for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
         ++work_idx) {
      auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end] =
          get_block_coord(params, work_idx);
      const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
      const uint32_t qo_upperbound =
          min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

      init_states_<KTraits>(o_frag, m, d, o_scale);

      int kv_tile_idx =
          ceil_div(
              (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                      : kv_end),
              CTA_TILE_KV) -
          1 - (kv_start / CTA_TILE_KV);

      int mask_tile_idx =
          (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
              CTA_TILE_KV -
          (kv_start / CTA_TILE_KV);

      consumer_wait(pipeline_q, smem_pipe_read_q);
      consumer_wait(pipeline_kv, smem_pipe_read_kv);
      compute_mla_qk<KTraits>(&smem_storage, smem_pipe_read_kv.index(), s_frag);
      warpgroup_wait<0>();
      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);
      update_md_<KTraits>(&smem_storage, variant, s_frag, m, d, o_scale);
      convert_s_to_p<KTraits>(s_frag, p_frag);
      write_p_rmem_smem<KTraits>(&smem_storage, smem_pipe_read_kv.index(), p_frag);

      // loop with mask
#pragma unroll 1
      for (; kv_tile_idx > mask_tile_idx && kv_tile_idx > 1; --kv_tile_idx) {
        auto smem_pipe_read_kv_cur = smem_pipe_read_kv;
        ++smem_pipe_read_kv;
        consumer_wait(pipeline_kv, smem_pipe_read_kv);
        compute_mla_qk<KTraits>(&smem_storage, smem_pipe_read_kv.index(), s_frag);
        rescale_o_<KTraits>(o_scale, o_frag);
        compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv_cur.index(), o_frag);
        warpgroup_wait<1>();
        logits_mask_<KTraits>(qo_packed_idx_base, kv_start + (kv_tile_idx - 1) * CTA_TILE_KV, q_len,
                              kv_len, kv_end, num_heads, s_frag);
        update_md_<KTraits>(&smem_storage, variant, s_frag, m, d, o_scale);
        convert_s_to_p<KTraits>(s_frag, p_frag);
        warpgroup_wait<0>();
        pipeline_kv.consumer_release(smem_pipe_read_kv_cur);
        write_p_rmem_smem<KTraits>(&smem_storage, smem_pipe_read_kv.index(), p_frag);
      }

      // loop without mask
#pragma unroll 2
      for (; kv_tile_idx > NUM_STAGES; --kv_tile_idx) {
        auto smem_pipe_read_kv_cur = smem_pipe_read_kv;
        ++smem_pipe_read_kv;
        consumer_wait(pipeline_kv, smem_pipe_read_kv);
        compute_mla_qk<KTraits>(&smem_storage, smem_pipe_read_kv.index(), s_frag);
        rescale_o_<KTraits>(o_scale, o_frag);
        compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv_cur.index(), o_frag);
        warpgroup_wait<1>();
        update_md_<KTraits>(&smem_storage, variant, s_frag, m, d, o_scale);
        convert_s_to_p<KTraits>(s_frag, p_frag);
        warpgroup_wait<0>();
        pipeline_kv.consumer_release(smem_pipe_read_kv_cur);
        write_p_rmem_smem<KTraits>(&smem_storage, smem_pipe_read_kv.index(), p_frag);
      }

      // last tiles
#pragma unroll
      for (; kv_tile_idx > 0; --kv_tile_idx) {
        auto smem_pipe_read_kv_cur = smem_pipe_read_kv;
        ++smem_pipe_read_kv;
        consumer_wait(pipeline_kv, smem_pipe_read_kv);
        compute_mla_qk<KTraits>(&smem_storage, smem_pipe_read_kv.index(), s_frag);
        rescale_o_<KTraits>(o_scale, o_frag);
        compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv_cur.index(), o_frag);
        warpgroup_wait<1>();
        logits_mask_<KTraits>(qo_packed_idx_base, kv_start + (kv_tile_idx - 1) * CTA_TILE_KV, q_len,
                              kv_len, kv_end, num_heads, s_frag);
        update_md_<KTraits>(&smem_storage, variant, s_frag, m, d, o_scale);
        convert_s_to_p<KTraits>(s_frag, p_frag);
        warpgroup_wait<0>();
        pipeline_kv.consumer_release(smem_pipe_read_kv_cur);
        write_p_rmem_smem<KTraits>(&smem_storage, smem_pipe_read_kv.index(), p_frag);
      }

      barrier_arrive(KTraits::NUM_COPY_THREADS + KTraits::NUM_MMA_THREADS,
                     NamedBarriers::kBarrierQ);
      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;

      rescale_o_<KTraits>(o_scale, o_frag);
      compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv.index(), o_frag);
      warpgroup_wait<0>();
      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;

      normalize_d_<KTraits>(&smem_storage, o_frag, m, d);
      finalize_m_<KTraits>(variant, m);

      write_o<KTraits>(
          &smem_storage, smem_pipe_read_kv.count() - 2, final_o + q_indptr * o_stride_n,
          final_lse ? final_lse + q_indptr * num_heads : nullptr,
          (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
          (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_frag, m, d, o_stride_n,
          o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads);
    }
  }

  auto grid = cg::this_grid();
  grid.sync();

  __syncthreads();
  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, partial_o, partial_lse, final_o, final_lse, o_stride_n,
      o_stride_h, num_heads);
}

}  // namespace hopper

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
cudaError_t BatchMLAPageAttentionHopper(Params params, uint32_t num_blks_x, uint32_t num_blks_y,
                                        cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;

  // get GPU shared memory size
  int device;
  int smem_limit_per_sm;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  constexpr uint32_t NUM_STAGES = 4;
  constexpr uint32_t CTA_TILE_Q = 64;
  constexpr uint32_t CTA_TILE_KV = 32;

  using KTraits =
      hopper::HopperKernelTraits<CAUSAL, NUM_STAGES, HEAD_DIM_CKV, HEAD_DIM_KPE, CTA_TILE_Q,
                                 CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(KTraits::NUM_THREADS);
  size_t smem_size = sizeof(typename KTraits::SharedStorage);

  auto kernel = hopper::BatchMLAPageAttentionHopperKernel<KTraits, Params>;
  void* args[] = {(void*)&params};

  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  FLASHINFER_CUDA_CALL(
      cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

  return cudaSuccess;
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_HOPPER_CUH_
