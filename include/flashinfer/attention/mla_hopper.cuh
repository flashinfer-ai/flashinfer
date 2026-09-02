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
#include <cuda/std/limits>
#include <sstream>

#include "hopper.cuh"
#include "mla.cuh"
#include "mla_params.cuh"
#include "prefill.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

namespace mla {

namespace hopper {

enum class ProfileEventType {
  kIssueLoadQ = 0U,
  kIssueLoadKV = 1U,
  kWriteO = 2U,
  kSoftmaxUpdate = 3U,
  kGemmQK = 4U,
  kGemmPV = 5U,
  kRescaleO = 6U,
  kWritePSmem = 7U,
  kSplitK = 8U,
};

enum class NamedBarriers {
  kOScaleReady = 1U,
  kBarrierO = 2U,
  kMDReady = 3U,
  // FP8 KV path only: 128-thread (consumer warpgroup) barriers used to bracket
  // the FP8->BF16 dequant of one KV tile. Producer warpgroup does not touch
  // these; cross-warpgroup ordering is provided by the existing kOScaleReady
  // (consumer arrives after QK -> producer reads BF16 staging in PV).
  kConsumerDequantBegin = 4U,
  kConsumerDequantEnd = 5U,
  // Swap-AB Q tiles only (see HopperKernelTraits): one 128-thread barrier per
  // warpgroup. The consumer uses its barrier to exchange per-warp softmax
  // partials through shared memory; both warpgroups use theirs to wait for
  // their own transposed O staging before copying rows out. cutlass reserves
  // the upper half of the 16 hardware barriers, so ids must stay below 8.
  kConsumerWarpgroup = 6U,
  kProducerWarpgroup = 7U,
};

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
  // FP8 KV path: store KV as FP8 (e4m3) in shmem, dequantize to BF16 in
  // dedicated staging buffers right before each WGMMA. WGMMA itself stays
  // BF16xBF16 because Hopper does not have a mixed BF16xFP8 wgmma instruction.
  // Must match HopperKernelTraits::USE_KV_REPACK exactly — same predicate
  // keeps the smem layout in sync with the kernel-side dequant.
  static constexpr bool USE_KV_REPACK = std::is_same_v<DTypeKV, __nv_fp8_e4m3>;
  // Per-stage KV data tile. `p` (softmax output) is always DTypeQ-typed so the
  // PV WGMMA can run as BF16xBF16 on both the BF16 KV and FP8 KV paths. On the
  // FP8 KV path the union with the FP8-typed `kpe` is sized for the larger
  // BF16 P, so the per-stage struct shrinks from 72KB (BF16) to 40KB (FP8).
  struct PerStageKV {
    alignas(16) DTypeKV ckv[CTA_TILE_KV * HEAD_DIM_CKV];
    union {
      alignas(16) DTypeKV kpe[CTA_TILE_KV * HEAD_DIM_KPE];
      alignas(16) DTypeQ p[CTA_TILE_Q * CTA_TILE_KV];
    };
  };
  // The output writeback overlay `o` lives at the top level (not per-stage):
  // it is written only after every stage has been consumed, so it can safely
  // share memory with the entire per-stage data path. This is critical for
  // the FP8 path, which needs ~72KB of shared BF16 dequant staging on top of
  // ~80KB of per-stage data; replicating an o overlay per stage would blow
  // the 228KB/SM budget.
  struct {
    struct {
      struct {
        alignas(16) DTypeQ nope[CTA_TILE_Q * HEAD_DIM_CKV];
        alignas(16) DTypeQ pe[CTA_TILE_Q * HEAD_DIM_KPE];
      } q_smem;
      union {
        struct {
          PerStageKV kv_o_smem[NUM_STAGES];
          // FP8-only BF16 dequant staging, shared across stages. On the BF16
          // path these collapse to a single element via std::conditional_t.
          alignas(16) std::conditional_t<USE_KV_REPACK, DTypeQ[CTA_TILE_KV * HEAD_DIM_CKV],
                                         DTypeQ[1]> ckv_bf16;
          // When HEAD_DIM_KPE=0 (no-PE MLA) the FP8 KPE staging is unused but
          // must stay a valid object, so size it as at least one element.
          alignas(16) std::conditional_t<USE_KV_REPACK,
                                         DTypeQ[HEAD_DIM_KPE > 0 ? CTA_TILE_KV* HEAD_DIM_KPE : 1],
                                         DTypeQ[1]> kpe_bf16;
        };
        alignas(16) DTypeO o[CTA_TILE_Q * HEAD_DIM_CKV];
      };
      alignas(16) float o_scale[CTA_TILE_Q];
      alignas(16) float m[CTA_TILE_Q];
      alignas(16) float d[CTA_TILE_Q];
      // Swap-AB Q tiles reduce softmax statistics across the four warps of the
      // consumer warpgroup; each warp parks its per-column partials here.
      alignas(16) float md_partial[4][CTA_TILE_Q];
    };

    typename MainloopPipeline::SharedStorage pipeline_q, pipeline_kv;
  };
};

template <bool CAUSAL_, uint32_t NUM_STAGES_, uint32_t HEAD_DIM_CKV_, uint32_t HEAD_DIM_KPE_,
          uint32_t CTA_TILE_Q_, uint32_t CTA_TILE_KV_, typename DTypeQ_, typename DTypeKV_,
          typename DTypeO_, typename IdType_>
struct HopperKernelTraits
    : KernelTraits<CAUSAL_, NUM_STAGES_, /*QK_SHARD_=*/false, HEAD_DIM_CKV_, HEAD_DIM_KPE_,
                   CTA_TILE_Q_, CTA_TILE_KV_, DTypeQ_, DTypeKV_, DTypeO_, IdType_> {
  static constexpr uint32_t NUM_THREADS = 256;
  static constexpr uint32_t NUM_COPY_THREADS = 128;
  static constexpr uint32_t NUM_QK_THREADS = 128;
  // FP8 KV path: KV stored as FP8 e4m3 in shmem, dequantized to DTypeQ
  // before WGMMA. Match on the exact type (not sizeof==1) — other 1-byte
  // JIT dtypes (e.g. __nv_fp4x2_e2m1) have no compatible vec_cast.
  static constexpr bool USE_KV_REPACK = std::is_same_v<DTypeKV_, __nv_fp8_e4m3>;
  // FP8 KV swizzle / dequant layout is only correct for the supported MLA
  // dims; AOT/JIT must not instantiate other sizes.
  static_assert(!USE_KV_REPACK || HEAD_DIM_CKV_ == 512,
                "FP8 KV MLA path currently only supports HEAD_DIM_CKV=512");
  static_assert(!USE_KV_REPACK || HEAD_DIM_KPE_ == 0 || HEAD_DIM_KPE_ == 64,
                "FP8 KV MLA path currently only supports HEAD_DIM_KPE=0 (no PE) or 64");
  // Strides for the BF16 dequant staging buffer and the P buffer, both of
  // which are DTypeQ-typed regardless of DTypeKV.
  static constexpr uint32_t UPCAST_STRIDE_CKV_BF16 = HEAD_DIM_CKV_ / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE_BF16 = HEAD_DIM_KPE_ / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_P_BF16 = CTA_TILE_KV_ / upcast_size<DTypeQ_>();
  // Dtype-aware TMA load loop bounds. The existing FA2-shared KernelTraits hard
  // codes `NUM_MMA_D_* / 4` based on a BF16 K=16 MMA shape, which overshoots by
  // 2x for FP8 (one b128 holds 16 fp8 elems vs 8 bf16). For HEAD_DIM_KPE=64 on
  // FP8 the row only has 4 b128, so we additionally gate 4 of 8 lanes off.
  static constexpr uint32_t CKV_B128_PER_ROW = HEAD_DIM_CKV_ * sizeof(DTypeKV_) / 16;
  static constexpr uint32_t KPE_B128_PER_ROW = HEAD_DIM_KPE_ * sizeof(DTypeKV_) / 16;
  static constexpr uint32_t LANES_PER_ROW_CKV = (CKV_B128_PER_ROW >= 8) ? 8 : CKV_B128_PER_ROW;
  static constexpr uint32_t LANES_PER_ROW_KPE =
      KPE_B128_PER_ROW == 0 ? 1 : ((KPE_B128_PER_ROW >= 8) ? 8 : KPE_B128_PER_ROW);
  static constexpr uint32_t INNER_LOADS_CKV = CKV_B128_PER_ROW / LANES_PER_ROW_CKV;
  static constexpr uint32_t INNER_LOADS_KPE = KPE_B128_PER_ROW / LANES_PER_ROW_KPE;
  // Swizzle for the FP8 *raw* KPE shmem storage. When KPE_B128_PER_ROW < 8 (FP8
  // KPE with HEAD_DIM_KPE=64 only has 4 b128 per row), the k128B swizzle's
  // N=8 col group makes (row=K, col=c) collide with (row=K+4, col=c) at the
  // same shmem offset, corrupting load_kv writes. Switch the raw-FP8 KPE
  // layout to a k64B swizzle (N=4) so the row-group bits stay distinct.
  // BF16 path is unaffected (KPE_B128_PER_ROW=8, k128B already correct).
  static constexpr SwizzleMode SWIZZLE_MODE_KPE_RAW =
      (USE_KV_REPACK && KPE_B128_PER_ROW < 8) ? SwizzleMode::k64B : SwizzleMode::k128B;
  using MainloopPipeline = cutlass::PipelineAsync<NUM_STAGES_>;
  using SharedStorage =
      HopperSharedStorageQKVO<MainloopPipeline, NUM_STAGES_, CTA_TILE_Q_, CTA_TILE_KV_,
                              HEAD_DIM_CKV_, HEAD_DIM_KPE_, DTypeQ_, DTypeKV_, DTypeO_>;

  // Q tile layout. CTA_TILE_Q == 64 is the original layout: Q is the WGMMA-M
  // operand, so a decode work item with 8 or 16 packed query rows pads to 64
  // rows and wastes most of the tensor-core work. Narrower tiles swap the GEMM
  // operands (SWAP_AB) so the query rows land on WGMMA-N, whose granularity is 8:
  //
  //   S^T[kv, q]  = K[kv, d]   * Q^T[d, q]     A = K   (K-major),  B = Q   (K-major)
  //   O^T[d, q]  += V^T[d, kv] * P^T[kv, q]    A = V^T (MN-major), B = P^T (K-major)
  //
  // Both products read the same shared-memory tiles as the original layout;
  // only the descriptor roles and the accumulator layout change. A thread then
  // owns two kv (or d) rows and CTA_TILE_Q / 4 query columns, so the softmax
  // statistics m/d/o_scale are per column, and the reduction over kv needs one
  // cross-warp step through shared memory per KV tile.
  static constexpr bool SWAP_AB = CTA_TILE_Q_ < 64;
  static_assert(CTA_TILE_Q_ % 16 == 0 && CTA_TILE_Q_ <= 64, "unsupported Q tile");
  static_assert(!SWAP_AB || HEAD_DIM_CKV_ % 128 == 0,
                "swap-AB tiles split each warpgroup's half of HEAD_DIM_CKV into 64-row chunks");

  // Accumulator registers per thread. S is [CTA_TILE_Q, CTA_TILE_KV], or its
  // transpose when swapped. Each warpgroup owns HEAD_DIM_CKV / 2 columns of O;
  // swapped, that half is NUM_O_CHUNKS independent 64-row WGMMA-M chunks.
  static constexpr uint32_t NUM_REGS_S_FRAG = (SWAP_AB ? CTA_TILE_Q_ : CTA_TILE_KV_) / 2;
  static constexpr uint32_t NUM_REGS_P_FRAG = NUM_REGS_S_FRAG / 2;
  static constexpr uint32_t NUM_O_CHUNKS = SWAP_AB ? HEAD_DIM_CKV_ / 128 : 1;
  static constexpr uint32_t NUM_REGS_O_CHUNK = SWAP_AB ? CTA_TILE_Q_ / 2 : HEAD_DIM_CKV_ / 4;
  static constexpr uint32_t NUM_REGS_O_FRAG = NUM_O_CHUNKS * NUM_REGS_O_CHUNK;
  // Softmax states (m, d, o_scale) per thread: two rows, or CTA_TILE_Q / 4 columns.
  static constexpr uint32_t NUM_MD = SWAP_AB ? CTA_TILE_Q_ / 4 : 2;

  // Softmax state that accumulator register `reg` belongs to. The WGMMA
  // accumulator layout is the same for S and for every O chunk: register
  // 4 * i + 2 * j + c holds (row lane / 4 + 8 * j, column 8 * i + 2 * (lane % 4) + c).
  static __device__ __forceinline__ uint32_t md_index(uint32_t reg) {
    if constexpr (SWAP_AB) {
      const uint32_t r = reg % NUM_REGS_O_CHUNK;
      return 2 * (r / 4) + r % 2;
    } else {
      return (reg % 4) / 2;
    }
  }

  // Query column within the tile of softmax state `md` on lane `lane_idx` (swap-AB).
  static __device__ __forceinline__ uint32_t md_column(uint32_t md, uint32_t lane_idx) {
    return 8 * (md / 2) + 2 * (lane_idx % 4) + md % 2;
  }
};

template <typename KTraits>
__device__ __forceinline__ void init_states_(float* o_frag, float* m, float* d, float* o_scale) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] = 0.f;
  }

#pragma unroll
  for (uint32_t j = 0; j < KTraits::NUM_MD; ++j) {
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
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem.nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem.pe);

  // Each pass covers 16 packed rows: 4 rows per warp, 8 lanes per row.
#pragma unroll
  for (uint32_t pass = 0; pass < KTraits::CTA_TILE_Q / 16; ++pass) {
    const uint32_t row = pass * 16 + warp_idx_in_wg * 4 + lane_idx / 8;
    uint32_t q, r;
    num_heads.divmod(packed_offset + row, q, r);
    DTypeQ* q_nope_ptr =
        q_nope + q * q_nope_stride_n + r * q_nope_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
    DTypeQ* q_pe_ptr =
        q_pe + q * q_pe_stride_n + r * q_pe_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
    uint32_t q_smem_nope_offset_w =
        get_swizzle_offset<KTraits::SWIZZLE_MODE_Q_NOPE, UPCAST_STRIDE_Q_NOPE>(row, lane_idx % 8);
    uint32_t q_smem_pe_offset_w =
        get_swizzle_offset<KTraits::SWIZZLE_MODE_Q_PE, UPCAST_STRIDE_Q_PE>(row, lane_idx % 8);

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

template <typename KTraits>
__device__ __forceinline__ void prefetch_offset(
    const uint32_t packed_block_iter_base, const uint32_t packed_kv_bound,
    const uint32_t ckv_stride_page, const uint32_t ckv_stride_n, const uint32_t kpe_stride_page,
    const uint32_t kpe_stride_n, const uint_fastdiv& block_size, typename KTraits::IdType* indices,
    int64_t (*ckv_offset)[2], int64_t (*kpe_offset)[2]) {
  using DTypeKV = typename KTraits::DTypeKV;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t q, r;
      uint32_t packed_block_iter =
          packed_block_iter_base + lane_idx / 8 + (j + mma_kv * 2) * 16 + warp_idx_in_wg * 4;
      block_size.divmod(packed_block_iter, q, r);
      // Widen page index to int64_t before multiplying to avoid overflow.
      ckv_offset[mma_kv][j] =
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              ckv_stride_page +
          r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      kpe_offset[mma_kv][j] =
          static_cast<int64_t>(packed_block_iter < packed_kv_bound ? indices[q] : 0) *
              kpe_stride_page +
          r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
    }
  }
}

template <bool predicate, typename KTraits>
__device__ __forceinline__ void load_kv(typename KTraits::SharedStorage* smem_storage,
                                        typename KTraits::DTypeKV* ckv,
                                        typename KTraits::DTypeKV* kpe,
                                        const uint32_t packed_kv_bound,
                                        const uint32_t packed_block_iter_base,
                                        const uint32_t stage_idx, int64_t (*ckv_offset)[2],
                                        int64_t (*kpe_offset)[2]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->kv_o_smem[stage_idx].ckv);
  // Raw FP8 KPE storage uses SWIZZLE_MODE_KPE_RAW (k64B for FP8, k128B for
  // BF16). Mismatch with the WGMMA-side k128B layout in BF16 staging is fine
  // because the FP8 raw storage is only ever read by the consumer-side
  // repack_fp8_kv_to_bf16 helper that uses the same SWIZZLE_MODE_KPE_RAW.
  smem_t<KTraits::SWIZZLE_MODE_KPE_RAW> kpe_smem(smem_storage->kv_o_smem[stage_idx].kpe);

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t packed_block_iter =
          packed_block_iter_base + lane_idx / 8 + (j + mma_kv * 2) * 16 + warp_idx_in_wg * 4;

      DTypeKV* ckv_ptr = ckv + ckv_offset[mma_kv][j];
      DTypeKV* kpe_ptr = kpe + kpe_offset[mma_kv][j];
      uint32_t ckv_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV>(
          32 * mma_kv + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);
      uint32_t kpe_smem_offset_w =
          get_swizzle_offset<KTraits::SWIZZLE_MODE_KPE_RAW, UPCAST_STRIDE_KPE>(
              32 * mma_kv + j * 16 + warp_idx_in_wg * 4 + lane_idx / 8, 8 * 0 + lane_idx % 8);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::INNER_LOADS_CKV; ++mma_d) {
        if constexpr (predicate) {
          ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(
              ckv_smem_offset_w, ckv_ptr, packed_block_iter < packed_kv_bound);
        } else {
          ckv_smem.load_128b_async(ckv_smem_offset_w, ckv_ptr);
        }
        ckv_smem_offset_w += 64;
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

      // KPE: when HEAD_DIM_KPE is too narrow for 8 lanes per row (FP8 case with
      // HEAD_DIM_KPE=64 → 4 b128/row), gate the extra lanes off.
      if (lane_idx % 8 < KTraits::LANES_PER_ROW_KPE) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::INNER_LOADS_KPE; ++mma_d) {
          if constexpr (predicate) {
            kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(
                kpe_smem_offset_w, kpe_ptr, packed_block_iter < packed_kv_bound);
          } else {
            kpe_smem.load_128b_async(kpe_smem_offset_w, kpe_ptr);
          }
          kpe_smem_offset_w += 64;
          kpe_ptr += 8 * upcast_size<DTypeKV>();
        }
      }
    }
  }
}

// FP8 KV path: dequantize one tile of CKV/KPE from packed FP8 shmem buffers into
// BF16 staging buffers, applying per-tensor scales. The destination layout uses
// the same k128B swizzle as the BF16 path, so existing WGMMA descriptors work
// unchanged after pointing them at the staging buffers.
//
// Each thread reads one 16-byte chunk (16 FP8 elems) and writes two
// 16-byte chunks (8 BF16 elems each); see the FP8 dequant idiom in prefill.cuh.
template <typename KTraits>
__device__ __forceinline__ void repack_fp8_kv_to_bf16(
    typename KTraits::SharedStorage* smem_storage, const uint32_t stage_idx, float ckv_scale,
    float kpe_scale, const float* ckv_scale_arr = nullptr, uint32_t row_base = 0,
    const typename KTraits::IdType* kv_indices = nullptr, uint_fastdiv block_size = uint_fastdiv(),
    uint32_t packed_kv_bound = 0) {
  using DTypeKV = typename KTraits::DTypeKV;
  using DTypeQ = typename KTraits::DTypeQ;
  static_assert(std::is_same_v<DTypeKV, __nv_fp8_e4m3>,
                "repack_fp8_kv_to_bf16 only supports DTypeKV == __nv_fp8_e4m3");

  // Only consumer warpgroup does the dequant; producer wg early-returns.
  // The helper has no internal barriers, so this is safe — the surrounding
  // __syncthreads() at the call sites covers cross-wg synchronization.
  if (threadIdx.x < KTraits::NUM_COPY_THREADS) return;
  const uint32_t thread_id = threadIdx.x - KTraits::NUM_COPY_THREADS;
  const uint32_t num_threads = KTraits::NUM_QK_THREADS;

  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t HEAD_DIM_KPE = KTraits::HEAD_DIM_KPE;
  // b128 column counts (16-byte chunks per row).
  constexpr uint32_t FP8_COLS_CKV = HEAD_DIM_CKV / upcast_size<DTypeKV>();
  constexpr uint32_t BF16_COLS_CKV = HEAD_DIM_CKV / upcast_size<DTypeQ>();
  constexpr uint32_t BF16_COLS_KPE = HEAD_DIM_KPE / upcast_size<DTypeQ>();
  constexpr uint32_t NUM_B128_CKV = CTA_TILE_KV * FP8_COLS_CKV;

  using packed2_t = std::conditional_t<std::is_same_v<DTypeQ, half>, half2, nv_bfloat162>;
  packed2_t ckv_scale_packed{static_cast<DTypeQ>(ckv_scale), static_cast<DTypeQ>(ckv_scale)};

  b128_t* src_ckv = (b128_t*)smem_storage->kv_o_smem[stage_idx].ckv;
  b128_t* dst_ckv = (b128_t*)smem_storage->ckv_bf16;
#pragma unroll
  for (uint32_t idx = thread_id; idx < NUM_B128_CKV; idx += num_threads) {
    uint32_t row = idx / FP8_COLS_CKV, col = idx % FP8_COLS_CKV;
    b128_t packed =
        src_ckv[get_swizzle_offset<KTraits::SWIZZLE_MODE_CKV, KTraits::UPCAST_STRIDE_CKV>(row,
                                                                                          col)];
    alignas(16) DTypeQ conv[16];
    vec_cast<DTypeQ, DTypeKV>::template cast<16>(conv, (DTypeKV*)&packed);
    if (ckv_scale_arr) {
      uint32_t page, off;
      block_size.divmod(row_base + row, page, off);
      uint32_t phys = (row_base + row) < packed_kv_bound ? kv_indices[page] * block_size + off : 0;
      float scale =
          ckv_scale_arr[phys * (HEAD_DIM_CKV / 128) + col / (128 / upcast_size<DTypeKV>())];
#pragma unroll
      for (uint32_t k = 0; k < 16; ++k) {
        conv[k] = static_cast<DTypeQ>(static_cast<float>(conv[k]) * scale);
      }
    } else {
#pragma unroll
      for (uint32_t k = 0; k < 8; ++k) {
        ((packed2_t*)&conv[0])[k] = __hmul2(((packed2_t*)&conv[0])[k], ckv_scale_packed);
      }
    }
    dst_ckv[get_swizzle_offset<KTraits::SWIZZLE_MODE_CKV, KTraits::UPCAST_STRIDE_CKV_BF16>(
        row, 2 * col)] = *(b128_t*)&conv[0];
    dst_ckv[get_swizzle_offset<KTraits::SWIZZLE_MODE_CKV, KTraits::UPCAST_STRIDE_CKV_BF16>(
        row, 2 * col + 1)] = *(b128_t*)&conv[8];
  }

  if constexpr (HEAD_DIM_KPE > 0) {
    constexpr uint32_t FP8_COLS_KPE = HEAD_DIM_KPE / upcast_size<DTypeKV>();
    constexpr uint32_t NUM_B128_KPE = CTA_TILE_KV * FP8_COLS_KPE;
    packed2_t kpe_scale_packed{static_cast<DTypeQ>(kpe_scale), static_cast<DTypeQ>(kpe_scale)};

    b128_t* src_kpe = (b128_t*)smem_storage->kv_o_smem[stage_idx].kpe;
    b128_t* dst_kpe = (b128_t*)smem_storage->kpe_bf16;
#pragma unroll
    for (uint32_t idx = thread_id; idx < NUM_B128_KPE; idx += num_threads) {
      uint32_t row = idx / FP8_COLS_KPE, col = idx % FP8_COLS_KPE;
      b128_t packed =
          src_kpe[get_swizzle_offset<KTraits::SWIZZLE_MODE_KPE_RAW, KTraits::UPCAST_STRIDE_KPE>(
              row, col)];
      alignas(16) DTypeQ conv[16];
      vec_cast<DTypeQ, DTypeKV>::template cast<16>(conv, (DTypeKV*)&packed);
#pragma unroll
      for (uint32_t k = 0; k < 8; ++k) {
        ((packed2_t*)&conv[0])[k] = __hmul2(((packed2_t*)&conv[0])[k], kpe_scale_packed);
      }
      dst_kpe[get_swizzle_offset<KTraits::SWIZZLE_MODE_KPE, KTraits::UPCAST_STRIDE_KPE_BF16>(
          row, 2 * col)] = *(b128_t*)&conv[0];
      dst_kpe[get_swizzle_offset<KTraits::SWIZZLE_MODE_KPE, KTraits::UPCAST_STRIDE_KPE_BF16>(
          row, 2 * col + 1)] = *(b128_t*)&conv[8];
    }
  }
}

// One K=16 step of the QK product. Swap-AB tiles issue K as the A operand and Q
// as the B operand; the descriptors are the same either way.
template <typename KTraits, typename wgmma, bool init>
__device__ __forceinline__ void qk_mma(uint64_t desc_q, uint64_t desc_k, float* s_frag) {
  if constexpr (KTraits::SWAP_AB) {
    wgmma::template op<init>(desc_k, desc_q, s_frag);
  } else {
    wgmma::template op<init>(desc_q, desc_k, s_frag);
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx, float* s_frag) {
  // After dequant, the BF16 staging buffers feed the WGMMA on the FP8 KV path;
  // otherwise the per-stage native BF16 buffers are used directly.
  using KVDescType = std::conditional_t<KTraits::USE_KV_REPACK, typename KTraits::DTypeQ,
                                        typename KTraits::DTypeKV>;
  auto* kpe_smem_ptr = KTraits::USE_KV_REPACK ? &smem_storage->kpe_bf16[0]
                                              : (KVDescType*)smem_storage->kv_o_smem[stage_idx].kpe;
  auto* ckv_smem_ptr = KTraits::USE_KV_REPACK ? &smem_storage->ckv_bf16[0]
                                              : (KVDescType*)smem_storage->kv_o_smem[stage_idx].ckv;

  auto desc_q_pe =
      make_smem_desc<KTraits::SWIZZLE_MODE_Q_PE, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_KPE * 16, typename KTraits::DTypeQ>(
          smem_storage->q_smem.pe);
  auto desc_k_pe =
      make_smem_desc<KTraits::SWIZZLE_MODE_KPE, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_KPE * 16, KVDescType>(kpe_smem_ptr);
  using wgmma = WGMMA_ASYNC_SS<KVDescType, float, 64,
                               KTraits::SWAP_AB ? KTraits::CTA_TILE_Q : KTraits::CTA_TILE_KV, 16,
                               Major::K, Major::K, ScaleIn::One, ScaleIn::One>;

  warpgroup_fence_frag<KTraits::NUM_REGS_S_FRAG>(s_frag);
  warpgroup_arrive();
#pragma unroll
  for (uint32_t mma_d_pe = 0; mma_d_pe < KTraits::NUM_MMA_D_KPE; ++mma_d_pe) {
    if (mma_d_pe == 0) {
      qk_mma<KTraits, wgmma, /*init=*/true>(desc_q_pe, desc_k_pe, s_frag);
    } else {
      qk_mma<KTraits, wgmma, /*init=*/false>(desc_q_pe, desc_k_pe, s_frag);
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
                     /*stride_byte_offset=*/KTraits::HEAD_DIM_CKV * 16, KVDescType>(ckv_smem_ptr);

  if constexpr (KTraits::NUM_MMA_D_KPE == 0) {
    qk_mma<KTraits, wgmma, /*init=*/true>(desc_q_nope, desc_ckv, s_frag);
    desc_q_nope += 2;
    desc_ckv += 2;
  }
#pragma unroll
  for (uint32_t mma_d_ckv = KTraits::NUM_MMA_D_KPE == 0 ? 1 : 0; mma_d_ckv < KTraits::NUM_MMA_D_CKV;
       ++mma_d_ckv) {
    qk_mma<KTraits, wgmma, /*init=*/false>(desc_q_nope, desc_ckv, s_frag);
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
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();

  // P is always DTypeQ-typed; V (= CKV) comes from the BF16 staging buffer on
  // the FP8 KV path and from the native BF16 per-stage buffer otherwise.
  using KVDescType = std::conditional_t<KTraits::USE_KV_REPACK, typename KTraits::DTypeQ,
                                        typename KTraits::DTypeKV>;
  KVDescType* ckv_base = KTraits::USE_KV_REPACK
                             ? (KVDescType*)smem_storage->ckv_bf16
                             : (KVDescType*)smem_storage->kv_o_smem[stage_idx].ckv;

  auto desc_p =
      make_smem_desc<KTraits::SWIZZLE_MODE_P, /*leading_byte_offset=*/16,
                     /*stride_byte_offset=*/KTraits::CTA_TILE_KV * 16, typename KTraits::DTypeQ>(
          smem_storage->kv_o_smem[stage_idx].p);
  // V is read along d (MN-major): one 128B swizzle atom per 64 d columns, 8 kv
  // rows apart. Each warpgroup owns the upper or lower half of HEAD_DIM_CKV.
  auto desc_ckv = make_smem_desc<KTraits::SWIZZLE_MODE_CKV,
                                 /*leading_byte_offset=*/KTraits::CTA_TILE_KV * 16,
                                 /*stride_byte_offset=*/KTraits::HEAD_DIM_CKV * 16, KVDescType>(
      ckv_base + warp_group_idx * 8 * (KTraits::HEAD_DIM_CKV / 2));
  warpgroup_fence_frag<KTraits::NUM_REGS_O_FRAG>(o_frag);
  warpgroup_arrive();

  if constexpr (!KTraits::SWAP_AB) {
    using wgmma = WGMMA_ASYNC_SS<KVDescType, float, 64, KTraits::HEAD_DIM_CKV / 2, 16, Major::K,
                                 Major::MN, ScaleIn::One, ScaleIn::One>;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      wgmma::template op</*init=*/false>(desc_p, desc_ckv, o_frag);
      desc_p += 2;
      desc_ckv += 1024;
    }
  } else {
    // O^T[d, q] += V^T[d, kv] * P^T[kv, q]: this warpgroup's half of d is
    // NUM_O_CHUNKS WGMMA-M chunks of 64 rows, each one swizzle atom (1024B)
    // further along d and accumulating into its own registers.
    using wgmma = WGMMA_ASYNC_SS<KVDescType, float, 64, KTraits::CTA_TILE_Q, 16, Major::MN,
                                 Major::K, ScaleIn::One, ScaleIn::One>;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t chunk = 0; chunk < KTraits::NUM_O_CHUNKS; ++chunk) {
        wgmma::template op</*init=*/false>(desc_ckv + chunk * 64, desc_p,
                                           o_frag + chunk * KTraits::NUM_REGS_O_CHUNK);
      }
      desc_p += 2;
      desc_ckv += 1024;
    }
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
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;

  auto keep = [&](uint32_t q_idx, uint32_t kv_idx) {
    return !(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                             : kv_idx >= kv_end);
  };

  if constexpr (!KTraits::SWAP_AB) {
    uint32_t q[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      q[j] = (qo_packed_idx_base + warp_idx_in_wg * 16 + lane_idx / 4 + 8 * j) / num_heads;
    }
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_S_FRAG; ++reg_id) {
      const uint32_t q_idx = q[(reg_id % 4) / 2],
                     kv_idx = kv_idx_base + 2 * (lane_idx % 4) + 8 * (reg_id / 4) + reg_id % 2;
      s_frag[reg_id] = keep(q_idx, kv_idx) ? s_frag[reg_id] : KTraits::MaskFillValue;
    }
  } else {
    // S^T: rows are kv positions, columns are packed query rows.
    uint32_t q[KTraits::NUM_MD];
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      q[md] = (qo_packed_idx_base + KTraits::md_column(md, lane_idx)) / num_heads;
    }
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_S_FRAG; ++reg_id) {
      const uint32_t q_idx = q[KTraits::md_index(reg_id)],
                     kv_idx =
                         kv_idx_base + warp_idx_in_wg * 16 + lane_idx / 4 + 8 * ((reg_id % 4) / 2);
      s_frag[reg_id] = keep(q_idx, kv_idx) ? s_frag[reg_id] : KTraits::MaskFillValue;
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void rescale_o_(float* o_scale, float* o_frag) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] *= o_scale[KTraits::md_index(reg_id)];
  }
}

// Online-softmax update for one KV tile: folds the tile into the running
// max/sum, rewrites s_frag as unnormalized probabilities, and leaves the O
// rescale factor in o_scale. d stays a per-thread partial sum until the end.
template <typename KTraits>
__device__ __forceinline__ void update_md_(typename KTraits::SharedStorage* smem_storage,
                                           typename KTraits::AttentionVariant variant,
                                           float* s_frag, float* m, float* d, float* o_scale) {
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  // Per-row clamp for fully masked rows (see update_mdo_states in prefill.cuh).
  auto scaled_max = [&](float m_val) {
    return max(m_val * sm_scale, -cuda::std::numeric_limits<float>::max());
  };

  if constexpr (!KTraits::SWAP_AB) {
    // Rows of S: a thread holds two rows, spread over the four lanes of a quad.
    float m_prev[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (uint32_t k = 0; k < KTraits::NUM_REGS_S_FRAG / 4; ++k) {
        float m_local = max(s_frag[k * 4 + j * 2 + 0], s_frag[k * 4 + j * 2 + 1]);
        m[j] = max(m[j], m_local);
      }
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
    }

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      const float m_scaled = scaled_max(m[j]);
      o_scale[j] = math::ptx_exp2(m_prev[j] * sm_scale - m_scaled);
      float d_local = 0.f;
#pragma unroll
      for (uint32_t k = 0; k < KTraits::NUM_REGS_S_FRAG / 4; ++k) {
        s_frag[k * 4 + j * 2 + 0] = math::ptx_exp2(s_frag[k * 4 + j * 2 + 0] * sm_scale - m_scaled);
        s_frag[k * 4 + j * 2 + 1] = math::ptx_exp2(s_frag[k * 4 + j * 2 + 1] * sm_scale - m_scaled);

        d_local += s_frag[k * 4 + j * 2 + 0] + s_frag[k * 4 + j * 2 + 1];
      }
      d[j] = d[j] * o_scale[j] + d_local;
    }
  } else {
    // Columns of S^T: a thread holds two kv rows of each column. The column
    // max is reduced over the eight lanes that share a column (lane bits 2..4)
    // and then across the four warps of the warpgroup through shared memory.
    float m_prev[KTraits::NUM_MD];
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      m_prev[md] = m[md];
      const uint32_t reg_id = 4 * (md / 2) + md % 2;
      float m_local = max(s_frag[reg_id], s_frag[reg_id + 2]);
      m_local = max(m_local, math::shfl_xor_sync(m_local, 0x4));
      m_local = max(m_local, math::shfl_xor_sync(m_local, 0x8));
      m_local = max(m_local, math::shfl_xor_sync(m_local, 0x10));
      if (lane_idx < 4) {
        smem_storage->md_partial[warp_idx_in_wg][KTraits::md_column(md, lane_idx)] = m_local;
      }
    }
    barrier_sync(KTraits::NUM_QK_THREADS, NamedBarriers::kConsumerWarpgroup);

#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      const uint32_t col = KTraits::md_column(md, lane_idx);
      const float m_tile =
          max(max(smem_storage->md_partial[0][col], smem_storage->md_partial[1][col]),
              max(smem_storage->md_partial[2][col], smem_storage->md_partial[3][col]));
      m[md] = max(m[md], m_tile);
      const float m_scaled = scaled_max(m[md]);
      o_scale[md] = math::ptx_exp2(m_prev[md] * sm_scale - m_scaled);
      const uint32_t reg_id = 4 * (md / 2) + md % 2;
      s_frag[reg_id] = math::ptx_exp2(s_frag[reg_id] * sm_scale - m_scaled);
      s_frag[reg_id + 2] = math::ptx_exp2(s_frag[reg_id + 2] * sm_scale - m_scaled);
      d[md] = d[md] * o_scale[md] + s_frag[reg_id] + s_frag[reg_id + 2];
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_p_rmem_smem(typename KTraits::SharedStorage* smem_storage,
                                                  const uint32_t stage_idx, uint32_t* p_frag) {
  // P is DTypeQ-typed, so use UPCAST_STRIDE_P_BF16 (BF16 elements per b128) for
  // the swizzle offset on both the BF16 and FP8 KV paths.
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kv_o_smem[stage_idx].p);
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      uint32_t p_smem_offset_w =
          get_swizzle_offset<KTraits::SWIZZLE_MODE_P, KTraits::UPCAST_STRIDE_P_BF16>(
              warp_idx_in_wg * 16 + lane_idx % 16, mma_kv * 2 + lane_idx / 16);
      p_smem.stmatrix_m8n8x4(p_smem_offset_w, p_frag + mma_kv * 4);
    }
  } else {
    // The registers hold P^T; a transposed stmatrix lands every 8x8 block as
    // P[q, kv], the K-major B operand of the PV product. Each x4 store covers
    // 16 query columns by this warp's 16 kv rows.
#pragma unroll
    for (uint32_t i = 0; i < KTraits::CTA_TILE_Q / 16; ++i) {
      uint32_t p_smem_offset_w =
          get_swizzle_offset<KTraits::SWIZZLE_MODE_P, KTraits::UPCAST_STRIDE_P_BF16>(
              i * 16 + 8 * (lane_idx / 16) + lane_idx % 8, warp_idx_in_wg * 2 + (lane_idx / 8) % 2);
      p_smem.stmatrix_m8n8x4_trans(p_smem_offset_w, p_frag + i * 4);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d_(float* o_frag, float* m, float* d) {
  float d_rcp[KTraits::NUM_MD];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t j = 0; j < KTraits::NUM_MD; ++j) {
    d_rcp[j] = (m[j] != -math::inf) ? math::ptx_rcp(d[j]) : 0.f;
  }

#pragma unroll
  for (uint32_t reg_id = 0; reg_id < KTraits::NUM_REGS_O_FRAG; ++reg_id) {
    o_frag[reg_id] = o_frag[reg_id] * d_rcp[KTraits::md_index(reg_id)];
  }
}

template <typename KTraits>
__device__ __forceinline__ void scale_m_(typename KTraits::AttentionVariant variant, float* m) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t j = 0; j < KTraits::NUM_MD; ++j) {
      if (m[j] != -math::inf) {
        m[j] *= variant.sm_scale_log2;
      }
    }
  }
}

template <bool write_lse, typename KTraits>
__device__ __forceinline__ void write_o(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeO* final_o,
    float* final_lse, typename KTraits::DTypeO* partial_o, float* partial_lse, float* o_frag,
    float* m, float* d, const uint32_t o_stride_n, const uint32_t o_stride_h, const uint32_t q_len,
    const uint32_t packed_offset, const uint_fastdiv& num_heads, const bool& return_lse_base_on_e) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O = KTraits::UPCAST_STRIDE_FINAL_O;
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  // o is a top-level (not per-stage) shmem region. Both warpgroups write
  // disjoint halves of the same buffer, offset by warp_group_idx * NUM_MMA_D_CKV.
  smem_t<KTraits::SWIZZLE_MODE_O> o_smem;
  o_smem = smem_storage->o;

  // step 0. rmem to smem
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t k = 0; k < HEAD_DIM_CKV / 32; ++k) {
      uint32_t o_frag_f16[8 / 2];
      vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, &o_frag[k * 8]);
      uint32_t o_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
          warp_idx_in_wg * 16 + lane_idx % 16,
          warp_group_idx * NUM_MMA_D_CKV + k * 2 + lane_idx / 16);
      o_smem.template stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
    }
  } else {
    // The registers hold O^T; a transposed stmatrix lands every 8x8 block as
    // O[q, d]. Each x4 store covers 16 query rows by this warp's 16 d columns
    // of one 64-column chunk.
#pragma unroll
    for (uint32_t chunk = 0; chunk < KTraits::NUM_O_CHUNKS; ++chunk) {
#pragma unroll
      for (uint32_t i = 0; i < KTraits::CTA_TILE_Q / 16; ++i) {
        uint32_t o_frag_f16[8 / 2];
        vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16,
                                         &o_frag[chunk * KTraits::NUM_REGS_O_CHUNK + i * 8]);
        uint32_t o_smem_offset_w =
            get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
                i * 16 + 8 * (lane_idx / 16) + lane_idx % 8, warp_group_idx * NUM_MMA_D_CKV +
                                                                 chunk * 8 + warp_idx_in_wg * 2 +
                                                                 (lane_idx / 8) % 2);
        o_smem.template stmatrix_m8n8x4_trans(o_smem_offset_w, o_frag_f16);
      }
    }
    // Every warp of this warpgroup wrote a slice of every row: wait for the
    // whole half of O before reading rows back.
    barrier_sync(KTraits::NUM_COPY_THREADS, warp_group_idx == 0
                                                ? NamedBarriers::kProducerWarpgroup
                                                : NamedBarriers::kConsumerWarpgroup);
  }

  // step 1. smem to gmem, 16 rows per pass: 4 rows per warp, 8 lanes per row.
  // In the original layout each warp copies back the 16 rows it staged itself.
#pragma unroll
  for (uint32_t pass = 0; pass < KTraits::CTA_TILE_Q / 16; ++pass) {
    const uint32_t row = KTraits::SWAP_AB ? pass * 16 + warp_idx_in_wg * 4 + lane_idx / 8
                                          : warp_idx_in_wg * 16 + pass * 4 + lane_idx / 8;
    uint32_t q, r;
    num_heads.divmod(packed_offset + row, q, r);
    DTypeO* o_ptr = (partial_o != nullptr)
                        ? partial_o + (blockIdx.x * KTraits::CTA_TILE_Q + row) * HEAD_DIM_CKV
                        : final_o + q * o_stride_n + r * o_stride_h;
    o_ptr += warp_group_idx * (HEAD_DIM_CKV / 2) + (lane_idx % 8) * upcast_size<DTypeO>();
    uint32_t o_smem_offset_w = get_swizzle_offset<KTraits::SWIZZLE_MODE_O, UPCAST_STRIDE_FINAL_O>(
        row, warp_group_idx * NUM_MMA_D_CKV + lane_idx % 8);
#pragma unroll
    for (uint32_t k = 0; k < HEAD_DIM_CKV / 128; ++k) {
      if (q < q_len) {
        o_smem.template store_128b(o_smem_offset_w, o_ptr);
      }
      o_ptr += 8 * upcast_size<DTypeO>();
      o_smem_offset_w += 64;
    }
  }

  if constexpr (write_lse) {
    auto store_lse = [&](uint32_t row, float m_val, float d_val) {
      uint32_t q, r;
      num_heads.divmod(packed_offset + row, q, r);
      if (q >= q_len) return;
      const float lse = (m_val == -math::inf) ? -cuda::std::numeric_limits<float>::infinity()
                                              : math::ptx_log2(d_val) + m_val;
      if (partial_o != nullptr) {
        partial_lse[blockIdx.x * KTraits::CTA_TILE_Q + row] = lse;
      } else if (final_lse) {
        final_lse[q * num_heads + r] = return_lse_base_on_e ? lse * math::loge2 : lse;
      }
    };
    if constexpr (!KTraits::SWAP_AB) {
      if (lane_idx % 4 == 0) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          store_lse(warp_idx_in_wg * 16 + 8 * j + lane_idx / 4, m[j], d[j]);
        }
      }
    } else {
      // Every thread holds the reduced statistics of its columns; warp 0's
      // first quad covers all CTA_TILE_Q of them.
      if (warp_idx_in_wg == 0 && lane_idx < 4) {
#pragma unroll
        for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
          store_lse(KTraits::md_column(md, lane_idx), m[md], d[md]);
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
__device__ __forceinline__ void convert_s_to_p(float* s_frag, uint32_t* p_frag) {
  // P is always DTypeQ-typed in shmem (see HopperSharedStorageQKVO::kv_o_smem.p)
  // so the PV WGMMA can run as BF16xBF16 on the FP8 KV path. On the BF16 KV
  // path DTypeQ == DTypeKV so this is unchanged.
#pragma unroll
  for (uint32_t i = 0; i < KTraits::NUM_REGS_S_FRAG / 8; ++i) {
    vec_cast<typename KTraits::DTypeQ, float>::cast<8>(((typename KTraits::DTypeQ*)p_frag) + i * 8,
                                                       s_frag + i * 8);
  }
}

// Consumer -> producer handoff of the per-tile O rescale factors.
template <typename KTraits>
__device__ __forceinline__ void write_o_scale_smem(typename KTraits::SharedStorage* smem_storage,
                                                   float* o_scale) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (lane_idx % 4 == 0) {
        smem_storage->o_scale[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = o_scale[j];
      }
    }
  } else {
    if (warp_idx_in_wg == 0 && lane_idx < 4) {
#pragma unroll
      for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
        smem_storage->o_scale[KTraits::md_column(md, lane_idx)] = o_scale[md];
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_o_scale_smem(typename KTraits::SharedStorage* smem_storage,
                                                  float* o_scale) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      o_scale[j] = smem_storage->o_scale[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];
    }
  } else {
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      o_scale[md] = smem_storage->o_scale[KTraits::md_column(md, lane_idx)];
    }
  }
}

// Consumer, after the last KV tile: complete the row sums d (per-thread partials
// until now) and publish m/d for the producer warpgroup through shared memory.
template <typename KTraits>
__device__ __forceinline__ void finalize_md_(typename KTraits::SharedStorage* smem_storage,
                                             float* m, float* d) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d[j] += math::shfl_xor_sync(d[j], 0x2);
      d[j] += math::shfl_xor_sync(d[j], 0x1);
      if (lane_idx % 4 == 0) {
        smem_storage->m[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = m[j];
        smem_storage->d[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = d[j];
      }
    }
  } else {
    // Sum over the eight lanes sharing a column, then across the four warps.
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      d[md] += math::shfl_xor_sync(d[md], 0x4);
      d[md] += math::shfl_xor_sync(d[md], 0x8);
      d[md] += math::shfl_xor_sync(d[md], 0x10);
      if (lane_idx < 4) {
        smem_storage->md_partial[warp_idx_in_wg][KTraits::md_column(md, lane_idx)] = d[md];
      }
    }
    barrier_sync(KTraits::NUM_QK_THREADS, NamedBarriers::kConsumerWarpgroup);
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      const uint32_t col = KTraits::md_column(md, lane_idx);
      d[md] = (smem_storage->md_partial[0][col] + smem_storage->md_partial[1][col]) +
              (smem_storage->md_partial[2][col] + smem_storage->md_partial[3][col]);
      if (warp_idx_in_wg == 0 && lane_idx < 4) {
        smem_storage->m[col] = m[md];
        smem_storage->d[col] = d[md];
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_md_(typename KTraits::SharedStorage* smem_storage, float* m,
                                         float* d) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  const uint32_t warp_idx_in_wg = cutlass::canonical_warp_idx() % 4;
  if constexpr (!KTraits::SWAP_AB) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[j] = smem_storage->m[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];
      d[j] = smem_storage->d[warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];
    }
  } else {
#pragma unroll
    for (uint32_t md = 0; md < KTraits::NUM_MD; ++md) {
      m[md] = smem_storage->m[KTraits::md_column(md, lane_idx)];
      d[md] = smem_storage->d[KTraits::md_column(md, lane_idx)];
    }
  }
}

template <typename Pipeline, typename PipelineState>
__device__ __forceinline__ void consumer_wait(Pipeline& pipeline, PipelineState& smem_pipe_read) {
  auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
  pipeline.consumer_wait(smem_pipe_read, barrier_token);
}

// Coordinates of one work item, shared by the per-tile steps below.
struct WorkCoord {
  uint32_t qo_packed_idx_base;
  uint32_t q_len;
  uint32_t kv_len;
  uint32_t kv_start;
  uint32_t kv_end;
  // First packed KV position of this work item and the end of the request's KV.
  uint32_t kv_block_iter_base;
  uint32_t packed_kv_bound;
};

// Producer warpgroup, one KV tile: take the consumer's rescale factors, rescale
// the running O and run this warpgroup's half of the PV product.
template <typename KTraits, typename Pipeline, typename PipelineState>
__device__ __forceinline__ void producer_pv_tile(typename KTraits::SharedStorage& smem_storage,
                                                 typename KTraits::AttentionVariant& variant,
                                                 Pipeline& pipeline_kv,
                                                 PipelineState& smem_pipe_read_kv, float* o_frag,
                                                 float* o_scale) {
  if constexpr (KTraits::USE_KV_REPACK) {
    // d1/d2: pair with the consumer's s1/s2 around its FP8->BF16 dequant. The
    // producer only passes through; after d2 the BF16 staging buffer holds
    // the current stage and is safe to read in the PV WGMMA below.
    __syncthreads();
    __syncthreads();
  }
  barrier_sync(KTraits::NUM_THREADS, NamedBarriers::kOScaleReady);
  load_o_scale_smem<KTraits>(&smem_storage, o_scale);
  PROFILER_EVENT_START(variant, ProfileEventType::kRescaleO);
  rescale_o_<KTraits>(o_scale, o_frag);
  PROFILER_EVENT_END(variant, ProfileEventType::kRescaleO);
  consumer_wait(pipeline_kv, smem_pipe_read_kv);
  __syncthreads();
  PROFILER_EVENT_START(variant, ProfileEventType::kGemmPV);
  compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv.index(), o_frag);
  warpgroup_wait<0>();
  PROFILER_EVENT_END(variant, ProfileEventType::kGemmPV);
  pipeline_kv.consumer_release(smem_pipe_read_kv);
  ++smem_pipe_read_kv;
  if constexpr (KTraits::USE_KV_REPACK) {
    // iter-end sync: both warpgroups must finish this tile's WGMMA reads of
    // the BF16 staging buffer before the next dequant overwrites it.
    __syncthreads();
  }
}

// Producer warpgroup, one work item: stream Q and the KV tiles into shared
// memory one tile ahead of the consumer, and own the lower half of O.
template <typename KTraits, typename Params, typename Pipeline, typename PipelineState>
__device__ __forceinline__ void producer_work(
    const Params& params, typename KTraits::SharedStorage& smem_storage,
    typename KTraits::AttentionVariant& variant, const typename Params::IdType work_idx,
    Pipeline& pipeline_q, Pipeline& pipeline_kv, PipelineState& smem_pipe_write_q,
    PipelineState& smem_pipe_write_kv, PipelineState& smem_pipe_read_kv) {
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr bool CAUSAL = KTraits::CAUSAL;
  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;

  auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end] =
      get_block_coord(params, work_idx);

  alignas(16) float o_frag[KTraits::NUM_REGS_O_FRAG];
  float m[KTraits::NUM_MD];
  float d[KTraits::NUM_MD];
  float o_scale[KTraits::NUM_MD];
  init_states_<KTraits>(o_frag, m, d, o_scale);

  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;
  const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
  const uint32_t qo_upperbound =
      min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

  const uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;
  int kv_tile_idx =
      ceil_div(
          (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                  : kv_end),
          CTA_TILE_KV) -
      1 - (kv_start / CTA_TILE_KV);
  const bool has_kv = kv_tile_idx >= 0;
  const uint32_t block_iter_base = kv_indptr * block_size + kv_start;

  int64_t ckv_offset[KTraits::NUM_MMA_KV / 2][2];
  int64_t kpe_offset[KTraits::NUM_MMA_KV / 2][2];

  if (has_kv) {
    prefetch_offset<KTraits>(block_iter_base + kv_tile_idx * CTA_TILE_KV, packed_kv_bound,
                             params.ckv_stride_page, params.ckv_stride_n, params.kpe_stride_page,
                             params.kpe_stride_n, block_size, params.kv_indices, ckv_offset,
                             kpe_offset);
    pipeline_kv.producer_acquire(smem_pipe_write_kv);
    PROFILER_EVENT_START(variant, ProfileEventType::kIssueLoadKV);
    load_kv<true, KTraits>(&smem_storage, params.ckv, params.kpe, packed_kv_bound,
                           block_iter_base + kv_tile_idx * CTA_TILE_KV, smem_pipe_write_kv.index(),
                           ckv_offset, kpe_offset);
    PROFILER_EVENT_END(variant, ProfileEventType::kIssueLoadKV);
    pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
    kv_tile_idx -= 1;
    ++smem_pipe_write_kv;
    if (kv_tile_idx >= 0) {
      prefetch_offset<KTraits>(block_iter_base + kv_tile_idx * CTA_TILE_KV, packed_kv_bound,
                               params.ckv_stride_page, params.ckv_stride_n, params.kpe_stride_page,
                               params.kpe_stride_n, block_size, params.kv_indices, ckv_offset,
                               kpe_offset);
    }
  }

  pipeline_q.producer_acquire(smem_pipe_write_q);
  PROFILER_EVENT_START(variant, ProfileEventType::kIssueLoadQ);
  load_q<KTraits>(&smem_storage, params.q_nope + q_indptr * params.q_nope_stride_n,
                  params.q_pe + q_indptr * params.q_pe_stride_n, params.q_nope_stride_n,
                  params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, qo_upperbound,
                  qo_packed_idx_base, num_heads);
  PROFILER_EVENT_END(variant, ProfileEventType::kIssueLoadQ);
  pipeline_q.producer_commit(smem_pipe_write_q, cutlass::arch::cpasync_barrier_arrive);
  ++smem_pipe_write_q;

#pragma unroll 1
  for (; kv_tile_idx >= 0; --kv_tile_idx) {
    pipeline_kv.producer_acquire(smem_pipe_write_kv);
    PROFILER_EVENT_START(variant, ProfileEventType::kIssueLoadKV);
    load_kv<false, KTraits>(&smem_storage, params.ckv, params.kpe, packed_kv_bound,
                            block_iter_base + kv_tile_idx * CTA_TILE_KV, smem_pipe_write_kv.index(),
                            ckv_offset, kpe_offset);
    PROFILER_EVENT_END(variant, ProfileEventType::kIssueLoadKV);
    if (kv_tile_idx > 0) {
      prefetch_offset<KTraits>(block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV, packed_kv_bound,
                               params.ckv_stride_page, params.ckv_stride_n, params.kpe_stride_page,
                               params.kpe_stride_n, block_size, params.kv_indices, ckv_offset,
                               kpe_offset);
    }
    pipeline_kv.producer_commit(smem_pipe_write_kv, cutlass::arch::cpasync_barrier_arrive);
    ++smem_pipe_write_kv;

    producer_pv_tile<KTraits>(smem_storage, variant, pipeline_kv, smem_pipe_read_kv, o_frag,
                              o_scale);
  }

  if (has_kv) {
    producer_pv_tile<KTraits>(smem_storage, variant, pipeline_kv, smem_pipe_read_kv, o_frag,
                              o_scale);
  }

  barrier_sync(KTraits::NUM_THREADS, NamedBarriers::kMDReady);
  load_md_<KTraits>(&smem_storage, m, d);
  normalize_d_<KTraits>(o_frag, m, d);
  PROFILER_EVENT_START(variant, ProfileEventType::kWriteO);
  write_o<false, KTraits>(
      &smem_storage, params.final_o + q_indptr * params.o_stride_n,
      params.final_lse ? params.final_lse + q_indptr * num_heads : nullptr,
      (partial_indptr == -1) ? nullptr : params.partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
      (partial_indptr == -1) ? nullptr : params.partial_lse + partial_indptr, o_frag, m, d,
      params.o_stride_n, params.o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads,
      params.return_lse_base_on_e);
  PROFILER_EVENT_END(variant, ProfileEventType::kWriteO);
  __syncthreads();
}

// Consumer warpgroup, one KV tile: QK, masking (for boundary tiles only),
// online softmax, P to shared memory, then this warpgroup's half of PV.
template <typename KTraits, bool MASK, typename Params, typename Pipeline, typename PipelineState>
__device__ __forceinline__ void consumer_tile(
    const Params& params, typename KTraits::SharedStorage& smem_storage,
    typename KTraits::AttentionVariant& variant, Pipeline& pipeline_kv,
    PipelineState& smem_pipe_read_kv, const WorkCoord& work, const int kv_tile_idx, float* s_frag,
    uint32_t* p_frag, float* o_frag, float* m, float* d, float* o_scale) {
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;

  consumer_wait(pipeline_kv, smem_pipe_read_kv);
  if constexpr (KTraits::USE_KV_REPACK) {
    // s1: pair with producer's d1 (matches per-iter __syncthreads count).
    __syncthreads();
    // Only consumer warpgroup (128 threads) dequants FP8 KV -> BF16
    // staging buffers. Producer wg passes through the d1/d2/d_end syncs.
    repack_fp8_kv_to_bf16<KTraits>(&smem_storage, smem_pipe_read_kv.index(), variant.ckv_scale,
                                   variant.kpe_scale, variant.ckv_scale_arr,
                                   work.kv_block_iter_base + kv_tile_idx * CTA_TILE_KV,
                                   params.kv_indices, params.block_size, work.packed_kv_bound);
    // s2: ensures dequant is complete before any wg reads BF16 staging.
    __syncthreads();
  }
  PROFILER_EVENT_START(variant, ProfileEventType::kGemmQK);
  compute_mla_qk<KTraits>(&smem_storage, smem_pipe_read_kv.index(), s_frag);
  warpgroup_wait<0>();
  PROFILER_EVENT_END(variant, ProfileEventType::kGemmQK);
  if constexpr (MASK) {
    logits_mask_<KTraits>(work.qo_packed_idx_base, work.kv_start + kv_tile_idx * CTA_TILE_KV,
                          work.q_len, work.kv_len, work.kv_end, params.num_heads, s_frag);
  }
  PROFILER_EVENT_START(variant, ProfileEventType::kSoftmaxUpdate);
  update_md_<KTraits>(&smem_storage, variant, s_frag, m, d, o_scale);
  PROFILER_EVENT_END(variant, ProfileEventType::kSoftmaxUpdate);
  write_o_scale_smem<KTraits>(&smem_storage, o_scale);
  convert_s_to_p<KTraits>(s_frag, p_frag);
  write_p_rmem_smem<KTraits>(&smem_storage, smem_pipe_read_kv.index(), p_frag);
  barrier_arrive(KTraits::NUM_THREADS, NamedBarriers::kOScaleReady);
  PROFILER_EVENT_START(variant, ProfileEventType::kRescaleO);
  rescale_o_<KTraits>(o_scale, o_frag);
  PROFILER_EVENT_END(variant, ProfileEventType::kRescaleO);
  __syncthreads();
  PROFILER_EVENT_START(variant, ProfileEventType::kGemmPV);
  compute_mla_pv<KTraits>(&smem_storage, smem_pipe_read_kv.index(), o_frag);
  warpgroup_wait<0>();
  PROFILER_EVENT_END(variant, ProfileEventType::kGemmPV);
  pipeline_kv.consumer_release(smem_pipe_read_kv);
  ++smem_pipe_read_kv;
  if constexpr (KTraits::USE_KV_REPACK) {
    // iter-end sync: both warpgroups must finish this tile's WGMMA reads of
    // the BF16 staging buffer before the next dequant overwrites it.
    __syncthreads();
  }
}

// Consumer warpgroup, one work item: QK and softmax for every KV tile, the
// upper half of O, and the final statistics / LSE.
template <typename KTraits, typename Params, typename Pipeline, typename PipelineState>
__device__ __forceinline__ void consumer_work(const Params& params,
                                              typename KTraits::SharedStorage& smem_storage,
                                              typename KTraits::AttentionVariant& variant,
                                              const typename Params::IdType work_idx,
                                              Pipeline& pipeline_q, Pipeline& pipeline_kv,
                                              PipelineState& smem_pipe_read_q,
                                              PipelineState& smem_pipe_read_kv) {
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  constexpr bool CAUSAL = KTraits::CAUSAL;
  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;

  auto [q_indptr, kv_indptr, partial_indptr, q_len, kv_len, packed_qo_start, kv_start, kv_end] =
      get_block_coord(params, work_idx);

  alignas(16) float o_frag[KTraits::NUM_REGS_O_FRAG];
  float m[KTraits::NUM_MD];
  float d[KTraits::NUM_MD];
  float o_scale[KTraits::NUM_MD];
  float s_frag[KTraits::NUM_REGS_S_FRAG];
  uint32_t p_frag[KTraits::NUM_REGS_P_FRAG];
  init_states_<KTraits>(o_frag, m, d, o_scale);

  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;
  const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
  const uint32_t qo_upperbound =
      min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));
  const WorkCoord work{qo_packed_idx_base,
                       static_cast<uint32_t>(q_len),
                       static_cast<uint32_t>(kv_len),
                       static_cast<uint32_t>(kv_start),
                       static_cast<uint32_t>(kv_end),
                       kv_indptr * block_size + kv_start,
                       kv_indptr * block_size + kv_len};

  int kv_tile_idx =
      ceil_div(
          (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                  : kv_end),
          CTA_TILE_KV) -
      1 - (kv_start / CTA_TILE_KV);

  int mask_tile_idx =
      (CAUSAL ? min(kv_end, kv_len - q_len + static_cast<uint32_t>(packed_qo_start) / num_heads)
              : kv_end) /
          CTA_TILE_KV -
      (kv_start / CTA_TILE_KV);

  consumer_wait(pipeline_q, smem_pipe_read_q);
  // Tiles crossing the causal / kv_end boundary need masking; the interior
  // tiles do not. The last NUM_STAGES tiles are handled as boundary tiles.
#pragma unroll 1
  for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
    consumer_tile<KTraits, /*MASK=*/true>(params, smem_storage, variant, pipeline_kv,
                                          smem_pipe_read_kv, work, kv_tile_idx, s_frag, p_frag,
                                          o_frag, m, d, o_scale);
  }
#pragma unroll 1
  for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
    consumer_tile<KTraits, /*MASK=*/false>(params, smem_storage, variant, pipeline_kv,
                                           smem_pipe_read_kv, work, kv_tile_idx, s_frag, p_frag,
                                           o_frag, m, d, o_scale);
  }
#pragma unroll 1
  for (; kv_tile_idx >= 0; --kv_tile_idx) {
    consumer_tile<KTraits, /*MASK=*/true>(params, smem_storage, variant, pipeline_kv,
                                          smem_pipe_read_kv, work, kv_tile_idx, s_frag, p_frag,
                                          o_frag, m, d, o_scale);
  }

  pipeline_q.consumer_release(smem_pipe_read_q);
  ++smem_pipe_read_q;

  finalize_md_<KTraits>(&smem_storage, m, d);
  normalize_d_<KTraits>(o_frag, m, d);
  scale_m_<KTraits>(variant, m);
  barrier_arrive(KTraits::NUM_THREADS, NamedBarriers::kMDReady);
  PROFILER_EVENT_START(variant, ProfileEventType::kWriteO);
  write_o<true, KTraits>(
      &smem_storage, params.final_o + q_indptr * params.o_stride_n,
      params.final_lse ? params.final_lse + q_indptr * num_heads : nullptr,
      (partial_indptr == -1) ? nullptr : params.partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
      (partial_indptr == -1) ? nullptr : params.partial_lse + partial_indptr, o_frag, m, d,
      params.o_stride_n, params.o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads,
      params.return_lse_base_on_e);
  PROFILER_EVENT_END(variant, ProfileEventType::kWriteO);
  __syncthreads();
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPageAttentionHopperKernel(
    const __grid_constant__ Params params) {
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);
  IdType* work_indptr = params.work_indptr;
  const uint32_t warp_group_idx = cutlass::canonical_warp_group_idx();

  PROFILER_INIT(params, smem_storage, variant, warp_group_idx, 2, (threadIdx.x % 128 == 0));

  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  PipelineParams pipeline_params;
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.producer_arv_count = 128;
  pipeline_params.consumer_arv_count = 128;
  MainloopPipeline pipeline_q(smem_storage.pipeline_q, pipeline_params);
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::ProducerConsumer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.producer_arv_count = 128;
  pipeline_params.consumer_arv_count = 256;
  MainloopPipeline pipeline_kv(smem_storage.pipeline_kv, pipeline_params);

  __syncthreads();

  if (warp_group_idx == 0) {
    // load q & kv, compute pv1
    PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_write_kv = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read_kv;

#pragma unroll 1
    for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
         ++work_idx) {
      producer_work<KTraits>(params, smem_storage, variant, work_idx, pipeline_q, pipeline_kv,
                             smem_pipe_write_q, smem_pipe_write_kv, smem_pipe_read_kv);
    }
  } else {
    // compute qk, pv2
    PipelineState smem_pipe_read_q;
    PipelineState smem_pipe_read_kv;

#pragma unroll 1
    for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
         ++work_idx) {
      consumer_work<KTraits>(params, smem_storage, variant, work_idx, pipeline_q, pipeline_kv,
                             smem_pipe_read_q, smem_pipe_read_kv);
    }
  }

  auto grid = cg::this_grid();
  grid.sync();

  PROFILER_EVENT_START(variant, ProfileEventType::kSplitK);

  __syncthreads();
  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      smem, params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, params.partial_o, params.partial_lse, params.final_o,
      params.final_lse, params.o_stride_n, params.o_stride_h, params.num_heads,
      params.return_lse_base_on_e);

  PROFILER_EVENT_END(variant, ProfileEventType::kSplitK);
}

}  // namespace hopper

// Q tiles the SM90 kernel is instantiated for; MLAPlan reports the planned one
// in MLAPlanInfo::cta_tile_q.
#define DISPATCH_MLA_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \
  switch (cta_tile_q) {                                        \
    case 16: {                                                 \
      constexpr uint32_t CTA_TILE_Q = 16;                      \
      __VA_ARGS__                                              \
      break;                                                   \
    }                                                          \
    case 32: {                                                 \
      constexpr uint32_t CTA_TILE_Q = 32;                      \
      __VA_ARGS__                                              \
      break;                                                   \
    }                                                          \
    case 64: {                                                 \
      constexpr uint32_t CTA_TILE_Q = 64;                      \
      __VA_ARGS__                                              \
      break;                                                   \
    }                                                          \
    default: {                                                 \
      std::ostringstream err_msg;                              \
      err_msg << "Unsupported MLA cta_tile_q: " << cta_tile_q; \
      FLASHINFER_ERROR(err_msg.str());                         \
    }                                                          \
  }

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t CTA_TILE_Q,
          typename Params>
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

  // NUM_STAGES=2 for both paths. The FP8 KV path fits within the 228KB/SM
  // budget by sharing a single (non-per-stage) `o` writeback overlay across
  // the whole data path, freeing up room for the BF16 dequant staging buffers.
  constexpr uint32_t NUM_STAGES = 2;
  constexpr uint32_t CTA_TILE_KV = 64;

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
