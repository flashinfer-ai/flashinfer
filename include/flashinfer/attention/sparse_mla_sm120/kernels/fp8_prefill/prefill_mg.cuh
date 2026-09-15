// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../arch/matrix_memory.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../common/d2_load_b.cuh"
#include "../../compute/online_softmax.cuh"
#include "../../compute/q_rope.cuh"
#include "../../compute/q_stage.cuh"
#include "../../compute/scale_mma.cuh"
#include "../../compute/warp_tiles.cuh"
#include "../../model/scale_convert.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "prefill_common.cuh"
#include "resources.cuh"
#include "smem_layout.cuh"
#include "xv_rope_mma.cuh"

using flashinfer::sparse_mla_sm120::pv_fp8_d2_16x8;
using flashinfer::sparse_mla_sm120::pv_fp8_d2_16x8_pair;
using flashinfer::sparse_mla_sm120::qk_bf16_from_fp8_nope_16x8;
using flashinfer::sparse_mla_sm120::qk_fp8_scale_group_16x8;
using flashinfer::sparse_mla_sm120::pipeline::BulkReady;
using flashinfer::sparse_mla_sm120::pipeline::RoleSync;
using flashinfer::sparse_mla_sm120::pipeline::SlotRelease;

// ============================================================================
// Multi-Group (MG) Prefill Kernel — 1 or 2 head groups per CTA
//
// MG_N_HG_T = 2 (HEADS_PER_CTA=32): NUM_HEADS in {32,64,128}, KV reused across
//             both groups (2× reuse, deferred row_sum, higher MMA utilization).
// MG_N_HG_T = 1 (HEADS_PER_CTA=16): NUM_HEADS in {8,16}; NH=8 zero-pads the
//             upper 8 rows and gates all global reads/writes.
// ============================================================================

// SmemLayoutMG / SmemPtrsMG are parameterised on the default MG_N_HG=2 layout;
// MG_N_HG_T=1 instantiations reuse them and waste ~half the MG smem to avoid a
// full retemplate (q_nope/q_sc/m_smem/l_smem/reduce_buf/w_smem).
static constexpr int MG_N_HG_DEFAULT = SmemLayoutMG<ModelType::DSV4, QkComputeMode::FP8>::N_HG;
static constexpr int MG_HEADS_PER_CTA_DEFAULT = MG_N_HG_DEFAULT * HPB;  // 32

// Shared MG implementation for single-cache and dual-cache prefill.
// topk (the main indices row width) is runtime via cold.topk.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE, bool DUAL_CACHE,
          int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T, bool ASSUME_FULL_TILES = false>
__device__ __forceinline__ void prefill_mg_impl(
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ KV_cache_extra,  // nullptr when !DUAL_CACHE
    const int32_t* __restrict__ indices_extra,   // nullptr when !DUAL_CACHE
    bf16* __restrict__ output, float* __restrict__ out_lse,
    const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
    const PrefillColdParams& cold) {
  constexpr int MG_N_HG = MG_N_HG_T;
  constexpr int MG_HEADS_PER_CTA = MG_N_HG_T * HPB;

  const float sm_scale = cold.sm_scale;
  const int num_tokens = cold.num_tokens;
  const size_t kv_stride_bytes = cold.kv_stride_bytes;
  [[maybe_unused]] const size_t extra_page_stride_bytes =
      DUAL_CACHE ? cold.extra_page_stride_bytes : (size_t)0;
  using KV = KVCacheTraits<MT>;
  constexpr int PAGE_MAIN = PAGE_BLOCK_SIZE;
  constexpr int PAGE_EXTRA =
      PAGE_BLOCK_SIZE == 0 && PAGE_BLOCK_SIZE_EXTRA != 2 ? 0 : PAGE_BLOCK_SIZE_EXTRA;
  const PageGeom pg_main = page_geom(PAGE_MAIN == 0 ? cold.page_block_size : PAGE_MAIN,
                                     KVIOTraits<MT>::IO_STRIDE, cold.main_div);
  [[maybe_unused]] const PageGeom pg_extra =
      page_geom(PAGE_EXTRA == 0 ? cold.extra_page_block_size : PAGE_EXTRA,
                KVIOTraits<MT>::IO_STRIDE, cold.extra_div);
  constexpr bool REUSE_ADDRESS = MT == ModelType::DSV4;
  // CT pinned to FP8: XV always uses FP8 W; QkMode only flips the QK side.
  using CT = ComputeTraits<MT, QkComputeMode::FP8, BI, N_MATH_WARPS>;
  using LMG = SmemLayoutMG<MT, QkMode>;
  using SMG = SmemPtrsMG<MT, QkMode>;

  static_assert(NUM_HEADS % MG_HEADS_PER_CTA == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill MG_HEADS_PER_CTA, except a single padded head group");
  static constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA - 1) / MG_HEADS_PER_CTA;
  static constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
  static constexpr bool USE_WFP8_ROW_XOR = DUAL_CACHE && (PAGE_BLOCK_SIZE_EXTRA == 2);

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_tile = blockIdx.x % REPLICATE_H;
  const int h_start = h_tile * MG_HEADS_PER_CTA;
  if (s_i >= num_tokens) return;

  // Whole-tile main-cache tile count (the binding guarantees topk % BI == 0).
  const int NI = cold.topk / BI;
  [[maybe_unused]] int topk_len =
      ASSUME_FULL_TILES ? cold.topk
                        : (cold.topk_length ? __ldg(cold.topk_length + s_i) : cold.topk);
  if constexpr (!ASSUME_FULL_TILES) {
    topk_len = topk_len < 0 ? 0 : (topk_len > cold.topk ? cold.topk : topk_len);
  }
  const int actual_ni = ASSUME_FULL_TILES ? NI : ((topk_len + BI - 1) / BI);
  const int main_ni = ASSUME_FULL_TILES ? NI : actual_ni;

  // Dual-cache runtime lengths.
  [[maybe_unused]] int topk_len_extra = 0;
  [[maybe_unused]] int topk_extra_declared = 0;
  int ni_total = main_ni;
  if constexpr (DUAL_CACHE) {
    topk_extra_declared = cold.extra_topk;
    if constexpr (ASSUME_FULL_TILES) {
      topk_len_extra = topk_extra_declared;
    } else {
      topk_len_extra =
          cold.extra_topk_length ? __ldg(cold.extra_topk_length + s_i) : topk_extra_declared;
      topk_len_extra =
          topk_len_extra < 0
              ? 0
              : (topk_len_extra > topk_extra_declared ? topk_extra_declared : topk_len_extra);
    }
    ni_total = main_ni + (topk_len_extra + BI - 1) / BI;
  }
  // Runtime-length variants skip empty main rows before switching to the
  // secondary cache; fulltile variants intentionally consume all declared rows.
  const int loop_bound = DUAL_CACHE ? ni_total : actual_ni;

  const int warp_rank = threadIdx.x / 32;
  const int wy = warp_rank / 4;

  extern __shared__ char smem_raw[];
  auto sm = SMG::init(smem_raw);

  if (threadIdx.x == 0)
    flashinfer::sparse_mla_sm120::pipeline::BulkReady::init_slots<2>(sm.mbar_kv(0));
  bar_sync_t<Fp8PrefillSync::CTA_INIT, BLOCK_THREADS>();

  constexpr bool H32_FP8_SINGLE =
      MT == ModelType::DSV4 && NUM_HEADS == 32 && QkMode == QkComputeMode::FP8 && !DUAL_CACHE;
  constexpr int IO_REGS = H32_FP8_SINGLE ? 24 : 32;
  constexpr int MATH_REGS = H32_FP8_SINGLE ? 240 : 232;
  static_assert(MATH_REGS * MATH_THREADS + IO_REGS * IO_THREADS <= 168 * BLOCK_THREADS);

  // ── IO warps ────────────────────────────────────────────────────
  if (wy == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(IO_REGS));

    const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
    const int32_t* idx_base = indices + (size_t)s_i * cold.topk;
    [[maybe_unused]] const int32_t* idx_base_extra = nullptr;
    if constexpr (DUAL_CACHE) {
      idx_base_extra = indices_extra + (size_t)s_i * topk_extra_declared;
    }
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    // This thread's candidate index for logical tile t, staged a tile ahead of
    // use so the index LDG latency is hidden behind the previous tile's
    // gather. main_ni equals NI under ASSUME_FULL_TILES, so the main/extra
    // split needs no branch on the tile-length mode. Lanes past the runtime
    // topk length are masked out: the last tile is partial and caller padding
    // may be stale, and a garbage index would gather a wild gmem address.
    auto load_idx = [&](int t) -> int {
      if (t >= loop_bound || io_tid >= BI) return -1;
      if constexpr (DUAL_CACHE) {
        const bool is_main = t < main_ni;
        if (is_main ? (t * BI + io_tid >= topk_len)
                    : ((t - main_ni) * BI + io_tid >= topk_len_extra)) {
          return -1;
        }
        const int32_t* p = is_main ? (idx_base + t * BI) : (idx_base_extra + (t - main_ni) * BI);
        return __ldg(p + io_tid);
      } else {
        return (t * BI + io_tid < topk_len) ? __ldg(idx_base + t * BI + io_tid) : -1;
      }
    };
    // DSV4 ordinary address/scale stores use a counted CTA store handoff;
    // bulk completion alone does not publish stores from all IO threads.
    auto issue_tile = [&](int logical_ti, int buf, int staged) {
      if constexpr (REUSE_ADDRESS) {
        const uint8_t* address;
        if constexpr (DUAL_CACHE) {
          if (logical_ti >= main_ni) {
            address = prefill_kv_entry_base<MT, PAGE_EXTRA>(KV_cache_extra, staged,
                                                            extra_page_stride_bytes, pg_extra);
            io_gather_scales<MT, PAGE_EXTRA, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged,
                                                             KV_cache_extra, io_tid,
                                                             extra_page_stride_bytes, pg_extra);
          } else {
            address =
                prefill_kv_entry_base<MT, PAGE_MAIN>(KV_cache, staged, kv_stride_bytes, pg_main);
            io_gather_scales<MT, PAGE_MAIN, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged, KV_cache,
                                                            io_tid, kv_stride_bytes, pg_main);
          }
        } else {
          address =
              prefill_kv_entry_base<MT, PAGE_MAIN>(KV_cache, staged, kv_stride_bytes, pg_main);
          io_gather_scales<MT, PAGE_MAIN, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged, KV_cache,
                                                          io_tid, kv_stride_bytes, pg_main);
        }
        if (io_tid == 0) BulkReady::expect(sm.mbar_kv(buf), BI * KV::KV_SMEM_COPY_BYTES);
        if (io_tid < BI) {
          auto addresses = reinterpret_cast<const uint8_t**>(smem_raw + LMG::OFF_KV_ADDRESS);
          addresses[buf * BI + io_tid] = address;
          cp_async_bulk_g2s_l2hint(sm.kv_buf(buf) + io_tid * KV::KV_SMEM_STRIDE, address,
                                   KV::KV_SMEM_COPY_BYTES, sm.mbar_kv(buf), kv_l2_policy);
        }
        flashinfer::sparse_mla_sm120::pipeline::StoreHandoff<6, 7, IO_THREADS,
                                                             MATH_THREADS>::publish(buf);
        return;
      }
      if constexpr (DUAL_CACHE) {
        if (logical_ti >= main_ni) {
          io_gather_scales<MT, PAGE_EXTRA, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged,
                                                           KV_cache_extra, io_tid,
                                                           extra_page_stride_bytes, pg_extra);
          __threadfence_block();
          io_bulk_gather_tile<MT, PAGE_EXTRA, true, BI, IO_THREADS>(
              sm.kv_buf(buf), staged, KV_cache_extra, sm.mbar_kv(buf), io_tid,
              extra_page_stride_bytes, kv_l2_policy, pg_extra);
          return;
        }
      }
      io_gather_scales<MT, PAGE_MAIN, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged, KV_cache,
                                                      io_tid, kv_stride_bytes, pg_main);
      __threadfence_block();
      io_bulk_gather_tile<MT, PAGE_MAIN, true, BI, IO_THREADS>(
          sm.kv_buf(buf), staged, KV_cache, sm.mbar_kv(buf), io_tid, kv_stride_bytes, kv_l2_policy,
          pg_main);
    };

    int staged = load_idx(0);
    if (loop_bound > 0) {
      issue_tile(0, 0, staged);
      staged = load_idx(1);
    }

#pragma unroll 1
    for (int ti = 0; ti < loop_bound; ti++) {
      if (ti + 1 < loop_bound) {
        const int next = load_idx(ti + 2);
        issue_tile(ti + 1, (ti + 1) & 1, staged);
        staged = next;
      }
      Fp8PrefillSync::KvFree<IO_THREADS, MATH_THREADS>::acquire(ti & 1);
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(MATH_REGS));

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    const int32_t* idx_base = indices + (size_t)s_i * cold.topk;
    [[maybe_unused]] const int32_t* idx_base_extra = nullptr;
    if constexpr (DUAL_CACHE) {
      idx_base_extra = indices_extra + (size_t)s_i * topk_extra_declared;
    }

    // ── Quantize Q for both groups ─────────────────────────────
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      const bf16* q_base_g =
          Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)(h_start + g * HPB) * KV::D_QK;
      if constexpr (QkMode == QkComputeMode::BF16) {
        load_q_bf16_to_smem<MT, MATH_THREADS>(sm.q_nope_bf16(g), sm.q_rope() + g * HPB * KV::D_ROPE,
                                              q_base_g, VALID_HPB);
      } else {
        quantize_q_to_smem<MT, MATH_THREADS>(sm.q_nope_fp8(g), sm.q_nope_sc(g),
                                             sm.q_rope() + g * HPB * KV::D_ROPE, q_base_g,
                                             VALID_HPB);
      }
    }

    // Preload Q rope to registers for both groups
    QRopeRegs<MT> q_rope_regs[MG_N_HG];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
      q_rope_regs[g] = preload_q_rope_regs<MT>(sm.q_rope() + g * HPB * KV::D_ROPE, lane);

    for (int i = threadIdx.x; i < MG_N_HG * HPB; i += MATH_THREADS) sm.m_smem()[i] = -1e30f;

    // Per-group accumulators
    float acc_o[MG_N_HG][CT::ACC_TILES][4];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
#pragma unroll
      for (int t = 0; t < CT::ACC_TILES; t++)
        acc_o[g][t][0] = acc_o[g][t][1] = acc_o[g][t][2] = acc_o[g][t][3] = 0.f;

    float acc_rope[MG_N_HG][4];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
      acc_rope[g][0] = acc_rope[g][1] = acc_rope[g][2] = acc_rope[g][3] = 0.f;

    // Deferred row_sum accumulators (register-only, no smem per tile)
    float warp_l_partial[MG_N_HG][2] = {};

    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
    if (ASSUME_FULL_TILES || loop_bound > 0) BulkReady::wait(sm.mbar_kv(0), 0);

    // ── Main loop ───────────────────────────────────────────────
#pragma unroll 1
    for (int ti = 0; ti < loop_bound; ti++) {
      uint8_t* kv_smem = sm.kv_buf(ti & 1);
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

      // Per-tile data source. Single-cache: always main. Dual: route on phase.
      const int32_t* ib;
      const uint8_t* kv_global;
      size_t kv_stride_bytes_now;
      bool is_main = true;
      if constexpr (DUAL_CACHE) {
        is_main = (ti < main_ni);
        ib = is_main ? (idx_base + ti * BI) : (idx_base_extra + (ti - main_ni) * BI);
        kv_global = is_main ? KV_cache : KV_cache_extra;
        kv_stride_bytes_now = is_main ? kv_stride_bytes : extra_page_stride_bytes;
      } else {
        ib = idx_base + ti * BI;
        kv_global = KV_cache;
        kv_stride_bytes_now = kv_stride_bytes;
      }

      // The store handoff publishes every IO writer; KvFree retires all XV
      // readers before IO can overwrite this slot two tiles later.
      const uint8_t* entry_base_gid;
      if constexpr (REUSE_ADDRESS) {
        flashinfer::sparse_mla_sm120::pipeline::StoreHandoff<6, 7, IO_THREADS, MATH_THREADS>::wait(
            ti & 1);
        auto addresses = reinterpret_cast<const uint8_t* const*>(smem_raw + LMG::OFF_KV_ADDRESS);
        entry_base_gid = addresses[(ti & 1) * BI + qk_nb + gid];
      } else {
        // Position and runtime length are cache-section-relative under dual cache.
        const int section_tile = (DUAL_CACHE && !is_main) ? (ti - main_ni) : ti;
        const int len_now = (DUAL_CACHE && !is_main) ? topk_len_extra : topk_len;
        const int idx =
            mask_idx_past_len(ib[qk_nb + gid], section_tile * BI + qk_nb + gid, len_now);
        if constexpr (DUAL_CACHE) {
          if (is_main) {
            entry_base_gid =
                prefill_kv_entry_base<MT, PAGE_MAIN>(KV_cache, idx, kv_stride_bytes, pg_main);
          } else {
            entry_base_gid = prefill_kv_entry_base<MT, PAGE_EXTRA>(
                KV_cache_extra, idx, extra_page_stride_bytes, pg_extra);
          }
        } else {
          entry_base_gid =
              prefill_kv_entry_base<MT, PAGE_MAIN>(KV_cache, idx, kv_stride_bytes, pg_main);
        }
      }

      KVRopePrefetch<MT> rope_pf = prefetch_kv_rope<MT>(
          reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

      // ── QK + softmax for both groups ────────────────────────
      float scores_log2[MG_N_HG][4];
      float vsc_cache[CT::N_V_CHUNKS][2];

      // BF16 MG: both head groups consume the same KV B operand. Fuse them so
      // FP8->BF16 KV dequantization runs once per K step.
      if constexpr (QkMode == QkComputeMode::BF16 && MG_N_HG == 2) {
        const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
        float qk_grp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

        const uint8_t* k_sc = KV::SCALE_IN_KV_SMEM ? kv_gid_base + KV::D_NOPE
                                                   : sm.kv_scale_buf(ti & 1) +
                                                         (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
        qk_bf16_from_fp8_nope_16x8<KV>(qk_grp, sm.q_nope_bf16(0),
                                       int(sm.q_nope_bf16(1) - sm.q_nope_bf16(0)), kv_gid_base,
                                       k_sc, lane);

#pragma unroll
        for (int g = 0; g < 2; g++) {
          float* qk = qk_grp[g];
          compute_qk_rope<MT>(qk, q_rope_regs[g], rope_pf);

          {
            int e0 = qk_nb + tid * 2, e1 = e0 + 1;
            if (ib[e0] < 0) {
              qk[0] = -1e30f;
              qk[2] = -1e30f;
            }
            if (ib[e1] < 0) {
              qk[1] = -1e30f;
              qk[3] = -1e30f;
            }
            if constexpr (DUAL_CACHE && !ASSUME_FULL_TILES) {
              if (is_main) {
                if (cold.topk_length != nullptr) {
                  int a0 = ti * BI + e0, a1 = ti * BI + e1;
                  if (a0 >= topk_len) {
                    qk[0] = -1e30f;
                    qk[2] = -1e30f;
                  }
                  if (a1 >= topk_len) {
                    qk[1] = -1e30f;
                    qk[3] = -1e30f;
                  }
                }
              } else {
                int a0 = (ti - main_ni) * BI + e0, a1 = (ti - main_ni) * BI + e1;
                if (a0 >= topk_len_extra) {
                  qk[0] = -1e30f;
                  qk[2] = -1e30f;
                }
                if (a1 >= topk_len_extra) {
                  qk[1] = -1e30f;
                  qk[3] = -1e30f;
                }
              }
            } else if constexpr (!DUAL_CACHE && !ASSUME_FULL_TILES) {
              if (cold.topk_length != nullptr) {
                int a0 = ti * BI + e0, a1 = ti * BI + e1;
                if (a0 >= topk_len) {
                  qk[0] = -1e30f;
                  qk[2] = -1e30f;
                }
                if (a1 >= topk_len) {
                  qk[1] = -1e30f;
                  qk[3] = -1e30f;
                }
              }
            }
          }

          float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e, qk[2] * sm_scale_log2e,
                        qk[3] * sm_scale_log2e};

          float lm0, lm1;
          softmax_warp_max(s, lm0, lm1);
          if (tid == 0) {
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
          }
          scores_log2[g][0] = s[0];
          scores_log2[g][1] = s[1];
          scores_log2[g][2] = s[2];
          scores_log2[g][3] = s[3];
        }
      } else {
#pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
          const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;

          // QK nope MMA. BF16: m16n8k16 with per-thread FP8→BF16 dequant on KV.
          float qk_storage[1][4] = {};
          auto& qk = qk_storage[0];
          if constexpr (QkMode == QkComputeMode::BF16) {
            const uint8_t* k_sc =
                KV::SCALE_IN_KV_SMEM
                    ? kv_gid_base + KV::D_NOPE
                    : sm.kv_scale_buf(ti & 1) + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
            qk_bf16_from_fp8_nope_16x8<KV>(qk_storage, sm.q_nope_bf16(g), 0, kv_gid_base, k_sc,
                                           lane);
          } else {
#pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
              uint8_t sfa = fp32_exponent_byte(
                  sm.q_nope_sc(g)[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
              float acc0, acc1, acc2, acc3;
              init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
              const uint8_t* k_scale_base;
              if constexpr (KV::SCALE_IN_KV_SMEM) {
                k_scale_base = kv_gid_base + KV::D_NOPE;
              } else {
                k_scale_base = sm.kv_scale_buf(ti & 1) + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
              }
              uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
              qk_fp8_scale_group_16x8<KV>(acc0, acc1, acc2, acc3, sm.q_nope_fp8(g), kv_warp_base,
                                          blk, sfa, sfb, lane);
              const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
              const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
              commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE,
                                e1_base + KV::D_NOPE, blk);
            }
          }

          // QK rope (reuses prefetched B operands)
          compute_qk_rope<MT>(qk, q_rope_regs[g], rope_pf);

          // Invalid index masking + topk_length overflow. Dual splits per phase
          // (main: absolute ti*BI+e vs topk_len; extra: relative
          // (ti-main_ni)*BI+e vs topk_len_extra).
          {
            int e0 = qk_nb + tid * 2, e1 = e0 + 1;
            if (ib[e0] < 0) {
              qk[0] = -1e30f;
              qk[2] = -1e30f;
            }
            if (ib[e1] < 0) {
              qk[1] = -1e30f;
              qk[3] = -1e30f;
            }
            if constexpr (DUAL_CACHE && !ASSUME_FULL_TILES) {
              if (is_main) {
                if (cold.topk_length != nullptr) {
                  int a0 = ti * BI + e0, a1 = ti * BI + e1;
                  if (a0 >= topk_len) {
                    qk[0] = -1e30f;
                    qk[2] = -1e30f;
                  }
                  if (a1 >= topk_len) {
                    qk[1] = -1e30f;
                    qk[3] = -1e30f;
                  }
                }
              } else {
                int a0 = (ti - main_ni) * BI + e0, a1 = (ti - main_ni) * BI + e1;
                if (a0 >= topk_len_extra) {
                  qk[0] = -1e30f;
                  qk[2] = -1e30f;
                }
                if (a1 >= topk_len_extra) {
                  qk[1] = -1e30f;
                  qk[3] = -1e30f;
                }
              }
            } else if constexpr (!DUAL_CACHE && !ASSUME_FULL_TILES) {
              if (cold.topk_length != nullptr) {
                int a0 = ti * BI + e0, a1 = ti * BI + e1;
                if (a0 >= topk_len) {
                  qk[0] = -1e30f;
                  qk[2] = -1e30f;
                }
                if (a1 >= topk_len) {
                  qk[1] = -1e30f;
                  qk[3] = -1e30f;
                }
              }
            }
          }

          float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e, qk[2] * sm_scale_log2e,
                        qk[3] * sm_scale_log2e};

          float lm0, lm1;
          softmax_warp_max(s, lm0, lm1);
          if (tid == 0) {
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
            sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
          }
          scores_log2[g][0] = s[0];
          scores_log2[g][1] = s[1];
          scores_log2[g][2] = s[2];
          scores_log2[g][3] = s[3];
        }
      }
      RoleSync<Fp8PrefillSync::MATH, MATH_THREADS>::wait();

      // All prior XV readers retired; the next math barrier publishes these resets.
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = 0.f;

      // Cross-warp max for both groups
      if (threadIdx.x < MG_N_HG * HPB) {
        int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
        float old_m = sm.m_smem()[g * SMG::ML_GRP_STRIDE + h], tm = -1e30f;
#pragma unroll
        for (int w = 0; w < N_MATH_WARPS; w++)
          tm = fmaxf(tm, sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h]);
        float nm = fmaxf(old_m, tm);
        float alpha = exp2f(old_m - nm);
        sm.m_smem()[g * SMG::ML_GRP_STRIDE + h] = nm;
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + h] = alpha;
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + h] = nm;
      }
      RoleSync<Fp8PrefillSync::MATH, MATH_THREADS>::wait();

      // V scales are shared by both head groups; cache them once for the tile.
      const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
      const uint8_t* e0_base = kv_warp_base + tid * 2 * KV::KV_SMEM_STRIDE;
      const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        if constexpr (KV::SCALE_IN_KV_SMEM) {
          vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
          vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
        } else {
          vsc_cache[vc][0] = fp32_from_exponent_byte(
              sm.kv_scale_buf(ti & 1)[e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          vsc_cache[vc][1] = fp32_from_exponent_byte(
              sm.kv_scale_buf(ti & 1)[e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
        }
      }

      // Scores and probabilities have disjoint lifetimes in the same registers.
      auto& p = scores_log2;
      // Rescale and exponentiate weights for both groups.
#pragma unroll
      for (int g = 0; g < MG_N_HG; g++) {
        float alpha0 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + gid];
        float alpha1 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + gid + 8];
        float nm0 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid];
        float nm1 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid + 8];

        if (alpha0 < 1.0f || alpha1 < 1.0f) {
#pragma unroll
          for (int t = 0; t < CT::ACC_TILES; t++) {
            acc_o[g][t][0] *= alpha0;
            acc_o[g][t][1] *= alpha0;
            acc_o[g][t][2] *= alpha1;
            acc_o[g][t][3] *= alpha1;
          }
          if constexpr (KV::V_HAS_ROPE) {
            acc_rope[g][0] *= alpha0;
            acc_rope[g][1] *= alpha0;
            acc_rope[g][2] *= alpha1;
            acc_rope[g][3] *= alpha1;
          }
          warp_l_partial[g][0] *= alpha0;
          warp_l_partial[g][1] *= alpha1;
        }

        float w0 = exp2f(scores_log2[g][0] - nm0), w1 = exp2f(scores_log2[g][1] - nm0);
        float w2 = exp2f(scores_log2[g][2] - nm1), w3 = exp2f(scores_log2[g][3] - nm1);
        p[g][0] = w0;
        p[g][1] = w1;
        p[g][2] = w2;
        p[g][3] = w3;

        float ls0, ls1;
        softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
        warp_l_partial[g][0] += ls0;
        warp_l_partial[g][1] += ls1;

        // V-scale max for W quantization.
#pragma unroll
        for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
          float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
          float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
          float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
          atomicMax(
              reinterpret_cast<int*>(&sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid]),
              __float_as_int(fmaxf(ws00, ws01)));
          atomicMax(reinterpret_cast<int*>(
                        &sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(ws10, ws11)));
        }
      }
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // Normalize w_head_sc_all (both groups)
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // ── XV nope MMA (per-vc barrier, D2 direct B) ────────────
      {
        if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            uint8_t* wfp8_parity = sm.w_fp8() + (vc & 1) * SMG::WFP8_PARITY_STRIDE;
            float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
            float xv_acc[MG_N_HG][CT::NT_PER_WARP_XV][4] = {0};
#pragma unroll
            for (int wpass = 0; wpass < 2; ++wpass) {
              if (wpass > 0) bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
#pragma unroll
              for (int g = 0; g < MG_N_HG; g++) {
                float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
                uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
                float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                float w0 = p[g][0], w1 = p[g][1];
                float w2 = p[g][2], w3 = p[g][3];
                float wn00 = w0 * vsc0 * si0, wn01 = w1 * vsc1 * si0;
                float wn10 = w2 * vsc0 * si1, wn11 = w3 * vsc1 * si1;
                Fp8WeightQuad wq =
                    quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
                int wrow0 = gid, wrow1 = gid + 8;
                if constexpr (USE_WFP8_ROW_XOR) {
                  wrow0 = wfp8_row_xor(wrow0);
                  wrow1 = wfp8_row_xor(wrow1);
                }
                p_vscale_fp8[wrow0 * (BI + 16) + e0i] = wq.h0_e0;
                p_vscale_fp8[wrow0 * (BI + 16) + e1i] = wq.h0_e1;
                p_vscale_fp8[wrow1 * (BI + 16) + e0i] = wq.h1_e0;
                p_vscale_fp8[wrow1 * (BI + 16) + e1i] = wq.h1_e1;
              }
              bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

#pragma unroll
              for (int g = 0; g < MG_N_HG; g++) {
                uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
                for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                  int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                  pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, BI + 16, CT::XV_KSTEPS, USE_WFP8_ROW_XOR>(
                      xv_acc[g][nt], p_vscale_fp8, kv_smem, dim, lane);
                }
              }
            }

#pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
              float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                acc_o[g][ti_acc][0] += xv_acc[g][nt][0] * sc0;
                acc_o[g][ti_acc][1] += xv_acc[g][nt][1] * sc0;
                acc_o[g][ti_acc][2] += xv_acc[g][nt][2] * sc1;
                acc_o[g][ti_acc][3] += xv_acc[g][nt][3] * sc1;
              }
            }
          }
        } else {
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            // W_FP8 ping-pong: writes go to buf[vc&1], reads go to same buf;
            // next vc writes to buf[(vc+1)&1] in parallel with this vc's reads.
            uint8_t* wfp8_parity = sm.w_fp8() + (vc & 1) * SMG::WFP8_PARITY_STRIDE;
            float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
#pragma unroll
            for (int g = 0; g < MG_N_HG; g++) {
              float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
              uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
              float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
              float w0 = p[g][0], w1 = p[g][1];
              float w2 = p[g][2], w3 = p[g][3];
              float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
              float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
              // vc_sc already bounds the normalized weights; FP8 conversion saturates rounding
              // overshoot.
              __nv_fp8_e4m3 f00(ws00 * si0);
              __nv_fp8_e4m3 f01(ws01 * si0);
              __nv_fp8_e4m3 f10(ws10 * si1);
              __nv_fp8_e4m3 f11(ws11 * si1);
              int wrow0 = gid, wrow1 = gid + 8;
              if constexpr (USE_WFP8_ROW_XOR) {
                wrow0 = wfp8_row_xor(wrow0);
                wrow1 = wfp8_row_xor(wrow1);
              }
              p_vscale_fp8[wrow0 * (BI + 16) + e0i] = f00.__x;
              p_vscale_fp8[wrow0 * (BI + 16) + e1i] = f01.__x;
              p_vscale_fp8[wrow1 * (BI + 16) + e0i] = f10.__x;
              p_vscale_fp8[wrow1 * (BI + 16) + e1i] = f11.__x;
            }
            bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

            // Both head groups use the same V B operand for a given chunk/dim.
            if constexpr (MG_N_HG == 2) {
              float* vc_sc0 = sm.w_head_sc_all() + vc * HPB;
              float* vc_sc1 = sm.w_head_sc_all() + SMG::WSC_GRP_STRIDE + vc * HPB;
              uint8_t* p_vscale_fp8_g0 = wfp8_parity;
              uint8_t* p_vscale_fp8_g1 = wfp8_parity + SMG::WFP8_GRP_SIZE;
#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                float xv0[4] = {0.f, 0.f, 0.f, 0.f};
                float xv1[4] = {0.f, 0.f, 0.f, 0.f};
                pv_fp8_d2_16x8_pair<KV::KV_SMEM_STRIDE, BI + 16, CT::XV_KSTEPS, USE_WFP8_ROW_XOR>(
                    xv0, xv1, p_vscale_fp8_g0, p_vscale_fp8_g1, kv_smem, dim, lane);
                float sc00 = vc_sc0[gid], sc01 = vc_sc0[gid + 8];
                acc_o[0][ti_acc][0] += xv0[0] * sc00;
                acc_o[0][ti_acc][1] += xv0[1] * sc00;
                acc_o[0][ti_acc][2] += xv0[2] * sc01;
                acc_o[0][ti_acc][3] += xv0[3] * sc01;

                float sc10 = vc_sc1[gid], sc11 = vc_sc1[gid + 8];
                acc_o[1][ti_acc][0] += xv1[0] * sc10;
                acc_o[1][ti_acc][1] += xv1[1] * sc10;
                acc_o[1][ti_acc][2] += xv1[2] * sc11;
                acc_o[1][ti_acc][3] += xv1[3] * sc11;
              }
            } else {
#pragma unroll
              for (int g = 0; g < MG_N_HG; g++) {
                float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
                uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
                for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                  int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                  int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                  float xv[4] = {0.f, 0.f, 0.f, 0.f};
                  pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, BI + 16, CT::XV_KSTEPS, USE_WFP8_ROW_XOR>(
                      xv, p_vscale_fp8, kv_smem, dim, lane);
                  float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
                  acc_o[g][ti_acc][0] += xv[0] * sc0;
                  acc_o[g][ti_acc][1] += xv[1] * sc0;
                  acc_o[g][ti_acc][2] += xv[2] * sc1;
                  acc_o[g][ti_acc][3] += xv[3] * sc1;
                }
              }
            }
            // W_FP8 ping-pong keeps adjacent chunks in different buffers; the
            // surrounding visibility barriers cover each chunk.
          }
        }
      }

      // ── XV rope BF16 MMA (DSV4, both groups) ──────────────
      if constexpr (KV::V_HAS_ROPE) {
        bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
        // DSV4 IO already maps negative indices and stale padding to the zero row.
        const int valid_len = (DUAL_CACHE && !is_main)
                                  ? min(BI, topk_len_extra - (ti - main_ni) * BI)
                                  : min(BI, topk_len - ti * BI);
        if constexpr (REUSE_ADDRESS) {
          auto addresses = reinterpret_cast<const uint8_t* const*>(smem_raw + LMG::OFF_KV_ADDRESS);
          xv_rope_mma_mg<MT, PAGE_MAIN, MG_N_HG>(
              acc_rope, p, ib, valid_len, kv_global, mwarp, lane, kv_stride_bytes_now,
              reinterpret_cast<bf16*>(sm.w_fp8()), addresses + (ti & 1) * BI);
        } else if constexpr (DUAL_CACHE) {
          if (is_main) {
            xv_rope_mma_mg<MT, PAGE_MAIN, MG_N_HG>(
                acc_rope, p, ib, valid_len, kv_global, mwarp, lane, kv_stride_bytes_now,
                reinterpret_cast<bf16*>(sm.w_fp8()), nullptr, pg_main);
          } else {
            xv_rope_mma_mg<MT, PAGE_EXTRA, MG_N_HG>(
                acc_rope, p, ib, valid_len, kv_global, mwarp, lane, kv_stride_bytes_now,
                reinterpret_cast<bf16*>(sm.w_fp8()), nullptr, pg_extra);
          }
        } else {
          xv_rope_mma_mg<MT, PAGE_MAIN, MG_N_HG>(
              acc_rope, p, ib, valid_len, kv_global, mwarp, lane, kv_stride_bytes_now,
              reinterpret_cast<bf16*>(sm.w_fp8()), nullptr, pg_main);
        }
      }
      Fp8PrefillSync::KvFree<IO_THREADS, MATH_THREADS>::release(ti & 1);
      if (ti + 1 < loop_bound) {
        const int next_phase = ((ti + 1) >> 1) & 1;
        BulkReady::wait(sm.mbar_kv((ti + 1) & 1), next_phase);
      }
    }

// ── Finalize deferred row_sum ───────────────────────────────
// Write warp_l_partial to smem for cross-warp reduction
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      if (tid == 0) {
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = warp_l_partial[g][0];
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = warp_l_partial[g][1];
      }
    }
    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

    if (threadIdx.x < MG_N_HG * HPB) {
      int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < N_MATH_WARPS; w++)
        ts += sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h];
      sm.l_smem()[g * SMG::ML_GRP_STRIDE + h] = ts;
    }
    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

    // ── Epilogue: BF16 output for both groups (serial) ─────────
    // Reuse kv_bufs[0] for BF16 staging (16KB needed, 29-33KB available)
    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_buf(0));
    constexpr int BF16_STAGING_STRIDE = KV::D_V;
    constexpr size_t h_stride = KV::D_V;
    constexpr size_t token_stride = (size_t)NUM_HEADS * KV::D_V;

#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      // attn_sink folded into the normalizer (FlashMLA V4 convention).
      // See SG epilogue for full derivation.
      float il0, il1;
      if (cold.attn_sink != nullptr) {
        int h0 = h_start + g * HPB + gid;
        float s0 = __ldg(cold.attn_sink + h0) * LOG2E;
        float d0 = sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid] +
                   exp2f(s0 - sm.m_smem()[g * SMG::ML_GRP_STRIDE + gid]);
        il0 = (d0 > 0.f) ? (1.f / d0) : 0.f;
        if constexpr (VALID_HPB > 8) {
          float s1 = __ldg(cold.attn_sink + h0 + 8) * LOG2E;
          float d1 = sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8] +
                     exp2f(s1 - sm.m_smem()[g * SMG::ML_GRP_STRIDE + gid + 8]);
          il1 = (d1 > 0.f) ? (1.f / d1) : 0.f;
        } else {
          il1 = 0.f;
        }
      } else {
        il0 = (sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid] > 0.f)
                  ? (1.f / sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid])
                  : 0.f;
        if constexpr (VALID_HPB > 8) {
          il1 = (sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8] > 0.f)
                    ? (1.f / sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8])
                    : 0.f;
        } else {
          il1 = 0.f;
        }
      }

#pragma unroll
      for (int t = 0; t < CT::ACC_TILES; t++) {
        constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
        int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
        int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
        staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[g][t][0] * il0);
        staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[g][t][1] * il0);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[g][t][2] * il1);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] =
            __float2bfloat16(acc_o[g][t][3] * il1);
      }

      if constexpr (KV::V_HAS_ROPE) {
        int n_start = mwarp * 8;
        int d0 = KV::D_NOPE + n_start + tid * 2;
        staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[g][0] * il0);
        staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_rope[g][1] * il0);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[g][2] * il1);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] =
            __float2bfloat16(acc_rope[g][3] * il1);
      }
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // Coalesced write
      {
        const int g_h_start = h_start + g * HPB;
        const size_t out_base = (size_t)s_i * token_stride + (size_t)g_h_start * h_stride;
        constexpr int BF16_PER_STORE = 8;
        constexpr int STORES_PER_HEAD = KV::D_V / BF16_PER_STORE;
        for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
          int h = idx / STORES_PER_HEAD;
          int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
          uint4 v = *reinterpret_cast<const uint4*>(&staging_bf16[h * BF16_STAGING_STRIDE + d8]);
          *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
        }
      }

      // Write LSE for this group (merged with attn_sink if present)
      if (threadIdx.x < VALID_HPB) {
        int h = threadIdx.x;
        float lse = softmax_lse(sm.m_smem()[g * SMG::ML_GRP_STRIDE + h],
                                sm.l_smem()[g * SMG::ML_GRP_STRIDE + h]);
        if (cold.attn_sink != nullptr) {
          float sink_log2 = __ldg(cold.attn_sink + h_start + g * HPB + h) * LOG2E;
          if (lse != -1e30f)
            lse += log2f(1.f + exp2f(sink_log2 - lse));
          else
            lse = sink_log2;
        }
        size_t lse_idx = (size_t)s_i * cold.out_lse_stride_elems + (h_start + g * HPB + h);
        out_lse[lse_idx] = lse;
      }

      if (g < MG_N_HG - 1) bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
    }
  }
}

// Single-cache __global__ wrapper.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    sparse_mla_prefill_mg_kernel(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                                 const int32_t* __restrict__ indices, bf16* __restrict__ output,
                                 float* __restrict__ out_lse,
                                 const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                                 __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/false,
                  /*PAGE_BLOCK_SIZE_EXTRA=*/PAGE_BLOCK_SIZE, MG_N_HG_T>(
      Q, KV_cache, indices, /*KV_cache_extra=*/nullptr, /*indices_extra=*/nullptr, output, out_lse,
      attn_sink, cold);
}

// Dual-cache __global__ wrapper. topk and topk_extra are runtime;
// PAGE_BLOCK_SIZE_EXTRA stays template because it changes the KV stride.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T = MG_N_HG_DEFAULT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    sparse_mla_prefill_mg_dual_kernel(const bf16* __restrict__ Q,
                                      const uint8_t* __restrict__ KV_cache,
                                      const int32_t* __restrict__ indices,
                                      const uint8_t* __restrict__ KV_cache_extra,
                                      const int32_t* __restrict__ indices_extra,
                                      bf16* __restrict__ output, float* __restrict__ out_lse,
                                      const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                                      __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/true,
                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>(
      Q, KV_cache, indices, KV_cache_extra, indices_extra, output, out_lse, attn_sink, cold);
}

// Dual-cache full-tile wrapper for fixed-length inputs.
template <ModelType MT, int NUM_HEADS, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1) sparse_mla_prefill_mg_dual_fulltile_kernel(
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices, const uint8_t* __restrict__ KV_cache_extra,
    const int32_t* __restrict__ indices_extra, bf16* __restrict__ output,
    float* __restrict__ out_lse, const float* __restrict__ attn_sink,
    __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<MT, QkComputeMode::BF16, NUM_HEADS, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/true,
                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T, /*ASSUME_FULL_TILES=*/true>(
      Q, KV_cache, indices, KV_cache_extra, indices_extra, output, out_lse, attn_sink, cold);
}
