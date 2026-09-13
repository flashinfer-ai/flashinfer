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
#include "../dsv41_fp8/prefill_gather.cuh"
#include "../dsv41_fp8/resources.cuh"
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
// Sparse MLA Prefill Kernel — single-pass (no split-KV, no combine)
//
// Structurally identical to decode main loop (QK→softmax→XV), but:
//   - Iterates over ALL NI = topk/BI tiles (no split)
//   - Writes direct BF16 output (no partial_O + combine)
//   - No PDL (no dependent kernel)
//
// Template params (all constexpr):
//   MT:              ModelType (DSV3_2 / DSV4 / GLM_NSA / GLM53_NOPE / DOTS3_SWA / DSV4_1)
//   QkMode:              QkComputeMode (FP8 / BF16) for the QK MMA; XV is always FP8
//   NUM_HEADS:       8, 16, 32, 64, 128 (NUM_HEADS < HPB=16 zero-pads + gates;
//                    NUM_HEADS > HPB replicates one CTA per 16-head tile)
//   PAGE_BLOCK_SIZE: 64; DSV4_1 gathers instead use cold's runtime page sizes.
//
// topk is runtime (cold.topk): the indices row width, a whole number of
// PrefillTileCfg<MT>::BI candidate tiles.
// ============================================================================

// ── Producer/consumer math path (SPLIT_QK_XV tiles: DOTS3_SWA) ─────────────
// The serial loop below alternates QK (QK_WARPS warps) and XV (all math
// warps) per tile, which parks the non-QK warps at barriers through
// QK+softmax+W-quant; at BI=32 that idle time is ~45% of all warp samples.
// Here the two halves pipeline instead: the QK warps produce w_fp8 +
// w_head_sc + alpha for tile ti+1 into one parity buffer while the remaining
// warps run the XV MMA for tile ti out of the other.
//
// Handoffs (named barriers; alternating id pairs per the #3700 rule):
//   w_ready  (ids 6/7):  producer arrives, consumer syncs — w_fp8/w_sc/alpha
//                        for the tile are visible. Implies the tile's KV smem
//                        landed (the producer waited mbar_kv before QK).
//   w_free   (ids 8/9):  consumer arrives, producer syncs before reusing the
//                        parity buffer two tiles later.
//   KV release (ids 1/5): both math groups arrive (the consumer for its XV
//                        reads, the producer for its QK/vsc reads of the same
//                        buffer), IO syncs before refilling the KV buffer.
// Softmax state (m, row sums) stays in producer registers; the consumer
// receives only the per-head alpha. The epilogue hands l_smem/m_smem across
// once (id 4) so the consumer can normalize and store the output. Producer
// internals use barrier id 2 with QK_THREADS, the consumer's staging
// readback uses id 10 with XV_THREADS — distinct ids because the two groups
// sync independently.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          typename GatherSchedule>
__device__ __forceinline__ void sparse_mla_prefill_math_pc(
    SmemPtrs<MT, QkMode, PrefillTileCfg<MT>::BI, PrefillTileCfg<MT>::MATH_WARPS> sm,
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices, const float* __restrict__ attn_sink,
    bf16* __restrict__ output, float* __restrict__ out_lse, const PrefillColdParams& cold, int s_i,
    int h_start, int topk_len) {
  using KV = KVCacheTraits<MT>;
  using Cfg = PrefillTileCfg<MT>;
  // CT pinned to FP8: XV always uses FP8 W; QkMode only flips the QK side. Only
  // warp-count-independent members (N_V_CHUNKS, V_CHUNK, W_FP8_STRIDE) are read
  // through CT, so it takes the XV warp count: at the full MATH_WARPS a
  // 32-wide-group model (DSV4_1) would floor NT_PER_WARP_XV to 0 and trip the
  // ComputeTraits assert even though no QK warp runs the XV mapping.
  using CT = ComputeTraits<MT, QkComputeMode::FP8, Cfg::BI, Cfg::MATH_WARPS - Cfg::QK_WARPS>;
  using CT_XV = ComputeTraits<MT, QkComputeMode::FP8, Cfg::BI, Cfg::MATH_WARPS - Cfg::QK_WARPS>;
  using L = SmemLayout<MT, QkMode, Cfg::BI, Cfg::MATH_WARPS>;
  static_assert(Cfg::SPLIT_QK_XV, "the producer/consumer path requires a QK/XV warp split");
  static_assert(!KV::V_HAS_ROPE, "the producer/consumer path has no XV rope MMA");
  static_assert(KV::SCALE_FORMAT != ScaleFormat::ARBITRARY_FP32,
                "the producer/consumer path implements only the batched W quantization");

  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
  constexpr int QK_WARPS = Cfg::QK_WARPS;
  constexpr int QK_THREADS = Cfg::QK_THREADS;
  constexpr int XV_WARPS = Cfg::MATH_WARPS - QK_WARPS;
  constexpr int XV_THREADS = XV_WARPS * 32;
  using WeightsReady = Fp8PrefillSync::WeightsReady<QK_THREADS, XV_THREADS>;
  using WeightsFree = Fp8PrefillSync::WeightsFree<QK_THREADS, XV_THREADS>;
  using KvFree = Fp8PrefillSync::KvFree<Cfg::IO_THREADS, Cfg::MATH_THREADS>;

  const float sm_scale = cold.sm_scale;
  const size_t page_stride_bytes = cold.page_stride_bytes;
  const int topk = cold.topk;
  const int extra_len = prefill_extra_length(cold, s_i);
  const int actual_ni = (topk_len + Cfg::BI - 1) / Cfg::BI + (extra_len + Cfg::BI - 1) / Cfg::BI;
  auto section = [&](int tile) {
    return prefill_section<Cfg::BI, PAGE_BLOCK_SIZE>(cold, KV_cache, indices, s_i, topk_len,
                                                     extra_len, tile);
  };
  auto wait_tile = [&](int tile) { GatherSchedule::template wait<Cfg>(sm, section(tile), tile); };

  const int warp_rank = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int mwarp = warp_rank;
  const int gid = lane >> 2, tid = lane & 3;
  const float sm_scale_log2e = sm_scale * LOG2E;
  const int32_t* idx_base = indices + (size_t)s_i * topk;

  if (mwarp < QK_WARPS) {
    // ── Producer: QK + softmax + W quant ─────────────────────────
    const int qk_nb = mwarp * Cfg::ENTRIES_PER_WARP;
    const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)h_start * KV::D_QK;

    if constexpr (QkMode == QkComputeMode::BF16) {
      load_q_bf16_to_smem<MT, QK_THREADS>(sm.q_nope_bf16, sm.q_rope, q_base, VALID_HPB);
    } else {
      quantize_q_to_smem<MT, QK_THREADS>(sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base, VALID_HPB);
    }
    QRopeRegs<MT> q_rope_regs = preload_q_rope_regs<MT>(sm.q_rope, lane);

    // BI=32 tiles are too short to cover the index->rope address-chain
    // latency, so the three per-tile index reads run one tile ahead in
    // registers.
    static_assert(Cfg::BI < 64, "a BI=64 split tile should re-examine staging");
    int idx_rope_cur = 0, mask0_cur = 0, mask1_cur = 0;
    if (actual_ni > 0) {
      const auto sec = section(0);
      idx_rope_cur = sec.index(qk_nb + gid, Cfg::BI);
      mask0_cur = sec.index(qk_nb + tid * 2, Cfg::BI);
      mask1_cur = sec.index(qk_nb + tid * 2 + 1, Cfg::BI);
    }

    float warp_l[2] = {0.f, 0.f};
    float m0 = -1e30f, m1 = -1e30f;

    bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();
    if (actual_ni > 0) wait_tile(0);

#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      const int buf = ti & 1;
      uint8_t* kv_smem = sm.kv_bufs[buf];
      uint8_t* wfp8_buf = sm.w_fp8 + buf * L::SMEM_W_FP8_ONE_PARITY;
      float* wsc_buf = sm.w_head_sc_all + buf * (CT::N_V_CHUNKS * HPB);
      float* alpha_buf = sm.alpha_buf + buf * HPB;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

      // Advance the staged index reads (issue ti+1's loads, consume ti's).
      int idx_rope_nxt = 0, mask0_nxt = 0, mask1_nxt = 0;
      if (ti + 1 < actual_ni) {
        const auto sec = section(ti + 1);
        idx_rope_nxt = sec.index(qk_nb + gid, Cfg::BI);
        mask0_nxt = sec.index(qk_nb + tid * 2, Cfg::BI);
        mask1_nxt = sec.index(qk_nb + tid * 2 + 1, Cfg::BI);
      }
      const int idx_rope = idx_rope_cur, mask0 = mask0_cur, mask1 = mask1_cur;
      idx_rope_cur = idx_rope_nxt;
      mask0_cur = mask0_nxt;
      mask1_cur = mask1_nxt;

      const uint8_t* entry_base_gid =
          prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(KV_cache, idx_rope, page_stride_bytes);
      KVRopePrefetch<MT> rope_pf = prefetch_kv_rope<MT>(
          reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

      // The handoff buffers were last consumed by tile ti-2; the wait sits
      // after the QK MMA's operand setup so a fast consumer is not on the
      // critical path.
      if (ti >= 2) WeightsFree::acquire(buf);
      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += QK_THREADS) wsc_buf[i] = 0.f;

      // ── QK nope MMA ─────────────────────
      float qk[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
      if constexpr (QkMode == QkComputeMode::BF16) {
#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
          float scale_f;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
          } else {
            scale_f = fp32_from_exponent_byte(
                sm.kv_scale_bufs[buf][(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
          }
#pragma unroll
          for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
            int ko = blk * KV::QUANT_TILE + ks * 16;
            uint32_t a0, a1, a2, a3;
            ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_nope_bf16 + ko, KV::Q_NOPE_BF16_STRIDE, lane);
            uint16_t p0 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid);
            uint16_t p1 = *reinterpret_cast<const uint16_t*>(kv_gid_base + ko + 2 * tid + 8);
            uint32_t f16x2_0, f16x2_1;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_0) : "h"(p0));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_1) : "h"(p1));
            __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
            __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
            float fk0 = __low2float(h2_0) * scale_f, fk1 = __high2float(h2_0) * scale_f;
            float fk2 = __low2float(h2_1) * scale_f, fk3 = __high2float(h2_1) * scale_f;
            uint32_t b0, b1;
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b0) : "f"(fk1), "f"(fk0));
            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b1) : "f"(fk3), "f"(fk2));
            MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
            qk[0] = r.d0;
            qk[1] = r.d1;
            qk[2] = r.d2;
            qk[3] = r.d3;
          }
        }
      } else {
#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
          uint8_t sfa =
              fp32_exponent_byte(sm.q_nope_sc[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
          float acc0, acc1, acc2, acc3;
          init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
          const uint8_t* k_scale_base;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            k_scale_base = kv_gid_base + KV::D_NOPE;
          } else {
            k_scale_base = sm.kv_scale_bufs[buf] + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
          }
          uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
          qk_fp8_scale_group_16x8<KV>(acc0, acc1, acc2, acc3, sm.q_nope_fp8, kv_warp_base, blk, sfa,
                                      sfb, lane);
          const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
          const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
          commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE, e1_base + KV::D_NOPE,
                            blk);
        }
      }

      compute_qk_rope<MT>(qk, q_rope_regs, rope_pf);

      if (mask0 < 0) qk[0] = qk[2] = -1e30f;
      if (mask1 < 0) qk[1] = qk[3] = -1e30f;

      float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e, qk[2] * sm_scale_log2e,
                    qk[3] * sm_scale_log2e};
      float lm0, lm1;
      softmax_warp_max(s, lm0, lm1);
      if (tid == 0) {
        sm.reduce_buf[mwarp * HPB + gid] = lm0;
        sm.reduce_buf[mwarp * HPB + gid + 8] = lm1;
      }
      bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();

      // Redundant per-thread reduction over the producer warps' row maxima;
      // identical ops on identical inputs as the serial path's reduction.
      float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
      for (int w = 0; w < QK_WARPS; w++) {
        tm0 = fmaxf(tm0, sm.reduce_buf[w * HPB + gid]);
        tm1 = fmaxf(tm1, sm.reduce_buf[w * HPB + gid + 8]);
      }
      float nm0 = fmaxf(m0, tm0), nm1 = fmaxf(m1, tm1);
      float alpha0 = exp2f(m0 - nm0), alpha1 = exp2f(m1 - nm1);
      m0 = nm0;
      m1 = nm1;
      alpha_buf[gid] = alpha0;
      alpha_buf[gid + 8] = alpha1;

      float w0 = mask0 < 0 ? 0.f : exp2f(s[0] - nm0);
      float w1 = mask1 < 0 ? 0.f : exp2f(s[1] - nm0);
      float w2 = mask0 < 0 ? 0.f : exp2f(s[2] - nm1);
      float w3 = mask1 < 0 ? 0.f : exp2f(s[3] - nm1);
      float ls0, ls1;
      softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
      warp_l[0] = warp_l[0] * alpha0 + ls0;
      warp_l[1] = warp_l[1] * alpha1 + ls1;

      // V scale cache + atomicMax.
      const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
      const uint8_t* e0_base = kv_smem + (size_t)e0i * KV::KV_SMEM_STRIDE;
      const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
      float vsc_cache[CT::N_V_CHUNKS][2];
#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        if constexpr (KV::SCALE_IN_KV_SMEM) {
          vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
          vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
        } else {
          vsc_cache[vc][0] =
              fp32_from_exponent_byte(sm.kv_scale_bufs[buf][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          vsc_cache[vc][1] =
              fp32_from_exponent_byte(sm.kv_scale_bufs[buf][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
        }
        float ws00 = w0 * vsc_cache[vc][0], ws01 = w1 * vsc_cache[vc][1];
        float ws10 = w2 * vsc_cache[vc][0], ws11 = w3 * vsc_cache[vc][1];
        atomicMax(reinterpret_cast<int*>(&wsc_buf[vc * HPB + gid]),
                  __float_as_int(fmaxf(ws00, ws01)));
        atomicMax(reinterpret_cast<int*>(&wsc_buf[vc * HPB + gid + 8]),
                  __float_as_int(fmaxf(ws10, ws11)));
      }
      bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();

      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += QK_THREADS)
        wsc_buf[i] = fmaxf(wsc_buf[i], 1e-10f) / FP8_MAX;
      bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();

      // ── Batch W quant for all chunks ──────────────────────
#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        float* vc_sc = wsc_buf + vc * HPB;
        uint8_t* wfp8 = wfp8_buf + vc * L::SMEM_W_FP8_ONE;
        float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
        float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
        float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
        float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
        __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
        __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
        __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
        __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
        wfp8[gid * CT::W_FP8_STRIDE + e0i] = f00.__x;
        wfp8[gid * CT::W_FP8_STRIDE + e1i] = f01.__x;
        wfp8[(gid + 8) * CT::W_FP8_STRIDE + e0i] = f10.__x;
        wfp8[(gid + 8) * CT::W_FP8_STRIDE + e1i] = f11.__x;
      }
      WeightsReady::publish(buf);
      KvFree::release(buf);

      if (ti + 1 < actual_ni) wait_tile(ti + 1);
    }

    // ── Producer epilogue: row sums, m/l publish, LSE ────────────
    if (tid == 0) {
      sm.reduce_buf[mwarp * HPB + gid] = warp_l[0];
      sm.reduce_buf[mwarp * HPB + gid + 8] = warp_l[1];
    }
    bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();
    if (threadIdx.x < HPB) {
      int h = threadIdx.x;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < QK_WARPS; w++) ts += sm.reduce_buf[w * HPB + h];
      sm.l_smem[h] = ts;
    }
    if (tid == 0) {
      sm.m_smem[gid] = m0;
      sm.m_smem[gid + 8] = m1;
    }
    bar_sync_t<Fp8PrefillSync::QK, QK_THREADS>();

    if (threadIdx.x < VALID_HPB) {
      int h = threadIdx.x;
      float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
      if (attn_sink != nullptr) {
        float sink_log2 = __ldg(attn_sink + h_start + h) * LOG2E;
        if (lse != -1e30f)
          lse += log2f(1.f + exp2f(sink_log2 - lse));
        else
          lse = sink_log2;
      }
      size_t lse_idx = (size_t)s_i * cold.out_lse_stride_elems + h_start + h;
      out_lse[lse_idx] = lse;
    }
    // Publish l_smem/m_smem to the consumer group for the output normalizer.
    bar_sync_t<Fp8PrefillSync::NORMALIZER, Cfg::MATH_THREADS>();

  } else {
    // ── Consumer: XV MMA + output epilogue ───────────────────────
    const int xw = mwarp - QK_WARPS;

    float acc_o[CT_XV::ACC_TILES][4];
#pragma unroll
    for (int t = 0; t < CT_XV::ACC_TILES; t++)
      acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      const int buf = ti & 1;
      const uint8_t* kv_smem = sm.kv_bufs[buf];
      const uint8_t* wfp8_buf = sm.w_fp8 + buf * L::SMEM_W_FP8_ONE_PARITY;
      const float* wsc_buf = sm.w_head_sc_all + buf * (CT::N_V_CHUNKS * HPB);
      const float* alpha_buf = sm.alpha_buf + buf * HPB;

      WeightsReady::wait(buf);

      const float alpha0 = alpha_buf[gid], alpha1 = alpha_buf[gid + 8];
      if (alpha0 < 1.0f || alpha1 < 1.0f) {
#pragma unroll
        for (int t = 0; t < CT_XV::ACC_TILES; t++) {
          acc_o[t][0] *= alpha0;
          acc_o[t][1] *= alpha0;
          acc_o[t][2] *= alpha1;
          acc_o[t][3] *= alpha1;
        }
      }

#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        const float* vc_sc = wsc_buf + vc * HPB;
        const uint8_t* wfp8 = wfp8_buf + vc * L::SMEM_W_FP8_ONE;
#pragma unroll
        for (int nt = 0; nt < CT_XV::NT_PER_WARP_XV; nt++) {
          int ti_acc = vc * CT_XV::NT_PER_WARP_XV + nt;
          int dim = vc * CT::V_CHUNK + xw * (CT_XV::NT_PER_WARP_XV * 8) + nt * 8;
          float xv[4] = {0.f, 0.f, 0.f, 0.f};
          pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, CT::W_FP8_STRIDE, CT_XV::XV_KSTEPS>(xv, wfp8, kv_smem,
                                                                                 dim, lane);
          float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
          acc_o[ti_acc][0] += xv[0] * sc0;
          acc_o[ti_acc][1] += xv[1] * sc0;
          acc_o[ti_acc][2] += xv[2] * sc1;
          acc_o[ti_acc][3] += xv[3] * sc1;
        }
      }
      WeightsFree::release(buf);
      KvFree::release(buf);
    }

    // ── Consumer epilogue: normalize + BF16 store ────────────────
    bar_sync_t<Fp8PrefillSync::NORMALIZER, Cfg::MATH_THREADS>();

    float il0, il1;
    if (attn_sink != nullptr) {
      float s0 = __ldg(attn_sink + h_start + gid) * LOG2E;
      float denom0 = sm.l_smem[gid] + exp2f(s0 - sm.m_smem[gid]);
      il0 = (denom0 > 0.f) ? (1.f / denom0) : 0.f;
      if constexpr (VALID_HPB > 8) {
        float s1 = __ldg(attn_sink + h_start + gid + 8) * LOG2E;
        float denom1 = sm.l_smem[gid + 8] + exp2f(s1 - sm.m_smem[gid + 8]);
        il1 = (denom1 > 0.f) ? (1.f / denom1) : 0.f;
      } else {
        il1 = 0.f;
      }
    } else {
      il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
      il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;
    }

    // The main loop is done with the KV double buffer; reuse its first half
    // to stage the BF16 output (same fit argument as the serial path).
    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_bufs[0]);
    constexpr int BF16_STAGING_STRIDE = KV::D_V;
    static_assert(HPB * BF16_STAGING_STRIDE * sizeof(bf16) <= L::SMEM_KV_BUF,
                  "BF16 output staging overruns the KV buffer it reuses");

#pragma unroll
    for (int t = 0; t < CT_XV::ACC_TILES; t++) {
      constexpr int _NT8 = CT_XV::NT_PER_WARP_XV * 8;
      int c = t / CT_XV::NT_PER_WARP_XV, lnt = t % CT_XV::NT_PER_WARP_XV;
      int d0 = c * CT::V_CHUNK + xw * _NT8 + lnt * 8 + tid * 2;
      staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[t][0] * il0);
      staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[t][1] * il0);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[t][2] * il1);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[t][3] * il1);
    }
    bar_sync_t<Fp8PrefillSync::OUTPUT, XV_THREADS>();

    // Coalesced BF16 write: uint4 = 128-bit = 8 bf16 per store.
    {
      constexpr size_t h_stride = KV::D_V;
      constexpr size_t token_stride = (size_t)NUM_HEADS * KV::D_V;
      const size_t out_base = (size_t)s_i * token_stride + (size_t)h_start * h_stride;
      constexpr int BF16_PER_STORE = 8;
      constexpr int STORES_PER_HEAD = KV::D_V / BF16_PER_STORE;
      for (int idx = threadIdx.x - QK_THREADS; idx < VALID_HPB * STORES_PER_HEAD;
           idx += XV_THREADS) {
        int h = idx / STORES_PER_HEAD;
        int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
        uint4 v = *reinterpret_cast<const uint4*>(&staging_bf16[h * BF16_STAGING_STRIDE + d8]);
        *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
      }
    }
  }
}

template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          typename GatherSchedule = Dsv41PrefillGatherSchedule>
__global__ void __launch_bounds__((Fp8PrefillResources<MT, QkMode, GatherSchedule>::BLOCK_THREADS),
                                  1)
    sparse_mla_prefill_kernel(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                              const int32_t* __restrict__ indices,
                              const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                              bf16* __restrict__ output, float* __restrict__ out_lse,
                              __grid_constant__ const PrefillColdParams cold) {
  const float sm_scale = cold.sm_scale;
  const int num_tokens = cold.num_tokens;
  const size_t page_stride_bytes = cold.page_stride_bytes;
  using KV = KVCacheTraits<MT>;
  using Cfg = PrefillTileCfg<MT>;
  // CT pinned to FP8: XV always uses FP8 W; QkMode only flips the QK side.
  // On a split tile the serial tail below is unreachable (the pc path above
  // returns first), but it is still instantiated — use the XV warp count there
  // so the ComputeTraits NT_PER_WARP_XV assert sees the warp count the XV MMA
  // would run at (a 32-wide-group model floors it to 0 at MATH_WARPS).
  using CT = ComputeTraits<MT, QkComputeMode::FP8, Cfg::BI,
                           Cfg::SPLIT_QK_XV ? Cfg::MATH_WARPS - Cfg::QK_WARPS : Cfg::MATH_WARPS>;
  using L = SmemLayout<MT, QkMode, Cfg::BI, Cfg::MATH_WARPS>;

  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  static constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  // smem layout always allocates HPB heads (zero-padded for invalid slots);
  // Q load and BF16 output / LSE write-back are gated by VALID_HPB to avoid
  // reading/writing past the caller's [num_tokens, NUM_HEADS, ...] buffers.
  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_tile = blockIdx.x % REPLICATE_H;
  const int h_start = h_tile * HPB;
  if (s_i >= num_tokens) return;

  const int topk = cold.topk;
  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : topk;
  topk_len = topk_len < 0 ? 0 : (topk_len > topk ? topk : topk_len);
  // A sliding-window model's candidate list *is* the window, so no token can
  // carry more valid entries than that however wide topk or a caller-supplied
  // topk_length is. Capping here (rather than validating in the FFI) makes an
  // over-large topk_length harmless; the binding requires topk >= WINDOW so
  // the window always fits the row. Mirrors DecodeTileCfg::WINDOW.
  if constexpr (Cfg::HAS_WINDOW) {
    topk_len = topk_len > Cfg::WINDOW ? Cfg::WINDOW : topk_len;
  }
  const int extra_len = prefill_extra_length(cold, s_i);
  const int actual_ni = (topk_len + Cfg::BI - 1) / Cfg::BI + (extra_len + Cfg::BI - 1) / Cfg::BI;

  const int warp_rank = threadIdx.x / 32;

  extern __shared__ char smem_raw[];
  auto sm = SmemPtrs<MT, QkMode, Cfg::BI, Cfg::MATH_WARPS>::init(smem_raw);

  if (threadIdx.x == 0) {
    flashinfer::sparse_mla_sm120::pipeline::BulkReady::init_slots<2>(sm.mbar_kv);
    if constexpr (GatherSchedule::RAW_PIPELINE)
      flashinfer::sparse_mla_sm120::pipeline::BulkReady::init_slots<2>(
          GatherSchedule::ready(smem_raw + L::TOTAL));
  }
  bar_sync_t<Fp8PrefillSync::CTA_INIT,
             Fp8PrefillResources<MT, QkMode, GatherSchedule>::BLOCK_THREADS>();

  if constexpr (GatherSchedule::RAW_PIPELINE) {
    if (threadIdx.x >= Cfg::MATH_THREADS) {
      auto section = [&](int t) {
        return prefill_section<Cfg::BI, PAGE_BLOCK_SIZE>(cold, KV_cache, indices, s_i, topk_len,
                                                         extra_len, t);
      };
      GatherSchedule::template produce<Cfg>(sm, smem_raw + L::TOTAL, section, actual_ni);
    } else {
      GatherSchedule::math_registers();
      sparse_mla_prefill_math_pc<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, GatherSchedule>(
          sm, Q, KV_cache, indices, attn_sink, output, out_lse, cold, s_i, h_start, topk_len);
    }
    return;
  }

  // ── IO warps ────────────────────────────────────────────────────
  if (warp_rank >= Cfg::MATH_WARPS) {
    if constexpr (Cfg::REG_REALLOC) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(32));
    }

    const int io_tid = threadIdx.x - Cfg::MATH_THREADS;
    const int32_t* idx_base = indices + (size_t)s_i * topk;
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    if constexpr (MT == ModelType::DSV4_1 && !GatherSchedule::RAW_PIPELINE) {
      {
        auto issue_dual_tile = [&](int t) {
          const auto sec = prefill_section<Cfg::BI, PAGE_BLOCK_SIZE>(cold, KV_cache, indices, s_i,
                                                                     topk_len, extra_len, t);
          GatherSchedule::template issue<Cfg>(sm, sec, t, io_tid, kv_l2_policy);
        };
        if (actual_ni > 0) issue_dual_tile(0);
        for (int ti = 0; ti < actual_ni; ++ti) {
          if (ti + 1 < actual_ni) issue_dual_tile(ti + 1);
          Fp8PrefillSync::KvFree<Cfg::IO_THREADS, Cfg::MATH_THREADS>::acquire(ti & 1);
        }
        return;
      }
    }

    // This thread's candidate index for tile t, staged a tile ahead of use so
    // the index LDG latency is hidden behind the previous tile's gather.
    // Bound by topk_len, not just actual_ni * BI: the last tile is partial and
    // caller padding past topk_len may be stale, and a garbage index would
    // gather a wild gmem address.
    auto load_idx = [&](int t) -> int {
      return (t < actual_ni && io_tid < Cfg::BI && t * Cfg::BI + io_tid < topk_len)
                 ? __ldg(idx_base + t * Cfg::BI + io_tid)
                 : -1;
    };
    // Scales first (plain stores, no mbar signal), then bulk gather
    // (cp.async.bulk signals mbar_kv on completion). Math warps wake on
    // mbar_kv, so reversing the order would race scales vs math reads.
    // threadfence_block ensures scale stores are visible to all threads before
    // the bulk completion event.
    auto issue_tile = [&](int t, int staged) {
      const int buf = t & 1;
      io_gather_scales<MT, PAGE_BLOCK_SIZE, Cfg::BI, Cfg::IO_THREADS>(
          sm.kv_scale_bufs[buf], staged, KV_cache, io_tid, page_stride_bytes);
      __threadfence_block();
      io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, Cfg::L2_EVICT_FIRST, Cfg::BI, Cfg::IO_THREADS>(
          sm.kv_bufs[buf], staged, KV_cache, sm.mbar_kv + buf, io_tid, page_stride_bytes,
          kv_l2_policy);
    };

    int staged = load_idx(0);
    if (actual_ni > 0) {
      issue_tile(0, staged);
      staged = load_idx(1);
    }
    int pf = load_idx(2);

#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      if (ti + 1 < actual_ni) {
        const int next = load_idx(ti + 3);
        // Warm L2 for tile ti+2; its index LDG was issued an iteration ago.
        io_bulk_prefetch_l2<MT, PAGE_BLOCK_SIZE, Cfg::L2_EVICT_FIRST, Cfg::BI, Cfg::IO_THREADS>(
            pf, KV_cache, io_tid, page_stride_bytes, kv_l2_policy);
        issue_tile(ti + 1, staged);
        staged = pf;
        pf = next;
      }
      Fp8PrefillSync::KvFree<Cfg::IO_THREADS, Cfg::MATH_THREADS>::acquire(ti & 1);
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    if constexpr (Cfg::REG_REALLOC) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg::MATH_MAXNREG));
    }

    if constexpr (Cfg::SPLIT_QK_XV) {
      sparse_mla_prefill_math_pc<MT, QkMode, NUM_HEADS, PAGE_BLOCK_SIZE, GatherSchedule>(
          sm, Q, KV_cache, indices, attn_sink, output, out_lse, cold, s_i, h_start, topk_len);
      return;
    }

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    // The serial path only runs on non-split tiles (SPLIT_QK_XV dispatches to
    // the producer/consumer path above), so every math warp is a QK warp here.
    const bool qk_warp = !Cfg::SPLIT_QK_XV || mwarp < Cfg::QK_WARPS;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)h_start * KV::D_QK;
    const int32_t* idx_base = indices + (size_t)s_i * topk;

    // Candidate slice of this warp. Meaningful only for a QK warp: past
    // QK_WARPS it runs off both the BI-wide KV buffer and the token's index
    // row, so everything derived from it stays inside `if (qk_warp)`.
    const int qk_nb = mwarp * Cfg::ENTRIES_PER_WARP;

    if constexpr (QkMode == QkComputeMode::BF16) {
      load_q_bf16_to_smem<MT, Cfg::MATH_THREADS>(sm.q_nope_bf16, sm.q_rope, q_base, VALID_HPB);
    } else {
      quantize_q_to_smem<MT, Cfg::MATH_THREADS>(sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base,
                                                VALID_HPB);
    }
    QRopeRegs<MT> q_rope_regs = preload_q_rope_regs<MT>(sm.q_rope, lane);

    float acc_o[CT::ACC_TILES][4];
#pragma unroll
    for (int t = 0; t < CT::ACC_TILES; t++)
      acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

    // Only V_HAS_ROPE models accumulate a rope tail in the output; sized to 1
    // otherwise so a D_V == D_NOPE model does not carry four dead registers.
    constexpr bool V_ROPE = KV::V_HAS_ROPE;
    float acc_rope[V_ROPE ? 4 : 1] = {0.f};
    float warp_l[2] = {0.f, 0.f};
    // Running row max for this thread's two heads, kept in registers: every
    // math thread redundantly reduces the QK warps' per-warp maxima each tile,
    // which removes one math-group barrier per tile vs a smem round-trip.
    float m0 = -1e30f, m1 = -1e30f;

    bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();
    if (actual_ni > 0) BulkReady::wait(sm.mbar_kv + 0, 0);

// ── Main loop — QK + softmax + XV ───────────────────────────
#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      uint8_t* kv_smem = sm.kv_bufs[ti & 1];
      const int32_t* ib = idx_base + ti * Cfg::BI;

      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += Cfg::MATH_THREADS)
        sm.w_head_sc_all[i] = 0.f;

      float s[4] = {0.f, 0.f, 0.f, 0.f};
      if (qk_warp) {
        uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

        const int idx_rope =
            mask_idx_past_len(ib[qk_nb + gid], ti * Cfg::BI + qk_nb + gid, topk_len);
        const int mask0 = ib[qk_nb + tid * 2];
        const int mask1 = ib[qk_nb + tid * 2 + 1];

        // Only this lane's own entry is needed for the QK rope prefetch — the
        // XV rope MMA re-derives its addresses from `ib` inside xv_rope_mma.
        const uint8_t* entry_base_gid =
            prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(KV_cache, idx_rope, page_stride_bytes);

        KVRopePrefetch<MT> rope_pf = prefetch_kv_rope<MT>(
            reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

        // ── QK nope MMA ─────────────────────
        float qk_storage[1][4] = {};
        auto& qk = qk_storage[0];
        const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
        if constexpr (QkMode == QkComputeMode::BF16) {
          const uint8_t* k_sc =
              KV::SCALE_IN_KV_SMEM
                  ? kv_gid_base + KV::D_NOPE
                  : sm.kv_scale_bufs[ti & 1] + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
          qk_bf16_from_fp8_nope_16x8<KV>(qk_storage, sm.q_nope_bf16, 0, kv_gid_base, k_sc, lane);
        } else {
#pragma unroll
          for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
            uint8_t sfa =
                fp32_exponent_byte(sm.q_nope_sc[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
            float acc0, acc1, acc2, acc3;
            init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
            const uint8_t* k_scale_base;
            if constexpr (KV::SCALE_IN_KV_SMEM) {
              k_scale_base = kv_gid_base + KV::D_NOPE;
            } else {
              k_scale_base = sm.kv_scale_bufs[ti & 1] + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
            }
            uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
            qk_fp8_scale_group_16x8<KV>(acc0, acc1, acc2, acc3, sm.q_nope_fp8, kv_warp_base, blk,
                                        sfa, sfb, lane);
            const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
            const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
            commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE,
                              e1_base + KV::D_NOPE, blk);
          }
        }  // closes the FP8/BF16 QK branch

        // ── QK rope (BF16 MMA, uses prefetched B operands) ──────
        compute_qk_rope<MT>(qk, q_rope_regs, rope_pf);

        // ── Invalid index masking + topk_length overflow ─────
        {
          int e0 = qk_nb + tid * 2, e1 = e0 + 1;
          if (mask0 < 0) {
            qk[0] = -1e30f;
            qk[2] = -1e30f;
          }
          if (mask1 < 0) {
            qk[1] = -1e30f;
            qk[3] = -1e30f;
          }
          // HAS_WINDOW makes topk_len a real bound even when the caller passes no
          // topk_length, so the mask has to run. For the unbounded models this
          // folds back to the original pointer test (and with topk_length null,
          // topk_len == topk makes the comparisons dead anyway).
          if (Cfg::HAS_WINDOW || cold.topk_length != nullptr) {
            int a0 = ti * Cfg::BI + e0, a1 = ti * Cfg::BI + e1;
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

        // ── Online softmax, warp-local half ──────────────────
        s[0] = qk[0] * sm_scale_log2e;
        s[1] = qk[1] * sm_scale_log2e;
        s[2] = qk[2] * sm_scale_log2e;
        s[3] = qk[3] * sm_scale_log2e;

        float lm0, lm1;
        softmax_warp_max(s, lm0, lm1);
        if (tid == 0) {
          sm.reduce_buf[mwarp * HPB + gid] = lm0;
          sm.reduce_buf[mwarp * HPB + gid + 8] = lm1;
        }
      }  // qk_warp
      bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

      // Redundant per-thread reduction over the QK warps' row maxima. Every
      // math thread holds heads gid and gid+8, so each reads QK_WARPS pairs
      // and updates its own running max; identical ops on identical inputs
      // keep this bitwise-equal to the old single-warp reduction.
      float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
      for (int w = 0; w < Cfg::QK_WARPS; w++) {
        tm0 = fmaxf(tm0, sm.reduce_buf[w * HPB + gid]);
        tm1 = fmaxf(tm1, sm.reduce_buf[w * HPB + gid + 8]);
      }
      float nm0 = fmaxf(m0, tm0), nm1 = fmaxf(m1, tm1);
      float alpha0 = exp2f(m0 - nm0), alpha1 = exp2f(m1 - nm1);
      m0 = nm0;
      m1 = nm1;

      if (alpha0 < 1.0f || alpha1 < 1.0f) {
#pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++) {
          acc_o[t][0] *= alpha0;
          acc_o[t][1] *= alpha0;
          acc_o[t][2] *= alpha1;
          acc_o[t][3] *= alpha1;
        }
        if constexpr (V_ROPE) {
          acc_rope[0] *= alpha0;
          acc_rope[1] *= alpha0;
          acc_rope[2] *= alpha1;
          acc_rope[3] *= alpha1;
        }
        warp_l[0] *= alpha0;
        warp_l[1] *= alpha1;
      }

      // The softmax weights and the per-entry V scales belong to the warp's own
      // candidates, so they stay on the QK warps. Both survive to the W
      // quantization below, which is why they are declared at loop scope.
      float w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
      float vsc_cache[CT::N_V_CHUNKS][2];
      if (qk_warp) {
        w0 = exp2f(s[0] - nm0);
        w1 = exp2f(s[1] - nm0);
        w2 = exp2f(s[2] - nm1);
        w3 = exp2f(s[3] - nm1);

        float ls0, ls1;
        softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
        warp_l[0] += ls0;
        warp_l[1] += ls1;

        // ── V scale cache + atomicMax ───────────────────────────
        const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
        const uint8_t* e0_base = kv_smem + (size_t)e0i * KV::KV_SMEM_STRIDE;
        const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
#pragma unroll
        for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
            vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
          } else {
            vsc_cache[vc][0] = fp32_from_exponent_byte(
                sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
            vsc_cache[vc][1] = fp32_from_exponent_byte(
                sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          }
          float ws00 = w0 * vsc_cache[vc][0], ws01 = w1 * vsc_cache[vc][1];
          float ws10 = w2 * vsc_cache[vc][0], ws11 = w3 * vsc_cache[vc][1];
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid]),
                    __float_as_int(fmaxf(ws00, ws01)));
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(ws10, ws11)));
        }
      }
      bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += Cfg::MATH_THREADS)
        sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
      bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

      // ── XV nope MMA (D2: direct B from kv_smem) ───────────
      {
        const int e0i = qk_nb + tid * 2, e1i = e0i + 1;

        if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
          // This variant interleaves W quantization with the MMA across two
          // passes, so the register-dependent and smem-only halves cannot be
          // separated — every math warp must own candidates.
          static_assert(!Cfg::SPLIT_QK_XV,
                        "the two-pass ARBITRARY_FP32 W quantization cannot run with a QK/XV warp "
                        "split; give this model type MATH_WARPS == QK_WARPS");
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            float* vc_sc = sm.w_head_sc_all + vc * HPB;
            uint8_t* wfp8 = sm.w_fp8 + vc * L::SMEM_W_FP8_ONE;
            float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
            float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
            float wn00 = w0 * vsc0 * si0, wn01 = w1 * vsc1 * si0;
            float wn10 = w2 * vsc0 * si1, wn11 = w3 * vsc1 * si1;
            float xv_acc[CT::NT_PER_WARP_XV][4] = {0};
#pragma unroll
            for (int wpass = 0; wpass < 2; ++wpass) {
              if (wpass > 0) bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();
              Fp8WeightQuad wq =
                  quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
              wfp8[gid * CT::W_FP8_STRIDE + e0i] = wq.h0_e0;
              wfp8[gid * CT::W_FP8_STRIDE + e1i] = wq.h0_e1;
              wfp8[(gid + 8) * CT::W_FP8_STRIDE + e0i] = wq.h1_e0;
              wfp8[(gid + 8) * CT::W_FP8_STRIDE + e1i] = wq.h1_e1;
              bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, CT::W_FP8_STRIDE, CT::XV_KSTEPS>(
                    xv_acc[nt], wfp8, kv_smem, dim, lane);
              }
            }
#pragma unroll
            for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
              int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
              float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
              acc_o[ti_acc][0] += xv_acc[nt][0] * sc0;
              acc_o[ti_acc][1] += xv_acc[nt][1] * sc0;
              acc_o[ti_acc][2] += xv_acc[nt][2] * sc1;
              acc_o[ti_acc][3] += xv_acc[nt][3] * sc1;
            }
          }
        } else {
          // Batch W quant for all chunks. Reads this warp's own softmax weights
          // and V scales, so it is QK-warp work; the MMA below is smem-only and
          // runs on every math warp. The barrier between them is what makes the
          // QK/XV warp split possible.
          if (qk_warp) {
#pragma unroll
            for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
              float* vc_sc = sm.w_head_sc_all + vc * HPB;
              uint8_t* wfp8 = sm.w_fp8 + vc * L::SMEM_W_FP8_ONE;
              float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
              float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
              float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
              float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
              __nv_fp8_e4m3 f00(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws00 * si0)));
              __nv_fp8_e4m3 f01(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws01 * si0)));
              __nv_fp8_e4m3 f10(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws10 * si1)));
              __nv_fp8_e4m3 f11(fmaxf(-FP8_MAX, fminf(FP8_MAX, ws11 * si1)));
              wfp8[gid * CT::W_FP8_STRIDE + e0i] = f00.__x;
              wfp8[gid * CT::W_FP8_STRIDE + e1i] = f01.__x;
              wfp8[(gid + 8) * CT::W_FP8_STRIDE + e0i] = f10.__x;
              wfp8[(gid + 8) * CT::W_FP8_STRIDE + e1i] = f11.__x;
            }
          }
          bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

// Batch MMA for all chunks
#pragma unroll
          for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
            float* vc_sc = sm.w_head_sc_all + vc * HPB;
            uint8_t* wfp8 = sm.w_fp8 + vc * L::SMEM_W_FP8_ONE;
#pragma unroll
            for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
              int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
              int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
              float xv[4] = {0.f, 0.f, 0.f, 0.f};
              pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, CT::W_FP8_STRIDE, CT::XV_KSTEPS>(xv, wfp8, kv_smem,
                                                                                  dim, lane);
              float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
              acc_o[ti_acc][0] += xv[0] * sc0;
              acc_o[ti_acc][1] += xv[1] * sc0;
              acc_o[ti_acc][2] += xv[2] * sc1;
              acc_o[ti_acc][3] += xv[3] * sc1;
            }
          }
        }
      }

      if constexpr (!V_ROPE) bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

      // ── XV rope BF16 MMA (DSV4 only) ─────────────────────
      if constexpr (V_ROPE) {
        bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();
        // xv_rope_mma still reads the namespace-scope BI/ENTRIES_PER_WARP, so
        // it is only correct at the default tile. No V_HAS_ROPE model runs a
        // custom tile today; the assert pins that.
        static_assert(Cfg::BI == BI && Cfg::MATH_WARPS == N_MATH_WARPS,
                      "xv_rope_mma is hardcoded to the 64/8 tile; parameterize it before running a "
                      "V_HAS_ROPE model at a different BI");
        xv_rope_mma<MT, PAGE_BLOCK_SIZE>(
            acc_rope, w0, w1, w2, w3, ib, min(Cfg::BI, topk_len - ti * Cfg::BI), KV_cache, mwarp,
            lane, page_stride_bytes, reinterpret_cast<bf16*>(sm.w_fp8));
      }

      Fp8PrefillSync::KvFree<Cfg::IO_THREADS, Cfg::MATH_THREADS>::release(ti & 1);
      if (ti + 1 < actual_ni) {
        const int next_phase = ((ti + 1) >> 1) & 1;
        BulkReady::wait(sm.mbar_kv + ((ti + 1) & 1), next_phase);
      }
    }

    // Publish the register-resident row max for the sink-aware normalizer in
    // the epilogue. Threads sharing a gid computed identical values.
    if (tid == 0) {
      sm.m_smem[gid] = m0;
      sm.m_smem[gid + 8] = m1;
    }

    // ── Finalize deferred row_sum ────────────────────────────────
    // Only the QK warps carry a partial row sum; warps past QK_WARPS never
    // touched warp_l and must not write a slot the reduction below reads.
    if (qk_warp && tid == 0) {
      sm.reduce_buf[mwarp * HPB + gid] = warp_l[0];
      sm.reduce_buf[mwarp * HPB + gid + 8] = warp_l[1];
    }
    bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();
    if (threadIdx.x < HPB) {
      int h = threadIdx.x;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < Cfg::QK_WARPS; w++) ts += sm.reduce_buf[w * HPB + h];
      sm.l_smem[h] = ts;
    }
    bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

    // ── Write BF16 output and LSE ────────────────────────────────
    // attn_sink convention (FlashMLA V4): output[h] *= sigmoid(lse_h - sink_h)
    // is folded directly into the normalizer:
    //   il = exp(lse) / (exp(lse) + exp(sink)) / exp(lse)
    //      = 1 / (l + exp(sink - m))   in log2 space
    // (working in log2 space: sum_l is in exp-domain of m, multiply sink by LOG2E).
    // Padded heads carry sink=-inf → exp2(-inf)=0 → no-op (collapses to 1/l).
    float il0, il1;
    if (cold.attn_sink != nullptr) {
      float s0 = __ldg(cold.attn_sink + h_start + gid) * LOG2E;
      float denom0 = sm.l_smem[gid] + exp2f(s0 - sm.m_smem[gid]);
      il0 = (denom0 > 0.f) ? (1.f / denom0) : 0.f;
      if constexpr (VALID_HPB > 8) {
        float s1 = __ldg(cold.attn_sink + h_start + gid + 8) * LOG2E;
        float denom1 = sm.l_smem[gid + 8] + exp2f(s1 - sm.m_smem[gid + 8]);
        il1 = (denom1 > 0.f) ? (1.f / denom1) : 0.f;
      } else {
        il1 = 0.f;
      }
    } else {
      il0 = (sm.l_smem[gid] > 0.f) ? (1.f / sm.l_smem[gid]) : 0.f;
      il1 = (sm.l_smem[gid + 8] > 0.f) ? (1.f / sm.l_smem[gid + 8]) : 0.f;
    }

    // The main loop is done with the KV double buffer, so its first half is
    // reused to stage the BF16 output. At the default tile this is slack;
    // DOTS3_SWA needs 16 * 1024 * 2 = 32768 B against a 32 * 1040 = 33280 B
    // buffer, so the fit is checked rather than assumed.
    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_bufs[0]);
    constexpr int BF16_STAGING_STRIDE = KV::D_V;
    static_assert(HPB * BF16_STAGING_STRIDE * sizeof(bf16) <= L::SMEM_KV_BUF,
                  "BF16 output staging overruns the KV buffer it reuses");

#pragma unroll
    for (int t = 0; t < CT::ACC_TILES; t++) {
      constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
      int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
      int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
      staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[t][0] * il0);
      staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[t][1] * il0);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[t][2] * il1);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[t][3] * il1);
    }

    if constexpr (V_ROPE) {
      static_assert(KV::D_V == KV::D_NOPE + KV::D_ROPE,
                    "the rope tail is written at D_NOPE within a D_V-wide staging row");
      int n_start = mwarp * 8;
      int d0 = KV::D_NOPE + n_start + tid * 2;
      staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[0] * il0);
      staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_rope[1] * il0);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[2] * il1);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_rope[3] * il1);
    } else {
      static_assert(KV::D_V == KV::D_NOPE, "a rope-free V must span exactly the nope dimensions");
    }
    bar_sync_t<Fp8PrefillSync::MATH, Cfg::MATH_THREADS>();

    // Coalesced BF16 write: uint4 = 128-bit = 8 bf16 per store
    {
      constexpr size_t h_stride = KV::D_V;
      constexpr size_t token_stride = (size_t)NUM_HEADS * KV::D_V;
      const size_t out_base = (size_t)s_i * token_stride + (size_t)h_start * h_stride;
      constexpr int BF16_PER_STORE = 8;
      constexpr int STORES_PER_HEAD = KV::D_V / BF16_PER_STORE;  // 64 at D_V=512
      for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += Cfg::MATH_THREADS) {
        int h = idx / STORES_PER_HEAD;
        int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
        uint4 v = *reinterpret_cast<const uint4*>(&staging_bf16[h * BF16_STAGING_STRIDE + d8]);
        *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
      }
    }

    // Write LSE (merged with attn_sink if present)
    if (threadIdx.x < VALID_HPB) {
      int h = threadIdx.x;
      float lse = softmax_lse(sm.m_smem[h], sm.l_smem[h]);
      if (cold.attn_sink != nullptr) {
        float sink_log2 = __ldg(cold.attn_sink + h_start + h) * LOG2E;
        if (lse != -1e30f)
          lse += log2f(1.f + exp2f(sink_log2 - lse));
        else
          lse = sink_log2;
      }
      size_t lse_idx = (size_t)s_i * cold.out_lse_stride_elems + h_start + h;
      out_lse[lse_idx] = lse;
    }
  }
}
