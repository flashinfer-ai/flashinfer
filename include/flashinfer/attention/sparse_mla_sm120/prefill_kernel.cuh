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

#include "arch/barrier.cuh"
#include "arch/common.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "arch/stmatrix_sm120.cuh"
#include "common/d2_load_b.cuh"
#include "common/fp8_quant.cuh"
#include "common/kv_cache_io.cuh"
#include "common/online_softmax.cuh"
#include "common/q_rope.cuh"
#include "common/scale_mma.cuh"
#include "common/smem_layout.cuh"
#include "common/xv_rope_mma.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

// ============================================================================
// Sparse MLA Prefill Kernel — single-pass (no split-KV, no combine)
//
// Structurally identical to decode main loop (QK→softmax→XV), but:
//   - Iterates over ALL NI = TOPK/BI tiles (no split)
//   - Writes direct BF16 output (no partial_O + combine)
//   - No PDL (no dependent kernel)
//
// Template params (all constexpr):
//   MT:              ModelType (DSV3_2 / DSV4)
//   CM:              ComputeMode (FP8 / BF16) for the QK MMA; XV is always FP8
//   NUM_HEADS:       8, 16, 32, 64, 128 (NUM_HEADS < HPB=16 zero-pads + gates)
//   TOPK:            128, 192, 256, 512, 1024, 2048
//   PAGE_BLOCK_SIZE: 64 (DSV3_2 and DSV4 both use the 64-token page layout)
// ============================================================================

struct PrefillColdParams {
  float sm_scale;
  int num_tokens;
  size_t stride_kv_block;
  // Dual-cache only (sparse_mla_prefill_mg_dual_kernel); ignored elsewhere.
  size_t stride_kv_block_extra;
  int topk_extra;          // dual-cache only. Runtime topk_extra so callers can
                           // pass any cdiv(max_model_len, compress_ratio) value
                           // without per-bound template instantiations.
  const float* attn_sink;  // [NUM_HEADS] float32, natural log domain. nullptr = disabled.
  const int* topk_length;  // [num_tokens] int32, nullptr = uniform TOPK.
  const int*
      topk_length_extra;  // [num_tokens] int32, dual-cache only. nullptr = uniform topk_extra.
};

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    sparse_mla_prefill_kernel(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                              const int32_t* __restrict__ indices,
                              const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                              bf16* __restrict__ output, float* __restrict__ out_lse,
                              __grid_constant__ const PrefillColdParams cold) {
  const float sm_scale = cold.sm_scale;
  const int num_tokens = cold.num_tokens;
  constexpr int page_block_size = PAGE_BLOCK_SIZE;
  const size_t stride_kv_block = cold.stride_kv_block;
  using KV = KVCacheTraits<MT>;
  // CT pinned to FP8: XV always uses FP8 W; CM only flips the QK side.
  using CT = ComputeTraits<MT, ComputeMode::FP8>;
  using L = SmemLayout<MT, CM>;
  using IO = KVIOTraits<MT>;

  static constexpr int NI = TOPK / BI;
  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  static constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;
  // smem layout always allocates HPB heads (zero-padded for invalid slots);
  // Q load and BF16 output / LSE write-back are gated by VALID_HPB to avoid
  // reading/writing past the caller's [num_tokens, NUM_HEADS, ...] buffers.
  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_tile = blockIdx.x % REPLICATE_H;
  const int h_start = h_tile * HPB;
  if (s_i >= num_tokens) return;

  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : TOPK;
  topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
  const int actual_ni = (topk_len + BI - 1) / BI;

  const int warp_rank = threadIdx.x / 32;
  const int wy = warp_rank / 4;

  extern __shared__ char smem_raw[];
  auto sm = SmemPtrs<MT, CM>::init(smem_raw);

  if (threadIdx.x == 0) {
    mbarrier_init(sm.mbar_kv + 0, 1);
    mbarrier_init(sm.mbar_kv + 1, 1);
  }
  bar_sync_t<3, BLOCK_THREADS>();

  // ── IO warps ────────────────────────────────────────────────────
  if (wy == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(32));

    const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
    const int32_t* idx_base = indices + (size_t)s_i * TOPK;
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    // Prologue: gather tile 0. Scales first (plain stores, no mbar signal),
    // then bulk gather (cp.async.bulk signals mbar_kv on completion). Math
    // warps wake on mbar_kv, so reversing the order would race scales vs
    // math reads. threadfence_block ensures scale stores are visible to all
    // threads before the bulk completion event.
    if (actual_ni > 0) {
      io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_bufs[0], idx_base, KV_cache, io_tid,
                                            stride_kv_block);
      __threadfence_block();
      io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
          sm.kv_bufs[0], idx_base, KV_cache, sm.mbar_kv + 0, io_tid, stride_kv_block, kv_l2_policy);
    }

#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      if (ti + 1 < actual_ni) {
        io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_bufs[(ti + 1) & 1],
                                              idx_base + (ti + 1) * BI, KV_cache, io_tid,
                                              stride_kv_block);
        __threadfence_block();
        io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
            sm.kv_bufs[(ti + 1) & 1], idx_base + (ti + 1) * BI, KV_cache,
            sm.mbar_kv + ((ti + 1) & 1), io_tid, stride_kv_block, kv_l2_policy);
      }
      bar_sync_t<1, BLOCK_THREADS>();
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    const bf16* q_base = Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)h_start * KV::D_QK;
    const int32_t* idx_base = indices + (size_t)s_i * TOPK;

    if constexpr (CM == ComputeMode::BF16) {
      load_q_bf16_to_smem<MT, MATH_THREADS>(sm.q_nope_bf16, sm.q_rope, q_base, VALID_HPB);
    } else {
      quantize_q_to_smem<MT, MATH_THREADS>(sm.q_nope_fp8, sm.q_nope_sc, sm.q_rope, q_base,
                                           sm.reduce_buf, VALID_HPB);
    }
    QRopeRegs q_rope_regs = preload_q_rope_regs(sm.q_rope, lane);

    for (int h = threadIdx.x; h < HPB; h += MATH_THREADS) sm.m_smem[h] = -1e30f;

    float acc_o[CT::ACC_TILES][4];
#pragma unroll
    for (int t = 0; t < CT::ACC_TILES; t++)
      acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;

    float acc_rope[4] = {0.f, 0.f, 0.f, 0.f};
    float warp_l[2] = {0.f, 0.f};

    bar_sync_t<2, MATH_THREADS>();
    if (actual_ni > 0) mbarrier_wait_parity(sm.mbar_kv + 0, 0);

// ── Main loop — QK + softmax + XV ───────────────────────────
#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      uint8_t* kv_smem = sm.kv_bufs[ti & 1];
      const int32_t* ib = idx_base + ti * BI;
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

      const uint8_t* entry_base[ENTRIES_PER_WARP];
      if constexpr (KV::V_HAS_ROPE) {
#pragma unroll
        for (int e = 0; e < ENTRIES_PER_WARP; e++) {
          int idx = ib[qk_nb + e];
          idx = (idx >= 0) ? idx : 0;
          int bi_e = idx / page_block_size;
          int li_e = idx % page_block_size;
          entry_base[e] = KV_cache + (size_t)bi_e * stride_kv_block + (size_t)li_e * IO::IO_STRIDE;
        }
      } else {
        int idx = ib[qk_nb + gid];
        idx = (idx >= 0) ? idx : 0;
        entry_base[gid] = KV_cache + (size_t)idx * IO::IO_STRIDE;
      }

      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all[i] = 0.f;

      KVRopePrefetch rope_pf = prefetch_kv_rope(
          reinterpret_cast<const bf16*>(entry_base[gid] + KV::KV_ROPE_GMEM_OFFSET), lane);

      // ── QK nope MMA ─────────────────────
      float qk[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
      if constexpr (CM == ComputeMode::BF16) {
        // BF16 m16n8k16 with per-thread FP8→BF16 dequant on KV.
#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
          float scale_f;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
          } else {
            scale_f = ue8m0_to_fp32(
                sm.kv_scale_bufs[ti & 1][(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
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
          uint8_t sfa = fp32_to_ue8m0(sm.q_nope_sc[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
          float acc0, acc1, acc2, acc3;
          init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
          const uint8_t* k_scale_base;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            k_scale_base = kv_gid_base + KV::D_NOPE;
          } else {
            k_scale_base = sm.kv_scale_bufs[ti & 1] + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
          }
          uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);

#pragma unroll
          for (int ks = 0; ks < QK_NOPE_KSTEPS; ks++) {
            int ko = blk * KV::QUANT_TILE + ks * 32;
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_nope_fp8 + ko, KV::Q_NOPE_STRIDE, lane);
            ldmatrix_load_B_fp8(b0, b1, kv_warp_base + ko, KV::KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1, acc0, acc1, acc2,
                                                           acc3, sfa, sfb);
            acc0 = r.d0;
            acc1 = r.d1;
            acc2 = r.d2;
            acc3 = r.d3;
          }
          const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
          const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
          commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE, e1_base + KV::D_NOPE,
                            blk);
        }
      }

      // ── QK rope (BF16 MMA, uses prefetched B operands) ──────
      compute_qk_rope(qk, q_rope_regs, rope_pf);

      // ── Invalid index masking + topk_length overflow ─────
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

      // ── Online softmax (deferred sum, conditional rescale) ──
      float s[4] = {qk[0] * sm_scale_log2e, qk[1] * sm_scale_log2e, qk[2] * sm_scale_log2e,
                    qk[3] * sm_scale_log2e};

      float lm0, lm1;
      softmax_warp_max(s, lm0, lm1);
      if (tid == 0) {
        sm.reduce_buf[mwarp * HPB + gid] = lm0;
        sm.reduce_buf[mwarp * HPB + gid + 8] = lm1;
      }
      bar_sync_t<2, MATH_THREADS>();

      if (threadIdx.x < HPB) {
        int h = threadIdx.x;
        float old_m = sm.m_smem[h], tm = -1e30f;
#pragma unroll
        for (int w = 0; w < N_MATH_WARPS; w++) tm = fmaxf(tm, sm.reduce_buf[w * HPB + h]);
        float nm = fmaxf(old_m, tm);
        float alpha = exp2f(old_m - nm);
        sm.m_smem[h] = nm;
        sm.reduce_buf[h] = alpha;
        sm.reduce_buf[HPB + h] = nm;
      }
      bar_sync_t<2, MATH_THREADS>();

      float alpha0 = sm.reduce_buf[gid], alpha1 = sm.reduce_buf[gid + 8];
      float nm0 = sm.reduce_buf[HPB + gid], nm1 = sm.reduce_buf[HPB + gid + 8];

      if (alpha0 < 1.0f || alpha1 < 1.0f) {
#pragma unroll
        for (int t = 0; t < CT::ACC_TILES; t++) {
          acc_o[t][0] *= alpha0;
          acc_o[t][1] *= alpha0;
          acc_o[t][2] *= alpha1;
          acc_o[t][3] *= alpha1;
        }
        if constexpr (KV::V_HAS_ROPE) {
          acc_rope[0] *= alpha0;
          acc_rope[1] *= alpha0;
          acc_rope[2] *= alpha1;
          acc_rope[3] *= alpha1;
        }
        warp_l[0] *= alpha0;
        warp_l[1] *= alpha1;
      }

      float w0 = exp2f(s[0] - nm0), w1 = exp2f(s[1] - nm0);
      float w2 = exp2f(s[2] - nm1), w3 = exp2f(s[3] - nm1);

      float ls0, ls1;
      softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
      warp_l[0] += ls0;
      warp_l[1] += ls1;

      // ── V scale cache + atomicMax ───────────────────────────
      float vsc_cache[CT::N_V_CHUNKS][2];
      {
        const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
        const uint8_t* e0_base = kv_warp_base + tid * 2 * KV::KV_SMEM_STRIDE;
        const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
#pragma unroll
        for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
            vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
          } else {
            vsc_cache[vc][0] =
                ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
            vsc_cache[vc][1] =
                ue8m0_to_fp32(sm.kv_scale_bufs[ti & 1][e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          }
          float ws00 = w0 * vsc_cache[vc][0], ws01 = w1 * vsc_cache[vc][1];
          float ws10 = w2 * vsc_cache[vc][0], ws11 = w3 * vsc_cache[vc][1];
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid]),
                    __float_as_int(fmaxf(ws00, ws01)));
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc_all[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(ws10, ws11)));
        }
      }
      bar_sync_t<2, MATH_THREADS>();

      for (int i = threadIdx.x; i < CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all[i] = fmaxf(sm.w_head_sc_all[i], 1e-10f) / FP8_MAX;
      bar_sync_t<2, MATH_THREADS>();

      // ── XV nope MMA (D2: direct B from kv_smem) ───────────
      {
        const int e0i = qk_nb + tid * 2, e1i = e0i + 1;

        if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
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
              if (wpass > 0) bar_sync_t<2, MATH_THREADS>();
              Fp8WeightQuad wq =
                  quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
              wfp8[gid * (BI + 16) + e0i] = wq.h0_e0;
              wfp8[gid * (BI + 16) + e1i] = wq.h0_e1;
              wfp8[(gid + 8) * (BI + 16) + e0i] = wq.h1_e0;
              wfp8[(gid + 8) * (BI + 16) + e1i] = wq.h1_e1;
              bar_sync_t<2, MATH_THREADS>();

#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
#pragma unroll
                for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                  int ko = kstep * 32;
                  uint32_t a0, a1, a2, a3, b0, b1;
                  ldmatrix_load_A_fp8(a0, a1, a2, a3, wfp8 + ko, BI + 16, lane);
                  d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                  MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv_acc[nt][0],
                                                    xv_acc[nt][1], xv_acc[nt][2], xv_acc[nt][3]);
                  xv_acc[nt][0] = r.d0;
                  xv_acc[nt][1] = r.d1;
                  xv_acc[nt][2] = r.d2;
                  xv_acc[nt][3] = r.d3;
                }
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
// Batch W quant for all chunks
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
            wfp8[gid * (BI + 16) + e0i] = f00.__x;
            wfp8[gid * (BI + 16) + e1i] = f01.__x;
            wfp8[(gid + 8) * (BI + 16) + e0i] = f10.__x;
            wfp8[(gid + 8) * (BI + 16) + e1i] = f11.__x;
          }
          bar_sync_t<2, MATH_THREADS>();

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
#pragma unroll
              for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                int ko = kstep * 32;
                uint32_t a0, a1, a2, a3, b0, b1;
                ldmatrix_load_A_fp8(a0, a1, a2, a3, wfp8 + ko, BI + 16, lane);
                d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                MmaFp8Result r =
                    mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
                xv[0] = r.d0;
                xv[1] = r.d1;
                xv[2] = r.d2;
                xv[3] = r.d3;
              }
              float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
              acc_o[ti_acc][0] += xv[0] * sc0;
              acc_o[ti_acc][1] += xv[1] * sc0;
              acc_o[ti_acc][2] += xv[2] * sc1;
              acc_o[ti_acc][3] += xv[3] * sc1;
            }
          }
        }
      }

      if constexpr (!KV::V_HAS_ROPE) bar_sync_t<2, MATH_THREADS>();

      // ── XV rope BF16 MMA (DSV4 only) ─────────────────────
      if constexpr (KV::V_HAS_ROPE) {
        bar_sync_t<2, MATH_THREADS>();
        xv_rope_mma<MT, PAGE_BLOCK_SIZE>(acc_rope, w0, w1, w2, w3, ib, KV_cache, mwarp, lane,
                                         stride_kv_block, reinterpret_cast<bf16*>(sm.w_fp8));
      }

      bar_arrive_t<1, BLOCK_THREADS>();
      if (ti + 1 < actual_ni) {
        const int next_phase = ((ti + 1) >> 1) & 1;
        mbarrier_wait_parity(sm.mbar_kv + ((ti + 1) & 1), next_phase);
      }
    }

    // ── Finalize deferred row_sum ────────────────────────────────
    if (tid == 0) {
      sm.reduce_buf[mwarp * HPB + gid] = warp_l[0];
      sm.reduce_buf[mwarp * HPB + gid + 8] = warp_l[1];
    }
    bar_sync_t<2, MATH_THREADS>();
    if (threadIdx.x < HPB) {
      int h = threadIdx.x;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < N_MATH_WARPS; w++) ts += sm.reduce_buf[w * HPB + h];
      sm.l_smem[h] = ts;
    }
    bar_sync_t<2, MATH_THREADS>();

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

    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_bufs[0]);
    constexpr int BF16_STAGING_STRIDE = D_V;

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

    if constexpr (KV::V_HAS_ROPE) {
      int n_start = mwarp * 8;
      int d0 = KV::D_NOPE + n_start + tid * 2;
      staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[0] * il0);
      staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_rope[1] * il0);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_rope[2] * il1);
      staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_rope[3] * il1);
    }
    bar_sync_t<2, MATH_THREADS>();

    // Coalesced BF16 write: uint4 = 128-bit = 8 bf16 per store
    {
      constexpr size_t h_stride = D_V;
      constexpr size_t token_stride = (size_t)NUM_HEADS * D_V;
      const size_t out_base = (size_t)s_i * token_stride + (size_t)h_start * h_stride;
      constexpr int BF16_PER_STORE = 8;
      constexpr int STORES_PER_HEAD = D_V / BF16_PER_STORE;  // 64
      for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
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
      size_t lse_idx = (size_t)s_i * NUM_HEADS + h_start + h;
      out_lse[lse_idx] = lse;
    }
  }
}

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
static constexpr int MG_N_HG_DEFAULT = 2;
static constexpr int MG_HEADS_PER_CTA_DEFAULT = MG_N_HG_DEFAULT * HPB;  // 32

template <ModelType MT, int PAGE_BLOCK_SIZE>
__device__ __forceinline__ const uint8_t* prefill_kv_entry_base(
    const uint8_t* __restrict__ kv_global, int idx, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  idx = (idx >= 0) ? idx : 0;
  if constexpr (KV::V_HAS_ROPE) {
    const int bi = idx / PAGE_BLOCK_SIZE;
    const int li = idx % PAGE_BLOCK_SIZE;
    return kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
  } else {
    return kv_global + (size_t)idx * IO::IO_STRIDE;
  }
}

// Shared MG implementation for single-cache and dual-cache prefill.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          bool DUAL_CACHE, int PAGE_BLOCK_SIZE_EXTRA, int MG_N_HG_T, bool ASSUME_FULL_TILES = false>
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
  const size_t stride_kv_block = cold.stride_kv_block;
  [[maybe_unused]] const size_t stride_kv_block_extra =
      DUAL_CACHE ? cold.stride_kv_block_extra : (size_t)0;
  using KV = KVCacheTraits<MT>;
  // CT pinned to FP8: XV always uses FP8 W; CM only flips the QK side.
  using CT = ComputeTraits<MT, ComputeMode::FP8>;
  using LMG = SmemLayoutMG<MT, CM>;
  using SMG = SmemPtrsMG<MT, CM>;

  static constexpr int NI = TOPK / BI;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA == 0 || (MG_N_HG_T == 1 && NUM_HEADS < HPB),
                "NUM_HEADS must fill MG_HEADS_PER_CTA, except a single padded head group");
  static constexpr int REPLICATE_H = (NUM_HEADS + MG_HEADS_PER_CTA - 1) / MG_HEADS_PER_CTA;
  static constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
  static constexpr int QK_NOPE_KSTEPS = KV::QUANT_TILE / 32;
  static constexpr bool USE_WFP8_ROW_XOR = DUAL_CACHE && (PAGE_BLOCK_SIZE_EXTRA == 2);

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_tile = blockIdx.x % REPLICATE_H;
  const int h_start = h_tile * MG_HEADS_PER_CTA;
  if (s_i >= num_tokens) return;

  [[maybe_unused]] int topk_len =
      ASSUME_FULL_TILES ? TOPK : (cold.topk_length ? __ldg(cold.topk_length + s_i) : TOPK);
  if constexpr (!ASSUME_FULL_TILES) {
    topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
  }
  const int actual_ni = ASSUME_FULL_TILES ? NI : ((topk_len + BI - 1) / BI);
  const int main_ni = ASSUME_FULL_TILES ? NI : actual_ni;

  // Dual-cache runtime lengths.
  [[maybe_unused]] int topk_len_extra = 0;
  [[maybe_unused]] int topk_extra_declared = 0;
  int ni_total = main_ni;
  if constexpr (DUAL_CACHE) {
    topk_extra_declared = cold.topk_extra;
    if constexpr (ASSUME_FULL_TILES) {
      topk_len_extra = topk_extra_declared;
    } else {
      topk_len_extra =
          cold.topk_length_extra ? __ldg(cold.topk_length_extra + s_i) : topk_extra_declared;
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

  if (threadIdx.x == 0) {
    mbarrier_init(sm.mbar_kv(0), 1);
    mbarrier_init(sm.mbar_kv(1), 1);
  }
  bar_sync_t<3, BLOCK_THREADS>();

  // ── IO warps ────────────────────────────────────────────────────
  if (wy == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(32));

    const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
    const int32_t* idx_base = indices + (size_t)s_i * TOPK;
    [[maybe_unused]] const int32_t* idx_base_extra = nullptr;
    if constexpr (DUAL_CACHE) {
      idx_base_extra = indices_extra + (size_t)s_i * topk_extra_declared;
    }
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    if constexpr (ASSUME_FULL_TILES) {
      io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf(0), idx_base, KV_cache, io_tid,
                                            stride_kv_block);
      __threadfence_block();
      io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(
          sm.kv_buf(0), idx_base, KV_cache, sm.mbar_kv(0), io_tid, stride_kv_block, kv_l2_policy);
#pragma unroll 1
      for (int ti = 0; ti < loop_bound; ti++) {
        if (ti + 1 < loop_bound) {
          if constexpr (DUAL_CACHE) {
            const bool next_main = (ti + 1) < NI;
            const int32_t* next_idx =
                next_main ? (idx_base + (ti + 1) * BI) : (idx_base_extra + (ti + 1 - NI) * BI);
            if (next_main) {
              io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf((ti + 1) & 1), next_idx,
                                                    KV_cache, io_tid, stride_kv_block);
              __threadfence_block();
              io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(sm.kv_buf((ti + 1) & 1), next_idx,
                                                             KV_cache, sm.mbar_kv((ti + 1) & 1),
                                                             io_tid, stride_kv_block, kv_l2_policy);
            } else {
              io_gather_scales<MT, PAGE_BLOCK_SIZE_EXTRA>(sm.kv_scale_buf((ti + 1) & 1), next_idx,
                                                          KV_cache_extra, io_tid,
                                                          stride_kv_block_extra);
              __threadfence_block();
              io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true>(
                  sm.kv_buf((ti + 1) & 1), next_idx, KV_cache_extra, sm.mbar_kv((ti + 1) & 1),
                  io_tid, stride_kv_block_extra, kv_l2_policy);
            }
          } else {
            const int32_t* next_idx = idx_base + (ti + 1) * BI;
            io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf((ti + 1) & 1), next_idx, KV_cache,
                                                  io_tid, stride_kv_block);
            __threadfence_block();
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(sm.kv_buf((ti + 1) & 1), next_idx,
                                                           KV_cache, sm.mbar_kv((ti + 1) & 1),
                                                           io_tid, stride_kv_block, kv_l2_policy);
          }
        }
        bar_sync_t<1, BLOCK_THREADS>();
      }
    } else {
      auto issue_tile = [&](int logical_ti, int buf) {
        if constexpr (DUAL_CACHE) {
          const bool is_main_tile = logical_ti < main_ni;
          const int32_t* tile_idx = is_main_tile ? (idx_base + logical_ti * BI)
                                                 : (idx_base_extra + (logical_ti - main_ni) * BI);
          if (is_main_tile) {
            io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf(buf), tile_idx, KV_cache, io_tid,
                                                  stride_kv_block);
            __threadfence_block();
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(sm.kv_buf(buf), tile_idx, KV_cache,
                                                           sm.mbar_kv(buf), io_tid, stride_kv_block,
                                                           kv_l2_policy);
          } else {
            io_gather_scales<MT, PAGE_BLOCK_SIZE_EXTRA>(
                sm.kv_scale_buf(buf), tile_idx, KV_cache_extra, io_tid, stride_kv_block_extra);
            __threadfence_block();
            io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE_EXTRA, true>(
                sm.kv_buf(buf), tile_idx, KV_cache_extra, sm.mbar_kv(buf), io_tid,
                stride_kv_block_extra, kv_l2_policy);
          }
        } else {
          const int32_t* tile_idx = idx_base + logical_ti * BI;
          io_gather_scales<MT, PAGE_BLOCK_SIZE>(sm.kv_scale_buf(buf), tile_idx, KV_cache, io_tid,
                                                stride_kv_block);
          __threadfence_block();
          io_bulk_gather_tile<MT, PAGE_BLOCK_SIZE, true>(sm.kv_buf(buf), tile_idx, KV_cache,
                                                         sm.mbar_kv(buf), io_tid, stride_kv_block,
                                                         kv_l2_policy);
        }
      };

      if (loop_bound > 0) issue_tile(0, 0);

#pragma unroll 1
      for (int ti = 0; ti < loop_bound; ti++) {
        if (ti + 1 < loop_bound) {
          issue_tile(ti + 1, (ti + 1) & 1);
        }
        bar_sync_t<1, BLOCK_THREADS>();
      }
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    const int32_t* idx_base = indices + (size_t)s_i * TOPK;
    [[maybe_unused]] const int32_t* idx_base_extra = nullptr;
    if constexpr (DUAL_CACHE) {
      idx_base_extra = indices_extra + (size_t)s_i * topk_extra_declared;
    }

    // ── Quantize Q for both groups ─────────────────────────────
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      const bf16* q_base_g =
          Q + (size_t)s_i * NUM_HEADS * KV::D_QK + (size_t)(h_start + g * HPB) * KV::D_QK;
      if constexpr (CM == ComputeMode::BF16) {
        load_q_bf16_to_smem<MT, MATH_THREADS>(sm.q_nope_bf16(g), sm.q_rope() + g * HPB * D_ROPE,
                                              q_base_g, VALID_HPB);
      } else {
        quantize_q_to_smem<MT, MATH_THREADS>(sm.q_nope_fp8(g), sm.q_nope_sc(g),
                                             sm.q_rope() + g * HPB * D_ROPE, q_base_g,
                                             sm.reduce_buf(), VALID_HPB);
      }
    }

    // Preload Q rope to registers for both groups
    QRopeRegs q_rope_regs[MG_N_HG];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
      q_rope_regs[g] = preload_q_rope_regs(sm.q_rope() + g * HPB * D_ROPE, lane);

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

    bar_sync_t<2, MATH_THREADS>();
    if constexpr (ASSUME_FULL_TILES) {
      mbarrier_wait_parity(sm.mbar_kv(0), 0);
    } else {
      if (loop_bound > 0) mbarrier_wait_parity(sm.mbar_kv(0), 0);
    }

    // ── Main loop ───────────────────────────────────────────────
#pragma unroll 1
    for (int ti = 0; ti < loop_bound; ti++) {
      uint8_t* kv_smem = sm.kv_buf(ti & 1);
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

      // Per-tile data source. Single-cache: always main. Dual: route on phase.
      const int32_t* ib;
      const uint8_t* kv_global;
      size_t stride_kv_block_now;
      bool is_main = true;
      if constexpr (DUAL_CACHE) {
        is_main = (ti < main_ni);
        ib = is_main ? (idx_base + ti * BI) : (idx_base_extra + (ti - main_ni) * BI);
        kv_global = is_main ? KV_cache : KV_cache_extra;
        stride_kv_block_now = is_main ? stride_kv_block : stride_kv_block_extra;
      } else {
        ib = idx_base + ti * BI;
        kv_global = KV_cache;
        stride_kv_block_now = stride_kv_block;
      }

      // Entry base: only gid's entry needed (rope prefetch + QK rope)
      const uint8_t* entry_base_gid;
      {
        const int idx = ib[qk_nb + gid];
        if constexpr (DUAL_CACHE) {
          if (is_main) {
            entry_base_gid =
                prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(KV_cache, idx, stride_kv_block);
          } else {
            entry_base_gid = prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE_EXTRA>(
                KV_cache_extra, idx, stride_kv_block_extra);
          }
        } else {
          entry_base_gid =
              prefill_kv_entry_base<MT, PAGE_BLOCK_SIZE>(KV_cache, idx, stride_kv_block);
        }
      }

      KVRopePrefetch rope_pf = prefetch_kv_rope(
          reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

      // Init per-group w_head_sc_all
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = 0.f;

      // ── QK + softmax for both groups ────────────────────────
      float w_grp[MG_N_HG][4];
      float vsc_cache[CT::N_V_CHUNKS][2];

      // BF16 MG: both head groups consume the same KV B operand. Fuse them so
      // FP8->BF16 KV dequantization runs once per K step.
      if constexpr (CM == ComputeMode::BF16 && MG_N_HG == 2) {
        const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;
        float qk_grp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
          float scale_f;
          if constexpr (KV::SCALE_IN_KV_SMEM) {
            scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
          } else {
            scale_f = ue8m0_to_fp32(
                sm.kv_scale_buf(ti & 1)[(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
          }
#pragma unroll
          for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
            int ko = blk * KV::QUANT_TILE + ks * 16;
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

            uint32_t a00, a01, a02, a03;
            ldmatrix_load_A_bf16(a00, a01, a02, a03, sm.q_nope_bf16(0) + ko, KV::Q_NOPE_BF16_STRIDE,
                                 lane);
            MmaBf16Result r0 = mma_bf16_m16n8k16(a00, a01, a02, a03, b0, b1, qk_grp[0][0],
                                                 qk_grp[0][1], qk_grp[0][2], qk_grp[0][3]);
            qk_grp[0][0] = r0.d0;
            qk_grp[0][1] = r0.d1;
            qk_grp[0][2] = r0.d2;
            qk_grp[0][3] = r0.d3;

            uint32_t a10, a11, a12, a13;
            ldmatrix_load_A_bf16(a10, a11, a12, a13, sm.q_nope_bf16(1) + ko, KV::Q_NOPE_BF16_STRIDE,
                                 lane);
            MmaBf16Result r1 = mma_bf16_m16n8k16(a10, a11, a12, a13, b0, b1, qk_grp[1][0],
                                                 qk_grp[1][1], qk_grp[1][2], qk_grp[1][3]);
            qk_grp[1][0] = r1.d0;
            qk_grp[1][1] = r1.d1;
            qk_grp[1][2] = r1.d2;
            qk_grp[1][3] = r1.d3;
          }
        }

#pragma unroll
        for (int g = 0; g < 2; g++) {
          float* qk = qk_grp[g];
          compute_qk_rope(qk, q_rope_regs[g], rope_pf);

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
          w_grp[g][0] = s[0];
          w_grp[g][1] = s[1];
          w_grp[g][2] = s[2];
          w_grp[g][3] = s[3];
        }
      } else {
#pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
          const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;

          // QK nope MMA. BF16: m16n8k16 with per-thread FP8→BF16 dequant on KV.
          float qk[4] = {0.f, 0.f, 0.f, 0.f};
          if constexpr (CM == ComputeMode::BF16) {
#pragma unroll
            for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
              float scale_f;
              if constexpr (KV::SCALE_IN_KV_SMEM) {
                scale_f = reinterpret_cast<const float*>(kv_gid_base + KV::D_NOPE)[blk];
              } else {
                scale_f = ue8m0_to_fp32(
                    sm.kv_scale_buf(ti & 1)[(qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN + blk]);
              }
#pragma unroll
              for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
                int ko = blk * KV::QUANT_TILE + ks * 16;
                uint32_t a0, a1, a2, a3;
                ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_nope_bf16(g) + ko, KV::Q_NOPE_BF16_STRIDE,
                                     lane);
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
                MmaBf16Result r =
                    mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
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
                  fp32_to_ue8m0(sm.q_nope_sc(g)[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
              float acc0, acc1, acc2, acc3;
              init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
              const uint8_t* k_scale_base;
              if constexpr (KV::SCALE_IN_KV_SMEM) {
                k_scale_base = kv_gid_base + KV::D_NOPE;
              } else {
                k_scale_base = sm.kv_scale_buf(ti & 1) + (qk_nb + gid) * KV::SCALE_BYTES_PER_TOKEN;
              }
              uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
#pragma unroll
              for (int ks = 0; ks < QK_NOPE_KSTEPS; ks++) {
                int ko = blk * KV::QUANT_TILE + ks * 32;
                uint32_t a0, a1, a2, a3, b0, b1;
                ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_nope_fp8(g) + ko, KV::Q_NOPE_STRIDE, lane);
                ldmatrix_load_B_fp8(b0, b1, kv_warp_base + ko, KV::KV_SMEM_STRIDE, lane);
                MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1, acc0, acc1,
                                                               acc2, acc3, sfa, sfb);
                acc0 = r.d0;
                acc1 = r.d1;
                acc2 = r.d2;
                acc3 = r.d3;
              }
              const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
              const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
              commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE,
                                e1_base + KV::D_NOPE, blk);
            }
          }

          // QK rope (reuses prefetched B operands)
          compute_qk_rope(qk, q_rope_regs[g], rope_pf);

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
          w_grp[g][0] = s[0];
          w_grp[g][1] = s[1];
          w_grp[g][2] = s[2];
          w_grp[g][3] = s[3];
        }
      }
      bar_sync_t<2, MATH_THREADS>();

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
      bar_sync_t<2, MATH_THREADS>();

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
          vsc_cache[vc][0] =
              ue8m0_to_fp32(sm.kv_scale_buf(ti & 1)[e0i * KV::SCALE_BYTES_PER_TOKEN + vc]);
          vsc_cache[vc][1] =
              ue8m0_to_fp32(sm.kv_scale_buf(ti & 1)[e1i * KV::SCALE_BYTES_PER_TOKEN + vc]);
        }
      }

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

        float w0 = exp2f(w_grp[g][0] - nm0), w1 = exp2f(w_grp[g][1] - nm0);
        float w2 = exp2f(w_grp[g][2] - nm1), w3 = exp2f(w_grp[g][3] - nm1);
        w_grp[g][0] = w0;
        w_grp[g][1] = w1;
        w_grp[g][2] = w2;
        w_grp[g][3] = w3;

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
      bar_sync_t<2, MATH_THREADS>();

      // Normalize w_head_sc_all (both groups)
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
      bar_sync_t<2, MATH_THREADS>();

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
              if (wpass > 0) bar_sync_t<2, MATH_THREADS>();
#pragma unroll
              for (int g = 0; g < MG_N_HG; g++) {
                float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
                uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
                float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
                float w0 = w_grp[g][0], w1 = w_grp[g][1];
                float w2 = w_grp[g][2], w3 = w_grp[g][3];
                float wn00 = w0 * vsc0 * si0, wn01 = w1 * vsc1 * si0;
                float wn10 = w2 * vsc0 * si1, wn11 = w3 * vsc1 * si1;
                Fp8WeightQuad wq =
                    quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
                int wrow0 = gid, wrow1 = gid + 8;
                if constexpr (USE_WFP8_ROW_XOR) {
                  wrow0 = wfp8_row_xor(wrow0);
                  wrow1 = wfp8_row_xor(wrow1);
                }
                cur_wfp8[wrow0 * (BI + 16) + e0i] = wq.h0_e0;
                cur_wfp8[wrow0 * (BI + 16) + e1i] = wq.h0_e1;
                cur_wfp8[wrow1 * (BI + 16) + e0i] = wq.h1_e0;
                cur_wfp8[wrow1 * (BI + 16) + e1i] = wq.h1_e1;
              }
              bar_sync_t<2, MATH_THREADS>();

#pragma unroll
              for (int g = 0; g < MG_N_HG; g++) {
                uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
                for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                  int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
#pragma unroll
                  for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                    int ko = kstep * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(a0, a1, a2, a3, cur_wfp8 + ko,
                                                                 BI + 16, lane);
                    d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                    MmaFp8Result r =
                        mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv_acc[g][nt][0], xv_acc[g][nt][1],
                                         xv_acc[g][nt][2], xv_acc[g][nt][3]);
                    xv_acc[g][nt][0] = r.d0;
                    xv_acc[g][nt][1] = r.d1;
                    xv_acc[g][nt][2] = r.d2;
                    xv_acc[g][nt][3] = r.d3;
                  }
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
              uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
              float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
              float w0 = w_grp[g][0], w1 = w_grp[g][1];
              float w2 = w_grp[g][2], w3 = w_grp[g][3];
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
              cur_wfp8[wrow0 * (BI + 16) + e0i] = f00.__x;
              cur_wfp8[wrow0 * (BI + 16) + e1i] = f01.__x;
              cur_wfp8[wrow1 * (BI + 16) + e0i] = f10.__x;
              cur_wfp8[wrow1 * (BI + 16) + e1i] = f11.__x;
            }
            bar_sync_t<2, MATH_THREADS>();

            // Both head groups use the same V B operand for a given chunk/dim.
            if constexpr (MG_N_HG == 2) {
              float* vc_sc0 = sm.w_head_sc_all() + vc * HPB;
              float* vc_sc1 = sm.w_head_sc_all() + SMG::WSC_GRP_STRIDE + vc * HPB;
              uint8_t* cur_wfp8_g0 = wfp8_parity;
              uint8_t* cur_wfp8_g1 = wfp8_parity + SMG::WFP8_GRP_SIZE;
#pragma unroll
              for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                float xv0[4] = {0.f, 0.f, 0.f, 0.f};
                float xv1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                  int ko = kstep * 32;
                  uint32_t b0, b1;
                  d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                  uint32_t a00, a01, a02, a03;
                  ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(a00, a01, a02, a03, cur_wfp8_g0 + ko,
                                                               BI + 16, lane);
                  uint32_t a10, a11, a12, a13;
                  ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(a10, a11, a12, a13, cur_wfp8_g1 + ko,
                                                               BI + 16, lane);

                  MmaFp8Result r0 =
                      mma_fp8_m16n8k32(a00, a01, a02, a03, b0, b1, xv0[0], xv0[1], xv0[2], xv0[3]);
                  xv0[0] = r0.d0;
                  xv0[1] = r0.d1;
                  xv0[2] = r0.d2;
                  xv0[3] = r0.d3;

                  MmaFp8Result r1 =
                      mma_fp8_m16n8k32(a10, a11, a12, a13, b0, b1, xv1[0], xv1[1], xv1[2], xv1[3]);
                  xv1[0] = r1.d0;
                  xv1[1] = r1.d1;
                  xv1[2] = r1.d2;
                  xv1[3] = r1.d3;
                }
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
                uint8_t* cur_wfp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
                for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
                  int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
                  int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
                  float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                  for (int kstep = 0; kstep < CT::XV_KSTEPS; kstep++) {
                    int ko = kstep * 32;
                    uint32_t a0, a1, a2, a3, b0, b1;
                    ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>(a0, a1, a2, a3, cur_wfp8 + ko,
                                                                 BI + 16, lane);
                    d2_load_b_fp8<KV::KV_SMEM_STRIDE>(b0, b1, kv_smem, kstep * 32, dim, lane);
                    MmaFp8Result r =
                        mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
                    xv[0] = r.d0;
                    xv[1] = r.d1;
                    xv[2] = r.d2;
                    xv[3] = r.d3;
                  }
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
        bar_sync_t<2, MATH_THREADS>();
        if constexpr (DUAL_CACHE) {
          if (is_main) {
            xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE, MG_N_HG>(acc_rope, w_grp, ib, kv_global, mwarp,
                                                         lane, stride_kv_block_now,
                                                         reinterpret_cast<bf16*>(sm.w_fp8()));
          } else {
            xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE_EXTRA, MG_N_HG>(acc_rope, w_grp, ib, kv_global,
                                                               mwarp, lane, stride_kv_block_now,
                                                               reinterpret_cast<bf16*>(sm.w_fp8()));
          }
        } else {
          xv_rope_mma_mg<MT, PAGE_BLOCK_SIZE, MG_N_HG>(acc_rope, w_grp, ib, kv_global, mwarp, lane,
                                                       stride_kv_block_now,
                                                       reinterpret_cast<bf16*>(sm.w_fp8()));
        }
      }
      bar_arrive_t<1, BLOCK_THREADS>();
      if (ti + 1 < loop_bound) {
        const int next_phase = ((ti + 1) >> 1) & 1;
        mbarrier_wait_parity(sm.mbar_kv((ti + 1) & 1), next_phase);
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
    bar_sync_t<2, MATH_THREADS>();

    if (threadIdx.x < MG_N_HG * HPB) {
      int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < N_MATH_WARPS; w++)
        ts += sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h];
      sm.l_smem()[g * SMG::ML_GRP_STRIDE + h] = ts;
    }
    bar_sync_t<2, MATH_THREADS>();

    // ── Epilogue: BF16 output for both groups (serial) ─────────
    // Reuse kv_bufs[0] for BF16 staging (16KB needed, 29-33KB available)
    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_buf(0));
    constexpr int BF16_STAGING_STRIDE = D_V;
    constexpr size_t h_stride = D_V;
    constexpr size_t token_stride = (size_t)NUM_HEADS * D_V;

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
      bar_sync_t<2, MATH_THREADS>();

      // Coalesced write
      {
        const int g_h_start = h_start + g * HPB;
        const size_t out_base = (size_t)s_i * token_stride + (size_t)g_h_start * h_stride;
        constexpr int BF16_PER_STORE = 8;
        constexpr int STORES_PER_HEAD = D_V / BF16_PER_STORE;
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
        size_t lse_idx = (size_t)s_i * NUM_HEADS + (h_start + g * HPB + h);
        out_lse[lse_idx] = lse;
      }

      if (g < MG_N_HG - 1) bar_sync_t<2, MATH_THREADS>();
    }
  }
}

// Single-cache __global__ wrapper.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    sparse_mla_prefill_mg_kernel(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                                 const int32_t* __restrict__ indices, bf16* __restrict__ output,
                                 float* __restrict__ out_lse,
                                 const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                                 __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/false,
                  /*PAGE_BLOCK_SIZE_EXTRA=*/PAGE_BLOCK_SIZE, MG_N_HG_T>(
      Q, KV_cache, indices, /*KV_cache_extra=*/nullptr, /*indices_extra=*/nullptr, output, out_lse,
      attn_sink, cold);
}

// Dual-cache __global__ wrapper. topk_extra is runtime; PAGE_BLOCK_SIZE_EXTRA
// stays template because it changes the KV stride.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
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
  prefill_mg_impl<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/true,
                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>(
      Q, KV_cache, indices, KV_cache_extra, indices_extra, output, out_lse, attn_sink, cold);
}

// Dual-cache full-tile wrapper for fixed-length inputs.
template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, int PAGE_BLOCK_SIZE_EXTRA,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
__global__ void __launch_bounds__(BLOCK_THREADS, 1) sparse_mla_prefill_mg_dual_fulltile_kernel(
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices, const uint8_t* __restrict__ KV_cache_extra,
    const int32_t* __restrict__ indices_extra, bf16* __restrict__ output,
    float* __restrict__ out_lse, const float* __restrict__ attn_sink,
    __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<MT, ComputeMode::BF16, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, /*DUAL_CACHE=*/true,
                  PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T, /*ASSUME_FULL_TILES=*/true>(
      Q, KV_cache, indices, KV_cache_extra, indices_extra, output, out_lse, attn_sink, cold);
}

// ============================================================================
// Sparse MLA Prefill Kernel — swapAB, warp specialized (DSV3_2 family)
//
// Single pass over all TOPK/BI tiles, with the MMA operands swapped:
//   - Candidates on M, heads on N, so one warp owns HEADS_PER_WARP heads and
//     softmax reduces inside the warp
//   - Q is register resident, so the KV ring gets the whole smem budget
//   - 8 math warps gather-fed by 4 IO warps
//   - A CTA covers HEADS_PER_CTA = 64 heads of one query token, so NUM_HEADS
//     128 launches two CTAs per token
//
// Template params (all constexpr):
//   MT:        ModelType (DSV3_2 / GLM_NSA)
//   NUM_HEADS: 64, 128
//   TOPK:      2048
// ============================================================================

// Unlike io_bulk_gather_tile this copies the whole gmem row, rope included, so
// the inline-scale ABI needs no rope buffer and the smem stride matches gmem.
template <ModelType MT>
__device__ __forceinline__ void io_bulk_gather_tile_swapab(uint8_t* dst, const int32_t* indices,
                                                           const uint8_t* __restrict__ kv_ptr,
                                                           uint64_t* mbar, int io_tid,
                                                           uint64_t cache_policy) {
  constexpr int STRIDE = SmemLayoutSwapAB<MT>::KV_STRIDE;

  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, BI * STRIDE);

#pragma unroll 1
  for (int bi = io_tid; bi < BI; bi += IO_THREADS) {
    const int idx = indices[bi];
    const uint8_t* src = kv_ptr + (size_t)(idx >= 0 ? idx : 0) * STRIDE;
    cp_async_bulk_g2s_l2hint(dst + bi * STRIDE, src, STRIDE, mbar, cache_policy);
  }
}

template <ModelType MT, int NUM_HEADS, int TOPK>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    sparse_mla_prefill_swapab_kernel(const bf16* __restrict__ Q,
                                     const uint8_t* __restrict__ KV_cache,
                                     const int32_t* __restrict__ indices,
                                     const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable
                                     bf16* __restrict__ output, float* __restrict__ out_lse,
                                     __grid_constant__ const PrefillColdParams cold) {
  using KV = KVCacheTraits<MT>;
  using CT = ComputeTraitsSwapAB<MT>;
  using L = SmemLayoutSwapAB<MT>;

  static constexpr int REPLICATE_H = NUM_HEADS / CT::HEADS_PER_CTA;
  static constexpr int HPW = CT::HEADS_PER_WARP;
  static constexpr bool SOFT_SCALE = (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32);

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_start = (blockIdx.x % REPLICATE_H) * CT::HEADS_PER_CTA;
  if (s_i >= cold.num_tokens) return;

  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : TOPK;
  topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
  const int actual_ni = (topk_len + BI - 1) / BI;

  const int warp_rank = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int32_t* idx_base = indices + (size_t)s_i * TOPK;

  extern __shared__ char smem_raw[];
  auto sm = SmemPtrsSwapAB<MT>::init(smem_raw, warp_rank < N_MATH_WARPS ? warp_rank : 0);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < 2; s++) {
      mbarrier_init(sm.mbar_kv + s, 1);             // the bulk gather signals once per tile
      mbarrier_init(sm.mbar_wr + s, N_MATH_WARPS);  // one release per math warp
    }
  }
  bar_sync_t<3, BLOCK_THREADS>();

  // ── IO warps ────────────────────────────────────────────────────
  if (warp_rank >= N_MATH_WARPS) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(24));

    const int io_tid = threadIdx.x - MATH_THREADS;
    const uint64_t kv_l2_policy = create_l2_evict_last_policy();

    int wr_phase = 1;
#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      const int buf = ti & 1;
      mbarrier_wait_parity(sm.mbar_wr + buf, wr_phase);
      io_bulk_gather_tile_swapab<MT>(sm.kv_bufs[buf], idx_base + ti * BI, KV_cache,
                                     sm.mbar_kv + buf, io_tid, kv_l2_policy);
      if (buf == 1) wr_phase ^= 1;
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(240));

    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = cold.sm_scale * LOG2E;
    // B fragment column n == gid, so this lane feeds head h_base + gid; the C
    // fragment hands back heads h_base + 2*tid and h_base + 2*tid + 1.
    const int h_base = h_start + mwarp * HPW;
    const bf16* q_base = Q + ((size_t)s_i * NUM_HEADS + h_base + gid) * KV::D_QK;

    QSwapABRegs<MT> q = quantize_q_to_regs_swapab<MT>(q_base, lane);
    KVRopePrefetch q_rope = prefetch_kv_rope(q_base + KV::D_NOPE, lane);

    uint8_t sfb[KV::NUM_SCALES];
    float q_sc[2][KV::NUM_SCALES];
    if constexpr (SOFT_SCALE) {
#pragma unroll
      for (int g = 0; g < KV::NUM_SCALES; g++) {
        q_sc[0][g] = __shfl_sync(0xffffffff, q.scale[g], 8 * tid);
        q_sc[1][g] = __shfl_sync(0xffffffff, q.scale[g], 8 * tid + 4);
      }
    } else {
#pragma unroll
      for (int g = 0; g < KV::NUM_SCALES; g++) sfb[g] = fp32_to_ue8m0(q.scale[g]);
    }

    float acc_o[CT::N_V_CHUNKS][CT::XV_MTILES][4];
#pragma unroll
    for (int c = 0; c < CT::N_V_CHUNKS; c++)
#pragma unroll
      for (int t = 0; t < CT::XV_MTILES; t++)
        acc_o[c][t][0] = acc_o[c][t][1] = acc_o[c][t][2] = acc_o[c][t][3] = 0.f;

    float warp_m[2] = {-1e30f, -1e30f};
    float warp_l[2] = {0.f, 0.f};
    int kv_phase = 0;

// ── Main loop — QK + softmax + XV ───────────────────────────
#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      const int buf = ti & 1;
      mbarrier_wait_parity(sm.mbar_kv + buf, kv_phase);
      const uint8_t* kv_smem = sm.kv_bufs[0] + buf * L::SMEM_KV_BUF;
      const int32_t* ib = idx_base + ti * BI;
      auto kv_scale = [&](int slot) {
        return reinterpret_cast<const float*>(kv_smem + (size_t)slot * L::KV_STRIDE + KV::D_NOPE);
      };

      const uint32_t vm_lo =
          __ballot_sync(0xffffffff, ib[lane] >= 0 && (ti * BI + lane) < topk_len);
      const uint32_t vm_hi =
          __ballot_sync(0xffffffff, ib[32 + lane] >= 0 && (ti * BI + 32 + lane) < topk_len);
      const uint64_t vmask = (uint64_t)vm_lo | ((uint64_t)vm_hi << 32);

      // ── QK nope MMA (A = candidates, B = heads) ─────────
      float qk[CT::MTILES][4];
#pragma unroll
      for (int m = 0; m < CT::MTILES; m++) qk[m][0] = qk[m][1] = qk[m][2] = qk[m][3] = 0.f;

#pragma unroll
      for (int mh = 0; mh < CT::MPASSES; mh++) {
        const int m0 = mh * CT::MPASS;
        uint32_t a[2][CT::MPASS][4];
#pragma unroll
        for (int mi = 0; mi < CT::MPASS; mi++)
          ldmatrix_load_A_fp8(a[0][mi][0], a[0][mi][1], a[0][mi][2], a[0][mi][3],
                              kv_smem + (size_t)((m0 + mi) * 16) * L::KV_STRIDE, L::KV_STRIDE,
                              lane);

#pragma unroll
        for (int g = 0; g < KV::NUM_SCALES; g++) {
          [[maybe_unused]] uint8_t sfa[CT::MPASS];
          [[maybe_unused]] float ks[CT::MPASS][2];
          [[maybe_unused]] float acc[CT::MPASS][4];
          if constexpr (SOFT_SCALE) {
#pragma unroll
            for (int mi = 0; mi < CT::MPASS; mi++) {
              ks[mi][0] = kv_scale((m0 + mi) * 16 + gid)[g];
              ks[mi][1] = kv_scale((m0 + mi) * 16 + gid + 8)[g];
              acc[mi][0] = acc[mi][1] = acc[mi][2] = acc[mi][3] = 0.f;
            }
          } else {
#pragma unroll
            for (int mi = 0; mi < CT::MPASS; mi++)
              sfa[mi] = KV::scale_to_ue8m0(kv_scale((m0 + mi) * 16 + gid + (lane & 1) * 8)[g]);
          }

#pragma unroll
          for (int j = 0; j < CT::STEPS_PER_GRP; j++) {
            const int ik = g * CT::STEPS_PER_GRP + j;
            if (ik + 1 < CT::NOPE_KSTEPS) {
#pragma unroll
              for (int mi = 0; mi < CT::MPASS; mi++)
                ldmatrix_load_A_fp8(
                    a[(ik + 1) & 1][mi][0], a[(ik + 1) & 1][mi][1], a[(ik + 1) & 1][mi][2],
                    a[(ik + 1) & 1][mi][3],
                    kv_smem + (size_t)((m0 + mi) * 16) * L::KV_STRIDE + (ik + 1) * 32, L::KV_STRIDE,
                    lane);
            }
#pragma unroll
            for (int mi = 0; mi < CT::MPASS; mi++) {
              const uint32_t* av = a[ik & 1][mi];
              if constexpr (SOFT_SCALE) {
                MmaFp8Result r =
                    mma_fp8_m16n8k32(av[0], av[1], av[2], av[3], q.nope[ik][0], q.nope[ik][1],
                                     acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3]);
                acc[mi][0] = r.d0;
                acc[mi][1] = r.d1;
                acc[mi][2] = r.d2;
                acc[mi][3] = r.d3;
              } else {
                float* o = qk[m0 + mi];
                MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(av[0], av[1], av[2], av[3],
                                                               q.nope[ik][0], q.nope[ik][1], o[0],
                                                               o[1], o[2], o[3], sfa[mi], sfb[g]);
                o[0] = r.d0;
                o[1] = r.d1;
                o[2] = r.d2;
                o[3] = r.d3;
              }
            }
          }

          if constexpr (SOFT_SCALE) {
#pragma unroll
            for (int mi = 0; mi < CT::MPASS; mi++) {
              float* o = qk[m0 + mi];
              o[0] = fmaf(acc[mi][0] * ks[mi][0], q_sc[0][g], o[0]);
              o[1] = fmaf(acc[mi][1] * ks[mi][0], q_sc[1][g], o[1]);
              o[2] = fmaf(acc[mi][2] * ks[mi][1], q_sc[0][g], o[2]);
              o[3] = fmaf(acc[mi][3] * ks[mi][1], q_sc[1][g], o[3]);
            }
          }
        }
      }

      // ── QK rope: stored raw, so it lands in the real domain ──
#pragma unroll
      for (int m = 0; m < CT::MTILES; m++)
        compute_qk_rope_swapab<L::KV_STRIDE>(
            qk[m],
            reinterpret_cast<const bf16*>(kv_smem + (size_t)(m * 16) * L::KV_STRIDE +
                                          KV::KV_ROPE_GMEM_OFFSET),
            q_rope, lane);

      // ── Masking + online softmax (warp-local) ───────────────
      float s[CT::MTILES][4];
#pragma unroll
      for (int m = 0; m < CT::MTILES; m++) {
        const bool v0 = (vmask >> (m * 16 + gid)) & 1ull;
        const bool v1 = (vmask >> (m * 16 + gid + 8)) & 1ull;
        s[m][0] = v0 ? qk[m][0] * sm_scale_log2e : -1e30f;
        s[m][1] = v0 ? qk[m][1] * sm_scale_log2e : -1e30f;
        s[m][2] = v1 ? qk[m][2] * sm_scale_log2e : -1e30f;
        s[m][3] = v1 ? qk[m][3] * sm_scale_log2e : -1e30f;
      }

      float alpha[2];
      float smax[2];
#pragma unroll
      for (int ih = 0; ih < 2; ih++) {
        float rm = -1e30f;
#pragma unroll
        for (int m = 0; m < CT::MTILES; m++) rm = fmaxf(rm, fmaxf(s[m][ih], s[m][2 + ih]));
        rm = warp8_reduce_max(rm);

        const float nm = fmaxf(warp_m[ih], rm);
        alpha[ih] = exp2f(warp_m[ih] - nm);
        smax[ih] = exp2f(rm - nm);
        warp_m[ih] = nm;

        float rs = 0.f;
#pragma unroll
        for (int m = 0; m < CT::MTILES; m++) {
          s[m][ih] = exp2f(s[m][ih] - nm);
          s[m][2 + ih] = exp2f(s[m][2 + ih] - nm);
          rs += s[m][ih] + s[m][2 + ih];
        }
        warp_l[ih] = fmaf(warp_l[ih], alpha[ih], rs);
      }

      // ── XV: fold V's group scale into the weights, quantize, MMA ──
#pragma unroll
      for (int g = 0; g < KV::NUM_SCALES; g++) {
        float vsc[CT::MTILES][2];
        float vsc_max = 0.f;
#pragma unroll
        for (int m = 0; m < CT::MTILES; m++) {
          vsc[m][0] = kv_scale(m * 16 + gid)[g];
          vsc[m][1] = kv_scale(m * 16 + gid + 8)[g];
          vsc_max = fmaxf(vsc_max, fmaxf(vsc[m][0], vsc[m][1]));
        }
        vsc_max = warp8_reduce_max(vsc_max);

        float wsc[2], si[2];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
          const float pmax = smax[ih] * vsc_max;
          wsc[ih] = pmax * FP8_MAX_INV;
          si[ih] = pmax > 0.f ? (FP8_MAX / pmax) : 0.f;
        }

        uint32_t pq[CT::P_PASSES][CT::MTILES];
#pragma unroll
        for (int m = 0; m < CT::MTILES; m++) {
          const float ws00 = s[m][0] * vsc[m][0] * si[0], ws10 = s[m][1] * vsc[m][0] * si[1];
          const float ws01 = s[m][2] * vsc[m][1] * si[0], ws11 = s[m][3] * vsc[m][1] * si[1];
          pq[0][m] = cvt_e4m3x4(ws00, ws10, ws01, ws11);
          if constexpr (CT::P_PASSES == 2)
            pq[1][m] = cvt_e4m3x4_residual(ws00, ws10, ws01, ws11, pq[0][m]);
        }

        __syncwarp();
#pragma unroll
        for (int p = 0; p < CT::P_PASSES; p++)
          StMatrixTransB8Tile<HPW, BI>::store(sm.p_buf + p * L::P_TILE_BYTES, pq[p], lane);
        __syncwarp();

        uint32_t pb[CT::P_PASSES][BI / 16];
#pragma unroll
        for (int p = 0; p < CT::P_PASSES; p++)
          StMatrixTransB8Tile<HPW, BI>::load_b(pb[p], sm.p_buf + p * L::P_TILE_BYTES, lane);

#pragma unroll
        for (int jc = 0; jc < CT::CHUNKS_PER_GRP; jc++) {
          const int vc = g * CT::CHUNKS_PER_GRP + jc;
          uint32_t v[CT::XV_MTILES][CT::XV_KSTEPS][4];
#pragma unroll
          for (int mt = 0; mt < CT::XV_MTILES; mt++)
#pragma unroll
            for (int kt = 0; kt < CT::XV_KSTEPS; kt++)
              ldmatrix_load_A_fp8_trans<L::KV_STRIDE>(v[mt][kt][0], v[mt][kt][1], v[mt][kt][2],
                                                      v[mt][kt][3], kv_smem, kt * 32,
                                                      vc * CT::V_CHUNK + mt * 16, lane);

          float xv[CT::XV_MTILES][4] = {};
#pragma unroll
          for (int kt = 0; kt < CT::XV_KSTEPS; kt++)
#pragma unroll
            for (int p = 0; p < CT::P_PASSES; p++)
#pragma unroll
              for (int mt = 0; mt < CT::XV_MTILES; mt++) {
                MmaFp8Result r = mma_fp8_m16n8k32(v[mt][kt][0], v[mt][kt][1], v[mt][kt][2],
                                                  v[mt][kt][3], pb[p][2 * kt], pb[p][2 * kt + 1],
                                                  xv[mt][0], xv[mt][1], xv[mt][2], xv[mt][3]);
                xv[mt][0] = r.d0;
                xv[mt][1] = r.d1;
                xv[mt][2] = r.d2;
                xv[mt][3] = r.d3;
              }

#pragma unroll
          for (int mt = 0; mt < CT::XV_MTILES; mt++) {
            float* o = acc_o[vc][mt];
            o[0] = fmaf(xv[mt][0], wsc[0], o[0] * alpha[0]);
            o[1] = fmaf(xv[mt][1], wsc[1], o[1] * alpha[1]);
            o[2] = fmaf(xv[mt][2], wsc[0], o[2] * alpha[0]);
            o[3] = fmaf(xv[mt][3], wsc[1], o[3] * alpha[1]);
          }
        }
      }

      if (lane == 0) mbarrier_arrive(sm.mbar_wr + buf);
      if (buf == 1) kv_phase ^= 1;
    }

    // ── Write BF16 output and LSE ────────────────────────────────
    // attn_sink convention (FlashMLA V4): output[h] *= sigmoid(lse_h - sink_h)
    // is folded directly into the normalizer:
    //   il = exp(lse) / (exp(lse) + exp(sink)) / exp(lse)
    //      = 1 / (l + exp(sink - m))   in log2 space
    // (working in log2 space: sum_l is in exp-domain of m, multiply sink by LOG2E).
    // A row whose candidates are all masked keeps m at -1e30f, where every slot
    // would contribute exp2(0)=1; drop l so it collapses to il=0, lse=-1e30f.
    float il[2];
    float lse[2];
#pragma unroll
    for (int ih = 0; ih < 2; ih++) {
      const bool empty = (warp_m[ih] <= -1e29f);
      const float l = warp8_reduce_sum(empty ? 0.f : warp_l[ih]);
      const float m = empty ? -1e30f : warp_m[ih];
      lse[ih] = softmax_lse(m, l);
      if (cold.attn_sink != nullptr) {
        const float sink_log2 = __ldg(cold.attn_sink + h_base + 2 * tid + ih) * LOG2E;
        const float denom = l + exp2f(sink_log2 - m);
        il[ih] = (denom > 0.f) ? (1.f / denom) : 0.f;
        lse[ih] =
            (lse[ih] != -1e30f) ? (lse[ih] + log2f(1.f + exp2f(sink_log2 - lse[ih]))) : sink_log2;
      } else {
        il[ih] = (l > 0.f) ? (1.f / l) : 0.f;
      }
    }

    // Write LSE (merged with attn_sink if present)
    if (gid == 0) {
#pragma unroll
      for (int ih = 0; ih < 2; ih++)
        out_lse[(size_t)s_i * NUM_HEADS + h_base + 2 * tid + ih] = lse[ih];
    }

    const size_t out_base = ((size_t)s_i * NUM_HEADS + h_base) * D_V;
    constexpr int O_VECS_PER_CHUNK = CT::V_CHUNK / OUT_VEC;
#pragma unroll
    for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
      __syncwarp();
#pragma unroll
      for (int mt = 0; mt < CT::XV_MTILES; mt++) {
        const int d0 = mt * 16 + gid;
        const int row0 = (2 * tid) * L::O_STAGE_STRIDE;
        const int row1 = row0 + L::O_STAGE_STRIDE;
        const float* o = acc_o[vc][mt];
        sm.o_buf[row0 + d0] = to_bf16(o[0] * il[0]);
        sm.o_buf[row1 + d0] = to_bf16(o[1] * il[1]);
        sm.o_buf[row0 + d0 + 8] = to_bf16(o[2] * il[0]);
        sm.o_buf[row1 + d0 + 8] = to_bf16(o[3] * il[1]);
      }
      __syncwarp();
#pragma unroll
      for (int i = lane; i < HPW * O_VECS_PER_CHUNK; i += 32) {
        const int h = i / O_VECS_PER_CHUNK;
        const int d8 = (i - h * O_VECS_PER_CHUNK) * OUT_VEC;
        uint4 v = *reinterpret_cast<const uint4*>(&sm.o_buf[h * L::O_STAGE_STRIDE + d8]);
        *reinterpret_cast<uint4*>(&output[out_base + (size_t)h * D_V + vc * CT::V_CHUNK + d8]) = v;
      }
    }
  }
}
