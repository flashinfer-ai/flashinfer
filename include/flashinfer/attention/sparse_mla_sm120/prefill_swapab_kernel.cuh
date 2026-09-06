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

#include "prefill_common.cuh"
// ============================================================================
// Sparse MLA Prefill Kernel — swapAB, warp specialized (DSV3_2 family)
//
// Single pass over all topk/BI tiles, with the MMA operands swapped:
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
//
// topk is runtime (cold.topk): the indices row width, a whole number of BI
// candidate tiles.
// ============================================================================

// Unlike io_bulk_gather_tile this copies the whole gmem row, rope included, so
// the inline-scale ABI needs no rope buffer and the smem stride matches gmem.
// `idx` is this IO thread's candidate index, staged in a register by the caller
// one tile ahead of use; with BI <= IO_THREADS the thread's slot is io_tid.
template <ModelType MT>
__device__ __forceinline__ void io_bulk_gather_tile_swapab(uint8_t* dst, int idx,
                                                           const uint8_t* __restrict__ kv_ptr,
                                                           uint64_t* mbar, int io_tid,
                                                           uint64_t cache_policy) {
  constexpr int STRIDE = SmemLayoutSwapAB<MT>::KV_STRIDE;
  static_assert(BI <= IO_THREADS, "per-thread index staging assumes one candidate per IO thread");

  if (io_tid == 0) mbarrier_arrive_expect_tx(mbar, BI * STRIDE);
  if (io_tid >= BI) return;

  const uint8_t* src = kv_ptr + (size_t)(idx >= 0 ? idx : 0) * STRIDE;
  cp_async_bulk_g2s_l2hint(dst + io_tid * STRIDE, src, STRIDE, mbar, cache_policy);
}

// Warm L2 for a tile gathered one iteration later; addressing matches
// io_bulk_gather_tile_swapab. Pure hint, padding indices skipped.
template <ModelType MT>
__device__ __forceinline__ void io_bulk_prefetch_l2_swapab(int idx,
                                                           const uint8_t* __restrict__ kv_ptr,
                                                           int io_tid, uint64_t cache_policy) {
  constexpr int STRIDE = SmemLayoutSwapAB<MT>::KV_STRIDE;
  if (io_tid >= BI || idx < 0) return;
  cp_async_bulk_prefetch_l2_hint(kv_ptr + (size_t)idx * STRIDE, STRIDE, cache_policy);
}

template <ModelType MT, int NUM_HEADS>
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

  const int topk = cold.topk;
  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : topk;
  topk_len = topk_len < 0 ? 0 : (topk_len > topk ? topk : topk_len);
  const int actual_ni = (topk_len + BI - 1) / BI;

  const int warp_rank = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int32_t* idx_base = indices + (size_t)s_i * topk;

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

    // Stage this thread's candidate index a tile ahead: the LDG for tile ti+2
    // is issued before the mbar_wr wait so its latency hides behind the wait.
    // `pf` (tile ti+1) warms L2 after the gather issue: this pipeline is one
    // tile deep, so the prefetch must not delay the gather the math waits on.
    auto ld_idx = [&](int t) -> int {
      return (t < actual_ni && io_tid < BI) ? __ldg(idx_base + t * BI + io_tid) : -1;
    };
    int staged = ld_idx(0);
    int pf = ld_idx(1);
    int wr_phase = 1;
#pragma unroll 1
    for (int ti = 0; ti < actual_ni; ti++) {
      const int buf = ti & 1;
      const int next = ld_idx(ti + 2);
      mbarrier_wait_parity(sm.mbar_wr + buf, wr_phase);
      io_bulk_gather_tile_swapab<MT>(sm.kv_bufs[buf], staged, KV_cache, sm.mbar_kv + buf, io_tid,
                                     kv_l2_policy);
      io_bulk_prefetch_l2_swapab<MT>(pf, KV_cache, io_tid, kv_l2_policy);
      staged = pf;
      pf = next;
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
    KVRopePrefetch<MT> q_rope = prefetch_kv_rope<MT>(q_base + KV::D_NOPE, lane);

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
        compute_qk_rope_swapab<L::KV_STRIDE, MT>(
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
        out_lse[(size_t)s_i * cold.stride_out_lse + h_base + 2 * tid + ih] = lse[ih];
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
