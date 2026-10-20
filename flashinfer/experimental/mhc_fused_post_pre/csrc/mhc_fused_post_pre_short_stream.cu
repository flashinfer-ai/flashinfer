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
// Fused mHC post -> pre: stream kernel for 1024 <= num_tokens < 1800 (the producer warp
// also runs the Sinkhorn of the last owned tokens, which shortens the kernel tail).

#include "mhc_fused_post_pre_kernels.cuh"

namespace flashinfer {
namespace mhc_fpp {
namespace {

// ================================================================================================
// Bandwidth-regime kernel (several tiles per cluster): warp-specialized.
//   warps 0-3  (WG0): TF32 projection; each warp keeps its 256-column x 24 slice of fn in
//                     registers (setmaxnreg 248) and runs MMA(k) while the elementwise warps work
//                     on tile k+1.
//   warps 4-11 (WG1-2): post-map(k), B2(k-2), B1(k-1).
//   The bf16 residual_cur sum of squares is accumulated by the MMA warps from their A fragments.
//   warp 12: TMA producer (+ L2 tensor prefetch BG_PF tiles ahead), 13: owner/finalize,
//   14: Sinkhorn; 15: TMA residual store plus owner-reduce/broadcast for RMS.
// ================================================================================================
constexpr int BG_S = 5;   // residual / residual_cur stages
constexpr int BG_SX = 3;  // x + post/comb metadata slots (free after post-map)
constexpr int BG_NTHREADS = 512;
constexpr int BG_NMW = 4;
constexpr int BG_EW0 = 4;
constexpr int BG_NEW = 8;
constexpr int BG_WARP_PROD = 12;
constexpr int BG_WARP_F0 = 13;
constexpr int BG_WARP_F1 = 14;
constexpr int BG_WARP_G2 = 15;
constexpr int BG_SK_TAIL = 4;  // producer handles the final owned tokens on short streams
// Residual stage: [16 tokens][4 streams][256 hidden] bf16, overwritten in place by residual_cur and
// held until B1.  x / post / comb live in a separate short ring that is released right after
// post-map, so the big ring can be one stage deeper.
constexpr int BG_RSTAGEB = TMAX * 4 * ROWB;  // 32 KB
constexpr int XS_PLM = TMAX * ROWB;
constexpr int XS_COMB = XS_PLM + TMAX * 16;
constexpr int BG_XSLOTB = XS_COMB + TMAX * 64;  // 9472 B
constexpr int BG_OFF_X = BG_S * BG_RSTAGEB;
constexpr int BG_OFF_G1 = BG_OFF_X + BG_SX * BG_XSLOTB;
constexpr int BG_OFF_PM = BG_OFF_G1 + BG_S * CL * G1_REC * 4;
constexpr int BG_OFF_G2P = BG_OFF_PM + BG_S * TMAX * 16;      // owner: [s][token][sender rank]
constexpr int BG_OFF_RS = BG_OFF_G2P + BG_S * TMAX * CL * 4;  // [s][token] final RMS factor
constexpr int BG_OFF_RED = BG_OFF_RS + BG_S * TMAX * 4;
constexpr int BG_RED_LD = 28;  // 24 mixes + sumsq, padded
constexpr int BG_OFF_Q = BG_OFF_RED + 2 * BG_NMW * TMAX * BG_RED_LD * 4;
constexpr int BG_OFF_BAR = BG_OFF_Q + QN * QREC * 4;
constexpr int BG_B_FULL = 0, BG_B_EMPTY = BG_S, BG_B_RCR = 2 * BG_S, BG_B_G1 = 3 * BG_S;
constexpr int BG_B_PM = 4 * BG_S, BG_B_G2 = 5 * BG_S, BG_B_G2P = 6 * BG_S, BG_B_XE = 7 * BG_S;
constexpr int BG_B_QF = 7 * BG_S + BG_SX, BG_B_QE = 7 * BG_S + BG_SX + QN;
constexpr int BG_NBAR = 7 * BG_S + BG_SX + 2 * QN;
constexpr int BG_SMEM_BYTES = BG_OFF_BAR + BG_NBAR * 8;
static_assert(BG_SMEM_BYTES <= 227 * 1024, "smem");
static_assert(BG_XSLOTB % 128 == 0 && BG_OFF_X % 1024 == 0, "align");
static_assert(BG_OFF_G1 % 16 == 0 && BG_OFF_PM % 16 == 0 && BG_OFF_G2P % 16 == 0, "align");
static_assert(BG_OFF_RED % 16 == 0 && BG_OFF_Q % 16 == 0 && BG_OFF_BAR % 8 == 0, "align");

__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* tm, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
                   reinterpret_cast<uint64_t>(tm)),
               "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* tm, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(
                   reinterpret_cast<uint64_t>(tm)),
               "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
__device__ __forceinline__ void mma_bar() {
  asm volatile("bar.sync 2, %0;" ::"n"(BG_NMW * 32) : "memory");
}

template <bool kLongRegs>
__global__ void __launch_bounds__(BG_NTHREADS, 1)
    mhc_fused_post_pre_short_stream_kernel(const Params p, const __grid_constant__ CUtensorMap tm_x,
                                           const __grid_constant__ CUtensorMap tm_res,
                                           const __grid_constant__ CUtensorMap tm_rc) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sbase = smem_u32(smem);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t r = cluster_ctarank();
  const int cid = blockIdx.x / CL;
  const int ncl = gridDim.x / CL;
  // The host only selects this kernel when ntiles(TMAX) exceeds the resident cluster count,
  // which always yields TM == TMAX; a compile-time tile height removes the per-token guards.
  constexpr int TM = TMAX;
  const int M = p.M;
  const int nlocal = (p.ntiles > cid) ? (p.ntiles - cid + ncl - 1) / ncl : 0;

#define BAR(i) (sbase + BG_OFF_BAR + (i) * 8)

  if (warp == BG_WARP_PROD && lane == 0) {
    // Pull the tensor-map descriptors in while the barriers are initialized.
    prefetch_tmap(&tm_x);
    prefetch_tmap(&tm_res);
    prefetch_tmap(&tm_rc);
  }

  // Barrier init spread over warp 0's lanes; CTA-local visibility (__syncthreads) is enough for
  // the producer's first TMA loads, which are issued before the cluster rendezvous.  Every warp
  // performs the rendezvous once, before its first DSMEM access.
  if (warp == 0) {
    if (lane < BG_S) {
      mbar_init(BAR(BG_B_FULL + lane), 1);
      mbar_init(BAR(BG_B_EMPTY + lane), BG_NEW + BG_NMW + 1);
      mbar_init(BAR(BG_B_RCR + lane), BG_NEW);
      mbar_init(BAR(BG_B_G1 + lane), 1);
      mbar_init(BAR(BG_B_PM + lane), 1);
      mbar_init(BAR(BG_B_G2 + lane), 1);
      mbar_init(BAR(BG_B_G2P + lane), 1);
    }
    if (lane < BG_SX) mbar_init(BAR(BG_B_XE + lane), BG_NEW);
    mbar_init(BAR(BG_B_QF + lane), 1);
    mbar_init(BAR(BG_B_QE + lane), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp < BG_NMW) {
    // ===================================== MMA warps =====================================
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");
    const int g = lane >> 2, q = lane & 3;
    const int tid = threadIdx.x;  // 0..127
    // fn columns of stream `warp`, hidden [r*256, r*256+256).  Convert while the
    // elementwise warps produce tile 0 so the first MMA does not pay this serially.
    uint32_t fnr[8][3][8];
    {
      const float* fb = p.fn + (size_t)warp * H + (size_t)r * W + 8 * q;
#pragma unroll
      for (int kg = 0; kg < 8; ++kg) {
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
          const float* src = fb + (size_t)(g + 8 * nt) * (4 * H) + 32 * kg;
          const float4 v0 = __ldg(reinterpret_cast<const float4*>(src));
          const float4 v1 = __ldg(reinterpret_cast<const float4*>(src + 4));
          fnr[kg][nt][0] = __float_as_uint(v0.x);
          fnr[kg][nt][1] = __float_as_uint(v0.y);
          fnr[kg][nt][2] = __float_as_uint(v0.z);
          fnr[kg][nt][3] = __float_as_uint(v0.w);
          fnr[kg][nt][4] = __float_as_uint(v1.x);
          fnr[kg][nt][5] = __float_as_uint(v1.y);
          fnr[kg][nt][6] = __float_as_uint(v1.z);
          fnr[kg][nt][7] = __float_as_uint(v1.w);
        }
      }
    }
    cluster_sync_init();
#pragma unroll
    for (int kg = 0; kg < 8; ++kg)
#pragma unroll
      for (int nt = 0; nt < 3; ++nt)
#pragma unroll
        for (int e = 0; e < 8; ++e) fnr[kg][nt][e] = f2tf32(__uint_as_float(fnr[kg][nt][e]));
    float* red = reinterpret_cast<float*>(smem + BG_OFF_RED);
    for (int k = 0; k < nlocal; ++k) {
      const int s = k % BG_S;
      mbar_wait(BAR(BG_B_RCR + s), (k / BG_S) & 1);
      const uint8_t* st = smem + s * BG_RSTAGEB + warp * ROWB + 16 * q;
      // Two accumulator chains (even / odd k-groups) halve the dependent-HMMA depth.
      float acc[2][3][4];
#pragma unroll
      for (int c2 = 0; c2 < 2; ++c2)
#pragma unroll
        for (int nt = 0; nt < 3; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[c2][nt][e] = 0.0f;
      // bf16 residual_cur sum of squares for tokens g (x) and g+8 (y), over this thread's
      // A-fragment elements (the fragments partition the 16 x 256 tile exactly).
      float2 sq = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int kg = 0; kg < 8; ++kg) {
        const uint4 A0 = *reinterpret_cast<const uint4*>(st + g * 4 * ROWB + 64 * kg);
        const uint4 A1 = *reinterpret_cast<const uint4*>(st + (g + 8) * 4 * ROWB + 64 * kg);
        const uint32_t w0[4] = {A0.x, A0.y, A0.z, A0.w};
        const uint32_t w1[4] = {A1.x, A1.y, A1.z, A1.w};
#pragma unroll
        for (int s4 = 0; s4 < 4; ++s4) {
          const uint32_t a0 = w0[s4] << 16, a2 = w0[s4] & 0xffff0000u;
          const uint32_t a1 = w1[s4] << 16, a3 = w1[s4] & 0xffff0000u;
#pragma unroll
          for (int nt = 0; nt < 3; ++nt)
            mma_tf32(acc[kg & 1][nt], a0, a1, a2, a3, fnr[kg][nt][2 * s4], fnr[kg][nt][2 * s4 + 1]);
          const float2 lo = make_float2(__uint_as_float(a0), __uint_as_float(a1));
          const float2 hi = make_float2(__uint_as_float(a2), __uint_as_float(a3));
          sq = __ffma2_rn(lo, lo, sq);
          sq = __ffma2_rn(hi, hi, sq);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(BAR(BG_B_EMPTY + s));  // stage reads done
      sq.x += __shfl_xor_sync(0xffffffffu, sq.x, 1);
      sq.y += __shfl_xor_sync(0xffffffffu, sq.y, 1);
      sq.x += __shfl_xor_sync(0xffffffffu, sq.x, 2);
      sq.y += __shfl_xor_sync(0xffffffffu, sq.y, 2);
      float* rw = red + ((k & 1) * BG_NMW + warp) * TMAX * BG_RED_LD;
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        *reinterpret_cast<float2*>(rw + g * BG_RED_LD + 8 * nt + 2 * q) =
            make_float2(acc[0][nt][0] + acc[1][nt][0], acc[0][nt][1] + acc[1][nt][1]);
        *reinterpret_cast<float2*>(rw + (g + 8) * BG_RED_LD + 8 * nt + 2 * q) =
            make_float2(acc[0][nt][2] + acc[1][nt][2], acc[0][nt][3] + acc[1][nt][3]);
      }
      if (q == 0) {
        rw[g * BG_RED_LD + NMIX] = sq.x;
        rw[(g + 8) * BG_RED_LD + NMIX] = sq.y;
      }
      mma_bar();
      const uint32_t g1rec = sbase + BG_OFF_G1 + ((s * CL + r) * G1_REC) * 4;
      const float* rb = red + (k & 1) * BG_NMW * TMAX * BG_RED_LD;
      if (tid < TM * 6) {
        const int t = tid / 6, n4 = tid % 6;
        float4 sum = *reinterpret_cast<const float4*>(rb + t * BG_RED_LD + 4 * n4);
#pragma unroll
        for (int w = 1; w < BG_NMW; ++w) {
          const float4 v =
              *reinterpret_cast<const float4*>(rb + w * TMAX * BG_RED_LD + t * BG_RED_LD + 4 * n4);
          sum.x += v.x;
          sum.y += v.y;
          sum.z += v.z;
          sum.w += v.w;
        }
        st_async_v4(mapa(g1rec + 16 * n4, t), sum.x, sum.y, sum.z, sum.w,
                    mapa(BAR(BG_B_G1 + s), t));
      } else if (tid >= 96 && tid < 96 + TM) {
        const int t = tid - 96;
        float sum = rb[t * BG_RED_LD + NMIX];
#pragma unroll
        for (int w = 1; w < BG_NMW; ++w) sum += rb[w * TMAX * BG_RED_LD + t * BG_RED_LD + NMIX];
        st_async_b32(mapa(g1rec + 4 * 24, t), sum, mapa(BAR(BG_B_G1 + s), t));
      }
    }
  } else if (warp < BG_EW0 + BG_NEW) {
    // ================================== elementwise warps =================================
    if constexpr (kLongRegs)
      asm volatile("setmaxnreg.dec.sync.aligned.u32 112;" ::: "memory");
    else
      asm volatile("setmaxnreg.dec.sync.aligned.u32 104;" ::: "memory");
    const int e = warp - BG_EW0;  // 0..7, tokens 2e, 2e+1
    uint2 nwr[2];
#pragma unroll
    for (int ch = 0; ch < 2; ++ch)
      nwr[ch] = __ldg(reinterpret_cast<const uint2*>(p.nw + (size_t)r * W + ch * 128 + 4 * lane));
    uint32_t lpb[2][2][2];
    uint2 rvb[2][2][4];
    cluster_sync_init();
    // Software pipeline: post-map(it), B2(it-3), B1(it-2).  B1(k) runs two iterations after
    // post-map(k), so the owner round trip (MMA -> partial mixes -> finalize -> pre-mix) is
    // covered by two post-maps instead of one.
    for (int it = 0; it < nlocal + 3; ++it) {
      // Pull tile it-2's finalized residual rows into registers before doing any work on
      // the current tile.  Once all lanes have loaded, this warp no longer owns the stage.
      if (it >= 2 && it - 2 < nlocal) {
        const int k = it - 2;
        const int s = k % BG_S;
        const uint8_t* st = smem + s * BG_RSTAGEB;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt)
#pragma unroll
          for (int ch = 0; ch < 2; ++ch)
#pragma unroll
            for (int i = 0; i < 4; ++i)
              rvb[tt][ch][i] = *reinterpret_cast<const uint2*>(
                  st + (2 * e + tt) * 4 * ROWB + 2 * (ch * 128 + 4 * lane) + i * ROWB);
        __syncwarp();
        if (lane == 0) mbar_arrive(BAR(BG_B_EMPTY + s));
      }
      // ------------------------------ post-map(it) ------------------------------
      if (it < nlocal) {
        const int k = it;
        const int s = k % BG_S;
        const int sx = k % BG_SX;
        if (e == 0 && lane == 0) {
          mbar_arrive_expect_tx(BAR(BG_B_PM + s), TM * 16);
          // Each token owner broadcasts one final RMS factor to this CTA.
          mbar_arrive_expect_tx(BAR(BG_B_G2 + s), TM * 4);
        }
        mbar_wait(BAR(BG_B_FULL + s), (k / BG_S) & 1);
        uint8_t* st = smem + s * BG_RSTAGEB;
        const uint8_t* xs = smem + BG_OFF_X + sx * BG_XSLOTB;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = 2 * e + tt;
          if (t < TM) {
            const float4 pl = reinterpret_cast<const float4*>(xs + XS_PLM)[t];
            const float4* cb = reinterpret_cast<const float4*>(xs + XS_COMB) + 4 * t;
            const float4 c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3];
            const float cm[4][4] = {{c0.x, c0.y, c0.z, c0.w},
                                    {c1.x, c1.y, c1.z, c1.w},
                                    {c2.x, c2.y, c2.z, c2.w},
                                    {c3.x, c3.y, c3.z, c3.w}};
            const float plv[4] = {pl.x, pl.y, pl.z, pl.w};
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              const int hb = 2 * (ch * 128 + 4 * lane);
              uint8_t* base = st + t * 4 * ROWB + hb;
              const uint2 xv = *reinterpret_cast<const uint2*>(xs + t * ROWB + hb);
              uint2 rv[4];
#pragma unroll
              for (int i = 0; i < 4; ++i) rv[i] = *reinterpret_cast<const uint2*>(base + i * ROWB);
              const float2 xf[2] = {bf2_to_f2(xv.x), bf2_to_f2(xv.y)};
              float2 rf[4][2];
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                rf[i][0] = bf2_to_f2(rv[i].x);
                rf[i][1] = bf2_to_f2(rv[i].y);
              }
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                uint32_t ow[2];
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                  float2 o = __fmul2_rn(xf[h2], f2s(plv[j]));
#pragma unroll
                  for (int i = 0; i < 4; ++i) o = __ffma2_rn(rf[i][h2], f2s(cm[i][j]), o);
                  ow[h2] = pack_bf2(o.x, o.y);
                }
                const uint2 ov = make_uint2(ow[0], ow[1]);
                *reinterpret_cast<uint2*>(base + j * ROWB) = ov;
              }
            }
          }
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(BAR(BG_B_RCR + s));
          mbar_arrive(BAR(BG_B_XE + sx));  // x / metadata slot no longer read
        }
      }
      // ------------------------------ B2(it-3) ------------------------------
      if (it >= 3) {
        const int k = it - 3;
        const int s = k % BG_S;
        const int t0 = (cid + k * ncl) * TM;
        const int nv = min(TM, M - t0);
        mbar_wait(BAR(BG_B_G2 + s), (k / BG_S) & 1);
        const float* rsbuf = reinterpret_cast<const float*>(smem + BG_OFF_RS) + s * TMAX;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = 2 * e + tt;
          if (t < nv) {
            const float rs = rsbuf[t];
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              uint2 ov;
              const float2 l0 = bf2_to_f2(lpb[tt][ch][0]), l1 = bf2_to_f2(lpb[tt][ch][1]);
              const float2 w0 = bf2_to_f2(nwr[ch].x), w1 = bf2_to_f2(nwr[ch].y);
              const float2 o0 = __fmul2_rn(__fmul2_rn(l0, f2s(rs)), w0);
              const float2 o1 = __fmul2_rn(__fmul2_rn(l1, f2s(rs)), w1);
              ov.x = pack_bf2(o0.x, o0.y);
              ov.y = pack_bf2(o1.x, o1.y);
              *reinterpret_cast<uint2*>(p.li_out + (size_t)(t0 + t) * H + (size_t)r * W + ch * 128 +
                                        4 * lane) = ov;
            }
          }
        }
      }
      // ------------------------------ B1(it-2) ------------------------------
      if (it >= 2 && it - 2 < nlocal) {
        const int k = it - 2;
        const int s = k % BG_S;
        mbar_wait(BAR(BG_B_PM + s), (k / BG_S) & 1);
        const float4* pm = reinterpret_cast<const float4*>(smem + BG_OFF_PM) + s * TMAX;
        float ssb[2] = {0.0f, 0.0f};
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = 2 * e + tt;
          if (t < TM) {
            const float4 pr = pm[t];
            float2 ss2 = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              const uint2* rv = rvb[tt][ch];
              float lv[4];
#pragma unroll
              for (int h2 = 0; h2 < 2; ++h2) {
                const uint32_t sel[4] = {h2 ? rv[0].y : rv[0].x, h2 ? rv[1].y : rv[1].x,
                                         h2 ? rv[2].y : rv[2].x, h2 ? rv[3].y : rv[3].x};
                float2 v = __fmul2_rn(bf2_to_f2(sel[0]), f2s(pr.x));
                v = __ffma2_rn(bf2_to_f2(sel[1]), f2s(pr.y), v);
                v = __ffma2_rn(bf2_to_f2(sel[2]), f2s(pr.z), v);
                v = __ffma2_rn(bf2_to_f2(sel[3]), f2s(pr.w), v);
                ss2 = __ffma2_rn(v, v, ss2);
                lv[2 * h2] = v.x;
                lv[2 * h2 + 1] = v.y;
              }
              lpb[tt][ch][0] = pack_bf2(lv[0], lv[1]);
              lpb[tt][ch][1] = pack_bf2(lv[2], lv[3]);
            }
            ssb[tt] = ss2.x + ss2.y;
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          ssb[0] += __shfl_xor_sync(0xffffffffu, ssb[0], o);
          ssb[1] += __shfl_xor_sync(0xffffffffu, ssb[1], o);
        }
        // One partial per token to its owner CTA (rank == token), which reduces the 16
        // partials and broadcasts the RMS factor (owner warp F0).
        if (lane < 2) {
          const int t = 2 * e + lane;
          const uint32_t dst = sbase + BG_OFF_G2P + ((s * TMAX + t) * CL + (int)r) * 4;
          st_async_b32(mapa(dst, t), lane ? ssb[1] : ssb[0], mapa(BAR(BG_B_G2P + s), t));
        }
      }
    }
  } else {
    if constexpr (kLongRegs)
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
    else
      asm volatile("setmaxnreg.dec.sync.aligned.u32 56;" ::: "memory");
    if (warp == BG_WARP_PROD) {
      // ================================ producer warp ================================
      // This warp never touches DSMEM: arrive on the cluster rendezvous immediately and
      // only wait (to keep the barrier phase consistent) before its first blocking wait.
      cluster_arrive();
      const uint64_t pol = policy_evict_first();
      auto prefetch = [&](int k) {
        const int t0 = (cid + k * ncl) * TM;
        tma_prefetch_2d(&tm_x, (int)r * W, t0);
        tma_prefetch_3d(&tm_res, (int)r * W, 0, t0);
      };
      const int pf = (nlocal <= 16) ? BG_S : ((nlocal <= 48) ? BG_S + 1 : BG_S + 2);
      for (int k = 0; k < nlocal; ++k) {
        const int s = k % BG_S;
        const int sx = k % BG_SX;
        const int t0 = (cid + k * ncl) * TM;
        const int nv = min(TM, M - t0);
        if (k == BG_SX) cluster_wait();
        if (k >= BG_SX) mbar_wait(BAR(BG_B_XE + sx), ((k / BG_SX) - 1) & 1);
        if (k >= BG_S) mbar_wait(BAR(BG_B_EMPTY + s), ((k / BG_S) - 1) & 1);
        if (lane == 0) {
          const uint32_t stg = sbase + s * BG_RSTAGEB;
          const uint32_t xsl = sbase + BG_OFF_X + sx * BG_XSLOTB;
          mbar_arrive_expect_tx(BAR(BG_B_FULL + s), TM * 5 * ROWB + nv * (16 + 64));
          tma_load_2d(xsl, &tm_x, (int)r * W, t0, BAR(BG_B_FULL + s), pol);
          tma_load_3d(stg, &tm_res, (int)r * W, 0, t0, BAR(BG_B_FULL + s), pol);
          bulk_g2s(xsl + XS_PLM, p.plm + (size_t)t0 * 4, nv * 16, BAR(BG_B_FULL + s), pol);
          bulk_g2s(xsl + XS_COMB, p.comb + (size_t)t0 * 16, nv * 64, BAR(BG_B_FULL + s), pol);
          // L2 prefetches start only once the first x-ring loads are queued, so they do
          // not delay the tiles the elementwise warps need first.
          if (k == BG_SX - 1 || (k == nlocal - 1 && k < BG_SX - 1)) {
            for (int kk = BG_S; kk <= k + pf && kk < nlocal; ++kk) prefetch(kk);
          } else if (k >= BG_SX && k + pf < nlocal) {
            prefetch(k + pf);
          }
        }
        __syncwarp();
      }
      if (nlocal <= BG_SX) cluster_wait();
      // For short persistent streams, the producer is idle after priming while the
      // batched Sinkhorn warp otherwise forms the kernel tail.  Consume the last four
      // owned tokens here, two at a time in the 16-lane element layout.
      if (nlocal <= 20) {
        int nown = 0;
        for (int k = 0; k < nlocal; ++k)
          if ((cid + k * ncl) * TM + (int)r < M) ++nown;
        const float* qbuf = reinterpret_cast<const float*>(smem + BG_OFF_Q);
        int head = nown - min(BG_SK_TAIL, nown);
        while (head < nown) {
          const int n = ((nown - head) & 1) ? 1 : 2;
          mbar_wait(BAR(BG_B_QF + ((head + n - 1) % QN)), ((head + n - 1) / QN) & 1);
          const int tb = lane >> 4, el = lane & 15;
          const bool act = tb < n;
          const int slot = (head + (act ? tb : 0)) % QN;
          bool bad = false;
          float c = sinkhorn_element<false, 0xffffffffu>(qbuf[slot * QREC + el], bad);
          if (__any_sync(0xffffffffu, bad)) {
            bad = false;
            c = sinkhorn_element<true, 0xffffffffu>(qbuf[slot * QREC + el], bad);
          }
          const int tok = __float_as_int(qbuf[slot * QREC + 16]);
          if (act) p.comb_out[(size_t)tok * 16 + el] = c;
          __syncwarp();
          if (lane < n) mbar_arrive(BAR(BG_B_QE + ((head + lane) % QN)));
          head += n;
        }
      }
    } else if (warp == BG_WARP_F0) {
      // ============================ owner / finalize warp =============================
      // CTA rank r owns token r of every tile.  Per tile k this warp
      //   (a) reduces the RMS square-sum partials of tile k-2 (B1 runs two tiles behind)
      //       and broadcasts the RMS factor, then
      //   (b) reduces the 16 partial projections of tile k, applies the prenorm RMS,
      //       broadcasts pre_mix, writes post_mix and queues the Sinkhorn logits.
      // (a) completes well before (b) in steady state, so (b) is never delayed by it.
      const float* g1 = reinterpret_cast<const float*>(smem + BG_OFF_G1);
      const float* part = reinterpret_cast<const float*>(smem + BG_OFF_G2P);
      if (lane == 0) {
#pragma unroll
        for (int s = 0; s < BG_S; ++s)
          if (s < nlocal) mbar_arrive_expect_tx(BAR(BG_B_G2P + s), CL * 4);
        if (nlocal > 0) mbar_arrive_expect_tx(BAR(BG_B_G1 + 0), CL * 100);
      }
      cluster_sync_init();
      const float s0 = __ldg(p.hc_scale + 0);
      const float s1 = __ldg(p.hc_scale + 1);
      const float s2 = __ldg(p.hc_scale + 2);
      const float base = (lane < NMIX) ? __ldg(p.hc_base + lane) : 0.0f;
      float* qbuf = reinterpret_cast<float*>(smem + BG_OFF_Q);
      int qidx = 0;
      for (int k = 0; k < nlocal + 2; ++k) {
        if (k >= 2) {
          const int kb = k - 2;
          const int s = kb % BG_S;
          mbar_wait(BAR(BG_B_G2P + s), (kb / BG_S) & 1);
          float rs = 0.0f;
          if (lane == 0) {
            const float4* gp = reinterpret_cast<const float4*>(part + (s * TMAX + (int)r) * CL);
            const float4 v0 = gp[0], v1 = gp[1], v2 = gp[2], v3 = gp[3];
            const float tot = (((v0.x + v0.y) + (v0.z + v0.w)) + ((v1.x + v1.y) + (v1.z + v1.w))) +
                              (((v2.x + v2.y) + (v2.z + v2.w)) + ((v3.x + v3.y) + (v3.z + v3.w)));
            rs = rsqrtf(tot * (1.0f / H) + NORM_EPS);
            if (kb + BG_S < nlocal) mbar_arrive_expect_tx(BAR(BG_B_G2P + s), CL * 4);
          }
          rs = __shfl_sync(0xffffffffu, rs, 0);
          if (lane < CL) {
            const uint32_t dst = sbase + BG_OFF_RS + (s * TMAX + (int)r) * 4;
            st_async_b32(mapa(dst, lane), rs, mapa(BAR(BG_B_G2 + s), lane));
          }
        }
        if (k < nlocal) {
          const int s = k % BG_S;
          const int t0 = (cid + k * ncl) * TM;
          mbar_wait(BAR(BG_B_G1 + s), (k / BG_S) & 1);
          float v = 0.0f;
          if (lane < 25) {
            const float* gs = g1 + s * CL * G1_REC + lane;
            float a[CL];
#pragma unroll
            for (int src = 0; src < CL; ++src) a[src] = gs[src * G1_REC];
            v = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) +
                (((a[8] + a[9]) + (a[10] + a[11])) + ((a[12] + a[13]) + (a[14] + a[15])));
          }
          // Arm the next tile's partial-mix barrier (its slot was consumed above).
          if (lane == 0 && k + 1 < nlocal)
            mbar_arrive_expect_tx(BAR(BG_B_G1 + ((k + 1) % BG_S)), CL * 100);
          const float sq = __shfl_sync(0xffffffffu, v, 24);
          const float rsq = rsqrtf(sq * (1.0f / (4 * H)) + RMS_EPS);
          const float m = __fmul_rn(v, rsq);
          float pre = 0.0f;
          if (lane < 4) pre = sigmoid_exact(__fadd_rn(__fmul_rn(m, s0), base)) + HC_PRE_EPS;
          const float p0 = __shfl_sync(0xffffffffu, pre, 0);
          const float p1 = __shfl_sync(0xffffffffu, pre, 1);
          const float p2 = __shfl_sync(0xffffffffu, pre, 2);
          const float p3 = __shfl_sync(0xffffffffu, pre, 3);
          if (lane < CL) {
            const uint32_t dst = sbase + BG_OFF_PM + ((s * TMAX + r) * 4) * 4;
            st_async_v4(mapa(dst, lane), p0, p1, p2, p3, mapa(BAR(BG_B_PM + s), lane));
          }
          const int tok = t0 + (int)r;
          if (tok < M) {
            if (lane >= 4 && lane < 8)
              p.post_out[(size_t)tok * 4 + (lane - 4)] =
                  sigmoid_exact(__fadd_rn(__fmul_rn(m, s1), base)) * HC_POST_MULT;
            const float cl = __fadd_rn(__fmul_rn(m, s2), base);
            const int slot = qidx % QN;
            if (qidx >= QN) mbar_wait(BAR(BG_B_QE + slot), ((qidx / QN) - 1) & 1);
            if (lane >= 8 && lane < 24) qbuf[slot * QREC + (lane - 8)] = cl;
            if (lane == 0) qbuf[slot * QREC + 16] = __int_as_float(tok);
            __syncwarp();
            if (lane == 0) mbar_arrive(BAR(BG_B_QF + slot));
            ++qidx;
          }
        }
      }
    } else if (warp == BG_WARP_F1) {
      // ================================ Sinkhorn warp ================================
      cluster_sync_init();
      int nown = 0;
      if ((int)r < TM) {
        for (int k = 0; k < nlocal; ++k)
          if ((cid + k * ncl) * TM + (int)r < M) ++nown;
      }
      if (nlocal <= 20) nown -= min(BG_SK_TAIL, nown);
      const float* qbuf = reinterpret_cast<const float*>(smem + BG_OFF_Q);
      int head = 0;
      while (head < nown) {
        // Wait for a full batch of 8 queued tokens (or the remainder): the pass costs the
        // same with 1 or 8 tokens, and running it once per token kept this warp ~95% busy,
        // stealing issue slots from the elementwise warps on its SM sub-partition.  The
        // owner fills slots in order, so the last slot's completion (release-cumulative)
        // covers every earlier slot of the batch.
        const int n = min(8, nown - head);
        mbar_wait(BAR(BG_B_QF + ((head + n - 1) % QN)), ((head + n - 1) / QN) & 1);
        const int tb = lane >> 2, i = lane & 3;
        const bool act = tb < n;
        const int slot = (head + (act ? tb : 0)) % QN;
        float c[4];
        {
          const float4 v = *reinterpret_cast<const float4*>(qbuf + slot * QREC + 4 * i);
          c[0] = v.x;
          c[1] = v.y;
          c[2] = v.z;
          c[3] = v.w;
        }
        const int tok = __float_as_int(qbuf[slot * QREC + 16]);
        bool bad = false;
        sinkhorn_rows<false, true>(c, bad);
        if (__any_sync(0xffffffffu, bad)) {
          const float4 v = *reinterpret_cast<const float4*>(qbuf + slot * QREC + 4 * i);
          c[0] = v.x;
          c[1] = v.y;
          c[2] = v.z;
          c[3] = v.w;
          sinkhorn_rows<true, true>(c, bad);
        }
        if (act)
          *reinterpret_cast<float4*>(p.comb_out + (size_t)tok * 16 + 4 * i) =
              make_float4(c[0], c[1], c[2], c[3]);
        __syncwarp();
        if (lane < n) mbar_arrive(BAR(BG_B_QE + ((head + lane) % QN)));
        head += n;
      }
    } else if (warp == BG_WARP_G2) {
      // ========================== TMA residual_cur store warp ==========================
      // Stores each residual_cur tile straight from its stage and releases the stage as
      // soon as the TMA engine has read it.  Square sums are exchanged all-to-all, so this
      // warp never waits on a cross-CTA reduction.
      cluster_sync_init();
      for (int k = 0; k < nlocal; ++k) {
        const int s = k % BG_S;
        mbar_wait(BAR(BG_B_RCR + s), (k / BG_S) & 1);
        if (lane == 0) {
          tma_store_3d(&tm_rc, (int)r * W, 0, (cid + k * ncl) * TM, sbase + s * BG_RSTAGEB);
          bulk_commit();
          bulk_wait_read<0>();
          mbar_arrive(BAR(BG_B_EMPTY + s));
        }
        __syncwarp();
      }
      if (lane == 0) bulk_wait<0>();
    }
  }
#undef BAR
}

}  // namespace

// Only the <false> register split is reachable in this regime (see kStreamLongRegs).
cudaError_t prepare_short_stream(int* stream_clusters) {
  return configure_cluster_kernel(
      reinterpret_cast<const void*>(mhc_fused_post_pre_short_stream_kernel<false>), BG_NTHREADS,
      BG_SMEM_BYTES, stream_clusters);
}

cudaError_t launch_short_stream(const LaunchArgs& args, int num_clusters, const TensorMaps& maps,
                                cudaStream_t stream) {
  return launch_cluster_kernel(mhc_fused_post_pre_short_stream_kernel<false>, BG_NTHREADS,
                               BG_SMEM_BYTES, num_clusters, make_params(args), maps, stream);
}

}  // namespace mhc_fpp
}  // namespace flashinfer
