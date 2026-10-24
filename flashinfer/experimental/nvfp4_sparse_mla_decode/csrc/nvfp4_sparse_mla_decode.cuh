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
// NVFP4 sparse-MLA decode for SM100 and SM103 (compute capability 10.0 and 10.3).
// Framework-agnostic: raw pointers only.
//
// Each query token t attends to the KV rows a sparse indexer picked for it (DeepSeek-V3.2 / GLM-5
// DSA style):
//   out[t, h] = output_scale * sum_k softmax_k(scale * q[t, h] . K[i_tk]) * V[i_tk]
// where K is a row's 576-dim latent (512 NoPE + 64 RoPE) and V its first 512 dims. Rows use vLLM's
// `nvfp4_ds_mla` cache format, 352 bytes per token:
//   bytes   0..255  512 NoPE values, e2m1, two per byte (element 2i in the low nibble)
//   bytes 256..319  64 RoPE values, e4m3
//   bytes 320..351  32 e4m3 scales, one per 16 NoPE values; block b's scale sits at byte 320 + 8 *
//   (b % 4) + b / 4
// q is e4m3 [T, 16, 576], the indices int32 [T, topk_width] flat row ids (-1 = no key), out bf16
// [T, 16, 512]. A token whose indices are all -1 gets zeros.
//
// One thread-block cluster of C CTAs (2..8, chosen per launch) serves one query token; CTA r walks
// stages [r * S / C, (r + 1) * S / C) of the token's S = topk_width / 32 key stages. 16 warps hold
// fixed roles and hand each 32-key stage down a ring of NSLOT slots:
//   warps  0-7   dequantize: fetch 4 rows each with cp.async (a -1 row is zero-filled without a
//   global read) into a
//                raw ring one slot deeper than the K ring, then expand e2m1 x e4m3 into an f16 K/V
//                tile (exact in f16)
//   warps  8-11  S = Q K^T on mma.sync m16n8k16, one k-quarter each with Q in registers, then the
//   online softmax for
//                4 heads each; P (f16) and the rescale factor go to the P.V warps
//   warps 12-15  O = alpha * O + P V, 128 output dims each, V fragments via ldmatrix.trans
// The cluster's CTAs then push f16 partial outputs and their (max, sum) over distributed shared
// memory to the CTA that owns each 8-dim output block and merge after one cluster barrier: no
// second kernel, no workspace. Shared memory: 218,624 bytes per CTA at NSLOT 3, so one CTA per SM.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace nvfp4_sparse_mla_decode {

namespace cg = cooperative_groups;

constexpr int H = 16, D = 576, DV = 512, ROWB = 352, RAWP = 368;
constexpr int THREADS = 512, SUB = 32, KP = D + 8, PP = SUB + 8, SP = SUB + 8, MAXKEYS = 1024,
              MAXC = 8;
constexpr int KSTEPS = D / 16, KQ = KSTEPS / 4;  // 36, 9 per k-quarter
constexpr int SLOTH =
    H * 8 + 8;  // halves per (src, dim block): 16 heads x 8 dims + 8 pad (bank spread in the merge)
constexpr int SM_RECVO = 19456;  // f16 partials [C][ceil(64/C)][SLOTH]: max over C = 19,040 (C=7)
#ifndef NSLOT_N
#define NSLOT_N 3
#endif
constexpr int NSLOT = NSLOT_N;
constexpr int NDEQ = 8, NQK = 4, NPV = 4;  // warps per role
constexpr int RSLOT = NSLOT + 1;           // raw ring: one slot deeper than the K ring
constexpr int SM_RAW = RSLOT * SUB * RAWP;
constexpr int SM_KF = NSLOT * SUB * KP * 2;
constexpr int SM_S = NSLOT * 4 * H * SP * 4;
constexpr int SM_P = NSLOT * H * PP * 2;
constexpr int SM_ALPHA = NSLOT * H * 4;
constexpr int SM_IDX = MAXKEYS * 4;  // key indices of this CTA
constexpr int SM_MB = 64;            // reserved (unused): keeps the validated layout
constexpr int SM_RECV = SM_RECVO + MAXC * H * 2 * 4;  // partials + recv_ml [C][H][2]
constexpr int SM_TOTAL =
    SM_RAW + SM_KF + SM_S + SM_P + SM_ALPHA + SM_IDX + SM_MB + SM_RECV;  // 218,624 at NSLOT 3
static_assert(H * D <= SM_S / NSLOT && NSLOT >= 2 && NSLOT <= 4,
              "Q staging fits in S slot 0; ring depth");

// named barriers (id, participant count)
constexpr int PSLOT = 2;
enum : int {
  B_KVQ0 = 1,
  B_KVP0 = B_KVQ0 + NSLOT,
  B_KV_FREE0 = B_KVP0 + NSLOT,
  B_P_FULL0 = B_KV_FREE0 + NSLOT,
  B_P_FREE0 = B_P_FULL0 + PSLOT,
  B_QK_SYNC = B_P_FREE0 + PSLOT
};
constexpr int N_KVQ = 32 * (NDEQ + NQK), N_KVP = 32 * (NDEQ + NPV);  // 384 each
static_assert(B_QK_SYNC <= 15, "named barrier ids");
constexpr int N_KV_FREE = 32 * (NQK + NPV + NDEQ);  // 512
constexpr int N_QK = 32 * NQK;                      // 128
constexpr int N_P = 32 * (NQK + NPV);               // 256

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cp_async16_zfill(
    void* dst, const void* src, uint32_t src_bytes) {  // src_bytes = 0: zero-fill, no global read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
               "r"(src_bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
// warp w fetches rows [4w, +4) of a stage: 88 16-B chunks over 32 lanes (row = j / 22, chunk = j %
// 22); -1 rows are zero-filled
__device__ __forceinline__ void load_rows(uint8_t* slot, const int32_t* idx_stage,
                                          const uint8_t* kv, int warp, int lane) {
#pragma unroll
  for (int j = lane; j < 88; j += 32) {
    const int ri = j / 22, ch = j - ri * 22, r = warp * 4 + ri;
    const int32_t ii = idx_stage[r];
    cp_async16_zfill(slot + r * RAWP + ch * 16, kv + (ii < 0 ? 0 : (size_t)ii * ROWB) + ch * 16,
                     ii < 0 ? 0u : 16u);  // -1 rows: zeros, no traffic
  }
}
__device__ __forceinline__ void mma_f16_16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t fp4x2_to_h2_u32(uint32_t byte) {
  uint32_t r;
  asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %1; cvt.rn.f16x2.e2m1x2 %0, b0; }"
      : "=r"(r)
      : "r"(byte));
  return r;
}
__device__ __forceinline__ uint32_t fp8x2_to_h2_u32(uint32_t pair) {
  uint32_t r;
  asm("{ .reg .b16 h0, h1; mov.b32 {h0, h1}, %1; cvt.rn.f16x2.e4m3x2 %0, h0; }"
      : "=r"(r)
      : "r"(pair));
  return r;
}
__device__ __forceinline__ __half fp8_to_h(uint8_t b) {
  uint32_t r = fp8x2_to_h2_u32((uint32_t)b);
  return __low2half(*reinterpret_cast<__half2*>(&r));
}
__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, __half2 s) {
  __half2 x = __hmul2(*reinterpret_cast<__half2*>(&a), s);
  return *reinterpret_cast<uint32_t*>(&x);
}

__global__ void __launch_bounds__(THREADS, 1)
    nvfp4_sparse_mla_decode_kernel(const uint8_t* __restrict__ kv, const uint8_t* __restrict__ q,
                                   const int32_t* __restrict__ topk,
                                   __nv_bfloat16* __restrict__ out, int topk_width,
                                   float sm_scale_log2, float o_scale) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* raw = smem;                                                  // [2][SUB][RAWP]
  __half* Kf = reinterpret_cast<__half*>(smem + SM_RAW);                // [2][SUB][KP]
  float* S = reinterpret_cast<float*>(smem + SM_RAW + SM_KF);           // [2][4][H][SP]
  __half* P = reinterpret_cast<__half*>(smem + SM_RAW + SM_KF + SM_S);  // [2][H][PP]
  float* alpha_sh = reinterpret_cast<float*>(smem + SM_RAW + SM_KF + SM_S + SM_P);  // [2][H]
  int32_t* idx = reinterpret_cast<int32_t*>(smem + SM_RAW + SM_KF + SM_S + SM_P + SM_ALPHA);
  __half* recv_o = reinterpret_cast<__half*>(smem + SM_RAW + SM_KF + SM_S + SM_P + SM_ALPHA +
                                             SM_IDX + SM_MB);  // [C][BPC][SLOTH] f16
  float* recv_ml = reinterpret_cast<float*>(smem + SM_RAW + SM_KF + SM_S + SM_P + SM_ALPHA +
                                            SM_IDX + SM_MB + SM_RECVO);  // [C][H][2]

  cg::cluster_group cluster = cg::this_cluster();
  const int C = cluster.num_blocks(), rank = cluster.block_rank(), t = blockIdx.x / C,
            BPC = (DV / 8 + C - 1) / C;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int nst_total = topk_width / SUB, s0 = (rank * nst_total) / C,
            s1 = ((rank + 1) * nst_total) / C;
  const int nstage = s1 - s0, keys_per_cta = nstage * SUB;
  const int32_t* my_topk = topk + (size_t)t * topk_width + s0 * SUB;

  // ---- prologue (all warps): Q staging, indices, the first NSLOT stages of rows
  constexpr int QV = H * D / 16;
  const uint8_t* qt = q + (size_t)t * H * D;
  uint4 qv0 = *reinterpret_cast<const uint4*>(qt + tid * 16), qv1 = make_uint4(0, 0, 0, 0);
  if (tid < QV - THREADS) qv1 = *reinterpret_cast<const uint4*>(qt + (THREADS + tid) * 16);
  for (int i = tid; i < keys_per_cta; i += THREADS) idx[i] = my_topk[i];
  __syncthreads();
  __syncthreads();
  if (warp < NDEQ) {  // DEQ warps: 8 x 4 = 32 rows per stage, NSLOT stages ahead
#pragma unroll
    for (int i = 0; i < NSLOT; ++i) {
      load_rows(raw + i * SUB * RAWP, idx + i * SUB, kv, warp, lane);
      cp_async_commit();
    }
  }
  uint8_t* Qs = reinterpret_cast<uint8_t*>(S);
  *reinterpret_cast<uint4*>(Qs + tid * 16) = qv0;
  if (tid < QV - THREADS) *reinterpret_cast<uint4*>(Qs + (THREADS + tid) * 16) = qv1;
  __syncthreads();

  if (warp < NDEQ) {
    // ================= DEQ role: warp -> rows [4*warp, +4) of each stage
    for (int s = 0; s < nstage; ++s) {
      const int b = s % NSLOT;
      {  // refill: rows of stage s + NSLOT go into the raw slot stage s - 1 used
        cp_async_wait<NSLOT - 1>();
        __syncwarp();  // this warp's raw rows for stage s landed
        if (s + NSLOT < nstage)
          load_rows(raw + ((s + NSLOT) % RSLOT) * SUB * RAWP, idx + (s + NSLOT) * SUB, kv, warp,
                    lane);  // slot of stage s-1: consumed
        cp_async_commit();  // one group per stage keeps wait_group<NSLOT-1> exact
      }
      if (s >= NSLOT)
        bar_sync(B_KV_FREE0 + b, N_KV_FREE);  // consumers released K slot b (stage s-NSLOT)
      uint8_t* rb = raw + (s % RSLOT) * SUB * RAWP;
      __half* kb = Kf + b * SUB * KP;
#pragma unroll
      for (int ri = 0; ri < 4; ++ri) {
        const int r = warp * 4 + ri;
        const uint8_t* rr = rb + r * RAWP;
        __half* kr = kb + r * KP;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int e0 = 256 * k + 8 * lane, blk = e0 >> 4;
          __half sc = fp8_to_h(rr[320 + 8 * (blk & 3) + (blk >> 2)]);
          __half2 sc2 = __halves2half2(sc, sc);
          uint32_t w = *reinterpret_cast<const uint32_t*>(rr + (e0 >> 1));
          *reinterpret_cast<uint4*>(kr + e0) = make_uint4(
              hmul2_u32(fp4x2_to_h2_u32(w), sc2), hmul2_u32(fp4x2_to_h2_u32(w >> 8), sc2),
              hmul2_u32(fp4x2_to_h2_u32(w >> 16), sc2), hmul2_u32(fp4x2_to_h2_u32(w >> 24), sc2));
        }
        *reinterpret_cast<uint32_t*>(kr + DV + 2 * lane) =
            fp8x2_to_h2_u32(*reinterpret_cast<const uint16_t*>(rr + 256 + 2 * lane));
      }
      bar_arrive(B_KVQ0 + b, N_KVQ);
      bar_arrive(B_KVP0 + b, N_KVP);  // K slot b ready for QK and for PV
    }
  } else if (warp < NDEQ + NQK) {
    // ================= QK + softmax role: warp -> k-quarter kq = warp - 8, all 4 n-blocks (32
    // keys)
    const int kq = warp - NDEQ;
    uint32_t qa[KQ][4];
#pragma unroll
    for (int i = 0; i < KQ; ++i) {
      int d = (kq * KQ + i) * 16 + 2 * c;
      qa[i][0] = fp8x2_to_h2_u32(*reinterpret_cast<const uint16_t*>(Qs + g * D + d));
      qa[i][1] = fp8x2_to_h2_u32(*reinterpret_cast<const uint16_t*>(Qs + (g + 8) * D + d));
      qa[i][2] = fp8x2_to_h2_u32(*reinterpret_cast<const uint16_t*>(Qs + g * D + d + 8));
      qa[i][3] = fp8x2_to_h2_u32(*reinterpret_cast<const uint16_t*>(Qs + (g + 8) * D + d + 8));
    }
    // softmax lanes: head hs = 4*kq + (lane >> 3), keys [4*(lane & 7), +4)
    const int hs = 4 * kq + (lane >> 3), k0 = 4 * (lane & 7);
    float m_run = -INFINITY, l_run = 0.f;
    bar_sync(B_QK_SYNC, N_QK);  // Q fragments built before slot 0 of S is reused
    for (int s = 0; s < nstage; ++s) {
      const int b = s % NSLOT;
      const int32_t* ib = idx + s * SUB;
      const int pb = s % PSLOT;
      bar_sync(B_KVQ0 + b, N_KVQ);
      if (s >= PSLOT) bar_sync(B_P_FREE0 + pb, N_P);  // PV done with P slot pb (stage s-PSLOT)
      const __half* kb = Kf + b * SUB * KP;
      float sacc[4][4], sacc2[4][4];
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        sacc[nb][0] = sacc[nb][1] = sacc[nb][2] = sacc[nb][3] = 0.f;
        sacc2[nb][0] = sacc2[nb][1] = sacc2[nb][2] = sacc2[nb][3] = 0.f;
      }
      const __half* krow = kb + (lane & 7) * KP + (((lane >> 3) & 1) << 3) + kq * KQ * 16;
#pragma unroll
      for (int i = 0; i < KQ; ++i) {
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
          uint32_t bf[2];
          ldmatrix_x2(bf, krow + nb * 8 * KP + i * 16);
          mma_f16_16816((i & 1) ? sacc2[nb] : sacc[nb], qa[i], bf);
        }
      }
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        sacc[nb][0] += sacc2[nb][0];
        sacc[nb][1] += sacc2[nb][1];
        sacc[nb][2] += sacc2[nb][2];
        sacc[nb][3] += sacc2[nb][3];
      }
      bar_arrive(B_KV_FREE0 + b, N_KV_FREE);  // done reading K slot b
      float* Sq = S + (b * 4 + kq) * H * SP;
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) {
        int key = nb * 8 + 2 * c;
        *reinterpret_cast<float2*>(&Sq[g * SP + key]) = make_float2(sacc[nb][0], sacc[nb][1]);
        *reinterpret_cast<float2*>(&Sq[(g + 8) * SP + key]) = make_float2(sacc[nb][2], sacc[nb][3]);
      }
      bar_sync(B_QK_SYNC, N_QK);  // all 4 partial S written
      {                           // online softmax over the stage's 32 keys
        const float* Sb = S + b * 4 * H * SP;
        float sv[4];
        float mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float v = Sb[hs * SP + k0 + k] + Sb[1 * H * SP + hs * SP + k0 + k] +
                    Sb[2 * H * SP + hs * SP + k0 + k] + Sb[3 * H * SP + hs * SP + k0 + k];
          sv[k] = (ib[k0 + k] >= 0) ? v * sm_scale_log2 : -INFINITY;
          mx = fmaxf(mx, sv[k]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        float m_new = fmaxf(m_run, mx), m_safe = (m_new == -INFINITY) ? 0.f : m_new;
        float alpha = (m_run == -INFINITY) ? 0.f : exp2f(m_run - m_safe), sum = 0.f;
        __half pv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          float pk = (sv[k] == -INFINITY) ? 0.f : exp2f(sv[k] - m_safe);
          sum += pk;
          pv[k] = __float2half(pk);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        l_run = l_run * alpha + sum;
        m_run = m_new;
        *reinterpret_cast<uint2*>(&P[(pb * H + hs) * PP + k0]) =
            make_uint2(*reinterpret_cast<uint32_t*>(&pv[0]), *reinterpret_cast<uint32_t*>(&pv[2]));
        if ((lane & 7) == 0) alpha_sh[pb * H + hs] = alpha;
      }
      bar_sync(B_QK_SYNC,
               N_QK);  // S slot b consumed before the next QK overwrites... (next use is s+2)
      bar_arrive(B_P_FULL0 + pb, N_P);
    }
    // publish m, l of head hs to every owner (lanes with (lane & 7) == 0)
    if ((lane & 7) == 0) {
#pragma unroll
      for (int r = 0; r < C; ++r)
        *reinterpret_cast<float2*>(cluster.map_shared_rank(recv_ml, r) + (rank * H + hs) * 2) =
            make_float2(m_run, l_run);
    }
  } else {
    // ================= PV role: warp -> dims [128*(warp-12), +128) = 16 n-blocks
    const int pw = warp - NDEQ - NQK;
    float oacc[16][4];
#pragma unroll
    for (int nb = 0; nb < 16; ++nb) {
      oacc[nb][0] = oacc[nb][1] = oacc[nb][2] = oacc[nb][3] = 0.f;
    }
    for (int s = 0; s < nstage; ++s) {
      const int b = s % NSLOT;
      const int pb = s % PSLOT;
      bar_sync(B_KVP0 + b, N_KVP);
      bar_sync(B_P_FULL0 + pb, N_P);
      const __half* kb = Kf + b * SUB * KP;
      float a0 = alpha_sh[pb * H + g], a1 = alpha_sh[pb * H + g + 8];
#pragma unroll
      for (int nb = 0; nb < 16; ++nb) {
        oacc[nb][0] *= a0;
        oacc[nb][1] *= a0;
        oacc[nb][2] *= a1;
        oacc[nb][3] *= a1;
      }
#pragma unroll
      for (int ks = 0; ks < SUB / 16; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, P + (pb * H + (lane & 15)) * PP + ks * 16 + ((lane >> 4) << 3));
        const __half* vrow = kb + (ks * 16 + (lane & 15)) * KP + pw * 128;
#pragma unroll
        for (int nb = 0; nb < 16; ++nb) {
          uint32_t bf[2];
          ldmatrix_x2_trans(bf, vrow + nb * 8);
          mma_f16_16816(oacc[nb], a, bf);
        }
      }
      bar_arrive(B_KV_FREE0 + b, N_KV_FREE);
      bar_arrive(B_P_FREE0 + pb, N_P);
    }
// push my O slices as f16: dim block j = 16 pw + nb -> owner j % C, slot j / C; [C][BPC][H][8]: 128
// B per warp store
#pragma unroll
    for (int nb = 0; nb < 16; ++nb) {
      const int j = 16 * pw + nb, owner = j % C, slot = j / C;
      __half* dst = cluster.map_shared_rank(recv_o, owner) + (rank * BPC + slot) * SLOTH + 2 * c;
      *reinterpret_cast<__half2*>(dst + g * 8) = __floats2half2_rn(oacc[nb][0], oacc[nb][1]);
      *reinterpret_cast<__half2*>(dst + (g + 8) * 8) = __floats2half2_rn(oacc[nb][2], oacc[nb][3]);
    }
  }
  cluster.sync();
  {  // merge the dim blocks this CTA owns (j = rank + C*slot): warp = head; lanes 0..C-1 compute
     // the source weights, then lanes stride over values
    const int h = warp;
    float mr = -INFINITY, lr = 0.f;
    if (lane < C) {
      float2 ml = *reinterpret_cast<const float2*>(recv_ml + (lane * H + h) * 2);
      mr = ml.x;
      lr = ml.y;
    }
    float M = mr;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    M = __shfl_sync(0xffffffff, M, 0);
    const float Ms = (M == -INFINITY) ? 0.f : M;
    const float wl = (lane < C && mr != -INFINITY) ? exp2f(mr - Ms) : 0.f;
    float Lp = wl * lr;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) Lp += __shfl_xor_sync(0xffffffff, Lp, o);
    const float L = __shfl_sync(0xffffffff, Lp, 0), inv = (L > 0.f) ? o_scale / L : 0.f;
    float wr[MAXC];
#pragma unroll
    for (int r = 0; r < MAXC; ++r) wr[r] = __shfl_sync(0xffffffff, wl, r);
    for (int v = lane; v < BPC * 8; v += 32) {
      const int slot = v >> 3, d = v & 7, j = rank + C * slot;
      if (j < DV / 8) {
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < MAXC; ++r)
          if (r < C) acc += wr[r] * __half2float(recv_o[(r * BPC + slot) * SLOTH + h * 8 + d]);
        out[((size_t)t * H + h) * DV + 8 * j + d] = __float2bfloat16(acc * inv);
      }
    }
  }
}

// A launch is valid when every CTA of the cluster gets between NSLOT and MAXKEYS / SUB stages of
// keys.
inline bool is_valid_config(int topk_width, int num_ctas) {
  if (topk_width <= 0 || topk_width % SUB != 0 || num_ctas < 2 || num_ctas > MAXC) return false;
  const int stages = topk_width / SUB;
  return (stages + num_ctas - 1) / num_ctas <= MAXKEYS / SUB && stages / num_ctas >= NSLOT;
}

// Opt the kernel into its dynamic shared memory once per device (cudaFuncSetAttribute is per
// device).
inline cudaError_t ensure_kernel_attributes() {
  constexpr int kMaxDevices = 64;
  static int status[kMaxDevices] =
      {};  // 0: not set yet, 1: set, other: 2 + the cudaError_t it failed with
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidDevice;
  if (status[device] == 0) {
    e = cudaFuncSetAttribute(nvfp4_sparse_mla_decode_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
    status[device] = (e == cudaSuccess) ? 1 : 2 + static_cast<int>(e);
  }
  return status[device] == 1 ? cudaSuccess : static_cast<cudaError_t>(status[device] - 2);
}

inline cudaError_t launch(const uint8_t* kv, const uint8_t* q, const int32_t* indices,
                          __nv_bfloat16* out, int num_tokens, int topk_width, float sm_scale_log2,
                          float output_scale, int num_ctas, cudaStream_t stream) {
  if (num_tokens <= 0) return cudaSuccess;
  if (!is_valid_config(topk_width, num_ctas)) return cudaErrorInvalidValue;
  cudaError_t e = ensure_kernel_attributes();
  if (e != cudaSuccess) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(num_tokens * num_ctas);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SM_TOTAL;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = num_ctas;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, nvfp4_sparse_mla_decode_kernel, kv, q, indices, out, topk_width,
                            sm_scale_log2, output_scale);
}

// How many clusters of `num_ctas` CTAs fit on the current device at once: a launch of up to that
// many query tokens runs in one wave.
inline cudaError_t max_active_clusters(int num_ctas, int* count) {
  *count = 0;
  if (num_ctas < 1 || num_ctas > MAXC) return cudaErrorInvalidValue;
  cudaError_t e = ensure_kernel_attributes();
  if (e != cudaSuccess) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(num_ctas * 64);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SM_TOTAL;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = num_ctas;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaOccupancyMaxActiveClusters(
      count, reinterpret_cast<void*>(nvfp4_sparse_mla_decode_kernel), &cfg);
}

}  // namespace nvfp4_sparse_mla_decode
}  // namespace flashinfer
