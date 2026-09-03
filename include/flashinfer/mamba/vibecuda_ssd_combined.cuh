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
 *
 * VibeCUDA Mamba2/SSD combined selective scan (forward).
 *
 * Hand-written mma.sync m16n8k16 (bf16/fp16 inputs, fp32 accumulation)
 * chunked-scan pipeline for the Mamba2 SSD operator at chunk_size=128,
 * headdim=64, dstate=128:
 *
 *   k_segstate<DtT,IdxT,StateT> : per (segment, head): fused chunk-state
 *                        accumulation (decay-weighted x^T b, bf16 MMA with
 *                        fp32 accumulators kept in registers) + sequential
 *                        inter-chunk state passing + state_in/final_states
 *                        stores.  Blocks return early on single-chunk
 *                        segments, which k_out closes itself.
 *   k_out<DtT,IdxT,StateT>    : per (logical chunk, head): masked-decay
 *                        C.B^T, M.X, C.state, D-skip, optional SiLU z-gate,
 *                        and the fused final-state MMA for single-chunk
 *                        segments.
 *
 * Single-chunk segments are closed entirely inside k_out, so for the
 * ubiquitous seqlen<=128 layouts the pipeline is exactly one launch.
 *
 * There is no separate preprocess kernel: segment bounds and the
 * dt->delta->cumsum(dA) chain are recomputed per block from (seq_idx or the
 * B*L layout) with a block-wide scan.  Logical chunk = (segment) intersect
 * (physical 128-token chunk).  All decay math is fp32; intra-chunk M.X is
 * fp16 MMA, everything else bf16 MMA.
 *
 * Outputs use the FlashInfer SSDCombined layouts:
 *   out           : (batch, nheads, headdim, nchunks, 128), io dtype
 *   final_states  : (state_batch, nheads, headdim, dstate), state dtype
 * state_in scratch: (nLCmax, nheads, headdim, dstate), bf16 (caller-owned).
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <mutex>
#include <type_traits>

namespace flashinfer {
namespace mamba {
namespace vibecuda {

#define DEVI __device__ __forceinline__

using bf16 = __nv_bfloat16;
using fp16 = half;

constexpr int CHUNK = 128;
constexpr int HDIM = 64;
constexpr int DSTATE = 128;
constexpr int STRIDE_BC = 128 + 8;  // halves, padded
constexpr int STRIDE_X = 64 + 8;

template <typename T>
DEVI float to_f32(T v);
template <>
DEVI float to_f32<float>(float v) {
  return v;
}
template <>
DEVI float to_f32<bf16>(bf16 v) {
  return __bfloat162float(v);
}
template <>
DEVI float to_f32<fp16>(fp16 v) {
  return __half2float(v);
}

// ---------------------------------------------------------------------------
// ldmatrix / mma helpers
// ---------------------------------------------------------------------------
DEVI uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVI void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
DEVI void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
DEVI void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
DEVI void ldsm_x2_t(uint32_t& r0, uint32_t& r1, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}

// cp.async helpers: async global->shared copies (LDGSTS). pred=false zfills.
DEVI void cp_async_16(void* dst, const void* src, bool pred) {
  uint32_t d = smem_addr(dst);
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
DEVI void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
DEVI void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n"); }

DEVI void mma_bf16(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1,
                   uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEVI void mma_f16(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1, uint32_t a2,
                  uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Address patterns (lane-dependent), strides in halves (2B), results in bytes.
DEVI uint32_t addr_A_nt(const void* base, int stride, int m0, int k0, int lane) {
  // SMEM stored [m][k]; A fragment m16k16.
  int row = m0 + (lane & 15);
  int col = k0 + (lane >> 4) * 8;
  return smem_addr(reinterpret_cast<const char*>(base) + ((int64_t)row * stride + col) * 2);
}
DEVI uint32_t addr_A_t(const void* base, int stride, int k0, int m0, int lane) {
  // SMEM stored [k][m]; A fragment m16k16 via ldmatrix.trans.
  int row = k0 + (lane & 7) + ((lane >> 4) * 8);
  int col = m0 + (((lane & 15) >> 3) * 8);
  return smem_addr(reinterpret_cast<const char*>(base) + ((int64_t)row * stride + col) * 2);
}
DEVI uint32_t addr_B_nt(const void* base, int stride, int n0, int k0, int lane) {
  // SMEM stored [n][k]; B fragment k16n8 (x2).
  int row = n0 + (lane & 7);
  int col = k0 + (((lane & 15) >> 3) * 8);
  return smem_addr(reinterpret_cast<const char*>(base) + ((int64_t)row * stride + col) * 2);
}
DEVI uint32_t addr_B_t(const void* base, int stride, int k0, int n0, int lane) {
  // SMEM stored [k][n]; B fragment k16n8 via ldmatrix.trans (x2).
  int row = k0 + (lane & 15);
  int col = n0;
  return smem_addr(reinterpret_cast<const char*>(base) + ((int64_t)row * stride + col) * 2);
}

// ---------------------------------------------------------------------------
// dtype pair conversion helpers
// ---------------------------------------------------------------------------
template <typename StateT>
DEVI float2 st_to_float2(uint32_t u);
template <>
DEVI float2 st_to_float2<bf16>(uint32_t u) {
  __nv_bfloat162 h2;
  h2.x = __ushort_as_bfloat16(u & 0xffffu);
  h2.y = __ushort_as_bfloat16(u >> 16);
  return __bfloat1622float2(h2);
}
template <>
DEVI float2 st_to_float2<fp16>(uint32_t u) {
  __half2 h2;
  h2.x = __ushort_as_half(u & 0xffffu);
  h2.y = __ushort_as_half(u >> 16);
  return __half22float2(h2);
}
template <typename StateT>
DEVI uint32_t float2_to_st(float2 v);
template <>
DEVI uint32_t float2_to_st<bf16>(float2 v) {
  __nv_bfloat162 h2 = __floats2bfloat162_rn(v.x, v.y);
  return (__bfloat16_as_ushort(h2.y) << 16) | __bfloat16_as_ushort(h2.x);
}
template <>
DEVI uint32_t float2_to_st<fp16>(float2 v) {
  __half2 h2 = __floats2half2_rn(v.x, v.y);
  return (__half_as_ushort(h2.y) << 16) | __half_as_ushort(h2.x);
}

// ---------------------------------------------------------------------------
// On-the-fly segment metadata (segment tables are derived from seq_idx or the
// B*L batched layout instead of host-side preprocessing)
// ---------------------------------------------------------------------------
template <typename IdxT>
DEVI int lb_seqidx(const IdxT* __restrict__ sid, int NT, int64_t v) {
  // lower_bound of v in non-decreasing sid[0..NT)
  int lo = 0, hi = NT;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (static_cast<int64_t>(sid[mid]) < v)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

template <typename IdxT>
DEVI void seg_bounds(const IdxT* __restrict__ sid, int NT, int L, bool varlen, int s, int& b0,
                     int& b1) {
  if (varlen) {
    b0 = lb_seqidx(sid, NT, s);
    b1 = lb_seqidx(sid, NT, (int64_t)s + 1);
  } else {
    b0 = s * L;
    b1 = b0 + L;
  }
}

DEVI int seg_chunk_count(int b0, int b1) { return ((b1 - 1) >> 7) - (b0 >> 7) + 1; }

// Resolve logical chunk lc -> (seg, t0, t1); returns total logical chunks.
template <typename IdxT>
DEVI int resolve_lc(const IdxT* __restrict__ sid, int NT, int L, int nseg, bool varlen, int lc,
                    int& seg, int& t0, int& t1) {
  int acc = 0;
  seg = -1;
  for (int s = 0; s < nseg; ++s) {
    int b0, b1;
    seg_bounds(sid, NT, L, varlen, s, b0, b1);
    int cnt = seg_chunk_count(b0, b1);
    if (seg < 0 && lc < acc + cnt) {
      int k = lc - acc;
      int pchi = ((b0 >> 7) + k) << 7;
      t0 = b0 > pchi ? b0 : pchi;
      t1 = b1 < pchi + CHUNK ? b1 : pchi + CHUNK;
      seg = s;
    }
    acc += cnt;
  }
  return acc;
}

// ---------------------------------------------------------------------------
// Block-wide inclusive scan (blockDim.x must be a multiple of 32, <= 1024)
// warp_sums: shared float array with >= blockDim.x/32 + a few cells.
// Contains barriers; must be called uniformly by all threads.
// ---------------------------------------------------------------------------
DEVI float block_incl_scan(float v, float* warp_sums, int tid, int nthreads, float& total) {
  int lane = tid & 31, warp = tid >> 5, nwarp = nthreads >> 5;
  float scan = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    float u = __shfl_up_sync(0xffffffffu, scan, o);
    if (lane >= o) scan += u;
  }
  __syncthreads();  // protect warp_sums reuse from previous call
  if (lane == 31) warp_sums[warp] = scan;
  __syncthreads();
  if (warp == 0) {
    float w = (lane < nwarp) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float u = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += u;
    }
    if (lane < nwarp) warp_sums[lane] = w;
  }
  __syncthreads();
  float add = (warp > 0) ? warp_sums[warp - 1] : 0.f;
  total = warp_sums[nwarp - 1];
  return scan + add;
}

// Compute delta = clamp(dt+d_bias or softplus(dt+d_bias), lo, hi) and dA
// (segment-inclusive cumsum of delta*a) for tokens [t0,t1) of segment [s0,s1);
// fills sdel/sdA (zero-filled to CHUNK), and writes entry (=dA[t0-1], 0 if
// t0==s0) and end (=dA[t1-1]) into s_meta[0]/s_meta[1]. s_meta is 2 shared
// floats + warp_sums scratch. With dt_lo<=0 and dt_hi=+inf the clamp is a
// no-op on the non-negative softplus range.
template <typename DtT>
DEVI void scan_chunk_deltas(const DtT* __restrict__ dt, const DtT* __restrict__ dt_bias,
                            const float* __restrict__ a, int softplus, float dt_lo, float dt_hi,
                            int H, int h, int s0, int s1, int t0, int t1, float* __restrict__ sdel,
                            float* __restrict__ sdA, float* s_meta, float* warp_sums, int tid,
                            int nthreads) {
  const float bias = (dt_bias != nullptr) ? to_f32<DtT>(dt_bias[h]) : 0.f;
  const float av = a[h];
  float carry = 0.f;
  for (int base = s0; base < s1; base += nthreads) {
    int g = base + tid;
    bool valid = g < s1;
    float delta = 0.f, da = 0.f;
    if (valid) {
      float v = to_f32<DtT>(dt[(size_t)g * H + h]) + bias;
      float sp = softplus ? ((v > 20.f) ? v : log1pf(__expf(v))) : v;
      delta = fminf(fmaxf(sp, dt_lo), dt_hi);
      da = delta * av;
    }
    float total;
    float inc = block_incl_scan(da, warp_sums, tid, nthreads, total);
    if (valid) {
      if (g >= t0 && g < t1) {
        sdel[g - t0] = delta;
        sdA[g - t0] = carry + inc;
      }
      if (g == t0 - 1) s_meta[0] = carry + inc;
      if (g == t1 - 1) s_meta[1] = carry + inc;
    }
    carry += total;
  }
  __syncthreads();
  int len = t1 - t0;
  for (int i = tid + len; i < CHUNK; i += nthreads) {
    sdel[i] = 0.f;
    sdA[i] = 0.f;
  }
  if (t0 == s0 && tid == 0) s_meta[0] = 0.f;
  __syncthreads();
}

// ---------------------------------------------------------------------------
// k_segstate: fused chunk-state accumulation + inter-chunk state passing, per
// (segment, head).  Chunk summaries (decay-weighted x^T b, mma bf16 fp32-acc)
// stay in registers, the sequential state recurrence runs on the same
// fragment layout, state_in entries are written bf16 per logical chunk, and
// final_states comes out in StateT.  Single-chunk segments return early —
// k_out closes those via its fused state MMA.
// Two cp.async slots hold (x rows | raw b rows) per chunk; b rows are
// decay-scaled in place after the per-chunk delta/dA scan.
// 256 threads; warp w owns rows [16*(w>>1),+16) x dstate-half (w&1).
// ---------------------------------------------------------------------------
constexpr int KS_SMEM = 2 * (CHUNK * STRIDE_X + CHUNK * STRIDE_BC) * 2;
constexpr int KS_SLOT = (CHUNK * STRIDE_X + CHUNK * STRIDE_BC);  // halves

template <typename DtT, typename IdxT, typename StateT>
__global__ __launch_bounds__(256) void k_segstate(
    const bf16* __restrict__ x, const bf16* __restrict__ bmat, const DtT* __restrict__ dt,
    const DtT* __restrict__ dt_bias, const float* __restrict__ a, int softplus, float dt_lo,
    float dt_hi, const IdxT* __restrict__ seq_idx, int NT, int L, int H, int hpg, int G, int nseg,
    int varlen, const StateT* __restrict__ initial, bf16* __restrict__ state_in,
    StateT* __restrict__ final_states) {
  __shared__ float sdA[CHUNK], sdel[CHUNK], s_meta[2], warp_sums[8];
  extern __shared__ bf16 smem[];
  int seg = blockIdx.x, h = blockIdx.y;
  int tid = threadIdx.x;
  int s0, s1;
  seg_bounds(seq_idx, NT, L, varlen != 0, seg, s0, s1);
  const int pc0 = s0 >> 7, pc1 = (s1 - 1) >> 7;
  const int cnt = pc1 - pc0 + 1;
  if (cnt == 1) return;  // single-chunk segments handled inside k_out

  // global index of this segment's first logical chunk
  int lc_base = 0;
  for (int s = 0; s < seg; ++s) {
    int q0, q1;
    seg_bounds(seq_idx, NT, L, varlen != 0, s, q0, q1);
    lc_base += seg_chunk_count(q0, q1);
  }
  const int grp = h / hpg;
  const int warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const int m0 = (warp >> 1) * 16, nh = warp & 1;

  // issue chunk k's x rows and raw b rows into slot (k&1), zfill past len; an
  // empty commit when k >= cnt keeps the wait_group accounting uniform.
  auto issue_slot = [&](int k) {
    if (k < cnt) {
      int pch = (pc0 + k) << 7;
      int t0 = s0 > pch ? s0 : pch;
      int t1 = s1 < pch + CHUNK ? s1 : pch + CHUNK;
      int len = t1 - t0;
      bf16* xs = smem + (k & 1) * KS_SLOT;
      bf16* sb = xs + CHUNK * STRIDE_X;
#pragma unroll
      for (int idx = tid; idx < CHUNK * 8; idx += 256) {
        int i = idx >> 3, c8 = idx & 7;
        bool v = i < len;
        const char* src =
            reinterpret_cast<const char*>(x + ((size_t)(t0 + (v ? i : 0)) * H + h) * HDIM) +
            c8 * 16;
        cp_async_16(xs + i * STRIDE_X + c8 * 8, src, v);
      }
#pragma unroll
      for (int idx = tid; idx < CHUNK * 16; idx += 256) {
        int i = idx >> 4, c8 = idx & 15;
        bool v = i < len;
        const char* src =
            reinterpret_cast<const char*>(bmat + ((size_t)(t0 + (v ? i : 0)) * G + grp) * DSTATE) +
            c8 * 16;
        cp_async_16(sb + i * STRIDE_BC + c8 * 8, src, v);
      }
    }
    cp_async_commit();
  };

  issue_slot(0);
  issue_slot(1);

  // state fragment: cur[j][0] = rows m0+gid, cur[j][1] = rows m0+gid+8;
  // cols n = nh*64 + j*8 + tig*2 (+1 in .y).  Seeded from initial[seg]; zero
  // seed when no initial states are provided.
  float2 cur[8][2];
  if (initial != nullptr) {
    const uint32_t* init32 =
        reinterpret_cast<const uint32_t*>(initial) + (size_t)(seg * H + h) * (HDIM * DSTATE / 2);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int n = nh * 64 + j * 8 + tig * 2;
      cur[j][0] = st_to_float2<StateT>(init32[((m0 + gid) * DSTATE + n) >> 1]);
      cur[j][1] = st_to_float2<StateT>(init32[((m0 + gid + 8) * DSTATE + n) >> 1]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) cur[j][0] = cur[j][1] = make_float2(0.f, 0.f);
  }

  for (int k = 0; k < cnt; ++k) {
    const int slot = k & 1;
    bf16* xs = smem + slot * KS_SLOT;
    bf16* sb = xs + CHUNK * STRIDE_X;
    // state entering chunk k -> state_in[lc] (bf16)
    {
      uint32_t* si = reinterpret_cast<uint32_t*>(state_in) +
                     ((size_t)(lc_base + k) * H + h) * (HDIM * DSTATE / 2);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int n = nh * 64 + j * 8 + tig * 2;
        si[((m0 + gid) * DSTATE + n) >> 1] = float2_to_st<bf16>(cur[j][0]);
        si[((m0 + gid + 8) * DSTATE + n) >> 1] = float2_to_st<bf16>(cur[j][1]);
      }
    }
    // per-chunk delta/dA scan (barriers overlap the in-flight cp.async groups)
    const int pch = (pc0 + k) << 7;
    const int t0 = s0 > pch ? s0 : pch;
    const int t1 = s1 < pch + CHUNK ? s1 : pch + CHUNK;
    const int len = t1 - t0;
    scan_chunk_deltas(dt, dt_bias, a, softplus, dt_lo, dt_hi, H, h, s0, s1, t0, t1, sdel, sdA,
                      s_meta, warp_sums, tid, 256);
    // groups k and k+1 outstanding -> wait until <=1 (slot k complete)
    cp_async_wait1();
    __syncthreads();
    // decay-scale b rows in place: w_i = delta_i * exp(dA_end - dA_i)
    const float s_end = s_meta[1];
    if (tid < CHUNK) {
      float w = (tid < len) ? (sdel[tid] * __expf(s_end - sdA[tid])) : 0.f;
      __nv_bfloat162 w2 = __float2bfloat162_rn(w);
      uint32_t* row32 = reinterpret_cast<uint32_t*>(sb + tid * STRIDE_BC);
#pragma unroll
      for (int j = 0; j < 68; ++j) {
        __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(row32 + j);
        *reinterpret_cast<__nv_bfloat162*>(row32 + j) = __hmul2(u, w2);
      }
    }
    __syncthreads();
    // chunk summary MMA: acc[d][n] = sum_t x[t,d] * wb[t,n]
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      uint32_t a0, a1, a2, a3;
      ldsm_x4_t(a0, a1, a2, a3, addr_A_t(xs, STRIDE_X, kt * 16, m0, lane));
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t b0, b1;
        ldsm_x2_t(b0, b1, addr_B_t(sb, STRIDE_BC, kt * 16, nh * 64 + j * 8, lane));
        mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], a0, a1, a2, a3, b0, b1);
      }
    }
    // state recurrence: cur = cur * exp(chunk cumsum) + chunk summary
    const float decay = __expf(s_meta[1] - s_meta[0]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      cur[j][0].x = fmaf(cur[j][0].x, decay, acc[j][0]);
      cur[j][0].y = fmaf(cur[j][0].y, decay, acc[j][1]);
      cur[j][1].x = fmaf(cur[j][1].x, decay, acc[j][2]);
      cur[j][1].y = fmaf(cur[j][1].y, decay, acc[j][3]);
    }
    __syncthreads();  // all ldmatrix reads done before slot reuse
    issue_slot(k + 2);
  }

  uint32_t* fs =
      reinterpret_cast<uint32_t*>(final_states) + (size_t)(seg * H + h) * (HDIM * DSTATE / 2);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int n = nh * 64 + j * 8 + tig * 2;
    fs[((m0 + gid) * DSTATE + n) >> 1] = float2_to_st<StateT>(cur[j][0]);
    fs[((m0 + gid + 8) * DSTATE + n) >> 1] = float2_to_st<StateT>(cur[j][1]);
  }
}

// ---------------------------------------------------------------------------
// k_out: per (logical chunk, head): y for each token in the chunk part.
//   M[t][s] = (c_t . b_s) * exp(dA_t - dA_s) * delta_s * [s <= t]     (fp16 MMA)
//   y_intra = M @ x
//   y_inter[t][d] = exp(dA_t - dA_entry) * (c_t . state_entry[d])
//   y = y_intra + y_inter + D (*) x ; optional y *= z*sigmoid(z)
// Single-chunk segments additionally read `initial` as the entry state and
// write final_states from the completed chunk state (no second kernel).
// The output tile is staged through shared memory so global stores land in
// the (batch, nheads, headdim, nchunks, 128) FlashInfer layout with 16B
// alignment.
// 512 threads = 16 warps; warp w covers rows [16*(w>>1),+16) x N-half (w&1).
// ---------------------------------------------------------------------------
constexpr int K3_SMEM =
    (2 * CHUNK * STRIDE_X + 3 * CHUNK * STRIDE_BC + HDIM * STRIDE_BC) * 2 + (2 * CHUNK + 20) * 4;

template <typename DtT, typename IdxT, typename StateT>
__global__ __launch_bounds__(512) void k_out(
    const bf16* __restrict__ x, const bf16* __restrict__ bmat, const bf16* __restrict__ cmat,
    const bf16* __restrict__ dmat, const bf16* __restrict__ z_bf, const fp16* __restrict__ z_hp,
    int z_is_f16, const DtT* __restrict__ dt, const DtT* __restrict__ dt_bias,
    const float* __restrict__ a, int softplus, float dt_lo, float dt_hi,
    const IdxT* __restrict__ seq_idx, int NT, int L, int H, int hpg, int G, int nseg, int varlen,
    int d_has_hdim, int has_z, const bf16* __restrict__ state_in,
    const StateT* __restrict__ initial, StateT* __restrict__ final_states, bf16* __restrict__ out,
    int64_t out_head_stride, int64_t out_row_stride) {
  // out[b, h, d, c, l] contiguous: batch stride = H * out_head_stride,
  // head stride = out_head_stride (= HDIM * nchunks * CHUNK), d-plane row
  // stride = out_row_stride (= nchunks * CHUNK).  A logical chunk never spans
  // batch rows, so the whole block writes into one batch plane.
  extern __shared__ char smem_raw[];
  int lc = blockIdx.x, h = blockIdx.y;
  int tid = threadIdx.x;
  int seg, t0, t1;
  int nlc = resolve_lc(seq_idx, NT, L, nseg, varlen != 0, lc, seg, t0, t1);
  if (lc >= nlc) return;
  int len = t1 - t0;
  // Single-chunk segment: this lc spans the whole segment, so the entry state
  // is `initial[seg]` (or zero) and final_states[seg] = initial*decay +
  // chunk_state[lc].
  int seg_prev = -1, seg_next = -1;
  {
    int s_, q0_, q1_;
    if (lc > 0) {
      resolve_lc(seq_idx, NT, L, nseg, varlen != 0, lc - 1, s_, q0_, q1_);
      seg_prev = s_;
    }
    if (lc + 1 < nlc) {
      resolve_lc(seq_idx, NT, L, nseg, varlen != 0, lc + 1, s_, q0_, q1_);
      seg_next = s_;
    }
  }
  const bool single = (seg_prev != seg) && (seg_next != seg);

  bf16* sbf = reinterpret_cast<bf16*>(smem_raw);                    // [128][136]
  bf16* sc = sbf + CHUNK * STRIDE_BC;                               // [128][136]
  bf16* sst = sc + CHUNK * STRIDE_BC;                               // [64][136]
  bf16* sxb = sst + HDIM * STRIDE_BC;                               // [128][72] raw x
  bf16* swb = sxb + CHUNK * STRIDE_X;                               // [128][136] w-scaled b
  fp16* sxhp = reinterpret_cast<fp16*>(swb + CHUNK * STRIDE_BC);    // [128][72]
  float* sdel = reinterpret_cast<float*>(sxhp + CHUNK * STRIDE_X);  // [128]
  float* sdA = sdel + CHUNK;                                        // [128]
  float* s_meta = sdA + CHUNK;                                      // [2]
  float* sShr = s_meta + 2;                                         // [2]
  float* warp_sums = sShr + 2;                                      // [16]

  int s0, s1;
  seg_bounds(seq_idx, NT, L, varlen != 0, seg, s0, s1);
  int grp = h / hpg;
  // b rows -> sbf, c rows -> sc (async 16B chunks, zfill past len)
#pragma unroll
  for (int idx = tid; idx < CHUNK * 16; idx += 512) {
    int i = idx >> 4, k = idx & 15;
    bool v = i < len;
    const char* bs =
        reinterpret_cast<const char*>(bmat + ((size_t)(t0 + (v ? i : 0)) * G + grp) * DSTATE) +
        k * 16;
    const char* cs_ =
        reinterpret_cast<const char*>(cmat + ((size_t)(t0 + (v ? i : 0)) * G + grp) * DSTATE) +
        k * 16;
    cp_async_16(sbf + i * STRIDE_BC + k * 8, bs, v);
    cp_async_16(sc + i * STRIDE_BC + k * 8, cs_, v);
  }
  // x rows -> sxb raw bf16 (async; converted to fp16 sxhp after the wait)
#pragma unroll
  for (int idx = tid; idx < CHUNK * 8; idx += 512) {
    int i = idx >> 3, k = idx & 7;
    bool v = i < len;
    const char* xs =
        reinterpret_cast<const char*>(x + ((size_t)(t0 + (v ? i : 0)) * H + h) * HDIM) + k * 16;
    cp_async_16(sxb + i * STRIDE_X + k * 8, xs, v);
  }
  // entry state rows -> sst (async raw copy; StateT and bf16 are both 16-bit)
  if (single) {
#pragma unroll
    for (int idx = tid; idx < HDIM * 16; idx += 512) {
      int i = idx >> 4, k = idx & 15;
      const char* src = reinterpret_cast<const char*>(
          initial + (size_t)(seg * H + h) * (HDIM * DSTATE) + i * DSTATE);
      cp_async_16(sst + i * STRIDE_BC + k * 8, src + k * 16, initial != nullptr);
    }
  } else {
#pragma unroll
    for (int idx = tid; idx < HDIM * 16; idx += 512) {
      int i = idx >> 4, k = idx & 15;
      const char* src = reinterpret_cast<const char*>(
          state_in + ((size_t)(lc * H + h)) * (HDIM * DSTATE) + i * DSTATE);
      cp_async_16(sst + i * STRIDE_BC + k * 8, src + k * 16, true);
    }
  }
  cp_async_commit();
  // block-wide softplus/cumsum (own barriers overlap the async loads above)
  scan_chunk_deltas(dt, dt_bias, a, softplus, dt_lo, dt_hi, H, h, s0, s1, t0, t1, sdel, sdA, s_meta,
                    warp_sums, tid, 512);
  cp_async_wait_all();
  __syncthreads();
  // smem->smem work: x bf16 -> fp16 (sxhp); fp16 initial -> bf16 (in place);
  // single-chunk segments additionally build the decay-scaled b tile (swb) that
  // feeds the fused final-state MMA, replacing the global state round trip.
  {
    int row = tid >> 2, part = tid & 3;  // 4 threads per x row, 16 elems each
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(sxb + row * STRIDE_X + part * 16);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(sxhp + row * STRIDE_X + part * 16);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint32_t u = src32[k];
      fp16 lo = __float2half_rn(__bfloat162float(__ushort_as_bfloat16(u & 0xffffu)));
      fp16 hi = __float2half_rn(__bfloat162float(__ushort_as_bfloat16(u >> 16)));
      dst32[k] = (__half_as_ushort(hi) << 16) | __half_as_ushort(lo);
    }
  }
  if (single && initial != nullptr && std::is_same_v<StateT, fp16>) {
    int row = tid >> 3, part = tid & 7;  // 8 threads per state row, 16 elems each
    uint32_t* p32 = reinterpret_cast<uint32_t*>(sst + row * STRIDE_BC + part * 16);
#pragma unroll
    for (int k = 0; k < 8; ++k) p32[k] = float2_to_st<bf16>(st_to_float2<fp16>(p32[k]));
  }
  if (single) {
    // swb[t][s] = b[t][s] * w_t, w_t = delta_t * exp(dA_end - dA_t) (bf16
    // scale); dead rows have delta=0 -> zeros.
    int row = tid >> 2, part = tid & 3;  // 4 threads per b row, 32 elems each
    float w = sdel[row] * __expf(s_meta[1] - sdA[row]);
    __nv_bfloat162 w2 = __float2bfloat162_rn(w);
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(sbf + row * STRIDE_BC + part * 32);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(swb + row * STRIDE_BC + part * 32);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      __nv_bfloat162 u = *reinterpret_cast<const __nv_bfloat162*>(src32 + k);
      *reinterpret_cast<__nv_bfloat162*>(dst32 + k) = __hmul2(u, w2);
    }
  }
  __syncthreads();
  if (tid == 0) {
    sShr[0] = s_meta[0];                                                    // entry cumsum
    sShr[1] = (dmat != nullptr) ? __bfloat162float(dmat[(size_t)h]) : 0.f;  // !d_has_hdim
  }
  __syncthreads();

  // 16 warps: warp w covers token rows [(w>>1)*16, +16) x N-half (w&1)
  int warp = tid >> 5, lane = tid & 31;
  int gid = lane >> 2, tig = lane & 3;
  int m0 = (warp >> 1) * 16, nh = warp & 1;

  // Fused final-state MMA for single-chunk segments: state[h][s] = sum_t
  // x[t,h] * swb[t,s], then final_states = initial * exp(dA_end) + state
  // (initial rows live in sst; zero when no initial states are provided).
  if (single) {
    int mg = warp >> 2, nq = (warp & 3) * 32;
    float accS[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) accS[j][0] = accS[j][1] = accS[j][2] = accS[j][3] = 0.f;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      uint32_t a0, a1, a2, a3;
      ldsm_x4_t(a0, a1, a2, a3, addr_A_t(sxb, STRIDE_X, kt * 16, mg * 16, lane));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t b0, b1;
        ldsm_x2_t(b0, b1, addr_B_t(swb, STRIDE_BC, kt * 16, nq + j * 8, lane));
        mma_bf16(accS[j][0], accS[j][1], accS[j][2], accS[j][3], a0, a1, a2, a3, b0, b1);
      }
    }
    float decay = __expf(s_meta[1]);
    uint32_t* fsb =
        reinterpret_cast<uint32_t*>(final_states) + (size_t)(seg * H + h) * (HDIM * DSTATE / 2);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int col = nq + j * 8 + tig * 2;
      int h_lo = mg * 16 + gid, h_hi = h_lo + 8;
      float2 i0 =
          st_to_float2<bf16>(*reinterpret_cast<const uint32_t*>(sst + h_lo * STRIDE_BC + col));
      float2 i1 =
          st_to_float2<bf16>(*reinterpret_cast<const uint32_t*>(sst + h_hi * STRIDE_BC + col));
      fsb[(h_lo * DSTATE + col) >> 1] = float2_to_st<StateT>(
          make_float2(fmaf(i0.x, decay, accS[j][0]), fmaf(i0.y, decay, accS[j][1])));
      fsb[(h_hi * DSTATE + col) >> 1] = float2_to_st<StateT>(
          make_float2(fmaf(i1.x, decay, accS[j][2]), fmaf(i1.y, decay, accS[j][3])));
    }
  }

  // Phase 1: M = C.B^T (accM, 64-col N-half), inter = C.state^T (accI, 32 cols)
  float accM[8][4], accI[4][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) accM[j][0] = accM[j][1] = accM[j][2] = accM[j][3] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) accI[j][0] = accI[j][1] = accI[j][2] = accI[j][3] = 0.f;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    uint32_t a0, a1, a2, a3;
    ldsm_x4(a0, a1, a2, a3, addr_A_nt(sc, STRIDE_BC, m0, kt * 16, lane));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t b0, b1;
      ldsm_x2(b0, b1, addr_B_nt(sbf, STRIDE_BC, nh * 64 + j * 8, kt * 16, lane));
      mma_bf16(accM[j][0], accM[j][1], accM[j][2], accM[j][3], a0, a1, a2, a3, b0, b1);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0, b1;
      ldsm_x2(b0, b1, addr_B_nt(sst, STRIDE_BC, nh * 32 + j * 8, kt * 16, lane));
      mma_bf16(accI[j][0], accI[j][1], accI[j][2], accI[j][3], a0, a1, a2, a3, b0, b1);
    }
  }

  // mask + decay-scale M, convert to fp16, store into the swb region
  // (raw b is dead after accM is formed; swb is likewise dead once the state
  // MMA consumed it)
  fp16* sM = reinterpret_cast<fp16*>(swb);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int s_col = nh * 64 + j * 8 + tig * 2;
    int t_lo = m0 + gid;
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
      int t = t_lo + half_row * 8;
      int i0 = half_row * 2;
      float e0 = 0.f, e1 = 0.f;
      if (s_col <= t) e0 = accM[j][i0] * __expf(sdA[t] - sdA[s_col]) * sdel[s_col];
      if (s_col + 1 <= t) e1 = accM[j][i0 + 1] * __expf(sdA[t] - sdA[s_col + 1]) * sdel[s_col + 1];
      fp16 h0 = __float2half_rn(e0), h1 = __float2half_rn(e1);
      reinterpret_cast<uint32_t*>(sM + t * STRIDE_BC)[(s_col) >> 1] =
          (__half_as_ushort(h1) << 16) | (__half_as_ushort(h0));
    }
  }
  __syncthreads();

  // scale inter accumulators by exp(dA_t - dA_entry)
  float entry = sShr[0];
  {
    int t_lo = m0 + gid;
    float e_lo = __expf(sdA[t_lo] - entry);
    float e_hi = __expf(sdA[t_lo + 8] - entry);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      accI[j][0] *= e_lo;
      accI[j][1] *= e_lo;
      accI[j][2] *= e_hi;
      accI[j][3] *= e_hi;
    }
  }

  // Phase 2: y_intra = M(f16) @ x(f16), 32-col N-half
  float accY[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) accY[j][0] = accY[j][1] = accY[j][2] = accY[j][3] = 0.f;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    uint32_t a0, a1, a2, a3;
    ldsm_x4(a0, a1, a2, a3, addr_A_nt(sM, STRIDE_BC, m0, kt * 16, lane));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t b0, b1;
      ldsm_x2_t(b0, b1, addr_B_t(sxhp, STRIDE_X, kt * 16, nh * 32 + j * 8, lane));
      mma_f16(accY[j][0], accY[j][1], accY[j][2], accY[j][3], a0, a1, a2, a3, b0, b1);
    }
  }

  // Epilogue: combine, D-skip, z-gate.  Finished (t, d) pairs are staged in a
  // [t][d] fp16 tile aliased onto the now-dead raw-x region (all MMA reads of
  // sxb completed before the barrier above; phase-2 reads sM/sxhp only), then
  // copied to global with 16B stores laid out as (b, h, d, c, l).
  bf16* sY = reinterpret_cast<bf16*>(sxb);  // [128][72] staged y pairs
  float dv_shared = sShr[1];
  int t_lo = m0 + gid;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int d_col = nh * 32 + j * 8 + tig * 2;
    float dv0, dv1;
    if (d_has_hdim) {
      dv0 = __bfloat162float(dmat[(size_t)h * HDIM + d_col]);
      dv1 = __bfloat162float(dmat[(size_t)h * HDIM + d_col + 1]);
    } else {
      dv0 = dv1 = dv_shared;
    }
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
      int t = t_lo + half_row * 8;
      int i0 = half_row * 2;
      float y0 = accY[j][i0] + accI[j][i0];
      float y1 = accY[j][i0 + 1] + accI[j][i0 + 1];
      if (dmat != nullptr) {
        float x0 = __half2float(sxhp[t * STRIDE_X + d_col]);
        float x1 = __half2float(sxhp[t * STRIDE_X + d_col + 1]);
        y0 = fmaf(dv0, x0, y0);
        y1 = fmaf(dv1, x1, y1);
      }
      if (has_z) {
        size_t off0 = ((size_t)(t0 + t) * H + h) * HDIM + d_col;
        float z0, z1;
        if (z_is_f16) {
          z0 = __half2float(z_hp[off0]);
          z1 = __half2float(z_hp[off0 + 1]);
        } else {
          z0 = __bfloat162float(z_bf[off0]);
          z1 = __bfloat162float(z_bf[off0 + 1]);
        }
        y0 *= z0 / (1.f + __expf(-z0));
        y1 *= z1 / (1.f + __expf(-z1));
      }
      bf16 b0 = __float2bfloat16_rn(y0), b1 = __float2bfloat16_rn(y1);
      reinterpret_cast<uint32_t*>(sY + t * STRIDE_X)[d_col >> 1] =
          (__bfloat16_as_ushort(b1) << 16) | (__bfloat16_as_ushort(b0));
    }
  }
  __syncthreads();

  // Global store pass: thread handles one 16B chunk = 8 tokens for one d row.
  // Destinations are contiguous in the token axis; tails at the segment edge
  // fall back to per-element stores.
  {
    const int64_t batch = varlen ? 0 : (t0 / L);
    bf16* obase = out + (size_t)h * out_head_stride + batch * (H * out_head_stride);
    const int64_t toff = varlen ? t0 : (t0 % L);
#pragma unroll
    for (int idx = tid; idx < HDIM * (CHUNK / 8); idx += 512) {
      int d = idx >> 4, tg = (idx & 15) * 8;
      bf16* dst = obase + d * out_row_stride + toff + tg;
      if (tg + 8 <= len) {
        uint32_t pk[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          uint32_t lo = __bfloat16_as_ushort(sY[(tg + 2 * k) * STRIDE_X + d]);
          uint32_t hi = __bfloat16_as_ushort(sY[(tg + 2 * k + 1) * STRIDE_X + d]);
          pk[k] = (hi << 16) | lo;
        }
        reinterpret_cast<uint4*>(dst)[0] = *reinterpret_cast<const uint4*>(pk);
      } else {
        for (int k = 0; k < 8 && tg + k < len; ++k) dst[k] = sY[(tg + k) * STRIDE_X + d];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher (single translation unit; dtype dispatch happens in csrc)
// ---------------------------------------------------------------------------
template <typename DtT, typename IdxT, typename StateT>
cudaError_t LaunchVibeCudaSsdCombined(const void* x, const void* dt, const void* dt_bias,
                                      const void* a, const void* b, const void* c, const void* dmat,
                                      const void* z, int z_is_f16, const void* initial,
                                      const IdxT* seq_idx, void* state_in, void* out,
                                      void* final_states, int Bsz, int L, int H, int G, int nseg,
                                      int nLCmax, int softplus, double dt_lo, double dt_hi,
                                      int d_has_hdim, int varlen, bool all_single_host,
                                      cudaStream_t stream) {
  const int NT = Bsz * L;
  const int hpg = H / G;
  const bf16* xp = reinterpret_cast<const bf16*>(x);
  const bf16* bp = reinterpret_cast<const bf16*>(b);
  const bf16* cp = reinterpret_cast<const bf16*>(c);
  const bf16* dp = reinterpret_cast<const bf16*>(dmat);
  const bf16* z_bf = reinterpret_cast<const bf16*>(z);
  const fp16* z_hp = reinterpret_cast<const fp16*>(z);

  if (!all_single_host) {
    static std::once_flag once;
    std::call_once(once, [] {
      cudaFuncSetAttribute((k_segstate<float, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<float, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<float, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<float, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<bf16, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<bf16, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<bf16, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<bf16, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<fp16, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<fp16, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<fp16, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
      cudaFuncSetAttribute((k_segstate<fp16, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM);
    });
    dim3 grid(nseg, H);
    k_segstate<DtT, IdxT, StateT><<<grid, 256, KS_SMEM, stream>>>(
        xp, bp, reinterpret_cast<const DtT*>(dt), reinterpret_cast<const DtT*>(dt_bias),
        reinterpret_cast<const float*>(a), softplus, (float)dt_lo, (float)dt_hi, seq_idx, NT, L, H,
        hpg, G, nseg, varlen, reinterpret_cast<const StateT*>(initial),
        reinterpret_cast<bf16*>(state_in), reinterpret_cast<StateT*>(final_states));
  }

  {
    static std::once_flag once3;
    std::call_once(once3, [] {
      cudaFuncSetAttribute((k_out<float, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<float, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<float, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<float, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<bf16, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<bf16, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<bf16, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<bf16, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<fp16, int32_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<fp16, int32_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<fp16, int64_t, bf16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
      cudaFuncSetAttribute((k_out<fp16, int64_t, fp16>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
    });
    dim3 grid(nLCmax, H);
    const int64_t out_row_stride = (int64_t)L;              // nchunks * CHUNK
    const int64_t out_head_stride = out_row_stride * HDIM;  // D * nchunks * CHUNK
    k_out<DtT, IdxT, StateT><<<grid, 512, K3_SMEM, stream>>>(
        xp, bp, cp, dp, z != nullptr && !z_is_f16 ? z_bf : nullptr,
        z != nullptr && z_is_f16 ? z_hp : nullptr, z_is_f16, reinterpret_cast<const DtT*>(dt),
        reinterpret_cast<const DtT*>(dt_bias), reinterpret_cast<const float*>(a), softplus,
        (float)dt_lo, (float)dt_hi, seq_idx, NT, L, H, hpg, G, nseg, varlen, d_has_hdim,
        z != nullptr ? 1 : 0, reinterpret_cast<const bf16*>(state_in),
        reinterpret_cast<const StateT*>(initial), reinterpret_cast<StateT*>(final_states),
        reinterpret_cast<bf16*>(out), out_head_stride, out_row_stride);
  }
  return cudaGetLastError();
}

#undef DEVI

}  // namespace vibecuda
}  // namespace mamba
}  // namespace flashinfer
