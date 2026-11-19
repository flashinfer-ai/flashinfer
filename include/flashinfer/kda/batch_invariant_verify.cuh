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
#ifndef FLASHINFER_KDA_BATCH_INVARIANT_VERIFY_CUH_
#define FLASHINFER_KDA_BATCH_INVARIANT_VERIFY_CUH_

// Bookkeeping of the multi-token verify of kda.bf16.c16: a snapshot of what a span of draft rows
// overwrites, and its restore. No row arithmetic: every row comes from batch_invariant_decode.cuh.

#include <flashinfer/kda/batch_invariant_decode.cuh>

namespace flashinfer {
namespace kda {
namespace bi_verify {

using bi_decode::CH;
using bi_decode::D;
using bi_decode::STAMP;
constexpr int NTHR = 256;
// per (slot, head): draft rows pending, count after them, count before them (-1: a replaced
// state), cached rows saved with the boundary state (-1: no boundary crossed)
constexpr int META = 4;

// the block moves `bytes` 16 at a time; a thread issues its loads (up to sixteen) before its first
// store
__device__ __forceinline__ void copy16(void* dst, const void* src, size_t bytes, int tid) {
  const unsigned char* s = static_cast<const unsigned char*>(src);
  unsigned char* d = static_cast<unsigned char*>(dst);
  const size_t step = (size_t)NTHR * 16;
  for (size_t u = (size_t)tid * 16; u < bytes; u += 16 * step) {
    uint4 x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (u + i * step < bytes) x[i] = *reinterpret_cast<const uint4*>(s + u + i * step);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (u + i * step < bytes) *reinterpret_cast<uint4*>(d + u + i * step) = x[i];
  }
}

// the step kernel's test for a state the caller replaced
__device__ __forceinline__ bool replaced(const float* S32, const float* Stamp, int c0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(S32);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(Stamp);
  uint32_t d = 0u;
#pragma unroll
  for (int i = 0; i < STAMP; ++i) d |= a[i] ^ b[i];
  return c0 < 0 || d != 0u;
}

// Before a span of n rows per sequence: save what it overwrites and the n token rows.
__global__ void __launch_bounds__(NTHR)
    SnapshotKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                   const uint16_t* __restrict__ v, const uint16_t* __restrict__ gt,
                   const uint16_t* __restrict__ bt, const int* __restrict__ idx, int n,
                   const float* __restrict__ state, const int* __restrict__ cnt,
                   const float* __restrict__ pref, const uint16_t* __restrict__ Kc,
                   const float* __restrict__ Kf, const uint16_t* __restrict__ Rc,
                   const uint16_t* __restrict__ Uc, const uint16_t* __restrict__ Lc,
                   const float* __restrict__ stamp, int* meta, float* vpref, float* vstate,
                   uint16_t* vK, float* vF, uint16_t* vR, uint16_t* vU, uint16_t* vL, uint16_t* tq,
                   uint16_t* tk, uint16_t* tv, uint16_t* tg, uint16_t* tb, int cap, int H, int S) {
  const int tid = threadIdx.x, h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  if (slot < 0 || slot >= S) return;
  const size_t sh = (size_t)slot * H + h;
  const float* S32 = state + sh * D * D;
  const int c0 = cnt[sh];
  const bool moved = replaced(S32, stamp + sh * STAMP, c0);
  const int p = moved ? 0 : (c0 & (CH - 1));
  const int over = p + n - CH;
  const int rows = over < 0 ? -1 : (over < p ? over : p);
  if (tid < META) {
    const int m[META] = {n, (p + n) & (CH - 1), moved ? -1 : p, rows};
    meta[sh * META + tid] = m[tid];
  }
  // 64 threads per token row: four rows per round
  const uint16_t* src[4] = {q, k, v, gt};
  uint16_t* dst[4] = {tq, tk, tv, tg};
  const int a = (tid >> 4) & 3, c8 = (tid & 15) * 8;
  for (int j = tid >> 6; j < n; j += NTHR / 64) {
    const size_t in = ((size_t)b * n + j) * H + h, sv = ((size_t)slot * cap + j) * H + h;
    *reinterpret_cast<uint4*>(dst[a] + sv * D + c8) =
        *reinterpret_cast<const uint4*>(src[a] + in * D + c8);
  }
  if (tid < n) tb[((size_t)slot * cap + tid) * H + h] = bt[((size_t)b * n + tid) * H + h];
  copy16(vpref + sh * D, pref + sh * D, (size_t)D * 4, tid);
  if (rows >= 0) {
    copy16(vstate + sh * D * D, S32, (size_t)D * D * 4, tid);
    const size_t r16 = sh * CH * D;
    copy16(vK + r16, Kc + r16, (size_t)rows * D * 2, tid);
    copy16(vF + r16, Kf + r16, (size_t)rows * D * 4, tid);
    copy16(vR + r16, Rc + r16, (size_t)rows * D * 2, tid);
    copy16(vL + sh * CH * CH, Lc + sh * CH * CH, (size_t)CH * CH * 2, tid);
    copy16(vU + r16, Uc + r16, (size_t)rows * D * 2, tid);
  }
}

// After a verify: a sequence that keeps fewer rows than were run goes back to the snapshot and
// gets the kept rows as a replay span (starts / lens); every pending record is cleared.
__global__ void __launch_bounds__(NTHR)
    AcceptKernel(const int* __restrict__ idx, const int* __restrict__ accepted, float* state,
                 uint16_t* state16, int* cnt, float* pref, uint16_t* Kc, float* Kf, uint16_t* Rc,
                 uint16_t* Uc, uint16_t* Lc, float* stamp, int* meta,
                 const float* __restrict__ vpref, const float* __restrict__ vstate,
                 const uint16_t* __restrict__ vK, const float* __restrict__ vF,
                 const uint16_t* __restrict__ vR, const uint16_t* __restrict__ vU,
                 const uint16_t* __restrict__ vL, int* starts, int* lens, int cap, int H, int S) {
  const int tid = threadIdx.x, h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  const bool inside = slot >= 0 && slot < S;
  const size_t sh = inside ? (size_t)slot * H + h : 0;
  int keep = 0, rows = -1, before = 0;
  bool back = false;
  if (inside) {
    float* S32 = state + sh * D * D;
    const int* m = meta + sh * META;
    const int n = m[0], c0 = cnt[sh];
    before = m[2];
    rows = m[3];
    // the record stands only while the slot is where the verify left it
    const bool live = n > 0 && c0 == m[1] && !replaced(S32, stamp + sh * STAMP, c0);
    const int a = accepted ? accepted[b] : n;
    keep = a < 0 ? 0 : a;
    back = live && keep < n;
    if (!back) keep = 0;
  }
  if (h == 0 && tid == 0) {
    starts[b] = inside ? slot * cap : 0;
    lens[b] = keep;
  }
  if (!inside) return;
  __syncthreads();
  if (tid == 0) meta[sh * META] = 0;
  if (!back) return;
  copy16(pref + sh * D, vpref + sh * D, (size_t)D * 4, tid);
  if (rows >= 0) {
    float* S32 = state + sh * D * D;
    uint16_t* S16 = state16 + sh * D * D;
    const float* V32 = vstate + sh * D * D;
    // the boundary state and its bf16 copy as the step header's begin writes it; eight pieces per
    // thread
    float4 x[8], y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x[i] = *reinterpret_cast<const float4*>(V32 + 8 * (tid + i * NTHR));
      y[i] = *reinterpret_cast<const float4*>(V32 + 8 * (tid + i * NTHR) + 4);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e8 = 8 * (tid + i * NTHR);
      *reinterpret_cast<float4*>(S32 + e8) = x[i];
      *reinterpret_cast<float4*>(S32 + e8 + 4) = y[i];
      *reinterpret_cast<uint4*>(S16 + e8) =
          make_uint4(bi_decode::f2bf_x2(x[i].x, x[i].y), bi_decode::f2bf_x2(x[i].z, x[i].w),
                     bi_decode::f2bf_x2(y[i].x, y[i].y), bi_decode::f2bf_x2(y[i].z, y[i].w));
    }
    if (tid < STAMP) stamp[sh * STAMP + tid] = V32[tid];
    const size_t r16 = sh * CH * D;
    copy16(Kc + r16, vK + r16, (size_t)rows * D * 2, tid);
    copy16(Kf + r16, vF + r16, (size_t)rows * D * 4, tid);
    copy16(Rc + r16, vR + r16, (size_t)rows * D * 2, tid);
    copy16(Lc + sh * CH * CH, vL + sh * CH * CH, (size_t)CH * CH * 2, tid);
    copy16(Uc + r16, vU + r16, (size_t)rows * D * 2, tid);
  }
  if (tid == 0) cnt[sh] = before;
}

inline cudaError_t Snapshot(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                            const uint16_t* g, const uint16_t* beta, const int* slots, int n,
                            const float* state, const int* cnt, const float* prefix,
                            const uint16_t* k_inv_rows, const float* k_inv_f32_rows,
                            const uint16_t* rhs_rows, const uint16_t* update_rows,
                            const uint16_t* l_rows, const float* stamp, int* meta, float* v_prefix,
                            float* v_state, uint16_t* v_k, float* v_kf, uint16_t* v_r,
                            uint16_t* v_u, uint16_t* v_l, uint16_t* t_q, uint16_t* t_k,
                            uint16_t* t_v, uint16_t* t_g, uint16_t* t_beta, int cap, int batch,
                            int num_heads, int num_slots, cudaStream_t stream) {
  SnapshotKernel<<<batch * num_heads, NTHR, 0, stream>>>(
      q, k, v, g, beta, slots, n, state, cnt, prefix, k_inv_rows, k_inv_f32_rows, rhs_rows,
      update_rows, l_rows, stamp, meta, v_prefix, v_state, v_k, v_kf, v_r, v_u, v_l, t_q, t_k, t_v,
      t_g, t_beta, cap, num_heads, num_slots);
  return cudaGetLastError();
}

inline cudaError_t Accept(const int* slots, const int* accepted, float* state, uint16_t* state_bf16,
                          int* cnt, float* prefix, uint16_t* k_inv_rows, float* k_inv_f32_rows,
                          uint16_t* rhs_rows, uint16_t* update_rows, uint16_t* l_rows, float* stamp,
                          int* meta, const float* v_prefix, const float* v_state,
                          const uint16_t* v_k, const float* v_kf, const uint16_t* v_r,
                          const uint16_t* v_u, const uint16_t* v_l, int* starts, int* lens, int cap,
                          int batch, int num_heads, int num_slots, cudaStream_t stream) {
  AcceptKernel<<<batch * num_heads, NTHR, 0, stream>>>(
      slots, accepted, state, state_bf16, cnt, prefix, k_inv_rows, k_inv_f32_rows, rhs_rows,
      update_rows, l_rows, stamp, meta, v_prefix, v_state, v_k, v_kf, v_r, v_u, v_l, starts, lens,
      cap, num_heads, num_slots);
  return cudaGetLastError();
}

}  // namespace bi_verify
}  // namespace kda
}  // namespace flashinfer

#endif  // FLASHINFER_KDA_BATCH_INVARIANT_VERIFY_CUH_
