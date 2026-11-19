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

// Bookkeeping of the multi-token verify of kda.bf16.c16: the record of what a span of draft rows
// overwrites, and the settlement that puts a slot where the kept rows leave it. No row arithmetic:
// every row and every prefix value comes from batch_invariant_decode.cuh.

#include <flashinfer/kda/batch_invariant_decode.cuh>

namespace flashinfer {
namespace kda {
namespace bi_verify {

using bi_decode::CH;
using bi_decode::D;
using bi_decode::STAMP;
constexpr int NTHR = 256;
// per (slot, head): draft rows pending, count after them, count before them, cached rows recorded
// with the boundary state (-1: no boundary crossed)
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

// One launch per verify or settlement. First each slot's pending verify is settled with
// accepted[b] of its rows kept (nullptr: all): a slot that keeps fewer gets the count and the decay
// prefix of the kept position, and the boundary state and cached rows the dropped rows replaced.
// Then, for n > 0 (at most CH), what a span of n rows will overwrite is recorded.
__global__ void __launch_bounds__(NTHR)
    SettleKernel(const int* __restrict__ idx, const int* __restrict__ accepted, int n, float* state,
                 uint16_t* state16, int* cnt, float* pref, uint16_t* Kc, float* Kf, uint16_t* Rc,
                 uint16_t* Uc, uint16_t* Lc, float* stamp, int* meta, float* vpref,
                 const float* __restrict__ phist, float* vstate, uint16_t* vK, float* vF,
                 uint16_t* vR, uint16_t* vU, uint16_t* vL, int cap, int H, int S) {
  const int tid = threadIdx.x, h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  if (slot < 0 || slot >= S) return;
  const size_t sh = (size_t)slot * H + h, r16 = sh * CH * D;
  float* S32 = state + sh * D * D;
  float* V32 = vstate + sh * D * D;
  const int* m = meta + sh * META;
  const int pending = m[0], after = m[1], before = m[2], saved = m[3];
  const int c0 = cnt[sh];
  const bool moved = replaced(S32, stamp + sh * STAMP, c0);
  // the record stands only while the slot is where the verify left it
  const bool live = pending > 0 && c0 == after && !moved;
  const int a = accepted ? (accepted[b] < 0 ? 0 : accepted[b]) : pending;
  const bool back = live && a < pending;
  const int pos = before + a;
  // kept rows that stop before the boundary the verify crossed: its state and rows come back
  const bool undo = back && saved >= 0 && pos < CH;
  float pv = 0.0f;
  if (tid < D)
    pv = !back ? pref[sh * D + tid]
               : (a > 0 ? phist[(sh * cap + a - 1) * D + tid] : vpref[sh * D + tid]);
  __syncthreads();
  if (back && tid < D) pref[sh * D + tid] = pv;
  if (undo) {
    uint16_t* S16 = state16 + sh * D * D;
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
    copy16(Kc + r16, vK + r16, (size_t)saved * D * 2, tid);
    copy16(Kf + r16, vF + r16, (size_t)saved * D * 4, tid);
    copy16(Rc + r16, vR + r16, (size_t)saved * D * 2, tid);
    copy16(Uc + r16, vU + r16, (size_t)saved * D * 2, tid);
    copy16(Lc + sh * CH * CH, vL + sh * CH * CH, (size_t)saved * CH * 2, tid);
  }
  // the slot's count now: where the kept rows leave it, else as found
  const int p = back ? (pos & (CH - 1)) : (moved ? 0 : (c0 & (CH - 1)));
  if (n <= 0) {
    if (tid == 0) {
      meta[sh * META] = 0;
      if (back) cnt[sh] = p;
    }
    return;
  }
  __syncthreads();
  const int rows = p + n - CH;
  if (tid < D) vpref[sh * D + tid] = pv;
  if (rows >= 0) {
    // after an undo the record already holds this boundary state
    if (!undo) copy16(V32, S32, (size_t)D * D * 4, tid);
    copy16(vK + r16, Kc + r16, (size_t)rows * D * 2, tid);
    copy16(vF + r16, Kf + r16, (size_t)rows * D * 4, tid);
    copy16(vR + r16, Rc + r16, (size_t)rows * D * 2, tid);
    copy16(vU + r16, Uc + r16, (size_t)rows * D * 2, tid);
    copy16(vL + sh * CH * CH, Lc + sh * CH * CH, (size_t)rows * CH * 2, tid);
  }
  if (tid < META) {
    const int w[META] = {n, (p + n) & (CH - 1), p, rows < 0 ? -1 : rows};
    meta[sh * META + tid] = w[tid];
  }
  if (tid == 0 && back) cnt[sh] = p;
}

inline cudaError_t Settle(const int* slots, const int* accepted, int n, float* state,
                          uint16_t* state_bf16, int* cnt, float* prefix, uint16_t* k_inv_rows,
                          float* k_inv_f32_rows, uint16_t* rhs_rows, uint16_t* update_rows,
                          uint16_t* l_rows, float* stamp, int* meta, float* v_prefix,
                          const float* prefix_rows, float* v_state, uint16_t* v_k, float* v_kf,
                          uint16_t* v_r, uint16_t* v_u, uint16_t* v_l, int cap, int batch,
                          int num_heads, int num_slots, cudaStream_t stream) {
  SettleKernel<<<batch * num_heads, NTHR, 0, stream>>>(
      slots, accepted, n, state, state_bf16, cnt, prefix, k_inv_rows, k_inv_f32_rows, rhs_rows,
      update_rows, l_rows, stamp, meta, v_prefix, prefix_rows, v_state, v_k, v_kf, v_r, v_u, v_l,
      cap, num_heads, num_slots);
  return cudaGetLastError();
}

}  // namespace bi_verify
}  // namespace kda
}  // namespace flashinfer

#endif  // FLASHINFER_KDA_BATCH_INVARIANT_VERIFY_CUH_
