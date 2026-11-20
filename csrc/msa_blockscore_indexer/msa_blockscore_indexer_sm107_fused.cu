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

// MSA stage-1 block-score top-k indexer for SM107 (Rubin).
// Kernel 1: persistent, work-balanced tcgen05 BF16 scoring whose epilogue keeps
//           a per-query top-15 in registers (scores never reach global memory).
// Kernel 2: merges the per-chunk top-15 lists and emits the ascending ids.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cfloat>
#include <cstdint>
#include <mutex>

namespace ffi = tvm::ffi;

namespace msa_fused {

#define KBLOCK 128
#define DHEAD 128
#define TOPK 16
#define NLIST 15
#define BMT 128
#define NAT 2
#define BM (NAT * BMT)
#define BN 256
#define NBLKT 2
#define STAGES 2
#define LOG2S 1
#define NTHR 256
#define MAXTILES 1024
#define MAXPAIRS 1536

// Primary and overflow workspaces keep the fused path reentrant while leaving
// the common first-stream launch free of acquisition overhead.
constexpr size_t PART_ELEMS = (size_t)MAXPAIRS * NLIST * BM;
__device__ __align__(256) uint64_t g_part_primary[PART_ELEMS];
__device__ __align__(256) uint64_t g_part_overflow[PART_ELEMS];
__device__ int g_cnt_primary[MAXTILES];
__device__ int g_cnt_overflow[MAXTILES];
__device__ int g_overflow_lock;
__device__ int g_overflow_done;

__global__ void acquire_workspace() {
  if (threadIdx.x == 0) {
    // Acquire CAS: pairs with the release store in finish_workspace(), so
    // taking the lock makes every write the previous overflow call published
    // before its release (its relaxed per-tile counter resets included)
    // visible here; kernel completion then orders this grid before the
    // stream-ordered consumer kernels that use the slot.
    int got;
    do {
      asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;"
                   : "=r"(got)
                   : "l"(&g_overflow_lock), "r"(0), "r"(1)
                   : "memory");
      if (got != 0) __nanosleep(64);
    } while (got != 0);
    g_overflow_done = 0;
  }
}

__device__ __forceinline__ void finish_workspace() {
  __syncthreads();
  if (threadIdx.x == 0) {
    // acq_rel completion increment: the release half publishes this CTA's
    // overflow-slot writes (a merger's relaxed counter reset included); the
    // increments form a release sequence on g_overflow_done, so the last
    // arrival's acquire half sees every earlier CTA's writes before the
    // release below hands the slot to the next call's acquire CAS.
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(old)
                 : "l"(&g_overflow_done)
                 : "memory");
    if (old == (int)gridDim.x - 1) {
      g_overflow_done = 0;
      asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(&g_overflow_lock), "r"(0)
                   : "memory");
    }
  }
}

template <bool GUARDED>
__device__ __forceinline__ uint64_t* part_workspace() {
  if constexpr (GUARDED)
    return g_part_overflow;
  else
    return g_part_primary;
}

template <bool GUARDED>
__device__ __forceinline__ int* count_workspace() {
  if constexpr (GUARDED)
    return g_cnt_overflow;
  else
    return g_cnt_primary;
}

#define ASZ (BMT * DHEAD * 2)
#define BSZ (BN * DHEAD * 2)

// ---------------- device helpers ----------------
__device__ __forceinline__ uint32_t elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t}"
      : "+r"(pred)
      : "r"(0xFFFFFFFFu));
  return pred;
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  uint64_t d = 0;
  d |= desc_enc(addr);
  d |= desc_enc(1024) << 32ULL;  // SBO = 8 rows * 128 B
  d |= 1ULL << 46ULL;            // gen5 flag
  d |= 2ULL << 61ULL;            // 128B swizzle
  return d;
}

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}

__device__ __forceinline__ void mbar_wait(uint32_t a, int phase) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_W%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE%=;\n\t"
      "bra.uni LAB_W%=;\n\t"
      "DONE%=:\n\t}" ::"r"(a),
      "r"(phase), "r"(0x989680u));
}

__device__ __forceinline__ void mbar_expect(uint32_t a, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tma_2d(uint32_t smem, const CUtensorMap* tm, int c, int r,
                                       uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem),
      "l"(tm), "r"(c), "r"(r), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void mma_f16(uint32_t taddr, uint64_t ad, uint64_t bd, uint32_t idesc,
                                        int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(taddr),
      "l"(ad), "l"(bd), "r"(idesc), "r"(acc)
      : "memory");
}

#define TC_TMP64                                                     \
  "t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,"           \
  "t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31," \
  "t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47," \
  "t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,t61,t62,t63"

// Max of a 64-column TMEM strip (hardware row statistic).
__device__ __forceinline__ float tmem_ld_max64(uint32_t a) {
  float red;
  asm volatile(
      "{ .reg .f32 t<64>; "
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {" TC_TMP64 "}, %0, [%1]; }"
      : "=f"(red)
      : "r"(a));
  return red;
}

// Score key: monotone u32 of the float in the high half, complemented block id
// in the low half, so one unsigned compare gives score-desc / id-asc order.
__device__ __forceinline__ uint64_t pack_key(float v, int j) {
  uint32_t b = __float_as_uint(v);
  if ((b << 1) == 0) b = 0;
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((uint64_t)b << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)j);
}

// Sorted-descending top-15 kept in registers; ascending candidate order makes
// the strict compare implement the lower-id tie-break.
#define LIST_DECL                                                                          \
  uint64_t L0 = 0, L1 = 0, L2 = 0, L3 = 0, L4 = 0, L5 = 0, L6 = 0, L7 = 0, L8 = 0, L9 = 0, \
           L10 = 0, L11 = 0, L12 = 0, L13 = 0, L14 = 0
#define CE(a, b)             \
  {                          \
    uint64_t _x = a, _y = b; \
    a = _x > _y ? _x : _y;   \
    b = _x > _y ? _y : _x;   \
  }
#define INSERT(x)                                                                             \
  {                                                                                           \
    uint64_t _v = (x);                                                                        \
    if (_v > L14) {                                                                           \
      L14 = _v;                                                                               \
      CE(L13, L14)                                                                            \
      CE(L12, L13)                                                                            \
      CE(L11, L12)                                                                            \
      CE(L10, L11)                                                                            \
      CE(L9, L10)                                                                             \
      CE(L8, L9)                                                                              \
      CE(L7, L8) CE(L6, L7) CE(L5, L6) CE(L4, L5) CE(L3, L4) CE(L2, L3) CE(L1, L2) CE(L0, L1) \
    }                                                                                         \
  }
#define LIST_AT(i) \
  (i == 0    ? L0  \
   : i == 1  ? L1  \
   : i == 2  ? L2  \
   : i == 3  ? L3  \
   : i == 4  ? L4  \
   : i == 5  ? L5  \
   : i == 6  ? L6  \
   : i == 7  ? L7  \
   : i == 8  ? L8  \
   : i == 9  ? L9  \
   : i == 10 ? L10 \
   : i == 11 ? L11 \
   : i == 12 ? L12 \
   : i == 13 ? L13 \
             : L14)

struct TileInfo {
  int r, qbase, nqt, nb;
};

__device__ __forceinline__ void tile_info(int g, const int* __restrict__ cu,
                                          const int* __restrict__ prefix, int B, TileInfo& t) {
  int acc = 0;
  t.r = -1;
  t.qbase = 0;
  t.nqt = 0;
  t.nb = 0;
  for (int i = 0; i < B; ++i) {
    int q0 = cu[i], q1 = cu[i + 1];
    int nt = (q1 - q0 + BM - 1) / BM;
    if (g < acc + nt) {
      t.r = i;
      t.qbase = q0 + (g - acc) * BM;
      t.nqt = min(BM, q1 - t.qbase);
      t.nb = (prefix[i] + (t.qbase - q0) + t.nqt - 1) >> 7;
      return;
    }
    acc += nt;
  }
}

__device__ __forceinline__ int tile_iters(const TileInfo& t) {
  if (t.nqt <= 0 || t.nb <= 15) return 0;
  return (t.nb + NBLKT - 1) / NBLKT;
}

__device__ __forceinline__ int count_tiles(const int* __restrict__ cu, int B) {
  int n = 0;
  for (int i = 0; i < B; ++i) n += (cu[i + 1] - cu[i] + BM - 1) / BM;
  return n;
}

// Which persistent CTA owns global iteration x, given start(c) = c*total/NCTA.
__device__ __forceinline__ int cta_of(int x, int total, int NCTA) {
  return (int)((((long long)(x + 1) * NCTA) - 1) / total);
}

// 256-thread exclusive scan; also returns the block total.
__device__ __forceinline__ int blk_scan_excl(int v, int* ssum, int& total) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) ssum[w] = x;
  __syncthreads();
  int off = 0, tot = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int s = ssum[i];
    tot += s;
    if (i < w) off += s;
  }
  total = tot;
  __syncthreads();
  return off + x - v;
}

// Builds s_cum[0..ntiles] (iteration prefix sums) and s_pb[] (pair offsets).
__device__ int build_tables(int ntiles, const int* __restrict__ cu, const int* __restrict__ prefix,
                            int B, int NCTA, int* s_cum, int* s_pb, int* ssum) {
  int run = 0;
  for (int base = 0; base < ntiles; base += NTHR) {
    int g = base + (int)threadIdx.x, v = 0;
    if (g < ntiles) {
      TileInfo t;
      tile_info(g, cu, prefix, B, t);
      v = tile_iters(t);
    }
    int tot, ex = blk_scan_excl(v, ssum, tot);
    if (g < ntiles) s_cum[g] = run + ex;
    run += tot;
  }
  if (threadIdx.x == 0) s_cum[ntiles] = run;
  __syncthreads();
  const int total = run;
  if (total == 0) return 0;
  run = 0;
  for (int base = 0; base < ntiles; base += NTHR) {
    int g = base + (int)threadIdx.x, sl = 0;
    if (g < ntiles && s_cum[g + 1] > s_cum[g]) {
      sl = cta_of(s_cum[g + 1] - 1, total, NCTA) - cta_of(s_cum[g], total, NCTA) + 1;
      const int iters = s_cum[g + 1] - s_cum[g];
      if (iters < sl) sl = iters;  // empty CTA ranges own no slot
    }
    int tot, ex = blk_scan_excl(sl, ssum, tot);
    if (g < ntiles) s_pb[g] = run + ex;
    run += tot;
  }
  return total;
}

#define TC_TMP128                                                                \
  "t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,"                       \
  "t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,"             \
  "t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,"             \
  "t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,t61,t62,t63,"             \
  "t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,"             \
  "t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90,t91,t92,t93,t94,t95,"             \
  "t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111," \
  "t112,t113,t114,t115,t116,t117,t118,t119,t120,t121,t122,t123,t124,t125,t126,t127"

// One 128-key block reduced by a single LDTM.STAT row-max (128 columns).
#define TC_TMP128                                                                \
  "t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,"                       \
  "t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,"             \
  "t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,"             \
  "t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,t61,t62,t63,"             \
  "t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,"             \
  "t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90,t91,t92,t93,t94,t95,"             \
  "t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111," \
  "t112,t113,t114,t115,t116,t117,t118,t119,t120,t121,t122,t123,t124,t125,t126,t127"
__device__ __forceinline__ float tmem_max_blk(uint32_t a) {
  float red;
  asm volatile(
      "{ .reg .f32 t<128>; "
      "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32 {" TC_TMP128
      "}, %0, [%1];\n\t"
      "tcgen05.wait::ld.sync.aligned; }"
      : "=f"(red)
      : "r"(a)
      : "memory");
  return red;
}

// One pipeline stage of the epilogue: reduce both key blocks of accumulator
// (a) and fold them into this thread's running top-15.
#define EPI(a, itv)                                                    \
  {                                                                    \
    if (fa == (a)) {                                                   \
      asm volatile("tcgen05.fence::after_thread_sync;");               \
      const uint32_t _b = taddr + (uint32_t)((a) * BN) + lane_off;     \
      const int _bk = (itv) * NBLKT;                                   \
      _Pragma("unroll 1") for (int _bq = 0; _bq < NBLKT; ++_bq) {      \
        float _s = tmem_max_blk(_b + _bq * KBLOCK);                    \
        if (_bk + _bq < own) INSERT(pack_key(_s, _bk + _bq))           \
      }                                                                \
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); \
    }                                                                  \
  }

// ---------------- per-tile merge (run by the tile's last CTA) ----------------
// One thread per query: merging two sorted-descending 16-lists is a 16-way
// compare-exchange plus a 4-stage bitonic clean-up, all register-resident.
#define MCE(a, b)                  \
  {                                \
    uint64_t _x = L[a], _y = L[b]; \
    L[a] = _x > _y ? _x : _y;      \
    L[b] = _x > _y ? _y : _x;      \
  }
#define ICE(a, b)             \
  {                           \
    int _x = I[a], _y = I[b]; \
    I[a] = _x < _y ? _x : _y; \
    I[b] = _x < _y ? _y : _x; \
  }

__device__ __forceinline__ void write_sel(const TileInfo& T, const int* I, int* __restrict__ sel,
                                          int* stage) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < TOPK; ++i) stage[tid * 17 + i] = I[i];
  __syncthreads();
  const int n = T.nqt * TOPK;
  for (int o = tid; o < n; o += NTHR)
    sel[(size_t)T.qbase * TOPK + o] = stage[(o >> 4) * 17 + (o & 15)];
  // The staging aliases the shared memory a later tile's TMA copies (async
  // proxy) write into; __syncthreads() alone gives no generic-to-async proxy
  // ordering, so publish these generic-proxy accesses to the async proxy
  // before the storage is reused (callers synchronize threads right after).
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fill_direct(const TileInfo& T, const int* __restrict__ cu,
                                            const int* __restrict__ prefix, int* __restrict__ sel,
                                            int* stage) {
  const int tid = threadIdx.x;
  int I[TOPK];
  const int own = (prefix[T.r] + (T.qbase + tid - cu[T.r])) >> 7;
#pragma unroll
  for (int i = 0; i < TOPK; ++i) I[i] = (i <= own) ? i : -1;
  write_sel(T, I, sel, stage);
}

template <bool GUARDED>
__device__ __forceinline__ void merge_tile(const TileInfo& T, int pb, int slots,
                                           const int* __restrict__ cu,
                                           const int* __restrict__ prefix, int* __restrict__ sel,
                                           int* stage) {
  const int tid = threadIdx.x;
  const int own = (prefix[T.r] + (T.qbase + tid - cu[T.r])) >> 7;
  int I[TOPK];
  if (own <= 15) {
#pragma unroll
    for (int i = 0; i < TOPK; ++i) I[i] = (i <= own) ? i : -1;
  } else {
    const uint64_t* base = part_workspace<GUARDED>() + (size_t)pb * (NLIST * BM) + tid;
    uint64_t L[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) L[i] = 0ull;
#pragma unroll 1
    for (int sl = 0; sl < slots; ++sl) {
      const uint64_t* p = base + (size_t)sl * (NLIST * BM);
      uint64_t nv[16];
      nv[0] = 0ull;  // pad the new list's small end
#pragma unroll
      for (int i = 1; i < 16; ++i) nv[i] = p[(15 - i) * BM];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        uint64_t t = nv[i];
        if (t > L[i]) L[i] = t;
      }
#pragma unroll
      for (int st = 8; st > 0; st >>= 1)
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if ((i & st) == 0) MCE(i, i + st)
    }
#pragma unroll
    for (int i = 0; i < NLIST; ++i)
      I[i] = (L[i] == 0ull) ? -1 : (int)(0xFFFFFFFFu - (uint32_t)L[i]);
    I[15] = own;
#pragma unroll
    for (int k = 2; k <= 16; k <<= 1)
#pragma unroll
      for (int jj = k >> 1; jj > 0; jj >>= 1)
#pragma unroll
        for (int i = 0; i < 16; ++i)
          if ((i & jj) == 0) {
            if ((i & k) == 0) ICE(i, i + jj) else ICE(i + jj, i)
          }
  }
  write_sel(T, I, sel, stage);
}

// ---------------- stage A : fused scoring + per-chunk top-15 ----------------
template <bool GUARDED>
__global__ __launch_bounds__(NTHR, 1) void score_kernel(const __grid_constant__ CUtensorMap qmap,
                                                        const __grid_constant__ CUtensorMap kmap,
                                                        const int* __restrict__ cu,
                                                        const int* __restrict__ kvbase,
                                                        const int* __restrict__ prefix,
                                                        int* __restrict__ sel, int B, int NCTA) {
  extern __shared__ __align__(1024) char smem_raw[];
  __shared__ int s_cum[MAXTILES + 1];
  __shared__ int s_pb[MAXTILES];
  __shared__ int ssum[8];
  __shared__ __align__(8) uint64_t bars[STAGES + 3];
  __shared__ int tmem_slot[1];
  __shared__ int s_last[1];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int ntiles = count_tiles(cu, B);
  const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sB = sA + NAT * ASZ;
  const uint32_t barB = (uint32_t)__cvta_generic_to_shared(&bars[0]);
  const uint32_t barM = barB + STAGES * 8;
  const uint32_t barA = barM + 16;

  if (warp == 0 && elect_one()) {
    for (int i = 0; i < STAGES + 3; ++i) mbar_init(barB + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp == 1) {
    const uint32_t ad = (uint32_t)__cvta_generic_to_shared(tmem_slot);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(ad),
                 "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  const int total = build_tables(ntiles, cu, prefix, B, NCTA, s_cum, s_pb, ssum);
  __syncthreads();
  const uint32_t taddr = (uint32_t)tmem_slot[0];

  const int c = (int)blockIdx.x;
// Tiles whose every query has own<=15 need no scoring; fill them here so a
// second kernel over all queries is never required.
#pragma unroll 1
  for (int g = c; g < ntiles; g += NCTA) {
    TileInfo T;
    tile_info(g, cu, prefix, B, T);
    if (T.nqt > 0 && tile_iters(T) == 0) {
      fill_direct(T, cu, prefix, sel, (int*)smem_raw);
      __syncthreads();
    }
  }
  if (total == 0) {
    if (warp == 1)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(512));
    if constexpr (GUARDED) finish_workspace();
    return;
  }

  const int it0 = (int)(((long long)c * total) / NCTA);
  const int it1 = (int)(((long long)(c + 1) * total) / NCTA);
  if (it0 >= it1) {
    if (warp == 1)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(512));
    if constexpr (GUARDED) finish_workspace();
    return;
  }

  int gstart = 0;
  {  // largest g with s_cum[g] <= it0
    int lo = 0, hi = ntiles;
    while (hi - lo > 1) {
      int m = (lo + hi) >> 1;
      if (s_cum[m] <= it0)
        lo = m;
      else
        hi = m;
    }
    gstart = lo;
  }

  constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(BN / 8) << 17) |
                             ((uint32_t)(BMT / 16) << 24);
  const int row = tid & 127;
  const int fa = tid >> 7;
  const uint32_t lane_off = (uint32_t)((row >> 5) * 32) << 16;
  int phA = 0, phM0 = 0, phM1 = 0;
  unsigned wcnt = 0;  // global B-stage wait counter: buffer = wcnt&1, phase = (wcnt>>1)&1

#pragma unroll 1
  for (int g = gstart; g < ntiles && s_cum[g] < it1; ++g) {
    const int lo = max(it0, s_cum[g]) - s_cum[g];
    const int hi = min(it1, s_cum[g + 1]) - s_cum[g];
    if (hi <= lo) continue;
    TileInfo T;
    tile_info(g, cu, prefix, B, T);
    const int krow0 = kvbase[T.r] * KBLOCK;
    const int fq = T.qbase + tid;
    const int own = (prefix[T.r] + (fq - cu[T.r])) >> 7;
    const int nit = hi - lo;

    if (warp == 0 && elect_one()) {
      mbar_expect(barA, NAT * ASZ);
      for (int a = 0; a < NAT; ++a) {
        int rows = T.qbase + a * BMT;
        tma_2d(sA + a * ASZ, &qmap, 0, rows, barA);
        tma_2d(sA + a * ASZ + ASZ / 2, &qmap, 64, rows, barA);
      }
      int pre = min(STAGES, nit);
      for (int st = 0; st < pre; ++st) {
        int bs = (int)((wcnt + st) & (STAGES - 1u));
        int kr = krow0 + (lo + st) * (NBLKT * KBLOCK);
        mbar_expect(barB + bs * 8, BSZ);
        tma_2d(sB + bs * BSZ, &kmap, 0, kr, barB + bs * 8);
        tma_2d(sB + bs * BSZ + BSZ / 2, &kmap, 64, kr, barB + bs * 8);
      }
    }
    LIST_DECL;
    mbar_wait(barA, phA);
    phA ^= 1;
#pragma unroll 1
    for (int i = 0; i < nit; ++i) {
      const int buf = (int)(wcnt & (STAGES - 1u));
      mbar_wait(barB + buf * 8, (int)((wcnt >> LOG2S) & 1u));
      ++wcnt;
      __syncthreads();
      if (warp == 0 && elect_one()) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t acc = taddr;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          mma_f16(acc, smem_desc(sA + (kk >> 2) * (ASZ / 2) + (kk & 3) * 32),
                  smem_desc(sB + buf * BSZ + (kk >> 2) * (BSZ / 2) + (kk & 3) * 32), idesc,
                  kk ? 1 : 0);
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                barM)
            : "memory");
      }
      if (i >= 1) {
        mbar_wait(barM + 8, phM1);
        phM1 ^= 1;
        EPI(1, lo + i - 1)
        __syncthreads();
        if (warp == 0 && elect_one()) {
          int nxt = i - 1 + STAGES;
          if (nxt < nit) {
            int bf = (buf + STAGES - 1) & (STAGES - 1);
            int kr = krow0 + (lo + nxt) * (NBLKT * KBLOCK);
            mbar_expect(barB + bf * 8, BSZ);
            tma_2d(sB + bf * BSZ, &kmap, 0, kr, barB + bf * 8);
            tma_2d(sB + bf * BSZ + BSZ / 2, &kmap, 64, kr, barB + bf * 8);
          }
        }
      }
      __syncthreads();
      if (warp == 0 && elect_one()) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint32_t acc = taddr + (uint32_t)BN;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
          mma_f16(acc, smem_desc(sA + ASZ + (kk >> 2) * (ASZ / 2) + (kk & 3) * 32),
                  smem_desc(sB + buf * BSZ + (kk >> 2) * (BSZ / 2) + (kk & 3) * 32), idesc,
                  kk ? 1 : 0);
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                barM + 8)
            : "memory");
      }
      mbar_wait(barM, phM0);
      phM0 ^= 1;
      EPI(0, lo + i)
    }
    mbar_wait(barM + 8, phM1);
    phM1 ^= 1;
    EPI(1, lo + nit - 1)
    __syncthreads();

    // Publish this chunk's top-15 per query row (entry-major: coalesced
    // here and coalesced again for the thread-per-query merge below).
    {
      const int cof = c - cta_of(s_cum[g], total, NCTA);
      const int pair = s_pb[g] + (cof < lo ? cof : lo);
      uint64_t* out = part_workspace<GUARDED>() + (size_t)pair * (NLIST * BM) + tid;
#pragma unroll
      for (int k = 0; k < NLIST; ++k) out[k * BM] = LIST_AT(k);
    }
    // The CTA that completes a tile owns its merge, so the per-query
    // selection never needs a second kernel launch over all queries.
    __syncthreads();
    if (tid == 0) {
      int slots = cta_of(s_cum[g + 1] - 1, total, NCTA) - cta_of(s_cum[g], total, NCTA) + 1;
      const int iters = s_cum[g + 1] - s_cum[g];
      if (iters < slots) slots = iters;
      // One acq_rel-ordered arrival publishes this chunk's list (release
      // half) and, for the last-arriving CTA, acquires every earlier
      // contributor's partial list (acquire half) *before* the CTA barrier
      // below, so the barrier distributes that visibility to the merger
      // threads; a full threadfence would drain far more than needed.
      int old;
      int* const counts = count_workspace<GUARDED>();
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                   : "=r"(old)
                   : "l"(&counts[g])
                   : "memory");
      s_last[0] = (old == slots - 1);
      if (old == slots - 1)
        asm volatile("st.relaxed.gpu.global.s32 [%0], 0;" ::"l"(&counts[g])
                     : "memory");  // armed for next call
    }
    __syncthreads();
    if (s_last[0]) {
      asm volatile("fence.acquire.gpu;" ::: "memory");
      int slots = cta_of(s_cum[g + 1] - 1, total, NCTA) - cta_of(s_cum[g], total, NCTA) + 1;
      const int iters2 = s_cum[g + 1] - s_cum[g];
      if (iters2 < slots) slots = iters2;
      merge_tile<GUARDED>(T, s_pb[g], slots, cu, prefix, sel, (int*)smem_raw);
      __syncthreads();
    }
  }
  __syncthreads();
  if (warp == 1)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(512));
  if constexpr (GUARDED) finish_workspace();
}

// Every query's own block is <=15 when the request has at most TOPK blocks:
// emit the forced prefix directly, no scoring and no tensor maps needed.
__global__ __launch_bounds__(256) void fill_kernel(const int* __restrict__ cu,
                                                   const int* __restrict__ prefix,
                                                   int* __restrict__ sel, int B, int TQ) {
  const int q = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (q >= TQ) return;
  const int lane = threadIdx.x & 31;
  int lo = 0, hi = B;
  while (hi - lo > 1) {
    int m = (lo + hi) >> 1;
    if (cu[m] <= q)
      lo = m;
    else
      hi = m;
  }
  const int own = (prefix[lo] + q - cu[lo]) >> 7;
  if (lane < TOPK) sel[(size_t)q * TOPK + lane] = (lane <= own) ? lane : -1;
}

// Allocation-free exact fallback for shapes outside the fast path's tile table.
__global__ __launch_bounds__(256, 2) void direct_fallback_kernel(
    const __nv_bfloat16* __restrict__ qi, const __nv_bfloat16* __restrict__ ki,
    const int* __restrict__ cu, const int* __restrict__ kvbase, const int* __restrict__ prefix,
    int* __restrict__ sel, int B, int TQ) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q = (int)blockIdx.x * 8 + warp;
  if (q >= TQ) return;
  int lo = 0, hi = B;
  while (hi - lo > 1) {
    int m = (lo + hi) >> 1;
    if (cu[m] <= q)
      lo = m;
    else
      hi = m;
  }
  const int own = (prefix[lo] + q - cu[lo]) >> 7;
  int* out = sel + (size_t)q * TOPK;
  if (own <= 15) {
    if (lane < TOPK) out[lane] = (lane <= own) ? lane : -1;
    return;
  }
  float qreg[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) qreg[j] = __bfloat162float(qi[(size_t)q * DHEAD + lane + j * 32]);
  __shared__ float sbest[8][15];
  __shared__ int sid[8][15];
  float worst = FLT_MAX;
  int worst_id = -1, worst_pos = 0, seen = 0;
  const int key0 = kvbase[lo] * KBLOCK;
  for (int block = 0; block < own; ++block) {
    float bmax = -FLT_MAX;
#pragma unroll 1
    for (int key = 0; key < KBLOCK; ++key) {
      const __nv_bfloat16* kp = ki + (size_t)(key0 + block * KBLOCK + key) * DHEAD;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) sum = fmaf(qreg[j], __bfloat162float(kp[lane + j * 32]), sum);
#pragma unroll
      for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
      if (lane == 0) bmax = fmaxf(bmax, sum);
    }
    if (lane == 0) {
      if (seen < 15) {
        sbest[warp][seen] = bmax;
        sid[warp][seen] = block;
        // Seed the minimum from the first admitted candidate: with the
        // FLT_MAX sentinel an all-+inf score set would never lower `worst`,
        // and block 15 would displace block 0 against the lower-id tie
        // contract. The comparison stays strict so later equal scores
        // cannot displace lower ids.
        if (seen == 0 || bmax < worst || (bmax == worst && block > worst_id)) {
          worst = bmax;
          worst_id = block;
          worst_pos = seen;
        }
        ++seen;
      } else if (bmax > worst) {
#pragma unroll
        for (int i = 0; i < 14; ++i)
          if (i >= worst_pos) {
            sbest[warp][i] = sbest[warp][i + 1];
            sid[warp][i] = sid[warp][i + 1];
          }
        sbest[warp][14] = bmax;
        sid[warp][14] = block;
        worst = sbest[warp][0];
        worst_id = sid[warp][0];
        worst_pos = 0;
#pragma unroll
        for (int i = 1; i < 15; ++i) {
          float x = sbest[warp][i];
          int id = sid[warp][i];
          if (x < worst || (x == worst && id > worst_id)) {
            worst = x;
            worst_id = id;
            worst_pos = i;
          }
        }
      }
    }
  }
  if (lane == 0) {
    int ids[16];
#pragma unroll
    for (int i = 0; i < 15; ++i) ids[i] = sid[warp][i];
    ids[15] = own;
#pragma unroll
    for (int i = 1; i < 16; ++i) {
      int x = ids[i], jj = i - 1;
      while (jj >= 0 && ids[jj] > x) {
        ids[jj + 1] = ids[jj];
        --jj;
      }
      ids[jj + 1] = x;
    }
#pragma unroll
    for (int i = 0; i < TOPK; ++i) out[i] = ids[i];
  }
}

// Only launched after an accepted acquire and a rejected scorer launch.
__global__ void release_rejected_workspace() { finish_workspace(); }

// ---------------- host side ----------------
// Only synchronous submission rejection establishes that no consumer ran.
static bool submission_rejected(cudaError_t status) {
  return status == cudaErrorInvalidConfiguration || status == cudaErrorLaunchOutOfResources ||
         status == cudaErrorInvalidValue || status == cudaErrorInvalidDeviceFunction ||
         status == cudaErrorNoKernelImageForDevice;
}

static void check_cuda(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    TVM_FFI_ICHECK(false) << op << ": " << cudaGetErrorString(status);
  }
}
static CUresult make_map(CUtensorMap* tm, const void* ptr, uint64_t rows, uint32_t box_rows) {
  uint64_t gdim[2] = {(uint64_t)DHEAD, rows};
  uint64_t gstr[1] = {(uint64_t)DHEAD * 2ull};
  uint32_t bdim[2] = {64u, box_rows};
  uint32_t bstr[2] = {1u, 1u};
  return cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, gdim, gstr,
                                bdim, bstr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// cudaFuncSetAttribute is per device: key the one-time setup by device id.
static int g_attr_set[64] = {};
static int g_sm_count = 0;
static std::mutex g_launch_mutex;
static bool g_primary_stream_set = false;
static cudaStream_t g_primary_stream = nullptr;

void Run(ffi::TensorView qi, ffi::TensorView ki, ffi::TensorView cu_seqlens,
         ffi::TensorView kv_base, ffi::TensorView seq_lens, ffi::TensorView prefix_lens,
         ffi::TensorView sel) {
  const int TQ = (int)qi.shape()[0];
  const int TKPAD = (int)ki.shape()[0];
  const int B = (int)kv_base.shape()[0];
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id);
  if (TQ <= 0) return;

  std::lock_guard<std::mutex> launch_guard(g_launch_mutex);
  if (!g_primary_stream_set) {
    g_primary_stream = stream;
    g_primary_stream_set = true;
  }
  const bool guarded = stream != g_primary_stream;

  check_cuda(cudaGetLastError(), "CUDA error before fused indexer submission");
  if (g_sm_count == 0) {
    int count;
    check_cuda(
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, qi.device().device_id),
        "cudaDeviceGetAttribute");
    g_sm_count = count;
  }
  const int pgtot = TKPAD / KBLOCK;
  const int ntiles_bound = (TQ + BM - 1) / BM + B - 1;

  if (ntiles_bound > MAXTILES || B < 1) {
    direct_fallback_kernel<<<(TQ + 7) / 8, 256, 0, stream>>>(
        (const __nv_bfloat16*)qi.data_ptr(), (const __nv_bfloat16*)ki.data_ptr(),
        (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
        (const int*)prefix_lens.data_ptr(), (int*)sel.data_ptr(), B, TQ);
    check_cuda(cudaGetLastError(), "direct_fallback_kernel");
    return;
  }

  if (pgtot <= TOPK) {
    fill_kernel<<<(TQ + 7) / 8, 256, 0, stream>>>((const int*)cu_seqlens.data_ptr(),
                                                  (const int*)prefix_lens.data_ptr(),
                                                  (int*)sel.data_ptr(), B, TQ);
    check_cuda(cudaGetLastError(), "fill_kernel");
    return;
  }
  {
    CUtensorMap qmap, kmap;
    const CUresult qres = make_map(&qmap, qi.data_ptr(), (uint64_t)TQ, BMT);
    const CUresult kres = make_map(&kmap, ki.data_ptr(), (uint64_t)TKPAD, BN);
    TVM_FFI_ICHECK(qres == CUDA_SUCCESS && kres == CUDA_SUCCESS)
        << "cuTensorMapEncodeTiled failed (qi: " << (int)qres << ", ki: " << (int)kres
        << "); qi/ki must be 16-byte-aligned contiguous BF16";
    const int smem = NAT * ASZ + STAGES * BSZ;
    const int attr_dev = qi.device().device_id;
    const bool cacheable = attr_dev >= 0 && attr_dev < 64;
    if (!cacheable || !g_attr_set[attr_dev]) {
      check_cuda(cudaFuncSetAttribute(score_kernel<false>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                 "cudaFuncSetAttribute");
      check_cuda(cudaFuncSetAttribute(score_kernel<true>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                 "cudaFuncSetAttribute");
      if (cacheable) g_attr_set[attr_dev] = 1;
    }
    int ncta = g_sm_count > 0 ? g_sm_count : 212;
    // Few query tiles would otherwise give each tile ncta/ntiles chunks to
    // merge; cap the split so the per-tile merge stays short.
    const int cap = ntiles_bound * 32;
    if (cap < ncta) ncta = cap < 32 ? 32 : cap;
    if (ncta > MAXPAIRS - MAXTILES) ncta = MAXPAIRS - MAXTILES;
    if (guarded) {
      acquire_workspace<<<1, 1, 0, stream>>>();
      check_cuda(cudaGetLastError(), "acquire_workspace");
      score_kernel<true><<<ncta, NTHR, smem, stream>>>(
          qmap, kmap, (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
          (const int*)prefix_lens.data_ptr(), (int*)sel.data_ptr(), B, ncta);
    } else {
      score_kernel<false><<<ncta, NTHR, smem, stream>>>(
          qmap, kmap, (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
          (const int*)prefix_lens.data_ptr(), (int*)sel.data_ptr(), B, ncta);
    }
    const cudaError_t status = cudaGetLastError();
    if (submission_rejected(status) && guarded) {
      release_rejected_workspace<<<1, 1, 0, stream>>>();
      const cudaError_t cleanup = cudaGetLastError();
      TVM_FFI_ICHECK(cleanup == cudaSuccess)
          << "fused scorer: " << cudaGetErrorString(status)
          << "; workspace release: " << cudaGetErrorString(cleanup);
    }
    check_cuda(status, "fused scorer");
  }
}

}  // namespace msa_fused
