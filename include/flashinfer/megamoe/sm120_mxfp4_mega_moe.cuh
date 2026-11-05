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
// SM120 (RTX PRO 6000 / 6000D class) MXFP4 MegaMoE for decode-size batches over PCIe P2P, one
// persistent kernel per rank: routing exchange -> dispatch (MXFP8 requantisation + 16 B P2P stores
// into the destination expert pool) -> FC1 + clamped SwiGLU (swap-AB W4A8 mma.sync) -> FC2 -> P2P
// push of each (token, k) partial back to its source rank -> top-k combine. No cross-rank barrier:
// every source writes an epoch-tagged flag into the consumer's memory (st.release.sys) after its
// data, the consumer polls its own memory (ld.acquire.sys); there are no remote atomics.
//
// Compute core: mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale (E2M1 weights as A, E4M3
// activations as B, UE8M0 scales per 32 on both): 16 weight rows x 8 tokens per MMA, weights loaded
// global -> registers in a lane-fragment layout (flashinfer/megamoe/weights.py, SM120 branch). FC1
// rows are interleaved per 16-row block (gate rows 8b..8b+7, then up rows 8b..8b+7), so each SwiGLU
// pair lands in one thread.
//
// Every spin wait times out and records an error code in the counter block instead of hanging the
// GPU; the last CTA to exit resets the per-call counters and advances the epoch (CUDA-graph replay
// safe).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashinfer::megamoe::sm120 {

constexpr int kWarps = 8;
constexpr int kNbPerWarp = 1;                           // 16-row weight blocks per warp per item
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 128 weight rows per work item
constexpr int kTile = 8;                                // tokens per item (MMA N)
constexpr int kPfBytes = 8192;  // per 16-row weight block, prefetched to L2 during dispatch
constexpr size_t kPf0Bytes = size_t(64)
                             << 20;  // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;           // K64 steps in flight per warp
constexpr int kMaxWorld = 8;         // ranks: one PCIe host
constexpr int kMaxEl = 256;          // local experts per rank
constexpr int kMaxTopk = 8;          // routes per token (pool metadata packs token * 8 + k)
constexpr uint64_t kSpinNs =
    30000000000ull;  // spin timeout: startup warm-ups can skew ranks by seconds

// error codes (counter kCtrErr)
constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;

// counters (local, reset by the last CTA to exit)
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4,
              kCtrEpoch = 5, kCtrL1 = 8;
constexpr int kCtrInts = kCtrL1 + kMaxEl;

struct Peers {
  uint8_t* pool_x[kMaxWorld];      // [EL][W][TCAP][H] e4m3            (dest-owned)
  uint32_t* pool_xs[kMaxWorld];    // [EL][W][TCAP][H/128] ue8m0 x4
  int* pool_meta[kMaxWorld];       // [EL][W][TCAP] src token*8 + k
  int* recv_cnt[kMaxWorld];        // [W][EL]
  int* recv_flag[kMaxWorld];       // [W] epoch
  __nv_bfloat16* comb[kMaxWorld];  // [TCAP][TOPK][H]                    (src-owned)
  int* comb_flag[kMaxWorld];       // [W] epoch
  int* route[kMaxWorld];       // [W][1 + TCAP*TOPK]: T, then the source's top-k ids (dest-owned)
  int* route_flag[kMaxWorld];  // [W] epoch
};

struct Args {
  const uint8_t* x;         // [T][H] e4m3, staged by the caller
  const float* x_sf;        // [T][H/128] fp32 scale per 128 channels
  const int64_t* topk_ids;  // [T][TOPK] global expert ids, -1 = masked route
  const float* topk_w;      // [T][TOPK]
  __nv_bfloat16* out;       // [T][H]
  const uint8_t* l1;        // [EL][l1_stride]: packed W13 (l1_w_bytes) then its scale words
  const uint8_t* l2;        // [EL][l2_stride]: packed W2 (l2_w_bytes) then its scale words
  size_t l1_stride, l1_w_bytes, l2_stride, l2_w_bytes;
  const float *gs1, *gs2;      // optional [EL] per-expert FC1 / FC2 output multipliers
  int* stats;                  // optional [EL] cumulative received-token counts
  uint8_t* inter;              // [EL][W*TCAP][I] e4m3 (local)
  uint32_t* inter_s;           // [EL][W*TCAP][I/128]
  int* ctr;                    // kCtrInts
  unsigned long long* stamps;  // optional [grid][16] globaltimer stamps (debug)
  Peers peers;
  int T, H, I, EL, W, TOPK, TCAP, rank;
  float swiglu_limit;
};

__device__ __forceinline__ const uint4* w13_of(const Args& a, int el) {
  return reinterpret_cast<const uint4*>(a.l1 + size_t(el) * a.l1_stride);
}
__device__ __forceinline__ const uint32_t* ws13_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l1 + size_t(el) * a.l1_stride + a.l1_w_bytes);
}
__device__ __forceinline__ const uint4* w2_of(const Args& a, int el) {
  return reinterpret_cast<const uint4*>(a.l2 + size_t(el) * a.l2_stride);
}
__device__ __forceinline__ const uint32_t* ws2_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l2 + size_t(el) * a.l2_stride + a.l2_w_bytes);
}

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// thread 0 spins until *p >= v (sys or gpu scope); returns false on timeout
__device__ __forceinline__ bool wait_ge(const int* p, int v, bool sys) {
  const uint64_t t0 = gtime();
  while ((sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) < v)
    if (gtime() - t0 > kSpinNs) return false;
  return true;
}

__device__ __forceinline__ void mma_w4a8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1, uint32_t sfa, uint16_t byte_a, uint32_t sfb,
                                         uint16_t byte_b) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3."
      "f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(byte_a),
        "h"(uint16_t{0}), "r"(sfb), "h"(byte_b), "h"(uint16_t{0}));
}
__device__ __forceinline__ void expand(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = (w & 0x0F0F0F0Fu) << 2;
  hi = (w >> 2) & 0x3C3C3C3Cu;
}

// warp GEMM over one 256-row item: acc[j] = rows (nb0+j)*16 + {g, g+8}, cols = tokens 2c, 2c+1.
// xrow: this lane's token row (e4m3, K bytes), xsrow: its ue8m0 x4 words (K/128).
// [kc_begin, kc_end) restricts K to a range of K64 chunks (split-K); both multiples of kUnroll (and
// of 2 for scales).
__device__ __forceinline__ void item_gemm(float (&acc)[kNbPerWarp][4], const uint4* W,
                                          const uint32_t* WS, int nblocks, int nb0, int K,
                                          const uint8_t* xrow, const uint32_t* xsrow, int lane,
                                          int kc_begin = 0, int kc_end = -1) {
  const int g = lane / 4, c = lane % 4;
  const int kch = K / 64, kq = K / 128;
  const uint4* wp[kNbPerWarp];
  const uint32_t* sp[kNbPerWarp];
#pragma unroll
  for (int j = 0; j < kNbPerWarp; ++j) {
    wp[j] = W + size_t(nb0 + j) * kch * 32 + lane;
    sp[j] = WS + size_t(nb0 + j) * kq * 16 + g + (lane & 1) * 8;
  }
  const uint32_t* xp = reinterpret_cast<const uint32_t*>(xrow) + c;
#pragma unroll
  for (int j = 0; j < kNbPerWarp; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  if (kc_end < 0) kc_end = kch;
  for (int k0 = kc_begin; k0 < kc_end; k0 += kUnroll) {
    uint4 wv[kNbPerWarp][kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u)
#pragma unroll
      for (int j = 0; j < kNbPerWarp; ++j) wv[j][u] = __ldcs(wp[j] + (k0 + u) * 32);
    uint32_t sfa[kNbPerWarp][(kUnroll + 1) / 2];
#pragma unroll
    for (int q = 0; q < (kUnroll + 1) / 2; ++q)
#pragma unroll
      for (int j = 0; j < kNbPerWarp; ++j) sfa[j][q] = __ldg(sp[j] + (k0 / 2 + q) * 16);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int kc = k0 + u;
      const uint32_t sfb = xsrow[kc / 2];
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int step = kc * 2 + s;
        const uint32_t b0 = xp[step * 8];
        const uint32_t b1 = xp[step * 8 + 4];
        const uint16_t sel = uint16_t(step & 3);
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) {
          const uint32_t w0 = s ? wv[j][u].z : wv[j][u].x;
          const uint32_t w1 = s ? wv[j][u].w : wv[j][u].y;
          uint32_t a[4];
          expand(w0, a[0], a[1]);
          expand(w1, a[2], a[3]);
          mma_w4a8(acc[j], a, b0, b1, sfa[j][u / 2], sel, sfb, sel);
        }
      }
    }
  }
}

// quantize 32 consecutive fp32 values (in registers) to e4m3 with one ue8m0 scale; returns packed
// bytes + scale byte
__device__ __forceinline__ uint8_t quant32(const float (&v)[32], uint32_t (&q)[8]) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  int e = amax > 0.f ? int(ceilf(log2f(amax / 448.f))) : -127;
  e = max(-127, min(127, e));
  const float inv = exp2f(float(-e));
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    uint32_t word = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const __nv_fp8_storage_t f =
          __nv_cvt_float_to_fp8(v[w * 4 + b] * inv, __NV_SATFINITE, __NV_E4M3);
      word |= uint32_t(f) << (8 * b);
    }
    q[w] = word;
  }
  return uint8_t(e + 127);
}

struct Map {  // per-expert token bookkeeping, built identically by every CTA after the dispatch
              // flags
  int cnt[kMaxWorld][kMaxEl];
  int ne[kMaxEl];
  int tiles_prefix[kMaxEl + 1];
};

// Top-k id of (token t, slot k) as every phase reads it: -1 for a masked route, an id outside
// [0, W * EL), or a repeat of an earlier slot of the same token, so a token reaches each expert at
// most once and its dispatch slot stays below TCAP
__device__ __forceinline__ int route_at(const Args& a, int t, int k) {
  const int64_t* ids = a.topk_ids + size_t(t) * a.TOPK;
  const int64_t id = ids[k];
  if (id < 0 || id >= int64_t(a.W) * a.EL) return -1;
  for (int j = 0; j < k; ++j)
    if (ids[j] == id) return -1;
  return int(id);
}

__device__ __forceinline__ void token_src(const Map& m, int el, int j, int W, int& w, int& i) {
  w = 0;
  i = j;
  while (w < W - 1 && i >= m.cnt[w][el]) {
    i -= m.cnt[w][el];
    ++w;
  }
}

__global__ void __launch_bounds__(kWarps * 32) sm120_mega_moe_kernel(Args a) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
  __shared__ Map m;
  __shared__ int s_item, s_ok, s_epoch;
  // L1: [tok][kRowsPerItem/2 inter] / L2: [tok][kRowsPerItem hidden] / dispatch: one MXFP8 token
  // row (H + H/32 bytes)
  __shared__ __align__(16) float s_tile[kTile][kRowsPerItem < 256 ? 256 : kRowsPerItem];

  unsigned long long st_[16] = {
      gtime()};  // +8 route-push 9 route-ok 10 map 11 pf 12 flag-sent 13 last FC1 end 14 first FC2
                 // start 15 comb-flag release  // 0 start 1 pushed 2 flags 3 work end 4 comb flags
                 // 5 end 6 L1-wait ns 7 items
  if (threadIdx.x == 0) s_epoch = *reinterpret_cast<volatile int*>(a.ctr + kCtrEpoch) + 1;
  // the error word reports this call only (a wait records one no earlier than kSpinNs from now)
  if (blockIdx.x == 0 && threadIdx.x == 0) a.ctr[kCtrErr] = 0;
  __syncthreads();
  const int epoch = s_epoch;
  int* ctr = a.ctr;

  // ---------------- routing exchange first (tiny): every rank learns the full expert load before
  // the rows land
  const int RS = 1 + TCAP * TOPK;
  if (blockIdx.x == (a.T < int(gridDim.x) ? a.T : int(gridDim.x) - 1)) {
    for (int u = threadIdx.x; u < W * (1 + a.T * TOPK); u += blockDim.x) {
      const int dst = u / (1 + a.T * TOPK), v = u % (1 + a.T * TOPK);
      a.peers.route[dst][a.rank * RS + v] =
          v == 0 ? a.T : route_at(a, (v - 1) / TOPK, (v - 1) % TOPK);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
      for (int dst = 0; dst < W; ++dst) st_release_sys(a.peers.route_flag[dst] + a.rank, epoch);
    st_[8] = gtime();
  }

  // ---------------- t=0: this rank's own tokens already tell which local experts are active; pull
  // the head of their FC1 weights into L2 while the rows and the other ranks' routing are in flight
  if (kPf0Bytes > 0 && EL <= 64) {
    __shared__ unsigned long long s_known;
    if (threadIdx.x == 0) s_known = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0 && e / EL == a.rank) atomicOr(&s_known, 1ull << (e % EL));
    }
    __syncthreads();
    const unsigned long long known = s_known;
    const int nk = __popcll(known);
    if (nk > 0) {
      const size_t per = min(a.l1_w_bytes, kPf0Bytes / nk) & ~size_t(127);
      const size_t lines = per / 128 * nk;
      for (size_t l = size_t(blockIdx.x) * blockDim.x + threadIdx.x; l < lines;
           l += size_t(gridDim.x) * blockDim.x) {
        const int ki = int(l / (per / 128));
        unsigned long long rest = known;
        for (int q = 0; q < ki; ++q) rest &= rest - 1;
        const int el = __ffsll(rest) - 1;
        const char* p = reinterpret_cast<const char*>(w13_of(a, el)) + (l % (per / 128)) * 128;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
    }
  }

  // ---------------- dispatch: one CTA per token: quantize once to MXFP8 in smem, push TOPK copies
  // with every thread
  const int nitems = a.T * TOPK;
  {
    uint8_t* s_q = reinterpret_cast<uint8_t*>(
        &s_tile[0][0]);  // [H] e4m3, then [H/32] ue8m0 (H + H/32 <= smem)
    uint8_t* s_sc = s_q + H;
    __shared__ int s_rowk[8], s_dstk[8];
    for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
      if (threadIdx.x <
          H / 32) {  // one thread per K32 group: dequantise the staged row, requantise to MXFP8
        const uint4* s4 = reinterpret_cast<const uint4*>(a.x + size_t(t) * H + threadIdx.x * 32);
        const float sf = a.x_sf[size_t(t) * (H / 128) + threadIdx.x / 4];
        float v[32];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const uint4 u = s4[q];
          const __nv_fp8_storage_t* f8 = reinterpret_cast<const __nv_fp8_storage_t*>(&u);
#pragma unroll
          for (int r = 0; r < 16; ++r) {
            const __half_raw hr = __nv_cvt_fp8_to_halfraw(f8[r], __NV_E4M3);
            v[q * 16 + r] = __half2float(__half(hr)) * sf;
          }
        }
        uint32_t q8[8];
        s_sc[threadIdx.x] = quant32(v, q8);
        uint4* d4 = reinterpret_cast<uint4*>(s_q + threadIdx.x * 32);
        d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
        d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
      }
      if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t' < t routed to the
                          // same expert)
        const int e = route_at(a, t, warp);  // < 0: masked route, nothing to send
        int slot = 0;
        for (int u = lane; u < t * TOPK; u += 32) slot += (route_at(a, u / TOPK, u % TOPK) == e);
#pragma unroll
        for (int o = 16; o; o >>= 1) slot += __shfl_xor_sync(0xffffffffu, slot, o);
        if (lane == 0) {
          if (e < 0) {
            s_rowk[warp] = -1;
          } else {
            const int dst = e / EL, el = e % EL;
            const int row = (el * W + a.rank) * TCAP + slot;
            s_rowk[warp] = row;
            s_dstk[warp] = dst;
            a.peers.pool_meta[dst][row] = t * 8 + warp;
          }
        }
      }
      __syncthreads();
      // per copy: H / 16 uint4 of data, then the H / 32 scale bytes as H / 256 uint2 (a scale row
      // is only 8-byte aligned when hidden % 512 != 0)
      const int per = H / 16 + H / 256;
      for (int u = threadIdx.x; u < TOPK * per; u += blockDim.x) {
        const int k = u / per, q = u % per;
        if (s_rowk[k] < 0) continue;
        const size_t row = size_t(s_rowk[k]);
        const int dst = s_dstk[k];
        if (q < H / 16)
          reinterpret_cast<uint4*>(a.peers.pool_x[dst] + row * H)[q] =
              reinterpret_cast<const uint4*>(s_q)[q];
        else
          reinterpret_cast<uint2*>(a.peers.pool_xs[dst] + row * (H / 128))[q - H / 16] =
              reinterpret_cast<const uint2*>(s_sc)[q - H / 16];
      }
      __syncthreads();
    }
  }
  __threadfence_system();
  __syncthreads();
  st_[1] = gtime();
  if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
  __syncthreads();
  if (s_item && threadIdx.x ==
                    0) {  // last CTA out of the dispatch: every row of this rank is visible -> flag
    __threadfence_system();
    for (int dst = 0; dst < W; ++dst) st_release_sys(a.peers.recv_flag[dst] + a.rank, epoch);
    st_[12] = gtime();
  }
  __syncthreads();

  // ---------------- per-expert map from the exchanged routing (same slot order as the dispatch)
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w)
      if (!wait_ge(a.peers.route_flag[a.rank] + w, epoch, true)) s_ok = 0;
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrRouteWait);
  }
  __syncthreads();
  st_[9] = gtime();
  if (!s_ok) goto exit_kernel;
  {  // every source's routing in one parallel pass (all loads in flight together), then a warp scan
     // for the prefix
    __shared__ int s_T[kMaxWorld];
    const int* rb = a.peers.route[a.rank];
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = 0;
    if (threadIdx.x < W) s_T[threadIdx.x] = __ldcv(rb + threadIdx.x * RS);
    __syncthreads();
    constexpr int kPer = 8;  // entries per thread per pass
    for (int base = 0; base < W * RS; base += kPer * int(blockDim.x)) {
      int e[kPer];
#pragma unroll
      for (int q = 0; q < kPer; ++q) {
        const int u = base + q * blockDim.x + threadIdx.x;
        const int w = u / RS, v = u % RS;
        e[q] = (u < W * RS && v > 0 && v - 1 < s_T[w] * TOPK) ? __ldcv(rb + u)
                                                              : -1;  // -1 also = padding id
      }
#pragma unroll
      for (int q = 0; q < kPer; ++q) {
        const int u = base + q * blockDim.x + threadIdx.x;
        if (e[q] >= 0 && e[q] / EL == a.rank)
          atomicAdd(&m.cnt[u / RS][e[q] % EL], 1);  // e < 0: padding
      }
    }
  }
  __syncthreads();
  if (warp == 0) {
    int tiles_sum = 0;
    for (int base = 0; base < EL; base += 32) {
      const int el = base + lane;
      int n = 0;
      if (el < EL)
        for (int w = 0; w < W; ++w) n += m.cnt[w][el];
      int incl = (n + kTile - 1) / kTile;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
      }
      if (el < EL) {
        m.ne[el] = n;
        m.tiles_prefix[el + 1] = tiles_sum + incl;
      }
      tiles_sum += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (lane == 0) m.tiles_prefix[0] = 0;
  }
  __syncthreads();
  if (a.stats && blockIdx.x == 0)
    for (int el = threadIdx.x; el < EL; el += blockDim.x) a.stats[el] += m.ne[el];
  st_[10] = gtime();
  {  // this CTA's first FC1 item is static (item = blockIdx.x): start pulling its weights into L2
     // now
    const int nblk1 = 2 * I / kRowsPerItem;
    const int item = blockIdx.x;
    if (item < m.tiles_prefix[EL] * nblk1) {
      const int tile_g = item / nblk1, rb = item % nblk1;
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      const int blk_bytes = (H / 64) * 32 * 16;
#pragma unroll
      for (int jb = 0; jb < kNbPerWarp; ++jb) {
        const char* base =
            reinterpret_cast<const char*>(w13_of(a, el) + size_t(nb0 + jb) * (H / 64) * 32);
        for (int off = lane * 128; off < min(kPfBytes, blk_bytes); off += 32 * 128)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(base + off));
      }
    }
  }
  st_[11] = gtime();

  // ---------------- wait for every source rank, build the per-expert map
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w)
      if (!wait_ge(a.peers.recv_flag[a.rank] + w, epoch, true)) s_ok = 0;
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrDispatchWait);
  }
  __syncthreads();
  st_[2] = gtime();
  if (!s_ok) goto exit_kernel;
  {
    const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
    const int ntiles = m.tiles_prefix[EL];
    const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
    const uint8_t* pool_x = a.peers.pool_x[a.rank];
    const uint32_t* pool_xs = a.peers.pool_xs[a.rank];
    const int* pool_meta = a.peers.pool_meta[a.rank];
    const int g = lane / 4, c = lane % 4;
    const int TOKE = W * TCAP;

    bool first = true;
    while (true) {
      if (threadIdx.x == 0)
        s_item = first ? int(blockIdx.x) : int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
      first = false;
      __syncthreads();
      const int item = s_item;
      if (item >= n1 + n2) break;
      st_[7] += 1;
      const bool l1 = item < n1;
      const int li = l1 ? item : item - n1;
      const int nblk = l1 ? nblk1 : nblk2;
      const int tile_g = li / nblk, rb = li % nblk;
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int tile = tile_g - m.tiles_prefix[el];
      const int ntok = min(kTile, m.ne[el] - tile * kTile);
      const int j = tile * kTile + (g < ntok ? g : 0);  // this lane's token (B column g)

      if (!l1) {  // FC2 needs every FC1 item of this expert
        const unsigned long long tw0 = gtime();
        if (threadIdx.x == 0) {
          const int need = ((m.ne[el] + kTile - 1) / kTile) * nblk1;
          if (!wait_ge(ctr + kCtrL1 + el, need, false)) {
            atomicExch(ctr + kCtrErr, kErrL1Wait);
            s_ok = 0;
          }
        }
        __syncthreads();
        st_[6] += gtime() - tw0;
        if (!st_[14]) st_[14] = gtime();
        if (!s_ok) goto exit_kernel;
      }

      float acc[kNbPerWarp][4];
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      if (l1) {
        int w, i;
        token_src(m, el, j, W, w, i);
        const size_t row = (size_t(el) * W + w) * TCAP + i;
        item_gemm(acc, w13_of(a, el), ws13_of(a, el), 2 * I / 16, nb0, H, pool_x + row * H,
                  pool_xs + row * (H / 128), lane);
        // SwiGLU: acc rows g (gate) / g+8 (up) of block nb -> inter col nb*8+g; tokens 2c, 2c+1
        constexpr int kItemInter = kRowsPerItem / 2;  // inter cols per item (multiple of 32)
        constexpr int kGroups = kItemInter / 32;
        const float g1 = a.gs1 ? a.gs1[el] : 1.f;
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 8 + g;  // 0..kItemInter-1 within the item
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float gate = fminf(acc[jb][h] * g1, a.swiglu_limit);
            const float up = fminf(fmaxf(acc[jb][2 + h] * g1, -a.swiglu_limit), a.swiglu_limit);
            s_tile[2 * c + h][col] = gate / (1.f + __expf(-gate)) * up;
          }
        }
        __syncthreads();
        if (threadIdx.x < kTile * kGroups) {  // one thread per (token, K32 group)
          const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
          float v[32];
#pragma unroll
          for (int u = 0; u < 32; ++u) v[u] = s_tile[tk][gi * 32 + u];
          uint32_t q8[8];
          const uint8_t sc = quant32(v, q8);
          if (tk < ntok) {
            const size_t r = size_t(el) * TOKE + tile * kTile + tk;
            const int col0 = rb * kItemInter + gi * 32;
            uint4* d4 = reinterpret_cast<uint4*>(a.inter + r * I + col0);
            d4[0] = make_uint4(q8[0], q8[1], q8[2], q8[3]);
            d4[1] = make_uint4(q8[4], q8[5], q8[6], q8[7]);
            reinterpret_cast<uint8_t*>(a.inter_s + r * (I / 128))[col0 / 32] = sc;
          }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(ctr + kCtrL1 + el, 1);
        st_[13] = gtime();
      } else {
        const size_t r = size_t(el) * TOKE + j;
        item_gemm(acc, w2_of(a, el), ws2_of(a, el), H / 16, nb0, I, a.inter + r * I,
                  a.inter_s + r * (I / 128), lane);
        const float g2 = a.gs2 ? a.gs2[el] : 1.f;
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 16 + g;  // 0..kRowsPerItem-1 within the item
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            s_tile[2 * c + h][col] = acc[jb][h] * g2;
            s_tile[2 * c + h][col + 8] = acc[jb][2 + h] * g2;
          }
        }
        __syncthreads();
        // push: token tk of this tile -> its source rank's combine slot, kRowsPerItem bf16 in 16 B
        // stores
        constexpr int kVec = kRowsPerItem / 8;
        for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
          const int tk = u / kVec, q = u % kVec;
          int w, i;
          token_src(m, el, tile * kTile + tk, W, w, i);
          const int meta = pool_meta[(size_t(el) * W + w) * TCAP + i];
          const int st = meta / 8, sk = meta % 8;
          uint32_t pk[4];
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const __nv_bfloat162 b2 =
                __floats2bfloat162_rn(s_tile[tk][q * 8 + 2 * p], s_tile[tk][q * 8 + 2 * p + 1]);
            pk[p] = *reinterpret_cast<const uint32_t*>(&b2);
          }
          __nv_bfloat16* dst =
              a.peers.comb[w] + (size_t(st) * TOPK + sk) * H + rb * kRowsPerItem + q * 8;
          *reinterpret_cast<uint4*>(dst) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          const int done = atomicAdd(ctr + kCtrL2Done, 1);
          if (done == n2 - 1) {  // this rank's last FC2 item: tell every source rank
            __threadfence_system();
            for (int w = 0; w < W; ++w) st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
            st_[15] = gtime();
          }
        }
      }
      __syncthreads();
    }
    st_[3] = gtime();
    if (n2 == 0 && blockIdx.x == 0 &&
        threadIdx.x == 0) {  // no local work: still release every source
      __threadfence_system();
      for (int w = 0; w < W; ++w) st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
    }
  }

  // ---------------- combine: wait for every expert rank, out[t] = sum_k w[t][k] * partial[t][k]
  if (threadIdx.x == 0) {
    s_ok = 1;
    for (int w = 0; w < W; ++w)
      if (!wait_ge(a.peers.comb_flag[a.rank] + w, epoch, true)) s_ok = 0;
    if (!s_ok) atomicExch(ctr + kCtrErr, kErrCombineWait);
  }
  __syncthreads();
  st_[4] = gtime();
  if (s_ok) {
    const __nv_bfloat16* comb = a.peers.comb[a.rank];
    const int nvec = a.T * H / 8;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nvec; v += gridDim.x * blockDim.x) {
      const int t = v / (H / 8), h8 = (v % (H / 8)) * 8;
      float s[8] = {};
      for (int k = 0; k < TOPK; ++k) {
        if (route_at(a, t, k) < 0) continue;  // masked route: never dispatched
        const float wk = a.topk_w[t * TOPK + k];
        const uint4 u = *reinterpret_cast<const uint4*>(comb + (size_t(t) * TOPK + k) * H + h8);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float2 f = __bfloat1622float2(b2[r]);
          s[2 * r] += wk * f.x;
          s[2 * r + 1] += wk * f.y;
        }
      }
      uint32_t pk[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const __nv_bfloat162 b2 = __floats2bfloat162_rn(s[2 * r], s[2 * r + 1]);
        pk[r] = *reinterpret_cast<const uint32_t*>(&b2);
      }
      *reinterpret_cast<uint4*>(a.out + size_t(t) * H + h8) =
          make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }

exit_kernel:
  __syncthreads();
  st_[5] = gtime();
  if (a.stamps && threadIdx.x == 0)
    for (int u = 0; u < 16; ++u) a.stamps[blockIdx.x * 16 + u] = st_[u];
  if (threadIdx.x == 0) {
    __threadfence();
    const int done = atomicAdd(ctr + kCtrExit, 1);
    if (done ==
        int(gridDim.x) - 1) {  // last CTA out: advance the epoch, reset the per-call counters
      ctr[kCtrEpoch] = epoch;
      ctr[kCtrDispatch] = 0;
      ctr[kCtrQueue] = 0;
      ctr[kCtrL2Done] = 0;
      for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
      __threadfence();
      ctr[kCtrExit] = 0;
    }
  }
}

}  // namespace flashinfer::megamoe::sm120
