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
// persistent kernel per rank: routing exchange -> dispatch (the staged MXFP8 rows, 16 B P2P stores
// into the destination expert pool) -> FC1 + clamped SwiGLU (swap-AB W4A8 mma.sync) -> FC2 -> P2P
// push of each (token, k) partial back to its source rank -> top-k combine. No cross-rank barrier:
// every source writes an epoch-tagged flag into the consumer's memory (st.release.sys) after its
// data, the consumer polls its own memory (ld.acquire.sys); there are no remote atomics.
//
// Compute core: mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale (E2M1 weights as A, E4M3
// activations as B, UE8M0 scales per 32 on both): 16 weight rows x 8 tokens per MMA, weights loaded
// global -> registers from the row-major packed E2M1 layout the sibling moe_ep SM120 backends
// (cake, cutedsl) use, so a layer can share one weight copy with them: [rows][K/2] bytes, element
// 2i in the low nibble of byte i, FC1 rows interleaved per 16-row block (gate rows 8b..8b+7, then
// up rows 8b..8b+7) so each SwiGLU pair lands in one thread. Within each 32-wide scale block the K
// order is permuted (MMA k = 16p + 4c + j reads element 8c + 4p + j, for both operands), so lane c
// needs weight bytes 4c..4c+3 of each row and block and activation bytes 8c..8c+7. Each lane loads
// 16 contiguous bytes (one block) of a row per K128, coalesced, and the four lanes of a quad
// exchange words with a 4x4 shuffle transpose. The UE8M0 weight scales are a separate per-expert
// array (mxfp4_mxfp8_bf16_nosync/weights.py).
//
// Every spin wait times out and records an error code in the counter block instead of hanging the
// GPU; the last CTA to exit resets the per-call counters and advances the epoch (CUDA-graph replay
// safe). The epoch is a wrapping 32-bit call number compared modulo 2^32 (wait_ge), so a flag one
// call old reads as behind after the wrap too; the GIN routing signal, a 64-bit count of calls,
// is compared with a 64-bit call count (kCtrCalls).
//
// NIC transport (built with FLASHINFER_MEGAMOE_SM120_GIN=1, NCCL Device API): ranks in Gin::mask
// are reached with NCCL GIN puts instead of P2P stores, for hosts where a pair's PCIe peer path is
// slow (e.g. across CPU sockets). The buffer is then one symmetric NCCL window; a GIN payload is
// first written to a local staging region of the window, then put to the peer's receive region.
// A token's MXFP8 row goes once to a GIN rank's inbox however many of that rank's experts it routes
// to: the rows for one GIN rank are staged packed (the source's tokens routed there, in token
// order) and put once after the dispatch (one put per token was bound by the NIC's message rate
// with many GIN ranks), with a table mapping the rank's pool slots (packed in (local expert, slot)
// order) to the packed rows. The table ends where the rows start (gin_block), so one put carries
// both. Partials come back one put per run of finished rows of an FC2 tile, packed in the same
// order, so a tile's rows are contiguous. The receiver counts GIN signals per source instead of
// polling a flag: the routing put adds 1, the rows put its token rows + 1, each pushed-back
// partial row 1 and a final per-call "done" 1. Signal values are cumulative; the expected totals
// per source live in the counter block and advance at exit.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef FLASHINFER_MEGAMOE_SM120_GIN
#define FLASHINFER_MEGAMOE_SM120_GIN 0
#endif
// GIN code is device-pass only: NCCL's device headers do not compile in nvcc's host pass, and the
// kernel's signature carries no NCCL type
#if FLASHINFER_MEGAMOE_SM120_GIN
// the proxy backend's device code names cuda:: types that the host stub pass also sees
#include <cuda/atomic>
#endif
#if FLASHINFER_MEGAMOE_SM120_GIN && defined(__CUDA_ARCH__)
#define MEGAMOE_SM120_GIN_DEVICE 1
#include <nccl_device.h>
#else
#define MEGAMOE_SM120_GIN_DEVICE 0
#endif

namespace flashinfer::megamoe::sm120 {

constexpr int kWarps = 8;
constexpr int kNbPerWarp = 1;                           // 16-row weight blocks per warp per item
constexpr int kRowsPerItem = kWarps * kNbPerWarp * 16;  // 128 weight rows per work item
constexpr int kTile = 8;  // tokens per MMA (N); a work item covers kTileN = 8, 16, 24 or 32 tokens
// row-major weights: each warp also prefetches its 16 rows this many bytes ahead of its loads, so
// DRAM sees longer runs per row than 16 interleaved 64-byte reads (RTX 6000D, cold L2: 128 B ahead
// was best of 0 / 128 / 256 / 512 / 1024 B; without it the shared layout was 25-30% slower)
constexpr int kPfAhead = 128;
constexpr size_t kPf0Bytes = size_t(64)
                             << 20;  // t=0 L2 prefetch budget over the locally known experts
constexpr int kUnroll = 4;           // K64 steps in flight per warp
// ranks: the default build takes one PCIe host (8 GPUs); groups across hosts load a build with
// FLASHINFER_MEGAMOE_SM120_MAX_WORLD=32 (jit/megamoe.py). Map::cnt keeps 2048 entries either way
// (the static shared memory next to s_tile), so a 32-rank build takes 64 local experts per rank.
#ifndef FLASHINFER_MEGAMOE_SM120_MAX_WORLD
#define FLASHINFER_MEGAMOE_SM120_MAX_WORLD 8
#endif
constexpr int kMaxWorld = FLASHINFER_MEGAMOE_SM120_MAX_WORLD;
static_assert(kMaxWorld == 8 || kMaxWorld == 32, "builds for up to 8 and up to 32 ranks");
constexpr int kMaxEl = 2048 / kMaxWorld;  // local experts per rank: 256, or 64 in the 32-rank build
constexpr int kMaxTopk = 8;               // routes per token (pool metadata packs token * 8 + k)
constexpr int kMaxTokens = 128;           // tokens per rank (the workspace capacity)
constexpr uint64_t kSpinNs =
    30000000000ull;  // spin timeout: startup warm-ups can skew ranks by seconds

// an expert-pool row: the MXFP8 token (H bytes), its UE8M0 scales (H/32 bytes) and the source's
// (token * 8 + k), padded to 16 bytes, so a dispatched copy is one contiguous store / put
__host__ __device__ constexpr size_t pool_row_bytes(int H) {
  return size_t(H) + ((size_t(H) / 32 + sizeof(int) + 15) & ~size_t(15));
}
// GIN block per rank of stage_row / inbox: the slot table ([TCAP*TOPK] ints, 16-byte aligned),
// then TCAP token rows. A call's table of n entries is the slot_words(n) ints right before the rows
__host__ __device__ constexpr size_t gin_slot_bytes(int TCAP, int TOPK) {
  return (size_t(TCAP) * TOPK * sizeof(int) + 15) & ~size_t(15);
}
__host__ __device__ constexpr size_t gin_block(int TCAP, int TOPK, int H) {
  return gin_slot_bytes(TCAP, TOPK) + size_t(TCAP) * pool_row_bytes(H);
}
__host__ __device__ constexpr int slot_words(int n) { return n > 4 ? (n + 3) & ~3 : 4; }

// error codes (counter kCtrErr)
constexpr int kErrDispatchWait = 1, kErrL1Wait = 2, kErrCombineWait = 3, kErrRouteWait = 4;

// counters (local, reset by the last CTA to exit; the epoch and the call count persist)
constexpr int kCtrDispatch = 0, kCtrQueue = 1, kCtrL2Done = 2, kCtrErr = 3, kCtrExit = 4,
              kCtrEpoch = 5, kCtrCalls = 6, kCtrFlag = 8, kCtrL1 = 12;
// kCtrFlag + kind: the epoch of the last call whose routing / rows / partials from every rank
// arrived (one CTA polls the peers and republishes here; the others wait on this local word)
constexpr int kFlagRoute = 0, kFlagRows = 1, kFlagComb = 2;
static_assert(kCtrCalls % 2 == 0, "the u64 call count must be 8-byte aligned");
// GIN transport state: per-tile FC2 completion counts, this call's expected GIN signal counts per
// source (dispatch, combine) and the cumulative totals (u64, persistent across calls)
constexpr int kMaxTiles =
    kMaxEl + kMaxWorld * kMaxTokens * kMaxTopk / 8;  // sum_el ceil(n_el / kTile)
constexpr int kCtrTile = kCtrL1 + kMaxEl, kCtrCnt = kCtrTile + kMaxTiles,
              kCtrCum = kCtrCnt + 2 * kMaxWorld;
constexpr int kCtrInts = kCtrCum + 2 * 2 * kMaxWorld;
static_assert(kCtrCum % 2 == 0, "cumulative GIN totals must be 8-byte aligned");
// GIN signal ids at the receiver, per source rank
constexpr int kSigRoute = 0, kSigDispatch = kMaxWorld, kSigComb = 2 * kMaxWorld,
              kGinSignals = 3 * kMaxWorld;

struct Peers {
  uint8_t* pool[kMaxWorld];        // [EL][W][TCAP] rows of pool_row_bytes(H)   (dest-owned)
  int* recv_cnt[kMaxWorld];        // [W][EL]
  int* recv_flag[kMaxWorld];       // [W] epoch
  __nv_bfloat16* comb[kMaxWorld];  // [TCAP][TOPK][H]                    (src-owned)
  int* comb_flag[kMaxWorld];       // [W] epoch
  int* route[kMaxWorld];       // [2][W][EL + 1]: the source's routes to each local expert, then its
                               // tokens routed here, by epoch parity
                               // (dest-owned)
  int* route_flag[kMaxWorld];  // [W] epoch
};

// NIC transport: window offsets of the receive regions (identical on every rank) and of this rank's
// staging regions, plus their local pointers. mask == 0 (always in the P2P build): no GIN pair.
struct Gin {
  const void* comm;  // ncclDevComm const*, device memory
  void* win;         // ncclWindow_t of the whole buffer
  uint32_t mask;     // bit w: rank w is reached over GIN
  int nctx;          // GIN contexts: CTA b posts on context b % nctx; a signal is the sum over them
  size_t off_route, off_inbox, off_comb_gin, off_stage_row, off_stage_comb, off_stage_route;
  uint8_t* stage_row;  // [W] gin_block: per GIN rank, its slot table, then the tokens routed there
  uint8_t* inbox;      // [W] gin_block: the same, received from each GIN source
  __nv_bfloat16* stage_comb;  // [W][TCAP*TOPK][H]: partials per GIN source, packed (el, slot)
  __nv_bfloat16* comb_gin;    // [W][TCAP*TOPK][H]: partials from each GIN expert rank, packed
  int* cidx;                  // [TCAP*TOPK]: packed row of (token, k) sent to a GIN rank
  int* stage_route;           // [W][EL + 1]: the routing counts sent to each GIN rank
};

struct Args {
  const uint8_t* x;         // [T][H] e4m3, staged by the caller
  const uint8_t* x_sf;      // [T][H/32] UE8M0 scale per 32 channels
  const int64_t* topk_ids;  // [T][TOPK] global expert ids, -1 = masked route
  const float* topk_w;      // [T][TOPK]
  __nv_bfloat16* out;       // [T][H]
  const uint8_t* l1;        // [EL][2I][H/2] packed E2M1 W13, row-major, gate/up rows interleaved
  const uint8_t* l2;        // [EL][H][I/2] packed E2M1 W2, row-major
  const uint8_t* l1s;       // [EL][2I/16][H/128][16] UE8M0 x4 words of W13
  const uint8_t* l2s;       // [EL][H/16][I/128][16] UE8M0 x4 words of W2
  size_t l1_stride, l2_stride, l1s_stride, l2s_stride;  // bytes per expert
  uint8_t* inter;                                       // [EL][W*TCAP][I] e4m3 (local)
  uint32_t* inter_s;                                    // [EL][W*TCAP][I/128]
  int* ctr;                                             // kCtrInts
  unsigned long long* stamps;  // optional [grid][16] globaltimer stamps (debug)
  Peers peers;
  Gin gin;
  int T, H, I, EL, W, TOPK, TCAP, rank;
  float swiglu_limit;
};

__device__ __forceinline__ const uint8_t* w13_of(const Args& a, int el) {
  return a.l1 + size_t(el) * a.l1_stride;
}
__device__ __forceinline__ const uint32_t* ws13_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l1s + size_t(el) * a.l1s_stride);
}
__device__ __forceinline__ const uint8_t* w2_of(const Args& a, int el) {
  return a.l2 + size_t(el) * a.l2_stride;
}
__device__ __forceinline__ const uint32_t* ws2_of(const Args& a, int el) {
  return reinterpret_cast<const uint32_t*>(a.l2s + size_t(el) * a.l2s_stride);
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
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// thread 0 spins until *p >= v modulo 2^32 (the difference read as signed, so it holds across the
// epoch's wrap; counts within a call are far below 2^31), sys or gpu scope; false on timeout
__device__ __forceinline__ bool wait_ge(const int* p, int v, bool sys) {
  const uint64_t t0 = gtime();
  while (static_cast<int>(static_cast<unsigned>(sys ? ld_acquire_sys(p) : ld_acquire_gpu(p)) -
                          static_cast<unsigned>(v)) < 0)
    if (gtime() - t0 > kSpinNs) return false;
  return true;
}

__device__ __forceinline__ unsigned long long* gin_cum(int* ctr) {  // [2][kMaxWorld]
  return reinterpret_cast<unsigned long long*>(ctr + kCtrCum);
}

#if MEGAMOE_SM120_GIN_DEVICE
__device__ __forceinline__ bool is_gin(const Args& a, int w) { return (a.gin.mask >> w) & 1u; }
__device__ __forceinline__ const ncclDevComm& gin_comm(const Args& a) {
  return *static_cast<const ncclDevComm*>(a.gin.comm);
}
__device__ __forceinline__ ncclWindow_t gin_win(const Args& a) {
  return static_cast<ncclWindow_t>(a.gin.win);
}
// a CTA posts every put to one peer on one context (its puts to that peer stay ordered, so a strong
// signal settles the earlier puts of the same thread); CTAs and peers spread over the contexts'
// queue pairs, so the puts one CTA posts to several peers at once go out in parallel
__device__ __forceinline__ int gin_ctx(const Args& a, int peer) {
  return (int(blockIdx.x) + peer) % a.gin.nctx;
}
template <typename Action>
__device__ __forceinline__ void gin_put(const Args& a, int peer, size_t dst_off, size_t src_off,
                                        size_t bytes, Action action) {
  const ncclDevComm& c = gin_comm(a);
  ncclGin g(c, gin_ctx(a, peer));
  g.put(ncclTeamWorld(c), peer, gin_win(a), dst_off, gin_win(a), src_off, bytes, action);
}
template <typename Action>
__device__ __forceinline__ void gin_signal(const Args& a, int peer, Action action) {
  const ncclDevComm& c = gin_comm(a);
  ncclGin g(c, gin_ctx(a, peer));
  g.signal(ncclTeamWorld(c), peer, action);
}
// thread 0 spins until GIN signal sig, summed over the contexts, reaches target (rolling 64-bit
// compare); false on timeout
__device__ __forceinline__ bool gin_wait(const Args& a, int sig, unsigned long long target) {
  const uint64_t t0 = gtime();
  while (true) {
    unsigned long long v = 0;
    for (int x = 0; x < a.gin.nctx; ++x)
      v += ncclGin(gin_comm(a), x).readSignal(static_cast<ncclGinSignal_t>(sig));
    if (static_cast<long long>(v - target) >= 0) return true;
    if (gtime() - t0 > kSpinNs) return false;
  }
}
#else
__device__ __forceinline__ constexpr bool is_gin(const Args&, int) { return false; }
#endif

// Wait until every rank w's flag flags[w] (written over PCIe) reaches epoch, or a GIN rank's
// signal sig + w reaches target(w). One CTA (the last) polls them, a thread per rank, and
// republishes on the local word ctr[kCtrFlag + kind]; the other CTAs wait on that word: with every
// CTA polling, the loads delay the flags' and signals' arrival. A failed wait records err first.
// Every thread calls it; the result is thread 0's.
template <class Target>
__device__ __forceinline__ bool wait_peers(const Args& a, const int* flags, int epoch, int kind,
                                           int sig, int err, Target target) {
  int* local = a.ctr + kCtrFlag + kind;
  if (blockIdx.x != gridDim.x - 1) {
    if (threadIdx.x != 0) return true;
    return wait_ge(local, epoch, false) && ld_acquire_gpu(a.ctr + kCtrErr) == 0;
  }
  __shared__ int s_fail;
  if (threadIdx.x == 0) s_fail = 0;
  __syncthreads();
  const int w = threadIdx.x;
  if (w < a.W) {
    bool ok;
#if MEGAMOE_SM120_GIN_DEVICE
    if (is_gin(a, w))
      ok = gin_wait(a, sig + w, target(w));
    else
#endif
      ok = wait_ge(flags + w, epoch, true);
    if (!ok) s_fail = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (s_fail) atomicExch(a.ctr + kCtrErr, err);
    __threadfence();
    st_release_gpu(local, epoch);
  }
  return threadIdx.x != 0 || !s_fail;
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
// 4x4 word transpose across the lanes of a quad: v[q] of lane c becomes v[c] of lane q
__device__ __forceinline__ void quad_transpose(uint32_t (&v)[4], int c) {
#pragma unroll
  for (int m = 2; m >= 1; m >>= 1) {
    const bool hi = (c & m) != 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      if (q & m) continue;
      const uint32_t r = __shfl_xor_sync(0xffffffffu, hi ? v[q] : v[q | m], m);
      if (hi)
        v[q] = r;
      else
        v[q | m] = r;
    }
  }
}

// 4 consecutive E2M1 nibbles (one 16-bit half of w, low nibble first) -> 4 bytes, each holding its
// value in bits 5..2 as the mxf8f6f4 E2M1 operand expects
__device__ __forceinline__ uint32_t spread4(uint32_t w, uint32_t half_selector) {
  const uint32_t t = __byte_perm(w, 0u, half_selector);  // 0x00 b1 0x00 b0 of the chosen half
  return ((t & 0x000F000Fu) << 2) | ((t & 0x00F000F0u) << 6);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
      "l"(src)
      : "memory");
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(uint32_t(__cvta_generic_to_shared(dst))),
      "l"(src)
      : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// activation staging of items wider than 8 tokens: per K step (kUnroll K64 chunks) a token's
// kStageB value bytes, rows kStageRow apart (the pad spreads the 8 token rows a warp reads at once
// over the banks), and its kStageW scale words
constexpr int kStageB = kUnroll * 64, kStageRow = kStageB + 32, kStageW = kUnroll / 2;
template <int kTileN>
constexpr int stage_bytes() {
  return 2 * kTileN * kStageRow + 2 * kTileN * kStageW * 4;
}

// warp GEMM over one item: acc[j][h] = rows (nb0+j)*16 + {g, g+8}, cols = tokens 8h + 2c,
// 8h + 2c + 1. W: row-major packed E2M1 [rows][K/2]; WS: UE8M0 x4 words per 16-row block and K128.
// xrow[h]: this lane's token 8h + g (e4m3, K bytes), xsrow[h]: its ue8m0 x4 words (K/128). The NH
// token groups share every weight load. Lane c loads block c of each K128 of rows g and g + 8 and
// the quad transposes, so lane c holds weight bytes 4c..4c+3 (elements 8c..8c+7) of every block,
// and reads activation bytes 8c..8c+7: the same K permutation on both operands (file comment).
// [kc_begin, kc_end) restricts K to a range of K64 chunks (split-K); both multiples of kUnroll.
// nsub: token groups of 8 that hold tokens (an expert's last tile is partly empty); the others
// take no activation loads and no MMAs (their accumulators stay 0).
// Items wider than 8 tokens: the item's tokens are the same for its 8 warps, so the whole CTA
// stages their activations (sxr / sxs: token tk's value row / scale words, the live groups only)
// one K step ahead with cp.async into sx / ss ([2][8 NH] rows, double buffered), and every warp
// reads them from shared memory instead of each loading them from global memory; every thread of
// the CTA calls it then.
template <int NH>
__device__ __forceinline__ void item_gemm(
    float (&acc)[kNbPerWarp][NH][4], const uint8_t* W, const uint32_t* WS, int nblocks, int nb0,
    int K, const uint8_t* const (&xrow)[NH], const uint32_t* const (&xsrow)[NH], int lane, int nsub,
    const uint8_t* const* sxr = nullptr, const uint32_t* const* sxs = nullptr,
    uint8_t* sx = nullptr, uint32_t* ss = nullptr, int kc_begin = 0, int kc_end = -1) {
  const int g = lane / 4, c = lane % 4;
  const int kch = K / 64, kq = K / 128;
  const uint4* wp[kNbPerWarp][2];  // rows g, g + 8 of block nb0 + j, at this lane's 16 bytes
  const uint32_t* sp[kNbPerWarp];
#pragma unroll
  for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
    for (int r = 0; r < 2; ++r)
      wp[j][r] =
          reinterpret_cast<const uint4*>(W + size_t((nb0 + j) * 16 + g + 8 * r) * (K / 2)) + c;
    sp[j] = WS + size_t(nb0 + j) * kq * 16 + g + (lane & 1) * 8;
  }
  const uint2* xp[NH];
#pragma unroll
  for (int h = 0; h < NH; ++h) xp[h] = reinterpret_cast<const uint2*>(xrow[h]) + c;
#pragma unroll
  for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
    for (int h = 0; h < NH; ++h) acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;
  if (kc_end < 0) kc_end = kch;
  // lanes 0..15 prefetch row lane of the block kPfAhead bytes ahead, one 128-byte line (an
  // iteration's K256) per iteration
  const char* pf = reinterpret_cast<const char*>(W) + size_t(nb0 * 16 + (lane & 15)) * (K / 2);
  // 8-token items (one MMA per weight fragment) leave the loop waiting on DRAM: the next step's
  // weights and scales are loaded into a second buffer while this step's MMAs run (both builds:
  // with the single-poller waits the GIN build is faster with it too, RTX 6000D cold L2,
  // DeepSeek-V4.1-Flash balanced: EP8 1 token per rank 228 -> 200 us, EP16 271 -> 240 us).
  if constexpr (NH == 1) {
    uint4 wq[kNbPerWarp][kUnroll / 2][2], wn[kNbPerWarp][kUnroll / 2][2];
    uint32_t sfa[kNbPerWarp][(kUnroll + 1) / 2], sfn[kNbPerWarp][(kUnroll + 1) / 2];
    auto load = [&](int k0, uint4(&q)[kNbPerWarp][kUnroll / 2][2],
                    uint32_t(&sf)[kNbPerWarp][(kUnroll + 1) / 2]) {
#pragma unroll
      for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) q[j][p][r] = __ldcs(wp[j][r] + (k0 / 2 + p) * 4);
#pragma unroll
      for (int s = 0; s < (kUnroll + 1) / 2; ++s)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) sf[j][s] = __ldg(sp[j] + (k0 / 2 + s) * 16);
    };
    if (kc_begin < kc_end) load(kc_begin, wq, sfa);
    for (int k0 = kc_begin; k0 < kc_end; k0 += kUnroll) {
      if (kPfAhead > 0 && lane < 16 && k0 * 32 + kPfAhead < K / 2)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + k0 * 32 + kPfAhead));
      if (k0 + kUnroll < kc_end) load(k0 + kUnroll, wn, sfn);
      uint32_t wv[kNbPerWarp][kUnroll][2][2];
#pragma unroll
      for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            uint32_t v[4] = {wq[j][p][r].x, wq[j][p][r].y, wq[j][p][r].z, wq[j][p][r].w};
            quad_transpose(v, c);
#pragma unroll
            for (int b = 0; b < 4; ++b) wv[j][p * 2 + b / 2][b % 2][r] = v[b];
          }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int kc = k0 + u;
        const uint32_t sfb = xsrow[0][kc / 2];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int step = kc * 2 + s;
          const uint16_t sel = uint16_t(step & 3);
          const uint2 b = xp[0][step * 4];
#pragma unroll
          for (int j = 0; j < kNbPerWarp; ++j) {
            const uint32_t wg = wv[j][u][s][0], wg8 = wv[j][u][s][1];
            const uint32_t a[4] = {spread4(wg, 0x4140u), spread4(wg8, 0x4140u),
                                   spread4(wg, 0x4342u), spread4(wg8, 0x4342u)};
            mma_w4a8(acc[j][0], a, b.x, b.y, sfa[j][u / 2], sel, sfb, sel);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kNbPerWarp; ++j) {
#pragma unroll
        for (int p = 0; p < kUnroll / 2; ++p) wq[j][p][0] = wn[j][p][0], wq[j][p][1] = wn[j][p][1];
#pragma unroll
        for (int s = 0; s < (kUnroll + 1) / 2; ++s) sfa[j][s] = sfn[j][s];
      }
    }
    return;
  }
  if constexpr (NH > 1) {
    constexpr int kT = kTile * NH;
    const int ntk = nsub * kTile;
    auto stage = [&](int k0, int b) {
      for (int u = threadIdx.x; u < ntk * (kStageB / 16); u += blockDim.x) {
        const int tk = u / (kStageB / 16), q = u % (kStageB / 16);
        cp_async16(sx + (b * kT + tk) * kStageRow + q * 16, sxr[tk] + k0 * 64 + q * 16);
      }
      for (int tk = threadIdx.x; tk < ntk; tk += blockDim.x)
        cp_async8(ss + (b * kT + tk) * kStageW, sxs[tk] + k0 / 2);
      cp_async_commit();
    };
    if (kc_begin < kc_end) stage(kc_begin, 0);
    for (int k0 = kc_begin; k0 < kc_end; k0 += kUnroll) {
      const int buf = ((k0 - kc_begin) / kUnroll) & 1;
      if (k0 + kUnroll < kc_end) {
        stage(k0 + kUnroll, buf ^ 1);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      if (kPfAhead > 0 && lane < 16 && k0 * 32 + kPfAhead < K / 2)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + k0 * 32 + kPfAhead));
      uint32_t wv[kNbPerWarp][kUnroll][2][2];
      {
        uint4 wq[kNbPerWarp][kUnroll / 2][2];
#pragma unroll
        for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
          for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
            for (int r = 0; r < 2; ++r) wq[j][p][r] = __ldcs(wp[j][r] + (k0 / 2 + p) * 4);
#pragma unroll
        for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
          for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              uint32_t v[4] = {wq[j][p][r].x, wq[j][p][r].y, wq[j][p][r].z, wq[j][p][r].w};
              quad_transpose(v, c);
#pragma unroll
              for (int b = 0; b < 4; ++b) wv[j][p * 2 + b / 2][b % 2][r] = v[b];
            }
      }
      uint32_t sfa[kNbPerWarp][(kUnroll + 1) / 2];
#pragma unroll
      for (int q = 0; q < (kUnroll + 1) / 2; ++q)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) sfa[j][q] = __ldg(sp[j] + (k0 / 2 + q) * 16);
      __syncthreads();  // this step's activations, staged by every thread
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int kc = k0 + u;
        uint32_t sfb[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h)
          sfb[h] = h < nsub ? ss[(buf * kT + h * kTile + g) * kStageW + u / 2] : 0u;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int step = kc * 2 + s;
          const uint16_t sel = uint16_t(step & 3);
          uint2 b[NH];
#pragma unroll
          for (int h = 0; h < NH; ++h)
            b[h] = h < nsub ? reinterpret_cast<const uint2*>(
                                  sx + (buf * kT + h * kTile + g) * kStageRow)[(u * 2 + s) * 4 + c]
                            : make_uint2(0u, 0u);
#pragma unroll
          for (int j = 0; j < kNbPerWarp; ++j) {
            const uint32_t wg = wv[j][u][s][0], wg8 = wv[j][u][s][1];
            const uint32_t a[4] = {spread4(wg, 0x4140u), spread4(wg8, 0x4140u),
                                   spread4(wg, 0x4342u), spread4(wg8, 0x4342u)};
#pragma unroll
            for (int h = 0; h < NH; ++h)
              if (h < nsub) mma_w4a8(acc[j][h], a, b[h].x, b[h].y, sfa[j][u / 2], sel, sfb[h], sel);
          }
        }
      }
      __syncthreads();  // every warp is done with this buffer before the next step stages into it
    }
    return;
  }
  for (int k0 = kc_begin; k0 < kc_end; k0 += kUnroll) {
    if (kPfAhead > 0 && lane < 16 && k0 * 32 + kPfAhead < K / 2)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + k0 * 32 + kPfAhead));
    uint32_t wv[kNbPerWarp][kUnroll][2][2];  // [block][K64 chunk][step][row g / g + 8]
    {
      uint4 wq[kNbPerWarp][kUnroll / 2][2];  // [block][K128 chunk][row]: this lane's 32-block
#pragma unroll
      for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) wq[j][p][r] = __ldcs(wp[j][r] + (k0 / 2 + p) * 4);
#pragma unroll
      for (int p = 0; p < kUnroll / 2; ++p)
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            uint32_t v[4] = {wq[j][p][r].x, wq[j][p][r].y, wq[j][p][r].z, wq[j][p][r].w};
            quad_transpose(v, c);  // v[b]: bytes 4c..4c+3 of 32-block b of this K128
#pragma unroll
            for (int b = 0; b < 4; ++b) wv[j][p * 2 + b / 2][b % 2][r] = v[b];
          }
    }
    uint32_t sfa[kNbPerWarp][(kUnroll + 1) / 2];
#pragma unroll
    for (int q = 0; q < (kUnroll + 1) / 2; ++q)
#pragma unroll
      for (int j = 0; j < kNbPerWarp; ++j) sfa[j][q] = __ldg(sp[j] + (k0 / 2 + q) * 16);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int kc = k0 + u;
      uint32_t sfb[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) sfb[h] = h < nsub ? xsrow[h][kc / 2] : 0u;
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int step = kc * 2 + s;
        const uint16_t sel = uint16_t(step & 3);
        uint2 b[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) b[h] = h < nsub ? xp[h][step * 4] : make_uint2(0u, 0u);
#pragma unroll
        for (int j = 0; j < kNbPerWarp; ++j) {
          const uint32_t wg = wv[j][u][s][0], wg8 = wv[j][u][s][1];
          const uint32_t a[4] = {spread4(wg, 0x4140u), spread4(wg8, 0x4140u), spread4(wg, 0x4342u),
                                 spread4(wg8, 0x4342u)};
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (h < nsub) mma_w4a8(acc[j][h], a, b[h].x, b[h].y, sfa[j][u / 2], sel, sfb[h], sel);
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
  int lag;  // queue order (item_decode): tiles whose FC1 items precede the first FC2 item
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

// Queue order over the tiles (a tile = one expert's kTileN tokens; nblk1 FC1 items, nblk2 FC2
// items): FC1 of the first `lag` tiles, then FC1 of tile t + lag followed by FC2 of tile t, then
// the last FC2s. The partials leave during the whole weight stream instead of after the last FC1,
// and an expert's FC1 has finished (or nearly) when its FC2 is claimed.
__device__ __forceinline__ void item_decode(int item, int ntiles, int nblk1, int nblk2, int lag,
                                            bool& l1, int& tile_g, int& rb) {
  const int head = lag * nblk1, per = nblk1 + nblk2, mid = (ntiles - lag) * per;
  if (item < head) {
    l1 = true, tile_g = item / nblk1, rb = item % nblk1;
  } else if (item < head + mid) {
    const int t = (item - head) / per, r = (item - head) % per;
    l1 = r < nblk1;
    tile_g = l1 ? t + lag : t;
    rb = l1 ? r : r - nblk1;
  } else {
    const int j = item - head - mid;
    l1 = false, tile_g = ntiles - lag + j / nblk2, rb = j % nblk2;
  }
}

// A CTA's first item is static. The first CTAs dispatch this rank's tokens (and send the routing)
// and reach the queue last, so the head of the queue goes to the last CTAs.
__device__ __forceinline__ int first_item(unsigned block, unsigned grid) {
  return int(grid - 1 - block);
}

__device__ __forceinline__ void token_src(const Map& m, int el, int j, int W, int& w, int& i) {
  w = 0;
  i = j;
  while (w < W - 1 && i >= m.cnt[w][el]) {
    i -= m.cnt[w][el];
    ++w;
  }
}

// kTileN tokens per work item (a multiple of 8): the kTileN / 8 MMAs of a step share each weight
// fragment, so larger items cut the weight traffic when the experts hold many rows each, smaller
// ones keep more of the grid busy when they hold few. Two CTAs per SM (the grid is sized from the
// occupancy): with one, 8-token items fill only one wave of the SMs' load slots
template <int kTileN>
__global__ void __launch_bounds__(kWarps * 32, 2) sm120_mega_moe_kernel(Args a) {
  constexpr int NH = kTileN / kTile;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int H = a.H, I = a.I, EL = a.EL, W = a.W, TOPK = a.TOPK, TCAP = a.TCAP;
  __shared__ Map m;
  __shared__ int s_item, s_ok, s_epoch;
  // L1: [tok][kRowsPerItem/2 inter] / L2: [tok][kRowsPerItem hidden] / dispatch: one MXFP8 token
  // row (H + H/32 bytes)
  __shared__ __align__(16) float s_tile[kTileN][kRowsPerItem < 256 ? 256 : kRowsPerItem];
  // items wider than 8 tokens stage their activations in s_tile's bytes while their GEMM runs (the
  // epilogues use s_tile only after it): the item's token rows (values / scale words)
  static_assert(sizeof(s_tile) >= size_t(stage_bytes<kTileN>()), "activation staging fits s_tile");
  __shared__ const uint8_t* s_xr[kTileN];
  __shared__ const uint32_t* s_xs[kTileN];
  uint8_t* const s_sx = reinterpret_cast<uint8_t*>(&s_tile[0][0]);
  uint32_t* const s_ss = reinterpret_cast<uint32_t*>(s_sx + 2 * kTileN * kStageRow);

  // Phase stamps for Args::stamps (debug). Kept even with stamps null: without them nvcc 13
  // allocates the 8-token loop 128 registers with spills instead of 168 (RTX 6000D, cold L2:
  // 2-11% slower at 1-128 tokens per rank).
  unsigned long long st_[16] = {
      gtime()};  // +8 route-push 9 route-ok 10 map 11 pf 12 flag-sent 13 last FC1 end 14 first FC2
                 // start 15 comb-flag release  // 0 start 1 pushed 2 flags 3 work end 4 comb flags
                 // 5 end 6 L1-wait ns 7 items
  if (threadIdx.x == 0)  // wraps modulo 2^32 (unsigned: no signed overflow)
    s_epoch = static_cast<int>(*reinterpret_cast<volatile unsigned*>(a.ctr + kCtrEpoch) + 1u);
  // the error word reports this call only (a wait records one no earlier than kSpinNs from now)
  if (blockIdx.x == 0 && threadIdx.x == 0) a.ctr[kCtrErr] = 0;
  __syncthreads();
  const int epoch = s_epoch;
  int* ctr = a.ctr;

  // ---------------- routing exchange first (tiny): every rank learns its experts' load from each
  // source before the rows land: per destination rank, this rank's routes to each of its local
  // experts and the number of its tokens routed there (one inbox row each over GIN)
  const int RS = EL + 1;
  // the routes go to the half of the peer's route region this call's epoch parity picks: a peer
  // whose done arrived may still have CTAs that have not read this call's routes when this rank
  // posts the next call's (it cannot get two calls ahead: each call waits for every rank's routes)
  const int rhalf = (epoch & 1) * W * RS;
  const size_t RX = pool_row_bytes(H);
  const size_t SB = gin_slot_bytes(TCAP, TOPK), GB = gin_block(TCAP, TOPK, H);
  if (blockIdx.x == (a.T < int(gridDim.x) ? a.T : int(gridDim.x) - 1)) {
    __shared__ int s_ntok[kMaxWorld];  // this rank's tokens routed to each rank
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x) m.cnt[u / EL][u % EL] = 0;
    if (threadIdx.x < W) s_ntok[threadIdx.x] = 0;
    __syncthreads();
    for (int t = threadIdx.x; t < a.T; t += blockDim.x) {  // m.cnt as scratch: [dst][el]
      unsigned to = 0;
      for (int k = 0; k < TOPK; ++k) {
        const int e = route_at(a, t, k);
        if (e < 0) continue;
        atomicAdd(&m.cnt[e / EL][e % EL], 1);
        to |= 1u << (e / EL);
      }
      for (; to; to &= to - 1) atomicAdd(&s_ntok[__ffs(to) - 1], 1);
    }
    __syncthreads();
    for (int u = threadIdx.x; u < W * RS; u += blockDim.x) {
      const int dst = u / RS, v = u % RS;
      const int val = v < EL ? m.cnt[dst][v] : s_ntok[dst];
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, dst)) {
        a.gin.stage_route[u] = val;
        continue;
      }
#endif
      a.peers.route[dst][rhalf + a.rank * RS + v] = val;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < W) {  // a thread per rank
      const int dst = threadIdx.x;
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, dst))
        gin_put(a, dst, a.gin.off_route + size_t(rhalf + a.rank * RS) * sizeof(int),
                a.gin.off_stage_route + size_t(dst) * RS * sizeof(int), size_t(RS) * sizeof(int),
                ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigRoute + a.rank), 1});
      else
#endif
        st_release_sys(a.peers.route_flag[dst] + a.rank, epoch);
    }
    st_[8] = gtime();
  }

  // ---------------- t=0: this rank's own tokens already tell which local experts are active; pull
  // the head of their FC1 weights into L2 while the rows and the other ranks' routing are in
  // flight, on the CTAs that neither dispatch nor send the routing: the rows must not wait for it
  const int ndisp = min(a.T + 1, int(gridDim.x));
  if (kPf0Bytes > 0 && EL <= 64 && int(blockIdx.x) >= ndisp) {
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
      const size_t per = min(a.l1_stride, kPf0Bytes / nk) & ~size_t(127);
      const size_t lines = per / 128 * nk;
      for (size_t l = size_t(int(blockIdx.x) - ndisp) * blockDim.x + threadIdx.x; l < lines;
           l += size_t(int(gridDim.x) - ndisp) * blockDim.x) {
        const int ki = int(l / (per / 128));
        unsigned long long rest = known;
        for (int q = 0; q < ki; ++q) rest &= rest - 1;
        const int el = __ffsll(rest) - 1;
        const char* p = reinterpret_cast<const char*>(w13_of(a, el)) + (l % (per / 128)) * 128;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
    }
  }

  // ---------------- dispatch: one CTA per token: the staged MXFP8 row into smem, push TOPK copies
  // with every thread
  const int nitems = a.T * TOPK;
  {
    uint8_t* s_q = reinterpret_cast<uint8_t*>(
        &s_tile[0][0]);  // [H] e4m3, then [H/32] ue8m0 (H + H/32 <= smem)
    uint8_t* s_sc = s_q + H;
    __shared__ int s_rowk[8], s_dstk[8], s_posk[8];
    // the rank's routes (route_at, masked: -1), read once: every copy's slot scans all of them
    __shared__ int16_t s_rt[kMaxTokens * kMaxTopk];
    if (int(blockIdx.x) < a.T) {
      for (int u = threadIdx.x; u < nitems; u += blockDim.x)
        s_rt[u] = int16_t(route_at(a, u / TOPK, u % TOPK));
      __syncthreads();
    }
    for (int t = blockIdx.x; t < a.T; t += gridDim.x) {
      if (threadIdx.x < H / 32) {  // one thread per K32 group: the staged codes and their scale
        const uint4* s4 = reinterpret_cast<const uint4*>(a.x + size_t(t) * H + threadIdx.x * 32);
        uint4* d4 = reinterpret_cast<uint4*>(s_q + threadIdx.x * 32);
        d4[0] = s4[0];
        d4[1] = s4[1];
        s_sc[threadIdx.x] = a.x_sf[size_t(t) * (H / 32) + threadIdx.x];
      }
      if (warp < TOPK) {  // warp k: destination row of copy k (slot = tokens t' < t routed to the
                          // same expert)
        const int e = s_rt[t * TOPK + warp];  // < 0: masked route, nothing to send
        int slot = 0;
        for (int u = lane; u < t * TOPK; u += 32) slot += (s_rt[u] == e);
#pragma unroll
        for (int o = 16; o; o >>= 1) slot += __shfl_xor_sync(0xffffffffu, slot, o);
#if MEGAMOE_SM120_GIN_DEVICE
        if (e >= 0 && is_gin(a, e / EL)) {  // slot table entry: packed (el, slot) -> packed row
          const int d = e / EL;
          int pre = 0, nd = 0;  // this rank's routes to lower experts of that rank / to it
          for (int u = lane; u < nitems; u += 32) {
            const int e2 = s_rt[u];
            nd += (e2 >= 0 && e2 / EL == d);
            pre += (e2 >= 0 && e2 / EL == d && e2 % EL < e % EL);
          }
          int pos = 0;  // the token's row among this rank's tokens t' < t routed to rank d
          for (int u = lane; u < t; u += 32) {
            bool to = false;
            for (int k = 0; k < TOPK; ++k) {
              const int e2 = s_rt[u * TOPK + k];
              to |= e2 >= 0 && e2 / EL == d;
            }
            pos += to;
          }
#pragma unroll
          for (int o = 16; o; o >>= 1) {
            pre += __shfl_xor_sync(0xffffffffu, pre, o);
            nd += __shfl_xor_sync(0xffffffffu, nd, o);
            pos += __shfl_xor_sync(0xffffffffu, pos, o);
          }
          if (lane == 0) {  // the same packed index places this copy's partial on the way back
            reinterpret_cast<int*>(a.gin.stage_row + d * GB + SB)[pre + slot - slot_words(nd)] =
                pos;
            a.gin.cidx[t * TOPK + warp] = pre + slot;
            s_posk[warp] = pos;
          }
        }
#endif
        if (lane == 0) {
          if (e < 0) {
            s_rowk[warp] = -1;
          } else {
            const int dst = e / EL, el = e % EL;
            const int row = (el * W + a.rank) * TCAP + slot;
            s_rowk[warp] = row;
            s_dstk[warp] = dst;
          }
        }
      }
      __syncthreads();
      // per copy: H / 16 uint4 of data, the H / 32 scale bytes as H / 256 uint2 (a scale row is
      // only 8-byte aligned when hidden % 512 != 0), then the int metadata. A copy to a GIN rank
      // goes to that rank's packed staging rows instead (no metadata), only for the token's first
      // copy there: GIN ranks get the token once, into their inbox, whatever number of their
      // experts it routes to.
      const int per = H / 16 + H / 256 + 1;
      for (int u = threadIdx.x; u < TOPK * per; u += blockDim.x) {
        const int k = u / per, q = u % per;
        if (s_rowk[k] < 0) continue;
        uint8_t* dst;
        if (is_gin(a, s_dstk[k])) {
          bool first = q != per - 1;
          for (int k2 = 0; k2 < k; ++k2) first &= !(s_rowk[k2] >= 0 && s_dstk[k2] == s_dstk[k]);
          if (!first) continue;
          dst = a.gin.stage_row + s_dstk[k] * GB + SB + s_posk[k] * RX;
        } else {
          dst = a.peers.pool[s_dstk[k]] + size_t(s_rowk[k]) * RX;
        }
        if (q < H / 16)
          reinterpret_cast<uint4*>(dst)[q] = reinterpret_cast<const uint4*>(s_q)[q];
        else if (q < H / 16 + H / 256)
          reinterpret_cast<uint2*>(dst + H)[q - H / 16] =
              reinterpret_cast<const uint2*>(s_sc)[q - H / 16];
        else
          *reinterpret_cast<int*>(dst + H + H / 32) = t * 8 + k;
      }
      __syncthreads();
    }
  }
  __threadfence_system();
  __syncthreads();
  st_[1] = gtime();
  if (threadIdx.x == 0) s_item = (atomicAdd(ctr + kCtrDispatch, 1) == int(gridDim.x) - 1);
  __syncthreads();
  if (s_item) {  // last CTA out of the dispatch: every row of this rank is visible -> a thread per
                 // P2P rank flags it, then a thread per GIN rank puts its slot table and rows
    if (threadIdx.x < W && !is_gin(a, threadIdx.x))
      st_release_sys(a.peers.recv_flag[threadIdx.x] + a.rank, epoch);
#if MEGAMOE_SM120_GIN_DEVICE
    if (a.gin.mask) {
      // the routing counts staged for each GIN rank: this rank's routes there (its slot table
      // length) and its tokens there (its packed rows)
      __shared__ int s_nd[kMaxWorld];
      if (threadIdx.x < W) s_nd[threadIdx.x] = 0;
      __syncthreads();
      for (int u = threadIdx.x; u < W * RS; u += blockDim.x)
        if (u % RS < EL && is_gin(a, u / RS))
          atomicAdd(&s_nd[u / RS], __ldcg(a.gin.stage_route + u));
      __syncthreads();
      if (threadIdx.x < W && is_gin(a, threadIdx.x)) {  // one put from the table's start through
                                                        // the last row
        const int d = threadIdx.x, nt = __ldcg(a.gin.stage_route + d * RS + EL);
        const size_t head = SB - size_t(slot_words(s_nd[d])) * sizeof(int);
        gin_put(a, d, a.gin.off_inbox + a.rank * GB + head, a.gin.off_stage_row + d * GB + head,
                SB - head + size_t(nt) * RX,
                ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigDispatch + a.rank), uint64_t(nt) + 1});
      }
    }
#endif
    st_[12] = gtime();
  }
  __syncthreads();

  // ---------------- per-expert map from the exchanged routing (same slot order as the dispatch)
  {  // one routing put per call from a GIN rank: the signal counts calls (64-bit)
    const unsigned long long calls =
        *reinterpret_cast<volatile unsigned long long*>(ctr + kCtrCalls);
    const bool ok = wait_peers(a, a.peers.route_flag[a.rank], epoch, kFlagRoute, kSigRoute,
                               kErrRouteWait, [&](int) { return calls + 1; });
    if (threadIdx.x == 0) s_ok = ok;
  }
  __syncthreads();
  st_[9] = gtime();
  if (!s_ok) goto exit_kernel;
#if MEGAMOE_SM120_GIN_DEVICE
  __shared__ int s_dn[kMaxWorld];  // GIN source w: tokens it sends here (one inbox row each)
#endif
  {  // every source's counts for this rank's experts, read in one parallel pass
    const int* rb = a.peers.route[a.rank] + rhalf;
    __syncthreads();  // the routing CTA's scratch use of m.cnt is over
    for (int u = threadIdx.x; u < W * EL; u += blockDim.x)
      m.cnt[u / EL][u % EL] = __ldcv(rb + (u / EL) * RS + u % EL);
#if MEGAMOE_SM120_GIN_DEVICE
    if (threadIdx.x < W) s_dn[threadIdx.x] = __ldcv(rb + threadIdx.x * RS + EL);
#endif
  }
  __syncthreads();
  if (warp == 0) {
    int tiles_sum = 0, tiles_max = 0;
    for (int base = 0; base < EL; base += 32) {
      const int el = base + lane;
      int n = 0;
      if (el < EL)
        for (int w = 0; w < W; ++w) n += m.cnt[w][el];
      int incl = (n + kTileN - 1) / kTileN;
      tiles_max = max(tiles_max, incl);
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
#pragma unroll
    for (int o = 16; o; o >>= 1)
      tiles_max = max(tiles_max, __shfl_xor_sync(0xffffffffu, tiles_max, o));
    if (lane == 0) {  // lag: one wave of CTAs (their static first items are FC1) + an expert's own
                      // later tiles
      m.tiles_prefix[0] = 0;
      const int nblk1 = 2 * I / kRowsPerItem;
      m.lag = min(tiles_sum, (int(gridDim.x) + nblk1 - 1) / nblk1 + tiles_max);
    }
  }
  __syncthreads();
  st_[10] = gtime();
  {  // this CTA's first item is static: pull this warp's share of its weights and scales into L2
     // now. The item is FC1 while the queue head lasts; with few tokens the later CTAs start on
     // FC2 items, whose weights then arrive while FC1 runs. Skipped up to 8 tiles, where it only
     // delays the rows wait
    const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
    const int ntiles = m.tiles_prefix[EL];
    const int item = first_item(blockIdx.x, gridDim.x);
    if (ntiles > 8 && item < ntiles * (nblk1 + nblk2)) {
      bool l1;
      int tile_g, rb;
      item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int K = l1 ? H : I;
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      const char* wb = reinterpret_cast<const char*>(l1 ? w13_of(a, el) : w2_of(a, el)) +
                       size_t(nb0) * 16 * (K / 2);
      const char* sb = reinterpret_cast<const char*>((l1 ? ws13_of(a, el) : ws2_of(a, el)) +
                                                     size_t(nb0) * (K / 128) * 16);
      for (int off = lane * 128; off < kNbPerWarp * 16 * (K / 2); off += 32 * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(wb + off));
      for (int off = lane * 128; off < kNbPerWarp * (K / 128) * 64; off += 32 * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(sb + off));
    }
  }
  st_[11] = gtime();

  // ---------------- wait for every source rank, build the per-expert map
  {  // GIN rank w: one signal per token it sends here, plus its slot table
#if MEGAMOE_SM120_GIN_DEVICE
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x < W && is_gin(a, threadIdx.x))
      ctr[kCtrCnt + threadIdx.x] = s_dn[threadIdx.x] + 1;
#endif
    const bool ok = wait_peers(a, a.peers.recv_flag[a.rank], epoch, kFlagRows, kSigDispatch,
                               kErrDispatchWait, [&](int w) {
#if MEGAMOE_SM120_GIN_DEVICE
                                 return gin_cum(ctr)[w] + s_dn[w] + 1;
#else
                                 return 0ull;
#endif
                               });
    if (threadIdx.x == 0) s_ok = ok;
  }
  __syncthreads();
  st_[2] = gtime();
  if (!s_ok) goto exit_kernel;
  {
    const int nblk1 = 2 * I / kRowsPerItem, nblk2 = H / kRowsPerItem;
    const int ntiles = m.tiles_prefix[EL];
    const int n1 = ntiles * nblk1, n2 = ntiles * nblk2;
    const uint8_t* pool = a.peers.pool[a.rank];
    const int g = lane / 4, c = lane % 4;
    const int TOKE = W * TCAP;

    bool first = true;
    while (true) {
      if (threadIdx.x == 0)
        s_item = first ? first_item(blockIdx.x, gridDim.x)
                       : int(gridDim.x) + atomicAdd(ctr + kCtrQueue, 1);
      first = false;
      __syncthreads();
      const int item = s_item;
      if (item >= n1 + n2) break;
      st_[7] += 1;
      bool l1;
      int tile_g, rb;
      item_decode(item, ntiles, nblk1, nblk2, m.lag, l1, tile_g, rb);
      int el = 0;
      while (m.tiles_prefix[el + 1] <= tile_g) ++el;
      const int tile = tile_g - m.tiles_prefix[el];
      const int ntok = min(kTileN, m.ne[el] - tile * kTileN);
      int jt[NH];  // this lane's tokens (B column g of each group of 8; a padding lane repeats one)
#pragma unroll
      for (int h = 0; h < NH; ++h) jt[h] = tile * kTileN + (h * 8 + g < ntok ? h * 8 + g : 0);
#if MEGAMOE_SM120_GIN_DEVICE
      __shared__ int s_pw[kMaxWorld];          // GIN source w: its slot table index of (el, slot 0)
      __shared__ const int* s_tab[kMaxWorld];  // ... and that entry in its inbox block
      if (a.gin.mask) {
        if (threadIdx.x < W && is_gin(a, threadIdx.x)) {
          const int w = threadIdx.x;
          int p = 0, n = 0;
          for (int e2 = 0; e2 < EL; ++e2) {
            p += e2 < el ? m.cnt[w][e2] : 0;
            n += m.cnt[w][e2];
          }
          s_pw[w] = p;
          s_tab[w] = reinterpret_cast<const int*>(a.gin.inbox + w * GB + SB) - slot_words(n) + p;
        }
        __syncthreads();
      }
#endif

      if (!l1) {  // FC2 needs every FC1 item of this expert
        const unsigned long long tw0 = gtime();
        if (threadIdx.x == 0) {
          const int need = ((m.ne[el] + kTileN - 1) / kTileN) * nblk1;
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

      float acc[kNbPerWarp][NH][4];
      const int nb0 = (rb * kRowsPerItem) / 16 + warp * kNbPerWarp;
      if (l1) {
        const uint8_t* xr[NH];
        const uint32_t* xs[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          int w, i;
          token_src(m, el, jt[h], W, w, i);
          const size_t row = (size_t(el) * W + w) * TCAP + i;
          xr[h] = pool + row * RX;
#if MEGAMOE_SM120_GIN_DEVICE
          if (is_gin(a, w)) xr[h] = a.gin.inbox + w * GB + SB + s_tab[w][i] * RX;
#endif
          xs[h] = reinterpret_cast<const uint32_t*>(xr[h] + H);
        }
        if constexpr (NH > 1) {
          if (threadIdx.x < kTileN) {  // token tk's row (past the tile's tokens: its first token)
            const int tk = int(threadIdx.x) < ntok ? int(threadIdx.x) : 0;
            int w, i;
            token_src(m, el, tile * kTileN + tk, W, w, i);
            const uint8_t* row = pool + ((size_t(el) * W + w) * TCAP + i) * RX;
#if MEGAMOE_SM120_GIN_DEVICE
            if (is_gin(a, w)) row = a.gin.inbox + w * GB + SB + s_tab[w][i] * RX;
#endif
            s_xr[threadIdx.x] = row;
            s_xs[threadIdx.x] = reinterpret_cast<const uint32_t*>(row + H);
          }
          __syncthreads();
        }
        item_gemm<NH>(acc, w13_of(a, el), ws13_of(a, el), 2 * I / 16, nb0, H, xr, xs, lane,
                      (ntok + kTile - 1) / kTile, s_xr, s_xs, s_sx, s_ss);
        // SwiGLU: acc rows g (gate) / g+8 (up) of block nb -> inter col nb*8+g; tokens 2c, 2c+1
        constexpr int kItemInter = kRowsPerItem / 2;  // inter cols per item (multiple of 32)
        constexpr int kGroups = kItemInter / 32;
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 8 + g;  // 0..kItemInter-1 within the item
#pragma unroll
          for (int th = 0; th < NH; ++th)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const float gate = fminf(acc[jb][th][h], a.swiglu_limit);
              const float up = fminf(fmaxf(acc[jb][th][2 + h], -a.swiglu_limit), a.swiglu_limit);
              s_tile[th * 8 + 2 * c + h][col] = gate / (1.f + __expf(-gate)) * up;
            }
        }
        __syncthreads();
        if (threadIdx.x < kTileN * kGroups) {  // one thread per (token, K32 group)
          const int tk = threadIdx.x / kGroups, gi = threadIdx.x % kGroups;
          float v[32];
#pragma unroll
          for (int u = 0; u < 32; ++u) v[u] = s_tile[tk][gi * 32 + u];
          uint32_t q8[8];
          const uint8_t sc = quant32(v, q8);
          if (tk < ntok) {
            const size_t r = size_t(el) * TOKE + tile * kTileN + tk;
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
        const uint8_t* xr[NH];
        const uint32_t* xs[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          const size_t r = size_t(el) * TOKE + jt[h];
          xr[h] = a.inter + r * I;
          xs[h] = a.inter_s + r * (I / 128);
        }
        if constexpr (NH > 1) {
          if (threadIdx.x < kTileN) {
            const size_t r = size_t(el) * TOKE + tile * kTileN +
                             (int(threadIdx.x) < ntok ? int(threadIdx.x) : 0);
            s_xr[threadIdx.x] = a.inter + r * I;
            s_xs[threadIdx.x] = a.inter_s + r * (I / 128);
          }
          __syncthreads();
        }
        item_gemm<NH>(acc, w2_of(a, el), ws2_of(a, el), H / 16, nb0, I, xr, xs, lane,
                      (ntok + kTile - 1) / kTile, s_xr, s_xs, s_sx, s_ss);
#pragma unroll
        for (int jb = 0; jb < kNbPerWarp; ++jb) {
          const int col = (warp * kNbPerWarp + jb) * 16 + g;  // 0..kRowsPerItem-1 within the item
#pragma unroll
          for (int th = 0; th < NH; ++th)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              s_tile[th * 8 + 2 * c + h][col] = acc[jb][th][h];
              s_tile[th * 8 + 2 * c + h][col + 8] = acc[jb][th][2 + h];
            }
        }
        __syncthreads();
        // push: token tk of this tile -> its source rank's combine slot, kRowsPerItem bf16 in 16 B
        // stores
        constexpr int kVec = kRowsPerItem / 8;
        for (int u = threadIdx.x; u < ntok * kVec; u += blockDim.x) {
          const int tk = u / kVec, q = u % kVec;
          int w, i;
          token_src(m, el, tile * kTileN + tk, W, w, i);
          uint32_t pk[4];
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const __nv_bfloat162 b2 =
                __floats2bfloat162_rn(s_tile[tk][q * 8 + 2 * p], s_tile[tk][q * 8 + 2 * p + 1]);
            pk[p] = *reinterpret_cast<const uint32_t*>(&b2);
          }
          __nv_bfloat16* dst;
#if MEGAMOE_SM120_GIN_DEVICE
          if (is_gin(a, w))  // staged in the source's packed (el, slot) order
            dst = a.gin.stage_comb + (size_t(w) * TCAP * TOPK + s_pw[w] + i) * H;
          else
#endif
          {
            const int meta = *reinterpret_cast<const int*>(
                pool + ((size_t(el) * W + w) * TCAP + i) * RX + H + H / 32);
            dst = a.peers.comb[w] + (size_t(meta / 8) * TOPK + meta % 8) * H;
          }
          *reinterpret_cast<uint4*>(dst + rb * kRowsPerItem + q * 8) =
              make_uint4(pk[0], pk[1], pk[2], pk[3]);
        }
        __threadfence_system();
        __syncthreads();
#if MEGAMOE_SM120_GIN_DEVICE
        if (a.gin.mask) {  // the tile's last FC2 item puts its finished rows to GIN sources
          if (threadIdx.x == 0) {
            s_item = atomicAdd(ctr + kCtrTile + tile_g, 1) == nblk2 - 1;
            __threadfence();
          }
          __syncthreads();
          // the tile's tokens are in (source, slot) order: one put per run of a GIN source
          __shared__ int s_tw[kTileN], s_ti[kTileN];
          if (s_item && threadIdx.x < ntok)
            token_src(m, el, tile * kTileN + threadIdx.x, W, s_tw[threadIdx.x], s_ti[threadIdx.x]);
          __syncthreads();
          if (s_item && threadIdx.x < ntok && is_gin(a, s_tw[threadIdx.x]) &&
              (threadIdx.x == 0 || s_tw[threadIdx.x - 1] != s_tw[threadIdx.x])) {
            const int w = s_tw[threadIdx.x];
            int n = 1;
            while (threadIdx.x + n < ntok && s_tw[threadIdx.x + n] == w) ++n;
            const size_t r = size_t(s_pw[w] + s_ti[threadIdx.x]);  // packed row, same both sides
            gin_put(a, w, a.gin.off_comb_gin + (size_t(a.rank) * TCAP * TOPK + r) * H * 2,
                    a.gin.off_stage_comb + (size_t(w) * TCAP * TOPK + r) * H * 2, size_t(n) * H * 2,
                    ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigComb + a.rank), uint64_t(n)});
          }
          __syncthreads();
        }
#endif
        if (threadIdx.x == 0) {
          s_item = atomicAdd(ctr + kCtrL2Done, 1) == n2 - 1;
          if (s_item) __threadfence_system();
        }
        __syncthreads();
        if (s_item && threadIdx.x < W) {  // this rank's last FC2 item: tell every source rank, a
                                          // thread per rank
          const int w = threadIdx.x;
#if MEGAMOE_SM120_GIN_DEVICE
          if (is_gin(a, w))  // "done": the source counts it after the rows
            gin_signal(a, w, ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigComb + a.rank), 1});
          else
#endif
            st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
          st_[15] = gtime();
        }
      }
      __syncthreads();
    }
    st_[3] = gtime();
    if (n2 == 0 && blockIdx.x == 0 && threadIdx.x < W) {  // no local work: still release every
                                                          // source, a thread per rank
      const int w = threadIdx.x;
      __threadfence_system();
#if MEGAMOE_SM120_GIN_DEVICE
      if (is_gin(a, w))
        gin_signal(a, w, ncclGin_WeakSignalAdd{ncclGinSignal_t(kSigComb + a.rank), 1});
      else
#endif
        st_release_sys(a.peers.comb_flag[w] + a.rank, epoch);
    }
  }

  // ---------------- combine: wait for every expert rank, out[t] = sum_k w[t][k] * partial[t][k]
#if MEGAMOE_SM120_GIN_DEVICE
  __shared__ int s_rows[kMaxWorld];  // this rank's routes per expert rank: GIN rows to expect
  if (a.gin.mask) {
    if (threadIdx.x < W) s_rows[threadIdx.x] = 0;
    __syncthreads();
    for (int u = threadIdx.x; u < a.T * TOPK; u += blockDim.x) {
      const int e = route_at(a, u / TOPK, u % TOPK);
      if (e >= 0) atomicAdd(&s_rows[e / EL], 1);
    }
    __syncthreads();
  }
#endif
  {  // GIN rank w: one signal per pushed-back row, plus its "done"
#if MEGAMOE_SM120_GIN_DEVICE
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x < W && is_gin(a, threadIdx.x))
      ctr[kCtrCnt + kMaxWorld + threadIdx.x] = s_rows[threadIdx.x] + 1;
#endif
    bool ok = wait_peers(a, a.peers.comb_flag[a.rank], epoch, kFlagComb, kSigComb, kErrCombineWait,
                         [&](int w) {
#if MEGAMOE_SM120_GIN_DEVICE
                           return gin_cum(ctr)[kMaxWorld + w] + s_rows[w] + 1;
#else
                           return 0ull;
#endif
                         });
#if MEGAMOE_SM120_GIN_DEVICE
    // the packed rows (cidx) were written by every CTA's dispatch
    if (threadIdx.x == 0 && a.gin.mask && !wait_ge(ctr + kCtrDispatch, int(gridDim.x), false)) {
      atomicExch(ctr + kCtrErr, kErrCombineWait);
      ok = false;
    }
#endif
    if (threadIdx.x == 0) s_ok = ok;
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
        const int e = route_at(a, t, k);
        if (e < 0) continue;  // masked route: never dispatched
        const float wk = a.topk_w[t * TOPK + k];
        const __nv_bfloat16* row = comb + (size_t(t) * TOPK + k) * H;
#if MEGAMOE_SM120_GIN_DEVICE
        if (is_gin(a, e / EL))
          row = a.gin.comb_gin + (size_t(e / EL) * TCAP * TOPK + a.gin.cidx[t * TOPK + k]) * H;
#endif
        const uint4 u = *reinterpret_cast<const uint4*>(row + h8);
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
      *reinterpret_cast<unsigned long long*>(ctr + kCtrCalls) += 1;
      ctr[kCtrDispatch] = 0;
      ctr[kCtrQueue] = 0;
      ctr[kCtrL2Done] = 0;
      for (int el = 0; el < EL; ++el) ctr[kCtrL1 + el] = 0;
      if (a.gin.mask) {  // advance the expected GIN totals by this call's counts
        for (int w = 0; w < W; ++w)
          for (int p = 0; p < 2; ++p) {
            gin_cum(ctr)[p * kMaxWorld + w] += ctr[kCtrCnt + p * kMaxWorld + w];
            ctr[kCtrCnt + p * kMaxWorld + w] = 0;
          }
        for (int i = 0; i < kMaxTiles; ++i) ctr[kCtrTile + i] = 0;
      }
      __threadfence();
      ctr[kCtrExit] = 0;
    }
  }
}

}  // namespace flashinfer::megamoe::sm120
