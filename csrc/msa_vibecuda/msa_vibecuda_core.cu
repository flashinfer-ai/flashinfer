// Block-sparse GQA attention kernel for SM100 (B300).
// One CTA handles one (query token, kv head) pair. The CTA streams the q2k-
// selected logical KV blocks (128 tokens each) in 64-token chunks through a
// TMA (cp.async.bulk.tensor) multi-stage ring buffer with mbarrier tracking,
// computes QK^T with mma.sync (m16n8k16), runs an online softmax in exp2
// space, and accumulates P.V in fp32 registers. Each of the 4 warps keeps an
// independent (m, l, acc) softmax state over its own 16-token slice of every
// chunk; the states are merged once at the end of the CTA (exact split-
// softmax merge), so no cross-warp communication is needed inside the loop.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdlib>

#include "msa_vibecuda_common.h"

namespace {

using bf16 = __nv_bfloat16;
using f16 = __half;

constexpr int kThreads = 128;   // 4 warps
constexpr int kHead = 128;      // head dim (fixed by the task)
constexpr int kRows = 16;       // query tile rows (one token x GQA group <= 16)
constexpr int kBlockTok = 128;  // KV block size (fixed by the task)

// Round-19 persistent work queue (PACK pair path). MSA_PERSIST_WAVES gates
// the queue to grids with at least this many full waves over the resident
// CTA pool (structural key: grid size only; 0 disables the queue entirely).
// MSA_PERSIST_REVERSE reverses the claim order so the heaviest causal tiles
// start in the first resident wave instead of forming the tail wave.
#ifndef MSA_PERSIST_WAVES
// MEASURED-OUT ON THIS SUITE (round 19, fixture9 A/B, 6 variants):
// static GigaThread launch 382.7 us; persistent queue 403-409 us (+5-7%)
// split into +11 us PERSIST-codegen overhead (FULLGRID=2 no-queue probe,
// 393.8 us, reverse-neutral) and +14 us per-tile claim atomics (position-
// independent: loop-top/pre-assigned/merge-hidden/final-pair claims all
// land 403-409), because tid0's atomicAdd resolves on the CTA's
// merge-barrier critical path. The hardware distributor is already
// work-conserving at CTA granularity on this 13.8-wave saturated grid, so
// a device queue only adds per-tile ops — there is no exploitable tail on
// the PACK cost gradient. Default OFF (0); the mechanism + knobs stay
// compiled for genuinely CTA-STARVED shapes (set >0 when waves are few and
// tiles long).
#define MSA_PERSIST_WAVES 0
#endif
#ifndef MSA_PERSIST_REVERSE
#define MSA_PERSIST_REVERSE 1
#endif
// A/B knob: 0 = grid capped at the resident pool (multi-claim CTAs);
// 1 = one work item per CTA (claim-exit bookkeeping per tile);
// 2 = one work item per CTA and NO queue op at all (pre-assigned tile,
//     direct return after one tile) — isolates tile-body codegen of the
//     PERSIST=1 instantiation from every queue mechanic.
#ifndef MSA_PERSIST_FULLGRID
#define MSA_PERSIST_FULLGRID 0
#endif
constexpr int kChunkTok = 64;  // tokens per pipeline stage
constexpr int kWarpTok = 16;   // tokens per warp per chunk

using Params = msa_vibecuda::CoreParams;
using msa_vibecuda::KvLayout;

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(src_size));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(mbar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(mbar)),
               "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      "@P bra LAB_DONE_%=;\n"
      "bra LAB_WAIT_%=;\n"
      "LAB_DONE_%=:\n"
      "}\n" ::"r"(smem_u32(mbar)),
      "r"(parity));
}

// ---------------------------------------------------------------------------
// 2-CTA cluster helpers (distributed shared memory): the row-packed prefill
// PAIRP variant runs each query tile as a cluster pair that splits the union
// block list by hardware rank and merges the two FP32 online-softmax states
// through the peer CTA's shared memory before a single final store.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t mapa_cta(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

__device__ __forceinline__ float ldsm_remote_f32(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ void ldsm_remote_v4f32(float (&v)[4], uint32_t addr) {
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
               : "r"(addr));
}

__device__ __forceinline__ void stsm_remote_v4u32(uint32_t addr, uint4 v) {
  asm volatile("st.shared::cluster.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ void stsm_remote_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(v));
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive;\n" ::: "memory");
  asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

template <int RANK>
__device__ __forceinline__ void tma_load(const CUtensorMap* map, uint64_t* mbar, uint32_t dst,
                                         const int32_t* c) {
  if constexpr (RANK == 3) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n" ::"r"(dst),
        "l"(map), "r"(smem_u32(mbar)), "r"(c[0]), "r"(c[1]), "r"(c[2])
        : "memory");
  } else if constexpr (RANK == 4) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n" ::"r"(dst),
        "l"(map), "r"(smem_u32(mbar)), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
        : "memory");
  } else if constexpr (RANK == 5) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n" ::"r"(dst),
        "l"(map), "r"(smem_u32(mbar)), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]), "r"(c[4])
        : "memory");
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

template <typename T>
struct Mma;
template <>
struct Mma<bf16> {
  static __device__ __forceinline__ void run(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  static __device__ __forceinline__ uint32_t pack(float x, float y) {
    __nv_bfloat162 h = __float22bfloat162_rn(make_float2(x, y));
    return *reinterpret_cast<uint32_t*>(&h);
  }
};
template <>
struct Mma<f16> {
  static __device__ __forceinline__ void run(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  static __device__ __forceinline__ uint32_t pack(float x, float y) {
    __half2 h = __float22half2_rn(make_float2(x, y));
    return *reinterpret_cast<uint32_t*>(&h);
  }
};

// TMA SWIZZLE_128B aware byte offset inside a 128B-pitch tile atom
// (64 rows x 64 x 16-bit elems = 8 KB). c16 is the 16 B index within the row.
__device__ __forceinline__ int tile_off(int row, int c16) {
  return (row << 7) + (((c16 ^ (row & 7))) << 4);
}

// Exact e4m3fn -> bf16 conversion of 4 packed bytes -> one u64 of 4 bf16.
// Every finite e4m3 value (and NaN) is exactly representable in bf16, so the
// f16/f32 intermediate steps (each exact) compose to an exact result.
// Replaces the shared-memory LUT version: the LUT cost 16 data-dependent 4B
// LDS per lane per (iteration,tensor) — ~16 KB/warp/chunk of random-bank smem
// reads, the dominant source of bank conflicts in the fp8 convert path.
__device__ __forceinline__ uint64_t cvt_fp8x4_bf16(uint32_t x) {
  uint32_t y01, y23;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(y01) : "h"((unsigned short)(x)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(y23) : "h"((unsigned short)(x >> 16)));
  float f0 = __half2float(__ushort_as_half((unsigned short)(y01)));
  float f1 = __half2float(__ushort_as_half((unsigned short)(y01 >> 16)));
  float f2 = __half2float(__ushort_as_half((unsigned short)(y23)));
  float f3 = __half2float(__ushort_as_half((unsigned short)(y23 >> 16)));
  uint32_t p01, p23;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p01) : "f"(f1), "f"(f0));
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p23) : "f"(f3), "f"(f2));
  return (uint64_t)p01 | ((uint64_t)p23 << 32);
}

// ---------------------------------------------------------------------------
// Shared memory map
// ---------------------------------------------------------------------------

// VSL = V ring slots (KVK != 2 only; fp8 uses joint K+V stages): 1 for the
// wide-topk prefill variant, NSTG for joint-depth pipelines.
template <int KVK, int NSTG, int VSL>
struct Smem {
  static constexpr int kStages = NSTG;
  // [Q 4KB][bar 16B][red 640B][list 160B][lut 512B][pad][RING][CONV(fp8)]
  // All TMA/ldmatrix atom bases are kept 1024B-aligned (SWIZZLE_128B key).
  static constexpr int kQOff = 0;
  static constexpr int kBarOff = kQOff + kRows * 256;
  // Split K/V arrival barriers: slot i tracks K at mbar[i], V at
  // mbar[kStages+i] so QK can start once K lands while V streams in.
  static constexpr int kBarCount = 16;
  static constexpr int kRedOff = kBarOff + kBarCount * 8;
  static constexpr int kListOff = kRedOff + 160 * 4;
  static constexpr int kLutOff = kListOff + 40 * 4;
  static constexpr int kRingOff = 6 * 1024;
  // Row-packed prefill small arrays — overlay the dead fp8 LUT region
  // (kLutOff, 512B). Union list (resolved TMA source + packed
  // (logical_blk << 4) | token_mask word), per-row causal position +
  // token bit, per-token batch/qpos scratch, and the dedup first-occurrence
  // flag word. Total: 160+160+64+64+16+16+8 = 488B <= 512B.
  static constexpr int kMaxUnion = 40;  // structural dispatch cap (T*topk)
  static constexpr int kUBaseOff = kLutOff;
  static constexpr int kUTokOff = kUBaseOff + kMaxUnion * 4;
  static constexpr int kRQPosOff = kUTokOff + kMaxUnion * 4;
  static constexpr int kRBitOff = kRQPosOff + 16 * 4;
  static constexpr int kTBatOff = kRBitOff + 16 * 4;
  static constexpr int kTQpOff = kTBatOff + 4 * 4;
  static constexpr int kUFirstOff = kTQpOff + 4 * 4;
  static constexpr int kStageBytes = (KVK == 2) ? (64 * 128 * 2) : (64 * 256 * 2);
  // bf16/fp16 ring: kStages x K(16KB) + 1 x V(16KB) single-buffered; fp8 ring:
  // kStages x joint K+V stages (16KB) feeding the smem convert tile.
  static constexpr int kTensorBytes = 64 * 128 * 2;  // one 64-tok x 128-dim tensor
  // V ring slots come from the dispatch (structural keys only):
  //  * wide-topk prefill (topk >= 8): 1 slot — the saved 16KB buys the 4th
  //    CTA per SM and a long chunk stream covers the zero-depth V fetch
  //    (b1 q4096 topk16: 1006 -> 918 us, round 8).
  //  * narrow-topk prefill (topk < 8): NSTG slots — <= 8 chunks per token
  //    cannot amortize a zero-depth V fetch (b3 q4096 topk4 paged:
  //    606 us single-slot vs 515 us joint-depth, round-9 A/B).
  //  * decode (NSTG=3): NSTG slots — short decode chunks cannot hide
  //    anything less than joint depth (b128 q1 flat: single-V 106.3 us,
  //    2-slot post-PV issue 95.5 us, joint-depth 88.7 us, rounds 8-9),
  //    and 102KB gives the same 2 CTA/SM as 86KB.
  static constexpr int kVSlots = VSL;
  static constexpr int kRingBytes =
      (KVK == 2) ? (kStages * kStageBytes) : ((kStages + kVSlots) * kTensorBytes);
  static constexpr int kConvOff = (KVK == 2) ? (kRingOff + kStages * kStageBytes) : 0;
  // fp32 merge accumulator overlays the ring (only touched after the loop)
  static constexpr int kAccOff = kRingOff;
  // Per-warp merge strip row pitch (floats). Round 23 measured out pitch
  // 136 (and 136+row-twist) against 132: pitch-136 swaps 1.6M store
  // conflicts for 1.6M load conflicts (st 2.25M->0.63M, ld 0.21M->1.80M)
  // and the row-twist did not collapse the load side (1.80M); fixture9
  // time flat in all three layouts (382.3/380.9/381.0 us, NCU 387 vs
  // 387.2/386.9 us). Conflicts on this path are latency-hidden; keep 132.
  static constexpr int kStripPitch = 132;
  static constexpr int kTotal = (KVK == 2) ? (kConvOff + 2 * 64 * 256) : (kRingOff + kRingBytes);
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// KV kinds: 0 = bf16, 1 = fp16, 2 = fp8e4m3 (converted to bf16 in smem)
// PACK: row-packed prefill — the 16-row tile holds pack_T = 16/group tokens
// x group heads and the CTA streams the DEDUPED UNION of the pack's per-token
// block lists; each union entry carries a group-bit token mask so every MMA
// row computes only the blocks its token selected (i.i.d. selections: ~14 of
// 16 blocks distinct at topk4, so the union costs <= the concatenation while
// quartering the per-CTA fixed costs).
// JV (joint V barriers): V chunks share the K chunk's mbarrier. JV=false
// splits V onto its own per-slot barriers (mbar[kStages + c%VSL]) with V
// issued at consume-top, short-lead style.
// PAIRP (pair loop): one 128-token union block per iteration — QK over both
// 64-token halves with four interleaved independent HMMA chains, one softmax
// chain over 128 tokens, one PV pass per half, one __syncthreads. Two ring
// forms: JV&&NSTG==2 (70KB, 3 CTAs/SM, K+V refilled after the loop-end
// barrier — cross-CTA overlap covers the flight) and !JV&&NSTG==4 (102KB,
// 2 CTAs/SM, K runs a full pair ahead with split V at pair top).
template <typename QT, int KVK, bool PAGED, int NSTG, int VSL, bool PACK, bool JV = (VSL != 1),
          bool PAIRP = false, bool CLUSTER = false, bool PERSIST = false>
__global__ void __launch_bounds__(kThreads, (KVK == 2 && NSTG == 2)
                                                ? 3
                                                : ((VSL == 1) ? 4 : ((PAIRP && JV) ? 3 : 1)))
    msa_sparse_kernel(const __grid_constant__ Params p, const __grid_constant__ CUtensorMap kmap,
                      const __grid_constant__ CUtensorMap vmap) {
  constexpr int TMA_RANK = PAGED ? 4 : 3;
  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // CLUSTER: the pair of CTAs covering clusterDim.x=2 shares one query tile;
  // the hardware cluster rank both selects this CTA's half of the union block
  // list and (rank 0) the final merge/store role.
  const int crank = CLUSTER ? (int)cluster_rank() : 0;
  int n = CLUSTER ? (int)(blockIdx.x >> 1) : (int)blockIdx.x;  // tile id
  int h = blockIdx.y;                                          // kv head

  static_assert(!PACK || KVK != 2, "packed path is bf16/fp16 only");
  // Pair forms: JV joint 4-slot ring (VSL=2), or the 3-buffer form (VSL=1,
  // !JV) — K in slots 0/1 with slot 1 re-armed as V(c+1) mid-pair; 54KB gives
  // 4 CTAs/SM instead of 3 (round 21).
  static_assert(!PAIRP || (NSTG == 2 && ((JV && VSL > 1) || (!JV && VSL == 1))),
                "pair form: JV 4-slot ring or !JV 3-buffer ring, NSTG=2");
  static_assert(!PAIRP || JV || !CLUSTER, "3-buffer pair is single-CTA only");
  static_assert(!PAIRP || JV || !PERSIST, "3-buffer pair is non-persistent");
  static_assert(!CLUSTER || (PACK && PAIRP && JV && NSTG == 2),
                "cluster form is the packed pair variant only");
  static_assert(!PERSIST || (PACK && PAIRP && JV && NSTG == 2 && !CLUSTER),
                "persistent queue form is the packed pair variant only");

  using S = Smem<KVK, NSTG, VSL>;
  constexpr int kStages = S::kStages;

  char* sm_q = smem_raw + S::kQOff;
  char* sm_ring = smem_raw + S::kRingOff;
  char* sm_conv = smem_raw + S::kConvOff;
  float* sm_acc = reinterpret_cast<float*>(smem_raw + S::kAccOff);
  float* sm_red = reinterpret_cast<float*>(smem_raw + S::kRedOff);
  int* sm_list = reinterpret_cast<int*>(smem_raw + S::kListOff);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + S::kBarOff);
  int* sm_cnt = reinterpret_cast<int*>(smem_raw + S::kListOff + 36 * 4);
  int* sm_ubase = reinterpret_cast<int*>(smem_raw + S::kUBaseOff);
  int* sm_utok = reinterpret_cast<int*>(smem_raw + S::kUTokOff);
  int* sm_rqpos = reinterpret_cast<int*>(smem_raw + S::kRQPosOff);
  int* sm_rtbit = reinterpret_cast<int*>(smem_raw + S::kRBitOff);
  int* sm_tbat = reinterpret_cast<int*>(smem_raw + S::kTBatOff);
  int* sm_tqp = reinterpret_cast<int*>(smem_raw + S::kTQpOff);

  // ---- persistent work scheduling (round 19; PACK pair path only) ---------
  // A device work queue replaces the one-CTA-per-tile grid: grid.x CTAs
  // (min of work count and the resident pool) claim (tile, kv-head) work
  // items with one atomicAdd per tile, so the last claimed tiles fill the
  // resident slots greedily instead of forming a launch-quantized tail wave
  // out of the heaviest causal tiles (claim order optionally reversed:
  // heaviest-first, LPT-style). wphase[] carries the two joint ring
  // barriers' phase counts across claimed tiles (the barriers are initialized
  // once and reused for every claim); the pair-loop wait sites' parities
  // become (carried + per-tile use) & 1. Non-persistent instantiations keep
  // wphase == 0, so their wait parities are bit-identical to the previous
  // form. The claim cell lives in the PACK array region's 24B tail (kMaxUnion
  // staging uses 488 of the 512B LUT overlay; the uint64 ufirst word is at
  // +480).
  int* sm_claim = reinterpret_cast<int*>(smem_raw + S::kUFirstOff + 8);
  int wphase[2] = {0, 0};
  bool first_tile = true;
  if constexpr (PERSIST) {
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < 2 * kStages; ++s) mbar_init(&mbar[s], 1);
    }
    asm volatile("fence.proxy.async.shared::cta;\n");
    __syncthreads();  // barriers visible
  }
  for (;;) {
    if constexpr (PERSIST) {
      int raw;
      if (first_tile) {
        // Work item blockIdx.x is pre-assigned: no queue traffic at wave 0.
        first_tile = false;
        raw = (int)blockIdx.x;
      } else {
        // The next work item was already claimed during the previous tile's
        // main loop; this barrier only drains the previous tile's smem reads
        // (merge strips, PACK arrays) before this tile's prologue overwrites.
        __syncthreads();
        raw = *sm_claim;
      }
      if (raw >= p.ws_total) return;
      const int w = MSA_PERSIST_REVERSE ? (p.ws_total - 1 - raw) : raw;
      n = w % p.ws_ntiles;
      h = w / p.ws_ntiles;
    }
    const int ntok0 = PACK ? n * p.pack_T : n;
    int b = 0, qpos = 0, cu_kb = 0;
    if constexpr (!PACK) {
      // ---- batch/locals -----------------------------------------------------
      int lo = 0, hi = p.nbatch - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (__ldg(p.cu_q + mid) <= n)
          lo = mid;
        else
          hi = mid - 1;
      }
      b = lo;
      const int local_q = n - __ldg(p.cu_q + b);
      const int kv_len = __ldg(p.cu_k + b + 1) - __ldg(p.cu_k + b);
      qpos = kv_len - p.seqlen_q + local_q;
      cu_kb = __ldg(p.cu_k + b);
    }

    // ---- zero reduction state -------------------------------------------------
    {
      if (tid < 16) sm_red[128 + tid] = 0.f;
      if (tid == 0) *sm_cnt = 0;
      // First-occurrence flag word — the PACK build's phase C1 publishes its
      // per-warp ballot fragments with atomicOr, so it must start at zero.
      if (tid == 16) *reinterpret_cast<uint64_t*>(smem_raw + S::kUFirstOff) = 0;
    }
    if constexpr (!PERSIST) {
      if (tid == 0) {
#pragma unroll
        for (int s = 0; s < 2 * kStages; ++s) mbar_init(&mbar[s], 1);
      }
    }
    asm volatile("fence.proxy.async.shared::cta;\n");
    __syncthreads();  // sm_cnt + barriers visible

    // ---- Q tile issue (cp.async, one group) — hoisted before the list/union
    // build (zero dependencies on it) so the global flight overlaps the
    // build's gmem latency chain. Two 128B-pitch half atoms.
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      int i = tid + it * kThreads;
      int r = i >> 4, s = i & 15;
      int t = PACK ? (r / p.group) : 0;
      int hh = PACK ? (r - t * p.group) : r;
      long nn = PACK ? (long)(ntok0 + t) : (long)n;
      bool rv = PACK ? (nn < p.total_q) : (r < p.group);
      int sz = rv ? 16 : 0;
      const QT* src = reinterpret_cast<const QT*>(p.q) + nn * p.q_tok +
                      (long)(h * p.group + hh) * p.q_head + s * 8;
      cp16(smem_u32(sm_q + (s >> 3) * (kRows * 128) + tile_off(r, s & 7)), src, sz);
    }
    cp_commit();

    // ---- compact the q2k block list -----------------------------------------
    if constexpr (PACK) {
      // Deduped union build, parallelized across all 4 warps (round 17). The
      // round-16 form ran the whole build on warp 0 — a serial gmem-latency
      // chain (cu_q search -> q2k gather -> page_table) plus O(nent^2) smem
      // scans while 3 warps idled at the barrier (~30% of CTA wall per the
      // round-15/16 clock and cluster A/B evidence). Phases below overlap the
      // independent load streams (maps || gather || the hoisted Q flight) and
      // split the filter/key/dedup passes over contiguous per-warp slices.
      // Dedup key: paged -> physical page id (globally unique), flat -> global
      // flat token base cu_k[b] + blk*128 (unique across batches); both +1 so 0
      // marks an invalid entry. Entry word: (logical_blk << 4) | token_bits.
      // Staging arrays overlay ring slot 0 — dead before the first chunk issue.
      // CLUSTER: only rank 0 builds the union; rank 1 consumes the DSM push
      // (union sharing is mandatory — duplicating the build per CTA of the
      // pair made the cluster form strictly worse, round-17 A/B).
      if (!CLUSTER || crank == 0) {
        int* skey = reinterpret_cast<int*>(smem_raw + S::kRingOff);
        int* stok = skey + S::kMaxUnion;
        uint64_t* sm_ufirst = reinterpret_cast<uint64_t*>(smem_raw + S::kUFirstOff);
        const int nent = p.pack_T * p.topk;
        // ---- phase A: token maps (warp 0) || raw q2k gather (warp 1) -------
        if (warp == 0) {
          if (lane < p.pack_T) {
            long nn = ntok0 + lane;
            int bat = 0, qp = -1;
            if (nn < p.total_q) {
              int lo2 = 0, hi2 = p.nbatch - 1;
              while (lo2 < hi2) {
                int mid = (lo2 + hi2 + 1) >> 1;
                if (__ldg(p.cu_q + mid) <= nn)
                  lo2 = mid;
                else
                  hi2 = mid - 1;
              }
              bat = lo2;
              int lq = (int)(nn - __ldg(p.cu_q + bat));
              qp = (int)(__ldg(p.cu_k + bat + 1) - __ldg(p.cu_k + bat)) - p.seqlen_q + lq;
            }
            sm_tbat[lane] = bat;
            sm_tqp[lane] = qp;
          }
          __syncwarp();
          for (int r = lane; r < kRows; r += 32) {
            int t = r / p.group;
            sm_rqpos[r] = sm_tqp[t];
            sm_rtbit[r] = (ntok0 + t < p.total_q) ? (1 << t) : 0;
          }
        } else if (warp == 1) {
          // Raw (unfiltered) block ids — the q2k load stream is independent of
          // the warp-0 cu_q/cu_k map chain, so the two gmem rounds overlap.
          // power-of-two topk uses shift/mask instead of the ~20-instr runtime
          // s32 division sequence; exhausted NCU sampler attribution shows this
          // idiom is a real hotspot in sibling kernels.
          if ((p.topk & (p.topk - 1)) == 0) {
            const int tsh = __ffs(p.topk) - 1;
            for (int idx = lane; idx < nent; idx += 32) {
              int t = idx >> tsh;
              long nn = ntok0 + t;
              int blk = -1;
              if (nn < p.total_q)
                blk = __ldg(p.q2k + (long)h * p.q2k_h + nn * p.q2k_n + (idx & (p.topk - 1)));
              skey[idx] = blk;
            }
          } else {
            for (int idx = lane; idx < nent; idx += 32) {
              int t = idx / p.topk;
              long nn = ntok0 + t;
              int blk = -1;
              if (nn < p.total_q)
                blk = __ldg(p.q2k + (long)h * p.q2k_h + nn * p.q2k_n + (idx - t * p.topk));
              skey[idx] = blk;
            }
          }
        }
        __syncthreads();
        // ---- phase B: filter + key/token words, 4-warp contiguous slices ----
        const int per = (nent + 3) >> 2;
        const int i0 = warp * per;
        const int tsh = __ffs(p.topk) - 1;
        const bool tpow2 = (p.topk & (p.topk - 1)) == 0;
        for (int idx = i0 + lane; idx < nent && idx < i0 + per; idx += 32) {
          int t = tpow2 ? (idx >> tsh) : (idx / p.topk);
          int blk = skey[idx];  // raw block id from phase A
          int key = 0, tk = 0;
          if (blk >= 0 && (!p.causal || blk * kBlockTok <= sm_tqp[t])) {
            long base = PAGED ? (long)__ldg(p.page_table + (long)sm_tbat[t] * p.pt_stride + blk)
                              : (long)__ldg(p.cu_k + sm_tbat[t]) + (long)blk * kBlockTok;
            key = (int)base + 1;
            tk = (blk << 4) | (1 << t);
          }
          skey[idx] = key;
          stok[idx] = tk;
        }
        __syncthreads();
        // ---- phase C1: first-occurrence flags (one ballot per warp slice) ---
        {
          bool f = false;
          int idx = i0 + lane;
          if (idx < nent && idx < i0 + per) {
            int ki = skey[idx];
            f = ki > 0;
            if (f)
              for (int j = 0; j < idx; ++j)
                if (skey[j] == ki) {
                  f = false;
                  break;
                }
          }
          unsigned bal = __ballot_sync(0xffffffffu, f);
          if (lane == 0 && bal)
            atomicOr(reinterpret_cast<unsigned long long*>(sm_ufirst),
                     (unsigned long long)bal << i0);
        }
        __syncthreads();
        const uint64_t fmask = *sm_ufirst;
        // ---- phase C2: survivors store; position counts DISTINCT smaller
        // keys (first-occurrence flags); duplicates merge token bits after ----
        {
          int idx = i0 + lane;
          bool f = false;
          if (idx < nent && idx < i0 + per) {
            int key = skey[idx];
            f = key > 0 && ((fmask >> idx) & 1);
            if (f) {
              int pos = 0;
              for (int j = 0; j < nent; ++j) {
                int kj = skey[j];
                pos += (kj > 0 && kj < key && ((fmask >> j) & 1));
              }
              sm_ubase[pos] = key - 1;
              sm_utok[pos] = stok[idx];
            }
          }
          unsigned sbal = __ballot_sync(0xffffffffu, f);
          if (lane == 0 && sbal) atomicAdd(sm_cnt, __popc(sbal));
        }
        __syncthreads();  // survivor stores visible before duplicate merge
        for (int idx = i0 + lane; idx < nent && idx < i0 + per; idx += 32) {
          int key = skey[idx];
          if (key > 0 && !((fmask >> idx) & 1)) {
            int pos = 0;
            for (int j = 0; j < nent; ++j) {
              int kj = skey[j];
              pos += (kj > 0 && kj < key && ((fmask >> j) & 1));
            }
            atomicOr(&sm_utok[pos], stok[idx] & 0xF);
          }
        }
        __syncthreads();  // union final CTA-wide
        // CLUSTER: hand the resolved union + row maps to rank 1 over DSM —
        // ubase[40] + utok[40] + rqpos[16] + rtbit[16] = 112 ints = 448
        // contiguous bytes from kUBaseOff (16B aligned), plus the entry count.
        // The cluster arrival below carries release semantics, and the CTA
        // barrier + arrival chain makes these stores visible to rank 1.
        if constexpr (CLUSTER) {
          if (warp == 0) {
            const uint32_t r_ub = mapa_cta(smem_u32(sm_ubase), 1);
            const uint4* src4 = reinterpret_cast<const uint4*>(sm_ubase);
            for (int i = lane; i < 28; ++i) stsm_remote_v4u32(r_ub + i * 16, src4[i]);
            if (lane == 0) stsm_remote_u32(mapa_cta(smem_u32(sm_cnt), 1), (uint32_t)*sm_cnt);
          }
        }
      }
      __syncthreads();                        // build + (CLUSTER: rank-0 push issued) CTA-wide
      if constexpr (CLUSTER) cluster_sync();  // pushed arrays visible to rank 1
    } else {
      if (warp == 0) {
        for (int j = lane; j < p.topk; j += 32) {
          int blk = __ldg(p.q2k + (long)h * p.q2k_h + (long)n * p.q2k_n + j);
          bool ok = (blk >= 0) && (!p.causal || (blk * kBlockTok) <= qpos);
          if (ok) sm_list[atomicAdd(sm_cnt, 1)] = blk;
        }
      }
      __syncthreads();  // list + count visible
    }
    const int nblk = *sm_cnt;
    const int nchunks = nblk * 2;  // 64-token chunks
    // CLUSTER: this CTA consumes union blocks crank, crank+2, ... — local chunk
    // count is 2x the local pair count; union block index of local chunk c is
    // crank + (c & ~1). Grid covers every union block exactly once per tile.
    const int nloc = CLUSTER ? (((nblk - crank + 1) >> 1) * 2) : nchunks;

    // ---- TMA chunk issue helpers -------------------------------------------------
    // bf16/fp16 joint-depth rings (VSL > 1): one mbarrier covers K+V per chunk
    // (round-7 form). The single-V prefill variant (VSL == 1) splits K and V
    // onto separate barriers so QK starts at K arrival while V streams in.
    // fp8 keeps one joint barrier because convert_chunk consumes both tensors
    // together.
    auto issue_chunk = [&](int c, int bidx) {
      // PACK: union entries carry the resolved TMA source (paged: physical
      // page; flat: global token base) so no page_table/cu_k lookup is needed.
      // bidx >= 0 (cluster pair) overrides the union index of chunk c.
      int blk = PACK ? sm_ubase[bidx < 0 ? (c >> 1) : bidx] : sm_list[c >> 1];
      int chalf = c & 1;
      char* dst = sm_ring + (c % kStages) * S::kStageBytes;
      if (KVK == 2) {
        uint64_t* mb = &mbar[c % kStages];
        mbar_expect_tx(mb, S::kStageBytes);
        int32_t gc[4];
        if (PAGED) {
          int page = __ldg(p.page_table + (long)b * p.pt_stride + blk);
          gc[0] = 0;
          gc[1] = chalf * kChunkTok;
          gc[2] = h;
          gc[3] = page;
        } else {
          gc[0] = 0;
          gc[1] = h;
          gc[2] = cu_kb + blk * kBlockTok + chalf * kChunkTok;
        }
        tma_load<TMA_RANK>(&kmap, mb, smem_u32(dst), gc);
        tma_load<TMA_RANK>(&vmap, mb, smem_u32(dst + 64 * 128), gc);
      } else {
        uint64_t* mbk = &mbar[c % kStages];
        // Joint-barrier regime (VSL > 1): ONE expect covers K+V — round-7's
        // form. Splitting K/V barriers measured a win only for the single-V
        // prefill variant; on joint-depth rings the extra PV-side mbarrier
        // spin cost ~3-4% (b128 q4 flat 211 -> 220 us, b3 q4096 topk4 paged
        // 556 -> 574 us, round-9 A/B).
        mbar_expect_tx(mbk, (JV ? 2 : 1) * S::kTensorBytes);
        char* dst_k = sm_ring + (c % kStages) * S::kTensorBytes;
        char* dst_v = sm_ring + (kStages + (c % S::kVSlots)) * S::kTensorBytes;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          int32_t gc[4];
          if (PAGED) {
            int page = PACK ? blk : __ldg(p.page_table + (long)b * p.pt_stride + blk);
            gc[0] = half * 64;
            gc[1] = chalf * kChunkTok;
            gc[2] = h;
            gc[3] = page;
          } else {
            gc[0] = half * 64;
            gc[1] = h;
            gc[2] =
                PACK ? (blk + chalf * kChunkTok) : (cu_kb + blk * kBlockTok + chalf * kChunkTok);
          }
          tma_load<TMA_RANK>(&kmap, mbk, smem_u32(dst_k + half * (64 * 128)), gc);
          if (JV) tma_load<TMA_RANK>(&vmap, mbk, smem_u32(dst_v + half * (64 * 128)), gc);
        }
      }
    };

    // V half of chunk c (single-V prefill variant only; joint rings load V
    // inside issue_chunk): the sole V slot lives past the K slots at
    // sm_ring + kStages. Issued at chunk-consume top (write window =
    // QK+softmax of c); PV(c)'s ldmatrix reads are fenced from the next V
    // write by the loop's per-chunk __syncthreads. Barrier phase toggles per
    // use (parity c&1).
    auto issue_v = [&](int c) {
      int blk = PACK ? sm_ubase[c >> 1] : sm_list[c >> 1];
      int chalf = c & 1;
      uint64_t* mbv = &mbar[kStages + (c % S::kVSlots)];
      mbar_expect_tx(mbv, S::kTensorBytes);
      char* dst_v = sm_ring + kStages * S::kTensorBytes + (c % S::kVSlots) * S::kTensorBytes;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int32_t gc[4];
        if (PAGED) {
          int page = PACK ? blk : __ldg(p.page_table + (long)b * p.pt_stride + blk);
          gc[0] = half * 64;
          gc[1] = chalf * kChunkTok;
          gc[2] = h;
          gc[3] = page;
        } else {
          gc[0] = half * 64;
          gc[1] = h;
          gc[2] = PACK ? (blk + chalf * kChunkTok) : (cu_kb + blk * kBlockTok + chalf * kChunkTok);
        }
        tma_load<TMA_RANK>(&vmap, mbv, smem_u32(dst_v + half * (64 * 128)), gc);
      }
    };

    // ---- 3-buffer pair-ring issues (PACK pair !JV form, round 21) -------------
    // Slot map in the 48KB ring: S0=[0,16KB) K, S1=[16KB,32KB) K then re-armed
    // as V mid-pair, SLV=[32KB,48KB) the single rolling V slot. Barriers: b0 /
    // b1 / bV = mbar[0/1/2]. Per pair p: b0 and bV see one use each (parity
    // p&1); b1 sees two uses — K(c+1) (use 2p, parity 0) then V(c+1) (use
    // 2p+1, parity 1) — all phases issued strictly in order by tid0.
    auto issue3b_k = [&](int c, int slot) {
      int blk = sm_ubase[c >> 1];
      int chalf = c & 1;
      uint64_t* mb = &mbar[slot];
      mbar_expect_tx(mb, S::kTensorBytes);
      char* dst = sm_ring + slot * S::kTensorBytes;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int32_t gc[4];
        if (PAGED) {
          gc[0] = half * 64;
          gc[1] = chalf * kChunkTok;
          gc[2] = h;
          gc[3] = blk;
        } else {
          gc[0] = half * 64;
          gc[1] = h;
          gc[2] = blk + chalf * kChunkTok;
        }
        tma_load<TMA_RANK>(&kmap, mb, smem_u32(dst + half * (64 * 128)), gc);
      }
    };
    auto issue3b_v = [&](int c, int slot) {
      int blk = sm_ubase[c >> 1];
      int chalf = c & 1;
      uint64_t* mb = &mbar[slot];
      mbar_expect_tx(mb, S::kTensorBytes);
      char* dst = sm_ring + slot * S::kTensorBytes;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        int32_t gc[4];
        if (PAGED) {
          gc[0] = half * 64;
          gc[1] = chalf * kChunkTok;
          gc[2] = h;
          gc[3] = blk;
        } else {
          gc[0] = half * 64;
          gc[1] = h;
          gc[2] = blk + chalf * kChunkTok;
        }
        tma_load<TMA_RANK>(&vmap, mb, smem_u32(dst + half * (64 * 128)), gc);
      }
    };

    if (tid == 0) {
      // Pair mode pre-issues only the first 128-token pair; the pair loop
      // pre-issues chunk c+2/c+3 at pair top (split-V) or past the loop-end
      // barrier (joint), so a kStages-1 prologue would double-issue chunk 2
      // (two arrives on one mbarrier phase -> trap).
      const int npro = PAIRP ? 2 : (kStages - 1);
      if constexpr (CLUSTER) {
        // Local chunks 0/1 are the two halves of this rank's first union block.
        for (int c = 0; c < npro && c < nloc; ++c) issue_chunk(c, crank);
      } else if constexpr (PAIRP && !JV) {
        // 3-buffer: K(0)->S0, K(1)->S1, V(0)->SLV (nloc is even when nonzero).
        if (nloc > 0) {
          issue3b_k(0, 0);
          issue3b_k(1, 1);
          issue3b_v(0, 2);
        }
      } else {
        for (int c = 0; c < npro && c < nchunks; ++c) issue_chunk(c, -1);
      }
    }

    cp_wait<0>();
    __syncthreads();  // Q tile visible

    // ---- load Q fragments into registers -------------------------------------
    // The 3-buffer pair form (!JV) skips this persistent staging to stay at the
    // 128-reg/thread bound of 4 CTAs/SM; its QK restages Q from smem per pair
    // (persistent-Q at 128 regs spilled 32B and measured 435.9 vs 394.3 f9).
    uint32_t qa[8][4];  // [k16 step][a0..a3]
    if constexpr (!(PAIRP && !JV)) {
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        int row = lane & 15;
        int seg = (kk & 3) * 2 + (lane >> 4);
        ldsm_x4(qa[kk], smem_u32(sm_q + (kk >> 2) * (kRows * 128) + tile_off(row, seg)));
      }
    }

    // ---- per-warp softmax state ----------------------------------------------
    float m0 = -1e38f, m1 = -1e38f, l0 = 0.f, l1 = 0.f;
    float acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

    const int kcol0 = (lane & 3) * 2;

    // PACK: this thread's two fragment rows belong to (possibly) two packed
    // tokens — hoist their causal positions and token-mask bits to registers.
    int qp0 = 0, qp1 = 0, bit0 = 0, bit1 = 0;
    if constexpr (PACK) {
      const int r0 = lane >> 2;
      qp0 = sm_rqpos[r0];
      qp1 = sm_rqpos[r0 + 8];
      bit0 = sm_rtbit[r0];
      bit1 = sm_rtbit[r0 + 8];
    }

    // fp8 -> bf16 conversion of this warp's 16 rows of K and V.
    // uint4 loads (16 fp8) -> four u64 stores into the swizzled split atoms.
    auto convert_chunk = [&](int c) {
      const char* rk = sm_ring + (c % kStages) * S::kStageBytes;
      const char* rv = rk + 64 * 128;
      const int r0 = warp * kWarpTok;
#pragma unroll
      for (int tensor = 0; tensor < 2; ++tensor) {
        const char* src = tensor ? rv : rk;
        char* ct = sm_conv + tensor * (2 * 64 * 128);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
          int i = lane + it * 32;  // 0..127
          int r = r0 + (i >> 3);   // row within tile
          int s = i & 7;           // 16B seg of raw row
          uint4 raw = *reinterpret_cast<const uint4*>(src + r * 128 + s * 16);
          char* hbase = ct + (s >> 2) * (64 * 128);
          int ws0 = 2 * (s & 3);  // within-half 16B seg base
          uint32_t xs[4] = {raw.x, raw.y, raw.z, raw.w};
          // Bank-conflict-free store schedule: the u64 for raw word t lands in
          // bank pair P = ((2*(s&3) + (t>>1)) ^ (r&7))*2 + (t&1). With lockstep
          // t-order every lane of the warp hits only 8 of the 16 bank pairs per
          // store instruction (~4-way conflict, measured 40% excessive smem
          // wavefronts). Rotating the per-lane store order by
          // g = ((s&3) ^ (r&3)) ^ ((s>>2)*2) spreads each instruction's 32
          // stores over all 16 bank pairs exactly twice (2 wavefronts, the
          // minimum for 8B stores) — verified by enumeration of (M,x,h)
          // groups; each lane still stores all 4 of its words over 4 steps.
          int g = ((s & 3) ^ (r & 3)) ^ ((s >> 2) << 1);
          // Convert the 4 raw words to bf16 u64s with STATIC indices (dynamic
          // xs[t] would force the array to local memory).
          uint64_t ws_w[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            ws_w[t] = cvt_fp8x4_bf16(xs[t]);
          }
          // Two static segment addresses; per-step SEL picks word/address per
          // the rotated schedule (lockstep, no divergence, no local memory).
          char* a0 = hbase + tile_off(r, ws0);
          char* a1 = hbase + tile_off(r, ws0 + 1);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int t = (j + g) & 3;
            uint64_t wlo = (t & 2) ? ws_w[2] : ws_w[0];
            uint64_t whi = (t & 2) ? ws_w[3] : ws_w[1];
            uint64_t w = (t & 1) ? whi : wlo;
            char* a = ((t & 2) ? a1 : a0) + ((t & 1) << 3);
            *reinterpret_cast<uint64_t*>(a) = w;
          }
        }
      }
      __syncwarp();
    };

    // ---- consume one 64-token chunk -------------------------------------------
    // bf16/fp16 ring layout: [K x kStages slots 16KB each | V x kVSlots slots].
    // Prefill (NSTG=2): single V slot — V(c) is issued at chunk top and its
    // write window only has to beat the post-softmax PV read; the saved 16KB
    // raises the CTA limit 3 -> 4 per SM. Decode (NSTG=3): full-depth V ring
    // issued at K-pipeline lookahead (see main loop). fp8 keeps the joint
    // K+V stage (convert path).
    auto consume_chunk = [&](int c) {
      const char* tk;
      const char* tv;
      if (KVK == 2) {
        tk = sm_conv;
        tv = sm_conv + 64 * 256;
      } else {
        tk = sm_ring + (c % kStages) * (64 * 128 * 2);
        tv = sm_ring + (kStages + (c % S::kVSlots)) * (64 * 128 * 2);
      }
      // PACK: union word gives the logical block (for the causal compare — the
      // TMA source is already resolved at issue time) and the token mask.
      int tm = 0;
      int blk;
      if constexpr (PACK) {
        int uw = sm_utok[c >> 1];
        tm = uw & 0xF;
        blk = uw >> 4;
      } else {
        blk = sm_list[c >> 1];
      }
      const int tok_base = blk * kBlockTok + (c & 1) * kChunkTok + warp * kWarpTok;

      float s[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) s[j][q4] = 0.f;

      // S = Q . K^T (this warp's 16 tokens) via x4 ldmatrix on 2 n8 tiles
      // ldsm x4 reg order: (tok0-7,klo),(tok8-15,klo),(tok0-7,khi),(tok8-15,khi)
      if constexpr (PACK) {
        // Dual accumulator chains per n8 tile: the 8-step k loop otherwise
        // serializes 8 dependent HMMA accumulations per tile (~130-190cyc of
        // fixed-latency wait, the top stall at 28.5%). Alternating kk between
        // sb[0]/sb[1] halves the chain; +8 fp32 regs (short-lived, QK scope).
        // The step kk+1 ldmatrix prefetch also rides the MMA issue window.
        float sb[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) sb[j][q4] = 0.f;
        uint32_t bn[4];
        {
          int row = warp * kWarpTok + (lane & 15);
          int seg = (lane >> 4);
          ldsm_x4(bn, smem_u32(tk + tile_off(row, seg)));
        }
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          uint32_t bc[4] = {bn[0], bn[1], bn[2], bn[3]};
          if (kk < 7) {
            int row = warp * kWarpTok + (lane & 15);
            int seg = ((kk + 1) & 3) * 2 + (lane >> 4);
            ldsm_x4(bn, smem_u32(tk + ((kk + 1) >> 2) * (64 * 128) + tile_off(row, seg)));
          }
          uint32_t b0[2] = {bc[0], bc[2]};  // n8 tile 0: klo, khi
          uint32_t b1[2] = {bc[1], bc[3]};  // n8 tile 1: klo, khi
          Mma<QT>::run(kk & 1 ? sb[0] : s[0], qa[kk], b0);
          Mma<QT>::run(kk & 1 ? sb[1] : s[1], qa[kk], b1);
        }
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          s[0][q4] += sb[0][q4];
          s[1][q4] += sb[1][q4];
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          uint32_t b[4];
          int row = warp * kWarpTok + (lane & 15);
          int seg = (kk & 3) * 2 + (lane >> 4);
          ldsm_x4(b, smem_u32(tk + (kk >> 2) * (64 * 128) + tile_off(row, seg)));
          uint32_t b0[2] = {b[0], b[2]};  // n8 tile 0: klo, khi
          uint32_t b1[2] = {b[1], b[3]};  // n8 tile 1: klo, khi
          Mma<QT>::run(s[0], qa[kk], b0);
          Mma<QT>::run(s[1], qa[kk], b1);
        }
      }

      // selection/causal mask in RAW score space (PACK: per-row token bit +
      // per-row causal position). The softmax scale is NOT applied here: the
      // max tree/online state stay unscaled and the scale enters once per row
      // as ns = -mn*scale, making each exp2 argument a single FFMA instead of
      // a scale-FMUL + FSUB pair. Max comparisons are scale-invariant.
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int tok = tok_base + j * 8 + kcol0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          if constexpr (PACK) {
            bool ok0 = (tm & bit0) && (!p.causal || (tok + i) <= qp0);
            bool ok1 = (tm & bit1) && (!p.causal || (tok + i) <= qp1);
            s[j][i] = ok0 ? s[j][i] : -1e38f;
            s[j][2 + i] = ok1 ? s[j][2 + i] : -1e38f;
          } else {
            bool ok = !p.causal || (tok + i) <= qpos;
            s[j][i] = ok ? s[j][i] : -1e38f;
            s[j][2 + i] = ok ? s[j][2 + i] : -1e38f;
          }
        }
      }

      // PV step-0 ldsm prefetch, issued before the exp2/shfl softmax chain so
      // its LDS latency hides under it (jointly-resident on VSL>1 rings);
      // discarded harmlessly on fully-masked chunks.
      uint32_t bv[4];
      if constexpr (PACK) {
        int row = warp * kWarpTok + (lane & 15);
        ldsm_x4_trans(bv, smem_u32(tv + tile_off(row, lane >> 4)));
      }

      float rmax0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
      float rmax1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
      if constexpr (PACK) {
        // 64-bit packed shfl tree: both row maxima reduced with 2 shfls
        // instead of 4 (softmax scalar chain is 28.5% of warp stalls).
        unsigned long long mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
                                (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x1 = __shfl_xor_sync(0xffffffffu, mp, 1);
        rmax0 = fmaxf(__uint_as_float((unsigned)(mp >> 32)), __uint_as_float((unsigned)(x1 >> 32)));
        rmax1 = fmaxf(__uint_as_float((unsigned)mp), __uint_as_float((unsigned)x1));
        mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
             (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x2 = __shfl_xor_sync(0xffffffffu, mp, 2);
        rmax0 = fmaxf(rmax0, __uint_as_float((unsigned)(x2 >> 32)));
        rmax1 = fmaxf(rmax1, __uint_as_float((unsigned)x2));
      } else {
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, 2));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, 2));
      }

      // PACK: this predicate is only quad-uniform — each quad's two rows carry
      // different packed-token bits/causal positions, so one row pair can be
      // fully masked while another is live. The branch body holds warp-
      // collective ops (shfl reduce, ldmatrix, HMMA); divergent entry
      // deadlocks. Promote the test to warp-uniform via __any_sync: fully-
      // masked quads contribute exact zeros (pe guards + the mn>m rescale skip
      // already handle all-(-inf) rows). Non-PACK rows share one token mask,
      // so the quad predicate is already warp-uniform there — keep it.
      const bool anylive = PACK ? __any_sync(0xffffffffu, (rmax0 > -1e37f) || (rmax1 > -1e37f))
                                : ((rmax0 > -1e37f) || (rmax1 > -1e37f));
      if (anylive) {
        float mn0 = fmaxf(m0, rmax0);
        float mn1 = fmaxf(m1, rmax1);
        const float ns0 = -mn0 * p.scale_log2e;  // raw-space max -> scaled exp2 bias
        const float ns1 = -mn1 * p.scale_log2e;
        // skip the accumulator rescale entirely when the row maxima did not move
        if (mn0 > m0 || mn1 > m1) {
          float f0 = exp2f(fmaf(m0, p.scale_log2e, ns0));
          float f1 = exp2f(fmaf(m1, p.scale_log2e, ns1));
          l0 *= f0;
          l1 *= f1;
#pragma unroll
          for (int dt = 0; dt < 16; ++dt) {
            acc[dt][0] *= f0;
            acc[dt][1] *= f0;
            acc[dt][2] *= f1;
            acc[dt][3] *= f1;
          }
        }
        m0 = mn0;
        m1 = mn1;
        float pe[2][4];
        pe[0][0] = (s[0][0] <= -1e37f) ? 0.f : exp2f(fmaf(s[0][0], p.scale_log2e, ns0));
        pe[0][1] = (s[0][1] <= -1e37f) ? 0.f : exp2f(fmaf(s[0][1], p.scale_log2e, ns0));
        pe[0][2] = (s[0][2] <= -1e37f) ? 0.f : exp2f(fmaf(s[0][2], p.scale_log2e, ns1));
        pe[0][3] = (s[0][3] <= -1e37f) ? 0.f : exp2f(fmaf(s[0][3], p.scale_log2e, ns1));
        pe[1][0] = (s[1][0] <= -1e37f) ? 0.f : exp2f(fmaf(s[1][0], p.scale_log2e, ns0));
        pe[1][1] = (s[1][1] <= -1e37f) ? 0.f : exp2f(fmaf(s[1][1], p.scale_log2e, ns0));
        pe[1][2] = (s[1][2] <= -1e37f) ? 0.f : exp2f(fmaf(s[1][2], p.scale_log2e, ns1));
        pe[1][3] = (s[1][3] <= -1e37f) ? 0.f : exp2f(fmaf(s[1][3], p.scale_log2e, ns1));
        float rs0 = pe[0][0] + pe[0][1] + pe[1][0] + pe[1][1];
        float rs1 = pe[0][2] + pe[0][3] + pe[1][2] + pe[1][3];
        if constexpr (PACK) {
          unsigned long long rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
                                  (unsigned long long)__float_as_uint(rs1);
          unsigned long long y1 = __shfl_xor_sync(0xffffffffu, rp, 1);
          rs0 += __uint_as_float((unsigned)(y1 >> 32));
          rs1 += __uint_as_float((unsigned)y1);
          rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
               (unsigned long long)__float_as_uint(rs1);
          unsigned long long y2 = __shfl_xor_sync(0xffffffffu, rp, 2);
          rs0 += __uint_as_float((unsigned)(y2 >> 32));
          rs1 += __uint_as_float((unsigned)y2);
        } else {
          rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);
          rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);
          rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);
          rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);
        }
        l0 += rs0;
        l1 += rs1;

        uint32_t a[4];
        a[0] = Mma<QT>::pack(pe[0][0], pe[0][1]);
        a[1] = Mma<QT>::pack(pe[0][2], pe[0][3]);
        a[2] = Mma<QT>::pack(pe[1][0], pe[1][1]);
        a[3] = Mma<QT>::pack(pe[1][2], pe[1][3]);

        // V of this chunk only needs to be resident by PV time: QK+softmax
        // already gave TMA a full compute window. Only the single-V prefill
        // variant tracks K/V on separate barriers; joint rings and fp8 waited
        // jointly at chunk top.
        if (KVK != 2 && !JV) mbar_wait(&mbar[kStages + (c % S::kVSlots)], (c / S::kVSlots) & 1);

#pragma unroll
        for (int dt = 0; dt < 16; dt += 2) {
          uint32_t b4[4];
          if constexpr (PACK) {
            b4[0] = bv[0];
            b4[1] = bv[1];
            b4[2] = bv[2];
            b4[3] = bv[3];
            if (dt < 14) {  // prefetch the next PV step under this step's HMMA
              int row = warp * kWarpTok + (lane & 15);
              int seg = ((dt + 2) & 7) + (lane >> 4);
              ldsm_x4_trans(bv, smem_u32(tv + ((dt + 2) >> 3) * (64 * 128) + tile_off(row, seg)));
            }
          } else {
            int row = warp * kWarpTok + (lane & 15);
            int seg = (dt & 7) + (lane >> 4);
            ldsm_x4_trans(b4, smem_u32(tv + (dt >> 3) * (64 * 128) + tile_off(row, seg)));
          }
          uint32_t b0[2] = {b4[0], b4[1]};
          uint32_t b1[2] = {b4[2], b4[3]};
          Mma<QT>::run(acc[dt], a, b0);
          Mma<QT>::run(acc[dt + 1], a, b1);
        }
      } else if (KVK != 2 && !JV) {
        // Fully masked chunk (all -inf): PV skipped, but the V barrier must
        // still be consumed before this ring slot is re-armed, otherwise a
        // late V(c) TMA write could race the next chunk's V fill.
        mbar_wait(&mbar[kStages + (c % S::kVSlots)], (c / S::kVSlots) & 1);
      }
    };

    // ---- consume one union block as a 2-chunk pair (PACK pair loop, JV) -------
    // A union block IS 2 chunks (64-token halves of one 128-token block), so the
    // block becomes the loop unit: QK over both halves, then ONE mask/scale +
    // max/rescale/exp2/sum softmax chain over 128 tokens, then PV of both
    // halves. Halves the per-block __syncthreads / softmax-chain / rescale cost
    // of the 64-token form (round-14 profile of the g4 paged PACK path: wait
    // 28% + barrier 19%, tensor pipe 36.6% — a latency-chained loop, not math
    // or DRAM bound). c is always even (block-aligned); the joint 2-slot ring
    // waits are staggered so QK of chunk c covers the refill flight of chunk
    // c+1 (removed the refill-stall the double-wait form paid at pair top).
    auto qk_pair = [&](int c, float(*s)[4]) {
      const char* tk0 = sm_ring + (c % kStages) * (64 * 128 * 2);
      const char* tk1 = sm_ring + ((c + 1) % kStages) * (64 * 128 * 2);
      auto ldsm_step = [&](const char* tk, int kk, uint32_t(&b)[4]) {
        int row = warp * kWarpTok + (lane & 15);
        int seg = (kk & 3) * 2 + (lane >> 4);
        ldsm_x4(b, smem_u32(tk + (kk >> 2) * (64 * 128) + tile_off(row, seg)));
      };
      // Chunk c's k-steps, then the staggered wait, then chunk c+1 (the refill
      // flight hides under chunk c's QK). Interleaving both chunks' steps for
      // extra accumulator-bank ILP measured -1% (395.4 vs 391.3 local fixture9):
      // the staggered split already leaves the MMA pipe fed.
      uint32_t bn0[4], bn1[4];
      ldsm_step(tk0, 0, bn0);
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t bc0[4] = {bn0[0], bn0[1], bn0[2], bn0[3]};
        if (kk < 7) ldsm_step(tk0, kk + 1, bn0);
        uint32_t b0[2] = {bc0[0], bc0[2]};
        uint32_t b1[2] = {bc0[1], bc0[3]};
        Mma<QT>::run(s[0], qa[kk], b0);
        Mma<QT>::run(s[1], qa[kk], b1);
      }
      mbar_wait(&mbar[(c + 1) % kStages], (((c + 1) >> 1) + wphase[1]) & 1);
      ldsm_step(tk1, 0, bn1);
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint32_t bc1[4] = {bn1[0], bn1[1], bn1[2], bn1[3]};
        if (kk < 7) ldsm_step(tk1, kk + 1, bn1);
        uint32_t b0[2] = {bc1[0], bc1[2]};
        uint32_t b1[2] = {bc1[1], bc1[3]};
        Mma<QT>::run(s[2], qa[kk], b0);
        Mma<QT>::run(s[3], qa[kk], b1);
      }
    };

    auto consume_pair = [&](int c, int bidx) {
      const char* tv0 = sm_ring + (kStages + (c % S::kVSlots)) * (64 * 128 * 2);
      const char* tv1 = sm_ring + (kStages + ((c + 1) % S::kVSlots)) * (64 * 128 * 2);
      const int uw = sm_utok[bidx];
      const int tm = uw & 0xF;
      const int blk = uw >> 4;
      const int tok0 = blk * kBlockTok + warp * kWarpTok;
      const int tok1 = tok0 + kChunkTok;

      mbar_wait(&mbar[c % kStages], ((c >> 1) + wphase[0]) & 1);

      // Warp-uniform early skip: tm bits are per-BLOCK (identical for both
      // halves) and every warp covers all 16 packed rows, so this only fires
      // on degenerate tail packs. BOTH barriers must still be waited before
      // returning — the post-syncthreads expect_tx is only phase-safe once
      // every warp observed this pair's phases complete.
      if (!__any_sync(0xffffffffu, (tm & (bit0 | bit1)) != 0)) {
        mbar_wait(&mbar[(c + 1) % kStages], (((c + 1) >> 1) + wphase[1]) & 1);
        return;
      }

      float s[4][4];  // [chunk*2 + n8 tile][frag cols]
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) s[j][q4] = 0.f;

      qk_pair(c, s);

      // selection/causal mask for all 4 tiles in RAW score space (halves share
      // tm; the second half is 64 tokens further along the same block). Scale
      // enters once per row via ns below (see the single-block site).
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int tok = (j < 2 ? tok0 : tok1) + (j & 1) * 8 + kcol0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          bool ok0 = (tm & bit0) && (!p.causal || (tok + i) <= qp0);
          bool ok1 = (tm & bit1) && (!p.causal || (tok + i) <= qp1);
          s[j][i] = ok0 ? s[j][i] : -1e38f;
          s[j][2 + i] = ok1 ? s[j][2 + i] : -1e38f;
        }
      }

      float rmax0 = fmaxf(fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1])),
                          fmaxf(fmaxf(s[2][0], s[2][1]), fmaxf(s[3][0], s[3][1])));
      float rmax1 = fmaxf(fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3])),
                          fmaxf(fmaxf(s[2][2], s[2][3]), fmaxf(s[3][2], s[3][3])));
      {
        unsigned long long mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
                                (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x1 = __shfl_xor_sync(0xffffffffu, mp, 1);
        rmax0 = fmaxf(__uint_as_float((unsigned)(mp >> 32)), __uint_as_float((unsigned)(x1 >> 32)));
        rmax1 = fmaxf(__uint_as_float((unsigned)mp), __uint_as_float((unsigned)x1));
        mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
             (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x2 = __shfl_xor_sync(0xffffffffu, mp, 2);
        rmax0 = fmaxf(rmax0, __uint_as_float((unsigned)(x2 >> 32)));
        rmax1 = fmaxf(rmax1, __uint_as_float((unsigned)x2));
      }

      const bool anylive = __any_sync(0xffffffffu, (rmax0 > -1e37f) || (rmax1 > -1e37f));
      if (anylive) {
        float mn0 = fmaxf(m0, rmax0);
        float mn1 = fmaxf(m1, rmax1);
        const float ns0 = -mn0 * p.scale_log2e;
        const float ns1 = -mn1 * p.scale_log2e;
        if (mn0 > m0 || mn1 > m1) {
          float f0 = exp2f(fmaf(m0, p.scale_log2e, ns0));
          float f1 = exp2f(fmaf(m1, p.scale_log2e, ns1));
          l0 *= f0;
          l1 *= f1;
#pragma unroll
          for (int dt = 0; dt < 16; ++dt) {
            acc[dt][0] *= f0;
            acc[dt][1] *= f0;
            acc[dt][2] *= f1;
            acc[dt][3] *= f1;
          }
        }
        m0 = mn0;
        m1 = mn1;
        float pe[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          pe[j][0] = (s[j][0] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][0], p.scale_log2e, ns0));
          pe[j][1] = (s[j][1] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][1], p.scale_log2e, ns0));
          pe[j][2] = (s[j][2] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][2], p.scale_log2e, ns1));
          pe[j][3] = (s[j][3] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][3], p.scale_log2e, ns1));
        }
        float rs0 =
            pe[0][0] + pe[0][1] + pe[1][0] + pe[1][1] + pe[2][0] + pe[2][1] + pe[3][0] + pe[3][1];
        float rs1 =
            pe[0][2] + pe[0][3] + pe[1][2] + pe[1][3] + pe[2][2] + pe[2][3] + pe[3][2] + pe[3][3];
        {
          unsigned long long rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
                                  (unsigned long long)__float_as_uint(rs1);
          unsigned long long y1 = __shfl_xor_sync(0xffffffffu, rp, 1);
          rs0 += __uint_as_float((unsigned)(y1 >> 32));
          rs1 += __uint_as_float((unsigned)y1);
          rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
               (unsigned long long)__float_as_uint(rs1);
          unsigned long long y2 = __shfl_xor_sync(0xffffffffu, rp, 2);
          rs0 += __uint_as_float((unsigned)(y2 >> 32));
          rs1 += __uint_as_float((unsigned)y2);
        }
        l0 += rs0;
        l1 += rs1;

        auto pv_half = [&](const char* tv, int half) {
          const float(*peh)[4] = &pe[half * 2];
          uint32_t a[4];
          a[0] = Mma<QT>::pack(peh[0][0], peh[0][1]);
          a[1] = Mma<QT>::pack(peh[0][2], peh[0][3]);
          a[2] = Mma<QT>::pack(peh[1][0], peh[1][1]);
          a[3] = Mma<QT>::pack(peh[1][2], peh[1][3]);
          uint32_t bvx[4];
          {
            int row = warp * kWarpTok + (lane & 15);
            ldsm_x4_trans(bvx, smem_u32(tv + tile_off(row, lane >> 4)));
          }
#pragma unroll
          for (int dt = 0; dt < 16; dt += 2) {
            uint32_t b4[4] = {bvx[0], bvx[1], bvx[2], bvx[3]};
            if (dt < 14) {  // prefetch the next PV step under this step's HMMA
              int row = warp * kWarpTok + (lane & 15);
              int seg = ((dt + 2) & 7) + (lane >> 4);
              ldsm_x4_trans(bvx, smem_u32(tv + ((dt + 2) >> 3) * (64 * 128) + tile_off(row, seg)));
            }
            uint32_t b0[2] = {b4[0], b4[1]};
            uint32_t b1[2] = {b4[2], b4[3]};
            Mma<QT>::run(acc[dt], a, b0);
            Mma<QT>::run(acc[dt + 1], a, b1);
          }
        };

        // Joint ring: K+V were both waited at/inside pair top; PV reads V of
        // each half directly from its joint slot.
        pv_half(tv0, 0);
        pv_half(tv1, 1);
      }
    };

    // ---- consume one union block as a 2-chunk pair (3-buffer form, PACK !JV) ---
    // 48KB ring + 6KB prologue = 54KB/CTA -> 4 CTAs/SM (vs 3 at 70KB joint).
    // K(c) in S0 and K(c+1) in S1 are pre-armed one pair ahead; V(c) pre-armed
    // one pair ahead into the single V slot; V(c+1) is issued INTO S1 right
    // after both QKs drained K from it — its TMA flight hides under the softmax
    // chain + PV(c). Q fragments restage from smem per half-pair (two 4-step
    // halves) — under the 128-reg cap persistent-Q spilled 32B and measured
    // 435.9 vs 394.3 on fixture9, so restage stays. Barrier discipline:
    // every phase is always waited (mask-skipped pairs too), so tid0's next
    // expect_tx after the loop-end barrier never folds into an open phase.
    auto consume_pair_3b = [&](int c, int bidx) {
      const char* s0 = sm_ring;
      const char* slv = sm_ring + 2 * S::kTensorBytes;
      const int uw = sm_utok[bidx];
      const int tm = uw & 0xF;
      const int blk = uw >> 4;
      const int tok0 = blk * kBlockTok + warp * kWarpTok;
      const int tok1 = tok0 + kChunkTok;
      const int pp = c >> 1;

      mbar_wait(&mbar[0], 0);  // K(c) in S0 (two uses/pair: K at use 2p -> parity 0)

      // Warp-uniform mask-skip: waits/syncs still execute on every warp.
      const bool live = __any_sync(0xffffffffu, (tm & (bit0 | bit1)) != 0);

      float s[4][4];  // [chunk*2 + n8 tile][frag cols]
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) s[j][q4] = 0.f;

      // QK of one 64-token half into s[base..base+1]; Q fragments restaged from
      // smem in two halves of 4 k16 steps (~16 live regs vs 32 persistent; the
      // persistent-Q variant spilled and measured 435.9 vs 394.3 — restage is
      // cheaper than the 32B spill the 128-reg cap forces).
      auto qk_half = [&](const char* tk, int base) {
        auto ldsm_step = [&](int kk, uint32_t(&b)[4]) {
          int row = warp * kWarpTok + (lane & 15);
          int seg = (kk & 3) * 2 + (lane >> 4);
          ldsm_x4(b, smem_u32(tk + (kk >> 2) * (64 * 128) + tile_off(row, seg)));
        };
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          uint32_t qx[4][4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int row = lane & 15;
            ldsm_x4(qx[j],
                    smem_u32(sm_q + kh * (kRows * 128) + tile_off(row, j * 2 + (lane >> 4))));
          }
          uint32_t bn[4];
          ldsm_step(kh * 4, bn);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int kk = kh * 4 + j;
            uint32_t bc[4] = {bn[0], bn[1], bn[2], bn[3]};
            if (j < 3) ldsm_step(kk + 1, bn);
            uint32_t b0[2] = {bc[0], bc[2]};  // n8 tile 0: klo, khi
            uint32_t b1[2] = {bc[1], bc[3]};  // n8 tile 1: klo, khi
            Mma<QT>::run(s[base + 0], qx[j], b0);
            Mma<QT>::run(s[base + 1], qx[j], b1);
          }
        }
      };

      if (live) qk_half(s0, 0);
      __syncthreads();                    // all warps done with S0's K; free for V(c+1)
      if (tid == 0) issue3b_v(c + 1, 0);  // V(c+1) -> S0 (phase 2pp+1, parity 1)
      const char* s1 = sm_ring + S::kTensorBytes;
      mbar_wait(&mbar[1], pp & 1);  // K(c+1) in S1 (one use/pair, parity pp&1)
      if (live) qk_half(s1, 2);

      // selection/causal mask, RAW space (identical to the JV pair form)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int tok = (j < 2 ? tok0 : tok1) + (j & 1) * 8 + kcol0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          bool ok0 = (tm & bit0) && (!p.causal || (tok + i) <= qp0);
          bool ok1 = (tm & bit1) && (!p.causal || (tok + i) <= qp1);
          s[j][i] = ok0 ? s[j][i] : -1e38f;
          s[j][2 + i] = ok1 ? s[j][2 + i] : -1e38f;
        }
      }

      float rmax0 = fmaxf(fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1])),
                          fmaxf(fmaxf(s[2][0], s[2][1]), fmaxf(s[3][0], s[3][1])));
      float rmax1 = fmaxf(fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3])),
                          fmaxf(fmaxf(s[2][2], s[2][3]), fmaxf(s[3][2], s[3][3])));
      {
        unsigned long long mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
                                (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x1 = __shfl_xor_sync(0xffffffffu, mp, 1);
        rmax0 = fmaxf(__uint_as_float((unsigned)(mp >> 32)), __uint_as_float((unsigned)(x1 >> 32)));
        rmax1 = fmaxf(__uint_as_float((unsigned)mp), __uint_as_float((unsigned)x1));
        mp = ((unsigned long long)__float_as_uint(rmax0) << 32) |
             (unsigned long long)__float_as_uint(rmax1);
        unsigned long long x2 = __shfl_xor_sync(0xffffffffu, mp, 2);
        rmax0 = fmaxf(rmax0, __uint_as_float((unsigned)(x2 >> 32)));
        rmax1 = fmaxf(rmax1, __uint_as_float((unsigned)x2));
      }

      const bool anylive = __any_sync(0xffffffffu, (rmax0 > -1e37f) || (rmax1 > -1e37f));
      mbar_wait(&mbar[2], pp & 1);  // V(c) in SLV (phase always consumed)
      if (anylive) {
        float mn0 = fmaxf(m0, rmax0);
        float mn1 = fmaxf(m1, rmax1);
        const float ns0 = -mn0 * p.scale_log2e;
        const float ns1 = -mn1 * p.scale_log2e;
        if (mn0 > m0 || mn1 > m1) {
          float f0 = exp2f(fmaf(m0, p.scale_log2e, ns0));
          float f1 = exp2f(fmaf(m1, p.scale_log2e, ns1));
          l0 *= f0;
          l1 *= f1;
#pragma unroll
          for (int dt = 0; dt < 16; ++dt) {
            acc[dt][0] *= f0;
            acc[dt][1] *= f0;
            acc[dt][2] *= f1;
            acc[dt][3] *= f1;
          }
        }
        m0 = mn0;
        m1 = mn1;
        float pe[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          pe[j][0] = (s[j][0] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][0], p.scale_log2e, ns0));
          pe[j][1] = (s[j][1] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][1], p.scale_log2e, ns0));
          pe[j][2] = (s[j][2] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][2], p.scale_log2e, ns1));
          pe[j][3] = (s[j][3] <= -1e37f) ? 0.f : exp2f(fmaf(s[j][3], p.scale_log2e, ns1));
        }
        float rs0 =
            pe[0][0] + pe[0][1] + pe[1][0] + pe[1][1] + pe[2][0] + pe[2][1] + pe[3][0] + pe[3][1];
        float rs1 =
            pe[0][2] + pe[0][3] + pe[1][2] + pe[1][3] + pe[2][2] + pe[2][3] + pe[3][2] + pe[3][3];
        {
          unsigned long long rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
                                  (unsigned long long)__float_as_uint(rs1);
          unsigned long long y1 = __shfl_xor_sync(0xffffffffu, rp, 1);
          rs0 += __uint_as_float((unsigned)(y1 >> 32));
          rs1 += __uint_as_float((unsigned)y1);
          rp = ((unsigned long long)__float_as_uint(rs0) << 32) |
               (unsigned long long)__float_as_uint(rs1);
          unsigned long long y2 = __shfl_xor_sync(0xffffffffu, rp, 2);
          rs0 += __uint_as_float((unsigned)(y2 >> 32));
          rs1 += __uint_as_float((unsigned)y2);
        }
        l0 += rs0;
        l1 += rs1;

        auto pv_half = [&](const char* tv, int half) {
          const float(*peh)[4] = &pe[half * 2];
          uint32_t a[4];
          a[0] = Mma<QT>::pack(peh[0][0], peh[0][1]);
          a[1] = Mma<QT>::pack(peh[0][2], peh[0][3]);
          a[2] = Mma<QT>::pack(peh[1][0], peh[1][1]);
          a[3] = Mma<QT>::pack(peh[1][2], peh[1][3]);
          uint32_t bvx[4];
          {
            int row = warp * kWarpTok + (lane & 15);
            ldsm_x4_trans(bvx, smem_u32(tv + tile_off(row, lane >> 4)));
          }
#pragma unroll
          for (int dt = 0; dt < 16; dt += 2) {
            uint32_t b4[4] = {bvx[0], bvx[1], bvx[2], bvx[3]};
            if (dt < 14) {  // prefetch the next PV step under this step's HMMA
              int row = warp * kWarpTok + (lane & 15);
              int seg = ((dt + 2) & 7) + (lane >> 4);
              ldsm_x4_trans(bvx, smem_u32(tv + ((dt + 2) >> 3) * (64 * 128) + tile_off(row, seg)));
            }
            uint32_t b0[2] = {b4[0], b4[1]};
            uint32_t b1[2] = {b4[2], b4[3]};
            Mma<QT>::run(acc[dt], a, b0);
            Mma<QT>::run(acc[dt + 1], a, b1);
          }
        };

        pv_half(slv, 0);
        mbar_wait(&mbar[0], 1);  // V(c+1) in S0 (V phase, parity 1)
        pv_half(s0, 1);
      } else {
        mbar_wait(&mbar[0], 1);  // drain the V(c+1) phase on skipped pairs too
      }
    };

    // ---- main loop -------------------------------------------------------------
    if constexpr (PAIRP) {
      // Pair form: one 128-token union block per iteration (joint 2-slot ring,
      // 3 CTAs/SM). Both chunks of the NEXT pair (K+V jointly) are issued only
      // after the loop-end __syncthreads frees their slots — the ring holds
      // exactly one pair, so the refill flight is covered by the staggered
      // waits inside consume_pair (QK of the resident chunk) and by co-resident
      // CTAs.
      for (int c = 0; c < nloc; c += 2) {
        if constexpr (PERSIST && MSA_PERSIST_FULLGRID != 2) {
          // Final pair: claim this CTA's NEXT work item now, so the queue
          // atomic's L2 round-trip hides under this pair's consume + the
          // merge/epilogue that follow (claiming at the loop-top barrier
          // measured +6% end-to-end — tid0's atomic stall held the whole CTA).
          if (tid == 0 && c + 2 >= nloc) *sm_claim = (int)gridDim.x + atomicAdd(p.ws_next, 1);
        }
        const int bidx = CLUSTER ? (crank + (c & ~1)) : (c >> 1);
        if constexpr (JV) {
          consume_pair(c, bidx);
        } else {
          consume_pair_3b(c, bidx);
        }
        __syncthreads();  // this pair's slots dead for reuse
        if (tid == 0) {
          if constexpr (JV) {
            if (c + 2 < nloc) issue_chunk(c + 2, CLUSTER ? (bidx + 2) : -1);
            if (c + 3 < nloc) issue_chunk(c + 3, CLUSTER ? (bidx + 2) : -1);
          } else {
            // 3-buffer: next pair's K halves + the V slot's next fill.
            if (c + 2 < nloc) issue3b_k(c + 2, 0);
            if (c + 3 < nloc) {
              issue3b_k(c + 3, 1);
              issue3b_v(c + 2, 2);
            }
          }
        }
      }
    } else {
      for (int c = 0; c < nchunks; ++c) {
        mbar_wait(&mbar[c % kStages], (c / kStages) & 1);
        // Issue the next TMA chunk as soon as its ring slot is provably free
        // (slot (c + kStages-1) % kStages last held chunk c-1, consumed and
        // barrier-synced in the previous iteration). With issue-at-loop-end
        // (previous form) a 2-stage ring kept ZERO chunks inflight while the
        // warps computed — every chunk paid the full serial TMA latency. Here
        // the fetch of chunk c+kStages-1 overlaps compute of chunk c.
        if (tid == 0 && c + kStages - 1 < nchunks) issue_chunk(c + kStages - 1, -1);
        if (KVK == 2) {
          convert_chunk(c);
        } else if (tid == 0 && !JV) {
          // single V slot: can only fire after the loop-end __syncthreads below,
          // so issue V for THIS chunk; the QK+softmax window covers its write.
          // (Joint-barrier rings load V inside issue_chunk.)
          issue_v(c);
        }
        consume_chunk(c);
        __syncthreads();
      }
    }

    if constexpr (PERSIST && MSA_PERSIST_FULLGRID != 2) {
      // Empty-union tiles never enter the pair loop, so their next-item claim
      // has no pair window to hide in — issue it here (rare path: empty tiles
      // have almost no other work to overlap anyway).
      if (tid == 0 && nloc == 0) *sm_claim = (int)gridDim.x + atomicAdd(p.ws_next, 1);
    }

    // ---- merge the 4 per-warp softmax states -----------------------------------
    // Stage each warp's scaled accumulator into its own smem strip (overlaid on
    // the now-idle TMA ring), then reduce strips with plain loads (no atomics).
    if ((lane & 3) == 0) {
      int r = lane >> 2;
      // the mainloop's online state now keeps raw (unscaled) maxima; convert to
      // scaled space here so the merge below (e0/e1, cluster fa/fb) is unchanged
      sm_red[warp * 16 + r] = m0 * p.scale_log2e;
      sm_red[warp * 16 + r + 8] = m1 * p.scale_log2e;
      sm_red[64 + warp * 16 + r] = l0;
      sm_red[64 + warp * 16 + r + 8] = l1;
    }
    __syncthreads();
    const int r0 = lane >> 2;
    float M0 = fmaxf(fmaxf(sm_red[r0], sm_red[16 + r0]), fmaxf(sm_red[32 + r0], sm_red[48 + r0]));
    float M1 = fmaxf(fmaxf(sm_red[r0 + 8], sm_red[16 + r0 + 8]),
                     fmaxf(sm_red[32 + r0 + 8], sm_red[48 + r0 + 8]));
    float e0 = exp2f(m0 * p.scale_log2e - M0);  // m raw -> scaled; M0 already scaled
    float e1 = exp2f(m1 * p.scale_log2e - M1);
    if ((lane & 3) == 0) {
      atomicAdd(&sm_red[128 + r0], l0 * e0);
      atomicAdd(&sm_red[128 + r0 + 8], l1 * e1);
    }
    float* strip = sm_acc + warp * (kRows * S::kStripPitch);
#pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
      int col = dt * 8 + kcol0;
      float2 x0 = make_float2(acc[dt][0] * e0, acc[dt][1] * e0);
      float2 x1 = make_float2(acc[dt][2] * e1, acc[dt][3] * e1);
      *reinterpret_cast<float2*>(&strip[r0 * S::kStripPitch + col]) = x0;
      *reinterpret_cast<float2*>(&strip[(r0 + 8) * S::kStripPitch + col]) = x1;
    }
    __syncthreads();

    // ---- reduce strips + epilogue write ------------------------------------------
    // warp w owns output cols [w*32, w*32+32) for all 16 rows
    if constexpr (CLUSTER) {
      // Cluster pair merge. Each CTA first reduces its 4 warp strips into a
      // compact 16x128 FP32 partial (still scaled to this CTA's per-row max,
      // like the l sums in sm_red[128+g]) and publishes M_cta per row. After a
      // cluster barrier, rank 0 pulls the peer's partial + (m, l) over DSM,
      // merges the two online-softmax states exactly (fa/fb rescale), and does
      // the single final store; rank 1 waits on a second barrier so its shared
      // memory stays live until every remote read has completed.
      float* sm_part = sm_acc + 4 * (kRows * S::kStripPitch);  // 16x128 fp32, past strips
      if (tid < 16) {
        sm_red[144 + tid] =
            fmaxf(fmaxf(sm_red[tid], sm_red[16 + tid]), fmaxf(sm_red[32 + tid], sm_red[48 + tid]));
      }
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        int i = lane + it * 32;  // 0..63
        int g = i >> 2;          // row 0..15
        int c8 = i & 3;          // 8-col group inside warp slice
        int col = warp * 32 + c8 * 8;
        float vals[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) vals[j] = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float* sp = sm_acc + s * (kRows * S::kStripPitch) + g * S::kStripPitch + col;
          float4 x0 = *reinterpret_cast<const float4*>(sp);
          float4 x1 = *reinterpret_cast<const float4*>(sp + 4);
          vals[0] += x0.x;
          vals[1] += x0.y;
          vals[2] += x0.z;
          vals[3] += x0.w;
          vals[4] += x1.x;
          vals[5] += x1.y;
          vals[6] += x1.z;
          vals[7] += x1.w;
        }
        float* pp = sm_part + g * 128 + col;
        *reinterpret_cast<float4*>(pp) = make_float4(vals[0], vals[1], vals[2], vals[3]);
        *reinterpret_cast<float4*>(pp + 4) = make_float4(vals[4], vals[5], vals[6], vals[7]);
      }
      __syncthreads();  // partial + M_cta visible CTA-wide before cluster sync
      cluster_sync();
      if (crank == 0) {
        const uint32_t r_part = mapa_cta(smem_u32(sm_part), 1);
        const uint32_t r_red = mapa_cta(smem_u32(sm_red), 1);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
          int i = lane + it * 32;
          int g = i >> 2;
          int c8 = i & 3;
          int col = warp * 32 + c8 * 8;
          // Row g's output token/head come from the packed row map; rows of
          // tokens past total_q carry bit 0 and are skipped.
          const int t = g / p.group;
          if (sm_rtbit[g] != 0) {
            const float* pp = sm_part + g * 128 + col;
            const float4 a0 = *reinterpret_cast<const float4*>(pp);
            const float4 a1 = *reinterpret_cast<const float4*>(pp + 4);
            const uint32_t roff = (uint32_t)((g * 128 + col) * 4);
            float b0[4], b1[4];
            ldsm_remote_v4f32(b0, r_part + roff);
            ldsm_remote_v4f32(b1, r_part + roff + 16);
            const float mA = sm_red[144 + g];
            const float lA = sm_red[128 + g];
            const float mB = ldsm_remote_f32(r_red + (144 + g) * 4);
            const float lB = ldsm_remote_f32(r_red + (128 + g) * 4);
            const float M = fmaxf(mA, mB);
            const float fa = exp2f(mA - M);
            const float fb = exp2f(mB - M);
            const float vals[8] = {a0.x * fa + b0[0] * fb, a0.y * fa + b0[1] * fb,
                                   a0.z * fa + b0[2] * fb, a0.w * fa + b0[3] * fb,
                                   a1.x * fa + b1[0] * fb, a1.y * fa + b1[1] * fb,
                                   a1.z * fa + b1[2] * fb, a1.w * fa + b1[3] * fb};
            const float l = lA * fa + lB * fb;
            const float inv = l > 0.f ? 1.f / l : 0.f;
            uint32_t pack4[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
              pack4[j] = Mma<QT>::pack(vals[j * 2] * inv, vals[j * 2 + 1] * inv);
            QT* dst = reinterpret_cast<QT*>(p.out) + (long)(ntok0 + t) * p.o_tok +
                      (long)(h * p.group + (g - t * p.group)) * p.o_head + col;
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4*>(pack4);
          }
        }
      }
      cluster_sync();  // rank 1's smem must stay live until rank 0 finished
      return;
    }
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      int i = lane + it * 32;  // 0..63
      int g = i >> 2;          // row 0..15
      int c8 = i & 3;          // 8-col group inside warp slice
      int col = warp * 32 + c8 * 8;
      float vals[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) vals[j] = 0.f;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float* sp = sm_acc + s * (kRows * S::kStripPitch) + g * S::kStripPitch + col;
        float4 x0 = *reinterpret_cast<const float4*>(sp);
        float4 x1 = *reinterpret_cast<const float4*>(sp + 4);
        vals[0] += x0.x;
        vals[1] += x0.y;
        vals[2] += x0.z;
        vals[3] += x0.w;
        vals[4] += x1.x;
        vals[5] += x1.y;
        vals[6] += x1.z;
        vals[7] += x1.w;
      }
      // PACK: row g's output token/head come from the packed row map; rows of
      // tokens past total_q carry bit 0 and are skipped.
      long otok = n;
      int oh = h * p.group + g;
      bool rvalid = g < p.group;
      if constexpr (PACK) {
        int t = g / p.group;
        otok = PACK ? (long)(ntok0 + t) : otok;
        oh = h * p.group + (g - t * p.group);
        rvalid = sm_rtbit[g] != 0;
      }
      if (rvalid) {
        float lr = sm_red[128 + g];
        float inv = lr > 0.f ? 1.f / lr : 0.f;
        uint32_t pack4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          pack4[j] = Mma<QT>::pack(vals[j * 2] * inv, vals[j * 2 + 1] * inv);
        QT* dst = reinterpret_cast<QT*>(p.out) + otok * p.o_tok + (long)oh * p.o_head + col;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4*>(pack4);
      }
    }
    if constexpr (PERSIST && MSA_PERSIST_FULLGRID == 2) {
      // A/B: static-equivalent traversal — no queue op anywhere; every CTA
      // runs its pre-assigned tile and exits.
      return;
    } else if constexpr (PERSIST) {
      // Pair mode: every union-block pair iteration consumed exactly one use
      // of both ring slots; account this tile's nloc/2 uses so the next
      // claimed tile's wait parities match the carried-over barrier phases.
      wphase[0] += nloc >> 1;
      wphase[1] += nloc >> 1;
    } else {
      break;
    }
  }  // PERSIST claim loop (single pass when !PERSIST)
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// kind: 0 = bf16, 1 = fp16, 2 = fp8-e4m3
CUtensorMapDataType tma_dtype(int kind) {
  switch (kind) {
    case 0:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case 1:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case 2:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default:
      MSAV_CHECK(false, "unsupported KV dtype kind %d", kind);
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
}

void encode_tmap(CUtensorMap* map, const void* base, const KvLayout& kv, int kind, bool paged,
                 int box_tok) {
  // Build the (rank, dims, strides, box) description shared by K and V.
  const bool is_fp8 = (kind == 2);
  const int es = is_fp8 ? 1 : 2;
  cuuint64_t dims[5];
  cuuint64_t strides[4];  // bytes, for dims 1..rank-1
  cuuint32_t box[5];
  int rank;
  if (!is_fp8) {
    if (!paged) {
      // [total_tokens, kvh, 128] -> dims {64 elem half, kvh, tokens}
      rank = 3;
      dims[0] = kHead;  // full head dim; box[0] selects the 64-elem half
      dims[1] = (cuuint64_t)kv.d1;
      dims[2] = (cuuint64_t)kv.d0;
      strides[0] = (cuuint64_t)kv.s1 * es;
      strides[1] = (cuuint64_t)kv.s0 * es;
      box[0] = 64;
      box[1] = 1;
      box[2] = box_tok;
    } else {
      // [pages, kvh, 128, 128] -> dims {64 elem half, 128 tok, kvh, pages}
      rank = 4;
      dims[0] = kHead;
      dims[1] = kBlockTok;
      dims[2] = (cuuint64_t)kv.d1;
      dims[3] = (cuuint64_t)kv.d0;
      strides[0] = (cuuint64_t)kv.s2 * es;
      strides[1] = (cuuint64_t)kv.s1 * es;
      strides[2] = (cuuint64_t)kv.s0 * es;
      box[0] = 64;
      box[1] = box_tok;
      box[2] = 1;
      box[3] = 1;
    }
  } else {
    if (!paged) {
      // dims {128, kvh, tokens}
      rank = 3;
      dims[0] = kHead;
      dims[1] = (cuuint64_t)kv.d1;
      dims[2] = (cuuint64_t)kv.d0;
      strides[0] = (cuuint64_t)kv.s1 * es;
      strides[1] = (cuuint64_t)kv.s0 * es;
      box[0] = kHead;
      box[1] = 1;
      box[2] = kChunkTok;
    } else {
      // dims {128, 128 tok, kvh, pages}
      rank = 4;
      dims[0] = kHead;
      dims[1] = kBlockTok;
      dims[2] = (cuuint64_t)kv.d1;
      dims[3] = (cuuint64_t)kv.d0;
      strides[0] = (cuuint64_t)kv.s2 * es;
      strides[1] = (cuuint64_t)kv.s1 * es;
      strides[2] = (cuuint64_t)kv.s0 * es;
      box[0] = kHead;
      box[1] = kChunkTok;
      box[2] = 1;
      box[3] = 1;
    }
  }
  cuuint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  CUresult r =
      cuTensorMapEncodeTiled(map, tma_dtype(kind), rank, const_cast<void*>(base), dims, strides,
                             box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             is_fp8 ? CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  MSAV_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: %d", (int)r);
}

template <typename QT, int KVK, bool PAGED, int NSTG, int VSL, bool PACK, bool JV = (VSL != 1),
          bool PAIRP = false>
void launch_msa(const Params& p, const CUtensorMap& kmap, const CUtensorMap& vmap, int grid_x,
                int kvh, cudaStream_t stream) {
  constexpr size_t kSmem = Smem<KVK, NSTG, VSL>::kTotal;
  auto kern = msa_sparse_kernel<QT, KVK, PAGED, NSTG, VSL, PACK, JV, PAIRP>;
  static bool inited = [&] {
    cudaError_t e =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem);
    MSAV_CHECK(e == cudaSuccess, "cudaFuncSetAttribute failed: %s smem=%d", cudaGetErrorString(e),
               (int)kSmem);
    return true;
  }();
  (void)inited;
  dim3 grid(grid_x, kvh);
  kern<<<grid, kThreads, kSmem, stream>>>(p, kmap, vmap);
  cudaError_t le = cudaGetLastError();
  MSAV_CHECK(le == cudaSuccess, "kernel launch failed: %s smem=%d grid=%dx%d",
             cudaGetErrorString(le), (int)kSmem, grid_x, kvh);
}

// PACK pair launcher (the default packed dispatch): one CTA per
// (union tile, KV head). Round-17 (2-CTA DSM cluster) and round-19
// (persistent device work queue) launch structures measured out —
// +36-51% and +5-7% respectively on the grid-saturated fixture9 —
// and were pruned in round 22 (evidence in ncu_evidence.md).
template <typename QT, int KVK, bool PAGED>
void launch_pack_pair(const Params& pin, const CUtensorMap& kmap, const CUtensorMap& vmap,
                      int ntiles, int kvh, cudaStream_t stream) {
  launch_msa<QT, KVK, PAGED, 2, 2, true, true, true>(pin, kmap, vmap, ntiles, kvh, stream);
}

template <typename QT, int KVK, bool PAGED>
void dispatch_stages(Params p, const CUtensorMap& kmap, const CUtensorMap& vmap, int total_q,
                     int kvh, cudaStream_t stream) {
  // Structural variant dispatch (keyed only on runtime tensor structure):
  //  * seqlen_q >= kRows (16): long-query/prefill variant with a shallow
  //    2-stage ring so three CTAs fit per SM for cross-CTA latency hiding
  //    (raising the boundary to 64 so q16 tiles took the deeper NSTG=3 ring
  //    measured 796 vs 779 us on the q16 flat decode case — cross-CTA
  //    parallelism wins there, so the boundary stays at kRows). Wide-topk
  //    prefills additionally drop to a single V slot for the 4th CTA/SM
  //    (topk bucket below).
  //  * shorter query tiles (decode): deeper pipeline (3 stages, 4 for fp8 KV)
  //    so a single CTA keeps more loads in flight, with V at joint depth.
  // Earlier union/dense tile-kernel experiments (deleted round 9) stayed
  // unreferenced: measured on this suite's i.i.d. per-token block selections,
  // chunk unions saturate the KV block space, so the extra masked MMA work
  // outweighs the fetch savings (0.27x-0.87x across prefill and decode
  // shapes; A/B digests in ncu_evidence.md).
  // Prefill V policy keyed on the topk bucket (structural: q2k width):
  // wide selections stream enough chunks to cover a zero-depth V fetch, so
  // a single V slot trades for +1 CTA/SM; narrow selections (few chunks per
  // token) need V prefetched at joint K depth or every chunk's PV stalls on
  // a late TMA (fixture evidence in Smem comment). fp8 ignores VSL (joint
  // K+V stages feed the smem convert tile).
  // Row-packed prefill (group < kRows): pack T = kRows/group tokens x group
  // heads into the 16-row tile and stream the union block list. Structural
  // guards: T in {2,4} (token-mask width), T*topk <= kMaxUnion (smem list
  // budget), bf16/fp16 KV only — anything else keeps the general path below.
  if constexpr (KVK != 2) {
    // Round-22 tiny-decode single-CTA fusion (PACK pair machinery at
    // grid = 1 CTA/KV-head for total_q*group <= kRows) MEASURED OUT:
    // fixture12 (b2 q1 kv257 paged g8 topk4) 11.2 us fused vs 7.1 us
    // per-token (+58% loss) — disjoint-batch queries have an empty union
    // intersection, so fusion serializes two streams and pays the PACK
    // union-build + staggered-ring prologue for zero fetch dedup
    // (ncu_evidence.md r22). Tiny decode stays on the general path below.
    if (p.seqlen_q >= kRows && p.group < kRows && (kRows % p.group) == 0) {
      const int T = kRows / p.group;
      // Round-20 supervisor A/B (non-union per-token dispatch on this class):
      // MEASURED OUT — the non-PACK template costs 382.3 -> 554.2 us on
      // fixture9 (+45%, corroborates the round-9 515-606 us A/B); the union
      // traversal's fetch/compute overhead (~14/16 blocks, ~12%) is far
      // smaller than paying the per-CTA fixed costs 4x more often
      // (24576 vs 6144 CTAs). PACK stays the default for this class.
      if (T <= 4 && T * p.topk <= Smem<KVK, 2, 2>::kMaxUnion) {
        p.pack_T = T;
        // Pair form (round-14): one 128-token union block per iteration,
        // single-CTA staggered-joint ring. Alternative pair structures all
        // measured out (rounds 17/21, ncu_evidence.md): 2-CTA DSM cluster
        // +36-51% loss (no idle SMs to absorb the duplicated per-tile fixed
        // cost), 3-buffer ring (54KB/CTA, 4 CTAs/SM) +3.1-14% loss in three
        // variants (4-CTA residency confirmed by NCU but the shorter ring
        // inflates per-pair warp latency 6.26 -> 7.41 cyc/inst).
        launch_pack_pair<QT, KVK, PAGED>(p, kmap, vmap, (total_q + T - 1) / T, kvh, stream);
        return;
      }
    }
  }
  if (p.seqlen_q >= kRows) {
    if (KVK == 2 || p.topk < 8) {
      launch_msa<QT, KVK, PAGED, 2, 2, false>(p, kmap, vmap, total_q, kvh, stream);
    } else {
      launch_msa<QT, KVK, PAGED, 2, 1, false>(p, kmap, vmap, total_q, kvh, stream);
    }
  } else {
    launch_msa<QT, KVK, PAGED, (KVK == 2) ? 4 : 3, (KVK == 2) ? 4 : 3, false>(p, kmap, vmap,
                                                                              total_q, kvh, stream);
  }
}

}  // namespace

namespace msa_vibecuda_core {

void core_forward(const msa_vibecuda::CoreParams& p, const msa_vibecuda::KvLayout& kv,
                  const void* k, const void* v, bool q_is_bf16, int kv_kind, bool paged,
                  cudaStream_t stream) {
  CUtensorMap kmap, vmap;
  encode_tmap(&kmap, k, kv, kv_kind, paged, kChunkTok);
  encode_tmap(&vmap, v, kv, kv_kind, paged, kChunkTok);
  const int total_q = p.total_q;
  const int kvh = p.num_kv_heads;
  if (kv_kind == 2) {
    if (paged) {
      dispatch_stages<bf16, 2, true>(p, kmap, vmap, total_q, kvh, stream);
    } else {
      dispatch_stages<bf16, 2, false>(p, kmap, vmap, total_q, kvh, stream);
    }
  } else if (q_is_bf16) {
    if (paged) {
      dispatch_stages<bf16, 0, true>(p, kmap, vmap, total_q, kvh, stream);
    } else {
      dispatch_stages<bf16, 0, false>(p, kmap, vmap, total_q, kvh, stream);
    }
  } else {
    if (paged) {
      dispatch_stages<f16, 1, true>(p, kmap, vmap, total_q, kvh, stream);
    } else {
      dispatch_stages<f16, 1, false>(p, kmap, vmap, total_q, kvh, stream);
    }
  }
}

}  // namespace msa_vibecuda_core
