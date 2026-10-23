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

// SM120 DSA indexer: scoring and exact top-k, the ragged (prefill) form and
// the pieces the decode form shares (backend="sm120" of
// flashinfer.dsa_indexer.dsa_indexer_topk).
//
//   s[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h, :] . k[j, :])
//
// is evaluated exactly as DeepGEMM's sm120_fp8_mqa_logits does it, so the
// scores are bit-identical to its logits: mma.sync m16n8k32 with keys as M (16
// a warp) and heads as N (8 a tile), K-steps accumulated from zero in order;
// per head tile t = fma(relu(d1), w1, relu(d0) * w0), partial += t; then
// partial * k_scale and two butterfly adds over the quad. MXFP4 scales enter
// the MMA instead (fp8_fp4_mqa_logits).
//
// Every row selects its top_k keys by score, ties to the larger key id. All
// launches are fixed by the call's shapes, with no host synchronization:
//   rows_kernel     each row's visible key range, and which resident classes
//                   the rows fall in.
//   RESIDENT        rows of at most 16K / 32K / 48K keys (one launch a class):
//                   a query block per CTA scores its rows into an L2 scratch
//                   row while selector warps select the previous block's rows
//                   (sel_row). No threshold, no cross-CTA step.
// Longer rows take a sampled-threshold scan, in chunks of rows:
//   SAMPLE + tau    exact scores of evenly spread key chunks give each row a
//                   threshold aimed at ~2 * top_k candidates;
//   SCAN + finalize every score; those above the threshold are appended to the
//                   row's candidates, then selected exactly;
//   EXACT           rows whose candidate count missed [want, capacity]: radix
//                   selection over rescanned scores (rare).
// With few rows, the per-row steps run inside the cooperative SAMPLE and SCAN
// launches after a grid barrier.
#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cfloat>
#include <cmath>
#include <type_traits>

namespace flashinfer::sm120_dsa_indexer {

constexpr int kHeadDim = 128;
constexpr int kBlockKV = 128;                          // keys a tile: 8 math warps x 16
constexpr int kSampleTiles = 64;                       // sampled tiles a row, at most
constexpr int kSampleSlots = kSampleTiles * kBlockKV;  // a row's sample slots
// A sampled tile gathers kBlockKV / kChunk chunks of 16 keys spread evenly
// over the block's range: score clusters (recent tokens, attention sinks) are
// what make a sparse sample misjudge the threshold, so the sample covers many
// regions of the row. Long rows also keep the rank at the candidate target at
// least kSampleRank (count noise ~1/sqrt(rank)).
constexpr int kMinSampleTiles = 32;
constexpr int kSampleRank = 64;
constexpr int kChunk = 16;
constexpr int kChunksPerTile = kBlockKV / kChunk;
constexpr int kSmemLimit = 101376;  // SM120's opt-in shared memory a block

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma_fp8(float (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// MXFP4 (e2m1, packed two per byte) with ue8m0 scales per 32 elements: one
// K-step covers 64 elements. sfa / sfb hold the scale bytes of two K-steps;
// kByte (0 or 2) selects this step's pair in the instruction, so both steps
// read one register instead of a shifted copy each.
template <int kByte>
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1."
      "f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%12, %13}, {%11}, "
      "{%12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb),
        "n"(kByte), "n"(0));
}

// K-step ks (0 or 1) of an MXFP4 MMA from the two-step scale words.
__device__ __forceinline__ void mma_fp4_step(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2], uint32_t sfa, uint32_t sfb,
                                             int ks) {
  if (ks == 0)
    mma_fp4<0>(d, a, b, sfa, sfb);
  else
    mma_fp4<2>(d, a, b, sfa, sfb);
}

// Row bytes per format: FP8 rows are 128 bytes (128B TMA swizzle), packed
// MXFP4 rows 64 bytes (64B swizzle). 16-byte chunks are XOR-swizzled so that
// ldmatrix over 8 consecutive rows at one chunk is bank-conflict free.
template <bool kFp4>
struct Fmt {
  static constexpr int kRow = kFp4 ? 64 : 128;
  static constexpr int kKSteps = kFp4 ? 2 : 4;  // 64-element / 32-element MMA K-steps
  __device__ __forceinline__ static int swz(int row, int chunk) {
    if constexpr (kFp4)
      return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
    else
      return row * 128 + ((chunk ^ (row & 7)) << 4);
  }
  // The swizzle repeats every 8 rows: swz(row + 8 * m, c) == swz(row, c) +
  // 8 * m * kRow. So a lane's fragment offset per K-step is computed once and
  // the MMA loops address rows 8 apart with immediates.
  __device__ __forceinline__ static void lane_offsets(int row, int chunk_hi, int (&off)[kKSteps]) {
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) off[ks] = swz(row, ks * 2 + chunk_hi);
  }
};

// Total order used for selection: score descending with +0.0 == -0.0, then the
// larger key id. Returns a key whose unsigned order is ascending in score.
__device__ __forceinline__ uint32_t score_order(float s) {
  uint32_t b = __float_as_uint(s);
  if ((b << 1) == 0) b = 0;  // fold -0.0 onto +0.0
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The score of an order key (score_order's inverse; -0.0 comes back +0.0).
__device__ __forceinline__ float order_value(uint32_t o) {
  return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}

// ---------------------------------------------------------------------------
// Per-row visible key range.
// ---------------------------------------------------------------------------
constexpr int kResClasses = 3;  // resident classes (res_class_cap)

struct RowsParams {
  const int32_t* cu_seqlens_q;      // [S + 1]
  const int32_t* cu_seqlens_k;      // [S + 1]
  const int64_t* q_causal_offsets;  // [S] or nullptr
  int32_t* row_ks;                  // [T] absolute start key
  int32_t* row_ke;                  // [T] absolute end key (exclusive)
  // Per CTA of 256 rows, the classes of its rows as a bit mask: bit i for
  // rows of (i * class_keys, (i + 1) * class_keys] keys, bit kResClasses for
  // longer ones. Or null.
  uint32_t* row_classes;
  int class_keys;
  int num_rows;
  int num_segments;
  int ratio;
};

// Programmatic dependent launch: wait for the previous grid, then let the next
// one launch (it waits for this grid's end itself).
__device__ __forceinline__ void pdl_enter() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
}

// A block whose rows all see at most cap keys is the resident route's; the
// scan takes the rest. cap 0: no split.
__device__ __forceinline__ bool short_block(const int32_t* row_ks, const int32_t* row_ke,
                                            int num_rows, int block_q, int qb, int cap) {
  if (cap <= 0) return false;
  for (int i = 0; i < block_q; ++i) {
    const int r = qb * block_q + i;
    if (r < num_rows && row_ke[r] - row_ks[r] > cap) return false;
  }
  return true;
}

// The most keys a row of block qb sees: short_block(cap) is block_keys <= cap,
// so a resident launch makes both of its tests from one pass.
__device__ __forceinline__ int block_keys(const int32_t* row_ks, const int32_t* row_ke,
                                          int num_rows, int block_q, int qb) {
  int m = 0;
  for (int i = 0; i < block_q; ++i) {
    const int r = qb * block_q + i;
    if (r < num_rows) m = max(m, row_ke[r] - row_ks[r]);
  }
  return m;
}

__device__ __forceinline__ void rows_row(const RowsParams& p, int t, int64_t& vis_out) {
  int lo = 0, hi = p.num_segments;  // last segment with cu_q[s] <= t
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (p.cu_seqlens_q[mid] <= t)
      lo = mid;
    else
      hi = mid;
  }
  const int s = lo;
  const int64_t lq = p.cu_seqlens_q[s + 1] - p.cu_seqlens_q[s];
  const int64_t lk = p.cu_seqlens_k[s + 1] - p.cu_seqlens_k[s];
  const int64_t u = t - p.cu_seqlens_q[s];
  int64_t offset;
  if (p.q_causal_offsets != nullptr)
    offset = p.q_causal_offsets[s];
  else
    offset = p.ratio == 1 ? lk - lq : 0;
  const int64_t num = offset + u + 1;
  int64_t vis = num >= 0 ? num / p.ratio : -((-num + p.ratio - 1) / p.ratio);  // floor
  vis = vis < 0 ? 0 : (vis > lk ? lk : vis);
  p.row_ks[t] = p.cu_seqlens_k[s];
  p.row_ke[t] = p.cu_seqlens_k[s] + static_cast<int32_t>(vis);
  vis_out = vis;
}

__global__ void rows_kernel(const __grid_constant__ RowsParams p) {
  pdl_enter();
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t vis = 0;
  if (t < p.num_rows) rows_row(p, t, vis);
  if (p.row_classes != nullptr) {
    uint32_t m = 0u;
    if (t < p.num_rows) {
      const int64_t c = vis > 0 ? (vis - 1) / p.class_keys : 0;
      m = 1u << (c < kResClasses ? static_cast<int>(c) : kResClasses);
    }
    m = __reduce_or_sync(0xffffffffu, m);
    __shared__ uint32_t wm[8];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t a = 0u;
      for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) a |= wm[w];
      p.row_classes[blockIdx.x] = a;
    }
  }
}

// ---------------------------------------------------------------------------
// TMA / mbarrier helpers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)),
               "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_addr(bar)),
      "r"(phase));
}
// mbar_wait with a suspend-time hint (CUTLASS's ClusterBarrier::wait): the
// warp sleeps until the phase completes instead of polling, which took issue
// slots from the math warps sharing its sub-partition.
constexpr uint32_t kSuspendHintNs = 10000000;
__device__ __forceinline__ void mbar_wait_sleep(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_addr(bar)),
      "r"(phase), "r"(kSuspendHintNs));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar,
                                            int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, "
      "%4}], [%2];\n" ::"r"(smem_addr(dst)),
      "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, uint64_t* bar,
                                            int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, "
      "%4, %5}], "
      "[%2];\n" ::"r"(smem_addr(dst)),
      "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n" ::"l"(map) : "memory");
}

// Device-wide barrier of a cooperative launch.
__device__ __forceinline__ void grid_barrier() { cooperative_groups::this_grid().sync(); }

__device__ __forceinline__ void math_barrier();

// Per-warpgroup register reconfiguration (every warp of the warpgroup).
template <int kRegs>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kRegs));
}
template <int kRegs>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kRegs));
}

// ---------------------------------------------------------------------------
// Scoring kernel: persistent and warp-specialized. 384 threads: warpgroups
// 0-1 are the 8 math warps (16 keys each of a 128-key tile); warpgroup 2's
// first warp issues the TMA loads (a query block's rows into a 2-stage ring,
// key tiles and their scales into a kKStages ring) and, in RESIDENT, its
// other three warps select.
// ---------------------------------------------------------------------------
constexpr int kNumMathThreads = 256;
constexpr int kNumWsThreads = kNumMathThreads + 128;
constexpr int kNumQStages = 2;

// Barrier over the 8 math warps (barrier 1; the producer warpgroup never joins).
__device__ __forceinline__ void math_barrier() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kNumMathThreads) : "memory");
}

// A ring of kN pipeline stages: the current stage and its mbarrier phase
// parity, stepped together (no divisions in the loops).
template <int kN>
struct Ring {
  int st = 0;
  uint32_t parity = 0u;
  __device__ __forceinline__ void next() {
    if (++st == kN) {
      st = 0;
      parity ^= 1u;
    }
  }
};

// SAMPLE: score the sampled chunks. SCAN: score every key, append those at or
// above the row's threshold. EXACT: radix selection of the rows the scan could
// not settle (ExactState). RESIDENT: every score of a query block into the
// CTA's scratch, selected there (sel_row).
enum class Mode { kSample = 0, kScan = 1, kExact = 2, kResident = 3 };

// Score outputs are optional (null: indices only).
__device__ __forceinline__ float* score_row(float* scores, int64_t off) {
  return scores != nullptr ? scores + off : nullptr;
}
__device__ __forceinline__ void put_score(float* s, int i, float v) {
  if (__builtin_expect(s != nullptr, 1)) s[i] = v;
}

// Exact selection of a listed row by rescanning it (bit-identical scores).
// Phases 0-2: radix histograms of the order key (11, 11, 10 bits) find the
// k-th key and how many keys equal it; phases 3-5, only when the keys at or
// above it overflow the candidate buffer: radix histograms of the tied ids
// find the id cut (ties keep the larger ids); phase 6 collects the selection
// into the candidate buffer for finalize. The CTA that completes a row's last
// unit of a phase picks that phase's bin; the row's histogram lives in its
// discarded candidate buffer.
constexpr int kExactBins = 2048;
constexpr int kExactPhases = 7;
struct ExactState {
  uint32_t prefix;  // selected high bits so far (order key, then id)
  int want;         // rank still to find within the prefix
  int eq;           // keys equal to kth (after phase 2)
  int above;        // keys greater than kth
  uint32_t kth;
  int need_ids;
  uint32_t id_cut;
  int done;  // units of the current phase finished
};

__device__ __forceinline__ void exact_pass(int phase, int& shift, uint32_t& bin_mask,
                                           uint32_t& hi_mask) {
  const int q = phase < 3 ? phase : phase - 3;
  shift = q == 0 ? 21 : (q == 1 ? 10 : 0);
  bin_mask = q == 2 ? 0x3ffu : 0x7ffu;
  hi_mask = q == 0 ? 0u : (q == 1 ? 0xffe00000u : 0xfffffc00u);
}

struct TauParams {
  int32_t* zero_words;  // zeroed by block 0 (the later launches' counters)
  int num_zero_words;
  int32_t* tile_prefix;  // the scan's, written by one extra block
  int block_q;
  int num_qblocks;
  const float* sample;
  const int32_t* sample_slots;
  const int32_t* row_ks;
  const int32_t* row_ke;
  float* tau;
  int32_t* count;
  int32_t* fallback_count;  // zeroed here, filled by finalize
  int num_rows;
  int capacity;
  int target;  // candidates a row
  int top_k;
  int res_cap;  // rows of short_block() blocks are the resident route's
};

struct FinalizeParams {
  const int32_t* row_ks;
  const int32_t* row_ke;
  const int32_t* count;
  const float* cand_score;
  const int32_t* cand_id;
  int32_t* indices;        // [T, top_k]
  float* scores;           // [T, top_k]
  int32_t* fallback_rows;  // [T]
  int32_t* fallback_count;
  int num_rows;
  int capacity;
  int top_k;
  int sort_slots;       // power of two >= top_k
  const int32_t* rows;  // EXACT: the listed rows
  const int32_t* rows_count;
  ExactState* xs;  // SCAN: flagged rows start the exact path
  int block_q;     // with res_cap: rows of short_block() blocks are skipped
  int res_cap;
};

struct ScoreParams {
  const int32_t* row_ks;
  const int32_t* row_ke;
  const float* tau;       // [T] (SCAN)
  float* sample;          // [T, kSampleSlots] (SAMPLE)
  int32_t* sample_slots;  // [T] (SAMPLE): slots written per row
  const void* k_scale;    // [Tkv] fp32 (FP8) or int32 ue8m0 x4 (MXFP4), read by SAMPLE
  int sample_target;      // candidates a row, sizes the sample
  int sample_unit_tiles;  // sampled tiles per SAMPLE work unit
  int32_t* count;         // [T] (SCAN)
  float* cand_score;      // [T, capacity] (SCAN)
  int32_t* cand_id;       // [T, capacity] (SCAN), segment-local ids
  int num_rows;
  int num_keys;
  int capacity;
  int kv_splits;  // EXACT: units sharing one row's key range
  int num_units;
  // EXACT: units are (listed row, split); the list length is read on the
  // device.
  const int32_t* unit_rows;
  const int32_t* num_unit_rows;
  // SCAN: inclusive prefix of tiles per query block. Each CTA takes an equal
  // contiguous share of all tiles (stream-K), so no wave is left partly idle.
  const int32_t* tile_prefix;  // [num_qblocks]
  int num_qblocks;
  // EXACT
  ExactState* xs;  // [T]
  // Fused tails (cooperative launches): after a grid barrier, SAMPLE computes
  // every row's threshold and the scan's tile prefix; SCAN and EXACT select
  // every (listed) row.
  int fuse_tail;
  TauParams tau_p;
  FinalizeParams fin_p;
  int top_k;
  // RESIDENT: res_cta_words scratch words a CTA: slots of kBlockQ score rows
  // of res_stride words (a multiple of 32, at least every resident row's
  // keys), then per selecting warp its list (kResList keys and ids) and its
  // kept-key bitmap. The output goes to fin_p.indices / fin_p.scores.
  float* res_scratch;
  int64_t res_stride;
  int64_t res_cta_words;
  // The resident route takes short_block() blocks, the scan the rest (0: no
  // split). A longer class's RESIDENT launch skips the blocks short at res_lo.
  int res_cap;
  int res_lo;
  // RESIDENT: rows_kernel's class masks; a launch with no row of its class
  // leaves at once.
  const uint32_t* row_classes;
  int res_class_ctas;
  int res_class;
};

template <int kNumHeads, bool kFp4>
struct ScoreSmem {
  static constexpr int kBlockQ = 128 / kNumHeads;
  // FP4 tiles are half the bytes: 5 stages, as DeepGEMM's FP4 logits kernel.
  static constexpr int kKStages = kFp4 ? 5 : 3;
  static constexpr int kQBytes = 128 * Fmt<kFp4>::kRow;  // 128 head rows
  static constexpr int kKBytes = kBlockKV * Fmt<kFp4>::kRow;
  static constexpr int kQOff = 0;
  static constexpr int kKOff = kQOff + kNumQStages * kQBytes;   // 1024-aligned (swizzle atom)
  static constexpr int kScaleOff = kKOff + kKStages * kKBytes;  // per-key fp32 scale / int32 sf
  static constexpr int kWOff = kScaleOff + kKStages * kBlockKV * 4;
  static constexpr int kQsfBytes = kFp4 ? kBlockQ * kNumHeads * 4 : 0;  // per (row, head) int32 sf
  static constexpr int kQsfOff = kWOff + kNumQStages * kBlockQ * kNumHeads * 4;
  // EXACT's histogram, or SCAN's candidate staging: a slice of kWarpCap
  // entries per math warp and row (scores, then ids).
  static constexpr int kHistOff = kQsfOff + kNumQStages * kQsfBytes;
  static constexpr int kBarBytes = (2 * kNumQStages + 2 * kKStages) * 8;
  // 99 KB less the static shared memory (the tails' and RESIDENT's state).
  static constexpr int kDynLimit = kSmemLimit - 4096;
  static constexpr int kSlices = (kNumMathThreads / 32) * kBlockQ;
  // 64 entries where they fit: more only shrinks the L1 cache.
  static constexpr int kWarpCapFit = ((kDynLimit - kHistOff - kBarBytes) / (kSlices * 8)) & ~31;
  static constexpr int kWarpCap = kWarpCapFit < 64 ? kWarpCapFit : 64;
  static constexpr int kStageBytes = kSlices * kWarpCap * 8;
  static constexpr int kHistBytes = kExactBins * 4 > kStageBytes ? kExactBins * 4 : kStageBytes;
  static constexpr int kBarOff = kHistOff + kHistBytes;
  static constexpr int kBytes = kBarOff + kBarBytes;
  static_assert(kWarpCap >= 32, "a warp's staging below two tiles");
  static_assert(kBytes <= kDynLimit, "SM120 shared memory");
  // RESIDENT: a histogram per selector warp, in the staging area and past
  // the barriers: 2048 bins where all three fit, else 1024.
  static constexpr int kSelAreaB = kDynLimit - ((kBarOff + kBarBytes + 15) & ~15);
  static constexpr int kSelHB =
      kHistBytes / 8192 + kSelAreaB / 8192 >= 3 ? 8192 : 4096;  // bytes per warp
  static constexpr int kSelHBits = kSelHB == 8192 ? 11 : 10;
  static constexpr int kSelInA = kHistBytes / kSelHB < 3 ? kHistBytes / kSelHB : 3;
  static constexpr int kSelOffB = (kBarOff + kBarBytes + 15) & ~15;
  static constexpr int kResBytes = kSelOffB + (3 - kSelInA) * kSelHB;
  // RESIDENT: once their scan is done, a histogram per math warp in the
  // pipeline's buffers.
  static constexpr int kMathHB = kHistOff >= 8 * 8192 ? 8192 : 4096;
  static constexpr int kMathHBits = kMathHB == 8192 ? 11 : 10;
  static_assert(8 * kMathHB <= kHistOff, "math warps' histograms");
  __host__ __device__ static constexpr int sel_hist_off(int w) {
    return w < kSelInA ? kHistOff + w * kSelHB : kSelOffB + (w - kSelInA) * kSelHB;
  }
  static_assert(kHistBytes / 4096 + kSelAreaB / 4096 >= 3 && kResBytes <= kDynLimit,
                "SM120 shared memory");
};

// Sampled tiles of a query block spanning ntiles_total tiles.
__device__ __forceinline__ int sample_count(const ScoreParams& p, int ntiles_total) {
  // rank at the target = target * sampled / visible >= kSampleRank
  const int64_t by_rank = (static_cast<int64_t>(kSampleRank) * ntiles_total + p.sample_target - 1) /
                          max(p.sample_target, 1);
  // Coverage: kMinSampleTiles chunked tiles for long rows, an eighth of a
  // shorter row's tiles (at least 8).
  const int min_tiles = min(kMinSampleTiles, max(8, ntiles_total / 8));
  const int n = by_rank > kSampleTiles
                    ? kSampleTiles
                    : (by_rank < min_tiles ? min_tiles : static_cast<int>(by_rank));
  return min(ntiles_total, n);
}

// A unit's tile range. Tiles are aligned to 128 keys, so every TMA box is;
// keys outside a row's visible range are masked.
template <int kBlockQ, Mode kMode>
__device__ __forceinline__ void unit_tiles(const ScoreParams& p, int phase, int unit, int& row0,
                                           int& only_qi, int& base, int& ntiles_total,
                                           int& tile_begin, int& ntiles) {
  const int sample_units = kSampleTiles / p.sample_unit_tiles;
  int q_block = kMode != Mode::kSample ? unit / p.kv_splits : unit / sample_units;
  const int split = kMode != Mode::kSample ? unit % p.kv_splits : unit % sample_units;
  only_qi = -1;
  if (p.unit_rows != nullptr) {
    const int row = __ldg(p.unit_rows + q_block);
    q_block = row / kBlockQ;
    only_qi = row % kBlockQ;
  }
  row0 = q_block * kBlockQ;
  int start = INT32_MAX, end = 0;
#pragma unroll
  for (int i = 0; i < kBlockQ; ++i) {
    const int r = row0 + i;
    if (r < p.num_rows && (only_qi < 0 || i == only_qi)) {
      const int ks = __ldg(p.row_ks + r), ke = __ldg(p.row_ke + r);
      if (ke > ks) {
        start = min(start, ks);
        end = max(end, ke);
      }
    }
  }
  // The resident route's blocks: nothing to sample. (SCAN's tile prefix
  // already gives them no tiles.)
  if constexpr (kMode == Mode::kSample) {
    if (p.unit_rows == nullptr &&
        short_block(p.row_ks, p.row_ke, p.num_rows, kBlockQ, q_block, p.res_cap))
      end = 0;
  }
  base = end > start ? start & ~(kBlockKV - 1) : 0;
  ntiles_total = end > start ? (end - base + kBlockKV - 1) / kBlockKV : 0;
  if constexpr (kMode == Mode::kSample) {
    // This unit's share [tile_begin, tile_begin + ntiles) of the block's
    // sample_count() sampled tiles.
    const int n = sample_count(p, ntiles_total);
    tile_begin = min(split * p.sample_unit_tiles, n);
    ntiles = min(n - tile_begin, p.sample_unit_tiles);
  } else {
    const int chunk = (ntiles_total + p.kv_splits - 1) / p.kv_splits;
    tile_begin = split * chunk;
    ntiles = max(0, min(chunk, ntiles_total - tile_begin));
    if (kMode == Mode::kExact && phase >= 3 && phase < 6 && !p.xs[row0 + only_qi].need_ids)
      ntiles = 0;
  }
}

// RESIDENT (H32): the query block a CTA takes at position pos (round *
// gridDim.x + blockIdx.x), from the last (the longest under a causal mask),
// the rounds dealt alternately forward and backward so that each CTA's long
// and short blocks pair up. -1: none.
template <bool kSnake>
__device__ __forceinline__ int res_unit(int pos, int num_units) {
  int i = pos;
  if constexpr (kSnake) {
    const int g = static_cast<int>(gridDim.x), r = pos / g;
    if (r & 1) i = r * g + (g - 1 - static_cast<int>(blockIdx.x));
  }
  return i < num_units ? num_units - 1 - i : -1;
}

__device__ __forceinline__ int num_units_of(const ScoreParams& p) {
  return p.num_unit_rows != nullptr ? *p.num_unit_rows * p.kv_splits : p.num_units;
}

// The sequence of work pieces one CTA processes; the producer and the math
// warps walk identical copies.
template <int kBlockQ, Mode kMode, bool kResLo = false>
struct WorkIter {
  int unit, num_units;  // unit mode
  int pos, end, qb;     // stream-K mode: global tile range [pos, end)

  __device__ __forceinline__ explicit WorkIter(const ScoreParams& p) {
    if (kMode == Mode::kScan && p.tile_prefix != nullptr) {
      const int total = p.num_qblocks > 0 ? __ldg(p.tile_prefix + p.num_qblocks - 1) : 0;
      const int per = (total + gridDim.x - 1) / gridDim.x;
      pos = min(static_cast<int>(blockIdx.x) * per, total);
      end = min(pos + per, total);
      int lo = 0, hi = p.num_qblocks;  // first block whose prefix exceeds pos
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (__ldg(p.tile_prefix + mid) > pos)
          hi = mid;
        else
          lo = mid + 1;
      }
      qb = lo;
      unit = num_units = 0;
    } else {
      unit = blockIdx.x;
      num_units = num_units_of(p);
      pos = end = qb = 0;
    }
  }

  __device__ __forceinline__ bool next(const ScoreParams& p, int phase, int& row0, int& only_qi,
                                       int& base, int& ntiles_total, int& tile_begin, int& ntiles) {
    if (kMode == Mode::kScan && p.tile_prefix != nullptr) {
      if (pos >= end) return false;
      while (__ldg(p.tile_prefix + qb) <= pos) ++qb;
      const int first = qb > 0 ? __ldg(p.tile_prefix + qb - 1) : 0;
      unit_tiles<kBlockQ, kMode>(p, phase, qb * p.kv_splits, row0, only_qi, base, ntiles_total,
                                 tile_begin, ntiles);
      tile_begin = pos - first;
      ntiles = min(end, __ldg(p.tile_prefix + qb)) - pos;
      pos += ntiles;
      ++qb;
      return true;
    }
    // RESIDENT: the last query blocks first (the longest under a causal
    // mask), so the persistent CTAs end together. H32 snakes the rounds
    // (res_unit); H64 keeps the plain loop, whose scan the snaking code slows
    // even when unused.
    constexpr bool kSnake = kMode == Mode::kResident && kBlockQ == 4;
    while (kSnake ? unit - static_cast<int>(blockIdx.x) < num_units : unit < num_units) {
      int u;
      if constexpr (kSnake) {
        u = res_unit<true>(unit, num_units);
        unit += gridDim.x;
        if (u < 0) continue;
      } else {
        u = kMode == Mode::kResident ? num_units - 1 - unit : unit;
        unit += gridDim.x;
      }
      // RESIDENT beside the scan: its blocks only; a longer class: not the
      // shorter class's blocks. One pass over the block's rows serves both.
      if (kMode == Mode::kResident && (p.res_cap > 0 || kResLo)) {
        const int bk = block_keys(p.row_ks, p.row_ke, p.num_rows, kBlockQ, u);
        if (p.res_cap > 0 && bk > p.res_cap) continue;
        if (kResLo && p.res_lo > 0 && bk <= p.res_lo) continue;
      }
      unit_tiles<kBlockQ, kMode>(p, phase, u, row0, only_qi, base, ntiles_total, tile_begin,
                                 ntiles);
      // SAMPLE: a short row samples fewer tiles than its block has units.
      if (kMode != Mode::kSample || ntiles > 0) return true;
    }
    return false;
  }
};

// Inclusive prefix of tiles per query block (one CTA), for stream-K scans.
__device__ __forceinline__ void tile_prefix_block(const int32_t* row_ks, const int32_t* row_ke,
                                                  int num_rows, int block_q, int num_qblocks,
                                                  int res_cap, int32_t* prefix) {
  __shared__ int s_warp[32];
  __shared__ int s_carry;
  if (threadIdx.x == 0) s_carry = 0;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int c = 0; c < num_qblocks; c += blockDim.x) {
    const int qb = c + threadIdx.x;
    int n = 0;
    if (qb < num_qblocks) {
      int start = INT32_MAX, end = 0;
      for (int i = 0; i < block_q; ++i) {
        const int r = qb * block_q + i;
        if (r < num_rows) {
          const int ks = row_ks[r], ke = row_ke[r];
          if (ke > ks) {
            start = min(start, ks);
            end = max(end, ke);
          }
        }
      }
      if (end > start && !short_block(row_ks, row_ke, num_rows, block_q, qb, res_cap)) {
        const int base = start & ~(kBlockKV - 1);
        n = (end - base + kBlockKV - 1) / kBlockKV;
      }
    }
    int x = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    __syncthreads();  // s_carry and s_warp of the previous chunk are consumed
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
      int t = lane < static_cast<int>(blockDim.x >> 5) ? s_warp[lane] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, t, o);
        if (lane >= o) t += y;
      }
      s_warp[lane] = t;
    }
    __syncthreads();
    const int incl = x + (warp > 0 ? s_warp[warp - 1] : 0) + s_carry;
    if (qb < num_qblocks) prefix[qb] = incl;
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) s_carry = incl;
  }
}

// SAMPLE: the c-th chunk of sampled tile i as a chunk index into the block's
// range (contiguous when every tile is sampled). 32-bit: below 2^32 for rows
// of fewer than 2^30 keys.
__device__ __forceinline__ int chunk_of(int i, int c, int ntiles_total, int ntiles) {
  return static_cast<int>(static_cast<uint32_t>(i * kChunksPerTile + c) *
                          static_cast<uint32_t>(ntiles_total) / static_cast<uint32_t>(ntiles));
}

// EXACT, end of a unit (all 256 math threads): add the unit's histogram to
// the row's; the CTA finishing the row's last unit of this phase picks the
// bin holding the wanted rank and advances the row's state.
__device__ __noinline__ void exact_unit_done(const ScoreParams& p, int phase, int r,
                                             uint32_t* shist, int tid) {
  __shared__ int s_last;
  uint32_t* ghist =
      reinterpret_cast<uint32_t*>(p.cand_score + static_cast<int64_t>(r) * p.capacity);
  ExactState& st = p.xs[r];
  math_barrier();
  for (int b = tid; b < kExactBins; b += kNumMathThreads) {
    const uint32_t c = shist[b];
    if (c) {
      atomicAdd(ghist + b, c);
      shist[b] = 0;
    }
  }
  __threadfence();
  math_barrier();
  if (tid == 0) s_last = atomicAdd(&st.done, 1) == p.kv_splits - 1;
  math_barrier();
  if (!s_last) return;
  __threadfence();
  int shift;
  uint32_t bin_mask, hi_mask;
  exact_pass(phase, shift, bin_mask, hi_mask);
  // Descending bins, 8 per thread: thread t owns bins 2047 - 8t ... 2040 - 8t.
  __shared__ int s_warp[8];
  __shared__ int s_found, s_want0;
  __shared__ uint32_t s_prefix0;
  const int lane = tid & 31, warp = tid >> 5;
  uint32_t c[8];
  int sum = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int b = kExactBins - 1 - 8 * tid - j;
    c[j] = __ldcg(ghist + b);
    sum += static_cast<int>(c[j]);
  }
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) s_warp[warp] = incl;
  if (tid == 0) {
    // Snapshot before the finder below rewrites the state.
    s_found = 0;
    s_want0 = st.want;
    s_prefix0 = st.prefix;
  }
  math_barrier();
  for (int w = 0; w < warp; ++w) incl += s_warp[w];
  const int want = s_want0;
  if (incl >= want && incl - sum < want) {
    // This thread's 8 bins hold the wanted rank.
    int rem = want - (incl - sum);
    int j = 0;
    for (; j < 7; ++j) {
      if (rem <= static_cast<int>(c[j])) break;
      rem -= static_cast<int>(c[j]);
    }
    const uint32_t b = static_cast<uint32_t>(kExactBins - 1 - 8 * tid - j);
    const uint32_t prefix = s_prefix0 | (b << shift);
    if (phase < 3) {
      if (phase == 2) {
        // prefix is the k-th order key; rem of the keys equal to it are selected.
        st.kth = prefix;
        st.eq = static_cast<int>(c[j]);
        st.prefix = 0;
        st.want = rem;
      } else {
        st.prefix = prefix;
        st.want = rem;
      }
    } else {
      if (phase == 5)
        st.id_cut = prefix;
      else {
        st.prefix = prefix;
        st.want = rem;
      }
    }
    s_found = 1;
  }
  math_barrier();
  if (tid == 0 && phase == 2) {
    // Keys >= kth: `above` greater, `eq` equal. Only when they overflow the
    // candidate buffer do the id passes pick which ties to keep.
    const int ties = st.want;
    const int total = min(p.top_k, p.row_ke[r] - __ldg(p.row_ks + r));
    st.above = total - ties;
    st.need_ids = st.above + st.eq > p.capacity;
  }
  for (int b = tid; b < kExactBins; b += kNumMathThreads) ghist[b] = 0;
  if (tid == 0) st.done = 0;
  __threadfence();
}

// ---------------------------------------------------------------------------
// Block-wide bitonic sort of (key, value) pairs in shared memory, descending
// by key. n must be a power of two.
// ---------------------------------------------------------------------------
template <typename K, typename V>
__device__ __forceinline__ void bitonic_sort_desc(K* key, V* val, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int i = threadIdx.x; i < (n >> 1); i += blockDim.x) {
        const int lo = 2 * i - (i & (stride - 1));
        const int hi = lo + stride;
        const bool desc = (lo & size) == 0;
        const K a = key[lo], b = key[hi];
        if ((a < b) == desc) {
          key[lo] = b;
          key[hi] = a;
          const V t = val[lo];
          val[lo] = val[hi];
          val[hi] = t;
        }
      }
    }
  }
  __syncthreads();
}

// Bitonic sort, descending by key, of n (a power of two, 32 * kE <= n <= 1024
// * kE) (key, value) pairs: thread t holds elements [t * kE, t * kE + kE) in
// registers. Compare-exchanges within a thread or a warp need no block
// barrier; only strides spanning warps go through shared memory (skey /
// sval, which also carry the input and output).
template <int kE>
__device__ __forceinline__ void bitonic_sort_desc_regs(uint32_t* skey, float* sval, int n) {
  const int t = threadIdx.x;
  const bool active = t < n / kE;  // whole warps: n / kE is a multiple of 32
  uint32_t k[kE];
  float v[kE];
#pragma unroll
  for (int e = 0; e < kE; ++e) {
    k[e] = active ? skey[t * kE + e] : 0u;
    v[e] = active ? sval[t * kE + e] : 0.0f;
  }
  for (int size = 2; size <= n; size <<= 1) {
    for (int j = size >> 1; j > 0; j >>= 1) {
      if (j < kE) {  // partner in this thread; the stride is resolved at compile
                     // time so the register arrays stay in registers
#pragma unroll
        for (int jj = kE / 2; jj >= 1; jj >>= 1) {
          if (jj != j) continue;
#pragma unroll
          for (int e = 0; e < kE; ++e) {
            if ((e & jj) != 0) continue;
            const int i = t * kE + e;
            const bool desc = (i & size) == 0;
            if ((k[e] < k[e + jj]) == desc) {
              const uint32_t tk = k[e];
              k[e] = k[e + jj];
              k[e + jj] = tk;
              const float tv = v[e];
              v[e] = v[e + jj];
              v[e + jj] = tv;
            }
          }
        }
      } else if (j < 32 * kE) {  // partner in this warp
        if (active) {
#pragma unroll
          for (int e = 0; e < kE; ++e) {
            const int i = t * kE + e;
            const uint32_t pk = __shfl_xor_sync(0xffffffffu, k[e], j / kE);
            const float pv = __shfl_xor_sync(0xffffffffu, v[e], j / kE);
            const bool lo = (i & j) == 0, desc = (i & size) == 0;
            if (lo == desc ? pk > k[e] : pk < k[e]) {
              k[e] = pk;
              v[e] = pv;
            }
          }
        }
      } else {  // partner in another warp
        __syncthreads();
        if (active) {
#pragma unroll
          for (int e = 0; e < kE; ++e) {
            skey[t * kE + e] = k[e];
            sval[t * kE + e] = v[e];
          }
        }
        __syncthreads();
        if (active) {
#pragma unroll
          for (int e = 0; e < kE; ++e) {
            const int i = t * kE + e;
            const uint32_t pk = skey[i ^ j];
            const float pv = sval[i ^ j];
            const bool lo = (i & j) == 0, desc = (i & size) == 0;
            if (lo == desc ? pk > k[e] : pk < k[e]) {
              k[e] = pk;
              v[e] = pv;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (active) {
#pragma unroll
    for (int e = 0; e < kE; ++e) {
      skey[t * kE + e] = k[e];
      sval[t * kE + e] = v[e];
    }
  }
  __syncthreads();
}

// Block-wide radix select (4 passes of 8 bits, 256-bin histogram) of the
// want-th largest key among the entries i < n for which fetch(i, k) returns
// true. On return *s_want holds the rank left inside the selected key (how
// many entries equal to it belong to the top `want`) and *s_eq how many
// entries equal it. Warp 0 locates each pass's bin in parallel.
template <typename Fetch>
__device__ __forceinline__ uint32_t radix_select_desc_if(int n, int want, Fetch fetch,
                                                         uint32_t* hist /* [256] */,
                                                         uint32_t* s_prefix, int* s_want,
                                                         int* s_eq) {
  if (threadIdx.x == 0) {
    *s_prefix = 0;
    *s_want = want;
  }
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    const uint32_t mask_hi = pass == 0 ? 0u : (0xffffffffu << (32 - 8 * pass));
    for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    const uint32_t prefix = *s_prefix;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      uint32_t k;
      if (fetch(i, k) && (k & mask_hi) == prefix) atomicAdd(hist + ((k >> shift) & 0xffu), 1u);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      // Lane l owns bins 255 - 8l ... 248 - 8l (descending key order).
      const int lane = threadIdx.x;
      int c[8], sum = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        c[j] = static_cast<int>(hist[255 - 8 * lane - j]);
        sum += c[j];
      }
      int incl = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      const int w = *s_want;
      const uint32_t hit = __ballot_sync(0xffffffffu, incl >= w);
      const int owner = hit ? __ffs(hit) - 1 : 31;
      if (lane == owner) {
        // The first of this lane's bins whose running count reaches w
        // (unrolled with constant indices: c stays in registers).
        int rem = w - (incl - sum), j = 7, eq = c[7];
        bool found = false;
#pragma unroll
        for (int jj = 0; jj < 7; ++jj) {
          if (!found && rem <= c[jj]) {
            found = true;
            j = jj;
            eq = c[jj];
          } else if (!found) {
            rem -= c[jj];
          }
        }
        const int b = 255 - 8 * lane - j;
        *s_want = rem;
        *s_eq = eq;
        *s_prefix = prefix | (static_cast<uint32_t>(b) << shift);
      }
    }
    __syncthreads();
  }
  return *s_prefix;
}

// ---------------------------------------------------------------------------
// A row's threshold from its sample: the m-th largest sampled score with
// m = ceil(target * sampled / visible); -inf when every visible key fits in
// the candidate buffer. All threads of the block; key: kSampleSlots words of
// shared memory.
// ---------------------------------------------------------------------------
__device__ __noinline__ void tau_row(const TauParams& p, int r, uint32_t* key) {
  if (p.res_cap > 0 &&
      short_block(p.row_ks, p.row_ke, p.num_rows, p.block_q, r / p.block_q, p.res_cap))
    return;  // the resident route's row
  __shared__ int s_valid;
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_prefix;
  __shared__ int s_want, s_eq;
  __syncthreads();  // the previous row's readers are done with the shared state
  const int visible = p.row_ke[r] - __ldg(p.row_ks + r);
  if (threadIdx.x == 0) p.count[r] = 0;
  if (visible <= p.capacity) {
    if (threadIdx.x == 0) p.tau[r] = -INFINITY;
    return;
  }
  const float* src = p.sample + static_cast<int64_t>(r) * kSampleSlots;
  const int slots = min(p.sample_slots[r], kSampleSlots);
  if (threadIdx.x == 0) s_valid = 0;
  __syncthreads();
  int local = 0;
  // 16-byte loads and stores; keys past `slots` are written but never read.
  static_assert(kSampleSlots % 4 == 0, "16-byte sample rows");
  for (int v = threadIdx.x; 4 * v < slots; v += blockDim.x) {
    const float4 f = reinterpret_cast<const float4*>(src)[v];
    const float x[4] = {f.x, f.y, f.z, f.w};
    uint32_t k[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const bool ok = 4 * v + j < slots && !(x[j] == -INFINITY);
      k[j] = ok ? score_order(x[j]) : 0u;
      local += ok;
    }
    reinterpret_cast<uint4*>(key)[v] = make_uint4(k[0], k[1], k[2], k[3]);
  }
  atomicAdd(&s_valid, local);
  __syncthreads();
  const int m = s_valid;
  if (m == 0) {
    if (threadIdx.x == 0) p.tau[r] = -INFINITY;
    return;
  }
  // The candidate count above the m-th sampled score has relative noise
  // ~1/sqrt(m): keep m >= 32 by raising the target when the sample is sparse
  // for the row, up to 3/4 of the capacity (the noise's headroom).
  int64_t target = p.target;
  const int64_t min_target = (32ll * visible + m - 1) / m;
  if (target < min_target) target = min_target;
  if (target > 3ll * p.capacity / 4) target = 3ll * p.capacity / 4;
  int64_t want = (target * m + visible - 1) / visible;
  want = want < 1 ? 1 : (want > m ? m : want);
  const uint32_t o = radix_select_desc_if(
      slots, static_cast<int>(want),
      [&](int i, uint32_t& k) {
        k = key[i];
        return true;
      },
      hist, &s_prefix, &s_want, &s_eq);
  if (threadIdx.x == 0) p.tau[r] = order_value(o);
}

// Setup shared by the threshold launch and the fused sample launch: zero the
// later launches' counters (no fill launch) and write the tile prefix.
__device__ __forceinline__ void tau_setup(const TauParams& p, bool zero_block, bool prefix_block) {
  if (zero_block && p.zero_words != nullptr && threadIdx.x < p.num_zero_words)
    p.zero_words[threadIdx.x] = 0;
  if (zero_block && threadIdx.x == 0) *p.fallback_count = 0;
  if (prefix_block && p.tile_prefix != nullptr)
    tile_prefix_block(p.row_ks, p.row_ke, p.num_rows, p.block_q, p.num_qblocks, p.res_cap,
                      p.tile_prefix);
}

// The parameters stay in parameter space (__grid_constant__): tau_row and
// finalize_row take them by reference, which otherwise copies them to every
// thread's stack (several times these kernels' early-exit cost).
__global__ void __launch_bounds__(1024) tau_kernel(const __grid_constant__ TauParams p) {
  __shared__ __align__(16) uint32_t key[kSampleSlots];
  pdl_enter();
  if (p.tile_prefix != nullptr && blockIdx.x == gridDim.x - 1) {
    tau_setup(p, false, true);
    return;
  }
  tau_setup(p, blockIdx.x == 0, false);
  if (static_cast<int>(blockIdx.x) < p.num_rows) tau_row(p, blockIdx.x, key);
}

// ---------------------------------------------------------------------------
// Exact top-k over the candidates; ascending ids with aligned scores.
// ---------------------------------------------------------------------------

// The k-th order key comes from a radix select over the candidates; among the
// entries tied with it, a second radix select over their ids keeps the
// largest. Only the k selected entries are ordered by id. All threads of the
// block; smem_f: finalize_smem_bytes.
__device__ __noinline__ void finalize_row(const FinalizeParams& p, int r, uint8_t* smem_f) {
  // Scores are rebuilt from their order keys: a selected -0.0 is written as
  // +0.0. The resident route's rows are its own.
  if (p.res_cap > 0 &&
      short_block(p.row_ks, p.row_ke, p.num_rows, p.block_q, r / p.block_q, p.res_cap))
    return;
  uint32_t* ord = reinterpret_cast<uint32_t*>(smem_f);              // [capacity]
  uint32_t* ids = ord + p.capacity;                                 // [capacity]
  uint32_t* id_key = ids + p.capacity;                              // [sort_slots]: ~id
  float* id_val = reinterpret_cast<float*>(id_key + p.sort_slots);  // [sort_slots]
  __shared__ uint32_t hist[256];
  __shared__ uint32_t s_prefix;
  __shared__ int s_want, s_n, s_eq;
  __syncthreads();  // the previous row's readers are done with the shared state
  const int visible = p.row_ke[r] - __ldg(p.row_ks + r);
  const int c = p.count[r];
  const int want = min(p.top_k, visible);
  int32_t* out_i = p.indices + static_cast<int64_t>(r) * p.top_k;
  float* out_s = score_row(p.scores, static_cast<int64_t>(r) * p.top_k);
  if (c < want || c > p.capacity) {
    if (threadIdx.x == 0) p.fallback_rows[atomicAdd(p.fallback_count, 1)] = r;
    if (p.xs != nullptr) {
      // The candidates are discarded: the buffer becomes the row's histogram.
      uint32_t* ghist = reinterpret_cast<uint32_t*>(const_cast<float*>(p.cand_score) +
                                                    static_cast<int64_t>(r) * p.capacity);
      for (int b = threadIdx.x; b < kExactBins; b += blockDim.x) ghist[b] = 0;
      if (threadIdx.x == 0) {
        p.xs[r] = ExactState{0u, want, 0, 0, 0u, 0, 0u, 0};
        const_cast<int32_t*>(p.count)[r] = 0;
      }
    }
    return;
  }
  const int64_t off = static_cast<int64_t>(r) * p.capacity;
  if ((p.capacity & 3) == 0) {
    // 16-byte loads and stores (rows are 16-byte aligned; entries past c are
    // stale and never read): scalar loads saturated the load queue.
    const float4* cs = reinterpret_cast<const float4*>(p.cand_score + off);
    const int4* ci = reinterpret_cast<const int4*>(p.cand_id + off);
    for (int v = threadIdx.x; 4 * v < c; v += blockDim.x) {
      const float4 sv = cs[v];
      const int4 iv = ci[v];
      reinterpret_cast<uint4*>(ord)[v] =
          make_uint4(score_order(sv.x), score_order(sv.y), score_order(sv.z), score_order(sv.w));
      reinterpret_cast<uint4*>(ids)[v] = make_uint4(iv.x, iv.y, iv.z, iv.w);
    }
  } else {
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
      ord[i] = score_order(p.cand_score[off + i]);
      ids[i] = static_cast<uint32_t>(p.cand_id[off + i]);
    }
  }
  if (threadIdx.x == 0) s_n = 0;
  __syncthreads();
  const uint32_t kth = radix_select_desc_if(
      c, want,
      [&](int i, uint32_t& k) {
        k = ord[i];
        return true;
      },
      hist, &s_prefix, &s_want, &s_eq);
  const int ties = s_want;  // entries equal to kth that are selected
  // Ties at the k-th score (rare unless scores repeat): keep the larger ids.
  uint32_t id_cut = 0;
  if (ties < s_eq)
    id_cut = radix_select_desc_if(
        c, ties,
        [&](int i, uint32_t& k) {
          k = ids[i];
          return ord[i] == kth;
        },
        hist, &s_prefix, &s_want, &s_eq);
  // Compact the selection: one shared-counter atomic per warp.
  const int c_round = (c + 31) & ~31;
  for (int i = threadIdx.x; i < c_round; i += blockDim.x) {
    const uint32_t o = i < c ? ord[i] : 0u;
    const bool take = i < c && (o > kth || (o == kth && ids[i] >= id_cut));
    const uint32_t m = __ballot_sync(0xffffffffu, take);
    const int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == 0 && m) base = atomicAdd(&s_n, __popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (take) {
      const int pos = base + __popc(m & ((1u << lane) - 1u));
      id_key[pos] = ~ids[i];
      id_val[pos] = order_value(o);
    }
  }
  __syncthreads();
  // Ascending ids, a deterministic order. The ids are distinct and below
  // `visible`: rank them through a bitmap of the row's keys (popcount
  // prefix) when it fits in the now free candidate buffers.
  const int words = (visible + 31) >> 5;
  if (words <= p.capacity) {  // ord and ids hold capacity words each
    uint32_t* bm = ord;       // [words] bits of the selected ids
    uint32_t* pre = ids;      // [words] selected ids in earlier words
    for (int w = threadIdx.x; w < words; w += blockDim.x) bm[w] = 0u;
    __syncthreads();
    for (int i = threadIdx.x; i < want; i += blockDim.x) {
      const uint32_t id = ~id_key[i];
      atomicOr(bm + (id >> 5), 1u << (id & 31));
    }
    __syncthreads();
    // Exclusive prefix of the words' popcounts: per-thread chunks, then a
    // block scan of the chunk sums.
    __shared__ int s_part[32];
    const int per = (words + blockDim.x - 1) / blockDim.x;
    const int w0 = threadIdx.x * per, w1 = min(words, w0 + per);
    int sum = 0;
    for (int w = w0; w < w1; ++w) sum += __popc(bm[w]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    if (lane == 31) s_part[warp] = incl;
    __syncthreads();
    if (warp == 0) {
      int x = lane < static_cast<int>(blockDim.x >> 5) ? s_part[lane] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      s_part[lane] = x;
    }
    __syncthreads();
    int run = incl - sum + (warp > 0 ? s_part[warp - 1] : 0);
    for (int w = w0; w < w1; ++w) {
      pre[w] = run;
      run += __popc(bm[w]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < want; i += blockDim.x) {
      const uint32_t id = ~id_key[i];
      const uint32_t w = id >> 5;
      const int rank = pre[w] + __popc(bm[w] & ((1u << (id & 31)) - 1u));
      out_i[rank] = static_cast<int32_t>(id);
      put_score(out_s, rank, id_val[i]);
    }
    for (int i = want + threadIdx.x; i < p.top_k; i += blockDim.x) {
      out_i[i] = -1;
      put_score(out_s, i, -INFINITY);
    }
    return;
  }
  for (int i = want + threadIdx.x; i < p.sort_slots; i += blockDim.x) {
    id_key[i] = 0;
    id_val[i] = -INFINITY;
  }
  int n_sort = 1;
  while (n_sort < want) n_sort <<= 1;
  // Rows too long for the bitmap: sort the selection by ascending id, kE
  // elements a thread (1024 threads in finalize_kernel, 256 of the fused
  // tail's 384).
  const int span = blockDim.x >= 1024 ? 1024 : 256;
  if (n_sort < 32)
    bitonic_sort_desc(id_key, id_val, n_sort);
  else if (n_sort <= span)
    bitonic_sort_desc_regs<1>(id_key, id_val, n_sort);
  else if (n_sort <= 2 * span)
    bitonic_sort_desc_regs<2>(id_key, id_val, n_sort);
  else if (n_sort <= 4 * span)
    bitonic_sort_desc_regs<4>(id_key, id_val, n_sort);
  else if (n_sort <= 8 * span)
    bitonic_sort_desc_regs<8>(id_key, id_val, n_sort);
  else
    bitonic_sort_desc_regs<16>(id_key, id_val, n_sort);
  for (int i = threadIdx.x; i < p.top_k; i += blockDim.x) {
    if (i < want) {
      out_i[i] = static_cast<int32_t>(~id_key[i]);
      put_score(out_s, i, id_val[i]);
    } else {
      out_i[i] = -1;
      put_score(out_s, i, -INFINITY);
    }
  }
}

// Shared memory the selection of one row needs: candidate keys and ids, then
// the selected ids and scores.
__host__ __device__ inline int finalize_smem_bytes(int capacity, int sort_slots) {
  return capacity * 2 * 4 + sort_slots * 2 * 4;
}

__global__ void __launch_bounds__(1024) finalize_kernel(const __grid_constant__ FinalizeParams p) {
  extern __shared__ __align__(16) uint8_t smem_f[];
  pdl_enter();
  if (static_cast<int>(blockIdx.x) < p.num_rows) finalize_row(p, blockIdx.x, smem_f);
}

// RESIDENT scratch. The warps that select, each with its own list and
// bitmap: the three selectors and, once their scan is done, the eight math
// warps.
constexpr int kSelClaimers = 11;
// Score slots a CTA: the scan runs ahead of the selection, which is the
// slower of the two on short rows.
constexpr int kResSlots = 4;
// A list holds a row's cut bin once narrowed to at most a histogram's bins
// (a larger bin is a single tied key: read from the row instead).
constexpr int kResList = 2048;
// A selecting warp's list and bitmap words (the bitmap padded to 128 bytes, so
// every region and the next CTA's score rows stay aligned for vector loads).
__host__ __device__ constexpr int64_t res_claimer_words(int64_t stride) {
  return 2 * kResList + (stride / 32 + 31) / 32 * 32;
}
// Words per CTA for `slots` slots of score rows of `stride` words (a
// multiple of 32).
__host__ __device__ constexpr int64_t res_cta_words(int block_q, int64_t stride,
                                                    int slots = kResSlots) {
  return slots * block_q * stride + kSelClaimers * res_claimer_words(stride);
}
// Resident class i takes the blocks of rows past res_class_cap(i - 1) and at
// most res_class_cap(i) caps of keys, with res_class_slots(i) slots of rows
// that long. Up to 48K keys (two slots) it beats the sampled-threshold scan's
// fixed costs; past that, or with a single slot (no overlap of selection and
// scan), it loses.
__host__ __device__ constexpr int res_class_slots(int i) {
  return i == 0 ? kResSlots : kResSlots / 2;
}
// In units of the first class's cap.
__host__ __device__ constexpr int res_class_cap(int i) { return i + 1; }

// RESIDENT selection: three warps of the producer warpgroup select a query
// block's rows while the math warps score the next block (mbarrier handoffs
// both ways through kResSlots scratch slots), and the math warps join once
// their scan is done. A warp selects whole rows alone, claimed from a CTA
// counter, with its own histogram, list and bitmap: warp-level
// synchronization only, and no warp waits on another's L2 round trips.
constexpr int kSelWarp0 = kNumMathThreads / 32 + 1;  // warps 9-11 (8 issues TMA)
constexpr int kSelWarps = 3;

struct SelShared {
  int bin, rem, cnt, m;
};

// The bin from the top of hist[0, nbins) holding rank want (1-based; lane t
// owns ceil(nbins / 32) bins from the top down); rem: the rank within it,
// cnt: its count.
__device__ __forceinline__ int sel_rank(const uint32_t* hist, int want, int t, SelShared& ss,
                                        int& rem, int& cnt, int nbins) {
  const int lane = t;
  const int pb = (nbins + 31) / 32;
  const int top = nbins - 1 - pb * t;  // this lane's first (highest) bin
  const int nb = max(0, min(pb, top + 1));
  int sum = 0;
  for (int j = 0; j < nb; ++j) sum += static_cast<int>(hist[top - j]);
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  __syncwarp();
  if (incl >= want && incl - sum < want) {
    int r = want - (incl - sum), j = 0;
    for (;; ++j) {
      const int c = static_cast<int>(hist[top - j]);
      if (r <= c || j == nb - 1) {
        ss.bin = top - j;
        ss.rem = r;
        ss.cnt = c;
        break;
      }
      r -= c;
    }
  }
  __syncwarp();
  const int bin = ss.bin;
  rem = ss.rem;
  cnt = ss.cnt;
  __syncwarp();  // read before the caller's next histogram
  return bin;
}

// f(i, v) on each score of a lane's share [b0, b1) (a multiple of 4 wide, so
// its 16-byte loads stay aligned). kSweep scores in flight a lane: a sweep of
// a row of n keys costs n / (32 * kSweep) L2 round trips. kSweep == 32: a
// sweep step's kept keys are one bitmap word.
constexpr int kSweep = 32;
template <typename F>
__device__ __forceinline__ void sel_sweep(const float* sc, int b0, int b1, F&& f) {
  int i = b0;
  for (; i + kSweep <= b1; i += kSweep) {
    float4 v[kSweep / 4];
#pragma unroll
    for (int j = 0; j < kSweep / 4; ++j) v[j] = *reinterpret_cast<const float4*>(sc + i + 4 * j);
#pragma unroll
    for (int j = 0; j < kSweep / 4; ++j) {
      f(i + 4 * j, v[j].x);
      f(i + 4 * j + 1, v[j].y);
      f(i + 4 * j + 2, v[j].z);
      f(i + 4 * j + 3, v[j].w);
    }
  }
  for (; i + 4 <= b1; i += 4) {
    const float4 a = *reinterpret_cast<const float4*>(sc + i);
    f(i, a.x);
    f(i + 1, a.y);
    f(i + 2, a.z);
    f(i + 3, a.w);
  }
  for (; i < b1; ++i) f(i, sc[i]);
}

// One warp's exact top-k of row r from its scores sc[0, n) (ids are
// positions); [lo, hi] spans their order keys.
//  1. The cut bin: a histogram of hb bins, by value over [lo, hi] for a row
//     of fewer than 8 * top_k keys; else by order key above a floor taken
//     from a 512-key sample (the bins then are fine at the cut). A crowded
//     cut bin is narrowed by further sweeps.
//  2. One sweep: keys above the cut bin into the lane's own bitmap words, the
//     bin's keys into a list (at most kResList; a larger bin is a single key
//     value, read from the row instead).
//  3. The k-th key from the list (a direct rank for <= 32 entries, else
//     histogram rounds), and the id cut when only some of the keys equal to
//     it are kept (ties keep the larger ids); the kept ones into the bitmap.
//  4. The writes in ascending id order from the bitmap.
// kBatch: list entries whose loads go out together. H32's inlined selectors
// take 1: at their 104 registers a batch made the other stages slower.
// kStageIdx: stage dense indices-only writes in shared memory.
template <int kBatch = 8, bool kStageIdx = false>
__device__ __forceinline__ void sel_row(const ScoreParams& p, int r, const float* sc, int n,
                                        uint32_t lo, uint32_t hi, uint32_t* list_k, int32_t* list_i,
                                        uint32_t* bm, uint32_t* hist, int hbits, int t,
                                        SelShared& ss) {
  constexpr int kT = 32;
  const int hb = 1 << hbits;
  // The list's rounds take at most 256 bins: zeroing and ranking a larger
  // histogram costs more than an extra round.
  const int rbits = hbits > 8 ? 8 : hbits;
  const int rhb = 1 << rbits;
  const int K = p.top_k;
  int32_t* out_i = p.fin_p.indices + static_cast<int64_t>(r) * K;
  float* out_s = score_row(p.fin_p.scores, static_cast<int64_t>(r) * K);
  const int want = min(K, max(n, 0));
  // Multiples of 32: each lane owns whole words of the kept bitmap.
  const int per = ((n + kT - 1) / kT + 31) & ~31;
  const int b0 = min(n, t * per), b1 = min(n, b0 + per);
  uint32_t kth = 0u, id_cut = 0u;
  bool all_ties = true;
  if (n <= K) {
    // Every key.
    if (out_s == nullptr) {
      for (int i = t; i < n; i += kT) out_i[i] = i;
    } else {
      for (int i = t; i < n; i += kT) {
        out_i[i] = i;
        out_s[i] = sc[i];
      }
    }
  } else {
    // A long row (at least 8 * want keys) histograms the order keys in
    // [flo, hi], flo near rank 2 * want of a 512-key sample (two 8-bit radix
    // rounds): the bins are fine at the cut and keys below flo cost no atomic.
    // If fewer than want keys reach flo, the whole range.
    uint32_t flo = lo;
    int rem = 0, cnt = 0, shift = 0;
    uint32_t blo = lo, bhi = hi;
    // A shorter row bins by value over [lo, hi]: order keys bin by exponent,
    // which crowds a short row's keys into few bins. The value bin is
    // monotone in the score, so the list sweep classifies by the same bin and
    // the rounds below take the listed keys' order-key range. A crowded value
    // bin (over hb keys) falls back to the order-key passes.
    const float vlo = order_value(lo);
    const float vscale = __fdiv_rn(static_cast<float>(hb), __fsub_rn(order_value(hi), vlo));
    auto vbin = [&](float v) {
      return min(hb - 1, max(0, __float2int_rz(__fmul_rn(__fsub_rn(v, vlo), vscale))));
    };
    int vcut = -1;  // the value-binned cut bin
    if (n < 8 * want && hi > lo && vscale <= 3.0e38f) {
      for (int b = t; b < hb; b += kT) hist[b] = 0u;
      if (t == 0) ss.m = 0;
      __syncwarp();
      sel_sweep(sc, b0, b1, [&](int, float v) { atomicAdd(hist + vbin(v), 1u); });
      __syncwarp();
      const int b = sel_rank(hist, want, t, ss, rem, cnt, hb);
      if (cnt <= hb) vcut = b;
    }
    constexpr int kSamp = 512;
    if (n >= 8 * want && n >= kSamp) {
      const int stride = n / kSamp;
      uint32_t so[kSamp / 32];
#pragma unroll
      for (int j = 0; j < kSamp / 32; ++j) so[j] = score_order(sc[(j * 32 + t) * stride]);
      int r = (2 * want * kSamp + n - 1) / n;  // the sample rank from the top
      uint32_t pre = 0u;
      for (int sh = 24; sh >= 16; sh -= 8) {
        for (int b = t; b < 256; b += kT) hist[b] = 0u;
        __syncwarp();
#pragma unroll
        for (int j = 0; j < kSamp / 32; ++j)
          if (sh == 24 || (so[j] >> 24) == pre) atomicAdd(hist + ((so[j] >> sh) & 255u), 1u);
        __syncwarp();
        int rem_, cnt_;
        const int b = sel_rank(hist, r, t, ss, rem_, cnt_, 256);
        r = rem_;
        pre = sh == 24 ? static_cast<uint32_t>(b) : (pre << 8) | static_cast<uint32_t>(b);
      }
      const uint32_t g = pre << 16;
      if (g > lo && g <= hi) flo = g;
    }
    if (vcut < 0) {
      int total;
      for (;;) {
        shift = 0;
        while (((hi - flo) >> shift) >= static_cast<uint32_t>(hb)) ++shift;
        for (int b = t; b < hb; b += kT) hist[b] = 0u;
        if (t == 0) ss.m = 0;
        __syncwarp();
        int c = 0;
        sel_sweep(sc, b0, b1, [&](int, float v) {
          const uint32_t o = score_order(v);
          if (o >= flo && o <= hi) {
            atomicAdd(hist + ((o - flo) >> shift), 1u);
            ++c;
          }
        });
        __syncwarp();
        total = __reduce_add_sync(0xffffffffu, c);
        if (total >= want || flo == lo) break;
        flo = lo;  // the sample put flo above the k-th: the whole range
      }
      const int bin = sel_rank(hist, want, t, ss, rem, cnt, hb);
      blo = flo + (static_cast<uint32_t>(bin) << shift);
      unsigned long long bend = static_cast<unsigned long long>(blo) + (1ull << shift) - 1ull;
      bhi = bend < hi ? static_cast<uint32_t>(bend) : hi;
      // A crowded cut bin: narrow it by another sweep before listing it.
      while (cnt > hb && shift > 0) {
        const int s2 = shift > hbits ? shift - hbits : 0;
        for (int b = t; b < hb; b += kT) hist[b] = 0u;
        __syncwarp();
        sel_sweep(sc, b0, b1, [&](int, float v) {
          const uint32_t o = score_order(v);
          if (o >= blo && o <= bhi) atomicAdd(hist + ((o - blo) >> s2), 1u);
        });
        __syncwarp();
        const int b2 = sel_rank(hist, rem, t, ss, rem, cnt, hb);
        blo += static_cast<uint32_t>(b2) << s2;
        bend = static_cast<unsigned long long>(blo) + (1ull << s2) - 1ull;
        bhi = bend < bhi ? static_cast<uint32_t>(bend) : bhi;
        shift = s2;
      }
    }
    // The cut bin goes to the list (narrowed to at most hb <= kResList keys)
    // unless it is a single key value with more: then the rounds below read
    // the row's keys equal to it instead.
    const bool listed = cnt <= kResList;
    // Each cut-bin key (order key, id) once, from the list or the row.
    auto each_cut = [&](auto&& f) {
      if (listed) {
        const int m_ = ss.m;
        for (int j0 = t; j0 < m_; j0 += kBatch * kT) {
          uint32_t k_[kBatch];
          int32_t i_[kBatch];
#pragma unroll
          for (int u = 0; u < kBatch; ++u) {
            const int j = j0 + u * kT;
            k_[u] = j < m_ ? list_k[j] : 0u;
            i_[u] = j < m_ ? list_i[j] : 0;
          }
#pragma unroll
          for (int u = 0; u < kBatch; ++u)
            if (j0 + u * kT < m_) f(k_[u], i_[u]);
        }
      } else {
        sel_sweep(sc, b0, b1, [&](int i, float v) {
          const uint32_t o = score_order(v);
          if (o == blo) f(o, i);
        });
      }
    };
    // Each lane its own range: one list reservation per sweep step.
    {
      for (int i = b0; i < b1; i += kSweep) {
        float v[kSweep];
        if (i + kSweep <= b1) {
#pragma unroll
          for (int j = 0; j < kSweep / 4; ++j) {
            const float4 a = *reinterpret_cast<const float4*>(sc + i + 4 * j);
            v[4 * j] = a.x, v[4 * j + 1] = a.y, v[4 * j + 2] = a.z, v[4 * j + 3] = a.w;
          }
        } else {
#pragma unroll
          for (int j = 0; j < kSweep; ++j) v[j] = i + j < b1 ? sc[i + j] : -INFINITY;
        }
        uint32_t inbin = 0u, up = 0u;
        if (vcut >= 0) {
#pragma unroll
          for (int j = 0; j < kSweep; ++j) {
            const int bb = vbin(v[j]);
            const bool ok = i + j < b1;
            up |= (ok && bb > vcut ? 1u : 0u) << j;
            inbin |= (ok && bb == vcut ? 1u : 0u) << j;
          }
        } else {
#pragma unroll
          for (int j = 0; j < kSweep; ++j) {
            const uint32_t o = score_order(v[j]);
            const bool ok = i + j < b1;
            up |= (ok && o > bhi ? 1u : 0u) << j;
            inbin |= (ok && o >= blo && o <= bhi ? 1u : 0u) << j;
          }
        }
        bm[i >> 5] = up;
        if (listed && inbin != 0u) {
          int at = atomicAdd(&ss.m, __popc(inbin));
#pragma unroll
          for (int j = 0; j < kSweep; ++j) {  // static indices: v stays in registers
            if ((inbin >> j) & 1u) {
              list_k[at] = score_order(v[j]);
              list_i[at] = i + j;
              ++at;
            }
          }
        }
      }
    }
    __syncwarp();
    const int m = ss.m;
    if (kBatch == 8 && listed && m <= 32) {
      // The common case, a few listed keys: each ranked directly against the
      // others (ties: the larger id first), the rem best into the bitmap.
      // Not in H32's inlined selectors (kBatch 1), where its code slowed the
      // other stages.
      const uint32_t mk = t < m ? list_k[t] : 0u;
      const int mi = t < m ? list_i[t] : -1;
      int rank = 0;
      for (int j = 0; j < m; ++j) {
        const uint32_t kj = __shfl_sync(0xffffffffu, mk, j);
        const int ij = __shfl_sync(0xffffffffu, mi, j);
        rank += kj > mk || (kj == mk && ij > mi);
      }
      if (t < m && rank < rem) atomicOr(bm + (mi >> 5), 1u << (mi & 31));
    } else {
      if (vcut >= 0) {
        // The listed cut bin's order-key range for the rounds below.
        uint32_t mlo = 0xffffffffu, mhi = 0u;
        for (int j = t; j < m; j += kT) {
          const uint32_t o = list_k[j];
          mlo = min(mlo, o);
          mhi = max(mhi, o);
        }
        blo = __reduce_min_sync(0xffffffffu, mlo);
        bhi = __reduce_max_sync(0xffffffffu, mhi);
        shift = 0;
        while ((static_cast<uint64_t>(bhi - blo) >> shift) != 0u) ++shift;
      }
      // The k-th within the list: rank rem among keys in [blo, bhi].
      uint32_t cur = blo;
      int sh = shift;
      while (sh > 0) {
        const int s2 = sh > rbits ? sh - rbits : 0;
        const unsigned long long tend = static_cast<unsigned long long>(cur) + (1ull << sh) - 1ull;
        const uint32_t top = tend < bhi ? static_cast<uint32_t>(tend) : bhi;
        for (int b = t; b < rhb; b += kT) hist[b] = 0u;
        __syncwarp();
        for (int j0 = t; j0 < m; j0 += kBatch * kT) {
          uint32_t k_[kBatch];
#pragma unroll
          for (int u = 0; u < kBatch; ++u) k_[u] = j0 + u * kT < m ? list_k[j0 + u * kT] : 0u;
#pragma unroll
          for (int u = 0; u < kBatch; ++u) {
            const uint32_t o = k_[u];
            if (j0 + u * kT < m && o >= cur && o <= top) atomicAdd(hist + ((o - cur) >> s2), 1u);
          }
        }
        __syncwarp();
        const int b2 = sel_rank(hist, rem, t, ss, rem, cnt, rhb);
        cur += static_cast<uint32_t>(b2) << s2;
        sh = s2;
      }
      kth = cur;
      if (rem < cnt) {
        // The rem largest ids among the list's keys equal to kth (ids < 2^22).
        all_ties = false;
        uint32_t ipre = 0u, ihi = 0u;
        for (int s3 = 22; s3 > 0;) {
          s3 = s3 > rbits ? s3 - rbits : 0;
          for (int b = t; b < rhb; b += kT) hist[b] = 0u;
          __syncwarp();
          each_cut([&](uint32_t o, int i) {
            const uint32_t id = static_cast<uint32_t>(i);
            if (o == kth && (id & ihi) == ipre)
              atomicAdd(hist + ((id >> s3) & static_cast<uint32_t>(rhb - 1)), 1u);
          });
          __syncwarp();
          const int b3 = sel_rank(hist, rem, t, ss, rem, cnt, rhb);
          ipre |= static_cast<uint32_t>(b3) << s3;
          ihi |= static_cast<uint32_t>(rhb - 1) << s3;
        }
        id_cut = ipre;
      }
      // The cut bin's kept keys into the bitmap (their owners' words).
      each_cut([&](uint32_t o, int id) {
        if (o > kth || (o == kth && (all_ties || static_cast<uint32_t>(id) >= id_cut)))
          atomicOr(bm + (id >> 5), 1u << (id & 31));
      });
    }
    __syncwarp();
    // The writes: the lanes interleaved over the bitmap words, so stores of
    // neighbouring kept keys coalesce.
    const int lane = t;
    const int w_lo = 0, w_hi = (n + 31) >> 5;
    int pos = 0;
    if (out_s == nullptr && 16 * want <= n) {
      // A sparse row, indices only: each lane writes the set bits of its own
      // words after one scan for the offsets, instead of walking every word
      // 32 at a time. (A lane past the row's end owns no word: b0 == b1 == n
      // would round to the last, partial word, its owner's.)
      const int x0 = b0 >> 5, x1 = b0 < b1 ? (b1 + 31) >> 5 : x0;
      int c = 0;
      for (int x = x0; x < x1; ++x) c += __popc(bm[x]);
      int incl = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      int at = incl - c;
      for (int x = x0; x < x1; ++x) {
        uint32_t bits = bm[x];
        while (bits != 0u) {
          out_i[at++] = 32 * x + __ffs(bits) - 1;
          bits &= bits - 1u;
        }
      }
    } else {
      // Staged writes for a dense row: with scores when a quarter or more of
      // its keys are kept; indices only, when half or more and kStageIdx.
      const bool staged = out_s != nullptr ? 4 * want >= n : kStageIdx && 2 * want >= n;
      for (int x0 = w_lo; x0 < w_hi; x0 += 32) {
        const int x = x0 + lane;
        const uint32_t bits = x < w_hi ? bm[x] : 0u;
        const int c = __popc(bits);
        int incl = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int y = __shfl_up_sync(0xffffffffu, incl, o);
          if (lane >= o) incl += y;
        }
        int at = pos + incl - c;
        const int end = pos + __shfl_sync(0xffffffffu, incl, 31);
        if (staged) {
          // A lane's kept keys are consecutive slots, so a direct store
          // instruction writes 32 pieces 128 bytes apart. Staged in the
          // finished histogram's space a half-warp at a time (padded a word
          // per 32 against bank conflicts), then stored whole.
          float v[32];
          if (out_s != nullptr) {
#pragma unroll
            for (int j = 0; j < 32; ++j) v[j] = (bits >> j) & 1u ? sc[32 * x + j] : 0.0f;
          }
          uint32_t* const st = hist;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int h0 = __shfl_sync(0xffffffffu, at, 16 * h);
            const int h1 = h == 0 ? __shfl_sync(0xffffffffu, at, 16) : end;
            if (h1 == h0) continue;
            const bool mine = (lane >> 4) == h;
            if (mine) {
              int l = at - h0;
#pragma unroll
              for (int j = 0; j < 32; ++j)
                if ((bits >> j) & 1u) {
                  st[l + (l >> 5)] = static_cast<uint32_t>(32 * x + j);
                  ++l;
                }
            }
            __syncwarp();
            for (int i = lane; i < h1 - h0; i += 32)
              out_i[h0 + i] = static_cast<int32_t>(st[i + (i >> 5)]);
            __syncwarp();
            if (out_s != nullptr) {
              if (mine) {
                int l = at - h0;
#pragma unroll
                for (int j = 0; j < 32; ++j)
                  if ((bits >> j) & 1u) {
                    st[l + (l >> 5)] = __float_as_uint(v[j]);
                    ++l;
                  }
              }
              __syncwarp();
              for (int i = lane; i < h1 - h0; i += 32)
                out_s[h0 + i] = __uint_as_float(st[i + (i >> 5)]);
              __syncwarp();
            }
          }
          pos = end;
          continue;
        }
        pos = end;
        if (bits == 0u) continue;
        if (out_s == nullptr) {  // not a check per store: it slows the scored form
#pragma unroll
          for (int j = 0; j < 32; ++j)
            if ((bits >> j) & 1u) out_i[at++] = 32 * x + j;
          continue;
        }
        float v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) v[j] = (bits >> j) & 1u ? sc[32 * x + j] : 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          if ((bits >> j) & 1u) {
            out_i[at] = 32 * x + j;
            out_s[at] = v[j];
            ++at;
          }
        }
      }
    }
  }
  if (out_s == nullptr) {
    for (int j = want + t; j < K; j += kT) out_i[j] = -1;
  } else {
    for (int j = want + t; j < K; j += kT) {
      out_i[j] = -1;
      out_s[j] = -INFINITY;
    }
  }
  __syncwarp();  // the shared state and the list are reused by the next row
}

// RESIDENT: the math warps' selection as a call (the selectors inline theirs).
template <bool kStageIdx>
__device__ __noinline__ void sel_row_call(const ScoreParams& p, int r, const float* sc, int n,
                                          uint32_t lo, uint32_t hi, uint32_t* list_k,
                                          int32_t* list_i, uint32_t* bm, uint32_t* hist, int hbits,
                                          int t, SelShared& ss) {
  sel_row<8, kStageIdx>(p, r, sc, n, lo, hi, list_k, list_i, bm, hist, hbits, t, ss);
}

template <int kNumHeads, Mode kMode, bool kFp4, int kResSlotsT = kResSlots>
__global__ void __launch_bounds__(kNumWsThreads, 1)
    score_kernel(const __grid_constant__ ScoreParams p, const __grid_constant__ CUtensorMap tmap_q,
                 const __grid_constant__ CUtensorMap tmap_w,
                 const __grid_constant__ CUtensorMap tmap_k,
                 const __grid_constant__ CUtensorMap tmap_ks,
                 const __grid_constant__ CUtensorMap tmap_qsf) {
  // EXACT runs its phases in one cooperative launch.
  constexpr int kNumPhases = kMode == Mode::kExact ? kExactPhases : 1;
  pdl_enter();
  // No listed row (the common case): every CTA leaves before any setup.
  if (p.num_unit_rows != nullptr && *p.num_unit_rows == 0) return;
  // RESIDENT: no row of this launch's class (the launches are fixed by the
  // shapes; the lengths are known only here): every CTA leaves at once.
  if constexpr (kMode == Mode::kResident) {
    if (p.row_classes != nullptr) {
      bool mine = false;
      for (int i = threadIdx.x; i < p.res_class_ctas; i += blockDim.x)
        mine |= (p.row_classes[i] >> p.res_class) & 1u;
      if (!__syncthreads_or(mine)) return;
    }
  }
  if constexpr (kMode == Mode::kSample) {
    if (p.fuse_tail && blockIdx.x == 0) tau_setup(p.tau_p, true, false);
  }
  using L = ScoreSmem<kNumHeads, kFp4>;
  using F = Fmt<kFp4>;
  constexpr int kKSteps = F::kKSteps;
  constexpr int kBlockQ = L::kBlockQ;
  constexpr int kNTiles = kNumHeads / 8;

  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::kBarOff);
  uint64_t* full_q = bars;
  uint64_t* empty_q = bars + kNumQStages;
  uint64_t* full_k = bars + 2 * kNumQStages;
  uint64_t* empty_k = bars + 2 * kNumQStages + L::kKStages;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // RESIDENT: per slot, scores ready (math -> selectors) and slot free
  // (selectors -> math), and each row's order-key range.
  constexpr int kSlots = kResSlotsT;
  constexpr bool kResLo = kResSlotsT != kResSlots;  // a longer class: skip the shorter's blocks
  __shared__ uint64_t res_ready[kSlots], res_free[kSlots];
  __shared__ int res_claim;         // rows claimed so far
  __shared__ int res_iters;         // the CTA's blocks (selector 0 counts them)
  __shared__ int res_row0[kSlots];  // each slot's block
  __shared__ SelShared res_ss[kSelClaimers];
  __shared__ uint32_t res_rng[kSlots][kBlockQ][2];
  if (warp == kNumMathThreads / 32 && lane == 0) {
    if constexpr (kMode == Mode::kResident) {
      res_claim = 0;
      res_iters = -1;
      for (int i = 0; i < kSlots; ++i) {
        mbar_init(res_ready + i, 1);
        mbar_init(res_free + i, kBlockQ);  // a row each
      }
    }
    prefetch_tmap(&tmap_q);
    prefetch_tmap(&tmap_w);
    prefetch_tmap(&tmap_k);
    prefetch_tmap(&tmap_ks);
    if constexpr (kFp4) prefetch_tmap(&tmap_qsf);
    for (int i = 0; i < kNumQStages; ++i) {
      mbar_init(full_q + i, 1);
      mbar_init(empty_q + i, kNumMathThreads);
    }
    for (int i = 0; i < L::kKStages; ++i) {
      mbar_init(full_k + i, 1);
      mbar_init(empty_k + i, kNumMathThreads);
    }
    fence_barrier_init();
  }
  __syncthreads();

  // RESIDENT: claimer warp c takes the CTA's rows in order from res_claim
  // (a block's rows once its scores are ready).
  auto res_claims = [&](int c, auto&& select) {
    if constexpr (kMode == Mode::kResident) {
      // The CTA's blocks, as WorkIter walks them: counted by the selectors
      // while they wait for the first block. The math warps read selector 0's
      // count after their scan, which keeps the count out of the scan's code.
      // The order and the tests must match WorkIter's, or the handshake on
      // res_ready / res_free never completes.
      int iters = 0;
      if (c < kSelWarps) {
        if constexpr (kBlockQ == 4) {
          for (int pos = blockIdx.x + lane * gridDim.x;
               pos - static_cast<int>(blockIdx.x) < p.num_units; pos += 32 * gridDim.x) {
            const int u = res_unit<true>(pos, p.num_units);
            iters += u >= 0 &&
                     (p.res_cap <= 0 ||
                      short_block(p.row_ks, p.row_ke, p.num_rows, kBlockQ, u, p.res_cap)) &&
                     !(kResLo && short_block(p.row_ks, p.row_ke, p.num_rows, kBlockQ, u, p.res_lo));
          }
        } else {
          for (int u = blockIdx.x + lane * gridDim.x; u < p.num_units; u += 32 * gridDim.x)
            iters += (p.res_cap <= 0 || short_block(p.row_ks, p.row_ke, p.num_rows, kBlockQ,
                                                    p.num_units - 1 - u, p.res_cap)) &&
                     !(kResLo && short_block(p.row_ks, p.row_ke, p.num_rows, kBlockQ,
                                             p.num_units - 1 - u, p.res_lo));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) iters += __shfl_xor_sync(0xffffffffu, iters, o);
        if (c == 0 && lane == 0) *reinterpret_cast<volatile int*>(&res_iters) = iters;
      } else {
        // Selector 0 wrote the count long before (it is the scan's length):
        // a volatile read in a loop suffices.
        while ((iters = *reinterpret_cast<volatile int*>(&res_iters)) < 0) {
        }
      }
      // Selectors: their own histograms; math warps: the pipeline's buffers.
      const bool sel = c < kSelWarps;
      uint32_t* hist = reinterpret_cast<uint32_t*>(
          smem + (sel ? L::sel_hist_off(c) : (c - kSelWarps) * L::kMathHB));
      // First-pass bins: 1024 (fewer narrowing sweeps); 256 for H64 MXFP4,
      // whose selectors (88 registers, beside the busiest scan) lose more to
      // zeroing and ranking a larger histogram.
      constexpr int kShortBits = kNumHeads == 64 && kFp4 ? 8 : 10;
      const int hbits = min(sel ? L::kSelHBits : L::kMathHBits, kShortBits);
      float* const cta_base = p.res_scratch + static_cast<int64_t>(blockIdx.x) * p.res_cta_words;
      uint32_t* const list_k = reinterpret_cast<uint32_t*>(cta_base) +
                               kSlots * kBlockQ * p.res_stride +
                               c * res_claimer_words(p.res_stride);
      int32_t* const list_i = reinterpret_cast<int32_t*>(list_k + kResList);
      uint32_t* const bm = list_k + 2 * kResList;
      for (;;) {
        int g = 0;
        if (lane == 0) g = atomicAdd(&res_claim, 1);
        g = __shfl_sync(0xffffffffu, g, 0);
        const int it = g / kBlockQ, qi = g - it * kBlockQ;
        if (it >= iters) break;
        const int slot = it % kSlots;
        mbar_wait_sleep(res_ready + slot, (it / kSlots) & 1);
        const int r = res_row0[slot] + qi;
        if (r < p.num_rows) {
          const int n = __ldg(p.row_ke + r) - __ldg(p.row_ks + r);
          select(r, cta_base + (slot * kBlockQ + qi) * p.res_stride, n, res_rng[slot][qi][0],
                 res_rng[slot][qi][1], list_k, list_i, bm, hist, hbits, lane, res_ss[c]);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(res_free + slot);
      }
    }
  };

  // SAMPLE and SCAN move registers from the producer warpgroup to the math
  // warps (40 / 232 around the launch's 168: 128 * (168 - 40) == 256 * (232 -
  // 168)). Each role ends back at 168, the math warps' release before the
  // producers' claim, so the fused tail and its calls run at the launch
  // count. ptxas keeps setmaxnreg in a kernel with calls only under this knob
  // (with -Xptxas -uumn). EXACT calls exact_unit_done from the math loop and
  // keeps the uniform count. RESIDENT rebalances for MXFP4 only (its query
  // fragments then stay in registers; FP8 gains nothing), leaving the
  // selector warps room for sel_row: H32's inlined selectors spill below 104,
  // and H64's scan wants 208.
  constexpr bool kRebalance =
      kMode == Mode::kSample || kMode == Mode::kScan || (kMode == Mode::kResident && kFp4);
  constexpr int kDecRegs = kMode != Mode::kResident ? 40 : kNumHeads == 32 ? 104 : 88;
  constexpr int kIncRegs = kMode != Mode::kResident ? 232 : kNumHeads == 32 ? 200 : 208;
  static_assert((kIncRegs - 168) * kNumMathThreads == (168 - kDecRegs) * 128, "setmaxnreg budget");
  if constexpr (kRebalance)
    asm volatile(".pragma \"global knob CtaReconfigOptInSeparateComp=1\";\n");
  if (kMode == Mode::kResident && warp >= kSelWarp0) {
    // ---------------- RESIDENT selectors ----------------
    if constexpr (kMode == Mode::kResident) {
      const int sw = warp - kSelWarp0;
      if constexpr (kRebalance) setmaxnreg_dec<kDecRegs>();
      res_claims(sw, [&](int r, const float* sc, int n, uint32_t lo, uint32_t hi, uint32_t* lk,
                         int32_t* li, uint32_t* b, uint32_t* h, int hbits, int t_, SelShared& s_) {
        sel_row<kNumHeads == 64 ? 8 : 1>(p, r, sc, n, lo, hi, lk, li, b, h, hbits, t_, s_);
      });
      if constexpr (kRebalance) {
        __syncthreads();
        setmaxnreg_inc<168>();
      }
    }
  } else if (tid >= kNumMathThreads) {
    // ---------------- producer warpgroup ----------------
    if constexpr (kRebalance) setmaxnreg_dec<kDecRegs>();
    const bool issuer = warp == kNumMathThreads / 32 && lane == 0;
    Ring<kNumQStages> qr;
    Ring<L::kKStages> kr;
    for (int phase = 0; phase < kNumPhases; ++phase) {
      if constexpr (kMode == Mode::kExact) {
        if (*p.num_unit_rows == 0) break;  // no flagged row: every CTA leaves at once
      }
      if (issuer) {
        WorkIter<kBlockQ, kMode, kResLo> work(p);
        int row0, only_qi, base, ntiles_total, tile_begin, ntiles;
        for (; work.next(p, phase, row0, only_qi, base, ntiles_total, tile_begin, ntiles);
             qr.next()) {
          const int qs = qr.st;
          mbar_wait_sleep(empty_q + qs, qr.parity ^ 1u);
          mbar_expect_tx(full_q + qs, L::kQBytes + kBlockQ * kNumHeads * 4 + L::kQsfBytes);
          tma_load_2d(smem + L::kQOff + qs * L::kQBytes, &tmap_q, full_q + qs, 0, row0 * kNumHeads);
          tma_load_2d(smem + L::kWOff + qs * kBlockQ * kNumHeads * 4, &tmap_w, full_q + qs, 0,
                      row0);
          if constexpr (kFp4)
            tma_load_2d(smem + L::kQsfOff + qs * L::kQsfBytes, &tmap_qsf, full_q + qs, 0, row0);
          const int nsamp = kMode == Mode::kSample ? sample_count(p, ntiles_total) : 0;
          for (int i = 0; i < ntiles; ++i, kr.next()) {
            const int st = kr.st;
            mbar_wait_sleep(empty_k + st, kr.parity ^ 1u);
            if constexpr (kMode == Mode::kSample) {
              // 16-key boxes (the map's box height is kChunk in this mode); the
              // math warps read the chunks' scales (64 B: below TMA's alignment).
              mbar_expect_tx(full_k + st, L::kKBytes);
              for (int c = 0; c < kChunksPerTile; ++c) {
                const int key0 = base + chunk_of(tile_begin + i, c, ntiles_total, nsamp) * kChunk;
                uint8_t* dst = smem + L::kKOff + st * L::kKBytes + c * kChunk * F::kRow;
                tma_load_2d(dst, &tmap_k, full_k + st, 0, key0);
              }
            } else {
              mbar_expect_tx(full_k + st, L::kKBytes + kBlockKV * 4);
              const int key0 = base + (tile_begin + i) * kBlockKV;
              tma_load_2d(smem + L::kKOff + st * L::kKBytes, &tmap_k, full_k + st, 0, key0);
              tma_load_2d(smem + L::kScaleOff + st * kBlockKV * 4, &tmap_ks, full_k + st, key0, 0);
            }
          }
        }
      }  // issuer
      if constexpr (kMode == Mode::kExact) grid_barrier();  // the phase's picks are done
    }  // phase
    if constexpr (kRebalance) {
      __syncthreads();
      setmaxnreg_inc<168>();
    }
  } else {
    // ---------------- math warpgroups ----------------
    // Hot parameters in registers.
    const int num_rows_ = p.num_rows, capacity_ = p.capacity;
    float* const cand_score_ = p.cand_score;
    int32_t* const cand_id_ = p.cand_id;
    int32_t* const count_ = p.count;
    if constexpr (kRebalance) setmaxnreg_inc<kIncRegs>();
    const int g = lane >> 2, t4 = lane & 3;
    const int a_row = warp * 16 + (lane & 15);  // ldmatrix x4: rows 0-15
    const int a_chunk_hi = lane >> 4;           // matrices 2,3 take k bytes 16..31
    const int b_row = lane & 7;                 // ldmatrix x2: 8 head rows
    const int b_chunk_hi = (lane >> 3) & 1;
    int a_off[kKSteps], b_off[kKSteps];  // per K-step, see Fmt::lane_offsets
    F::lane_offsets(a_row, a_chunk_hi, a_off);
    F::lane_offsets(b_row, b_chunk_hi, b_off);

    uint32_t* shist = reinterpret_cast<uint32_t*>(smem + L::kHistOff);
    if constexpr (kMode == Mode::kExact) {
      for (int b = tid; b < kExactBins; b += kNumMathThreads) shist[b] = 0;
      math_barrier();
    }

    Ring<kNumQStages> qr;
    Ring<L::kKStages> kr;
    for (int phase = 0; phase < kNumPhases; ++phase) {
      if constexpr (kMode == Mode::kExact) {
        if (*p.num_unit_rows == 0) break;
      }
      WorkIter<kBlockQ, kMode, kResLo> work(p);
      int row0, only_qi, base, ntiles_total, tile_begin, ntiles;
      int res_it = 0;  // RESIDENT: the block's scratch slot is res_it % kSlots
      for (; work.next(p, phase, row0, only_qi, base, ntiles_total, tile_begin, ntiles);
           qr.next()) {
        if constexpr (kMode == Mode::kResident)
          mbar_wait(res_free + res_it % kSlots,
                    ((res_it / kSlots) & 1) ^ 1u);  // its selection done
        int rks[kBlockQ], rke[kBlockQ];
        float rtau[kBlockQ];
#pragma unroll
        for (int qi = 0; qi < kBlockQ; ++qi) {
          const int r = row0 + qi;
          const bool ok = r < num_rows_ && (only_qi < 0 || qi == only_qi);
          rks[qi] = ok ? __ldg(p.row_ks + r) : 0;
          rke[qi] = ok ? __ldg(p.row_ke + r) : 0;
          rtau[qi] = kMode == Mode::kScan && ok ? __ldg(p.tau + r) : INFINITY;
        }
        // EXACT: this phase's view of the listed row's state (stable during the
        // launch; the previous launch's last pick wrote it).
        int x_shift = 0;
        uint32_t x_bin_mask = 0, x_hi_mask = 0, x_prefix = 0, x_kth = 0, x_id_cut = 0;
        bool x_ties_all = true, x_active = false;
        if constexpr (kMode == Mode::kExact) {
          const ExactState& st = p.xs[row0 + only_qi];
          x_active = phase < 3 || phase == 6 || st.need_ids;
          exact_pass(phase, x_shift, x_bin_mask, x_hi_mask);
          x_prefix = st.prefix;
          x_kth = st.kth;
          x_id_cut = st.id_cut;
          x_ties_all = !st.need_ids;
        }
        if constexpr (kMode == Mode::kSample) {
          // Only the first ntiles * kBlockKV slots of each row hold samples.
          if (tile_begin == 0 && tid < kBlockQ && row0 + tid < num_rows_)
            p.sample_slots[row0 + tid] = sample_count(p, ntiles_total) * kBlockKV;
        }
        const int qs = qr.st;
        mbar_wait(full_q + qs, qr.parity);
        const uint8_t* sq = smem + L::kQOff + qs * L::kQBytes;
        const float* sw =
            reinterpret_cast<const float*>(smem + L::kWOff + qs * kBlockQ * kNumHeads * 4);
        const uint32_t* sqsf =
            reinterpret_cast<const uint32_t*>(smem + L::kQsfOff + qs * L::kQsfBytes);
        // FP4 under the rebalance: the block's B fragments and their scales
        // stay in registers for all its tiles instead of an ldmatrix per MMA.
        constexpr bool kQRegs = kRebalance && kFp4;
        constexpr int kQR = kQRegs ? kBlockQ : 1, kQK = kQRegs ? kKSteps : 1,
                      kQN = kQRegs ? kNTiles : 1;
        uint32_t qf[kQR][kQK][kQN][2];
        uint32_t qsf_r[kQR][kQN];
        if constexpr (kQRegs) {
#pragma unroll
          for (int qi = 0; qi < kQR; ++qi)
#pragma unroll
            for (int nt = 0; nt < kQN; ++nt) {
#pragma unroll
              for (int ks = 0; ks < kQK; ++ks)
                ldmatrix_x2(qf[qi][ks][nt], sq + b_off[ks] + (qi * kNumHeads + nt * 8) * F::kRow);
              qsf_r[qi][nt] = sqsf[qi * kNumHeads + nt * 8 + g];
            }
        }

        // Exact scores of this warp's 16 keys (fragments kf) for query row qi:
        // DeepGEMM's MMA order and epilogue, so scores are bit-identical.
        // FP8: scale_0 / scale_1 are the keys' fp32 scales (multiplied after the
        // head sum). MXFP4: ksf is the scale word of this lane's A row (key
        // g + (t4 & 1) * 8); the scales enter the MMA and no multiply follows.
        auto exact_scores = [&](const uint32_t(&kf)[kKSteps][4], int qi, float scale_0,
                                float scale_1, uint32_t ksf, float& v_0, float& v_1) {
          // Independent accumulators per head tile interleave the MMA chains
          // (each still sums its K-steps 0..3 in order, as DeepGEMM does).
          float acc[kNTiles][4];
#pragma unroll
          for (int nt = 0; nt < kNTiles; ++nt)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
          for (int ks = 0; ks < kKSteps; ++ks) {
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
              uint32_t b[2];
              uint32_t qsf = 0u;
              if constexpr (kQRegs) {
                b[0] = qf[kQRegs ? qi : 0][kQRegs ? ks : 0][kQRegs ? nt : 0][0];
                b[1] = qf[kQRegs ? qi : 0][kQRegs ? ks : 0][kQRegs ? nt : 0][1];
                qsf = qsf_r[kQRegs ? qi : 0][kQRegs ? nt : 0];
              } else {
                ldmatrix_x2(b, sq + b_off[ks] + (qi * kNumHeads + nt * 8) * F::kRow);
                if constexpr (kFp4) qsf = sqsf[qi * kNumHeads + nt * 8 + g];
              }
              if constexpr (kFp4) {
                mma_fp4_step(acc[nt], kf[ks], b, ksf, qsf, ks);
              } else {
                mma_fp8(acc[nt], kf[ks], b);
              }
            }
          }
          float partial_0 = 0.0f, partial_1 = 0.0f;
#pragma unroll
          for (int nt = 0; nt < kNTiles; ++nt) {
            const float* d = acc[nt];
            const int h0 = nt * 8 + t4 * 2;
            const float w0 = sw[qi * kNumHeads + h0];
            const float w1 = sw[qi * kNumHeads + h0 + 1];
            partial_0 = __fadd_rn(
                partial_0, __fmaf_rn(fmaxf(d[0], 0.0f), w0, __fmul_rn(fmaxf(d[1], 0.0f), w1)));
            partial_1 = __fadd_rn(
                partial_1, __fmaf_rn(fmaxf(d[2], 0.0f), w0, __fmul_rn(fmaxf(d[3], 0.0f), w1)));
          }
          if constexpr (kFp4) {
            v_0 = partial_0;
            v_1 = partial_1;
          } else {
            v_0 = __fmul_rn(partial_0, scale_0);
            v_1 = __fmul_rn(partial_1, scale_1);
          }
          v_0 = __fadd_rn(v_0, __shfl_xor_sync(0xffffffffu, v_0, 1));
          v_1 = __fadd_rn(v_1, __shfl_xor_sync(0xffffffffu, v_1, 1));
          v_0 = __fadd_rn(v_0, __shfl_xor_sync(0xffffffffu, v_0, 2));
          v_1 = __fadd_rn(v_1, __shfl_xor_sync(0xffffffffu, v_1, 2));
        };

        // SCAN stages each warp's candidates per row in its own shared-memory
        // slice (no atomic, no barrier) and copies a slice to the row's
        // candidates when it has no room for another tile: one global atomic
        // per flush and coalesced stores, instead of an atomic per warp, row
        // and tile and scattered 4-byte stores.
        constexpr int kWarpCap = L::kWarpCap;
        float* ws_score = reinterpret_cast<float*>(smem + L::kHistOff) + warp * kBlockQ * kWarpCap;
        int32_t* ws_id = reinterpret_cast<int32_t*>(smem + L::kHistOff) +
                         (L::kSlices + warp * kBlockQ) * kWarpCap;
        int ws_n[kBlockQ];  // staged entries per row (warp-uniform)
        // RESIDENT: the range of each row's scores (this thread's share). H32
        // tracks floats and converts to order keys once a block, sparing a
        // score_order per score; H64 is faster with order keys.
        constexpr bool kFloatRange = kNumHeads == 32;
        using RangeT = std::conditional_t<kFloatRange, float, uint32_t>;
        RangeT rlo[kBlockQ], rhi[kBlockQ];
#pragma unroll
        for (int qi = 0; qi < kBlockQ; ++qi) {
          if constexpr (kFloatRange) {
            rlo[qi] = INFINITY;
            rhi[qi] = -INFINITY;
          } else {
            rlo[qi] = 0xffffffffu;
            rhi[qi] = 0u;
          }
        }
#pragma unroll
        for (int qi = 0; qi < kBlockQ; ++qi) ws_n[qi] = 0;
        auto flush_row = [&](int qi) {
          __syncwarp();
          int base = 0;
          if (lane == 0) base = atomicAdd(count_ + row0 + qi, ws_n[qi]);
          base = __shfl_sync(0xffffffffu, base, 0);
          const int64_t row_off = static_cast<int64_t>(row0 + qi) * capacity_;
          for (int j = lane; j < ws_n[qi] && base + j < capacity_; j += 32) {
            cand_score_[row_off + base + j] = ws_score[qi * kWarpCap + j];
            cand_id_[row_off + base + j] = ws_id[qi * kWarpCap + j];
          }
          __syncwarp();
          ws_n[qi] = 0;
        };
        const int nsamp = kMode == Mode::kSample ? sample_count(p, ntiles_total) : 0;
        for (int i = 0; i < ntiles; ++i, kr.next()) {
          const int st = kr.st;
          int key_base = 0;
          if constexpr (kMode == Mode::kSample)
            key_base = base + chunk_of(tile_begin + i, warp, ntiles_total, nsamp) * kChunk;
          float scale_0 = 1.0f, scale_1 = 1.0f;
          uint32_t ksf = 0;
          // SAMPLE reads its chunk's scales from global memory, issued before
          // the wait for the chunk's keys.
          auto scale_word = [&](int key) -> uint32_t {
            return key < p.num_keys ? __ldg(static_cast<const uint32_t*>(p.k_scale) + key) : 0u;
          };
          if constexpr (kMode == Mode::kSample) {
            if constexpr (kFp4) {
              ksf = scale_word(key_base + g + (t4 & 1) * 8);
            } else {
              scale_0 = __uint_as_float(scale_word(key_base + g));
              scale_1 = __uint_as_float(scale_word(key_base + g + 8));
            }
          }
          mbar_wait(full_k + st, kr.parity);
          const uint8_t* kt = smem + L::kKOff + st * L::kKBytes;
          const float* sks =
              reinterpret_cast<const float*>(smem + L::kScaleOff + st * kBlockKV * 4);
          if constexpr (kMode != Mode::kSample) {
            key_base = base + (tile_begin + i) * kBlockKV + warp * 16;
            if constexpr (kFp4) {
              ksf = reinterpret_cast<const uint32_t*>(sks)[warp * 16 + g + (t4 & 1) * 8];
            } else {
              scale_0 = sks[warp * 16 + g];
              scale_1 = sks[warp * 16 + g + 8];
            }
          }
          uint32_t kf[kKSteps][4];
#pragma unroll
          for (int ks = 0; ks < kKSteps; ++ks) ldmatrix_x4(kf[ks], kt + a_off[ks]);

          if constexpr (kMode == Mode::kSample) {
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) {
              float v_0, v_1;
              const int r = row0 + qi;
              const int key_0 = key_base + g, key_1 = key_base + g + 8;
              const bool in_0 = r < num_rows_ && key_0 >= rks[qi] && key_0 < rke[qi];
              const bool in_1 = r < num_rows_ && key_1 >= rks[qi] && key_1 < rke[qi];
              exact_scores(kf, qi, scale_0, scale_1, ksf, v_0, v_1);
              if (t4 == 0 && r < num_rows_) {
                float* dst = p.sample + static_cast<int64_t>(r) * kSampleSlots +
                             (tile_begin + i) * kBlockKV + warp * 16;
                dst[g] = in_0 ? v_0 : -INFINITY;
                dst[g + 8] = in_1 ? v_1 : -INFINITY;
              }
            }
          } else if constexpr (kMode == Mode::kResident) {
            const int key_0 = key_base + g, key_1 = key_base + g + 8;
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) {
              float v_0, v_1;
              exact_scores(kf, qi, scale_0, scale_1, ksf, v_0, v_1);
              const bool in_0 = key_0 >= rks[qi] && key_0 < rke[qi];
              const bool in_1 = key_1 >= rks[qi] && key_1 < rke[qi];
              if (t4 == 0) {
                float* dst = p.res_scratch + static_cast<int64_t>(blockIdx.x) * p.res_cta_words +
                             (res_it % kSlots * kBlockQ + qi) * p.res_stride;
                if (in_0) dst[key_0 - rks[qi]] = v_0;
                if (in_1) dst[key_1 - rks[qi]] = v_1;
              }
              if constexpr (kFloatRange) {
                rlo[qi] = fminf(rlo[qi], fminf(in_0 ? v_0 : INFINITY, in_1 ? v_1 : INFINITY));
                rhi[qi] = fmaxf(rhi[qi], fmaxf(in_0 ? v_0 : -INFINITY, in_1 ? v_1 : -INFINITY));
              } else {
                const uint32_t o_0 = score_order(v_0), o_1 = score_order(v_1);
                if (in_0) {
                  rlo[qi] = min(rlo[qi], o_0);
                  rhi[qi] = max(rhi[qi], o_0);
                }
                if (in_1) {
                  rlo[qi] = min(rlo[qi], o_1);
                  rhi[qi] = max(rhi[qi], o_1);
                }
              }
            }
          } else if constexpr (kMode == Mode::kExact) {
            // Only the listed row's scores; histogram or collect by phase.
            const int qi = only_qi;
            float v_0, v_1;
            exact_scores(kf, qi, scale_0, scale_1, ksf, v_0, v_1);
            const int key_0 = key_base + g, key_1 = key_base + g + 8;
            const bool in_0 = t4 == 0 && key_0 >= rks[qi] && key_0 < rke[qi];
            const bool in_1 = t4 == 0 && key_1 >= rks[qi] && key_1 < rke[qi];
            const uint32_t o_0 = score_order(v_0), o_1 = score_order(v_1);
            const uint32_t id_0 = static_cast<uint32_t>(key_0 - rks[qi]);
            const uint32_t id_1 = static_cast<uint32_t>(key_1 - rks[qi]);
            if (phase < 3) {
              if (in_0 && (o_0 & x_hi_mask) == x_prefix)
                atomicAdd(shist + ((o_0 >> x_shift) & x_bin_mask), 1u);
              if (in_1 && (o_1 & x_hi_mask) == x_prefix)
                atomicAdd(shist + ((o_1 >> x_shift) & x_bin_mask), 1u);
            } else if (phase < 6) {
              if (in_0 && o_0 == x_kth && (id_0 & x_hi_mask) == x_prefix)
                atomicAdd(shist + ((id_0 >> x_shift) & x_bin_mask), 1u);
              if (in_1 && o_1 == x_kth && (id_1 & x_hi_mask) == x_prefix)
                atomicAdd(shist + ((id_1 >> x_shift) & x_bin_mask), 1u);
            } else {
              const bool keep_0 =
                  in_0 && (o_0 > x_kth || (o_0 == x_kth && (x_ties_all || id_0 >= x_id_cut)));
              const bool keep_1 =
                  in_1 && (o_1 > x_kth || (o_1 == x_kth && (x_ties_all || id_1 >= x_id_cut)));
              const uint32_t m0 = __ballot_sync(0xffffffffu, keep_0);
              const uint32_t m1 = __ballot_sync(0xffffffffu, keep_1);
              const int n0 = __popc(m0), n = n0 + __popc(m1);
              if (n > 0) {
                const int r = row0 + qi;
                int slot0 = 0;
                if (lane == 0) slot0 = atomicAdd(count_ + r, n);
                slot0 = __shfl_sync(0xffffffffu, slot0, 0);
                const uint32_t below = (1u << lane) - 1u;
                const int64_t row_off = static_cast<int64_t>(r) * capacity_;
                // The selection holds at most capacity keys by construction.
                if (keep_0) {
                  const int slot = slot0 + __popc(m0 & below);
                  cand_score_[row_off + slot] = v_0;
                  cand_id_[row_off + slot] = static_cast<int32_t>(id_0);
                }
                if (keep_1) {
                  const int slot = slot0 + n0 + __popc(m1 & below);
                  cand_score_[row_off + slot] = v_1;
                  cand_id_[row_off + slot] = static_cast<int32_t>(id_1);
                }
              }
            }
          } else {
            // All rows' scores first, in one branch-free block, so the compiler
            // overlaps row qi's epilogue with row qi + 1's MMAs; then the appends,
            // with one counter round trip per tile for all rows.
            float v_0[kBlockQ], v_1[kBlockQ];
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) {
              exact_scores(kf, qi, scale_0, scale_1, ksf, v_0[qi], v_1[qi]);
            }
            const int key_0 = key_base + g, key_1 = key_base + g + 8;
            uint32_t m0[kBlockQ], m1[kBlockQ];
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) {
              m0[qi] = __ballot_sync(0xffffffffu, t4 == 0 && key_0 >= rks[qi] && key_0 < rke[qi] &&
                                                      v_0[qi] >= rtau[qi]);
              m1[qi] = __ballot_sync(0xffffffffu, t4 == 0 && key_1 >= rks[qi] && key_1 < rke[qi] &&
                                                      v_1[qi] >= rtau[qi]);
            }
            // Most tiles add nothing: one uniform test of every row's masks.
            uint32_t any = 0u;
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) any |= m0[qi] | m1[qi];
            const uint32_t below = (1u << lane) - 1u;
#pragma unroll
            for (int qi = 0; qi < kBlockQ && any != 0u; ++qi) {
              const int n0 = __popc(m0[qi]), n = n0 + __popc(m1[qi]);
              if (n == 0) continue;
              float* sc = ws_score + qi * kWarpCap + ws_n[qi];
              int32_t* id = ws_id + qi * kWarpCap + ws_n[qi];
              if ((m0[qi] >> lane) & 1u) {
                const int at = __popc(m0[qi] & below);
                sc[at] = v_0[qi];
                id[at] = key_0 - rks[qi];
              }
              if ((m1[qi] >> lane) & 1u) {
                const int at = n0 + __popc(m1[qi] & below);
                sc[at] = v_1[qi];
                id[at] = key_1 - rks[qi];
              }
              ws_n[qi] += n;
              if (ws_n[qi] > kWarpCap - 16) flush_row(qi);
            }
          }
          fence_proxy_async();  // order this stage's generic reads before its TMA refill
          mbar_arrive(empty_k + st);
        }
        if constexpr (kMode == Mode::kScan) {
#pragma unroll
          for (int qi = 0; qi < kBlockQ; ++qi)
            if (ws_n[qi] > 0) flush_row(qi);
        }
        if constexpr (kMode == Mode::kExact) {
          if (x_active && phase < 6) exact_unit_done(p, phase, row0 + only_qi, shist, tid);
        }
        fence_proxy_async();
        mbar_arrive(empty_q + qs);
        if constexpr (kMode == Mode::kResident) {
          // Each row's order-key range over the block: per warp, then over the
          // 8 warps.
          __shared__ uint32_t s_rng[kNumMathThreads / 32][kBlockQ][2];
#pragma unroll
          for (int qi = 0; qi < kBlockQ; ++qi) {
            uint32_t a, b;
            if constexpr (kFloatRange) {
              a = __reduce_min_sync(0xffffffffu, score_order(rlo[qi]));
              b = __reduce_max_sync(0xffffffffu, score_order(rhi[qi]));
            } else {
              a = __reduce_min_sync(0xffffffffu, rlo[qi]);
              b = __reduce_max_sync(0xffffffffu, rhi[qi]);
            }
            if (lane == 0) {
              s_rng[warp][qi][0] = a;
              s_rng[warp][qi][1] = b;
            }
          }
          math_barrier();  // every score of the block written, the ranges posted
          if (tid == 0) {
#pragma unroll
            for (int qi = 0; qi < kBlockQ; ++qi) {
              uint32_t lo = 0xffffffffu, hi = 0u;
              for (int w = 0; w < kNumMathThreads / 32; ++w) {
                lo = min(lo, s_rng[w][qi][0]);
                hi = max(hi, s_rng[w][qi][1]);
              }
              res_rng[res_it % kSlots][qi][0] = lo;
              res_rng[res_it % kSlots][qi][1] = hi;
            }
            res_row0[res_it % kSlots] = row0;
            mbar_arrive(res_ready + res_it % kSlots);  // releases the scores and ranges
          }
          math_barrier();  // s_rng read before the next block's writes
          ++res_it;
        }
      }
      if constexpr (kMode == Mode::kExact) grid_barrier();
    }  // phase
    if constexpr (kMode == Mode::kResident) {
      math_barrier();  // every math warp done with the pipeline's buffers
      res_claims(kSelWarps + warp,
                 [&](int r, const float* sc, int n, uint32_t lo, uint32_t hi, uint32_t* lk,
                     int32_t* li, uint32_t* b, uint32_t* h, int hbits, int t_, SelShared& s_) {
                   sel_row_call<kNumHeads == 32>(p, r, sc, n, lo, hi, lk, li, b, h, hbits, t_, s_);
                 });
    }
    if constexpr (kRebalance) {
      setmaxnreg_dec<168>();
      __syncthreads();
    }
  }  // roles

  // Fused tails: every thread of every CTA, the whole dynamic shared memory.
  if (p.fuse_tail) {
    if constexpr (kMode == Mode::kSample) {
      grid_barrier();  // every row's samples written
      for (int r = blockIdx.x; r < p.num_rows; r += gridDim.x)
        tau_row(p.tau_p, r, reinterpret_cast<uint32_t*>(smem));
      if (blockIdx.x == gridDim.x - 1) {
        __syncthreads();
        tau_setup(p.tau_p, false, true);
      }
    } else if constexpr (kMode == Mode::kScan) {
      grid_barrier();  // every row's candidates appended
      for (int r = blockIdx.x; r < p.num_rows; r += gridDim.x) finalize_row(p.fin_p, r, smem);
    } else {
      // The last phase's barrier ordered the collected selections.
      const int n = *p.fin_p.rows_count;
      for (int i = blockIdx.x; i < n; i += gridDim.x) finalize_row(p.fin_p, p.fin_p.rows[i], smem);
    }
  }
}

}  // namespace flashinfer::sm120_dsa_indexer
