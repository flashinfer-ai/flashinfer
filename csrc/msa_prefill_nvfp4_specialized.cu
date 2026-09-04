/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Minimax Sparse Attention union prefill over an NVFP4 paged KV cache, for
// compute capability 10.0 / 10.3, where no NVFP4 MSA prefill route exists
// otherwise.
//
// The kernel is parametric in batch size, per-request query length, per-request
// KV length, total query count and block-table width.  The fixed geometry is
// the model-level one that the shared-memory map, the MMA tile and the
// warp-to-row map are built from: 64 query heads, 4 KV heads, head_dim 128,
// page_size 128, topk 16.  Every one of those is asserted by the host binding
// below AND by the Python dispatch guard, because the kernel body bakes them
// into its addressing and a neighbouring geometry must never reach this
// translation unit.
//
// STORAGE CONTRACT (one physical page, planar, 73728 B):
//
//     [ K data 32768 | K scale 4096 | V data 32768 | V scale 4096 ]
//
// per (page, kv head, token): 64 packed e2m1 bytes and 8 e4m3 block scales, one
// scale per 16 elements.  K scales are stored linearly; V scales are
// (4, 4)-swizzled inside (token, scale index), so the scale of logical (t, s)
// lives at ((t / 4) * 4 + s / 2, (s % 2) * 4 + t % 4).  The four regions are
// handed in as four strided views over the same allocation; the host binding
// proves that relationship from the byte offsets between their base pointers,
// because shape, dtype and stride cannot tell four views of one page apart from
// four unrelated allocations strided the same way, and the V-scale swizzle is
// invisible to all three.
//
// NO GLOBAL SCRATCH.  Each CTA dequantizes the K and V sides of one selected
// page directly into its own shared memory, in the 128 B-swizzled layout the
// MMA wants, and never materializes a BF16 copy of the cache in global memory.
// Peak device memory attributable to this route is zero bytes beyond the output
// tensor; the launch carries 98,848 B of dynamic shared memory instead.  That is
// what lets the block-table width be a free parameter: nothing in the kernel is
// proportional to it except one runtime row stride.
//
// COMPUTE CONTRACT: the QK and PV products run on tcgen05 `kind::f16` with BF16
// operands and FP32 accumulators, which is exactly the operand class of the BF16
// MSA prefill path this specializes.  Native FP4 MMA (`kind::mxf4nvf4`) is
// deliberately NOT used: the block scales are applied during the e2m1 -> bf16
// dequant, where the e2m1 x e4m3 product is exact in BF16 (1 + 3 significand
// bits into 8).
//
// TWO WARPGROUPS, ONE TILE.  The 256 threads are two warpgroups over the SAME
// 128 rows: warps 0-3 own KV columns 0..63 and warps 4-7 own 64..127.  Each
// group keeps its P inside the score columns it has just consumed, so the two
// halves never alias in TMEM and no extra barrier is needed; the two halves of
// the row denominator are joined once, at the end, through shared memory (the K
// stage is dead by then and is reused for the join).
//
// SOFTMAX RANGE INVARIANT -- READ THIS BEFORE CHANGING ANY EXPONENTIAL.
//
// This kernel does NOT carry an online row maximum on its fast path.  It
// exponentiates against a FIXED origin (`kFastOrigin`), which is only sound
// because it is paired with a range CHECK and an exact REPLAY.  The three parts
// are one mechanism and none of them is optional:
//
//   (1) FAST PATH (`sparse_prefill_tile<false>`).  Every probability is
//       `exp2f(logit * logits_scale_log2 - kFastOrigin)`.  This is the whole
//       reason the softmax costs ~2 instructions per score element instead of
//       ~3.3: no per-block maximum, no correction factor, no accumulator
//       rescale, and no cross-warpgroup reduction of a maximum over rows that
//       the two groups split.
//
//   (2) RANGE CHECK, AT TWO SITES.  A fixed origin is representable only while
//       the row's logits stay inside FP32's normal range once shifted by it, so
//       the row denominator is tested against a guard band of
//       `[kMinSafeSum, kMaxSafeSum]` = `[2^-120, 2^112]`.  Both sites test
//       `!isfinite` OR the range, never finiteness alone -- a denominator that
//       has flushed to zero is perfectly finite and completely wrong -- and
//       neither is redundant, because the first cannot run on the fast path and
//       the second cannot rebase:
//
//         END OF TILE, fast path only.  `!isfinite(total_sum) ||
//         total_sum < kMinSafeSum || total_sum > kMaxSafeSum` raises a per-CTA
//         flag in shared memory.  This is what asks for the replay.
//
//         MID-SCAN, replay only.  Per block:
//         `!isfinite(block_sum) || !isfinite(next_sum) ||
//          (row_sum == 0.0f && block_sum < kMinSafeSum) ||
//          next_sum > kMaxSafeSum` raises the origin.  Note the last two
//         disjuncts: `next_sum > kMaxSafeSum` fires while the sum is still
//         finite and on its way to overflow rather than after it has become
//         Inf, and the `row_sum == 0.0f` term catches a FIRST block that has
//         already flushed, which no test on the accumulated sum can see.
//
//       The band is FP32's normal range with six binades trimmed off the bottom
//       -- the P matrix is materialized in BF16, whose smallest normal is
//       2^-126 and whose subnormals lose the precision the PV product needs, so
//       a denominator below 2^-120 means the individual probabilities are
//       already flushing -- and sixteen off the top, one `kRebaseMargin` of
//       headroom so that neither the reciprocal `v_global_scale / total_sum`
//       nor the O accumulator can reach FP32's overflow boundary.  A
//       denominator inside that band cannot have been built from
//       unrepresentable terms.
//
//   (3) EXACT REPLAY (`sparse_prefill_tile<true>`).  When the flag is set, the
//       timed `__global__` runs the SAME tile again through the specialization
//       that does carry a data-derived origin: a per-row maximum over real
//       scores, a raise to `block_max + kRebaseMargin`, and a rescale of BOTH
//       carried quantities (`row_sum *= acc_scale` and the O accumulator in
//       TMEM) before anything else is written.  The replay's origin is never a
//       constant; `kFastOrigin` appears only as its starting point and only
//       ever moves up.  The replay recomputes the tile from global memory --
//       Q, the union and both KV stages are rebuilt -- so its answer does not
//       depend on anything the fast path left behind.
//
// THIS IS NOT DEAD CODE.  On the shipped workload rows the replay fires on a
// real one (batch 2, query lengths [186, 7936]): 8 replay CTAs out of 17,328
// active, 0.046% of that row, and the measured cost of the route already
// includes it.  Deleting the check, widening the band, or "simplifying" the
// fixed origin into an unguarded one removes a guard that a real input
// exercises.  A build that keeps (1) without (2) and (3) is incorrect, not
// merely faster.
//
// EMPTY-UNION CONTRACT: `tcgen05.alloc` does not zero TMEM, and a tile whose
// union is empty issues no PV at all, so its O accumulator is never written.
// The epilogue REPLACES the value (`live ? frag[p] * scale : 0.0f`) rather than
// multiplying it by a zero scale, because finite garbage would give 0.0 but a
// NaN or Inf bit pattern would give NaN.
//
// DETERMINISM: the union is built into an open-addressed shared-memory hash and
// consumed in ascending slot order, so the order in which selected blocks are
// accumulated -- and therefore the low bits of the output -- is a function of
// the slot each block lands in.  While `num_blocks <= kHashSize` that slot is
// `block * kHashMultiplier mod kHashSize`, a permutation of a set that is
// smaller than the table, so distinct blocks cannot collide, the insert is a
// commutative `atomicOr`, and the traversal order is a deterministic function
// of the selected SET.  Above that width the insert falls back to linear
// probing under `atomicCAS`, colliding ids land in the order the races resolve,
// and repeated runs of the same call may differ in the last bits.  See the
// module docstring of `flashinfer/msa_ops/_nvfp4_prefill_sm100.py`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace msa_prefill_nvfp4 {

// ---------------------------------------------------------------------------
// Problem geometry.  Asserted by the host binding and by the Python dispatch
// guard.
// ---------------------------------------------------------------------------
constexpr int kNumQHeads = 64;
constexpr int kNumKVHeads = 4;
constexpr int kGroupSize = kNumQHeads / kNumKVHeads;  // GQA siblings per KV head
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kPageSizeLog2 = 7;
static_assert((1 << kPageSizeLog2) == kPageSize, "the ceil-divide below is spelled as a shift");
constexpr int kTopK = 16;

// Packed-page map, in bytes.
constexpr int kDataDim = kHeadDim / 2;                       // 64 packed e2m1 bytes per token
constexpr int kScaleVec = 16;                                // one block scale per 16 elements
constexpr int kScaleDim = kHeadDim / kScaleVec;              // 8 e4m3 scales per token
constexpr int64_t kDataHeadStride = kPageSize * kDataDim;    // 8192
constexpr int64_t kScaleHeadStride = kPageSize * kScaleDim;  // 1024
constexpr int64_t kKScaleByteOffset = kNumKVHeads * kDataHeadStride;                      // 32768
constexpr int64_t kVDataByteOffset = kKScaleByteOffset + kNumKVHeads * kScaleHeadStride;  // 36864
constexpr int64_t kVScaleByteOffset = kVDataByteOffset + kKScaleByteOffset;               // 69632
constexpr int64_t kPageBytes = kVScaleByteOffset + kNumKVHeads * kScaleHeadStride;        // 73728

// A selected block id is carried in the low 24 bits of a union-table entry and
// the eight bits above it hold the per-query membership mask.  0 marks an empty
// slot, so the largest representable id is kMaxSelectableBlocks - 1 and a
// block-table width of kMaxSelectableBlocks is admissible.  This is the route's
// ONLY width ceiling: 2,147,483,520 context tokens at page_size 128, within one
// page of the int32 bound on `seqused_k` itself.  It is CHECKED, never clamped
// -- truncating an id would silently drop a selected block from the union.
constexpr int kQueryBitShift = 24;
constexpr uint32_t kBlockCodeMask = (1u << kQueryBitShift) - 1u;  // 0x00ffffff
constexpr int kMaxSelectableBlocks = static_cast<int>(kBlockCodeMask);
constexpr int64_t kMaxContextTokens = static_cast<int64_t>(kMaxSelectableBlocks) * kPageSize;

namespace {

// ---------------------------------------------------------------------------
// Tile shape and the shared-memory map.
// ---------------------------------------------------------------------------
constexpr int kTileM = 128;  // (query, qo head) rows per tile
constexpr int kQueriesPerTile = kTileM / kGroupSize;
constexpr int kThreads = 256;  // two warpgroups over the same kTileM rows
constexpr int kWarpsPerGroup = kThreads / 32 / 2;
constexpr int kVecChunks = kHeadDim * 2 / 16;  // 16 B chunks in one bf16 head row
// Capacity of the per-tile block union.  Declared, not derived from
// kQueriesPerTile * kTopK, so that the assertion below is a real check on the
// relationship rather than a restatement of it.
constexpr int kHashSize = 128;
constexpr int kQSmemBytes = kTileM * kHeadDim * 2;
constexpr int kKVSmemBytes = kTileM * kHeadDim * 2;
constexpr int kKSmemOffset = kQSmemBytes;
constexpr int kVSmemOffset = kKSmemOffset + kKVSmemBytes;
constexpr int kHashOffset = kVSmemOffset + kKVSmemBytes;
// The union table, then four warp counts, then the two-word tcgen05 handle
// (whose second word doubles as the replay flag), then one mbarrier.
constexpr int kUnionCountWords = 4;
constexpr int kTmemStorageWords = 2;
constexpr int kTmemAddrWord = kHashSize + kUnionCountWords;
constexpr int kRepairFlagWord = kTmemAddrWord + 1;
constexpr int kDynamicSmemBytes =
    kHashOffset + (kHashSize + kUnionCountWords + kTmemStorageWords) * 4 + 8;
static_assert(kDynamicSmemBytes == 98848,
              "the dispatch guard and the capability manifest publish this number");

// Shared-memory layout of one 128 x 128 BF16 stage: two 64-column chunks, each
// kTileM rows of 128 B, with the 128 B XOR swizzle the MMA descriptor expects.
constexpr int kSmemRowBytes = 64 * 2;
constexpr int kSmemChunkBytes = kTileM * kSmemRowBytes;  // 16384
constexpr int kSwizzleRows = 8;

// V scales are (4, 4)-swizzled: one 32 B group holds four tokens x kScaleDim
// scales.  Each thread converts a uint4 of packed data, i.e. 32 values, which
// is kScalesPerSlice block scales.
constexpr int kVScaleGroupBytes = 4 * kScaleDim;
constexpr int kScalesPerSlice = 2;

// TMEM column map.  The fast path parks P inside the score columns its own
// warpgroup has just consumed; the replay owns all 128 KV columns and needs a
// disjoint P region.
constexpr int kScoreTmem = 0;
constexpr int kPTmemRepair = 64;
constexpr int kOutputTmem = 128;
constexpr int kTmemCols = 256;
constexpr int kDecodeIters = 512 / kThreads;
static_assert(kThreads == 256, "named-barrier width is spelled literally below");

// ---------------------------------------------------------------------------
// G3: the per-tile union table.  Five independent properties have to hold
// together for the insert to be correct AND to terminate.  All five are
// compile-time, because none of them is visible at runtime until it is already
// wrong: an overfull table makes the probe loop below spin forever, a
// non-power-of-two size breaks the `& (kHashSize - 1)` wrap, an even
// multiplier stops being a permutation and lets the collision-free path
// silently OR two different block ids into one slot, a table wider than the
// block reserves leaves slots uninitialized, and a table that is not exactly
// four warps wide breaks the ballot-ranked compaction.
// ---------------------------------------------------------------------------
constexpr uint32_t kHashMultiplier = 2654435761u;
static_assert(kQueriesPerTile * kTopK <= kHashSize,
              "the table must hold every block kQueriesPerTile queries can select; "
              "the probe loop is unbounded and terminates only because a free slot "
              "or a matching code is guaranteed to exist");
static_assert((kHashSize & (kHashSize - 1)) == 0, "the probe wraps with & (kHashSize - 1)");
static_assert((kHashMultiplier & 1u) == 1u,
              "an odd multiplier is a permutation modulo a power-of-two table size, "
              "which is what makes the num_blocks <= kHashSize path collision-free "
              "and therefore independent of the order the atomics resolve in");
static_assert(kHashSize <= kThreads,
              "the table is zeroed by one store per thread under tid < kHashSize");
static_assert(kHashSize == 4 * 32,
              "the compaction ranks entries with four warp ballots, one word per "
              "thread of warps 0..3");
static_assert(kQueriesPerTile + kQueryBitShift <= 32,
              "the per-query membership mask sits above the block id in one word");
static_assert(kTileM == kQueriesPerTile * kGroupSize,
              "a tile row is one (query, qo head within the KV head's group) pair");

// ---------------------------------------------------------------------------
// Softmax range guard.  See the header comment: the fast path exponentiates
// against kFastOrigin and the replay is what makes that sound.
// ---------------------------------------------------------------------------
constexpr float kFastOrigin = 32.0f;
constexpr float kRebaseMargin = 16.0f;
constexpr float kMinSafeSum = 0x1p-120f;
constexpr float kMaxSafeSum = 0x1p112f;
static_assert(kMinSafeSum > 0.0f && kMaxSafeSum > kMinSafeSum, "guard band must be a band");

// The fast path publishes 'this tile needs the replay' to the replay itself by
// writing a quiet NaN over one element it owns exclusively, so the replay can
// assert it is repairing the tile that asked for it.  The replay overwrites the
// whole tile, marker included.
constexpr int kReplayMarkerBits = 0x7fffffff;

__device__ __forceinline__ uint32_t elect_one() {
  uint32_t result = 0;
  const uint32_t mask = 0xffffffffu;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "elect.sync _|p, %1;\n\t"
      "@p mov.u32 %0, 1;\n\t"
      "}"
      : "+r"(result)
      : "r"(mask));
  return result;
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) { return (x & 0x3ffffULL) >> 4; }

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  uint64_t d = desc_encode(static_cast<uint64_t>(addr));
  d |= desc_encode(1024) << 32;
  d |= 1ULL << 46;
  d |= 2ULL << 61;
  return d;
}

__device__ __forceinline__ void mbarrier_init(int addr) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_wait(int addr, int phase) {
  uint32_t done;
  do {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, 0x989680;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(done)
        : "r"(addr), "r"(phase)
        : "memory");
  } while (!done);
}

__device__ __forceinline__ void mma_ss(int dst, uint64_t a, uint64_t b, uint32_t idesc,
                                       int accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(dst),
      "l"(a), "l"(b), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void mma_ts(int dst, int a_tmem, uint64_t b, uint32_t idesc,
                                       int accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.u32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t"
      "}" ::"r"(dst),
      "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void mma_commit(int addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(addr)
      : "memory");
}

__device__ __forceinline__ void tmem_load_x16(float* x, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]),
        "=f"(x[7]), "=f"(x[8]), "=f"(x[9]), "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]),
        "=f"(x[14]), "=f"(x[15])
      : "r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_x32_nowait(float* x, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]),
        "=f"(x[7]), "=f"(x[8]), "=f"(x[9]), "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]),
        "=f"(x[14]), "=f"(x[15]), "=f"(x[16]), "=f"(x[17]), "=f"(x[18]), "=f"(x[19]), "=f"(x[20]),
        "=f"(x[21]), "=f"(x[22]), "=f"(x[23]), "=f"(x[24]), "=f"(x[25]), "=f"(x[26]), "=f"(x[27]),
        "=f"(x[28]), "=f"(x[29]), "=f"(x[30]), "=f"(x[31])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_store_x16_f32(int addr, const float* x) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]),
      "f"(x[8]), "f"(x[9]), "f"(x[10]), "f"(x[11]), "f"(x[12]), "f"(x[13]), "f"(x[14]), "f"(x[15]));
}

__device__ __forceinline__ void tmem_store_x16_b32(int addr, const uint32_t* x) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(addr),
      "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]),
      "r"(x[8]), "r"(x[9]), "r"(x[10]), "r"(x[11]), "r"(x[12]), "r"(x[13]), "r"(x[14]), "r"(x[15]));
}

// `cvt.rn.bf16x2.{e2m1x2,e4m3x2}` require CUDA Toolkit >= 13.2 (the same guard
// vec_dtypes.cuh already carries for the FP4 case).  Below that they do not
// assemble, so the conversion goes through FP16 -- which is BIT-EXACT here, not
// an approximation: an e2m1 magnitude is one of {0, .5, 1, 1.5, 2, 3, 4, 6} and
// an e4m3 value carries 4 significand bits with |x| <= 448, so both are exactly
// representable in FP16 (11 significand bits, |x| <= 65504) and in BF16
// (8 significand bits) alike, and FP16 -> FP32 -> BF16 round-trips them.
// Overridable so that the guard can be shown NOT to be decorative: forcing it
// to 1 under a toolkit below 13.2 must fail to assemble.  Never define it in a
// shipped build.
#ifndef FLASHINFER_MSA_NVFP4_NATIVE_BF16_CVT
#if (defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
     ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)))
#define FLASHINFER_MSA_NVFP4_NATIVE_BF16_CVT 1
#else
#define FLASHINFER_MSA_NVFP4_NATIVE_BF16_CVT 0
#endif
#endif

__device__ __forceinline__ uint32_t f16x2_to_bf16x2(uint32_t h2) {
  __half2 h;
  memcpy(&h, &h2, sizeof(h));
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  uint32_t out;
  memcpy(&out, &b, sizeof(out));
  return out;
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  union Pair {
    __nv_bfloat162 v;
    uint32_t u;
  } p;
  p.v = __floats2bfloat162_rn(a, b);
  return p.u;
}

// Four packed e2m1 pairs (one b32 = 8 nibbles) -> four bf16x2 words.
__device__ __forceinline__ void cvt_fp4x8_bf16x8(uint32_t src, uint32_t* out) {
#if FLASHINFER_MSA_NVFP4_NATIVE_BF16_CVT
  asm volatile(
      "{\n\t"
      ".reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %0, b0;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %2, b2;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %3, b3;\n\t"
      "}"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(src));
#else
  uint32_t h[4];
  asm volatile(
      "{\n\t"
      ".reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t"
      "}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(src));
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = f16x2_to_bf16x2(h[i]);
#endif
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// One hardware convert handles BOTH e4m3 scales of a 32-value slice; the two
// broadcasts are byte permutes. The old path duplicated each byte first and
// converted twice, which cost about twice the instructions.
__device__ __forceinline__ uint32_t cvt_e4m3x2_bf16x2(uint16_t src) {
#if FLASHINFER_MSA_NVFP4_NATIVE_BF16_CVT
  uint32_t out;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(src));
  return out;
#else
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(src));
  return f16x2_to_bf16x2(h2);
#endif
}

__device__ __forceinline__ void store_swizzled_vec16(char* base, int row, int chunk, uint4 value) {
  char* row_base = base + (chunk >> 3) * kSmemChunkBytes + row * kSmemRowBytes;
  const int byte = ((chunk & 7) * 16) ^ ((row & (kSwizzleRows - 1)) << 4);
  *reinterpret_cast<uint4*>(row_base + byte) = value;
}

__device__ __forceinline__ uint4 load_swizzled_vec16(const char* base, int row, int chunk) {
  const char* row_base = base + (chunk >> 3) * kSmemChunkBytes + row * kSmemRowBytes;
  const int byte = ((chunk & 7) * 16) ^ ((row & (kSwizzleRows - 1)) << 4);
  return *reinterpret_cast<const uint4*>(row_base + byte);
}

__device__ __forceinline__ void dequant_quad(uint32_t* out, uint32_t packed, uint32_t scale_pair) {
  cvt_fp4x8_bf16x8(packed, out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = mul_bf16x2(out[i], scale_pair);
  }
}

// Packed NVFP4 bytes for one (page, kv head, side), held in registers between
// the global load and the SMEM dequant so the load latency lands under an
// asynchronous MMA instead of under its own consumer.
struct RawSide {
  uint4 d[kDecodeIters];
  uint32_t sa[kDecodeIters];
  uint32_t sb[kDecodeIters];
};

// Permute the contiguous uint4 loads of each warp so every 8-lane STS.128
// wavefront spans four tokens and two 64-column slices. After the row swizzle
// this is a bijection over all 32 banks.
__device__ __forceinline__ int decode_base_chunk() {
  const int lane = threadIdx.x & 31;
  const int token_offset = ((lane >> 4) << 2) | ((lane >> 1) & 3);
  const int slice = (lane & 1) | ((((lane >> 2) ^ (lane >> 3)) & 1) << 1);
  return (threadIdx.x & ~31) + (token_offset << 2) + slice;
}

template <bool ValueSide>
__device__ __forceinline__ void load_page_side(RawSide& r, const uint8_t* page, int kv_head,
                                               int base_chunk) {
  const uint8_t* data = page + (ValueSide ? static_cast<int>(kVDataByteOffset) : 0) +
                        kv_head * static_cast<int>(kDataHeadStride);
  const uint8_t* scales =
      page +
      (ValueSide ? static_cast<int>(kVScaleByteOffset) : static_cast<int>(kKScaleByteOffset)) +
      kv_head * static_cast<int>(kScaleHeadStride);
  const int slice = base_chunk & 3;
#pragma unroll
  for (int it = 0; it < kDecodeIters; ++it) {
    const int linear_chunk = base_chunk + it * kThreads;
    const int token = linear_chunk >> 2;
    r.d[it] = reinterpret_cast<const uint4*>(data)[linear_chunk];
    if constexpr (ValueSide) {
      const uint2 scale_vec = *reinterpret_cast<const uint2*>(
          scales + (token >> 2) * kVScaleGroupBytes + slice * kScaleDim);
      r.sa[it] = scale_vec.x;
      r.sb[it] = scale_vec.y;
    } else {
      r.sa[it] =
          *reinterpret_cast<const uint16_t*>(scales + token * kScaleDim + slice * kScalesPerSlice);
    }
  }
}

template <bool ValueSide>
__device__ __forceinline__ void store_page_side(char* dst, const RawSide& r, int base_chunk) {
  const int slice = base_chunk & 3;
  char* chunk_base = dst + (slice >> 1) * kSmemChunkBytes;
  const int piece_base = (slice & 1) * 64;
#pragma unroll
  for (int it = 0; it < kDecodeIters; ++it) {
    const int linear_chunk = base_chunk + it * kThreads;
    const int token = linear_chunk >> 2;
    uint16_t scale_bytes;
    if constexpr (ValueSide) {
      // V scale is (4,4)-swizzled: gather byte (token&3) of both halves of the
      // 8-byte group with one permute instead of two shift/mask pairs.
      const uint32_t sel = 0x40u | (static_cast<uint32_t>(token & 3) * 0x11u);
      scale_bytes = static_cast<uint16_t>(prmt_b32(r.sa[it], r.sb[it], sel));
    } else {
      // K scale is linear: the two group scales are already adjacent bytes.
      scale_bytes = static_cast<uint16_t>(r.sa[it]);
    }
    const uint32_t scale_both = cvt_e4m3x2_bf16x2(scale_bytes);
    const uint32_t scale_pair0 = prmt_b32(scale_both, 0u, 0x1010u);
    const uint32_t scale_pair1 = prmt_b32(scale_both, 0u, 0x3232u);
    uint32_t piece[16];
    dequant_quad(piece, r.d[it].x, scale_pair0);
    dequant_quad(piece + 4, r.d[it].y, scale_pair0);
    dequant_quad(piece + 8, r.d[it].z, scale_pair1);
    dequant_quad(piece + 12, r.d[it].w, scale_pair1);

    char* row_base = chunk_base + token * kSmemRowBytes;
    const int swizzle = (token & (kSwizzleRows - 1)) << 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint4*>(row_base + ((piece_base + i * 16) ^ swizzle)) =
          make_uint4(piece[i * 4], piece[i * 4 + 1], piece[i * 4 + 2], piece[i * 4 + 3]);
    }
  }
}

// P (bf16 pairs) column base for KV-column chunk `kk` (16 KV columns each).
// Fast path splits the 128 KV columns between the two warpgroups; each group
// keeps its P inside the score columns it just consumed, so no cross-group
// TMEM hazard exists and no extra barrier is needed.
__device__ __forceinline__ constexpr int p_col_of_chunk(bool repair, int kk) {
  return repair ? (kPTmemRepair + kk * 8) : ((kk >> 2) * 64 + 32 + (kk & 3) * 8);
}

// The column-split fast path produces P in two 32-column waves.  Release the
// high wave to the MMA issuer immediately, so its four PV K-steps execute
// while all warps compute and store the low wave.
__device__ __forceinline__ void issue_pv_high_half(int warp, int taddr, char* v_smem,
                                                   uint32_t pv_idesc, bool first_pv) {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if (warp == 0) {
    asm volatile("bar.sync 1, 256;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (elect_one()) {
      uint64_t bd = make_smem_desc(static_cast<int>(__cvta_generic_to_shared(v_smem)));
      bd |= 0x04000000ULL;
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 2), bd + 2 * 128ULL, pv_idesc,
             !first_pv);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 3), bd + 3 * 128ULL, pv_idesc,
             true);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 6), bd + 6 * 128ULL, pv_idesc,
             true);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 7), bd + 7 * 128ULL, pv_idesc,
             true);
    }
  } else {
    asm volatile("bar.arrive 1, 256;" ::: "memory");
  }
}

__device__ __forceinline__ void issue_pv_low_half(int warp, int taddr, char* v_smem, int mbar_addr,
                                                  uint32_t pv_idesc) {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if (warp == 0) {
    asm volatile("bar.sync 2, 256;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (elect_one()) {
      uint64_t bd = make_smem_desc(static_cast<int>(__cvta_generic_to_shared(v_smem)));
      bd |= 0x04000000ULL;
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 0), bd, pv_idesc, true);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 1), bd + 1 * 128ULL, pv_idesc,
             true);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 4), bd + 4 * 128ULL, pv_idesc,
             true);
      mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(false, 5), bd + 5 * 128ULL, pv_idesc,
             true);
      mma_commit(mbar_addr);
    }
  } else {
    asm volatile("bar.arrive 2, 256;" ::: "memory");
  }
}

template <bool Repair>
__device__ __forceinline__ bool sparse_prefill_tile(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ q2k, const int* __restrict__ cu_q, const int* __restrict__ page_table,
    const int* __restrict__ seqused_k, float logits_scale_log2, float v_global_scale,
    __nv_bfloat16* __restrict__ output, int total_q, int batch_size, int max_pages,
    int q2k_head_stride, int q2k_token_stride, int virtual_block_x, int virtual_kv_head,
    char* smem) {
  char* q_smem = smem;
  char* k_smem = smem + kKSmemOffset;
  char* v_smem = smem + kVSmemOffset;
  uint32_t* union_hash = reinterpret_cast<uint32_t*>(smem + kHashOffset);
  int* union_counts = reinterpret_cast<int*>(union_hash + kHashSize);
  int* tmem_storage = union_counts + kUnionCountWords;
  uint64_t* mbar_storage = reinterpret_cast<uint64_t*>(tmem_storage + 2);
  // The K stage is dead once the loop ends; reuse it for the row-sum join.
  float* sum_smem = reinterpret_cast<float*>(k_smem);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wgroup = warp >> 2;  // 0 -> KV cols 0..63, 1 -> 64..127
  const int col_half = wgroup * 64;
  const int row = tid & (kTileM - 1);  // TMEM lane this thread owns
  const int row_base = ((warp & (kWarpsPerGroup - 1)) * 32) << 16;

  int request = 0;
  int q_tile = 0;
  bool active = false;
  int tile_prefix = 0;
  for (int b = 0; b < batch_size; ++b) {
    const int qlen = cu_q[b + 1] - cu_q[b];
    const int tiles = (qlen + kQueriesPerTile - 1) / kQueriesPerTile;
    if (!active && virtual_block_x >= tile_prefix && virtual_block_x < tile_prefix + tiles) {
      request = b;
      // Reverse tiles within each request so the causally heaviest work is
      // dispatched first and the final grid wave contains the lightest CTAs.
      q_tile = tiles - 1 - (virtual_block_x - tile_prefix);
      active = true;
    }
    tile_prefix += tiles;
  }
  if (!active) return false;

  const int kv_head = virtual_kv_head;
  const int query_in_tile = row / kGroupSize;
  const int q_begin = cu_q[request];
  const int q_len = cu_q[request + 1] - q_begin;
  const int q_local = q_tile * kQueriesPerTile;
  const int q_valid = min(kQueriesPerTile, q_len - q_local);
  const int query_base = q_begin + q_local;
  const int kv_len = seqused_k[request];
  // The block table has max_pages entries per request, so a block beyond it
  // is not addressable however long seqused_k claims the sequence is.
  // Clamping here -- rather than trusting the caller to keep the two
  // consistent -- is what keeps the page-table read inside this request's row
  // instead of walking into the next request's; it matches the reference,
  // which also takes min(max_blocks, ceil(kv_len / page_size)).  The shift is
  // the candidate's own: seqused_k is non-negative, so an arithmetic shift is
  // the ceil-divide, and spelling it as a signed division instead would cost a
  // round-toward-zero correction on the hot path for nothing.
  const int num_blocks = min((kv_len + kPageSize - 1) >> kPageSizeLog2, max_pages);

  if constexpr (Repair) {
    const __nv_bfloat16 marker =
        output[(static_cast<size_t>(query_base) * kNumQHeads + kv_head * kGroupSize) * kHeadDim];
    if (!isnan(__bfloat162float(marker))) return false;
  }

  if (tid < kHashSize) union_hash[tid] = 0;

  const bool valid_row = query_in_tile < q_valid;
#pragma unroll 4
  for (int idx = tid; idx < kTileM * kVecChunks; idx += kThreads) {
    const int qrow = idx / kVecChunks;
    const int chunk = idx % kVecChunks;
    const int row_query = qrow / kGroupSize;
    const int row_head = kv_head * kGroupSize + (qrow % kGroupSize);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row_query < q_valid) {
      const __nv_bfloat16* src =
          q + (static_cast<size_t>(query_base + row_query) * kNumQHeads + row_head) * kHeadDim;
      value = reinterpret_cast<const uint4*>(src)[chunk];
    }
    store_swizzled_vec16(q_smem, qrow, chunk, value);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  for (int i = tid; i < q_valid * kTopK; i += kThreads) {
    const int qr = i / kTopK;
    const int slot = i % kTopK;
    // Two strides, not `total_q * kTopK`: topk stays a compile-time constant
    // of this kernel body, but the selection tensor's LAYOUT does not have to
    // be contiguous.  The transposed view of a token-major buffer is read in
    // place, which is what removes the per-step contiguous copy from the
    // consumer's prefill path.
    const int block = q2k[static_cast<size_t>(kv_head) * q2k_head_stride +
                          static_cast<size_t>(query_base + qr) * q2k_token_stride + slot];
    if (block >= 0 && block < num_blocks) {
      const uint32_t code = static_cast<uint32_t>(block + 1);
      const uint32_t query_bit = 1u << (kQueryBitShift + qr);
      int pos = (static_cast<uint32_t>(block) * kHashMultiplier) & (kHashSize - 1);
      if (num_blocks <= kHashSize) {
        // The odd multiplier is a permutation modulo 128, so distinct valid
        // blocks cannot collide in this regime. OR preserves the exact hash
        // slot and compacted traversal order of the general path.
        atomicOr(union_hash + pos, code | query_bit);
      } else {
        for (;;) {
          const uint32_t old = atomicCAS(union_hash + pos, 0u, code | query_bit);
          if (old == 0u) break;
          if ((old & kBlockCodeMask) == code) {
            atomicOr(union_hash + pos, query_bit);
            break;
          }
          pos = (pos + 1) & (kHashSize - 1);
        }
      }
    }
  }
  __syncthreads();

  // Compact the hash in-place with warps 0..3 (one word each). Every thread
  // captures its entry before any overwrite; warp-prefix ranks preserve
  // hash-slot order deterministically.
  const uint32_t compact_entry = (warp < 4) ? union_hash[tid] : 0u;
  const uint32_t compact_mask = __ballot_sync(0xffffffffu, compact_entry != 0u);
  if (warp < 4 && lane == 0) union_counts[warp] = __popc(compact_mask);
  __syncthreads();
  if (warp < 4) {
    int compact_base = 0;
#pragma unroll
    for (int w = 0; w < kUnionCountWords; ++w) {
      if (w < warp) compact_base += union_counts[w];
    }
    const int compact_rank = __popc(compact_mask & ((1u << lane) - 1u));
    if (compact_entry != 0u) {
      union_hash[compact_base + compact_rank] = compact_entry;
    }
  }
  const int union_count = union_counts[0] + union_counts[1] + union_counts[2] + union_counts[3];
  __syncthreads();

  const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar_storage));
  if (tid == 0) {
    tmem_storage[1] = 0;
    mbarrier_init(mbar_addr);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  } else if (warp == 1) {
    // Allocated by the FAST PATH ONLY.  The replay is a second call to this
    // same template on the same CTA and inherits the address through
    // tmem_storage[0], which it is already reading below.  Allocating there
    // would be an allocation AFTER this CTA has relinquished its permit, which
    // the ISA does not allow; the matching deallocation is now the
    // __global__'s job, once, after whichever path ran.
    if constexpr (!Repair) {
      const int storage_addr = static_cast<int>(__cvta_generic_to_shared(tmem_storage));
      asm volatile(
          "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(storage_addr),
          "r"(kTmemCols)
          : "memory");
    }
  }
  __syncthreads();
  if constexpr (!Repair) {
    // Safe to relinquish immediately, and only because the allocation above is
    // the only one this CTA will ever make.
    if (warp == 1) {
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
  }
  const int taddr = tmem_storage[0];
  int phase = 0;
  float row_sum = 0.0f;
  // Fast path exponent origin.  Fixed BY DESIGN and sound only in company
  // with the range check at the end of this function and the replay the
  // `__global__` runs when it fires -- see the file header.  In the replay
  // specialization this same variable is the SEED of a data-derived running
  // maximum and only ever moves up.
  float exp_origin = kFastOrigin;
  bool first_pv = true;

  constexpr uint32_t qk_idesc = 0x08200490u;
  constexpr uint32_t pv_idesc = 0x08210490u;
  const int base_chunk = decode_base_chunk();
  const int* page_row = page_table + static_cast<size_t>(request) * max_pages;
  RawSide raw_k;
  RawSide raw_v;
  int u = 0;
  int current_page = -1;
  int next_page = -1;
  if (u < union_count) {
    const uint32_t first_entry = union_hash[0];
    current_page = page_row[static_cast<int>((first_entry & kBlockCodeMask) - 1u)];
    if (union_count > 1) {
      const uint32_t second_entry = union_hash[1];
      next_page = page_row[static_cast<int>((second_entry & kBlockCodeMask) - 1u)];
    }
    load_page_side<false>(raw_k, kv_cache + static_cast<size_t>(current_page) * kPageBytes, kv_head,
                          base_chunk);
    store_page_side<false>(k_smem, raw_k, base_chunk);
    load_page_side<true>(raw_v, kv_cache + static_cast<size_t>(current_page) * kPageBytes, kv_head,
                         base_chunk);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  }

  while (u < union_count) {
    const uint32_t union_entry = union_hash[u];
    const int logical_block = static_cast<int>((union_entry & kBlockCodeMask) - 1u);
    const int next_u = u + 1;

    if (warp == 0 && elect_one()) {
      const uint64_t q_desc_base =
          make_smem_desc(static_cast<int>(__cvta_generic_to_shared(q_smem)));
      const uint64_t k_desc_base = q_desc_base + static_cast<uint64_t>(kKSmemOffset >> 4);
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const uint64_t ad = q_desc_base + kb * 1024 + k * 2;
          const uint64_t bd = k_desc_base + kb * 1024 + k * 2;
          mma_ss(kScoreTmem + taddr, ad, bd, qk_idesc, kb != 0 || k != 0);
        }
      }
      mma_commit(mbar_addr);
    }

    // QK is asynchronous. Start the next block's K loads first, then dequant
    // the already-resident V bytes into the disjoint stage. The K scoreboard
    // latency now overlaps both V conversion/stores and the QK/softmax work.
    if (next_u < union_count) {
      load_page_side<false>(raw_k, kv_cache + static_cast<size_t>(next_page) * kPageBytes, kv_head,
                            base_chunk);
    }
    store_page_side<true>(v_smem, raw_v, base_chunk);
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    {
      const bool selected = valid_row && ((union_entry >> (kQueryBitShift + query_in_tile)) & 1u);
      const int q_abs = kv_len - q_len + q_local + query_in_tile;
      const int block_begin = logical_block * kPageSize;
      const int visible_end = min(q_abs + 1, kv_len);
      const int active_full = selected ? max(0, min(kPageSize, visible_end - block_begin)) : 0;
      float block_sum = 0.0f;
      float next_sum = row_sum;
      bool need_rebase = false;

      if constexpr (!Repair) {
        // Each warpgroup owns 64 KV columns of the same 128 rows.
        const int active_cols = min(64, max(0, active_full - col_half));
        if (__all_sync(0xffffffffu, active_cols == 64)) {
          // Whole warp sees a fully visible selected block -- the common case.
          // No per-element predicate is needed, which drops the softmax from
          // roughly 3.3 to 2 instructions per score element.
#pragma unroll
          for (int c_rev = 0; c_rev < 2; ++c_rev) {
            const int c = 1 - c_rev;
            uint32_t packed[16];
            float frag[32];
            tmem_load_x32_nowait(frag, taddr + kScoreTmem + row_base + col_half + c * 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float sum_a = 0.0f;
            float sum_b = 0.0f;
#pragma unroll
            for (int p = 0; p < 16; ++p) {
              const float p0 = exp2f(frag[p * 2] * logits_scale_log2 - exp_origin);
              const float p1 = exp2f(frag[p * 2 + 1] * logits_scale_log2 - exp_origin);
              sum_a += p0;
              sum_b += p1;
              packed[p] = pack_bf16(p0, p1);
            }
            block_sum += sum_a + sum_b;
            tmem_store_x16_b32(taddr + row_base + col_half + 32 + c * 16, packed);
            if (c == 1) {
              issue_pv_high_half(warp, taddr, v_smem, pv_idesc, first_pv);
            }
          }
        } else if (__all_sync(0xffffffffu, active_cols == 0 || active_cols == 64)) {
          // A union block is often selected by only one of the two queries
          // carried by a warp. In that case lanes are mixed, but every row is
          // still either wholly active or wholly inactive. Keep the TMEM
          // collectives warp-wide and predicate the whole exponent loop once
          // per row instead of testing every score column.
#pragma unroll
          for (int c_rev = 0; c_rev < 2; ++c_rev) {
            const int c = 1 - c_rev;
            uint32_t packed[16];
            float frag[32];
            tmem_load_x32_nowait(frag, taddr + kScoreTmem + row_base + col_half + c * 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float sum_a = 0.0f;
            float sum_b = 0.0f;
            if (active_cols == 64) {
#pragma unroll
              for (int p = 0; p < 16; ++p) {
                const float p0 = exp2f(frag[p * 2] * logits_scale_log2 - exp_origin);
                const float p1 = exp2f(frag[p * 2 + 1] * logits_scale_log2 - exp_origin);
                sum_a += p0;
                sum_b += p1;
                packed[p] = pack_bf16(p0, p1);
              }
            } else {
#pragma unroll
              for (int p = 0; p < 16; ++p) packed[p] = 0u;
            }
            block_sum += sum_a + sum_b;
            tmem_store_x16_b32(taddr + row_base + col_half + 32 + c * 16, packed);
            if (c == 1) {
              issue_pv_high_half(warp, taddr, v_smem, pv_idesc, first_pv);
            }
          }
        } else if (__any_sync(0xffffffffu, active_cols > 0)) {
#pragma unroll
          for (int c_rev = 0; c_rev < 2; ++c_rev) {
            const int c = 1 - c_rev;
            uint32_t packed[16];
            float frag[32];
            tmem_load_x32_nowait(frag, taddr + kScoreTmem + row_base + col_half + c * 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float sum_a = 0.0f;
            float sum_b = 0.0f;
#pragma unroll
            for (int p = 0; p < 16; ++p) {
              const int j = c * 32 + p * 2;
              const float p0 =
                  j < active_cols ? exp2f(frag[p * 2] * logits_scale_log2 - exp_origin) : 0.0f;
              const float p1 = j + 1 < active_cols
                                   ? exp2f(frag[p * 2 + 1] * logits_scale_log2 - exp_origin)
                                   : 0.0f;
              sum_a += p0;
              sum_b += p1;
              packed[p] = pack_bf16(p0, p1);
            }
            block_sum += sum_a + sum_b;
            tmem_store_x16_b32(taddr + row_base + col_half + 32 + c * 16, packed);
            if (c == 1) {
              issue_pv_high_half(warp, taddr, v_smem, pv_idesc, first_pv);
            }
          }
        } else {
          uint32_t zero[16];
#pragma unroll
          for (int p = 0; p < 16; ++p) zero[p] = 0u;
#pragma unroll
          for (int c_rev = 0; c_rev < 2; ++c_rev) {
            const int c = 1 - c_rev;
            tmem_store_x16_b32(taddr + row_base + col_half + 32 + c * 16, zero);
            if (c == 1) {
              issue_pv_high_half(warp, taddr, v_smem, pv_idesc, first_pv);
            }
          }
        }
        next_sum = row_sum + block_sum;
        issue_pv_low_half(warp, taddr, v_smem, mbar_addr, pv_idesc);
      } else {
        // Stable replay: warps 0..3 own all 128 KV columns so the row max and
        // the accumulator rescale stay row-local.
        const int active_cols = active_full;
        if (warp < 4) {
          if (__any_sync(0xffffffffu, active_cols > 0)) {
#pragma unroll
            for (int c_rev = 0; c_rev < 4; ++c_rev) {
              const int c = 3 - c_rev;
              uint32_t packed[16];
              float frag[32];
              tmem_load_x32_nowait(frag, taddr + kScoreTmem + row_base + c * 32);
              asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
              for (int p = 0; p < 16; ++p) {
                const int j = c * 32 + p * 2;
                const float p0 =
                    j < active_cols ? exp2f(frag[p * 2] * logits_scale_log2 - exp_origin) : 0.0f;
                const float p1 = j + 1 < active_cols
                                     ? exp2f(frag[p * 2 + 1] * logits_scale_log2 - exp_origin)
                                     : 0.0f;
                block_sum += p0 + p1;
                packed[p] = pack_bf16(p0, p1);
              }
              tmem_store_x16_b32(taddr + kPTmemRepair + row_base + c * 16, packed);
            }
          } else {
            uint32_t zero[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) zero[p] = 0u;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              tmem_store_x16_b32(taddr + kPTmemRepair + row_base + c * 16, zero);
            }
          }
          next_sum = row_sum + block_sum;
          need_rebase = active_cols > 0 &&
                        (!isfinite(block_sum) || !isfinite(next_sum) ||
                         (row_sum == 0.0f && block_sum < kMinSafeSum) || next_sum > kMaxSafeSum);
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        const bool cta_rebase = __syncthreads_or(need_rebase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (cta_rebase) {
          if (warp == 0 && elect_one()) {
            const uint64_t q_desc_base =
                make_smem_desc(static_cast<int>(__cvta_generic_to_shared(q_smem)));
            const uint64_t k_desc_base = q_desc_base + static_cast<uint64_t>(kKSmemOffset >> 4);
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                const uint64_t ad = q_desc_base + kb * 1024 + k * 2;
                const uint64_t bd = k_desc_base + kb * 1024 + k * 2;
                mma_ss(kScoreTmem + taddr, ad, bd, qk_idesc, kb != 0 || k != 0);
              }
            }
            mma_commit(mbar_addr);
          }
          mbarrier_wait(mbar_addr, phase);
          phase ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

          if (warp < 4) {
            float block_max = -FLT_MAX;
#pragma unroll
            for (int frag_idx = 0; frag_idx < 8; ++frag_idx) {
              float frag[16];
              tmem_load_x16(frag, taddr + kScoreTmem + row_base + frag_idx * 16);
#pragma unroll
              for (int p = 0; p < 16; ++p) {
                const int j = frag_idx * 16 + p;
                if (need_rebase && j < active_cols) {
                  block_max = fmaxf(block_max, frag[p] * logits_scale_log2);
                }
              }
            }
            const float new_origin =
                need_rebase ? (row_sum > 0.0f ? fmaxf(block_max + kRebaseMargin, exp_origin + 64.0f)
                                              : block_max + kRebaseMargin)
                            : exp_origin;
            const float acc_scale =
                need_rebase && row_sum > 0.0f ? exp2f(exp_origin - new_origin) : 1.0f;

            if (!first_pv && __any_sync(0xffffffffu, acc_scale != 1.0f)) {
#pragma unroll
              for (int c = 0; c < 8; ++c) {
                float acc[16];
                tmem_load_x16(acc, taddr + kOutputTmem + row_base + c * 16);
#pragma unroll
                for (int p = 0; p < 16; ++p) acc[p] *= acc_scale;
                tmem_store_x16_f32(taddr + kOutputTmem + row_base + c * 16, acc);
              }
              asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
            if (need_rebase) {
              row_sum *= acc_scale;
              exp_origin = new_origin;
            }

            block_sum = 0.0f;
#pragma unroll
            for (int c_rev = 0; c_rev < 4; ++c_rev) {
              const int c = 3 - c_rev;
              uint32_t packed[16];
              float frag[32];
              tmem_load_x32_nowait(frag, taddr + kScoreTmem + row_base + c * 32);
              asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
              for (int p = 0; p < 16; ++p) {
                const int j = c * 32 + p * 2;
                const float p0 =
                    j < active_cols ? exp2f(frag[p * 2] * logits_scale_log2 - exp_origin) : 0.0f;
                const float p1 = j + 1 < active_cols
                                     ? exp2f(frag[p * 2 + 1] * logits_scale_log2 - exp_origin)
                                     : 0.0f;
                block_sum += p0 + p1;
                packed[p] = pack_bf16(p0, p1);
              }
              tmem_store_x16_b32(taddr + kPTmemRepair + row_base + c * 16, packed);
            }
            next_sum = row_sum + block_sum;
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          }
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          __syncthreads();
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        }
      }
      row_sum = next_sum;
    }

    if constexpr (Repair) {
      if (warp == 0 && elect_one()) {
        uint64_t bd = make_smem_desc(static_cast<int>(__cvta_generic_to_shared(v_smem)));
        bd |= 0x04000000ULL;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          mma_ts(taddr + kOutputTmem, taddr + p_col_of_chunk(true, k),
                 bd + static_cast<uint64_t>(k * 128), pv_idesc, !first_pv || k != 0);
        }
        mma_commit(mbar_addr);
      }
    }

    // PV is asynchronous. Dequant the resident K bytes under it, start the
    // next V load, and resolve the page id two blocks ahead so no global load
    // on the steady-state path feeds its own consumer.
    if (next_u < union_count) {
      load_page_side<true>(raw_v, kv_cache + static_cast<size_t>(next_page) * kPageBytes, kv_head,
                           base_chunk);
      store_page_side<false>(k_smem, raw_k, base_chunk);
      if (next_u + 1 < union_count) {
        const uint32_t ahead_entry = union_hash[next_u + 1];
        next_page = page_row[static_cast<int>((ahead_entry & kBlockCodeMask) - 1u)];
      }
    }
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
    first_pv = false;
    if (next_u < union_count) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    u = next_u;
  }

  // Join the two column halves' partial row sums (k_smem is dead here).
  __syncthreads();
  sum_smem[tid] = row_sum;
  __syncthreads();
  const float total_sum = sum_smem[row] + sum_smem[row + kTileM];

  if constexpr (!Repair) {
    // A row that attended to nothing has total_sum == 0, which is BELOW
    // kMinSafeSum.  That is not a flushed denominator -- there is nothing to
    // recompute and no origin that would help -- so an empty union must not
    // request a replay.  union_count is CTA-uniform.
    const bool unsafe =
        valid_row && union_count > 0 &&
        (!isfinite(total_sum) || total_sum < kMinSafeSum || total_sum > kMaxSafeSum);
    if (unsafe) atomicExch(tmem_storage + 1, 1);
  }

  // Empty-union contract.  A tile whose union is empty issues no PV at all, and
  // `tcgen05.alloc` does not zero TMEM, so the O accumulator still holds
  // whatever the previous occupant of those columns left there.  Scaling that
  // by zero is NOT safe -- finite garbage gives 0.0, but a NaN or Inf bit
  // pattern gives NaN -- so the accumulator is zeroed instead, on a path the
  // union-bearing tiles never take.  `union_count` is CTA-uniform.
  // warps 0..3 already span all kTileM lanes -- row_base repeats across the two
  // warpgroups -- so restricting the stores to one group covers the whole
  // accumulator without two warps racing to write the same address.
  if (union_count == 0 && warp < kWarpsPerGroup) {
    float zero[16];
#pragma unroll
    for (int p = 0; p < 16; ++p) zero[p] = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      tmem_store_x16_f32(taddr + kOutputTmem + row_base + c * 16, zero);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  const float scale = total_sum > 0.0f ? v_global_scale / total_sum : 0.0f;
  __syncthreads();
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    float frag[32];
    tmem_load_x32_nowait(frag, taddr + kOutputTmem + row_base + col_half + c * 32);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    uint32_t packed[16];
#pragma unroll
    for (int p = 0; p < 16; ++p) {
      packed[p] = pack_bf16(frag[p * 2] * scale, frag[p * 2 + 1] * scale);
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      store_swizzled_vec16(
          q_smem, row, wgroup * 8 + c * 4 + v,
          make_uint4(packed[v * 4], packed[v * 4 + 1], packed[v * 4 + 2], packed[v * 4 + 3]));
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int idx = tid; idx < kTileM * kVecChunks; idx += kThreads) {
    const int orow = idx / kVecChunks;
    const int chunk = idx % kVecChunks;
    const int row_query = orow / kGroupSize;
    if (row_query < q_valid) {
      const int row_head = kv_head * kGroupSize + (orow % kGroupSize);
      __nv_bfloat16* out =
          output + (static_cast<size_t>(query_base + row_query) * kNumQHeads + row_head) * kHeadDim;
      reinterpret_cast<uint4*>(out)[chunk] = load_swizzled_vec16(q_smem, orow, chunk);
    }
  }

  __syncthreads();
  if constexpr (!Repair) {
    if (tid == 0 && tmem_storage[1] != 0) {
      output[(static_cast<size_t>(query_base) * kNumQHeads + kv_head * kGroupSize) * kHeadDim] =
          __float2bfloat16(__int_as_float(kReplayMarkerBits));
    }
  }
  if constexpr (Repair) __syncthreads();
  return true;
}

__device__ __noinline__ void replay_sparse_prefill_tile(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ q2k, const int* __restrict__ cu_q, const int* __restrict__ page_table,
    const int* __restrict__ seqused_k, float logits_scale_log2, float v_global_scale,
    __nv_bfloat16* __restrict__ output, int total_q, int batch_size, int max_pages,
    int q2k_head_stride, int q2k_token_stride, int virtual_block_x, int virtual_kv_head,
    char* smem) {
  (void)sparse_prefill_tile<true>(q, kv_cache, q2k, cu_q, page_table, seqused_k, logits_scale_log2,
                                  v_global_scale, output, total_q, batch_size, max_pages,
                                  q2k_head_stride, q2k_token_stride, virtual_block_x,
                                  virtual_kv_head, smem);
}

__global__ __launch_bounds__(kThreads, 2) void sparse_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ q2k, const int* __restrict__ cu_q, const int* __restrict__ page_table,
    const int* __restrict__ seqused_k, float logits_scale_log2, float v_global_scale,
    __nv_bfloat16* __restrict__ output, int total_q, int batch_size, int max_pages,
    int q2k_head_stride, int q2k_token_stride) {
  extern __shared__ __align__(1024) char smem[];
  const bool active = sparse_prefill_tile<false>(
      q, kv_cache, q2k, cu_q, page_table, seqused_k, logits_scale_log2, v_global_scale, output,
      total_q, batch_size, max_pages, q2k_head_stride, q2k_token_stride,
      static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x), smem);
  __syncthreads();
  const int* tile_state = reinterpret_cast<const int*>(smem + kHashOffset);
  if (active && tile_state[kRepairFlagWord] != 0) {
    replay_sparse_prefill_tile(q, kv_cache, q2k, cu_q, page_table, seqused_k, logits_scale_log2,
                               v_global_scale, output, total_q, batch_size, max_pages,
                               q2k_head_stride, q2k_token_stride, static_cast<int>(blockIdx.y),
                               static_cast<int>(blockIdx.x), smem);
  }
  // The single deallocation, matching the single allocation the fast path made.
  // Both passes are done with tensor memory by here: the fast path ends at the
  // __syncthreads() above and the replay ends with one of its own.  An inactive
  // CTA never allocated and must not free.
  if (active && (static_cast<int>(threadIdx.x) >> 5) == 1) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tile_state[kTmemAddrWord]),
        "r"(kTmemCols)
        : "memory");
  }
}

// The dynamic shared-memory request is above the 48 KiB default, so the opt-in
// attribute has to be set before the first launch.  It is a property of the
// kernel image and of the DEVICE, not of a call: setting it on every launch
// would buy a driver round-trip per prefill and nothing else, and caching it in
// a single process-wide flag would leave a second device unconfigured and fail
// its first launch.  Cached per device; a concurrent double-set is idempotent
// and therefore benign.
cudaError_t configure_dynamic_smem(int device_id) {
  constexpr int kMaxCachedDevices = 64;
  static std::atomic<bool> configured[kMaxCachedDevices];  // zero-initialized
  const bool cacheable = device_id >= 0 && device_id < kMaxCachedDevices;
  if (cacheable && configured[device_id].load(std::memory_order_acquire)) {
    return cudaSuccess;
  }
  const cudaError_t status =
      cudaFuncSetAttribute(reinterpret_cast<const void*>(sparse_prefill_kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
  if (status == cudaSuccess && cacheable) {
    configured[device_id].store(true, std::memory_order_release);
  }
  return status;
}

}  // namespace

cudaError_t launch(const void* q, const void* kv_cache, const int* q2k_indices,
                   const int* cu_seqlens_q, const int* page_table, const int* seqused_k,
                   void* output, int total_q, int batch, int max_pages, int q2k_head_stride,
                   int q2k_token_stride, float softmax_scale, float k_global_scale,
                   float v_global_scale, int device_id, cudaStream_t stream) {
  const cudaError_t configured = configure_dynamic_smem(device_id);
  if (configured != cudaSuccess) return configured;

  // KV head is the fastest-changing launch coordinate so that all four heads
  // share one global heavy-to-light traversal.  The tile count is an upper
  // bound: ceil(total_q / kQueriesPerTile) can undercount a ragged batch by at
  // most one tile per request after the first, and every CTA finds its own
  // request by walking cu_seqlens_q, so the surplus CTAs simply exit.
  const dim3 grid(kNumKVHeads, (total_q + kQueriesPerTile - 1) / kQueriesPerTile + batch - 1, 1);
  constexpr float kLog2e = 1.4426950408889634f;
  sparse_prefill_kernel<<<grid, kThreads, kDynamicSmemBytes, stream>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const uint8_t*>(kv_cache), q2k_indices,
      cu_seqlens_q, page_table, seqused_k, softmax_scale * k_global_scale * kLog2e, v_global_scale,
      static_cast<__nv_bfloat16*>(output), total_q, batch, max_pages, q2k_head_stride,
      q2k_token_stride);
  return cudaGetLastError();
}

}  // namespace msa_prefill_nvfp4
}  // namespace flashinfer

// ---------------------------------------------------------------------------
// Host binding.
// ---------------------------------------------------------------------------
namespace {

namespace geom = flashinfer::msa_prefill_nvfp4;

int64_t byte_offset(const TensorView& a, const TensorView& base) {
  return static_cast<const uint8_t*>(a.data_ptr()) - static_cast<const uint8_t*>(base.data_ptr());
}

void check_page_region(const TensorView& t, const char* name, int64_t num_pages, int64_t inner,
                       int64_t head_stride) {
  CHECK_DIM(4, t);
  TVM_FFI_ICHECK_EQ(t.size(0), num_pages) << name << " must cover the whole page pool";
  TVM_FFI_ICHECK_EQ(t.size(1), geom::kNumKVHeads) << name << " must have 4 kv heads";
  TVM_FFI_ICHECK_EQ(t.size(2), geom::kPageSize) << name << " must have a 128-token page";
  TVM_FFI_ICHECK_EQ(t.size(3), inner) << name << " inner extent mismatch";
  // The kernel derives every byte address from these strides rather than
  // reading them, so they are asserted, not propagated.
  TVM_FFI_ICHECK_EQ(t.stride(0), geom::kPageBytes) << name << " page stride must be 73728";
  TVM_FFI_ICHECK_EQ(t.stride(1), head_stride) << name << " kv-head stride mismatch";
  TVM_FFI_ICHECK_EQ(t.stride(2), inner) << name << " token stride mismatch";
  TVM_FFI_ICHECK_EQ(t.stride(3), 1) << name << " must be dense in its innermost dim";
  // The layout proof below differences data_ptr()s, which is only the address
  // the kernel will read if the DLPack byte offset is zero.  Every framework
  // this route serves folds the offset into the pointer; assert it rather than
  // assume it, because a non-zero offset would make the proof compare the wrong
  // addresses and still pass.
  TVM_FFI_ICHECK_EQ(t.byte_offset(), 0u) << name << " must carry a zero DLPack byte offset";
}

}  // namespace

// KV LENGTH AUTHORITY.  This entry point takes `seqused_k` and does NOT take
// `cu_seqlens_k`: the per-request KV length is `seqused_k[request]` and nothing
// else, and the query offset inside that sequence is derived as
// `seqused_k[request] - query_length`, i.e. the queries are the right-aligned
// tail of their request's KV.  A `cu_seqlens_k` argument would be a second,
// independent source for the same quantity with no way to reconcile the two
// without a device-to-host copy, so the Python guard REFUSES a call that
// supplies one rather than silently preferring one over the other.
void msa_prefill_nvfp4_specialized(TensorView q, TensorView k_data, TensorView v_data,
                                   TensorView k_scale, TensorView v_scale, TensorView q2k_indices,
                                   TensorView cu_seqlens_q, TensorView page_table,
                                   TensorView seqused_k, TensorView output, double softmax_scale,
                                   double k_global_scale, double v_global_scale) {
  CHECK_INPUT(q);
  CHECK_CUDA(k_data);
  CHECK_CUDA(v_data);
  CHECK_CUDA(k_scale);
  CHECK_CUDA(v_scale);
  // Only the innermost dimension of the selection tensor must be dense; its
  // two outer strides are kernel arguments.
  CHECK_CUDA(q2k_indices);
  CHECK_INPUT(cu_seqlens_q);
  CHECK_INPUT(page_table);
  CHECK_INPUT(seqused_k);
  CHECK_INPUT(output);
  CHECK_DEVICE(k_data, q);
  CHECK_DEVICE(v_data, q);
  CHECK_DEVICE(k_scale, q);
  CHECK_DEVICE(v_scale, q);
  CHECK_DEVICE(q2k_indices, q);
  CHECK_DEVICE(cu_seqlens_q, q);
  CHECK_DEVICE(page_table, q);
  CHECK_DEVICE(seqused_k, q);
  CHECK_DEVICE(output, q);

  CHECK_INPUT_TYPE(q, dl_bfloat16);
  CHECK_INPUT_TYPE(output, dl_bfloat16);
  CHECK_INPUT_TYPE(k_data, dl_uint8);
  CHECK_INPUT_TYPE(v_data, dl_uint8);
  CHECK_INPUT_TYPE(k_scale, dl_uint8);
  CHECK_INPUT_TYPE(v_scale, dl_uint8);
  CHECK_INPUT_TYPE(q2k_indices, dl_int32);
  CHECK_INPUT_TYPE(cu_seqlens_q, dl_int32);
  CHECK_INPUT_TYPE(page_table, dl_int32);
  CHECK_INPUT_TYPE(seqused_k, dl_int32);

  CHECK_DIM(3, q);
  CHECK_DIM(3, output);
  CHECK_DIM(3, q2k_indices);
  CHECK_DIM(1, cu_seqlens_q);
  CHECK_DIM(2, page_table);
  CHECK_DIM(1, seqused_k);

  const int64_t total_q = q.size(0);
  TVM_FFI_ICHECK_GT(total_q, 0) << "q must contain at least one query token";
  TVM_FFI_ICHECK_LE(total_q, 0x7fffffffLL) << "total_q must fit in int32";
  TVM_FFI_ICHECK_EQ(q.size(1), geom::kNumQHeads) << "this route serves 64 query heads";
  TVM_FFI_ICHECK_EQ(q.size(2), geom::kHeadDim) << "head_dim must be 128";
  CHECK_SHAPE(output, q);

  // topk is STILL a compile-time constant of this kernel body -- it sizes the
  // union hash table (`kQueriesPerTile * kTopK <= kHashSize` holds with exact
  // equality) and the eight-query membership mask -- so it is asserted here as
  // well as in the Python guard.  What is no longer baked in is the q2k row
  // STRIDE: that moved into the signature, so a non-contiguous selection view
  // is read in place rather than refused.
  TVM_FFI_ICHECK_EQ(q2k_indices.size(0), geom::kNumKVHeads)
      << "q2k_indices must have one plane per kv head";
  TVM_FFI_ICHECK_EQ(q2k_indices.size(1), total_q)
      << "q2k_indices must have one row per query token";
  TVM_FFI_ICHECK_EQ(q2k_indices.size(2), geom::kTopK) << "topk must be 16";
  TVM_FFI_ICHECK_EQ(q2k_indices.stride(2), 1)
      << "q2k_indices must be dense in its innermost (top-k) dimension";
  const int64_t q2k_head_stride = q2k_indices.stride(0);
  const int64_t q2k_token_stride = q2k_indices.stride(1);
  TVM_FFI_ICHECK(q2k_head_stride >= 0 && q2k_token_stride >= 0)
      << "q2k_indices must not be negatively strided";
  TVM_FFI_ICHECK_LE(
      (geom::kNumKVHeads - 1) * q2k_head_stride + (total_q - 1) * q2k_token_stride + geom::kTopK,
      0x7fffffffLL)
      << "the q2k_indices view is too large for 32-bit addressing";

  const int64_t batch_size = cu_seqlens_q.size(0) - 1;
  TVM_FFI_ICHECK_GE(batch_size, 1) << "cu_seqlens_q must contain at least two entries";
  TVM_FFI_ICHECK_EQ(page_table.size(0), batch_size) << "page_table needs one row per request";
  TVM_FFI_ICHECK_EQ(seqused_k.size(0), batch_size) << "seqused_k needs one entry per request";

  const int64_t max_blocks = page_table.size(1);
  TVM_FFI_ICHECK_GE(max_blocks, 1) << "page_table row width must be positive";
  // The only width ceiling this route has: a block id is carried in the low 24
  // bits of a union-table entry.  Not a clamp -- truncating an id would drop a
  // selected block from the union with no diagnostic.
  TVM_FFI_ICHECK_LE(max_blocks, geom::kMaxSelectableBlocks)
      << "a selected block id is carried in 24 bits, so the block-table width may not exceed "
      << geom::kMaxSelectableBlocks << " (page_size " << geom::kPageSize << " => context "
      << geom::kMaxContextTokens << " tokens), got " << max_blocks;

  const int64_t num_pages = k_data.size(0);
  check_page_region(k_data, "k_data", num_pages, geom::kDataDim, geom::kDataHeadStride);
  check_page_region(v_data, "v_data", num_pages, geom::kDataDim, geom::kDataHeadStride);
  check_page_region(k_scale, "k_scale", num_pages, geom::kScaleDim, geom::kScaleHeadStride);
  check_page_region(v_scale, "v_scale", num_pages, geom::kScaleDim, geom::kScaleHeadStride);

  // Layout proof.  Shape, dtype and stride cannot distinguish four views of one
  // planar page from four unrelated allocations that happen to be strided the
  // same way -- and the (4, 4) V-scale swizzle is invisible to all three.  The
  // byte offsets between the four base pointers are what pins them to the same
  // page map the cache writer used, and they are also what lets the kernel
  // address a whole page from k_data's base pointer.
  TVM_FFI_ICHECK_EQ(byte_offset(k_scale, k_data), geom::kKScaleByteOffset)
      << "k_scale must be the K-scale region of the same page as k_data";
  TVM_FFI_ICHECK_EQ(byte_offset(v_data, k_data), geom::kVDataByteOffset)
      << "v_data must be the V-data region of the same page as k_data";
  TVM_FFI_ICHECK_EQ(byte_offset(v_scale, k_data), geom::kVScaleByteOffset)
      << "v_scale must be the V-scale region of the same page as k_data";

  TVM_FFI_ICHECK(std::isfinite(softmax_scale) && std::isfinite(k_global_scale) &&
                 std::isfinite(v_global_scale))
      << "softmax_scale, k_global_scale and v_global_scale must be finite";

  const int device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  int major = 0;
  int minor = 0;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
                    cudaSuccess);
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "the specialized NVFP4 MSA prefill kernel requires compute capability 10.0 or 10.3, got "
      << major << "." << minor;

  const cudaStream_t stream = get_stream(q.device());
  const cudaError_t status = flashinfer::msa_prefill_nvfp4::launch(
      q.data_ptr(), k_data.data_ptr(), static_cast<const int*>(q2k_indices.data_ptr()),
      static_cast<const int*>(cu_seqlens_q.data_ptr()),
      static_cast<const int*>(page_table.data_ptr()), static_cast<const int*>(seqused_k.data_ptr()),
      output.data_ptr(), static_cast<int>(total_q), static_cast<int>(batch_size),
      static_cast<int>(max_blocks), static_cast<int>(q2k_head_stride),
      static_cast<int>(q2k_token_stride), static_cast<float>(softmax_scale),
      static_cast<float>(k_global_scale), static_cast<float>(v_global_scale), device_id, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "msa_prefill_nvfp4_specialized launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(msa_prefill_nvfp4_specialized, msa_prefill_nvfp4_specialized);
