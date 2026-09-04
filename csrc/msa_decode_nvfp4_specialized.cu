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

// Minimax Sparse Attention decode over an NVFP4 paged KV cache, for compute
// capability 10.0 / 10.3, where no NVFP4 MSA route exists otherwise.
//
// TWO INSTANTIATION FAMILIES, ONE CONTRACT
// ----------------------------------------
// `general` is parametric in the query/KV head counts, the GQA group, the page
// size, the top-k width, the block-table width, the batch size, the per-request
// KV length and the query length.  `pinned` is the same algorithm with the
// deployment's geometry (64 query heads, 4 KV heads, head_dim 128, page 128,
// top-k 16, block-table width 128, one query token) resolved at compile time,
// which lets it carry a deeper page pipeline.  Both compute the same function;
// the pinned family is a faster instantiation, never a narrower capability.
//
// The host binding decides between them from the geometry alone, and refuses to
// take the pinned family for anything outside it.  The Python route computes
// the same predicate and passes its answer in; the two must AGREE, and a
// disagreement is a hard error rather than a silent fall to the slower family.
// That is deliberate: the pinned envelope IS the deployment envelope, and a pin
// that quietly stops matching the deployment is a performance regression with
// no failing test attached to it.
//
// Storage contract (one physical page, planar, 73728 B at the deployment's
// geometry; parametric in general):
//
//     [ K data 32768 | K scale 4096 | V data 32768 | V scale 4096 ]
//
// per (page, kv head, token): head_dim/2 packed e2m1 bytes and head_dim/16 e4m3
// block scales, one scale per 16 elements.  K scales are stored linearly; V
// scales are (4, 4)-swizzled inside (token, scale index) so that the scale of
// logical (t, s) lives at ((t / 4) * 4 + s / 2, (s % 2) * 4 + t % 4).  The four
// regions are handed in as four strided views over the same allocation; the
// kernel consumes them in place and never materializes a dense copy.
//
// Compute contract: the QK and PV products run on tcgen05 `kind::f8f6f4` with
// FP32 accumulators.  Native FP4 MMA (`kind::mxf4nvf4`) is deliberately NOT
// used -- the block scales are applied during the e2m1 -> e4m3 dequant so that
// the operand precision matches the uniform-FP8 decode route.
//
// SOFTMAX STABILITY INVARIANT (do not weaken):
//     every exponential argument resolves to a difference against a running row
//     maximum -- `ex2(x + shift)` with `shift = kLog2PScale - new_max` in the
//     pinned family, `ex2(fma(x, ls2, sh))` with `sh = kPreScale -
//     run_max_origin` in the general one, where `run_max_origin` tracks the
//     running maximum within a bounded slack.  A fixed-exponent form such as
//     `ex2(x + kConst)` makes representability depend on the absolute logit and
//     saturates e4m3 once the query aligns with a selected key, which is exactly
//     the regime a trained attention layer selects for.  The prescale is applied
//     ALONGSIDE the maximum, never in place of it, and an entirely masked tile is
//     clamped so that it contributes exactly zero instead of exp2(0).
//
// SELECTION VALIDITY INVARIANT: a selected logical block is used
// only after `blk >= 0`, `blk` inside the request's own block count AND the
// block-table row width, and the physical page it resolves to is inside the
// pool.  Duplicates are suppressed in first-occurrence order -- by
// `__match_any_sync` in the pinned family, by a prior-slot scan in the general
// one -- and the survivors are COMPACTED, so an interior eviction cannot shift a
// later valid page out of the loop.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace msa_decode_nvfp4 {

// ---------------------------------------------------------------------------
// Deployment geometry.  These are the values the host binding and the Python
// dispatch guard assert, the page map the cache writer produces, and the
// compile-time constants the `pinned` family bakes in.  `general` reads every
// one of them at runtime instead.
// ---------------------------------------------------------------------------
constexpr int kNumQHeads = 64;
constexpr int kNumKVHeads = 4;
constexpr int kHeadsPerKV = kNumQHeads / kNumKVHeads;  // 16
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kTopK = 16;
constexpr int kMaxBlocks = 128;  // ceil(max_model_len 16384 / page 128)

constexpr int kDataDim = kHeadDim / 2;                            // 64 packed e2m1 bytes per token
constexpr int kScaleDim = kHeadDim / 16;                          // 8 e4m3 block scales per token
constexpr int kDataHeadStride = kPageSize * kDataDim;             // 8192
constexpr int kScaleHeadStride = kPageSize * kScaleDim;           // 1024
constexpr int kKScaleByteOffset = kNumKVHeads * kDataHeadStride;  // 32768
constexpr int kVDataByteOffset = kKScaleByteOffset + kNumKVHeads * kScaleHeadStride;  // 36864
constexpr int kVScaleByteOffset = kVDataByteOffset + kKScaleByteOffset;               // 69632
constexpr int kPageBytes = kVScaleByteOffset + kNumKVHeads * kScaleHeadStride;        // 73728

// --- tiling relations -------------------------------------------------------
// The relations both kernel bodies assume.  They exist so that a port to
// another geometry fails to COMPILE instead of silently misreading memory.
static_assert(kHeadsPerKV == 16, "one warp owns one query head during softmax");
static_assert(kHeadDim == 128, "MMA K extent and the 128B smem swizzle");
static_assert(kPageSize == 128, "one MMA tile covers exactly one KV page");
static_assert(kTopK == 16, "the pinned selected-block list is one 16-lane ballot");
static_assert(kMaxBlocks == 128, "the pinned block-table row is staged whole in smem");
static_assert(kDataDim * 2 == kHeadDim, "e2m1 packs two elements per byte");
static_assert(kScaleDim * 16 == kHeadDim, "one e4m3 block scale per 16 elements");
static_assert(kPageBytes == 73728, "the planar page the NVFP4 cache writer produces");

// ---------------------------------------------------------------------------
// BF16 -> E4M3, and the only place in this file that names a toolkit-gated
// instruction.  Shared by both instantiation families on purpose: a second copy
// would be a second thing to forget to guard.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t pack_e4m3x2(float low, float high) {
  uint16_t out;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(high), "f"(low));
  return out;
}

// `cvt.rn.satfinite.e4m3x2.bf16x2` does not assemble before CUDA Toolkit 13.1:
// ptxas 13.0 rejects it with "Unexpected instruction types specified for 'cvt'"
// for a .b16 destination, for a .b32 destination, and for the
// `__nv_fp8x2_e4m3(__nv_bfloat162)` intrinsic alike -- the intrinsic lowers to
// the same instruction. This is the same class of toolkit gap that
// `vec_dtypes.cuh` already guards for the FP4 conversions and that the sibling
// NVFP4 prefill unit guards for `cvt.rn.bf16x2.{e2m1x2,e4m3x2}`; only the
// version differs, because it is a different instruction (13.1 here, 13.2
// there). Override the macro to exercise either path in a test.
#ifndef FLASHINFER_MSA_NVFP4_NATIVE_BF16_TO_E4M3
#if (defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
     ((__CUDACC_VER_MAJOR__ > 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)))
#define FLASHINFER_MSA_NVFP4_NATIVE_BF16_TO_E4M3 1
#else
#define FLASHINFER_MSA_NVFP4_NATIVE_BF16_TO_E4M3 0
#endif
#endif

// BF16 pair -> E4M3 pair, low half to low byte.
//
// The fallback is BIT-EXACT, not an approximation. BF16 is FP32 truncated to 7
// explicit significand bits and carries the same 8-bit exponent, so widening
// BF16 -> FP32 is lossless for every one of the 2^16 bit patterns, NaNs,
// infinities, denormals and signed zeros included. `cvt.rn.satfinite.e4m3x2.f32`
// then applies the same round-to-nearest-even and the same saturation to the
// same real value that the native instruction would have seen, so the two paths
// agree bit-for-bit by construction rather than by tolerance. Checked
// exhaustively over the whole BF16 domain, not sampled.
__device__ __forceinline__ uint16_t pack_bf16x2_to_e4m3x2(uint32_t src) {
#if FLASHINFER_MSA_NVFP4_NATIVE_BF16_TO_E4M3
  uint16_t out;
  asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(out) : "r"(src));
  return out;
#else
  const float low = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(src)));
  const float high = __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(src >> 16)));
  return pack_e4m3x2(low, high);
#endif
}

// ---------------------------------------------------------------------------
// Is this call inside the pinned family's envelope?
//
// Host-side, pure, and duplicated in flashinfer/msa_ops/_nvfp4_decode_sm100.py
// as `selects_pinned_path`.  The duplication is the point: the Python copy is
// what a GPU-free serviceability preflight can execute over every coordinate a
// serving run reaches, and the binding cross-checks the two so the copies
// cannot drift apart unnoticed.
// ---------------------------------------------------------------------------
struct PinnedEnvelope {
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int page_size;
  int topk;
  int max_blocks;
  int seqlen_q;
  int total_q;
  int num_pages;
};

inline bool selects_pinned_path(const PinnedEnvelope& e) {
  // A batch of 32 whose page pool is smaller than 32 rows per request is an
  // out-of-deployment shape (the serving pool is ~2.3M pages); the pinned
  // family's short-cache instantiation is not tuned for it, so it is routed to
  // the general one rather than guessed at.
  const bool short_batch32 = e.total_q == 32 && e.num_pages < 32 * e.total_q;
  return e.num_q_heads == kNumQHeads && e.num_kv_heads == kNumKVHeads && e.head_dim == kHeadDim &&
         e.page_size == kPageSize && e.topk == kTopK && e.max_blocks == kMaxBlocks &&
         e.seqlen_q == 1 && !short_batch32;
}

namespace general {

constexpr int kHeadDim = 128;
constexpr int kHeadCapacity = 16;
constexpr int kTokenTile = 128;
// Every selection slot is one lane of warp 0's `__ballot_sync`, and the two
// compaction arrays below are sized by it.  A 33rd slot would be owned by a
// thread outside that ballot and would be dropped silently, so this is the
// runtime top-k ceiling the binding refuses above -- a structural bound of the
// compaction, not a policy.
constexpr int kSelectedCapacity = 32;
static_assert(kSelectedCapacity <= 32, "one selection slot per lane of warp 0's ballot");
// Staging capacity for the request's block-table row.  Purely a shared-memory
// budget: any row longer than this falls back to the direct global lookup, so
// max_blocks stays a runtime value.
constexpr int kPageTableCapacity = 128;
constexpr int kThreads = 512;
constexpr int kMmaAlignment = 1 << 10;
constexpr float kLog2e = 1.4426950408889634f;

// Softmax origin policy.  The exponential origin tracks the running row maximum
// with a fixed prescale: p = exp2(z - origin + kPreScale).  kPreScale puts a
// probability that sits exactly on the origin at 2^5 = 32; e4m3 reaches down to
// 2^-9, so 14 binades of the row still resolve below it.  kOriginSlack is the
// guard band the origin is allowed to trail the running maximum by before it is
// bumped, and it buys the rescale-free path: p can never exceed
// 2^(kPreScale + kOriginSlack) = 2^8.5 = 362 < e4m3's 448 ceiling, so the
// operand cannot saturate however the running maximum evolves, while a tile
// whose maximum only creeps up costs no accumulator rescale at all.
constexpr float kPreScale = 5.0f;
constexpr float kOriginSlack = 3.5f;
constexpr float kNegRunMaxOrigin = -3.0e38f;

// Two KV pages are folded into a single QK/PV MMA round trip.
constexpr int kPagesPerMma = 2;
static_assert(kPagesPerMma == 2, "two KV pages are folded into one QK/PV MMA round trip");

constexpr int kSStride = 2 * kTokenTile + 4;
constexpr int kOStride = kHeadDim + 4;
constexpr int kKvTileBytes = kTokenTile * kHeadDim;
constexpr int kProbTileBytes = kHeadCapacity * kTokenTile;

// Shared-memory map. These are MMA tile dimensions, not cache geometry.
constexpr int kOffK = 0;
constexpr int kOffV = kOffK + 2 * kKvTileBytes;
constexpr int kOffP = kOffV + 2 * kKvTileBytes;
constexpr int kOffQ = kOffP + 2 * kProbTileBytes;
constexpr int kOffS = kOffQ + kProbTileBytes;
constexpr int kOffO = kOffS + kHeadCapacity * kSStride * 4;
constexpr int kSmemBytes = kOffO + kHeadCapacity * kOStride * 4;

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "elect.sync _|%%p, %1;\n\t"
      "@%%p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xffffffff));
  return pred;
}

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3ffffULL) >> 4ULL;
}

// 128B swizzle, rows of 128 bytes, SBO = 8 rows.
__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  constexpr int kSBO = 8 * 128;
  return desc_encode(addr) | (desc_encode(kSBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void mbarrier_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "WAIT%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra.uni WAIT%=;\n\t"
      "}" ::"r"(addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ uint32_t map_rank(uint32_t addr, int rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

__device__ __forceinline__ float ld_dsmem(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ float2 ld_dsmem_v2(uint32_t addr) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
  return v;
}

__device__ __forceinline__ float4 ld_dsmem_v4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void mma_f8(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
                                       int accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(accumulate));
}

__device__ __forceinline__ void commit_mma(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tmem_ld16(int addr, float (&v)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
        "=f"(v[14]), "=f"(v[15])
      : "r"(addr));
}

// Half-width TMEM accesses.  Eight live fp32 results per access instead of
// sixteen keeps the 64-register budget of a 512-thread / 2-CTA launch and
// removes the local-memory spill the x16 form provokes.
__device__ __forceinline__ void tmem_ld8(int addr, float (&v)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_st8(int addr, const float (&v)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], "
      "{%0,%1,%2,%3,%4,%5,%6,%7};" ::"f"(v[0]),
      "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "r"(addr));
}

__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }

__device__ __forceinline__ void tmem_st16(int addr, const float (&v)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"f"(v[0]),
      "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]),
      "f"(v[9]), "f"(v[10]), "f"(v[11]), "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]), "r"(addr));
}

__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }

__device__ __forceinline__ float ex2(float x) { return exp2f(x); }

__device__ __forceinline__ float warp_max(float x) {
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(x));
  return r;
}

union F32x2Bits {
  float2 f;
  uint64_t u;
};

union BF16x2Bits {
  __nv_bfloat162 b;
  uint32_t u;
};

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
  F32x2Bits pa, pb, pr;
  pa.f = a;
  pb.f = b;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u));
  return pr.f;
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
  F32x2Bits pa, pb, pr;
  pa.f = a;
  pb.f = b;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u));
  return pr.f;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
  F32x2Bits pa, pb, pc, pr;
  pa.f = a;
  pb.f = b;
  pc.f = c;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u), "l"(pc.u));
  return pr.f;
}

__device__ __forceinline__ uint64_t load_stream_u64(const void* ptr) {
  uint64_t v;
  asm("ld.global.L1::no_allocate.L2::256B.u64 %0, [%1];" : "=l"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ ulonglong2 load_stream_u128(const void* ptr) {
  ulonglong2 v;
  asm("ld.global.L1::no_allocate.L2::256B.v2.u64 {%0, %1}, [%2];"
      : "=l"(v.x), "=l"(v.y)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t load_stream_u16(const void* ptr) {
  uint16_t v;
  asm("ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];" : "=h"(v) : "l"(ptr));
  return static_cast<uint32_t>(v);
}

__device__ __forceinline__ uint32_t load_stream_u8(const void* ptr) {
  uint32_t v;
  asm("ld.global.L1::no_allocate.L2::256B.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

// The V block scales are (4,4)-swizzled, so the two scales of one token sit in
// the same byte lane of the two halves of a 64-bit quad.  One prmt with a
// thread-invariant selector replaces the shift/mask/or sequence.
__device__ __forceinline__ uint32_t pick_v_scales(uint64_t quad, uint32_t sel) {
  const uint32_t lo = static_cast<uint32_t>(quad);
  const uint32_t hi = static_cast<uint32_t>(quad >> 32);
  uint32_t out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(lo), "r"(hi), "r"(sel));
  return out;
}

// 128B-swizzled byte offset inside a tile whose rows are 128 bytes wide.
__device__ __forceinline__ int swz(int row, int col) {
  return row * 128 + (col ^ ((row & 7) << 4));
}

struct F16ScalePair {
  uint32_t lo;
  uint32_t hi;
};

// Convert two adjacent E4M3 scales together, then broadcast each converted
// half. Packed conversion preserves low-byte -> low-half ordering.
__device__ __forceinline__ F16ScalePair convert_scale_pair(uint32_t e4m3x2) {
  F16ScalePair out;
  asm volatile(
      "{\n\t"
      ".reg .b32 both;\n\t"
      ".reg .b16 packed, s0, s1;\n\t"
      "mov.b32 {packed, _}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 both, packed;\n\t"
      "mov.b32 {s0, s1}, both;\n\t"
      "mov.b32 %0, {s0, s0};\n\t"
      "mov.b32 %1, {s1, s1};\n\t"
      "}"
      : "=r"(out.lo), "=r"(out.hi)
      : "r"(e4m3x2));
  return out;
}

// 16 packed e2m1 values * one broadcast f16x2 block scale -> 16 e4m3 values.
__device__ __forceinline__ uint4 dequant_fp4x16_to_fp8(uint64_t src, uint32_t sf_f16x2) {
  uint4 out;
  asm volatile(
      "{\n\t"
      ".reg .b32 lo, hi, h0, h1, h2, h3;\n\t"
      ".reg .b16 e0, e1, e2, e3;\n\t"
      ".reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b64 {lo, hi}, %4;\n\t"
      "mov.b32 {b0, b1, b2, b3}, lo;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
      "mul.rn.f16x2 h0, h0, %5;\n\t"
      "mul.rn.f16x2 h1, h1, %5;\n\t"
      "mul.rn.f16x2 h2, h2, %5;\n\t"
      "mul.rn.f16x2 h3, h3, %5;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
      "mov.b32 %0, {e0, e1};\n\t"
      "mov.b32 %1, {e2, e3};\n\t"
      "mov.b32 {b0, b1, b2, b3}, hi;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
      "mul.rn.f16x2 h0, h0, %5;\n\t"
      "mul.rn.f16x2 h1, h1, %5;\n\t"
      "mul.rn.f16x2 h2, h2, %5;\n\t"
      "mul.rn.f16x2 h3, h3, %5;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
      "mov.b32 %2, {e0, e1};\n\t"
      "mov.b32 %3, {e2, e3};\n\t"
      "}"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(src), "r"(sf_f16x2));
  return out;
}

// Register staging for one MMA batch (two pages) of one tensor.
struct RawBatch {
  uint64_t d[4];
  uint32_t sf;  // four e4m3 block scales, one per byte
};

struct CacheGeometry {
  int page_size;
  int data_head_stride;
  int scale_head_stride;
};

// Each thread owns two (token, scale-group) tiles in each MMA token tile.
template <bool kFullPageTile>
__device__ __forceinline__ void load_k_batch(const uint8_t* __restrict__ k_data,
                                             const uint8_t* __restrict__ k_scale, uint64_t off0,
                                             uint64_t off1, int hkv, int tid,
                                             const CacheGeometry& g, RawBatch& r) {
  const uint64_t offsets[2] = {off0, off1};
  const int tok = tid >> 2;
  const int pair = tid & 3;
  uint32_t sf = 0;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    if constexpr (!kFullPageTile) {
      if (tok >= g.page_size) {
        r.d[p * 2] = 0;
        r.d[p * 2 + 1] = 0;
        continue;
      }
    }
    const size_t base = static_cast<size_t>(offsets[p]);
    const uint8_t* db = k_data + base + hkv * g.data_head_stride;
    const uint8_t* sb = k_scale + base + hkv * g.scale_head_stride;
    const ulonglong2 data = load_stream_u128(db + tok * 64 + pair * 16);
    r.d[p * 2] = data.x;
    r.d[p * 2 + 1] = data.y;
    sf |= load_stream_u16(sb + tok * 8 + pair * 2) << (16 * p);
  }
  r.sf = sf;
}

template <bool kFullPageTile>
__device__ __forceinline__ void load_v_batch(const uint8_t* __restrict__ v_data,
                                             const uint8_t* __restrict__ v_scale, uint64_t off0,
                                             uint64_t off1, int hkv, int tid,
                                             const CacheGeometry& g, RawBatch& r) {
  const uint64_t offsets[2] = {off0, off1};
  const int tok = tid >> 2;
  const int pair = tid & 3;
  const uint32_t vsel = (tok & 3) | ((4 + (tok & 3)) << 4);
  uint32_t sf = 0;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    if constexpr (!kFullPageTile) {
      if (tok >= g.page_size) {
        r.d[p * 2] = 0;
        r.d[p * 2 + 1] = 0;
        continue;
      }
    }
    const size_t base = static_cast<size_t>(offsets[p]);
    const uint8_t* db = v_data + base + hkv * g.data_head_stride;
    const uint8_t* sb = v_scale + base + hkv * g.scale_head_stride;
    const ulonglong2 data = load_stream_u128(db + tok * 64 + pair * 16);
    r.d[p * 2] = data.x;
    r.d[p * 2 + 1] = data.y;
    const int scale_base = (tok >> 2) * 32 + pair * 8;
    const uint64_t scale_quad = load_stream_u64(sb + scale_base);
    sf |= (pick_v_scales(scale_quad, vsel) & 0xffffu) << (16 * p);
  }
  r.sf = sf;
}

template <bool kFullPageTile>
__device__ __forceinline__ void load_k_one(const uint8_t* __restrict__ k_data,
                                           const uint8_t* __restrict__ k_scale, uint64_t page_off,
                                           int hkv, int tid, const CacheGeometry& g, RawBatch& r) {
  const int tok = tid >> 2;
  const int pair = tid & 3;
  if constexpr (!kFullPageTile) {
    if (tok >= g.page_size) {
      r.d[0] = 0;
      r.d[1] = 0;
      r.sf = 0;
      return;
    }
  }
  const size_t base = static_cast<size_t>(page_off);
  const uint8_t* db = k_data + base + hkv * g.data_head_stride;
  const uint8_t* sb = k_scale + base + hkv * g.scale_head_stride;
  const ulonglong2 data = load_stream_u128(db + tok * 64 + pair * 16);
  r.d[0] = data.x;
  r.d[1] = data.y;
  r.sf = load_stream_u16(sb + tok * 8 + pair * 2);
}

template <bool kFullPageTile>
__device__ __forceinline__ void load_v_one(const uint8_t* __restrict__ v_data,
                                           const uint8_t* __restrict__ v_scale, uint64_t page_off,
                                           int hkv, int tid, const CacheGeometry& g, RawBatch& r) {
  const int tok = tid >> 2;
  const int pair = tid & 3;
  const uint32_t vsel = (tok & 3) | ((4 + (tok & 3)) << 4);
  if constexpr (!kFullPageTile) {
    if (tok >= g.page_size) {
      r.d[0] = 0;
      r.d[1] = 0;
      r.sf = 0;
      return;
    }
  }
  const size_t base = static_cast<size_t>(page_off);
  const uint8_t* db = v_data + base + hkv * g.data_head_stride;
  const uint8_t* sb = v_scale + base + hkv * g.scale_head_stride;
  const ulonglong2 data = load_stream_u128(db + tok * 64 + pair * 16);
  r.d[0] = data.x;
  r.d[1] = data.y;
  const int scale_base = (tok >> 2) * 32 + pair * 8;
  const uint64_t scale_quad = load_stream_u64(sb + scale_base);
  r.sf = pick_v_scales(scale_quad, vsel) & 0xffffu;
}

template <bool kOddTail, bool kFullPageTile>
__device__ __forceinline__ void load_k_pages(const uint8_t* k_data, const uint8_t* k_scale,
                                             uint64_t off0, uint64_t off1, bool has2, int hkv,
                                             int tid, const CacheGeometry& g, RawBatch& r) {
  if constexpr (kOddTail) {
    if (!has2) return load_k_one<kFullPageTile>(k_data, k_scale, off0, hkv, tid, g, r);
  }
  load_k_batch<kFullPageTile>(k_data, k_scale, off0, off1, hkv, tid, g, r);
}

template <bool kOddTail, bool kFullPageTile>
__device__ __forceinline__ void load_v_pages(const uint8_t* v_data, const uint8_t* v_scale,
                                             uint64_t off0, uint64_t off1, bool has2, int hkv,
                                             int tid, const CacheGeometry& g, RawBatch& r) {
  if constexpr (kOddTail) {
    if (!has2) return load_v_one<kFullPageTile>(v_data, v_scale, off0, hkv, tid, g, r);
  }
  load_v_batch<kFullPageTile>(v_data, v_scale, off0, off1, hkv, tid, g, r);
}

__device__ __forceinline__ void store_batch(uint8_t* dst, int tid, const RawBatch& r, int npage) {
  const int tok = tid >> 2;
  const int pair = tid & 3;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    if (p >= npage) continue;
    uint8_t* base = dst + p * kKvTileBytes;
    const F16ScalePair sf = convert_scale_pair(r.sf >> (16 * p));
    const int col = pair * 32;
    *reinterpret_cast<uint4*>(base + swz(tok, col)) = dequant_fp4x16_to_fp8(r.d[p * 2], sf.lo);
    *reinterpret_cast<uint4*>(base + swz(tok, col + 16)) =
        dequant_fp4x16_to_fp8(r.d[p * 2 + 1], sf.hi);
  }
}

template <int kSplit, bool kFastSQ1, bool kOddTail, bool kFullPageTile>
__global__ __launch_bounds__(kThreads, 2) void kernel_msa_decode_nvfp4_kv_paged(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_data,
    const uint8_t* __restrict__ v_data, const uint8_t* __restrict__ k_scale,
    const uint8_t* __restrict__ v_scale, const int* __restrict__ q2k_indices,
    const int* __restrict__ page_table, const int* __restrict__ seqused_k, int q2k_head_stride,
    int q2k_token_stride, int num_pages, int num_q_heads, int num_kv_heads, int heads_per_kv,
    int page_size, int topk, int max_blocks, int page_bytes, int data_head_stride,
    int scale_head_stride, int seqlen_q, int causal, float softmax_scale, float k_global_scale,
    float v_global_scale, __nv_bfloat16* __restrict__ output) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rank = blockIdx.x & (kSplit - 1);
  const int hkv = blockIdx.x / kSplit;
  const int qi = blockIdx.y;
  // Fold runtime-derived KV-head strides into the plane bases once per CTA.
  // The page loaders can then use a zero head index on every prefetch path.
  k_data += static_cast<size_t>(hkv) * data_head_stride;
  v_data += static_cast<size_t>(hkv) * data_head_stride;
  k_scale += static_cast<size_t>(hkv) * scale_head_stride;
  v_scale += static_cast<size_t>(hkv) * scale_head_stride;
  const CacheGeometry cache{page_size, data_head_stride, scale_head_stride};

  extern __shared__ __align__(kMmaAlignment) uint8_t smem[];
  uint8_t* sK = smem + kOffK;
  uint8_t* sV = smem + kOffV;
  uint8_t* sP = smem + kOffP;
  uint8_t* sQ = smem + kOffQ;
  float* sS = reinterpret_cast<float*>(smem + kOffS);
  float* sO = reinterpret_cast<float*>(smem + kOffO);

  __shared__ __align__(8) uint64_t mma_bar;
  __shared__ int tmem_slot;
  __shared__ __align__(8) float2 sStat[kHeadCapacity];
  __shared__ float sInv[kHeadCapacity];
  __shared__ float sAlpha[kHeadCapacity];
  __shared__ float sScale[kHeadCapacity];
  __shared__ uint32_t sRemoteO[kSplit];
  __shared__ int sTokenBase[kSelectedCapacity];
  __shared__ uint64_t sPageOff[kSelectedCapacity];
  __shared__ int sNpages;
  __shared__ int sPT[kPageTableCapacity];

  const int mbar = static_cast<int>(__cvta_generic_to_shared(&mma_bar));
  if (warp == 0 && elect_sync()) {
    mbarrier_init(mbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp == 1) {
    const int slot = static_cast<int>(__cvta_generic_to_shared(&tmem_slot));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot),
                 "r"(64));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  int request, qpos;
  if constexpr (kFastSQ1) {
    request = qi;
    qpos = 0;
  } else {
    request = qi / seqlen_q;
    qpos = qi - request * seqlen_q;
  }
  const int pages_per_cta = (topk + kSplit - 1 - rank) / kSplit;
  const int qrow = tid >> 5;
  const int qcol = (tid & 31) * 4;

  // ---- Dependency level 1: everything the CTA can ask DRAM for at once. ----
  // seqused_k, the selected logical block ids, the request's block-table row and
  // the query rows carry no address dependence on each other, so they are all
  // issued before anything is consumed.  Only the KV pages themselves have to
  // wait, and they now wait on one round trip instead of a chain of three.
  const int seq = seqused_k[request];
  // The selection row for (kv head, q row).  Its two strides are passed rather
  // than derived from `total_q * topk`, so a caller may hand this kernel any
  // layout whose LAST dimension is contiguous -- in particular the transposed
  // view of a token-major (total_q, num_kv_heads, topk) buffer, which is what
  // the MSA indexer produces and what every consumer had to materialise a
  // contiguous copy of before.
  const int* __restrict__ q2k_row = q2k_indices + static_cast<size_t>(hkv) * q2k_head_stride +
                                    static_cast<size_t>(qi) * q2k_token_stride;
  const int blk = tid < pages_per_cta ? q2k_row[rank + tid * kSplit] : -1;
  const int pt_staged = max_blocks < kPageTableCapacity ? max_blocks : kPageTableCapacity;
  if constexpr (kSplit != 8) {
    for (int i = tid; i < pt_staged; i += kThreads) sPT[i] = page_table[request * max_blocks + i];
  }
  const uint64_t qquad =
      qrow < heads_per_kv
          ? *reinterpret_cast<const uint64_t*>(
                q + (static_cast<size_t>(qi) * num_q_heads + hkv * heads_per_kv + qrow) * kHeadDim +
                qcol)
          : 0;

  const int causal_limit = kFastSQ1 ? seq - 1 : qpos + seq - seqlen_q;
  const int request_blocks = (seq + page_size - 1) / page_size;
  if constexpr (kSplit != 8) __syncthreads();

  const bool blk_ok = blk >= 0 && blk < request_blocks && blk < max_blocks;
  int physical = -1;
  if (blk_ok) {
    if constexpr (kSplit == 8)
      physical = page_table[request * max_blocks + blk];
    else
      physical = blk < pt_staged ? sPT[blk] : page_table[request * max_blocks + blk];
  }
  bool selected_ok = blk_ok && physical >= 0 && physical < num_pages;
  if (selected_ok) {
    const int slot = rank + tid * kSplit;
    const int* selected = q2k_row;
#pragma unroll 1
    for (int prior = 0; prior < slot; ++prior) selected_ok &= selected[prior] != blk;
  }

  // Only warp 0 owns selection slots (topk is validated <= 32).  Compact
  // every individually valid entry so an interior eviction or an out-of-range
  // score-ordered selection cannot shift a later valid page out of the loop.
  // The prior-slot scan also implements first-occurrence duplicate suppression
  // across cluster ranks, matching the reference contract.
  if (warp == 0) {
    const uint32_t mask = __ballot_sync(0xffffffffu, selected_ok);
    uint32_t lanes_before;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanes_before));
    const int compact = __popc(mask & lanes_before);
    if (selected_ok) {
      sTokenBase[compact] = blk * page_size;
      sPageOff[compact] = static_cast<uint64_t>(physical) * static_cast<uint64_t>(page_bytes);
    }
    if (lane == 0) sNpages = __popc(mask);
  }
  __syncthreads();
  RawBatch rawK, rawV;
  const int npages = sNpages;
  const int nbatch = (npages + 1) >> 1;
  if (npages == 0) {
    for (int e = tid; e < heads_per_kv * kHeadDim; e += kThreads)
      sO[(e >> 7) * kOStride + (e & (kHeadDim - 1))] = 0.0f;
  }

  const int taddrS = tmem_slot;       // columns  0..31 : two score tiles
  const int taddrO = tmem_slot + 32;  // columns 32..47 : running O^T
  constexpr uint32_t qk_desc = (1U << 4U) | (16U >> 3U << 17U) | (128U >> 4U << 24U);
  constexpr uint32_t pv_desc = qk_desc | (1U << 15U);

  const float ls2 = softmax_scale * k_global_scale * kLog2e;
  float2 run_sum2 = make_float2(0.0f, 0.0f);
  // Online softmax state, private to the warp that owns this query head.
  float run_max = kNegRunMaxOrigin;         // running row maximum, in log2 units
  float run_max_origin = kNegRunMaxOrigin;  // exponential origin, trails run_max by <= slack
  int phase = 0;

  const int qk_a = static_cast<int>(__cvta_generic_to_shared(sK));
  const int qk_b = static_cast<int>(__cvta_generic_to_shared(sQ));
  const int pv_a = static_cast<int>(__cvta_generic_to_shared(sV));
  const int pv_b = static_cast<int>(__cvta_generic_to_shared(sP));

  const int shead = warp;  // one warp per query head

  if (npages > 0) {
    const uint64_t p0 = sPageOff[0];
    const uint64_t p1 = npages > 1 ? sPageOff[1] : p0;
    // Both sides of the first batch are in flight before either is consumed:
    // K and V of a page live in the same page, so their misses overlap instead
    // of costing two serial round trips on the critical path.
    load_k_pages<kOddTail, kFullPageTile>(k_data, k_scale, p0, p1, npages > 1, 0, tid, cache, rawK);
    load_v_pages<kOddTail, kFullPageTile>(v_data, v_scale, p0, p1, npages > 1, 0, tid, cache, rawV);
    store_batch(sK, tid, rawK, npages > 1 ? 2 : 1);
    if (nbatch > 1) {
      const uint64_t q0 = sPageOff[2];
      const uint64_t q1 = npages > 3 ? sPageOff[3] : q0;
      load_k_pages<kOddTail, kFullPageTile>(k_data, k_scale, q0, q1, npages > 3, 0, tid, cache,
                                            rawK);
    }
  }
  // Q converts while the KV pages are still in flight.  sQ is the K-major B
  // operand of S^T = K Q^T.
  {
    const uint32_t packed =
        static_cast<uint32_t>(pack_bf16x2_to_e4m3x2(static_cast<uint32_t>(qquad))) |
        (static_cast<uint32_t>(pack_bf16x2_to_e4m3x2(static_cast<uint32_t>(qquad >> 32))) << 16);
    *reinterpret_cast<uint32_t*>(sQ + swz(qrow, qcol)) = packed;
  }
  // All warps publish Q/K, but only warp 0 consumes those tiles for QK.
  // Producers arrive and advance into V staging while warp 0 waits.
  if (warp == 0)
    asm volatile("bar.sync 1, 512;" ::: "memory");
  else
    asm volatile("bar.arrive 1, 512;" ::: "memory");
  if (warp == 0) asm volatile("tcgen05.fence::after_thread_sync;");

#pragma unroll 1
  for (int it = 0; it < nbatch; ++it) {
    const int first = it * 2;
    const bool has2 = (first + 1) < npages;
    const int token0 = sTokenBase[first];
    const int token1 = has2 ? sTokenBase[first + 1] : token0;

    if (warp == 0 && elect_sync()) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        mma_f8(taddrS, make_smem_desc(qk_a + kk * 32), make_smem_desc(qk_b + kk * 32), qk_desc,
               kk != 0);
      }
      if (has2) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          mma_f8(taddrS + 16, make_smem_desc(qk_a + kKvTileBytes + kk * 32),
                 make_smem_desc(qk_b + kk * 32), qk_desc, kk != 0);
        }
      }
      commit_mma(mbar);
    }

    // ---- V dequant + next-batch V prefetch, overlapped with the QK MMA ----
    store_batch(sV, tid, rawV, has2 ? 2 : 1);
    if (it + 1 < nbatch) {
      const int n0 = first + 2;
      const uint64_t p0 = sPageOff[n0];
      const uint64_t p1 = (n0 + 1) < npages ? sPageOff[n0 + 1] : p0;
      load_v_pages<kOddTail, kFullPageTile>(v_data, v_scale, p0, p1, (n0 + 1) < npages, 0, tid,
                                            cache, rawV);
    }

    if (warp < 8) mbarrier_wait(mbar, phase);
    phase ^= 1;
    if (warp < 8) asm volatile("tcgen05.fence::after_thread_sync;");

    // ---- Read both S^T tiles (128 tokens x 16 heads each) with 8 warps ----
    if (warp < 8) {
      const int pg = warp >> 2;
      const bool loaded = pg == 0 || has2;
      float* dst = sS + pg * kTokenTile + (warp & 3) * 32 + lane;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float s[8];
        if (loaded) {
          tmem_ld8(taddrS + pg * 16 + half * 8, s);
        } else {
#pragma unroll
          for (int h = 0; h < 8; ++h) s[h] = -1e30f;
        }
#pragma unroll
        for (int h = 0; h < 8; ++h) dst[(half * 8 + h) * kSStride] = s[h];
      }
      if (loaded) tmem_wait_ld();
    }
    __syncthreads();

    // ---- Softmax over both pages: warp `shead` owns query head `shead` ----
    float alpha_h = 1.0f;
    {
      const float* row = sS + shead * kSStride + lane * 4;
      const float4 v0 = *reinterpret_cast<const float4*>(row);
      const float4 v1 = *reinterpret_cast<const float4*>(row + kTokenTile);
      const float2 scale2 = make_float2(ls2, ls2);
      float x[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
      // Masked lanes are driven to -inf, which survives the max reduction and
      // which exp2 flushes to zero, so no separate validity mask has to stay
      // live across the reductions.
      const bool full = kFastSQ1 && kFullPageTile && has2 && (token0 + page_size <= seq) &&
                        (token1 + page_size <= seq);
      if (!full) {
        const int c0 = token0 + lane * 4;
        const int c1 = token1 + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int local = lane * 4 + j;
          const int col = c0 + j;
          if (!(local < page_size && col < seq && (kFastSQ1 || !causal || col <= causal_limit)))
            x[j] = -INFINITY;
          const int col1 = c1 + j;
          if (!(has2 && local < page_size && col1 < seq &&
                (kFastSQ1 || !causal || col1 <= causal_limit)))
            x[4 + j] = -INFINITY;
        }
      }
      // Online row maximum.  The raw dot products are reduced first and scaled
      // once afterwards; ls2 > 0 so max(x)*ls2 == max(x*ls2).
      const float rm = warp_max(fmaxf(fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3])),
                                      fmaxf(fmaxf(x[4], x[5]), fmaxf(x[6], x[7]))));
      run_max = fmaxf(run_max, rm * ls2);
      // Bump the exponential origin onto the running maximum whenever the
      // maximum has outgrown the guard band.  alpha rescales everything that
      // was accumulated against the previous origin.
      if (run_max > run_max_origin + kOriginSlack) {
        // The first transition starts from an exactly zero accumulator, so its
        // scale factor is known without spending an SFU exponential.
        alpha_h = run_max_origin == kNegRunMaxOrigin ? 0.0f : ex2(run_max_origin - run_max);
        run_max_origin = run_max;
      }
      const float sh = kPreScale - run_max_origin;
      const float2 shift2 = make_float2(sh, sh);
      float2 bs2 = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const float2 z = fma_f32x2(make_float2(x[j], x[j + 1]), scale2, shift2);
        x[j] = ex2(z.x);
        x[j + 1] = ex2(z.y);
        bs2 = add_f32x2(bs2, make_float2(x[j], x[j + 1]));
      }
      // Accumulate lane-local denominator partials against the current origin
      // and reduce once after all selected pages, instead of paying a warp
      // reduction for every batch.
      run_sum2 = fma_f32x2(run_sum2, make_float2(alpha_h, alpha_h), bs2);
#pragma unroll
      for (int pg = 0; pg < 2; ++pg) {
        const uint16_t lo = pack_e4m3x2(x[pg * 4], x[pg * 4 + 1]);
        const uint16_t hi = pack_e4m3x2(x[pg * 4 + 2], x[pg * 4 + 3]);
        *reinterpret_cast<uint32_t*>(sP + pg * kProbTileBytes + swz(shead, lane * 4)) =
            static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
      if (lane == 0) sAlpha[shead] = alpha_h;
    }
    // Only warps 0..3 consume P/O. Other warps publish their probability rows
    // and immediately advance toward the next K stage. Every consumer warp
    // derives the same CTA-wide origin decision with one shared load per lane.
    if (warp < 4)
      asm volatile("bar.sync 2, 512;" ::: "memory");
    else
      asm volatile("bar.arrive 2, 512;" ::: "memory");
    int origin_held = 1;
    if (warp < 4) {
      const bool lane_held = lane >= heads_per_kv || sAlpha[lane] == 1.0f;
      origin_held = __all_sync(0xffffffff, lane_held);
    }
    if (it > 0 && warp < 4 && !origin_held) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float o[8];
        tmem_ld8(taddrO + half * 8, o);
        tmem_wait_ld();
#pragma unroll
        for (int h = 0; h < 8; ++h) o[h] *= sAlpha[half * 8 + h];
        tmem_st8(taddrO + half * 8, o);
      }
      tmem_wait_st();
      // All four O fragments must be rescaled before warp 0 accumulates PV.
      asm volatile("bar.sync 3, 128;" ::: "memory");
    }
    if (warp < 4) asm volatile("tcgen05.fence::after_thread_sync;");

    if (warp == 0 && elect_sync()) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        mma_f8(taddrO, make_smem_desc(pv_a + kk * 32 * kHeadDim), make_smem_desc(pv_b + kk * 32),
               pv_desc, it != 0 || kk != 0);
      }
      if (has2) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          mma_f8(taddrO, make_smem_desc(pv_a + kKvTileBytes + kk * 32 * kHeadDim),
                 make_smem_desc(pv_b + kProbTileBytes + kk * 32), pv_desc, 1);
        }
      }
      commit_mma(mbar);
    }

    // ---- Next batch's K dequant, overlapped with the PV MMA ----
    if (it + 1 < nbatch) {
      const int n0 = first + 2;
      store_batch(sK, tid, rawK, (n0 + 1) < npages ? 2 : 1);
      if (it + 2 < nbatch) {
        const int m0 = first + 4;
        const uint64_t p0 = sPageOff[m0];
        const uint64_t p1 = (m0 + 1) < npages ? sPageOff[m0 + 1] : p0;
        load_k_pages<kOddTail, kFullPageTile>(k_data, k_scale, p0, p1, (m0 + 1) < npages, 0, tid,
                                              cache, rawK);
      }
    }

    // Intermediate completion only gates warp 0, which reuses the barrier to
    // issue the next QK. O-reading warps need their own wait only at epilogue.
    const bool last_it = (it + 1 == nbatch);
    const bool wait_pv = (warp == 0) || (last_it && warp < 4);
    if (wait_pv) mbarrier_wait(mbar, phase);
    phase ^= 1;
    if (wait_pv) asm volatile("tcgen05.fence::after_thread_sync;");

    // The epilogue begins with its own CTA/cluster rendezvous, so only an
    // intermediate iteration needs to protect the next QK shared-memory tile.
    if (it + 1 < nbatch) __syncthreads();
  }

  float run_sum = run_sum2.x + run_sum2.y;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) run_sum += __shfl_xor_sync(0xffffffff, run_sum, off);

  __nv_bfloat16* dst =
      output + (static_cast<size_t>(qi) * num_q_heads + hkv * heads_per_kv) * kHeadDim;

  if constexpr (kSplit == 1) {
    if (lane == 0 && warp < heads_per_kv)
      sInv[warp] = run_sum > 0.0f ? v_global_scale / run_sum : 0.0f;
    __syncthreads();
    if (warp < 4) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float o[8];
        if (npages > 0) {
          tmem_ld8(taddrO + half * 8, o);
          tmem_wait_ld();
        }
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          const int hh = half * 8 + h;
          if (hh >= heads_per_kv) continue;
          const float x = npages > 0 ? o[h] * sInv[hh] : 0.0f;
          dst[hh * kHeadDim + warp * 32 + lane] = __float2bfloat16_rn(x);
        }
      }
    }
    __syncthreads();
    if (warp == 1) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_slot),
                   "r"(64));
    }
  } else {
    // Materialize the local FP32 partial once for split-cluster reduction.
    if (warp < 4 && npages > 0) {
      float* dst_o = sO + warp * 32 + lane;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float o[8];
        tmem_ld8(taddrO + half * 8, o);
        tmem_wait_ld();
#pragma unroll
        for (int h = 0; h < 8; ++h) {
          const int hh = half * 8 + h;
          if (hh < heads_per_kv) dst_o[hh * kOStride] = o[h];
        }
      }
    }
    __syncthreads();
    if (warp == 1) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_slot),
                   "r"(64));
    }
    // Each rank ran its own online maximum, so its partial numerator and
    // denominator live on its own exponential origin.  Both are published as one
    // pair and renormalized onto the cluster-wide maximum origin before the
    // reduction; pairing them turns two dependent DSMEM rounds into one, and the
    // remote O handles are mapped before the rendezvous that already exists.
    if (lane == 0 && warp < heads_per_kv) sStat[warp] = make_float2(run_sum, run_max_origin);
    const uint32_t aStat = static_cast<uint32_t>(__cvta_generic_to_shared(sStat));
    const uint32_t aO = static_cast<uint32_t>(__cvta_generic_to_shared(sO));
    if (tid < kSplit) sRemoteO[tid] = map_rank(aO, tid);
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    const int heads_per_rank = heads_per_kv / kSplit;
    const int first_head = rank * heads_per_rank;
    if (tid < heads_per_rank) {
      const int h = first_head + tid;
      float2 st[kSplit];
      float gmax = kNegRunMaxOrigin;
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        st[r] = ld_dsmem_v2(map_rank(aStat, r) + h * 8);
        gmax = fmaxf(gmax, st[r].y);
      }
      float denom = 0.0f;
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        const float sc = ex2(st[r].y - gmax);
        sScale[r * heads_per_rank + tid] = sc;
        denom = fmaf(st[r].x, sc, denom);
      }
      sInv[h] = denom > 0.0f ? v_global_scale / denom : 0.0f;
    }
    __syncthreads();
    for (int e = tid; e < heads_per_rank * (kHeadDim / 4); e += kThreads) {
      const int hl = e >> 5;
      const int h = first_head + hl;
      const int d = (e & 31) * 4;
      const int oe = h * kOStride + d;
      float2 num01 = make_float2(0.0f, 0.0f);
      float2 num23 = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        const float4 x = ld_dsmem_v4(sRemoteO[r] + oe * 4);
        const float sc = sScale[r * heads_per_rank + hl];
        const float2 sc2 = make_float2(sc, sc);
        num01 = fma_f32x2(make_float2(x.x, x.y), sc2, num01);
        num23 = fma_f32x2(make_float2(x.z, x.w), sc2, num23);
      }
      const float inv = sInv[h];
      BF16x2Bits out01, out23;
      out01.b = __floats2bfloat162_rn(num01.x * inv, num01.y * inv);
      out23.b = __floats2bfloat162_rn(num23.x * inv, num23.y * inv);
      *reinterpret_cast<uint64_t*>(dst + h * kHeadDim + d) =
          static_cast<uint64_t>(out01.u) | (static_cast<uint64_t>(out23.u) << 32);
    }
    cluster.sync();
  }
}

// ---------------------------------------------------------------------------
// Launch.  The shared-memory attributes are set on the SAME function pointer
// that is then launched, immediately before launching it, rather than from a
// process-global "already configured" table.  Two reasons, both learned here:
// the attribute is per (device, function), so a process-global flag poisons
// every device after the first; and configure-then-launch makes "launched an
// instantiation nobody configured" structurally impossible instead of a
// property of two lists staying in sync.  Neither call is stream-ordered, so
// both are legal inside a CUDA graph capture region.
// ---------------------------------------------------------------------------
cudaError_t launch(const void* q, const void* k_data, const void* v_data, const void* k_scale,
                   const void* v_scale, const int* q2k_indices, const int* page_table,
                   const int* seqused_k, int total_q, int q2k_head_stride, int q2k_token_stride,
                   int num_pages, int num_q_heads, int num_kv_heads, int heads_per_kv,
                   int page_size, int topk, int max_blocks, int page_bytes, int data_head_stride,
                   int scale_head_stride, int seqlen_q, int causal, float softmax_scale,
                   float k_global_scale, float v_global_scale, void* output, cudaStream_t stream) {
  const int tiles = total_q * num_kv_heads;
  int split;
  if (tiles <= 32)
    split = 8;
  else if (tiles <= 64)
    split = 4;
  else if (tiles <= 128)
    split = 2;
  else
    split = 1;
  while (split > topk || split > heads_per_kv || (heads_per_kv % split) != 0) {
    split >>= 1;
  }
  const bool fast = (seqlen_q == 1);
  const bool odd_tail = fast && num_pages < 32 * total_q;

  auto pick = [&](auto full_page_tag, int s, bool f, bool o) {
    constexpr bool full_page = decltype(full_page_tag)::value;
    if (f) {
      if (o) {
        switch (s) {
          case 8:
            return kernel_msa_decode_nvfp4_kv_paged<8, true, true, full_page>;
          case 4:
            return kernel_msa_decode_nvfp4_kv_paged<4, true, true, full_page>;
          case 2:
            return kernel_msa_decode_nvfp4_kv_paged<2, true, true, full_page>;
          default:
            return kernel_msa_decode_nvfp4_kv_paged<1, true, true, full_page>;
        }
      }
      switch (s) {
        case 8:
          return kernel_msa_decode_nvfp4_kv_paged<8, true, false, full_page>;
        case 4:
          return kernel_msa_decode_nvfp4_kv_paged<4, true, false, full_page>;
        case 2:
          return kernel_msa_decode_nvfp4_kv_paged<2, true, false, full_page>;
        default:
          return kernel_msa_decode_nvfp4_kv_paged<1, true, false, full_page>;
      }
    } else {
      switch (s) {
        case 8:
          return kernel_msa_decode_nvfp4_kv_paged<8, false, false, full_page>;
        case 4:
          return kernel_msa_decode_nvfp4_kv_paged<4, false, false, full_page>;
        case 2:
          return kernel_msa_decode_nvfp4_kv_paged<2, false, false, full_page>;
        default:
          return kernel_msa_decode_nvfp4_kv_paged<1, false, false, full_page>;
      }
    }
  };
  const bool full_page = page_size == kTokenTile;
  auto kernel = full_page ? pick(std::true_type{}, split, fast, odd_tail)
                          : pick(std::false_type{}, split, fast, odd_tail);

  cudaError_t status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 85);
  if (status != cudaSuccess) return status;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(num_kv_heads * split, total_q);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = stream;
  // The default (spread) cluster placement measured faster than an explicit
  // load-balancing preference at every split this heuristic selects.
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = split;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = split == 1 ? nullptr : &attr;
  config.numAttrs = split == 1 ? 0 : 1;
  return cudaLaunchKernelEx(
      &config, kernel, static_cast<const __nv_bfloat16*>(q), static_cast<const uint8_t*>(k_data),
      static_cast<const uint8_t*>(v_data), static_cast<const uint8_t*>(k_scale),
      static_cast<const uint8_t*>(v_scale), q2k_indices, page_table, seqused_k, q2k_head_stride,
      q2k_token_stride, num_pages, num_q_heads, num_kv_heads, heads_per_kv, page_size, topk,
      max_blocks, page_bytes, data_head_stride, scale_head_stride, seqlen_q, causal, softmax_scale,
      k_global_scale, v_global_scale, static_cast<__nv_bfloat16*>(output));
}

}  // namespace general

namespace pinned {

constexpr int kNumQHeads = 64;
constexpr int kNumKVHeads = 4;
constexpr int kHeadsPerKV = 16;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kTopK = 16;
constexpr int kMaxBlocks = 128;
constexpr int kThreads = 512;
constexpr int kMmaAlignment = 1 << 10;
constexpr float kLog2PScale = 8.0f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNegInf = -1e30f;

constexpr int kTileBytes = 16384;  // 128 tokens x 128 e4m3, 128B-swizzled rows
constexpr int kOStride = 132;      // floats per head row of the running output
constexpr int kScoreBytesPerPage = kHeadsPerKV * kPageSize * sizeof(float);
constexpr int kProbBytesPerPage = kHeadsPerKV * kPageSize;

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "elect.sync _|%%p, %1;\n\t"
      "@%%p mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xffffffff));
  return pred;
}

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3ffffULL) >> 4ULL;
}

// 128B swizzle, rows of 128 bytes, SBO = 8 rows.
__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  constexpr int kSBO = 8 * 128;
  return desc_encode(addr) | (desc_encode(kSBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

__device__ __forceinline__ void mbarrier_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "WAIT%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra.uni WAIT%=;\n\t"
      "}" ::"r"(addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

__device__ __forceinline__ uint32_t map_rank(uint32_t addr, int rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

__device__ __forceinline__ float ld_dsmem(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr));
  return v;
}

__device__ __forceinline__ float2 ld_dsmem_v2(uint32_t addr) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
  return v;
}

__device__ __forceinline__ float4 ld_dsmem_v4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void mma_f8(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
                                       int accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(taddr),
      "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(accumulate));
}

__device__ __forceinline__ void commit_mma(int mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tmem_ld16(int addr, float (&v)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7]), "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]),
        "=f"(v[14]), "=f"(v[15])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_ld8(int addr, float (&v)[8]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]),
        "=f"(v[7])
      : "r"(addr));
}

__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }

__device__ __forceinline__ void tmem_st16(int addr, const float (&v)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};" ::"f"(v[0]),
      "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "f"(v[8]),
      "f"(v[9]), "f"(v[10]), "f"(v[11]), "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]), "r"(addr));
}

__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }

__device__ __forceinline__ float ex2(float x) { return exp2f(x); }

__device__ __forceinline__ float warp_max(float x) {
  float r;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(x));
  return r;
}

union F32x2Bits {
  float2 f;
  uint64_t u;
};

union BF16x2Bits {
  __nv_bfloat162 b;
  uint32_t u;
};

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
  F32x2Bits pa, pb, pr;
  pa.f = a;
  pb.f = b;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u));
  return pr.f;
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
  F32x2Bits pa, pb, pr;
  pa.f = a;
  pb.f = b;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u));
  return pr.f;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
  F32x2Bits pa, pb, pc, pr;
  pa.f = a;
  pb.f = b;
  pc.f = c;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(pr.u) : "l"(pa.u), "l"(pb.u), "l"(pc.u));
  return pr.f;
}

__device__ __forceinline__ uint4 load_stream_v4(const void* ptr) {
  uint4 v;
  asm("ld.global.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t load_stream_u16(const void* ptr) {
  uint32_t v;
  asm("ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint64_t load_stream_u64(const void* ptr) {
  uint64_t v;
  asm("ld.global.L1::no_allocate.L2::256B.u64 %0, [%1];" : "=l"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// 128B-swizzled byte offset inside a tile whose rows are 128 bytes wide.
__device__ __forceinline__ int swz(int row, int col) {
  return row * 128 + (col ^ ((row & 7) << 4));
}

// two e4m3 scale bytes (packed in the low 16 bits) -> f16x2
__device__ __forceinline__ uint32_t scales_to_f16x2(uint32_t two_e4m3) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((uint16_t)two_e4m3));
  return r;
}

// 16 packed e2m1 values * one broadcast f16x2 block scale -> 16 e4m3 values.
__device__ __forceinline__ uint4 dq16(uint32_t lo, uint32_t hi, uint32_t sfh) {
  uint4 out;
  asm("{\n\t"
      ".reg .b32 h0, h1, h2, h3;\n\t"
      ".reg .b16 e0, e1, e2, e3;\n\t"
      ".reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
      "mul.rn.f16x2 h0, h0, %6;\n\t"
      "mul.rn.f16x2 h1, h1, %6;\n\t"
      "mul.rn.f16x2 h2, h2, %6;\n\t"
      "mul.rn.f16x2 h3, h3, %6;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
      "mov.b32 %0, {e0, e1};\n\t"
      "mov.b32 %1, {e2, e3};\n\t"
      "mov.b32 {b0, b1, b2, b3}, %5;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 h1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 h3, b3;\n\t"
      "mul.rn.f16x2 h0, h0, %6;\n\t"
      "mul.rn.f16x2 h1, h1, %6;\n\t"
      "mul.rn.f16x2 h2, h2, %6;\n\t"
      "mul.rn.f16x2 h3, h3, %6;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
      "mov.b32 %2, {e0, e1};\n\t"
      "mov.b32 %3, {e2, e3};\n\t"
      "}"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "r"(lo), "r"(hi), "r"(sfh));
  return out;
}

// One (page, kv-head) worth of raw NVFP4 for this thread: 16 data bytes that
// cover two adjacent 16-element scale groups, plus the two block scales.
struct Raw {
  uint4 d;
  uint32_t sf;  // two e4m3 bytes in the low 16 bits
};

// Per-thread offsets are invariant across every paged K/V load and dequant.
struct Addr {
  unsigned dat;
  unsigned ksc;
  unsigned vsc;
  unsigned vsel;
  int st0;
  int st1;
};

__device__ __forceinline__ Addr make_addr(int tid, int hkv, int data_head_stride,
                                          int scale_head_stride) {
  Addr a;
  a.dat = hkv * data_head_stride + static_cast<unsigned>(tid) * 16u;
  a.ksc = hkv * scale_head_stride + static_cast<unsigned>(tid) * 2u;
  a.vsc = hkv * scale_head_stride + static_cast<unsigned>(((tid >> 4) * 4 + (tid & 3)) * 8);
  const unsigned t3 = static_cast<unsigned>((tid >> 2) & 3);
  a.vsel = t3 | ((t3 + 4) << 4);
  a.st0 = swz(tid >> 2, (tid & 3) * 32);
  a.st1 = swz(tid >> 2, (tid & 3) * 32 + 16);
  return a;
}

template <bool kIsV>
__device__ __forceinline__ void load_raw(const uint8_t* __restrict__ data,
                                         const uint8_t* __restrict__ scale, size_t base,
                                         const Addr& a, Raw& r) {
  r.d = load_stream_v4(data + base + a.dat);
  if constexpr (kIsV) {
    // (4,4)-swizzle: logical (t, s) lives at (t/4)*32 + (s/2)*8 + (s&1)*4 + t%4
    const uint64_t oct = load_stream_u64(scale + base + a.vsc);
    const uint32_t w0 = static_cast<uint32_t>(oct);
    const uint32_t w1 = static_cast<uint32_t>(oct >> 32);
    r.sf = prmt(w0, w1, a.vsel) & 0xffffu;
  } else {
    r.sf = load_stream_u16(scale + base + a.ksc);
  }
}

__device__ __forceinline__ void store_tile(uint8_t* tile, const Addr& a, const Raw& r) {
  const uint32_t sfh = scales_to_f16x2(r.sf);
  const uint32_t s0 = prmt(sfh, sfh, 0x1010u);
  const uint32_t s1 = prmt(sfh, sfh, 0x3232u);
  *reinterpret_cast<uint4*>(tile + a.st0) = dq16(r.d.x, r.d.y, s0);
  *reinterpret_cast<uint4*>(tile + a.st1) = dq16(r.d.z, r.d.w, s1);
}

constexpr int smem_bytes_for(int chunk) {
  return chunk * kTileBytes + chunk * kScoreBytesPerPage + chunk * kProbBytesPerPage +
         kProbBytesPerPage;
}
template <int kSplit, int kChunk, bool kFastSQ1, bool kTailAware = false>
__global__ __launch_bounds__(kThreads, 2) void kernel_msa_decode_nvfp4_kv_paged_pinned(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ k_data,
    const uint8_t* __restrict__ v_data, const uint8_t* __restrict__ k_scale,
    const uint8_t* __restrict__ v_scale, const int* __restrict__ q2k_indices,
    const int* __restrict__ page_table, const int* __restrict__ seqused_k, int q2k_head_stride,
    int q2k_token_stride, int num_pages, int page_bytes, int data_head_stride,
    int scale_head_stride, int seqlen_q, int causal, float softmax_scale, float k_global_scale,
    float v_global_scale, __nv_bfloat16* __restrict__ output) {
  // This family is a canonical-geometry specialization selected only after the
  // host validates the actual strided views.  Repeat that validation at the
  // device boundary, expressing every derived stride from the specialized
  // geometry; mismatches remain the generalized kernel's responsibility.
  constexpr int kExpectedDataHeadStride = kPageSize * (kHeadDim / 2);
  constexpr int kExpectedScaleHeadStride = kPageSize * (kHeadDim / 16);
  constexpr int kExpectedPageBytes =
      2 * kNumKVHeads * (kExpectedDataHeadStride + kExpectedScaleHeadStride);
  if (page_bytes != kExpectedPageBytes || data_head_stride != kExpectedDataHeadStride ||
      scale_head_stride != kExpectedScaleHeadStride)
    return;

  constexpr int kPages = kTopK / kSplit;
  constexpr int kChunks = kPages / kChunk;
  constexpr int kHalf = kChunk / 2;  // pages per ping-pong half

  constexpr int kOffT = 0;
  constexpr int kOffS = kChunk * kTileBytes;
  constexpr int kOffP = kOffS + kChunk * kScoreBytesPerPage;
  constexpr int kOffQ = kOffP + kChunk * kProbBytesPerPage;
  constexpr int kSmemBytes = kOffQ + kProbBytesPerPage;
  static_assert(kSmemBytes == smem_bytes_for(kChunk),
                "the launcher must request exactly this kernel's shared-memory map");
  constexpr int kSStride = kChunk * kPageSize;  // floats per head row

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rank = blockIdx.x & (kSplit - 1);
  const int task = blockIdx.x / kSplit;
  const int qi = task >> 2;
  const int hkv = task & 3;

  extern __shared__ __align__(kMmaAlignment) uint8_t smem[];
  uint8_t* sT = smem + kOffT;
  float* sS = reinterpret_cast<float*>(smem + kOffS);
  float* sO = reinterpret_cast<float*>(smem + kOffS);  // aliased, epilogue only
  uint8_t* sP = smem + kOffP;
  uint8_t* sQ = smem + kOffQ;

  __shared__ __align__(8) uint64_t bar[2];
  __shared__ int tmem_slot;
  __shared__ float sAlpha[kHeadsPerKV];
  __shared__ float sSum[kHeadsPerKV];
  __shared__ __align__(8) float2 sStat[kHeadsPerKV];
  __shared__ float sScale[kSplit * kHeadsPerKV];
  __shared__ float sInv[kHeadsPerKV];
  __shared__ uint32_t sRemoteO[kSplit];
  __shared__ size_t sPageOff[kPages];
  __shared__ int sBlk[kPages];
  __shared__ int sNpages;
  __shared__ int sSelected[kTopK];
  __shared__ int sPT[kMaxBlocks];

  const int mbarA = static_cast<int>(__cvta_generic_to_shared(&bar[0]));
  const int mbarB = static_cast<int>(__cvta_generic_to_shared(&bar[1]));
  if (warp == 0 && elect_sync()) {
    mbarrier_init(mbarA, 1);
    mbarrier_init(mbarB, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp == 1) {
    const int slot = static_cast<int>(__cvta_generic_to_shared(&tmem_slot));
    constexpr int kCols = (kChunk * 16 + 16) <= 64 ? 64 : 128;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot),
                 "r"(kCols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  int request, qpos;
  if constexpr (kFastSQ1) {
    request = qi;
    qpos = 0;
  } else {
    request = qi / seqlen_q;
    qpos = qi - request * seqlen_q;
  }

  // Independent global loads issued together: seq length, the block list and
  // the whole page-table row (so the blk -> page lookup is an SMEM hit).
  const int seq = seqused_k[request];
  int blk_reg = -1;
  // Two strides, not `total_q * kTopK`: this family's topk is compile-time but
  // its q2k LAYOUT is not, so a transposed (total_q, kv head, topk) view is
  // read in place instead of being copied contiguous by the caller.
  if (warp == 0 && lane < kTopK)
    sSelected[lane] = q2k_indices[static_cast<size_t>(hkv) * q2k_head_stride +
                                  static_cast<size_t>(qi) * q2k_token_stride + lane];
  {
    const int* prow = page_table + request * kMaxBlocks;
    for (int i = tid; i < kMaxBlocks; i += kThreads) sPT[i] = prow[i];
  }
  const int causal_limit = kFastSQ1 ? seq - 1 : qpos + seq - seqlen_q;

  // Stage Q (bf16 -> e4m3) once.  sQ is the K-major B operand of S^T = K Q^T.
  {
    const int row = tid >> 5;
    const int col = (tid & 31) * 4;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(
        q + (static_cast<size_t>(qi) * kNumQHeads + hkv * kHeadsPerKV + row) * kHeadDim + col);
    const uint32_t a = src[0];
    const uint32_t b = src[1];
    const uint32_t lo = pack_bf16x2_to_e4m3x2(a);
    const uint32_t hi = pack_bf16x2_to_e4m3x2(b);
    *reinterpret_cast<uint32_t*>(sQ + swz(row, col)) = lo | (hi << 16);
  }
  __syncthreads();

  bool selected_ok = false;
  int physical = -1;
  if (warp == 0 && lane < kTopK) {
    const int slot = lane;
    blk_reg = sSelected[slot];
    // Every lane represents one global selection slot.  match_any identifies
    // the first occurrence in one warp instruction; each cluster rank then
    // retains only its strided subset before the ballot compacts it locally.
    const uint32_t peers = __match_any_sync(0x0000ffffu, blk_reg);
    const bool first = lane == (__ffs(peers) - 1);
    const bool assigned = (lane & (kSplit - 1)) == rank;
    selected_ok =
        assigned && first && blk_reg >= 0 && blk_reg < kMaxBlocks && blk_reg * kPageSize < seq;
    if (selected_ok) {
      physical = sPT[blk_reg];
      selected_ok = physical >= 0 && physical < num_pages;
    }
  }

  // Preserve first-occurrence order while compacting every valid page.  This
  // keeps later pages live after an interior eviction and suppresses repeated
  // score-ordered selections across cluster ranks.
  if (warp == 0) {
    const uint32_t mask = __ballot_sync(0xffffffffu, selected_ok);
    uint32_t lanes_before;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanes_before));
    const int compact = __popc(mask & lanes_before);
    if (selected_ok) {
      sBlk[compact] = blk_reg;
      sPageOff[compact] = static_cast<size_t>(physical) * page_bytes;
    }
    if (lane == 0) sNpages = __popc(mask);
  }
  __syncthreads();
  const int npages = sNpages;

  const int taddrS = tmem_slot;
  const int taddrO = tmem_slot + kChunk * 16;
  constexpr uint32_t qk_desc = (1U << 4U) | (16U >> 3U << 17U) | (128U >> 4U << 24U);
  constexpr uint32_t pv_desc = qk_desc | (1U << 15U);

  const float ls2 = softmax_scale * k_global_scale * kLog2e;
  float run_max = kNegInf;
  float run_sum = 0.0f;
  int phaseA = 0, phaseB = 0;

  const int tile_a = static_cast<int>(__cvta_generic_to_shared(sT));
  const int q_a = static_cast<int>(__cvta_generic_to_shared(sQ));
  const int p_a = static_cast<int>(__cvta_generic_to_shared(sP));

  const Addr addr = make_addr(tid, hkv, data_head_stride, scale_head_stride);

  if (npages == 0) {
    for (int e = tid; e < kHeadsPerKV * kHeadDim; e += kThreads)
      sO[(e >> 7) * kOStride + (e & (kHeadDim - 1))] = 0.0f;
  }

  // Clamp absent trailing pages onto page 0 so every tile holds finite data;
  // their scores are masked to -inf so they contribute nothing.
  if (tid < kPages && tid >= npages && npages > 0) sPageOff[tid] = sPageOff[0];
  __syncthreads();

  Raw rk[kHalf], rv[kHalf];

  if (npages > 0) {
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      if (!kTailAware || i < npages) load_raw<false>(k_data, k_scale, sPageOff[i], addr, rk[i]);
    }
  }

#pragma unroll 1
  for (int c = 0; c < kChunks; ++c) {
    const int p0 = c * kChunk;
    if (p0 >= npages) break;
    const int nvalid = kTailAware ? min(kChunk, npages - p0) : kChunk;
    const int nfirst = kTailAware ? min(kHalf, nvalid) : kHalf;
    const int nsecond = kTailAware ? nvalid - nfirst : kHalf;

    // ---- first half: K -> tiles [0, kHalf) ----
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nfirst) store_tile(sT + i * kTileBytes, addr, rk[i]);
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nsecond)
        load_raw<false>(k_data, k_scale, sPageOff[p0 + kHalf + i], addr, rk[i]);
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (warp == 0 && elect_sync()) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (kTailAware && i >= nfirst) continue;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_f8(taddrS + i * 16, make_smem_desc(tile_a + i * kTileBytes + kk * 32),
                 make_smem_desc(q_a + kk * 32), qk_desc, kk != 0);
      }
      commit_mma(mbarA);
    }

    // ---- second half: K -> tiles [kHalf, kChunk) ----
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nsecond) store_tile(sT + (kHalf + i) * kTileBytes, addr, rk[i]);
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nfirst) load_raw<true>(v_data, v_scale, sPageOff[p0 + i], addr, rv[i]);
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
    if ((!kTailAware || nsecond > 0) && warp == 0 && elect_sync()) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (kTailAware && i >= nsecond) continue;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_f8(taddrS + (kHalf + i) * 16,
                 make_smem_desc(tile_a + (kHalf + i) * kTileBytes + kk * 32),
                 make_smem_desc(q_a + kk * 32), qk_desc, kk != 0);
      }
      commit_mma(mbarB);
    }

    // ---- V for the first half, overlapped with the second half's QK MMA ----
    mbarrier_wait(mbarA, phaseA);
    phaseA ^= 1;
    asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nfirst) store_tile(sT + i * kTileBytes, addr, rv[i]);
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nsecond)
        load_raw<true>(v_data, v_scale, sPageOff[p0 + kHalf + i], addr, rv[i]);

    // ---- read every score tile of this chunk out of TMEM ----
    if (!kTailAware || nsecond > 0) {
      mbarrier_wait(mbarB, phaseB);
      phaseB ^= 1;
    }
    asm volatile("tcgen05.fence::after_thread_sync;");
    if (warp < (kTailAware ? nvalid * 4 : kChunk * 4)) {
      const int pg = warp >> 2;
      float s[16];
      tmem_ld16(taddrS + pg * 16, s);
      float* dst = sS + pg * kPageSize + (warp & 3) * 32 + lane;
#pragma unroll
      for (int h = 0; h < 16; ++h) dst[h * kSStride] = s[h];
      tmem_wait_ld();
    }
    __syncthreads();

    // ---- softmax: warp h owns query head h over all kChunk pages ----
    {
      const int shead = warp;
      float x[kChunk * 4];
      const float2 scale2 = make_float2(ls2, ls2);
#pragma unroll
      for (int p = 0; p < kChunk; ++p) {
        const float4 v =
            *reinterpret_cast<const float4*>(sS + shead * kSStride + p * kPageSize + lane * 4);
        const float2 a = mul_f32x2(make_float2(v.x, v.y), scale2);
        const float2 b = mul_f32x2(make_float2(v.z, v.w), scale2);
        x[p * 4 + 0] = a.x;
        x[p * 4 + 1] = a.y;
        x[p * 4 + 2] = b.x;
        x[p * 4 + 3] = b.y;
      }
      bool tail = true;
      if (kFastSQ1 && npages == kPages) {
        tail = false;
#pragma unroll
        for (int p = 0; p < kChunk; ++p)
          if ((sBlk[p0 + p] + 1) * kPageSize > seq) tail = true;
      }
      if (tail) {
#pragma unroll
        for (int p = 0; p < kChunk; ++p) {
          const int gi = p0 + p;
          const int base = (gi < npages) ? sBlk[gi] * kPageSize + lane * 4 : -1;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int col = base + j;
            const bool ok =
                (base >= 0) && col < seq && (kFastSQ1 || !causal || col <= causal_limit);
            if (!ok) x[p * 4 + j] = kNegInf;
          }
        }
      }
      float m0 = fmaxf(x[0], x[1]);
      float m1 = fmaxf(x[2], x[3]);
#pragma unroll
      for (int j = 4; j < kChunk * 4; j += 4) {
        m0 = fmaxf(m0, fmaxf(x[j], x[j + 1]));
        m1 = fmaxf(m1, fmaxf(x[j + 2], x[j + 3]));
      }
      float m = warp_max(fmaxf(m0, m1));
      const float new_max = fmaxf(fmaxf(run_max, m), -1e29f);
      const float alpha = (c == 0) ? 0.0f : ex2(run_max - new_max);
      const float shift = kLog2PScale - new_max;
      float2 bs2 = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int j = 0; j < kChunk * 4; j += 2) {
        x[j] = ex2(x[j] + shift);
        x[j + 1] = ex2(x[j + 1] + shift);
        bs2 = add_f32x2(bs2, make_float2(x[j], x[j + 1]));
      }
      float bs = bs2.x + bs2.y;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) bs += __shfl_xor_sync(0xffffffff, bs, off);
      run_sum = (c == 0) ? bs : run_sum * alpha + bs;
      run_max = new_max;
#pragma unroll
      for (int pgi = 0; pgi < kChunk; ++pgi) {
        const uint32_t lo = pack_e4m3x2(x[pgi * 4], x[pgi * 4 + 1]);
        const uint32_t hi = pack_e4m3x2(x[pgi * 4 + 2], x[pgi * 4 + 3]);
        *reinterpret_cast<uint32_t*>(sP + pgi * 2048 + swz(shead, lane * 4)) = lo | (hi << 16);
      }
      if (lane == 0 && c != 0) sAlpha[shead] = alpha;
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    // Rescale the running FP32 numerator in TMEM when the max rose.
    if (c != 0) {
      if (warp < 4) {
        float o[16];
        tmem_ld16(taddrO, o);
        tmem_wait_ld();
#pragma unroll
        for (int h = 0; h < 16; h += 2) {
          const float2 z =
              mul_f32x2(make_float2(o[h], o[h + 1]), make_float2(sAlpha[h], sAlpha[h + 1]));
          o[h] = z.x;
          o[h + 1] = z.y;
        }
        tmem_st16(taddrO, o);
        tmem_wait_st();
        asm volatile("tcgen05.fence::before_thread_sync;");
      }
      __syncthreads();
      asm volatile("tcgen05.fence::after_thread_sync;");
    }

    // ---- PV for the first half ----
    if (warp == 0 && elect_sync()) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (kTailAware && i >= nfirst) continue;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_f8(taddrO, make_smem_desc(tile_a + i * kTileBytes + kk * 32 * 128),
                 make_smem_desc(p_a + i * 2048 + kk * 32), pv_desc, (c != 0) || i != 0 || kk != 0);
      }
      // A single-CTA launch can queue both PV halves behind one completion;
      // split clusters retain the established two-commit ordering.
      if constexpr (kSplit != 1) commit_mma(mbarA);
    }

    // ---- V for the second half, overlapped with the first half's PV MMA ----
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      if (!kTailAware || i < nsecond) store_tile(sT + (kHalf + i) * kTileBytes, addr, rv[i]);
    if (c + 1 < kChunks && p0 + kChunk < npages) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i)
        if (!kTailAware || p0 + kChunk + i < npages)
          load_raw<false>(k_data, k_scale, sPageOff[p0 + kChunk + i], addr, rk[i]);
    }
    if constexpr (kSplit != 1) {
      mbarrier_wait(mbarA, phaseA);
      phaseA ^= 1;
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    if ((!kTailAware || nsecond > 0) && warp == 0 && elect_sync()) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        if (kTailAware && i >= nsecond) continue;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          mma_f8(taddrO, make_smem_desc(tile_a + (kHalf + i) * kTileBytes + kk * 32 * 128),
                 make_smem_desc(p_a + (kHalf + i) * 2048 + kk * 32), pv_desc, 1);
      }
      if constexpr (kSplit == 1)
        commit_mma(mbarA);
      else
        commit_mma(mbarB);
    }
    if (!kTailAware || nsecond > 0) {
      if constexpr (kSplit == 1) {
        mbarrier_wait(mbarA, phaseA);
        phaseA ^= 1;
      } else {
        mbarrier_wait(mbarB, phaseB);
        phaseB ^= 1;
      }
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");
  }

  // Materialize the completed local partial for the output epilogue and, for a
  // split cluster, peer CTA consumption.
  if (warp < 4 && npages > 0) {
    float o[16];
    tmem_ld16(taddrO, o);
    tmem_wait_ld();
    float* dst_o = sO + warp * 32 + lane;
#pragma unroll
    for (int h = 0; h < 16; ++h) dst_o[h * kOStride] = o[h];
  }
  if (lane == 0 && warp < kHeadsPerKV) {
    if constexpr (kSplit == 1)
      sSum[warp] = run_sum;
    else
      sStat[warp] = make_float2(run_sum, run_max);
  }
  __syncthreads();
  // All four reader warps have completed their TMEM loads before warp 1
  // releases the allocation.  Reuse the statistics-publication barrier so
  // the lifetime fix adds no synchronization to the epilogue.
  if (warp == 1) {
    constexpr int kCols = (kChunk * 16 + 16) <= 64 ? 64 : 128;
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_slot),
                 "r"(kCols));
  }

  __nv_bfloat16* dst =
      output + (static_cast<size_t>(qi) * kNumQHeads + hkv * kHeadsPerKV) * kHeadDim;

  if constexpr (kSplit == 1) {
    if (tid < kHeadsPerKV) sInv[tid] = sSum[tid] > 0.0f ? v_global_scale / sSum[tid] : 0.0f;
    __syncthreads();
    for (int e = tid; e < kHeadsPerKV * kHeadDim; e += kThreads) {
      const int h = e >> 7;
      const int d = e & (kHeadDim - 1);
      dst[e] = __float2bfloat16_rn(sO[h * kOStride + d] * sInv[h]);
    }
  } else {
    const uint32_t aStat = static_cast<uint32_t>(__cvta_generic_to_shared(sStat));
    const uint32_t aO = static_cast<uint32_t>(__cvta_generic_to_shared(sO));
    if (tid < kSplit) sRemoteO[tid] = map_rank(aO, tid);
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    constexpr int kHeadsPerRank = kHeadsPerKV / kSplit;
    const int first_head = rank * kHeadsPerRank;
    if (tid < kHeadsPerRank) {
      const int h = first_head + tid;
      float2 st[kSplit];
      float gmax = -FLT_MAX;
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        st[r] = ld_dsmem_v2(map_rank(aStat, r) + h * sizeof(float2));
        gmax = fmaxf(gmax, st[r].y);
      }
      float denom = 0.0f;
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        const float a = ex2(st[r].y - gmax);
        sScale[r * kHeadsPerRank + tid] = a;
        denom = fmaf(st[r].x, a, denom);
      }
      sInv[h] = denom > 0.0f ? v_global_scale / denom : 0.0f;
    }
    __syncthreads();
    for (int e = tid; e < kHeadsPerRank * (kHeadDim / 4); e += kThreads) {
      const int hl = e >> 5;
      const int h = first_head + hl;
      const int d = (e & 31) * 4;
      const int oe = h * kOStride + d;
      float2 num01 = make_float2(0.0f, 0.0f);
      float2 num23 = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int r = 0; r < kSplit; ++r) {
        const float4 x = ld_dsmem_v4(sRemoteO[r] + oe * sizeof(float));
        const float sc = sScale[r * kHeadsPerRank + hl];
        const float2 sc2 = make_float2(sc, sc);
        num01 = fma_f32x2(make_float2(x.x, x.y), sc2, num01);
        num23 = fma_f32x2(make_float2(x.z, x.w), sc2, num23);
      }
      const float inv = sInv[h];
      BF16x2Bits out01, out23;
      out01.b = __floats2bfloat162_rn(num01.x * inv, num01.y * inv);
      out23.b = __floats2bfloat162_rn(num23.x * inv, num23.y * inv);
      *reinterpret_cast<uint64_t*>(dst + h * kHeadDim + d) =
          static_cast<uint64_t>(out01.u) | (static_cast<uint64_t>(out23.u) << 32);
    }
    cluster.sync();
  }
}

// ---------------------------------------------------------------------------
// Launch.  Same configure-then-launch discipline as the general family.
//
// `seqlen_q == 1` is a PRECONDITION, not a runtime branch: it is one of the
// conjuncts of `selects_pinned_path`, and the binding does not take this family
// without it.  The multi-token instantiations that used to sit here were
// unreachable for the same reason, and their shared-memory attributes were
// never configured, so reaching one would have failed the launch rather than
// run slowly.  They are gone; the parametric family serves seqlen_q > 1.
// ---------------------------------------------------------------------------
cudaError_t launch(const void* q, const void* k_data, const void* v_data, const void* k_scale,
                   const void* v_scale, const int* q2k_indices, const int* page_table,
                   const int* seqused_k, int total_q, int q2k_head_stride, int q2k_token_stride,
                   int num_pages, int page_bytes, int data_head_stride, int scale_head_stride,
                   int causal, float softmax_scale, float k_global_scale, float v_global_scale,
                   void* output, cudaStream_t stream) {
  const int tiles = total_q * kNumKVHeads;
  int split;
  if (tiles <= 32)
    split = 8;
  else if (tiles <= 64)
    split = 4;
  else if (tiles <= 128)
    split = 2;
  else
    split = 1;
  const bool short_cache = num_pages < 32 * total_q;
  // A full-cache batch of eight requests has enough tasks to fill the device
  // with four-rank clusters; split-8 doubles cluster/DSMEM overhead without
  // adding useful latency hiding on that shape.
  if (total_q == 8 && !short_cache) split = 4;

  cudaError_t status = cudaSuccess;
  auto go = [&](auto kernel, int smem_bytes) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (status != cudaSuccess) return;
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (status != cudaSuccess) return;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(tiles * split);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = split;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = split == 1 ? nullptr : &attr;
    config.numAttrs = split == 1 ? 0 : 1;
    status = cudaLaunchKernelEx(
        &config, kernel, static_cast<const __nv_bfloat16*>(q), static_cast<const uint8_t*>(k_data),
        static_cast<const uint8_t*>(v_data), static_cast<const uint8_t*>(k_scale),
        static_cast<const uint8_t*>(v_scale), q2k_indices, page_table, seqused_k, q2k_head_stride,
        q2k_token_stride, num_pages, page_bytes, data_head_stride, scale_head_stride,
        /*seqlen_q=*/1, causal, softmax_scale, k_global_scale, v_global_scale,
        static_cast<__nv_bfloat16*>(output));
  };

  switch (split) {
    case 8:
      go(kernel_msa_decode_nvfp4_kv_paged_pinned<8, 2, true>, smem_bytes_for(2));
      break;
    case 4:
      if (short_cache)
        go(kernel_msa_decode_nvfp4_kv_paged_pinned<4, 4, true, true>, smem_bytes_for(4));
      else
        go(kernel_msa_decode_nvfp4_kv_paged_pinned<4, 4, true>, smem_bytes_for(4));
      break;
    case 2:
      if (short_cache)
        go(kernel_msa_decode_nvfp4_kv_paged_pinned<2, 2, true, true>, smem_bytes_for(2));
      else
        go(kernel_msa_decode_nvfp4_kv_paged_pinned<2, 4, true>, smem_bytes_for(4));
      break;
    default:
      go(kernel_msa_decode_nvfp4_kv_paged_pinned<1, 4, true>, smem_bytes_for(4));
      break;
  }
  return status;
}

}  // namespace pinned

}  // namespace msa_decode_nvfp4
}  // namespace flashinfer

namespace {

namespace geom = flashinfer::msa_decode_nvfp4;

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
  // Both kernel families derive addresses from these strides, so they are
  // asserted here and then PASSED, never re-derived from the shape.
  TVM_FFI_ICHECK_EQ(t.stride(0), geom::kPageBytes) << name << " page stride must be 73728";
  TVM_FFI_ICHECK_EQ(t.stride(1), head_stride) << name << " kv-head stride mismatch";
  TVM_FFI_ICHECK_EQ(t.stride(2), inner) << name << " token stride mismatch";
  TVM_FFI_ICHECK_EQ(t.stride(3), 1) << name << " must be dense in its innermost dim";
}

}  // namespace

// `pinned_path`: -1 decide here, 0 force the general family, 1 force the pinned
// one.  The Python route computes the same predicate and passes 0 or 1; a
// disagreement with what this binding derives from the tensors is a hard error,
// because the two copies existing without a cross-check is exactly how a pin
// silently stops matching its deployment.
void msa_decode_nvfp4_specialized(TensorView q, TensorView k_data, TensorView v_data,
                                  TensorView k_scale, TensorView v_scale, TensorView q2k_indices,
                                  TensorView page_table, TensorView seqused_k, TensorView output,
                                  int64_t seqlen_q, int64_t causal, double softmax_scale,
                                  double k_global_scale, double v_global_scale,
                                  int64_t pinned_path) {
  CHECK_INPUT(q);
  CHECK_CUDA(k_data);
  CHECK_CUDA(v_data);
  CHECK_CUDA(k_scale);
  CHECK_CUDA(v_scale);
  // q2k_indices is the ONE input whose outer strides are read rather than
  // derived.  Requiring full contiguity here is what forced every consumer to
  // materialise `topk.transpose(0, 1).contiguous()` on the serving path.
  CHECK_CUDA(q2k_indices);
  CHECK_INPUT(page_table);
  CHECK_INPUT(seqused_k);
  CHECK_INPUT(output);
  CHECK_DEVICE(k_data, q);
  CHECK_DEVICE(v_data, q);
  CHECK_DEVICE(k_scale, q);
  CHECK_DEVICE(v_scale, q);
  CHECK_DEVICE(q2k_indices, q);
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
  CHECK_INPUT_TYPE(page_table, dl_int32);
  CHECK_INPUT_TYPE(seqused_k, dl_int32);

  CHECK_DIM(3, q);
  CHECK_DIM(3, output);
  CHECK_DIM(3, q2k_indices);
  CHECK_DIM(2, page_table);
  CHECK_DIM(1, seqused_k);

  const int64_t total_q = q.size(0);
  TVM_FFI_ICHECK_GT(total_q, 0) << "q must contain at least one decode token";
  TVM_FFI_ICHECK_LE(total_q, 0x7fffffffLL) << "total_q must fit in int32";
  TVM_FFI_ICHECK_EQ(q.size(1), geom::kNumQHeads) << "this route serves 64 query heads";
  TVM_FFI_ICHECK_EQ(q.size(2), geom::kHeadDim) << "head_dim must be 128";
  CHECK_SHAPE(output, q);

  TVM_FFI_ICHECK_EQ(q2k_indices.size(0), geom::kNumKVHeads);
  TVM_FFI_ICHECK_EQ(q2k_indices.size(1), total_q);
  const int64_t topk = q2k_indices.size(2);
  // The parametric family reads topk at runtime; its ceiling is a REAL
  // structural bound, not a policy: every selection slot is one lane of warp
  // 0's `__ballot_sync`, and `sTokenBase` / `sPageOff` are sized by it.  A 33rd
  // slot would land in warp 1 and be dropped from the ballot with no
  // diagnostic, so it is refused by name here instead.
  TVM_FFI_ICHECK(topk >= 1 && topk <= geom::general::kSelectedCapacity)
      << "top-k must be in [1, " << geom::general::kSelectedCapacity << "], got " << topk;
  // Only the LAST dimension has to be dense; the two outer strides are passed
  // to the kernel.  This is what lets a caller hand in the transposed view of a
  // token-major (total_q, num_kv_heads, topk) selection buffer.
  TVM_FFI_ICHECK_EQ(q2k_indices.stride(2), 1)
      << "q2k_indices must be dense in its innermost (top-k) dimension";
  const int64_t q2k_head_stride = q2k_indices.stride(0);
  const int64_t q2k_token_stride = q2k_indices.stride(1);
  TVM_FFI_ICHECK(q2k_head_stride >= 0 && q2k_token_stride >= 0)
      << "q2k_indices must not be negatively strided";
  // Both kernels do the row arithmetic in int; bound the largest offset either
  // can form rather than letting a very large batch wrap silently.
  TVM_FFI_ICHECK_LE(
      (geom::kNumKVHeads - 1) * q2k_head_stride + (total_q - 1) * q2k_token_stride + topk,
      0x7fffffffLL)
      << "the q2k_indices view is too large for 32-bit addressing";
  TVM_FFI_ICHECK(seqlen_q >= 1 && total_q % seqlen_q == 0)
      << "q rows must equal batch_size * positive seqlen_q";
  const int64_t batch_size = total_q / seqlen_q;
  TVM_FFI_ICHECK_EQ(page_table.size(0), batch_size) << "page_table needs one row per request";
  const int64_t max_blocks = page_table.size(1);
  TVM_FFI_ICHECK(max_blocks >= 1 && max_blocks <= 0x7fffffffLL)
      << "page_table row width must be a positive int32";
  TVM_FFI_ICHECK_EQ(seqused_k.size(0), batch_size);

  const int64_t num_pages = k_data.size(0);
  TVM_FFI_ICHECK(num_pages >= 1 && num_pages <= 0x7fffffffLL)
      << "the page pool must be a positive int32 number of pages";
  check_page_region(k_data, "k_data", num_pages, geom::kDataDim, geom::kDataHeadStride);
  check_page_region(v_data, "v_data", num_pages, geom::kDataDim, geom::kDataHeadStride);
  check_page_region(k_scale, "k_scale", num_pages, geom::kScaleDim, geom::kScaleHeadStride);
  check_page_region(v_scale, "v_scale", num_pages, geom::kScaleDim, geom::kScaleHeadStride);

  // Layout proof.  Shape, dtype and stride cannot distinguish four views of one
  // planar page from four unrelated allocations that happen to be strided the
  // same way -- and the (4, 4) V-scale swizzle is invisible to all three.  The
  // byte offsets between the four base pointers are what pins them to the same
  // page map the cache writer used.
  TVM_FFI_ICHECK_EQ(byte_offset(k_scale, k_data), geom::kKScaleByteOffset)
      << "k_scale must be the K-scale region of the same page as k_data";
  TVM_FFI_ICHECK_EQ(byte_offset(v_data, k_data), geom::kVDataByteOffset)
      << "v_data must be the V-data region of the same page as k_data";
  TVM_FFI_ICHECK_EQ(byte_offset(v_scale, k_data), geom::kVScaleByteOffset)
      << "v_scale must be the V-scale region of the same page as k_data";

  TVM_FFI_ICHECK(std::isfinite(softmax_scale) && std::isfinite(k_global_scale) &&
                 std::isfinite(v_global_scale))
      << "softmax_scale, k_global_scale and v_global_scale must be finite";
  TVM_FFI_ICHECK(softmax_scale > 0.0 && k_global_scale > 0.0 && v_global_scale > 0.0)
      << "softmax_scale, k_global_scale and v_global_scale must be positive";
  TVM_FFI_ICHECK(causal == 0 || causal == 1) << "causal must be 0 or 1";

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  int major = 0;
  int minor = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "the specialized NVFP4 MSA decode kernel requires compute capability 10.0 or 10.3, got "
      << major << "." << minor;

  const geom::PinnedEnvelope envelope{static_cast<int>(q.size(1)),
                                      static_cast<int>(k_data.size(1)),
                                      static_cast<int>(q.size(2)),
                                      static_cast<int>(k_data.size(2)),
                                      static_cast<int>(q2k_indices.size(2)),
                                      static_cast<int>(max_blocks),
                                      static_cast<int>(seqlen_q),
                                      static_cast<int>(total_q),
                                      static_cast<int>(num_pages)};
  const bool pinned_here = geom::selects_pinned_path(envelope);
  TVM_FFI_ICHECK(pinned_path == -1 || pinned_path == 0 || pinned_path == 1)
      << "pinned_path must be -1 (decide here), 0 (general) or 1 (pinned), got " << pinned_path;
  TVM_FFI_ICHECK(pinned_path == -1 || (pinned_path == 1) == pinned_here)
      << "the caller's pinned-path decision (" << pinned_path << ") disagrees with the geometry "
      << "this binding sees (" << (pinned_here ? 1 : 0)
      << "); the Python mirror of selects_pinned_path has drifted from the C++ one";
  const bool use_pinned = pinned_path == -1 ? pinned_here : pinned_path == 1;

  const int page_bytes = static_cast<int>(k_data.stride(0));
  const int data_head_stride = static_cast<int>(k_data.stride(1));
  const int scale_head_stride = static_cast<int>(k_scale.stride(1));
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status;
  if (use_pinned) {
    status = geom::pinned::launch(
        q.data_ptr(), k_data.data_ptr(), v_data.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        static_cast<const int*>(q2k_indices.data_ptr()),
        static_cast<const int*>(page_table.data_ptr()),
        static_cast<const int*>(seqused_k.data_ptr()), static_cast<int>(total_q),
        static_cast<int>(q2k_head_stride), static_cast<int>(q2k_token_stride),
        static_cast<int>(num_pages), page_bytes, data_head_stride, scale_head_stride,
        static_cast<int>(causal != 0), static_cast<float>(softmax_scale),
        static_cast<float>(k_global_scale), static_cast<float>(v_global_scale), output.data_ptr(),
        stream);
  } else {
    const int num_q_heads = static_cast<int>(q.size(1));
    const int num_kv_heads = static_cast<int>(k_data.size(1));
    status = geom::general::launch(
        q.data_ptr(), k_data.data_ptr(), v_data.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
        static_cast<const int*>(q2k_indices.data_ptr()),
        static_cast<const int*>(page_table.data_ptr()),
        static_cast<const int*>(seqused_k.data_ptr()), static_cast<int>(total_q),
        static_cast<int>(q2k_head_stride), static_cast<int>(q2k_token_stride),
        static_cast<int>(num_pages), num_q_heads, num_kv_heads, num_q_heads / num_kv_heads,
        static_cast<int>(k_data.size(2)), static_cast<int>(topk), static_cast<int>(max_blocks),
        page_bytes, data_head_stride, scale_head_stride, static_cast<int>(seqlen_q),
        static_cast<int>(causal != 0), static_cast<float>(softmax_scale),
        static_cast<float>(k_global_scale), static_cast<float>(v_global_scale), output.data_ptr(),
        stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "msa_decode_nvfp4_specialized launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(msa_decode_nvfp4_specialized, msa_decode_nvfp4_specialized);
