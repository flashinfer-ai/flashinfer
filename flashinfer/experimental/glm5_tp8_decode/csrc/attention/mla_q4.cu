/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "ll_poll.cuh"
#include "mla_q4_types.cuh"

// Per-launch values that may come from device words.  Op bodies read
// them from these two shared scalars (set by thread 0 before the prologue barrier), so every other
// MegaArgs field stays a constant-bank operand.
__shared__ uint32_t g_mega_epoch;
__shared__ int32_t g_mega_cur_pos;
__shared__ uint32_t g_ll_tag;         // MegaArgs::ll_tag: LL ids record tag for this step
__shared__ const uint8_t* g_ll_recv;  // MegaArgs::ll_recv: this rank's recv_buf (LL records {id0,
                                      // tag, id1, tag}, 16 B, row = cta*32 + lane)
__shared__ int32_t
    g_ll_ids[64];  // the 64 selected ids of this K3 CTA when consumed from the LL rows (kK3OptRows)
// Default persistent K3: the producer warpgroup stages this CTA's ten
// post_q_nope head rows while the consumers run the PE score MMA.  A separate
// barrier is required because the four page-ring barriers continue directly
// into the K4/K5 weight stream.
__shared__ __align__(8) uint64_t g_k3_qnope_bar;

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace flashinfer::glm5_tp8::mla_q4 {

// ---- bf16 helpers --------------------------------
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 to_bf16(float value) { return __float2bfloat16_rn(value); }

// ---- MMA fragments -----------------------------------
// The packed 16x16 byte tile is already in the lane order required by the
// warp-level m16n8k16 BF16 MMA A operand.  Decode each lane's eight E4M3 values
// directly into four BF16x2 registers; the four query rows occupy N=0..3.
union Bf16Pair {
  __nv_bfloat162 pair;
  uint32_t bits;
};

struct HmmaAcc {
  float value[4];
};

struct RawFp8x8 {
  uint32_t lo;
  uint32_t hi;
};

// ---- helpers -----------------------------------------------------------------------
// (q_b_proj + W_UK: packed f32 pair / bf16x2 pack; packed f32x2 FMA/ADD, bf16 pair extraction and
// vectorized RMS-normalize staging; ldmatrix fragment loaders.)
union F32Pair {
  uint64_t bits;
  float value[2];
};

// PTX packs its first f32 operand into the high half.  The reversed operands
// keep the lower-address row in bits 15:0 for a little-endian b32 store.
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  uint32_t bits;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(high), "f"(low));
  return bits;
}

__device__ __forceinline__ float bf16_pair_low(uint32_t bits) {
  return __uint_as_float(bits << 16);
}

__device__ __forceinline__ float bf16_pair_high(uint32_t bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

// SM100 can issue two independent FP32 FMAs from one packed b64 operand.  Pair
// query-token accumulators while preserving each token's low/high
// BF16 element order and therefore its exact sequence of FP32 roundings.
__device__ __forceinline__ uint64_t pack_f32x2(float low, float high) {
  return static_cast<uint64_t>(__float_as_uint(low)) |
         (static_cast<uint64_t>(__float_as_uint(high)) << 32);
}

__device__ __forceinline__ uint64_t fma_f32x2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t result;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(result) : "l"(a), "l"(b), "l"(c));
  return result;
}

__device__ __forceinline__ uint64_t add_f32x2(uint64_t a, uint64_t b) {
  uint64_t result;
  asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
  return result;
}

__device__ __forceinline__ uint64_t accumulate_bf16_pair_f32x2(uint64_t sums, uint32_t token0,
                                                               uint32_t token1) {
  uint64_t values = pack_f32x2(bf16_pair_low(token0), bf16_pair_low(token1));
  sums = fma_f32x2(values, values, sums);
  values = pack_f32x2(bf16_pair_high(token0), bf16_pair_high(token1));
  return fma_f32x2(values, values, sums);
}

__device__ __forceinline__ uint64_t accumulate_bf16x8_f32x2(uint64_t sums, uint4 token0,
                                                            uint4 token1) {
  sums = accumulate_bf16_pair_f32x2(sums, token0.x, token1.x);
  sums = accumulate_bf16_pair_f32x2(sums, token0.y, token1.y);
  sums = accumulate_bf16_pair_f32x2(sums, token0.z, token1.z);
  return accumulate_bf16_pair_f32x2(sums, token0.w, token1.w);
}

__device__ __forceinline__ float f32x2_low(uint64_t packed) {
  return __uint_as_float(static_cast<uint32_t>(packed));
}

__device__ __forceinline__ float f32x2_high(uint64_t packed) {
  return __uint_as_float(static_cast<uint32_t>(packed >> 32));
}

__device__ __forceinline__ uint32_t normalize_bf16_pair(uint32_t bits, float inverse,
                                                        float gamma_low, float gamma_high) {
  const float low = __fmul_rn(__fmul_rn(bf16_pair_low(bits), inverse), gamma_low);
  const float high = __fmul_rn(__fmul_rn(bf16_pair_high(bits), inverse), gamma_high);
  uint32_t packed;
  // cvt.bf16x2 places its first FP32 source in the high half and its second
  // source in the low half, hence the intentionally reversed operands.
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
  return packed;
}

__device__ __forceinline__ uint4 normalize_bf16x8(uint4 packed, float inverse, float4 gamma0,
                                                  float4 gamma1) {
  return make_uint4(normalize_bf16_pair(packed.x, inverse, gamma0.x, gamma0.y),
                    normalize_bf16_pair(packed.y, inverse, gamma0.z, gamma0.w),
                    normalize_bf16_pair(packed.z, inverse, gamma1.x, gamma1.y),
                    normalize_bf16_pair(packed.w, inverse, gamma1.z, gamma1.w));
}

__device__ __forceinline__ void stage_retained_x_chunk(__nv_bfloat16* xnorm, int col, float4 gamma0,
                                                       float4 gamma1, float inverse0,
                                                       float inverse1, float inverse2,
                                                       float inverse3, uint4 x0, uint4 x1, uint4 x2,
                                                       uint4 x3) {
  *reinterpret_cast<uint4*>(xnorm + col) = normalize_bf16x8(x0, inverse0, gamma0, gamma1);
  *reinterpret_cast<uint4*>(xnorm + smem::kXnormStride + col) =
      normalize_bf16x8(x1, inverse1, gamma0, gamma1);
  *reinterpret_cast<uint4*>(xnorm + 2 * smem::kXnormStride + col) =
      normalize_bf16x8(x2, inverse2, gamma0, gamma1);
  *reinterpret_cast<uint4*>(xnorm + 3 * smem::kXnormStride + col) =
      normalize_bf16x8(x3, inverse3, gamma0, gamma1);
}

__device__ __forceinline__ void fp8_to_bf16_fragment(const RawFp8x8& raw, uint32_t (&fragment)[4]) {
  asm volatile(
      "{\n\t"
      ".reg .b16 h0,h1,h2,h3;\n\t"
      "mov.b32 {h0,h1},%4;\n\t"
      "mov.b32 {h2,h3},%5;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0,h0;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1,h1;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %2,h2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %3,h3;\n\t"
      "}"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "r"(raw.lo), "r"(raw.hi));
}

__device__ __forceinline__ uint32_t hmma_smem_addr(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void hmma_ldsm_x4(uint32_t (&value)[4], const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void hmma_ldsm_x4(uint32_t (&value)[2][2],
                                             const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(value[0][0]), "=r"(value[0][1]), "=r"(value[1][0]), "=r"(value[1][1])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void hmma_ldsm_x2(uint32_t (&value)[2], const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
               : "=r"(value[0]), "=r"(value[1])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void hmma_ldsm_x4_trans(uint32_t (&value)[4],
                                                   const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
      "{%0,%1,%2,%3}, [%4];"
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void hmma_load_a_16x16(const __nv_bfloat16* base, int stride,
                                                  uint32_t (&value)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int source_row = (lane & 7) + ((matrix & 1) << 3);
  const int source_col = (matrix >> 1) << 3;
  hmma_ldsm_x4(value, base + source_row * stride + source_col);
}

// Load a row-major 16x16 tile transposed into the row-major MMA A fragment.
// x4.trans needs source quadrants in TL, TR, BL, BR address order.
__device__ __forceinline__ void hmma_load_a_16x16_trans(const __nv_bfloat16* base, int stride,
                                                        uint32_t (&value)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int source_row = (lane & 7) + ((matrix >> 1) << 3);
  const int source_col = (matrix & 1) << 3;
  hmma_ldsm_x4_trans(value, base + source_row * stride + source_col);
}

__device__ __forceinline__ void hmma(HmmaAcc& accumulator, const uint32_t (&a)[4],
                                     const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(accumulator.value[0]), "+f"(accumulator.value[1]), "+f"(accumulator.value[2]),
        "+f"(accumulator.value[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- warp reductions ------------------------------------
__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// ---- block reductions  emulated for a 384-thread block ------------

// lane 0 -> scratch[warp], then warp 0 sums scratch[0..11] with the same tree.
// The consumer group has 256 threads, so each thread carries two virtual
// values:  for virtual thread tid (virtual warp w = tid>>5, 0..7) and  for
// virtual thread tid+256 (virtual warp w+8, only tid < 128).  Real warp w
// reduces virtual warp w, and real warps 0..3 also reduce virtual warps 8..11.

__device__ __forceinline__ float block_sum_v384(const OpCtx& ctx, float v0, float v1,
                                                float* scratch) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  float w0 = warp_sum(v0);
  float w1 = warp_sum(v1);
  if (lane == 0) {
    scratch[warp] = w0;
    if (warp < (kSourceBlock - kOpThreads) / 32) scratch[warp + kOpWarps] = w1;
  }
  ctx.sync();
  if (warp == 0) {
    float total = lane < (kSourceBlock >> 5) ? scratch[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) scratch[0] = total;
  }
  ctx.sync();
  return scratch[0];
}

}  // namespace flashinfer::glm5_tp8::mla_q4
// weight-page lists consumed by the producer-warp page ring AND walked in the same order by the
// consumer op bodies (acquire/release).
//   qkv_a_proj  82 blocks x 6 pages of 0x8080 B  (K1: CTA c < 82 owns 32-row block c)
//   q_b_proj    80 CTAs   x 2 pages of 0x8080 B  (K2QbProj)
//   W_UK        80 CTAs   x 1 page  of 0x3080 B  (K2Uk, query absorption)
//   W_UV        80 CTAs   x 1 page  of 0x4080 B  (K4)
//   o_proj     128 CTAs   x 5 pages of 0x6080 B  (K5)
// Every page is copied whole (one cp.async.bulk); bytes is a multiple of 16.
// rank7 pads (heads 4..9) are skipped with bytes = 0 exactly where the math
// skips them: a bytes == 0 entry is not a ring item on either side.
// CTAs outside an op's page set get no entries.

// t = cta, cta+128 (kK1Ctas) from global memory; its list keeps the per-tile
// half-page + scale segments so the drained bytes equal what the body reads.

namespace flashinfer::glm5_tp8::mla_q4 {

static __device__ bool op_page(int op, const MegaArgs& args, int cta, int index,
                               const uint8_t** ptr, int* bytes) {
  switch (op) {
    case kOpK1: {
      if (cta >= kQkvAProjCtaBlocks || index >= kQkvAProjPagesPerBlock) return false;
      *ptr = args.qkv_a_proj +
             static_cast<int64_t>(cta * kQkvAProjPagesPerBlock + index) * kPageBytes8080;
      *bytes = kPageBytes8080;
      return true;
    }
    case kOpK2QbProj: {
      if (cta >= kK2Ctas || index >= 2) return false;
      int row_base = 32 * cta;
      int head =
          row_base < kQbProjNopeRows ? row_base / kNope : (row_base - kQbProjNopeRows) / kRope;
      bool active = args.rank != 7 || head < 4;
      *ptr = args.q_b_proj + static_cast<int64_t>(2 * cta + index) * kPageBytes8080;
      *bytes = active ? kPageBytes8080 : 0;
      return true;
    }
    case kOpK2Uk: {
      if (cta >= kK2Ctas || index != 0) return false;
      int head = (64 * cta) / kKvlora;
      bool active = args.rank != 7 || head < 4;
      *ptr = args.w_uk + static_cast<int64_t>(cta) * kUkPageBytes;
      *bytes = active ? kUkPageBytes : 0;
      return true;
    }
    case kOpK2Fused: {
      // Fused q-chain (kK2Fused* in mla_q4_types.cuh): cohort k of head h streams the two 0x8080
      // pages of the NOPE block 6h + k/2, then (k < 8) the W_UK page 8h + k or (k >= 8) the two
      // pages of PE block 60 + 2h + (k-8)/2.  rank7 heads 4..9: bytes 0 (not items), matching the
      // zeroed outputs.
      if (cta >= kK2FusedCtas) return false;
      const int head = cta / kK2FusedCohort;
      const int cohort = cta - head * kK2FusedCohort;
      const bool active = args.rank != 7 || head < 4;
      if (index < 2) {
        const int block = head * (kNope / 32) + (cohort >> 1);
        *ptr = args.q_b_proj + static_cast<int64_t>(2 * block + index) * kPageBytes8080;
        *bytes = active ? kPageBytes8080 : 0;
        return true;
      }
      if (cohort < kK2FusedUkPerHead) {
        if (index != 2) return false;
        *ptr = args.w_uk + static_cast<int64_t>(head * kK2FusedUkPerHead + cohort) * kUkPageBytes;
        *bytes = active ? kUkPageBytes : 0;
        return true;
      }
      if (index >= 4) return false;
      const int pe_block =
          kQbProjNopeRows / 32 + head * (kRope / 32) + ((cohort - kK2FusedUkPerHead) >> 1);
      *ptr = args.q_b_proj + static_cast<int64_t>(2 * pe_block + (index - 2)) * kPageBytes8080;
      *bytes = active ? kPageBytes8080 : 0;
      return true;
    }
    case kOpK1Tails:
    case kOpK3:
      return false;  // no weight pages; the K3 KV gather is issued by producer_stream_k3,
                     // 2 ring items
    case kOpK4: {
      if (cta >= kK4Ctas || index >= kK4FusedPages) return false;
      const int head = cta >> 3;
      const int row_half = cta & 1;
      const bool active = args.rank != 7 || head < 4;
      *ptr = args.w_uv +
             static_cast<int64_t>(8 * head + kK4FusedPages * row_half + index) * kUvPageBytes;
      *bytes = active ? kUvPageBytes : 0;
      return true;
    }
    case kOpK5:
    case kOpK5OProj:
    case kOpK5OProjOnly: {
      if (cta >= kK5Ctas || index >= 5) return false;
      *ptr = args.o_proj + static_cast<int64_t>(cta * 5 + index) * kOProjPageBytes;
      // The o_proj consumer reads the 0x6000 B matrix and the 12 scales (48 B) at +0x6000; the blob
      // keeps its 0x6080 page stride while the bulk transaction skips the unused 80 B tail
      // (the same for every K5 op id).
      *bytes = kOProjPageBulkBytes;
      return true;
    }
    default:
      return false;
  }
}

}  // namespace flashinfer::glm5_tp8::mla_q4

//   producer_stream_op(op):  for every page of this CTA's op (op_page order,
//     bytes > 0), all 128 producer threads wait empty[slot] with the flipped
//     parity (first round passes), then producer thread 0 alone does
//     mbarrier.arrive.expect_tx(full[slot], bytes) + ONE cp.async.bulk of the
//     whole page into the slot, and lane 0 of each other producer warp does a
//     plain mbarrier.arrive on full[slot] (4 arrivals + tx bytes complete the
//     phase).  No bar.sync anywhere on the weight path.  The item counter is
//     monotonic across all ops of the CTA; the 4-slot depth bounds the lookahead
//     (the producer may run up to four pages into later ops -- weights are
//     constant inputs, so that is safe).
//   producer_stream_k3: the attention CTA's KV
//     gather.  CTA c < 128 owns (query c >> 5, split c & 31); after the ids /
//     kv-cache words (C2 + C3 in the persistent kernel; IdsRx in kModeChain) the
//     four producer warps copy up to 16 KV rows each (one 1 KB cp.async.bulk per
//     row, id validated against length / cache_rows / cur_pos) into TWO ring
//     slots ([32 rows][512 + 8 pad] bf16); every producer warp arrives once on
//     both slots' full barriers (4 arrivals each) and carries the tx bytes of
//     its own rows.  This is the only place a producer waits on an arena word,
//     and the words it waits on are arrived by CTAs that never wait on this
//     CTA's ring (tail CTAs / K3 CTA 0), so liveness is preserved.
//   producer_main: the per-CTA op order (K1 -> K2 fused -> K3 gather -> K4 -> K5)
//     shared by the persistent and null_stream modes (the K3 gather is skipped
//     in null_stream).
//   consumer_drain_op: acquire + release every page of an op without reading
//     it, for consumers that do not take the op's weights from the ring
//     (null_stream) but must keep the ring live.
// The producer and the consumers must enumerate exactly the same (op, page)
// items in the same order; both go through op_page with the same skip rule
// (K3: exactly two items per attention CTA c < 128, none for c >= 128).

namespace flashinfer::glm5_tp8::mla_q4 {
constexpr size_t kRecQnd = arena::kScratch;
static_assert(size_t(kHeads) * kQlen * (kNope / 4) * 16 <= arena::kScratchBytes,
              "q_nope-down records fit the record scratch");
// K4 -> K5 hand-off (persistent mode): each K4 CTA publishes its 128 post_uv values as
// launch-tagged records (four BF16 each) and every K5 warp polls the records of the columns it owns
// straight into its activation slice, instead of bar 5 -> bulk store -> bulk wait -> C7 release ->
// C7 poll -> activation reload.  post_uv is still written; C7 keeps counting (relaxed). K2 -> K3
// query hand-off (persistent mode): the absorbed q_nope (W_UK) and the rotated q_pe (PE cohorts)
// are published as the same launch-tagged b128 records, four BF16 per record, and every K3 split
// CTA polls its query's 1440 records straight into its query tile (one L2 round trip after the
// store) instead of bulk store -> bulk wait -> C4/C4Pe release -> counter poll -> bulk load.
// post_qnope / post_qpe are still written with the same values.
constexpr size_t kRecQnope = kRecQnd + 32768;
constexpr size_t kRecQpe = kRecQnope + size_t(kQlen) * kHeads * (kKvlora / 4) * 16;
static_assert(kRecQpe + size_t(kQlen) * kHeads * (kRope / 4) * 16 <= kRecQnd + 196608,
              "query records before post_uv records");
constexpr bool kK3QueryRecords = true;
constexpr size_t kRecUv = kRecQnd + 196608;
static_assert(kRecQnd + size_t(kHeads) * kQlen * (kNope / 4) * 16 <= kRecUv &&
                  kRecUv + size_t(kQlen) * kUvRows / 4 * 16 <=
                      arena::kScratch + arena::kScratchBytes,
              "post_uv records fit the record scratch");
__device__ __forceinline__ void rec_store_b128(void* address, uint32_t w0, uint32_t w1,
                                               uint32_t tag) {
  const uint64_t lo = (uint64_t(tag) << 32) | w0;
  const uint64_t hi = (uint64_t(tag) << 32) | w1;
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.gpu.global.b128 [%0], r;\n\t}" ::
          "l"(address),
      "l"(lo), "l"(hi)
      : "memory");
}
__device__ __forceinline__ uint2 rec_poll(const void* address, uint32_t tag) {
  uint64_t lo, hi;
  for (;;) {
    asm volatile(
        "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
        : "=l"(lo), "=l"(hi)
        : "l"(address)
        : "memory");
    if (static_cast<uint32_t>(lo >> 32) == tag && static_cast<uint32_t>(hi >> 32) == tag) break;
  }
  return make_uint2(static_cast<uint32_t>(lo), static_cast<uint32_t>(hi));
}

// K3's validated-id slab sits above every live K2 allocation, so the
// producer can fill it before handing its C2/C3 acquire to the consumers.
// The canonical K3 layout below statically checks these early constants.
constexpr uint32_t kK3ProducerScratchIds = 34816u;
constexpr uint32_t kK3ProducerScratchAux =
    kK3ProducerScratchIds + kK3OptRows * uint32_t(sizeof(int32_t));

__device__ __forceinline__ void store_shared_i32(uint32_t address, int value) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(address), "r"(value) : "memory");
}

__device__ __forceinline__ int load_shared_i32(uint32_t address) {
  int value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
  return value;
}

// All weight and selected-KV streams are single-touch within one cold call.
// Give them the permitted evict-first priority so they do not displace
// activations, counters, or partials that are reused later in-call.
__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void bulk_copy_g2s_l2(uint32_t dst_smem, const void* src_global,
                                                 uint32_t bytes, uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1], %2, [%3], %4;" ::"r"(dst_smem),
      "l"(src_global), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}
static __device__ void producer_stream_k3(const MegaArgs& args, int cta, uint32_t ring_s,
                                          uint32_t full_s, uint32_t empty_s, uint32_t& item) {
  if (cta >= kK3SplitCtas || args.mode == kModeNullStream) return;

  const int ptid = static_cast<int>(threadIdx.x) - kOpThreads;
  const int pwarp = ptid >> 5;
  const int lane = ptid & 31;
  const int query = cta >> 5;
  const int split = cta & 31;

  uint8_t* local = args.peers[args.rank];
  const bool producer_stages_ids = args.ids_ready && args.mode == kModePersistent;
  const bool ll_mode =
      args.ll_ids && g_ll_recv != nullptr && (g_mega_cur_pos + query + 1) > kSelected;

  // L2 span prefetch of the live causal cache window: while producer warp 0
  // polls the C2/C3 boundaries, warp 1's lane 0 streams this CTA's dense
  // slice of [0, cur_pos + 4) rows of both caches so the id-directed row
  // gather below completes against L2 instead of scattered cold HBM reads.
  // Bounded to spans of the same order as the gather itself (<= 8192 rows);
  // larger caches keep the plain gather, correct for the whole domain.
  // L2 is coherent with the kv-norm/RoPE tail writes that C3/C2 order, so a
  // prefetched line never masks those rows' freshly written values.
  // A complete dense gather already streams the live contiguous cache.
  // Keep the span prefetch for sparse selections, whose rows are scattered.
  if (ptid == 32 && g_mega_cur_pos + query + 1 > kSelected) {
    const int span_rows = min(args.cache_rows, g_mega_cur_pos + kQlen);
    if (span_rows > 0 && span_rows <= 8192) {
      const size_t kv_bytes = size_t(span_rows) * (kKvlora * 2);
      const size_t pe_bytes = size_t(span_rows) * (kRope * 2);
      const size_t b = (kv_bytes * size_t(cta) / kK3SplitCtas) & ~size_t(15);
      const size_t e = (cta + 1 == kK3SplitCtas)
                           ? kv_bytes
                           : ((kv_bytes * size_t(cta + 1) / kK3SplitCtas) & ~size_t(15));
      if (e > b) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(
                         reinterpret_cast<const uint8_t*>(args.kv_cache) + b),
                     "r"(static_cast<uint32_t>(e - b))
                     : "memory");
      }
      const size_t pb = (pe_bytes * size_t(cta) / kK3SplitCtas) & ~size_t(15);
      const size_t pf = (cta + 1 == kK3SplitCtas)
                            ? pe_bytes
                            : ((pe_bytes * size_t(cta + 1) / kK3SplitCtas) & ~size_t(15));
      if (pf > pb) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(
                         reinterpret_cast<const uint8_t*>(args.pe_cache) + pb),
                     "r"(static_cast<uint32_t>(pf - pb))
                     : "memory");
      }
    }
  }
  {
    // Persistent kernel (ids_ready): the ids were consumed by tail CTA 136 (C2) and the kv_cache
    // rows cur_pos..cur_pos+3 were written by the kv-norm tails (C3) -- wait for both words
    // whatever the ids source (ids_direct still gathers freshly written cache rows).  With
    // ids_ready = 0 (kModeChain): kv_cache is final by stream order;
    // wait for CTA 0's consume (IdsRx) unless the ids come straight from the host tensors
    // (ids_direct). Relaxed poll, then one acquire fence.
    const uint32_t* counters = reinterpret_cast<const uint32_t*>(local + arena::kCountersPadded);
    const bool needs_wait = args.ids_ready || !args.ids_direct;
    // One producer warp polls, then a 128-thread named barrier broadcasts the acquire to the other
    // three. Keeping the idle producer warps off the load loop leaves more issue slots for the
    // concurrently-running consumer math.
    if (pwarp == 0) {
      if (args.ids_ready) {
        const uint32_t ids_target = ArrivalCounter::target(kArriveC2, g_mega_epoch);
        const uint32_t kv_target = ArrivalCounter::target(kArriveC3, g_mega_epoch);
        while (!word_reached(counters + cnt_word(kCntC2), ids_target) ||
               !word_reached(counters + cnt_word(kCntC3), kv_target)) {
          spin_backoff();
        }
      } else if (!args.ids_direct) {
        const uint32_t target = ArrivalCounter::target(kArriveIdsRx, g_mega_epoch);
        while (!word_reached(counters + cnt_word(kCntIdsRx), target)) spin_backoff();
      }
      if (needs_wait) fence_acquire_gpu();
    }

    // Rank 0 publishes this CTA's LL ids records into recv_buf + (cta*32 + lane)*16.  Producer
    // warp 0 polls them;
    // bar 6 below publishes g_ll_ids to the other producer warps and barrier 7 to the consumers.
    if (ll_mode && pwarp == 0) {
      const uint8_t* ll_row = g_ll_recv + (size_t(cta) * 32 + size_t(lane)) * 16;
      uint64_t lo = 0, hi = 0;
      ld_relaxed_sys_b128(ll_row, lo, hi);
      uint32_t spins = 0;
      while (static_cast<uint32_t>(lo >> 32) != g_ll_tag ||
             static_cast<uint32_t>(hi >> 32) != g_ll_tag) {
        ld_relaxed_sys_b128(ll_row, lo, hi);
        if (++spins == (1u << 23)) {  // cold deadline branch: the final reload may already be valid
          if (ll_wire::deadline_missing_tags(spins, 1u << 23, static_cast<uint32_t>(lo >> 32),
                                             static_cast<uint32_t>(hi >> 32), g_ll_tag)) {
            ll_wire::timeout_trap();
          }
          break;
        }
      }
      g_ll_ids[2 * lane] = static_cast<int32_t>(lo);
      g_ll_ids[2 * lane + 1] = static_cast<int32_t>(hi);
      // LL consume: the ids tag is a constant (1) shared by every layer and step, so a row left
      // behind by the previous layer / step satisfies the poll above before rank 0 has published
      // this layer's ids. Clear this lane's tag words right after consuming the row, so the next
      // poll on this row can only be satisfied by a fresh publish.  Rank 0 cannot re-publish before
      // this CTA's K5 push, so no fresh row is ever erased.  One 16 B sys-scope store per lane;
      // keeps the ids words (harmless).
      if (!(lo == 0xffffffffffffffffull && hi == 0xffffffffffffffffull)) {
        asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                         const_cast<uint8_t*>(ll_row)),
                     "r"(static_cast<uint32_t>(lo)), "r"(0u), "r"(static_cast<uint32_t>(hi)),
                     "r"(0u)
                     : "memory");
      }
    }
    if (needs_wait || ll_mode) {
      asm volatile("bar.sync 6, 128;" ::: "memory");
    }
  }

  const size_t ids_parity = arena::kIdsRx + size_t(g_mega_epoch & 1u) * arena::kIdsRxParityBytes;
  const int32_t* lengths = args.ids_direct ? args.selected_lengths_src
                                           : wsp<int32_t>(local, ids_parity + arena::kIdsRxLengths);
  const int32_t* ids = args.ids_direct ? args.selected_ids_src : wsp<int32_t>(local, ids_parity);

  // The public operation guarantees unique causal IDs and, when this
  // query has at most 2048 visible tokens, includes every visible token.
  // Traversing that complete set in token order preserves the operation
  // without assuming anything about the supplied permutation. Decide per
  // query because one invocation can straddle the dense/sparse boundary;
  // sparse queries still consume their live selected IDs below.  For such a
  // guaranteed-complete prefix the live length is the public cur_pos scalar
  // plus the query offset, so the cold lengths read leaves the gather's
  // critical path entirely; sparse queries keep their live lengths read.
  const bool canonical_dense = g_mega_cur_pos + query + 1 <= kSelected;
  const int length = min(
      kSelected, (canonical_dense || args.dense_ids) ? g_mega_cur_pos + query + 1 : lengths[query]);
  // Balanced split geometry (see k3_split_row_lo in mla_q4_types.cuh): contiguous ascending
  // 16-row granule ranges per split; 64-key splits exactly on full
  // selections.
  const int rows_per_split = k3_split_row_cnt(length, split);
  const int slot_base = k3_split_row_lo(length, split);
  const uint32_t staged_ids_s = ring_s - smem::kOpScratchBytes + kK3ProducerScratchIds;

  // Validate once in the producer warpgroup and publish all 64 entries plus
  // the length through the named barrier that already joins the two
  // warpgroups, so the K3 consumers skip those 64 global loads, bounds checks
  // and stores and their own CTA barrier.
  if (producer_stages_ids) {
    const int row = pwarp * 16 + lane;
    if (lane < 16) {
      const int slot = slot_base + row;
      int id = -1;
      if (row < rows_per_split && slot < length) {
        const int candidate =
            canonical_dense ? slot : (ll_mode ? g_ll_ids[row] : ids[query * kSelected + slot]);
        if (candidate >= 0 && candidate < args.cache_rows && candidate <= g_mega_cur_pos + query) {
          id = candidate;
        }
      }
      store_shared_i32(staged_ids_s + row * uint32_t(sizeof(int32_t)), id);
    }
    if (ptid == 0) {
      store_shared_i32(ring_s - smem::kOpScratchBytes + kK3ProducerScratchAux, length);
    }
    // Hand the C2/C3 acquire and the completed shared slab to all 256
    // consumers.  Every producer thread arrives, then continues to KV gather.
    asm volatile("barrier.cta.arrive 7, 384;" ::: "memory");
  }

  const uint32_t first_item = item;
  const uint32_t second_item = item + 1;
  const uint32_t first_slot = first_item & (smem::kRingSlots - 1);
  const uint32_t second_slot = second_item & (smem::kRingSlots - 1);
  mbar_wait_parity(empty_s + first_slot * 8, ((first_item >> 2) & 1u) ^ 1u);
  mbar_wait_parity(empty_s + second_slot * 8, ((second_item >> 2) & 1u) ^ 1u);

  int warp_rows = rows_per_split - pwarp * 16;
  warp_rows = warp_rows < 0 ? 0 : (warp_rows > 16 ? 16 : warp_rows);
  if (slot_base >= length) warp_rows = 0;
  const bool first_half = pwarp < 2;
  const uint32_t owned_slot = first_half ? first_slot : second_slot;
  const uint32_t other_slot = first_half ? second_slot : first_slot;
  const uint32_t owned_full = full_s + owned_slot * 8;

  // Every producer warp contributes one arrival to each slot.  Its owned
  // half carries the byte expectation for that warp's row-copy transactions.
  if (lane == 0) {
    if (warp_rows > 0) {
      mbar_arrive_expect_tx(owned_full, warp_rows * kKvlora * int(sizeof(__nv_bfloat16)));
    } else {
      mbar_arrive(owned_full);
    }
    mbar_arrive(full_s + other_slot * 8);
  }
  __syncwarp();

  if (lane < warp_rows) {
    const int row = pwarp * 16 + lane;
    const int slot = slot_base + row;
    int safe_id;
    if (producer_stages_ids) {
      safe_id = load_shared_i32(staged_ids_s + row * uint32_t(sizeof(int32_t)));
      if (safe_id < 0) safe_id = 0;
    } else {
      safe_id = 0;
      if (slot < length) {
        const int candidate =
            canonical_dense ? slot : (ll_mode ? g_ll_ids[row] : ids[query * kSelected + slot]);
        if (candidate >= 0 && candidate < args.cache_rows && candidate <= g_mega_cur_pos + query) {
          safe_id = candidate;
        }
      }
    }
    const int local_row = row & (kK3OptHalfRows - 1);
    bulk_copy_g2s_l2(ring_s + owned_slot * smem::kRingSlotBytes +
                         local_row * kK3OptRingRowStride * sizeof(__nv_bfloat16),
                     args.kv_cache + static_cast<int64_t>(safe_id) * kKvlora,
                     kKvlora * sizeof(__nv_bfloat16), owned_full, l2_policy_evict_first());
    // Warm this split's pe_cache rows (one 128 B line each) for the
    // consumer's cp.async PE gather, which a K2-working consumer can only
    // issue after its q-chain role ends (~2 us from now): by then the row is
    // an L2 hit instead of a cold HBM round trip.
    asm volatile(
        "prefetch.global.L2 [%0];" ::"l"(args.pe_cache + static_cast<int64_t>(safe_id) * kRope)
        : "memory");
  }

  // The query-NOPE half of K3 depends on the 80 W_UK publications (C4), but
  // the consumer's PE score half does not.  Move the C4 acquire and the ten
  // 1-KiB copies to producer lane 0 so both are hidden under the PE MMA and
  // the selected-KV ring flight.  Each head lands in the padded K3 query row;
  // the PE suffix at +kKvlora is disjoint and may be read concurrently.
  const bool has_work = slot_base < length;
  if (has_work && args.ids_ready && args.mode == kModePersistent && ptid == 0 && !kK3QueryRecords) {
    const uint32_t* counters = reinterpret_cast<const uint32_t*>(local + arena::kCountersPadded);
    const uint32_t c4_target = ArrivalCounter::target(kArriveC4, g_mega_epoch);
    const uint32_t c4pe_target = ArrivalCounter::target(kArriveC4Pe, g_mega_epoch);
    // post_qnope lands in the same scratch prefix used by both fused-K2
    // cohorts.  C4 covers W_UK and C4Pe covers the PE cohort; require both
    // before reusing that storage.  C4Pe normally completes first.
    while (!word_reached(counters + cnt_word(kCntC4), c4_target) ||
           !word_reached(counters + cnt_word(kCntC4Pe), c4pe_target)) {
      spin_backoff();
    }
    fence_acquire_gpu();

    constexpr uint32_t kHeadBytes = kKvlora * uint32_t(sizeof(__nv_bfloat16));
    constexpr uint32_t kAllHeadBytes = kHeads * kHeadBytes;
    constexpr uint32_t kQueryStrideBytes = (kKvlora + kRope + 16) * uint32_t(sizeof(__nv_bfloat16));
    const uint32_t query_s = ring_s - smem::kOpScratchBytes;
    const uint32_t qbar_s = smem_u32(&g_k3_qnope_bar);
    mbar_arrive_expect_tx(qbar_s, kAllHeadBytes);
#pragma unroll
    for (int head = 0; head < kHeads; ++head) {
      // Every query row is consumed by all 32 split CTAs.  Preserve it in L2
      // instead of applying the one-shot page-streaming evict-first policy.
      bulk_copy_g2s(query_s + head * kQueryStrideBytes,
                    args.post_qnope + (query * kHeads + head) * kKvlora, kHeadBytes, qbar_s);
    }
  }
  item += 2;
}

// Fused K2 consumes one 16-row half from each 0x8080-byte q_b_proj page.  Preserve
// the logical two-item schedule, but stage only that contiguous 0x4000-byte
// half followed by the eight FP32 K-slice scales that the consumer reads.
static __device__ void producer_stream_q_b_proj_half_compact(const MegaArgs& args, int block,
                                                             int row_half, int head,
                                                             uint32_t ring_s, uint32_t full_s,
                                                             uint32_t empty_s, uint32_t& item) {
  if (args.rank == 7 && head >= 4) return;

  const int ptid = static_cast<int>(threadIdx.x) - kOpThreads;
  const int pwarp = ptid >> 5;
  const int plane = ptid & 31;
#pragma unroll
  for (int page = 0; page < 2; ++page) {
    constexpr uint32_t kHalfBytes = 0x4000u;
    constexpr uint32_t kScaleBytes = 0x20u;
    const uint8_t* base = args.q_b_proj + static_cast<int64_t>(2 * block + page) * kPageBytes8080;
    const uint32_t slot = item & (smem::kRingSlots - 1);
    const uint32_t full = full_s + slot * 8;
    const uint32_t dst = ring_s + slot * smem::kRingSlotBytes;
    if (plane == 0) {
      mbar_wait_parity(empty_s + slot * 8, ((item >> 2) & 1u) ^ 1u);
    }
    if (pwarp == 0) {
      if (plane == 0) {
        mbar_arrive_expect_tx(full, kHalfBytes + kScaleBytes);
        bulk_copy_g2s_l2(dst, base + row_half * kHalfBytes, kHalfBytes, full,
                         l2_policy_evict_first());
        bulk_copy_g2s_l2(dst + kHalfBytes, base + 0x8000, kScaleBytes, full,
                         l2_policy_evict_first());
      }
    } else if (plane == 0) {
      mbar_arrive(full);
    }
    ++item;
  }
}

static __device__ void producer_stream_k2_fused_compact(const MegaArgs& args, int cta,
                                                        uint32_t ring_s, uint32_t full_s,
                                                        uint32_t empty_s, uint32_t& item) {
  if (cta >= kK2FusedCtas) return;
  const int head = cta / kK2FusedCohort;
  const int cohort = cta - head * kK2FusedCohort;
  const bool active = args.rank != 7 || head < 4;

  const int nope_block = head * (kNope / 32) + (cohort >> 1);
  producer_stream_q_b_proj_half_compact(args, nope_block, cohort & 1, head, ring_s, full_s, empty_s,
                                        item);

  if (cohort < kK2FusedUkPerHead) {
    if (!active) return;
    const int ptid = static_cast<int>(threadIdx.x) - kOpThreads;
    const int pwarp = ptid >> 5;
    const int plane = ptid & 31;
    const uint8_t* ptr =
        args.w_uk + static_cast<int64_t>(head * kK2FusedUkPerHead + cohort) * kUkPageBytes;
    const uint32_t slot = item & (smem::kRingSlots - 1);
    const uint32_t full = full_s + slot * 8;
    mbar_wait_parity(empty_s + slot * 8, ((item >> 2) & 1u) ^ 1u);
    if (pwarp == 0) {
      if (plane == 0) {
        mbar_arrive_expect_tx(full, kUkPageBytes);
        bulk_copy_g2s_l2(ring_s + slot * smem::kRingSlotBytes, ptr, kUkPageBytes, full,
                         l2_policy_evict_first());
      }
    } else if (plane == 0) {
      mbar_arrive(full);
    }
    ++item;
  } else {
    const int pe_tile = cohort - kK2FusedUkPerHead;
    const int pe_block = kQbProjNopeRows / 32 + head * (kRope / 32) + (pe_tile >> 1);
    producer_stream_q_b_proj_half_compact(args, pe_block, pe_tile & 1, head, ring_s, full_s,
                                          empty_s, item);
  }
}

static __device__ void producer_stream_op(int op, const MegaArgs& args, int cta, uint32_t ring_s,
                                          uint32_t full_s, uint32_t empty_s, uint32_t& item,
                                          int first_index = 0) {
  if (op == kOpK3) {
    producer_stream_k3(args, cta, ring_s, full_s, empty_s, item);
    return;
  }
  if (op == kOpK2Fused) {
    producer_stream_k2_fused_compact(args, cta, ring_s, full_s, empty_s, item);
    return;
  }
  const int ptid = static_cast<int>(threadIdx.x) - kOpThreads;  // 0..127
  const int pwarp = ptid >> 5;
  const int plane = ptid & 31;
  const uint8_t* ptr = nullptr;
  int bytes = 0;
  for (int index = first_index; op_page(op, args, cta, index, &ptr, &bytes); ++index) {
    if (bytes == 0) continue;  // rank7 pad page: not an item
    const uint32_t slot = item & (smem::kRingSlots - 1);
    const uint32_t full = full_s + slot * 8;
    mbar_wait_parity(empty_s + slot * 8,
                     ((item >> 2) & 1u) ^ 1u);  // slot free (first round: immediate)
    if (pwarp == 0) {
      if (plane == 0) {
        mbar_arrive_expect_tx(full, static_cast<uint32_t>(bytes));
        bulk_copy_g2s_l2(ring_s + slot * smem::kRingSlotBytes, ptr, static_cast<uint32_t>(bytes),
                         full, l2_policy_evict_first());
      }
    } else if (plane == 0) {
      mbar_arrive(full);
    }
    ++item;
  }
}

// Bulk L2 prefetch of one weight page (fire-and-forget; no mbarrier).
__device__ __forceinline__ void prefetch_l2_bulk(const void* src_global, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src_global), "r"(bytes)
               : "memory");
}

// The final six tail CTAs' producer warpgroups are idle after the
// prologue.  Warm the K4/K5 weight stream into L2 during the DRAM trough
// between the q-chain and o_proj: one thread polls the PE half of the q-chain
// boundary (by then the qkv_a_proj and q_b_proj streams are finished, so this stream does
// not contend with them; the flush write-back is long done), a producer-group
// barrier releases the rest, and threads 0..719 across the six CTAs issue one
// bulk L2 prefetch per W_UV / o_proj page.  The K4 ring TMAs (issued ~1 us
// later) and the late K5 pages (issued during the o_proj GEMV) then hit L2
// instead of HBM.  Default eviction; the weights are constant inputs so no ordering is
// required.
static __device__ void producer_tail_prefetch_weights(const MegaArgs& args, int cta) {
  if (args.mode != kModePersistent) return;
  constexpr int kPrefetchCtas = kTailCtas;
  const int prefetch_cta = cta - (kGrid - kPrefetchCtas);
  if (prefetch_cta < 0) return;
  const int ptid = static_cast<int>(threadIdx.x) - kOpThreads;  // 0..127
  if (ptid == 0) {
    const uint32_t* counters =
        reinterpret_cast<const uint32_t*>(args.peers[args.rank] + arena::kCountersPadded);
    const uint32_t target = ArrivalCounter::target(kArriveC0, g_mega_epoch);
    while (!word_reached(counters + cnt_word(kCntC0), target)) {
    }
  }
  asm volatile("bar.sync 6, %0;" ::"n"(kProducerThreads) : "memory");
  const int slot = ptid * kPrefetchCtas +
                   prefetch_cta;  // interleaved: consecutive pages land on different tail CTAs
  if (slot < kK4Ctas) {
    prefetch_l2_bulk(args.w_uv + static_cast<int64_t>(slot) * kUvPageBytes, kUvPageBytes);
  } else if (slot < kK4Ctas + kK5Ctas * 5) {
    const int page = slot - kK4Ctas;
    prefetch_l2_bulk(args.o_proj + static_cast<int64_t>(page) * kOProjPageBytes, kOProjPageBytes);
  }
}

// Launch scalars (epoch counter RMW, cur_pos, LL receive buffer / tag / epoch) of this CTA,
// published to the shared launch-scalar slots.  Thread 0 of the CTA's last producer warp runs it
// (rank 0: thread 0 at entry); the consumers read the slots only after named barrier 12
// (producer_publish_launch_scalars), so the RMW and the two dependent loads leave the consumer
// critical path.
__device__ __forceinline__ void mega_load_launch_scalars(const MegaArgs& args_in, int cta) {
  int32_t cur_pos_v = args_in.cur_pos;
  uint32_t epoch_v = args_in.epoch;
  const int32_t* cp_ptr = args_in.cur_pos_dev;
  uint32_t* ctr = args_in.epoch_ctr;
  const uint8_t* ll_recv_v = args_in.ll_recv;
  const uint32_t ll_tag_v = args_in.ll_tag;
  uint32_t ctr_old = 0;
  int32_t cp_dev = 0;
  if (ctr != nullptr) ctr_old = ctr[cta];
  if (cp_ptr != nullptr) cp_dev = *cp_ptr;
  if (ctr != nullptr) {
    epoch_v = ctr_old + 1u;
    ctr[cta] = epoch_v;
  } else if (args_in.epoch_dev != nullptr) {
    epoch_v =
        atomicAdd(const_cast<uint32_t*>(args_in.epoch_dev), 1u) / static_cast<uint32_t>(kGrid) + 1u;
  }
  if (cp_ptr != nullptr) cur_pos_v = cp_dev;
  g_mega_epoch = epoch_v;
  g_mega_cur_pos = cur_pos_v;
  g_ll_recv = ll_recv_v;
  g_ll_tag = ll_tag_v;
}

// The last producer warp loads the launch scalars and arrives on barrier 12 (consumers + this warp)
// and 13 (producers); the other producer warps wait on 13 before continuing.
__device__ __forceinline__ void producer_publish_launch_scalars(const MegaArgs& args, int cta) {
  const int pwarp = (static_cast<int>(threadIdx.x) - kOpThreads) >> 5;
  if (pwarp == kProducerWarps - 1) {
    if ((threadIdx.x & 31) == 0) mega_load_launch_scalars(args, cta);
    __syncwarp(0xffffffffu);
    asm volatile("bar.arrive 12, %0;" ::"n"(kOpThreads + 32) : "memory");
    asm volatile("bar.arrive 13, %0;" ::"n"(kProducerThreads) : "memory");
  } else {
    asm volatile("bar.sync 13, %0;" ::"n"(kProducerThreads) : "memory");
  }
}

// Per-CTA page schedule of the persistent kernel (consumers run K1 -> K2 fused -> K3 ->
// K4 -> K5).  op_page returns no pages for CTAs outside an op's page set (tail CTAs
// 128..147 stream nothing); the K3 gather has no pages for CTAs >= 128 either.
static __device__ void producer_main(const MegaArgs& args, int cta, uint32_t ring_s,
                                     uint32_t full_s, uint32_t empty_s, bool k1_page0_preissued) {
  if (args.mode == kModeNullOrchestration) {  // orchestration floor: no streaming
    producer_publish_launch_scalars(args, cta);
    return;
  }
  if (cta >= kGroup1Ctas) {
    // Tail CTAs own no ring items; every tail CTA warms part of the pages.
    producer_publish_launch_scalars(args, cta);
    producer_tail_prefetch_weights(args, cta);
    return;
  }
  uint32_t item = k1_page0_preissued ? 1u : 0u;
  producer_stream_op(kOpK1, args, cta, ring_s, full_s, empty_s, item, k1_page0_preissued ? 1 : 0);
  producer_publish_launch_scalars(args, cta);  // K1 weight stream already in flight
  if (cta >= kQkvAProjCtaBlocks && cta < kK2FusedCtas && args.mode == kModePersistent) {
    if (static_cast<int>(threadIdx.x) == kOpThreads) {
      const uint32_t* counters =
          reinterpret_cast<const uint32_t*>(args.peers[args.rank] + arena::kCountersPadded);
      const uint32_t c0_first = ArrivalCounter::target(kArriveC0, g_mega_epoch - 1u) + 1u;
      while (!word_reached(counters + cnt_word(kCntC0), c0_first)) {
      }
    }
    asm volatile("bar.sync 6, %0;" ::"n"(kProducerThreads) : "memory");
  }
  producer_stream_op(kOpK2Fused, args, cta, ring_s, full_s, empty_s, item);
  producer_stream_op(kOpK3, args, cta, ring_s, full_s, empty_s, item);
  producer_stream_op(kOpK4, args, cta, ring_s, full_s, empty_s, item);
  producer_stream_op(kOpK5, args, cta, ring_s, full_s, empty_s, item);
}

// Consumer side: walk the op's pages, wait for each and free it at once.
static __device__ void consumer_drain_op(int op, const MegaArgs& args, OpCtx& ctx) {
  const uint8_t* ptr = nullptr;
  int bytes = 0;
  for (int index = 0; op_page(op, args, ctx.cta, index, &ptr, &bytes); ++index) {
    if (bytes == 0) continue;
    ctx.acquire_page();
    ctx.release_page();
  }
}

// null_stream: consumers mirror producer_main so every page is waited for and
// released (the stream floor then measures completed bulk copies; the K3 gather
// is not issued in this mode).
static __device__ void consumer_drain_all(const MegaArgs& args, OpCtx& ctx) {
  consumer_drain_op(kOpK1, args, ctx);
  consumer_drain_op(kOpK2Fused, args, ctx);
  consumer_drain_op(kOpK4, args, ctx);
  consumer_drain_op(kOpK5, args, ctx);
}

}  // namespace flashinfer::glm5_tp8::mla_q4
// smem-operand W8A16 GEMV building blocks.

// Packed tile layout: every 16-row x 16-K tile is 256 B in exactly the lane order
// of the m16n8k16 BF16 MMA A fragment (lane l holds bytes [8l, 8l+8)),
// followed by fp32 block scales.  Per
// page, warp w owns one K chunk of all rows:
// The 8 warp partials are stored to smem as [unit][token][row] and folded in a
// fixed order (warp 0..7 / chunk 0..2) by one thread per (token, row), then
// rounded to bf16.  This fp32 association (K split across warps, per-warp

namespace flashinfer::glm5_tp8::mla_q4 {

// ---- shared-memory loads by 32-bit shared-window address ------------------------------
// asm volatile keeps program order relative to the (volatile) mbarrier arrive of
// release_page: every shared load of a page is issued before the slot is freed.
__device__ __forceinline__ RawFp8x8 lds_fp8x8(uint32_t address) {
  RawFp8x8 raw;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(raw.lo), "=r"(raw.hi) : "r"(address));
  return raw;
}

__device__ __forceinline__ uint32_t lds_u32(uint32_t address) {
  uint32_t value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
  return value;
}

__device__ __forceinline__ float lds_f32(uint32_t address) {
  float value;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"(address));
  return value;
}

// B fragment of one k16 step: token (lane >> 2) columns k + 2*(lane & 3) + {0, 1}
// and + 8, zero for lanes 16..31 (tokens 4..7).
__device__ __forceinline__ void lds_act_fragment(uint32_t act_s, uint32_t row_stride_bytes, int k,
                                                 uint32_t (&b)[2]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  b[0] = 0u;
  b[1] = 0u;
  if (lane < 16) {
    const uint32_t address = act_s + static_cast<uint32_t>(lane >> 2) * row_stride_bytes +
                             static_cast<uint32_t>(k + 2 * (lane & 3)) * 2u;
    b[0] = lds_u32(address);
    b[1] = lds_u32(address + 16u);
  }
}

// One warp's share of one page: RowTiles 16-row tiles (stride row_tile_stride
// bytes) x KSteps k16 tiles (stride k_step_stride bytes) starting at a_base
// (= slot + chunk offset + lane*8), against activation columns k0 + 16 t.
// All loads are issued first (independent shared loads in flight), then converted and
// fed to the tensor cores; acc[r] accumulates row tile r over the KSteps.
template <int RowTiles, int KSteps>
__device__ __forceinline__ void gemv_page_steps(uint32_t a_base, uint32_t row_tile_stride,
                                                uint32_t k_step_stride, uint32_t act_s,
                                                uint32_t act_stride_bytes, int k0,
                                                HmmaAcc (&acc)[RowTiles]) {
  RawFp8x8 raw[RowTiles][KSteps];
  uint32_t b[KSteps][2];
#pragma unroll
  for (int t = 0; t < KSteps; ++t) {
#pragma unroll
    for (int r = 0; r < RowTiles; ++r) {
      raw[r][t] = lds_fp8x8(a_base + static_cast<uint32_t>(r) * row_tile_stride +
                            static_cast<uint32_t>(t) * k_step_stride);
    }
    lds_act_fragment(act_s, act_stride_bytes, k0 + 16 * t, b[t]);
  }
#pragma unroll
  for (int t = 0; t < KSteps; ++t) {
#pragma unroll
    for (int r = 0; r < RowTiles; ++r) {
      uint32_t a[4];
      fp8_to_bf16_fragment(raw[r][t], a);
      hmma(acc[r], a, b[t]);
    }
  }
}

// K1 always computes two row tiles against the same activation fragment.
// Decode both FP8 A fragments before the adjacent MMAs so the B registers can
// remain collector-resident across the pair.
template <int KSteps>
__device__ __forceinline__ float gemv_page_steps_b_reuse_release(
    OpCtx& ctx, uint32_t a_base, uint32_t row_tile_stride, uint32_t k_step_stride, uint32_t act_s,
    uint32_t act_stride_bytes, int k0, uint32_t scale_address, HmmaAcc (&acc)[2]) {
  RawFp8x8 raw[2][KSteps];
  uint32_t b[KSteps][2];
#pragma unroll
  for (int t = 0; t < KSteps; ++t) {
    raw[0][t] = lds_fp8x8(a_base + static_cast<uint32_t>(t) * k_step_stride);
    raw[1][t] = lds_fp8x8(a_base + row_tile_stride + static_cast<uint32_t>(t) * k_step_stride);
    lds_act_fragment(act_s, act_stride_bytes, k0 + 16 * t, b[t]);
  }
  // The ring page is dead once its raw fragments and scale are resident in
  // registers. Free the slot before FP8 decode/MMA so the producer can land
  // the next page while this warp consumes the current one.
  const float scale = lds_f32(scale_address);
  ctx.release_page();
#pragma unroll
  for (int t = 0; t < KSteps; ++t) {
    uint32_t a0[4];
    uint32_t a1[4];
    fp8_to_bf16_fragment(raw[0][t], a0);
    fp8_to_bf16_fragment(raw[1][t], a1);
    hmma(acc[0], a0, b[t]);
    hmma(acc[1], a1, b[t]);
  }
  return scale;
}

// Cross-warp partial buffer layout: partial[unit][token 0..3][rows_per_cta] fp32.
// An MMA accumulator holds rows (lane>>2) and (lane>>2)+8 of the 16-row tile for
// token columns 2*(lane&3) and +1; lanes with (lane&3) >= 2 hold the zero-padded
// tokens 4..7 and store nothing.
__device__ __forceinline__ void store_partial_tile(float* unit_partial, int rows_per_cta,
                                                   int row_tile, const float (&v)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int tp = lane & 3;
  if (tp < 2) {
    const int row = row_tile * 16 + g;
    unit_partial[(2 * tp) * rows_per_cta + row] = v[0];
    unit_partial[(2 * tp + 1) * rows_per_cta + row] = v[1];
    unit_partial[(2 * tp) * rows_per_cta + row + 8] = v[2];
    unit_partial[(2 * tp + 1) * rows_per_cta + row + 8] = v[3];
  }
}

// Fold two adjacent rows with one SM100 packed add while retaining each row's
// scalar warp-0..7 association.  Even-row callers make every u64 shared load aligned.
template <int Units, int RowsPerCta>
__device__ __forceinline__ F32Pair reduce_partial_pair(const float* partial, int token,
                                                       int even_row) {
  static_assert((RowsPerCta & 1) == 0, "paired reduction needs even rows");
  uint64_t value[Units];
  const uint32_t base = smem_u32(partial + token * RowsPerCta + even_row);
#pragma unroll
  for (int u = 0; u < Units; ++u) {
    const uint32_t address = base + u * kQlen * RowsPerCta * static_cast<int>(sizeof(float));
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(value[u]) : "r"(address));
  }
  F32Pair sum{};
#pragma unroll
  for (int u = 0; u < Units; ++u) {
    asm volatile("add.rn.f32x2 %0, %0, %1;" : "+l"(sum.bits) : "l"(value[u]));
  }
  return sum;
}

// Stage the four token rows of a bf16 activation into smem with a padded row stride, for tiles of
// <= 256 16 B chunks (one cp.async per thread, one commit group per thread): the caller overlaps
// the copy with its ring-page acquire, then waits (wait_activation_rows_async) and publishes the
// tile with the consumer barrier.
__device__ __forceinline__ void stage_activation_rows_async(__nv_bfloat16* dst, int dst_stride,
                                                            const __nv_bfloat16* src,
                                                            int src_stride, int cols) {
  const int chunks_per_row = cols / 8;
  const int i = static_cast<int>(threadIdx.x);
  if (i < kQlen * chunks_per_row) {
    const int row = i / chunks_per_row;
    const int c = i - row * chunks_per_row;
    const uint32_t dst_s = smem_u32(dst + row * dst_stride + c * 8);
    const __nv_bfloat16* src_g = src + row * src_stride + c * 8;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst_s), "l"(src_g) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_activation_rows_async() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// o_proj warp w consumes exactly five disjoint 64-column slices of post_uv (page p: columns
// [512 p + 64 w, +64) of all four tokens).  Each lane copies one 16 B vector per page, so the
// staging has no cross-warp dependency and a warp fence replaces the CTA-wide publication barrier
// of the generic helper.
__device__ __forceinline__ void stage_o_proj_activation_kowned(__nv_bfloat16* dst, int dst_stride,
                                                               const __nv_bfloat16* src,
                                                               int src_stride, int warp, int lane) {
  const int token = lane >> 3;
  const int chunk = lane & 7;
#pragma unroll
  for (int page = 0; page < 5; ++page) {
    const int col = page * 512 + warp * 64 + chunk * 8;
    *reinterpret_cast<uint4*>(dst + token * dst_stride + col) =
        *reinterpret_cast<const uint4*>(src + token * src_stride + col);
  }
  __syncwarp(0xffffffffu);
}
__device__ __forceinline__ void stage_o_proj_activation_records(__nv_bfloat16* dst, int dst_stride,
                                                                const uint8_t* records, int warp,
                                                                int lane, uint32_t tag) {
  const int token = lane >> 3;
  const int chunk = lane & 7;
  uint64_t lo[10], hi[10];
#pragma unroll
  for (int page = 0; page < 5; ++page) {
    const int element = token * (kHeads * kVdim) + page * 512 + warp * 64 + chunk * 8;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint8_t* rec = records + size_t(element / 4 + h) * 16;
      asm volatile(
          "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
          : "=l"(lo[2 * page + h]), "=l"(hi[2 * page + h])
          : "l"(rec)
          : "memory");
    }
  }
  for (;;) {
    bool done = true;
#pragma unroll
    for (int page = 0; page < 5; ++page) {
      const int element = token * (kHeads * kVdim) + page * 512 + warp * 64 + chunk * 8;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int i = 2 * page + h;
        if (static_cast<uint32_t>(lo[i] >> 32) != tag ||
            static_cast<uint32_t>(hi[i] >> 32) != tag) {
          done = false;
          const uint8_t* rec = records + size_t(element / 4 + h) * 16;
          asm volatile(
              "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, "
              "r;\n\t}"
              : "=l"(lo[i]), "=l"(hi[i])
              : "l"(rec)
              : "memory");
        }
      }
    }
    if (done) break;
  }
#pragma unroll
  for (int page = 0; page < 5; ++page) {
    const int col = page * 512 + warp * 64 + chunk * 8;
    *reinterpret_cast<uint4*>(dst + token * dst_stride + col) = make_uint4(
        static_cast<uint32_t>(lo[2 * page]), static_cast<uint32_t>(hi[2 * page]),
        static_cast<uint32_t>(lo[2 * page + 1]), static_cast<uint32_t>(hi[2 * page + 1]));
  }
  __syncwarp(0xffffffffu);
}

__device__ __forceinline__ void fma_scale_into(float (&run)[4], float scale, const HmmaAcc& acc) {
#pragma unroll
  for (int i = 0; i < 4; ++i) run[i] = fmaf(scale, acc.value[i], run[i]);
}

}  // namespace flashinfer::glm5_tp8::mla_q4

// K0 (rank 0 publication of the selected ids, bulk-copy wire): CTA d of the K0 set copies the
// 32 KiB ids + four lengths into attention rank d+1's parity-buffered inbox and releases its
// kFlagPub; CTA 0 also writes the dummy outputs and arrives once on kCntK0Copy (bookkeeping only,
// nobody waits on it).  The other CTAs of the K0 set retire at once.  CTA 0 then polls the seven
// early acks in parallel and never waits for the dones.

// Single-rank loopback (MegaArgs::single_rank; launch_mla_q4 leaves it 0): the
// same body runs as a publish-only launch on the attention rank's own GPU,
// stream-ordered before the attention launch.  It copies ids/lengths into the
// local arena (which is also peers[0]) and releases kFlagPub = epoch; it must
// NOT zero the outputs (they are the attention rank's real outputs) and must NOT
// wait for kFlagAck/kFlagDone: those words are written by K3/K5 of the same
// rank, which run AFTER this kernel in stream order -> waiting here would
// deadlock the stream.

namespace flashinfer::glm5_tp8::mla_q4 {

// Rank-0 publication is a regular contiguous payload once the four lengths are
// appended to the 32 KiB id matrix.  Stage it through this CTA's otherwise-idle
// operation scratch, then let the bulk engine perform the peer write instead
// of issuing one system-scope 128-bit store from every lane for every vector.
__device__ __forceinline__ void bulk_copy_s2g(uint8_t* dst_global, uint32_t src_smem,
                                              uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global),
               "r"(src_smem), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void bulk_copy_s2g_wait() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

static __device__ void op_k0_rank0_publish(const MegaArgs& args, OpCtx& ctx) {
  // Publisher CTA d pushes this epoch's complete payload into peer d+1.  The
  // seven peer writes therefore proceed independently instead of making every
  // attention rank pull the same rank-0 buffer over NVLink.
  const int dst = ctx.cta + 1;
  if (dst >= kWorld) return;
  if (args.single_rank && ctx.cta != 0) return;
  uint8_t* peer = args.peers[args.single_rank ? args.rank : dst];
  uint8_t* inbox = peer + arena::kIdsRx + size_t(g_mega_epoch & 1u) * arena::kIdsRxParityBytes;
  constexpr uint32_t kIdsPayloadBytes = kQlen * kSelected * uint32_t(sizeof(int32_t));
  constexpr uint32_t kLengthsPayloadBytes = kQlen * uint32_t(sizeof(int32_t));
  constexpr uint32_t kPublishPayloadBytes = kIdsPayloadBytes + kLengthsPayloadBytes;
  static_assert(kIdsPayloadBytes == arena::kIdsRxLengths && (kPublishPayloadBytes & 15u) == 0 &&
                    kPublishPayloadBytes <= smem::kOpScratchBytes,
                "rank0 ids + lengths bulk payload");
  if (ctx.tid == 0 && ctx.cta == 0 && !args.single_rank) {
    __nv_bfloat16 zero = to_bf16(0.0f);
    args.post_q[0] = zero;
    args.post_kv[0] = zero;
    args.post_qnd[0] = zero;
    args.post_qpe[0] = zero;
    args.post_qnope[0] = zero;
    args.post_o[0] = zero;
    args.post_uv[0] = zero;
    args.endpoint[0] = zero;
  }

  if (ctx.tid == 0) {
    const uint32_t payload_s = smem_u32(ctx.smem);
    mbar_init(ctx.full_s, 1);
    mbar_fence_init();
    mbar_arrive_expect_tx(ctx.full_s, kPublishPayloadBytes);
    bulk_copy_g2s(payload_s, args.selected_ids_src, kIdsPayloadBytes, ctx.full_s);
    bulk_copy_g2s(payload_s + kIdsPayloadBytes, args.selected_lengths_src, kLengthsPayloadBytes,
                  ctx.full_s);
    mbar_wait_parity(ctx.full_s, 0);

    // Both operations use the async proxy.  The explicit proxy fence makes the
    // completed shared writes visible before the bulk engine reads the payload.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bulk_copy_s2g(inbox, payload_s, kPublishPayloadBytes);
    bulk_copy_s2g_wait();

    // Observing cp.async.bulk.wait_group completion includes the ISA-defined
    // async-to-generic proxy fence.  The following system-scope release store
    // therefore orders the complete P2P payload without an extra membar.sys.
    if (ctx.cta == 0) red_release_gpu_add(ctx.counters + cnt_word(kCntK0Copy), 1u);
    store_release_sys(wsp<uint32_t>(peer, arena::kFlagPub), g_mega_epoch);
  }
  if (ctx.cta != 0) return;
  // rank0 paces on the seven EARLY acknowledgements (each attention rank releases kFlagAck right
  // after copying this epoch's ids into its own arena -- the last of its eight ids tail CTAs), not
  // on the endpoint completion tokens: the ack is exactly the buffer-reuse proof rank0 needs before
  // it may overwrite its single ids buffer next epoch, and it lands ~25 us earlier than kFlagDone.
  // Attention ranks can lag each other by at most one epoch (rank0 publishes e+1 only after every
  // rank acked e), which the epoch-parity double-buffered K5 inboxes tolerate; nobody publishes or
  // waits for kFlagDone. The seven waits are independent, so lanes 1..7 of warp 0 poll
  // them concurrently. null_stream: attention ranks never ack; loopback/single-GPU: no peers;
  // single_rank: the acks come from K3 of the attention launch that follows this one (waiting here
  // would deadlock the stream).
  if (!args.skip_done_wait && args.mode != kModeNullStream && !args.single_rank) {
    if (ctx.tid >= 1 && ctx.tid < kWorld) {
      wait_epoch(wsp<uint32_t>(args.peers[ctx.tid], arena::kFlagAck), g_mega_epoch);
    }
  }
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

template <int CopyCtas>
static __device__ __forceinline__ void consume_selected_ids(const MegaArgs& args, OpCtx& ctx,
                                                            int copy_cta = 0) {
  static_assert(CopyCtas >= 1, "selected-id consume needs at least one CTA");
  // The payload is already in this rank's local parity buffer.  A single CTA
  // acquires the local publish word and immediately returns the ack used to
  // pace rank 0's parity-buffer reuse; no data copy or copy-counter remains.
  if (args.ids_direct || copy_cta != 0) return;
  if (ctx.tid == 0) {
    wait_epoch(wsp<uint32_t>(ctx.arena, arena::kFlagPub), g_mega_epoch);
    store_release_sys(wsp<uint32_t>(ctx.arena, arena::kFlagAck), g_mega_epoch);
  }
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

// The four KV-normalization tails gate the selected-KV gather.  Issue every
// source/gamma load before consuming any of them, retain the values across the
// virtual-384 reduction, and write each output element exactly once.
// The sum uses the virtual-384 ownership of block_sum_v384: tids 0..127 own
// {tid, tid+384} in sum0 and {tid+256} in sum1, while tids 128..255 own {tid}.
__device__ __forceinline__ uint16_t ldg_u16_kv(const void* address) {
  uint16_t value;
  asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ float ldg_f32_kv(const float* address) {
  float value;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}
struct TailPreloads {
  float g0, g256, g384;
  float cosine, sine;
};

__device__ __forceinline__ void normalize_kv_row_retained(const OpCtx& ctx,
                                                          const __nv_bfloat16* input,
                                                          const TailPreloads& pre,
                                                          __nv_bfloat16* output, float epsilon,
                                                          float* scratch) {
  const int tid = ctx.tid;

  // These independent loads are deliberately grouped ahead of their first
  // use so the SM carries one memory round trip instead of a sum/reload/gamma
  // dependency chain.
  const uint16_t raw0 = ldg_u16_kv(input + tid);
  uint16_t raw384 = 0;
  uint16_t raw256 = 0;
  if (tid < 128) {
    raw384 = ldg_u16_kv(input + tid + 384);
    raw256 = ldg_u16_kv(input + tid + 256);
  }
  const float gamma0 = pre.g0;
  float gamma384 = 0.0f;
  float gamma256 = 0.0f;
  if (tid < 128) {
    gamma384 = pre.g384;
    gamma256 = pre.g256;
  }

  const float value0 = bf16_bits_to_float(raw0);
  float sum0 = fmaf(value0, value0, 0.0f);
  float value384 = 0.0f;
  float value256 = 0.0f;
  float sum1 = 0.0f;
  if (tid < 128) {
    value384 = bf16_bits_to_float(raw384);
    value256 = bf16_bits_to_float(raw256);
    sum0 = fmaf(value384, value384, sum0);
    sum1 = fmaf(value256, value256, 0.0f);
  }

  const float sum = block_sum_v384(ctx, sum0, sum1, scratch);
  const float inverse = rsqrtf(sum * (4.0f / 2048.0f) + epsilon);

  output[tid] = to_bf16((value0 * inverse) * gamma0);
  if (tid < 128) {
    output[tid + 384] = to_bf16((value384 * inverse) * gamma384);
    output[tid + 256] = to_bf16((value256 * inverse) * gamma256);
  }
}

// Tail-CTA work (kv-norm rows, pe RoPE rows, ids copy).  `ctx.cta` = tail index.
__device__ __forceinline__ TailPreloads tail_preload(const MegaArgs& args, int t, int tid) {
  TailPreloads pre{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (args.mode == kModeNullOrchestration) return pre;
  if (t < kKvNormCtas) {
    pre.g0 = ldg_f32_kv(args.kv_gamma + tid);
    if (tid < 128) {
      pre.g384 = ldg_f32_kv(args.kv_gamma + tid + 384);
      pre.g256 = ldg_f32_kv(args.kv_gamma + tid + 256);
    }
  } else if (t < kTailRope + kQlen && tid < kRope / 2) {
    const int query = t - kTailRope;
    pre.cosine = args.rope[(size_t(g_mega_cur_pos) + query) * kRope + tid * 2];
    pre.sine = args.rope[(size_t(g_mega_cur_pos) + query) * kRope + tid * 2 + 1];
  }
  return pre;
}

static __device__ void op_k1_tails(const MegaArgs& args, OpCtx& ctx, const TailPreloads& pre) {
  const bool null_math = args.mode == kModeNullOrchestration;
  // Tail normalization does not run a GEMV; reuse its dead partial prefix.
  float* reduce_scratch = ctx.scratch<float>(smem::kGemmPartial);
  const __nv_bfloat16* pe_raw = wsp<const __nv_bfloat16>(ctx.arena, arena::kPeRaw);
  const int t = ctx.cta;
  if (t < kKvNormCtas) {
    // kv-norm tail: post_kv row t into kv_cache[cur_pos + t].
    if (!null_math) {
      normalize_kv_row_retained(ctx, args.post_kv + t * kKvlora, pre,
                                args.kv_cache + static_cast<int64_t>(g_mega_cur_pos + t) * kKvlora,
                                args.kv_eps, reduce_scratch);
    }
  } else if (t < kTailRope + kQlen) {
    // pe RoPE tail: pe_raw query t-4 into pe_cache[cur_pos + query].
    const int query = t - kTailRope;
    if (!null_math && ctx.tid < kRope / 2) {
      int pair = ctx.tid;
      int col = pair * 2;
      float real = bf16_to_float(pe_raw[query * kRope + col]);
      float imag = bf16_to_float(pe_raw[query * kRope + col + 1]);
      float cosine = pre.cosine;
      float sine = pre.sine;
      int64_t cache = static_cast<int64_t>(g_mega_cur_pos + query) * kRope + col;
      args.pe_cache[cache] = to_bf16(__fsub_rn(__fmul_rn(real, cosine), __fmul_rn(imag, sine)));
      args.pe_cache[cache + 1] = to_bf16(__fadd_rn(__fmul_rn(imag, cosine), __fmul_rn(real, sine)));
    }
  } else if (t >= kTailIds && t < kTailIds + kIdsCopyCtas) {
    if (args.dense_ids && args.ids_direct) {
      // Replace mode, dense selection: the constant table ids[q][i] = i was written once at bind;
      // lengths come from cur_pos.
    } else {
      // Selected-ID consume off the consumer critical path (also in null_orchestration: it is
      // orchestration): slice t - kTailIds of the eight coalesced 4 KiB ids slices.
      consume_selected_ids<kIdsCopyCtas>(args, ctx, t - kTailIds);
    }
  }
  // The first ID tail (CTA 136) performs both its publish-word acquire/ack and
  // the following C2 arrival in thread 0; no peer-thread writes need ordering.
  // KV-normalization and RoPE tails still require the consumer rendezvous.
  if (t != kTailIds) ctx.sync();
}

static __device__ void op_k1_tails(const MegaArgs& args, OpCtx& ctx) {
  op_k1_tails(args, ctx, tail_preload(args, ctx.cta, ctx.tid));
}

constexpr int kK1SegmentCells = kQlen * 32;  // (token, row) cells of one 32-row block
constexpr int kK1PublishRows = 32;
constexpr int kK1PublishOffset = 2 * kOpWarps * kK1SegmentCells * int(sizeof(float));
constexpr int kK1PublishBytes = kQlen * kK1PublishRows * int(sizeof(__nv_bfloat16));
constexpr int kK1QnormPublishOffset = kK1PublishOffset + kK1PublishBytes;
constexpr int kK1QnormPublishBytes = kQlen * int(sizeof(float));
static_assert(2 * kOpWarps * kK1SegmentCells * int(sizeof(float)) <= smem::kGemmPartialBytes,
              "two local K1 segments x 8 warps must fit the gemm partial scratch");
static_assert(kK1QnormPublishOffset + kK1QnormPublishBytes <= smem::kGemmPartialBytes,
              "K1 output and Q-norm publication slabs fit the partial scratch");
static_assert(kOpWarps * kQlen * int(sizeof(float)) <= smem::kGemmPartialBytes, "RMS scratch fits");

// Entry-hoisted K1 loads.  The twelve 16 B x vectors and six gamma
// float4s consumed by stage_xnorm_all_retained_f32x2 are issued at kernel
// entry -- before the ring-barrier init, the 384-thread prologue rendezvous
// and setmaxnreg -- so their HBM round trip (82 CTAs cold-missing the same
// 72 KiB) overlaps the prologue and the producer's page-0 TMA flight instead
// of serializing in front of the RMS reduction and the first MMA.  Same
// addresses, same 16 B widths, same arithmetic order downstream as loads
// issued inside the stage.
struct K1EntryLoads {
  uint4 x[3][kQlen];  // [band][token] at x + token*kHidden + 8*tid + 2048*band
  float4 g[3][2];     // [band][half]  at x_gamma + 8*tid + 2048*band + 4*half
};

__device__ __forceinline__ uint4 ldg_v4_b32(const void* address) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(address));
  return v;
}
__device__ __forceinline__ float4 ldg_v4_f32(const void* address) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(address));
  return v;
}

__device__ __forceinline__ void k1_entry_issue(const MegaArgs& args, int tid, K1EntryLoads& out) {
  const int col_base = 8 * tid;
#pragma unroll
  for (int band = 0; band < 3; ++band) {
    const int col = col_base + band * 2048;
#pragma unroll
    for (int token = 0; token < kQlen; ++token) {
      out.x[band][token] = ldg_v4_b32(args.x + token * kHidden + col);
    }
  }
#pragma unroll
  for (int band = 0; band < 3; ++band) {
    const int col = col_base + band * 2048;
    out.g[band][0] = ldg_v4_f32(args.x_gamma + col);
    out.g[band][1] = ldg_v4_f32(args.x_gamma + col + 4);
  }
}

__device__ __forceinline__ void stage_xnorm_all_retained_f32x2(const MegaArgs& args, OpCtx& ctx,
                                                               float* reduce_scratch,
                                                               __nv_bfloat16* xnorm,
                                                               const K1EntryLoads* pre = nullptr) {
  const int col0 = 8 * ctx.tid;
  const int col1 = col0 + 2048;
  const int col2 = col1 + 2048;

  const uint4 x00 = pre ? pre->x[0][0] : *reinterpret_cast<const uint4*>(args.x + col0);
  const uint4 x10 = pre ? pre->x[0][1] : *reinterpret_cast<const uint4*>(args.x + kHidden + col0);
  const uint4 x20 =
      pre ? pre->x[0][2] : *reinterpret_cast<const uint4*>(args.x + 2 * kHidden + col0);
  const uint4 x30 =
      pre ? pre->x[0][3] : *reinterpret_cast<const uint4*>(args.x + 3 * kHidden + col0);

  const uint4 x01 = pre ? pre->x[1][0] : *reinterpret_cast<const uint4*>(args.x + col1);
  const uint4 x11 = pre ? pre->x[1][1] : *reinterpret_cast<const uint4*>(args.x + kHidden + col1);
  const uint4 x21 =
      pre ? pre->x[1][2] : *reinterpret_cast<const uint4*>(args.x + 2 * kHidden + col1);
  const uint4 x31 =
      pre ? pre->x[1][3] : *reinterpret_cast<const uint4*>(args.x + 3 * kHidden + col1);

  const uint4 x02 = pre ? pre->x[2][0] : *reinterpret_cast<const uint4*>(args.x + col2);
  const uint4 x12 = pre ? pre->x[2][1] : *reinterpret_cast<const uint4*>(args.x + kHidden + col2);
  const uint4 x22 =
      pre ? pre->x[2][2] : *reinterpret_cast<const uint4*>(args.x + 2 * kHidden + col2);
  const uint4 x32 =
      pre ? pre->x[2][3] : *reinterpret_cast<const uint4*>(args.x + 3 * kHidden + col2);

  // Gamma vectors: prefetched at kernel entry (persistent path) or issued here,
  // before the RMS rendezvous, so the load latency overlaps the reduction and
  // barrier in both cases (the K2 q-norm uses this shape).
  float4 g00, g01, g10, g11, g20, g21;
  if (pre) {
    g00 = pre->g[0][0];
    g01 = pre->g[0][1];
    g10 = pre->g[1][0];
    g11 = pre->g[1][1];
    g20 = pre->g[2][0];
    g21 = pre->g[2][1];
  } else {
    g00 = *reinterpret_cast<const float4*>(args.x_gamma + col0);
    g01 = *reinterpret_cast<const float4*>(args.x_gamma + col0 + 4);
    g10 = *reinterpret_cast<const float4*>(args.x_gamma + col1);
    g11 = *reinterpret_cast<const float4*>(args.x_gamma + col1 + 4);
    g20 = *reinterpret_cast<const float4*>(args.x_gamma + col2);
    g21 = *reinterpret_cast<const float4*>(args.x_gamma + col2 + 4);
  }

  uint64_t sums01 = 0;
  uint64_t sums23 = 0;
  sums01 = accumulate_bf16x8_f32x2(sums01, x00, x10);
  sums01 = accumulate_bf16x8_f32x2(sums01, x01, x11);
  sums01 = accumulate_bf16x8_f32x2(sums01, x02, x12);
  sums23 = accumulate_bf16x8_f32x2(sums23, x20, x30);
  sums23 = accumulate_bf16x8_f32x2(sums23, x21, x31);
  sums23 = accumulate_bf16x8_f32x2(sums23, x22, x32);

  float sum0 = warp_sum(f32x2_low(sums01));
  float sum1 = warp_sum(f32x2_high(sums01));
  float sum2 = warp_sum(f32x2_low(sums23));
  float sum3 = warp_sum(f32x2_high(sums23));
  if (ctx.lane == 0) {
    reduce_scratch[ctx.warp * kQlen + 0] = sum0;
    reduce_scratch[ctx.warp * kQlen + 1] = sum1;
    reduce_scratch[ctx.warp * kQlen + 2] = sum2;
    reduce_scratch[ctx.warp * kQlen + 3] = sum3;
  }
  ctx.sync();

  float inverse_lane = 0.0f;
  if (ctx.lane < kQlen) {
    inverse_lane = reduce_scratch[ctx.lane];
#pragma unroll
    for (int warp = 1; warp < kOpWarps; ++warp) {
      inverse_lane = __fadd_rn(inverse_lane, reduce_scratch[warp * kQlen + ctx.lane]);
    }
    inverse_lane = rsqrtf(inverse_lane * (1.0f / static_cast<float>(kHidden)) + args.x_eps);
  }
  const float inverse0 = __shfl_sync(0xffffffffu, inverse_lane, 0);
  const float inverse1 = __shfl_sync(0xffffffffu, inverse_lane, 1);
  const float inverse2 = __shfl_sync(0xffffffffu, inverse_lane, 2);
  const float inverse3 = __shfl_sync(0xffffffffu, inverse_lane, 3);

  stage_retained_x_chunk(xnorm, col0, g00, g01, inverse0, inverse1, inverse2, inverse3, x00, x10,
                         x20, x30);
  stage_retained_x_chunk(xnorm, col1, g10, g11, inverse0, inverse1, inverse2, inverse3, x01, x11,
                         x21, x31);
  stage_retained_x_chunk(xnorm, col2, g20, g21, inverse0, inverse1, inverse2, inverse3, x02, x12,
                         x22, x32);
  ctx.sync();
}

// Warp record of one page segment: pairs (segment 0, segment 1) per (token, row) cell,
// [warp][cell][2].
__device__ __forceinline__ void store_k1_segment_local(float* local, int owned, int warp,
                                                       const float (&run)[2][4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int tp = lane & 3;
  if (tp < 2) {
    float* warp_pairs = local + 2 * warp * kK1SegmentCells + owned;
    const int token0 = 2 * tp;
    const int token1 = token0 + 1;
    warp_pairs[2 * (token0 * 32 + g)] = run[0][0];
    warp_pairs[2 * (token1 * 32 + g)] = run[0][1];
    warp_pairs[2 * (token0 * 32 + g + 8)] = run[0][2];
    warp_pairs[2 * (token1 * 32 + g + 8)] = run[0][3];
    warp_pairs[2 * (token0 * 32 + g + 16)] = run[1][0];
    warp_pairs[2 * (token1 * 32 + g + 16)] = run[1][1];
    warp_pairs[2 * (token0 * 32 + g + 24)] = run[1][2];
    warp_pairs[2 * (token1 * 32 + g + 24)] = run[1][3];
  }
}

static __device__ __forceinline__ void op_k1_qkv_a_proj_cache_pre(const MegaArgs& args, OpCtx& ctx,
                                                                  const K1EntryLoads* pre) {
  const bool null_math = args.mode == kModeNullOrchestration;
  // RMS scratch f32[8 warps][4 tokens] during the x-norm, then the segment records [8][128
  // cells][2] f32.
  float* reduce_scratch = ctx.scratch<float>(smem::kGemmPartial);
  __nv_bfloat16* xnorm = ctx.scratch<__nv_bfloat16>(smem::kActivation);  // bf16 [4][kXnormStride]
  __nv_bfloat16* publish_s = ctx.scratch<__nv_bfloat16>(kK1PublishOffset);
  float* qnorm_publish_s = ctx.scratch<float>(kK1QnormPublishOffset);
  __nv_bfloat16* pe_raw = wsp<__nv_bfloat16>(ctx.arena, arena::kPeRaw);
  const int block = ctx.cta;
  if (block >= kQkvAProjCtaBlocks) return;  // CTAs 82..127: no rows, no pages in K1

  // x-norm into smem (padded row stride), all three 2048-column bands retained in registers.
  if (!null_math) {
    stage_xnorm_all_retained_f32x2(args, ctx, reduce_scratch, xnorm, pre);
  }

  // qkv_a_proj GEMV over the block's 6 pages as two segments (4+2 / 2+4).  Page layout:
  // row half h at +h*0x4000, k16 step t of chunk w at +t*0x800 + w*0x100; scale at +0x8000 + 4w.
  // Keep one cold instruction body for both segments and all pages.  These loops
  // execute only once per call, so rolling them trades a few uniform branches
  // for substantially less first-touch I-cache pressure.
  const int cut = (block & 1) ? 2 : 4;
#pragma unroll 1
  for (int owned = 0; owned < 2; ++owned) {
    const int local_page0 = owned == 0 ? 0 : cut;
    const int pages_per_segment = owned == 0 ? cut : kQkvAProjPagesPerBlock - cut;
    float run[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    if (!null_math) {
      const uint32_t act_s = smem_u32(xnorm);
      const uint32_t a_lane =
          static_cast<uint32_t>(ctx.warp) * 0x100u + static_cast<uint32_t>(ctx.lane) * 8u;
#pragma unroll 1
      for (int page = 0; page < 4; ++page) {
        if (page < pages_per_segment) {
          const uint32_t slot = ctx.acquire_page();
          HmmaAcc acc[2] = {{{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}};
          const float scale = gemv_page_steps_b_reuse_release<8>(
              ctx, slot + a_lane, 0x4000u, 0x800u, act_s,
              smem::kXnormStride * static_cast<int>(sizeof(__nv_bfloat16)),
              (local_page0 + page) * 1024 + ctx.warp * 128,
              slot + 0x8000u + static_cast<uint32_t>(ctx.warp) * 4u, acc);
          fma_scale_into(run[0], scale, acc[0]);
          fma_scale_into(run[1], scale, acc[1]);
        }
      }
    }
    store_k1_segment_local(reduce_scratch, owned, ctx.warp, run);
  }
  ctx.sync();

  // Epilogue: thread (token, row) folds the eight warp records (packed pairs) in order 0..7, then
  // the two segments; bf16 to post_q / post_kv / pe_raw.
  if (ctx.tid < kK1SegmentCells) {
    const uint64_t* records = reinterpret_cast<const uint64_t*>(reduce_scratch);
    uint64_t value_pair = records[ctx.tid];
#pragma unroll
    for (int warp = 1; warp < kOpWarps; ++warp) {
      value_pair = add_f32x2(value_pair, records[warp * kK1SegmentCells + ctx.tid]);
    }
    const float folded = __fadd_rn(f32x2_low(value_pair), f32x2_high(value_pair));
    const int token = ctx.tid >> 5;
    const int row = ctx.tid & 31;
    const __nv_bfloat16 result = to_bf16(null_math ? 0.0f : folded);
    publish_s[token * kK1PublishRows + row] = result;
    // Publish the square sum of this 32-row, BF16-rounded post_q block while
    // its values are still in registers.  C0 orders these partials together
    // with post_q, eliminating the redundant K2-side CTA reduction.
    if (block < kQkvAProjQBlocks) {
      const float rounded = bf16_to_float(result);
      float square_sum = __fmul_rn(rounded, rounded);
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        square_sum = __fadd_rn(square_sum, __shfl_down_sync(0xffffffffu, square_sum, delta));
      }
      if (row == 0) {
        qnorm_publish_s[token] = square_sum;
      }
    }
  }
  ctx.sync();
  if (ctx.tid == 0) {
    constexpr uint32_t kTokenBytes = kK1PublishRows * uint32_t(sizeof(__nv_bfloat16));
    const int grow_base = block * kK1PublishRows;
    __nv_bfloat16* output_base;
    int output_stride;
    int output_row;
    if (grow_base < kQlora) {
      output_base = args.post_q;
      output_stride = kQlora;
      output_row = grow_base;
    } else if (grow_base < kQlora + kKvlora) {
      output_base = args.post_kv;
      output_stride = kKvlora;
      output_row = grow_base - kQlora;
    } else {
      output_base = pe_raw;
      output_stride = kRope;
      output_row = grow_base - kQlora - kKvlora;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
    for (int token = 0; token < kQlen; ++token) {
      bulk_copy_s2g(reinterpret_cast<uint8_t*>(output_base + token * output_stride + output_row),
                    smem_u32(publish_s + token * kK1PublishRows), kTokenBytes);
    }
    if (block < kQkvAProjQBlocks) {
      bulk_copy_s2g(
          reinterpret_cast<uint8_t*>(wsp<float>(ctx.arena, arena::kQnormPartial) + block * kQlen),
          smem_u32(qnorm_publish_s), kK1QnormPublishBytes);
    }
    bulk_copy_s2g_wait();
    red_release_gpu_add(ctx.counters + cnt_word(kCntC0), 1u);
  }
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

constexpr int kK2QnormOffset = 0;  // bf16 [4][kQnormStride]
constexpr int kK2QnormBytes = kQlen * smem::kQnormStride * int(sizeof(__nv_bfloat16));
constexpr int kK2WPartialOffset =
    (kK2QnormBytes + 127) & ~127;  // f64 [8][4][16] q_b_proj partials + f32 RMS [8][4]
constexpr int kK2WPartialDoubles = kOpWarps * kQlen * 16;  // fp64 q_b_proj fold
constexpr int kK2WPartialBytes =
    (kK2WPartialDoubles * int(sizeof(double)) + kOpWarps * kQlen * int(sizeof(float)) + 127) & ~127;
constexpr int kK2WPublishOffset = kK2WPartialOffset + kK2WPartialBytes;
constexpr int kK2WPublishRows = 16;
constexpr int kK2WPublishBytes = kQlen * kK2WPublishRows * int(sizeof(__nv_bfloat16));
constexpr int kK2FinalPublishOffset = kK2WPublishOffset + kK2WPublishBytes;
constexpr int kK2FinalPublishRows = 64;
constexpr int kK2FinalPublishBytes = kQlen * kK2FinalPublishRows * int(sizeof(__nv_bfloat16));
constexpr int kK2QndOffset = 0;  // W_UK slice [4][200], after q_b_proj
static_assert(kK2WPartialOffset + kK2WPartialBytes <= smem::kOpScratchBytes,
              "fused K2 scratch exceeds the carve-out");
static_assert(kK2WPublishOffset + kK2WPublishBytes <= smem::kOpScratchBytes,
              "fused K2 q_b_proj publication slab exceeds the carve-out");
static_assert(kK2FinalPublishOffset + kK2FinalPublishBytes <= smem::kOpScratchBytes,
              "fused K2 final publication slab exceeds the carve-out");
static_assert(kK2WPartialDoubles * int(sizeof(double)) + kOpWarps * kQlen * int(sizeof(float)) <=
                  kK2WPartialBytes,
              "K-owned qnorm RMS partials overlap or exceed the q_b_proj partial scratch");
static_assert(kK2WPartialOffset % 16 == 0 && (kQlen * 16 * int(sizeof(double))) % 16 == 0,
              "paired q_b_proj partial loads must stay naturally aligned");
static_assert(kQlen * smem::kUkActStride * int(sizeof(__nv_bfloat16)) <= smem::kOpScratchBytes,
              "W_UK slice fits");
struct K2GammaLoads {
  float4 g[2];
};

__device__ __forceinline__ void k2_gamma_issue(const MegaArgs& args, int warp, int lane,
                                               K2GammaLoads& out) {
#pragma unroll
  for (int page = 0; page < 2; ++page) {
    const int col = page * 1024 + warp * 128 + lane * 4;
    out.g[page] = *reinterpret_cast<const float4*>(args.q_gamma + col);
  }
}

// Full-rank specialization: cache the two warp-owned K128 post_q slices and
// both gamma vectors.  RMS totals come from the 64 K1-published block
// partials covered by C0, retiring the K2-side cross-warp rendezvous.
static __device__ __forceinline__ void compute_qnorm_kowned_gamma_prefetch(
    const MegaArgs& args, OpCtx& ctx, __nv_bfloat16* qnorm, const K2GammaLoads* pre = nullptr) {
  const int warp = ctx.warp;
  const int lane = ctx.lane;
  uint2 cached[kQlen][2];

#pragma unroll
  for (int token = 0; token < kQlen; ++token) {
#pragma unroll
    for (int page = 0; page < 2; ++page) {
      const int col = page * 1024 + warp * 128 + lane * 4;
      cached[token][page] = *reinterpret_cast<const uint2*>(args.post_q + token * kQlora + col);
    }
  }

  float4 cached_gamma[2];
  if (pre != nullptr) {
    cached_gamma[0] = pre->g[0];
    cached_gamma[1] = pre->g[1];
  } else {
#pragma unroll
    for (int page = 0; page < 2; ++page) {
      const int col = page * 1024 + warp * 128 + lane * 4;
      cached_gamma[page] = *reinterpret_cast<const float4*>(args.q_gamma + col);
    }
  }

  // Lane l owns adjacent block records 2l and 2l+1.  K1 publishes each
  // record as one float4, so two coalesced 16-byte loads provide all four
  // token partials while keeping the pairwise reduction order of the contract.
  const float* partial_rows = wsp<const float>(ctx.arena, arena::kQnormPartial);
  const float4 block0 = *reinterpret_cast<const float4*>(partial_rows + (2 * lane) * kQlen);
  const float4 block1 = *reinterpret_cast<const float4*>(partial_rows + (2 * lane + 1) * kQlen);
  const float block0_values[kQlen] = {block0.x, block0.y, block0.z, block0.w};
  const float block1_values[kQlen] = {block1.x, block1.y, block1.z, block1.w};

  float inverse[kQlen];
#pragma unroll
  for (int token = 0; token < kQlen; ++token) {
    float sum = __fadd_rn(block0_values[token], block1_values[token]);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, delta));
    }
    sum = __shfl_sync(0xffffffffu, sum, 0);
    inverse[token] = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / float(kQlora)), args.q_eps));
  }

#pragma unroll
  for (int page = 0; page < 2; ++page) {
    const int col = page * 1024 + warp * 128 + lane * 4;
    const float* g = reinterpret_cast<const float*>(&cached_gamma[page]);
#pragma unroll
    for (int token = 0; token < kQlen; ++token) {
      const uint2 packed = cached[token][page];
      const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
      uint2 packed_out;
      __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(&packed_out);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        output[i] = to_bf16(__fmul_rn(__fmul_rn(bf16_to_float(values[i]), inverse[token]), g[i]));
      }
      *reinterpret_cast<uint2*>(qnorm + token * smem::kQnormStride + col) = packed_out;
    }
  }
  __syncwarp(0xffffffffu);
}

static __device__ __forceinline__ void consume_q_b_proj_half(const MegaArgs& args, OpCtx& ctx,
                                                             __nv_bfloat16* qnorm, double* partial,
                                                             int row_base, int row_half,
                                                             bool active,
                                                             __nv_bfloat16* publish_s) {
  float* fpartial = reinterpret_cast<float*>(partial);  // same carve-out, fp32 records
  float run[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (active) {
    const uint32_t act_s = smem_u32(qnorm);
    const uint32_t a_lane =
        static_cast<uint32_t>(ctx.warp) * 0x800u + static_cast<uint32_t>(ctx.lane) * 8u;
#pragma unroll
    for (int page = 0; page < 2; ++page) {
      const uint32_t slot = ctx.acquire_page();
      HmmaAcc acc[1] = {{{0.0f, 0.0f, 0.0f, 0.0f}}};
      gemv_page_steps<1, 8>(slot + a_lane, 0x4000u, 0x100u, act_s, smem::kQnormStride * 2,
                            page * 1024 + ctx.warp * 128, acc);
      const float scale = lds_f32(slot + 0x4000u + static_cast<uint32_t>(ctx.warp) * 4u);
      ctx.release_page();
      fma_scale_into(run, scale, acc[0]);
    }
    store_partial_tile(fpartial + ctx.warp * (kQlen * 16), 16, 0, run);
  }
  ctx.sync();

  if (ctx.tid < kQlen * 8) {
    const int token = ctx.tid >> 3;
    const int row = (ctx.tid & 7) << 1;
    F32Pair value{};
    if (active) {
      value = reduce_partial_pair<kOpWarps, 16>(fpartial, token, row);
    }
    Bf16Pair rounded;
    rounded.bits = pack_bf16_pair(value.value[0], value.value[1]);
    const int grow = row_base + row;
    if (grow < kQbProjNopeRows) {
      __nv_bfloat16* output = publish_s ? publish_s + token * kK2WPublishRows + row
                                        : args.post_qnd + token * kQbProjNopeRows + grow;
      *reinterpret_cast<uint32_t*>(output) = rounded.bits;
    } else {
      const int pe_index = grow - kQbProjNopeRows;
      __nv_bfloat16* output = publish_s ? publish_s + token * kK2WPublishRows + row
                                        : args.post_qpe + token * kQbProjPeRows + pe_index;
      if (active) {
        const int dim = pe_index & (kRope - 1);
        const float real = __uint_as_float((rounded.bits & 0xffffu) << 16);
        const float imag = __uint_as_float(rounded.bits & 0xffff0000u);
        const float cosine = args.rope[(size_t(g_mega_cur_pos) + token) * kRope + dim];
        const float sine = args.rope[(size_t(g_mega_cur_pos) + token) * kRope + dim + 1];
        const float out_real = __fsub_rn(__fmul_rn(real, cosine), __fmul_rn(imag, sine));
        const float out_imag = __fadd_rn(__fmul_rn(imag, cosine), __fmul_rn(real, sine));
        rounded.bits = pack_bf16_pair(out_real, out_imag);
        *reinterpret_cast<uint32_t*>(output) = rounded.bits;
      } else {
        *reinterpret_cast<uint32_t*>(output) = 0u;
      }
    }
  }
}

// W_UK step of the fused op: rows [64 uk_cta, +64) of head uk_cta / 8 after the head's 12
// cohort arrivals (post_q_nope_down of the head complete).  Inactive (rank7 pad head / null math):
// zeros.
static __device__ void k2_uk_fused(const MegaArgs& args, OpCtx& ctx, int uk_cta, bool active,
                                   __nv_bfloat16* publish_s) {
  __nv_bfloat16* qnd_tile = ctx.scratch<__nv_bfloat16>(kK2QndOffset);
  constexpr int kUkRowsPerCta = 64;
  const int uk_row_base = kUkRowsPerCta * uk_cta;
  const int uk_head = uk_row_base / kKvlora;
  if (!active) {
    // Inactive heads have no ring item but still require exact zeroes.
    const int token = ctx.tid >> 6;
    const int row = ctx.tid & 63;
    publish_s[token * kK2FinalPublishRows + row] = to_bf16(0.0f);
    return;
  }
  uint32_t slot;
  if (args.mode == kModePersistent) {
    if (ctx.tid < kQlen * (kNope / 4)) {
      const int token = ctx.tid / (kNope / 4);
      const int k = ctx.tid - token * (kNope / 4);
      const uint2 w =
          rec_poll(ctx.arena + kRecQnd + (size_t(uk_head * kQlen + token) * (kNope / 4) + k) * 16,
                   g_mega_epoch);
      *reinterpret_cast<uint2*>(qnd_tile + token * smem::kUkActStride + 4 * k) = w;
    }
    slot = ctx.acquire_page();
    ctx.sync();
  } else {
    if (ctx.tid == 0) {
      const uint32_t qnd_target =
          ArrivalCounter::target(kArriveHeadCycle, g_mega_epoch - 1u) + kArriveK2Head;
      ctx.counter(kCntK2Head + uk_head).wait(qnd_target);
    }
    ctx.sync();

    stage_activation_rows_async(qnd_tile, smem::kUkActStride, args.post_qnd + uk_head * kNope,
                                kHeads * kNope, kNope);
    slot = ctx.acquire_page();
    wait_activation_rows_async();
    ctx.sync();
  }
  const uint32_t act_s = smem_u32(qnd_tile);

  // Four warps own the four 16-row groups.  Interleave the three independent
  // K-chunk accumulators at each k16 step, then fold them in registers.
  HmmaAcc chunk_acc[3] = {
      {{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}};
  float chunk_scale[3] = {0.0f, 0.0f, 0.0f};
  if (ctx.warp < 4) {
    const int rg = ctx.warp;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      RawFp8x8 raw[3];
      uint32_t b[3][2];
#pragma unroll
      for (int chunk = 0; chunk < 3; ++chunk) {
        raw[chunk] = lds_fp8x8(
            slot + static_cast<uint32_t>(rg) * 0xC00u + static_cast<uint32_t>(chunk) * 0x400u +
            static_cast<uint32_t>(kt) * 0x100u + static_cast<uint32_t>(ctx.lane) * 8u);
        lds_act_fragment(act_s, smem::kUkActStride * 2, chunk * 64 + kt * 16, b[chunk]);
      }
#pragma unroll
      for (int chunk = 0; chunk < 3; ++chunk) {
        uint32_t a[4];
        fp8_to_bf16_fragment(raw[chunk], a);
        hmma(chunk_acc[chunk], a, b[chunk]);
      }
    }
#pragma unroll
    for (int chunk = 0; chunk < 3; ++chunk) {
      chunk_scale[chunk] = lds_f32(slot + 0x3000u + static_cast<uint32_t>(chunk) * 4u);
    }
  }
  // The remaining arithmetic uses registers, so the page can be released now.
  ctx.release_page();

  if (ctx.warp < 4) {
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Preserve chunk order while contracting the two dependent add/multiply
      // pairs.  The final BF16 store keeps the stage's single BF16 rounding boundary.
      value[i] = __fmul_rn(chunk_acc[0].value[i], chunk_scale[0]);
      value[i] = __fmaf_rn(chunk_acc[1].value[i], chunk_scale[1], value[i]);
      value[i] = __fmaf_rn(chunk_acc[2].value[i], chunk_scale[2], value[i]);
    }

    const int g = ctx.lane >> 2;
    const int token = (ctx.lane & 3) * 2;
    if ((ctx.lane & 3) < 2) {
      const int row = ctx.warp * 16 + g;
      publish_s[token * kK2FinalPublishRows + row] = to_bf16(value[0]);
      publish_s[(token + 1) * kK2FinalPublishRows + row] = to_bf16(value[1]);
      publish_s[token * kK2FinalPublishRows + row + 8] = to_bf16(value[2]);
      publish_s[(token + 1) * kK2FinalPublishRows + row + 8] = to_bf16(value[3]);
    }
  }
  // No trailing consumer barrier: the persistent caller's next
  // boundary_arrive supplies its own.
}

static __device__ void op_k2_fused_pre(const MegaArgs& args, OpCtx& ctx,
                                       const K2GammaLoads* gamma_pre) {
  const bool null_math = args.mode == kModeNullOrchestration;
  const int head = ctx.cta / kK2FusedCohort;
  const int cohort = ctx.cta - head * kK2FusedCohort;
  const bool active = (args.rank != 7 || head < 4) && !null_math;
  double* partial = ctx.scratch<double>(kK2WPartialOffset);  // fp64 q_b_proj fold partials
  __nv_bfloat16* qnorm = ctx.scratch<__nv_bfloat16>(kK2QnormOffset);
  __nv_bfloat16* publish_s = ctx.scratch<__nv_bfloat16>(kK2WPublishOffset);
  __nv_bfloat16* final_publish_s = ctx.scratch<__nv_bfloat16>(kK2FinalPublishOffset);

  // K-owned q-norm slices; C0 also acquires the K1 RMS partials, so
  // no CTA-wide reduction barrier remains before the first q_b_proj page.
  if (active) {
    compute_qnorm_kowned_gamma_prefetch(args, ctx, qnorm, gamma_pre);
  }
  // NOPE half tile of this cohort (pages 0, 1 of the block), then publish the head.
  const int nope_row_base = head * kNope + cohort * 16;
  consume_q_b_proj_half(args, ctx, qnorm, partial, nope_row_base, cohort & 1, active, publish_s);
  // All twelve cohorts publish, rank-7 pads included; this barrier also frees the partial buffer.
  ctx.sync();
  if (args.mode == kModePersistent) {
    if (ctx.tid < kQlen * (kK2WPublishRows / 4)) {
      const int token = ctx.tid / (kK2WPublishRows / 4);
      const int k = ctx.tid - token * (kK2WPublishRows / 4);
      const uint2 w = *reinterpret_cast<const uint2*>(publish_s + token * kK2WPublishRows + 4 * k);
      rec_store_b128(
          ctx.arena + kRecQnd +
              (size_t(head * kQlen + token) * (kNope / 4) + cohort * (kK2WPublishRows / 4) + k) *
                  16,
          w.x, w.y, g_mega_epoch);
      *reinterpret_cast<uint2*>(args.post_qnd + token * kQbProjNopeRows + nope_row_base + 4 * k) =
          w;
    }
    if (ctx.tid == 0) red_relaxed_gpu_add(ctx.counters + cnt_word(kCntK2Head + head), 1u);
  } else if (ctx.tid == 0) {
    constexpr uint32_t kTokenBytes = kK2WPublishRows * uint32_t(sizeof(__nv_bfloat16));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
    for (int token = 0; token < kQlen; ++token) {
      bulk_copy_s2g(
          reinterpret_cast<uint8_t*>(args.post_qnd + token * kQbProjNopeRows + nope_row_base),
          smem_u32(publish_s + token * kK2WPublishRows), kTokenBytes);
    }
    bulk_copy_s2g_wait();
    red_release_gpu_add(ctx.counters + cnt_word(kCntK2Head + head), 1u);
  }
  if (cohort >= kK2FusedUkPerHead) {
    // PE half tile (pages 2, 3), RoPE fused into the store.
    const int pe_tile = cohort - kK2FusedUkPerHead;
    const int pe_row_base = kQbProjNopeRows + head * kRope + pe_tile * 16;
    consume_q_b_proj_half(args, ctx, qnorm, partial, pe_row_base, pe_tile & 1, active,
                          final_publish_s);
    return;
  }
  // W_UK tile of this cohort (page 2) after the head's 12 arrivals.
  k2_uk_fused(args, ctx, head * kK2FusedUkPerHead + cohort, active, final_publish_s);
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

constexpr int kK3QkDim = kKvlora + kRope;
constexpr int kK3QStride = kK3QkDim + 16;
constexpr int kK3PeStride = 72;
constexpr int kK3ScoreStride = 68;
constexpr int kK3ProbStride = 72;

constexpr int kK3ScratchQuery = 0;
constexpr int kK3ScratchPe =
    kK3ScratchQuery + kK3HeadsPadded * kK3QStride * int(sizeof(__nv_bfloat16));
constexpr int kK3ScratchScore =
    kK3ScratchPe + kK3OptRows * kK3PeStride * int(sizeof(__nv_bfloat16));
constexpr int kK3ScratchProb =
    kK3ScratchScore + kK3HeadsPadded * kK3ScoreStride * int(sizeof(float));
constexpr int kK3ScratchIds =
    kK3ScratchProb + kK3HeadsPadded * kK3ProbStride * int(sizeof(__nv_bfloat16));
constexpr int kK3ScratchAux = kK3ScratchIds + kK3OptRows * int(sizeof(int32_t));
constexpr int kK3ScratchCombine = kK3ScratchAux + kK3OptSplits * int(sizeof(float));
constexpr int kK3ScratchEnd = kK3ScratchCombine + 4 * 64 * int(sizeof(float4));
static_assert(kK3ProducerScratchIds == uint32_t(kK3ScratchIds) &&
                  kK3ProducerScratchAux == uint32_t(kK3ScratchAux),
              "producer and consumer K3 shared layouts diverged");
constexpr int kK3PublishPartialFloats = kHeads * kKvlora;
// Compact FP32 split-partial records: ten contiguous live 512-element head
// rows plus one 128 B pad per split (20608 B stride instead of the padded
// 16-head 32 KiB stride).  The pad rotates successive split bases across L2
// sets; the live partial address span of one call is 2.64 MB (4.19 MB with the
// padded stride).  Arithmetic, publication and combine order do not depend on
// the stride.
constexpr int kK3PartialStrideFloats = kHeads * kKvlora + 32;
constexpr int kK3PublishPartialBytes = kK3PublishPartialFloats * int(sizeof(float));
constexpr int kK3PublishStatsFloats = 2 * kK3HeadsPadded;
constexpr int kK3PublishStatsBytes = kK3PublishStatsFloats * int(sizeof(float));
static_assert(kK3ScratchEnd <= smem::kOpScratchBytes,
              "K3 scratch exceeds the op scratch region (smem::kOpScratchBytes)");
static_assert(kK3PublishPartialBytes + kK3PublishStatsBytes <= kK3ScratchIds,
              "K3 partial + statistics publication slab overlaps live ids");
static_assert((kK3PublishPartialBytes & 15) == 0 && (kK3PublishStatsBytes & 15) == 0,
              "K3 bulk publication sizes must be 16-byte multiples");
// Combine prefetch: kK3OptSplits x 64 float4 (32 KB) land in the dead query/pe/score/prob region
// [kK3ScratchQuery, kK3ScratchIds) once the PV has passed its consumer barrier; ids / aux / combine
// stay intact.
static_assert(kK3OptSplits * 64 * int(sizeof(float4)) <= kK3ScratchIds - kK3ScratchQuery,
              "K3 combine partial prefetch overflows the dead query/pe/score/prob scratch region");
static_assert(size_t(kQlen) * kK3OptSplits * kK3PartialStrideFloats * sizeof(float) <=
                  arena::kOAccBytes,
              "K3 partials exceed arena::kOAcc");
static_assert(size_t(kQlen) * kK3OptSplits * kK3HeadsPadded * 2 * sizeof(float) <=
                  arena::kScoresBytes,
              "K3 stats exceed arena::kScores");

__device__ __forceinline__ float k3_negative_infinity() { return __int_as_float(0xff800000u); }

__device__ __forceinline__ float k3_warp_max_broadcast(float value) {
  float result;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float k3_halfwarp_max_broadcast(float value, unsigned mask) {
#pragma unroll
  for (int offset = 8; offset; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset, 16));
  }
  return value;
}

__device__ __forceinline__ float k3_halfwarp_sum_broadcast(float value, unsigned mask) {
#pragma unroll
  for (int offset = 8; offset; offset >>= 1) {
    value += __shfl_xor_sync(mask, value, offset, 16);
  }
  return value;
}

// Publish the two accumulator rows directly to the globally coherent cache
// level.  The combine CTAs consume these values only after the release-counter
// acquire, so retaining private copies in each producer SM's L1 is wasted.
__device__ __forceinline__ void k3_store_partial_cg(float* first, float first_value, float* second,
                                                    float second_value) {
  asm volatile(
      "st.global.cg.f32 [%0], %2;\n"
      "st.global.cg.f32 [%1], %3;" ::"l"(first),
      "l"(second), "f"(first_value), "f"(second_value)
      : "memory");
}

union K3F32x2 {
  float2 values;
  unsigned long long bits;
};

__device__ __forceinline__ float2 k3_add_f32x2(float2 left, float2 right) {
  K3F32x2 a;
  K3F32x2 b;
  K3F32x2 result;
  a.values = left;
  b.values = right;
  asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(result.bits) : "l"(a.bits), "l"(b.bits));
  return result.values;
}

__device__ __forceinline__ const int32_t* k3_ids_source(const MegaArgs& args, uint8_t* local) {
  return args.ids_direct ? args.selected_ids_src
                         : wsp<int32_t>(local, arena::kIdsRx + size_t(g_mega_epoch & 1u) *
                                                                   arena::kIdsRxParityBytes);
}

__device__ __forceinline__ const int32_t* k3_lengths_source(const MegaArgs& args, uint8_t* local) {
  return args.ids_direct
             ? args.selected_lengths_src
             : wsp<int32_t>(local, arena::kIdsRx +
                                       size_t(g_mega_epoch & 1u) * arena::kIdsRxParityBytes +
                                       arena::kIdsRxLengths);
}

__device__ __forceinline__ void k3_cp_async_16(void* destination, const void* source) {
  const uint32_t shared = hmma_smem_addr(destination);
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(shared), "l"(source)
               : "memory");
}

__device__ __forceinline__ void k3_cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void k3_cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void k3_stage_pe(const MegaArgs& args, const int32_t* ids_s,
                                            __nv_bfloat16* pe_s, int rows_per_split, int tid) {
  constexpr int kPeVectors = kRope / 8;
  for (int index = tid; index < rows_per_split * kPeVectors; index += kOpThreads) {
    const int row = index / kPeVectors;
    const int vector = index - row * kPeVectors;
    const int id = ids_s[row];
    if (id >= 0) {
      k3_cp_async_16(pe_s + row * kK3PeStride + vector * 8,
                     args.pe_cache + static_cast<int64_t>(id) * kRope + vector * 8);
    } else {
      *reinterpret_cast<uint4*>(pe_s + row * kK3PeStride + vector * 8) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

// The query slab split by consumer.  The QK-PE MMA needs only the 1.3 KB
// post_q_pe slice; the 10 KB post_q_nope slice is consumed by the QK-KV MMA
// after the ring wait.  Staging them as separate commit groups lets the PE MMA
// start behind a wait_group 1 while the NOPE group is still in flight.
__device__ __forceinline__ void k3_stage_query_nope(const MegaArgs& args, int query,
                                                    __nv_bfloat16* query_s, int tid) {
  constexpr int kNopeVectors = kKvlora / 8;
  for (int index = tid; index < kHeads * kNopeVectors; index += kOpThreads) {
    const int head = index / kNopeVectors;
    const int vector = index - head * kNopeVectors;
    k3_cp_async_16(query_s + head * kK3QStride + vector * 8,
                   args.post_qnope + (query * kHeads + head) * kKvlora + vector * 8);
  }
}
__device__ __forceinline__ void k3_stage_query_pe(const MegaArgs& args, int query,
                                                  __nv_bfloat16* query_s, int tid) {
  constexpr int kPeVectors = kRope / 8;
  for (int index = tid; index < kHeads * kPeVectors; index += kOpThreads) {
    const int head = index / kPeVectors;
    const int vector = index - head * kPeVectors;
    k3_cp_async_16(query_s + head * kK3QStride + kKvlora + vector * 8,
                   args.post_qpe + (query * kHeads + head) * kRope + vector * 8);
  }
}
__device__ __forceinline__ void k3_poll_query_records(const uint8_t* arena, int query,
                                                      __nv_bfloat16* query_s, int tid,
                                                      uint32_t tag) {
  constexpr int kNopeRecs = kHeads * (kKvlora / 4);
  constexpr int kPeRecs = kHeads * (kRope / 4);
  constexpr int kRecs = kNopeRecs + kPeRecs;
  constexpr int kPer = (kRecs + kOpThreads - 1) / kOpThreads;
  const uint8_t* address[kPer];
  __nv_bfloat16* destination[kPer];
  uint64_t lo[kPer], hi[kPer];
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const int r = tid + i * kOpThreads;
    address[i] = nullptr;
    destination[i] = nullptr;
    if (r < kNopeRecs) {
      const int head = r / (kKvlora / 4);
      const int r4 = r - head * (kKvlora / 4);
      address[i] = arena + kRecQnope + (size_t(query * kHeads + head) * (kKvlora / 4) + r4) * 16;
      destination[i] = query_s + head * kK3QStride + 4 * r4;
    } else if (r < kRecs) {
      const int p = r - kNopeRecs;
      const int head = p / (kRope / 4);
      const int d4 = p - head * (kRope / 4);
      address[i] = arena + kRecQpe + (size_t(query * kHeads + head) * (kRope / 4) + d4) * 16;
      destination[i] = query_s + head * kK3QStride + kKvlora + 4 * d4;
    }
    lo[i] = 0ull;
    hi[i] = 0ull;
    if (address[i] != nullptr) {
      asm volatile(
          "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
          : "=l"(lo[i]), "=l"(hi[i])
          : "l"(address[i])
          : "memory");
    }
  }
  for (;;) {
    bool done = true;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (address[i] != nullptr && (static_cast<uint32_t>(lo[i] >> 32) != tag ||
                                    static_cast<uint32_t>(hi[i] >> 32) != tag)) {
        done = false;
        asm volatile(
            "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, "
            "r;\n\t}"
            : "=l"(lo[i]), "=l"(hi[i])
            : "l"(address[i])
            : "memory");
      }
    }
    if (done) break;
  }
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    if (destination[i] != nullptr) {
      *reinterpret_cast<uint2*>(destination[i]) =
          make_uint2(static_cast<uint32_t>(lo[i]), static_cast<uint32_t>(hi[i]));
    }
  }
}

// Folded softmax constant c = softmax_scale * log2(e), computed once per CTA from the runtime
// softmax_scale.
__device__ __forceinline__ float k3_exp2_scale(float softmax_scale) {
  return __fmul_rn(softmax_scale, 1.4426950408889634074f);
}

// One-pass exponential of one raw score against the split's raw local max: exp2f((s - m) * c).
__device__ __forceinline__ float k3_exp_local(float score, bool valid, float local_max, float c) {
  return valid ? exp2f(__fmul_rn(__fsub_rn(score, local_max), c)) : 0.0f;
}

// Warps 2..7 retain exp2f((score - local_max) * c) in the dead score tile as
// FP32 values. The FP32 row sum and local maximum are published separately;
// probability rounding occurs only after the global normalization below.
// An all-masked split publishes (-inf, 0) and retains zero exponentials.
__device__ __forceinline__ void k3_softmax_head_adjacent(int query, int split, int head, float c,
                                                         const int32_t* ids_s, float* score_s,
                                                         __nv_bfloat16* probability_s,
                                                         float* stats_s, int lane) {
  const int row0 = 2 * lane;
  const int2 id_pair = *reinterpret_cast<const int2*>(ids_s + row0);
  const float2 score_pair =
      *reinterpret_cast<const float2*>(score_s + head * kK3ScoreStride + row0);
  const float neg_inf = k3_negative_infinity();
  const bool valid0 = id_pair.x >= 0;
  const bool valid1 = id_pair.y >= 0;
  const float s0 = valid0 ? score_pair.x : neg_inf;
  const float s1 = valid1 ? score_pair.y : neg_inf;
  const float local_max = k3_warp_max_broadcast(fmaxf(s0, s1));
  float e0 = 0.0f;
  float e1 = 0.0f;
  if (local_max != neg_inf) {
    e0 = k3_exp_local(s0, valid0, local_max, c);
    e1 = k3_exp_local(s1, valid1, local_max, c);
  }
  *reinterpret_cast<float2*>(score_s + head * kK3ScoreStride + row0) = make_float2(e0, e1);
  // Only lane 0 publishes; retain the down-tree result there (no broadcast).
  // The (max, sum) pair goes straight to this split's globally coherent
  // statistics record (head-major stride): no shared staging slab and no
  // separate copy pass sit between the softmax and the release.
  const float local_sum = warp_sum(e0 + e1);
  if (lane == 0) {
    *reinterpret_cast<float2*>(stats_s + head * (2 * kK3OptSplits)) =
        make_float2(local_max, local_sum);
  }
}

// Warps 0 and 1: two heads each (h and h+8) with 16-lane reductions, so both heads progress
// together instead of serializing a second full-warp pass after heads 0 and 1.  The four row groups
// cover the largest 64-row split; ids_s masks unused rows for 16/32/48-row geometries.  Same
// arithmetic as the adjacent form.
__device__ __forceinline__ void k3_softmax_head_pair(int query, int split, int low_head, float c,
                                                     const int32_t* ids_s, float* score_s,
                                                     __nv_bfloat16* probability_s, float* stats_s,
                                                     int lane) {
  const int half = lane >> 4;
  const int lane16 = lane & 15;
  const int head = low_head + 8 * half;
  const unsigned mask = half ? 0xffff0000u : 0x0000ffffu;
  const int row0 = 2 * lane16;
  const int row2 = row0 + 32;
  const int2 ids01 = *reinterpret_cast<const int2*>(ids_s + row0);
  const int2 ids23 = *reinterpret_cast<const int2*>(ids_s + row2);
  const float2 scores01 = *reinterpret_cast<const float2*>(score_s + head * kK3ScoreStride + row0);
  const float2 scores23 = *reinterpret_cast<const float2*>(score_s + head * kK3ScoreStride + row2);
  const float neg_inf = k3_negative_infinity();
  const bool valid0 = ids01.x >= 0;
  const bool valid1 = ids01.y >= 0;
  const bool valid2 = ids23.x >= 0;
  const bool valid3 = ids23.y >= 0;
  const float s0 = valid0 ? scores01.x : neg_inf;
  const float s1 = valid1 ? scores01.y : neg_inf;
  const float s2 = valid2 ? scores23.x : neg_inf;
  const float s3 = valid3 ? scores23.y : neg_inf;
  float local_max = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
  local_max = k3_halfwarp_max_broadcast(local_max, mask);
  float e0 = 0.0f;
  float e1 = 0.0f;
  float e2 = 0.0f;
  float e3 = 0.0f;
  if (local_max != neg_inf) {
    e0 = k3_exp_local(s0, valid0, local_max, c);
    e1 = k3_exp_local(s1, valid1, local_max, c);
    e2 = k3_exp_local(s2, valid2, local_max, c);
    e3 = k3_exp_local(s3, valid3, local_max, c);
  }
  *reinterpret_cast<float2*>(score_s + head * kK3ScoreStride + row0) = make_float2(e0, e1);
  *reinterpret_cast<float2*>(score_s + head * kK3ScoreStride + row2) = make_float2(e2, e3);
  float local_sum = (e0 + e1) + (e2 + e3);
  local_sum = k3_halfwarp_sum_broadcast(local_sum, mask);
  if (lane16 == 0) {
    *reinterpret_cast<float2*>(stats_s + head * (2 * kK3OptSplits)) =
        make_float2(local_max, local_sum);
  }
}

// One head per 16-lane half warp. Pairing split l with l+16 locally
// reproduces the first stage of the full-warp reduction tree; the
// remaining width-16 down tree therefore keeps the exact FP32 association.
__device__ __forceinline__ void k3_global_probability_head_halfwarp(
    int query, int split, int head, float c, const int32_t* ids_s, const float* score_s,
    const float* stats_g, __nv_bfloat16* probability_s, int lane) {
  const int lane16 = lane & 15;
  const unsigned mask = (lane & 16) ? 0xffff0000u : 0x0000ffffu;
  const int stat_base = (query * kK3HeadsPadded + head) * kK3OptSplits * 2;
  const float2 stats_lo = *reinterpret_cast<const float2*>(stats_g + stat_base + 2 * lane16);
  const float2 stats_hi = *reinterpret_cast<const float2*>(stats_g + stat_base + 2 * (lane16 + 16));
  const float neg_inf = k3_negative_infinity();
  const float global_max = k3_halfwarp_max_broadcast(fmaxf(stats_lo.x, stats_hi.x), mask);
  float weight_lo = 0.0f;
  float term_lo = 0.0f;
  float weight_hi = 0.0f;
  float term_hi = 0.0f;
  if (stats_lo.x != neg_inf) {
    weight_lo = exp2f(__fmul_rn(__fsub_rn(stats_lo.x, global_max), c));
    term_lo = __fmul_rn(stats_lo.y, weight_lo);
  }
  if (stats_hi.x != neg_inf) {
    weight_hi = exp2f(__fmul_rn(__fsub_rn(stats_hi.x, global_max), c));
    term_hi = __fmul_rn(stats_hi.y, weight_hi);
  }
  float denominator = term_lo + term_hi;
#pragma unroll
  for (int offset = 8; offset; offset >>= 1) {
    denominator += __shfl_down_sync(mask, denominator, offset, 16);
  }
  denominator = __shfl_sync(mask, denominator, 0, 16);
  const float inverse = denominator > 0.0f ? __frcp_rn(denominator) : 0.0f;
  const float own_weight = __shfl_sync(mask, split < 16 ? weight_lo : weight_hi, split & 15, 16);
  const int row = 4 * lane16;
  const int4 ids = *reinterpret_cast<const int4*>(ids_s + row);
  const float4 exponentials =
      *reinterpret_cast<const float4*>(score_s + head * kK3ScoreStride + row);
  const bool live = inverse > 0.0f;
  const float p0 =
      live && ids.x >= 0 ? __fmul_rn(__fmul_rn(exponentials.x, own_weight), inverse) : 0.0f;
  const float p1 =
      live && ids.y >= 0 ? __fmul_rn(__fmul_rn(exponentials.y, own_weight), inverse) : 0.0f;
  const float p2 =
      live && ids.z >= 0 ? __fmul_rn(__fmul_rn(exponentials.z, own_weight), inverse) : 0.0f;
  const float p3 =
      live && ids.w >= 0 ? __fmul_rn(__fmul_rn(exponentials.w, own_weight), inverse) : 0.0f;
  __nv_bfloat162* out_pair =
      reinterpret_cast<__nv_bfloat162*>(probability_s + head * kK3ProbStride + row);
  out_pair[0] = __floats2bfloat162_rn(p0, p1);
  out_pair[1] = __floats2bfloat162_rn(p2, p3);
}

static __device__ void op_k3_attention_split(const MegaArgs& args, OpCtx& ctx) {
  uint8_t* local = ctx.arena;
  // Padded rank-7 queries are zero by the public rank layout. Compute their
  // common attention result in head 4; combine replicates it to heads 5..9.
  // These raw latent outputs are not suppressed until the downstream W_UV.
  const int computed_heads = args.rank == 7 ? 5 : kHeads;

  // With ids_direct the ids come straight from selected_ids_src / the LL records, and with
  // ids_ready the tail CTAs already consumed them.  Otherwise CTA 0 consumes the bulk-copy
  // publication into this rank's arena first.
  if (!args.ids_direct && !args.ids_ready) {
    if (ctx.cta == 0) {
      consume_selected_ids<1>(args, ctx);  // single-CTA consume, direct kFlagAck release
      if (ctx.tid == 0) ctx.counter(kCntIdsRx).arrive();
    }
    if (ctx.tid == 0) {
      ctx.counter(kCntIdsRx).wait(ArrivalCounter::target(kArriveIdsRx, g_mega_epoch));
    }
    ctx.sync();
  }

  if (ctx.cta >= kQlen * kK3OptSplits) return;

  // Persistent kernel: the ids / cache-row boundaries (C2, C3) land ~3 us before
  // the q-chain boundary (C4).  Wait for them first, fill ids_s and issue the PE
  // gather, and only then poll C4 -- the query slab is the only staging left on
  // the post-C4 path.  With ids_ready = 0 (kModeChain) stream order gates the inputs (no waits).
  if (args.ids_ready) {
    if (args.mode == kModePersistent) {
      asm volatile("barrier.cta.sync 7, 384;" ::: "memory");
    } else {
      // Null-orchestration has no producer warpgroup, so it keeps the direct
      // global-counter acquire.
      ctx.boundary_wait2(kCntC2, ArrivalCounter::target(kArriveC2, g_mega_epoch), kCntC3,
                         ArrivalCounter::target(kArriveC3, g_mega_epoch));
    }
  }

  const int query = ctx.cta >> 5;
  const int split = ctx.cta & 31;
  const bool producer_stages_ids = args.ids_ready && args.mode == kModePersistent;
  const int32_t* ids_source = k3_ids_source(args, local);
  const int32_t* lengths_source = k3_lengths_source(args, local);

  __nv_bfloat16* query_s = ctx.scratch<__nv_bfloat16>(kK3ScratchQuery);
  __nv_bfloat16* pe_s = ctx.scratch<__nv_bfloat16>(kK3ScratchPe);
  float* score_s = ctx.scratch<float>(kK3ScratchScore);
  __nv_bfloat16* probability_s = ctx.scratch<__nv_bfloat16>(kK3ScratchProb);
  int32_t* ids_s = ctx.scratch<int32_t>(kK3ScratchIds);
  float* partial_g = wsp<float>(local, arena::kOAcc);
  float* partial = partial_g + (query * kK3OptSplits + split) * kK3PartialStrideFloats;

  // In kModeChain with ids_ready = 0 stream order covers every input, so both query slabs can
  // stream into shared memory from the first cycle and overlap the id staging and the producer's
  // selected-KV ring flight.  The persistent schedule keeps its C4/C4Pe-gated staging below.
  const bool chain_entry_stage = args.mode == kModeChain && !args.ids_ready;
  if (chain_entry_stage) {
    k3_stage_query_pe(args, query, query_s, ctx.tid);
    k3_stage_query_nope(args, query, query_s, ctx.tid);
    k3_cp_async_commit();
  }

  // Match the producer's query-local traversal of a guaranteed full causal
  // prefix, using the live per-query length and the acquire paths above.
  // A guaranteed-complete prefix's live length equals the public cur_pos
  // scalar plus the query offset (the contract masks the whole visible
  // prefix for such queries), so no cold lengths read gates the id setup;
  // sparse queries keep their live lengths read.
  const bool canonical_dense = g_mega_cur_pos + query + 1 <= kSelected;
  const int length = min(kSelected, producer_stages_ids ? ids_s[kK3OptRows]
                                                        : ((canonical_dense || args.dense_ids)
                                                               ? g_mega_cur_pos + query + 1
                                                               : lengths_source[query]));
  // Balanced split geometry (see k3_split_row_lo in mla_q4_types.cuh): contiguous ascending
  // 16-row granule ranges per split, 64-key splits exactly on full
  // selections, every live key range shrunk evenly on shorter selections.
  const int rows_per_split = k3_split_row_cnt(length, split);
  const int row_tiles = rows_per_split >> 4;
  const int slot_base = k3_split_row_lo(length, split);
  const bool producer_stages_qnope = args.ids_ready && args.mode == kModePersistent;
  if (!producer_stages_ids) {
    const bool ll_mode_c =
        args.ll_ids && g_ll_recv != nullptr && (g_mega_cur_pos + query + 1) > kSelected;
    if (ctx.tid < kK3OptRows) {
      const int slot = slot_base + ctx.tid;
      int id = -1;
      if (ctx.tid < rows_per_split && slot < length) {
        const int candidate = canonical_dense ? slot
                                              : (ll_mode_c ? g_ll_ids[ctx.tid]
                                                           : ids_source[query * kSelected + slot]);
        if (candidate >= 0 && candidate < args.cache_rows && candidate <= g_mega_cur_pos + query) {
          id = candidate;
        }
      }
      ids_s[ctx.tid] = id;
    }
    ctx.sync();
  }
  // Whether this logical split has slots is uniform across the CTA.  Invalid
  // IDs are still masked by ids_s below, so a ballot and CTA broadcast are not
  // needed to preserve the all-invalid-row contract.
  const bool has_work =
      slot_base < length && args.mode != kModeNullOrchestration && args.mode != kModeNullStream;
  if (has_work) {
    k3_stage_pe(args, ids_s, pe_s, rows_per_split, ctx.tid);
    k3_cp_async_commit();  // group: PE gather, flies under the C4 poll
  }
  if (args.ids_ready && !producer_stages_qnope) {
    // arrivals, done well before the W_UK tail).  The W_UK half (kCntC4, 80)
    // is waited later, right before the q_nope staging, so the whole query-PE
    // path runs under the q-chain tail.
    ctx.boundary_wait(kCntC4Pe, ArrivalCounter::target(kArriveC4Pe, g_mega_epoch));
  }
  if (has_work) {
    // The QK-PE MMA waits for the PE gather + the 1.3 KB q_pe slice; the
    // 10 KB q_nope slice is staged after the kCntC4 wait below and lands
    // under the KV ring wait, awaited (with a consumer barrier) just before
    // the QK-KV MMA.  Under chain_entry_stage both query slabs are staged already.
    if (producer_stages_qnope) {
      k3_poll_query_records(ctx.arena, query, query_s, ctx.tid, g_mega_epoch);
    } else if (!chain_entry_stage) {
      k3_stage_query_pe(args, query, query_s, ctx.tid);
      k3_cp_async_commit();
    }
    k3_cp_async_wait_all();
  }
  ctx.sync();

  const uint32_t first_item = ctx.item;
  const uint32_t second_item = ctx.item + 1;
  const uint32_t first_slot = first_item & (smem::kRingSlots - 1);
  const uint32_t second_slot = second_item & (smem::kRingSlots - 1);

  {
    HmmaAcc accumulator[2]{};
    if (has_work && ctx.warp < row_tiles) {
      if (args.rank == 7) {
        // Five distinct rank-7 heads fit the MMA N=8 dimension. Transpose
        // QK to [16 keys, 8 heads], retaining the PE-then-latent FP32 sum.
        const int source_head = ctx.lane & 7;
        const int source_khalf = ((ctx.lane >> 3) & 1) * 8;
#pragma unroll
        for (int kk = 0; kk < kRope; kk += 16) {
          uint32_t fragment_k[4];
          hmma_load_a_16x16(pe_s + ctx.warp * 16 * kK3PeStride + kk, kK3PeStride, fragment_k);
          uint32_t fragment_q[2];
          hmma_ldsm_x2(fragment_q,
                       query_s + source_head * kK3QStride + kKvlora + kk + source_khalf);
          hmma(accumulator[0], fragment_k, fragment_q);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < kRope; kk += 16) {
          uint32_t fragment_q[4];
          hmma_load_a_16x16(query_s + kKvlora + kk, kK3QStride, fragment_q);
          // One ldmatrix.x4 covers both 8-row output halves of this 16-key
          // tile: same fragments, same MMA order, half the load issues.
          const int source_row = (ctx.lane & 7) + ((ctx.lane >> 4) << 3);
          const int source_khalf = ((ctx.lane >> 3) & 1) * 8;
          uint32_t fragment_k[2][2];
          hmma_ldsm_x4(fragment_k,
                       pe_s + (ctx.warp * 16 + source_row) * kK3PeStride + kk + source_khalf);
          hmma(accumulator[0], fragment_q, fragment_k[0]);
          hmma(accumulator[1], fragment_q, fragment_k[1]);
        }
      }
    }

    // In the default persistent path the producer has moved this C4 wait and
    // q_nope stage under the PE MMA.  Other modes keep the
    // consumer-side wait and cp.async stage.
    if (args.ids_ready && !producer_stages_qnope) {
      ctx.boundary_wait(kCntC4, ArrivalCounter::target(kArriveC4, g_mega_epoch));
    }
    if (has_work && !producer_stages_qnope && !chain_entry_stage) {
      k3_stage_query_nope(args, query, query_s, ctx.tid);
      k3_cp_async_commit();
    }

    const bool ring_live = args.mode != kModeNullOrchestration && args.mode != kModeNullStream;
    if (ring_live) {
      if (has_work || args.mode != kModeChain) {
        mbar_wait_parity(ctx.full_s + first_slot * 8, (first_item >> 2) & 1u);
        mbar_wait_parity(ctx.full_s + second_slot * 8, (second_item >> 2) & 1u);
      }
      ctx.item += 2;
    }

    // The producer-side mbar or the fallback cp.async group was left in flight
    // across the PE MMA and KV ring wait.  Complete it before QK-KV reads the
    // q_nope rows.  Only the QK worker warps consume those rows, so they wait
    // on the producer mbar directly; the remaining warps can park at the
    // post-QK consumer barrier below.
    if (has_work && !producer_stages_qnope) {
      k3_cp_async_wait_all();
      ctx.sync();
    }

    if (has_work && ctx.warp < row_tiles) {
      const int ring_half = ctx.warp >> 1;
      const int local_row_base = (ctx.warp & 1) * 16;
      const uint32_t slot = ring_half ? second_slot : first_slot;
      const __nv_bfloat16* kv_s =
          reinterpret_cast<const __nv_bfloat16*>(ctx.ring + slot * smem::kRingSlotBytes);
      if (args.rank == 7) {
        const int source_head = ctx.lane & 7;
        const int source_khalf = ((ctx.lane >> 3) & 1) * 8;
#pragma unroll
        for (int kk = 0; kk < kKvlora; kk += 16) {
          uint32_t fragment_k[4];
          hmma_load_a_16x16(kv_s + local_row_base * kK3OptRingRowStride + kk, kK3OptRingRowStride,
                            fragment_k);
          uint32_t fragment_q[2];
          hmma_ldsm_x2(fragment_q, query_s + source_head * kK3QStride + kk + source_khalf);
          hmma(accumulator[0], fragment_k, fragment_q);
        }

        // MMA rows are selected keys and columns are heads. Scatter only
        // the five computed heads into the head-major score tile.
        const int key_row = ctx.warp * 16 + (ctx.lane >> 2);
        const int head_pair = 2 * (ctx.lane & 3);
        if (head_pair < computed_heads) {
          score_s[head_pair * kK3ScoreStride + key_row] = accumulator[0].value[0];
          score_s[head_pair * kK3ScoreStride + key_row + 8] = accumulator[0].value[2];
        }
        if (head_pair + 1 < computed_heads) {
          score_s[(head_pair + 1) * kK3ScoreStride + key_row] = accumulator[0].value[1];
          score_s[(head_pair + 1) * kK3ScoreStride + key_row + 8] = accumulator[0].value[3];
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < kKvlora; kk += 16) {
          uint32_t fragment_q[4];
          hmma_load_a_16x16(query_s + kk, kK3QStride, fragment_q);
          // One ldmatrix.x4 covers both 8-row output halves of this 16-key
          // tile: same fragments, same MMA order, half the load issues.
          const int source_row = (ctx.lane & 7) + ((ctx.lane >> 4) << 3);
          const int source_khalf = ((ctx.lane >> 3) & 1) * 8;
          uint32_t fragment_k[2][2];
          hmma_ldsm_x4(fragment_k, kv_s + (local_row_base + source_row) * kK3OptRingRowStride + kk +
                                       source_khalf);
          hmma(accumulator[0], fragment_q, fragment_k[0]);
          hmma(accumulator[1], fragment_q, fragment_k[1]);
        }

        const int output_row = ctx.lane >> 2;
        const int output_pair = ctx.lane & 3;
#pragma unroll
        for (int output_half = 0; output_half < 2; ++output_half) {
          const int output_column = ctx.warp * 16 + output_half * 8 + 2 * output_pair;
          *reinterpret_cast<float2*>(score_s + output_row * kK3ScoreStride + output_column) =
              make_float2(accumulator[output_half].value[0], accumulator[output_half].value[1]);
          if (output_row + 8 < kHeads) {
            *reinterpret_cast<float2*>(score_s + (output_row + 8) * kK3ScoreStride +
                                       output_column) =
                make_float2(accumulator[output_half].value[2], accumulator[output_half].value[3]);
          }
        }
      }
    }
  }
  ctx.sync();

  // Retain FP32 local exponentials until the global normalization is known.
  // There is no BF16 rounding before the globally normalized PV operand.
  // The (max, sum) statistics publish straight to this split's globally
  // coherent head-major record: no shared staging slab, no copy pass.
  const float c = k3_exp2_scale(args.softmax_scale);
  float* stats_g = wsp<float>(local, arena::kScores);
  float* stats_split_g = stats_g + (query * kK3HeadsPadded * kK3OptSplits + split) * 2;
  if (has_work) {
    if (ctx.warp < 2) {
      // Keep the half-warp reduction tree for real heads 0 and 1.
      // The upper half-warps would compute unused padded heads 8 and 9.
      if (args.rank != 7 || ctx.lane < 16) {
        k3_softmax_head_pair(query, split, ctx.warp, c, ids_s, score_s, probability_s,
                             stats_split_g, ctx.lane);
      }
    } else if (ctx.warp < 8 && ctx.warp < computed_heads) {
      k3_softmax_head_adjacent(query, split, ctx.warp, c, ids_s, score_s, probability_s,
                               stats_split_g, ctx.lane);
    }
    if (ctx.tid >= computed_heads && ctx.tid < (args.rank == 7 ? computed_heads : kK3HeadsPadded)) {
      *reinterpret_cast<float2*>(stats_split_g + ctx.tid * (2 * kK3OptSplits)) =
          make_float2(k3_negative_infinity(), 0.0f);
    }
  } else if (ctx.tid < (args.rank == 7 ? computed_heads : kK3HeadsPadded)) {
    *reinterpret_cast<float2*>(stats_split_g + ctx.tid * (2 * kK3OptSplits)) =
        make_float2(k3_negative_infinity(), 0.0f);
  }
  // Preload this warp's first 16-key V fragments from the resident ring rows
  // while the statistics rendezvous is in flight: the loads depend only on
  // local shared memory, so they retire under the rendezvous instead of
  // extending the post-rendezvous PV critical path.
  uint32_t preload_v[4][4];
  ctx.sync();
  // Arrive on the split statistics first: the V fragment preload below only reads this CTA's KV
  // ring slot.
  if (ctx.tid == 0) {
    red_release_gpu_add(ctx.counters + cnt_word(kCntK3Stats + query), 1u);
  }
  if (has_work) {
    const __nv_bfloat16* kv_first =
        reinterpret_cast<const __nv_bfloat16*>(ctx.ring + first_slot * smem::kRingSlotBytes);
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      hmma_load_a_16x16_trans(kv_first + (ctx.warp + tile * 8) * 16, kK3OptRingRowStride,
                              preload_v[tile]);
    }
  }
  // One poller per CTA keeps a single coalesced request stream on the
  // counter line (more pollers measurably delay the releasing reductions);
  // the CTA barrier broadcasts the acquire.  Dead splits, which publish
  // (-inf, 0) but read no statistics, skip the poll and rendezvous at the
  // pre-PV consumer barrier below.
  if (has_work) {
    if (ctx.tid == 0) {
      const uint32_t target = ArrivalCounter::target(kK3OptSplits, g_mega_epoch);
      while (!word_reached(ctx.counters + cnt_word(kCntK3Stats + query), target)) {
        spin_backoff();
      }
      fence_acquire_gpu();
    }
    ctx.sync();
  }
  if (has_work) {
    // Five warps normalize all ten heads in one pass; rank 7 uses the lower
    // five half-warps. The half-warp helper preserves the full-warp reduction
    // association while doubling row ownership per lane.
    const int head_slot = ctx.warp * 2 + (ctx.lane >> 4);
    if (head_slot < computed_heads) {
      k3_global_probability_head_halfwarp(query, split, head_slot, c, ids_s, score_s, stats_g,
                                          probability_s, ctx.lane);
    }
    // The rank-7 PV tile has eight physical columns but only five computed
    // heads. Initialize its three unused columns inside this invocation.
    if (args.rank == 7 && ctx.tid < 3 * (kK3OptRows / 2)) {
      const int head = 5 + ctx.tid / (kK3OptRows / 2);
      const int row = 2 * (ctx.tid % (kK3OptRows / 2));
      *reinterpret_cast<uint32_t*>(probability_s + head * kK3ProbStride + row) = 0u;
    }
  }
  ctx.sync();

  if (has_work) {
    HmmaAcc output[4][2]{};
#pragma unroll
    for (int kk = 0; kk < kK3OptRows; kk += 16) {
      if (kk >= rows_per_split) continue;
      const int ring_half = kk / kK3OptHalfRows;
      const int local_kk = kk & (kK3OptHalfRows - 1);
      const uint32_t slot = ring_half ? second_slot : first_slot;
      const __nv_bfloat16* kv_s =
          reinterpret_cast<const __nv_bfloat16*>(ctx.ring + slot * smem::kRingSlotBytes);
      uint32_t fragment_p[2][2];
      const int source_head = ctx.lane & 7;
      const int source_khalf = ((ctx.lane >> 3) & 1) * 8;
      const int address_head_half = ctx.lane >> 4;
      if (args.rank == 7) {
        hmma_ldsm_x2(fragment_p[0],
                     probability_s + source_head * kK3ProbStride + kk + source_khalf);
      } else {
        hmma_ldsm_x4(fragment_p, probability_s +
                                     (address_head_half * 8 + source_head) * kK3ProbStride + kk +
                                     source_khalf);
      }
      uint32_t fragment_v[4][4];
      if (kk == 0) {
        // The first 16-key V fragments were preloaded under the statistics
        // rendezvous above; identical loads, identical values.
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
          for (int reg = 0; reg < 4; ++reg) {
            fragment_v[tile][reg] = preload_v[tile][reg];
          }
        }
      } else {
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
          const int dimension = (ctx.warp + tile * 8) * 16;
          hmma_load_a_16x16_trans(kv_s + local_kk * kK3OptRingRowStride + dimension,
                                  kK3OptRingRowStride, fragment_v[tile]);
        }
      }
#pragma unroll
      for (int head_half = 0; head_half < 2; ++head_half) {
        if (head_half * 8 >= computed_heads) continue;
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
          hmma(output[tile][head_half], fragment_v[tile], fragment_p[head_half]);
        }
      }
    }

    // V^T P^T produces [dimension, head].  Lanes sharing a head write eight
    // adjacent dimensions, keeping scalar stores sector-coalesced.  The FP32
    // accumulators publish straight to the globally coherent partial record
    // (same values, same addresses the combine reads): no shared staging
    // slab, no async-proxy round trip, and no serialized bulk-store wait sit
    // between the PV and the partial release below.
    const int dimension_row = ctx.lane >> 2;
    const int head_pair = 2 * (ctx.lane & 3);
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int head_half = 0; head_half < 2; ++head_half) {
        if (head_half * 8 >= computed_heads) continue;
        const int head0 = head_half * 8 + head_pair;
        const int dimension0 = (ctx.warp + tile * 8) * 16 + dimension_row;
        const HmmaAcc& value = output[tile][head_half];
        if (head0 < computed_heads) {
          float* base = partial + head0 * kKvlora + dimension0;
          k3_store_partial_cg(base, value.value[0], base + 8, value.value[2]);
        }
        if (head0 + 1 < computed_heads) {
          float* base = partial + (head0 + 1) * kKvlora + dimension0;
          k3_store_partial_cg(base, value.value[1], base + 8, value.value[3]);
        }
      }
    }
  }
  ctx.sync();

  // In kModeChain the producer would exit after these two
  // items, so its empty releases and the barrier that joins those releases
  // would be dead.  Persistent mode keeps both because its producer reuses the
  // slots for K4/K5.
  const bool ring_reused = args.mode != kModeChain && args.mode != kModeNullOrchestration &&
                           args.mode != kModeNullStream;
  if (ring_reused && ctx.tid == 0) {
    // The PV-store barrier above has joined all 256 consumers.  One counted
    // arrival now releases each ring slot on their behalf, avoiding 512
    // per-thread mbarrier instructions and the otherwise redundant join.
    mbar_arrive_count(ctx.empty_s + first_slot * 8, kOpThreads);
    mbar_arrive_count(ctx.empty_s + second_slot * 8, kOpThreads);
  }
  // The PV-store barrier above ordered every thread's coherent partial store
  // before this release, so the publication is one fire-and-forget reduction:
  // no staging drain sits between the last PV store and the combine's poll.
  if (ctx.tid == 7 * 32) {
    red_release_gpu_add(ctx.counters + cnt_word(kCntK3Partials + query), 1u);
  }
}

// One body for every cache size.  There is no separate small-cache variant: the deferred
// normalization in the combine covers 1..2048 selected rows alike.
static __device__ void op_k3_attention(const MegaArgs& args, OpCtx& ctx) {
  op_k3_attention_split(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

// K4 CTA c = 8h + 2q + r owns head h, query q and W_UV rows [256h + 128r, +128).
// It waits only for query q's 32 split partials, folds all 512 dimensions of
// (q, h) itself in the contract order ((((0 + p0) + .. + p7) per split group,
// then g0 + g1 + g2 + g3), publishes its half of post_o and runs the four
// 32-row W_UV pages of its rows on that BF16 vector.  No per-head post_o
// rendezvous is needed; the head counter still receives eight arrivals per
// epoch, which keeps the 20-arrival K2 + K3 head cycle.
static __device__ void op_k4_fused_combine_uv(const MegaArgs& args, OpCtx& ctx) {
  const bool null_math = args.mode == kModeNullOrchestration;
  const int head = ctx.cta >> 3;
  const int query = (ctx.cta & 7) >> 1;
  const int row_half = ctx.cta & 1;
  const bool active = (args.rank != 7 || head < 4) && !null_math;
  const int source_head = args.rank == 7 && head >= 5 ? 4 : head;
  constexpr int kRows = kK4FusedPages * 32;
  // With dense_ids (as launched) the length is the public cur_pos scalar plus the query offset and
  // the lengths array is never read, so it is formed before the partials wait; with dense_ids == 0
  // this read would have to stay after the acquire below.
  const bool canonical_dense = g_mega_cur_pos + query + 1 <= kSelected;
  const int length = min(kSelected, (canonical_dense || args.dense_ids)
                                        ? g_mega_cur_pos + query + 1
                                        : k3_lengths_source(args, args.peers[args.rank])[query]);

  if (ctx.warp == 0) {
    const uint32_t target = ArrivalCounter::target(kK3OptSplits, g_mega_epoch);
    const uint32_t* partial_counter = ctx.counters + cnt_word(kCntK3Partials + query);
    while (!word_reached(partial_counter, target)) spin_backoff();
    asm volatile("fence.acquire.gpu;" ::: "memory");
  }
  ctx.sync();
  const int live_splits = (args.mode == kModeNullOrchestration || args.mode == kModeNullStream)
                              ? 0
                              : k3_live_splits(length);
  const float* partial_g = wsp<float>(ctx.arena, arena::kOAcc) +
                           static_cast<size_t>(query) * kK3OptSplits * kK3PartialStrideFloats +
                           source_head * kKvlora + 2 * ctx.tid;
  float2 split_values[kK3OptSplits];
#pragma unroll
  for (int split = 0; split < kK3OptSplits; ++split) {
    if (split < live_splits) {
      float2 v;
      asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];"
                   : "=f"(v.x), "=f"(v.y)
                   : "l"(partial_g + static_cast<size_t>(split) * kK3PartialStrideFloats)
                   : "memory");
      split_values[split] = v;
    } else {
      split_values[split] = make_float2(0.0f, 0.0f);
    }
  }
  float2 group_total[4];
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    float2 value = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int inner = 0; inner < kK3OptSplits / 4; ++inner) {
      value = k3_add_f32x2(value, split_values[group * (kK3OptSplits / 4) + inner]);
    }
    group_total[group] = value;
  }
  float2 total = group_total[0];
#pragma unroll
  for (int group = 1; group < 4; ++group) total = k3_add_f32x2(total, group_total[group]);
  const uint32_t post_bits = pack_bf16_pair(total.x, total.y);
  if ((ctx.tid >> 7) == row_half) {
    *reinterpret_cast<uint32_t*>(args.post_o + (query * kHeads + head) * kKvlora + 2 * ctx.tid) =
        post_bits;
  }

  float* partial = ctx.scratch<float>(smem::kGemmPartial);
  __nv_bfloat16* act = ctx.scratch<__nv_bfloat16>(smem::kActivation);
  if (active) {
#pragma unroll
    for (int token = 0; token < kQlen; ++token) {
      *reinterpret_cast<uint32_t*>(act + token * smem::kUvActStride + 2 * ctx.tid) = post_bits;
    }
    __syncwarp(0xffffffffu);
    const uint32_t a_lane =
        static_cast<uint32_t>(ctx.warp) * 0x400u + static_cast<uint32_t>(ctx.lane) * 8u;
#pragma unroll 1
    for (int page = 0; page < kK4FusedPages; ++page) {
      const uint32_t slot = ctx.acquire_page();
      HmmaAcc acc[2] = {{{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}};
      gemv_page_steps<2, 4>(slot + a_lane, 0x2000u, 0x100u, smem_u32(act), smem::kUvActStride * 2,
                            ctx.warp * 64, acc);
      const float scale = lds_f32(slot + 0x4000u + static_cast<uint32_t>(ctx.warp >> 1) * 4u);
      ctx.release_page();
      float out[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
      fma_scale_into(out[0], scale, acc[0]);
      fma_scale_into(out[1], scale, acc[1]);
      if ((ctx.lane & 3) == 0) {
        const int g = ctx.lane >> 2;
        float* unit = partial + ctx.warp * kRows + page * 32 + g;
        unit[0] = out[0][0];
        unit[8] = out[0][2];
        unit[16] = out[1][0];
        unit[24] = out[1][2];
      }
    }
  }
  ctx.sync();
  uint32_t* out_stage = reinterpret_cast<uint32_t*>(act);
  if (ctx.tid < kRows / 2) {
    F32Pair value{};
    if (active) {
      uint64_t v[kOpWarps];
      const uint32_t base = smem_u32(partial + 2 * ctx.tid);
#pragma unroll
      for (int u = 0; u < kOpWarps; ++u) {
        asm volatile("ld.shared.u64 %0, [%1];"
                     : "=l"(v[u])
                     : "r"(base + static_cast<uint32_t>(u * kRows * sizeof(float))));
      }
#pragma unroll
      for (int u = 0; u < kOpWarps; ++u) {
        asm volatile("add.rn.f32x2 %0, %0, %1;" : "+l"(value.bits) : "l"(v[u]));
      }
    }
    out_stage[ctx.tid] = pack_bf16_pair(value.value[0], value.value[1]);
    if (args.mode == kModePersistent) {
      const uint32_t bits = pack_bf16_pair(value.value[0], value.value[1]);
      const uint32_t next = __shfl_down_sync(0xffffffffu, bits, 1);
      if ((ctx.tid & 1) == 0) {
        const int element = query * kUvRows + head * kVdim + row_half * kRows + 2 * ctx.tid;
        rec_store_b128(ctx.arena + kRecUv + size_t(element / 4) * 16, bits, next, g_mega_epoch);
        *reinterpret_cast<uint2*>(args.post_uv + element) = make_uint2(bits, next);
      }
    } else {
      asm volatile("barrier.cta.sync 5, %0;" ::"n"(kRows / 2) : "memory");
    }
  }
  if (args.mode == kModePersistent) {
    if (ctx.tid == 0) {
      red_relaxed_gpu_add(ctx.counters + cnt_word(kCntC7), 1u);
      red_relaxed_gpu_add(ctx.counters + cnt_word(kCntK2Head + head), 1u);
    }
  } else if (ctx.tid == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bulk_copy_s2g(reinterpret_cast<uint8_t*>(args.post_uv + query * kUvRows + head * kVdim +
                                             row_half * kRows),
                  smem_u32(out_stage), kRows * uint32_t(sizeof(__nv_bfloat16)));
    bulk_copy_s2g_wait();
    red_release_gpu_add(ctx.counters + cnt_word(kCntC7), 1u);
    red_relaxed_gpu_add(ctx.counters + cnt_word(kCntK2Head + head), 1u);
  }
}

// K4 owns both the one-to-one combine role and its W_UV tile.
// All eight roles of a head publish before any of its eight W_UV CTAs stages
// post_o.
static __device__ void op_k4_attention_combine_uv(const MegaArgs& args, OpCtx& ctx) {
  op_k4_fused_combine_uv(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

// ----  push collective helpers ------------------------------------------------------------------
// The i-th (0..5) other attention rank of `rank`, in rank order 1..7.
__device__ __forceinline__ int k5_peer_rank(int rank, int i) {
  const int r = 1 + i;
  return r < rank ? r : r + 1;
}
// Byte offset of record k (0..7) of token q of CTA `cta` in source slot `slot` of an inbox.
__device__ __forceinline__ size_t k5_rx_offset(uint32_t epoch, int slot, int cta, int q, int k) {
  return arena::kHiddenRx + size_t(epoch & 1u) * arena::kRxParityBytes +
         size_t(slot) * arena::kRxSlotBytes + size_t(cta) * arena::kRxCtaBytes +
         size_t(q) * arena::kRxRecordsPerToken * arena::kRxRecordBytes +
         size_t(k) * arena::kRxRecordBytes;
}

__device__ __forceinline__ uint64_t k5_add_bf16_pair(uint64_t total, uint32_t term) {
  return add_f32x2(total, pack_f32x2(bf16_pair_low(term), bf16_pair_high(term)));
}

__device__ __forceinline__ void k5_reduce_direct_record(const MegaArgs& args, const OpCtx& ctx,
                                                        int token, int rec_k, int row_base,
                                                        uint32_t hidden_bits, uint32_t word1,
                                                        uint32_t word2, bool null_math) {
  const int pair_base = token * kHidden + row_base + rec_k * arena::kRxRecordElems;
  uint32_t x_pairs[3];
#pragma unroll
  for (int pair = 0; pair < 3; ++pair) {
    x_pairs[pair] =
        null_math ? 0u : *reinterpret_cast<const uint32_t*>(args.x + pair_base + 2 * pair);
  }
  uint64_t peer_lo[kWorld - 2];
  uint64_t peer_hi[kWorld - 2];
#pragma unroll
  for (int src_i = 0; src_i < kWorld - 2; ++src_i) {
    const int src = k5_peer_rank(args.rank, src_i);
    const uint8_t* record = ctx.arena + k5_rx_offset(g_mega_epoch, src - 1, ctx.cta, token, rec_k);
    ld_relaxed_sys_b128(record, peer_lo[src_i], peer_hi[src_i]);
  }
  for (;;) {
    uint32_t stale = 0u;
#pragma unroll
    for (int src_i = 0; src_i < kWorld - 2; ++src_i) {
      if (static_cast<uint32_t>(peer_hi[src_i] >> 32) != g_mega_epoch) {
        stale |= 1u << src_i;
      }
    }
    if (stale == 0u) break;
#pragma unroll
    for (int src_i = 0; src_i < kWorld - 2; ++src_i) {
      if (stale & (1u << src_i)) {
        const int src = k5_peer_rank(args.rank, src_i);
        const uint8_t* record =
            ctx.arena + k5_rx_offset(g_mega_epoch, src - 1, ctx.cta, token, rec_k);
        ld_relaxed_sys_b128(record, peer_lo[src_i], peer_hi[src_i]);
      }
    }
  }

  if (null_math) {
#pragma unroll
    for (int pair = 0; pair < 3; ++pair) {
      *reinterpret_cast<uint32_t*>(args.endpoint + pair_base + 2 * pair) = 0u;
    }
    return;
  }

  uint64_t total[3] = {0ull, 0ull, 0ull};
#pragma unroll
  for (int src_i = 0; src_i < kWorld - 2; ++src_i) {
    if (args.rank == src_i + 1) {
      total[0] = k5_add_bf16_pair(total[0], hidden_bits);
      total[1] = k5_add_bf16_pair(total[1], word1);
      total[2] = k5_add_bf16_pair(total[2], word2);
    }
    total[0] = k5_add_bf16_pair(total[0], static_cast<uint32_t>(peer_lo[src_i]));
    total[1] = k5_add_bf16_pair(total[1], static_cast<uint32_t>(peer_lo[src_i] >> 32));
    total[2] = k5_add_bf16_pair(total[2], static_cast<uint32_t>(peer_hi[src_i]));
  }
  if (args.rank == kWorld - 1) {
    total[0] = k5_add_bf16_pair(total[0], hidden_bits);
    total[1] = k5_add_bf16_pair(total[1], word1);
    total[2] = k5_add_bf16_pair(total[2], word2);
  }
#pragma unroll
  for (int pair = 0; pair < 3; ++pair) {
    const F32Pair sum{total[pair]};
    const uint32_t x_bits = x_pairs[pair];
    *reinterpret_cast<uint32_t*>(args.endpoint + pair_base + 2 * pair) =
        pack_bf16_pair(__fadd_rn(bf16_pair_low(x_bits), sum.value[0]),
                       __fadd_rn(bf16_pair_high(x_bits), sum.value[1]));
  }
}

// o_proj GEMV of this CTA's 48-row tile into local_hidden (bf16 [4][6144]).  With
// no_collective == kCollectiveFull the epilogue also pushes this CTA's collective records to
// the peers (push_records) and, for kOpK5, reduces the matching peer records in registers
// (k5_reduce_direct_record).  endpoint_local != nullptr (no-collective kCollectiveLocal):
// the epilogue also writes endpoint[token][row] = BF16(x + BF16(local_hidden)) for the
// CTA's rows (zeros in null_orchestration, like the collective phase does).  There is no
// trailing consumer barrier (see the note at the end of the function).
static __device__ void op_k5_o_proj_gemv(const MegaArgs& args, OpCtx& ctx,
                                         __nv_bfloat16* local_hidden,
                                         __nv_bfloat16* endpoint_local = nullptr) {
  const bool null_math = args.mode == kModeNullOrchestration;
  float* partial = ctx.scratch<float>(smem::kGemmPartial);             // f32 [8][4][48]
  __nv_bfloat16* act = ctx.scratch<__nv_bfloat16>(smem::kActivation);  // bf16 [4][kOProjActStride]
  constexpr int kOProjCols = kHeads * kVdim;                           // 2560
  constexpr int kRows = kOProjRowsPerCta;                              // 48
  constexpr int kPages = 5;
  const int row_base = kRows * ctx.cta;
  // Rank 0's exit credit for this CTA (arena::kExitCredit), read before the GEMV so the rank-0 push
  // normally finds it satisfied. Every warp reads it: a warp-dependent guard here costs the GEMV's
  // warp-collective ops divergence checks.  Same 16 B access as rank 0's store (single-copy
  // atomic).
  const bool exit_credit_gate = ctx.op == kOpK5 && args.mode == kModePersistent;
  const uint8_t* exit_credit =
      ctx.arena + arena::kExitCredit + size_t(ctx.cta) * arena::kExitCreditLineBytes;
  uint64_t credit_lo = 0u, credit_hi = 0u;
  if (exit_credit_gate) ld_relaxed_sys_b128(exit_credit, credit_lo, credit_hi);
  (void)credit_hi;  // the four words are equal

  if (!null_math) {
    // Warp w stages exactly the five 64-column slices its MMAs consume (warp fence only).
    if (ctx.op == kOpK5 && args.mode == kModePersistent) {
      stage_o_proj_activation_records(act, smem::kOProjActStride, ctx.arena + kRecUv, ctx.warp,
                                      ctx.lane, g_mega_epoch);
    } else {
      stage_o_proj_activation_kowned(act, smem::kOProjActStride, args.post_uv, kOProjCols, ctx.warp,
                                     ctx.lane);
    }
    // Page layout: row tile r at +r*0x2000, k16 tile t at +t*0x100,
    // scale (r, chunk j) at +0x6000 + (4r + j)*4.  Warp w: tiles 4w..4w+3 (chunk w>>1).
    const uint32_t act_s = smem_u32(act);
    const uint32_t a_lane =
        static_cast<uint32_t>(ctx.warp) * 0x400u + static_cast<uint32_t>(ctx.lane) * 8u;
    float run[3][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll 1
    for (int page = 0; page < kPages; ++page) {
      const uint32_t slot = ctx.acquire_page();
      HmmaAcc acc[3] = {
          {{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}};
      gemv_page_steps<3, 4>(slot + a_lane, 0x2000u, 0x100u, act_s, smem::kOProjActStride * 2,
                            page * 512 + ctx.warp * 64, acc);
      float scale[3];
#pragma unroll
      for (int rt = 0; rt < 3; ++rt) {
        scale[rt] = lds_f32(slot + 0x6000u + static_cast<uint32_t>(4 * rt + (ctx.warp >> 1)) * 4u);
      }
      // K5 is the final ring op and has five pages over four slots.  Only page
      // zero's slot is reused (by page four); pages one through four are never
      // overwritten before kernel exit.  Retain the logical item progression
      // without issuing dead empty[] arrivals for those terminal slots.
      if (page == 0) {
        ctx.release_page();
      } else {
        ++ctx.item;
      }
#pragma unroll
      for (int rt = 0; rt < 3; ++rt) fma_scale_into(run[rt], scale[rt], acc[rt]);
    }
#pragma unroll
    for (int rt = 0; rt < 3; ++rt) {
      store_partial_tile(partial + ctx.warp * (kQlen * kRows), kRows, rt, run[rt]);
    }
  }
  ctx.sync();
  // PDL: the dependent MoE may launch now (it waits for this grid's completion before touching
  // per-call memory)
  if (ctx.op == kOpK5 && args.mode == kModePersistent)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // Paired epilogue on a (warp = token, lane = pair) map: add.rn.f32x2 folds the 8 warp
  // partials of both rows in the same 0 + p0 + ... + p7 order as the scalar fold,
  // cvt.rn.bf16x2 rounds.  The warp layout keeps each 3-pair record inside one warp,
  // so the collective push assembles records from two shfl.down and issues its six
  // sys-scope stores here -- no shared staging read and no extra consumer barrier on
  // the push path.  The rounded pairs still land in the dead activation scratch for
  // the collective reduce's own term.
  uint32_t* hidden_stage =
      reinterpret_cast<uint32_t*>(act);  // u32 [4][kRows/2], dead after the GEMV
  const bool push_records =
      args.no_collective == kCollectiveFull && (ctx.op == kOpK5 || ctx.op == kOpK5OProj);
  if (ctx.warp < kQlen) {
    const int token = ctx.warp;
    const int row = ctx.lane * 2;
    const bool pair_lane = ctx.lane < kRows / 2;
    uint32_t hidden_bits = 0u;
    if (pair_lane) {
      F32Pair value{};
      if (!null_math) {
        value = reduce_partial_pair<kOpWarps, kRows>(partial, token, row);
      }
      hidden_bits = pack_bf16_pair(value.value[0], value.value[1]);
      const int index = token * kHidden + row_base + row;
      // The fused persistent collective consumes its own term directly from
      // hidden_bits and pushes peer records from the same register value.  Its
      // arena mirror is needed only by the single-phase K5 ops (kOpK5OProj / kOpK5Collective).
      if (local_hidden != nullptr) {
        *reinterpret_cast<uint32_t*>(local_hidden + index) = hidden_bits;
      }
      // Fused K5 retains the own record in this token warp; only the single-phase K5 ops
      // need the shared handoff.
      if (ctx.op != kOpK5) {
        hidden_stage[token * (kRows / 2) + ctx.lane] = hidden_bits;
      }
      if (endpoint_local != nullptr) {
        // endpoint = BF16(x + h).  null_orchestration writes zeros like the collective phase.
        uint32_t endpoint_bits = 0u;
        if (!null_math) {
          const uint32_t x_bits = *reinterpret_cast<const uint32_t*>(args.x + index);
          const float x0 = __uint_as_float(x_bits << 16);
          const float x1 = __uint_as_float(x_bits & 0xffff0000u);
          const float h0 = __uint_as_float(hidden_bits << 16);
          const float h1 = __uint_as_float(hidden_bits & 0xffff0000u);
          endpoint_bits = pack_bf16_pair(__fadd_rn(x0, h0), __fadd_rn(x1, h1));
        }
        *reinterpret_cast<uint32_t*>(endpoint_local + index) = endpoint_bits;
      }
    }

    if (pair_lane && push_records) {
      constexpr uint32_t kPairLanes = (1u << (kRows / 2)) - 1u;  // lanes 0..23
      const uint32_t word1 = __shfl_down_sync(kPairLanes, hidden_bits, 1);
      const uint32_t word2 = __shfl_down_sync(kPairLanes, hidden_bits, 2);
      const uint32_t word3 = __shfl_down_sync(kPairLanes, hidden_bits, 3);

      // The inbox store must not queue behind the arena records' drain.  Its inbox block for (q =
      // token, source slot = rank-1, parity, cta) is 128 B = 8 lines x 16 B: lines 0..5 = the 48
      // bf16 of this CTA's rows (line l = pair lanes 4l..4l+3 = elements 8l..8l+7), lines 6,7 =
      // {E,E,E,E} with E = low 32 bits of the launch epoch; the receiver polls line 7
      // dwords 2,3 == E and then reads lines 0..5.  Eight lanes (0,4,8,12,16,20 -> lines 0..5; 1,2
      // -> lines 6,7) issue ONE predicated st.relaxed.sys.b128 so the whole 128 B line leaves in
      // one warp store.

      const int rec_k = ctx.lane / 3;
      if (ctx.lane - rec_k * 3 == 0) {  // eight record leaders per token warp
        for (int dst_i = 0; dst_i < kWorld - 2; ++dst_i) {
          const int dst = k5_peer_rank(args.rank, dst_i);
          const int slot = (args.single_rank ? dst : args.rank) - 1;
          st_relaxed_sys_b128(
              args.peers[dst] + k5_rx_offset(g_mega_epoch, slot, ctx.cta, token, rec_k),
              hidden_bits, word1, word2, g_mega_epoch);
        }
        if (args.rank0_reduce &&
            !args.single_rank) {  // replace mode: rank 0 reduces the seven slices itself
          // This slot last held epoch - 2; overwrite it only after rank 0 has consumed that epoch.
          while (exit_credit_gate &&
                 static_cast<int32_t>(static_cast<uint32_t>(credit_lo) - (g_mega_epoch - 2u)) < 0) {
            __nanosleep(32);
            ld_relaxed_sys_b128(exit_credit, credit_lo, credit_hi);
          }
          st_relaxed_sys_b128(
              args.peers[0] + k5_rx_offset(g_mega_epoch, args.rank - 1, ctx.cta, token, rec_k),
              hidden_bits, word1, word2, g_mega_epoch);
        }
      }
      // Keep the eight-leader sequential-destination issue shape: spreading the 48 stores over 24
      // lanes x 2 wide instructions bursts the NVLink egress queues and delays record
      // publication.  No fence drains the system-scope record stores here: the records are
      // epoch-tagged relaxed stores polled by the receivers, and a sys-scope fence would only
      // shape the drain timing (~1.2-1.5 us of exit time in a CUDA graph).
      // Physical K5 record owners retain the three local pairs and reduce the
      // matching six peer records directly in registers.  The single-phase K5 ops keep the
      // shared receive/reducer implementation.
      if (ctx.op == kOpK5 && ctx.lane - rec_k * 3 == 0) {
        k5_reduce_direct_record(args, ctx, token, rec_k, row_base, hidden_bits, word1, word2,
                                null_math);
      }
    }
  }
  // The persistent kernel proceeds straight to the collective phase:
  // its receive region lives in the dead activation scratch past the staged
  // pairs, so no consumer barrier is needed between the GEMV epilogue and the
  // inbox polls -- the collective's own pre-reduce barrier orders the staged
  // pairs for their cross-thread read.
}

constexpr int kK5RxThreads =
    (kWorld - 2) * kQlen * arena::kRxRecordsPerToken;  // 6 peers x 4 tokens x 8 records = 192
constexpr int kK5RxTokenWords =
    arena::kRxRecordsPerToken *
    (arena::kRxRecordBytes / sizeof(uint32_t));  // 32 u32: eight aligned wire records
static_assert(kK5RxThreads <= kOpThreads && kQlen * kOProjRowsPerCta <= kOpThreads,
              "K5 push / reduce thread maps");
static_assert((kWorld - 2) * kQlen * kK5RxTokenWords * int(sizeof(uint32_t)) <=
                  smem::kGemmPartialBytes,
              "received slices fit the gemm partial scratch (free after the GEMV)");

// Phase A: o_proj GEMV into arena::kLocalHidden; the GEMV epilogue itself assembles the
// records from shfl.down pairs and pushes this CTA's slice to the six peers.  Ends with the
// 192 record stores issued; no counter, no flag, no trailing consumer barrier.
static __device__ void op_k5_o_proj_phase(const MegaArgs& args, OpCtx& ctx) {
  const bool preserve_local_hidden = ctx.op != kOpK5 || args.mode != kModePersistent ||
                                     args.single_rank || args.ll_recv != nullptr ||
                                     args.rank0_reduce;
  __nv_bfloat16* local_hidden =
      preserve_local_hidden ? wsp<__nv_bfloat16>(ctx.arena, arena::kLocalHidden) : nullptr;
  op_k5_o_proj_gemv(args, ctx, local_hidden);
}

// Phase B: receive the six peer slices of this CTA's 48 rows from the own inbox, rank-ordered fp32
// sum with the own local_hidden term, BF16 round, residual, endpoint.  No done token.
static __device__ void op_k5_collective_phase(const MegaArgs& args, OpCtx& ctx) {
  const bool null_math = args.mode == kModeNullOrchestration;
  // The fused K5 already reduced and published each record in its token warp.  Only the
  // independent rank-0 pacing role remains here; the single-phase K5 ops use the path below.
  if (ctx.op == kOpK5) {
    return;
  }

  // The receive slab lives in the dead activation scratch past the staged own
  // pairs (not in kGemmPartial): the epilogue's partial reads and the pollers'
  // rx stores are then disjoint, so no barrier is needed between the GEMV
  // epilogue and the inbox polls.
  uint32_t* rx_s =
      ctx.scratch<uint32_t>(smem::kActivation) + 128;  // [6 peers][4 tokens][32 u32] at +512 B
  uint32_t* hidden_stage =
      ctx.scratch<uint32_t>(smem::kActivation);  // own rounded pairs (fallback reload)

  // A single-phase K5 op starts in a fresh CTA and finds its own term only in
  // arena::kLocalHidden.  Rebuild the shared handoff before the receive/reducer
  // barriers; the fused K5 returned above.
  const bool stage_own_from_arena = ctx.op != kOpK5;
  if (stage_own_from_arena && ctx.warp < kQlen && ctx.lane < kOProjRowsPerCta / 2) {
    const int own_pair = ctx.warp * kHidden + kOProjRowsPerCta * ctx.cta + 2 * ctx.lane;
    hidden_stage[ctx.warp * (kOProjRowsPerCta / 2) + ctx.lane] = *reinterpret_cast<const uint32_t*>(
        wsp<const __nv_bfloat16>(ctx.arena, arena::kLocalHidden) + own_pair);
  }

  // Warps 4..6 remain the 96 adjacent-pair reducers. Six receive warps map one-to-one to the six
  // peer slabs: 4,5,6 handle sources 0,1,2 while sender warps 0,1,2 handle sources 3,4,5 after
  // issuing their outbound records.  Warp 3 stays push-only and warp 7 remains free for rank-0
  // pacing.
  constexpr int kPairs = kOProjRowsPerCta / 2;
  constexpr int kRxWarpBase = kQlen;
  constexpr int kReduceWarps = 3;
  const bool pair_owner = ctx.warp >= kRxWarpBase && ctx.warp < kRxWarpBase + kReduceWarps;
  const int pair_thread = ctx.tid - kRxWarpBase * 32;
  const int pair_q = pair_thread / kPairs;
  const int pair_p = pair_thread - pair_q * kPairs;
  const int pair_index = pair_q * kHidden + kOProjRowsPerCta * ctx.cta + 2 * pair_p;
  uint32_t x_bits = 0u;
  if (pair_owner && !null_math) {
    x_bits = *reinterpret_cast<const uint32_t*>(args.x + pair_index);
  }

  // Each receive warp owns one source slab.  First only the final record of
  // each token polls: all eight records of that token are co-issued by one
  // sender-warp instruction, so its arrival strongly predicts that the other
  // seven are resident.  After the warp join every lane loads its own record;
  // the per-record epoch fallback keeps correctness independent of line
  // visibility without a second serial pass over the sources.
  const bool receive_owner =
      ctx.warp < 3 || (ctx.warp >= kRxWarpBase && ctx.warp < kRxWarpBase + kReduceWarps);
  if (receive_owner) {
    const int src_i = ctx.warp < 3 ? ctx.warp + kReduceWarps : ctx.warp - kRxWarpBase;
    const int rec = ctx.lane;
    const int q = rec >> 3;
    const int k = rec & 7;
    const int src = k5_peer_rank(args.rank, src_i);
    const uint8_t* record = ctx.arena + k5_rx_offset(g_mega_epoch, src - 1, ctx.cta, q, k);
    uint64_t lo = 0, hi = 0;
    if (k == arena::kRxRecordsPerToken - 1) {
      ld_relaxed_sys_b128(record, lo, hi);
      while (static_cast<uint32_t>(hi >> 32) != g_mega_epoch) {
        __nanosleep(32);
        ld_relaxed_sys_b128(record, lo, hi);
      }
    }
    __syncwarp();
    if (k != arena::kRxRecordsPerToken - 1) {
      ld_relaxed_sys_b128(record, lo, hi);
      while (static_cast<uint32_t>(hi >> 32) != g_mega_epoch) {
        __nanosleep(32);
        ld_relaxed_sys_b128(record, lo, hi);
      }
    }
    uint32_t* out = rx_s + (src_i * kQlen + q) * kK5RxTokenWords + 4 * k;
    // Keep the epoch padding from the 16-byte wire record so ptxas can emit
    // one aligned shared-memory vector store instead of three scalar stores.
    *reinterpret_cast<uint4*>(out) =
        make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32),
                   static_cast<uint32_t>(hi), static_cast<uint32_t>(hi >> 32));
  }
  // Only the single-phase K5 op (kOpK5Collective) keeps the own-hidden
  // rendezvous.  The persistent kernel (ctx.op == kOpK5) folds this publication
  // into barrier 8 below.
  if (ctx.op != kOpK5) {
    if (ctx.warp < kRxWarpBase) {
      asm volatile("barrier.cta.arrive 2, 224;" ::: "memory");
    } else if (ctx.warp < kRxWarpBase + kReduceWarps) {
      asm volatile("barrier.cta.sync 2, 224;" ::: "memory");
    }
  }

  // One seven-warp rendezvous orders both sources used by the reduction:
  // token warps 0..3 publish own hidden pairs, while the six receive owners
  // (0..2 and 4..6) publish every peer slab.  Reducers 4..6 acquire both.
  if (ctx.warp < kRxWarpBase) {
    asm volatile("barrier.cta.arrive 8, 224;" ::: "memory");
  } else if (ctx.warp >= kRxWarpBase && ctx.warp < kRxWarpBase + kReduceWarps) {
    asm volatile("barrier.cta.sync 8, 224;" ::: "memory");
  }

  // Collective sum: rank order 1..7, __fadd_rn chain from 0, + FP32 x, final BF16.
  // Paired form: each add.rn.f32x2 lane is one row's scalar chain in the same rank order,
  // one final BF16 rounding stays per row; the pair leaves as one aligned b32 store.
  if (pair_owner) {
    constexpr int kPairsPerRecord = arena::kRxRecordElems / 2;
    const int peer_word = pair_p + pair_p / kPairsPerRecord;
    if (null_math) {
      *reinterpret_cast<uint32_t*>(args.endpoint + pair_index) = 0u;
    } else {
      uint64_t total_pair = 0ull;
      int src_i = 0;
      for (int rank = 1; rank < kWorld; ++rank) {
        uint32_t term_bits;
        if (rank == args.rank) {
          term_bits = hidden_stage[pair_q * kPairs + pair_p];
        } else {
          term_bits = rx_s[(src_i * kQlen + pair_q) * kK5RxTokenWords + peer_word];
          ++src_i;
        }
        total_pair =
            add_f32x2(total_pair, pack_f32x2(bf16_pair_low(term_bits), bf16_pair_high(term_bits)));
      }
      const F32Pair total{total_pair};
      *reinterpret_cast<uint32_t*>(args.endpoint + pair_index) =
          pack_bf16_pair(__fadd_rn(bf16_pair_low(x_bits), total.value[0]),
                         __fadd_rn(bf16_pair_high(x_bits), total.value[1]));
    }
  }
  // Kernel completion orders the endpoint stores; no later phase reuses this
  // scratch or consumes a CTA-local completion barrier.

  // No completion step: rank 0 is paced by the early kFlagAck (op_k0_rank0_publish), so nothing
  // consumes kFlagDone and K5 arrives on no completion counter.
}

static __device__ void op_k5_o_proj_collective(const MegaArgs& args, OpCtx& ctx) {
  if (args.no_collective != kCollectiveFull) {
    // Single-GPU, no TP all-reduce: o_proj into the arena (locally visible),
    // local endpoint (kCollectiveLocal) or none (kCollectiveOProjOnly); no inbox push / poll, no
    // kFlagDone.
    op_k5_o_proj_gemv(args, ctx, wsp<__nv_bfloat16>(ctx.arena, arena::kLocalHidden),
                      args.no_collective == kCollectiveLocal ? args.endpoint : nullptr);
    return;
  }
  op_k5_o_proj_phase(args, ctx);
  op_k5_collective_phase(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::mla_q4

namespace flashinfer::glm5_tp8::mla_q4 {

// Replace mode: rank 0 runs the collective side itself.  CTAs 0..127 receive the seven attention
// ranks' 48-row slices of their rows from rank 0's own inbox (slot r-1 = attention rank r,
// epoch-tagged records, same layout the attention ranks use) and write hidden_out = BF16(x +
// rank-ordered fp32 sum)
// -- the attention ranks' phase-B math without an own term. 96 pair workers per CTA (4 tokens x 24
// pairs); no smem, no counters.
static __device__ void op_k0_rank0_reduce(const MegaArgs& args, OpCtx& ctx) {
  if (ctx.cta >= kOProjCtas) return;
  constexpr int kPairs = kOProjRowsPerCta / 2;
  if (ctx.tid >= kQlen * kPairs) return;
  const int q = ctx.tid / kPairs;
  const int p = ctx.tid - q * kPairs;
  const int k = p / 3;
  const int w = p - k * 3;
  const int pair_index = q * kHidden + kOProjRowsPerCta * ctx.cta + 2 * p;
  const uint32_t x_bits = *reinterpret_cast<const uint32_t*>(args.x + pair_index);
  uint64_t total = 0ull;
  for (int src = 1; src < kWorld; ++src) {
    const uint8_t* record = ctx.arena + k5_rx_offset(g_mega_epoch, src - 1, ctx.cta, q, k);
    uint64_t lo = 0, hi = 0;
    ld_relaxed_sys_b128(record, lo, hi);
    while (static_cast<uint32_t>(hi >> 32) != g_mega_epoch) {
      __nanosleep(32);
      ld_relaxed_sys_b128(record, lo, hi);
    }
    const uint32_t term =
        w == 0 ? static_cast<uint32_t>(lo)
               : (w == 1 ? static_cast<uint32_t>(lo >> 32) : static_cast<uint32_t>(hi));
    total = add_f32x2(total, pack_f32x2(bf16_pair_low(term), bf16_pair_high(term)));
  }
  const F32Pair t{total};
  *reinterpret_cast<uint32_t*>(args.endpoint + pair_index) = pack_bf16_pair(
      __fadd_rn(bf16_pair_low(x_bits), t.value[0]), __fadd_rn(bf16_pair_high(x_bits), t.value[1]));
}
static __device__ void tail_ll_prefetch(const MegaArgs& args, const OpCtx& ctx, int helper) {
  constexpr int kHelpers = kTailCtas - (kTailIds + 1);
  const uint32_t stats_target = ArrivalCounter::target(kArriveK3Stats, g_mega_epoch);
  const uint32_t tag = g_ll_tag;
  for (int r = helper * kOpThreads + ctx.tid; r < kK3SplitCtas * 32; r += kHelpers * kOpThreads) {
    const int query = r >> 10;
    if (g_mega_cur_pos + query + 1 <= kSelected) continue;
    const uint8_t* ll_row = g_ll_recv + size_t(r) * 16;
    const uint32_t* stats = ctx.counters + cnt_word(kCntK3Stats + query);
    uint64_t lo = 0, hi = 0;
    bool seen = false;
    for (;;) {
      ld_relaxed_sys_b128(ll_row, lo, hi);
      if (static_cast<uint32_t>(lo >> 32) == tag && static_cast<uint32_t>(hi >> 32) == tag) {
        seen = true;
        break;
      }
      if (word_reached(stats, stats_target)) break;
    }
    if (!seen) continue;
    const int ids[2] = {static_cast<int32_t>(lo), static_cast<int32_t>(hi)};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int id = ids[j];
      if (id >= 0 && id < args.cache_rows && id <= g_mega_cur_pos + query) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(
                         args.kv_cache + static_cast<int64_t>(id) * kKvlora),
                     "r"(static_cast<uint32_t>(kKvlora * sizeof(__nv_bfloat16)))
                     : "memory");
        asm volatile(
            "prefetch.global.L2 [%0];" ::"l"(args.pe_cache + static_cast<int64_t>(id) * kRope)
            : "memory");
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The persistent kernel.  148 CTAs x 384 threads, 1 CTA/SM (asserted at init). Prologue: threads
// 0..3 init the 8 ring mbarriers, then the ONLY block-wide barrier.  Producers (warps 8..11) stream
// this CTA's weight pages K1 -> K2 fused -> [K3 KV gather after C2 + C3] -> K4 -> K5 through the
// 4-slot ring, throttled by the consumers' empty[] arrivals; the K3 gather is the only producer
// wait on arena words (arrived by the tail CTAs).  Consumers (warps 0..7):
//   Group1 (CTAs 0..127): K1 qkv_a_proj (CTAs < 82 own pages) -> arrive C0
//                         -> [CTAs < 120: wait C0, fused q_b_proj half tiles -> per-head word
//                             kCntK2Head (12 arrivals) -> W_UK (cohorts 0..7) | PE half tiles
//                             (8..11)]   -> arrive C4
//   tails (CTAs 128..147): tails 0..7 wait C0; kv-norm row t (0..3) -> arrive C3;
//                         pe RoPE query t-4 (4..7), ids copy slice t-8 (8..15; the last of the
//                         eight releases kFlagAck), idle (16..19) -> arrive C2 -> exit (no
//                         attention phase)
//   Group1 (128):         wait C4 + C2 + C3 (one poll loop, one acquire fence)
//                         -> K3 split-KV attention (CTAs < 128: (query, split); the producer warps
//                         gathered
//                            the split's KV rows into two ring slots after C2 + C3; split CTAs run
//                            the one-pass softmax (local max, unnormalized bf16 exp2, fp32 (max,
//                            sum) statistics), PV, arrive kCntK3Partials[query], splits 0..19
//                            combine with LSE weights x rcp(Z))
//                            -> arrive C6 (128)
//   Group1:               [CTAs < 80: wait C6, W_UV]     -> arrive C7
//                         -> wait C7 -> K5 o_proj -> push the 48-row slice into the 6 peers'
//                         inboxes
//                            (16 B {6 x bf16, epoch} records, st.relaxed.sys.b128) -> poll the
//                            own inbox -> rank-ordered sum + residual -> endpoint;
//                            rank-zero slot reuse additionally waits for its per-CTA exit credit
// Every CTA of a boundary's group arrives once per epoch whether or not it had
// work (target = group size x epoch, monotonic, no reset, no last-arriver);
// idle CTAs skip the wait of a phase they have no work in and arrive at once.
// Public dispatch uses this kernel on ranks 1..7 and the indexer on rank zero.
// Phase counters and the exit-credit protocol protect the parity-buffered K5
// inbox. Initialization checks full-grid residency, and the package launches it cooperatively.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kBlock, 1) void mega_attn_persistent(
    const __grid_constant__ MegaArgs args_in) {
  asm volatile("griddepcontrol.wait;" ::
                   : "memory");  // PDL: launched behind the previous layer's MoE; nothing before
                                 // this touches memory
  extern __shared__ __align__(128) uint8_t dynamic_shared[];
  __shared__ __align__(8) uint64_t ring_bars[smem::kRingBarriers];  // full[4], empty[4]
  uint8_t* op_smem = dynamic_shared;
  uint8_t* ring = dynamic_shared + smem::kOpScratchBytes;
  uint8_t* arena_base = args_in.peers[args_in.rank];
  const int cta = static_cast<int>(blockIdx.x);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const bool producer = warp >= kOpWarps;

  if (threadIdx.x == 0 && args_in.rank == 0)
    mega_load_launch_scalars(args_in, cta);  // ranks 1..7: producer_publish_launch_scalars
  // Rank 0 only publishes selected ids and never touches the page ring.  Skip
  // its eight mbarrier initializations and the 384-thread prologue rendezvous;
  // K0 synchronizes its 256 consumers internally before publishing.  The 256
  // consumers first meet on named barrier 1 so that the shared epoch is visible.
  if (args_in.rank == 0) {
    if (!producer) {
      asm volatile("bar.sync 1, 256;" ::: "memory");
      OpCtx ctx = make_op_ctx(kOpK0, cta, kK0Ctas, op_smem, ring, ring_bars, arena_base);
      if (!(args_in.dense_ids && args_in.ids_direct))
        op_k0_rank0_publish(args_in, ctx);  // dense replace mode: nobody consumes the publish
      if (args_in.rank0_reduce) op_k0_rank0_reduce(args_in, ctx);
    }

    return;
  }

  // Preissue the correctness-bearing first qkv_a_proj page before the only
  // block-wide prologue rendezvous.  Producer lane 0 owns slot 0, contributes
  // all four logical arrivals, and resumes the normal stream at page 1.  This
  // overlaps the cold HBM miss without duplicate traffic or an L2-residency
  // control.
  const bool k1_page0_preissued = cta < kQkvAProjCtaBlocks && args_in.mode == kModePersistent;
  if (k1_page0_preissued && threadIdx.x == kOpThreads) {
    const uint32_t full0 = smem_u32(ring_bars);
    mbar_init(full0, kProducerWarps);
    mbar_init(smem_u32(ring_bars + smem::kRingSlots), kOpThreads);
    mbar_fence_init();
    const uint8_t* page0 =
        args_in.qkv_a_proj + static_cast<int64_t>(cta) * kQkvAProjPagesPerBlock * kPageBytes8080;
    mbar_arrive_expect_tx(full0, kPageBytes8080);
    mbar_arrive_count(full0, kProducerWarps - 1);
    bulk_copy_g2s_l2(smem_u32(ring), page0, kPageBytes8080, full0, l2_policy_evict_first());
  }

  // Issue this CTA's K1 x / x-gamma loads before the prologue
  // rendezvous (asm-pinned): 82 CTAs cold-miss the same 72 KiB, and issuing at
  // entry overlaps that round trip with the ring init, setmaxnreg and the
  // page-0 TMA flight.  Values are consumed unchanged by op_k1_qkv_a_proj_cache_pre.
  K1EntryLoads k1_pre;
  const bool k1_pre_ok = !producer && cta < kQkvAProjCtaBlocks && args_in.mode == kModePersistent;
  if (k1_pre_ok) k1_entry_issue(args_in, static_cast<int>(threadIdx.x), k1_pre);

  ring_init_barriers(ring_bars, k1_page0_preissued ? 1 : 0);
  // Producer warp 9 lanes 0..3 initialize the ring slots; lane 4 initializes
  // the independent K3 q_nope completion barrier in parallel with them.
  if (threadIdx.x == kOpThreads + 32 + smem::kRingSlots) {
    mbar_init(smem_u32(&g_k3_qnope_bar), 1);
    mbar_fence_init();
  }
  __syncthreads();                 // the only block-wide barrier: all 12 warps are still together
  const MegaArgs& args = args_in;  // constant-bank operands; the per-launch epoch / cur_pos live in
                                   // g_mega_epoch / g_mega_cur_pos

  // The producer warpgroup releases registers that the two compute warpgroups can use for the
  // deeply unrolled GEMV/attention paths. Each branch is warpgroup-uniform, as required by
  // setmaxnreg.sync.aligned.
  if (producer) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
    producer_main(args, cta, smem_u32(ring), smem_u32(ring_bars),
                  smem_u32(ring_bars + smem::kRingSlots), k1_page0_preissued);
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");

  uint32_t epoch =
      0u;  // read after named barrier 12 (launch scalars published by the last producer warp)

  OpCtx ctx = make_op_ctx(kOpK1, cta, kK1Ctas, op_smem, ring, ring_bars, arena_base);
  if (args.mode == kModeNullStream) {  // weight-stream floor: consumers only drain the ring
    consumer_drain_all(args, ctx);
    return;
  }

  if (cta < kGroup1Ctas) {
    // K1 qkv_a_proj: CTAs < 82 own one 32-row block each; only those producers arrive C0.
    // Always pass the entry-load struct: op_k1_qkv_a_proj_cache_pre returns before any
    // use for cta >= kQkvAProjCtaBlocks and skips staging in the null modes, i.e.
    // it reads the struct exactly when k1_pre_ok loaded it.  A compile-time
    // non-null pointer lets SROA keep the struct in registers.
    // The K1 epilogue bulk-publishes its four token slices and optional block-major Q-norm
    // record, then arrives C0 after completion.
    op_k1_qkv_a_proj_cache_pre(args, ctx, &k1_pre);
    asm volatile("bar.sync 12, %0;" ::"n"(kOpThreads + 32) : "memory");
    epoch = g_mega_epoch;

    // K2 fused q-chain (CTAs < 120 = 10 heads x 12 cohorts): needs every post_q block (C0); the
    // W_UK of a head waits only for its 12 cohort arrivals (kCntK2Head, page-local hand-off) -- no
    // C1 boundary.  Only the 120 fused-K2 producers arrive C4.
    if (cta < kK2FusedCtas) {
      ctx.op = kOpK2Fused;
      ctx.num_ctas = kK2FusedCtas;
      K2GammaLoads k2_gamma;
      k2_gamma_issue(args, ctx.warp, ctx.lane, k2_gamma);
      ctx.boundary_wait(kCntC0, ArrivalCounter::target(kArriveC0, epoch));
      op_k2_fused_pre(args, ctx, &k2_gamma);
    }
    if (cta < kK2FusedCtas && args.mode == kModePersistent && args.ids_ready) {
      // Query records for the K3 split CTAs (kRecQnope / kRecQpe) plus the same post_qnope /
      // post_qpe values, stored directly by the owning threads: no bulk store, no bulk wait.  C4 /
      // C4Pe keep counting (relaxed) so every launch adds the same arrivals.
      ctx.sync();
      const int c4_head = cta / kK2FusedCohort;
      const int c4_cohort = cta - c4_head * kK2FusedCohort;
      const __nv_bfloat16* final_publish_s = ctx.scratch<__nv_bfloat16>(kK2FinalPublishOffset);
      if (c4_cohort < kK2FusedUkPerHead) {
        if (ctx.tid < kQlen * (kK2FinalPublishRows / 4)) {
          const int token = ctx.tid / (kK2FinalPublishRows / 4);
          const int r4 = ctx.tid - token * (kK2FinalPublishRows / 4);
          const uint2 w = *reinterpret_cast<const uint2*>(final_publish_s +
                                                          token * kK2FinalPublishRows + 4 * r4);
          const int row = c4_cohort * kK2FinalPublishRows + 4 * r4;
          rec_store_b128(ctx.arena + kRecQnope +
                             (size_t(token * kHeads + c4_head) * (kKvlora / 4) + row / 4) * 16,
                         w.x, w.y, g_mega_epoch);
          *reinterpret_cast<uint2*>(args.post_qnope + token * kUkRows + c4_head * kKvlora + row) =
              w;
        }
      } else {
        if (ctx.tid < kQlen * (kK2WPublishRows / 4)) {
          const int token = ctx.tid / (kK2WPublishRows / 4);
          const int r4 = ctx.tid - token * (kK2WPublishRows / 4);
          const uint2 w =
              *reinterpret_cast<const uint2*>(final_publish_s + token * kK2WPublishRows + 4 * r4);
          const int dim = (c4_cohort - kK2FusedUkPerHead) * kK2WPublishRows + 4 * r4;
          rec_store_b128(
              ctx.arena + kRecQpe + (size_t(token * kHeads + c4_head) * (kRope / 4) + dim / 4) * 16,
              w.x, w.y, g_mega_epoch);
          *reinterpret_cast<uint2*>(args.post_qpe + token * kQbProjPeRows + c4_head * kRope + dim) =
              w;
        }
      }
      if (ctx.tid == 0) {
        red_relaxed_gpu_add(
            ctx.counters + cnt_word(c4_cohort < kK2FusedUkPerHead ? kCntC4 : kCntC4Pe), 1u);
      }
    } else if (cta < kK2FusedCtas) {
      // The W_UK/PE epilogues need this barrier before their release.

      // of each head finish first (no per-head W_UK rendezvous): they release
      // kCntC4Pe (40 arrivals), which is all the QK-PE MMA consumes; the eight
      // W_UK cohorts release kCntC4 (80 arrivals), which gates only the q_nope
      // staging in K3 -- issued so its latency hides under the KV ring wait.
      ctx.sync();
      if (ctx.tid == 0) {
        const int c4_head = cta / kK2FusedCohort;
        const int c4_cohort = cta - c4_head * kK2FusedCohort;
        __nv_bfloat16* final_publish_s = ctx.scratch<__nv_bfloat16>(kK2FinalPublishOffset);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (c4_cohort < kK2FusedUkPerHead) {
          constexpr uint32_t kTokenBytes = kK2FinalPublishRows * uint32_t(sizeof(__nv_bfloat16));
          const int row_base = (c4_head * kK2FusedUkPerHead + c4_cohort) * kK2FinalPublishRows;
#pragma unroll
          for (int token = 0; token < kQlen; ++token) {
            bulk_copy_s2g(reinterpret_cast<uint8_t*>(args.post_qnope + token * kUkRows + row_base),
                          smem_u32(final_publish_s + token * kK2FinalPublishRows), kTokenBytes);
          }
        } else {
          constexpr uint32_t kTokenBytes = kK2WPublishRows * uint32_t(sizeof(__nv_bfloat16));
          const int row_base = c4_head * kRope + (c4_cohort - kK2FusedUkPerHead) * kK2WPublishRows;
#pragma unroll
          for (int token = 0; token < kQlen; ++token) {
            bulk_copy_s2g(
                reinterpret_cast<uint8_t*>(args.post_qpe + token * kQbProjPeRows + row_base),
                smem_u32(final_publish_s + token * kK2WPublishRows), kTokenBytes);
          }
        }
        bulk_copy_s2g_wait();
        red_release_gpu_add(
            ctx.counters + cnt_word(c4_cohort < kK2FusedUkPerHead ? kCntC4 : kCntC4Pe), 1u);
      }
    }
  } else {
    asm volatile("bar.sync 12, %0;" ::"n"(kOpThreads + 32) : "memory");
    epoch = g_mega_epoch;
    // Tail group: kv-norm rows -> C3; four RoPE rows plus the one local
    // ids-acquire CTA -> C2.  The other ids-copy CTAs retire without arriving.
    const int tail = cta - kGroup1Ctas;
    ctx.op = kOpK1Tails;
    ctx.cta = tail;
    ctx.num_ctas = kTailCtas;
    const TailPreloads tail_pre = tail_preload(args, tail, ctx.tid);
    if (tail < kTailRope + kQlen) {  // kv-norm / RoPE rows read post_kv / pe_raw of K1
      ctx.boundary_wait(kCntC0, ArrivalCounter::target(kArriveC0, epoch));
    }
    if (tail < kTailIds + 1) {
      op_k1_tails(args, ctx, tail_pre);
      if (tail < kKvNormCtas) {
        // op_k1_tails ends with a consumer barrier.
        ctx.boundary_arrive_presynced(kCntC3);
      } else {
        ctx.boundary_arrive_presynced(kCntC2);
      }
    }
    if (tail > kTailIds && args.mode == kModePersistent && args.ll_ids && g_ll_recv != nullptr) {
      tail_ll_prefetch(args, ctx, tail - kTailIds - 1);
    }
    // the split-KV attention is owned entirely by CTAs 0..127
    // (op_k3_attention returns at once for cta >= kK3SplitCtas), so once a tail has published C2 /
    // C3 it retires.
    return;
  }

  // K3 attention (the 128 Group1 CTAs): pe_cache rows and the consumed ids (C2),
  // kv_cache rows (C3), then post_q_nope (C4, which implies C1 and C0).  The op
  // body waits C2 + C3 first, prestages ids / PE under the C4 poll, and only the
  // query slab staging remains on the post-C4 path.
  ctx.op = kOpK3;
  ctx.cta = cta;
  ctx.num_ctas = kK3Ctas;
  op_k3_attention(args, ctx);

  // K4 W_UV (CTAs < 80): each CTA first owns one head-local K3 combine role,
  // then enters W_UV when all eight roles of that head have published.
  if (cta < kK4Ctas) {
    ctx.op = kOpK4;
    ctx.num_ctas = kK4Ctas;
    // The W_UV epilogue bulk-publishes its four token slices and arrives C7 after completion.
    op_k4_attention_combine_uv(args, ctx);
  }

  // K5 o_proj + collective (all 128 Group1 CTAs): needs post_uv (C7).
  ctx.op = kOpK5;
  ctx.num_ctas = kK5Ctas;
  if (args.mode != kModePersistent) {
    ctx.boundary_wait(kCntC7, ArrivalCounter::target(kArriveC7, epoch));
  } else {
    ctx.sync();
  }
  op_k5_o_proj_collective(args, ctx);
}

// ---------------------------------------------------------------------------

}  // namespace flashinfer::glm5_tp8::mla_q4

#include "launch_util.cuh"
namespace flashinfer::glm5_tp8 {
KernelResources initialize_mla_q4(int device) {
  using namespace mla_q4;
  return attention_detail::configure(reinterpret_cast<const void*>(&mega_attn_persistent), kGrid,
                                     kBlock, smem::kPersistentDynamicBytes, device);
}
cudaError_t launch_mla_q4(const AttentionArguments& p, cudaStream_t stream, bool pdl,
                          bool cooperative) {
  using namespace mla_q4;
  const auto& m = p.mla;
  MegaArgs a{};
  a.rank = p.rank;
  for (int r = 0; r < 8; ++r) a.peers[r] = p.peer_arenas[r];
  a.x = reinterpret_cast<const __nv_bfloat16*>(m.x);
  a.x_gamma = m.x_gamma;
  a.q_gamma = m.q_gamma;
  a.kv_gamma = m.kv_gamma;
  a.rope = m.rope;
  a.qkv_a_proj = m.qkv_a_proj;
  a.q_b_proj = m.q_b_proj;
  a.w_uk = m.w_uk;
  a.w_uv = m.w_uv;
  a.o_proj = m.o_proj;
  a.kv_cache = reinterpret_cast<__nv_bfloat16*>(m.kv_cache);
  a.pe_cache = reinterpret_cast<__nv_bfloat16*>(m.pe_cache);
  a.selected_ids_src = m.selected_ids;
  a.selected_lengths_src = m.selected_lengths;
  a.cache_rows = m.cache_rows;
  a.x_eps = a.q_eps = a.kv_eps = 1e-5f;
  a.softmax_scale = 0.0625f;
  a.post_q = reinterpret_cast<__nv_bfloat16*>(m.post_q);
  a.post_kv = reinterpret_cast<__nv_bfloat16*>(m.post_kv);
  a.post_qnd = reinterpret_cast<__nv_bfloat16*>(m.post_qnd);
  a.post_qpe = reinterpret_cast<__nv_bfloat16*>(m.post_qpe);
  a.post_qnope = reinterpret_cast<__nv_bfloat16*>(m.post_qnope);
  a.post_o = reinterpret_cast<__nv_bfloat16*>(m.post_o);
  a.post_uv = reinterpret_cast<__nv_bfloat16*>(m.post_uv);
  a.endpoint = reinterpret_cast<__nv_bfloat16*>(m.out);
  a.mode = kModePersistent;
  a.ids_direct = a.skip_done_wait = a.ids_ready = a.rank0_reduce = a.dense_ids = a.ll_ids = 1;
  a.cur_pos_dev = m.cur_pos;
  a.epoch_ctr = p.epoch_ctr;
  a.ll_recv = m.ll_recv;
  a.ll_tag = m.ll_tag;
  return attention_detail::launch(reinterpret_cast<const void*>(&mega_attn_persistent), a, kGrid,
                                  kBlock, smem::kPersistentDynamicBytes, stream, pdl, cooperative);
}
}  // namespace flashinfer::glm5_tp8
