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
#ifndef FLASHINFER_FUSED_MOE_BF16_MOE_DECODE_SM103_CUH_
#define FLASHINFER_FUSED_MOE_BF16_MOE_DECODE_SM103_CUH_

// Fused BF16 MoE decode kernels for SM103, the bf16_decode_sm103_glm52 kernel of the
// specialized MoELayer backend
// (flashinfer/fused_moe/backends/specialized/kernels/bf16_decode_sm103_glm52):
// DeepSeekV3 routing with one expert group (sigmoid scores, top-8 by score + bf16(bias),
// weights renormalized, scaled by the routed scaling factor and rounded to bf16), SwiGLU,
// 256 experts, hidden 6144, intermediate 256, bf16 weights in the shuffled BlockMajorK
// layout of trtllm_bf16_moe.
//
// Each call is one cooperative persistent launch with one CTA per SM. Every CTA computes
// the routing of the call's tokens; the selected experts' W13 and W2 are streamed once
// through a shared-memory ring (bulk / tensor-map copies) into mma.sync tiles with the
// tokens on N, GEMM1 split over the CTAs with fp32 partials, SwiGLU rounded to bf16,
// GEMM2 per row slice, and the top-k weighted sum accumulated in fp32 and written as
// bf16. The launches synchronize CTAs through device-global counters (hence the
// cooperative launch); the per-launch accumulators are kept clean by the kernels
// themselves, and launches serialize on that device state in the kernels (a launch starts
// on it once every CTA of every earlier launch has left), so calls may come from any
// stream or CUDA graph. Specializations: one token, two tokens, six and ten tokens, and a generic
// path for up to 16 tokens; a call is one launch, so 16 tokens is the maximum (the dispatch
// never sends more).
//
// Numerics match the trtllm-gen BF16 path: fp32 accumulation, routing weights / SwiGLU
// output / GEMM2 output rounded to bf16 where trtllm-gen stores them, bf16 output.
// Fused BF16 MoE (DeepSeekV3 routing, SwiGLU) for decode sizes on Blackwell (sm_103).
// Two persistent cooperative kernels, dispatched on the token count of a launch (<= 16 tokens):
//   small (T <= 2): kernel A - GEMM1 split-K over all CTAs, one grid barrier, then GEMM2
//   larger (T >= 3): kernel B - per-expert readiness counters instead of the grid barrier, so
//                    the weight stream of GEMM2 never waits for all of GEMM1
// See the per-namespace comments for details.
//
// This header is framework-agnostic (raw pointers, tensor maps and device globals only). The
// TVM-FFI entry point (tensor maps, launch) lives in csrc/fused_moe/bf16_moe_decode_sm103.cu;
// include it from one translation unit: the kernels keep cross-CTA state in device globals.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace fused_moe {
namespace bf16_decode_sm103 {

namespace kA {
// Fused BF16 MoE (DeepSeekV3 routing, SwiGLU) for decode sizes on Blackwell (sm_103).
// One persistent cooperative kernel, one CTA per SM (8 consumer warps + 1 producer warp):
//   1. every CTA computes the routing for the (<=16) tokens of this launch
//   2. GEMM1: the flattened (expert, 128-row group, k-block) work list is split evenly over
//      CTAs; the producer streams 16 KB weight pieces (+ the needed x rows) with bulk copies
//      into an smem ring; consumers run mma.sync (tokens on N) and red.add their fp32
//      split-K partials into a global accumulator
//   3. grid barrier; meanwhile the producer keeps prefetching this CTA's W2 rows
//   4. GEMM2: each CTA owns a contiguous slice of the 6144 output rows for all experts,
//      builds the bf16 SwiGLU table from the accumulator, and writes the bf16 output.
// Fragment trick: the dot product does not care about k order, so each thread feeds the MMA
// from contiguous 16 B smem chunks (same physical k permutation for A and B).

constexpr int E = 256, TOPK = 8, H = 6144, I = 256, M1 = 512, KB1 = H / 64, KB2 = I / 64;
constexpr int TCH = 16;  // tokens per launch
constexpr int NCW = 8;   // consumer warps
constexpr int NCT = NCW * 32;
constexpr int NTHREADS = NCT + 32;  // + producer warp
constexpr int MAXNS = 16;
constexpr int PIECE1 = 128 * 128;  // 128 rows x 64 bf16
constexpr int MAX_PAIRS = TCH * TOPK;
constexpr int OUT_ROWS = 64;      // padded W2 rows per CTA in the output accumulator
constexpr int W2_TILE_ROWS = 48;  // 3 MMA row tiles, padded for one swizzled TMA
constexpr int W2_TX_ROWS = 41;    // maximum real CTA row slice on 152 SMs
constexpr int SMEM_LIMIT = 232448;

struct SStatic {
  unsigned long long full[MAXNS];
  unsigned long long empty[MAXNS];
  unsigned long long full2[32];
  unsigned long long empty2[8];
  unsigned long long bar_base;
  int tk_e[TCH][TOPK];
  float tk_w[TCH][TOPK];
  unsigned mask[E];
  int slot_expert[E];
  int slot_pstart[E + 1];
  int pair_tok[MAX_PAIRS];
  float pair_w[MAX_PAIRS];
  int wsum[NCW];
  int n_active, n_pairs, max_tok, epoch, tlslot;
  int dirty[2];
};
constexpr int DSMEM = SMEM_LIMIT - 6144;
static_assert(sizeof(SStatic) <= 6144, "static smem");
// More than half of an SM's 228 KB shared memory per CTA: one CTA per SM, so two launches of these
// kernels (each a cooperative grid over all SMs) overlap only while one's CTAs exit and the
// next one's start (see wait_device_state).
static_assert(SMEM_LIMIT > 228 * 1024 / 2, "one CTA per SM");

__device__ float g_acc[2][MAX_PAIRS * M1];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ unsigned long long g_bar;
__device__ unsigned long long g_w1;  // T = 1: CTAs that issued all their W13 pieces (monotonic)
__device__ int g_dirty[2];

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(unsigned long long* b, unsigned cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_arrive(unsigned long long* b, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(tx)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes,
                                         unsigned long long* b, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_w1_g2s(void* dst, const CUtensorMap& map, int row, int kb,
                                           int expert, unsigned long long* b, uint64_t policy) {
  const uint64_t desc = reinterpret_cast<uint64_t>(&map);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(smem_u32(dst)),
      "l"(desc), "r"(0), "r"(row), "r"(kb), "r"(expert), "r"(smem_u32(b)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_x_g2s(void* dst, const CUtensorMap& map, int token, int kb,
                                          unsigned long long* b, uint64_t policy) {
  const uint64_t desc = reinterpret_cast<uint64_t>(&map);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(smem_u32(dst)),
      "l"(desc), "r"(0), "r"(token), "r"(kb), "r"(smem_u32(b)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_x_gather4_g2s(void* dst, const CUtensorMap& map, int kb, int t0,
                                                  int t1, int t2, int t3, unsigned long long* b,
                                                  uint64_t policy) {
  const uint64_t desc = reinterpret_cast<uint64_t>(&map);
  asm volatile(
      "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(smem_u32(dst)),
      "l"(desc), "r"(kb * 64), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(smem_u32(b)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_x_gather4_cached_g2s(void* dst, const CUtensorMap& map, int kb,
                                                         int t0, int t1, int t2, int t3,
                                                         unsigned long long* b) {
  const uint64_t desc = reinterpret_cast<uint64_t>(&map);
  asm volatile(
      "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(smem_u32(dst)),
      "l"(desc), "r"(kb * 64), "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(smem_u32(b))
      : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(NCT) : "memory");
}
__device__ __forceinline__ uint4 lds128(const char* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ unsigned f2key(float f) {
  unsigned b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ unsigned long long ld_acquire64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ float atom_add_f32(float* p, float v) {
  float r;
  asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(v) : "memory");
  return r;
}
__device__ __forceinline__ void atom_add_v2_f32(float* p, float x, float y, float& rx, float& ry) {
  asm volatile("atom.global.v2.f32.add {%0, %1}, [%2], {%3, %4};"
               : "=f"(rx), "=f"(ry)
               : "l"(p), "f"(x), "f"(y)
               : "memory");
}
__device__ __forceinline__ float2 ldcg2(const float* p) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ void red_relaxed_add64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
// Launches of kernel A serialize on its device state (g_acc, g_dirty and g_bar itself), whatever
// stream or CUDA graph issued them: g_bar counts G barrier arrivals and then G exits per launch
// (each CTA counts its exit with a release once it is done with that state, see producer_exit),
// and a launch starts
// on the state once the count is a multiple of 2G, i.e. every CTA of every earlier launch has
// left; the acquire makes their writes visible. Two launches can overlap only while the CTAs of
// one exit and those of the next start (each is a cooperative grid of one CTA per SM over all
// SMs, see SMEM_LIMIT), and a CTA exits only after its grid barrier, when every CTA of its launch
// has read the count. Returns the count: 2G x the launches before this one.
__device__ __forceinline__ unsigned long long wait_device_state(unsigned long long G) {
  unsigned long long v = ld_acquire64(&g_bar);
  while (v % (2ull * G) != 0ull) {
    __nanosleep(128);
    v = ld_acquire64(&g_bar);
  }
  return v;
}
// Counts this CTA's exit from the device state (see wait_device_state). The consumers arrive on
// named barrier 6 once the accumulators are read into the SwiGLU table, their last access to the
// state; the producer warp, idle by then, waits on it and releases, so the release fence overlaps
// GEMM2 instead of delaying the output.
__device__ __forceinline__ void consumers_done_with_device_state() {
  asm volatile("bar.arrive 6, %0;" ::"n"(NTHREADS) : "memory");
}
__device__ __forceinline__ void producer_exit(int lane) {
  asm volatile("bar.sync 6, %0;" ::"n"(NTHREADS) : "memory");
  if (lane == 0) red_release_add64(&g_bar, 1ull);
}
__device__ __forceinline__ void red_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ unsigned u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
// D = A * B + D, m16n8k16, bf16 inputs, fp32 accumulate
__device__ __forceinline__ void mma_bf16(float* d, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16_02(float (&d)[2], unsigned a0, unsigned a1, unsigned a2,
                                            unsigned a3, unsigned b0, unsigned b1) {
  float d1 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d1), "+f"(d[1]), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 16-row x 64-k block: rows g / g+8 (16 B chunks t and t+4 of each), B chunks of one n-block.
__device__ __forceinline__ void mma_k64(float* d, const uint4 (&qa)[2][2], const uint4 (&qb)[2]) {
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int h = s >> 1, sub = (s & 1) * 2;
    mma_bf16(d, u4get(qa[0][h], sub), u4get(qa[1][h], sub), u4get(qa[0][h], sub + 1),
             u4get(qa[1][h], sub + 1), u4get(qb[h], sub), u4get(qb[h], sub + 1));
  }
}

__device__ __forceinline__ void lda(uint4 (&qa)[2][2], const char* p) {
  qa[0][0] = lds128(p);
  qa[0][1] = lds128(p + 64);
  qa[1][0] = lds128(p + 8 * 128);
  qa[1][1] = lds128(p + 8 * 128 + 64);
}
__device__ __forceinline__ uint4 lds128_swz(const char* p, int row, int chunk) {
  return lds128(p + row * 128 + ((chunk ^ (row & 7)) * 16));
}
__device__ __forceinline__ void ldmatrix4(unsigned (&v)[4], const char* p) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix4_smem(unsigned (&v)[4], unsigned p) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(p));
}
__device__ __forceinline__ void ldmatrix2(unsigned (&v)[2], const char* p) {
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared::cta.b16 {%0,%1}, [%2];"
               : "=r"(v[0]), "=r"(v[1])
               : "r"(smem_u32(p)));
}

struct Layout {
  int NS, slot_bytes, NS2, slot2_bytes, R_lo, R_c, P, it_lo, it_hi, nq, out_stride;
  int sh_lo, nsh, nf;  // T = 1: top-1 expert share, total W13 pieces of the CTA
  char* ring;
  char* table;
  float* outacc;
};

struct Ring {
  int slot = 0;
  unsigned phase = 0;
  __device__ __forceinline__ void next(int ns) {
    if (++slot == ns) {
      slot = 0;
      phase ^= 1u;
    }
  }
};

// ------------------------------------------------------------- GEMM1 segment
// Warp w owns rows w*16 + {g, g+8} of the 128-row group; NB n-blocks of 8 tokens.
template <int NB>
__device__ __forceinline__ float w1_segment(SStatic& S, const Layout& L, int it, int it_end,
                                            Ring& rs, int s, int rg, float* gacc,
                                            float (&pend)[4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int pst = S.slot_pstart[s], nt = S.slot_pstart[s + 1] - pst;
  float acc[NB][4];
#pragma unroll
  for (int n = 0; n < NB; ++n)
#pragma unroll
    for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
#pragma unroll 1
  for (; it < it_end; ++it) {
    const int slot = rs.slot;
    mbar_wait(&S.full[slot], rs.phase);
    const char* base = L.ring + slot * L.slot_bytes;
    unsigned qb_all[NB][2][4];
#pragma unroll
    for (int n = 0; n < NB; ++n) {
      const int brow = min(n * 8 + lane8, nt - 1);
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        const int bchunk = (brow & 7) ^ (kh * 4 + quad8);
        ldmatrix4(qb_all[n][kh], base + PIECE1 + brow * 128 + bchunk * 16);
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      unsigned qa[4];
      const int achunk = (lrow_a & 7) ^ (2 * kk + acol_half);
      ldmatrix4(qa, base + (warp * 16 + lrow_a) * 128 + achunk * 16);
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        const int h = kk >> 1, sub = (kk & 1) * 2;
        mma_bf16(acc[n], qa[0], qa[1], qa[2], qa[3], qb_all[n][h][sub], qb_all[n][h][sub + 1]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&S.empty[slot]);
    rs.next(L.NS);
  }
  // Returned values of the split-K atomics: the partials are performed in L2 once they are
  // read. Those of this segment are only read a segment later (or before the grid barrier),
  // so the round trip does not stall the next segment.
  float chk = (pend[0] + pend[1]) + (pend[2] + pend[3]);
  float* dst = gacc + (size_t)pst * M1 + rg * 128 + warp * 16 + g;
#pragma unroll
  for (int n = 0; n < NB; ++n) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int i = n * 8 + 2 * t + (q & 1);
      float r = 0.f;
      if (i < nt) r = atom_add_f32(dst + (size_t)i * M1 + (q >> 1) * 8, acc[n][q]);
      if (NB == 1)
        pend[q] = r;
      else
        chk += r;
    }
  }
  return chk;
}

// ------------------------------------------------------------- GEMM2 (consumers)
// Warp w handles the experts with slot index s % 8 == w: all m-tiles (16 rows each) of this
// CTA's row slice, full K = 256 (4 pieces of 64), then rounds o to bf16 and accumulates
// weight * o into the smem output accumulator. Every warp waits for and releases every fill.
template <int NB>
__device__ __forceinline__ void gemm2(SStatic& S, const Layout& L, Ring& rs) {
  // (all activations are in the table once the grid barrier has passed)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
  float acc[3][NB][4];
  int pst = 0, nt = 0;
#pragma unroll 1
  for (int u0 = 0; u0 < L.nq; u0 += L.P) {
    mbar_wait(&S.full2[rs.slot], rs.phase);
    const int u1 = min(u0 + L.P, L.nq);
    const char* fbase = L.ring + rs.slot * L.slot2_bytes;
#pragma unroll 1
    for (int u = u0; u < u1; ++u) {
      const int s = u >> 2, kb = u & 3;
      if ((s & (NCW - 1)) != warp) continue;
      if (kb == 0) {
        pst = S.slot_pstart[s];
        nt = S.slot_pstart[s + 1] - pst;
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
          for (int n = 0; n < NB; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[m][n][c] = 0.f;
      }
      const char* base = fbase + (u - u0) * W2_TILE_ROWS * 128;
      unsigned qb[NB][2][4];
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        const int brow = min(n * 8 + lane8, nt - 1);
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          const int bchunk = kh * 4 + quad8;
          ldmatrix4(qb[n][kh], L.table + (pst + brow) * (I * 2) + kb * 128 + bchunk * 16);
        }
      }
#pragma unroll
      for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          unsigned qa[4];
          const int wr = m * 16 + lrow_a;
          const int achunk = (wr & 7) ^ (2 * kk + acol_half);
          ldmatrix4(qa, base + wr * 128 + achunk * 16);
#pragma unroll
          for (int n = 0; n < NB; ++n) {
            const int h = kk >> 1, sub = (kk & 1) * 2;
            mma_bf16(acc[m][n], qa[0], qa[1], qa[2], qa[3], qb[n][h][sub], qb[n][h][sub + 1]);
          }
        }
      }
      if (kb == KB2 - 1) {
#pragma unroll
        for (int n = 0; n < NB; ++n) {
#pragma unroll
          for (int c2 = 0; c2 < 2; ++c2) {
            const int i = n * 8 + 2 * t + c2;
            if (i < nt) {
              const int tok = S.pair_tok[pst + i];
              const float w = S.pair_w[pst + i];
              float* orow = L.outacc + warp * L.out_stride + tok * OUT_ROWS + g;
#pragma unroll
              for (int m = 0; m < 3; ++m) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                  const int r = m * 16 + g + h * 8;
                  if (r < L.R_c) {
                    const float o = __bfloat162float(__float2bfloat16_rn(acc[m][n][h * 2 + c2]));
                    orow[m * 16 + h * 8] += o * w;
                  }
                }
              }
            }
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&S.empty2[rs.slot]);
    rs.next(L.NS2);
  }
}

// ------------------------------------------------------------- T == 1 W2 placement
// one 4D TMA box per expert: 41 rows x 4 k-blocks, flattened row R = kb * 41 + r (swizzle on R)
constexpr int W2_POS = 22 * 1024;  // 4 * 41 rows + 7 rows of ldmatrix over-read, 1 KB aligned
// first W2 position inside the first W1 slot to drain (f = pieces this CTA streamed)
__device__ __forceinline__ int w2_jstart(const Layout& L, int f) {
  return ((f % L.NS) * L.slot_bytes + W2_POS - 1) / W2_POS;
}

// GEMM2 for one token: warp w owns expert slot w (all 4 k-blocks of this CTA's row slice).
__device__ __forceinline__ void gemm2_t1(SStatic& S, const Layout& L, int jstart, int dry = 0) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
  const int npos = (L.NS * L.slot_bytes) / W2_POS;
  float acc[2][3][4];  // even / odd k-blocks: two independent MMA chains per m-tile
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int m = 0; m < 3; ++m)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[p][m][q] = 0.f;
  int j = jstart + warp;
  if (j >= npos) j -= npos;
  const char* base = L.ring + j * W2_POS;
  if (!dry) mbar_wait(&S.full2[warp], 0);
#pragma unroll
  for (int kb = 0; kb < KB2; ++kb) {
    unsigned qb[2][4];
#pragma unroll
    for (int kh = 0; kh < 2; ++kh)
      ldmatrix4(qb[kh], L.table + warp * (I * 2) + kb * 128 + (kh * 4 + quad8) * 16);
#pragma unroll
    for (int m = 0; m < 3; ++m) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        unsigned qa[4];
        const int wr = kb * L.R_c + m * 16 + lrow_a;
        const int achunk = (wr & 7) ^ (2 * kk + acol_half);
        ldmatrix4(qa, base + wr * 128 + achunk * 16);
        const int h = kk >> 1, sub = (kk & 1) * 2;
        mma_bf16(acc[kb & 1][m], qa[0], qa[1], qa[2], qa[3], qb[h][sub], qb[h][sub + 1]);
      }
    }
  }
  if (t == 0) {
    const float w = S.pair_w[warp];
    float* orow = L.outacc + warp * L.out_stride + g;
#pragma unroll
    for (int m = 0; m < 3; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m * 16 + g + h * 8;
        if (r < L.R_c) {
          const float o =
              __bfloat162float(__float2bfloat16_rn(acc[0][m][h * 2] + acc[1][m][h * 2]));
          orow[m * 16 + h * 8] += o * w;
        }
      }
    }
  }
}

// ------------------------------------------------------------------ kernel
template <int T>
__global__ void __launch_bounds__(NTHREADS, 1)
    moe_fused_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                     const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w1,
                     const __nv_bfloat16* __restrict__ w2, __nv_bfloat16* __restrict__ out,
                     const __grid_constant__ CUtensorMap w1_map,
                     const __grid_constant__ CUtensorMap x_map,
                     const __grid_constant__ CUtensorMap w2_map32,
                     const __grid_constant__ CUtensorMap w2_map9, int token_base,
                     float route_scale) {
  __shared__ SStatic S;
  extern __shared__ __align__(1024) char dsm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;

  // ---------------- routing
  if (tid == NCT) {
    const unsigned long long v = wait_device_state(G);
    const unsigned long long ep = v / (2ull * (unsigned long long)G);
    S.bar_base = v;
    S.epoch = (int)(ep & 1);
    S.tlslot = (int)(ep & 63);
    S.dirty[0] = g_dirty[0];
    S.dirty[1] = g_dirty[1];
  }
  if (T != 1 && tid < E) S.mask[tid] = 0u;
  if (tid == 0) S.max_tok = 0;
  if (T == 1) {
    if (warp == NCW) {
      // the producer warp is idle during routing: ring barriers + descriptor prefetch
      if (lane < MAXNS) {
        mbar_init(&S.full[lane], 1);
        mbar_init(&S.empty[lane], NCW);
      }
      mbar_init(&S.full2[lane], 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if (lane == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w1_map))
                     : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&x_map))
                     : "memory");
        if constexpr (T != 1)
          asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w2_map32))
                       : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w2_map9))
                     : "memory");
      }
    }
    // rank-based top-8: per-warp rank among 32 (shuffles), then rank among the 64 survivors.
    // Order: larger key first, lower expert index on ties.
    unsigned* cand_k = reinterpret_cast<unsigned*>(dsm);
    int* cand_e = reinterpret_cast<int*>(cand_k + 64);
    float* cand_s = reinterpret_cast<float*>(cand_e + 64);
    if (warp < NCW) {
      const int e = warp * 32 + lane;
      const float sv = 1.0f / (1.0f + expf(-logits[e]));
      const unsigned key = f2key(sv + __bfloat162float(__float2bfloat16_rn(bias[e])));
      int r = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        const unsigned kj = __shfl_sync(0xffffffffu, key, j);
        r += (kj > key) | ((kj == key) & (j < lane));
      }
      if (r < TOPK) {
        const int q = warp * TOPK + r;
        cand_k[q] = key;
        cand_e[q] = e;
        cand_s[q] = sv;
      }
    }
    __syncthreads();
    if (warp < 2) {
      const unsigned key = cand_k[tid];
      const int e = cand_e[tid];
      const uint4* ck = reinterpret_cast<const uint4*>(cand_k);
      const int4* ce = reinterpret_cast<const int4*>(cand_e);
      int r = 0;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint4 k4 = ck[j];
        const int4 e4 = ce[j];
        r += (k4.x > key) | ((k4.x == key) & (e4.x < e));
        r += (k4.y > key) | ((k4.y == key) & (e4.y < e));
        r += (k4.z > key) | ((k4.z == key) & (e4.z < e));
        r += (k4.w > key) | ((k4.w == key) & (e4.w < e));
      }
      if (r < TOPK) {
        S.tk_e[0][r] = e;
        S.tk_w[0][r] = cand_s[tid];
      }
      asm volatile("bar.sync 2, 64;" ::: "memory");
      if (warp == 0) {
        const float s = S.tk_w[0][lane & (TOPK - 1)];
        const int se = S.tk_e[0][lane & (TOPK - 1)];
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) sum += __shfl_sync(0xffffffffu, s, k);
        const float den = sum + 1e-20f;
        if (lane < TOPK) {
          const float wv = __bfloat162float(__float2bfloat16_rn((s * route_scale) / den));
          S.tk_w[0][lane] = wv;
          S.slot_expert[lane] = se;
          S.slot_pstart[lane] = lane;
          S.pair_tok[lane] = 0;
          S.pair_w[lane] = wv;
        }
        if (lane == 0) {
          S.slot_pstart[TOPK] = TOPK;
          S.n_active = TOPK;
          S.n_pairs = TOPK;
          S.max_tok = 1;
        }
      }
    }
  } else {
    float* sc_s = reinterpret_cast<float*>(dsm);                  // [T][E] sigmoid scores
    unsigned* sc_k = reinterpret_cast<unsigned*>(dsm) + TCH * E;  // [T][E] selection keys
#pragma unroll 1
    for (int i = tid; i < T * E; i += NTHREADS) {
      const float l = logits[i];
      const float b = __bfloat162float(__float2bfloat16_rn(bias[i & (E - 1)]));
      const float sv = 1.0f / (1.0f + expf(-l));
      sc_s[i] = sv;
      sc_k[i] = f2key(sv + b);
    }
    __syncthreads();
    if (warp < NCW) {
#pragma unroll 1
      for (int t = warp; t < T; t += NCW) {
        const uint4 k0 = *reinterpret_cast<const uint4*>(sc_k + t * E + lane * 8);
        const uint4 k1 = *reinterpret_cast<const uint4*>(sc_k + t * E + lane * 8 + 4);
        unsigned key[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k) {
          const unsigned b = max(max(max(key[0], key[1]), max(key[2], key[3])),
                                 max(max(key[4], key[5]), max(key[6], key[7])));
          const unsigned m = __reduce_max_sync(0xffffffffu, b);
          const unsigned bal = __ballot_sync(0xffffffffu, b == m);
          if (lane == __ffs(bal) - 1) {
            int bi = 7;
#pragma unroll
            for (int i = 6; i >= 0; --i)
              if (key[i] == m) bi = i;
#pragma unroll
            for (int i = 0; i < 8; ++i)
              if (i == bi) key[i] = 0u;
            const int e = lane * 8 + bi;
            S.tk_e[t][k] = e;
            S.tk_w[t][k] = sc_s[t * E + e];
          }
        }
        __syncwarp();
        if (lane == 0) {
          float sum = 0.f;
#pragma unroll
          for (int k = 0; k < TOPK; ++k) sum += S.tk_w[t][k];
          const float den = sum + 1e-20f;
#pragma unroll
          for (int k = 0; k < TOPK; ++k) {
            S.tk_w[t][k] =
                __bfloat162float(__float2bfloat16_rn((S.tk_w[t][k] * route_scale) / den));
            if (T != 1) atomicOr(&S.mask[S.tk_e[t][k]], 1u << t);
          }
        }
      }
    }
  }
  if (T != 1) {
    if (tid < MAXNS) {
      mbar_init(&S.full[tid], 1);
      mbar_init(&S.empty[tid], NCW);
    }
    if (tid < 8) {
      mbar_init(&S.full2[tid], 1);
      mbar_init(&S.empty2[tid], NCW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // ---------------- expert compaction (thread e handles expert e)
  if (T != 1) {
    int incl = 0, cnt = 0;
    if (tid < E) {
      cnt = __popc(S.mask[tid]);
      incl = ((cnt > 0) << 16) | cnt;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
      }
      if (lane == 31) S.wsum[warp] = incl;
      const int mx = __reduce_max_sync(0xffffffffu, cnt);
      if (lane == 0) atomicMax(&S.max_tok, mx);
    }
    __syncthreads();
    if (tid < E) {
      int pre = 0;
#pragma unroll
      for (int w = 0; w < NCW; ++w) pre += (w < warp) ? S.wsum[w] : 0;
      incl += pre;
      const int excl = incl - (((cnt > 0) << 16) | cnt);
      if (cnt > 0) {
        const int slot = excl >> 16, pst = excl & 0xffff;
        S.slot_expert[slot] = tid;
        S.slot_pstart[slot] = pst;
        unsigned m = S.mask[tid];
        int r = 0;
        while (m) {
          const int t = __ffs(m) - 1;
          m &= m - 1;
          S.pair_tok[pst + r] = t;
          float wv = 0.f;
#pragma unroll
          for (int k = 0; k < TOPK; ++k)
            if (S.tk_e[t][k] == tid) wv = S.tk_w[t][k];
          S.pair_w[pst + r] = wv;
          ++r;
        }
      }
      if (tid == E - 1) {
        S.n_active = incl >> 16;
        S.n_pairs = incl & 0xffff;
        S.slot_pstart[incl >> 16] = incl & 0xffff;
      }
    }
    __syncthreads();
  }
  const int n_active = S.n_active, n_pairs = S.n_pairs;

  // ---------------- layout
  Layout L;
  {
    const int xbytes = ((S.max_tok * 128) + 1023) & ~1023;
    L.slot_bytes = 17 * 1024;
    L.slot2_bytes = W2_TILE_ROWS * 128 * 4;
    L.R_lo = (int)((unsigned)c * H / (unsigned)G);
    L.R_c = (int)((unsigned)(c + 1) * H / (unsigned)G) - L.R_lo;
    L.P = 4;
    const int table_bytes = n_pairs * I * 2;
    const int outacc_bytes = NCW * T * OUT_ROWS * 4;
    const int ring_avail = DSMEM - table_bytes - outacc_bytes;
    const int ns = ring_avail / L.slot_bytes;
    L.NS = ns < MAXNS ? ns : MAXNS;
    const int ns2 = ring_avail / L.slot2_bytes;
    L.NS2 = ns2 < 8 ? ns2 : 8;
    const int ring_bytes1 = L.NS * L.slot_bytes;
    const int ring_bytes2 = L.NS2 * L.slot2_bytes;
    const int ring_bytes = max(ring_bytes1, ring_bytes2);
    L.ring = dsm;
    L.table = dsm + ring_bytes;
    L.outacc = reinterpret_cast<float*>(dsm + ring_bytes + table_bytes);
    L.out_stride = T * OUT_ROWS;
    const unsigned N1 = (unsigned)n_active * 4 * KB1;  // <= 16 * 384
    L.it_lo = (int)(N1 * c / (unsigned)G);
    L.it_hi = (int)(N1 * (c + 1) / (unsigned)G);
    L.nq = n_active * KB2;
  }
  const int par = S.epoch;
  float* gacc = g_acc[par];

  if (warp == NCW) {
    // ================= producer warp
    const uint64_t pol_w = policy_evict_first();
    int f = 0;
#pragma unroll 1
    for (int it = L.it_lo; it < L.it_hi; ++it, ++f) {
      const int slot = f % L.NS;
      const int seg = it / KB1, kb = it - seg * KB1;
      const int s = seg >> 2, rg = seg & 3;
      const int e = S.slot_expert[s];
      const int pst = S.slot_pstart[s], nt = S.slot_pstart[s + 1] - pst;
      if (lane == 0) {
        mbar_wait(&S.empty[slot], ((f / L.NS) & 1) ^ 1);
        mbar_expect_arrive(&S.full[slot], PIECE1 + 256);
      }
      __syncwarp();
      char* dst = L.ring + slot * L.slot_bytes;
      if (lane == 0) {
        tma_w1_g2s(dst, w1_map, rg * 128, kb, e, &S.full[slot], pol_w);
        tma_x_g2s(dst + PIECE1, x_map, token_base + S.pair_tok[pst], kb, &S.full[slot], pol_w);
      }
    }
    if constexpr (T == 1) {
      // W2 tiles go into the W1 ring as its slots drain for the last time: tile u sits at
      // 6 KB position (jstart + u) mod npos, in the slot order the consumers release them, so
      // the DRAM stream runs from W1 straight into W2 while GEMM1 drains and the barrier runs.
      {
        const int b0 = f % L.NS;
        const int jstart = w2_jstart(L, f);
        const int npos = (L.NS * L.slot_bytes) / W2_POS;
        unsigned freed = 0;
        int u = 0;
#pragma unroll 1
        for (int k = 0; k < L.NS && u < TOPK; ++k) {
          const int sl = (b0 + k) % L.NS;
          const int uses = f / L.NS + (sl < b0 ? 1 : 0);
          if (lane == 0 && uses > 0) mbar_wait(&S.empty[sl], (uses - 1) & 1);
          __syncwarp();
          freed |= 1u << sl;
          int u_end = u;
#pragma unroll 1
          for (; u_end < TOPK; ++u_end) {
            int j = jstart + u_end;
            if (j >= npos) j -= npos;
            const int lo = (j * W2_POS) / L.slot_bytes;
            const int hi = (j * W2_POS + W2_POS - 1) / L.slot_bytes;
            if (!((freed >> lo) & (freed >> hi) & 1u)) break;
          }
          const int uu = u + lane;
          if (uu < u_end) {
            int j = jstart + uu;
            if (j >= npos) j -= npos;
            const int e = S.slot_expert[uu];
            char* pdst = L.ring + j * W2_POS;
            mbar_expect_arrive(&S.full2[uu], KB2 * W2_TX_ROWS * 128);
            tma_w1_g2s(pdst, w2_map9, L.R_lo, 0, e, &S.full2[uu], pol_w);
          }
          u = u_end;
        }
      }
      producer_exit(lane);
      return;
    }
    if (lane == 0) {
      const int used = min(f, L.NS);
      for (int slot = 0; slot < used; ++slot) {
        const int last = slot + ((f - 1 - slot) / L.NS) * L.NS;
        mbar_wait(&S.empty[slot], (last / L.NS) & 1);
      }
    }
    __syncwarp();
    const int pbytes = W2_TILE_ROWS * 128;
#pragma unroll 1
    for (int q = 0; q * L.P < L.nq; ++q) {
      const int slot = q % L.NS2;
      const unsigned phase = (q / L.NS2) & 1;
      const int np = min(L.P, L.nq - q * L.P);
      if (lane == 0) {
        mbar_wait(&S.empty2[slot], phase ^ 1);
        mbar_expect_arrive(&S.full2[slot], np * W2_TX_ROWS * 128);
      }
      __syncwarp();
      if (lane < np) {
        const int u = q * L.P + lane;
        const int s = u / KB2, kb = u % KB2;
        const int e = S.slot_expert[s];
        char* pdst = L.ring + slot * L.slot2_bytes + lane * pbytes;
        tma_w1_g2s(pdst, w2_map32, L.R_lo, kb, e, &S.full2[slot], pol_w);
        tma_w1_g2s(pdst + 32 * 128, w2_map9, L.R_lo + 32, kb, e, &S.full2[slot], pol_w);
      }
    }
    producer_exit(lane);
    return;
  }

  // ================= consumers: GEMM1
  Ring w1rs;
  float chk = 0.f;
  float pend[4] = {0.f, 0.f, 0.f, 0.f};
  {
    int it = L.it_lo;
#pragma unroll 1
    while (it < L.it_hi) {
      const int seg = it / KB1;
      const int it_end = min(L.it_hi, (seg + 1) * KB1);
      const int s = seg >> 2, rg = seg & 3;
      const int nt = S.slot_pstart[s + 1] - S.slot_pstart[s];
      if (nt <= 8)
        chk += w1_segment<1>(S, L, it, it_end, w1rs, s, rg, gacc, pend);
      else
        chk += w1_segment<2>(S, L, it, it_end, w1rs, s, rg, gacc, pend);
      it = it_end;
    }
  }

  // ================= grid barrier (monotonic 64-bit arrival counter)
  // Every split-K atomic has returned (chk is read below), so it is performed in L2 before the
  // arrival is issued. The arrival is a release and the poll an acquire (gpu scope): with the
  // CTA barriers around them, each CTA's partials happen-before the ld.cg reads of every other
  // CTA below, which the PTX memory model does not guarantee for relaxed counter accesses.
  chk += (pend[0] + pend[1]) + (pend[2] + pend[3]);
  if (__float_as_uint(chk) == 0x7fbadbadu) S.wsum[0] = 1;
  consumer_sync();
  if (tid == 0) {
    if (c == 0) g_dirty[par] = n_pairs;
    red_release_add64(&g_bar, 1ull);
    const unsigned long long target = S.bar_base + (unsigned long long)G;
    while (ld_acquire64(&g_bar) < target) {
    }
  } else {
    float4* z = reinterpret_cast<float4*>(g_acc[par ^ 1]);
    const int tot = S.dirty[par ^ 1] * (M1 / 4);
    const int lo = (int)((long long)tot * c / G), hi = (int)((long long)tot * (c + 1) / G);
    for (int i = lo + tid - 1; i < hi; i += NCT - 1) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    float4* out4 = reinterpret_cast<float4*>(L.outacc);
    for (int i = tid - 1; i < NCW * T * (OUT_ROWS / 4); i += NCT - 1)
      out4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  consumer_sync();

  // The non-signalling threads cleared both the other parity and output planes during the wait.

  // ================= SwiGLU table: a[p][j] = bf16(silu(gate_j) * up_j), canonical j
  if constexpr (T == 1) {
    // warp w builds row w (its own expert slot); no CTA-wide sync needed before GEMM2
    const float* ga = gacc + warp * M1;
    __nv_bfloat16* tab = reinterpret_cast<__nv_bfloat16*>(L.table) + warp * I;
    float u[8], gt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j = lane + 32 * i;
      const int up = (j >> 4) * 32 + (j & 1) * 16 + ((j & 15) >> 1);
      u[i] = __ldcg(ga + up);
      gt[i] = __ldcg(ga + up + 8);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float sl = gt[i] / (1.0f + expf(-gt[i]));
      tab[lane + 32 * i] = __float2bfloat16_rn(sl * u[i]);
    }
    __syncwarp();
  } else if (tid < I) {
    const int j = tid;
    const int up = (j >> 4) * 32 + (j & 1) * 16 + ((j & 15) >> 1);
    __nv_bfloat16* tab = reinterpret_cast<__nv_bfloat16*>(L.table);
#pragma unroll 1
    for (int p0 = 0; p0 < n_pairs; p0 += 8) {
      float u[8], gt[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int p = min(p0 + k, n_pairs - 1);
        u[k] = __ldcg(gacc + (size_t)p * M1 + up);
        gt[k] = __ldcg(gacc + (size_t)p * M1 + up + 8);
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (p0 + k < n_pairs) {
          const float sl = gt[k] / (1.0f + expf(-gt[k]));
          tab[(p0 + k) * I + j] = __float2bfloat16_rn(sl * u[k]);
        }
      }
    }
  }
  if constexpr (T != 1) consumer_sync();
  consumers_done_with_device_state();

  // ================= GEMM2
  Ring rs;
  if constexpr (T == 1)
    gemm2_t1(S, L, w2_jstart(L, L.it_hi - L.it_lo));
  else if (S.max_tok <= 8)
    gemm2<1>(S, L, rs);
  else
    gemm2<2>(S, L, rs);
  consumer_sync();
#pragma unroll 1
  for (int i = tid; i < T * L.R_c; i += NCT) {
    const int t = i / L.R_c, r = i - t * L.R_c;
    const int pr = L.R_lo + r;
    const int q = pr & 31;
    const int canon = (pr & ~31) + 4 * (q & 7) + (q >> 3);
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < NCW; ++w) v += L.outacc[w * L.out_stride + t * OUT_ROWS + r];
    out[(size_t)t * H + canon] = __float2bfloat16_rn(v);
  }
  __syncthreads();
}

// GEMM1 pieces [it, it_end) of one segment for one token, accumulated into acc (no flush)
__device__ __forceinline__ void w1_run(SStatic& S, const Layout& L, int it, int it_end, Ring& rs,
                                       float (&acc)[2][2]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
#pragma unroll 1
  for (; it < it_end; ++it) {
    const int slot = rs.slot;
    mbar_wait(&S.full[slot], rs.phase);
    const char* base = L.ring + slot * L.slot_bytes;
    const unsigned sbase = smem_u32(base);
    unsigned qb[2][4], qa[4][4];
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) ldmatrix4_smem(qb[kh], sbase + PIECE1 + (kh * 4 + quad8) * 16);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int achunk = (lrow_a & 7) ^ (2 * kk + acol_half);
      ldmatrix4_smem(qa[kk], sbase + (warp * 16 + lrow_a) * 128 + achunk * 16);
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int h = kk >> 1, sub = (kk & 1) * 2;
      mma_bf16_02(acc[kk & 1], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], qb[h][sub],
                  qb[h][sub + 1]);
    }
    if (lane == 0) mbar_arrive(&S.empty[slot]);
    rs.next(L.NS);
  }
}

// Everything after GEMM1 for one token: grid barrier, SwiGLU table, GEMM2, output.
__device__ __forceinline__ void t1_tail(SStatic& S, const Layout& L, float* gacc,
                                        __nv_bfloat16* __restrict__ out, int par, int dry) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  consumer_sync();
  if (tid == 0) {
    if (!dry) {
      // grid barrier: release arrival, acquire poll (see moe_fused_kernel)
      if (c == 0) g_dirty[par] = TOPK;
      red_release_add64(&g_bar, 1ull);
      const unsigned long long target = S.bar_base + (unsigned long long)G;
      while (ld_acquire64(&g_bar) < target) {
      }
    }
  } else if (!dry) {
    float4* z = reinterpret_cast<float4*>(g_acc[par ^ 1]);
    const int tot = S.dirty[par ^ 1] * (M1 / 4);
    const int lo = (int)((long long)tot * c / G), hi = (int)((long long)tot * (c + 1) / G);
    for (int i = lo + tid - 1; i < hi; i += NCT - 1) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  consumer_sync();
  // Four warps split the K blocks of each expert.  This shortens the dependent MMA chain;
  // partials stay fp32 and are reduced in fixed K-block order before the required bf16 round.
  const int kq = warp & 3, u0 = warp >> 2;
  {
    const float* ga = gacc + u0 * M1;
    __nv_bfloat16* tab = reinterpret_cast<__nv_bfloat16*>(L.table) + u0 * I + kq * 64 + lane;
    float u[8], gt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j = kq * 64 + lane + 32 * (i & 1);
      const int up = (i >> 1) * 2 * M1 + (j >> 4) * 32 + (j & 1) * 16 + (j & 14);
      const float2 v = dry ? make_float2(0.f, 0.f) : ldcg2(ga + up);
      u[i] = v.x;
      gt[i] = v.y;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float sl = gt[i] / (1.0f + expf(-gt[i]));
      tab[(i >> 1) * 2 * I + 32 * (i & 1)] = __float2bfloat16_rn(sl * u[i]);
    }
    __syncwarp();
  }
  consumers_done_with_device_state();
  {
    const int g = lane >> 2, t = lane & 3;
    const int lane8 = lane & 7, quad8 = lane >> 3;
    const int lrow_a = lane8 + ((quad8 & 1) << 3);
    const int acol_half = quad8 >> 1;
    const int npos = (L.NS * L.slot_bytes) / W2_POS;
    const int jstart = w2_jstart(L, L.nf);
    unsigned qb[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int kh = 0; kh < 2; ++kh)
        ldmatrix4(qb[i][kh], L.table + (u0 + 2 * i) * (I * 2) + kq * 128 + (kh * 4 + quad8) * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ex = u0 + 2 * i;
      int j = jstart + ex;
      if (j >= npos) j -= npos;
      const char* base = L.ring + j * W2_POS;
      if (!dry) mbar_wait(&S.full2[ex], 0);
      float acc[3][4];
#pragma unroll
      for (int m = 0; m < 3; ++m)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][q] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          unsigned qa[4];
          const int wr = kq * L.R_c + m * 16 + lrow_a;
          const int achunk = (wr & 7) ^ (2 * kk + acol_half);
          ldmatrix4(qa, base + wr * 128 + achunk * 16);
          const int h = kk >> 1, sub = (kk & 1) * 2;
          mma_bf16(acc[m], qa[0], qa[1], qa[2], qa[3], qb[i][h][sub], qb[i][h][sub + 1]);
        }
      }
      if (t == 0) {
        float* pp = L.outacc + (ex * KB2 + kq) * W2_TILE_ROWS + g;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          pp[m * 16] = acc[m][0];
          pp[m * 16 + 8] = acc[m][2];
        }
      }
    }
  }
  consumer_sync();
#pragma unroll 1
  for (int r = tid; r < L.R_c; r += NCT) {
    const int pr = L.R_lo + r;
    const int q = pr & 31;
    const int canon = (pr & ~31) + 4 * (q & 7) + (q >> 3);
    float v = 0.f;
#pragma unroll
    for (int e = 0; e < TOPK; ++e) {
      const float* pp = L.outacc + e * KB2 * W2_TILE_ROWS + r;
      const float o = __bfloat162float(__float2bfloat16_rn(
          ((pp[0] + pp[W2_TILE_ROWS]) + pp[2 * W2_TILE_ROWS]) + pp[3 * W2_TILE_ROWS]));
      v = __fadd_rn(v, __fmul_rn(o, S.pair_w[e]));
    }
    if (!dry) out[canon] = __float2bfloat16_rn(v);
  }
}

// ------------------------------------------------------------------ T == 1 kernel
// Same pipeline as moe_fused_kernel<1>, but the producer warp computes the routing on its own
// (one warp, no CTA-wide barrier) and issues its first W13 copies straight away; the consumers
// only wait on named barrier 3 for the routing results and the ring barriers.
__global__ void __launch_bounds__(NTHREADS, 1)
    moe_t1_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                  __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap w1_map,
                  const __grid_constant__ CUtensorMap x_map,
                  const __grid_constant__ CUtensorMap w2_map40,
                  const __grid_constant__ CUtensorMap w2_map41, int token_base, float route_scale) {
  __shared__ SStatic S;
  extern __shared__ __align__(1024) char dsm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  float lg = 0.f, bs = 0.f;
  if (warp < NCW) {
    lg = __ldg(logits + tid);
    bs = __ldg(bias + tid);
  }

  // ---------------- layout (constant for one token: 8 pairs)
  Layout L;
  {
    L.slot_bytes = 17 * 1024;
    L.slot2_bytes = W2_TILE_ROWS * 128 * 4;
    L.R_lo = (int)((unsigned)c * H / (unsigned)G);
    L.R_c = (int)((unsigned)(c + 1) * H / (unsigned)G) - L.R_lo;
    L.P = 4;
    constexpr int table_bytes = TOPK * I * 2;
    constexpr int outacc_bytes = TOPK * KB2 * W2_TILE_ROWS * 4;
    constexpr int ring_avail = DSMEM - table_bytes - outacc_bytes;
    constexpr int ns = ring_avail / (17 * 1024);
    L.NS = ns < MAXNS ? ns : MAXNS;
    L.NS2 = 8;
    const int ring_bytes = L.NS * L.slot_bytes;
    L.ring = dsm;
    L.table = dsm + ring_bytes;
    L.outacc = reinterpret_cast<float*>(dsm + ring_bytes + table_bytes);
    L.out_stride = OUT_ROWS;
    constexpr unsigned NSH = 4 * KB1, NMN = (TOPK - 1) * 4 * KB1;
    L.sh_lo = (int)(NSH * c / (unsigned)G);
    L.nsh = (int)(NSH * (c + 1) / (unsigned)G) - L.sh_lo;
    L.it_lo = (int)NSH + (int)(NMN * c / (unsigned)G);
    L.it_hi = (int)NSH + (int)(NMN * (c + 1) / (unsigned)G);
    L.nf = L.nsh + L.it_hi - L.it_lo;
    L.nq = TOPK * KB2;
  }

  if (warp == NCW) {
    // ================= producer warp: ring barriers, then the weight stream once routed
    // Barrier init, its fence and the descriptor prefetches come before any global load: the
    // release fence would otherwise wait for that load's DRAM round trip, delaying the
    // descriptor fetch the first (top-1) copies need.
    if (lane < MAXNS) {
      mbar_init(&S.full[lane], 1);
      mbar_init(&S.empty[lane], NCW);
    }
    mbar_init(&S.full2[lane], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w1_map))
                   : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&x_map)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w2_map40))
                   : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w2_map41))
                   : "memory");
    }
    const unsigned long long w1_raw = ld_relaxed64(&g_w1);
    __syncwarp();
    asm volatile("bar.arrive 4, %0;" ::"n"(NTHREADS) : "memory");  // ring barriers ready
    // first ring fill, one piece per lane: armed before the routing is known, issued right
    // after it (the issue code runs once dry first so it is in the instruction cache)
    const uint64_t pol_w = policy_evict_first();
    const uint64_t pol_x = pol_w;
    const int n0 = min(L.NS, L.nf);
    const int it0 = lane < L.nsh ? L.sh_lo + lane : L.it_lo + (lane - L.nsh);
    const int seg0 = it0 / KB1, kb0 = it0 - seg0 * KB1;
    char* dst0 = L.ring + lane * L.slot_bytes;
    if (lane < n0) mbar_expect_arrive(&S.full[lane], PIECE1 + 128);
#pragma unroll 1
    for (int pass = 1; pass < 3; ++pass) {
      if (pass == 1) asm volatile("bar.sync 5, %0;" ::"n"(64) : "memory");
      if (pass == 2) asm volatile("bar.sync 3, %0;" ::"n"(NTHREADS) : "memory");
      const unsigned go = (pass == 1 && lane < L.nsh) || (pass == 2 && lane >= L.nsh && lane < n0);
      const int e = go ? S.slot_expert[seg0 >> 2] : 0;
      const uint64_t desc_w = reinterpret_cast<uint64_t>(&w1_map);
      const uint64_t desc_x = reinterpret_cast<uint64_t>(&x_map);
      asm volatile(
          "{\n .reg .pred p;\n setp.ne.u32 p, %9, 0;\n"
          " @p "
          "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_"
          "hint "
          "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n"
          " @p "
          "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_"
          "hint "
          "[%10], [%8, {%2, %11, %4}], [%6], %12;\n}\n" ::"r"(smem_u32(dst0)),
          "l"(desc_w), "r"(0), "r"((seg0 & 3) * 128), "r"(kb0), "r"(e),
          "r"(smem_u32(&S.full[lane])), "l"(pol_w), "l"(desc_x), "r"(go),
          "r"(smem_u32(dst0 + PIECE1)), "r"(token_base), "l"(pol_x)
          : "memory");
    }
    int f = n0;
#pragma unroll 1
    for (; f < L.nf; ++f) {
      const int it = f < L.nsh ? L.sh_lo + f : L.it_lo + (f - L.nsh);
      const int slot = f % L.NS;
      const int seg = it / KB1, kb = it - seg * KB1;
      const int s = seg >> 2, rg = seg & 3;
      if (lane == 0) {
        const int e = S.slot_expert[s];
        mbar_wait(&S.empty[slot], ((f / L.NS) & 1) ^ 1);
        mbar_expect_arrive(&S.full[slot], PIECE1 + 128);
        char* dst = L.ring + slot * L.slot_bytes;
        tma_w1_g2s(dst, w1_map, rg * 128, kb, e, &S.full[slot], pol_w);
        tma_x_g2s(dst + PIECE1, x_map, token_base, kb, &S.full[slot], pol_x);
      }
    }
    // The W2 requests are held back by one L2 round trip (count this CTA's W13 as issued, then
    // read the counter until it shows an increment of this launch): measured ~0.3 us faster
    // than issuing W2 right away (which crowds the W13 tail) and than waiting until every CTA
    // has issued its W13 (which idles the early CTAs' copy engines)
    const unsigned long long w1_target =
        (w1_raw / (unsigned long long)G) * (unsigned long long)G + 1ull;
    if (lane == 0) {
      red_relaxed_add64(&g_w1, 1ull);
      while (ld_relaxed64(&g_w1) < w1_target) {
      }
    }
    __syncwarp();
    // W2 tiles go into the W1 ring as its slots drain for the last time (see moe_fused_kernel)
    {
      const int b0s = f % L.NS;
      const int jstart = w2_jstart(L, f);
      const int npos = (L.NS * L.slot_bytes) / W2_POS;
      unsigned freed = 0;
      int u = 0;
#pragma unroll 1
      for (int k = 0; k < L.NS && u < TOPK; ++k) {
        const int sl = (b0s + k) % L.NS;
        const int uses = f / L.NS + (sl < b0s ? 1 : 0);
        if (lane == 0 && uses > 0) mbar_wait(&S.empty[sl], (uses - 1) & 1);
        __syncwarp();
        freed |= 1u << sl;
        int u_end = u;
#pragma unroll 1
        for (; u_end < TOPK; ++u_end) {
          int j = jstart + u_end;
          if (j >= npos) j -= npos;
          const int lo = (j * W2_POS) / L.slot_bytes;
          const int hi = (j * W2_POS + W2_POS - 1) / L.slot_bytes;
          if (!((freed >> lo) & (freed >> hi) & 1u)) break;
        }
        const int uu = u + lane;
        if (uu < u_end) {
          int j = jstart + uu;
          if (j >= npos) j -= npos;
          char* pdst = L.ring + j * W2_POS;
          mbar_expect_arrive(&S.full2[uu], KB2 * L.R_c * 128);
          const CUtensorMap& w2_map = L.R_c == 40 ? w2_map40 : w2_map41;
          tma_w1_g2s(pdst, w2_map, L.R_lo, 0, S.slot_expert[uu], &S.full2[uu], pol_w);
        }
        u = u_end;
      }
    }
    producer_exit(lane);
    return;
  }

  // ================= consumers: routing (thread e = expert e)
  {
    // tau = min over the 8 warps of the warp maximum key bounds the 8th largest key from below,
    // so the top 8 are among the keys >= tau (typically ~8-16); those are ranked exactly
    // (larger key first, lower expert on ties)
    unsigned* cand_k = reinterpret_cast<unsigned*>(L.table);  // the table is free until GEMM2
    int* cand_e = reinterpret_cast<int*>(cand_k + E);
    float* cand_s = reinterpret_cast<float*>(cand_e + E);
    if (tid == 0) S.n_active = 0;  // selected-expert counter
    const float sv = 1.0f / (1.0f + expf(-lg));
    const unsigned key = f2key(sv + __bfloat162float(__float2bfloat16_rn(bs)));
    const unsigned wm = __reduce_max_sync(0xffffffffu, key);
    if (lane == 0) S.wsum[warp] = (int)wm;
    consumer_sync();
    const unsigned wv = lane < NCW ? (unsigned)S.wsum[lane] : 0u;
    const unsigned top = __reduce_max_sync(0xffffffffu, wv);
    const unsigned tv = lane < NCW ? wv : 0xffffffffu;
    const unsigned tau = __reduce_min_sync(0xffffffffu, tv);
    const int wtop = __ffs(__ballot_sync(0xffffffffu, lane < NCW && wv == top)) - 1;
    if (warp == wtop) {
      const unsigned bt = __ballot_sync(0xffffffffu, key == top);
      if (lane == __ffs(bt) - 1) {
        S.slot_expert[0] = tid;
        S.tk_w[0][0] = sv;
      }
      asm volatile("bar.arrive 5, %0;" ::"n"(64) : "memory");
    }
    const bool isc = key >= tau;
    const unsigned bal = __ballot_sync(0xffffffffu, isc);
    int cbase = 0;
    if (lane == 0) cbase = atomicAdd(&S.n_active, __popc(bal));
    cbase = __shfl_sync(0xffffffffu, cbase, 0);
    if (isc) {
      const int q = cbase + __popc(bal & ((1u << lane) - 1u));
      cand_k[q] = key;
      cand_e[q] = tid;
      cand_s[q] = sv;
    }
    consumer_sync();
    const int nc = S.n_active;
#pragma unroll 1
    for (int c0 = 0; c0 < nc; c0 += NCT / 8) {
      const int ci = c0 + (tid >> 3), sub = tid & 7;
      int rr = 0;
      unsigned ki = 0u;
      int ei = 0;
      if (ci < nc) {
        ki = cand_k[ci];
        ei = cand_e[ci];
#pragma unroll 4
        for (int j = sub; j < nc; j += 8) {
          const unsigned kj = cand_k[j];
          rr += (kj > ki) | ((kj == ki) & (cand_e[j] < ei));
        }
      }
      rr += __shfl_xor_sync(0xffffffffu, rr, 1);
      rr += __shfl_xor_sync(0xffffffffu, rr, 2);
      rr += __shfl_xor_sync(0xffffffffu, rr, 4);
      if (ci < nc && sub == 0 && rr > 0 && rr < TOPK) {
        S.slot_expert[rr] = ei;
        S.tk_w[0][rr] = cand_s[ci];
      }
    }
    if (tid <= TOPK) S.slot_pstart[tid] = tid;
    asm volatile("bar.arrive 3, %0;" ::"n"(NTHREADS) : "memory");  // release the producer
  }
  // The producer can issue the remaining W13 pieces as soon as barrier 3 is released. Finish
  // the exact BF16 routing-weight normalization behind those copies instead of on the GEMM2 tail.
  consumer_sync();
  {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sum += S.tk_w[0][k];
    if (lane == 0)
      S.pair_w[warp] =
          __bfloat162float(__float2bfloat16_rn((S.tk_w[0][warp] * route_scale) / (sum + 1e-20f)));
  }
  if (tid == 0) {
    const unsigned long long v = wait_device_state(G);
    const unsigned long long ep = v / (2ull * (unsigned long long)G);
    S.bar_base = v;
    S.epoch = (int)(ep & 1);
    S.tlslot = (int)(ep & 63);
    S.dirty[0] = g_dirty[0];
    S.dirty[1] = g_dirty[1];
  }
  asm volatile("bar.sync 4, %0;" ::"n"(NTHREADS) : "memory");  // ring barriers initialised
  const int par = S.epoch;
  float* gacc = g_acc[par];
  Ring w1rs;
  float chk = 0.f;
  float pend[4] = {0.f, 0.f, 0.f, 0.f};
  {
    float acc[2][2] = {};
    int cur = -1;
#pragma unroll 1
    for (int f = 0; f < L.nf;) {
      const int it = f < L.nsh ? L.sh_lo + f : L.it_lo + (f - L.nsh);
      const int seg = it / KB1;
      const int f_end = f < L.nsh ? L.nsh : L.nf;
      const int fe = min(f_end, f + ((seg + 1) * KB1 - it));
      if (seg != cur) {
        if (cur >= 0 && (lane & 3) == 0) {
          chk += pend[0] + pend[1];
          float* d = gacc + (size_t)(cur >> 2) * M1 + (cur & 3) * 128 + warp * 16 + 2 * (lane >> 2);
          atom_add_v2_f32(d, acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], pend[0], pend[1]);
        }
        cur = seg;
#pragma unroll
        for (int q = 0; q < 2; ++q) acc[0][q] = acc[1][q] = 0.f;
      }
      w1_run(S, L, f, fe, w1rs, acc);
      f = fe;
    }
    if (cur >= 0 && (lane & 3) == 0) {
      chk += pend[0] + pend[1];
      float* d = gacc + (size_t)(cur >> 2) * M1 + (cur & 3) * 128 + warp * 16 + 2 * (lane >> 2);
      atom_add_v2_f32(d, acc[0][0] + acc[1][0], acc[0][1] + acc[1][1], pend[0], pend[1]);
    }
  }
  chk += (pend[0] + pend[1]) + (pend[2] + pend[3]);
  if (__float_as_uint(chk) == 0x7fbadbadu) S.wsum[0] = 1;
  t1_tail(S, L, gacc, out, par, 0);
}

}  // namespace kA
#ifndef T6_W2S
#define T6_W2S 6
#endif
namespace kB {
// Fused BF16 MoE (DeepSeekV3 routing, SwiGLU) for decode sizes on Blackwell (sm_103).
// One persistent cooperative kernel, one CTA per SM. Warp roles:
//   warps 0-7  consumers (mma.sync, tokens on N)
//   warp  8    producer: bulk copies of 16 KB weight pieces (+ x rows) into an smem ring
//   warp  9    signaller: publishes "this CTA finished expert s" (fence + red on a counter)
//   warp 10    table builder: waits for an expert's GEMM1 to be complete grid-wide, then
//              builds its bf16 SwiGLU activations in smem
// Flow: every CTA computes the routing for the (<=16) tokens of this launch. GEMM1 work
// (expert, k-block, 128-row group) is dealt round-robin over CTAs so all CTAs finish the
// experts in order; fp32 split-K partials go to a global accumulator with red.add. Each CTA
// owns a contiguous slice of the 6144 GEMM2 rows for all experts and streams them after its
// GEMM1 work; GEMM2 of an expert only waits for that expert's activations, so the weight
// stream never stops on a grid-wide barrier.
// Fragment trick: the dot product does not care about k order, so each thread feeds the MMA
// from contiguous 16 B smem chunks (same physical k permutation for A and B).

constexpr int E = 256, TOPK = 8, H = 6144, I = 256, M1 = 512, KB1 = H / 64, KB2 = I / 64;
constexpr int ITEMS_PER_EXPERT = KB1 * 4;  // (k-block, 128-row group) pieces of W13
constexpr int TCH = 16;                    // tokens per launch
constexpr int NCW = 8;                     // consumer warps
constexpr int NCT = NCW * 32;
constexpr int WARP_PROD = NCW, WARP_SIG = NCW + 1, WARP_TAB = NCW + 2;
constexpr int NTHREADS = (NCW + 3) * 32;
constexpr int MAXNS = 16;
constexpr int PIECE1 = 128 * 128;  // 128 rows x 64 bf16
constexpr int MAX_PAIRS = TCH * TOPK;
constexpr int MAX_ACTIVE = TCH * TOPK;
constexpr int OUT_ROWS = 48;  // padded W2 rows per CTA in the output accumulator
constexpr int OUT_PLANES = NCW;
constexpr int OUTB = OUT_PLANES * TCH * OUT_ROWS * 4;
#ifndef KB_SUB
#define KB_SUB 2
#endif
constexpr int TBARB = MAX_ACTIVE * 8;  // per-expert "SwiGLU rows ready" mbarriers
constexpr int SMEM_LIMIT = 232448;
constexpr int FSTAGE_BYTES = NCW * 2 * 8 * 16 * 4;  // warps x buffers x tokens x rows x fp32

struct SStatic {
  unsigned long long full[MAXNS];
  unsigned long long empty[MAXNS];
  // T=6 uses a separate three-stage W2 ring. T=10 deliberately keeps the
  // proven shared ring because the late separate-ring variant replay-raced.
  unsigned long long full2[8];
  unsigned long long empty2[8];
  unsigned char epst[E];  // first pair index of each active expert
  unsigned mask[E];
  int slot_expert[E];
  int slot_pstart[E + 1];
  int pair_tok[MAX_PAIRS];
  float pair_w[MAX_PAIRS];
  int done_cnt[MAX_ACTIVE];
  int wsum[NCW];
  int n_active, n_pairs, max_tok, par;
  int tbl_ready;
  unsigned long long stage_bar;
  int tlslot;
  // T = 6: distinct per-token top-1 experts take the first expert slots (early W13 issue)
  alignas(16) unsigned char fidx[E];  // slot of an early expert, 0xff otherwise
  int top1[TCH];
  int fexp[TCH];
  int n_first;
};
using kA::gtime;
constexpr int DSMEM = SMEM_LIMIT - 6144;
static_assert(sizeof(SStatic) <= 6144, "static smem");

__device__ float g_acc[2][MAX_PAIRS * M1];  // GEMM1 fp32 accumulators, by launch parity
__device__ unsigned g_cnt[2][MAX_ACTIVE];   // per-expert "CTA done" counters, by parity
__device__ float g_ack[152 * NCW];          // per-CTA/warp completion acknowledgements
__device__ unsigned long long g_epoch;      // CTA exits so far (launches so far x grid size)
__device__ int g_prev[2][2];                // {n_pairs, n_active} of the launch of a parity

// Launches of kernel B serialize on its device state (g_acc, g_cnt, g_prev, g_epoch), as in kernel
// A (kA::wait_device_state): each CTA counts its exit in g_epoch with a release once it is done
// with the state (the table builder, once every table is built and the consumers' GEMM1 partials
// are flushed, named barrier 11), and a launch starts on it once the count is a multiple of G. A
// CTA counts its exit only after its table builder saw every expert's counter reach G, i.e. after
// every CTA of its launch read the count. Returns the count: G x the launches before this one.
__device__ __forceinline__ unsigned long long wait_device_state(unsigned long long G) {
  unsigned long long v = kA::ld_acquire64(&g_epoch);
  while (v % G != 0ull) {
    __nanosleep(128);
    v = kA::ld_acquire64(&g_epoch);
  }
  return v;
}

// ---------------------------------------------------------------- PTX helpers
// n / d for 0 <= n < 2^24, 0 < d: float estimate, then exact integer correction
__device__ __forceinline__ int fdiv(int n, int d) {
  int q = __float2int_rz(__fdividef((float)n, (float)d));
  q -= (q * d > n);
  q += ((q + 1) * d <= n);
  return q;
}
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(unsigned long long* b, unsigned cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_arrive(unsigned long long* b, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(tx)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes,
                                         unsigned long long* b, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 1, %0;" ::"n"(NCT) : "memory");
}
__device__ __forceinline__ uint4 lds128(const char* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ unsigned f2key(float f) {
  unsigned b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// The six T=6 activation rows for k-block kb.
__device__ __forceinline__ void tma_x6_g2s(void* dst, const CUtensorMap& map, int kb, int t0,
                                           unsigned long long* b) {
  const uint64_t desc = reinterpret_cast<uint64_t>(&map);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(desc), "r"(kb * 64), "r"(t0), "r"(smem_u32(b))
      : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_acquire_cta_s(const int* p) {
  int v;
  asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ int ld_volatile_s(const int* p) {
  int v;
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_cta_s(int* p, int v) {
  asm volatile("st.release.cta.shared.u32 [%0], %1;" ::"r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ void red_add_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ float atom_add_f32(float* p, float v) {
  float r;
  asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(v) : "memory");
  return r;
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_add_u32(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ float4 ldcg4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
// D = A * B + D, m16n8k16, bf16 inputs, fp32 accumulate
__device__ __forceinline__ void mma_bf16(float* d, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One 16-row x 64-k block: rows g / g+8 (16 B chunks t and t+4 of each), B chunks of one n-block.
__device__ __forceinline__ void mma_k64(float* d, const uint4 (&qa)[2][2], const uint4 (&qb)[2]) {
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int h = s >> 1, sub = (s & 1) * 2;
    mma_bf16(d, u4get(qa[0][h], sub), u4get(qa[1][h], sub), u4get(qa[0][h], sub + 1),
             u4get(qa[1][h], sub + 1), u4get(qb[h], sub), u4get(qb[h], sub + 1));
  }
}
__device__ __forceinline__ void lda(uint4 (&qa)[2][2], const char* p) {
  qa[0][0] = lds128(p);
  qa[0][1] = lds128(p + 64);
  qa[1][0] = lds128(p + 8 * 128);
  qa[1][1] = lds128(p + 8 * 128 + 64);
}
__device__ __forceinline__ float silu_mul(float gt, float up) {
  return gt / (1.0f + expf(-gt)) * up;
}

struct Layout {
  int NS, slot_bytes, NS2, slot2_bytes, sub_bytes, R_lo, R_c;
  int P, P2, N1, nq, out_stride;
  char* ring;
  char* table;
  float* outacc;
  unsigned long long* tbar;
  float* fstage;  // T=6 per-warp double-buffered split-K bulk-reduction staging
};

struct Ring {
  int slot = 0;
  unsigned phase = 0;
  __device__ __forceinline__ void next(int ns) {
    if (++slot == ns) {
      slot = 0;
      phase ^= 1u;
    }
  }
};

// ------------------------------------------------------------- GEMM1 (consumers)
// Warp w owns rows w*16 + {g, g+8} of each 128-row group; NB n-blocks of 8 tokens.
template <int NB, bool SWZ = false, bool ACK = false, bool XTOK = false>
__device__ __forceinline__ void gemm1(SStatic& S, const Layout& L, Ring& rs, float* gacc) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
  const int G = gridDim.x;
  const int aoff = (warp * 16 + g) * 128 + t * 16;
  float acc[NB][4];
  int boff[NB];
  int cur_s = -1, cur_rg = 0, pst = 0, nt = 0;
  // Partials are flushed with returning atomics; the returned values are consumed at the next
  // flush, which proves the earlier atomics were performed at L2 before the expert is
  // reported done (no memory fence on the consumer path).
  int signalled = 0;  // experts [0, signalled) reported to the signaller
  float pend[NB][4];
#pragma unroll
  for (int n = 0; n < NB; ++n)
#pragma unroll
    for (int q = 0; q < 4; ++q) pend[n][q] = 0.f;
  // T=6 ACK path: each warp double-buffers its 16-row partials in shared memory and issues
  // one bulk async reduction per live token row.  wait_group supplies a real completion point
  // before expert readiness is published, without a returning atomic for every scalar partial.
  int fbuf = 0, s_newest = -1;
  auto bulk_signal = [&](int s_end, bool all) {
    if constexpr (XTOK) {
      if (warp == 0 && lane == 0) {
        if (all)
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        else
          asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
        for (int w = 0; w < NCW; ++w) st_release_cta_s(&S.done_cnt[w], s_end);
      }
      consumer_sync();
    } else {
      if (lane == 0) {
        if (all)
          asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        else
          asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
        atomicExch(&S.done_cnt[warp], s_end);
      }
      __syncwarp();
    }
  };
  auto resolve_and_signal = [&](int s_end) {
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < NB; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) sum += pend[n][q];
    const int bad = __any_sync(0xffffffffu, sum != sum) ? 1 : 0;
    if (lane == 0) atomicExch(&S.done_cnt[warp], s_end | (bad << 16));
    signalled = max(signalled, s_end);
  };
  auto flush = [&]() {
    float* dst = gacc + (size_t)pst * M1 + cur_rg * 128 + warp * 16 + g;
#pragma unroll
    for (int n = 0; n < NB; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int i = n * 8 + 2 * t + (q & 1);
        if constexpr (ACK) {
          if (i < nt && !(lane == 0 && n == 0 && q == 0))
            red_add_f32(dst + (size_t)i * M1 + (q >> 1) * 8, acc[n][q]);
          if (!(lane == 0 && n == 0 && q == 0)) pend[n][q] = 0.f;
        } else {
          pend[n][q] =
              (i < nt) ? atom_add_f32(dst + (size_t)i * M1 + (q >> 1) * 8, acc[n][q]) : 0.f;
        }
      }
    if constexpr (ACK) {
      __syncwarp();
      if (lane == 0) pend[0][0] = atom_add_f32(dst, acc[0][0]);
    }
  };
  auto flush_bulk = [&]() {
    if constexpr (XTOK) {
      float* stg = L.fstage + fbuf * (8 * 128);
      if (warp == 0 && lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      consumer_sync();
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int i = 2 * t + (q & 1);
        if (i < nt) stg[i * 128 + warp * 16 + g + (q >> 1) * 8] = acc[0][q];
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      consumer_sync();
      if (warp == 0 && lane == 0) {
        for (int i = 0; i < nt; ++i)
          asm volatile(
              "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 512;" ::"l"(
                  gacc + (size_t)(pst + i) * M1 + cur_rg * 128),
              "r"(smem_u32(stg + i * 128))
              : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    } else {
      float* stg = L.fstage + (warp * 2 + fbuf) * (8 * 16);
      if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      __syncwarp();
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int i = 2 * t + (q & 1);
        if (i < nt) stg[i * 16 + g + (q >> 1) * 8] = acc[0][q];
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) {
        for (int i = 0; i < nt; ++i)
          asm volatile(
              "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 64;" ::"l"(
                  gacc + (size_t)(pst + i) * M1 + cur_rg * 128 + warp * 16),
              "r"(smem_u32(stg + i * 16))
              : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
    fbuf ^= 1;
  };
  int sub = 0;
#pragma unroll 2
  for (int p = blockIdx.x; p < L.N1; p += G) {
    const int s = p / ITEMS_PER_EXPERT;
    const int q = p - s * ITEMS_PER_EXPERT;
    const int kb = q >> 2, rg = q & 3;
    if (s != cur_s || rg != cur_rg) {
      if (cur_s >= 0) {
        if constexpr (ACK) {
          if (s_newest >= 0) bulk_signal(s_newest, false);
          flush_bulk();
          s_newest = cur_s;
        } else {
          resolve_and_signal(cur_s);
          flush();
        }
      }
      cur_s = s;
      cur_rg = rg;
      pst = S.slot_pstart[s];
      nt = S.slot_pstart[s + 1] - pst;
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        boff[n] = PIECE1 + min(n * 8 + g, nt - 1) * 128 + t * 16;
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
      }
    }
    (void)kb;
    if (sub == 0) mbar_wait(&S.full[rs.slot], rs.phase);
    const char* base = L.ring + rs.slot * L.slot_bytes + sub * L.sub_bytes;
    if constexpr (SWZ) {
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        // XTOK: the x tile holds every token of the launch, row = token
        const int brow =
            XTOK ? S.pair_tok[pst + min(n * 8 + lane8, nt - 1)] : min(n * 8 + lane8, nt - 1);
        unsigned qb_all[2][4];
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          const int bchunk = (brow & 7) ^ (kh * 4 + quad8);
          kA::ldmatrix4(qb_all[kh], base + PIECE1 + brow * 128 + bchunk * 16);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          unsigned qa[4];
          const int achunk = (lrow_a & 7) ^ (2 * kk + acol_half);
          kA::ldmatrix4(qa, base + (warp * 16 + lrow_a) * 128 + achunk * 16);
          const int h = kk >> 1, qsub = (kk & 1) * 2;
          mma_bf16(acc[n], qa[0], qa[1], qa[2], qa[3], qb_all[h][qsub], qb_all[h][qsub + 1]);
        }
      }
    } else {
      uint4 qa[2][2];
      lda(qa, base + aoff);
#pragma unroll
      for (int n = 0; n < NB; ++n) {
        uint4 qb[2];
        qb[0] = lds128(base + boff[n]);
        qb[1] = lds128(base + boff[n] + 64);
        mma_k64(acc[n], qa, qb);
      }
    }
    if (++sub == KB_SUB || p + G >= L.N1) {
      mbar_arrive(&S.empty[rs.slot]);  // every consumer thread releases its own reads
      sub = 0;
      rs.next(L.NS);
    }
  }
  if constexpr (ACK) {
    if (cur_s >= 0) {
      if (s_newest >= 0) bulk_signal(s_newest, false);
      flush_bulk();
    }
    bulk_signal(S.n_active, true);
  } else {
    if (cur_s >= 0) {
      resolve_and_signal(cur_s);
      flush();
    }
    resolve_and_signal(S.n_active);
  }
}

// ------------------------------------------------------------- GEMM2 (consumers)
// The three reachable 16-row tiles of every expert are flattened and striped over all
// eight consumer warps. Each warp has a private fp32 output plane, avoiding atomics while
// balancing 3*n_active tasks instead of leaving the fourth-tile warps idle.
template <int NB, bool SWZ = false, bool SEP_W2 = false, bool EXACT_R = false>
__device__ __forceinline__ void gemm2(SStatic& S, const Layout& L, Ring& rs) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int lane8 = lane & 7, quad8 = lane >> 3;
  const int lrow_a = lane8 + ((quad8 & 1) << 3);
  const int acol_half = quad8 >> 1;
  float acc[NB][4];
#pragma unroll 1
  for (int u0 = 0; u0 < L.nq; u0 += (SEP_W2 ? L.P2 : L.P)) {
    if constexpr (SEP_W2)
      mbar_wait(&S.full2[rs.slot], rs.phase);
    else
      mbar_wait(&S.full[rs.slot], rs.phase);
    const int u1 = min(u0 + (SEP_W2 ? L.P2 : L.P), L.nq);
    const char* fbase = L.ring + rs.slot * (SEP_W2 ? L.slot2_bytes : L.slot_bytes);
    if constexpr (SWZ) {
      // one piece per expert: flattened rows R = kb * 41 + r of the 4D copy (128B swizzle on R)
#pragma unroll 1
      for (int s = u0; s < u1; ++s) {
        const int mt = (warp - 3 * s) & (NCW - 1);
        if (mt >= 3) continue;
        const int pst = S.slot_pstart[s], nt = S.slot_pstart[s + 1] - pst;
        mbar_wait(&L.tbar[s], 0);  // SwiGLU rows of expert s are in the table
        const char* base = fbase + (s - u0) * kA::W2_POS;
#pragma unroll
        for (int n = 0; n < NB; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
        for (int kb = 0; kb < KB2; ++kb) {
          unsigned qb_all[NB][2][4];
#pragma unroll
          for (int n = 0; n < NB; ++n) {
            const int brow = min(n * 8 + lane8, nt - 1);
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
              int bchunk = kh * 4 + quad8;
              if constexpr (SEP_W2) bchunk ^= ((pst + brow) & 7);
              kA::ldmatrix4(qb_all[n][kh],
                            L.table + (pst + brow) * (I * 2) + kb * 128 + bchunk * 16);
            }
          }
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            unsigned qa[4];
            const int wr = kb * (EXACT_R ? L.R_c : kA::W2_TX_ROWS) + mt * 16 + lrow_a;
            const int achunk = (wr & 7) ^ (2 * kk + acol_half);
            kA::ldmatrix4(qa, base + wr * 128 + achunk * 16);
#pragma unroll
            for (int n = 0; n < NB; ++n) {
              const int h = kk >> 1, sub = (kk & 1) * 2;
              mma_bf16(acc[n], qa[0], qa[1], qa[2], qa[3], qb_all[n][h][sub],
                       qb_all[n][h][sub + 1]);
            }
          }
        }
#pragma unroll
        for (int n = 0; n < NB; ++n) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int i = n * 8 + 2 * t + (c & 1);
            const int r = mt * 16 + g + (c >> 1) * 8;
            if (i < nt && r < L.R_c) {
              const float o = __bfloat162float(__float2bfloat16_rn(acc[n][c]));
              const int plane = SEP_W2 ? (rs.slot * L.P2 + (s - u0)) : warp;
              float* oacc = L.outacc + plane * L.out_stride + mt * 16 + g;
              oacc[S.pair_tok[pst + i] * OUT_ROWS + (c >> 1) * 8] += o * S.pair_w[pst + i];
            }
          }
        }
      }
      if constexpr (SEP_W2)
        mbar_arrive(&S.empty2[rs.slot]);
      else
        mbar_arrive(&S.empty[rs.slot]);
      rs.next(SEP_W2 ? L.NS2 : L.NS);
      continue;
    }
#pragma unroll 1
    for (int u = u0; u < u1; ++u) {
      const int s = u >> 2, kb = u & 3;
      const int mt = (warp - 3 * s) & (NCW - 1);
      if (mt >= 3) continue;
      const int pst = S.slot_pstart[s], nt = S.slot_pstart[s + 1] - pst;
      if (kb == 0) {
        mbar_wait(&L.tbar[s], 0);  // SwiGLU rows of expert s are in the table
#pragma unroll
        for (int n = 0; n < NB; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
      }
      const int piece_bytes = SWZ ? kA::W2_TILE_ROWS * 128 : L.R_c * 128;
      const char* base = fbase + (u - u0) * piece_bytes;
      const int r0 = min(mt * 16 + g, L.R_c - 1);
      const int r1 = min(mt * 16 + g + 8, L.R_c - 1);
      if constexpr (SWZ) {
        unsigned qb_all[NB][2][4];
#pragma unroll
        for (int n = 0; n < NB; ++n) {
          const int brow = min(n * 8 + lane8, nt - 1);
#pragma unroll
          for (int kh = 0; kh < 2; ++kh) {
            const int bchunk = kh * 4 + quad8;
            kA::ldmatrix4(qb_all[n][kh], L.table + (pst + brow) * (I * 2) + kb * 128 + bchunk * 16);
          }
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          unsigned qa[4];
          const int wr = mt * 16 + lrow_a;
          const int achunk = (wr & 7) ^ (2 * kk + acol_half);
          kA::ldmatrix4(qa, base + wr * 128 + achunk * 16);
#pragma unroll
          for (int n = 0; n < NB; ++n) {
            const int h = kk >> 1, sub = (kk & 1) * 2;
            mma_bf16(acc[n], qa[0], qa[1], qa[2], qa[3], qb_all[n][h][sub], qb_all[n][h][sub + 1]);
          }
        }
      } else {
        uint4 qa[2][2];
        qa[0][0] = lds128(base + r0 * 128 + t * 16);
        qa[0][1] = lds128(base + r0 * 128 + t * 16 + 64);
        qa[1][0] = lds128(base + r1 * 128 + t * 16);
        qa[1][1] = lds128(base + r1 * 128 + t * 16 + 64);
#pragma unroll
        for (int n = 0; n < NB; ++n) {
          const char* tb = L.table + (pst + min(n * 8 + g, nt - 1)) * (I * 2) + kb * 128 + t * 16;
          uint4 qb[2];
          qb[0] = lds128(tb);
          qb[1] = lds128(tb + 64);
          mma_k64(acc[n], qa, qb);
        }
      }
      if (kb == KB2 - 1) {
#pragma unroll
        for (int n = 0; n < NB; ++n) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int i = n * 8 + 2 * t + (c & 1);
            const int r = mt * 16 + g + (c >> 1) * 8;
            if (i < nt && r < L.R_c) {
              const float o = __bfloat162float(__float2bfloat16_rn(acc[n][c]));
              float* oacc = L.outacc + warp * L.out_stride + mt * 16 + g;
              oacc[S.pair_tok[pst + i] * OUT_ROWS + (c >> 1) * 8] += o * S.pair_w[pst + i];
            }
          }
        }
      }
    }
    mbar_arrive(&S.empty[rs.slot]);  // every consumer thread releases its own reads
    rs.next(L.NS);
  }
}

// ------------------------------------------------------------------ kernel
template <int TT>
__global__ void __launch_bounds__(NTHREADS, 1)
    moe_fused_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                     const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w1,
                     const __nv_bfloat16* __restrict__ w2, __nv_bfloat16* __restrict__ out,
                     const __grid_constant__ CUtensorMap w1_map,
                     const __grid_constant__ CUtensorMap x_map1,
                     const __grid_constant__ CUtensorMap w2_map32,
                     const __grid_constant__ CUtensorMap w2_map9, int Targ, int token_base,
                     float route_scale) {
  __shared__ SStatic S;
  extern __shared__ __align__(1024) char dsm[];
  const int T = TT ? TT : Targ;
  constexpr bool ROUTE_COALESCED = TT == 6;
  constexpr int ROUTE_WARPS = TT == 10 ? 10 : NCW;
  constexpr int ROUTE_ROUNDS = TT == 6 || TT == 10 ? 1 : 2;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, c = blockIdx.x;
  // T = 6: one W2 plane per ring stage (T6_W2S single-expert stages); T = 10: 3 stages x 2
  const int out_planes = TT == 6 ? T6_W2S : TT == 10 ? 6 : OUT_PLANES;
  const int outb = out_planes * T * OUT_ROWS * 4;
  float* outacc = reinterpret_cast<float*>(dsm + DSMEM - outb);
  // Start the first routing pass before shared-state initialization so its cold DRAM latency
  // overlaps the fixed launch setup. Each routing warp owns one token and each lane owns 8 experts.
  float4 pl[4] = {};
  if (warp < ROUTE_WARPS && warp < T) {
    const float* lp = logits + warp * E + lane * (ROUTE_COALESCED ? 4 : 8);
    const float* bp = bias + lane * (ROUTE_COALESCED ? 4 : 8);
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(pl[0].x), "=f"(pl[0].y), "=f"(pl[0].z), "=f"(pl[0].w)
                 : "l"(lp));
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(pl[1].x), "=f"(pl[1].y), "=f"(pl[1].z), "=f"(pl[1].w)
                 : "l"(lp + (ROUTE_COALESCED ? 128 : 4)));
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(pl[2].x), "=f"(pl[2].y), "=f"(pl[2].z), "=f"(pl[2].w)
                 : "l"(bp));
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(pl[3].x), "=f"(pl[3].y), "=f"(pl[3].z), "=f"(pl[3].w)
                 : "l"(bp + (ROUTE_COALESCED ? 128 : 4)));
  }

  if (tid == NCT) {
    const unsigned long long v = wait_device_state(G);
    S.par = (int)((v / (unsigned long long)G) & 1);
    S.tlslot = (int)((v / (unsigned long long)G) & 63);
  }
  if (tid < NCW) S.done_cnt[tid] = 0;
  if (tid == 0) {
    if constexpr (TT != 6) S.max_tok = 0;
  }
  for (int i = tid; i < outb / 4; i += NTHREADS) outacc[i] = 0.f;

  // ---------------- routing: top-8 per token, one warp per token, straight from global.
  // Lane l holds experts 8l..8l+7 (T = 6: 4l..4l+3 and 128+4l..128+4l+3); 8 rounds of warp max
  // over the lane maxima, ties -> lowest expert (the trtllm-gen and reference rule). Lane k < 8
  // ends with the k-th pick and its renormalised weight.
  int re[ROUTE_ROUNDS] = {};
  float rw[ROUTE_ROUNDS] = {};
  if (warp < ROUTE_WARPS) {
    if (tid < E) S.mask[tid] = 0u;
#pragma unroll 1
    for (int j = 0; j < ROUTE_ROUNDS; ++j) {
      const int t = warp + j * NCW;
      if (t < T) {
        const float4* lg4 =
            reinterpret_cast<const float4*>(logits + t * E + lane * (ROUTE_COALESCED ? 4 : 8));
        const float4* bs4 =
            reinterpret_cast<const float4*>(bias + lane * (ROUTE_COALESCED ? 4 : 8));
        float4 l0, l1, b0, b1;
        if (j == 0) {
          l0 = pl[0];
          l1 = pl[1];
          b0 = pl[2];
          b1 = pl[3];
        } else {
          l0 = lg4[0];
          l1 = lg4[ROUTE_COALESCED ? 32 : 1];
          b0 = bs4[0];
          b1 = bs4[ROUTE_COALESCED ? 32 : 1];
        }
        const float lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        unsigned key[8];
        float sv[8];
        int ex[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          sv[i] = 1.0f / (1.0f + expf(-lv[i]));
          key[i] = f2key(sv[i] + __bfloat162float(__float2bfloat16_rn(bv[i])));
          if constexpr (ROUTE_COALESCED)
            ex[i] = i < 4 ? lane * 4 + i : 128 + lane * 4 + (i - 4);
          else
            ex[i] = lane * 8 + i;
        }
        if constexpr (TT == 6) {
          // this token's top-1 = the first pick below: the lowest expert among those with the
          // largest key (a lane holds two expert ranges, so not the lowest such lane)
          unsigned lm = key[0];
#pragma unroll
          for (int i = 1; i < 8; ++i) lm = max(lm, key[i]);
          int le = E;
#pragma unroll
          for (int i = 0; i < 8; ++i) le = (key[i] == lm) ? min(le, ex[i]) : le;
          const unsigned M = __reduce_max_sync(0xffffffffu, lm);
          const int e1 = (int)__reduce_min_sync(0xffffffffu, lm == M ? (unsigned)le : (unsigned)E);
          if (lane == 0) S.top1[t] = e1;
          asm volatile("bar.arrive 8, %0;" ::"n"((6 + 1) * 32) : "memory");  // -> producer
        }
        // odd-even transposition sort, descending key, ascending expert on ties
#pragma unroll
        for (int r = 0; r < 8; ++r) {
#pragma unroll
          for (int i = r & 1; i + 1 < 8; i += 2) {
            const bool sw = key[i + 1] > key[i];
            const unsigned ka = sw ? key[i + 1] : key[i], kb2 = sw ? key[i] : key[i + 1];
            const int ea = sw ? ex[i + 1] : ex[i], eb = sw ? ex[i] : ex[i + 1];
            const float sa = sw ? sv[i + 1] : sv[i], sb = sw ? sv[i] : sv[i + 1];
            key[i] = ka;
            key[i + 1] = kb2;
            ex[i] = ea;
            ex[i + 1] = eb;
            sv[i] = sa;
            sv[i + 1] = sb;
          }
        }
        float my_s = 0.f;
        int my_e = 0;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
          const unsigned m = __reduce_max_sync(0xffffffffu, key[0]);
          int win, we;
          if constexpr (ROUTE_COALESCED) {
            // a lane holds two expert ranges: ties go to the lowest expert, not the lowest lane
            const bool top = key[0] == m;
            we = (int)__reduce_min_sync(0xffffffffu, top ? (unsigned)ex[0] : (unsigned)E);
            win = __ffs(__ballot_sync(0xffffffffu, top && ex[0] == we)) - 1;
          } else {
            // lane l holds experts 8l..8l+7: the lowest tied lane holds the lowest tied expert
            win = __ffs(__ballot_sync(0xffffffffu, key[0] == m)) - 1;
            we = __shfl_sync(0xffffffffu, ex[0], win);
          }
          const float ws = __shfl_sync(0xffffffffu, sv[0], win);
          if (lane == win) {
#pragma unroll
            for (int i = 0; i < 7; ++i) {
              key[i] = key[i + 1];
              ex[i] = ex[i + 1];
              sv[i] = sv[i + 1];
            }
            key[7] = 0u;
          }
          if (lane == k) {
            my_e = we;
            my_s = ws;
          }
        }
        if (j == 0)
          re[0] = my_e;
        else
          re[1] = my_e;
        // renormalise in parallel (same summation order as a sequential loop)
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) sum += __shfl_sync(0xffffffffu, my_s, k);
        const float den = sum + 1e-20f;
        const float wv = __bfloat162float(__float2bfloat16_rn((my_s * route_scale) / den));
        if (j == 0)
          rw[0] = wv;
        else
          rw[1] = wv;
      }
    }
    asm volatile("bar.sync 6, %0;" ::"n"(ROUTE_WARPS * 32) : "memory");  // masks cleared
#pragma unroll
    for (int j = 0; j < ROUTE_ROUNDS; ++j) {
      const int t = warp + j * NCW;
      if (t < T && lane < TOPK) atomicOr(&S.mask[re[j]], 1u << t);
    }
  }
  int n_early = 0;  // producer (T = 6): W13 items issued before the full routing
  if constexpr (TT == 6) {
    if (warp == WARP_PROD) {
      *reinterpret_cast<uint2*>(S.fidx + lane * 8) = make_uint2(0xffffffffu, 0xffffffffu);
      if (lane < MAXNS) {
        mbar_init(&S.full[lane], 1);
        mbar_init(&S.empty[lane], NCT);
      }
      if (lane < 8) {
        mbar_init(&S.full2[lane], 1);
        mbar_init(&S.empty2[lane], NCT);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      if (lane == 0) {
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&w1_map))
                     : "memory");
        asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&x_map1))
                     : "memory");
      }
      const uint64_t pol = policy_evict_first();
      __syncwarp();
      asm volatile("bar.sync 8, %0;" ::"n"((6 + 1) * 32) : "memory");  // top-1 of every token
      // distinct top-1 experts in token order -> expert slots 0 .. nF-1
      const int e = lane < 6 ? S.top1[lane] : -1;
      bool first = lane < 6;
#pragma unroll
      for (int k = 0; k < 6; ++k) {
        const int ek = __shfl_sync(0xffffffffu, e, k);
        if (k < lane && ek == e) first = false;
      }
      const unsigned fm = __ballot_sync(0xffffffffu, first);
      const int nF = __popc(fm);
      if (first) {
        const int fi = __popc(fm & ((1u << lane) - 1u));
        S.fexp[fi] = e;
        S.fidx[e] = (unsigned char)fi;
      }
      if (lane == 0) S.n_first = nF;
      __syncwarp();
      // whole ring slots (2 items each) whose items p = c + k G all lie in the early slots
      while (n_early < 2 * 4 && (c + (n_early + 1) * G) / ITEMS_PER_EXPERT < nF) n_early += 2;
      constexpr int SUBB = (PIECE1 + 768 + 1023) & ~1023;
      if (lane < n_early / 2) mbar_expect_arrive(&S.full[lane], 2 * (PIECE1 + 768));
      __syncwarp();
      if (lane < n_early) {
        const int p = c + lane * G;
        const int s = p / ITEMS_PER_EXPERT, q = p - s * ITEMS_PER_EXPERT;
        char* dst = dsm + lane * SUBB;  // slot lane / 2, sub lane % 2
        kA::tma_w1_g2s(dst, w1_map, (q & 3) * 128, q >> 2, S.fexp[s], &S.full[lane >> 1], pol);
        tma_x6_g2s(dst + PIECE1, x_map1, q >> 2, token_base, &S.full[lane >> 1]);
      }
    }
  } else {
    if (tid < MAXNS) {
      mbar_init(&S.full[tid], 1);
      mbar_init(&S.empty[tid], NCT);
    }
    if (tid < 3) {
      mbar_init(&S.full2[tid], 1);
      mbar_init(&S.empty2[tid], NCT);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // ---------------- expert compaction (thread e handles expert e)
  int incl = 0, cnt = 0;
  bool in_f = false;
  if (tid < E) {
    cnt = __popc(S.mask[tid]);
    if constexpr (TT == 6) in_f = S.fidx[tid] != 0xff;
    incl = in_f ? 0 : (((cnt > 0) << 16) | cnt);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    if (lane == 31) S.wsum[warp] = incl;
    if constexpr (TT != 6) {
      const int mx = __reduce_max_sync(0xffffffffu, cnt);
      if (lane == 0) atomicMax(&S.max_tok, mx);
    }
  }
  if constexpr (TT == 6) {
    if (warp < NCW) asm volatile("bar.sync 10, %0;" ::"n"(NCT) : "memory");
  } else {
    __syncthreads();
  }
  if (tid < E) {
    int wpre = lane < NCW ? S.wsum[lane] : 0;
#pragma unroll
    for (int o = 1; o < NCW; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, wpre, o);
      if (lane >= o) wpre += y;
    }
    const int pre = warp == 0 ? 0 : __shfl_sync(0xffffffffu, wpre, warp - 1);
    incl += pre;
    const int excl = incl - (in_f ? 0 : (((cnt > 0) << 16) | cnt));
    int nF = 0, sumF = 0, myF = 0;
    if constexpr (TT == 6) {
      // early experts: slots 0 .. nF-1 in token order, their pairs first
      nF = S.n_first;
      const int fi = in_f ? S.fidx[tid] : 0;
#pragma unroll
      for (int k = 0; k < 6; ++k) {
        if (k < nF) {
          const int ck = __popc(S.mask[S.fexp[k]]);
          sumF += ck;
          if (k < fi) myF += ck;
        }
      }
      if (in_f) {
        S.slot_expert[fi] = tid;
        S.slot_pstart[fi] = myF;
        S.epst[tid] = (unsigned char)myF;
      }
    }
    if (cnt > 0 && !in_f) {
      const int slot = nF + (excl >> 16), pst = sumF + (excl & 0xffff);
      S.slot_expert[slot] = tid;
      S.slot_pstart[slot] = pst;
      S.epst[tid] = (unsigned char)pst;
    }
    if (tid == E - 1) {
      S.n_active = nF + (incl >> 16);
      S.n_pairs = sumF + (incl & 0xffff);
      S.slot_pstart[nF + (incl >> 16)] = sumF + (incl & 0xffff);
    }
  }
  if constexpr (TT == 6) {
    if (warp < NCW) asm volatile("bar.sync 10, %0;" ::"n"(NCT) : "memory");
  } else {
    __syncthreads();
  }
  // pairs: grouped by expert slot, tokens ascending within an expert
  if (warp < ROUTE_WARPS) {
#pragma unroll
    for (int j = 0; j < ROUTE_ROUNDS; ++j) {
      const int t = warp + j * NCW;
      if (t < T && lane < TOPK) {
        const int e = re[j];
        const int pi = S.epst[e] + __popc(S.mask[e] & ((1u << t) - 1u));
        S.pair_tok[pi] = t;
        S.pair_w[pi] = rw[j];
      }
    }
  }
  __syncthreads();
  const int n_active = S.n_active, n_pairs = S.n_pairs;
  const int par = S.par;
  float* gacc = g_acc[par];
  // ---------------- layout
  Layout L;
  {
    const int xbytes = TT == 6 ? 768 : ((S.max_tok * 128) + 127) & ~127;
    L.sub_bytes = PIECE1 + xbytes;  // one W13 piece + its x rows
    if constexpr (TT == 6 || TT == 10) L.sub_bytes = (L.sub_bytes + 1023) & ~1023;
    L.slot_bytes = KB_SUB * L.sub_bytes;  // KB_SUB pieces per ring slot: fewer barrier round trips
    if constexpr (TT == 6 || TT == 10) L.slot_bytes = (L.slot_bytes + 1023) & ~1023;
    // T = 6: deeper W2 stream (the phase is bound by bytes in flight): T6_W2S single-expert
    // stages over the W13 ring and past it (the SwiGLU staging area starts after them)
    L.NS2 = TT == 6 ? T6_W2S : 3;
    L.slot2_bytes = TT == 6 ? kA::W2_POS : 2 * kA::W2_POS;
    L.R_lo = fdiv(c * H, G);
    L.R_c = fdiv((c + 1) * H, G) - L.R_lo;
    // T = 6 / 10: one 4D bulk tensor copy per expert (41 rows x 4 k-blocks, kA::W2_POS bytes)
    L.P = (TT == 6 || TT == 10) ? fdiv(L.slot_bytes, kA::W2_POS) : fdiv(L.slot_bytes, L.R_c * 128);
    L.P2 = TT == 6 ? 1 : TT == 10 ? 2 : L.P;
    const int table_bytes = n_pairs * I * 2;
    const int ns = fdiv(DSMEM - outb - TBARB - table_bytes, L.slot_bytes);
    L.NS = (TT == 6 && ns > 4) ? 4 : (ns < MAXNS ? ns : MAXNS);
    L.ring = dsm;
    L.table = dsm + DSMEM - outb - TBARB - table_bytes;
    L.tbar = reinterpret_cast<unsigned long long*>(dsm + DSMEM - outb - TBARB);
    L.fstage = reinterpret_cast<float*>(L.table - FSTAGE_BYTES);
    L.outacc = outacc;
    L.out_stride = T * OUT_ROWS;
    L.N1 = n_active * ITEMS_PER_EXPERT;
    L.nq = (TT == 6 || TT == 10) ? n_active : n_active * KB2;
  }

  if (warp == WARP_PROD) {
    // ================= producer
    const uint64_t pol_w = policy_evict_first();
    Ring rs;
    int sub = 0;
    bool started = false;
    int pfirst = c;
    if constexpr (TT == 6) {
      pfirst = c + n_early * G;  // the first n_early / 2 slots were issued during the routing
      for (int k = 0; k < n_early / 2; ++k) rs.next(L.NS);
      // Slot zero is already in flight. Let consumers wait on its full barrier now instead
      // of withholding them until the remaining ring slots have also been issued.
      asm volatile("bar.arrive 7, %0;" : : "n"(NCT + 32) : "memory");
      started = true;
    }
#pragma unroll 1
    for (int p = pfirst; p < L.N1; p += G) {
      const int s = p / ITEMS_PER_EXPERT;
      const int q = p - s * ITEMS_PER_EXPERT;
      const int kb = q >> 2, rg = q & 3;
      const int e = S.slot_expert[s];
      const int pst = S.slot_pstart[s], nt = S.slot_pstart[s + 1] - pst;
      if (sub == 0) {
        mbar_wait(&S.empty[rs.slot], rs.phase ^ 1);  // all lanes: each issues copies into the slot
        if (lane == 0) {
          unsigned bytes = PIECE1 + (TT == 6 ? 768 : nt * 128);
          for (int j = 1; j < KB_SUB && p + j * G < L.N1; ++j) {
            const int s2 = (p + j * G) / ITEMS_PER_EXPERT;
            bytes += PIECE1 + (TT == 6 ? 768 : (S.slot_pstart[s2 + 1] - S.slot_pstart[s2]) * 128);
          }
          mbar_expect_arrive(&S.full[rs.slot], bytes);
        }
        __syncwarp();
      }
      char* dst = L.ring + rs.slot * L.slot_bytes + sub * L.sub_bytes;
      if (lane == 0) {
        if constexpr (TT == 6 || TT == 10)
          kA::tma_w1_g2s(dst, w1_map, rg * 128, kb, e, &S.full[rs.slot], pol_w);
        else
          bulk_g2s(dst, w1 + ((size_t)(e * KB1 + kb) * M1 + rg * 128) * 64, PIECE1,
                   &S.full[rs.slot], pol_w);
      } else if constexpr (TT == 6) {
        if (lane == 1) tma_x6_g2s(dst + PIECE1, x_map1, kb, token_base, &S.full[rs.slot]);
      } else if constexpr (TT == 10) {
        if (lane <= nt)
          kA::tma_x_g2s(dst + PIECE1 + (lane - 1) * 128, x_map1,
                        token_base + S.pair_tok[pst + lane - 1], kb, &S.full[rs.slot], pol_w);
      } else if (lane <= nt) {
        const int tk = S.pair_tok[pst + lane - 1];
        bulk_g2s(dst + PIECE1 + (lane - 1) * 128, x + (size_t)tk * H + kb * 64, 128,
                 &S.full[rs.slot], pol_w);
      }
      if (++sub == KB_SUB || p + G >= L.N1) {
        sub = 0;
        rs.next(L.NS);
        // the consumers start polling only once the first ring fills are on their way
        if (!started && ((TT == 10 && rs.slot == 1) || (TT != 10 && rs.slot == 0))) {
          asm volatile("bar.arrive 7, %0;" ::"n"(NCT + 32) : "memory");
          started = true;
        }
      }
    }
    if (!started) asm volatile("bar.arrive 7, %0;" ::"n"(NCT + 32) : "memory");
    const int pbytes = L.R_c * 128;
    if constexpr (TT == 6 || TT == 10) {
      Ring rs2;
      const int nitems = (L.N1 > c) ? (L.N1 - 1 - c) / G + 1 : 0;
      const int nfills = (nitems + KB_SUB - 1) / KB_SUB;
#pragma unroll 1
      for (int u0 = 0; u0 < L.nq; u0 += L.P2) {
        const int np = min(L.P2, L.nq - u0);
        if (u0 < L.NS2 * L.P2) {
          const int lo = (rs2.slot * L.slot2_bytes) / L.slot_bytes;
          const int hi =
              min(L.NS - 1, (rs2.slot * L.slot2_bytes + L.slot2_bytes - 1) / L.slot_bytes);
          if (lane == 0) {
            for (int sl = lo; sl <= hi; ++sl) {
              const int uses = nfills / L.NS + (sl < (nfills % L.NS));
              if (uses > 0) mbar_wait(&S.empty[sl], (uses - 1) & 1);
            }
          }
          __syncwarp();
        }
        mbar_wait(&S.empty2[rs2.slot], rs2.phase ^ 1);
        if (lane == 0)
          mbar_expect_arrive(&S.full2[rs2.slot],
                             np * KB2 * ((TT == 6 || TT == 10) ? L.R_c : kA::W2_TX_ROWS) * 128);
        __syncwarp();
        if (lane < np) {
          const int u = u0 + lane;
          char* pdst = L.ring + rs2.slot * L.slot2_bytes + lane * kA::W2_POS;
          kA::tma_w1_g2s(pdst, (TT == 6 || TT == 10) && L.R_c == 40 ? w2_map32 : w2_map9, L.R_lo, 0,
                         S.slot_expert[u], &S.full2[rs2.slot], pol_w);
        }
        rs2.next(L.NS2);
      }
    } else {
#pragma unroll 1
      for (int u0 = 0; u0 < L.nq; u0 += L.P) {
        const int np = min(L.P, L.nq - u0);
        mbar_wait(&S.empty[rs.slot], rs.phase ^ 1);
        if (lane == 0)
          mbar_expect_arrive(&S.full[rs.slot],
                             np * ((TT == 10) ? KB2 * kA::W2_TX_ROWS * 128 : pbytes));
        __syncwarp();
        if (lane < np) {
          const int u = u0 + lane;
          if constexpr (TT == 10) {
            char* pdst = L.ring + rs.slot * L.slot_bytes + lane * kA::W2_POS;
            kA::tma_w1_g2s(pdst, w2_map9, L.R_lo, 0, S.slot_expert[u], &S.full[rs.slot], pol_w);
          } else {
            const int s = u >> 2, kb = u & 3;
            const int e = S.slot_expert[s];
            bulk_g2s(L.ring + rs.slot * L.slot_bytes + lane * pbytes,
                     w2 + ((size_t)(e * KB2 + kb) * H + L.R_lo) * 64, pbytes, &S.full[rs.slot],
                     pol_w);
          }
        }
        rs.next(L.NS);
      }
    }
    return;
  }

  if (warp == WARP_SIG) {
    // ================= signaller: CTA-wide "expert s done" -> grid-wide counter
    if (lane == 0) {
      int s = 0;
#pragma unroll 1
      while (s < n_active) {
        int s1 = n_active;
#pragma unroll
        for (int w = 0; w < NCW; ++w) s1 = min(s1, atomicAdd(&S.done_cnt[w], 0) & 0xffff);
        if (s1 <= s) {
          __nanosleep(32);
          continue;
        }
        for (int k = s; k < s1; ++k) red_add_u32(&g_cnt[par][k], 1u);
        s = s1;
      }
    }
    return;
  }

  if (warp == WARP_TAB) {
    // per-expert "SwiGLU rows ready" barriers (only the active experts); the consumers pass
    // named barrier 5 before their first wait on them
    for (int k = lane; k < n_active; k += 32) mbar_init(&L.tbar[k], 32);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncwarp();
    asm volatile("bar.arrive 5, %0;" ::"n"(NCT + 32) : "memory");
    // ================= table builder
    // clear the other parity's state, dirtied by the previous launch
    {
      // read after the CTA barriers that follow wait_device_state
      const int pp = g_prev[par ^ 1][0], pa = g_prev[par ^ 1][1];
      float4* z = reinterpret_cast<float4*>(g_acc[par ^ 1]);
      const int tot = pp * (M1 / 4);
      const int lo = (int)((long long)tot * c / G), hi = (int)((long long)tot * (c + 1) / G);
      for (int i = lo + lane; i < hi; i += 32) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int s = c + lane * G; s < pa; s += 32 * G) g_cnt[par ^ 1][s] = 0u;
      if (c == 0 && lane == 0) {
        g_prev[par][0] = n_pairs;
        g_prev[par][1] = n_active;
      }
    }
    const unsigned target = (unsigned)min(G, ITEMS_PER_EXPERT);
    // lane l covers canonical j = 8l .. 8l+7: up rows base + {0..3} (even j), base + 16 + {0..3}
    // (odd j); gate rows are 8 above the matching up rows.
    const int rbase = (lane >> 1) * 32 + (lane & 1) * 4;
    __nv_bfloat16* tab = reinterpret_cast<__nv_bfloat16*>(L.table);
    char* stage =
        L.ring + max(L.NS * L.slot_bytes, (TT == 6 || TT == 10) ? L.NS2 * L.slot2_bytes : 0);
    char* stage_end = (TT == 6 || TT == 10) ? reinterpret_cast<char*>(L.fstage) : L.table;
    const int stage_cap = min(TT == 6    ? 26
                              : TT == 10 ? 24
                                         : 16,
                              (int)((stage_end - stage) / (M1 * 4)));
    unsigned sph = 0;
    if (lane == 0) {
      mbar_init(&S.stage_bar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncwarp();
    int s0 = 0;
#pragma unroll 1
    while (s0 < n_active) {
      const int s = s0 + lane;
      const bool rdy = s < n_active && ld_relaxed_gpu(&g_cnt[par][s]) >= target;
      const unsigned b = __ballot_sync(0xffffffffu, rdy);
      int nr = (b == 0xffffffffu) ? 32 : (__ffs(~b) - 1);
      nr = min(nr, n_active - s0);
      if (nr == 0) {
        __nanosleep(64);
        continue;
      }
      const int p0 = S.slot_pstart[s0], p1 = S.slot_pstart[s0 + nr];
      if constexpr (TT == 6 || TT == 10) {
        // Pull complete fp32 rows into spare smem with bulk copies, then apply SwiGLU.
        asm volatile("fence.proxy.async.global;" ::: "memory");
#pragma unroll 1
        for (int pc = p0; pc < p1; pc += stage_cap) {
          const int n = min(stage_cap, p1 - pc);
          if (lane == 0) {
            mbar_expect_arrive(&S.stage_bar, n * M1 * 4);
            const int step = (TT == 6 || TT == 10) ? 4 : 1;
            for (int k = 0; k < n; k += step) {
              const int nr = min(step, n - k);
              asm volatile(
                  "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], "
                  "%2, [%3];" ::"r"(smem_u32(stage + k * M1 * 4)),
                  "l"(gacc + (size_t)(pc + k) * M1), "r"(nr * M1 * 4), "r"(smem_u32(&S.stage_bar))
                  : "memory");
            }
          }
          mbar_wait(&S.stage_bar, sph);
          sph ^= 1u;
#pragma unroll 1
          for (int k = 0; k < n; ++k) {
            const float* src = reinterpret_cast<const float*>(stage + k * M1 * 4) + rbase;
            const float4 u0 = *reinterpret_cast<const float4*>(src);
            const float4 g0 = *reinterpret_cast<const float4*>(src + 8);
            const float4 u1 = *reinterpret_cast<const float4*>(src + 16);
            const float4 g1 = *reinterpret_cast<const float4*>(src + 24);
            __nv_bfloat162 o[4];
            o[0] = __floats2bfloat162_rn(silu_mul(g0.x, u0.x), silu_mul(g1.x, u1.x));
            o[1] = __floats2bfloat162_rn(silu_mul(g0.y, u0.y), silu_mul(g1.y, u1.y));
            o[2] = __floats2bfloat162_rn(silu_mul(g0.z, u0.z), silu_mul(g1.z, u1.z));
            o[3] = __floats2bfloat162_rn(silu_mul(g0.w, u0.w), silu_mul(g1.w, u1.w));
            const int dst_chunk = (TT == 6 || TT == 10) ? (lane ^ ((pc + k) & 7)) : lane;
            *reinterpret_cast<uint4*>(tab + (size_t)(pc + k) * I + dst_chunk * 8) =
                *reinterpret_cast<const uint4*>(o);
          }
          __syncwarp();
        }
      } else {
#pragma unroll 1
        for (int p = p0; p < p1; p += 2) {
          float4 u0[2], u1[2], g0[2], g1[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            const float* src = gacc + (size_t)min(p + k, p1 - 1) * M1 + rbase;
            u0[k] = ldcg4(src);
            g0[k] = ldcg4(src + 8);
            u1[k] = ldcg4(src + 16);
            g1[k] = ldcg4(src + 24);
          }
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            if (p + k < p1) {
              __nv_bfloat162 o[4];
              o[0] = __floats2bfloat162_rn(silu_mul(g0[k].x, u0[k].x), silu_mul(g1[k].x, u1[k].x));
              o[1] = __floats2bfloat162_rn(silu_mul(g0[k].y, u0[k].y), silu_mul(g1[k].y, u1[k].y));
              o[2] = __floats2bfloat162_rn(silu_mul(g0[k].z, u0[k].z), silu_mul(g1[k].z, u1[k].z));
              o[3] = __floats2bfloat162_rn(silu_mul(g0[k].w, u0[k].w), silu_mul(g1[k].w, u1[k].w));
              *reinterpret_cast<uint4*>(tab + (size_t)(p + k) * I + lane * 8) =
                  *reinterpret_cast<const uint4*>(o);
            }
          }
        }
      }
      __syncwarp();
      for (int k = s0; k < s0 + nr; ++k) mbar_arrive(&L.tbar[k]);  // every writer lane arrives
      s0 += nr;
    }
    // Every table is built (the last reads of the partials and counters) and the consumers have
    // flushed their GEMM1 partials (named barrier 11): this CTA is done with the device state.
    // Counting its exit here, on a warp that is otherwise idle, keeps the release fence off the
    // GEMM2 / output path (wait_device_state).
    asm volatile("bar.sync 11, %0;" ::"n"(NCT + 32) : "memory");
    if (lane == 0) kA::red_release_add64(&g_epoch, 1ull);
    return;
  }

  // ================= consumers
  asm volatile("bar.sync 7, %0;" ::"n"(NCT + 32) : "memory");
  Ring rs;
  if constexpr (TT == 6 || TT == 10) {
    // T = 10: more than 8 tokens on one expert needs the two n-block (16 column) variants
    const bool wide = TT == 10 && S.max_tok > 8;
    // One real accumulator partial uses a returning atomic after the warp's reductions; its
    // consumed return orders the actual accumulator traffic before expert publication.
    if (!wide)
      gemm1<1, true, TT == 6 || TT == 10, TT == 6>(S, L, rs, gacc);
    else
      gemm1<2, true, false, false>(S, L, rs, gacc);
    asm volatile("bar.arrive 11, %0;" ::"n"(NCT + 32) : "memory");  // partials flushed
    asm volatile("bar.sync 5, %0;" ::"n"(NCT + 32) : "memory");
    Ring rs2;
    if (!wide)
      gemm2<1, true, true, TT == 6 || TT == 10>(S, L, rs2);
    else
      gemm2<2, true, true, TT == 10>(S, L, rs2);
  } else {
    if (S.max_tok <= 8) {
      gemm1<1>(S, L, rs, gacc);
      asm volatile("bar.arrive 11, %0;" ::"n"(NCT + 32) : "memory");  // partials flushed
      asm volatile("bar.sync 5, %0;" ::"n"(NCT + 32) : "memory");
      gemm2<1>(S, L, rs);
    } else {
      gemm1<2>(S, L, rs, gacc);
      asm volatile("bar.arrive 11, %0;" ::"n"(NCT + 32) : "memory");  // partials flushed
      asm volatile("bar.sync 5, %0;" ::"n"(NCT + 32) : "memory");
      gemm2<2>(S, L, rs);
    }
  }
  consumer_sync();
#pragma unroll 1
  for (int i = tid; i < T * L.R_c; i += NCT) {
    const int t = i / L.R_c, r = i - t * L.R_c;
    const int pr = L.R_lo + r;
    const int q = pr & 31;
    const int canon = (pr & ~31) + 4 * (q & 7) + (q >> 3);
    float v = 0.f;
#pragma unroll
    for (int p = 0; p < out_planes; ++p) v += outacc[(p * T + t) * OUT_ROWS + r];
    out[(size_t)t * H + canon] = __float2bfloat16_rn(v);
  }
}

}  // namespace kB

}  // namespace bf16_decode_sm103
}  // namespace fused_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_BF16_MOE_DECODE_SM103_CUH_
