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
#ifndef FLASHINFER_COMM_ALLREDUCE_FUSION_WS2_SPECIALIZED_CUH_
#define FLASHINFER_COMM_ALLREDUCE_FUSION_WS2_SPECIALIZED_CUH_

// Fused all-reduce over two ranks + residual add + RMSNorm, hidden size 6144, bf16, for
// compute capability 10.3. Serves `allreduce_fusion(pattern=kARResidualRMSNorm, fp32_acc=True)`
// on a two-rank single-node TRT-LLM workspace; see
// flashinfer/comm/_allreduce_fusion_ws2_specialized.py.
//
// Host interface (the Python side relies on exactly these; a replacement kernel keeps the names,
// signatures and semantics and may change everything else in this file):
//
//   region_bytes(capacity)  bytes of the peer-mapped region each rank allocates for calls of up to
//                           `capacity` tokens. Both ranks use the same capacity.
//   epoch_words(grid)       uint32 words of the rank-local control buffer (per-CTA call counters
//                           and row-claim counters) for a grid of `grid` CTAs.
//   init(region, capacity, epoch, epoch_words, stream)
//                           puts a rank's own region and its control buffer into the idle state.
//                           Run on both ranks (and synchronised) before either rank launches.
//   max_grid(device)        CTAs per launch on `device` (all co-resident); <= 0 on error. The two
//                           ranks launch the same grid (the minimum of their values) in every call.
//   launch(x, residual_in, gamma, residual_out, norm_out, own_region, peer_region, epoch,
//          capacity, num_tokens, grid, eps, weight_bias, launch_with_pdl, stream)
//                           one call, 1 <= num_tokens <= capacity, on `stream`. The inputs of a row
//                           are read before the outputs of that row are written, so `residual_out`
//                           and/or `norm_out` may be the same tensor as `x` or `residual_in`.
//
// Kernel design. One persistent kernel, grid = all co-resident CTAs (fixed per device, independent
// of the token count), 768 threads per CTA. A row is 24 segments of 256 elements; warp w owns
// segment w of every row its CTA processes and does both jobs for it:
//   - push:    encode segment w of the row kL rows ahead (this rank's x, from the shared-memory
//              ring) and store it from registers into the PEER's receive slot;
//   - consume: wait for the peer's segment w of the current row in its OWN receive slot, decode,
//              add, add the residual, block-reduce sum(h^2) (one CTA barrier per row) and write
//              residual_out / norm_out.
// x / residual rows are loaded with bulk async copies (cp.async.bulk + mbarrier) into a kNB-row
// shared-memory ring by thread 0. Start-up: each thread loads its own 16 B of the first kL rows
// straight from global memory and pushes them from registers.
//
// Rows: CTA b takes rows b + s*G for its first kNB ring slots, then claims further rows from a
// per-call counter (so faster SMs take more rows). Every row is read and written by exactly one
// CTA on each rank; the two ranks may assign a row to different CTAs (the receive slots are indexed
// by row). The claim counters live in the rank-local control buffer after the per-CTA call
// counters and are recycled by the call counter modulo kNumClaimCounters: call e claims from
// counter e % 4 and clears counter (e + 1) % 4, last used by call e - 3. Four (a power of two)
// keeps the cycle consistent when the 32-bit call counter wraps.
//
// Link format (lossless): a segment is sent either raw (16 B per lane, -0 mapped to +0) or packed
// into 12 B per lane (sign + mantissa bytes plus a 4-bit exponent code relative to the segment's
// max exponent, see encode2). A per-row header (one 16-bit unit per segment, 0x4000 = raw,
// 0x4100 | B = packed) tells the consumer the format.
//
// Arrival protocol: Lamport sentinel. Receive slots and headers hold 0x8000 in every 16-bit unit
// when idle. Senders never produce a 0x8000 unit (raw: -0 -> +0; packed: a segment with a 0x8000
// unit is sent raw), so a unit has arrived iff it is not 0x8000. The consumer polls its own memory
// with volatile loads (L2, never L1) and writes the sentinel back after use.
//
// Buffer reuse: two slots, selected by a per-CTA call counter (epoch[b] & 1) that every CTA
// advances once per call; the grid is the same on both ranks and in every call, so epoch[b] equals
// the call count on both ranks. A rank pushes call e+2 into the peer's slot (e & 1) only after its
// call e+1 kernel has finished, which needs all of the peer's call e+1 rows, which the peer pushes
// only from its call e+1 kernel, i.e. after its call e kernel (whose consumers restored slot
// (e & 1) to the sentinel) has completed. With programmatic dependent launch the kernel waits for
// the preceding grid (griddepcontrol.wait) before anything else and never triggers its dependents
// early, so this ordering (and the claim-counter recycling) holds with and without PDL.
//
// Liveness: iteration i of a CTA pushes its row i+kL before it waits for the peer's copy of its
// row i; rows of a CTA are increasing, so the smallest row awaited anywhere has always been pushed
// (all CTAs are co-resident).
//
// Waits have no deadline (as in the TRT-LLM kernels): a peer that never arrives stalls the call.
// For debugging only, -DFLASHINFER_ALLREDUCE_FUSION_WS2_SPIN_TIMEOUT_NS=<ns> makes a wait that
// exceeds <ns> trap.

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>

namespace flashinfer {
namespace allreduce_fusion_ws2 {

constexpr int kHidden = 6144;
constexpr int kRowVec = kHidden / 8;  // 768 x 16 B per row
constexpr int kNT = 768;              // one 16 B chunk per thread per row
constexpr int kNW = kNT / 32;         // 24 warps = 24 segments per row
constexpr int kL = 2;                 // push lookahead (rows)
constexpr int kNB = 4;                // x / residual ring depth (rows)
constexpr int kHdrVec = 8;            // 128 B header record per row (48 B used: 24 x u16)
constexpr int kNumClaimCounters = 4;  // row-claim counters, recycled by call counter modulo 4
constexpr int kCtlWords = 32;         // claim counters, padded to one 128 B line
constexpr uint32_t kNeg0x2 = 0x80008000u;
constexpr uint16_t kHdrRaw = 0x4000, kHdrPacked = 0x4100;
constexpr uint32_t kRowB = (uint32_t)kHidden * 2;      // 12288 B
constexpr size_t kRowBytes = (size_t)kHidden * 2;      // 12288 B per row
constexpr size_t kHdrRowBytes = (size_t)kHdrVec * 16;  // 128 B per row

static_assert(kRowVec == kNT, "one 16 B chunk per thread per row");
static_assert(kNW <= 32, "one row total per lane of a warp");
static_assert(kL >= 1 && kL < kNB, "lookahead must be below the ring depth");
static_assert((kNumClaimCounters & (kNumClaimCounters - 1)) == 0 && kNumClaimCounters >= 3,
              "claim counters: a power of two (wrap-safe), at least three in flight");
static_assert(kNumClaimCounters <= kCtlWords, "claim counters exceed the control words");

struct Params {
  const int4* x;
  const int4* res;
  const int4* gamma;
  int4* res_out;
  int4* norm_out;
  int4* recv;       // own data slots (slot 1 at + slot_vec)
  int4* peer;       // peer data slots
  int4* recv_hdr;   // own header slots (slot 1 at + hdr_vec)
  int4* peer_hdr;   // peer header slots
  uint32_t* epoch;  // [G] per-CTA call counters
  uint32_t* ctl;    // [kNumClaimCounters] row-claim counters
  size_t slot_vec;
  size_t hdr_vec;
  int T;
  float eps;
  float weight_bias;
};

__device__ __forceinline__ int4 ld_volatile(const void* p) {
  int4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t ld_volatile32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void ld_volatile2(const void* p, uint32_t& x, uint32_t& y) {
  asm volatile("ld.volatile.global.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "l"(p) : "memory");
}

__device__ __forceinline__ uint32_t ld_volatile16(const void* p) {
  uint16_t v;
  asm volatile("ld.volatile.global.b16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ int4 ld_stream(const int4* p) {
  int4 v;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void st_v4(void* p, const int4& v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ void st_b32(void* p, uint32_t v) {
  asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_v2(void* p, uint32_t x, uint32_t y) {
  asm volatile("st.global.v2.b32 [%0], {%1,%2};" ::"l"(p), "r"(x), "r"(y) : "memory");
}

// Streaming store (outputs are not re-read by this kernel).
__device__ __forceinline__ void st_cs(void* p, const int4& v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

// Debug-only deadline for the spin waits (see the file comment); a no-op by default.
#ifdef FLASHINFER_ALLREDUCE_FUSION_WS2_SPIN_TIMEOUT_NS
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct SpinDeadline {
  unsigned long long t0;
  unsigned spins;
  __device__ __forceinline__ SpinDeadline() : t0(global_ns()), spins(0) {}
  __device__ __forceinline__ void check() {
    if ((++spins & 255u) == 0u &&
        global_ns() - t0 > (unsigned long long)(FLASHINFER_ALLREDUCE_FUSION_WS2_SPIN_TIMEOUT_NS))
      __trap();
  }
};
#else
struct SpinDeadline {
  __device__ __forceinline__ void check() {}
};
#endif

// any 16-bit unit of w equal to 0x8000 (exact: a borrow into the high unit needs a low unit equal
// to 0x8000)
__device__ __forceinline__ uint32_t has_neg0_u(uint32_t w) {
  const uint32_t t = w ^ kNeg0x2;
  return (t - 0x00010001u) & ~t & 0x80008000u;
}

// -0 -> +0 per bf16 lane (keeps the sentinel unambiguous; only the sign of a zero changes).
__device__ __forceinline__ uint32_t sanitize(uint32_t w) {
  return w & ~(__vcmpeq2(w, kNeg0x2) & kNeg0x2);
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// round-to-nearest-even bf16 pair (low half = a)
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}

// bf16x2 add with one round-to-nearest-even of the exact sum. For two bf16 operands this equals
// bf16(fp32(a) + fp32(b)) of the TRT-LLM kernel: the fp32 sum is exact unless the exponents differ
// by more than 16, and then the smaller operand is far below half a bf16 ulp of the larger, so both
// paths return the same bf16.
__device__ __forceinline__ uint32_t add_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// Receive-side probe: the header unit and both possible data layouts of this lane's chunk of one
// segment.
struct Probe {
  uint32_t h, q[3];
  int4 v;
};

__device__ __forceinline__ void probe_issue(Probe& pb, const char* hp, const char* sp, int lane) {
  pb.h = ld_volatile16(hp);
  pb.v = ld_volatile(sp + 16 * lane);
  ld_volatile2(sp + 8 * lane, pb.q[0], pb.q[1]);
  pb.q[2] = ld_volatile32(sp + 256 + 4 * lane);
}

__device__ __forceinline__ bool probe_ready(const Probe& pb) {
  if (pb.h == 0x8000u) return false;
  if (pb.h & 0x100u) return (has_neg0_u(pb.q[0]) | has_neg0_u(pb.q[1]) | has_neg0_u(pb.q[2])) == 0u;
  return (has_neg0_u((uint32_t)pb.v.x) | has_neg0_u((uint32_t)pb.v.y) |
          has_neg0_u((uint32_t)pb.v.z) | has_neg0_u((uint32_t)pb.v.w)) == 0u;
}

// -------------------------------------------------------------------------------------------------
// Packed link format (12 B per lane). Per element: the sign + mantissa byte (w0: elements 0..3,
// w1: elements 4..7, byte j = sign << 7 | mantissa) and a 4-bit code n = B + 1 - e in 1..15
// (B = segment max exponent), so a segment packs iff every exponent is in [B - 14, B]. w2 holds the
// codes of the even elements in its low half and of the odd elements in its high half (element 2k
// at bit 4k, element 2k+1 at bit 16 + 4k), so the decoder moves a pair of codes into exponent
// position with one shift. Codes are never 0, so w2 never holds a 0x8000 unit. Zero / subnormal
// elements pack only when B < 15 (they then fit the range).
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

__device__ __forceinline__ uint32_t encode2(const int4& xv, uint32_t& w0, uint32_t& w1,
                                            uint32_t& w2, int4& v) {
  const uint32_t w[4] = {(uint32_t)xv.x, (uint32_t)xv.y, (uint32_t)xv.z, (uint32_t)xv.w};
  uint32_t e2[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) e2[k] = (w[k] >> 7) & 0x00ff00ffu;  // exponents in 16-bit lanes
  const uint32_t m2 = __vmaxu2(__vmaxu2(e2[0], e2[1]), __vmaxu2(e2[2], e2[3]));
  const uint32_t B = __reduce_max_sync(0xffffffffu, max(m2 & 0xffffu, m2 >> 16));
  // e + 0x10e - B: bit 8 set iff e >= B - 14, low nibble = 15 - n
  const uint32_t C = (0x10eu - B) * 0x00010001u;
  uint32_t fit = 0x01000100u, codes = 0u;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t t = e2[k] + C;
    fit &= t;
    codes |= ~(t << (4 * k)) & (0x000f000fu << (4 * k));
  }
  w2 = codes;
  uint32_t s2[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) s2[k] = ((w[k] >> 8) & 0x00800080u) | (w[k] & 0x007f007fu);
  w0 = __byte_perm(s2[0], s2[1], 0x6420);
  w1 = __byte_perm(s2[2], s2[3], 0x6420);
  const bool ok = fit == 0x01000100u && (has_neg0_u(w0) | has_neg0_u(w1)) == 0u;
  if (__all_sync(0xffffffffu, ok)) return kHdrPacked | B;
  v = make_int4((int)sanitize(w[0]), (int)sanitize(w[1]), (int)sanitize(w[2]), (int)sanitize(w[3]));
  return kHdrRaw;
}

// decode one lane's packed chunk (q0, q1: sign + mantissa bytes, q2: codes)
__device__ __forceinline__ int4 decode2(uint32_t h, uint32_t q0, uint32_t q1, uint32_t q2) {
  const uint32_t KB = (((h & 0xffu) + 1u) << 7) * 0x00010001u;  // (B + 1) << 7 in both lanes
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t n7 = (k == 0   ? q2 << 7
                         : k == 1 ? q2 << 3
                         : k == 2 ? q2 >> 1
                                  : q2 >> 5) &
                        0x07800780u;
    // sign-extended sign + mantissa byte in each 16-bit lane: bit 15 = sign, bits 0..6 = mantissa
    const uint32_t sx = prmt(k < 2 ? q0 : q1, 0u, (k & 1) ? 0xb3a2u : 0x9180u);
    w[k] = (sx & 0x807f807fu) | (KB - n7);
  }
  return make_int4((int)w[0], (int)w[1], (int)w[2], (int)w[3]);
}

// Finish receiving this lane's chunk: re-poll until arrived (the first probe was issued ahead of
// time), restore the sentinel and decode.
__device__ __forceinline__ int4 probe_finish2(Probe& pb, const char* hp, char* sp, int lane) {
  if (!probe_ready(pb)) {
    SpinDeadline deadline;
    do {
      probe_issue(pb, hp, sp, lane);
      deadline.check();
    } while (!probe_ready(pb));
  }
  if (pb.h & 0x100u) {
    st_v2(sp + 8 * lane, kNeg0x2, kNeg0x2);
    st_b32(sp + 256 + 4 * lane, kNeg0x2);
    return decode2(pb.h, pb.q[0], pb.q[1], pb.q[2]);
  }
  st_v4(sp + 16 * lane, make_int4((int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2));
  return pb.v;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar))
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n"
      "@!P bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// Streaming read (x / residual are read once): L2 evict-first, so it does not displace lines worth
// keeping.
__device__ __forceinline__ void bulk_load(void* sdst, const void* gsrc, uint32_t bytes,
                                          uint64_t* bar) {
  asm volatile(
      "{\n"
      ".reg .b64 pol;\n"
      "createpolicy.fractional.L2::evict_first.b64 pol, 1.0;\n"
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], pol;\n"
      "}\n" ::"r"(smem_u32(sdst)),
      "l"(gsrc), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

struct Smem {
  int4 xs[kNB][kRowVec];
  int4 rs[kNB][kRowVec];
  uint64_t xfull[kNB];  // x of the slot's row landed
  uint64_t rfull[kNB];  // residual of the slot's row landed
  float red[2][kNW];
  uint16_t hst[kNB][32];  // header units of the row pushed from each ring slot (24 used)
  int row[kNB];           // row held by each ring slot (-1: none)
  uint32_t e;
};
static_assert(sizeof(Smem) <= 227 * 1024, "shared memory exceeds the per-CTA limit");

// encode this lane's chunk of segment `warp` and store it into the peer's slot of `row`; returns
// the header unit
__device__ __forceinline__ uint32_t push_chunk(const int4& xv, char* dst, int lane) {
  uint32_t w0, w1, w2;
  int4 v;
  const uint32_t u = encode2(xv, w0, w1, w2, v);
  if (u & 0x100u) {
    st_v2(dst + 8 * lane, w0, w1);
    st_b32(dst + 256 + 4 * lane, w2);
  } else {
    st_v4(dst + 16 * lane, v);
  }
  return u;
}

// Ring slot s of local index j holds that row's x (for the push at iteration j - kL and the add at
// iteration j) and residual (iteration j).
__global__ void __launch_bounds__(kNT, 1) ar_rmsnorm_ws2_kernel(Params p) {
  // Programmatic dependent launch: wait for the preceding grid before touching the call counter,
  // the claim counters, the inputs or the receive slots. Dependents are never triggered early
  // (implicit trigger at exit).
  asm volatile("griddepcontrol.wait;" ::: "memory");

  extern __shared__ __align__(128) char smem_raw[];
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);

  const int b = blockIdx.x, G = gridDim.x, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int T = p.T;
  // start-up loads: this thread's 16 B of the first kL rows and the call counter (every thread, no
  // barrier needed)
  int4 xe[kL];
  const uint32_t e_early = p.epoch[b];
#pragma unroll
  for (int j = 0; j < kL; ++j) {
    const int row = b + j * G;
    if (row < T) xe[j] = ld_stream(p.x + (size_t)row * kRowVec + tid);
  }
  auto early_push = [&]() {
    char* const peer0 =
        reinterpret_cast<char*>(p.peer + ((e_early & 1u) ? p.slot_vec : 0)) + 512 * warp;
#pragma unroll
    for (int j = 0; j < kL; ++j) {
      const int row = b + j * G;
      if (row < T) {
        const uint32_t u = push_chunk(xe[j], peer0 + (size_t)row * kRowBytes, lane);
        if (lane == 0) S.hst[j][warp] = (uint16_t)u;
        S.xs[j][tid] = xe[j];  // read back by this thread only (its add of the row)
      }
    }
  };
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kNB; ++s) {
      mbar_init(&S.xfull[s], 1);
      mbar_init(&S.rfull[s], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // every x first: the pushes need only x, the residuals are needed kL rows later
    for (int s = 0; s < kNB; ++s) {
      const int row = b + s * G;
      S.row[s] = row < T ? row : -1;
      if (row < T) {
        if (s < kL) {
          mbar_arrive(&S.xfull[s]);  // filled by the start-up loads (early_push)
        } else {
          mbar_expect_tx(&S.xfull[s], kRowB);
          bulk_load(S.xs[s], p.x + (size_t)row * kRowVec, kRowB, &S.xfull[s]);
        }
      } else {
        mbar_arrive(&S.xfull[s]);
        mbar_arrive(&S.rfull[s]);
      }
    }
    for (int s = 0; s < kNB; ++s) {
      const int row = b + s * G;
      if (row < T) {
        mbar_expect_tx(&S.rfull[s], kRowB);
        bulk_load(S.rs[s], p.res + (size_t)row * kRowVec, kRowB, &S.rfull[s]);
      }
    }
    S.e = e_early;
  }
  early_push();
  __syncthreads();
  if (tid ==
      0)  // consume phase 0 of the start-up slots' x barriers (completed by the arrive above)
    for (int j = 0; j < kL; ++j) mbar_wait(&S.xfull[j], 0u);
  const uint32_t e = S.e;
  uint32_t* const ctr = p.ctl + (e % (uint32_t)kNumClaimCounters);
  if (tid == 0 && b == 0)  // next call's counter was last used by call e - (kNumClaimCounters - 1)
    p.ctl[(e + 1u) % (uint32_t)kNumClaimCounters] = 0u;
  char* const peer = reinterpret_cast<char*>(p.peer + ((e & 1u) ? p.slot_vec : 0)) + 512 * warp;
  char* const recv = reinterpret_cast<char*>(p.recv + ((e & 1u) ? p.slot_vec : 0)) + 512 * warp;
  int4* const peer_hdr = p.peer_hdr + ((e & 1u) ? p.hdr_vec : 0);
  int4* const recv_hdr = p.recv_hdr + ((e & 1u) ? p.hdr_vec : 0);
  const char* const my_hdr = reinterpret_cast<const char*>(recv_hdr) + 2 * warp;

  // Push one ring slot at its current barrier phase.
  auto push_slot = [&](int s, uint32_t phase) {
    mbar_wait(&S.xfull[s], phase);
    const int row = S.row[s];
    if (row < 0) return;
    const uint32_t u = push_chunk(S.xs[s][tid], peer + (size_t)row * kRowBytes, lane);
    if (lane == 0) S.hst[s][warp] = (uint16_t)u;
  };
  // after a CTA barrier: lanes 0..2 of the header warp send the 48 B header of ring slot s
  auto send_hdr_slot = [&](int s) {
    const int row = S.row[s];
    if (row >= 0)
      st_v4(peer_hdr + (size_t)row * kHdrVec + lane, reinterpret_cast<const int4*>(S.hst[s])[lane]);
  };

  if (warp == 0 && lane < 3)
    for (int j = 0; j < kL; ++j) send_hdr_slot(j);
  // tid 0 keeps one claim in flight so the atomic's latency is off the refill path; every issued
  // claim is checked by a later refill (the indices before a valid claim are all valid, so their
  // refills run)
  bool fail = false;
  uint32_t claim = 0;
  if (tid == 0) claim = atomicAdd(ctr, 1u);

  // gamma kept packed; (weight_bias + gamma) formed in fp32 once
  float gw[8];
  {
    const int4 g = ld_stream(p.gamma + tid);
    const uint32_t gu[4] = {(uint32_t)g.x, (uint32_t)g.y, (uint32_t)g.z, (uint32_t)g.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      gw[2 * k] = p.weight_bias + lo(gu[k]);
      gw[2 * k + 1] = p.weight_bias + hi(gu[k]);
    }
  }

  Probe pb;
  int slot = 0;
  uint32_t phase = 0u;
  int red_slot = 0;
  int crow = S.row[0];
  if (crow >= 0)
    probe_issue(pb, my_hdr + (size_t)crow * kHdrVec * 16, recv + (size_t)crow * kRowBytes, lane);
  while (crow >= 0) {
    const int next_slot = slot + 1 < kNB ? slot + 1 : 0;
    const size_t row = (size_t)crow;
    const int push_sum = slot + kL;
    const int push_slot_idx = push_sum < kNB ? push_sum : push_sum - kNB;
    push_slot(push_slot_idx, phase ^ (uint32_t)(push_sum >= kNB));
    const int4 pr = probe_finish2(pb, my_hdr + row * kHdrVec * 16, recv + row * kRowBytes, lane);
    // The next slot was assigned in the prologue or by its prior refill.
    const int nxt = S.row[next_slot];
    if (nxt >= 0)  // probe the next row now: its L2 round trip overlaps this row's processing
      probe_issue(pb, my_hdr + (size_t)nxt * kHdrVec * 16, recv + (size_t)nxt * kRowBytes, lane);
    // This slot's x was already consumed by its earlier lookahead push.
    mbar_wait(&S.rfull[slot], phase);
    const int4 xv = S.xs[slot][tid], rv = S.rs[slot][tid];
    const uint32_t xa[4] = {(uint32_t)xv.x, (uint32_t)xv.y, (uint32_t)xv.z, (uint32_t)xv.w};
    const uint32_t qa[4] = {(uint32_t)pr.x, (uint32_t)pr.y, (uint32_t)pr.z, (uint32_t)pr.w};
    const uint32_t ra[4] = {(uint32_t)rv.x, (uint32_t)rv.y, (uint32_t)rv.z, (uint32_t)rv.w};
    uint32_t h2[4];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      // all-reduce sum, then residual add (one bf16 rounding each)
      h2[k] = add_bf16x2(add_bf16x2(xa[k], qa[k]), ra[k]);
      ss = fmaf(lo(h2[k]), lo(h2[k]), ss);
      ss = fmaf(hi(h2[k]), hi(h2[k]), ss);
    }
    const size_t base = row * kRowVec + tid;
    // the x / residual row is in shared memory already: safe under in-place aliasing
    st_cs(p.res_out + base, make_int4((int)h2[0], (int)h2[1], (int)h2[2], (int)h2[3]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float* rb = S.red[red_slot];
    if (lane == 0) rb[warp] = ss;
    __syncthreads();  // all warps consumed this ring slot and published their partial sums
    if (tid == 0) {
      int r2 = -1;  // refill the just-consumed slot for its next barrier phase
      if (!fail) {
        r2 = kNB * G + (int)claim;
        if (r2 >= T) {
          r2 = -1;
          fail = true;
        } else {
          claim = atomicAdd(ctr, 1u);
        }
      }
      S.row[slot] = r2;
      if (r2 >= 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_expect_tx(&S.xfull[slot], kRowB);
        bulk_load(S.xs[slot], p.x + (size_t)r2 * kRowVec, kRowB, &S.xfull[slot]);
        mbar_expect_tx(&S.rfull[slot], kRowB);
        bulk_load(S.rs[slot], p.res + (size_t)r2 * kRowVec, kRowB, &S.rfull[slot]);
      } else {
        mbar_arrive(&S.xfull[slot]);
        mbar_arrive(&S.rfull[slot]);
      }
    }
    if (warp == 1 && lane < 3)
      st_v4(recv_hdr + row * kHdrVec + lane,
            make_int4((int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2));
    if (warp == 0 && lane < 3) send_hdr_slot(push_slot_idx);
    // row total: one smem load per lane, then a fixed-order (deterministic) shuffle tree
    float tot = lane < kNW ? rb[lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
    // IEEE division whatever the fast-math flags of the build
    const float rstd = rsqrtf(__fdiv_rn(tot, (float)kHidden) + p.eps);
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      o[k] = pack2(lo(h2[k]) * rstd * gw[2 * k], hi(h2[k]) * rstd * gw[2 * k + 1]);
    // the x row is in shared memory already: safe under in-place aliasing
    st_cs(p.norm_out + base, make_int4((int)o[0], (int)o[1], (int)o[2], (int)o[3]));
    crow = nxt;
    slot = next_slot;
    if (slot == 0) phase ^= 1u;
    red_slot ^= 1;
  }
  if (tid == 0) p.epoch[b] = e + 1u;
}

__global__ void fill_sentinel_kernel(int4* p, size_t n) {
  const int4 v = make_int4((int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2, (int)kNeg0x2);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    p[i] = v;
}

// ---------------------------------------------------------------------------------------------
// Host interface
// ---------------------------------------------------------------------------------------------

// Region layout, identical on both ranks: [data slot 0][data slot 1][hdr slot 0][hdr slot 1],
// each data slot capacity x 12288 B, each header slot capacity x 128 B.
inline size_t slot_bytes(int64_t capacity) { return (size_t)capacity * kRowBytes; }
inline size_t hdr_bytes(int64_t capacity) { return (size_t)capacity * kHdrRowBytes; }

inline size_t region_bytes(int64_t capacity) {
  return 2 * slot_bytes(capacity) + 2 * hdr_bytes(capacity);
}

// Control buffer layout: [per-CTA call counters: grid words, padded to 128 B][claim counters:
// kCtlWords words]. All zero when idle.
inline int64_t ctl_offset(int64_t grid) { return (grid + 31) / 32 * 32; }

inline int64_t epoch_words(int64_t grid) { return ctl_offset(grid) + kCtlWords; }

inline cudaError_t init(void* region, int64_t capacity, uint32_t* epoch, int64_t num_epoch_words,
                        cudaStream_t stream) {
  fill_sentinel_kernel<<<1024, 256, 0, stream>>>(reinterpret_cast<int4*>(region),
                                                 region_bytes(capacity) / sizeof(int4));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  return cudaMemsetAsync(epoch, 0, (size_t)num_epoch_words * sizeof(uint32_t), stream);
}

inline int max_grid(int device) {
  int sms = 0, per_sm = 0, prev = 0;
  if (cudaGetDevice(&prev) != cudaSuccess) return -1;
  if (cudaSetDevice(device) != cudaSuccess) return -1;
  int grid = -1;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
      cudaFuncSetAttribute(ar_rmsnorm_ws2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)sizeof(Smem)) == cudaSuccess &&
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ar_rmsnorm_ws2_kernel, kNT,
                                                    sizeof(Smem)) == cudaSuccess) {
    grid = sms * per_sm;
  }
  (void)cudaGetLastError();
  (void)cudaSetDevice(prev);
  return grid;
}

inline cudaError_t launch(const void* x, const void* residual_in, const void* gamma,
                          void* residual_out, void* norm_out, void* own_region, void* peer_region,
                          uint32_t* epoch, int64_t capacity, int num_tokens, int grid, float eps,
                          float weight_bias, bool launch_with_pdl, cudaStream_t stream) {
  Params p;
  p.x = reinterpret_cast<const int4*>(x);
  p.res = reinterpret_cast<const int4*>(residual_in);
  p.gamma = reinterpret_cast<const int4*>(gamma);
  p.res_out = reinterpret_cast<int4*>(residual_out);
  p.norm_out = reinterpret_cast<int4*>(norm_out);
  p.recv = reinterpret_cast<int4*>(own_region);
  p.peer = reinterpret_cast<int4*>(peer_region);
  p.recv_hdr = reinterpret_cast<int4*>(static_cast<char*>(own_region) + 2 * slot_bytes(capacity));
  p.peer_hdr = reinterpret_cast<int4*>(static_cast<char*>(peer_region) + 2 * slot_bytes(capacity));
  p.epoch = epoch;
  p.ctl = epoch + ctl_offset(grid);
  p.slot_vec = slot_bytes(capacity) / sizeof(int4);
  p.hdr_vec = hdr_bytes(capacity) / sizeof(int4);
  p.T = num_tokens;
  p.eps = eps;
  p.weight_bias = weight_bias;

  cudaError_t err = cudaFuncSetAttribute(
      ar_rmsnorm_ws2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (err != cudaSuccess) return err;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(kNT);
  cfg.dynamicSmemBytes = sizeof(Smem);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, ar_rmsnorm_ws2_kernel, p);
}

}  // namespace allreduce_fusion_ws2
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_ALLREDUCE_FUSION_WS2_SPECIALIZED_CUH_
