// MSA block-sparse GQA decode (q_len 1, 16 heads, 1 KV head, D=128, fp8 paged KV).
// Native-16-bit tensor-core core: on SM100 `mma.sync ... e4m3` is emulated
// (F2FP + HMMA.16816 + FADD), and the emulation re-converts the loop-invariant
// A operand on every instruction.  Dequantise K/V once in registers and issue
// HMMA.16816 directly instead.  QK runs on bf16 (Q is the caller's bf16
// verbatim; e4m3 K widens to bf16 exactly), so no per-head Q scaling is needed
// and wide per-head exponent spreads survive.  P stays f16 for the PV mma but
// is staged as exp2(s - m + PSHIFT): the 2^PSHIFT factor rides O and l alike
// and cancels at normalization, keeping tiny probabilities above the f16
// subnormal floor instead of flushing them to zero against an fp32 denominator.
//
// Framework-independent device code: the kernels and their launch-geometry
// constants, relocated verbatim from the former csrc kernel TU. Included by
// exactly one binding TU, csrc/msa_sparse_decode_fp8kv_sm103_f16core.cu, which
// owns the host-side launch glue and the TVM-FFI entry point. The geometry
// macros below are file-scope by design: never include two of these headers
// in the same translation unit.
#pragma once
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace cg = cooperative_groups;

#define HQ 16
#define DH 128
#define BLKT 128
#define TOPKC 16

#define PGB 16384         // one 128x128 fp8 page, swizzled, row stride 128
#define STAGEB (2 * PGB)  // K then V
#define FSTRIDE 272       // 128 f16 + 16B pad -> conflict-free ldmatrix
#define QBYTES (HQ * FSTRIDE)
#define PBYTES (HQ * FSTRIDE)
#define SCRB 1152  // 128 maxes, 128 row sums (16 former q-scale slots unused)
template <int CLU>
struct Geo {
  static constexpr int NSLOT = ((HQ + CLU - 1) / CLU) * CLU;
  // fp32 partial numerators + (m, l) pairs cross the cluster merge; a
  // single-CTA grid never merges, so it carries no mailbox at all.
  static constexpr int MERGE = (CLU == 1) ? 0 : ((NSLOT * DH * 4 + NSLOT * 8) + 127) / 128 * 128;
};
// P aliases the Q staging buffer: Q is hoisted to registers before the page
// loop on every compiled occupancy (static_assert in the kernel).
#define SMEM_N(N, C) ((N) * STAGEB + Geo<C>::MERGE + QBYTES + SCRB)
#define SMEM_O(N, C, O) SMEM_N(N, C)

#define QSCALE 0.08838834764831845f
#define LOG2EF 1.4426950408889634f
#define SCL (QSCALE * LOG2EF)
// P is staged as exp2(s - m + PSHIFT); the exact power of two rides the O
// accumulators and the row sums alike, so it cancels in every O/l ratio.
#define PSHIFT 14.0f

__device__ __forceinline__ uint32_t sptr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// src-size form: full ? copy 16 bytes : zero-fill the 16-byte destination.
// Tokens at or beyond the sequence end in a partial last page are unwritten
// allocator memory; an E4M3 NaN byte there would survive the score mask as
// 0 * NaN in the PV mma, so the stage itself must hold zeros for them.
__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, bool full) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(full ? 16u : 0u));
}
__device__ __forceinline__ void cpcommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cpwait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_sync_raw() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_init_expect(uint64_t* bar, uint32_t bytes) {
  const uint32_t addr = sptr(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(addr) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr),
               "r"(bytes)
               : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_wait_local(uint64_t* bar) {
  const uint32_t addr = sptr(bar);
  uint32_t done;
  do {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(addr)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void st_async_u32(uint32_t dst, uint32_t value, uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 [%0], %1, [%2];\n" ::"r"(dst),
      "r"(value), "r"(remote_bar)
      : "memory");
}
__device__ __forceinline__ void st_async_u64(uint32_t dst, uint64_t value, uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];\n" ::"r"(dst),
      "l"(value), "r"(remote_bar)
      : "memory");
}
__device__ __forceinline__ void ldm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                     uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(addr));
}
__device__ __forceinline__ void ldm2t_b8(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                         uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(addr));
}
// 4 packed e4m3 -> two f16x2 registers (B fragment of one m16n8k16 mma).
__device__ __forceinline__ void cvt8(uint32_t s, uint32_t& lo, uint32_t& hi) {
  asm("{\n .reg .b16 l_, h_;\n mov.b32 {l_, h_}, %2;\n"
      " cvt.rn.f16x2.e4m3x2 %0, l_;\n cvt.rn.f16x2.e4m3x2 %1, h_;\n}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(s));
}
__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                      uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t f2h2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  union PackedBF16x2 {
    __nv_bfloat162 bf;
    uint32_t u;
  } r;
  r.bf = __floats2bfloat162_rn(lo, hi);
  return r.u;
}
// four packed e4m3 -> the two bf16x2 halves of one m16n8k16 B fragment.
// e4m3 -> f16 is exact and every e4m3 value is exactly representable in
// bf16, so the widening chain loses nothing.
__device__ __forceinline__ void cvt8bf(uint32_t s, uint32_t& lo, uint32_t& hi) {
  uint32_t a, b;
  cvt8(s, a, b);
  float2 fa = __half22float2(*reinterpret_cast<__half2*>(&a));
  float2 fb = __half22float2(*reinterpret_cast<__half2*>(&b));
  lo = f2bf2(fa.x, fa.y);
  hi = f2bf2(fb.x, fb.y);
}
__device__ __forceinline__ void mma_bf16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                         uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Logical contraction index -> slot inside a 16-element group.  ldmatrix hands
// each lane four consecutive bytes while the mma fragment wants {2j,2j+1} and
// {2j+8,2j+9}; applying this permutation to the operand we build ourselves (Q,
// P) makes both sides agree without touching the gathered K/V bytes.
__device__ __forceinline__ int perm16(int t) {
  return (t & ~15) + 2 * ((t & 15) >> 2) + (t & 1) + ((t & 2) ? 8 : 0);
}

template <int NSTAGE, int CLU, int OCC, int OSPLIT = 1>
__global__ void __cluster_dims__(CLU, 1, 1) __launch_bounds__(256, OCC)
    msaf16_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                  const uint8_t* __restrict__ vc, const int* __restrict__ topk,
                  const int* __restrict__ ptab, const int* __restrict__ seqlens,
                  __nv_bfloat16* __restrict__ outp, int pages, int bpc, int nblk) {
  constexpr bool QUARTER = (OSPLIT > 1);  // any output split
  constexpr bool ASYNC_MERGE =
      (CLU == 16 && (OSPLIT == 4 || OSPLIT == 2)) ||
      (CLU == 3 && (OSPLIT == 8 || OSPLIT == 4 || OSPLIT == 2 || OSPLIT == 1));
  const int b = (OSPLIT > 1) ? (blockIdx.y / OSPLIT) : blockIdx.y;
  const int dgroup = (OSPLIT > 1) ? (blockIdx.y % OSPLIT) : 0;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int tig = lane & 3;

  extern __shared__ __align__(128) uint8_t sm[];
  // The last 64 bytes of SCRB are unused (scr/scrl occupy 1088 of 1152 B).
  // One owner-local transaction barrier replaces the all-CTA cluster barrier
  // on selected output-split merge paths.
  uint64_t* merge_bar = reinterpret_cast<uint64_t*>(sm + SMEM_O(NSTAGE, CLU, OSPLIT) - 64);
  if constexpr (ASYNC_MERGE) {
    const uint32_t owned_heads = (HQ - sp + CLU - 1) / CLU;
    // fp32 partials: DH * 4 bytes of O per contributing CTA (split OSPLIT
    // ways) plus the 8-byte (m, l) pair.
    if (tid == 0) mbar_init_expect(merge_bar, owned_heads * CLU * (512 / OSPLIT + 8));
  }
  if constexpr (CLU > 1) cluster_arrive_relaxed();

  const __nv_bfloat16* qptr = q + (size_t)b * (HQ * DH) + (tid >> 4) * DH + (tid & 15) * 8;
  uint4 qw;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(qw.x), "=r"(qw.y), "=r"(qw.z), "=r"(qw.w)
               : "l"(qptr));
  int seqlen;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(seqlen) : "l"(seqlens + b));
  const int* pt = ptab + (size_t)b * pages;
  int pv0 = 0, pv1 = 0;
  if (pages <= 64) {
    if (lane < pages) pv0 = pt[lane];
    if (lane + 32 < pages) pv1 = pt[lane + 32];
  }

  const int i0 = sp * bpc;
  const int i1 = min(i0 + bpc, nblk);
  const int lim = i1 - i0;
  int jj = i0 + (lane & 15);
  if (jj > TOPKC - 1) jj = TOPKC - 1;
  int blkr;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(blkr) : "l"(topk + b * TOPKC + jj));

  // Q dies into registers (QHOIST below) before the first P write, so P
  // aliases the Q staging buffer on every route.
  static_assert(OCC < 3, "P aliases Q; requires the register-hoisted Q path");
  uint8_t* Sb = sm;
  uint8_t* Mg = Sb + NSTAGE * STAGEB;
  uint8_t* Qs = Mg + Geo<CLU>::MERGE;
  uint8_t* Pb = Qs;
  float* scr = (float*)(Pb + QBYTES);
  float* scrl = scr + 144;

  int pgr;
  if (pages <= 64) {
    int bc = blkr >= 0 ? blkr : 0;
    int s0 = __shfl_sync(0xffffffffu, pv0, bc & 31);
    int s1 = __shfl_sync(0xffffffffu, pv1, bc & 31);
    pgr = (bc < 32) ? s0 : s1;
  } else {
    pgr = pt[blkr >= 0 ? blkr : 0];
  }
  unsigned vmask = __ballot_sync(
      0xffffffffu, lane < 16 && lane < lim && blkr >= 0 && int64_t(blkr) * BLKT < seqlen);
  const int nb = __popc(vmask & 0xffffu);
  const bool needs_mask = (seqlen != pages * BLKT);

  const int tbase0 = 16 * warp;
  int soff[4];
#pragma unroll
  for (int t2 = 0; t2 < 4; ++t2) {
    int u_ = tid + 256 * t2, tok_ = u_ >> 3, c_ = u_ & 7;
    soff[t2] = (tok_ << 7) | ((c_ ^ (tok_ & 7)) << 4);
  }
  const uint8_t* kbase = kc + (size_t)tid * 16;
  const uint8_t* vbase = vc + (size_t)tid * 16;
  const uint32_t soff0 = (uint32_t)soff[0];
#define ISSUEK(J, ST)                                                                           \
  do {                                                                                          \
    if ((J) < nb) {                                                                             \
      int pg_ = __shfl_sync(0xffffffffu, pgr, (J));                                             \
      const uint8_t* kg_ = kbase + (size_t)pg_ * PGB;                                           \
      const uint32_t sd_ = sptr(Sb + (ST) * STAGEB) + soff0;                                    \
      _Pragma("unroll") for (int t2 = 0; t2 < 4; ++t2) cp16(sd_ + 4096u * t2, kg_ + 4096 * t2); \
    }                                                                                           \
    cpcommit();                                                                                 \
  } while (0)
#define ISSUEV(J, ST)                                                              \
  do {                                                                             \
    if ((J) < nb) {                                                                \
      int pg_ = __shfl_sync(0xffffffffu, pgr, (J));                                \
      int lim_ = BLKT;                                                             \
      if (needs_mask) {                                                            \
        int cb_ = __shfl_sync(0xffffffffu, blkr, (J));                             \
        lim_ = seqlen - cb_ * BLKT;                                                \
      }                                                                            \
      uint8_t* kd_ = Sb + (ST) * STAGEB;                                           \
      const uint8_t* vg_ = vc + (size_t)pg_ * PGB;                                 \
      uint8_t* vd_ = kd_ + PGB;                                                    \
      if constexpr (OSPLIT == 8) {                                                 \
        if (tid < 128) {                                                           \
          int tok_ = tid, c_ = dgroup;                                             \
          int vo_ = (tok_ << 7) | ((c_ ^ (tok_ & 7)) << 4);                        \
          cp16z(sptr(vd_ + vo_), vg_ + ((size_t)tok_ * 8 + c_) * 16, tok_ < lim_); \
        }                                                                          \
      } else if constexpr (OSPLIT == 4) {                                          \
        int ig_ = lane >> 3;                                                       \
        int tok_ = 16 * warp + 8 * (ig_ >> 1) + 2 * ((lane & 7) >> 1) + (ig_ & 1); \
        int c_ = 2 * dgroup + (lane & 1);                                          \
        int vo_ = (tok_ << 7) | ((c_ ^ (tok_ & 7)) << 4);                          \
        cp16z(sptr(vd_ + vo_), vg_ + ((size_t)tok_ * 8 + c_) * 16, tok_ < lim_);   \
      } else if constexpr (OSPLIT == 2) {                                          \
        _Pragma("unroll") for (int t2 = 0; t2 < 2; ++t2) {                         \
          int tok_, c_;                                                            \
          if constexpr (CLU == 16) {                                               \
            tok_ = 64 * t2 + 8 * warp + (lane >> 3) + 4 * ((lane & 7) >> 2);       \
            c_ = 4 * dgroup + (lane & 3);                                          \
          } else {                                                                 \
            int u_ = tid + 256 * t2;                                               \
            tok_ = u_ >> 2;                                                        \
            c_ = 4 * dgroup + (u_ & 3);                                            \
          }                                                                        \
          int vo_ = (tok_ << 7) | ((c_ ^ (tok_ & 7)) << 4);                        \
          cp16z(sptr(vd_ + vo_), vg_ + ((size_t)tok_ * 8 + c_) * 16, tok_ < lim_); \
        }                                                                          \
      } else {                                                                     \
        const uint32_t vsd_ = sptr(vd_) + soff0;                                   \
        _Pragma("unroll") for (int t2 = 0; t2 < 4; ++t2)                           \
            cp16z(vsd_ + 4096u * t2, vbase + (size_t)pg_ * PGB + 4096 * t2,        \
                  ((tid >> 3) + 32 * t2) < lim_);                                  \
      }                                                                            \
    }                                                                              \
    cpcommit();                                                                    \
  } while (0)

#pragma unroll
  for (int s_ = 0; s_ < NSTAGE; ++s_) {
    ISSUEK(s_, s_);
    ISSUEV(s_, s_);
  }
  if constexpr (CLU > 1) cluster_wait_acquire();

  // ---- Q: the caller's bf16 verbatim (QK runs on bf16 HMMA, so no per-head
  // scaling is needed and no exponent range is clipped).
  {
    const int h = tid >> 4, dg = tid & 15;
    uint32_t raw[4] = {qw.x, qw.y, qw.z, qw.w};
    const int d0 = dg * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int d = d0 + 2 * i;
      *(uint32_t*)(Qs + h * FSTRIDE + perm16(d) * 2) = raw[i];
    }
  }
  if (nb == 0 && tid < 128) scrl[tid] = 0.f;
  __syncthreads();
  const float fac0 = SCL;
  const float fac1 = SCL;

  float O[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) O[i][j] = 0.f;
  float m0 = 0.f, m1 = 0.f, l0 = 0.f, l1 = 0.f;

  const uint32_t frow = (uint32_t)(((lane & 7) + ((lane >> 3) & 1) * 8) * FSTRIDE);
  const uint32_t fhalf = (uint32_t)((lane >> 4) * 16);
  const uint32_t qbase = sptr(Qs) + frow + fhalf;
  const uint32_t pbase = sptr(Pb) + frow + fhalf;
  constexpr bool QHOIST = (OCC < 3);
  uint32_t qa[QHOIST ? 8 : 1][4];
  if constexpr (QHOIST) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
      ldm4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], qbase + (uint32_t)(ks * 32));
  }
  const uint32_t kswz = (uint32_t)(((lane >> 3) ^ (lane & 7)) << 4);
  const uint32_t kswz2 = (uint32_t)((((lane >> 3) | 4) ^ (lane & 7)) << 4);
  const uint32_t krow = (uint32_t)((tbase0 + (lane & 7)) << 7);
  constexpr int OWARPS = 8 / OSPLIT;
  const int owarp = (OSPLIT > 1) ? (warp + OWARPS * dgroup) : warp;
  const uint32_t vswz = (uint32_t)(((owarp ^ (lane & 7)) << 4));
  int st = 0;
  for (int j = 0; j < nb; ++j) {
    const int cb = needs_mask ? __shfl_sync(0xffffffffu, blkr, j) : 0;
    cpwait<2 * NSTAGE - 1>();
    __syncthreads();

    uint8_t* kd = Sb + st * STAGEB;
    uint8_t* vd = kd + PGB;

    // ---- QK^T : 16 heads x 16 tokens per warp, 8 k-steps of 16 dims ----
    float S[2][4];
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2)
#pragma unroll
      for (int x = 0; x < 4; ++x) S[t2][x] = 0.f;
    {
      const uint32_t kb = sptr(kd) + krow;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t ka[4], kbb[4];
        ldm4(ka[0], ka[1], ka[2], ka[3], kb + (half ? kswz2 : kswz));
        ldm4(kbb[0], kbb[1], kbb[2], kbb[3], kb + (uint32_t)(1 << 10) + (half ? kswz2 : kswz));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int ks = half * 4 + c;
          uint32_t a0, a1, a2, a3;
          if constexpr (QHOIST) {
            a0 = qa[ks][0];
            a1 = qa[ks][1];
            a2 = qa[ks][2];
            a3 = qa[ks][3];
          } else {
            ldm4(a0, a1, a2, a3, qbase + (uint32_t)(ks * 32));
          }
          uint32_t b0, b1;
          cvt8bf(ka[c], b0, b1);
          mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], a0, a1, a2, a3, b0, b1);
          cvt8bf(kbb[c], b0, b1);
          mma_bf16(S[1][0], S[1][1], S[1][2], S[1][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }

    // ---- scale + mask + row max ----
    const int cnt = needs_mask ? min(seqlen - cb * BLKT, BLKT) : BLKT;
    float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2) {
      int t0 = tbase0 + 8 * t2 + 2 * tig;
      float v0 = S[t2][0] * fac0, v1 = S[t2][1] * fac0;
      float v2 = S[t2][2] * fac1, v3 = S[t2][3] * fac1;
      if (needs_mask) {
        if (t0 >= cnt) {
          v0 = -INFINITY;
          v2 = -INFINITY;
        }
        if (t0 + 1 >= cnt) {
          v1 = -INFINITY;
          v3 = -INFINITY;
        }
      }
      S[t2][0] = v0;
      S[t2][1] = v1;
      S[t2][2] = v2;
      S[t2][3] = v3;
      rmax0 = fmaxf(rmax0, fmaxf(v0, v1));
      rmax1 = fmaxf(rmax1, fmaxf(v2, v3));
    }
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));
    if (tig == 0) {
      scr[g * 8 + warp] = rmax0;
      scr[(g + 8) * 8 + warp] = rmax1;
    }
    cpwait<2 * NSTAGE - 2>();
    __syncthreads();

    if constexpr (NSTAGE > 1) ISSUEK(j + NSTAGE, st);

    float4 la = *(const float4*)(scr + g * 8);
    float4 lb = *(const float4*)(scr + g * 8 + 4);
    float4 lc = *(const float4*)(scr + (g + 8) * 8);
    float4 ld = *(const float4*)(scr + (g + 8) * 8 + 4);
    float bm0 = fmaxf(fmaxf(fmaxf(la.x, la.y), fmaxf(la.z, la.w)),
                      fmaxf(fmaxf(lb.x, lb.y), fmaxf(lb.z, lb.w)));
    float bm1 = fmaxf(fmaxf(fmaxf(lc.x, lc.y), fmaxf(lc.z, lc.w)),
                      fmaxf(fmaxf(ld.x, ld.y), fmaxf(ld.z, ld.w)));
    float nm0 = j == 0 ? bm0 : fmaxf(m0, bm0);
    float nm1 = j == 0 ? bm1 : fmaxf(m1, bm1);
    if (j != 0) {
      float al0 = exp2f(m0 - nm0), al1 = exp2f(m1 - nm1);
#pragma unroll
      for (int nd = 0; nd < 2; ++nd) {
        O[nd][0] *= al0;
        O[nd][1] *= al0;
        O[nd][2] *= al1;
        O[nd][3] *= al1;
      }
      l0 *= al0;
      l1 *= al1;
    }
    m0 = nm0;
    m1 = nm1;

    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2) {
      // 2^PSHIFT-scaled staging: p <= 2^PSHIFT stays inside f16 range while
      // probabilities down to 2^-24-PSHIFT of the max survive staging.  O and
      // l both carry the factor, so every O/l normalization cancels it.
      float p0 = exp2f(S[t2][0] - nm0 + PSHIFT);
      float p1 = exp2f(S[t2][1] - nm0 + PSHIFT);
      float p2 = exp2f(S[t2][2] - nm1 + PSHIFT);
      float p3 = exp2f(S[t2][3] - nm1 + PSHIFT);
      sum0 += p0 + p1;
      sum1 += p2 + p3;
      int tok = perm16(tbase0 + 8 * t2 + 2 * tig);
      *(uint32_t*)(Pb + g * FSTRIDE + tok * 2) = f2h2(p0, p1);
      *(uint32_t*)(Pb + (g + 8) * FSTRIDE + tok * 2) = f2h2(p2, p3);
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    l0 += sum0;
    l1 += sum1;
    if (tig == 0) {
      scrl[g * 8 + warp] = l0;
      scrl[(g + 8) * 8 + warp] = l1;
    }
    __syncthreads();

    // ---- PV : this warp owns 16 output dims, 8 k-steps of 16 tokens ----
    if (!QUARTER || warp < OWARPS) {
      const uint32_t vb = sptr(vd) + (uint32_t)(lane << 7) + vswz;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t vf[4];
        ldm2t_b8(vf[0], vf[1], vf[2], vf[3], vb + (uint32_t)(ks << 12));
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t a0, a1, a2, a3;
          ldm4(a0, a1, a2, a3, pbase + (uint32_t)((2 * ks + half) * 32));
          uint32_t b0, b1;
          cvt8(vf[0 + 2 * half], b0, b1);
          mma16(O[0][0], O[0][1], O[0][2], O[0][3], a0, a1, a2, a3, b0, b1);
          cvt8(vf[1 + 2 * half], b0, b1);
          mma16(O[1][0], O[1][1], O[1][2], O[1][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }
    if constexpr (NSTAGE > 1) {
      if (j + NSTAGE < nb) __syncthreads();
      ISSUEV(j + NSTAGE, st);
    } else if (j + 1 < nb) {
      // Single-stage pipelines still have to refill the one stage they own;
      // the barrier covers the V reads the PV above just finished.
      __syncthreads();
      ISSUEK(j + 1, 0);
      ISSUEV(j + 1, 0);
    }
    if (++st == NSTAGE) st = 0;
  }

  // ---------------- epilogue ----------------
  if constexpr (CLU == 1) {
    float4 la = *(const float4*)(scrl + g * 8);
    float4 lb = *(const float4*)(scrl + g * 8 + 4);
    float4 lc = *(const float4*)(scrl + (g + 8) * 8);
    float4 ld = *(const float4*)(scrl + (g + 8) * 8 + 4);
    float L0 = (la.x + la.y) + (la.z + la.w) + (lb.x + lb.y) + (lb.z + lb.w);
    float L1 = (lc.x + lc.y) + (lc.z + lc.w) + (ld.x + ld.y) + (ld.z + ld.w);
    // empty selection -> zero output (packed-backend convention)
    float i0f = L0 > 0.f ? 1.0f / L0 : 0.f, i1f = L1 > 0.f ? 1.0f / L1 : 0.f;
#pragma unroll
    for (int nd = 0; nd < 2; ++nd) {
      int d = 16 * warp + 8 * nd + 2 * tig;
      __nv_bfloat162 r0 = __floats2bfloat162_rn(O[nd][0] * i0f, O[nd][1] * i0f);
      __nv_bfloat162 r1 = __floats2bfloat162_rn(O[nd][2] * i1f, O[nd][3] * i1f);
      *(__nv_bfloat162*)(outp + (size_t)(b * HQ + g) * DH + d) = r0;
      *(__nv_bfloat162*)(outp + (size_t)(b * HQ + g + 8) * DH + d) = r1;
    }
    return;
  }

  {
    constexpr int NSLOT = Geo<CLU>::NSLOT;
    // fp32 partial numerators: rounding them to bf16 here can cancel the
    // entire residual of adversarially opposing pages in the merge.
    float* oh = (float*)Mg;
    float* ml = oh + NSLOT * DH;
    cg::cluster_group clu = cg::this_cluster();
    const int ha = g, hb = g + 8;
    float* oa = (float*)clu.map_shared_rank(oh, ha % CLU);
    float* ob = (float*)clu.map_shared_rank(oh, hb % CLU);
    const int sa = (ha / CLU) * CLU + sp;
    const int sb = (hb / CLU) * CLU + sp;
    uint64_t* bara = clu.map_shared_rank(merge_bar, ha % CLU);
    uint64_t* barb = clu.map_shared_rank(merge_bar, hb % CLU);
    if (!QUARTER || warp < OWARPS) {
#pragma unroll
      for (int nd = 0; nd < 2; ++nd) {
        int d = 16 * owarp + 8 * nd + 2 * tig;
        union Pack64 {
          float2 f;
          uint64_t u;
        } va, vb;
        va.f = make_float2(O[nd][0], O[nd][1]);
        vb.f = make_float2(O[nd][2], O[nd][3]);
        if constexpr (ASYNC_MERGE) {
          st_async_u64(sptr(oa + sa * DH + d), va.u, sptr(bara));
          st_async_u64(sptr(ob + sb * DH + d), vb.u, sptr(barb));
        } else {
          *reinterpret_cast<float2*>(oa + sa * DH + d) = va.f;
          *reinterpret_cast<float2*>(ob + sb * DH + d) = vb.f;
        }
      }
    }
    if (warp == 0 && tig == 0) {
      float4 la = *(const float4*)(scrl + g * 8);
      float4 lb = *(const float4*)(scrl + g * 8 + 4);
      float4 lc = *(const float4*)(scrl + (g + 8) * 8);
      float4 ld = *(const float4*)(scrl + (g + 8) * 8 + 4);
      float L0 = (la.x + la.y) + (la.z + la.w) + (lb.x + lb.y) + (lb.z + lb.w);
      float L1 = (lc.x + lc.y) + (lc.z + lc.w) + (ld.x + ld.y) + (ld.z + ld.w);
      float* mla = (float*)clu.map_shared_rank(ml, ha % CLU);
      float* mlb = (float*)clu.map_shared_rank(ml, hb % CLU);
      if constexpr (ASYNC_MERGE) {
        union Pack64 {
          float2 f;
          uint64_t u;
        } pa, pb;
        pa.f = make_float2(m0, L0);
        pb.f = make_float2(m1, L1);
        st_async_u64(sptr(mla + 2 * sa), pa.u, sptr(bara));
        st_async_u64(sptr(mlb + 2 * sb), pb.u, sptr(barb));
      } else {
        *(float2*)(mla + 2 * sa) = make_float2(m0, L0);
        *(float2*)(mlb + 2 * sb) = make_float2(m1, L1);
      }
    }
    const int nh = (HQ - sp + CLU - 1) / CLU;
    constexpr int DLANES = 64 / OSPLIT;  // threads covering this CTA's dims
    const int dd = dgroup * (128 / OSPLIT) + (tid & (DLANES - 1)) * 2;
    const int htid = tid / DLANES;
    constexpr int HSTEP = 256 / DLANES;
    if constexpr (ASYNC_MERGE) {
      // Only threads that consume this owner-local mailbox need to acquire it;
      // other threads can retire without paying a CTA-wide tail barrier.
      // This is especially sparse for C16 (one owned head), but also avoids
      // releasing idle head groups on the C3 output-split routes.
      if (htid >= nh) return;
      mbar_wait_local(merge_bar);
    } else {
      cluster_sync_raw();
    }
    for (int hi = htid; hi < nh; hi += HSTEP) {
      float wv[CLU];
      float mm = -INFINITY;
#pragma unroll
      for (int c = 0; c < CLU; ++c) {
        wv[c] = ml[2 * (hi * CLU + c)];
        if (ml[2 * (hi * CLU + c) + 1] > 0.f) mm = fmaxf(mm, wv[c]);
      }
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < CLU; ++c) {
        const float mass = ml[2 * (hi * CLU + c) + 1];
        wv[c] = mass > 0.f ? exp2f(wv[c] - mm) : 0.f;
        den += mass * wv[c];
      }
      float inv = (den > 0.f) ? (1.0f / den) : 0.f;
      float a0 = 0.f, a1 = 0.f;
#pragma unroll
      for (int c = 0; c < CLU; ++c) {
        float2 o = *(const float2*)(oh + (hi * CLU + c) * DH + dd);
        float w = wv[c] * inv;
        a0 += o.x * w;
        a1 += o.y * w;
      }
      *(__nv_bfloat162*)(outp + (size_t)b * (HQ * DH) + (sp + hi * CLU) * DH + dd) =
          __floats2bfloat162_rn(a0, a1);
    }
  }
}

// ---------------------------------------------------------------------------
