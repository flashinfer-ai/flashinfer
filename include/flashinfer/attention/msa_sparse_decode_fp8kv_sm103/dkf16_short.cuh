// MSA block-sparse GQA decode -- direct-K variant.
//
// K never enters shared memory: each warp pulls its own 16 KV rows straight from
// global into the mma B operand with 16-byte vector loads, using a contraction
// order that makes the four k-steps the four words of one uint4.  Only V is
// cp.async staged, so the page loop needs just two __syncthreads (row-max
// exchange + P publication) instead of four, shared-memory traffic per page
// drops by ~45%, and the K latency is prefetched a whole page ahead in registers.
//
// Framework-independent device code: the kernels and their launch-geometry
// constants, relocated verbatim from the former csrc kernel TU. Included by
// exactly one binding TU, csrc/msa_sparse_decode_fp8kv_sm103_dkf16_short.cu, which
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

namespace dkf16short {

#define HQ 16
#define DH 128
#define BLKT 128
#define TOPKC 16

#define PGB 16384
#define FSTRIDE 272            // 128 f16 + 16B pad
#define PBYTES (HQ * FSTRIDE)  // 4352, also hosts the q staging buffer
#define SCRB 1152              // 144 maxes/scales + 128 row sums

template <int CLU>
struct Geo {
  static constexpr int NSLOT = ((HQ + CLU - 1) / CLU) * CLU;
  // fp32 partial numerators + (m, l) pairs cross the cluster merge; a
  // single-CTA grid never merges, so it carries no mailbox at all.
  static constexpr int MERGE = (CLU == 1) ? 0 : ((NSLOT * DH * 4 + NSLOT * 8) + 127) / 128 * 128;
};
#define SMEM_F(N, C) ((N) * PGB + Geo<C>::MERGE + PBYTES + SCRB)

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
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire_merge() {
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
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
      "[%0], %1, [%2];\n" ::"r"(dst),
      "r"(value), "r"(remote_bar)
      : "memory");
}
__device__ __forceinline__ void st_async_u64(uint32_t dst, uint64_t value, uint32_t remote_bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 "
      "[%0], %1, [%2];\n" ::"r"(dst),
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
__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                      uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four packed e4m3 -> the two f16x2 halves of one m16n8k16 B fragment
__device__ __forceinline__ void cvt8(uint32_t s, uint32_t& lo, uint32_t& hi) {
  asm("{\n .reg .b16 l_, h_;\n mov.b32 {l_, h_}, %2;\n"
      " cvt.rn.f16x2.e4m3x2 %0, l_;\n cvt.rn.f16x2.e4m3x2 %1, h_;\n}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(s));
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
// ldmatrix.trans.b8 hands a lane four consecutive V tokens while the mma wants
// k slots {2j,2j+1} and {2j+8,2j+9}; P is built by us, so permute P instead.
__device__ __forceinline__ int perm16(int t) {
  return (t & ~15) + 2 * ((t & 15) >> 2) + (t & 1) + ((t & 2) ? 8 : 0);
}
__device__ __forceinline__ uint4 ldg128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ void ldg256(const void* p, uint4& lo, uint4& hi) {
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w), "=r"(hi.x), "=r"(hi.y), "=r"(hi.z),
                 "=r"(hi.w)
               : "l"(p));
}

template <int NSTAGE, int CLU, int OCC, bool HALFOUT = false>
__global__ void __cluster_dims__(CLU, 1, 1) __launch_bounds__(256, OCC)
    dkf_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
               const uint8_t* __restrict__ vc, const int* __restrict__ topk,
               const int* __restrict__ ptab, const int* __restrict__ seqlens,
               __nv_bfloat16* __restrict__ outp, int pages, int bpc, int nblk) {
  const int b = HALFOUT ? (blockIdx.y >> 1) : blockIdx.y;
  const int dgroup = HALFOUT ? (blockIdx.y & 1) : 0;
  const int sp = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;

  const __nv_bfloat16* qptr = q + (size_t)b * (HQ * DH) + (tid >> 4) * DH + (tid & 15) * 8;
  uint4 qw = ldg128(qptr);
  int seqlen;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(seqlen) : "l"(seqlens + b));

  const int i0 = sp * bpc;
  const int i1 = min(i0 + bpc, nblk);
  const int lim = i1 - i0;
  int jj = i0 + (lane & 15);
  if (jj > TOPKC - 1) jj = TOPKC - 1;
  int blkr;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(blkr) : "l"(topk + b * TOPKC + jj));
  const int* pt = ptab + (size_t)b * pages;
  int pv0 = 0, pv1 = 0;
  if (pages <= 64) {
    if (lane < pages) pv0 = pt[lane];
    if (lane + 32 < pages) pv1 = pt[lane + 32];
  }

  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* Vb = sm;
  uint8_t* Mg = Vb + NSTAGE * PGB;
  uint8_t* Pb = Mg + Geo<CLU>::MERGE;
  uint8_t* Qs = Pb;  // q dies before the first P write
  float* scr = (float*)(Pb + PBYTES);
  float* scrl = scr + 144;
  uint64_t* merge_bar = reinterpret_cast<uint64_t*>(sm + SMEM_F(NSTAGE, CLU) - 64);
  if constexpr (CLU > 1) {
    const uint32_t owned_heads = (HQ - sp + CLU - 1) / CLU;
    // fp32 partials: DH * 4 bytes of O per contributing CTA (halved under
    // HALFOUT) plus the 8-byte (m, l) pair.
    constexpr uint32_t out_bytes = HALFOUT ? 256 : 512;
    if (tid == 0) mbar_init_expect(merge_bar, owned_heads * CLU * (out_bytes + 8));
  }
  if constexpr (CLU > 1) cluster_arrive_relaxed();

  int pgr;
  if (pages <= 64) {
    int bc = blkr >= 0 ? blkr : 0;
    int s0 = __shfl_sync(0xffffffffu, pv0, bc & 31);
    int s1 = __shfl_sync(0xffffffffu, pv1, bc & 31);
    pgr = (bc < 32) ? s0 : s1;
  } else {
    pgr = pt[blkr >= 0 ? blkr : 0];
  }
  // Contract (documented at the Python wrapper): valid q2k entries are an
  // ascending prefix, -1-padded at the tail only, so counting them and
  // consuming lanes 0..nb-1 is exact. An interior -1 is out of contract.
  unsigned vmask = __ballot_sync(0xffffffffu, (lane < 16) && (lane < lim) && (blkr >= 0));
  const int nb = __popc(vmask & 0xffffu);
  const bool needs_mask = (seqlen != pages * BLKT);

  const int tok0 = tid >> 3, c0 = tid & 7;
  const uint32_t soff0 = (uint32_t)((tok0 << 7) | ((c0 ^ (tok0 & 7)) << 4));
  const uint8_t* vbase = vc + (size_t)tid * 16;
  const uint32_t svbase = sptr(Vb) + soff0;
  const int htok0 = tid >> 2;
  const int hc0 = 4 * dgroup + (tid & 3);
  const uint32_t hsoff0 = (uint32_t)((htok0 << 7) | ((hc0 ^ (htok0 & 7)) << 4));
  const uint8_t* hvbase = vc + (size_t)htok0 * DH + hc0 * 16;
  const uint32_t hsvbase = sptr(Vb) + hsoff0;
#define ISSUEV(J, ST)                                    \
  do {                                                   \
    if ((J) < nb) {                                      \
      int pg_ = __shfl_sync(0xffffffffu, pgr, (J));      \
      int lim_ = BLKT;                                   \
      if (needs_mask) {                                  \
        int cb_ = __shfl_sync(0xffffffffu, blkr, (J));   \
        lim_ = seqlen - cb_ * BLKT;                      \
      }                                                  \
      if constexpr (HALFOUT) {                           \
        uint32_t d_ = hsvbase + (uint32_t)((ST) * PGB);  \
        const uint8_t* s_ = hvbase + (size_t)pg_ * PGB;  \
        cp16z(d_, s_, htok0 < lim_);                     \
        cp16z(d_ + 8192, s_ + 8192, htok0 + 64 < lim_);  \
      } else {                                           \
        uint32_t d_ = svbase + (uint32_t)((ST) * PGB);   \
        const uint8_t* s_ = vbase + (size_t)pg_ * PGB;   \
        cp16z(d_, s_, tok0 < lim_);                      \
        cp16z(d_ + 4096, s_ + 4096, tok0 + 32 < lim_);   \
        cp16z(d_ + 8192, s_ + 8192, tok0 + 64 < lim_);   \
        cp16z(d_ + 12288, s_ + 12288, tok0 + 96 < lim_); \
      }                                                  \
    }                                                    \
    cpcommit();                                          \
  } while (0)

  const int tbase0 = 16 * warp;
  const uint32_t koff = (uint32_t)((tbase0 + g) * DH + 32 * tig);
  uint4 ka[2], kb[2];
#define LOADK(J)                                        \
  do {                                                  \
    int pg_ = __shfl_sync(0xffffffffu, pgr, (J));       \
    const uint8_t* kg_ = kc + (size_t)pg_ * PGB + koff; \
    ldg256(kg_, ka[0], kb[0]);                          \
    ldg256(kg_ + 8 * DH, ka[1], kb[1]);                 \
  } while (0)
  if (nb > 0) LOADK(0);
#pragma unroll
  for (int s_ = 0; s_ < NSTAGE; ++s_) ISSUEV(s_, s_);

  // ---- q: the caller's bf16 verbatim, in the natural d order (QK runs on
  // bf16 HMMA, so no per-head scaling is needed and no exponent is clipped) ----
  {
    const int h = tid >> 4, dg = tid & 15;
    uint32_t raw[4] = {qw.x, qw.y, qw.z, qw.w};
    const int qchunk = dg ^ ((dg >> 2) & 2);
#pragma unroll
    for (int i = 0; i < 4; ++i) *(uint32_t*)(Qs + h * FSTRIDE + qchunk * 16 + 4 * i) = raw[i];
  }
  __syncthreads();

  // Permute 16-byte Q chunks within each padded row so every vector-load
  // instruction spans distinct bank groups while preserving fragment order.
  uint4 qa[4], qb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = 4 * tig + i;
    const int qchunk = c ^ ((c >> 2) & 2);
    qa[i] = *(const uint4*)(Qs + g * FSTRIDE + 16 * qchunk);
    qb[i] = *(const uint4*)(Qs + (g + 8) * FSTRIDE + 16 * qchunk);
  }
  const float fac0 = SCL;
  const float fac1 = SCL;

  float O[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) O[i][j] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  const int prow = (lane & 7) + ((lane >> 3) & 1) * 8;
  const uint32_t pbase = sptr(Pb) + (uint32_t)(prow * FSTRIDE + (lane >> 4) * 16);
  const int owarp = HALFOUT ? warp + 4 * dgroup : warp;
  const uint32_t vswz = (uint32_t)(((owarp ^ (lane & 7)) << 4));
  int st = 0;
#pragma unroll 1
  for (int j = 0; j < nb; ++j) {
    const int cb = needs_mask ? __shfl_sync(0xffffffffu, blkr, j) : 0;

    float S[2][4];
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2)
#pragma unroll
      for (int x = 0; x < 4; ++x) S[t2][x] = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const int i = w >> 1;
      const uint32_t a0 = (w & 1) ? qa[i].z : qa[i].x;
      const uint32_t a1 = (w & 1) ? qb[i].z : qb[i].x;
      const uint32_t a2 = (w & 1) ? qa[i].w : qa[i].y;
      const uint32_t a3 = (w & 1) ? qb[i].w : qb[i].y;
      uint32_t kw0 = (w < 4) ? ((const uint32_t*)&ka[0])[w] : ((const uint32_t*)&kb[0])[w - 4];
      uint32_t kw1 = (w < 4) ? ((const uint32_t*)&ka[1])[w] : ((const uint32_t*)&kb[1])[w - 4];
      uint32_t b0, b1;
      cvt8bf(kw0, b0, b1);
      mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], a0, a1, a2, a3, b0, b1);
      cvt8bf(kw1, b0, b1);
      mma_bf16(S[1][0], S[1][1], S[1][2], S[1][3], a0, a1, a2, a3, b0, b1);
    }
    if (j + 1 < nb) LOADK(j + 1);

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

    cpwait<NSTAGE - 1>();
    __syncthreads();  // row maxes visible + V stage ready

    uint8_t* vd = Vb + st * PGB;
    uint32_t vf[4][4];
    const uint32_t vb = sptr(vd) + (uint32_t)(lane << 7) + vswz;
    if (!HALFOUT || warp < 4) {
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
        ldm2t_b8(vf[gq][0], vf[gq][1], vf[gq][2], vf[gq][3], vb + (uint32_t)(gq << 12));
    }

    float4 r0a = *(const float4*)(scr + g * 8);
    float4 r0b = *(const float4*)(scr + g * 8 + 4);
    float bm0 = fmaxf(fmaxf(fmaxf(r0a.x, r0a.y), fmaxf(r0a.z, r0a.w)),
                      fmaxf(fmaxf(r0b.x, r0b.y), fmaxf(r0b.z, r0b.w)));

    float4 r1a = *(const float4*)(scr + (g + 8) * 8);
    float4 r1b = *(const float4*)(scr + (g + 8) * 8 + 4);
    float bm1 = fmaxf(fmaxf(fmaxf(r1a.x, r1a.y), fmaxf(r1a.z, r1a.w)),
                      fmaxf(fmaxf(r1b.x, r1b.y), fmaxf(r1b.z, r1b.w)));

    float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
    if (j != 0) {
      float al0 = exp2f(m0 - nm0), al1 = exp2f(m1 - nm1);
      if (!HALFOUT || warp < 4) {
#pragma unroll
        for (int nd = 0; nd < 2; ++nd) {
          O[nd][0] *= al0;
          O[nd][1] *= al0;
          O[nd][2] *= al1;
          O[nd][3] *= al1;
        }
      }
      l0 *= al0;
      l1 *= al1;
    }
    m0 = nm0;
    m1 = nm1;

    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int t2 = 0; t2 < 2; ++t2) {
      // 2^PSHIFT-scaled staging keeps tiny probabilities above the f16
      // subnormal floor; O and l both carry the factor, so it cancels.
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
    l0 += sum0;
    l1 += sum1;

    __syncthreads();  // P complete + V stage consumed
    ISSUEV(j + NSTAGE, st);

    if (!HALFOUT || warp < 4) {
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int ks = 2 * gq + half;
          uint32_t a0, a1, a2, a3;
          ldm4(a0, a1, a2, a3, pbase + (uint32_t)(ks * 32));
          uint32_t b0, b1;
          cvt8(vf[gq][0 + 2 * half], b0, b1);
          mma16(O[0][0], O[0][1], O[0][2], O[0][3], a0, a1, a2, a3, b0, b1);
          cvt8(vf[gq][1 + 2 * half], b0, b1);
          mma16(O[1][0], O[1][1], O[1][2], O[1][3], a0, a1, a2, a3, b0, b1);
        }
    }
    st = (st + 1 == NSTAGE) ? 0 : st + 1;
  }

  // Start the remote output publication before the independent denominator
  // reduction so its DSM store latency is hidden by the remaining CTA work.
  if constexpr (CLU > 1) {
    // A peer's merge mbarrier only has to be visible before the first remote
    // store, so the cluster rendezvous waits here rather than on the launch
    // path, where every CTA stalled until its last peer had started.
    cluster_wait_acquire();
    constexpr int NSLOT = Geo<CLU>::NSLOT;
    // fp32 partial numerators: rounding them to bf16 here can cancel the
    // entire residual of adversarially opposing pages in the merge.
    float* oh = (float*)Mg;
    cg::cluster_group clu = cg::this_cluster();
    const int ha = g, hb = g + 8;
    float* oa = (float*)clu.map_shared_rank(oh, ha % CLU);
    float* ob = (float*)clu.map_shared_rank(oh, hb % CLU);
    const int sa = (ha / CLU) * CLU + sp;
    const int sb = (hb / CLU) * CLU + sp;
    uint64_t* bara = clu.map_shared_rank(merge_bar, ha % CLU);
    uint64_t* barb = clu.map_shared_rank(merge_bar, hb % CLU);
    if (!HALFOUT || warp < 4) {
#pragma unroll
      for (int nd = 0; nd < 2; ++nd) {
        int d = 16 * owarp + 8 * nd + 2 * tig;
        union Pack64 {
          float2 f;
          uint64_t u;
        } va, vb;
        va.f = make_float2(O[nd][0], O[nd][1]);
        vb.f = make_float2(O[nd][2], O[nd][3]);
        st_async_u64(sptr(oa + sa * DH + d), va.u, sptr(bara));
        st_async_u64(sptr(ob + sb * DH + d), vb.u, sptr(barb));
      }
    }
  }

  // The denominator is linear within each head, so reduce it once after all
  // pages instead of paying four cross-lane shuffles on every page.
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  if (tig == 0) {
    scrl[g * 8 + warp] = l0;
    scrl[(g + 8) * 8 + warp] = l1;
  }
  __syncthreads();

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
      *(__nv_bfloat162*)(outp + (size_t)(b * HQ + g) * DH + d) =
          __floats2bfloat162_rn(O[nd][0] * i0f, O[nd][1] * i0f);
      *(__nv_bfloat162*)(outp + (size_t)(b * HQ + g + 8) * DH + d) =
          __floats2bfloat162_rn(O[nd][2] * i1f, O[nd][3] * i1f);
    }
    return;
  }

  {
    constexpr int NSLOT = Geo<CLU>::NSLOT;
    float* oh = (float*)Mg;
    float* ml = oh + NSLOT * DH;
    cg::cluster_group clu = cg::this_cluster();
    const int ha = g, hb = g + 8;
    const int sa = (ha / CLU) * CLU + sp;
    const int sb = (hb / CLU) * CLU + sp;
    uint64_t* bara = clu.map_shared_rank(merge_bar, ha % CLU);
    uint64_t* barb = clu.map_shared_rank(merge_bar, hb % CLU);
    if (warp == 0 && tig == 0) {
      float4 la = *(const float4*)(scrl + g * 8);
      float4 lb = *(const float4*)(scrl + g * 8 + 4);
      float4 lc = *(const float4*)(scrl + (g + 8) * 8);
      float4 ld = *(const float4*)(scrl + (g + 8) * 8 + 4);
      float L0 = (la.x + la.y) + (la.z + la.w) + (lb.x + lb.y) + (lb.z + lb.w);
      float L1 = (lc.x + lc.y) + (lc.z + lc.w) + (ld.x + ld.y) + (ld.z + ld.w);
      float* mla = (float*)clu.map_shared_rank(ml, ha % CLU);
      float* mlb = (float*)clu.map_shared_rank(ml, hb % CLU);
      union Pack64 {
        float2 f;
        uint64_t u;
      } pa, pb;
      pa.f = make_float2(m0, L0);
      pb.f = make_float2(m1, L1);
      st_async_u64(sptr(mla + 2 * sa), pa.u, sptr(bara));
      st_async_u64(sptr(mlb + 2 * sb), pb.u, sptr(barb));
    }
    const int nh = (HQ - sp + CLU - 1) / CLU;
    const int dd = HALFOUT ? dgroup * 64 + (tid & 31) * 2 : (tid & 63) * 2;
    const int htid = HALFOUT ? (tid >> 5) : (tid >> 6);
    const int hstep = HALFOUT ? 8 : 4;
    if (htid >= nh) return;
    mbar_wait_local(merge_bar);
    for (int hi = htid; hi < nh; hi += hstep) {
      float wv[CLU];
      float mm = -1e30f;
#pragma unroll
      for (int c = 0; c < CLU; ++c) {
        wv[c] = ml[2 * (hi * CLU + c)];
        mm = fmaxf(mm, wv[c]);
      }
      float den = 0.f;
#pragma unroll
      for (int c = 0; c < CLU; ++c) {
        wv[c] = exp2f(wv[c] - mm);
        den += ml[2 * (hi * CLU + c) + 1] * wv[c];
      }
      // empty selection -> zero output (packed-backend convention)
      float inv = den > 0.f ? 1.0f / den : 0.f;
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

}  // namespace dkf16short
