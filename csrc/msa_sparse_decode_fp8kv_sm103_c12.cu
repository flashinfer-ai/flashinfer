// MSA block-sparse GQA decode (q_len 1, 16 heads, 1 KV head, D=128, fp8 paged KV).
// Single fused kernel: gather pages -> cp.async into smem -> fp8 tensor-core QK^T ->
// online softmax -> tensor-core PV -> optional in-kernel split-K merge.
// C12 keeps Q in its original bf16 and P in f16; fp8 K/V convert in registers.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <algorithm>
#include <cstdint>

namespace ffi = tvm::ffi;
namespace cg = cooperative_groups;

#define HQ 16
#define DH 128
#define BLKT 128
#define TOPKC 16

#define KSTRIDE 128
#define VSTRIDE 128
#define V2STRIDE 272
#define PSTRIDE 144
#define KSTAGE (BLKT * KSTRIDE)
#define VSTAGE (BLKT * VSTRIDE)
#define V2BYTES (64 * V2STRIDE)
#define PBYTES (HQ * PSTRIDE)
#define QSTRIDE 144
#define QSBYTES (HQ * QSTRIDE)
#define SSTRIDE 12
#define LBASE 256
#define SCRBYTES 5120
#define SMEM_N(N) ((N) * (KSTAGE + VSTAGE) + V2BYTES + PBYTES + QSBYTES + SCRBYTES)
// The resource fallback has three ranks and bf16 Q, so it does not use
// the fp8 scale table or page-table scratch beyond the row reductions.
#define COMPACT_FALLBACK(N, C, H, W) ((W) && !(H) && (C) == 3 && (N) == 1)
#define FALLBACK_V2 ((((HQ + 2) / 3 * 3 * (DH * 2 + 12)) + 127) / 128 * 128)
#define SMEM_VARIANT(N, C, H, W)                                                          \
  (COMPACT_FALLBACK(N, C, H, W) ? KSTAGE + VSTAGE + FALLBACK_V2 + HQ * (272 + 272) + 2048 \
                                : SMEM_N(N))
static_assert((LBASE + (HQ - 1) * SSTRIDE + 8) * 4 + 64 <= 2048);

#define QSCALE 0.08838834764831845f
#define LOG2EF 1.4426950408889634f
#define PEXP 8.0f  // P is stored as exp2(s - m + PEXP) so e4m3 keeps range

__device__ __forceinline__ uint32_t sptr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ int kvsw(int row, int chunk) {
  return row * 128 + ((chunk ^ (row & 7)) << 4);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
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
__device__ __forceinline__ void ldm2(uint32_t& a, uint32_t& b, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a), "=r"(b)
               : "r"(addr));
}
__device__ __forceinline__ void ldm4t(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                      uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(addr));
}
__device__ __forceinline__ void ldm2t_b8(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                         uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(addr));
}
__device__ __forceinline__ void mma_fp8(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                        uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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
// Logical contraction index -> slot inside a 16-element group.  ldmatrix hands
// each lane four consecutive bytes while the mma fragment wants {2j,2j+1} and
// {2j+8,2j+9}; applying this permutation to the operand we build ourselves (Q,
// P) makes both sides agree without touching the gathered K/V bytes.
__device__ __forceinline__ int perm16(int t) {
  return (t & ~15) + 2 * ((t & 15) >> 2) + (t & 1) + ((t & 2) ? 8 : 0);
}

__device__ __forceinline__ uint16_t fp8x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
  return ((uint32_t)fp8x2(c, d) << 16) | (uint32_t)fp8x2(a, b);
}
__device__ __forceinline__ float bf2f(uint32_t bits16) { return __int_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  union PackedBF16x2 {
    __nv_bfloat162 bf;
    uint32_t u;
  } out;
  out.bf = __floats2bfloat162_rn(lo, hi);
  return out.u;
}

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

// 4 bf16 packed in uint2 -> floats
__device__ __forceinline__ void unpack4(uint2 w, float& a, float& b, float& c, float& d) {
  a = __int_as_float(w.x << 16);
  b = __int_as_float(w.x & 0xFFFF0000u);
  c = __int_as_float(w.y << 16);
  d = __int_as_float(w.y & 0xFFFF0000u);
}

template <int NSTAGE, int CLU, bool HALFOUT = false, bool QUARTER = false, bool HALFPAGE = false,
          int CTA_THREADS = 256, bool BF16_Q_F16_P = false>
__global__ void __cluster_dims__(CLU, 1, 1)
    __launch_bounds__(CTA_THREADS,
                      (CTA_THREADS == 128 ? 1 : ((NSTAGE == 2 || BF16_Q_F16_P) ? 2 : 1)))
        msa_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc,
                   const uint8_t* __restrict__ vc, const int* __restrict__ topk,
                   const int* __restrict__ ptab, const int* __restrict__ seqlens,
                   __nv_bfloat16* __restrict__ outp, int pages, int bpc, int nblk) {
  static_assert(CTA_THREADS == 128 || CTA_THREADS == 256);
  constexpr int QSTEP = BF16_Q_F16_P ? 272 : QSTRIDE;
  constexpr int PSTEP = BF16_Q_F16_P ? (HALFPAGE ? 80 : 272) : PSTRIDE;
  constexpr bool ASYNC_MERGE = (CLU == 12 && HALFOUT && HALFPAGE);
  const int b = QUARTER ? (blockIdx.y >> 2) : (HALFOUT ? (blockIdx.y >> 1) : blockIdx.y);
  const int dgroup = QUARTER ? (blockIdx.y & 3) : (HALFOUT ? (blockIdx.y & 1) : 0);
  const int sp = blockIdx.x;
  const int psp = HALFPAGE ? (sp >> 2) : sp;
  const int hoff = HALFPAGE ? 32 * (sp & 3) : 0;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int tig = lane & 3;
  extern __shared__ __align__(16) uint8_t sm[];
  uint64_t* merge_bar =
      reinterpret_cast<uint64_t*>(sm + SMEM_VARIANT(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P) - 64);

  // The few-page routes have one selected rank per CTA.  Start both
  // independent metadata requests before barrier setup and Q so the later
  // shuffle consumes them only after useful work has covered their latency.
  int early_row = 0, early_blkr = 0;
  if constexpr (NSTAGE == 1 && (CLU == 3 || CLU == 4 || HALFPAGE)) {
    if (lane < pages)
      asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(early_row) : "l"(ptab + b * pages + lane));
    asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(early_blkr) : "l"(topk + b * TOPKC + psp));
  }
  if constexpr (ASYNC_MERGE) {
    // Each owner CTA receives one 128-byte BF16 output half plus one
    // 8-byte (m,l) pair from every cluster rank for each head it owns.
    // With HQ=16 and CLU=12, ranks 0..3 own two heads while the rest own
    // one, so a uniform transaction count over-completes four barriers.
    const uint32_t owned_heads = 1u + (sp < 4);
    if (tid == 0) mbar_init_expect(merge_bar, owned_heads * CLU * (128 + 8));
  }
  if constexpr (CLU > 1) cluster_arrive_relaxed();

  // Force the independent q/sequence fetches to start before the page-table
  // pointer chase.  The 128-thread C12 route assigns two query heads to each
  // half warp; the generic 256-thread route retains one.
  constexpr int QREPEAT = 256 / CTA_THREADS;
  uint4 qw[QREPEAT];
#pragma unroll
  for (int qh = 0; qh < QREPEAT; ++qh) {
    const int h = (tid >> 4) + qh * (CTA_THREADS >> 4);
    const __nv_bfloat16* qptr = q + (size_t)b * (HQ * DH) + h * DH + (tid & 15) * 8;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(qw[qh].x), "=r"(qw[qh].y), "=r"(qw[qh].z), "=r"(qw[qh].w)
                 : "l"(qptr));
  }
  int seqlen;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(seqlen) : "l"(seqlens + b));

  uint8_t* Kb = sm;
  uint8_t* Vb = sm + NSTAGE * KSTAGE;
  uint8_t* V2 = Vb + NSTAGE * VSTAGE;
  // C12's V2 region only holds the merge mailbox. Reuse its unused tail
  // for wider Q/P, keeping the original per-CTA shared-memory footprint.
  constexpr int V2ROOM =
      COMPACT_FALLBACK(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P)
          ? FALLBACK_V2
          : V2BYTES - (BF16_Q_F16_P ? HQ * (QSTEP + PSTEP - QSTRIDE - PSTRIDE) : 0);
  static_assert(!BF16_Q_F16_P || V2ROOM >= ((HQ + CLU - 1) / CLU) * CLU * (DH * 2 + 12));
  uint8_t* Pb = V2 + V2ROOM;
  uint8_t* Qs = Pb + HQ * PSTEP;
  float* scr = (float*)(Qs + HQ * QSTEP);

  // ---- block-id / page-id prefetch: 2 dependent loads once for ALL blocks of this CTA ----
  int blkr, pgr, nb;
  if constexpr (NSTAGE == 1 && (CLU == 3 || CLU == 4 || HALFPAGE)) {
    // Few-page dispatches give one rank per selected block.  The top-k entry
    // and the (short) page-table row are independent loads, so both issue in
    // the same round trip and the physical page is picked in registers:
    // honouring an arbitrary ascending top-k costs one shuffle, not a
    // second dependent global load.
    int row = early_row;
    blkr = early_blkr;  // warp-uniform: broadcast, no shuffle
    nb = (blkr >= 0);
    // An empty rank (-1) lands on lane 31, whose row is 0 and whose pages are
    // never issued, so the shuffle index needs no select.
    pgr = __shfl_sync(0xffffffffu, row, blkr & 31);
  } else {
    const int i0 = sp * bpc;
    const int i1 = min(i0 + bpc, nblk);
    const int lim = i1 - i0;
    const int* tkp = topk + b * TOPKC;
    int jj = i0 + (lane & 15);
    if (jj > TOPKC - 1) jj = TOPKC - 1;
    blkr = tkp[jj];
    if (pages <= 256) {
      int* ptsm = (int*)(scr + 768);
      for (int p = tid; p < pages; p += 256) ptsm[p] = ptab[b * pages + p];
      __syncthreads();
      pgr = (blkr >= 0) ? ptsm[blkr] : 0;
    } else {
      pgr = ptab[b * pages + (blkr >= 0 ? blkr : 0)];
    }
    unsigned vmask = __ballot_sync(0xffffffffu, (lane < 16) && (lane < lim) && (blkr >= 0));
    nb = __popc(vmask & 0xffffu);
  }

  constexpr int NT2 = HALFPAGE ? 1 : 2;
  constexpr int KSN = HALFPAGE ? 1 : 4;
  const int tbase0 = (HALFPAGE ? 8 : 16) * warp;
#define ISSUE(J, ST)                                                                         \
  do {                                                                                       \
    if ((J) < nb) {                                                                          \
      int pg_ = __shfl_sync(0xffffffffu, pgr, (J));                                          \
      const uint8_t* kg_ = kc + (size_t)pg_ * (BLKT * DH) + hoff * DH;                       \
      const uint8_t* vg_ = vc + (size_t)pg_ * (BLKT * DH) + hoff * DH;                       \
      uint8_t* kd_ = Kb + (ST) * KSTAGE;                                                     \
      uint8_t* vd_ = Vb + (ST) * VSTAGE;                                                     \
      _Pragma("unroll") for (int t2 = 0; t2 < (HALFPAGE ? 256 : 1024) / CTA_THREADS; ++t2) { \
        int u_ = tid + CTA_THREADS * t2;                                                     \
        int tok_ = u_ >> 3, c_ = u_ & 7;                                                     \
        cp16(sptr(kd_ + kvsw(tok_, c_)), kg_ + (size_t)u_ * 16);                             \
      }                                                                                      \
      cpcommit();                                                                            \
      if constexpr (HALFPAGE && QUARTER) {                                                   \
        if (tid < 64) {                                                                      \
          int tok_ = tid >> 1, c_ = 2 * dgroup + (tid & 1);                                  \
          cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + ((size_t)tok_ * 8 + c_) * 16);              \
        }                                                                                    \
      } else if constexpr (HALFPAGE && HALFOUT) {                                            \
        if (tid < 128) {                                                                     \
          int tok_ = tid >> 2, c_ = 4 * dgroup + (tid & 3);                                  \
          cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + ((size_t)tok_ * 8 + c_) * 16);              \
        }                                                                                    \
      } else if constexpr (HALFPAGE) {                                                       \
        _Pragma("unroll") for (int t2 = 0; t2 < 256 / CTA_THREADS; ++t2) {                   \
          int u_ = tid + CTA_THREADS * t2;                                                   \
          int tok_ = u_ >> 3, c_ = u_ & 7;                                                   \
          cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + (size_t)u_ * 16);                           \
        }                                                                                    \
      } else if constexpr (QUARTER) {                                                        \
        int tok_ = tid >> 1, c_ = 2 * dgroup + (tid & 1);                                    \
        cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + ((size_t)tok_ * 8 + c_) * 16);                \
      } else if constexpr (HALFOUT) {                                                        \
        _Pragma("unroll") for (int t2 = 0; t2 < 512 / CTA_THREADS; ++t2) {                   \
          int u_ = tid + CTA_THREADS * t2;                                                   \
          int tok_ = u_ >> 2, c_ = 4 * dgroup + (u_ & 3);                                    \
          cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + ((size_t)tok_ * 8 + c_) * 16);              \
        }                                                                                    \
      } else {                                                                               \
        _Pragma("unroll") for (int t2 = 0; t2 < 1024 / CTA_THREADS; ++t2) {                  \
          int u_ = tid + CTA_THREADS * t2;                                                   \
          int tok_ = u_ >> 3, c_ = u_ & 7;                                                   \
          cp16(sptr(vd_ + kvsw(tok_, c_)), vg_ + (size_t)u_ * 16);                           \
        }                                                                                    \
      }                                                                                      \
      cpcommit();                                                                            \
    } else {                                                                                 \
      cpcommit();                                                                            \
      cpcommit();                                                                            \
    }                                                                                        \
  } while (0)

#pragma unroll
  for (int s_ = 0; s_ < NSTAGE; ++s_) ISSUE(s_, s_);

  // C12 preserves every bf16 query channel, including heads with a wide
  // exponent range. The legacy fp8 variants retain their original scaling.
  {
    const int dg = tid & 15;
#pragma unroll
    for (int qh = 0; qh < QREPEAT; ++qh) {
      const int h = (tid >> 4) + qh * (CTA_THREADS >> 4);
      uint32_t raw[4] = {qw[qh].x, qw[qh].y, qw[qh].z, qw[qh].w};
      if constexpr (BF16_Q_F16_P) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          *(uint32_t*)(Qs + h * QSTEP + 2 * perm16(dg * 8 + 2 * i)) = raw[i];
      } else {
        float v[8];
        float am = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw[i]));
          v[2 * i] = f.x;
          v[2 * i + 1] = f.y;
          am = fmaxf(am, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
        float scq = (am > 0.f) ? (448.0f / am) : 1.0f;
        *(uint2*)(Qs + h * QSTRIDE + dg * 8) =
            make_uint2(fp8x4(v[0] * scq, v[1] * scq, v[2] * scq, v[3] * scq),
                       fp8x4(v[4] * scq, v[5] * scq, v[6] * scq, v[7] * scq));
        if (dg == 0) scr[1024 + h] = QSCALE * LOG2EF * am * (1.0f / 448.0f);
      }
    }
    // One clear per thread, not one per staged head.
    if (nb == 0 && tig == 0) {
      scr[LBASE + g * SSTRIDE + warp] = 0.f;
      scr[LBASE + (g + 8) * SSTRIDE + warp] = 0.f;
    }
  }
  __syncthreads();
  uint32_t qa[BF16_Q_F16_P ? 8 : 4][4];
  if (!HALFPAGE || warp < 4) {
    const int qrow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int qhalf = (lane >> 4) * 16;
#pragma unroll
    for (int ks = 0; ks < (BF16_Q_F16_P ? 8 : 4); ++ks)
      ldm4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], sptr(Qs + qrow * QSTEP + 32 * ks + qhalf));
  }
  float fac0 = QSCALE * LOG2EF, fac1 = QSCALE * LOG2EF;
  if constexpr (!BF16_Q_F16_P) {
    if (!HALFPAGE || warp < 4) {
      fac0 = scr[1024 + g];
      fac1 = scr[1024 + g + 8];
    }
  }

  // ---------------- accumulators ----------------
  constexpr int OWARPS = QUARTER ? 2 : (HALFOUT ? 4 : 8);
  float O[2][4];
  if ((!HALFOUT && !QUARTER) || warp < OWARPS) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) O[i][j] = 0.f;
  }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  const int prow = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int phalf = (lane >> 4) * 16;
  const int owarp = (HALFOUT || QUARTER) ? (warp + dgroup * OWARPS) : warp;
  const bool needs_mask = seqlen != pages * BLKT;
  for (int j = 0; j < nb; ++j) {
    const int st = j & (NSTAGE - 1);
    int cb = 0;
    if (needs_mask) cb = __shfl_sync(0xffffffffu, blkr, j);
    // K and V use distinct copy groups: begin QK as soon as K is visible,
    // leaving the corresponding V page in flight behind score computation.
    cpwait<2 * NSTAGE - 1>();
    __syncthreads();

    uint8_t* kd = Kb + st * KSTAGE;
    uint8_t* vd = Vb + st * VSTAGE;

    // ---- QK^T ----
    float S[2][4];
    if (!HALFPAGE || warp < 4) {
#pragma unroll
      for (int t2 = 0; t2 < NT2; ++t2)
#pragma unroll
        for (int x = 0; x < 4; ++x) S[t2][x] = 0.f;
      const int krow = tbase0 + (lane & 7) + 8 * ((lane >> 3) & 1);
      const int khalf = (lane >> 4) & 1;
      if constexpr (!HALFPAGE) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          uint32_t k0, k1, k2, k3;
          ldm4(k0, k1, k2, k3, sptr(kd + kvsw(krow, 2 * ks + khalf)));
          if constexpr (BF16_Q_F16_P) {
            uint32_t b0, b1;
            cvt8bf(k0, b0, b1);
            mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], qa[2 * ks][0], qa[2 * ks][1],
                     qa[2 * ks][2], qa[2 * ks][3], b0, b1);
            cvt8bf(k1, b0, b1);
            mma_bf16(S[1][0], S[1][1], S[1][2], S[1][3], qa[2 * ks][0], qa[2 * ks][1],
                     qa[2 * ks][2], qa[2 * ks][3], b0, b1);
            cvt8bf(k2, b0, b1);
            mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], qa[2 * ks + 1][0], qa[2 * ks + 1][1],
                     qa[2 * ks + 1][2], qa[2 * ks + 1][3], b0, b1);
            cvt8bf(k3, b0, b1);
            mma_bf16(S[1][0], S[1][1], S[1][2], S[1][3], qa[2 * ks + 1][0], qa[2 * ks + 1][1],
                     qa[2 * ks + 1][2], qa[2 * ks + 1][3], b0, b1);
          } else {
            mma_fp8(S[0][0], S[0][1], S[0][2], S[0][3], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3],
                    k0, k2);
            mma_fp8(S[1][0], S[1][1], S[1][2], S[1][3], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3],
                    k1, k3);
          }
        }
      } else {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          uint32_t k0, k2;
          // x4 used to fetch two 8-token matrices that this quarter-page
          // route never consumed.  For x2, lanes 0..7 name the even chunk
          // and lanes 8..15 name the odd chunk for the same eight rows.
          const int x2row = tbase0 + (lane & 7);
          const int x2half = (lane >> 3) & 1;
          ldm2(k0, k2, sptr(kd + kvsw(x2row, 2 * ks + x2half)));
          if constexpr (BF16_Q_F16_P) {
            uint32_t b0, b1;
            cvt8bf(k0, b0, b1);
            mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], qa[2 * ks][0], qa[2 * ks][1],
                     qa[2 * ks][2], qa[2 * ks][3], b0, b1);
            cvt8bf(k2, b0, b1);
            mma_bf16(S[0][0], S[0][1], S[0][2], S[0][3], qa[2 * ks + 1][0], qa[2 * ks + 1][1],
                     qa[2 * ks + 1][2], qa[2 * ks + 1][3], b0, b1);
          } else {
            mma_fp8(S[0][0], S[0][1], S[0][2], S[0][3], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3],
                    k0, k2);
          }
        }
      }
    }

    // ---- scale + mask + row max ----
    int cnt = BLKT;
    if (needs_mask) cnt = min(seqlen - cb * BLKT, BLKT);
    float rmax0 = -INFINITY, rmax1 = -INFINITY;
    if (!HALFPAGE || warp < 4) {
#pragma unroll
      for (int t2 = 0; t2 < NT2; ++t2) {
        int t0 = hoff + tbase0 + 8 * t2 + 2 * tig;
        float v0 = S[t2][0] * fac0, v1 = S[t2][1] * fac0;
        float v2 = S[t2][2] * fac1, v3 = S[t2][3] * fac1;
        if constexpr (HALFPAGE) {
          if (warp >= 4) v0 = v1 = v2 = v3 = -INFINITY;
        }
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
        scr[g * SSTRIDE + warp] = rmax0;
        scr[(g + 8) * SSTRIDE + warp] = rmax1;
      }
    }
    cpwait<2 * NSTAGE - 2>();
    __syncthreads();

    if (!HALFPAGE || warp < 4) {
      float bm0 = scr[g * SSTRIDE + tig];
      float bm1 = scr[(g + 8) * SSTRIDE + tig];
      if constexpr (!HALFPAGE) {
        bm0 = fmaxf(bm0, scr[g * SSTRIDE + tig + 4]);
        bm1 = fmaxf(bm1, scr[(g + 8) * SSTRIDE + tig + 4]);
      }
      bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 1));
      bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 2));
      bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 1));
      bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 2));
      float nm0, nm1;
      if constexpr (HALFPAGE) {
        // The C12 B1-short route assigns exactly one 32-token quarter-page to
        // each CTA.  Initialize online softmax from that sole tile instead of
        // exponentiating -infinity and rescaling zero state.  Preserve the
        // finite sentinel when a quarter of a partial page is fully masked.
        nm0 = fmaxf(m0, bm0);
        nm1 = fmaxf(m1, bm1);
        m0 = nm0;
        m1 = nm1;
      } else {
        nm0 = fmaxf(m0, bm0);
        nm1 = fmaxf(m1, bm1);
        float al0 = exp2f(m0 - nm0), al1 = exp2f(m1 - nm1);
        m0 = nm0;
        m1 = nm1;
        if ((!HALFOUT && !QUARTER) || warp < OWARPS) {
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

      float sum0 = 0.f, sum1 = 0.f;
      if (!HALFPAGE || warp < 4) {
#pragma unroll
        for (int t2 = 0; t2 < NT2; ++t2) {
          float p0 = exp2f(S[t2][0] - nm0 + PEXP);
          float p1 = exp2f(S[t2][1] - nm0 + PEXP);
          float p2 = exp2f(S[t2][2] - nm1 + PEXP);
          float p3 = exp2f(S[t2][3] - nm1 + PEXP);
          sum0 += p0 + p1;
          sum1 += p2 + p3;
          int tok = tbase0 + 8 * t2 + 2 * tig;
          if constexpr (BF16_Q_F16_P) {
            *(uint32_t*)(Pb + g * PSTEP + 2 * perm16(tok)) = f2h2(p0, p1);
            *(uint32_t*)(Pb + (g + 8) * PSTEP + 2 * perm16(tok)) = f2h2(p2, p3);
          } else {
            *(uint16_t*)(Pb + g * PSTRIDE + tok) = fp8x2(p0, p1);
            *(uint16_t*)(Pb + (g + 8) * PSTRIDE + tok) = fp8x2(p2, p3);
          }
        }
      }
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
      l0 += sum0;
      l1 += sum1;
      if (j + 1 == nb && tig == 0) {
        scr[LBASE + g * SSTRIDE + warp] = l0;
        scr[LBASE + (g + 8) * SSTRIDE + warp] = l1;
      }
    }
    __syncthreads();

    // ---- PV ----
    if ((!HALFOUT && !QUARTER) || warp < OWARPS) {
#pragma unroll
      for (int ks = 0; ks < KSN; ++ks) {
        uint32_t vf[4];
        ldm2t_b8(vf[0], vf[1], vf[2], vf[3], sptr(vd + kvsw(32 * ks + lane, owarp)));
        if constexpr (BF16_Q_F16_P) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            uint32_t a0, a1, a2, a3, b0, b1;
            ldm4(a0, a1, a2, a3, sptr(Pb + prow * PSTEP + (2 * ks + half) * 32 + phalf));
            cvt8(vf[2 * half], b0, b1);
            mma16(O[0][0], O[0][1], O[0][2], O[0][3], a0, a1, a2, a3, b0, b1);
            cvt8(vf[2 * half + 1], b0, b1);
            mma16(O[1][0], O[1][1], O[1][2], O[1][3], a0, a1, a2, a3, b0, b1);
          }
        } else {
          uint32_t p0_, p1_, p2_, p3_;
          ldm4(p0_, p1_, p2_, p3_, sptr(Pb + prow * PSTRIDE + 32 * ks + phalf));
          mma_fp8(O[0][0], O[0][1], O[0][2], O[0][3], p0_, p1_, p2_, p3_, vf[0], vf[2]);
          mma_fp8(O[1][0], O[1][1], O[1][2], O[1][3], p0_, p1_, p2_, p3_, vf[1], vf[3]);
        }
      }
    }
    if constexpr (NSTAGE > 1) {
      __syncthreads();
      ISSUE(j + NSTAGE, st);
    }
  }

  // ---------------- epilogue ----------------
  if constexpr (CLU == 1) {
    float L0 = scr[LBASE + g * SSTRIDE + tig];
    float L1 = scr[LBASE + (g + 8) * SSTRIDE + tig];
    if constexpr (!HALFPAGE) {
      L0 += scr[LBASE + g * SSTRIDE + tig + 4];
      L1 += scr[LBASE + (g + 8) * SSTRIDE + tig + 4];
    }
    L0 += __shfl_xor_sync(0xffffffffu, L0, 1);
    L0 += __shfl_xor_sync(0xffffffffu, L0, 2);
    L1 += __shfl_xor_sync(0xffffffffu, L1, 1);
    L1 += __shfl_xor_sync(0xffffffffu, L1, 2);
    float i0f = 1.0f / L0, i1f = 1.0f / L1;
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

  // ---- Push bf16 partials into each owner CTA's local mailbox.  One cluster
  // barrier then makes the combine entirely local.  Issue the O stores before
  // the independent row-sum reduction so their DSM latency is hidden by it.
  cluster_wait_acquire();
  {
    constexpr int NSLOT = ((HQ + CLU - 1) / CLU) * CLU;
    __nv_bfloat16* oh = (__nv_bfloat16*)V2;  // [NSLOT][128]
    float* ml = (float*)(oh + NSLOT * DH);   // [NSLOT][2] = (m, l)
    float* wt = ml + 2 * NSLOT;              // [NSLOT] normalized weights
    cg::cluster_group clu = cg::this_cluster();
    const int ha = g, hb = g + 8;
    __nv_bfloat16* oa = (__nv_bfloat16*)clu.map_shared_rank(oh, ha % CLU);
    __nv_bfloat16* ob = (__nv_bfloat16*)clu.map_shared_rank(oh, hb % CLU);
    float* mla = (float*)clu.map_shared_rank(ml, ha % CLU);
    float* mlb = (float*)clu.map_shared_rank(ml, hb % CLU);
    uint64_t* bara = clu.map_shared_rank(merge_bar, ha % CLU);
    uint64_t* barb = clu.map_shared_rank(merge_bar, hb % CLU);
    const int sa = (ha / CLU) * CLU + sp;
    const int sb = (hb / CLU) * CLU + sp;
    if ((!HALFOUT && !QUARTER) || warp < OWARPS) {
#pragma unroll
      for (int nd = 0; nd < 2; ++nd) {
        int d = 16 * owarp + 8 * nd + 2 * tig;
        if constexpr (ASYNC_MERGE) {
          st_async_u32(sptr(oa + sa * DH + d), f2bf2(O[nd][0], O[nd][1]), sptr(bara));
          st_async_u32(sptr(ob + sb * DH + d), f2bf2(O[nd][2], O[nd][3]), sptr(barb));
        } else {
          *(__nv_bfloat162*)(oa + sa * DH + d) = __floats2bfloat162_rn(O[nd][0], O[nd][1]);
          *(__nv_bfloat162*)(ob + sb * DH + d) = __floats2bfloat162_rn(O[nd][2], O[nd][3]);
        }
      }
    }
    if (warp == 0) {
      float L0 = 0.f, L1 = 0.f;
      if constexpr (ASYNC_MERGE) {
        if (tig == 0) {
          float4 a = *(const float4*)(scr + LBASE + g * SSTRIDE);
          float4 b = *(const float4*)(scr + LBASE + (g + 8) * SSTRIDE);
          L0 = (a.x + a.y) + (a.z + a.w);
          L1 = (b.x + b.y) + (b.z + b.w);
        }
      } else {
        L0 = scr[LBASE + g * SSTRIDE + tig];
        L1 = scr[LBASE + (g + 8) * SSTRIDE + tig];
        if constexpr (!HALFPAGE) {
          L0 += scr[LBASE + g * SSTRIDE + tig + 4];
          L1 += scr[LBASE + (g + 8) * SSTRIDE + tig + 4];
        }
        L0 += __shfl_xor_sync(0xffffffffu, L0, 1);
        L0 += __shfl_xor_sync(0xffffffffu, L0, 2);
        L1 += __shfl_xor_sync(0xffffffffu, L1, 1);
        L1 += __shfl_xor_sync(0xffffffffu, L1, 2);
      }
      if (tig == 0) {
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
    }
    const int nh = ASYNC_MERGE ? (1 + (sp < 4)) : ((HQ - sp + CLU - 1) / CLU);
    if constexpr (ASYNC_MERGE) {
      // Only owner warps consume this CTA's mailbox.  A per-lane acquire wait
      // removes the critical-tail CTA barrier while preserving visibility for
      // every lane that subsequently reads remote async-store data.
      if (warp < nh) mbar_wait_local(merge_bar);
    } else {
      clu.sync();
    }

    if constexpr (CLU == 12 && HALFOUT && HALFPAGE) {
      // Ranks 0..3 own two heads and the other ranks own one.  Give each
      // owned head a warp; lanes 0..11 hold its twelve merge weights while
      // all 32 lanes vectorize the 64-d output half.
      if (warp < nh) {
        const int cc = min(lane, 11);
        float2 t = *(const float2*)(ml + 2 * (warp * CLU + cc));
        float mm = -1e30f;
#pragma unroll
        for (int c = 0; c < 12; ++c) mm = fmaxf(mm, __shfl_sync(0xffffffffu, t.x, c));
        float w = exp2f(t.x - mm);
        float den = 0.f;
#pragma unroll
        for (int c = 0; c < 12; ++c) den += __shfl_sync(0xffffffffu, t.y * w, c);
        w = (den > 0.f) ? (w / den) : 0.f;
        const int d = dgroup * 64 + 2 * lane;
        // Three vector loads replace twelve lane broadcasts of the same
        // twelve weights on the critical tail.
        if (lane < 12) wt[warp * CLU + lane] = w;
        __syncwarp();
        float wv[12];
        *(float4*)(wv + 0) = *(const float4*)(wt + warp * CLU + 0);
        *(float4*)(wv + 4) = *(const float4*)(wt + warp * CLU + 4);
        *(float4*)(wv + 8) = *(const float4*)(wt + warp * CLU + 8);
        float2 acc = make_float2(0.f, 0.f);
#pragma unroll
        for (int c = 0; c < 12; ++c) {
          float wc = wv[c];
          __nv_bfloat162 x = *(__nv_bfloat162*)(oh + (warp * CLU + c) * DH + d);
          float2 xf = __bfloat1622float2(x);
          acc.x += xf.x * wc;
          acc.y += xf.y * wc;
        }
        *(__nv_bfloat162*)(outp + (size_t)(b * HQ + sp + warp * CLU) * DH + d) =
            __floats2bfloat162_rn(acc.x, acc.y);
      }
    } else {
      float lane_w = 0.f;
      if (warp == 0) {
        const int hi = lane / CLU;
        const int cc = lane - hi * CLU;
        float mv = -1e30f, lv = 0.f;
        if (hi < nh) {
          float2 t = *(const float2*)(ml + 2 * (hi * CLU + cc));
          mv = t.x;
          lv = t.y;
        }
        float mm;
        if constexpr ((CLU & (CLU - 1)) == 0) {
          mm = mv;
#pragma unroll
          for (int off = CLU >> 1; off > 0; off >>= 1)
            mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, off, CLU));
        } else {
          const int hbase = (hi < nh) ? hi * CLU : 0;
          mm = -1e30f;
#pragma unroll
          for (int c = 0; c < CLU; ++c) mm = fmaxf(mm, __shfl_sync(0xffffffffu, mv, hbase + c));
        }
        float w = exp2f(mv - mm);
        float den;
        if constexpr ((CLU & (CLU - 1)) == 0) {
          den = lv * w;
#pragma unroll
          for (int off = CLU >> 1; off > 0; off >>= 1)
            den += __shfl_xor_sync(0xffffffffu, den, off, CLU);
        } else {
          const int hbase = (hi < nh) ? hi * CLU : 0;
          den = 0.f;
#pragma unroll
          for (int c = 0; c < CLU; ++c) den += __shfl_sync(0xffffffffu, lv * w, hbase + c);
        }
        lane_w = (hi < nh && den > 0.f) ? (w / den) : 0.f;
        if constexpr (!(CLU == 16 && (QUARTER || HALFOUT))) {
          if (hi < nh) wt[lane] = lane_w;
        }
      }
      if constexpr (CLU == 16 && (QUARTER || HALFOUT)) {
        if (warp == 0) {
          constexpr unsigned MASK = QUARTER ? 0x0000ffffu : 0xffffffffu;
          constexpr int WIDTH = QUARTER ? 16 : 32;
          const bool active = QUARTER ? (lane < 16) : true;
          if (active) {
            const int d = QUARTER ? (dgroup * 32 + 2 * lane) : (dgroup * 64 + 2 * lane);
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              float w = __shfl_sync(MASK, lane_w, c, WIDTH);
              __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(oh + c * DH + d);
              float2 xf = __bfloat1622float2(x);
              acc.x += xf.x * w;
              acc.y += xf.y * w;
            }
            *reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * HQ + sp) * DH + d) =
                __floats2bfloat162_rn(acc.x, acc.y);
          }
        }
      } else {
        __syncthreads();
        if constexpr (CLU <= 4) {
          const int d = QUARTER ? (dgroup * 32 + 2 * (tid & 15))
                                : (HALFOUT ? (dgroup * 64 + 2 * (tid & 31)) : 2 * (tid & 63));
          const int htid = QUARTER ? (tid >> 4) : (HALFOUT ? (tid >> 5) : (tid >> 6));
          const int hstep = QUARTER ? 16 : (HALFOUT ? 8 : 4);
          for (int hi = htid; hi < nh; hi += hstep) {
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for (int c = 0; c < CLU; ++c) {
              float w = wt[hi * CLU + c];
              __nv_bfloat162 x = *(__nv_bfloat162*)(oh + (hi * CLU + c) * DH + d);
              float2 xf = __bfloat1622float2(x);
              acc.x += xf.x * w;
              acc.y += xf.y * w;
            }
            *(__nv_bfloat162*)(outp + (size_t)b * (HQ * DH) + (sp + hi * CLU) * DH + d) =
                __floats2bfloat162_rn(acc.x, acc.y);
          }
        } else {
          const int d = QUARTER ? (dgroup * 32 + (tid & 31))
                                : (HALFOUT ? (dgroup * 64 + (tid & 63)) : (tid & 127));
          const int htid = QUARTER ? (tid >> 5) : (HALFOUT ? (tid >> 6) : (tid >> 7));
          const int hstep = QUARTER ? 8 : (HALFOUT ? 4 : 2);
          for (int hi = htid; hi < nh; hi += hstep) {
            float acc = 0.f;
#pragma unroll
            for (int c = 0; c < CLU; ++c)
              acc += __bfloat162float(oh[(hi * CLU + c) * DH + d]) * wt[hi * CLU + c];
            outp[(size_t)b * (HQ * DH) + (sp + hi * CLU) * DH + d] = __float2bfloat16(acc);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
static bool g_attr_set[64] = {};  // per-device one-time attribute opt-in

template <int NSTAGE, int CLU, bool HALFOUT = false, bool QUARTER = false, bool HALFPAGE = false,
          int CTA_THREADS = 256, bool BF16_Q_F16_P = false>
static cudaError_t launch(int B, const void* qp, const void* kp, const void* vp, const void* tkp,
                          const void* ptp, const void* slp, void* op, int pages, int bpc, int nblk,
                          cudaStream_t stream) {
  dim3 grid(CLU, QUARTER ? B * 4 : (HALFOUT ? B * 2 : B));
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(CTA_THREADS);
  config.dynamicSmemBytes = SMEM_VARIANT(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P);
  config.stream = stream;
  cudaLaunchAttribute scheduling{};
  if constexpr (COMPACT_FALLBACK(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P)) {
    scheduling.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    scheduling.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = &scheduling;
    config.numAttrs = 1;
  }
  if constexpr (BF16_Q_F16_P) {
    cudaStreamCaptureStatus capture;
    cudaError_t status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return status;
    if (capture != cudaStreamCaptureStatusNone) {
      // Capture can defer launch validation until graph instantiation.
      // Check residency before recording a node that cannot be launched.
      int active = 0;
      status = cudaOccupancyMaxActiveClusters(
          &active, msa_kernel<NSTAGE, CLU, HALFOUT, QUARTER, HALFPAGE, CTA_THREADS, BF16_Q_F16_P>,
          &config);
      if (status != cudaSuccess) return status;
      if (active == 0) return cudaErrorInvalidClusterSize;
    }
  }
  return cudaLaunchKernelEx(
      &config, msa_kernel<NSTAGE, CLU, HALFOUT, QUARTER, HALFPAGE, CTA_THREADS, BF16_Q_F16_P>,
      (const __nv_bfloat16*)qp, (const uint8_t*)kp, (const uint8_t*)vp, (const int*)tkp,
      (const int*)ptp, (const int*)slp, (__nv_bfloat16*)op, pages, bpc, nblk);
}

void C12Run(ffi::TensorView q, ffi::TensorView kc, ffi::TensorView vc, ffi::TensorView topk,
            ffi::TensorView ptab, ffi::TensorView seqlens, ffi::TensorView out) {
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, q.device().device_id);
  const int B = (int)q.shape()[0];
  const int pages = (int)ptab.shape()[1];
  const int nblk = pages < TOPKC ? pages : TOPKC;

  int cap = 1;
  while (cap < nblk && cap < 8) cap *= 2;
  int splits = 1;
  while (splits < cap && B * splits * 2 <= 304) splits *= 2;
  int bpc = (nblk + splits - 1) / splits;

  int attr_di = 0;
  cudaGetDevice(&attr_di);
  if (!g_attr_set[attr_di & 63]) {
    cudaFuncSetAttribute(msa_kernel<4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(4));
    cudaFuncSetAttribute(msa_kernel<1, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(msa_kernel<1, 16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 16, true>, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
    cudaFuncSetAttribute(msa_kernel<1, 16, false, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 16, false, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(msa_kernel<1, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 4, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 12, true, false, true, 128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 12, true, false, true, 128>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(msa_kernel<1, 12, true, false, true, 128, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(1));
    cudaFuncSetAttribute(msa_kernel<1, 12, true, false, true, 128, true>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(msa_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(2));
    cudaFuncSetAttribute(msa_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(2));
    cudaFuncSetAttribute(msa_kernel<2, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_N(2));
    g_attr_set[attr_di & 63] = true;
  }

  const void* qp = q.data_ptr();
  const void* kp = kc.data_ptr();
  const void* vp = vc.data_ptr();
  const void* tkp = topk.data_ptr();
  const void* ptp = ptab.data_ptr();
  const void* slp = seqlens.data_ptr();
  void* op = out.data_ptr();
  if (nblk == 16 && B == 1)
    launch<1, 16, false, true>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3 && B <= 2) {
    const cudaError_t status = launch<1, 12, true, false, true, 128, true>(
        B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
    if (status == cudaErrorInvalidClusterSize) {
      // Some GB300 green contexts cannot place this non-portable cluster.
      // A rejected launch has not touched out. Use this core's existing
      // full-page/three-rank geometry, retaining bf16 Q on the same stream.
      (void)cudaGetLastError();
      const cudaError_t fallback_status = launch<1, 3, true, false, false, 256, true>(
          B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
      if (fallback_status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(fallback_status);
    } else if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(status);
    }
  } else if (nblk == 3 && B <= 16)
    launch<1, 4, true>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3 && B <= 32)
    launch<1, 4>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3)
    launch<1, 3>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (splits == 1)
    launch<4, 1>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else if (splits == 2)
    launch<2, 2>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else if (splits == 4)
    launch<2, 4>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else
    launch<2, 8>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(c12_split_run, C12Run);
