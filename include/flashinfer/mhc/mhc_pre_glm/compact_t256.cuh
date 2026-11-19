#pragma once
// GLM-5.3 hyper-connection (mHC) PRE op, fully fused single-launch CUDA kernel
// for GB300 (SM100/SM103).
//
// One cluster of 16 CTAs owns a tile of TT tokens.  CTA rank r owns the hidden
// slice h in [r*256, (r+1)*256) across all 4 residual streams, i.e. 1024 of the
// 16384 contraction elements.  The residual tile stays resident in shared
// memory so it is read from HBM exactly once: first for the projection GEMM
// (bf16 mma.sync), then again for the pre-gated layer-input reduction after the
// cluster-wide logit reduction.  The 24 projection weights for the CTA's k
// slice live in registers for the whole (persistent) tile loop.
//
// The 20-iteration sinkhorn is a ~1100-cycle serial dependency chain per token,
// so comb logits are buffered and the sinkhorn is run in bulk, one token per
// thread, instead of on every tile's critical path.

// Framework-independent device code: the kernel, device helpers and
// launch-geometry constants, relocated verbatim from the former
// csrc/mhc_pre_glm/compact_t256.cuh. Included (through that file's remaining
// TVM-FFI binding tail) by exactly one TU,
// csrc/mhc_pre_glm/mhc_pre_glm_sm103.cu.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace persistent_compact_256 {
namespace {
namespace cg = cooperative_groups;

constexpr int HCN = 4;
constexpr int HD = 4096;
constexpr int KK = 16384;
constexpr int NFR = 24;
constexpr int CLU = 16;          // CTAs per cluster == hidden slices
constexpr int HSL = HD / CLU;    // 256 hidden elements per CTA
constexpr int KCTA = HCN * HSL;  // 1024 contraction elements per CTA
constexpr int NWARP = 8;
constexpr int NTH = NWARP * 32;    // 256
constexpr int TT = 16;             // tokens per tile
constexpr int KPW = KCTA / NWARP;  // 128 k per warp
constexpr int KST = KPW / 16;      // 8 mma k-steps per warp
constexpr int RS = KCTA + 8;       // smem residual row stride (bf16)
constexpr int FSB = KPW + 8;       // fn staging row stride (bf16)
constexpr int ACCW = 28;           // 24 logits + sqrsum, float4 aligned
constexpr int CAP = 64;            // deferred sinkhorn slots per CTA
constexpr int NCMAX = 16;          // resident clusters
constexpr float EPSF = 1.0e-6f;

__device__ __forceinline__ void cpa16(void* dst, const void* src) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpa_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void ldmat4(uint32_t (&r)[4], const void* p) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldmat2(uint32_t (&r)[2], const void* p) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t cvt_bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t fma_bf2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
template <int PublishMode>
__device__ __forceinline__ void cluster_bar() {
  if constexpr (PublishMode == 2) {
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n" ::: "memory");
  } else if constexpr (PublishMode == 1) {
    if (threadIdx.x == 0)
      asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_bar_publish(int writer_threads) {
  if (static_cast<int>(threadIdx.x) < writer_threads) {
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t peer_addr(const void* p, int rank) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint32_t d;
  asm("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}
// The memory clobber is required: without it these peer reads are not memory
// operations to the compiler and can be hoisted above the cluster barrier.
__device__ __forceinline__ float ld_cluster(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_cluster_v4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cluster_v4_b32(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_cluster(uint32_t a, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ void rownorm(float (&m)[16]) {
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    float s = (m[4 * a] + m[4 * a + 1]) + (m[4 * a + 2] + m[4 * a + 3]) + EPSF;
    float inv = 1.0f / s;
#pragma unroll
    for (int b = 0; b < 4; ++b) m[4 * a + b] *= inv;
  }
}
__device__ __forceinline__ void colnorm(float (&m)[16]) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    float s = (m[b] + m[4 + b]) + (m[8 + b] + m[12 + b]) + EPSF;
    float inv = 1.0f / s;
#pragma unroll
    for (int a = 0; a < 4; ++a) m[4 * a + b] *= inv;
  }
}

__device__ __forceinline__ float rcp_fast(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void sinkhorn20_factored(float (&m)[16]) {
  float R0 = 1.0f, R1 = 1.0f, R2 = 1.0f, R3 = 1.0f;
  float S0 = ((m[0] + m[4]) + (m[8] + m[12])) + EPSF;
  float S1 = ((m[1] + m[5]) + (m[9] + m[13])) + EPSF;
  float S2 = ((m[2] + m[6]) + (m[10] + m[14])) + EPSF;
  float S3 = ((m[3] + m[7]) + (m[11] + m[15])) + EPSF;
#pragma unroll 1
  for (int it = 0; it < 19; ++it) {
    const float e0 = EPSF * R0, e1 = EPSF * R1;
    const float e2 = EPSF * R2, e3 = EPSF * R3;
    const float T0 = rcp_fast(S0), T1 = rcp_fast(S1);
    const float T2 = rcp_fast(S2), T3 = rcp_fast(S3);
    R0 = fmaf(m[1], T1, fmaf(m[0], T0, e0)) + fmaf(m[3], T3, m[2] * T2);
    R1 = fmaf(m[5], T1, fmaf(m[4], T0, e1)) + fmaf(m[7], T3, m[6] * T2);
    R2 = fmaf(m[9], T1, fmaf(m[8], T0, e2)) + fmaf(m[11], T3, m[10] * T2);
    R3 = fmaf(m[13], T1, fmaf(m[12], T0, e3)) + fmaf(m[15], T3, m[14] * T2);
    const float f0 = EPSF * S0, f1 = EPSF * S1;
    const float f2 = EPSF * S2, f3 = EPSF * S3;
    const float U0 = rcp_fast(R0), U1 = rcp_fast(R1);
    const float U2 = rcp_fast(R2), U3 = rcp_fast(R3);
    S0 = fmaf(m[4], U1, fmaf(m[0], U0, f0)) + fmaf(m[12], U3, m[8] * U2);
    S1 = fmaf(m[5], U1, fmaf(m[1], U0, f1)) + fmaf(m[13], U3, m[9] * U2);
    S2 = fmaf(m[6], U1, fmaf(m[2], U0, f2)) + fmaf(m[14], U3, m[10] * U2);
    S3 = fmaf(m[7], U1, fmaf(m[3], U0, f3)) + fmaf(m[15], U3, m[11] * U2);
  }
  const float T0 = rcp_fast(S0), T1 = rcp_fast(S1);
  const float T2 = rcp_fast(S2), T3 = rcp_fast(S3);
  const float U0 = rcp_fast(R0), U1 = rcp_fast(R1);
  const float U2 = rcp_fast(R2), U3 = rcp_fast(R3);
  m[0] *= U0 * T0;
  m[1] *= U0 * T1;
  m[2] *= U0 * T2;
  m[3] *= U0 * T3;
  m[4] *= U1 * T0;
  m[5] *= U1 * T1;
  m[6] *= U1 * T2;
  m[7] *= U1 * T3;
  m[8] *= U2 * T0;
  m[9] *= U2 * T1;
  m[10] *= U2 * T2;
  m[11] *= U2 * T3;
  m[12] *= U3 * T0;
  m[13] *= U3 * T1;
  m[14] *= U3 * T2;
  m[15] *= U3 * T3;
}

// softmax over the last dim, then 20 sinkhorn repeats, one token per thread.
__device__ __forceinline__ void sinkhorn_flush(const float* __restrict__ defv,
                                               const int* __restrict__ deft, int cnt,
                                               float* __restrict__ comb_g, int tid) {
  for (int c = tid; c < cnt; c += NTH) {
    alignas(16) float m[16];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      *reinterpret_cast<float4*>(m + q * 4) =
          *reinterpret_cast<const float4*>(defv + c * 16 + q * 4);
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      float mx = fmaxf(fmaxf(m[4 * a], m[4 * a + 1]), fmaxf(m[4 * a + 2], m[4 * a + 3]));
      float e0 = __expf(m[4 * a] - mx);
      float e1 = __expf(m[4 * a + 1] - mx);
      float e2 = __expf(m[4 * a + 2] - mx);
      float e3 = __expf(m[4 * a + 3] - mx);
      float inv = 1.0f / ((e0 + e1) + (e2 + e3));
      m[4 * a] = e0 * inv + EPSF;
      m[4 * a + 1] = e1 * inv + EPSF;
      m[4 * a + 2] = e2 * inv + EPSF;
      m[4 * a + 3] = e3 * inv + EPSF;
    }
    sinkhorn20_factored(m);
    float* o = comb_g + (size_t)deft[c] * 16;
#pragma unroll
    for (int q = 0; q < 4; ++q)
      *reinterpret_cast<float4*>(o + q * 4) = *reinterpret_cast<const float4*>(m + q * 4);
  }
}

template <bool SPLIT, bool MULTI>
__global__ void __cluster_dims__(CLU, 1, 1) __launch_bounds__(NTH, 2)
    mhc_kernel(const __nv_bfloat16* __restrict__ res_g, const float* __restrict__ fn_g,
               const float* __restrict__ sc_g, const float* __restrict__ ba_g,
               float* __restrict__ post_g, float* __restrict__ comb_g,
               __nv_bfloat16* __restrict__ li_g, int T, int ntiles, int NC) {
  constexpr int NTA = SPLIT ? 2 : 3;
  // A cluster that owns several tiles must double buffer the published logit
  // block: the comb group has no full cluster barrier between a tile's peer
  // reads and the next tile's writes, so a single stage would race.
  constexpr bool ONETILE = SPLIT && !MULTI;
  constexpr int NSTAGE = ONETILE ? 1 : 2;
  constexpr int DEFCAP = ONETILE ? 1 : CAP;
  constexpr size_t RES_BYTES = (size_t)TT * RS * 2;
  constexpr size_t FNS_BYTES = (size_t)KST * NTA * NWARP * FSB * 2;
  constexpr size_t SECOND_BYTES = FNS_BYTES > RES_BYTES ? FNS_BYTES : RES_BYTES;
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* resb = reinterpret_cast<__nv_bfloat16*>(smem);
  float* stg = reinterpret_cast<float*>(smem + RES_BYTES + SECOND_BYTES);
  float* stq = stg + NWARP * TT * NFR;
  float* accs = stq + NWARP * TT;  // [NSTAGE][TT][ACCW], read by peers
  uint32_t* gateb =
      reinterpret_cast<uint32_t*>(accs + NSTAGE * TT * ACCW);  // [4]            read by peers
  uint32_t* gcache = gateb + 4;                                // [TT][4]        CTA local
  float* cst = reinterpret_cast<float*>(gcache + TT * 4);
  float* defv = cst + 28;
  int* deft = reinterpret_cast<int*>(defv + DEFCAP * 16);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int rank = blockIdx.x;  // 0..CLU-1
  const int cid0 = blockIdx.y;  // cluster index
  // When the token count leaves clusters idle, two cluster groups split the
  // fn rows instead: group 0 takes n-tile 0 (pre/post gates + layer_input),
  // group 1 takes n-tiles 1-2 (the 16 comb logits + sinkhorn).  The critical
  // CTA then reads 64KB of fn instead of 98KB, and the sinkhorn chain overlaps
  // group 0's layer_input.
  const int cid = SPLIT ? (cid0 >> 1) : cid0;
  const int grp = SPLIT ? (cid0 & 1) : 0;
  const int nc = SPLIT ? (NC >> 1) : NC;
  const int ntbase = (SPLIT && grp) ? 1 : 0;
  const int ntcount = SPLIT ? (grp ? 2 : 1) : 3;
  const bool do_gate = (ntbase == 0);
  const bool do_comb = (ntbase + ntcount == 3);
  cluster_arrive_relaxed();

  // The scale/bias tensors are rotated per timed iteration, so these are cold
  // global reads.  Keeping them in warp-0 registers instead of publishing them
  // through shared memory moves the full miss latency off the prologue: the
  // loads issue here and are not consumed until the epilogue.
  float sc_reg = 0.0f;
  float ba_reg = 0.0f;
  if (warp == 0) {
    if (lane < 3) sc_reg = sc_g[lane];
    if (lane < NFR) ba_reg = ba_g[lane];
  }

  // ---- staging coordinates for residual cp.async ------------------------
  const int cp_t = tid >> 7;        // 0..1 token sub-index
  const int cp_i = (tid >> 5) & 3;  // 0..3 stream
  const int cp_o = (tid & 31) * 8;  // bf16 offset inside the 256 slice

  // ---- first residual tile ----------------------------------------------
  int tb = cid * TT;
  {
#pragma unroll
    for (int s = 0; s < TT / 2; ++s) {
      const int t = s * 2 + cp_t;
      const int g = tb + t;
      __nv_bfloat16* dst = resb + (size_t)t * RS + cp_i * HSL + cp_o;
      if (g < T) {
        cpa16(dst, res_g + (size_t)g * KK + cp_i * HD + rank * HSL + cp_o);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      }
    }
    cpa_commit();
  }

  // ---- projection weights for this CTA's k slice, into registers ---------
  // Staged through shared memory: the global reads are fully coalesced
  // (one warp covers a 512B run of one fn row) and the register extraction is
  // a single ldmatrix per (kstep, n-tile) instead of 48 strided 8B loads.
  uint32_t Bf[KST][NTA][2];
  {
    __nv_bfloat16* fns = resb + (size_t)TT * RS;
    const int lk = lane * 4;
    float4 gall[KST][NTA];
#pragma unroll
    for (int u = 0; u < KST; ++u) {
      const int gk = (u >> 1) * HD + rank * HSL + (u & 1) * KPW;
#pragma unroll
      for (int r = 0; r < NTA; ++r)
        gall[u][r] = (r < ntcount)
                         ? *reinterpret_cast<const float4*>(
                               fn_g + (size_t)((ntbase + r) * NWARP + warp) * KK + gk + lk)
                         : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < KST; ++u) {
      __nv_bfloat16* buf = fns + (size_t)u * (NTA * NWARP * FSB);
#pragma unroll
      for (int r = 0; r < NTA; ++r) {
        if (r < ntcount) {
          uint32_t pk[2];
          pk[0] = cvt_bf2(gall[u][r].x, gall[u][r].y);
          pk[1] = cvt_bf2(gall[u][r].z, gall[u][r].w);
          *reinterpret_cast<uint2*>(buf + (size_t)(r * NWARP + warp) * FSB + lk) =
              *reinterpret_cast<uint2*>(pk);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < KST; ++u) {
      const __nv_bfloat16* buf = fns + (size_t)u * (NTA * NWARP * FSB);
      const int t16 = lane & 15;
      const __nv_bfloat16* bp = buf + (size_t)(t16 & 7) * FSB + warp * 16 + (t16 >> 3) * 8;
#pragma unroll
      for (int nt = 0; nt < NTA; ++nt) {
        if (nt < ntcount) {
          ldmat2(Bf[u][nt], bp + (size_t)nt * 8 * FSB);
        } else {
          Bf[u][nt][0] = 0u;
          Bf[u][nt][1] = 0u;
        }
      }
    }
  }

  cluster_wait();

  int cur = 0;
  int cnt = 0;
  const int ntl = (ntiles - cid + nc - 1) / nc;  // tiles for this cluster

  for (int it = 0; it < ntl; ++it) {
    const int tile = cid + it * nc;
    tb = tile * TT;
    const int active = min(T - tb, TT);
    __nv_bfloat16* rbuf = resb + (size_t)cur * TT * RS;

    cpa_wait0();
    __syncthreads();

    // prefetch next tile
    if (it + 1 < ntl) {
      const int ntb = (cid + (it + 1) * nc) * TT;
      __nv_bfloat16* nbuf = resb + (size_t)(cur ^ 1) * TT * RS;
#pragma unroll
      for (int s = 0; s < TT / 2; ++s) {
        const int t = s * 2 + cp_t;
        const int g = ntb + t;
        __nv_bfloat16* dst = nbuf + (size_t)t * RS + cp_i * HSL + cp_o;
        if (g < T) {
          cpa16(dst, res_g + (size_t)g * KK + cp_i * HD + rank * HSL + cp_o);
        } else {
          *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
      }
      cpa_commit();
    }

    // ---------------- projection GEMM -----------------------------------
    float acc3[NTA][4];
#pragma unroll
    for (int nt = 0; nt < NTA; ++nt)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc3[nt][q] = 0.0f;
    float sq0 = 0.0f, sq1 = 0.0f;
    {
      const __nv_bfloat16* ap = rbuf + (size_t)(lane & 15) * RS + warp * 16 + (lane >> 4) * 8;
      uint32_t ab[2][4];
      ldmat4(ab[0], ap);
#pragma unroll
      for (int s = 0; s < KST; ++s) {
        if (s + 1 < KST) ldmat4(ab[(s + 1) & 1], ap + (s + 1) * KPW);
        uint32_t(&a)[4] = ab[s & 1];
        const int sqrow = lane >> 2;
        if (sqrow < active) {
          float2 f0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[0]));
          float2 f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[2]));
          sq0 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, sq0));
          sq0 = fmaf(f2.x, f2.x, fmaf(f2.y, f2.y, sq0));
        }
        if (sqrow + 8 < active) {
          float2 f1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[1]));
          float2 f3 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[3]));
          sq1 = fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, sq1));
          sq1 = fmaf(f3.x, f3.x, fmaf(f3.y, f3.y, sq1));
        }
#pragma unroll
        for (int nt = 0; nt < NTA; ++nt) {
          if (nt < ntcount) {
            mma16816(acc3[nt], a, Bf[s][nt][0], Bf[s][nt][1]);
          }
        }
      }
    }
    {
      const int r0 = lane >> 2;
      const int c0 = (lane & 3) << 1;
      float* sp = stg + warp * (TT * NFR);
#pragma unroll
      for (int nt = 0; nt < NTA; ++nt) {
        if (nt < ntcount) {
          const int cc = (ntbase + nt) * 8 + c0;
          if (r0 < active) {
            sp[r0 * NFR + cc] = acc3[nt][0];
            sp[r0 * NFR + cc + 1] = acc3[nt][1];
          }
          if (r0 + 8 < active) {
            sp[(r0 + 8) * NFR + cc] = acc3[nt][2];
            sp[(r0 + 8) * NFR + cc + 1] = acc3[nt][3];
          }
        }
      }
      sq0 += __shfl_xor_sync(0xffffffffu, sq0, 1);
      sq0 += __shfl_xor_sync(0xffffffffu, sq0, 2);
      sq1 += __shfl_xor_sync(0xffffffffu, sq1, 1);
      sq1 += __shfl_xor_sync(0xffffffffu, sq1, 2);
      if ((lane & 3) == 0) {
        if (r0 < active) stq[warp * TT + r0] = sq0;
        if (r0 + 8 < active) stq[warp * TT + r0 + 8] = sq1;
      }
    }
    __syncthreads();

    // Complete the previous tile's peer-lifetime phase before reusing the
    // distributed record/gate buffers.  The wait is deliberately deferred
    // until after this tile's local projection so its latency overlaps useful
    // MMA work, but it must precede the first new remote store.
    if (it > 0) cluster_wait();

    // --------- reduce over warps into this CTA's published buffer -------
    float* acct = accs + (ONETILE ? 0 : (it & 1)) * (TT * ACCW);
    {
#pragma unroll
      for (int b = 0; b < (TT * NTA * 8 + NTH - 1) / NTH; ++b) {
        const int i = tid + b * NTH;
        const int owned_cols = ntcount * 8;
        if (i < active * owned_cols) {
          const int t = i / owned_cols;
          const int col = ntbase * 8 + i - t * owned_cols;
          const int off = t * NFR + col;
          float v0 = stg[off] + stg[4 * (TT * NFR) + off];
          float v1 = stg[1 * (TT * NFR) + off] + stg[5 * (TT * NFR) + off];
          float v2 = stg[2 * (TT * NFR) + off] + stg[6 * (TT * NFR) + off];
          float v3 = stg[3 * (TT * NFR) + off] + stg[7 * (TT * NFR) + off];
          float v = (v0 + v1) + (v2 + v3);
          st_cluster(peer_addr(acct + rank * ACCW + col, t), v);
        }
      }
      if (tid < active) {
        float v0 = stq[tid] + stq[4 * TT + tid];
        float v1 = stq[1 * TT + tid] + stq[5 * TT + tid];
        float v2 = stq[2 * TT + tid] + stq[6 * TT + tid];
        float v3 = stq[3 * TT + tid] + stq[7 * TT + tid];
        float v = (v0 + v1) + (v2 + v3);
        st_cluster(peer_addr(acct + rank * ACCW + 24, tid), v);
      }
    }
    cluster_bar_publish(active * ntcount * 8);

    // --------- epilogue on the owning CTA --------------------------------
    {
      const int g = tb + rank;
      if (warp == 0 && g < T) {
        float v = 0.0f;
        const bool owned_lane = lane == 24 || (lane >= ntbase * 8 && lane < (ntbase + ntcount) * 8);
        if (lane < 25 && owned_lane) {
          float t4[CLU];
#pragma unroll
          for (int s = 0; s < CLU; ++s) t4[s] = acct[s * ACCW + lane];
#pragma unroll
          for (int d = CLU >> 1; d; d >>= 1)
#pragma unroll
            for (int s = 0; s < d; ++s) t4[s] += t4[s + d];
          v = t4[0];
        }
        const float ss = __shfl_sync(0xffffffffu, v, 24);
        const float rsq = rsqrtf(ss * (1.0f / 16384.0f) + EPSF);
        const float mix = v * rsq;
        const float bq = ba_reg;
        const float sc0 = __shfl_sync(0xffffffffu, sc_reg, 0);
        const float sc1 = __shfl_sync(0xffffffffu, sc_reg, 1);
        const float sc2 = __shfl_sync(0xffffffffu, sc_reg, 2);
        if (do_gate && lane < 4) {
          const float pre = sigmoidf(mix * sc0 + bq) + EPSF;
          gateb[lane] = cvt_bf2(pre, pre);
        }
        if (do_gate && lane >= 4 && lane < 8)
          post_g[(size_t)g * 4 + lane - 4] = sigmoidf(mix * sc1 + bq);
        if (do_comb && lane >= 8 && lane < NFR) defv[cnt * 16 + (lane - 8)] = mix * sc2 + bq;
        if (do_comb && lane == 0) deft[cnt] = g;
      }
      if (g < T && do_comb) ++cnt;
    }
    if (do_gate) cluster_bar_publish(4);

    // --------- pre-gated layer input -------------------------------------
    if (do_gate) {
      const int active_gate = min(T - tb, TT);
      if (tid < active_gate) {
        const uint4 pg = ld_cluster_v4_b32(peer_addr(gateb, tid));
        *reinterpret_cast<uint4*>(gcache + tid * 4) = pg;
      }
      __syncthreads();
      cluster_arrive_relaxed();
      {
        const int tsub = tid >> 5;
        const int hl = (tid & 31) * 8;
#pragma unroll
        for (int t = tsub; t < TT; t += NWARP) {
          const int g = tb + t;
          if (g >= T) break;
          const uint4 pg = *reinterpret_cast<const uint4*>(gcache + t * 4);
          const uint32_t p0 = pg.x;
          const uint32_t p1 = pg.y;
          const uint32_t p2 = pg.z;
          const uint32_t p3 = pg.w;
          const __nv_bfloat16* rp = rbuf + (size_t)t * RS + hl;
          uint4 w0 = *reinterpret_cast<const uint4*>(rp);
          uint4 w1 = *reinterpret_cast<const uint4*>(rp + HSL);
          uint4 w2 = *reinterpret_cast<const uint4*>(rp + 2 * HSL);
          uint4 w3 = *reinterpret_cast<const uint4*>(rp + 3 * HSL);
          const uint32_t* q0 = reinterpret_cast<const uint32_t*>(&w0);
          const uint32_t* q1 = reinterpret_cast<const uint32_t*>(&w1);
          const uint32_t* q2 = reinterpret_cast<const uint32_t*>(&w2);
          const uint32_t* q3 = reinterpret_cast<const uint32_t*>(&w3);
          uint4 out;
          uint32_t* op = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            uint32_t v = fma_bf2(q0[e], p0, 0u);
            v = fma_bf2(q1[e], p1, v);
            v = fma_bf2(q2[e], p2, v);
            op[e] = fma_bf2(q3[e], p3, v);
          }
          *reinterpret_cast<uint4*>(li_g + (size_t)g * HD + rank * HSL + hl) = out;
        }
      }
    }
    if (!do_gate) cluster_arrive_relaxed();

    if (cnt == DEFCAP) {
      __syncthreads();
      sinkhorn_flush(defv, deft, cnt, comb_g, tid);
      if (it + 1 < ntl) __syncthreads();
      cnt = 0;
    }
    cur ^= 1;
  }
  if (cnt > 0) {
    __syncthreads();
    sinkhorn_flush(defv, deft, cnt, comb_g, tid);
  }
  cluster_wait();
}

template <bool SPLIT, bool MULTI>
size_t smem_bytes() {
  constexpr int NTA = SPLIT ? 2 : 3;
  constexpr bool ONETILE = SPLIT && !MULTI;
  constexpr int NSTAGE = ONETILE ? 1 : 2;
  constexpr int DEFCAP = ONETILE ? 1 : CAP;
  constexpr size_t res_bytes = (size_t)TT * RS * 2;
  constexpr size_t fns_bytes = (size_t)KST * NTA * NWARP * FSB * 2;
  const size_t second_bytes = fns_bytes > res_bytes ? fns_bytes : res_bytes;
  return res_bytes + second_bytes +
         (size_t)(NWARP * TT * NFR + NWARP * TT + NSTAGE * TT * ACCW + 4 + TT * 4 + 28) * 4 +
         (size_t)DEFCAP * 16 * 4 + (size_t)DEFCAP * 4;
}

}  // namespace

}  // namespace persistent_compact_256
