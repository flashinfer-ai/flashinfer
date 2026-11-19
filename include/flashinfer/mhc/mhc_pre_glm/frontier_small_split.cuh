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
// csrc/mhc_pre_glm/frontier_small_split.cuh. Included (through that file's remaining
// TVM-FFI binding tail) by exactly one TU,
// csrc/mhc_pre_glm/mhc_pre_glm_sm103.cu.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace persistent_split {
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
constexpr int CAP = 1;             // one token owned by each one-tile CTA
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
// Cluster rendezvous.  PublishMode: 0 = lifetime only (relaxed arrival),
// 1 = publish CTA-local shared memory written by thread 0 (restricted
// release fence by the producer, then a relaxed arrival), 2 = publish
// owner-push st.shared::cluster stores into peer CTA windows.  Mode 2 must
// use a release arrival: fence.release.sync_restrict::shared::cta.cluster
// covers only the issuing CTA's own shared::cta window, so it does not
// order remote DSMEM stores.  The release arrival orders every prior
// shared-memory access of the arriving thread at cluster scope, and the
// barrier.cluster.wait below is an acquire, so the token owner's reduction
// reads the pushed partials with a happens-before edge.
template <int PublishMode>
__device__ __forceinline__ void cluster_bar() {
  if constexpr (PublishMode == 2) {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  } else {
    if constexpr (PublishMode == 1) {
      if (threadIdx.x == 0)
        asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  }
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
// Release arrival without a wait: orders this thread's prior shared-memory
// accesses -- reads included -- before anything peers do after their
// (acquire) cluster_wait().  Used where earlier ldmatrix reads of a staging
// slab must complete before peers are allowed to reuse that storage.
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
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
__device__ __forceinline__ void st_cluster(uint32_t a, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(a), "f"(v) : "memory");
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
    float m[16];
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

__global__ void __cluster_dims__(CLU, 1, 1) __launch_bounds__(NTH, 1)
    mhc_kernel(const __nv_bfloat16* __restrict__ res_g, const float* __restrict__ fn_g,
               const float* __restrict__ sc_g, const float* __restrict__ ba_g,
               float* __restrict__ post_g, float* __restrict__ comb_g,
               __nv_bfloat16* __restrict__ li_g, int split_pairs) {
  extern __shared__ __align__(16) char smem[];
  __nv_bfloat16* resb = reinterpret_cast<__nv_bfloat16*>(smem);
  // T=128 dispatches exactly one tile per cluster.  Weight conversion reuses
  // the workspace after the single resident residual buffer, then the same
  // bytes become the warp-reduction staging area.
  float* stg = reinterpret_cast<float*>(smem + (size_t)TT * RS * 2);
  float* stq = stg + NWARP * TT * NFR;
  float* accs = stq + NWARP * TT;   // [TT][ACCW] read by peers
  float* gateb = accs + TT * ACCW;  // [4] read by peers
  float* gcache = gateb + 4;        // [TT][4]        CTA local
  float* cst = gcache + TT * 4;
  float* defv = cst + 28;
  int* deft = reinterpret_cast<int*>(defv + CAP * 16);

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
  const bool paired = cid0 < 2 * split_pairs;
  const int cid = paired ? (cid0 >> 1) : (cid0 - split_pairs);
  const int grp = paired ? (cid0 & 1) : 0;
  const int nt0 = (paired && grp) ? 1 : 0;
  const int nt1 = (paired && !grp) ? 1 : 3;
  const bool do_gate = (nt0 == 0);
  const bool do_comb = (nt1 == 3);

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
      cpa16(dst, res_g + (size_t)g * KK + cp_i * HD + rank * HSL + cp_o);
    }
    cpa_commit();
  }

  // ---- projection weights for this CTA's k slice, into registers ---------
  // Staged through shared memory: the global reads are fully coalesced
  // (one warp covers a 512B run of one fn row) and the register extraction is
  // a single ldmatrix per (kstep, n-tile) instead of 48 strided 8B loads.
  // Unit u covers k-local [u*128, u*128+128); warp w takes k-local u*128+w*16.
  uint32_t Bf[KST][3][2];
  {
    __nv_bfloat16* fns = resb + (size_t)TT * RS;
    const int lk = lane * 4;
    // All 24 float4 are issued up front: one memory round trip for the whole
    // 98KB slice instead of one per staged unit.
    float4 gall[KST][3];
#pragma unroll
    for (int u = 0; u < KST; ++u) {
      const int gk = (u >> 1) * HD + rank * HSL + (u & 1) * KPW;
#pragma unroll
      for (int r = 0; r < 3; ++r)
        gall[u][r] =
            (r >= nt0 && r < nt1)
                ? *reinterpret_cast<const float4*>(fn_g + (size_t)(r * NWARP + warp) * KK + gk + lk)
                : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < KST; ++u) {
      __nv_bfloat16* buf = fns + (u & 1) * (NFR * FSB);
#pragma unroll
      for (int r = 0; r < 3; ++r) {
        uint32_t pk[2];
        pk[0] = cvt_bf2(gall[u][r].x, gall[u][r].y);
        pk[1] = cvt_bf2(gall[u][r].z, gall[u][r].w);
        *reinterpret_cast<uint2*>(buf + (size_t)(r * NWARP + warp) * FSB + lk) =
            *reinterpret_cast<uint2*>(pk);
      }
      __syncthreads();
      const int t16 = lane & 15;
      const __nv_bfloat16* bp = buf + (size_t)(t16 & 7) * FSB + warp * 16 + (t16 >> 3) * 8;
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        if (nt >= nt0 && nt < nt1) {
          ldmat2(Bf[u][nt], bp + (size_t)nt * 8 * FSB);
        } else {
          Bf[u][nt][0] = 0u;
          Bf[u][nt][1] = 0u;
        }
      }
      // The next unit writes the other parity buffer. Its pre-load barrier
      // proves all threads finished this ldmatrix before either parity is
      // reused two units later.
    }
  }

  // The weight-conversion workspace aliases the later reduction records.
  // All peers must finish staging before any CTA can push into that storage.
  // The arrival must be a release so the ldmatrix reads of the staging slab
  // above are ordered before peers' remote stores reuse it (an unpaired
  // tile's last weight rows overlap the first token owner's accumulator
  // records); a relaxed arrival would leave those reads unordered against
  // the stores after the peers' acquire wait.
  cluster_arrive_release();

  int cnt = 0;
  constexpr int ntl = 1;

  for (int it = 0; it < ntl; ++it) {
    const int tile = cid;
    tb = tile * TT;
    __nv_bfloat16* rbuf = resb;

    cpa_wait0();
    __syncthreads();

    // ---------------- projection GEMM -----------------------------------
    float acc3[3][4];
#pragma unroll
    for (int nt = 0; nt < 3; ++nt)
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
        float2 f0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[0]));
        float2 f1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[1]));
        float2 f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[2]));
        float2 f3 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&a[3]));
        sq0 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, sq0));
        sq0 = fmaf(f2.x, f2.x, fmaf(f2.y, f2.y, sq0));
        sq1 = fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, sq1));
        sq1 = fmaf(f3.x, f3.x, fmaf(f3.y, f3.y, sq1));
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
          if (nt >= nt0 && nt < nt1) mma16816(acc3[nt], a, Bf[s][nt][0], Bf[s][nt][1]);
        }
      }
    }
    {
      const int r0 = lane >> 2;
      const int c0 = (lane & 3) << 1;
      float* sp = stg + warp * (TT * NFR);
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        const int cc = nt * 8 + c0;
        sp[r0 * NFR + cc] = acc3[nt][0];
        sp[r0 * NFR + cc + 1] = acc3[nt][1];
        sp[(r0 + 8) * NFR + cc] = acc3[nt][2];
        sp[(r0 + 8) * NFR + cc + 1] = acc3[nt][3];
      }
      sq0 += __shfl_xor_sync(0xffffffffu, sq0, 1);
      sq0 += __shfl_xor_sync(0xffffffffu, sq0, 2);
      sq1 += __shfl_xor_sync(0xffffffffu, sq1, 1);
      sq1 += __shfl_xor_sync(0xffffffffu, sq1, 2);
      if ((lane & 3) == 0) {
        stq[warp * TT + r0] = sq0;
        stq[warp * TT + r0 + 8] = sq1;
      }
    }
    __syncthreads();

    // Local projection overlaps the cluster rendezvous.  Complete it before
    // the first owner-push remote shared-memory store below.
    cluster_wait();

    // --------- reduce over warps into this CTA's published buffer -------
    float* acct = accs;
    {
#pragma unroll
      for (int b = 0; b < (TT * NFR + NTH - 1) / NTH; ++b) {
        const int i = tid + b * NTH;
        if (i < TT * NFR) {
          float v = stg[i];
#pragma unroll
          for (int w = 1; w < NWARP; ++w) v += stg[w * (TT * NFR) + i];
          const int t = i / NFR;
          st_cluster(peer_addr(acct + rank * ACCW + (i - t * NFR), t), v);
        }
      }
      if (tid < TT) {
        float v = stq[tid];
#pragma unroll
        for (int w = 1; w < NWARP; ++w) v += stq[w * TT + tid];
        st_cluster(peer_addr(acct + rank * ACCW + 24, tid), v);
      }
    }
    cluster_bar<2>();

    // --------- epilogue on the owning CTA --------------------------------
    {
      const int g = tb + rank;
      if (warp == 0) {
        float v = 0.0f;
        const bool owned_lane = lane == 24 || (lane >= nt0 * 8 && lane < nt1 * 8);
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
        if (do_gate) {
          const float pre = lane < 4 ? sigmoidf(mix * sc0 + bq) + EPSF : 0.0f;
          const float p0 = __shfl_sync(0xffffffffu, pre, 0);
          const float p1 = __shfl_sync(0xffffffffu, pre, 1);
          const float p2 = __shfl_sync(0xffffffffu, pre, 2);
          const float p3 = __shfl_sync(0xffffffffu, pre, 3);
          if (lane >= 4 && lane < 8) post_g[(size_t)g * 4 + (lane - 4)] = sigmoidf(mix * sc1 + bq);
          if (lane == 0) {
            uint32_t* packed_gateb = reinterpret_cast<uint32_t*>(gateb);
            packed_gateb[0] = cvt_bf2(p0, p0);
            packed_gateb[1] = cvt_bf2(p1, p1);
            packed_gateb[2] = cvt_bf2(p2, p2);
            packed_gateb[3] = cvt_bf2(p3, p3);
          }
        }
        if (do_comb && lane >= 8 && lane < NFR) defv[cnt * 16 + (lane - 8)] = mix * sc2 + bq;
        if (do_comb && lane == 0) deft[cnt] = g;
      }
      if (do_comb) ++cnt;
    }
    if (do_gate) cluster_bar<1>();

    // --------- pre-gated layer input -------------------------------------
    if (do_gate) {
      const bool overlap_sink = do_comb;
      if (overlap_sink && warp == 0) {
        __syncwarp();
        sinkhorn_flush(defv, deft, cnt, comb_g, tid);
      }
      if (!overlap_sink || warp > 0) {
        {
          const int tsub = overlap_sink ? (warp - 1) : warp;
          const int tstep = overlap_sink ? (NWARP - 1) : NWARP;
          const int hl = (tid & 31) * 8;
#pragma unroll
          for (int t = tsub; t < TT; t += tstep) {
            const int g = tb + t;
            uint4 pg = make_uint4(0u, 0u, 0u, 0u);
            if (lane == 0) pg = ld_cluster_v4_b32(peer_addr(gateb, t));
            pg.x = __shfl_sync(0xffffffffu, pg.x, 0);
            pg.y = __shfl_sync(0xffffffffu, pg.y, 0);
            pg.z = __shfl_sync(0xffffffffu, pg.z, 0);
            pg.w = __shfl_sync(0xffffffffu, pg.w, 0);
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
      if (overlap_sink) cnt = 0;
    }

    if (cnt == CAP) {
      __syncwarp();
      sinkhorn_flush(defv, deft, cnt, comb_g, tid);
      cnt = 0;
    }
  }
  cluster_bar<0>();

  if (cnt > 0) {
    __syncthreads();
    sinkhorn_flush(defv, deft, cnt, comb_g, tid);
  }
}

size_t smem_bytes() {
  return (size_t)TT * RS * 2 +
         (size_t)(NWARP * TT * NFR + NWARP * TT + TT * ACCW + 4 + TT * 4 + 28) * 4 +
         (size_t)CAP * 16 * 4 + (size_t)CAP * 4;
}

}  // namespace

}  // namespace persistent_split
