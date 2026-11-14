// Fused silu-gated block-fp8 MoE for sm_120 (RTX PRO 6000 Blackwell).
// Hand written CUDA C++ ; no external compute libraries.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;
namespace cg = cooperative_groups;

#define HID   2048
#define INT_  512
#define TWO_I 1024
#define NEXP  256
#define TOPK  8

// e4m3 byte placed in bits 31..24 of p -> float value * 2^-120 (exact, incl. subnormals)
__device__ __forceinline__ float f8(uint32_t p) {
  uint32_t bits = (p & 0x80000000u) | ((p >> 4) & 0x07F00000u);
  return __int_as_float(bits);
}
#ifndef T1NR
#define T1NR 4
#endif
#ifndef T1PF
#define T1PF 0
#endif
#define P120F 1.3292279957849159e36f

struct X16 { float v[16]; };

__device__ __forceinline__ void unpack_bf16x16(uint4 a, uint4 b, X16 &o) {
  o.v[0]  = __int_as_float(a.x << 16); o.v[1]  = __int_as_float(a.x & 0xFFFF0000u);
  o.v[2]  = __int_as_float(a.y << 16); o.v[3]  = __int_as_float(a.y & 0xFFFF0000u);
  o.v[4]  = __int_as_float(a.z << 16); o.v[5]  = __int_as_float(a.z & 0xFFFF0000u);
  o.v[6]  = __int_as_float(a.w << 16); o.v[7]  = __int_as_float(a.w & 0xFFFF0000u);
  o.v[8]  = __int_as_float(b.x << 16); o.v[9]  = __int_as_float(b.x & 0xFFFF0000u);
  o.v[10] = __int_as_float(b.y << 16); o.v[11] = __int_as_float(b.y & 0xFFFF0000u);
  o.v[12] = __int_as_float(b.z << 16); o.v[13] = __int_as_float(b.z & 0xFFFF0000u);
  o.v[14] = __int_as_float(b.w << 16); o.v[15] = __int_as_float(b.w & 0xFFFF0000u);
}

__device__ __forceinline__ float dot16(uint4 w, const X16 &x) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  uint32_t u;
  u = w.x;
  s0 = fmaf(f8(u << 24), x.v[0], s0);  s1 = fmaf(f8(u << 16), x.v[1], s1);
  s2 = fmaf(f8(u << 8),  x.v[2], s2);  s3 = fmaf(f8(u),       x.v[3], s3);
  u = w.y;
  s0 = fmaf(f8(u << 24), x.v[4], s0);  s1 = fmaf(f8(u << 16), x.v[5], s1);
  s2 = fmaf(f8(u << 8),  x.v[6], s2);  s3 = fmaf(f8(u),       x.v[7], s3);
  u = w.z;
  s0 = fmaf(f8(u << 24), x.v[8], s0);  s1 = fmaf(f8(u << 16), x.v[9], s1);
  s2 = fmaf(f8(u << 8),  x.v[10], s2); s3 = fmaf(f8(u),       x.v[11], s3);
  u = w.w;
  s0 = fmaf(f8(u << 24), x.v[12], s0); s1 = fmaf(f8(u << 16), x.v[13], s1);
  s2 = fmaf(f8(u << 8),  x.v[14], s2); s3 = fmaf(f8(u),       x.v[15], s3);
  return (s0 + s1) + (s2 + s3);
}

__device__ __forceinline__ float dot16_native_generic(uint4 w,
                                                       const X16 &x) {
  const uint32_t *wp = reinterpret_cast<const uint32_t *>(&w);
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01)
        : "h"(static_cast<unsigned short>(wp[j])));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23)
        : "h"(static_cast<unsigned short>(wp[j] >> 16)));
    const float2 f01 =
        __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 f23 =
        __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    s0 = fmaf(f01.x, x.v[4 * j], s0);
    s1 = fmaf(f01.y, x.v[4 * j + 1], s1);
    s2 = fmaf(f23.x, x.v[4 * j + 2], s2);
    s3 = fmaf(f23.y, x.v[4 * j + 3], s3);
  }
  return (s0 + s1) + (s2 + s3);
}

template <bool NATIVE>
__device__ __forceinline__ float dot16_mode(uint4 w, const X16 &x) {
  if constexpr (NATIVE) return dot16_native_generic(w, x);
  return dot16(w, x);
}

__device__ __forceinline__ uint4 ldcg4(const void *p) {
  int4 t = __ldcg((const int4 *)p);
  uint4 r; r.x = t.x; r.y = t.y; r.z = t.z; r.w = t.w; return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------------------------------------------------------- routing prep
__global__ __launch_bounds__(256) void prep_kernel(
    const int *__restrict__ ids, int P, int MT,
    int *__restrict__ pair_idx, int *__restrict__ pos_of,
    int *__restrict__ tile_e, int *__restrict__ tile_p0, int *__restrict__ tile_cnt,
    int *__restrict__ ntiles) {
  __shared__ int cnt[NEXP];
  __shared__ int excl[NEXP];
  __shared__ int cursor[NEXP];
  __shared__ int wsum[8];
  __shared__ int wofs[8];
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  cnt[tid] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += 256) atomicAdd(&cnt[ids[p] & 255], 1);
  __syncthreads();
  int v = cnt[tid];
  int incl = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int n = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += n;
  }
  if (lane == 31) wsum[wid] = incl;
  __syncthreads();
  if (tid < 8) { int t = 0; for (int i = 0; i < tid; i++) t += wsum[i]; wofs[tid] = t; }
  __syncthreads();
  const int ex = incl - v + wofs[wid];
  excl[tid] = ex;
  cursor[tid] = ex;
  __syncthreads();
  for (int p = tid; p < P; p += 256) {
    int e = ids[p] & 255;
    int pos = atomicAdd(&cursor[e], 1);
    pair_idx[pos] = p;
    pos_of[p] = pos;
  }
  // tile scan
  int nt = (v + MT - 1) / MT;
  int inc2 = nt;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int n = __shfl_up_sync(0xffffffffu, inc2, off);
    if (lane >= off) inc2 += n;
  }
  __syncthreads();
  if (lane == 31) wsum[wid] = inc2;
  __syncthreads();
  if (tid < 8) { int t = 0; for (int i = 0; i < 8; i++) { if (i < tid) t += wsum[i]; }
                 wofs[tid] = t; if (tid == 7) ntiles[0] = t + wsum[7]; }
  __syncthreads();
  const int tb = inc2 - nt + wofs[wid];
  const int p0 = excl[tid];
  for (int q = 0; q < nt; q++) {
    tile_e[tb + q] = tid;
    tile_p0[tb + q] = p0 + q * MT;
    int c = v - q * MT; if (c > MT) c = MT;
    tile_cnt[tb + q] = c;
  }
}

// ---------------------------------------------------------------- FC1 + silu
// block = KW*RW warps.  warp -> (wj = k-chunk of 512, wr = row group of NR channels)
// lane owns 16 contiguous k inside its 512-k chunk.
template <int MT, int NR, int RW, bool IDENT, bool SKIP_PAD,
          bool NATIVE = false>
__global__ __launch_bounds__(32 * 4 * RW) void fc1_kernel(
    const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w13,
    const float *__restrict__ w13s, const int *__restrict__ tile_e,
    const int *__restrict__ tile_p0, const int *__restrict__ tile_cnt,
    const int *__restrict__ pair_idx, const int *__restrict__ ntiles,
    const int *__restrict__ ids, __nv_bfloat16 *__restrict__ inter) {
  constexpr int KW = 4;
  __shared__ float red[RW * NR * MT * 2][KW];
  const int tile = blockIdx.x;
  int e, cnt, p0;
  if (IDENT) { e = ids[tile] & 255; cnt = 1; p0 = tile; }
  else {
    if (tile >= ntiles[0]) return;
    e = tile_e[tile]; cnt = tile_cnt[tile]; p0 = tile_p0[tile];
  }
  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int wj = w & (KW - 1);
  const int wr = w >> 2;
  const int i0 = (blockIdx.y * RW + wr) * NR;
  const int lsub = lane >> 3;
  const float *ws = w13s + (size_t)e * 128;
  const int rbg = i0 >> 7;
  const float fp8_comp = NATIVE ? 1.0f : P120F;
  const float sg = ws[rbg * 16 + 4 * wj + lsub] * fp8_comp;
  const float su = ws[(rbg + 4) * 16 + 4 * wj + lsub] * fp8_comp;

  int tok[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    if (IDENT) tok[m] = p0 >> 3;
    else tok[m] = (m < cnt) ? (pair_idx[p0 + m] >> 3) : 0;
  }
  X16 xv[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    if (m < cnt) {
      const uint4 *xp = (const uint4 *)(x + (size_t)tok[m] * HID + wj * 512 + lane * 16);
      unpack_bf16x16(xp[0], xp[1], xv[m]);
    }
  }
  const uint8_t *wb = w13 + (size_t)e * TWO_I * HID + (size_t)i0 * HID + wj * 512 + lane * 16;
  uint4 wg[NR], wu[NR];
#pragma unroll
  for (int r = 0; r < NR; r++) {
    wg[r] = *(const uint4 *)(wb + (size_t)r * HID);
    wu[r] = *(const uint4 *)(wb + (size_t)(r + 512) * HID);
  }
#pragma unroll
  for (int r = 0; r < NR; r++) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if constexpr (SKIP_PAD) {
        if (m < cnt) {
          float g = warp_sum(dot16_mode<NATIVE>(wg[r], xv[m]) * sg);
          float u = warp_sum(dot16_mode<NATIVE>(wu[r], xv[m]) * su);
          if (lane == 0) {
            red[((wr * NR + r) * MT + m) * 2 + 0][wj] = g;
            red[((wr * NR + r) * MT + m) * 2 + 1][wj] = u;
          }
        }
      } else {
        float g = 0.f, u = 0.f;
        if (m < cnt) {
          g = dot16_mode<NATIVE>(wg[r], xv[m]) * sg;
          u = dot16_mode<NATIVE>(wu[r], xv[m]) * su;
        }
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane == 0) {
          red[((wr * NR + r) * MT + m) * 2 + 0][wj] = g;
          red[((wr * NR + r) * MT + m) * 2 + 1][wj] = u;
        }
      }
    }
  }
  __syncthreads();
  const int nout = RW * NR * MT;
  if (threadIdx.x < nout) {
    const int q = threadIdx.x;
    const int m = q % MT;
    const int rr = (q / MT) % NR;
    const int wwr = q / (MT * NR);
    if (m < cnt) {
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int c = 0; c < KW; c++) {
        g += red[((wwr * NR + rr) * MT + m) * 2 + 0][c];
        u += red[((wwr * NR + rr) * MT + m) * 2 + 1][c];
      }
      float s = g / (1.f + expf(-g));
      int ii = (blockIdx.y * RW + wwr) * NR + rr;
      inter[(size_t)(p0 + m) * INT_ + ii] = __float2bfloat16(s * u);
    }
  }
}

// ---------------------------------------------------------------- FC2 + combine
// block = TOPK warps; warp k handles expert slot k for NRH output channels.
template <int NRH, bool IDENT, bool NATIVE = false>
__global__ __launch_bounds__(32 * TOPK) void fc2_kernel(
    const __nv_bfloat16 *__restrict__ inter, const uint8_t *__restrict__ w2,
    const float *__restrict__ w2s, const int *__restrict__ topk_ids,
    const float *__restrict__ topk_weights, const int *__restrict__ pos_of,
    __nv_bfloat16 *__restrict__ out) {
  __shared__ float red[TOPK][NRH];
  __shared__ uint4 si[TOPK][64];
  const int t = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int k = threadIdx.x >> 5;
  const int h0 = blockIdx.y * NRH;
  const int lsub = lane >> 3;
  const int hb = h0 >> 7;
  const int rid = topk_ids[t * TOPK + k];
  const int e = rid & 255;
  const float tw =
      (unsigned)rid < (unsigned)NEXP ? topk_weights[t * TOPK + k] : 0.f;
  const int p = IDENT ? (t * TOPK + k) : pos_of[t * TOPK + k];
  const float sc = w2s[(size_t)e * 64 + hb * 4 + lsub] *
                   (NATIVE ? 1.0f : P120F) * tw;
  const uint4 *ip = (const uint4 *)(inter + (size_t)p * INT_);
  X16 xv;
  const uint8_t *wb = w2 + ((size_t)e * HID + h0) * INT_ + lane * 16;
  uint4 wr[NRH];
#pragma unroll
  for (int r = 0; r < NRH; r++) wr[r] = *(const uint4 *)(wb + (size_t)r * INT_);
  const uint4 ia = ip[lane];
  const uint4 ib = ip[lane + 32];
  const int s0 = (lane >> 1) + ((lane & 1) << 5);
  si[k][s0] = ia;
  si[k][s0 + 16] = ib;
  __syncwarp();
  unpack_bf16x16(si[k][lane], si[k][lane + 32], xv);
#pragma unroll
  for (int r = 0; r < NRH; r++) {
    float a = warp_sum(dot16_mode<NATIVE>(wr[r], xv) * sc);
    if (lane == 0) red[k][r] = a;
  }
  __syncthreads();
  if (threadIdx.x < NRH) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < TOPK; c++) s += red[c][threadIdx.x];
    out[(size_t)t * HID + h0 + threadIdx.x] = __float2bfloat16(s);
  }
}

// Higher-residency cooperative T=1 kernel. Each CTA computes 2*NR FC1 rows
// for one route and NR FC2 output rows, with FC1 stored in FC2 lane order.
__device__ __forceinline__ void t1_cp_async16(uint32_t s, const void *g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
}

// Cluster-hierarchical grid barrier.  The flat form has all 512 CTAs issue a
// read-modify-write to one 32-bit word; same-address atomics serialise in a
// single L2 slice, so that chain alone is a large part of the phase boundary.
// Blackwell schedules a cluster's CTAs together in one GPC and gives them a
// hardware barrier, so a cluster of 8 can elect one CTA to carry the global
// arrival: 64 atomics instead of 512, with no extra round trip through L2
// (which is what sank the software two-level and separate-release-line forms).
#define T1CLU 8
__device__ __forceinline__ void t1_cluster_barrier(unsigned int *bar) {
  cg::cluster_group cl = cg::this_cluster();
  __syncthreads();
  if (threadIdx.x == 0) {
    // Device-scope release of this CTA's intermediate stores, so the elected
    // CTA's atomic publishes the whole cluster's work.
    asm volatile("fence.release.gpu;\n" ::: "memory");
  }
  cl.sync();
  if (cl.block_rank() == 0 && threadIdx.x == 0) {
    unsigned int old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n"
                 : "=r"(old) : "l"(bar) : "memory");
    const unsigned int target = (old & ~63u) + 64u;
    if (old + 1u != target) {
      // 64 cluster leaders only: a bare acquire spin beats any backoff here,
      // the arrivals are spread over far less than one nanosleep quantum.
      unsigned int observed;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n"
                     : "=r"(observed) : "l"(bar) : "memory");
      } while (static_cast<int32_t>(observed - target) < 0);
    }
  }
  cl.sync();
  if (threadIdx.x == 0 && cl.block_rank() != 0) {
    // One acquire load per CTA (not a fence): gives the other seven CTAs of
    // the cluster the device-scope acquire that only rank 0 performed, at the
    // cost of a single L2 read rather than a membar.
    unsigned int dummy;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n"
                 : "=r"(dummy) : "l"(bar) : "memory");
  }
  __syncthreads();
}

__device__ __forceinline__ void t1_grid_barrier(unsigned int *bar,
                                                unsigned int blocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n"
                 : "=r"(old) : "l"(bar) : "memory");
    const unsigned int target = (old & ~511u) + 512u;
    if (old + 1u != target) {
      unsigned int observed;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n"
                     : "=r"(observed) : "l"(bar) : "memory");
        if (static_cast<int32_t>(observed - target) < 0) {
          const unsigned int gap = target - observed;
          __nanosleep(gap > 96u ? 1024u : 48u);
        }
      } while (static_cast<int32_t>(observed - target) < 0);
    }
  }
  __syncthreads();
}

#define T1_XLO(u) __int_as_float((u) << 16)
#define T1_XHI(u) __int_as_float((u) & 0xFFFF0000u)

__device__ __forceinline__ float t1_dot16_packed(
    uint4 w, const uint4 &a, const uint4 &b) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  s0 = fmaf(f8(w.x << 24), T1_XLO(a.x), s0);
  s1 = fmaf(f8(w.x << 16), T1_XHI(a.x), s1);
  s2 = fmaf(f8(w.x << 8), T1_XLO(a.y), s2);
  s3 = fmaf(f8(w.x), T1_XHI(a.y), s3);
  s0 = fmaf(f8(w.y << 24), T1_XLO(a.z), s0);
  s1 = fmaf(f8(w.y << 16), T1_XHI(a.z), s1);
  s2 = fmaf(f8(w.y << 8), T1_XLO(a.w), s2);
  s3 = fmaf(f8(w.y), T1_XHI(a.w), s3);
  s0 = fmaf(f8(w.z << 24), T1_XLO(b.x), s0);
  s1 = fmaf(f8(w.z << 16), T1_XHI(b.x), s1);
  s2 = fmaf(f8(w.z << 8), T1_XLO(b.y), s2);
  s3 = fmaf(f8(w.z), T1_XHI(b.y), s3);
  s0 = fmaf(f8(w.w << 24), T1_XLO(b.z), s0);
  s1 = fmaf(f8(w.w << 16), T1_XHI(b.z), s1);
  s2 = fmaf(f8(w.w << 8), T1_XLO(b.w), s2);
  s3 = fmaf(f8(w.w), T1_XHI(b.w), s3);
  return (s0 + s1) + (s2 + s3);
}

// T=1-only native FP8 widening.  Unlike f8(), cvt returns the true E4M3
// value, so callers use the checkpoint block scale directly (without P120F).
__device__ __forceinline__ float t1_dot16_native(uint4 w, const X16 &x) {
  const uint32_t *wp = reinterpret_cast<const uint32_t *>(&w);
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t h01, h23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01)
        : "h"(static_cast<unsigned short>(wp[j])));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23)
        : "h"(static_cast<unsigned short>(wp[j] >> 16)));
    const float2 f01 =
        __half22float2(*reinterpret_cast<const __half2 *>(&h01));
    const float2 f23 =
        __half22float2(*reinterpret_cast<const __half2 *>(&h23));
    s0 = fmaf(f01.x, x.v[4 * j], s0);
    s1 = fmaf(f01.y, x.v[4 * j + 1], s1);
    s2 = fmaf(f23.x, x.v[4 * j + 2], s2);
    s3 = fmaf(f23.y, x.v[4 * j + 3], s3);
  }
  return (s0 + s1) + (s2 + s3);
}

__device__ __forceinline__ float2 t1_dot16_native_pair(
    uint4 wg, uint4 wu, const X16 &x) {
  const uint32_t *gp = reinterpret_cast<const uint32_t *>(&wg);
  const uint32_t *up = reinterpret_cast<const uint32_t *>(&wu);
  float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
  float u0 = 0.f, u1 = 0.f, u2 = 0.f, u3 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t gh01, gh23, uh01, uh23;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(gh01)
        : "h"(static_cast<unsigned short>(gp[j])));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(gh23)
        : "h"(static_cast<unsigned short>(gp[j] >> 16)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(uh01)
        : "h"(static_cast<unsigned short>(up[j])));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(uh23)
        : "h"(static_cast<unsigned short>(up[j] >> 16)));
    const float2 gf01 =
        __half22float2(*reinterpret_cast<const __half2 *>(&gh01));
    const float2 gf23 =
        __half22float2(*reinterpret_cast<const __half2 *>(&gh23));
    const float2 uf01 =
        __half22float2(*reinterpret_cast<const __half2 *>(&uh01));
    const float2 uf23 =
        __half22float2(*reinterpret_cast<const __half2 *>(&uh23));
    g0 = fmaf(gf01.x, x.v[4 * j], g0);
    g1 = fmaf(gf01.y, x.v[4 * j + 1], g1);
    g2 = fmaf(gf23.x, x.v[4 * j + 2], g2);
    g3 = fmaf(gf23.y, x.v[4 * j + 3], g3);
    u0 = fmaf(uf01.x, x.v[4 * j], u0);
    u1 = fmaf(uf01.y, x.v[4 * j + 1], u1);
    u2 = fmaf(uf23.x, x.v[4 * j + 2], u2);
    u3 = fmaf(uf23.y, x.v[4 * j + 3], u3);
  }
  return make_float2((g0 + g1) + (g2 + g3),
                     (u0 + u1) + (u2 + u3));
}

// PF: 0 = load FC2 weights after the grid barrier, 1 = cp.async to shared,
// 2 = hold FC2 weights in registers across the barrier.
template <int NR, int PF, bool CLU = false>
__global__ __launch_bounds__(256, 12 / NR) void t1_v3_kernel(
    const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w13,
    const float *__restrict__ w13s, const uint8_t *__restrict__ w2,
    const float *__restrict__ w2s, const int *__restrict__ ids,
    const float *__restrict__ tws, __nv_bfloat16 *__restrict__ inter,
    __nv_bfloat16 *__restrict__ out, unsigned int *__restrict__ bar) {
  __shared__ uint8_t s_w2[PF == 1 ? TOPK * NR * 512 : 16];
  __shared__ __align__(16) uint32_t s_x[1024];
  __shared__ float red[4 * NR][4];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int lsub = lane >> 3;

  const int route = blockIdx.x & (TOPK - 1);
  const int rt = blockIdx.x >> 3;
  const int wj = warp & 3;
  const int wr = warp >> 2;
  const int i0 = rt * (2 * NR) + wr * NR;

  // Resolve the route first, then put the tiny activation slice in flight
  // before the id-dependent weight stream fills the LSU queue.
  const int rid1 = ids[route];
  const int e1 = rid1 & 255;
  const float *ws1 = w13s + (size_t)e1 * 128;
  const int rbg = i0 >> 7;
  const float sg = ws1[rbg * 16 + 4 * wj + lsub];
  const float su = ws1[(rbg + 4) * 16 + 4 * wj + lsub] *
                   ((unsigned)rid1 < (unsigned)NEXP ? 1.f : 0.f);
  const uint4 *xp =
      reinterpret_cast<const uint4 *>(x + wj * 512 + lane * 16);
  const uint4 xa = xp[0], xb = xp[1];
  const uint8_t *wb1 =
      w13 + ((size_t)e1 * TWO_I + i0) * HID + wj * 512 + lane * 16;
  uint4 wg[NR], wu[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r)
    wg[r] = *reinterpret_cast<const uint4 *>(wb1 + (size_t)r * HID);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    wu[r] = *reinterpret_cast<const uint4 *>(
        wb1 + (size_t)(INT_ + r) * HID);

  X16 xv;
  unpack_bf16x16(xa, xb, xv);

  const int h0 = blockIdx.x * NR;
  int e2 = ids[warp];
  asm("and.b32 %0, %0, 255;" : "+r"(e2));
  const uint8_t *wb2 = w2 + ((size_t)e2 * HID + h0) * INT_ + lane * 16;
  uint4 w2r[NR];
  if (PF == 1) {
    const uint32_t sb =
        __cvta_generic_to_shared(&s_w2[(warp * NR) * 512 + lane * 16]);
#pragma unroll
    for (int r = 0; r < NR; ++r)
      t1_cp_async16(sb + r * 512, wb2 + (size_t)r * INT_);
    asm volatile("cp.async.commit_group;\n" ::);
  } else if (PF == 2) {
#pragma unroll
    for (int r = 0; r < NR; ++r)
      w2r[r] = *reinterpret_cast<const uint4 *>(wb2 + (size_t)r * INT_);
  }

#pragma unroll
  for (int r = 0; r < NR; ++r) {
    const float2 gu = t1_dot16_native_pair(wg[r], wu[r], xv);
    const float g = warp_sum(gu.x * sg);
    const float u = warp_sum(gu.y * su);
    if (lane == 0) {
      red[(wr * NR + r) * 2][wj] = g;
      red[(wr * NR + r) * 2 + 1][wj] = u;
    }
  }
  __syncthreads();
  const int k = warp;
  const float sc =
      w2s[(size_t)e2 * 64 + (h0 >> 7) * 4 + lsub] * tws[k];
  if (tid < 2 * NR) {
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      g += red[tid * 2][c];
      u += red[tid * 2 + 1][c];
    }
    const float s = g / (1.f + expf(-g));
    const int i = rt * (2 * NR) + tid;
    const int pos = ((i >> 4) * 8) + (i & 7) + (((i >> 3) & 1) << 8);
    inter[(size_t)route * INT_ + pos] = __float2bfloat16(s * u);
  }

  if constexpr (CLU) t1_cluster_barrier(bar);
  else t1_grid_barrier(bar, gridDim.x);

  const uint4 *ip = reinterpret_cast<const uint4 *>(inter + (size_t)k * INT_);
  X16 iv;
  unpack_bf16x16(ip[lane], ip[lane + 32], iv);
  if (PF == 1) {
    asm volatile("cp.async.wait_all;\n" ::);
#pragma unroll
    for (int r = 0; r < NR; ++r)
      w2r[r] = *reinterpret_cast<const uint4 *>(
          &s_w2[(k * NR + r) * 512 + lane * 16]);
  } else if (PF == 0) {
#pragma unroll
    for (int r = 0; r < NR; ++r)
      w2r[r] = *reinterpret_cast<const uint4 *>(wb2 + (size_t)r * INT_);
  }
#pragma unroll
  for (int r = 0; r < NR; r += 2) {
    const float2 ab = t1_dot16_native_pair(w2r[r], w2r[r + 1], iv);
    const float a0 = warp_sum(ab.x * sc);
    const float a1 = warp_sum(ab.y * sc);
    if (lane == 0) {
      red[k][r] = a0;
      red[k][r + 1] = a1;
    }
  }
  __syncthreads();
  if (tid < NR) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < TOPK; ++c) s += red[c][tid];
    out[h0 + tid] = __float2bfloat16(s);
  }
}

// Join the two T=1 GEMV phases with a cooperative grid barrier, preserving
// the established arithmetic and work mapping while removing one launch.
__global__ __launch_bounds__(256) void t1_fused_kernel(
    const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w13,
    const float *__restrict__ w13s, const uint8_t *__restrict__ w2,
    const float *__restrict__ w2s, const int *__restrict__ ids,
    const float *__restrict__ tws, __nv_bfloat16 *__restrict__ inter,
    __nv_bfloat16 *__restrict__ out) {
  cg::grid_group grid = cg::this_grid();
  // A five-float pitch keeps the eight FC1 reduction rows on distinct banks.
  __shared__ float phase_red[16][5];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int lsub = lane >> 3;

  const int route = blockIdx.x & (TOPK - 1);
  const int row_tile = blockIdx.x >> 3;
  const int e1 = ids[route] & 255;
  const int wj = warp & 3;
  const int wr = warp >> 2;
  const int i0 = (row_tile * 2 + wr) * 4;
  const int rbg = i0 >> 7;
  const float *ws1 = w13s + (size_t)e1 * 128;
  const float sg = ws1[rbg * 16 + 4 * wj + lsub] * P120F;
  const float su = ws1[(rbg + 4) * 16 + 4 * wj + lsub] * P120F;
  const uint4 *xp =
      reinterpret_cast<const uint4 *>(x + wj * 512 + lane * 16);
  X16 xv;
  unpack_bf16x16(xp[0], xp[1], xv);
  const uint8_t *wb1 =
      w13 + ((size_t)e1 * TWO_I + i0) * HID + wj * 512 + lane * 16;
  uint4 wg[4], wu[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    wg[r] = *reinterpret_cast<const uint4 *>(wb1 + (size_t)r * HID);
    wu[r] =
        *reinterpret_cast<const uint4 *>(wb1 + (size_t)(r + INT_) * HID);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = warp_sum(dot16(wg[r], xv) * sg);
    const float u = warp_sum(dot16(wu[r], xv) * su);
    if (lane == 0) {
      phase_red[(wr * 4 + r) * 2][wj] = g;
      phase_red[(wr * 4 + r) * 2 + 1][wj] = u;
    }
  }
  __syncthreads();
  if (tid < 8) {
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      g += phase_red[tid * 2][c];
      u += phase_red[tid * 2 + 1][c];
    }
    const float s = g / (1.f + expf(-g));
    const int pq = (row_tile >> 1) | ((row_tile & 1) << 5);
    inter[(size_t)route * INT_ + pq * 8 + tid] =
        __float2bfloat16(s * u);
  }

  grid.sync();

  const int k = warp;
  const int h0 = blockIdx.x * 4;
  const int rid2 = ids[k];
  const int e2 = rid2 & 255;
  const float sc =
      w2s[(size_t)e2 * 64 + (h0 >> 7) * 4 + lsub] * P120F *
      ((unsigned)rid2 < (unsigned)NEXP ? tws[k] : 0.f);
  const uint4 *ip =
      reinterpret_cast<const uint4 *>(inter + (size_t)k * INT_);
  const uint8_t *wb2 =
      w2 + ((size_t)e2 * HID + h0) * INT_ + lane * 16;
  uint4 wr2[4];
#pragma unroll
  for (int r = 0; r < 4; ++r)
    wr2[r] = *reinterpret_cast<const uint4 *>(wb2 + (size_t)r * INT_);
  const uint4 ia = ip[lane];
  const uint4 ib = ip[lane + 32];
  X16 iv;
  unpack_bf16x16(ia, ib, iv);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float a = warp_sum(dot16(wr2[r], iv) * sc);
    if (lane == 0) phase_red[k][r] = a;
  }
  __syncthreads();
  if (tid < 4) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < TOPK; ++c) s += phase_red[c][tid];
    out[h0 + tid] = __float2bfloat16(s);
  }
}


// ================================================================ tiled GEMM path (large T)
// BM=128 tokens x BN=128 columns, BK=32, 256 threads (16x16), 8x8 micro tile.
#define BK_ 32

__device__ __forceinline__ void unpack8_bf(uint4 v, float *o) {
  o[0] = __int_as_float(v.x << 16); o[1] = __int_as_float(v.x & 0xFFFF0000u);
  o[2] = __int_as_float(v.y << 16); o[3] = __int_as_float(v.y & 0xFFFF0000u);
  o[4] = __int_as_float(v.z << 16); o[5] = __int_as_float(v.z & 0xFFFF0000u);
  o[6] = __int_as_float(v.w << 16); o[7] = __int_as_float(v.w & 0xFFFF0000u);
}
__device__ __forceinline__ uint32_t bf_at(const uint32_t *a, int s) {
  return (s & 1) ? (a[s >> 1] >> 16) : (a[s >> 1] & 0xFFFFu);
}

// ---- FC1 : inter = silu(x@Wg^T) * (x@Wu^T)
__global__ __launch_bounds__(256, 2) void gemm_fc1(
    const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w13,
    const float *__restrict__ w13s, const int *__restrict__ tile_e,
    const int *__restrict__ tile_p0, const int *__restrict__ tile_cnt,
    const int *__restrict__ pair_idx, const int *__restrict__ ntiles,
    __nv_bfloat16 *__restrict__ inter) {
  __shared__ uint32_t As[BK_][64];
  __shared__ float Bs[BK_][128];
  const int tile = blockIdx.x;
  if (tile >= ntiles[0]) return;
  const int e = tile_e[tile], cnt = tile_cnt[tile], p0 = tile_p0[tile];
  const int n0 = blockIdx.y * 64;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

  const int am = (tid & 63) * 2, akk = tid >> 6;           // A loader: rows am,am+1 ; k+akk*8
  const int bn = tid & 127, bkk = tid >> 7;                 // B loader: col bn ; k+bkk*16
  const int brow = (bn < 64) ? (n0 + bn) : (n0 + 512 + bn - 64);
  const uint8_t *bptr = w13 + ((size_t)e * TWO_I + brow) * HID;
  const float *wsb = w13s + (size_t)e * 128 + ((bn < 64) ? (n0 >> 7) : ((n0 + 512) >> 7)) * 16;

  long long xo0 = -1, xo1 = -1;
  if (am < cnt) xo0 = (long long)(pair_idx[p0 + am] >> 3) * HID;
  if (am + 1 < cnt) xo1 = (long long)(pair_idx[p0 + am + 1] >> 3) * HID;

  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

  uint4 a0, a1, bv; float bsc;
  const int NIT = HID / BK_;
  {
    a0 = (xo0 >= 0) ? *(const uint4 *)(x + xo0 + akk * 8) : make_uint4(0, 0, 0, 0);
    a1 = (xo1 >= 0) ? *(const uint4 *)(x + xo1 + akk * 8) : make_uint4(0, 0, 0, 0);
    bv = ldcg4(bptr + bkk * 16);
    bsc = wsb[0] * P120F;
  }
#pragma unroll 1
  for (int it = 0; it < NIT; it++) {
    {
      const uint32_t *pa0 = (const uint32_t *)&a0, *pa1 = (const uint32_t *)&a1;
#pragma unroll
      for (int s = 0; s < 8; s++)
        As[akk * 8 + s][tid & 63] = (bf_at(pa1, s) << 16) | bf_at(pa0, s);
      const uint32_t *pb = (const uint32_t *)&bv;
#pragma unroll
      for (int s = 0; s < 16; s++)
        Bs[bkk * 16 + s][bn] = f8(pb[s >> 2] << (24 - 8 * (s & 3))) * bsc;
    }
    __syncthreads();
    if (it + 1 < NIT) {
      const int k1 = (it + 1) * BK_;
      a0 = (xo0 >= 0) ? *(const uint4 *)(x + xo0 + k1 + akk * 8) : make_uint4(0, 0, 0, 0);
      a1 = (xo1 >= 0) ? *(const uint4 *)(x + xo1 + k1 + akk * 8) : make_uint4(0, 0, 0, 0);
      bv = ldcg4(bptr + k1 + bkk * 16);
      bsc = wsb[k1 >> 7] * P120F;
    }
#pragma unroll 4
    for (int k = 0; k < BK_; k++) {
      float a[8], b[8];
      unpack8_bf(*(const uint4 *)&As[k][ty * 4], a);
      float4 q0 = *(const float4 *)&Bs[k][tx * 4];
      float4 q1 = *(const float4 *)&Bs[k][64 + tx * 4];
      b[0] = q0.x; b[1] = q0.y; b[2] = q0.z; b[3] = q0.w;
      b[4] = q1.x; b[5] = q1.y; b[6] = q1.z; b[7] = q1.w;
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int m = ty * 8 + i;
    if (m < cnt) {
      __nv_bfloat16 o[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float g = acc[i][j];
        o[j] = __float2bfloat16(g / (1.f + expf(-g)) * acc[i][j + 4]);
      }
      *(uint2 *)(inter + (size_t)(p0 + m) * INT_ + n0 + tx * 4) = *(uint2 *)o;
    }
  }
}

// ---- FC2 : out_f32[token] += tw * (inter @ W2^T)
__global__ __launch_bounds__(256, 2) void gemm_fc2(
    const __nv_bfloat16 *__restrict__ inter, const uint8_t *__restrict__ w2,
    const float *__restrict__ w2s, const float *__restrict__ topk_weights,
    const int *__restrict__ tile_e, const int *__restrict__ tile_p0,
    const int *__restrict__ tile_cnt, const int *__restrict__ pair_idx,
    const int *__restrict__ ntiles, float *__restrict__ outf) {
  __shared__ uint32_t As[BK_][64];
  __shared__ float Bs[BK_][128];
  const int tile = blockIdx.x;
  if (tile >= ntiles[0]) return;
  const int e = tile_e[tile], cnt = tile_cnt[tile], p0 = tile_p0[tile];
  const int h0 = blockIdx.y * 128;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int am = (tid & 63) * 2, akk = tid >> 6;
  const int bn = tid & 127, bkk = tid >> 7;
  const uint8_t *bptr = w2 + ((size_t)e * HID + h0 + bn) * INT_;
  const float *wsb = w2s + (size_t)e * 64 + (h0 >> 7) * 4;
  const __nv_bfloat16 *ia0 = (am < cnt) ? (inter + (size_t)(p0 + am) * INT_) : nullptr;
  const __nv_bfloat16 *ia1 = (am + 1 < cnt) ? (inter + (size_t)(p0 + am + 1) * INT_) : nullptr;

  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

  uint4 a0, a1, bv; float bsc;
  const int NIT = INT_ / BK_;
  a0 = ia0 ? *(const uint4 *)(ia0 + akk * 8) : make_uint4(0, 0, 0, 0);
  a1 = ia1 ? *(const uint4 *)(ia1 + akk * 8) : make_uint4(0, 0, 0, 0);
  bv = ldcg4(bptr + bkk * 16);
  bsc = wsb[0] * P120F;
#pragma unroll 1
  for (int it = 0; it < NIT; it++) {
    {
      const uint32_t *pa0 = (const uint32_t *)&a0, *pa1 = (const uint32_t *)&a1;
#pragma unroll
      for (int s = 0; s < 8; s++)
        As[akk * 8 + s][tid & 63] = (bf_at(pa1, s) << 16) | bf_at(pa0, s);
      const uint32_t *pb = (const uint32_t *)&bv;
#pragma unroll
      for (int s = 0; s < 16; s++)
        Bs[bkk * 16 + s][bn] = f8(pb[s >> 2] << (24 - 8 * (s & 3))) * bsc;
    }
    __syncthreads();
    if (it + 1 < NIT) {
      const int k1 = (it + 1) * BK_;
      a0 = ia0 ? *(const uint4 *)(ia0 + k1 + akk * 8) : make_uint4(0, 0, 0, 0);
      a1 = ia1 ? *(const uint4 *)(ia1 + k1 + akk * 8) : make_uint4(0, 0, 0, 0);
      bv = ldcg4(bptr + k1 + bkk * 16);
      bsc = wsb[k1 >> 7] * P120F;
    }
#pragma unroll 4
    for (int k = 0; k < BK_; k++) {
      float a[8], b[8];
      unpack8_bf(*(const uint4 *)&As[k][ty * 4], a);
      float4 q0 = *(const float4 *)&Bs[k][tx * 4];
      float4 q1 = *(const float4 *)&Bs[k][64 + tx * 4];
      b[0] = q0.x; b[1] = q0.y; b[2] = q0.z; b[3] = q0.w;
      b[4] = q1.x; b[5] = q1.y; b[6] = q1.z; b[7] = q1.w;
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int m = ty * 8 + i;
    if (m < cnt) {
      int pr = pair_idx[p0 + m];
      float tw = topk_weights[pr];
      float *dst = outf + (size_t)(pr >> 3) * HID + h0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        atomicAdd(dst + tx * 4 + j, acc[i][j] * tw);
        atomicAdd(dst + 64 + tx * 4 + j, acc[i][j + 4] * tw);
      }
    }
  }
}

__global__ __launch_bounds__(256) void f32_to_bf16(const float *__restrict__ src,
                                                   __nv_bfloat16 *__restrict__ dst, long long n) {
  long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  long long stride = (long long)gridDim.x * 256;
  for (; i < n; i += stride) dst[i] = __float2bfloat16(src[i]);
}

// ---------------------------------------------------------------- workspace
namespace {
struct Workspace {
  void *ptr = nullptr;
  size_t cap = 0;
  void ensure(size_t bytes) {
    if (bytes <= cap) return;
    if (ptr) cudaFree(ptr);
    cudaMalloc(&ptr, bytes);
    cudaMemset(ptr, 0, 256);
    cap = bytes;
  }
};
Workspace g_ws;
int g_t1_v3_ok = -1;
int g_t1_clu = -1;
inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
}  // namespace

void RunGemv(ffi::TensorView hidden_states, ffi::TensorView w13,
             ffi::TensorView w13_scale, ffi::TensorView w2,
             ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out,
             cudaStream_t stream) {
  const int T = static_cast<int>(hidden_states.shape()[0]);
  const int P = T * TOPK;

  const int MT = (T == 1) ? 1 : ((T <= 8) ? 2 : ((T >= 1024) ? 128 : 4));
  long long nt_bound = (P <= NEXP)
                           ? P
                           : NEXP + (long long)(P - NEXP) / MT;

  size_t off = 256;  // persistent counter used only by the T=1 phase barrier
  size_t o_pair = off;  off = align_up(off + (size_t)P * 4, 256);
  size_t o_pos  = off;  off = align_up(off + (size_t)P * 4, 256);
  size_t o_te   = off;  off = align_up(off + (size_t)nt_bound * 4, 256);
  size_t o_tp   = off;  off = align_up(off + (size_t)nt_bound * 4, 256);
  size_t o_tc   = off;  off = align_up(off + (size_t)nt_bound * 4, 256);
  size_t o_nt   = off;  off = align_up(off + 4, 256);
  size_t o_int  = off;  off = align_up(off + (size_t)P * INT_ * 2, 256);
  const bool BIG = (T >= 1024);
  size_t o_of   = off;  if (BIG) off = align_up(off + (size_t)T * HID * 4, 256);
  g_ws.ensure(off);
  char *base = (char *)g_ws.ptr;
  int *pair_idx = (int *)(base + o_pair);
  int *pos_of   = (int *)(base + o_pos);
  int *tile_e   = (int *)(base + o_te);
  int *tile_p0  = (int *)(base + o_tp);
  int *tile_cnt = (int *)(base + o_tc);
  int *ntiles   = (int *)(base + o_nt);
  __nv_bfloat16 *inter = (__nv_bfloat16 *)(base + o_int);
  float *outf = (float *)(base + o_of);

  const int *ids = (const int *)topk_ids.data_ptr();
  const float *tws = (const float *)topk_weights.data_ptr();
  const __nv_bfloat16 *x = (const __nv_bfloat16 *)hidden_states.data_ptr();
  const uint8_t *pw13 = (const uint8_t *)w13.data_ptr();
  const uint8_t *pw2 = (const uint8_t *)w2.data_ptr();
  const float *pw13s = (const float *)w13_scale.data_ptr();
  const float *pw2s = (const float *)w2_scale.data_ptr();
  __nv_bfloat16 *pout = (__nv_bfloat16 *)out.data_ptr();

  if (T == 1) {
    if (g_t1_v3_ok < 0) {
      int coop = 0, sms = 0, occ = 0;
      const int dev = hidden_states.device().device_id;
      cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occ, t1_v3_kernel<T1NR, T1PF>, 256, 0);
      g_t1_v3_ok = coop && occ * sms >= (2048 / T1NR);
    }
    if (g_t1_v3_ok) {
      unsigned int *bar = reinterpret_cast<unsigned int *>(base);
      if (g_t1_clu < 0) {
        int cl = 0;
        cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch,
                               hidden_states.device().device_id);
        int mac = 0;
        if (cl) {
          cudaLaunchConfig_t probe = {};
          probe.gridDim = dim3(2048 / T1NR);
          probe.blockDim = dim3(256);
          cudaLaunchAttribute pa[1];
          pa[0].id = cudaLaunchAttributeClusterDimension;
          pa[0].val.clusterDim.x = T1CLU;
          pa[0].val.clusterDim.y = 1;
          pa[0].val.clusterDim.z = 1;
          probe.attrs = pa;
          probe.numAttrs = 1;
          cudaOccupancyMaxActiveClusters(
              &mac, reinterpret_cast<const void *>(
                        t1_v3_kernel<T1NR, T1PF, true>),
              &probe);
        }
        g_t1_clu = (cl && mac >= (2048 / T1NR) / T1CLU) ? 1 : 0;
      }
      if (g_t1_clu) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(2048 / T1NR);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attrs[3];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = T1CLU;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        attrs[1].id = cudaLaunchAttributeCooperative;
        attrs[1].val.cooperative = 1;
        attrs[2].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[2].val.clusterSchedulingPolicyPreference =
            cudaClusterSchedulingPolicySpread;
        cfg.attrs = attrs;
        cfg.numAttrs = 3;
        if (cudaLaunchKernelEx(&cfg, t1_v3_kernel<T1NR, T1PF, true>, x, pw13,
                               pw13s, pw2, pw2s, ids, tws, inter, pout,
                               bar) == cudaSuccess)
          return;
        g_t1_clu = 0;
      }
      // Keep the 512-arrival fallback on its own epoch counter: a prior
      // successful cluster launch advances the cluster counter by only 64.
      unsigned int *flat_bar = bar + 1;
      void *args[] = {&x, &pw13, &pw13s, &pw2, &pw2s,
                      &ids, &tws, &inter, &pout, &flat_bar};
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void *>(t1_v3_kernel<T1NR, T1PF>),
          dim3(2048 / T1NR), dim3(256), args, 0, stream);
      return;
    }
    void *args[] = {&x, &pw13, &pw13s, &pw2, &pw2s,
                    &ids, &tws, &inter, &pout};
    cudaLaunchCooperativeKernel(reinterpret_cast<void *>(t1_fused_kernel),
                                dim3(512), dim3(256), args, 0, stream);
    return;
  }

  prep_kernel<<<1, 256, 0, stream>>>(ids, P, MT, pair_idx, pos_of, tile_e, tile_p0,
                                     tile_cnt, ntiles);
  if (BIG) {
    cudaMemsetAsync(outf, 0, (size_t)T * HID * 4, stream);
    dim3 gg((unsigned)nt_bound, INT_ / 64);
    gemm_fc1<<<gg, 256, 0, stream>>>(x, pw13, pw13s, tile_e, tile_p0, tile_cnt,
                                     pair_idx, ntiles, inter);
    dim3 gh((unsigned)nt_bound, HID / 128);
    gemm_fc2<<<gh, 256, 0, stream>>>(inter, pw2, pw2s, tws, tile_e, tile_p0, tile_cnt,
                                     pair_idx, ntiles, outf);
    long long n = (long long)T * HID;
    int nb = (int)((n + 255) / 256); if (nb > 8192) nb = 8192;
    f32_to_bf16<<<nb, 256, 0, stream>>>(outf, pout, n);
    return;
  }
  if (MT == 2) {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<2, NR, RW, false, false><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  } else if (T == 64) {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<4, NR, RW, false, true, true><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  } else {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<4, NR, RW, false, false><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  }
  if (T == 64) {
    dim3 g2((unsigned)T, HID / 8);
    fc2_kernel<8, false, true><<<g2, 32 * TOPK, 0, stream>>>(
        inter, pw2, pw2s, ids, tws, pos_of, pout);
  } else {
    dim3 g2((unsigned)T, HID / 4);
    fc2_kernel<4, false><<<g2, 32 * TOPK, 0, stream>>>(
        inter, pw2, pw2s, ids, tws, pos_of, pout);
  }
}












