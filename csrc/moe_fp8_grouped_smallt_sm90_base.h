#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FULLM 0xffffffffu

__device__ __forceinline__ void unpack4(uint32_t v, __half2& a, __half2& b) {
  __half2_raw ra = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  __half2_raw rb = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v >> 16), __NV_E4M3);
  a = ra;
  b = rb;
}

// The small-T fc1 kernels accumulate each 16-element chunk in __half (8
// products per half lane) against RAW e4m3 weight mantissas — which the
// [128,128] block quantizer drives to ~448 at every block's peak — folding
// the fp32 block scale only once per chunk. Staging the bf16 activation
// row prescaled by 1/16 keeps those chunk partials inside half range for
// activation magnitudes up to ~2^9 instead of ~2^5 (outlier channels of
// O(10..100) occur in real post-norm hidden states). The exact power of
// two is restored in fp32 (FC1_XS_UNSCALE on accU/accG) before the
// nonlinearity, so in-range results are unchanged up to half rounding.
#define FC1_XS_PRESCALE 0.0625f
#define FC1_XS_UNSCALE 16.f

__device__ __forceinline__ __half2 bf2_to_h2_prescaled(__nv_bfloat162 v) {
  const float2 f = __bfloat1622float2(v);
  return __floats2half2_rn(f.x * FC1_XS_PRESCALE, f.y * FC1_XS_PRESCALE);
}

// The small-T fc2 inner products accumulate RAW e4m3 mantissas (the
// [128,128] quantizer drives every block's peak to ~448) against the fp16
// fc1 intermediate, whose magnitude is bounded only by half range: one
// 8-product __hfma2 chain can reach 8 * 448 * 65504 and overflow to inf
// long before the fp32 block scale is applied. Prescaling the unpacked
// weights by 2^-14 — exact for EVERY e4m3 value (4 significant bits, the
// multiply is a pure exponent shift, subnormal halves included) — bounds
// the chain by 8 * 448 * 2^-14 * 65504 = 14336, safely inside half range;
// the exact 2^14 is restored in fp32 where the block scale is applied
// (FC2_W_UNSCALE folded into the scale), so in-range results keep the
// original rounding behavior.
#define FC2_W_PRESCALE 6.103515625e-5f /* 2^-14 */
#define FC2_W_UNSCALE 16384.f

// ------------------------------------------------------------------
// fc1 for small T: one warp per intermediate column j, one block row
// per (token,slot) pair.  Weight-streaming GEMV, memory bound.
// ------------------------------------------------------------------
extern "C" __global__ void fc1_small(const __nv_bfloat16* __restrict__ X,
                                     const uint8_t* __restrict__ W13,
                                     const float* __restrict__ W13S,
                                     const int* __restrict__ topk_ids, __half* __restrict__ Hb,
                                     int H, int I, int Sh, int Sn1, int KTOP) {
  extern __shared__ __half xs[];
  const int tid = threadIdx.x;
  const int s = blockIdx.x;
  const int t = s / KTOP;
  const int e = topk_ids[s];

  const uint4* xg = (const uint4*)(X + (size_t)t * H);
  uint4* xd = (uint4*)xs;
  for (int i = tid; i < (H >> 3); i += blockDim.x) {
    uint4 v = xg[i];
    const __nv_bfloat162* bp = (const __nv_bfloat162*)&v;
    __half2 h0 = bf2_to_h2_prescaled(bp[0]);
    __half2 h1 = bf2_to_h2_prescaled(bp[1]);
    __half2 h2 = bf2_to_h2_prescaled(bp[2]);
    __half2 h3 = bf2_to_h2_prescaled(bp[3]);
    uint4 o;
    o.x = *(const unsigned int*)&h0;
    o.y = *(const unsigned int*)&h1;
    o.z = *(const unsigned int*)&h2;
    o.w = *(const unsigned int*)&h3;
    xd[i] = o;
  }
  float* ss = (float*)(xs + H);
  for (int i = tid; i < Sn1 * Sh; i += blockDim.x) ss[i] = W13S[(size_t)e * Sn1 * Sh + i];
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int warps = blockDim.x >> 5;
  for (int j = blockIdx.y * warps + warp; j < I; j += gridDim.y * warps) {
    const uint8_t* Wu = W13 + ((size_t)e * (2 * I) + j) * H;
    const uint8_t* Wg = W13 + ((size_t)e * (2 * I) + I + j) * H;
    const float* Su = ss + (j >> 7) * Sh;
    const float* Sg = ss + ((I + j) >> 7) * Sh;

    float accU = 0.f, accG = 0.f;
#pragma unroll 2
    for (int k0 = lane * 16; k0 < H; k0 += 512) {
      uint4 wu = *(const uint4*)(Wu + k0);
      uint4 wg = *(const uint4*)(Wg + k0);
      const __half2* xp = (const __half2*)(xs + k0);
      __half2 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
      __half2 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
      __half2 zero = __float2half2_rn(0.f);
      __half2 lu = zero, lg = zero, a, b;
      unpack4(wu.x, a, b);
      lu = __hfma2(a, x0, lu);
      lu = __hfma2(b, x1, lu);
      unpack4(wu.y, a, b);
      lu = __hfma2(a, x2, lu);
      lu = __hfma2(b, x3, lu);
      unpack4(wu.z, a, b);
      lu = __hfma2(a, x4, lu);
      lu = __hfma2(b, x5, lu);
      unpack4(wu.w, a, b);
      lu = __hfma2(a, x6, lu);
      lu = __hfma2(b, x7, lu);
      unpack4(wg.x, a, b);
      lg = __hfma2(a, x0, lg);
      lg = __hfma2(b, x1, lg);
      unpack4(wg.y, a, b);
      lg = __hfma2(a, x2, lg);
      lg = __hfma2(b, x3, lg);
      unpack4(wg.z, a, b);
      lg = __hfma2(a, x4, lg);
      lg = __hfma2(b, x5, lg);
      unpack4(wg.w, a, b);
      lg = __hfma2(a, x6, lg);
      lg = __hfma2(b, x7, lg);
      const int kb = k0 >> 7;
      float2 fu = __half22float2(lu), fg = __half22float2(lg);
      accU = fmaf(Su[kb], fu.x + fu.y, accU);
      accG = fmaf(Sg[kb], fg.x + fg.y, accG);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      accU += __shfl_xor_sync(FULLM, accU, o);
      accG += __shfl_xor_sync(FULLM, accG, o);
    }
    if (lane == 0) {
      const float g = accG * FC1_XS_UNSCALE;
      const float u = accU * FC1_XS_UNSCALE;
      Hb[(size_t)s * I + j] = __float2half_rn(((g / (1.f + __expf(-g))) * u) * (1.f / 256.f));
    }
  }
}

// ------------------------------------------------------------------
template <int UNR, int NT = 384, int MINB = 2>
__global__ __launch_bounds__(NT, MINB) void fc1_t1_u4(const __nv_bfloat16* __restrict__ X,
                                                      const uint8_t* __restrict__ W13,
                                                      const float* __restrict__ W13S,
                                                      const int* __restrict__ topk_ids,
                                                      __half* __restrict__ Hb, int H, int I, int Sh,
                                                      int Sn1, int KTOP) {
  extern __shared__ __half xs[];
  const int tid = threadIdx.x;
  const int s = blockIdx.x;
  const int t = s / KTOP;
  const int e = topk_ids[s];

  const uint4* xg = (const uint4*)(X + (size_t)t * H);
  __half2* xd = (__half2*)xs;
  for (int i = tid; i < (H >> 3); i += blockDim.x) {
    uint4 v = xg[i];
    const __nv_bfloat162* bp = (const __nv_bfloat162*)&v;
    __half2 h0 = bf2_to_h2_prescaled(bp[0]);
    __half2 h1 = bf2_to_h2_prescaled(bp[1]);
    __half2 h2 = bf2_to_h2_prescaled(bp[2]);
    __half2 h3 = bf2_to_h2_prescaled(bp[3]);
    const int k = i << 3;
    const int chunk = k >> 9;
    const int owner = (k >> 4) & 31;
    const int pair = (k >> 1) & 7;
    __half2* xdst = xd + chunk * 256 + pair * 32 + owner;
    xdst[0 * 32] = h0;
    xdst[1 * 32] = h1;
    xdst[2 * 32] = h2;
    xdst[3 * 32] = h3;
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int j = blockIdx.y * (blockDim.x >> 5) + warp;
  if (j < I) {
    const uint8_t* Wu = W13 + ((size_t)e * (2 * I) + j) * H;
    const uint8_t* Wg = W13 + ((size_t)e * (2 * I) + I + j) * H;
    const float* Su = W13S + ((size_t)e * Sn1 + (j >> 7)) * Sh;
    const float* Sg = W13S + ((size_t)e * Sn1 + ((I + j) >> 7)) * Sh;

    const int STEP = 512;
    float accU = 0.f, accG = 0.f;
    int k0 = lane * 16;
    // main pipelined body: UNR independent (up,gate) 16B loads in flight
    for (; k0 + STEP * (UNR - 1) < H; k0 += STEP * UNR) {
      uint4 wu[UNR], wg[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        wu[u] = *(const uint4*)(Wu + k0 + u * STEP);
        wg[u] = *(const uint4*)(Wg + k0 + u * STEP);
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int kk = k0 + u * STEP;
        const __half2* xp = (const __half2*)xs + (kk >> 9) * 256 + lane;
        __half2 x0 = xp[0 * 32], x1 = xp[1 * 32];
        __half2 x2 = xp[2 * 32], x3 = xp[3 * 32];
        __half2 x4 = xp[4 * 32], x5 = xp[5 * 32];
        __half2 x6 = xp[6 * 32], x7 = xp[7 * 32];
        __half2 zero = __float2half2_rn(0.f);
        __half2 lu = zero, lg = zero, a, b;
        unpack4(wu[u].x, a, b);
        lu = __hfma2(a, x0, lu);
        lu = __hfma2(b, x1, lu);
        unpack4(wu[u].y, a, b);
        lu = __hfma2(a, x2, lu);
        lu = __hfma2(b, x3, lu);
        unpack4(wu[u].z, a, b);
        lu = __hfma2(a, x4, lu);
        lu = __hfma2(b, x5, lu);
        unpack4(wu[u].w, a, b);
        lu = __hfma2(a, x6, lu);
        lu = __hfma2(b, x7, lu);
        unpack4(wg[u].x, a, b);
        lg = __hfma2(a, x0, lg);
        lg = __hfma2(b, x1, lg);
        unpack4(wg[u].y, a, b);
        lg = __hfma2(a, x2, lg);
        lg = __hfma2(b, x3, lg);
        unpack4(wg[u].z, a, b);
        lg = __hfma2(a, x4, lg);
        lg = __hfma2(b, x5, lg);
        unpack4(wg[u].w, a, b);
        lg = __hfma2(a, x6, lg);
        lg = __hfma2(b, x7, lg);
        const int kb = kk >> 7;
        float2 fu = __half22float2(lu), fg = __half22float2(lg);
        accU = fmaf(Su[kb], fu.x + fu.y, accU);
        accG = fmaf(Sg[kb], fg.x + fg.y, accG);
      }
    }
    for (; k0 < H; k0 += STEP) {
      uint4 wu = *(const uint4*)(Wu + k0);
      uint4 wg = *(const uint4*)(Wg + k0);
      const __half2* xp = (const __half2*)xs + (k0 >> 9) * 256 + lane;
      __half2 x0 = xp[0 * 32], x1 = xp[1 * 32];
      __half2 x2 = xp[2 * 32], x3 = xp[3 * 32];
      __half2 x4 = xp[4 * 32], x5 = xp[5 * 32];
      __half2 x6 = xp[6 * 32], x7 = xp[7 * 32];
      __half2 zero = __float2half2_rn(0.f);
      __half2 lu = zero, lg = zero, a, b;
      unpack4(wu.x, a, b);
      lu = __hfma2(a, x0, lu);
      lu = __hfma2(b, x1, lu);
      unpack4(wu.y, a, b);
      lu = __hfma2(a, x2, lu);
      lu = __hfma2(b, x3, lu);
      unpack4(wu.z, a, b);
      lu = __hfma2(a, x4, lu);
      lu = __hfma2(b, x5, lu);
      unpack4(wu.w, a, b);
      lu = __hfma2(a, x6, lu);
      lu = __hfma2(b, x7, lu);
      unpack4(wg.x, a, b);
      lg = __hfma2(a, x0, lg);
      lg = __hfma2(b, x1, lg);
      unpack4(wg.y, a, b);
      lg = __hfma2(a, x2, lg);
      lg = __hfma2(b, x3, lg);
      unpack4(wg.z, a, b);
      lg = __hfma2(a, x4, lg);
      lg = __hfma2(b, x5, lg);
      unpack4(wg.w, a, b);
      lg = __hfma2(a, x6, lg);
      lg = __hfma2(b, x7, lg);
      const int kb = k0 >> 7;
      float2 fu = __half22float2(lu), fg = __half22float2(lg);
      accU = fmaf(Su[kb], fu.x + fu.y, accU);
      accG = fmaf(Sg[kb], fg.x + fg.y, accG);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      accU += __shfl_xor_sync(FULLM, accU, o);
      accG += __shfl_xor_sync(FULLM, accG, o);
    }
    if (lane == 0) {
      const float g = accG * FC1_XS_UNSCALE;
      const float u = accU * FC1_XS_UNSCALE;
      Hb[(size_t)s * I + j] = __float2half_rn(((g / (1.f + __expf(-g))) * u) * (1.f / 256.f));
    }
  }
  // every warp's Hb store must be visible before this CTA declares its
  // portion of the dependent (fc2) launch ready.
  __syncthreads();
  if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
}

// fc1 specialised for T==1: a flat (slot,column) work list spread over
// exactly ceil(work/warps) CTAs so every SM gets one CTA and no warp is
// idle.  The default (slot, col-tile) grid is 198 CTAs on 132 SMs, which
// leaves half the SMs idle for half the kernel.
// ------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(576) void fc1_t1(const __nv_bfloat16* __restrict__ X,
                                                         const uint8_t* __restrict__ W13,
                                                         const float* __restrict__ W13S,
                                                         const int* __restrict__ topk_ids,
                                                         __half* __restrict__ Hb, int H, int I,
                                                         int Sh, int Sn1, int total) {
  extern __shared__ __half xs[];
  const int tid = threadIdx.x;
  const uint4* xg = (const uint4*)X;
  uint4* xd = (uint4*)xs;
  for (int i = tid; i < (H >> 3); i += blockDim.x) {
    uint4 v = xg[i];
    const __nv_bfloat162* bp = (const __nv_bfloat162*)&v;
    __half2 h0 = bf2_to_h2_prescaled(bp[0]);
    __half2 h1 = bf2_to_h2_prescaled(bp[1]);
    __half2 h2 = bf2_to_h2_prescaled(bp[2]);
    __half2 h3 = bf2_to_h2_prescaled(bp[3]);
    uint4 o;
    o.x = *(const unsigned int*)&h0;
    o.y = *(const unsigned int*)&h1;
    o.z = *(const unsigned int*)&h2;
    o.w = *(const unsigned int*)&h3;
    xd[i] = o;
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int item = blockIdx.x * (blockDim.x >> 5) + warp;
  if (item >= total) return;
  const int s = item / I;
  const int j = item - s * I;
  const int e = topk_ids[s];

  const uint8_t* Wu = W13 + ((size_t)e * (2 * I) + j) * H;
  const uint8_t* Wg = W13 + ((size_t)e * (2 * I) + I + j) * H;
  const float* Su = W13S + ((size_t)e * Sn1 + (j >> 7)) * Sh;
  const float* Sg = W13S + ((size_t)e * Sn1 + ((I + j) >> 7)) * Sh;

  float accU = 0.f, accG = 0.f;
#pragma unroll 2
  for (int k0 = lane * 16; k0 < H; k0 += 512) {
    uint4 wu = *(const uint4*)(Wu + k0);
    uint4 wg = *(const uint4*)(Wg + k0);
    const __half2* xp = (const __half2*)(xs + k0);
    __half2 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    __half2 x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
    __half2 zero = __float2half2_rn(0.f);
    __half2 lu = zero, lg = zero, a, b;
    unpack4(wu.x, a, b);
    lu = __hfma2(a, x0, lu);
    lu = __hfma2(b, x1, lu);
    unpack4(wu.y, a, b);
    lu = __hfma2(a, x2, lu);
    lu = __hfma2(b, x3, lu);
    unpack4(wu.z, a, b);
    lu = __hfma2(a, x4, lu);
    lu = __hfma2(b, x5, lu);
    unpack4(wu.w, a, b);
    lu = __hfma2(a, x6, lu);
    lu = __hfma2(b, x7, lu);
    unpack4(wg.x, a, b);
    lg = __hfma2(a, x0, lg);
    lg = __hfma2(b, x1, lg);
    unpack4(wg.y, a, b);
    lg = __hfma2(a, x2, lg);
    lg = __hfma2(b, x3, lg);
    unpack4(wg.z, a, b);
    lg = __hfma2(a, x4, lg);
    lg = __hfma2(b, x5, lg);
    unpack4(wg.w, a, b);
    lg = __hfma2(a, x6, lg);
    lg = __hfma2(b, x7, lg);
    const int kb = k0 >> 7;
    float2 fu = __half22float2(lu), fg = __half22float2(lg);
    accU = fmaf(Su[kb], fu.x + fu.y, accU);
    accG = fmaf(Sg[kb], fg.x + fg.y, accG);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    accU += __shfl_xor_sync(FULLM, accU, o);
    accG += __shfl_xor_sync(FULLM, accG, o);
  }
  if (lane == 0) {
    const float g = accG * FC1_XS_UNSCALE;
    const float u = accU * FC1_XS_UNSCALE;
    Hb[(size_t)s * I + j] = __float2half_rn(((g / (1.f + __expf(-g))) * u) * (1.f / 256.f));
  }
}

// ------------------------------------------------------------------
template <int KT>
__global__ __launch_bounds__(512, 3) void fc2_t1_w32(
    const __half* __restrict__ Hb, const uint8_t* __restrict__ W2, const float* __restrict__ W2S,
    const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
    __nv_bfloat16* __restrict__ out, int H, int I, int Si) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int d = blockIdx.x * (blockDim.x >> 5) + warp;
  if (d >= H) return;
  const int Sh = (H + 127) >> 7;
  const int j0 = lane * 8;

  int eid[KT];
  float tws[KT];
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) {
    eid[kk] = topk_ids[kk];
    tws[kk] = topk_w[kk];
  }

  uint2 w[KT];
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) w[kk] = *(const uint2*)(W2 + ((size_t)eid[kk] * H + d) * I + j0);

  cudaGridDependencySynchronize();
  float acc = 0.f;
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) {
    const uint4 hv = *(const uint4*)(Hb + kk * I + j0);
    const float sc = W2S[((size_t)eid[kk] * Sh + (d >> 7)) * Si + (j0 >> 7)] * FC2_W_UNSCALE;
    const __half2* hp = (const __half2*)&hv;
    const __half2 wps = __float2half2_rn(FC2_W_PRESCALE);
    __half2 lh = __float2half2_rn(0.f), a, b;
    unpack4(w[kk].x, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[0], lh);
    lh = __hfma2(__hmul2(b, wps), hp[1], lh);
    unpack4(w[kk].y, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[2], lh);
    lh = __hfma2(__hmul2(b, wps), hp[3], lh);
    float2 f = __half22float2(lh);
    acc = fmaf(tws[kk] * sc, f.x + f.y, acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULLM, acc, o);
  if (lane == 0) out[d] = __float2bfloat16(acc * 256.f);
}

// fc2 T=1, paired rows: each warp owns two adjacent hidden columns, so a
// single LDG.128 per lane fetches a whole 256B W2 row (16 lanes/row) and the
// two half-warps reduce independently.  Halves the load instruction count
// versus the one-row/32-lane LDG.64 form at identical bytes in flight.
// ------------------------------------------------------------------
template <int KT>
__global__ __launch_bounds__(256, 3) void fc2_t1_pr(
    const __half* __restrict__ Hb, const uint8_t* __restrict__ W2, const float* __restrict__ W2S,
    const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
    __nv_bfloat16* __restrict__ out, int H, int I, int Si) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int sl = lane & 15, hf = lane >> 4;
  const int d = ((blockIdx.x * (blockDim.x >> 5)) + warp) * 2 + hf;
  if (d >= H) return;
  const int Sh = (H + 127) >> 7;
  const int j0 = sl * 16;

  int eid[KT];
  float tws[KT];
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) {
    eid[kk] = topk_ids[kk];
    tws[kk] = topk_w[kk];
  }

  uint4 w[KT];
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) w[kk] = *(const uint4*)(W2 + ((size_t)eid[kk] * H + d) * I + j0);

  // Scale data is independent of FC1.  Fetch it with W2 so these nine
  // cache misses can overlap the producer instead of extending the wait.
  float wsc[KT];
#pragma unroll
  for (int kk = 0; kk < KT; ++kk)
    wsc[kk] = W2S[((size_t)eid[kk] * Sh + (d >> 7)) * Si + (j0 >> 7)] * FC2_W_UNSCALE;

  cudaGridDependencySynchronize();
  float acc = 0.f;
#pragma unroll
  for (int kk = 0; kk < KT; ++kk) {
    const __half* hbp = Hb + kk * I + j0;
    uint4 h0 = *(const uint4*)(hbp);
    uint4 h1 = *(const uint4*)(hbp + 8);
    const float sc = wsc[kk];
    const __half2* p0 = (const __half2*)&h0;
    const __half2* p1 = (const __half2*)&h1;
    const __half2 wps = __float2half2_rn(FC2_W_PRESCALE);
    __half2 lh = __float2half2_rn(0.f), a, b;
    unpack4(w[kk].x, a, b);
    lh = __hfma2(__hmul2(a, wps), p0[0], lh);
    lh = __hfma2(__hmul2(b, wps), p0[1], lh);
    unpack4(w[kk].y, a, b);
    lh = __hfma2(__hmul2(a, wps), p0[2], lh);
    lh = __hfma2(__hmul2(b, wps), p0[3], lh);
    unpack4(w[kk].z, a, b);
    lh = __hfma2(__hmul2(a, wps), p1[0], lh);
    lh = __hfma2(__hmul2(b, wps), p1[1], lh);
    unpack4(w[kk].w, a, b);
    lh = __hfma2(__hmul2(a, wps), p1[2], lh);
    lh = __hfma2(__hmul2(b, wps), p1[3], lh);
    float2 f = __half22float2(lh);
    acc = fmaf(tws[kk] * sc, f.x + f.y, acc);
  }
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(FULLM, acc, o);
  if (sl == 0) out[d] = __float2bfloat16(acc * 256.f);
}

// fc2 for small T: one block row per token, loops the token's KTOP
// slots in registers so no atomics / no second pass are needed.
// 16 lanes cooperate on one hidden column d.
// ------------------------------------------------------------------
extern "C" __global__ void fc2_l16(const __half* __restrict__ Hb, const uint8_t* __restrict__ W2,
                                   const float* __restrict__ W2S, const int* __restrict__ topk_ids,
                                   const float* __restrict__ topk_w,
                                   __nv_bfloat16* __restrict__ out, int H, int I, int Si,
                                   int KTOP) {
  extern __shared__ __half hs[];
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  const int NH = KTOP * I;
  const uint4* hsrc = (const uint4*)(Hb + (size_t)t * NH);
  for (int i = tid; i < (NH >> 3); i += blockDim.x) {
    uint4 v = hsrc[i];
    const int h0 = i << 3;
    const int kk = h0 >> 8;
    const int within = h0 & 255;
    const int src_sub = within >> 4;
    const int pair = (within & 15) >> 1;
    uint32_t* dst = (uint32_t*)((__half2*)hs + kk * 144 + src_sub * 9 + pair);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int dsub = lane >> 4, sub = lane & 15;
  const int d = (blockIdx.y * (blockDim.x >> 5) + warp) * 2 + dsub;
  if (d >= H) return;
  const int Sh = (H + 127) >> 7;

  float acc = 0.f;
  for (int kk = 0; kk < KTOP; ++kk) {
    const int e = topk_ids[t * KTOP + kk];
    const uint8_t* wp = W2 + ((size_t)e * H + d) * I;
    const float* scp = W2S + ((size_t)e * Sh + (d >> 7)) * Si;
    float part = 0.f;
    const int j0 = sub * 16;
    uint4 w = *(const uint4*)(wp + j0);
    const __half2* hp = (const __half2*)hs + kk * 144 + sub * 9;
    const __half2 wps = __float2half2_rn(FC2_W_PRESCALE);
    __half2 lh = __float2half2_rn(0.f), a, b;
    unpack4(w.x, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[0], lh);
    lh = __hfma2(__hmul2(b, wps), hp[1], lh);
    unpack4(w.y, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[2], lh);
    lh = __hfma2(__hmul2(b, wps), hp[3], lh);
    unpack4(w.z, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[4], lh);
    lh = __hfma2(__hmul2(b, wps), hp[5], lh);
    unpack4(w.w, a, b);
    lh = __hfma2(__hmul2(a, wps), hp[6], lh);
    lh = __hfma2(__hmul2(b, wps), hp[7], lh);
    float2 f = __half22float2(lh);
    part = fmaf(scp[j0 >> 7] * FC2_W_UNSCALE, f.x + f.y, part);
    acc = fmaf(topk_w[t * KTOP + kk], part, acc);
  }
  acc += __shfl_down_sync(FULLM, acc, 8);
  acc += __shfl_down_sync(FULLM, acc, 4);
  acc += __shfl_down_sync(FULLM, acc, 2);
  acc += __shfl_down_sync(FULLM, acc, 1);
  if (sub == 0) out[(size_t)t * H + d] = __float2bfloat16(acc * 256.f);
}

extern "C" __global__ void fc2_l8(const __half* __restrict__ Hb, const uint8_t* __restrict__ W2,
                                  const float* __restrict__ W2S, const int* __restrict__ topk_ids,
                                  const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ out,
                                  int H, int I, int Si, int KTOP) {
  extern __shared__ __half hs[];
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  const int NH = KTOP * I;
  const uint4* hsrc = (const uint4*)(Hb + (size_t)t * NH);
  uint4* hdst = (uint4*)hs;
  for (int i = tid; i < (NH >> 3); i += blockDim.x) hdst[i] = hsrc[i];
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int dsub = lane >> 3, sub = lane & 7;
  const int d = (blockIdx.y * (blockDim.x >> 5) + warp) * 4 + dsub;
  if (d >= H) return;
  const int Sh = (H + 127) >> 7;

  float acc = 0.f;
  for (int kk = 0; kk < KTOP; ++kk) {
    const int e = topk_ids[t * KTOP + kk];
    const uint8_t* wp = W2 + ((size_t)e * H + d) * I;
    const float* scp = W2S + ((size_t)e * Sh + (d >> 7)) * Si;
    float part = 0.f;
#pragma unroll 2
    for (int c = 0; c < I; c += 128) {
      const int j0 = sub * 16 + c;
      uint4 w = *(const uint4*)(wp + j0);
      const __half2* hp = (const __half2*)(hs + kk * I + j0);
      const __half2 wps = __float2half2_rn(FC2_W_PRESCALE);
      __half2 lh = __float2half2_rn(0.f), a, b;
      unpack4(w.x, a, b);
      lh = __hfma2(__hmul2(a, wps), hp[0], lh);
      lh = __hfma2(__hmul2(b, wps), hp[1], lh);
      unpack4(w.y, a, b);
      lh = __hfma2(__hmul2(a, wps), hp[2], lh);
      lh = __hfma2(__hmul2(b, wps), hp[3], lh);
      unpack4(w.z, a, b);
      lh = __hfma2(__hmul2(a, wps), hp[4], lh);
      lh = __hfma2(__hmul2(b, wps), hp[5], lh);
      unpack4(w.w, a, b);
      lh = __hfma2(__hmul2(a, wps), hp[6], lh);
      lh = __hfma2(__hmul2(b, wps), hp[7], lh);
      float2 f = __half22float2(lh);
      part = fmaf(scp[j0 >> 7] * FC2_W_UNSCALE, f.x + f.y, part);
    }
    acc = fmaf(topk_w[t * KTOP + kk], part, acc);
  }
  acc += __shfl_down_sync(FULLM, acc, 4);
  acc += __shfl_down_sync(FULLM, acc, 2);
  acc += __shfl_down_sync(FULLM, acc, 1);
  if (sub == 0) out[(size_t)t * H + d] = __float2bfloat16(acc * 256.f);
}

void launch_moe_small(const void* X, const void* W13, const float* W13S, const void* W2,
                      const float* W2S, const int* ids, const float* tw, void* out, void* Hb, int T,
                      int H, int I, int E, int KTOP, cudaStream_t st) {
  const int Sh = (H + 127) >> 7;
  const int Si = (I + 127) >> 7;
  const int Sn1 = (2 * I + 127) >> 7;
  const int S = T * KTOP;
  if (T == 1) {
    dim3 grid(S, (I + 11) / 12);
    fc1_t1_u4<6, 384, 2><<<grid, 384, (size_t)H * sizeof(__half), st>>>(
        (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, ids, (__half*)Hb, H, I, Sh, Sn1, KTOP);
  } else {
    dim3 grid(S, T == 4 ? 16 : (I + 11) / 12);
    fc1_small<<<grid, 384, (size_t)H * sizeof(__half) + (size_t)Sn1 * Sh * sizeof(float), st>>>(
        (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, ids, (__half*)Hb, H, I, Sh, Sn1, KTOP);
  }
  if (T == 1) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3((H + 15) / 16);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, fc2_t1_pr<9>, (const __half*)Hb, (const uint8_t*)W2, W2S, ids, tw,
                       (__nv_bfloat16*)out, H, I, Si);
  } else if (T == 4) {
    dim3 grid(T, (H + 63) / 64);
    fc2_l8<<<grid, 512, (size_t)KTOP * I * sizeof(__half), st>>>(
        (const __half*)Hb, (const uint8_t*)W2, W2S, ids, tw, (__nv_bfloat16*)out, H, I, Si, KTOP);
  } else {
    dim3 grid(T, (H + 15) / 16);
    fc2_l8<<<grid, 128, (size_t)KTOP * I * sizeof(__half), st>>>(
        (const __half*)Hb, (const uint8_t*)W2, W2S, ids, tw, (__nv_bfloat16*)out, H, I, Si, KTOP);
  }
}

// ==================================================================
// Grouped (expert-binned) tensor-core path for large T.
// ==================================================================
#include <mma.h>

#define TM 128
#define TKC 64
#define ALD 72

__device__ __forceinline__ void cvt16(const uint4 w, float sc, __nv_bfloat16* dst) {
  const uint32_t* wp = (const uint32_t*)&w;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __half2 a, b;
    unpack4(wp[q], a, b);
    float2 fa = __half22float2(a), fb = __half22float2(b);
    *(__nv_bfloat162*)(dst + q * 4) = __floats2bfloat162_rn(fa.x * sc, fa.y * sc);
    *(__nv_bfloat162*)(dst + q * 4 + 2) = __floats2bfloat162_rn(fb.x * sc, fb.y * sc);
  }
}

__device__ __forceinline__ void cvt8h(const uint4 v, __half* dst) {
  const __nv_bfloat162* bp = (const __nv_bfloat162*)&v;
#pragma unroll
  for (int q = 0; q < 4; ++q)
    *(__half2*)(dst + q * 2) = __float22half2_rn(__bfloat1622float2(bp[q]));
}

// Dequantize against the fp32 block scale without ever pushing the scale
// through half's limited range: a [128,128] block scale is amax/448, so a
// small-magnitude block's fp32 scale can sit at or below half's subnormal
// range — narrowing it BEFORE the multiply flushes it toward zero (or
// strips most of its mantissa), zeroing or warping the whole dequantized
// tile. When the scale is outside half's safe window the raw e4m3 value is
// multiplied by the fp32 scale and the PRODUCT is narrowed, rounding the
// true effective weight once (subnormal halves included). Inside the
// window (normal-range scale, products of raw mantissas <= 448 cannot
// overflow) the single-instruction half multiply is kept: it differs from
// the fp32-scaled product by at most one extra half ulp, nine bits below
// the e4m3 quantization floor, and keeps the wgmma staging loops at their
// original instruction count. The branch is uniform per tile (one scale).
__device__ __forceinline__ bool cvt_sc_half_safe(float sc) {
  const float a = fabsf(sc);
  return a >= 6.103515625e-5f /* half min normal */ && a < 128.f /* 448*128 < 65504 */;
}

__device__ __forceinline__ void cvt16h(const uint4 w, float sc, __half* dst) {
  const uint32_t* wp = (const uint32_t*)&w;
  if (cvt_sc_half_safe(sc)) {
    const __half2 hs = __float2half2_rn(sc);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      __half2 a, b;
      unpack4(wp[q], a, b);
      *(__half2*)(dst + q * 4) = __hmul2(a, hs);
      *(__half2*)(dst + q * 4 + 2) = __hmul2(b, hs);
    }
    return;
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __half2 a, b;
    unpack4(wp[q], a, b);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    *(__half2*)(dst + q * 4) = __floats2half2_rn(fa.x * sc, fa.y * sc);
    *(__half2*)(dst + q * 4 + 2) = __floats2half2_rn(fb.x * sc, fb.y * sc);
  }
}

__device__ __forceinline__ void cvt16h_sw128(const uint4 w, float sc, __half* base, int row,
                                             int group2) {
  const uint32_t* wp = (const uint32_t*)&w;
  if (cvt_sc_half_safe(sc)) {
    const __half2 hs = __float2half2_rn(sc);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      __half2 a, b;
      unpack4(wp[q], a, b);
      const int g = (group2 + (q >> 1)) ^ (row & 7);
      __half* dst = base + (row * 8 + g) * 8 + (q & 1) * 4;
      *(__half2*)(dst) = __hmul2(a, hs);
      *(__half2*)(dst + 2) = __hmul2(b, hs);
    }
    return;
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __half2 a, b;
    unpack4(wp[q], a, b);
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    const int g = (group2 + (q >> 1)) ^ (row & 7);
    __half* dst = base + (row * 8 + g) * 8 + (q & 1) * 4;
    *(__half2*)(dst) = __floats2half2_rn(fa.x * sc, fa.y * sc);
    *(__half2*)(dst + 2) = __floats2half2_rn(fb.x * sc, fb.y * sc);
  }
}

__device__ __forceinline__ uint64_t wgmma_desc(const __half* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return ((uint64_t)(a & 0x3ffff) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma128_f16(float d[64], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "%64,%65,1,1,1,0,0;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
        "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
        "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
        "+f"(d[63])
      : "l"(da), "l"(db));
}

extern "C" __global__ void k_count(const int* __restrict__ ids, int* __restrict__ cnt, int pairs) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned active = __ballot_sync(FULLM, p < pairs);
  if (p >= pairs) return;
  const int e = ids[p];
  const unsigned peers = __match_any_sync(active, e);
  const int leader = __ffs(peers) - 1;
  if ((threadIdx.x & 31) == leader) atomicAdd(cnt + e, __popc(peers));
}

template <int BM>
__global__ __launch_bounds__(288) void k_prefix(const int* __restrict__ cnt, int* __restrict__ off,
                                                int* __restrict__ cur, int* __restrict__ toff,
                                                int2* __restrict__ tasks, int E) {
  __shared__ int warp_c[9], warp_t[9];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int c = (tid < E) ? cnt[tid] : 0;
  const int t = (c + BM - 1) / BM;
  int scan_c = c, scan_t = t;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(FULLM, scan_c, o);
    const int b = __shfl_up_sync(FULLM, scan_t, o);
    if (lane >= o) {
      scan_c += a;
      scan_t += b;
    }
  }
  if (lane == 31) {
    warp_c[warp] = scan_c;
    warp_t[warp] = scan_t;
  }
  __syncthreads();
  if (warp == 0) {
    int wc = (lane < 9) ? warp_c[lane] : 0;
    int wt = (lane < 9) ? warp_t[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a = __shfl_up_sync(FULLM, wc, o);
      const int b = __shfl_up_sync(FULLM, wt, o);
      if (lane >= o) {
        wc += a;
        wt += b;
      }
    }
    if (lane < 9) {
      warp_c[lane] = wc;
      warp_t[lane] = wt;
    }
  }
  __syncthreads();

  const int prev_c = (warp == 0) ? 0 : warp_c[warp - 1];
  const int prev_t = (warp == 0) ? 0 : warp_t[warp - 1];
  const int start = prev_c + scan_c - c;
  const int base = prev_t + scan_t - t;
  if (tid < E) {
    off[tid] = start;
    cur[tid] = start;
    toff[tid] = base;
    if (tid == E - 1) toff[E] = base + t;
    for (int q = 0; q < t; ++q) tasks[base + q] = make_int2(tid, start + q * BM);
  }
}

extern "C" __global__ void k_tasks(const int* __restrict__ cnt, const int* __restrict__ off,
                                   const int* __restrict__ toff, int2* __restrict__ tasks) {
  int e = blockIdx.x;
  int nt = (cnt[e] + TM - 1) / TM;
  for (int t = threadIdx.x; t < nt; t += blockDim.x)
    tasks[toff[e] + t] = make_int2(e, off[e] + t * TM);
}

extern "C" __global__ void k_scatter(const int* __restrict__ ids, int* __restrict__ cur,
                                     int* __restrict__ order, int pairs) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned active = __ballot_sync(FULLM, p < pairs);
  if (p >= pairs) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];
  const unsigned peers = __match_any_sync(active, e);
  const int leader = __ffs(peers) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(cur + e, __popc(peers));
  base = __shfl_sync(peers, base, leader);
  const unsigned lower = (lane == 0) ? 0u : ((1u << lane) - 1u);
  order[base + __popc(peers & lower)] = p;
}

// ---- FC1: M=64 x N=128 x K=H, 4 warps (2m x 2n), warp tile 32x64 ----
// Chunk n+1 is fetched into registers while chunk n is consumed by the
// tensor cores, so the global-memory latency is hidden by the MMAs.
#define FC1_LOAD(kb)                                                            \
  _Pragma("unroll") for (int q = 0; q < BM * 8 / THREADS; ++q) {                \
    const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7;                  \
    ra[q] = make_uint4(0, 0, 0, 0);                                             \
    const int tkn = sTok[r];                                                    \
    if (tkn >= 0) ra[q] = *(const uint4*)(X + (size_t)tkn * H + (kb) + c * 8);  \
  }                                                                             \
  _Pragma("unroll") for (int q = 0; q < BN / 128; ++q) rs[q] =                  \
      W13S[((size_t)e * Sn1 + nt * (BN / 128) + q) * Sh + ((kb) >> 7)];         \
  _Pragma("unroll") for (int q = 0; q < BN * 4 / THREADS; ++q) {                \
    const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;                  \
    const int grow = nt * BN + r;                                               \
    rb[q] = *(const uint4*)(W13 + ((size_t)e * I2 + grow) * H + (kb) + c * 16); \
  }

#define TILE_STORE(VA, VB, VS)                                 \
  _Pragma("unroll") for (int q = 0; q < 4; ++q) {              \
    const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7; \
    cvt8h(VA[q], sA + r * ALD + c * 8);                        \
  }                                                            \
  _Pragma("unroll") for (int q = 0; q < 512 / THREADS; ++q) {  \
    const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3; \
    cvt16h(VB[q], VS, sB + r * ALD + c * 16);                  \
  }

#define TILE_MMA(KLIM)                                                      \
  _Pragma("unroll") for (int kk = 0; kk < (KLIM); kk += 16) {               \
    wmma::load_matrix_sync(fa0, sA + (wm * 32) * ALD + kk, ALD);            \
    wmma::load_matrix_sync(fa1, sA + (wm * 32 + 16) * ALD + kk, ALD);       \
    _Pragma("unroll") for (int ni = 0; ni < 4; ++ni) {                      \
      wmma::load_matrix_sync(fb, sB + (wn * 64 + ni * 16) * ALD + kk, ALD); \
      wmma::mma_sync(acc[0][ni], fa0, fb, acc[0][ni]);                      \
      wmma::mma_sync(acc[1][ni], fa1, fb, acc[1][ni]);                      \
    }                                                                       \
  }

constexpr int fc1_wmma_smem(int BM, int BN) { return 1024 + BM * 128 + BN * 128; }

template <int BM, int BN, int THREADS>
__global__ __launch_bounds__(THREADS) void fc1_wmma(
    const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W13,
    const float* __restrict__ W13S, const int* __restrict__ order, const int* __restrict__ cnt,
    const int* __restrict__ off, const int* __restrict__ toff, const int2* __restrict__ tasks,
    __nv_bfloat16* __restrict__ raw, int H, int I2, int KTOP, int Sh, int Sn1, int E) {
  using namespace nvcuda;
  const int ntl = I2 / BN;
  const int ti = blockIdx.x / ntl;
  const int nt = blockIdx.x - ti * ntl;
  if (ti >= toff[E]) return;
  const int2 tk = tasks[ti];
  const int e = tk.x, start = tk.y;
  const int rows = min(BM, off[e] + cnt[e] - start);

  // Dynamic shared memory (sized by fc1_wmma_smem at the launch site): the
  // <128, 256> variant needs 50,176 bytes, over the 48 KiB static-smem
  // ptxas limit on sm_90a — a static array is a hard build error on CUDA
  // 12.x toolkits. The over-48 KiB variant is opted in with
  // cudaFuncAttributeMaxDynamicSharedMemorySize before launch.
  extern __shared__ __align__(1024) unsigned char swg1[];
  int* sPair = (int*)swg1;
  int* sTok = sPair + BM;
  __half* sA = (__half*)(swg1 + 1024);
  __half* sB = (__half*)(swg1 + 1024 + BM * 128);

  const int tid = threadIdx.x, wgid = tid >> 7;
  const int wgm = wgid % (BM / 64), wgn = wgid / (BM / 64);
  const int ltid = tid & 127, warp = ltid >> 5, lane = ltid & 31;

  for (int r = tid; r < BM; r += THREADS) {
    int p = -1, t = -1;
    if (r < rows) {
      p = order[start + r];
      t = p / KTOP;
    }
    sPair[r] = p;
    sTok[r] = t;
  }
  __syncthreads();

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;

  uint4 ra[BM * 8 / THREADS], rb[BN * 4 / THREADS];
  float rs[BN / 128];
  FC1_LOAD(0)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int kb = 0; kb < H; kb += TKC) {
#pragma unroll
    for (int q = 0; q < BM * 8 / THREADS; ++q) {
      const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7;
      cvt8h(ra[q], sA + (r * 8 + (c ^ (r & 7))) * 8);
    }
#pragma unroll
    for (int q = 0; q < BN * 4 / THREADS; ++q) {
      const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;
      cvt16h_sw128(rb[q], rs[r >> 7], sB + (r >> 7) * 128 * 64, r & 127, c * 2);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (kb + TKC < H) {
      FC1_LOAD(kb + TKC)
    }
    const uint64_t da = wgmma_desc(sA + wgm * 64 * 64);
    const uint64_t db = wgmma_desc(sB + wgn * 128 * 64);
#pragma unroll
    for (int kk = 0; kk < TKC; kk += 16) wgmma128_f16(acc, da + (kk >> 3), db + (kk >> 3));
    asm volatile(
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;" ::
            : "memory");
    __syncthreads();
  }

  const int row = wgm * 64 + warp * 16 + lane / 4;
  const int col0 = nt * BN + wgn * 128;
  const int cb = 2 * (lane & 3);
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    const int c = w * 16 + cb;
    if (row < rows) {
      __nv_bfloat16* dst = raw + (size_t)sPair[row] * I2 + col0 + c;
      *(__nv_bfloat162*)dst = __floats2bfloat162_rn(acc[w * 8], acc[w * 8 + 1]);
      *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 4], acc[w * 8 + 5]);
    }
    if (row + 8 < rows) {
      __nv_bfloat16* dst = raw + (size_t)sPair[row + 8] * I2 + col0 + c;
      *(__nv_bfloat162*)dst = __floats2bfloat162_rn(acc[w * 8 + 2], acc[w * 8 + 3]);
      *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 6], acc[w * 8 + 7]);
    }
  }
}

extern "C" __global__ void k_silu(const __nv_bfloat16* __restrict__ raw,
                                  __nv_bfloat16* __restrict__ Hb, int I, int pairs) {
  int p = blockIdx.x;
  for (int i = threadIdx.x * 2; i < I; i += blockDim.x * 2) {
    float2 up = __bfloat1622float2(*(__nv_bfloat162*)(raw + (size_t)p * 2 * I + i));
    float2 g = __bfloat1622float2(*(__nv_bfloat162*)(raw + (size_t)p * 2 * I + I + i));
    float h0 = (g.x / (1.f + __expf(-g.x))) * up.x;
    float h1 = (g.y / (1.f + __expf(-g.y))) * up.y;
    *(__nv_bfloat162*)(Hb + (size_t)p * I + i) = __floats2bfloat162_rn(h0, h1);
  }
}

// ---- FC2: M=64 x N=128 x K=I, 4 warps, same register-staged pipeline ----
#define FC2_LOAD(kb)                                                          \
  _Pragma("unroll") for (int q = 0; q < BM * 8 / THREADS; ++q) {              \
    const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7;                \
    ra[q] = make_uint4(0, 0, 0, 0);                                           \
    const int p = sPair[r];                                                   \
    if (p >= 0) ra[q] = *(const uint4*)(Hb + (size_t)p * I + (kb) + c * 8);   \
  }                                                                           \
  _Pragma("unroll") for (int q = 0; q < BN / 128; ++q) rs[q] =                \
      W2S[((size_t)e * Sh + nt * (BN / 128) + q) * Si + ((kb) >> 7)];         \
  _Pragma("unroll") for (int q = 0; q < BN * 4 / THREADS; ++q) {              \
    const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;                \
    const int grow = nt * BN + r;                                             \
    rb[q] = *(const uint4*)(W2 + ((size_t)e * H + grow) * I + (kb) + c * 16); \
  }

#define FC2_LOAD_NP(kb, ntv)                                                     \
  rs_np = W2S[((size_t)e * Sh + (ntv)) * Si + ((kb) >> 7)];                      \
  _Pragma("unroll") for (int q = 0; q < 512 / THREADS; ++q) {                    \
    const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;                   \
    const int grow = (ntv) * 128 + r;                                            \
    rb_np[q] = *(const uint4*)(W2 + ((size_t)e * H + grow) * I + (kb) + c * 16); \
  }

// The per-slot partial buffer is BF16: the final output is BF16, and a
// partial legitimately reaches any magnitude the combined output can (the
// combine weights are a convex sum), so a half-typed buffer overflows to
// inf at 65504 while the true result is still well inside BF16 range.
template <int BM, int BN, int THREADS>
__global__ __launch_bounds__(THREADS) void fc2_wmma(
    const __nv_bfloat16* __restrict__ Hb, const uint8_t* __restrict__ W2,
    const float* __restrict__ W2S, const int* __restrict__ order, const int* __restrict__ cnt,
    const int* __restrict__ off, const int* __restrict__ toff, const int2* __restrict__ tasks,
    __nv_bfloat16* __restrict__ partial, int H, int I, int KTOP, int Si, int Sh, int E) {
  using namespace nvcuda;
  const int ntl = H / BN;
  const int ti = blockIdx.x / ntl;
  const int nt = blockIdx.x - ti * ntl;
  if (ti >= toff[E]) return;
  const int2 tk = tasks[ti];
  const int e = tk.x, start = tk.y;
  const int rows = min(BM, off[e] + cnt[e] - start);

  __shared__ __align__(1024) unsigned char swg[1024 + BM * 128 + BN * 128];
  int* sPair = (int*)swg;
  __half* sA = (__half*)(swg + 1024);
  __half* sB = (__half*)(swg + 1024 + BM * 128);

  const int tid = threadIdx.x, wgid = tid >> 7;
  const int wgm = wgid % (BM / 64), wgn = wgid / (BM / 64);
  const int ltid = tid & 127, warp = ltid >> 5, lane = ltid & 31;

  for (int r = tid; r < BM; r += THREADS) {
    int p = -1;
    if (r < rows) p = order[start + r];
    sPair[r] = p;
  }
  __syncthreads();

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;

  uint4 ra[BM * 8 / THREADS], rb[BN * 4 / THREADS];
  float rs[BN / 128];
  FC2_LOAD(0)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  for (int kb = 0; kb < I; kb += TKC) {
#pragma unroll
    for (int q = 0; q < BM * 8 / THREADS; ++q) {
      const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7;
      cvt8h(ra[q], sA + (r * 8 + (c ^ (r & 7))) * 8);
    }
#pragma unroll
    for (int q = 0; q < BN * 4 / THREADS; ++q) {
      const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;
      cvt16h_sw128(rb[q], rs[r >> 7], sB + (r >> 7) * 128 * 64, r & 127, c * 2);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (kb + TKC < I) {
      FC2_LOAD(kb + TKC)
    }
    const uint64_t da = wgmma_desc(sA + wgm * 64 * 64);
    const uint64_t db = wgmma_desc(sB + wgn * 128 * 64);
#pragma unroll
    for (int kk = 0; kk < TKC; kk += 16) wgmma128_f16(acc, da + (kk >> 3), db + (kk >> 3));
    asm volatile(
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;" ::
            : "memory");
    __syncthreads();
  }

  const int row = wgm * 64 + warp * 16 + lane / 4;
  const int col0 = nt * BN + wgn * 128;
  const int cb = 2 * (lane & 3);
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    const int c = w * 16 + cb;
    if (row < rows) {
      __nv_bfloat16* dst = partial + (size_t)sPair[row] * H + col0 + c;
      *(__nv_bfloat162*)(dst) = __floats2bfloat162_rn(acc[w * 8], acc[w * 8 + 1]);
      *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 4], acc[w * 8 + 5]);
    }
    if (row + 8 < rows) {
      __nv_bfloat16* dst = partial + (size_t)sPair[row + 8] * H + col0 + c;
      *(__nv_bfloat162*)(dst) = __floats2bfloat162_rn(acc[w * 8 + 2], acc[w * 8 + 3]);
      *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 6], acc[w * 8 + 7]);
    }
  }
}

template <int BM, int THREADS, int NGROUP>
__global__ __launch_bounds__(THREADS) void fc2_wmma_np(
    const __nv_bfloat16* __restrict__ Hb, const uint8_t* __restrict__ W2,
    const float* __restrict__ W2S, const int* __restrict__ order, const int* __restrict__ cnt,
    const int* __restrict__ off, const int* __restrict__ toff, const int2* __restrict__ tasks,
    __nv_bfloat16* __restrict__ partial, int H, int I, int KTOP, int Si, int Sh, int E) {
  const int ntl = H >> 7;
  const int ngl = (ntl + NGROUP - 1) / NGROUP;
  const int ti = blockIdx.x / ngl;
  const int nt0 = (blockIdx.x - ti * ngl) * NGROUP;
  if (ti >= toff[E]) return;
  const int2 tk = tasks[ti];
  const int e = tk.x, start = tk.y;
  const int rows = min(BM, off[e] + cnt[e] - start);

  extern __shared__ __align__(1024) unsigned char swg_np[];
  int* sPair = (int*)swg_np;
  __half* sA = (__half*)(swg_np + 1024);
  __half* sB = (__half*)(swg_np + 1024 + BM * 512);
  const int tid = threadIdx.x, wgid = tid >> 7;
  const int ltid = tid & 127, warp = ltid >> 5, lane = ltid & 31;

  for (int r = tid; r < BM; r += THREADS) {
    int p = -1;
    if (r < rows) p = order[start + r];
    sPair[r] = p;
  }
  __syncthreads();

#pragma unroll
  for (int kb = 0; kb < I; kb += TKC) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ii = tid + q * THREADS, r = ii >> 3, c = ii & 7;
      uint4 v = make_uint4(0, 0, 0, 0);
      const int p = sPair[r];
      if (p >= 0) v = *(const uint4*)(Hb + (size_t)p * I + kb + c * 8);
      cvt8h(v, sA + kb * BM + (r * 8 + (c ^ (r & 7))) * 8);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  const int row = wgid * 64 + warp * 16 + lane / 4;
  const int cb = 2 * (lane & 3);
  uint4 rb_np[4];
  float rs_np;
  for (int ni = 0; ni < NGROUP && nt0 + ni < ntl; ++ni) {
    const int nt = nt0 + ni;
    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    FC2_LOAD_NP(0, nt)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    for (int kb = 0; kb < I; kb += TKC) {
#pragma unroll
      for (int q = 0; q < 512 / THREADS; ++q) {
        const int ii = tid + q * THREADS, r = ii >> 2, c = ii & 3;
        cvt16h_sw128(rb_np[q], rs_np, sB, r, c * 2);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (kb + TKC < I) {
        FC2_LOAD_NP(kb + TKC, nt)
      }
      const uint64_t da = wgmma_desc(sA + kb * BM + wgid * 64 * 64);
      const uint64_t db = wgmma_desc(sB);
#pragma unroll
      for (int kk = 0; kk < TKC; kk += 16) wgmma128_f16(acc, da + (kk >> 3), db + (kk >> 3));
      asm volatile(
          "wgmma.commit_group.sync.aligned;\n"
          "wgmma.wait_group.sync.aligned 0;" ::
              : "memory");
      __syncthreads();
    }

#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const int c = w * 16 + cb;
      if (row < rows) {
        __nv_bfloat16* dst = partial + (size_t)sPair[row] * H + nt * 128 + c;
        *(__nv_bfloat162*)dst = __floats2bfloat162_rn(acc[w * 8], acc[w * 8 + 1]);
        *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 4], acc[w * 8 + 5]);
      }
      if (row + 8 < rows) {
        __nv_bfloat16* dst = partial + (size_t)sPair[row + 8] * H + nt * 128 + c;
        *(__nv_bfloat162*)dst = __floats2bfloat162_rn(acc[w * 8 + 2], acc[w * 8 + 3]);
        *(__nv_bfloat162*)(dst + 8) = __floats2bfloat162_rn(acc[w * 8 + 6], acc[w * 8 + 7]);
      }
    }
  }
}

extern "C" __global__ void k_reduce9(const __nv_bfloat16* __restrict__ partial,
                                     const float* __restrict__ tw, __nv_bfloat16* __restrict__ out,
                                     int H) {
  __shared__ float sw[9];
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  if (tid < 9) sw[tid] = tw[t * 9 + tid];
  __syncthreads();
  for (int d4 = tid; d4 < (H >> 2); d4 += blockDim.x) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      uint2 hv = *(const uint2*)(partial + ((size_t)t * 9 + k) * H + d4 * 4);
      const __nv_bfloat162* hh = (const __nv_bfloat162*)&hv;
      float2 f0 = __bfloat1622float2(hh[0]);
      float2 f1 = __bfloat1622float2(hh[1]);
      const float w = sw[k];
      a0 = fmaf(w, f0.x, a0);
      a1 = fmaf(w, f0.y, a1);
      a2 = fmaf(w, f1.x, a2);
      a3 = fmaf(w, f1.y, a3);
    }
    ((__nv_bfloat162*)out)[(size_t)t * (H >> 1) + d4 * 2] = __floats2bfloat162_rn(a0, a1);
    ((__nv_bfloat162*)out)[(size_t)t * (H >> 1) + d4 * 2 + 1] = __floats2bfloat162_rn(a2, a3);
  }
}

void launch_moe_grouped(const void* X, const void* W13, const float* W13S, const void* W2,
                        const float* W2S, const int* ids, const float* tw, void* out, void* raw,
                        void* Hb, void* partial, int* ibuf, int2* tasks, int T, int H, int I, int E,
                        int KTOP, int max_tasks, cudaStream_t st) {
  const int pairs = T * KTOP;
  const int Sh1 = (H + 127) >> 7;      // 48  (w13 scale inner dim)
  const int Si = (I + 127) >> 7;       // 2
  const int Sn1 = (2 * I + 127) >> 7;  // 4
  int* cnt = ibuf;
  int* off = cnt + E;
  int* cur = off + E;
  int* toff = cur + E;  // E+1
  int* order = toff + E + 1;

  cudaMemsetAsync(cnt, 0, sizeof(int) * E, st);
  k_count<<<(pairs + 255) / 256, 256, 0, st>>>(ids, cnt, pairs);
  const bool wide_m = T >= 2048;
  const bool wide_n = T >= 256;
  if (wide_m)
    k_prefix<128><<<1, 288, 0, st>>>(cnt, off, cur, toff, tasks, E);
  else
    k_prefix<64><<<1, 288, 0, st>>>(cnt, off, cur, toff, tasks, E);
  k_scatter<<<(pairs + 255) / 256, 256, 0, st>>>(ids, cur, order, pairs);

  if (wide_m) {
    constexpr int FC1_SMEM = fc1_wmma_smem(128, 256);
    cudaFuncSetAttribute(fc1_wmma<128, 256, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         FC1_SMEM);
    fc1_wmma<128, 256, 512><<<max_tasks * ((2 * I) / 256), 512, FC1_SMEM, st>>>(
        (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)raw, H, 2 * I, KTOP, Sh1, Sn1, E);
  } else if (wide_n)
    fc1_wmma<64, 256, 256><<<max_tasks * ((2 * I) / 256), 256, fc1_wmma_smem(64, 256), st>>>(
        (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)raw, H, 2 * I, KTOP, Sh1, Sn1, E);
  else
    fc1_wmma<64, 128, 128><<<max_tasks * ((2 * I) >> 7), 128, fc1_wmma_smem(64, 128), st>>>(
        (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)raw, H, 2 * I, KTOP, Sh1, Sn1, E);
  k_silu<<<pairs, 128, 0, st>>>((const __nv_bfloat16*)raw, (__nv_bfloat16*)Hb, I, pairs);
  if (wide_m) {
    constexpr int FC2_NGROUP = 8;
    constexpr int FC2_SMEM = 1024 + 128 * 512 + 16384;
    const int fc2_ng = ((H >> 7) + FC2_NGROUP - 1) / FC2_NGROUP;
    cudaFuncSetAttribute(fc2_wmma_np<128, 256, FC2_NGROUP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, FC2_SMEM);
    fc2_wmma_np<128, 256, FC2_NGROUP><<<max_tasks * fc2_ng, 256, FC2_SMEM, st>>>(
        (const __nv_bfloat16*)Hb, (const uint8_t*)W2, W2S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)partial, H, I, KTOP, Si, Sh1, E);
  } else if (wide_n)
    fc2_wmma<64, 256, 256><<<max_tasks*(H / 256), 256, 0, st>>>(
        (const __nv_bfloat16*)Hb, (const uint8_t*)W2, W2S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)partial, H, I, KTOP, Si, Sh1, E);
  else
    fc2_wmma<64, 128, 128><<<max_tasks*(H >> 7), 128, 0, st>>>(
        (const __nv_bfloat16*)Hb, (const uint8_t*)W2, W2S, order, cnt, off, toff, tasks,
        (__nv_bfloat16*)partial, H, I, KTOP, Si, Sh1, E);
  if (wide_m)
    k_reduce9<<<T, 512, 0, st>>>((const __nv_bfloat16*)partial, tw, (__nv_bfloat16*)out, H);
  else
    k_reduce9<<<T, 256, 0, st>>>((const __nv_bfloat16*)partial, tw, (__nv_bfloat16*)out, H);
}
