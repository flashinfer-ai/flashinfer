// Multi-warpgroup, intra-CTA split-K FP8 block-scaled decode GEMM (SM90).
//
// The cluster split-K kernels spend ~39% of their stall samples inside
// `cluster.sync()` (MEMBAR.ALL.GPU + CGA barrier) and run at ~12% achieved
// occupancy because each CTA only carries 4 warps.  For shapes with enough
// N tiles to cover the GPU (N/64 >= ~96) the K split can live *inside* one
// CTA instead: NWG warpgroups each own a disjoint K range, stage their own
// weight tile, and the cross-split reduction becomes a plain __syncthreads()
// over shared memory.  That removes the cluster barrier entirely and raises
// the per-SM warp count from 8 to 24-32.

//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants, relocated verbatim from the former csrc kernel
// TU. Included by exactly one binding TU, csrc/fp8bs_gemm_decodem_sm90_mwg.cu, which owns the
// host-side launch glue and the exported entry point. The geometry macros
// are file-scope by design: never include two of these headers in the same
// translation unit.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace mwg {

__device__ __forceinline__ uint16_t fp8x2(float x0, float x1) {
  uint16_t result;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(result) : "f"(x0), "f"(x1));
  return result;
}

__device__ __forceinline__ uint64_t wgmma_desc(const uint8_t* ptr) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return ((static_cast<uint64_t>(addr) & 0x3ffffull) >> 4) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 62);
}

template <bool Add>
__device__ __forceinline__ void wgmma_ss_m64n8(float (&d)[4], uint64_t a, uint64_t b) {
  if constexpr (Add) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, %4, %5, 1, 1, 1;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(a), "l"(b));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3}, %4, %5, 0, 1, 1;"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "l"(a), "l"(b));
  }
}

template <bool Add>
__device__ __forceinline__ void wgmma_ss_m64n16(float (&d)[8], uint64_t a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, %10, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7])
      : "l"(a), "l"(b), "n"(Add ? 1 : 0));
}

template <bool Add>
__device__ __forceinline__ void wgmma_ss_m64n24(float (&d)[12], uint64_t a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, %12, %13, %14, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "l"(a), "l"(b), "n"(Add ? 1 : 0));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if (valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;" ::"r"(smem), "l"(src) : "memory");
  }
}

__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}

// One warp owns one 1x128 activation group: 32 lanes x 4 bf16.  The group
// amax collapses to a single `redux.sync` instead of a five-deep shuffle
// chain, and every lane already holds the result so no broadcast is needed.
template <bool SparseM1 = false>
__device__ __forceinline__ void quantize_row_warp(uint2 raw, uint8_t* bq, float* ascale, float eps,
                                                  int row, int lane, bool valid = true) {
  union Bf16Pair {
    uint32_t u32;
    __nv_bfloat162 bf16x2;
  };
  const int logical = (row & 7) * 128 + lane * 4;
  const int physical = logical ^ ((logical & 0x380) >> 3);
  if constexpr (SparseM1) {
    if (!valid) {
      if (lane == 0) ascale[row] = 0.0f;
      *reinterpret_cast<uint32_t*>(bq + (row >> 3) * 1024 + physical) = 0;
      return;
    }
  }
  const uint32_t w0 = raw.x, w1 = raw.y;
  uint32_t m = w0 & 0x7fffu;
  uint32_t o = (w0 >> 16) & 0x7fffu;
  m = m > o ? m : o;
  o = w1 & 0x7fffu;
  m = m > o ? m : o;
  o = (w1 >> 16) & 0x7fffu;
  m = m > o ? m : o;
  m = warp_max_u32(m);
  const float gmax = __uint_as_float(m << 16);
  // Stock scale_1x128 semantics: the clamp floor is 2^-117 and fires only
  // for amax == 0 (exactly the stock zero contribution) or an amax below
  // 2^-117 — far under any BF16-normal magnitude — where both routes'
  // outputs round to the same BF16 value.  Every other amax scales by
  // amax/448 unclamped (bitwise-stock for power-of-two amax, <=1 ulp
  // otherwise).  448/cl stays finite (<= 448*2^117 < FLT_MAX), so no 0*inf
  // NaN can form.
  const float cl = fmaxf(gmax, 6.019e-36f);
  const float scale = cl * (1.0f / 448.0f);
  const float inv_scale = __fdividef(448.0f, cl);
  if (lane == 0) ascale[row] = scale;
  Bf16Pair p0, p1;
  p0.u32 = w0;
  p1.u32 = w1;
  const float2 v0 = __bfloat1622float2(p0.bf16x2);
  const float2 v1 = __bfloat1622float2(p1.bf16x2);
  const uint32_t packed = static_cast<uint32_t>(fp8x2(v0.x * inv_scale, v0.y * inv_scale)) |
                          (static_cast<uint32_t>(fp8x2(v1.x * inv_scale, v1.y * inv_scale)) << 16);
  *reinterpret_cast<uint32_t*>(bq + (row >> 3) * 1024 + physical) = packed;
}

__device__ __forceinline__ unsigned half_warp_mask() {
  unsigned laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return 0x0000ffffu << (laneid & 16u);
}

__device__ __forceinline__ uint32_t redux_max_u32(uint32_t v, unsigned mask) {
  uint32_t r;
  asm("redux.sync.max.u32 %0, %1, %2;" : "=r"(r) : "r"(v), "r"(mask));
  return r;
}

// One 1x128 group per half-warp: eight bf16 values per lane.  This is used
// only by the 1024-thread MT24/NWG8 route, where the extra live values do not
// reduce residency and two independent groups retire per warp instruction.
template <bool BitAmax = false, bool FixedHalf = false>
__device__ __forceinline__ void quantize_row_half(uint4 raw, uint8_t* bq, float* ascale, float eps,
                                                  int row, int sublane) {
  float f[8];
  f[0] = __uint_as_float(raw.x << 16);
  f[1] = __uint_as_float(raw.x & 0xffff0000u);
  f[2] = __uint_as_float(raw.y << 16);
  f[3] = __uint_as_float(raw.y & 0xffff0000u);
  f[4] = __uint_as_float(raw.z << 16);
  f[5] = __uint_as_float(raw.z & 0xffff0000u);
  f[6] = __uint_as_float(raw.w << 16);
  f[7] = __uint_as_float(raw.w & 0xffff0000u);
  float gmax;
  if constexpr (BitAmax) {
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
    uint32_t m = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t m0 = words[i] & 0x7fffu;
      const uint32_t m1 = (words[i] >> 16) & 0x7fffu;
      m = m > m0 ? m : m0;
      m = m > m1 ? m : m1;
    }
    gmax = __uint_as_float(redux_max_u32(m, half_warp_mask()) << 16);
  }
  if constexpr (!BitAmax) {
    float m = fmaxf(fabsf(f[0]), fabsf(f[1]));
#pragma unroll
    for (int i = 2; i < 8; ++i) m = fmaxf(m, fabsf(f[i]));
    uint32_t m_bits = __float_as_uint(m);
    if constexpr (FixedHalf) {
#pragma unroll
      for (int delta = 8; delta > 0; delta >>= 1) {
        const uint32_t other = __shfl_xor_sync(0xffffffffu, m_bits, delta, 16);
        m_bits = m_bits > other ? m_bits : other;
      }
    } else {
      m_bits = redux_max_u32(m_bits, half_warp_mask());
    }
    gmax = __uint_as_float(m_bits);
  }
  // Stock scale_1x128 semantics: the clamp floor is 2^-117 and fires only
  // for amax == 0 (exactly the stock zero contribution) or an amax below
  // 2^-117 — far under any BF16-normal magnitude — where both routes'
  // outputs round to the same BF16 value.  Every other amax scales by
  // amax/448 unclamped (bitwise-stock for power-of-two amax, <=1 ulp
  // otherwise).  448/cl stays finite (<= 448*2^117 < FLT_MAX), so no 0*inf
  // NaN can form.
  const float clamped = fmaxf(gmax, 6.019e-36f);
  const float scale = clamped * (1.0f / 448.0f);
  if (sublane == 0) ascale[row] = scale;
  const float inv_scale = __fdividef(448.0f, clamped);
  const uint2 result =
      make_uint2(static_cast<uint32_t>(fp8x2(f[0] * inv_scale, f[1] * inv_scale)) |
                     (static_cast<uint32_t>(fp8x2(f[2] * inv_scale, f[3] * inv_scale)) << 16),
                 static_cast<uint32_t>(fp8x2(f[4] * inv_scale, f[5] * inv_scale)) |
                     (static_cast<uint32_t>(fp8x2(f[6] * inv_scale, f[7] * inv_scale)) << 16));
  const int logical = (row & 7) * 128 + sublane * 8;
  const int physical = logical ^ ((logical & 0x380) >> 3);
  *reinterpret_cast<uint2*>(bq + (row >> 3) * 1024 + physical) = result;
}

template <int MT, bool HoistDesc = false>
__device__ __forceinline__ void issue_mwg_wgmma(float (&partial)[MT / 2], const uint8_t* wtile,
                                                const uint8_t* qtile) {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  uint64_t da = 0;
  uint64_t db = 0;
  if constexpr (HoistDesc) {
    da = wgmma_desc(wtile);
    db = wgmma_desc(qtile);
  }
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    if constexpr (!HoistDesc) {
      da = wgmma_desc(wtile + ks * 32);
      db = wgmma_desc(qtile + ks * 32);
    }
    if constexpr (MT == 8) {
      if (ks == 0)
        wgmma_ss_m64n8<false>(partial, da, db);
      else
        wgmma_ss_m64n8<true>(partial, da, db);
    } else if constexpr (MT == 16) {
      if (ks == 0)
        wgmma_ss_m64n16<false>(partial, da, db);
      else
        wgmma_ss_m64n16<true>(partial, da, db);
    } else {
      if (ks == 0)
        wgmma_ss_m64n24<false>(partial, da, db);
      else
        wgmma_ss_m64n24<true>(partial, da, db);
    }
    if constexpr (HoistDesc) {
      da += 2;
      db += 2;
    }
  }
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int MT, int NWG, int KU, bool ExactOneRound = false, bool StaggerMT8 = false,
          bool SparseM1 = false, bool HalfQuant = false, bool PipelineMMA = false,
          bool DenseM = false>
__global__ __launch_bounds__(128 * NWG, 1) void mwg_kernel(const __nv_bfloat16* __restrict__ x,
                                                           const uint8_t* __restrict__ weight,
                                                           const float* __restrict__ weight_scale,
                                                           float eps,
                                                           __nv_bfloat16* __restrict__ out, int M,
                                                           int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24, "token tile");
  static_assert(!StaggerMT8 || ((MT == 8 && NWG == 6 && KU == 2 && ExactOneRound) ||
                                (MT == 16 && NWG == 8 && KU == 2 && ExactOneRound)));
  static_assert(!SparseM1 || (MT == 8 && NWG == 6 && KU == 2 && ExactOneRound));
  static_assert(!HalfQuant || (KU == 2 && ExactOneRound &&
                               ((MT == 24 && (NWG == 6 || NWG == 8)) || (MT == 16 && NWG == 8))));
  static_assert(!PipelineMMA || (KU == 2 && ExactOneRound && NWG == 6 && MT == 8));
  static_assert(!DenseM || (ExactOneRound && HalfQuant && (MT == 16 || MT == 24)));
  constexpr int NACC = MT / 2;
  constexpr int ATOMS = MT / 8;
  constexpr bool WeightPipeW1 = MT == 24 && (NWG == 6 || NWG == 8) && KU == 2 && ExactOneRound;
  constexpr bool FixedHalfW1 = MT == 24 && NWG == 6 && KU == 2 && ExactOneRound;
  constexpr bool LateAccumInitW1 = FixedHalfW1 && DenseM;
  constexpr bool HoistPipelineDesc = MT == 8 && NWG == 6 && KU == 2 && ExactOneRound && !SparseM1;
  constexpr bool PrefetchScaleW6 = MT == 16 && NWG == 8 && KU == 2 && ExactOneRound;
  constexpr int WBYTES = 64 * 128;      // one 64x128 fp8 weight tile
  constexpr int QBYTES = ATOMS * 1024;  // one MTx128 fp8 activation tile
  constexpr int REDUCE_BYTES = (NACC / 2) * 128 * sizeof(float2);
  static_assert(MT != 24 || REDUCE_BYTES <= KU * WBYTES,
                "MT24 reduction exceeds per-warpgroup weight scratch");

  extern __shared__ uint8_t dyn_smem[];
  uint8_t* const base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(dyn_smem) + 1023u) &
                                                   ~static_cast<uintptr_t>(1023u));
  uint8_t* const wsmem = base;
  uint8_t* const qsmem = base + NWG * KU * WBYTES;
  float* const ssmem = reinterpret_cast<float*>(qsmem + NWG * KU * QBYTES);

  const int tid = static_cast<int>(threadIdx.x);
  const int wg = tid >> 7;
  const int t = tid & 127;
  const int warp = t >> 5;
  const int lane = t & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int bar = wg + 1;

  const int tile_n = static_cast<int>(blockIdx.x) * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* const tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;
  const int kb_begin = ExactOneRound ? KU * wg : (Sk * wg) / NWG;
  const int kb_end = ExactOneRound ? kb_begin + KU : (Sk * (wg + 1)) / NWG;

  uint8_t* const my_w = wsmem + wg * KU * WBYTES;
  uint8_t* const my_q = qsmem + wg * KU * QBYTES;
  float* const my_s = ssmem + wg * KU * MT;

  const int kseg = t & 7;
  const int wrow0 = t >> 3;
  int wsoff[4];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int row = wrow0 + 16 * it;
    const int logical = (row & 7) * 128 + kseg * 16;
    wsoff[it] = (row >> 3) * 1024 + (logical ^ ((logical & 0x380) >> 3));
  }
  constexpr int RPW = MT / 4;  // activation rows handled by each warp
  constexpr int RPH = MT / 8;  // activation rows handled by each half-warp
  const int sublane = t & 15;
  const int arow0 = t >> 4;
  const int nrow = tile_n + wrow0;
  const float* const wsp = weight_scale + (tile_n >> 7) * Sk;
  const uint8_t* const wbase = weight + static_cast<int64_t>(nrow) * K + kseg * 16;
  const int64_t wrowstep = static_cast<int64_t>(16) * K;

  float accum[NACC];
  if constexpr (!LateAccumInitW1) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) accum[i] = 0.0f;
  }

#pragma unroll
  for (int kc = kb_begin; kc < kb_end; kc += KU) {
    uint2 araw[KU][HalfQuant ? 1 : RPW];
    uint4 araw4[KU][HalfQuant ? RPH : 1];
    if constexpr (StaggerMT8) {
      // Spread the independent global-memory issue streams.  MT8 alternates
      // by warpgroup; W6 reverses only its last two non-owner warpgroups.
      const bool weight_first = MT == 8 ? ((wg & 1) != 0) : (wg >= NWG - 2);
      if (weight_first) {
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
          for (int it = 0; it < 4; ++it) {
            const bool valid = live && (nrow + 16 * it) < N;
            cp_async16(my_w + u * WBYTES + wsoff[it], wbase + it * wrowstep + (kc + u) * 128,
                       valid);
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
          if constexpr (HalfQuant) {
#pragma unroll
            for (int a = 0; a < RPH; ++a) {
              const int row = arow0 + 8 * a;
              if constexpr (DenseM) {
                araw4[u][a] = *reinterpret_cast<const uint4*>(
                    tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + sublane * 8);
              } else {
                araw4[u][a] = make_uint4(0, 0, 0, 0);
                if (live && row < tile_rows)
                  araw4[u][a] = *reinterpret_cast<const uint4*>(
                      tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + sublane * 8);
              }
            }
          } else {
#pragma unroll
            for (int a = 0; a < RPW; ++a) {
              const int row = warp + 4 * a;
              araw[u][a] = make_uint2(0, 0);
              if (live && row < tile_rows)
                araw[u][a] = *reinterpret_cast<const uint2*>(
                    tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + lane * 4);
            }
          }
        }
      } else {
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
          if constexpr (HalfQuant) {
#pragma unroll
            for (int a = 0; a < RPH; ++a) {
              const int row = arow0 + 8 * a;
              if constexpr (DenseM) {
                araw4[u][a] = *reinterpret_cast<const uint4*>(
                    tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + sublane * 8);
              } else {
                araw4[u][a] = make_uint4(0, 0, 0, 0);
                if (live && row < tile_rows)
                  araw4[u][a] = *reinterpret_cast<const uint4*>(
                      tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + sublane * 8);
              }
            }
          } else {
#pragma unroll
            for (int a = 0; a < RPW; ++a) {
              const int row = warp + 4 * a;
              araw[u][a] = make_uint2(0, 0);
              if (live && row < tile_rows)
                araw[u][a] = *reinterpret_cast<const uint2*>(
                    tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + lane * 4);
            }
          }
        }
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
          for (int it = 0; it < 4; ++it) {
            const bool valid = live && (nrow + 16 * it) < N;
            cp_async16(my_w + u * WBYTES + wsoff[it], wbase + it * wrowstep + (kc + u) * 128,
                       valid);
          }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    } else {
      // Activation rows go out first: their LDG latency is then covered by
      // weight address arithmetic and cp.async issue below.
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
        if constexpr (HalfQuant) {
#pragma unroll
          for (int a = 0; a < RPH; ++a) {
            const int row = arow0 + 8 * a;
            if constexpr (DenseM) {
              araw4[u][a] = *reinterpret_cast<const uint4*>(tile_x + static_cast<int64_t>(row) * K +
                                                            (kc + u) * 128 + sublane * 8);
            } else {
              araw4[u][a] = make_uint4(0, 0, 0, 0);
              if (live && row < tile_rows)
                araw4[u][a] = *reinterpret_cast<const uint4*>(
                    tile_x + static_cast<int64_t>(row) * K + (kc + u) * 128 + sublane * 8);
            }
          }
        } else {
#pragma unroll
          for (int a = 0; a < RPW; ++a) {
            const int row = warp + 4 * a;
            araw[u][a] = make_uint2(0, 0);
            if (live && row < tile_rows)
              araw[u][a] = *reinterpret_cast<const uint2*>(tile_x + static_cast<int64_t>(row) * K +
                                                           (kc + u) * 128 + lane * 4);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const bool live = ExactOneRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
          const bool valid = live && (nrow + 16 * it) < N;
          cp_async16(my_w + u * WBYTES + wsoff[it], wbase + it * wrowstep + (kc + u) * 128, valid);
        }
        if constexpr (WeightPipeW1) asm volatile("cp.async.commit_group;" ::: "memory");
      }
      if constexpr (!WeightPipeW1) asm volatile("cp.async.commit_group;" ::: "memory");
    }

    if constexpr (LateAccumInitW1) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) accum[i] = 0.0f;
    }

#pragma unroll
    for (int u = 0; u < KU; ++u) {
      if constexpr (HalfQuant) {
#pragma unroll
        for (int a = 0; a < RPH; ++a)
          quantize_row_half<false, FixedHalfW1>(araw4[u][a], my_q + u * QBYTES, my_s + u * MT, eps,
                                                arow0 + 8 * a, sublane);
      } else {
#pragma unroll
        for (int a = 0; a < RPW; ++a)
          quantize_row_warp<SparseM1>(araw[u][a], my_q + u * QBYTES, my_s + u * MT, eps,
                                      warp + 4 * a, lane, (warp + 4 * a) < tile_rows);
      }
    }

    // Order matters: the proxy fence publishes only writes that are complete
    // when it executes.  Drain the cp.async group(s) consumed by the next
    // WGMMA first, then fence (covering both the drained weight copies and
    // the generic-proxy activation stores), then join at the warpgroup
    // barrier.  A fence issued before the wait would leave any copy that
    // completes after it unpublished to the async proxy.
    if constexpr (WeightPipeW1) {
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");

    if constexpr (PipelineMMA) {
      float partial0[NACC];
      float partial1[NACC];
#pragma unroll
      for (int i = 0; i < NACC; ++i) {
        partial0[i] = 0.0f;
        partial1[i] = 0.0f;
      }
      issue_mwg_wgmma<MT, HoistPipelineDesc>(partial0, my_w, my_q);
      issue_mwg_wgmma<MT, HoistPipelineDesc>(partial1, my_w + WBYTES, my_q + QBYTES);
      // Form both rescale vectors while the tensor groups are in flight;
      // the WGMMA drains otherwise hold these shared-scale loads behind the
      // wait instruction's memory clobber.
      const float ws0 = wsp[kc];
      const float ws1 = wsp[kc + 1];
      float sc0[ATOMS * 2];
      float sc1[ATOMS * 2];
#pragma unroll
      for (int j = 0; j < ATOMS; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int token = 8 * j + 2 * lane_in_group + r;
          sc0[2 * j + r] = my_s[token] * ws0;
          sc1[2 * j + r] = my_s[MT + token] * ws1;
        }
      }
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
      for (int i = 0; i < NACC; ++i)
        accum[i] = fmaf(partial0[i], sc0[2 * (i >> 2) + (i & 1)], accum[i]);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
      for (int i = 0; i < NACC; ++i)
        accum[i] = fmaf(partial1[i], sc1[2 * (i >> 2) + (i & 1)], accum[i]);
    } else {
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        if constexpr (WeightPipeW1) {
          if (u > 0) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            // Publish the just-drained second weight stage to the async
            // proxy before the barrier hands it to WGMMA.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
          }
        }
        float partial[NACC];
#pragma unroll
        for (int i = 0; i < NACC; ++i) partial[i] = 0.0f;
        issue_mwg_wgmma<MT, FixedHalfW1 || SparseM1 || (MT == 16 && NWG == 8)>(
            partial, my_w + u * WBYTES, my_q + u * QBYTES);
        const int kk = kc + u;
        float prefetched_ws = 0.0f;
        if constexpr (WeightPipeW1 || PrefetchScaleW6) prefetched_ws = wsp[kk];
        if constexpr (SparseM1) {
          // M=1 only observes the row-zero fragments.  Compile out the padded
          // row scale reads and FMAs instead of guarding each fragment at run
          // time; WGMMA still sees the required zero-padded 8-row tile.  The
          // single scale read is cheap, so keep the original drain ordering.
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
          const float ws = (WeightPipeW1 || PrefetchScaleW6)
                               ? prefetched_ws
                               : wsp[ExactOneRound ? kk : (kk < Sk ? kk : Sk - 1)];
          if (lane_in_group == 0) {
            const float scale = my_s[u * MT] * ws;
            accum[0] = fmaf(partial[0], scale, accum[0]);
            accum[2] = fmaf(partial[2], scale, accum[2]);
          }
        } else {
          // Pre-drain rescale hoist: the drain's memory clobber would hold
          // these shared activation-scale loads behind the tensor-core wait.
          const float ws = (WeightPipeW1 || PrefetchScaleW6)
                               ? prefetched_ws
                               : wsp[ExactOneRound ? kk : (kk < Sk ? kk : Sk - 1)];
          const float* const cs = my_s + u * MT;
          float sc_e[ATOMS * 2];
#pragma unroll
          for (int j = 0; j < ATOMS; ++j)
#pragma unroll
            for (int r = 0; r < 2; ++r) sc_e[2 * j + r] = cs[8 * j + 2 * lane_in_group + r] * ws;
          asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
          for (int i = 0; i < NACC; ++i)
            accum[i] = fmaf(partial[i], sc_e[2 * (i >> 2) + (i & 1)], accum[i]);
        }
      }
    }
    if constexpr (!ExactOneRound) {
      if (kc + KU < kb_end) asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
    }
  }

  // Cross-warpgroup reduction through each warpgroup's dead weight scratch.
  // wgmma.wait_group 0 has completed all async reads, so every warpgroup can
  // overwrite its private tile without a CTA-wide pre-store barrier.
  if constexpr (SparseM1) {
    // Only row zero is live: two fragment halves cover the 64 output columns.
    // Keep the required CTA rendezvous but reduce peer loads to the 64 lanes
    // that own observable results.
    float* const red = reinterpret_cast<float*>(my_w);
    if (lane_in_group == 0) {
      red[t] = accum[0];
      red[128 + t] = accum[2];
    }
    if (wg < 2)
      asm volatile("bar.sync 15, 768;" ::: "memory");
    else
      asm volatile("bar.arrive 15, 768;" ::: "memory");
    if (wg < 2 && lane_in_group == 0) {
      const int p = wg;
      float sum = 0.0f;
#pragma unroll
      for (int r = 0; r < NWG; ++r) {
        const float* const peer = reinterpret_cast<const float*>(wsmem + r * KU * WBYTES);
        sum += peer[p * 128 + t];
      }
      const int n = tile_n + warp * 16 + lane_group + 8 * p;
      if (n < N) out[static_cast<int64_t>(tile_m) * N + n] = __float2bfloat16_rn(sum);
    }
    return;
  }
  // Cross-warpgroup split-K partials stay FP32: BF16-packed partials of
  // opposite sign can cancel to a rounded residual far from the exact sum.
#pragma unroll
  for (int i = 0; i < NACC / 2; ++i)
    reinterpret_cast<float2*>(my_w)[i * 128 + t] = make_float2(accum[2 * i], accum[2 * i + 1]);
  if constexpr (MT == 8 && NWG == 6 && KU == 2 && PipelineMMA) {
    if (wg < 2)
      asm volatile("bar.sync 15, 768;" ::: "memory");
    else
      asm volatile("bar.arrive 15, 768;" ::: "memory");
  } else {
    __syncthreads();
  }
  for (int p = wg; p < NACC / 2; p += NWG) {
    float s0, s1;
    if constexpr (NWG == 6 && !SparseM1) {
      const float2* const peer0 = reinterpret_cast<const float2*>(wsmem);
      const float2* const peer1 = reinterpret_cast<const float2*>(wsmem + KU * WBYTES);
      float2 sum0 = peer0[p * 128 + t];
      float2 sum1 = peer1[p * 128 + t];
#pragma unroll
      for (int r = 2; r < NWG; r += 2) {
        const float2* const peer_even = reinterpret_cast<const float2*>(wsmem + r * KU * WBYTES);
        const float2* const peer_odd =
            reinterpret_cast<const float2*>(wsmem + (r + 1) * KU * WBYTES);
        const float2 v0 = peer_even[p * 128 + t];
        const float2 v1 = peer_odd[p * 128 + t];
        sum0.x += v0.x;
        sum0.y += v0.y;
        sum1.x += v1.x;
        sum1.y += v1.y;
      }
      s0 = sum0.x + sum1.x;
      s1 = sum0.y + sum1.y;
    } else {
      s0 = 0.0f;
      s1 = 0.0f;
#pragma unroll
      for (int r = 0; r < NWG; ++r) {
        const float2* const peer_red = reinterpret_cast<const float2*>(wsmem + r * KU * WBYTES);
        const float2 v = peer_red[p * 128 + t];
        s0 += v.x;
        s1 += v.y;
      }
    }
    const int n = tile_n + warp * 16 + lane_group + 8 * (p & 1);
    const int token = 8 * (p >> 1) + 2 * lane_in_group;
    if (DenseM || n < N) {
      const int64_t bo = static_cast<int64_t>(tile_m + token) * N + n;
      if (DenseM || token < tile_rows) out[bo] = __float2bfloat16_rn(s0);
      if (DenseM || token + 1 < tile_rows) out[bo + N] = __float2bfloat16_rn(s1);
    }
  }
}

}  // namespace mwg
