#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

#include "fp8bs_gemm_decodem_sm90.h"

namespace ffi = tvm::ffi;

template <class T>
static inline T* tensor_ptr(ffi::TensorView t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}

__device__ __forceinline__ uint32_t fp8x4(float x0, float x1, float x2, float x3) {
  uint16_t p01, p23;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p01) : "f"(x0), "f"(x1));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(p23) : "f"(x2), "f"(x3));
  return static_cast<uint32_t>(p01) | (static_cast<uint32_t>(p23) << 16);
}

__device__ __forceinline__ uint16_t fp8x2(float x0, float x1) {
  uint16_t result;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(result) : "f"(x0), "f"(x1));
  return result;
}

__device__ __forceinline__ void mma_fp8_m16n8k32(float& d0, float& d1, float& d2, float& d3,
                                                 uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One CTA computes a 16x64 output tile.  The two warps cooperatively quantize
// a 16x128 activation tile, then each warp computes a 16x32 output tile.
__global__ __launch_bounds__(64, 2) void fp8_blockscale_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale, float eps, __nv_bfloat16* __restrict__ out, int M,
    int N, int K) {
  __shared__ __align__(16) uint8_t aq[16 * 128];
  __shared__ float activation_scale[16];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int tile_m = static_cast<int>(blockIdx.y) * 16;
  const int tile_n = static_cast<int>(blockIdx.x) * 64;
  const int quant_row_local = warp * 8 + lane_group;
  const int quant_row = tile_m + quant_row_local;
  const int Sk = K >> 7;

  float accum[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int r = 0; r < 4; ++r) accum[j][r] = 0.0f;
  }

  for (int kb = 0; kb < Sk; ++kb) {
    const int kbase = kb * 128;
    float local_max = 0.0f;
    if (quant_row < M) {
      const __nv_bfloat16* xp =
          x + static_cast<int64_t>(quant_row) * K + kbase + lane_in_group * 32;
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        local_max = fmaxf(local_max, fabsf(__bfloat162float(xp[i])));
      }
    }
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2));
    const float scale = fmaxf(local_max, eps) * (1.0f / 448.0f);
    const float inv_scale = 1.0f / scale;
    if (lane_in_group == 0) activation_scale[quant_row_local] = scale;

    uint32_t* aq_dst = reinterpret_cast<uint32_t*>(aq + quant_row_local * 128 + lane_in_group * 32);
    if (quant_row < M) {
      const __nv_bfloat16* xp =
          x + static_cast<int64_t>(quant_row) * K + kbase + lane_in_group * 32;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float x0 = __bfloat162float(xp[i * 4 + 0]) * inv_scale;
        float x1 = __bfloat162float(xp[i * 4 + 1]) * inv_scale;
        float x2 = __bfloat162float(xp[i * 4 + 2]) * inv_scale;
        float x3 = __bfloat162float(xp[i * 4 + 3]) * inv_scale;
        aq_dst[i] = fp8x4(x0, x1, x2, x3);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) aq_dst[i] = 0;
    }
    __syncthreads();

    float partial[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int r = 0; r < 4; ++r) partial[j][r] = 0.0f;
    }

#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int ko = ks * 32 + lane_in_group * 4;
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(aq + lane_group * 128 + ko);
      const uint32_t a1 = *reinterpret_cast<const uint32_t*>(aq + (lane_group + 8) * 128 + ko);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(aq + lane_group * 128 + ko + 16);
      const uint32_t a3 = *reinterpret_cast<const uint32_t*>(aq + (lane_group + 8) * 128 + ko + 16);

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int n = tile_n + warp * 32 + j * 8 + lane_group;
        uint32_t b0 = 0, b1 = 0;
        if (n < N) {
          const uint8_t* wp =
              weight + static_cast<int64_t>(n) * K + kbase + ks * 32 + lane_in_group * 4;
          b0 = *reinterpret_cast<const uint32_t*>(wp);
          b1 = *reinterpret_cast<const uint32_t*>(wp + 16);
        }
        mma_fp8_m16n8k32(partial[j][0], partial[j][1], partial[j][2], partial[j][3], a0, a1, a2, a3,
                         b0, b1);
      }
    }

    const float ws = weight_scale[(tile_n >> 7) * Sk + kb];
    const float s0 = activation_scale[lane_group] * ws;
    const float s1 = activation_scale[lane_group + 8] * ws;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      accum[j][0] = fmaf(partial[j][0], s0, accum[j][0]);
      accum[j][1] = fmaf(partial[j][1], s0, accum[j][1]);
      accum[j][2] = fmaf(partial[j][2], s1, accum[j][2]);
      accum[j][3] = fmaf(partial[j][3], s1, accum[j][3]);
    }
    if (kb + 1 < Sk) __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = tile_m + lane_group + ((r >= 2) ? 8 : 0);
      const int col = tile_n + warp * 32 + j * 8 + lane_in_group * 2 + (r & 1);
      if (row < M && col < N) {
        out[static_cast<int64_t>(row) * N + col] = __float2bfloat16_rn(accum[j][r]);
      }
    }
  }
}

// Split-K partials traverse DSM, so pack adjacent output rows to bf16x2.
// The final output is bf16 and the campaign tolerance already includes FP8
// quantization noise; this halves both DSM bytes and peer load transactions.
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u));
}

__device__ __forceinline__ uint64_t wgmma_b_desc(const uint8_t* ptr) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return ((static_cast<uint64_t>(addr) & 0x3ffffull) >> 4) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 62);
}

template <bool Add>
__device__ __forceinline__ void wgmma_m64n8(float (&d)[4], const uint32_t (&a)[4], uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, %9, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "n"(Add ? 1 : 0));
}

template <bool Add>
__device__ __forceinline__ void wgmma_m64n16(float (&d)[8], const uint32_t (&a)[4], uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, %13, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "n"(Add ? 1 : 0));
}

template <bool Add>
__device__ __forceinline__ void wgmma_m64n24(float (&d)[12], const uint32_t (&a)[4], uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, "
      "{%12,%13,%14,%15}, %16, %17, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "n"(Add ? 1 : 0));
}

template <bool Add>
__device__ __forceinline__ void wgmma_ss_m64n8(float (&d)[4], uint64_t a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3}, %4, %5, %6, 1, 1;"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(a), "l"(b), "n"(Add ? 1 : 0));
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

template <int N>
__device__ __forceinline__ void wgmma_wait_n() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

template <int MT>
__device__ __forceinline__ void issue_ku_wgmma_group(float (&d)[MT / 2], uint64_t da, uint64_t db) {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const uint64_t a = da + static_cast<uint64_t>(ks * 2);
    const uint64_t b = db + static_cast<uint64_t>(ks * 2);
    if constexpr (MT == 8) {
      if (ks == 0)
        wgmma_ss_m64n8<false>(d, a, b);
      else
        wgmma_ss_m64n8<true>(d, a, b);
    } else if constexpr (MT == 16) {
      if (ks == 0)
        wgmma_ss_m64n16<false>(d, a, b);
      else
        wgmma_ss_m64n16<true>(d, a, b);
    } else {
      if (ks == 0)
        wgmma_ss_m64n24<false>(d, a, b);
      else
        wgmma_ss_m64n24<true>(d, a, b);
    }
  }
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int GROUP>
__device__ __forceinline__ void load_activation_row(const __nv_bfloat16* __restrict__ x,
                                                    uint32_t* __restrict__ raw, int row,
                                                    int subgroup_lane, int M, int K, int kbase) {
  static_assert(GROUP == 8 || GROUP == 16);
  constexpr int PAIRS = 64 / GROUP;
  const bool valid = row < M;
  const uint32_t* xp = reinterpret_cast<const uint32_t*>(x + static_cast<int64_t>(row) * K + kbase);
#pragma unroll
  for (int i = 0; i < PAIRS; ++i) {
    raw[i] = valid ? xp[i * GROUP + subgroup_lane] : 0;
  }
}

template <int GROUP>
__device__ __forceinline__ void quantize_activation_row_raw(const uint32_t* __restrict__ raw,
                                                            uint8_t* __restrict__ bq,
                                                            float* __restrict__ activation_scale,
                                                            float eps, int row, int subgroup_lane) {
  static_assert(GROUP == 8 || GROUP == 16);
  constexpr int PAIRS = 64 / GROUP;
  union Bf16Pair {
    uint32_t u32;
    __nv_bfloat162 bf16x2;
  };

  uint32_t local_absmax = 0;
#pragma unroll
  for (int i = 0; i < PAIRS; ++i) {
    const uint32_t abs0 = raw[i] & 0x7fffu;
    const uint32_t abs1 = (raw[i] >> 16) & 0x7fffu;
    const uint32_t pair_max = abs0 > abs1 ? abs0 : abs1;
    local_absmax = local_absmax > pair_max ? local_absmax : pair_max;
  }
#pragma unroll
  for (int delta = 1; delta < GROUP; delta <<= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, local_absmax, delta);
    local_absmax = local_absmax > other ? local_absmax : other;
  }

  float inv_scale = 0.0f;
  if (subgroup_lane == 0) {
    const float local_max = __uint_as_float(local_absmax << 16);
    const float scale = fmaxf(local_max, eps) * (1.0f / 448.0f);
    activation_scale[row] = scale;
    inv_scale = 1.0f / scale;
  }
  inv_scale = __shfl_sync(0xffffffffu, inv_scale, 0, GROUP);

  const int atom = row >> 3;
  const int row_in_atom = row & 7;
#pragma unroll
  for (int i = 0; i < PAIRS; ++i) {
    Bf16Pair pair;
    pair.u32 = raw[i];
    const float2 value = __bfloat1622float2(pair.bf16x2);
    const int logical = row_in_atom * 128 + 2 * (i * GROUP + subgroup_lane);
    const int physical = logical ^ ((logical & 0x380) >> 3);
    *reinterpret_cast<uint16_t*>(bq + atom * 1024 + physical) =
        fp8x2(value.x * inv_scale, value.y * inv_scale);
  }
}

template <int GROUP>
__device__ __forceinline__ void quantize_activation_row(const __nv_bfloat16* __restrict__ x,
                                                        uint8_t* __restrict__ bq,
                                                        float* __restrict__ activation_scale,
                                                        float eps, int row, int subgroup_lane,
                                                        int M, int K, int kbase) {
  uint32_t raw[64 / GROUP];
  load_activation_row<GROUP>(x, raw, row, subgroup_lane, M, K, kbase);
  quantize_activation_row_raw<GROUP>(raw, bq, activation_scale, eps, row, subgroup_lane);
}

__device__ __forceinline__ uint64_t make_l2_evict_last_policy() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

template <bool KeepX>
__device__ __forceinline__ uint4
load_activation_row_contiguous16(const __nv_bfloat16* __restrict__ x, int row, int subgroup_lane,
                                 int M, int K, int kbase, uint64_t cache_policy) {
  if (row >= M) return make_uint4(0, 0, 0, 0);
  const __nv_bfloat16* ptr = x + static_cast<int64_t>(row) * K + kbase + subgroup_lane * 8;
  if constexpr (KeepX) {
    uint4 value;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr), "l"(cache_policy));
    return value;
  } else {
    return *reinterpret_cast<const uint4*>(ptr);
  }
}

__device__ __forceinline__ uint4 load_activation_row_contiguous16_l1_last(
    const __nv_bfloat16* __restrict__ x, int row, int subgroup_lane, int M, int K, int kbase) {
  if (row >= M) return make_uint4(0, 0, 0, 0);
  const __nv_bfloat16* ptr = x + static_cast<int64_t>(row) * K + kbase + subgroup_lane * 8;
  uint4 value;
  asm volatile("ld.global.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr));
  return value;
}

template <bool BalancedAmax = false, bool FastAmax = false>
__device__ __forceinline__ void quantize_activation_row_contiguous16(
    uint4 raw, uint8_t* __restrict__ bq, float* __restrict__ activation_scale, float eps, int row,
    int subgroup_lane) {
  union Bf16Pair {
    uint32_t u32;
    __nv_bfloat162 bf16x2;
  };
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
  uint32_t local_absmax;
  constexpr int AMAX_SHIFT = FastAmax ? 0 : 16;
  if constexpr (FastAmax) {
    const uint32_t m0 = raw.x & 0x7fff7fffu;
    const uint32_t m1 = raw.y & 0x7fff7fffu;
    const uint32_t m2 = raw.z & 0x7fff7fffu;
    const uint32_t m3 = raw.w & 0x7fff7fffu;
    const uint32_t hi = max(max(m0, m1), max(m2, m3)) & 0xffff0000u;
    const uint32_t lo = max(max(m0 << 16, m1 << 16), max(m2 << 16, m3 << 16));
    local_absmax = max(hi, lo);
  } else if constexpr (BalancedAmax) {
    const uint32_t p0 = max(words[0] & 0x7fffu, (words[0] >> 16) & 0x7fffu);
    const uint32_t p1 = max(words[1] & 0x7fffu, (words[1] >> 16) & 0x7fffu);
    const uint32_t p2 = max(words[2] & 0x7fffu, (words[2] >> 16) & 0x7fffu);
    const uint32_t p3 = max(words[3] & 0x7fffu, (words[3] >> 16) & 0x7fffu);
    local_absmax = max(max(p0, p1), max(p2, p3));
  } else {
    local_absmax = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t abs0 = words[i] & 0x7fffu;
      const uint32_t abs1 = (words[i] >> 16) & 0x7fffu;
      const uint32_t pair_max = abs0 > abs1 ? abs0 : abs1;
      local_absmax = local_absmax > pair_max ? local_absmax : pair_max;
    }
  }
#pragma unroll
  for (int delta = 1; delta < 16; delta <<= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, local_absmax, delta);
    local_absmax = local_absmax > other ? local_absmax : other;
  }

  float inv_scale = 0.0f;
  if (subgroup_lane == 0) {
    const float local_max = __uint_as_float(local_absmax << AMAX_SHIFT);
    const float scale = fmaxf(local_max, eps) * (1.0f / 448.0f);
    activation_scale[row] = scale;
    inv_scale = 1.0f / scale;
  }
  inv_scale = __shfl_sync(0xffffffffu, inv_scale, 0, 16);

  uint16_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    Bf16Pair pair;
    pair.u32 = words[i];
    const float2 value = __bfloat1622float2(pair.bf16x2);
    packed[i] = fp8x2(value.x * inv_scale, value.y * inv_scale);
  }
  const uint2 result =
      make_uint2(static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16),
                 static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16));
  const int atom = row >> 3;
  const int row_in_atom = row & 7;
  const int logical = row_in_atom * 128 + subgroup_lane * 8;
  const int physical = logical ^ ((logical & 0x380) >> 3);
  *reinterpret_cast<uint2*>(bq + atom * 1024 + physical) = result;
}

// Eight lanes cover one full 128-element row with two 16-byte vectors each.
// This removes one shuffle stage from the amax reduction and combines each
// lane's two shared writes into one 16-byte transaction.
__device__ __forceinline__ void quantize_activation_row_contiguous8_fast(
    uint4 raw0, uint4 raw1, uint8_t* __restrict__ bq, float* __restrict__ activation_scale,
    float eps, int row, int subgroup_lane) {
  union Bf16Pair {
    uint32_t u32;
    __nv_bfloat162 bf16x2;
  };
  const uint32_t words[8] = {raw0.x, raw0.y, raw0.z, raw0.w, raw1.x, raw1.y, raw1.z, raw1.w};
  uint32_t m[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) m[i] = words[i] & 0x7fff7fffu;
  const uint32_t hi =
      max(max(max(m[0], m[1]), max(m[2], m[3])), max(max(m[4], m[5]), max(m[6], m[7]))) &
      0xffff0000u;
  const uint32_t lo = max(max(max(m[0] << 16, m[1] << 16), max(m[2] << 16, m[3] << 16)),
                          max(max(m[4] << 16, m[5] << 16), max(m[6] << 16, m[7] << 16)));
  uint32_t local_absmax = max(hi, lo);
#pragma unroll
  for (int delta = 1; delta < 8; delta <<= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, local_absmax, delta, 8);
    local_absmax = max(local_absmax, other);
  }

  float inv_scale = 0.0f;
  if (subgroup_lane == 0) {
    const float local_max = __uint_as_float(local_absmax);
    const float scale = fmaxf(local_max, eps) * (1.0f / 448.0f);
    activation_scale[row] = scale;
    inv_scale = 1.0f / scale;
  }
  inv_scale = __shfl_sync(0xffffffffu, inv_scale, 0, 8);

  uint16_t packed[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    Bf16Pair pair;
    pair.u32 = words[i];
    const float2 value = __bfloat1622float2(pair.bf16x2);
    packed[i] = fp8x2(value.x * inv_scale, value.y * inv_scale);
  }
  const uint4 result =
      make_uint4(static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16),
                 static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16),
                 static_cast<uint32_t>(packed[4]) | (static_cast<uint32_t>(packed[5]) << 16),
                 static_cast<uint32_t>(packed[6]) | (static_cast<uint32_t>(packed[7]) << 16));
  const int atom = row >> 3;
  const int row_in_atom = row & 7;
  const int logical = row_in_atom * 128 + subgroup_lane * 16;
  const int physical = logical ^ ((logical & 0x380) >> 3);
  *reinterpret_cast<uint4*>(bq + atom * 1024 + physical) = result;
}

template <bool Streaming>
__device__ __forceinline__ uint4 load_weight_vector(const uint8_t* __restrict__ ptr) {
  if constexpr (Streaming) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr));
    return value;
  } else {
    return *reinterpret_cast<const uint4*>(ptr);
  }
}

__device__ __forceinline__ void cp_async_weight16(void* __restrict__ dst,
                                                  const void* __restrict__ src, bool valid) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if (valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;" ::"r"(smem), "l"(src) : "memory");
  }
}

__device__ __forceinline__ void cp_async_activation16(void* __restrict__ dst,
                                                      const void* __restrict__ src, bool valid) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if (valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;" ::"r"(smem), "l"(src) : "memory");
  }
}

__device__ __forceinline__ uint32_t cluster_id_x() {
  uint32_t id;
  asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(id));
  return id;
}

// cooperative_groups::map_shared_rank returns a generic pointer, which can
// leave the store on the generic ST.E path. Form the cluster shared address
// explicitly so owner-push publication lowers to STS.
__device__ __forceinline__ uint32_t dsm_addr(const void* p, int rank) {
  const uint32_t local = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  uint32_t remote;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank));
  return remote;
}

__device__ __forceinline__ void dsm_st_u32(uint32_t addr, uint32_t value) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(value) : "memory");
}

__device__ __forceinline__ void dsm_st_f32(uint32_t addr, float value) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(value) : "memory");
}

__device__ __forceinline__ void wgmma_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

// Publish freshly initialized transaction barriers without imposing a full
// release fence on unrelated shared state.
__device__ __forceinline__ void mbar_init_publish() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

// Asynchronous owner-push publication.  st.async signals completion through
// the destination CTA's mbarrier, so publishing CTAs need not drain their
// remote stores at a full-cluster release rendezvous.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect(uint32_t addr, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t addr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P%=;\n"
      "W%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P%=, [%0], %1;\n"
      "@!P%= bra W%=;\n"
      "}\n" ::"r"(addr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void dsm_st_async_u32(uint32_t addr, uint32_t value, uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
      "[%0], %1, [%2];" ::"r"(addr),
      "r"(value), "r"(mbar)
      : "memory");
}

// Flat owner-push split-K reduction.  Each source rank distributes its
// accumulator elements across owner CTA mailboxes.  After one cluster-wide
// rendezvous, every owner reduces only local shared memory and writes its
// disjoint flat output slice.
template <int MT, int SPLIT, bool Pack>
struct SplitkMailbox {
  static constexpr int NP = Pack ? MT / 4 : MT / 2;
  static constexpr int TOT = NP * 128;
  static constexpr int CH = (TOT + SPLIT - 1) / SPLIT;
  static constexpr int WORDS = SPLIT * CH;
};

template <int MT, int SPLIT, bool Pack, bool TwoChain = false>
__device__ __forceinline__ void splitk_flat_push_reduce(
    const cooperative_groups::cluster_group& cluster, uint32_t* box, const float (&accum)[MT / 2],
    __nv_bfloat16* __restrict__ out, int split, int tid, int tile_m, int tile_n, int tile_rows,
    int N) {
  using B = SplitkMailbox<MT, SPLIT, Pack>;
  float* const fbox = reinterpret_cast<float*>(box);
#pragma unroll
  for (int p = 0; p < B::NP; ++p) {
    const int e = p * 128 + tid;
    const int owner = e / B::CH;
    const int slot = split * B::CH + (e - owner * B::CH);
    if constexpr (Pack) {
      dsm_st_u32(dsm_addr(box + slot, owner), pack_bf16x2(accum[2 * p], accum[2 * p + 1]));
    } else {
      dsm_st_f32(dsm_addr(fbox + slot, owner), accum[p]);
    }
  }
  cluster.sync();
  for (int j = tid; j < B::CH; j += 128) {
    const int e = split * B::CH + j;
    if (e >= B::TOT) break;
    const int p = e >> 7;
    const int t = e & 127;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int lgrp = lane >> 2;
    const int lsub = lane & 3;
    if constexpr (Pack) {
      float s0;
      float s1;
      if constexpr (TwoChain) {
        float s0_even = 0.0f;
        float s1_even = 0.0f;
        float s0_odd = 0.0f;
        float s1_odd = 0.0f;
#pragma unroll
        for (int r = 0; r + 1 < SPLIT; r += 2) {
          const float2 ve = unpack_bf16x2(box[r * B::CH + j]);
          const float2 vo = unpack_bf16x2(box[(r + 1) * B::CH + j]);
          s0_even += ve.x;
          s1_even += ve.y;
          s0_odd += vo.x;
          s1_odd += vo.y;
        }
        if constexpr ((SPLIT & 1) != 0) {
          const float2 v = unpack_bf16x2(box[(SPLIT - 1) * B::CH + j]);
          s0_even += v.x;
          s1_even += v.y;
        }
        s0 = s0_even + s0_odd;
        s1 = s1_even + s1_odd;
      } else {
        s0 = 0.0f;
        s1 = 0.0f;
#pragma unroll
        for (int r = 0; r < SPLIT; ++r) {
          const float2 v = unpack_bf16x2(box[r * B::CH + j]);
          s0 += v.x;
          s1 += v.y;
        }
      }
      const int n = tile_n + warp * 16 + lgrp + 8 * (p & 1);
      const int token = 8 * (p >> 1) + 2 * lsub;
      if (n < N) {
        const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
        if (token < tile_rows) out[base] = __float2bfloat16_rn(s0);
        if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(s1);
      }
    } else {
      float v = 0.0f;
#pragma unroll
      for (int r = 0; r < SPLIT; ++r) v += fbox[r * B::CH + j];
      const int n = tile_n + warp * 16 + lgrp + 8 * ((p >> 1) & 1);
      const int token = 8 * (p >> 2) + 2 * lsub + (p & 1);
      if (token < tile_rows && n < N)
        out[static_cast<int64_t>(tile_m + token) * N + n] = __float2bfloat16_rn(v);
    }
  }
}

// Exact M12/MT16 flat mailbox: omit the four padded rows while keeping every
// active owner mailbox dense and every per-rank slice 32 words wide.
template <bool Async = false>
__device__ __forceinline__ void splitk_flat_push_reduce_m12_s16(
    const cooperative_groups::cluster_group& cluster, uint32_t* box, const float (&accum)[8],
    __nv_bfloat16* __restrict__ out, int split, int tid, int tile_m, int tile_n, int N,
    uint64_t* rbar = nullptr) {
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lgrp = lane >> 2;
  const int lsub = lane & 3;
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    const int slot = split * 32 + lane;
    const int owner = 4 * p + warp;
    const uint32_t value = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
    if constexpr (Async)
      dsm_st_async_u32(dsm_addr(box + slot, owner), value, dsm_addr(rbar, owner));
    else {
      uint32_t* const remote = cluster.map_shared_rank(box, owner);
      remote[slot] = value;
    }
  }
  if (lsub < 2) {
    const int c = 2 * lgrp + lsub;
#pragma unroll
    for (int p = 2; p < 4; ++p) {
      const int slot = split * 32 + 16 * (p - 2) + c;
      const int owner = 8 + warp;
      const uint32_t value = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
      if constexpr (Async)
        dsm_st_async_u32(dsm_addr(box + slot, owner), value, dsm_addr(rbar, owner));
      else {
        uint32_t* const remote = cluster.map_shared_rank(box, owner);
        remote[slot] = value;
      }
    }
  }
  if constexpr (Async) {
    if (split < 12 && tid < 32) mbar_wait(smem_addr(rbar), 0);
  } else {
    // All sixteen ranks have published.  Only the twelve owner ranks consume
    // mailboxes, so let non-owners retire after a release arrival.
    wgmma_cluster_arrive_release();
    if (split < 12) wgmma_cluster_wait_acquire();
  }
  if (split < 12 && tid < 32) {
    float2 sum0 = unpack_bf16x2(box[tid]);
    float2 sum1 = unpack_bf16x2(box[32 + tid]);
    float2 sum2 = unpack_bf16x2(box[64 + tid]);
    float2 sum3 = unpack_bf16x2(box[96 + tid]);
#pragma unroll
    for (int r = 4; r < 16; r += 4) {
      const float2 v0 = unpack_bf16x2(box[r * 32 + tid]);
      const float2 v1 = unpack_bf16x2(box[(r + 1) * 32 + tid]);
      const float2 v2 = unpack_bf16x2(box[(r + 2) * 32 + tid]);
      const float2 v3 = unpack_bf16x2(box[(r + 3) * 32 + tid]);
      sum0.x += v0.x;
      sum0.y += v0.y;
      sum1.x += v1.x;
      sum1.y += v1.y;
      sum2.x += v2.x;
      sum2.y += v2.y;
      sum3.x += v3.x;
      sum3.y += v3.y;
    }
    const float s0 = (sum0.x + sum2.x) + (sum1.x + sum3.x);
    const float s1 = (sum0.y + sum2.y) + (sum1.y + sum3.y);
    int p;
    int out_warp;
    int out_lgrp;
    int out_lsub;
    if (split < 8) {
      p = split >> 2;
      out_warp = split & 3;
      out_lgrp = tid >> 2;
      out_lsub = tid & 3;
    } else {
      p = 2 + (tid >> 4);
      out_warp = split - 8;
      const int c = tid & 15;
      out_lgrp = c >> 1;
      out_lsub = c & 1;
    }
    const int n = tile_n + out_warp * 16 + out_lgrp + 8 * (p & 1);
    const int token = 8 * (p >> 1) + 2 * out_lsub;
    const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
    out[base] = __float2bfloat16_rn(s0);
    out[base + N] = __float2bfloat16_rn(s1);
  }
}

// Exact M4/MT8 flat mailbox: only two lane-in-group values are live, so pack
// each warp's valid pairs into one dense 16-word owner slice.
template <bool Async = false>
__device__ __forceinline__ void splitk_flat_push_reduce_m4_s16(
    const cooperative_groups::cluster_group& cluster, uint32_t* box, const float (&accum)[4],
    __nv_bfloat16* __restrict__ out, int split, int tid, int tile_m, int tile_n, int N,
    uint64_t* rbar = nullptr) {
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lgrp = lane >> 2;
  const int lsub = lane & 3;
  if (lsub < 2) {
    const int c = 2 * lgrp + lsub;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const uint32_t value = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
      if constexpr (Async) {
        dsm_st_async_u32(dsm_addr(box + split * 16 + c, 4 * p + warp), value,
                         dsm_addr(rbar, 4 * p + warp));
      } else {
        uint32_t* const remote = cluster.map_shared_rank(box, 4 * p + warp);
        remote[split * 16 + c] = value;
      }
    }
  }
  if constexpr (Async) {
    if (split < 8 && tid < 16) mbar_wait(smem_addr(rbar), 0);
  } else {
    // Only the eight owner ranks consume the compact M4 mailboxes.  A release
    // arrival lets the other half of the cluster retire immediately.
    wgmma_cluster_arrive_release();
    if (split < 8) wgmma_cluster_wait_acquire();
  }
  if (split < 8 && tid < 16) {
    float2 sum0 = unpack_bf16x2(box[tid]);
    float2 sum1 = unpack_bf16x2(box[16 + tid]);
    float2 sum2 = unpack_bf16x2(box[32 + tid]);
    float2 sum3 = unpack_bf16x2(box[48 + tid]);
#pragma unroll
    for (int r = 4; r < 16; r += 4) {
      const float2 v0 = unpack_bf16x2(box[r * 16 + tid]);
      const float2 v1 = unpack_bf16x2(box[(r + 1) * 16 + tid]);
      const float2 v2 = unpack_bf16x2(box[(r + 2) * 16 + tid]);
      const float2 v3 = unpack_bf16x2(box[(r + 3) * 16 + tid]);
      sum0.x += v0.x;
      sum0.y += v0.y;
      sum1.x += v1.x;
      sum1.y += v1.y;
      sum2.x += v2.x;
      sum2.y += v2.y;
      sum3.x += v3.x;
      sum3.y += v3.y;
    }
    const float s0 = (sum0.x + sum2.x) + (sum1.x + sum3.x);
    const float s1 = (sum0.y + sum2.y) + (sum1.y + sum3.y);
    const int p = split >> 2;
    const int out_warp = split & 3;
    const int n = tile_n + out_warp * 16 + (tid >> 1) + 8 * p;
    const int token = 2 * (tid & 1);
    const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
    out[base] = __float2bfloat16_rn(s0);
    out[base + N] = __float2bfloat16_rn(s1);
  }
}

// Exact M2/MT8 split-8 mailbox: only lane-in-group zero owns live rows.
// Preserve two reducer CTAs and send each one 32 dense words per source rank.
template <bool Async = false>
__device__ __forceinline__ void splitk_push_reduce_m2_s8(
    const cooperative_groups::cluster_group& cluster, uint32_t* box, const float (&accum)[4],
    __nv_bfloat16* __restrict__ out, int split, int tid, int tile_m, int tile_n, int N,
    uint64_t* rbar = nullptr) {
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  if (lane_in_group == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int slot = split * 32 + warp * 8 + lane_group;
      const uint32_t value = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
      if constexpr (Async)
        dsm_st_async_u32(dsm_addr(box + slot, p), value, dsm_addr(rbar, p));
      else
        dsm_st_u32(dsm_addr(box + slot, p), value);
    }
  }
  if constexpr (Async) {
    if (split < 2 && tid < 32) mbar_wait(smem_addr(rbar), 0);
  } else {
    wgmma_cluster_arrive_release();
    if (split < 2) wgmma_cluster_wait_acquire();
  }
  if (split < 2 && tid < 32) {
    float2 sum0 = unpack_bf16x2(box[tid]);
    float2 sum1 = unpack_bf16x2(box[32 + tid]);
    float2 sum2 = unpack_bf16x2(box[64 + tid]);
    float2 sum3 = unpack_bf16x2(box[96 + tid]);
#pragma unroll
    for (int rank = 4; rank < 8; rank += 4) {
      const float2 v0 = unpack_bf16x2(box[rank * 32 + tid]);
      const float2 v1 = unpack_bf16x2(box[(rank + 1) * 32 + tid]);
      const float2 v2 = unpack_bf16x2(box[(rank + 2) * 32 + tid]);
      const float2 v3 = unpack_bf16x2(box[(rank + 3) * 32 + tid]);
      sum0.x += v0.x;
      sum0.y += v0.y;
      sum1.x += v1.x;
      sum1.y += v1.y;
      sum2.x += v2.x;
      sum2.y += v2.y;
      sum3.x += v3.x;
      sum3.y += v3.y;
    }
    const float s0 = (sum0.x + sum2.x) + (sum1.x + sum3.x);
    const float s1 = (sum0.y + sum2.y) + (sum1.y + sum3.y);
    const int out_warp = tid >> 3;
    const int n = tile_n + out_warp * 16 + (tid & 7) + 8 * split;
    // The compact split-8 helper is dispatched only for exact
    // M2/N7168/K2048, so every N64 output tile is complete.
    const int64_t base = static_cast<int64_t>(tile_m) * N + n;
    out[base] = __float2bfloat16_rn(s0);
    out[base + N] = __float2bfloat16_rn(s1);
  }
}

// Exact M2 mailbox for the split-14 KU4 route.  Only lane-in-group zero owns
// live rows, so preserve the existing two-owner topology while sending 32
// dense words per source rank instead of 128 padded words.
template <int SPLIT>
__device__ __forceinline__ void splitk_push_reduce_m2_ku(
    const cooperative_groups::cluster_group& cluster, uint32_t* box, const float (&accum)[4],
    __nv_bfloat16* __restrict__ out, int split, int tid, int tile_m, int tile_n, int N,
    uint64_t* rbar) {
  static_assert(SPLIT >= 8);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  if (lane_in_group == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int slot = split * 32 + warp * 8 + lane_group;
      dsm_st_async_u32(dsm_addr(box + slot, p), pack_bf16x2(accum[2 * p], accum[2 * p + 1]),
                       dsm_addr(rbar, p));
    }
  }
  if (split < 2) {
    mbar_wait(smem_addr(rbar), 0);
    if (tid < 32) {
      float2 sum0 = unpack_bf16x2(box[tid]);
      float2 sum1 = unpack_bf16x2(box[32 + tid]);
      float2 sum2 = unpack_bf16x2(box[64 + tid]);
      float2 sum3 = unpack_bf16x2(box[96 + tid]);
#pragma unroll
      for (int rank = 4; rank + 3 < SPLIT; rank += 4) {
        const float2 v0 = unpack_bf16x2(box[rank * 32 + tid]);
        const float2 v1 = unpack_bf16x2(box[(rank + 1) * 32 + tid]);
        const float2 v2 = unpack_bf16x2(box[(rank + 2) * 32 + tid]);
        const float2 v3 = unpack_bf16x2(box[(rank + 3) * 32 + tid]);
        sum0.x += v0.x;
        sum0.y += v0.y;
        sum1.x += v1.x;
        sum1.y += v1.y;
        sum2.x += v2.x;
        sum2.y += v2.y;
        sum3.x += v3.x;
        sum3.y += v3.y;
      }
      if constexpr ((SPLIT & 3) >= 1) {
        const float2 v = unpack_bf16x2(box[(SPLIT - (SPLIT & 3)) * 32 + tid]);
        sum0.x += v.x;
        sum0.y += v.y;
      }
      if constexpr ((SPLIT & 3) >= 2) {
        const float2 v = unpack_bf16x2(box[(SPLIT - (SPLIT & 3) + 1) * 32 + tid]);
        sum1.x += v.x;
        sum1.y += v.y;
      }
      const float s0 = (sum0.x + sum2.x) + (sum1.x + sum3.x);
      const float s1 = (sum0.y + sum2.y) + (sum1.y + sum3.y);
      const int out_warp = tid >> 3;
      const int n = tile_n + out_warp * 16 + (tid & 7) + 8 * split;
      if (n < N) {
        const int64_t base = static_cast<int64_t>(tile_m) * N + n;
        out[base] = __float2bfloat16_rn(s0);
        out[base + N] = __float2bfloat16_rn(s1);
      }
    }
  }
}

template <int MT>
__global__ __launch_bounds__(128, 2) void fp8_blockscale_wgmma_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale, float eps, __nv_bfloat16* __restrict__ out, int M,
    int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24);
  __shared__ __align__(1024) uint8_t wa[64 * 128];
  __shared__ __align__(1024) uint8_t bq[(MT / 8) * 1024];
  __shared__ float activation_scale[MT];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int tile_n = static_cast<int>(blockIdx.x) * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;

  float accum[MT / 2];
#pragma unroll
  for (int i = 0; i < MT / 2; ++i) accum[i] = 0.0f;

  for (int kb = 0; kb < Sk; ++kb) {
    const int kbase = kb * 128;
    if constexpr (MT == 8) {
      quantize_activation_row<16>(tile_x, bq, activation_scale, eps, tid >> 4, tid & 15, tile_rows,
                                  K, kbase);
    } else {
      quantize_activation_row<8>(tile_x, bq, activation_scale, eps, tid >> 3, tid & 7, tile_rows, K,
                                 kbase);
      if constexpr (MT == 24) {
        if (tid < 64) {
          quantize_activation_row<8>(tile_x, bq, activation_scale, eps, 16 + (tid >> 3), tid & 7,
                                     tile_rows, K, kbase);
        }
      }
    }

#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int seg = tid + it * 128;
      const int row = seg >> 3;
      const int kseg = seg & 7;
      const int atom = row >> 3;
      const int row_in_atom = row & 7;
      const int logical = row_in_atom * 128 + kseg * 16;
      const int physical = logical ^ ((logical & 0x380) >> 3);
      uint4 value = make_uint4(0, 0, 0, 0);
      if (tile_n + row < N) {
        value = *reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(tile_n + row) * K +
                                                kbase + kseg * 16);
      }
      *reinterpret_cast<uint4*>(wa + atom * 1024 + physical) = value;
    }
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                     : "memory");
    __syncthreads();

    float partial[MT / 2];
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) partial[i] = 0.0f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t desc_a = wgmma_b_desc(wa + ks * 32);
      const uint64_t desc_b = wgmma_b_desc(bq + ks * 32);
      if constexpr (MT == 8) {
        if (ks == 0)
          wgmma_ss_m64n8<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n8<true>(partial, desc_a, desc_b);
      } else if constexpr (MT == 16) {
        if (ks == 0)
          wgmma_ss_m64n16<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n16<true>(partial, desc_a, desc_b);
      } else {
        if (ks == 0)
          wgmma_ss_m64n24<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n24<true>(partial, desc_a, desc_b);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    const float ws = weight_scale[(tile_n >> 7) * Sk + kb];
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) {
      const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
      accum[i] = fmaf(partial[i], activation_scale[token] * ws, accum[i]);
    }
    if (kb + 1 < Sk) __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < MT / 2; ++i) {
    const int n = tile_n + warp * 16 + lane_group + 8 * ((i >> 1) & 1);
    const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
    if (n < N && token < tile_rows) {
      out[static_cast<int64_t>(tile_m + token) * N + n] = __float2bfloat16_rn(accum[i]);
    }
  }
}

template <int MT, int SPLIT, bool KeepX = false, bool PackMT8 = false, bool PushDSM = false,
          bool FlatPush = false, bool FlatPack = false, bool CompactM4 = false,
          bool CompactM2 = false, bool ScaleOpt = false, bool AsyncFlat = false,
          bool AsyncM2 = false, bool DenseN = false>
__global__ __launch_bounds__(128, 2) void __cluster_dims__(SPLIT, 1, 1)
    fp8_blockscale_wgmma_splitk_kernel(const __nv_bfloat16* __restrict__ x,
                                       const uint8_t* __restrict__ weight,
                                       const float* __restrict__ weight_scale, float eps,
                                       __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24);
  static_assert(!(PushDSM && FlatPush));
  static_assert(!CompactM4 || (MT == 8 && SPLIT == 16 && FlatPush && FlatPack));
  static_assert(!AsyncFlat || CompactM4);
  static_assert(!AsyncM2 || CompactM2);
  static_assert(!CompactM2 || (MT == 8 && SPLIT == 8 && PackMT8 && PushDSM));
  static_assert(!DenseN || CompactM2);
  __shared__ __align__(1024) uint8_t wa[64 * 128];
  __shared__ __align__(1024) uint8_t bq[(MT / 8) * 1024];
  __shared__ __align__(16) uint8_t raw_async[AsyncM2 ? MT * 256 : 16];
  __shared__ float activation_scale[MT];
  __shared__ uint32_t
      split_partial[FlatPush ? SplitkMailbox<MT, SPLIT, (MT >= 16 || PackMT8 || FlatPack)>::WORDS
                             : (MT / 2) * 128];
  __shared__ uint32_t push_partial[PushDSM ? (CompactM2 ? SPLIT * 32 : SPLIT * 128) : 1];
  __shared__ __align__(8) uint64_t rbar[1];

  const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  if constexpr (AsyncM2) {
    if (cluster.block_rank() < 2) {
      if (threadIdx.x == 0) {
        const uint32_t addr = smem_addr(&rbar[0]);
        mbar_init(addr, 1);
        mbar_arrive_expect(addr, SPLIT * 32 * 4);
      }
      if (threadIdx.x < 32) __syncwarp();
    }
    mbar_init_publish();
  } else if constexpr (AsyncFlat) {
    constexpr int NOWNER = 8;
    if (cluster.block_rank() < NOWNER) {
      if (threadIdx.x == 0) {
        const uint32_t addr = smem_addr(&rbar[0]);
        mbar_init(addr, 1);
        mbar_arrive_expect(addr, SPLIT * 16 * 4);
      }
      if (threadIdx.x < 32) __syncwarp();
    }
    mbar_init_publish();
  } else if constexpr (PushDSM || FlatPush) {
    wgmma_cluster_arrive_relaxed();
  }
  const int split = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  static_assert(SPLIT == 2 || SPLIT == 3 || SPLIT == 4 || SPLIT == 6 || SPLIT == 8 || SPLIT == 16);
  const int tile_n = (static_cast<int>(blockIdx.x) / SPLIT) * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;
  const int kb_begin = (Sk * split) / SPLIT;
  const int kb_end = (Sk * (split + 1)) / SPLIT;

  float accum[MT / 2];
#pragma unroll
  for (int i = 0; i < MT / 2; ++i) accum[i] = 0.0f;
  uint64_t x_cache_policy = 0;
  if constexpr (KeepX) x_cache_policy = make_l2_evict_last_policy();

  for (int kb = kb_begin; kb < kb_end; ++kb) {
    const int kbase = kb * 128;
    const int subgroup_lane = tid & 15;
    const int row0 = tid >> 4;
    uint4 raw0;
    if constexpr (AsyncM2) {
      const bool valid = row0 < tile_rows;
      const __nv_bfloat16* const src =
          valid ? tile_x + static_cast<int64_t>(row0) * K + kbase + subgroup_lane * 8 : tile_x;
      cp_async_activation16(raw_async + row0 * 256 + subgroup_lane * 16, src, valid);
      asm volatile("cp.async.commit_group;" ::: "memory");
    } else {
      raw0 = load_activation_row_contiguous16<KeepX>(tile_x, row0, subgroup_lane, tile_rows, K,
                                                     kbase, x_cache_policy);
    }
    uint4 raw1;
    uint4 raw2;
    if constexpr (MT >= 16) {
      raw1 = load_activation_row_contiguous16<KeepX>(tile_x, 8 + row0, subgroup_lane, tile_rows, K,
                                                     kbase, x_cache_policy);
    }
    if constexpr (MT == 24) {
      raw2 = load_activation_row_contiguous16<KeepX>(tile_x, 16 + row0, subgroup_lane, tile_rows, K,
                                                     kbase, x_cache_policy);
    }
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int seg = tid + it * 128;
      const int row = seg >> 3;
      const int kseg = seg & 7;
      const int atom = row >> 3;
      const int row_in_atom = row & 7;
      const int logical = row_in_atom * 128 + kseg * 16;
      const int physical = logical ^ ((logical & 0x380) >> 3);
      const bool valid = DenseN || tile_n + row < N;
      const uint8_t* src =
          valid ? weight + static_cast<int64_t>(tile_n + row) * K + kbase + kseg * 16 : weight;
      if constexpr (PackMT8) {
        cp_async_weight16(wa + atom * 1024 + physical, src, valid);
      } else {
        uint4 value = make_uint4(0, 0, 0, 0);
        if (valid) value = load_weight_vector<KeepX>(src);
        *reinterpret_cast<uint4*>(wa + atom * 1024 + physical) = value;
      }
    }
    if constexpr (PackMT8) asm volatile("cp.async.commit_group;" ::: "memory");

    if constexpr (AsyncM2) {
      asm volatile("cp.async.wait_group 1;" ::: "memory");
      raw0 = *reinterpret_cast<const uint4*>(raw_async + row0 * 256 + subgroup_lane * 16);
    }

    quantize_activation_row_contiguous16(raw0, bq, activation_scale, eps, row0, subgroup_lane);
    if constexpr (MT >= 16) {
      quantize_activation_row_contiguous16(raw1, bq, activation_scale, eps, 8 + row0,
                                           subgroup_lane);
    }
    if constexpr (MT == 24) {
      quantize_activation_row_contiguous16(raw2, bq, activation_scale, eps, 16 + row0,
                                           subgroup_lane);
    }
    if constexpr (PackMT8) asm volatile("cp.async.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                     : "memory");
    __syncthreads();

    float partial[MT / 2];
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) partial[i] = 0.0f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t desc_a0 = 0;
    uint64_t desc_b0 = 0;
    if constexpr (CompactM4 || CompactM2) {
      desc_a0 = wgmma_b_desc(wa);
      desc_b0 = wgmma_b_desc(bq);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t desc_a = (CompactM4 || CompactM2) ? desc_a0 + static_cast<uint64_t>(ks * 2)
                                                       : wgmma_b_desc(wa + ks * 32);
      const uint64_t desc_b = (CompactM4 || CompactM2) ? desc_b0 + static_cast<uint64_t>(ks * 2)
                                                       : wgmma_b_desc(bq + ks * 32);
      if constexpr (MT == 8) {
        if (ks == 0)
          wgmma_ss_m64n8<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n8<true>(partial, desc_a, desc_b);
      } else if constexpr (MT == 16) {
        if (ks == 0)
          wgmma_ss_m64n16<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n16<true>(partial, desc_a, desc_b);
      } else {
        if (ks == 0)
          wgmma_ss_m64n24<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n24<true>(partial, desc_a, desc_b);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    float scale_pre[MT / 4];
    if constexpr (ScaleOpt) {
      const float ws_pre = weight_scale[(tile_n >> 7) * Sk + kb];
#pragma unroll
      for (int atom = 0; atom < MT / 8; ++atom) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const int token = 8 * atom + 2 * lane_in_group + row;
          scale_pre[2 * atom + row] = activation_scale[token] * ws_pre;
        }
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    if constexpr (ScaleOpt) {
#pragma unroll
      for (int atom = 0; atom < MT / 8; ++atom) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const int i = 4 * atom + row;
          const float scale = scale_pre[2 * atom + row];
          accum[i] = fmaf(partial[i], scale, accum[i]);
          accum[i + 2] = fmaf(partial[i + 2], scale, accum[i + 2]);
        }
      }
    } else {
      const float ws = weight_scale[(tile_n >> 7) * Sk + kb];
#pragma unroll
      for (int i = 0; i < MT / 2; ++i) {
        const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
        accum[i] = fmaf(partial[i], activation_scale[token] * ws, accum[i]);
      }
    }
    if (kb + 1 < kb_end) __syncthreads();
  }

  if constexpr (FlatPush) {
    wgmma_cluster_wait_acquire();
    if constexpr (CompactM4) {
      splitk_flat_push_reduce_m4_s16<AsyncFlat>(cluster, split_partial, accum, out, split, tid,
                                                tile_m, tile_n, N, &rbar[0]);
    } else {
      splitk_flat_push_reduce<MT, SPLIT, (MT >= 16 || PackMT8 || FlatPack)>(
          cluster, split_partial, accum, out, split, tid, tile_m, tile_n, tile_rows, N);
    }
  } else if constexpr (PushDSM) {
    wgmma_cluster_wait_acquire();
    if constexpr (CompactM2) {
      splitk_push_reduce_m2_s8<AsyncM2>(cluster, push_partial, accum, out, split, tid, tile_m,
                                        tile_n, N, &rbar[0]);
    } else if constexpr (MT >= 16 || PackMT8) {
      constexpr int NPAIR = MT / 4;
      static_assert(NPAIR <= SPLIT);
#pragma unroll
      for (int p = 0; p < NPAIR; ++p) {
        uint32_t* const remote = cluster.map_shared_rank(push_partial + split * 128 + tid, p);
        *remote = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
      }
      cluster.sync();
      if (split < NPAIR) {
        float2 sum = unpack_bf16x2(push_partial[tid]);
#pragma unroll
        for (int rank = 1; rank < SPLIT; ++rank) {
          const float2 v = unpack_bf16x2(push_partial[rank * 128 + tid]);
          sum.x += v.x;
          sum.y += v.y;
        }
        const int n_local = warp * 16 + lane_group + 8 * (split & 1);
        const int token = 8 * (split >> 1) + 2 * lane_in_group;
        if (tile_n + n_local < N) {
          const int64_t base = static_cast<int64_t>(tile_m + token) * N + tile_n + n_local;
          if (token < tile_rows) out[base] = __float2bfloat16_rn(sum.x);
          if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(sum.y);
        }
      }
    } else {
      static_assert(MT == 8);
      constexpr int NACC = MT / 2;
      static_assert(NACC <= SPLIT);
      float* const push_fp = reinterpret_cast<float*>(push_partial);
#pragma unroll
      for (int i = 0; i < NACC; ++i) {
        float* const remote = cluster.map_shared_rank(push_fp + split * 128 + tid, i);
        *remote = accum[i];
      }
      cluster.sync();
      if (split < NACC) {
        float value = push_fp[tid];
#pragma unroll
        for (int rank = 1; rank < SPLIT; ++rank) value += push_fp[rank * 128 + tid];
        const int n_local = warp * 16 + lane_group + 8 * ((split >> 1) & 1);
        const int token = 8 * (split >> 2) + 2 * lane_in_group + (split & 1);
        if (token < tile_rows && tile_n + n_local < N) {
          out[static_cast<int64_t>(tile_m + token) * N + tile_n + n_local] =
              __float2bfloat16_rn(value);
        }
      }
    }
  } else if constexpr (MT >= 16) {
#pragma unroll
    for (int i = 0; i < MT / 4; ++i) {
      split_partial[i * 128 + tid] = pack_bf16x2(accum[2 * i], accum[2 * i + 1]);
    }
    cluster.sync();

    for (int p = split; p < MT / 4; p += SPLIT) {
      uint32_t* const q = split_partial + p * 128 + tid;
      float s0 = 0.0f;
      float s1 = 0.0f;
#pragma unroll
      for (int rank = 0; rank < SPLIT; ++rank) {
        const float2 v = unpack_bf16x2(*cluster.map_shared_rank(q, rank));
        s0 += v.x;
        s1 += v.y;
      }
      const int n_local = warp * 16 + lane_group + 8 * (p & 1);
      const int token = 8 * (p >> 1) + 2 * lane_in_group;
      if (tile_n + n_local < N) {
        const int64_t base = static_cast<int64_t>(tile_m + token) * N + tile_n + n_local;
        if (token < tile_rows) out[base] = __float2bfloat16_rn(s0);
        if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(s1);
      }
    }
    cluster.sync();
  } else if constexpr (PackMT8) {
#pragma unroll
    for (int i = 0; i < MT / 4; ++i) {
      split_partial[i * 128 + tid] = pack_bf16x2(accum[2 * i], accum[2 * i + 1]);
    }
    cluster.sync();

    uint32_t* remote_partial[SPLIT];
#pragma unroll
    for (int rank = 0; rank < SPLIT; ++rank) {
      remote_partial[rank] = cluster.map_shared_rank(split_partial, rank);
    }

    for (int p = split; p < MT / 4; p += SPLIT) {
      const int index = p * 128 + tid;
      float s0 = 0.0f;
      float s1 = 0.0f;
#pragma unroll
      for (int rank = 0; rank < SPLIT; ++rank) {
        const float2 v = unpack_bf16x2(remote_partial[rank][index]);
        s0 += v.x;
        s1 += v.y;
      }
      const int n_local = warp * 16 + lane_group + 8 * (p & 1);
      const int token = 8 * (p >> 1) + 2 * lane_in_group;
      if (tile_n + n_local < N) {
        const int64_t base = static_cast<int64_t>(tile_m + token) * N + tile_n + n_local;
        if (token < tile_rows) out[base] = __float2bfloat16_rn(s0);
        if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(s1);
      }
    }
    cluster.sync();
  } else {
    float* const fp = reinterpret_cast<float*>(split_partial);
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) {
      fp[i * 128 + tid] = accum[i];
    }
    cluster.sync();

    float* remote_partial[SPLIT];
#pragma unroll
    for (int rank = 0; rank < SPLIT; ++rank) {
      remote_partial[rank] = cluster.map_shared_rank(fp, rank);
    }

    for (int i = split; i < MT / 2; i += SPLIT) {
      const int index = i * 128 + tid;
      float value = 0.0f;
#pragma unroll
      for (int rank = 0; rank < SPLIT; ++rank) {
        value += remote_partial[rank][index];
      }
      const int n_local = warp * 16 + lane_group + 8 * ((i >> 1) & 1);
      const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
      if (token < tile_rows && tile_n + n_local < N) {
        out[static_cast<int64_t>(tile_m + token) * N + tile_n + n_local] =
            __float2bfloat16_rn(value);
      }
    }
    cluster.sync();
  }
}

template <int MT, int SPLIT>
__global__ __launch_bounds__(256, 2) void __cluster_dims__(SPLIT, 1, 1)
    fp8_blockscale_wgmma_splitk_2wg_kernel(const __nv_bfloat16* __restrict__ x,
                                           const uint8_t* __restrict__ weight,
                                           const float* __restrict__ weight_scale, float eps,
                                           __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24);
  static_assert(SPLIT == 2 || SPLIT == 3 || SPLIT == 4 || SPLIT == 6 || SPLIT == 8 || SPLIT == 16);
  __shared__ __align__(1024) uint8_t wa[2][64 * 128];
  __shared__ __align__(1024) uint8_t bq[(MT / 8) * 1024];
  __shared__ float activation_scale[MT];
  __shared__ float split_partial[MT * 128];

  const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  const int split = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wg_tid = tid & 127;
  const int warp = wg_tid >> 5;
  const int lane = wg_tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int tile_n = (static_cast<int>(blockIdx.x) / SPLIT) * 128;
  const int wg_tile_n = tile_n + wg * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;
  const int kb_begin = (Sk * split) / SPLIT;
  const int kb_end = (Sk * (split + 1)) / SPLIT;

  float accum[MT / 2];
#pragma unroll
  for (int i = 0; i < MT / 2; ++i) accum[i] = 0.0f;
  const uint64_t x_cache_policy = make_l2_evict_last_policy();

  for (int kb = kb_begin; kb < kb_end; ++kb) {
    const int kbase = kb * 128;
    const int subgroup_lane = wg_tid & 15;
    const int row0 = wg_tid >> 4;
    uint4 raw0;
    uint4 raw1;
    uint4 raw2;
    if (wg == 0) {
      raw0 = load_activation_row_contiguous16<true>(tile_x, row0, subgroup_lane, tile_rows, K,
                                                    kbase, x_cache_policy);
      if constexpr (MT >= 16) {
        raw1 = load_activation_row_contiguous16<true>(tile_x, 8 + row0, subgroup_lane, tile_rows, K,
                                                      kbase, x_cache_policy);
      }
      if constexpr (MT == 24) {
        raw2 = load_activation_row_contiguous16<true>(tile_x, 16 + row0, subgroup_lane, tile_rows,
                                                      K, kbase, x_cache_policy);
      }
    }

#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int seg = wg_tid + it * 128;
      const int row = seg >> 3;
      const int kseg = seg & 7;
      const int atom = row >> 3;
      const int row_in_atom = row & 7;
      const int logical = row_in_atom * 128 + kseg * 16;
      const int physical = logical ^ ((logical & 0x380) >> 3);
      uint4 value = make_uint4(0, 0, 0, 0);
      if (wg_tile_n + row < N) {
        value = *reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(wg_tile_n + row) * K +
                                                kbase + kseg * 16);
      }
      *reinterpret_cast<uint4*>(wa[wg] + atom * 1024 + physical) = value;
    }

    if (wg == 0) {
      quantize_activation_row_contiguous16(raw0, bq, activation_scale, eps, row0, subgroup_lane);
      if constexpr (MT >= 16) {
        quantize_activation_row_contiguous16(raw1, bq, activation_scale, eps, 8 + row0,
                                             subgroup_lane);
      }
      if constexpr (MT == 24) {
        quantize_activation_row_contiguous16(raw2, bq, activation_scale, eps, 16 + row0,
                                             subgroup_lane);
      }
    }
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                     : "memory");
    __syncthreads();

    float partial[MT / 2];
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) partial[i] = 0.0f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t desc_a = wgmma_b_desc(wa[wg] + ks * 32);
      const uint64_t desc_b = wgmma_b_desc(bq + ks * 32);
      if constexpr (MT == 8) {
        if (ks == 0)
          wgmma_ss_m64n8<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n8<true>(partial, desc_a, desc_b);
      } else if constexpr (MT == 16) {
        if (ks == 0)
          wgmma_ss_m64n16<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n16<true>(partial, desc_a, desc_b);
      } else {
        if (ks == 0)
          wgmma_ss_m64n24<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n24<true>(partial, desc_a, desc_b);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    const float ws = weight_scale[(tile_n >> 7) * Sk + kb];
#pragma unroll
    for (int i = 0; i < MT / 2; ++i) {
      const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
      accum[i] = fmaf(partial[i], activation_scale[token] * ws, accum[i]);
    }
    if (kb + 1 < kb_end) __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < MT / 2; ++i) {
    split_partial[i * 256 + tid] = accum[i];
  }
  cluster.sync();

  float* remote_partial[SPLIT];
#pragma unroll
  for (int rank = 0; rank < SPLIT; ++rank) {
    remote_partial[rank] = cluster.map_shared_rank(split_partial, rank);
  }

  for (int i = split; i < MT / 2; i += SPLIT) {
    const int index = i * 256 + tid;
    float value = 0.0f;
#pragma unroll
    for (int rank = 0; rank < SPLIT; ++rank) {
      value += remote_partial[rank][index];
    }
    const int n_local = wg * 64 + warp * 16 + lane_group + 8 * ((i >> 1) & 1);
    const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
    if (token < tile_rows && tile_n + n_local < N) {
      out[static_cast<int64_t>(tile_m + token) * N + tile_n + n_local] = __float2bfloat16_rn(value);
    }
  }
  cluster.sync();
}

// SS-WGMMA split-K body that consumes KU k-groups per shared-memory round.
// The one-k-group version spends ~28% of its stall cycles in the
// the generic-to-async publication fence and ~16% in
// the following __syncthreads; batching KU k-groups halves that per weight
// byte and doubles the number of weight vectors in flight per thread, without
// touching the grid shape (so CTA count / occupancy are unchanged).
template <int MT, int SPLIT, int KU, bool KeepX, bool PipeWeights = false, bool PackMT8 = false,
          bool ExactRound = false, bool PairPipe = false, bool PushDSM = false,
          bool FlatPush = false, bool CompactM2 = false, bool ExplicitSTS = false,
          bool DenseTile = false, bool DenseOut = false, bool PrefetchScale = false,
          bool AsyncPush = false, bool DenseN = false, bool MmaPipe = false, bool Quant8 = false,
          bool PerWarpMbar = false, bool PreWaitScale = false, bool SpreadOwners = false>
__global__ __launch_bounds__(128, 2) void __cluster_dims__(SPLIT, 1, 1)
    fp8_blockscale_wgmma_ku_kernel(const __nv_bfloat16* __restrict__ x,
                                   const uint8_t* __restrict__ weight,
                                   const float* __restrict__ weight_scale, float eps,
                                   __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24);
  static_assert(!PipeWeights || KU == 2 || KU == 4);
  static_assert(!PairPipe || (KU == 4 && ExactRound && !PipeWeights));
  static_assert(!(PushDSM && FlatPush));
  static_assert(!CompactM2 || (MT == 8 && SPLIT == 14 && KU == 4 && PackMT8 && ExactRound &&
                               PairPipe && PushDSM));
  static_assert(!DenseTile || ((MT == 24 || MT == 16) && SPLIT == 8 && PushDSM && ExactRound));
  static_assert(!DenseOut || (PushDSM && ((MT == 24 && SPLIT == 12) || (MT == 16 && SPLIT == 6))));
  static_assert(!AsyncPush || (PushDSM && (MT >= 16 || PackMT8)));
  static_assert(!DenseN || CompactM2 || (MT == 24 && SPLIT == 12));
  static_assert(!MmaPipe || ((KU == 2 && !PairPipe) || (KU == 4 && PairPipe)));
  static_assert(!Quant8 || (MT == 16 && SPLIT == 6 && KU == 2 && DenseOut));
  static_assert(!PerWarpMbar ||
                (PushDSM && AsyncPush &&
                 ((MT == 24 && (SPLIT == 8 || SPLIT == 12)) || (MT == 16 && SPLIT == 8))));
  static_assert(!PreWaitScale || PrefetchScale);
  static_assert(!SpreadOwners || (MT == 8 && SPLIT == 8 && PushDSM && AsyncPush && !PerWarpMbar));
  constexpr int NACC = MT / 2;
  constexpr int ATOMS = MT / 8;
  __shared__ __align__(1024) uint8_t wa[KU][64 * 128];
  __shared__ __align__(1024) uint8_t bq[KU][ATOMS * 1024];
  __shared__ float activation_scale[KU][MT];
  __shared__ uint32_t
      push_partial[FlatPush
                       ? SplitkMailbox<MT, SPLIT, (MT >= 16 || PackMT8)>::WORDS
                       : ((PushDSM || AsyncPush) ? (CompactM2 ? SPLIT * 32 : SPLIT * 128) : 1)];
  __shared__ __align__(8) uint64_t rbar[PerWarpMbar ? 4 : 1];
  uint32_t* const split_partial = reinterpret_cast<uint32_t*>(wa);
  uint64_t wa_desc0 = 0;
  uint64_t bq_desc0 = 0;
  if constexpr (MT == 16 || MT == 24) {
    wa_desc0 = wgmma_b_desc(&wa[0][0]);
    bq_desc0 = wgmma_b_desc(&bq[0][0]);
  }

  const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  if constexpr (AsyncPush) {
    constexpr int NOWNER = SpreadOwners ? SPLIT : MT / 4;
    if constexpr (CompactM2) {
      if (threadIdx.x == 0) {
        const uint32_t addr = smem_addr(&rbar[0]);
        mbar_init(addr, 1);
        mbar_arrive_expect(addr, SPLIT * 32 * 4);
      }
      __syncthreads();
    } else if (cluster.block_rank() < NOWNER) {
      if constexpr (PerWarpMbar) {
        if (threadIdx.x == 0) {
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            const uint32_t addr = smem_addr(&rbar[w]);
            mbar_init(addr, 1);
            mbar_arrive_expect(addr, SPLIT * 32 * 4);
          }
        }
      } else {
        if (threadIdx.x == 0) {
          const uint32_t addr = smem_addr(&rbar[0]);
          mbar_init(addr, 1);
          mbar_arrive_expect(addr, SPLIT * (SpreadOwners ? 32 : 128) * 4);
        }
      }
      if (threadIdx.x < 32) __syncwarp();
    }
    mbar_init_publish();
  } else if constexpr (PushDSM || FlatPush) {
    wgmma_cluster_arrive_relaxed();
  }
  const int split = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int cluster_x =
      SPLIT == 6 ? static_cast<int>(cluster_id_x()) : static_cast<int>(blockIdx.x) / SPLIT;
  const int tile_n = cluster_x * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;
  const int kb_begin = (Sk * split) / SPLIT;
  const int kb_end = (Sk * (split + 1)) / SPLIT;

  float accum[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) accum[i] = 0.0f;

  uint64_t x_cache_policy = 0;
  if constexpr (KeepX) x_cache_policy = make_l2_evict_last_policy();

  const int kseg = tid & 7;
  const int wrow0 = tid >> 3;
  int wsoff[4];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int row = wrow0 + 16 * it;
    const int logical = (row & 7) * 128 + kseg * 16;
    wsoff[it] = (row >> 3) * 1024 + (logical ^ ((logical & 0x380) >> 3));
  }
  const int subgroup_lane = tid & 15;
  const int row0 = tid >> 4;
  const int nvalid = tile_n + wrow0;
  const float* const wsp = weight_scale + (tile_n >> 7) * Sk;

  int kc = kb_begin;
  do {
    uint4 qraw[KU][2];
    if constexpr (Quant8) {
      const int qlane = tid & 7;
      const int qrow = tid >> 3;
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        qraw[u][0] = load_activation_row_contiguous16_l1_last(tile_x, qrow, 2 * qlane, tile_rows, K,
                                                              (kc + u) * 128);
        qraw[u][1] = load_activation_row_contiguous16_l1_last(tile_x, qrow, 2 * qlane + 1,
                                                              tile_rows, K, (kc + u) * 128);
      }
    }
    // Compact M2 separates the two committed weight pairs with activation
    // LDGs. This gives the first pair a longer head start while useful global
    // issue work delays the second pair.
#pragma unroll
    for (int u = 0; u < (CompactM2 && PairPipe ? 2 : KU); ++u) {
      const bool live = ExactRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int n = nvalid + 16 * it;
        if constexpr (DenseTile || DenseN) {
          const uint8_t* src = weight + static_cast<int64_t>(n) * K + (kc + u) * 128 + kseg * 16;
          cp_async_weight16(wa[u] + wsoff[it], src, true);
        } else {
          const bool valid = live && n < N;
          const uint8_t* src =
              valid ? weight + static_cast<int64_t>(n) * K + (kc + u) * 128 + kseg * 16 : weight;
          cp_async_weight16(wa[u] + wsoff[it], src, valid);
        }
      }
      if constexpr (PipeWeights) {
        asm volatile("cp.async.commit_group;" ::: "memory");
      } else if constexpr (PairPipe) {
        if ((u & 1) != 0) asm volatile("cp.async.commit_group;" ::: "memory");
      }
    }
    if constexpr (!PipeWeights && !PairPipe) {
      asm volatile("cp.async.commit_group;" ::: "memory");
    }

    if constexpr (Quant8) {
      const int qlane = tid & 7;
      const int qrow = tid >> 3;
#pragma unroll
      for (int u = 0; u < KU; ++u)
        quantize_activation_row_contiguous8_fast(qraw[u][0], qraw[u][1], bq[u], activation_scale[u],
                                                 eps, qrow, qlane);
    } else {
      uint4 araw[KU][ATOMS];
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const bool live = ExactRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
        for (int a = 0; a < ATOMS; ++a)
          if constexpr (DenseTile) {
            araw[u][a] =
                *reinterpret_cast<const uint4*>(tile_x + static_cast<int64_t>(row0 + 8 * a) * K +
                                                (kc + u) * 128 + subgroup_lane * 8);
          } else {
            araw[u][a] = live ? load_activation_row_contiguous16<KeepX>(
                                    tile_x, row0 + 8 * a, subgroup_lane, tile_rows, K,
                                    (kc + u) * 128, x_cache_policy)
                              : make_uint4(0, 0, 0, 0);
          }
      }
      if constexpr (CompactM2 && PairPipe) {
#pragma unroll
        for (int u = 2; u < KU; ++u) {
          const bool live = ExactRound || (u == 0) || (kc + u < kb_end);
#pragma unroll
          for (int it = 0; it < 4; ++it) {
            const int n = nvalid + 16 * it;
            if constexpr (DenseTile || DenseN) {
              const uint8_t* src =
                  weight + static_cast<int64_t>(n) * K + (kc + u) * 128 + kseg * 16;
              cp_async_weight16(wa[u] + wsoff[it], src, true);
            } else {
              const bool valid = live && n < N;
              const uint8_t* src =
                  valid ? weight + static_cast<int64_t>(n) * K + (kc + u) * 128 + kseg * 16
                        : weight;
              cp_async_weight16(wa[u] + wsoff[it], src, valid);
            }
          }
          if ((u & 1) != 0) asm volatile("cp.async.commit_group;" ::: "memory");
        }
      }
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int a = 0; a < ATOMS; ++a)
          quantize_activation_row_contiguous16<false, MT == 24 && SPLIT == 8>(
              araw[u][a], bq[u], activation_scale[u], eps, row0 + 8 * a, subgroup_lane);
    }

    float early_ws[KU];
    if constexpr (PrefetchScale && PairPipe) {
#pragma unroll
      for (int u = 0; u < KU; ++u)
        early_ws[u] = wsp[ExactRound ? kc + u : (kc + u < Sk ? kc + u : Sk - 1)];
    }

    constexpr bool EarlyActFence =
        CompactM2 || (DenseOut && MT == 24) || (MT == 16 && SPLIT == 8 && ExactRound && AsyncPush);
    if constexpr (EarlyActFence) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if constexpr (PairPipe) {
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else if constexpr (PipeWeights && KU == 2) {
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else if constexpr (PipeWeights && KU == 4) {
      asm volatile("cp.async.wait_group 3;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    if constexpr (EarlyActFence) {
      // The local activation-publication fence was issued above the weight
      // drain; the following CTA barrier joins both paths.
    } else {
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                       : "memory");
    }
    __syncthreads();

    if constexpr (MmaPipe) {
      // Commit all staged K groups before draining any, so tensor work from
      // the later group overlaps the scale/rescale path of the earlier one.
      float pg[KU][NACC];
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int i = 0; i < NACC; ++i) pg[u][i] = 0.0f;
      uint64_t da[KU];
      uint64_t db[KU];
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        if constexpr (MT == 16 || MT == 24) {
          da[u] = wa_desc0 + static_cast<uint64_t>(u * 512);
          db[u] = bq_desc0 + static_cast<uint64_t>(u * (ATOMS * 64));
        } else {
          da[u] = wgmma_b_desc(wa[u]);
          db[u] = wgmma_b_desc(bq[u]);
        }
      }
      float pipe_ws[KU];
      if constexpr (PerWarpMbar) {
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const int kk = kc + u;
          pipe_ws[u] = wsp[ExactRound ? kk : (kk < Sk ? kk : Sk - 1)];
        }
      }
      if constexpr (KU == 2) {
        issue_ku_wgmma_group<MT>(pg[0], da[0], db[0]);
        if constexpr (PipeWeights) {
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          __syncthreads();
        }
        issue_ku_wgmma_group<MT>(pg[1], da[1], db[1]);
      } else {
        issue_ku_wgmma_group<MT>(pg[0], da[0], db[0]);
        issue_ku_wgmma_group<MT>(pg[1], da[1], db[1]);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        issue_ku_wgmma_group<MT>(pg[2], da[2], db[2]);
        issue_ku_wgmma_group<MT>(pg[3], da[3], db[3]);
      }

      // Stage scale products while tensor-core work is outstanding; the
      // wait instructions otherwise pin these loads behind the drain.
      float sc[KU][PerWarpMbar ? 1 : ATOMS * 2];
      if constexpr (!PerWarpMbar) {
#pragma unroll
        for (int u = 0; u < KU; ++u) {
          const int kk = kc + u;
          const float wsu = wsp[ExactRound ? kk : (kk < Sk ? kk : Sk - 1)];
#pragma unroll
          for (int atom = 0; atom < ATOMS; ++atom)
#pragma unroll
            for (int row = 0; row < 2; ++row)
              sc[u][2 * atom + row] = activation_scale[u][8 * atom + 2 * lane_in_group + row] * wsu;
        }
      }
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        if constexpr (KU == 2) {
          if (u == 0)
            wgmma_wait_n<1>();
          else
            wgmma_wait_n<0>();
        } else {
          if (u == 0)
            wgmma_wait_n<3>();
          else if (u == 1)
            wgmma_wait_n<2>();
          else if (u == 2)
            wgmma_wait_n<1>();
          else
            wgmma_wait_n<0>();
        }
#pragma unroll
        for (int atom = 0; atom < ATOMS; ++atom) {
#pragma unroll
          for (int row = 0; row < 2; ++row) {
            const int i = 4 * atom + row;
            float scale;
            if constexpr (PerWarpMbar)
              scale = activation_scale[u][8 * atom + 2 * lane_in_group + row] * pipe_ws[u];
            else
              scale = sc[u][2 * atom + row];
            accum[i] = fmaf(pg[u][i], scale, accum[i]);
            accum[i + 2] = fmaf(pg[u][i + 2], scale, accum[i + 2]);
          }
        }
      }
    } else
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const int kk = kc + u;
        float prefetched_ws = 0.0f;
        if constexpr (PrefetchScale && !PairPipe)
          prefetched_ws = wsp[ExactRound ? kk : (kk < Sk ? kk : Sk - 1)];
        if constexpr (PipeWeights) {
          if (u > 0) {
            if constexpr (KU == 2) {
              asm volatile("cp.async.wait_group 0;" ::: "memory");
            } else {
              if (u == 1)
                asm volatile("cp.async.wait_group 2;" ::: "memory");
              else if (u == 2)
                asm volatile("cp.async.wait_group 1;" ::: "memory");
              else
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
          }
        } else if constexpr (PairPipe) {
          if (u == 2) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
          }
        }
        float partial[NACC];
#pragma unroll
        for (int i = 0; i < NACC; ++i) partial[i] = 0.0f;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          uint64_t desc_a;
          uint64_t desc_b;
          if constexpr (MT == 16 || MT == 24) {
            desc_a = wa_desc0 + static_cast<uint64_t>(u * 512 + ks * 2);
            desc_b = bq_desc0 + static_cast<uint64_t>(u * (ATOMS * 64) + ks * 2);
          } else {
            desc_a = wgmma_b_desc(wa[u] + ks * 32);
            desc_b = wgmma_b_desc(bq[u] + ks * 32);
          }
          if constexpr (MT == 8) {
            if (ks == 0)
              wgmma_ss_m64n8<false>(partial, desc_a, desc_b);
            else
              wgmma_ss_m64n8<true>(partial, desc_a, desc_b);
          } else if constexpr (MT == 16) {
            if (ks == 0)
              wgmma_ss_m64n16<false>(partial, desc_a, desc_b);
            else
              wgmma_ss_m64n16<true>(partial, desc_a, desc_b);
          } else {
            if (ks == 0)
              wgmma_ss_m64n24<false>(partial, desc_a, desc_b);
            else
              wgmma_ss_m64n24<true>(partial, desc_a, desc_b);
          }
        }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float prefetched_scale[ATOMS * 2];
        if constexpr (PreWaitScale) {
#pragma unroll
          for (int atom = 0; atom < ATOMS; ++atom)
#pragma unroll
            for (int row = 0; row < 2; ++row) {
              const int token = 8 * atom + 2 * lane_in_group + row;
              prefetched_scale[2 * atom + row] = activation_scale[u][token] * prefetched_ws;
            }
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

        const float ws =
            (PrefetchScale && PairPipe)
                ? early_ws[u]
                : (PrefetchScale ? prefetched_ws : wsp[ExactRound ? kk : (kk < Sk ? kk : Sk - 1)]);
        if constexpr (MT == 16) {
#pragma unroll
          for (int atom = 0; atom < 2; ++atom) {
#pragma unroll
            for (int row = 0; row < 2; ++row) {
              const int i = 4 * atom + row;
              const int token = 8 * atom + 2 * lane_in_group + row;
              const float scale =
                  PreWaitScale ? prefetched_scale[2 * atom + row] : activation_scale[u][token] * ws;
              accum[i] = fmaf(partial[i], scale, accum[i]);
              accum[i + 2] = fmaf(partial[i + 2], scale, accum[i + 2]);
            }
          }
        } else {
#pragma unroll
          for (int atom = 0; atom < ATOMS; ++atom) {
#pragma unroll
            for (int row = 0; row < 2; ++row) {
              const int i = 4 * atom + row;
              const int token = 8 * atom + 2 * lane_in_group + row;
              const float scale =
                  PreWaitScale ? prefetched_scale[2 * atom + row] : activation_scale[u][token] * ws;
              accum[i] = fmaf(partial[i], scale, accum[i]);
              accum[i + 2] = fmaf(partial[i + 2], scale, accum[i + 2]);
            }
          }
        }
      }
    if (!ExactRound && kc + KU < kb_end) __syncthreads();
    kc += KU;
  } while (!ExactRound && kc < kb_end);

  if constexpr (SPLIT == 1) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) {
      const int n = tile_n + warp * 16 + lane_group + 8 * ((i >> 1) & 1);
      const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
      if (token < tile_rows && n < N)
        out[static_cast<int64_t>(tile_m + token) * N + n] = __float2bfloat16_rn(accum[i]);
    }
  } else if constexpr (FlatPush) {
    wgmma_cluster_wait_acquire();
    splitk_flat_push_reduce<MT, SPLIT, (MT >= 16 || PackMT8)>(
        cluster, push_partial, accum, out, split, tid, tile_m, tile_n, tile_rows, N);
  } else if constexpr (PushDSM) {
    static_assert(MT >= 16 || PackMT8);
    constexpr int NPAIR = NACC / 2;
    static_assert(NPAIR <= SPLIT);
    wgmma_cluster_wait_acquire();
    if constexpr (CompactM2) {
      splitk_push_reduce_m2_ku<SPLIT>(cluster, push_partial, accum, out, split, tid, tile_m, tile_n,
                                      N, &rbar[0]);
    } else {
#pragma unroll
      for (int p = 0; p < NPAIR; ++p) {
        const uint32_t value = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
        const int owner = SpreadOwners ? p + NPAIR * warp : p;
        if constexpr (AsyncPush) {
          if constexpr (PerWarpMbar)
            dsm_st_async_u32(dsm_addr(push_partial + split * 128 + tid, owner), value,
                             dsm_addr(&rbar[warp], owner));
          else
            dsm_st_async_u32(dsm_addr(push_partial + split * 128 + tid, owner), value,
                             dsm_addr(&rbar[0], owner));
        } else if constexpr (ExplicitSTS) {
          dsm_st_u32(dsm_addr(push_partial + split * 128 + tid, p), value);
        } else {
          uint32_t* const remote = cluster.map_shared_rank(push_partial + split * 128 + tid, p);
          *remote = value;
        }
      }
      if constexpr (!AsyncPush) wgmma_cluster_arrive_release();
      const bool reduce_here =
          SpreadOwners ? (split < 4 * NPAIR && warp == (split / NPAIR)) : (split < NPAIR);
      if (reduce_here) {
        if constexpr (AsyncPush) {
          if constexpr (PerWarpMbar)
            mbar_wait(smem_addr(&rbar[warp]), 0);
          else
            mbar_wait(smem_addr(&rbar[0]), 0);
        } else
          wgmma_cluster_wait_acquire();
        float2 sum;
        if constexpr (ExplicitSTS && SPLIT == 12) {
          float2 sum0 = unpack_bf16x2(push_partial[tid]);
          float2 sum1 = unpack_bf16x2(push_partial[128 + tid]);
          float2 sum2 = unpack_bf16x2(push_partial[256 + tid]);
          float2 sum3 = unpack_bf16x2(push_partial[384 + tid]);
#pragma unroll
          for (int rank = 4; rank < 12; rank += 4) {
            const float2 v0 = unpack_bf16x2(push_partial[rank * 128 + tid]);
            const float2 v1 = unpack_bf16x2(push_partial[(rank + 1) * 128 + tid]);
            const float2 v2 = unpack_bf16x2(push_partial[(rank + 2) * 128 + tid]);
            const float2 v3 = unpack_bf16x2(push_partial[(rank + 3) * 128 + tid]);
            sum0.x += v0.x;
            sum0.y += v0.y;
            sum1.x += v1.x;
            sum1.y += v1.y;
            sum2.x += v2.x;
            sum2.y += v2.y;
            sum3.x += v3.x;
            sum3.y += v3.y;
          }
          sum = make_float2((sum0.x + sum2.x) + (sum1.x + sum3.x),
                            (sum0.y + sum2.y) + (sum1.y + sum3.y));
        } else if constexpr (ExplicitSTS && (SPLIT % 2 == 0)) {
          float2 sum0 = unpack_bf16x2(push_partial[tid]);
          float2 sum1 = unpack_bf16x2(push_partial[128 + tid]);
#pragma unroll
          for (int rank = 2; rank < SPLIT; rank += 2) {
            const float2 v0 = unpack_bf16x2(push_partial[rank * 128 + tid]);
            const float2 v1 = unpack_bf16x2(push_partial[(rank + 1) * 128 + tid]);
            sum0.x += v0.x;
            sum0.y += v0.y;
            sum1.x += v1.x;
            sum1.y += v1.y;
          }
          sum = make_float2(sum0.x + sum1.x, sum0.y + sum1.y);
        } else {
          sum = unpack_bf16x2(push_partial[tid]);
#pragma unroll
          for (int rank = 1; rank < SPLIT; ++rank) {
            const float2 v = unpack_bf16x2(push_partial[rank * 128 + tid]);
            sum.x += v.x;
            sum.y += v.y;
          }
        }
        const int owner_p = SpreadOwners ? split % NPAIR : split;
        const int n = tile_n + warp * 16 + lane_group + 8 * (owner_p & 1);
        const int token = 8 * (owner_p >> 1) + 2 * lane_in_group;
        if (DenseTile || DenseOut || n < N) {
          const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
          if (DenseTile || DenseOut || token < tile_rows) out[base] = __float2bfloat16_rn(sum.x);
          if (DenseTile || DenseOut || token + 1 < tile_rows)
            out[base + N] = __float2bfloat16_rn(sum.y);
        }
      }
    }
  } else {
    if constexpr (!PairPipe) __syncthreads();
    if constexpr (MT >= 16 || PackMT8) {
#pragma unroll
      for (int i = 0; i < NACC / 2; ++i) {
        split_partial[i * 128 + tid] = pack_bf16x2(accum[2 * i], accum[2 * i + 1]);
      }
      cluster.sync();
      for (int p = split; p < NACC / 2; p += SPLIT) {
        uint32_t* const q = split_partial + p * 128 + tid;
        float s0 = 0.0f;
        float s1 = 0.0f;
#pragma unroll
        for (int rank = 0; rank < SPLIT; ++rank) {
          const float2 v = unpack_bf16x2(*cluster.map_shared_rank(q, rank));
          s0 += v.x;
          s1 += v.y;
        }
        const int n = tile_n + warp * 16 + lane_group + 8 * (p & 1);
        const int token = 8 * (p >> 1) + 2 * lane_in_group;
        if (n < N) {
          const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
          if (token < tile_rows) out[base] = __float2bfloat16_rn(s0);
          if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(s1);
        }
      }
      cluster.sync();
    } else {
      float* const fp = reinterpret_cast<float*>(split_partial);
#pragma unroll
      for (int i = 0; i < NACC; ++i) fp[i * 128 + tid] = accum[i];
      cluster.sync();
      for (int i = split; i < NACC; i += SPLIT) {
        float* const q = fp + i * 128 + tid;
        float value = 0.0f;
#pragma unroll
        for (int rank = 0; rank < SPLIT; ++rank) value += *cluster.map_shared_rank(q, rank);
        const int n = tile_n + warp * 16 + lane_group + 8 * ((i >> 1) & 1);
        const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
        if (token < tile_rows && n < N)
          out[static_cast<int64_t>(tile_m + token) * N + n] = __float2bfloat16_rn(value);
      }
      cluster.sync();
    }
  }
}

template <int MT, int SPLIT, bool KeepX, int KU = 2, bool PipeWeights = false, bool PackMT8 = false,
          bool ExactRound = false, bool PairPipe = false, bool PushDSM = false,
          bool FlatPush = false, bool CompactM2 = false, bool ExplicitSTS = false,
          bool DenseTile = false, bool DenseOut = false, bool PrefetchScale = false,
          bool AsyncPush = false, bool DenseN = false, bool MmaPipe = false, bool Quant8 = false,
          bool PerWarpMbar = false, bool PreWaitScale = false, bool SpreadOwners = false>
static void launch_ku(dim3 grid, bool lb, cudaStream_t stream, const __nv_bfloat16* x,
                      const uint8_t* weight, const float* weight_scale, float eps,
                      __nv_bfloat16* out, int M, int N, int K) {
  if constexpr (SPLIT > 8) {
    static bool opted = false;
    if (!opted) {
      cudaFuncSetAttribute(
          fp8_blockscale_wgmma_ku_kernel<MT, SPLIT, KU, KeepX, PipeWeights, PackMT8, ExactRound,
                                         PairPipe, PushDSM, FlatPush, CompactM2, ExplicitSTS,
                                         DenseTile, DenseOut, PrefetchScale, AsyncPush, DenseN,
                                         MmaPipe, Quant8, PerWarpMbar, PreWaitScale, SpreadOwners>,
          cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      opted = true;
    }
  }
  if (lb) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(128, 1, 1);
    config.stream = stream;
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attribute.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaLaunchKernelEx(
        &config,
        fp8_blockscale_wgmma_ku_kernel<MT, SPLIT, KU, KeepX, PipeWeights, PackMT8, ExactRound,
                                       PairPipe, PushDSM, FlatPush, CompactM2, ExplicitSTS,
                                       DenseTile, DenseOut, PrefetchScale, AsyncPush, DenseN,
                                       MmaPipe, Quant8, PerWarpMbar, PreWaitScale, SpreadOwners>,
        x, weight, weight_scale, eps, out, M, N, K);
  } else {
    fp8_blockscale_wgmma_ku_kernel<MT, SPLIT, KU, KeepX, PipeWeights, PackMT8, ExactRound, PairPipe,
                                   PushDSM, FlatPush, CompactM2, ExplicitSTS, DenseTile, DenseOut,
                                   PrefetchScale, AsyncPush, DenseN, MmaPipe, Quant8, PerWarpMbar,
                                   PreWaitScale, SpreadOwners>
        <<<grid, 128, 0, stream>>>(x, weight, weight_scale, eps, out, M, N, K);
  }
}

template <int MT, bool KeepX>
static void launch_ku_split(int split, dim3 grid, bool lb, cudaStream_t stream,
                            const __nv_bfloat16* x, const uint8_t* weight,
                            const float* weight_scale, float eps, __nv_bfloat16* out, int M, int N,
                            int K) {
  switch (split) {
    case 2:
      launch_ku<MT, 2, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 4:
      launch_ku<MT, 4, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 6:
      launch_ku<MT, 6, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    case 8:
      launch_ku<MT, 8, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
    default:
      launch_ku<MT, 16, KeepX>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
      break;
  }
}

static void launch_ku_all(int mt, int split, bool keepx, bool lb, dim3 grid, cudaStream_t stream,
                          const __nv_bfloat16* x, const uint8_t* weight, const float* weight_scale,
                          float eps, __nv_bfloat16* out, int M, int N, int K) {
  if (mt == 8) {
    if (keepx)
      launch_ku_split<8, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
    else
      launch_ku_split<8, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
  } else if (mt == 16) {
    if (keepx)
      launch_ku_split<16, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
    else
      launch_ku_split<16, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                 K);
  } else {
    if (keepx && split == 8)
      launch_ku<24, 8, true, 2, true>(grid, lb, stream, x, weight, weight_scale, eps, out, M, N, K);
    else if (keepx)
      launch_ku_split<24, true>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                K);
    else
      launch_ku_split<24, false>(split, grid, lb, stream, x, weight, weight_scale, eps, out, M, N,
                                 K);
  }
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// Two-stage weight pipeline for long-K split-16 shapes. The next weight tile
// is issued before the current WGMMA, overlapping HBM/L2 latency with quant
// and tensor-core work while keeping activation staging single-buffered.
template <int MT, int SPLIT, bool KeepX, bool PushDSM = false, bool FlatPush = false,
          bool CompactM12 = false, bool BalancedAmax = false, bool FastAmax = false,
          bool AsyncM12 = false, bool SparseM12Quant = false, bool PipeNextWs = false>
__global__ __launch_bounds__(128, 2) void __cluster_dims__(SPLIT, 1, 1)
    fp8_blockscale_wgmma_cpa_kernel(const __nv_bfloat16* __restrict__ x,
                                    const uint8_t* __restrict__ weight,
                                    const float* __restrict__ weight_scale, float eps,
                                    __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  static_assert(MT == 8 || MT == 16 || MT == 24);
  static_assert(!(PushDSM && FlatPush));
  static_assert(!CompactM12 || (MT == 16 && SPLIT == 16 && FlatPush));
  static_assert(!AsyncM12 || CompactM12);
  static_assert(!SparseM12Quant || (MT == 16 && CompactM12 && FastAmax));
  static_assert(!PipeNextWs || (MT == 16 && SPLIT == 16 && CompactM12 && FastAmax));
  constexpr int NACC = MT / 2;
  constexpr int ATOMS = MT / 8;
  __shared__ __align__(1024) uint8_t wa[2][64 * 128];
  __shared__ __align__(1024) uint8_t bq[ATOMS * 1024];
  __shared__ float activation_scale[MT];
  __shared__ uint32_t
      split_partial[FlatPush ? SplitkMailbox<MT, SPLIT, (MT >= 16)>::WORDS : NACC * 128];
  __shared__ uint32_t push_partial[PushDSM ? SPLIT * 128 : 1];
  __shared__ __align__(8) uint64_t rbar[1];

  const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  if constexpr (AsyncM12) {
    if (cluster.block_rank() < 12) {
      if (threadIdx.x == 0) {
        const uint32_t addr = smem_addr(&rbar[0]);
        mbar_init(addr, 1);
        mbar_arrive_expect(addr, SPLIT * 32 * 4);
      }
      if (threadIdx.x < 32) __syncwarp();
    }
    mbar_init_publish();
  } else if constexpr (PushDSM || FlatPush) {
    wgmma_cluster_arrive_relaxed();
  }
  const int split = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int tile_n = (static_cast<int>(blockIdx.x) / SPLIT) * 64;
  const int tile_m = static_cast<int>(blockIdx.y) * MT;
  const int tile_rows = M - tile_m;
  const __nv_bfloat16* tile_x = x + static_cast<int64_t>(tile_m) * K;
  const int Sk = K >> 7;
  const int kb_begin = (Sk * split) / SPLIT;
  const int kb_end = (Sk * (split + 1)) / SPLIT;

  float accum[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) accum[i] = 0.0f;

  uint64_t x_cache_policy = 0;
  if constexpr (KeepX) x_cache_policy = make_l2_evict_last_policy();

  const int kseg = tid & 7;
  const int wrow0 = tid >> 3;
  uint32_t wsoff[4];
  int wrow[4];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int row = wrow0 + 16 * it;
    const int logical = (row & 7) * 128 + kseg * 16;
    wrow[it] = tile_n + row;
    wsoff[it] = (row >> 3) * 1024 + (logical ^ ((logical & 0x380) >> 3));
  }
  const uint32_t wa_base = static_cast<uint32_t>(__cvta_generic_to_shared(&wa[0][0]));
  const uint64_t wa_desc0 = wgmma_b_desc(&wa[0][0]);
  const uint64_t bq_desc0 = wgmma_b_desc(&bq[0]);
  const uint8_t* const wbase = weight + static_cast<int64_t>(kseg) * 16;
  const int subgroup_lane = tid & 15;
  const int row0 = tid >> 4;

#define ISSUE_WTILE(kb_, buf_)                                                                \
  {                                                                                           \
    const int64_t koff_ = static_cast<int64_t>(kb_) * 128;                                    \
    _Pragma("unroll") for (int it = 0; it < 4; ++it) {                                        \
      const int ok_ = wrow[it] < N;                                                           \
      cp_async16(wa_base + (buf_) * 8192 + wsoff[it],                                         \
                 wbase + static_cast<int64_t>(ok_ ? wrow[it] : 0) * K + koff_, ok_ ? 16 : 0); \
    }                                                                                         \
  }

  uint4 fast_raw[2];
  if constexpr (FastAmax && MT == 16) {
    const int qlane = tid & 7;
    const int qrow = tid >> 3;
    const int kbase = kb_begin * 128;
    fast_raw[0] = load_activation_row_contiguous16<KeepX>(tile_x, qrow, 2 * qlane, tile_rows, K,
                                                          kbase, x_cache_policy);
    fast_raw[1] = load_activation_row_contiguous16<KeepX>(tile_x, qrow, 2 * qlane + 1, tile_rows, K,
                                                          kbase, x_cache_policy);
  }
  if (kb_begin < kb_end) ISSUE_WTILE(kb_begin, 0);
  cp_async_commit();

  const float* const wsp = weight_scale + (tile_n >> 7) * Sk;
  float ws_pipe = 0.0f;
  if constexpr (PipeNextWs) ws_pipe = wsp[kb_begin];
  int buf = 0;
  for (int kb = kb_begin; kb < kb_end; ++kb) {
    const int kbase = kb * 128;
    // Issue the block-scale load before WGMMA memory clobbers pin it on the
    // critical path after compute.
    const float ws_pre = PipeNextWs ? ws_pipe : wsp[kb];
    uint4 raw[ATOMS];
    if constexpr (FastAmax && MT == 16) {
      const int qlane = tid & 7;
      const int qrow = tid >> 3;
      if constexpr (SparseM12Quant) {
        if (qrow < 12) {
          quantize_activation_row_contiguous8_fast(fast_raw[0], fast_raw[1], bq, activation_scale,
                                                   eps, qrow, qlane);
        } else {
          const int atom = qrow >> 3;
          const int row_in_atom = qrow & 7;
          const int logical = row_in_atom * 128 + qlane * 16;
          const int physical = logical ^ ((logical & 0x380) >> 3);
          *reinterpret_cast<uint4*>(bq + atom * 1024 + physical) = make_uint4(0, 0, 0, 0);
          if (qlane == 0) activation_scale[qrow] = 1.0f;
        }
      } else {
        quantize_activation_row_contiguous8_fast(fast_raw[0], fast_raw[1], bq, activation_scale,
                                                 eps, qrow, qlane);
      }
    } else {
#pragma unroll
      for (int a = 0; a < ATOMS; ++a) {
        raw[a] = load_activation_row_contiguous16<KeepX>(tile_x, row0 + 8 * a, subgroup_lane,
                                                         tile_rows, K, kbase, x_cache_policy);
      }
#pragma unroll
      for (int a = 0; a < ATOMS; ++a) {
        quantize_activation_row_contiguous16<BalancedAmax, FastAmax>(
            raw[a], bq, activation_scale, eps, row0 + 8 * a, subgroup_lane);
      }
    }
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                     : "memory");
    if (kb + 1 < kb_end) {
      ISSUE_WTILE(kb + 1, buf ^ 1);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    float partial[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) partial[i] = 0.0f;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t desc_a = wa_desc0 + static_cast<uint64_t>(buf * 512 + ks * 2);
      const uint64_t desc_b = bq_desc0 + static_cast<uint64_t>(ks * 2);
      if constexpr (MT == 8) {
        if (ks == 0)
          wgmma_ss_m64n8<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n8<true>(partial, desc_a, desc_b);
      } else if constexpr (MT == 16) {
        if (ks == 0)
          wgmma_ss_m64n16<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n16<true>(partial, desc_a, desc_b);
      } else {
        if (ks == 0)
          wgmma_ss_m64n24<false>(partial, desc_a, desc_b);
        else
          wgmma_ss_m64n24<true>(partial, desc_a, desc_b);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    if constexpr (PipeNextWs) {
      if (kb + 1 < kb_end) ws_pipe = wsp[kb + 1];
    }
    float fast_scale[ATOMS * 2];
    if constexpr (FastAmax && MT == 16) {
      if (kb + 1 < kb_end) {
        const int qlane = tid & 7;
        const int qrow = tid >> 3;
        const int next_kbase = (kb + 1) * 128;
        fast_raw[0] = load_activation_row_contiguous16<KeepX>(tile_x, qrow, 2 * qlane, tile_rows, K,
                                                              next_kbase, x_cache_policy);
        fast_raw[1] = load_activation_row_contiguous16<KeepX>(
            tile_x, qrow, 2 * qlane + 1, tile_rows, K, next_kbase, x_cache_policy);
      }
#pragma unroll
      for (int atom = 0; atom < ATOMS; ++atom) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const int token = 8 * atom + 2 * lane_in_group + row;
          fast_scale[2 * atom + row] = activation_scale[token] * ws_pre;
        }
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    const float ws = ws_pre;
#pragma unroll
    for (int atom = 0; atom < ATOMS; ++atom) {
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        const int i = 4 * atom + row;
        const int token = 8 * atom + 2 * lane_in_group + row;
        const float scale =
            (FastAmax && MT == 16) ? fast_scale[2 * atom + row] : activation_scale[token] * ws;
        accum[i] = fmaf(partial[i], scale, accum[i]);
        accum[i + 2] = fmaf(partial[i + 2], scale, accum[i + 2]);
      }
    }
    buf ^= 1;
    if (kb + 1 < kb_end) __syncthreads();
  }
#undef ISSUE_WTILE

  if constexpr (FlatPush) {
    wgmma_cluster_wait_acquire();
    if constexpr (CompactM12) {
      splitk_flat_push_reduce_m12_s16<AsyncM12>(cluster, split_partial, accum, out, split, tid,
                                                tile_m, tile_n, N, &rbar[0]);
    } else {
      splitk_flat_push_reduce<MT, SPLIT, (MT >= 16)>(cluster, split_partial, accum, out, split, tid,
                                                     tile_m, tile_n, tile_rows, N);
    }
  } else if constexpr (PushDSM) {
    static_assert(MT >= 16);
    constexpr int NPAIR = NACC / 2;
    static_assert(NPAIR <= SPLIT);
    wgmma_cluster_wait_acquire();
#pragma unroll
    for (int p = 0; p < NPAIR; ++p) {
      uint32_t* const remote = cluster.map_shared_rank(push_partial + split * 128 + tid, p);
      *remote = pack_bf16x2(accum[2 * p], accum[2 * p + 1]);
    }
    cluster.sync();
    if (split < NPAIR) {
      float2 sum = unpack_bf16x2(push_partial[tid]);
#pragma unroll
      for (int rank = 1; rank < SPLIT; ++rank) {
        const float2 v = unpack_bf16x2(push_partial[rank * 128 + tid]);
        sum.x += v.x;
        sum.y += v.y;
      }
      const int n = tile_n + warp * 16 + lane_group + 8 * (split & 1);
      const int token = 8 * (split >> 1) + 2 * lane_in_group;
      if (n < N) {
        const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
        if (token < tile_rows) out[base] = __float2bfloat16_rn(sum.x);
        if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(sum.y);
      }
    }
  } else if constexpr (MT >= 16) {
#pragma unroll
    for (int i = 0; i < NACC / 2; ++i) {
      split_partial[i * 128 + tid] = pack_bf16x2(accum[2 * i], accum[2 * i + 1]);
    }
    cluster.sync();
    for (int p = split; p < NACC / 2; p += SPLIT) {
      uint32_t* const q = split_partial + p * 128 + tid;
      float s0 = 0.0f;
      float s1 = 0.0f;
#pragma unroll
      for (int rank = 0; rank < SPLIT; ++rank) {
        const float2 v = unpack_bf16x2(*cluster.map_shared_rank(q, rank));
        s0 += v.x;
        s1 += v.y;
      }
      const int n = tile_n + warp * 16 + lane_group + 8 * (p & 1);
      const int token = 8 * (p >> 1) + 2 * lane_in_group;
      if (n < N) {
        const int64_t base = static_cast<int64_t>(tile_m + token) * N + n;
        if (token < tile_rows) out[base] = __float2bfloat16_rn(s0);
        if (token + 1 < tile_rows) out[base + N] = __float2bfloat16_rn(s1);
      }
    }
    cluster.sync();
  } else {
    float* const fp = reinterpret_cast<float*>(split_partial);
#pragma unroll
    for (int i = 0; i < NACC; ++i) fp[i * 128 + tid] = accum[i];
    cluster.sync();
    for (int i = split; i < NACC; i += SPLIT) {
      float* const q = fp + i * 128 + tid;
      float value = 0.0f;
#pragma unroll
      for (int rank = 0; rank < SPLIT; ++rank) value += *cluster.map_shared_rank(q, rank);
      const int n = tile_n + warp * 16 + lane_group + 8 * ((i >> 1) & 1);
      const int token = 8 * (i >> 2) + 2 * lane_in_group + (i & 1);
      if (token < tile_rows && n < N)
        out[static_cast<int64_t>(tile_m + token) * N + n] = __float2bfloat16_rn(value);
    }
    cluster.sync();
  }
}

template <int MT, int SPLIT, bool KeepX = false, bool PackMT8 = false, bool PushDSM = false,
          bool CompactM2 = false, bool AsyncM2 = false, bool DenseN = false, bool ScaleOpt = false>
static void launch_splitk_load_balanced(dim3 grid, cudaStream_t stream, const __nv_bfloat16* x,
                                        const uint8_t* weight, const float* weight_scale, float eps,
                                        __nv_bfloat16* out, int M, int N, int K) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(128, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute = {};
  attribute.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attribute.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(
      &config,
      fp8_blockscale_wgmma_splitk_kernel<MT, SPLIT, KeepX, PackMT8, PushDSM, false, false, false,
                                         CompactM2, ScaleOpt, false, AsyncM2, DenseN>,
      x, weight, weight_scale, eps, out, M, N, K);
}

void Run(ffi::TensorView x, ffi::TensorView weight, ffi::TensorView weight_scale, float eps,
         ffi::TensorView out) {
  const auto dev = x.device();
  TVM_FFI_ICHECK_EQ(dev.device_type, kDLCUDA);
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight_scale.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);

  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(weight.size(0));
  TVM_FFI_ICHECK_EQ(K & 127, 0);

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const auto* xp = tensor_ptr<const __nv_bfloat16>(x);
  const auto* wp = tensor_ptr<const uint8_t>(weight);
  const auto* wsp = tensor_ptr<const float>(weight_scale);
  auto* op = tensor_ptr<__nv_bfloat16>(out);

  // Wide-N K=2048 rows need more than one CTA per column tile to cover all
  // 132 SMs.  Split-8 clusters replace the single 768/1024-thread MWG CTA.
  if (M == 24 && N == 6144 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, 1);
    launch_ku<24, 8, false, 2, false, false, true, false, true, false, false, true, true, false,
              true, true, false, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps,
                                                            op, M, N, K);
    return;
  }
  if (M == 16 && N == 7168 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, 1);
    launch_ku<16, 8, false, 2, true, false, true, false, true, false, false, true, true, false,
              false, true, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M,
                                                      N, K);
    return;
  }

  // Preserve c797's tuned dispatch and exact MWG routes for W6 and W9.  W5
  // uses the conventional split-K PushDSM path below.
  const bool use_mwg = (N == 8192 && K == 1536 && (M == 24 || M == 1 || M == 8)) ||
                       (K == 2048 && ((N == 7168 && M == 16) || (N == 6144 && M == 24)));
  if (use_mwg) {
    if (fp8bs_mwg_launch(op, xp, wp, wsp, M, N, K, eps, stream)) return;
  }

  // Exact W10/W11 experiment: retain every existing route, but replace the
  // selected row's pull/conventional reduction with c131's flat owner-push
  // mailbox.  Each specialization gets its own cluster-size opt-in.
  if (M == 12 && N == 2624 && K == 6144) {
    static bool opted_flat_w10 = false;
    if (!opted_flat_w10) {
      cudaFuncSetAttribute(fp8_blockscale_wgmma_cpa_kernel<16, 16, false, false, true, true, false,
                                                           true, true, true, true>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      opted_flat_w10 = true;
    }
    const dim3 grid((N + 63) / 64 * 16, (M + 15) / 16);
    fp8_blockscale_wgmma_cpa_kernel<16, 16, false, false, true, true, false, true, true, true, true>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  if (M == 4 && N == 2048 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, (M + 7) / 8);
    launch_ku<8, 8, false, 2, true, true, true, false, true, false, false, true, false, false,
              false, true, false, true, false, false, false, true>(grid, true, stream, xp, wp, wsp,
                                                                   eps, op, M, N, K);
    return;
  }

  if (N > 2304 && N <= 3072 && K >= 4096 && M > 16 && M <= 24 && ((K >> 7) % 24 == 0)) {
    const dim3 grid((N + 63) / 64 * 12, (M + 23) / 24);
    if (M % 24 == 0 && (N & 63) == 0)
      launch_ku<24, 12, false, 2, false, false, false, false, true, false, false, true, false, true,
                true, true, true, false, false, false, true>(grid, true, stream, xp, wp, wsp, eps,
                                                             op, M, N, K);
    else
      launch_ku<24, 12, false, 2, false, false, false, false, true, false, false, true>(
          grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  if (M == 2 && N == 7168 && K == 2048) {
    const dim3 grid((N + 63) / 64 * 8, (M + 7) / 8);
    launch_splitk_load_balanced<8, 8, false, true, true, true, true, true, true>(
        grid, stream, xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  const bool use_hmma = (N > 3072 && K < 2048 && M > 16 && M <= 24) ||
                        (N > 3072 && K >= 2048 && M > 8 && M <= 16) ||
                        (N > 2304 && N <= 3072 && K >= 4096 && M > 16 && M <= 24);
  if (use_hmma) {
    fp8bs_launch(op, xp, wp, wsp, M, N, K, eps, stream);
    return;
  }
  if (N > 2304 && N <= 3072 && K >= 4096 && M > 8 && M <= 16 && (K >> 7) >= 16) {
    static bool opted_mid = false;
    if (!opted_mid) {
      cudaFuncSetAttribute(fp8_blockscale_wgmma_cpa_kernel<16, 16, true, true>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      opted_mid = true;
    }
    const dim3 grid((N + 63) / 64 * 16, (M + 15) / 16);
    fp8_blockscale_wgmma_cpa_kernel<16, 16, true, true>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    return;
  }
  const bool use_split8 = N <= 2304 && K >= 4096;
  const bool use_split4 = N <= 2304 && K >= 1536 && K < 4096;
  const bool use_split6 = N > 2304 && N <= 3072 && K >= 4096;
  const bool use_split2 = N > 3072 && M <= 24;
  const bool use_wide_large_k = N > 3072 && K >= 1536;
  const dim3 row8_grid((N + 63) / 64, (M + 7) / 8);
  const bool use_wide_mid_n = N > 2304 && N <= 3072 && K >= 4096;
  if (use_wide_large_k && K >= 2048 && M > 16 && M <= 24) {
    const dim3 grid((N + 63) / 64 * 4, (M + 23) / 24);
    launch_ku<24, 4, true, 2, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
  } else if (use_wide_large_k && M > 8 && M <= 16) {
    if (K >= 2048) {
      const dim3 grid((N + 63) / 64 * 4, (M + 15) / 16);
      fp8_blockscale_wgmma_splitk_kernel<16, 4, true>
          <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
    } else {
      const dim3 grid((N + 63) / 64 * 6, (M + 15) / 16);
      if (M == 16 && N == 8192 && K == 1536)
        launch_ku<16, 6, false, 2, true, false, false, false, true, false, false, true, false, true,
                  true, true, false, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N,
                                                  K);
      else
        launch_ku<16, 6, false, 2, true, false, false, false, true, false, false, true, false,
                  false, true>(grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    }
  } else if (use_wide_mid_n && M > 8 && M <= 16) {
    const dim3 grid((N + 63) / 64 * 8, (M + 15) / 16);
    fp8_blockscale_wgmma_splitk_kernel<16, 8>
        <<<grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M <= 8) {
    static bool opted16 = false;
    if (!opted16) {
      cudaFuncSetAttribute(fp8_blockscale_wgmma_splitk_kernel<8, 16>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      opted16 = true;
    }
    fp8_blockscale_wgmma_splitk_kernel<8, 16>
        <<<dim3(row8_grid.x * 16, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M > 16 && M <= 24) {
    // Narrow N leaves few column tiles, so the grid is small; batching two
    // k-groups per shared round doubles the weight vectors in flight per
    // thread and halves the publish fence/barrier count per weight byte.
    const dim3 grid((N + 63) / 64 * 8, (M + 23) / 24);
    if (M == 24 && N == 2048 && K == 2048) {
      launch_ku<24, 8, false, 2, true, false, true, false, true, false, false, true, true, false,
                false, true, false, true, false, true>(grid, true, stream, xp, wp, wsp, eps, op, M,
                                                       N, K);
    } else {
      launch_ku<24, 8, false, 2, true, false, false, false, true, false, false, true>(
          grid, true, stream, xp, wp, wsp, eps, op, M, N, K);
    }
  } else if (use_split2 && K >= 2048 && M <= 8) {
    if (M > 1) {
      launch_splitk_load_balanced<8, 8, true, true>(dim3(row8_grid.x * 8, row8_grid.y), stream, xp,
                                                    wp, wsp, eps, op, M, N, K);
    } else {
      launch_splitk_load_balanced<8, 8, true>(dim3(row8_grid.x * 8, row8_grid.y), stream, xp, wp,
                                              wsp, eps, op, M, N, K);
    }
  } else if (use_split2 && M <= 8) {
    if (M > 1) {
      launch_splitk_load_balanced<8, 6, false, true>(dim3(row8_grid.x * 6, row8_grid.y), stream, xp,
                                                     wp, wsp, eps, op, M, N, K);
    } else {
      launch_splitk_load_balanced<8, 6>(dim3(row8_grid.x * 6, row8_grid.y), stream, xp, wp, wsp,
                                        eps, op, M, N, K);
    }
  } else if (use_split2) {
    fp8_blockscale_wgmma_splitk_kernel<8, 2>
        <<<dim3(row8_grid.x * 2, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split8 && M <= 8) {
    if ((K >> 7) == 14 * 4) {
      if (M == 2) {
        if (N == 2048)
          launch_ku<8, 14, false, 4, false, true, true, true, true, false, true, false, false,
                    false, true, true, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp,
                                             wp, wsp, eps, op, M, N, K);
        else
          launch_ku<8, 14, false, 4, false, true, true, true, true, false, true, false, false,
                    false, true, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp, wp,
                                       wsp, eps, op, M, N, K);
      } else {
        launch_ku<8, 14, false, 4, false, true, true, true, true, false, false, false, false, false,
                  true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp, wp, wsp, eps, op, M,
                        N, K);
      }
    } else {
      launch_ku<8, 14, false, 4, false, true>(dim3(row8_grid.x * 14, row8_grid.y), true, stream, xp,
                                              wp, wsp, eps, op, M, N, K);
    }
  } else if (use_split8 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 8>
        <<<dim3(row8_grid.x * 8, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split4 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 4>
        <<<dim3(row8_grid.x * 4, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split6 && M > 16 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 8>
        <<<dim3(row8_grid.x * 8, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (use_split6 && M <= 24) {
    fp8_blockscale_wgmma_splitk_kernel<8, 6>
        <<<dim3(row8_grid.x * 6, row8_grid.y), 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else if (M <= 24) {
    fp8_blockscale_wgmma_kernel<8><<<row8_grid, 128, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  } else {
    const dim3 block(64);
    const dim3 grid((N + 63) / 64, (M + 15) / 16);
    fp8_blockscale_kernel<<<grid, block, 0, stream>>>(xp, wp, wsp, eps, op, M, N, K);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
