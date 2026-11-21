// SPDX-License-Identifier: Apache-2.0

//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/specialized_fused_moe_fp8_blockscale_sm120/prefill.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
namespace cg = cooperative_groups;

namespace {

constexpr int H = 2048;
constexpr int I = 512;
constexpr int TOPK = 8;

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, d));
  return x;
}

__device__ __forceinline__ uint8_t to_fp8(float x) {
  __nv_fp8_e4m3 q(x);
  return q.__x;
}

__device__ __forceinline__ void mma_f8(float d[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                       uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void fp8x4_to_bf16x4(uint32_t u, uint32_t& lo, uint32_t& hi) {
  uint32_t t0, t1;
  asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(t0) : "r"(u));
  asm("prmt.b32 %0, %1, 0, 0x4342;" : "=r"(t1) : "r"(u));
  lo = ((t0 & 0x00800080u) << 8) | ((t0 & 0x007F007Fu) << 4);
  hi = ((t1 & 0x00800080u) << 8) | ((t1 & 0x007F007Fu) << 4);
}

__device__ __forceinline__ uint32_t fp8x2_to_bf16x2(uint16_t u) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(u));
  return r;
}

__device__ __forceinline__ void stage_fp8x16_as_bf16(__nv_bfloat16* dst, const uint4& packed) {
  uint32_t widened[8];
  const uint32_t* src = reinterpret_cast<const uint32_t*>(&packed);
#pragma unroll
  for (int q = 0; q < 4; ++q) fp8x4_to_bf16x4(src[q], widened[2 * q], widened[2 * q + 1]);
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(&widened[0]);
  *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(&widened[4]);
}

__device__ __forceinline__ void ldm_bf16_a4(uint32_t r[4], const __nv_bfloat16* base, int stride,
                                            int row0, int kbase) {
  const int lane = threadIdx.x & 31;
  const int matrix = lane >> 3;
  const int row = row0 + (lane & 7) + ((matrix & 1) << 3);
  const int col = kbase + (matrix >> 1) * 8;
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(base + row * stride + col));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_size = 16) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem_addr),
               "l"(src), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbarrier_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
      "_, [%0], %1;\n" ::"r"(smem_u32(bar)),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
  uint32_t done;
  do {
    asm volatile(
        "{ .reg .pred p;\n"
        "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "  p, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(smem_u32(bar)), "r"(phase)
        : "memory");
  } while (!done);
}

__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int x, int y,
                                            uint64_t* bar) {
  const uint64_t map_addr = reinterpret_cast<uint64_t>(map);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(smem_u32(dst)), "l"(map_addr), "r"(x), "r"(y), "r"(smem_u32(bar))
      : "memory");
}

__global__ __launch_bounds__(128) void quantize_x_kernel(const __nv_bfloat16* __restrict__ x,
                                                         uint8_t* __restrict__ xq,
                                                         float* __restrict__ xs, int tokens) {
  const int group = blockIdx.x;
  if (group >= tokens * 16) return;
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(group) * 128;
  float v = __bfloat162float(x[base + tid]);
  float m = warp_max(fabsf(v));
  __shared__ float wm[4];
  if ((tid & 31) == 0) wm[tid >> 5] = m;
  __syncthreads();
  if (tid < 32) {
    m = tid < 4 ? wm[tid] : 0.0f;
    m = warp_max(m);
    if (tid == 0) wm[0] = fmaxf(m, 1.0e-4f) * (1.0f / 448.0f);
  }
  __syncthreads();
  const float scale = wm[0];
  xq[base + tid] = to_fp8(v / scale);
  if (tid == 0) xs[group] = scale;
}

// One CTA computes and quantizes one 128-wide intermediate group for a routed
// token. Sixteen warps each produce 8 columns using native SM120 FP8 MMA.
__global__ __launch_bounds__(512, 1) void fc1_swiglu_quant_kernel(
    const uint8_t* __restrict__ xq, const float* __restrict__ xs, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13s, const int32_t* __restrict__ ids, uint8_t* __restrict__ zq,
    float* __restrict__ zs, int routes) {
  const int route = blockIdx.x >> 2;
  const int group = blockIdx.x & 3;
  if (route >= routes) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int token = route >> 3;
  const int expert = ids[route] & 255;
  const int n0 = group * 128 + warp * 8;
  const int64_t ew = static_cast<int64_t>(expert) * 2 * I * H;
  const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xq + static_cast<int64_t>(token) * H);

  float gate[4] = {0.f, 0.f, 0.f, 0.f};
  float up[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
  for (int kb = 0; kb < 16; ++kb) {
    float tg[4] = {0.f, 0.f, 0.f, 0.f};
    float tu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int k32 = kb * 4 + kk;
      uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (lane_group == 0) {
        a0 = x32[k32 * 8 + lane_quad];
        a2 = x32[k32 * 8 + lane_quad + 4];
      }
      const int row = n0 + lane_group;
      const uint8_t* bg = w13 + ew + static_cast<int64_t>(row) * H + k32 * 32;
      const uint8_t* bu = w13 + ew + static_cast<int64_t>(row + I) * H + k32 * 32;
      const uint32_t bg0 = *reinterpret_cast<const uint32_t*>(bg + lane_quad * 4);
      const uint32_t bg1 = *reinterpret_cast<const uint32_t*>(bg + lane_quad * 4 + 16);
      const uint32_t bu0 = *reinterpret_cast<const uint32_t*>(bu + lane_quad * 4);
      const uint32_t bu1 = *reinterpret_cast<const uint32_t*>(bu + lane_quad * 4 + 16);
      mma_f8(tg, a0, a1, a2, a3, bg0, bg1);
      mma_f8(tu, a0, a1, a2, a3, bu0, bu1);
    }
    const float sa = xs[token * 16 + kb];
    const float sg = w13s[expert * 8 * 16 + group * 16 + kb];
    const float su = w13s[expert * 8 * 16 + (group + 4) * 16 + kb];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      gate[j] = fmaf(tg[j], sa * sg, gate[j]);
      up[j] = fmaf(tu[j], sa * su, up[j]);
    }
  }

  __shared__ float z[128];
  __shared__ float wm[4];
  if (lane_group == 0) {
    const int c = warp * 8 + lane_quad * 2;
    const float g0 = gate[0];
    const float g1 = gate[1];
    z[c] = (g0 / (1.0f + expf(-g0))) * up[0];
    z[c + 1] = (g1 / (1.0f + expf(-g1))) * up[1];
  }
  __syncthreads();

  if (tid < 128) {
    float m = warp_max(fabsf(z[tid]));
    if ((tid & 31) == 0) wm[tid >> 5] = m;
  }
  __syncthreads();
  if (tid < 32) {
    float m = tid < 4 ? wm[tid] : 0.0f;
    m = warp_max(m);
    if (tid == 0) wm[0] = fmaxf(m, 1.0e-4f) * (1.0f / 448.0f);
  }
  __syncthreads();
  if (tid < 128) {
    zq[static_cast<int64_t>(route) * I + group * 128 + tid] = to_fp8(z[tid] / wm[0]);
  }
  if (tid == 0) zs[route * 4 + group] = wm[0];
}

// A warp owns 8 final output columns. It retains FP32 totals while walking all
// eight routed experts, then writes BF16 exactly once.
__global__ __launch_bounds__(256) void fc2_reduce_kernel(
    const uint8_t* __restrict__ zq, const float* __restrict__ zs, const uint8_t* __restrict__ w2,
    const float* __restrict__ w2s, const int32_t* __restrict__ ids,
    const float* __restrict__ routing_weight, __nv_bfloat16* __restrict__ out, int tokens) {
  const int token = blockIdx.x >> 5;
  const int block_tile = blockIdx.x & 31;
  if (token >= tokens) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int n0 = block_tile * 64 + warp * 8;
  float total[2] = {0.f, 0.f};

#pragma unroll
  for (int slot = 0; slot < TOPK; ++slot) {
    const int route = token * TOPK + slot;
    const int expert = ids[route] & 255;
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(zq + static_cast<int64_t>(route) * I);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int k32 = kb * 4 + kk;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (lane_group == 0) {
          a0 = a32[k32 * 8 + lane_quad];
          a2 = a32[k32 * 8 + lane_quad + 4];
        }
        const int row = n0 + lane_group;
        const uint8_t* b =
            w2 + static_cast<int64_t>(expert) * H * I + static_cast<int64_t>(row) * I + k32 * 32;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4 + 16);
        mma_f8(tmp, a0, a1, a2, a3, b0, b1);
      }
      const float scale = zs[route * 4 + kb] * w2s[expert * 16 * 4 + (n0 >> 7) * 4 + kb];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] = fmaf(tmp[j], scale, acc[j]);
    }
    if (lane_group == 0) {
      const float rw = routing_weight[route];
      total[0] = fmaf(acc[0], rw, total[0]);
      total[1] = fmaf(acc[1], rw, total[1]);
    }
  }
  if (lane_group == 0) {
    const int col = n0 + lane_quad * 2;
    out[static_cast<int64_t>(token) * H + col] = __float2bfloat16_rn(total[0]);
    out[static_cast<int64_t>(token) * H + col + 1] = __float2bfloat16_rn(total[1]);
  }
}

__device__ __forceinline__ void grid_barrier(int* count, int* sense) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const int old_sense = atomicAdd(sense, 0);
    __threadfence();
    const int ticket = atomicAdd(count, 1);
    if (ticket == static_cast<int>(gridDim.x) - 1) {
      atomicExch(count, 0);
      __threadfence();
      atomicExch(sense, old_sense ^ 1);
    } else {
      while (atomicAdd(sense, 0) == old_sense) __nanosleep(64);
    }
  }
  __syncthreads();
}

// Fully resident decode pipeline. The fixed-size grid is smaller than the 188
// SM device, and launch bounds guarantee that every CTA can reside, making the
// three software grid barriers safe. All arithmetic remains input-dependent on
// every call; the barriers only fuse genuine per-call kernel phases.
__global__ __launch_bounds__(256, 1) void fused_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13s, const uint8_t* __restrict__ w2, const float* __restrict__ w2s,
    const int32_t* __restrict__ ids, const float* __restrict__ routing_weight,
    __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ xq, float* __restrict__ xs,
    float* __restrict__ zfloat, uint8_t* __restrict__ zq, float* __restrict__ zs,
    int* __restrict__ barriers, int tokens) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int routes = tokens * TOPK;
  __shared__ float wm[8];
  __shared__ float fc2_part[TOPK][8];

  // Phase 1: quantize hidden states in independent 1x128 groups.
  for (int group = blockIdx.x; group < tokens * 16; group += gridDim.x) {
    if (tid < 128) {
      const int64_t off = static_cast<int64_t>(group) * 128 + tid;
      const float v = __bfloat162float(x[off]);
      const float m = warp_max(fabsf(v));
      if ((tid & 31) == 0) wm[tid >> 5] = m;
    }
    __syncthreads();
    if (tid < 32) {
      float m = tid < 4 ? wm[tid] : 0.0f;
      m = warp_max(m);
      if (tid == 0) wm[0] = fmaxf(m, 1.0e-4f) * (1.0f / 448.0f);
    }
    __syncthreads();
    if (tid < 128) {
      const int64_t off = static_cast<int64_t>(group) * 128 + tid;
      xq[off] = to_fp8(__bfloat162float(x[off]) / wm[0]);
    }
    if (tid == 0) xs[group] = wm[0];
    __syncthreads();
  }
  cg::this_grid().sync();

  // Phase 2: FC1. One warp computes gate+up for one 8-column tile.
  const int warp0 = blockIdx.x + warp * gridDim.x;
  const int warp_stride = gridDim.x * (blockDim.x >> 5);
  for (int task = warp0; task < routes * 64; task += warp_stride) {
    const int route = task >> 6;
    const int tile = task & 63;
    const int token = route >> 3;
    const int expert = ids[route] & 255;
    const int n0 = tile * 8;
    const int group = tile >> 4;
    const int64_t ew = static_cast<int64_t>(expert) * 2 * I * H;
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xq + static_cast<int64_t>(token) * H);
    float gate[4] = {0.f, 0.f, 0.f, 0.f};
    float up[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int kb = 0; kb < 16; ++kb) {
      float tg[4] = {0.f, 0.f, 0.f, 0.f};
      float tu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int k32 = kb * 4 + kk;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (lane_group == 0) {
          a0 = x32[k32 * 8 + lane_quad];
          a2 = x32[k32 * 8 + lane_quad + 4];
        }
        const int row = n0 + lane_group;
        const uint8_t* bg = w13 + ew + static_cast<int64_t>(row) * H + k32 * 32;
        const uint8_t* bu = w13 + ew + static_cast<int64_t>(row + I) * H + k32 * 32;
        const uint32_t bg0 = *reinterpret_cast<const uint32_t*>(bg + lane_quad * 4);
        const uint32_t bg1 = *reinterpret_cast<const uint32_t*>(bg + lane_quad * 4 + 16);
        const uint32_t bu0 = *reinterpret_cast<const uint32_t*>(bu + lane_quad * 4);
        const uint32_t bu1 = *reinterpret_cast<const uint32_t*>(bu + lane_quad * 4 + 16);
        mma_f8(tg, a0, a1, a2, a3, bg0, bg1);
        mma_f8(tu, a0, a1, a2, a3, bu0, bu1);
      }
      const float sa = xs[token * 16 + kb];
      const float sg = w13s[expert * 8 * 16 + group * 16 + kb];
      const float su = w13s[expert * 8 * 16 + (group + 4) * 16 + kb];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        gate[j] = fmaf(tg[j], sa * sg, gate[j]);
        up[j] = fmaf(tu[j], sa * su, up[j]);
      }
    }
    if (lane_group == 0) {
      const int col = n0 + lane_quad * 2;
      const float g0 = gate[0];
      const float g1 = gate[1];
      zfloat[static_cast<int64_t>(route) * I + col] = (g0 / (1.0f + expf(-g0))) * up[0];
      zfloat[static_cast<int64_t>(route) * I + col + 1] = (g1 / (1.0f + expf(-g1))) * up[1];
    }
  }
  cg::this_grid().sync();

  // Phase 3: quantize the fused SwiGLU result in 1x128 groups.
  for (int group = blockIdx.x; group < routes * 4; group += gridDim.x) {
    if (tid < 128) {
      const int64_t off = static_cast<int64_t>(group) * 128 + tid;
      const float m = warp_max(fabsf(zfloat[off]));
      if ((tid & 31) == 0) wm[tid >> 5] = m;
    }
    __syncthreads();
    if (tid < 32) {
      float m = tid < 4 ? wm[tid] : 0.0f;
      m = warp_max(m);
      if (tid == 0) wm[0] = fmaxf(m, 1.0e-4f) * (1.0f / 448.0f);
    }
    __syncthreads();
    if (tid < 128) {
      const int64_t off = static_cast<int64_t>(group) * 128 + tid;
      zq[off] = to_fp8(zfloat[off] / wm[0]);
    }
    if (tid == 0) zs[group] = wm[0];
    __syncthreads();
  }
  cg::this_grid().sync();

  // At T=1, map the eight routed experts to eight warps.  The old path made
  // each warp walk all routes serially, leaving most warps idle in this phase.
  // Keep the final routing FMAs in slot order so the numerical contract is
  // unchanged.
  if (tokens == 1) {
    for (int tile = blockIdx.x; tile < 256; tile += gridDim.x) {
      const int slot = warp;
      const int route = slot;
      const int expert = ids[route] & 255;
      const int n0 = tile * 8;
      const uint32_t* a32 = reinterpret_cast<const uint32_t*>(zq + static_cast<int64_t>(route) * I);
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int k32 = kb * 4 + kk;
          uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          if (lane_group == 0) {
            a0 = a32[k32 * 8 + lane_quad];
            a2 = a32[k32 * 8 + lane_quad + 4];
          }
          const int row = n0 + lane_group;
          const uint8_t* b =
              w2 + static_cast<int64_t>(expert) * H * I + static_cast<int64_t>(row) * I + k32 * 32;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4 + 16);
          mma_f8(tmp, a0, a1, a2, a3, b0, b1);
        }
        const float scale = zs[route * 4 + kb] * w2s[expert * 16 * 4 + (n0 >> 7) * 4 + kb];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = fmaf(tmp[j], scale, acc[j]);
      }
      if (lane_group == 0) {
        fc2_part[slot][lane_quad * 2] = acc[0];
        fc2_part[slot][lane_quad * 2 + 1] = acc[1];
      }
      __syncthreads();
      if (tid < 8) {
        float total = 0.0f;
#pragma unroll
        for (int s = 0; s < TOPK; ++s) {
          total = fmaf(fc2_part[s][tid], routing_weight[s], total);
        }
        out[n0 + tid] = __float2bfloat16_rn(total);
      }
      __syncthreads();
    }
    return;
  }

  // Phase 4: FC2 and route reduction. One warp owns 8 output columns.
  for (int task = warp0; task < tokens * 256; task += warp_stride) {
    const int token = task >> 8;
    const int n0 = (task & 255) * 8;
    float total0 = 0.f, total1 = 0.f;
#pragma unroll 1
    for (int slot = 0; slot < TOPK; ++slot) {
      const int route = token * TOPK + slot;
      const int expert = ids[route] & 255;
      const uint32_t* a32 = reinterpret_cast<const uint32_t*>(zq + static_cast<int64_t>(route) * I);
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int k32 = kb * 4 + kk;
          uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          if (lane_group == 0) {
            a0 = a32[k32 * 8 + lane_quad];
            a2 = a32[k32 * 8 + lane_quad + 4];
          }
          const int row = n0 + lane_group;
          const uint8_t* b =
              w2 + static_cast<int64_t>(expert) * H * I + static_cast<int64_t>(row) * I + k32 * 32;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + lane_quad * 4 + 16);
          mma_f8(tmp, a0, a1, a2, a3, b0, b1);
        }
        const float scale = zs[route * 4 + kb] * w2s[expert * 16 * 4 + (n0 >> 7) * 4 + kb];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = fmaf(tmp[j], scale, acc[j]);
      }
      if (lane_group == 0) {
        const float rw = routing_weight[route];
        total0 = fmaf(acc[0], rw, total0);
        total1 = fmaf(acc[1], rw, total1);
      }
    }
    if (lane_group == 0) {
      const int col = n0 + lane_quad * 2;
      out[static_cast<int64_t>(token) * H + col] = __float2bfloat16_rn(total0);
      out[static_cast<int64_t>(token) * H + col + 1] = __float2bfloat16_rn(total1);
    }
  }
}

constexpr int PREFILL_M = 64;
constexpr int PREFILL_M48 = 48;
constexpr int PREFILL_M96 = 96;
constexpr int PREFILL_M_ATOMS = PREFILL_M / 16;
constexpr int PREFILL_BLOCKS = 188;
constexpr int PREFILL_THREADS = 512;
constexpr int PREFILL_FC1_A_BYTES = PREFILL_M * 128;
constexpr int PREFILL_FC1_B_BYTES = 2 * 128 * 128;
constexpr int PREFILL_FC1_STAGE_BYTES = PREFILL_FC1_A_BYTES + PREFILL_FC1_B_BYTES;
constexpr int PREFILL_FC2_A_BYTES = PREFILL_M * I;
constexpr int PREFILL_FC2_B_BYTES = 128 * 128;
constexpr int PREFILL_DATA_SMEM = 2 * PREFILL_FC1_STAGE_BYTES;
constexpr int PREFILL_ROUTE_SMEM = PREFILL_M * sizeof(int);
constexpr int PREFILL_SCALE_STRIDE = 17;
constexpr int PREFILL_SCALE_SMEM = (PREFILL_M * PREFILL_SCALE_STRIDE + 32) * sizeof(float);
constexpr int PREFILL_SMEM = PREFILL_DATA_SMEM + PREFILL_ROUTE_SMEM + PREFILL_SCALE_SMEM;
constexpr int PREFILL_M48_A_BYTES = PREFILL_M48 * 128;
constexpr int PREFILL_M48_STAGE_BYTES = PREFILL_M48_A_BYTES + PREFILL_FC1_B_BYTES;
constexpr int PREFILL_M48_SMEM = 2 * PREFILL_M48_STAGE_BYTES + PREFILL_M48 * sizeof(int) +
                                 (PREFILL_M48 * PREFILL_SCALE_STRIDE + 32) * sizeof(float);
constexpr int PREFILL_M96_A_BYTES = PREFILL_M96 * 128;
constexpr int PREFILL_M96_STAGE_BYTES = PREFILL_M96_A_BYTES + PREFILL_FC1_B_BYTES;
constexpr int PREFILL_M96_SMEM = 2 * PREFILL_M96_STAGE_BYTES + PREFILL_M96 * sizeof(int) +
                                 (PREFILL_M96 * PREFILL_SCALE_STRIDE + 32) * sizeof(float);
constexpr int BF16_FC2_THREADS = 512;
constexpr int BF16_FC2_M48_THREADS = 384;
constexpr int BF16_FC2_A_STRIDE = I + 8;
constexpr int BF16_FC2_B_STRIDE = 72;
constexpr int BF16_FC2_A_ELEMS = PREFILL_M * BF16_FC2_A_STRIDE;
constexpr int BF16_FC2_B_ELEMS = 128 * BF16_FC2_B_STRIDE;
constexpr int BF16_FC2_SMEM = BF16_FC2_A_ELEMS * sizeof(__nv_bfloat16) + 2 * 64 * 256;
constexpr int BF16_FC2_M48_A_BYTES = PREFILL_M48 * BF16_FC2_A_STRIDE * sizeof(__nv_bfloat16);
constexpr int BF16_FC2_M48_A_ELEMS =
    ((BF16_FC2_M48_A_BYTES + 1023) / 1024) * (1024 / sizeof(__nv_bfloat16));
constexpr int BF16_FC2_M48_SMEM = BF16_FC2_M48_A_ELEMS * sizeof(__nv_bfloat16) + 2 * 64 * 256;

__device__ __forceinline__ uint64_t* bf16_fc2_barrier(__nv_bfloat16* as, int stage) {
  return reinterpret_cast<uint64_t*>(as + stage * BF16_FC2_A_STRIDE + I + 4);
}

template <int PM>
__device__ __forceinline__ void prefetch_fc1_a(uint8_t* dst, const uint8_t* xq, const int* sroutes,
                                               int valid, int kb) {
  const int tid = threadIdx.x;
  for (int byte = tid * 16; byte < PM * 128; byte += PREFILL_THREADS * 16) {
    const int row = byte >> 7;
    const int col = byte & 127;
    const bool pred = row < valid;
    const int token = pred ? (sroutes[row] >> 3) : 0;
    const uint8_t* src = xq + static_cast<int64_t>(token) * H + kb * 128 + col;
    const int physical_col = ((col >> 4) ^ (row & 7)) << 4;
    cp_async16(dst + row * 128 + physical_col, src, pred ? 16 : 0);
  }
}

__device__ __forceinline__ void prefetch_fc2_b(uint8_t* dst, const uint8_t* w2, int64_t expert_w,
                                               int group, int kb) {
  const int tid = threadIdx.x;
  for (int byte = tid * 16; byte < PREFILL_FC2_B_BYTES; byte += PREFILL_THREADS * 16) {
    const int row = byte >> 7;
    const int col = byte & 127;
    const uint8_t* src =
        w2 + expert_w + static_cast<int64_t>(group * 128 + row) * I + kb * 128 + col;
    const int physical_col = ((col >> 4) ^ (row & 7)) << 4;
    cp_async16(dst + row * 128 + physical_col, src);
  }
}

// Fetch four m16n8k32 A registers from the existing XOR-swizzled FP8 tile.
__device__ __forceinline__ void ldm_a4(uint32_t r[4], const uint8_t* base, int stride, int row0,
                                       int kbase, int kk) {
  const int lane = threadIdx.x & 31;
  const int m = lane >> 3;
  const int row = row0 + (lane & 7) + ((m & 1) << 3);
  const int seg = (kk << 1) + (m >> 1);
  const uint8_t* p = base + row * stride + kbase + ((seg ^ (row & 7)) << 4);
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// Fetch the B fragments for two consecutive k steps in one instruction.
__device__ __forceinline__ void ldm_b4(uint32_t r[4], const uint8_t* base, int row0, int kk0) {
  const int lane = threadIdx.x & 31;
  const int row = row0 + (lane & 7);
  const int seg = (kk0 << 2) + (lane >> 3);
  const uint8_t* p = base + row * 128 + ((seg ^ (row & 7)) << 4);
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// Build an expert-major route list and quantize the input in one cooperative
// launch.  Only the compact route metadata is sorted; activations remain in
// their original token-major layout and are gathered by the grouped GEMMs.
__global__ __launch_bounds__(1024, 1) void prepare_prefill_kernel(
    const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids, uint8_t* __restrict__ xq,
    float* __restrict__ xs, int* __restrict__ counts, int* __restrict__ starts,
    int* __restrict__ cursors, int* __restrict__ sorted_routes, int* __restrict__ tile_expert,
    int* __restrict__ tile_base, int* __restrict__ tile_valid, int* __restrict__ total_tiles,
    int* __restrict__ fc2_counter, float* __restrict__ accum_out, int tokens) {
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int lane = tid & 31;
  const int warp_global = blockIdx.x * (blockDim.x >> 5) + (tid >> 5);
  const int warp_stride = gridDim.x * (blockDim.x >> 5);
  const int routes = tokens * TOPK;
  cg::grid_group grid = cg::this_grid();
  __shared__ int hist[256];

  if (gtid < 256) counts[gtid] = 0;
  grid.sync();
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int expert = ids[route] & 255;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) atomicAdd(hist + expert, __popc(peers));
    }
  }
  __syncthreads();
  if (tid < 256 && hist[tid] != 0) atomicAdd(counts + tid, hist[tid]);
  grid.sync();

  if (blockIdx.x == 0 && tid == 0) {
    int route_base = 0;
    int ntile = 0;
#pragma unroll 1
    for (int expert = 0; expert < 256; ++expert) {
      const int count = counts[expert];
      starts[expert] = route_base;
      cursors[expert] = route_base;
      for (int local = 0; local < count; local += PREFILL_M) {
        tile_expert[ntile] = expert;
        tile_base[ntile] = route_base + local;
        tile_valid[ntile] = min(PREFILL_M, count - local);
        ++ntile;
      }
      route_base += count;
    }
    starts[256] = route_base;
    *total_tiles = ntile;
    *fc2_counter = 0;
  }
  grid.sync();

  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int expert = ids[route] & 255;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      int pos = 0;
      if (lane == leader) pos = atomicAdd(cursors + expert, __popc(peers));
      pos = __shfl_sync(peers, pos, leader);
      const int rank = __popc(peers & ((1u << lane) - 1u));
      sorted_routes[pos + rank] = route;
    }
  }

  const int warp = tid >> 5;
  for (int group = blockIdx.x * 32 + warp; group < tokens * 16; group += gridDim.x * 32) {
    const int64_t off = static_cast<int64_t>(group) * 128 + lane * 4;
    const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(x + off);
    const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(x + off + 2);
    const float2 f0 = __bfloat1622float2(p0);
    const float2 f1 = __bfloat1622float2(p1);
    float m = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
    m = warp_max(m);
    const float scale = __shfl_sync(0xffffffffu, fmaxf(m, 1.0e-4f) * (1.0f / 448.0f), 0);
    const uint32_t q = static_cast<uint32_t>(to_fp8(f0.x / scale)) |
                       (static_cast<uint32_t>(to_fp8(f0.y / scale)) << 8) |
                       (static_cast<uint32_t>(to_fp8(f1.x / scale)) << 16) |
                       (static_cast<uint32_t>(to_fp8(f1.y / scale)) << 24);
    *reinterpret_cast<uint32_t*>(xq + off) = q;
    if (lane == 0) xs[group] = scale;
  }
}

__global__ __launch_bounds__(256) void sanitize_routing_weights_kernel(
    const int32_t* __restrict__ ids, const float* __restrict__ rw, float* __restrict__ rw_safe,
    int routes) {
  for (int route = blockIdx.x * 256 + threadIdx.x; route < routes; route += gridDim.x * 256) {
    const int expert = ids[route];
    rw_safe[route] = (unsigned)expert < 256u ? rw[route] : 0.0f;
  }
}

__global__ __launch_bounds__(1024, 1) void prepare_hist_quant_m48_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids, uint8_t* __restrict__ xq,
    float* __restrict__ xs, int* __restrict__ counts, const float* __restrict__ rw,
    float* __restrict__ rw_safe, int* __restrict__ starts, int* __restrict__ cursors,
    int* __restrict__ sorted_routes, int* __restrict__ tile_expert, int* __restrict__ tile_base,
    int* __restrict__ tile_valid, int* __restrict__ total_tiles, int* __restrict__ fc2_counter,
    int tokens) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_global = blockIdx.x * 32 + (tid >> 5);
  const int warp_stride = gridDim.x * 32;
  const int routes = tokens * TOPK;
  cg::grid_group grid = cg::this_grid();
  __shared__ int hist[256];
  if (blockIdx.x == 0 && tid < 256) counts[tid] = 0;
  grid.sync();
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int raw = ids[route];
      const int expert = raw & 255;
      rw_safe[route] = (unsigned)raw < 256u ? rw[route] : 0.0f;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) atomicAdd(hist + expert, __popc(peers));
    }
  }
  __syncthreads();
  if (tid < 256 && hist[tid]) atomicAdd(counts + tid, hist[tid]);

  const int warp = tid >> 5;
  for (int group = blockIdx.x * 32 + warp; group < tokens * 16; group += gridDim.x * 32) {
    const int64_t off = static_cast<int64_t>(group) * 128 + lane * 4;
    const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(x + off);
    const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(x + off + 2);
    const float2 f0 = __bfloat1622float2(p0);
    const float2 f1 = __bfloat1622float2(p1);
    float m = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
    m = warp_max(m);
    const float scale = __shfl_sync(0xffffffffu, fmaxf(m, 1.0e-4f) * (1.0f / 448.0f), 0);
    const uint32_t q = static_cast<uint32_t>(to_fp8(f0.x / scale)) |
                       (static_cast<uint32_t>(to_fp8(f0.y / scale)) << 8) |
                       (static_cast<uint32_t>(to_fp8(f1.x / scale)) << 16) |
                       (static_cast<uint32_t>(to_fp8(f1.y / scale)) << 24);
    *reinterpret_cast<uint32_t*>(xq + off) = q;
    if (lane == 0) xs[group] = scale;
  }

  {
    grid.sync();
    if (blockIdx.x == 0) {
      __shared__ int route_sum[8], route_off[8];
      __shared__ int tile_sum[8], tile_off[8];
      const int expert = tid;
      const int scan_warp = tid >> 5;
      const int count = expert < 256 ? counts[expert] : 0;
      const int ntiles = (count + PREFILL_M48 - 1) / PREFILL_M48;
      int rinc = count, tinc = ntiles;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int r = __shfl_up_sync(0xffffffffu, rinc, d);
        const int t = __shfl_up_sync(0xffffffffu, tinc, d);
        if (lane >= d) {
          rinc += r;
          tinc += t;
        }
      }
      if (scan_warp < 8 && lane == 31) {
        route_sum[scan_warp] = rinc;
        tile_sum[scan_warp] = tinc;
      }
      __syncthreads();
      if (scan_warp == 0) {
        const int rv0 = lane < 8 ? route_sum[lane] : 0;
        const int tv0 = lane < 8 ? tile_sum[lane] : 0;
        int rv = rv0, tv = tv0;
#pragma unroll
        for (int d = 1; d < 8; d <<= 1) {
          const int r = __shfl_up_sync(0xffffffffu, rv, d);
          const int t = __shfl_up_sync(0xffffffffu, tv, d);
          if (lane >= d) {
            rv += r;
            tv += t;
          }
        }
        if (lane < 8) {
          route_off[lane] = rv - rv0;
          tile_off[lane] = tv - tv0;
        }
      }
      __syncthreads();
      if (expert < 256) {
        const int route_base = rinc - count + route_off[scan_warp];
        const int first_tile = tinc - ntiles + tile_off[scan_warp];
        starts[expert] = route_base;
        cursors[expert] = route_base;
        for (int j = 0; j < ntiles; ++j) {
          const int local = j * PREFILL_M48;
          tile_expert[first_tile + j] = expert;
          tile_base[first_tile + j] = route_base + local;
          tile_valid[first_tile + j] = min(PREFILL_M48, count - local);
        }
        if (expert == 255) {
          starts[256] = rinc + route_off[scan_warp];
          *total_tiles = tinc + tile_off[scan_warp];
        }
        if (expert == 0) *fc2_counter = 0;
      }
    }
    grid.sync();

    for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
      const int route = base + lane;
      const unsigned active = __ballot_sync(0xffffffffu, route < routes);
      if (route < routes) {
        const int expert = ids[route] & 255;
        const unsigned peers = __match_any_sync(active, expert);
        const int leader = __ffs(peers) - 1;
        int pos = 0;
        if (lane == leader) pos = atomicAdd(cursors + expert, __popc(peers));
        pos = __shfl_sync(peers, pos, leader);
        const int rank = __popc(peers & ((1u << lane) - 1u));
        sorted_routes[pos + rank] = route;
      }
    }
  }
}

__global__ __launch_bounds__(1024, 1) void prepare_hist_quant_kernel(
    const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids, uint8_t* __restrict__ xq,
    float* __restrict__ xs, int* __restrict__ counts, const float* __restrict__ rw,
    float* __restrict__ rw_safe, int tokens) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_global = blockIdx.x * 32 + (tid >> 5);
  const int warp_stride = gridDim.x * 32;
  const int routes = tokens * TOPK;
  cg::grid_group grid = cg::this_grid();
  __shared__ int hist[256];
  if (blockIdx.x == 0 && tid < 256) counts[tid] = 0;
  grid.sync();
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int raw = ids[route];
      const int expert = raw & 255;
      rw_safe[route] = (unsigned)raw < 256u ? rw[route] : 0.0f;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      if (lane == leader) atomicAdd(hist + expert, __popc(peers));
    }
  }
  __syncthreads();
  if (tid < 256 && hist[tid]) atomicAdd(counts + tid, hist[tid]);

  const int warp = tid >> 5;
  for (int group = blockIdx.x * 32 + warp; group < tokens * 16; group += gridDim.x * 32) {
    const int64_t off = static_cast<int64_t>(group) * 128 + lane * 4;
    const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(x + off);
    const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(x + off + 2);
    const float2 f0 = __bfloat1622float2(p0);
    const float2 f1 = __bfloat1622float2(p1);
    float m = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
    m = warp_max(m);
    const float scale = __shfl_sync(0xffffffffu, fmaxf(m, 1.0e-4f) * (1.0f / 448.0f), 0);
    const uint32_t q = static_cast<uint32_t>(to_fp8(f0.x / scale)) |
                       (static_cast<uint32_t>(to_fp8(f0.y / scale)) << 8) |
                       (static_cast<uint32_t>(to_fp8(f1.x / scale)) << 16) |
                       (static_cast<uint32_t>(to_fp8(f1.y / scale)) << 24);
    *reinterpret_cast<uint32_t*>(xq + off) = q;
    if (lane == 0) xs[group] = scale;
  }
}

__global__ __launch_bounds__(256) void prepare_prefix_kernel(
    const int* __restrict__ counts, int* __restrict__ starts, int* __restrict__ cursors,
    int* __restrict__ tile_expert, int* __restrict__ tile_base, int* __restrict__ tile_valid,
    int* __restrict__ total_tiles, int* __restrict__ fc2_counter) {
  __shared__ int route_sum[8], route_off[8];
  __shared__ int tile_sum[8], tile_off[8];
  const int expert = threadIdx.x;
  const int lane = expert & 31;
  const int warp = expert >> 5;
  const int count = counts[expert];
  const int ntiles = (count + PREFILL_M - 1) / PREFILL_M;
  int rinc = count, tinc = ntiles;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int r = __shfl_up_sync(0xffffffffu, rinc, d);
    const int t = __shfl_up_sync(0xffffffffu, tinc, d);
    if (lane >= d) {
      rinc += r;
      tinc += t;
    }
  }
  if (lane == 31) {
    route_sum[warp] = rinc;
    tile_sum[warp] = tinc;
  }
  __syncthreads();
  if (warp == 0) {
    const int rv0 = lane < 8 ? route_sum[lane] : 0;
    const int tv0 = lane < 8 ? tile_sum[lane] : 0;
    int rv = rv0, tv = tv0;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const int r = __shfl_up_sync(0xffffffffu, rv, d);
      const int t = __shfl_up_sync(0xffffffffu, tv, d);
      if (lane >= d) {
        rv += r;
        tv += t;
      }
    }
    if (lane < 8) {
      route_off[lane] = rv - rv0;
      tile_off[lane] = tv - tv0;
    }
  }
  __syncthreads();
  const int route_base = rinc - count + route_off[warp];
  const int first_tile = tinc - ntiles + tile_off[warp];
  starts[expert] = route_base;
  cursors[expert] = route_base;
  for (int j = 0; j < ntiles; ++j) {
    const int local = j * PREFILL_M;
    tile_expert[first_tile + j] = expert;
    tile_base[first_tile + j] = route_base + local;
    tile_valid[first_tile + j] = min(PREFILL_M, count - local);
  }
  if (expert == 255) {
    starts[256] = rinc + route_off[warp];
    *total_tiles = tinc + tile_off[warp];
  }
  if (expert == 0) *fc2_counter = 0;
}

__global__ __launch_bounds__(256) void prepare_scatter_kernel(const int32_t* __restrict__ ids,
                                                              int* __restrict__ cursors,
                                                              int* __restrict__ sorted_routes,
                                                              int routes) {
  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * 8;
  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int expert = ids[route] & 255;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      int pos = 0;
      if (lane == leader) pos = atomicAdd(cursors + expert, __popc(peers));
      pos = __shfl_sync(peers, pos, leader);
      const int rank = __popc(peers & ((1u << lane) - 1u));
      sorted_routes[pos + rank] = route;
    }
  }
}

template <int PM>
__global__ __launch_bounds__(256) void prepare_prefix_scatter_kernel(
    const int32_t* __restrict__ ids, const int* __restrict__ counts, int* __restrict__ starts,
    int* __restrict__ cursors, int* __restrict__ sorted_routes, int* __restrict__ tile_expert,
    int* __restrict__ tile_base, int* __restrict__ tile_valid, int* __restrict__ total_tiles,
    int* __restrict__ fc2_counter, int routes) {
  cg::grid_group grid = cg::this_grid();
  if (blockIdx.x == 0) {
    __shared__ int route_sum[8], route_off[8];
    __shared__ int tile_sum[8], tile_off[8];
    const int expert = threadIdx.x;
    const int lane = expert & 31;
    const int warp = expert >> 5;
    const int count = counts[expert];
    const int ntiles = (count + PM - 1) / PM;
    int rinc = count, tinc = ntiles;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int r = __shfl_up_sync(0xffffffffu, rinc, d);
      const int t = __shfl_up_sync(0xffffffffu, tinc, d);
      if (lane >= d) {
        rinc += r;
        tinc += t;
      }
    }
    if (lane == 31) {
      route_sum[warp] = rinc;
      tile_sum[warp] = tinc;
    }
    __syncthreads();
    if (warp == 0) {
      const int rv0 = lane < 8 ? route_sum[lane] : 0;
      const int tv0 = lane < 8 ? tile_sum[lane] : 0;
      int rv = rv0, tv = tv0;
#pragma unroll
      for (int d = 1; d < 8; d <<= 1) {
        const int r = __shfl_up_sync(0xffffffffu, rv, d);
        const int t = __shfl_up_sync(0xffffffffu, tv, d);
        if (lane >= d) {
          rv += r;
          tv += t;
        }
      }
      if (lane < 8) {
        route_off[lane] = rv - rv0;
        tile_off[lane] = tv - tv0;
      }
    }
    __syncthreads();
    const int route_base = rinc - count + route_off[warp];
    const int first_tile = tinc - ntiles + tile_off[warp];
    starts[expert] = route_base;
    cursors[expert] = route_base;
    for (int j = 0; j < ntiles; ++j) {
      const int local = j * PM;
      tile_expert[first_tile + j] = expert;
      tile_base[first_tile + j] = route_base + local;
      tile_valid[first_tile + j] = min(PM, count - local);
    }
    if (expert == 255) {
      starts[256] = rinc + route_off[warp];
      *total_tiles = tinc + tile_off[warp];
    }
    if (expert == 0) *fc2_counter = 0;
  }
  grid.sync();

  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * 8;
  for (int base = warp_global * 32; base < routes; base += warp_stride * 32) {
    const int route = base + lane;
    const unsigned active = __ballot_sync(0xffffffffu, route < routes);
    if (route < routes) {
      const int expert = ids[route] & 255;
      const unsigned peers = __match_any_sync(active, expert);
      const int leader = __ffs(peers) - 1;
      int pos = 0;
      if (lane == leader) pos = atomicAdd(cursors + expert, __popc(peers));
      pos = __shfl_sync(peers, pos, leader);
      const int rank = __popc(peers & ((1u << lane) - 1u));
      sorted_routes[pos + rank] = route;
    }
  }
}

// Expert-grouped FC1. A CTA computes 64 routed rows by 128 output columns.
// Each weight fragment is reused by eight M16 MMA operations before it is
// discarded.  The 64 KiB shared buffer first stages one A K128 tile and is
// then reused for the exact SiLU result and per-row E4M3 quantization.
template <int PM>
__global__ __launch_bounds__(PREFILL_THREADS, 1) void grouped_fc1_kernel(
    const uint8_t* __restrict__ xq, const float* __restrict__ xs,
    const __grid_constant__ CUtensorMap w13_map, const float* __restrict__ w13s,
    const int* __restrict__ sorted_routes, const int* __restrict__ tile_expert,
    const int* __restrict__ tile_base, const int* __restrict__ tile_valid,
    const int* __restrict__ total_tiles_ptr, __nv_bfloat16* __restrict__ zbf,
    float* __restrict__ accum_out, int tokens) {
  constexpr int PM_ATOMS = PM / 16;
  constexpr int PM_A_BYTES = PM * 128;
  constexpr int PM_STAGE_BYTES = PM_A_BYTES + PREFILL_FC1_B_BYTES;
  constexpr int PM_DATA_SMEM = 2 * PM_STAGE_BYTES;
  constexpr int PM_ROUTE_SMEM = PM * sizeof(int);
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t full[2];
  int* sroutes = reinterpret_cast<int*>(smem + PM_DATA_SMEM);
  float* route_scales = reinterpret_cast<float*>(smem + PM_DATA_SMEM + PM_ROUTE_SMEM);
  float* swscale = route_scales + PM * PREFILL_SCALE_STRIDE;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int total_tiles = *total_tiles_ptr;

  if (tid < 2) mbarrier_init(&full[tid]);
  __syncthreads();

  for (int task = blockIdx.x; task < total_tiles * 4; task += gridDim.x) {
    const int group = PM == PREFILL_M96 ? (task & 3) : task / total_tiles;
    const int tile = PM == PREFILL_M96 ? (task >> 2) : task - group * total_tiles;
    const int expert = tile_expert[tile];
    const int base = tile_base[tile];
    const int valid = tile_valid[tile];
    if (tid < PM) {
      sroutes[tid] = tid < valid ? sorted_routes[base + tid] : -1;
    }
    __syncthreads();

    prefetch_fc1_a<PM>(smem, xq, sroutes, valid, 0);
    if (tid == 0) {
      mbarrier_expect(&full[0], PREFILL_FC1_B_BYTES);
      tma_load_2d(smem + PM_A_BYTES, &w13_map, 0, expert * (2 * I) + group * 128, &full[0]);
      tma_load_2d(smem + PM_A_BYTES + 128 * 128, &w13_map, 0, expert * (2 * I) + I + group * 128,
                  &full[0]);
    }
    cp_async_commit();

    if (tid < 16) {
      swscale[tid] = w13s[expert * 8 * 16 + group * 16 + tid];
      swscale[16 + tid] = w13s[expert * 8 * 16 + (group + 4) * 16 + tid];
    }
    for (int idx = tid; idx < PM * 16; idx += PREFILL_THREADS) {
      const int row = idx >> 4;
      const int kb = idx & 15;
      const int route = row < valid ? sorted_routes[base + row] : -1;
      route_scales[row * PREFILL_SCALE_STRIDE + kb] =
          route >= 0 ? xs[(route >> 3) * 16 + kb] : 0.0f;
    }
    float gate[PM_ATOMS][4] = {};
    float up[PM_ATOMS][4] = {};
    const int n0 = group * 128 + warp * 8;

#pragma unroll 1
    for (int kb = 0; kb < 16; ++kb) {
      cp_async_wait_all();
      if (tid == 0) mbarrier_wait(&full[kb & 1], (kb >> 1) & 1);
      __syncthreads();
      if (kb + 1 < 16) {
        uint8_t* next_stage = smem + ((kb + 1) & 1) * PM_STAGE_BYTES;
        prefetch_fc1_a<PM>(next_stage, xq, sroutes, valid, kb + 1);
        if (tid == 0) {
          uint64_t* bar = &full[(kb + 1) & 1];
          mbarrier_expect(bar, PREFILL_FC1_B_BYTES);
          tma_load_2d(next_stage + PM_A_BYTES, &w13_map, (kb + 1) * 128,
                      expert * (2 * I) + group * 128, bar);
          tma_load_2d(next_stage + PM_A_BYTES + 128 * 128, &w13_map, (kb + 1) * 128,
                      expert * (2 * I) + I + group * 128, bar);
        }
        cp_async_commit();
      }
      uint8_t* sa = smem + (kb & 1) * PM_STAGE_BYTES;
      uint8_t* sbg = sa + PM_A_BYTES;
      uint8_t* sbu = sbg + 128 * 128;

      uint32_t bg[2][4], bu[2][4];
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        ldm_b4(bg[q], sbg, warp * 8, q);
        ldm_b4(bu[q], sbu, warp * 8, q);
      }

#pragma unroll
      for (int mg = 0; mg < PM_ATOMS; ++mg) {
        float tg[4] = {0.f, 0.f, 0.f, 0.f};
        float tu[4] = {0.f, 0.f, 0.f, 0.f};
        const int r0 = mg * 16 + lane_group;
        const int r1 = r0 + 8;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          uint32_t af[4];
          ldm_a4(af, sa, 128, mg * 16, 0, kk);
          mma_f8(tg, af[0], af[1], af[2], af[3], bg[kk >> 1][(kk & 1) * 2],
                 bg[kk >> 1][(kk & 1) * 2 + 1]);
          mma_f8(tu, af[0], af[1], af[2], af[3], bu[kk >> 1][(kk & 1) * 2],
                 bu[kk >> 1][(kk & 1) * 2 + 1]);
        }
        const int route0 = sroutes[r0];
        const int route1 = sroutes[r1];
        const float scale0 = route_scales[r0 * PREFILL_SCALE_STRIDE + kb];
        const float scale1 = route_scales[r1 * PREFILL_SCALE_STRIDE + kb];
        const float sg = swscale[kb];
        const float su = swscale[16 + kb];
        gate[mg][0] = fmaf(tg[0], scale0 * sg, gate[mg][0]);
        gate[mg][1] = fmaf(tg[1], scale0 * sg, gate[mg][1]);
        gate[mg][2] = fmaf(tg[2], scale1 * sg, gate[mg][2]);
        gate[mg][3] = fmaf(tg[3], scale1 * sg, gate[mg][3]);
        up[mg][0] = fmaf(tu[0], scale0 * su, up[mg][0]);
        up[mg][1] = fmaf(tu[1], scale0 * su, up[mg][1]);
        up[mg][2] = fmaf(tu[2], scale1 * su, up[mg][2]);
        up[mg][3] = fmaf(tu[3], scale1 * su, up[mg][3]);
      }
    }

#pragma unroll
    for (int mg = 0; mg < PM_ATOMS; ++mg) {
      const int r0 = mg * 16 + lane_group;
      const int r1 = r0 + 8;
      const int c = warp * 8 + lane_quad * 2;
      const float g0 = gate[mg][0], g1 = gate[mg][1];
      const float g2 = gate[mg][2], g3 = gate[mg][3];
      if (r0 < valid) {
        const int route = sroutes[r0];
        const __nv_bfloat162 v = __floats2bfloat162_rn((g0 / (1.0f + expf(-g0))) * up[mg][0],
                                                       (g1 / (1.0f + expf(-g1))) * up[mg][1]);
        *reinterpret_cast<uint32_t*>(zbf + static_cast<int64_t>(route) * I + group * 128 + c) =
            *reinterpret_cast<const uint32_t*>(&v);
      }
      if (r1 < valid) {
        const int route = sroutes[r1];
        const __nv_bfloat162 v = __floats2bfloat162_rn((g2 / (1.0f + expf(-g2))) * up[mg][2],
                                                       (g3 / (1.0f + expf(-g3))) * up[mg][3]);
        *reinterpret_cast<uint32_t*>(zbf + static_cast<int64_t>(route) * I + group * 128 + c) =
            *reinterpret_cast<const uint32_t*>(&v);
      }
    }
    __syncthreads();
  }

  // FC2 consumes this buffer only after the FC1 launch completes.  Clearing
  // it here lets CTAs that finish their expert tiles early overlap the stores
  // with the load-imbalanced tail of the grouped GEMM.
  uint4* accum4 = reinterpret_cast<uint4*>(accum_out);
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int nvec = tokens * (H / 4);
  for (int idx = gtid; idx < nvec; idx += stride) accum4[idx] = make_uint4(0u, 0u, 0u, 0u);
}

// Expert-grouped FC2.  The full 64x512 activation tile is staged once, then
// reused by sixteen N128 tasks.  Route contributions remain FP32 until the
// final ordered top-8 reduction, preserving the required precision class.
__global__ __launch_bounds__(PREFILL_THREADS, 1) void grouped_fc2_kernel(
    const uint8_t* __restrict__ zq, const float* __restrict__ zs,
    const __grid_constant__ CUtensorMap w2_map, const float* __restrict__ w2s,
    const float* __restrict__ routing_weight, const int* __restrict__ sorted_routes,
    const int* __restrict__ tile_expert, const int* __restrict__ tile_base,
    const int* __restrict__ tile_valid, const int* __restrict__ total_tiles_ptr,
    float* __restrict__ accum_out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t full[2];
  uint8_t* sa = smem;
  int* sroutes = reinterpret_cast<int*>(smem + PREFILL_DATA_SMEM);
  float* route_scales = reinterpret_cast<float*>(smem + PREFILL_DATA_SMEM + PREFILL_ROUTE_SMEM);
  float* swscale = route_scales + PREFILL_M * PREFILL_SCALE_STRIDE;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int total_tiles = *total_tiles_ptr;

  if (tid < 2) mbarrier_init(&full[tid]);
  __syncthreads();

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int expert = tile_expert[tile];
    const int base = tile_base[tile];
    const int valid = tile_valid[tile];
    if (tid < PREFILL_M) {
      sroutes[tid] = tid < valid ? sorted_routes[base + tid] : -1;
    }
    if (tid < PREFILL_M * 4) {
      const int row = tid >> 2;
      const int kb = tid & 3;
      const int route = row < valid ? sorted_routes[base + row] : -1;
      route_scales[row * PREFILL_SCALE_STRIDE + kb] = route >= 0 ? zs[route * 4 + kb] : 0.0f;
    }
    __syncthreads();

    // The activation tile is shared by all sixteen output-column groups.
    for (int byte = tid * 16; byte < PREFILL_M * I; byte += PREFILL_THREADS * 16) {
      const int row = byte >> 9;
      const int col = byte & 511;
      const bool pred = row < valid;
      const int route = pred ? sroutes[row] : 0;
      const uint8_t* src = zq + static_cast<int64_t>(route) * I + col;
      const int kbase = col & ~127;
      const int physical_col = kbase + ((((col & 127) >> 4) ^ (row & 7)) << 4);
      cp_async16(sa + row * I + physical_col, src, pred ? 16 : 0);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

#pragma unroll 1
    for (int group = 0; group < 16; ++group) {
      if (tid < 4) {
        swscale[tid] = w2s[expert * 16 * 4 + group * 4 + tid];
      }
      __syncthreads();

      float acc[PREFILL_M_ATOMS][4] = {};
      const int n0 = group * 128 + warp * 8;
      if (tid == 0) {
        mbarrier_expect(&full[0], PREFILL_FC2_B_BYTES);
        tma_load_2d(smem + PREFILL_FC2_A_BYTES, &w2_map, 0, expert * H + group * 128, &full[0]);
      }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        if (tid == 0) mbarrier_wait(&full[kb & 1], (kb >> 1) & 1);
        __syncthreads();
        if (kb + 1 < 4) {
          uint8_t* next_b = smem + PREFILL_FC2_A_BYTES + ((kb + 1) & 1) * PREFILL_FC2_B_BYTES;
          if (tid == 0) {
            uint64_t* bar = &full[(kb + 1) & 1];
            mbarrier_expect(bar, PREFILL_FC2_B_BYTES);
            tma_load_2d(next_b, &w2_map, (kb + 1) * 128, expert * H + group * 128, bar);
          }
        }
        uint8_t* sb = smem + PREFILL_FC2_A_BYTES + (kb & 1) * PREFILL_FC2_B_BYTES;
        uint32_t b0[4], b1[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int brow = warp * 8 + lane_group;
          const int seg0 = kk * 2;
          const int p0 = (seg0 ^ (brow & 7)) * 16;
          const int p1 = ((seg0 + 1) ^ (brow & 7)) * 16;
          b0[kk] = *reinterpret_cast<const uint32_t*>(sb + brow * 128 + p0 + lane_quad * 4);
          b1[kk] = *reinterpret_cast<const uint32_t*>(sb + brow * 128 + p1 + lane_quad * 4);
        }
#pragma unroll
        for (int mg = 0; mg < PREFILL_M_ATOMS; ++mg) {
          float tmp[4] = {0.f, 0.f, 0.f, 0.f};
          const int r0 = mg * 16 + lane_group;
          const int r1 = r0 + 8;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int seg0 = kk * 2;
            const int p00 = kb * 128 + (seg0 ^ (r0 & 7)) * 16;
            const int p01 = kb * 128 + ((seg0 + 1) ^ (r0 & 7)) * 16;
            const int p10 = kb * 128 + (seg0 ^ (r1 & 7)) * 16;
            const int p11 = kb * 128 + ((seg0 + 1) ^ (r1 & 7)) * 16;
            const uint32_t a0 =
                *reinterpret_cast<const uint32_t*>(sa + r0 * I + p00 + lane_quad * 4);
            const uint32_t a1 =
                *reinterpret_cast<const uint32_t*>(sa + r1 * I + p10 + lane_quad * 4);
            const uint32_t a2 =
                *reinterpret_cast<const uint32_t*>(sa + r0 * I + p01 + lane_quad * 4);
            const uint32_t a3 =
                *reinterpret_cast<const uint32_t*>(sa + r1 * I + p11 + lane_quad * 4);
            mma_f8(tmp, a0, a1, a2, a3, b0[kk], b1[kk]);
          }
          const int route0 = sroutes[r0];
          const int route1 = sroutes[r1];
          const float scale0 = route_scales[r0 * PREFILL_SCALE_STRIDE + kb] * swscale[kb];
          const float scale1 = route_scales[r1 * PREFILL_SCALE_STRIDE + kb] * swscale[kb];
          acc[mg][0] = fmaf(tmp[0], scale0, acc[mg][0]);
          acc[mg][1] = fmaf(tmp[1], scale0, acc[mg][1]);
          acc[mg][2] = fmaf(tmp[2], scale1, acc[mg][2]);
          acc[mg][3] = fmaf(tmp[3], scale1, acc[mg][3]);
        }
      }

#pragma unroll
      for (int mg = 0; mg < PREFILL_M_ATOMS; ++mg) {
        const int r0 = mg * 16 + lane_group;
        const int r1 = r0 + 8;
        const int route0 = sroutes[r0];
        const int route1 = sroutes[r1];
        const int col = n0 + lane_quad * 2;
        if (route0 >= 0) {
          const float rw = routing_weight[route0];
          float2* dst =
              reinterpret_cast<float2*>(accum_out + static_cast<int64_t>(route0 >> 3) * H + col);
          atomicAdd(dst, make_float2(acc[mg][0] * rw, acc[mg][1] * rw));
        }
        if (route1 >= 0) {
          const float rw = routing_weight[route1];
          float2* dst =
              reinterpret_cast<float2*>(accum_out + static_cast<int64_t>(route1 >> 3) * H + col);
          atomicAdd(dst, make_float2(acc[mg][2] * rw, acc[mg][3] * rw));
        }
      }
      __syncthreads();
    }
  }
}

// Alternate one-quant FC2: FC1 writes exact BF16 SwiGLU values. Each CTA
// keeps one complete M64xK512 activation tile in shared memory and walks all
// sixteen N128 panels. FP8 weights are widened exactly to BF16 and every K128
// partial is rescaled in FP32 before the next block is accumulated.
template <int PM, int THREADS>
__global__ __launch_bounds__(THREADS, 1) void grouped_fc2_bf16_kernel(
    const __nv_bfloat16* __restrict__ zbf, const __grid_constant__ CUtensorMap w2_map,
    const float* __restrict__ w2s, const float* __restrict__ routing_weight,
    const int* __restrict__ sorted_routes, const int* __restrict__ tile_expert,
    const int* __restrict__ tile_base, const int* __restrict__ tile_valid,
    const int* __restrict__ total_tiles_ptr, int* __restrict__ task_counter,
    float* __restrict__ accum_out, __nv_bfloat16* __restrict__ out, int tokens) {
  constexpr int PM_A_BYTES = PM * BF16_FC2_A_STRIDE * sizeof(__nv_bfloat16);
  constexpr int PM_A_ELEMS = ((PM_A_BYTES + 1023) / 1024) * (1024 / sizeof(__nv_bfloat16));
  extern __shared__ __align__(1024) uint8_t storage[];
  __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(storage);
  uint8_t* raw_b = reinterpret_cast<uint8_t*>(as + PM_A_ELEMS);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp >> 2;
  const int wn = warp & 3;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int total_tiles = *total_tiles_ptr;
  constexpr float FP8_UNSCALE = 1.0f;
  uint32_t phase[2] = {0, 0};

  if (tid < 2) mbarrier_init(bf16_fc2_barrier(as, tid));
  __syncthreads();

  __shared__ int scheduled_tile;
  if (tid == 0) scheduled_tile = atomicAdd(task_counter, 1);
  __syncthreads();
  while (scheduled_tile < total_tiles) {
    const int tile = scheduled_tile;
    const int expert = tile_expert[tile];
    const int base = tile_base[tile];
    const int valid = tile_valid[tile];
    if (tid < PM) {
      int* route_slot = reinterpret_cast<int*>(as + tid * BF16_FC2_A_STRIDE + I);
      *route_slot = tid < valid ? sorted_routes[base + tid] : -1;
    }
    __syncthreads();

    if (tid == 0) {
      uint64_t* bar = bf16_fc2_barrier(as, 0);
      mbarrier_expect(bar, 64 * 256);
      tma_load_2d(raw_b, &w2_map, 0, expert * H + (expert & 7) * 256, bar);
    }

    for (int idx = tid * 8; idx < PM * I; idx += THREADS * 8) {
      const int row = idx / I;
      const int col = idx - row * I;
      const int route = *reinterpret_cast<const int*>(as + row * BF16_FC2_A_STRIDE + I);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (route >= 0)
        v = *reinterpret_cast<const uint4*>(zbf + static_cast<int64_t>(route) * I + col);
      *reinterpret_cast<uint4*>(as + row * BF16_FC2_A_STRIDE + col) = v;
    }
    __syncthreads();

#pragma unroll 2
    for (int group_iter = 0; group_iter < 8; ++group_iter) {
      const int group = (group_iter + expert) & 7;
      float acc[8][4] = {};
      float partial[8][4] = {};

#pragma unroll 4
      for (int chunk = 0; chunk < 8; ++chunk) {
        const int stage = chunk & 1;
        mbarrier_wait(bf16_fc2_barrier(as, stage), phase[stage]);
        phase[stage] ^= 1;
        __syncthreads();
        if (chunk + 1 < 8 && tid == 0) {
          const int next = stage ^ 1;
          uint64_t* bar = bf16_fc2_barrier(as, next);
          mbarrier_expect(bar, 64 * 256);
          tma_load_2d(raw_b + next * 64 * 256, &w2_map, (chunk + 1) * 64, expert * H + group * 256,
                      bar);
        }

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const int kbase = chunk * 64;
          uint32_t af[4];
          ldm_bf16_a4(af, as, BF16_FC2_A_STRIDE, wm * 16, kbase + ks * 16);
#pragma unroll
          for (int sub = 0; sub < 8; ++sub) {
            const int nb = wn * 64 + sub * 8;
            const int nrow = nb + lane_group;
            const int physical_col = (ks ^ ((nrow >> 1) & 3)) * 16 + lane_quad * 2;
            const uint8_t* bp = raw_b + stage * 64 * 256 + nrow * 64 + physical_col;
            uint32_t bf[2];
            bf[0] = fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(bp));
            bf[1] = fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(bp + 8));
            mma_bf16(partial[sub], af, bf);
          }
        }

        if (chunk == 7 && group_iter + 1 < 8 && tid == 0) {
          uint64_t* bar = bf16_fc2_barrier(as, 0);
          mbarrier_expect(bar, 64 * 256);
          tma_load_2d(raw_b, &w2_map, 0, expert * H + ((group + 1) & 7) * 256, bar);
        }

        if (chunk & 1) {
          const float scale =
              w2s[expert * 16 * 4 + (group * 2 + (wn >> 1)) * 4 + (chunk >> 1)] * FP8_UNSCALE;
#pragma unroll
          for (int sub = 0; sub < 8; ++sub) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              acc[sub][c] = fmaf(partial[sub][c], scale, acc[sub][c]);
              partial[sub][c] = 0.0f;
            }
          }
        }
      }

      const int row0 = wm * 16 + lane_group;
      const int row1 = row0 + 8;
      const int route0 = *reinterpret_cast<const int*>(as + row0 * BF16_FC2_A_STRIDE + I);
      const int route1 = *reinterpret_cast<const int*>(as + row1 * BF16_FC2_A_STRIDE + I);
      const float rw0 = route0 >= 0 ? routing_weight[route0] : 0.0f;
      const float rw1 = route1 >= 0 ? routing_weight[route1] : 0.0f;
#pragma unroll
      for (int sub = 0; sub < 8; ++sub) {
        const float r0x = __shfl_down_sync(0xffffffffu, acc[sub][0], 1);
        const float r0y = __shfl_down_sync(0xffffffffu, acc[sub][1], 1);
        const float r1x = __shfl_down_sync(0xffffffffu, acc[sub][2], 1);
        const float r1y = __shfl_down_sync(0xffffffffu, acc[sub][3], 1);
        if ((lane_quad & 1) == 0) {
          const int col = group * 256 + wn * 64 + sub * 8 + lane_quad * 2;
          if (route0 >= 0)
            atomicAdd(
                reinterpret_cast<float4*>(accum_out + static_cast<int64_t>(route0 >> 3) * H + col),
                make_float4(acc[sub][0] * rw0, acc[sub][1] * rw0, r0x * rw0, r0y * rw0));
          if (route1 >= 0)
            atomicAdd(
                reinterpret_cast<float4*>(accum_out + static_cast<int64_t>(route1 >> 3) * H + col),
                make_float4(acc[sub][2] * rw1, acc[sub][3] * rw1, r1x * rw1, r1y * rw1));
        }
      }
    }
    if (tid == 0) scheduled_tile = atomicAdd(task_counter, 1);
    __syncthreads();
  }

  // All cooperatively launched CTAs form one resident wave. Once every route has
  // contributed its FP32 atomics, reuse those CTAs for the final BF16 cast so
  // the prefill path does not need a separate output kernel launch.
  cg::this_grid().sync();
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int nvec = tokens * (H / 4);
  for (int idx = gtid; idx < nvec; idx += stride) {
    const float4 v = reinterpret_cast<const float4*>(accum_out)[idx];
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&lo);
    packed.y = *reinterpret_cast<const uint32_t*>(&hi);
    reinterpret_cast<uint2*>(out)[idx] = packed;
  }
}

__global__ __launch_bounds__(512) void cast_output_kernel(const float* __restrict__ accum_out,
                                                          __nv_bfloat16* __restrict__ out,
                                                          int tokens) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int idx = token * (H / 4) + threadIdx.x;
  const float4 v = reinterpret_cast<const float4*>(accum_out)[idx];
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
  uint2 packed;
  packed.x = *reinterpret_cast<const uint32_t*>(&lo);
  packed.y = *reinterpret_cast<const uint32_t*>(&hi);
  reinterpret_cast<uint2*>(out)[idx] = packed;
}

}  // namespace
