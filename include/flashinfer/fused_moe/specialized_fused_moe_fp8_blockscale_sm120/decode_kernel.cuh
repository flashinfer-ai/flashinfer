// Fused block-FP8 silu-gated MoE for sm120 (RTX PRO 6000 Blackwell).
// E=256, top_k=8, H=2048, I=512.
//
// Math: fp8-e4m3 weights widen to fp16 exactly (3 mantissa bits), activations
// bf16 -> fp16 exactly, mma.sync m16n8k16 with fp32 accumulate, 128-wide block
// scale folded into the fp32 partial after each 128-k chunk.  Matches the
// fp32-dequant reference up to accumulation order.
//
// Memory: weight tiles are staged into shared memory with cp.async so every
// global load instruction reads 512 contiguous bytes of one weight row; the
// mma A-fragments are then gathered from smem.  Reading only 64 B per row
// directly into registers costs ~1/3 of DRAM bandwidth on this part.

//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/specialized_fused_moe_fp8_blockscale_sm120/decode_kernel.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define H_DIM 2048
#define I_DIM 512
#define TWO_I 1024
#define E_NUM 256
#define TOPK 8
#define NT 8

#define XSTRIDE 2056 /* halves: 4112 B pitch, conflict free LDS.128 */
#define WPITCH 576   /* bytes per staged weight row (512 + 64) */

// ---------------------------------------------------------------- helpers

__device__ __forceinline__ uint32_t cvt_e4m3x2_to_f16x2(uint32_t v16) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)v16));
  return r;
}

__device__ __forceinline__ void unpack16(const uint4& w, uint32_t h[8]) {
  h[0] = cvt_e4m3x2_to_f16x2(w.x & 0xffffu);
  h[1] = cvt_e4m3x2_to_f16x2(w.x >> 16);
  h[2] = cvt_e4m3x2_to_f16x2(w.y & 0xffffu);
  h[3] = cvt_e4m3x2_to_f16x2(w.y >> 16);
  h[4] = cvt_e4m3x2_to_f16x2(w.z & 0xffffu);
  h[5] = cvt_e4m3x2_to_f16x2(w.z >> 16);
  h[6] = cvt_e4m3x2_to_f16x2(w.w & 0xffffu);
  h[7] = cvt_e4m3x2_to_f16x2(w.w >> 16);
}

__device__ __forceinline__ void mma16816(float d[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void tile64(float acc[4], const uint4& w0, const uint4& w1,
                                       const uint4& x0, const uint4& x1) {
  uint32_t h0[8], h1[8], xq[8];
  unpack16(w0, h0);
  unpack16(w1, h1);
  xq[0] = x0.x;
  xq[1] = x0.y;
  xq[2] = x0.z;
  xq[3] = x0.w;
  xq[4] = x1.x;
  xq[5] = x1.y;
  xq[6] = x1.z;
  xq[7] = x1.w;
#pragma unroll
  for (int s = 0; s < 4; ++s)
    mma16816(acc, h0[2 * s], h1[2 * s], h0[2 * s + 1], h1[2 * s + 1], xq[2 * s], xq[2 * s + 1]);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float silu_mul(float g, float u) {
  return g * __frcp_rn(1.f + __expf(-g)) * u;
}

__device__ __forceinline__ int blk_escan(int v, int* sw, int tid) {
  const int lane = tid & 31, warp = tid >> 5;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) sw[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = (lane < 8) ? sw[lane] : 0;
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += y;
    }
    if (lane < 8) sw[lane] = w;
  }
  __syncthreads();
  return ((warp > 0) ? sw[warp - 1] : 0) + x - v;
}

// In-block deterministic tile routing for the decode path (P <= 1024).
// Fills s_pair[0..n-1] with the pair indices of tile `tileIdx`; returns false
// when the tile is empty.  All blocks derive the same assignment because the
// pair order inside an expert is the natural topk_ids order.
__device__ __forceinline__ bool route_tile(const int* __restrict__ topk_ids, int P, int tileIdx,
                                           int tid, int* cnt, int* sw, int* sh, int* s_pair) {
  cnt[tid] = 0;
  if (tid == 0) sh[0] = -1;
  __syncthreads();
  for (int p = tid; p < P; p += 256) atomicAdd(&cnt[topk_ids[p]], 1);
  __syncthreads();
  const int v = cnt[tid];
  const int nt = (v + NT - 1) / NT;
  const int toff = blk_escan(nt, sw, tid);
  if (tileIdx >= toff && tileIdx < toff + nt) {
    sh[0] = tid;
    sh[1] = tileIdx - toff;
    const int rem = v - (tileIdx - toff) * NT;
    sh[2] = rem < NT ? rem : NT;
  }
  __syncthreads();
  const int e = sh[0];
  if (e < 0) return false;
  const int lo = sh[1] * NT, n = sh[2];
  int f[4], pp[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    pp[q] = tid * 4 + q;
    f[q] = (pp[q] < P && topk_ids[pp[q]] == e) ? 1 : 0;
  }
  int base = blk_escan(f[0] + f[1] + f[2] + f[3], sw, tid) - lo;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    if (f[q]) {
      if (base >= 0 && base < n) s_pair[base] = pp[q];
      ++base;
    }
  }
  __syncthreads();
  return true;
}

__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned G) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned g = atomicAdd(&bar[1], 0u);
    if (atomicAdd(&bar[0], 1u) == G - 1u) {
      atomicExch(&bar[0], 0u);
      atomicAdd(&bar[1], 1u);
    } else {
      while (atomicAdd(&bar[1], 0u) == g) __nanosleep(64);
    }
  }
  __syncthreads();
  __threadfence();
}

// Latency path for T<=16: one resident CTA per SM performs FC1, crosses a
// generation-counted grid barrier, then performs FC2 and the top-k reduction.
// The intermediate is per-call scratch and is completely overwritten before
// it is consumed.
__device__ __forceinline__ float2 p_bf2f(uint32_t v) {
  float2 r;
  r.x = __uint_as_float(v << 16);
  r.y = __uint_as_float(v & 0xffff0000u);
  return r;
}

__device__ __forceinline__ uint4 p_ld_inter_l1_evict_last(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.L1::evict_last.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

__device__ __forceinline__ void p_cvt_fp8x4(uint32_t v, float* o) {
  uint32_t b01, b23;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(b01) : "h"(static_cast<unsigned short>(v)));
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(b23) : "h"(static_cast<unsigned short>(v >> 16)));
  const float2 f01 = p_bf2f(b01);
  const float2 f23 = p_bf2f(b23);
  o[0] = f01.x;
  o[1] = f01.y;
  o[2] = f23.x;
  o[3] = f23.y;
}

__device__ __forceinline__ float p_dot16(const uint4& w, const float* __restrict__ x) {
  const uint32_t* wu = reinterpret_cast<const uint32_t*>(&w);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_cvt_fp8x4(wu[i], v);
    a0 = fmaf(v[0], x[4 * i], a0);
    a1 = fmaf(v[1], x[4 * i + 1], a1);
    a2 = fmaf(v[2], x[4 * i + 2], a2);
    a3 = fmaf(v[3], x[4 * i + 3], a3);
  }
  return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ float2 p_dot16_pair(const uint4& wa, const uint4& wb,
                                               const float* __restrict__ x) {
  const uint32_t* au = reinterpret_cast<const uint32_t*>(&wa);
  const uint32_t* bu = reinterpret_cast<const uint32_t*>(&wb);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
  float av[4], bv[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p_cvt_fp8x4(au[i], av);
    p_cvt_fp8x4(bu[i], bv);
    a0 = fmaf(av[0], x[4 * i], a0);
    b0 = fmaf(bv[0], x[4 * i], b0);
    a1 = fmaf(av[1], x[4 * i + 1], a1);
    b1 = fmaf(bv[1], x[4 * i + 1], b1);
    a2 = fmaf(av[2], x[4 * i + 2], a2);
    b2 = fmaf(bv[2], x[4 * i + 2], b2);
    a3 = fmaf(av[3], x[4 * i + 3], a3);
    b3 = fmaf(bv[3], x[4 * i + 3], b3);
  }
  return make_float2((a0 + a1) + (a2 + a3), (b0 + b1) + (b2 + b3));
}

__device__ __forceinline__ float p_warp_sum(float v) {
#pragma unroll
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  return v;
}

// The persistent epoch counters are 64-bit: each epoch target is derived by
// integer division, and the arrival counts (gridDim.x, 11..13-CTA token
// groups) do not divide 2^32, so a 32-bit counter would misalign its epoch
// grid after wrapping and let early arrivals pass before the epoch completes.
// A 64-bit counter cannot wrap in any realistic process lifetime. Counters
// keep a 32-byte stride so no two share one L2 sector.
//
// Barrier counters and the FC1->FC2 intermediate live in the per-call
// workspace, NOT in module __device__ globals: module globals are one
// mutable allocation per device, so two persistent-decode launches running
// concurrently (two streams, or two independently captured CUDA graphs
// replayed together) would merge their barrier arrivals and overwrite each
// other's intermediates. The host passes bars/inter pointers carved from the
// call's own workspace.
//
// Workspace contract shared by all four translation units: the first
// kWorkspaceDataOffset bytes are a RESERVED counter region that the Python
// owner zeroes once when the buffer is allocated (the zeroing is captured
// when the buffer is capture-owned, so each graph replay restarts its
// epochs); every path's data begins at kWorkspaceDataOffset, so no path
// ever overwrites another path's counters when the buffer is reused across
// token counts. Counter words are at FIXED offsets, each used by exactly
// one launch variant, and every counter is epoch-based: a completed launch
// leaves it at a multiple of its own arrival count, so stream-ordered
// launches over the same buffer need no per-launch reset (this is exactly
// the zero-once lifetime the previous module globals had, made per-buffer).
// The gemv T=1 phase counters own [0..256); persistent decode owns
// [256..2048), laid out below in u64 units from the workspace base:
//   [32..48)   T=4 token counters   (4 tokens x 32-byte stride)
//   [48..112)  T=16 token counters  (16 tokens x 32-byte stride)
//   [112..176) T=16 dependency counters
//   [176]      release barrier: {u32 count, u32 generation}
// The bf16 intermediate (data) starts at kWorkspaceDataOffset.
constexpr size_t kWorkspaceDataOffset = 2048;
constexpr int kPersistentT4Off = 32;
constexpr int kPersistentT16TokenOff = 48;
constexpr int kPersistentT16DepOff = 112;
constexpr int kPersistentReleaseOff = 176;
constexpr size_t kPersistentInterBytes = (size_t)16 * TOPK * I_DIM * sizeof(__nv_bfloat16);
constexpr size_t kPersistentDecodeWsBytes = kWorkspaceDataOffset + kPersistentInterBytes;

__device__ __forceinline__ void p_t16_release_barrier(unsigned long long* bars) {
  unsigned* count = reinterpret_cast<unsigned*>(bars + kPersistentReleaseOff);
  unsigned* generation = count + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned gen = *reinterpret_cast<volatile unsigned*>(generation);
    if (atomicAdd(count, 1u) == gridDim.x - 1u) {
      atomicAdd(generation, 1u);
      atomicExch(count, 0u);
    } else {
      volatile unsigned* observed = generation;
      int spin = 0;
      while (*observed == gen) {
        if (++spin > 4) __nanosleep(192);
      }
    }
  }
  __syncthreads();
}

template <unsigned ARRIVALS>
__device__ __forceinline__ void p_t4_token_barrier(int token, unsigned long long* bars) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long* counter = bars + kPersistentT4Off + token * 4;
    unsigned long long old;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;\n"
                 : "=l"(old)
                 : "l"(counter)
                 : "memory");
    const unsigned long long target = (old / ARRIVALS + 1ull) * ARRIVALS;
    if (old + 1ull != target) {
      unsigned long long observed;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n"
                     : "=l"(observed)
                     : "l"(counter)
                     : "memory");
        if (observed < target) __nanosleep(32);
      } while (observed < target);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void p_t16_token_barrier(int token, unsigned arrivals,
                                                    unsigned long long* bars) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long* counter = bars + kPersistentT16TokenOff + token * 4;
    unsigned long long old;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;\n"
                 : "=l"(old)
                 : "l"(counter)
                 : "memory");
    const unsigned long long target = (old / arrivals + 1ull) * arrivals;
    if (old + 1ull != target) {
      unsigned long long observed;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n"
                     : "=l"(observed)
                     : "l"(counter)
                     : "memory");
        if (observed < target) __nanosleep(32);
      } while (observed < target);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void p_t16_dependency_barrier(int token0, int token1,
                                                         unsigned long long* bars) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long targets[2], old[2];
    const int nt = token1 != token0 ? 2 : 1;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (i < nt) {
        const int token = i == 0 ? token0 : token1;
        const unsigned long long arrivals = (token == 0 || (token & 3) == 3) ? 12ull : 13ull;
        unsigned long long* counter = bars + kPersistentT16DepOff + token * 4;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;\n"
                     : "=l"(old[i])
                     : "l"(counter)
                     : "memory");
        targets[i] = (old[i] / arrivals + 1ull) * arrivals;
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (i < nt && old[i] + 1ull != targets[i]) {
        const int token = i == 0 ? token0 : token1;
        const unsigned long long* counter = bars + kPersistentT16DepOff + token * 4;
        unsigned long long observed;
        do {
          asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n"
                       : "=l"(observed)
                       : "l"(counter)
                       : "memory");
          if (observed < targets[i]) __nanosleep(32);
        } while (observed < targets[i]);
      }
    }
  }
  __syncthreads();
}

template <int NWARP, int TOKEN_LOCAL = 0, bool T4_PAIR_CLUSTER = false>
__global__ __launch_bounds__(NWARP * 32, NWARP == 13 ? 2 : 1) void k_persistent_decode(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13s, const uint8_t* __restrict__ w2, const float* __restrict__ w2s,
    const int* __restrict__ ids, const float* __restrict__ tws, __nv_bfloat16* __restrict__ out,
    int T, int q1, int r1, int q2, int r2, unsigned long long* bars,
    __nv_bfloat16* __restrict__ p_inter) {
  __shared__ unsigned char s_t4_route_perm[T4_PAIR_CLUSTER ? TOPK : 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int P = T * TOPK;
  const int lg = lane >> 3;
  const int bx = blockIdx.x;
  // T4 cluster rank pairs tokens (0,3) and (1,2).  Nibble index is
  // pair*2+rank, so consecutive blockIdx values form one cross-token cluster.
  constexpr unsigned T4_PAIR_TOKENS = 0x2130u;
  const int t4_cluster = T4_PAIR_CLUSTER ? (bx >> 1) : 0;
  const int t4_pair = T4_PAIR_CLUSTER ? t4_cluster / 94 : 0;
  const int t4_rank = T4_PAIR_CLUSTER ? (bx & 1) : 0;
  const int token_slot = TOKEN_LOCAL == 2 ? (bx < 144 ? bx / 12 : 12 + (bx - 144) / 11) : 0;
  const int token_group =
      TOKEN_LOCAL == 1
          ? (T4_PAIR_CLUSTER ? int((T4_PAIR_TOKENS >> (4 * (2 * t4_pair + t4_rank))) & 15u)
                             : (bx >= 282   ? 3
                                : bx >= 188 ? 2
                                : bx >= 94  ? 1
                                            : 0))
          : (TOKEN_LOCAL == 2
                 ? (token_slot < 5
                        ? token_slot
                        : (token_slot < 8
                               ? token_slot + 1
                               : (token_slot < 10
                                      ? token_slot + 2
                                      : (token_slot < 12
                                             ? token_slot + 4
                                             : (token_slot == 12
                                                    ? 5
                                                    : (token_slot == 13 ? 9 : token_slot - 2))))))
                 : 0);
  const int token_blocks =
      TOKEN_LOCAL == 1 ? 94 : (TOKEN_LOCAL == 2 ? (token_slot < 12 ? 12 : 11) : gridDim.x);
  const int local_block =
      TOKEN_LOCAL == 1
          ? (T4_PAIR_CLUSTER ? t4_cluster - t4_pair * 94 : bx - token_group * 94)
          : (TOKEN_LOCAL == 2
                 ? (token_slot < 12 ? bx - token_slot * 12 : bx - 144 - (token_slot - 12) * 11)
                 : bx);
  const int fc1_base = (TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2) ? token_group * 2048 : 0;

  if constexpr (T4_PAIR_CLUSTER) {
    if (threadIdx.x == 0) {
      if (t4_rank == 0) {
#pragma unroll
        for (int r = 0; r < TOPK; ++r) s_t4_route_perm[r] = r;
      } else {
        const int token_a = int((T4_PAIR_TOKENS >> (8 * t4_pair)) & 15u);
        unsigned used = 0;
#pragma unroll
        for (int r = 0; r < TOPK; ++r) s_t4_route_perm[r] = 0xffu;
        // Reserve every exact expert match before unmatched routes can consume
        // their positions; then fill the remaining ranks into a bijection.
#pragma unroll
        for (int r = 0; r < TOPK; ++r) {
          const int wanted = ids[token_a * TOPK + r];
#pragma unroll
          for (int j = 0; j < TOPK; ++j) {
            if (s_t4_route_perm[r] == 0xffu && !(used & (1u << j)) &&
                ids[token_group * TOPK + j] == wanted) {
              s_t4_route_perm[r] = static_cast<unsigned char>(j);
              used |= 1u << j;
            }
          }
        }
#pragma unroll
        for (int r = 0; r < TOPK; ++r) {
#pragma unroll
          for (int j = 0; j < TOPK; ++j) {
            if (s_t4_route_perm[r] == 0xffu && !(used & (1u << j))) {
              s_t4_route_perm[r] = static_cast<unsigned char>(j);
              used |= 1u << j;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  const int tq1 = TOKEN_LOCAL == 2 ? 2048 / token_blocks : 21;
  const int tr1 = TOKEN_LOCAL == 2 ? 2048 % token_blocks : 74;
  const int u10 = (TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2)
                      ? fc1_base + tq1 * local_block + min(local_block, tr1)
                      : q1 * local_block + min(local_block, r1);
  const int u11 = (TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2) ? u10 + tq1 + (local_block < tr1)
                                                         : u10 + q1 + (local_block < r1);
  for (int u = u10 + warp; u < u11; u += NWARP) {
    const int logical_p = u >> 8;
    const int n = (u & (I_DIM / 2 - 1)) * 2;
    int p = logical_p;
    if constexpr (T4_PAIR_CLUSTER) p = token_group * TOPK + s_t4_route_perm[logical_p & (TOPK - 1)];
    const int rp = ids[p];
    const int t = p >> 3, e = rp & 255;
    const __nv_bfloat16* xp = x + (size_t)t * H_DIM + lane * 16;
    const uint8_t* gp = w13 + ((size_t)e * TWO_I + n) * H_DIM + lane * 16;
    const uint8_t* up = gp + (size_t)I_DIM * H_DIM;
    const float* sg = w13s + ((size_t)e * 8 + (n >> 7)) * 16;
    const float* su = sg + 4 * 16;
    float ag0 = 0.f, ag1 = 0.f, au0 = 0.f, au1 = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float xr[16];
      const uint4* src = reinterpret_cast<const uint4*>(xp + c * 512);
      uint4 xa = src[0], xb = src[1];
      const uint32_t* av = reinterpret_cast<const uint32_t*>(&xa);
      const uint32_t* bv = reinterpret_cast<const uint32_t*>(&xb);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = p_bf2f(av[j]), g = p_bf2f(bv[j]);
        xr[2 * j] = f.x;
        xr[2 * j + 1] = f.y;
        xr[8 + 2 * j] = g.x;
        xr[8 + 2 * j + 1] = g.y;
      }
      const uint4 wg0 = *reinterpret_cast<const uint4*>(gp + c * 512);
      const uint4 wg1 = *reinterpret_cast<const uint4*>(gp + H_DIM + c * 512);
      const uint4 wu0 = *reinterpret_cast<const uint4*>(up + c * 512);
      const uint4 wu1 = *reinterpret_cast<const uint4*>(up + H_DIM + c * 512);
      const float scg = sg[c * 4 + lg], scu = su[c * 4 + lg];
      if constexpr (TOKEN_LOCAL == 2) {
        const float2 dg = p_dot16_pair(wg0, wg1, xr);
        const float2 du = p_dot16_pair(wu0, wu1, xr);
        ag0 = fmaf(dg.x, scg, ag0);
        ag1 = fmaf(dg.y, scg, ag1);
        au0 = fmaf(du.x, scu, au0);
        au1 = fmaf(du.y, scu, au1);
      } else {
        ag0 = fmaf(p_dot16(wg0, xr), scg, ag0);
        ag1 = fmaf(p_dot16(wg1, xr), scg, ag1);
        au0 = fmaf(p_dot16(wu0, xr), scu, au0);
        au1 = fmaf(p_dot16(wu1, xr), scu, au1);
      }
    }
    ag0 = p_warp_sum(ag0);
    ag1 = p_warp_sum(ag1);
    au0 = p_warp_sum(au0);
    au1 = p_warp_sum(au1);
    if (lane == 0) {
      const float okf = (unsigned)rp < (unsigned)E_NUM ? 1.f : 0.f;
      const float z0 = (ag0 / (1.f + expf(-ag0))) * au0 * okf;
      const float z1 = (ag1 / (1.f + expf(-ag1))) * au1 * okf;
      *reinterpret_cast<__nv_bfloat162*>(p_inter + (size_t)p * I_DIM + n) =
          __floats2bfloat162_rn(z0, z1);
    }
  }

  if constexpr (TOKEN_LOCAL == 1)
    p_t4_token_barrier<94>(token_group, bars);
  else if constexpr (TOKEN_LOCAL == 2)
    p_t16_token_barrier(token_group, token_blocks, bars);
  else if constexpr (TOKEN_LOCAL == 3)
    p_t16_dependency_barrier(u10 >> 11, (u11 - 1) >> 11, bars);
  else
    p_t16_release_barrier(bars);

  constexpr int FC2_UNITS = TOKEN_LOCAL == 1 ? H_DIM / 4 : H_DIM / 2;
  const int fc2_base = (TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2) ? token_group * FC2_UNITS : 0;
  const int tq2 =
      TOKEN_LOCAL == 1 ? FC2_UNITS / token_blocks : (TOKEN_LOCAL == 2 ? 1024 / token_blocks : 10);
  const int tr2 =
      TOKEN_LOCAL == 1 ? FC2_UNITS % token_blocks : (TOKEN_LOCAL == 2 ? 1024 % token_blocks : 84);
  const int u20 = TOKEN_LOCAL == 3 ? (u10 >> 1)
                                   : ((TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2)
                                          ? fc2_base + tq2 * local_block + min(local_block, tr2)
                                          : q2 * local_block + min(local_block, r2));
  const int u21 = TOKEN_LOCAL == 3
                      ? (u11 >> 1)
                      : ((TOKEN_LOCAL == 1 || TOKEN_LOCAL == 2) ? u20 + tq2 + (local_block < tr2)
                                                                : u20 + q2 + (local_block < r2));
  for (int u = u20 + warp; u < u21; u += NWARP) {
    const int t = u / FC2_UNITS;
    const int h = (u - t * FC2_UNITS) * (TOKEN_LOCAL == 1 ? 4 : 2);
    const int* idp = ids + t * TOPK;
    const float* twp = tws + t * TOPK;
    if constexpr (TOKEN_LOCAL == 1) {
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const int e = idp[k] & 255;
        const __nv_bfloat16* ip = p_inter + (size_t)(t * TOPK + k) * I_DIM + lane * 16;
        float ir[16];
        const uint4 ia = p_ld_inter_l1_evict_last(ip);
        const uint4 ib = p_ld_inter_l1_evict_last(ip + 8);
        const uint32_t* av = reinterpret_cast<const uint32_t*>(&ia);
        const uint32_t* bv = reinterpret_cast<const uint32_t*>(&ib);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2 f = p_bf2f(av[j]), g = p_bf2f(bv[j]);
          ir[2 * j] = f.x;
          ir[2 * j + 1] = f.y;
          ir[8 + 2 * j] = g.x;
          ir[8 + 2 * j + 1] = g.y;
        }
        const uint8_t* wp = w2 + ((size_t)e * H_DIM + h) * I_DIM + lane * 16;
        const float sc = w2s[((size_t)e * 16 + (h >> 7)) * 4 + lg] * twp[k];
        const uint4 w0 = *reinterpret_cast<const uint4*>(wp);
        const uint4 w1 = *reinterpret_cast<const uint4*>(wp + I_DIM);
        const uint4 w2v = *reinterpret_cast<const uint4*>(wp + 2 * I_DIM);
        const uint4 w3 = *reinterpret_cast<const uint4*>(wp + 3 * I_DIM);
        const float2 d01 = p_dot16_pair(w0, w1, ir);
        const float2 d23 = p_dot16_pair(w2v, w3, ir);
        acc0 = fmaf(d01.x, sc, acc0);
        acc1 = fmaf(d01.y, sc, acc1);
        acc2 = fmaf(d23.x, sc, acc2);
        acc3 = fmaf(d23.y, sc, acc3);
      }
      acc0 = p_warp_sum(acc0);
      acc1 = p_warp_sum(acc1);
      acc2 = p_warp_sum(acc2);
      acc3 = p_warp_sum(acc3);
      if (lane == 0) {
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * H_DIM + h) =
            __floats2bfloat162_rn(acc0, acc1);
        *reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * H_DIM + h + 2) =
            __floats2bfloat162_rn(acc2, acc3);
      }
      continue;
    }
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int e = idp[k] & 255;
      const __nv_bfloat16* ip = p_inter + (size_t)(t * TOPK + k) * I_DIM + lane * 16;
      float ir[16];
      const uint4 ia = reinterpret_cast<const uint4*>(ip)[0];
      const uint4 ib = reinterpret_cast<const uint4*>(ip)[1];
      const uint32_t* av = reinterpret_cast<const uint32_t*>(&ia);
      const uint32_t* bv = reinterpret_cast<const uint32_t*>(&ib);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = p_bf2f(av[j]), g = p_bf2f(bv[j]);
        ir[2 * j] = f.x;
        ir[2 * j + 1] = f.y;
        ir[8 + 2 * j] = g.x;
        ir[8 + 2 * j + 1] = g.y;
      }
      const uint8_t* wp = w2 + ((size_t)e * H_DIM + h) * I_DIM + lane * 16;
      if constexpr (TOKEN_LOCAL == 1) {
        // At T=4, issue the tiny independent scale before the row payload so
        // its latency overlaps the two 16-byte weight loads.
        const float sc = w2s[((size_t)e * 16 + (h >> 7)) * 4 + lg] * twp[k];
        const uint4 w0 = *reinterpret_cast<const uint4*>(wp);
        const uint4 w1 = *reinterpret_cast<const uint4*>(wp + I_DIM);
        acc0 = fmaf(p_dot16(w0, ir), sc, acc0);
        acc1 = fmaf(p_dot16(w1, ir), sc, acc1);
      } else {
        const uint4 w0 = *reinterpret_cast<const uint4*>(wp);
        const uint4 w1 = *reinterpret_cast<const uint4*>(wp + I_DIM);
        const float sc = w2s[((size_t)e * 16 + (h >> 7)) * 4 + lg] * twp[k];
        acc0 = fmaf(p_dot16(w0, ir), sc, acc0);
        acc1 = fmaf(p_dot16(w1, ir), sc, acc1);
      }
    }
    acc0 = p_warp_sum(acc0);
    acc1 = p_warp_sum(acc1);
    if (lane == 0) {
      *reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * H_DIM + h) =
          __floats2bfloat162_rn(acc0, acc1);
    }
  }
}

// ---------------------------------------------------------------- routing

__global__ __launch_bounds__(256) void k_route(const int* __restrict__ topk_ids, int P,
                                               int* __restrict__ sorted, int* __restrict__ tile_e,
                                               int* __restrict__ tile_base,
                                               int* __restrict__ tile_n, int max_tiles, int ntk) {
  __shared__ int cnt[256], off[256], cur[256], sw[8];
  const int tid = threadIdx.x;
  for (int i = tid; i < max_tiles; i += 256) tile_n[i] = 0;
  cnt[tid] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += 256) atomicAdd(&cnt[topk_ids[p]], 1);
  __syncthreads();
  const int v = cnt[tid];
  const int o = blk_escan(v, sw, tid);
  off[tid] = o;
  cur[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += 256) {
    const int e = topk_ids[p];
    sorted[atomicAdd(&cur[e], 1)] = p;
  }
  const int nt = (v + ntk - 1) / ntk;
  const int to = blk_escan(nt, sw, tid);
  for (int j = 0; j < nt; ++j) {
    const int idx = to + j;
    if (idx < max_tiles) {
      tile_e[idx] = tid;
      tile_base[idx] = o + j * ntk;
      const int rem = v - j * ntk;
      tile_n[idx] = rem < ntk ? rem : ntk;
    }
  }
}

// ---------------------------------------------------------------- FC1

// grid (64 row tiles of 16, n_tiles); 256 threads = 8 warps, warp = k slice.
__global__ __launch_bounds__(256) void k_gemm1(
    const __nv_bfloat16* __restrict__ hs, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13_scale, const int* __restrict__ topk_ids, int P,
    __half* __restrict__ gu, __nv_bfloat16* __restrict__ out, int nout4, int* __restrict__ tinfo) {
  const int tid = threadIdx.x;
  // clear the destination (FC2 accumulates into it atomically)
  {
    const int flat = blockIdx.y * gridDim.x + blockIdx.x;
    const int stride = gridDim.x * gridDim.y * 256;
    uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = flat * 256 + tid; i < nout4; i += stride) ((uint4*)out)[i] = z;
  }
  extern __shared__ char smem[];
  uint8_t* wb = (uint8_t*)smem;  // [2][16][WPITCH]
  __half* xs = (__half*)(smem + 2 * 16 * WPITCH);
  __shared__ float red[8 * 128];
  __shared__ int s_pair[NT];
  __shared__ int s_cnt[256], s_sw[8], s_sh[4];
  const bool has = route_tile(topk_ids, P, blockIdx.y, tid, s_cnt, s_sw, s_sh, s_pair);
  if (blockIdx.x == 0) {
    if (tid == 0) tinfo[blockIdx.y * 16] = has ? (s_sh[0] | (s_sh[2] << 16)) : 0;
    if (has && tid < s_sh[2]) tinfo[blockIdx.y * 16 + 1 + tid] = s_pair[tid];
  }
  if (!has) return;
  const int n = s_sh[2];
  const int e = s_sh[0];
  const int r0 = blockIdx.x * 16;

  const uint8_t* wptr = w13 + ((size_t)e * TWO_I + r0) * H_DIM;
  // chunk 0
  {
    const int row = tid >> 5, off = (tid & 31) * 16;
    cp_async16(wb + row * WPITCH + off, wptr + (size_t)row * H_DIM + off);
    cp_async16(wb + (row + 8) * WPITCH + off, wptr + (size_t)(row + 8) * H_DIM + off);
    cp_commit();
  }

  for (int rr = 0; rr < n; ++rr) {
    const uint4 b = ((const uint4*)(hs + (size_t)(s_pair[rr] >> 3) * H_DIM))[tid];
    __half2 c0 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.x));
    __half2 c1 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.y));
    __half2 c2 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.z));
    __half2 c3 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.w));
    uint4 o;
    o.x = *(const uint32_t*)&c0;
    o.y = *(const uint32_t*)&c1;
    o.z = *(const uint32_t*)&c2;
    o.w = *(const uint32_t*)&c3;
    *(uint4*)(xs + (size_t)rr * XSTRIDE + tid * 8) = o;
  }

  const int warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, c = lane & 3;
  const float* wsc = w13_scale + ((size_t)e * 8 + (r0 >> 7)) * 16;
  const __half* xptr = xs + (size_t)((r < n) ? r : 0) * XSTRIDE + warp * 64 + c * 16;
  const int wo = warp * 64 + c * 16;

  float facc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ch = 0; ch < 4; ++ch) {
    if (ch < 3) {
      const int row = tid >> 5, off = (tid & 31) * 16;
      const int k = (ch + 1) * 512;
      cp_async16(wb + (ch + 1) % 2 * (16 * WPITCH) + row * WPITCH + off,
                 wptr + (size_t)row * H_DIM + k + off);
      cp_async16(wb + (ch + 1) % 2 * (16 * WPITCH) + (row + 8) * WPITCH + off,
                 wptr + (size_t)(row + 8) * H_DIM + k + off);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    const uint8_t* wc = wb + (ch % 2) * (16 * WPITCH) + wo;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    tile64(acc, *(const uint4*)(wc + r * WPITCH), *(const uint4*)(wc + (r + 8) * WPITCH),
           *(const uint4*)(xptr + ch * 512), *(const uint4*)(xptr + ch * 512 + 8));
    const float sc = wsc[ch * 4 + (warp >> 1)];
#pragma unroll
    for (int i = 0; i < 4; ++i) facc[i] += acc[i] * sc;
    __syncthreads();
  }

  float* rb = red + warp * 128;
  rb[r * 8 + c * 2 + 0] = facc[0];
  rb[r * 8 + c * 2 + 1] = facc[1];
  rb[(r + 8) * 8 + c * 2 + 0] = facc[2];
  rb[(r + 8) * 8 + c * 2 + 1] = facc[3];
  __syncthreads();
  if (tid < 128) {
    const int tok = tid >> 4, row = tid & 15;
    if (tok < n) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) s += red[w * 128 + row * 8 + tok];
      gu[(size_t)s_pair[tok] * TWO_I + r0 + row] = __float2half_rn(s);
    }
  }
}

// ---------------------------------------------------------------- FC2

// grid (64 h tiles of 32, n_tiles); 256 threads = 8 warps = 2 row tiles x 4 k.
__global__ __launch_bounds__(256) void k_gemm2(const __half* __restrict__ gu,
                                               const uint8_t* __restrict__ w2,
                                               const float* __restrict__ w2_scale,
                                               const float* __restrict__ topk_weights,
                                               const int* __restrict__ tinfo,
                                               __nv_bfloat16* __restrict__ out) {
  const int tid = threadIdx.x;
  __shared__ uint8_t wb[32 * WPITCH];
  __shared__ __half si[NT * 520];
  __shared__ float red[8 * 128];
  __shared__ int s_inf[16];
  if (tid < 16) s_inf[tid] = tinfo[blockIdx.y * 16 + tid];
  __syncthreads();
  const int n = s_inf[0] >> 16;
  if (n == 0) return;
  const int e = s_inf[0] & 0xffff;
  const int h0 = blockIdx.x * 32;

  const uint8_t* wptr = w2 + ((size_t)e * H_DIM + h0) * I_DIM;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int g = q * 4096 + tid * 16;
    const int row = g >> 9, off = g & 511;
    cp_async16(wb + row * WPITCH + off, wptr + (size_t)row * I_DIM + off);
  }
  cp_commit();

  for (int i = tid; i < n * 64; i += 256) {
    const int j = i >> 6, o = (i & 63) * 8;
    const __half* gp = gu + (size_t)s_inf[1 + j] * TWO_I;
    uint4 ga = *(const uint4*)(gp + o);
    uint4 ub = *(const uint4*)(gp + I_DIM + o);
    const __half2* ha = (const __half2*)&ga;
    const __half2* hb = (const __half2*)&ub;
    uint4 ov;
    uint32_t* op = (uint32_t*)&ov;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float2 g = __half22float2(ha[q]);
      float2 u = __half22float2(hb[q]);
      __half2 hv = __float22half2_rn(make_float2(silu_mul(g.x, u.x), silu_mul(g.y, u.y)));
      op[q] = *(const uint32_t*)&hv;
    }
    *(uint4*)(si + j * 520 + o) = ov;
  }
  cp_wait<0>();
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int rt = warp >> 2, kc = warp & 3;
  const int r = lane >> 2, c = lane & 3;
  const float sc = w2_scale[((size_t)e * 16 + ((h0 + rt * 16) >> 7)) * 4 + kc];
  const uint8_t* wbase = wb + rt * (16 * WPITCH) + c * 16;
  const __half* xptr = si + (size_t)((r < n) ? r : 0) * 520 + kc * 128 + c * 16;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int k = it * 64;
    tile64(acc, *(const uint4*)(wbase + r * WPITCH + kc * 128 + k),
           *(const uint4*)(wbase + (r + 8) * WPITCH + kc * 128 + k), *(const uint4*)(xptr + k),
           *(const uint4*)(xptr + k + 8));
  }

  float* rb = red + warp * 128;
  rb[r * 8 + c * 2 + 0] = acc[0] * sc;
  rb[r * 8 + c * 2 + 1] = acc[1] * sc;
  rb[(r + 8) * 8 + c * 2 + 0] = acc[2] * sc;
  rb[(r + 8) * 8 + c * 2 + 1] = acc[3] * sc;
  __syncthreads();

  if (tid < n * 16) {
    const int tok = tid >> 4, q = tid & 15;
    const int rtt = q >> 3, rr0 = (q & 7) * 2;
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      a += red[(rtt * 4 + kk) * 128 + rr0 * 8 + tok];
      b += red[(rtt * 4 + kk) * 128 + (rr0 + 1) * 8 + tok];
    }
    const int p = s_inf[1 + tok];
    const float tw = topk_weights[p];
    __nv_bfloat162 v = __floats2bfloat162_rn(a * tw, b * tw);
    atomicAdd((__nv_bfloat162*)(out + (size_t)(p >> 3) * H_DIM + h0 + rtt * 16 + rr0), v);
  }
}

// ------------------------------------------------- fused decode kernel
// grid (64, n_tiles) -- blockIdx.x indexes a 16-row FC1 tile in phase 1 and a
// 32-row FC2 tile in phase 2, so one routing pass serves both phases and the
// two GEMMs cost a single launch and a single ramp.
__global__ __launch_bounds__(256) void k_fused(
    const __nv_bfloat16* __restrict__ hs, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13_scale, const uint8_t* __restrict__ w2,
    const float* __restrict__ w2_scale, const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights, int P, __half* __restrict__ gu,
    __nv_bfloat16* __restrict__ out, int nout4, unsigned* __restrict__ bar, int xrows) {
  const int tid = threadIdx.x;
  {
    const int flat = blockIdx.y * gridDim.x + blockIdx.x;
    const int stride = gridDim.x * gridDim.y * 256;
    uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = flat * 256 + tid; i < nout4; i += stride) ((uint4*)out)[i] = z;
  }
  extern __shared__ char smem[];
  uint8_t* wb = (uint8_t*)smem;
  __half* xs = (__half*)(smem + 2 * 16 * WPITCH);
  __half* si = (__half*)(smem + 2 * 16 * WPITCH);
  __shared__ float red[8 * 128];
  __shared__ int s_pair[NT];
  __shared__ int s_cnt[256], s_sw[8], s_sh[4];

  const bool has = route_tile(topk_ids, P, blockIdx.y, tid, s_cnt, s_sw, s_sh, s_pair);
  const int n = has ? s_sh[2] : 0;
  const int e = has ? s_sh[0] : 0;
  const int warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, c = lane & 3;

  if (has) {
    const int r0 = blockIdx.x * 16;
    const uint8_t* wptr = w13 + ((size_t)e * TWO_I + r0) * H_DIM;
    {
      const int row = tid >> 5, off = (tid & 31) * 16;
      cp_async16(wb + row * WPITCH + off, wptr + (size_t)row * H_DIM + off);
      cp_async16(wb + (row + 8) * WPITCH + off, wptr + (size_t)(row + 8) * H_DIM + off);
      cp_commit();
    }
    for (int rr = 0; rr < n; ++rr) {
      const uint4 b = ((const uint4*)(hs + (size_t)(s_pair[rr] >> 3) * H_DIM))[tid];
      __half2 c0 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.x));
      __half2 c1 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.y));
      __half2 c2 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.z));
      __half2 c3 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.w));
      uint4 o;
      o.x = *(const uint32_t*)&c0;
      o.y = *(const uint32_t*)&c1;
      o.z = *(const uint32_t*)&c2;
      o.w = *(const uint32_t*)&c3;
      *(uint4*)(xs + (size_t)rr * XSTRIDE + tid * 8) = o;
    }
    const float* wsc = w13_scale + ((size_t)e * 8 + (r0 >> 7)) * 16;
    const __half* xptr = xs + (size_t)((r < n) ? r : 0) * XSTRIDE + warp * 64 + c * 16;
    const int wo = warp * 64 + c * 16;
    float facc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      if (ch < 3) {
        const int row = tid >> 5, off = (tid & 31) * 16;
        const int k = (ch + 1) * 512;
        cp_async16(wb + (ch + 1) % 2 * (16 * WPITCH) + row * WPITCH + off,
                   wptr + (size_t)row * H_DIM + k + off);
        cp_async16(wb + (ch + 1) % 2 * (16 * WPITCH) + (row + 8) * WPITCH + off,
                   wptr + (size_t)(row + 8) * H_DIM + k + off);
        cp_commit();
        cp_wait<1>();
      } else {
        cp_wait<0>();
      }
      __syncthreads();
      const uint8_t* wc = wb + (ch % 2) * (16 * WPITCH) + wo;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      tile64(acc, *(const uint4*)(wc + r * WPITCH), *(const uint4*)(wc + (r + 8) * WPITCH),
             *(const uint4*)(xptr + ch * 512), *(const uint4*)(xptr + ch * 512 + 8));
      const float sc = wsc[ch * 4 + (warp >> 1)];
#pragma unroll
      for (int i = 0; i < 4; ++i) facc[i] += acc[i] * sc;
      __syncthreads();
    }
    float* rb = red + warp * 128;
    rb[r * 8 + c * 2 + 0] = facc[0];
    rb[r * 8 + c * 2 + 1] = facc[1];
    rb[(r + 8) * 8 + c * 2 + 0] = facc[2];
    rb[(r + 8) * 8 + c * 2 + 1] = facc[3];
    __syncthreads();
    if (tid < 128) {
      const int tok = tid >> 4, row = tid & 15;
      if (tok < n) {
        float sacc = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) sacc += red[w * 128 + row * 8 + tok];
        gu[(size_t)s_pair[tok] * TWO_I + r0 + row] = __float2half_rn(sacc);
      }
    }
  }

  grid_barrier(bar, gridDim.x * gridDim.y);
  if (!has) return;

  // ---- phase 2: 32 output rows of w2[e]
  const int h0 = blockIdx.x * 32;
  const uint8_t* w2p = w2 + ((size_t)e * H_DIM + h0) * I_DIM;
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int g = q * 4096 + tid * 16;
    const int row = g >> 9, off = g & 511;
    cp_async16(wb + row * WPITCH + off, w2p + (size_t)row * I_DIM + off);
  }
  cp_commit();
  for (int i = tid; i < n * 64; i += 256) {
    const int j = i >> 6, o = (i & 63) * 8;
    const __half* gp = gu + (size_t)s_pair[j] * TWO_I;
    uint4 ga = *(const uint4*)(gp + o);
    uint4 ub = *(const uint4*)(gp + I_DIM + o);
    const __half2* ha = (const __half2*)&ga;
    const __half2* hb = (const __half2*)&ub;
    uint4 ov;
    uint32_t* op = (uint32_t*)&ov;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float2 g = __half22float2(ha[q]);
      float2 u = __half22float2(hb[q]);
      __half2 hv = __float22half2_rn(make_float2(silu_mul(g.x, u.x), silu_mul(g.y, u.y)));
      op[q] = *(const uint32_t*)&hv;
    }
    *(uint4*)(si + j * 520 + o) = ov;
  }
  cp_wait<0>();
  __syncthreads();

  const int rt = warp >> 2, kc = warp & 3;
  const float sc2 = w2_scale[((size_t)e * 16 + ((h0 + rt * 16) >> 7)) * 4 + kc];
  const uint8_t* wbase = wb + rt * (16 * WPITCH) + c * 16;
  const __half* xp2 = si + (size_t)((r < n) ? r : 0) * 520 + kc * 128 + c * 16;
  float acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int k = it * 64;
    tile64(acc2, *(const uint4*)(wbase + r * WPITCH + kc * 128 + k),
           *(const uint4*)(wbase + (r + 8) * WPITCH + kc * 128 + k), *(const uint4*)(xp2 + k),
           *(const uint4*)(xp2 + k + 8));
  }
  float* rb2 = red + warp * 128;
  rb2[r * 8 + c * 2 + 0] = acc2[0] * sc2;
  rb2[r * 8 + c * 2 + 1] = acc2[1] * sc2;
  rb2[(r + 8) * 8 + c * 2 + 0] = acc2[2] * sc2;
  rb2[(r + 8) * 8 + c * 2 + 1] = acc2[3] * sc2;
  __syncthreads();
  if (tid < n * 16) {
    const int tok = tid >> 4, q = tid & 15;
    const int rtt = q >> 3, rr0 = (q & 7) * 2;
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      a += red[(rtt * 4 + kk) * 128 + rr0 * 8 + tok];
      b += red[(rtt * 4 + kk) * 128 + (rr0 + 1) * 8 + tok];
    }
    const int p = s_pair[tok];
    const float tw = topk_weights[p];
    __nv_bfloat162 v = __floats2bfloat162_rn(a * tw, b * tw);
    atomicAdd((__nv_bfloat162*)(out + (size_t)(p >> 3) * H_DIM + h0 + rtt * 16 + rr0), v);
  }
}

// ---------------------------------------------------------------- big-T path

__global__ __launch_bounds__(256) void k_cvt_x(const __nv_bfloat16* __restrict__ hs,
                                               __half* __restrict__ hsf,
                                               __nv_bfloat16* __restrict__ out, int nvec) {
  const int stride = gridDim.x * 256;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < nvec; i += stride) {
    const uint4 b = ((const uint4*)hs)[i];
    __half2 c0 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.x));
    __half2 c1 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.y));
    __half2 c2 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.z));
    __half2 c3 = __float22half2_rn(__bfloat1622float2(*(const __nv_bfloat162*)&b.w));
    uint4 o;
    o.x = *(const uint32_t*)&c0;
    o.y = *(const uint32_t*)&c1;
    o.z = *(const uint32_t*)&c2;
    o.w = *(const uint32_t*)&c3;
    ((uint4*)hsf)[i] = o;
    ((uint4*)out)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
}

__global__ __launch_bounds__(256) void k_silu(const __half* __restrict__ gu,
                                              __half* __restrict__ inter, int nvec) {
  const int stride = gridDim.x * 256;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < nvec; i += stride) {
    const int p = i >> 6, o = (i & 63) * 8;
    const __half* gp = gu + (size_t)p * TWO_I;
    uint4 ga = *(const uint4*)(gp + o);
    uint4 ub = *(const uint4*)(gp + I_DIM + o);
    const __half2* ha = (const __half2*)&ga;
    const __half2* hb = (const __half2*)&ub;
    uint4 ov;
    uint32_t* op = (uint32_t*)&ov;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float2 g = __half22float2(ha[q]);
      float2 u = __half22float2(hb[q]);
      __half2 hv = __float22half2_rn(make_float2(silu_mul(g.x, u.x), silu_mul(g.y, u.y)));
      op[q] = *(const uint32_t*)&hv;
    }
    *(uint4*)(inter + (size_t)p * I_DIM + o) = ov;
  }
}

#define NTILES 8
#define NTOK (8 * NTILES)

// FC1 for large T: block = 128 weight rows x NTOK tokens, warp = 16 rows.
__global__ __launch_bounds__(256) void k_gemm1_big(
    const __half* __restrict__ hsf, const uint8_t* __restrict__ w13,
    const float* __restrict__ w13_scale, const int* __restrict__ sorted,
    const int* __restrict__ tile_e, const int* __restrict__ tile_base,
    const int* __restrict__ tile_n, __half* __restrict__ gu) {
  const int tile = blockIdx.y;
  const int n = tile_n[tile];
  if (n == 0) return;
  const int e = tile_e[tile];
  const int rb = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, c = lane & 3;

  __shared__ int s_tok[NTOK];
  __shared__ float xf[128 * NTOK];
  if (tid < NTOK) s_tok[tid] = (tid < n) ? (sorted[tile_base[tile] + tid] >> 3) : 0;
  __syncthreads();

  const int rowbase = rb * 128 + warp * 16;
  const uint8_t* wptr = w13 + ((size_t)e * TWO_I + rowbase) * H_DIM + c * 16;
  const float* wsc = w13_scale + ((size_t)e * 8 + (rowbase >> 7)) * 16;
  int xoff[NTILES];
#pragma unroll
  for (int t = 0; t < NTILES; ++t) xoff[t] = s_tok[8 * t + r] * H_DIM + c * 16;

  float facc[NTILES][4];
#pragma unroll
  for (int t = 0; t < NTILES; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) facc[t][i] = 0.f;

  const int ntl = (n + 7) >> 3;
#pragma unroll 1
  for (int ks = 0; ks < 32; ++ks) {
    const int k = ks * 64;
    const __half2 sc2 = __float2half2_rn(wsc[ks >> 1]);
    uint4 w0 = *(const uint4*)(wptr + (size_t)r * H_DIM + k);
    uint4 w1 = *(const uint4*)(wptr + (size_t)(r + 8) * H_DIM + k);
    uint32_t h0[8], h1[8];
    unpack16(w0, h0);
    unpack16(w1, h1);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      __half2 a = __hmul2(*(const __half2*)&h0[q], sc2);
      __half2 b = __hmul2(*(const __half2*)&h1[q], sc2);
      h0[q] = *(const uint32_t*)&a;
      h1[q] = *(const uint32_t*)&b;
    }
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      if (t >= ntl) continue;
      uint4 x0 = *(const uint4*)(hsf + xoff[t] + k);
      uint4 x1 = *(const uint4*)(hsf + xoff[t] + k + 8);
      uint32_t xq[8];
      xq[0] = x0.x;
      xq[1] = x0.y;
      xq[2] = x0.z;
      xq[3] = x0.w;
      xq[4] = x1.x;
      xq[5] = x1.y;
      xq[6] = x1.z;
      xq[7] = x1.w;
#pragma unroll
      for (int sx = 0; sx < 4; ++sx)
        mma16816(facc[t], h0[2 * sx], h1[2 * sx], h0[2 * sx + 1], h1[2 * sx + 1], xq[2 * sx],
                 xq[2 * sx + 1]);
    }
  }

#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    xf[(warp * 16 + r) * NTOK + 8 * t + c * 2 + 0] = facc[t][0];
    xf[(warp * 16 + r) * NTOK + 8 * t + c * 2 + 1] = facc[t][1];
    xf[(warp * 16 + r + 8) * NTOK + 8 * t + c * 2 + 0] = facc[t][2];
    xf[(warp * 16 + r + 8) * NTOK + 8 * t + c * 2 + 1] = facc[t][3];
  }
  __syncthreads();
  const int tb = tile_base[tile];
  for (int i = tid; i < n * 16; i += 256) {
    const int j = i % n, cch = i / n;
    const int row0 = cch * 8;
    uint4 ov;
    uint32_t* op = (uint32_t*)&ov;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      __half2 hv = __float22half2_rn(
          make_float2(xf[(row0 + 2 * q) * NTOK + j], xf[(row0 + 2 * q + 1) * NTOK + j]));
      op[q] = *(const uint32_t*)&hv;
    }
    *(uint4*)(gu + (size_t)sorted[tb + j] * TWO_I + rb * 128 + row0) = ov;
  }
}

// FC2 for large T: block = 128 output rows x NTOK tokens, warp = 16 rows.
__global__ __launch_bounds__(256) void k_gemm2_big(
    const __half* __restrict__ inter, const uint8_t* __restrict__ w2,
    const float* __restrict__ w2_scale, const float* __restrict__ topk_weights,
    const int* __restrict__ sorted, const int* __restrict__ tile_e,
    const int* __restrict__ tile_base, const int* __restrict__ tile_n,
    __nv_bfloat16* __restrict__ out) {
  const int tile = blockIdx.y;
  const int n = tile_n[tile];
  if (n == 0) return;
  const int e = tile_e[tile];
  const int hb = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r = lane >> 2, c = lane & 3;

  __shared__ int s_pair[NTOK];
  __shared__ float xf[128 * NTOK];
  if (tid < NTOK) s_pair[tid] = (tid < n) ? sorted[tile_base[tile] + tid] : 0;
  __syncthreads();

  const int rowbase = hb * 128 + warp * 16;
  const uint8_t* wptr = w2 + ((size_t)e * H_DIM + rowbase) * I_DIM + c * 16;
  const float* wsc = w2_scale + ((size_t)e * 16 + (rowbase >> 7)) * 4;
  int xoff[NTILES];
#pragma unroll
  for (int t = 0; t < NTILES; ++t) xoff[t] = s_pair[8 * t + r] * I_DIM + c * 16;

  float facc[NTILES][4];
#pragma unroll
  for (int t = 0; t < NTILES; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) facc[t][i] = 0.f;

  const int ntl = (n + 7) >> 3;
#pragma unroll 1
  for (int ks = 0; ks < 8; ++ks) {
    const int k = ks * 64;
    const __half2 sc2 = __float2half2_rn(wsc[ks >> 1]);
    uint4 w0 = *(const uint4*)(wptr + (size_t)r * I_DIM + k);
    uint4 w1 = *(const uint4*)(wptr + (size_t)(r + 8) * I_DIM + k);
    uint32_t h0[8], h1[8];
    unpack16(w0, h0);
    unpack16(w1, h1);
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      __half2 a = __hmul2(*(const __half2*)&h0[q], sc2);
      __half2 b = __hmul2(*(const __half2*)&h1[q], sc2);
      h0[q] = *(const uint32_t*)&a;
      h1[q] = *(const uint32_t*)&b;
    }
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      if (t >= ntl) continue;
      uint4 x0 = *(const uint4*)(inter + xoff[t] + k);
      uint4 x1 = *(const uint4*)(inter + xoff[t] + k + 8);
      uint32_t xq[8];
      xq[0] = x0.x;
      xq[1] = x0.y;
      xq[2] = x0.z;
      xq[3] = x0.w;
      xq[4] = x1.x;
      xq[5] = x1.y;
      xq[6] = x1.z;
      xq[7] = x1.w;
#pragma unroll
      for (int sx = 0; sx < 4; ++sx)
        mma16816(facc[t], h0[2 * sx], h1[2 * sx], h0[2 * sx + 1], h1[2 * sx + 1], xq[2 * sx],
                 xq[2 * sx + 1]);
    }
  }

#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    xf[(warp * 16 + r) * NTOK + 8 * t + c * 2 + 0] = facc[t][0];
    xf[(warp * 16 + r) * NTOK + 8 * t + c * 2 + 1] = facc[t][1];
    xf[(warp * 16 + r + 8) * NTOK + 8 * t + c * 2 + 0] = facc[t][2];
    xf[(warp * 16 + r + 8) * NTOK + 8 * t + c * 2 + 1] = facc[t][3];
  }
  __syncthreads();
  for (int i = tid; i < n * 64; i += 256) {
    const int j = i % n, rp = i / n;
    const int p = s_pair[j];
    const float tw = topk_weights[p];
    __nv_bfloat162 v =
        __floats2bfloat162_rn(xf[(2 * rp) * NTOK + j] * tw, xf[(2 * rp + 1) * NTOK + j] * tw);
    atomicAdd((__nv_bfloat162*)(out + (size_t)(p >> 3) * H_DIM + hb * 128 + 2 * rp), v);
  }
}
