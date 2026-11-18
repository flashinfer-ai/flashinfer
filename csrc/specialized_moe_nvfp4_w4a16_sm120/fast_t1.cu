// Fused NVFP4-weight / bf16-activation MoE (non-gated relu^2) for sm_120.
//
//   prep    : counting-sort of topk_ids into expert groups, bf16->fp16 copy of
//             the activations, zero-init of the bf16 destination.
//   phase 1 : h   = relu(a1 * x . W1^T)^2          (fp16 scratch)
//   phase 2 : out = sum_slots tw * a2 * h . W2^T   (bf16 atomics)
//
// Both phases use one templated GEMV kernel: A is (BT x K) fp16 with K
// contiguous, B is (N x K) NVFP4 (e2m1 nibble pairs, one e4m3 scale per 16
// values along K, one fp32 per-expert global scale).  A lane owns 8 contiguous
// K values, i.e. 4 packed weight bytes and 16 activation bytes, so every
// global access of both operands is a fully coalesced 128 B / 512 B burst.
// The weight stream is software pipelined one step ahead to keep enough
// DRAM requests in flight; the activation stream is served by L1.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <vector>

namespace ffi = tvm::ffi;

void RunLarge(ffi::TensorView hidden_states, ffi::TensorView w1_packed, ffi::TensorView w1_gs,
              ffi::TensorView w1_alpha, ffi::TensorView w2_packed, ffi::TensorView w2_gs,
              ffi::TensorView w2_alpha, ffi::TensorView topk_ids, ffi::TensorView topk_weights,
              ffi::TensorView out);

#define HID 2688
#define INTM 1856
#define NEXP 128
#define TOPKV 6
// Token rows batched per expert group at T=64: larger BT reuses each
// dequantized weight tile across more rows and cuts L2 weight re-reads.
#define T64_BT 2
// T=64 ran at 8 CTAs/SM (64 registers) = 32 of 48 warps while DRAM sat at
// 91%; ask for 10 so more weight streams are in flight per SM.
#define T64_MINB 4
#define T64_NWARP 4
// n-rows per warp: each is an independent 128-bit weight load issued against
// one shared activation register set, so RR sets the per-warp load depth.
#define T64_RR 4

// ---------------------------------------------------------------- helpers

__host__ __device__ constexpr int ilog2c(int n) { return (n <= 1) ? 0 : 1 + ilog2c(n >> 1); }

// Packed bf16x2 held in a 32-bit register.  BF16 keeps the activation
// exponent range of the public contract end to end; FP16 would clamp legal
// bf16 magnitudes at 65504 and overflow on relu^2 intermediates.
typedef uint32_t bf2;

__device__ __forceinline__ void dq8(uint32_t w, bf2* o) {
  asm("{\n\t.reg .b8 x0,x1,x2,x3;\n\t"
      "mov.b32 {x0,x1,x2,x3}, %4;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %0, x0;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %1, x1;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %2, x2;\n\t"
      "cvt.rn.bf16x2.e2m1x2 %3, x3;\n\t}"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
      : "r"(w));
}

__device__ __forceinline__ bf2 sc2(uint32_t s) {
  uint32_t r;
  unsigned short packed = (unsigned short)((s & 0xFF) | ((s & 0xFF) << 8));
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(packed));
  return r;
}

__device__ __forceinline__ bf2 bfma(bf2 a, bf2 b, bf2 c) {
  bf2 d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ bf2 bmul(bf2 a, bf2 b) {
  bf2 d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float blow(bf2 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhigh(bf2 v) { return __uint_as_float(v & 0xffff0000u); }
__device__ __forceinline__ float2 b2f2(bf2 v) { return make_float2(blow(v), bhigh(v)); }

__device__ __forceinline__ uint32_t ldcg32(const void* p) {
  uint32_t r;
  asm("ld.global.cg.L2::128B.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldcg8(const void* p) {
  uint32_t r;
  asm("ld.global.cg.L2::256B.u8 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
struct U64x2 {
  uint64_t x, y;
};
__device__ __forceinline__ U64x2 ldcg128(const void* p) {
  U64x2 v;
  asm("ld.global.cs.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}
// Two adjacent e4m3 group scales -> one bf16x2 (low half = first group).
__device__ __forceinline__ bf2 sc2pair(uint32_t s) {
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(s)));
  return r;
}
__device__ __forceinline__ void ld_l1_evict_last_32B(const void* p, bf2* o) {
  asm volatile(
      "ld.global.L1::evict_last.v8.u32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]), "=r"(o[4]), "=r"(o[5]), "=r"(o[6]),
        "=r"(o[7])
      : "l"(p));
}
__device__ __forceinline__ void ld_cg_32B(const void* p, bf2* o) {
  asm volatile(
      "ld.global.nc.v8.u32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]), "=r"(o[4]), "=r"(o[5]), "=r"(o[6]),
        "=r"(o[7])
      : "l"(p));
}
__device__ __forceinline__ uint64_t ldcg64(const void* p) {
  uint64_t r;
  asm("ld.global.cg.u64 %0, [%1];" : "=l"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ld_l1_evict_last_u128(const void* p) {
  uint4 v;
  asm volatile("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ float warp_sum32(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
  return x;
}

__device__ __forceinline__ void red_bf16x2_v2(__nv_bfloat162* dst, __nv_bfloat162 v0,
                                              __nv_bfloat162 v1) {
  const uint32_t a = *reinterpret_cast<const uint32_t*>(&v0);
  const uint32_t b = *reinterpret_cast<const uint32_t*>(&v1);
  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n"
               :
               : "l"(dst), "r"(a), "r"(b)
               : "memory");
}

// ---------------------------------------------------------------- prep

template <int BT>
__global__ __launch_bounds__(256) void prep_kernel(const int* __restrict__ topk_ids, int T,
                                                   const __nv_bfloat16* __restrict__ hs,
                                                   int* __restrict__ sorted_route,
                                                   uint32_t* __restrict__ grp_desc, int Gmax,
                                                   __nv_bfloat16* __restrict__ outp) {
  const int tid = threadIdx.x;

  if (blockIdx.x == 0) {
    __shared__ int cnt[NEXP];
    __shared__ int off[NEXP];
    __shared__ int gof[NEXP];
    __shared__ int cur[NEXP];
    const int P = T * TOPKV;
    for (int g = tid; g < Gmax; g += blockDim.x) grp_desc[g] = 0u;
    if (tid < NEXP) {
      cnt[tid] = 0;
      cur[tid] = 0;
    }
    __syncthreads();
    for (int i = tid; i < P; i += 256) {
      const int e = topk_ids[i];
      if (unsigned(e) < NEXP) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    // Two-level inclusive scans over 128 experts: scan within each warp,
    // then scan the four warp totals.
    int c = (tid < NEXP) ? cnt[tid] : 0;
    int gc = (tid < NEXP) ? (c + BT - 1) / BT : 0;
    int a = c, b = gc;
    const int lane = tid & 31;
    const int warp = tid >> 5;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int pa = __shfl_up_sync(0xffffffffu, a, d);
      const int pb = __shfl_up_sync(0xffffffffu, b, d);
      if (lane >= d) {
        a += pa;
        b += pb;
      }
    }
    if (tid < NEXP && lane == 31) {
      off[warp] = a;
      gof[warp] = b;
    }
    __syncthreads();
    if (tid < 32) {
      int wa = lane < 4 ? off[lane] : 0;
      int wb = lane < 4 ? gof[lane] : 0;
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        const int pa = __shfl_up_sync(0xffffffffu, wa, d);
        const int pb = __shfl_up_sync(0xffffffffu, wb, d);
        if (lane >= d) {
          wa += pa;
          wb += pb;
        }
      }
      if (lane < 4) {
        cnt[lane] = wa;
        gof[lane] = wb;
      }
    }
    __syncthreads();
    int myoff = 0, mygof = 0;
    if (tid < NEXP) {
      const int abase = warp ? cnt[warp - 1] : 0;
      const int bbase = warp ? gof[warp - 1] : 0;
      myoff = abase + a - c;
      mygof = bbase + b - gc;
      off[tid] = myoff;
    }
    __syncthreads();
    for (int i = tid; i < P; i += 256) {
      int e = topk_ids[i];
      if (unsigned(e) < NEXP) {
        int pos = off[e] + atomicAdd(&cur[e], 1);
        sorted_route[pos] = i;
      }
    }
    if (tid < NEXP) {
      int g = mygof;
      for (int bb = 0; bb < c; bb += BT) {
        const uint32_t poff = uint32_t(myoff + bb);
        const uint32_t nn = uint32_t(min(BT, c - bb));
        grp_desc[g] = uint32_t(tid) | (poff << 8) | (nn << 24);
        ++g;
      }
    }
  }

  const long total = (long)T * (HID / 2);
  const long stride = (long)gridDim.x * 256;
  uint32_t* dsto = reinterpret_cast<uint32_t*>(outp);
  for (long i = (long)blockIdx.x * 256 + tid; i < total; i += stride) dsto[i] = 0u;
}

__global__ __launch_bounds__(256) void prep_direct(__nv_bfloat16* __restrict__ outp, int T) {
  const long total = (long)T * (HID / 2);
  const long stride = (long)gridDim.x * blockDim.x;
  uint32_t* dsto = reinterpret_cast<uint32_t*>(outp);
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) dsto[i] = 0u;
}

// ---------------------------------------------------------------- gemv

#define LOADW(WB, SB, SS)                                                   \
  {                                                                         \
    const int hg_ = lane + 32 * (SS);                                       \
    const bool a_ = (hg_ < KH);                                             \
    SB = 0u;                                                                \
    _Pragma("unroll") for (int r = 0; r < RR; ++r) {                        \
      WB[r] = a_ ? ldcg32(wp + (size_t)r * KB + hg_ * 4) : 0u;              \
      SB |= (a_ ? ldcg8(wg + (size_t)r * KG + (hg_ >> 1)) : 0u) << (8 * r); \
    }                                                                       \
  }

#define COMPUTE(WB, SB, SS)                                                        \
  {                                                                                \
    const int hg_ = lane + 32 * (SS);                                              \
    const bool a_ = (hg_ < KH);                                                    \
    bf2 wq[RR][4];                                                                 \
    _Pragma("unroll") for (int r = 0; r < RR; ++r) {                               \
      dq8(WB[r], wq[r]);                                                           \
      bf2 sv = sc2((SB >> (8 * r)) & 0xFFu);                                       \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) wq[r][j] = bmul(wq[r][j], sv); \
    }                                                                              \
    _Pragma("unroll") for (int tb = 0; tb < BT; tb += TS) {                        \
      uint4 xv[TS];                                                                \
      _Pragma("unroll") for (int u = 0; u < TS; ++u) xv[u] =                       \
          a_ ? __ldg(reinterpret_cast<const uint4*>(A + arow[tb + u] + hg_ * 8))   \
             : make_uint4(0, 0, 0, 0);                                             \
      _Pragma("unroll") for (int u = 0; u < TS; ++u) {                             \
        const bf2* xp = reinterpret_cast<const bf2*>(&xv[u]);                      \
        _Pragma("unroll") for (int r = 0; r < RR; ++r)                             \
            _Pragma("unroll") for (int j = 0; j < 4; ++j) acc[tb + u][r] =         \
                bfma(wq[r][j], xp[j], acc[tb + u][r]);                             \
      }                                                                            \
    }                                                                              \
  }

template <int BT, int RR, int PHASE, bool DIRECT = false>
__device__ __forceinline__ void gemv_body(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha,
    const int* __restrict__ sorted_tok, const float* __restrict__ sorted_w,
    __nv_bfloat16* __restrict__ Hbuf, __nv_bfloat16* __restrict__ Out, int e, int poff, int nn,
    int lane, int n0) {
  constexpr int K = (PHASE == 1) ? HID : INTM;
  constexpr int N = (PHASE == 1) ? INTM : HID;
  constexpr int KH = K / 8;
  constexpr int KG = K / 16;
  constexpr int KB = K / 2;
  constexpr int NSTEP = (KH + 31) / 32;
  constexpr int NV = BT * RR;
  constexpr int TS = (BT < 4) ? BT : 4;
  constexpr int LOG2NV = ilog2c(NV);
  constexpr int LOWB = 5 - LOG2NV;

  int arow[BT];
#pragma unroll
  for (int t = 0; t < BT; ++t) {
    int tt = (t < nn) ? t : 0;
    if constexpr (PHASE == 1) {
      const int tok = DIRECT ? (poff + tt) / TOPKV : sorted_tok[poff + tt];
      arow[t] = tok * HID;
    } else {
      arow[t] = (poff + tt) * INTM;
    }
  }

  const uint8_t* wp = Wp + ((size_t)e * N + n0) * KB;
  const uint8_t* wg = Wg + ((size_t)e * N + n0) * KG;

  bf2 acc[BT][RR];
#pragma unroll
  for (int t = 0; t < BT; ++t)
#pragma unroll
    for (int r = 0; r < RR; ++r) acc[t][r] = 0u;

  uint32_t wa[RR], wb[RR];
  uint32_t sa = 0u, sb = 0u;
  LOADW(wa, sa, 0)
  for (int s = 0; s < NSTEP; s += 2) {
    LOADW(wb, sb, s + 1)
    COMPUTE(wa, sa, s)
    LOADW(wa, sa, s + 2)
    if (s + 1 < NSTEP) {
      COMPUTE(wb, sb, s + 1)
    }
  }

  float v[NV];
#pragma unroll
  for (int t = 0; t < BT; ++t)
#pragma unroll
    for (int r = 0; r < RR; ++r) v[t * RR + r] = blow(acc[t][r]) + bhigh(acc[t][r]);
  {
    int cntv = NV;
#pragma unroll
    for (int lvl = 0; lvl < LOG2NV; ++lvl) {
      const int bit = 16 >> lvl;
      const int hf = cntv >> 1;
#pragma unroll
      for (int i = 0; i < (NV > 1 ? NV / 2 : 1); ++i) {
        if (i < hf) {
          float snd = (lane & bit) ? v[i] : v[i + hf];
          float kp = (lane & bit) ? v[i + hf] : v[i];
          v[i] = kp + __shfl_xor_sync(0xffffffffu, snd, bit);
        }
      }
      cntv = hf;
    }
#pragma unroll
    for (int bit = (1 << LOWB) >> 1; bit > 0; bit >>= 1)
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], bit);
  }

  const bool active = ((lane & ((1 << LOWB) - 1)) == 0);
  const int myidx = lane >> LOWB;
  const float al = alpha[e];
  const float res = v[0];
  const float res_hi = __shfl_down_sync(0xffffffffu, res, 1 << LOWB);

  if (PHASE == 1) {
#pragma unroll
    for (int t = 0; t < BT; ++t)
#pragma unroll
      for (int r = 0; r < RR; ++r) {
        if (active && myidx == t * RR + r && t < nn) {
          float z = res * al;
          z = (z > 0.f) ? z * z : 0.f;
          Hbuf[(size_t)(poff + t) * INTM + n0 + r] = __float2bfloat16_rn(z);
        }
      }
  } else {
#pragma unroll
    for (int t = 0; t < BT; ++t)
#pragma unroll
      for (int r = 0; r < RR; r += 2) {
        if (active && myidx == t * RR + r && t < nn) {
          float sc = al * sorted_w[poff + t];
          int tk;
          if constexpr (DIRECT)
            tk = (poff + t) / TOPKV;
          else
            tk = sorted_tok[poff + t];
          __nv_bfloat162 val = __floats2bfloat162_rn(res * sc, res_hi * sc);
          atomicAdd(reinterpret_cast<__nv_bfloat162*>(Out + (size_t)tk * HID + n0 + r), val);
        }
      }
  }
}

template <int BTMAX, int NWARP, int RR, int MINB, int PHASE>
__global__ __launch_bounds__(NWARP * 32, MINB) void moe_gemv(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha,
    const int* __restrict__ sorted_tok, const float* __restrict__ sorted_w,
    const int* __restrict__ grp_e, const int* __restrict__ grp_off, const int* __restrict__ grp_n,
    const int* __restrict__ nG, __nv_bfloat16* __restrict__ Hbuf, __nv_bfloat16* __restrict__ Out) {
  const int g = blockIdx.x;
  if (g >= *nG) return;
  const int e = grp_e[g];
  const int poff = grp_off[g];
  const int nn = grp_n[g];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.y * (NWARP * RR) + warp * RR;

  if (BTMAX >= 8 && nn > 4) {
    gemv_body<(BTMAX >= 8 ? 8 : BTMAX), RR, PHASE>(A, Wp, Wg, alpha, sorted_tok, sorted_w, Hbuf,
                                                   Out, e, poff, nn, lane, n0);
  } else if (BTMAX >= 4 && nn > 2) {
    gemv_body<(BTMAX >= 4 ? 4 : BTMAX), RR, PHASE>(A, Wp, Wg, alpha, sorted_tok, sorted_w, Hbuf,
                                                   Out, e, poff, nn, lane, n0);
  } else if (BTMAX >= 2 && nn > 1) {
    gemv_body<(BTMAX >= 2 ? 2 : BTMAX), RR, PHASE>(A, Wp, Wg, alpha, sorted_tok, sorted_w, Hbuf,
                                                   Out, e, poff, nn, lane, n0);
  } else {
    gemv_body<1, RR, PHASE>(A, Wp, Wg, alpha, sorted_tok, sorted_w, Hbuf, Out, e, poff, nn, lane,
                            n0);
  }
}

template <int NWARP, int RR, int MINB, int PHASE>
__global__ __launch_bounds__(NWARP * 32, MINB) void moe_gemv_direct(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha, const int* __restrict__ ids,
    const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ Hbuf,
    __nv_bfloat16* __restrict__ Out) {
  const int route = blockIdx.x;
  const int e = ids[route];
  if (unsigned(e) >= NEXP) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.y * (NWARP * RR) + warp * RR;
  gemv_body<1, RR, PHASE, true>(A, Wp, Wg, alpha, nullptr, topk_w, Hbuf, Out, e, route, 1, lane,
                                n0);
}

template <int NWARP, int RR, int MINB, int PHASE>
__global__ __launch_bounds__(NWARP * 32, MINB) void moe_gemv_group16_direct(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha, const int* __restrict__ ids,
    const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ Hbuf,
    __nv_bfloat16* __restrict__ Out) {
  constexpr int K = (PHASE == 1) ? HID : INTM;
  constexpr int N = (PHASE == 1) ? INTM : HID;
  constexpr int KG = K / 16;
  constexpr int KB = K / 2;
  const int route = blockIdx.x;
  if constexpr (PHASE == 1) {
    const int bid = int(blockIdx.y) * int(gridDim.x) + int(blockIdx.x);
    const int z = bid * int(blockDim.x) + int(threadIdx.x);
    const int total_pairs = (int(gridDim.x) / TOPKV) * (HID / 2);
    if (z < total_pairs) reinterpret_cast<uint32_t*>(Out)[z] = 0u;
  }
  const int e = ids[route];
  if (unsigned(e) >= NEXP) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.y * (NWARP * RR) + warp * RR;
  const int arow = (PHASE == 1) ? (route / TOPKV) * HID : route * INTM;
  const __nv_bfloat16* ap = A + arow;
  const uint8_t* wp = Wp + ((size_t)e * N + n0) * KB;
  const uint8_t* wg = Wg + ((size_t)e * N + n0) * KG;

  float acc[RR];
#pragma unroll
  for (int r = 0; r < RR; ++r) acc[r] = 0.0f;

  for (int gb = 0; gb < KG; gb += 32) {
    const int group = gb + lane;
    if (group < KG) {
      uint4 xv[2];
      xv[0] = __ldg(reinterpret_cast<const uint4*>(ap + group * 16));
      xv[1] = __ldg(reinterpret_cast<const uint4*>(ap + group * 16 + 8));
      const bf2* xp = reinterpret_cast<const bf2*>(xv);
#pragma unroll
      for (int r = 0; r < RR; ++r) {
        const uint64_t q8 = ldcg64(wp + (size_t)r * KB + group * 8);
        bf2 wq[8];
        dq8(uint32_t(q8), wq);
        dq8(uint32_t(q8 >> 32), wq + 4);
        bf2 sum = 0u;
#pragma unroll
        for (int p = 0; p < 8; ++p) sum = bfma(wq[p], xp[p], sum);
        const float2 pair = b2f2(sum);
        const float sf = blow(sc2(ldcg8(wg + (size_t)r * KG + group)));
        acc[r] = fmaf(pair.x + pair.y, sf, acc[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RR; ++r) acc[r] = warp_sum32(acc[r]);
  if (lane == 0) {
    const float al = alpha[e];
    if constexpr (PHASE == 1) {
#pragma unroll
      for (int r = 0; r < RR; ++r) {
        float z = acc[r] * al;
        z = (z > 0.0f) ? z * z : 0.0f;
        Hbuf[(size_t)route * INTM + n0 + r] = __float2bfloat16_rn(z);
      }
    } else {
      const float sc = al * topk_w[route];
#pragma unroll
      for (int r = 0; r < RR; r += 2) {
        const __nv_bfloat162 val = __floats2bfloat162_rn(acc[r] * sc, acc[r + 1] * sc);
        atomicAdd(reinterpret_cast<__nv_bfloat162*>(Out + (size_t)(route / TOPKV) * HID + n0 + r),
                  val);
      }
    }
  }
}

// One expert group: BT token rows x RR n-rows per warp.  The bin size is
// fixed by prep, but the body is instantiated per actual row count so a short
// tail bin does not pay for the rows it does not own.
template <int BT, int RR, int PHASE>
__device__ __forceinline__ void g16_body(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha,
    const int* __restrict__ sorted_route, const float* __restrict__ topk_w,
    __nv_bfloat16* __restrict__ Hbuf, __nv_bfloat16* __restrict__ Out, int e, int poff, int nn,
    int lane, int n0) {
  constexpr int K = (PHASE == 1) ? HID : INTM;
  constexpr int N = (PHASE == 1) ? INTM : HID;
  constexpr int KG = K / 16;
  constexpr int KB = K / 2;
  const uint8_t* wp = Wp + ((size_t)e * N + n0) * KB;
  const uint8_t* wg = Wg + ((size_t)e * N + n0) * KG;

  int arow[BT];
#pragma unroll
  for (int t = 0; t < BT; ++t) {
    const int tt = (t < nn) ? t : 0;
    const int route = sorted_route[poff + tt];
    arow[t] = (PHASE == 1) ? (route / TOPKV) * HID : (poff + tt) * INTM;
  }
  float acc[BT][RR];
#pragma unroll
  for (int t = 0; t < BT; ++t)
#pragma unroll
    for (int r = 0; r < RR; ++r) acc[t][r] = 0.0f;

  // Two group-16 columns per lane: one 128-bit weight load and one 16-bit
  // scale-pair load replace two of each, halving the memory instructions
  // that carry the dominant weight stream.
  static_assert(KG % 2 == 0, "paired group iteration needs an even KG");
  for (int gb = 0; gb < KG; gb += 64) {
    const int group = gb + lane * 2;
    if (group < KG) {
      bf2 xv[BT][16];
#pragma unroll
      for (int t = 0; t < BT; ++t) {
        const __nv_bfloat16* ap = A + arow[t] + group * 16;
        if constexpr (PHASE == 1) {
          ld_l1_evict_last_32B(ap, xv[t]);
          ld_l1_evict_last_32B(ap + 16, xv[t] + 8);
        } else {
          ld_cg_32B(ap, xv[t]);
          ld_cg_32B(ap + 16, xv[t] + 8);
        }
      }
#pragma unroll
      for (int r = 0; r < RR; ++r) {
        const U64x2 q16 = ldcg128(wp + (size_t)r * KB + group * 8);
        const bf2 sfp = sc2pair(*reinterpret_cast<const uint16_t*>(wg + (size_t)r * KG + group));
        const float sf0 = blow(sfp);
        const float sf1 = bhigh(sfp);
        bf2 wq[8];
        dq8(uint32_t(q16.x), wq);
        dq8(uint32_t(q16.x >> 32), wq + 4);
#pragma unroll
        for (int t = 0; t < BT; ++t) {
          bf2 sum = 0u;
#pragma unroll
          for (int p = 0; p < 8; ++p) sum = bfma(wq[p], xv[t][p], sum);
          const float2 pair = b2f2(sum);
          acc[t][r] = fmaf(pair.x + pair.y, sf0, acc[t][r]);
        }
        dq8(uint32_t(q16.y), wq);
        dq8(uint32_t(q16.y >> 32), wq + 4);
#pragma unroll
        for (int t = 0; t < BT; ++t) {
          bf2 sum = 0u;
#pragma unroll
          for (int p = 0; p < 8; ++p) sum = bfma(wq[p], xv[t][8 + p], sum);
          const float2 pair = b2f2(sum);
          acc[t][r] = fmaf(pair.x + pair.y, sf1, acc[t][r]);
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < BT; ++t)
#pragma unroll
    for (int r = 0; r < RR; ++r) acc[t][r] = warp_sum32(acc[t][r]);
  if (lane == 0) {
    const float al = alpha[e];
#pragma unroll
    for (int t = 0; t < BT; ++t) {
      if (t < nn) {
        if constexpr (PHASE == 1) {
#pragma unroll
          for (int r = 0; r < RR; ++r) {
            float z = acc[t][r] * al;
            z = (z > 0.0f) ? z * z : 0.0f;
            Hbuf[(size_t)(poff + t) * INTM + n0 + r] = __float2bfloat16_rn(z);
          }
        } else {
          const int route = sorted_route[poff + t];
          const float sc = al * topk_w[route];
          const int tok = route / TOPKV;
          if constexpr (RR == 4) {
            const __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[t][0] * sc, acc[t][1] * sc);
            const __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[t][2] * sc, acc[t][3] * sc);
            red_bf16x2_v2(reinterpret_cast<__nv_bfloat162*>(Out + (size_t)tok * HID + n0), v0, v1);
          } else {
#pragma unroll
            for (int r = 0; r < RR; r += 2) {
              const __nv_bfloat162 val = __floats2bfloat162_rn(acc[t][r] * sc, acc[t][r + 1] * sc);
              atomicAdd(reinterpret_cast<__nv_bfloat162*>(Out + (size_t)tok * HID + n0 + r), val);
            }
          }
        }
      }
    }
  }
}

template <int BTMAX, int NWARP, int RR, int MINB, int PHASE>
__global__ __launch_bounds__(NWARP * 32, MINB) void moe_gemv_group16(
    const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Wp,
    const uint8_t* __restrict__ Wg, const float* __restrict__ alpha,
    const int* __restrict__ sorted_route, const float* __restrict__ topk_w,
    const uint32_t* __restrict__ grp_desc, __nv_bfloat16* __restrict__ Hbuf,
    __nv_bfloat16* __restrict__ Out) {
  const uint32_t desc = grp_desc[blockIdx.x];
  if (desc == 0u) return;
  const int e = int(desc & 0xFFu);
  const int poff = int((desc >> 8) & 0xFFFFu);
  const int nn = int(desc >> 24);
  const int lane = threadIdx.x & 31;
  const int n0 = blockIdx.y * (NWARP * RR) + (threadIdx.x >> 5) * RR;
  if (BTMAX >= 4 && nn > 2) {
    g16_body<(BTMAX >= 4 ? 4 : BTMAX), RR, PHASE>(A, Wp, Wg, alpha, sorted_route, topk_w, Hbuf, Out,
                                                  e, poff, nn, lane, n0);
  } else if (BTMAX >= 2 && nn > 1) {
    g16_body<(BTMAX >= 2 ? 2 : BTMAX), RR, PHASE>(A, Wp, Wg, alpha, sorted_route, topk_w, Hbuf, Out,
                                                  e, poff, nn, lane, n0);
  } else {
    g16_body<1, RR, PHASE>(A, Wp, Wg, alpha, sorted_route, topk_w, Hbuf, Out, e, poff, nn, lane,
                           n0);
  }
}

// ---------------------------------------------------------------- host

// Device scratch arena (workspace only - never holds results between calls).
// Buffers retired by growth stay alive for the process lifetime: a CUDA graph
// captured earlier may still replay kernels holding the old address. Growth
// is monotone, so the retained total stays bounded (< 2x the final capacity).
static void* g_ws = nullptr;
static size_t g_ws_bytes = 0;
static std::vector<void*> g_retired_ws;
static void* ws_get(size_t bytes) {
  if (bytes > g_ws_bytes) {
    if (g_ws) g_retired_ws.push_back(g_ws);
    g_ws = nullptr;
    g_ws_bytes = 0;
    cudaError_t err = cudaMalloc(&g_ws, bytes);
    TVM_FFI_ICHECK(err == cudaSuccess && g_ws != nullptr)
        << "specialized MoE workspace cudaMalloc(" << bytes
        << ") failed: " << cudaGetErrorString(err);
    g_ws_bytes = bytes;
  }
  return g_ws;
}

template <int NWARP, int RR, int MINB>
static void launch_direct(int T, const __nv_bfloat16* hs, const uint8_t* w1p, const uint8_t* w1g,
                          const float* w1a, const uint8_t* w2p, const uint8_t* w2g,
                          const float* w2a, const int* ids, const float* tw, __nv_bfloat16* out,
                          void* ws, cudaStream_t st) {
  const int P = T * TOPKV;
  __nv_bfloat16* hbuf = reinterpret_cast<__nv_bfloat16*>(ws);

  constexpr int BN = NWARP * RR;
  dim3 g1(P, INTM / BN), g2(P, HID / BN);
  if (T <= 4) {
    moe_gemv_group16_direct<NWARP, RR, MINB, 1>
        <<<g1, NWARP * 32, 0, st>>>(hs, w1p, w1g, w1a, ids, tw, hbuf, out);
    moe_gemv_group16_direct<NWARP, RR, MINB, 2>
        <<<g2, NWARP * 32, 0, st>>>(hbuf, w2p, w2g, w2a, ids, tw, hbuf, out);
  } else {
    moe_gemv_direct<NWARP, RR, MINB, 1>
        <<<g1, NWARP * 32, 0, st>>>(hs, w1p, w1g, w1a, ids, tw, hbuf, out);
    moe_gemv_direct<NWARP, RR, MINB, 2>
        <<<g2, NWARP * 32, 0, st>>>(hbuf, w2p, w2g, w2a, ids, tw, hbuf, out);
  }
}

template <int BT, int NWARP, int RR, int MINB>
static void launch_all(int T, int P, int Gmax, const __nv_bfloat16* hs, const uint8_t* w1p,
                       const uint8_t* w1g, const float* w1a, const uint8_t* w2p, const uint8_t* w2g,
                       const float* w2a, const int* ids, const float* tw, __nv_bfloat16* out,
                       void* ws, cudaStream_t st) {
  char* p = reinterpret_cast<char*>(ws);
  __nv_bfloat16* hbuf = reinterpret_cast<__nv_bfloat16*>(p);
  p += (size_t)P * INTM * sizeof(__nv_bfloat16);
  int* sorted_route = reinterpret_cast<int*>(p);
  p += (size_t)P * 4;
  uint32_t* grp_desc = reinterpret_cast<uint32_t*>(p);

  int pb = (int)(((long)T * (HID / 2) + 255) / 256);
  if (pb < 1) pb = 1;
  if (pb > 2048) pb = 2048;
  prep_kernel<BT><<<pb, 256, 0, st>>>(ids, T, hs, sorted_route, grp_desc, Gmax, out);

  constexpr int BN = NWARP * RR;
  dim3 g1(Gmax, INTM / BN), g2(Gmax, HID / BN);
  moe_gemv_group16<BT, NWARP, RR, MINB, 1>
      <<<g1, NWARP * 32, 0, st>>>(hs, w1p, w1g, w1a, sorted_route, tw, grp_desc, hbuf, out);
  moe_gemv_group16<BT, NWARP, RR, MINB, 2>
      <<<g2, NWARP * 32, 0, st>>>(hbuf, w2p, w2g, w2a, sorted_route, tw, grp_desc, hbuf, out);
}

void RunFastDirect(ffi::TensorView hidden_states, ffi::TensorView w1_packed, ffi::TensorView w1_gs,
                   ffi::TensorView w1_alpha, ffi::TensorView w2_packed, ffi::TensorView w2_gs,
                   ffi::TensorView w2_alpha, ffi::TensorView topk_ids, ffi::TensorView topk_weights,
                   ffi::TensorView out, cudaStream_t st) {
  const int T = int(hidden_states.size(0));
  const int P = T * TOPKV;
  // sum_e ceil(count_e / BT) <= (P + (BT - 1) * active_experts) / BT.
  const int Gmax = (P + (T64_BT - 1) * NEXP) / T64_BT + 1;
  void* ws =
      ws_get((size_t)P * INTM * sizeof(__nv_bfloat16) + (size_t)P * 4 + (size_t)Gmax * 12 + 1028);

  const __nv_bfloat16* hs = (const __nv_bfloat16*)hidden_states.data_ptr();
  const uint8_t* w1p = (const uint8_t*)w1_packed.data_ptr();
  const uint8_t* w1g = (const uint8_t*)w1_gs.data_ptr();
  const float* w1a = (const float*)w1_alpha.data_ptr();
  const uint8_t* w2p = (const uint8_t*)w2_packed.data_ptr();
  const uint8_t* w2g = (const uint8_t*)w2_gs.data_ptr();
  const float* w2a = (const float*)w2_alpha.data_ptr();
  const int* ids = (const int*)topk_ids.data_ptr();
  const float* tw = (const float*)topk_weights.data_ptr();
  __nv_bfloat16* o = (__nv_bfloat16*)out.data_ptr();

  if (T == 64) {
    launch_all<T64_BT, T64_NWARP, T64_RR, T64_MINB>(T, P, Gmax, hs, w1p, w1g, w1a, w2p, w2g, w2a,
                                                    ids, tw, o, ws, st);
  } else if (T == 1) {
    launch_direct<8, 2, 6>(T, hs, w1p, w1g, w1a, w2p, w2g, w2a, ids, tw, o, ws, st);
  } else {
    launch_direct<4, 4, 6>(T, hs, w1p, w1g, w1a, w2p, w2g, w2a, ids, tw, o, ws, st);
  }
}
