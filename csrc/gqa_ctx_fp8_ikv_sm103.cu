// fp8 GQA chunked-prefill attention over a token-interleaved paged KV cache.
// HQ=16 query heads share 1 KV head, D=128, PAGE=128 tokens.
// One warp owns one query token: its 16 heads are the M=16 rows of the mma tiles,
// so the causal mask is uniform over the tile and the QK accumulator feeds the
// PV A-fragment with no data movement.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <cstring>

#ifndef Q2_CAUSAL_PRUNE
#define Q2_CAUSAL_PRUNE 0
#endif
#ifndef TINY_KVCAP
#define TINY_KVCAP 0
#endif

#ifndef TINY_DS
#define TINY_DS 2
#endif

namespace ffi = tvm::ffi;

#define HQ 16
#define DH 128
#define HQD 2048  // HQ*DH : bytes per fp8 query token / elems per bf16 token
#define PAGE 128
#define KVTOK 256  // bytes per token in kv_buf (K then V)
#define VSTRH 136  // halves per Vh row  (272B -> 16B bank rotation)
#define KSTR 144   // bytes per Kraw row (16B bank rotation)
#define OSTRH 136  // halves per output-staging row
#define QSTR 144   // bytes per shared-memory query row (16B bank rotation)

#define SCALE_LOG2E 0.12751743082459868f  // 128**-0.5 * log2(e)
#define RTHRESH 7.84207f                  // 1 binade in raw-score units
#define RMARGIN 15.68414f                 // 2 binades in raw-score units

__device__ __forceinline__ uint32_t smemu32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t fp8x2_to_h2(unsigned short s) {
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)s, __NV_E4M3);
  return (uint32_t)hr.x | ((uint32_t)hr.y << 16);
}

#define CVLO(x) fp8x2_to_h2((unsigned short)((x) & 0xffffu))
#define CVHI(x) fp8x2_to_h2((unsigned short)((x) >> 16))

__device__ __forceinline__ float dequant_sum4(uint32_t x, uint32_t y) {
  uint32_t a0 = fp8x2_to_h2((unsigned short)(x & 0xffffu));
  uint32_t a1 = fp8x2_to_h2((unsigned short)(x >> 16));
  uint32_t a2 = fp8x2_to_h2((unsigned short)(y & 0xffffu));
  uint32_t a3 = fp8x2_to_h2((unsigned short)(y >> 16));
  __half2 s = __hadd2(__hadd2(*reinterpret_cast<__half2*>(&a0), *reinterpret_cast<__half2*>(&a1)),
                      __hadd2(*reinterpret_cast<__half2*>(&a2), *reinterpret_cast<__half2*>(&a3)));
  return __low2float(s) + __high2float(s);
}

// Packed half2 exponential: one ex2.approx.f16x2 replaces two MUFU.EX2 and the
// pack that followed them.  P only ever feeds an f16 MMA, so f16 accuracy here
// is the accuracy the probabilities are stored at anyway.
__device__ __forceinline__ uint32_t ex2_h2(uint32_t x) {
  uint32_t d;
  asm("ex2.approx.f16x2 %0, %1;\n" : "=r"(d) : "r"(x));
  return d;
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  uint32_t d;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(b), "f"(a));
  return d;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  uint32_t d;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(b), "f"(a));
  return d;
}

struct alignas(8) F32x2 {
  float x, y;
};
__device__ __forceinline__ F32x2 fma_f32x2(F32x2 a, F32x2 b, F32x2 c) {
  // memcpy, not reinterpret_cast references: defined object access (the
  // copies compile away; the asm sees the same 64-bit registers).
  uint64_t ua, ub, uc, ud;
  std::memcpy(&ua, &a, sizeof(ua));
  std::memcpy(&ub, &b, sizeof(ub));
  std::memcpy(&uc, &c, sizeof(uc));
  asm("fma.rn.f32x2 %0,%1,%2,%3;\n" : "=l"(ud) : "l"(ua), "l"(ub), "l"(uc));
  F32x2 d;
  std::memcpy(&d, &ud, sizeof(d));
  return d;
}

__device__ __forceinline__ void mma_f8(float& d0, float& d1, float& d2, float& d3,
                                       const uint32_t a[4], const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16_k8(float* d, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16_k16(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                            uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x4_trans(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2_trans(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2_t8(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(b), "f"(a));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(d), "f"(c));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Keep metadata loads at kernel entry instead of allowing them to sink below
// the tiny-path early exit; GPU-scope relaxed loads still retain L2 caching.
__device__ __forceinline__ int ldv32(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ int ldu32(const int* p) {
  int v;
  asm volatile("ldu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void cpasync16z(uint32_t dst, const void* src, int nbytes) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(nbytes));
}

__device__ __forceinline__ void cpasync16(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

// Stage this warp's 16x128 fp8 query tile into shared memory as one more
// cp.async group.  The scattered 4-byte global reads the register path needs
// become four fully coalesced 16B copies that retire with the KV prefetch, and
// ldmatrix.x4 on a 16-row/32-byte slab hands back exactly the four fp8 A
// fragment registers of the k=32 block at that column offset.
__device__ __forceinline__ void stage_q(uint8_t* qs, const uint8_t* __restrict__ q, int tok,
                                        int lane) {
  const uint32_t base = smemu32(qs);
  const uint8_t* g = q + (size_t)tok * HQD + lane * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    cpasync16(base + (uint32_t)((i * 4 + (lane >> 3)) * QSTR + (lane & 7) * 16), g + i * 512);
}

__device__ __forceinline__ void widen_q_smem(uint32_t ah[8][4], const uint8_t* qs, int lane) {
  const uint32_t qb = smemu32(qs) + (uint32_t)((lane & 15) * QSTR + (lane >> 4) * 16);
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t x[4];
    ldm_x4(x, qb + (uint32_t)(kk * 32));
    ah[2 * kk][0] = CVLO(x[0]);
    ah[2 * kk][1] = CVLO(x[1]);
    ah[2 * kk][2] = CVHI(x[0]);
    ah[2 * kk][3] = CVHI(x[1]);
    ah[2 * kk + 1][0] = CVLO(x[2]);
    ah[2 * kk + 1][1] = CVLO(x[3]);
    ah[2 * kk + 1][2] = CVHI(x[2]);
    ah[2 * kk + 1][3] = CVHI(x[3]);
  }
}

// 16 fp8 -> 16 half, written to Vh row l at column c
__device__ __forceinline__ void cvt_store_v(__half* vh, int l, int c, uint4 raw) {
  unsigned short bs[8];  // defined access into the raw payload
  std::memcpy(bs, &raw, sizeof(bs));
  uint4 h0, h1;
  h0.x = fp8x2_to_h2(bs[0]);
  h0.y = fp8x2_to_h2(bs[1]);
  h0.z = fp8x2_to_h2(bs[2]);
  h0.w = fp8x2_to_h2(bs[3]);
  h1.x = fp8x2_to_h2(bs[4]);
  h1.y = fp8x2_to_h2(bs[5]);
  h1.z = fp8x2_to_h2(bs[6]);
  h1.w = fp8x2_to_h2(bs[7]);
  *reinterpret_cast<uint4*>(&vh[l * VSTRH + c]) = h0;
  *reinterpret_cast<uint4*>(&vh[l * VSTRH + c + 8]) = h1;
}

// Stage the 16 x (8*NJ) bf16 tile in shared memory with stmatrix (the PV
// accumulator fragment already has the stmatrix register layout), then write it
// back fully coalesced.  NJ < 16 means this CTA owns only a slice of the head
// dimension, so each staged row covers 8*NJ of the 128 output columns.
__device__ __forceinline__ void stm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2,
                                       uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr),
               "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
__device__ __forceinline__ void stm_x2(uint32_t addr, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared::cta.b16 [%0], {%1,%2};\n" ::"r"(addr),
               "r"(r0), "r"(r1));
}
template <bool MUL, int NJ, bool DRAIN64 = false, bool PAIR4 = false>
__device__ __forceinline__ void store_tile(__nv_bfloat16* ob, float acc[NJ][4], float ra, float rb,
                                           uint16_t* sb, int lane, bool active) {
  constexpr int OSTR = 8 * NJ + 8;  // halves per staged row
  const uint32_t base = smemu32(sb) + (uint32_t)(((lane & 7) * OSTR + (lane >> 3) * 8) * 2);
#pragma unroll
  for (int R = 0; R < 2; ++R) {
    const float sc = R ? rb : ra;
    if constexpr (NJ == 2) {
      float v0 = acc[0][2 * R], v1 = acc[0][2 * R + 1];
      float v2 = acc[1][2 * R], v3 = acc[1][2 * R + 1];
      if constexpr (MUL) {
        v0 *= sc;
        v1 *= sc;
        v2 *= sc;
        v3 *= sc;
      }
      stm_x2(base + (uint32_t)(R * 8 * OSTR * 2), pack_bf16x2(v0, v1), pack_bf16x2(v2, v3));
    } else {
#pragma unroll
      for (int g = 0; g < NJ / 4; ++g) {
        uint32_t rg[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const int j = 4 * g + m;
          float v0 = acc[j][2 * R], v1 = acc[j][2 * R + 1];
          if constexpr (MUL) {
            if constexpr (DRAIN64 && PAIR4) {
              const F32x2 vv = {v0, v1};
              const F32x2 ss = {sc, sc};
              const F32x2 zz = {0.f, 0.f};
              const F32x2 vr = fma_f32x2(vv, ss, zz);
              v0 = vr.x;
              v1 = vr.y;
            } else {
              v0 *= sc;
              v1 *= sc;
            }
          }
          rg[m] = pack_bf16x2(v0, v1);
        }
        stm_x4(base + (uint32_t)((R * 8 * OSTR + 32 * g) * 2), rg[0], rg[1], rg[2], rg[3]);
      }
    }
  }
  __syncwarp();
  if (!active) return;
  if constexpr (PAIR4) {
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int g = lane >> 2;
      const int row = 8 * it + (g >> 1) + 4 * (g & 1);
      const int col = (lane & 3) * 8;
      *reinterpret_cast<uint4*>(ob + row * DH + col) =
          *reinterpret_cast<const uint4*>(sb + row * OSTR + col);
    }
  } else if constexpr (DRAIN64) {
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int row = ((it >> 1) << 3) + (it & 1) + 2 * (lane >> 3);
      const int col = (lane & 7) * 4;
      *reinterpret_cast<uint2*>(ob + row * DH + col) =
          *reinterpret_cast<const uint2*>(sb + row * OSTR + col);
    }
  } else {
#pragma unroll
    for (int it = 0; it < NJ / 2; ++it) {
      const int lin = it * 256 + lane * 8;
      const int row = lin / (8 * NJ), col = lin % (8 * NJ);
      *reinterpret_cast<uint4*>(ob + row * DH + col) =
          *reinterpret_cast<const uint4*>(sb + row * OSTR + col);
    }
  }
}

// Full-D output using only an 8-row shared slab.  The two 8-head halves are
// staged and drained sequentially, allowing a dead per-warp Q slice to replace
// the CTA-wide KV/output overlay and its final block barrier.
__device__ __forceinline__ void store_tile_rows8(__nv_bfloat16* ob, float acc[16][4], float ra,
                                                 float rb, uint16_t* sb, int lane) {
  constexpr int OSTR = 136;
  const uint32_t base = smemu32(sb) + (uint32_t)(((lane & 7) * OSTR + (lane >> 3) * 8) * 2);
#pragma unroll
  for (int R = 0; R < 2; ++R) {
    const float sc = R ? rb : ra;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      uint32_t rg[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int j = 4 * g + m;
        rg[m] = pack_bf16x2(acc[j][2 * R] * sc, acc[j][2 * R + 1] * sc);
      }
      stm_x4(base + (uint32_t)(32 * g * 2), rg[0], rg[1], rg[2], rg[3]);
    }
    __syncwarp();
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int lin = it * 256 + lane * 8;
      const int row = lin >> 7, col = lin & 127;
      *reinterpret_cast<uint4*>(ob + (R * 8 + row) * DH + col) =
          *reinterpret_cast<const uint4*>(sb + row * OSTR + col);
    }
    // Only the first half needs to protect this slab from the next stmatrix.
    if (R == 0) __syncwarp();
  }
}

// Load a query tile once in fp8, then widen it to the native f16 MMA fragment
// layout.  GB300 lowers fp8 mma.sync to F2FP + HMMA + fix-up adds, so doing the
// widening explicitly and issuing m16n8k16.f16 directly is faster on the
// saturated B15 path.
__device__ __forceinline__ void load_q_raw(uint32_t qr[4][4], const uint8_t* __restrict__ q,
                                           int tok, int gid, int tig) {
  const uint8_t* qa0 = q + (size_t)tok * HQD + gid * DH + tig * 4;
  const uint8_t* qa1 = qa0 + 8 * DH;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    qr[kk][0] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32);
    qr[kk][1] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32);
    qr[kk][2] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32 + 16);
    qr[kk][3] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32 + 16);
  }
}

__device__ __forceinline__ void widen_q(uint32_t ah[8][4], const uint32_t qr[4][4]) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    ah[2 * kk][0] = CVLO(qr[kk][0]);
    ah[2 * kk][1] = CVLO(qr[kk][1]);
    ah[2 * kk][2] = CVHI(qr[kk][0]);
    ah[2 * kk][3] = CVHI(qr[kk][1]);
    ah[2 * kk + 1][0] = CVLO(qr[kk][2]);
    ah[2 * kk + 1][1] = CVLO(qr[kk][3]);
    ah[2 * kk + 1][2] = CVHI(qr[kk][2]);
    ah[2 * kk + 1][3] = CVHI(qr[kk][3]);
  }
}

__device__ __forceinline__ void qk_d128_split(float* d, const uint32_t ah[8][4],
                                              const uint32_t* kf) {
  // Two interleaved accumulator chains halve the serial MMA dependency depth
  // of one n-tile.  On the B=1 path a scheduler owns a single warp, so that
  // depth is exposed directly; the four extra FADDs are cheaper than it.
  float e[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    mma_f16_k16(d, ah[2 * s][0], ah[2 * s][1], ah[2 * s][2], ah[2 * s][3], CVLO(kf[2 * s]),
                CVHI(kf[2 * s]));
    mma_f16_k16(e, ah[2 * s + 1][0], ah[2 * s + 1][1], ah[2 * s + 1][2], ah[2 * s + 1][3],
                CVLO(kf[2 * s + 1]), CVHI(kf[2 * s + 1]));
  }
  d[0] += e[0];
  d[1] += e[1];
  d[2] += e[2];
  d[3] += e[3];
}

__device__ __forceinline__ void qk_d128(float* d, const uint32_t ah[8][4], const uint32_t* kf) {
  // One accumulator chain: the split-chain form needs four extra FADDs and a
  // second live quad, and the surrounding n-tiles already supply the ILP.
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    mma_f16_k16(d, ah[s][0], ah[s][1], ah[s][2], ah[s][3], CVLO(kf[s]), CVHI(kf[s]));
  }
}

// ---------------------------------------------------------------------------
// Path A: every request has kv_len <= 8 -> one n-tile, no online softmax
// ---------------------------------------------------------------------------
template <int WARPS, bool U2 = false, int KVCAP = 8, int DS = 1>
__global__ __launch_bounds__(32 * WARPS) void attn_tiny(const uint8_t* __restrict__ q,
                                                        const uint8_t* __restrict__ kvb,
                                                        const int* __restrict__ block_table,
                                                        const int* __restrict__ cu_q,
                                                        const int* __restrict__ seq_lens,
                                                        __nv_bfloat16* __restrict__ out, int btw) {
  __shared__ __half Vh[8 * VSTRH];
  __shared__ uint16_t Ost[WARPS * 16 * OSTRH];

  const int r = blockIdx.y;
  // one dependent round trip for all per-request metadata: issued together,
  // before any branch, so q and kv are only two round trips from kernel start.
  // U2 speculates uniform packing (q0 = 2r, qlen = kvlen = 2) and VERIFIES
  // it against cu_q/seq_lens (the loads issue together, one round trip);
  // on mismatch the loaded metadata drives the same masked body, so the
  // route stays exact for callers whose packing differs (e.g. extra
  // storage rows past cu_q[-1], or short requests under max_q_len).
  const int q0r = __ldg(cu_q + r);
  const int q1r = __ldg(cu_q + r + 1);
  const int kvr = __ldg(seq_lens + r);
  const bool uni = !U2 || (q0r == r * 2 && q1r == r * 2 + 2 && kvr == 2);
  const int q0 = (U2 && uni) ? r * 2 : q0r;
  const int kvlen = (U2 && uni) ? 2 : kvr;
  const int pg = __ldg(block_table + r * btw);
  const int qlen = (U2 && uni) ? 2 : (q1r - q0r);
  const int pbase = U2 ? 0 : blockIdx.x * WARPS;
  if (pbase >= qlen) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int dpart = DS == 1 ? 0 : (U2 ? (int)blockIdx.x : (int)blockIdx.z);
  const uint8_t* vbase = kvb + (size_t)pg * (PAGE * KVTOK) + DH;

  for (int idx = tid; idx < 64; idx += 32 * WARPS) {
    const int l = idx >> 3, c = (idx & 7) << 4;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (l < kvlen) raw = *reinterpret_cast<const uint4*>(vbase + (size_t)l * KVTOK + c);
    cvt_store_v(Vh, l, c, raw);
  }

  const int gid = lane >> 2, tig = lane & 3;
  const int p = U2 ? warp : (pbase + warp);
  const bool active = p < qlen;
  const int tok = q0 + (active ? p : 0);
  const int vis = kvlen - qlen + p + 1;

  const uint8_t* qa0 = q + (size_t)tok * HQD + gid * DH + tig * 4;
  const uint8_t* qa1 = qa0 + 8 * DH;
  uint32_t a[4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    a[kk][0] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32);
    a[kk][1] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32);
    a[kk][2] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32 + 16);
    a[kk][3] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32 + 16);
  }
  const uint8_t* kb = kvb + (size_t)pg * (PAGE * KVTOK) + (size_t)gid * KVTOK + tig * 4;
  uint32_t bq[4][2] = {};
  if constexpr (!U2) {
    if constexpr (KVCAP == 8) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        bq[kk][0] = *reinterpret_cast<const uint32_t*>(kb + kk * 32);
        bq[kk][1] = *reinterpret_cast<const uint32_t*>(kb + kk * 32 + 16);
      }
    } else if (gid < KVCAP) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        bq[kk][0] = *reinterpret_cast<const uint32_t*>(kb + kk * 32);
        bq[kk][1] = *reinterpret_cast<const uint32_t*>(kb + kk * 32 + 16);
      }
    }
  } else if (Q2_CAUSAL_PRUNE ? (gid < vis) : (gid < 2)) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      bq[kk][0] = *reinterpret_cast<const uint32_t*>(kb + kk * 32);
      bq[kk][1] = *reinterpret_cast<const uint32_t*>(kb + kk * 32 + 16);
    }
  }

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) mma_f8(c0, c1, c2, c3, a[kk], bq[kk]);

  const int l0 = tig * 2;
  float s0 = (l0 < vis) ? c0 * SCALE_LOG2E : -INFINITY;
  float s1 = (l0 + 1 < vis) ? c1 * SCALE_LOG2E : -INFINITY;
  float s2 = (l0 < vis) ? c2 * SCALE_LOG2E : -INFINITY;
  float s3 = (l0 + 1 < vis) ? c3 * SCALE_LOG2E : -INFINITY;
  float ma = fmaxf(s0, s1), mb = fmaxf(s2, s3);
  float p0, p1, p2, p3, ra, rb;
  if constexpr (KVCAP <= 4) {
    if (vis > 2) {
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
    }
    p0 = (l0 < vis) ? exp2f(s0 - ma) : 0.f;
    p1 = (l0 + 1 < vis) ? exp2f(s1 - ma) : 0.f;
    p2 = (l0 < vis) ? exp2f(s2 - mb) : 0.f;
    p3 = (l0 + 1 < vis) ? exp2f(s3 - mb) : 0.f;
    float la = p0 + p1, lb = p2 + p3;
    if (vis > 2) {
      la += __shfl_xor_sync(0xffffffffu, la, 1);
      lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    }
    ra = (l0 < vis) ? 1.0f / la : 0.f;
    rb = (l0 < vis) ? 1.0f / lb : 0.f;
  } else {
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
    p0 = exp2f(s0 - ma);
    p1 = exp2f(s1 - ma);
    p2 = exp2f(s2 - mb);
    p3 = exp2f(s3 - mb);
    float la = p0 + p1, lb = p2 + p3;
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    ra = 1.0f / la;
    rb = 1.0f / lb;
  }
  const uint32_t pa0 = pack_h2(p0 * ra, p1 * ra);
  const uint32_t pa1 = pack_h2(p2 * rb, p3 * rb);

  __syncthreads();

  // V is now fully staged CTA-wide.  Tail warps do not participate in any
  // later CTA barrier, so avoid their 16 PV MMAs and output staging work.
  if constexpr (!U2) {
    if (!active) return;
  }

  constexpr int NJ = 16 / DS;
  uint32_t vb[NJ];
  const uint32_t sb = smemu32(&Vh[(lane & 7) * VSTRH + ((lane >> 3) << 3)]);
  if constexpr (NJ == 2) {
    ldm_x2_trans(vb, sb + (uint32_t)(dpart * NJ * 16));
  } else {
#pragma unroll
    for (int g = 0; g < NJ / 4; ++g)
      ldm_x4_trans(&vb[g * 4], sb + (uint32_t)((g + dpart * (NJ / 4)) * 64));
  }

  float acc[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    acc[j][0] = 0.f;
    acc[j][1] = 0.f;
    acc[j][2] = 0.f;
    acc[j][3] = 0.f;
    mma_f16_k8(acc[j], pa0, pa1, vb[j]);
  }

  store_tile<false, NJ>(out + (size_t)tok * HQD + dpart * (DH / DS), acc, 1.f, 1.f,
                        Ost + warp * (16 * OSTRH), lane, active);
}

template <int WARPS, bool U2 = false, bool ASYNC_QK = false, int DS = 1, int KVCAP = 8,
          bool RFAST = false, bool BTW1 = false, bool RAWV = false, bool PB0 = false,
          bool LDMQ2 = false>
__global__ __launch_bounds__(32 * WARPS) void attn_tiny_async(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ kvb,
    const int* __restrict__ block_table, const int* __restrict__ cu_q,
    const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, int btw) {
  __shared__ __align__(16) uint8_t Vstage[RAWV ? (2 * (DH / DS)) : (8 * VSTRH * sizeof(__half))];
  __half* Vh = reinterpret_cast<__half*>(Vstage);
  __shared__ uint16_t Ost[WARPS * 16 * OSTRH];
  __shared__ __align__(128) uint8_t Qraw[WARPS * HQD];
  __shared__ __align__(128) uint8_t Kraw[8 * DH];

  // U2 takes the request from grid x (2^31-1 capacity): one launch covers any
  // batch size, so cu_q's absolute packed offsets never need host rebasing.
  const int r = (RFAST || U2) ? (int)blockIdx.x : (int)blockIdx.y;
  // one dependent round trip for all per-request metadata: issued together,
  // before any branch, so q and kv are only two round trips from kernel start.
  // U2 speculation verified against cu_q/seq_lens (loads issue together);
  // on mismatch the loaded metadata drives the same masked body.
  const int q0r = __ldg(cu_q + r);
  const int q1r = __ldg(cu_q + r + 1);
  const int kvr = ldv32(seq_lens + r);
  const bool uni = !U2 || (q0r == r * 2 && q1r == r * 2 + 2 && kvr == 2);
  const int q0 = (U2 && uni) ? r * 2 : q0r;
  const int kvlen = (U2 && uni) ? 2 : kvr;
  const int pg = (U2 && BTW1) ? __ldg(block_table + r) : ldv32(block_table + r * btw);
  const int qlen = (U2 && uni) ? 2 : (q1r - q0r);
  const int pbase = (U2 || PB0) ? 0
                                : (RFAST ? (DS == 2 ? ((int)gridDim.y - 1 - (int)blockIdx.y) * WARPS
                                                    : (int)blockIdx.y * WARPS)
                                         : (int)blockIdx.x * WARPS);
  if (pbase >= qlen) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int dpart = DS == 1 ? 0 : (int)blockIdx.z;
  const int gid = lane >> 2, tig = lane & 3;
  const int p = U2 ? warp : (pbase + warp);
  const bool active = p < qlen;
  const int tok = q0 + (active ? p : 0);
  const uint8_t* vbase = kvb + (size_t)pg * (PAGE * KVTOK) + DH;

  if constexpr (ASYNC_QK) {
    // threadIdx.x < 32*WARPS, so the token index and the 64-element row jump
    // of each step are compile-time: one invariant shared address plus a
    // constant offset replaces rebuilding head/ch/the swizzle every step.
    const uint32_t qs = smemu32(Qraw);
    constexpr int T = 32 * WARPS;
    const int ch = tid & 7, hlo = (tid >> 3) & 7;
    const uint32_t qdst = qs + (uint32_t)((tid >> 3) * DH + ((ch ^ hlo) << 4));
    const uint8_t* qsrc = q + (size_t)tid * 16;
#pragma unroll
    for (int u = 0; u < (PB0 ? KVCAP : 4); ++u) {
      const int off = (T * u) & 127;  // 0 or 64 (W=2), always 0 (W=4)
      const int qw = (T * u) >> 7;
      const int pp = pbase + qw;
      const int safe_p = (pp < qlen) ? pp : 0;
      cpasync16(qdst + (uint32_t)(qw * HQD + off * DH / 8),
                qsrc + (size_t)(q0 + safe_p) * HQD + off * 16);
    }
    // KVCAP is a compile-time bound from max_kv_len; within it the per-request
    // kv_len still bounds the transfer with cp.async src-size 0, which keeps
    // every thread issuing (and so the access coalesced) while moving only the
    // bytes a causal row can actually see.
    if (tid < (RFAST ? 64 : 8 * KVCAP)) {
      const int l = tid >> 3, ch = tid & 7;
      cpasync16z(smemu32(Kraw) + (uint32_t)(l * DH + ((ch ^ l) << 4)),
                 kvb + (size_t)pg * (PAGE * KVTOK) + (size_t)l * KVTOK + ch * 16,
                 (l < kvlen) ? 16 : 0);
    }
    if constexpr (RAWV) {
      constexpr int VCH = (DH / DS) / 16;
      if (tid < 2 * VCH) {
        const int l = tid / VCH, ch = tid - l * VCH;
        cpasync16(smemu32(Vstage) + (uint32_t)((l * VCH + ch) * 16),
                  vbase + (size_t)l * KVTOK + dpart * (DH / DS) + ch * 16);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }

  if constexpr (!RAWV) {
    for (int idx = tid; idx < 64; idx += 32 * WARPS) {
      const int l = idx >> 3, c = (idx & 7) << 4;
      uint4 raw = make_uint4(0u, 0u, 0u, 0u);
      if (l < kvlen) raw = *reinterpret_cast<const uint4*>(vbase + (size_t)l * KVTOK + c);
      cvt_store_v(Vh, l, c, raw);
    }
  }

  uint32_t a[4][4];
  uint32_t bq[4][2] = {};
  if constexpr (ASYNC_QK) {
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // Q/K/V publication is the only CTA-wide synchronization in this path.
    // Tail warps can leave immediately afterwards instead of executing a
    // complete QK/softmax sequence whose result would never be stored.
    if (!active) return;
    if constexpr (RFAST) {
      const uint32_t qrb = smemu32(Qraw) + (uint32_t)(warp * HQD + (lane & 15) * DH +
                                                      (((lane >> 4) ^ (lane & 7)) << 4));
      const uint32_t krb =
          smemu32(Kraw) + (uint32_t)((lane & 7) * DH + ((((lane >> 3) & 1) ^ (lane & 7)) << 4));
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        ldm_x4(a[kk], qrb ^ (uint32_t)(kk * 32));
        ldm_x2(bq[kk], krb ^ (uint32_t)(kk * 32));
      }
    } else {
      if constexpr (KVCAP >= 3 || LDMQ2) {
        const uint32_t qrb = smemu32(Qraw) + (uint32_t)(warp * HQD + (lane & 15) * DH +
                                                        (((lane >> 4) ^ (lane & 7)) << 4));
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          ldm_x4(a[kk], qrb ^ (uint32_t)(kk * 32));
          if (gid < KVCAP) {
            bq[kk][0] = *reinterpret_cast<const uint32_t*>(Kraw + gid * DH +
                                                           (((2 * kk) ^ gid) << 4) + tig * 4);
            bq[kk][1] = *reinterpret_cast<const uint32_t*>(Kraw + gid * DH +
                                                           (((2 * kk + 1) ^ gid) << 4) + tig * 4);
          }
        }
      } else {
        const uint8_t* qa0 = Qraw + warp * HQD + gid * DH;
        const uint8_t* qa1 = qa0 + 8 * DH;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int c0 = 2 * kk;
          a[kk][0] = *reinterpret_cast<const uint32_t*>(qa0 + ((c0 ^ (gid & 7)) << 4) + tig * 4);
          a[kk][1] = *reinterpret_cast<const uint32_t*>(qa1 + ((c0 ^ gid) << 4) + tig * 4);
          a[kk][2] =
              *reinterpret_cast<const uint32_t*>(qa0 + (((c0 + 1) ^ (gid & 7)) << 4) + tig * 4);
          a[kk][3] = *reinterpret_cast<const uint32_t*>(qa1 + (((c0 + 1) ^ gid) << 4) + tig * 4);
          if (gid < KVCAP) {
            bq[kk][0] = *reinterpret_cast<const uint32_t*>(Kraw + gid * DH +
                                                           (((2 * kk) ^ gid) << 4) + tig * 4);
            bq[kk][1] = *reinterpret_cast<const uint32_t*>(Kraw + gid * DH +
                                                           (((2 * kk + 1) ^ gid) << 4) + tig * 4);
          }
        }
      }
    }
  } else {
    const uint8_t* qa0 = q + (size_t)tok * HQD + gid * DH + tig * 4;
    const uint8_t* qa1 = qa0 + 8 * DH;
    const uint8_t* kb = kvb + (size_t)pg * (PAGE * KVTOK) + (size_t)gid * KVTOK + tig * 4;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      a[kk][0] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32);
      a[kk][1] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32);
      a[kk][2] = *reinterpret_cast<const uint32_t*>(qa0 + kk * 32 + 16);
      a[kk][3] = *reinterpret_cast<const uint32_t*>(qa1 + kk * 32 + 16);
      bq[kk][0] = *reinterpret_cast<const uint32_t*>(kb + kk * 32);
      bq[kk][1] = *reinterpret_cast<const uint32_t*>(kb + kk * 32 + 16);
    }
  }

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  if constexpr (RFAST) {
    // SM103 executes this tiny shape faster through native f16 MMA after
    // explicit widening than through its fp8 warp-MMA fix-up path.
    float d0[4] = {0.f, 0.f, 0.f, 0.f};
    float d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      mma_f16_k16(d0, CVLO(a[kk][0]), CVLO(a[kk][1]), CVHI(a[kk][0]), CVHI(a[kk][1]),
                  CVLO(bq[kk][0]), CVHI(bq[kk][0]));
      mma_f16_k16(d1, CVLO(a[kk][2]), CVLO(a[kk][3]), CVHI(a[kk][2]), CVHI(a[kk][3]),
                  CVLO(bq[kk][1]), CVHI(bq[kk][1]));
    }
    c0 = d0[0] + d1[0];
    c1 = d0[1] + d1[1];
    c2 = d0[2] + d1[2];
    c3 = d0[3] + d1[3];
  } else {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) mma_f8(c0, c1, c2, c3, a[kk], bq[kk]);
  }

  const int vis = kvlen - qlen + p + 1;  // == p + 1 on the verified-uniform U2 path
  const int l0 = tig * 2;
  float s0 = (l0 < vis) ? c0 * SCALE_LOG2E : -INFINITY;
  float s1 = (l0 + 1 < vis) ? c1 * SCALE_LOG2E : -INFINITY;
  float s2 = (l0 < vis) ? c2 * SCALE_LOG2E : -INFINITY;
  float s3 = (l0 + 1 < vis) ? c3 * SCALE_LOG2E : -INFINITY;
  float ma = fmaxf(s0, s1), mb = fmaxf(s2, s3);
  float p0, p1, p2, p3, ra, rb;
  if constexpr (KVCAP <= 4) {
    if (vis > 2) {
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
    }
    p0 = (l0 < vis) ? exp2f(s0 - ma) : 0.f;
    p1 = (l0 + 1 < vis) ? exp2f(s1 - ma) : 0.f;
    p2 = (l0 < vis) ? exp2f(s2 - mb) : 0.f;
    p3 = (l0 + 1 < vis) ? exp2f(s3 - mb) : 0.f;
    float la = p0 + p1, lb = p2 + p3;
    if (vis > 2) {
      la += __shfl_xor_sync(0xffffffffu, la, 1);
      lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    }
    ra = (l0 < vis) ? 1.0f / la : 0.f;
    rb = (l0 < vis) ? 1.0f / lb : 0.f;
  } else {
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
    p0 = exp2f(s0 - ma);
    p1 = exp2f(s1 - ma);
    p2 = exp2f(s2 - mb);
    p3 = exp2f(s3 - mb);
    float la = p0 + p1, lb = p2 + p3;
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    ra = 1.0f / la;
    rb = 1.0f / lb;
  }
  const uint32_t pa0 = pack_h2(p0 * ra, p1 * ra);
  const uint32_t pa1 = pack_h2(p2 * rb, p3 * rb);

  if constexpr (!ASYNC_QK) __syncthreads();

  // V is now fully staged CTA-wide.  Tail warps do not participate in any
  // later CTA barrier, so avoid their 16 PV MMAs and output staging work.
  if (!active) return;

  constexpr int NJ = 16 / DS;
  uint32_t vb[NJ];
  if constexpr (RAWV) {
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      if (tig == 0) {
        const int c = j * 8 + gid;
        const uint16_t x = (uint16_t)Vstage[c] | ((uint16_t)Vstage[DH / DS + c] << 8);
        vb[j] = fp8x2_to_h2(x);
      } else {
        vb[j] = 0;
      }
    }
  } else {
    const uint32_t sb = smemu32(&Vh[(lane & 7) * VSTRH + ((lane >> 3) << 3)]);
    if constexpr (NJ == 2) {
      ldm_x2_trans(vb, sb + (uint32_t)(dpart * NJ * 16));
    } else {
#pragma unroll
      for (int g = 0; g < NJ / 4; ++g)
        ldm_x4_trans(&vb[g * 4], sb + (uint32_t)((g + dpart * (NJ / 4)) * 64));
    }
  }

  float acc[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    acc[j][0] = 0.f;
    acc[j][1] = 0.f;
    acc[j][2] = 0.f;
    acc[j][3] = 0.f;
    mma_f16_k8(acc[j], pa0, pa1, vb[j]);
  }

  store_tile<false, NJ, false, (NJ == 4)>(out + (size_t)tok * HQD + dpart * (DH / DS), acc, 1.f,
                                          1.f, Ost + warp * (16 * OSTRH), lane, active);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Path B: general kv length, 32-wide kv tiles, fully async double-buffered KV.
// K and V both stay fp8 in shared memory: K feeds mma.m16n8k32.e4m3 through
// ldmatrix.b16, V feeds it through ldmatrix.m16n16.trans.b8.  That transposing
// 8-bit load hands back 4 kv rows {2i, 2i+1, 2i+8, 2i+9} per B register, a
// permutation of the contraction index -- applying the same permutation when
// packing P into the A fragment makes the product exact again.
// ---------------------------------------------------------------------------
#define VROW(j) (((j) & 1) + 2 * (((j) >> 2) & 3) + 8 * (((j) >> 1) & 1))
#define VSWZ(l) (((l) & 1) | ((((l) >> 3) & 1) << 1) | ((((l) >> 1) & 1) << 2))

// Fold the two shortest four-row tiles into otherwise short resident CTAs on
// B15.  Keeping this as a separate device function prevents its accumulator
// set from raising the register allocation of the long-query main path.
__device__ __noinline__ void attn_gen_tail_u128(const uint8_t* __restrict__ q,
                                                __nv_bfloat16* __restrict__ out, int q0, int pbase,
                                                uint8_t* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tig = lane & 3;
  const int p = pbase + warp;
  const int tok = q0 + p;
  const int vis = p + 1;

  // K/V buffer zero still holds the first 32 cache tokens.  Buffers one and
  // two are dead, and together form a conflict-padded per-warp Q/output slab.
  uint8_t* qtmp = smem + 32 * KSTR + warp * (16 * QSTR);
  stage_q(qtmp, q, tok, lane);
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncwarp();

  uint32_t ah[8][4];
  widen_q_smem(ah, qtmp, lane);
  const uint32_t kbase = smemu32(smem) + (uint32_t)((lane & 7) * KSTR + (lane >> 3) * 16);
  uint32_t kf[8];
  ldm_x4(kf, kbase);
  ldm_x4(kf + 4, kbase + 64);
  float dq[4] = {0.f, 0.f, 0.f, 0.f};
  qk_d128(dq, ah, kf);
  const int lc = tig * 2;
  float s0 = (lc < vis) ? dq[0] : -INFINITY;
  float s1 = (lc + 1 < vis) ? dq[1] : -INFINITY;
  float s2 = (lc < vis) ? dq[2] : -INFINITY;
  float s3 = (lc + 1 < vis) ? dq[3] : -INFINITY;
  float ma = fmaxf(s0, s1), mb = fmaxf(s2, s3);
  ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
  ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
  mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
  mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));

  const F32x2 sc2 = {SCALE_LOG2E, SCALE_LOG2E};
  const F32x2 xa = fma_f32x2({s0, s1}, sc2, {-ma * SCALE_LOG2E, -ma * SCALE_LOG2E});
  const F32x2 xb = fma_f32x2({s2, s3}, sc2, {-mb * SCALE_LOG2E, -mb * SCALE_LOG2E});
  const uint32_t pa0[4] = {ex2_h2(pack_h2(xa.x, xa.y)), ex2_h2(pack_h2(xb.x, xb.y)), 0u, 0u};
  const uint32_t ONE2 = 0x3c003c00u;
  float den[4] = {0.f, 0.f, 0.f, 0.f};
  mma_f16_k16(den, pa0[0], pa0[1], 0u, 0u, ONE2, ONE2);

  float acc[16][4];
  const uint32_t vcode =
      ((uint32_t)((lane & 16) + VROW(lane & 15)) << 7) | ((uint32_t)(lane & 7) << 4);
  const uint32_t vbase = smemu32(smem + 3 * 32 * KSTR);
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    uint32_t vb[4];
    ldm_x2_t8(vb, vbase + (vcode ^ ((uint32_t)jj << 4)));
    acc[2 * jj][0] = acc[2 * jj][1] = 0.f;
    acc[2 * jj][2] = acc[2 * jj][3] = 0.f;
    acc[2 * jj + 1][0] = acc[2 * jj + 1][1] = 0.f;
    acc[2 * jj + 1][2] = acc[2 * jj + 1][3] = 0.f;
    mma_f16_k16(acc[2 * jj], pa0[0], pa0[1], 0u, 0u, CVLO(vb[0]), CVHI(vb[0]));
    mma_f16_k16(acc[2 * jj + 1], pa0[0], pa0[1], 0u, 0u, CVLO(vb[1]), CVHI(vb[1]));
  }
  store_tile_rows8(out + (size_t)tok * HQD, acc, 1.0f / den[0], 1.0f / den[2],
                   reinterpret_cast<uint16_t*>(qtmp), lane);
}

template <int WARPS, int DS, int NB, bool U128 = false, bool LPT = false, bool FUSETAIL = false>
__global__ __launch_bounds__(32 * WARPS, WARPS == 4 ? 3 : 1) void attn_gen(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ kvb,
    const int* __restrict__ block_table, const int* __restrict__ cu_q,
    const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ out, int btw) {
  constexpr int NT = 32 * WARPS;
  constexpr int NCH = 256 / NT;
  constexpr int NJ = 16 / DS;  // n-tiles of the head dim owned here
  constexpr int NP = 8 / DS;   // 16-wide d blocks owned here
  constexpr int QPITCH = (U128 && DS == 1) ? 2176 : HQD;
  const int dh = (DS == 1) ? 0 : (LPT ? (int)blockIdx.x % DS : (int)blockIdx.z);
  // one block: 2x K tile (padded) + 2x V tile (xor-swizzled); reused after the
  // loop as the per-warp bf16 output staging area (needs WARPS*16*OSTRH*2 <= size)
  __shared__ __align__(16) uint8_t smem[NB * 32 * KSTR + NB * 32 * DH + WARPS * QPITCH];
  static_assert(WARPS * 16 * (8 * NJ + 8) * sizeof(uint16_t) <= sizeof(smem),
                "KV/output shared-memory overlay is too small");
#define KBUF(b) (smem + (b) * (32 * KSTR))
#define VBUF(b) (smem + NB * 32 * KSTR + (b) * (32 * DH))
#define QBUF (smem + NB * 32 * KSTR + NB * 32 * DH)

  // Uniform q128 batches are causally imbalanced: later query blocks execute
  // more KV tiles.  Flatten and transpose the physical grid so adjacent CTAs
  // cover different requests at the same (descending) query block, placing
  // the longest work first and avoiding a long-query tail on one request.
  const int r = LPT ? (int)blockIdx.x / DS : (int)blockIdx.y;
  const int pblock = FUSETAIL ? (int)gridDim.y + 1 - (int)blockIdx.y
                     : LPT    ? (int)gridDim.y - 1 - (int)blockIdx.y
                              : (int)blockIdx.x;
  // U128 speculation verified against cu_q/seq_lens (loads issue together);
  // on mismatch the loaded metadata drives the same masked body.
  const int q0r = __ldg(cu_q + r);
  const int q1r = __ldg(cu_q + r + 1);
  const int kvr = __ldg(seq_lens + r);
  const bool uni = !U128 || (q0r == r * 128 && q1r == r * 128 + 128 && kvr == 128);
  const int q0 = (U128 && uni) ? r * 128 : q0r;
  const int qlen = (U128 && uni) ? 128 : (q1r - q0r);
  const int kvlen = (U128 && uni) ? 128 : kvr;
  const int pg0 = U128 ? ldu32(block_table + r * btw) : __ldg(block_table + r * btw);
  const int pbase = pblock * WARPS;
  if (pbase >= qlen) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2, tig = lane & 3;
  const int p = pbase + warp;
  const bool active = (p < qlen);
  const int tok = q0 + (active ? p : 0);
  const int vis = kvlen - qlen + p + 1;

  const int pmax = (U128 && uni) ? (pbase + WARPS - 1) : min(pbase + WARPS - 1, qlen - 1);
  const int vmax = kvlen - qlen + pmax + 1;
  const int ntiles = (vmax + 31) >> 5;

  uint32_t qr[4][4], ah[8][4];

  float acc[NJ][4];
#pragma unroll
  for (int j = 0; j < NJ; ++j) {
    acc[j][0] = 0.f;
    acc[j][1] = 0.f;
    acc[j][2] = 0.f;
    acc[j][3] = 0.f;
  }
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;

  const int wrow = tid >> 3, wcol = tid & 7;  // staging: 16B chunk of a kv row
  // Fuse the V row offset and XOR-column permutation.  Their bit fields are
  // disjoint, leaving one invariant address code live across the tile loop.
  const uint32_t vcode =
      ((uint32_t)((lane & 16) + VROW(lane & 15)) << 7) | ((uint32_t)(lane & 7) << 4);

#define ISSUE(t)                                                                                \
  {                                                                                             \
    const int pg_ = U128 ? pg0 : (((t) >> 2) ? block_table[r * btw + ((t) >> 2)] : pg0);        \
    const uint8_t* tb_ = kvb + (size_t)pg_ * (PAGE * KVTOK) + (size_t)(((t) & 3) * 32) * KVTOK; \
    const int lb_ = (t) * 32;                                                                   \
    const int buf_ = (U128 && NB == 4) ? (t) : ((t) % NB);                                      \
    _Pragma("unroll") for (int u = 0; u < NCH; ++u) {                                           \
      const int l_ = wrow + u * (NT >> 3);                                                      \
      const int nb_ = (lb_ + l_ < vmax) ? 16 : 0;                                               \
      cpasync16z(smemu32(KBUF(buf_)) + (uint32_t)(l_ * KSTR + wcol * 16),                       \
                 tb_ + (size_t)l_ * KVTOK + wcol * 16, nb_);                                    \
      cpasync16z(smemu32(VBUF(buf_)) + (uint32_t)(l_ * DH + ((wcol ^ VSWZ(l_)) << 4)),          \
                 tb_ + (size_t)l_ * KVTOK + DH + wcol * 16, nb_);                               \
    }                                                                                           \
  }

  // Coalesced 16-byte query staging removes the strided lane-wise global loads
  // (one 2 KiB swizzled tile per warp).  It is committed ahead of KV tile 0,
  // so the existing wait_group retires Q and the first KV tile together.
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int lin = tid + u * NT;
    const int qw = lin >> 7, tt = lin & 127;
    const int head = tt >> 3, ch = tt & 7;
    const uint32_t qdst =
        smemu32(QBUF) + (uint32_t)(qw * QPITCH + head * DH + ((ch ^ (head & 7)) << 4));
    if constexpr (U128) {
      if (uni) {
        cpasync16(qdst, q + (size_t)(q0 + pbase + qw) * HQD + tt * 16);
      } else {
        // Failed speculation: rows past the request's real qlen must not be
        // fetched (they may lie outside the packed q rows), exactly like the
        // generic path below.
        const bool valid = (pbase + qw < qlen);
        const int safe_qw = valid ? qw : 0;
        cpasync16z(qdst, q + (size_t)(q0 + pbase + safe_qw) * HQD + tt * 16, valid ? 16 : 0);
      }
    } else {
      const bool valid = (pbase + qw < qlen);
      const int safe_qw = valid ? qw : 0;
      cpasync16z(qdst, q + (size_t)(q0 + pbase + safe_qw) * HQD + tt * 16, valid ? 16 : 0);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);

  // 3 buffers, 2 tiles of prefetch in flight: one in-flight tile is not enough
  // to cover the global latency when the grid is too small to give a scheduler
  // more than one warp.  An empty commit_group keeps the group count constant
  // so `wait_group 1` always retires exactly the tile about to be consumed.
  ISSUE(0)
  asm volatile("cp.async.commit_group;\n" ::);
  if (ntiles > 1) ISSUE(1)
  // Keep an empty second group for one-tile CTAs so wait_group 1 has the same
  // pipeline meaning without fetching a KV tile that can never be consumed.
  asm volatile("cp.async.commit_group;\n" ::);
  if constexpr (NB >= 4) {
    if (ntiles > 2) ISSUE(2)
    asm volatile("cp.async.commit_group;\n" ::);
  }

  for (int t = 0; t < ntiles; ++t) {
    const int buf = (U128 && NB == 4) ? t : (t % NB);
    const int lbase = t * 32;
    if constexpr (NB >= 4) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 1;\n" ::);
    }
    __syncthreads();
    if (t == 0) {
      const int l8 = lane & 7, h8 = (lane >> 3) & 3;
      const uint32_t qbase = smemu32(QBUF) + (uint32_t)(warp * QPITCH + (l8 + 8 * (h8 & 1)) * DH);
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        ldm_x4(qr[st], qbase + (uint32_t)((((2 * st + (h8 >> 1)) ^ l8)) << 4));
      }
      widen_q(ah, qr);
      // Uniform q128 has exactly four tiles and four buffers.  Issue tile 3
      // once and keep the steady-state loop free of modulus/refill control.
      if constexpr (U128 && NB == 4) {
        if (ntiles == 4) ISSUE(3)
      }
    }
    if constexpr (!(U128 && NB == 4)) {
      if (t + NB - 1 < ntiles) ISSUE(t + NB - 1)
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // U128 keeps this gate compile-time (the hot loop stays branch-free).
    // On failed speculation inactive warps compute garbage here, but every
    // cross-lane op is warp-local and their store below is masked, so the
    // waste is bounded by the tail warps of one CTA per request.
    if (U128 || (active && lbase < vis)) {
      const uint32_t kbase = smemu32(KBUF(buf)) + (uint32_t)((lane & 7) * KSTR + (lane >> 3) * 16);
      const bool full = (lbase + 32 <= vis);
      float s[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t kf[8];
        ldm_x4(kf, kbase + (uint32_t)(nt * 8 * KSTR));
        ldm_x4(kf + 4, kbase + (uint32_t)(nt * 8 * KSTR + 64));
        float dq[4] = {0.f, 0.f, 0.f, 0.f};
        qk_d128(dq, ah, kf);
        const float d0 = dq[0], d1 = dq[1], d2 = dq[2], d3 = dq[3];
        if (full) {
          s[nt][0] = d0;
          s[nt][1] = d1;
          s[nt][2] = d2;
          s[nt][3] = d3;
        } else {
          const int lc = lbase + nt * 8 + tig * 2;
          s[nt][0] = (lc < vis) ? d0 : -INFINITY;
          s[nt][1] = (lc + 1 < vis) ? d1 : -INFINITY;
          s[nt][2] = (lc < vis) ? d2 : -INFINITY;
          s[nt][3] = (lc + 1 < vis) ? d3 : -INFINITY;
        }
      }
      float na = -INFINITY, nb = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        na = fmaxf(na, fmaxf(s[nt][0], s[nt][1]));
        nb = fmaxf(nb, fmaxf(s[nt][2], s[nt][3]));
      }
      // Softmax is invariant to the reference, so instead of tracking the exact
      // running max we keep a reference two binades above it and rebase only
      // when a tile gets within one binade of the reference.  With
      // a stable reference the 64-register accumulator rescale is skipped on
      // nearly every tile, which is the largest block of work left per tile.
      if (t == 0) {
        // la/lb and every accumulator are exactly zero on the first tile, so
        // establish the lazy reference directly instead of multiplying all
        // accumulator registers by exp2(-inf).
        na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 1));
        na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 2));
        nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 1));
        nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 2));
        ma = na + (U128 ? 2.0f * RMARGIN : RMARGIN);
        mb = nb + (U128 ? 2.0f * RMARGIN : RMARGIN);
      } else {
        // A lane-local maximum is sufficient for the vote: if the tile-wide
        // maximum crosses the threshold, the lane holding it votes true.
        if (__any_sync(0xffffffffu, (na > ma - RTHRESH) || (nb > mb - RTHRESH))) {
          na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 1));
          na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 2));
          nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 1));
          nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 2));
          const bool reba = (na > ma - RTHRESH), rebb = (nb > mb - RTHRESH);
          const float rma = reba ? (na + RMARGIN) : ma;
          const float rmb = rebb ? (nb + RMARGIN) : mb;
          const float sca = exp2f((ma - rma) * SCALE_LOG2E);
          const float scb = exp2f((mb - rmb) * SCALE_LOG2E);
          ma = rma;
          mb = rmb;
          la *= sca;
          lb *= scb;
#pragma unroll
          for (int j = 0; j < NJ; ++j) {
            acc[j][0] *= sca;
            acc[j][1] *= sca;
            acc[j][2] *= scb;
            acc[j][3] *= scb;
          }
        }
      }
      // P remains f16, so no e4m3 binade bias is needed.
      const float oa = -ma * SCALE_LOG2E, ob2 = -mb * SCALE_LOG2E;
      const F32x2 sc2 = {SCALE_LOG2E, SCALE_LOG2E};
      const F32x2 ba2 = {oa, oa}, bb2 = {ob2, ob2};
      uint32_t pa[4][2];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const F32x2 xa = fma_f32x2({s[nt][0], s[nt][1]}, sc2, ba2);
        const F32x2 xb = fma_f32x2({s[nt][2], s[nt][3]}, sc2, bb2);
        pa[nt][0] = ex2_h2(pack_h2(xa.x, xa.y));
        pa[nt][1] = ex2_h2(pack_h2(xb.x, xb.y));
      }
      const uint32_t pa0[4] = {pa[0][0], pa[0][1], pa[1][0], pa[1][1]};
      const uint32_t pa1[4] = {pa[2][0], pa[2][1], pa[3][0], pa[3][1]};
      const uint32_t ONE2 = 0x3c003c00u;
      float z[4] = {0.f, 0.f, 0.f, 0.f};
      mma_f16_k16(z, pa0[0], pa0[1], pa0[2], pa0[3], ONE2, ONE2);
      mma_f16_k16(z, pa1[0], pa1[1], pa1[2], pa1[3], ONE2, ONE2);
      la += z[0];
      lb += z[2];
#pragma unroll
      for (int jj = 0; jj < NP; ++jj) {
        uint32_t vb[4];
        ldm_x2_t8(vb, smemu32(VBUF(buf)) + (vcode ^ ((uint32_t)(dh * NP + jj) << 4)));
        mma_f16_k16(acc[2 * jj], pa0[0], pa0[1], pa0[2], pa0[3], CVLO(vb[0]), CVHI(vb[0]));
        mma_f16_k16(acc[2 * jj + 1], pa0[0], pa0[1], pa0[2], pa0[3], CVLO(vb[1]), CVHI(vb[1]));
        mma_f16_k16(acc[2 * jj], pa1[0], pa1[1], pa1[2], pa1[3], CVLO(vb[2]), CVHI(vb[2]));
        mma_f16_k16(acc[2 * jj + 1], pa1[0], pa1[1], pa1[2], pa1[3], CVLO(vb[3]), CVHI(vb[3]));
      }
    }
  }
  // Uniform q128 has consumed all four real groups; only empty bookkeeping
  // groups remain, so no terminal wait is needed before output staging.
  if constexpr (!(U128 && NB == 4)) {
    asm volatile("cp.async.wait_group 0;\n" ::);
  }
#undef ISSUE
#undef KBUF
#undef VBUF
#undef QBUF
  if constexpr (U128 && DS == 1) {
    // On failed U128 speculation a tail warp past the real qlen holds
    // tok == q0; its store would race the CTA that owns the request's first
    // row.  The store is warp-wide (store_tile_rows8 only uses __syncwarp),
    // so the whole inactive warp can skip it.
    if (active) {
      store_tile_rows8(
          out + (size_t)tok * HQD, acc, 1.0f / la, 1.0f / lb,
          reinterpret_cast<uint16_t*>(smem + NB * 32 * KSTR + NB * 32 * DH + warp * QPITCH), lane);
    }
  } else {
    __syncthreads();  // all warps and async copies are done with KV buffers
    store_tile<true, NJ>(out + (size_t)tok * HQD + dh * (8 * NJ), acc, 1.0f / la, 1.0f / lb,
                         reinterpret_cast<uint16_t*>(smem) + warp * (16 * (8 * NJ + 8)), lane,
                         active);
  }
  if constexpr (FUSETAIL) {
    if (pblock <= 3) attn_gen_tail_u128(q, out, q0, (pblock - 2) * WARPS, smem);
  }
}

// A warp-owned row against a full 128-token K/V page already resident in
// shared memory.  The output fragment is written directly: four neighboring
// lanes form each naturally aligned 16-byte sector, avoiding a CTA-wide
// output-staging barrier while the resident page is still live.
__device__ __forceinline__ void attn_persist_row_u128(const uint8_t* __restrict__ q,
                                                      const uint8_t* ks, const uint8_t* vs,
                                                      __nv_bfloat16* __restrict__ out, int tok,
                                                      int vis, int lane) {
  const int gid = lane >> 2, tig = lane & 3;
  uint32_t qr[4][4], ah[8][4];
  load_q_raw(qr, q, tok, gid, tig);
  widen_q(ah, qr);

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    acc[j][0] = 0.f;
    acc[j][1] = 0.f;
    acc[j][2] = 0.f;
    acc[j][3] = 0.f;
  }
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  const int vrow = (lane & 16) + VROW(lane & 15);
  const uint32_t vswz = (uint32_t)VSWZ(vrow);
  const int ntiles = (vis + 31) >> 5;

#pragma unroll
  for (int t = 0; t < 4; ++t) {
    if (t >= ntiles) break;
    const int lbase = t * 32;
    const uint32_t kbase = smemu32(ks) + (uint32_t)((lbase + (lane & 7)) * KSTR + (lane >> 3) * 16);
    const uint32_t vbase = smemu32(vs) + (uint32_t)((lbase + vrow) * DH);
    const bool full = (lbase + 32 <= vis);
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      uint32_t kf[8];
      ldm_x4(kf, kbase + (uint32_t)(nt * 8 * KSTR));
      ldm_x4(kf + 4, kbase + (uint32_t)(nt * 8 * KSTR + 64));
      float dq[4] = {0.f, 0.f, 0.f, 0.f};
      qk_d128(dq, ah, kf);
      if (full) {
        s[nt][0] = dq[0];
        s[nt][1] = dq[1];
        s[nt][2] = dq[2];
        s[nt][3] = dq[3];
      } else {
        const int lc = lbase + nt * 8 + tig * 2;
        s[nt][0] = (lc < vis) ? dq[0] : -INFINITY;
        s[nt][1] = (lc + 1 < vis) ? dq[1] : -INFINITY;
        s[nt][2] = (lc < vis) ? dq[2] : -INFINITY;
        s[nt][3] = (lc + 1 < vis) ? dq[3] : -INFINITY;
      }
    }
    float na = -INFINITY, nb = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      na = fmaxf(na, fmaxf(s[nt][0], s[nt][1]));
      nb = fmaxf(nb, fmaxf(s[nt][2], s[nt][3]));
    }
    na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 1));
    na = fmaxf(na, __shfl_xor_sync(0xffffffffu, na, 2));
    nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 1));
    nb = fmaxf(nb, __shfl_xor_sync(0xffffffffu, nb, 2));
    if (t == 0) {
      ma = na + RMARGIN;
      mb = nb + RMARGIN;
    } else {
      const bool reba = (na > ma - RTHRESH), rebb = (nb > mb - RTHRESH);
      if (__any_sync(0xffffffffu, reba || rebb)) {
        const float rma = reba ? (na + RMARGIN) : ma;
        const float rmb = rebb ? (nb + RMARGIN) : mb;
        const float sca = exp2f((ma - rma) * SCALE_LOG2E);
        const float scb = exp2f((mb - rmb) * SCALE_LOG2E);
        ma = rma;
        mb = rmb;
        la *= sca;
        lb *= scb;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          acc[j][0] *= sca;
          acc[j][1] *= sca;
          acc[j][2] *= scb;
          acc[j][3] *= scb;
        }
      }
    }
    const float oa = -ma * SCALE_LOG2E, ob = -mb * SCALE_LOG2E;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      s[nt][0] = exp2f(fmaf(s[nt][0], SCALE_LOG2E, oa));
      s[nt][1] = exp2f(fmaf(s[nt][1], SCALE_LOG2E, oa));
      s[nt][2] = exp2f(fmaf(s[nt][2], SCALE_LOG2E, ob));
      s[nt][3] = exp2f(fmaf(s[nt][3], SCALE_LOG2E, ob));
    }
    uint32_t pa0[4], pa1[4];
    pa0[0] = pack_h2(s[0][0], s[0][1]);
    pa0[1] = pack_h2(s[0][2], s[0][3]);
    pa0[2] = pack_h2(s[1][0], s[1][1]);
    pa0[3] = pack_h2(s[1][2], s[1][3]);
    pa1[0] = pack_h2(s[2][0], s[2][1]);
    pa1[1] = pack_h2(s[2][2], s[2][3]);
    pa1[2] = pack_h2(s[3][0], s[3][1]);
    pa1[3] = pack_h2(s[3][2], s[3][3]);
    const uint32_t ONE2 = 0x3c003c00u;
    float z[4] = {0.f, 0.f, 0.f, 0.f};
    mma_f16_k16(z, pa0[0], pa0[1], pa0[2], pa0[3], ONE2, ONE2);
    mma_f16_k16(z, pa1[0], pa1[1], pa1[2], pa1[3], ONE2, ONE2);
    la += z[0];
    lb += z[2];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      uint32_t vb[4];
      ldm_x2_t8(vb, vbase + (((uint32_t)jj ^ vswz) << 4));
      mma_f16_k16(acc[2 * jj], pa0[0], pa0[1], pa0[2], pa0[3], CVLO(vb[0]), CVHI(vb[0]));
      mma_f16_k16(acc[2 * jj], pa1[0], pa1[1], pa1[2], pa1[3], CVLO(vb[2]), CVHI(vb[2]));
      mma_f16_k16(acc[2 * jj + 1], pa0[0], pa0[1], pa0[2], pa0[3], CVLO(vb[1]), CVHI(vb[1]));
      mma_f16_k16(acc[2 * jj + 1], pa1[0], pa1[1], pa1[2], pa1[3], CVLO(vb[3]), CVHI(vb[3]));
    }
  }

  const float ra = 1.0f / la, rb = 1.0f / lb;
  const int i8 = lane >> 2, u = lane & 3;
  uint32_t* op = reinterpret_cast<uint32_t*>(out + (size_t)tok * HQD);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    op[i8 * 64 + 4 * j + u] = pack_bf16x2(acc[j][0] * ra, acc[j][1] * ra);
    op[(i8 + 8) * 64 + 4 * j + u] = pack_bf16x2(acc[j][2] * rb, acc[j][3] * rb);
  }
}

// Uniform q128 request-persistent path.  Each CTA stages one full interleaved
// K/V page once, then its eight independent warps consume a serpentine,
// workload-balanced subset of query rows.
template <int CPR>
__global__ __launch_bounds__(256, 1) void attn_persist_u128(const uint8_t* __restrict__ q,
                                                            const uint8_t* __restrict__ kvb,
                                                            const int* __restrict__ block_table,
                                                            __nv_bfloat16* __restrict__ out,
                                                            int btw) {
  __shared__ __align__(16) uint8_t smem[128 * KSTR + 128 * DH];
  uint8_t* ks = smem;
  uint8_t* vs = smem + 128 * KSTR;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r = blockIdx.y, slot = blockIdx.x;
  const int pg = __ldg(block_table + r * btw);
  const uint8_t* src = kvb + (size_t)pg * (PAGE * KVTOK);
  const int wrow = tid >> 3, wcol = tid & 7;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int l = wrow + u * 32;
    cpasync16(smemu32(ks) + (uint32_t)(l * KSTR + wcol * 16), src + (size_t)l * KVTOK + wcol * 16);
    cpasync16(smemu32(vs) + (uint32_t)(l * DH + ((wcol ^ VSWZ(l)) << 4)),
              src + (size_t)l * KVTOK + DH + wcol * 16);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  constexpr int NJOB = (128 + CPR - 1) / CPR;
#pragma unroll
  for (int j = warp; j < NJOB; j += 8) {
    const int rank = j * CPR + ((j & 1) ? (CPR - 1 - slot) : slot);
    if (rank < 128) {
      const int p = 127 - rank;
      attn_persist_row_u128(q, ks, vs, out, r * 128 + p, p + 1, lane);
    }
  }
}

// ---------------------------------------------------------------------------
// Path C: small grid -- one query token per CTA, four warps.
// Warp-per-token leaves a single warp on a scheduler once the grid is smaller
// than the SM count, so the token's whole kv range sits on one warp's critical
// path.  Here the four warps split the kv range for Q*K^T (phase A, writing P
// to shared memory) and then split the head dimension for P*V (phase B): the
// total MMA count is unchanged but the per-warp path is quartered.  Phase B's
// accumulator is only 4 n-tiles wide, so nothing but the 16 row maxima and row
// sums has to cross warps -- no accumulator merge.
// ---------------------------------------------------------------------------
#define PSTRH 136  // halves per P row: 128 kv + 8 pad (16B bank rotation)

template <bool ONEROUND, bool U128>
__global__ __launch_bounds__(128, 1) void attn_split(const uint8_t* __restrict__ q,
                                                     const uint8_t* __restrict__ kvb,
                                                     const int* __restrict__ block_table,
                                                     const int* __restrict__ cu_q,
                                                     const int* __restrict__ seq_lens,
                                                     __nv_bfloat16* __restrict__ out, int btw) {
  __shared__ __align__(16) uint8_t smem[4 * 32 * KSTR + 4 * 32 * DH];
  __shared__ __align__(16) uint16_t Pb[16 * PSTRH];
  // Head-major partials: a warp's four per-chunk entries for one head group
  // are contiguous, so the cross-warp fold is two 128-bit shared loads instead
  // of eight scalar ones.
  __shared__ __align__(16) float wmax[16][4];
  __shared__ __align__(16) float wsum[16][4];
  __shared__ __align__(16) uint8_t Qs[4 * 16 * QSTR];
#define SKBUF(b) (smem + (b) * (32 * KSTR))
#define SVBUF(b) (smem + 4 * (32 * KSTR) + (b) * (32 * DH))

  const int r = U128 ? 0 : (int)blockIdx.y;
  // U128 speculation verified against cu_q/seq_lens; on mismatch the loaded
  // metadata drives the same masked body (q0 == cu_q[0] == 0 by contract).
  const int q1r = __ldg(cu_q + r + 1);
  const int kvr = __ldg(seq_lens + r);
  const bool uni = !U128 || (q1r == 128 && kvr == 128);
  const int q0 = U128 ? 0 : cu_q[r];
  const int qlen = (U128 && uni) ? 128 : (q1r - q0);
  const int kvlen = (U128 && uni) ? 128 : kvr;
  // The grid is one CTA per query token and never exceeds the SM count, so the
  // kernel ends with token 127's CTA; launch it first instead of last.
  const int p = U128 ? (127 - (int)blockIdx.x) : (int)blockIdx.x;
  if (p >= qlen) return;

  const int tok = q0 + p;
  const int vis = kvlen - qlen + p + 1;
  const int nrounds = ONEROUND ? 1 : ((vis + 127) >> 7);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2, tig = lane & 3;

  // The four score warps consume the same query tile.  Copy its 128 aligned
  // segments once CTA-wide; each warp widens the shared read-only tile below.
  cpasync16(smemu32(Qs) + (uint32_t)((tid >> 3) * QSTR + (tid & 7) * 16),
            q + (size_t)tok * HQD + tid * 16);
  asm volatile("cp.async.commit_group;\n" ::);

  // Resolve the page-table route while the independent Q transfer is in flight.
  const int pg0 = U128 ? ldu32(block_table) : ldv32(block_table + r * btw);

  const int wrow = tid >> 3, wcol = tid & 7;
  const int vrow = (lane & 16) + VROW(lane & 15);
  const uint32_t vswz = (uint32_t)VSWZ(vrow);
  const uint32_t kbase = smemu32(SKBUF(warp)) + (uint32_t)((lane & 7) * KSTR + (lane >> 3) * 16);
  const uint32_t pbase = smemu32(Pb) + (uint32_t)(((lane & 15) * PSTRH + (lane >> 4) * 8) * 2);

  float acc[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    acc[j][0] = 0.f;
    acc[j][1] = 0.f;
    acc[j][2] = 0.f;
    acc[j][3] = 0.f;
  }
  float mg = -INFINITY, mg8 = -INFINITY, lg = 0.f, lg8 = 0.f;
  uint32_t ah[8][4];

  // Stage K and V as separate async-copy groups.  QK can begin as soon as K
  // is resident while the larger, later-consumed V transfer remains in flight.
#define STAGE4_K(rd)                                                                               \
  {                                                                                                \
    _Pragma("unroll") for (int j_ = 0; j_ < 4; ++j_) {                                             \
      const int t_ = (rd) * 4 + j_;                                                                \
      const int lb_ = t_ * 32;                                                                     \
      if (lb_ < vis) {                                                                             \
        const int pi_ = t_ >> 2;                                                                   \
        const int pg_ = (pi_ == 0) ? pg0 : block_table[r * btw + pi_];                             \
        const uint8_t* tb_ = kvb + (size_t)pg_ * (PAGE * KVTOK) + (size_t)((t_ & 3) * 32) * KVTOK; \
        _Pragma("unroll") for (int u = 0; u < 2; ++u) {                                            \
          const int l_ = wrow + u * 16;                                                            \
          const int nb_ = (lb_ + l_ < vis) ? 16 : 0;                                               \
          cpasync16z(smemu32(SKBUF(j_)) + (uint32_t)(l_ * KSTR + wcol * 16),                       \
                     tb_ + (size_t)l_ * KVTOK + wcol * 16, nb_);                                   \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
    asm volatile("cp.async.commit_group;\n" ::);                                                   \
  }
#define STAGE4_V(rd)                                                                               \
  {                                                                                                \
    _Pragma("unroll") for (int j_ = 0; j_ < 4; ++j_) {                                             \
      const int t_ = (rd) * 4 + j_;                                                                \
      const int lb_ = t_ * 32;                                                                     \
      if (lb_ < vis) {                                                                             \
        const int pi_ = t_ >> 2;                                                                   \
        const int pg_ = (pi_ == 0) ? pg0 : block_table[r * btw + pi_];                             \
        const uint8_t* tb_ = kvb + (size_t)pg_ * (PAGE * KVTOK) + (size_t)((t_ & 3) * 32) * KVTOK; \
        _Pragma("unroll") for (int u = 0; u < 2; ++u) {                                            \
          const int l_ = wrow + u * 16;                                                            \
          const int nb_ = (lb_ + l_ < vis) ? 16 : 0;                                               \
          cpasync16z(smemu32(SVBUF(j_)) + (uint32_t)(l_ * DH + ((wcol ^ VSWZ(l_)) << 4)),          \
                     tb_ + (size_t)l_ * KVTOK + DH + wcol * 16, nb_);                              \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
    asm volatile("cp.async.commit_group;\n" ::);                                                   \
  }

  if constexpr (U128) {
    // Each warp publishes only the 32-key tile it consumes.  This makes K
    // publication warp-local; V remains in the following async group until
    // the cooperative PV phase needs it.
    const int lbw = warp * 32;
    const int l0 = lane >> 3, c0 = lane & 7;
    const int krem = vis - lbw - l0;
    const uint32_t kdst = smemu32(SKBUF(warp)) + (uint32_t)(l0 * KSTR + c0 * 16);
    const uint8_t* ksrc = kvb + (size_t)pg0 * (PAGE * KVTOK) + (size_t)(lbw + l0) * KVTOK + c0 * 16;
    if (lbw < vis) {
#pragma unroll
      // A lane's row advances by a constant 4 each step, so both addresses and
      // the visibility bound are strength-reduced to one invariant plus a
      // compile-time offset instead of being rebuilt from ix every iteration.
      for (int u = 0; u < 8; ++u) {
        cpasync16z(kdst + (uint32_t)(u * (4 * KSTR)), ksrc + (size_t)(u * 4) * KVTOK,
                   (4 * u < krem) ? 16 : 0);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    if (lbw < vis) {
      // VSWZ's only u-dependent bit is bit 1, which lands on bit 5 of the
      // shared address and is disjoint from the u*4*DH row stride.
      const uint32_t vdst = smemu32(SVBUF(warp)) + (uint32_t)(l0 * DH + ((c0 ^ VSWZ(l0)) << 4));
      const uint8_t* vsrc =
          kvb + (size_t)pg0 * (PAGE * KVTOK) + (size_t)(lbw + l0) * KVTOK + DH + c0 * 16;
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        cpasync16z((vdst + (uint32_t)(u * (4 * DH))) ^ (uint32_t)(((u >> 1) & 1) << 5),
                   vsrc + (size_t)(u * 4) * KVTOK, (4 * u < krem) ? 16 : 0);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  } else {
    STAGE4_K(0)
    STAGE4_V(0)
  }
  asm volatile("cp.async.wait_group 2;\n" ::);
  __syncthreads();
  widen_q_smem(ah, Qs, lane);

  for (int rd = 0; rd < nrounds; ++rd) {
    asm volatile("cp.async.wait_group 1;\n" ::);
    if constexpr (U128)
      __syncwarp();
    else
      __syncthreads();

    // ---- phase A: this warp owns kv tile rd*4 + warp ----
    const int lbase = (rd * 4 + warp) * 32;
    const bool haveT = (lbase < vis);
    const int plb = warp * 32;
    float s[4][4];
    float nmg = -INFINITY, nmg8 = -INFINITY;
    if (haveT) {
      const bool full = (lbase + 32 <= vis);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t kf[8];
        ldm_x4(kf, kbase + (uint32_t)(nt * 8 * KSTR));
        ldm_x4(kf + 4, kbase + (uint32_t)(nt * 8 * KSTR + 64));
        float dq[4] = {0.f, 0.f, 0.f, 0.f};
        qk_d128_split(dq, ah, kf);
        if (full) {
          s[nt][0] = dq[0];
          s[nt][1] = dq[1];
          s[nt][2] = dq[2];
          s[nt][3] = dq[3];
        } else {
          const int lc = lbase + nt * 8 + tig * 2;
          s[nt][0] = (lc < vis) ? dq[0] : -INFINITY;
          s[nt][1] = (lc + 1 < vis) ? dq[1] : -INFINITY;
          s[nt][2] = (lc < vis) ? dq[2] : -INFINITY;
          s[nt][3] = (lc + 1 < vis) ? dq[3] : -INFINITY;
        }
        nmg = fmaxf(nmg, fmaxf(s[nt][0], s[nt][1]));
        nmg8 = fmaxf(nmg8, fmaxf(s[nt][2], s[nt][3]));
      }
      nmg = fmaxf(nmg, __shfl_xor_sync(0xffffffffu, nmg, 1));
      nmg = fmaxf(nmg, __shfl_xor_sync(0xffffffffu, nmg, 2));
      nmg8 = fmaxf(nmg8, __shfl_xor_sync(0xffffffffu, nmg8, 1));
      nmg8 = fmaxf(nmg8, __shfl_xor_sync(0xffffffffu, nmg8, 2));
    }
    if (tig == 0) {
      wmax[gid][warp] = nmg;
      wmax[gid + 8][warp] = nmg8;
    }
    __syncthreads();

    // the round's true per-head maximum, folded into the running one
    const float4 wm0 = *reinterpret_cast<const float4*>(&wmax[gid][0]);
    const float4 wm8 = *reinterpret_cast<const float4*>(&wmax[gid + 8][0]);
    const float rmg = fmaxf(fmaxf(wm0.x, wm0.y), fmaxf(wm0.z, wm0.w));
    const float rmg8 = fmaxf(fmaxf(wm8.x, wm8.y), fmaxf(wm8.z, wm8.w));
    const float nmgR = fmaxf(mg, rmg), nmg8R = fmaxf(mg8, rmg8);

    if (haveT) {
      const float oa = -nmgR * SCALE_LOG2E, ob = -nmg8R * SCALE_LOG2E;
      uint32_t pk[4][2];
      const F32x2 sc2 = {SCALE_LOG2E, SCALE_LOG2E};
      const F32x2 ba2 = {oa, oa}, bb2 = {ob, ob};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const F32x2 xa = fma_f32x2({s[nt][0], s[nt][1]}, sc2, ba2);
        const F32x2 xb = fma_f32x2({s[nt][2], s[nt][3]}, sc2, bb2);
        pk[nt][0] = ex2_h2(pack_h2(xa.x, xa.y));
        pk[nt][1] = ex2_h2(pack_h2(xb.x, xb.y));
        const int col = plb + nt * 8 + tig * 2;
        *reinterpret_cast<uint32_t*>(&Pb[gid * PSTRH + col]) = pk[nt][0];
        *reinterpret_cast<uint32_t*>(&Pb[(gid + 8) * PSTRH + col]) = pk[nt][1];
      }
      const uint32_t ONE2 = 0x3c003c00u;
      float z[4] = {0.f, 0.f, 0.f, 0.f};
      mma_f16_k16(z, pk[0][0], pk[0][1], pk[1][0], pk[1][1], ONE2, ONE2);
      mma_f16_k16(z, pk[2][0], pk[2][1], pk[3][0], pk[3][1], ONE2, ONE2);
      if (tig == 0) {
        wsum[gid][warp] = z[0];
        wsum[gid + 8][warp] = z[2];
      }
    } else if (tig == 0) {
      wsum[gid][warp] = 0.f;
      wsum[gid + 8][warp] = 0.f;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // ---- phase B: this warp owns output columns [32*warp, 32*warp+32) ----
    // With one round the denominator is only needed by the final rescale, so
    // its cross-warp fold is deferred past the PV MMAs and off the path
    // between the barrier and the first ldmatrix.
    float rsg = 0.f, rsg8 = 0.f;
    if (!ONEROUND) {
      const float4 ws0 = *reinterpret_cast<const float4*>(&wsum[gid][0]);
      const float4 ws8 = *reinterpret_cast<const float4*>(&wsum[gid + 8][0]);
      rsg = (ws0.x + ws0.y) + (ws0.z + ws0.w);
      rsg8 = (ws8.x + ws8.y) + (ws8.z + ws8.w);
    }
    if (!ONEROUND && rd) {
      const float sc = exp2f((mg - nmgR) * SCALE_LOG2E);
      const float sc8 = exp2f((mg8 - nmg8R) * SCALE_LOG2E);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[j][0] *= sc;
        acc[j][1] *= sc;
        acc[j][2] *= sc8;
        acc[j][3] *= sc8;
      }
      lg = lg * sc + rsg;
      lg8 = lg8 * sc8 + rsg8;
    } else {
      lg = rsg;
      lg8 = rsg8;
    }
    mg = nmgR;
    mg8 = nmg8R;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if ((rd * 4 + j) * 32 >= vis) break;
      uint32_t a0[4], a1[4];
      ldm_x4(a0, pbase + (uint32_t)(j * 64));
      ldm_x4(a1, pbase + (uint32_t)(j * 64 + 32));
      const uint32_t vb0 = smemu32(SVBUF(j)) + (uint32_t)(vrow * DH);
      uint32_t vb0r[4], vb1r[4];
      ldm_x2_t8(vb0r, vb0 + (((uint32_t)(2 * warp) ^ vswz) << 4));
      ldm_x2_t8(vb1r, vb0 + (((uint32_t)(2 * warp + 1) ^ vswz) << 4));
      mma_f16_k16(acc[0], a0[0], a0[1], a0[2], a0[3], CVLO(vb0r[0]), CVHI(vb0r[0]));
      mma_f16_k16(acc[1], a0[0], a0[1], a0[2], a0[3], CVLO(vb0r[1]), CVHI(vb0r[1]));
      mma_f16_k16(acc[2], a0[0], a0[1], a0[2], a0[3], CVLO(vb1r[0]), CVHI(vb1r[0]));
      mma_f16_k16(acc[3], a0[0], a0[1], a0[2], a0[3], CVLO(vb1r[1]), CVHI(vb1r[1]));
      mma_f16_k16(acc[0], a1[0], a1[1], a1[2], a1[3], CVLO(vb0r[2]), CVHI(vb0r[2]));
      mma_f16_k16(acc[1], a1[0], a1[1], a1[2], a1[3], CVLO(vb0r[3]), CVHI(vb0r[3]));
      mma_f16_k16(acc[2], a1[0], a1[1], a1[2], a1[3], CVLO(vb1r[2]), CVHI(vb1r[2]));
      mma_f16_k16(acc[3], a1[0], a1[1], a1[2], a1[3], CVLO(vb1r[3]), CVHI(vb1r[3]));
    }

    if (!ONEROUND && rd + 1 < nrounds) {
      __syncthreads();
      STAGE4_K(rd + 1)
      STAGE4_V(rd + 1)
    }
  }
#undef STAGE4_K
#undef STAGE4_V
#undef SKBUF
#undef SVBUF

  if constexpr (ONEROUND) {
    const float4 ws0 = *reinterpret_cast<const float4*>(&wsum[gid][0]);
    const float4 ws8 = *reinterpret_cast<const float4*>(&wsum[gid + 8][0]);
    lg = (ws0.x + ws0.y) + (ws0.z + ws0.w);
    lg8 = (ws8.x + ws8.y) + (ws8.z + ws8.w);
  }
  store_tile<true, 4, true, true>(out + (size_t)tok * HQD + warp * 32, acc, 1.0f / lg, 1.0f / lg8,
                                  reinterpret_cast<uint16_t*>(Qs + warp * (16 * QSTR)), lane, true);
}

// --- helpers for the 4-warp cooperative path -------------------------------
__device__ __forceinline__ void cpcg16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
               "r"(bytes));
}
__device__ __forceinline__ void cp16n_fast(uint32_t dst, const void* src) {
  // Inline cp.async.cg takes the shared-window address directly.  The
  // __pipeline_memcpy_async helper round-trips through a generic pointer and a
  // runtime size switch, which NCU attributed ~13% of this kernel's issued
  // instructions to.
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }
// cp.async.wait_group with a literal group count: the __pipeline_* helpers add
// a predicated-off fallback path that still consumes issue slots.
template <int N>
__device__ __forceinline__ void cp_waitn() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                      uint32_t p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3},[%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(p));
}
__device__ __forceinline__ void ldsm_t8(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                        uint32_t p) {
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {%0,%1,%2,%3},[%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(p));
}
__device__ __forceinline__ void mma8(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma8z(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  float z = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}
__device__ __forceinline__ void barn(int id, int cnt) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(cnt));
}
__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.f32 %0,%1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_h2(uint32_t x) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0,%1;\n" : "=h"(h) : "r"(x));
  return (uint32_t)h;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2(float hi, float lo) {
  uint16_t h;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;\n" : "=h"(h) : "f"(hi), "f"(lo));
  return (uint32_t)h;
}
__device__ __forceinline__ uint32_t cvt_bf16x2(float hi, float lo) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

template <bool U128 = false>
__global__ __launch_bounds__(128, 6) void attn_coop_b7(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ kvp, const int* __restrict__ bt,
    const int* __restrict__ cuq, const int* __restrict__ sl, uint32_t* __restrict__ outp, int btw) {
  constexpr int DT = 4;  // 8-wide head-dim tiles per warp
  constexpr int DG = 2;  // 16-byte head-dim groups per warp
  constexpr int OSTR = 8 * DT + 8;
  constexpr int OUT_SCRATCH_BYTES = 4 * 16 * OSTR * sizeof(uint16_t);
  constexpr int P_BYTES = 16 * PSTRH * sizeof(uint16_t);
  constexpr int PSUM_BYTES = 4 * 16 * sizeof(float);
  static_assert(OUT_SCRATCH_BYTES + P_BYTES + PSUM_BYTES <= 128 * 128);
  constexpr uint32_t ONE2C = 0x3c003c00u;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31, sp = tid >> 5;
  const int r = U128 ? (int)blockIdx.x : (int)blockIdx.y;
  const int t0 = U128 ? (127 - (int)blockIdx.y) : ((int)gridDim.x - 1 - (int)blockIdx.x);

  // U128 speculation verified against cuq/sl (loads issue together); on
  // mismatch the loaded metadata drives the same masked body.
  const int q0r = __ldg(cuq + r);
  const int q1r = __ldg(cuq + r + 1);
  const int kvr = __ldg(sl + r);
  const bool uni = !U128 || (q0r == r * 128 && q1r == r * 128 + 128 && kvr == 128);
  const int q0 = (U128 && uni) ? r * 128 : q0r;
  const int kvlen = (U128 && uni) ? 128 : kvr;
  const int page0 = U128 ? ldu32(bt + r) : __ldg(bt + r * btw);
  const int qlen = (U128 && uni) ? 128 : (q1r - q0r);
  if (t0 >= qlen) return;
  const int vis = kvlen - qlen + t0 + 1;
  const int spad = (vis + 31) & ~31;
  const int nchunks = spad >> 5;
  const bool hasc = (sp < nchunks);

  const uint32_t QS = smemu32(smem_raw);
  const uint32_t KS = QS + 2048;
  const uint32_t VS = KS + 128 * 128;
  const uint32_t PS = KS + OUT_SCRATCH_BYTES;
  float* psum = reinterpret_cast<float*>(smem_raw + 2048 + OUT_SCRATCH_BYTES + P_BYTES);
  float* mbuf = reinterpret_cast<float*>(smem_raw + 2048 + 2 * 128 * 128);

  const int u = lane & 3, i8 = lane >> 2, l8 = lane & 7, h8 = (lane >> 3) & 3;

  {
    const int head = tid >> 3, ch = tid & 7;
    cpasync16(QS + (uint32_t)(head * 128 + ((ch ^ (head & 7)) << 4)),
              qp + (size_t)(q0 + t0) * 2048 + tid * 16);
  }
  // K and V are staged by disjoint warps rather than by alternating lanes:
  // a lane-level split makes every warp run both loops at half occupancy, so
  // the staging costs twice the instructions for the same traffic.
  const int ht = tid & 63, jk = ht & 7, lt0 = ht >> 3;
  const uint8_t* srcb = kvp + (size_t)page0 * 32768 + jk * 16;
  const int nit = spad >> 3;
  const int vmax = vis - 1;
  if (tid < 64) {
    uint32_t d = KS + (uint32_t)(lt0 * 128 + ((jk ^ lt0) << 4));
    // Two kv rows per iteration: the loop control was costing as much as the
    // copies it issued.
#pragma unroll 1
    for (int it = 0; it < nit; it += 2) {
      const int la = lt0 + 8 * it, lb = la + 8;
      cp16n_fast(d, srcb + (size_t)(la < vmax ? la : vmax) * 256);
      cp16n_fast(d + 1024, srcb + (size_t)(lb < vmax ? lb : vmax) * 256);
      d += 2048;
    }
  }
  cp_commit();
  if (tid >= 64) {
    const int sA = lt0 & 3;
    uint32_t dA = VS + (uint32_t)(lt0 * 128 + ((jk ^ sA) << 4));
    uint32_t dB = VS + (uint32_t)((lt0 + 8) * 128 + ((jk ^ (sA + 4)) << 4));
    const uint8_t* srcv = srcb + 128;
#pragma unroll 1
    for (int it = 0; it < nit; it += 2) {
      const int la = lt0 + 8 * it, lb = la + 8;
      cp16n_fast(dA, srcv + (size_t)(la < vmax ? la : vmax) * 256);
      cp16n_fast(dB, srcv + (size_t)(lb < vmax ? lb : vmax) * 256);
      dA += 2048;
      dB += 2048;
    }
  }
  cp_commit();
  cp_waitn<1>();  // Q + K
  __syncthreads();

  const int cV = (lane & 1) + (((lane >> 1) & 1) << 3) + (((lane >> 2) & 3) << 1);
  const int fV = (cV & 3) + (((cV >> 3) & 1) << 2);
  const uint32_t vbase = VS + (uint32_t)((16 * (lane >> 4) + cV) * 128);
  const uint32_t pqrow = (uint32_t)((l8 + 8 * (h8 & 1)) * 128);

  float s[4][4];
  float m0 = -INFINITY, m1 = -INFINITY;
  if (hasc) {
    uint32_t ah[8][4];
    {
      uint32_t qa[4][4];
#pragma unroll
      for (int st = 0; st < 4; ++st)
        ldm_x4(qa[st], QS + pqrow + (uint32_t)(((2 * st + (h8 >> 1)) ^ l8) << 4));
      widen_q(ah, qa);
    }
    const int rem = vis - sp * 32;
    const uint32_t kb0base = KS + (uint32_t)((sp * 32 + l8) * 128);
    if (rem >= 32) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        uint32_t kf[8];
        const uint32_t kb0 = kb0base + (uint32_t)(nt * 1024);
        ldm_x4(kf, kb0 + (uint32_t)((h8 ^ l8) << 4));
        ldm_x4(kf + 4, kb0 + (uint32_t)(((h8 + 4) ^ l8) << 4));
        float dq[4] = {0.f, 0.f, 0.f, 0.f};
        qk_d128(dq, ah, kf);
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] = dq[e];
      }
    } else {
      const int nnt = (rem + 7) >> 3;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        if (nt < nnt) {
          uint32_t kf[8];
          const uint32_t kb0 = kb0base + (uint32_t)(nt * 1024);
          ldm_x4(kf, kb0 + (uint32_t)((h8 ^ l8) << 4));
          ldm_x4(kf + 4, kb0 + (uint32_t)(((h8 + 4) ^ l8) << 4));
          float dq[4] = {0.f, 0.f, 0.f, 0.f};
          qk_d128(dq, ah, kf);
#pragma unroll
          for (int e = 0; e < 4; ++e) s[nt][e] = dq[e];
          const int cc = nt * 8 + 2 * u;
          if (cc >= rem) {
            s[nt][0] = -INFINITY;
            s[nt][2] = -INFINITY;
          }
          if (cc + 1 >= rem) {
            s[nt][1] = -INFINITY;
            s[nt][3] = -INFINITY;
          }
        } else {
          s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = -INFINITY;
        }
      }
    }
    m0 = fmaxf(s[0][0], s[0][1]);
    m1 = fmaxf(s[0][2], s[0][3]);
#pragma unroll
    for (int nt = 1; nt < 4; ++nt) {
      m0 = fmaxf(m0, fmaxf(s[nt][0], s[nt][1]));
      m1 = fmaxf(m1, fmaxf(s[nt][2], s[nt][3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
  }
  float nm0 = m0, nm1 = m1;
  if (nchunks > 1) {
    if (u == 0) {
      mbuf[sp * 16 + i8] = m0;
      mbuf[sp * 16 + i8 + 8] = m1;
    }
    __syncthreads();
    nm0 = fmaxf(mbuf[i8], mbuf[16 + i8]);
    nm1 = fmaxf(mbuf[i8 + 8], mbuf[24 + i8]);
    if (nchunks > 2) {
      nm0 = fmaxf(nm0, mbuf[32 + i8]);
      nm1 = fmaxf(nm1, mbuf[40 + i8]);
    }
    if (nchunks > 3) {
      nm0 = fmaxf(nm0, mbuf[48 + i8]);
      nm1 = fmaxf(nm1, mbuf[56 + i8]);
    }
  }

  float lsum0 = 0.f, lsum1 = 0.f;
  if (hasc) {
    uint32_t ph[4][2];
    const F32x2 sc = {SCALE_LOG2E, SCALE_LOG2E};
    const F32x2 bs0 = {-nm0 * SCALE_LOG2E, -nm0 * SCALE_LOG2E};
    const F32x2 bs1 = {-nm1 * SCALE_LOG2E, -nm1 * SCALE_LOG2E};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const F32x2 xa = fma_f32x2({s[nt][0], s[nt][1]}, sc, bs0);
      const F32x2 xb = fma_f32x2({s[nt][2], s[nt][3]}, sc, bs1);
      ph[nt][0] = ex2_h2(pack_h2(xa.x, xa.y));
      ph[nt][1] = ex2_h2(pack_h2(xb.x, xb.y));
      const int col = sp * 32 + nt * 8 + 2 * u;
      const uint32_t paddr = PS + (uint32_t)((i8 * PSTRH + col) * sizeof(uint16_t));
      asm volatile("st.shared.b32 [%0], %1;\n" : : "r"(paddr), "r"(ph[nt][0]));
      asm volatile("st.shared.b32 [%0], %1;\n"
                   :
                   : "r"(paddr + (uint32_t)(8 * PSTRH * sizeof(uint16_t))), "r"(ph[nt][1]));
    }
    float z[4] = {0.f, 0.f, 0.f, 0.f};
    mma_f16_k16(z, ph[0][0], ph[0][1], ph[1][0], ph[1][1], ONE2C, ONE2C);
    mma_f16_k16(z, ph[2][0], ph[2][1], ph[3][0], ph[3][1], ONE2C, ONE2C);
    lsum0 = z[0];
    lsum1 = z[2];
  }
  if (u == 0) {
    psum[sp * 16 + i8] = lsum0;
    psum[sp * 16 + i8 + 8] = lsum1;
  }
  cp_waitn<0>();  // V resident
  __syncthreads();

  float den0 = psum[i8], den1 = psum[i8 + 8];
#pragma unroll
  for (int k = 1; k < 4; ++k) {
    den0 += psum[k * 16 + i8];
    den1 += psum[k * 16 + i8 + 8];
  }

  float o[DT][4];
#pragma unroll
  for (int j = 0; j < DT; ++j) {
    o[j][0] = 0.f;
    o[j][1] = 0.f;
    o[j][2] = 0.f;
    o[j][3] = 0.f;
  }
  const uint32_t pbase =
      PS + (uint32_t)(((lane & 15) * PSTRH + (lane >> 4) * 8) * sizeof(uint16_t));
#pragma unroll 1
  for (int c2 = 0; c2 < nchunks; ++c2) {
    uint32_t pf0[4], pf1[4];
    ldm_x4(pf0, pbase + (uint32_t)(c2 * 64));
    ldm_x4(pf1, pbase + (uint32_t)(c2 * 64 + 32));
    const uint32_t vb0 = vbase + (uint32_t)(c2 * 4096);
#pragma unroll
    for (int g = 0; g < DG; ++g) {
      uint32_t vb[4];
      ldm_x2_t8(vb, vb0 + (uint32_t)(((sp * DG + g) ^ fV) << 4));
      mma_f16_k16(o[2 * g], pf0[0], pf0[1], pf0[2], pf0[3], CVLO(vb[0]), CVHI(vb[0]));
      mma_f16_k16(o[2 * g + 1], pf0[0], pf0[1], pf0[2], pf0[3], CVLO(vb[1]), CVHI(vb[1]));
      mma_f16_k16(o[2 * g], pf1[0], pf1[1], pf1[2], pf1[3], CVLO(vb[2]), CVHI(vb[2]));
      mma_f16_k16(o[2 * g + 1], pf1[0], pf1[1], pf1[2], pf1[3], CVLO(vb[3]), CVHI(vb[3]));
    }
  }

  const float rs0 = 1.0f / den0, rs1 = 1.0f / den1;
  store_tile<true, DT, false, true>(
      reinterpret_cast<__nv_bfloat16*>(outp) + (size_t)(q0 + t0) * HQD + sp * (8 * DT), o, rs0, rs1,
      reinterpret_cast<uint16_t*>(smem_raw + HQD) + sp * (16 * OSTR), lane, true);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fast path: max_kv_len <= 128.  CTA = NTOK tokens x 4 warps.
// Phase A: warp sp computes S for kv chunk sp (32 kv), local row max.
// Exchange max, build f16 P, publish to smem (aliased over the dead K tile).
// Phase B: warp sp owns d-tiles [4*sp,4*sp+4) and accumulates over all chunks.
// ---------------------------------------------------------------------------
template <int NTOK, int KVCAP, int NTMAX, bool U128 = false, bool LPT = false>
__global__ void __launch_bounds__((128 * NTOK), (NTOK == 4 ? 2 : (NTOK == 2 ? 4 : 6)))
    attn_fast(const uint8_t* __restrict__ qp, const uint8_t* __restrict__ kvp,
              const int* __restrict__ bt, const int* __restrict__ cuq, const int* __restrict__ sl,
              uint32_t* __restrict__ outp, int btw) {
  constexpr int SPL = 4;
  constexpr int DT = 16 / SPL;  // d-tiles (8 wide) per warp
  constexpr int DG = 8 / SPL;   // 16-byte d-groups per warp

  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int tok = warp >> 2, sp = warp & 3;
  const int r = LPT ? (int)blockIdx.x : (int)blockIdx.y;
  const int tile = LPT ? (int)gridDim.y - 1 - (int)blockIdx.y : (int)blockIdx.x;

  // U128 speculation verified against cuq/sl (loads issue together); on
  // mismatch the loaded metadata drives the same masked body.
  const int q0r = __ldg(cuq + r);
  const int q1r = __ldg(cuq + r + 1);
  const int kvr = __ldg(sl + r);
  const bool uni = !U128 || (q0r == r * 128 && q1r == r * 128 + 128 && kvr == 128);
  const int q0 = (U128 && uni) ? r * 128 : q0r;
  const int qlen = (U128 && uni) ? 128 : (q1r - q0r);
  const int kvlen = (U128 && uni) ? 128 : kvr;
  const int page0 = __ldg(bt + r * btw);
  const int t0 = tile * NTOK;
  if (t0 >= qlen) return;
  const int ntok = min(NTOK, qlen - t0);
  const int delta = kvlen - qlen;
  const int kvneed = delta + t0 + ntok;

  const uint32_t QS = smemu32(smem_raw);
  const uint32_t KS = QS + NTOK * 2048;
  const uint32_t VS = KS + KVCAP * 128;
  float* mbuf = reinterpret_cast<float*>(smem_raw + NTOK * 2048 + 2 * KVCAP * 128);

  // ---- Q (one 16B copy per thread covers all NTOK tokens)
  {
    const int it = tid >> 7, tt = tid & 127;
    const int head = tt >> 3, ch = tt & 7;
    const uint32_t dq = QS + it * 2048 + head * 128 + (uint32_t)((ch ^ (head & 7)) << 4);
    const int ti = (it < ntok) ? it : 0;
    cp16n_fast(dq, qp + (size_t)(q0 + t0 + ti) * 2048 + tt * 16);
  }
  // ---- K/V.  Disjoint producer warps keep every issued copy coalesced: the
  // first two warps stage K, then the second pair stage V in a later group so
  // QK can overlap the still-pending V transfer.
  const int ht = tid & 63, jk = ht & 7, lt0 = ht >> 3;
  const uint8_t* srcb = kvp + (size_t)page0 * 32768 + jk * 16;
  const int spad = (kvneed + 31) & ~31;
  const int nit = spad >> 3;
  const int vmax = kvneed - 1;
  if (tid < 64) {
    uint32_t d = KS + (uint32_t)(lt0 * 128 + ((jk ^ lt0) << 4));
    // Two kv rows per iteration: the loop control was costing as much as the
    // copies it issued.
#pragma unroll 1
    for (int it = 0; it < nit; it += 2) {
      const int la = lt0 + 8 * it, lb = la + 8;
      cp16n_fast(d, srcb + (size_t)(la < vmax ? la : vmax) * 256);
      cp16n_fast(d + 1024, srcb + (size_t)(lb < vmax ? lb : vmax) * 256);
      d += 2048;
    }
  }
  cp_commit();
  if (tid >= 64) {
    const int sA = lt0 & 3;
    uint32_t dA = VS + (uint32_t)(lt0 * 128 + ((jk ^ sA) << 4));
    uint32_t dB = VS + (uint32_t)((lt0 + 8) * 128 + ((jk ^ (sA + 4)) << 4));
    const uint8_t* srcv = srcb + 128;
#pragma unroll 1
    for (int it = 0; it < nit; it += 2) {
      const int la = lt0 + 8 * it, lb = la + 8;
      cp16n_fast(dA, srcv + (size_t)(la < vmax ? la : vmax) * 256);
      cp16n_fast(dB, srcv + (size_t)(lb < vmax ? lb : vmax) * 256);
      dA += 2048;
      dB += 2048;
    }
  }

  const int u = lane & 3;
  const int i8 = lane >> 2;
  const int l8 = lane & 7;
  const int h8 = (lane >> 3) & 3;
  const int cV = (lane & 1) + (((lane >> 1) & 1) << 3) + (((lane >> 2) & 3) << 1);
  const int fV = (cV & 3) + (((cV >> 3) & 1) << 2);
  const uint32_t vbase = VS + (uint32_t)((16 * (lane >> 4) + cV) * 128);
  const uint32_t pqrow = (uint32_t)(tok * 2048 + (l8 + 8 * (h8 & 1)) * 128);

  const int vis = (tok < ntok) ? (delta + t0 + tok + 1) : 0;
  const int nchunks = (vis + 31) >> 5;
  const bool hasc = (sp < nchunks);

  cp_commit();
  cp_waitn<1>();
  __syncthreads();

  // ---- replicated path: whole kv fits one 32-wide chunk, every warp of the
  // token computes the same S/P, then splits the output head-dim.  No max
  // exchange, no P round-trip, no extra barriers.
  if (KVCAP == 32) {
    if (tok >= ntok) return;
    uint32_t qa[4][4];
#pragma unroll
    for (int st = 0; st < 4; ++st)
      ldsm4(qa[st][0], qa[st][1], qa[st][2], qa[st][3],
            QS + pqrow + (uint32_t)((((2 * st + (h8 >> 1)) ^ l8)) << 4));
    float sr[NTMAX][4];
    const int ncols = vis;
    const int nnt = (ncols + 7) >> 3;
    const uint32_t kb0base = KS + (uint32_t)(l8 * 128);
#pragma unroll
    for (int nt = 0; nt < NTMAX; ++nt) {
      if (NTMAX == 1 || nt < nnt) {
        uint32_t kb[8];
        const uint32_t kb0 = kb0base + (uint32_t)(nt * 1024);
        ldsm4(kb[0], kb[1], kb[2], kb[3], kb0 + (uint32_t)((h8 ^ l8) << 4));
        ldsm4(kb[4], kb[5], kb[6], kb[7], kb0 + (uint32_t)(((h8 + 4) ^ l8) << 4));
        mma8z(sr[nt], qa[0], kb[0], kb[1]);
#pragma unroll
        for (int st = 1; st < 4; ++st) mma8(sr[nt], qa[st], kb[2 * st], kb[2 * st + 1]);
#pragma unroll
        for (int e = 0; e < 4; ++e) sr[nt][e] *= SCALE_LOG2E;
        const int cc = nt * 8 + 2 * u;
        if (cc >= ncols) {
          sr[nt][0] = -INFINITY;
          sr[nt][2] = -INFINITY;
        }
        if (cc + 1 >= ncols) {
          sr[nt][1] = -INFINITY;
          sr[nt][3] = -INFINITY;
        }
      } else {
        sr[nt][0] = sr[nt][1] = sr[nt][2] = sr[nt][3] = -INFINITY;
      }
    }
    float a0 = fmaxf(sr[0][0], sr[0][1]);
    float a1 = fmaxf(sr[0][2], sr[0][3]);
#pragma unroll
    for (int nt = 1; nt < NTMAX; ++nt) {
      a0 = fmaxf(a0, fmaxf(sr[nt][0], sr[nt][1]));
      a1 = fmaxf(a1, fmaxf(sr[nt][2], sr[nt][3]));
    }
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1));
    a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1));
    a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));
    uint32_t pr[4];
#pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
      const int A = 2 * pp, Bn = 2 * pp + 1;
      uint32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
      if (A < NTMAX) {
        lo0 = cvt_e4m3x2(ex2f(sr[A < NTMAX ? A : 0][1] - a0), ex2f(sr[A < NTMAX ? A : 0][0] - a0));
        lo1 = cvt_e4m3x2(ex2f(sr[A < NTMAX ? A : 0][3] - a1), ex2f(sr[A < NTMAX ? A : 0][2] - a1));
      }
      if (Bn < NTMAX) {
        hi0 = cvt_e4m3x2(ex2f(sr[Bn < NTMAX ? Bn : 0][1] - a0),
                         ex2f(sr[Bn < NTMAX ? Bn : 0][0] - a0));
        hi1 = cvt_e4m3x2(ex2f(sr[Bn < NTMAX ? Bn : 0][3] - a1),
                         ex2f(sr[Bn < NTMAX ? Bn : 0][2] - a1));
      }
      pr[2 * pp] = lo0 | (hi0 << 16);
      pr[2 * pp + 1] = lo1 | (hi1 << 16);
    }
    float oo[DT + 1][4];
#pragma unroll
    for (int g = 0; g < DG; ++g) {
      uint32_t v0, v1, v2, v3;
      ldsm_t8(v0, v1, v2, v3, vbase + (uint32_t)((((sp * DG + g) ^ fV)) << 4));
      mma8z(oo[2 * g], pr, v0, v2);
      mma8z(oo[2 * g + 1], pr, v1, v3);
    }
    mma8z(oo[DT], pr, 0x38383838u, 0x38383838u);
    const float q0r = 1.0f / oo[DT][0], q1r = 1.0f / oo[DT][2];
    uint32_t* op0 = outp + (size_t)(q0 + t0 + tok) * 1024 + i8 * 64 + u + 4 * (sp * DT);
#pragma unroll
    for (int j = 0; j < DT; ++j) {
      op0[4 * j] = cvt_bf16x2(oo[j][1] * q0r, oo[j][0] * q0r);
      op0[512 + 4 * j] = cvt_bf16x2(oo[j][3] * q1r, oo[j][2] * q1r);
    }
    return;
  }

  float s[NTMAX][4];
  float m0 = -INFINITY, m1 = -INFINITY;
  if (hasc) {
    uint32_t ah[8][4];
    {
      uint32_t qa[4][4];
#pragma unroll
      for (int st = 0; st < 4; ++st)
        ldsm4(qa[st][0], qa[st][1], qa[st][2], qa[st][3],
              QS + pqrow + (uint32_t)((((2 * st + (h8 >> 1)) ^ l8)) << 4));
      widen_q(ah, qa);
    }
    const int ncols = min(32, vis - sp * 32);
    const int nnt = (ncols + 7) >> 3;
    const uint32_t kb0base = KS + (uint32_t)((sp * 32 + l8) * 128);
#pragma unroll
    for (int nt = 0; nt < NTMAX; ++nt) {
      if (NTMAX == 1 || nt < nnt) {
        uint32_t kb[8];
        const uint32_t kb0 = kb0base + (uint32_t)(nt * 1024);
        ldsm4(kb[0], kb[1], kb[2], kb[3], kb0 + (uint32_t)((h8 ^ l8) << 4));
        ldsm4(kb[4], kb[5], kb[6], kb[7], kb0 + (uint32_t)(((h8 + 4) ^ l8) << 4));
        float dq[4] = {0.f, 0.f, 0.f, 0.f};
        qk_d128(dq, ah, kb);
#pragma unroll
        for (int e = 0; e < 4; ++e) s[nt][e] = dq[e] * SCALE_LOG2E;
        const int cc = nt * 8 + 2 * u;
        if (cc >= ncols) {
          s[nt][0] = -INFINITY;
          s[nt][2] = -INFINITY;
        }
        if (cc + 1 >= ncols) {
          s[nt][1] = -INFINITY;
          s[nt][3] = -INFINITY;
        }
      } else {
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = -INFINITY;
      }
    }
    m0 = fmaxf(s[0][0], s[0][1]);
    m1 = fmaxf(s[0][2], s[0][3]);
#pragma unroll
    for (int nt = 1; nt < NTMAX; ++nt) {
      m0 = fmaxf(m0, fmaxf(s[nt][0], s[nt][1]));
      m1 = fmaxf(m1, fmaxf(s[nt][2], s[nt][3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
  }
  if (u == 0) {
    mbuf[warp * 16 + i8] = m0;
    mbuf[warp * 16 + i8 + 8] = m1;
  }
  if (NTOK > 1)
    barn(tok + 1, 128);
  else
    __syncthreads();

  float nm0 = mbuf[tok * 64 + i8], nm1 = mbuf[tok * 64 + i8 + 8];
#pragma unroll
  for (int k = 1; k < SPL; ++k) {
    nm0 = fmaxf(nm0, mbuf[tok * 64 + k * 16 + i8]);
    nm1 = fmaxf(nm1, mbuf[tok * 64 + k * 16 + i8 + 8]);
  }

  if (hasc) {
#pragma unroll
    for (int nt = 0; nt < NTMAX; ++nt) {
      const uint32_t ph0 = pack_h2(ex2f(s[nt][0] - nm0), ex2f(s[nt][1] - nm0));
      const uint32_t ph1 = pack_h2(ex2f(s[nt][2] - nm1), ex2f(s[nt][3] - nm1));
      const int col = sp * 32 + nt * 8 + 2 * u;
      const uint32_t paddr = KS + (uint32_t)((tok * 16 * PSTRH + i8 * PSTRH + col) * 2);
      asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(paddr), "r"(ph0));
      asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(paddr + 8 * PSTRH * 2), "r"(ph1));
    }
  }
  cp_waitn<0>();
  if (NTOK > 1)
    barn(tok + 1, 128);
  else
    __syncthreads();

  if (tok >= ntok) return;

  float o[DT + 1][4];
#pragma unroll
  for (int j = 0; j <= DT; ++j) {
    o[j][0] = 0.f;
    o[j][1] = 0.f;
    o[j][2] = 0.f;
    o[j][3] = 0.f;
  }
#pragma unroll 1
  for (int c2 = 0; c2 < nchunks; ++c2) {
    const uint32_t pbase =
        KS + (uint32_t)((tok * 16 * PSTRH + (lane & 15) * PSTRH + (lane >> 4) * 8) * 2);
    uint32_t pf0[4], pf1[4];
    ldm_x4(pf0, pbase + (uint32_t)(c2 * 64));
    ldm_x4(pf1, pbase + (uint32_t)(c2 * 64 + 32));
    const uint32_t vb0 = vbase + (uint32_t)(c2 * 4096);
#pragma unroll
    for (int g = 0; g < DG; ++g) {
      uint32_t vb[4];
      ldm_x2_t8(vb, vb0 + (uint32_t)(((sp * DG + g) ^ fV) << 4));
      mma_f16_k16(o[2 * g], pf0[0], pf0[1], pf0[2], pf0[3], CVLO(vb[0]), CVHI(vb[0]));
      mma_f16_k16(o[2 * g], pf1[0], pf1[1], pf1[2], pf1[3], CVLO(vb[2]), CVHI(vb[2]));
      mma_f16_k16(o[2 * g + 1], pf0[0], pf0[1], pf0[2], pf0[3], CVLO(vb[1]), CVHI(vb[1]));
      mma_f16_k16(o[2 * g + 1], pf1[0], pf1[1], pf1[2], pf1[3], CVLO(vb[3]), CVHI(vb[3]));
    }
    mma_f16_k16(o[DT], pf0[0], pf0[1], pf0[2], pf0[3], 0x3c003c00u, 0x3c003c00u);
    mma_f16_k16(o[DT], pf1[0], pf1[1], pf1[2], pf1[3], 0x3c003c00u, 0x3c003c00u);
  }

  const float rs0 = 1.0f / o[DT][0], rs1 = 1.0f / o[DT][2];
  uint32_t* op = outp + (size_t)(q0 + t0 + tok) * 1024 + i8 * 64 + u + 4 * (sp * DT);
#pragma unroll
  for (int j = 0; j < DT; ++j) {
    op[4 * j] = cvt_bf16x2(o[j][1] * rs0, o[j][0] * rs0);
    op[512 + 4 * j] = cvt_bf16x2(o[j][3] * rs1, o[j][2] * rs1);
  }
}

// ---------------------------------------------------------------------------

#define LAUNCH(KER, W) \
  KER<W><<<dim3((max_q + W - 1) / W, B), 32 * W, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw)
#define LAUNCH_CAP(KER, W, CAP) \
  KER<W, false, CAP>            \
      <<<dim3((max_q + W - 1) / W, B), 32 * W, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw)
#define LAUNCH_GEN(W, DS, NB) \
  attn_gen<W, DS, NB>         \
      <<<dim3((max_q + W - 1) / W, B, DS), 32 * W, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw)
#define LAUNCH_GEN_U128(W, DS, NB) \
  attn_gen<W, DS, NB, true, true>  \
      <<<dim3(B* DS, (max_q + W - 1) / W), 32 * W, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw)

void Run(ffi::TensorView q, ffi::TensorView kv_buf, ffi::TensorView block_table,
         ffi::TensorView cu_q, ffi::TensorView cu_kv, ffi::TensorView seq_lens, int64_t max_q_len,
         int64_t max_kv_len, ffi::TensorView out) {
  const int total_B = static_cast<int>(seq_lens.shape()[0]);
  const int btw = static_cast<int>(block_table.shape()[1]);
  const int max_q = static_cast<int>(max_q_len);
  const bool uniform128 =
      max_q == 128 && max_kv_len == 128 && q.shape()[0] == (int64_t)total_B * 128;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q.device().device_id));

  const bool uniform2 = max_q == 2 && max_kv_len == 2 && q.shape()[0] == (int64_t)total_B * 2;
  // Routes whose request index rides grid x never chunk (x holds 2^31-1).
  // uniform2 must be here: its kernels verify the uniform speculation against
  // cu_q's ABSOLUTE packed offsets, which a chunk-local request index and a
  // rebased q/out pointer would both break on every chunk after the first.
  const bool batch_on_x =
      uniform128 || uniform2 ||
      (max_kv_len <= 8 && max_q == max_kv_len && (max_q == 6 || max_q == 8));
  const int batch_step = batch_on_x || total_B <= 65535 ? total_B : 65535;
  int rbase = 0;
  do {
    const int B = (total_B - rbase < batch_step) ? total_B - rbase : batch_step;
    // Every kernel consumes absolute packed offsets from cu_q (the U2/U128
    // speculations are verified against them), so chunked launches advance
    // only the per-request views and keep q/out at their global bases.
    const uint8_t* qp = static_cast<const uint8_t*>(q.data_ptr());
    const uint8_t* kvp = static_cast<const uint8_t*>(kv_buf.data_ptr());
    const int* btp = static_cast<const int*>(block_table.data_ptr()) + (size_t)rbase * btw;
    const int* cqp = static_cast<const int*>(cu_q.data_ptr()) + rbase;
    const int* slp = static_cast<const int*>(seq_lens.data_ptr()) + rbase;
    __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());

    if (max_kv_len <= 8) {
      if (uniform2) {
        // Staging Q and K through cp.async turns the per-lane 4-byte strided
        // global reads into coalesced 16-byte copies and takes the global
        // load-use scoreboard chain off the QK critical path.
        if (B <= 16) {
          if (btw == 1) {
            attn_tiny_async<2, true, true, 4, 2, false, true, true, false, true>
                <<<dim3(B, 1, 4), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
          } else {
            attn_tiny_async<2, true, true, 4, 2, false, false, true, false, true>
                <<<dim3(B, 1, 4), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
          }
        } else if (btw == 1) {
          attn_tiny_async<2, true, true, 4, 2, false, true, true>
              <<<dim3(B, 1, 4), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
        } else {
          attn_tiny_async<2, true, true, 4, 2, false, false, true>
              <<<dim3(B, 1, 4), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
        }
      } else if (max_q == 3 && max_kv_len == 3) {
        // One CTA per request covers every query token, so no CTA is launched
        // only to find pbase >= q_len, except for empty requests.
        attn_tiny_async<4, false, true, 2, 3, false, false, false, true>
            <<<dim3(1, B, 2), 128, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
      } else if (max_q == 4 && max_kv_len == 4) {
        attn_tiny_async<4, false, true, 2, 4, false, false, false, true>
            <<<dim3(1, B, 2), 128, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
      } else if (max_q == 6 && max_kv_len == 6) {
        // Requests vary fastest so neighbouring CTAs miss on distinct pages.
        attn_tiny_async<2, false, true, 2, 6, true>
            <<<dim3(B, (max_q + 1) / 2, 2), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
      } else if (max_q == 8 && max_kv_len == 8) {
        attn_tiny_async<2, false, true, 1, 8, true>
            <<<dim3(B, (max_q + 1) / 2, 1), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
      } else if (max_q >= 3) {
        attn_tiny_async<2, false, true, TINY_DS>
            <<<dim3((max_q + 1) / 2, B, TINY_DS), 64, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
#if TINY_KVCAP
      } else if (max_kv_len == 3) {
        LAUNCH_CAP(attn_tiny, 2, 3);
      } else if (max_kv_len == 4) {
        LAUNCH_CAP(attn_tiny, 2, 4);
#endif
      } else {
        LAUNCH(attn_tiny, 2);
      }
    } else if (B == 1 && uniform128) {
      attn_split<true, true><<<dim3(128), 128, 0, st>>>(qp, kvp, btp, cqp, slp, op, btw);
    } else if (max_kv_len <= 128 && (long)max_q * B <= 1200) {
      // Short-context chunk with a grid that cannot by itself fill the machine:
      // give every query token four cooperating warps (kv-split for QK, head-dim
      // split for PV) so each SM has enough independent warps to hide latency.
      const int tpc = 1;
      dim3 g2((max_q + tpc - 1) / tpc, B);
      size_t sm = (size_t)tpc * 2048 + 2 * 128 * 128 + (size_t)tpc * 256;
      if (uniform128 && B > 1 && btw == 1) {
        attn_coop_b7<true><<<dim3(B, max_q), 128, sm, st>>>(qp, kvp, btp, cqp, slp,
                                                            reinterpret_cast<uint32_t*>(op), btw);
      } else if (uniform128) {
        attn_fast<1, 128, 4, true>
            <<<g2, 128, sm, st>>>(qp, kvp, btp, cqp, slp, reinterpret_cast<uint32_t*>(op), btw);
      } else {
        attn_fast<1, 128, 4>
            <<<g2, 128, sm, st>>>(qp, kvp, btp, cqp, slp, reinterpret_cast<uint32_t*>(op), btw);
      }
    } else {
      // 4 warps/CTA amortises the per-tile KV staging, but when the grid would
      // not reach every SM, shrink the CTA so the same warps spread over more
      // SMs and each gets a whole SM's shared-memory/SFU/LSU bandwidth.
      int W = (max_q <= 1) ? 1 : (max_q <= 2 ? 2 : 4);
      while (W > 1 && (long)((max_q + W - 1) / W) * B < 148) W >>= 1;
      if (W == 4) {
        if (uniform128) {
          LAUNCH_GEN_U128(4, 1, 3);
        } else {
          LAUNCH_GEN(4, 1, 4);
        }
      } else if (W == 2) {
        LAUNCH_GEN(2, 1, 4);
      } else if ((long)max_q * B < 152) {
        // Still fewer blocks than SMs: split the head dimension so each token is
        // worked by two CTAs writing disjoint output columns (no combine needed).
        LAUNCH_GEN(1, 2, 3);
      } else {
        LAUNCH_GEN(1, 1, 4);
      }
    }
    rbase += B;
  } while (rbase < total_B);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gqa_ctx_fp8_ikv_run, Run);
