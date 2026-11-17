/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Specialized SM90 batch-decode MLA attention for an fp8 (e4m3) packed query
// over an fp8 (e4m3) packed paged compressed-KV cache: 8 query heads,
// head_dim_ckv 512 + head_dim_kpe 64 (576-byte packed rows), page size 64,
// causal, bf16 output, per-tensor float32 Q and KV descales.
//
// Algorithm (the numeric-core REPAIR of the earlier all-fp8 fast-path):
// QK^T runs as fp8 mma (m16n8k32 e4m3, fp32 accumulate); softmax statistics
// are fp32 online-softmax with unit-max probability quantisation on long
// splits. Three launcher families cover the measured envelope: a cp.async
// mma.sync kernel for shallow multi-query tiles (MT<=2), a 2-compute-WG +
// TMA-producer warp-group kernel (dual 128B/64B-swizzled TMA maps over the
// ckv and kpe halves of the 576-byte rows, two-page smem queue), and a
// compact five-warp dual kernel (two CTAs per SM) for the deep multi-batch
// classes, with P.V overlapped against the TMA-full dependency. Per-split
// fp8 partials with fp32 M/L land in a caller-owned workspace, merged by
// `combine_kernel` variants with the KV descale folded into the final bf16
// write. The split plan and launch selection are carried verbatim from the
// measured launcher.
//
// PRECISION (part of the API contract, disclosed in the dispatcher): this
// is an all-fp8-compute kernel (fp8 q x fp8 KV end-to-end, the FlashMLA
// dense-fp8 class) — NOT the stock wrapper's bf16-q-over-dequantised-KV
// arithmetic. It is deterministic (atomics-free, bit-stable); its measured
// record (x4-first magnitude battery with rival parity, per-operand battery
// including the descale axes, fresh seeds, bit-determinism) accompanies the
// change.
//
// `init` must run once per device before the first `run` (SM-count query +
// dynamic smem opt-in for the MT<=2 kernels), outside CUDA-graph capture.
// `run` performs no allocation and no synchronisation; the warp-group/dual
// kernel variants arm their dynamic-smem attribute once on their first
// EAGER call (std::call_once, carried from the measured launcher), so each
// launch class needs one eager warm-up call before CUDA-graph capture —
// the same documented contract as the dispatcher enforces. The dual TMA
// descriptors are host-encoded and cached per thread by (base, rows,
// promotion); steady-state calls re-encode nothing.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

namespace {

constexpr int DIMQ = 576;
constexpr int CKV = 512;
constexpr int PAGE = 64;
constexpr int NH = 8;
constexpr int TT = 128;    // tokens per kv tile == 2 pages
constexpr int KSTR = 592;  // linear fp8 byte stride of a Q/K row (592%128==80)
constexpr int PSTR = 144;  // fp8 byte stride of a P row (128 slots + pad)
constexpr float LOG2E = 1.4426950408889634f;

static PFN_cuTensorMapEncodeTiled_v12000 get_tmap_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult status;
    cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                                       cudaEnableDefault, &status);
    TVM_FFI_ICHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess && p)
        << "cuTensorMapEncodeTiled is unavailable: " << cudaGetErrorString(err);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }();
  return fn;
}

__device__ __forceinline__ uint32_t prm(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}
__device__ __forceinline__ uint32_t sptr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mma32(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma32h(uint32_t* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                       uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// SM90 lowers fp8 mma.sync through explicit operand conversion. Convert each
// P/V fragment once and reuse it across all three row groups.
__device__ __forceinline__ uint32_t cvtl(uint32_t x) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)x));
  return r;
}
__device__ __forceinline__ uint32_t cvth(uint32_t x) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(x >> 16)));
  return r;
}
// Reduce packed half accumulators while propagating NaNs.  The sign bits of
// (abs(x) exponent + 1) are set exactly for Inf/NaN, so the final test is a
// cheap exceptional-path guard and leaves the common WGMMA path unchanged.
__device__ __forceinline__ uint32_t hmax_abs_nan(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.NaN.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ bool half2_nonfinite(uint32_t max_abs) {
  return (((max_abs & 0x7c007c00u) + 0x04000400u) & 0x80008000u) != 0;
}
__device__ __forceinline__ void mma16h(uint32_t* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                       uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16f(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void sts16(uint32_t a, unsigned short v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(a), "h"(v) : "memory");
}

// Guard rail for the packed-FP16 PV accumulator (|O| <= l * 448): if the
// running denominator bound ever approached the format ceiling, an extra power
// of two would be folded into the reference, which leaves L * 2^M invariant.
// The threshold sits far above anything the softmax can reach here, so the
// shift is always zero in practice; a tighter threshold was measured to drift
// the reference by a fixed amount per tile and underflow L on long splits.
__device__ __forceinline__ float lcap_shift(float lbound) {
  const int eb = (int)(__float_as_uint(lbound) >> 23) - 127;
  return (float)(eb > 24 ? eb - 24 : 0);
}
__device__ __forceinline__ float exp2i(float ex) {
  return __uint_as_float((unsigned int)(127 - (int)ex) << 23);
}

__device__ __forceinline__ void row_group_sync(int id) {
  asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

// Exceptional replay for one 16-row slice of a split.  The fast kernels use
// packed-FP16 WGMMA PV accumulation; when that representation overflows, the
// CTA reuses its retired shared storage and recomputes only this split with
// tile-local FP8 probabilities and FP32 mma.sync accumulation.  b/sp are
// passed explicitly because the fast rectangular grid is not the flattened
// grid used by the legacy path.
__device__ __noinline__ void repair_split(
    const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg, const int* __restrict__ btab,
    const int* __restrict__ slen, const float* __restrict__ dqp, const float* __restrict__ dkp,
    float sm_scale, __nv_bfloat16* __restrict__ outp, uint8_t* __restrict__ Op,
    float* __restrict__ Mp, float* __restrict__ Lp, int QL, int pps, int ns, int pps_split, int b,
    int sp, int qbase, char* smraw) {
  constexpr int NTHR = 128;
  constexpr int ROWS = 16;
  constexpr int NLD = (PAGE * 36) / NTHR;

  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);
  uint8_t* Ks = Qs + ROWS * KSTR;
  uint8_t* Ps = Ks + TT * KSTR;
  float* red = reinterpret_cast<float*>(Ps + ROWS * PSTR);
  float* red2 = red + ROWS * 4;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m = warp >> 2, s = warp & 3;
  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg0 = sp * pps_split;
  const int pg1 = min(ppr, pg0 + pps_split);
  const int q0 = qbase;
  const int vrows = min(ROWS, (QL - q0) * NH);
  const int rows_tot = QL * NH;

  if (pg0 >= pg1) {
    if (ns == 1) {
      for (int i = tid; i < vrows * CKV; i += NTHR)
        outp[(size_t)(b * rows_tot + q0 * NH) * CKV + i] = __float2bfloat16(0.f);
    } else {
      for (int r = tid; r < vrows; r += NTHR) {
        const size_t gi = (size_t)(b * rows_tot + q0 * NH + r) * ns + sp;
        Mp[gi] = -INFINITY;
        Lp[gi] = 0.f;
      }
      for (int i = tid; i < vrows * CKV; i += NTHR)
        Op[((size_t)(b * ns + sp) * rows_tot + q0 * NH) * CKV + i] = 0;
    }
    return;
  }

  {
    const uint8_t* qb = qg + (size_t)(b * QL + q0) * NH * DIMQ;
    for (int j = tid; j < ROWS * 36; j += NTHR) {
      const int row = j / 36, c = j % 36;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < vrows) v = *reinterpret_cast<const uint4*>(qb + (size_t)j * 16);
      *reinterpret_cast<uint4*>(Qs + row * KSTR + (c << 4)) = v;
    }
  }

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = dkp[0];
  constexpr float pshift = 8.f;

  float oacc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int k = 0; k < 4; ++k) oacc[i][k] = 0.f;
  float mrun0 = -INFINITY, mrun1 = -INFINITY;
  float lrun0 = 0.f, lrun1 = 0.f;

  const int rA = m * 16 + (lane >> 2);
  const int rB = rA + 8;
  const int gA = q0 * NH + rA, gB = q0 * NH + rB;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - QL + (gA / NH) + 1), ptop);
  const int limB = min(max(1, L - QL + (gB / NH) + 1), ptop);

  auto loadpair = [&](int pg) {
    uint8_t* dst = Ks;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int pgh = min(pg + h, pg1 - 1);
      const uint8_t* kb = kvg + (size_t)btab[b * pps + pgh] * PAGE * DIMQ;
      uint8_t* d2 = dst + h * PAGE * KSTR;
#pragma unroll
      for (int i = 0; i < NLD; ++i) {
        const int j = tid + i * NTHR, row = j / 36, c = j % 36;
        const uint32_t a = sptr(d2 + row * KSTR + (c << 4));
        if ((pgh + 1) * PAGE <= L)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a),
                       "l"(kb + (size_t)j * 16));
        else {
          const int bytes = (pgh * PAGE + row < L) ? 16 : 0;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a),
                       "l"(kb + (size_t)j * 16), "r"(bytes));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  const int mi = lane >> 3, l7 = lane & 7;
  const int qrow = m * 16 + ((mi & 1) << 3) + l7;
  const int bcA = (mi >> 1) << 4;
  const int krow = s * 32 + ((mi >> 1) << 3) + l7;
  const int bcB = (mi & 1) << 4;
  const int prow = m * 16 + ((mi & 1) << 3) + l7;
  const int pbc = (mi >> 1) << 4;
  const uint32_t qA = sptr(Qs) + qrow * KSTR + bcA;
  const uint32_t kA = krow * KSTR + bcB;
  const uint32_t Ksb = sptr(Ks);
  const uint32_t pA = sptr(Ps) + prow * PSTR + pbc;
  const uint32_t vofs = lane * KSTR + s * 128;
  uint32_t psA[4], psB[4];
  const uint32_t Psb = sptr(Ps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = s * 32 + ((j >> 1) << 4) + ((j & 1) << 3) + 2 * (lane & 3);
    const int t16 = col & 15;
    const int slot =
        (col >> 5) * 32 + (((col >> 4) & 1) << 4) + 4 * ((t16 & 7) >> 1) + 2 * (t16 >> 3);
    psA[j] = Psb + rA * PSTR + slot;
    psB[j] = Psb + rB * PSTR + slot;
  }
  const int cbase = s * 32 + 2 * (lane & 3);

  for (int pg = pg0; pg < pg1; pg += 2) {
    loadpair(pg);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    row_group_sync(12);

    const uint32_t Kbase = Ksb;
    const int tok0 = pg * PAGE;
    float sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) sacc[j][k] = 0.f;

    {
      uint32_t qp = qA, kp = Kbase + kA;
#pragma unroll 2
      for (int kk = 0; kk < 18; ++kk) {
        uint32_t a[4], b0[4], b1[4];
        ldm4(a, qp);
        ldm4(b0, kp);
        ldm4(b1, kp + 16 * KSTR);
        mma32(sacc[0], a[0], a[1], a[2], a[3], b0[0], b0[1]);
        mma32(sacc[1], a[0], a[1], a[2], a[3], b0[2], b0[3]);
        mma32(sacc[2], a[0], a[1], a[2], a[3], b1[0], b1[1]);
        mma32(sacc[3], a[0], a[1], a[2], a[3], b1[2], b1[3]);
        qp += 32;
        kp += 32;
      }
    }

    float mxA = -INFINITY, mxB = -INFINITY;
    const int remA = limA - tok0, remB = limB - tok0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c0 = cbase + ((j >> 1) << 4) + ((j & 1) << 3);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float xa = (c0 + e < remA) ? sacc[j][e] * qk_scale : -INFINITY;
        const float xb = (c0 + e < remB) ? sacc[j][2 + e] * qk_scale : -INFINITY;
        sacc[j][e] = xa;
        sacc[j][2 + e] = xb;
        mxA = fmaxf(mxA, xa);
        mxB = fmaxf(mxB, xb);
      }
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
    }
    if ((lane & 3) == 0) {
      red[rA * 4 + s] = mxA;
      red[rB * 4 + s] = mxB;
    }
    row_group_sync(m + 1);
    const float4 ia = *reinterpret_cast<const float4*>(red + rA * 4);
    const float4 ib = *reinterpret_cast<const float4*>(red + rB * 4);
    const float tileA = fmaxf(fmaxf(ia.x, ia.y), fmaxf(ia.z, ia.w));
    const float tileB = fmaxf(fmaxf(ib.x, ib.y), fmaxf(ib.z, ib.w));
    row_group_sync(m + 1);

    const float eA = tileA - pshift;
    const float eB = tileB - pshift;
    float sA = 0.f, sB = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float pa[2], pb[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float xa = (eA == -INFINITY) ? 0.f : exp2f(sacc[j][e] - eA);
        const float xb = (eB == -INFINITY) ? 0.f : exp2f(sacc[j][2 + e] - eB);
        sA += xa;
        sB += xb;
        pa[e] = xa;
        pb[e] = xb;
      }
      sts16(psA[j], __nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]), __NV_SATFINITE, __NV_E4M3));
      sts16(psB[j], __nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]), __NV_SATFINITE, __NV_E4M3));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      sA += __shfl_xor_sync(0xffffffff, sA, d);
      sB += __shfl_xor_sync(0xffffffff, sB, d);
    }
    if ((lane & 3) == 0) {
      red[rA * 4 + s] = mxA;
      red[rB * 4 + s] = mxB;
      red2[rA * 4 + s] = sA;
      red2[rB * 4 + s] = sB;
    }
    row_group_sync(m + 1);
    const float4 fa = *reinterpret_cast<const float4*>(red + rA * 4);
    const float4 fb = *reinterpret_cast<const float4*>(red + rB * 4);
    const float4 ga = *reinterpret_cast<const float4*>(red2 + rA * 4);
    const float4 gb = *reinterpret_cast<const float4*>(red2 + rB * 4);
    const float mnA = fmaxf(mrun0, fmaxf(fmaxf(fa.x, fa.y), fmaxf(fa.z, fa.w)));
    const float mnB = fmaxf(mrun1, fmaxf(fmaxf(fb.x, fb.y), fmaxf(fb.z, fb.w)));
    const float corrA = (mnA == -INFINITY) ? 1.f : exp2f(mrun0 - mnA);
    const float corrB = (mnB == -INFINITY) ? 1.f : exp2f(mrun1 - mnB);
    const float tcA = (tileA == -INFINITY) ? 0.f : exp2f(tileA - mnA);
    const float tcB = (tileB == -INFINITY) ? 0.f : exp2f(tileB - mnB);
    mrun0 = mnA;
    mrun1 = mnB;
    lrun0 = lrun0 * corrA + ((ga.x + ga.y) + (ga.z + ga.w)) * tcA;
    lrun1 = lrun1 * corrB + ((gb.x + gb.y) + (gb.z + gb.w)) * tcB;

    float tile_acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) tile_acc[i][k] = 0.f;

    {
      const uint32_t vb0 = Kbase + vofs;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t a[4];
        ldm4(a, pA + kk * 32);
        const uint32_t vb = vb0 + kk * (32 * KSTR);
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          uint32_t r[4];
          ldm4t(r, vb + (uint32_t)(g << 4));
          const uint32_t be0 = prm(r[0], r[1], 0x6420u);
          const uint32_t be1 = prm(r[2], r[3], 0x6420u);
          mma32(tile_acc[g * 2], a[0], a[1], a[2], a[3], be0, be1);
          const uint32_t bo0 = prm(r[0], r[1], 0x7531u);
          const uint32_t bo1 = prm(r[2], r[3], 0x7531u);
          mma32(tile_acc[g * 2 + 1], a[0], a[1], a[2], a[3], bo0, bo1);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
#pragma unroll
      for (int k = 0; k < 2; ++k) oacc[i][k] = fmaf(tile_acc[i][k], tcA, oacc[i][k] * corrA);
#pragma unroll
      for (int k = 2; k < 4; ++k) oacc[i][k] = fmaf(tile_acc[i][k], tcB, oacc[i][k] * corrB);
    }
    row_group_sync(12);
  }

  const bool single = (ns == 1);
  const float rAi = (lrun0 > 0.f) ? (single ? o_scale : 1.f) / lrun0 : 0.f;
  const float rBi = (lrun1 > 0.f) ? (single ? o_scale : 1.f) / lrun1 : 0.f;
  if (!single) {
    const int giA = (b * rows_tot + gA) * ns + sp;
    const int giB = (b * rows_tot + gB) * ns + sp;
    if (s == 0 && (lane & 3) == 0) {
      if (rA < vrows) {
        Mp[giA] = mrun0;
        Lp[giA] = lrun0 * 0x1p-8f;
      }
      if (rB < vrows) {
        Mp[giB] = mrun1;
        Lp[giB] = lrun1 * 0x1p-8f;
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    const int col = s * 128 + g * 16 + 4 * (lane & 3);
    const float a0 = oacc[2 * g][0], a1 = oacc[2 * g + 1][0];
    const float a2 = oacc[2 * g][1], a3 = oacc[2 * g + 1][1];
    const float b0 = oacc[2 * g][2], b1 = oacc[2 * g + 1][2];
    const float b2 = oacc[2 * g][3], b3 = oacc[2 * g + 1][3];
    if (single) {
      if (rA < vrows) {
        __nv_bfloat162* d =
            reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * QL * NH + gA) * CKV + col);
        d[0] = __floats2bfloat162_rn(a0 * rAi, a1 * rAi);
        d[1] = __floats2bfloat162_rn(a2 * rAi, a3 * rAi);
      }
      if (rB < vrows) {
        __nv_bfloat162* d =
            reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * QL * NH + gB) * CKV + col);
        d[0] = __floats2bfloat162_rn(b0 * rBi, b1 * rBi);
        d[1] = __floats2bfloat162_rn(b2 * rBi, b3 * rBi);
      }
    } else {
      const size_t base = (size_t)((b * ns + sp) * rows_tot) * CKV;
      if (rA < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(a0 * rAi, a1 * rAi, a2 * rAi, a3 * rAi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gA * CKV + col) = t.__x;
      }
      if (rB < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(b0 * rBi, b1 * rBi, b2 * rBi, b3 * rBi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gB * CKV + col) = t.__x;
      }
    }
  }
}

template <int MT, bool HALF_ACC>
__global__ __launch_bounds__(128 * MT, 1) void mla_kernel(
    const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg, const int* __restrict__ btab,
    const int* __restrict__ slen, const float* __restrict__ dqp, const float* __restrict__ dkp,
    float sm_scale, __nv_bfloat16* __restrict__ outp, uint8_t* __restrict__ Op,
    float* __restrict__ Mp, float* __restrict__ Lp, int B, int QL, int pps, int ns, int pps_split,
    int QPC) {
  constexpr int NTHR = 128 * MT;
  constexpr int ROWS = 16 * MT;
  constexpr int NLD = (PAGE * 36) / NTHR;

  extern __shared__ char smraw[];
  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);
  uint8_t* Ks = Qs + ROWS * KSTR;
  uint8_t* Ps = Ks + 2 * TT * KSTR;
  float* red = reinterpret_cast<float*>(Ps + ROWS * PSTR);  // [ROWS][4] max
  float* red2 = red + ROWS * 4;                             // [ROWS][4] sum

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m = warp >> 2, s = warp & 3;
  const int qc = blockIdx.z;
  int b = 0, sp = blockIdx.x;
  for (; b < B; ++b) {
    const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
    const int nsp_b = (ppr_b + pps_split - 1) / pps_split;
    if (sp < nsp_b) break;
    sp -= nsp_b;
  }
  if (b == B) return;

  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg0 = sp * pps_split;
  const int pg1 = min(ppr, pg0 + pps_split);
  const int q0 = qc * QPC;
  const int vrows = min(ROWS, (QL - q0) * NH);
  const int rows_tot = QL * NH;

  if (pg0 >= pg1) {
    for (int r = tid; r < vrows; r += NTHR) {
      int gi = (b * rows_tot + q0 * NH + r) * ns + sp;
      Mp[gi] = -INFINITY;
      Lp[gi] = 0.f;
    }
    return;
  }

  // ---- load Q tile (raw fp8, linear stride) ----
  {
    const uint8_t* qb = qg + (size_t)(b * QL + q0) * NH * DIMQ;
    for (int j = tid; j < ROWS * 36; j += NTHR) {
      int row = j / 36, c = j % 36;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < vrows) v = *reinterpret_cast<const uint4*>(qb + (size_t)j * 16);
      *reinterpret_cast<uint4*>(Qs + row * KSTR + (c << 4)) = v;
    }
  }

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = dkp[0];

  constexpr int OREGS = HALF_ACC ? 2 : 4;
  uint32_t oacc[16][OREGS];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int k = 0; k < OREGS; ++k) oacc[i][k] = 0;
  float mrun0 = -INFINITY, mrun1 = -INFINITY, lrun0 = 0.f, lrun1 = 0.f;

  const int rA = m * 16 + (lane >> 2);
  const int rB = rA + 8;
  const int gA = q0 * NH + rA, gB = q0 * NH + rB;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - QL + (gA / NH) + 1), ptop);
  const int limB = min(max(1, L - QL + (gB / NH) + 1), ptop);

  auto loadpair = [&](int pg, int buf) {
    uint8_t* dst = Ks + buf * TT * KSTR;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int pgh = pg + h;
      if (pgh >= pg1) pgh = pg1 - 1;  // clamp: duplicate data is masked out
      const uint8_t* kb = kvg + (size_t)btab[b * pps + pgh] * PAGE * DIMQ;
      uint8_t* d2 = dst + h * PAGE * KSTR;
#pragma unroll
      for (int i = 0; i < NLD; ++i) {
        int j = tid + i * NTHR, row = j / 36, c = j % 36;
        uint32_t a = sptr(d2 + row * KSTR + (c << 4));
        if ((pgh + 1) * PAGE <= L)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a),
                       "l"(kb + (size_t)j * 16));
        else {
          // Zero-fill the rows past the sequence in a partially-filled page.
          // Their probabilities are exactly 0, but an unwritten fp8 cache
          // byte can be NaN (0x7f) and 0 * NaN would poison the PV sum.
          const int bytes = (pgh * PAGE + row < L) ? 16 : 0;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a),
                       "l"(kb + (size_t)j * 16), "r"(bytes));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  const int mi = lane >> 3, l7 = lane & 7;
  const int qrow = m * 16 + ((mi & 1) << 3) + l7;
  const int bcA = (mi >> 1) << 4;
  const int krow = s * 32 + ((mi >> 1) << 3) + l7;
  const int bcB = (mi & 1) << 4;
  const int prow = m * 16 + ((mi & 1) << 3) + l7;
  const int pbc = (mi >> 1) << 4;

  // --- page-invariant base addresses (linear layout: base + immediate) ---
  const uint32_t qA = sptr(Qs) + qrow * KSTR + bcA;
  const uint32_t kA = krow * KSTR + bcB;  // offset within one K buffer
  const uint32_t Ksb = sptr(Ks);
  const uint32_t pA = sptr(Ps) + prow * PSTR + pbc;
  const uint32_t vofs = lane * KSTR + s * 128;
  // P store addresses (tile invariant)
  uint32_t psA[4];
  uint32_t psB[4];
  const uint32_t Psb = sptr(Ps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int col = s * 32 + ((j >> 1) << 4) + ((j & 1) << 3) + 2 * (lane & 3);
    int t16 = col & 15;
    int slot = (col >> 5) * 32 + (((col >> 4) & 1) << 4) + 4 * ((t16 & 7) >> 1) + 2 * (t16 >> 3);
    psA[j] = Psb + rA * PSTR + slot;
    psB[j] = Psb + rB * PSTR + slot;
  }
  const int cbase = s * 32 + 2 * (lane & 3);  // tile-local token col for j=0

  loadpair(pg0, 0);
  int cur = 0;

  for (int pg = pg0; pg < pg1; pg += 2) {
    const bool more = (pg + 2) < pg1;
    if (more) loadpair(pg + 2, cur ^ 1);
    if (more)
      asm volatile("cp.async.wait_group 1;\n" ::);
    else
      asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    const uint32_t Kbase = Ksb + cur * (TT * KSTR);
    const int tok0 = pg * PAGE;

    float sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) sacc[j][k] = 0.f;

    {
      uint32_t qp = qA, kp = Kbase + kA;
#pragma unroll 2
      for (int kk = 0; kk < 18; ++kk) {
        uint32_t a[4], b0[4], b1[4];
        ldm4(a, qp);
        ldm4(b0, kp);
        ldm4(b1, kp + 16 * KSTR);
        mma32(sacc[0], a[0], a[1], a[2], a[3], b0[0], b0[1]);
        mma32(sacc[1], a[0], a[1], a[2], a[3], b0[2], b0[3]);
        mma32(sacc[2], a[0], a[1], a[2], a[3], b1[0], b1[1]);
        mma32(sacc[3], a[0], a[1], a[2], a[3], b1[2], b1[3]);
        qp += 32;
        kp += 32;
      }
    }

    float mxA = -INFINITY, mxB = -INFINITY;
    const int remA = limA - tok0, remB = limB - tok0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c0 = cbase + ((j >> 1) << 4) + ((j & 1) << 3);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float xa = (c0 + e < remA) ? sacc[j][e] * qk_scale : -INFINITY;
        float xb = (c0 + e < remB) ? sacc[j][2 + e] * qk_scale : -INFINITY;
        sacc[j][e] = xa;
        sacc[j][2 + e] = xb;
        mxA = fmaxf(mxA, xa);
        mxB = fmaxf(mxB, xb);
      }
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
    }
    // Exact online-softmax reference: the four score warps exchange this
    // tile's row maxima first, so P is quantized against max(running, tile)
    // and can never exceed 1.  A lagged reference lets a record-setting tile
    // produce P up to 2^(s-m_lagged), which saturates E4M3 and overflows the
    // packed-FP16 PV accumulator on long-context rows.
    if ((lane & 3) == 0) {
      red[rA * 4 + s] = mxA;
      red[rB * 4 + s] = mxB;
    }
    row_group_sync(m + 1);
    {
      float4 ia = *reinterpret_cast<const float4*>(red + rA * 4);
      float4 ib = *reinterpret_cast<const float4*>(red + rB * 4);
      mxA = fmaxf(fmaxf(ia.x, ia.y), fmaxf(ia.z, ia.w));
      mxB = fmaxf(fmaxf(ib.x, ib.y), fmaxf(ib.z, ib.w));
    }
    float mnA = fmaxf(mrun0, mxA), mnB = fmaxf(mrun1, mxB);
    float corrA = (mrun0 == -INFINITY) ? 1.f : exp2f(mrun0 - mnA);
    float corrB = (mrun1 == -INFINITY) ? 1.f : exp2f(mrun1 - mnB);
    {
      const float exA = lcap_shift(fmaf(lrun0, corrA, (float)TT));
      const float exB = lcap_shift(fmaf(lrun1, corrB, (float)TT));
      mnA += exA;
      corrA *= exp2i(exA);
      mnB += exB;
      corrB *= exp2i(exB);
    }
    mrun0 = mnA;
    mrun1 = mnB;
    // O carries the previous reference, so rescale it before this tile's PV.
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if constexpr (HALF_ACC) {
        __half2 ha = *reinterpret_cast<__half2*>(&oacc[i][0]);
        __half2 hb = *reinterpret_cast<__half2*>(&oacc[i][1]);
        ha = __hmul2(ha, __float2half2_rn(corrA));
        hb = __hmul2(hb, __float2half2_rn(corrB));
        oacc[i][0] = *reinterpret_cast<uint32_t*>(&ha);
        oacc[i][1] = *reinterpret_cast<uint32_t*>(&hb);
      } else {
        float* of = reinterpret_cast<float*>(oacc[i]);
        of[0] *= corrA;
        of[1] *= corrA;
        of[2] *= corrB;
        of[3] *= corrB;
      }
    }
    const float eA = mnA, eB = mnB;
    float sA = 0.f, sB = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float pa[2], pb[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float xa = (eA == -INFINITY) ? 0.f : exp2f(sacc[j][e] - eA);
        float xb = (eB == -INFINITY) ? 0.f : exp2f(sacc[j][2 + e] - eB);
        sA += xa;
        sB += xb;
        pa[e] = xa;
        pb[e] = xb;
      }
      unsigned short qa =
          __nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]), __NV_SATFINITE, __NV_E4M3);
      unsigned short qb2 =
          __nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]), __NV_SATFINITE, __NV_E4M3);
      sts16(psA[j], qa);
      sts16(psB[j], qb2);
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      sA += __shfl_xor_sync(0xffffffff, sA, d);
      sB += __shfl_xor_sync(0xffffffff, sB, d);
    }
    if ((lane & 3) == 0) {
      red2[rA * 4 + s] = sA;
      red2[rB * 4 + s] = sB;
    }
    // Publishes both the tile sums and the probability tile itself.
    row_group_sync(m + 1);
    {
      float4 ga = *reinterpret_cast<const float4*>(red2 + rA * 4);
      float4 gb = *reinterpret_cast<const float4*>(red2 + rB * 4);
      lrun0 = lrun0 * corrA + ((ga.x + ga.y) + (ga.z + ga.w));
      lrun1 = lrun1 * corrB + ((gb.x + gb.y) + (gb.z + gb.w));
    }

    // ---- PV (fp8 m16n8k32, V^T fragments from ldmatrix.trans + prmt) ----
    {
      const uint32_t vb0 = Kbase + vofs;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t a[4];
        ldm4(a, pA + kk * 32);
        const uint32_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
        const uint32_t vb = vb0 + kk * (32 * KSTR);
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          uint32_t r[4];
          ldm4t(r, vb + (uint32_t)(g << 4));
          uint32_t be0 = prm(r[0], r[1], 0x6420u), be1 = prm(r[2], r[3], 0x6420u);
          if constexpr (HALF_ACC)
            mma32h(oacc[g * 2], a0, a1, a2, a3, be0, be1);
          else
            mma32(reinterpret_cast<float*>(oacc[g * 2]), a0, a1, a2, a3, be0, be1);
          uint32_t bo0 = prm(r[0], r[1], 0x7531u), bo1 = prm(r[2], r[3], 0x7531u);
          if constexpr (HALF_ACC)
            mma32h(oacc[g * 2 + 1], a0, a1, a2, a3, bo0, bo1);
          else
            mma32(reinterpret_cast<float*>(oacc[g * 2 + 1]), a0, a1, a2, a3, bo0, bo1);
        }
      }
    }

    cur ^= 1;
    __syncthreads();
  }

  if (ns > 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // ---- epilogue ----
  const bool single = (ns == 1);
  float rAi = (lrun0 > 0.f) ? (single ? o_scale : 1.f) / lrun0 : 0.f;
  float rBi = (lrun1 > 0.f) ? (single ? o_scale : 1.f) / lrun1 : 0.f;
  if (!single) {
    int giA = (b * rows_tot + gA) * ns + sp, giB = (b * rows_tot + gB) * ns + sp;
    if (s == 0 && (lane & 3) == 0) {
      if (rA < vrows) {
        Mp[giA] = mrun0;
        Lp[giA] = lrun0;
      }
      if (rB < vrows) {
        Mp[giB] = mrun1;
        Lp[giB] = lrun1;
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    int col = s * 128 + g * 16 + 4 * (lane & 3);
    float a0, a1, a2, a3, b0, b1, b2, b3;
    if constexpr (HALF_ACC) {
      float2 av0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g][0]));
      float2 av1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g + 1][0]));
      float2 bv0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g][1]));
      float2 bv1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g + 1][1]));
      a0 = av0.x;
      a1 = av1.x;
      a2 = av0.y;
      a3 = av1.y;
      b0 = bv0.x;
      b1 = bv1.x;
      b2 = bv0.y;
      b3 = bv1.y;
    } else {
      float* oa0 = reinterpret_cast<float*>(oacc[2 * g]);
      float* oa1 = reinterpret_cast<float*>(oacc[2 * g + 1]);
      a0 = oa0[0];
      a1 = oa1[0];
      a2 = oa0[1];
      a3 = oa1[1];
      b0 = oa0[2];
      b1 = oa1[2];
      b2 = oa0[3];
      b3 = oa1[3];
    }
    if (single) {
      if (rA < vrows) {
        __nv_bfloat162 t0 = __floats2bfloat162_rn(a0 * rAi, a1 * rAi);
        __nv_bfloat162 t1 = __floats2bfloat162_rn(a2 * rAi, a3 * rAi);
        __nv_bfloat162* d =
            reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * QL * NH + gA) * CKV + col);
        d[0] = t0;
        d[1] = t1;
      }
      if (rB < vrows) {
        __nv_bfloat162 t0 = __floats2bfloat162_rn(b0 * rBi, b1 * rBi);
        __nv_bfloat162 t1 = __floats2bfloat162_rn(b2 * rBi, b3 * rBi);
        __nv_bfloat162* d =
            reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * QL * NH + gB) * CKV + col);
        d[0] = t0;
        d[1] = t1;
      }
    } else {
      size_t base = (size_t)((b * ns + sp) * rows_tot) * CKV;
      if (rA < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(a0 * rAi, a1 * rAi, a2 * rAi, a3 * rAi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gA * CKV + col) = t.__x;
      }
      if (rB < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(b0 * rBi, b1 * rBi, b2 * rBi, b3 * rBi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gB * CKV + col) = t.__x;
      }
    }
  }
}

// ===== wgmma (sm_90a) 48-row kernel, 2 compute WGs + 1 TMA producer WG =====
constexpr int CMS = 144;        // LBO: byte stride between 16B core-matrix cols
constexpr int SBOX = 36 * CMS;  // 5184: stride between 8-row groups
constexpr int PGB = 8 * SBOX;   // 41472: bytes for 64 rows (one page)

// Native 128B WGMMA swizzle: five 64x128B K slabs cover the 576-byte row.
// Each 8x128B atom is contiguous and XORs its 16B column by row&7.
constexpr int SW_ATOM = 8 * 128;
constexpr int SW_LBO = 64 * 128;
constexpr int SW_PAGE = 5 * SW_LBO;
constexpr uint32_t TMA_PAIR_BYTES = 2 * 5 * SW_LBO;

static_assert((SW_ATOM % 1024) == 0 && (SW_LBO % 1024) == 0 && (SW_PAGE % 1024) == 0,
              "SW128 slabs must preserve the 1024-byte swizzle origin");

__device__ __forceinline__ uint32_t swofs64(int row, int c16) {
  const int lr = row & 7;
  return (uint32_t)((c16 >> 3) * SW_LBO + (row >> 3) * SW_ATOM + lr * 128 + ((c16 & 7) ^ lr) * 16);
}

__device__ __forceinline__ uint64_t gdesc_sw(uint32_t addr) {
  return ((uint64_t)((addr & 0x3FFFFu) >> 4)) | ((uint64_t)((uint32_t)SW_LBO >> 4) << 16) |
         ((uint64_t)((uint32_t)SW_ATOM >> 4) << 32) | (1ull << 62);
}

// Packed page: four 128B-swizzled CKV slabs plus one 64B-swizzled KPE slab.
// Dropping the 64 wasted bytes per row of the old fifth 128B slab shrinks a
// page from 40960 to 36864 bytes, which buys a fifth ring slot.
constexpr int KPE_OFS = 4 * SW_LBO;         // 32768
constexpr int SW64_ATOM = 8 * 64;           // 512
constexpr int PAGE_SZ = KPE_OFS + 64 * 64;  // 36864
constexpr uint32_t TMA_PAGE_BYTES = PAGE_SZ;
constexpr int NSLOT = 5;
constexpr int MAXPG = 288;  // page ids staged in shared memory

__device__ __forceinline__ uint32_t swofs64b(int row, int c16) {
  const int lr = (row >> 1) & 3;
  return (uint32_t)((row >> 3) * SW64_ATOM + (row & 7) * 64 + ((c16 ^ lr) << 4));
}

__device__ __forceinline__ uint64_t gdesc_sw64(uint32_t addr) {
  return ((uint64_t)((addr & 0x3FFFFu) >> 4)) | (1ull << 16) |
         ((uint64_t)((uint32_t)SW64_ATOM >> 4) << 32) | (2ull << 62);
}

__device__ __forceinline__ void tma_bar_init(uint64_t* bar, uint32_t arrivals = 1) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sptr(bar)), "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ void tma_bar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(sptr(bar)),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void tma_bar_wait(uint64_t* bar, int phase) {
  uint32_t done;
  do {
    asm volatile(
        "{ .reg .pred p;\n"
        "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(sptr(bar)), "r"(phase)
        : "memory");
  } while (!done);
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;" ::"r"(sptr(bar))
               : "memory");
}

__device__ __forceinline__ void tma_load_slab(uint8_t* dst, const CUtensorMap* map, int col,
                                              int row, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];" ::"r"(sptr(dst)),
      "l"(map), "r"(col), "r"(row), "r"(sptr(bar))
      : "memory");
}

// The KV stream has no reuse at all: every page is read once by one CTA, so
// the lines are marked evict-first to keep them from displacing anything else
// and to cut L2 replacement pressure (L2 is the busiest unit at 61% of peak).
__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ void tma_load_slab_ef(uint8_t* dst, const CUtensorMap* map, int col,
                                                 int row, uint64_t* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(sptr(dst)),
      "l"(map), "r"(col), "r"(row), "r"(sptr(bar)), "l"(pol)
      : "memory");
}

__device__ __forceinline__ void compute_sync() { asm volatile("bar.sync 8, 256;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void warpgroup_reg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}

template <int N>
__device__ __forceinline__ void warpgroup_reg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}

#define WGMMA64(SD)                                                                                \
  asm volatile(                                                                                    \
      "{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"                                                \
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                                      \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%"   \
      "24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"                                   \
      : "+f"(sa[0]), "+f"(sa[1]), "+f"(sa[2]), "+f"(sa[3]), "+f"(sa[4]), "+f"(sa[5]), "+f"(sa[6]), \
        "+f"(sa[7]), "+f"(sa[8]), "+f"(sa[9]), "+f"(sa[10]), "+f"(sa[11]), "+f"(sa[12]),           \
        "+f"(sa[13]), "+f"(sa[14]), "+f"(sa[15]), "+f"(sa[16]), "+f"(sa[17]), "+f"(sa[18]),        \
        "+f"(sa[19]), "+f"(sa[20]), "+f"(sa[21]), "+f"(sa[22]), "+f"(sa[23]), "+f"(sa[24]),        \
        "+f"(sa[25]), "+f"(sa[26]), "+f"(sa[27]), "+f"(sa[28]), "+f"(sa[29]), "+f"(sa[30]),        \
        "+f"(sa[31])                                                                               \
      : "l"(da), "l"(db), "n"(SD))

#define WGMMA64_INIT                                                                               \
  asm volatile(                                                                                    \
      "{\n.reg .pred p;\n setp.ne.b32 p, 0, 0;\n"                                                  \
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                                      \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%"   \
      "24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"                                   \
      : "=f"(sa[0]), "=f"(sa[1]), "=f"(sa[2]), "=f"(sa[3]), "=f"(sa[4]), "=f"(sa[5]), "=f"(sa[6]), \
        "=f"(sa[7]), "=f"(sa[8]), "=f"(sa[9]), "=f"(sa[10]), "=f"(sa[11]), "=f"(sa[12]),           \
        "=f"(sa[13]), "=f"(sa[14]), "=f"(sa[15]), "=f"(sa[16]), "=f"(sa[17]), "=f"(sa[18]),        \
        "=f"(sa[19]), "=f"(sa[20]), "=f"(sa[21]), "=f"(sa[22]), "=f"(sa[23]), "=f"(sa[24]),        \
        "=f"(sa[25]), "=f"(sa[26]), "=f"(sa[27]), "=f"(sa[28]), "=f"(sa[29]), "=f"(sa[30]),        \
        "=f"(sa[31])                                                                               \
      : "l"(da), "l"(db))

constexpr int PSZ = 6 * 1024;  // 48 query rows x 128 kappa slots, SW128
constexpr int SRS = 528;       // epilogue transpose row stride (fp8)

// Dual-resident MT3 path: only the 48 live query rows are stored.  Each
// 128-byte CKV slab therefore has six 8-row SW128 atoms rather than eight;
// the KPE tail analogously has six SW64 atoms.  WGMMA's unused rows 48..63
// may alias the following slab, but they map only to the ignored fourth warp.
constexpr int Q_SW_LBO = 6 * SW_ATOM;           // 6144 bytes / CKV slab
constexpr int Q_KPE_OFS = 4 * Q_SW_LBO;         // 24576
constexpr int Q_PAGE_SZ = Q_KPE_OFS + 48 * 64;  // 27648
constexpr int Q_GUARD_SZ = 1024;                // ignored KPE rows 48..63
constexpr int Q_STORAGE_SZ = Q_PAGE_SZ + Q_GUARD_SZ;
constexpr int D_NSLOT = 2;
static_assert((Q_SW_LBO % 1024) == 0 && (Q_STORAGE_SZ % 1024) == 0,
              "compact Q slabs must preserve WGMMA swizzle alignment");

__device__ __forceinline__ uint32_t swofs48(int row, int c16) {
  const int lr = row & 7;
  return (uint32_t)((c16 >> 3) * Q_SW_LBO + (row >> 3) * SW_ATOM + lr * 128 +
                    ((c16 & 7) ^ lr) * 16);
}

__device__ __forceinline__ uint64_t gdesc_q(uint32_t addr) {
  return ((uint64_t)((addr & 0x3FFFFu) >> 4)) | ((uint64_t)((uint32_t)Q_SW_LBO >> 4) << 16) |
         ((uint64_t)((uint32_t)SW_ATOM >> 4) << 32) | (1ull << 62);
}

__device__ __forceinline__ void dual_compute_sync() {
  asm volatile("bar.sync 8, 128;" ::: "memory");
}

// O^T = V^T . P^T : the V^T operand lives in registers (built by ldmatrix.trans
// + prmt), P^T is read straight from shared memory by the descriptor, so the
// PV GEMM runs natively in fp8 with no operand up-conversion at all.
#define WGMMA_PV(ACC, A0, A1, A2, A3, DB)                                                    \
  asm volatile(                                                                              \
      "{\n.reg .pred p;\n setp.ne.b32 p, 1, 0;\n"                                            \
      "wgmma.mma_async.sync.aligned.m64n48k32.f16.e4m3.e4m3 "                                \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, "                                            \
      "{%12,%13,%14,%15}, %16, p, 1, 1;\n}\n"                                                \
      : "+r"(ACC[0]), "+r"(ACC[1]), "+r"(ACC[2]), "+r"(ACC[3]), "+r"(ACC[4]), "+r"(ACC[5]),  \
        "+r"(ACC[6]), "+r"(ACC[7]), "+r"(ACC[8]), "+r"(ACC[9]), "+r"(ACC[10]), "+r"(ACC[11]) \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "l"(DB))

// L2-only tile prefetch: costs no shared memory, so the producer can run far
// further ahead in the KV stream than the two-slot smem queue allows.
__device__ __forceinline__ void tma_prefetch_slab(const CUtensorMap* map, int col, int row) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" ::"l"(map), "r"(col),
               "r"(row)
               : "memory");
}

template <bool HALF_ACC, bool FAST_SPLIT, bool ONE_B, bool QLEN6, bool RECT_B, bool LCHECK = true,
          bool REPAIR_CHECK = true, int SLOTS = NSLOT>
__global__ __launch_bounds__(288, 1) void mla_kernel_tmaproducer(
    const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg, const int* __restrict__ btab,
    const int* __restrict__ slen, const float* __restrict__ dqp, const float* __restrict__ dkp,
    float sm_scale, __nv_bfloat16* __restrict__ outp, uint8_t* __restrict__ Op,
    float* __restrict__ Mp, float* __restrict__ Lp, const __grid_constant__ CUtensorMap kv_map,
    const __grid_constant__ CUtensorMap pe_map, int B, int QL, int pps, int ns, int pps_split,
    int QPC, int rect_hint) {
  constexpr int NTHR = 288;
  constexpr int ROWS = 48;

  extern __shared__ __align__(1024) char smraw[];
  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);      // 64 rows, packed page
  uint8_t* Ks = Qs + PAGE_SZ;                           // SLOTS single-page slots
  uint8_t* Ps = Ks + SLOTS * PAGE_SZ;                   // wgmma B operand, SW128
  float* tmx = reinterpret_cast<float*>(Ps + PSZ);      // [48][2]
  float* tsm = tmx + 2 * ROWS;                          // [48][2]
  float* gl = tsm + 2 * ROWS;                           // [48]
  float* rinv = gl + ROWS;                              // [48]
  __half* hc = reinterpret_cast<__half*>(rinv + ROWS);  // [48]
  int* pgt = reinterpret_cast<int*>(hc + 64);           // [MAXPG] page ids
  __shared__ __align__(8) uint64_t tma_full[SLOTS];
  __shared__ __align__(8) uint64_t slot_empty[SLOTS];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = warp >> 2, w = warp & 3;
  const int qc = blockIdx.z;
  __shared__ int repair_needed_cta;
  if constexpr (REPAIR_CHECK) {
    if (tid == 0) repair_needed_cta = 0;
  }
  int b = 0, sp = blockIdx.x;
  if constexpr (RECT_B) {
    // Rectangular B2 launch: each sequence owns ns ranges.  A uniform early
    // return keeps non-live tails out of the Q/TMA prologue while preserving
    // correctness for arbitrary in-scope seq_lens/block-table layouts.
    b = (sp >= ns);
    if (b) sp -= ns;
    // The host hint recovers the packed-generator hot prefix, but never
    // controls correctness: underestimated suffixes are checked against the
    // actual length here, while overestimates hit the later empty-split guard.
    if (b && sp >= rect_hint) {
      const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
      if (sp * pps_split >= ppr_b) return;
    }
  } else if constexpr (!ONE_B) {
    if (B > 1) {
      for (; b < B; ++b) {
        const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
        const int nsp_b = (ppr_b + pps_split - 1) / pps_split;
        if (sp < nsp_b) break;
        sp -= nsp_b;
      }
      if (b == B) return;
    }
  }

  const int qlen = QLEN6 ? 6 : QL;
  const int q0 = QLEN6 ? 0 : qc * QPC;
  const int vrows = FAST_SPLIT ? ROWS : min(ROWS, (qlen - q0) * NH);
  const int rows_tot = qlen * NH;

  // Q depends only on the block index, so it is issued as cp.async before the
  // sequence-length load: the two round trips then overlap instead of chaining.
  {
    const uint8_t* qb = qg + (size_t)(b * qlen + q0) * NH * DIMQ;
    // Only the two consumer warpgroups stage Q.  The producer then owes them
    // no barrier after mbarrier initialization and can start the KV ring while
    // the Q copies are still in flight.
    for (int j = (wg < 2) ? tid : (64 * 36); j < 64 * 36; j += 256) {
      const int row = j / 36, c = j % 36;
      const uint32_t o = (c < 32) ? swofs64(row, c) : (KPE_OFS + swofs64b(row, c - 32));
      if (row < (FAST_SPLIT ? ROWS : vrows))
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sptr(Qs + o)),
                     "l"(qb + (size_t)j * 16)
                     : "memory");
      else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n" ::"r"(sptr(Qs + o)), "l"(qb)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  const int pg0 = sp * pps_split;
  // Gather the producer's page ids in parallel with the sequence-length load.
  // The mbarrier-init CTA barrier below publishes these generic shared stores.
  {
    const int nload = min((pps_split + 1) & ~1, MAXPG);
    const int* bt0 = btab + b * pps;
    if (wg == 2)
      for (int i = tid - 256; i < nload; i += 32) pgt[i] = bt0[min(pg0 + i, pps - 1)];
  }

  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg1 = min(ppr, pg0 + pps_split);

  if (pg0 >= pg1) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    for (int r = tid; r < vrows; r += NTHR) {
      int gi = (b * rows_tot + q0 * NH + r) * ns + sp;
      Mp[gi] = -INFINITY;
      Lp[gi] = 0.f;
    }
    return;
  }

  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < SLOTS; ++i) {
      tma_bar_init(&tma_full[i]);
      tma_bar_init(&slot_empty[i], 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int npg = ((pg1 - pg0 + 1) >> 1) << 1;  // even, last page may repeat

  if (wg == 2) {
    // Five single-page ring slots let the producer run up to three pages ahead
    // of the two the compute warpgroups hold, 50% more KV bytes in flight.
    if (w == 0 && lane == 0) {
      const int* bt = btab + b * pps;
      const uint64_t pol = evict_first_policy();
      int ps = 0, kcnt = 0;
      for (int p = 0; p < npg; ++p) {
        if (p >= SLOTS) tma_bar_wait(&slot_empty[ps], (kcnt - 1) & 1);
        tma_bar_expect(&tma_full[ps], TMA_PAGE_BYTES);
        const int phys = (p < MAXPG) ? pgt[p] : bt[min(pg0 + p, pg1 - 1)];
        uint8_t* dst = Ks + ps * PAGE_SZ;
#pragma unroll
        for (int slab = 0; slab < 4; ++slab)
          tma_load_slab_ef(dst + slab * SW_LBO, &kv_map, slab * 128, phys * PAGE, &tma_full[ps],
                           pol);
        tma_load_slab_ef(dst + KPE_OFS, &pe_map, 0, phys * PAGE, &tma_full[ps], pol);
        if (++ps == SLOTS) {
          ps = 0;
          ++kcnt;
        }
      }
    }
    return;
  }

  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  // gl[] is fully overwritten by the epilogue and hc[] by the first tile's
  // recurrence update before either is read, so no prologue stores are needed.
  // Last prologue barrier, now consumer-only; the producer is already issuing
  // TMA loads and synchronizes with consumers solely through the page queue.
  compute_sync();

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = FAST_SPLIT ? 1.f : dkp[0];

  // PV accumulators: four 64-dim M blocks, m64n48k32 f16 -> 12 regs each
  uint32_t oacc[4][12];
#pragma unroll
  for (int mb = 0; mb < 4; ++mb)
#pragma unroll
    for (int i = 0; i < 12; ++i) oacc[mb][i] = 0;
  // The six real score warps own disjoint halves of the 48 query rows, so
  // update only the recurrence for the rows each warp actually produced.
  // Both the reference and the denominator are tracked redundantly for the two
  // rows a thread owns, so the l-cap shift needs no extra exchange.
  float mq0 = -INFINITY, mq1 = -INFINITY;
  float lq0 = 0.f, lq1 = 0.f;

  // QK rows (row group = w, 3 of the 4 warps cover the 48 real rows)
  const int qrA = w * 16 + (lane >> 2);
  const int qrB = qrA + 8;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - qlen + ((q0 * NH + qrA) / NH) + 1), ptop);
  const int limB = min(max(1, L - qlen + ((q0 * NH + qrB) / NH) + 1), ptop);

  const uint32_t Psb = sptr(Ps);
  const uint32_t Ksb = sptr(Ks);

  // P^T stores: the four probabilities a thread owns inside one 16-token group
  // land on four consecutive kappa slots, so one 32-bit store suffices.
  uint32_t psA[4], psB[4];
  const int cbase = wg * 64 + 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int slot = wg * 64 + 16 * j + 4 * (lane & 3);
    psA[j] = Psb + swofs64(qrA, slot >> 4) + (slot & 15);
    psB[j] = Psb + swofs64(qrB, slot >> 4) + (slot & 15);
  }

  // V^T register-A fragments: warp w of warpgroup wg owns the 16-byte latent
  // column group c16 = wg*16 + mb*4 + w of each 64-dim M block.
  const uint32_t vlane = (uint32_t)((lane >> 3) * SW_ATOM + (lane & 7) * 128);
  uint32_t vadd[4];
#pragma unroll
  for (int mb = 0; mb < 4; ++mb) {
    const int c16 = wg * 16 + mb * 4 + w;
    vadd[mb] = (uint32_t)((c16 >> 3) * SW_LBO + (((c16 & 7) ^ (lane & 7)) << 4));
  }

  const uint64_t dqbase = gdesc_sw(sptr(Qs));
  const uint64_t dqpe = gdesc_sw64(sptr(Qs) + KPE_OFS);
  const uint64_t pdesc = gdesc_sw(Psb);
  uint64_t qd[18];
#pragma unroll
  for (int i = 0; i < 16; ++i) qd[i] = dqbase + (uint64_t)((i >> 2) * 512 + (i & 3) * 2);
  qd[16] = dqpe;
  qd[17] = dqpe + 2;
  // Each tile consumes two independent single-page ring slots; a warpgroup
  // only blocks on the page its own score GEMM reads.
  int s0 = 0, s1 = 1, ph0 = 0, ph1 = 0;
  tma_bar_wait(&tma_full[(wg == 0) ? s0 : s1], (wg == 0) ? ph0 : ph1);

  for (int pg = pg0; pg < pg1; pg += 2) {
    const int sw = (wg == 0) ? s0 : s1;
    const int tok0 = pg * PAGE;
    float sa[32];
    float mxA = -INFINITY, mxB = -INFINITY, sA = 0.f, sB = 0.f;

    {
      uint64_t da = qd[0];
      uint64_t db = gdesc_sw(Ksb + sw * PAGE_SZ);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      WGMMA64_INIT;
      db += 2;
#pragma unroll
      for (int kk = 1; kk < 16; ++kk) {
        da = qd[kk];
        WGMMA64(1);
        db += ((kk & 3) == 3) ? 506 : 2;
      }
      // The 64-byte rope tail lives in its own 64B-swizzled slab.
      da = qd[16];
      db = gdesc_sw64(Ksb + sw * PAGE_SZ + KPE_OFS);
      WGMMA64(1);
      da = qd[17];
      db += 2;
      WGMMA64(1);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }

    if (w < 3) {
      const int remA = limA - tok0, remB = limB - tok0;
      if (remA >= 128 && remB >= 128) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            mxA = fmaxf(mxA, sa[4 * j + e]);
            mxB = fmaxf(mxB, sa[4 * j + 2 + e]);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int c0 = cbase + 8 * j;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float xa = (c0 + e < remA) ? sa[4 * j + e] : -INFINITY;
            float xb = (c0 + e < remB) ? sa[4 * j + 2 + e] : -INFINITY;
            sa[4 * j + e] = xa;
            sa[4 * j + 2 + e] = xb;
            mxA = fmaxf(mxA, xa);
            mxB = fmaxf(mxB, xb);
          }
        }
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
      }
      mxA *= qk_scale;
      mxB *= qk_scale;
    }

    if (w < 3 && (lane & 3) == 0) {
      tmx[qrA * 2 + wg] = mxA;
      tmx[qrB * 2 + wg] = mxB;
    }
    // Exact online-softmax reference.  The two compute warpgroups own disjoint
    // halves of the token tile, so they exchange this tile's row maxima before
    // the probabilities are formed: P then never exceeds 1.  Against a lagged
    // reference a record-setting tile yields P = 2^(s - m_lagged), which
    // saturates E4M3 and overflows the packed-FP16 PV accumulator.
    compute_sync();
    float c0 = 1.f, c1 = 1.f;
    if (w < 3) {
      const float2 qa = *reinterpret_cast<const float2*>(tmx + qrA * 2);
      const float2 qb = *reinterpret_cast<const float2*>(tmx + qrB * 2);
      const float mp0 = mq0, mp1 = mq1;
      mq0 = fmaxf(mq0, fmaxf(qa.x, qa.y));
      mq1 = fmaxf(mq1, fmaxf(qb.x, qb.y));
      c0 = (mp0 == -INFINITY) ? 1.f : exp2f(mp0 - mq0);
      c1 = (mp1 == -INFINITY) ? 1.f : exp2f(mp1 - mq1);
      {
        const float ex0 = LCHECK ? lcap_shift(fmaf(lq0, c0, (float)TT)) : 0.f;
        const float ex1 = LCHECK ? lcap_shift(fmaf(lq1, c1, (float)TT)) : 0.f;
        mq0 += ex0;
        c0 *= exp2i(ex0);
        mq1 += ex1;
        c1 *= exp2i(ex1);
      }
      if ((lane & 3) == 0) {
        const int r = w * 16 + (lane >> 2) + wg * 8;
        hc[r] = __float2half((wg == 0) ? c0 : c1);
      }
      const float eA = (mq0 == -INFINITY) ? 0.f : -mq0;
      const float eB = (mq1 == -INFINITY) ? 0.f : -mq1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float pa[4], pb[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float xa = exp2f(fmaf(sa[8 * j + 4 * (e >> 1) + (e & 1)], qk_scale, eA));
          float xb = exp2f(fmaf(sa[8 * j + 4 * (e >> 1) + 2 + (e & 1)], qk_scale, eB));
          sA += xa;
          sB += xb;
          pa[e] = xa;
          pb[e] = xb;
        }
        uint32_t qa = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]), __NV_SATFINITE,
                                                         __NV_E4M3) |
                      ((uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pa[2], pa[3]), __NV_SATFINITE,
                                                          __NV_E4M3)
                       << 16);
        uint32_t qb = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]), __NV_SATFINITE,
                                                         __NV_E4M3) |
                      ((uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pb[2], pb[3]), __NV_SATFINITE,
                                                          __NV_E4M3)
                       << 16);
        sts32(psA[j], qa);
        sts32(psB[j], qb);
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        sA += __shfl_xor_sync(0xffffffff, sA, d);
        sB += __shfl_xor_sync(0xffffffff, sB, d);
      }
      if ((lane & 3) == 0) {
        tsm[qrA * 2 + wg] = sA;
        tsm[qrB * 2 + wg] = sB;
      }
    }
    // Publishes the probability tile, the per-row rescale and the tile sums.
    compute_sync();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (w < 3) {
      const float2 ua = *reinterpret_cast<const float2*>(tsm + qrA * 2);
      const float2 ub = *reinterpret_cast<const float2*>(tsm + qrB * 2);
      lq0 = fmaf(lq0, c0, ua.x + ua.y);
      lq1 = fmaf(lq1, c1, ub.x + ub.y);
    }
    // O still carries the previous reference; rescale it before this tile's PV.
    {
      uint32_t hcv[6];
#pragma unroll
      for (int j = 0; j < 6; ++j)
        hcv[j] = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(hc) +
                                                    2 * (8 * j + 2 * (lane & 3)));
#pragma unroll
      for (int mb = 0; mb < 4; ++mb)
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          const __half2 h = *reinterpret_cast<__half2*>(&hcv[j]);
          __half2 x0 = __hmul2(*reinterpret_cast<__half2*>(&oacc[mb][2 * j]), h);
          __half2 x1 = __hmul2(*reinterpret_cast<__half2*>(&oacc[mb][2 * j + 1]), h);
          oacc[mb][2 * j] = *reinterpret_cast<uint32_t*>(&x0);
          oacc[mb][2 * j + 1] = *reinterpret_cast<uint32_t*>(&x1);
        }
    }

    // ---- PV: native fp8 wgmma, O^T = V^T . P^T, A = V^T from registers ----
    {
      const uint32_t vbA = Ksb + (uint32_t)(s0 * PAGE_SZ) + vlane;
      const uint32_t vbB = Ksb + (uint32_t)(s1 * PAGE_SZ) + vlane;
      uint32_t Ar[2][16];
      int buf = 0;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        ldm4t(&Ar[0][4 * kk],
              (kk < 2 ? vbA : vbB) + vadd[0] + (uint32_t)((kk & 1) * (4 * SW_ATOM)));
      }
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const uint32_t r0 = Ar[0][4 * kk + 0];
        const uint32_t r1 = Ar[0][4 * kk + 1];
        const uint32_t r2 = Ar[0][4 * kk + 2];
        const uint32_t r3 = Ar[0][4 * kk + 3];
        Ar[0][4 * kk + 0] = prm(r0, r1, 0x6420u);
        Ar[0][4 * kk + 1] = prm(r0, r1, 0x7531u);
        Ar[0][4 * kk + 2] = prm(r2, r3, 0x6420u);
        Ar[0][4 * kk + 3] = prm(r2, r3, 0x7531u);
      }
#pragma unroll
      for (int mb = 0; mb < 4; ++mb) {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
          WGMMA_PV(oacc[mb], Ar[buf][4 * kk + 0], Ar[buf][4 * kk + 1], Ar[buf][4 * kk + 2],
                   Ar[buf][4 * kk + 3], pdesc + 2 * kk);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        // Retire only the previous M block, so this one stays in flight while
        // its successor's V^T fragments are fetched: no pipeline drain per block.
        if (mb >= 1 && mb < 3) asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        if (mb < 3) {
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            ldm4t(&Ar[buf ^ 1][4 * kk],
                  (kk < 2 ? vbA : vbB) + vadd[mb + 1] + (uint32_t)((kk & 1) * (4 * SW_ATOM)));
          }
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const uint32_t r0 = Ar[buf ^ 1][4 * kk + 0];
            const uint32_t r1 = Ar[buf ^ 1][4 * kk + 1];
            const uint32_t r2 = Ar[buf ^ 1][4 * kk + 2];
            const uint32_t r3 = Ar[buf ^ 1][4 * kk + 3];
            Ar[buf ^ 1][4 * kk + 0] = prm(r0, r1, 0x6420u);
            Ar[buf ^ 1][4 * kk + 1] = prm(r0, r1, 0x7531u);
            Ar[buf ^ 1][4 * kk + 2] = prm(r2, r3, 0x6420u);
            Ar[buf ^ 1][4 * kk + 3] = prm(r2, r3, 0x7531u);
          }
          if (mb == 2 && lane == 0) {
            mbar_arrive(&slot_empty[s0]);
            mbar_arrive(&slot_empty[s1]);
          }
        }
        buf ^= 1;
      }
      // Retire the current PV group while waiting for the next KV slot.  This
      // overlaps the TMA-full dependency with tensor-core execution instead
      // of serializing it after a complete WGMMA drain.
      int ns0 = s0 + 2, ns1 = s1 + 2;
      int nph0 = ph0, nph1 = ph1;
      if (ns0 >= SLOTS) {
        ns0 -= SLOTS;
        nph0 ^= 1;
      }
      if (ns1 >= SLOTS) {
        ns1 -= SLOTS;
        nph1 ^= 1;
      }
      if (pg + 2 < pg1) tma_bar_wait(&tma_full[(wg == 0) ? ns0 : ns1], (wg == 0) ? nph0 : nph1);
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      s0 = ns0;
      s1 = ns1;
      ph0 = nph0;
      ph1 = nph1;
    }
  }

  // Packed-FP16 WGMMA is the fast path.  If any final lane is Inf/NaN,
  // publish one CTA-wide exceptional flag; the existing epilogue barrier
  // below makes it visible before deciding whether to replay.  A split of at
  // most two pages is statically safe: 128 tokens * 448 = 57344 < 65504.
  if constexpr (REPAIR_CHECK) {
    uint32_t nonfinite_chain[4] = {0, 0, 0, 0};
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      if (8 * g + 2 * (lane & 3) < vrows) {
#pragma unroll
        for (int mb = 0; mb < 4; ++mb)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint32_t v = oacc[mb][2 * g + j];
            nonfinite_chain[(2 * mb + j) & 3] = hmax_abs_nan(nonfinite_chain[(2 * mb + j) & 3], v);
          }
      }
    }
    const uint32_t nonfinite_bits =
        hmax_abs_nan(hmax_abs_nan(nonfinite_chain[0], nonfinite_chain[1]),
                     hmax_abs_nan(nonfinite_chain[2], nonfinite_chain[3]));
    if (__any_sync(0xffffffffu, half2_nonfinite(nonfinite_bits)) && lane == 0)
      atomicExch(&repair_needed_cta, 1);
  }

  if constexpr (FAST_SPLIT)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  else if (ns > 1)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // ---- epilogue ----
  const bool single = FAST_SPLIT ? false : (ns == 1);
  if (w < 3 && (lane & 3) == 0) {
    const int r = w * 16 + (lane >> 2) + wg * 8;
    const float mrun = (wg == 0) ? mq0 : mq1;
    const float lrun = (wg == 0) ? lq0 : lq1;
    gl[r] = lrun;
    if (!single) {
      int gi = (b * rows_tot + q0 * NH + r) * ns + sp;
      if constexpr (FAST_SPLIT) {
        Mp[gi] = mrun;
        Lp[gi] = lrun;
      } else if (r < vrows) {
        Mp[gi] = mrun;
        Lp[gi] = lrun;
      }
    }
  }
  compute_sync();
  if constexpr (REPAIR_CHECK) {
    if (repair_needed_cta) {
      if (tid < 128)
        for (int j = 0; j < 3 && q0 + 2 * j < qlen; ++j)
          repair_split(qg, kvg, btab, slen, dqp, dkp, sm_scale, outp, Op, Mp, Lp, qlen, pps, ns,
                       pps_split, b, sp, q0 + 2 * j, smraw);
      return;
    }
  }
  if (tid < ROWS) {
    const float l0 = gl[tid];
    rinv[tid] = (l0 > 0.f) ? (single ? o_scale : 1.f) / l0 : 0.f;
  }
  compute_sync();

  const int a3 = lane & 3, l4 = lane >> 2;
  if constexpr (!FAST_SPLIT)
    if (single) {
#pragma unroll
      for (int mb = 0; mb < 4; ++mb) {
        const int c = wg * 256 + mb * 64 + w * 16 + 2 * l4;
#pragma unroll
        for (int g = 0; g < 6; ++g) {
          const int n0 = 8 * g + 2 * a3;
          float2 ri = *reinterpret_cast<const float2*>(rinv + n0);
          float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g]));
          float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g + 1]));
          if (n0 < vrows)
            *reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * qlen * NH + q0 * NH + n0) * CKV +
                                               c) = __floats2bfloat162_rn(f0.x * ri.x, f1.x * ri.x);
          if (n0 + 1 < vrows)
            *reinterpret_cast<__nv_bfloat162*>(
                outp + (size_t)(b * qlen * NH + q0 * NH + n0 + 1) * CKV + c) =
                __floats2bfloat162_rn(f0.y * ri.y, f1.y * ri.y);
        }
      }
      return;
    }

  uint8_t* S = Ks;
#pragma unroll
  for (int mb = 0; mb < 4; ++mb) {
    const int c = wg * 256 + mb * 64 + w * 16 + 2 * l4;
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      const int n0 = 8 * g + 2 * a3;
      float2 ri = *reinterpret_cast<const float2*>(rinv + n0);
      float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g]));
      float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g + 1]));
      sts16(sptr(S + n0 * SRS + c), __nv_cvt_float2_to_fp8x2(make_float2(f0.x * ri.x, f1.x * ri.x),
                                                             __NV_SATFINITE, __NV_E4M3));
      sts16(sptr(S + (n0 + 1) * SRS + c),
            __nv_cvt_float2_to_fp8x2(make_float2(f0.y * ri.y, f1.y * ri.y), __NV_SATFINITE,
                                     __NV_E4M3));
    }
  }
  compute_sync();
  {
    const size_t obase = (size_t)((b * ns + sp) * rows_tot + q0 * NH) * CKV;
#pragma unroll 1
    for (int ch = tid; ch < vrows * 32; ch += 256) {
      const int row = ch >> 5, cc = (ch & 31) << 4;
      uint4 v = *reinterpret_cast<const uint4*>(S + row * SRS + cc);
      *reinterpret_cast<uint4*>(Op + obase + (size_t)row * CKV + cc) = v;
    }
  }
}

// Dual-resident long-context path.  A single consumer warpgroup handles all
// 48 live query rows and all 512 output columns, while a dedicated producer
// warp keeps a two-page TMA ring full.  Five warps plus the 110208-byte shared
// footprint permit two CTAs per H200 SM without a 128-register launch cap.
template <bool HALF_ACC, bool QLEN6, bool LCHECK = true, bool REPAIR_CHECK = true>
__global__ __launch_bounds__(160, 2) void mla_kernel_dual(
    const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg, const int* __restrict__ btab,
    const int* __restrict__ slen, const float* __restrict__ dqp, const float* __restrict__ dkp,
    float sm_scale, __nv_bfloat16* __restrict__ outp, uint8_t* __restrict__ Op,
    float* __restrict__ Mp, float* __restrict__ Lp, const __grid_constant__ CUtensorMap kv_map,
    const __grid_constant__ CUtensorMap pe_map, int B, int QL, int pps, int ns, int pps_split,
    int QPC) {
  constexpr int NTHR = 160;
  constexpr int ROWS = 48;

  extern __shared__ __align__(1024) char smraw[];
  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);
  // WGMMA still addresses an m64 Q operand.  The final KPE slab's ignored
  // rows 48..63 land in this initialized guard instead of racing a KV TMA.
  uint8_t* Qguard = Qs + Q_PAGE_SZ;
  uint8_t* Ks = Qs + Q_STORAGE_SZ;
  uint8_t* Ps = Ks + D_NSLOT * PAGE_SZ;
  float* gl = reinterpret_cast<float*>(Ps + PSZ);       // [48]
  float* rinv = gl + ROWS;                              // [48]
  __half* hc = reinterpret_cast<__half*>(rinv + ROWS);  // [64]
  int* pgt = reinterpret_cast<int*>(hc + 64);           // [MAXPG]
  __shared__ __align__(8) uint64_t tma_full[D_NSLOT];
  __shared__ __align__(8) uint64_t slot_empty[D_NSLOT];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = warp >> 2, w = warp & 3;
  const int qc = blockIdx.z;
  __shared__ int repair_needed_cta;
  if constexpr (REPAIR_CHECK) {
    if (tid == 0) repair_needed_cta = 0;
  }
  int b = 0, sp = blockIdx.x;
  for (; b < B; ++b) {
    const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
    const int nsp_b = (ppr_b + pps_split - 1) / pps_split;
    if (sp < nsp_b) break;
    sp -= nsp_b;
  }
  if (b == B) return;

  const int qlen = QLEN6 ? 6 : QL;
  const int q0 = QLEN6 ? 0 : qc * QPC;
  const int vrows = min(ROWS, (qlen - q0) * NH);
  const int rows_tot = qlen * NH;

  // Consumer threads stage compact Q while the producer gathers page IDs.
  // Their disjoint prologues meet at barrier initialization, after which TMA
  // can start before the consumer's cp.async copies retire.
  {
    const uint8_t* qb = qg + (size_t)(b * qlen + q0) * NH * DIMQ;
    for (int j = (wg == 0) ? tid : ROWS * 36; j < ROWS * 36; j += 128) {
      const int row = j / 36, c = j % 36;
      const uint32_t o = (c < 32) ? swofs48(row, c) : (Q_KPE_OFS + swofs64b(row, c - 32));
      if (row < vrows)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sptr(Qs + o)),
                     "l"(qb + (size_t)j * 16)
                     : "memory");
      else
        *reinterpret_cast<uint4*>(Qs + o) = make_uint4(0, 0, 0, 0);
    }
    for (int j = (wg == 0) ? tid : Q_GUARD_SZ / 16; j < Q_GUARD_SZ / 16; j += 128)
      *reinterpret_cast<uint4*>(Qguard + 16 * j) = make_uint4(0, 0, 0, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg0 = sp * pps_split;
  const int pg1 = min(ppr, pg0 + pps_split);

  if (pg0 >= pg1) {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    for (int r = tid; r < vrows; r += NTHR) {
      const int gi = (b * rows_tot + q0 * NH + r) * ns + sp;
      Mp[gi] = -INFINITY;
      Lp[gi] = 0.f;
    }
    return;
  }

  const int npg = pg1 - pg0;
  if (wg == 1) {
    const int nload = min(npg, MAXPG);
    const int* bt = btab + b * pps;
    for (int i = tid - 128; i < nload; i += 32) pgt[i] = bt[pg0 + i];
  }
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < D_NSLOT; ++i) {
      tma_bar_init(&tma_full[i]);
      tma_bar_init(&slot_empty[i], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (wg == 1) {
    if (w == 0 && lane == 0) {
      const int* bt = btab + b * pps;
      int ps = 0, cycle = 0;
      for (int p = 0; p < npg; ++p) {
        if (p >= D_NSLOT) tma_bar_wait(&slot_empty[ps], (cycle - 1) & 1);
        tma_bar_expect(&tma_full[ps], TMA_PAGE_BYTES);
        const int phys = (p < MAXPG) ? pgt[p] : bt[pg0 + p];
        uint8_t* dst = Ks + ps * PAGE_SZ;
#pragma unroll
        for (int slab = 0; slab < 4; ++slab)
          tma_load_slab(dst + slab * SW_LBO, &kv_map, slab * 128, phys * PAGE, &tma_full[ps]);
        tma_load_slab(dst + KPE_OFS, &pe_map, 0, phys * PAGE, &tma_full[ps]);
        if (++ps == D_NSLOT) {
          ps = 0;
          ++cycle;
        }
      }
    }
    return;
  }

  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  dual_compute_sync();

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = dkp[0];

  // Eight 64-column blocks cover the complete 512-dimensional latent output.
  uint32_t oacc[8][12];
#pragma unroll
  for (int mb = 0; mb < 8; ++mb)
#pragma unroll
    for (int i = 0; i < 12; ++i) oacc[mb][i] = 0;

  // Each four-lane subgroup tracks its two query rows.  The state is
  // duplicated within the subgroup, avoiding shared-memory reductions.
  float mrun0 = -INFINITY, mrun1 = -INFINITY;
  float lrun0 = 0.f, lrun1 = 0.f;
  const int qrA = w * 16 + (lane >> 2);
  const int qrB = qrA + 8;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - qlen + ((q0 * NH + qrA) / NH) + 1), ptop);
  const int limB = min(max(1, L - qlen + ((q0 * NH + qrB) / NH) + 1), ptop);

  const uint32_t Psb = sptr(Ps);
  const uint32_t Ksb = sptr(Ks);
  uint32_t psA[4], psB[4];
  const int cbase = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int slot = 16 * j + 4 * (lane & 3);
    psA[j] = Psb + swofs64(qrA, slot >> 4) + (slot & 15);
    psB[j] = Psb + swofs64(qrB, slot >> 4) + (slot & 15);
  }

  const uint32_t vlane = (uint32_t)((lane >> 3) * SW_ATOM + (lane & 7) * 128);
  uint32_t vadd[8];
#pragma unroll
  for (int mb = 0; mb < 8; ++mb) {
    const int c16 = mb * 4 + w;
    vadd[mb] = (uint32_t)((c16 >> 3) * SW_LBO + (((c16 & 7) ^ (lane & 7)) << 4));
  }

  const uint64_t dqbase = gdesc_q(sptr(Qs));
  const uint64_t dqpe = gdesc_sw64(sptr(Qs) + Q_KPE_OFS);
  const uint64_t pdesc = gdesc_sw(Psb);
  int sw = 0, phase = 0;
  tma_bar_wait(&tma_full[sw], phase);

  for (int pg = pg0; pg < pg1; ++pg) {
    const int tok0 = pg * PAGE;
    float sa[32];
    float mxA = -INFINITY, mxB = -INFINITY;
    float sA = 0.f, sB = 0.f;

    {
      uint64_t da = dqbase;
      uint64_t db = gdesc_sw(Ksb + sw * PAGE_SZ);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      WGMMA64_INIT;
      da += 2;
      db += 2;
#pragma unroll
      for (int kk = 1; kk < 16; ++kk) {
        WGMMA64(1);
        // Compact Q advances 6144 bytes per 128-byte slab; KV retains its
        // full 8192-byte stride.  Descriptor units are 16 bytes.
        if ((kk & 3) == 3) {
          da += 378;
          db += 506;
        } else {
          da += 2;
          db += 2;
        }
      }
      da = dqpe;
      db = gdesc_sw64(Ksb + sw * PAGE_SZ + KPE_OFS);
      WGMMA64(1);
      da += 2;
      db += 2;
      WGMMA64(1);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");

      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }

    if (w < 3) {
      const int remA = limA - tok0, remB = limB - tok0;
      if (remA >= PAGE && remB >= PAGE) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            mxA = fmaxf(mxA, sa[4 * j + e]);
            mxB = fmaxf(mxB, sa[4 * j + 2 + e]);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int c0 = cbase + 8 * j;
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float xa = (c0 + e < remA) ? sa[4 * j + e] : -INFINITY;
            const float xb = (c0 + e < remB) ? sa[4 * j + 2 + e] : -INFINITY;
            sa[4 * j + e] = xa;
            sa[4 * j + 2 + e] = xb;
            mxA = fmaxf(mxA, xa);
            mxB = fmaxf(mxB, xb);
          }
        }
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
      }
      mxA *= qk_scale;
      mxB *= qk_scale;
      // Exact online-softmax reference: a four-lane subgroup already holds the
      // complete row max of this page, so P is quantized against
      // max(running, tile) and never exceeds 1.  Against a lagged reference a
      // record-setting page yields P = 2^(s - m_lagged), which saturates E4M3
      // and overflows the packed-FP16 PV accumulator on long-context rows.
      float mn0 = fmaxf(mrun0, mxA), mn1 = fmaxf(mrun1, mxB);
      float corr0 = (mrun0 == -INFINITY) ? 1.f : exp2f(mrun0 - mn0);
      float corr1 = (mrun1 == -INFINITY) ? 1.f : exp2f(mrun1 - mn1);
      {
        const float ex0 = LCHECK ? lcap_shift(fmaf(lrun0, corr0, (float)PAGE)) : 0.f;
        const float ex1 = LCHECK ? lcap_shift(fmaf(lrun1, corr1, (float)PAGE)) : 0.f;
        mn0 += ex0;
        corr0 *= exp2i(ex0);
        mn1 += ex1;
        corr1 *= exp2i(ex1);
      }
      mrun0 = mn0;
      mrun1 = mn1;
      if ((lane & 3) == 0) {
        hc[qrA] = __float2half(corr0);
        hc[qrB] = __float2half(corr1);
      }
      const float eA = (mn0 == -INFINITY) ? 0.f : -mn0;
      const float eB = (mn1 == -INFINITY) ? 0.f : -mn1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float pa[4], pb[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float xa = exp2f(fmaf(sa[8 * j + 4 * (e >> 1) + (e & 1)], qk_scale, eA));
          const float xb = exp2f(fmaf(sa[8 * j + 4 * (e >> 1) + 2 + (e & 1)], qk_scale, eB));
          sA += xa;
          sB += xb;
          pa[e] = xa;
          pb[e] = xb;
        }
        const uint32_t qa = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]),
                                                               __NV_SATFINITE, __NV_E4M3) |
                            ((uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pa[2], pa[3]),
                                                                __NV_SATFINITE, __NV_E4M3)
                             << 16);
        const uint32_t qb = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]),
                                                               __NV_SATFINITE, __NV_E4M3) |
                            ((uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(pb[2], pb[3]),
                                                                __NV_SATFINITE, __NV_E4M3)
                             << 16);
        sts32(psA[j], qa);
        sts32(psB[j], qb);
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        sA += __shfl_xor_sync(0xffffffff, sA, d);
        sB += __shfl_xor_sync(0xffffffff, sB, d);
      }
      lrun0 = lrun0 * corr0 + sA;
      lrun1 = lrun1 * corr1 + sB;
    }
    dual_compute_sync();
    // Publish the probability tile from the generic shared-memory proxy to
    // the async proxy used by WGMMA before any consumer issues PV.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    // O carries the previous reference; rescale it before this page's PV.
    {
      uint32_t hcv[6];
#pragma unroll
      for (int j = 0; j < 6; ++j)
        hcv[j] = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(hc) +
                                                    2 * (8 * j + 2 * (lane & 3)));
#pragma unroll
      for (int mb = 0; mb < 8; ++mb)
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          const __half2 h = *reinterpret_cast<__half2*>(&hcv[j]);
          __half2 x0 = __hmul2(*reinterpret_cast<__half2*>(&oacc[mb][2 * j]), h);
          __half2 x1 = __hmul2(*reinterpret_cast<__half2*>(&oacc[mb][2 * j + 1]), h);
          oacc[mb][2 * j] = *reinterpret_cast<uint32_t*>(&x0);
          oacc[mb][2 * j + 1] = *reinterpret_cast<uint32_t*>(&x1);
        }
    }
    // Every fold has consumed hc before the next page overwrites it.
    dual_compute_sync();
    // One page contributes two k32 native-FP8 PV operations per output block.
    {
      const uint32_t vb = Ksb + (uint32_t)(sw * PAGE_SZ) + vlane;
      uint32_t Ar[2][8];
      int buf = 0;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t r[4];
        ldm4t(r, vb + vadd[0] + (uint32_t)(kk * 4 * SW_ATOM));
        Ar[0][4 * kk + 0] = prm(r[0], r[1], 0x6420u);
        Ar[0][4 * kk + 1] = prm(r[0], r[1], 0x7531u);
        Ar[0][4 * kk + 2] = prm(r[2], r[3], 0x6420u);
        Ar[0][4 * kk + 3] = prm(r[2], r[3], 0x7531u);
      }
#pragma unroll
      for (int mb = 0; mb < 8; ++mb) {
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
          WGMMA_PV(oacc[mb], Ar[buf][4 * kk + 0], Ar[buf][4 * kk + 1], Ar[buf][4 * kk + 2],
                   Ar[buf][4 * kk + 3], pdesc + 2 * kk);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        if (mb >= 1 && mb < 7) asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        if (mb < 7) {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            uint32_t r[4];
            ldm4t(r, vb + vadd[mb + 1] + (uint32_t)(kk * 4 * SW_ATOM));
            Ar[buf ^ 1][4 * kk + 0] = prm(r[0], r[1], 0x6420u);
            Ar[buf ^ 1][4 * kk + 1] = prm(r[0], r[1], 0x7531u);
            Ar[buf ^ 1][4 * kk + 2] = prm(r[2], r[3], 0x6420u);
            Ar[buf ^ 1][4 * kk + 3] = prm(r[2], r[3], 0x7531u);
          }
          if (mb == 6 && lane == 0) mbar_arrive(&slot_empty[sw]);
        }
        buf ^= 1;
      }
      int nsw = sw + 1, nphase = phase;
      if (nsw == D_NSLOT) {
        nsw = 0;
        nphase ^= 1;
      }
      if (pg + 1 < pg1) tma_bar_wait(&tma_full[nsw], nphase);
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      sw = nsw;
      phase = nphase;
    }
  }

  if constexpr (REPAIR_CHECK) {
    uint32_t nonfinite_chain[4] = {0, 0, 0, 0};
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      if (8 * g + 2 * (lane & 3) < vrows) {
#pragma unroll
        for (int mb = 0; mb < 8; ++mb)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint32_t v = oacc[mb][2 * g + j];
            nonfinite_chain[(2 * mb + j) & 3] = hmax_abs_nan(nonfinite_chain[(2 * mb + j) & 3], v);
          }
      }
    }
    const uint32_t nonfinite_bits =
        hmax_abs_nan(hmax_abs_nan(nonfinite_chain[0], nonfinite_chain[1]),
                     hmax_abs_nan(nonfinite_chain[2], nonfinite_chain[3]));
    if (__any_sync(0xffffffffu, half2_nonfinite(nonfinite_bits)) && lane == 0)
      atomicExch(&repair_needed_cta, 1);
  }

  if (ns > 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const bool single = (ns == 1);
  if (w < 3 && (lane & 3) == 0) {
    gl[qrA] = lrun0;
    gl[qrB] = lrun1;
    if (!single) {
      const int giA = (b * rows_tot + q0 * NH + qrA) * ns + sp;
      const int giB = (b * rows_tot + q0 * NH + qrB) * ns + sp;
      if (qrA < vrows) {
        Mp[giA] = mrun0;
        Lp[giA] = lrun0;
      }
      if (qrB < vrows) {
        Mp[giB] = mrun1;
        Lp[giB] = lrun1;
      }
    }
  }
  dual_compute_sync();
  if constexpr (REPAIR_CHECK) {
    if (repair_needed_cta) {
      for (int j = 0; j < 3 && q0 + 2 * j < qlen; ++j)
        repair_split(qg, kvg, btab, slen, dqp, dkp, sm_scale, outp, Op, Mp, Lp, qlen, pps, ns,
                     pps_split, b, sp, q0 + 2 * j, smraw);
      return;
    }
  }
  if (tid < ROWS) {
    const float l0 = gl[tid];
    rinv[tid] = (l0 > 0.f) ? (single ? o_scale : 1.f) / l0 : 0.f;
  }
  dual_compute_sync();

  const int a3 = lane & 3, l4 = lane >> 2;
  if (single) {
#pragma unroll
    for (int mb = 0; mb < 8; ++mb) {
      const int c = mb * 64 + w * 16 + 2 * l4;
#pragma unroll
      for (int g = 0; g < 6; ++g) {
        const int n0 = 8 * g + 2 * a3;
        const float2 ri = *reinterpret_cast<const float2*>(rinv + n0);
        const float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g]));
        const float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g + 1]));
        if (n0 < vrows)
          *reinterpret_cast<__nv_bfloat162*>(outp + (size_t)(b * qlen * NH + q0 * NH + n0) * CKV +
                                             c) = __floats2bfloat162_rn(f0.x * ri.x, f1.x * ri.x);
        if (n0 + 1 < vrows)
          *reinterpret_cast<__nv_bfloat162*>(outp +
                                             (size_t)(b * qlen * NH + q0 * NH + n0 + 1) * CKV + c) =
              __floats2bfloat162_rn(f0.y * ri.y, f1.y * ri.y);
      }
    }
    return;
  }

  uint8_t* S = Ks;
#pragma unroll
  for (int mb = 0; mb < 8; ++mb) {
    const int c = mb * 64 + w * 16 + 2 * l4;
#pragma unroll
    for (int g = 0; g < 6; ++g) {
      const int n0 = 8 * g + 2 * a3;
      const float2 ri = *reinterpret_cast<const float2*>(rinv + n0);
      const float2 f0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g]));
      const float2 f1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[mb][2 * g + 1]));
      sts16(sptr(S + n0 * SRS + c), __nv_cvt_float2_to_fp8x2(make_float2(f0.x * ri.x, f1.x * ri.x),
                                                             __NV_SATFINITE, __NV_E4M3));
      sts16(sptr(S + (n0 + 1) * SRS + c),
            __nv_cvt_float2_to_fp8x2(make_float2(f0.y * ri.y, f1.y * ri.y), __NV_SATFINITE,
                                     __NV_E4M3));
    }
  }
  dual_compute_sync();
  const size_t obase = (size_t)((b * ns + sp) * rows_tot + q0 * NH) * CKV;
#pragma unroll 1
  // The producer warp has returned; only the 128-thread consumer warpgroup
  // participates in this cooperative transpose copy.
  for (int ch = tid; ch < vrows * 32; ch += 128) {
    const int row = ch >> 5, cc = (ch & 31) << 4;
    const uint4 v = *reinterpret_cast<const uint4*>(S + row * SRS + cc);
    *reinterpret_cast<uint4*>(Op + obase + (size_t)row * CKV + cc) = v;
  }
}

#define CWARPS 16
#define C1WARPS 16
#define CSHORTWARPS 8
template <int GROUP, bool PREFETCH, int NWARPS, int VMAX>
__global__ __launch_bounds__(32 * NWARPS, 1) void combine_kernel(
    const uint8_t* __restrict__ Op, const float* __restrict__ Mp, const float* __restrict__ Lp,
    const float* __restrict__ dkp, const int* __restrict__ slen, __nv_bfloat16* __restrict__ outp,
    int ns, int rows_tot, int pps_split) {
  constexpr int NT = 32 * NWARPS;
  constexpr int PARTS = NWARPS / GROUP;
  const int b = blockIdx.z;
  const int r = blockIdx.x;
  const int row = b * rows_tot + r;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wp = tid >> 5;
  const int chunk = blockIdx.y * GROUP + wp / PARTS;
  const int part = wp % PARTS;

  // A nonpositive split span marks the single-sequence case, where every
  // logical split is populated and no sequence-length round trip is needed.
  int nsb = ns;
  if (pps_split > 0) {
    const int span = PAGE * pps_split;
    if constexpr (PREFETCH) {
      nsb = (slen[b] + span - 1) / span;
      if (nsb > ns) nsb = ns;
      if (nsb < 1) nsb = 1;
    } else {
      if (lane == 0) {
        nsb = (slen[b] + span - 1) / span;
        if (nsb > ns) nsb = ns;
        if (nsb < 1) nsb = 1;
      }
      nsb = __shfl_sync(0xffffffffu, nsb, 0);
    }
  }

  const int c = chunk * 128 + lane * 4;
  const uint8_t* base = Op + (size_t)(b * ns) * rows_tot * CKV + (size_t)r * CKV + c;
  const size_t stride = (size_t)rows_tot * CKV;
  // Resolve launch indices and immutable-input addressing before waiting on
  // the predecessor.  This waits for full prerequisite-grid completion and
  // global-memory visibility, not merely the earlier programmatic launch
  // trigger.  No split partial is read until this ordering point.
  cudaGridDependencySynchronize();
  uint32_t vraw[PREFETCH ? VMAX : 1];
  if constexpr (PREFETCH) {
#pragma unroll
    for (int i = 0; i < VMAX; ++i) {
      int s = part + i * PARTS;
      vraw[i] = (s < nsb) ? *reinterpret_cast<const uint32_t*>(base + (size_t)s * stride) : 0u;
    }
  }

  __shared__ float sred[NWARPS][128];
  __shared__ float smw[NWARPS];
  __shared__ float sls[NWARPS];

  // Each warp normalizes only the splits it reduces. The final cross-warp
  // pass reconciles the independent reference maxima, removing the initial
  // warp-0 weight-table phase and one CTA barrier.
  const size_t stat_base = (size_t)(b * rows_tot + r) * ns;
  float mv0 = -INFINITY, lv0 = 0.f;
  {
    const int sx = part + lane * PARTS;
    if (sx < nsb) {
      mv0 = Mp[stat_base + sx];
      lv0 = Lp[stat_base + sx];
    }
  }
  float mw = mv0;
  for (int sx = part + (32 + lane) * PARTS; sx < nsb; sx += 32 * PARTS)
    mw = fmaxf(mw, Mp[stat_base + sx]);
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, d));

  float wv = (mv0 > -INFINITY) ? exp2f(mv0 - mw) * lv0 : 0.f;
  float lsum = wv;
  for (int sx = part + (32 + lane) * PARTS; sx < nsb; sx += 32 * PARTS) {
    const float mm = Mp[stat_base + sx];
    lsum += (mm > -INFINITY) ? exp2f(mm - mw) * Lp[stat_base + sx] : 0.f;
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, d);

  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  if constexpr (PREFETCH) {
#pragma unroll
    for (int i = 0; i < VMAX; ++i) {
      const int sx = part + i * PARTS;
      if (sx < nsb) {
        const float ww = __shfl_sync(0xffffffffu, wv, i);
        // A planned-but-empty split publishes M=-inf/L=0 and never writes its
        // O partial; skip zero-weight splits entirely so an uninitialized fp8
        // byte cannot poison the sum through fmaf(0, NaN, acc).
        if (ww != 0.f) {
          __nv_fp8x4_e4m3 v;
          v.__x = vraw[i];
          const float4 a = static_cast<float4>(v);
          acc.x = fmaf(ww, a.x, acc.x);
          acc.y = fmaf(ww, a.y, acc.y);
          acc.z = fmaf(ww, a.z, acc.z);
          acc.w = fmaf(ww, a.w, acc.w);
        }
      }
    }
  }
  {
    int i = PREFETCH ? VMAX : 0;
    for (int sx = part + i * PARTS; sx < nsb; sx += PARTS, ++i) {
      float ww;
      if (i < 32)
        ww = __shfl_sync(0xffffffffu, wv, i);
      else {
        const float mm = Mp[stat_base + sx];
        ww = (mm > -INFINITY) ? exp2f(mm - mw) * Lp[stat_base + sx] : 0.f;
      }
      if (ww != 0.f) {
        __nv_fp8x4_e4m3 v;
        v.__x = *reinterpret_cast<const uint32_t*>(base + (size_t)sx * stride);
        const float4 a = static_cast<float4>(v);
        acc.x = fmaf(ww, a.x, acc.x);
        acc.y = fmaf(ww, a.y, acc.y);
        acc.z = fmaf(ww, a.z, acc.z);
        acc.w = fmaf(ww, a.w, acc.w);
      }
    }
  }

  *reinterpret_cast<float4*>(&sred[wp][lane * 4]) = acc;
  if (lane == 0) {
    smw[wp] = mw;
    sls[wp] = lsum;
  }
  __syncthreads();
  if (tid < 128 * GROUP) {
    const int out_chunk = blockIdx.y * GROUP + tid / 128;
    const int out_col = tid & 127;
    const int first_warp = (tid / 128) * PARTS;
    float m = -INFINITY;
#pragma unroll
    for (int w = 0; w < PARTS; ++w) m = fmaxf(m, smw[first_warp + w]);
    float o = 0.f, den = 0.f;
#pragma unroll
    for (int w = 0; w < PARTS; ++w) {
      const float wm = smw[first_warp + w];
      const float sc = (wm > -INFINITY) ? exp2f(wm - m) : 0.f;
      o = fmaf(sred[first_warp + w][out_col], sc, o);
      den = fmaf(sls[first_warp + w], sc, den);
    }
    outp[(size_t)row * CKV + out_chunk * 128 + out_col] =
        __float2bfloat16((den > 0.f) ? o * (dkp[0] / den) : 0.f);
  }
}

template <int GROUP, bool PREFETCH, int NWARPS, int VMAX, typename... Args>
static void launch_combine(dim3 grid, int nthr, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(nthr, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, combine_kernel<GROUP, PREFETCH, NWARPS, VMAX>, args...);
}

thread_local int g_sm = 0;

struct TmapCache {
  alignas(64) CUtensorMap kv{};
  alignas(64) CUtensorMap pe{};
  const void* base = nullptr;
  unsigned long long rows = 0;
  bool no_promote = false;
  bool valid = false;
};
thread_local TmapCache g_tmap;

}  // namespace

constexpr int kMaxDevices = 128;
int g_sm_count[kMaxDevices] = {0};

namespace {

struct MlaSplitPlan {
  int MT;
  int QPC;
  int nqc;
  int ns;
  int pps_split;
  int nblocks;
  int rows_tot;
  int rect_hint;
};

// Verbatim split arithmetic of the measured launcher (`run`), parameterised
// on the plan-time facts it read from its tensors.
MlaSplitPlan mla_decode_fp8_split_plan(int B, int QL, int pps, int total_pages, int sm) {
  using std::max;
  MlaSplitPlan P{};
  const int maxsp = (pps + 1) / 2;  // tiles cover 2 pages
  int MT = (QL + 1) / 2;
  if (MT > 3) MT = 3;
  if (MT < 1) MT = 1;
  if (B == 1 && QL < 5 && maxsp * ((QL + 1) / 2) <= 2 * sm) MT = 1;
  int QPC = 2 * MT;
  int nqc = (QL + QPC - 1) / QPC;

  int ns, pps_split, nblocks, rect_hint = 0;
  if (B == 1) {
    ns = sm;
    if (ns > maxsp) ns = maxsp;
    if (ns > 256) ns = 256;
    if (ns < 1) ns = 1;
    pps_split = (((pps + ns - 1) / ns) + 1) & ~1;
    ns = (pps + pps_split - 1) / pps_split;
    nblocks = ns;
  } else {
    // total_pages is a parameter
    if (B >= 5 && MT == 3) {
      // The compact five-warp kernel permits two CTAs per H200 SM.  Its TMA
      // ring consumes one page per iteration, so split width need not be even.
      // Reserve only one CTA slot per pair of sequences.  This keeps the
      // compact kernel within its two-CTA/SM capacity while filling the last
      // wave more completely for B6/B10.
      const int target = max(1, 2 * sm - (B + 1) / 2);
      pps_split = max(1, (total_pages + target - 1) / target);
      ns = (pps + pps_split - 1) / pps_split;
      if (ns > 256) {
        // Keep the dispatcher's 256-split workspace bound: with one-page
        // splits ns otherwise reaches `target` (up to 2*sm), e.g. a ragged
        // B>=5 batch whose longest sequence nearly fills the page pool.
        ns = 256;
        pps_split = (pps + ns - 1) / ns;
        ns = (pps + pps_split - 1) / pps_split;
      }
    } else {
      // Preserve the champion's two-page planner for B2-B4 (and lower-Q MTs).
      // Roughly B/2 extra ranges compensate the per-sequence ceil.
      const int target = max(1, sm - B / 2 + 1);
      pps_split = (max(1, (total_pages + target - 1) / target) + 1) & ~1;
      ns = (pps + pps_split - 1) / pps_split;
      if (ns > 256) {
        ns = 256;
        pps_split = (((pps + ns - 1) / ns) + 1) & ~1;
        ns = (pps + pps_split - 1) / pps_split;
      }
    }
    if (B == 2) {
      // Keep the direct rectangular mapping valid for every in-scope pair of
      // sequence lengths; non-live tails return before staging Q or issuing TMA.
      const int ppr1_hint = max(1, total_pages - pps);
      rect_hint = (ppr1_hint + pps_split - 1) / pps_split;
      nblocks = B * ns;
    } else {
      nblocks = B * ns;
    }
  }

  P.MT = MT;
  P.QPC = QPC;
  P.nqc = nqc;
  P.ns = ns;
  P.pps_split = pps_split;
  P.nblocks = nblocks;
  P.rows_tot = QL * NH;
  P.rect_hint = rect_hint;
  return P;
}

size_t mla_decode_fp8_ws_bytes(const MlaSplitPlan& P, int B) {
  if (P.ns <= 1) return 0;
  const size_t nrow = (size_t)B * P.ns * P.rows_tot;
  return nrow * CKV + 2 * nrow * sizeof(float);
}

}  // namespace

// One-time per-device SM-count query + dynamic shared memory opt-in for the
// MT<=2 kernel instantiations. Outside CUDA-graph capture, before the first
// `run` on the device. (The warp-group/dual/ring variants arm their own
// attribute on first eager call — see the header note.)
void mla_decode_fp8_paged_sm90_init(int64_t device_id) {
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  int sm = 0;
  cudaError_t st =
      cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, static_cast<int>(device_id));
  TVM_FFI_ICHECK(st == cudaSuccess && sm > 0)
      << "mla_decode_fp8_paged_sm90: SM-count query failed: " << cudaGetErrorString(st);
  g_sm_count[device_id] = sm;
  for (int MT = 1; MT <= 2; ++MT) {
    const int ROWS = 16 * MT;
    const size_t smem =
        (size_t)ROWS * KSTR + (size_t)2 * TT * KSTR + (size_t)ROWS * PSTR + (size_t)8 * ROWS * 4;
    cudaError_t s1, s2;
    if (MT == 1) {
      s1 = cudaFuncSetAttribute(mla_kernel<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
      s2 = cudaFuncSetAttribute(mla_kernel<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
    } else {
      s1 = cudaFuncSetAttribute(mla_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
      s2 = cudaFuncSetAttribute(mla_kernel<2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
    }
    TVM_FFI_ICHECK(s1 == cudaSuccess && s2 == cudaSuccess)
        << "mla_decode_fp8_paged_sm90: cudaFuncSetAttribute failed";
  }
}

// Workspace bytes the dispatcher must plan-allocate for a
// (batch, q_len, pages_per_seq, total_pages) plan on `device_id`.
int64_t mla_decode_fp8_paged_sm90_plan_workspace_bytes(int64_t batch_size, int64_t q_len,
                                                       int64_t pages_per_seq, int64_t total_pages,
                                                       int64_t device_id) {
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  TVM_FFI_ICHECK_GT(g_sm_count[device_id], 0)
      << "mla_decode_fp8_paged_sm90: init(device_id) must run before "
         "plan_workspace_bytes";
  const MlaSplitPlan P = mla_decode_fp8_split_plan(
      static_cast<int>(batch_size), static_cast<int>(q_len), static_cast<int>(pages_per_seq),
      static_cast<int>(total_pages), g_sm_count[device_id]);
  return (int64_t)mla_decode_fp8_ws_bytes(P, static_cast<int>(batch_size));
}

// Launch the measured kernel sequence on the caller's current stream over
// the verbatim split plan, then the measured combine selection when the
// plan splits. No allocation, no sync; see the header for the per-class
// one-eager-warm-up capture contract.
void mla_decode_fp8_paged_sm90_run(TensorView q, TensorView kv_cache, TensorView block_table,
                                   TensorView seq_lens, TensorView q_descale, TensorView k_descale,
                                   double sm_scale, TensorView out, TensorView workspace) {
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(q_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(k_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_CUDA(kv_cache);
  CHECK_INPUT_TYPE(kv_cache, dl_float8_e4m3fn);
  CHECK_DIM(4, q);
  CHECK_DIM(4, out);
  CHECK_DIM(2, block_table);

  const int B = static_cast<int>(q.size(0));
  const int QL = static_cast<int>(q.size(1));
  const int pps = static_cast<int>(block_table.size(1));

  TVM_FFI_ICHECK_EQ(q.size(2), NH);
  TVM_FFI_ICHECK_EQ(q.size(3), DIMQ);
  TVM_FFI_ICHECK_EQ(out.size(0), B);
  TVM_FFI_ICHECK_EQ(out.size(1), QL);
  TVM_FFI_ICHECK_EQ(out.size(2), NH);
  TVM_FFI_ICHECK_EQ(out.size(3), CKV);
  TVM_FFI_ICHECK_EQ(block_table.size(0), B);
  TVM_FFI_ICHECK_GE(seq_lens.numel(), B);
  TVM_FFI_ICHECK_GE(q_descale.numel(), 1);
  TVM_FFI_ICHECK_GE(k_descale.numel(), 1);
  // packed paged cache: (num_pages, 64, 576) or (num_pages, 64, 1, 576),
  // 576-byte contiguous rows, page-contiguous
  const int kv_dim = kv_cache.dim();
  TVM_FFI_ICHECK(kv_dim == 3 || kv_dim == 4) << "kv_cache must be (pages, 64, [1,] 576)";
  TVM_FFI_ICHECK_EQ(kv_cache.size(1), PAGE);
  TVM_FFI_ICHECK_EQ(kv_cache.size(kv_dim - 1), DIMQ);
  if (kv_dim == 4) TVM_FFI_ICHECK_EQ(kv_cache.size(2), 1);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(kv_dim - 1), 1);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(1), DIMQ);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(0), (int64_t)PAGE * DIMQ)
      << "kv pages must be 64x576 contiguous bytes";
  const int total_pages = static_cast<int>(kv_cache.size(0));

  const int device_id = q.device().device_id;
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  const int sm_count = g_sm_count[device_id];
  TVM_FFI_ICHECK_GT(sm_count, 0)
      << "mla_decode_fp8_paged_sm90: init(device_id) must run before run";

  const MlaSplitPlan P = mla_decode_fp8_split_plan(B, QL, pps, total_pages, sm_count);
  const size_t need = mla_decode_fp8_ws_bytes(P, B);
  TVM_FFI_ICHECK_GE((size_t)workspace.numel(), need)
      << "workspace too small: plan with plan_workspace_bytes(" << B << ", " << QL << ", " << pps
      << ", " << total_pages << ", " << device_id << ")";

  CHECK_DEVICE(q, kv_cache);
  CHECK_DEVICE(q, block_table);
  CHECK_DEVICE(q, seq_lens);
  CHECK_DEVICE(q, q_descale);
  CHECK_DEVICE(q, k_descale);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, workspace);

  ffi::CUDADeviceGuard device_guard(device_id);
  const cudaStream_t st = get_stream(q.device());
  g_sm = sm_count;  // the verbatim selection logic reads the TLS mirror

  // verbatim locals of the measured launcher, sourced from the plan
  const int MT = P.MT;
  const int QPC = P.QPC;
  const int nqc = P.nqc;
  const int ns = P.ns;
  const int pps_split = P.pps_split;
  const int nblocks = P.nblocks;
  const int rows_tot = P.rows_tot;
  const int rect_hint = P.rect_hint;
  uint8_t* Opp = nullptr;
  float* Mpp = nullptr;
  float* Lpp = nullptr;
  if (ns > 1) {
    const size_t nrow = (size_t)B * ns * rows_tot;
    const size_t obytes = nrow * CKV;
    char* p = static_cast<char*>(workspace.data_ptr());
    Opp = reinterpret_cast<uint8_t*>(p);
    Mpp = reinterpret_cast<float*>(p + obytes);
    Lpp = Mpp + nrow;
  }

  const int* btp = static_cast<const int*>(block_table.data_ptr());
  const int* slp = static_cast<const int*>(seq_lens.data_ptr());
  const float* dqp = static_cast<const float*>(q_descale.data_ptr());
  const float* dkp = static_cast<const float*>(k_descale.data_ptr());
  int ROWS = 16 * MT;
  size_t smem =
      (size_t)ROWS * KSTR + (size_t)2 * TT * KSTR + (size_t)ROWS * PSTR + (size_t)8 * ROWS * 4;
  dim3 grid(nblocks, 1, nqc);
  int nthr = 128 * MT;

  const uint8_t* qp = reinterpret_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* kp = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

  const uint64_t map_rows = static_cast<uint64_t>(total_pages) * PAGE;
  const bool no_promote = total_pages <= g_sm;
  if (MT == 3 && (!g_tmap.valid || g_tmap.base != static_cast<const void*>(kp) ||
                  g_tmap.rows != map_rows || g_tmap.no_promote != no_promote)) {
    CUtensorMap& kv_map = g_tmap.kv;
    CUtensorMap& pe_map = g_tmap.pe;
    const uint64_t rows = map_rows;
    const uint64_t strides[1] = {DIMQ};
    const uint32_t elem_strides[2] = {1, 1};
    {
      const uint64_t dims[2] = {CKV, rows};
      const uint32_t box[2] = {128, PAGE};
      CUresult result = get_tmap_encoder()(
          &kv_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(kp), dims, strides, box,
          elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          no_promote ? CU_TENSOR_MAP_L2_PROMOTION_NONE : CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(result == CUDA_SUCCESS)
          << "cuTensorMapEncodeTiled(ckv) failed: " << static_cast<int>(result);
    }
    {
      const uint64_t dims[2] = {DIMQ - CKV, rows};
      const uint32_t box[2] = {64, PAGE};
      CUresult result = get_tmap_encoder()(
          &pe_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(kp) + CKV, dims, strides,
          box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
          no_promote ? CU_TENSOR_MAP_L2_PROMOTION_NONE : CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(result == CUDA_SUCCESS)
          << "cuTensorMapEncodeTiled(kpe) failed: " << static_cast<int>(result);
    }
    g_tmap.base = static_cast<const void*>(kp);
    g_tmap.rows = map_rows;
    g_tmap.no_promote = no_promote;
    g_tmap.valid = true;
  }
  const CUtensorMap& kv_map = g_tmap.kv;
  const CUtensorMap& pe_map = g_tmap.pe;

#define LAUNCH(M, H, TAG)                                                                       \
  do {                                                                                          \
    static std::once_flag init_##TAG;                                                           \
    std::call_once(init_##TAG, [&] {                                                            \
      cudaFuncSetAttribute(mla_kernel<M, H>, cudaFuncAttributeMaxDynamicSharedMemorySize,       \
                           (int)smem);                                                          \
    });                                                                                         \
    mla_kernel<M, H><<<grid, nthr, smem, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, \
                                               Opp, Mpp, Lpp, B, QL, pps, ns, pps_split, QPC);  \
  } while (0)

  // Unit-max probability quantization on long splits keeps packed-FP16 PV
  // below overflow while preserving the same softmax ratio in the epilogue.
  const bool half_acc = true;
  const size_t smw = (size_t)(NSLOT + 1) * PAGE_SZ + (size_t)PSZ + 1280 + (size_t)MAXPG * 4;
  const size_t smw4 = (size_t)(4 + 1) * PAGE_SZ + (size_t)PSZ + 1280 + (size_t)MAXPG * 4;
  const size_t smw_dual =
      (size_t)Q_STORAGE_SZ + (size_t)D_NSLOT * PAGE_SZ + (size_t)PSZ + 512 + (size_t)MAXPG * 4;
#define LAUNCHW(H, F, O, Q6, R, TAG)                                                             \
  do {                                                                                           \
    static std::once_flag initw_##TAG;                                                           \
    std::call_once(initw_##TAG, [&] {                                                            \
      cudaFuncSetAttribute(mla_kernel_tmaproducer<H, F, O, Q6, R>,                               \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smw);               \
    });                                                                                          \
    mla_kernel_tmaproducer<H, F, O, Q6, R>                                                       \
        <<<grid, 288, smw, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, Lpp, \
                                 kv_map, pe_map, B, QL, pps, ns, pps_split, QPC, rect_hint);     \
  } while (0)
#define LAUNCHD(H, Q6, TAG)                                                                      \
  do {                                                                                           \
    static std::once_flag initd_##TAG;                                                           \
    std::call_once(initd_##TAG, [&] {                                                            \
      cudaFuncSetAttribute(mla_kernel_dual<H, Q6>, cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                           (int)smw_dual);                                                       \
    });                                                                                          \
    mla_kernel_dual<H, Q6>                                                                       \
        <<<grid, 160, smw_dual, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, \
                                      Lpp, kv_map, pe_map, B, QL, pps, ns, pps_split, QPC);      \
  } while (0)
#define LAUNCHW_NOCAP(H, F, O, Q6, R, TAG)                                                       \
  do {                                                                                           \
    static std::once_flag initw_##TAG;                                                           \
    std::call_once(initw_##TAG, [&] {                                                            \
      cudaFuncSetAttribute(mla_kernel_tmaproducer<H, F, O, Q6, R, false>,                        \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smw);               \
    });                                                                                          \
    mla_kernel_tmaproducer<H, F, O, Q6, R, false>                                                \
        <<<grid, 288, smw, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, Lpp, \
                                 kv_map, pe_map, B, QL, pps, ns, pps_split, QPC, rect_hint);     \
  } while (0)
#define LAUNCHW4_NOCAP(H, F, O, Q6, R, TAG)                                                       \
  do {                                                                                            \
    static std::once_flag initw_##TAG;                                                            \
    std::call_once(initw_##TAG, [&] {                                                             \
      cudaFuncSetAttribute(mla_kernel_tmaproducer<H, F, O, Q6, R, false, true, 4>,                \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smw4);               \
    });                                                                                           \
    mla_kernel_tmaproducer<H, F, O, Q6, R, false, true, 4>                                        \
        <<<grid, 288, smw4, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, Lpp, \
                                  kv_map, pe_map, B, QL, pps, ns, pps_split, QPC, rect_hint);     \
  } while (0)
#define LAUNCHD_NOCAP(H, Q6, TAG)                                                                \
  do {                                                                                           \
    static std::once_flag initd_##TAG;                                                           \
    std::call_once(initd_##TAG, [&] {                                                            \
      cudaFuncSetAttribute(mla_kernel_dual<H, Q6, false>,                                        \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smw_dual);          \
    });                                                                                          \
    mla_kernel_dual<H, Q6, false>                                                                \
        <<<grid, 160, smw_dual, st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, \
                                      Lpp, kv_map, pe_map, B, QL, pps, ns, pps_split, QPC);      \
  } while (0)
  // A split can increase L by at most 64*pps_split.  The guard cannot fire
  // below 2^25 tokens, so use a specialization with those instructions
  // compiled out while preserving the guarded path for unbounded contexts.
  const bool lcheck = pps_split >= (1 << 19);
  // Packed-FP16 PV accumulation in the MT<=2 kernels is statically safe only
  // while a split spans at most two pages (128 tokens * 448 = 57344 < 65504);
  // unlike the warp-group kernels they have no nonfinite repair path, so
  // deeper splits take the FP32-accumulator instantiations instead.
  const bool shallow_half = (ns == 1 ? pps : pps_split) <= 2;
  if (half_acc) {
    if (MT == 1) {
      if (shallow_half)
        LAUNCH(1, true, h1);
      else
        LAUNCH(1, false, h1f);
    } else if (MT == 2) {
      if (shallow_half)
        LAUNCH(2, true, h2);
      else
        LAUNCH(2, false, h2f);
    } else if (B >= 5) {
      if (QL == 6) {
        if (lcheck)
          LAUNCHD(true, true, dh6c);
        else
          LAUNCHD_NOCAP(true, true, dh6);
      } else {
        if (lcheck)
          LAUNCHD(true, false, dhc);
        else
          LAUNCHD_NOCAP(true, false, dh);
      }
    } else if (QL == 6 && ns > 1) {
      if (B == 1) {
        // No REPAIR_CHECK-free two-page variant: static overflow safety
        // (128 * 448 < 65504) does not cover NaN bytes in the unwritten tail
        // rows of a partially-filled page, which the repair replay recomputes
        // with zero-filled loads.
        if (lcheck)
          LAUNCHW(true, true, true, true, false, wh6b1c);
        else if (pps_split >= 16)
          LAUNCHW4_NOCAP(true, true, true, true, false, wh6b1r4);
        else
          LAUNCHW_NOCAP(true, true, true, true, false, wh6b1);
      } else if (B == 2) {
        if (lcheck)
          LAUNCHW(true, true, false, true, true, wh6b2c);
        else
          LAUNCHW_NOCAP(true, true, false, true, true, wh6b2);
      } else {
        if (lcheck)
          LAUNCHW(true, true, false, true, false, wh6fc);
        else
          LAUNCHW_NOCAP(true, true, false, true, false, wh6f);
      }
    } else if ((QL % 6) == 0 && ns > 1) {
      if (B == 1) {
        if (lcheck)
          LAUNCHW(true, true, true, false, false, whb1c);
        else
          LAUNCHW_NOCAP(true, true, true, false, false, whb1);
      } else {
        if (lcheck)
          LAUNCHW(true, true, false, false, false, whfc);
        else
          LAUNCHW_NOCAP(true, true, false, false, false, whf);
      }
    } else {
      if (lcheck)
        LAUNCHW(true, false, false, false, false, whc);
      else
        LAUNCHW_NOCAP(true, false, false, false, false, wh);
    }
  } else {
    if (MT == 1)
      LAUNCH(1, false, f1);
    else if (MT == 2)
      LAUNCH(2, false, f2);
    else if (B >= 5) {
      if (QL == 6)
        LAUNCHD(false, true, df6);
      else
        LAUNCHD(false, false, df);
    } else if (QL == 6 && ns > 1) {
      if (B == 1)
        LAUNCHW(false, true, true, true, false, wf6b1);
      else if (B == 2)
        LAUNCHW(false, true, false, true, true, wf6b2);
      else
        LAUNCHW(false, true, false, true, false, wf6f);
    } else if ((QL % 6) == 0 && ns > 1) {
      if (B == 1)
        LAUNCHW(false, true, true, false, false, wfb1);
      else
        LAUNCHW(false, true, false, false, false, wff);
    } else
      LAUNCHW(false, false, false, false, false, wf);
  }
#undef LAUNCHD
#undef LAUNCHW
#undef LAUNCHD_NOCAP
#undef LAUNCHW_NOCAP
#undef LAUNCHW4_NOCAP
#undef LAUNCH

  if (ns > 1) {
    const int nrows = B * rows_tot;
    const int combine_span = (B == 1) ? 0 : pps_split;
    if (B >= 5 && ns <= 72) {
      // Multi-batch rows have only a few splits.  Four independent 4-warp
      // CTAs keep the same total warp count as the 16-warp GROUP=4 block but
      // reduce each shared reduction/barrier domain to one 128-column chunk.
      dim3 cgrid(rows_tot, CKV / 128, B);
      if (ns <= 40)
        launch_combine<1, false, 4, 1>(cgrid, 128, ns * sizeof(float), st, Opp, Mpp, Lpp, dkp, slp,
                                       op, ns, rows_tot, combine_span);
      else
        launch_combine<1, true, 4, 13>(cgrid, 128, ns * sizeof(float), st, Opp, Mpp, Lpp, dkp, slp,
                                       op, ns, rows_tot, combine_span);
    } else {
      const int group = (nrows >= sm_count) ? 4 : 1;
      dim3 cgrid(rows_tot, (CKV / 128) / group, B);
      if (group == 4)
        launch_combine<4, false, CWARPS, 12>(cgrid, 32 * CWARPS, ns * sizeof(float), st, Opp, Mpp,
                                             Lpp, dkp, slp, op, ns, rows_tot, combine_span);
      else if (B == 1 && ns < 64)
        launch_combine<1, true, CSHORTWARPS, 5>(cgrid, 32 * CSHORTWARPS, ns * sizeof(float), st,
                                                Opp, Mpp, Lpp, dkp, slp, op, ns, rows_tot,
                                                combine_span);
      else if (B == 1 && ns >= 96) {
        if (ns <= 112)
          launch_combine<1, true, C1WARPS, 7>(cgrid, 32 * C1WARPS, ns * sizeof(float), st, Opp, Mpp,
                                              Lpp, dkp, slp, op, ns, rows_tot, combine_span);
        else if (ns <= 128)
          launch_combine<1, true, C1WARPS, 8>(cgrid, 32 * C1WARPS, ns * sizeof(float), st, Opp, Mpp,
                                              Lpp, dkp, slp, op, ns, rows_tot, combine_span);
        else
          launch_combine<1, true, C1WARPS, 9>(cgrid, 32 * C1WARPS, ns * sizeof(float), st, Opp, Mpp,
                                              Lpp, dkp, slp, op, ns, rows_tot, combine_span);
      } else if (B == 2)
        launch_combine<1, true, 8, 10>(cgrid, 256, ns * sizeof(float), st, Opp, Mpp, Lpp, dkp, slp,
                                       op, ns, rows_tot, combine_span);
      else
        launch_combine<1, false, CWARPS, 12>(cgrid, 32 * CWARPS, ns * sizeof(float), st, Opp, Mpp,
                                             Lpp, dkp, slp, op, ns, rows_tot, combine_span);
    }
  }

  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mla_decode_fp8_paged_sm90 launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mla_decode_fp8_paged_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, mla_decode_fp8_paged_sm90_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_workspace_bytes, mla_decode_fp8_paged_sm90_plan_workspace_bytes);
