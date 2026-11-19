// SPDX-License-Identifier: Apache-2.0
// Sparse MLA (DSA) decode with precomputed top-k indices over the
// fp8_ds_mla 656-byte/token Page64 paged KV cache, for SM90 (H100/H200).
//
// Contract (DeepSeek-V3.2 / GLM-class DSA decode, attention stage only):
//   q         bf16  [B, q_len, 8, 576]   q_nope(512) + q_pe(64) per head
//   kv_cache  uint8 [num_pages, 64, 656] per-token row:
//               [0:512)   512 x fp8_e4m3 quantized ckv (NoPE latent)
//               [512:528) 4 x fp32 scales, one per 128-element tile
//               [528:656) 64 x bf16 RoPE (kpe), unquantized
//   indices   int32 [B, q_len, 2048]     physical token slots into the
//             flattened (num_pages*64)-row token cache, -1 = invalid pad
//             (accepted anywhere); only listed tokens are attended; the
//             kernel is non-causal (causality lives in the index lists).
//   seq_lens  int32 [B]                  scheduling metadata only
//   out       bf16  [B, q_len, 8, 512]   latent-space output; rows whose
//             index list is all-invalid are zero.
//
// Key design: both GEMMs run TOKEN-MAJOR on the m16n8k16 tensor-core shape so
// the 8 real q-heads land exactly on N=8 (no head padding at all):
//   GEMM1  S[16 tok][8 head] = K[16 tok][16 d] * Q^T[16 d][8 head]
//   GEMM2  O[16 d][8 head]   = V^T[16 d][16 tok] * P^T[16 tok][8 head]
//
// The device code and launch heuristic are carried from the kernel
// validated on H200 (same-node two-arm study re-time) with two
// numerical hardenings: (1) Q and RoPE keys retain their BF16 bits through
// staging; QK uses TF32 operands, which represent both BF16 and the staged
// FP16 latents without further rounding; (2) probabilities are staged
// as exp2(s - m + 14) plus an fp16 residual plane, so ~22 mantissa bits of
// every probability reach PV and representable output contributions are not
// rounded away. The host binding differs only in workspace ownership
// (caller-provided buffer in place of a cudaMalloc'd static) and FlashInfer
// TVM-FFI conventions.
// Launches are plain per-stream kernel launches (no cooperative launch, no
// allocation, no host sync), so calls are CUDA-graph capturable once the
// module is built and warmed eagerly.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>

#include "tvm_ffi_utils.h"

#define CKV 512
#define KPE 64
#define DQ 576
#define BPT 656
#define TOPK 2048
#define NWARP 8
#define NTHREAD 256
#define CS 584  // conv row stride (halves): 576 + 8 pad -> conflict-free
#define QS 584
#define OSTR 516  // epilogue float staging stride

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t lo16) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)lo16));
  return r;
}
__device__ __forceinline__ uint32_t dequant_pair(uint32_t bytes, float scale) {
  const uint32_t h = fp8x2_to_f16x2(bytes);
  const float x = __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
  const float y = __half2float(__ushort_as_half(static_cast<unsigned short>(h >> 16)));
  const __half2 v = __floats2half2_rn(x * scale, y * scale);
  return __half_as_ushort(__low2half(v)) | (uint32_t(__half_as_ushort(__high2half(v))) << 16);
}

__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

// Each packed K=16 fragment holds pairs of consecutive features. Taking
// the same half of each pair in A and B gives the lane layout of a K=8
// TF32 fragment: two MMAs cover the even and odd features respectively.
template <bool ROPE>
__device__ __forceinline__ void mma_qk(float* c, const uint32_t* a, const uint32_t* b, float qmul) {
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    uint32_t k[4], q[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned short bits = static_cast<unsigned short>(a[i] >> (16 * j));
      k[i] =
          tf32(ROPE ? __uint_as_float(uint32_t(bits) << 16) : __half2float(__ushort_as_half(bits)));
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float x = __uint_as_float((b[i] >> (16 * j)) << 16);
      q[i] = tf32(x * qmul);
    }
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(k[0]), "r"(k[1]), "r"(k[2]), "r"(k[3]), "r"(q[0]), "r"(q[1]));
  }
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldm_x4(uint32_t* d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldm_x4t(uint32_t* d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldm_x2t(uint32_t* d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}

template <int TILE, int STATIC_TPS>
__global__ __launch_bounds__(NTHREAD, 2) void mla_main(
    const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kv,
    const int* __restrict__ indices, __half* __restrict__ po, float* __restrict__ plse,
    __nv_bfloat16* __restrict__ outp, int nsplits, int tps, float qscale) {
  static_assert(STATIC_TPS == 0 || STATIC_TPS == 1, "unsupported static TPS");
  extern __shared__ __align__(16) unsigned char smem[];
  __half* conv = (__half*)smem;            // TILE * CS
  __half* qs = conv + TILE * CS;           // 8 * QS
  __half* ps = qs + 8 * QS;                // TILE * 8  (P, exp2(s - m + 14))
  __half* psl = ps + TILE * 8;             // TILE * 8  (P fp16 residual plane)
  float* red0 = (float*)(psl + TILE * 8);  // 32
  float* red1 = red0 + 32;                 // 32
  float* hsc = red1 + 32;                  // 8 inverse Q multipliers
  int* sl = (int*)(hsc + 8);               // tps * TILE

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m4 = lane & 3, l4 = lane >> 2;
  const int row = blockIdx.x, split = blockIdx.y;

  const int tps_eff = (STATIC_TPS == 0) ? tps : STATIC_TPS;
  const int tile0 = split * tps_eff;
  int nt = 1;
  if constexpr (STATIC_TPS == 0) {
    nt = tps;
    constexpr int maxt = (TOPK + TILE - 1) / TILE;
    if (tile0 + nt > maxt) nt = maxt - tile0;
    if (nt < 0) nt = 0;
  }
  const int tok_begin = tile0 * TILE;
  const int* idxp = indices + (size_t)row * TOPK;
  if constexpr (STATIC_TPS == 0) {
    for (int i = TILE + tid; i < nt * TILE; i += NTHREAD) {
      int g = tok_begin + i;
      sl[i] = (g < TOPK) ? idxp[g] : -1;
    }
  }

  // The first tile's KV loads are issued straight from the global index list
  // (no shared round trip, so no barrier needed first) and the Q dequantize
  // below then runs inside their memory latency instead of ahead of it.
  int slots[TILE / 8];
  uint4 nv[TILE / 8];
  float sc0[TILE / 8];
  uint32_t rp[TILE / 8];
  if (nt > 0) {
    const int gb = tok_begin + warp * (TILE / 8);
#pragma unroll
    for (int i = 0; i < TILE / 8; ++i) {
      int s;
      if constexpr (STATIC_TPS == 1)
        s = (gb + i < TOPK) ? idxp[gb + i] : -1;
      else
        s = (gb + i < TOPK) ? idxp[gb + i] : -1;
      slots[i] = s;
      if (lane == 0) sl[warp * (TILE / 8) + i] = s;
      const uint8_t* rw = kv + (size_t)(s < 0 ? 0 : s) * BPT;
      nv[i] = *(const uint4*)(rw + lane * 16);
      sc0[i] = *(const float*)(rw + CKV + ((lane >> 3) << 2));
      rp[i] = *(const uint32_t*)(rw + 528 + lane * 4);
    }
  }

  // Keep packed BF16 in shared memory; scale only after unpacking to FP32.
  const uint32_t* qp = (const uint32_t*)(q + (size_t)row * (8 * DQ)) + warp * (DQ / 2);
  float amax = 0.f;
#pragma unroll
  for (int it = 0; it < 9; ++it) {
    const int d2 = lane + it * 32;
    const uint32_t v = qp[d2];
    ((uint32_t*)(qs + warp * QS))[d2] = v;
    amax = fmaxf(amax, fabsf(__uint_as_float(v << 16)));
    amax = fmaxf(amax, fabsf(__uint_as_float(v & 0xffff0000u)));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (lane == 0) hsc[warp] = amax >= 65536.f ? exp2f(float(ilogbf(amax) - 15)) : 1.f;
  __syncthreads();

  // B fragment rows read Q head l4; C columns belong to heads 2*m4 and 2*m4+1.
  const float qmul = 1.f / hsc[l4];
  const float qsa = qscale * hsc[2 * m4];
  const float qsb = qscale * hsc[2 * m4 + 1];

  float acc[4][4];
#pragma unroll
  for (int p = 0; p < 4; ++p)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[p][k] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  for (int t = 0; t < nt; ++t) {
    // ---- direct gather + dequantize tile -> conv (fp16) ----
    // Hoist the warp's rows before converting to expose memory-level parallelism.
    {
#pragma unroll
      for (int i = 0; i < TILE / 8; ++i) {
        __half* crow = conv + (warp * (TILE / 8) + i) * CS;
        const uint32_t vm = (slots[i] < 0) ? 0u : 0xffffffffu;
        const uint4 n = nv[i];
        uint4 o;
        o.x = dequant_pair(n.x & 0xffffu, sc0[i]) & vm;
        o.y = dequant_pair(n.x >> 16, sc0[i]) & vm;
        o.z = dequant_pair(n.y & 0xffffu, sc0[i]) & vm;
        o.w = dequant_pair(n.y >> 16, sc0[i]) & vm;
        *(uint4*)(crow + lane * 8) = o;
        uint4 o2;
        o2.x = dequant_pair(n.z & 0xffffu, sc0[i]) & vm;
        o2.y = dequant_pair(n.z >> 16, sc0[i]) & vm;
        o2.z = dequant_pair(n.w & 0xffffu, sc0[i]) & vm;
        o2.w = dequant_pair(n.w >> 16, sc0[i]) & vm;
        *(uint4*)(crow + 256 + lane * 8) = o2;
        *(uint32_t*)(crow + CKV + lane * 2) = rp[i] & vm;
      }
    }
    __syncthreads();

    const int tbase = t * TILE;
    // ---- GEMM1 : S[16 tok][8 head], one 16-token tile per active warp ----
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    bool va = false, vb = false;
    if (warp < TILE / 16) {
      const int t0 = warp * 16;
      const int arow = t0 + (lane & 7) + (((lane >> 3) & 1) << 3);
      const int cg = (lane >> 4) & 1;
      const __half* ab = conv + arow * CS + (cg << 8);
      const __half* abr = conv + arow * CS + CKV + (cg << 3);
      const __half* qr = qs + l4 * QS + m4 * 2;
      float s2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int k = 0; k < CKV / 16; k += 2) {
        uint32_t A[4], B[2];
        ldm_x4(A, smem_u32(ab + k * 8));
        B[0] = *(const uint32_t*)(qr + k * 16);
        B[1] = *(const uint32_t*)(qr + k * 16 + 8);
        mma_qk<false>(s, A, B, qmul);
        uint32_t A1[4], B1[2];
        ldm_x4(A1, smem_u32(ab + k * 8 + 8));
        B1[0] = *(const uint32_t*)(qr + k * 16 + 16);
        B1[1] = *(const uint32_t*)(qr + k * 16 + 24);
        mma_qk<false>(s2, A1, B1, qmul);
      }
#pragma unroll
      for (int k = 0; k < (DQ - CKV) / 16; k += 2) {
        uint32_t A[4], B[2];
        ldm_x4(A, smem_u32(abr + k * 16));
        B[0] = *(const uint32_t*)(qr + CKV + k * 16);
        B[1] = *(const uint32_t*)(qr + CKV + k * 16 + 8);
        mma_qk<true>(s, A, B, qmul);
        uint32_t A1[4], B1[2];
        ldm_x4(A1, smem_u32(abr + k * 16 + 16));
        B1[0] = *(const uint32_t*)(qr + CKV + k * 16 + 16);
        B1[1] = *(const uint32_t*)(qr + CKV + k * 16 + 24);
        mma_qk<true>(s2, A1, B1, qmul);
      }
      // Apply the compensated softmax scale, never materializing the raw dot.
      s[0] = (s[0] + s2[0]) * qsa;
      s[1] = (s[1] + s2[1]) * qsb;
      s[2] = (s[2] + s2[2]) * qsa;
      s[3] = (s[3] + s2[3]) * qsb;
      va = (sl[tbase + t0 + l4] >= 0);
      vb = (sl[tbase + t0 + l4 + 8] >= 0);
      float mx0 = fmaxf(va ? s[0] : -INFINITY, vb ? s[2] : -INFINITY);
      float mx1 = fmaxf(va ? s[1] : -INFINITY, vb ? s[3] : -INFINITY);
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) {
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
      }
      if (l4 == 0) {
        red0[warp * 8 + 2 * m4] = mx0;
        red0[warp * 8 + 2 * m4 + 1] = mx1;
      }
    }
    __syncthreads();
    // Fold the QK-warp rows; unused logical lane groups are identities.
    const int wr = lane >> 3;
    float mv = (wr < TILE / 16) ? red0[wr * 8 + (lane & 7)] : -INFINITY;
    mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 8));
    mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, 16));
    float mt0 = __shfl_sync(0xffffffffu, mv, 2 * m4);
    float mt1 = __shfl_sync(0xffffffffu, mv, 2 * m4 + 1);
    const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
    if (warp < TILE / 16) {
      const int t0 = warp * 16;
      // P is staged as exp2(s - m + 14): the exact power-of-two shift rides
      // the O accumulators and the row sums alike and cancels in every O/l
      // normalization (split LSEs all carry the same +14), while
      // probabilities down to 2^-38 of the row max survive fp16 staging.
      // A second fp16 plane holds the staging residual, so ~22 mantissa
      // bits of every probability reach PV: differences that remain
      // significant after multiplying V are preserved instead of rounding
      // away at 11 bits.
      float p0 = va ? exp2f(s[0] - mn0 + 14.f) : 0.f;
      float p1 = va ? exp2f(s[1] - mn1 + 14.f) : 0.f;
      float p2 = vb ? exp2f(s[2] - mn0 + 14.f) : 0.f;
      float p3 = vb ? exp2f(s[3] - mn1 + 14.f) : 0.f;
      const __half2 h01 = __floats2half2_rn(p0, p1);
      const __half2 h23 = __floats2half2_rn(p2, p3);
      *(__half2*)(ps + (t0 + l4) * 8 + 2 * m4) = h01;
      *(__half2*)(ps + (t0 + l4 + 8) * 8 + 2 * m4) = h23;
      *(__half2*)(psl + (t0 + l4) * 8 + 2 * m4) = __floats2half2_rn(
          p0 - __half2float(__low2half(h01)), p1 - __half2float(__high2half(h01)));
      *(__half2*)(psl + (t0 + l4 + 8) * 8 + 2 * m4) = __floats2half2_rn(
          p2 - __half2float(__low2half(h23)), p3 - __half2float(__high2half(h23)));
      float ls0 = p0 + p2, ls1 = p1 + p3;
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) {
        ls0 += __shfl_xor_sync(0xffffffffu, ls0, o);
        ls1 += __shfl_xor_sync(0xffffffffu, ls1, o);
      }
      if (l4 == 0) {
        red1[warp * 8 + 2 * m4] = ls0;
        red1[warp * 8 + 2 * m4 + 1] = ls1;
      }
    }
    __syncthreads();
    float sv = (wr < TILE / 16) ? red1[wr * 8 + (lane & 7)] : 0.f;
    sv += __shfl_xor_sync(0xffffffffu, sv, 8);
    sv += __shfl_xor_sync(0xffffffffu, sv, 16);
    float lt0 = __shfl_sync(0xffffffffu, sv, 2 * m4);
    float lt1 = __shfl_sync(0xffffffffu, sv, 2 * m4 + 1);
    const float a0 = (m0 == -INFINITY) ? 0.f : exp2f(m0 - mn0);
    const float a1 = (m1 == -INFINITY) ? 0.f : exp2f(m1 - mn1);
    l0 = l0 * a0 + lt0;
    l1 = l1 * a1 + lt1;
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      acc[p][0] *= a0;
      acc[p][1] *= a1;
      acc[p][2] *= a0;
      acc[p][3] *= a1;
    }

    // ---- GEMM2 : O[16 d][8 head] ----
    {
      const int gb0 = warp * 32;
      const int arow_o = (lane & 7) + (((lane >> 4) & 1) << 3);
      const int acol_o = (((lane >> 3) & 1) << 8);
#pragma unroll
      for (int kt = 0; kt < TILE / 16; ++kt) {
        uint32_t B[2], Bl[2];
        ldm_x2t(B, smem_u32(ps + (kt * 16 + (lane & 15)) * 8));
        ldm_x2t(Bl, smem_u32(psl + (kt * 16 + (lane & 15)) * 8));
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t A[4];
          ldm_x4t(A, smem_u32(conv + (kt * 16 + arow_o) * CS + gb0 + p * 8 + acol_o));
          mma16816(acc[p], A, B);
          mma16816(acc[p], A, Bl);
        }
      }
    }
    if constexpr (STATIC_TPS == 0) {
      if (t + 1 < nt) {
        // Issue the next tile's low-half gather before waiting for every warp to
        // finish the current PV reads.  The barrier still precedes any overwrite
        // of conv or ps in the next loop iteration.
        const int tb_next = (t + 1) * TILE + warp * (TILE / 8);
#pragma unroll
        for (int i = 0; i < TILE / 8; ++i) {
          int s = sl[tb_next + i];
          slots[i] = s;
          const uint8_t* rw = kv + (size_t)(s < 0 ? 0 : s) * BPT;
          nv[i] = *(const uint4*)(rw + lane * 16);
          sc0[i] = *(const float*)(rw + CKV + ((lane >> 3) << 2));
          rp[i] = *(const uint32_t*)(rw + 528 + lane * 4);
        }
        __syncthreads();
      }
    }
  }

  // ---- epilogue ----
  const float i0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
  const float i1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
  __syncthreads();
  float* ob = (float*)conv;  // [8][OSTR]
  {
    const int dbw = warp * 64;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      int d = dbw + p * 16 + l4;
      ob[(2 * m4) * OSTR + d] = acc[p][0] * i0;
      ob[(2 * m4 + 1) * OSTR + d] = acc[p][1] * i1;
      ob[(2 * m4) * OSTR + d + 8] = acc[p][2] * i0;
      ob[(2 * m4 + 1) * OSTR + d + 8] = acc[p][3] * i1;
    }
  }
  __syncthreads();
  if (nsplits == 1) {
    __nv_bfloat16* dst = outp + (size_t)row * (8 * CKV);
    for (int i = tid * 4; i < 8 * CKV; i += NTHREAD * 4) {
      int h = i >> 9, d = i & 511;
      float4 v = *(const float4*)(ob + h * OSTR + d);
      __nv_bfloat162 o0 = __floats2bfloat162_rn(v.x, v.y);
      __nv_bfloat162 o1 = __floats2bfloat162_rn(v.z, v.w);
      uint2 w;
      w.x = *(const uint32_t*)&o0;
      w.y = *(const uint32_t*)&o1;
      *(uint2*)(dst + i) = w;
    }
  } else {
    __half* dst = po + ((size_t)row * nsplits + split) * (8 * CKV);
    for (int i = tid * 4; i < 8 * CKV; i += NTHREAD * 4) {
      int h = i >> 9, d = i & 511;
      float4 v = *(const float4*)(ob + h * OSTR + d);
      __half2 o0 = __floats2half2_rn(v.x, v.y);
      __half2 o1 = __floats2half2_rn(v.z, v.w);
      uint2 w;
      w.x = *(const uint32_t*)&o0;
      w.y = *(const uint32_t*)&o1;
      *(uint2*)(dst + i) = w;
    }
    if (warp == 0 && l4 == 0) {
      float* lp = plse + ((size_t)row * nsplits + split) * 8;
      lp[2 * m4] = (l0 > 0.f) ? (m0 + __log2f(l0)) : -INFINITY;
      lp[2 * m4 + 1] = (l1 > 0.f) ? (m1 + __log2f(l1)) : -INFINITY;
    }
  }
}

// ---------------------------------------------------------------------------
// Split combine.  One CTA per (row, head, 128-dim chunk) -> 4x more CTAs than a
// per-(row,head) combine, so the reduction saturates HBM instead of 1/3 of the
// SMs.  nsplits is a template constant for the shapes the splitter produces, so
// the whole split loop unrolls and every partial load is issued up front.
template <int NS>
__global__ __launch_bounds__(64, 8) void mla_combine_t(const __half* __restrict__ po,
                                                       const float* __restrict__ plse,
                                                       __nv_bfloat16* __restrict__ outp,
                                                       int nsplits) {
  const int row = blockIdx.z, h = blockIdx.y;
  const int d = blockIdx.x * 128 + (int)threadIdx.x * 2;
  const float* lp = plse + (size_t)row * nsplits * 8 + h;
  float e[NS];
  float M = -INFINITY;
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    e[s] = lp[(size_t)s * 8];
    M = fmaxf(M, e[s]);
  }
  float Z = 0.f;
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    e[s] = (M > -INFINITY) ? exp2f(e[s] - M) : 0.f;
    Z += e[s];
  }
  const float iz = (Z > 0.f) ? (1.f / Z) : 0.f;
  const __half* src = po + (size_t)row * nsplits * (8 * CKV) + h * CKV + d;
  __half2 v[NS];
#pragma unroll
  for (int s = 0; s < NS; ++s) v[s] = *(const __half2*)(src + (size_t)s * (8 * CKV));
  float2 a = make_float2(0.f, 0.f);
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    float2 f = __half22float2(v[s]);
    a.x = fmaf(f.x, e[s], a.x);
    a.y = fmaf(f.y, e[s], a.y);
  }
  *(__nv_bfloat162*)(outp + (size_t)row * (8 * CKV) + h * CKV + d) =
      __floats2bfloat162_rn(a.x * iz, a.y * iz);
}

template <int NS>
__global__ __launch_bounds__(64, 8) void mla_combine_warp_t(const __half* __restrict__ po,
                                                            const float* __restrict__ plse,
                                                            __nv_bfloat16* __restrict__ outp) {
  static_assert(NS <= 64);
  constexpr unsigned FULL = 0xffffffffu;
  const int row = blockIdx.z, h = blockIdx.y;
  const int lane = (int)threadIdx.x & 31;
  const float* lp = plse + (size_t)row * NS * 8 + h;
  float x0 = -INFINITY, x1 = -INFINITY;
  if (lane < NS) x0 = lp[(size_t)lane * 8];
  if (lane + 32 < NS) x1 = lp[(size_t)(lane + 32) * 8];
  const int d = blockIdx.x * 128 + (int)threadIdx.x * 2;
  const __half* src = po + (size_t)row * NS * (8 * CKV) + h * CKV + d;
  __half2 v[NS];
#pragma unroll
  for (int s = 0; s < NS; ++s) v[s] = *(const __half2*)(src + (size_t)s * (8 * CKV));
  float M = fmaxf(x0, x1);
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULL, M, o));
  float w0 = (lane < NS && M > -INFINITY) ? exp2f(x0 - M) : 0.f;
  float w1 = (lane + 32 < NS && M > -INFINITY) ? exp2f(x1 - M) : 0.f;
  float Z = w0 + w1;
#pragma unroll
  for (int o = 16; o; o >>= 1) Z += __shfl_xor_sync(FULL, Z, o);
  const float iz = (Z > 0.f) ? (1.f / Z) : 0.f;
  float2 a = make_float2(0.f, 0.f);
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const float ws = (s < 32) ? __shfl_sync(FULL, w0, s) : __shfl_sync(FULL, w1, s - 32);
    const float2 f = __half22float2(v[s]);
    a.x = fmaf(f.x, ws, a.x);
    a.y = fmaf(f.y, ws, a.y);
  }
  *(__nv_bfloat162*)(outp + (size_t)row * (8 * CKV) + h * CKV + d) =
      __floats2bfloat162_rn(a.x * iz, a.y * iz);
}

__global__ __launch_bounds__(64, 8) void mla_combine_g(const __half* __restrict__ po,
                                                       const float* __restrict__ plse,
                                                       __nv_bfloat16* __restrict__ outp,
                                                       int nsplits) {
  const int row = blockIdx.z, h = blockIdx.y;
  const int d = blockIdx.x * 128 + (int)threadIdx.x * 2;
  const float* lp = plse + (size_t)row * nsplits * 8 + h;
  float M = -INFINITY;
  for (int s = 0; s < nsplits; ++s) M = fmaxf(M, lp[(size_t)s * 8]);
  float Z = 0.f;
  for (int s = 0; s < nsplits; ++s) Z += (M > -INFINITY) ? exp2f(lp[(size_t)s * 8] - M) : 0.f;
  const float iz = (Z > 0.f) ? (1.f / Z) : 0.f;
  const __half* src = po + (size_t)row * nsplits * (8 * CKV) + h * CKV + d;
  float2 a = make_float2(0.f, 0.f);
  for (int s = 0; s < nsplits; ++s) {
    float w = (M > -INFINITY) ? exp2f(lp[(size_t)s * 8] - M) : 0.f;
    float2 f = __half22float2(*(const __half2*)(src + (size_t)s * (8 * CKV)));
    a.x = fmaf(f.x, w, a.x);
    a.y = fmaf(f.y, w, a.y);
  }
  *(__nv_bfloat162*)(outp + (size_t)row * (8 * CKV) + h * CKV + d) =
      __floats2bfloat162_rn(a.x * iz, a.y * iz);
}

// Launch heuristic (verbatim from the validated kernel): tile size, tiles per
// split and split count as a pure function of nq = B * q_len.
struct SplitPlan {
  bool small_batch;
  int tile;
  int tps;
  int nsplits;
};

inline SplitPlan make_split_plan(int nq) {
  SplitPlan p;
  p.small_batch = nq <= 12;
  p.tile = p.small_batch ? 48 : 64;
  const int maxt = (TOPK + p.tile - 1) / p.tile;
  int per = 264 / nq;
  if (per < 1) per = 1;
  if (per > maxt) per = maxt;
  p.tps = (maxt + per - 1) / per;
  if (p.tps < 1) p.tps = 1;
  p.nsplits = (maxt + p.tps - 1) / p.tps;
  return p;
}

inline int64_t workspace_bytes(int nq) {
  SplitPlan p = make_split_plan(nq);
  if (p.nsplits <= 1) return 0;
  return (int64_t)nq * p.nsplits * (8 * CKV) * 2 + (int64_t)nq * p.nsplits * 8 * 4;
}

bool device_supported(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int major, minor, smem;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
                    cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                    cudaSuccess);
  return major == 9 && minor == 0 && smem >= 94624;
}

int64_t get_workspace_bytes(int64_t nq) {
  TVM_FFI_ICHECK(nq >= 1) << "nq must be positive";
  // The split plan is not monotonic in token rows (e.g. 12 rows need more
  // bytes than 64), so the sizing surface returns the running maximum over
  // every row count up to the requested one: a buffer sized at the largest
  // row count a caller will use is valid for every smaller call. run()
  // checks the exact per-call need, which never exceeds this.
  int64_t cap = 0;
  for (int k = 1; k <= static_cast<int>(nq); ++k) {
    const int64_t b = workspace_bytes(k);
    if (b > cap) cap = b;
  }
  return cap;
}

void set_max_smem_attrs(int device) {
  // Opt-in dynamic shared memory: 48-tile path 75488 B, 64-tile path 94624 B.
  // cudaFuncSetAttribute applies to the calling thread's current device, so
  // the opt-in is tracked per device ordinal (not per process): set once per
  // device, before any launch there (and therefore before any CUDA-graph
  // capture of this op on that device). Caller holds the device guard.
  static std::atomic<uint64_t> done_mask{0};
  const bool trackable = device >= 0 && device < 64;
  const uint64_t bit = trackable ? (uint64_t(1) << device) : 0;
  if (trackable && (done_mask.load(std::memory_order_acquire) & bit)) return;
  cudaFuncSetAttribute(mla_main<48, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 75488);
  cudaFuncSetAttribute(mla_main<48, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 75488);
  cudaFuncSetAttribute(mla_main<64, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 94624);
  cudaFuncSetAttribute(mla_main<64, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 94624);
  if (trackable) done_mask.fetch_or(bit, std::memory_order_release);
}

void run(TensorView q, TensorView kv_cache, TensorView indices, TensorView seq_lens,
         TensorView workspace, TensorView out, double sm_scale) {
  CHECK_CUDA(q);
  for (auto t : {kv_cache, indices, seq_lens, workspace, out}) {
    CHECK_CUDA(t);
    CHECK_DEVICE(t, q);
  }
  CHECK_INPUT_TYPE(q, dl_bfloat16);
  CHECK_INPUT_TYPE(kv_cache, dl_uint8);
  CHECK_INPUT_TYPE(indices, dl_int32);
  CHECK_INPUT_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  CHECK_INPUT_TYPE(out, dl_bfloat16);
  CHECK_DIM(4, q);
  CHECK_DIM(3, kv_cache);
  CHECK_DIM(3, indices);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, workspace);
  CHECK_DIM(4, out);
  for (auto t : {q, kv_cache, indices, seq_lens, workspace, out}) {
    CHECK_CONTIGUOUS(t);
  }
  const int64_t b = q.size(0), ql = q.size(1);
  TVM_FFI_ICHECK_EQ(q.size(2), 8) << "q heads must be 8 (TP-rank local MLA head count)";
  TVM_FFI_ICHECK_EQ(q.size(3), DQ) << "q head_dim must be 576 (512 ckv + 64 kpe)";
  TVM_FFI_ICHECK_EQ(kv_cache.size(1), 64) << "page_size must be 64";
  TVM_FFI_ICHECK_EQ(kv_cache.size(2), BPT) << "kv_cache rows must be 656 bytes (fp8_ds_mla)";
  TVM_FFI_ICHECK_EQ(indices.size(0), b);
  TVM_FFI_ICHECK_EQ(indices.size(1), ql);
  TVM_FFI_ICHECK_EQ(indices.size(2), TOPK) << "top-k must be 2048";
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), b);
  TVM_FFI_ICHECK_EQ(out.size(0), b);
  TVM_FFI_ICHECK_EQ(out.size(1), ql);
  TVM_FFI_ICHECK_EQ(out.size(2), 8);
  TVM_FFI_ICHECK_EQ(out.size(3), CKV);
  TVM_FFI_ICHECK(std::isfinite(sm_scale) && sm_scale > 0) << "sm_scale must be positive finite";
  for (auto t : {q, kv_cache, workspace, out}) {
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16, 0)
        << "tensors require 16-byte alignment";
  }
  TVM_FFI_ICHECK(device_supported(q.device().device_id))
      << "Requires SM90 with at least 94624 bytes opt-in shared memory";

  const int nq = static_cast<int>(b * ql);
  TVM_FFI_ICHECK(nq >= 1) << "batch * q_len must be positive";
  const SplitPlan plan = make_split_plan(nq);
  const int tile = plan.tile, tps = plan.tps, nsplits = plan.nsplits;

  const int64_t need = workspace_bytes(nq);
  TVM_FFI_ICHECK_GE(workspace.size(0), need)
      << "workspace too small: need " << need << " bytes for nq=" << nq;

  ffi::CUDADeviceGuard guard(q.device().device_id);
  set_max_smem_attrs(static_cast<int>(q.device().device_id));

  size_t smem = (size_t)tile * CS * 2 + (size_t)8 * QS * 2 + (size_t)tile * 8 * 2 * 2 + 64 * 4 +
                8 * sizeof(float) + (size_t)tps * tile * 4;

  __half* po = nullptr;
  float* plse = nullptr;
  if (nsplits > 1) {
    size_t obytes = (size_t)nq * nsplits * (8 * CKV) * sizeof(__half);
    po = reinterpret_cast<__half*>(workspace.data_ptr());
    plse = reinterpret_cast<float*>(static_cast<char*>(workspace.data_ptr()) + obytes);
  }

  // Current-stream re-query on every call (CUDA-graph requirement: the launch
  // stream is read from the environment at call time, never cached).
  cudaStream_t stream = get_stream(q.device());

  const float qscale = (float)(sm_scale * 1.4426950408889634);
  const auto* qptr = static_cast<const __nv_bfloat16*>(q.data_ptr());
  const auto* kvptr = static_cast<const uint8_t*>(kv_cache.data_ptr());
  const auto* iptr = static_cast<const int*>(indices.data_ptr());
  auto* optr = static_cast<__nv_bfloat16*>(out.data_ptr());
  dim3 grid(nq, nsplits);
  if (plan.small_batch && tps == 1) {
    mla_main<48, 1>
        <<<grid, NTHREAD, smem, stream>>>(qptr, kvptr, iptr, po, plse, optr, nsplits, tps, qscale);
  } else if (plan.small_batch) {
    mla_main<48, 0>
        <<<grid, NTHREAD, smem, stream>>>(qptr, kvptr, iptr, po, plse, optr, nsplits, tps, qscale);
  } else if (tps == 1) {
    mla_main<64, 1>
        <<<grid, NTHREAD, smem, stream>>>(qptr, kvptr, iptr, po, plse, optr, nsplits, tps, qscale);
  } else {
    mla_main<64, 0>
        <<<grid, NTHREAD, smem, stream>>>(qptr, kvptr, iptr, po, plse, optr, nsplits, tps, qscale);
  }
  if (nsplits > 1) {
    dim3 cg(4, 8, nq);
#define CMB(N) mla_combine_t<N><<<cg, 64, 0, stream>>>(po, plse, optr, nsplits)
    switch (nsplits) {
      case 2:
        CMB(2);
        break;
      case 3:
        CMB(3);
        break;
      case 4:
        CMB(4);
        break;
      case 5:
        CMB(5);
        break;
      case 6:
        CMB(6);
        break;
      case 7:
        CMB(7);
        break;
      case 8:
        CMB(8);
        break;
      case 11:
        CMB(11);
        break;
      case 16:
        mla_combine_warp_t<16><<<cg, 64, 0, stream>>>(po, plse, optr);
        break;
      case 22:
        mla_combine_warp_t<22><<<cg, 64, 0, stream>>>(po, plse, optr);
        break;
      case 32:
        mla_combine_warp_t<32><<<cg, 64, 0, stream>>>(po, plse, optr);
        break;
      case 43:
        mla_combine_warp_t<43><<<cg, 64, 0, stream>>>(po, plse, optr);
        break;
      default:
        mla_combine_g<<<cg, 64, 0, stream>>>(po, plse, optr, nsplits);
        break;
    }
#undef CMB
  }
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(device_supported, device_supported);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_workspace_bytes, get_workspace_bytes);
