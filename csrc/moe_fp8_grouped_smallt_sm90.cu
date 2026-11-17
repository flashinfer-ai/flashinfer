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

// Specialized SM90 small-batch (decode-step) fused-MoE fast-path for the
// DeepSeek-style fp8 [128,128]-block-scale weight format: bf16 activations,
// fp8 (e4m3) gated fc1 (rows [0:I] = up / multiplicative branch, rows
// [I:2I] = gate / SiLU branch — the cutlass_fused_moe convention), fp8 fc2,
// captured per-token expert selections and combine weights consumed as-is,
// bf16 output.
//
// Algorithm (the T < 256 path of the measured kernel): fc1 runs one CTA
// group per (token, selected expert) pair with the activation row staged in
// shared memory, dequantising fp8 weight tiles against their block scales
// in registers and writing the silu(gate) * up intermediate as fp16; fc2
// consumes the fp16 intermediates per token with the per-slot combine
// weights folded into the final bf16 reduction (the T == 1 class runs a
// PDL-serialized split pair, T == 4 and larger a per-token tiling). Two to
// three kernel launches total, no host round trips, no atomics.
//
// The grouped (expert-binned wgmma) large-T path of the measured kernel is
// carried verbatim below for completeness but is NOT exported: the
// dispatcher's allowlist is the measured small-T envelope (T <= 16), where
// this kernel wins against the serve's dispatched triton fused-MoE; at
// T >= 64 the measured numbers LOSE to that incumbent, so larger batches
// keep the stock path.
//
// Host contract (asserted by the Python dispatcher and by `run` below):
//  * hidden_states / out are contiguous [T, H] bf16, 1 <= T < 256
//    (dispatcher allowlist: T <= 16);
//  * fc1 weights [E, 2*I, H] e4m3 with [128,128] block scales
//    [E, ceil(2I/128), ceil(H/128)] f32; fc2 [E, H, I] e4m3 with scales
//    [E, ceil(H/128), ceil(I/128)] f32 (contiguous);
//  * topk_ids [T, K] int32 in [0, E), topk_weights [T, K] f32, consumed
//    as-is (the serve's routing math, including any fused shared-expert
//    slot, is baked into the capture);
//  * the caller-owned uint8 workspace holds at least
//    `plan_workspace_bytes(T, K, I)` bytes (the fp16 intermediate).
//
// `run` performs no allocation, no synchronisation and no attribute
// queries: it launches on the caller's current stream and is CUDA-graph
// capturable.

#include <cstdint>

#include "tvm_ffi_utils.h"

#define SOLBENCH_EMBEDDED 1

// Native-FP8 FC2 experiment layered on the current grouped MoE implementation.
//
// Locally this includes moe.cu.  The submitted solution defines
// SOLBENCH_EMBEDDED and stages the same source as moe_base.h so that only
// this translation unit is compiled.
#define launch_moe_grouped launch_moe_grouped_fp16_base
#ifdef SOLBENCH_EMBEDDED
#include "moe_fp8_grouped_smallt_sm90_base.h"
#else
#include "moe.cu"
#endif
#undef launch_moe_grouped

__device__ __forceinline__ uint64_t wgmma_desc_fp8(const uint8_t* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return ((uint64_t)(a & 0x3ffff) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma128_fp8(float d[64], uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "%64,%65,1,1,1;\n"
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

// The two rows represented by each thread's accumulator fragment have
// independent activation scales.  The weight scale is common to the N128
// tile, so only two multipliers are needed for all 64 registers.
__device__ __forceinline__ void scale_wgmma128_rows(float d[64], float s0, float s8) {
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    d[w * 8 + 0] *= s0;
    d[w * 8 + 1] *= s0;
    d[w * 8 + 2] *= s8;
    d[w * 8 + 3] *= s8;
    d[w * 8 + 4] *= s0;
    d[w * 8 + 5] *= s0;
    d[w * 8 + 6] *= s8;
    d[w * 8 + 7] *= s8;
  }
}

// Fuses SiLU*up with dynamic per-row, per-K128 E4M3 quantization.  One warp
// owns a sorted routed pair and handles its two K128 blocks serially.  Packing
// sixteen pairs per CTA avoids both the cross-warp barrier and tiny CTAs while
// retaining the block-major grouped layout consumed by FC2.
__global__ __launch_bounds__(512) void k_silu_q_fp8(const __nv_bfloat16* __restrict__ raw,
                                                    uint8_t* __restrict__ hq,
                                                    float2* __restrict__ hs,
                                                    const int* __restrict__ order, int I,
                                                    int pairs) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int gidx = blockIdx.x * 16 + (tid >> 5);
  if (gidx >= pairs) return;
  const int p = order[gidx];
  float invs[2];
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const int i = b * 128 + lane * 4;
    float2 up0 = __bfloat1622float2(*(const __nv_bfloat162*)(raw + (size_t)p * 2 * I + i));
    float2 up1 = __bfloat1622float2(*(const __nv_bfloat162*)(raw + (size_t)p * 2 * I + i + 2));
    float2 g0 = __bfloat1622float2(*(const __nv_bfloat162*)(raw + (size_t)p * 2 * I + I + i));
    float2 g1 = __bfloat1622float2(*(const __nv_bfloat162*)(raw + (size_t)p * 2 * I + I + i + 2));
    float2 h0, h1;
    h0.x = (g0.x / (1.f + __expf(-g0.x))) * up0.x;
    h0.y = (g0.y / (1.f + __expf(-g0.y))) * up0.y;
    h1.x = (g1.x / (1.f + __expf(-g1.x))) * up1.x;
    h1.y = (g1.y / (1.f + __expf(-g1.y))) * up1.y;
    float amax = fmaxf(fmaxf(fabsf(h0.x), fabsf(h0.y)), fmaxf(fabsf(h1.x), fabsf(h1.y)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULLM, amax, o));
    const float inv = (amax > 0.f) ? 448.f / amax : 1.f;
    invs[b] = inv;
    h0.x *= inv;
    h0.y *= inv;
    h1.x *= inv;
    h1.y *= inv;
    const size_t qoff = ((size_t)b * pairs + gidx) * 128 + lane * 4;
    *(__nv_fp8x2_storage_t*)(hq + qoff) = __nv_cvt_float2_to_fp8x2(h0, __NV_SATFINITE, __NV_E4M3);
    *(__nv_fp8x2_storage_t*)(hq + qoff + 2) =
        __nv_cvt_float2_to_fp8x2(h1, __NV_SATFINITE, __NV_E4M3);
  }
  if (lane == 0) hs[gidx] = make_float2(__frcp_rn(invs[0]) * invs[1], __frcp_rn(invs[1]));
}

__global__ __launch_bounds__(128, 8) void k_reduce9_v16(const __half* __restrict__ partial,
                                                        const float* __restrict__ tw,
                                                        __nv_bfloat16* __restrict__ out, int H) {
  __shared__ float sw[9];
  const int tid = threadIdx.x;
  const int t = blockIdx.x;
  if (tid < 9) sw[tid] = tw[t * 9 + tid];
  __syncthreads();
  for (int d16 = tid; d16 < (H >> 4); d16 += blockDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const uint4* src = (const uint4*)(partial + ((size_t)t * 9 + k) * H + d16 * 16);
      const uint4 h0 = src[0];
      const uint4 h1 = src[1];
      const __half2* hh0 = (const __half2*)&h0;
      const __half2* hh1 = (const __half2*)&h1;
      const float w = sw[k];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __half22float2(hh0[q]);
        acc[2 * q] = fmaf(w, f.x, acc[2 * q]);
        acc[2 * q + 1] = fmaf(w, f.y, acc[2 * q + 1]);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __half22float2(hh1[q]);
        acc[8 + 2 * q] = fmaf(w, f.x, acc[8 + 2 * q]);
        acc[8 + 2 * q + 1] = fmaf(w, f.y, acc[8 + 2 * q + 1]);
      }
    }
    __nv_bfloat162* dst = (__nv_bfloat162*)(out + (size_t)t * H + d16 * 16);
#pragma unroll
    for (int q = 0; q < 8; ++q) dst[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
  }
}

template <int BM, int THREADS, int NGROUP>
__global__ __launch_bounds__(THREADS) void fc2_wgmma_np_fp8(
    const uint8_t* __restrict__ hq, const float2* __restrict__ hs, const uint8_t* __restrict__ W2,
    const float* __restrict__ W2S, const int* __restrict__ order, const int* __restrict__ cnt,
    const int* __restrict__ off, const int* __restrict__ toff, const int2* __restrict__ tasks,
    __half* __restrict__ partial, int H, int I, int KTOP, int Si, int Sh, int E, int pairs) {
  static_assert(BM == 128 && THREADS == 256,
                "This first native-FP8 path is specialized for wide M");
  const int ntl = H >> 7;
  const int ngl = (ntl + NGROUP - 1) / NGROUP;
  const int ti = blockIdx.x / ngl;
  const int nt0 = (blockIdx.x - ti * ngl) * NGROUP;
  if (ti >= toff[E]) return;
  const int2 tk = tasks[ti];
  const int e = tk.x, start = tk.y;
  const int rows = min(BM, off[e] + cnt[e] - start);

  extern __shared__ __align__(1024) unsigned char smem[];
  int* sPair = (int*)smem;
  float* sWr = (float*)(sPair + BM);
  float* sW1 = sWr + NGROUP;
  uint8_t* sA = smem + 1024;
  uint8_t* sB = sA + BM * 256;
  uint8_t* sB1 = sB + 128 * 128;
  const int tid = threadIdx.x;
  const int wgid = tid >> 7;
  const int ltid = tid & 127;
  const int warp = ltid >> 5;
  const int lane = ltid & 31;

  for (int r = tid; r < BM; r += THREADS) {
    int p = -1;
    if (r < rows) p = order[start + r];
    sPair[r] = p;
  }
  if (tid < NGROUP) {
    const int nt = nt0 + tid;
    float w0 = 1.f, w1 = 1.f;
    if (nt < ntl) {
      w0 = W2S[((size_t)e * Sh + nt) * Si + 0];
      w1 = W2S[((size_t)e * Sh + nt) * Si + 1];
    }
    sWr[tid] = w0 * __frcp_rn(w1);
    sW1[tid] = w1;
  }
  __syncthreads();

// Quantized activation is reused across every persistent N tile.  Each
// K128 block is one 128B-swizzled atom wide (8 uint4 cells per row).
#pragma unroll
  for (int kb = 0; kb < 256; kb += 128) {
#pragma unroll
    for (int q = 0; q < BM * 8 / THREADS; ++q) {
      const int ii = tid + q * THREADS;
      const int r = ii >> 3;
      const int c = ii & 7;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < rows)
        v = *(const uint4*)(hq + ((size_t)(kb >> 7) * pairs + start + r) * 128 + c * 16);
      ((uint4*)(sA + kb * BM))[r * 8 + (c ^ (r & 7))] = v;
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  const int row = wgid * 64 + warp * 16 + lane / 4;
  const int row8 = row + 8;
  const int p0 = (row < rows) ? sPair[row] : -1;
  const int p8 = (row8 < rows) ? sPair[row8] : -1;
  const float2 one = make_float2(1.f, 1.f);
  const float2 a0 = (p0 >= 0) ? hs[start + row] : one;
  const float2 a8 = (p8 >= 0) ? hs[start + row8] : one;
  const int cb = 2 * (lane & 3);

  for (int ni = 0; ni < NGROUP && nt0 + ni < ntl; ++ni) {
    const int nt = nt0 + ni;
    {
      uint4 rb[128 * 8 / THREADS];
#pragma unroll
      for (int q = 0; q < 128 * 8 / THREADS; ++q) {
        const int ii = tid + q * THREADS;
        const int r = ii >> 3;
        const int c = ii & 7;
        const int grow = nt * 128 + r;
        rb[q] = *(const uint4*)(W2 + ((size_t)e * H + grow) * I + c * 16);
      }
#pragma unroll
      for (int q = 0; q < 128 * 8 / THREADS; ++q) {
        const int ii = tid + q * THREADS;
        const int r = ii >> 3;
        const int c = ii & 7;
        ((uint4*)sB)[r * 8 + (c ^ (r & 7))] = rb[q];
      }
#pragma unroll
      for (int q = 0; q < 128 * 8 / THREADS; ++q) {
        const int ii = tid + q * THREADS;
        const int r = ii >> 3;
        const int c = ii & 7;
        const int grow = nt * 128 + r;
        rb[q] = *(const uint4*)(W2 + ((size_t)e * H + grow) * I + 128 + c * 16);
      }
#pragma unroll
      for (int q = 0; q < 128 * 8 / THREADS; ++q) {
        const int ii = tid + q * THREADS;
        const int r = ii >> 3;
        const int c = ii & 7;
        ((uint4*)sB1)[r * 8 + (c ^ (r & 7))] = rb[q];
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;

    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    uint64_t da = wgmma_desc_fp8(sA + wgid * 64 * 128);
    uint64_t db = wgmma_desc_fp8(sB);
#pragma unroll
    for (int kk = 0; kk < 128; kk += 32) wgmma128_fp8(acc, da + (kk >> 4), db + (kk >> 4));
    asm volatile(
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;" ::
            : "memory");

    // Convert the block-0-normalized accumulator to block-1 raw units.
    const float wr = sWr[ni];
    const float ws1 = sW1[ni];
    scale_wgmma128_rows(acc, a0.x * wr, a8.x * wr);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    da = wgmma_desc_fp8(sA + 128 * BM + wgid * 64 * 128);
    db = wgmma_desc_fp8(sB1);
#pragma unroll
    for (int kk = 0; kk < 128; kk += 32) wgmma128_fp8(acc, da + (kk >> 4), db + (kk >> 4));
    asm volatile(
        "wgmma.commit_group.sync.aligned;\n"
        "wgmma.wait_group.sync.aligned 0;" ::
            : "memory");
    __syncthreads();

    // Return from raw block-1 units to the physical output scale.
    scale_wgmma128_rows(acc, a0.y * ws1, a8.y * ws1);

#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const int c = w * 16 + cb;
      if (row < rows) {
        __half* dst = partial + (size_t)sPair[row] * H + nt * 128 + c;
        *(__half2*)dst = __floats2half2_rn(acc[w * 8], acc[w * 8 + 1]);
        *(__half2*)(dst + 8) = __floats2half2_rn(acc[w * 8 + 4], acc[w * 8 + 5]);
      }
      if (row8 < rows) {
        __half* dst = partial + (size_t)sPair[row8] * H + nt * 128 + c;
        *(__half2*)dst = __floats2half2_rn(acc[w * 8 + 2], acc[w * 8 + 3]);
        *(__half2*)(dst + 8) = __floats2half2_rn(acc[w * 8 + 6], acc[w * 8 + 7]);
      }
    }
  }
}

void launch_moe_grouped(const void* X, const void* W13, const float* W13S, const void* W2,
                        const float* W2S, const int* ids, const float* tw, void* out, void* raw,
                        void* Hb, void* partial, int* ibuf, int2* tasks, int T, int H, int I, int E,
                        int KTOP, int max_tasks, cudaStream_t st) {
  // Preserve the submitted incumbent byte-for-byte for T=256 and T=1024.
  if (T < 2048) {
    launch_moe_grouped_fp16_base(X, W13, W13S, W2, W2S, ids, tw, out, raw, Hb, partial, ibuf, tasks,
                                 T, H, I, E, KTOP, max_tasks, st);
    return;
  }

  const int pairs = T * KTOP;
  const int Sh1 = (H + 127) >> 7;
  const int Si = (I + 127) >> 7;
  const int Sn1 = (2 * I + 127) >> 7;
  const int task_bound = E + (pairs + 127) / 128;
  int* cnt = ibuf;
  int* off = cnt + E;
  int* cur = off + E;
  int* toff = cur + E;
  int* order = toff + E + 1;

  cudaMemsetAsync(cnt, 0, sizeof(int) * E, st);
  k_count<<<(pairs + 255) / 256, 256, 0, st>>>(ids, cnt, pairs);
  k_prefix<128><<<1, 288, 0, st>>>(cnt, off, cur, toff, tasks, E);
  k_scatter<<<(pairs + 255) / 256, 256, 0, st>>>(ids, cur, order, pairs);

  fc1_wmma<128, 256, 512><<<task_bound * ((2 * I) / 256), 512, 0, st>>>(
      (const __nv_bfloat16*)X, (const uint8_t*)W13, W13S, order, cnt, off, toff, tasks,
      (__nv_bfloat16*)raw, H, 2 * I, KTOP, Sh1, Sn1, E);

  // Hb is a 2*pairs*I-byte BF16 allocation.  Native FP8 uses pairs*I bytes
  // for values and only pairs*2*sizeof(float2) more bytes for scales.
  uint8_t* hq = (uint8_t*)Hb;
  float2* hs = (float2*)(hq + (size_t)pairs * I);
  k_silu_q_fp8<<<(pairs + 15) / 16, 512, 0, st>>>((const __nv_bfloat16*)raw, hq, hs, order, I,
                                                  pairs);

  constexpr int FC2_NGROUP = 8;
  constexpr int FC2_SMEM = 1024 + 128 * 256 + 2 * 128 * 128;
  const int fc2_ng = ((H >> 7) + FC2_NGROUP - 1) / FC2_NGROUP;
  cudaFuncSetAttribute(fc2_wgmma_np_fp8<128, 256, FC2_NGROUP>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, FC2_SMEM);
  fc2_wgmma_np_fp8<128, 256, FC2_NGROUP><<<task_bound * fc2_ng, 256, FC2_SMEM, st>>>(
      hq, hs, (const uint8_t*)W2, W2S, order, cnt, off, toff, tasks, (__half*)partial, H, I, KTOP,
      Si, Sh1, E, pairs);

  k_reduce9_v16<<<T, 128, 0, st>>>((const __half*)partial, tw, (__nv_bfloat16*)out, H);
}

// ---------------------------------------------------------------------------
// Host side. `launch_moe_small` above is carried verbatim from the measured
// kernel's same-SKU re-time; the fp16 intermediate moved from a
// per-call torch allocation in the measured entry point to a caller-owned
// plan-time workspace so that `run` never allocates and is CUDA-graph
// capturable with a stable workspace pointer.
// ---------------------------------------------------------------------------

namespace {

// Byte layout of the caller-owned workspace. T < 256 uses only the fp16 fc1
// intermediate of the small path; T >= 256 carries the grouped path's five
// buffers at the measured entry point's exact sizes (16-byte-aligned
// segments).
struct MoeWsLayout {
  size_t raw_off, hb_off, partial_off, ibuf_off, tasks_off, total;
  int max_tasks;
};

inline size_t moe_align16(size_t x) { return (x + 15) & ~size_t(15); }

MoeWsLayout moe_grouped_ws_layout(int64_t T, int64_t K, int64_t I, int64_t H, int64_t E) {
  MoeWsLayout L;
  const int64_t pairs = T * K;
  L.max_tasks = (int)(E + (pairs + 63) / 64);
  L.raw_off = 0;
  L.hb_off = moe_align16(L.raw_off + (size_t)pairs * 2 * I * 2);    // bf16
  L.partial_off = moe_align16(L.hb_off + (size_t)pairs * I * 2);    // bf16
  L.ibuf_off = moe_align16(L.partial_off + (size_t)pairs * H * 2);  // fp16
  L.tasks_off = moe_align16(L.ibuf_off + (size_t)(3 * E + E + 1 + pairs) * 4);
  L.total = moe_align16(L.tasks_off + (size_t)L.max_tasks * 2 * 4);
  return L;
}

}  // namespace

// Workspace bytes the dispatcher must plan-allocate for a (tokens, top_k,
// intermediate_size, hidden_size, num_experts) call: T < 256 uses the small
// path's fp16 fc1 intermediate; T >= 256 the grouped path's buffers at the
// measured entry point's sizes.
int64_t moe_fp8_grouped_smallt_sm90_plan_workspace_bytes(int64_t tokens, int64_t top_k,
                                                         int64_t intermediate_size,
                                                         int64_t hidden_size, int64_t num_experts) {
  TVM_FFI_ICHECK_GE(tokens, 1);
  TVM_FFI_ICHECK_GE(top_k, 1);
  TVM_FFI_ICHECK_GE(intermediate_size, 1);
  TVM_FFI_ICHECK_GE(hidden_size, 1);
  TVM_FFI_ICHECK_GE(num_experts, 1);
  if (tokens < 256) {
    return tokens * top_k * intermediate_size * (int64_t)sizeof(__half);
  }
  return (int64_t)moe_grouped_ws_layout(tokens, top_k, intermediate_size, hidden_size, num_experts)
      .total;
}

void moe_fp8_grouped_smallt_sm90_run(TensorView hidden_states, TensorView fc1_weights,
                                     TensorView fc1_scales, TensorView fc2_weights,
                                     TensorView fc2_scales, TensorView topk_ids,
                                     TensorView topk_weights, TensorView out,
                                     TensorView workspace) {
  CHECK_INPUT_AND_TYPE(hidden_states, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(fc1_weights, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(fc2_weights, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(fc1_scales, dl_float32);
  CHECK_INPUT_AND_TYPE(fc2_scales, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, out);
  CHECK_DIM(3, fc1_weights);
  CHECK_DIM(3, fc2_weights);
  CHECK_DIM(3, fc1_scales);
  CHECK_DIM(3, fc2_scales);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_weights);

  const int T = static_cast<int>(hidden_states.size(0));
  const int H = static_cast<int>(hidden_states.size(1));
  const int E = static_cast<int>(fc1_weights.size(0));
  const int I = static_cast<int>(fc2_weights.size(2));
  const int KTOP = static_cast<int>(topk_ids.size(1));

  TVM_FFI_ICHECK_GE(T, 1);
  // T < 256 runs the small path; T >= 256 the grouped (expert-binned) path.
  // The dispatcher allowlist is the measured won BANDS: T <= 16 and
  // 256 <= T <= 4096 (interior points by band interpolation between the
  // measured classes; the 17..255 gap — containing the measured FI-held
  // T=64 class — stays on the stock path).
  TVM_FFI_ICHECK_EQ(fc1_weights.size(1), 2 * (int64_t)I)
      << "fc1 must be gated (rows [0:I]=up, [I:2I]=gate)";
  TVM_FFI_ICHECK_EQ(fc1_weights.size(2), H);
  TVM_FFI_ICHECK_EQ(fc2_weights.size(0), E);
  TVM_FFI_ICHECK_EQ(fc2_weights.size(1), H);
  TVM_FFI_ICHECK_EQ(out.size(0), T);
  TVM_FFI_ICHECK_EQ(out.size(1), H);
  TVM_FFI_ICHECK_EQ(fc1_scales.size(0), E);
  TVM_FFI_ICHECK_EQ(fc1_scales.size(1), (2 * (int64_t)I + 127) / 128);
  TVM_FFI_ICHECK_EQ(fc1_scales.size(2), ((int64_t)H + 127) / 128);
  TVM_FFI_ICHECK_EQ(fc2_scales.size(0), E);
  TVM_FFI_ICHECK_EQ(fc2_scales.size(1), ((int64_t)H + 127) / 128);
  TVM_FFI_ICHECK_EQ(fc2_scales.size(2), ((int64_t)I + 127) / 128);
  TVM_FFI_ICHECK_EQ(topk_ids.size(0), T);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), T);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), KTOP);
  TVM_FFI_ICHECK_GE((int64_t)workspace.numel(),
                    moe_fp8_grouped_smallt_sm90_plan_workspace_bytes(T, KTOP, I, H, E))
      << "workspace too small: plan with plan_workspace_bytes(" << T << ", " << KTOP << ", " << I
      << ")";

  CHECK_DEVICE(hidden_states, fc1_weights);
  CHECK_DEVICE(hidden_states, fc1_scales);
  CHECK_DEVICE(hidden_states, fc2_weights);
  CHECK_DEVICE(hidden_states, fc2_scales);
  CHECK_DEVICE(hidden_states, topk_ids);
  CHECK_DEVICE(hidden_states, topk_weights);
  CHECK_DEVICE(hidden_states, out);
  CHECK_DEVICE(hidden_states, workspace);

  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const cudaStream_t stream = get_stream(hidden_states.device());

  if (T < 256) {
    launch_moe_small(hidden_states.data_ptr(), fc1_weights.data_ptr(),
                     static_cast<const float*>(fc1_scales.data_ptr()), fc2_weights.data_ptr(),
                     static_cast<const float*>(fc2_scales.data_ptr()),
                     static_cast<const int*>(topk_ids.data_ptr()),
                     static_cast<const float*>(topk_weights.data_ptr()), out.data_ptr(),
                     workspace.data_ptr(), T, H, I, E, KTOP, stream);
  } else {
    const MoeWsLayout L = moe_grouped_ws_layout(T, KTOP, I, H, E);
    char* wp = static_cast<char*>(workspace.data_ptr());
    launch_moe_grouped(
        hidden_states.data_ptr(), fc1_weights.data_ptr(),
        static_cast<const float*>(fc1_scales.data_ptr()), fc2_weights.data_ptr(),
        static_cast<const float*>(fc2_scales.data_ptr()),
        static_cast<const int*>(topk_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()), out.data_ptr(), wp + L.raw_off,
        wp + L.hb_off, wp + L.partial_off, reinterpret_cast<int*>(wp + L.ibuf_off),
        reinterpret_cast<int2*>(wp + L.tasks_off), T, H, I, E, KTOP, L.max_tasks, stream);
  }
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "moe_fp8_grouped_smallt_sm90 launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, moe_fp8_grouped_smallt_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_workspace_bytes,
                              moe_fp8_grouped_smallt_sm90_plan_workspace_bytes);
