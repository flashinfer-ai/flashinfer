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
 *
 * SM90 (Hopper) NVFP4-weight x BF16-activation (W4A16) fused MoE: shared PTX
 * helpers and the packed weight layout.
 *
 * Adapted from gauravjain14/muse-okc-submissions (muse_moe_w4a16, Apache-2.0),
 * which itself builds on vjeux/okc-2026 and alygg77/okc-muse-spark-kernels
 * (Apache-2.0); see NOTICE. Changes from the upstream kernel: ported from an
 * NVRTC module into FlashInfer's nvcc JIT, the model constants (hidden size,
 * intermediate size, expert count) became template / runtime parameters, and
 * the multi-GPU paths (sequence-parallel scatter, replica plans, remote
 * experts) were removed.
 */
#ifndef FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_COMMON_CUH_
#define FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_COMMON_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace sm90_w4a16_moe {

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef long long i64;

#define FI_W4A16_DEVI __device__ __forceinline__
#define FI_W4A16_AS(T, p) reinterpret_cast<T*>(p)

// ---------------------------------------------------------------------------
// Packed weight layout (produced by flashinfer.fused_moe.sm90_nvfp4_w4a16
// prepare_sm90_nvfp4_w4a16_weights). A matrix [N, K] of one expert is cut into
// blocks of 256 output rows x 256 reduction columns. Each block is
// kWeightBlockBytes: 16 row groups x 2048 B of FP4 codes (stored in the MMA
// fragment order the in-register decoder expects), then 16 x 192 B of 6-bit
// block scales. Blocks are ordered [expert][k_chunk][row_block].
// ---------------------------------------------------------------------------
constexpr int kWeightBlockBytes = 35840;
constexpr int kWeightCodeBytes = 32768;
// Tokens are routed through exactly this many top-k slots; the host pads
// smaller top-k with id -1 / weight 0.
constexpr int kTopKSlots = 8;
// Experts are tracked as one bit per warp lane.
constexpr int kMaxLocalExperts = 32;

FI_W4A16_DEVI const u8* weight_block(const void* w, int e, int nkc, int nrg, int kc, int rg) {
  return FI_W4A16_AS(const u8, w) +
         ((size_t)(e * nkc + kc) * (nrg >> 4) + (rg >> 4)) * kWeightBlockBytes + (rg & 15) * 2048;
}
FI_W4A16_DEVI const u8* weight_scales(const void* w, int e, int nkc, int nrg, int kc, int rg) {
  return weight_block(w, e, nkc, nrg, kc, rg & ~15) + kWeightCodeBytes + (rg & 15) * 192;
}

template <class T>
FI_W4A16_DEVI T shfl(T v, int src) {
  return __shfl_sync(~0u, v, src);
}
template <class T>
FI_W4A16_DEVI T shfl_xor(T v, int m) {
  return __shfl_xor_sync(~0u, v, m);
}
template <class T>
FI_W4A16_DEVI T shfl_up(T v, int d) {
  return __shfl_up_sync(~0u, v, d);
}
FI_W4A16_DEVI u32 ballot(bool p) { return __ballot_sync(~0u, p); }
// inclusive prefix sum over the warp
FI_W4A16_DEVI int wscan(int v, int lane) {
  for (int o = 1; o < 32; o <<= 1) {
    const int a = shfl_up(v, o);
    if (lane >= o) v += a;
  }
  return v;
}
template <int N, class T>
FI_W4A16_DEVI void advance(T& s, u32& ph) {
  if (++s == N) {
    s = 0;
    ph ^= 1;
  }
}
FI_W4A16_DEVI u32 smem_u32(const void* p) { return static_cast<u32>(__cvta_generic_to_shared(p)); }
FI_W4A16_DEVI void mbar_init(u32 bar, u32 n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
FI_W4A16_DEVI void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
FI_W4A16_DEVI void mbar_arrive(u32 bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
FI_W4A16_DEVI void mbar_expect(u32 bar, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
FI_W4A16_DEVI void mbar_wait(u32 bar, u32 ph) {
  asm volatile(
      "{ .reg .pred P1; W%=: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; @!P1 bra W%=; "
      "}" ::"r"(bar),
      "r"(ph)
      : "memory");
}
FI_W4A16_DEVI u32 cl_rank() {
  u32 r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
FI_W4A16_DEVI u32 mapa(u32 a, u32 rank) {
  u32 d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}
FI_W4A16_DEVI void cl_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::
                   : "memory");
}
FI_W4A16_DEVI void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
FI_W4A16_DEVI void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
FI_W4A16_DEVI void stmatrix_t(u32 a, const u32 (&r)[4]) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(a),
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
               : "memory");
}
FI_W4A16_DEVI void sts32(u32 a, u32 v) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
FI_W4A16_DEVI uint4 lds128(u32 a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}
FI_W4A16_DEVI void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
FI_W4A16_DEVI void fence_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
FI_W4A16_DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, "
      "[%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
FI_W4A16_DEVI void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar, u64 pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(pol)
      : "memory");
}
FI_W4A16_DEVI void bulk_s2g(void* dst, u32 src, u32 bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src),
               "r"(bytes)
               : "memory");
}
FI_W4A16_DEVI void prefetch_l2(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
FI_W4A16_DEVI void cpa16(u32 dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
FI_W4A16_DEVI void bulk_commit_wait_read() {
  asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
}
FI_W4A16_DEVI void bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
FI_W4A16_DEVI void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
FI_W4A16_DEVI void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
FI_W4A16_DEVI int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
FI_W4A16_DEVI void red_release(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
FI_W4A16_DEVI u64 policy_evict_first() {
  u64 p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
FI_W4A16_DEVI u64 policy_evict_last() {
  u64 p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
template <int kRegs>
FI_W4A16_DEVI void nreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kRegs) : "memory");
}
template <int kRegs>
FI_W4A16_DEVI void nreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kRegs) : "memory");
}
FI_W4A16_DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
FI_W4A16_DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int kPending>
FI_W4A16_DEVI void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(kPending) : "memory");
}
// 128B-swizzled K-major shared-memory descriptor (wgmma B operand).
FI_W4A16_DEVI u64 sw128_desc(u32 a) {
  return (u64)((a & 0x3FFFF) >> 4) | ((u64)1 << 16) | ((u64)64 << 32) | ((u64)1 << 62);
}
FI_W4A16_DEVI float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
FI_W4A16_DEVI u16 f2bf(float f) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
FI_W4A16_DEVI float bf_lo(u32 v) { return __uint_as_float(v << 16); }
FI_W4A16_DEVI float bf_hi(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
FI_W4A16_DEVI u32 bf2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
FI_W4A16_DEVI u32 rotl(u32 x, u32 r) { return __funnelshift_l(x, x, r); }
// bf16x2 a times the low (H = 0) or high bf16 of b
#define FI_W4A16_BMUL2S(X)                                                         \
  asm("{ .reg .b16 l, h; .reg .b32 t; mov.b32 {l, h}, %2; mov.b32 t, {" #X ", " #X \
      "}; mul.rn.bf16x2 %0, %1, t; }"                                              \
      : "=r"(d)                                                                    \
      : "r"(a), "r"(b))
template <int H>
FI_W4A16_DEVI u32 bmul2s(u32 a, u32 b) {
  u32 d;
  if (H == 0)
    FI_W4A16_BMUL2S(l);
  else
    FI_W4A16_BMUL2S(h);
  return d;
}
#undef FI_W4A16_BMUL2S

// wgmma.m64nNk16.f32.bf16.bf16 with A from registers. The explicit
// specializations (one per N, each with N / 2 accumulator operands) are
// generated by flashinfer/jit/sm90_nvfp4_w4a16_moe.py into the JIT source
// directory and must be included by the translation unit before the kernels
// below are instantiated.
template <int N>
FI_W4A16_DEVI void wgmma_rs(float* d, const u32* a, u64 desc);

}  // namespace sm90_w4a16_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_COMMON_CUH_
