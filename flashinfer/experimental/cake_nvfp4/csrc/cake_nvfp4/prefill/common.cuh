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
#pragma once
// PTX helpers shared by the prefill kernels of the Nemotron 3.5 Super NVFP4-KV backend (sm_100a).
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "the cake-nvfp4 prefill kernels need sm_100: tcgen05 and the three-input max.f32"
#endif

namespace flashinfer {
namespace cake_nvfp4 {
namespace prefill {

// The TMA descriptors of a kernel, passed by value as its last parameter. Each takes a 128-byte
// slot: descriptor i is at offset 128 * i.
template <int N>
struct alignas(128) TensorMaps {
  CUtensorMap maps[N];
};
static_assert(sizeof(TensorMaps<3>) == 3 * 128, "tensor map slots");

// 1 in one lane of a converged warp, 0 in the others.
__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}\n"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

// Blocking wait; the suspend-time hint lets a waiting warp sleep instead of spinning.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase), "r"(ticks)
      : "memory");
}

// One elected lane makes the barrier track the MMAs issued so far.
__device__ __forceinline__ void elect_commit(int mbar_addr) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];\n\t"
      "}\n" ::"r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

// mbarrier.init in the lanes where pred is set.
__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %2, 0;\n\t"
      "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(count), "r"(pred));
}

// exp2, flushing denormals to zero.
__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

// Packed f32x2 arithmetic: one instruction for two values.
__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
  float2 r;
  asm("add.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
  return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
  float2 r;
  asm("sub.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
  return r;
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
  float2 r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
        "l"(*(unsigned long long*)&c));
  return r;
}

__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
  asm volatile(
      "{\n\t"
      ".reg .pred leader;\n\t"
      "elect.sync _|leader, 0xFFFFFFFF;\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%0];\n\t"
      "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
      ".shared::cluster.b64 [%1];\n\t"
      "}\n" ::"r"(mbar_addr0),
      "r"(mbar_addr1)
      : "memory");
}

__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int w, int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
      : "memory");
}

// Broadcast of lane 0, so that values derived from it are provably warp-uniform.
__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
  uint32_t result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(result) : "r"(val));
  return result;
}

// Bits of a value as an unsigned word.
__device__ __forceinline__ unsigned int as_u32(float v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
  return u;
}

__device__ __forceinline__ unsigned int as_u32(unsigned int v) { return v; }

__device__ __forceinline__ unsigned int as_u32(int v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
  return u;
}

// As make_warp_uniform, for the kernels without TMA and tcgen05.
__device__ __forceinline__ int broadcast_lane0(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

}  // namespace prefill
}  // namespace cake_nvfp4
}  // namespace flashinfer
