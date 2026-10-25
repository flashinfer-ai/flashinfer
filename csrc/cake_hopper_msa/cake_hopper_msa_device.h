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

// Shared device preamble of the Cake Hopper (sm_90a) MSA programs: typedefs, tensor-map ABI, CUDA
// includes and the compiler helper library every kernel translation unit carried inline.  Hand
// fold of exporter output pending exporter support; the bodies are the generated ones, verbatim.
// A translation unit that defines CAKE_WARP_UNIFORM_INT before including this header gets the
// int make_warp_uniform(int) body (the top-k programs); every other unit gets uint32_t(uint32_t).
#ifndef CAKE_HOPPER_MSA_DEVICE_H_
#define CAKE_HOPPER_MSA_DEVICE_H_

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(64) CakeTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F

#include <math_constants.h>

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

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr), "r"(count) : "memory");
}

// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
      " P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}\n" ::"r"(mbar_addr),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr) : "memory");
}

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                                 int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tma_2d_gmem2smem(int dst, const void* tmap_ptr, int x, int y,
                                                 int mbar_addr) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void cp_async_bulk_gmem2smem(unsigned smem_addr, const void* gmem_ptr,
                                                        unsigned bytes, int mbar_addr) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];" ::"r"(smem_addr),
      "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
      : "memory");
}

#if defined(CAKE_WARP_UNIFORM_INT)
__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}
#else
__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
  uint32_t result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(result) : "r"(val));
  return result;
}
#endif

__device__ __forceinline__ unsigned int __as_u32(float v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
  return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
  return u;
}

#endif  // CAKE_HOPPER_MSA_DEVICE_H_
