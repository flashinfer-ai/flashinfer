/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

// Framework-independent host/device primitives for the SM100 PatchShift
// 3x3x3 Conv3d kernels. Standalone CLI and benchmark concerns intentionally
// do not belong in this header.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <utility>

namespace flashinfer::conv3d::patchshift {

using Element = __nv_bfloat16;
using TensorMap = CUtensorMap;
constexpr CUtensorMapDataType kTensorMapDataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

__host__ __device__ __forceinline__ Element element_from_float(float value) {
  return __float2bfloat16(value);
}

__host__ __device__ __forceinline__ float element_to_float(Element value) {
  return __bfloat162float(value);
}

__host__ __device__ constexpr int round_up(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int warp_id() { return int(threadIdx.x) >> 5; }
__device__ __forceinline__ int lane_id() { return int(threadIdx.x) & 31; }

__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t elected_lane = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 rx;\n\t"
      ".reg .pred px;\n\t"
      "elect.sync rx|px, %2;\n\t"
      "@px mov.u32 %1, 1;\n\t"
      "mov.u32 %0, rx;\n\t"
      "}\n"
      : "+r"(elected_lane), "+r"(pred)
      : "r"(0xffffffffu));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, int arrive_count) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(arrive_count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier,
                                                          uint32_t transaction_bytes) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n" ::"r"(transaction_bytes),
               "r"(addr)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_remote(uint64_t* local_barrier,
                                                                 uint32_t transaction_bytes,
                                                                 uint32_t remote_cta_rank) {
  uint32_t local_addr = smem_ptr_to_uint(local_barrier);
  asm volatile(
      "{\n\t"
      ".reg .b32 remote_addr;\n\t"
      "mapa.shared::cluster.u32 remote_addr, %0, %1;\n\t"
      "mbarrier.arrive.expect_tx.shared::cluster.b64 _, "
      "[remote_addr], %2;\n\t"
      "}\n" ::"r"(local_addr),
      "r"(remote_cta_rank), "r"(transaction_bytes)
      : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* barrier, int phase) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  uint32_t complete = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n"
      : "=r"(complete)
      : "r"(addr), "r"(phase)
      : "memory");
  return complete != 0;
}

__device__ __forceinline__ void tma_descriptor_fence_acquire(void const* map) {
  uint64_t map_address = reinterpret_cast<uint64_t>(map);
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
               :
               : "l"(map_address)
               : "memory");
}

__device__ __forceinline__ void tma_load_5d(void const* map, uint64_t* barrier, void* smem, int c0,
                                            int c1, int c2, int c3, int c4) {
  cute::SM90_TMA_LOAD_5D::copy(map, barrier,
                               static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL), smem,
                               c0, c1, c2, c3, c4);
}

__device__ __forceinline__ void tma_load_5d_multicast(void const* map, uint64_t* barrier,
                                                      uint16_t cta_mask, void* smem, int c0, int c1,
                                                      int c2, int c3, int c4) {
  cute::SM90_TMA_LOAD_MULTICAST_5D::copy(
      map, barrier, cta_mask, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL), smem,
      c0, c1, c2, c3, c4);
}

__device__ __forceinline__ void fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_alloc(uint32_t* tmem_base, int columns) {
  uint32_t addr = smem_ptr_to_uint(tmem_base);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(addr),
               "r"(columns)
               : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_base, int columns) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_base),
               "r"(columns)
               : "memory");
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint64_t* barrier) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  if (elect_one_sync()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n" ::"r"(addr)
        : "memory");
  }
}

__device__ __forceinline__ void tcgen05_commit_multicast(uint64_t* barrier, uint16_t cta_mask) {
  uint32_t addr = smem_ptr_to_uint(barrier);
  if (elect_one_sync()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
        "shared::cluster.multicast::cluster.b64 [%0], %1;\n" ::"r"(addr),
        "h"(cta_mask)
        : "memory");
  }
}

__device__ __forceinline__ void tcgen05_wait_tmem_load() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_load_32dp32b_x32(uint32_t address, uint32_t* values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]),
        "=r"(values[5]), "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]),
        "=r"(values[10]), "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]),
        "=r"(values[15]), "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]),
        "=r"(values[20]), "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]),
        "=r"(values[25]), "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]),
        "=r"(values[30]), "=r"(values[31])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void tcgen05_load_32dp32b_x64(uint32_t address, uint32_t* values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, [%64];\n"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]),
        "=r"(values[5]), "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]),
        "=r"(values[10]), "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]),
        "=r"(values[15]), "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]),
        "=r"(values[20]), "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]),
        "=r"(values[25]), "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]),
        "=r"(values[30]), "=r"(values[31]), "=r"(values[32]), "=r"(values[33]), "=r"(values[34]),
        "=r"(values[35]), "=r"(values[36]), "=r"(values[37]), "=r"(values[38]), "=r"(values[39]),
        "=r"(values[40]), "=r"(values[41]), "=r"(values[42]), "=r"(values[43]), "=r"(values[44]),
        "=r"(values[45]), "=r"(values[46]), "=r"(values[47]), "=r"(values[48]), "=r"(values[49]),
        "=r"(values[50]), "=r"(values[51]), "=r"(values[52]), "=r"(values[53]), "=r"(values[54]),
        "=r"(values[55]), "=r"(values[56]), "=r"(values[57]), "=r"(values[58]), "=r"(values[59]),
        "=r"(values[60]), "=r"(values[61]), "=r"(values[62]), "=r"(values[63])
      : "r"(address)
      : "memory");
}

template <size_t... Indices>
__device__ __forceinline__ void tcgen05_load_32dp32b_x128_impl(uint32_t address, uint32_t* values,
                                                               std::index_sequence<Indices...>) {
  cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b128x::copy(address, values[Indices]...);
}

__device__ __forceinline__ void tcgen05_load_32dp32b_x128(uint32_t address, uint32_t* values) {
  tcgen05_load_32dp32b_x128_impl(address, values, std::make_index_sequence<128>{});
}

__device__ __forceinline__ uint64_t pack_k16_desc(Element* ptr, int rows) {
  uint32_t start_address = smem_ptr_to_uint(ptr) >> 4;
  uint64_t desc = 0;
  desc |= uint64_t(start_address & 0x3fffu);
  desc |= uint64_t(uint32_t(rows) & 0x3fffu) << 16;
  desc |= uint64_t(8u) << 32;
  desc |= uint64_t(1u) << 46;
  return desc;
}

__device__ __forceinline__ uint16_t element_bits_from_float(float value) {
  uint16_t bits;
  asm volatile("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(bits) : "f"(value));
  return bits;
}

__host__ __device__ __forceinline__ uint64_t shift_desc(int shift) {
  return uint64_t(shift & 0x3f) << 56;
}

}  // namespace flashinfer::conv3d::patchshift
