/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
namespace flashinfer::glm5_tp8::q1_detail {

__device__ __forceinline__ void red_relaxed_gpu_add(uint32_t* address, uint32_t value) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ float k3_warp_max_broadcast(float value) {
  float result;
  asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float k3_warp_sum_broadcast(float value) {
  value = warp_sum(value);
  return __shfl_sync(0xffffffffu, value, 0);
}
union Bf16Pair {
  __nv_bfloat162 pair;
  uint32_t bits;
};
struct HmmaAcc {
  float value[4];
};
struct RawFp8x8 {
  uint32_t lo;
  uint32_t hi;
};
union F32Pair {
  uint64_t bits;
  float value[2];
};

__device__ __forceinline__ uint32_t load_relaxed_gpu(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

__device__ __forceinline__ void red_release_gpu_add(uint32_t* address, uint32_t value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void st_relaxed_sys_b128(void* address, uint32_t w0, uint32_t w1,
                                                    uint32_t w2, uint32_t w3) {
  const uint64_t lo = uint64_t(w0) | (uint64_t(w1) << 32);
  const uint64_t hi = uint64_t(w2) | (uint64_t(w3) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.sys.global.b128 [%0], r;\n\t}" ::
          "l"(address),
      "l"(lo), "l"(hi)
      : "memory");
}

__device__ __forceinline__ void ld_relaxed_sys_b128(const void* address, uint64_t& lo,
                                                    uint64_t& hi) {
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.sys.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo), "=l"(hi)
      : "l"(address)
      : "memory");
}

// The aligned LL record contains both IDs and tags. Unpack only after one
// scalar system-scope load, matching the publisher's scalar b128 store.
__device__ __forceinline__ uint4 load_ll_record(const void* address) {
  uint64_t lo, hi;
  ld_relaxed_sys_b128(address, lo, hi);
  return make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32),
                    static_cast<uint32_t>(hi), static_cast<uint32_t>(hi >> 32));
}

__device__ __forceinline__ uint32_t smem_u32(const void* generic) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(generic));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t arrival_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(arrival_count) : "memory");
}

__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "MBAR_WAIT_LOOP:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra MBAR_WAIT_DONE;\n\t"
      "bra MBAR_WAIT_LOOP;\n\t"
      "MBAR_WAIT_DONE:\n\t"
      "}" ::"r"(bar),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ void fp8_to_bf16_fragment(const RawFp8x8& raw, uint32_t (&fragment)[4]) {
  asm volatile(
      "{\n\t"
      ".reg .b16 h0,h1,h2,h3;\n\t"
      "mov.b32 {h0,h1},%4;\n\t"
      "mov.b32 {h2,h3},%5;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0,h0;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1,h1;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %2,h2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %3,h3;\n\t"
      "}"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "r"(raw.lo), "r"(raw.hi));
}

__device__ __forceinline__ uint32_t hmma_smem_addr(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void hmma_ldsm_x4(uint32_t (&value)[4], const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void hmma_ldsm_x4(uint32_t (&value)[2][2],
                                             const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(value[0][0]), "=r"(value[0][1]), "=r"(value[1][0]), "=r"(value[1][1])
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void hmma_ldsm_x4_trans(uint32_t (&value)[4],
                                                   const __nv_bfloat16* pointer) {
  const uint32_t address = hmma_smem_addr(pointer);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 "
      "{%0,%1,%2,%3}, [%4];"
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void hmma_load_a_16x16(const __nv_bfloat16* base, int stride,
                                                  uint32_t (&value)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int source_row = (lane & 7) + ((matrix & 1) << 3);
  const int source_col = (matrix >> 1) << 3;
  hmma_ldsm_x4(value, base + source_row * stride + source_col);
}

__device__ __forceinline__ void hmma_load_a_16x16_trans(const __nv_bfloat16* base, int stride,
                                                        uint32_t (&value)[4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int matrix = lane >> 3;
  const int source_row = (lane & 7) + ((matrix >> 1) << 3);
  const int source_col = (matrix & 1) << 3;
  hmma_ldsm_x4_trans(value, base + source_row * stride + source_col);
}

__device__ __forceinline__ void hmma(HmmaAcc& accumulator, const uint32_t (&a)[4],
                                     const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(accumulator.value[0]), "+f"(accumulator.value[1]), "+f"(accumulator.value[2]),
        "+f"(accumulator.value[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

__device__ __forceinline__ void bulk_copy_g2s_l2(uint32_t dst_smem, const void* src_global,
                                                 uint32_t bytes, uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1], %2, [%3], %4;" ::"r"(dst_smem),
      "l"(src_global), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}

__device__ __forceinline__ RawFp8x8 lds_fp8x8(uint32_t address) {
  RawFp8x8 raw;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(raw.lo), "=r"(raw.hi) : "r"(address));
  return raw;
}

__device__ __forceinline__ uint32_t lds_u32(uint32_t address) {
  uint32_t value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
  return value;
}

__device__ __forceinline__ float lds_f32(uint32_t address) {
  float value;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"(address));
  return value;
}
}  // namespace flashinfer::glm5_tp8::q1_detail
