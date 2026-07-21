#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>  // CUtensorMap (Driver API) for the tma_load_2d signature

#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90.hpp>

namespace monomoe {

// ── cp.async scalar helpers (sm_80+) ──────────────────────────────────────
//
// Non-blocking 4-byte global→shared copies for small latency-sensitive
// prefetches (e.g. the per-expert block-scale tile).  Issue with
// `cp_async_cg_4`, checkpoint with `cp_async_commit_group`, drain with
// `cp_async_wait_group<N>()` before reading the destination.

__device__ static inline void cp_async_cg_4(void* smem_dst, const void* gmem_src) {
  const std::uint32_t smem_addr = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ static inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <std::uint32_t N>
__device__ static inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ── Hopper WGMMA (sm_90a) helpers ─────────────────────────────────────────
//
// Wrappers for `wgmma.mma_async` with fp8 (e4m3) operands and fp32
// accumulators.  Key semantics vs `mma.sync`:
//
//   * Issued by a full warpgroup (4 warps); every thread must execute it.
//   * A and B live in shared memory, addressed by 64-bit matrix
//     descriptors (base address + leading/stride byte offsets + swizzle).
//   * The accumulator D stays in registers, distributed across the WG.
//   * Asynchronous: `commit_group` checkpoints, `wait_group<N>` waits for
//     all but the last N groups, and `wgmma_fence` orders prior register
//     writes before the SHM reads of subsequent WGMMAs.
//
// Reference: PTX ISA §9.7.15 (Asynchronous Warpgroup MMA).

/**
 * @brief Build a WGMMA 64-bit shared-memory matrix descriptor.
 *
 * Layout (bit ranges, LSB first):
 *   [13: 0]  start address                 : (addr >> 4) & 0x3FFF
 *   [15:14]  reserved                      : 0
 *   [29:16]  leading-dim byte offset (LBO) : (lbo  >> 4) & 0x3FFF
 *   [31:30]  reserved                      : 0
 *   [45:32]  stride-dim byte offset (SBO)  : (sbo  >> 4) & 0x3FFF
 *   [48:46]  reserved                      : 0
 *   [50:49]  base offset (unused here, 0)  : 0
 *   [51:52]  reserved
 *   [52:52]  LBO mode (unused for SWZ_NONE): 0
 *   [62:53]  reserved
 *   [63:62]  swizzle mode                  : 0=none, 1=128B, 2=64B, 3=32B
 *
 * For SWIZZLE_NONE with a canonical K-major layout:
 *   - `addr` is the SHM byte address of the (0,0) element of the tile.
 *   - `lbo` is the byte offset between consecutive "core matrices" along
 *     the leading dimension.  A core matrix is 8 rows × 16 bytes (for
 *     e4m3 k=32 that's 8×16 = 128 B per 8-row strip in K).
 *   - `sbo` is the byte offset between consecutive core matrices along
 *     the stride dimension (same 128 B for k=32, 8-row tiling).
 *
 * The shifts-by-4 reflect the 16-byte alignment requirement on all three.
 */
__device__ static __forceinline__ std::uint64_t make_wgmma_desc(const void* addr,
                                                                std::uint64_t leading_byte_offset,
                                                                std::uint64_t stride_byte_offset,
                                                                std::uint32_t swizzle_mode = 0) {
  std::uint64_t shm_addr;
  asm volatile("cvta.to.shared.u64 %0, %1;\n"
               : "=l"(shm_addr)
               : "l"(reinterpret_cast<std::uint64_t>(addr)));

  std::uint64_t desc = 0;
  desc |= (shm_addr >> 4) & 0x3FFFULL;
  desc |= ((leading_byte_offset >> 4) & 0x3FFFULL) << 16;
  desc |= ((stride_byte_offset >> 4) & 0x3FFFULL) << 32;
  desc |= (static_cast<std::uint64_t>(swizzle_mode) & 0x3ULL) << 62;
  return desc;
}

/**
 * @brief WGMMA fence — order prior register writes before SHM reads by
 *        subsequent WGMMA instructions in the same warpgroup.
 *
 * Required between register-based accumulator zeroing and the first
 * WGMMA in a group.  Not needed between back-to-back WGMMAs that chain
 * accumulators.
 */
__device__ static __forceinline__ void wgmma_fence() { cute::warpgroup_arrive(); }

/**
 * @brief Close the currently-outstanding WGMMA group and start a new one.
 */
__device__ static __forceinline__ void wgmma_commit_group() { cute::warpgroup_commit_batch(); }

/**
 * @brief Wait until at most N WGMMA groups are still outstanding.
 *        `wgmma_wait_group<0>()` waits for all pending groups.
 */
template <std::uint32_t N>
__device__ __forceinline__ void wgmma_wait_group() {
  cute::warpgroup_wait<static_cast<int>(N)>();
}

/**
 * @brief wgmma.mma_async m64n8k32 fp8×fp8 → fp32, SHM A and SHM B.
 *
 * Accumulator layout per thread (lane l, warp w in the WG), same mapping as
 * mma.sync m16n8k32 replicated 4× across the warps:
 *   d[0]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 0
 *   d[1]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 1
 *   d[2]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 0
 *   d[3]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 1
 *
 * Hard-codes scale_D = 1 (accumulate) and scaleA = scaleB = +1.
 */
__device__ static __forceinline__ void wgmma_m64n8k32_e4m3_e4m3_f32(std::uint64_t desc_a,
                                                                    std::uint64_t desc_b, float& d0,
                                                                    float& d1, float& d2,
                                                                    float& d3) {
  constexpr std::uint32_t scale_D = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3}, %4, %5, p, %7, %8;\n"
      "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1));
}

// ── Hopper mbarrier helpers (sm_90a) ──────────────────────────────────────
//
// Wrappers around the `mbarrier.*` family used to gate TMA transfers.  An
// mbarrier is a 64-bit, 16-byte-aligned SHM object with two counters:
//
//   1. Arrival counter   — set by `mbarrier.init`, decremented by arrives.
//   2. Transaction bytes — set by `arrive.expect_tx`, decremented by the
//      TMA engine as bytes land in SHM.
//
// The barrier completes (flips its parity bit) when both reach zero.
// Consumers spin on `try_wait.parity` with a self-cycling parity register,
// so no explicit reset is needed between uses.
//
// Reference: PTX ISA §9.7.12; CUDA Hopper Tuning Guide §1.4.1.2.

/**
 * @brief Convert a generic SHM pointer to a 32-bit shared state-space
 *        address via `cvta.to.shared.u64`.
 *
 * PTX shared-memory operands are 32-bit addresses within the shared
 * state space. We issue `cvta.to.shared.u64` for consistency with
 * `make_wgmma_desc` above and then truncate to 32 bits — the upper
 * bits of a shared pointer are always zero on SM90.
 */
__device__ static __forceinline__ std::uint32_t cvta_to_shared_u32(const void* ptr) {
  std::uint64_t shm_u64;
  asm volatile("cvta.to.shared.u64 %0, %1;\n"
               : "=l"(shm_u64)
               : "l"(reinterpret_cast<std::uint64_t>(ptr)));
  return static_cast<std::uint32_t>(shm_u64);
}

/**
 * @brief Initialize an mbarrier in SHM.
 *
 * Must run exactly once per barrier before any arrive/wait, and must be
 * followed by `fence_mbarrier_init_release_cluster()` before any remote
 * arrival or TMA issue targeting the barrier.
 */
__device__ static __forceinline__ void mbarrier_init(std::uint64_t* bar,
                                                     std::uint32_t arrival_count) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_addr), "r"(arrival_count));
#else
  (void)bar;
  (void)arrival_count;
  asm volatile("trap;");
#endif
}

/**
 * @brief Publish prior `mbarrier.init` writes so that cluster-visible
 *        arrivals (including TMA-engine completions) observe them.
 *
 * Emits: `fence.mbarrier_init.release.cluster;`
 *
 * Required after the last `mbarrier.init` on a block's set of barriers
 * and before the first `mbarrier.arrive*` or `cp.async.bulk.tensor.*`
 * that targets any of them.
 */
__device__ static __forceinline__ void fence_mbarrier_init_release_cluster() {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#else
  asm volatile("trap;");
#endif
}

/**
 * @brief Arrive on an mbarrier and set its transaction-bytes counter.
 *
 * The caller must issue the TMA(s) targeting this barrier AFTER this call;
 * `tx_bytes` must cover the total bytes of every TMA pointing at it.
 */
__device__ static __forceinline__ void mbarrier_arrive_expect_tx(std::uint64_t* bar,
                                                                 std::uint32_t tx_bytes) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  [[maybe_unused]] std::uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
               : "=l"(state)
               : "r"(bar_addr), "r"(tx_bytes)
               : "memory");
#else
  (void)bar;
  (void)tx_bytes;
  asm volatile("trap;");
#endif
}

/**
 * @brief Plain arrive on an mbarrier (no transaction bytes).
 *
 * Consumer-side "slot empty" signal for producer/consumer pipelines: each
 * consuming warp arrives once per phase after its last read of the guarded
 * buffer; the producer waits the phase via mbarrier_try_wait_parity before
 * overwriting.  The arrive carries release semantics for the consumer's
 * prior SHM reads.
 */
__device__ static __forceinline__ void mbarrier_arrive(std::uint64_t* bar) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  [[maybe_unused]] std::uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
               : "=l"(state)
               : "r"(bar_addr)
               : "memory");
#else
  (void)bar;
  asm volatile("trap;");
#endif
}

/**
 * @brief Non-blocking parity-based wait on an mbarrier.
 *
 * Emits:
 *   {
 *     .reg .pred P;
 *     mbarrier.try_wait.parity.shared::cta.b64 P, [bar], parity;
 *     selp.u32 out, 1, 0, P;
 *   }
 *
 * Returns true iff the barrier's current parity matches `parity`,
 * i.e. the barrier has completed for the expected phase. Callers
 * typically loop:
 *
 *   while (!mbarrier_try_wait_parity(bar, parity)) {}
 *   parity ^= 1;  // flip for next use of the same slot
 *
 * The barrier is self-cycling — no reset is needed between uses;
 * each completion toggles the internal parity bit.
 *
 * @param bar     16-B aligned pointer to the barrier in SHM.
 * @param parity  Expected parity bit (0 or 1).
 * @return  `true` if the barrier has reached the expected phase.
 */
__device__ static __forceinline__ bool mbarrier_try_wait_parity(std::uint64_t* bar,
                                                                std::uint32_t parity) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  std::uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
      "selp.u32 %0, 1, 0, P;\n"
      "}\n"
      : "=r"(done)
      : "r"(bar_addr), "r"(parity)
      : "memory");
  return done != 0;
#else
  (void)bar;
  (void)parity;
  asm volatile("trap;");
  return false;
#endif
}

// ── Hopper TMA bulk-tensor copy (sm_90a) ──────────────────────────────────

/**
 * @brief Issue one 2D TMA bulk-tensor load, tracked by an mbarrier.
 *
 * The TMA engine copies one `boxDim` tile of the tensor described by
 * `desc` (a `__grid_constant__ CUtensorMap` kernel parameter — parameter
 * space is coherent with all threads on SM90+) into SHM, decrementing the
 * barrier's transaction-bytes counter as bytes land.  Returns immediately.
 *
 * Caller contract:
 *   - Issued by exactly one thread in the block (the TMA launcher).
 *   - `bar_smem` must be pre-armed with `mbarrier_arrive_expect_tx`
 *     covering the total byte count of every TMA issue targeting it.
 *   - Coordinate order matches `cuTensorMapEncodeTiled`: coord0 = innermost
 *     (fastest) global axis, coord1 = outer axis.
 */
__device__ static __forceinline__ void tma_load_2d(CUtensorMap const& desc, std::uint32_t coord0,
                                                   std::uint32_t coord1, void* dst_smem,
                                                   std::uint64_t* bar_smem) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t dst_addr = cvta_to_shared_u32(dst_smem);
  std::uint32_t bar_addr = cvta_to_shared_u32(bar_smem);
  std::uint64_t desc_addr = reinterpret_cast<std::uint64_t>(&desc);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(dst_addr), "l"(desc_addr), "r"(coord0), "r"(coord1), "r"(bar_addr)
      : "memory");
#else
  (void)desc;
  (void)coord0;
  (void)coord1;
  (void)dst_smem;
  (void)bar_smem;
  asm volatile("trap;");
#endif
}

}  // namespace monomoe

#endif
