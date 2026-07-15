#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>  // CUtensorMap (Driver API) for the tma_load_2d signature

#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90.hpp>

namespace monomoe {

// ── Hopper WGMMA (sm_90a) helpers ─────────────────────────────────────────
//
// These thin wrappers expose the `wgmma.mma_async` family of instructions
// for fp8 (e4m3) operands with fp32 accumulators.
//
// WGMMA semantics (key differences from `mma.sync`):
//
//   * Issued by a full **warpgroup** (4 consecutive warps = 128 threads).
//     Every thread in the warpgroup must execute the instruction.
//
//   * A and B operands live in **shared memory**, addressed by 64-bit
//     "matrix descriptors" that encode base address, leading- and stride-
//     dimension byte offsets, and a swizzle mode.  We use SWIZZLE_NONE for
//     correctness-first bring-up; upgrade to 128B swizzle later for perf.
//
//   * The accumulator D is kept in **registers**, distributed across the
//     warpgroup.  For `m64n8k32` each thread holds 4 fp32 values; for
//     larger N, more.
//
//   * WGMMA is **asynchronous**.  After issuing one or more WGMMAs you
//     must `wgmma.commit_group.sync` to checkpoint, and
//     `wgmma.wait_group.sync N` to wait for all but the last N groups
//     to finish.  Before loading new operands into SHM slots that a
//     pending WGMMA reads, use `wgmma.fence.sync.aligned` to establish
//     ordering between register writes and the SHM reads.
//
// References:
//   - PTX ISA 8.5 §9.7.15 "Asynchronous Warpgroup Level Matrix Multiply-
//     Accumulate Instructions"
//   - NVIDIA Hopper H100 architecture whitepaper

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
 * @brief wgmma.mma_async m64n8k32 fp8×fp8 → fp32 with SHM A and SHM B.
 *
 * Accumulator layout (per thread in the warpgroup, for N=8):
 *   d[0..3] — 4 fp32 values.  Thread (lane l, warp w in {0..3}) in the
 *   warpgroup owns:
 *     d[0]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 0
 *     d[1]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 1
 *     d[2]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 0
 *     d[3]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 1
 *   (Same mapping as mma.sync m16n8k32, replicated 4× across the warps.)
 *
 * PTX form (from CUTLASS mma_sm90_gmma.hpp, MMA_64x8x32_F32E4M3E4M3_SS_TN):
 *   {
 *     .reg .pred p;
 *     setp.ne.b32 p, <scale_D_reg>, 0;
 *     wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3
 *         {d0, d1, d2, d3}, desc_a, desc_b, p, <scaleA>, <scaleB>;
 *   }
 *   where scaleA / scaleB are immediates (+1 or -1).
 *
 * This wrapper hard-codes:
 *   scale_D = 1  (accumulate: D = A·B + D)
 *   scaleA  = 1, scaleB = 1
 * which are the only modes our kernel uses.
 *
 * @param desc_a   Matrix descriptor for operand A (weights, M×K)
 * @param desc_b   Matrix descriptor for operand B (activations, K×N)
 * @param d0..d3   4 fp32 accumulator registers (read-modify-write)
 */
__device__ static __forceinline__ void wgmma_m64n8k32_e4m3_e4m3_f32(std::uint64_t desc_a,
                                                                    std::uint64_t desc_b, float& d0,
                                                                    float& d1, float& d2,
                                                                    float& d3) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  // scale_D = 1 → accumulate into d0..d3.
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
#else
  (void)desc_a;
  (void)desc_b;
  (void)d0;
  (void)d1;
  (void)d2;
  (void)d3;
  asm volatile("trap;");
#endif
}

__device__ inline std::uint32_t rotate_col_32(std::uint32_t col, std::uint32_t row) {
  std::uint32_t col_base = col & 0xff9f;
  std::uint32_t col_rot = (col + 0x20 * row) & 0x60;
  return col_base | col_rot;
}

// ── Hopper mbarrier helpers (sm_90a) ──────────────────────────────────────
//
// Thin PTX wrappers around the `mbarrier.*` family used to gate TMA
// (`cp.async.bulk.tensor.*`) transfers. An mbarrier is a 64-bit shared-
// memory object (16-byte aligned) with two internal counters:
//
//   1. Arrival counter   — decremented by `mbarrier.arrive*`.
//                          Initialized by `mbarrier.init` to `arrival_count`.
//   2. Transaction-bytes — decremented by the TMA engine as bytes land in
//                          SMEM. Initialized to 0, then *set* by
//                          `mbarrier.arrive.expect_tx` (or `expect_tx`).
//
// The barrier "completes" (flips its parity bit) when both counters reach
// zero. Consumers spin on `mbarrier.try_wait.parity` with a self-cycling
// parity register, avoiding explicit reset between uses.
//
// All of these instructions require SM90+; the bodies are compiled out on
// older targets to avoid emitting invalid PTX (`ptxas` would otherwise
// reject them).
//
// References:
//   - PTX ISA 8.5 §9.7.12 "Parallel Synchronization Instructions:
//     mbarrier"
//   - CUDA Hopper Tuning Guide §1.4.1.2 "Asynchronous Memory Copy with
//     mbarrier"

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
 * Sets the arrival counter to `arrival_count` and the transaction-bytes
 * counter to 0. Must be called exactly once per barrier before any
 * arrive / wait, and must be followed by a release fence
 * (`fence.mbarrier_init.release.cluster`) before any remote arrival
 * or TMA issue targeting this barrier.
 *
 * @param bar            16-B aligned pointer to the barrier in SHM.
 * @param arrival_count  Expected number of `mbarrier.arrive*` calls
 *                       (typically 1 for TMA-only completion).
 */
__device__ static __forceinline__ void mbarrier_init(std::uint64_t* bar,
                                                     std::uint32_t arrival_count) {
  cute::initialize_barrier(*bar, static_cast<int>(arrival_count));
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
 * Combined arrive + expect_tx operation:
 *   - Decrements the arrival counter by 1.
 *   - Adds `tx_bytes` to the transaction-bytes counter.
 *
 * Consumers synchronize via `mbarrier_try_wait_parity`.  Callers must issue
 * the TMA (`cp.async.bulk.tensor.*`) targeting this same barrier *after* this
 * call so the hardware-emitted `complete_tx` decrements the counter we just
 * primed.
 *
 * @param bar       16-B aligned pointer to the barrier in SHM.
 * @param tx_bytes  Expected total bytes the TMA engine will deliver.
 */
__device__ static __forceinline__ void mbarrier_arrive_expect_tx(std::uint64_t* bar,
                                                                 std::uint32_t tx_bytes) {
  cute::set_barrier_transaction_bytes(*bar, tx_bytes);
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
//
// Thin PTX wrapper around `cp.async.bulk.tensor.2d.shared::cluster.global.tile.
// mbarrier::complete_tx::bytes`. The TMA engine reads a rectangular tile
// from a tensor in global memory (described by a host-built `CUtensorMap`)
// and writes it into shared memory, decrementing the transaction-bytes
// counter of the supplied mbarrier as bytes land.
//
// Caller contract:
//   - Issued by exactly one thread in the block (the "TMA launcher").
//   - `desc` is a `__grid_constant__ CUtensorMap const` kernel parameter,
//     or otherwise lives in memory coherent with every thread. Passing a
//     generic C++ reference keeps the source readable; the PTX operand
//     is its address, fetched via `&desc`. Parameter-space addresses are
//     coherent with all threads on SM90+, so no `cvta.param.u64` is
//     required — the driver-supplied descriptor address is already usable
//     as a 64-bit global-coherent pointer (this matches the usage pattern
//     in CUTLASS SM90 TMA helpers and in `gpt_oss_router_gemm.cuh`).
//   - The caller must pre-arm `bar_smem` once with
//     `mbarrier_arrive_expect_tx(bar_smem, tx_bytes)` where `tx_bytes`
//     covers every TMA issue pointing at this same barrier, before
//     calling this function. The hardware `complete_tx` emitted by each
//     TMA decrements that counter.
//
// Coordinate convention matches `cuTensorMapEncodeTiled`:
//   - `coord0` is along the innermost (fastest) global axis.
//   - `coord1` is along the outer axis.
// For our use cases both descriptors have innermost = K, so `coord0 =
// k_start`.
//
// References:
//   - PTX ISA §9.7.8.24 "cp.async.bulk.tensor"
//   - CUDA Hopper Tuning Guide §1.4.1.2
//   - CUTLASS `include/cute/arch/copy_sm90_tma.hpp`

/**
 * @brief Issue one 2D TMA bulk-tensor load, tracked by an mbarrier.
 *
 * Emits:
 *   `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes`
 *   `  [dst_smem_addr], [desc, {coord0, coord1}], [bar_smem_addr];`
 *
 * The instruction returns immediately; the TMA engine performs the copy
 * asynchronously and signals completion on `bar_smem` via the
 * transaction-bytes mechanism. Consumers wait via
 * `mbarrier_try_wait_parity`.
 *
 * Byte count per issue is determined by the descriptor's `boxDim` and
 * element type, NOT by this wrapper — it is the caller's responsibility
 * to sum all issues' byte counts into `tx_bytes` when arming the barrier.
 *
 * @param desc      `__grid_constant__` 2D tensor descriptor.
 * @param coord0    Tile coordinate along the innermost global axis.
 * @param coord1    Tile coordinate along the outer global axis.
 * @param dst_smem  16-B aligned SHM destination pointer.
 * @param bar_smem  16-B aligned SHM barrier (pre-armed with `expect_tx`).
 */
__device__ static __forceinline__ void tma_load_2d(CUtensorMap const& desc, std::uint32_t coord0,
                                                   std::uint32_t coord1, void* dst_smem,
                                                   std::uint64_t* bar_smem) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t dst_addr = cvta_to_shared_u32(dst_smem);
  std::uint32_t bar_addr = cvta_to_shared_u32(bar_smem);
  std::uint64_t desc_addr = reinterpret_cast<std::uint64_t>(&desc);
  // `.shared::cluster` dst (not `.shared::cta`, which needs PTX ISA 8.6 /
  // CUDA 12.8+ and fails on older ptxas).  Equivalent for a non-cluster
  // launch; same form CUTLASS emits in cute/arch/copy_sm90_tma.hpp.
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
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
