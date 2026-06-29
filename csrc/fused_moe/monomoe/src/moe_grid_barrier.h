#pragma once
#ifndef MOE_GRID_BARRIER_H
#define MOE_GRID_BARRIER_H

#ifndef INSIDE_MONOMOE_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

// Software grid / partial barriers for the MonoMoe kernel.
//
// Full protocol (seed + high-bit, ping-pong slots, fence discipline,
// self-maintaining reset) and the per-site arrival-set recipe live in
// docs/design_docs/monomoe_kernel.md §2.  Deadlock-freedom rests on the co-residency invariant
// (docs/design_docs/monomoe_kernel.md §3).  Comments here only flag the per-step intent.

namespace monomoe {

/**
 * @brief Software grid-wide barrier (all `GRID_SIZE_STATIC` blocks),
 *        cooperative-groups-equivalent happens-before with a standard
 *        launch.  Protocol: docs/design_docs/monomoe_kernel.md §2.
 *
 * @tparam GRID_SIZE_STATIC   Compile-time block count
 *                            (`Dims::KernelConfig::GRID_SIZE`); folds the
 *                            seed value and degenerate gate.
 * @param  counters           Two-slot ping-pong Counter_Pair (e.g.
 *                            `&spec->grid_barrier.slot[0]`).  Host
 *                            zero-inits on first use; self-maintaining
 *                            thereafter (docs/design_docs/monomoe_kernel.md §4).
 * @param  phase              In/out phase counter; active slot is
 *                            `phase & 1`.  Init to 0 at kernel entry.
 */
template <uint32_t GRID_SIZE_STATIC>
__device__ __forceinline__ void grid_barrier(uint32_t* __restrict__ counters, uint32_t& phase) {
  // Degenerate single-block grid: nothing to order; still bump phase.
  if constexpr (GRID_SIZE_STATIC == 1) {
    ++phase;
    return;
  }

  // Slot holds 0x80000000u (bit 31 set) once all N blocks contribute;
  // partial counts stay in [SEED, 0x7FFFFFFFu].  N <= 132 ⇒ no wrap.
  constexpr uint32_t SEED = 0x80000000u - (GRID_SIZE_STATIC - 1u);

  const uint32_t slot = phase & 1u;
  uint32_t* c = counters + slot;

  __syncthreads();  // step 1: publish pre-barrier block writes

  if (threadIdx.x == 0) {
    __threadfence();  // step 2: release pre-barrier global writes

    if (blockIdx.x == 0) {
      // step 3: seed.  Mask strips the previous call's leftover high-bit
      // marker, folding back only real early arrivals (docs/design_docs/monomoe_kernel.md §2).
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      atomicAdd(c, 1u);  // non-seed blocks arrive once
    }
  }

  // step 4: thread 0 alone spins; atomicAdd(c, 0u) is an uncached
  // device-scope read (== ld.acquire.gpu on SM90).  Others wait at the
  // step-6 sync.  Happens-before chain: docs/design_docs/monomoe_kernel.md §2.
  if (threadIdx.x == 0) {
    while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
    }
  }

  __threadfence();  // step 5: post-barrier accesses happen-after bit-31
  __syncthreads();  // step 6: re-gather block, release non-spinners
  ++phase;          // step 7: next call targets the other slot
}

/**
 * @brief Software sub-grid (partial) barrier over a caller-specified
 *        arrival set.  Underlies `expert_barrier` (site #2) and
 *        `colstripe_barrier` (site #3).  Protocol, caller contract, and
 *        per-site seed/arrival recipe: docs/design_docs/monomoe_kernel.md §2.
 *
 * Identical to `grid_barrier` except the slot is per-id
 * (`counter_region + id*2 + (phase&1)`) and the seed block /
 * arrival_count are runtime args.  `arrival_count` is runtime (not a
 * template param) so the primitive isn't instantiated per site; every
 * call site passes a compile-time constant so `SEED` and the degenerate
 * gate still fold.
 *
 * @param counter_region          Base of this region's Counter_Pair array
 *                                (`expert_slot`/`colstripe_slot`).  Host
 *                                zero-inits on first use (docs/design_docs/monomoe_kernel.md §4).
 * @param id                      Barrier id (expert group or col stripe).
 * @param arrival_count           Blocks arriving for this `id`
 *                                (`UP_GRID` or `DOWN_GROUPS`).
 * @param seed_thread_blockidx    `blockIdx.x` of the seed block (lowest
 *                                in the arrival set).
 * @param phase                   In/out phase counter for this
 *                                (region, id-at-call-site); active slot
 *                                `phase & 1`.  Init to 0 at kernel entry.
 */
__device__ __forceinline__ void partial_barrier(uint32_t* __restrict__ counter_region, uint32_t id,
                                                uint32_t arrival_count,
                                                uint32_t seed_thread_blockidx, uint32_t& phase) {
  // Degenerate single-block arrival set: nothing to order; still bump
  // phase.  Folds away since arrival_count is constant per call site.
  if (arrival_count == 1u) {
    ++phase;
    return;
  }

  // Slot holds 0x80000000u once all arrival_count blocks contribute.
  const uint32_t SEED = 0x80000000u - (arrival_count - 1u);

  // Per-id Counter_Pair; `id * 2` stride matches the [N_IDS][2] layout
  // in `MoEGemmSpec<Dims>::partial_barrier`.
  const uint32_t slot = phase & 1u;
  uint32_t* c = counter_region + id * 2u + slot;

  __syncthreads();  // step 1: publish pre-barrier block writes

  if (threadIdx.x == 0) {
    __threadfence();  // step 2: release pre-barrier global writes

    if (blockIdx.x == seed_thread_blockidx) {
      // step 3: seed (mask strips leftover high-bit marker; docs/design_docs/monomoe_kernel.md §2)
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      atomicAdd(c, 1u);  // non-seed members arrive once
    }
  }

  // step 4: thread 0 alone spins on an uncached device-scope read.
  if (threadIdx.x == 0) {
    while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
    }
  }

  __threadfence();  // step 5: post-barrier accesses happen-after bit-31
  __syncthreads();  // step 6: re-gather block, release non-spinners
  ++phase;          // step 7: next call targets the other slot
}

/**
 * @brief Phase 3→4 expert-local barrier (site #2).  Thin `__forceinline__`
 *        alias for `partial_barrier` (zero runtime cost).  Recipe:
 *        `id = up_group`, `arrival_count = UP_GRID`,
 *        `seed_blockidx = up_group * UP_GRID` (docs/design_docs/monomoe_kernel.md §2).
 */
__device__ __forceinline__ void expert_barrier(uint32_t* __restrict__ expert_counters,
                                               uint32_t expert_id, uint32_t arrival_count,
                                               uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(expert_counters, expert_id, arrival_count, seed_thread_blockidx, phase);
}

/**
 * @brief Phase 4→5 col-stripe-local barrier (site #3).  Thin
 *        `__forceinline__` alias for `partial_barrier` (zero runtime
 *        cost).  Recipe: `id = blockIdx.x % DOWN_GRID`,
 *        `arrival_count = DOWN_GROUPS`, `seed_blockidx = id` (the
 *        Phase-5 writer is its own seed; docs/design_docs/monomoe_kernel.md §2).
 */
__device__ __forceinline__ void colstripe_barrier(uint32_t* __restrict__ colstripe_counters,
                                                  uint32_t col_stripe, uint32_t arrival_count,
                                                  uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(colstripe_counters, col_stripe, arrival_count, seed_thread_blockidx, phase);
}

}  // namespace monomoe

#endif  // MOE_GRID_BARRIER_H
