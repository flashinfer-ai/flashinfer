#pragma once
#ifndef MOE_GRID_BARRIER_H
#define MOE_GRID_BARRIER_H

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

// Software grid-wide barrier for the MoE monokernel.
//
// Replaces `cooperative_groups::this_grid().sync()` so the kernel can
// launch via the standard `cudaLaunchKernel` / `<<<>>>` path and thus be
// captured into a CUDA Graph (see the
// `moe-monokernel-software-grid-sync` spec, Requirement 1 and
// Requirement 5).  The primitive implements Component A ("Software
// `Grid_Barrier` primitive") of the design document.
//
// Safety: the software barrier is a spin on a global-memory flag.
// Deadlock is only possible if the grid runs in multiple temporal waves.
// The monokernel enforces one-block-per-SM co-residency
// (`GRID_SIZE <= SM_count`, `__launch_bounds__(BLOCK_SIZE, 1)`, opt-in
// dynamic SHM > per-SM/2), so every block is scheduled from launch and
// cannot forever-pend while another spins.  See Requirement 4 for the
// full co-residency invariant; the host-side launcher enforces the
// runtime half of those checks.

namespace moe_monokernel {

/**
 * @brief Software grid-wide barrier with cooperative-groups-equivalent
 *        happens-before semantics.
 *
 * Every calling thread in every participating block blocks until all
 * `GRID_SIZE_STATIC` blocks have arrived at this call.  On exit every
 * global-memory write issued by any block before the call is visible
 * to every global-memory read issued by any block after the call
 * (Req 2.1, 2.6).
 *
 * Protocol (design Component A, "Protocol" and "Fence discipline"):
 *
 *   1. `__syncthreads()` publishes pre-barrier per-thread writes
 *      across the block.
 *   2. Thread 0 issues `__threadfence()` so the block's pre-barrier
 *      global writes are visible to other SMs before the arrival.
 *   3. Block 0 seeds the active counter slot with
 *      `SEED = 0x80000000u - (GRID_SIZE_STATIC - 1u)` via
 *      `atomicExch` and folds back any arrivals that landed before
 *      the exchange.  On entry the slot can be in one of four
 *      states:
 *        * `0` — fresh scratchpad, no racers yet.
 *        * `k` with `k ∈ [1, N-1]` — fresh scratchpad, `k` early
 *          arrivals already committed before the seed.
 *        * `0x80000000u` — leftover exit state from the previous
 *          call that used this slot, no racers yet on this call.
 *        * `0x80000000u + k` with `k ∈ [1, N-1]` — previous-call
 *          leftover plus `k` early arrivals for the current call.
 *      The high bit (`0x80000000u`) is the exit-state marker from
 *      the previous call on this slot, NOT an arrival count.  Only
 *      the low 31 bits represent early arrivals that need to be
 *      folded back in, so we mask with `0x7FFFFFFFu` before the
 *      fold-back `atomicAdd`.  This preserves the high-bit
 *      invariant `SEED + (N - 1) = 0x80000000u` for the current
 *      call regardless of the interleaving of atomicExch vs
 *      atomicAdd (Req 2.2, 2.3).
 *   4. Non-seed blocks issue `atomicAdd(c, 1u)` (Req 2.3).
 *   5. Every thread spins on `atomicAdd(c, 0u)` until bit 31 flips
 *      set.  Using `atomicAdd(c, 0u)` (vs a plain load) forces an
 *      uncached device-scope read so the high-bit transition is
 *      observed promptly (Req 2.4).
 *   6. `__threadfence()` pairs with the pre-arrival release so
 *      post-barrier reads happen-after the bit-31 observation, and
 *      `__syncthreads()` gathers the block before the caller uses
 *      the post-barrier invariant (Req 2.6).
 *   7. `++phase` advances the register-resident phase counter so
 *      the next call targets the other ping-pong slot.  The two
 *      slots guarantee that call N+2's seeder cannot observe a
 *      stale high-bit from call N (Req 2.5 — "ping-pong reset").
 *
 * The degenerate `GRID_SIZE_STATIC == 1` branch short-circuits: a
 * single block has no cross-block ordering to enforce and its prior
 * writes are already self-visible (Req 2.7).  We still `++phase` so
 * the call count stays consistent across code paths.
 *
 * @tparam GRID_SIZE_STATIC   Compile-time block count
 *                            (`Dims::KernelConfig::GRID_SIZE`).  Passed
 *                            as a template non-type parameter so the
 *                            seed value and the degenerate-case gate
 *                            fold at compile time.
 * @param  counters           Device pointer to the two-slot ping-pong
 *                            counter pair (e.g.
 *                            `&spec->grid_barrier.slot[0]`, Req 2.8).
 *                            Must be zero-initialized by the host on
 *                            first use; self-maintaining thereafter.
 * @param  phase              In/out register-resident phase counter;
 *                            the active slot is `phase & 1`.  Callers
 *                            initialize to `0` at kernel entry and
 *                            pass by reference so this call can bump
 *                            it.
 */
template <uint32_t GRID_SIZE_STATIC>
__device__ __forceinline__ void grid_barrier(uint32_t* __restrict__ counters, uint32_t& phase) {
  // Degenerate case (Req 2.7): single-block grid has no cross-block
  // ordering to enforce; the block's prior writes are already
  // self-visible.  We still bump `phase` so the call count stays
  // consistent if a later call ever takes a different branch.
  if constexpr (GRID_SIZE_STATIC == 1) {
    ++phase;
    return;
  }

  // Seed value: when all GRID_SIZE_STATIC blocks have added their +1
  // contribution (the seeder's exchange acts as the seed + a fold-in
  // of any prior arrivals; see step 3 below), the slot holds
  // `SEED + (GRID_SIZE_STATIC - 1) = 0x80000000u`, i.e. bit 31 set.
  // For any partial arrival count `j ∈ [0, GRID_SIZE_STATIC - 1)`
  // the slot value is in `[SEED, 0x7FFFFFFFu]` (bit 31 clear), so
  // waiters never exit early.  Bound: `GRID_SIZE_STATIC <= 132` on
  // H200, so `SEED >= 0x80000000u - 131` never wraps.
  constexpr uint32_t SEED = 0x80000000u - (GRID_SIZE_STATIC - 1u);

  // Ping-pong slot selection (Req 2.5, design "Ping-pong reset"):
  // successive calls alternate slots so a block racing one call ahead
  // cannot observe a stale high-bit from the previous call on the
  // same slot.
  const uint32_t slot = phase & 1u;
  uint32_t* c = counters + slot;

  // Step 1 (fence discipline, step 1): publish every thread's
  // pre-barrier writes to block-shared / block-visible state.
  __syncthreads();

  if (threadIdx.x == 0) {
    // Step 2 (fence discipline, step 2): release pre-barrier global
    // writes to the device so other SMs observe them before our
    // arrival on the counter (Req 2.6).
    __threadfence();

    if (blockIdx.x == 0) {
      // Step 3 (seed correctness argument, design Component A):
      // Block 0 seeds the slot with `atomicExch` so the store is
      // atomic with the simultaneous `atomicAdd(+1)` from other
      // blocks.  The slot on entry can be in one of these states:
      //   * 0                        — first use, no racers
      //   * k ∈ [1, N-1]             — first use, k early arrivals
      //   * 0x80000000u              — leftover exit-state marker
      //                                from the previous call on
      //                                this slot, no racers yet
      //   * 0x80000000u + k          — leftover + k early arrivals
      //                                for the current call
      // The high bit is NOT a count — it is the exit-state marker
      // from the previous call that used this slot.  Only the low
      // 31 bits represent early arrivals that need to be folded
      // back.  Masking with `0x7FFFFFFFu` strips the leftover
      // marker and preserves the real count, maintaining the
      // high-bit invariant `SEED + (N - 1) = 0x80000000u`.
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      // Non-seed blocks arrive exactly once (Req 2.3).  The
      // atomicAdd is device-scope; its ordering relative to the
      // seeder's exchange is provided by the atomic RMW itself plus
      // the preceding `__threadfence()`.
      atomicAdd(c, 1u);
    }
  }

  // Step 4 (fence discipline, step 4 — spin with acquire-like
  // semantics): every thread polls the slot until bit 31 flips.
  // Using `atomicAdd(c, 0u)` forces an uncached device-scope read;
  // it compiles to the same ld.acquire.gpu pattern as an inline-PTX
  // acquire load on SM90 and is portable across CUDA versions.
  // (Req 2.4.)
  while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
    // Empty spin body.  The memory subsystem coalesces repeated
    // reads of the same address from the same SM, so the spin's
    // L2 bandwidth cost is negligible.
  }

  // Step 5 (fence discipline, step 5): pair with the arrival-side
  // release so post-barrier global reads/writes happen-after the
  // bit-31 observation and cannot be reordered above the spin.
  __threadfence();

  // Step 6 (fence discipline, step 6): re-gather the block so every
  // thread proceeds together with a consistent post-barrier view.
  __syncthreads();

  // Step 7 (Req 2.5): advance the phase so the next call targets
  // the other ping-pong slot.  The slot we just exited is left at
  // `0x80000000u`; the self-maintaining reset discipline overwrites
  // it via `atomicExch` on the next call that lands on this slot
  // (two calls from now), folding in any stray arrivals.
  ++phase;
}

/**
 * @brief Software sub-grid (partial) barrier with
 *        cooperative-groups-equivalent happens-before semantics over
 *        a caller-specified arrival set.
 *
 * Generic primitive underlying both `Expert_Barrier` (site #2) and
 * `ColStripe_Barrier` (site #3) in the BS8 TMA+WGMMA migration
 * (design Component B — "Software `Partial_Barrier` primitive
 * (Phase 2)").  Every calling thread in every block of the arrival
 * set blocks until all `arrival_count` blocks have arrived at this
 * call for this `id`.  On exit every global-memory write issued by
 * any block in the arrival set before the call is visible to every
 * global-memory read issued by any block in the arrival set after
 * the call (Req 8.1, 8.3, 8.6).
 *
 * The protocol is identical to `grid_barrier` above; only the slot
 * address, seed thread, and seed value differ:
 *
 *   - Slot address: `counter_region + id * 2 + (phase & 1)`.  The
 *     per-id Counter_Pair makes disjoint `id` namespaces
 *     non-interfering (Req 8.2, 8.5).
 *   - Seed_Thread: thread 0 of the block whose `blockIdx.x ==
 *     seed_thread_blockidx` (the lowest `blockIdx.x` in the arrival
 *     set by convention).
 *   - Seed value: `0x80000000u - (arrival_count - 1u)` (runtime,
 *     not constexpr — see note below).
 *   - `++phase` on exit; the two ping-pong slots cover reuse of the
 *     same (region, id) across successive kernel invocations
 *     (Req 8.4).
 *
 * See the `grid_barrier` doxygen above for the step-by-step fence
 * discipline and the seed correctness argument (both carry over
 * unchanged: the 32-bit high-bit protocol works for any
 * `1 <= arrival_count <= 2^31`; bound here is much smaller —
 * `arrival_count <= GRID_SIZE <= 132`).
 *
 * Degenerate case: `arrival_count == 1` short-circuits (a single
 * block has no cross-block ordering to enforce).  The value is a
 * runtime argument so this gate is a runtime branch rather than a
 * compile-time `if constexpr`; at every expected call site the value
 * is a compile-time constant (`UP_GRID`, `DOWN_GROUPS`) so the
 * compiler folds both the gate and the `SEED` computation (Req 9.8).
 *
 * Caller contract:
 *
 *   - Callers that aren't in the arrival set for `id` MUST NOT call
 *     `partial_barrier` for that `id`.  The arrival set is defined
 *     per call site by the producer-set / consumer-set analysis in
 *     the design document; e.g. for `Expert_Barrier` at site #2 the
 *     arrival set for `up_group = g` is
 *     `{blockIdx.x | blockIdx.x / UP_GRID == g}`.
 *   - `phase` is per-(region, id)-call-site register state; callers
 *     maintain one `uint32_t` per region (expert-counters,
 *     colstripe-counters, ...) in the block-local register file
 *     (design "Per-sub-grid phase state").
 *
 * Seed-block selection per call site (design Component B table):
 *
 *   - Site #2 (Expert_Barrier):
 *       `id                    = up_group`
 *       `arrival_count         = UP_GRID`
 *       `seed_thread_blockidx  = up_group * UP_GRID`
 *     (The lowest `blockIdx.x` whose `up_group == g` is `g *
 *     UP_GRID`, and it is in the arrival set by construction.)
 *
 *   - Site #3 (ColStripe_Barrier):
 *       `id                    = blockIdx.x % DOWN_GRID`
 *       `arrival_count         = DOWN_GROUPS`
 *       `seed_thread_blockidx  = blockIdx.x % DOWN_GRID`
 *     (The seed blockidx is the same as the `id` because the
 *     Phase-5 writer for col stripe `c` is `blockIdx.x == c`, and
 *     that block is in the arrival set
 *     `{c + DOWN_GRID * g : g ∈ [0, DOWN_GROUPS)}` at `g == 0`.)
 *
 * Design note — why not template on `arrival_count`?  The primitive
 * is reused for multiple call sites whose arrival counts differ
 * (UP_GRID = 8 for Expert_Barrier, DOWN_GROUPS = 16 for
 * ColStripe_Barrier).  Making `arrival_count` a template non-type
 * parameter would force two separate device-function instantiations
 * for what is otherwise identical code.  The value is a compile-time
 * constant at every call site, so the compiler constant-folds `SEED`
 * and the degenerate-case gate just as effectively without the extra
 * instantiation.
 *
 * Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.6, 9.8.
 *
 * @param counter_region          Device pointer to the base of the
 *                                Counter_Pair array for this region
 *                                (e.g.
 *                                `&spec->partial_barrier.expert_slot[0][0]`
 *                                for site #2, or
 *                                `&spec->partial_barrier.colstripe_slot[0][0]`
 *                                for site #3).  Must be
 *                                zero-initialized by the host on first
 *                                use; self-maintaining thereafter via
 *                                the ping-pong reset discipline
 *                                (Req 8.5).
 * @param id                      Barrier id within the region
 *                                (expert-group index or col-stripe
 *                                index).
 * @param arrival_count           Number of blocks that will arrive at
 *                                this call for this `id`
 *                                (`UP_GRID` or `DOWN_GROUPS`).
 * @param seed_thread_blockidx    `blockIdx.x` of the block whose
 *                                thread 0 seeds the slot.  Must be a
 *                                member of the arrival set; by
 *                                convention the lowest `blockIdx.x`
 *                                in the set.
 * @param phase                   In/out register-resident phase
 *                                counter for this (region, id-at-
 *                                call-site) pair; the active slot is
 *                                `phase & 1`.  Callers initialize to
 *                                `0` at kernel entry.
 */
__device__ __forceinline__ void partial_barrier(uint32_t* __restrict__ counter_region, uint32_t id,
                                                uint32_t arrival_count,
                                                uint32_t seed_thread_blockidx, uint32_t& phase) {
  // Degenerate case (Req 9.8): a single-block arrival set has no
  // cross-block ordering to enforce; the block's prior writes are
  // already self-visible.  We still bump `phase` so the call count
  // stays consistent across kernel invocations (the slot still
  // ping-pongs).  At all current call sites `arrival_count` is a
  // compile-time constant (UP_GRID = 8 or DOWN_GROUPS = 16) so the
  // compiler folds this branch away.
  if (arrival_count == 1u) {
    ++phase;
    return;
  }

  // Seed value: when all `arrival_count` blocks have added their +1
  // contribution (the seeder's exchange acts as the seed + a fold-in
  // of any prior arrivals; see step 3 below), the slot holds
  // `SEED + (arrival_count - 1) = 0x80000000u`, i.e. bit 31 set.
  // For any partial arrival count `j ∈ [0, arrival_count - 1)` the
  // slot value is in `[SEED, 0x7FFFFFFFu]` (bit 31 clear), so
  // waiters never exit early.  Bound: `arrival_count <= GRID_SIZE
  // <= 132` on H200, so `SEED >= 0x80000000u - 131` never wraps.
  // Runtime (not constexpr) by design — see the "why not template"
  // note in the doxygen above.
  const uint32_t SEED = 0x80000000u - (arrival_count - 1u);

  // Slot address: per-id Counter_Pair with ping-pong slot selection
  // (Req 8.4, design Component B "Protocol").  The `id * 2` stride
  // matches the `uint32_t counter[N_IDS][2]` layout declared in
  // `MoEGemmSpec<Dims>::partial_barrier`.
  const uint32_t slot = phase & 1u;
  uint32_t* c = counter_region + id * 2u + slot;

  // Step 1 (fence discipline, step 1): publish every thread's
  // pre-barrier writes to block-shared / block-visible state.
  __syncthreads();

  if (threadIdx.x == 0) {
    // Step 2 (fence discipline, step 2): release pre-barrier global
    // writes to the device so other arrival-set members observe
    // them before our arrival on the counter (Req 8.6).
    __threadfence();

    if (blockIdx.x == seed_thread_blockidx) {
      // Step 3 (seed correctness argument, design Component B —
      // identical to `grid_barrier`): the seed block writes
      // `SEED` with `atomicExch` so the store is atomic with the
      // simultaneous `atomicAdd(+1)` from other blocks in the
      // arrival set.  The slot on entry can be in one of these
      // states:
      //   * 0                         — first use, no racers
      //   * k ∈ [1, arrival_count-1]  — first use, k early arrivals
      //   * 0x80000000u               — leftover exit-state marker
      //                                 from the previous call on
      //                                 this slot, no racers yet
      //   * 0x80000000u + k           — leftover + k early arrivals
      //                                 for the current call
      // The high bit is NOT a count — it is the exit-state marker
      // from the previous call on this (region, id) slot.  Only
      // the low 31 bits represent early arrivals that need to be
      // folded back.  Masking with `0x7FFFFFFFu` strips the
      // leftover marker and preserves the real count, maintaining
      // the high-bit invariant
      // `SEED + (arrival_count - 1) = 0x80000000u`.
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      // Non-seed arrival-set members arrive exactly once (Req 8.3).
      // The atomicAdd is device-scope; its ordering relative to the
      // seeder's exchange is provided by the atomic RMW itself plus
      // the preceding `__threadfence()`.
      atomicAdd(c, 1u);
    }
  }

  // Step 4 (fence discipline, step 4 — spin with acquire-like
  // semantics): every thread polls the slot until bit 31 flips.
  // Using `atomicAdd(c, 0u)` forces an uncached device-scope read;
  // it compiles to the same ld.acquire.gpu pattern as an inline-PTX
  // acquire load on SM90 and is portable across CUDA versions.
  while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
    // Empty spin body.  Repeated reads of the same address from
    // the same SM are coalesced in the memory subsystem so the
    // spin's L2 bandwidth cost is negligible.
  }

  // Step 5 (fence discipline, step 5): pair with the arrival-side
  // release so post-barrier global reads/writes happen-after the
  // bit-31 observation and cannot be reordered above the spin.
  __threadfence();

  // Step 6 (fence discipline, step 6): re-gather the block so every
  // thread proceeds together with a consistent post-barrier view.
  __syncthreads();

  // Step 7 (Req 8.4): advance the phase so the next call targets
  // the other ping-pong slot.  The slot we just exited is left at
  // `0x80000000u`; the self-maintaining reset discipline overwrites
  // it via `atomicExch` on the next call that lands on this slot
  // (two calls from now — typically the next kernel invocation for
  // Partial_Barrier, since each block participates at most once per
  // (region, id) per kernel call).
  ++phase;
}

/**
 * @brief Thin call-site alias for `partial_barrier` used at site #2
 *        (BS8 Phase 3 → Phase 4, expert-local barrier).
 *
 * Exists purely for call-site clarity — it forwards every argument
 * unchanged to `partial_barrier` and adds no logic (design
 * Component B, "Thin aliases for call-site clarity").  Keeping the
 * alias `__device__ __forceinline__` ensures the compiler inlines
 * through the wrapper so there is zero runtime cost vs. calling
 * `partial_barrier` directly.
 *
 * Call-site parameter recipe (design Component B, seed-block
 * selection table):
 *
 *   - `id                    = up_group`
 *   - `arrival_count         = UP_GRID`
 *   - `seed_thread_blockidx  = up_group * UP_GRID`
 *
 * (The lowest `blockIdx.x` whose `up_group == g` is `g * UP_GRID`,
 * and it is in the arrival set by construction.)
 *
 * See the `partial_barrier` doxygen above for the full protocol,
 * fence discipline, and caller contract.
 *
 * Validates: Requirements 9.6, 9.8.
 *
 * @param expert_counters         Device pointer to the base of the
 *                                expert Counter_Pair array
 *                                (`&spec->partial_barrier.expert_slot[0][0]`).
 * @param expert_id               Expert-group index (`up_group`) in
 *                                `[0, NUM_EXPERTS)`.
 * @param arrival_count           Blocks per expert group
 *                                (`UP_GRID`).
 * @param seed_thread_blockidx    `blockIdx.x` of the seed block
 *                                (`up_group * UP_GRID`).
 * @param phase                   In/out register-resident phase
 *                                counter for the expert-barrier
 *                                region.
 */
__device__ __forceinline__ void expert_barrier(uint32_t* __restrict__ expert_counters,
                                               uint32_t expert_id, uint32_t arrival_count,
                                               uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(expert_counters, expert_id, arrival_count, seed_thread_blockidx, phase);
}

/**
 * @brief Thin call-site alias for `partial_barrier` used at site #3
 *        (BS8 Phase 4 → Phase 5, col-stripe-local barrier).
 *
 * Exists purely for call-site clarity — it forwards every argument
 * unchanged to `partial_barrier` and adds no logic (design
 * Component B, "Thin aliases for call-site clarity").  Keeping the
 * alias `__device__ __forceinline__` ensures the compiler inlines
 * through the wrapper so there is zero runtime cost vs. calling
 * `partial_barrier` directly.
 *
 * Call-site parameter recipe (design Component B, seed-block
 * selection table):
 *
 *   - `id                    = blockIdx.x % DOWN_GRID`
 *   - `arrival_count         = DOWN_GROUPS`
 *   - `seed_thread_blockidx  = blockIdx.x % DOWN_GRID`
 *
 * (The seed blockidx is the same as the `id` because the Phase-5
 * writer for col stripe `c` is `blockIdx.x == c`, and that block is
 * in the arrival set
 * `{c + DOWN_GRID * g : g ∈ [0, DOWN_GROUPS)}` at `g == 0`.)
 *
 * See the `partial_barrier` doxygen above for the full protocol,
 * fence discipline, and caller contract.
 *
 * Validates: Requirements 9.7, 9.8.
 *
 * @param colstripe_counters      Device pointer to the base of the
 *                                col-stripe Counter_Pair array
 *                                (`&spec->partial_barrier.colstripe_slot[0][0]`).
 * @param col_stripe              Col-stripe index
 *                                (`blockIdx.x % DOWN_GRID`) in
 *                                `[0, DOWN_GRID)`.
 * @param arrival_count           Blocks per col stripe
 *                                (`DOWN_GROUPS`).
 * @param seed_thread_blockidx    `blockIdx.x` of the seed block
 *                                (`blockIdx.x % DOWN_GRID` — i.e.
 *                                the Phase-5 writer for this col
 *                                stripe).
 * @param phase                   In/out register-resident phase
 *                                counter for the col-stripe-barrier
 *                                region.
 */
__device__ __forceinline__ void colstripe_barrier(uint32_t* __restrict__ colstripe_counters,
                                                  uint32_t col_stripe, uint32_t arrival_count,
                                                  uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(colstripe_counters, col_stripe, arrival_count, seed_thread_blockidx, phase);
}

}  // namespace moe_monokernel

#endif  // MOE_GRID_BARRIER_H
