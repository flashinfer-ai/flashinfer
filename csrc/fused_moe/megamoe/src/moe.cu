/**
 * This is the main file of the MoE monokernel for Qwen3-Coder (top-K path).
 * It is designed so that you just need to build this file. It includes all
 * relevant implementations. For documentation of the main entry function
 * moe_kernel_topk, see moe_interface.h
 */

#include <cstdint>

#include "moe_interface.h"

#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "moe_debug.h"
#include "moe_down_projection.cu"
#include "moe_grid_barrier.h"
#include "moe_internal.h"
#include "moe_prepare.cu"
#include "moe_routing.cu"
#include "moe_scale_inputs.cu"
#include "moe_tma.h"
#include "moe_up_projection.cu"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

namespace moe_monokernel {

/**
 * @brief Top-K MoE kernel — split-phase WGMMA path for BS <= 8.
 *
 * Uses the v1 dual-warpgroup K=128 streaming WGMMA pipeline for both
 * up- and down-projections.
 *
 * Pipeline:
 *   Phase 1: routing + topK (running in parallel with the routing-window
 *            TMA: a single 16-issue load that fetches the full per-block
 *            BF16 input tile into `bf16_in_full`, completion signalled
 *            on `bar_rwin`)
 *   Phase 2: warp 0 runs `prepare_moe_topk_BS8`; warps 1..11 wait on
 *            `bar_rwin` and quantize the full bf16 tile into
 *            `fp8_act_full` + `act_scale`.
 *   Phase 3: up-proj — streaming WGMMA reads FP8 directly from
 *            `fp8_act_full` (no per-K-step bf16 TMA, no `bar_a`,
 *            no QUANT half) → SiLU → fp8 writeback to spec->temp_fp8.
 *   grid.sync()
 *   Phase 4: down-proj — streaming WGMMA; each block atomicAdds
 *            its fp32 partial sum into the single-buffer
 *            spec->down_partial_out[tok][col]
 *   grid.sync()
 *   Phase 5: cast fp32 → bf16, write activations_out.
 *
 * 2 grid syncs total.
 */
template <typename Dims>
__device__ void moe_kernel_topk_BS8(
    const A_element* __restrict__ activations_in, std::uint32_t batch_size,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up, const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, R_element* __restrict__ activations_out,
    uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    CUtensorMap const& down_weights_desc, CUtensorMap const& down_activations_desc,
    uint32_t* __restrict__ grid_counters, uint32_t& grid_phase,
    uint32_t* __restrict__ expert_counters, uint32_t& expert_phase,
    uint32_t* __restrict__ colstripe_counters, uint32_t& colstripe_phase) {
  static_assert(Dims::BS <= 8);
  static_assert(use_wgmma<Dims>::value, "BS8 path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma<Dims>::value, "BS8 path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  MONO_PHASE_TIMESTAMP(t_start);

  // ── Zero-init the single-buffer `down_partial_out[BS][HIDDEN]` ─────
  //
  // Phase 4 atomicAdds into this buffer (one cell summed across 16
  // blocks).  All 128 blocks zero a chunk in parallel; the up-proj
  // expert_barrier (#2) acts as the cross-block visibility fence
  // before any Phase-4 atomicAdd fires.
  //
  // The buffer was allocated with the legacy [DOWN_GROUPS][BS][HIDDEN]
  // shape; we zero only the first `[BS][HIDDEN]` portion (= 64 KB)
  // since that's all atomicAdd ever touches.
  {
    const uint32_t partial_n = Dims::BS * Dims::HIDDEN_STATES;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < partial_n;
         i += blockDim.x * gridDim.x) {
      spec->down_partial_out[i] = 0.f;
    }
  }

  // ── Phase 1: routing (topK) + routing-window BF16 prefetch ─────────────
  // Phase 1 runs routing (topK / prepare_moe_topk) which writes
  // shmem->experts and shmem->topk_ids_flat that later phases depend on.
  //
  // In parallel with routing, the prefetch warps issue the routing-
  // window TMA: a single 16-issue load that pulls the full per-block
  // BF16 input tile (`BS × HIDDEN_STATES`) into `bf16_in_full` (Req 1.4,
  // 1.5).  Completion is signalled on the single mbarrier `bar_rwin`
  // (`arrival_count = 1`, `tx_bytes = BS * K_BLOCKS_TOTAL *
  // K_STEP_WGMMA * sizeof(A_element)`).  Phase 2 then quantizes the
  // full tile into `fp8_act_full`, which the up-projection K-loop
  // reads directly — no per-K-step bf16 TMA, no `bar_a` (Req 3.1, 3.7).
  //
  // Correctness requirements:
  //   * mbarriers must be initialized before any `arrive_expect_tx`.
  //     The init and the arm run on the same launcher thread, so
  //     program order guarantees local visibility.  The
  //     `fence_mbarrier_init_release_cluster()` below (and the
  //     block-wide `__syncthreads()` at the end of Phase 2) publishes
  //     the init to every consumer warp before it waits on
  //     `bar_rwin` / `bar_w[*]`.
  auto* u_tma = &shmem->u.tiny_wgmma_tma;
  if (is_tma_launcher_thread<Dims>()) {
    // mbarrier inits are kept regardless of the profile flags: they are
    // cheap SHM writes and make the SHM state well-defined even when
    // SKIP_PREFETCH elides every arrive/TMA below.  The K-loop waits
    // inside the up-proj helper are themselves gated on SKIP_PREFETCH,
    // so a consumer will never block on an uninitialized parity.
    mbarrier_init(&u_tma->bar_w[0], 1u);
    mbarrier_init(&u_tma->bar_w[1], 1u);
#ifdef MONO_PROFILE_BARW_4DEEP
    // 2-deep up-proj weight pipeline: slots 2 and 3 hold lookahead
    // tiles `(s+2) & 3` issued at iter `s`.  Initializing them in
    // the prologue (rather than re-initializing in the up-proj
    // helper) reuses the existing fence_mbarrier_init_release_cluster
    // sequence already covering bar_w[0..1].  The down-proj phase
    // doesn't use slots 2 and 3 — they're left untouched at the
    // down-proj re-init and stay armed-or-idle accordingly.
    mbarrier_init(&u_tma->bar_w[2], 1u);
    mbarrier_init(&u_tma->bar_w[3], 1u);
#endif
    // Phase-1 routing-window mbarrier (Req 1.6). Single mbarrier with
    // arrival_count = 1 and tx_bytes = BS * K_BLOCKS_TOTAL *
    // K_STEP_WGMMA * sizeof(A_element) (= 32 KB for Qwen3.5). Armed by
    // the TMA launcher thread at the start of Phase 1; waited on by
    // every warp in [1, 12) at the start of Phase 2 before reading
    // bf16_in_full. Init shares the same launcher-thread / fence
    // discipline as bar_w so consumers never block on an
    // uninitialized parity.
    mbarrier_init(&u_tma->bar_rwin, 1u);
    fence_mbarrier_init_release_cluster();
  }
  // Block-wide barrier publishes the mbarrier inits to every warp
  // before any warp issues a `try_wait.parity` against them.  Without
  // this sync, warp 9 (a non-launcher prefetch warp) can race past
  // the launcher's `mbarrier_init(&bar_rwin, 1u)` and hit the Phase-2
  // wait loop while `bar_rwin` is still in an undefined state, which
  // compute-sanitizer flags as `Unknown Error` at the
  // `SYNCS.PHASECHK.TRANS64.TRYWAIT` instruction (mbarrier state
  // corruption).  `bar_w[*]` was previously protected by the calc-warp
  // path's early `topK_BS8` cost; `bar_rwin` is the only mbarrier
  // waited on by EVERY warp 1..11 with no intervening prior work, so
  // the discipline must be made explicit here.
  //
  // `fence_mbarrier_init_release_cluster()` alone is not sufficient:
  // it pairs with a matching acquire on the consuming side, but
  // `mbarrier.try_wait.parity` is not an acquire of the init; it
  // assumes the init has already been published.  The `__syncthreads()`
  // is what publishes the launcher-thread-only init writes to all
  // warps.
  __syncthreads();
  // ── Phase 1 — Routing-window concurrent dispatch (Req 1.1, 1.2,
  // 1.3, 1.7, 1.8).
  //
  // Re-organized as an if-elif-else over warp identity:
  //   * warp ∈ [8, 12) (prefetch warps):
  //       - TMA launcher thread (warp 8, lane 0) arms `bar_rwin` once
  //         with `tx_bytes = BS * K_BLOCKS_TOTAL * K_STEP_WGMMA *
  //         sizeof(A_element)` (= 32 KB for Qwen3.5) and issues
  //         K_BLOCKS_TOTAL `cp.async.bulk.tensor.2d` instructions
  //         covering the full per-block BF16 input tile via
  //         `moe_load_full_bf16_input` (Option B, design
  //         "TMA-granularity decision").  Both gated under
  //         `MONO_PROFILE_SKIP_PREFETCH_UP` so the matching wait in
  //         the Phase-2 dispatch (added in task 4.2) is paired-elided.
  //       - Other prefetch lanes do nothing in Phase 1.
  //   * warp ∈ [0, 8) (calc warps): unchanged `topK_BS8` +
  //     `sync_calc_threads<>()` (256-thread `bar.sync 15`).
  //     `prepare_moe_topk_BS8` runs in Phase 2 on warp 0 alongside
  //     `routing_phase_quantize` on warps 1..11.
  const unsigned warp_id = get_any_warp<Dims>();
  if (warp_id >= CoreDims::CALC_WARP_COUNT) {
    // Prefetch warps + TMA launcher thread (warp ∈ [8, 12)).
    if (is_tma_launcher_thread<Dims>()) {
#ifndef MONO_PROFILE_SKIP_PREFETCH_UP
      // Single mbarrier arm covers all K_BLOCKS_TOTAL bulk loads.
      // The helper itself does not arm — see the doc comment on
      // `moe_load_full_bf16_input` for the caller contract.
      constexpr std::uint32_t RWIN_TX_BYTES =
          Dims::BS * MoE_SHM<Dims>::U::TinyDataWGMMA_TMA::K_BLOCKS_TOTAL * CoreDims::K_STEP_WGMMA *
          static_cast<std::uint32_t>(sizeof(A_element));
      mbarrier_arrive_expect_tx(&u_tma->bar_rwin,
                                /*tx_bytes=*/RWIN_TX_BYTES);
      moe_load_full_bf16_input<Dims>(activations_desc, u_tma->bf16_in_full, &u_tma->bar_rwin);
#endif
    }
    // Other prefetch lanes (warp 8 lanes 1..31, warps 9..11) do
    // nothing in Phase 1.  Phase 2 (task 4.2) re-engages them as
    // BF16→FP8 quantization workers.
  } else {
    // Calc warps (warp ∈ [0, 8)).  Routing is intentionally NOT
    // guarded by MONO_PROFILE_SKIP_CALC_UP: `shmem->expert_count` /
    // `shmem->experts[e].id` drive the helper's expert loop bounds
    // and an uninitialized expert_count could be anything from 0
    // to 2^32 (runaway loop).  The BS64 path handles
    // MONO_PROFILE_SKIP_CALC_{UP,DOWN} the same way — `topK_BS64`
    // and `prepare_moe_topk_BSx_Ey` run regardless; only the
    // per-expert QUANT / WGMMA / writeback work is compiled out.
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size, shmem);
    MONO_PHASE_TIMESTAMP(t_after_topk);
    sync_calc_threads<Dims>();
    MONO_PHASE_TIMESTAMP(t_after_sync_calc);
    // `prepare_moe_topk_BS8` is no longer called from the calc-warp
    // branch — it now runs in the Phase-2 dispatch below on warp 0
    // only, alongside `routing_phase_quantize` on warps 1..11
    // (Req 2.1, 2.2; design "Phase 2 — Prepare (concurrent across 12
    // warps)").
  }

  // ── Phase 2 — Prepare-phase concurrent dispatch (Req 2.1, 2.2,
  // 2.5, 2.9, 2.10).
  //
  // Warp dispatch over the 12 warps in the block:
  //   * warp 0: runs `prepare_moe_topk_BS8` (builds expert ids,
  //     `sorted_slot`, `expert_count`, `expert_slot_start[]`).  Does
  //     NOT wait on `bar_rwin` because warp 0 does not read
  //     `bf16_in_full`.
  //   * warps 1..11: wait on `bar_rwin` (paired with the Phase-1
  //     16-issue TMA load armed in 4.1) and then run
  //     `routing_phase_quantize`, which calls
  //     `moe_streaming_quantize_k128` once per (token, k_block) pair
  //     across the 11 warps in stride-11 partition (Req 2.4).
  //
  // The wait on `bar_rwin` and the `routing_phase_quantize` body are
  // gated on different `MONO_PROFILE_SKIP_*` flags so they can be
  // toggled independently:
  //   * `MONO_PROFILE_SKIP_PREFETCH_UP` elides BOTH the Phase-1
  //     `bar_rwin` arm + 16 TMA issues AND this Phase-2 wait
  //     (paired-elision, Req 1.8, 8.7) — so warps 1..11 never block
  //     on a routing-window mbarrier that was never armed.
  //   * `MONO_PROFILE_SKIP_CALC_UP` elides the
  //     `routing_phase_quantize` body itself (Req 2.9); warp 0's
  //     `prepare_moe_topk_BS8` keeps running so downstream phases
  //     still see a valid `expert_count` / `experts[]`.
  //
  // The trailing `__syncthreads()` is the SINGLE block-wide sync
  // that ends Phase 2 (Req 2.10): it publishes BOTH warp 0's routing
  // metadata writes AND warps 1..11's `fp8_act_full` / `act_scale`
  // writes to all warps before Phase 3 begins.  No additional
  // intra-Phase-2 sync between warp 0 and warps 1..11 is required —
  // they touch disjoint SHM (warp 0 writes
  // `experts[]`/`sorted_slot[]`/...; warps 1..11 write
  // `fp8_act_full`/`act_scale`).
  if (warp_id == 0) {
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem, spec);
  } else {
#ifndef MONO_PROFILE_SKIP_PREFETCH_UP
    uint32_t parity_rwin = 0;
    while (!mbarrier_try_wait_parity(&u_tma->bar_rwin, parity_rwin)) {
    }
#endif
#ifndef MONO_PROFILE_SKIP_CALC_UP
    routing_phase_quantize<Dims>(u_tma->bf16_in_full, u_tma->fp8_act_full, shmem->act_scale,
                                 batch_size);
#endif
  }
  __syncthreads();

  MONO_PHASE_TIMESTAMP(t_after_routing);

  // ── Phase 2: setup up-projection group mapping ──────────────────────────
  // GRID=128 design, expert-group parallelism (WGMMA path):
  //   UP_GRID = 2*N / W_UP_TILE_EFFECTIVE blocks cover the full 2*N weight
  //   rows for one expert.  With GRID_SIZE=128, we run
  //   UP_GROUPS = GRID_SIZE / UP_GRID groups processing DIFFERENT experts
  //   in parallel, with each group's blocks indexed by
  //   blockIdx.x % UP_GRID.
  //
  //   WGMMA v1 path (W_UP_TILE_EFFECTIVE=128): UP_GRID = 2*N/128,
  //   UP_GROUPS = 128 / UP_GRID.  For N=512: UP_GRID=8, UP_GROUPS=16
  //   (sixteen experts processed in parallel per grid).
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE_EFFECTIVE;
  constexpr std::uint32_t UP_GROUPS = Dims::KernelConfig::GRID_SIZE / UP_GRID;
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  // UP_GROUPS = number of expert groups processed in parallel per grid.
  // Each token contributes at most `top_k` virtual_row slots in
  // spec->temp_bf16, so at most `top_k` blocks write to any given token
  // (one per expert in the token's top-K list).  Blocks processing
  // experts NOT in a token's top-K silently skip the write.  Therefore
  // UP_GROUPS has no upper bound from a correctness standpoint — only
  // a wasted-work concern (higher UP_GROUPS ⇒ more WGMMAs whose
  // experts aren't in any active token's top-K list).
  //
  // We cap at UP_GROUPS <= NUM_EXPERTS (trivially always true) and
  // leave perf tuning to the caller's choice of GRID_SIZE / UP_GRID.
  static_assert(UP_GROUPS <= Dims::NUM_EXPERTS,
                "UP_GROUPS cannot exceed the total number of experts.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  // Phase 3 (`moe_up_projection_BS8_allexperts_wgmma_tma`) reads FP8
  // activations directly from `fp8_act_full`, which Phase 2 produced
  // and the trailing `__syncthreads()` above published.  The up-proj
  // helper does its own weight-tile priming via the pre-loop
  // `bar_w[0]` arm + first-expert weight TMA.  No __syncthreads()
  // here: the Phase-2 trailing sync already published both the
  // barrier init and shmem->expert_count, and computing up_group /
  // up_block_idx / in_up is pure register work.

  // ── Phase 3: Up-projection — expert groups in parallel ────────────────
  // Group `g` (blocks [g*UP_GRID, (g+1)*UP_GRID)) iterates experts starting
  // at index `g`, stepping by UP_GROUPS. Each group writes to DIFFERENT
  // virtual_row slots of spec->temp_bf16 (because each expert has its own
  // k index within a token's top-K list), so the groups never have a
  // write conflict.
  //
  // The BS8 path is TMA+WGMMA only; the kernel asserts
  // `use_wgmma<Dims>::value` and `use_tma<Dims>::value` at the top of
  // this function, so dispatch is unconditional.
  if (in_up && up_group < shmem->expert_count) {
    moe_up_projection_BS8_allexperts_wgmma_tma<Dims>(
        activations_in, expert_weights_up, expert_scales_up, top_k, batch_size, spec, shmem,
        up_weights_desc, activations_desc, up_block_idx,
        /*expert_start=*/up_group,
        /*expert_stride=*/UP_GROUPS);
  }

  MONO_PHASE_TIMESTAMP(t_after_up);

  // ── Site #2 — Expert-local barrier (Phase 2b) ────────────────────────
  //
  // Phase 2a aligned `DOWN_GROUPS == UP_GROUPS` so the producer-set
  // (8 blocks with `up_group == g` writing `spec->temp_fp8` rows for
  // expert group `g`) is identical to the consumer-set (same 8 blocks,
  // now reading those rows in Phase 4 as `down_group == g`).  An
  // `expert_barrier` with `arrival_count = UP_GRID = 8` and `id = up_group`
  // is therefore sufficient: the 8 blocks rendezvous on one of
  // `UP_GROUPS = 16` independent expert-keyed Counter_Pairs, reducing
  // per-barrier atomic contention from 128 → 8 and allowing 16 expert
  // groups to sync concurrently (Design "Site #2 Phase 2b change
  // summary", Requirements 9.6, 9.8).
  //
  // `in_up` is always true in the GRID_SIZE=128, UP_GRID=8, UP_GROUPS=16
  // configuration (every block maps to a valid up_group), but the gate
  // is kept defensively so a future config with UP_GROUPS < GRID_SIZE /
  // UP_GRID won't silently deadlock.
  if (in_up) {
    moe_monokernel::expert_barrier(expert_counters,
                                   /*expert_id=*/up_group,
                                   /*arrival_count=*/UP_GRID,
                                   /*seed_blockidx=*/up_group * UP_GRID, expert_phase);
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier2);

  // ── Phase 4 (WGMMA): dual-WG streaming down-projection ────────────────
  // Each block owns DOWN_COL_TILE output cols; blocks partition into
  // DOWN_GROUPS expert groups × DOWN_GRID col-blocks.  Every
  // contributing block atomicAdds its partial sum into the SAME
  // single-buffer `spec->down_partial_out[BS][HIDDEN_STATES]`; Phase 5
  // reads each cell ONCE and casts to bf16 (no cross-group
  // reduction).  The single-buffer is zero-initialized at kernel entry.
  //
  // The WGMMA down-projection function zeroes its own per-block
  // out_accum in SHM internally, so no pre-zero is needed here.
  //
  // The BS8 path is TMA+WGMMA only; the kernel asserts
  // `use_wgmma<Dims>::value` and `use_tma<Dims>::value` at the top of
  // this function, so dispatch is unconditional.
  moe_down_projection_BS8_allexperts_wgmma_tma<Dims>(expert_weights_down, expert_scales_down, top_k,
                                                     batch_size, spec, shmem, down_weights_desc,
                                                     down_activations_desc);

  MONO_PHASE_TIMESTAMP(t_after_down);

  // ── Site #3 — Col-stripe-local barrier (Phase 2b) ────────────────────
  //
  // Phase 4 atomicAdded into `spec->down_partial_out[tok][col_stripe *
  // DOWN_COL_TILE .. +DOWN_COL_TILE-1]`.  Phase 5 on block `b` reads
  // those cells at its own col stripe `b`, so its producer-set is
  // exactly the `DOWN_GROUPS` blocks with
  // `blockIdx.x % DOWN_GRID == b`.  That sub-grid is also the arrival
  // set of `colstripe_barrier(col_stripe = b, arrival_count =
  // DOWN_GROUPS)`.  Every block (including those with
  // `down_group_r > 0` that don't enter Phase 5) calls the barrier to
  // publish its Phase-4 atomicAdd; the block with `blockIdx.x = b` is
  // the Phase-5 reader and also the seed block (its ID == its col
  // stripe).
  //
  // Per-barrier atomic contention drops from 128 → 16; DOWN_GRID = 8
  // independent col-stripe barriers run concurrently.
  {
    const uint32_t col_stripe_id = blockIdx.x % MoECoreDims<Dims>::DOWN_GRID;
    moe_monokernel::colstripe_barrier(colstripe_counters,
                                      /*col_stripe=*/col_stripe_id,
                                      /*arrival_count=*/MoECoreDims<Dims>::DOWN_GROUPS,
                                      /*seed_blockidx=*/col_stripe_id, colstripe_phase);
  }

  MONO_PHASE_TIMESTAMP(t_after_barrier3);

  // ── Phase 5 (WGMMA): bf16 cast + writeback ─────────────────────────
  // Each Phase-5 block reads its own DOWN_COL_TILE output cols ×
  // Dims::BS tokens of fp32 sums (already accumulated by Phase 4
  // atomicAdds across all DOWN_GROUPS contributing blocks) and casts
  // them to bf16 in `activations_out`.  No cross-group reduction —
  // the work is just a streaming load + cast + store.
  //
  // Block-to-col mapping mirrors Phase 4a: only blocks with
  // `blockIdx.x < DOWN_GRID` are responsible for writing (the first
  // DOWN_GRID blocks cover the full HIDDEN_STATES output).  Blocks
  // beyond DOWN_GRID would map to duplicate cols via
  // `blockIdx.x % DOWN_GRID`, so we gate on the primary group
  // (down_group == 0) to avoid redundant writes.
  //
  // For the BS8 TMA+WGMMA path: DOWN_COL_TILE=256, DOWN_GRID=8,
  // DOWN_GROUPS=16.  For BS64 / non-TMA: DOWN_COL_TILE=128,
  // DOWN_GRID=16, DOWN_GROUPS=8.  Bounds expressed via `CoreDims` so
  // both variants share this code.
  constexpr std::uint32_t DOWN_GRID_LOCAL = CoreDims::DOWN_GRID;
  constexpr std::uint32_t DOWN_COL_TILE_LOCAL = CoreDims::DOWN_COL_TILE;
  const std::uint32_t down_group_r = blockIdx.x / DOWN_GRID_LOCAL;
  const std::uint32_t down_block_idx_r = blockIdx.x % DOWN_GRID_LOCAL;
  const std::uint32_t base_col_r = down_block_idx_r * DOWN_COL_TILE_LOCAL;

  if (down_group_r == 0) {
    // Phase 5 with atomicAdd writeback: read the SINGLE fp32 cell at
    // `partial[tok][col]` (already the sum across all 16 contributing
    // blocks via Phase-4 atomicAdds) and cast to bf16.  No DOWN_GROUPS
    // dimension to reduce over — this is just a streaming
    // load + cast + store.
    for (std::uint32_t flat = threadIdx.x; flat < batch_size * DOWN_COL_TILE_LOCAL;
         flat += blockDim.x) {
      const std::uint32_t tok = flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      const float v = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col];
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)v;
    }

    // Zero out activations_out[tok] for tok in [batch_size, Dims::BS)
    // for this block's DOWN_COL_TILE col stripe.
    for (std::uint32_t flat = threadIdx.x; flat < (Dims::BS - batch_size) * DOWN_COL_TILE_LOCAL;
         flat += blockDim.x) {
      const std::uint32_t tok = batch_size + flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)0.0f;
    }
  }

  MONO_PHASE_TIMESTAMP(t_after_phase5);
}

/**
 * @brief Top-K MoE kernel — single-pass path for BS > 8.
 *
 * Two-phase pipeline:
 *
 *  Phase 1: Calc warps compute all K expert selections into shmem flat arrays,
 *           then sort the virtual batch (num_tokens * top_k) by expert.
 *
 *  Phase 2: Quantize activations once per original token. Separate act_scale
 *           (for up-proj inside silu) from routing_weight (for down-proj).
 *
 *  Then: up-projection over sorted virtual batch → grid.sync →
 *        down-projection accumulating += into original token positions.
 *
 * The output buffer must be zeroed before calling this function.
 */
template <typename Dims>
__device__ void moe_kernel_topk_BS64(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up, const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, R_element* __restrict__ activations_out,
    uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    uint32_t* __restrict__ grid_counters, uint32_t& grid_phase) {
  static_assert(Dims::BS > 8);

  // Step 1: compute all K selections into shmem flat arrays
  if (is_calc_warp<Dims>()) {
    topK_BS64<Dims>(top_k, scoring_func, renormalize, router_logits, token_count, shmem);
  }
  __syncthreads();

  // Step 2: sort BS*top_k virtual rows by expert, build token_indexes_topk
  //         and token_weights
  prepare_moe_topk_BSx_Ey<Dims>(token_count, top_k, shmem);
  __syncthreads();

  // Step 3: quantize activations once per original token.
  // Writes spec->activations[tok] (fp8) and shmem->act_scale[blk][tok].
  //
  // Note: Stage 3b (copy routing_weight → topk_weights_flat) was removed.
  // The down-projection now reads path.bs64.token_weights[sorted_pos]
  // directly — the copy-back was a redundant pass.
  moe_scale_activation_BSx<Dims>(activations_in, token_count, spec, shmem, grid_counters,
                                 grid_phase);

  // Step 4: up-projection (reads token_weights per sorted slot)
  moe_up_projection_topk<Dims>(expert_weights_up, expert_scales_up, spec, shmem);
  moe_monokernel::grid_barrier<Dims::KernelConfig::GRID_SIZE>(grid_counters, grid_phase);

  // Step 5: down-projection (accumulates += into original token positions)
  moe_down_projection_topk<Dims>(expert_weights_down, expert_scales_down, activations_out, spec,
                                 shmem);
}

/**
 * @brief Top-K MoE kernel with configurable scoring and renormalization.
 *
 * Dispatches to moe_kernel_topk_BS8 (BS <= 8) or moe_kernel_topk_BS64 (BS > 8).
 *
 * `up_weights_desc` and `activations_desc` are the host-built TMA
 * descriptors consumed by `moe_up_projection_BS8_allexperts_wgmma_tma` when
 * `use_tma<Dims>::value` is true.  For non-TMA variants the torch-binding
 * wrapper passes zero-initialized `CUtensorMap` values and the descriptors
 * are never read (spec R6.1, R6.3).  The `__grid_constant__` qualifier
 * places them in constant memory coherent with all threads without SMEM
 * cost.
 */
// Requirement 4.4: pin the kernel to 1 block per SM at compile time.
// The software grid / partial barriers rely on the co-residency invariant
// (GRID_SIZE <= SM_count and max_active_blocks_per_SM == 1) so every
// launched block is guaranteed to be running when any other block spins
// on its arrival counter. `__launch_bounds__(BLOCK_SIZE, 1)` is the
// compile-time half of that invariant; the launcher enforces the runtime
// half via `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`.
template <typename Dims>
__global__ __launch_bounds__(Dims::KernelConfig::BLOCK_SIZE, 1) void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up, const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad, size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize, __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc) {
  // ── Compile-time preconditions on `Dims` (spec R7.3, R11.3) ─────────────
  // These fire at the first point where `Dims` is instantiated, so any
  // misconfigured variant is caught at compile time before any TMA /
  // WGMMA code is instantiated below.
  //
  //  * R7.3: `USE_TMA` requires `USE_WGMMA`. There is no TMA support for
  //    the scalar up-projection path.
  //  * R11.3: For every TMA-enabled variant, `MoE_SHM<Dims>` must fit in
  //    the H100 opt-in 228 KB per-block SHM budget. The existing 224 KB
  //    check inside `get_moe_shmem_size<Dims>()` is tighter, but this
  //    assertion documents the per-variant TMA budget and catches future
  //    SHM-layout regressions that loosen the opt-in cap.
  static_assert(!use_tma<Dims>::value || use_wgmma<Dims>::value,
                "USE_TMA requires USE_WGMMA; no TMA support for the scalar "
                "path.");
  static_assert(!use_tma<Dims>::value || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "MoE_SHM<Dims> exceeds the 228 KB per-block SHM budget "
                "for TMA variants.");
  // Phase-2a layout alignment (software-grid-sync spec, Req 9.4):
  // For the BS8 TMA+WGMMA variant, `DOWN_COL_TILE` is bumped to 256,
  // which doubles the down-proj weight tile in SHM from 16 KB to 32 KB
  // per double-buffer slot (the `w_wgmma` / `w_down_wgmma` union grows
  // from 32 KB to 64 KB total).  The re-assertion below makes this
  // explicit at the BS8 TMA+WGMMA instantiation site so that any
  // future layout regression that overflows the 228 KB opt-in budget
  // after the Phase-2a alignment is flagged with a pointed error.
  //
  // Also enforces Req 4.6 of the topk-bs8-tma-prefetch-quant-fusion
  // spec ("`sizeof(MoE_SHM<Dims>)` <= 233472 for every BS8 TMA+WGMMA
  // Dims variant"): `228 * 1024 == 233472`, and the predicate
  // `use_tma<Dims>::value && Dims::BS <= 8` matches every BS8
  // TMA+WGMMA Dims variant.  No second per-Dims-variant assert is
  // required because this one is per-Dims-variant by construction —
  // the kernel is instantiated once per Dims, so the static_assert
  // fires once per BS8 TMA+WGMMA variant and once per non-BS8
  // TMA variant (the latter via the broader assert above).
  static_assert(!(use_tma<Dims>::value && Dims::BS <= 8) || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "Exceeds 228 KB opt-in SHM budget for BS8 TMA+WGMMA after "
                "Phase 2a layout alignment (DOWN_COL_TILE = 256 doubles "
                "the per-block down-proj weight tile).  Also enforces "
                "the topk-bs8-tma-prefetch-quant-fusion Req 4.6 budget.");

  assert(MoECoreDims<Dims>::THREADS_PER_WARP == 32);
  assert(blockDim.x == Dims::KernelConfig::BLOCK_SIZE);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.x == Dims::KernelConfig::GRID_SIZE);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  assert(token_count <= Dims::BS);
  assert(token_count > 0);
  assert(top_k >= 1 && top_k <= MoE_SHM<Dims>::MAX_TOPK);

  MoEGemmSpec<Dims>* spec = reinterpret_cast<MoEGemmSpec<Dims>*>(scratchpad);

  extern __shared__ char shmem_buffer[];
  MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);

  // ── Software barrier pointers and per-region phase state ─────────────
  //
  // Design Component A ("Device-side function signature") + Component B
  // ("Per-sub-grid phase state").  The barrier primitives
  // (`grid_barrier`, `expert_barrier`, `colstripe_barrier`) are pure
  // device functions that take a counter-region pointer + an in/out
  // register-resident phase counter; we materialize the pointers once in
  // the kernel prologue from the scratchpad base and keep the phase
  // counters in the block-local register file.
  //
  //   * `grid_counters` — used by every site that stays at
  //     `Grid_Barrier` (BS64 sites #1, #4, #5 in Phase 1 and BS8 sites
  //     #2, #3 in Phase 1 before Phase-2b downgrades them to
  //     Expert_Barrier / ColStripe_Barrier).
  //   * `expert_counters` — reserved for site #2 in Phase 2b
  //     (task 12.1).  Threaded into `moe_kernel_topk_BS8` so the BS8
  //     Phase-2b migration can wire it into the Expert_Barrier call
  //     without re-plumbing the call chain.
  //   * `colstripe_counters` — reserved for site #3 in Phase 2b
  //     (task 12.2).  Same plumbing treatment as `expert_counters`.
  //   * `grid_phase` / `expert_phase` / `colstripe_phase` — per-region
  //     block-local phase counters; initialized to 0 at kernel entry,
  //     bumped by each barrier call on that region.
  //   * `GRID_SIZE_STATIC` — template non-type arg to
  //     `grid_barrier<>`; the compile-time `Dims::KernelConfig::GRID_SIZE`
  //     value lets the primitive fold its seed value and degenerate-case
  //     gate.
  //
  // Validates: Requirements 2.8, 3.1.
  uint32_t* grid_counters = spec->grid_barrier.slot;
  uint32_t* expert_counters = &spec->partial_barrier.expert_slot[0][0];
  uint32_t* colstripe_counters = &spec->partial_barrier.colstripe_slot[0][0];
  uint32_t grid_phase = 0;
  uint32_t expert_phase = 0;
  uint32_t colstripe_phase = 0;
  constexpr uint32_t GRID_SIZE_STATIC = Dims::KernelConfig::GRID_SIZE;

  // Site #1 — top-of-kernel output zero-out + sync.
  //
  // For BS8 (TMA+WGMMA): ELIMINATED. The Phase 5 reduction in
  // moe_kernel_topk_BS8 `=`-writes every element of activations_out
  // (assigns reduced sum for tokens [0, batch_size) and explicitly
  // zeros [batch_size, Dims::BS) per block col stripe), so the
  // pre-zero + sync is dead work. See Requirements 3.4, 3.5 and
  // Design Migration Plan Site #1.
  //
  // For BS64: PRESERVED. The BS64 down-projection uses `+=` into
  // activations_out and needs the buffer to start zeroed.
  // cooperative_groups::this_grid().sync() → grid_barrier<>.
  if constexpr (Dims::BS > 8) {
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < token_count * Dims::HIDDEN_STATES;
         i += blockDim.x * gridDim.x) {
      activations_out[i] = (__nv_bfloat16)0.0f;
    }
    moe_monokernel::grid_barrier<GRID_SIZE_STATIC>(grid_counters, grid_phase);
  }

  if constexpr (Dims::BS <= 8) {
    moe_kernel_topk_BS8<Dims>(activations_in, token_count, router_logits, expert_weights_up,
                              expert_scales_up, expert_weights_down, expert_scales_down,
                              activations_out, top_k, scoring_func, renormalize, spec, shmem,
                              up_weights_desc, activations_desc, down_weights_desc,
                              down_activations_desc, grid_counters, grid_phase, expert_counters,
                              expert_phase, colstripe_counters, colstripe_phase);
  } else {
    moe_kernel_topk_BS64<Dims>(activations_in, token_count, router_logits, expert_weights_up,
                               expert_scales_up, expert_weights_down, expert_scales_down,
                               activations_out, top_k, scoring_func, renormalize, spec, shmem,
                               grid_counters, grid_phase);
  }
}

}  // namespace moe_monokernel
