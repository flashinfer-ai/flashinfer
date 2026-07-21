/**
 * This is the main file of the MonoMoe kernel for Qwen3-Coder (top-K path).
 * It is designed so that you just need to build this file. It includes all
 * relevant implementations. For documentation of the main entry function
 * moe_kernel_topk, see moe_interface.h
 */

#include <cstdint>

#include "moe_interface.h"

#define INSIDE_MONOMOE_IMPLEMENTATION
#include "moe_down_projection.cuh"
#include "moe_grid_barrier.h"
#include "moe_internal.h"
#include "moe_routing.cuh"
#include "moe_scale_inputs.cuh"
#include "moe_tma.h"
#include "moe_up_projection.cuh"
#undef INSIDE_MONOMOE_IMPLEMENTATION

namespace monomoe {

/**
 * @brief Top-K MoE kernel — split-phase WGMMA path for BS <= 8.
 *
 * Design doc: docs/design_docs/monomoe_kernel.md.  The five-phase pipeline
 * (routing+prefetch / quantize / up-proj / down-proj / writeback) and the two
 * grid syncs between them are described in §1; the barriers in §2.  Phase
 * markers below tie each block of code to that table.
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
  static_assert(use_wgmma_v<Dims>, "BS8 path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma_v<Dims>, "BS8 path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  // Zero-init `down_partial_out[BS][HIDDEN]` that Phase 4 atomicAdds into.
  // All blocks zero a chunk in parallel; the Site-#2 expert_barrier is the
  // cross-block visibility fence before any Phase-4 atomicAdd fires.
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
  // BF16 input tile (`BS × HIDDEN_STATES`) into `bf16_in_full`.
  // Completion is signalled on the single mbarrier `bar_rwin`
  // (`arrival_count = 1`, `tx_bytes = BS * K_BLOCKS_TOTAL *
  // K_STEP_WGMMA * sizeof(A_element)`).  Phase 2 then quantizes the
  // full tile into `fp8_act_full`, which the up-projection K-loop
  // reads directly — no per-K-step bf16 TMA, no `bar_a`.
  //
  // Correctness requirements:
  //   * mbarriers must be initialized before any `arrive_expect_tx`.
  //     The init and the arm run on the same launcher thread, so
  //     program order guarantees local visibility.  The
  //     `fence_mbarrier_init_release_cluster()` below (and the
  //     block-wide `__syncthreads()` at the end of Phase 2) publishes
  //     the init to every consumer warp before it waits on
  //     `bar_rwin` / `bar_w[*]`.
  auto* u_tma = &shmem->tiny_wgmma_tma;
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
    // Phase-1 routing-window mbarrier. Single mbarrier with
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
  // ── Phase 1 — Routing-window concurrent dispatch.
  //
  // Re-organized as an if-elif-else over warp identity:
  //   * warp ∈ [8, 12) (prefetch warps):
  //       - TMA launcher thread (warp 8, lane 0) arms `bar_rwin` once
  //         with `tx_bytes = BS * K_BLOCKS_TOTAL * K_STEP_WGMMA *
  //         sizeof(A_element)` (= 32 KB for Qwen3.5) and issues
  //         K_BLOCKS_TOTAL `cp.async.bulk.tensor.2d` instructions
  //         covering the full per-block BF16 input tile via
  //         `moe_load_full_bf16_input`.  The matching wait
  //         lives in the Phase-2 dispatch.
  //       - Other prefetch lanes do nothing in Phase 1.
  //   * warp ∈ [0, 8) (calc warps): unchanged `topK_BS8` +
  //     `sync_calc_threads<>()` (256-thread `bar.sync 15`).
  //     `prepare_moe_topk_BS8` runs in Phase 2 on warp 0 alongside
  //     `routing_phase_quantize` on warps 1..11.
  const unsigned warp_id = get_any_warp<Dims>();
  if (warp_id >= CoreDims::CALC_WARP_COUNT) {
    // Prefetch warps + TMA launcher thread (warp ∈ [8, 12)).
    if (is_tma_launcher_thread<Dims>()) {
      // Single mbarrier arm covers all K_BLOCKS_TOTAL bulk loads.
      // The helper itself does not arm — see the doc comment on
      // `moe_load_full_bf16_input` for the caller contract.
      constexpr std::uint32_t RWIN_TX_BYTES =
          Dims::BS * MoE_SHM<Dims>::TinyDataWGMMA_TMA::K_BLOCKS_TOTAL * CoreDims::K_STEP_WGMMA *
          static_cast<std::uint32_t>(sizeof(A_element));
      mbarrier_arrive_expect_tx(&u_tma->bar_rwin,
                                /*tx_bytes=*/RWIN_TX_BYTES);
      moe_load_full_bf16_input<Dims>(activations_desc, u_tma->bf16_in_full, &u_tma->bar_rwin);
    }
    // Other prefetch lanes (warp 8 lanes 1..31, warps 9..11) do
    // nothing in Phase 1.  Phase 2 re-engages them as
    // BF16→FP8 quantization workers.
  } else {
    // Calc warps (warp ∈ [0, 8)).  `topK_BS8` and `prepare_moe_topk_BS8`
    // build `shmem->expert_count` / `shmem->experts[e].id`, which drive
    // the downstream per-expert loop bounds.
    topK_BS8<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size, shmem);
    sync_calc_threads<Dims>();
    // `prepare_moe_topk_BS8` is no longer called from the calc-warp
    // branch — it now runs in the Phase-2 dispatch below on warp 0
    // only, alongside `routing_phase_quantize` on warps 1..11.
  }

  // ── Phase 2 — Prepare-phase concurrent dispatch.
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
  //     across the 11 warps in stride-11 partition.
  //
  // The trailing `__syncthreads()` is the SINGLE block-wide sync
  // that ends Phase 2: it publishes BOTH warp 0's routing
  // metadata writes AND warps 1..11's `fp8_act_full` / `act_scale`
  // writes to all warps before Phase 3 begins.  No additional
  // intra-Phase-2 sync between warp 0 and warps 1..11 is required —
  // they touch disjoint SHM (warp 0 writes
  // `experts[]`/`sorted_slot[]`/...; warps 1..11 write
  // `fp8_act_full`/`act_scale`).
  if (warp_id == 0) {
    prepare_moe_topk_BS8<Dims>(batch_size, top_k, shmem, spec);
  } else {
    uint32_t parity_rwin = 0;
    while (!mbarrier_try_wait_parity(&u_tma->bar_rwin, parity_rwin)) {
    }
    routing_phase_quantize<Dims>(u_tma->bf16_in_full, u_tma->fp8_act_full, shmem->act_scale,
                                 batch_size);
  }
  __syncthreads();

  // ── Up-projection group mapping (docs/design_docs/monomoe_kernel.md §6) ────────────────────────
  // UP_GRID blocks cover one expert's 2*N rows; UP_GROUPS expert groups run
  // in parallel.  For this shape: UP_GRID=8, UP_GROUPS=16.
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE_EFFECTIVE;
  constexpr std::uint32_t UP_GROUPS = Dims::KernelConfig::GRID_SIZE / UP_GRID;
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  // UP_GROUPS has no correctness upper bound (blocks whose expert isn't in a
  // token's top-K just skip the write) — higher only means wasted WGMMAs.
  static_assert(UP_GROUPS <= Dims::NUM_EXPERTS,
                "UP_GROUPS cannot exceed the total number of experts.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  // No __syncthreads() before Phase 3: the Phase-2 trailing sync already
  // published `fp8_act_full` / barrier init / `expert_count`, and the
  // up_group/up_block_idx math is pure register work.  The up-proj helper
  // primes its own weight tile (pre-loop `bar_w[0]` arm + first TMA).

  // ── Phase 3: Up-projection — expert groups in parallel (docs/design_docs/monomoe_kernel.md §1)
  // Group `g` iterates experts from index `g` stepping by UP_GROUPS, writing
  // disjoint `temp_fp8` slabs (no write conflict).  TMA+WGMMA dispatch is
  // unconditional (asserted at function top).
  if (in_up && up_group < shmem->expert_count) {
    moe_up_projection_BS8_allexperts_wgmma_tma<Dims>(
        activations_in, expert_weights_up, expert_scales_up, top_k, batch_size, spec, shmem,
        up_weights_desc, activations_desc, up_block_idx,
        /*expert_start=*/up_group,
        /*expert_stride=*/UP_GROUPS);
  }

  // ── Site #2 — Expert-local barrier, Phase 3→4 (docs/design_docs/monomoe_kernel.md §2) ─────────
  // DOWN_GROUPS == UP_GROUPS aligns the producer set (8 blocks that wrote
  // expert group `g`'s temp_fp8 rows) with the consumer set, so an
  // expert_barrier keyed on `up_group` is sufficient (vs a full grid sync).
  // `in_up` is always true here but gated defensively for future configs.
  if (in_up) {
    monomoe::expert_barrier(expert_counters,
                            /*expert_id=*/up_group,
                            /*arrival_count=*/UP_GRID,
                            /*seed_blockidx=*/up_group * UP_GRID, expert_phase);
  }

  // ── Phase 4 (WGMMA): dual-WG streaming down-projection ──
  // (docs/design_docs/monomoe_kernel.md §1,§6)
  // Blocks partition into DOWN_GROUPS expert groups × DOWN_GRID col-blocks,
  // each owning DOWN_COL_TILE output cols.  Every contributing block
  // atomicAdds its partial into the single-buffer `down_partial_out` (zeroed
  // at kernel entry); Phase 5 reads each cell once.  Dispatch unconditional.
  moe_down_projection_BS8_allexperts_wgmma_tma<Dims>(expert_weights_down, expert_scales_down, top_k,
                                                     batch_size, spec, shmem, down_weights_desc,
                                                     down_activations_desc);

  // ── Site #3 — Col-stripe-local barrier, Phase 4→5 (docs/design_docs/monomoe_kernel.md §2) ─────
  // Phase 5 reader for col stripe `b` is `blockIdx.x == b`; its producer set
  // is the DOWN_GROUPS blocks with `blockIdx.x % DOWN_GRID == b`, which is
  // exactly the colstripe_barrier arrival set (and `b` is its own seed).
  // Every block calls it to publish its Phase-4 atomicAdd.
  {
    const uint32_t col_stripe_id = blockIdx.x % MoECoreDims<Dims>::DOWN_GRID;
    monomoe::colstripe_barrier(colstripe_counters,
                               /*col_stripe=*/col_stripe_id,
                               /*arrival_count=*/MoECoreDims<Dims>::DOWN_GROUPS,
                               /*seed_blockidx=*/col_stripe_id, colstripe_phase);
  }

  // ── Phase 5 (WGMMA): bf16 cast + writeback (docs/design_docs/monomoe_kernel.md §1) ─────────────
  // Each Phase-5 block reads its own DOWN_COL_TILE cols × BS tokens of fp32
  // sums (already reduced by the Phase-4 atomicAdds) and casts to bf16 — a
  // streaming load+cast+store, no cross-group reduction.  Only the first
  // DOWN_GRID blocks write (gate on `down_group == 0`); the rest would map to
  // duplicate cols via `blockIdx.x % DOWN_GRID`.
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
}

/**
 * @brief Top-K MoE kernel with configurable scoring and renormalization.
 *
 * Dispatches to moe_kernel_topk_BS8 (BS <= 8).
 *
 * `up_weights_desc` and `activations_desc` are the host-built TMA
 * descriptors consumed by `moe_up_projection_BS8_allexperts_wgmma_tma` when
 * `use_tma_v<Dims>` is true.  For non-TMA variants the torch-binding
 * wrapper passes zero-initialized `CUtensorMap` values and the descriptors
 * are never read.  The `__grid_constant__` qualifier
 * places them in constant memory coherent with all threads without SMEM
 * cost.
 */
// `__launch_bounds__(BLOCK_SIZE, 1)` pins one block per SM — the compile-time
// half of the co-residency invariant the barrier spin depends on
// (docs/design_docs/monomoe_kernel.md §3); the launcher enforces
// `GRID_SIZE <= SM_count` at runtime.
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
  // Compile-time preconditions on `Dims` (fire at first instantiation):
  // TMA requires WGMMA (no scalar TMA path), and the SHM footprint must fit
  // the H100 opt-in 228 KB per-block budget.  The third assert is the same
  // budget specialized to BS8 TMA+WGMMA, where DOWN_COL_TILE=256 doubles the
  // per-slot down-proj weight tile — kept pointed so a layout regression
  // names that cause.
  static_assert(!use_tma_v<Dims> || use_wgmma_v<Dims>,
                "USE_TMA requires USE_WGMMA; no TMA support for the scalar "
                "path.");
  static_assert(!use_tma_v<Dims> || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "MoE_SHM<Dims> exceeds the 228 KB per-block SHM budget "
                "for TMA variants.");
  static_assert(!(use_tma_v<Dims> && Dims::BS <= 8) || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "Exceeds 228 KB opt-in SHM budget for BS8 TMA+WGMMA after "
                "Phase 2a layout alignment (DOWN_COL_TILE = 256 doubles "
                "the per-block down-proj weight tile).");

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

  // Barrier counter-region pointers (into the scratchpad tail; see
  // docs/design_docs/monomoe_kernel.md §4) and per-region phase counters
  // (block-local registers, init 0, bumped by each barrier call; §2).
  uint32_t* grid_counters = spec->grid_barrier.slot;
  uint32_t* expert_counters = &spec->partial_barrier.expert_slot[0][0];
  uint32_t* colstripe_counters = &spec->partial_barrier.colstripe_slot[0][0];
  uint32_t grid_phase = 0;
  uint32_t expert_phase = 0;
  uint32_t colstripe_phase = 0;
  constexpr uint32_t GRID_SIZE_STATIC = Dims::KernelConfig::GRID_SIZE;

  // No top-of-kernel output zero-out: Phase 5 `=`-writes every element of
  // activations_out (reduced sum for [0, batch_size), zeros for the rest), so
  // a pre-zero would be dead work.

  static_assert(Dims::BS <= 8, "Only BS <= 8 is supported.");
  moe_kernel_topk_BS8<Dims>(activations_in, token_count, router_logits, expert_weights_up,
                            expert_scales_up, expert_weights_down, expert_scales_down,
                            activations_out, top_k, scoring_func, renormalize, spec, shmem,
                            up_weights_desc, activations_desc, down_weights_desc,
                            down_activations_desc, grid_counters, grid_phase, expert_counters,
                            expert_phase, colstripe_counters, colstripe_phase);
}

}  // namespace monomoe
