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
#include "moe_internal.h"
#include "moe_routing.cuh"
#include "moe_scale_inputs.cuh"
#include "moe_tma.h"
#include "moe_up_projection.cuh"
#undef INSIDE_MONOMOE_IMPLEMENTATION

namespace monomoe {

/**
 * @brief Top-K MoE kernel — split-phase TMA+WGMMA path for BS <= 8.
 *
 * Design doc: docs/design_docs/monomoe_kernel.md.  The five-phase pipeline
 * (routing+prefetch / quantize / up-proj / down-proj / writeback). Phase
 * markers below tie each block of code to that table.
 */
template <typename Dims>
__device__ void moe_kernel_topk_impl(
    const A_element* __restrict__ activations_in, std::uint32_t batch_size,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up, const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, R_element* __restrict__ activations_out,
    uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
    const float* __restrict__ expert_bias, float routed_scaling_factor,
    MoEGemmSpec<Dims>* __restrict__ spec, MoE_SHM<Dims>* __restrict__ shmem,
    CUtensorMap const& up_weights_desc, CUtensorMap const& activations_desc,
    CUtensorMap const& down_weights_desc, CUtensorMap const& down_activations_desc) {
  static_assert(Dims::BS <= 8,
                "moe_kernel_topk_impl supports BS<=8 (token_count > 8 is "
                "rejected by the host dispatcher).");
  static_assert(use_wgmma<Dims>::value,
                "TMA+WGMMA path requires the WGMMA configuration (use_wgmma).");
  static_assert(use_tma<Dims>::value, "TMA+WGMMA path requires USE_TMA");
  using CoreDims = MoECoreDims<Dims>;

  // Zero the Phase-4 accumulator (all blocks cooperate).  No cross-block
  // sync guards this against Phase-4 atomicAdds from fast blocks; the
  // margin is structural — every block runs the full routing + quantize +
  // up-projection pipeline (tens of µs) before its first Phase-4
  // atomicAdd, while this zero-fill retires within the first ~µs of the
  // launch.
  {
    const uint32_t partial_n = Dims::BS * Dims::HIDDEN_STATES;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < partial_n;
         i += blockDim.x * gridDim.x) {
      spec->down_partial_out[i] = 0.f;
    }
  }

  // ── mbarrier init + sentinel-handoff launch parity ───────────────────
  // Inits run unconditionally (cheap SHM writes) so the barrier state is
  // well-defined before any warp reaches its first try_wait.
  auto* u_tma = &shmem->tiny_wgmma_tma;
  if (is_tma_launcher_thread<Dims>()) {
#pragma unroll
    for (uint32_t i = 0; i < MoECoreDims<Dims>::UP_W_SLOTS; ++i) {
      mbarrier_init(&u_tma->bar_w[i], 1u);
    }
    mbarrier_init(&u_tma->bar_rwin, 1u);
    fence_mbarrier_init_release_cluster();
  }
  if (threadIdx.x == 0) {
    // Per-block private launch counter → scale-buffer parity for the
    // sentinel Phase-3→4 handoff.  No cross-block sync needed: each block
    // increments only its own word, once per launch, so all blocks agree.
    const uint32_t flip = spec->launch_flip[blockIdx.x];
    shmem->scale_parity = flip & 1u;
    spec->launch_flip[blockIdx.x] = flip + 1u;
  }
  // Publish the launcher-thread inits to every warp before any
  // try_wait.parity.  fence_mbarrier_init alone is not sufficient: it pairs
  // with a matching acquire, but try_wait.parity is not an acquire of the
  // init — without this sync a prefetch warp can hit the Phase-2 bar_rwin
  // wait while the barrier is still uninitialized (mbarrier state
  // corruption under compute-sanitizer).  Also publishes scale_parity.
  __syncthreads();

  // Zero-refill the OTHER parity's handoff state for the NEXT launch
  // (sentinel reset, off the critical path — nobody reads those words
  // this launch; kernel completion publishes them before the next launch
  // starts).  The host-side torch.zeros allocation establishes the same
  // invariant before the first launch.
  {
    constexpr uint32_t SCALE_N =
        MoEGemmSpec<Dims>::TEMP_ROWS * MoEGemmSpec<Dims>::TEMP_ACT_SCALE_COLS;
    const uint32_t next_parity = shmem->scale_parity ^ 1u;
    float* next_buf = moe_act_scale_buf<Dims>(spec, next_parity);
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < SCALE_N;
         i += blockDim.x * gridDim.x) {
      reinterpret_cast<uint32_t*>(next_buf)[i] = 0u;  // +0.0f sentinel
    }
    // Phase-4→5 readiness counters for the next launch (site #3 flags).
    if (threadIdx.x < MoEGemmSpec<Dims>::DOWN_GRID && blockIdx.x == 0) {
      spec->down_ready[next_parity][threadIdx.x] = 0u;
    }
  }

  // ── Phase 1: routing ∥ routing-window BF16 prefetch ───────────────────
  const unsigned warp_id = get_any_warp<Dims>();
  if (warp_id >= CoreDims::CALC_WARP_COUNT) {
    // Prefetch warps: the launcher arms bar_rwin once for the whole tile
    // and issues the K_BLOCKS_TOTAL bulk loads; other lanes idle.
    if (is_tma_launcher_thread<Dims>()) {
      constexpr std::uint32_t RWIN_TX_BYTES =
          Dims::BS * MoE_SHM<Dims>::TinyDataWGMMA_TMA::K_BLOCKS_TOTAL * CoreDims::K_STEP_WGMMA *
          static_cast<std::uint32_t>(sizeof(A_element));
      mbarrier_arrive_expect_tx(&u_tma->bar_rwin,
                                /*tx_bytes=*/RWIN_TX_BYTES);
      moe_load_full_bf16_input<Dims>(activations_desc, u_tma->bf16_in_full, &u_tma->bar_rwin);
    }
  } else {
    // Calc warps run routing: shmem->expert_count / experts[] drive
    // downstream loop bounds and must always be valid.
    topK<Dims>(top_k, scoring_func, renormalize, router_logits, batch_size, shmem, expert_bias,
               routed_scaling_factor);
    sync_calc_threads<Dims>();
  }

  // ── Phase 2: prepare (warp 0) ∥ quantize (warps 1..11) ────────────────
  // Warp 0 builds the routing tables and does not read bf16_in_full, so it
  // skips the bar_rwin wait.  Warps 1..11 wait for the Phase-1 load, then
  // quantize bf16 → fp8_act_full + act_scale.  The two sides touch
  // disjoint SHM; the single trailing __syncthreads() publishes both to
  // all warps before Phase 3.
  if (warp_id == 0) {
    prepare_moe_topk<Dims>(batch_size, top_k, shmem);
  } else {
    uint32_t parity_rwin = 0;
    while (!mbarrier_try_wait_parity(&u_tma->bar_rwin, parity_rwin)) {
    }
    routing_phase_quantize<Dims>(u_tma->bf16_in_full, u_tma->fp8_act_full, shmem->act_scale,
                                 batch_size);
  }
  __syncthreads();

  // ── Phase 3: up-projection — expert groups in parallel ────────────────
  // Group g (blocks [g*UP_GRID, (g+1)*UP_GRID)) iterates experts starting
  // at index g, stepping by UP_GROUPS.  Groups write disjoint temp_fp8
  // rows (each routed (tok, expert) pair has its own sorted_slot row).
  constexpr std::uint32_t UP_GRID = 2 * Dims::N / CoreDims::W_UP_TILE_EFFECTIVE;
  constexpr std::uint32_t UP_GROUPS = Dims::KernelConfig::GRID_SIZE / UP_GRID;
  static_assert(Dims::KernelConfig::GRID_SIZE % UP_GRID == 0,
                "GRID_SIZE must be a multiple of UP_GRID.");
  static_assert(UP_GROUPS <= Dims::NUM_EXPERTS,
                "UP_GROUPS cannot exceed the total number of experts.");
  const std::uint32_t up_group = blockIdx.x / UP_GRID;
  const std::uint32_t up_block_idx = blockIdx.x % UP_GRID;
  const bool in_up = (up_group < UP_GROUPS);

  if (in_up && up_group < shmem->expert_count) {
    // Single-atom interleaved up-projection (UP_COL_HALVES == 1): one
    // 128-row A-tile packs 64 gate + 64 up rows in the gate/up pair layout.
    static_assert(CoreDims::UP_COL_HALVES == 1u,
                  "this shape uses the interleaved (UCH==1) up-projection.");
    moe_up_projection_allexperts_wgmma_tma<Dims>(activations_in, expert_weights_up,
                                                 expert_scales_up, top_k, batch_size, spec, shmem,
                                                 up_weights_desc, activations_desc, up_block_idx,
                                                 /*expert_start=*/up_group,
                                                 /*expert_stride=*/UP_GROUPS);
  }

  // ── Site #2: Phase 3 → Phase 4 handoff — SENTINEL, NO BARRIER ─────────
  //
  // Replaced by data-path readiness: each `temp_act_scale` cell is
  // release-published by its producing warp AFTER the covering fp8
  // payload segment (moe_publish_act_scale), and the down-projection
  // polls exactly the cells it consumes until they turn non-NaN before
  // reading/TMA-loading the payload (see the act-scale loader and the
  // inter-expert lookahead poll in moe_down_projection.cu).
  //
  // This gives per-expert granularity — a down-block starts an expert as
  // soon as THAT expert's rows are published, instead of waiting for its
  // whole up-group — and covers the coupled and decoupled carves
  // uniformly (consumers wait on data, so producer set != consumer set
  // needs no special protocol).  Barrier-counter state is no longer
  // involved in this handoff, which also removes the decoupled
  // consume-wait stale-marker hazard.

  // ── Phase 4: down-projection ──────────────────────────────────────────
  moe_down_projection_allexperts_wgmma_tma<Dims>(expert_weights_down, expert_scales_down, top_k,
                                                 batch_size, spec, shmem, down_weights_desc,
                                                 down_activations_desc);

  // ── Site #3: Phase 4 → Phase 5 handoff — FLAGS, NO BARRIER ────────────
  // Phase 5 on block b reads down_partial_out cells at col stripe b, whose
  // producers are the DOWN_GROUPS blocks with blockIdx.x % DOWN_GRID == b.
  //
  // down_partial_out is atomicAdd-accumulated, so readiness cannot live
  // in the data itself; instead every block publishes its Phase-4 adds
  // by bumping its stripe's parity-selected arrival counter (release
  // fence + add), and ONLY the stripe's Phase-5 writer polls it up to
  // DOWN_GROUPS.  The other blocks publish and run to kernel exit — no
  // grid-wide mutual spin.  Counter reset is the launch-parity
  // double-buffer refilled in the prologue (same discipline as the
  // site-#2 scale sentinel).
  {
    const uint32_t col_stripe_id = blockIdx.x % MoECoreDims<Dims>::DOWN_GRID;
    // All threads' Phase-4 atomicAdds are issued before this sync; the
    // fence releases them at device scope before the arrival is visible.
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(&spec->down_ready[shmem->scale_parity][col_stripe_id], 1u);
    }
  }

  // ── Phase 5: fp32 → bf16 cast + writeback ─────────────────────────────
  // Each cell of down_partial_out already holds the full sum (Phase-4
  // atomicAdds); the first DOWN_GRID blocks stream-cast their own col
  // stripe.  Every element of the M-row output is `=`-written, so no
  // output pre-zero pass exists anywhere.
  constexpr std::uint32_t DOWN_GRID_LOCAL = CoreDims::DOWN_GRID;
  constexpr std::uint32_t DOWN_COL_TILE_LOCAL = CoreDims::DOWN_COL_TILE;
  const std::uint32_t down_group_r = blockIdx.x / DOWN_GRID_LOCAL;
  const std::uint32_t down_block_idx_r = blockIdx.x % DOWN_GRID_LOCAL;
  const std::uint32_t base_col_r = down_block_idx_r * DOWN_COL_TILE_LOCAL;

  if (down_group_r == 0) {
    // Readiness wait: this block is the single Phase-5 writer for col
    // stripe `down_block_idx_r` (== blockIdx.x here).  One thread polls
    // the stripe's arrival counter, acquires, and the __syncthreads
    // broadcasts the observation to the whole block.
    if (threadIdx.x == 0) {
      uint32_t* ctr = &spec->down_ready[shmem->scale_parity][down_block_idx_r];
      while (atomicAdd(ctr, 0u) < MoECoreDims<Dims>::DOWN_GROUPS) {
      }
      __threadfence();
    }
    __syncthreads();

    for (std::uint32_t flat = threadIdx.x; flat < batch_size * DOWN_COL_TILE_LOCAL;
         flat += blockDim.x) {
      const std::uint32_t tok = flat / DOWN_COL_TILE_LOCAL;
      const std::uint32_t col_in_block = flat % DOWN_COL_TILE_LOCAL;
      const std::uint32_t col = base_col_r + col_in_block;
      const float v = spec->down_partial_out[tok * Dims::HIDDEN_STATES + col];
      activations_out[tok * Dims::HIDDEN_STATES + col] = (R_element)v;
    }

    // No padding-row writes: activations_out is an M-row tensor and rows
    // [batch_size, Dims::BS) do not exist (writing them was an
    // out-of-bounds store for M < 8; no consumer ever read them).
  }
}

/**
 * @brief Kernel entry point.  Dispatches to moe_kernel_topk_impl.
 *
 * The TMA descriptors are built host-side (moe_wrapper.cu) and passed as
 * `__grid_constant__` parameters.
 *
 * `__launch_bounds__(BLOCK_SIZE, 1)` is the compile-time half of the
 * one-block-per-SM co-residency invariant the software barriers rely on
 * (every participating block must be scheduled from launch); the host
 * launcher checks the runtime half (GRID_SIZE <= SM count).
 */
template <typename Dims>
__global__ __launch_bounds__(Dims::KernelConfig::BLOCK_SIZE, 1) void moe_kernel_topk(
    const A_element* __restrict__ activations_in, std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up, const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down, R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad, size_t scratchpad_size, size_t shmem_size, std::uint32_t top_k,
    ScoringFunc scoring_func, bool renormalize, const float* __restrict__ expert_bias,
    float routed_scaling_factor, __grid_constant__ CUtensorMap const up_weights_desc,
    __grid_constant__ CUtensorMap const activations_desc,
    __grid_constant__ CUtensorMap const down_weights_desc,
    __grid_constant__ CUtensorMap const down_activations_desc) {
  static_assert(!use_tma<Dims>::value || use_wgmma<Dims>::value,
                "USE_TMA requires USE_WGMMA; no TMA support for the scalar "
                "path.");
  static_assert(!use_tma<Dims>::value || sizeof(MoE_SHM<Dims>) <= 228 * 1024,
                "MoE_SHM<Dims> exceeds the 228 KB per-block SHM budget "
                "for TMA variants.");

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

  // Cross-block ordering is entirely flag/sentinel-based (sites #2 and
  // #3 inside moe_kernel_topk_impl); no software barrier counters remain.

  moe_kernel_topk_impl<Dims>(activations_in, token_count, router_logits, expert_weights_up,
                             expert_scales_up, expert_weights_down, expert_scales_down,
                             activations_out, top_k, scoring_func, renormalize, expert_bias,
                             routed_scaling_factor, spec, shmem, up_weights_desc, activations_desc,
                             down_weights_desc, down_activations_desc);
}

}  // namespace monomoe
