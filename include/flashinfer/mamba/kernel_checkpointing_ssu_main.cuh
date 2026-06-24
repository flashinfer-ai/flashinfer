/*
 * Copyright (c) 2025 by FlashInfer team.
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
// =============================================================================
// Two-kernel split: MAIN kernel (.plans/ssu_split.md, S4).  Validated bit-exact
// vs the monolithic via test_two_kernel_matches_monolithic (T0).
//
// The CONSUMER of the precompute's scratch.  Grid (D_SPLIT, batch, nheads) —
// per (batch, head, d-tile), like the monolithic (per-head state recurrence).
// Reuses the monolithic's ssu_checkpoint / ssu_nocheckpoint (replay + output)
// with READ_PRECOMPUTED_CB=true: the matmul-4 CB A-operand is LDG'd from gmem
// (cb_scaled / cb_old, fragA-native) instead of computed+LDSM'd, and β comes
// from the loaded raw cumAdt (cumAdt_vec → smem.cumAdt, exp'd on the fly by the
// existing epilogue).  No CB compute, no C1/C2 here.
//
// Load (load_main_data): a SUBSET of the monolithic's load_data composed from
// the shared building blocks — load_state_*, load_tile_async (C, x, old_x,
// old_B, z), load_old_dt_cumAdt — MINUS new-B and dt (+ the C1/C2 compute),
// PLUS the cumAdt_vec → smem.cumAdt load.  Kept as composable pieces so a
// future persistent variant can loop (batch, head) reusing them.
//
// Cache ownership (see precompute header): the main writes old_x (it loads x)
// and state (the replay's C8); the precompute owns old_B / old_dt / old_cumAdt.
//
// PDL overlap: the CACHED loads (state, old_*, z) + the state replay run BEFORE
// gdc_wait, so under the PDL chain they overlap the precompute (the replay is the
// heaviest work — folding old tokens into state + the C8 HBM write).  The conv1d
// OUTPUTS (C, new-token x) and the precompute outputs (cb_scaled / cumAdt_vec)
// load + are consumed POST-gdc_wait: they are NOT available earlier — when the
// main co-launches, conv1d may still be writing x/C (the precompute is itself
// blocked on its conv1d gdc_wait), so a pre-wait load would read stale data.
// gdc_wait is a no-op without a programmatic predecessor (T0 / non-PDL).
// =============================================================================
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_

// Brings ssu_checkpoint / ssu_nocheckpoint, CheckpointingSsuStorage,
// store_old_x, ENABLE_PDL — and (transitively) the shared load helpers in
// kernel_checkpointing_ssu_common.cuh (load_tile_async, load_state_*,
// load_old_dt_cumAdt, load_cb_fragA).
#include "kernel_checkpointing_ssu.cuh"

namespace flashinfer::mamba::checkpointing {

// -----------------------------------------------------------------------------
// Lean smem storage for the MAIN kernel.  Same layout as CheckpointingSsuStorage EXCEPT it
// omits the three buffers the main never touches — it reads cb_scaled / cb_old from gmem
// (fragA-native, the READ_PRECOMPUTED_CB path in compute_*_output), never loads new-B (CB is
// precomputed), and uses the precomputed cumAdt instead of computing dt_proc:
//   • CB_scaled (~2 KB)   • B (~2 KB)   • dt_proc (tiny)
// Dropping ~4 KB takes the main from 7 → 8 resident blocks/SM (smem was the co-limiter with
// regs), adding a wave of warps to hide its DRAM-latency stalls (long_scoreboard ~39%,
// eligible-warps 0.71).  The static constexpr layout constants mirror CheckpointingSsuStorage
// exactly so the shared load/compute helpers (templated on SmemT) bind unchanged.
// STATE_PIPE = PIPELINE_STAGES.  >1 double-buffers ALL per-head buffers (z, old_x, old_dt,
// old_cumAdt, cumAdt, x, state) so head h+1's loads overlap head h's replay/output.
// Per-GROUP buffers (C, old_B) stay single-slot — they are shared across all heads in the CTA.
// STATE_PIPE=1 ⇒ single slot == the original layout, bit-identical.
template <typename input_t, typename state_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE, int STATE_PIPE = 1>
struct CheckpointingSsuMainStorage {
  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;

  // C (matmul-3 A-operand) — per-group conv1d output.  Single slot (per-GROUP, not per-head).
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];
  // old_B — replay B-operand; per-group cache.  Single slot (per-GROUP).
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];
  // All per-HEAD buffers — STATE_PIPE slots each.  Slot b at offset b × <per-slot size>.
  // STATE_PIPE=1 → single slot, bit-identical to the original single-buffer layout.
  static constexpr int Z_ELEMS = NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  static constexpr int OLD_X_ELEMS = MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
  static constexpr int X_ELEMS = NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  static constexpr int STATE_ELEMS = D_PER_CTA * DSTATE;
  alignas(16) input_t z[STATE_PIPE * Z_ELEMS];
  alignas(16) input_t old_x[STATE_PIPE * OLD_X_ELEMS];
  float old_dt[STATE_PIPE * MAX_WINDOW];
  float old_cumAdt[STATE_PIPE * MAX_WINDOW];
  float cumAdt[STATE_PIPE * NPREDICTED];
  alignas(16) input_t x[STATE_PIPE * X_ELEMS];
  alignas(16) state_t state[STATE_PIPE * STATE_ELEMS];
};

// -----------------------------------------------------------------------------
// Main-kernel load — the conv1d-INDEPENDENT (pre-gdc_wait) set: state, old_x, z
// always; old_B + old_dt + old_cumAdt only when must_checkpoint (they feed the
// replay).  On the no-write path the precompute baked old_B/old_dt/old_cumAdt
// into cb_old, so only the tail old_cumAdt[prev_k-1] is loaded (for the β).  All
// are cached from the previous step (or z, which bypasses conv1d), so they are
// safe to load before gdc_wait and overlap the precompute.
//
// The conv1d OUTPUTS — C and the new-token x — are deliberately NOT loaded here.
// They are produced by conv1d, which the main co-launches ahead of (the
// precompute is still blocked on its own conv1d gdc_wait when the main starts),
// so a pre-wait load would race conv1d's writes.  They load POST-gdc_wait via
// load_conv_inputs (Triton does the same: history pre-wait, new x / C post-wait,
// replay_selective_state_update.py:1845 vs 1857).  cumAdt (precompute output)
// likewise loads post-wait, in the kernel.  Drops the monolithic's new-B
// (cb_scaled is precomputed) and dt + cumAdt scan.  Single async group, drained
// by one commit + wait + syncwarp (like load_data).
// load_head: the per-head PRE-gdc_wait cached loads (old_x / z / old_B / old_dt).  ISSUE ONLY —
// no __pipeline_commit / wait / syncwarp; head_loop owns the cp.async pipeline so these loads and
// the state prefetch can be committed + drained together at the right depth.  State is NOT loaded
// here — its sole loader is prefetch_state (the one tile the cross-head pipeline double-buffers).
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, bool IS_FIRST, typename SmemT>
__device__ __forceinline__ void load_head(SmemT& smem, CheckpointingSsuParams const& params,
                                          int lane, int warp, int d_tile, int head, int group_idx,
                                          int64_t cache_slot, int buf_read, int64_t outer,
                                          int seq_len, bool must_checkpoint, int prev_k,
                                          int tile_buf) {
  using namespace cute;
  int const d_tile_off = d_tile * D_PER_CTA;

  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ old_x_ptr = reinterpret_cast<input_t const*>(params.old_x);
  auto const* __restrict__ old_B_ptr = reinterpret_cast<input_t const*>(params.old_B);

  int64_t const ox_base = cache_slot * params.old_x_stride_seq + (int64_t)head * DIM + d_tile_off;
  int64_t const oB_base = cache_slot * params.old_B_stride_seq +
                          (int64_t)buf_read * params.old_B_stride_dbuf +
                          (int64_t)group_idx * DSTATE;

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  using ZShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<D_PER_CTA>>;
  using OldBShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  using OxShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

  // Per-head smem slots: offset by tile_buf into the STATE_PIPE-strided arrays.
  auto* old_x_slot = smem.old_x + tile_buf * SmemT::OLD_X_ELEMS;
  float* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  float* old_cumAdt_slot = smem.old_cumAdt + tile_buf * MAX_WINDOW;
  auto* z_slot = smem.z + tile_buf * SmemT::Z_ELEMS;

  if (warp == 0)
    load_tile_async<OxShape, MAX_WINDOW>(old_x_slot, old_x_ptr + ox_base, params.old_x_stride_token,
                                         lane);
  if (must_checkpoint) {
    if constexpr (IS_FIRST)
      if (warp == 1)
        load_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, old_B_ptr + oB_base,
                                               params.old_B_stride_token, lane);
    if (warp == 2)
      load_old_dt_cumAdt(params, lane, cache_slot, buf_read, head, MAX_WINDOW, old_dt_slot,
                         old_cumAdt_slot);
  } else if (prev_k > 0 && warp == 0 && lane == 0) {
    auto const* __restrict__ oca_ptr = reinterpret_cast<float const*>(params.old_cumAdt);
    int64_t const ca_base = cache_slot * params.old_cumAdt_stride_seq +
                            (int64_t)buf_read * params.old_cumAdt_stride_dbuf +
                            (int64_t)head * params.old_cumAdt_stride_head;
    old_cumAdt_slot[prev_k - 1] = oca_ptr[ca_base + prev_k - 1];  // tail for β only
  }
  if (warp == 3 && z_ptr) {
    int64_t const z_base = outer * params.z_stride_seq + (int64_t)head * DIM + d_tile_off;
    load_tile_async<ZShape, NPREDICTED>(z_slot, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }
  // NO commit/wait/syncwarp — head_loop drains (see header).
}

template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void prefetch_state(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int state_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
  int64_t const state_base = cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE +
                             (int64_t)d_tile_off * DSTATE;
  if constexpr (DIM == D_PER_CTA) {
    load_state_per_warp<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp,
                                                               lane, state_buf);
  } else {
    int const tid = warp * warpSize + lane;
    load_state_cta<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, tid,
                                                          state_buf);
  }
  // NOTE: issue ONLY — no __pipeline_commit / wait here.  head_loop owns the whole cp.async
  // pipeline (commit + wait_prior) so the prefetched states can stay in flight across the
  // gdc_wait / replay; committing here would let an internal drain complete them too early.
}

// load_x: loads x (and C for IS_FIRST) from gmem → smem slot tile_buf.  ISSUE ONLY —
// no __pipeline_commit / wait / syncwarp.  head_loop owns the whole cp.async pipeline.
// C is per-GROUP (W1); x is per-HEAD (W0).  Both loads run in parallel across warps.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, bool IS_FIRST,
          typename SmemT>
__device__ __forceinline__ void load_x(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int warp, int d_tile, int head, int group_idx, int64_t outer,
                                       int seq_len, int tile_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const x_base = outer * params.x_stride_seq + (int64_t)head * DIM + d_tile_off;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // C is per-GROUP → load once per head-tile (IS_FIRST only), on W1.
  if constexpr (IS_FIRST)
    if (warp == 1)
      load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane,
                                          seq_len);
  // x is per-HEAD → every head, on W0 (parallel with C on W1).
  auto* x_slot = smem.x + tile_buf * SmemT::X_ELEMS;
  if (warp == 0)
    load_tile_async<XShape, NPREDICTED>(x_slot, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
  // NOTE: NO commit/wait — head_loop owns the pipeline.
}

// replay_head: state replay (MUST_CHECKPOINT) + cross-warp sync + one-shot gdc_wait (IS_FIRST).
// Pre-condition: state and old_tiles for this head are in smem slot tile_buf (committed+waited).
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, bool MUST_CHECKPOINT,
          bool IS_FIRST, typename SmemT>
__device__ __forceinline__ void replay_head(SmemT& smem, CheckpointingSsuParams const& params,
                                            int lane, int warp, int d_tile, int head,
                                            int64_t cache_slot, int prev_k, int64_t rand_seed,
                                            int tile_buf) {
  if constexpr (MUST_CHECKPOINT) {
    __syncthreads();  // state (per-warp split) visible cross-warp before the replay
    int64_t const state_ptr_offset =
        cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;
    state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                  (int64_t)d_tile * D_PER_CTA * DSTATE;
    replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
        smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
        /*must_checkpoint=*/true, tile_buf);
  }
  __syncthreads();  // publish replayed state cross-warp; also serves as the gdc_wait fence
  // ONE-SHOT PDL gdc_wait (IS_FIRST only).  Once head 0 has waited, precompute is visible
  // to the whole CTA; subsequent heads read cb_scaled / cumAdt_vec without re-waiting.
  if constexpr (IS_FIRST) cudaGridDependencySynchronize();
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS,
          bool MUST_CHECKPOINT, typename SmemT>
__device__ __forceinline__ void output_head(SmemT& smem, CheckpointingSsuParams const& params,
                                            int lane, int warp, int d_tile, int head, int seq,
                                            int64_t cache_slot, int prev_k, int64_t outer,
                                            int seq_len, int64_t out_seq_base, int write_offset,
                                            int64_t rand_seed, int tile_buf, float D_val) {
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;
  constexpr int CB_OLD_REGS = SmemT::MAX_WINDOW_PAD_MMA_K / 2;
  auto const* __restrict__ cb_gmem_head =
      reinterpret_cast<input_t const*>(params.cb_scaled) +
      (int64_t)(seq * params.nheads + head) * warpSize * CB_NEW_REGS;

  if constexpr (MUST_CHECKPOINT) {
    compute_and_store_output<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                             PHILOX_ROUNDS, /*READ_PRECOMPUTED_CB=*/true>(
        smem, params, warp, lane, d_tile, out_seq_base, head, cache_slot, D_val,
        /*must_checkpoint=*/true, seq_len, cb_gmem_head, tile_buf);
  } else {
    auto const* __restrict__ cb_old_head =
        reinterpret_cast<input_t const*>(params.cb_old) +
        (int64_t)(seq * params.nheads + head) * warpSize * CB_OLD_REGS;
    compute_no_write_output<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                            NUM_WARPS, /*READ_PRECOMPUTED_CB=*/true>(
        smem, params, warp, lane, prev_k, d_tile, out_seq_base, head, cache_slot, D_val, seq_len,
        cb_gmem_head, cb_old_head, tile_buf);
  }
  // store_old_x: 128-thread cooperative copy (16×8 layout), only first 4 warps participate.
  if (warp < 4)
    store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                     cache_slot, write_offset, seq_len, tile_buf);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS,
          bool MUST_CHECKPOINT, int MAIN_HEADS_PER_CTA, int PIPELINE_STAGES, typename SmemT>
__device__ __forceinline__ void head_loop(SmemT& smem, CheckpointingSsuParams const& params,
                                          int lane, int warp, int d_tile, int first_head,
                                          int group_idx, int seq, int64_t cache_slot, int buf_read,
                                          int prev_k, int64_t outer, int seq_len,
                                          int64_t out_seq_base, int write_offset) {
  int64_t const rand_seed = (MUST_CHECKPOINT && PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  // D is static model weight — safe to issue before the PDL gdc_wait; hides LDG latency
  // under the async state/tile prefetches and replay_head MMA.
  float const D_val_0 = D_ptr ? toFloat(D_ptr[first_head]) : 0.f;

  static_assert(PIPELINE_STAGES <= 2, "head_loop generic for STAGES > 2 not yet implemented");

  // ── PRE-GDC PROLOGUE ──
  // G0: state_0 alone (MUST be its own group so wait_prior(1) can keep state_1 in flight).
  prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, lane, warp, d_tile,
                                                             first_head, cache_slot, /*buf=*/0);
  __pipeline_commit();  // G0: state_0

  // G1: old_tiles_0 (load_head is issue-only).
  load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
            /*IS_FIRST=*/true>(smem, params, lane, warp, d_tile, first_head, group_idx, cache_slot,
                               buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k, /*tile_buf=*/0);
  __pipeline_commit();  // G1: old_tiles_0

  // G2: state_1 + old_tiles_1 — pre-issued for STAGES=2 so they load across replay_head(0).
  if constexpr (PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1) {
    prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
        smem, params, lane, warp, d_tile, first_head + 1, cache_slot, /*buf=*/1);
    load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
              /*IS_FIRST=*/false>(smem, params, lane, warp, d_tile, first_head + 1, group_idx,
                                  cache_slot, buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k,
                                  /*tile_buf=*/1);
    __pipeline_commit();  // G2: state_1 + old_tiles_1
  }

  // Drain G0+G1 (state_0+old_tiles_0 ready); keep G2 in flight across replay_head(0).
  __pipeline_wait_prior(PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1 ? 1 : 0);
  __syncwarp();

  // ── HEAD 0: REPLAY (includes one-shot gdc_wait) ──
  replay_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
              NUM_WARPS, MUST_CHECKPOINT, /*IS_FIRST=*/true>(
      smem, params, lane, warp, d_tile, first_head, cache_slot, prev_k, rand_seed, /*tile_buf=*/0);
  // gdc_wait has fired; precompute output (cumAdt_vec, cb_scaled) is now globally visible.
  // Load cumAdt from one warp only — all warps would write the same values (same lane/head/seq).
  if (warp == 0 && lane < seq_len) {
    (smem.cumAdt + 0 * NPREDICTED)[lane] =
        cumAdt_ptr[(int64_t)(seq * params.nheads + first_head) * NPREDICTED_PAD_MMA_M + lane];
  }

  // ── POST-GDC X PIPELINE ──
  // G3: C + x_0.  (For STAGES=1 or MHC=1 this is the only post-gdc group.)
  load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/true>(
      smem, params, lane, warp, d_tile, first_head, group_idx, outer, seq_len, /*tile_buf=*/0);
  __pipeline_commit();  // G3: C + x_0

  // G4: x_1 — pre-issued for STAGES=2 so x_1 loads across output_head(0).
  if constexpr (PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1) {
    load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/false>(
        smem, params, lane, warp, d_tile, first_head + 1, group_idx, outer, seq_len,
        /*tile_buf=*/1);
    __pipeline_commit();  // G4: x_1
  }

  // Drain G2+G3 (old_tiles_1+state_1+C+x_0 ready); keep G4=x_1 in flight.
  __pipeline_wait_prior(PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1 ? 1 : 0);
  __syncthreads();  // fence warp-0 cumAdt write + C + x_0 pipeline data across all warps

  // ── HEAD 0: OUTPUT ──
  output_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
              PHILOX_ROUNDS, NUM_WARPS, MUST_CHECKPOINT>(
      smem, params, lane, warp, d_tile, first_head, seq, cache_slot, prev_k, outer, seq_len,
      out_seq_base, write_offset, rand_seed, /*tile_buf=*/0, D_val_0);

  // ── HEADS 1 .. MAIN_HEADS_PER_CTA-1 ──
  for (int h = 1; h < MAIN_HEADS_PER_CTA; ++h) {
    int const head = first_head + h;
    int const cur_buf = h % PIPELINE_STAGES;
    int const next_buf = (h + 1) % PIPELINE_STAGES;
    bool const has_next = (h + 1 < MAIN_HEADS_PER_CTA);

    __syncthreads();  // inter-head barrier: prev head's output / store_old_x done

    // Hoist D and cumAdt LDGs before the pipeline branches so their latency (~200-400 cycles)
    // is hidden under the cp.async pipeline setup and replay_head MMA compute.
    // gdc_wait fired during head 0's replay_head — precompute output is visible from h=1 onward.
    float const D_val_h = D_ptr ? toFloat(D_ptr[head]) : 0.f;
    float my_cumAdt = 0.f;
    if (warp == 0 && lane < seq_len)
      my_cumAdt = cumAdt_ptr[(int64_t)(seq * params.nheads + head) * NPREDICTED_PAD_MMA_M + lane];

    if constexpr (PIPELINE_STAGES == 1) {
      // STAGES=1: issue state_h + old_tiles_h here (no pre-fetch from prologue/prior iter).
      prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, lane, warp, d_tile,
                                                                 head, cache_slot, /*buf=*/0);
      load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                /*IS_FIRST=*/false>(smem, params, lane, warp, d_tile, head, group_idx, cache_slot,
                                    buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k,
                                    /*tile_buf=*/0);
      __pipeline_commit();  // G_tiles: state_h + old_tiles_h

      // Issue x_h in a SEPARATE group so it overlaps with replay_head below.
      load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, head, group_idx, outer, seq_len, /*tile_buf=*/0);
      __pipeline_commit();  // G_x: x_h

      __pipeline_wait_prior(1);  // drain G_tiles (state+tiles ready); keep G_x in flight
      __syncwarp();
      replay_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
                  NUM_WARPS, MUST_CHECKPOINT, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, head, cache_slot, prev_k, rand_seed, /*tile_buf=*/0);
      // my_cumAdt is in a register (loaded before replay); write it to smem now.
      if (warp == 0 && lane < seq_len) (smem.cumAdt + 0 * NPREDICTED)[lane] = my_cumAdt;
      __pipeline_wait_prior(0);  // drain G_x → x_h ready for output_head
      __syncthreads();           // fence warp-0 cumAdt write + x_h pipeline data across all warps
    } else {
      // STAGES=2: state_h + old_tiles_h already in smem (pre-issued in prologue or prior iter).
      // Issue state_{h+1} + old_tiles_{h+1} together in G_tiles (if there's a next head).
      if (has_next) {
        prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, lane, warp, d_tile,
                                                                   head + 1, cache_slot, next_buf);
        load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                  /*IS_FIRST=*/false>(smem, params, lane, warp, d_tile, head + 1, group_idx,
                                      cache_slot, buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k,
                                      next_buf);
      }
      __pipeline_commit();  // G_tiles: [state_{h+1} + old_tiles_{h+1}] if has_next, else empty

      // Issue x_{h+1} in its own group so it stays in flight across replay + output of head h.
      if (has_next) {
        load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/false>(
            smem, params, lane, warp, d_tile, head + 1, group_idx, outer, seq_len, next_buf);
        __pipeline_commit();  // G_xnext: x_{h+1}
      }

      // Always drain only the oldest outstanding group, keeping the 2 most recent in flight:
      //   has_next:  in-flight = G_xprev(x_h), G_tiles(state_{h+1}+tiles_{h+1}), G_xnext(x_{h+1})
      //              → wait_prior(2) drains G_xprev(x_h); x_h lands in smem; G_tiles+G_xnext
      //              stay in flight across replay_head(h) which doesn't need either.
      //   !has_next: in-flight = G_tiles(state_h+tiles_h), G_xnext(x_h), G_empty
      //              → wait_prior(2) drains G_tiles(state_h+tiles_h); replay_head(h) proceeds
      //              while x_h (G_xnext) still loads in the background.
      __pipeline_wait_prior(2);
      __syncwarp();
      replay_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
                  NUM_WARPS, MUST_CHECKPOINT, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, head, cache_slot, prev_k, rand_seed, cur_buf);
      // my_cumAdt is in a register (loaded before replay); write it to smem now.
      if (warp == 0 && lane < seq_len) (smem.cumAdt + cur_buf * NPREDICTED)[lane] = my_cumAdt;
      // !has_next: x_h (G_xnext) is still in flight — drain it now before the sync.
      //  has_next: x_h was already drained by wait_prior(2) above; this is a no-op.
      if (!has_next) __pipeline_wait_prior(0);
      __syncthreads();  // fence warp-0 cumAdt write + x_h across all warps
    }

    output_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                PHILOX_ROUNDS, NUM_WARPS, MUST_CHECKPOINT>(
        smem, params, lane, warp, d_tile, head, seq, cache_slot, prev_k, outer, seq_len,
        out_seq_base, write_offset, rand_seed, cur_buf, D_val_h);
  }
}

// =============================================================================
// Main kernel.  Template params mirror checkpointing_ssu_kernel so the launcher
// dispatches both with the same args.
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1,
          bool VARLEN = false, int MAIN_HEADS_PER_CTA = 1, int PIPELINE_STAGES = 1>
// __maxnreg__(64): 65536/(64·128)=8 blocks/SM at 50% occupancy.  Spills ~4 scalar regs
// (from tile_buf address arithmetic), but fewer blocks/SM from a higher cap hurts more.
__global__ __maxnreg__(64) void checkpointing_ssu_main_kernel(CheckpointingSsuParams params) {
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32, "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 layout)");
  static_assert(NPREDICTED <= MAX_WINDOW, "NPREDICTED must be <= MAX_WINDOW");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG, "MAX_WINDOW must be <= MMA::K_BIG=16");
  assert(params.d_split == D_SPLIT);

  using SmemT = CheckpointingSsuMainStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA,
                                            DSTATE, PIPELINE_STAGES>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── Grid (D_SPLIT, batch, ngroups·ceil(HPG/MAIN_HEADS_PER_CTA)) ──
  // grid.z packs (group, head-tile): each CTA owns MAIN_HEADS_PER_CTA CONSECUTIVE heads
  // within ONE group, looped below so the per-group C / old_B load once and reuse across
  // them.  At MAIN_HEADS_PER_CTA==1, HEAD_TILES==HPG and first_head==blockIdx.z (one head
  // per CTA) — bit-identical to the old (D_SPLIT, batch, nheads) per-head grid.
  static_assert(HEADS_PER_GROUP % MAIN_HEADS_PER_CTA == 0,
                "MAIN_HEADS_PER_CTA must divide HEADS_PER_GROUP");
  constexpr int HEAD_TILES = HEADS_PER_GROUP / MAIN_HEADS_PER_CTA;
  int const d_tile = blockIdx.x;
  int const seq = blockIdx.y;
  int group_idx, first_head;
  if constexpr (MAIN_HEADS_PER_CTA == 1) {
    // Degenerate head-tile (one head per CTA): first_head == blockIdx.z, exactly the old
    // per-head grid.  Skip the div/mod/mul so the compiler addresses straight off %ctaid.z
    // instead of materializing first_head/head_tile in registers.
    first_head = blockIdx.z;
    group_idx = blockIdx.z / HEADS_PER_GROUP;
  } else {
    group_idx = blockIdx.z / HEAD_TILES;
    int const head_tile = blockIdx.z % HEAD_TILES;
    first_head = group_idx * HEADS_PER_GROUP + head_tile * MAIN_HEADS_PER_CTA;
  }
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;

  // ── Per-slot setup ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
  if (cache_slot == params.pad_slot_id) return;
  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  int seq_len;
  int64_t outer;
  if constexpr (VARLEN) {
    auto const* __restrict__ cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);
    int const bos = __ldg(&cu[seq]);
    int const eos = __ldg(&cu[seq + 1]);
    seq_len = eos - bos;
    if (seq_len <= 0) return;
    outer = (int64_t)bos;
  } else {
    seq_len = NPREDICTED;
    outer = (int64_t)seq;
  }
  int64_t const out_seq_base = outer * params.out_stride_seq;

  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  int const write_offset = must_checkpoint ? 0 : prev_k;

  if (must_checkpoint) {
    head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
              PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/true, MAIN_HEADS_PER_CTA,
              PIPELINE_STAGES>(smem, params, lane, warp, d_tile, first_head, group_idx, seq,
                               cache_slot, buf_read, prev_k, outer, seq_len, out_seq_base,
                               write_offset);
  } else {
    head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
              PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/false, MAIN_HEADS_PER_CTA,
              PIPELINE_STAGES>(smem, params, lane, warp, d_tile, first_head, group_idx, seq,
                               cache_slot, buf_read, prev_k, outer, seq_len, out_seq_base,
                               write_offset);
  }

  // ── EXTERNAL PDL: signal a programmatic DOWNSTREAM kernel that `output` is fully written
  // (ALL heads done; the per-head cache writes are next-step-only).  Gated by ENABLE_PDL.
  // (The internal precompute→main chain uses the one-shot gdc_wait inside replay_head.) ──
  if constexpr (ENABLE_PDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
