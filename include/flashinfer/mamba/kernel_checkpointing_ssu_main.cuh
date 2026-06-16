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
template <typename input_t, typename state_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE>
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

  // C (matmul-3 A-operand) — per-group conv1d output.
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];
  // x (matmul-4 B-operand) — per-head new-token conv1d output.
  alignas(16) input_t x[NPREDICTED_PAD_MMA_M * D_SMEM_COLS];
  // z — the gate (conv1d-independent).
  alignas(16) input_t z[NPREDICTED_SWIZZLE_R * D_SMEM_COLS];
  // old_x — replay A-operand (write path) / cb_old@old_x (no-write); per-head cache.
  alignas(16) input_t old_x[MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS];
  // old_B — replay B-operand (write path); per-group cache.
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];
  float old_dt[MAX_WINDOW];
  float old_cumAdt[MAX_WINDOW];
  // cumAdt — precompute output (cumAdt_vec) → exp() β epilogue.
  float cumAdt[NPREDICTED];
  // state — per-CTA D-slice of the SSM state.
  alignas(16) state_t state[D_PER_CTA * DSTATE];
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
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, bool IS_FIRST, typename SmemT>
__device__ __forceinline__ void load_main_data(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int group_idx, int64_t cache_slot, int buf_read,
                                               int64_t outer, int seq_len, bool must_checkpoint,
                                               int prev_k) {
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

  // ── State: per-CTA D-slice ([D_PER_CTA, DSTATE]) ──
  {
    auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
    int64_t const state_base = cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE +
                               (int64_t)d_tile_off * DSTATE;
    if constexpr (DIM == D_PER_CTA) {
      load_state_per_warp<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp,
                                                                 lane);
    } else {
      int const tid = warp * warpSize + lane;
      load_state_cta<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, tid);
    }
  }

  // ── Cached tiles, ONE warp each (no longer all-warp redundant) — every consumer
  // runs after a __syncthreads (the replay's, or the hoisted pre-gdc_wait one),
  // which publishes each warp's tile cross-warp, so a single load per tile
  // suffices.  Spreading them across warps parallelizes the cp.async and cuts the
  // redundant LDGSTS that fed mio_throttle.  conv1d-INDEPENDENT (C / new-token x
  // are conv1d outputs → POST-gdc_wait via load_conv_inputs; no new-B). ──
  // old_x (warp 0): BOTH paths — no-write cb_old@old_x, write replay.
  if (warp == 0)
    load_tile_async<OxShape, MAX_WINDOW>(smem.old_x, old_x_ptr + ox_base, params.old_x_stride_token,
                                         lane);
  // old_B (warp 1) + old_dt/old_cumAdt (warp 2): WRITE-PATH ONLY (the replay folds
  // old tokens into state from them).  On the (common) no-write path the precompute
  // already baked them into cb_old, so the main needs NONE — except the single
  // tail old_cumAdt[prev_k-1] that compute_no_write_output exp's for the β total
  // decay.  (The monolithic can't skip them — it computes CB_old inline.)
  if (must_checkpoint) {
    // old_B is per-GROUP (no head term in oB_base) → load ONCE for the head-tile (is_first)
    // and reuse across the loop's heads.  old_dt/old_cumAdt are per-HEAD → load every head.
    if constexpr (IS_FIRST)
      if (warp == 1)
        load_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, old_B_ptr + oB_base,
                                               params.old_B_stride_token, lane);
    if (warp == 2)
      load_old_dt_cumAdt(params, lane, cache_slot, buf_read, head, MAX_WINDOW, smem.old_dt,
                         smem.old_cumAdt);
  } else if (prev_k > 0 && warp == 0 && lane == 0) {
    auto const* __restrict__ oca_ptr = reinterpret_cast<float const*>(params.old_cumAdt);
    int64_t const ca_base = cache_slot * params.old_cumAdt_stride_seq +
                            (int64_t)buf_read * params.old_cumAdt_stride_dbuf +
                            (int64_t)head * params.old_cumAdt_stride_head;
    smem.old_cumAdt[prev_k - 1] = oca_ptr[ca_base + prev_k - 1];  // tail for β only
  }
  // z (warp 3): the gate — bypasses conv1d, conv1d-INDEPENDENT.
  if (warp == 3 && z_ptr) {
    int64_t const z_base = outer * params.z_stride_seq + (int64_t)head * DIM + d_tile_off;
    load_tile_async<ZShape, NPREDICTED>(smem.z, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }

  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
}

// -----------------------------------------------------------------------------
// Post-gdc_wait load of the conv1d OUTPUTS the main consumes: C (matmul-3
// A-operand) and the new-token x (matmul-4 B-operand).  Called AFTER gdc_wait so
// conv1d's writes are guaranteed visible (the main co-launches ahead of conv1d's
// completion — see load_main_data header).
//
// Both are loaded REDUNDANTLY by every warp (no warp guard): matmul-3 reads the
// full C from every warp, and each warp's matmul-4 reads its own D-slice of x
// out of the full tile.  Loading them per-warp (idempotent) means there is NO
// cross-warp hazard — only the cross-LANE LDSM read inside each warp — so the
// caller publishes with a cheap __syncwarp instead of a block barrier, keeping
// the post-wait critical path free of __syncthreads.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, bool IS_FIRST,
          typename SmemT>
__device__ __forceinline__ void load_conv_inputs(SmemT& smem, CheckpointingSsuParams const& params,
                                                 int lane, int d_tile, int head, int group_idx,
                                                 int64_t outer, int seq_len) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const x_base = outer * params.x_stride_seq + (int64_t)head * DIM + d_tile_off;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // C is per-GROUP (C_base has no head term) → load ONCE for the head-tile (is_first) and
  // reuse across the loop's heads.  x is per-HEAD → load every head.
  if constexpr (IS_FIRST)
    load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane,
                                        seq_len);
  load_tile_async<XShape, NPREDICTED>(smem.x, x_ptr + x_base, params.x_stride_token, lane, seq_len);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
}

// -----------------------------------------------------------------------------
// Per-head work for the head-tiled main: load (cache + conv1d + precompute outputs),
// state replay (write path), output, and the old_x cache writeback — for ONE head.
// Templated on MUST_CHECKPOINT (uniform per CTA: must_checkpoint is per-seq).  State
// stays in smem the whole time — loaded once per head, replayed in place, consumed by the
// output — it NEVER round-trips to gmem (the replay's C8 HBM write is the cache output,
// not a re-read).  `is_first` gates the per-GROUP loads (C, old_B) and the one-shot
// gdc_wait so they run once per head-tile and are reused across the loop's heads.
template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS,
          bool MUST_CHECKPOINT, bool IS_FIRST, typename SmemT>
__device__ __forceinline__ void process_head(SmemT& smem, CheckpointingSsuParams const& params,
                                             int lane, int warp, int d_tile, int head,
                                             int group_idx, int seq, int64_t cache_slot,
                                             int buf_read, int prev_k, int64_t outer, int seq_len,
                                             int64_t out_seq_base, int write_offset,
                                             int64_t rand_seed) {
  // Inter-head barrier: the PREVIOUS head's output / store_old_x READ smem.state & smem.x
  // while THIS head's load_main_data / load_conv_inputs WRITE them — sync so the buffer
  // reuse can't race.  (C / old_B are write-once on IS_FIRST, unaffected.)  Compile-time-
  // elided for the tile's first head (nothing precedes it) so MHC=1 keeps the flat body.
  if constexpr (!IS_FIRST) __syncthreads();

  // D (tie_hdim scalar); A / dt_bias not needed here (no C1/C2).
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ── PRE-gdc_wait: cache + state replay (precompute-INDEPENDENT, overlaps precompute) ──
  load_main_data<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                 IS_FIRST>(smem, params, lane, warp, d_tile, head, group_idx, cache_slot, buf_read,
                           outer, seq_len, MUST_CHECKPOINT, prev_k);

  if constexpr (MUST_CHECKPOINT) {
    __syncthreads();  // state (per-warp split) visible cross-warp before the replay
    int64_t const state_ptr_offset =
        cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;
    state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                  (int64_t)d_tile * D_PER_CTA * DSTATE;
    replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS>(
        smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
        /*must_checkpoint=*/true);
  }

  __syncthreads();  // publish cached/replayed data cross-warp before the (head-0) gdc_wait

  // INTERNAL PDL gdc_wait — ONE-SHOT (first head of the tile only).  Once head 0 has waited,
  // the precompute is finished + visible for the WHOLE CTA, so later heads read
  // cb_scaled / cumAdt_vec without re-waiting.  No-op without a programmatic predecessor.
  if constexpr (IS_FIRST) cudaGridDependencySynchronize();

  // ── POST-gdc_wait: conv1d outputs (C once per tile, x per head) + precompute cumAdt_vec ──
  load_conv_inputs<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, IS_FIRST>(
      smem, params, lane, d_tile, head, group_idx, outer, seq_len);
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  if (lane < seq_len) {
    smem.cumAdt[lane] =
        cumAdt_ptr[(int64_t)(seq * params.nheads + head) * NPREDICTED_PAD_MMA_M + lane];
  }
  __syncwarp();  // publish x (per-warp) + cumAdt for the cross-lane β broadcast

  // Precompute CB pointers (fragA-native, per (batch_slot, head)).  REGS = K/2.
  constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;
  constexpr int CB_OLD_REGS = SmemT::MAX_WINDOW_PAD_MMA_K / 2;
  auto const* __restrict__ cb_gmem_head =
      reinterpret_cast<input_t const*>(params.cb_scaled) +
      (int64_t)(seq * params.nheads + head) * warpSize * CB_NEW_REGS;

  if constexpr (MUST_CHECKPOINT) {
    // Output: matmul-3 (C@folded-state) + β + matmul-4 (cb_scaled@x).  Old tokens' contribution
    // is already in the folded state (no cb_old).
    compute_and_store_output<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                             PHILOX_ROUNDS, /*READ_PRECOMPUTED_CB=*/true>(
        smem, params, warp, lane, d_tile, out_seq_base, head, cache_slot, D_val,
        /*must_checkpoint=*/true, seq_len, cb_gmem_head);
  } else {
    // No replay (state = s_0); output = β·C@s_0 + cb_scaled@x + cb_old@old_x.
    auto const* __restrict__ cb_old_head =
        reinterpret_cast<input_t const*>(params.cb_old) +
        (int64_t)(seq * params.nheads + head) * warpSize * CB_OLD_REGS;
    compute_no_write_output<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                            NUM_WARPS, /*READ_PRECOMPUTED_CB=*/true>(
        smem, params, warp, lane, prev_k, d_tile, out_seq_base, head, cache_slot, D_val, seq_len,
        cb_gmem_head, cb_old_head);
  }

  // ── Cache: the main owns old_x (it loaded x).  old_B / old_dt / old_cumAdt are the
  // precompute's.  Per-head writeback. ──
  store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                   cache_slot, write_offset, seq_len);
}

// =============================================================================
// Main kernel.  Template params mirror checkpointing_ssu_kernel so the launcher
// dispatches both with the same args.
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1,
          bool VARLEN = false, int MAIN_HEADS_PER_CTA = 1>
__global__ void checkpointing_ssu_main_kernel(CheckpointingSsuParams params) {
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32, "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 layout)");
  static_assert(NPREDICTED <= MAX_WINDOW, "NPREDICTED must be <= MAX_WINDOW");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG, "MAX_WINDOW must be <= MMA::K_BIG=16");
  assert(params.d_split == D_SPLIT);

  using SmemT =
      CheckpointingSsuMainStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>;
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

  // ════════════════════════════════════════════════════════════════════════
  // Head loop: each CTA processes MAIN_HEADS_PER_CTA consecutive heads of ONE group.
  // must_checkpoint is per-seq (uniform per CTA) → dispatch it to the process_head template
  // ONCE, outside the loop, so the per-head path carries no runtime branch.  process_head
  // fires the one-shot gdc_wait + the per-group C / old_B loads only on the tile's first head;
  // state stays in smem per head (no gmem round-trip — the replay's C8 write is the only HBM
  // state I/O).  External PDL fires once, after ALL heads' outputs are written.
  // ════════════════════════════════════════════════════════════════════════
  // PEEL head 0 (IS_FIRST=true) from the rest (IS_FIRST=false) so IS_FIRST is COMPILE-TIME:
  // the per-group C / old_B loads, the one-shot gdc_wait and the inter-head __syncthreads all
  // fold via `if constexpr`.  At MAIN_HEADS_PER_CTA==1 the `h = 1` loop is empty, so this
  // collapses to a single inlined IS_FIRST=true body — bit-AND-perf-identical to the old flat
  // per-head main (no loop, no runtime is_first branch, no extra address math).
  if (must_checkpoint) {
    // rand_seed: computed ONCE for the write path (replay), reused by every head of the tile.
    int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
    process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/true, /*IS_FIRST=*/true>(
        smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read, prev_k,
        outer, seq_len, out_seq_base, write_offset, rand_seed);
    for (int h = 1; h < MAIN_HEADS_PER_CTA; ++h) {
      process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                   PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/true, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, first_head + h, group_idx, seq, cache_slot, buf_read,
          prev_k, outer, seq_len, out_seq_base, write_offset, rand_seed);
    }
  } else {
    process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/false, /*IS_FIRST=*/true>(
        smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read, prev_k,
        outer, seq_len, out_seq_base, write_offset, /*rand_seed=*/0);
    for (int h = 1; h < MAIN_HEADS_PER_CTA; ++h) {
      process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                   PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/false, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, first_head + h, group_idx, seq, cache_slot, buf_read,
          prev_k, outer, seq_len, out_seq_base, write_offset, /*rand_seed=*/0);
    }
  }

  // ── EXTERNAL PDL: signal a programmatic DOWNSTREAM kernel that `output` is fully written
  // (ALL heads done; the per-head cache writes are next-step-only).  Gated by ENABLE_PDL —
  // the SSU participating in the broader pipeline chain.  (The internal precompute→main chain
  // is the per-head one-shot gdc_wait inside process_head.) ──
  if constexpr (ENABLE_PDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
