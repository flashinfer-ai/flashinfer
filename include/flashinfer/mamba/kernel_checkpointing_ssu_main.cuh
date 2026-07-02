/*
 * Copyright (c) 2026 by FlashInfer team.
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
// gdc_wait is a no-op without a programmatic predecessor (T0 / non-PDL)
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
// STATE_PIPE = ring depth (STAGES).  >1 makes EVERY per-tile buffer (C, old_B, z, old_x, old_dt,
// old_cumAdt, cumAdt, x, state) STATE_PIPE-slotted so the persistent all-async ring can prefetch a
// whole work-unit's bundle (incl per-GROUP C / old_B) into one slot — a grid-stride CTA's
// consecutive work-units are different groups, so there's no same-group C/old_B reuse to preserve.
// STATE_PIPE=1 ⇒ single slot == the original layout, bit-identical (the monolith's use).
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

  static constexpr int C_ELEMS = NPREDICTED_SWIZZLE_R * DSTATE;
  static constexpr int OLD_B_ELEMS = MAX_WINDOW_PAD_MMA_K * DSTATE;
  static constexpr int Z_ELEMS = NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  static constexpr int OLD_X_ELEMS = MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
  static constexpr int X_ELEMS = NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  static constexpr int STATE_ELEMS = D_PER_CTA * DSTATE;
  static constexpr int CUMADT_ELEMS = NPREDICTED_PAD_MMA_M;
  // CB fragB-native blocks (operand swap): one warp of lane-major Packs (REGS_B =
  // NUM_OUT_NTILES·K/4 — HALF the old fragA store at mtp≤8).  Staged here so the output MMA's
  // B-operand loads from smem (LDS) instead of a just-in-time LDG.  Precompute writes this exact
  // layout (scale_store_cb).
  static constexpr int NUM_OUT_NTILES = NPREDICTED_PAD_MMA_N / MMA_prop::N;  // 1 (mtp≤8) or 2
  static constexpr int CB_NEW_REGS_B = NUM_OUT_NTILES * (NPREDICTED_PAD_MMA_M / 4);
  static constexpr int CB_OLD_REGS_B = NUM_OUT_NTILES * (MAX_WINDOW_PAD_MMA_K / 4);
  static constexpr int CB_NEW_ELEMS = 32 * CB_NEW_REGS_B;
  static constexpr int CB_OLD_ELEMS = 32 * CB_OLD_REGS_B;
  alignas(16) input_t C[STATE_PIPE * C_ELEMS];          // matmul-3 A-operand (conv1d output)
  alignas(16) input_t old_B[STATE_PIPE * OLD_B_ELEMS];  // replay B-operand (per-group cache)
  alignas(16) input_t z[STATE_PIPE * Z_ELEMS];
  alignas(16) input_t old_x[STATE_PIPE * OLD_X_ELEMS];
  float old_dt[STATE_PIPE * MAX_WINDOW];
  float old_cumAdt[STATE_PIPE * MAX_WINDOW];
  float cumAdt[STATE_PIPE * CUMADT_ELEMS];
  alignas(16) input_t x[STATE_PIPE * X_ELEMS];
  alignas(
      16) input_t cb_new[STATE_PIPE * CB_NEW_ELEMS];  // matmul-4 A-operand (precompute cb_scaled)
  alignas(16) input_t cb_old[STATE_PIPE * CB_OLD_ELEMS];  // no-write CB_old A-operand (precompute)
  alignas(16) state_t state[STATE_PIPE * STATE_ELEMS];
};

// Ring depth (STAGES) for the PERSISTENT main's cross-iteration prefetch pipeline.  Each of the
// STAGES smem slots holds one work-unit's full bundle; the grid loop prefetches STAGES work-units
// ahead so the 16 KB state load (the long_scoreboard pole) issues well before it's consumed.
// Higher = deeper overlap but more smem (lower occupancy); 2 is the sweet spot at this footprint.
constexpr int MAIN_STATE_PIPE = 2;

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
// no __pipeline_commit / wait / syncwarp; process_head owns the cp.async pipeline so these loads
// and the state prefetch can be committed + drained together at the right depth.  State is NOT
// loaded here — its sole loader is prefetch_state (the one tile the cross-head pipeline
// double-buffers).
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

  // Indices into contiguous inner dims (head·DIM, group·DSTATE) are ≪ 2^31 → fold in 32-bit and
  // cast ONCE onto the int64 base (native IMAD, not a 64-bit chain).  Buffer/slot-distance strides
  // (*_stride_seq, *_stride_dbuf) stay 64-bit — they can exceed 2^31 in production layouts.
  int64_t const ox_base = cache_slot * params.old_x_stride_seq + (int64_t)(head * DIM + d_tile_off);
  int64_t const oB_base = cache_slot * params.old_B_stride_seq +
                          (int64_t)buf_read * params.old_B_stride_dbuf +
                          (int64_t)(group_idx * DSTATE);

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  using ZShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<D_PER_CTA>>;
  using OldBShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  using OxShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

  // Per-tile smem slots: offset by tile_buf into the STATE_PIPE-strided ring arrays.
  auto* old_x_slot = smem.old_x + tile_buf * SmemT::OLD_X_ELEMS;
  float* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  float* old_cumAdt_slot = smem.old_cumAdt + tile_buf * MAX_WINDOW;
  auto* z_slot = smem.z + tile_buf * SmemT::Z_ELEMS;
  auto* old_B_slot = smem.old_B + tile_buf * SmemT::OLD_B_ELEMS;

  // Per-head load distribution: one tensor per warp, each a single-warp (conflict-free)
  // load_tile_async.  old_x is the variable / largest small tensor (up to max_window rows), so
  // it gets two warps; x and z (each seq_len valid rows) get one each:
  //   W0 → old_x[0:8]   W1 → old_x[8:16]   W2 → x (load_x)   W3 → z
  // old_x is PREDICATED to prev_k (PNAT): prev_k=0 loads nothing, high PNAT fills both halves —
  // its DRAM scales with the window occupancy instead of always reading max_window rows.  The
  // split is at row 8 = the Swizzle<3,3,3> period, so the upper sub-tile's addresses match
  // rows 8-15 of the full layout exactly.  W1's half ZFILLs when prev_k≤8 (cheap), and at
  // max_window=8 (no upper half) W1 is idle — accepted, not generalized per combination.
  using OxHalfShape = cute::Shape<cute::Int<8>, cute::Int<D_PER_CTA>>;
  if (warp == 0)
    load_tile_async<OxHalfShape, /*VALID_ROWS=*/8>(old_x_slot, old_x_ptr + ox_base,
                                                   params.old_x_stride_token, lane,
                                                   /*valid_rows_rt=*/prev_k);
  if constexpr (MAX_WINDOW_PAD_MMA_K > 8)
    if (warp == 1)
      load_tile_async<OxHalfShape, /*VALID_ROWS=*/8>(
          old_x_slot + 8 * SmemT::D_SMEM_COLS, old_x_ptr + ox_base + 8 * params.old_x_stride_token,
          params.old_x_stride_token, lane, /*valid_rows_rt=*/prev_k - 8);
  if (must_checkpoint) {
    if constexpr (IS_FIRST)
      if (warp == 1)
        load_tile_async<OldBShape, MAX_WINDOW>(old_B_slot, old_B_ptr + oB_base,
                                               params.old_B_stride_token, lane);
    if (warp == 2)
      load_old_dt_cumAdt(params, lane, cache_slot, buf_read, head, MAX_WINDOW, old_dt_slot,
                         old_cumAdt_slot);
  } else if (prev_k > 0 && warp == 0 && lane == 0) {
    auto const* __restrict__ oca_ptr = reinterpret_cast<float const*>(params.old_cumAdt);
    int64_t const ca_base = cache_slot * params.old_cumAdt_stride_seq +
                            (int64_t)buf_read * params.old_cumAdt_stride_dbuf +
                            (int64_t)(head * (int)params.old_cumAdt_stride_head);
    // β tail (one float).  cp.async, NOT a synchronous LDG→STS: this is the only blocking load left
    // in the prefetch path, and as a single-lane sync load it made W0 the laggard at the publish
    // barrier.  As cp.async it's non-blocking (W0 issues + continues) and drains with the bundle.
    __pipeline_memcpy_async(&old_cumAdt_slot[prev_k - 1], &oca_ptr[ca_base + prev_k - 1],
                            sizeof(float));
  }
  if (warp == 3 && z_ptr) {
    int64_t const z_base = outer * params.z_stride_seq + (int64_t)(head * DIM + d_tile_off);
    load_tile_async<ZShape, NPREDICTED>(z_slot, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }
  // NO commit/wait/syncwarp — process_head drains (see header).
}

template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void prefetch_state(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int state_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
  int64_t const state_base =
      cache_slot * params.state_stride_seq + (int64_t)(head * DIM * DSTATE + d_tile_off * DSTATE);
  if constexpr (DIM == D_PER_CTA) {
    load_state_per_warp<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp,
                                                               lane, state_buf);
  } else {
    int const tid = warp * warpSize + lane;
    load_state_cta<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, tid,
                                                          state_buf);
  }
  // NOTE: issue ONLY — no __pipeline_commit / wait here.  process_head owns the whole cp.async
  // pipeline (commit + wait_prior) so the prefetched states can stay in flight across the
  // gdc_wait / replay; committing here would let an internal drain complete them too early.
}

// load_x: loads x (and C for IS_FIRST) from gmem → smem slot tile_buf.  ISSUE ONLY —
// no __pipeline_commit / wait / syncwarp.  process_head owns the whole cp.async pipeline.
// Per-head load distribution (see load_head): C per-GROUP on W1 (IS_FIRST only); x per-HEAD on
// W2, single-warp (conflict-free).  old_x (W0-1) and z (W3) load in parallel — all 4 warps busy
// without the CTA-wide multi-warp writes that caused the d_split=2 LDGSTS bank conflict.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, bool IS_FIRST,
          typename SmemT>
__device__ __forceinline__ void load_x(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int warp, int d_tile, int head, int group_idx, int64_t outer,
                                       int seq_len, int tile_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)(group_idx * DSTATE);
  int64_t const x_base = outer * params.x_stride_seq + (int64_t)(head * DIM + d_tile_off);
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // C → slot, on W1.
  auto* C_slot = smem.C + tile_buf * SmemT::C_ELEMS;
  if constexpr (IS_FIRST)
    if (warp == 1)
      load_tile_async<CShape, NPREDICTED>(C_slot, C_ptr + C_base, params.C_stride_token, lane,
                                          seq_len);
  // x → slot, single-warp on W2 (W0-1 carry old_x, W3 carries z).
  auto* x_slot = smem.x + tile_buf * SmemT::X_ELEMS;
  if (warp == 2)
    load_tile_async<XShape, NPREDICTED>(x_slot, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
  // NOTE: NO commit/wait — process_head owns the pipeline.
}

// load_cumAdt: load ONE work-unit's cumAdt (precompute output) gmem → smem ring slot tile_buf.
// WARP-0 ONLY (the caller guards with `if (warp == 0)`); lanes 0..seq_len-1 each carry one element.
// Plain stores (not cp.async) → never touches the cp.async ring FIFO; the caller's __syncthreads()
// broadcasts to all warps.  MUST run AFTER the gdc_wait that makes cumAdt_vec globally visible.
template <typename SmemT>
__device__ __forceinline__ void load_cumAdt(SmemT& smem, CheckpointingSsuParams const& params,
                                            int lane, int seq, int first_head, int seq_len,
                                            int tile_buf) {
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  float* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;
  if (lane < seq_len)
    cumAdt_slot[lane] =
        cumAdt_ptr[(int64_t)(seq * params.nheads + first_head) * NPREDICTED_PAD_MMA_M + lane];
}

// load_cumAdt_async: cp.async variant of load_cumAdt — joins the post-gdc cp.async group instead of
// a synchronous LDG+STS (which was the #1 long_scoreboard pole: the STS stalled on its feeding
// LDG). WARP-0 ONLY; lanes 0..seq_len-1 each cp.async one float.  Drained + published by the
// caller's pipeline_wait + __syncthreads, same as the rest of the bundle.  ISSUE ONLY — caller
// commits.
template <typename SmemT>
__device__ __forceinline__ void load_cumAdt_async(SmemT& smem, CheckpointingSsuParams const& params,
                                                  int lane, int seq, int first_head, int seq_len,
                                                  int tile_buf) {
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  float* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;
  if (lane < seq_len)
    __pipeline_memcpy_async(
        &cumAdt_slot[lane],
        &cumAdt_ptr[(int64_t)(seq * params.nheads + first_head) * NPREDICTED_PAD_MMA_M + lane],
        sizeof(float));
}

// load_cb_async: cp.async the per-(seq,head) CB blocks (precompute outputs, fragA-native,
// lane-major) gmem → smem ring slot tile_buf, so the output MMA's A-operand loads from smem (LDS,
// short scoreboard) instead of a just-in-time LDG straight to registers — the latter was the #1
// long_scoreboard pole (the HMMA stalled on the cb LDG).  All NUM_WARPS warps consume the SAME 32
// Packs, so ONE warp's cp.async (32) feeds them all, replacing 128 redundant LDGs.  cb_scaled
// always; cb_old only on the no-write path (mirrors old_B's gating).  W3 (idle in load_x's post-gdc
// set) issues both.  Precompute output ⇒ MUST run AFTER gdc_wait.  ISSUE ONLY — caller commits
// (joins the post-gdc cp.async group).
template <typename input_t, typename SmemT>
__device__ __forceinline__ void load_cb_async(SmemT& smem, CheckpointingSsuParams const& params,
                                              int lane, int warp, int seq, int head,
                                              bool must_checkpoint, int tile_buf) {
  // fragB-native (operand swap): REGS_B = NUM_OUT_NTILES·K/4 regs/lane (precompute scale_store_cb).
  constexpr int CB_NEW_REGS_B = SmemT::CB_NEW_REGS_B;
  constexpr int CB_OLD_REGS_B = SmemT::CB_OLD_REGS_B;
  // Whole flattened offset stays 32-bit (batch·nheads·32·REGS_B ≪ 2³¹); widen only at the ptr add.
  auto const* __restrict__ cb_new_g = reinterpret_cast<input_t const*>(params.cb_scaled) +
                                      (int64_t)((seq * params.nheads + head) * 32 * CB_NEW_REGS_B);
  input_t* cb_new_s = smem.cb_new + tile_buf * SmemT::CB_NEW_ELEMS;
  __pipeline_memcpy_async(cb_new_s + lane * CB_NEW_REGS_B, cb_new_g + lane * CB_NEW_REGS_B,
                          CB_NEW_REGS_B * sizeof(input_t));
  if (!must_checkpoint) {  // no-write path also needs CB_old @ old_x
    auto const* __restrict__ cb_old_g =
        reinterpret_cast<input_t const*>(params.cb_old) +
        (int64_t)((seq * params.nheads + head) * 32 * CB_OLD_REGS_B);
    input_t* cb_old_s = smem.cb_old + tile_buf * SmemT::CB_OLD_ELEMS;
    __pipeline_memcpy_async(cb_old_s + lane * CB_OLD_REGS_B, cb_old_g + lane * CB_OLD_REGS_B,
                            CB_OLD_REGS_B * sizeof(input_t));
  }
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

// add_init_out_ring (OPERAND SWAP): OUT.1 = state @ Cᵀ → frag_y[d, t] = Σ_n state[d,n]·C[t,n].
// A = state (SM75_U32x4_LDSM_N; state smem is dstate=K contiguous → non-transpose), B = C
// (SM75_U32x2_LDSM_N).  Reuses the shared, operand-generic pipelined_kloop_gemm with A/B swapped
// (the monolith's add_init_out call A=C,B=state is untouched).  Warps split M=DIM (tiled_mma
// Shape<NUM_WARPS,1>); C is the broadcast B operand.  Both C and state are slot-indexed for the
// persistent ring.  2-byte state only — the A-operand LDSM path (4-byte would need UniversalCopy).
template <typename input_t, typename state_t, int D_PER_CTA, int DSTATE, typename SmemT,
          typename TiledMma, typename ThrMma, typename... FragY>
__device__ __forceinline__ void add_init_out_ring(SmemT const& smem, TiledMma const& tiled_mma,
                                                  ThrMma const& thr_mma, int tid, int state_buf,
                                                  FragY&... frag_y) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr int M_TILE = cute::tile_size<0>(TiledMma{});  // 16·NUM_WARPS = D_PER_CTA
  static_assert(sizeof(state_t) == 2, "operand-swap OUT.1 needs 2-byte state (A-operand LDSM)");
  using AView = MMA_prop::operand_t;  // state viewed as the 2-byte MMA operand (bf16)
  // A = state [D_PER_CTA, DSTATE] (dstate contiguous = K-major → x4 non-transpose LDSM),
  // slot-indexed.
  auto const layout_state = make_swizzled_layout_rc<AView, D_PER_CTA, DSTATE>();
  Tensor smem_state = make_tensor(
      make_smem_ptr(reinterpret_cast<AView const*>(smem.state + state_buf * SmemT::STATE_ELEMS)),
      layout_state);
  Tensor smem_state_ktiled =
      local_tile(smem_state, make_tile(Int<M_TILE>{}, Int<K_TILE>{}), make_coord(_0{}, _));
  // B = C [NPRED_pad, DSTATE] (dstate contiguous → x2 non-transpose LDSM), slot-indexed.
  auto const layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(
                                  smem.C + state_buf * SmemT::C_ELEMS)),
                              layout_C);
  pipelined_kloop_gemm<3, NUM_K_TILES, AView, input_t, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_state_ktiled, smem_C, frag_y...);
}

// ── OPERAND-SWAP output helpers ([DIM,NPRED]) ────────────────────────────────
// x / old_x are the A-operands (transpose ldmatrix from the [token,d] smem via the [d,token]
// view); CB / CB_old are the B-operands (fragB, in registers).  Kept separate from the shared
// add_cb_x / add_D_skip / compute_z_gating (the monolith + 8-bit still use those, unswapped).

// OUT.2/3 (swap): frag_y[d,t] += Σ_c operand[c,d]·CB[t,c].  A = operand [M=d, K=c] via transpose
// LDSM (operand smem is [token,d]; smem_trans is the [d,token] view), B = CB (fragB).  Single
// K-tile (K = NPREDICTED_PAD_MMA_M for new, MAX_WINDOW_PAD_MMA_K for old).  n = output N-tile (t).
template <typename MmaT, int M_TILE, int K_CONTRACT, typename FragY, typename FragCB,
          typename SmemTrans, typename S2RA, typename S2RThrA, typename ThrMma, typename TiledMma>
__device__ __forceinline__ void add_cbx_swapped(FragY& frag_y, FragCB const& frag_CB,
                                                SmemTrans const& smem_trans, S2RA const& s2r_A,
                                                S2RThrA const& s2r_thr_A, ThrMma const& thr_mma,
                                                TiledMma const& tiled_mma, int n) {
  using namespace cute;
  Tensor a_tile =
      local_tile(smem_trans, make_tile(Int<M_TILE>{}, Int<K_CONTRACT>{}), make_coord(_0{}, _0{}));
  auto a_s2r = s2r_thr_A.partition_S(a_tile);
  auto frag_A = thr_mma.partition_fragment_A(
      make_tensor((MmaT*)0x0, make_shape(Int<M_TILE>{}, Int<K_CONTRACT>{})));
  auto frag_A_view = s2r_thr_A.retile_D(frag_A);
  cute::copy(s2r_A, a_s2r, frag_A_view);
  // frag_CB is the caller's pre-selected output-N-tile B-fragment (frag_CB_new[n]/frag_CB_old[n]);
  // x (frag_A) is the same for all output N-tiles, so n only selects which t-tile accumulates here.
  (void)n;
  cute::gemm(tiled_mma, frag_y, frag_A, frag_CB, frag_y);
}

// OUT.4 (swap): frag_y[d,t] += D·x[t,d].  Read x at the output (d,t) via the transpose view
// smem_x_trans[d,token] (== x[token,d]).  Scalar: adjacent frag elems are adjacent tokens (strided
// by out_stride_token / D_SMEM_COLS), so no float2.
template <typename input_t, int M_TILE, int N_TILE, typename FragY, typename SmemXTrans,
          typename ThrMma>
__device__ __forceinline__ void add_D_skip_swapped(FragY& frag_y, SmemXTrans const& smem_x_trans,
                                                   ThrMma const& thr_mma, float D_val, int n) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "swapped D_skip requires 2-byte input_t");
  if (D_val == 0.f) return;
  Tensor x_tile =
      local_tile(smem_x_trans, make_tile(Int<M_TILE>{}, Int<N_TILE>{}), make_coord(_0{}, n));
  Tensor x_part = thr_mma.partition_C(x_tile);
#pragma unroll
  for (int i = 0; i < size(frag_y); ++i) frag_y(i) += D_val * static_cast<float>(x_part(i));
}

// z-gate (swap): frag_y[d,t] *= z·sigmoid(z), z read at (d,t) via smem_z_trans[d,token].  Scalar.
template <typename input_t, int M_TILE, int N_TILE, typename FragY, typename SmemZTrans,
          typename ThrMma>
__device__ __forceinline__ void compute_z_gating_swapped(FragY& frag_y,
                                                         SmemZTrans const& smem_z_trans,
                                                         ThrMma const& thr_mma, void const* z_ptr,
                                                         int n) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "swapped z-gate requires 2-byte input_t");
  if (!z_ptr) return;
  Tensor z_tile =
      local_tile(smem_z_trans, make_tile(Int<M_TILE>{}, Int<N_TILE>{}), make_coord(_0{}, n));
  Tensor z_part = thr_mma.partition_C(z_tile);
#pragma unroll
  for (int i = 0; i < size(frag_y); ++i) {
    float const z = static_cast<float>(z_part(i));
    frag_y(i) *= z * __fdividef(1.f, (1.f + __expf(-z)));
  }
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, bool MUST_CHECKPOINT,
          typename FragCBNew, typename FragCBOld, typename SmemT>
__device__ __forceinline__ void output_head_2k(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int prev_k, int64_t out_seq_base,
                                               int write_offset, int seq_len, float D_val,
                                               int tile_buf, FragCBNew const& frag_CB_new,
                                               FragCBOld const& frag_CB_old) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "output_head_2k requires 2-byte input type");
  static_assert(sizeof(state_t) == 2, "operand-swap output requires 2-byte state");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int NPREDICTED_SWIZZLE_R = SmemT::NPREDICTED_SWIZZLE_R;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA (OPERAND SWAP): warps split M = DIM (16 rows/warp); N = t (NPRED); output [d, t].
  // NUM_M_WARPS = D_PER_CTA/16 warps run the output MMA (== NUM_WARPS at d_split=1; fewer at
  // d_split=2, where the surplus warps still do store_state/store_old_x but skip the output). ──
  constexpr int NUM_M_WARPS = D_PER_CTA / MMA_prop::M;
  static_assert(D_PER_CTA % MMA_prop::M == 0, "D_PER_CTA must be a multiple of MMA::M (16)");
  bool const m_active = (warp < NUM_M_WARPS);
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                  Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto thr_mma = tiled_mma.get_slice(tid);
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});  // 8 = one t N-tile
  constexpr int M_TILE = cute::tile_size<0>(decltype(tiled_mma){});  // = D_PER_CTA
  constexpr int NUM_N_TILES = NPREDICTED_PAD_MMA_N / N_TILE;         // 1 (mtp≤8) or 2 (mtp>8)
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2, "output_head_2k: NUM_N_TILES must be 1 or 2");

  // ── smem transpose views [d, token]: x is OUT.2's A-operand source + the D·x read; z the z-gate
  // read.  x/old_x are stored [token,d] (d contiguous) → transpose-LDSM as the A operand.  z is
  // aliased-stored; its physical NPREDICTED_SWIZZLE_R rows == make_swizzled_layout_rc for token<8.
  // ──
  auto const* x_base = smem.x + tile_buf * NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  auto layout_x_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(x_base)), layout_x_trans);
  auto const* z_base = smem.z + tile_buf * NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  auto layout_z_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_SWIZZLE_R, D_SMEM_COLS>();
  Tensor smem_z_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(z_base)), layout_z_trans);

  // ── x A-operand transpose LDSM (SM75_U16x8_LDSM_T: M=d 16 rows/warp, K=j 16) ──
  auto s2r_A_x = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A_x = s2r_A_x.get_slice(tid);

  // ── Gmem output base (token 0, d = head·DIM + d_tile·D_PER_CTA); M=d stride 1, N=t stride token
  // ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // ── Column (t) predicate identity; token = N so predicate on get<1> ──
  auto id_tile = make_identity_tensor(make_shape(Int<M_TILE>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  float const* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;

  if constexpr (MUST_CHECKPOINT) {
    // ── Write path: OUT.1 + store_state + (decay + CB@x + D·x + z-gate + scatter store) ──
    constexpr bool kSkipSmemToGmemState = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;

    auto epilogue = [&](auto& frag_y, int n) {
      // decay: frag_y[d,t] *= exp(cumAdt[t])  (t = N; broadcast over d = M) ──
      auto decay_bcast = make_tensor(
          make_smem_ptr(cumAdt_slot + n * N_TILE),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}), make_stride(_0{}, _1{})));
      auto decay_part = thr_mma.partition_C(decay_bcast);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= __expf(decay_part(i));
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new[n], smem_x_trans, s2r_A_x, s2r_thr_A_x, thr_mma, tiled_mma, n);
      add_D_skip_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_x_trans, thr_mma, D_val, n);
      compute_z_gating_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_z_trans, thr_mma, params.z, n);
      auto gOut = make_tensor(
          make_gmem_ptr(output_ptr + out_base + (int64_t)(n * N_TILE) * params.out_stride_token),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}),
                      make_stride(_1{}, params.out_stride_token)));
      auto gOut_part = thr_mma.partition_C(gOut);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) {
        int const t = n * N_TILE + get<1>(id_part(i));
        if (t < seq_len) gOut_part(i) = static_cast<input_t>(frag_y(i));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      if (m_active)  // only the D_PER_CTA/16 M-warps run the output MMA
        add_init_out_ring<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0, frag_y_1);
      if constexpr (!kSkipSmemToGmemState)  // store_state is cooperative over ALL warps
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      if (m_active) {
        epilogue(frag_y_0, 0);
        epilogue(frag_y_1, 1);
      }
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      if (m_active)  // only the D_PER_CTA/16 M-warps run the output MMA
        add_init_out_ring<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0);
      if constexpr (!kSkipSmemToGmemState)  // store_state is cooperative over ALL warps
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      if (m_active) epilogue(frag_y_0, 0);
    }
  } else {
    // ── No-write path: OUT.1 + (β·decay + CB@x + CB_old@old_x + D·x + z-gate + scatter store) ──
    using MmaAtomOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
    using LdsmAOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                        SM75_U16x4_LDSM_T>;
    auto tiled_mma_old =
        make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld>>{}, Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
    auto thr_mma_old = tiled_mma_old.get_slice(tid);

    auto const* old_x_base = smem.old_x + tile_buf * MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
    auto layout_old_x_trans =
        make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
    Tensor smem_old_x_trans =
        make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(old_x_base)),
                    layout_old_x_trans);
    // old_x A-operand transpose LDSM (K=i = MAX_WINDOW_PAD_MMA_K ∈ {8,16}).
    auto s2r_A_old_x = make_tiled_copy_A(Copy_Atom<LdsmAOld, MMA_prop::operand_t>{}, tiled_mma_old);
    auto s2r_thr_A_old_x = s2r_A_old_x.get_slice(tid);

    float const* old_cumAdt_slot = smem.old_cumAdt + tile_buf * MAX_WINDOW;
    float const total_old_cumAdt = (prev_k > 0) ? old_cumAdt_slot[prev_k - 1] : 0.f;
    float const beta_extra = __expf(total_old_cumAdt);

    auto epilogue = [&](auto& frag_y, int n) {
      auto decay_bcast = make_tensor(
          make_smem_ptr(cumAdt_slot + n * N_TILE),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}), make_stride(_0{}, _1{})));
      auto decay_part = thr_mma.partition_C(decay_bcast);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= beta_extra * __expf(decay_part(i));
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new[n], smem_x_trans, s2r_A_x, s2r_thr_A_x, thr_mma, tiled_mma, n);
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, MAX_WINDOW_PAD_MMA_K>(
          frag_y, frag_CB_old[n], smem_old_x_trans, s2r_A_old_x, s2r_thr_A_old_x, thr_mma_old,
          tiled_mma_old, n);
      add_D_skip_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_x_trans, thr_mma, D_val, n);
      compute_z_gating_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_z_trans, thr_mma, params.z, n);
      auto gOut = make_tensor(
          make_gmem_ptr(output_ptr + out_base + (int64_t)(n * N_TILE) * params.out_stride_token),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}),
                      make_stride(_1{}, params.out_stride_token)));
      auto gOut_part = thr_mma.partition_C(gOut);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) {
        int const t = n * N_TILE + get<1>(id_part(i));
        if (t < seq_len) gOut_part(i) = static_cast<input_t>(frag_y(i));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      if (m_active) {  // only the D_PER_CTA/16 M-warps run the output MMA + epilogue
        add_init_out_ring<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0, frag_y_1);
        epilogue(frag_y_0, 0);
        epilogue(frag_y_1, 1);
      }
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      if (m_active) {  // only the D_PER_CTA/16 M-warps run the output MMA + epilogue
        add_init_out_ring<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0);
        epilogue(frag_y_0, 0);
      }
    }
  }

  // store_old_x: 128-thread cooperative copy (16×8 layout), only first 4 warps participate.
  if (warp < 4)
    store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                     cache_slot, write_offset, seq_len, tile_buf);
}

// replay_state_mma_ring: main-local copy of ssu.cuh's replay_state_mma, but with old_B ALSO
// slot-indexed (by tile_buf) for the persistent ring — old_B is per-slot here, single-slot in
// the monolith.  The shared replay_state_mma is left untouched.  Only the old_B read changes.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void replay_state_mma_ring(SmemT& smem,
                                                      CheckpointingSsuParams const& params,
                                                      int warp, int lane, int prev_k, int d_tile,
                                                      int64_t state_ptr_offset,
                                                      state_t* state_w_base, int64_t rand_seed,
                                                      bool must_checkpoint, int tile_buf = 0) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma requires 2-byte input type");
  static_assert(D_PER_CTA % 16 == 0, "D_PER_CTA must be divisible by 16 (m16n8 atom)");
  static_assert(D_PER_CTA >= 16, "D_PER_CTA must be at least 16");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Atom K matches the cache-window tile (MAX_WINDOW_PAD_MMA_K).
  //   K == MMA_prop::K_BIG   (16) → m16n8k16 + x4/x2 ldmatrix.trans
  //   K == MMA_prop::K_SMALL (8)  → m16n8k8  + x2/x1 ldmatrix.trans
  using MmaAtomType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16,
                                         MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // Warp layout (M_WARPS, 4): always 4 warps along N=DSTATE; M_WARPS = NUM_WARPS/4 warps
  // split M=D_PER_CTA (each covers D_PER_CTA/M_WARPS/16 m-atoms + its own M-slice of the
  // A operand, cutting the redundant old_x LDSM by M_WARPS).  NUM_WARPS=4 → (1,4) = the
  // original _1x4 (byte-identical); 8 → (2,4); 16 → (4,4).
  // TODO(int8-8warp): the int8/fp8 amax (kernel_checkpointing_ssu_8bit.cuh) is warp-local
  // ONLY with full-N-per-warp (_W×1); this N-split layout would need a cross-warp amax
  // reduce.  bf16/fp16 are unaffected (deterministic state / order-free SR).
  constexpr int M_WARPS = NUM_WARPS / 4;
  static_assert(NUM_WARPS % 4 == 0, "replay_state_mma needs NUM_WARPS a multiple of 4");
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<Int<M_WARPS>, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Per-pass output tile is (D_PER_CTA/M_WARPS, N_PER_PASS).  N_PER_PASS = 4 warps × n8 = 32.
  constexpr int N_PER_PASS = 4 * MMA_prop::N;
  static_assert(DSTATE % N_PER_PASS == 0,
                "DSTATE must be divisible by 4 * MMA_prop::N for the (M_WARPS,4) warp layout");
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;

  // tile_buf selects the pipeline slot for all per-head smem arrays (old_cumAdt, old_dt, old_x).
  constexpr int MAX_WINDOW = SmemT::MAX_WINDOW;
  float const* old_cumAdt_slot = smem.old_cumAdt + tile_buf * MAX_WINDOW;
  float const* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  float total_cumAdt = (prev_k > 0) ? old_cumAdt_slot[prev_k - 1] : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] Swizzle<3,3,3>, transposed
  // view [M=D_SMEM_COLS, K=MAX_WINDOW_PAD_MMA_K].  D_SMEM_COLS may be padded above
  // D_PER_CTA when D_PER_CTA < swizzle atom; local_tile to D_PER_CTA
  // restricts the LDSM to the valid sub-tile.  Each warp loads the FULL M (4×
  // redundant across warps).  See header comment for traffic accounting. ──
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  auto const* old_x_slot = smem.old_x + tile_buf * MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(old_x_slot)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == MMA_prop::operand_t (bf16) — no conversion needed.

  // ── B operand: old_B [MAX_WINDOW_PAD_MMA_K, DSTATE] swizzled, transposed view
  // [N=DSTATE, K=MAX_WINDOW_PAD_MMA_K].  Per pass loads N_PER_PASS=32 cols across
  // 4 warps; partition_S splits — each warp gets its disjoint 8-col slice. ──
  auto layout_B = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(
                                       smem.old_B + tile_buf * SmemT::OLD_B_ELEMS)),
                                   layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: per-CTA swizzle layout [D_PER_CTA, DSTATE].  tile_buf selects the
  // double-buffered slot (cross-head prefetch); 0 ⇒ the single original buffer. ──
  auto layout_state_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  state_t* state_base = reinterpret_cast<state_t*>(smem.state) + tile_buf * D_PER_CTA * DSTATE;

  // ── Per-pass identity for (row, col) coords ──
  // partition_C of an identity tensor of the per-pass output shape gives this
  // thread's (row, col) at every C-frag position, including warp-N offset.
  // Frag size per thread = (M_atoms=D_PER_CTA/16) × (N_atoms_per_warp=1) × 4 elts.
  auto id_tile = make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  // Linear order from CuTe's column-major partition_C with m16n8 atom:
  //   i=0,1: same row (= row_lo of M-atom 0), adjacent cols (col_off, col_off+1)
  //   i=2,3: same row (= row_hi of M-atom 0), adjacent cols
  //   i=4,5: same row (= row_lo of M-atom 1)
  //   ... (V index 0..3 inside each m16n8, then M-atoms in M-major order)
  // Pair load at (i, i+1) covers two consecutive bf16 elts → one 32-bit LDS.

  // Precompute dB coefficients once — depend only on K (lane), not on N.
  constexpr int LANES_PER_N_COL = warpSize / MMA_prop::N;  // = 4 for m16n8k_
  constexpr int DB_COEFFS_PER_LANE = MAX_WINDOW_PAD_MMA_K / LANES_PER_N_COL;
  float dB_coeff[DB_COEFFS_PER_LANE];
  precompute_dB_coeff<DB_COEFFS_PER_LANE>(dB_coeff, old_cumAdt_slot, old_dt_slot, total_cumAdt,
                                          prev_k, lane);

  using pair_t = Pair<state_t>;

  // Philox state amortized across 4 consecutive pair conversions: each call
  // returns 4 randints, all 4 get consumed before the next refresh (vs. 1-of-4
  // in the Triton-bit-equal layout — see writeback loop below).  Compile-time
  // pair_idx (n-loop and i-loop both unrolled) keeps `rand_idx[pair_idx & 3]`
  // as a known register access — no local-memory spill.
  constexpr bool kPhiloxF16 = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;
  [[maybe_unused]] uint32_t rand_idx[4];
  // state_w_base is the pre-combined (params.state + state_gmem_off) base
  // pointer — see the function header.  No separate state_w / state_gmem_off
  // alive in this scope.

  // ── Vectorized state writeback (cross-pass STG.64 fusion) ──────────
  // smem always gets nearest-even f32→state_t (consumed by matmul 3 — must
  // match Triton's f32→bf16 path as closely as possible).  Gmem cache, when
  // PHILOX_ROUNDS > 0 and state_t == __half, gets PTX cvt.rs.f16x2.f32
  // stochastic rounding direct from registers via cross-pass STG.64; the
  // smem→gmem `store_state` is gated off in compute_and_store_output.
  //
  // Cross-pass STG fusion: do PASS n0 and PASS n1 back-to-back, buffering
  // the post-cvt_rs packed u32s of n0 across n1's HMMA + cvt_rs.  Then issue
  // ONE STG.64 instruction per pair iter, all 32 lanes active:
  //   - even lane stores PASS n0 data at the warp's n0 column slice
  //   - odd  lane stores PASS n1 data at the warp's n1 column slice
  // Halves the STG instruction count vs per-pass writeback (16 STG.64/thread
  // per 2 passes vs 16 + 16 = 32 STG.64/thread previously — same byte volume).
  //
  // Randint amortization: rand_idx[4] refreshed every 4 pairs; each pair's
  // cvt_rs uses one of the 4 randints.  Triton bit-equality is intentionally
  // given up; unbiasedness still holds.
  // Per-warp M-rows = D_PER_CTA / M_WARPS, so pairs/pass = (D_PER_CTA/M_WARPS)/8.  (NUM_WARPS=4
  // → M_WARPS=1 → D_PER_CTA/8, the original.)  Keeps the philox my_packed buffer + STG fusion
  // sized to this warp's actual fragment under the (M_WARPS,4) split.
  constexpr int PAIRS_PER_PASS = (D_PER_CTA / M_WARPS) / 8;  // = (per-warp M-atoms) × 2 row-pairs
  static_assert(NUM_N_PASSES % 2 == 0, "Cross-pass STG fusion requires even NUM_N_PASSES");

#pragma unroll
  for (int np = 0; np < NUM_N_PASSES; np += 2) {
    // Buffer of post-cvt_rs packed u32s for both passes (philox path only).
    [[maybe_unused]] uint32_t my_packed[2][PAIRS_PER_PASS];

#pragma unroll
    for (int local_n = 0; local_n < 2; ++local_n) {
      int const n = np + local_n;
      int const n_base = n * N_PER_PASS;

      // ── Allocate per-pass C-frag (4 × M_atoms fp32 elts/thread) ──
      Tensor frag_h = thr_mma.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

      // ── Load state × total_decay into frag_h. ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);
        pair_t const p = *reinterpret_cast<pair_t const*>(&state_base[off]);
        frag_h(i) = toFloat(p[cute::Int<0>{}]) * total_decay;
        frag_h(i + 1) = toFloat(p[cute::Int<1>{}]) * total_decay;
      }

      // ── LDSM.T per-pass B (per warp = 1 atom of 8 cols of N) ──
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);

      Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_view = s2r_thr_B.retile_D(frag_B);

      cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

      compute_dB_scaling<DB_COEFFS_PER_LANE>(frag_B, dB_coeff);

      // ── HMMA: frag_h += frag_A @ frag_B ──
      cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

      // ── Smem write (always) + cvt_rs into my_packed (philox path) ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);

        // Smem write — always nearest-even (output's matmul 3 reads this).
        pair_t const q = pack_float2<state_t>(make_float2(frag_h(i), frag_h(i + 1)));
        *reinterpret_cast<pair_t*>(&state_base[off]) = q;

        if constexpr (kPhiloxF16) {
          static_assert(sizeof(state_t) == 2, "STG.64 cooperative path requires 2-byte state_t");
          int const pair_idx = n * PAIRS_PER_PASS + i / 2;
          // Per-lane philox_off is unique per (thread, refresh group) — each
          // pair gets its own randint bits.  Always computed; only consumed
          // by the refresh branch inside the helper.
          int64_t const philox_off =
              state_ptr_offset + (int64_t)(d_tile * D_PER_CTA + row) * DSTATE + col;
          // Buffer the SR'd packed u32 — store happens after BOTH passes.
          my_packed[local_n][i / 2] = stochastic_round_pair_with_philox_refresh<PHILOX_ROUNDS>(
              frag_h(i), frag_h(i + 1), pair_idx, rand_seed, philox_off, rand_idx);
        }
      }
    }

    // ── Cross-pass STG.64: all 32 lanes active. ─────────────────────────
    // m16n8 lane layout: lane k → row k/4, cols (k%4)*2..(k%4)*2+1.  Lanes
    // (2k, 2k+1) hold adjacent col-pairs of the same row.  After shfl_xor,
    // the even/odd lane each has a 4-col contiguous block (in different
    // bit-orders).  Even lane STG.64s the n0-pass block at its own col
    // base; odd lane STG.64s the n1-pass block at the peer's (lower) col
    // — both 8-byte aligned for state_t = f16.
    // Runtime-gated on must_checkpoint: non-checkpoint steps skip the gmem
    // STGs entirely (state HBM remains the prior checkpoint).  The cvt_rs
    // SR + philox refresh above still ran — only the STGs are elided —
    // because skipping them would require routing must_checkpoint into the
    // pair_idx amortization logic, which lives across the n-loop.
    if constexpr (kPhiloxF16) {
      if (must_checkpoint) {
        exchange_ntile_state_store_global<PAIRS_PER_PASS, N_PER_PASS, DSTATE>(
            state_w_base, np, lane, my_packed, id_part);
      }
    }
  }
}

// HeadMetaSSU: the per-work-unit ring entry — the grid-stride tile index, its resolved cache slot,
// and prev_k.  cache_slot = sbi[seq] heads the dependent chain (cache_slot → prev_num_accepted /
// cache_buf_idx[cache_slot]); resolving cache_slot AND prev_k together at fetch (prefetched STAGES
// ahead) means the consumer reads prev_k from a register, never reloading prev_num_accepted.
// prev_k was the #1 long_scoreboard site when re-fetched on the fly (mc_of + derive_head reload it
// 3–5×/unit); storing it costs +1 int/entry and keeps the ring register-resident.  buf_read stays a
// single load off cache_slot in derive_head (write-path only).  cache_slot == pad_slot_id ⇒ skip.
struct HeadMetaSSU {
  int tile{-1};
  int prev_k{0};
  int64_t cache_slot{-1};
  float D_val{
      0.f};  // params.D[head] skip coeff — the one un-prefetched output input; resolve at fetch
};

// derive_head: expand the ring entry {tile, cache_slot, prev_k} into the working scalars, ON THE
// FLY at each use (no held struct).  Tile-derived (d_tile/seq/head/group) are constexpr divides;
// prev_k comes straight from the entry (resolved at fetch); buf_read is one load off the
// already-resolved cache_slot (write-path only); seq_len stays the NPREDICTED constexpr
// (non-varlen).  __forceinline__ so unused outputs fold away per call site.
template <int NHEADS, int HEADS_PER_GROUP, int D_SPLIT, int NPREDICTED, bool VARLEN>
__device__ __forceinline__ void derive_head(CheckpointingSsuParams const& params,
                                            HeadMetaSSU const& m, int& d_tile, int& seq,
                                            int& first_head, int& group_idx, int& buf_read,
                                            int& prev_k, int& seq_len, int64_t& outer) {
  first_head = m.tile % NHEADS;  // compile-time NHEADS divisor
  int const t = m.tile / NHEADS;
  d_tile = t % D_SPLIT;                      // compile-time D_SPLIT divisor
  seq = t / D_SPLIT;                         // batch is the range, never a divisor
  group_idx = first_head / HEADS_PER_GROUP;  // compile-time HEADS_PER_GROUP divisor
  buf_read = __ldg(reinterpret_cast<int32_t const*>(params.cache_buf_idx) + m.cache_slot);
  prev_k = m.prev_k;  // resolved at fetch (prefetched STAGES ahead), not reloaded here
  if constexpr (VARLEN) {
    auto const* __restrict__ cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);
    int const bos = __ldg(&cu[seq]);
    seq_len = __ldg(&cu[seq + 1]) - bos;
    outer = (int64_t)bos;
  } else {
    seq_len = NPREDICTED;  // keep the compile-time constant
    outer = (int64_t)seq;
  }
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, int NHEADS, int HEADS_PER_GROUP, bool VARLEN,
          typename SmemT>
__device__ __forceinline__ void prefetch_async_pre_gdc(SmemT& smem,
                                                       CheckpointingSsuParams const& params,
                                                       int lane, int warp, HeadMetaSSU const& m,
                                                       int slot, bool must_checkpoint) {
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  int d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len;
  int64_t outer;
  derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(
      params, m, d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len, outer);
  prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, lane, warp, d_tile,
                                                             first_head, m.cache_slot, slot);
  load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
            /*IS_FIRST=*/true>(smem, params, lane, warp, d_tile, first_head, group_idx,
                               m.cache_slot, buf_read, outer, seq_len, must_checkpoint, prev_k,
                               slot);
}

template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, typename SmemT>
__device__ __forceinline__ void prefetch_async_post_gdc(SmemT& smem,
                                                        CheckpointingSsuParams const& params,
                                                        int lane, int warp, HeadMetaSSU const& m,
                                                        int slot, bool must_checkpoint) {
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  int d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len;
  int64_t outer;
  derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(
      params, m, d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len, outer);
  if (warp == 0) {
    load_cumAdt_async(smem, params, lane, seq, first_head, seq_len, slot);
  } else if (warp == 1 || warp == 2) {
    load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/true>(
        smem, params, lane, warp, d_tile, first_head, group_idx, outer, seq_len, slot);
  } else if (warp == 3) {
    load_cb_async<input_t>(smem, params, lane, warp, seq, first_head, must_checkpoint, slot);
  }
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, int NHEADS, int HEADS_PER_GROUP, bool VARLEN,
          typename SmemT>
__device__ __forceinline__ void prefetch_async(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, HeadMetaSSU const& m, int slot,
                                               bool must_checkpoint) {
  prefetch_async_pre_gdc<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                         NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, m,
                                                                     slot, must_checkpoint);
  prefetch_async_post_gdc<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NHEADS, HEADS_PER_GROUP,
                          VARLEN>(smem, params, lane, warp, m, slot, must_checkpoint);
}

// replay_state: write-path state-checkpoint replay for one work-unit, reading its ring slot `slot`.
// The bundle (state + old-tiles) must already be drained and published cross-warp by the caller.
template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, typename SmemT>
__device__ __forceinline__ void replay_state(SmemT& smem, CheckpointingSsuParams const& params,
                                             int lane, int warp, HeadMetaSSU const& m, int slot) {
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  int d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len;
  int64_t outer;
  derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(
      params, m, d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len, outer);
  int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  int64_t const state_ptr_offset =
      m.cache_slot * params.state_stride_seq + (int64_t)first_head * DIM * DSTATE;
  state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                (int64_t)d_tile * D_PER_CTA * DSTATE;
  replay_state_mma_ring<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
      smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
      /*must_checkpoint=*/true, slot);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, bool MUST_CHECKPOINT, typename SmemT>
__device__ __forceinline__ void compute_output_and_store(SmemT& smem,
                                                         CheckpointingSsuParams const& params,
                                                         int lane, int warp, HeadMetaSSU const& m,
                                                         int slot) {
  using namespace cute;
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  int d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len;
  int64_t outer;
  derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(
      params, m, d_tile, seq, first_head, group_idx, buf_read, prev_k, seq_len, outer);
  float const D_val = m.D_val;  // resolved at fetch (prefetched STAGES ahead), not reloaded here
  int const tid = warp * warpSize + lane;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int NUM_OUT_NTILES = NPREDICTED_PAD_MMA_N / MMA_prop::N;  // 1 (mtp≤8) or 2 (mtp>8)
  constexpr int CB_NEW_REGS_B = SmemT::CB_NEW_REGS_B;  // NUM_OUT_NTILES · (K/4) total
  constexpr int CB_OLD_REGS_B = SmemT::CB_OLD_REGS_B;
  constexpr int CB_NEW_REGS_PER =
      CB_NEW_REGS_B / NUM_OUT_NTILES;  // one output N-tile's B-frag = K/4
  constexpr int CB_OLD_REGS_PER = CB_OLD_REGS_B / NUM_OUT_NTILES;
  using MmaAtomOld_t = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
  // Operand swap: CB is the B-operand, warps split M=DIM (Shape<NUM_M_WARPS,1>, matching
  // output_head_2k).  ONE m16n8 N-atom per output N-tile → a partition_fragment_B ARRAY (a single
  // multi-N-tile partition_fragment_B trips make_fragment_like at mtp>8).  The precompute stored
  // REGS_B = NUM_OUT_NTILES·(K/4) regs/lane g-major (g = out_ntile·REGS_PER + r) — one vectorized
  // LDS, then distribute to the per-tile B-frags.
  constexpr int NUM_M_WARPS = D_PER_CTA / MMA_prop::M;
  auto tiled_mma_cb = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                     Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto tiled_mma_old_cb =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld_t>>{}, Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto thr_mma_cb = tiled_mma_cb.get_slice(tid);
  auto thr_mma_old_cb = tiled_mma_old_cb.get_slice(tid);
  using FragCBNewT = decltype(thr_mma_cb.partition_fragment_B(make_tensor(
      (MMA_prop::operand_t*)nullptr, make_shape(Int<MMA_prop::N>{}, Int<NPREDICTED_PAD_MMA_M>{}))));
  using FragCBOldT = decltype(thr_mma_old_cb.partition_fragment_B(make_tensor(
      (MMA_prop::operand_t*)nullptr, make_shape(Int<MMA_prop::N>{}, Int<MAX_WINDOW_PAD_MMA_K>{}))));
  FragCBNewT frag_CB_new[NUM_OUT_NTILES];
  FragCBOldT frag_CB_old[NUM_OUT_NTILES];
  // The fragment element is the MMA operand type (cutlass::bfloat16_t), bit-compatible with input_t
  // but a distinct C++ type — read the gmem/smem bytes AS that type so the assign is well-typed.
  using cb_new_frag_t = cute::remove_cvref_t<decltype(frag_CB_new[0](0))>;
  using cb_old_frag_t = cute::remove_cvref_t<decltype(frag_CB_old[0](0))>;
  {
    input_t const* cb_new_s = smem.cb_new + slot * SmemT::CB_NEW_ELEMS;
    auto const raw =
        reinterpret_cast<PackedAligned<cb_new_frag_t, CB_NEW_REGS_B> const*>(cb_new_s)[lane];
#pragma unroll
    for (int on = 0; on < NUM_OUT_NTILES; ++on)
#pragma unroll
      for (int r = 0; r < CB_NEW_REGS_PER; ++r)
        frag_CB_new[on](r) = raw.val[on * CB_NEW_REGS_PER + r];
  }
  if constexpr (!MUST_CHECKPOINT) {
    input_t const* cb_old_s = smem.cb_old + slot * SmemT::CB_OLD_ELEMS;
    auto const raw =
        reinterpret_cast<PackedAligned<cb_old_frag_t, CB_OLD_REGS_B> const*>(cb_old_s)[lane];
#pragma unroll
    for (int on = 0; on < NUM_OUT_NTILES; ++on)
#pragma unroll
      for (int r = 0; r < CB_OLD_REGS_PER; ++r)
        frag_CB_old[on](r) = raw.val[on * CB_OLD_REGS_PER + r];
  }
  int64_t const out_seq_base = outer * params.out_stride_seq;
  int const write_offset = MUST_CHECKPOINT ? 0 : prev_k;
  output_head_2k<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
                 NUM_WARPS, MUST_CHECKPOINT>(smem, params, lane, warp, d_tile, first_head,
                                             m.cache_slot, prev_k, out_seq_base, write_offset,
                                             seq_len, D_val, slot, frag_CB_new, frag_CB_old);
}

// process_head: PURE CONSUMER of a prefetched ring slot — replay (write path) then output.  The
// slot's bundle is already drained by the caller's pipeline_wait; this issues NO cp.async (only the
// on-the-fly scalar derives + cb LDGs).  The barriers publish the drained slot cross-warp.
template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, bool MUST_CHECKPOINT, typename SmemT>
__device__ __forceinline__ void process_head(SmemT& smem, CheckpointingSsuParams const& params,
                                             int lane, int warp, HeadMetaSSU const& m, int slot) {
  if constexpr (MUST_CHECKPOINT) {
    replay_state<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane,
                                                                            warp, m, slot);
    __syncthreads();  // publish the in-place-replayed state cross-warp before the output MMA
  }
  compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                           DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                           MUST_CHECKPOINT>(smem, params, lane, warp, m, slot);
}

// =============================================================================
// Main kernel.  Template params mirror checkpointing_ssu_kernel so the launcher
// dispatches both with the same args.
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1,
          bool VARLEN = false, int NGROUPS = 1>
// __maxnreg__ tracks the ring depth, because the two regimes are opposite:
//   STATE_PIPE=1: ≈19 KB smem allows >8 blocks/SM, so the kernel is REGISTER-bound — cap at 64 to
//     hold 8 blocks (65536/(64·128)=8).
//   STATE_PIPE≥2: SMEM-bound (≈38 KB → 5 blocks at depth 2).  The old 96-cap forced the compiler to
//     recompute per-work-unit addressing it could otherwise hold in registers.  Lifting to 128 →
//     4 blocks/SM (118 regs, no spills) lets it hold the addressing: per-work-unit recompute drops
//     (total executed −6.5%, nw0/nw8 −~6 µs).  The Triton regime (more regs, fewer blocks).  Fully
//     uncapping (255) is worse: the compiler grabs 150 regs → 3 blocks → +20 µs.  Tune via the
//     macro.
#ifndef SSU_MAIN_MAXNREG_PIPE
#define SSU_MAIN_MAXNREG_PIPE 128
#endif
__global__ __maxnreg__(
    MAIN_STATE_PIPE == 1
        ? 64
        : SSU_MAIN_MAXNREG_PIPE) void checkpointing_ssu_main_kernel(CheckpointingSsuParams params) {
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32, "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 layout)");
  static_assert(NPREDICTED <= MAX_WINDOW, "NPREDICTED must be <= MAX_WINDOW");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG, "MAX_WINDOW must be <= MMA::K_BIG=16");
  assert(params.d_split == D_SPLIT);

  // ── EXTERNAL PDL: signal a programmatic DOWNSTREAM kernel that `output` is fully written
  // (ALL work-units done; the per-head cache writes are next-step-only).  Gated by ENABLE_PDL.
  // (The internal precompute→main chain uses the one-shot gdc_wait on the first work-unit.) ──
  if constexpr (ENABLE_PDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }

  using SmemT = CheckpointingSsuMainStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA,
                                            DSTATE, MAIN_STATE_PIPE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── 1D grid-stride persistent loop over single-head work-units ──
  // Work-unit = (d_tile, seq, head); the launcher sizes the grid to
  // min(cta_per_sm·NUM_SMS, total_work) and each CTA grid-strides over work-units.  Default
  // cta_per_sm = occupancy ⇒ grid = the resident set, each CTA strides; bigger ⇒ grid==total_work.
  // Flatten head-fastest (head innermost) so consecutive work-units are consecutive heads of one
  // (d_tile, seq) → per-group C/old_B stay L2-hot.
  //
  // NHEADS = NGROUPS·HEADS_PER_GROUP is COMPILE-TIME, so the unflatten divides only by compile-time
  // constants (NHEADS, D_SPLIT, HEADS_PER_GROUP) — no runtime div/mod.  seq is the top quotient, so
  // `batch` is never a divisor; gridDim.x is the loop stride, never a divisor.
  constexpr int NHEADS = NGROUPS * HEADS_PER_GROUP;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const batch = (int)params.batch;
  int const total_work = D_SPLIT * batch * NHEADS;
  int const stride = (int)gridDim.x;

  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  auto const* __restrict__ cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);

  // fetch: resolve the ring entry {tile, cache_slot, prev_k}.  cache_slot = sbi[seq] heads the
  // dependent chain (cache_slot → prev_num_accepted[cache_slot]); resolving BOTH here, STAGES ahead
  // (the meta is prefetched), means the consumer reads prev_k from a register — the chain latency
  // is overlapped at prefetch instead of stalling process (prev_k was the #1 long_scoreboard site
  // when reloaded on the fly).  prev_k is loaded only for a real slot (pad ⇒ skipped, so prev_k
  // unused). buf_read/seq_len stay derived on the fly at use (see derive_head).
  auto fetch = [&](int tile) -> HeadMetaSSU {
    int const seq = (tile / NHEADS) / D_SPLIT;  // compile-time NHEADS, D_SPLIT divisors
    HeadMetaSSU m;
    m.tile = tile;
    m.cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
    m.prev_k = (m.cache_slot != params.pad_slot_id) ? prev_ptr[m.cache_slot] : 0;
    m.D_val = D_ptr ? toFloat(D_ptr[tile % NHEADS]) : 0.f;  // D skip-coeff, prefetched STAGES ahead
    return m;
  };

  // ── CROSS-ITERATION N-STAGE RING ──
  // head_meta[] is a register-resident SHIFT-REGISTER of the trimmed {tile, cache_slot}, prefetched
  // STAGES ahead so cache_slot is resolved before it's consumed.  CRITICAL: it is indexed ONLY by
  // compile-time-constant subscripts (head_meta[0], head_meta[k] inside #pragma unroll,
  // head_meta[STAGES-1]) so SROA keeps it in registers.  A runtime subscript (the old meta[j %
  // STAGES]) would force the whole array to LOCAL memory — then "prefetched cache_slot" is read
  // back from local at ~the same latency as the global load it replaced (measured: the #1
  // long_scoreboard site + 3.3 MB local traffic).  The runtime smem-buffer index `slot = j %
  // STAGES` is kept SEPARATE and used only to address the smem ring (a runtime offset into shared
  // memory is fine; it is not local memory).
  //
  // Invariant at the top of steady iteration j: head_meta[k] == meta(j + k) for k = 0..STAGES-1, so
  // head_meta[0] is always the unit to process now.  Each iteration: process head_meta[0] from smem
  // buffer `slot`; shift the queue left (constant indices); append meta(j+STAGES) into
  // head_meta[STAGES-1] and prefetch it into the freed buffer `slot` (since (j+STAGES) % STAGES ==
  // j % STAGES).  The physical smem layout (unit u in buffer u % STAGES) and the cp.async FIFO
  // depth are UNCHANGED, so the result is bit-identical. tile 0 is split pre/post-gdc (single
  // scalar m0) so its 16 KB state load overlaps the precompute and its replay runs in the gdc
  // window.  Every work-unit is exactly one cp.async group (EMPTY for a pad / past-the-end), so
  // wait_prior(STAGES-1) drains exactly the work-unit being processed.
  constexpr int STAGES = MAIN_STATE_PIPE;
  HeadMetaSSU head_meta[STAGES];
  auto fetch_head_meta = [&](int i) -> HeadMetaSSU {
    int const wu = blockIdx.x + i * stride;
    if (wu >= total_work) {
      HeadMetaSSU m;
      m.tile = wu;
      m.cache_slot = params.pad_slot_id;  // sentinel ⇒ skip (past-the-end)
      m.prev_k = 0;                       // unused (skipped); keep deterministic
      m.D_val = 0.f;
      return m;
    }
    return fetch(wu);
  };
  // valid / must_checkpoint from the register-resident entry — non-varlen reads only registers
  // (cache_slot for validity, prev_k for mc); varlen also touches cu_seqlens for the row length.
  auto is_valid = [&](HeadMetaSSU const& m) -> bool {
    if (m.cache_slot == params.pad_slot_id) return false;
    if constexpr (VARLEN) {
      int const seq = (m.tile / NHEADS) / D_SPLIT;
      return __ldg(&cu[seq + 1]) - __ldg(&cu[seq]) > 0;  // empty varlen row ⇒ skip
    }
    return true;
  };
  auto mc_of = [&](HeadMetaSSU const& m) -> bool {
    if constexpr (VARLEN) {
      int const seq = (m.tile / NHEADS) / D_SPLIT;
      return m.prev_k + (__ldg(&cu[seq + 1]) - __ldg(&cu[seq])) > MAX_WINDOW;
    }
    return m.prev_k + NPREDICTED > MAX_WINDOW;
  };

  // ── PROLOGUE: tile 0 (smem buffer 0), pre-gdc half first to overlap the precompute. ──
  HeadMetaSSU const m0 =
      fetch_head_meta(0);  // wu(0)=blockIdx.x < total_work (CTA has ≥1 work-unit)
  bool const v0 = is_valid(m0);
  bool const mc0 = v0 && mc_of(m0);
  if (v0)
    prefetch_async_pre_gdc<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                           NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, m0,
                                                                       /*slot=*/0, mc0);
  __pipeline_commit();  // G_pre0 (empty if tile 0 is a pad)
  // Resolve units 1..STAGES-1's cache_slot ahead (overlaps tile-0's pre-gdc load); unit p →
  // head_meta[p-1].
#pragma unroll
  for (int p = 1; p < STAGES; ++p) head_meta[p - 1] = fetch_head_meta(p);

  __pipeline_wait_prior(0);  // tile-0 pre-gdc bundle ready for replay
  __syncthreads();           // publish it cross-warp
  if (mc0)
    replay_state<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane,
                                                                            warp, m0, 0);
  __syncthreads();                  // publish replayed state + gdc convergence
  cudaGridDependencySynchronize();  // gdc ONCE — precompute outputs (cb / cumAdt / x / C) now
                                    // visible

  if (v0)
    prefetch_async_post_gdc<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NHEADS, HEADS_PER_GROUP,
                            VARLEN>(smem, params, lane, warp, m0, 0, mc0);
  __pipeline_commit();  // G_post0

  // Prefetch units 1..STAGES-1 (full bundles; gdc already fired) — unit p into smem buffer p.
#pragma unroll
  for (int p = 1; p < STAGES; ++p) {
    if (is_valid(head_meta[p - 1]))
      prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                     NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, head_meta[p - 1],
                                                      /*slot=*/p, mc_of(head_meta[p - 1]));
    __pipeline_commit();  // G_full_p (empty if pad / past-the-end)
  }

  // FIFO: [G_post0, G_full_1 .. G_full_{STAGES-1}] = STAGES groups.  Drain + output tile 0.
  __pipeline_wait_prior(STAGES - 1);
  __syncthreads();  // publish tile-0 post-gdc data (C / x / cumAdt) cross-warp before its output
  if (v0) {
    if (mc0)
      compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                               DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                               /*MUST_CHECKPOINT=*/true>(smem, params, lane, warp, m0, 0);
    else
      compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                               DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                               /*MUST_CHECKPOINT=*/false>(smem, params, lane, warp, m0, 0);
  }
  __syncthreads();  // tile-0 output done reading buffer 0 before the tile-STAGES prefetch
                    // overwrites it

  // Prefetch unit STAGES into buffer 0 (freed by tile-0's output) → head_meta[STAGES-1].  Now the
  // queue holds units 1..STAGES (head_meta[k] == meta(k+1)), establishing the steady invariant for
  // j == 1.
  head_meta[STAGES - 1] = fetch_head_meta(STAGES);
  if (is_valid(head_meta[STAGES - 1]))
    prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                   NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, head_meta[STAGES - 1],
                                                    /*slot=*/0, mc_of(head_meta[STAGES - 1]));
  __pipeline_commit();

  // ── STEADY STATE: process head_meta[0] (== unit j, prefetched STAGES ago, cache_slot in a
  // register), shift
  //    the queue, append + prefetch unit (j+STAGES) into the freed buffer `slot`. ──
  for (int j = 1; blockIdx.x + j * stride < total_work; ++j) {
    int const slot = j % STAGES;         // smem buffer of unit j (== buffer of unit j+STAGES)
    HeadMetaSSU const m = head_meta[0];  // register-resident: unit j, cache_slot already resolved
    bool const v = is_valid(m);
    __pipeline_wait_prior(STAGES - 1);  // drain unit j's bundle (the oldest group)
    if (v) {
      __syncthreads();  // publish the drained bundle (state + old-tiles + C/x/cumAdt) cross-warp
      if (mc_of(m))
        process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                     /*MUST_CHECKPOINT=*/true>(smem, params, lane, warp, m, slot);
      else
        process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                     /*MUST_CHECKPOINT=*/false>(smem, params, lane, warp, m, slot);
    }
    __syncthreads();  // output done reading `slot` before the prefetch below overwrites it

    // Shift the queue left (constant indices → stays in registers), append the next unit.
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) head_meta[k] = head_meta[k + 1];
    head_meta[STAGES - 1] = fetch_head_meta(j + STAGES);

    if (is_valid(head_meta[STAGES - 1]))
      prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                     NHEADS, HEADS_PER_GROUP, VARLEN>(
          smem, params, lane, warp, head_meta[STAGES - 1], slot, mc_of(head_meta[STAGES - 1]));
    __pipeline_commit();  // keep the FIFO at STAGES groups (empty for pad / past-the-end)
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
