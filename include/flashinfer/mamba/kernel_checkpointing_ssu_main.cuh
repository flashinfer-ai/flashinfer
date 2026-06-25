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
          int DSTATE, int STATE_PIPE = 1, int MHC = STATE_PIPE>
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
  float cumAdt[MHC * NPREDICTED_PAD_MMA_M];  // one slot per CTA head (not per pipeline stage)
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
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const x_base = outer * params.x_stride_seq + (int64_t)head * DIM + d_tile_off;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // C is per-GROUP → load once per head-tile (IS_FIRST only), on W1.
  if constexpr (IS_FIRST)
    if (warp == 1)
      load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane,
                                          seq_len);
  // x is per-HEAD → every head, single-warp on W2 (W0-1 carry old_x, W3 carries z).
  auto* x_slot = smem.x + tile_buf * SmemT::X_ELEMS;
  if (warp == 2)
    load_tile_async<XShape, NPREDICTED>(x_slot, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
  // NOTE: NO commit/wait — head_loop owns the pipeline.
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

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, bool MUST_CHECKPOINT,
          typename FragCBNew, typename FragCBOld, typename SmemT>
__device__ __forceinline__ void output_head_2k(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int prev_k, int64_t out_seq_base,
                                               int write_offset, int seq_len, float D_val,
                                               int tile_buf, int cumAdt_h,
                                               FragCBNew const& frag_CB_new,
                                               FragCBOld const& frag_CB_old) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "output_head_2k requires 2-byte input type");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA for matmul-3 + matmul-4-new (K = NPREDICTED_PAD_MMA_M = 16) ──
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                  Layout<Shape<_1, Int<NUM_WARPS>>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  auto const* x_base = smem.x + tile_buf * NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(x_base)),
                              layout_x_swz);
  auto layout_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(x_base)), layout_x_trans_swz);

  auto layout_z_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS, NPREDICTED>();
  auto const* z_base = smem.z + tile_buf * SmemT::NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  Tensor smem_z =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(z_base)), layout_z_swz);

  // ── S2R copies (matmul-4-new: CB A-operand and x B-operand) ──
  // s2r_A is dead code on the two-kernel path (CB lives in registers, not smem.CB_scaled)
  // but declaring it alongside s2r_B_trans preserves the code-gen pattern from
  // compute_no_write_output<READ_PRECOMPUTED_CB=true>, which avoids bank conflicts on
  // the s2r_B_old_trans LDSM path.
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B_trans =
      make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B_trans = s2r_B_trans.get_slice(tid);

  // ── Decay broadcast: cumAdt[t] with stride-0 on N ──
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  float const* cumAdt_slot = smem.cumAdt + cumAdt_h * NPREDICTED_PAD_MMA_M;
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(cumAdt_slot),
      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output base ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // ── Row predicates (2 unique rows in m16n8k16 C-frag per thread) ──
  auto id_tile = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < seq_len;
  bool const pred_row_hi = get<0>(id_part(2)) < seq_len;

  constexpr int NUM_N_TILES = D_PER_CTA / N_TILE;
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2,
                "output_head_2k: NUM_N_TILES = D_PER_CTA / N_TILE must be 1 or 2");

  if constexpr (MUST_CHECKPOINT) {
    // ── Write path: matmul-3 + store_state + (decay + CB@x + D*x + z-gate + store) ──
    constexpr bool kSkipSmemToGmemState = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;

    auto epilogue = [&](auto& frag_y, int n) {
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= __expf(decay_part(i));
      add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);
      add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);
      compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);
      auto gOut_tile =
          make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                  make_stride(params.out_stride_token, _1{})));
      auto gOut_part = thr_mma.partition_C(gOut_tile);
#pragma unroll
      for (int i = 0; i < size(frag_y); i += 2) {
        bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
        if (pred_i)
          *reinterpret_cast<Pair<input_t>*>(&gOut_part(i)) =
              pack_float2<input_t>(make_float2(frag_y(i), frag_y(i + 1)));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, tile_buf,
                                                        frag_y_0, frag_y_1);
      if constexpr (!kSkipSmemToGmemState)
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      epilogue(frag_y_0, 0);
      epilogue(frag_y_1, 1);
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, tile_buf,
                                                        frag_y_0);
      if constexpr (!kSkipSmemToGmemState)
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      epilogue(frag_y_0, 0);
    }
  } else {
    // ── No-write path: matmul-3 + (β*decay + CB@x + CB_old@old_x + D*x + z-gate + store) ──
    using MmaAtomOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
    using LdsmBOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                        SM75_U16x2_LDSM_T>;
    auto tiled_mma_old =
        make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld>>{}, Layout<Shape<_1, Int<NUM_WARPS>>>{});
    auto thr_mma_old = tiled_mma_old.get_slice(tid);

    auto layout_old_x_trans_swz =
        make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
    auto const* old_x_base = smem.old_x + tile_buf * MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
    Tensor smem_old_x_trans =
        make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(old_x_base)),
                    layout_old_x_trans_swz);

    // s2r_A_old is dead code on the two-kernel path (CB_old in registers), kept to
    // match compute_no_write_output's declaration order and preserve code-gen for s2r_B_old_trans.
    using LdsmAOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U32x4_LDSM_N,
                                        SM75_U32x2_LDSM_N>;
    auto s2r_A_old = make_tiled_copy_A(Copy_Atom<LdsmAOld, MMA_prop::operand_t>{}, tiled_mma_old);
    auto s2r_thr_A_old = s2r_A_old.get_slice(tid);
    auto s2r_B_old_trans =
        make_tiled_copy_B(Copy_Atom<LdsmBOld, MMA_prop::operand_t>{}, tiled_mma_old);
    auto s2r_thr_B_old_trans = s2r_B_old_trans.get_slice(tid);

    float const* old_cumAdt_slot = smem.old_cumAdt + tile_buf * MAX_WINDOW;
    float const total_old_cumAdt = (prev_k > 0) ? old_cumAdt_slot[prev_k - 1] : 0.f;
    float const beta_extra = __expf(total_old_cumAdt);

    auto epilogue = [&](auto& frag_y, int n) {
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= beta_extra * __expf(decay_part(i));
      add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);
      add_cb_old_x<input_t, MMA_prop::operand_t, N_TILE, MAX_WINDOW_PAD_MMA_K>(
          frag_y, frag_CB_old, smem_old_x_trans, s2r_B_old_trans, s2r_thr_B_old_trans, thr_mma_old,
          tiled_mma_old, n);
      add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);
      compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);
      auto gOut_tile =
          make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                  make_stride(params.out_stride_token, _1{})));
      auto gOut_part = thr_mma.partition_C(gOut_tile);
#pragma unroll
      for (int i = 0; i < size(frag_y); i += 2) {
        bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
        if (pred_i)
          *reinterpret_cast<Pair<input_t>*>(&gOut_part(i)) =
              pack_float2<input_t>(make_float2(frag_y(i), frag_y(i + 1)));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, tile_buf,
                                                        frag_y_0, frag_y_1);
      epilogue(frag_y_0, 0);
      epilogue(frag_y_1, 1);
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, tile_buf,
                                                        frag_y_0);
      epilogue(frag_y_0, 0);
    }
  }

  // store_old_x: 128-thread cooperative copy (16×8 layout), only first 4 warps participate.
  if (warp < 4)
    store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                     cache_slot, write_offset, seq_len, tile_buf);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS,
          bool MUST_CHECKPOINT, int MAIN_HEADS_PER_CTA, int PIPELINE_STAGES,
          bool DO_GDC_WAIT = true, typename SmemT>
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
  // under the async state/tile prefetches and the replay MMA.
  float const D_val_0 = D_ptr ? toFloat(D_ptr[first_head]) : 0.f;

  // CB fragment infrastructure for hoisted LDGs (output_head_2k).
  // These compile-time objects are zero-cost; declared once and reused per head iteration.
  using namespace cute;
  int const tid = warp * warpSize + lane;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;
  constexpr int CB_OLD_REGS = MAX_WINDOW_PAD_MMA_K / 2;
  using MmaAtomOld_t = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
  auto tiled_mma_cb = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                     Layout<Shape<_1, Int<NUM_WARPS>>>{});
  auto tiled_mma_old_cb =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld_t>>{}, Layout<Shape<_1, Int<NUM_WARPS>>>{});
  auto thr_mma_cb = tiled_mma_cb.get_slice(tid);
  auto thr_mma_old_cb = tiled_mma_old_cb.get_slice(tid);

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

  // G2: state_1 + old_tiles_1 — pre-issued for STAGES=2 so they load across the head-0 replay.
  if constexpr (PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1) {
    prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
        smem, params, lane, warp, d_tile, first_head + 1, cache_slot, /*buf=*/1);
    load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
              /*IS_FIRST=*/false>(smem, params, lane, warp, d_tile, first_head + 1, group_idx,
                                  cache_slot, buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k,
                                  /*tile_buf=*/1);
    __pipeline_commit();  // G2: state_1 + old_tiles_1
  }

  // Drain G0+G1 (state_0+old_tiles_0 ready); keep G2 in flight across the head-0 replay.
  __pipeline_wait_prior(PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1 ? 1 : 0);
  __syncwarp();

  // ── HEAD 0: REPLAY (write path only) ──
  // Pre-replay fence: publish the prefetch_state-loaded state cross-warp before replay_state_mma
  // reads it by M-shard (a different partition than the coalesced load).
  if constexpr (MUST_CHECKPOINT) {
    __syncthreads();
    int64_t const state_ptr_offset =
        cache_slot * params.state_stride_seq + (int64_t)first_head * DIM * DSTATE;
    state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                  (int64_t)d_tile * D_PER_CTA * DSTATE;
    replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
        smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
        /*must_checkpoint=*/true, /*tile_buf=*/0);
  }
  // Unconditional CTA barrier: publishes the prefetch-loaded (+ replayed, write path) state
  // cross-warp before the output MMA reads it.  Must run on every work-unit (BOTH paths).
  // (Also serves as the warp-convergence fence for the gdc_wait below, when it fires.)
  __syncthreads();
  // ONE-SHOT PDL gdc_wait — fires only on a CTA's FIRST work-unit (DO_GDC_WAIT).  Once any
  // work-unit has waited, the precompute is done + globally visible, so subsequent grid-stride
  // iterations read cb_scaled / cumAdt_vec without re-waiting.  gdc_wait is a no-op without a
  // programmatic predecessor (non-PDL / T0).
  if constexpr (DO_GDC_WAIT) cudaGridDependencySynchronize();
  // precompute output (cumAdt_vec, cb_scaled) is now globally visible.
  // Batch-load ALL MAIN_HEADS_PER_CTA heads' cumAdt at once.  MAX_WINDOW ≤ 16 floats/head;
  // we cover 2 heads per pass by using all 32 warp-0 lanes: lanes 0..15 → head hh,
  // lanes 16..31 → head hh+1.  The __syncthreads() below broadcasts to all warps once.
  if (warp == 0) {
    int const h_off = lane >> 4;                           // 0 for lanes 0-15, 1 for lanes 16-31
    int const h_lane = lane & (NPREDICTED_PAD_MMA_M - 1);  // 0..NPREDICTED_PAD_MMA_M-1
#pragma unroll
    for (int hh = 0; hh < MAIN_HEADS_PER_CTA; hh += 2) {
      int const hh_abs = hh + h_off;
      if (hh_abs < MAIN_HEADS_PER_CTA && h_lane < seq_len)
        (smem.cumAdt + hh_abs * NPREDICTED_PAD_MMA_M)[h_lane] =
            cumAdt_ptr[(int64_t)(seq * params.nheads + first_head + hh_abs) * NPREDICTED_PAD_MMA_M +
                       h_lane];
    }
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

  // CB LDG for head 0: issued here so its HBM latency hides behind __pipeline_wait_prior +
  // __syncthreads + add_init_out K-loop.  Placed after G3/G4 commits so the x pipeline
  // is already full before we issue another LDG stream.
  auto frag_CB_new_0 = thr_mma_cb.partition_fragment_A(
      make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_CB_old_0 = thr_mma_old_cb.partition_fragment_A(
      make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  {
    auto const* cb_ptr_0 = reinterpret_cast<input_t const*>(params.cb_scaled) +
                           (int64_t)(seq * params.nheads + first_head) * warpSize * CB_NEW_REGS;
    load_cb_fragA<CB_NEW_REGS, input_t>(frag_CB_new_0, lane, cb_ptr_0);
  }
  if constexpr (!MUST_CHECKPOINT) {
    auto const* cb_old_ptr_0 = reinterpret_cast<input_t const*>(params.cb_old) +
                               (int64_t)(seq * params.nheads + first_head) * warpSize * CB_OLD_REGS;
    load_cb_fragA<CB_OLD_REGS, input_t>(frag_CB_old_0, lane, cb_old_ptr_0);
  }
  // Drain G2+G3 (old_tiles_1+state_1+C+x_0 ready); keep G4=x_1 in flight.
  __pipeline_wait_prior(PIPELINE_STAGES > 1 && MAIN_HEADS_PER_CTA > 1 ? 1 : 0);
  __syncthreads();  // broadcast all-MHC-heads cumAdt from warp-0 to all warps

  // ── HEAD 0: OUTPUT ──
  output_head_2k<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
                 NUM_WARPS, MUST_CHECKPOINT>(
      smem, params, lane, warp, d_tile, first_head, cache_slot, prev_k, out_seq_base, write_offset,
      seq_len, D_val_0, /*tile_buf=*/0, /*cumAdt_h=*/0, frag_CB_new_0, frag_CB_old_0);

  // ── HEADS 1 .. MAIN_HEADS_PER_CTA-1 ──
  for (int h = 1; h < MAIN_HEADS_PER_CTA; ++h) {
    int const head = first_head + h;
    int const cur_buf = h % PIPELINE_STAGES;
    int const next_buf = (h + 1) % PIPELINE_STAGES;
    bool const has_next = (h + 1 < MAIN_HEADS_PER_CTA);

    __syncthreads();  // inter-head barrier: prev head's output / store_old_x done

    // Hoist D LDG before the pipeline branches to hide its latency under cp.async setup + replay
    // MMA. cumAdt for all heads was batch-prefetched in the head-0 prologue — no per-head load
    // needed.
    float const D_val_h = D_ptr ? toFloat(D_ptr[head]) : 0.f;

    if constexpr (PIPELINE_STAGES == 1) {
      // STAGES=1: issue state_h + old_tiles_h here (no pre-fetch from prologue/prior iter).
      prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, lane, warp, d_tile,
                                                                 head, cache_slot, /*buf=*/0);
      load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                /*IS_FIRST=*/false>(smem, params, lane, warp, d_tile, head, group_idx, cache_slot,
                                    buf_read, outer, seq_len, MUST_CHECKPOINT, prev_k,
                                    /*tile_buf=*/0);
      __pipeline_commit();  // G_tiles: state_h + old_tiles_h

      // Issue x_h in a SEPARATE group so it overlaps with the replay below.
      load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/false>(
          smem, params, lane, warp, d_tile, head, group_idx, outer, seq_len, /*tile_buf=*/0);
      __pipeline_commit();  // G_x: x_h

      __pipeline_wait_prior(1);  // drain G_tiles (state+tiles ready); keep G_x in flight
      __syncwarp();
      // REPLAY (write path only).  Pre-replay fence: publish loaded state cross-warp before the
      // M-shard replay read (a different partition than prefetch_state's coalesced load).  No
      // post-replay fence: the pre-output __syncthreads below publishes the in-place-replayed
      // state (only gmem CB LDGs run in between → no cross-warp state read → that barrier
      // suffices).
      if constexpr (MUST_CHECKPOINT) {
        __syncthreads();
        int64_t const state_ptr_offset =
            cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;
        state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                      (int64_t)d_tile * D_PER_CTA * DSTATE;
        replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
            smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
            /*must_checkpoint=*/true, /*tile_buf=*/0);
      }
      // CB LDG after the replay: declared here to keep live range short (not across replay).
      auto frag_CB_new_h = thr_mma_cb.partition_fragment_A(make_identity_tensor(
          make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<NPREDICTED_PAD_MMA_M>{})));
      auto frag_CB_old_h = thr_mma_old_cb.partition_fragment_A(make_identity_tensor(
          make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      {
        auto const* cb_ptr_h = reinterpret_cast<input_t const*>(params.cb_scaled) +
                               (int64_t)(seq * params.nheads + head) * warpSize * CB_NEW_REGS;
        load_cb_fragA<CB_NEW_REGS, input_t>(frag_CB_new_h, lane, cb_ptr_h);
      }
      if constexpr (!MUST_CHECKPOINT) {
        auto const* cb_old_ptr_h = reinterpret_cast<input_t const*>(params.cb_old) +
                                   (int64_t)(seq * params.nheads + head) * warpSize * CB_OLD_REGS;
        load_cb_fragA<CB_OLD_REGS, input_t>(frag_CB_old_h, lane, cb_old_ptr_h);
      }
      __pipeline_wait_prior(0);  // drain G_x → x_h in smem
      __syncthreads();           // fence smem.x_h visible to all warps; cumAdt already broadcast

      output_head_2k<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, MUST_CHECKPOINT>(
          smem, params, lane, warp, d_tile, head, cache_slot, prev_k, out_seq_base, write_offset,
          seq_len, D_val_h, cur_buf, /*cumAdt_h=*/h, frag_CB_new_h, frag_CB_old_h);
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
        __pipeline_commit();  // G_tiles: state_{h+1} + old_tiles_{h+1}
      }

      // Issue x_{h+1} in its own group so it stays in flight across replay + output of head h.
      if (has_next) {
        load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/false>(
            smem, params, lane, warp, d_tile, head + 1, group_idx, outer, seq_len, next_buf);
        __pipeline_commit();  // G_xnext: x_{h+1}
      }

      // ── DRAIN TILES FOR REPLAY ──
      // has_next:  3 in-flight = [x_h(prev), G_tiles(h+1), G_xnext(h+1)].
      //            wait_prior(3) keeps all 3 in flight — the replay does NOT need x_h,
      //            so its HBM latency is hidden behind the replay MMA.
      // !has_next: 1 in-flight = [x_h].
      //            wait_prior(1) = no stall (already ≤1). x_h loads during the replay.
      __pipeline_wait_prior(has_next ? 3 : 1);
      __syncwarp();

      // REPLAY (write path only).  Pre-replay fence: publish loaded state cross-warp before the
      // M-shard replay read (a different partition than prefetch_state's coalesced load).  No
      // post-replay fence: the pre-output __syncthreads below publishes the in-place-replayed
      // state (only gmem CB LDGs run in between → no cross-warp state read → that barrier
      // suffices).
      if constexpr (MUST_CHECKPOINT) {
        __syncthreads();
        int64_t const state_ptr_offset =
            cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;
        state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                      (int64_t)d_tile * D_PER_CTA * DSTATE;
        replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
            smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
            /*must_checkpoint=*/true, cur_buf);
      }
      // CB LDG after the replay: declared here to keep live range short (not across replay).
      auto frag_CB_new_h = thr_mma_cb.partition_fragment_A(make_identity_tensor(
          make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<NPREDICTED_PAD_MMA_M>{})));
      auto frag_CB_old_h = thr_mma_old_cb.partition_fragment_A(make_identity_tensor(
          make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      {
        auto const* cb_ptr_h = reinterpret_cast<input_t const*>(params.cb_scaled) +
                               (int64_t)(seq * params.nheads + head) * warpSize * CB_NEW_REGS;
        load_cb_fragA<CB_NEW_REGS, input_t>(frag_CB_new_h, lane, cb_ptr_h);
      }
      if constexpr (!MUST_CHECKPOINT) {
        auto const* cb_old_ptr_h = reinterpret_cast<input_t const*>(params.cb_old) +
                                   (int64_t)(seq * params.nheads + head) * warpSize * CB_OLD_REGS;
        load_cb_fragA<CB_OLD_REGS, input_t>(frag_CB_old_h, lane, cb_old_ptr_h);
      }
      // ── DRAIN X FOR OUTPUT ──
      // has_next:  x_h still in flight; wait_prior(2) drains it (drops G_xprev(x_h),
      //            leaving G_tiles(h+1)+G_xnext(h+1) in flight for next iteration).
      // !has_next: x_h still in flight; wait_prior(0) drains it.
      __pipeline_wait_prior(has_next ? 2 : 0);  // drain x_h
      __syncthreads();  // fence smem.x_h visible to all warps; cumAdt already broadcast
      output_head_2k<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, MUST_CHECKPOINT>(
          smem, params, lane, warp, d_tile, head, cache_slot, prev_k, out_seq_base, write_offset,
          seq_len, D_val_h, cur_buf, /*cumAdt_h=*/h, frag_CB_new_h, frag_CB_old_h);
    }
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
                                            DSTATE, PIPELINE_STAGES, MAIN_HEADS_PER_CTA>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── 1D grid-stride persistent loop over work-units ──
  // Work-unit = (d_tile, seq, group, head_tile); a CTA processes MAIN_HEADS_PER_CTA heads per
  // work-unit (head_tile).  The launcher sizes the grid to min(cta_per_sm·NUM_SMS, total_work):
  // at the default (grid==total_work) each CTA runs exactly one work-unit → bit-identical to the
  // old 3D per-work-unit launch; with fewer CTAs each grid-strides over several, leaving SM room
  // to co-reside with conv1d (the precompute) and keeping per-group C/old_B L2-hot.
  // Flatten head_tile-fastest, then group, seq, d_tile → consecutive work-units are consecutive
  // head-tiles of one (d_tile, seq, group) (same C/old_B).
  static_assert(HEADS_PER_GROUP % MAIN_HEADS_PER_CTA == 0,
                "MAIN_HEADS_PER_CTA must divide HEADS_PER_GROUP");
  constexpr int HEAD_TILES = HEADS_PER_GROUP / MAIN_HEADS_PER_CTA;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const ngroups = (int)params.ngroups;
  int const batch = (int)params.batch;
  int const tiles_per_dseq = ngroups * HEAD_TILES;  // work-units per (d_tile, seq)
  int const total_work = D_SPLIT * batch * tiles_per_dseq;

  // gdc_wait fires only on a CTA's FIRST work-unit that actually runs (skips pad slots); once any
  // CTA work-unit has waited, the precompute is done + globally visible for the rest of the loop.
  bool did_gdc = false;

  for (int tile = blockIdx.x; tile < total_work; tile += gridDim.x) {
    // Unflatten head_tile-fastest.
    int const head_tile = tile % HEAD_TILES;
    int t1 = tile / HEAD_TILES;
    int const group_idx = t1 % ngroups;
    int t2 = t1 / ngroups;
    int const seq = t2 % batch;
    int const d_tile = t2 / batch;
    int const first_head = group_idx * HEADS_PER_GROUP + head_tile * MAIN_HEADS_PER_CTA;

    // ── Per-slot setup ──
    auto const* __restrict__ sbi =
        reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
    int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
    if (cache_slot == params.pad_slot_id) continue;  // padded slot: skip (uniform across CTA)
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
      if (seq_len <= 0) continue;
      outer = (int64_t)bos;
    } else {
      seq_len = NPREDICTED;
      outer = (int64_t)seq;
    }
    int64_t const out_seq_base = outer * params.out_stride_seq;

    bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
    int const write_offset = must_checkpoint ? 0 : prev_k;
    bool const do_gdc = !did_gdc;  // uniform across the CTA
    did_gdc = true;

    // Runtime do_gdc → compile-time DO_GDC_WAIT (gates the one-shot gdc_wait inside head_loop).
    if (must_checkpoint) {
      if (do_gdc)
        head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                  PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/true, MAIN_HEADS_PER_CTA,
                  PIPELINE_STAGES, /*DO_GDC_WAIT=*/true>(
            smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read,
            prev_k, outer, seq_len, out_seq_base, write_offset);
      else
        head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                  PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/true, MAIN_HEADS_PER_CTA,
                  PIPELINE_STAGES, /*DO_GDC_WAIT=*/false>(
            smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read,
            prev_k, outer, seq_len, out_seq_base, write_offset);
    } else {
      if (do_gdc)
        head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                  PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/false, MAIN_HEADS_PER_CTA,
                  PIPELINE_STAGES, /*DO_GDC_WAIT=*/true>(
            smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read,
            prev_k, outer, seq_len, out_seq_base, write_offset);
      else
        head_loop<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                  PHILOX_ROUNDS, NUM_WARPS, /*MUST_CHECKPOINT=*/false, MAIN_HEADS_PER_CTA,
                  PIPELINE_STAGES, /*DO_GDC_WAIT=*/false>(
            smem, params, lane, warp, d_tile, first_head, group_idx, seq, cache_slot, buf_read,
            prev_k, outer, seq_len, out_seq_base, write_offset);
    }

    // Loop-boundary fence: this work-unit's output_head_2k / store_old_x reads of smem must finish
    // before the next work-unit's prefetch_state / load_head / load_x overwrite the same buffers
    // (single-buffered across work-units).  Only when a next iteration exists (uniform condition).
    if (tile + (int)gridDim.x < total_work) __syncthreads();
  }

  // ── EXTERNAL PDL: signal a programmatic DOWNSTREAM kernel that `output` is fully written
  // (ALL work-units done; the per-head cache writes are next-step-only).  Gated by ENABLE_PDL.
  // (The internal precompute→main chain uses the one-shot gdc_wait on the first work-unit.) ──
  if constexpr (ENABLE_PDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
