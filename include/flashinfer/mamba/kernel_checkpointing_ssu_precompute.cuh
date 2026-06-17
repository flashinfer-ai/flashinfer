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
// Two-kernel split: PRECOMPUTE kernel (.plans/ssu_split.md, S3).  Validated
// bit-exact vs the monolithic via test_two_kernel_matches_monolithic (T0).
//
// Computes the conv1d-coefficient block (equations C1/C2/C5/C6) and owns the
// cache writes for the data only IT holds — old_dt/old_cumAdt (C7, from its
// dt/cumAdt) and old_B (from smem.B; the main reads cb_scaled and never loads
// B).  The main owns old_x (it loads x) and state.  Stores three scratch
// tensors the `main` kernel consumes:
//   cb_scaled : bf16, FRAGMENT-NATIVE [batch, nheads, lane(0..31), reg(0..7)] —
//               equation C5 laid out as matmul-4's fragA, so the main reads it
//               with one LDG.128 per thread straight into fragA.
//   cb_old    : bf16, same fragA-native layout — equation C6 (NO-WRITE path
//               only; the write path folds old tokens into state via the
//               replay).  Main does OUT.3 = cb_old @ old_x.
//   cumAdt_vec: f32, (batch, nheads, NPREDICTED_PAD_MMA_M) — raw cumAdt[t].  The
//               main loads it into smem.cumAdt; the existing exp(smem.cumAdt)
//               epilogue computes β on the fly (OUT.1) — no decay branch.
//
// Launch granularity (see .plans/ssu_split.md "Granularity"):
//   grid = (batch, ngroups, ceil(HEADS_PER_GROUP / HEADS_PER_CTA))
//   First cut: HEADS_PER_CTA == HEADS_PER_GROUP → tiles=1 → 1 CTA/group.
//   The raw C·B matmul (C5's C·B contraction over DSTATE) is PER-GROUP — B/C
//   are (.,.,ngroups,dstate) — so it is computed ONCE per CTA and the result
//   (frag_acc, registers) is reused across the tile's heads; only the per-head
//   decay scaling differs.  That is the whole point of the per-group grid.
//
// CB handling: the 2-warp MMA writes raw C·B as FP32 to swizzled smem.CB
// (scale in fp32, like the monolithic).  Each warp then scales its head and
// writes the result to gmem with one STG.128/thread in fragA-native layout —
// the MAIN never LDSMs CB; it reads its 8-bf16 fragment with one LDG.128
// straight into matmul-4's fragA.  bf16 conversion happens only at the gmem
// store, so numerics match the monolithic.
// =============================================================================
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_

#include <cute/tensor.hpp>

#include "checkpointing_ssu.cuh"
#include "common.cuh"  // PackedAligned
#include "kernel_checkpointing_ssu_common.cuh"

namespace flashinfer::mamba::checkpointing {

// -----------------------------------------------------------------------------
// Lean precompute smem: only the C·B-path buffers (B, C, CB) + PER-HEAD
// coefficients (every head's inputs loaded up front in PHASE 1).  Drops the
// monolithic CheckpointingSsuStorage's state / old_x / old_B / x / z — none
// touched here — so the precompute uses far less smem → more CTAs/SM.
//
// B / C / CB_scaled keep byte-identical swizzled layouts to
// CheckpointingSsuStorage, so the same `make_swizzled_layout_rc` accessors in
// the MMA / LDSM helpers index them unchanged.  dt / cumAdt / old_dt / old_cumAdt
// are PER-HEAD (PHASE 1 loads every head's inputs with all 128 threads; PHASE 2
// scans + scales per head) — see the storage struct + kernel body below.
template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, int NUM_WARPS,
          int HEADS_PER_CTA>
struct CheckpointingSsuPrecomputeStorage {
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  // Raw C·B (C5) / C·old_B (C6) kept FP32 (scale in fp32 like the monolithic; cast
  // to bf16 only at the gmem STG).  Stored ELEMENT-MAJOR in matmul-4 fragA order —
  // smem[e*32 + lane] — NOT swizzled.  The C·B m16n8 accumulator (regs d0..d3)
  // IS the matmul-4 m16n8k16 A-fragment, same lane: W0 → fragA[0..3] (cols 0-7),
  // W1 → fragA[4..7] (cols 8-15).  So no relayout: compute_cb stores its 4
  // accumulator regs element-major (per element, 32 lanes → 32 contiguous f32 =
  // conflict-free), and scale_store reads the whole fragA back with one
  // conflict-free element-major load (once per warp, not per head).  REGS = K/2 =
  // matmul-4 A-frag size.
  static constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;  // m16n8k16 → 8
  static constexpr int CB_OLD_REGS = MAX_WINDOW_PAD_MMA_K / 2;  // m16n8k{K_old} → 4 or 8
  float CB[CB_NEW_REGS * warpSize];      // C5 new-token raw C·B, fragA element-major (e*W+lane)
  float CB_old[CB_OLD_REGS * warpSize];  // C6 old-token raw C·old_B (no-write), element-major

  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];      // matmul-1 N-operand (new B, swizzled)
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];      // matmul-1 A-operand (C, swizzled)
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];  // C6 N-operand (old B, no-write only)

  // Per-HEAD coefficients, indexed by the LOCAL head (0..HEADS_PER_CTA).  PHASE 1
  // loads ALL heads' dt / old_dt / old_cumAdt / dt_bias / A up front with all 128
  // threads; PHASE 2's warp-per-head loop reads them — so it issues NO gmem loads
  // and can't stall on tiny per-head fetches (was the #1 long_scoreboard source,
  // the per-head old_dt/old_cumAdt load).  No `decay` slot: exp(cumAdt) is
  // computed on the fly by scale_store_cb / the main's OUT.1; cumAdt itself is
  // what cumAdt_vec stores.  Smem scales with HEADS_PER_CTA (~4 KB at 16 heads);
  // a large HEADS_PER_GROUP (e.g. 64) would need head tiling to stay in budget.
  float dt[HEADS_PER_CTA][NPREDICTED];      // C1: raw dt (PHASE 1) → softplus in place (PHASE 2)
  float cumAdt[HEADS_PER_CTA][NPREDICTED];  // C2 scan output (PHASE 2)
  float old_dt[HEADS_PER_CTA][MAX_WINDOW];  // C6/C7 (NO-WRITE)
  float old_cumAdt[HEADS_PER_CTA][MAX_WINDOW];  // C6/C7 (NO-WRITE)
  float dt_bias[HEADS_PER_CTA];                 // per-head bias (PHASE 1)
  float A[HEADS_PER_CTA];                       // per-head A   (PHASE 1)
};

// -----------------------------------------------------------------------------
// Load this group's C (conv1d output) into the swizzled smem.C via cp.async —
// gmem→smem direct, no register footprint for the ~4 KB tile.  C is the A-operand
// of BOTH the new-token MMA (W0/1, compute_cb_2warp) and the old-token MMA (W2/3,
// compute_cb_old_2warp on the no-write path), so it is loaded by ALL warps (each
// warp's own cp.async — no cross-warp sync before the MMA — idempotent same-payload
// writes, exactly like load_post_pdl_wait_data, common.cuh:671).
// Committed, NOT drained here — the caller waits ONCE before the CB matmul so C/B
// overlap old_B (load_old_B) in flight instead of serializing.  Caller must have
// gdc_wait'd first (C is a conv1d output).
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_C(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int group_idx, int64_t outer, int seq_len) {
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane, seq_len);
  __pipeline_commit();
}

// -----------------------------------------------------------------------------
// Load this group's B (conv1d output) into the swizzled smem.B via cp.async.  B
// (new) is the N-operand of the new-token MMA, only read by W0/1 — so only they
// call this (the caller gates on warp < 2, the same warps that store_old_B).
// Committed, NOT drained here (see load_C).  Caller must have gdc_wait'd first.
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_B(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int group_idx, int64_t outer, int seq_len) {
  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  int64_t const B_base = outer * params.B_stride_seq + (int64_t)group_idx * DSTATE;
  using BShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_token, lane, seq_len);
  __pipeline_commit();
}

// Load this group's OLD B (cache, the buffered input-proj) into swizzled
// smem.old_B via cp.async — the N-operand of the C6 MMA (compute_cb_old_2warp).
// Called on the NO-WRITE path only, by W2/3 (the warps that run the old MMA);
// each loads the full tile (idempotent).  Valid rows = prev_k (the buffered
// tokens); the rest are masked by load_tile_async.  old_B is double-buffered:
// (state_cache_size, 2, MAX_WINDOW, ngroups, dstate).
template <typename input_t, int MAX_WINDOW, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_old_B(SmemT& smem, CheckpointingSsuParams const& params,
                                           int lane, int64_t cache_slot, int buf_read,
                                           int group_idx, int prev_k) {
  auto const* __restrict__ oldB_ptr = reinterpret_cast<input_t const*>(params.old_B);
  int64_t const base = cache_slot * params.old_B_stride_seq +
                       (int64_t)buf_read * params.old_B_stride_dbuf + (int64_t)group_idx * DSTATE;
  using OldBShape = cute::Shape<cute::Int<SmemT::MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  load_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, oldB_ptr + base, params.old_B_stride_token,
                                         lane, prev_k);
  __pipeline_commit();  // NOT drained here — see load_C/load_B; caller drains once before the MMA.
}

// -----------------------------------------------------------------------------
// PHASE 1: load ALL heads' scalar coefficients into per-head smem, ALL 128 threads
// (dt + dt_bias + A always; old_dt / old_cumAdt on the no-write path).  Batching every
// head's LDGs lets the global-load latency OVERLAP; PHASE 2 then does ZERO gmem loads.
// bias + softplus deferred to PHASE 2 (smem).  conv1d-INDEPENDENT (only B/C are conv1d
// outputs), so WHERE the caller runs it is a free choice — see the two call sites.
template <typename dt_t, typename weight_t, typename matrixA_t, int NPREDICTED, int MAX_WINDOW,
          int HEADS_PER_CTA, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_phase1_coeffs(SmemT& smem,
                                                   CheckpointingSsuParams const& params, int warp,
                                                   int lane, int first_head, int seq_len,
                                                   int64_t outer, int64_t cache_slot, int buf_read,
                                                   bool must_checkpoint, int prev_k) {
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  int const flat_tid = warp * warpSize + lane;
  constexpr int CTA_THREADS = NUM_WARPS * 32;
  // dt[t, head]: heads contiguous in gmem (stride 1), tokens strided (dt_stride_token).
  // idx → (t, h) with h = idx % HEADS_PER_CTA so consecutive threads hit consecutive
  // heads → coalesced read.
  auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);
  constexpr int DT_N = HEADS_PER_CTA * NPREDICTED;
#pragma unroll
  for (int idx = flat_tid; idx < DT_N; idx += CTA_THREADS) {
    int const h = idx % HEADS_PER_CTA;
    int const t = idx / HEADS_PER_CTA;
    float v = 0.f;
    if (t < seq_len)
      v = toFloat(dt_ptr[outer * params.dt_stride_seq + (int64_t)t * params.dt_stride_token +
                         first_head + h]);
    smem.dt[h][t] = v;  // raw; bias + softplus applied in PHASE 2
  }
  // Per-head dt_bias + A (one thread per head; heads contiguous → coalesced).
  if (flat_tid < HEADS_PER_CTA) {
    smem.A[flat_tid] = toFloat(A_ptr[first_head + flat_tid]);
    smem.dt_bias[flat_tid] = dt_bias_ptr ? toFloat(dt_bias_ptr[first_head + flat_tid]) : 0.f;
  }
  // NO-WRITE only: old_dt / old_cumAdt[head, t] (f32 cache) — tokens contiguous (stride 1),
  // heads strided (stride_head).  idx → (h, t) with t = idx % MAX_WINDOW so consecutive
  // threads hit consecutive tokens → coalesced read.  Masked to the valid window
  // (t < prev_k); the rest is zeroed so the C7 tail read (old_cumAdt[prev_k-1]) and the
  // scale_store mask stay well-defined.
  if (!must_checkpoint) {
    auto const* __restrict__ odt_ptr = reinterpret_cast<float const*>(params.old_dt);
    auto const* __restrict__ oca_ptr = reinterpret_cast<float const*>(params.old_cumAdt);
    int64_t const odt_base =
        cache_slot * params.old_dt_stride_seq + (int64_t)buf_read * params.old_dt_stride_dbuf;
    int64_t const oca_base = cache_slot * params.old_cumAdt_stride_seq +
                             (int64_t)buf_read * params.old_cumAdt_stride_dbuf;
    constexpr int OLD_N = HEADS_PER_CTA * MAX_WINDOW;
#pragma unroll
    for (int idx = flat_tid; idx < OLD_N; idx += CTA_THREADS) {
      int const h = idx / MAX_WINDOW;
      int const t = idx % MAX_WINDOW;
      float dv = 0.f, cv = 0.f;
      if (t < prev_k) {
        int64_t const hh = (int64_t)(first_head + h);
        dv = odt_ptr[odt_base + hh * params.old_dt_stride_head + t];
        cv = oca_ptr[oca_base + hh * params.old_cumAdt_stride_head + t];
      }
      smem.old_dt[h][t] = dv;
      smem.old_cumAdt[h][t] = cv;
    }
  }
}

// -----------------------------------------------------------------------------
// C1: scalar per-lane LDG of dt[outer, t, head] + bias + softplus → per-warp
// smem.dt[warp].  dt is tiny (NPREDICTED floats/head, T-axis stride
// dt_stride_token = nheads), so an on-the-fly strided LDG is fine — no cp.async.
// Same compute as load_pre_pdl_wait_data's dt→dt_proc block (common.cuh:613).
// DT_SOFTPLUS is JIT-stamped (the launcher reads params.dt_softplus once and
// dispatches), so the softplus branch folds away at compile time.
template <typename dt_t, int NPREDICTED, bool DT_SOFTPLUS, typename SmemT>
__device__ __forceinline__ void load_dt(SmemT& smem, CheckpointingSsuParams const& params, int warp,
                                        int lane, int head, int64_t outer, float dt_bias_val,
                                        int seq_len) {
  if (lane < seq_len) {
    auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);
    int64_t const base = outer * params.dt_stride_seq + head;
    float dt_val = toFloat(dt_ptr[base + (int64_t)lane * params.dt_stride_token]) + dt_bias_val;
    if constexpr (DT_SOFTPLUS) dt_val = thresholded_softplus(dt_val);
    smem.dt[warp][lane] = dt_val;
  }
}

// -----------------------------------------------------------------------------
// C2: cumAdt = cumsum(A * dt) (inclusive Hillis-Steele warp scan) → per-head
// smem.cumAdt[h] (h = LOCAL head index).  The scanning warp reads its head's
// PHASE-1-loaded (and softplus'd) dt.  decay = exp(cumAdt) is NOT materialized —
// scale_store_cb / the main's OUT.1 exp it on the fly.
template <int NPREDICTED, typename SmemT>
__device__ __forceinline__ void compute_cumAdt_pw(SmemT& smem, int h, int lane, float A_val) {
  float val = (lane < NPREDICTED) ? A_val * smem.dt[h][lane] : 0.f;
  for (int offset = 1; offset < NPREDICTED; offset *= 2) {
    float other = __shfl_up_sync(constants::MASK_ALL_LANES, val, offset);
    if (lane >= offset) val += other;
  }
  if (lane < NPREDICTED) {
    smem.cumAdt[h][lane] = val;
  }
}

// -----------------------------------------------------------------------------
// Raw C·B MMA (matmul-1) for ONE group, 2-warp N-split → fp32 swizzled smem.CB
// (UNMASKED, UNSCALED).  This is the MMA half of compute_CB_scaled_2warp
// (kernel_checkpointing_ssu_common.cuh:806) with the scale/mask epilogue
// dropped — the per-head decay/dt scaling + causal mask are deferred to
// scale_store_cb.  Reads the same swizzled smem.C / smem.B; stores RAW
// fp32 via a ROW-MAJOR layout so scale_store reads by logical (t,j) = t*S+j.
// Run by warps 0/1:
//   NPREDICTED_PAD_MMA_N == 16: warp 0 → cols [0,8), warp 1 → cols [8,16).
//   NPREDICTED_PAD_MMA_N == 8 : warp 0 covers all valid j (<NPREDICTED); warp 1
//     returns (its cols [8,16) are j>=8>=seq_len, never read by scale_store).
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_cb_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;
  static_assert(N_HALF % MMA_prop::N == 0, "N_HALF must be a multiple of MMA::N");

  if constexpr (NPREDICTED_PAD_MMA_N == 8) {
    if (warp == 1) return;  // no valid B rows; cols [8,16) never read.
  }

  // ── Swizzled smem views (C/B byte-identical to the monolithic layout) ──
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_B = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_N, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_B = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));
  Tensor smem_B_half =
      local_tile(smem_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp, _));

  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_B_half(_, _, _0{}));
  auto layout_cb_half = make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  Tensor frag_acc = thr_mma.partition_fragment_C(make_tensor((float*)nullptr, layout_cb_half));
  clear(frag_acc);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(lane);
  Tensor smem_C_s2r = s2r_thr_A.partition_S(smem_C_tiled);
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(lane);
  Tensor smem_B_s2r = s2r_thr_B.partition_S(smem_B_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  // ── Gemm: DSTATE/K_TILE K-tiles, 1 HMMA each ──
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_B_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  // ── Store the m16n8 accumulator ELEMENT-MAJOR → smem.CB (no swizzle) ──
  // No scaling / mask (deferred to scale_store_cb).  This warp's 4 accumulator
  // regs ARE matmul-4 fragA[warp*4 .. warp*4+3], same lane (W0 → fragA[0..3] =
  // cols 0-7, W1 → fragA[4..7] = cols 8-15), so we dump them element-major:
  // smem.CB[(warp*4 + i)*32 + lane].  Per element, 32 lanes write 32 contiguous
  // f32 → conflict-free; scale_store reads the whole fragA back the same way.
  int const warp_base = warp * (int)size(frag_acc);  // 4 regs/lane (m16n8 C)
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem.CB[(warp_base + i) * warpSize + lane] = frag_acc(i);  // raw fp32
  }
}

// -----------------------------------------------------------------------------
// Raw C·old_B MMA (C6, matmul-1 for the OLD tokens) for ONE group → fp32
// swizzled smem.CB_old (UNMASKED, UNSCALED).  Identical to compute_cb_2warp
// except the N-operand is smem.old_B (the buffered B, MAX_WINDOW_PAD_MMA_K rows)
// and the output is smem.CB_old.  The C6 decay/coeff scaling + write are
// deferred to scale_store_cb (IS_OLD=true, the no-write counterpart of the
// IS_OLD=false path).  Run by warps 2/3 on the NO-WRITE path only; the
// `warp_in_pair` arg is (threadIdx.y - 2) ∈ {0,1} so the N-split logic matches
// compute_cb_2warp's (warp 0 → cols [0,8), warp 1 → cols [8,16)).
//   MAX_WINDOW_PAD_MMA_K == 8: warp_in_pair 1 returns (only one N-tile).
template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_cb_old_2warp(SmemT& smem, int warp_in_pair, int lane) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int N_HALF = MMA_prop::N;  // 8 = one m16n8 N-tile per warp
  static_assert(MAX_WINDOW_PAD_MMA_K % N_HALF == 0, "MAX_WINDOW_PAD_MMA_K must be a multiple of N");

  if constexpr (MAX_WINDOW_PAD_MMA_K == N_HALF) {
    if (warp_in_pair == 1) return;  // single N-tile; cols [8,16) never read.
  }

  // ── Swizzled smem views (C/old_B byte-identical to the monolithic layout) ──
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_oldB = make_swizzled_layout_rc<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_oldB =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.old_B)), layout_oldB);

  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));
  Tensor smem_oldB_half =
      local_tile(smem_oldB, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp_in_pair, _));

  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_oldB_half(_, _, _0{}));
  auto layout_cb_half = make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  Tensor frag_acc = thr_mma.partition_fragment_C(make_tensor((float*)nullptr, layout_cb_half));
  clear(frag_acc);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(lane);
  Tensor smem_C_s2r = s2r_thr_A.partition_S(smem_C_tiled);
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(lane);
  Tensor smem_oldB_s2r = s2r_thr_B.partition_S(smem_oldB_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_oldB_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  // ── Store the m16n8 accumulator ELEMENT-MAJOR → smem.CB_old (no swizzle) ──
  // Same scheme as compute_cb_2warp: this N-tile's 4 regs ARE matmul-4-old fragA
  // [warp_in_pair*4 .. +3], same lane (single N-tile at K=8 → all of fragA[0..3];
  // two N-tiles at K=16 → W2 fragA[0..3], W3 fragA[4..7]).  Element-major,
  // conflict-free.
  int const warp_base = warp_in_pair * (int)size(frag_acc);  // 4 regs/lane (m16n8 C)
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem.CB_old[(warp_base + i) * warpSize + lane] = frag_acc(i);  // raw fp32
  }
}

// -----------------------------------------------------------------------------
// Scale one head's raw C·B (IS_OLD=false → equation C5, new tokens) or C·old_B
// (IS_OLD=true → equation C6, old tokens, NO-WRITE path) and store it to gmem
// FRAGMENT-NATIVE, one STG per lane.  Each thread emits the REGS values that ==
// matmul-4's `fragA` for this lane (mma.m16n8k{2·REGS} A operand), so the MAIN
// kernel reads them with a single LDG straight into fragA — no LDSM, no swizzle,
// neither side puts CB in smem as bf16.  The two paths share the ENTIRE fragA
// mapping + swizzled read + pack + STG; only the per-column scale and the mask
// differ, so those fold behind `if constexpr (IS_OLD)` (zero runtime cost) — the
// caller just points cb_smem / gmem_head at the right buffers.
//
// fragA element e → CB coords (row t, col c), on the fly (folds at unroll):
//   r0=lane/4, c0=(lane%4)*2; t=r0+(((e>>1)&1)<<3), c=c0+(((e>>2)&1)<<3)+(e&1).
//   (REGS=8 → 16 cols m16n8k16; REGS=4 → 8 cols m16n8k8 = the [0,8) prefix.)
// New (C5): CB_scaled[t,c] = (C·B)·exp(cumAdt[t]−cumAdt[c])·dt[c],  mask c≤t
//   (causal) ∧ c<seq_len ∧ t<seq_len.
// Old (C6): CB_old[t,c] = (C·old_B)·exp(cumAdt[t]+Σ_old−old_cumAdt[c])·old_dt[c],
//   mask c<prev_k ∧ t<seq_len (no causal — old tokens all precede the new ones).
//   Σ_old = old_cumAdt[prev_k−1] (0 when prev_k==0 → all-zero cb_old).
// Both use the SINGLE-__expf form, matching the monolithic bit-for-bit
// (kernel_checkpointing_ssu_common.cuh:933 new / :1054 old).  Scaling is fp32,
// cast to bf16 only at the pack.
// `raw` is this lane's matmul-4 fragA — the unscaled C·B (C5) / C·old_B (C6),
// loaded ONCE per warp from the element-major smem (raw[e] == CB[(t,c)_e]) and
// reused across all this group's heads.  REGS = K/2: new = NPREDICTED_PAD_MMA_M/2
// (=8, STG.128); old = MAX_WINDOW_PAD_MMA_K/2 (4 @ K=8 → STG.64, 8 @ K=16 → .128).
template <bool IS_OLD, int REGS, typename input_t, typename SmemT>
__device__ __forceinline__ void scale_store_cb(SmemT& smem, int h, int lane, int seq_len,
                                               int prev_k, float const (&raw)[REGS],
                                               input_t* __restrict__ gmem_head) {
  float total_old = 0.f;
  if constexpr (IS_OLD) {
    total_old = (prev_k > 0) ? smem.old_cumAdt[h][prev_k - 1] : 0.f;
  }
  int const r0 = lane / 4;
  int const c0 = (lane % 4) * 2;
  PackedAligned<input_t, REGS> packed;
#pragma unroll
  for (int e = 0; e < REGS; ++e) {
    int const t = r0 + (((e >> 1) & 1) << 3);
    int const c = c0 + (((e >> 2) & 1) << 3) + (e & 1);
    float val = 0.f;
    if constexpr (IS_OLD) {
      // C6: old tokens all precede the new ones — mask c<prev_k, no causal.
      if (c < prev_k && t < seq_len)
        val = raw[e] * __expf(smem.cumAdt[h][t] + total_old - smem.old_cumAdt[h][c]) *
              smem.old_dt[h][c];
    } else {
      // C5: causal CB_scaled[t,c]; per-head coeffs (this head's PHASE-1/scan data).
      if (c <= t && t < seq_len && c < seq_len)
        val = raw[e] * __expf(smem.cumAdt[h][t] - smem.cumAdt[h][c]) * smem.dt[h][c];
    }
    packed.val[e] = static_cast<input_t>(val);
  }
  reinterpret_cast<PackedAligned<input_t, REGS>*>(gmem_head)[lane] = packed;  // one STG
}

// -----------------------------------------------------------------------------
// PRECOMPUTE kernel.  Template params mirror checkpointing_ssu_kernel.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NPREDICTED, int MAX_WINDOW, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int HEADS_PER_CTA, int NUM_WARPS, bool DT_SOFTPLUS,
          bool VARLEN = false>
__global__ void checkpointing_ssu_precompute_kernel(CheckpointingSsuParams params) {
  // HEADS_PER_CTA <= HEADS_PER_GROUP: the group's heads are tiled across
  // ceil(HEADS_PER_GROUP / HEADS_PER_CTA) CTAs (grid.z) so small batches fill the
  // GPU.  It sizes the per-head coeff smem, so a small HEADS_PER_CTA shrinks smem
  // too.  The per-group C / B load + C·B (and C·old_B) matmul are recomputed by
  // every head-tile — cheap (Tensor-core-light) and L2-hot, hidden under the load
  // latency on the latency-bound small-batch path.  The launcher picks
  // HEADS_PER_CTA from the heuristic hc ~= (batch·nheads)/(SMs·occ).
  using SmemT = CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE,
                                                  NUM_WARPS, HEADS_PER_CTA>;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;

  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── INTERNAL PDL (precompute → main): UNCONDITIONAL — the split's mechanism,
  // not gated by ENABLE_PDL.  Fired FIRST (Triton fires gdc_launch_dependents at
  // the top of its precompute, replay_selective_state_update.py:971).
  // launch_dependents is a launch-eligibility hint, NOT a memory release: the
  // main's own gdc_wait blocks until THIS grid fully completes + its stores are
  // visible, so cb_scaled / cumAdt_vec / cb_old are always seen regardless of
  // where we fire this.  Firing it early lets the main's (precompute-independent)
  // state replay run concurrently with our conv1d wait + CB matmul — the whole
  // point of the split.  The main is launched with the programmatic attr
  // unconditionally (launch_checkpointing_ssu.cuh), so this always pairs.
  cudaTriggerProgrammaticLaunchCompletion();  // main may co-launch now

  // ── Grid (batch, ngroups, head_tiles) ──
  int const seq = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const head_tile = blockIdx.z;  // 0 for first cut (tiles==1)
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const first_head = group_idx * HEADS_PER_GROUP + head_tile * HEADS_PER_CTA;

  // ── Per-slot setup (shared across the group's heads) ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
  if (cache_slot == params.pad_slot_id) return;
  // prev_k is on the critical path — must_checkpoint (and load_old_B's row extent) consume it
  // first, and there's no executed work before that consume to hide the load.  Issue it FIRST,
  // via __ldg (read-only path), so its latency overlaps the buf_read load instead of fully
  // stalling must_checkpoint on a cold dependent global load.
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = __ldg(&prev_ptr[cache_slot]);
  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);

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
  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  // ── old_B (cache, conv1d-INDEPENDENT cp.async): hoisted BEFORE gdc_wait so it
  // overlaps the conv1d wait (cliff) / the B/C cp.async (standalone) — old_B is the
  // buffered cache, NOT a conv1d output.  The cp.async stays in flight across the
  // grid-dep sync; the single drain below completes it.  NO-WRITE + W2/3 (the C6-MMA
  // warps); must_checkpoint is uniform per CTA. ──
  if (!must_checkpoint && warp >= 2) {
    load_old_B<input_t, MAX_WINDOW, DSTATE>(smem, params, lane, cache_slot, buf_read, group_idx,
                                            prev_k);
  }

  // ── EXTERNAL PDL (conv1d → precompute): gated by ENABLE_PDL — conv1d produces B/C,
  // so wait before load_C/load_B.  PHASE 1 (conv1d-independent) runs HERE too, before
  // the wait, so its LDGs fill the conv1d wait window.  No-op without programmatic conv1d.
  if constexpr (ENABLE_PDL) {
    load_phase1_coeffs<dt_t, weight_t, matrixA_t, NPREDICTED, MAX_WINDOW, HEADS_PER_CTA, NUM_WARPS>(
        smem, params, warp, lane, first_head, seq_len, outer, cache_slot, buf_read, must_checkpoint,
        prev_k);
    cudaGridDependencySynchronize();
  }

  // ── Load this group's C (all warps) + B (W0/1 only — the new-token N-operand,
  // read only by the C·B warps) into swizzled smem (cp.async). ──
  load_C<input_t, NPREDICTED, DSTATE>(smem, params, lane, group_idx, outer, seq_len);
  if (warp < 2) {
    load_B<input_t, NPREDICTED, DSTATE>(smem, params, lane, group_idx, outer, seq_len);
  }

  // Standalone (!ENABLE_PDL): no conv1d wait to hide behind, so run PHASE 1 HERE to
  // overlap the B/C + old_B cp.async issued above instead of exposing its LDG latency.
  if constexpr (!ENABLE_PDL) {
    load_phase1_coeffs<dt_t, weight_t, matrixA_t, NPREDICTED, MAX_WINDOW, HEADS_PER_CTA, NUM_WARPS>(
        smem, params, warp, lane, first_head, seq_len, outer, cache_slot, buf_read, must_checkpoint,
        prev_k);
  }

  // ── Drain ALL cp.async ONCE (C/B from load_C/load_B + old_B from load_old_B) BEFORE
  // any consumer — the helpers only commit, so the loads overlap in flight rather than
  // each draining its own group (which serialized B/C → old_B).  MUST precede
  // store_old_B (reads smem.B) AND the CB matmul (reads smem.B/C/old_B).  __syncwarp
  // publishes them within each warp (loads are redundant per-warp). ──
  __pipeline_wait_prior(0);
  __syncwarp();

  // ── old_B cache writeback (per-group, D-independent): the new B tokens become the
  // buffered "old" for the next step.  Reads the now-drained smem.B.  W0/1 hold smem.B;
  // the precompute owns this writeback (the main reads cb_scaled, never loads B).
  // store_old_B self-gates on head % HEADS_PER_GROUP — first_head qualifies. ──
  if (warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(smem, params, warp, lane, first_head,
                                                              group_idx, cache_slot, buf_write,
                                                              write_offset, seq_len);
  }

  // ── Raw matmul-1 — ONCE per group, all 4 warps (no-write).  W0/1 → C·B (C5) →
  // smem.CB; W2/3 → C·old_B (C6, no-write) → smem.CB_old.  Independent of PHASE 1
  // (different smem); both land before the __syncthreads below. ──
  if (warp < 2) {
    compute_cb_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);
  } else if (!must_checkpoint) {
    compute_cb_old_2warp<input_t, NPREDICTED, MAX_WINDOW, DSTATE>(smem, warp - 2, lane);
  }

  // ── Publish PHASE-1 loads + the raw-CB stores cross-warp before PHASE 2 ──
  __syncthreads();

  // ── PHASE 2: warp-per-head, NO gmem loads.  Each loop iteration is one head:
  // bias+softplus (from smem) → warp scan → cumAdt; then store cumAdt_vec, the C7
  // cache tape, and scale_store_cb C5 / C6.  Every input is already in smem
  // (PHASE 1 + the scan), so the loop never stalls on a load.  NUM_ITER is
  // uniform, so every warp runs the same iteration count. ──
  auto* __restrict__ cb_gmem = reinterpret_cast<input_t*>(params.cb_scaled);   // CB_NEW_REGS/lane
  auto* __restrict__ cb_old_gmem = reinterpret_cast<input_t*>(params.cb_old);  // K_old/2 regs/lane
  auto* __restrict__ cumAdt_gmem = reinterpret_cast<float*>(params.cumAdt_vec);
  constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;         // m16n8k16 A-frag size (=8)
  constexpr int CB_OLD_REGS = SmemT::MAX_WINDOW_PAD_MMA_K / 2;  // m16n8k{K_old} A-frag size

  // Load the raw C·B / C·old_B fragA ONCE per warp (it's the same for all this
  // group's heads — the per-head decay scaling is applied below).  Element-major
  // (smem[e*32+lane]) → one conflict-free LDS per element.  Replaces the per-head
  // re-reads + the swizzled-store bank conflict.
  float raw_cb[CB_NEW_REGS];
#pragma unroll
  for (int e = 0; e < CB_NEW_REGS; ++e) raw_cb[e] = smem.CB[e * warpSize + lane];

  float raw_cb_old[CB_OLD_REGS];  // NO-WRITE only (loaded + used iff !must_checkpoint)

  if (!must_checkpoint) {
#pragma unroll
    for (int e = 0; e < CB_OLD_REGS; ++e) raw_cb_old[e] = smem.CB_old[e * warpSize + lane];
  }

  constexpr int NUM_ITER = (HEADS_PER_CTA + NUM_WARPS - 1) / NUM_WARPS;  // ceil, uniform
#pragma unroll
  for (int iter = 0; iter < NUM_ITER; ++iter) {
    int const h = warp + iter * NUM_WARPS;
    bool const has_head = (h < HEADS_PER_CTA);
    int const head = first_head + h;

    if (has_head) {
      // C1: bias + softplus on the pre-loaded raw dt (own-lane, in place).
      if (lane < seq_len) {
        float dv = smem.dt[h][lane] + smem.dt_bias[h];
        if constexpr (DT_SOFTPLUS) dv = thresholded_softplus(dv);
        smem.dt[h][lane] = dv;
      }
      // C2: cumsum scan (own-lane read + register shfl) → smem.cumAdt[h].
      compute_cumAdt_pw<NPREDICTED>(smem, h, lane, smem.A[h]);
      // Publish dt[h] (softplus) + cumAdt[h] cross-lane for the cross-lane reads
      // (C7 tail + scale_store_cb) below.
      __syncwarp();

      // cumAdt_vec[batch, head, t] = raw cumAdt[t] (the main exp's it for β).
      if (lane < seq_len)
        cumAdt_gmem[(int64_t)(seq * params.nheads + head) * NPREDICTED_PAD_MMA_M + lane] =
            smem.cumAdt[h][lane];
      // C7: persist this head's new dt / cumAdt into the cache tape at write_offset.
      // cumAdt continuity: no-write steps add the previous buffer's tail
      // (old_cumAdt[h][prev_k-1]) so the cache stays a monotone cumsum.
      if (lane < seq_len) {
        auto* __restrict__ old_dt_w = reinterpret_cast<float*>(params.old_dt);
        auto* __restrict__ old_ca_w = reinterpret_cast<float*>(params.old_cumAdt);
        int64_t const dt_w_base = cache_slot * params.old_dt_stride_seq +
                                  (int64_t)buf_write * params.old_dt_stride_dbuf +
                                  (int64_t)head * params.old_dt_stride_head;
        int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_seq +
                                  (int64_t)buf_write * params.old_cumAdt_stride_dbuf +
                                  (int64_t)head * params.old_cumAdt_stride_head;
        float const cumAdt_prefix =
            (!must_checkpoint && prev_k > 0) ? smem.old_cumAdt[h][prev_k - 1] : 0.f;
        old_dt_w[dt_w_base + write_offset + lane] = smem.dt[h][lane];
        old_ca_w[ca_w_base + write_offset + lane] = smem.cumAdt[h][lane] + cumAdt_prefix;
      }
      // C5: scale the register-resident raw C·B by this head + STG → cb_scaled.
      auto* cb_gmem_head = cb_gmem + (int64_t)(seq * params.nheads + head) * warpSize * CB_NEW_REGS;
      scale_store_cb</*IS_OLD=*/false, CB_NEW_REGS, input_t>(smem, h, lane, seq_len, prev_k, raw_cb,
                                                             cb_gmem_head);
      // C6 (NO-WRITE): scale the register-resident raw C·old_B + STG → cb_old.
      if (!must_checkpoint) {
        auto* cb_old_head =
            cb_old_gmem + (int64_t)(seq * params.nheads + head) * warpSize * CB_OLD_REGS;
        scale_store_cb</*IS_OLD=*/true, CB_OLD_REGS, input_t>(smem, h, lane, seq_len, prev_k,
                                                              raw_cb_old, cb_old_head);
      }
    }
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
