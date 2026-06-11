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
// Two-kernel split: PRECOMPUTE kernel (.plans/ssu_split.md, S3).
//
// PROTOTYPE / DESIGN DRAFT — structure for review, expect compile iteration.
//
// Computes the conv1d-coefficient block (equations C1/C2/C5 + the
// old_dt/old_cumAdt cache writes) and stores two scratch tensors the `main`
// kernel consumes:
//   cb_scaled : bf16, FRAGMENT-NATIVE [batch, nheads, lane(0..31), reg(0..7)] —
//               equation C5 laid out as matmul-4's fragA, so the main reads it
//               with one LDG.128 per thread straight into fragA.
//   decay_vec : f32, (batch, nheads, NPREDICTED_PAD_MMA_M) — exp(cumAdt[t]),
//               the per-head β factor for the main's OUT.1 (β·C@state).
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
// Lean precompute smem: only the C·B-path buffers (B, C, CB) + per-warp
// coefficients (warp-per-head).  Drops the monolithic CheckpointingSsuStorage's
// state / old_x / old_B / x / z — none touched here — so the precompute uses
// far less smem → more CTAs/SM → better occupancy (the lever we're chasing).
//
// B / C / CB_scaled keep byte-identical swizzled layouts to
// CheckpointingSsuStorage, so the same `make_swizzled_layout_rc` accessors in
// the MMA / LDSM helpers index them unchanged.  cumAdt/dt/decay are
// PER-WARP (each warp owns its head's C1/C2) — unlike the monolithic's single
// copy (one head per CTA).
template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, int NUM_WARPS>
struct CheckpointingSsuPrecomputeStorage {
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  // Raw C·B (C5, new tokens) and C·old_B (C6, old tokens — no-write path only)
  // kept in FP32 (scale in fp32 like the monolithic; cast to bf16 only at the
  // final gmem STG.128).  Swizzled via SmemSwizzle<float> (like the monolithic's
  // bf16 CB uses SmemSwizzle<input_t>) so the accumulator store + per-warp (t,j)
  // reads are bank-conflict-free; CB_STRIDE = the f32 swizzle atom cols (32).
  static constexpr int CB_STRIDE = SmemSwizzle<float>::ATOM_COLS;
  float CB[NPREDICTED_PAD_MMA_M * CB_STRIDE];      // C5 new-token raw C·B   (16×32 f32)
  float CB_old[NPREDICTED_PAD_MMA_M * CB_STRIDE];  // C6 old-token raw C·old_B (no-write)

  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];      // matmul-1 N-operand (new B, swizzled)
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];      // matmul-1 A-operand (C, swizzled)
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];  // C6 N-operand (old B, no-write only)

  // Per-warp coefficients — warp w owns its head's C1/C2 results.
  float dt[NUM_WARPS][NPREDICTED];
  float cumAdt[NUM_WARPS][NPREDICTED];
  float decay[NUM_WARPS][NPREDICTED];
};

// -----------------------------------------------------------------------------
// Load this group's C and B (conv1d outputs) into the swizzled smem.C / smem.B
// via cp.async — gmem→smem direct, no register footprint for the ~4 KB tiles.
// C is the A-operand of BOTH the new-token MMA (W0/1, compute_cb_2warp) and the
// old-token MMA (W2/3, compute_cb_old_2warp on the no-write path), so it is
// loaded by ALL warps (each warp's own cp.async — no cross-warp sync before the
// MMA — idempotent same-payload writes, exactly like load_post_pdl_wait_data,
// common.cuh:671).  B (new) is only read by W0/1, so only they load it.  Caller
// must have gdc_wait'd first (B/C are conv1d outputs).
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_group_BC(SmemT& smem, CheckpointingSsuParams const& params,
                                              int warp, int lane, int group_idx, int64_t outer,
                                              int seq_len) {
  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  int64_t const B_base = outer * params.B_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using BShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane, seq_len);
  if (warp < 2) {
    load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_token, lane,
                                        seq_len);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
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
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
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
// C2: cumAdt = cumsum(A * dt) (inclusive Hillis-Steele warp scan) + decay =
// exp(cumAdt) → per-warp smem.cumAdt[warp] / smem.decay[warp].  Per-warp variant
// of compute_cumAdt (common.cuh:451).
template <int NPREDICTED, typename SmemT>
__device__ __forceinline__ void compute_cumAdt_pw(SmemT& smem, int warp, int lane, float A_val) {
  float val = (lane < NPREDICTED) ? A_val * smem.dt[warp][lane] : 0.f;
  for (int offset = 1; offset < NPREDICTED; offset *= 2) {
    float other = __shfl_up_sync(constants::MASK_ALL_LANES, val, offset);
    if (lane >= offset) val += other;
  }
  if (lane < NPREDICTED) {
    smem.cumAdt[warp][lane] = val;
    smem.decay[warp][lane] = __expf(val);
  }
}

// -----------------------------------------------------------------------------
// Raw C·B MMA (matmul-1) for ONE group, 2-warp N-split → fp32 swizzled smem.CB
// (UNMASKED, UNSCALED).  This is the MMA half of compute_CB_scaled_2warp
// (kernel_checkpointing_ssu_common.cuh:806) with the scale/mask epilogue
// dropped — the per-head decay/dt scaling + causal mask are deferred to
// scale_store_cb_gmem.  Reads the same swizzled smem.C / smem.B; stores RAW
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

  // ── Store RAW fp32 accumulator → swizzled smem.CB, this warp's N-half ──
  // No scaling, no mask (deferred to scale_store_cb_gmem).  Swizzled
  // (SmemSwizzle<float>) for bank-conflict-free store/read; scale_store reads
  // the identical (t,j) through the same make_swizzled_layout_rc<float>.
  constexpr int CB_STRIDE = SmemT::CB_STRIDE;
  auto layout_cb =
      make_swizzled_layout_rc<float, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_STRIDE>();
  Tensor smem_CB = make_tensor(make_smem_ptr(smem.CB), layout_cb);
  Tensor smem_CB_half = local_tile(smem_CB, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}),
                                   make_coord(_0{}, warp));
  Tensor smem_CB_part = thr_mma.partition_C(smem_CB_half);
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem_CB_part(i) = frag_acc(i);  // raw fp32
  }
}

// -----------------------------------------------------------------------------
// Raw C·old_B MMA (C6, matmul-1 for the OLD tokens) for ONE group → fp32
// swizzled smem.CB_old (UNMASKED, UNSCALED).  Identical to compute_cb_2warp
// except the N-operand is smem.old_B (the buffered B, MAX_WINDOW_PAD_MMA_K rows)
// and the output is smem.CB_old.  The C6 decay/coeff scaling + write are
// deferred to scale_store_cb_old (the no-write counterpart of
// scale_store_cb_gmem).  Run by warps 2/3 on the NO-WRITE path only; the
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

  // ── Store RAW fp32 accumulator → swizzled smem.CB_old, this warp's N-tile ──
  constexpr int CB_STRIDE = SmemT::CB_STRIDE;
  auto layout_cb =
      make_swizzled_layout_rc<float, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_STRIDE>();
  Tensor smem_CB_old = make_tensor(make_smem_ptr(smem.CB_old), layout_cb);
  Tensor smem_CB_old_half =
      local_tile(smem_CB_old, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}),
                 make_coord(_0{}, warp_in_pair));
  Tensor smem_CB_old_part = thr_mma.partition_C(smem_CB_old_half);
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem_CB_old_part(i) = frag_acc(i);  // raw fp32
  }
}

// -----------------------------------------------------------------------------
// Scale one head's raw C·B (read fp32 from smem.CB) and store to gmem in
// FRAGMENT-NATIVE layout, one STG.128 per thread.  Each thread emits the 8
// values that == matmul-4's `fragA` for this lane (mma.m16n8k16 A operand), so
// the MAIN kernel reads them with a single LDG.128 straight into fragA — no
// LDSM, no swizzle, neither side puts CB in smem as bf16.
//
// The 8 fragA elements for this lane map to CB coords (t,j) computed on the fly
// from the register index e:
//   e0=(r0,c0)   e1=(r0,c0+1) e2=(r1,c0)   e3=(r1,c0+1)   [cols 0..7]
//   e4=(r0,c0+8) e5=(r0,c0+9) e6=(r1,c0+8) e7=(r1,c0+9)   [cols 8..15]
// with r0=lane/4, r1=r0+8, c0=(lane%4)*2.
// Scaling is done in fp32 (raw read fp32 from smem.CB), cast to bf16 only at
// the pack — matches the monolithic's precision.
// cb_gmem_head = &cb_scaled[batch_slot, head, 0]  (32 × PackedAligned<input_t>).
template <typename input_t, typename SmemT>
__device__ __forceinline__ void scale_store_cb_gmem(
    SmemT& smem, int warp, int lane, int seq_len,
    PackedAligned<input_t>* __restrict__ cb_gmem_head) {
  static_assert(PackedAligned<input_t>::count == 8,
                "cb_scaled fragA store assumes 8 input_t per 16 B pack (bf16)");
  using namespace cute;
  // Same swizzled f32 layout compute_cb_2warp stored CB through.
  constexpr int M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const layout_cb = make_swizzled_layout_rc<float, M, M, SmemT::CB_STRIDE>();
  int const r0 = lane / 4;
  int const c0 = (lane % 4) * 2;
  PackedAligned<input_t> packed;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    // fragA(m16n8k16) element e → (row t, col j), on the fly (folds at unroll).
    int const t = r0 + (((e >> 1) & 1) << 3);
    int const j = c0 + (((e >> 2) & 1) << 3) + (e & 1);
    float val = 0.f;
    if (j <= t && t < seq_len && j < seq_len) {
      // C5: CB_scaled[t,j] = (C·B) * exp(cumAdt[t]-cumAdt[j]) * dt[j].
      // Raw C·B read fp32 from swizzled smem (same layout compute_cb_2warp
      // stored); scaled in fp32, per-warp coeffs (this warp owns this head).
      val = smem.CB[layout_cb(t, j)] * __expf(smem.cumAdt[warp][t] - smem.cumAdt[warp][j]) *
            smem.dt[warp][j];
    }
    packed.val[e] = static_cast<input_t>(val);
  }
  cb_gmem_head[lane] = packed;  // one STG.128
}

// -----------------------------------------------------------------------------
// PRECOMPUTE kernel.  Template params mirror checkpointing_ssu_kernel.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NPREDICTED, int MAX_WINDOW, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int NUM_WARPS, bool DT_SOFTPLUS, bool VARLEN = false>
__global__ void checkpointing_ssu_precompute_kernel(CheckpointingSsuParams params) {
  using SmemT =
      CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE, NUM_WARPS>;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;
  // First cut: one CTA per (batch, group) — every head of the group handled by
  // this CTA's head loop (HEADS_PER_CTA == HEADS_PER_GROUP).
  constexpr int HEADS_PER_CTA = HEADS_PER_GROUP;  // TODO(tiling): heuristic for HPG=64.

  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

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
  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  if constexpr (ENABLE_PDL) {
    cudaGridDependencySynchronize();  // conv1d produces B/C — wait before the load.
  }

  // ── Load this group's C, B (conv1d outputs) into swizzled smem (cp.async) ──
  load_group_BC<input_t, NPREDICTED, DSTATE>(smem, params, warp, lane, group_idx, outer, seq_len);
  // NO-WRITE only: W2/3 also load this group's old B (cache) for the C6 MMA.
  // must_checkpoint is uniform per CTA, so this branch is divergence-free.
  if (!must_checkpoint && warp >= 2) {
    load_old_B<input_t, MAX_WINDOW, DSTATE>(smem, params, lane, cache_slot, buf_read, group_idx,
                                            prev_k);
  }

  // ── Raw matmul-1 — ONCE per group, all 4 warps (no-write) ──
  // W0/1 → C·B (C5, new tokens) → smem.CB; W2/3 → C·old_B (C6, old tokens,
  // no-write only) → smem.CB_old.  On the WRITE path W2/3 idle here (the main
  // folds old tokens into state via the replay instead) and pick up heads in the
  // loop below.  smem.CB / smem.CB_old become visible cross-warp at the loop's
  // iter==0 __syncthreads, before any scale_store reads them.
  if (warp < 2) {
    compute_cb_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);
  } else if (!must_checkpoint) {
    compute_cb_old_2warp<input_t, NPREDICTED, MAX_WINDOW, DSTATE>(smem, warp - 2, lane);
  }

  // ── Warp-per-head ──
  // Each warp LDSMs the shared raw CB → fragA, scales by ITS head, STG.128s
  // fragA-native to gmem.  NUM_ITER is uniform across warps, so the iter==0
  // __syncthreads() is reached by every warp exactly once — deadlock-proof even
  // when HEADS_PER_CTA < NUM_WARPS (tail warps just no-op the h-guarded work).
  // The barrier is deferred past the cumAdt compute so the raw-CB smem store
  // (warps 0/1) overlaps it.
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto* __restrict__ cb_gmem = reinterpret_cast<PackedAligned<input_t>*>(params.cb_scaled);
  auto* __restrict__ decay_gmem = reinterpret_cast<float*>(params.decay_vec);

  constexpr int NUM_ITER = (HEADS_PER_CTA + NUM_WARPS - 1) / NUM_WARPS;  // ceil, uniform
#pragma unroll
  for (int iter = 0; iter < NUM_ITER; ++iter) {
    int const h = warp + iter * NUM_WARPS;
    bool const has_head = (h < HEADS_PER_CTA);
    int const head = first_head + h;

    if (has_head) {
      float const A_val = toFloat(A_ptr[head]);
      float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      load_dt<dt_t, NPREDICTED, DT_SOFTPLUS>(smem, params, warp, lane, head, outer, dt_bias_val,
                                             seq_len);         // C1
      compute_cumAdt_pw<NPREDICTED>(smem, warp, lane, A_val);  // C2
    }

    // Sync before the reads below.  iter 0 uses __syncthreads — it also makes
    // the warps-0/1 raw-CB store visible cross-warp before scale_store; later
    // iters only need __syncwarp (CB unchanged since iter 0; this publishes this
    // warp's freshly-written dt/cumAdt for the cross-lane reads).  NUM_ITER is
    // uniform so the __syncthreads is reached by every warp exactly once.
    if (iter == 0)
      __syncthreads();
    else
      __syncwarp();

    if (has_head) {
      // decay_vec[batch, head, t] = exp(cumAdt[t]).
      if (lane < seq_len)
        decay_gmem[(int64_t)(seq * params.nheads + head) * NPREDICTED_PAD_MMA_M + lane] =
            smem.decay[warp][lane];
      // C5: scale the shared raw C·B (fp32 from smem.CB) by this head's per-warp
      // coeffs + STG.128 fragA-native to gmem.
      auto* cb_gmem_head = cb_gmem + (int64_t)(seq * params.nheads + head) * warpSize;
      scale_store_cb_gmem<input_t>(smem, warp, lane, seq_len, cb_gmem_head);
      // C7 + cache: store old_dt / old_cumAdt at write_offset (reuse
      // store_old_dt / store_old_cumAdt, common.cuh:1582+).  TODO(cache).
    }

    // WAR: this iter's cross-lane reads of smem.dt/cumAdt[warp] must complete
    // before the next iter overwrites them (single-slot reuse).  A double-buffer
    // [NUM_WARPS][2] would remove this and let the next scan overlap the store.
    __syncwarp();
  }
  (void)buf_write;
  (void)write_offset;
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
