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
#ifndef FLASHINFER_MAMBA_KERNEL_SSU_INCREMENTAL_CUH_
#define FLASHINFER_MAMBA_KERNEL_SSU_INCREMENTAL_CUH_

// Incremental SSU kernel — matmul-based, tensor-core MMA (single path).
// Single CTA per (batch, head). Grid: (batch, nheads).
// 4 warps per CTA, 128 threads total.
//
// v11.0: single __syncthreads() via per-warp data ownership.  Every smem
// read before the final barrier is served by data the same warp loaded.
// No mbarriers, no cross-warp visibility for the first half of the kernel.
//
// Phase 0 (per-warp cp.async, no cross-warp sync):
//   State: each warp loads own DIM slice (rows [16W : 16W+16]).
//   B, C:  redundant on W0, W1 (both do 2-warp CB).
//   old_B: redundant on all 4 warps (each warp's replay needs full DSTATE).
//   old_x: redundant on all 4 warps.
//   x:     W2 only  (Phase-2 read — covered by the single syncthreads).
//   z:     W3 only  (Phase-2 read — covered by the single syncthreads).
//   Scalars + cumAdt: redundant on each warp's first NTOKENS lanes.
//   Each warp: __pipeline_commit → __pipeline_wait_prior(0) → __syncwarp.
//
// Phase 1 (runs with *no* barrier; CB ‖ replay parallelism preserved):
//   - store_old_B hoisted here — W0,W1 only (they hold valid smem.B).
//   - Warps 0,1: compute_CB_scaled_2warp (bf16 HMMA → swizzled smem.CB_scaled).
//   - All warps: replay_state_mma (HMMA; state in smem updated in-place,
//                each warp touches only its own DIM rows).
//
// __syncthreads()   ← THE ONE.  Provides cross-warp visibility of:
//                      CB_scaled (W0,W1→all), x (W2→all), z (W3→all).
//
// Phase 2: compute_and_store_output_cute
//   = (C @ state^T) * decay + CB_scaled @ x + D*x, z-gate → direct gmem STG.
//   State writeback hoisted inside the orchestrator once matmul 3 has
//   finished consuming smem.state.
//
// Phase 3: old_x / old_dt_proc / old_cumAdt cache writes.

#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "cute/tensor.hpp"
#include "ssu_incremental.cuh"
#include "ssu_mtp_common.cuh"

namespace flashinfer::mamba::incremental {

using namespace conversion;

// Round x up to the next multiple of Y (Y must be a power of 2).
template <int Y>
constexpr int next_multiple_of(int x) {
  static_assert(Y > 0 && (Y & (Y - 1)) == 0, "Y must be a power of 2");
  return (x + Y - 1) & ~(Y - 1);
}

// NativeOf<T>::type: scalar T → 2-wide native CUDA vector type.
template <typename T>
struct NativeOf;
template <>
struct NativeOf<float> {
  using type = float2;
};
template <>
struct NativeOf<__half> {
  using type = __half2;
};
template <>
struct NativeOf<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

// Pair<T>: thin wrapper over the native 2-wide type, adding compile-time
// `[cute::Int<I>{}]` indexing so index-driven loops stay branchless.  Same
// layout as the native type — `=` compiles to one LDS.U32 / STS.U32 and the
// pair stays in one register.
template <typename T>
struct Pair {
  typename NativeOf<T>::type raw;
  template <int I>
  __device__ __forceinline__ auto operator[](cute::Int<I>) const {
    static_assert(I == 0 || I == 1, "Pair index must be 0 or 1");
    if constexpr (I == 0)
      return raw.x;
    else
      return raw.y;
  }
};

// pack_float2<T>: float2 → Pair<T>, using packed hardware cvt when available.
template <typename T>
__device__ __forceinline__ Pair<T> pack_float2(float2 val);
template <>
__device__ __forceinline__ Pair<float> pack_float2<float>(float2 val) {
  return {val};
}
template <>
__device__ __forceinline__ Pair<__half> pack_float2<__half>(float2 val) {
  return {__float22half2_rn(val)};
}
template <>
__device__ __forceinline__ Pair<__nv_bfloat16> pack_float2<__nv_bfloat16>(float2 val) {
  return {conversion::fromFloat2(val)};
}

// Swizzled smem layout for mma.sync operands (bf16/fp16, row-major).
// Atom: 8 rows × 64 cols (128 bytes/row = full bank cycle).
// Swizzle<3,3,3> XORs upper 3 col bits with row bits → bank-conflict-free ldmatrix.
// 8 consecutive columns (128-bit / 16 bytes) remain physically contiguous,
// so cp.async 16B writes to swizzled addresses work correctly.
template <int ROWS, int COLS>
__device__ __forceinline__ auto make_swizzled_layout_rc() {
  using namespace cute;
  static_assert(ROWS % 8 == 0, "ROWS must be multiple of 8 for swizzle atom");
  static_assert(COLS % 64 == 0, "COLS must be multiple of 64 for swizzle atom");
  auto atom = composition(Swizzle<3, 3, 3>{},
                          make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{})));
  return tile_to_shape(atom, make_shape(Int<ROWS>{}, Int<COLS>{}));
}

// Transposed swizzled smem layout: maps (col, row) → same physical offset as
// make_swizzled_layout_rc maps (row, col). Enables bank-conflict-free ldmatrix.trans
// reads on data stored with make_swizzled_layout_rc.
// Built by swapping modes of the original inner layout (before swizzle), which
// guarantees correct cross-atom offsets when both dimensions have multiple atoms.
template <int ROWS, int COLS>
__device__ __forceinline__ auto make_swizzled_layout_rc_transpose() {
  using namespace cute;
  static_assert(ROWS % 8 == 0, "ROWS must be multiple of 8 for swizzle atom");
  static_assert(COLS % 64 == 0, "COLS must be multiple of 64 for swizzle atom");
  // Build the inner (un-swizzled) tiled layout for the original (ROWS, COLS) layout
  auto inner = tile_to_shape(make_layout(make_shape(_8{}, _64{}), make_stride(_64{}, _1{})),
                             make_shape(Int<ROWS>{}, Int<COLS>{}));
  // Swap modes to get true transpose: result(c, r) == original(r, c)
  auto inner_T = make_layout(get<1>(inner), get<0>(inner));
  return composition(Swizzle<3, 3, 3>{}, inner_T);
}

// =============================================================================
// Shared memory layout
// =============================================================================
// `smem_state_t` is the in-smem representation of the state. For 2-byte
// `state_t` (bf16/f16) it matches `state_t` and we cp.async from gmem
// directly. For f32 `state_t` we narrow to bf16 on load (and widen back on
// writeback) so the tensor-core path stays uniform — there is no SIMT
// fallback for f32 state.
template <typename state_t>
using smem_state_t_of = std::conditional_t<sizeof(state_t) == 2, state_t, __nv_bfloat16>;

template <typename input_t, typename state_t, int NTOKENS, int DIM, int DSTATE>
struct SsuIncrementalStorage {
  // M-dim of the output MMAs (C, x, z, CB_scaled): always m16-tiled.
  static constexpr int NTOKENS_PAD_MMA_M = next_multiple_of<16>(NTOKENS);
  // K-dim of the replay MMA (B, old_x, old_B).  Padded to the LDSM unit (8).
  // For NTOKENS ≤ 8 this is 8 → replay uses m16n8k8 (1 K-tile, smaller smem,
  // +1 CTA/SM occupancy).  For NTOKENS > 8 this is 16 → replay uses m16n8k16
  // (1 K-tile, fewer MMA ops).  Assumes NTOKENS ≤ 16 (asserted in wrapper).
  static constexpr int NTOKENS_PAD_MMA_K = next_multiple_of<8>(NTOKENS);

  using smem_state_t = smem_state_t_of<state_t>;

  // CB_scaled: input_t[NTOKENS_PAD_MMA_M * 64] — padded row stride 128 bytes
  // (32 banks), accessed via Swizzle<3,3,3> CuTe layout for conflict-free
  // LDSM in the matmul-4 A-operand read.
  alignas(16) char CB_scaled[NTOKENS_PAD_MMA_M * 64 * sizeof(input_t)];

  // B: NTOKENS_PAD_MMA_K rows.  Padding rows inside [NTOKENS, NTOKENS_PAD_MMA_K)
  // contain garbage — valid output uses only [0, NTOKENS).  Warp-1 of
  // compute_CB_scaled_2warp reads rows ≥ 8 of a 16-row view; those reads spill
  // into C/old_B smem but are masked to 0 by the (j < NTOKENS) CB-store
  // predicate since j ≥ 8 ≥ NTOKENS when NTOKENS_PAD_MMA_K == 8.
  alignas(16) input_t B[NTOKENS_PAD_MMA_K][DSTATE];

  // C: padded to NTOKENS_PAD_MMA_M (matmul 3 A operand via LDSM.x4).
  alignas(16) input_t C[NTOKENS_PAD_MMA_M][DSTATE];

  // x: padded to NTOKENS_PAD_MMA_M rows for mma operand alignment (matmul 4 B operand).
  alignas(16) input_t x[NTOKENS_PAD_MMA_M][DIM];

  // z: padded to NTOKENS_PAD_MMA_M so partition_C reads don't go OOB.
  alignas(16) input_t z[NTOKENS_PAD_MMA_M][DIM];

  // Old cache data loaded in Phase 0 (consumed in Phase 1 replay).
  // old_x: NTOKENS_PAD_MMA_K rows, Swizzle<3,3,3> [NTOKENS_PAD_MMA_K, DIM] layout
  // — ldmatrix.trans feeds replay MMA A-operand directly.
  alignas(16) input_t old_x[NTOKENS_PAD_MMA_K][DIM];

  // old_B: NTOKENS_PAD_MMA_K rows, Swizzle<3,3,3> [NTOKENS_PAD_MMA_K, DSTATE] layout.
  // Replay MMA reads via ldmatrix.trans (LDSM_T) + register scaling.
  // Padding rows zero-filled via cp.async ZFILL.
  alignas(16) input_t old_B[NTOKENS_PAD_MMA_K][DSTATE];

  float old_dt_proc[NTOKENS];
  float old_cumAdt[NTOKENS];

  // Processed dt for new tokens (Phase 1a uses this for CB_scaled + cumAdt)
  float dt_proc[NTOKENS];

  // Cumulative A*dt — computed once by warp 0, read by all warps after sync
  float cumAdt[NTOKENS];

  // State buffer, always 2-byte in smem so the MMA path is uniform.  For
  // f32 state_t, load_data converts on ingress and store_state widens on
  // egress — the body of the kernel never sees f32.
  alignas(16) smem_state_t state[DIM][DSTATE];
};

// =============================================================================
// B/C/x/z load helper: single-warp cp.async into swizzled smem.
// =============================================================================

// Generic swizzled cp.async with ZFILL for padding rows — the six [ROWS_PAD,
// COLS] single-warp loaders (B, old_B, x, z, old_x, C) all collapse into this.
// Gmem tile is [ROWS_PAD, COLS] with runtime row stride; rows >= VALID_ROWS
// are zero-filled in smem without touching gmem (cp.async.ca.ZFILL).  Thread
// layout Shape<_4,_8>×val Shape<_1,_8> = 32 threads × 16B each = one warp
// covers 4 rows × 64 cols per step.
//
// Template args are all compile-time so the ZFILL predicate constant-folds;
// the caller pre-offsets `gmem_src` by the tile base, keeping 64-bit pointer
// math at the callsite.  `SmemShape` is a CuTe static shape, e.g.
// `cute::Shape<cute::Int<16>, cute::Int<128>>` — (rows_pad, cols_pad).  The
// shape travels as a single type so later we can pad cols too (e.g. DSTATE=96
// rounded up to a full bank cycle) without growing the parameter list.
template <typename SmemShape, int VALID_ROWS, typename input_t>
__device__ __forceinline__ void load_tile_async(input_t* __restrict__ smem_dst,
                                                input_t const* __restrict__ gmem_src,
                                                int gmem_row_stride, int lane) {
  using namespace cute;
  constexpr int ROWS_PAD = size<0>(SmemShape{});
  constexpr int COLS = size<1>(SmemShape{});
  Tensor s = make_tensor(make_smem_ptr(smem_dst), make_swizzled_layout_rc<ROWS_PAD, COLS>());
  Tensor g = make_tensor(make_gmem_ptr(gmem_src),
                         make_layout(SmemShape{}, make_stride(gmem_row_stride, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(SmemShape{});
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < VALID_ROWS;
  }
  copy_if(g2s, pred, thr.partition_S(g), thr.partition_D(s));
}

// State load helpers — v11.0: per-warp partitioning.
//
// Warp W loads its own contiguous DIM slice (rows [W*DIM_PER_WARP :
// (W+1)*DIM_PER_WARP]) into the shared swizzled smem.state buffer.  Writes
// go to different rows per warp, so no cross-warp smem collisions.  After
// each warp's own __pipeline_wait_prior + __syncwarp, that warp sees its
// own rows — sufficient because replay and Phase 2 read state with the
// same per-warp DIM partitioning (no cross-warp state reads).
//
// 32 lanes cover (DIM_PER_WARP=16, DSTATE=128) via thread layout (4, 8) ×
// val (1, 8) — **atom-aligned** with the Swizzle<3,3,3> (8, 64) atom.  Per-
// tile 4 × 64 is full-atom-width in N, half-atom-height in M.  This matches
// the proven conflict-free pattern used by `load_tile_async`.  Iterations:
// (DIM_PER_WARP/4, DSTATE/64) = (4, 2) = 8 tiles/lane, 64 elts/lane total.
// Writes across iterations XOR-spread across banks correctly because each
// in-flight cp.async covers one full swizzle atom.
template <typename state_t, int DIM, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_per_warp(SmemT& smem,
                                                    state_t const* __restrict__ state_ptr,
                                                    int64_t state_base, int warp, int lane) {
  using namespace cute;
  static_assert(sizeof(state_t) == 2, "load_state_per_warp requires 2-byte state type");
  static_assert(NUM_WARPS == 4, "Expected 4 warps");
  static_assert(DIM % NUM_WARPS == 0, "DIM must be divisible by NUM_WARPS");
  constexpr int DIM_PER_WARP = DIM / NUM_WARPS;

  Tensor sState_full = make_tensor(make_smem_ptr(reinterpret_cast<state_t*>(&smem.state[0][0])),
                                   make_swizzled_layout_rc<DIM, DSTATE>());
  Tensor gState_full = make_tensor(
      make_gmem_ptr(state_ptr + state_base),
      make_layout(make_shape(Int<DIM>{}, Int<DSTATE>{}), make_stride(Int<DSTATE>{}, Int<1>{})));

  Tensor sState = local_tile(sState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));
  Tensor gState = local_tile(gState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, state_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
}

// Narrowing state load: gmem f32 → swizzled smem bf16.  Used only when
// state_t is f32 — the tensor-core path wants a 2-byte state in smem, so we
// pay the conversion once on ingress and let the body of the kernel run
// through the uniform bf16 fast path.  Plain LDG (cp.async cannot narrow).
// Element-at-a-time write so we don't rely on swizzle-adjacency of (col,
// col+1) — correctness first, since f32 state isn't a hot path.
//
// v11.0: per-warp partitioning — warp W narrows rows [W*DIM_PER_WARP :
// (W+1)*DIM_PER_WARP].
template <int DIM, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_f32_to_bf16_per_warp(SmemT& smem,
                                                                float const* __restrict__ state_ptr,
                                                                int64_t state_base, int warp,
                                                                int lane) {
  using namespace cute;
  static_assert(std::is_same_v<typename SmemT::smem_state_t, __nv_bfloat16>,
                "narrowing path assumes bf16 smem state");
  static_assert(NUM_WARPS == 4, "Expected 4 warps");
  static_assert(DIM % NUM_WARPS == 0, "DIM must be divisible by NUM_WARPS");
  constexpr int DIM_PER_WARP = DIM / NUM_WARPS;

  auto layout_smem_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(&smem.state[0][0]);
  float const* gmem_state = state_ptr + state_base;

  int const row_base = warp * DIM_PER_WARP;
  constexpr int TOTAL_PER_WARP = DIM_PER_WARP * DSTATE;

  for (int i = lane; i < TOTAL_PER_WARP; i += warpSize) {
    int const row = row_base + i / DSTATE;
    int const col = i % DSTATE;
    smem_state[layout_smem_swz(row, col)] = __float2bfloat16(gmem_state[row * DSTATE + col]);
  }
}

// =============================================================================
// Phase 0: cooperative data load into smem (all warps).
// Compute cumAdt[T] = cumsum(A * dt_proc) → smem.
// Warp-level inclusive prefix sum using Hillis-Steele shuffles.
// Only the first NTOKENS lanes participate; the rest are idle.
template <int NTOKENS, typename SmemT>
__device__ __forceinline__ void compute_cumAdt(SmemT& smem, int lane, float A_val) {
  float val = (lane < NTOKENS) ? A_val * smem.dt_proc[lane] : 0.f;
  // Inclusive prefix sum (Hillis-Steele)
  for (int offset = 1; offset < NTOKENS; offset *= 2) {
    float other = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane >= offset) val += other;
  }
  if (lane < NTOKENS) {
    smem.cumAdt[lane] = val;
  }
}

// Loads B, C, x, z, state, old_x, old_B (cp.async 16B), old_dt_proc,
// old_cumAdt, dt → dt_proc (LDG + softplus → smem), cumAdt (warp shuffle).
//
// v11.0: per-warp data ownership.  Each warp loads exactly what it (and
// its warp-peers) will consume before the single CTA-wide barrier.
// Every warp waits for its own cp.async and issues __syncwarp for
// intra-warp visibility — no cross-warp sync needed here.  The kernel's
// sole __syncthreads comes after CB + replay, covering CB_scaled, x, z
// visibility for Phase 2.
//
// Load distribution:
//   state:  per-warp contiguous DIM slice (warp W owns rows [16W : 16W+16]).
//   B, C:   redundant on W0, W1 (both compute 2-warp CB, both need full).
//   old_B:  redundant on all 4 warps (each warp's replay reads full DSTATE).
//   old_x:  redundant on all 4 warps (small, ~2 KB — partitioning not worth
//           the complication).
//   x:      W2 only (Phase-2 read, covered by final __syncthreads).
//   z:      W3 only (Phase-2 read, covered by final __syncthreads).
//   scalars (old_dt_proc, old_cumAdt, dt→dt_proc) + cumAdt cumsum:
//           redundant on each warp's first NTOKENS lanes.  Writes are
//           idempotent across warps (identical payloads to same slots).
// =============================================================================
template <typename input_t, typename dt_t, typename state_t, int NTOKENS, int DIM, int DSTATE,
          int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_data(SmemT& smem, SsuIncrementalParams const& params, int lane,
                                          int warp, int batch_idx, int head, int group_idx,
                                          int64_t cache_slot, int buf_read, float A_val,
                                          float dt_bias_val) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(DIM % INPUT_PACK == 0, "DIM must be divisible by input pack size");

  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ old_x_ptr = reinterpret_cast<input_t const*>(params.old_x);
  auto const* __restrict__ old_B_ptr = reinterpret_cast<input_t const*>(params.old_B);
  auto const* __restrict__ old_dt_proc_ptr = reinterpret_cast<float const*>(params.old_dt_proc);
  auto const* __restrict__ old_cumAdt_ptr = reinterpret_cast<float const*>(params.old_cumAdt);
  auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);

  int64_t const B_base = (int64_t)batch_idx * params.B_stride_batch + (int64_t)group_idx * DSTATE;
  int64_t const C_base = (int64_t)batch_idx * params.C_stride_batch + (int64_t)group_idx * DSTATE;
  int64_t const x_base = (int64_t)batch_idx * params.x_stride_batch + head * DIM;
  int64_t const ox_base = cache_slot * params.old_x_stride_cache + head * DIM;
  int64_t const oB_base = cache_slot * params.old_B_stride_cache +
                          buf_read * params.old_B_stride_dbuf + group_idx * DSTATE;

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;
  using CShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_M>, cute::Int<DSTATE>>;
  using BShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_K>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_M>, cute::Int<DIM>>;
  using OxShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_K>, cute::Int<DIM>>;

  // ── State: per-warp DIM slice (rows [16W : 16W+16]) ──
  {
    auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
    int64_t const state_base =
        cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE;
    if constexpr (sizeof(state_t) == 2) {
      load_state_per_warp<state_t, DIM, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp, lane);
    } else {
      static_assert(sizeof(state_t) == 4, "unexpected state_t size");
      load_state_f32_to_bf16_per_warp<DIM, DSTATE, NUM_WARPS>(
          smem, reinterpret_cast<float const*>(state_ptr), state_base, warp, lane);
    }
  }

  // ── B + C: redundant on W0, W1 (both do 2-warp CB compute) ──
  if (warp < 2) {
    load_tile_async<BShape, NTOKENS>(&smem.B[0][0], B_ptr + B_base, params.B_stride_mtp, lane);
    load_tile_async<CShape, NTOKENS>(&smem.C[0][0], C_ptr + C_base, params.C_stride_mtp, lane);
  }

  // ── old_B: redundant on all 4 warps (each warp's replay consumes full
  // DSTATE).  Identical payloads to same smem dest — final bytes
  // deterministic. ──
  load_tile_async<BShape, NTOKENS>(&smem.old_B[0][0], old_B_ptr + oB_base, params.old_B_stride_mtp,
                                   lane);

  // ── old_x: redundant on all 4 warps (small, simpler than partitioning). ──
  load_tile_async<OxShape, NTOKENS>(&smem.old_x[0][0], old_x_ptr + ox_base, params.old_x_stride_mtp,
                                    lane);

  // ── x: W2 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 2) {
    load_tile_async<XShape, NTOKENS>(&smem.x[0][0], x_ptr + x_base, params.x_stride_mtp, lane);
  }

  // ── z: W3 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 3 && z_ptr) {
    int64_t const z_base = (int64_t)batch_idx * params.z_stride_batch + head * DIM;
    load_tile_async<XShape, NTOKENS>(&smem.z[0][0], z_ptr + z_base, params.z_stride_mtp, lane);
  }

  // ── Scalar loads + cumAdt cumsum: redundant per warp (first NTOKENS
  // lanes).  Synchronous LDG + plain smem stores — no cp.async.  Writes
  // from 4 warps to the same slots are idempotent (same payloads). ──
  static_assert(NTOKENS <= warpSize, "NTOKENS must fit in a single warp");
  if (lane < NTOKENS) {
    int64_t const dt_rd_base = cache_slot * params.old_dt_proc_stride_cache +
                               buf_read * params.old_dt_proc_stride_dbuf +
                               head * params.old_dt_proc_stride_head;
    smem.old_dt_proc[lane] = old_dt_proc_ptr[dt_rd_base + lane];

    int64_t const ca_rd_base = cache_slot * params.old_cumAdt_stride_cache +
                               buf_read * params.old_cumAdt_stride_dbuf +
                               head * params.old_cumAdt_stride_head;
    smem.old_cumAdt[lane] = old_cumAdt_ptr[ca_rd_base + lane];

    float dt_val = toFloat(
        dt_ptr[(int64_t)batch_idx * params.dt_stride_batch + lane * params.dt_stride_mtp + head]);
    dt_val += dt_bias_val;
    if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
    smem.dt_proc[lane] = dt_val;
  }
  // cumAdt = cumsum(A * dt_proc) — warp-local Hillis-Steele shuffle.  Each
  // of the 4 warps runs the same reduction on identical inputs (dt_proc
  // just written above) and writes the same smem.cumAdt slots.
  compute_cumAdt<NTOKENS>(smem, lane, A_val);

  // Commit this thread's cp.async group and wait for own completion.
  // __syncwarp() provides acquire semantics across the 32 lanes of each
  // warp — each lane now sees all its warp's cp.async writes.  No
  // cross-warp sync here; the only __syncthreads is after CB + replay.
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
}

// (compute_cumAdt moved above load_data so it can be called from there)

// Compute CB_scaled[T,T] = (C @ B^T) * decay * dt_proc * causal_mask.
// Split across 2 warps: warp 0 computes columns 0:8, warp 1 computes columns 8:16.
// Result stored to swizzled smem.CB_scaled (input_t, row stride 64, Swizzle<3,3,3>).
// Called between the two __syncthreads by warps 0 and 1 only.
template <typename input_t, typename mma_type, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_scaled_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;
  constexpr int N_HALF = 8;  // each warp computes [NTOKENS_PAD_MMA_M, 8]

  // CB_scaled output tile layout (used by both warp 0 compute and warp 1
  // zero-fill when smem.B has only 8 rows).
  auto layout_cb_swz =
      composition(Swizzle<3, 3, 3>{},
                  make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<NTOKENS_PAD_MMA_M>{}),
                              make_stride(Int<64>{}, _1{})));

  // ── NTOKENS_PAD_MMA_K == 8: warp 1 has no valid B rows to read.  But
  // CB_scaled[:, 8:16] must still be zero so matmul-4's K-reduction sees
  // zeros for k ≥ NTOKENS.  Do a simple 32-thread zero-fill and return. ──
  if constexpr (NTOKENS_PAD_MMA_K == 8) {
    if (warp == 1) {
      auto* __restrict__ cb = reinterpret_cast<mma_type*>(smem.CB_scaled);
      constexpr int COLS_TO_CLEAR = NTOKENS_PAD_MMA_M - N_HALF;  // 8
#pragma unroll
      for (int i = lane; i < NTOKENS_PAD_MMA_M * COLS_TO_CLEAR; i += warpSize) {
        int const r = i / COLS_TO_CLEAR;
        int const c = N_HALF + (i % COLS_TO_CLEAR);
        cb[layout_cb_swz(r, c)] = mma_type(0.f);
      }
      return;
    }
  }

  // Always bf16 MMA — activations are bf16; f16 state is converted in add_init_out_cute.
  using MmaAtomType = SM80_16x8x16_F32BF16BF16F32_TN;

  // ── Swizzled smem views ──
  // C is padded to NTOKENS_PAD_MMA_M; B has NTOKENS_PAD_MMA_K rows.  Use
  // NTOKENS_PAD_MMA_K for smem_B so the physical layout matches the write
  // layout from load_tile_async — `tile_to_shape` produces different outer
  // strides for (8, 128) vs (16, 128).
  auto layout_C = make_swizzled_layout_rc<NTOKENS_PAD_MMA_M, DSTATE>();
  auto layout_B = make_swizzled_layout_rc<NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor smem_C =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.C[0][0])), layout_C);
  Tensor smem_B =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.B[0][0])), layout_B);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  // ── K-tile A operand (C): full [NTOKENS_PAD_MMA_M, K_TILE], shared by both warps ──
  constexpr int K_TILE = 16;
  Tensor smem_C_tiled =
      local_tile(smem_C, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<K_TILE>{}), make_coord(_0{}, _));

  // ── K-tile B operand ──
  //   NTOKENS_PAD_MMA_K == 16: warp 0 → N=[0,8), warp 1 → N=[8,16).
  //   NTOKENS_PAD_MMA_K == 8 : only warp 0 runs (warp 1 took the early
  //     exit above), tile at (_0, _).
  Tensor smem_B_half =
      local_tile(smem_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp, _));

  // ── Register fragments ──
  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_B_half(_, _, _0{}));

  // ── Output accumulator: [NTOKENS_PAD_MMA_M, N_HALF] f32 ──
  auto layout_cb_half = make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_HALF>{}));
  Tensor frag_acc = thr_mma.partition_fragment_C(make_tensor((float*)nullptr, layout_cb_half));
  clear(frag_acc);

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(lane);
  Tensor smem_C_s2r = s2r_thr_A.partition_S(smem_C_tiled);
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(lane);
  Tensor smem_B_s2r = s2r_thr_B.partition_S(smem_B_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  // ── Gemm: 8 K-tiles, 1 HMMA each ──
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_B_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  // ── Elementwise: decay * dt_proc * causal mask, convert f32 → mma_type ──
  auto id_half = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_HALF>{}));
  auto id_part = thr_mma.partition_C(id_half);

  // ── Store to swizzled smem.CB_scaled ──
  Tensor smem_CB =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type*>(smem.CB_scaled)), layout_cb_swz);
  // Tile into [NTOKENS_PAD_MMA_M, N_HALF] halves; warp selects its half
  Tensor smem_CB_half = local_tile(smem_CB, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<N_HALF>{}),
                                   make_coord(_0{}, warp));
  Tensor smem_CB_part = thr_mma.partition_C(smem_CB_half);

#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    int t = get<0>(id_part(i));
    int j = warp * N_HALF + get<1>(id_part(i));
    float val;
    if (j <= t && t < NTOKENS && j < NTOKENS) {
      val = frag_acc(i) * __expf(smem.cumAdt[t] - smem.cumAdt[j]) * smem.dt_proc[j];
    } else {
      val = 0.f;
    }
    smem_CB_part(i) = mma_type(val);
  }
}

// =============================================================================
// Precompute dB scaling coefficients (called once before the N-tile loop).
// Returns DB_COEFFS_PER_LANE floats in coeff[], one per B-fragment element of
// the replay MMA — equal to (replay K) / 4 (each m16n8k* B-frag holds K*8/32
// elts per lane).
//   DB_COEFFS_PER_LANE = 4 for m16n8k16 B fragment
//   DB_COEFFS_PER_LANE = 2 for m16n8k8  B fragment
// coeff[i] = 0 when k >= prev_k, embedding the causal mask so the inner
// loop needs no branch.
//
// K-index derivation (row-major TN MMA, lane = tid % 32):
//   K_base = (lane % 4) * 2
//   m16n8k16 B frag (4 elts): 0→K_base, 1→K_base+1, 2→K_base+8, 3→K_base+9
//   m16n8k8  B frag (2 elts): 0→K_base, 1→K_base+1
// =============================================================================
template <int DB_COEFFS_PER_LANE, typename SmemT>
__device__ __forceinline__ void precompute_dB_coeff(float coeff[DB_COEFFS_PER_LANE],
                                                    SmemT const& smem, float total_cumAdt,
                                                    int prev_k, int lane) {
  static_assert(DB_COEFFS_PER_LANE == 2 || DB_COEFFS_PER_LANE == 4,
                "DB_COEFFS_PER_LANE must be 2 (k8) or 4 (k16)");
  int const K_base = (lane % 4) * 2;
  int k_idx[DB_COEFFS_PER_LANE];
  k_idx[0] = K_base;
  k_idx[1] = K_base + 1;
  if constexpr (DB_COEFFS_PER_LANE == 4) {
    k_idx[2] = K_base + 8;
    k_idx[3] = K_base + 9;
  }
#pragma unroll
  for (int i = 0; i < DB_COEFFS_PER_LANE; ++i) {
    int const k = k_idx[i];
    coeff[i] = (k < prev_k) ? __expf(total_cumAdt - smem.old_cumAdt[k]) * smem.old_dt_proc[k] : 0.f;
  }
}

// Apply precomputed dB coefficients to frag_B in-place.
// Handles input_t → mma_type conversion + scaling in one pass.
// coeff[i] = 0 encodes both causal mask and zero-fill for k >= prev_k.
// =============================================================================
template <int DB_COEFFS_PER_LANE, typename mma_type, typename FragB>
__device__ __forceinline__ void compute_dB_scaling(FragB& frag_B,
                                                   float const coeff[DB_COEFFS_PER_LANE]) {
  using namespace cute;
  static_assert(size(FragB{}) == DB_COEFFS_PER_LANE, "frag_B size must match DB_COEFFS_PER_LANE");
  using frag_t = typename FragB::value_type;
#pragma unroll
  for (int i = 0; i < DB_COEFFS_PER_LANE; ++i) {
    frag_B(i) = frag_t(toFloat(frag_B(i)) * coeff[i]);
  }
}

// =============================================================================
// Phase 1b: Replay — tensor-core MMA path.
// state[D, dstate] = state * total_decay + old_x^T @ (coeff * old_B)
// All 128 threads cooperate. TiledMMA covers [M=64, N=8] per step (4 atoms in M).
// A operand read directly from old_x via ldmatrix.trans (no transpose buffer).
// B operand read from old_B via ldmatrix.trans, scaled in registers by coeff[t].
// =============================================================================
template <typename input_t, int DIM, int DSTATE, typename SmemT>
__device__ __forceinline__ void replay_state_mma(SmemT& smem, int warp, int lane, int prev_k) {
  using namespace cute;
  using smem_state_t = typename SmemT::smem_state_t;
  static_assert(sizeof(smem_state_t) == 2, "smem state type must be 2-byte for MMA replay");
  static_assert(sizeof(input_t) == 2, "replay_state_mma requires 2-byte input type");
  static_assert(DIM == 64, "replay_state_mma requires DIM=64 for _4x1 MMA tiling");

  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Always bf16 MMA.  Atom K matches the token-axis tile (NTOKENS_PAD_MMA_K).
  //   K == 16 → m16n8k16 + x4/x2 ldmatrix.trans
  //   K == 8  → m16n8k8  + x2/x1 ldmatrix.trans
  using mma_type = __nv_bfloat16;
  using MmaAtomType = std::conditional_t<NTOKENS_PAD_MMA_K == 16, SM80_16x8x16_F32BF16BF16F32_TN,
                                         SM80_16x8x8_F32BF16BF16F32_TN>;
  using LdsmA = std::conditional_t<NTOKENS_PAD_MMA_K == 16, SM75_U16x8_LDSM_T, SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<NTOKENS_PAD_MMA_K == 16, SM75_U16x4_LDSM_T, SM75_U16x2_LDSM_T>;
  // # of dB coefficients each lane precomputes per replay K-tile = K * 8 / 32.
  constexpr int DB_COEFFS_PER_LANE = NTOKENS_PAD_MMA_K / 4;

  // TiledMMA: 128 threads, 4 atoms in M direction → covers [M=64, N=8]
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  float total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [NTOKENS_PAD_MMA_K, DIM] Swizzle<3,3,3>, transposed
  // view [M=DIM, K=NTOKENS_PAD_MMA_K]. ldmatrix.trans reads K-stride-1 smem
  // → K-stride-1 registers. ──
  auto layout_A = make_swizzled_layout_rc_transpose<NTOKENS_PAD_MMA_K, DIM>();
  Tensor smem_A =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.old_x[0][0])), layout_A);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, mma_type>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(
      make_tensor((mma_type*)0x0, make_shape(Int<DIM>{}, Int<NTOKENS_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == mma_type (bf16) — no conversion needed.

  // ── B operand: old_B [NTOKENS_PAD_MMA_K, DSTATE] Swizzle<3,3,3>, transposed
  // view [N=DSTATE, K=NTOKENS_PAD_MMA_K]. ──
  auto layout_B = make_swizzled_layout_rc_transpose<NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.old_B[0][0])), layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, mma_type>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: [DIM, DSTATE] swizzled ──
  auto layout_state_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  smem_state_t* state_base = reinterpret_cast<smem_state_t*>(&smem.state[0][0]);

  constexpr int N_TILE = 8;
  constexpr int NUM_N_TILES = DSTATE / N_TILE;

  // Identity tensor for C coordinates (covers full [DIM=64, N_TILE=8] output tile).
  // The replay C-frag per thread has 4 elts at (row_lo, col_off{,+1}) and
  // (row_hi=row_lo+8, col_off{,+1}).  Hoist the 2 unique row values and the
  // single col offset out of the N-tile loop — they don't depend on n.
  auto id_tile = make_identity_tensor(make_shape(Int<DIM>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  int const row_lo = get<0>(id_part(0));
  int const row_hi = get<0>(id_part(2));
  int const col_off = get<1>(id_part(0));

  // Precompute dB coefficients once — they depend only on K (lane), not on N.
  // coeff[i] = 0 when k ≥ prev_k (causal mask); since prev_k ≤ NTOKENS ≤
  // NTOKENS_PAD_MMA_K and we only read K < NTOKENS_PAD_MMA_K from smem, no
  // garbage feeds the MMA.
  float dB_coeff[DB_COEFFS_PER_LANE];
  precompute_dB_coeff<DB_COEFFS_PER_LANE>(dB_coeff, smem, total_cumAdt, prev_k, lane);

  // Process two N-tiles per iteration: interleave LDSM.T + dB_scaling across tiles
  // so PRMT has more latency cover from the other tile's loads.
  static_assert(NUM_N_TILES % 2 == 0, "NUM_N_TILES must be even for 2-wide N-tile loop");

#pragma unroll
  for (int n = 0; n < NUM_N_TILES; n += 2) {
    // Swizzled smem offsets for the 4 (row, col) positions this thread
    // owns in the 2-N-tile output.  Shared between the initial state read
    // and the post-MMA writeback to the same positions — 4 swizzle
    // evaluations per iter instead of 8.
    int const col0 = n * N_TILE + col_off;
    int const col1 = col0 + N_TILE;
    int const off_lo_0 = layout_state_swz(row_lo, col0);
    int const off_lo_1 = layout_state_swz(row_lo, col1);
    int const off_hi_0 = layout_state_swz(row_hi, col0);
    int const off_hi_1 = layout_state_swz(row_hi, col1);

    // ── Load state × total_decay into frag_h_{0,1} (accumulator init) ──
    Tensor frag_h_0 = thr_mma.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<DIM>{}, Int<N_TILE>{})));
    Tensor frag_h_1 = thr_mma.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<DIM>{}, Int<N_TILE>{})));

    using pair_t = Pair<smem_state_t>;
    int const offs[2][2] = {{off_lo_0, off_hi_0}, {off_lo_1, off_hi_1}};
    decltype(frag_h_0)* frag_h[2] = {&frag_h_0, &frag_h_1};

#pragma unroll
    for (int t = 0; t < 2; ++t) {
      auto& f = *frag_h[t];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        pair_t const p = *reinterpret_cast<pair_t const*>(&state_base[offs[t][h]]);
        cute::for_each(cute::make_seq<2>{}, [&](auto K) {
          f(h * 2 + decltype(K)::value) = toFloat(p[K]) * total_decay;
        });
      }
    }

    // ── LDSM.T both B tiles, then scale both — interleaved for latency cover ──
    Tensor smem_B_ntile_0 = local_tile(
        smem_B_full, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_K>{}), make_coord(n, _0{}));
    Tensor smem_B_ntile_1 = local_tile(
        smem_B_full, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_K>{}), make_coord(n + 1, _0{}));

    auto smem_B_s2r_0 = s2r_thr_B.partition_S(smem_B_ntile_0);
    auto smem_B_s2r_1 = s2r_thr_B.partition_S(smem_B_ntile_1);

    Tensor frag_B_0 = thr_mma.partition_fragment_B(
        make_tensor((mma_type*)0x0, make_shape(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_K>{})));
    Tensor frag_B_1 = make_fragment_like(frag_B_0);
    auto frag_B_view_0 = s2r_thr_B.retile_D(frag_B_0);
    auto frag_B_view_1 = s2r_thr_B.retile_D(frag_B_1);

    cute::copy(s2r_B, smem_B_s2r_0, frag_B_view_0);
    cute::copy(s2r_B, smem_B_s2r_1, frag_B_view_1);

    compute_dB_scaling<DB_COEFFS_PER_LANE, mma_type>(frag_B_0, dB_coeff);
    compute_dB_scaling<DB_COEFFS_PER_LANE, mma_type>(frag_B_1, dB_coeff);

    // ── Two independent HMMAs ──
    cute::gemm(tiled_mma, frag_h_0, frag_A, frag_B_0, frag_h_0);
    cute::gemm(tiled_mma, frag_h_1, frag_A, frag_B_1, frag_h_1);

    // ── Vectorized state store for both N-tiles (offsets reused from load) ──
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      auto const& f = *frag_h[t];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        pair_t const q = pack_float2<smem_state_t>(make_float2(f(h * 2), f(h * 2 + 1)));
        *reinterpret_cast<pair_t*>(&state_base[offs[t][h]]) = q;
      }
    }
  }
}

// ── CuTe mma.sync output sub-functions ──────────────────────────────────────
// Each operates on a register-resident frag_y accumulator (f32).
// Called from compute_output_cute's N-tile loop.

// Convert fragment elements from src_t to mma_type in-place.
// No-op when src_t == mma_type.  For the cross-dtype case: reads a src_t pair,
// converts via f32 intermediate, writes an mma_type pair.  `pack_float2`
// dispatches to the native packed cvt for the destination type (e.g.
// cvt.rn.bf16x2.f32 for bf16).
template <typename src_t, typename mma_type, typename Frag>
__device__ __forceinline__ void convert_frag(Frag& frag) {
  if constexpr (!std::is_same_v<src_t, mma_type>) {
#pragma unroll
    for (int i = 0; i < cute::size(frag); i += 2) {
      float2 const vals = toFloat2(reinterpret_cast<src_t const*>(&frag(i)));
      *reinterpret_cast<Pair<mma_type>*>(&frag(i)) = pack_float2<mma_type>(vals);
    }
  }
}

// 2b. frag_y += CB_scaled @ x  (matmul 4, single K-tile)
//     CB_scaled A operand loaded from swizzled smem via LDSM (precomputed by warps 0,1).
//     x B operand loaded from smem via ldmatrix.trans.
template <typename input_t, typename mma_type, int N_TILE, int NTOKENS_PAD_MMA_M, typename FragY,
          typename FragCB, typename SmemXTrans, typename S2RBTrans, typename S2RThrBTrans,
          typename ThrMma, typename TiledMma>
__device__ __forceinline__ void add_cb_x_cute(FragY& frag_y, FragCB const& frag_CB,
                                              SmemXTrans const& smem_x_trans,
                                              S2RBTrans const& s2r_B_trans,
                                              S2RThrBTrans const& s2r_thr_B_trans,
                                              ThrMma const& thr_mma, TiledMma const& tiled_mma,
                                              int n) {
  using namespace cute;
  Tensor smem_x_trans_ntile = local_tile(
      smem_x_trans, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_M>{}), make_coord(n, _0{}));
  auto smem_x_trans_s2r = s2r_thr_B_trans.partition_S(smem_x_trans_ntile);
  auto frag_B_x = thr_mma.partition_fragment_B(
      make_tensor((mma_type*)0x0, make_shape(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_M>{})));
  auto frag_B_x_view = s2r_thr_B_trans.retile_D(frag_B_x);

  cute::copy(s2r_B_trans, smem_x_trans_s2r, frag_B_x_view);
  // x is input_t == mma_type (bf16) — no conversion needed.

  cute::gemm(tiled_mma, frag_y, frag_CB, frag_B_x, frag_y);
}

// 3b. frag_y += D * x[t, d]  (per-thread skip connection via partition_C)
template <typename input_t, int NTOKENS_PAD_MMA_M, int N_TILE, typename FragY, typename SmemX,
          typename ThrMma>
__device__ __forceinline__ void add_D_skip_cute(FragY& frag_y, SmemX const& smem_x,
                                                ThrMma const& thr_mma, float D_val, int n) {
  using namespace cute;
  if (D_val == 0.f) return;
  Tensor smem_x_tile =
      local_tile(smem_x, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}), make_coord(_0{}, n));
  Tensor x_part = thr_mma.partition_C(smem_x_tile);
  // Load pairs of consecutive bf16 elements and convert via paired toFloat2.
  // m16n8k16 partition_C places consecutive N-column pairs adjacent in smem.
  static_assert(sizeof(input_t) == 2, "vectorized D_skip requires 2-byte input_t");
#pragma unroll
  for (int i = 0; i < size(frag_y); i += 2) {
    float2 vals = toFloat2(reinterpret_cast<input_t const*>(&x_part(i)));
    frag_y(i) += D_val * vals.x;
    frag_y(i + 1) += D_val * vals.y;
  }
}

// 4b. frag_y *= z * sigmoid(z)  (z-gating via partition_C)
template <typename input_t, int NTOKENS_PAD_MMA_M, int N_TILE, typename FragY, typename SmemZ,
          typename ThrMma>
__device__ __forceinline__ void compute_z_gating_cute(FragY& frag_y, SmemZ const& smem_z,
                                                      ThrMma const& thr_mma, void const* z_ptr,
                                                      int n) {
  using namespace cute;
  if (!z_ptr) return;
  Tensor smem_z_tile =
      local_tile(smem_z, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}), make_coord(_0{}, n));
  Tensor z_part = thr_mma.partition_C(smem_z_tile);
  static_assert(sizeof(input_t) == 2, "vectorized z-gating requires 2-byte input_t");
#pragma unroll
  for (int i = 0; i < size(frag_y); i += 2) {
    input_t pair[2];
    memcpy(pair, &z_part(i), 4);
    float z0 = toFloat(pair[0]);
    float z1 = toFloat(pair[1]);
    frag_y(i) *= z0 * __fdividef(1.f, (1.f + __expf(-z0)));
    frag_y(i + 1) *= z1 * __fdividef(1.f, (1.f + __expf(-z1)));
  }
}

// ── Matmul 3: init_out = C @ state^T ────────────────────────────────────────
// 3-stage software-pipelined K-loop with triple-buffered A and B register fragments.
// 3 frag_A + 6 frag_B in registers at steady state.
// Loads k+2 while HMMA consumes k — 2 iterations of latency cover between
// LDSM load and HMMA consume, reducing smem pipeline stalls.
template <typename input_t, int DIM, int DSTATE, typename SmemT, typename TiledMma, typename ThrMma,
          typename FragY>
__device__ __forceinline__ void add_init_out_cute(SmemT const& smem, TiledMma const& tiled_mma,
                                                  ThrMma const& thr_mma, int tid, FragY& frag_y_0,
                                                  FragY& frag_y_1) {
  using namespace cute;
  using smem_state_t = typename SmemT::smem_state_t;
  // Always bf16 MMA — f16 smem state converted via convert_frag<smem_state_t, mma_type>.
  using mma_type = __nv_bfloat16;

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int K_TILE = 16;
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr int N_TILE = 32;

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── Swizzled smem views ──
  auto layout_C_swz = make_swizzled_layout_rc<NTOKENS_PAD_MMA_M, DSTATE>();
  Tensor smem_C =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.C[0][0])), layout_C_swz);
  auto layout_state_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  Tensor smem_state = make_tensor(
      make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.state[0][0])), layout_state_swz);

  // ── K-tiled C (A operand) ──
  Tensor smem_C_ktiled =
      local_tile(smem_C, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<K_TILE>{}), make_coord(_0{}, _));
  auto smem_C_s2r = s2r_thr_A.partition_S(smem_C_ktiled);

  // ── State B operands for both N-tiles ──
  Tensor smem_state_nk_0 =
      local_tile(smem_state, make_tile(Int<N_TILE>{}, Int<K_TILE>{}), make_coord(0, _));
  Tensor smem_state_nk_1 =
      local_tile(smem_state, make_tile(Int<N_TILE>{}, Int<K_TILE>{}), make_coord(1, _));
  auto smem_state_s2r_0 = s2r_thr_B.partition_S(smem_state_nk_0);
  auto smem_state_s2r_1 = s2r_thr_B.partition_S(smem_state_nk_1);

  // ── Triple-buffered A fragments (3 K-slots) ──
  auto frag_A_0 = thr_mma.partition_fragment_A(smem_C_ktiled(_, _, _0{}));
  auto frag_A_1 = make_fragment_like(frag_A_0);
  auto frag_A_2 = make_fragment_like(frag_A_0);
  auto frag_A_view_0 = s2r_thr_A.retile_D(frag_A_0);
  auto frag_A_view_1 = s2r_thr_A.retile_D(frag_A_1);
  auto frag_A_view_2 = s2r_thr_A.retile_D(frag_A_2);

  // ── Triple-buffered B fragments (2 N-tiles × 3 K-slots = 6 total) ──
  auto frag_B0_0 = thr_mma.partition_fragment_B(smem_state_nk_0(_, _, _0{}));
  auto frag_B0_1 = make_fragment_like(frag_B0_0);
  auto frag_B0_2 = make_fragment_like(frag_B0_0);
  auto frag_B_view0_0 = s2r_thr_B.retile_D(frag_B0_0);
  auto frag_B_view0_1 = s2r_thr_B.retile_D(frag_B0_1);
  auto frag_B_view0_2 = s2r_thr_B.retile_D(frag_B0_2);

  auto frag_B1_0 = thr_mma.partition_fragment_B(smem_state_nk_1(_, _, _0{}));
  auto frag_B1_1 = make_fragment_like(frag_B1_0);
  auto frag_B1_2 = make_fragment_like(frag_B1_0);
  auto frag_B_view1_0 = s2r_thr_B.retile_D(frag_B1_0);
  auto frag_B_view1_1 = s2r_thr_B.retile_D(frag_B1_1);
  auto frag_B_view1_2 = s2r_thr_B.retile_D(frag_B1_2);

  // ── Clear accumulators ──
  clear(frag_y_0);
  clear(frag_y_1);

  // ── Prologue: load and convert k=0,1 into slots 0,1 ──
  cute::copy(s2r_A, smem_C_s2r(_, _, _, 0), frag_A_view_0);
  cute::copy(s2r_B, smem_state_s2r_0(_, _, _, 0), frag_B_view0_0);
  cute::copy(s2r_B, smem_state_s2r_1(_, _, _, 0), frag_B_view1_0);
  convert_frag<input_t, mma_type>(frag_A_0);
  convert_frag<smem_state_t, mma_type>(frag_B0_0);
  convert_frag<smem_state_t, mma_type>(frag_B1_0);

  cute::copy(s2r_A, smem_C_s2r(_, _, _, 1), frag_A_view_1);
  cute::copy(s2r_B, smem_state_s2r_0(_, _, _, 1), frag_B_view0_1);
  cute::copy(s2r_B, smem_state_s2r_1(_, _, _, 1), frag_B_view1_1);
  convert_frag<input_t, mma_type>(frag_A_1);
  convert_frag<smem_state_t, mma_type>(frag_B0_1);
  convert_frag<smem_state_t, mma_type>(frag_B1_1);

  // ── 3-stage pipelined K-loop: load k+2, gemm k, convert k+2 ──
  // Each slot has 2 iterations of latency cover between load and consume.
  // k%3=0: compute slot 0, load into slot 2
  // k%3=1: compute slot 1, load into slot 0
  // k%3=2: compute slot 2, load into slot 1
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    if (k % 3 == 0) {
      if (k + 2 < NUM_K_TILES) {
        cute::copy(s2r_A, smem_C_s2r(_, _, _, k + 2), frag_A_view_2);
        cute::copy(s2r_B, smem_state_s2r_0(_, _, _, k + 2), frag_B_view0_2);
        cute::copy(s2r_B, smem_state_s2r_1(_, _, _, k + 2), frag_B_view1_2);
      }
      cute::gemm(tiled_mma, frag_y_0, frag_A_0, frag_B0_0, frag_y_0);
      cute::gemm(tiled_mma, frag_y_1, frag_A_0, frag_B1_0, frag_y_1);
      if (k + 2 < NUM_K_TILES) {
        convert_frag<input_t, mma_type>(frag_A_2);
        convert_frag<smem_state_t, mma_type>(frag_B0_2);
        convert_frag<smem_state_t, mma_type>(frag_B1_2);
      }
    } else if (k % 3 == 1) {
      if (k + 2 < NUM_K_TILES) {
        cute::copy(s2r_A, smem_C_s2r(_, _, _, k + 2), frag_A_view_0);
        cute::copy(s2r_B, smem_state_s2r_0(_, _, _, k + 2), frag_B_view0_0);
        cute::copy(s2r_B, smem_state_s2r_1(_, _, _, k + 2), frag_B_view1_0);
      }
      cute::gemm(tiled_mma, frag_y_0, frag_A_1, frag_B0_1, frag_y_0);
      cute::gemm(tiled_mma, frag_y_1, frag_A_1, frag_B1_1, frag_y_1);
      if (k + 2 < NUM_K_TILES) {
        convert_frag<input_t, mma_type>(frag_A_0);
        convert_frag<smem_state_t, mma_type>(frag_B0_0);
        convert_frag<smem_state_t, mma_type>(frag_B1_0);
      }
    } else {  // k % 3 == 2
      if (k + 2 < NUM_K_TILES) {
        cute::copy(s2r_A, smem_C_s2r(_, _, _, k + 2), frag_A_view_1);
        cute::copy(s2r_B, smem_state_s2r_0(_, _, _, k + 2), frag_B_view0_1);
        cute::copy(s2r_B, smem_state_s2r_1(_, _, _, k + 2), frag_B_view1_1);
      }
      cute::gemm(tiled_mma, frag_y_0, frag_A_2, frag_B0_2, frag_y_0);
      cute::gemm(tiled_mma, frag_y_1, frag_A_2, frag_B1_2, frag_y_1);
      if (k + 2 < NUM_K_TILES) {
        convert_frag<input_t, mma_type>(frag_A_1);
        convert_frag<smem_state_t, mma_type>(frag_B0_1);
        convert_frag<smem_state_t, mma_type>(frag_B1_1);
      }
    }
  }
}

// store_state: vectorized smem → gmem state writeback (128 threads).
// Defined here (rather than alongside the other Phase 3 store helpers
// below) because compute_and_store_output_cute calls it inline for the
// v10.3 hoist — issued right after matmul 3 so the STGs fire-and-forget in
// parallel with matmul 4 + epilogue.  For 2-byte `state_t` the smem and
// gmem dtypes match → 128-bit UniversalCopy.  For f32 `state_t` the smem
// holds bf16, so we read bf16 pairs and widen to f32x2 on the way out.
template <typename state_t, int DIM, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void store_state(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int head, int64_t cache_slot) {
  using namespace cute;
  using smem_state_t = typename SmemT::smem_state_t;
  int const flat_tid = warp * warpSize + lane;
  auto* __restrict__ state_w = reinterpret_cast<state_t*>(params.state);
  int64_t const state_base = cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE;

  auto layout_smem_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  smem_state_t const* smem_state_base = reinterpret_cast<smem_state_t const*>(&smem.state[0][0]);

  if constexpr (std::is_same_v<state_t, smem_state_t>) {
    // 2-byte state_t: direct 128-bit vectorized copy from swizzled smem → gmem.
    Tensor sState = make_tensor(make_smem_ptr(smem_state_base), layout_smem_swz);
    Tensor gState = make_tensor(
        make_gmem_ptr(state_w + state_base),
        make_layout(make_shape(Int<DIM>{}, Int<DSTATE>{}), make_stride(Int<DSTATE>{}, Int<1>{})));
    auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, state_t>{},
                               Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
    auto thr = s2g.get_slice(flat_tid);
    copy(s2g, thr.partition_S(sState), thr.partition_D(gState));
  } else {
    // f32 state_t ⇐ bf16 smem: element-at-a-time widen + store.
    static_assert(sizeof(state_t) == 4, "unexpected state_t size");
    static_assert(std::is_same_v<smem_state_t, __nv_bfloat16>, "narrow path expects bf16 smem");
    constexpr int NUM_THREADS = NUM_WARPS * warpSize;
    constexpr int TOTAL = DIM * DSTATE;
    float* gmem_state = state_w + state_base;
    for (int i = flat_tid; i < TOTAL; i += NUM_THREADS) {
      int const row = i / DSTATE;
      int const col = i % DSTATE;
      gmem_state[row * DSTATE + col] = __bfloat162float(smem_state_base[layout_smem_swz(row, col)]);
    }
  }
}

// ── Orchestrator: compute_and_store_output_cute ─────────────────────────────
//     out = (C @ state^T) * decay + CB_scaled @ x + D*x, then z-gate.
//     All operations on register-resident frag_y — no smem round-trip.
//     Result converted f32 → input_t in registers and stored directly to gmem
//     via partition_C of the global output tensor (like CUTLASS sgemm_sm80 epilogue).
template <typename input_t, typename state_t, int NTOKENS, int DIM, int DSTATE, int NUM_WARPS,
          typename SmemT>
__device__ __forceinline__ void compute_and_store_output_cute(SmemT& smem,
                                                              SsuIncrementalParams const& params,
                                                              int warp, int lane, int batch_idx,
                                                              int head, int64_t cache_slot,
                                                              float D_val) {
  using namespace cute;
  static_assert(sizeof(typename SmemT::smem_state_t) == 2,
                "smem state type must be 2-byte for MMA output");
  static_assert(sizeof(input_t) == 2, "compute_and_store_output_cute requires 2-byte input type");

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // Always bf16 MMA — activations are bf16; f16 state converted in add_init_out_cute.
  using mma_type = __nv_bfloat16;
  using MmaAtomType = SM80_16x8x16_F32BF16BF16F32_TN;

  // ── TiledMMA: 128 threads, covers [16, 32] output per step ──
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  // x: swizzled [NTOKENS_PAD_MMA_M, DIM]
  auto layout_x_swz = make_swizzled_layout_rc<NTOKENS_PAD_MMA_M, DIM>();
  Tensor smem_x =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.x[0][0])), layout_x_swz);
  auto layout_x_trans_swz = make_swizzled_layout_rc_transpose<NTOKENS_PAD_MMA_M, DIM>();
  Tensor smem_x_trans = make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.x[0][0])),
                                    layout_x_trans_swz);

  // z: swizzled [NTOKENS_PAD_MMA_M, DIM]
  auto layout_z_swz = make_swizzled_layout_rc<NTOKENS_PAD_MMA_M, DIM>();
  Tensor smem_z =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.z[0][0])), layout_z_swz);

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);
  auto s2r_B_trans = make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, mma_type>{}, tiled_mma);
  auto s2r_thr_B_trans = s2r_B_trans.get_slice(tid);

  // ── Load CB_scaled A operand from smem (precomputed by warps 0,1 between syncs) ──
  auto layout_cb_swz =
      composition(Swizzle<3, 3, 3>{},
                  make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<NTOKENS_PAD_MMA_M>{}),
                              make_stride(Int<64>{}, _1{})));
  Tensor smem_CB =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // Decay broadcast: cumAdt[t] → [NTOKENS_PAD_MMA_M, N_TILE] with stride-0 on N
  constexpr int N_TILE = 32;
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output: partition_C for direct register → gmem store ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = (int64_t)batch_idx * params.out_stride_batch + (int64_t)head * DIM;

  // Row predicate for padding.  The epilogue store loop iterates i in steps
  // of 2 and only consults pred(0) and pred(2) — m16n8k16 C-frag per thread
  // has 4 elts at rows {t/4, t/4, t/4+8, t/4+8}, so there are only 2 unique
  // row predicates.  Compute them once and skip the 4-wide pred tensor.
  auto id_tile = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < NTOKENS;
  bool const pred_row_hi = get<0>(id_part(2)) < NTOKENS;

  // ── Matmul 3: init_out = C @ state^T (K-pipelined, see add_init_out_cute) ──
  Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
  Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
  add_init_out_cute<input_t, DIM, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0, frag_y_1);

  // v10.3 (Option A): state writeback hoisted here — after matmul 3 has
  // finished consuming smem.state, before matmul 4 which reads only
  // smem.x / smem.CB_scaled / smem.z.  ~16 KB of STGs fire-and-forget
  // onto the memory subsystem and complete in parallel with the
  // epilogue (matmul 4 + D*x + z-gate + output STG).
  store_state<state_t, DIM, DSTATE, NUM_WARPS>(smem, params, warp, lane, head, cache_slot);

  // ── Epilogue: decay + matmul 4 + D_skip + z-gate + store, per N-tile ──
  // Lambda to avoid duplicating the epilogue for each N-tile.
  auto epilogue = [&](auto& frag_y, int n) {
  // Decay: frag_y *= exp(cumAdt[t])
#pragma unroll
    for (int i = 0; i < size(frag_y); ++i) {
      frag_y(i) *= __expf(decay_part(i));
    }

    // frag_y += CB_scaled @ x (CB from smem LDSM, x from smem via ldmatrix.trans)
    add_cb_x_cute<input_t, mma_type, N_TILE, NTOKENS_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // frag_y += D * x[t, d]
    add_D_skip_cute<input_t, NTOKENS_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // frag_y *= z * sigmoid(z)
    compute_z_gating_cute<input_t, NTOKENS_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Store frag_y directly to gmem (register → gmem, no smem round-trip).
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}),
                                             make_stride(params.out_stride_mtp, _1{})));
    auto gOut_part = thr_mma.partition_C(gOut_tile);
    // Vectorized 32-bit store: elements i and i+1 are same-row, consecutive columns
    // in the m16n8k16 partition_C layout, so &gOut_part(i+1) == &gOut_part(i) + 1.
    // Address is 4-byte aligned because MMA column index = (lane%4)*2 → even.
    static_assert(sizeof(input_t) == 2, "vectorized output store requires 2-byte input_t");
#pragma unroll
    for (int i = 0; i < size(frag_y); i += 2) {
      // Bit 1 of i toggles between the two row groups of the m16n8k16
      // C-frag: i∈{0,1} → row t/4, i∈{2,3} → row t/4+8 (repeats per M-atom).
      bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
      if (pred_i) {
        input_t pair[2] = {input_t(frag_y(i)), input_t(frag_y(i + 1))};
        uint32_t packed;
        memcpy(&packed, pair, 4);
        *reinterpret_cast<uint32_t*>(&gOut_part(i)) = packed;
      }
    }
  };

  epilogue(frag_y_0, 0);
  epilogue(frag_y_1, 1);
}

// ── Store functions (called from kernel after compute_y + sync) ──
// (store_state moved above compute_and_store_output_cute — used there for
// the v10.3 state-writeback hoist.)

template <typename input_t, int NTOKENS, int DIM, typename SmemT>
__device__ __forceinline__ void store_old_x(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int head, int64_t cache_slot) {
  using namespace cute;
  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_x_w = reinterpret_cast<input_t*>(params.old_x);
  int64_t const ox_w_base = cache_slot * params.old_x_stride_cache + head * DIM;

  auto layout_x_swz = make_swizzled_layout_rc<NTOKENS_PAD_MMA_M, DIM>();
  Tensor sX =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.x[0][0])), layout_x_swz);
  Tensor gX = make_tensor(make_gmem_ptr(old_x_w + ox_w_base),
                          make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<DIM>{}),
                                      make_stride(params.old_x_stride_mtp, Int<1>{})));

  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsX = thr_s2g.partition_S(sX);
  auto tSgX = thr_s2g.partition_D(gX);

  if constexpr (NTOKENS == NTOKENS_PAD_MMA_M) {
    copy(s2g, tSsX, tSgX);
  } else {
    // All elements in a thread's partition share the same row (thread layout [16,8]).
    // Check the first element's row coordinate to predicate the entire copy.
    auto cX = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<DIM>{}));
    auto tScX = thr_s2g.partition_D(cX);
    if (get<0>(tScX(_0{})) < NTOKENS) {
      copy(s2g, tSsX, tSgX);
    }
  }
}

// v11.0: store_old_B runs on W0, W1 only (64 threads).  Caller must gate
// with `if (warp < 2)` — these are the warps that hold valid smem.B
// after their own cp.async + wait.  Halving the thread count keeps the
// v10.1 overlap (writeback fires before CB+replay consume smem.B).
//
// Thread layout `(8, 8) × (1, 8)` — **atom-aligned** with the Swizzle<3,3,3>
// (8, 64) atom for conflict-free smem reads.  Per-tile 8 × 64 covers one
// full atom.  For K=16: iters (2, 2) = 4 tiles, each thread owns 2 rows
// (t/8 and t/8+8) → per-iteration row predicate.  For K=8: iters (1, 2) = 2
// tiles, each thread owns 1 row.  The per-element predicate works for both.
template <typename input_t, int NTOKENS, int DSTATE, int HEADS_PER_GROUP, typename SmemT>
__device__ __forceinline__ void store_old_B(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int head, int group_idx,
                                            int64_t cache_slot, int buf_write) {
  using namespace cute;
  if (head % HEADS_PER_GROUP != 0) return;
  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;  // matches smem.B row count
  // Called only from warps 0, 1 — flat_tid ∈ [0, 64).
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_B_w = reinterpret_cast<input_t*>(params.old_B);
  int64_t const oB_base = cache_slot * params.old_B_stride_cache +
                          buf_write * params.old_B_stride_dbuf + group_idx * DSTATE;

  auto layout_B_swz = make_swizzled_layout_rc<NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor sB =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.B[0][0])), layout_B_swz);
  Tensor gB = make_tensor(make_gmem_ptr(old_B_w + oB_base),
                          make_layout(make_shape(Int<NTOKENS_PAD_MMA_K>{}, Int<DSTATE>{}),
                                      make_stride(params.old_B_stride_mtp, Int<1>{})));

  // 64 threads, (8, 8) × (1, 8) = atom-aligned per-tile (8, 64).
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_8, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsB = thr_s2g.partition_S(sB);
  auto tSgB = thr_s2g.partition_D(gB);

  if constexpr (NTOKENS == NTOKENS_PAD_MMA_K) {
    copy(s2g, tSsB, tSgB);
  } else {
    // Per-element predicate: for K=16 each thread owns 2 rows, so a
    // single `tScB(_0{})` check is insufficient.  copy_if masks each
    // iteration independently.
    auto cB = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_K>{}, Int<DSTATE>{}));
    auto tScB = thr_s2g.partition_D(cB);
    auto pred = make_tensor<bool>(shape(tScB));
    CUTE_UNROLL
    for (int i = 0; i < size(pred); ++i) {
      pred(i) = get<0>(tScB(i)) < NTOKENS;
    }
    copy_if(s2g, pred, tSsB, tSgB);
  }
}

// =============================================================================
// Kernel
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NTOKENS, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS>
__global__ void ssu_incremental_kernel(SsuIncrementalParams params) {
  using SmemT = SsuIncrementalStorage<input_t, state_t, NTOKENS, DIM, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  int const batch_idx = blockIdx.x;
  int const head = blockIdx.y;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const group_idx = head / HEADS_PER_GROUP;

  // ── Resolve cache slot ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[batch_idx]) : batch_idx;
  if (cache_slot == params.pad_slot_id) return;

  // ── Double-buffer index ──
  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);
  int const buf_write = 1 - buf_read;

  // ── prev_num_accepted_tokens ──
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  // ── Load A (scalar, tie_hdim), dt_bias, and D (hoisted to hide gmem latency) ──
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ════════════════════════════════════════════════════════════════════════
  // Phase 0: Load all data into smem (v11.0: per-warp ownership)
  // ════════════════════════════════════════════════════════════════════════
  // load_data ends with __pipeline_wait_prior(0) + __syncwarp() per warp.
  // No cross-warp sync here — every smem read before the post-replay
  // __syncthreads is served by data the *same warp* loaded.
  load_data<input_t, dt_t, state_t, NTOKENS, DIM, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, batch_idx, head, group_idx, cache_slot, buf_read, A_val,
      dt_bias_val);

  // v10.1: old_B writeback hoisted ahead of Phase 1.  Source (smem.B) is
  // consumed only by Phase 1a CB; the STGs fire-and-forget onto the
  // memory subsystem and complete in parallel with all subsequent compute.
  // v11.0: only W0, W1 hold valid smem.B at this point (they're the ones
  // that cp.async'd B).  Gate accordingly — store halves its thread
  // count but B is small (4 KB) so still cheap.
  if (warp < 2) {
    store_old_B<input_t, NTOKENS, DSTATE, HEADS_PER_GROUP>(smem, params, warp, lane, head,
                                                           group_idx, cache_slot, buf_write);
  }

  // Warps 0,1 compute CB_scaled into smem (split N=16 → 2×[16,8]).  No
  // barrier needed — warps 2,3 start replay immediately; warps 0,1 join
  // after CB.  Always bf16 MMA because activations are bf16.
  using mma_type = __nv_bfloat16;
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, mma_type, NTOKENS, DSTATE>(smem, warp, lane);
  }
  // Phase 1b: MMA replay (all 4 warps, independent M-rows).  Each warp
  // reads its own DIM slice of state + old_x (loaded into smem by this
  // same warp in load_data), plus smem.old_B (redundantly loaded by all
  // warps — each warp sees its own copy).
  replay_state_mma<input_t, DIM, DSTATE>(smem, warp, lane, prev_k);

  __syncthreads();

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: Output — y[t,d] = init_out + cb_out + D*x, then z-gate
  // ════════════════════════════════════════════════════════════════════════
  // D_val already loaded in preamble (gmem latency hidden behind Phase 0+1).
  //
  // Fused: matmul 3 + state-writeback + matmul 4 + D*x + z-gate → direct gmem store
  compute_and_store_output_cute<input_t, state_t, NTOKENS, DIM, DSTATE, NUM_WARPS>(
      smem, params, warp, lane, batch_idx, head, cache_slot, D_val);

  // ── Phase 3: Store to global memory ──
  // (old_B hoisted to pre-Phase-1 in v10.1;
  //  state hoisted into compute_and_store_output_cute in v10.3.)

  // Cache writes — old_x uses all warps (vectorized), dt/cumAdt one warp each
  store_old_x<input_t, NTOKENS, DIM>(smem, params, warp, lane, head, cache_slot);
  if (warp == 0 && lane < NTOKENS) {
    auto* __restrict__ old_dt_proc_w = reinterpret_cast<float*>(params.old_dt_proc);
    int64_t const dt_w_base = cache_slot * params.old_dt_proc_stride_cache +
                              buf_write * params.old_dt_proc_stride_dbuf +
                              head * params.old_dt_proc_stride_head;
    old_dt_proc_w[dt_w_base + lane] = smem.dt_proc[lane];
  }
  if (warp == 1 && lane < NTOKENS) {
    auto* __restrict__ old_cumAdt_w = reinterpret_cast<float*>(params.old_cumAdt);
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_cache +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + lane] = smem.cumAdt[lane];
  }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchSsuIncremental(SsuIncrementalParams& params, cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;

  FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                   ") must be divisible by ngroups (", params.ngroups, ")");

  // cp.async.ca with .L2::128B requires 16B-aligned pointers (128-bit / sizeof element).
  // The .L2::128B hint further requires the base address to be 128B-aligned for full
  // cache line utilization, but the hardware only faults on < 16B alignment.
  FLASHINFER_CHECK_ALIGNMENT(params.B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.C, 16);

  auto dispatch_hpg = [&]<int HEADS_PER_GROUP>() {
    auto func = ssu_incremental_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                       state_scale_t, NTOKENS_MTP, DIM, DSTATE, HEADS_PER_GROUP,
                                       PHILOX_ROUNDS, NUM_WARPS>;

    constexpr size_t smem_size =
        sizeof(SsuIncrementalStorage<input_t, state_t, NTOKENS_MTP, DIM, DSTATE>);

    dim3 grid(params.batch, params.nheads);
    dim3 block(warpSize, NUM_WARPS);

    if constexpr (smem_size > 0) {
      FLASHINFER_CUDA_CHECK(
          cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    func<<<grid, block, smem_size, stream>>>(params);
  };

  dispatchRatio(params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{},
                [&]<int HPG>() { dispatch_hpg.template operator()<HPG>(); });
}

}  // namespace flashinfer::mamba::incremental

#endif  // FLASHINFER_MAMBA_KERNEL_SSU_INCREMENTAL_CUH_
