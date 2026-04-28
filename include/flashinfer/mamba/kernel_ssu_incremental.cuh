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
// Phase 2: compute_and_store_output
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

// =============================================================================
// cp.async copy atoms
// =============================================================================
// 128-bit vector loads, shared by every gmem→smem copy in the kernel.  The
// ldmatrix unit has the same vector width so `vec_bytes` is derived from the
// atom's source-register type and reused as the LDSM vector width.
struct Copy_prop {
  using Atom = cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
  using AtomZFill = cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint128_t>;
  static constexpr int vec_bytes = sizeof(std::remove_extent_t<typename Atom::SRegisters>);
};

// =============================================================================
// MMA constants
// =============================================================================
// All MMA-related atom types, dtype, and dims for this kernel, grouped so the
// header doesn't sprinkle loose aliases.  The replay step chooses between the
// k=8 and k=16 atoms at compile time (NTOKENS ≤ 8 picks K8 for smaller smem,
// +1 CTA/SM); dims are pulled from MMA_Traits so they stay in sync with the
// atom choice (e.g. m16n8k32 for int8 would just need AtomK16/K8 swapped).
struct MMA_prop {
  using AtomK16 = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using AtomK8 = cute::SM80_16x8x8_F32BF16BF16F32_TN;
  // Operand dtype — matches the bf16 input of the atoms above.
  using operand_t = __nv_bfloat16;

  static constexpr int M = cute::size<0>(typename cute::MMA_Traits<AtomK16>::Shape_MNK{});
  static constexpr int N = cute::size<1>(typename cute::MMA_Traits<AtomK16>::Shape_MNK{});
  static constexpr int K_BIG = cute::size<2>(typename cute::MMA_Traits<AtomK16>::Shape_MNK{});
  static constexpr int K_SMALL = cute::size<2>(typename cute::MMA_Traits<AtomK8>::Shape_MNK{});
};

// =============================================================================
// Swizzled smem layout for mma.sync operands (row-major).
// =============================================================================
// The swizzle picks the `M` parameter to make each ldmatrix / cp.async atom
// exactly 16 bytes of contiguous element data (one 128-bit vector), and keeps
// B = S = 3 so that each 8-row block XORs row↔column bits to stay
// bank-conflict-free on the 128-byte bank cycle.
//
//   sizeof(T)  Swizzle<B,M,S>     atom rows × cols    row bytes
//     2B       Swizzle<3, 3, 3>       8  × 64             128
//     4B       Swizzle<3, 2, 3>       8  × 32             128
//     1B       Swizzle<3, 4, 3>       8  × 128            128
//
// The MMA operand element type dictates the smem buffer element type, which in
// turn dictates the swizzle — so every call site passes its own element type.
constexpr int log2_pow2(int x) {
  int r = 0;
  while (x > 1) {
    x >>= 1;
    ++r;
  }
  return r;
}

template <typename Elem>
struct SmemSwizzle {
  static_assert(Copy_prop::vec_bytes % sizeof(Elem) == 0,
                "element size must divide LDSM atom (16 bytes)");
  static constexpr int ELEMS_PER_ATOM = Copy_prop::vec_bytes / sizeof(Elem);
  using type = cute::Swizzle<3, log2_pow2(ELEMS_PER_ATOM), 3>;
  static constexpr int ATOM_ROWS = 1 << type::num_bits;
  static constexpr int ATOM_COLS = 1 << (type::num_base + type::num_shft);
};

// Default (ROW_STRIDE == COLS): tile the swizzle atom into a (ROWS, COLS)
// physical extent — the canonical CuTe pattern.
// Padded (ROW_STRIDE > COLS): logical (ROWS, COLS) view with the row stride
// inflated to ROW_STRIDE.  Used when COLS doesn't tile cleanly with the
// swizzle atom's col extent (e.g. CB_scaled: logical 16 cols, atom 64) but
// we want the atom-aligned bank pattern.  The extra cols-per-row are not
// "wasted padding" — the swizzle XOR scatters logical cells across the full
// ROW_STRIDE, so the physical extent is what the bijection actually needs.
template <typename Elem, int ROWS, int COLS, int ROW_STRIDE = COLS>
__device__ __forceinline__ auto make_swizzled_layout_rc() {
  using namespace cute;
  using S = SmemSwizzle<Elem>;
  static_assert(ROWS % S::ATOM_ROWS == 0, "ROWS must be a multiple of the swizzle atom rows");
  static_assert(ROW_STRIDE % S::ATOM_COLS == 0,
                "ROW_STRIDE must be a multiple of the swizzle atom cols");
  static_assert(ROW_STRIDE >= COLS, "ROW_STRIDE must be at least COLS");
  if constexpr (ROW_STRIDE == COLS) {
    auto atom = composition(typename S::type{},
                            make_layout(make_shape(Int<S::ATOM_ROWS>{}, Int<S::ATOM_COLS>{}),
                                        make_stride(Int<S::ATOM_COLS>{}, _1{})));
    return tile_to_shape(atom, make_shape(Int<ROWS>{}, Int<COLS>{}));
  } else {
    return composition(typename S::type{}, make_layout(make_shape(Int<ROWS>{}, Int<COLS>{}),
                                                       make_stride(Int<ROW_STRIDE>{}, _1{})));
  }
}

// Aliased-row swizzled smem layout: logical (LOGICAL_ROWS, COLS) view over a
// physical buffer sized to next_multiple_of<ATOM_ROWS>(VALID_ROWS) rows.
// When VALID_ROWS ≤ ATOM_ROWS the physical buffer is just one row-atom tall
// (e.g. 8 rows for bf16) but the MMA still wants to address LOGICAL_ROWS=16
// rows (m16n8k16's M).  We achieve the alias with a stride-0 outer mode on
// the row-tile axis: row r ∈ [0, LOGICAL_ROWS) maps to physical row
// (r mod PHYS_ROWS), col c maps unchanged.  The first m-tile carries the
// real C data; the second m-tile reads the same bytes, feeds garbage into
// MMA accumulator rows ≥ NTOKENS, predicated out at gmem store.
//
// When VALID_ROWS > ATOM_ROWS (NTOKENS > 8 here), PHYS_ROWS == LOGICAL_ROWS
// and the alias factor collapses to 1 — this then degenerates to the same
// layout that `make_swizzled_layout_rc<Elem, LOGICAL_ROWS, COLS>` produces.
template <typename Elem, int LOGICAL_ROWS, int COLS, int VALID_ROWS>
__device__ __forceinline__ auto make_aliased_swizzled_layout_rc() {
  using namespace cute;
  using S = SmemSwizzle<Elem>;
  static_assert(LOGICAL_ROWS % S::ATOM_ROWS == 0,
                "LOGICAL_ROWS must be a multiple of the swizzle atom rows");
  static_assert(COLS % S::ATOM_COLS == 0, "COLS must be a multiple of the swizzle atom cols");
  constexpr int PHYS_ROWS = next_multiple_of<S::ATOM_ROWS>(VALID_ROWS);
  constexpr int LOG_M_TILES = LOGICAL_ROWS / S::ATOM_ROWS;
  constexpr int PHYS_M_TILES = PHYS_ROWS / S::ATOM_ROWS;
  static_assert(LOG_M_TILES % PHYS_M_TILES == 0,
                "LOGICAL_ROWS must be a multiple of PHYS_ROWS for clean alias");
  constexpr int ALIAS = LOG_M_TILES / PHYS_M_TILES;
  constexpr int N_TILES = COLS / S::ATOM_COLS;
  auto atom = composition(typename S::type{},
                          make_layout(make_shape(Int<S::ATOM_ROWS>{}, Int<S::ATOM_COLS>{}),
                                      make_stride(Int<S::ATOM_COLS>{}, _1{})));
  // Outer layout (in atom-units): row-tile mode = (PHYS_M_TILES, ALIAS) strides
  // (1, 0); col-tile mode = N_TILES stride PHYS_M_TILES.  blocked_product
  // scales these by the atom cosize (= ATOM_ROWS * ATOM_COLS).
  auto outer =
      make_layout(make_shape(make_shape(Int<PHYS_M_TILES>{}, Int<ALIAS>{}), Int<N_TILES>{}),
                  make_stride(make_stride(_1{}, _0{}), Int<PHYS_M_TILES>{}));
  return blocked_product(atom, outer);
}

// Transposed swizzled smem layout: maps (col, row) → same physical offset as
// make_swizzled_layout_rc maps (row, col). Enables bank-conflict-free ldmatrix.trans
// reads on data stored with make_swizzled_layout_rc.
// Built by swapping modes of the original inner layout (before swizzle), which
// guarantees correct cross-atom offsets when both dimensions have multiple atoms.
template <typename Elem, int ROWS, int COLS>
__device__ __forceinline__ auto make_swizzled_layout_rc_transpose() {
  using namespace cute;
  using S = SmemSwizzle<Elem>;
  static_assert(ROWS % S::ATOM_ROWS == 0, "ROWS must be a multiple of the swizzle atom rows");
  static_assert(COLS % S::ATOM_COLS == 0, "COLS must be a multiple of the swizzle atom cols");
  // Build the inner (un-swizzled) tiled layout for the original (ROWS, COLS) layout
  auto inner = tile_to_shape(make_layout(make_shape(Int<S::ATOM_ROWS>{}, Int<S::ATOM_COLS>{}),
                                         make_stride(Int<S::ATOM_COLS>{}, _1{})),
                             make_shape(Int<ROWS>{}, Int<COLS>{}));
  // Swap modes to get true transpose: result(c, r) == original(r, c)
  auto inner_T = make_layout(get<1>(inner), get<0>(inner));
  return composition(typename S::type{}, inner_T);
}

// =============================================================================
// Shared memory layout
// =============================================================================
// smem holds the state in its native dtype (`state_t`).  cp.async pulls the
// native dtype straight into smem (with the matching `SmemSwizzle<state_t>`),
// and the conversion to `MMA_prop::operand_t` happens on the register read inside
// add_init_out / replay_state_mma.
// `D_PER_CTA` (v12 §59) is the per-CTA D dimension after D-split.
// For D_SPLIT = 1 (default), D_PER_CTA == DIM (per-head DIM).  At D_SPLIT > 1
// the storage is sliced: each CTA owns a contiguous D_PER_CTA-row slice of
// the head's D axis.  Buffers that aren't D-owned (B, C, old_B, scalars) are
// unaffected.
//
// Note on D_SMEM_COLS: the Swizzle<3,3,3> atom for bf16 is (8, 64), so
// `make_swizzled_layout_rc` requires col counts to be multiples of 64.  When
// D_PER_CTA < 64 (e.g. D_SPLIT=2 → D_PER_CTA=32) we pad the D-owned buffer
// cols up to the swizzle atom width.  The cp.async only fills the first
// D_PER_CTA cols; the padded tail is unused but keeps the swizzle layout
// well-formed.  Cost: 1 KB per [NTOKENS_PAD_MMA_M, 64] buffer at D_PER_CTA=32.
template <typename input_t, typename state_t, int NTOKENS_, int D_PER_CTA, int DSTATE>
struct SsuIncrementalStorage {
  // Re-export NTOKENS so helpers that only see SmemT can build NTOKENS-sized
  // shapes (e.g. aliased C layout).
  static constexpr int NTOKENS = NTOKENS_;
  // Swizzle atom width for input_t (= 64 cols for 2-byte types).
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  // M-dim of the output MMAs (C, x, z, CB_scaled): always m16-tiled.
  static constexpr int NTOKENS_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NTOKENS);
  // K-dim of the replay MMA (B, old_x, old_B).  Padded to the small atom's K
  // (== the LDSM unit for 2-byte elements).  For NTOKENS ≤ MMA_prop::K_SMALL the
  // replay uses the small atom (1 K-tile, smaller smem, +1 CTA/SM occupancy);
  // otherwise it uses the big atom (1 K-tile, fewer MMA ops).
  // Assumes NTOKENS ≤ MMA_prop::K_BIG (asserted in wrapper).
  static constexpr int NTOKENS_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(NTOKENS);
  // Row count for buffers padded only to the input-type swizzle atom's row
  // extent (8 for 2-byte, 4 for 4-byte) — used by C and z, which alias the
  // second m-tile back onto the first via `make_aliased_swizzled_layout_rc`.
  static constexpr int INPUT_SWIZZLE_ROWS =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NTOKENS);

  // All 2D smem buffers below are stored as flat 1D arrays — the actual
  // physical layout is determined by `make_swizzled_layout_rc<...>` at each
  // access site, which scrambles (row, col) → physical offset via the
  // Swizzle XOR.  Declaring them as `T[ROWS][COLS]` would falsely suggest a
  // row-major C-array layout that nobody ever uses; the only thing that
  // matters here is total byte count and 16-byte alignment.

  // CB_scaled — logical (NTOKENS_PAD_MMA_M, CB_ROW_STRIDE) Swizzle<3,3,3>.
  // CB_ROW_STRIDE pads each row to one bank cycle (128 B = 32 banks × 4 B)
  // worth of `input_t` so LDSM reads in matmul-4's A operand are
  // conflict-free.  Equals the swizzle atom's col extent for `input_t`
  // (64 for 2-byte, 32 for 4-byte).  Logical CB matrix is
  // (NTOKENS_PAD_MMA_M, NTOKENS_PAD_MMA_M); trailing cols are padding.
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;
  alignas(16) input_t CB_scaled[NTOKENS_PAD_MMA_M * CB_ROW_STRIDE];

  // B — logical (NTOKENS_PAD_MMA_K, DSTATE).  Padding rows inside
  // [NTOKENS, NTOKENS_PAD_MMA_K) contain garbage — valid output uses only
  // [0, NTOKENS).  Warp-1 of compute_CB_scaled_2warp reads rows ≥ 8 of a
  // 16-row view; those reads spill into C/old_B smem but are masked to 0 by
  // the (j < NTOKENS) CB-store predicate since j ≥ 8 ≥ NTOKENS when
  // NTOKENS_PAD_MMA_K == 8.
  alignas(16) input_t B[NTOKENS_PAD_MMA_K * DSTATE];

  // C — physical (next_multiple_of<ATOM_ROWS>(NTOKENS), DSTATE).  Padded only
  // to the swizzle atom's row extent (8 for 2-byte, 4 for 4-byte), not to
  // MMA_prop::M=16.  cp.async writes to this exact extent (CShape's first
  // dim shrunk to match — see load_data).  The MMA still views it as
  // NTOKENS_PAD_MMA_M=16 rows via `make_aliased_swizzled_layout_rc`, which
  // aliases the second m-tile back onto the first via stride-0 row-tile
  // mode.  Garbage feeds output rows ≥ NTOKENS — predicated out at gmem
  // store.  Saves up to 2 KB of smem at NTOKENS ≤ ATOM_ROWS (= MTP=8 in
  // practice), no-op when NTOKENS > ATOM_ROWS.
  alignas(16) input_t C[INPUT_SWIZZLE_ROWS * DSTATE];

  // x — logical (NTOKENS_PAD_MMA_M, D_SMEM_COLS).  Cols padded to
  // D_SMEM_COLS for swizzle atom alignment; cp.async only fills cols
  // [0, D_PER_CTA), the tail is unused.
  alignas(16) input_t x[NTOKENS_PAD_MMA_M * D_SMEM_COLS];

  // z — physical (next_multiple_of<ATOM_ROWS>(NTOKENS), D_SMEM_COLS).  Padded
  // only to the swizzle atom's row extent (8 for 2-byte, 4 for 4-byte), not
  // to MMA_prop::M=16.  z is never an MMA operand — the z-gating epilogue
  // reads it via `partition_C` of the m16n8 c-frag, so the MMA still views
  // it as NTOKENS_PAD_MMA_M=16 rows via `make_aliased_swizzled_layout_rc`,
  // which aliases the second m-tile back onto the first via stride-0
  // row-tile mode.  Garbage feeds output rows ≥ NTOKENS — predicated out at
  // gmem store.  Saves up to 1 KB of smem at NTOKENS ≤ ATOM_ROWS, no-op
  // when NTOKENS > ATOM_ROWS.
  alignas(16) input_t z[INPUT_SWIZZLE_ROWS * D_SMEM_COLS];

  // Old cache data loaded in Phase 0 (consumed in Phase 1 replay).
  // old_x — logical (NTOKENS_PAD_MMA_K, D_SMEM_COLS); ldmatrix.trans feeds
  // replay MMA A-operand (only the first D_PER_CTA cols are valid data).
  alignas(16) input_t old_x[NTOKENS_PAD_MMA_K * D_SMEM_COLS];

  // old_B — logical (NTOKENS_PAD_MMA_K, DSTATE) Swizzle<3,3,3>.  Replay MMA
  // reads via ldmatrix.trans (LDSM_T) + register scaling.  Padding rows
  // zero-filled via cp.async ZFILL.
  alignas(16) input_t old_B[NTOKENS_PAD_MMA_K * DSTATE];

  float old_dt_proc[NTOKENS];
  float old_cumAdt[NTOKENS];

  // Processed dt for new tokens (Phase 1a uses this for CB_scaled + cumAdt)
  float dt_proc[NTOKENS];

  // Cumulative A*dt — computed once by warp 0, read by all warps after sync
  float cumAdt[NTOKENS];

  // state — logical (D_PER_CTA, DSTATE) in `state_t` (native dtype).  The
  // MMA path reinterprets 2-byte state as bf16 for LDSM; f32 state is loaded
  // via UniversalCopy and converted to bf16 in registers inside
  // add_init_out.
  alignas(16) state_t state[D_PER_CTA * DSTATE];
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
  constexpr int VALID_COLS = size<1>(SmemShape{});
  // Smem cols are padded up to the swizzle atom width.  We always use the
  // wide thread layout (4 thread-rows × 8 thread-cols × 1×8 val = 4 rows ×
  // 64 cols/pass for bf16) so each thread-row covers all 8 vec-cols of the
  // Swizzle<B,M,3> atom — the design contract that makes cp.async writes
  // bank-conflict-free (each row consumes one full bank cycle, rows
  // serialize across cycles).  A "narrow" layout (½-atom-width per row)
  // would force adjacent rows to compete for the same 16 banks, costing
  // ~3-4× replay (observed as 12-way LDGSTS conflicts in d_split=2 ncu).
  // When VALID_COLS < SMEM_COLS (D_SPLIT > 1 path), cp.async ZFILL drops
  // the predicated-out cols as zeros without touching gmem — same mechanism
  // as ZFILL'ing rows ≥ VALID_ROWS.  The padded smem cells are unused.
  constexpr int SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(VALID_COLS);
  Tensor s_full =
      make_tensor(make_smem_ptr(smem_dst), make_swizzled_layout_rc<input_t, ROWS_PAD, SMEM_COLS>());
  Tensor g_full = make_tensor(make_gmem_ptr(gmem_src),
                              make_layout(make_shape(Int<ROWS_PAD>{}, Int<SMEM_COLS>{}),
                                          make_stride(gmem_row_stride, Int<1>{})));

  constexpr int VAL_COLS_PER_THREAD = Copy_prop::vec_bytes / sizeof(input_t);
  static_assert(SMEM_COLS % VAL_COLS_PER_THREAD == 0,
                "SMEM_COLS must be divisible by VAL_COLS_PER_THREAD");
  using ThrLayout = Layout<Shape<_4, _8>, Stride<_8, _1>>;
  static_assert(size<1>(ThrLayout{}) * VAL_COLS_PER_THREAD == SmemSwizzle<input_t>::ATOM_COLS,
                "wide thread layout must cover one full swizzle atom width per row");
  auto g2s = make_tiled_copy(Copy_Atom<Copy_prop::AtomZFill, input_t>{}, ThrLayout{},
                             Layout<Shape<_1, Int<VAL_COLS_PER_THREAD>>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<ROWS_PAD>{}, Int<SMEM_COLS>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = (get<0>(thr_id(i)) < VALID_ROWS) && (get<1>(thr_id(i)) < VALID_COLS);
  }
  copy_if(g2s, pred, thr.partition_S(g_full), thr.partition_D(s_full));
}

// State load — D_SPLIT-conditional dispatch (v12.2):
//
//   D_SPLIT == 1: per-warp partition (warp W loads rows
//     [W*DIM/4 : (W+1)*DIM/4)).  v11.0's pattern — large coalesced gmem
//     reads per warp.  Tests pass without an extra CTA-wide barrier
//     because the post-replay __syncthreads covers the eventual
//     cross-warp state reads.
//
//   D_SPLIT >= 2: 128-thread cooperative load.  Required because at
//     D_PER_CTA = 16 / 4 = 4 D-rows per warp the per-warp layout no
//     longer divides cleanly into the (4, 8) thread-tile atom that
//     `Copy_prop::Atom` expects.  Cooperative load works for any
//     D_PER_CTA ∈ {DIM, DIM/2, DIM/4} that's a multiple of 16.
//
// Both variants write through `make_swizzled_layout_rc<state_t, DIM, DSTATE>`
// followed by a `local_tile` to the (D_PER_CTA, DSTATE) slice this CTA
// owns — the swizzle outer-stride is invariant across D_SPLIT.
template <typename state_t, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_per_warp(SmemT& smem,
                                                    state_t const* __restrict__ state_ptr,
                                                    int64_t state_base, int warp, int lane) {
  using namespace cute;
  static_assert(NUM_WARPS == 4, "Expected 4 warps");
  static_assert(D_PER_CTA % NUM_WARPS == 0, "D_PER_CTA must be divisible by NUM_WARPS");
  constexpr int DIM_PER_WARP = D_PER_CTA / NUM_WARPS;

  // v11.0 single-local_tile path — swizzle layout sized to this CTA's
  // D_PER_CTA slice; one local_tile splits it directly per-warp.
  Tensor sState_full = make_tensor(make_smem_ptr(reinterpret_cast<state_t*>(smem.state)),
                                   make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>());
  Tensor gState_full = make_tensor(make_gmem_ptr(state_ptr + state_base),
                                   make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                               make_stride(Int<DSTATE>{}, Int<1>{})));

  Tensor sState = local_tile(sState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));
  Tensor gState = local_tile(gState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));

  constexpr int VAL_COLS = Copy_prop::vec_bytes / sizeof(state_t);
  auto g2s =
      make_tiled_copy(Copy_Atom<Copy_prop::Atom, state_t>{},
                      Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = g2s.get_slice(lane);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
}

template <typename state_t, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_cta(SmemT& smem, state_t const* __restrict__ state_ptr,
                                               int64_t state_base, int tid) {
  using namespace cute;
  static_assert(NUM_WARPS == 4, "Expected 4 warps");

  Tensor sState = make_tensor(make_smem_ptr(reinterpret_cast<state_t*>(smem.state)),
                              make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>());
  Tensor gState = make_tensor(make_gmem_ptr(state_ptr + state_base),
                              make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                          make_stride(Int<DSTATE>{}, Int<1>{})));

  constexpr int VAL_COLS = Copy_prop::vec_bytes / sizeof(state_t);
  using ThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  constexpr int THR_ROWS = decltype(size<0>(ThrLayout{}))::value;
  static_assert(D_PER_CTA % THR_ROWS == 0,
                "D_PER_CTA must be divisible by the thread layout's row count");
  auto g2s = make_tiled_copy(Copy_Atom<Copy_prop::Atom, state_t>{}, ThrLayout{},
                             Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = g2s.get_slice(tid);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
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
template <typename input_t, typename dt_t, typename state_t, int NTOKENS, int DIM, int D_PER_CTA,
          int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_data(SmemT& smem, SsuIncrementalParams const& params, int lane,
                                          int warp, int d_tile, int batch_idx, int head,
                                          int group_idx, int64_t cache_slot, int buf_read,
                                          float A_val, float dt_bias_val) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(D_PER_CTA % INPUT_PACK == 0, "D_PER_CTA must be divisible by input pack size");

  // v12 §59: D-owned tensors (x, z, old_x, state) sliced by d_tile along D.
  // The d_tile_off is added to the per-head gmem base to land on the CTA's
  // D-slice; SmemT's D-owned buffers are sized to D_PER_CTA so the cp.async
  // load fills the slice exactly.  Non-D-owned (B, C, old_B, scalars) are
  // unchanged — every CTA loads them redundantly (L2 covers).
  int const d_tile_off = d_tile * D_PER_CTA;

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
  int64_t const x_base = (int64_t)batch_idx * params.x_stride_batch + head * DIM + d_tile_off;
  int64_t const ox_base = cache_slot * params.old_x_stride_cache + head * DIM + d_tile_off;
  int64_t const oB_base = cache_slot * params.old_B_stride_cache +
                          buf_read * params.old_B_stride_dbuf + group_idx * DSTATE;

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;
  // CShape: first dim shrunk to the swizzle atom's row extent so cp.async
  // writes don't spill past the (also shrunk) C smem buffer.  When NTOKENS
  // > ATOM_ROWS this falls back to NTOKENS_PAD_MMA_M.
  using CShape = cute::Shape<cute::Int<SmemT::INPUT_SWIZZLE_ROWS>, cute::Int<DSTATE>>;
  using BShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_K>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // ZShape: same shrink as CShape — z is read via partition_C alias, so the
  // physical buffer only needs to be one swizzle row-atom tall.
  using ZShape = cute::Shape<cute::Int<SmemT::INPUT_SWIZZLE_ROWS>, cute::Int<D_PER_CTA>>;
  using OxShape = cute::Shape<cute::Int<NTOKENS_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

  // ── State: per-CTA D-slice ([D_PER_CTA, DSTATE]).  Dispatch on D_SPLIT
  // (= DIM == D_PER_CTA): per-warp coalesced load when one CTA owns the
  // full head's D, cooperative 128-thread load when D is sharded. ──
  {
    auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
    int64_t const state_base = cache_slot * params.state_stride_batch +
                               (int64_t)head * DIM * DSTATE + (int64_t)d_tile_off * DSTATE;
    if constexpr (DIM == D_PER_CTA) {
      // D_SPLIT=1: per-warp partition (warp w loads contiguous DIM/4 D-rows).
      load_state_per_warp<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp,
                                                                 lane);
    } else {
      // D_SPLIT>=2: 128-thread cooperative load (per-warp doesn't divide
      // cleanly when D_PER_CTA/4 is too small for the (4,8) thread atom).
      int const tid = warp * warpSize + lane;
      load_state_cta<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, tid);
    }
  }

  // ── B + C: redundant on W0, W1 (both do 2-warp CB compute) ──
  if (warp < 2) {
    load_tile_async<BShape, NTOKENS>(smem.B, B_ptr + B_base, params.B_stride_mtp, lane);
    load_tile_async<CShape, NTOKENS>(smem.C, C_ptr + C_base, params.C_stride_mtp, lane);
  }

  // ── old_B: redundant on all 4 warps (each warp's replay consumes full
  // DSTATE).  Identical payloads to same smem dest — final bytes
  // deterministic. ──
  load_tile_async<BShape, NTOKENS>(smem.old_B, old_B_ptr + oB_base, params.old_B_stride_mtp, lane);

  // ── old_x: redundant on all 4 warps (small, simpler than partitioning). ──
  load_tile_async<OxShape, NTOKENS>(smem.old_x, old_x_ptr + ox_base, params.old_x_stride_mtp, lane);

  // ── x: W2 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 2) {
    load_tile_async<XShape, NTOKENS>(smem.x, x_ptr + x_base, params.x_stride_mtp, lane);
  }

  // ── z: W3 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 3 && z_ptr) {
    int64_t const z_base = (int64_t)batch_idx * params.z_stride_batch + head * DIM + d_tile_off;
    load_tile_async<ZShape, NTOKENS>(smem.z, z_ptr + z_base, params.z_stride_mtp, lane);
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
template <typename input_t, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_scaled_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;
  // 2-warp output split: each warp owns NTOKENS_PAD_MMA_M / 2 cols of the
  // (NTOKENS_PAD_MMA_M, NTOKENS_PAD_MMA_M) CB tile.  Must be a multiple of
  // the MMA atom's N for the partition to be atom-aligned (currently 8 ==
  // MMA_prop::N for NTOKENS_PAD_MMA_M=16; if M-pad ever grows, this still
  // holds as long as M-pad is a multiple of 2 * MMA_prop::N).
  constexpr int N_HALF = NTOKENS_PAD_MMA_M / 2;
  static_assert(N_HALF % MMA_prop::N == 0,
                "compute_CB_scaled_2warp: NTOKENS_PAD_MMA_M / 2 must be a multiple of MMA::N");

  // CB_scaled output tile layout (used by both warp 0 compute and warp 1
  // zero-fill when smem.B has only 8 rows).  Row stride matches the buffer's
  // padded width (one swizzle atom of `input_t`).
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  auto layout_cb_swz =
      make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, NTOKENS_PAD_MMA_M, CB_ROW_STRIDE>();

  // ── NTOKENS_PAD_MMA_K == 8: warp 1 has no valid B rows to read.  But
  // CB_scaled[:, 8:16] must still be zero so matmul-4's K-reduction sees
  // zeros for k ≥ NTOKENS.  Do a simple 32-thread zero-fill and return. ──
  if constexpr (NTOKENS_PAD_MMA_K == 8) {
    if (warp == 1) {
      auto* __restrict__ cb = reinterpret_cast<MMA_prop::operand_t*>(smem.CB_scaled);
      constexpr int COLS_TO_CLEAR = NTOKENS_PAD_MMA_M - N_HALF;  // 8
#pragma unroll
      for (int i = lane; i < NTOKENS_PAD_MMA_M * COLS_TO_CLEAR; i += warpSize) {
        int const r = i / COLS_TO_CLEAR;
        int const c = N_HALF + (i % COLS_TO_CLEAR);
        cb[layout_cb_swz(r, c)] = MMA_prop::operand_t(0.f);
      }
      return;
    }
  }

  // ── Swizzled smem views ──
  // C is padded to NTOKENS_PAD_MMA_M; B has NTOKENS_PAD_MMA_K rows.  Use
  // NTOKENS_PAD_MMA_K for smem_B so the physical layout matches the write
  // layout from load_tile_async — `tile_to_shape` produces different outer
  // strides for (8, 128) vs (16, 128).
  // Aliased C view: physical buffer is just next_multiple_of<ATOM_ROWS>(NTOKENS)
  // rows tall but the MMA atom needs M=16; second m-tile aliases first m-tile
  // (predicated rows discarded at output store).
  auto layout_C = make_aliased_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, DSTATE, NTOKENS>();
  auto layout_B = make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_B = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  // ── K-tile A operand (C): full [NTOKENS_PAD_MMA_M, K_TILE], shared by both warps ──
  constexpr int K_TILE = MMA_prop::K_BIG;
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

  // ── Elementwise: decay * dt_proc * causal mask, convert f32 → MMA_prop::operand_t ──
  auto id_half = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_HALF>{}));
  auto id_part = thr_mma.partition_C(id_half);

  // ── Store to swizzled smem.CB_scaled ──
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t*>(smem.CB_scaled)), layout_cb_swz);
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
    smem_CB_part(i) = MMA_prop::operand_t(val);
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
// Scales frag_B in-place by per-coefficient multiplier; frag dtype inferred
// from FragB::value_type.
// coeff[i] = 0 encodes both causal mask and zero-fill for k >= prev_k.
// =============================================================================
template <int DB_COEFFS_PER_LANE, typename FragB>
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
// Phase 1b: Replay — tensor-core MMA path (matmul 2: state recurrence).
// state[D, dstate] = state * total_decay + old_x^T @ (coeff * old_B)
// All 128 threads cooperate.
//
// v12 prep (warps along N=DSTATE; was M=DIM in v11.x):
//   TiledMMA uses Layout<_1, _4> — per pass covers (M=DIM, N=4×MMA_prop::N=32).
//   Each warp owns: full M (DIM/16 m-atoms) and one n-atom of 8 cols.
//   Why: A is small (DIM × K), B is bigger (DSTATE × K).  M-split (`_4×1`)
//   redundantly loaded full B from each warp (4× × 4 KB = 16 KB).  N-split
//   (`_1×4`) instead redundantly loads full A (4× × 2 KB = 8 KB) and reads
//   B disjointly across warps — net smem read drops 18 KB → 12 KB per replay
//   (~33%) at K_BIG.  Also unlocks D-split D_PER_CTA < 64 (planned next).
// =============================================================================
// state_gmem_off (f16+philox path): full gmem element offset to this CTA's
// owned [D_PER_CTA, DSTATE] state slice (cache_slot * state_stride_batch +
// head * DIM*DSTATE + d_tile * D_PER_CTA*DSTATE).  Computed in the kernel
// preamble and threaded through; replay adds row*DSTATE + col per pair.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          typename SmemT>
__device__ __forceinline__ void replay_state_mma(SmemT& smem, SsuIncrementalParams const& params,
                                                 int warp, int lane, int prev_k, int d_tile,
                                                 uint32_t state_ptr_offset, int64_t state_gmem_off,
                                                 int64_t rand_seed) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma requires 2-byte input type");
  static_assert(D_PER_CTA % 16 == 0, "D_PER_CTA must be divisible by 16 (m16n8 atom)");
  static_assert(D_PER_CTA >= 16, "D_PER_CTA must be at least 16");

  constexpr int NTOKENS_PAD_MMA_K = SmemT::NTOKENS_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Atom K matches the token-axis tile (NTOKENS_PAD_MMA_K).
  //   K == MMA_prop::K_BIG   (16) → m16n8k16 + x4/x2 ldmatrix.trans
  //   K == MMA_prop::K_SMALL (8)  → m16n8k8  + x2/x1 ldmatrix.trans
  using MmaAtomType =
      std::conditional_t<NTOKENS_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16, MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<NTOKENS_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<NTOKENS_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // 4 warps along N=DSTATE; each warp covers full M (D_PER_CTA/16 m-atoms).
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Per-pass output tile is (D_PER_CTA, N_PER_PASS).  N_PER_PASS = 4 warps × n8 = 32 cols.
  constexpr int N_PER_PASS = 4 * MMA_prop::N;
  static_assert(DSTATE % N_PER_PASS == 0,
                "DSTATE must be divisible by 4 * MMA_prop::N for _1x4 warp layout");
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;

  float total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [NTOKENS_PAD_MMA_K, D_SMEM_COLS] Swizzle<3,3,3>, transposed
  // view [M=D_SMEM_COLS, K=NTOKENS_PAD_MMA_K].  D_SMEM_COLS may be padded above
  // D_PER_CTA when D_PER_CTA < swizzle atom (v12 §59); local_tile to D_PER_CTA
  // restricts the LDSM to the valid sub-tile.  Each warp loads the FULL M (4×
  // redundant across warps).  See header comment for traffic accounting. ──
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full = make_swizzled_layout_rc_transpose<input_t, NTOKENS_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<NTOKENS_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<NTOKENS_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == MMA_prop::operand_t (bf16) — no conversion needed.

  // ── B operand: old_B [NTOKENS_PAD_MMA_K, DSTATE] swizzled, transposed view
  // [N=DSTATE, K=NTOKENS_PAD_MMA_K].  Per pass loads N_PER_PASS=32 cols across
  // 4 warps; partition_S splits — each warp gets its disjoint 8-col slice. ──
  auto layout_B = make_swizzled_layout_rc_transpose<input_t, NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: per-CTA swizzle layout [D_PER_CTA, DSTATE]. ──
  auto layout_state_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  state_t* state_base = reinterpret_cast<state_t*>(smem.state);

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
  constexpr int DB_COEFFS_PER_LANE = NTOKENS_PAD_MMA_K / LANES_PER_N_COL;
  float dB_coeff[DB_COEFFS_PER_LANE];
  precompute_dB_coeff<DB_COEFFS_PER_LANE>(dB_coeff, smem, total_cumAdt, prev_k, lane);

  using pair_t = Pair<state_t>;

#pragma unroll
  for (int n = 0; n < NUM_N_PASSES; ++n) {
    int const n_base = n * N_PER_PASS;

    // ── Allocate per-pass C-frag (4 × M_atoms fp32 elts/thread; M_atoms = D_PER_CTA/16) ──
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
    Tensor smem_B_n = local_tile(
        smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<NTOKENS_PAD_MMA_K>{}), make_coord(n, _0{}));
    auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);

    Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<NTOKENS_PAD_MMA_K>{})));
    auto frag_B_view = s2r_thr_B.retile_D(frag_B);

    cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

    compute_dB_scaling<DB_COEFFS_PER_LANE>(frag_B, dB_coeff);

    // ── HMMA: frag_h += frag_A @ frag_B (D_PER_CTA/16 m-atoms × 1 n-atom HMMAs) ──
    cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

    // ── Vectorized state writeback ──
    // smem always gets nearest-even f32→state_t (consumed by matmul 3 — must
    // match Triton's f32→bf16 path as closely as possible).  Gmem cache, when
    // PHILOX_ROUNDS > 0 and state_t == __half, gets PTX cvt.rs.f16x2.f32
    // stochastic rounding direct from registers via STG.32, bypassing the
    // smem→gmem `store_state` copy.  Offset matches Triton's replay reference:
    // per (m, col_pair) take 4 randints from one philox_randint4x at offset
    // base + m*DSTATE + (col_base & ~3).  The tl.join interleave selects r0
    // for col_base%4==0 and r2 for col_base%4==2 (r1, r3 are discarded).
    constexpr bool kPhiloxF16 = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;
    [[maybe_unused]] auto* __restrict__ state_w = reinterpret_cast<state_t*>(params.state);
#pragma unroll
    for (int i = 0; i < size(frag_h); i += 2) {
      int const row = get<0>(id_part(i));
      int const col = get<1>(id_part(i)) + n_base;
      int const off = layout_state_swz(row, col);

      // Smem write — always nearest-even (output's matmul 3 reads this).
      pair_t const q = pack_float2<state_t>(make_float2(frag_h(i), frag_h(i + 1)));
      *reinterpret_cast<pair_t*>(&state_base[off]) = q;

      // Gmem write — SR f16 from registers (philox path only).  Skips the
      // smem→gmem `store_state` (gated off in compute_and_store_output).
      if constexpr (kPhiloxF16) {
        uint32_t const philox_off = state_ptr_offset +
                                    static_cast<uint32_t>(d_tile * D_PER_CTA + row) * DSTATE +
                                    static_cast<uint32_t>(col & ~3);
        uint32_t r0, r1, r2, r3;
        conversion::philox_randint4x<PHILOX_ROUNDS>(rand_seed, philox_off, r0, r1, r2, r3);
        uint32_t const rbits = (col & 2) ? r2 : r0;
        uint32_t const packed = conversion::cvt_rs_f16x2_f32(frag_h(i), frag_h(i + 1), rbits);
        int64_t const gmem_off = state_gmem_off + (int64_t)row * DSTATE + col;
        *reinterpret_cast<uint32_t*>(&state_w[gmem_off]) = packed;
      }
    }
  }
}

// ── CuTe mma.sync output sub-functions ──────────────────────────────────────
// Each operates on a register-resident frag_y accumulator (f32).
// Called from compute_and_store_output's N-tile loop.

// Convert fragment elements from src_t to MmaT in-place.
// No-op when src_t == MmaT.  For the cross-dtype case: reads a src_t pair,
// converts via f32 intermediate, writes an MmaT pair.  `pack_float2`
// dispatches to the native packed cvt for the destination type (e.g.
// cvt.rn.bf16x2.f32 for bf16).
template <typename src_t, typename MmaT, typename Frag>
__device__ __forceinline__ void convert_frag(Frag& frag) {
  if constexpr (!std::is_same_v<src_t, MmaT>) {
#pragma unroll
    for (int i = 0; i < cute::size(frag); i += 2) {
      float2 const vals = toFloat2(reinterpret_cast<src_t const*>(&frag(i)));
      *reinterpret_cast<Pair<MmaT>*>(&frag(i)) = pack_float2<MmaT>(vals);
    }
  }
}

// State → MMA B operand: dtype-aware TiledCopy.
//   2-byte smem: LDSM (SM75_U32x2_LDSM_N) — vectorized 16-bit ldmatrix.
//   4-byte smem: scalar UniversalCopy<state_t>; pairs are converted to
//                bf16 in registers by `convert_frag` after the load.
template <typename state_t, typename MmaT, typename TiledMma>
__device__ __forceinline__ auto make_state_b_s2r(TiledMma const& tm) {
  using namespace cute;
  if constexpr (sizeof(state_t) == 2) {
    return make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MmaT>{}, tm);
  } else {
    static_assert(sizeof(state_t) == 4, "wide state path expects 4-byte smem");
    return make_tiled_copy_B(Copy_Atom<UniversalCopy<state_t>, state_t>{}, tm);
  }
}

// Src → dst fragment conversion — a strict superset of the in-place overload
// above: supports narrowing (e.g. f32 → bf16) via a separate src fragment.
// Three paths:
//   (1) src_t == dst_t: bit copy via Pair<dst_t> (sidesteps cutlass-wrapper vs
//       native dtype mismatches like cutlass::bfloat16_t vs __nv_bfloat16).
//   (2) Same width, different dtype (e.g. fp16 → bf16): paired cvt through f32.
//       Works in-place when `src` aliases `dst`.
//   (3) Different width (e.g. f32 → bf16): paired element load + pack_float2.
template <typename src_t, typename dst_t, typename SrcFrag, typename DstFrag>
__device__ __forceinline__ void convert_frag(SrcFrag const& src, DstFrag& dst) {
  using namespace cute;
  if constexpr (std::is_same_v<src_t, dst_t>) {
#pragma unroll
    for (int i = 0; i < size(src); i += 2) {
      *reinterpret_cast<Pair<dst_t>*>(&dst(i)) = *reinterpret_cast<Pair<dst_t> const*>(&src(i));
    }
  } else if constexpr (sizeof(src_t) == sizeof(dst_t)) {
#pragma unroll
    for (int i = 0; i < size(src); i += 2) {
      float2 const vals = toFloat2(reinterpret_cast<src_t const*>(&src(i)));
      *reinterpret_cast<Pair<dst_t>*>(&dst(i)) = pack_float2<dst_t>(vals);
    }
  } else {
    static_assert(sizeof(dst_t) == 2, "only narrowing to 2-byte dst supported");
#pragma unroll
    for (int i = 0; i < size(src); i += 2) {
      *reinterpret_cast<Pair<dst_t>*>(&dst(i)) =
          pack_float2<dst_t>(make_float2(src(i), src(i + 1)));
    }
  }
}

// 2b. frag_y += CB_scaled @ x  (matmul 4, single K-tile)
//     CB_scaled A operand loaded from swizzled smem via LDSM (precomputed by warps 0,1).
//     x B operand loaded from smem via ldmatrix.trans.
template <typename input_t, typename MmaT, int N_TILE, int NTOKENS_PAD_MMA_M, typename FragY,
          typename FragCB, typename SmemXTrans, typename S2RBTrans, typename S2RThrBTrans,
          typename ThrMma, typename TiledMma>
__device__ __forceinline__ void add_cb_x(FragY& frag_y, FragCB const& frag_CB,
                                         SmemXTrans const& smem_x_trans,
                                         S2RBTrans const& s2r_B_trans,
                                         S2RThrBTrans const& s2r_thr_B_trans, ThrMma const& thr_mma,
                                         TiledMma const& tiled_mma, int n) {
  using namespace cute;
  Tensor smem_x_trans_ntile = local_tile(
      smem_x_trans, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_M>{}), make_coord(n, _0{}));
  auto smem_x_trans_s2r = s2r_thr_B_trans.partition_S(smem_x_trans_ntile);
  auto frag_B_x = thr_mma.partition_fragment_B(
      make_tensor((MmaT*)0x0, make_shape(Int<N_TILE>{}, Int<NTOKENS_PAD_MMA_M>{})));
  auto frag_B_x_view = s2r_thr_B_trans.retile_D(frag_B_x);

  cute::copy(s2r_B_trans, smem_x_trans_s2r, frag_B_x_view);
  cute::gemm(tiled_mma, frag_y, frag_CB, frag_B_x, frag_y);
}

// 3b. frag_y += D * x[t, d]  (per-thread skip connection via partition_C)
template <typename input_t, int NTOKENS_PAD_MMA_M, int N_TILE, typename FragY, typename SmemX,
          typename ThrMma>
__device__ __forceinline__ void add_D_skip(FragY& frag_y, SmemX const& smem_x,
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
__device__ __forceinline__ void compute_z_gating(FragY& frag_y, SmemZ const& smem_z,
                                                 ThrMma const& thr_mma, void const* z_ptr, int n) {
  using namespace cute;
  if (!z_ptr) return;
  Tensor smem_z_tile =
      local_tile(smem_z, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}), make_coord(_0{}, n));
  Tensor z_part = thr_mma.partition_C(smem_z_tile);
#pragma unroll
  for (int i = 0; i < size(frag_y); i += 2) {
    float2 const z = toFloat2(reinterpret_cast<input_t const*>(&z_part(i)));
    frag_y(i) *= z.x * __fdividef(1.f, (1.f + __expf(-z.x)));
    frag_y(i + 1) *= z.y * __fdividef(1.f, (1.f + __expf(-z.y)));
  }
}

// =============================================================================
// Pipelined K-loop GEMM
// =============================================================================
// Computes  frag_y[n] += A @ B[n]  for n ∈ [0, NumNTiles), where A is shared
// across N-tiles and B[n] is the n-th N-tile of `smem_B` (sliced inside).
//
// NumStages-deep register pipeline hides LDSM → HMMA latency: at steady state
// slot (k+NumStages-1) is loading while HMMA consumes slot k.  ATypeIn → MmaT
// and BTypeIn → MmaT conversions happen in registers between load and consume
// (in-place when widths match — see `convert_frag`).
//
// Used by matmul 3 (init_out += C @ state^T): A = C (shared), B = state.
// NumNTiles = sizeof...(FragY) = D_PER_CTA / N_TILE  (1 for D_SPLIT=2, 2 for
// D_SPLIT=1).
template <int NumStages, int NumKTiles, typename ATypeIn, typename BTypeIn, typename MmaT,
          typename TiledMma, typename ThrMma, typename SmemAKtiled, typename SmemB,
          typename... FragY>
__device__ __forceinline__ void pipelined_kloop_gemm(TiledMma const& tiled_mma,
                                                     ThrMma const& thr_mma, int tid,
                                                     SmemAKtiled const& smem_A_ktiled,
                                                     SmemB const& smem_B, FragY&... frag_y) {
  using namespace cute;
  constexpr int NumNTiles = sizeof...(FragY);
  static_assert(NumStages >= 2, "NumStages must be >= 2 for pipelining");
  static_assert(NumKTiles >= NumStages - 1, "NumKTiles must be >= NumStages - 1 for full prologue");
  static_assert(NumNTiles >= 1, "NumNTiles must be >= 1");

  constexpr int N_TILE = cute::tile_size<1>(TiledMma{});
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MmaT>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_state_b_s2r<BTypeIn, MmaT>(tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── Tile B by (N, K): shape (N_TILE, K_TILE, N_OUTER, NumKTiles) ──
  auto smem_B_tiled = local_tile(smem_B, make_tile(Int<N_TILE>{}, Int<K_TILE>{}), make_coord(_, _));

  // ── Partitioned smem (A shared, B per-N-tile) ──
  auto smem_A_s2r = s2r_thr_A.partition_S(smem_A_ktiled);
  auto sample_smem_B_n = smem_B_tiled(_, _, _0{}, _);
  using SmemBS2RType = decltype(s2r_thr_B.partition_S(sample_smem_B_n));
  SmemBS2RType smem_B_s2r[NumNTiles];
  CUTE_UNROLL
  for (int n = 0; n < NumNTiles; ++n) {
    smem_B_s2r[n] = s2r_thr_B.partition_S(smem_B_tiled(_, _, n, _));
  }

  // ── Fragment / view types ──
  using FragA = decltype(thr_mma.partition_fragment_A(smem_A_ktiled(_, _, _0{})));
  using FragB = decltype(thr_mma.partition_fragment_B(sample_smem_B_n(_, _, _0{})));
  using b_view_t = std::conditional_t<sizeof(BTypeIn) == sizeof(MmaT), MmaT, BTypeIn>;
  using FragBStg = decltype(make_fragment_like<b_view_t>(std::declval<FragB>()));
  using FragAView = decltype(s2r_thr_A.retile_D(std::declval<FragA&>()));
  using FragBStgView = decltype(s2r_thr_B.retile_D(std::declval<FragBStg&>()));

  // ── Multi-stage register fragments ──
  // Storage type matches the MMA fragment for A; for B the staging buffer is
  // BTypeIn-typed (when narrowing) or MmaT-typed (when widths match — the two
  // alias the same registers and `convert_frag` collapses to a bit-copy /
  // in-place reinterpret).
  FragA frag_A[NumStages];
  FragB frag_B[NumNTiles][NumStages];
  FragBStg frag_B_stg[NumNTiles][NumStages];
  FragAView frag_A_view[NumStages];
  FragBStgView frag_B_stg_view[NumNTiles][NumStages];
  CUTE_UNROLL
  for (int s = 0; s < NumStages; ++s) {
    frag_A_view[s] = s2r_thr_A.retile_D(frag_A[s]);
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      frag_B_stg_view[n][s] = s2r_thr_B.retile_D(frag_B_stg[n][s]);
    }
  }

  // Pack frag_y into a pointer array for indexed access (replay kernel pattern).
  using FragY0 = std::tuple_element_t<0, std::tuple<FragY...>>;
  static_assert((std::is_same_v<FragY0, FragY> && ...),
                "all FragY parameters must be the same type");
  FragY0* frag_y_p[NumNTiles] = {(&frag_y)...};

  // ── Per-stage operations (slot is constant after #pragma unroll) ──
  auto load_one = [&](int k_src, int slot) {
    cute::copy(s2r_A, smem_A_s2r(_, _, _, k_src), frag_A_view[slot]);
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      cute::copy(s2r_B, smem_B_s2r[n](_, _, _, k_src), frag_B_stg_view[n][slot]);
    }
  };
  auto convert_one = [&](int slot) {
    convert_frag<ATypeIn, MmaT>(frag_A[slot]);
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      convert_frag<BTypeIn, MmaT>(frag_B_stg[n][slot], frag_B[n][slot]);
    }
  };
  auto compute_one = [&](int slot) {
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      cute::gemm(tiled_mma, *frag_y_p[n], frag_A[slot], frag_B[n][slot], *frag_y_p[n]);
    }
  };

  // ── Clear accumulators ──
  CUTE_UNROLL
  for (int n = 0; n < NumNTiles; ++n) clear(*frag_y_p[n]);

  // ── Prologue: load + convert stages 0..NumStages-2 ──
  CUTE_UNROLL
  for (int s = 0; s < NumStages - 1; ++s) {
    load_one(s, s);
    convert_one(s);
  }

  // ── Main K-loop: load slot (k+NumStages-1) % NumStages, compute slot k % NumStages ──
#pragma unroll
  for (int k = 0; k < NumKTiles; ++k) {
    int const k_load = k + NumStages - 1;
    int const slot_load = k_load % NumStages;
    int const slot_compute = k % NumStages;
    if (k_load < NumKTiles) load_one(k_load, slot_load);
    compute_one(slot_compute);
    if (k_load < NumKTiles) convert_one(slot_load);
  }
}

// ── Matmul 3: init_out = C @ state^T ────────────────────────────────────────
// Thin wrapper: builds the swizzled smem views for C and state, then dispatches
// to `pipelined_kloop_gemm`.  NumNTiles = sizeof...(FragY) = D_PER_CTA / N_TILE
// (1 for D_SPLIT=2, 2 for D_SPLIT=1).
//
// On C: aliased view (see compute_CB_scaled_2warp).  On state: 2-byte smem is
// reinterpret-cast to MMA_prop::operand_t so the 16-bit LDSM atom matches the view
// (actual element type recovered inside `convert_frag`); ≥4-byte smem keeps
// the native dtype and uses scalar UniversalCopy + register conversion.
template <typename input_t, typename state_t, int D_PER_CTA, int DSTATE, typename SmemT,
          typename TiledMma, typename ThrMma, typename... FragY>
__device__ __forceinline__ void add_init_out(SmemT const& smem, TiledMma const& tiled_mma,
                                             ThrMma const& thr_mma, int tid, FragY&... frag_y) {
  using namespace cute;
  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr bool is_2byte_smem = (sizeof(state_t) == 2);
  using state_view_t = std::conditional_t<is_2byte_smem, MMA_prop::operand_t, state_t>;

  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, DSTATE, SmemT::NTOKENS>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  Tensor smem_C_ktiled =
      local_tile(smem_C, make_tile(Int<NTOKENS_PAD_MMA_M>{}, Int<K_TILE>{}), make_coord(_0{}, _));

  auto layout_state_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  Tensor smem_state = make_tensor(make_smem_ptr(reinterpret_cast<state_view_t const*>(smem.state)),
                                  layout_state_swz);

  pipelined_kloop_gemm<3, NUM_K_TILES, input_t, state_t, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_C_ktiled, smem_state, frag_y...);
}

// store_state: vectorized smem → gmem state writeback (128 threads).
// Defined here (rather than alongside the other Phase 3 store helpers
// below) because compute_and_store_output calls it inline for the
// v10.3 hoist — issued right after matmul 3 so the STGs fire-and-forget in
// parallel with matmul 4 + epilogue.  smem and gmem hold the same dtype now
// (no on-egress conversion) so this is always a direct 128-bit copy.
template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void store_state(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot) {
  using namespace cute;
  int const flat_tid = warp * warpSize + lane;
  auto* __restrict__ state_w = reinterpret_cast<state_t*>(params.state);
  // v12 §59: gmem dest = head's full state base + d_tile's row slice.
  int64_t const state_base = cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE +
                             (int64_t)d_tile * D_PER_CTA * DSTATE;

  // ── Per-CTA smem swizzle layout [D_PER_CTA, DSTATE]. ──
  auto layout_smem_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  state_t const* smem_state_base = reinterpret_cast<state_t const*>(smem.state);

  Tensor sState = make_tensor(make_smem_ptr(smem_state_base), layout_smem_swz);
  Tensor gState = make_tensor(make_gmem_ptr(state_w + state_base),
                              make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                          make_stride(Int<DSTATE>{}, Int<1>{})));
  // Each store is 16 bytes — adjust val cols to the dtype.
  constexpr int VAL_COLS = Copy_prop::vec_bytes / sizeof(state_t);
  auto s2g =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, state_t>{},
                      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = s2g.get_slice(flat_tid);
  copy(s2g, thr.partition_S(sState), thr.partition_D(gState));
}

// ── Orchestrator: compute_and_store_output ─────────────────────────────
//     out = (C @ state^T) * decay + CB_scaled @ x + D*x, then z-gate.
//     All operations on register-resident frag_y — no smem round-trip.
//     Result converted f32 → input_t in registers and stored directly to gmem
//     via partition_C of the global output tensor (like CUTLASS sgemm_sm80 epilogue).
template <typename input_t, typename state_t, int NTOKENS, int DIM, int D_PER_CTA, int DSTATE,
          int NUM_WARPS, int PHILOX_ROUNDS, typename SmemT>
__device__ __forceinline__ void compute_and_store_output(SmemT& smem,
                                                         SsuIncrementalParams const& params,
                                                         int warp, int lane, int d_tile,
                                                         int batch_idx, int head,
                                                         int64_t cache_slot, float D_val) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_and_store_output requires 2-byte input type");

  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA: 128 threads, covers [16, 32] output per step ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  // v12 §59: when D_PER_CTA < swizzle atom (= 64 for bf16), the underlying
  // smem buffer is padded to D_SMEM_COLS so the swizzle layout is well-formed.
  // Per-pass MMA loops only iterate D_PER_CTA / N_TILE tiles → never touch
  // the padded tail.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;

  // x: swizzled [NTOKENS_PAD_MMA_M, D_SMEM_COLS]
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)),
                              layout_x_swz);
  auto layout_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, NTOKENS_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans_swz);

  // z: aliased swizzled [NTOKENS_PAD_MMA_M, D_SMEM_COLS] — physical buffer
  // is only next_multiple_of<ATOM_ROWS>(NTOKENS) rows tall; second m-tile
  // aliases first.  Ghost rows feed predicated-out output rows.
  auto layout_z_swz =
      make_aliased_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, D_SMEM_COLS, NTOKENS>();
  Tensor smem_z =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.z)), layout_z_swz);

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);
  auto s2r_B_trans =
      make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B_trans = s2r_B_trans.get_slice(tid);

  // ── Load CB_scaled A operand from smem (precomputed by warps 0,1 between syncs) ──
  // Row stride matches the buffer's padded width (one swizzle atom of `input_t`).
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  auto layout_cb_swz =
      make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, NTOKENS_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // Decay broadcast: cumAdt[t] → [NTOKENS_PAD_MMA_M, N_TILE] with stride-0 on N.
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output: partition_C for direct register → gmem store ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  // v12 §59: out_base lands on this CTA's D-slice within the head.
  int64_t const out_base = (int64_t)batch_idx * params.out_stride_batch + (int64_t)head * DIM +
                           (int64_t)d_tile * D_PER_CTA;

  // Row predicate for padding.  The epilogue store loop iterates i in steps
  // of 2 and only consults pred(0) and pred(2) — m16n8k16 C-frag per thread
  // has 4 elts at rows {t/4, t/4, t/4+8, t/4+8}, so there are only 2 unique
  // row predicates.  Compute them once and skip the 4-wide pred tensor.
  auto id_tile = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < NTOKENS;
  bool const pred_row_hi = get<0>(id_part(2)) < NTOKENS;

  // v12 §59: number of output N-tiles per pass = D_PER_CTA / N_TILE.
  // D_SPLIT=1, D_PER_CTA=64, N_TILE=32 → NUM_N_TILES = 2 (current behavior).
  // D_SPLIT=2, D_PER_CTA=32                → NUM_N_TILES = 1 (uses _n1 variant).
  constexpr int NUM_N_TILES = D_PER_CTA / N_TILE;
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2,
                "Output epilogue supports NUM_N_TILES = D_PER_CTA / N_TILE in {1, 2}");

  // ── Epilogue lambda (defined once; called per N-tile from each branch) ──
  auto epilogue = [&](auto& frag_y, int n) {
  // Decay: frag_y *= exp(cumAdt[t])
#pragma unroll
    for (int i = 0; i < size(frag_y); ++i) {
      frag_y(i) *= __expf(decay_part(i));
    }

    // frag_y += CB_scaled @ x (CB from smem LDSM, x from smem via ldmatrix.trans)
    add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NTOKENS_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // frag_y += D * x[t, d]
    add_D_skip<input_t, NTOKENS_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // frag_y *= z * sigmoid(z)
    compute_z_gating<input_t, NTOKENS_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Store frag_y directly to gmem (register → gmem, no smem round-trip).
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<N_TILE>{}),
                                             make_stride(params.out_stride_mtp, _1{})));
    auto gOut_part = thr_mma.partition_C(gOut_tile);
    // Vectorized pair store: elements i and i+1 are same-row, consecutive columns
    // in the m16n8k16 partition_C layout, so &gOut_part(i+1) == &gOut_part(i) + 1.
    // Address is naturally aligned to sizeof(Pair<input_t>) since MMA column
    // index = (lane%4)*2 → even.  pack_float2 dispatches to the native packed
    // cvt (e.g. cvt.rn.bf16x2.f32 for bf16) — one instruction for the pair.
#pragma unroll
    for (int i = 0; i < size(frag_y); i += 2) {
      // Bit 1 of i toggles between the two row groups of the m16n8k16
      // C-frag: i∈{0,1} → row t/4, i∈{2,3} → row t/4+8 (repeats per M-atom).
      bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
      if (pred_i) {
        *reinterpret_cast<Pair<input_t>*>(&gOut_part(i)) =
            pack_float2<input_t>(make_float2(frag_y(i), frag_y(i + 1)));
      }
    }
  };

  // Skip the smem→gmem state copy when philox+f16 — `replay_state_mma` already
  // did the gmem store with stochastic rounding direct from registers.
  constexpr bool kSkipSmemToGmemState = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;

  // ── Matmul 3 + store_state + epilogue, dispatching on NUM_N_TILES ──
  // (NumNTiles is deduced from the variadic frag_y... pack in `add_init_out`.)
  if constexpr (NUM_N_TILES == 2) {
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0,
                                                      frag_y_1);
    // v10.3 (Option A): state writeback hoisted here — after matmul 3 has
    // finished consuming smem.state, before matmul 4 which reads only
    // smem.x / smem.CB_scaled / smem.z.  STGs fire-and-forget alongside
    // the epilogue (matmul 4 + D*x + z-gate + output STG).
    if constexpr (!kSkipSmemToGmemState) {
      store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                              head, cache_slot);
    }
    epilogue(frag_y_0, 0);
    epilogue(frag_y_1, 1);
  } else {  // NUM_N_TILES == 1 (D_SPLIT = 2 path)
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0);
    // No sync needed before store_state: the post-replay __syncthreads()
    // in the kernel already established cross-warp visibility of replay's
    // writes to smem.state, and nothing after that point writes to it
    // (add_init_out is read-only on smem.state).
    if constexpr (!kSkipSmemToGmemState) {
      store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                              head, cache_slot);
    }
    epilogue(frag_y_0, 0);
  }
}

// ── Store functions (called from kernel after compute_y + sync) ──
// (store_state moved above compute_and_store_output — used there for
// the v10.3 state-writeback hoist.)

template <typename input_t, int NTOKENS, int DIM, int D_PER_CTA, typename SmemT>
__device__ __forceinline__ void store_old_x(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot) {
  using namespace cute;
  constexpr int NTOKENS_PAD_MMA_M = SmemT::NTOKENS_PAD_MMA_M;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_x_w = reinterpret_cast<input_t*>(params.old_x);
  // v12 §59: gmem dest = head's full slot + d_tile's D-slice offset.
  int64_t const ox_w_base =
      cache_slot * params.old_x_stride_cache + head * DIM + (int64_t)d_tile * D_PER_CTA;

  // Smem and gmem are both viewed at the full atom-padded width D_SMEM_COLS.
  // The wide thread layout (16 row × 8 col × 1×8 val = 16 rows × 64 cols/pass
  // for bf16) covers one full atom width per thread-row, which is the
  // swizzle's bank-conflict-free contract on the LDS side (load-from-smem).
  // A narrow layout would (a) waste 64 threads (warps 2, 3 idle) and
  // (b) cause LDS bank conflicts on the smem-read side (observed as
  // 4-way LDS conflict in d_split=2 ncu).  Cols ≥ D_PER_CTA are predicated
  // off via copy_if so STG never fires for them — no OOB write into the
  // next d_tile / next head's gmem region.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_M, D_SMEM_COLS>();
  Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.x)), layout_x_swz);
  Tensor gX = make_tensor(make_gmem_ptr(old_x_w + ox_w_base),
                          make_layout(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}),
                                      make_stride(params.old_x_stride_mtp, Int<1>{})));

  using ThrLayoutX = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{}, ThrLayoutX{},
                             Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);

  auto tSsX = thr_s2g.partition_S(sX);
  auto tSgX = thr_s2g.partition_D(gX);

  // Per-(row, col) predicate: skip rows ≥ NTOKENS (m-padding) and cols ≥
  // D_PER_CTA (atom-padding past the d_tile's data).
  auto cX = make_identity_tensor(make_shape(Int<NTOKENS_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}));
  auto tScX = thr_s2g.partition_D(cX);
  auto pred = make_tensor<bool>(shape(tScX));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = (get<0>(tScX(i)) < NTOKENS) && (get<1>(tScX(i)) < D_PER_CTA);
  }
  copy_if(s2g, pred, tSsX, tSgX);
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

  auto layout_B_swz = make_swizzled_layout_rc<input_t, NTOKENS_PAD_MMA_K, DSTATE>();
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B_swz);
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
          int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1>
__global__ void ssu_incremental_kernel(SsuIncrementalParams params) {
  // v12 §59: per-head DIM is sharded across `D_SPLIT` CTAs (D_PER_CTA each).
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32,
                "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 warp layout). "
                "D_SPLIT=4 (D_PER_CTA=16) needs warp-count restructure — deferred to v12.x.");
  // Cross-check: host launcher must dispatch the template specialization
  // matching the runtime params.d_split it stamped into the struct.
  assert(params.d_split == D_SPLIT);
  using SmemT = SsuIncrementalStorage<input_t, state_t, NTOKENS, D_PER_CTA, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // v12 §59: grid layout (D_SPLIT, batch, nheads).
  int const d_tile = blockIdx.x;
  int const batch_idx = blockIdx.y;
  int const head = blockIdx.z;
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
  load_data<input_t, dt_t, state_t, NTOKENS, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, d_tile, batch_idx, head, group_idx, cache_slot, buf_read, A_val,
      dt_bias_val);

  // v12.1: dropped the post-load_data __syncthreads.  Cooperative
  // `load_state_cta` issues cp.async from every thread; each thread's own
  // `__pipeline_wait_prior(0)` at the end of `load_data` retires its own
  // group, so each warp sees its own writes after the per-warp __syncwarp.
  // Cross-warp visibility is established by the post-replay __syncthreads
  // below — replay reads of state are now safe because (a) replay's frag_h
  // initial load sees only the current warp's lane positions, and
  // (b) the actual `_1×4` cross-warp dependency is on writes that haven't
  // happened yet at this point.
  // (See v12.0 §60 dirty-bits #4 — this sync was added speculatively while
  // chasing what turned out to be a `store_old_x` OOB bug.)

  // v10.1: old_B writeback hoisted ahead of Phase 1.  Source (smem.B) is
  // consumed only by Phase 1a CB; the STGs fire-and-forget onto the
  // memory subsystem and complete in parallel with all subsequent compute.
  // v11.0: only W0, W1 hold valid smem.B at this point (they're the ones
  // that cp.async'd B).  Gate accordingly — store halves its thread
  // count but B is small (4 KB) so still cheap.
  // v12 §59 step 3: old_B is D-independent (per-group, full DSTATE) — only
  // d_tile == 0 writes; other d_tiles would emit identical payloads.
  if (d_tile == 0 && warp < 2) {
    store_old_B<input_t, NTOKENS, DSTATE, HEADS_PER_GROUP>(smem, params, warp, lane, head,
                                                           group_idx, cache_slot, buf_write);
  }

  // Warps 0,1 compute CB_scaled into smem (split N=16 → 2×[16,8]).  No
  // barrier needed — warps 2,3 start replay immediately; warps 0,1 join
  // after CB.
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, NTOKENS, DSTATE>(smem, warp, lane);
  }
  // Phase 1b: MMA replay (all 4 warps, independent M-rows).  Each warp
  // reads its own DIM slice of state + old_x (loaded into smem by this
  // same warp in load_data), plus smem.old_B (redundantly loaded by all
  // warps — each warp sees its own copy).
  // ── Philox seed/offset for f16 state stochastic rounding (no-op for
  // PHILOX_ROUNDS == 0 — compiles out via if constexpr inside replay).  Offset
  // matches Triton's `base_rand = cache_batch * stride_state_batch + pid_h *
  // (DIM*DSTATE)` (per-row m-offset and per-col offset added inside replay).
  int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  uint32_t const state_ptr_offset =
      static_cast<uint32_t>(cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE);
  // Full element offset to this CTA's state slice in gmem — used by the
  // philox path's direct register→gmem STG.32 writeback.
  int64_t const state_gmem_off = cache_slot * params.state_stride_batch +
                                 (int64_t)head * DIM * DSTATE +
                                 (int64_t)d_tile * D_PER_CTA * DSTATE;
  replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS>(
      smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_gmem_off, rand_seed);

  __syncthreads();

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: Output — y[t,d] = init_out + cb_out + D*x, then z-gate
  // ════════════════════════════════════════════════════════════════════════
  // D_val already loaded in preamble (gmem latency hidden behind Phase 0+1).
  //
  // Fused: matmul 3 + state-writeback + matmul 4 + D*x + z-gate → direct gmem store
  compute_and_store_output<input_t, state_t, NTOKENS, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                           PHILOX_ROUNDS>(smem, params, warp, lane, d_tile, batch_idx, head,
                                          cache_slot, D_val);

  // ── Phase 3: Store to global memory ──
  // (old_B hoisted to pre-Phase-1 in v10.1;
  //  state hoisted into compute_and_store_output in v10.3.)

  // Cache writes — old_x uses all warps (vectorized), dt/cumAdt one warp each
  store_old_x<input_t, NTOKENS, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head, cache_slot);
  // dt_proc / cumAdt are D-independent — only d_tile == 0 writes.
  if (d_tile == 0 && warp == 0 && lane < NTOKENS) {
    auto* __restrict__ old_dt_proc_w = reinterpret_cast<float*>(params.old_dt_proc);
    int64_t const dt_w_base = cache_slot * params.old_dt_proc_stride_cache +
                              buf_write * params.old_dt_proc_stride_dbuf +
                              head * params.old_dt_proc_stride_head;
    old_dt_proc_w[dt_w_base + lane] = smem.dt_proc[lane];
  }
  if (d_tile == 0 && warp == 1 && lane < NTOKENS) {
    auto* __restrict__ old_cumAdt_w = reinterpret_cast<float*>(params.old_cumAdt);
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_cache +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + lane] = smem.cumAdt[lane];
  }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` (v12 §59) splits each head's DIM axis across `D_SPLIT` CTAs.
// `launchSsuIncrementalImpl` is the per-D_SPLIT specialization;
// `launchSsuIncremental` (below) is the runtime dispatcher.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int D_SPLIT>
void launchSsuIncrementalImpl(SsuIncrementalParams& params, cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;

  FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                   ") must be divisible by ngroups (", params.ngroups, ")");

  // cp.async.ca with .L2::128B requires 16B-aligned pointers (128-bit / sizeof element).
  // The .L2::128B hint further requires the base address to be 128B-aligned for full
  // cache line utilization, but the hardware only faults on < 16B alignment.
  FLASHINFER_CHECK_ALIGNMENT(params.B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.C, 16);

  // v12 §59: per-CTA D = DIM / D_SPLIT.  Smem footprint shrinks for D-owned
  // buffers (state, x, z, old_x); non-D buffers (B, C, old_B, scalars) unchanged.
  constexpr int D_PER_CTA = DIM / D_SPLIT;

  auto dispatch_hpg = [&]<int HEADS_PER_GROUP>() {
    auto func = ssu_incremental_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                       state_scale_t, NTOKENS_MTP, DIM, DSTATE, HEADS_PER_GROUP,
                                       PHILOX_ROUNDS, NUM_WARPS, D_SPLIT>;

    constexpr size_t smem_size =
        sizeof(SsuIncrementalStorage<input_t, state_t, NTOKENS_MTP, D_PER_CTA, DSTATE>);

    // v12 §59: grid is (D_SPLIT, batch, nheads).  D-tile is the fastest axis
    // so the `D_SPLIT` CTAs of the same head land on adjacent SMs and share
    // L2 lines for the redundantly-loaded inputs (C, B, dt, ...).
    dim3 grid(D_SPLIT, params.batch, params.nheads);
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

// Public dispatcher: routes on `params.d_split` (v12 §59).  Allowed values
// for v12: {1, 2}.  D_SPLIT=4 (D_PER_CTA=16) requires warp-count restructure
// for the output MMA's `_1×4` layout — deferred to v12.x.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchSsuIncremental(SsuIncrementalParams& params, cudaStream_t stream) {
  auto launch = [&]<int D_SPLIT>() {
    launchSsuIncrementalImpl<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                             state_scale_t, D_SPLIT>(params, stream);
  };
  switch (params.d_split) {
    case 1:
      launch.template operator()<1>();
      break;
    case 2:
      launch.template operator()<2>();
      break;
    default:
      FLASHINFER_CHECK(false, "Unsupported d_split: ", params.d_split,
                       ".  Allowed values for v12: {1, 2}.  d_split=4 is "
                       "deferred to v12.x (needs warp-count restructure).");
  }
}

}  // namespace flashinfer::mamba::incremental

#endif  // FLASHINFER_MAMBA_KERNEL_SSU_INCREMENTAL_CUH_
