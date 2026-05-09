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
// Single __syncthreads() via per-warp data ownership.  Every smem
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
//   Scalars + cumAdt: redundant on each warp's first NPREDICTED/MAX_WINDOW lanes.
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

// ldmatrix.b8 (SM100_U8x16_LDSM_T) was tried as a replacement for per-lane
// LDS.16 int8 state loads.  It's 5-18% slower (bench v16.7b vs v16.8) because:
//   (1) inherent 2-way bank conflicts (16 threads × 16B vs 128B banks),
//   (2) state is the accumulator (C-frag), not an A/B operand — layout
//       remapping costs 8 shuffles + byte extractions,
//   (3) dynamic byte selection via SHF adds 15%+ short_scoreboard stalls.
// Keep the code behind this flag for reference; see .plans/ssu_checkpointing.md
// section 3b.1 for full investigation notes.
#define USE_LDMATRIX_INT8 0

namespace constants {
constexpr unsigned int MASK_ALL_LANES = 0xFFFFFFFFu;
constexpr unsigned int num_bits_uint32 = 32u;
}  // namespace constants

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

// Pair<int8_t>: explicit 16-bit packed specialization.  A struct of two
// `int8_t` fields would let the compiler split the pair into two 32-bit
// registers (CUDA registers are 32-bit; sub-word fields are zero-extended
// per access).  By backing the pair with a single 16-bit word and
// extracting via shift+cast, we keep the two elements in one register
// throughout the load → unpack → cast pipeline.
template <>
struct Pair<int8_t> {
  uint16_t raw;  // [bits 7:0] = element 0, [bits 15:8] = element 1
  template <int I>
  __device__ __forceinline__ int8_t operator[](cute::Int<I>) const {
    static_assert(I == 0 || I == 1, "Pair index must be 0 or 1");
    if constexpr (I == 0)
      return static_cast<int8_t>(raw & 0xFFu);
    else
      return static_cast<int8_t>(raw >> 8);
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
// k=8 and k=16 atoms at compile time (MAX_WINDOW ≤ 8 picks K8 for smaller smem,
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
// MMA accumulator rows ≥ VALID_ROWS, predicated out at gmem store.
//
// When VALID_ROWS > ATOM_ROWS (VALID_ROWS > 8 for 2-byte), PHYS_ROWS == LOGICAL_ROWS
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
// `D_PER_CTA` is the per-CTA D dimension after D-split.
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
// well-formed.  Cost: 1 KB per [NPREDICTED_PAD_MMA_M, 64] buffer at D_PER_CTA=32.
template <typename input_t, typename state_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE>
struct SsuIncrementalStorage {
  // Re-export the two T/W axis sizes so helpers that only see SmemT can
  // recover them.
  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  // Swizzle atom width for input_t (= 64 cols for 2-byte types).
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  // M-dim of the output MMAs (C, x, z, CB_scaled): always m16-tiled (keyed
  // off NPREDICTED, the new-tokens count).
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  // N-dim of the precompute-CB MMA (matmul-1: C @ B^T).  B's row count is
  // the matmul N-axis → padded to MMA::N=8.  When NPREDICTED ≤ 8 only warp
  // 0 has valid B rows; warp 1 zero-fills its CB slice.
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  // K-dim of the replay MMA (matmul-2: old_x^T @ dB_scaled).  Padded to
  // the small atom's K (== the LDSM unit for 2-byte elements).  When
  // MAX_WINDOW ≤ MMA::K_SMALL=8, replay picks the small atom (1 K-tile,
  // smaller smem, +1 CTA/SM occupancy); otherwise the big atom.  Assumes
  // MAX_WINDOW ≤ MMA::K_BIG (asserted in the wrapper).
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  // Row count for buffers padded only to the input-type swizzle atom's row
  // extent (8 for 2-byte, 4 for 4-byte) — used by C and z, which alias the
  // second m-tile back onto the first via `make_aliased_swizzled_layout_rc`.
  // Keyed off NPREDICTED.
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);

  // All 2D smem buffers below are stored as flat 1D arrays — the actual
  // physical layout is determined by `make_swizzled_layout_rc<...>` at each
  // access site, which scrambles (row, col) → physical offset via the
  // Swizzle XOR.  Declaring them as `T[ROWS][COLS]` would falsely suggest a
  // row-major C-array layout that nobody ever uses; the only thing that
  // matters here is total byte count and 16-byte alignment.

  // CB_scaled — logical (NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE) Swizzle<3,3,3>.
  // CB_ROW_STRIDE pads each row to one bank cycle (128 B = 32 banks × 4 B)
  // worth of `input_t` so LDSM reads in matmul-4's A operand are
  // conflict-free.  Equals the swizzle atom's col extent for `input_t`
  // (64 for 2-byte, 32 for 4-byte).  Logical CB matrix is
  // (NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M); trailing cols are padding.
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;
  alignas(16) input_t CB_scaled[NPREDICTED_PAD_MMA_M * CB_ROW_STRIDE];

  // B — logical (NPREDICTED_PAD_MMA_N, DSTATE).  Row count is matmul-1's
  // N-axis (since matmul-1 = C @ B^T).  Padding rows inside [NPREDICTED,
  // NPREDICTED_PAD_MMA_N) contain garbage — valid output uses only
  // [0, NPREDICTED).  Warp-1 of compute_CB_scaled_2warp reads rows ≥ 8 of a
  // 16-row view; those reads spill into C/old_B smem but are masked to 0 by
  // the (j < NPREDICTED) CB-store predicate since j ≥ 8 ≥ NPREDICTED when
  // NPREDICTED_PAD_MMA_N == 8.
  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];

  // C — physical (next_multiple_of<ATOM_ROWS>(NPREDICTED), DSTATE).  Padded
  // only to the swizzle atom's row extent (8 for 2-byte, 4 for 4-byte), not
  // to MMA_prop::M=16.  cp.async writes to this exact extent (CShape's first
  // dim shrunk to match — see load_data).  The MMA still views it as
  // NPREDICTED_PAD_MMA_M=16 rows via `make_aliased_swizzled_layout_rc`,
  // which aliases the second m-tile back onto the first via stride-0
  // row-tile mode.  Garbage feeds output rows ≥ NPREDICTED — predicated
  // out at gmem store.  Saves up to 2 KB of smem at NPREDICTED ≤ ATOM_ROWS,
  // no-op when NPREDICTED > ATOM_ROWS.
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];

  // x — logical (NPREDICTED_PAD_MMA_M, D_SMEM_COLS).  Cols padded to
  // D_SMEM_COLS for swizzle atom alignment; cp.async only fills cols
  // [0, D_PER_CTA), the tail is unused.
  alignas(16) input_t x[NPREDICTED_PAD_MMA_M * D_SMEM_COLS];

  // z — physical (next_multiple_of<ATOM_ROWS>(NPREDICTED), D_SMEM_COLS).
  // Padded only to the swizzle atom's row extent (8 for 2-byte, 4 for
  // 4-byte), not to MMA_prop::M=16.  z is never an MMA operand — the
  // z-gating epilogue reads it via `partition_C` of the m16n8 c-frag, so
  // the MMA still views it as NPREDICTED_PAD_MMA_M=16 rows via
  // `make_aliased_swizzled_layout_rc`, which aliases the second m-tile back
  // onto the first via stride-0 row-tile mode.  Garbage feeds output rows
  // ≥ NPREDICTED — predicated out at gmem store.  Saves up to 1 KB of smem
  // at NPREDICTED ≤ ATOM_ROWS, no-op when NPREDICTED > ATOM_ROWS.
  alignas(16) input_t z[NPREDICTED_SWIZZLE_R * D_SMEM_COLS];

  // Old cache data loaded in Phase 0 (consumed in Phase 1 replay).
  // old_x — logical (MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS); ldmatrix.trans
  // feeds replay MMA A-operand (only the first D_PER_CTA cols are valid
  // data).
  alignas(16) input_t old_x[MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS];

  // old_B — logical (MAX_WINDOW_PAD_MMA_K, DSTATE) Swizzle<3,3,3>.  Replay
  // MMA reads via ldmatrix.trans (LDSM_T) + register scaling.  Padding
  // rows zero-filled via cp.async ZFILL.
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];

  float old_dt_proc[MAX_WINDOW];
  float old_cumAdt[MAX_WINDOW];

  // Processed dt for new tokens (Phase 1a uses this for CB_scaled + cumAdt)
  float dt_proc[NPREDICTED];

  // Cumulative A*dt — computed once by warp 0, read by all warps after sync
  float cumAdt[NPREDICTED];

  // state — logical (D_PER_CTA, DSTATE) in `state_t` (native dtype).  The
  // MMA path reinterprets 2-byte state as bf16 for LDSM; f32 state is loaded
  // via UniversalCopy and converted to bf16 in registers inside
  // add_init_out.
  alignas(16) state_t state[D_PER_CTA * DSTATE];

  // new_state — post-replay state in bf16, used by matmul-3 ONLY
  // (init_out = C @ state^T).  Stays bf16 (not fp32) so matmul-3 takes
  // the LDSM fast path (2-byte) and to halve the smem footprint.
  // The int8 encode pass does NOT read this — it re-runs the replay
  // matmul (replay-again) and encodes from fresh fp32 frags in registers,
  // avoiding the bf16-rounding-before-encode bug while keeping smem small.
  // Conditionally sized only for sizeof(state_t) == 1 (int8); 1-element
  // placeholder for non-int8 paths so the storage struct stays valid.
  // Logical shape (D_PER_CTA, DSTATE), Swizzle<3,3,3> for bf16 (8×64 atom).
  static constexpr int NEW_STATE_ELEMS = (sizeof(state_t) == 1) ? D_PER_CTA * DSTATE : 1;
  alignas(16) MMA_prop::operand_t new_state[NEW_STATE_ELEMS];

  // (Previously held an `amax_smem` cross-warp reduction scratchpad for
  // the int8 path.  Removed once `replay_state_mma_int8` switched to a
  // per-warp M-shard layout (Layout<_4, _1>) where amax is fully
  // warp-local — reduced via `__shfl_xor` across the 8 col-lanes — and no
  // cross-warp combine is needed.)
};

// =============================================================================
// Int8 chain-rewrite storage (sibling of SsuIncrementalStorage)
// =============================================================================
// Used by `ssu_incremental_kernel_int8` only.  Differs from the generic
// `SsuIncrementalStorage<input_t, int8_t, ...>` in two ways:
//   1. No `new_state` staging buffer — matmul-3 chains state's fp32 C-frag
//      directly into the next mma's A-operand in registers (à la
//      `convert_layout_acc_Aregs`), so no smem round-trip is needed.
//   2. Adds an `output_transpose` buffer used to flip the (D, T) output frag
//      back to (T, D) before the gmem STG.  Matmul-3/4 in the chain path
//      compute init_out^T[D, T] (M=D), so the per-warp M-shard frags must be
//      transposed via smem before storing into the (T, D) gmem layout.
//
// All shared Phase 0/1 buffers (CB_scaled, B, C, x, z, old_x, old_B, scalars,
// state) are byte-for-byte identical to the generic struct — Phase 0/1
// helpers (`compute_CB_scaled_2warp`, B/C/x/z loaders, etc.) are templated on
// `SmemT` and read these by name, so they work unchanged.
template <typename input_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA, int DSTATE>
struct SsuIncrementalStorageInt8 {
  using state_t = int8_t;

  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;

  // Shared Phase 0/1 buffers (same shape/swizzle as `SsuIncrementalStorage`).
  alignas(16) input_t CB_scaled[NPREDICTED_PAD_MMA_M * CB_ROW_STRIDE];
  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];
  alignas(16) input_t x[NPREDICTED_PAD_MMA_M * D_SMEM_COLS];
  alignas(16) input_t z[NPREDICTED_SWIZZLE_R * D_SMEM_COLS];
  alignas(16) input_t old_x[MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS];
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];

  float old_dt_proc[MAX_WINDOW];
  float old_cumAdt[MAX_WINDOW];
  float dt_proc[NPREDICTED];
  float cumAdt[NPREDICTED];

  // state — int8 input, only LDS'd in the single replay pass.  After replay
  // completes its dequant + matmul into the C-frag, smem.state is dead — so
  // `output_transpose` could in principle alias it (8 KB int8 vs 2 KB bf16
  // overlap easily), but for clarity we keep them separate; alias is a
  // Phase-4 micro-optimization.
  alignas(16) state_t state[D_PER_CTA * DSTATE];

  // output_transpose — physical (NPREDICTED_PAD_MMA_M, OUTPUT_TRANSPOSE_ROW_STRIDE)
  // input_t scratch buffer with PADDED row stride for bank-conflict-free per-thread
  // STS + 16-byte-aligned cooperative LDS.128.  Used by `compute_output_int8` to
  // flip the per-warp `frag_y_DxT[D, T]` register layout into `(T, D)` gmem order.
  //
  // Row stride: D_PER_CTA + 8 = 72 bf16 elts = 144 bytes.  The 8-elt (16-byte) pad
  // gives:
  //   - 144 % 16 == 0 → LDS.128 / STG.128 stays 16-byte aligned across all rows.
  //   - 144 / 4 % 32 == 4 → adjacent t-rows shift bank assignment by 4 banks.
  // For the m16n8 partition_C STS pattern (per-elt: 4 lanes write at fixed d,
  // t ∈ {0, 2, 4, 6} → banks {0, 4, 8, 12} on the padded layout — all distinct,
  // no conflicts), the padded layout cuts STS bank conflicts from ~63% of
  // wavefronts (NCU v16.0) down to 0%.
  // Volume: 16 × 72 × 2 B = 2.25 KB (vs unswizzled 2 KB; +256 B).
  static constexpr int OUTPUT_TRANSPOSE_ROW_STRIDE = D_PER_CTA + 8;
  alignas(16) input_t output_transpose[NPREDICTED_PAD_MMA_M * OUTPUT_TRANSPOSE_ROW_STRIDE];
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

// State load — D_SPLIT-conditional dispatch:
//
//   D_SPLIT == 1: per-warp partition (warp W loads rows
//     [W*DIM/4 : (W+1)*DIM/4)).  Large coalesced gmem
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

  // Single-local_tile path — swizzle layout sized to this CTA's
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
// Only the first NPREDICTED lanes participate; the rest are idle.
template <int NPREDICTED, typename SmemT>
__device__ __forceinline__ void compute_cumAdt(SmemT& smem, int lane, float A_val) {
  float val = (lane < NPREDICTED) ? A_val * smem.dt_proc[lane] : 0.f;
  // Inclusive prefix sum (Hillis-Steele)
  for (int offset = 1; offset < NPREDICTED; offset *= 2) {
    float other = __shfl_up_sync(constants::MASK_ALL_LANES, val, offset);
    if (lane >= offset) val += other;
  }
  if (lane < NPREDICTED) {
    smem.cumAdt[lane] = val;
  }
}

// Loads B, C, x, z, state, old_x, old_B (cp.async 16B), old_dt_proc,
// old_cumAdt, dt → dt_proc (LDG + softplus → smem), cumAdt (warp shuffle).
//
// Per-warp data ownership.  Each warp loads exactly what it (and
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
//           redundant on each warp's first NPREDICTED/MAX_WINDOW lanes.  Writes are
//           idempotent across warps (identical payloads to same slots).
// =============================================================================
template <typename input_t, typename dt_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_data(SmemT& smem, SsuIncrementalParams const& params, int lane,
                                          int warp, int d_tile, int batch_idx, int head,
                                          int group_idx, int64_t cache_slot, int buf_read,
                                          float A_val, float dt_bias_val) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(D_PER_CTA % INPUT_PACK == 0, "D_PER_CTA must be divisible by input pack size");

  // D-owned tensors (x, z, old_x, state) sliced by d_tile along D.
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

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  // CShape: first dim shrunk to the swizzle atom's row extent so cp.async
  // writes don't spill past the (also shrunk) C smem buffer.  When NPREDICTED
  // > ATOM_ROWS this falls back to NPREDICTED_PAD_MMA_M.
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  // B's smem row count = matmul-1 N-axis = NPREDICTED_PAD_MMA_N.
  using BShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // ZShape: same shrink as CShape — z is read via partition_C alias, so the
  // physical buffer only needs to be one swizzle row-atom tall.
  using ZShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<D_PER_CTA>>;
  // old_B / old_x's smem row count = replay matmul K-axis = MAX_WINDOW_PAD_MMA_K.
  using OldBShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  using OxShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

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
  // VALID_ROWS = NPREDICTED (new-token tile rows).
  if (warp < 2) {
    load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_mtp, lane);
    load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_mtp, lane);
  }

  // ── old_B: redundant on all 4 warps (each warp's replay consumes full
  // DSTATE).  Identical payloads to same smem dest — final bytes
  // deterministic.  VALID_ROWS = MAX_WINDOW (cache rows). ──
  load_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, old_B_ptr + oB_base, params.old_B_stride_mtp,
                                         lane);

  // ── old_x: redundant on all 4 warps (small, simpler than partitioning).
  // VALID_ROWS = MAX_WINDOW (cache rows). ──
  load_tile_async<OxShape, MAX_WINDOW>(smem.old_x, old_x_ptr + ox_base, params.old_x_stride_mtp,
                                       lane);

  // ── x: W2 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 2) {
    load_tile_async<XShape, NPREDICTED>(smem.x, x_ptr + x_base, params.x_stride_mtp, lane);
  }

  // ── z: W3 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 3 && z_ptr) {
    int64_t const z_base = (int64_t)batch_idx * params.z_stride_batch + head * DIM + d_tile_off;
    load_tile_async<ZShape, NPREDICTED>(smem.z, z_ptr + z_base, params.z_stride_mtp, lane);
  }

  // ── Scalar loads + cumAdt cumsum: redundant per warp.
  // old_dt_proc / old_cumAdt: load up to MAX_WINDOW lanes (cache scalars).
  // dt_proc: load up to NPREDICTED lanes (new-token scalars).
  // Synchronous LDG + plain smem stores — no cp.async.  Writes from 4
  // warps to the same slots are idempotent (same payloads). ──
  static_assert(MAX_WINDOW <= warpSize, "MAX_WINDOW must fit in a single warp");
  if (lane < MAX_WINDOW) {
    int64_t const dt_rd_base = cache_slot * params.old_dt_proc_stride_cache +
                               buf_read * params.old_dt_proc_stride_dbuf +
                               head * params.old_dt_proc_stride_head;
    smem.old_dt_proc[lane] = old_dt_proc_ptr[dt_rd_base + lane];

    int64_t const ca_rd_base = cache_slot * params.old_cumAdt_stride_cache +
                               buf_read * params.old_cumAdt_stride_dbuf +
                               head * params.old_cumAdt_stride_head;
    smem.old_cumAdt[lane] = old_cumAdt_ptr[ca_rd_base + lane];
  }
  if (lane < NPREDICTED) {
    float dt_val = toFloat(
        dt_ptr[(int64_t)batch_idx * params.dt_stride_batch + lane * params.dt_stride_mtp + head]);
    dt_val += dt_bias_val;
    if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
    smem.dt_proc[lane] = dt_val;
  }
  // cumAdt = cumsum(A * dt_proc) — warp-local Hillis-Steele shuffle.  Each
  // of the 4 warps runs the same reduction on identical inputs (dt_proc
  // just written above) and writes the same smem.cumAdt slots.
  compute_cumAdt<NPREDICTED>(smem, lane, A_val);

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
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_scaled_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  // 2-warp output split: each warp owns NPREDICTED_PAD_MMA_M / 2 cols of
  // the (NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M) CB tile.  Must be a
  // multiple of the MMA atom's N for the partition to be atom-aligned
  // (currently 8 == MMA_prop::N for NPREDICTED_PAD_MMA_M=16; if M-pad ever
  // grows, this still holds as long as M-pad is a multiple of 2 * MMA_prop::N).
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;
  static_assert(N_HALF % MMA_prop::N == 0,
                "compute_CB_scaled_2warp: NPREDICTED_PAD_MMA_M / 2 must be a multiple of MMA::N");

  // CB_scaled output tile layout (used by both warp 0 compute and warp 1
  // zero-fill when smem.B has only 8 rows).  Row stride matches the buffer's
  // padded width (one swizzle atom of `input_t`).
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  auto layout_cb_swz =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();

  // ── NPREDICTED_PAD_MMA_N == 8: warp 1 has no valid B rows to read.  But
  // CB_scaled[:, 8:16] must still be zero so matmul-4's K-reduction sees
  // zeros for j ≥ NPREDICTED.  Do a simple 32-thread zero-fill and return.
  if constexpr (NPREDICTED_PAD_MMA_N == 8) {
    if (warp == 1) {
      auto* __restrict__ cb = reinterpret_cast<MMA_prop::operand_t*>(smem.CB_scaled);
      constexpr int COLS_TO_CLEAR = NPREDICTED_PAD_MMA_M - N_HALF;  // 8
#pragma unroll
      for (int i = lane; i < NPREDICTED_PAD_MMA_M * COLS_TO_CLEAR; i += warpSize) {
        int const r = i / COLS_TO_CLEAR;
        int const c = N_HALF + (i % COLS_TO_CLEAR);
        cb[layout_cb_swz(r, c)] = MMA_prop::operand_t(0.f);
      }
      return;
    }
  }

  // ── Swizzled smem views ──
  // C is padded to NPREDICTED_PAD_MMA_M; B has NPREDICTED_PAD_MMA_N rows.
  // Use NPREDICTED_PAD_MMA_N for smem_B so the physical layout matches the
  // write layout from load_tile_async — `tile_to_shape` produces different
  // outer strides for (8, 128) vs (16, 128).
  // Aliased C view: physical buffer is just next_multiple_of<ATOM_ROWS>(NPREDICTED)
  // rows tall but the MMA atom needs M=16; second m-tile aliases first m-tile
  // (predicated rows discarded at output store).
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_B = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_N, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_B = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  // ── K-tile A operand (C): full [NPREDICTED_PAD_MMA_M, K_TILE], shared by both warps ──
  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));

  // ── K-tile B operand ──
  //   NPREDICTED_PAD_MMA_N == 16: warp 0 → N=[0,8), warp 1 → N=[8,16).
  //   NPREDICTED_PAD_MMA_N == 8 : only warp 0 runs (warp 1 took the early
  //     exit above), tile at (_0, _).
  Tensor smem_B_half =
      local_tile(smem_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp, _));

  // ── Register fragments ──
  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_B_half(_, _, _0{}));

  // ── Output accumulator: [NPREDICTED_PAD_MMA_M, N_HALF] f32 ──
  auto layout_cb_half = make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
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
  auto id_half = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  auto id_part = thr_mma.partition_C(id_half);

  // ── Store to swizzled smem.CB_scaled ──
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t*>(smem.CB_scaled)), layout_cb_swz);
  // Tile into [NPREDICTED_PAD_MMA_M, N_HALF] halves; warp selects its half
  Tensor smem_CB_half = local_tile(smem_CB, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}),
                                   make_coord(_0{}, warp));
  Tensor smem_CB_part = thr_mma.partition_C(smem_CB_half);

#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    int t = get<0>(id_part(i));
    int j = warp * N_HALF + get<1>(id_part(i));
    float val;
    if (j <= t && t < NPREDICTED && j < NPREDICTED) {
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
#pragma unroll
  for (int i = 0; i < DB_COEFFS_PER_LANE; ++i) {
    // m16n8k_ V-index → K-offset: (V & 1) is the col-pair offset; (V & 2) ? 8 : 0
    // covers the second K-tile inside the K_BIG (k16) atom.
    int const k = K_base + (i & 1) + ((i & 2) << 2);
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

// Scale frag_A by dB coefficients ONCE before the N-pass loop, replacing
// 16 per-N-pass compute_dB_scaling calls on frag_B (64 scale ops → 8 or 4).
// Identity: sum_k A[m,k]*(c[k]*B[k,n]) == sum_k (c[k]*A[m,k])*B[k,n].
//
// K-index derivation (PTX ISA, m16n8k{8,16} mma.sync, A operand row-major):
//   groupID = lane / 4,  threadID_in_group = lane % 4
//   K_base = (lane % 4) * 2       (same formula as B operand)
//   m16n8k8  A regs: a0 = A[groupID, K_base:K_base+2]
//                    a1 = A[groupID+8, K_base:K_base+2]
//            frag (4 elts): {0,2}→K_base, {1,3}→K_base+1  (2 unique K)
//   m16n8k16 A regs: a0..a1 as above,
//                    a2 = A[groupID, K_base+8:K_base+10]
//                    a3 = A[groupID+8, K_base+8:K_base+10]
//            frag (8 elts): {0,2}→K_base, {1,3}→K_base+1,
//                           {4,6}→K_base+8, {5,7}→K_base+9  (4 unique K)
// =============================================================================
template <int MAX_WINDOW_PAD_MMA_K, typename FragA, typename SmemT>
__device__ __forceinline__ void apply_dA_coeff(FragA& frag_A, SmemT const& smem, float total_cumAdt,
                                               int prev_k, int lane) {
  using namespace cute;
  constexpr int FRAG_A_SIZE = size(FragA{});
  static_assert((MAX_WINDOW_PAD_MMA_K == 16 && FRAG_A_SIZE == 8) ||
                    (MAX_WINDOW_PAD_MMA_K == 8 && FRAG_A_SIZE == 4),
                "apply_dA_coeff: unsupported MMA K / frag_A size combination");
  using frag_t = typename FragA::value_type;

  int const K_base = (lane % 4) * 2;

  if constexpr (MAX_WINDOW_PAD_MMA_K == 8) {
    float const c0 = (K_base < prev_k)
                         ? __expf(total_cumAdt - smem.old_cumAdt[K_base]) * smem.old_dt_proc[K_base]
                         : 0.f;
    float const c1 = (K_base + 1 < prev_k) ? __expf(total_cumAdt - smem.old_cumAdt[K_base + 1]) *
                                                 smem.old_dt_proc[K_base + 1]
                                           : 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      frag_A(i) = frag_t(toFloat(frag_A(i)) * ((i & 1) ? c1 : c0));
    }
  } else {
    float c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const k = K_base + (j & 1) + ((j & 2) ? 8 : 0);
      c[j] = (k < prev_k) ? __expf(total_cumAdt - smem.old_cumAdt[k]) * smem.old_dt_proc[k] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int const ci = (i & 1) | ((i & 4) >> 1);
      frag_A(i) = frag_t(toFloat(frag_A(i)) * c[ci]);
    }
  }
}

// =============================================================================
// Stochastic-round one fp32 pair to a packed f16x2 u32 with amortized philox
// refresh.  rand_idx[4] is mutated in place every 4th call (when pair_idx & 3
// == 0): a single philox_randint4x feeds 4 consecutive cvt_rs calls, then
// gets refreshed.  Each refresh uses a per-lane unique `philox_off` so the
// generated randints don't collide across threads.  Triton bit-equality is
// intentionally given up here; unbiasedness still holds since each pair's
// cvt_rs gets its own dedicated 32-bit randint.
// =============================================================================
template <int PHILOX_ROUNDS>
__device__ __forceinline__ uint32_t
stochastic_round_pair_with_philox_refresh(float a, float b, int pair_idx, int64_t rand_seed,
                                          uint32_t philox_off, uint32_t (&rand_idx)[4]) {
  int const rand_pos = pair_idx & 3;
  if (rand_pos == 0) {
    conversion::philox_randint4x<PHILOX_ROUNDS>(rand_seed, philox_off, rand_idx[0], rand_idx[1],
                                                rand_idx[2], rand_idx[3]);
  }
  return conversion::cvt_rs_f16x2_f32(a, b, rand_idx[rand_pos]);
}

// =============================================================================
// Cross-pass shfl_xor + STG.64 state writeback.
//
// Given two passes' worth of post-cvt_rs packed u32s buffered in `my_packed`
// (pass-0 in [0][:], pass-1 in [1][:]), exchange via shfl_xor across lane^1
// neighbors so that all 32 lanes can issue ONE STG.64 each per pair iter:
//   - even lane k stores PASS n0 (cols (k%4)*2..(k%4)*2+3 of warp's n0 slice)
//   - odd  lane k stores PASS n1 (cols (k%4)*2-2..(k%4)*2+1 of warp's n1 slice)
//
// Halves the STG instruction count vs per-pass writeback: 1 STG.64 per pair
// iter covers BOTH passes' data via cross-lane participation.
// =============================================================================
template <int PAIRS_PER_PASS, int N_PER_PASS, int DSTATE, typename state_t, typename IdPart>
__device__ __forceinline__ void exchange_ntile_state_store_global(
    state_t* __restrict__ state_w_base, int np, int lane,
    uint32_t const (&my_packed)[2][PAIRS_PER_PASS], IdPart const& id_part) {
  using namespace cute;
  static_assert(sizeof(state_t) == 2,
                "exchange_ntile_state_store_global requires 2-byte state_t for STG.64 alignment");
  int const n_base_p0 = np * N_PER_PASS;
  int const n_base_p1 = (np + 1) * N_PER_PASS;
#pragma unroll
  for (int p = 0; p < PAIRS_PER_PASS; ++p) {
    int const i = p * 2;
    // xor mask = 1 swaps neighbor lanes: lane 0 <-> lane 1, lane 2 <-> lane 3, ...
    uint32_t const peer_p0 = __shfl_xor_sync(constants::MASK_ALL_LANES, my_packed[0][p], 1);
    uint32_t const peer_p1 = __shfl_xor_sync(constants::MASK_ALL_LANES, my_packed[1][p], 1);

    int const row = get<0>(id_part(i));
    int const col_p0 = get<1>(id_part(i)) + n_base_p0;
    int const col_p1 = get<1>(id_part(i)) + n_base_p1;

    uint64_t combined;
    int32_t gmem_off;
    if ((lane & 1) == 0) {
      // Even lane: store PASS n0 — my (lower col) in low, peer in high.
      combined = static_cast<uint64_t>(my_packed[0][p]) |
                 (static_cast<uint64_t>(peer_p0) << constants::num_bits_uint32);
      gmem_off = row * DSTATE + col_p0;
    } else {
      // Odd lane: store PASS n1 — peer (lower col) in low, my in high.
      // STG addr = gmem[row*DSTATE + (peer's col base)] = col_p1 - 2.
      combined = static_cast<uint64_t>(peer_p1) |
                 (static_cast<uint64_t>(my_packed[1][p]) << constants::num_bits_uint32);
      gmem_off = row * DSTATE + (col_p1 - 2);
    }
    *reinterpret_cast<uint64_t*>(&state_w_base[gmem_off]) = combined;
  }
}

// =============================================================================
// Phase 1b: Replay for QUANTIZED state (int8) with RN encoding.
// =============================================================================
// state[D, dstate] = dequant(state_q, decode_scale) * total_decay
//                   + old_x^T @ (coeff * old_B)
//
// Layout: per-warp M-shard via TiledMma `Layout<_4, _1>`.  Each warp owns
// D_PER_CTA / 4 D-rows × full DSTATE.  This makes amax-over-dstate fully
// warp-local (no atomic, no cross-warp __syncthreads), at the cost of
// loading full B (old_B) from smem in every warp (vs partitioning N
// across warps in the bf16/fp16 path).  Constraint: per-warp M must equal
// the m16n8 atom M (=16), so D_PER_CTA must be 64 — the wrapper enforces
// d_split == 1 for int8.
//
// Pipeline:
//   1. Replay n-loop: m16n8 matmul, write fp32 frag → smem.new_state.
//   2. STG redistribution + amax + encode pass (warp-local):
//        Each warp covers M_PER_WARP = 16 D-rows × 128 cols of new_state.
//        Re-tile 32 lanes as 4 row-groups × 8 col-segments.  Per round
//        r ∈ [0, 4): each lane reads 16 fp32 (4× LDS.128) for one D-row,
//        computes a lane-amax (16 fmaxf), `__shfl_xor` over the 8
//        col-lanes (mask 1, 2, 4) for the full-row amax, encodes 16 int8,
//        and STG.128's them to gmem.  One writer per row stores
//        decode_scale = amax/QUANT_MAX to params.state_scale.
//
// matmul-3 reads new_state (fp32) on the same M-shard partition, so no
// cross-warp visibility is needed.  The `__syncthreads` after this
// function returns is for dt_proc / cumAdt visibility (Phase 2), not for
// state.

// ─────────────────────────────────────────────────────────────────────────
// ldmatrix.b8.x2.trans helper (SM100+): issues the PTX instruction and
// applies the cutlass byte reshuffle that aligns with the DstLayout from
// Copy_Traits<SM100_U8x16_LDSM_T>.  Caller provides a 16-byte-aligned
// smem pointer; receives 4 .b32 regs (16 bytes/lane) post-reshuffle.
__device__ __forceinline__ void ldsm_b8x2_trans(void const* smem_ptr, uint32_t& dst0,
                                                uint32_t& dst1, uint32_t& dst2, uint32_t& dst3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "r"(smem_addr));
  dst0 = __byte_perm(tmp0, tmp1, 0x5140);  // {tmp0.b0, tmp0.b1, tmp1.b0, tmp1.b1}
  dst1 = __byte_perm(tmp0, tmp1, 0x7362);  // {tmp0.b2, tmp0.b3, tmp1.b2, tmp1.b3}
  dst2 = __byte_perm(tmp2, tmp3, 0x5140);
  dst3 = __byte_perm(tmp2, tmp3, 0x7362);
#else
  dst0 = dst1 = dst2 = dst3 = 0;
#endif
}

// Column-major variant: packed[i] = byte i from all 4 dst regs above.
// After shuffle, destination extracts byte[reg_sel] with a shift — no
// intermediate from_cu arrays, peak live regs drops from 16 to 6.
__device__ __forceinline__ void ldsm_b8x2_trans_packed(void const* smem_ptr, uint32_t& p0,
                                                       uint32_t& p1, uint32_t& p2, uint32_t& p3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "r"(smem_addr));
  // Transpose: packed[i] = {dst0.byte_i, dst1.byte_i, dst2.byte_i, dst3.byte_i}
  p0 = __byte_perm(tmp0, tmp2, 0x6420);
  p1 = __byte_perm(tmp1, tmp3, 0x6420);
  p2 = __byte_perm(tmp0, tmp2, 0x7531);
  p3 = __byte_perm(tmp1, tmp3, 0x7531);
#else
  p0 = p1 = p2 = p3 = 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────
// replay_state_mma_int8_chain: int8-state chain rewrite — PASS 1 only.
//
// Drops bf16 new_state smem buffer entirely; matmul-3 is fused inline with
// replay HMMA on a per-K-pair cadence (1 K-atom of A in flight at a time).
//
// Pipeline (single fused loop over K-pairs):
//   - For kpair ∈ [0, NUM_K_PAIRS=8):
//     - Replay 2 m16n8 N-atoms → fp32 frag_h × 2 (16 dstate cols of state).
//     - Update per-thread amax (fp32, bit-exact).
//     - Cast fp32 → bf16, pack into K-atom-sized A frag (`a_kpair`,
//       8 bf16/thread = 4 32-bit regs).  Layout matches the m16n8k16 A
//       operand directly (`partition_fragment_A` of the chain TiledMma).
//     - LDS one K-atom of B from `smem.C[T_pad, kpair*16..+16]`.
//     - `cute::gemm` accumulates one K-atom into `frag_y_DxT`:
//          `frag_y_DxT[D, T] += new_state[D, kpair*16..+16]
//                              @ smem.C[T_pad, kpair*16..+16]^T`.
//     - Both `a_kpair` and `b_kpair` go out of scope at iter end.
// Post-loop:
//   - Warp-local amax reduce (`__shfl_xor` over 4 col-lanes per row pair).
//   - Compute `decode_scale = amax/127`, `encode_scale = 127/amax` per row.
//   - STG `decode_scale` to gmem (one writer per (cache, head, d_row)).
//   - Return `encode_scale_per_row[2]` to the caller — needed by the
//     PASS 2 helper (`encode_state_replay_int8`) which runs *after*
//     `compute_output_int8` so that `frag_y_DxT`'s 8 fp32 regs are dead by
//     the time PASS 2's replay-again runs.
//
// Math identity for the chain (why writing `frag_h(j)` to `a_kpair(local_n*4+j)`
// places the bytes in the m16n8k16 A operand's expected position):
//   linear(cp, rp, kh, _, mma_k) = cp + 2*rp + 4*kh + 8*mma_k
//                                = cp + 2*rp + 4*(mma_n%2) + 8*(mma_n/2)
//                                = cp + 2*rp + 4*mma_n
// = same linear index as the C-frag for the 2 m16n8 N-atoms making up this
// K-pair.  No layout helper needed.
//
// `__syncthreads` placement: BEFORE the K-pair loop, so chain matmul-3's
// LDS of smem.C (which is loaded only by warps 0,1) sees cross-warp data.
// Cost: warps 2,3 wait here for warps 0,1 to finish CB precompute, but this
// is off the critical path because warps 0,1 must do CB → replay HMMA
// serially anyway.  This sync also subsumes the post-replay sync that
// `compute_output_int8` would have needed for smem.x / smem.z / smem.CB_scaled.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, typename SmemT,
          typename FragYDxT>
__device__ __forceinline__ void replay_state_mma_int8_chain(
    SmemT& smem, SsuIncrementalParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t cache_slot, int head, bool must_checkpoint, FragYDxT& frag_y_DxT,
    float (&encode_scale_per_row_out)[2], float (&total_scale_out)[2]) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma_int8_chain requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1,
                "replay_state_mma_int8_chain is for 1-byte state_t (int8) only");
  static_assert(D_PER_CTA == 64,
                "replay_state_mma_int8_chain requires D_PER_CTA == 64 (M-shard, per-warp M=16).");

  constexpr int NUM_WARPS = 4;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;  // 16
  static_assert(M_PER_WARP == MMA_prop::M, "Per-warp M must equal m16n8 atom M (=16)");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // Atom-K dispatch: K_BIG=16 (default), K_SMALL=8 if MAX_WINDOW ≤ 8.
  using MmaAtomReplayType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                               MMA_prop::AtomK16, MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // Replay TiledMma: M-shard, 4 warps along M, 1 along N.  Output is
  // ((2,2), 1, NUM_N_PASSES) per thread of fp32 (or bf16 view for new_state).
  auto tiled_mma_replay =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomReplayType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_replay = tiled_mma_replay.get_slice(tid);

  // Chain TiledMma: m16n8k16 (always K_BIG=16 since K=DSTATE/16 atoms ≥ 1),
  // same M-shard layout as replay.  M_per_warp=16 (1 m-atom),
  // N=NPREDICTED_PAD_MMA_M (T_pad, ≤ 16 = up to 2 n-atoms per warp), K=DSTATE.
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);

  constexpr int N_PER_PASS = MMA_prop::N;            // 8
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;  // 16
  constexpr int FRAG_SIZE = 4;
  constexpr int D_ROWS_PER_THREAD = 2;
  constexpr float QUANT_MAX = 127.0f;

  float const total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float const total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  int const lane_d = lane / 4;
  int const warp_d_base = warp * M_PER_WARP;

  // ── Per-row decode_scale for state init.
  auto const* __restrict__ state_scale_ptr = reinterpret_cast<float const*>(params.state_scale);
  int64_t const state_scale_base = cache_slot * params.state_scale_stride_batch +
                                   (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;
  float decode_scale_in[D_ROWS_PER_THREAD];
  decode_scale_in[0] = state_scale_ptr[state_scale_base + warp_d_base + lane_d];
  decode_scale_in[1] = state_scale_ptr[state_scale_base + warp_d_base + lane_d + 8];
  float total_scale[D_ROWS_PER_THREAD];
  total_scale[0] = decode_scale_in[0] * total_decay;
  total_scale[1] = decode_scale_in[1] * total_decay;

  // ── A operand (replay): old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] → LDSM_T.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A_replay = thr_mma_replay.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_replay_view = s2r_thr_A.retile_D(frag_A_replay);
  cute::copy(s2r_A, smem_A_s2r, frag_A_replay_view);

  // ── Bake dB coefficients into frag_A once (8 scale ops), replacing 16×
  // per-N-pass compute_dB_scaling on frag_B (64 scale ops).
  // dB coefficients c[k] baked into frag_A once, replacing per-N-pass B scaling.
  apply_dA_coeff<MAX_WINDOW_PAD_MMA_K>(frag_A_replay, smem, total_cumAdt, prev_k, lane);

  // ── B operand (replay): old_B per-pass.
  auto layout_B_replay = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B_replay);
  auto s2r_B_replay = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_B_replay = s2r_B_replay.get_slice(tid);

  // ── State: int8 input pointer + manual swizzle offsets (read in BOTH passes).
  // Drop bf16 new_state staging — replay's fp32 frag flows directly into the
  // register-resident `new_state` tensor below.
  state_t* state_int8_base = reinterpret_cast<state_t*>(smem.state);

  // Manual swizzle offsets for m16n8 C-fragment layout (int8 Swizzle<3,4,3>).
  // off = row * 128 + (col ^ ((row & 7) << 4)).
  // row_hi = row_lo + 8;  (row+8)&7 == row&7  ⇒  off_hi = off_lo + 1024.
  // Fragment col within each N_PER_PASS=8 tile: (lane % 4) * 2.
  int const row_lo = warp_d_base + lane_d;
  int const frag_col_base = (lane & 3) << 1;
  int const int8_base_lo = row_lo << 7;  // row_lo * DSTATE
  int const int8_xor = (row_lo & 7) << 4;

  float per_thread_amax[D_ROWS_PER_THREAD] = {0.f, 0.f};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
  // ── ldmatrix.b8.x2.trans per-lane addressing constants ──
  // SrcLayout lane decomposition: t = t0 + 2*t1 + 4*t2 + 16*t3.
  int const ldsm_t0 = lane & 1;
  int const ldsm_t1 = (lane >> 1) & 1;
  int const ldsm_t2 = (lane >> 2) & 3;
  int const ldsm_t3 = (lane >> 4) & 1;
  int const ldsm_row = ldsm_t1 * 4 + ldsm_t2 + ldsm_t3 * 8;
  int const ldsm_col_strip = ldsm_t0;

  // m16n8 C-frag → ldmatrix lane remapping for warp shuffles.
  int const t0_mma = lane & 3;
  int const src_lane_cu0 = (lane_d & 3) | (t0_mma << 3);
  int const src_lane_cu1 = src_lane_cu0 + 4;
  // reg_sel picks the register pair for this lane's D-rows:
  //   lane_d 0..3 → regs {0,2}; lane_d 4..7 → regs {1,3}.
  int const reg_sel = (lane_d >> 2);
  int const byte_sel_ru0 = reg_sel * 8;
  int const byte_sel_ru1 = (reg_sel + 2) * 8;

  uint32_t packed[4] = {};
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

  // ── Cross-warp visibility for smem.C / smem.x / smem.z / smem.CB_scaled
  // BEFORE the K-pair fusion loop below — chain matmul-3 reads smem.C from
  // all 4 warps, but smem.C is only loaded by warps 0,1.  Placement before
  // the loop also subsumes the post-replay sync that compute_output_int8
  // would otherwise need.
  // Cost: warps 2,3 wait here for warps 0,1 to finish CB precompute.  Off
  // the critical path because warps 0,1 must do CB → replay HMMA serially
  // anyway.
  __syncthreads();

  // ── smem.C view + B-operand TiledCopy for chain matmul-3 (hoisted before
  // the loop; same view per K-pair, B sliced per K-atom inside the loop).
  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  auto s2r_B_chain =
      make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_B_chain = s2r_B_chain.get_slice(tid);

  constexpr int NUM_K_PAIRS = NUM_N_PASSES / 2;  // 8 for DSTATE=128
  static_assert(NUM_N_PASSES % 2 == 0, "Per-K-pair fusion requires NUM_N_PASSES to be even");
  static_assert(MMA_prop::K_BIG == 16, "Chain mma assumes m16n8k16 K-atom = 16");

  // ════════════════════════════════════════════════════════════════════════
  // PASS 1 — fused replay + chain matmul-3 (per-K-pair):
  //   For each kpair ∈ [0, NUM_K_PAIRS):
  //     - Run 2 replay HMMAs (N-passes 2*kpair, 2*kpair+1) → fp32 frag_h × 2.
  //     - Update per-thread amax (bit-exact fp32).
  //     - Pack each pair's 4 fp32 → 4 bf16 into a tiny K-atom-sized A frag
  //       (`a_kpair` shape ((2,2,2), 1, 1) of bf16 = 8 elts/thread = 4
  //       32-bit regs).  Linear positions [local_n*4 .. local_n*4+3]
  //       within `a_kpair` map to the m16n8k16 A operand's (kh=local_n)
  //       slice — proven by the linear-index identity in the deleted
  //       `new_state`-tensor comment above.
  //     - LDS one K-atom of B (smem.C[T_pad, kpair*16..+16]) into a
  //       similarly small `b_kpair` frag (4 32-bit regs / thread).
  //     - `cute::gemm` accumulates one K-atom into `frag_y_DxT`.
  //     - Both `a_kpair` and `b_kpair` go out of scope at iter end → the
  //       compiler frees those ~8 32-bit regs/thread for the next iter.
  // Net: register footprint drops from the 32 regs of the old register-
  // resident `new_state` array (held across the whole loop) to ~8 regs in
  // flight.  Frees ~24 regs/thread → potentially +1-2 blocks/SM occupancy.
  //
  // CRITICAL: the outer kpair loop is `#pragma unroll 1` (runtime loop).
  // Fully unrolling it (8 iters) makes the compiler see 8 *independent*
  // `a_kpair` / `b_kpair` allocations and keep them ALL live in parallel —
  // which is the opposite of what we want, and worse than v16.0's
  // register-resident `new_state` (32 regs).  Runtime loop = same regs
  // reused across iters, ~8 regs in flight at a time.  The inner local_n
  // loop is compile-time unrolled (only 2 iters; data-dependent on
  // frag_h's lane positions, no inter-iter reuse anyway).
  // ════════════════════════════════════════════════════════════════════════
#pragma unroll
  for (int kpair = 0; kpair < NUM_K_PAIRS; ++kpair) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
    // ── ldmatrix.b8.x2.trans: load 16×32 byte tile every 2 kpairs ──
    // Column-major packed layout: packed[i] holds byte i from all 4 dst regs.
    // Shuffles are deferred to the local_n loop (2 per iteration instead of
    // 8 bunched), eliminating the from_cu/cu_ru intermediate arrays.
    if (kpair % 2 == 0) {
      int const col_base = (kpair / 2) * 32;
      int const abs_row = warp_d_base + ldsm_row;
      int const abs_col = col_base + ldsm_col_strip * 16;
      int const swizzled_col = abs_col ^ ((abs_row & 7) << 4);

      ldsm_b8x2_trans_packed(state_int8_base + abs_row * DSTATE + swizzled_col, packed[0],
                             packed[1], packed[2], packed[3]);
    }
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

    // K-atom-sized A frag for chain matmul-3 (filled across the 2 N-passes).
    Tensor a_kpair = thr_mma_chain.partition_fragment_A(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MMA_prop::K_BIG>{})));
    static_assert(decltype(size(a_kpair))::value == 8,
                  "a_kpair must hold 1 m16n8k16 K-atom of A = 8 bf16/thread");

#pragma unroll
    for (int local_n = 0; local_n < 2; ++local_n) {
      int const n = kpair * 2 + local_n;
      int const n_base = n * N_PER_PASS;

      Tensor frag_h = thr_mma_replay.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));
      static_assert(decltype(size(frag_h))::value == FRAG_SIZE,
                    "FRAG_SIZE must match the partitioned C-fragment size");

      // Zero-init accumulator — MMA from scratch, state added after.
      clear(frag_h);

      // Replay B operand load.
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B_replay.partition_S(smem_B_n);
      Tensor frag_B_replay = thr_mma_replay.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_replay_view = s2r_thr_B_replay.retile_D(frag_B_replay);
      cute::copy(s2r_B_replay, smem_B_s2r_n, frag_B_replay_view);

      // Replay HMMA: frag_h = frag_A_scaled @ frag_B (c[k] baked into A).
      cute::gemm(tiled_mma_replay, frag_h, frag_A_replay, frag_B_replay, frag_h);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
      {
        int const np_in_tile = (kpair & 1) * 2 + local_n;
        uint32_t cu0 = __shfl_sync(constants::MASK_ALL_LANES, packed[np_in_tile], src_lane_cu0);
        uint32_t cu1 = __shfl_sync(constants::MASK_ALL_LANES, packed[np_in_tile], src_lane_cu1);

        frag_h(0) += (float)(int8_t)(cu0 >> byte_sel_ru0) * total_scale[0];
        frag_h(1) += (float)(int8_t)(cu1 >> byte_sel_ru0) * total_scale[0];
        frag_h(2) += (float)(int8_t)(cu0 >> byte_sel_ru1) * total_scale[1];
        frag_h(3) += (float)(int8_t)(cu1 >> byte_sel_ru1) * total_scale[1];
      }
#else
      {
        int const off_lo = int8_base_lo + ((frag_col_base + n_base) ^ int8_xor);
        Pair<int8_t> const p0 = *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo]);
        frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
        frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
        Pair<int8_t> const p1 =
            *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo + 1024]);
        frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
        frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
      }
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

      // Update amax (fp32, bit-exact) AND pack 4 fp32 → 4 bf16 into a_kpair
      // at offset local_n*4 (matches A-frag's (kh=local_n) slice).
#pragma unroll
      for (int i = 0; i < FRAG_SIZE; i += 2) {
        int const d_idx = i / 2;
        float const a0 = fabsf(frag_h(i));
        float const a1 = fabsf(frag_h(i + 1));
        per_thread_amax[d_idx] = fmaxf(per_thread_amax[d_idx], fmaxf(a0, a1));

        Pair<MMA_prop::operand_t> const q =
            pack_float2<MMA_prop::operand_t>(make_float2(frag_h(i), frag_h(i + 1)));
        *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&a_kpair(local_n * FRAG_SIZE + i)) = q;
      }
    }

    // ── B operand for chain matmul-3 K-atom: smem.C[T_pad, kpair*16..+16] ──
    Tensor smem_C_k =
        local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<MMA_prop::K_BIG>{}),
                   make_coord(_0{}, kpair));
    auto smem_C_k_s2r = s2r_thr_B_chain.partition_S(smem_C_k);
    Tensor b_kpair = thr_mma_chain.partition_fragment_B(
        make_tensor((MMA_prop::operand_t*)0x0,
                    make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<MMA_prop::K_BIG>{})));
    auto b_kpair_view = s2r_thr_B_chain.retile_D(b_kpair);
    cute::copy(s2r_B_chain, smem_C_k_s2r, b_kpair_view);

    // Single K-atom chain matmul-3: frag_y_DxT += a_kpair @ b_kpair
    // frag_y_DxT (pre-zeroed by caller) accumulates across all 8 K-atoms.
    cute::gemm(tiled_mma_chain, frag_y_DxT, a_kpair, b_kpair, frag_y_DxT);
  }

  // ── Warp-local amax reduce (Layout<_4,_1> → fully warp-local; no atomics).
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 1));
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 2));
  }

  // ── encode/decode scales (Triton fall-through for amax==0).
  float encode_scale_per_row[D_ROWS_PER_THREAD];
  float decode_scale_per_row[D_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    float const a = per_thread_amax[i];
    encode_scale_per_row[i] = (a == 0.f) ? 1.f : (QUANT_MAX / a);
    decode_scale_per_row[i] = (a == 0.f) ? 1.f : (a / QUANT_MAX);
  }

  // ── STG decode_scale (one writer per (cache, head, d_row)).
  if (must_checkpoint && (lane & 3) == 0) {
    auto* __restrict__ state_scale_w = reinterpret_cast<float*>(params.state_scale);
#pragma unroll
    for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
      int const d_row_in_atom = lane_d + (i & 1) * 8;
      int const d_row = warp_d_base + d_row_in_atom;
      state_scale_w[state_scale_base + d_row] = decode_scale_per_row[i];
    }
  }

  // Hand `encode_scale_per_row` AND `total_scale` (= OLD decode_scale_in ×
  // total_decay) to the caller so PASS 2 (encode replay-again) can:
  //   - dequantize the OLD int8 state with the OLD decode_scale (NOT the NEW
  //     one we just STG'd above — re-reading params.state_scale in PASS 2
  //     would pick up the new value and corrupt the encode), and
  //   - encode the NEW state with the right encode_scale = 127 / amax.
  encode_scale_per_row_out[0] = encode_scale_per_row[0];
  encode_scale_per_row_out[1] = encode_scale_per_row[1];
  total_scale_out[0] = total_scale[0];
  total_scale_out[1] = total_scale[1];
}

// ─────────────────────────────────────────────────────────────────────────
// encode_state_replay_int8: PASS 2 of the int8 chain rewrite.
//
// Re-runs the replay matmul fresh (replay-again), encodes the post-replay
// state fp32 → int8 using `encode_scale_per_row[]` from PASS 1, and STG.16's
// the int8 pairs to gmem.  Bit-exact with Triton's fp32-encode path.
//
// Called *after* `compute_output_int8` so that:
//   - `frag_y_DxT`'s 8 fp32 regs are dead (chain matmul-3's accumulator
//     was consumed by the output STG).
//   - PASS 2's gmem STGs fire alongside `store_old_x` / dt_proc / cumAdt
//     writes — all gmem traffic at the kernel tail where there's nothing
//     else to do.
//
// The setup (TiledMma, frag_A_replay, smem layouts) is duplicated
// from `replay_state_mma_int8_chain` — separate stack frame keeps register
// allocation simple and avoids cross-function lifetime tracking.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, typename SmemT>
__device__ __forceinline__ void encode_state_replay_int8(SmemT& smem,
                                                         SsuIncrementalParams const& params,
                                                         int warp, int lane, int prev_k, int d_tile,
                                                         int64_t cache_slot, int head,
                                                         float const (&encode_scale_per_row)[2],
                                                         float const (&total_scale)[2]) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "encode_state_replay_int8 requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1, "encode_state_replay_int8 is for 1-byte state_t (int8) only");
  static_assert(D_PER_CTA == 64,
                "encode_state_replay_int8 requires D_PER_CTA == 64 (M-shard, per-warp M=16).");

  constexpr int NUM_WARPS = 4;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;
  static_assert(M_PER_WARP == MMA_prop::M, "Per-warp M must equal m16n8 atom M (=16)");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  int const tid = warp * warpSize + lane;

  using MmaAtomReplayType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                               MMA_prop::AtomK16, MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  auto tiled_mma_replay =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomReplayType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_replay = tiled_mma_replay.get_slice(tid);

  constexpr int N_PER_PASS = MMA_prop::N;
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;
  constexpr int FRAG_SIZE = 4;
  constexpr int D_ROWS_PER_THREAD = 2;

  float const total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;

  // total_scale (= OLD decode_scale_in × total_decay) was computed in PASS 1
  // and is passed in by reference.  We MUST NOT re-load decode_scale_in from
  // params.state_scale here — by the time PASS 2 runs, PASS 1 has already
  // STG'd the NEW decode_scale to that same gmem location.

  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A_replay = thr_mma_replay.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_replay_view = s2r_thr_A.retile_D(frag_A_replay);
  cute::copy(s2r_A, smem_A_s2r, frag_A_replay_view);

  // dB coefficients baked into frag_A (same identity as PASS 1).
  apply_dA_coeff<MAX_WINDOW_PAD_MMA_K>(frag_A_replay, smem, total_cumAdt, prev_k, lane);

  auto layout_B_replay = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B_replay);
  auto s2r_B_replay = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_B_replay = s2r_B_replay.get_slice(tid);

  state_t* state_int8_base = reinterpret_cast<state_t*>(smem.state);

  // Manual swizzle offsets (same derivation as replay_state_mma_int8_chain).
  int const lane_d = lane / 4;
  int const warp_d_base = warp * M_PER_WARP;
  int const row_lo = warp_d_base + lane_d;
  int const frag_col_base = (lane & 3) << 1;
  int const int8_base_lo = row_lo << 7;
  int const int8_xor = (row_lo & 7) << 4;

  // Keep layout_state_int8_swz for the cooperative STG.128 at the end.
  auto layout_state_int8_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
  int const lane_d_p2 = lane / 4;
  int const warp_d_base_p2 = warp * M_PER_WARP;

  int const ldsm_t0_p2 = lane & 1;
  int const ldsm_t1_p2 = (lane >> 1) & 1;
  int const ldsm_t2_p2 = (lane >> 2) & 3;
  int const ldsm_t3_p2 = (lane >> 4) & 1;
  int const ldsm_row_p2 = ldsm_t1_p2 * 4 + ldsm_t2_p2 + ldsm_t3_p2 * 8;
  int const ldsm_col_strip_p2 = ldsm_t0_p2;

  int const t0_mma_p2 = lane & 3;
  int const src_lane_cu0_p2 = (lane_d_p2 & 3) | (t0_mma_p2 << 3);
  int const src_lane_cu1_p2 = src_lane_cu0_p2 + 4;
  int const reg_sel_p2 = (lane_d_p2 >> 2);
  int const byte_sel_ru0_p2 = reg_sel_p2 * 8;
  int const byte_sel_ru1_p2 = (reg_sel_p2 + 2) * 8;

  uint32_t packed_p2[4] = {};
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

#pragma unroll
  for (int n = 0; n < NUM_N_PASSES; ++n) {
    int const n_base = n * N_PER_PASS;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
    if (n % 4 == 0) {
      int const col_base = (n / 4) * 32;
      int const abs_row = warp_d_base_p2 + ldsm_row_p2;
      int const abs_col = col_base + ldsm_col_strip_p2 * 16;
      int const swizzled_col = abs_col ^ ((abs_row & 7) << 4);

      ldsm_b8x2_trans_packed(state_int8_base + abs_row * DSTATE + swizzled_col, packed_p2[0],
                             packed_p2[1], packed_p2[2], packed_p2[3]);
    }
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

    Tensor frag_h = thr_mma_replay.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

    // Zero-init accumulator — MMA from scratch, state added after.
    clear(frag_h);

    Tensor smem_B_n =
        local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                   make_coord(n, _0{}));
    auto smem_B_s2r_n = s2r_thr_B_replay.partition_S(smem_B_n);
    Tensor frag_B_replay = thr_mma_replay.partition_fragment_B(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
    auto frag_B_replay_view = s2r_thr_B_replay.retile_D(frag_B_replay);
    cute::copy(s2r_B_replay, smem_B_s2r_n, frag_B_replay_view);

    // HMMA: frag_h = frag_A_scaled @ frag_B (c[k] baked into A).
    cute::gemm(tiled_mma_replay, frag_h, frag_A_replay, frag_B_replay, frag_h);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8
    {
      int const np_in_tile = n & 3;
      uint32_t cu0 = __shfl_sync(constants::MASK_ALL_LANES, packed_p2[np_in_tile], src_lane_cu0_p2);
      uint32_t cu1 = __shfl_sync(constants::MASK_ALL_LANES, packed_p2[np_in_tile], src_lane_cu1_p2);

      frag_h(0) += (float)(int8_t)(cu0 >> byte_sel_ru0_p2) * total_scale[0];
      frag_h(1) += (float)(int8_t)(cu1 >> byte_sel_ru0_p2) * total_scale[0];
      frag_h(2) += (float)(int8_t)(cu0 >> byte_sel_ru1_p2) * total_scale[1];
      frag_h(3) += (float)(int8_t)(cu1 >> byte_sel_ru1_p2) * total_scale[1];
    }
#else
    {
      int const off_lo = int8_base_lo + ((frag_col_base + n_base) ^ int8_xor);
      Pair<int8_t> const p0 = *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo]);
      frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
      frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
      Pair<int8_t> const p1 =
          *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo + 1024]);
      frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
      frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
    }
#endif  // __CUDA_ARCH__ >= 1000 && USE_LDMATRIX_INT8

    // ── Encode + in-place STS to smem.state at cols [n*8, n*8+8) ──
    // Overwrites the OLD int8 input *for this n-pass's cols only*.  The next
    // n-pass dequants from a DIFFERENT col band [(n+1)*8, +8) — still OLD —
    // so no read-after-write hazard.  Each warp writes only to its own
    // M-shard rows; cross-warp visibility is established by the
    // __syncthreads after the loop, before the cooperative STG.128 below.
    // (Replaces the previous per-thread STG.16 path, which generated 32
    // STG.16 per lane and ~29% uncoalesced gmem sectors per NCU v16.0.)
    {
      int const off_lo = int8_base_lo + ((frag_col_base + n_base) ^ int8_xor);
      // d_idx=0: row_lo
      float const e0 = encode_scale_per_row[0];
      int8_t const q0_lo = cvt_rni_sat_s8(frag_h(0) * e0);
      int8_t const q1_lo = cvt_rni_sat_s8(frag_h(1) * e0);
      Pair<int8_t> q_lo;
      q_lo.raw = static_cast<uint16_t>(static_cast<uint8_t>(q0_lo) |
                                       (static_cast<uint16_t>(static_cast<uint8_t>(q1_lo)) << 8));
      *reinterpret_cast<Pair<int8_t>*>(&state_int8_base[off_lo]) = q_lo;
      // d_idx=1: row_hi = row_lo + 8, off_hi = off_lo + 1024
      float const e1 = encode_scale_per_row[1];
      int8_t const q0_hi = cvt_rni_sat_s8(frag_h(2) * e1);
      int8_t const q1_hi = cvt_rni_sat_s8(frag_h(3) * e1);
      Pair<int8_t> q_hi;
      q_hi.raw = static_cast<uint16_t>(static_cast<uint8_t>(q0_hi) |
                                       (static_cast<uint16_t>(static_cast<uint8_t>(q1_hi)) << 8));
      *reinterpret_cast<Pair<int8_t>*>(&state_int8_base[off_lo + 1024]) = q_hi;
    }
  }

  // ── Cross-warp visibility: the cooperative STG.128 below reads ALL D-rows
  // of smem.state, but each warp wrote only its own M-shard.  Sync once,
  // then 128 threads cooperatively transfer the full 8 KB to gmem at 16 B
  // per STG.128 — 64 STG.128 per warp × 4 warps = 256 STG.128 total per CTA
  // (vs. 4096 STG.16 for the old per-thread path).
  __syncthreads();

  auto* __restrict__ state_w_base =
      reinterpret_cast<state_t*>(params.state) + cache_slot * params.state_stride_batch +
      (int64_t)head * DIM * DSTATE + (int64_t)d_tile * D_PER_CTA * DSTATE;

  Tensor sState = make_tensor(make_smem_ptr(reinterpret_cast<state_t const*>(smem.state)),
                              layout_state_int8_swz);
  Tensor gState = make_tensor(make_gmem_ptr(state_w_base),
                              make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                          make_stride(Int<DSTATE>{}, Int<1>{})));

  // 16 bytes per STG = 16 int8 elts.  Thread layout (16, 8) × val (1, 16)
  // = 128 threads × 16 cols/thread, covers 16 × 128 = 2 KB per "row block" ×
  // 4 row blocks (= 64 D-rows total) = 8 KB.
  constexpr int kValCols = 16 / sizeof(state_t);  // = 16 for int8
  auto s2g =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, state_t>{},
                      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, Int<kValCols>>>{});
  auto thr_s2g = s2g.get_slice(tid);
  cute::copy(s2g, thr_s2g.partition_S(sState), thr_s2g.partition_D(gState));
}

template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, typename SmemT>
__device__ __forceinline__ void replay_state_mma_int8(SmemT& smem,
                                                      SsuIncrementalParams const& params, int warp,
                                                      int lane, int prev_k, int d_tile,
                                                      int64_t cache_slot, int head,
                                                      bool must_checkpoint) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma_int8 requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1, "replay_state_mma_int8 is for 1-byte state_t (int8) only");
  static_assert(D_PER_CTA == 64,
                "replay_state_mma_int8 requires D_PER_CTA == 64 (d_split == 1).  "
                "The M-shard-per-warp layout needs per-warp M = D_PER_CTA / 4 == 16.");

  constexpr int NUM_WARPS = 4;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;  // 16
  static_assert(M_PER_WARP == MMA_prop::M, "Per-warp M must equal m16n8 atom M (=16)");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Atom-K dispatch (same as replay_state_mma).
  using MmaAtomType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16,
                                         MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // 4 warps along M (D-axis); each warp owns M_PER_WARP D-rows × full DSTATE.
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Per pass: 1 m16n8 atom in N direction (= 8 cols).  16 passes for DSTATE=128.
  constexpr int N_PER_PASS = MMA_prop::N;  // 8
  static_assert(DSTATE % N_PER_PASS == 0, "DSTATE must be divisible by MMA_prop::N");
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;  // 16
  constexpr int FRAG_SIZE = 4;                       // 4 fp32 / thread / pass (one m16n8 atom)
  constexpr int D_ROWS_PER_THREAD = 2;               // one m-atom row pair per thread
  constexpr float QUANT_MAX = 127.0f;                // int8 symmetric range

  float const total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float const total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── Per-thread D-row formula (m16n8 lane layout, M-shard per warp) ──
  // Lane k of warp w → rows {w*M_PER_WARP + k/4, w*M_PER_WARP + k/4 + 8}.
  // The MMA frag pair index `i ∈ {0, 1}` maps to row_lo (i=0) / row_hi (i=1).
  int const lane_d = lane / 4;
  int const warp_d_base = warp * M_PER_WARP;

  // ── Load decode_scale for this thread's 2 D-rows ──
  auto const* __restrict__ state_scale_ptr = reinterpret_cast<float const*>(params.state_scale);
  int64_t const state_scale_base = cache_slot * params.state_scale_stride_batch +
                                   (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  float decode_scale_in[D_ROWS_PER_THREAD];
  decode_scale_in[0] = state_scale_ptr[state_scale_base + warp_d_base + lane_d];
  decode_scale_in[1] = state_scale_ptr[state_scale_base + warp_d_base + lane_d + 8];

  // Combined per-row scale for state init: dequant + total_decay in one mul.
  float total_scale[D_ROWS_PER_THREAD];
  total_scale[0] = decode_scale_in[0] * total_decay;
  total_scale[1] = decode_scale_in[1] * total_decay;

  // ── Load A operand once (shared across all passes) ──
  // CuTe's `make_tiled_copy_A(_, tiled_mma)` partitions A by M across the
  // 4 warps; each warp's `partition_S` returns its M-shard slice.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);
  cute::copy(s2r_A, smem_A_s2r, frag_A_view);

  // Bake dB coefficients into frag_A once (shared across both passes).
  apply_dA_coeff<MAX_WINDOW_PAD_MMA_K>(frag_A, smem, total_cumAdt, prev_k, lane);

  // ── B operand layout (per-pass slice) ──
  // With Layout<_4, _1>, all warps share full B (no warp partition along N).
  auto layout_B = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: smem pointers + manual swizzle offsets ──
  // smem.state holds the int8 input (read in BOTH replay passes for
  // dequant).  smem.new_state holds the post-replay BF16 state — only
  // matmul-3 reads it (via fast LDSM, 2-byte path).  The encode pass
  // re-runs the replay matmul ("replay-again", below) and encodes from
  // fresh fp32 frags in registers — bit-exact with Triton's encode-from-fp32
  // and avoids the bf16-rounding-before-encode bug that would cost ±1 int8
  // cell on rare boundary elements.
  state_t* state_int8_base = reinterpret_cast<state_t*>(smem.state);
  MMA_prop::operand_t* new_state_base = reinterpret_cast<MMA_prop::operand_t*>(smem.new_state);

  // Manual swizzle offsets for m16n8 C-fragment layout.
  // Int8 Swizzle<3,4,3>: off = row*128 + (col ^ ((row&7)<<4)).
  // BF16 Swizzle<3,3,3> tiled (8,64) to (64,128):
  //   off = (row/8*2 + col/64)*512 + (row&7)*64 + ((col&63) ^ ((row&7)<<3)).
  // row_hi = row_lo+8: same &7 ⇒ off_hi = off_lo + 1024 for both layouts.
  int const row_lo = warp_d_base + lane_d;
  int const frag_col_base = (lane & 3) << 1;
  int const int8_base_lo = row_lo << 7;
  int const int8_xor = (row_lo & 7) << 4;
  int const bf16_r_fixed_lo = (row_lo >> 3) * 1024 + (row_lo & 7) * 64;
  int const bf16_xor = (row_lo & 7) << 3;

  // ── Per-thread amax accumulator (per D-row) ──
  // Updated during PASS 1's n-loop from fp32 frag_h values (bit-exact
  // amax, before bf16 cast for new_state).
  float per_thread_amax[D_ROWS_PER_THREAD] = {0.f, 0.f};

  // ════════════════════════════════════════════════════════════════════════
  // PASS 1: replay matmul → bf16 to new_state (matmul-3 input) +
  //         per-thread amax accumulation in fp32 registers
  // ════════════════════════════════════════════════════════════════════════
#pragma unroll
  for (int n = 0; n < NUM_N_PASSES; ++n) {
    int const n_base = n * N_PER_PASS;

    Tensor frag_h = thr_mma.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));
    static_assert(decltype(size(frag_h))::value == FRAG_SIZE,
                  "FRAG_SIZE must match the partitioned C-fragment size");

    // Zero-init accumulator — MMA from scratch, state added after.
    clear(frag_h);

    // Per-pass B operand load.
    Tensor smem_B_n =
        local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                   make_coord(n, _0{}));
    auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);
    Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
    auto frag_B_view = s2r_thr_B.retile_D(frag_B);
    cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

    // HMMA: frag_h = frag_A_scaled @ frag_B (c[k] baked into A).
    cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

    {
      int const off_lo = int8_base_lo + ((frag_col_base + n_base) ^ int8_xor);
      Pair<int8_t> const p0 = *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo]);
      frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
      frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
      Pair<int8_t> const p1 =
          *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo + 1024]);
      frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
      frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
    }

    // Update per-thread amax (fp32, bit-exact) AND write bf16 cast to
    // smem.new_state for matmul-3.  Single fused loop so frag_h can be
    // discarded right after.
    {
      int const col = frag_col_base + n_base;
      int const bf16_off_lo = bf16_r_fixed_lo + ((col & 64) << 3) + ((col & 63) ^ bf16_xor);

      float const a0_lo = fabsf(frag_h(0));
      float const a1_lo = fabsf(frag_h(1));
      per_thread_amax[0] = fmaxf(per_thread_amax[0], fmaxf(a0_lo, a1_lo));
      Pair<MMA_prop::operand_t> const q_lo =
          pack_float2<MMA_prop::operand_t>(make_float2(frag_h(0), frag_h(1)));
      *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&new_state_base[bf16_off_lo]) = q_lo;

      float const a0_hi = fabsf(frag_h(2));
      float const a1_hi = fabsf(frag_h(3));
      per_thread_amax[1] = fmaxf(per_thread_amax[1], fmaxf(a0_hi, a1_hi));
      Pair<MMA_prop::operand_t> const q_hi =
          pack_float2<MMA_prop::operand_t>(make_float2(frag_h(2), frag_h(3)));
      *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&new_state_base[bf16_off_lo + 1024]) = q_hi;
    }
  }

  // ── Per-warp amax reduction across the 4 col-lanes within an m16n8 atom ──
  // Each m16n8 atom has 4 col-lanes per row pair (cols {2*(k%4), 2*(k%4)+1}).
  // Across all NUM_N_PASSES n-atoms, each lane's accumulator covers 8 cols ×
  // NUM_N_PASSES = 32 cols of one row pair.  After two `__shfl_xor`s (mask 1,
  // 2), all 4 col-lanes of a row pair share the same per-row amax over the
  // FULL 128 N-cols (since per-warp covers full N=DSTATE in the M-shard
  // layout, all amax data is warp-local — no atomic, no __syncthreads).
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 1));
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 2));
  }

  // ── Compute encode_scale + decode_scale per row ──
  // Match Triton's `where(amax == 0, 1, QUANT_MAX/amax)` fall-through.
  float encode_scale_per_row[D_ROWS_PER_THREAD];
  float decode_scale_per_row[D_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    float const a = per_thread_amax[i];
    encode_scale_per_row[i] = (a == 0.f) ? 1.f : (QUANT_MAX / a);
    decode_scale_per_row[i] = (a == 0.f) ? 1.f : (a / QUANT_MAX);
  }

  // ── Store decode_scale to gmem (one writer per (cache, head, d_row)) ──
  // 4 col-lanes share the same per-row amax; pick lane%4==0 as the writer.
  // Per warp: 8 writers × 2 D-rows = 16 D-rows.  Across 4 warps: 64 D-rows
  // = D_PER_CTA. ✓
  if (must_checkpoint && (lane & 3) == 0) {
    auto* __restrict__ state_scale_w = reinterpret_cast<float*>(params.state_scale);
#pragma unroll
    for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
      int const d_row_in_atom = lane_d + (i & 1) * 8;
      int const d_row = warp_d_base + d_row_in_atom;
      state_scale_w[state_scale_base + d_row] = decode_scale_per_row[i];
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // PASS 2 (replay-again): re-run the replay matmul, encode + STG int8 to
  //                        gmem from fresh fp32 frags (bit-exact).
  // ════════════════════════════════════════════════════════════════════════
  // Cost: 1× extra replay matmul vs the staging-based path.  Avoids the
  // 32 KB fp32 staging buffer and the bf16-rounding precision loss; lets
  // matmul-3 keep its bf16 LDSM fast path via smem.new_state.
  if (must_checkpoint) {
    auto* __restrict__ state_w =
        reinterpret_cast<state_t*>(params.state) + cache_slot * params.state_stride_batch +
        (int64_t)head * DIM * DSTATE + (int64_t)d_tile * D_PER_CTA * DSTATE;

#pragma unroll
    for (int n = 0; n < NUM_N_PASSES; ++n) {
      int const n_base = n * N_PER_PASS;

      Tensor frag_h = thr_mma.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

      // Zero-init accumulator (same pattern as PASS 1).
      clear(frag_h);

      // B operand load (same as PASS 1).
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);
      Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_view = s2r_thr_B.retile_D(frag_B);
      cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

      // HMMA — re-derive the post-replay state (c[k] in frag_A).
      cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

      {
        int const off_lo = int8_base_lo + ((frag_col_base + n_base) ^ int8_xor);
        Pair<int8_t> const p0 = *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo]);
        frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
        frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
        Pair<int8_t> const p1 =
            *reinterpret_cast<Pair<int8_t> const*>(&state_int8_base[off_lo + 1024]);
        frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
        frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
      }

      // Encode + STG int8 in m16n8 layout.  Each lane has 4 int8 values
      // (2 row-pairs × 2 col-pairs).  Per row-pair, 2 contiguous int8 cols
      // → 1 STG.16 (uint16_t).  2 STG.16 per lane per atom × NUM_N_PASSES
      // atoms.
      {
        int const col = frag_col_base + n_base;
        int const off_gmem_lo = int8_base_lo + col;  // row_lo * DSTATE + col (no swizzle)
        // d_idx=0: row_lo
        float const e0 = encode_scale_per_row[0];
        int8_t const q0_lo = cvt_rni_sat_s8(frag_h(0) * e0);
        int8_t const q1_lo = cvt_rni_sat_s8(frag_h(1) * e0);
        Pair<int8_t> q_lo;
        q_lo.raw = static_cast<uint16_t>(static_cast<uint8_t>(q0_lo) |
                                         (static_cast<uint16_t>(static_cast<uint8_t>(q1_lo)) << 8));
        *reinterpret_cast<Pair<int8_t>*>(&state_w[off_gmem_lo]) = q_lo;
        // d_idx=1: row_hi, off = off_gmem_lo + 8*DSTATE = off_gmem_lo + 1024
        float const e1 = encode_scale_per_row[1];
        int8_t const q0_hi = cvt_rni_sat_s8(frag_h(2) * e1);
        int8_t const q1_hi = cvt_rni_sat_s8(frag_h(3) * e1);
        Pair<int8_t> q_hi;
        q_hi.raw = static_cast<uint16_t>(static_cast<uint8_t>(q0_hi) |
                                         (static_cast<uint16_t>(static_cast<uint8_t>(q1_hi)) << 8));
        *reinterpret_cast<Pair<int8_t>*>(&state_w[off_gmem_lo + 8 * DSTATE]) = q_hi;
      }
    }
  }
}

// =============================================================================
// Phase 1b: Replay — tensor-core MMA path (matmul 2: state recurrence).
// state[D, dstate] = state * total_decay + old_x^T @ (coeff * old_B)
// All 128 threads cooperate.
//
// Warps along N=DSTATE:
//   TiledMMA uses Layout<_1, _4> — per pass covers (M=DIM, N=4×MMA_prop::N=32).
//   Each warp owns: full M (DIM/16 m-atoms) and one n-atom of 8 cols.
//   Why: A is small (DIM × K), B is bigger (DSTATE × K).  M-split (`_4×1`)
//   redundantly loaded full B from each warp (4× × 4 KB = 16 KB).  N-split
//   (`_1×4`) instead redundantly loads full A (4× × 2 KB = 8 KB) and reads
//   B disjointly across warps — net smem read drops 18 KB → 12 KB per replay
//   (~33%) at K_BIG.  Also unlocks D-split D_PER_CTA < 64.
// =============================================================================
// state_w_base (f16+philox path): pre-offset gmem pointer to this CTA's owned
// [D_PER_CTA, DSTATE] state slice (params.state + cache_slot *
// state_stride_batch + head * DIM*DSTATE + d_tile * D_PER_CTA*DSTATE).
// Computed in the kernel preamble.  Combining base + offset into one i64
// pointer drops the cross-iter live-range cost from 4 regs (state_w ptr +
// state_gmem_off) to 2 regs (just the base), and the per-pair STG.32 uses an
// i32 element offset inside the chunk.  Use this instead of separately
// holding params.state-ptr and state_gmem_off.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          typename SmemT>
__device__ __forceinline__ void replay_state_mma(SmemT& smem, SsuIncrementalParams const& params,
                                                 int warp, int lane, int prev_k, int d_tile,
                                                 uint32_t state_ptr_offset, state_t* state_w_base,
                                                 int64_t rand_seed, bool must_checkpoint) {
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

  // ── A operand: old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] Swizzle<3,3,3>, transposed
  // view [M=D_SMEM_COLS, K=MAX_WINDOW_PAD_MMA_K].  D_SMEM_COLS may be padded above
  // D_PER_CTA when D_PER_CTA < swizzle atom; local_tile to D_PER_CTA
  // restricts the LDSM to the valid sub-tile.  Each warp loads the FULL M (4×
  // redundant across warps).  See header comment for traffic accounting. ──
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
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
  constexpr int DB_COEFFS_PER_LANE = MAX_WINDOW_PAD_MMA_K / LANES_PER_N_COL;
  float dB_coeff[DB_COEFFS_PER_LANE];
  precompute_dB_coeff<DB_COEFFS_PER_LANE>(dB_coeff, smem, total_cumAdt, prev_k, lane);

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
  constexpr int PAIRS_PER_PASS = D_PER_CTA / 8;  // = (D_PER_CTA/16) × 2 row-pair iters
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
          uint32_t const philox_off = state_ptr_offset +
                                      static_cast<uint32_t>(d_tile * D_PER_CTA + row) * DSTATE +
                                      static_cast<uint32_t>(col);
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

// SM80 m16n8k16 C-frag → A-frag layout reshape for chained mma (state →
// matmul-3 in the int8 chain rewrite).
//
// Pattern mirrors the SM90 helper at attention/hopper/utils.cuh:103, but:
//   - SM80 m16n8 C-frag inner per-thread layout is rank-2 ((col_pair=2,
//     row_pair=2)) — there's no inner "N/8" stride mode like SM90.
//   - We instead `logical_divide` the *outer* MMA_N axis by 2: each pair of
//     m16n8 N-atoms (= 16 cols of the producing mma's N) becomes one K=16
//     atom of the chained m16n8k16 mma's A operand.
//
// Lane-element mapping (verified by hand on the m16n8k16 PTX layout):
//   C-frag at (cp, rp, mma_n=2k+kh) maps to:    row=tid/4+rp*8,  col=4*(tid%2)+cp+(2k+kh)*8
//   A-frag at (cp, rp, kh, mma_k=k)   maps to:    row=tid/4+rp*8,  col=4*(tid%2)+cp+8*kh + 16k
//   Same element: (2k+kh)*8 + cp == 8*kh + cp + 16k. ✓
//
// Input layout:  ((2, 2),    MMA_M, MMA_N)            — m16n8 C-frag
// Output layout: ((2, 2, 2), MMA_M, MMA_N / 2)        — m16n8k16 A-frag,
//                                                       MMA_K = MMA_N / 2
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs_sm80(Layout acc_layout) {
  using namespace cute;
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2,
                "C-frag inner mode must be (col_pair=2, row_pair=2)");
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2,
                "C-frag inner mode must be (col_pair=2, row_pair=2)");
  static_assert(decltype(rank(acc_layout))::value == 3,
                "C-frag must be rank-3 ((C0,C1), MMA_M, MMA_N)");
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 2,
                "SM80 m16n8 C-frag inner is rank-2 (no inner stride mode like SM90)");
  // logical_divide the outer MMA_N axis by 2 → ((2, 2), MMA_M, (2, MMA_N/2))
  auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
  return make_layout(
      make_layout(get<0, 0>(l), get<0, 1>(l), get<2, 0>(l)),  // ((col_pair, row_pair, k_half))
      get<1>(l),                                              // MMA_M
      get<2, 1>(l));                                          // MMA_K = MMA_N / 2
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
template <typename input_t, typename MmaT, int N_TILE, int NPREDICTED_PAD_MMA_M, typename FragY,
          typename FragCB, typename SmemXTrans, typename S2RBTrans, typename S2RThrBTrans,
          typename ThrMma, typename TiledMma>
__device__ __forceinline__ void add_cb_x(FragY& frag_y, FragCB const& frag_CB,
                                         SmemXTrans const& smem_x_trans,
                                         S2RBTrans const& s2r_B_trans,
                                         S2RThrBTrans const& s2r_thr_B_trans, ThrMma const& thr_mma,
                                         TiledMma const& tiled_mma, int n) {
  using namespace cute;
  Tensor smem_x_trans_ntile = local_tile(
      smem_x_trans, make_tile(Int<N_TILE>{}, Int<NPREDICTED_PAD_MMA_M>{}), make_coord(n, _0{}));
  auto smem_x_trans_s2r = s2r_thr_B_trans.partition_S(smem_x_trans_ntile);
  auto frag_B_x = thr_mma.partition_fragment_B(
      make_tensor((MmaT*)0x0, make_shape(Int<N_TILE>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_B_x_view = s2r_thr_B_trans.retile_D(frag_B_x);

  cute::copy(s2r_B_trans, smem_x_trans_s2r, frag_B_x_view);
  cute::gemm(tiled_mma, frag_y, frag_CB, frag_B_x, frag_y);
}

// 3b. frag_y += D * x[t, d]  (per-thread skip connection via partition_C)
template <typename input_t, int NPREDICTED_PAD_MMA_M, int N_TILE, typename FragY, typename SmemX,
          typename ThrMma>
__device__ __forceinline__ void add_D_skip(FragY& frag_y, SmemX const& smem_x,
                                           ThrMma const& thr_mma, float D_val, int n) {
  using namespace cute;
  if (D_val == 0.f) return;
  Tensor smem_x_tile = local_tile(smem_x, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                  make_coord(_0{}, n));
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
template <typename input_t, int NPREDICTED_PAD_MMA_M, int N_TILE, typename FragY, typename SmemZ,
          typename ThrMma>
__device__ __forceinline__ void compute_z_gating(FragY& frag_y, SmemZ const& smem_z,
                                                 ThrMma const& thr_mma, void const* z_ptr, int n) {
  using namespace cute;
  if (!z_ptr) return;
  Tensor smem_z_tile = local_tile(smem_z, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                  make_coord(_0{}, n));
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
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  // Smem source dtype for matmul-3:
  //   - sizeof(state_t) == 1 (int8): read from `smem.new_state` (bf16) — the
  //     post-replay state staged by `replay_state_mma_int8` PASS 1 as bf16
  //     (cast from fp32 frag).  Goes through pipelined_kloop_gemm's 2-byte
  //     LDSM fast path; the bf16 in smem matches `MMA_prop::operand_t` so no
  //     conversion is needed in `convert_frag`.  smem.state holds the
  //     pre-replay int8 input and is consumed by replay_state_mma_int8's
  //     PASS 2 (encode replay-again) — not read by matmul-3.
  //   - sizeof(state_t) == 2 (fp16/bf16): LDSM the native 16-bit, view as bf16.
  //   - sizeof(state_t) == 4 (fp32): scalar UniversalCopy + on-the-fly convert.
  constexpr bool is_int8_smem = (sizeof(state_t) == 1);
  constexpr bool is_2byte_smem = (sizeof(state_t) == 2);
  using state_view_t =
      std::conditional_t<is_int8_smem, MMA_prop::operand_t,
                         std::conditional_t<is_2byte_smem, MMA_prop::operand_t, state_t>>;
  // BTypeIn into pipelined_kloop_gemm: int8 path uses `MMA_prop::operand_t`
  // (bf16) since `smem.new_state` is bf16.  pipelined_kloop_gemm dispatches
  // its B-load atom on `sizeof(BTypeIn)` — 2-byte → LDSM (SM75_U32x2_LDSM_N)
  // and `convert_frag<bf16, bf16>` is a no-op bit copy.
  using BTypeIn = std::conditional_t<is_int8_smem, MMA_prop::operand_t, state_t>;

  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  Tensor smem_C_ktiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                    make_coord(_0{}, _));

  // Swizzle layout matches the dtype of the buffer being viewed.  For int8
  // we point at `smem.new_state` (fp32); for non-int8 we point at `smem.state`.
  auto const layout_state_swz = make_swizzled_layout_rc<BTypeIn, D_PER_CTA, DSTATE>();
  state_view_t const* smem_state_ptr = [&]() -> state_view_t const* {
    if constexpr (is_int8_smem) {
      return reinterpret_cast<state_view_t const*>(smem.new_state);
    } else {
      return reinterpret_cast<state_view_t const*>(smem.state);
    }
  }();
  Tensor smem_state = make_tensor(make_smem_ptr(smem_state_ptr), layout_state_swz);

  pipelined_kloop_gemm<3, NUM_K_TILES, input_t, BTypeIn, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_C_ktiled, smem_state, frag_y...);
}

// store_state: vectorized smem → gmem state writeback (128 threads).
// Defined here (rather than alongside the other Phase 3 store helpers
// below) because compute_and_store_output calls it inline — issued right
// after matmul 3 so the STGs fire-and-forget in parallel with matmul 4 +
// epilogue.  smem and gmem hold the same dtype now (no on-egress
// conversion) so this is always a direct 128-bit copy.
template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void store_state(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot) {
  using namespace cute;
  int const flat_tid = warp * warpSize + lane;
  auto* __restrict__ state_w = reinterpret_cast<state_t*>(params.state);
  // gmem dest = head's full state base + d_tile's row slice.
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
template <typename input_t, typename state_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE,
          int NUM_WARPS, int PHILOX_ROUNDS, typename SmemT>
__device__ __forceinline__ void compute_and_store_output(
    SmemT& smem, SsuIncrementalParams const& params, int warp, int lane, int d_tile, int batch_idx,
    int head, int64_t cache_slot, float D_val, bool must_checkpoint) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_and_store_output requires 2-byte input type");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA: 128 threads, covers [16, 32] output per step ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  // When D_PER_CTA < swizzle atom (= 64 for bf16), the underlying
  // smem buffer is padded to D_SMEM_COLS so the swizzle layout is well-formed.
  // Per-pass MMA loops only iterate D_PER_CTA / N_TILE tiles → never touch
  // the padded tail.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;

  // x: swizzled [NPREDICTED_PAD_MMA_M, D_SMEM_COLS]
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)),
                              layout_x_swz);
  auto layout_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans_swz);

  // z: aliased swizzled [NPREDICTED_PAD_MMA_M, D_SMEM_COLS] — physical buffer
  // is only next_multiple_of<ATOM_ROWS>(NPREDICTED) rows tall; second m-tile
  // aliases first.  Ghost rows feed predicated-out output rows.
  auto layout_z_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS, NPREDICTED>();
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
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // Decay broadcast: cumAdt[t] → [NPREDICTED_PAD_MMA_M, N_TILE] with stride-0 on N.
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output: partition_C for direct register → gmem store ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  // out_base lands on this CTA's D-slice within the head.
  int64_t const out_base = (int64_t)batch_idx * params.out_stride_batch + (int64_t)head * DIM +
                           (int64_t)d_tile * D_PER_CTA;

  // Row predicate for padding.  The epilogue store loop iterates i in steps
  // of 2 and only consults pred(0) and pred(2) — m16n8k16 C-frag per thread
  // has 4 elts at rows {t/4, t/4, t/4+8, t/4+8}, so there are only 2 unique
  // row predicates.  Compute them once and skip the 4-wide pred tensor.
  auto id_tile = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < NPREDICTED;
  bool const pred_row_hi = get<0>(id_part(2)) < NPREDICTED;

  // Number of output N-tiles per pass = D_PER_CTA / N_TILE.
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
    add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // frag_y += D * x[t, d]
    add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // frag_y *= z * sigmoid(z)
    compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Store frag_y directly to gmem (register → gmem, no smem round-trip).
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
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

  // Skip the smem→gmem state copy when:
  //   - philox+f16: `replay_state_mma` already did the gmem store with
  //     stochastic rounding direct from registers.
  //   - int8 (sizeof == 1): `replay_state_mma_int8` already did the gmem
  //     store with RN encoding direct from registers (and smem.state holds
  //     the *input* int8, not the post-replay output, so a copy would
  //     overwrite gmem with stale data).
  constexpr bool kSkipSmemToGmemState =
      ((PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>) || (sizeof(state_t) == 1);

  // ── Matmul 3 + store_state + epilogue, dispatching on NUM_N_TILES ──
  // (NumNTiles is deduced from the variadic frag_y... pack in `add_init_out`.)
  if constexpr (NUM_N_TILES == 2) {
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0,
                                                      frag_y_1);
    // State writeback hoisted here — after matmul 3 has finished consuming
    // smem.state, before matmul 4 which reads only smem.x / smem.CB_scaled
    // / smem.z.  STGs fire-and-forget alongside the epilogue (matmul 4 +
    // D*x + z-gate + output STG).  Runtime-gated on must_checkpoint:
    // non-checkpoint steps leave the prior state HBM intact (saving
    // bandwidth — that's the perf win of the checkpointing design).
    if constexpr (!kSkipSmemToGmemState) {
      if (must_checkpoint) {
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot);
      }
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
      if (must_checkpoint) {
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot);
      }
    }
    epilogue(frag_y_0, 0);
  }
}

// ────────────────────────────────────────────────────────────────────────
// compute_output_int8: transposed matmul-4 + epilogue + smem-transpose STG
// ────────────────────────────────────────────────────────────────────────
// Companion to `replay_state_mma_int8_chain`.  Consumes the per-warp
// `frag_y_DxT` (shape ((2,2), 1, T_pad/8) of fp32 per thread; M=D-shard,
// N=T_pad) — pre-loaded with init_out^T from chain matmul-3 — and:
//   1. Decay broadcast: frag_y_DxT *= exp(cumAdt[t])  (per T-col, scalar LDS).
//   2. Chain matmul-4 transposed: frag_y_DxT += x^T[D, T] @ CB_scaled^T[T, T]
//        A operand: smem.x viewed via x_trans (D, T) → LDSM_N feeds A(M=D, K=T).
//        B operand: smem.CB_scaled (T, T) → LDSM_T feeds B(K=T, N=T).
//   3. D*x skip: frag_y_DxT(d, t) += D_val * x[t, d]  (scalar LDS per element;
//        consecutive frag elts at fixed D, varying T → not pair-loadable).
//   4. z-gate: frag_y_DxT *= z * sigmoid(z)            (scalar LDS per element).
//   5. fp32 → input_t pack (in-place register cvt via pack_float2).
//   6. Per-thread STS to smem.output_transpose at (T, D) layout.
//   7. __syncthreads.
//   8. Cooperative STG.128 from smem.output_transpose (T, D) to gmem (T, D).
//
// Cross-warp dependencies (smem.x, smem.z, smem.CB_scaled) are already
// visible because `replay_state_mma_int8_chain` did a __syncthreads before
// chain matmul-3, which subsumes the post-replay sync.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS,
          typename SmemT, typename FragYDxT>
__device__ __forceinline__ void compute_output_int8(SmemT& smem, SsuIncrementalParams const& params,
                                                    int warp, int lane, int d_tile, int batch_idx,
                                                    int head, int64_t cache_slot, float D_val,
                                                    FragYDxT& frag_y_DxT) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_output_int8 requires 2-byte input_t");
  static_assert(D_PER_CTA == 64, "compute_output_int8 requires D_PER_CTA == 64");
  static_assert(NUM_WARPS == 4, "compute_output_int8 requires 4 warps");

  int const tid = warp * warpSize + lane;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;  // 16

  // Same TiledMma as replay_state_mma_int8_chain (M-shard, m16n8k16).
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);

  // ── Smem views ──
  // x_trans: x physically stored at (T, D); transposed view at (D, T).
  // Used as the A operand of the chain matmul-4.
  auto layout_x_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans);
  Tensor smem_x_trans_tile =
      local_tile(smem_x_trans, make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{}),
                 make_coord(_0{}, _0{}));

  // x natural (T, D) view — for D-skip + z-gate per-element scalar LDS.
  auto layout_x = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();

  // z natural (T, D) view (aliased so padded rows alias valid rows).
  auto layout_z = make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS,
                                                  SmemT::NPREDICTED>();

  // CB_scaled (T, T_pad) within (NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE).
  auto layout_cb =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb);

  // ── Per-thread (d, t) coord lookup for epilogue scalar reads + smem-transpose write ──
  auto id_tile = make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{}));
  auto id_part = thr_mma_chain.partition_C(id_tile);

  // ── 1. Decay broadcast: frag_y(i) *= exp(cumAdt[t]) ──
  // Per-element scalar read from smem.cumAdt indexed by T-col.  For padded
  // T-cols (t >= NPREDICTED), the read returns garbage but the STG at the
  // end is predicated on t < NPREDICTED, so the garbage never reaches gmem.
#pragma unroll
  for (int i = 0; i < size(frag_y_DxT); ++i) {
    int const t = get<1>(id_part(i));
    if (t < SmemT::NPREDICTED) {
      frag_y_DxT(i) *= __expf(smem.cumAdt[t]);
    }
  }

  // ── 2. Chain matmul-4: frag_y_DxT += x^T @ CB^T ──
  // A operand: smem.x physically (T, D); transposed view (D, T) used as
  // A(M=D, K=T).  The transposed view has D-stride=1, T-stride=D — same
  // pattern as replay's A from old_x — so use LDSM_T to produce row-major
  // A from this column-wise smem source.
  auto s2r_A_x =
      make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_A_x = s2r_A_x.get_slice(tid);
  auto smem_x_s2r = s2r_thr_A_x.partition_S(smem_x_trans_tile);
  Tensor frag_A_x = thr_mma_chain.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_A_x_view = s2r_thr_A_x.retile_D(frag_A_x);
  cute::copy(s2r_A_x, smem_x_s2r, frag_A_x_view);

  // B operand for chain matmul-4 = CB^T.  smem.CB natural view shape (T, T)
  // already has T_inner stride 1 = K-major.  Use LDSM_N (no transpose).
  auto s2r_B_CB =
      make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_B_CB = s2r_B_CB.get_slice(tid);
  auto smem_CB_s2r = s2r_thr_B_CB.partition_S(smem_CB);
  Tensor frag_B_CB = thr_mma_chain.partition_fragment_B(
      make_tensor((MMA_prop::operand_t*)0x0,
                  make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_B_CB_view = s2r_thr_B_CB.retile_D(frag_B_CB);
  cute::copy(s2r_B_CB, smem_CB_s2r, frag_B_CB_view);

  cute::gemm(tiled_mma_chain, frag_y_DxT, frag_A_x, frag_B_CB, frag_y_DxT);

  // ── 3. D*x skip: frag_y(d, t) += D_val * x[t, d] (scalar LDS per element) ──
  if (D_val != 0.f) {
    auto* __restrict__ smem_x_base = reinterpret_cast<input_t const*>(smem.x);
#pragma unroll
    for (int i = 0; i < size(frag_y_DxT); ++i) {
      int const d = get<0>(id_part(i));
      int const t = get<1>(id_part(i));
      if (t < SmemT::NPREDICTED) {
        int const off = layout_x(t, d);
        frag_y_DxT(i) += D_val * toFloat(smem_x_base[off]);
      }
    }
  }

  // ── 4. z-gate: frag_y *= z * sigmoid(z) (scalar LDS per element) ──
  if (params.z != nullptr) {
    auto* __restrict__ smem_z_base = reinterpret_cast<input_t const*>(smem.z);
#pragma unroll
    for (int i = 0; i < size(frag_y_DxT); ++i) {
      int const d = get<0>(id_part(i));
      int const t = get<1>(id_part(i));
      if (t < SmemT::NPREDICTED) {
        int const off = layout_z(t, d);
        float const z = toFloat(smem_z_base[off]);
        frag_y_DxT(i) *= z * __fdividef(1.f, (1.f + __expf(-z)));
      }
    }
  }

  // ── 5. Pack fp32 → input_t per element + 6. STS to smem.output_transpose (T, D) ──
  // Padded row stride (D_PER_CTA + 8 = 72 bf16 = 144 bytes) gives:
  //   - 16-byte-aligned LDS.128 / STG.128 across all rows.
  //   - 4-bank shift per row → m16n8 STS pattern hits {bank 0, 4, 8, 12} for the
  //     4 t-rows of an elt → bank-conflict-free (vs 4-way conflict at stride 64).
  // See SsuIncrementalStorageInt8::OUTPUT_TRANSPOSE_ROW_STRIDE for derivation.
  constexpr int kSmemRowStride = SmemT::OUTPUT_TRANSPOSE_ROW_STRIDE;  // 72 bf16 elts
  auto* __restrict__ smem_out_base = reinterpret_cast<input_t*>(smem.output_transpose);
#pragma unroll
  for (int i = 0; i < size(frag_y_DxT); ++i) {
    int const d = get<0>(id_part(i));
    int const t = get<1>(id_part(i));
    if (t < SmemT::NPREDICTED) {
      // Pack via pack_float2(f, 0.f) and take low elt — emits a single cvt
      // (compiler folds the dummy into a no-op for the discarded high half).
      smem_out_base[t * kSmemRowStride + d] =
          pack_float2<input_t>(make_float2(frag_y_DxT(i), 0.f))[Int<0>{}];
    }
  }

  // ── 7. No sync needed ──
  // Cross-warp __syncthreads is NOT needed: each warp owns 16 D-rows of
  // `frag_y_DxT` (M-shard layout) and writes ONLY to its own D-rows of
  // `output_transpose`.  The warp-local cooperative STG below reads from
  // the same warp's 16 D-rows → no cross-warp dep.
  //
  // No intra-warp __syncwarp either: lanes don't diverge between the STS
  // loop above and the STG loop below, so SM80/90 hardware naturally
  // orders the memory ops in program order across all 32 lanes.  (Formal
  // PTX consistency model would call for a fence; in practice on Ampere/
  // Hopper the lockstep within a non-divergent warp gives us this for
  // free.  cp.async is the case that *does* need an explicit warp sync
  // because the loads complete asynchronously.)

  // ── 8. Warp-local cooperative STG.128: 32 lanes → one warp's 16 D-rows ──
  // Each warp's data: 16 D-rows × T_pad=16 cols × 2 B = 512 B.
  // Re-tile 32 lanes: (t = lane%16, d_group = lane/16 ∈ {0, 1}) → covers
  // T_pad × 2 D-groups = 32 slots, each STG.128 = 8 D-cols × 2 B = 16 B.
  // No cross-warp coordination → no __syncthreads.
  constexpr int kElsPerSTG = 16 / sizeof(input_t);          // 8 bf16 elts per STG.128
  constexpr int kDGroupsPerWarp = M_PER_WARP / kElsPerSTG;  // = 16 / 8 = 2
  static_assert(NPREDICTED_PAD_MMA_M * kDGroupsPerWarp == 32,
                "warp-local STG re-tile: T_pad × dGroupsPerWarp must equal warpSize");

  int const stg_t = lane % NPREDICTED_PAD_MMA_M;
  int const stg_d_group = lane / NPREDICTED_PAD_MMA_M;
  int const warp_d_base = warp * M_PER_WARP;
  int const stg_d = warp_d_base + stg_d_group * kElsPerSTG;

  if (stg_t < NPREDICTED) {
    int const smem_off = stg_t * kSmemRowStride + stg_d;

    auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
    int64_t const out_base = (int64_t)batch_idx * params.out_stride_batch + (int64_t)head * DIM +
                             (int64_t)d_tile * D_PER_CTA;
    int64_t const gmem_off = out_base + (int64_t)stg_t * params.out_stride_mtp + stg_d;

    // 128-bit copy.  smem_off * 2 B = (t * 144 + d * 2) is 16-byte aligned
    // for any t when d % 8 == 0 (here d_offset_within_warp = 0 or 8).
    using Vec = uint4;
    *reinterpret_cast<Vec*>(&output_ptr[gmem_off]) =
        *reinterpret_cast<Vec const*>(&smem_out_base[smem_off]);
  }
}

// ── Store functions (called from kernel after compute_y + sync) ──
// (store_state moved above compute_and_store_output — used there for
// the state-writeback hoist.)

template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, typename SmemT>
__device__ __forceinline__ void store_old_x(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot, int write_offset) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_x_w = reinterpret_cast<input_t*>(params.old_x);
  // gmem dest = head's full slot + d_tile's D-slice offset, shifted by
  // `write_offset` along the T-axis (must_checkpoint ? 0 : prev_k).
  int64_t const ox_w_base = cache_slot * params.old_x_stride_cache +
                            (int64_t)write_offset * params.old_x_stride_mtp + head * DIM +
                            (int64_t)d_tile * D_PER_CTA;

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
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.x)), layout_x_swz);
  Tensor gX = make_tensor(make_gmem_ptr(old_x_w + ox_w_base),
                          make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}),
                                      make_stride(params.old_x_stride_mtp, Int<1>{})));

  using ThrLayoutX = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{}, ThrLayoutX{},
                             Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);

  auto tSsX = thr_s2g.partition_S(sX);
  auto tSgX = thr_s2g.partition_D(gX);

  // Per-(row, col) predicate: skip rows ≥ NPREDICTED (m-padding) and cols ≥
  // D_PER_CTA (atom-padding past the d_tile's data).
  auto cX = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}));
  auto tScX = thr_s2g.partition_D(cX);
  auto pred = make_tensor<bool>(shape(tScX));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = (get<0>(tScX(i)) < NPREDICTED) && (get<1>(tScX(i)) < D_PER_CTA);
  }
  copy_if(s2g, pred, tSsX, tSgX);
}

// store_old_B runs on W0, W1 only (64 threads).  Caller must gate
// with `if (warp < 2)` — these are the warps that hold valid smem.B
// after their own cp.async + wait.  Halving the thread count keeps the
// overlap (writeback fires before CB+replay consume smem.B).
//
// Source: smem.B with NPREDICTED_PAD_MMA_N rows.
// Destination: gmem old_B[buf_write][write_offset:write_offset+NPREDICTED, :].
// The `write_offset` argument shifts the gmem T-axis base — it's added to the
// base pointer below; the per-element predicate masks rows ≥ NPREDICTED.
//
// Thread layout `(8, 8) × (1, 8)` — **atom-aligned** with the Swizzle<3,3,3>
// (8, 64) atom for conflict-free smem reads.  Per-tile 8 × 64 covers one
// full atom.  For NPREDICTED_PAD_MMA_N=16: iters (2, 2) = 4 tiles, each
// thread owns 2 rows (t/8 and t/8+8) → per-iteration row predicate.  For
// =8: iters (1, 2) = 2 tiles, each thread owns 1 row.  The per-element
// predicate works for both.
template <typename input_t, int NPREDICTED, int DSTATE, int HEADS_PER_GROUP, typename SmemT>
__device__ __forceinline__ void store_old_B(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int head, int group_idx,
                                            int64_t cache_slot, int buf_write, int write_offset) {
  using namespace cute;
  if (head % HEADS_PER_GROUP != 0) return;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;  // matches smem.B row count
  // Called only from warps 0, 1 — flat_tid ∈ [0, 64).
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_B_w = reinterpret_cast<input_t*>(params.old_B);
  int64_t const oB_base = cache_slot * params.old_B_stride_cache +
                          buf_write * params.old_B_stride_dbuf +
                          (int64_t)write_offset * params.old_B_stride_mtp + group_idx * DSTATE;

  auto layout_B_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_N, DSTATE>();
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B_swz);
  Tensor gB = make_tensor(make_gmem_ptr(old_B_w + oB_base),
                          make_layout(make_shape(Int<NPREDICTED_PAD_MMA_N>{}, Int<DSTATE>{}),
                                      make_stride(params.old_B_stride_mtp, Int<1>{})));

  // 64 threads, (8, 8) × (1, 8) = atom-aligned per-tile (8, 64).
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_8, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsB = thr_s2g.partition_S(sB);
  auto tSgB = thr_s2g.partition_D(gB);

  if constexpr (NPREDICTED == NPREDICTED_PAD_MMA_N) {
    copy(s2g, tSsB, tSgB);
  } else {
    // Per-element predicate: when smem rows > NPREDICTED, mask each
    // iteration independently against the valid row count.
    auto cB = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_N>{}, Int<DSTATE>{}));
    auto tScB = thr_s2g.partition_D(cB);
    auto pred = make_tensor<bool>(shape(tScB));
    CUTE_UNROLL
    for (int i = 0; i < size(pred); ++i) {
      pred(i) = get<0>(tScB(i)) < NPREDICTED;
    }
    copy_if(s2g, pred, tSsB, tSgB);
  }
}

// =============================================================================
// Kernel
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1>
__global__ void ssu_incremental_kernel(SsuIncrementalParams params) {
  // Per-head DIM is sharded across `D_SPLIT` CTAs (D_PER_CTA each).
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32,
                "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 warp layout). "
                "D_SPLIT=4 (D_PER_CTA=16) needs warp-count restructure.");
  static_assert(NPREDICTED <= MAX_WINDOW,
                "NPREDICTED must be <= MAX_WINDOW (new tokens must fit in cache)");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG,
                "MAX_WINDOW must be <= MMA::K_BIG=16 (single replay K-tile assumption)");
  // Cross-check: host launcher must dispatch the template specialization
  // matching the runtime params.d_split it stamped into the struct.
  assert(params.d_split == D_SPLIT);
  using SmemT = SsuIncrementalStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // Grid layout (D_SPLIT, batch, nheads).
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

  // ── prev_num_accepted_tokens ──
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  // ── Per-CTA implicit checkpoint criterion ──
  // When the new tokens would overflow the cache buffer, we must checkpoint:
  // replay [0, prev_k) into state, write state to HBM, write the new tokens
  // to the **staging** buffer (1 - buf_read) at offset 0.  Otherwise, we
  // append the new tokens to the **active** buffer (buf_read) at offset
  // prev_k and skip the state HBM write entirely.  Cache writes always
  // happen — only their target buffer + offset depends on must_checkpoint.
  bool const must_checkpoint = (prev_k + NPREDICTED > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  // ── Load A (scalar, tie_hdim), dt_bias, and D (hoisted to hide gmem latency) ──
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ════════════════════════════════════════════════════════════════════════
  // Phase 0: Load all data into smem (per-warp ownership)
  // ════════════════════════════════════════════════════════════════════════
  // load_data ends with __pipeline_wait_prior(0) + __syncwarp() per warp.
  // No cross-warp sync here — every smem read before the post-replay
  // __syncthreads is served by data the *same warp* loaded.
  load_data<input_t, dt_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, d_tile, batch_idx, head, group_idx, cache_slot, buf_read, A_val,
      dt_bias_val);

  // No post-load_data __syncthreads.  Cooperative `load_state_cta` issues
  // cp.async from every thread; each thread's own `__pipeline_wait_prior(0)`
  // at the end of `load_data` retires its own group, so each warp sees its
  // own writes after the per-warp __syncwarp.  Cross-warp visibility is
  // established by the post-replay __syncthreads below — replay reads of
  // state are now safe because (a) replay's frag_h initial load sees only
  // the current warp's lane positions, and (b) the actual `_1×4` cross-warp
  // dependency is on writes that haven't happened yet at this point.

  // old_B writeback hoisted ahead of Phase 1.  Source (smem.B) is consumed
  // only by Phase 1a CB; the STGs fire-and-forget onto the memory subsystem
  // and complete in parallel with all subsequent compute.  Only W0, W1 hold
  // valid smem.B at this point (they're the ones that cp.async'd B).  Gate
  // accordingly — store halves its thread count but B is small (4 KB) so
  // still cheap.  old_B is D-independent (per-group, full DSTATE) — only
  // d_tile == 0 writes; other d_tiles would emit identical payloads.
  if (d_tile == 0 && warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(
        smem, params, warp, lane, head, group_idx, cache_slot, buf_write, write_offset);
  }

  // Warps 0,1 compute CB_scaled into smem (split N=16 → 2×[16,8]).  No
  // barrier needed — warps 2,3 start replay immediately; warps 0,1 join
  // after CB.
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);
  }
  // Phase 1b: MMA replay (all 4 warps, independent M-rows).  Each warp
  // reads its own DIM slice of state + old_x (loaded into smem by this
  // same warp in load_data), plus smem.old_B (redundantly loaded by all
  // warps — each warp sees its own copy).
  // ── Philox seed/offsets for f16 state stochastic rounding (no-op for
  // PHILOX_ROUNDS == 0 — compiles out via if constexpr inside replay).
  // `state_ptr_offset` matches Triton's `base_rand = cache_batch *
  // stride_state_batch + pid_h * (DIM*DSTATE)`; per-row m-offset and per-col
  // offset are added inside replay.  `state_w_base` is the gmem pointer to
  // this CTA's state slice (params.state already advanced by all the
  // CTA-invariant offsets) — replay's STG.32 path uses an i32 element
  // offset against this base.  Combining base + offset into one i64 ptr
  // saves 2 cross-iter regs vs holding params.state and state_gmem_off
  // separately.
  // ── DO NOT HOIST `rand_seed` ──
  // Hoisting the LDG into the preamble next to A/D/dt_bias looks intuitive
  // (one cache-coherent uniform load, latency hidden behind Phase 0) but
  // empirically pessimizes the kernel.  Measured at batch=64, mtp ∈
  // {4..16}, pk ∈ {0, mtp/2, mtp}, f16-philox-5 cache:
  //   - hoisted: spread across pk values flattens to 0.1–0.2 us
  //   - in-place (here): spread is 0.5–1.0 us at mtp ≥ 10
  //   - hoisted mean is ~0.4 us *worse* per config — it pulls every pk up
  //     to the worst case (pk = mtp) instead of letting low/mid pk benefit
  //     from the slack.
  // Hypothesis: this kernel is register-constrained at higher mtp; widening
  // `rand_seed`'s live range across Phase 0 + CB precompute + replay setup
  // eats the register headroom that the prev_k < mtp paths use to run
  // faster.  Net: tighter tail but higher mean, so we keep the read here.
  // See chat-log analysis comparing the variants.
  int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  uint32_t const state_ptr_offset =
      static_cast<uint32_t>(cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE);
  state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) +
                                cache_slot * params.state_stride_batch +
                                (int64_t)head * DIM * DSTATE + (int64_t)d_tile * D_PER_CTA * DSTATE;
  // Dispatch on state_t: int8 takes the quantized M-shard path (per-row
  // dequant on load, fp32 frag write to smem.new_state, warp-local amax
  // via shfl_xor, encode + STG.128 from registers).  fp16 / bf16 / fp32
  // take the existing native path.
  //
  // The int8 path requires D_PER_CTA == 64 (per-warp M = 16 = m16n8 atom M);
  // the wrapper enforces d_split == 1 for int8.  We additionally gate the
  // call site with `if constexpr (D_PER_CTA == 64)` so the kernel template
  // can still instantiate at D_PER_CTA == 32 (D_SPLIT=2) without trying to
  // instantiate the int8 helper at an unsupported tile size.
  if constexpr (sizeof(state_t) == 1) {
    if constexpr (D_PER_CTA == 64) {
      replay_state_mma_int8<input_t, state_t, DIM, D_PER_CTA, DSTATE>(
          smem, params, warp, lane, prev_k, d_tile, cache_slot, head, must_checkpoint);
    }
    // (D_PER_CTA != 64 with int8) is unreachable at runtime per the
    // wrapper's d_split == 1 assert.
  } else {
    replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS>(
        smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
        must_checkpoint);
  }

  __syncthreads();

  // ════════════════════════════════════════════════════════════════════════
  // Phase 2: Output — y[t,d] = init_out + cb_out + D*x, then z-gate
  // ════════════════════════════════════════════════════════════════════════
  // D_val already loaded in preamble (gmem latency hidden behind Phase 0+1).
  //
  // Fused: matmul 3 + state-writeback + matmul 4 + D*x + z-gate → direct gmem store
  compute_and_store_output<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                           PHILOX_ROUNDS>(smem, params, warp, lane, d_tile, batch_idx, head,
                                          cache_slot, D_val, must_checkpoint);

  // ── Phase 3: Store to global memory ──
  // (old_B hoisted to pre-Phase-1; state hoisted into compute_and_store_output.)

  // Cache writes — old_x uses all warps (vectorized), dt/cumAdt one warp each.
  // Each writes the new NPREDICTED tokens at gmem offset `write_offset` into
  // buffer `buf_write` (computed above from must_checkpoint).
  store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                   cache_slot, write_offset);
  // dt_proc / cumAdt are D-independent — only d_tile == 0 writes.
  if (d_tile == 0 && warp == 0 && lane < NPREDICTED) {
    auto* __restrict__ old_dt_proc_w = reinterpret_cast<float*>(params.old_dt_proc);
    int64_t const dt_w_base = cache_slot * params.old_dt_proc_stride_cache +
                              buf_write * params.old_dt_proc_stride_dbuf +
                              head * params.old_dt_proc_stride_head;
    old_dt_proc_w[dt_w_base + write_offset + lane] = smem.dt_proc[lane];
  }
  if (d_tile == 0 && warp == 1 && lane < NPREDICTED) {
    auto* __restrict__ old_cumAdt_w = reinterpret_cast<float*>(params.old_cumAdt);
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_cache +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + write_offset + lane] = smem.cumAdt[lane];
  }
}

// =============================================================================
// Kernel — int8 chain rewrite (separate kernel from the generic path)
// =============================================================================
// The int8 path uses a fundamentally different output computation:
//   1. M-shard replay (Layout<_4, _1>) — same as v15.4.
//   2. Chained matmul-3: replay's fp32 C-frag → bf16 A-frag in registers via
//      `convert_layout_acc_Aregs_sm80` (no smem.new_state staging).
//   3. Transposed matmul-4: x as A (M=D), CB^T as B → output^T(D, T) in regs.
//   4. Smem-transpose + cooperative STG.128 to (T, D) gmem.
// To keep the generic kernel uncluttered (no `if constexpr (sizeof(state_t) == 1)`
// branches), the int8 kernel is a standalone function that calls the new
// helpers (`replay_state_mma_int8_chain`, `compute_output_int8`) and uses
// `SsuIncrementalStorageInt8` for smem.  Phase 0/1 helpers (`load_data`,
// `store_old_B`, `compute_CB_scaled_2warp`) are reused verbatim — they only
// touch shared smem fields that both storage structs expose by name.
//
// PHILOX_ROUNDS template kept for ABI parity but currently must be 0 (int8+SR
// is the next workstream — see `.plans/int8_full_chain_rewrite.md`).
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS>
__global__ void ssu_incremental_kernel_int8(SsuIncrementalParams params) {
  using namespace cute;
  static_assert(sizeof(state_t) == 1, "ssu_incremental_kernel_int8 requires 1-byte state_t (int8)");
  static_assert(NPREDICTED <= MAX_WINDOW);
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG);
  // int8 path uses M-shard layout (Layout<_4,_1>): per-warp M = 16 = m16n8
  // atom M.  D_PER_CTA must equal DIM (D_SPLIT=1) to give 4×16=64 D-rows/CTA.
  // The wrapper enforces d_split == 1 for int8.
  constexpr int D_PER_CTA = DIM;
  static_assert(D_PER_CTA == 64, "int8 chain kernel requires DIM == 64");
  assert(params.d_split == 1);

  using SmemT = SsuIncrementalStorageInt8<input_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // Grid: (1, batch, nheads).  D-tile is always 0 for int8 (D_SPLIT=1).
  int const d_tile = blockIdx.x;
  int const batch_idx = blockIdx.y;
  int const head = blockIdx.z;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const tid = warp * warpSize + lane;
  int const group_idx = head / HEADS_PER_GROUP;

  // ── Resolve cache slot ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[batch_idx]) : batch_idx;
  if (cache_slot == params.pad_slot_id) return;

  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);

  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  bool const must_checkpoint = (prev_k + NPREDICTED > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  // ── Load scalars (A, dt_bias, D) ──
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ── Phase 0: load_data (shared with generic path) ──
  load_data<input_t, dt_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, d_tile, batch_idx, head, group_idx, cache_slot, buf_read, A_val,
      dt_bias_val);

  // ── store_old_B hoist (warps 0,1 only, d_tile == 0) ──
  if (d_tile == 0 && warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(
        smem, params, warp, lane, head, group_idx, cache_slot, buf_write, write_offset);
  }

  // ── compute_CB_scaled (warps 0,1) ──
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);
  }

  // ── Allocate per-warp frag_y_DxT (chain mma C-frag, fp32) ──
  // Layout ((2, 2), MMA_M=1, MMA_N=NPREDICTED_PAD_MMA_M/8) per thread.
  // Caller must zero before chain matmul-3 accumulates.
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);
  auto id_DxT =
      make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<SmemT::NPREDICTED_PAD_MMA_M>{}));
  Tensor frag_y_DxT = thr_mma_chain.partition_fragment_C(id_DxT);
  cute::clear(frag_y_DxT);

  // ── Phase 1b: replay + amax + chain matmul-3 → frag_y_DxT (init_out^T).
  // `encode_scale_per_row[]` is computed at the end of PASS 1 (from the warp-
  // reduced amax) and consumed by `encode_state_replay_int8` further below —
  // *after* `compute_output_int8` consumes `frag_y_DxT` and STGs the output.
  // PASS 2 deferred: lets PASS 2's replay-again HMMA + fp32→int8 encode + STG
  // run with `frag_y_DxT`'s 8 fp32 regs already dead, and lets its STGs fire
  // alongside the kernel's tail-end gmem traffic (store_old_x, dt/cumAdt).
  float encode_scale_per_row[2];
  float total_scale[2];
  replay_state_mma_int8_chain<input_t, state_t, DIM, D_PER_CTA, DSTATE>(
      smem, params, warp, lane, prev_k, d_tile, cache_slot, head, must_checkpoint, frag_y_DxT,
      encode_scale_per_row, total_scale);

  // ── Phase 2: transposed matmul-4 + epilogue + smem-transpose STG ──
  compute_output_int8<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, warp, lane, d_tile, batch_idx, head, cache_slot, D_val, frag_y_DxT);

  // ── PASS 2 (replay-again): re-run replay HMMA, encode fp32 → int8, STG ──
  // frag_y_DxT is dead now (already STG'd by compute_output_int8).
  if (must_checkpoint) {
    encode_state_replay_int8<input_t, state_t, DIM, D_PER_CTA, DSTATE>(
        smem, params, warp, lane, prev_k, d_tile, cache_slot, head, encode_scale_per_row,
        total_scale);
  }

  // ── Phase 3: cache writes (old_x, dt_proc, cumAdt) ──
  store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                   cache_slot, write_offset);
  if (d_tile == 0 && warp == 0 && lane < NPREDICTED) {
    auto* __restrict__ old_dt_proc_w = reinterpret_cast<float*>(params.old_dt_proc);
    int64_t const dt_w_base = cache_slot * params.old_dt_proc_stride_cache +
                              buf_write * params.old_dt_proc_stride_dbuf +
                              head * params.old_dt_proc_stride_head;
    old_dt_proc_w[dt_w_base + write_offset + lane] = smem.dt_proc[lane];
  }
  if (d_tile == 0 && warp == 1 && lane < NPREDICTED) {
    auto* __restrict__ old_cumAdt_w = reinterpret_cast<float*>(params.old_cumAdt);
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_cache +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + write_offset + lane] = smem.cumAdt[lane];
  }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` splits each head's DIM axis across `D_SPLIT` CTAs.
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
  // All cp.async-loaded operands need 16B alignment; output is also vectorized
  // (Pair<input_t> stores partitioned by m16n8k16 partition_C — base must be at
  // least 16B-aligned for the stride math to keep per-thread stores aligned).
  FLASHINFER_CHECK_ALIGNMENT(params.B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.C, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.x, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.state, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.old_x, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.old_B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.output, 16);
  if (params.z != nullptr) {
    FLASHINFER_CHECK_ALIGNMENT(params.z, 16);
  }

  // Per-CTA D = DIM / D_SPLIT.  Smem footprint shrinks for D-owned
  // buffers (state, x, z, old_x); non-D buffers (B, C, old_B, scalars) unchanged.
  constexpr int D_PER_CTA = DIM / D_SPLIT;

  auto dispatch_hpg = [&]<int HEADS_PER_GROUP>() {
    if constexpr (sizeof(state_t) == 1) {
      // int8 chain rewrite — uses ssu_incremental_kernel_int8 +
      // SsuIncrementalStorageInt8.  Only D_SPLIT == 1 is valid (the wrapper
      // asserts this); D_SPLIT == 2 still gets template-instantiated by the
      // public dispatcher's switch but is unreachable at runtime — gate the
      // body with `if constexpr (D_SPLIT == 1)` so that path doesn't launch.
      if constexpr (D_SPLIT == 1) {
        auto func =
            ssu_incremental_kernel_int8<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                        state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                        HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS>;
        constexpr size_t smem_size =
            sizeof(SsuIncrementalStorageInt8<input_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);

        dim3 grid(D_SPLIT, params.batch, params.nheads);
        dim3 block(warpSize, NUM_WARPS);

        if constexpr (smem_size > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        func<<<grid, block, smem_size, stream>>>(params);
      }
    } else {
      // Generic kernel: bf16 / fp16 / fp32 state, supports D_SPLIT ∈ {1, 2}.
      auto func = ssu_incremental_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                         state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                         HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, D_SPLIT>;

      constexpr size_t smem_size = sizeof(
          SsuIncrementalStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);

      // Grid is (D_SPLIT, batch, nheads).  D-tile is the fastest axis so the
      // `D_SPLIT` CTAs of the same head land on adjacent SMs and share L2
      // lines for the redundantly-loaded inputs (C, B, dt, ...).
      dim3 grid(D_SPLIT, params.batch, params.nheads);
      dim3 block(warpSize, NUM_WARPS);

      if constexpr (smem_size > 0) {
        FLASHINFER_CUDA_CHECK(
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }
      func<<<grid, block, smem_size, stream>>>(params);
    }
  };

  dispatchRatio(params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{},
                [&]<int HPG>() { dispatch_hpg.template operator()<HPG>(); });
}

// Public dispatcher: routes on `params.d_split`.  Allowed values: {1, 2}.
// D_SPLIT=4 (D_PER_CTA=16) requires warp-count restructure for the output
// MMA's `_1×4` layout.
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
                       ".  Allowed values: {1, 2}.  d_split=4 needs "
                       "warp-count restructure.");
  }
}

}  // namespace flashinfer::mamba::incremental

#endif  // FLASHINFER_MAMBA_KERNEL_SSU_INCREMENTAL_CUH_
