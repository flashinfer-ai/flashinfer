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
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_COMMON_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_COMMON_CUH_

// Shared infrastructure for the incremental SSU kernel: utilities, loaders,
// stores, MMA helpers, and functions used by both the 2/4-byte (bf16/fp16/fp32)
// and 8-bit (int8, future e4m3) kernel paths.

#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "checkpointing_ssu.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "cute/tensor.hpp"
#include "ssu_mtp_common.cuh"

namespace flashinfer::mamba::checkpointing {

using namespace conversion;

// ldmatrix.b8 (SM100_U8x16_LDSM_T) was tried as a replacement for per-lane
// LDS.16 int8 state loads.  It's 5-18% slower (bench v16.7b vs v16.8) because:
//   (1) inherent 2-way bank conflicts (16 threads × 16B vs 128B banks),
//   (2) state is the accumulator (C-frag), not an A/B operand — layout
//       remapping costs 8 shuffles + byte extractions,
//   (3) dynamic byte selection via SHF adds 15%+ short_scoreboard stalls.
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

// Pair<__nv_fp8_e4m3>: same single-u16 backing as `Pair<int8_t>` — fp8 e4m3
// is also a 1-byte storage type, so the load → unpack → cast pipeline runs
// through one 16-bit register.
template <>
struct Pair<__nv_fp8_e4m3> {
  uint16_t raw;
  template <int I>
  __device__ __forceinline__ __nv_fp8_e4m3 operator[](cute::Int<I>) const {
    static_assert(I == 0 || I == 1, "Pair index must be 0 or 1");
    __nv_fp8_storage_t const byte = (I == 0) ? static_cast<__nv_fp8_storage_t>(raw & 0xFFu)
                                             : static_cast<__nv_fp8_storage_t>(raw >> 8);
    return reinterpret_cast<__nv_fp8_e4m3 const&>(byte);
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
  // L2-only (.cg) variant for read-once streams: the state cache is touched
  // exactly once per step, so pulling it through L1 evicts the reused tiles.
  using AtomCG = cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
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
// at least 16 bytes of contiguous element data (one 128-bit vector), and keeps
// B = S = 3 so that each 8-row block XORs row↔column bits to stay
// bank-conflict-free on the 128-byte bank cycle.
//
//   sizeof(T)  Swizzle<B,M,S>     atom rows × cols    row bytes
//     2B       Swizzle<3, 3, 3>       8  × 64             128
//     4B       Swizzle<3, 3, 3>       8  × 64             256
//     1B       Swizzle<3, 4, 3>       8  × 128            128
//
// M is floored at 3 (8 contiguous elements).  For 4-byte elements the natural
// M=2 (16B chunks) is NOT conflict-free for the fragment loads the f32 state
// path emits: an LDS.64 float2 wavefront phase is 16 lanes = 4 fragment rows,
// and the M=2 XOR moves each row's pair-set within an XOR-closed 16B-chunk set
// — two rows per phase land on the same 16 banks (2-way, every access).  M=3
// rotates 32B chunks, mapping the 4 rows of a phase onto 4 disjoint 32B bank
// ranges (verified in tiler.py's fragment-map bank simulator).  16B cp.async /
// LDS vectors stay intact under M=3 (32B ⊃ 16B).
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
  static constexpr int SWIZZLE_BASE = log2_pow2(ELEMS_PER_ATOM) < 3 ? 3 : log2_pow2(ELEMS_PER_ATOM);
  using type = cute::Swizzle<3, SWIZZLE_BASE, 3>;
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
// `valid_rows_rt` is a runtime bound that overrides the compile-time
// `VALID_ROWS` template parameter — used by the varlen (v20) path to tighten
// the predicate from `< NPREDICTED` to `< seq_len`.  When the caller omits it,
// the default is the constexpr `VALID_ROWS` so non-varlen call sites fold to
// the same SASS as before.
// `ZFILL` selects the cp.async variant: true (default) writes ZEROS wherever
// the predicate is false (rows ≥ valid, padded cols) — the legacy contract;
// false SKIPS false-predicate elements entirely (plain cp.async), which the
// ring gather's second segment needs so it can't clobber the first segment's
// rows.  `first_valid_row` tightens the predicate from below (default 0).
template <typename SmemShape, int VALID_ROWS, bool ZFILL = true, typename input_t>
__device__ __forceinline__ void load_tile_async(input_t* __restrict__ smem_dst,
                                                input_t const* __restrict__ gmem_src,
                                                int gmem_row_stride, int lane,
                                                int valid_rows_rt = VALID_ROWS,
                                                int first_valid_row = 0) {
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
  using CopyAtomT =
      std::conditional_t<ZFILL, typename Copy_prop::AtomZFill, typename Copy_prop::Atom>;
  auto g2s = make_tiled_copy(Copy_Atom<CopyAtomT, input_t>{}, ThrLayout{},
                             Layout<Shape<_1, Int<VAL_COLS_PER_THREAD>>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<ROWS_PAD>{}, Int<SMEM_COLS>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = (get<0>(thr_id(i)) >= first_valid_row) && (get<0>(thr_id(i)) < valid_rows_rt) &&
              (get<1>(thr_id(i)) < VALID_COLS);
  }
  copy_if(g2s, pred, thr.partition_S(g_full), thr.partition_D(s_full));
}

// Ring gather: smem rows j ∈ [0, count_rt) come from gmem ring row
// (start + j) mod ring_len (row stride `gmem_row_stride`); rows
// [count_rt, ROWS_PAD) end up zero (the MMA K-pad contract).  Hand-rolled
// version of load_tile_async's wide tiled copy (same 4×8 thread layout, same
// swizzled dst addresses) with two differences:
//   - the per-row source is the RING row (one cond-subtract per row — the
//     tiled copy can't express the mod in an affine gmem layout), and
//   - each 16B element is issued as ONE cp.async whose src-size operand is 0
//     when predicated off (the ZFILL mechanism), so pad rows zero-fill through
//     the async path instead of synchronous smem stores.  (A two-segment
//     copy_if + STS tail was tried first: the tail stores dominated no-write
//     units — 14× smem-store wavefronts, +22% main at pnat=4, ncu 2026-07-10.)
// Every element is written by exactly one cp.async — no same-address ordering
// hazard.  count_rt may be ≤ 0 (callers pass prev_k−8 residues): all rows
// predicate off and the whole tile zero-fills.
template <typename SmemShape, int COUNT, typename input_t>
__device__ __forceinline__ void load_ring_tile_async(input_t* __restrict__ smem_dst,
                                                     input_t const* __restrict__ gmem_base,
                                                     int gmem_row_stride, int lane, int ring_start,
                                                     int ring_len, int count_rt = COUNT) {
  using namespace cute;
  constexpr int ROWS_PAD = size<0>(SmemShape{});
  constexpr int VALID_COLS = size<1>(SmemShape{});
  constexpr int SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(VALID_COLS);
  constexpr int VAL = Copy_prop::vec_bytes / sizeof(input_t);  // elems per 16B cp.async
  static_assert(Copy_prop::vec_bytes == 16, "the inline cp.async hardcodes 16B atoms");
  constexpr int THR_ROWS = 4, THR_COLS = 8;  // wide layout (see load_tile_async)
  static_assert(SMEM_COLS % (THR_COLS * VAL) == 0, "col passes must tile SMEM_COLS");
  static_assert(ROWS_PAD % THR_ROWS == 0, "row passes must tile ROWS_PAD");
  auto s_full =
      make_tensor(make_smem_ptr(smem_dst), make_swizzled_layout_rc<input_t, ROWS_PAD, SMEM_COLS>());
  int const tr = lane >> 3, tc = lane & 7;
  CUTE_UNROLL
  for (int rp = 0; rp < ROWS_PAD / THR_ROWS; ++rp) {
    int const r = rp * THR_ROWS + tr;
    int rr = ring_start + r;
    if (rr >= ring_len) rr -= ring_len;
    auto const* src_row = gmem_base + (int64_t)rr * gmem_row_stride;
    CUTE_UNROLL
    for (int cp = 0; cp < SMEM_COLS / (THR_COLS * VAL); ++cp) {
      int const c = (cp * THR_COLS + tc) * VAL;
      bool const valid = (r < count_rt) && (c < VALID_COLS);
      uint32_t const dst = cast_smem_ptr_to_uint(&s_full(r, c));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src_row + c),
                   "r"(valid ? 16 : 0));
    }
  }
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
                                                    int64_t state_base, int warp, int lane,
                                                    int state_buf = 0) {
  using namespace cute;
  static_assert(D_PER_CTA % NUM_WARPS == 0, "D_PER_CTA must be divisible by NUM_WARPS");
  constexpr int DIM_PER_WARP = D_PER_CTA / NUM_WARPS;
  // Each warp loads its DIM_PER_WARP-row slice with the (4,8) thread tile (4 rows/pass), so
  // DIM_PER_WARP must be a multiple of 4.  NUM_WARPS=4 → 16 rows/warp (4 passes); 8 → 8 (2).
  static_assert(DIM_PER_WARP % 4 == 0,
                "DIM_PER_WARP (D_PER_CTA/NUM_WARPS) must be a multiple of 4 for the (4,8) load");

  // Single-local_tile path — swizzle layout sized to this CTA's D_PER_CTA slice; one local_tile
  // splits it directly per-warp.  state_buf selects the double-buffered state slot (cross-head
  // prefetch); state_buf=0 ⇒ the single original buffer.
  Tensor sState_full = make_tensor(
      make_smem_ptr(reinterpret_cast<state_t*>(smem.state) + state_buf * D_PER_CTA * DSTATE),
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
      make_tiled_copy(Copy_Atom<Copy_prop::AtomCG, state_t>{},
                      Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = g2s.get_slice(lane);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
}

template <typename state_t, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_cta(SmemT& smem, state_t const* __restrict__ state_ptr,
                                               int64_t state_base, int tid, int state_buf = 0) {
  using namespace cute;
  static_assert(NUM_WARPS == 4, "Expected 4 warps");

  Tensor sState = make_tensor(
      make_smem_ptr(reinterpret_cast<state_t*>(smem.state) + state_buf * D_PER_CTA * DSTATE),
      make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>());
  Tensor gState = make_tensor(make_gmem_ptr(state_ptr + state_base),
                              make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                          make_stride(Int<DSTATE>{}, Int<1>{})));

  constexpr int VAL_COLS = Copy_prop::vec_bytes / sizeof(state_t);
  using ThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  constexpr int THR_ROWS = decltype(size<0>(ThrLayout{}))::value;
  static_assert(D_PER_CTA % THR_ROWS == 0,
                "D_PER_CTA must be divisible by the thread layout's row count");
  auto g2s = make_tiled_copy(Copy_Atom<Copy_prop::AtomCG, state_t>{}, ThrLayout{},
                             Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = g2s.get_slice(tid);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
}

// =============================================================================
// Phase 0: cooperative data load into smem (all warps).
// Compute cumAdt[T] = cumsum(A * dt_proc) → smem.
// Warp-level inclusive prefix sum using Hillis-Steele shuffles.
// Only the first NPREDICTED lanes participate; the rest are idle.
//
// If the storage struct exposes a `decay` field, this also writes
// `decay[lane] = exp(val)` — fuses the EX2 with the cumsum write so the output
// decay broadcast in `compute_output_8bit` becomes a plain LDS (no per-element
// __expf).  Detected via SFINAE so the 2/4-byte storage (no decay field) is
// unaffected.
namespace detail {
template <typename T, typename = void>
struct has_decay : std::false_type {};
template <typename T>
struct has_decay<T, std::void_t<decltype(std::declval<T&>().decay[0])>> : std::true_type {};
}  // namespace detail

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
    if constexpr (detail::has_decay<SmemT>::value) {
      smem.decay[lane] = __expf(val);
    }
  }
}

// Load this head's old dt from the RING (row (start+t) % L) and recompute its
// decay into the caller's smem slices (dt_dst / ca_dst).  The ring caches only
// dt: cumAdt (prefix sums) is not ring-shift-invariant — a flush would
// invalidate every cached entry — so it is recomputed here as
// A · inclusive_scan(dt) over the window (≤ 16 lanes, 4 shfl steps).  All 32
// lanes join the scan (lanes ≥ count contribute masked zeros) so the shuffles
// stay convergent; only lanes < count write.  Used by the MONOLITH — the
// two-kernel main runs its own in-register recompute (warp_scan_old_cumAdt).
__device__ __forceinline__ void load_old_dt_cumAdt(CheckpointingSsuParams const& params, int lane,
                                                   int64_t cache_slot, int ring_start, int head,
                                                   int count, float A_val,
                                                   float* __restrict__ dt_dst,
                                                   float* __restrict__ ca_dst) {
  auto const* __restrict__ dt_cache_ptr = reinterpret_cast<float const*>(params.dt_cache);
  // head·head_stride is an index into a contiguous inner dim → 32-bit;
  // slot-distance stride (*_stride_seq) stays 64-bit.
  int64_t const dt_base =
      cache_slot * params.dt_cache_stride_seq + (int64_t)(head * (int)params.dt_cache_stride_head);
  int ring_row = ring_start + lane;
  if (ring_row >= params.ring_buffer_len) ring_row -= params.ring_buffer_len;
  float dt_v = (lane < count) ? dt_cache_ptr[dt_base + ring_row] : 0.f;
  // Inclusive scan over the first 16 lanes (count ≤ MAX_WINDOW ≤ 16).
  float scan = dt_v;
#pragma unroll
  for (int off = 1; off < 16; off <<= 1) {
    float const up = __shfl_up_sync(0xffffffffu, scan, off);
    if ((lane & 15) >= off) scan += up;
  }
  if (lane < count) {
    dt_dst[lane] = dt_v;
    ca_dst[lane] = A_val * scan;
  }
}

// Load phase.  Split into two halves around the PDL barrier (`gdc_wait`):
//
//   load_pre_pdl_wait_data:  data NOT produced by the immediate upstream
//     kernel (conv1d) — state and old_* are cache from the previous SSU
//     step; dt and z are in_proj outputs (in_proj fully completed before
//     conv1d began, so they are visible by the time we hit `gdc_wait`).
//     Issues cp.async for cache + z, runs the scalar LDGs (old_dt,
//     old_cumAdt, dt→dt_proc) and the cumAdt warp scan.  No commit/wait —
//     the cp.async stays in flight while we `gdc_wait` on conv1d.
//
//   load_post_pdl_wait_data: x/B/C cp.async (conv1d outputs — must wait)
//     and the single `__pipeline_commit + __pipeline_wait_prior(0) +
//     __syncwarp` that drains both halves.  Cache cp.async issued in the
//     pre-wait half share the per-thread async group with these, so one
//     wait_prior(0) covers them all.
//
// Per-warp data ownership (unchanged from the pre-split version):
//   state:  per-warp contiguous DIM slice (warp W owns rows [16W : 16W+16]).
//   B, C:   redundant on W0, W1 (both compute 2-warp CB, both need full).
//   old_B:  redundant on all 4 warps (each warp's replay reads full DSTATE).
//   old_x:  redundant on all 4 warps (small, ~2 KB — partitioning not worth
//           the complication).
//   x:      W2 only (Phase-2 read, covered by final __syncthreads).
//   z:      W3 only (Phase-2 read, covered by final __syncthreads).
//   scalars (old_dt, old_cumAdt, dt→dt_proc) + cumAdt cumsum:
//           redundant on each warp's first NPREDICTED/MAX_WINDOW lanes.  Writes
//           are idempotent across warps (identical payloads to same slots).
// =============================================================================
// Per-sequence gmem base offsets (`x_seq_base`, etc.) are computed once in
// the kernel prologue — they encode the "start of this sequence" along the
// outer axis (batch in non-varlen, packed-token in varlen).  Helper indexing
// is then uniform `seq_base + inner`.
//
// `seq_len` is the per-sequence new-token count (== NPREDICTED constexpr in
// non-varlen, runtime int in varlen).  Used as the cp.async row predicate
// and the dt/scalar lane predicate so trailing rows past `seq_len` ZFILL to
// zero in smem.
template <typename input_t, typename dt_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_pre_pdl_wait_data(
    SmemT& smem, CheckpointingSsuParams const& params, int lane, int warp, int d_tile, int head,
    int group_idx, int64_t cache_slot, int ring_start, float A_val, float dt_bias_val,
    int64_t dt_seq_base, int64_t z_seq_base, int seq_len) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(D_PER_CTA % INPUT_PACK == 0, "D_PER_CTA must be divisible by input pack size");

  int const d_tile_off = d_tile * D_PER_CTA;

  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ old_x_ptr = reinterpret_cast<input_t const*>(params.x_cache);
  auto const* __restrict__ old_B_ptr = reinterpret_cast<input_t const*>(params.B_cache);
  auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);

  int64_t const ox_base = cache_slot * params.x_cache_stride_seq +
                          (int64_t)head * params.x_cache_stride_head + d_tile_off;
  int64_t const oB_base =
      cache_slot * params.B_cache_stride_seq + (int64_t)group_idx * params.B_cache_stride_group;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  // ZShape: shrunk to the swizzle atom's row extent (z is read via
  // partition_C alias, the physical buffer only needs to be one swizzle
  // row-atom tall).
  using ZShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<D_PER_CTA>>;
  // old_B / old_x's smem row count = replay matmul K-axis = MAX_WINDOW_PAD_MMA_K.
  using OldBShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  using OxShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

  // ── State: per-CTA D-slice ([D_PER_CTA, DSTATE]).  Dispatch on D_SPLIT
  // (= DIM == D_PER_CTA): per-warp coalesced load when one CTA owns the
  // full head's D, cooperative 128-thread load when D is sharded. ──
  {
    auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
    int64_t const state_base = cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE +
                               (int64_t)d_tile_off * DSTATE;
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

  // ── old_B: redundant on all 4 warps (each warp's replay consumes full
  // DSTATE).  Identical payloads to same smem dest — final bytes
  // deterministic.  VALID_ROWS = MAX_WINDOW (cache rows). ──
  load_ring_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, old_B_ptr + oB_base,
                                              (int)params.B_cache_stride_pos, lane, ring_start,
                                              params.ring_buffer_len);

  // ── old_x: redundant on all 4 warps (small, simpler than partitioning).
  // VALID_ROWS = MAX_WINDOW (cache rows). ──
  load_ring_tile_async<OxShape, MAX_WINDOW>(smem.old_x, old_x_ptr + ox_base,
                                            (int)params.x_cache_stride_pos, lane, ring_start,
                                            params.ring_buffer_len);

  // ── z: W3 only (Phase-2 read, final __syncthreads makes it visible).
  // Sourced from in_proj — not from conv1d — so safe to issue pre-wait. ──
  if (warp == 3 && z_ptr) {
    int64_t const z_base = z_seq_base + head * DIM + d_tile_off;
    load_tile_async<ZShape, NPREDICTED>(smem.z, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }

  // Commit the cache cp.async group BEFORE the caller's `gdc_wait()` so the
  // hardware actually issues the gmem→smem transfers while the wait is in
  // flight (without commit, the operations sit pending and only kick off
  // once the post-wait commit fires — no overlap).  Placed immediately after
  // the last cp.async (z); the synchronous LDGs + cumAdt scan below are not
  // part of any pipeline group and run in parallel with the in-flight
  // transfers.  The post half issues a second group; `__pipeline_wait_prior(0)`
  // there drains both.
  __pipeline_commit();

  // ── Scalar loads + cumAdt cumsum: redundant per warp.
  // old_dt / old_cumAdt: load up to MAX_WINDOW lanes (cache scalars).
  // dt_proc: load up to NPREDICTED lanes (new-token scalars from in_proj).
  // Synchronous LDG + plain smem stores — no cp.async.  Writes from 4
  // warps to the same slots are idempotent (same payloads). ──
  load_old_dt_cumAdt(params, lane, cache_slot, ring_start, head, MAX_WINDOW, A_val, smem.old_dt,
                     smem.old_cumAdt);
  // dt → softplus → smem.dt_proc.  Under varlen the active lane range is
  // `[0, seq_len)`; lanes `[seq_len, NPREDICTED)` are left uninitialized —
  // `compute_cumAdt` will scan over them and produce garbage in the
  // `cumAdt[seq_len:NPREDICTED]` tail, but every downstream consumer
  // (`compute_CB_scaled_2warp` mask, output STG, dt_proc/cumAdt tape writes)
  // is gated on `seq_len`, so the garbage never reaches gmem or contaminates
  // valid rows.
  //
  // Per-lane stride along the T-axis is `dt_stride_token` in both layouts
  // (4D batch and 1D packed varlen); the caller bakes `head` into
  // `dt_seq_base` so the inner indexing is `dt_seq_base + lane *
  // dt_stride_token`.
  if (lane < seq_len) {
    float dt_val = toFloat(dt_ptr[dt_seq_base + (int64_t)lane * params.dt_stride_token]);
    dt_val += dt_bias_val;
    if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
    smem.dt_proc[lane] = dt_val;
  }
  // cumAdt = cumsum(A * dt_proc) — warp-local Hillis-Steele shuffle.  Each
  // of the 4 warps runs the same reduction on identical inputs (dt_proc
  // just written above) and writes the same smem.cumAdt slots.
  compute_cumAdt<NPREDICTED>(smem, lane, A_val);
}

// Post-wait half.  Issues cp.async for conv1d outputs (x, B, C) and drains
// the per-thread async group (which includes both the cache cp.async issued
// in `load_pre_pdl_wait_data` and these conv1d cp.async).  Caller must have
// called `gdc_wait()` between the two halves; otherwise this reads stale
// conv1d data.
//
// Takes `outer` (the per-sequence outer index) rather than pre-multiplied
// `*_seq_base` scalars.  Computing `outer * stride_seq` inside this function
// keeps the multipliers transient instead of pinning them across the
// `gdc_wait()` asm-volatile barrier (which the compiler can't reorder around
// and thus can't rematerialize through).  Saves ~6 registers vs. pre-computed
// bases.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_post_pdl_wait_data(SmemT& smem,
                                                        CheckpointingSsuParams const& params,
                                                        int lane, int warp, int d_tile, int head,
                                                        int group_idx, int64_t outer, int seq_len) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(D_PER_CTA % INPUT_PACK == 0, "D_PER_CTA must be divisible by input pack size");

  int const d_tile_off = d_tile * D_PER_CTA;

  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);

  int64_t const B_base = outer * params.B_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const x_base = outer * params.x_stride_seq + head * DIM + d_tile_off;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  // CShape: first dim shrunk to the swizzle atom's row extent so cp.async
  // writes don't spill past the (also shrunk) C smem buffer.  When NPREDICTED
  // > ATOM_ROWS this falls back to NPREDICTED_PAD_MMA_M.
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  // B's smem row count = matmul-1 N-axis = NPREDICTED_PAD_MMA_N.
  using BShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;

  // ── B: redundant on W0, W1 (both do 2-warp CB compute) ──
  if (warp < 2) {
    load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_token, lane,
                                        seq_len);
  }
  // ── C: redundant on all 4 warps — chain matmul-3 reads smem.C from every
  // warp, so each warp must see its own cp.async without a cross-warp sync.
  // Identical payloads to same smem dest (same pattern as old_B / old_x). ──
  load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane, seq_len);

  // ── x: W2 only (Phase-2 read, final __syncthreads makes it visible) ──
  if (warp == 2) {
    load_tile_async<XShape, NPREDICTED>(smem.x, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
  }

  // Commit the conv1d cp.async group and drain BOTH groups: the cache
  // group committed in `load_pre_pdl_wait_data` (pre-`gdc_wait`) and this
  // conv1d group.  `__pipeline_wait_prior(0)` waits for ≤0 pending groups.
  // __syncwarp() provides acquire semantics across the 32 lanes of each
  // warp.  No cross-warp sync here; the only __syncthreads is after
  // CB + replay.
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
}

// Single-pass load — used when `params.enable_pdl == false`.  All cp.async
// (state, B, C, old_B, old_x, x, z) issue together into one async group;
// scalars + cumAdt scan run while the cp.async are in flight; one commit +
// wait_prior(0) + syncwarp drains the whole thing.  This restores the v21.0
// load order: the synchronous LDG-then-STS for old_dt/old_cumAdt/dt benefits
// from overlap with the conv1d cp.async (B/C/x) — which the split (pre +
// gdc_wait + post) form sacrifices since conv1d cp.async only issue after
// the wait.  When PDL is paired with an upstream conv1d, the split's
// cache-load-during-wait overlap dominates; when not paired, the split is
// pure overhead (gdc_wait is a no-op, but the cp.async are delayed).
template <typename input_t, typename dt_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_data(SmemT& smem, CheckpointingSsuParams const& params,
                                          int lane, int warp, int d_tile, int head, int group_idx,
                                          int64_t cache_slot, int ring_start, float A_val,
                                          float dt_bias_val, int64_t outer, int seq_len) {
  constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
  static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
  static_assert(D_PER_CTA % INPUT_PACK == 0, "D_PER_CTA must be divisible by input pack size");

  int const d_tile_off = d_tile * D_PER_CTA;

  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ old_x_ptr = reinterpret_cast<input_t const*>(params.x_cache);
  auto const* __restrict__ old_B_ptr = reinterpret_cast<input_t const*>(params.B_cache);
  auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);

  int64_t const B_base = outer * params.B_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  int64_t const x_base = outer * params.x_stride_seq + head * DIM + d_tile_off;
  int64_t const ox_base = cache_slot * params.x_cache_stride_seq +
                          (int64_t)head * params.x_cache_stride_head + d_tile_off;
  int64_t const oB_base =
      cache_slot * params.B_cache_stride_seq + (int64_t)group_idx * params.B_cache_stride_group;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using BShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
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

  if (warp < 2) {
    load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_token, lane,
                                        seq_len);
  }
  load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane, seq_len);

  load_ring_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, old_B_ptr + oB_base,
                                              (int)params.B_cache_stride_pos, lane, ring_start,
                                              params.ring_buffer_len);
  load_ring_tile_async<OxShape, MAX_WINDOW>(smem.old_x, old_x_ptr + ox_base,
                                            (int)params.x_cache_stride_pos, lane, ring_start,
                                            params.ring_buffer_len);

  if (warp == 2) {
    load_tile_async<XShape, NPREDICTED>(smem.x, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
  }
  if (warp == 3 && z_ptr) {
    int64_t const z_base = outer * params.z_stride_seq + head * DIM + d_tile_off;
    load_tile_async<ZShape, NPREDICTED>(smem.z, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }

  // ── Scalar loads (overlap with cp.async) + cumAdt cumsum ──
  load_old_dt_cumAdt(params, lane, cache_slot, ring_start, head, MAX_WINDOW, A_val, smem.old_dt,
                     smem.old_cumAdt);
  int64_t const dt_seq_base_local = outer * params.dt_stride_seq + head;
  if (lane < seq_len) {
    float dt_val = toFloat(dt_ptr[dt_seq_base_local + (int64_t)lane * params.dt_stride_token]);
    dt_val += dt_bias_val;
    if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
    smem.dt_proc[lane] = dt_val;
  }
  compute_cumAdt<NPREDICTED>(smem, lane, A_val);

  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncwarp();
}

// (compute_cumAdt moved above load_pre_pdl_wait_data so it can be called from there)

// Compute CB_scaled[T,T] = (C @ B^T) * decay * dt_proc * causal_mask.
// Split across 2 warps: warp 0 computes columns 0:8, warp 1 computes columns 8:16.
// Result stored to swizzled smem.CB_scaled (input_t, row stride 64, Swizzle<3,3,3>).
// Called between the two __syncthreads by warps 0 and 1 only.
// `seq_len` is the runtime row/col bound on the (T, T) CB_scaled tile.
// Caller passes `NPREDICTED` (constexpr) for non-varlen — the mask
// `j <= t && t < seq_len && j < seq_len` then folds to today's SASS.
// Varlen passes the per-sequence `seq_len ≤ NPREDICTED`; rows/cols past it
// get zeroed so downstream matmul-4 / chain matmul-3 see zeros there.
template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_scaled_2warp(SmemT& smem, int warp, int lane,
                                                        int seq_len) {
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
    if (j <= t && t < seq_len && j < seq_len) {
      val = frag_acc(i) * __expf(smem.cumAdt[t] - smem.cumAdt[j]) * smem.dt_proc[j];
    } else {
      val = 0.f;
    }
    smem_CB_part(i) = MMA_prop::operand_t(val);
  }
}

// Compute CB_old[t, i] = (C @ old_B^T)[t, i] * exp(cumAdt[t]) * dB_old(i)  for
// i ∈ [0, prev_k); 0 otherwise.
//   dB_old(i)   = exp(total_old_cumAdt − smem.old_cumAdt[i]) * smem.old_dt[i].
// The per-t factor exp(cumAdt[t]) is baked in here (vs at matmul-4 time) so the
// epilogue's β-scale = exp(total_old_cumAdt) * exp(cumAdt[t]) on init_out
// composes with a single CB_old @ old_x add — no extra elementwise pass.
// Identity (matches Triton's combined-sequence SSU):
//   y_old_contrib[t, d] = exp(cumAdt[t]) * Σ_i dB_old(i) * x_old[i, d] * (C[t] · B_old[i])
//                      = Σ_i CB_old[t, i] * old_x[i, d].
// Written into smem.CB_scaled at cols [NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M +
// MAX_WINDOW_PAD_MMA_K). Sibling of compute_CB_scaled_2warp — runs on warps 2, 3 in parallel with
// warps 0, 1 writing the new-token half at cols [0, NPREDICTED_PAD_MMA_M).  Uses the no-write
// path's CB_old region of the same swizzled buffer (32 cols total ≤ CB_ROW_STRIDE=64).
//
// 2-warp N-split: each warp owns one m16n8 N-atom (MMA::N=8 cols).
//   MAX_WINDOW_PAD_MMA_K == 16:  warp 2 → cols [0, 8); warp 3 → cols [8, 16).
//   MAX_WINDOW_PAD_MMA_K == 8 :  warp 2 covers all 8 cols; warp 3 returns early.
template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_old_2warp(SmemT& smem, int warp, int lane, int prev_k,
                                                     int seq_len) {
  using namespace cute;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  constexpr int N_HALF = MMA_prop::N;  // 8 — one m16n8 atom per warp
  constexpr int NUM_N_ATOMS = MAX_WINDOW_PAD_MMA_K / N_HALF;
  static_assert(MAX_WINDOW_PAD_MMA_K % N_HALF == 0,
                "compute_CB_old_2warp: MAX_WINDOW_PAD_MMA_K must be a multiple of MMA::N");
  static_assert(NPREDICTED_PAD_MMA_M + MAX_WINDOW_PAD_MMA_K <= CB_ROW_STRIDE,
                "CB_scaled buffer must fit both CB_new (cols [0,T_pad)) and CB_old "
                "(cols [T_pad, T_pad+K_old)) within its physical row stride");

  int const sub_warp = warp - 2;  // ∈ {0, 1}
  if (sub_warp >= NUM_N_ATOMS) return;

  float const total_old_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;

  // ── Swizzled smem views (A = C, B = old_B; same shapes as the replay path's
  //    C/old_B reads, so we get cache locality with no extra cp.async). ──
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_old_B = make_swizzled_layout_rc<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_old_B =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.old_B)), layout_old_B);

  // ── TiledMMA: 32 threads, single m16n8k16 atom (K-loops over DSTATE) ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  // ── K-tile A operand (C): full M, K-loop dim ──
  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));

  // ── K-tile B operand (old_B): warp picks its 8-col N-atom slice ──
  Tensor smem_old_B_half =
      local_tile(smem_old_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(sub_warp, _));

  // ── Register fragments ──
  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_old_B_half(_, _, _0{}));

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
  Tensor smem_old_B_s2r = s2r_thr_B.partition_S(smem_old_B_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  // ── GEMM: DSTATE / K_BIG = 8 K-tiles ──
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_old_B_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  // ── Identity coords for elementwise / store ──
  auto id_half = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  auto id_part = thr_mma.partition_C(id_half);

  // ── Store to swizzled smem.CB_scaled at the CB_old region (cols [T_pad, T_pad+K_old)).
  //    Use the full physical (NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE) padded swizzle view.
  //    Byte-compatible with compute_CB_scaled_2warp's (T_pad, T_pad, CB_ROW_STRIDE)
  //    padded view: both produce inner offset r*CB_ROW_STRIDE + c, same Swizzle. ──
  auto layout_cb_full = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t*>(smem.CB_scaled)), layout_cb_full);
  // (16, CB_ROW_STRIDE) tiled by (16, N_HALF) → (1, CB_ROW_STRIDE/N_HALF) tiles.
  // Coord (0, T_pad/N_HALF + sub_warp) lands inside the CB_old region.
  constexpr int CB_OLD_TILE_BASE = NPREDICTED_PAD_MMA_M / N_HALF;
  Tensor smem_CB_half = local_tile(smem_CB, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}),
                                   make_coord(_0{}, CB_OLD_TILE_BASE + sub_warp));
  Tensor smem_CB_part = thr_mma.partition_C(smem_CB_half);

#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    int t = get<0>(id_part(i));
    int j = sub_warp * N_HALF + get<1>(id_part(i));
    float val;
    if (j < prev_k && t < seq_len) {
      val = frag_acc(i) * __expf(smem.cumAdt[t] + total_old_cumAdt - smem.old_cumAdt[j]) *
            smem.old_dt[j];
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
template <int DB_COEFFS_PER_LANE>
__device__ __forceinline__ void precompute_dB_coeff(float coeff[DB_COEFFS_PER_LANE],
                                                    float const* old_cumAdt, float const* old_dt,
                                                    float total_cumAdt, int prev_k, int lane) {
  static_assert(DB_COEFFS_PER_LANE == 2 || DB_COEFFS_PER_LANE == 4,
                "DB_COEFFS_PER_LANE must be 2 (k8) or 4 (k16)");
  int const K_base = (lane % 4) * 2;
#pragma unroll
  for (int i = 0; i < DB_COEFFS_PER_LANE; ++i) {
    // m16n8k_ V-index → K-offset: (V & 1) is the col-pair offset; (V & 2) ? 8 : 0
    // covers the second K-tile inside the K_BIG (k16) atom.
    int const k = K_base + (i & 1) + ((i & 2) << 2);
    coeff[i] = (k < prev_k) ? __expf(total_cumAdt - old_cumAdt[k]) * old_dt[k] : 0.f;
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
template <int MAX_WINDOW_PAD_MMA_K, typename FragA>
__device__ __forceinline__ void apply_dA_coeff(FragA& frag_A, float const* old_cumAdt,
                                               float const* old_dt, float total_cumAdt, int prev_k,
                                               int lane) {
  using namespace cute;
  constexpr int FRAG_A_SIZE = size(FragA{});
  static_assert((MAX_WINDOW_PAD_MMA_K == 16 && FRAG_A_SIZE == 8) ||
                    (MAX_WINDOW_PAD_MMA_K == 8 && FRAG_A_SIZE == 4),
                "apply_dA_coeff: unsupported MMA K / frag_A size combination");
  using frag_t = typename FragA::value_type;

  int const K_base = (lane % 4) * 2;

  if constexpr (MAX_WINDOW_PAD_MMA_K == 8) {
    float const c0 =
        (K_base < prev_k) ? __expf(total_cumAdt - old_cumAdt[K_base]) * old_dt[K_base] : 0.f;
    float const c1 = (K_base + 1 < prev_k)
                         ? __expf(total_cumAdt - old_cumAdt[K_base + 1]) * old_dt[K_base + 1]
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
      c[j] = (k < prev_k) ? __expf(total_cumAdt - old_cumAdt[k]) * old_dt[k] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int const ci = (i & 1) | ((i & 4) >> 1);
      frag_A(i) = frag_t(toFloat(frag_A(i)) * c[ci]);
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

// MMA A operand: dtype-aware TiledCopy.  A = C in the monolith's matmul 3
// (always 2-byte) and A = state in the persistent main's operand-swapped OUT.1
// (2- or 4-byte f32 cache).
//   2-byte smem: LDSM (SM75_U32x4_LDSM_N) — vectorized 16-bit ldmatrix.
//   4-byte smem: UniversalCopy<uint64_t> — one LDS.64 per k-adjacent float2
//     pair (the fragment's innermost value mode).  ldmatrix cannot feed a
//     16-bit MMA from 32-bit elements: its fixed distribution hands each lane
//     ONE whole f32 (the tf32 fragment layout), not the k-adjacent pair the
//     bf16 fragment wants — repairing that costs cross-lane shfl chains that
//     exceed the LDS path.  Pairs are narrowed to bf16 in registers by
//     `convert_frag` after the load (mirrors the B side).
template <typename state_t, typename MmaT, typename TiledMma>
__device__ __forceinline__ auto make_a_s2r(TiledMma const& tm) {
  using namespace cute;
  if constexpr (sizeof(state_t) == 2) {
    return make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MmaT>{}, tm);
  } else {
    static_assert(sizeof(state_t) == 4, "wide state path expects 4-byte smem");
    return make_tiled_copy_A(Copy_Atom<UniversalCopy<uint64_t>, state_t>{}, tm);
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

// Load a lane's REGS-element fragA chunk from gmem (fragA-native cb_scaled /
// cb_old) straight into a matmul-4 A-operand fragment — one vectorized LDG, no
// smem / LDSM / swizzle.  The two-kernel main's substitute for the monolithic's
// LDSM of the smem-computed CB: the precompute STG'd each lane's fragA, so the
// reverse LDG repopulates the identical fragment.  REGS = M·K/32 = K/2 (M=16):
// m16n8k16 → 8 (LDG.128), m16n8k8 → 4 (LDG.64).  cb_head = &cb[batch_slot, head, 0].
template <int REGS, typename input_t, typename FragCB>
__device__ __forceinline__ void load_cb_fragA(FragCB& frag_CB, int lane,
                                              input_t const* __restrict__ cb_head) {
  // The fragment element is the MMA operand type (bit-compatible with input_t
  // but a distinct C++ type — e.g. cutlass bf16 vs nv_bfloat16).  Read the gmem
  // bytes AS that type so the per-element assignment is well-typed (no cast).
  using frag_t = cute::remove_cvref_t<decltype(frag_CB(0))>;
  using Pack = PackedAligned<frag_t, REGS>;
  Pack const packed = reinterpret_cast<Pack const*>(cb_head)[lane];  // one vectorized LDG
#pragma unroll
  for (int e = 0; e < REGS; ++e) frag_CB(e) = packed.val[e];
}

// 2b. frag_y += CB_scaled @ x  (matmul 4, single K-tile)
//     CB_scaled A operand: monolithic LDSMs swizzled smem; two-kernel main does
//     one LDG.128 via load_cb_fragA — either way frag_CB is in registers here.
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

// 2c. frag_y += CB_old @ old_x  (matmul-4 over old tokens; sibling of add_cb_x).
//     CB_old A-operand: pre-loaded by caller (m16n8k_old A-frag).
//     old_x  B-operand: ldmatrix.trans from smem.old_x viewed transposed.
// K_OLD = MAX_WINDOW_PAD_MMA_K ∈ {8, 16}.  Caller's tiled_mma_old uses the
// matching m16n8k_OLD atom (K_BIG=16 or K_SMALL=8).  frag_y partitioned by a
// different (K_BIG) tiled_mma is layout-compatible — the m16n8 C-frag shape is
// the same regardless of K.
template <typename input_t, typename MmaT, int N_TILE, int MAX_WINDOW_PAD_MMA_K, typename FragY,
          typename FragCBOld, typename SmemOldXTrans, typename S2RBTransOld,
          typename S2RThrBTransOld, typename ThrMmaOld, typename TiledMmaOld>
__device__ __forceinline__ void add_cb_old_x(FragY& frag_y, FragCBOld const& frag_CB_old,
                                             SmemOldXTrans const& smem_old_x_trans,
                                             S2RBTransOld const& s2r_B_trans_old,
                                             S2RThrBTransOld const& s2r_thr_B_trans_old,
                                             ThrMmaOld const& thr_mma_old,
                                             TiledMmaOld const& tiled_mma_old, int n) {
  using namespace cute;
  Tensor smem_old_x_ntile = local_tile(
      smem_old_x_trans, make_tile(Int<N_TILE>{}, Int<MAX_WINDOW_PAD_MMA_K>{}), make_coord(n, _0{}));
  auto smem_old_x_s2r = s2r_thr_B_trans_old.partition_S(smem_old_x_ntile);
  auto frag_B_old_x = thr_mma_old.partition_fragment_B(
      make_tensor((MmaT*)0x0, make_shape(Int<N_TILE>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  auto frag_B_old_x_view = s2r_thr_B_trans_old.retile_D(frag_B_old_x);

  cute::copy(s2r_B_trans_old, smem_old_x_s2r, frag_B_old_x_view);
  cute::gemm(tiled_mma_old, frag_y, frag_CB_old, frag_B_old_x, frag_y);
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

// x4 ldmatrix at a precomputed swizzled smem address (uint).  Mirrors cute's
// SM75_U32x4_LDSM_N::copy but takes the address directly, so the caller supplies
// a precomputed (off0 + Δk) ^ swizzle_mask offset instead of re-deriving the
// swizzle per access (see pipelined_kloop_gemm's hand-rolled A path).
__device__ __forceinline__ void ldmatrix_x4_addr(uint32_t addr, uint32_t& d0, uint32_t& d1,
                                                 uint32_t& d2, uint32_t& d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(addr));
#endif
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
  auto s2r_A = make_a_s2r<ATypeIn, MmaT>(tiled_mma);
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

  // A staging is only live when ATypeIn is wider than MmaT (f32 state as the
  // swapped OUT.1 A operand): the LDS.64 copy lands raw f32 pairs in FragAStg
  // and `convert_frag` narrows them into the MMA fragment.  2-byte ATypeIn
  // keeps the landed direct-load + in-place-convert path bit-identical (the
  // staging arrays below are dead and eliminated).
  constexpr bool kWideA = sizeof(ATypeIn) != sizeof(MmaT);

  // ── Fragment / view types ──
  using FragA = decltype(thr_mma.partition_fragment_A(smem_A_ktiled(_, _, _0{})));
  using FragB = decltype(thr_mma.partition_fragment_B(sample_smem_B_n(_, _, _0{})));
  using a_view_t = std::conditional_t<sizeof(ATypeIn) == sizeof(MmaT), MmaT, ATypeIn>;
  using b_view_t = std::conditional_t<sizeof(BTypeIn) == sizeof(MmaT), MmaT, BTypeIn>;
  using FragAStg = decltype(make_fragment_like<a_view_t>(std::declval<FragA>()));
  using FragBStg = decltype(make_fragment_like<b_view_t>(std::declval<FragB>()));
  // The direct A view only exists on the 2-byte path (retiling the MmaT
  // fragment against the wide copy atom would be ill-formed) — resolve the
  // conditional BEFORE taking the decltype so it is never instantiated.
  using FragAView =
      decltype(s2r_thr_A.retile_D(std::declval<std::conditional_t<kWideA, FragAStg, FragA>&>()));
  using FragAStgView = decltype(s2r_thr_A.retile_D(std::declval<FragAStg&>()));
  using FragBStgView = decltype(s2r_thr_B.retile_D(std::declval<FragBStg&>()));

  // ── Multi-stage register fragments ──
  // Storage type matches the MMA fragment for A; for B the staging buffer is
  // BTypeIn-typed (when narrowing) or MmaT-typed (when widths match — the two
  // alias the same registers and `convert_frag` collapses to a bit-copy /
  // in-place reinterpret).
  FragA frag_A[NumStages];
  FragB frag_B[NumNTiles][NumStages];
  FragAStg frag_A_stg[NumStages];
  FragBStg frag_B_stg[NumNTiles][NumStages];
  FragAView frag_A_view[NumStages];
  FragAStgView frag_A_stg_view[NumStages];
  FragBStgView frag_B_stg_view[NumNTiles][NumStages];
  CUTE_UNROLL
  for (int s = 0; s < NumStages; ++s) {
    if constexpr (!kWideA) {
      frag_A_view[s] = s2r_thr_A.retile_D(frag_A[s]);
    } else {
      frag_A_stg_view[s] = s2r_thr_A.retile_D(frag_A_stg[s]);
    }
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

  // ── Hand-rolled A-operand (C) LDSM source addressing ──
  // The LDSM source for k-tile k is A_base + byteoffA[k].  byteoffA[k] is the
  // swizzled byte offset of this lane's k-tile *relative to the C slot base* — a
  // pure function of (lane, k): the ring-slot pointer lives in A_base (the engine
  // .data()), not in the layout, so byteoffA is invariant across heads.  Computing
  // it from the static swizzled layout lets LICM hoist the 8 swizzle evals out of
  // the head loop (once per thread) instead of cute re-deriving them per access,
  // bundled with the per-head base pointer.  recast<uint32_t> of the retiled
  // fragment matches what cute::copy writes internally, so the dst register order
  // is identical — only the source-address path is replaced.
  using SwizA = decltype(get_swizzle_portion(smem_A_ktiled.layout()));
#if defined(SSU_NO_HANDROLL_A)
  constexpr bool kHandrollA = false;  // A/B switch: -DSSU_NO_HANDROLL_A reverts to cute::copy
#else
  // Hand-rolled addressing is the ldmatrix path — 2-byte A only (wide f32 A
  // loads via the LDS.64 tiled copy below).
  constexpr bool kHandrollA = (SwizA::num_bits > 0) && (sizeof(MmaT) == 2) && !kWideA;
#endif
  uint32_t const A_base = cast_smem_ptr_to_uint(raw_pointer_cast(smem_A_s2r.data()));
  // byteoffA[k] is the stored swizzled byte offset (swizzle + offset baked in); per
  // access we only add A_base.  Computed via the explicit closed form: off0 + Ck
  // (compile-time k-delta) swizzled by the invariant mask X_A = (row&7)<<3 — cheap
  // integer ALU rather than cute's per-k layout machinery.  off0_A recovered by the
  // involution SwizA{}(swizzled) == unswizzled (Swizzle is its own inverse).
  int const swz0_A = smem_A_s2r.layout()(make_coord(_0{}, _0{}, _0{}, _0{}));
  int const off0_A = SwizA{}(swz0_A);
  int const X_A = swz0_A ^ off0_A;
  auto const plainA = get_nonswizzle_portion(smem_A_s2r.layout());
  uint32_t byteoffA[NumKTiles];
  CUTE_UNROLL
  for (int k = 0; k < NumKTiles; ++k)
    byteoffA[k] = uint32_t((off0_A + int(plainA(make_coord(_0{}, _0{}, _0{}, k)))) ^ X_A) *
                  uint32_t(sizeof(MmaT));

  // NOTE: the state (B) operand was tried with the same hand-rolled precomputed
  // addressing (2D table, inline layout-eval, and off0+invariant-mask forms — all
  // measured) and consistently regressed ~1-1.5us at b1024: unlike A's swizzle (a
  // hoistable per-access recompute), B's LDSM is not the bottleneck, and the
  // just-in-time address compute only lengthens the load→HMMA dependency chain in
  // the software pipeline.  B stays on cute::copy.

  // ── Per-stage operations (slot is constant after #pragma unroll) ──
  auto load_one = [&](int k_src, int slot) {
    if constexpr (kHandrollA) {
      auto rA = recast<uint32_t>(frag_A_view[slot]);
      ldmatrix_x4_addr(A_base + byteoffA[k_src], rA(0), rA(1), rA(2), rA(3));
#if defined(SSU_VERIFY_LDSM)
      assert(A_base + byteoffA[k_src] ==
             cast_smem_ptr_to_uint(&smem_A_s2r(_0{}, _0{}, _0{}, k_src)));
#endif
    } else if constexpr (kWideA) {
      cute::copy(s2r_A, smem_A_s2r(_, _, _, k_src), frag_A_stg_view[slot]);
    } else {
      cute::copy(s2r_A, smem_A_s2r(_, _, _, k_src), frag_A_view[slot]);
    }
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      cute::copy(s2r_B, smem_B_s2r[n](_, _, _, k_src), frag_B_stg_view[n][slot]);
    }
  };
  auto convert_one = [&](int slot) {
    if constexpr (kWideA) {
      convert_frag<ATypeIn, MmaT>(frag_A_stg[slot], frag_A[slot]);
    } else {
      convert_frag<ATypeIn, MmaT>(frag_A[slot]);
    }
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      convert_frag<BTypeIn, MmaT>(frag_B_stg[n][slot], frag_B[n][slot]);
    }
  };
  // ── 2-way accumulator split (C@state, ≥2 k-tiles) ──
  // frag_y += A[k]@B[k] is a serial chain: HMMA[k] reads HMMA[k-1]'s frag_y.
  // Routing odd-k tiles into a second accumulator gives two independent chains → 2
  // HMMAs in flight, filling the tensor pipe (otherwise ~10% utilized) via per-warp
  // ILP — needed because occupancy is only ~0.95 eligible warps/cycle, too low to
  // hide HMMA latency across warps.  Reduced into frag_y_p after the loop.  Single-
  // k-tile matmuls have no chain to break.
#if defined(SSU_NO_ACC_SPLIT)
  constexpr bool kSplitAcc = false;  // A/B switch: -DSSU_NO_ACC_SPLIT keeps one accumulator
#else
  constexpr bool kSplitAcc = (NumKTiles >= 2);
#endif
  FragY0 acc2[NumNTiles];

  // ── Clear accumulators ──
  CUTE_UNROLL
  for (int n = 0; n < NumNTiles; ++n) {
    clear(*frag_y_p[n]);
    if constexpr (kSplitAcc) clear(acc2[n]);
  }

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
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      if constexpr (kSplitAcc) {
        // (k & 1) folds at compile time (loop fully unrolled) → one static
        // accumulator per tile; even→frag_y, odd→acc2 are independent chains.
        if (k & 1)
          cute::gemm(tiled_mma, acc2[n], frag_A[slot_compute], frag_B[n][slot_compute], acc2[n]);
        else
          cute::gemm(tiled_mma, *frag_y_p[n], frag_A[slot_compute], frag_B[n][slot_compute],
                     *frag_y_p[n]);
      } else {
        cute::gemm(tiled_mma, *frag_y_p[n], frag_A[slot_compute], frag_B[n][slot_compute],
                   *frag_y_p[n]);
      }
    }
    if (k_load < NumKTiles) convert_one(slot_load);
  }

  // ── Reduce the odd-k accumulator into frag_y_p ──
  if constexpr (kSplitAcc) {
    CUTE_UNROLL
    for (int n = 0; n < NumNTiles; ++n) {
      CUTE_UNROLL
      for (int i = 0; i < cute::size(*frag_y_p[n]); ++i) (*frag_y_p[n])(i) += acc2[n](i);
    }
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
                                             ThrMma const& thr_mma, int tid, int state_buf,
                                             FragY&... frag_y) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  // Smem source dtype for matmul-3 (generic kernel only; 8-bit state goes
  // through the dedicated `checkpointing_ssu_kernel_8bit` path):
  //   - sizeof(state_t) == 2 (fp16/bf16): LDSM the native 16-bit, view as bf16.
  //   - sizeof(state_t) == 4 (fp32): scalar UniversalCopy + on-the-fly convert.
  static_assert(sizeof(state_t) != 1,
                "add_init_out is the 2/4-byte path; 1-byte state goes through "
                "compute_output_8bit");
  constexpr bool is_2byte_smem = (sizeof(state_t) == 2);
  using state_view_t = std::conditional_t<is_2byte_smem, MMA_prop::operand_t, state_t>;
  using BTypeIn = state_t;

  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  Tensor smem_C_ktiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                    make_coord(_0{}, _));

  // Swizzle layout matches the dtype of the buffer being viewed.
  auto const layout_state_swz = make_swizzled_layout_rc<BTypeIn, D_PER_CTA, DSTATE>();
  // state_buf selects the double-buffered slot; offset in state_t units before the view cast.
  state_view_t const* smem_state_ptr = reinterpret_cast<state_view_t const*>(
      reinterpret_cast<state_t const*>(smem.state) + state_buf * D_PER_CTA * DSTATE);
  Tensor smem_state = make_tensor(make_smem_ptr(smem_state_ptr), layout_state_swz);

  pipelined_kloop_gemm<3, NUM_K_TILES, input_t, BTypeIn, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_C_ktiled, smem_state, frag_y...);
}

// ── Matmul 3 (OPERAND SWAP): init_out^T = state @ C^T ────────────────────────
// frag_y[d, t] = Σ_n state[d,n]·C[t,n].  A = state (K-major smem; 2-byte via
// LDSM, 4-byte f32 via LDS.64 — dispatched inside pipelined_kloop_gemm by
// make_a_s2r), B = C (aliased swizzled view, LDSM_N, broadcast across the
// M-warps).  The monolith counterpart of the main's add_init_out_main; the
// caller's tiled_mma splits M = D_PER_CTA across warps (Shape<M_WARPS, 1>).
// NumStages = 2 (not the main's 3): the monolith's register context is fatter
// (replay + CB compute inline, no maxnreg cap), and the f32 wide-A staging at
// depth 3 (3×8 f32 + 3×4 u32 per thread) tips it over — depth 2 halves that.
template <typename input_t, typename state_t, int D_PER_CTA, int DSTATE, typename SmemT,
          typename TiledMma, typename ThrMma, typename... FragY>
__device__ __forceinline__ void add_init_out_swapped(SmemT const& smem, TiledMma const& tiled_mma,
                                                     ThrMma const& thr_mma, int tid, int state_buf,
                                                     FragY&... frag_y) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr int M_TILE = cute::tile_size<0>(TiledMma{});
  static_assert(sizeof(state_t) != 1, "8-bit state goes through the dedicated 8-bit kernel");
  using AView = std::conditional_t<sizeof(state_t) == 2, MMA_prop::operand_t, state_t>;
  auto const layout_state = make_swizzled_layout_rc<AView, D_PER_CTA, DSTATE>();
  Tensor smem_state = make_tensor(
      make_smem_ptr(reinterpret_cast<AView const*>(reinterpret_cast<state_t const*>(smem.state) +
                                                   state_buf * D_PER_CTA * DSTATE)),
      layout_state);
  Tensor smem_state_ktiled =
      local_tile(smem_state, make_tile(Int<M_TILE>{}, Int<K_TILE>{}), make_coord(_0{}, _));
  auto const layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C =
      make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)), layout_C);
  pipelined_kloop_gemm<2, NUM_K_TILES, state_t, input_t, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_state_ktiled, smem_C, frag_y...);
}

// ── OPERAND-SWAP output helpers ([DIM,NPRED]) ────────────────────────────────
// x / old_x are the A-operands (transpose ldmatrix from the [token,d] smem via the [d,token]
// view); CB / CB_old are the B-operands (fragB, in registers).  Kept separate from the shared
// add_cb_x / add_D_skip / compute_z_gating (the 8-bit kernel still uses those, unswapped).

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

// store_state: vectorized smem → gmem state writeback (128 threads).
// Defined here (rather than alongside the other Phase 3 store helpers
// below) because compute_and_store_output calls it inline — issued right
// after matmul 3 so the STGs fire-and-forget in parallel with matmul 4 +
// epilogue.  smem and gmem hold the same dtype now (no on-egress
// conversion) so this is always a direct 128-bit copy.
template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void store_state(SmemT& smem, CheckpointingSsuParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot, int state_buf = 0) {
  using namespace cute;
  static_assert(D_PER_CTA % NUM_WARPS == 0, "D_PER_CTA must be divisible by NUM_WARPS");
  constexpr int DIM_PER_WARP = D_PER_CTA / NUM_WARPS;
  static_assert(DIM_PER_WARP % 4 == 0,
                "DIM_PER_WARP (D_PER_CTA/NUM_WARPS) must be a multiple of 4 for the (4,8) store");
  auto* __restrict__ state_w = reinterpret_cast<state_t*>(params.state);
  // gmem dest = head's full state base + d_tile's row slice.
  int64_t const state_base = cache_slot * params.state_stride_seq +
                             (int64_t)(head * DIM * DSTATE + d_tile * D_PER_CTA * DSTATE);

  // Per-warp D-slice store — the exact inverse of load_state_per_warp: each warp writes its
  // own DIM_PER_WARP rows with a 32-thread (4,8) copy, so ALL NUM_WARPS warps participate
  // (symmetric with the load; no 128-thread cap, no idle warps).  NUM_WARPS=4 → 16 rows/warp
  // (byte-identical gmem result to the old cooperative copy); 8 → 8 rows/warp.
  auto layout_smem_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  Tensor sState_full = make_tensor(
      make_smem_ptr(reinterpret_cast<state_t const*>(smem.state) + state_buf * D_PER_CTA * DSTATE),
      layout_smem_swz);
  Tensor gState_full = make_tensor(make_gmem_ptr(state_w + state_base),
                                   make_layout(make_shape(Int<D_PER_CTA>{}, Int<DSTATE>{}),
                                               make_stride(Int<DSTATE>{}, Int<1>{})));
  Tensor sState = local_tile(sState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));
  Tensor gState = local_tile(gState_full, make_shape(Int<DIM_PER_WARP>{}, Int<DSTATE>{}),
                             make_coord(warp, _0{}));
  // Each store is 16 bytes — adjust val cols to the dtype.
  constexpr int VAL_COLS = Copy_prop::vec_bytes / sizeof(state_t);
  auto s2g =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, state_t>{},
                      Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, Int<VAL_COLS>>>{});
  auto thr = s2g.get_slice(lane);
  copy(s2g, thr.partition_S(sState), thr.partition_D(gState));
}

// ── Store functions (called from kernel after compute_y + sync) ──
// (store_state moved above compute_and_store_output — used there for
// the state-writeback hoist.)

template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, typename SmemT>
__device__ __forceinline__ void store_old_x(SmemT& smem, CheckpointingSsuParams const& params,
                                            int warp, int lane, int d_tile, int head,
                                            int64_t cache_slot, int ring_start, int write_offset,
                                            int seq_len, int tile_buf = 0) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_x_w = reinterpret_cast<input_t*>(params.x_cache);
  // gmem dest = ring rows (ring_start + write_offset + i) % RING_BUFFER_LEN;
  // write_offset is the LOGICAL append offset (= prev_k on both branches —
  // a flush only advances ring_start, host-side, after the call).
  int r0 = ring_start + write_offset;
  if (r0 >= params.ring_buffer_len) r0 -= params.ring_buffer_len;
  int const n1 = params.ring_buffer_len - r0 >= seq_len ? seq_len : params.ring_buffer_len - r0;
  int64_t const ox_w_base = cache_slot * params.x_cache_stride_seq +
                            (int64_t)head * params.x_cache_stride_head +
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
  auto const* x_base = smem.x + tile_buf * NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(x_base)), layout_x_swz);

  // Wide layout (16×8 threads): uses all 128 threads for maximum STG.128 bandwidth.
  // UniversalCopy<uint128_t> has no .with(bool) → copy_if falls back to software predication
  // at copy_atom.hpp:149, which produces 4-way LDS.128 bank conflicts (Swizzle<3,3,3> is
  // designed for ldmatrix, not plain LDS.128, and 4 rows-per-warp always hit the same 8 bank
  // groups).  The conflicts are intentional: switching to ldmatrix would require a full
  // MMA-layout writeback and lose the wide STG.128 path.  NCU will show excess wavefronts here.
  using ThrLayoutX = Layout<Shape<_16, _8>, Stride<_8, _1>>;
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{}, ThrLayoutX{},
                             Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsX = thr_s2g.partition_S(sX);
  auto cX = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}));
  auto tScX = thr_s2g.partition_D(cX);

  // Two disjoint-row segments cover the ring wrap: smem rows [0, n1) land at
  // ring rows r0+…, rows [n1, seq_len) at ring rows 0+… (base − n1·stride).
  int const seg_lo[2] = {0, n1};
  int const seg_hi[2] = {n1, seq_len};
  int64_t const seg_base[2] = {(int64_t)r0, -(int64_t)n1};
  CUTE_UNROLL
  for (int sg = 0; sg < 2; ++sg) {
    if (seg_lo[sg] >= seg_hi[sg]) continue;
    Tensor gX =
        make_tensor(make_gmem_ptr(old_x_w + ox_w_base + seg_base[sg] * params.x_cache_stride_pos),
                    make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<D_SMEM_COLS>{}),
                                make_stride(params.x_cache_stride_pos, Int<1>{})));
    auto tSgX = thr_s2g.partition_D(gX);
    auto pred = make_tensor<bool>(shape(tScX));
    CUTE_UNROLL
    for (int i = 0; i < size(pred); ++i) {
      pred(i) = (get<0>(tScX(i)) >= seg_lo[sg]) && (get<0>(tScX(i)) < seg_hi[sg]) &&
                (get<1>(tScX(i)) < D_PER_CTA);
    }
    copy_if(s2g, pred, tSsX, tSgX);
  }
}

// store_old_B runs on W0, W1 only (64 threads).  Caller must gate
// with `if (warp < 2)` — these are the warps that hold valid smem.B
// after their own cp.async + wait.  Halving the thread count keeps the
// overlap (writeback fires before CB+replay consume smem.B).
//
// Source: smem.B with NPREDICTED_PAD_MMA_N rows.
// Destination: B_cache ring rows (ring_start + write_offset + i) % RING_BUFFER_LEN.
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
__device__ __forceinline__ void store_old_B(SmemT& smem, CheckpointingSsuParams const& params,
                                            int warp, int lane, int head, int group_idx,
                                            int64_t cache_slot, int ring_start, int write_offset,
                                            int seq_len) {
  using namespace cute;
  if (head % HEADS_PER_GROUP != 0) return;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;  // matches smem.B row count
  // Called only from warps 0, 1 — flat_tid ∈ [0, 64).
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_B_w = reinterpret_cast<input_t*>(params.B_cache);
  int r0 = ring_start + write_offset;
  if (r0 >= params.ring_buffer_len) r0 -= params.ring_buffer_len;
  int const n1 = params.ring_buffer_len - r0 >= seq_len ? seq_len : params.ring_buffer_len - r0;
  int64_t const oB_base =
      cache_slot * params.B_cache_stride_seq + (int64_t)group_idx * params.B_cache_stride_group;

  auto layout_B_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_N, DSTATE>();
  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B_swz);

  // 64 threads, (8, 8) × (1, 8) = atom-aligned per-tile (8, 64).
  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_8, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsB = thr_s2g.partition_S(sB);
  auto cB = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_N>{}, Int<DSTATE>{}));
  auto tScB = thr_s2g.partition_D(cB);

  // Two disjoint-row segments cover the ring wrap (see store_old_x).  The
  // pre-ring seq_len == NPREDICTED unpredicated fast path is subsumed: the
  // no-wrap case is a single predicated pass with rows [0, seq_len).
  int const seg_lo[2] = {0, n1};
  int const seg_hi[2] = {n1, seq_len};
  int64_t const seg_base[2] = {(int64_t)r0, -(int64_t)n1};
  CUTE_UNROLL
  for (int sg = 0; sg < 2; ++sg) {
    if (seg_lo[sg] >= seg_hi[sg]) continue;
    Tensor gB =
        make_tensor(make_gmem_ptr(old_B_w + oB_base + seg_base[sg] * params.B_cache_stride_pos),
                    make_layout(make_shape(Int<NPREDICTED_PAD_MMA_N>{}, Int<DSTATE>{}),
                                make_stride(params.B_cache_stride_pos, Int<1>{})));
    auto tSgB = thr_s2g.partition_D(gB);
    auto pred = make_tensor<bool>(shape(tScB));
    CUTE_UNROLL
    for (int i = 0; i < size(pred); ++i) {
      pred(i) = (get<0>(tScB(i)) >= seg_lo[sg]) && (get<0>(tScB(i)) < seg_hi[sg]);
    }
    copy_if(s2g, pred, tSsB, tSgB);
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_COMMON_CUH_
