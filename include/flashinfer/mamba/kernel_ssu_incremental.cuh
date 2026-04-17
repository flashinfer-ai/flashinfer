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
// Phase 0 (overlapped across warps, before the single __syncthreads):
//   Warp 0: cp.async B + old_B (swizzled).
//   Warp 1: scalar loads (old_dt_proc, old_cumAdt, dt→dt_proc) + cumAdt cumsum.
//   Warp 2: cp.async x + old_x (swizzled).
//   Warp 3: cp.async C + z (swizzled).
//   All warps: state load into swizzled smem (cp.async for 2-byte state_t,
//              plain LDG + bf16 narrow for f32 state_t).
//
// __syncthreads() + cp.async.wait
//
// Phase 1:
//   - store_old_B hoisted here (fires STGs while compute proceeds).
//   - Warps 0,1: compute_CB_scaled_2warp (bf16 HMMA, scaled & masked result in
//     swizzled smem).
//   - All warps: replay_state_mma (HMMA; state in smem updated in-place).
//
// __syncthreads()
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

// =============================================================================
// cp.async helper (SM80+)
// =============================================================================
// cp.async with L2::128B eviction hint and L1 caching (.ca).
// CuTe's SM80_CP_ASYNC_CACHEALWAYS uses .ca (cache all levels), not .cg (bypass L1).
// On B200 (SM100), .ca resolves smem bank conflicts without replay; .cg does not.
__device__ __forceinline__ void cp_async_16B_L2(void* __restrict__ smem_dst,
                                                void const* __restrict__ gmem_src) {
  unsigned int smem_addr = __cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem_src)
               : "memory");
}

// Round up to next multiple of 16 (for mma.sync.m16n8k16 tile alignment).
constexpr int next_multiple_of_16(int x) { return (x + 15) & ~15; }

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
  // Padded token count for mma.sync.m16n8k16 tile alignment.
  static constexpr int NTOKENS_PAD = next_multiple_of_16(NTOKENS);

  using smem_state_t = smem_state_t_of<state_t>;

  // CB_scaled: input_t[NTOKENS_PAD * 64] — padded row stride 128 bytes
  // (32 banks), accessed via Swizzle<3,3,3> CuTe layout for conflict-free
  // LDSM in the matmul-4 A-operand read.
  alignas(16) char CB_scaled[NTOKENS_PAD * 64 * sizeof(input_t)];

  // B and C: padded to NTOKENS_PAD rows for mma.sync operand alignment.
  // Padding rows contain garbage — valid output uses only [0, NTOKENS).
  alignas(16) input_t B[NTOKENS_PAD][DSTATE];
  alignas(16) input_t C[NTOKENS_PAD][DSTATE];

  // x: padded to NTOKENS_PAD rows for mma operand alignment (matmul 4 B operand).
  alignas(16) input_t x[NTOKENS_PAD][DIM];

  // z: padded to NTOKENS_PAD so partition_C reads don't go OOB.
  alignas(16) input_t z[NTOKENS_PAD][DIM];

  // Old cache data loaded in Phase 0 (consumed in Phase 1 replay).
  // old_x: padded to NTOKENS_PAD rows, Swizzle<3,3,3> [NTOKENS_PAD, DIM]
  // layout — ldmatrix.trans feeds replay MMA A-operand directly.
  alignas(16) input_t old_x[NTOKENS_PAD][DIM];

  // old_B: padded to NTOKENS_PAD rows, Swizzle<3,3,3> [NTOKENS_PAD, DSTATE]
  // layout. Replay MMA reads via ldmatrix.trans (LDSM_T) + register scaling.
  // Padding rows zero-filled via cp.async ZFILL.
  alignas(16) input_t old_B[NTOKENS_PAD][DSTATE];

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
// B/C/x/z load helpers: single-warp cp.async into swizzled smem.
// =============================================================================

// Swizzled load via CuTe TiledCopy with ZFILL predicate.
// Both gmem and smem use NTOKENS_PAD shape. Rows >= NTOKENS are zero-filled
// in smem without reading from gmem (cp.async ZFILL).
template <typename input_t, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_B_cute(SmemT& smem, input_t const* __restrict__ B_ptr,
                                            int64_t B_base, int B_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.B[0][0])),
                          make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>());
  Tensor gB = make_tensor(make_gmem_ptr(B_ptr + B_base),
                          make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}),
                                      make_stride(B_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  // Row predicate: true for rows < NTOKENS, false for padding (zero-filled, no gmem read)
  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gB), thr.partition_D(sB));
}

// Single-warp swizzled load for old_B via CuTe TiledCopy with ZFILL predicate.
// Mirrors load_B_cute but targets smem.old_B.
template <typename input_t, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_old_B_cute(SmemT& smem, input_t const* __restrict__ old_B_ptr,
                                                int64_t oB_base, int old_B_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sOB = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.old_B[0][0])),
                           make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>());
  Tensor gOB = make_tensor(make_gmem_ptr(old_B_ptr + oB_base),
                           make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}),
                                       make_stride(old_B_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gOB), thr.partition_D(sOB));
}

// Swizzled load for old_x via CuTe TiledCopy with ZFILL predicate.
// Smem shape [NTOKENS_PAD, DIM] swizzled. Rows >= NTOKENS are zero-filled.
template <typename input_t, int NTOKENS, int DIM, typename SmemT>
__device__ __forceinline__ void load_old_x_cute(SmemT& smem, input_t const* __restrict__ old_x_ptr,
                                                int64_t ox_base, int ox_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sOX = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.old_x[0][0])),
                           make_swizzled_layout_rc<NTOKENS_PAD, DIM>());
  Tensor gOX = make_tensor(make_gmem_ptr(old_x_ptr + ox_base),
                           make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}),
                                       make_stride(ox_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gOX), thr.partition_D(sOX));
}

// Swizzled x load via CuTe TiledCopy with ZFILL predicate.
template <typename input_t, int NTOKENS, int DIM, typename SmemT>
__device__ __forceinline__ void load_x_cute(SmemT& smem, input_t const* __restrict__ x_ptr,
                                            int64_t x_base, int x_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.x[0][0])),
                          make_swizzled_layout_rc<NTOKENS_PAD, DIM>());
  Tensor gX = make_tensor(
      make_gmem_ptr(x_ptr + x_base),
      make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}), make_stride(x_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gX), thr.partition_D(sX));
}

// Swizzled z load via CuTe TiledCopy with ZFILL predicate.
template <typename input_t, int NTOKENS, int DIM, typename SmemT>
__device__ __forceinline__ void load_z_cute(SmemT& smem, input_t const* __restrict__ z_ptr,
                                            int64_t z_base, int z_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sZ = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.z[0][0])),
                          make_swizzled_layout_rc<NTOKENS_PAD, DIM>());
  Tensor gZ = make_tensor(
      make_gmem_ptr(z_ptr + z_base),
      make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}), make_stride(z_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gZ), thr.partition_D(sZ));
}

// Swizzled C load via CuTe TiledCopy with ZFILL predicate.
template <typename input_t, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_C_cute(SmemT& smem, input_t const* __restrict__ C_ptr,
                                            int64_t C_base, int C_stride_mtp, int lane) {
  using namespace cute;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;

  Tensor sC = make_tensor(make_smem_ptr(reinterpret_cast<input_t*>(&smem.C[0][0])),
                          make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>());
  Tensor gC = make_tensor(make_gmem_ptr(C_ptr + C_base),
                          make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}),
                                      make_stride(C_stride_mtp, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, input_t>{},
                             Layout<Shape<_4, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(lane);

  auto id = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}));
  auto thr_id = thr.partition_S(id);
  auto pred = make_tensor<bool>(shape(thr_id));
  CUTE_UNROLL
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(thr_id(i)) < NTOKENS;
  }
  copy_if(g2s, pred, thr.partition_S(gC), thr.partition_D(sC));
}

// Manual swizzled C load using cp_async_16B_L2 (with .L2::128B hint).
// Same logic as the original load_C_cute manual loop, but with the L2 hint
// that CuTe's SM80_CP_ASYNC generates. For A/B testing vs TiledCopy.
template <typename input_t, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_C_cute_manual(SmemT& smem, input_t const* __restrict__ C_ptr,
                                                   int64_t C_base, int C_stride_mtp, int lane) {
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  constexpr int INPUT_PACK = 16 / sizeof(input_t);
  auto layout = make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>();
  input_t* base = &smem.C[0][0];

  constexpr int num_packs = NTOKENS * DSTATE / INPUT_PACK;
  constexpr int packs_per_row = DSTATE / INPUT_PACK;
  for (int i = lane; i < num_packs; i += warpSize) {
    int const t = i / packs_per_row;
    int const col = (i % packs_per_row) * INPUT_PACK;
    cp_async_16B_L2(base + layout(t, col), &C_ptr[C_base + t * C_stride_mtp + col]);
  }
}

// State load helpers.
// All warps cooperate, using flat_tid (warp * 32 + lane).
// Swizzled state load via CuTe TiledCopy.
// 128 threads (all warps), thread layout (16,8) stride (8,1).
template <typename state_t, int DIM, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_cute(SmemT& smem, state_t const* __restrict__ state_ptr,
                                                int64_t state_base, int flat_tid) {
  using namespace cute;
  static_assert(sizeof(state_t) == 2, "load_state_cute requires 2-byte state type");
  static_assert(NUM_WARPS * warpSize == 128, "Expected 128 threads for state TiledCopy");

  Tensor sState = make_tensor(make_smem_ptr(reinterpret_cast<state_t*>(&smem.state[0][0])),
                              make_swizzled_layout_rc<DIM, DSTATE>());
  Tensor gState = make_tensor(
      make_gmem_ptr(state_ptr + state_base),
      make_layout(make_shape(Int<DIM>{}, Int<DSTATE>{}), make_stride(Int<DSTATE>{}, Int<1>{})));

  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, state_t>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr = g2s.get_slice(flat_tid);
  copy(g2s, thr.partition_S(gState), thr.partition_D(sState));
}

// Narrowing state load: gmem f32 → swizzled smem bf16.  Used only when
// state_t is f32 — the tensor-core path wants a 2-byte state in smem, so we
// pay the conversion once on ingress and let the body of the kernel run
// through the uniform bf16 fast path.  Plain LDG (cp.async cannot narrow).
// Element-at-a-time write so we don't rely on swizzle-adjacency of (col,
// col+1) — correctness first, since f32 state isn't a hot path.
template <int DIM, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_state_f32_to_bf16(SmemT& smem,
                                                       float const* __restrict__ state_ptr,
                                                       int64_t state_base, int flat_tid) {
  using namespace cute;
  static_assert(std::is_same_v<typename SmemT::smem_state_t, __nv_bfloat16>,
                "narrowing path assumes bf16 smem state");

  auto layout_smem_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(&smem.state[0][0]);
  float const* gmem_state = state_ptr + state_base;

  constexpr int NUM_THREADS = NUM_WARPS * warpSize;
  constexpr int TOTAL = DIM * DSTATE;

  for (int i = flat_tid; i < TOTAL; i += NUM_THREADS) {
    int const row = i / DSTATE;
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

// Loads B, C, x, z, state (cp.async 16B), old_x, old_B (plain LDG),
// old_dt_proc, old_cumAdt (plain LDG), dt → dt_proc (LDG + softplus → smem).
// Commits cp.async and waits (caller must __syncthreads after).
// =============================================================================
template <typename input_t, typename dt_t, typename state_t, int NTOKENS, int DIM, int DSTATE,
          int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_data(SmemT& smem, SsuIncrementalParams const& params, int lane,
                                          int warp, int batch_idx, int head, int group_idx,
                                          int64_t cache_slot, int buf_read, float A_val,
                                          float dt_bias_val) {
  int const flat_tid = warp * warpSize + lane;
  constexpr int num_threads = NUM_WARPS * warpSize;

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

  // ── Warp 0: cp.async B[T, dstate] + old_B[T, dstate] (swizzled smem) ──
  if (warp == 0) {
    load_B_cute<input_t, NTOKENS, DSTATE>(smem, B_ptr, B_base, params.B_stride_mtp, lane);
    load_old_B_cute<input_t, NTOKENS, DSTATE>(smem, old_B_ptr, oB_base, params.old_B_stride_mtp,
                                              lane);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  // ── Warp 1: scalar loads + cumAdt reduction (all synchronous, no cp.async) ──
  static_assert(NTOKENS <= warpSize, "NTOKENS must fit in a single warp");
  if (warp == 1) {
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
    // cumAdt = cumsum(A * dt_proc). dt_proc written above (synchronous).
    // cumAdt must be in smem before __syncthreads so all warps can read it.
    compute_cumAdt<NTOKENS>(smem, lane, A_val);
  }

  // ── Warp 2: cp.async x[T, dim] + old_x[T, dim] (swizzled) ──
  if (warp == 2) {
    load_x_cute<input_t, NTOKENS, DIM>(smem, x_ptr, x_base, params.x_stride_mtp, lane);
    load_old_x_cute<input_t, NTOKENS, DIM>(smem, old_x_ptr, ox_base, params.old_x_stride_mtp, lane);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  // ── Warp 3: cp.async C[T, dstate] + z[T, dim] (swizzled) ──
  if (warp == 3) {
    load_C_cute_manual<input_t, NTOKENS, DSTATE>(smem, C_ptr, C_base, params.C_stride_mtp, lane);
    // load_C_cute<input_t, NTOKENS, DSTATE>(smem, C_ptr, C_base, params.C_stride_mtp, lane);
    if (z_ptr) {
      int64_t const z_base = (int64_t)batch_idx * params.z_stride_batch + head * DIM;
      load_z_cute<input_t, NTOKENS, DIM>(smem, z_ptr, z_base, params.z_stride_mtp, lane);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }

  // ── All warps: state[DIM, DSTATE] → swizzled smem ──
  // 2-byte state_t: cp.async directly.  f32 state_t: plain LDG + narrow to
  // bf16, since cp.async cannot narrow dtypes.
  {
    auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
    int64_t const state_base =
        cache_slot * params.state_stride_batch + (int64_t)head * DIM * DSTATE;
    if constexpr (sizeof(state_t) == 2) {
      load_state_cute<state_t, DIM, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, flat_tid);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    } else {
      static_assert(sizeof(state_t) == 4, "unexpected state_t size");
      load_state_f32_to_bf16<DIM, DSTATE, NUM_WARPS>(
          smem, reinterpret_cast<float const*>(state_ptr), state_base, flat_tid);
      // No cp.async issued in the narrow path, so nothing to commit.
    }
  }

  // Wait for all cp.async groups to complete
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");

  // x padding rows are already zero-filled by load_x_cute's ZFILL predicate.
}

// (compute_cumAdt moved above load_data so it can be called from there)

// Compute CB_scaled[T,T] = (C @ B^T) * decay * dt_proc * causal_mask.
// Split across 2 warps: warp 0 computes columns 0:8, warp 1 computes columns 8:16.
// Result stored to swizzled smem.CB_scaled (input_t, row stride 64, Swizzle<3,3,3>).
// Called between the two __syncthreads by warps 0 and 1 only.
template <typename input_t, typename mma_type, int NTOKENS, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_CB_scaled_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;

  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  constexpr int N_HALF = 8;  // each warp computes [NTOKENS_PAD, 8]

  // Always bf16 MMA — activations are bf16; f16 state is converted in add_init_out_cute.
  using MmaAtomType = SM80_16x8x16_F32BF16BF16F32_TN;

  // ── Swizzled smem views ──
  auto layout_bc = make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>();
  Tensor smem_C =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.C[0][0])), layout_bc);
  Tensor smem_B =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.B[0][0])), layout_bc);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  // ── K-tile A operand (C): full [NTOKENS_PAD, K_TILE], shared by both warps ──
  constexpr int K_TILE = 16;
  Tensor smem_C_tiled =
      local_tile(smem_C, make_tile(Int<NTOKENS_PAD>{}, Int<K_TILE>{}), make_coord(_0{}, _));

  // ── K-tile B operand: each warp gets its N-half of B [N_HALF, K_TILE] ──
  Tensor smem_B_half =
      local_tile(smem_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp, _));

  // ── Register fragments ──
  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_B_half(_, _, _0{}));

  // ── Output accumulator: [NTOKENS_PAD, N_HALF] f32 ──
  auto layout_cb_half = make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<N_HALF>{}));
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
  auto id_half = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<N_HALF>{}));
  auto id_part = thr_mma.partition_C(id_half);

  // ── Store to swizzled smem.CB_scaled ──
  auto layout_cb_swz = composition(Swizzle<3, 3, 3>{},
                                   make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<NTOKENS_PAD>{}),
                                               make_stride(Int<64>{}, _1{})));
  Tensor smem_CB =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type*>(smem.CB_scaled)), layout_cb_swz);
  // Tile into [NTOKENS_PAD, N_HALF] halves; warp selects its half
  Tensor smem_CB_half =
      local_tile(smem_CB, make_tile(Int<NTOKENS_PAD>{}, Int<N_HALF>{}), make_coord(_0{}, warp));
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
// Returns 4 floats in coeff[], one per m16n8k16 B-fragment element.
// coeff[i] = 0 when k >= prev_k, embedding the causal mask so the inner
// loop needs no branch.
//
// K-index derivation (m16n8k16 _TN MMA, B fragment = 4 bf16 per thread):
//   lane = tid % 32;  K_base = (lane % 4) * 2;
//   elem 0 → K_base,   elem 1 → K_base+1,
//   elem 2 → K_base+8, elem 3 → K_base+9
// =============================================================================
template <typename SmemT>
__device__ __forceinline__ void precompute_dB_coeff(float coeff[4], SmemT const& smem,
                                                    float total_cumAdt, int prev_k, int lane) {
  int const K_base = (lane % 4) * 2;
  int const k_idx[4] = {K_base, K_base + 1, K_base + 8, K_base + 9};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int const k = k_idx[i];
    coeff[i] = (k < prev_k) ? __expf(total_cumAdt - smem.old_cumAdt[k]) * smem.old_dt_proc[k] : 0.f;
  }
}

// Apply precomputed dB coefficients to frag_B in-place.
// Handles input_t → mma_type conversion + scaling in one pass.
// coeff[i] = 0 encodes both causal mask and zero-fill for k >= prev_k.
// =============================================================================
template <typename mma_type, typename FragB>
__device__ __forceinline__ void compute_dB_scaling(FragB& frag_B, float const coeff[4]) {
  using namespace cute;
  static_assert(size(FragB{}) == 4, "m16n8k16 B fragment must have 4 elements");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float val = toFloat(frag_B(i)) * coeff[i];
    mma_type result(val);
    memcpy(&frag_B(i), &result, 2);
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

  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  int const tid = warp * warpSize + lane;

  // Always bf16 MMA — activations are bf16; f16 state is converted via convert_frag.
  using mma_type = __nv_bfloat16;
  using MmaAtomType = SM80_16x8x16_F32BF16BF16F32_TN;

  // TiledMMA: 128 threads, 4 atoms in M direction → covers [M=64, N=8]
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  float total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [NTOKENS_PAD=16, DIM=64] Swizzle<3,3,3>, transposed view [M=DIM,
  // K=NTOKENS_PAD] ── ldmatrix.trans reads from M-stride-1 smem and transposes → K stride-1 in
  // registers.
  auto layout_A =
      make_swizzled_layout_rc_transpose<NTOKENS_PAD, DIM>();  // shape (DIM, NTOKENS_PAD)
  Tensor smem_A =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.old_x[0][0])), layout_A);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, mma_type>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(
      make_tensor((mma_type*)0x0, make_shape(Int<DIM>{}, Int<NTOKENS_PAD>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == mma_type (bf16) — no conversion needed.

  // ── B operand: old_B [NTOKENS_PAD, DSTATE] Swizzle<3,3,3>, transposed view [N=DSTATE,
  // K=NTOKENS_PAD] ── ldmatrix.trans reads from N-stride-1 smem and transposes → K stride-1 in
  // registers.  Scaling by coeff[k] applied in registers after load.
  auto layout_B =
      make_swizzled_layout_rc_transpose<NTOKENS_PAD, DSTATE>();  // shape (DSTATE, NTOKENS_PAD)
  Tensor smem_B_full =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.old_B[0][0])), layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, mma_type>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: [DIM, DSTATE] swizzled ──
  auto layout_state_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  smem_state_t* state_base = reinterpret_cast<smem_state_t*>(&smem.state[0][0]);

  constexpr int N_TILE = 8;
  constexpr int NUM_N_TILES = DSTATE / N_TILE;

  // Identity tensor for C coordinates (covers full [DIM=64, N_TILE=8] output tile)
  auto id_tile = make_identity_tensor(make_shape(Int<DIM>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);

  // Precompute dB coefficients once — they depend only on K (lane), not on N.
  // Embeds the k < prev_k check: coeff[i] = 0 for k >= prev_k.
  float dB_coeff[4];
  precompute_dB_coeff(dB_coeff, smem, total_cumAdt, prev_k, lane);

  // Process two N-tiles per iteration: interleave LDSM.T + dB_scaling across tiles
  // so PRMT has more latency cover from the other tile's loads.
  static_assert(NUM_N_TILES % 2 == 0, "NUM_N_TILES must be even for 2-wide N-tile loop");

#pragma unroll
  for (int n = 0; n < NUM_N_TILES; n += 2) {
    // ── Load state for both N-tiles ──
    Tensor frag_h_0 = thr_mma.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<DIM>{}, Int<N_TILE>{})));
    Tensor frag_h_1 = thr_mma.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<DIM>{}, Int<N_TILE>{})));

#pragma unroll
    for (int i = 0; i < size(frag_h_0); i += 2) {
      int row = get<0>(id_part(i));
      int col0 = n * N_TILE + get<1>(id_part(i));
      int col1 = (n + 1) * N_TILE + get<1>(id_part(i));
      smem_state_t p0[2], p1[2];
      memcpy(p0, &state_base[layout_state_swz(row, col0)], 4);
      memcpy(p1, &state_base[layout_state_swz(row, col1)], 4);
      frag_h_0(i) = toFloat(p0[0]) * total_decay;
      frag_h_0(i + 1) = toFloat(p0[1]) * total_decay;
      frag_h_1(i) = toFloat(p1[0]) * total_decay;
      frag_h_1(i + 1) = toFloat(p1[1]) * total_decay;
    }

    // ── LDSM.T both B tiles, then scale both — interleaved for latency cover ──
    Tensor smem_B_ntile_0 =
        local_tile(smem_B_full, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD>{}), make_coord(n, _0{}));
    Tensor smem_B_ntile_1 = local_tile(smem_B_full, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD>{}),
                                       make_coord(n + 1, _0{}));

    auto smem_B_s2r_0 = s2r_thr_B.partition_S(smem_B_ntile_0);
    auto smem_B_s2r_1 = s2r_thr_B.partition_S(smem_B_ntile_1);

    Tensor frag_B_0 = thr_mma.partition_fragment_B(
        make_tensor((mma_type*)0x0, make_shape(Int<N_TILE>{}, Int<NTOKENS_PAD>{})));
    Tensor frag_B_1 = make_fragment_like(frag_B_0);
    auto frag_B_view_0 = s2r_thr_B.retile_D(frag_B_0);
    auto frag_B_view_1 = s2r_thr_B.retile_D(frag_B_1);

    cute::copy(s2r_B, smem_B_s2r_0, frag_B_view_0);
    cute::copy(s2r_B, smem_B_s2r_1, frag_B_view_1);

    compute_dB_scaling<mma_type>(frag_B_0, dB_coeff);
    compute_dB_scaling<mma_type>(frag_B_1, dB_coeff);

    // ── Two independent HMMAs ──
    cute::gemm(tiled_mma, frag_h_0, frag_A, frag_B_0, frag_h_0);
    cute::gemm(tiled_mma, frag_h_1, frag_A, frag_B_1, frag_h_1);

    // ── Vectorized state store for both N-tiles ──
#pragma unroll
    for (int i = 0; i < size(frag_h_0); i += 2) {
      int row = get<0>(id_part(i));
      int col0 = n * N_TILE + get<1>(id_part(i));
      int col1 = (n + 1) * N_TILE + get<1>(id_part(i));
      smem_state_t p0[2], p1[2];
      p0[0] = smem_state_t(frag_h_0(i));
      p0[1] = smem_state_t(frag_h_0(i + 1));
      p1[0] = smem_state_t(frag_h_1(i));
      p1[1] = smem_state_t(frag_h_1(i + 1));
      memcpy(&state_base[layout_state_swz(row, col0)], p0, 4);
      memcpy(&state_base[layout_state_swz(row, col1)], p1, 4);
    }
  }
}

// ── CuTe mma.sync output sub-functions ──────────────────────────────────────
// Each operates on a register-resident frag_y accumulator (f32).
// Called from compute_output_cute's N-tile loop.

// Convert fragment elements from src_t to bf16 (mma_type) in-place.
// No-op when src_t == mma_type.  For f16→bf16: reads f16 pair, converts
// via f32 intermediate, writes bf16 pair.  Uses fromFloat2 for efficient
// paired conversion (single cvt.rn.bf16x2.f32 on Ampere+).
template <typename src_t, typename mma_type, typename Frag>
__device__ __forceinline__ void convert_frag(Frag& frag) {
  if constexpr (!std::is_same_v<src_t, mma_type>) {
    static_assert(std::is_same_v<mma_type, __nv_bfloat16>, "mma_type must be bf16");
#pragma unroll
    for (int i = 0; i < cute::size(frag); i += 2) {
      float2 vals = toFloat2(reinterpret_cast<src_t const*>(&frag(i)));
      __nv_bfloat162 packed = fromFloat2(vals);
      memcpy(&frag(i), &packed, 4);
    }
  }
}

// 2b. frag_y += CB_scaled @ x  (matmul 4, single K-tile)
//     CB_scaled A operand loaded from swizzled smem via LDSM (precomputed by warps 0,1).
//     x B operand loaded from smem via ldmatrix.trans.
template <typename input_t, typename mma_type, int N_TILE, int NTOKENS_PAD, typename FragY,
          typename FragCB, typename SmemXTrans, typename S2RBTrans, typename S2RThrBTrans,
          typename ThrMma, typename TiledMma>
__device__ __forceinline__ void add_cb_x_cute(FragY& frag_y, FragCB const& frag_CB,
                                              SmemXTrans const& smem_x_trans,
                                              S2RBTrans const& s2r_B_trans,
                                              S2RThrBTrans const& s2r_thr_B_trans,
                                              ThrMma const& thr_mma, TiledMma const& tiled_mma,
                                              int n) {
  using namespace cute;
  Tensor smem_x_trans_ntile =
      local_tile(smem_x_trans, make_tile(Int<N_TILE>{}, Int<NTOKENS_PAD>{}), make_coord(n, _0{}));
  auto smem_x_trans_s2r = s2r_thr_B_trans.partition_S(smem_x_trans_ntile);
  auto frag_B_x = thr_mma.partition_fragment_B(
      make_tensor((mma_type*)0x0, make_shape(Int<N_TILE>{}, Int<NTOKENS_PAD>{})));
  auto frag_B_x_view = s2r_thr_B_trans.retile_D(frag_B_x);

  cute::copy(s2r_B_trans, smem_x_trans_s2r, frag_B_x_view);
  // x is input_t == mma_type (bf16) — no conversion needed.

  cute::gemm(tiled_mma, frag_y, frag_CB, frag_B_x, frag_y);
}

// 3b. frag_y += D * x[t, d]  (per-thread skip connection via partition_C)
template <typename input_t, int NTOKENS_PAD, int N_TILE, typename FragY, typename SmemX,
          typename ThrMma>
__device__ __forceinline__ void add_D_skip_cute(FragY& frag_y, SmemX const& smem_x,
                                                ThrMma const& thr_mma, float D_val, int n) {
  using namespace cute;
  if (D_val == 0.f) return;
  Tensor smem_x_tile =
      local_tile(smem_x, make_tile(Int<NTOKENS_PAD>{}, Int<N_TILE>{}), make_coord(_0{}, n));
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
template <typename input_t, int NTOKENS_PAD, int N_TILE, typename FragY, typename SmemZ,
          typename ThrMma>
__device__ __forceinline__ void compute_z_gating_cute(FragY& frag_y, SmemZ const& smem_z,
                                                      ThrMma const& thr_mma, void const* z_ptr,
                                                      int n) {
  using namespace cute;
  if (!z_ptr) return;
  Tensor smem_z_tile =
      local_tile(smem_z, make_tile(Int<NTOKENS_PAD>{}, Int<N_TILE>{}), make_coord(_0{}, n));
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

  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  constexpr int K_TILE = 16;
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr int N_TILE = 32;

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, mma_type>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── Swizzled smem views ──
  auto layout_C_swz = make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>();
  Tensor smem_C =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.C[0][0])), layout_C_swz);
  auto layout_state_swz = make_swizzled_layout_rc<DIM, DSTATE>();
  Tensor smem_state = make_tensor(
      make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.state[0][0])), layout_state_swz);

  // ── K-tiled C (A operand) ──
  Tensor smem_C_ktiled =
      local_tile(smem_C, make_tile(Int<NTOKENS_PAD>{}, Int<K_TILE>{}), make_coord(_0{}, _));
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

  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  int const tid = warp * warpSize + lane;

  // Always bf16 MMA — activations are bf16; f16 state converted in add_init_out_cute.
  using mma_type = __nv_bfloat16;
  using MmaAtomType = SM80_16x8x16_F32BF16BF16F32_TN;

  // ── TiledMMA: 128 threads, covers [16, 32] output per step ──
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  // x: swizzled [NTOKENS_PAD, DIM]
  auto layout_x_swz = make_swizzled_layout_rc<NTOKENS_PAD, DIM>();
  Tensor smem_x =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.x[0][0])), layout_x_swz);
  auto layout_x_trans_swz = make_swizzled_layout_rc_transpose<NTOKENS_PAD, DIM>();
  Tensor smem_x_trans = make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(&smem.x[0][0])),
                                    layout_x_trans_swz);

  // z: swizzled [NTOKENS_PAD, DIM]
  auto layout_z_swz = make_swizzled_layout_rc<NTOKENS_PAD, DIM>();
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
  auto layout_cb_swz = composition(Swizzle<3, 3, 3>{},
                                   make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<NTOKENS_PAD>{}),
                                               make_stride(Int<64>{}, _1{})));
  Tensor smem_CB =
      make_tensor(make_smem_ptr(reinterpret_cast<mma_type const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // Decay broadcast: cumAdt[t] → [NTOKENS_PAD, N_TILE] with stride-0 on N
  constexpr int N_TILE = 32;
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output: partition_C for direct register → gmem store ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = (int64_t)batch_idx * params.out_stride_batch + (int64_t)head * DIM;

  // Row predicate for padding: identity tensor partitioned the same way as frag_y.
  // pred(i) is true when the element's row < NTOKENS (skip padding rows).
  auto id_tile = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  auto pred = make_tensor<bool>(shape(id_part));
#pragma unroll
  for (int i = 0; i < size(pred); ++i) {
    pred(i) = get<0>(id_part(i)) < NTOKENS;
  }

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

    // 2b. frag_y += CB_scaled @ x (CB from smem LDSM, x from smem via ldmatrix.trans)
    add_cb_x_cute<input_t, mma_type, N_TILE, NTOKENS_PAD>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // 3b. frag_y += D * x[t, d]
    add_D_skip_cute<input_t, NTOKENS_PAD, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // 4b. frag_y *= z * sigmoid(z)
    compute_z_gating_cute<input_t, NTOKENS_PAD, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Store frag_y directly to gmem (register → gmem, no smem round-trip).
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<N_TILE>{}),
                                             make_stride(params.out_stride_mtp, _1{})));
    auto gOut_part = thr_mma.partition_C(gOut_tile);
    // Vectorized 32-bit store: elements i and i+1 are same-row, consecutive columns
    // in the m16n8k16 partition_C layout, so &gOut_part(i+1) == &gOut_part(i) + 1.
    // Address is 4-byte aligned because MMA column index = (lane%4)*2 → even.
    static_assert(sizeof(input_t) == 2, "vectorized output store requires 2-byte input_t");
#pragma unroll
    for (int i = 0; i < size(frag_y); i += 2) {
      if (pred(i)) {
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
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_x_w = reinterpret_cast<input_t*>(params.old_x);
  int64_t const ox_w_base = cache_slot * params.old_x_stride_cache + head * DIM;

  auto layout_x_swz = make_swizzled_layout_rc<NTOKENS_PAD, DIM>();
  Tensor sX =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.x[0][0])), layout_x_swz);
  Tensor gX = make_tensor(make_gmem_ptr(old_x_w + ox_w_base),
                          make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}),
                                      make_stride(params.old_x_stride_mtp, Int<1>{})));

  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsX = thr_s2g.partition_S(sX);
  auto tSgX = thr_s2g.partition_D(gX);

  if constexpr (NTOKENS == NTOKENS_PAD) {
    copy(s2g, tSsX, tSgX);
  } else {
    // All elements in a thread's partition share the same row (thread layout [16,8]).
    // Check the first element's row coordinate to predicate the entire copy.
    auto cX = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DIM>{}));
    auto tScX = thr_s2g.partition_D(cX);
    if (get<0>(tScX(_0{})) < NTOKENS) {
      copy(s2g, tSsX, tSgX);
    }
  }
}

template <typename input_t, int NTOKENS, int DSTATE, int HEADS_PER_GROUP, typename SmemT>
__device__ __forceinline__ void store_old_B(SmemT& smem, SsuIncrementalParams const& params,
                                            int warp, int lane, int head, int group_idx,
                                            int64_t cache_slot, int buf_write) {
  using namespace cute;
  if (head % HEADS_PER_GROUP != 0) return;
  constexpr int NTOKENS_PAD = SmemT::NTOKENS_PAD;
  int const flat_tid = warp * warpSize + lane;

  auto* __restrict__ old_B_w = reinterpret_cast<input_t*>(params.old_B);
  int64_t const oB_base = cache_slot * params.old_B_stride_cache +
                          buf_write * params.old_B_stride_dbuf + group_idx * DSTATE;

  auto layout_B_swz = make_swizzled_layout_rc<NTOKENS_PAD, DSTATE>();
  Tensor sB =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(&smem.B[0][0])), layout_B_swz);
  Tensor gB = make_tensor(make_gmem_ptr(old_B_w + oB_base),
                          make_layout(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}),
                                      make_stride(params.old_B_stride_mtp, Int<1>{})));

  auto s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, input_t>{},
                             Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
  auto thr_s2g = s2g.get_slice(flat_tid);
  auto tSsB = thr_s2g.partition_S(sB);
  auto tSgB = thr_s2g.partition_D(gB);

  if constexpr (NTOKENS == NTOKENS_PAD) {
    copy(s2g, tSsB, tSgB);
  } else {
    // All elements in a thread's partition share the same row (thread layout [16,8]).
    // For DSTATE=128 there are 2 column repetitions, but both are in the same row.
    auto cB = make_identity_tensor(make_shape(Int<NTOKENS_PAD>{}, Int<DSTATE>{}));
    auto tScB = thr_s2g.partition_D(cB);
    if (get<0>(tScB(_0{})) < NTOKENS) {
      copy(s2g, tSsB, tSgB);
    }
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
  // Phase 0: Load all data into smem
  // ════════════════════════════════════════════════════════════════════════
  load_data<input_t, dt_t, state_t, NTOKENS, DIM, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, batch_idx, head, group_idx, cache_slot, buf_read, A_val,
      dt_bias_val);

  __syncthreads();

  // v10.1: old_B writeback hoisted ahead of Phase 1.  Source (smem.B) is
  // consumed only by Phase 1a CB; the STGs fire-and-forget onto the
  // memory subsystem and complete in parallel with all subsequent compute.
  // No barrier stalls on outstanding STGs (bar.sync has no membar.gl).
  store_old_B<input_t, NTOKENS, DSTATE, HEADS_PER_GROUP>(smem, params, warp, lane, head, group_idx,
                                                         cache_slot, buf_write);

  // Warps 0,1 compute CB_scaled into smem (split N=16 → 2×[16,8]).  No
  // barrier needed — warps 2,3 start replay immediately; warps 0,1 join
  // after CB.  Always bf16 MMA because activations are bf16.
  using mma_type = __nv_bfloat16;
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, mma_type, NTOKENS, DSTATE>(smem, warp, lane);
  }
  // Phase 1b: MMA replay (all 4 warps, independent M-rows)
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
