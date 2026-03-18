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

// Horizontal MTP kernel for selective_state_update.
// Split from the vertical kernel for independent development.
//
// Warp layout: 12 compute + 3 TMA = 15 warps.
// Each compute warp does its own epilogue (z-gate + convert + gmem store) after
// finishing all MTP steps. No dedicated epilogue warp — 4 compute warps per group
// process DIM/4 contiguous elements each with vectorized stores.
//
// Key difference from vertical: DSTATE is split across lanesPerRow=8 adjacent
// lanes (horizontal traversal), reducing warp reduction from 32 to 8 lanes.

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cuda/barrier>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"
#include "ssu_mtp_common.cuh"

namespace flashinfer::mamba::mtp {

using namespace conversion;

// Horizontal kernel constants — may diverge from vertical in future.
namespace horiz {
static constexpr int NUM_COMPUTE_GROUPS = 3;
static constexpr int NUM_COMPUTE_WARPS_PER_GROUP = 4;
static constexpr int NUM_WARPS = 15;  // 12 compute + 3 TMA (no dedicated epilogue warp)
static constexpr int LANES_PER_ROW = 8;
static constexpr int ROWS_PER_WARP = 32 / LANES_PER_ROW;  // 4
// Total rows produced per pass across all compute warps in a group:
static constexpr int ROWS_PER_PASS = NUM_COMPUTE_WARPS_PER_GROUP * ROWS_PER_WARP;  // 16
}  // namespace horiz

// Horizontal warp role dispatch (different from vertical — no epilogue warp).
enum class HorizWarpRole { kCompute, kTMALoad };

__device__ __forceinline__ HorizWarpRole get_horiz_warp_role(int warp) {
  if (warp < 12) return HorizWarpRole::kCompute;
  return HorizWarpRole::kTMALoad;
}

// =============================================================================
// Shared memory layout (horizontal)
// =============================================================================

// Per-group shared memory. Full out[NTOKENS][DIM] buffer — compute warps do
// their own epilogue after all passes complete.
template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES>
struct GroupStorageHorizontal {
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];
  float dt[TOKENS_MTP];
  alignas(128) state_t state_in[NUM_IN_STAGES][DIM * DSTATE];
  alignas(128) input_t x[NUM_IN_STAGES][TOKENS_MTP][DIM];
  float out[TOKENS_MTP][DIM];

  barrier_t bar_BC_full;
  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
  barrier_t bar_out_ready;  // sync compute warps before epilogue
};

template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES>
struct SharedStorageHorizontal {
  GroupStorageHorizontal<input_t, state_t, TOKENS_MTP, DIM, DSTATE, NUM_IN_STAGES>
      group[horiz::NUM_COMPUTE_GROUPS];
};

// =============================================================================
// role_load_horizontal — one TMA warp per group: loads BC + state_in + x.
// TMA warps exit after loads complete (no epilogue duty).
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, int NTOKENS, int DIM,
          int DSTATE, int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_load_horizontal(
    SramT& sram, int lane, SelectiveStateMTPParams const& params, int batch, int head, int kv_group,
    CUtensorMap const& tensorState, CUtensorMap const& tensorX, CUtensorMap const& tensorB,
    CUtensorMap const& tensorC) {
  namespace cde = cuda::device::experimental;
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch = state_batch_indices ? (int)state_batch_indices[batch] : batch;

  // ── Load B and C ──────────────────────────────────────────────────────
  if (lane == 0) {
    constexpr int bytesBC = 2 * NTOKENS * DSTATE * (int)sizeof(input_t);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, kv_group, 0, batch,
                                                  sram.bar_BC_full);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, kv_group, 0, batch,
                                                  sram.bar_BC_full);
    cuda::device::barrier_arrive_tx(sram.bar_BC_full, warpSize, bytesBC);
  }

  // ── Load state_in + x ────────────────────────────────────────────────
  constexpr int bytesState = DIM * DSTATE * (int)sizeof(state_t);
  constexpr int bytesX = NTOKENS * DIM * (int)sizeof(input_t);
  constexpr int in_slot = 0;

  // Wait for compute to release this in-stage
  sram.bar_state_in_empty[in_slot].wait(sram.bar_state_in_empty[in_slot].arrive());

  if (lane == 0) {
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[in_slot][0], &tensorState, 0, 0,
                                                  head, state_batch,
                                                  sram.bar_state_in_full[in_slot]);

    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.x[in_slot][0][0], &tensorX, 0, head, 0,
                                                  batch, sram.bar_state_in_full[in_slot]);

    auto const _ =
        cuda::device::barrier_arrive_tx(sram.bar_state_in_full[in_slot], 1, bytesState + bytesX);
  }
}

// =============================================================================
// role_update_state_horizontal — horizontal DSTATE traversal + inline epilogue.
//
// lanesPerRow=8 adjacent lanes cooperate on one DIM row, splitting DSTATE.
// After all passes complete, compute warps sync via bar_out_ready and each
// warp processes DIM/NUM_COMPUTE_WARPS_PER_GROUP contiguous output elements
// per step with vectorized stores.
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int PHILOX_ROUNDS,
          int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_update_state_horizontal(SramT& sram, int lane,
                                                             int compute_warp,
                                                             SelectiveStateMTPParams const& params,
                                                             int batch, int head, bool is_pad) {
  constexpr int lanesPerRow = horiz::LANES_PER_ROW;
  constexpr int rowsPerWarp = horiz::ROWS_PER_WARP;
  constexpr int totalRowsPerWarp = DIM / horiz::NUM_COMPUTE_WARPS_PER_GROUP;
  constexpr int numPasses = totalRowsPerWarp / rowsPerWarp;
  constexpr int stateValuesPerThread = DSTATE / lanesPerRow;
  static_assert(DSTATE % lanesPerRow == 0, "DSTATE must be divisible by lanesPerRow");
  static_assert(totalRowsPerWarp % rowsPerWarp == 0,
                "DIM/NUM_COMPUTE_WARPS_PER_GROUP must be divisible by rowsPerWarp");

  constexpr int bankSize = sizeof(uint32_t);
  constexpr int stateValuesPerBank = bankSize / sizeof(state_t);
  constexpr int numBanks = 32;
  constexpr int sramReadsPerThreadPerTile = numBanks / lanesPerRow;                   // 8
  constexpr int elemsPerTileMember = sramReadsPerThreadPerTile * stateValuesPerBank;  // 16
  constexpr int elemsPerTile = elemsPerTileMember * lanesPerRow;                      // 64
  constexpr int numTiles = stateValuesPerThread / elemsPerTileMember;                 // 2
  constexpr int sizeof_dtype = sizeof(state_t);
  constexpr int cycle_length = DSTATE * sizeof_dtype;  // row stride in bytes
  using packed_tile_t = PackedAligned<state_t, elemsPerTileMember>;

  int const member = lane % lanesPerRow;  // which position along DSTATE (0..7)
  int const group = lane / lanesPerRow;   // which row within current 4 active rows

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);
  auto* __restrict__ istate_ptr = reinterpret_cast<state_t*>(params.intermediate_states);

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);

  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;
  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;

  float const A_val = toFloat(A_ptr[head]);
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;

  // Pre-arrive: unblock load warp for state_in
  for (int s = 0; s < NUM_IN_STAGES; s++) {
    sram.bar_state_in_empty[s].arrive();
  }

  // Wait for B/C to be loaded
  sram.bar_BC_full.wait(sram.bar_BC_full.arrive());

  constexpr int in_slot = 0;

  // Load dt values into smem — distributed across compute warps
  {
    for (int step = compute_warp; step < NTOKENS; step += horiz::NUM_COMPUTE_WARPS_PER_GROUP) {
      if (lane == 0) {
        float dt_val =
            toFloat(dt_ptr[batch * params.dt_stride_batch + step * params.dt_stride_mtp + head]) +
            dt_bias_val;
        if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
        sram.dt[step] = dt_val;
      }
    }
  }

  // Wait for state_in + x to be loaded
  sram.bar_state_in_full[in_slot].wait(sram.bar_state_in_full[in_slot].arrive());

  // ═══════════════════════════════════════════════════════════════════════
  // Compute phase: update state + write out[step][dd] for all passes
  // ═══════════════════════════════════════════════════════════════════════

  for (int pass = 0; pass < numPasses; pass++) {
    int const dd = compute_warp * totalRowsPerWarp + pass * rowsPerWarp + group;

    // XOR swizzle: given logical column c, return physical smem column for
    // bank-conflict-free access at row dd.  Mask dd to tile's bank range so
    // the permutation stays within each tile (low 3 bits = group index).
    int const dd_eff = dd & (sramReadsPerThreadPerTile - 1);
    auto smem_col = [&](int c) -> int {
      int const bank = c / stateValuesPerBank;
      int const intra = c % stateValuesPerBank;
      return (bank ^ dd_eff) * stateValuesPerBank + intra;
    };

    // Logical base column for tile t, element e
    auto baseCol = [&](int t, int e) -> int {
      return t * elemsPerTile + member * elemsPerTileMember + e;
    };

    // Load state — simple linear load (no swizzle), matching vertical kernel style.
    // TODO: re-add bank-conflict-free swizzle on the *source* address side
    float rState[numTiles][elemsPerTileMember];
#pragma unroll
    for (int t = 0; t < numTiles; t++) {
#pragma unroll
      for (int e = 0; e < elemsPerTileMember; e++) {
        int const c = baseCol(t, e);
        rState[t][e] = toFloat(sram.state_in[in_slot][dd * DSTATE + c]);
      }
    }

    for (int step = 0; step < NTOKENS; step++) {
      float const dt_value = sram.dt[step];
      float const dA = __expf(A_val * dt_value);
      float const x_value = toFloat(sram.x[in_slot][step][dd]);

      float out_value = 0.f;

#pragma unroll
      for (int t = 0; t < numTiles; t++) {
#pragma unroll
        for (int e = 0; e < elemsPerTileMember; e++) {
          int const c = baseCol(t, e);
          float dB = toFloat(sram.B[step][c]) * dt_value;
          rState[t][e] = rState[t][e] * dA + dB * x_value;
          out_value += rState[t][e] * toFloat(sram.C[step][c]);
        }
      }

      // Reduce across lanesPerRow adjacent lanes
#pragma unroll
      for (int offset = lanesPerRow / 2; offset >= 1; offset /= 2) {
        out_value += __shfl_down_sync(UINT32_MAX, out_value, offset);
      }

      if (member == 0) {
        sram.out[step][dd] = out_value + D_val * x_value;
      }

      // Write intermediate state — vectorized packed_tile_t at logical baseCol (always aligned)
      if (istate_ptr && !is_pad) {
        auto const istate_base = icache_idx * params.intermediate_state_stride_batch +
                                 step * params.nheads * DIM * DSTATE + head * DIM * DSTATE +
                                 dd * DSTATE;
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
          packed_tile_t rOut;
          int const col0 = baseCol(t, 0);
#pragma unroll
          for (int e = 0; e < elemsPerTileMember; e += stateValuesPerBank) {
            if constexpr (PHILOX_ROUNDS > 0) {
              [[maybe_unused]] uint32_t rand_ints[4];
              if (e % 4 == 0)
                philox_randint4x<PHILOX_ROUNDS>(
                    rand_seed, state_ptr_offset + dd * DSTATE + col0 + e, rand_ints[0],
                    rand_ints[1], rand_ints[2], rand_ints[3]);
              uint32_t packed =
                  cvt_rs_f16x2_f32(rState[t][e], rState[t][e + 1], rand_ints[e / 2 % 2]);
              rOut.val[e] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
              rOut.val[e + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
            } else {
              convertAndStore(&rOut.val[e], rState[t][e]);
              convertAndStore(&rOut.val[e + 1], rState[t][e + 1]);
            }
          }
          *reinterpret_cast<packed_tile_t*>(&istate_ptr[istate_base + col0]) = rOut;
        }
      }

      // Write final state directly to gmem at last step
      if (step == NTOKENS - 1 && params.update_state && !is_pad) {
        auto const state_base =
            state_batch * params.state_stride_batch + head * DIM * DSTATE + dd * DSTATE;
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
          packed_tile_t rOut;
          int const col0 = baseCol(t, 0);
#pragma unroll
          for (int e = 0; e < elemsPerTileMember; e += stateValuesPerBank) {
            if constexpr (PHILOX_ROUNDS > 0) {
              [[maybe_unused]] uint32_t rand_ints[4];
              if (e % 4 == 0)
                philox_randint4x<PHILOX_ROUNDS>(
                    rand_seed, state_ptr_offset + dd * DSTATE + col0 + e, rand_ints[0],
                    rand_ints[1], rand_ints[2], rand_ints[3]);
              uint32_t packed =
                  cvt_rs_f16x2_f32(rState[t][e], rState[t][e + 1], rand_ints[e / 2 % 2]);
              rOut.val[e] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
              rOut.val[e + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
            } else {
              convertAndStore(&rOut.val[e], rState[t][e]);
              convertAndStore(&rOut.val[e + 1], rState[t][e + 1]);
            }
          }
          *reinterpret_cast<packed_tile_t*>(&state_ptr[state_base + col0]) = rOut;
        }
      }
    }  // step loop
  }  // pass loop

  // ═══════════════════════════════════════════════════════════════════════
  // Epilogue phase: sync all 4 compute warps, then each warp processes
  // DIM/4 contiguous output elements per step with vectorized stores.
  // ═══════════════════════════════════════════════════════════════════════

  // Sync: all 4 warps must finish writing out[] before any warp reads it
  sram.bar_out_ready.wait(sram.bar_out_ready.arrive());

  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);

  constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
  using load_output_t = PackedAligned<input_t, outputLoadSize>;
  constexpr int elemsPerThread = DIM / warpSize;

  // Each warp handles full DIM for its own steps (striped across warps)
  for (int step = compute_warp; step < NTOKENS; step += horiz::NUM_COMPUTE_WARPS_PER_GROUP) {
    int const base_offset =
        batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM;

    for (int ii = 0; ii < elemsPerThread; ii += load_output_t::count) {
      int const d = lane * load_output_t::count +
                    (ii / load_output_t::count) * warpSize * load_output_t::count;
      load_output_t packed_out;
      load_output_t packed_z;
      if (z_ptr) {
        packed_z = *reinterpret_cast<load_output_t const*>(&z_ptr[base_offset + d]);
      }
#pragma unroll
      for (int k = 0; k < load_output_t::count; k++) {
        float out_value = sram.out[step][d + k];
        if (z_ptr) {
          float z_value = toFloat(packed_z.val[k]);
          float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
          out_value *= z_value * sig_z;
        }
        convertAndStore(&packed_out.val[k], out_value);
      }
      *reinterpret_cast<load_output_t*>(&output[base_offset + d]) = packed_out;
    }
  }
}

// =============================================================================
// Kernel entry point
// =============================================================================

// Grid: (batch, ceil(nheads / NUM_COMPUTE_GROUPS))
// Block: (32, NUM_WARPS)  where NUM_WARPS = 15
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int PHILOX_ROUNDS, int NUM_IN_STAGES>
__global__ void __launch_bounds__(horiz::NUM_WARPS * 32, 2)
    selective_state_update_kernel_horizontal_mtp(SelectiveStateMTPParams params,
                                                 __grid_constant__ CUtensorMap const tensorState,
                                                 __grid_constant__ CUtensorMap const tensorB,
                                                 __grid_constant__ CUtensorMap const tensorC,
                                                 __grid_constant__ CUtensorMap const tensorX) {
  extern __shared__ __align__(128) char smem[];
  using sram_t = SharedStorageHorizontal<input_t, state_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const chunk_idx = blockIdx.y;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const total_heads = params.nheads;

  // Each warp computes only its own group's head index — no arrays, no spills.
  int const group = warp < 12 ? warp / horiz::NUM_COMPUTE_WARPS_PER_GROUP : warp - 12;
  int const head = chunk_idx * horiz::NUM_COMPUTE_GROUPS + group;
  int const num_active_groups =
      min(horiz::NUM_COMPUTE_GROUPS, total_heads - chunk_idx * horiz::NUM_COMPUTE_GROUPS);

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  // ── Init barriers (warp 0, lane 0) ──────────────────────────────────────
  if (warp == 0 && lane == 0) {
    for (int g = 0; g < num_active_groups; g++) {
      auto& gsram = sram.group[g];
      // bar_BC_full: 1 TMA warp + 4 compute warps
      init(&gsram.bar_BC_full, (1 + horiz::NUM_COMPUTE_WARPS_PER_GROUP) * warpSize);
      for (int s = 0; s < NUM_IN_STAGES; s++) {
        // bar_state_in_empty: 4 compute warps + 1 TMA warp
        init(&gsram.bar_state_in_empty[s], (horiz::NUM_COMPUTE_WARPS_PER_GROUP + 1) * warpSize);
        // bar_state_in_full: 1 TMA (arrive_tx) + 4 compute warps (wait+arrive)
        init(&gsram.bar_state_in_full[s], 1 + horiz::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
      }
      // bar_out_ready: 4 compute warps only (no TMA/epilogue participation)
      init(&gsram.bar_out_ready, horiz::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
    }
  }
  __syncthreads();

  // ── Warp role dispatch ─────────────────────────────────────────────────
  auto const role = get_horiz_warp_role(warp);

  if (role == HorizWarpRole::kCompute) {
    int const compute_warp = warp % horiz::NUM_COMPUTE_WARPS_PER_GROUP;
    if (group < num_active_groups) {
      role_update_state_horizontal<input_t, state_t, matrixA_t, weight_t, stateIndex_t, NTOKENS,
                                   DIM, DSTATE, PHILOX_ROUNDS, NUM_IN_STAGES>(
          sram.group[group], lane, compute_warp, params, batch, head, is_pad);
    }

  } else /* role == HorizWarpRole::kTMALoad */ {
    if (group < num_active_groups) {
      int const kv_group = head / HEADS_PER_GROUP;
      role_load_horizontal<input_t, state_t, stateIndex_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES>(
          sram.group[group], lane, params, batch, head, kv_group, tensorState, tensorX, tensorB,
          tensorC);
    }
  }
}

}  // namespace flashinfer::mamba::mtp
