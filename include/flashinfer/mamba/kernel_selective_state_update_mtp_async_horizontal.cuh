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

// Async horizontal MTP kernel for selective_state_update.
// Uses cp.async instead of TMA — works on SM80+ (Ampere, Hopper, Blackwell).
// All warps are compute warps (no dedicated TMA warp).
// State is loaded directly into registers from global memory.
// Shared memory B/C rows are padded to avoid bank conflicts for non-power-of-2 DSTATE.
//
// Execution flow:
// 1. All warps cooperatively cp.async B/C/x/dt into smem.
// 2. Each thread loads its state columns from global memory directly into rState[] registers.
// 3. cp_async_wait_group<0>() + __syncthreads() — single sync.
// 4. Step loop: pure register compute + smem reads for B/C/x. No further syncs until epilogue.

#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "ssu_mtp_common.cuh"

namespace flashinfer::mamba::mtp {

using namespace conversion;

// Async horizontal kernel constants.
namespace async_horiz {
static constexpr int LANES_PER_ROW = 8;
static constexpr int ROWS_PER_WARP = warpSize / LANES_PER_ROW;  // 4
}  // namespace async_horiz

// Pad DSTATE to next multiple of 32 banks (128 bytes) to avoid bank conflicts.
template <typename T>
constexpr int padDstate(int dstate) {
  constexpr int alignment = 128;  // 32 banks * 4 bytes/bank
  int row_bytes = dstate * (int)sizeof(T);
  int padded_bytes = (row_bytes + alignment - 1) / alignment * alignment;
  return padded_bytes / (int)sizeof(T);
}

// =============================================================================
// Shared memory layout for async horizontal kernel.
// No state_in buffer — state goes directly to registers.
// =============================================================================

template <typename input_t, int NTOKENS, int DIM_PER_CTA, int DSTATE_PAD>
struct AsyncHorizontalStorage {
  alignas(128) input_t B[NTOKENS][DSTATE_PAD];
  alignas(128) input_t C[NTOKENS][DSTATE_PAD];
  alignas(128) input_t x[NTOKENS][DIM_PER_CTA];
  float dt[NTOKENS];
  float out[NTOKENS][DIM_PER_CTA];
};

// =============================================================================
// Cooperative async load: all warps load B, C, x, dt into smem.
// =============================================================================

template <typename input_t, typename weight_t, bool IS_PAD, int NTOKENS, int DIM, int DSTATE,
          int DSTATE_PAD, int DIM_PER_CTA, int NUM_WARPS, typename SramT>
__device__ __forceinline__ void load_async_horizontal(SramT& sram, int lane, int warp,
                                                      SelectiveStateMTPParams const& params,
                                                      int batch, int head, int kv_group,
                                                      int dim_offset) {
  int const flat_tid = warp * warpSize + lane;
  int const num_threads = NUM_WARPS * warpSize;

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  if constexpr (!IS_PAD) {
    // Load B[step][dstate] → sram.B[step][dstate_pad] (contiguous DSTATE, skip padding)
    auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
    for (int idx = flat_tid; idx < NTOKENS * DSTATE; idx += num_threads) {
      int const step = idx / DSTATE;
      int const col = idx % DSTATE;
      int64_t const src = (int64_t)batch * params.B_stride_batch + step * params.B_stride_mtp +
                          kv_group * DSTATE + col;
      sram.B[step][col] = B_ptr[src];
    }

    // Load C[step][dstate] → sram.C[step][dstate_pad]
    auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
    for (int idx = flat_tid; idx < NTOKENS * DSTATE; idx += num_threads) {
      int const step = idx / DSTATE;
      int const col = idx % DSTATE;
      int64_t const src = (int64_t)batch * params.C_stride_batch + step * params.C_stride_mtp +
                          kv_group * DSTATE + col;
      sram.C[step][col] = C_ptr[src];
    }

    // Load x[step][dim_per_cta] → sram.x[step][dim_per_cta]
    auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
    for (int idx = flat_tid; idx < NTOKENS * DIM_PER_CTA; idx += num_threads) {
      int const step = idx / DIM_PER_CTA;
      int const col = idx % DIM_PER_CTA;
      int64_t const src = (int64_t)batch * params.x_stride_batch + step * params.x_stride_mtp +
                          head * DIM + dim_offset + col;
      sram.x[step][col] = x_ptr[src];
    }

    // Load dt[step] (scalar per step) — only a few values, single thread suffices
    if (flat_tid < NTOKENS) {
      int const step = flat_tid;
      float dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      float dt_val =
          toFloat(dt_ptr[batch * params.dt_stride_batch + step * params.dt_stride_mtp + head]);
      dt_val += dt_bias_val;
      if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
      sram.dt[step] = dt_val;
    }
  }
}

// =============================================================================
// State update: horizontal DSTATE traversal with state in registers.
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, bool IS_PAD, int NTOKENS, int DIM, int DSTATE, int DSTATE_PAD,
          int DIM_PER_CTA, int PHILOX_ROUNDS, int NUM_WARPS, typename SramT>
__device__ __forceinline__ void update_state_async_horizontal(SramT& sram, int lane, int warp,
                                                              SelectiveStateMTPParams const& params,
                                                              int batch, int head, int dim_offset,
                                                              int64_t state_batch) {
  constexpr int lanesPerRow = async_horiz::LANES_PER_ROW;
  constexpr int rowsPerWarp = async_horiz::ROWS_PER_WARP;
  constexpr int ROWS_PER_PASS = NUM_WARPS * rowsPerWarp;
  constexpr int DSTATE_PADDED = nextPow2(DSTATE);
  constexpr int stateValuesPerThread = DSTATE_PADDED / lanesPerRow;

  constexpr int bankSize = sizeof(uint32_t);
  constexpr int stateValuesPerBank = bankSize / sizeof(state_t);
  constexpr int numBanks = 32;
  constexpr int sramReadsPerThreadPerTile = numBanks / lanesPerRow;
  constexpr int elemsPerTileMember = sramReadsPerThreadPerTile * stateValuesPerBank;
  constexpr int elemsPerTile = elemsPerTileMember * lanesPerRow;
  constexpr int numTiles = stateValuesPerThread / elemsPerTileMember;
  using packed_tile_t = PackedAligned<state_t, elemsPerTileMember>;

  static_assert(DSTATE % lanesPerRow == 0, "DSTATE must be divisible by lanesPerRow");
  static_assert(DIM_PER_CTA % ROWS_PER_PASS == 0, "DIM_PER_CTA must be divisible by ROWS_PER_PASS");
  static_assert(elemsPerTileMember % 2 == 0, "elemsPerTileMember must be even for f32x2");
  constexpr int pairsPerTileMember = elemsPerTileMember / 2;

  int const member = lane % lanesPerRow;
  int const group = lane / lanesPerRow;

  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);

  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);
  auto* __restrict__ istate_ptr = reinterpret_cast<state_t*>(params.intermediate_states);

  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;

  // Logical column helpers (same as TMA horizontal kernel)
  auto baseCol = [&](int t, int e) -> int {
    return t * elemsPerTile + member * elemsPerTileMember + e;
  };

  // Output pointers (for epilogue)
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  // Guard: outputLoadSize is only meaningful when DIM_PER_CTA >= warpSize
  constexpr auto outputLoadSize =
      DIM_PER_CTA >= warpSize ? getVectorLoadSizeForFullUtilization<input_t, DIM_PER_CTA>() : 1;
  using load_output_t = PackedAligned<input_t, outputLoadSize>;

  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;
  float const A_val = toFloat(A_ptr[head]);
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  constexpr int numPasses = DIM_PER_CTA / ROWS_PER_PASS;

  for (int pass = 0; pass < numPasses; pass++) {
    int const local_row = pass * ROWS_PER_PASS + warp * rowsPerWarp + group;
    int const dd = dim_offset + local_row;  // global DIM index

    // Load state directly from global memory into registers
    float2 rState[numTiles][pairsPerTileMember];
#pragma unroll
    for (int t = 0; t < numTiles; t++) {
#pragma unroll
      for (int p = 0; p < pairsPerTileMember; p++) {
        int const c0 = baseCol(t, p * 2);
        if (c0 >= DSTATE || IS_PAD) {
          rState[t][p] = {0.f, 0.f};
        } else {
          rState[t][p] = toFloat2(&state_ptr[state_ptr_offset + dd * DSTATE + c0]);
        }
      }
    }

    // Precompute intermediate-state pointer and stride
    int64_t const istate_base_dd = icache_idx * params.intermediate_state_stride_batch +
                                   (int64_t)head * DIM * DSTATE + (int64_t)dd * DSTATE;
    int64_t const istate_step_stride = (int64_t)params.nheads * DIM * DSTATE;
    state_t* __restrict__ istate_dd_ptr = istate_ptr + istate_base_dd;

    // Strength-reduce step-dependent shared memory indexing
    auto const* __restrict__ B_step = &sram.B[0][0];
    auto const* __restrict__ C_step = &sram.C[0][0];
    auto const* __restrict__ x_step = &sram.x[0][0];
    float const* __restrict__ dt_step = &sram.dt[0];
    float* __restrict__ out_step = &sram.out[0][0];

    for (int step = 0; step < NTOKENS; step++) {
      float const dt_value = *dt_step;
      float const dA = __expf(A_val * dt_value);
      float const x_value = toFloat(x_step[local_row]);

      // f32x2 packed recurrence
      float2 out2 = {0.f, 0.f};
      float2 const dA2 = {dA, dA};
      float const dtx_value = dt_value * x_value;
      float2 const dtx2 = {dtx_value, dtx_value};

#pragma unroll
      for (int t = 0; t < numTiles; t++) {
#pragma unroll
        for (int p = 0; p < pairsPerTileMember; p++) {
          int const c0 = baseCol(t, p * 2);
          if (c0 >= DSTATE) continue;
          float2 const B2 = toFloat2(&B_step[c0]);
          float2 const C2 = toFloat2(&C_step[c0]);
          float2 dBx;
          mul_f32x2(dBx, B2, dtx2);
          fma_f32x2(rState[t][p], dA2, rState[t][p], dBx);
          fma_f32x2(out2, rState[t][p], C2, out2);
        }
      }
      float out_value = out2.x + out2.y;

      // Reduce across lanesPerRow adjacent lanes
#pragma unroll
      for (int offset = lanesPerRow / 2; offset >= 1; offset /= 2) {
        out_value += __shfl_down_sync(UINT32_MAX, out_value, offset);
      }

      if (member == 0) {
        out_step[local_row] = out_value + D_val * x_value;
      }

      // Advance step pointers
      B_step += DSTATE_PAD;
      C_step += DSTATE_PAD;
      x_step += DIM_PER_CTA;
      dt_step += 1;
      out_step += DIM_PER_CTA;

      // Write intermediate state
      if (istate_ptr && !IS_PAD) {
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
          int const col0 = baseCol(t, 0);
          if (col0 >= DSTATE) continue;
          packed_tile_t rOut;
          [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
          for (int e = 0; e < elemsPerTileMember; e += 2) {
            float const s0 = rState[t][e / 2].x;
            float const s1 = rState[t][e / 2].y;
            convertAndStoreSRHorizontal<state_t, DSTATE, PHILOX_ROUNDS>(
                rOut.val[e], rOut.val[e + 1], s0, s1, rand_seed, state_ptr_offset, dd, col0, e,
                rand_ints);
          }
          *reinterpret_cast<packed_tile_t*>(&istate_dd_ptr[col0]) = rOut;
        }
        istate_dd_ptr += istate_step_stride;
      }

      // Write final state at last step
      if (step == NTOKENS - 1 && params.update_state && !IS_PAD) {
        auto const state_base =
            state_batch * params.state_stride_batch + head * DIM * DSTATE + dd * DSTATE;
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
          int const col0 = baseCol(t, 0);
          if (col0 >= DSTATE) continue;
          packed_tile_t rOut;
          [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
          for (int e = 0; e < elemsPerTileMember; e += 2) {
            float const s0 = rState[t][e / 2].x;
            float const s1 = rState[t][e / 2].y;
            convertAndStoreSRHorizontal<state_t, DSTATE, PHILOX_ROUNDS>(
                rOut.val[e], rOut.val[e + 1], s0, s1, rand_seed, state_ptr_offset, dd, col0, e,
                rand_ints);
          }
          *reinterpret_cast<packed_tile_t*>(&state_ptr[state_base + col0]) = rOut;
        }
      }
    }  // step loop
  }  // pass loop

  // ── Epilogue: sync all warps, z-gate + vectorized store ───
  __syncthreads();

  if constexpr (DIM_PER_CTA >= warpSize) {
    // Fast path: each lane handles >= 1 element, use vectorized loads/stores
    constexpr int elemsPerThreadEpilogue = DIM_PER_CTA / warpSize;

    for (int step = warp; step < NTOKENS; step += NUM_WARPS) {
      int const out_offset =
          batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM + dim_offset;
      int const z_offset =
          batch * params.z_stride_batch + step * params.z_stride_mtp + head * DIM + dim_offset;

      for (int ii = 0; ii < elemsPerThreadEpilogue; ii += load_output_t::count) {
        int const d = lane * load_output_t::count +
                      (ii / load_output_t::count) * warpSize * load_output_t::count;
        load_output_t packed_out;
        load_output_t packed_z;
        if (z_ptr) {
          packed_z = *reinterpret_cast<load_output_t const*>(&z_ptr[z_offset + d]);
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
        *reinterpret_cast<load_output_t*>(&output[out_offset + d]) = packed_out;
      }
    }
  } else {
    // Narrow path: DIM_PER_CTA < warpSize, only first DIM_PER_CTA lanes participate
    for (int step = warp; step < NTOKENS; step += NUM_WARPS) {
      if (lane < DIM_PER_CTA) {
        int const out_offset = batch * params.out_stride_batch + step * params.out_stride_mtp +
                               head * DIM + dim_offset;
        float out_value = sram.out[step][lane];
        if (z_ptr) {
          int const z_offset =
              batch * params.z_stride_batch + step * params.z_stride_mtp + head * DIM + dim_offset;
          float z_value = toFloat(z_ptr[z_offset + lane]);
          float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
          out_value *= z_value * sig_z;
        }
        convertAndStore(&output[out_offset + lane], out_value);
      }
    }
  }
}

// =============================================================================
// Kernel entry point
// Grid: (batch, nheads, CTAS_PER_HEAD)
// Block: (32, NUM_WARPS)
// =============================================================================

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int PHILOX_ROUNDS, int NUM_WARPS, int CTAS_PER_HEAD>
__global__ void __launch_bounds__(NUM_WARPS * 32)
    selective_state_update_kernel_async_horizontal_mtp(SelectiveStateMTPParams params) {
  constexpr int DSTATE_PAD = padDstate<input_t>(DSTATE);
  constexpr int DIM_PER_CTA = DIM / CTAS_PER_HEAD;
  constexpr int ROWS_PER_PASS = NUM_WARPS * async_horiz::ROWS_PER_WARP;

  static_assert(DIM % CTAS_PER_HEAD == 0, "DIM must be divisible by CTAS_PER_HEAD");
  static_assert(DIM_PER_CTA % ROWS_PER_PASS == 0, "DIM_PER_CTA must be divisible by ROWS_PER_PASS");

  extern __shared__ __align__(128) char smem[];
  using sram_t = AsyncHorizontalStorage<input_t, NTOKENS, DIM_PER_CTA, DSTATE_PAD>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const head = blockIdx.y;
  int const cta_z = blockIdx.z;
  int const dim_offset = cta_z * DIM_PER_CTA;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  int const kv_group = head / HEADS_PER_GROUP;

  auto run = [&]<bool IS_PAD>() {
    // Phase 1: cooperative async load of B/C/x/dt into smem
    load_async_horizontal<input_t, weight_t, IS_PAD, NTOKENS, DIM, DSTATE, DSTATE_PAD, DIM_PER_CTA,
                          NUM_WARPS>(sram, lane, warp, params, batch, head, kv_group, dim_offset);

    // Phase 2: single sync — ensures all smem writes are visible
    __syncthreads();

    // Phase 3: compute (state in registers, B/C/x from smem)
    update_state_async_horizontal<input_t, state_t, matrixA_t, weight_t, stateIndex_t, IS_PAD,
                                  NTOKENS, DIM, DSTATE, DSTATE_PAD, DIM_PER_CTA, PHILOX_ROUNDS,
                                  NUM_WARPS>(sram, lane, warp, params, batch, head, dim_offset,
                                             state_batch);
  };

  if (is_pad)
    run.template operator()<true>();
  else
    run.template operator()<false>();
}

}  // namespace flashinfer::mamba::mtp
