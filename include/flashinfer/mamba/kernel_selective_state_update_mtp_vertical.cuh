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

// Vertical MTP kernel for selective_state_update.
// Processes all heads in a single ngroups-group with one CTA, amortizing B and C
// loads across all HEADS_PER_GROUP heads.
//
// See .plans/mtp_persistent.md for the full design document.

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

namespace flashinfer::mamba::mtp {

using namespace conversion;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

// =============================================================================
// Constants
// =============================================================================

static constexpr int NUM_COMPUTE_GROUPS = 2;
static constexpr int WARPS_PER_COMPUTE_GROUP = 4;

// Warp index layout:
//   0       = load warp (per-tile state_in + x + dt)
//   1       = z warp (idle — epilogue loads z from gmem)
//   2       = store warp (drain state_out ring -> gmem)
//   3       = BC warp (issues TMA for B and C once at startup, then idle)
//   4..7    = compute group 0
//   8..11   = compute group 1
//   12      = epilogue warp (silu(z) * out -> gmem)
static constexpr int WARP_LOAD_STATE_B_C_DT_A_D = 0;
static constexpr int WARP_Z = 1;
static constexpr int WARP_STORE = 2;
static constexpr int WARP_IDLE = 3;
static constexpr int WARP_COMPUTE_BASE = 4;
static constexpr int WARP_EPILOGUE =
    WARP_COMPUTE_BASE + NUM_COMPUTE_GROUPS * WARPS_PER_COMPUTE_GROUP;  // 12
static constexpr int NUM_WARPS = WARP_EPILOGUE + 1;                    // 13

// =============================================================================
// Shared memory layout (single struct, per-group fields indexed by [g])
// =============================================================================

template <typename input_t, typename state_t, int TOKENS_MTP, int HEADS_PER_GROUP,
          int ROWS_PER_TILE, int DSTATE, int NUM_OUT_STAGES>
struct SharedStorageVertical {
  // ── Shared across all compute groups (loaded once via TMA) ─────────────
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];
  // dt: per-group, per-step, scalar per head (stride(dim)=0 in global tensor → broadcast).
  // Final value: raw dt + dt_bias + optional softplus, written by compute warps before
  // bar_state_in_full.wait(). Visible to all warps after the barrier (release-acquire).
  float dt[NUM_COMPUTE_GROUPS][TOKENS_MTP];

  // ── Per-compute-group data (g = 0..NUM_COMPUTE_GROUPS-1) ───────────────
  alignas(128) state_t state_in[NUM_COMPUTE_GROUPS][ROWS_PER_TILE * DSTATE];
  alignas(128) state_t state_out[NUM_COMPUTE_GROUPS][NUM_OUT_STAGES][ROWS_PER_TILE * DSTATE];

  alignas(alignof(PackedAligned<input_t>)) input_t x[NUM_COMPUTE_GROUPS][TOKENS_MTP][ROWS_PER_TILE];
  float out[NUM_COMPUTE_GROUPS][TOKENS_MTP][ROWS_PER_TILE];

  int current_head[NUM_COMPUTE_GROUPS];
  int current_dim_offset[NUM_COMPUTE_GROUPS];

  // ── Barriers ───────────────────────────────────────────────────────────
  barrier_t bar_BC_full;
  barrier_t bar_state_in_empty[NUM_COMPUTE_GROUPS];
  barrier_t bar_state_in_full[NUM_COMPUTE_GROUPS];
  barrier_t bar_out_empty[NUM_COMPUTE_GROUPS][NUM_OUT_STAGES];
  barrier_t bar_out_full[NUM_COMPUTE_GROUPS][NUM_OUT_STAGES];
  barrier_t bar_out_ready[NUM_COMPUTE_GROUPS];
  barrier_t bar_epilogue_done[NUM_COMPUTE_GROUPS];

  // ── Tile scheduler ─────────────────────────────────────────────────────
  int tile_counter;
  int total_tiles;
  int dim_tiles;
};

// =============================================================================
// role_load_state_x_dt — warp 0: per-tile TMA state_in + async x + scalar dt load
// Lane 0 issues all async copies, scalar dt stores, and calls barrier_arrive_tx;
// other lanes idle.
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, int NTOKENS, int DIM,
          int DSTATE, int ROWS_PER_TILE, int HEADS_PER_GROUP, typename SramT>
__device__ __forceinline__ void role_load_state_x_dt(SramT& sram, int lane,
                                                     SelectiveStateMTPParams const& params,
                                                     int batch, int first_head,
                                                     CUtensorMap const& tensorState) {
  namespace cde = cuda::device::experimental;
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch = state_batch_indices ? (int)state_batch_indices[batch] : batch;

  constexpr int bytesState = ROWS_PER_TILE * DSTATE * (int)sizeof(state_t);
  constexpr int bytesXPerStep = ROWS_PER_TILE * (int)sizeof(input_t);
  constexpr int bytesX = NTOKENS * bytesXPerStep;

  int const dim_tiles = (DIM + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
  int const total_tiles = HEADS_PER_GROUP * dim_tiles;

  for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
    int const g = tile_idx % NUM_COMPUTE_GROUPS;
    int const h_local = tile_idx / dim_tiles;
    int const d_tile = tile_idx % dim_tiles;
    int const head = first_head + h_local;
    int const dim_offset = d_tile * ROWS_PER_TILE;

    // Wait for compute group to release this slot
    sram.bar_state_in_empty[g].wait(sram.bar_state_in_empty[g].arrive());

    // Store tile metadata for compute group
    sram.current_head[g] = h_local;
    sram.current_dim_offset[g] = dim_offset;

    if (lane == 0) {
      // TMA load: state_in[g] <- state[state_batch, head, dim_offset:+ROWS_PER_TILE, 0:DSTATE]
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[g][0], &tensorState, 0,
                                                    dim_offset, head, state_batch,
                                                    sram.bar_state_in_full[g]);

      // Bulk async load: x[g][step] <- x[batch, step, head, dim_offset:+ROWS_PER_TILE]
      for (int step = 0; step < NTOKENS; step++) {
        cuda::device::memcpy_async_tx(&sram.x[g][step][0],
                                      &x_ptr[batch * params.x_stride_batch +
                                             step * params.x_stride_mtp + head * DIM + dim_offset],
                                      cuda::aligned_size_t<16>(bytesXPerStep),
                                      sram.bar_state_in_full[g]);
      }
      auto const _ =
          cuda::device::barrier_arrive_tx(sram.bar_state_in_full[g], 1, bytesState + bytesX);
    }
  }
}

// =============================================================================
// role_load_z — warp 1: per-tile z load, arrive bar_z_full
// =============================================================================

// =============================================================================
// role_store_state — warp 2: drain state_out ring → gmem
// =============================================================================

template <typename state_t, typename stateIndex_t, int NTOKENS, int DIM, int DSTATE,
          int ROWS_PER_TILE, int HEADS_PER_GROUP, int NUM_OUT_STAGES, typename SramT>
__device__ __forceinline__ void role_store_state(SramT& sram, int lane,
                                                 SelectiveStateMTPParams const& params, int batch,
                                                 int first_head, bool is_pad,
                                                 CUtensorMap const& tensorState,
                                                 CUtensorMap const& tensorIntermediateState) {
  namespace cde = cuda::device::experimental;

  auto const* __restrict__ intermediate_states = params.intermediate_states;
  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);

  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;
  int const dim_tiles = (DIM + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
  int const total_tiles = HEADS_PER_GROUP * dim_tiles;

  // Pre-arrive: initial ring slots start empty, allow compute to write first batch
  for (int g_idx = 0; g_idx < NUM_COMPUTE_GROUPS; g_idx++) {
    for (int s = 0; s < NUM_OUT_STAGES; s++) {
      sram.bar_out_empty[g_idx][s].arrive();
    }
  }

  for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
    int const g = tile_idx % NUM_COMPUTE_GROUPS;
    int const h_local = tile_idx / dim_tiles;
    int const d_tile = tile_idx % dim_tiles;
    int const head = first_head + h_local;
    int const dim_offset = d_tile * ROWS_PER_TILE;

    for (int step = 0; step < NTOKENS; step++) {
      int slot = step % NUM_OUT_STAGES;

      // Wait for compute group to fill this slot
      sram.bar_out_full[g][slot].wait(sram.bar_out_full[g][slot].arrive());

      if (!is_pad) {
        bool did_tma_store = false;

        // Write intermediate states via TMA store
        if (intermediate_states) {
          if (lane == 0) {
            cde::cp_async_bulk_tensor_5d_shared_to_global(&tensorIntermediateState, 0, dim_offset,
                                                          head, step, (int)icache_idx,
                                                          &sram.state_out[g][slot][0]);
          }
          did_tma_store = true;
        }

        // Write final state via TMA store (last token step only)
        if (step == NTOKENS - 1 && params.update_state) {
          if (lane == 0) {
            cde::cp_async_bulk_tensor_4d_shared_to_global(
                &tensorState, 0, dim_offset, head, (int)state_batch, &sram.state_out[g][slot][0]);
          }
          did_tma_store = true;
        }

        // commit_group + wait_group_read ensures TMA has read from smem before we release
        // the slot back to the compute warp. No barrier_arrive_tx for stores.
        if (did_tma_store) {
          if (lane == 0) {
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
          }
          __syncwarp();
        }
      }

      // Release slot back to compute group
      sram.bar_out_empty[g][slot].arrive();
    }
  }
}

// =============================================================================
// role_state_update — warps 4-11: SSM update per tile
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t, int NTOKENS,
          int DIM, int DSTATE, int ROWS_PER_TILE, int HEADS_PER_GROUP, int NUM_OUT_STAGES,
          typename SramT>
__device__ __forceinline__ void role_state_update(SramT& sram, int lane, int compute_warp, int g,
                                                  SelectiveStateMTPParams const& params, int batch,
                                                  int first_head) {
  using load_input_t = PackedAligned<input_t>;
  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;
  constexpr int stateRowsPerWarp = ROWS_PER_TILE / WARPS_PER_COMPUTE_GROUP;  // = 4
  constexpr auto stateValuesPerThread = DSTATE / warpSize;
  constexpr auto maxPackedElements = sizeof(uint64_t) / sizeof(input_t);
  constexpr auto packedSramLdInputElements =
      (stateValuesPerThread >= maxPackedElements) ? maxPackedElements : stateValuesPerThread;
  using packed_input_t = PackedAligned<input_t, packedSramLdInputElements>;

  // dim_tiles is compile-time: both DIM and ROWS_PER_TILE are template params.
  // This lets us compute h_local and dim_offset from tile_idx without reading sram.
  constexpr int dim_tiles = (DIM + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
  int const total_tiles = sram.total_tiles;
  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  // Wait for shared B/C/dt/D
  sram.bar_BC_full.wait(sram.bar_BC_full.arrive());

  // Pre-arrive: unblock load warp for the first tile (STP pattern)
  sram.bar_state_in_empty[g].arrive();

  // Process tiles assigned to this group
  for (int tile_idx = g; tile_idx < total_tiles; tile_idx += NUM_COMPUTE_GROUPS) {
    int const h_local = tile_idx / dim_tiles;
    int const dim_offset = (tile_idx % dim_tiles) * ROWS_PER_TILE;

    // Pre-load A and D from gmem before the barrier wait to hide latency.
    // h_local is computed from tile_idx (compile-time dim_tiles), so no sram read needed.
    float const A_val = toFloat(A_ptr[first_head + h_local]);
    float const D_val = D_ptr ? toFloat(D_ptr[first_head + h_local]) : 0.f;

    // Load dt into sram.dt[g] — final value = raw_dt + dt_bias + optional softplus.
    // Distributed across compute warps (warp compute_warp handles steps compute_warp,
    // compute_warp + WARPS_PER_COMPUTE_GROUP, ...). Lane 0 writes; all writes are released
    // through the arrive() below and acquired by all warps after wait().
    {
      float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[first_head + h_local]) : 0.f;
      for (int step = compute_warp; step < NTOKENS; step += WARPS_PER_COMPUTE_GROUP) {
        if (lane == 0) {
          float dt_val = toFloat(dt_ptr[batch * params.dt_stride_batch +
                                        step * params.dt_stride_mtp + (first_head + h_local)]) +
                         dt_bias_val;
          if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
          sram.dt[g][step] = dt_val;
        }
      }
    }

    // Wait for state_in + x to be loaded (TMA + memcpy_async_tx)
    sram.bar_state_in_full[g].wait(sram.bar_state_in_full[g].arrive());

    // Load state for ALL warpRows into registers before the step loop.
    // rState[wr][ii] persists across token steps for each row.
    float rState[stateRowsPerWarp][stateValuesPerThread];
    packed_input_t rB, rC;

    for (int wr = 0; wr < stateRowsPerWarp; wr++) {
      int dd = compute_warp * stateRowsPerWarp + wr;
      if (dim_offset + dd >= DIM) {
        for (int ii = 0; ii < stateValuesPerThread; ii++) rState[wr][ii] = 0.f;
        continue;
      }
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        int i = lane * packed_input_t::count +
                (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                (ii % packed_input_t::count);
        rState[wr][ii] = (i < DSTATE) ? toFloat(sram.state_in[g][dd * DSTATE + i]) : 0.f;
      }
    }

    // Step-major loop: barriers fire once per step (matching store warp)
    for (int step = 0; step < NTOKENS; step++) {
      int slot = step % NUM_OUT_STAGES;
      float dt_value = sram.dt[g][step];  // final: raw + bias + softplus (written before barrier)
      float dA = __expf(A_val * dt_value);

      // Wait for store warp to free this slot
      sram.bar_out_empty[g][slot].wait(sram.bar_out_empty[g][slot].arrive());

      for (int wr = 0; wr < stateRowsPerWarp; wr++) {
        int dd = compute_warp * stateRowsPerWarp + wr;
        if (dim_offset + dd >= DIM) break;

        float x_value = toFloat(sram.x[g][step][dd]);
        float out_value = D_val * x_value * int(lane == 0);

        for (int ii = 0; ii < stateValuesPerThread; ii += packed_input_t::count) {
          int base_i = lane * packed_input_t::count +
                       (ii / packed_input_t::count) * warpSize * packed_input_t::count;
          rB = *reinterpret_cast<packed_input_t const*>(&sram.B[step][base_i]);
          rC = *reinterpret_cast<packed_input_t const*>(&sram.C[step][base_i]);

#pragma unroll
          for (int k = 0; k < packed_input_t::count; k++) {
            float& sv = rState[wr][ii + k];
            float dB = toFloat(rB.val[k]) * dt_value;
            sv = sv * dA + dB * x_value;
            out_value += sv * toFloat(rC.val[k]);
          }
        }

        out_value = warpReduceSum(out_value);

        // Write output to smem immediately
        if (lane == 0) {
          sram.out[g][step][dd] = out_value;
        }

        // Write rState for this row to state_out ring
        for (int ii = 0; ii < stateValuesPerThread; ii++) {
          int i = lane * packed_input_t::count +
                  (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                  (ii % packed_input_t::count);
          if (i < DSTATE) {
            convertAndStore(&sram.state_out[g][slot][dd * DSTATE + i], rState[wr][ii]);
          }
        }
      }  // warpRow loop

      // All rows written for this step — signal store warp
      sram.bar_out_full[g][slot].arrive();
    }  // step loop

    // Signal epilogue: out[] ready
    sram.bar_out_ready[g].arrive();

    // Wait for epilogue to finish before releasing tile slot
    sram.bar_epilogue_done[g].wait(sram.bar_epilogue_done[g].arrive());

    // Release tile slot for next load
    sram.bar_state_in_empty[g].arrive();
  }
}

// =============================================================================
// role_epilogue — warp 12: silu(z) * out -> gmem
// =============================================================================

template <typename input_t, int NTOKENS, int DIM, int ROWS_PER_TILE, int HEADS_PER_GROUP,
          typename SramT>
__device__ __forceinline__ void role_epilogue(SramT& sram, int lane,
                                              SelectiveStateMTPParams const& params, int batch,
                                              int first_head) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  int const dim_tiles = (DIM + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
  int const total_tiles = HEADS_PER_GROUP * dim_tiles;

  for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
    int const g = tile_idx % NUM_COMPUTE_GROUPS;
    int const h_local = tile_idx / dim_tiles;
    int const d_tile = tile_idx % dim_tiles;
    int const head = first_head + h_local;
    int const dim_offset = d_tile * ROWS_PER_TILE;

    // Wait for out to be ready
    sram.bar_out_ready[g].wait(sram.bar_out_ready[g].arrive());

    // Apply silu(z) * out -> gmem (z loaded directly from gmem, no smem needed)
    for (int step = 0; step < NTOKENS; step++) {
      for (int d = lane; d < ROWS_PER_TILE; d += warpSize) {
        if (dim_offset + d < DIM) {
          float out_value = sram.out[g][step][d];
          if (z_ptr) {
            float z_value =
                toFloat(z_ptr[batch * params.z_stride_batch + step * params.z_stride_mtp +
                              head * DIM + dim_offset + d]);
            float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
            out_value *= z_value * sig_z;
          }
          convertAndStore(&output[batch * params.out_stride_batch + step * params.out_stride_mtp +
                                  head * DIM + dim_offset + d],
                          out_value);
        }
      }
    }

    // Signal compute group: epilogue done, tile slot can be reused
    sram.bar_epilogue_done[g].arrive();
  }
}

// =============================================================================
// Kernel entry point
// =============================================================================

// Grid: (batch, ngroups)
// Block: (32, NUM_WARPS)
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int ROWS_PER_TILE,
          int HEADS_PER_GROUP, int NUM_OUT_STAGES>
__global__ void selective_state_update_kernel_vertical_mtp(
    SelectiveStateMTPParams params, __grid_constant__ CUtensorMap const tensorState,
    __grid_constant__ CUtensorMap const tensorB, __grid_constant__ CUtensorMap const tensorC,
    __grid_constant__ CUtensorMap const tensorIntermediateState) {
  extern __shared__ __align__(128) char smem[];
  using sram_t = SharedStorageVertical<input_t, state_t, NTOKENS, HEADS_PER_GROUP, ROWS_PER_TILE,
                                       DSTATE, NUM_OUT_STAGES>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const first_head = group_idx * HEADS_PER_GROUP;
  int const dim_tiles = (DIM + ROWS_PER_TILE - 1) / ROWS_PER_TILE;

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  // ── Init barriers (warp 0, lane 0) ──────────────────────────────────────
  if (warp == 0 && lane == 0) {
    sram.total_tiles = HEADS_PER_GROUP * dim_tiles;
    sram.dim_tiles = dim_tiles;
    sram.tile_counter = 0;

    // Warp counts for barrier participant calculation
    constexpr int numBCLoadWarps = 1;                         // warp 3 (IDLE): issues B/C TMA
    constexpr int numStateLoadWarps = 1;                      // warp 0: load state_in + x
    constexpr int numStoreWarps = 1;                          // warp 2: drain state_out ring
    constexpr int numEpilogueWarps = 1;                       // warp 12: silu(z)*out
    constexpr int numComputeWarps = WARPS_PER_COMPUTE_GROUP;  // 4 per group

    init(&sram.bar_BC_full, (numBCLoadWarps + NUM_COMPUTE_GROUPS * numComputeWarps) * warpSize);

    for (int g = 0; g < NUM_COMPUTE_GROUPS; g++) {
      init(&sram.bar_state_in_empty[g], (numComputeWarps + numStateLoadWarps) * warpSize);
      init(&sram.bar_state_in_full[g], 1 + numComputeWarps * warpSize);
      init(&sram.bar_out_ready[g], (numComputeWarps + numEpilogueWarps) * warpSize);
      init(&sram.bar_epilogue_done[g], (numEpilogueWarps + numComputeWarps) * warpSize);

      for (int s = 0; s < NUM_OUT_STAGES; s++) {
        init(&sram.bar_out_empty[g][s], (numStoreWarps + numComputeWarps) * warpSize);
        init(&sram.bar_out_full[g][s], (numComputeWarps + numStoreWarps) * warpSize);
      }
    }
  }
  __syncthreads();

  // ── Warp role dispatch ──────────────────────────────────────────────────
  if (warp == WARP_LOAD_STATE_B_C_DT_A_D) {
    role_load_state_x_dt<input_t, state_t, stateIndex_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE,
                         HEADS_PER_GROUP>(sram, lane, params, batch, first_head, tensorState);

  } else if (warp == WARP_Z) {
    // Z warp is idle — epilogue loads z directly from gmem.

  } else if (warp == WARP_STORE) {
    role_store_state<state_t, stateIndex_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE, HEADS_PER_GROUP,
                     NUM_OUT_STAGES>(sram, lane, params, batch, first_head, is_pad, tensorState,
                                     tensorIntermediateState);

  } else if (warp == WARP_IDLE) {
    // Issue TMA loads for B and C; stay idle afterward. Only lane 0 acts; barrier_arrive_tx
    // contributes warpSize arrivals on behalf of the whole warp.
    if (lane == 0) {
      namespace cde = cuda::device::experimental;
      constexpr int bytesBC = 2 * NTOKENS * DSTATE * (int)sizeof(input_t);  // B + C combined
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, group_idx, 0, batch,
                                                    sram.bar_BC_full);
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, group_idx, 0, batch,
                                                    sram.bar_BC_full);
      cuda::device::barrier_arrive_tx(sram.bar_BC_full, warpSize, bytesBC);
    }

  } else if (warp >= WARP_COMPUTE_BASE && warp < WARP_EPILOGUE) {
    int const compute_warp = (warp - WARP_COMPUTE_BASE) % WARPS_PER_COMPUTE_GROUP;
    int const g = (warp - WARP_COMPUTE_BASE) / WARPS_PER_COMPUTE_GROUP;
    role_state_update<input_t, state_t, matrixA_t, weight_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE,
                      HEADS_PER_GROUP, NUM_OUT_STAGES>(sram, lane, compute_warp, g, params, batch,
                                                       first_head);

  } else if (warp == WARP_EPILOGUE) {
    role_epilogue<input_t, NTOKENS, DIM, ROWS_PER_TILE, HEADS_PER_GROUP>(sram, lane, params, batch,
                                                                         first_head);
  }
}

}  // namespace flashinfer::mamba::mtp
