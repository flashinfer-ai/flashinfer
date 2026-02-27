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
//   0       = load warp (B, C, dt, A, D, then per-tile state_in + x)
//   1       = z warp (per-tile z load)
//   2       = store warp (drain state_out ring -> gmem)
//   3       = idle (completes TMA warp group of 4)
//   4..7    = compute group 0
//   8..11   = compute group 1
//   12      = epilogue warp (silu(z) * out -> gmem)
static constexpr int WARP_LOAD = 0;
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
  // ── Shared across all compute groups (loaded once) ──────────────────────
  alignas(alignof(PackedAligned<input_t>)) input_t B[TOKENS_MTP][DSTATE];
  alignas(alignof(PackedAligned<input_t>)) input_t C[TOKENS_MTP][DSTATE];
  float dt[TOKENS_MTP][HEADS_PER_GROUP];
  float A[HEADS_PER_GROUP];
  float D[HEADS_PER_GROUP];

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
// role_load_BCdtAD — warps 0-3 cooperatively load shared B, C, dt, A, D
// =============================================================================

template <typename input_t, typename weight_t, typename matrixA_t, int NTOKENS, int DSTATE,
          int HEADS_PER_GROUP, typename SramT>
__device__ __forceinline__ void role_load_BCdtAD(SramT& sram, int lane, int warp,
                                                 SelectiveStateMTPParams const& params, int batch,
                                                 int group_idx, int first_head) {
  using load_input_t = PackedAligned<input_t>;
  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);

  // warp 0: load B
  if (warp == 0) {
    for (int step = 0; step < NTOKENS; step++) {
      for (int i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        *reinterpret_cast<load_input_t*>(&sram.B[step][i]) = *reinterpret_cast<load_input_t const*>(
            &B_ptr[batch * params.B_stride_batch + step * params.B_stride_mtp + group_idx * DSTATE +
                   i]);
      }
    }
  }
  // warp 1: load C
  else if (warp == 1) {
    for (int step = 0; step < NTOKENS; step++) {
      for (int i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        *reinterpret_cast<load_input_t*>(&sram.C[step][i]) = *reinterpret_cast<load_input_t const*>(
            &C_ptr[batch * params.C_stride_batch + step * params.C_stride_mtp + group_idx * DSTATE +
                   i]);
      }
    }
  }
  // warp 2: load dt (with dt_bias + softplus)
  else if (warp == 2) {
    for (int step = 0; step < NTOKENS; step++) {
      for (int h = lane; h < HEADS_PER_GROUP; h += warpSize) {
        float val = toFloat(
            dt_ptr[batch * params.dt_stride_batch + step * params.dt_stride_mtp + first_head + h]);
        if (dt_bias) val += toFloat(dt_bias[first_head + h]);
        if (params.dt_softplus) val = thresholded_softplus(val);
        sram.dt[step][h] = val;
      }
    }
  }
  // warp 3: load A, D
  else if (warp == 3) {
    for (int h = lane; h < HEADS_PER_GROUP; h += warpSize) {
      sram.A[h] = toFloat(A_ptr[first_head + h]);
      sram.D[h] = D_ptr ? toFloat(D_ptr[first_head + h]) : 0.f;
    }
  }

  // All 4 load warps arrive; compute warps arrive in role_state_update.
  sram.bar_BC_full.arrive();
}

// =============================================================================
// role_load_state_x — warp 0: per-tile state_in + x load, arrive bar_state_in_full
// Called by lane 0 only (single-thread producer for barriers).
// =============================================================================

template <typename input_t, typename state_t, int NTOKENS, int DIM, int DSTATE, int ROWS_PER_TILE,
          int HEADS_PER_GROUP, typename SramT>
__device__ __forceinline__ void role_load_state_x(SramT& sram, int lane,
                                                  SelectiveStateMTPParams const& params, int batch,
                                                  int first_head, bool is_pad) {
  using load_input_t = PackedAligned<input_t>;
  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);

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

    // Load state_in: state[state_batch, head, dim_offset:dim_offset+ROWS_PER_TILE, 0:DSTATE]
    auto const state_batch_idx =
        params.state_batch_indices
            ? reinterpret_cast<int64_t const*>(params.state_batch_indices)[batch]
            : (int64_t)batch;
    state_t* state_head =
        &state_ptr[state_batch_idx * params.state_stride_batch + head * DIM * DSTATE];

    for (int row = 0; row < ROWS_PER_TILE; row++) {
      int d = dim_offset + row;
      if (d < DIM && !is_pad) {
        for (int i = lane * load_state_t::count; i < DSTATE; i += warpSize * load_state_t::count) {
          *reinterpret_cast<load_state_t*>(&sram.state_in[g][row * DSTATE + i]) =
              *reinterpret_cast<load_state_t const*>(&state_head[d * DSTATE + i]);
        }
      } else {
        for (int i = lane * load_state_t::count; i < DSTATE; i += warpSize * load_state_t::count) {
          *reinterpret_cast<load_state_t*>(&sram.state_in[g][row * DSTATE + i]) =
              make_zeros<load_state_t>();
        }
      }
    }

    // Load x: x[batch, step, head, dim_offset:dim_offset+ROWS_PER_TILE]
    for (int step = 0; step < NTOKENS; step++) {
      for (int d = lane * load_input_t::count; d < ROWS_PER_TILE;
           d += warpSize * load_input_t::count) {
        if (dim_offset + d < DIM) {
          *reinterpret_cast<load_input_t*>(&sram.x[g][step][d]) =
              *reinterpret_cast<load_input_t const*>(
                  &x_ptr[batch * params.x_stride_batch + step * params.x_stride_mtp + head * DIM +
                         dim_offset + d]);
        }
      }
    }

    // Signal: state_in + x ready
    sram.bar_state_in_full[g].arrive();
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
                                                 int first_head, bool is_pad) {
  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);
  auto* __restrict__ intermediate_states = reinterpret_cast<state_t*>(params.intermediate_states);
  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);

  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;
  int const nheads = params.nheads;
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
        // Write intermediate states
        if (intermediate_states) {
          for (int row = 0; row < ROWS_PER_TILE; row++) {
            int d = dim_offset + row;
            if (d < DIM) {
              for (int i = lane * load_state_t::count; i < DSTATE;
                   i += warpSize * load_state_t::count) {
                *reinterpret_cast<load_state_t*>(
                    &intermediate_states[icache_idx * params.intermediate_state_stride_batch +
                                         step * nheads * DIM * DSTATE + head * DIM * DSTATE +
                                         d * DSTATE + i]) =
                    *reinterpret_cast<load_state_t const*>(
                        &sram.state_out[g][slot][row * DSTATE + i]);
              }
            }
          }
        }

        // Write final state (last token step, if update_state)
        if (step == NTOKENS - 1 && params.update_state) {
          state_t* state_head =
              &state_ptr[state_batch * params.state_stride_batch + head * DIM * DSTATE];
          for (int row = 0; row < ROWS_PER_TILE; row++) {
            int d = dim_offset + row;
            if (d < DIM) {
              for (int i = lane * load_state_t::count; i < DSTATE;
                   i += warpSize * load_state_t::count) {
                *reinterpret_cast<load_state_t*>(&state_head[d * DSTATE + i]) =
                    *reinterpret_cast<load_state_t const*>(
                        &sram.state_out[g][slot][row * DSTATE + i]);
              }
            }
          }
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

template <typename input_t, typename state_t, int NTOKENS, int DIM, int DSTATE, int ROWS_PER_TILE,
          int NUM_OUT_STAGES, typename SramT>
__device__ __forceinline__ void role_state_update(SramT& sram, int lane, int compute_warp, int g) {
  using load_input_t = PackedAligned<input_t>;
  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;
  constexpr int stateRowsPerWarp = ROWS_PER_TILE / WARPS_PER_COMPUTE_GROUP;  // = 4
  constexpr auto stateValuesPerThread = DSTATE / warpSize;
  constexpr auto maxPackedElements = sizeof(uint64_t) / sizeof(input_t);
  constexpr auto packedSramLdInputElements =
      (stateValuesPerThread >= maxPackedElements) ? maxPackedElements : stateValuesPerThread;
  using packed_input_t = PackedAligned<input_t, packedSramLdInputElements>;

  int const dim_tiles = sram.dim_tiles;
  int const total_tiles = sram.total_tiles;

  // Wait for shared B/C/dt/A/D
  sram.bar_BC_full.wait(sram.bar_BC_full.arrive());

  // Pre-arrive: unblock load warp for the first tile (STP pattern)
  sram.bar_state_in_empty[g].arrive();

  // Process tiles assigned to this group
  for (int tile_idx = g; tile_idx < total_tiles; tile_idx += NUM_COMPUTE_GROUPS) {
    // Wait for state_in + x to be loaded
    sram.bar_state_in_full[g].wait(sram.bar_state_in_full[g].arrive());

    int const h_local = sram.current_head[g];
    int const dim_offset = sram.current_dim_offset[g];

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
      float dt_value = sram.dt[step][h_local];
      float dA = __expf(sram.A[h_local] * dt_value);

      // Wait for store warp to free this slot
      sram.bar_out_empty[g][slot].wait(sram.bar_out_empty[g][slot].arrive());

      for (int wr = 0; wr < stateRowsPerWarp; wr++) {
        int dd = compute_warp * stateRowsPerWarp + wr;
        if (dim_offset + dd >= DIM) break;

        float x_value = toFloat(sram.x[g][step][dd]);
        float out_value = sram.D[h_local] * x_value * int(lane == 0);

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
__global__ void selective_state_update_kernel_vertical_mtp(SelectiveStateMTPParams params) {
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
    constexpr int numBCLoadWarps = 4;                         // warps 0-3: load B, C, dt, A, D
    constexpr int numStateLoadWarps = 1;                      // warp 0: load state_in + x
    constexpr int numStoreWarps = 1;                          // warp 2: drain state_out ring
    constexpr int numEpilogueWarps = 1;                       // warp 12: silu(z)*out
    constexpr int numComputeWarps = WARPS_PER_COMPUTE_GROUP;  // 4 per group

    init(&sram.bar_BC_full, (numBCLoadWarps + NUM_COMPUTE_GROUPS * numComputeWarps) * warpSize);

    for (int g = 0; g < NUM_COMPUTE_GROUPS; g++) {
      init(&sram.bar_state_in_empty[g], (numComputeWarps + numStateLoadWarps) * warpSize);
      init(&sram.bar_state_in_full[g], (numStateLoadWarps + numComputeWarps) * warpSize);
      init(&sram.bar_out_ready[g], (numComputeWarps + numEpilogueWarps) * warpSize);
      init(&sram.bar_epilogue_done[g], (numEpilogueWarps + numComputeWarps) * warpSize);

      for (int s = 0; s < NUM_OUT_STAGES; s++) {
        init(&sram.bar_out_empty[g][s], (numStoreWarps + numComputeWarps) * warpSize);
        init(&sram.bar_out_full[g][s], (numComputeWarps + numStoreWarps) * warpSize);
      }
    }
  }
  __syncthreads();

  // ── Phase 1: Load shared B, C, dt, A, D (warps 0-3) ────────────────────
  if (warp < 4) {
    role_load_BCdtAD<input_t, weight_t, matrixA_t, NTOKENS, DSTATE, HEADS_PER_GROUP>(
        sram, lane, warp, params, batch, group_idx, first_head);
  }

  // ── Phase 2: Warp role dispatch ─────────────────────────────────────────
  if (warp == WARP_LOAD) {
    // Compute warps pre-arrive on bar_state_in_empty after their bar_BC_full wait,
    // so the first tile load won't deadlock.
    role_load_state_x<input_t, state_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE, HEADS_PER_GROUP>(
        sram, lane, params, batch, first_head, is_pad);

  } else if (warp == WARP_Z) {
    // Z warp is idle — epilogue loads z directly from gmem.

  } else if (warp == WARP_STORE) {
    role_store_state<state_t, stateIndex_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE, HEADS_PER_GROUP,
                     NUM_OUT_STAGES>(sram, lane, params, batch, first_head, is_pad);

  } else if (warp == WARP_IDLE) {
    // Already arrived on bar_BC_full above. Nothing else to do.

  } else if (warp >= WARP_COMPUTE_BASE && warp < WARP_EPILOGUE) {
    int const compute_warp = (warp - WARP_COMPUTE_BASE) % WARPS_PER_COMPUTE_GROUP;
    int const g = (warp - WARP_COMPUTE_BASE) / WARPS_PER_COMPUTE_GROUP;
    role_state_update<input_t, state_t, NTOKENS, DIM, DSTATE, ROWS_PER_TILE, NUM_OUT_STAGES>(
        sram, lane, compute_warp, g);

  } else if (warp == WARP_EPILOGUE) {
    role_epilogue<input_t, NTOKENS, DIM, ROWS_PER_TILE, HEADS_PER_GROUP>(sram, lane, params, batch,
                                                                         first_head);
  }
}

}  // namespace flashinfer::mamba::mtp
