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
// All 8 compute warps process a single full DIM×DSTATE head tile together,
// each warp handling DIM/8 rows of the state.
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

static constexpr int NUM_COMPUTE_WARPS = 8;

// Warp index layout:
//   0       = load warp (per-head state_in + x)
//   1       = idle (was z warp)
//   2       = store warp (drain state_out ring → gmem)
//   3       = BC warp (issues TMA for B and C once at startup, then idle)
//   4..11   = 8 compute warps (all process same head, split on DIM)
//   12      = epilogue warp (silu(z) * out → gmem)
static constexpr int WARP_LOAD = 0;
static constexpr int WARP_IDLE_1 = 1;
static constexpr int WARP_STORE = 2;
static constexpr int WARP_BC = 3;
static constexpr int WARP_COMPUTE_BASE = 4;
static constexpr int WARP_EPILOGUE = WARP_COMPUTE_BASE + NUM_COMPUTE_WARPS;  // 12
static constexpr int NUM_WARPS = WARP_EPILOGUE + 1;                          // 13

// =============================================================================
// Shared memory layout
// =============================================================================

template <typename input_t, typename state_t, int TOKENS_MTP, int HEADS_PER_GROUP, int DIM,
          int DSTATE, int NUM_IN_STAGES, int NUM_OUT_STAGES>
struct SharedStorageVertical {
  // ── Shared across all heads (loaded once via TMA at startup) ────────────
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];

  // dt: scalar per head, per step. Written by compute warps before barrier.
  float dt[TOKENS_MTP];

  // ── Double-buffered state_in + x (load warp fills, compute reads) ───────
  alignas(128) state_t state_in[NUM_IN_STAGES][DIM * DSTATE];
  alignas(128) input_t x[NUM_IN_STAGES][TOKENS_MTP][DIM];

  // ── Ring-buffered state_out (compute writes, store warp drains) ─────────
  alignas(128) state_t state_out[NUM_OUT_STAGES][DIM * DSTATE];

  // ── Output values: single buffer, all tokens for one head ───────────────
  float out[TOKENS_MTP][DIM];

  // ── Barriers ────────────────────────────────────────────────────────────
  barrier_t bar_BC_full;
  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
  barrier_t bar_out_empty[NUM_OUT_STAGES];
  barrier_t bar_out_full[NUM_OUT_STAGES];
  barrier_t bar_out_ready;
  barrier_t bar_epilogue_done;
};

// =============================================================================
// role_load — warp 0: per-head TMA state_in + x load
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, int NTOKENS, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_load(SramT& sram, int lane,
                                          SelectiveStateMTPParams const& params, int batch,
                                          int first_head, CUtensorMap const& tensorState,
                                          CUtensorMap const& tensorX) {
  namespace cde = cuda::device::experimental;
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch = state_batch_indices ? (int)state_batch_indices[batch] : batch;

  constexpr int bytesState = DIM * DSTATE * (int)sizeof(state_t);
  constexpr int bytesX = NTOKENS * DIM * (int)sizeof(input_t);

  for (int h_local = 0; h_local < HEADS_PER_GROUP; h_local++) {
    int const in_slot = h_local % NUM_IN_STAGES;
    int const head = first_head + h_local;

    // Wait for compute to release this in-stage
    sram.bar_state_in_empty[in_slot].wait(sram.bar_state_in_empty[in_slot].arrive());

    if (lane == 0) {
      // TMA load: full head state
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[in_slot][0], &tensorState, 0, 0,
                                                    head, state_batch,
                                                    sram.bar_state_in_full[in_slot]);

      // TMA load: x for all tokens, full DIM
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.x[in_slot][0][0], &tensorX, 0, head, 0,
                                                    batch, sram.bar_state_in_full[in_slot]);

      auto const _ =
          cuda::device::barrier_arrive_tx(sram.bar_state_in_full[in_slot], 1, bytesState + bytesX);
    }
  }
}

// =============================================================================
// role_store — warp 2: drain state_out ring → gmem via TMA
// =============================================================================

template <typename state_t, typename stateIndex_t, int NTOKENS, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int NUM_OUT_STAGES, typename SramT>
__device__ __forceinline__ void role_store(SramT& sram, int lane,
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

  // Pre-arrive: initial out slots start empty
  for (int s = 0; s < NUM_OUT_STAGES; s++) {
    sram.bar_out_empty[s].arrive();
  }

  for (int h_local = 0; h_local < HEADS_PER_GROUP; h_local++) {
    int const head = first_head + h_local;

    for (int step = 0; step < NTOKENS; step++) {
      int const slot = step % NUM_OUT_STAGES;

      // Wait for compute to fill this slot
      sram.bar_out_full[slot].wait(sram.bar_out_full[slot].arrive());

      if (!is_pad) {
        bool did_tma_store = false;

        // Write intermediate states via TMA store
        if (intermediate_states) {
          if (lane == 0) {
            cde::cp_async_bulk_tensor_5d_shared_to_global(&tensorIntermediateState, 0, 0, head,
                                                          step, (int)icache_idx,
                                                          &sram.state_out[slot][0]);
          }
          did_tma_store = true;
        }

        // Write final state via TMA store (last token step only)
        if (step == NTOKENS - 1 && params.update_state) {
          if (lane == 0) {
            cde::cp_async_bulk_tensor_4d_shared_to_global(
                &tensorState, 0, 0, head, (int)state_batch, &sram.state_out[slot][0]);
          }
          did_tma_store = true;
        }

        if (did_tma_store) {
          if (lane == 0) {
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
          }
          __syncwarp();
        }
      }

      // Release slot back to compute
      sram.bar_out_empty[slot].arrive();
    }
  }
}

// =============================================================================
// role_compute — warps 4-11: SSM update, all warps process same head
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t, int NTOKENS,
          int DIM, int DSTATE, int HEADS_PER_GROUP, int NUM_IN_STAGES, int NUM_OUT_STAGES,
          typename SramT>
__device__ __forceinline__ void role_compute(SramT& sram, int lane, int compute_warp,
                                             SelectiveStateMTPParams const& params, int batch,
                                             int first_head) {
  using load_input_t = PackedAligned<input_t>;
  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;
  constexpr int rowsPerWarp = DIM / NUM_COMPUTE_WARPS;
  constexpr auto stateValuesPerThread = DSTATE / warpSize;
  constexpr auto maxPackedElements = sizeof(uint64_t) / sizeof(input_t);
  constexpr auto packedSramLdInputElements =
      (stateValuesPerThread >= maxPackedElements) ? maxPackedElements : stateValuesPerThread;
  using packed_input_t = PackedAligned<input_t, packedSramLdInputElements>;

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  // Wait for B/C to be loaded
  sram.bar_BC_full.wait(sram.bar_BC_full.arrive());

  // Pre-arrive: unblock load warp for first in-stage
  for (int s = 0; s < NUM_IN_STAGES; s++) {
    sram.bar_state_in_empty[s].arrive();
  }

  for (int h_local = 0; h_local < HEADS_PER_GROUP; h_local++) {
    int const in_slot = h_local % NUM_IN_STAGES;
    int const head = first_head + h_local;

    // Pre-load A, D scalars from gmem
    float const A_val = toFloat(A_ptr[head]);
    float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

    // Load dt values — distributed across compute warps
    {
      float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      for (int step = compute_warp; step < NTOKENS; step += NUM_COMPUTE_WARPS) {
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

    // Load state into registers: rState[wr][ii] persists across token steps
    float rState[rowsPerWarp][stateValuesPerThread];
    packed_input_t rB, rC;

    for (int wr = 0; wr < rowsPerWarp; wr++) {
      int const dd = compute_warp * rowsPerWarp + wr;
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        int i = lane * packed_input_t::count +
                (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                (ii % packed_input_t::count);
        rState[wr][ii] = (i < DSTATE) ? toFloat(sram.state_in[in_slot][dd * DSTATE + i]) : 0.f;
      }
    }

    // Step-major loop
    for (int step = 0; step < NTOKENS; step++) {
      int const slot = step % NUM_OUT_STAGES;
      float const dt_value = sram.dt[step];
      float const dA = __expf(A_val * dt_value);

      // Wait for store warp to free this out-slot
      sram.bar_out_empty[slot].wait(sram.bar_out_empty[slot].arrive());

      for (int wr = 0; wr < rowsPerWarp; wr++) {
        int const dd = compute_warp * rowsPerWarp + wr;

        float x_value = toFloat(sram.x[in_slot][step][dd]);
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

        if (lane == 0) {
          sram.out[step][dd] = out_value;
        }

        // Write rState to state_out ring
        for (int ii = 0; ii < stateValuesPerThread; ii++) {
          int i = lane * packed_input_t::count +
                  (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                  (ii % packed_input_t::count);
          if (i < DSTATE) {
            convertAndStore(&sram.state_out[slot][dd * DSTATE + i], rState[wr][ii]);
          }
        }
      }  // warpRow loop

      // All rows written for this step — signal store warp
      sram.bar_out_full[slot].arrive();
    }  // step loop

    // Signal epilogue: out[] ready
    sram.bar_out_ready.arrive();

    // Wait for epilogue to finish before releasing in-stage
    sram.bar_epilogue_done.wait(sram.bar_epilogue_done.arrive());

    // Release in-stage for next head's load
    sram.bar_state_in_empty[in_slot].arrive();
  }
}

// =============================================================================
// role_epilogue — warp 12: silu(z) * out → gmem
// =============================================================================

template <typename input_t, int NTOKENS, int DIM, int HEADS_PER_GROUP, typename SramT>
__device__ __forceinline__ void role_epilogue(SramT& sram, int lane,
                                              SelectiveStateMTPParams const& params, int batch,
                                              int first_head) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);

  for (int h_local = 0; h_local < HEADS_PER_GROUP; h_local++) {
    int const head = first_head + h_local;

    // Wait for all tokens' output to be ready
    sram.bar_out_ready.wait(sram.bar_out_ready.arrive());

    constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
    using load_output_t = PackedAligned<input_t, outputLoadSize>;

    constexpr int elemsPerThread = DIM / warpSize;
    for (int step = 0; step < NTOKENS; step++) {
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

    // Signal compute: epilogue done, in-stage can be reused
    sram.bar_epilogue_done.arrive();
  }
}

// =============================================================================
// Kernel entry point
// =============================================================================

// Grid: (batch, ngroups)
// Block: (32, NUM_WARPS)
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int NUM_IN_STAGES, int NUM_OUT_STAGES>
__global__ void selective_state_update_kernel_vertical_mtp(
    SelectiveStateMTPParams params, __grid_constant__ CUtensorMap const tensorState,
    __grid_constant__ CUtensorMap const tensorB, __grid_constant__ CUtensorMap const tensorC,
    __grid_constant__ CUtensorMap const tensorIntermediateState,
    __grid_constant__ CUtensorMap const tensorX) {
  extern __shared__ __align__(128) char smem[];
  using sram_t = SharedStorageVertical<input_t, state_t, NTOKENS, HEADS_PER_GROUP, DIM, DSTATE,
                                       NUM_IN_STAGES, NUM_OUT_STAGES>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const first_head = group_idx * HEADS_PER_GROUP;

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  // ── Init barriers (warp 0, lane 0) ──────────────────────────────────────
  if (warp == 0 && lane == 0) {
    constexpr int numBCLoadWarps = 1;
    constexpr int numLoadWarps = 1;
    constexpr int numStoreWarps = 1;
    constexpr int numEpilogueWarps = 1;

    init(&sram.bar_BC_full, (numBCLoadWarps + NUM_COMPUTE_WARPS) * warpSize);

    for (int s = 0; s < NUM_IN_STAGES; s++) {
      init(&sram.bar_state_in_empty[s], (NUM_COMPUTE_WARPS + numLoadWarps) * warpSize);
      init(&sram.bar_state_in_full[s], 1 + NUM_COMPUTE_WARPS * warpSize);
    }
    for (int s = 0; s < NUM_OUT_STAGES; s++) {
      init(&sram.bar_out_empty[s], (numStoreWarps + NUM_COMPUTE_WARPS) * warpSize);
      init(&sram.bar_out_full[s], (NUM_COMPUTE_WARPS + numStoreWarps) * warpSize);
    }
    init(&sram.bar_out_ready, (NUM_COMPUTE_WARPS + numEpilogueWarps) * warpSize);
    init(&sram.bar_epilogue_done, (numEpilogueWarps + NUM_COMPUTE_WARPS) * warpSize);
  }
  __syncthreads();

  // ── Warp role dispatch ──────────────────────────────────────────────────
  if (warp == WARP_LOAD) {
    role_load<input_t, state_t, stateIndex_t, NTOKENS, DIM, DSTATE, HEADS_PER_GROUP, NUM_IN_STAGES>(
        sram, lane, params, batch, first_head, tensorState, tensorX);

  } else if (warp == WARP_IDLE_1) {
    // Idle

  } else if (warp == WARP_STORE) {
    role_store<state_t, stateIndex_t, NTOKENS, DIM, DSTATE, HEADS_PER_GROUP, NUM_OUT_STAGES>(
        sram, lane, params, batch, first_head, is_pad, tensorState, tensorIntermediateState);

  } else if (warp == WARP_BC) {
    // Issue TMA loads for B and C at startup; then idle.
    if (lane == 0) {
      namespace cde = cuda::device::experimental;
      constexpr int bytesBC = 2 * NTOKENS * DSTATE * (int)sizeof(input_t);
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, group_idx, 0, batch,
                                                    sram.bar_BC_full);
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, group_idx, 0, batch,
                                                    sram.bar_BC_full);
      cuda::device::barrier_arrive_tx(sram.bar_BC_full, warpSize, bytesBC);
    }

  } else if (warp >= WARP_COMPUTE_BASE && warp < WARP_EPILOGUE) {
    int const compute_warp = warp - WARP_COMPUTE_BASE;
    role_compute<input_t, state_t, matrixA_t, weight_t, NTOKENS, DIM, DSTATE, HEADS_PER_GROUP,
                 NUM_IN_STAGES, NUM_OUT_STAGES>(sram, lane, compute_warp, params, batch,
                                                first_head);

  } else if (warp == WARP_EPILOGUE) {
    role_epilogue<input_t, NTOKENS, DIM, HEADS_PER_GROUP>(sram, lane, params, batch, first_head);
  }
}

}  // namespace flashinfer::mamba::mtp
