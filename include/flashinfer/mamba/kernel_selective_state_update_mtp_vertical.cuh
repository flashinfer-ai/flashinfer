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
//   1       = idle
//   2       = idle (was store warp — eliminated in step 18)
//   3       = BC warp (issues TMA for B and C once at startup, then idle)
//   4..11   = 8 compute warps (all process same head, split on DIM)
//   12      = epilogue warp (silu(z) * out → gmem)
static constexpr int WARP_LOAD = 0;
static constexpr int WARP_IDLE_1 = 1;
static constexpr int WARP_IDLE_2 = 2;
static constexpr int WARP_BC = 3;
static constexpr int WARP_COMPUTE_BASE = 4;
static constexpr int WARP_EPILOGUE = WARP_COMPUTE_BASE + NUM_COMPUTE_WARPS;  // 12
static constexpr int NUM_WARPS = WARP_EPILOGUE + 1;                          // 13

// =============================================================================
// Shared memory layout
// =============================================================================

template <typename input_t, typename state_t, int TOKENS_MTP, int HEADS_PER_GROUP, int DIM,
          int DSTATE, int NUM_IN_STAGES>
struct SharedStorageVertical {
  // ── Shared across all heads (loaded once via TMA at startup) ────────────
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];

  // dt: scalar per head, per step. Written by compute warps before barrier.
  float dt[TOKENS_MTP];

  // ── Double-buffered state_in + x (load warp fills, compute reads) ───────
  alignas(128) state_t state_in[NUM_IN_STAGES][DIM * DSTATE];
  alignas(128) input_t x[NUM_IN_STAGES][TOKENS_MTP][DIM];

  // ── Output values: single buffer, all tokens for one head ───────────────
  float out[TOKENS_MTP][DIM];

  // ── Barriers ────────────────────────────────────────────────────────────
  barrier_t bar_BC_full;
  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
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
// role_update_state — warps 4-11: SSM update, all warps process same head
// Compute warps write intermediate_states and final state directly to gmem
// using vectorized STG (contiguous-per-thread DSTATE layout).
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_update_state(SramT& sram, int lane, int compute_warp,
                                                  SelectiveStateMTPParams const& params, int batch,
                                                  int first_head, bool is_pad) {
  // sizeof(state_t) == sizeof(input_t) is enforced by the launcher via FLASHINFER_CHECK + if
  // constexpr guard
  constexpr int rowsPerWarp = DIM / NUM_COMPUTE_WARPS;
  constexpr auto stateValuesPerThread = DSTATE / warpSize;
  // Packed type for vectorized gmem stores: stateValuesPerThread contiguous state_t values
  using packed_state_t = PackedAligned<state_t, stateValuesPerThread>;

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  // Gmem pointers for direct STG writes
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
    // Contiguous-per-thread indexing: thread t holds elements [t*N, t*N+1, ..., t*N+(N-1)]
    float rState[rowsPerWarp][stateValuesPerThread];

    for (int wr = 0; wr < rowsPerWarp; wr++) {
      int const dd = compute_warp * rowsPerWarp + wr;
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        rState[wr][ii] =
            toFloat(sram.state_in[in_slot][dd * DSTATE + lane * stateValuesPerThread + ii]);
      }
    }

    // Step-major loop
    for (int step = 0; step < NTOKENS; step++) {
      float const dt_value = sram.dt[step];
      float const dA = __expf(A_val * dt_value);

      for (int wr = 0; wr < rowsPerWarp; wr++) {
        int const dd = compute_warp * rowsPerWarp + wr;

        float x_value = toFloat(sram.x[in_slot][step][dd]);
        float out_value = D_val * x_value * int(lane == 0);

#pragma unroll
        for (int ii = 0; ii < stateValuesPerThread; ii++) {
          float& sv = rState[wr][ii];
          float dB = toFloat(sram.B[step][lane * stateValuesPerThread + ii]) * dt_value;
          sv = sv * dA + dB * x_value;
          out_value += sv * toFloat(sram.C[step][lane * stateValuesPerThread + ii]);
        }

        out_value = warpReduceSum(out_value);

        if (lane == 0) {
          sram.out[step][dd] = out_value;
        }

        // Write intermediate state directly to gmem via vectorized STG
        if (istate_ptr && !is_pad) {
          packed_state_t rStateOut;
#pragma unroll
          for (int k = 0; k < stateValuesPerThread; k++) {
            convertAndStore(&rStateOut.val[k], rState[wr][k]);
          }
          *reinterpret_cast<packed_state_t*>(
              &istate_ptr[icache_idx * params.intermediate_state_stride_batch +
                          step * params.nheads * DIM * DSTATE + head * DIM * DSTATE + dd * DSTATE +
                          lane * stateValuesPerThread]) = rStateOut;
        }

        // Write final state directly to gmem at last step
        if (step == NTOKENS - 1 && params.update_state && !is_pad) {
          packed_state_t rStateOut;
#pragma unroll
          for (int k = 0; k < stateValuesPerThread; k++) {
            convertAndStore(&rStateOut.val[k], rState[wr][k]);
          }
          *reinterpret_cast<packed_state_t*>(
              &state_ptr[state_batch * params.state_stride_batch + head * DIM * DSTATE +
                         dd * DSTATE + lane * stateValuesPerThread]) = rStateOut;
        }
      }  // warpRow loop
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
          int NUM_IN_STAGES>
__global__ void selective_state_update_kernel_vertical_mtp(
    SelectiveStateMTPParams params, __grid_constant__ CUtensorMap const tensorState,
    __grid_constant__ CUtensorMap const tensorB, __grid_constant__ CUtensorMap const tensorC,
    __grid_constant__ CUtensorMap const tensorX) {
  extern __shared__ __align__(128) char smem[];
  using sram_t =
      SharedStorageVertical<input_t, state_t, NTOKENS, HEADS_PER_GROUP, DIM, DSTATE, NUM_IN_STAGES>;
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
    constexpr int numEpilogueWarps = 1;

    init(&sram.bar_BC_full, (numBCLoadWarps + NUM_COMPUTE_WARPS) * warpSize);

    for (int s = 0; s < NUM_IN_STAGES; s++) {
      init(&sram.bar_state_in_empty[s], (NUM_COMPUTE_WARPS + numLoadWarps) * warpSize);
      init(&sram.bar_state_in_full[s], 1 + NUM_COMPUTE_WARPS * warpSize);
    }
    init(&sram.bar_out_ready, (NUM_COMPUTE_WARPS + numEpilogueWarps) * warpSize);
    init(&sram.bar_epilogue_done, (numEpilogueWarps + NUM_COMPUTE_WARPS) * warpSize);
  }
  __syncthreads();

  // ── Warp role dispatch ──────────────────────────────────────────────────
  if (warp == WARP_LOAD) {
    role_load<input_t, state_t, stateIndex_t, NTOKENS, DIM, DSTATE, HEADS_PER_GROUP, NUM_IN_STAGES>(
        sram, lane, params, batch, first_head, tensorState, tensorX);

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
    role_update_state<input_t, state_t, matrixA_t, weight_t, stateIndex_t, NTOKENS, DIM, DSTATE,
                      HEADS_PER_GROUP, NUM_IN_STAGES>(sram, lane, compute_warp, params, batch,
                                                      first_head, is_pad);

  } else if (warp == WARP_EPILOGUE) {
    role_epilogue<input_t, NTOKENS, DIM, HEADS_PER_GROUP>(sram, lane, params, batch, first_head);
  }
  // Warps 1, 2 are idle
}

}  // namespace flashinfer::mamba::mtp
