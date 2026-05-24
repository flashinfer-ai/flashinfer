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
// 3 fully independent compute groups per CTA, each with 4 state update warps,
// 3 TMA load warps (one per group), and 1 shared epilogue warp.
// Each CTA processes up to 3 heads from the flattened head list (across all KV groups).
//

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

// =============================================================================
// Constants
// =============================================================================

namespace vertical_constants {
static constexpr int NUM_COMPUTE_GROUPS = 3;
static constexpr int NUM_COMPUTE_WARPS_PER_GROUP = 4;
static constexpr int NUM_WARPS = 16;  // 12 compute + 3 TMA + 1 epilogue
}  // namespace vertical_constants

// Warp layout (no setmaxnreg, no warp-group alignment required):
// Warps  0..3:  compute group 0
// Warps  4..7:  compute group 1
// Warps  8..11: compute group 2
// Warp  12:     TMA load for group 0
// Warp  13:     TMA load for group 1
// Warp  14:     TMA load for group 2
// Warp  15:     epilogue (shared across all 3 groups)

// =============================================================================
// Shared memory layout
// =============================================================================

// Per-group shared memory: fully independent, each group has its own data + barriers
template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES>
struct GroupStorage {
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];
  float dt[TOKENS_MTP];
  alignas(128) state_t state_in[NUM_IN_STAGES][DIM * DSTATE];
  alignas(128) input_t x[NUM_IN_STAGES][TOKENS_MTP][DIM];
  float out[TOKENS_MTP][DIM];

  barrier_t bar_BC_full;
  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
  barrier_t bar_out_ready;
  barrier_t bar_epilogue_done;
};

template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES>
struct SharedStorageVertical {
  GroupStorage<input_t, state_t, TOKENS_MTP, DIM, DSTATE, NUM_IN_STAGES>
      group[vertical_constants::NUM_COMPUTE_GROUPS];
};

// =============================================================================
// role_load — one TMA warp per group: loads BC + state_in + x for a single head
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, bool IS_PAD, int NTOKENS,
          int DIM, int DSTATE, int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_load(SramT& sram, int lane,
                                          SelectiveStateMTPParams const& params, int batch,
                                          int head, int kv_group, int64_t state_batch,
                                          CUtensorMap const& tensorState,
                                          CUtensorMap const& tensorX, CUtensorMap const& tensorB,
                                          CUtensorMap const& tensorC) {
  namespace cde = cuda::device::experimental;

  // ── Load B and C (always, even for pad slots — output must be valid) ──
  if (lane == 0) {
    constexpr int bytesBC = 2 * NTOKENS * DSTATE * (int)sizeof(input_t);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, kv_group, 0, batch,
                                                  sram.bar_BC_full);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, kv_group, 0, batch,
                                                  sram.bar_BC_full);
    cuda::device::barrier_arrive_tx(sram.bar_BC_full, warpSize, bytesBC);
  }

  // ── Load state_in + x ────────────────────────────────────────────────
  // x is always loaded; state is only loaded for non-pad slots (pad uses zero state).
  constexpr int bytesState = DIM * DSTATE * (int)sizeof(state_t);
  constexpr int bytesX = NTOKENS * DIM * (int)sizeof(input_t);
  constexpr int in_slot = 0;  // single head, single slot

  // Wait for compute to release this in-stage
  sram.bar_state_in_empty[in_slot].wait(sram.bar_state_in_empty[in_slot].arrive());

  if (lane == 0) {
    if constexpr (!IS_PAD) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[in_slot][0], &tensorState, 0, 0,
                                                    head, state_batch,
                                                    sram.bar_state_in_full[in_slot]);
    }

    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.x[in_slot][0][0], &tensorX, 0, head, 0,
                                                  batch, sram.bar_state_in_full[in_slot]);

    int constexpr bytes = IS_PAD ? bytesX : bytesState + bytesX;
    auto const _ = cuda::device::barrier_arrive_tx(sram.bar_state_in_full[in_slot], 1, bytes);
  }
}

// =============================================================================
// convertAndStoreSRVertical — convert float state registers to packed half.
// When PHILOX_ROUNDS > 0: stochastic rounding via f16x2 pairs + scalar tail.
// When PHILOX_ROUNDS == 0: plain nearest-even conversion.
// =============================================================================

template <typename state_t, int DSTATE, int stateValuesPerThread, int PHILOX_ROUNDS>
__device__ __forceinline__ void convertAndStoreSRVertical(
    PackedAligned<state_t, stateValuesPerThread>& rStateOut, float const* rStateRow,
    int64_t rand_seed, int state_ptr_offset, int dd, int lane) {
  if constexpr (PHILOX_ROUNDS > 0) {
    constexpr int pairedEnd = (stateValuesPerThread / 2) * 2;
    [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
    for (int k = 0; k < pairedEnd; k += 2) {
      if (k % 4 == 0)
        philox_randint4x<PHILOX_ROUNDS>(
            rand_seed, state_ptr_offset + dd * DSTATE + lane * stateValuesPerThread + k,
            rand_ints[0], rand_ints[1], rand_ints[2], rand_ints[3]);
      uint32_t packed = cvt_rs_f16x2_f32(rStateRow[k], rStateRow[k + 1], rand_ints[k / 2]);
      rStateOut.val[k] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
      rStateOut.val[k + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
    }
    if constexpr (stateValuesPerThread % 2 == 1) {
      constexpr int k = pairedEnd;
      if (k % 4 == 0)
        philox_randint4x<PHILOX_ROUNDS>(
            rand_seed, state_ptr_offset + dd * DSTATE + lane * stateValuesPerThread + k,
            rand_ints[0], rand_ints[1], rand_ints[2], rand_ints[3]);
      rStateOut.val[k] = cvt_rs_f16_f32(rStateRow[k], rand_ints[k / 2] & 0x1FFFu);
    }
  } else {
#pragma unroll
    for (int k = 0; k < stateValuesPerThread; k++) {
      convertAndStore(&rStateOut.val[k], rStateRow[k]);
    }
  }
}

// =============================================================================
// role_update_state — 4 state update warps per group process a single head
// State update warps write intermediate_states and final state directly to gmem
// using vectorized STG (contiguous-per-thread DSTATE layout).
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, bool IS_PAD, int NTOKENS, int DIM, int DSTATE, int PHILOX_ROUNDS,
          int NUM_IN_STAGES, typename SramT>
__device__ __forceinline__ void role_update_state(SramT& sram, int lane, int compute_warp,
                                                  SelectiveStateMTPParams const& params, int batch,
                                                  int head, int64_t state_batch) {
  using namespace vertical_constants;
  constexpr int rowsPerWarp = DIM / NUM_COMPUTE_WARPS_PER_GROUP;
  constexpr int rowsPerWarpPerPass = 4;
  static_assert(rowsPerWarp % rowsPerWarpPerPass == 0,
                "rowsPerWarp must be divisible by rowsPerWarpPerPass");
  constexpr int numPasses = rowsPerWarp / rowsPerWarpPerPass;
  constexpr auto stateValuesPerThread = DSTATE / warpSize;
  using packed_state_t = PackedAligned<state_t, stateValuesPerThread>;

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  // Load device-side Philox seed once into a register
  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);
  auto* __restrict__ istate_ptr = reinterpret_cast<state_t*>(params.intermediate_states);

  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;
  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;

  // Issue scalar gmem loads early so their latency overlaps with barrier waits below.
  float const A_val = toFloat(A_ptr[head]);
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;

  // Pre-arrive: unblock load warp for state_in
  for (int s = 0; s < NUM_IN_STAGES; s++) {
    sram.bar_state_in_empty[s].arrive();
  }

  // Wait for B/C to be loaded
  sram.bar_BC_full.wait(sram.bar_BC_full.arrive());

  constexpr int in_slot = 0;  // single head, single slot

  // Load dt values into smem — distributed across compute warps.
  // A_val, D_val, dt_bias_val are already in registers (loaded above).
  {
    for (int step = compute_warp; step < NTOKENS; step += NUM_COMPUTE_WARPS_PER_GROUP) {
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

  // Process state in passes of rowsPerWarpPerPass rows to reduce register pressure.
  for (int pass = 0; pass < numPasses; pass++) {
    int const row_offset = compute_warp * rowsPerWarp + pass * rowsPerWarpPerPass;

    float rState[rowsPerWarpPerPass][stateValuesPerThread];

#pragma unroll
    for (int wr = 0; wr < rowsPerWarpPerPass; wr++) {
      int const dd = row_offset + wr;
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        if constexpr (IS_PAD) {
          rState[wr][ii] = 0.f;
        } else {
          rState[wr][ii] =
              toFloat(sram.state_in[in_slot][dd * DSTATE + lane * stateValuesPerThread + ii]);
        }
      }
    }

    for (int step = 0; step < NTOKENS; step++) {
      float const dt_value = sram.dt[step];
      float const dA = __expf(A_val * dt_value);

#pragma unroll
      for (int wr = 0; wr < rowsPerWarpPerPass; wr++) {
        int const dd = row_offset + wr;

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
        if (istate_ptr && !IS_PAD) {
          packed_state_t rStateOut;
          convertAndStoreSRVertical<state_t, DSTATE, stateValuesPerThread, PHILOX_ROUNDS>(
              rStateOut, rState[wr], rand_seed, state_ptr_offset, dd, lane);
          *reinterpret_cast<packed_state_t*>(
              &istate_ptr[icache_idx * params.intermediate_state_stride_batch +
                          step * params.nheads * DIM * DSTATE + head * DIM * DSTATE + dd * DSTATE +
                          lane * stateValuesPerThread]) = rStateOut;
        }

        // Write final state directly to gmem at last step
        if (step == NTOKENS - 1 && params.update_state && !IS_PAD) {
          packed_state_t rStateOut;
          convertAndStoreSRVertical<state_t, DSTATE, stateValuesPerThread, PHILOX_ROUNDS>(
              rStateOut, rState[wr], rand_seed, state_ptr_offset, dd, lane);
          *reinterpret_cast<packed_state_t*>(
              &state_ptr[state_batch * params.state_stride_batch + head * DIM * DSTATE +
                         dd * DSTATE + lane * stateValuesPerThread]) = rStateOut;
        }
      }  // warpRow loop
    }  // step loop
  }  // pass loop

  // Signal epilogue: out[] ready
  sram.bar_out_ready.arrive();

  // Wait for epilogue to finish (needed before CTA exits so epilogue can read sram.out)
  sram.bar_epilogue_done.wait(sram.bar_epilogue_done.arrive());
}

// =============================================================================
// role_epilogue — single warp (warp 15) processes all active compute groups
// =============================================================================

template <typename input_t, int NTOKENS, int DIM, typename SharedSramT>
__device__ __forceinline__ void role_epilogue(
    SharedSramT& sram, int lane, SelectiveStateMTPParams const& params, int batch,
    int const heads[vertical_constants::NUM_COMPUTE_GROUPS], int num_active_groups) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);

  constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
  using load_output_t = PackedAligned<input_t, outputLoadSize>;
  constexpr int elemsPerThread = DIM / warpSize;

  for (int g = 0; g < num_active_groups; g++) {
    int const head = heads[g];
    auto& gsram = sram.group[g];

    // Wait for compute group g to finish writing out[]
    gsram.bar_out_ready.wait(gsram.bar_out_ready.arrive());

    for (int step = 0; step < NTOKENS; step++) {
      int const base_offset =
          batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM;
      int const z_base_offset =
          batch * params.z_stride_batch + step * params.z_stride_mtp + head * DIM;
      for (int ii = 0; ii < elemsPerThread; ii += load_output_t::count) {
        int const d = lane * load_output_t::count +
                      (ii / load_output_t::count) * warpSize * load_output_t::count;
        load_output_t packed_out;
        load_output_t packed_z;
        if (z_ptr) {
          packed_z = *reinterpret_cast<load_output_t const*>(&z_ptr[z_base_offset + d]);
        }
#pragma unroll
        for (int k = 0; k < load_output_t::count; k++) {
          float out_value = gsram.out[step][d + k];
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

    // Signal compute group g: epilogue done
    gsram.bar_epilogue_done.arrive();
  }
}

// =============================================================================
// Kernel entry point
// =============================================================================

// Grid: (batch, ceil(nheads / NUM_COMPUTE_GROUPS))
// Block: (32, NUM_WARPS)
// Each CTA processes up to NUM_COMPUTE_GROUPS heads from the flattened head list.
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int PHILOX_ROUNDS, int NUM_IN_STAGES>
__global__ void __launch_bounds__(vertical_constants::NUM_WARPS * 32, 2)
    selective_state_update_kernel_vertical_mtp(SelectiveStateMTPParams params,
                                               __grid_constant__ CUtensorMap const tensorState,
                                               __grid_constant__ CUtensorMap const tensorB,
                                               __grid_constant__ CUtensorMap const tensorC,
                                               __grid_constant__ CUtensorMap const tensorX) {
  using namespace vertical_constants;
  extern __shared__ __align__(128) char smem[];
  using sram_t = SharedStorageVertical<input_t, state_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const chunk_idx = blockIdx.y;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const total_heads = params.nheads;

  // Compute head indices for the 3 compute groups
  int heads[NUM_COMPUTE_GROUPS];
  int num_active_groups = 0;
  for (int g = 0; g < NUM_COMPUTE_GROUPS; g++) {
    heads[g] = chunk_idx * NUM_COMPUTE_GROUPS + g;
    if (heads[g] < total_heads) num_active_groups = g + 1;
  }

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  // ── Init barriers (warp 0, lane 0) ──────────────────────────────────────
  if (warp == 0 && lane == 0) {
    for (int g = 0; g < num_active_groups; g++) {
      auto& gsram = sram.group[g];
      init(&gsram.bar_BC_full, (1 + NUM_COMPUTE_WARPS_PER_GROUP) * warpSize);
      for (int s = 0; s < NUM_IN_STAGES; s++) {
        init(&gsram.bar_state_in_empty[s], (NUM_COMPUTE_WARPS_PER_GROUP + 1) * warpSize);
        init(&gsram.bar_state_in_full[s], 1 + NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
      }
      init(&gsram.bar_out_ready, (NUM_COMPUTE_WARPS_PER_GROUP + 1) * warpSize);
      init(&gsram.bar_epilogue_done, (1 + NUM_COMPUTE_WARPS_PER_GROUP) * warpSize);
    }
  }
  __syncthreads();

  // ── Warp role dispatch (no setmaxnreg) ─────────────────────────────────
  auto const role = get_warp_role(warp);

  auto dispatch = [&]<bool IS_PAD>() {
    if (role == WarpRole::kCompute) {
      int const g = warp / NUM_COMPUTE_WARPS_PER_GROUP;  // 0, 1, or 2
      int const compute_warp = warp % NUM_COMPUTE_WARPS_PER_GROUP;
      if (g < num_active_groups) {
        role_update_state<input_t, state_t, matrixA_t, weight_t, stateIndex_t, IS_PAD, NTOKENS, DIM,
                          DSTATE, PHILOX_ROUNDS, NUM_IN_STAGES>(
            sram.group[g], lane, compute_warp, params, batch, heads[g], state_batch);
      }

    } else if (role == WarpRole::kTMALoad) {
      int const g = warp - 12;  // 0, 1, or 2
      if (g < num_active_groups) {
        int const kv_group = heads[g] / HEADS_PER_GROUP;
        role_load<input_t, state_t, stateIndex_t, IS_PAD, NTOKENS, DIM, DSTATE, NUM_IN_STAGES>(
            sram.group[g], lane, params, batch, heads[g], kv_group, state_batch, tensorState,
            tensorX, tensorB, tensorC);
      }

    } else /* role == WarpRole::kEpilogue */ {
      role_epilogue<input_t, NTOKENS, DIM>(sram, lane, params, batch, heads, num_active_groups);
    }
  };
  if (is_pad)
    dispatch.template operator()<true>();
  else
    dispatch.template operator()<false>();
}

}  // namespace flashinfer::mamba::mtp
