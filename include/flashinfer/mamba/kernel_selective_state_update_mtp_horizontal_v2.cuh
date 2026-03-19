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

// Small-CTA horizontal MTP kernel for selective_state_update.
// 5 warps (4 compute + 1 TMA), HEADS_PER_CTA heads per CTA, pass-level TMA pipelining.
//
// vs. the original horizontal kernel (3 heads per CTA, 15 warps, 1 stage):
// - Pass-level TMA pipelining: while compute processes state chunk N,
//   TMA preloads chunk N+1 (overlaps memory latency with compute).
// - More CTAs in the grid → better warp diversity across SMs, less
//   scheduling overhead from correlated barrier arrivals.
// - Same compute logic per head: 4 warps, horizontal DSTATE traversal,
//   f32x2 packed SIMD, inline epilogue.

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

// Small-CTA horizontal constants: 5 warps per CTA.
namespace horiz_v2 {
static constexpr int NUM_COMPUTE_WARPS_PER_GROUP = 4;
static constexpr int NUM_TMA_WARPS = 1;
static constexpr int NUM_WARPS = NUM_COMPUTE_WARPS_PER_GROUP + NUM_TMA_WARPS;  // 5
static constexpr int LANES_PER_ROW = 8;
static constexpr int ROWS_PER_WARP = 32 / LANES_PER_ROW;                           // 4
static constexpr int ROWS_PER_PASS = NUM_COMPUTE_WARPS_PER_GROUP * ROWS_PER_WARP;  // 16
}  // namespace horiz_v2

// =============================================================================
// Shared memory layout — pass-level pipelining.
// B/C/x loaded once. state_in pipelined across passes (ROWS_PER_PASS chunks).
// =============================================================================

template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES, int HEADS_PER_CTA = 1>
struct GroupStorageHorizontalV2 {
  alignas(128) input_t B[TOKENS_MTP][DSTATE];
  alignas(128) input_t C[TOKENS_MTP][DSTATE];
  alignas(128) state_t state_in[NUM_IN_STAGES][horiz_v2::ROWS_PER_PASS * DSTATE];
  alignas(128) input_t x[HEADS_PER_CTA][TOKENS_MTP][DIM];
  float dt[TOKENS_MTP];
  float out[TOKENS_MTP][DIM];

  barrier_t bar_input_full;  // B + C + x (all heads) loaded
  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
  barrier_t bar_out_ready;  // sync compute warps before/after epilogue
};

// =============================================================================
// TMA warp: loads B, C, x (once), then pipelines state_in across passes.
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, int NTOKENS, int DIM,
          int DSTATE, int NUM_IN_STAGES, int HEADS_PER_CTA, typename SramT>
__device__ __forceinline__ void role_load_horizontal_v2(
    SramT& sram, int lane, SelectiveStateMTPParams const& params, int batch, int base_head,
    int kv_group, CUtensorMap const& tensorState, CUtensorMap const& tensorX,
    CUtensorMap const& tensorB, CUtensorMap const& tensorC) {
  namespace cde = cuda::device::experimental;
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch = state_batch_indices ? (int)state_batch_indices[batch] : batch;

  constexpr int numPasses = (DIM / horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP) / horiz_v2::ROWS_PER_WARP;

  // ── Load B, C (once), and x for all heads ─────────────────────────────
  if (lane == 0) {
    constexpr int bytesBCX = 2 * NTOKENS * DSTATE * (int)sizeof(input_t) +
                             HEADS_PER_CTA * NTOKENS * DIM * (int)sizeof(input_t);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, kv_group, 0, batch,
                                                  sram.bar_input_full);
    cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, kv_group, 0, batch,
                                                  sram.bar_input_full);
#pragma unroll
    for (int h = 0; h < HEADS_PER_CTA; h++) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.x[h][0][0], &tensorX, 0, base_head + h, 0,
                                                    batch, sram.bar_input_full);
    }
    cuda::device::barrier_arrive_tx(sram.bar_input_full, warpSize, bytesBCX);
  }

  // ── Pipeline state_in loads across heads and passes ───────────────────
  uint32_t parity_empty[NUM_IN_STAGES] = {};  // all start at phase 0
#pragma unroll
  for (int h = 0; h < HEADS_PER_CTA; h++) {
    int const head = base_head + h;
    for (int pass = 0; pass < numPasses; pass++) {
      int const slot = pass % NUM_IN_STAGES;
      constexpr int bytesChunk = horiz_v2::ROWS_PER_PASS * DSTATE * (int)sizeof(state_t);

      // Wait for compute to release this slot (tight spin, no NANOSLEEP)
      arrive_and_wait_parity(sram.bar_state_in_empty[slot], parity_empty[slot]);

      if (lane == 0) {
        cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[slot][0], &tensorState, 0,
                                                      pass * horiz_v2::ROWS_PER_PASS, head,
                                                      state_batch, sram.bar_state_in_full[slot]);
        cuda::device::barrier_arrive_tx(sram.bar_state_in_full[slot], 1, bytesChunk);
      }
    }
  }
}

// =============================================================================
// Compute warp: processes one head with pass-level pipelining.
// Same horizontal DSTATE traversal + f32x2 packed SIMD as the v1 kernel.
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int PHILOX_ROUNDS,
          int NUM_IN_STAGES, int HEADS_PER_CTA, typename SramT>
__device__ __forceinline__ void role_update_state_horizontal_v2(
    SramT& sram, int lane, int compute_warp, SelectiveStateMTPParams const& params, int batch,
    int base_head, bool is_pad) {
  constexpr int lanesPerRow = horiz_v2::LANES_PER_ROW;
  constexpr int rowsPerWarp = horiz_v2::ROWS_PER_WARP;
  constexpr int totalRowsPerWarp = DIM / horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP;
  constexpr int numPasses = totalRowsPerWarp / rowsPerWarp;
  constexpr int stateValuesPerThread = DSTATE / lanesPerRow;
  static_assert(DSTATE % lanesPerRow == 0, "DSTATE must be divisible by lanesPerRow");
  static_assert(DIM % horiz_v2::ROWS_PER_PASS == 0, "DIM must be divisible by ROWS_PER_PASS");

  constexpr int bankSize = sizeof(uint32_t);
  constexpr int stateValuesPerBank = bankSize / sizeof(state_t);
  constexpr int numBanks = 32;
  constexpr int sramReadsPerThreadPerTile = numBanks / lanesPerRow;                   // 8
  constexpr int elemsPerTileMember = sramReadsPerThreadPerTile * stateValuesPerBank;  // 16
  constexpr int elemsPerTile = elemsPerTileMember * lanesPerRow;                      // 64
  constexpr int numTiles = stateValuesPerThread / elemsPerTileMember;                 // 2
  using packed_tile_t = PackedAligned<state_t, elemsPerTileMember>;

  static_assert(elemsPerTileMember % 2 == 0, "elemsPerTileMember must be even for f32x2");
  constexpr int pairsPerTileMember = elemsPerTileMember / 2;

  int const member = lane % lanesPerRow;  // position along DSTATE (0..7)
  int const group = lane / lanesPerRow;   // row within current 4 active rows

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

  // Logical column helpers
  auto baseCol = [&](int t, int e) -> int {
    return t * elemsPerTile + member * elemsPerTileMember + e;
  };

  // Output pointers (for epilogue)
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  constexpr auto outputLoadSize = getVectorLoadSizeForFullUtilization<input_t, DIM>();
  using load_output_t = PackedAligned<input_t, outputLoadSize>;
  constexpr int elemsPerThreadEpilogue = DIM / warpSize;

  // Pre-arrive: unblock TMA for state_in slots (once, before first head)
  for (int s = 0; s < NUM_IN_STAGES; s++) {
    sram.bar_state_in_empty[s].arrive();
  }

  // Parity trackers for tight-spin barrier waits
  uint32_t parity_input_full = 0;
  uint32_t parity_full[NUM_IN_STAGES] = {};
  uint32_t parity_out_ready = 0;

  // Wait for B/C/x (all heads) to be loaded
  arrive_and_wait_parity(sram.bar_input_full, parity_input_full);

  // ═══════════════════════════════════════════════════════════════════════
  // Head loop: process HEADS_PER_CTA heads sequentially.
  // Pipeline barriers carry over naturally between heads (phase-based).
  // ═══════════════════════════════════════════════════════════════════════

#pragma unroll
  for (int h = 0; h < HEADS_PER_CTA; h++) {
    int const head = base_head + h;

    // Per-head constants
    auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;
    float const A_val = toFloat(A_ptr[head]);
    float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;
    float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;

    // Load dt into smem (distributed across warps)
    for (int step = compute_warp; step < NTOKENS; step += horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP) {
      if (lane == 0) {
        float dt_val =
            toFloat(dt_ptr[batch * params.dt_stride_batch + step * params.dt_stride_mtp + head]) +
            dt_bias_val;
        if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
        sram.dt[step] = dt_val;
      }
    }

    // ── Pass loop: each iteration processes ROWS_PER_PASS rows of DIM ───
    for (int pass = 0; pass < numPasses; pass++) {
      int const slot = pass % NUM_IN_STAGES;
      int const sram_row = compute_warp * rowsPerWarp + group;   // row within ROWS_PER_PASS chunk
      int const dd = pass * horiz_v2::ROWS_PER_PASS + sram_row;  // global DIM row

      // Wait for state_in chunk to be loaded by TMA (tight spin, no NANOSLEEP)
      arrive_and_wait_parity(sram.bar_state_in_full[slot], parity_full[slot]);

      // Load state from smem (pass-local smem, not full DIM)
      float2 rState[numTiles][pairsPerTileMember];
#pragma unroll
      for (int t = 0; t < numTiles; t++) {
#pragma unroll
        for (int p = 0; p < pairsPerTileMember; p++) {
          int const c0 = baseCol(t, p * 2);
          int const c1 = baseCol(t, p * 2 + 1);
          // Index into pass-local smem: sram_row within the ROWS_PER_PASS chunk
          rState[t][p] = {toFloat(sram.state_in[slot][sram_row * DSTATE + c0]),
                          toFloat(sram.state_in[slot][sram_row * DSTATE + c1])};
        }
      }

      for (int step = 0; step < NTOKENS; step++) {
        float const dt_value = sram.dt[step];
        float const dA = __expf(A_val * dt_value);
        float const x_value = toFloat(sram.x[h][step][dd]);

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
            int const c1 = baseCol(t, p * 2 + 1);
            float2 B2 = {toFloat(sram.B[step][c0]), toFloat(sram.B[step][c1])};
            float2 dBx;
            mul_f32x2(dBx, B2, dtx2);                         // dBx = B * (dt * x)
            fma_f32x2(rState[t][p], dA2, rState[t][p], dBx);  // state = dA * state + dBx
            float2 C2 = {toFloat(sram.C[step][c0]), toFloat(sram.C[step][c1])};
            fma_f32x2(out2, rState[t][p], C2, out2);  // out += state * C
          }
        }
        float out_value = out2.x + out2.y;

        // Reduce across lanesPerRow adjacent lanes
#pragma unroll
        for (int offset = lanesPerRow / 2; offset >= 1; offset /= 2) {
          out_value += __shfl_down_sync(UINT32_MAX, out_value, offset);
        }

        if (member == 0) {
          sram.out[step][dd] = out_value + D_val * x_value;
        }

        // Write intermediate state
        if (istate_ptr && !is_pad) {
          auto const istate_base = icache_idx * params.intermediate_state_stride_batch +
                                   step * params.nheads * DIM * DSTATE + head * DIM * DSTATE +
                                   dd * DSTATE;
#pragma unroll
          for (int t = 0; t < numTiles; t++) {
            packed_tile_t rOut;
            int const col0 = baseCol(t, 0);
#pragma unroll
            for (int e = 0; e < elemsPerTileMember; e += 2) {
              float const s0 = rState[t][e / 2].x;
              float const s1 = rState[t][e / 2].y;
              if constexpr (PHILOX_ROUNDS > 0) {
                [[maybe_unused]] uint32_t rand_ints[4];
                if (e % 4 == 0)
                  philox_randint4x<PHILOX_ROUNDS>(
                      rand_seed, state_ptr_offset + dd * DSTATE + col0 + e, rand_ints[0],
                      rand_ints[1], rand_ints[2], rand_ints[3]);
                uint32_t packed = cvt_rs_f16x2_f32(s0, s1, rand_ints[e / 2 % 2]);
                rOut.val[e] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
                rOut.val[e + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
              } else {
                convertAndStore(&rOut.val[e], s0);
                convertAndStore(&rOut.val[e + 1], s1);
              }
            }
            *reinterpret_cast<packed_tile_t*>(&istate_ptr[istate_base + col0]) = rOut;
          }
        }

        // Write final state at last step
        if (step == NTOKENS - 1 && params.update_state && !is_pad) {
          auto const state_base =
              state_batch * params.state_stride_batch + head * DIM * DSTATE + dd * DSTATE;
#pragma unroll
          for (int t = 0; t < numTiles; t++) {
            packed_tile_t rOut;
            int const col0 = baseCol(t, 0);
#pragma unroll
            for (int e = 0; e < elemsPerTileMember; e += 2) {
              float const s0 = rState[t][e / 2].x;
              float const s1 = rState[t][e / 2].y;
              if constexpr (PHILOX_ROUNDS > 0) {
                [[maybe_unused]] uint32_t rand_ints[4];
                if (e % 4 == 0)
                  philox_randint4x<PHILOX_ROUNDS>(
                      rand_seed, state_ptr_offset + dd * DSTATE + col0 + e, rand_ints[0],
                      rand_ints[1], rand_ints[2], rand_ints[3]);
                uint32_t packed = cvt_rs_f16x2_f32(s0, s1, rand_ints[e / 2 % 2]);
                rOut.val[e] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
                rOut.val[e + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
              } else {
                convertAndStore(&rOut.val[e], s0);
                convertAndStore(&rOut.val[e + 1], s1);
              }
            }
            *reinterpret_cast<packed_tile_t*>(&state_ptr[state_base + col0]) = rOut;
          }
        }
      }  // step loop

      // Release state_in slot for TMA to reuse
      sram.bar_state_in_empty[slot].arrive();
    }  // pass loop

    // ── Epilogue: sync all 4 compute warps, z-gate + vectorized store ───
    arrive_and_wait_parity(sram.bar_out_ready, parity_out_ready);

    for (int step = compute_warp; step < NTOKENS; step += horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP) {
      int const out_offset =
          batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM;

      for (int ii = 0; ii < elemsPerThreadEpilogue; ii += load_output_t::count) {
        int const d = lane * load_output_t::count +
                      (ii / load_output_t::count) * warpSize * load_output_t::count;
        load_output_t packed_out;
        load_output_t packed_z;
        if (z_ptr) {
          packed_z = *reinterpret_cast<load_output_t const*>(&z_ptr[out_offset + d]);
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

    // Sync compute warps after epilogue before next head's dt load
    if (h < HEADS_PER_CTA - 1) {
      arrive_and_wait_parity(sram.bar_out_ready, parity_out_ready);
    }
  }  // head loop
}

// =============================================================================
// Kernel entry point
// Grid: (batch, nheads / HEADS_PER_CTA)
// Block: (32, 5)  — 4 compute + 1 TMA
// =============================================================================

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NTOKENS, int DIM, int DSTATE, int HEADS_PER_GROUP,
          int PHILOX_ROUNDS, int NUM_IN_STAGES, int HEADS_PER_CTA>
__global__ void __launch_bounds__(horiz_v2::NUM_WARPS * 32, 6)
    selective_state_update_kernel_horizontal_v2_mtp(SelectiveStateMTPParams params,
                                                    __grid_constant__ CUtensorMap const tensorState,
                                                    __grid_constant__ CUtensorMap const tensorB,
                                                    __grid_constant__ CUtensorMap const tensorC,
                                                    __grid_constant__ CUtensorMap const tensorX) {
  static_assert(HEADS_PER_GROUP % HEADS_PER_CTA == 0,
                "HEADS_PER_GROUP must be divisible by HEADS_PER_CTA");

  extern __shared__ __align__(128) char smem[];
  using sram_t = GroupStorageHorizontalV2<input_t, state_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES,
                                          HEADS_PER_CTA>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const batch = blockIdx.x;
  int const base_head = blockIdx.y * HEADS_PER_CTA;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;

  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const state_batch =
      state_batch_indices ? (int64_t)state_batch_indices[batch] : (int64_t)batch;
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  // ── Init barriers (warp 0, lane 0) ──────────────────────────────────────
  if (warp == 0 && lane == 0) {
    // bar_input_full: 1 TMA warp (arrive_tx for warpSize) + 4 compute warps
    init(&sram.bar_input_full,
         (horiz_v2::NUM_TMA_WARPS + horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP) * warpSize);
    for (int s = 0; s < NUM_IN_STAGES; s++) {
      // bar_state_in_empty: 4 compute warps + 1 TMA warp
      init(&sram.bar_state_in_empty[s],
           (horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP + horiz_v2::NUM_TMA_WARPS) * warpSize);
      // bar_state_in_full: 1 TMA (arrive_tx, count=1) + 4 compute warps
      init(&sram.bar_state_in_full[s], 1 + horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
    }
    // bar_out_ready: 4 compute warps only
    init(&sram.bar_out_ready, horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
  }
  __syncthreads();

  // ── Warp role dispatch ─────────────────────────────────────────────────
  if (warp < horiz_v2::NUM_COMPUTE_WARPS_PER_GROUP) {
    role_update_state_horizontal_v2<input_t, state_t, matrixA_t, weight_t, stateIndex_t, NTOKENS,
                                    DIM, DSTATE, PHILOX_ROUNDS, NUM_IN_STAGES, HEADS_PER_CTA>(
        sram, lane, warp, params, batch, base_head, is_pad);
  } else {
    int const kv_group = base_head / HEADS_PER_GROUP;
    role_load_horizontal_v2<input_t, state_t, stateIndex_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES,
                            HEADS_PER_CTA>(sram, lane, params, batch, base_head, kv_group,
                                           tensorState, tensorX, tensorB, tensorC);
  }
}

}  // namespace flashinfer::mamba::mtp
