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
// 5 warps (4 compute + 1 TMA), HEADS_PER_CTA heads per CTA, pass-level TMA pipelining.
//
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

// Horizontal kernel constants: 5 warps per CTA.
namespace horiz {
static constexpr int NUM_COMPUTE_WARPS_PER_GROUP = 4;
static constexpr int NUM_TMA_WARPS = 1;
static constexpr int NUM_WARPS = NUM_COMPUTE_WARPS_PER_GROUP + NUM_TMA_WARPS;  // 5
static constexpr int LANES_PER_ROW = 8;
static constexpr int ROWS_PER_WARP = warpSize / LANES_PER_ROW;                     // 4
static constexpr int ROWS_PER_PASS = NUM_COMPUTE_WARPS_PER_GROUP * ROWS_PER_WARP;  // 16
}  // namespace horiz

// =============================================================================
// Shared memory layout — TMA-level pipelining.
// B/C/x loaded once. state_in pipelined across TMA loads (TMA_STATE_ROWS chunks).
// =============================================================================

template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE,
          int NUM_IN_STAGES, int TMA_STATE_ROWS, int HEADS_PER_CTA = 1>
struct GroupStorageHorizontal {
  // Pad DSTATE to next multiple of 32 banks (128 bytes) to eliminate bank conflicts.
  // TMA loads a single wide tile of DSTATE_PAD columns; OOB columns (DSTATE..DSTATE_PAD-1)
  // are skipped by TMA (FILL_NONE) and zeroed in registers at load time.
  static constexpr int BANK_CYCLE_BYTES = 32 * sizeof(uint32_t);  // 128 bytes
  static constexpr int DSTATE_PAD = (DSTATE * (int)sizeof(state_t) + BANK_CYCLE_BYTES - 1) /
                                    BANK_CYCLE_BYTES * BANK_CYCLE_BYTES / (int)sizeof(state_t);

  alignas(128) input_t B[TOKENS_MTP][DSTATE_PAD];
  alignas(128) input_t C[TOKENS_MTP][DSTATE_PAD];
  alignas(128) state_t state_in[NUM_IN_STAGES][TMA_STATE_ROWS * DSTATE_PAD];
  alignas(128) input_t x[HEADS_PER_CTA][TOKENS_MTP][DIM];
  float dt[HEADS_PER_CTA][TOKENS_MTP];
  float out[TOKENS_MTP][DIM];

  barrier_t bar_state_in_empty[NUM_IN_STAGES];
  barrier_t bar_state_in_full[NUM_IN_STAGES];
  barrier_t bar_out_ready;  // sync compute warps before/after epilogue
};

// =============================================================================
// TMA warp: loads B, C, x (once), then pipelines state_in in TMA_STATE_ROWS chunks.
// =============================================================================

template <typename input_t, typename state_t, typename stateIndex_t, bool IS_PAD, int NTOKENS,
          int DIM, int DSTATE, int NUM_IN_STAGES, int TMA_STATE_ROWS, int HEADS_PER_CTA,
          typename SramT>
__device__ __forceinline__ void role_load_horizontal(
    SramT& sram, int lane, SelectiveStateMTPParams const& params, int batch, int base_head,
    int kv_group, int64_t state_batch, CUtensorMap const& tensorState, CUtensorMap const& tensorX,
    CUtensorMap const& tensorB, CUtensorMap const& tensorC) {
  namespace cde = cuda::device::experimental;

  constexpr int numTmaLoads = DIM / TMA_STATE_ROWS;
  static_assert(DIM % TMA_STATE_ROWS == 0, "DIM must be divisible by TMA_STATE_ROWS");

  // ── Issue B/C/X TMA loads targeting bar_state_in_full[0] ─────────────
  // These are merged with the first state tile into a single barrier transaction,
  // eliminating a separate bar_input_full barrier and letting TMA instructions
  // stream back-to-back without serialization.
  constexpr int DSTATE_PAD = SramT::DSTATE_PAD;
  constexpr int bytesBCX = 2 * NTOKENS * DSTATE_PAD * (int)sizeof(input_t) +
                           HEADS_PER_CTA * NTOKENS * DIM * (int)sizeof(input_t);
  if (lane == 0) {
    if constexpr (!IS_PAD) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.B[0][0], &tensorB, 0, kv_group, 0, batch,
                                                    sram.bar_state_in_full[0]);
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.C[0][0], &tensorC, 0, kv_group, 0, batch,
                                                    sram.bar_state_in_full[0]);
#pragma unroll
      for (int h = 0; h < HEADS_PER_CTA; h++) {
        cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.x[h][0][0], &tensorX, 0, base_head + h,
                                                      0, batch, sram.bar_state_in_full[0]);
      }
    }
  }

  // ── Pipeline state_in loads (TMA_STATE_ROWS per transaction) ──────────
  // Single wide TMA load of DSTATE_PAD columns per chunk.
  // OOB padding columns are handled in registers, not smem.
  constexpr int bytesChunk = TMA_STATE_ROWS * DSTATE_PAD * (int)sizeof(state_t);
  uint32_t parity_empty[NUM_IN_STAGES] = {};  // all start at phase 0
#pragma unroll
  for (int h = 0; h < HEADS_PER_CTA; h++) {
    int const head = base_head + h;
    for (int tl = 0; tl < numTmaLoads; tl++) {
      int const slot = tl % NUM_IN_STAGES;

      // Wait for compute to release this slot (tight spin, no NANOSLEEP)
      arrive_and_wait_parity(sram.bar_state_in_empty[slot], parity_empty[slot]);

      if (lane == 0) {
        if constexpr (!IS_PAD) {
          cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state_in[slot][0], &tensorState, 0,
                                                        tl * TMA_STATE_ROWS, head, state_batch,
                                                        sram.bar_state_in_full[slot]);
          int const bytes = (h == 0 && tl == 0) ? bytesBCX + bytesChunk : bytesChunk;
          cuda::device::barrier_arrive_tx(sram.bar_state_in_full[slot], 1, bytes);
        } else {
          cuda::device::barrier_arrive_tx(sram.bar_state_in_full[slot], 1, 0);
        }
      }
    }
  }
}

// =============================================================================
// Compute warp: processes one head with TMA-level pipelining.
// Each TMA chunk (TMA_STATE_ROWS) is processed in sub-passes of ROWS_PER_PASS.
// Same horizontal DSTATE traversal + f32x2 packed SIMD.
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, bool IS_PAD, int NTOKENS, int DIM, int DSTATE, int PHILOX_ROUNDS,
          int NUM_IN_STAGES, int TMA_STATE_ROWS, int HEADS_PER_CTA, typename SramT>
__device__ __forceinline__ void role_update_state_horizontal(SramT& sram, int lane,
                                                             int compute_warp,
                                                             SelectiveStateMTPParams const& params,
                                                             int batch, int base_head,
                                                             int64_t state_batch) {
  constexpr int lanesPerRow = horiz::LANES_PER_ROW;
  constexpr int rowsPerWarp = horiz::ROWS_PER_WARP;
  constexpr int numTmaLoads = DIM / TMA_STATE_ROWS;
  constexpr int subPassesPerTma = TMA_STATE_ROWS / horiz::ROWS_PER_PASS;
  constexpr int DSTATE_PAD = SramT::DSTATE_PAD;
  constexpr int stateValuesPerThread = DSTATE_PAD / lanesPerRow;
  static_assert(DSTATE % lanesPerRow == 0, "DSTATE must be divisible by lanesPerRow");
  static_assert(DIM % TMA_STATE_ROWS == 0, "DIM must be divisible by TMA_STATE_ROWS");
  static_assert(TMA_STATE_ROWS % horiz::ROWS_PER_PASS == 0,
                "TMA_STATE_ROWS must be a multiple of ROWS_PER_PASS");

  constexpr int bankSize = sizeof(uint32_t);
  constexpr int stateValuesPerBank = bankSize / sizeof(state_t);
  constexpr int numBanks = 32;
  constexpr int sramReadsPerThreadPerTile = numBanks / lanesPerRow;
  constexpr int elemsPerTileMember = sramReadsPerThreadPerTile * stateValuesPerBank;
  constexpr int elemsPerTile = elemsPerTileMember * lanesPerRow;
  constexpr int numTiles = stateValuesPerThread / elemsPerTileMember;
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

  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  auto const icache_idx =
      intermediate_state_indices ? (int64_t)intermediate_state_indices[batch] : state_batch;

  // Logical column within DSTATE_PAD (for B/C/state smem access and global store bounds)
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
  uint32_t parity_full[NUM_IN_STAGES] = {};
  uint32_t parity_out_ready = 0;

  // Cooperatively load dt for all heads (global mem, no barrier needed).
  // 4 compute warps × 32 lanes = 128 threads cover HEADS_PER_CTA * NTOKENS values.
  {
    constexpr int totalDtValues = HEADS_PER_CTA * NTOKENS;
    constexpr int numComputeThreads = horiz::NUM_COMPUTE_WARPS_PER_GROUP * warpSize;  // 128
    int const flatThread = compute_warp * warpSize + lane;
    for (int idx = flatThread; idx < totalDtValues; idx += numComputeThreads) {
      int const h = idx / NTOKENS;
      int const step = idx % NTOKENS;
      int const head = base_head + h;
      float dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      float dt_val =
          toFloat(dt_ptr[batch * params.dt_stride_batch + step * params.dt_stride_mtp + head]) +
          dt_bias_val;
      if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
      sram.dt[h][step] = dt_val;
    }
  }

  // B/C/X are merged into bar_state_in_full[0] — no separate wait needed.
  // The first arrive_and_wait(bar_state_in_full[0]) at h=0, tl=0 delivers everything.

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

    // ── TMA load loop: each iteration processes TMA_STATE_ROWS rows of DIM ─
    // Within each TMA chunk, sub-passes of ROWS_PER_PASS are processed
    // without barrier sync (data already in smem). #pragma unroll 1 prevents
    // register pressure increase from unrolling the sub-pass loop.
    for (int tl = 0; tl < numTmaLoads; tl++) {
      int const slot = tl % NUM_IN_STAGES;

      // Wait for TMA to deliver this chunk (tight spin, no NANOSLEEP)
      arrive_and_wait_parity(sram.bar_state_in_full[slot], parity_full[slot]);

      // Process sub-passes within the TMA chunk (NO unroll to avoid register bloat)
#pragma unroll 1
      for (int sp = 0; sp < subPassesPerTma; sp++) {
        int const sram_row = sp * horiz::ROWS_PER_PASS + compute_warp * rowsPerWarp + group;
        int const dd = tl * TMA_STATE_ROWS + sram_row;  // global DIM row

        // Load state from smem; zero padding beyond DSTATE in registers
        float2 rState[numTiles][pairsPerTileMember];
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
#pragma unroll
          for (int p = 0; p < pairsPerTileMember; p++) {
            int const c0 = baseCol(t, p * 2);
            if (c0 >= DSTATE || IS_PAD) {
              rState[t][p] = {0.f, 0.f};
            } else {
              rState[t][p] = toFloat2(&sram.state_in[slot][sram_row * DSTATE_PAD + c0]);
            }
          }
        }

        // Precompute intermediate-state pointer (step=0) and per-step stride.
        // Using a direct pointer avoids 64-bit index add + shift-to-bytes per store.
        int64_t const istate_base_dd = icache_idx * params.intermediate_state_stride_batch +
                                       (int64_t)head * DIM * DSTATE + (int64_t)dd * DSTATE;
        int64_t const istate_step_stride = (int64_t)params.nheads * DIM * DSTATE;
        state_t* __restrict__ istate_dd_ptr = istate_ptr + istate_base_dd;

        // Strength-reduce step-dependent shared memory indexing:
        // replace step * stride multiplies with pointer increments.
        auto const* __restrict__ B_step = &sram.B[0][0];
        auto const* __restrict__ C_step = &sram.C[0][0];
        auto const* __restrict__ x_step = &sram.x[h][0][0];
        float const* __restrict__ dt_step = &sram.dt[h][0];
        float* __restrict__ out_step = &sram.out[0][0];

        for (int step = 0; step < NTOKENS; step++) {
          float const dt_value = *dt_step;
          float const dA = __expf(A_val * dt_value);
          float const x_value = toFloat(x_step[dd]);

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
              mul_f32x2(dBx, B2, dtx2);                         // dBx = B * (dt * x)
              fma_f32x2(rState[t][p], dA2, rState[t][p], dBx);  // state = dA * state + dBx
              fma_f32x2(out2, rState[t][p], C2, out2);          // out += state * C
            }
          }
          float out_value = out2.x + out2.y;

          // Reduce across lanesPerRow adjacent lanes
#pragma unroll
          for (int offset = lanesPerRow / 2; offset >= 1; offset /= 2) {
            out_value += __shfl_down_sync(UINT32_MAX, out_value, offset);
          }

          if (member == 0) {
            out_step[dd] = out_value + D_val * x_value;
          }

          // Advance step pointers (addition instead of multiply)
          B_step += DSTATE_PAD;
          C_step += DSTATE_PAD;
          x_step += DIM;
          dt_step += 1;
          out_step += DIM;

          // Write intermediate state (only real DSTATE columns, not padding)
          if (istate_ptr && !IS_PAD) {
#pragma unroll
            for (int t = 0; t < numTiles; t++) {
              int const col0 = baseCol(t, 0);
              if (col0 < DSTATE) {
                packed_tile_t rOut;
                [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
                for (int e = 0; e < elemsPerTileMember; e += 2) {
                  float const s0 = rState[t][e / 2].x;
                  float const s1 = rState[t][e / 2].y;
                  convertAndStoreSRHorizontal<state_t, DSTATE, PHILOX_ROUNDS>(
                      rOut.val[e], rOut.val[e + 1], s0, s1, rand_seed, state_ptr_offset, dd, col0,
                      e, rand_ints);
                }
                *reinterpret_cast<packed_tile_t*>(&istate_dd_ptr[col0]) = rOut;
              }
            }
            istate_dd_ptr += istate_step_stride;
          }

          // Write final state at last step (only real DSTATE columns, not padding)
          if (step == NTOKENS - 1 && params.update_state && !IS_PAD) {
            auto const state_base =
                state_batch * params.state_stride_batch + head * DIM * DSTATE + dd * DSTATE;
#pragma unroll
            for (int t = 0; t < numTiles; t++) {
              int const col0 = baseCol(t, 0);
              if (col0 < DSTATE) {
                packed_tile_t rOut;
                [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
                for (int e = 0; e < elemsPerTileMember; e += 2) {
                  float const s0 = rState[t][e / 2].x;
                  float const s1 = rState[t][e / 2].y;
                  convertAndStoreSRHorizontal<state_t, DSTATE, PHILOX_ROUNDS>(
                      rOut.val[e], rOut.val[e + 1], s0, s1, rand_seed, state_ptr_offset, dd, col0,
                      e, rand_ints);
                }
                *reinterpret_cast<packed_tile_t*>(&state_ptr[state_base + col0]) = rOut;
              }
            }
          }
        }  // step loop
      }  // sub-pass loop

      // Release state_in slot for TMA to reuse (after all sub-passes done)
      sram.bar_state_in_empty[slot].arrive();
    }  // TMA load loop

    // ── Epilogue: sync all 4 compute warps, z-gate + vectorized store ───
    arrive_and_wait_parity(sram.bar_out_ready, parity_out_ready);

    for (int step = compute_warp; step < NTOKENS; step += horiz::NUM_COMPUTE_WARPS_PER_GROUP) {
      int const out_offset =
          batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM;
      int const z_offset = batch * params.z_stride_batch + step * params.z_stride_mtp + head * DIM;

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

    // Sync compute warps after epilogue before next head reuses sram.out
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
          int PHILOX_ROUNDS, int NUM_IN_STAGES, int TMA_STATE_ROWS, int HEADS_PER_CTA>
__global__ void __launch_bounds__(horiz::NUM_WARPS * 32, 6)
    selective_state_update_kernel_horizontal_mtp(SelectiveStateMTPParams params,
                                                 __grid_constant__ CUtensorMap const tensorState,
                                                 __grid_constant__ CUtensorMap const tensorB,
                                                 __grid_constant__ CUtensorMap const tensorC,
                                                 __grid_constant__ CUtensorMap const tensorX) {
  static_assert(HEADS_PER_GROUP % HEADS_PER_CTA == 0,
                "HEADS_PER_GROUP must be divisible by HEADS_PER_CTA");

  extern __shared__ __align__(128) char smem[];
  using sram_t = GroupStorageHorizontal<input_t, state_t, NTOKENS, DIM, DSTATE, NUM_IN_STAGES,
                                        TMA_STATE_ROWS, HEADS_PER_CTA>;
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
    for (int s = 0; s < NUM_IN_STAGES; s++) {
      // bar_state_in_empty: 4 compute warps + 1 TMA warp
      init(&sram.bar_state_in_empty[s],
           (horiz::NUM_COMPUTE_WARPS_PER_GROUP + horiz::NUM_TMA_WARPS) * warpSize);
      // bar_state_in_full: 1 TMA (arrive_tx, count=1) + 4 compute warps
      init(&sram.bar_state_in_full[s], 1 + horiz::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
    }
    // bar_out_ready: 4 compute warps only
    init(&sram.bar_out_ready, horiz::NUM_COMPUTE_WARPS_PER_GROUP * warpSize);
  }
  __syncthreads();

  // ── Warp role dispatch ─────────────────────────────────────────────────
  auto dispatch = [&]<bool IS_PAD>() {
    if (warp < horiz::NUM_COMPUTE_WARPS_PER_GROUP) {
      role_update_state_horizontal<input_t, state_t, matrixA_t, weight_t, stateIndex_t, IS_PAD,
                                   NTOKENS, DIM, DSTATE, PHILOX_ROUNDS, NUM_IN_STAGES,
                                   TMA_STATE_ROWS, HEADS_PER_CTA>(sram, lane, warp, params, batch,
                                                                  base_head, state_batch);
    } else {
      int const kv_group = base_head / HEADS_PER_GROUP;
      role_load_horizontal<input_t, state_t, stateIndex_t, IS_PAD, NTOKENS, DIM, DSTATE,
                           NUM_IN_STAGES, TMA_STATE_ROWS, HEADS_PER_CTA>(
          sram, lane, params, batch, base_head, kv_group, state_batch, tensorState, tensorX,
          tensorB, tensorC);
    }
  };
  if (is_pad)
    dispatch.template operator()<true>();
  else
    dispatch.template operator()<false>();
}

}  // namespace flashinfer::mamba::mtp
