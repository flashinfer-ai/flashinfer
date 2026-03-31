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
static constexpr int64_t SKIP_WRITE_STATE = -1;
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
// Includes state_in buffer for cp.async prefetch from global memory.
// =============================================================================

template <typename input_t, typename state_t, int NTOKENS, int DIM_PER_CTA, int DSTATE_PAD,
          int ROWS_PER_PASS, int STATE_STAGES>
struct AsyncHorizontalStorage {
  alignas(128) input_t B[NTOKENS][DSTATE_PAD];
  alignas(128) input_t C[NTOKENS][DSTATE_PAD];
  alignas(128) input_t x[NTOKENS][DIM_PER_CTA];
  float dt[NTOKENS];
  float out[NTOKENS][DIM_PER_CTA];
  // Precomputed per-step destination batch indices for state writes.
  // -1 means "skip this step".
  int64_t state_dst_slots[NTOKENS];
  // State prefetch buffer: cp.async loads state here before the barrier.
  // Single stage for DPC=16 (1 pass), 2 stages for DPC>16 (pipelined).
  alignas(128) state_t state_in[STATE_STAGES][ROWS_PER_PASS][DSTATE_PAD];
};

// =============================================================================
// cp.async helpers: 8-byte and 16-byte async copy from global to shared memory.
// =============================================================================

__device__ __forceinline__ void cp_async_16B(void* __restrict__ smem_dst,
                                             void const* __restrict__ gmem_src) {
  unsigned int smem_addr = __cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src)
               : "memory");
}

// =============================================================================
// Cooperative state cp.async: all threads load state via flat 16-byte chunks.
// =============================================================================

template <typename state_t, int DSTATE, int DSTATE_PAD, int ROWS_PER_PASS, int NUM_WARPS,
          typename SramT>
__device__ __forceinline__ void cp_async_state_cooperative(SramT& sram, int lane, int warp,
                                                           int state_stage, int dim_offset,
                                                           state_t const* __restrict__ state_ptr,
                                                           int64_t state_base) {
  constexpr int STATE_PACK = 16 / sizeof(state_t);
  constexpr int state_packs_per_row = DSTATE / STATE_PACK;
  constexpr int num_state_chunks = ROWS_PER_PASS * DSTATE / STATE_PACK;
  int const flat_tid = warp * warpSize + lane;
  constexpr int num_threads = NUM_WARPS * warpSize;

  for (int i = flat_tid; i < num_state_chunks; i += num_threads) {
    int const row = i / state_packs_per_row;
    int const col = (i % state_packs_per_row) * STATE_PACK;
    int const dd = dim_offset + row;
    cp_async_16B(&sram.state_in[state_stage][row][col], &state_ptr[state_base + dd * DSTATE + col]);
  }
}

// =============================================================================
// Cooperative async load: all warps cp.async B, C, x, state into smem.
// dt is loaded via LDG (needs softplus computation).
// =============================================================================

template <typename input_t, typename state_t, typename weight_t, typename stateIndex_t, bool IS_PAD,
          int NTOKENS, int DIM, int DSTATE, int DSTATE_PAD, int DIM_PER_CTA, int ROWS_PER_PASS,
          int NUM_WARPS, typename SramT>
__device__ __forceinline__ void load_async_horizontal(SramT& sram, int lane, int warp,
                                                      SelectiveStateMTPParams const& params,
                                                      int seq_idx, int head, int kv_group,
                                                      int dim_offset, int bos, int seq_len,
                                                      int64_t state_batch, int state_stage) {
  int const flat_tid = warp * warpSize + lane;
  constexpr auto num_threads = NUM_WARPS * warpSize;

  auto const* __restrict__ dt_ptr = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);

  // Varlen: tokens are at (bos + step) with stride_batch; non-varlen: seq_idx with stride_mtp
  bool const has_cu_seqlens = (params.cu_seqlens != nullptr);
  int64_t const B_base = has_cu_seqlens ? (int64_t)bos * params.B_stride_batch
                                        : (int64_t)seq_idx * params.B_stride_batch;
  int64_t const B_tstride = has_cu_seqlens ? params.B_stride_batch : params.B_stride_mtp;
  int64_t const C_base = has_cu_seqlens ? (int64_t)bos * params.C_stride_batch
                                        : (int64_t)seq_idx * params.C_stride_batch;
  int64_t const C_tstride = has_cu_seqlens ? params.C_stride_batch : params.C_stride_mtp;
  int64_t const x_base = has_cu_seqlens ? (int64_t)bos * params.x_stride_batch
                                        : (int64_t)seq_idx * params.x_stride_batch;
  int64_t const x_tstride = has_cu_seqlens ? params.x_stride_batch : params.x_stride_mtp;
  int64_t const dt_base = has_cu_seqlens ? (int64_t)bos * params.dt_stride_batch
                                         : (int64_t)seq_idx * params.dt_stride_batch;
  int64_t const dt_tstride = has_cu_seqlens ? params.dt_stride_batch : params.dt_stride_mtp;

  if constexpr (!IS_PAD) {
    // ── Per-warp cp.async loads to avoid cross-array bank conflicts ──
    // Warps 0/1/2 load B/C/x respectively; all 4 warps cooperatively load state.
    constexpr int INPUT_PACK = 16 / sizeof(input_t);  // 8 for bf16
    constexpr int STATE_PACK = 16 / sizeof(state_t);  // 4 for f32, 8 for f16/bf16
    static_assert(DSTATE % INPUT_PACK == 0, "DSTATE must be divisible by input pack size");
    static_assert(DSTATE % STATE_PACK == 0, "DSTATE must be divisible by state pack size");
    static_assert(DIM_PER_CTA % INPUT_PACK == 0,
                  "DIM_PER_CTA must be divisible by input pack size");

    constexpr int B_packs_per_row = DSTATE / INPUT_PACK;
    constexpr int C_packs_per_row = DSTATE / INPUT_PACK;
    constexpr int x_packs_per_row = DIM_PER_CTA / INPUT_PACK;
    constexpr int state_packs_per_row = DSTATE / STATE_PACK;

    // Warp 0: load B[step][DSTATE] → sram.B[step][DSTATE_PAD]
    if (warp == 0) {
      auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
      constexpr int num_B_chunks = NTOKENS * DSTATE / INPUT_PACK;
      for (int i = lane; i < num_B_chunks; i += warpSize) {
        int const step = i / B_packs_per_row;
        int const col = (i % B_packs_per_row) * INPUT_PACK;
        if (step < seq_len) {
          cp_async_16B(&sram.B[step][col],
                       &B_ptr[B_base + step * B_tstride + kv_group * DSTATE + col]);
        }
      }
    }

    // Warp 1: load C[step][DSTATE] → sram.C[step][DSTATE_PAD]
    if (warp == 1) {
      auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
      constexpr int num_C_chunks = NTOKENS * DSTATE / INPUT_PACK;
      for (int i = lane; i < num_C_chunks; i += warpSize) {
        int const step = i / C_packs_per_row;
        int const col = (i % C_packs_per_row) * INPUT_PACK;
        if (step < seq_len) {
          cp_async_16B(&sram.C[step][col],
                       &C_ptr[C_base + step * C_tstride + kv_group * DSTATE + col]);
        }
      }
    }

    // All warps load x: each warp handles different tokens to avoid bank conflicts.
    // With DPC=16 bf16, each row is 32 bytes (8 banks). One row per warp per iteration
    // means zero bank aliasing within any warp-wide cp.async instruction.
    {
      auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
      for (int step = warp; step < seq_len; step += NUM_WARPS) {
        for (int col = lane * INPUT_PACK; col < DIM_PER_CTA; col += warpSize * INPUT_PACK) {
          cp_async_16B(&sram.x[step][col],
                       &x_ptr[x_base + step * x_tstride + head * DIM + dim_offset + col]);
        }
      }
    }

    // All 4 warps: tiled state load (bank-conflict-free, 8-byte cp.async)
    {
      auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
      auto const state_base = state_batch * params.state_stride_batch + head * DIM * DSTATE;
      cp_async_state_cooperative<state_t, DSTATE, DSTATE_PAD, ROWS_PER_PASS, NUM_WARPS>(
          sram, lane, warp, state_stage, dim_offset, state_ptr, state_base);
    }

    // Load dt[step] via LDG (needs softplus computation, can't use cp.async)
    if (flat_tid < seq_len) {
      int const step = flat_tid;
      float dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      float dt_val = toFloat(dt_ptr[dt_base + step * dt_tstride + head]);
      dt_val += dt_bias_val;
      if (params.dt_softplus) dt_val = thresholded_softplus(dt_val);
      sram.dt[step] = dt_val;
    }
  }

  // Precompute per-step destination slot indices for state writes.
  // Three mutually exclusive modes:
  //   1. dst_state_batch_indices → varlen: prefetch per-step indices (or -1 for pad_slot_id)
  //   2. intermediate_states    → MTP cache: consecutive slots within icache entry
  //   3. neither                → only write final state at last step
  if (flat_tid < NTOKENS) {
    int const step = flat_tid;
    constexpr int64_t SKIP = async_horiz::SKIP_WRITE_STATE;
    auto const* __restrict__ dst_state_batch_indices =
        reinterpret_cast<stateIndex_t const*>(params.dst_state_batch_indices);
    if (IS_PAD || step >= seq_len) {
      sram.state_dst_slots[step] = SKIP;
    } else if (dst_state_batch_indices) {
      // Varlen: read per-step destination index, mark pad slots as SKIP
      auto const dst_idx = static_cast<int64_t>(
          dst_state_batch_indices[seq_idx * params.dst_state_batch_indices_stride_batch +
                                  step * params.dst_state_batch_indices_stride_T]);
      sram.state_dst_slots[step] = (dst_idx == params.pad_slot_id) ? SKIP : dst_idx;
    } else if (params.intermediate_states) {
      // MTP cache: consecutive step slots within the icache entry
      auto const* __restrict__ intermediate_state_indices =
          reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
      auto const icache_idx = intermediate_state_indices
                                  ? static_cast<int64_t>(intermediate_state_indices[seq_idx])
                                  : state_batch;
      sram.state_dst_slots[step] = icache_idx * params.cache_steps + step;
    } else {
      // Final-state only: write at last step
      sram.state_dst_slots[step] =
          (step == seq_len - 1 && params.update_state) ? state_batch : SKIP;
    }
  }

  // Commit all cp.async and wait for completion
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// =============================================================================
// State update: horizontal DSTATE traversal with state in registers.
// =============================================================================

template <typename input_t, typename state_t, typename matrixA_t, typename weight_t,
          typename stateIndex_t, typename state_scale_t, bool IS_PAD, int NTOKENS, int DIM,
          int DSTATE, int DSTATE_PAD, int DIM_PER_CTA, int PHILOX_ROUNDS, int NUM_WARPS,
          typename SramT>
__device__ __forceinline__ void update_state_async_horizontal(
    SramT& sram, int lane, int warp, SelectiveStateMTPParams const& params, int seq_idx, int head,
    int dim_offset, int64_t state_batch, int bos, int seq_len, float A_val, float D_val) {
  constexpr bool scaleState = !std::is_same_v<state_scale_t, void>;
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

  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  auto* __restrict__ state_ptr = reinterpret_cast<state_t*>(params.state);

  // Unified state write path: pick destination pointer and stride based on mode.
  // The per-step slot indices are precomputed in sram.state_dst_slots[].
  // For istate: dst_slot = icache_idx * cache_steps + step, so the flat slot
  // index works with both state stride (nheads*DIM*DSTATE) and scale stride (nheads*DIM).
  auto* __restrict__ write_state_ptr = state_ptr;
  int64_t write_state_stride = params.state_stride_batch;
  [[maybe_unused]] auto* __restrict__ write_scale_ptr =
      reinterpret_cast<float*>(params.state_scale);
  [[maybe_unused]] int64_t write_scale_stride = params.state_scale_stride_batch;
  if (params.intermediate_states) {
    write_state_ptr = reinterpret_cast<state_t*>(params.intermediate_states);
    write_state_stride = (int64_t)params.nheads * DIM * DSTATE;
    write_scale_ptr = reinterpret_cast<float*>(params.intermediate_state_scales);
    write_scale_stride = (int64_t)params.nheads * DIM;
  }

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

  // State scale pointer (only used when scaleState == true)
  [[maybe_unused]] auto* __restrict__ state_scale_ptr =
      reinterpret_cast<state_scale_t*>(params.state_scale);

  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;

  constexpr int numPasses = DIM_PER_CTA / ROWS_PER_PASS;

  constexpr int STATE_STAGES = (numPasses == 1) ? 1 : 2;

  for (int pass = 0; pass < numPasses; pass++) {
    int const pass_row = warp * rowsPerWarp + group;  // row within current pass [0, ROWS_PER_PASS)
    int const local_row = pass * ROWS_PER_PASS + pass_row;
    int const dd = dim_offset + local_row;  // global DIM index
    int const stage = pass % STATE_STAGES;

    // Load state from smem (prefetched via cp.async) into registers
    float2 rState[numTiles][pairsPerTileMember];

    // Load decode scale for this DIM row (scaleState only)
    [[maybe_unused]] float state_decode_scale = 1.f;
    if constexpr (scaleState) {
      if constexpr (!IS_PAD) {
        state_decode_scale = toFloat(
            state_scale_ptr[state_batch * params.state_scale_stride_batch + head * DIM + dd]);
      }
    }

#pragma unroll
    for (int t = 0; t < numTiles; t++) {
#pragma unroll
      for (int p = 0; p < pairsPerTileMember; p++) {
        int const c0 = baseCol(t, p * 2);
        if (c0 >= DSTATE || IS_PAD) {
          rState[t][p] = {0.f, 0.f};
        } else {
          rState[t][p] = toFloat2(&sram.state_in[stage][pass_row][c0]);
          if constexpr (scaleState) {
            rState[t][p].x *= state_decode_scale;
            rState[t][p].y *= state_decode_scale;
          }
        }
      }
    }

    // Strength-reduce step-dependent shared memory indexing
    auto const* __restrict__ B_step = &sram.B[0][0];
    auto const* __restrict__ C_step = &sram.C[0][0];
    auto const* __restrict__ x_step = &sram.x[0][0];
    float const* __restrict__ dt_step = &sram.dt[0];
    float* __restrict__ out_step = &sram.out[0][0];

    for (int step = 0; step < NTOKENS; step++) {
      if (step >= seq_len) break;
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

      // Unified state write: use precomputed slot index from sram
      {
        int64_t const dst_slot = sram.state_dst_slots[step];
        if (dst_slot != async_horiz::SKIP_WRITE_STATE) {
          [[maybe_unused]] float encode_scale = 1.f;
          if constexpr (scaleState) {
            encode_scale =
                computeBlockScaleEncode<state_t, numTiles, pairsPerTileMember, lanesPerRow,
                                        elemsPerTile, elemsPerTileMember, DSTATE>(rState, lane,
                                                                                  member);
          }
          auto const dst_base =
              dst_slot * write_state_stride + (int64_t)head * DIM * DSTATE + (int64_t)dd * DSTATE;
#pragma unroll
          for (int t = 0; t < numTiles; t++) {
            int const col0 = baseCol(t, 0);
            if (col0 >= DSTATE) continue;
            packed_tile_t rOut;
            [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
            for (int e = 0; e < elemsPerTileMember; e += 2) {
              float2 s2 = rState[t][e / 2];
              if constexpr (scaleState) {
                float2 const scale2 = {encode_scale, encode_scale};
                mul_f32x2(s2, s2, scale2);
              }
              convertAndStoreSRHorizontal<state_t, DSTATE, PHILOX_ROUNDS>(
                  rOut.val[e], rOut.val[e + 1], s2.x, s2.y, rand_seed, state_ptr_offset, dd, col0,
                  e, rand_ints);
            }
            *reinterpret_cast<packed_tile_t*>(&write_state_ptr[dst_base + col0]) = rOut;
          }
          // Write decode scale
          if constexpr (scaleState) {
            if (member == 0) {
              write_scale_ptr[dst_slot * write_scale_stride + head * DIM + dd] = 1.f / encode_scale;
            }
          }
        }
      }
    }  // step loop

    // Multi-pass pipeline: prefetch next pass's state into the other smem stage
    if constexpr (numPasses > 1) {
      if (pass < numPasses - 1) {
        int const next_stage = (pass + 1) % STATE_STAGES;
        int const next_dim_base = dim_offset + (pass + 1) * ROWS_PER_PASS;

        if constexpr (!IS_PAD) {
          auto const* __restrict__ state_ptr_r = reinterpret_cast<state_t const*>(params.state);
          auto const state_base = state_batch * params.state_stride_batch + head * DIM * DSTATE;
          cp_async_state_cooperative<state_t, DSTATE, DSTATE_PAD, ROWS_PER_PASS, NUM_WARPS>(
              sram, lane, warp, next_stage, next_dim_base, state_ptr_r, state_base);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
      }
    }
  }  // pass loop

  // ── Epilogue: sync all warps, z-gate + vectorized store ───
  __syncthreads();

  // Varlen output indexing: (bos + step) * stride_batch; non-varlen: seq_idx * stride_batch + step
  // * stride_mtp
  bool const has_cu_seqlens = (params.cu_seqlens != nullptr);
  auto out_addr = [&](int step) -> int64_t {
    return has_cu_seqlens
               ? (int64_t)(bos + step) * params.out_stride_batch + head * DIM + dim_offset
               : (int64_t)seq_idx * params.out_stride_batch + step * params.out_stride_mtp +
                     head * DIM + dim_offset;
  };
  auto z_addr = [&](int step) -> int64_t {
    return has_cu_seqlens ? (int64_t)(bos + step) * params.z_stride_batch + head * DIM + dim_offset
                          : (int64_t)seq_idx * params.z_stride_batch + step * params.z_stride_mtp +
                                head * DIM + dim_offset;
  };

  if constexpr (DIM_PER_CTA >= warpSize) {
    // Fast path: each lane handles >= 1 element, use vectorized loads/stores
    constexpr int elemsPerThreadEpilogue = DIM_PER_CTA / warpSize;

    for (int step = warp; step < seq_len; step += NUM_WARPS) {
      int64_t const out_offset = out_addr(step);
      int64_t const z_offset = z_addr(step);

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
    for (int step = warp; step < seq_len; step += NUM_WARPS) {
      if (lane < DIM_PER_CTA) {
        int64_t const out_offset = out_addr(step);
        float out_value = sram.out[step][lane];
        if (z_ptr) {
          int64_t const z_offset = z_addr(step);
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
// Grid: (batch_or_n_sequences, nheads, CTAS_PER_HEAD)
// Block: (32, NUM_WARPS)
// =============================================================================

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NTOKENS, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int CTAS_PER_HEAD>
__global__ void __launch_bounds__(NUM_WARPS * 32)
    selective_state_update_kernel_async_horizontal_mtp(SelectiveStateMTPParams params) {
  constexpr int DSTATE_PAD = padDstate<input_t>(DSTATE);
  constexpr int DIM_PER_CTA = DIM / CTAS_PER_HEAD;
  constexpr int ROWS_PER_PASS = NUM_WARPS * async_horiz::ROWS_PER_WARP;

  static_assert(DIM % CTAS_PER_HEAD == 0, "DIM must be divisible by CTAS_PER_HEAD");
  static_assert(DIM_PER_CTA % ROWS_PER_PASS == 0, "DIM_PER_CTA must be divisible by ROWS_PER_PASS");

  constexpr int numPasses = DIM_PER_CTA / ROWS_PER_PASS;
  constexpr int STATE_STAGES = (numPasses == 1) ? 1 : 2;

  extern __shared__ __align__(128) char smem[];
  using sram_t = AsyncHorizontalStorage<input_t, state_t, NTOKENS, DIM_PER_CTA, DSTATE_PAD,
                                        ROWS_PER_PASS, STATE_STAGES>;
  auto& sram = *reinterpret_cast<sram_t*>(smem);

  int const seq_idx = blockIdx.x;
  int const head = blockIdx.y;
  int const cta_z = blockIdx.z;
  int const dim_offset = cta_z * DIM_PER_CTA;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;

  // ── Varlen: compute bos, seq_len ──
  auto const* __restrict__ cu_seqlens =
      reinterpret_cast<cuSeqlensIndex_t const*>(params.cu_seqlens);
  int bos;
  int seq_len;
  if (cu_seqlens) {
    bos = __ldg(&cu_seqlens[seq_idx]);
    int const eos = __ldg(&cu_seqlens[seq_idx + 1]);
    seq_len = eos - bos;
    if (seq_len <= 0) return;
  } else {
    bos = 0;
    seq_len = NTOKENS;
  }

  // ── num_accepted_tokens → init_token_idx ──
  auto const* __restrict__ num_accepted_tokens =
      reinterpret_cast<numAcceptedIndex_t const*>(params.num_accepted_tokens);
  int init_token_idx = 0;
  if (num_accepted_tokens) {
    int num_accepted = __ldg(&num_accepted_tokens[seq_idx]);
    init_token_idx = max(num_accepted - 1, 0);
  }

  // ── State batch index: 2D (seq_idx, init_token_idx) or 1D ──
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t state_batch;
  if (state_batch_indices) {
    state_batch = static_cast<int64_t>(
        state_batch_indices[seq_idx * params.state_batch_indices_stride_batch +
                            init_token_idx * params.state_batch_indices_stride_T]);
  } else {
    state_batch = static_cast<int64_t>(seq_idx);
  }
  bool const is_pad = (state_batch == (int64_t)params.pad_slot_id);

  int const kv_group = head / HEADS_PER_GROUP;

  // Load A and D before the barrier so global memory latency overlaps with barrier wait
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  auto run = [&]<bool IS_PAD>() {
    // Phase 1: cooperative cp.async load of B/C/x/dt/state into smem
    load_async_horizontal<input_t, state_t, weight_t, stateIndex_t, IS_PAD, NTOKENS, DIM, DSTATE,
                          DSTATE_PAD, DIM_PER_CTA, ROWS_PER_PASS, NUM_WARPS>(
        sram, lane, warp, params, seq_idx, head, kv_group, dim_offset, bos, seq_len, state_batch,
        /*state_stage=*/0);

    // Phase 2: single sync — ensures all smem writes (cp.async + LDG dt) are visible
    __syncthreads();

    // Phase 3: compute (state in registers, B/C/x from smem)
    update_state_async_horizontal<input_t, state_t, matrixA_t, weight_t, stateIndex_t,
                                  state_scale_t, IS_PAD, NTOKENS, DIM, DSTATE, DSTATE_PAD,
                                  DIM_PER_CTA, PHILOX_ROUNDS, NUM_WARPS>(
        sram, lane, warp, params, seq_idx, head, dim_offset, state_batch, bos, seq_len, A_val,
        D_val);
  };

  if (is_pad)
    run.template operator()<true>();
  else
    run.template operator()<false>();
}

}  // namespace flashinfer::mamba::mtp
