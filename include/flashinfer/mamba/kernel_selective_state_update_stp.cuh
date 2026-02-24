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
// #ifndef FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
// #define FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cuda/barrier>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"

namespace flashinfer::mamba {

using namespace conversion;

// Computes a conflict-free column index for shared memory access.
// This permutation avoids bank conflicts when threads access strided patterns.
//
// Without permutation (baseCol directly):
//   Thread 0 -> Bank 0, Thread 32 -> Bank 0, Thread 64 -> Bank 0  (conflict!)
//
// With permutation (adding bankCycle offset):
//   bankCycle = which "round" of 32 banks we're in
//   By offsetting each round by 1 bank:
//   Thread 0  -> Bank 0
//   Thread 32 -> Bank 1  (offset by 1)
//   Thread 64 -> Bank 2  (offset by 2)
//
// Visual: (stateValuesPerBank=1, numBanks=32, colsPerStage=128)
//   baseCol:    0  1  2 ... 31 | 32 33 34 ... 63 | 64 ...
//   bankCycle:  0  0  0 ...  0 |  1  1  1 ...  1 |  2 ...
//   ii:         0  1  2 ... 31 | 33 34 35 ... 64 | 66 ...  (mod colsPerStage)
template <int colsPerStage, int stateValuesPerBank, int numBanks>
__device__ __forceinline__ int conflict_free_column(int group, int baseCol) {
  auto const seq_index = group * colsPerStage + baseCol;
  auto const bankCycle = (seq_index / stateValuesPerBank) / numBanks;
  return (baseCol + stateValuesPerBank * bankCycle) % colsPerStage;
}

template <typename input_t, int rows_per_block, int dstate>
struct SharedStorageSimple {
  alignas(alignof(PackedAligned<input_t>)) input_t x[rows_per_block];
  alignas(alignof(PackedAligned<input_t>)) input_t z[rows_per_block];
  alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];
  float out[rows_per_block];
};

// Grid: (batch, nheads, cdiv(DIM, ROWS_PER_BLOCK))
// When ROWS_PER_BLOCK == DIM, degenerates to the non-tiled case (blockIdx.z == 0 always).
// Used when batch*nheads is too small to saturate the GPU: set ROWS_PER_BLOCK < DIM to
// split dim across blocks for better occupancy.
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int DIM, int DSTATE, int ROWS_PER_BLOCK, int numWarps>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);

  auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  bool const dt_softplus = params.dt_softplus;

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  constexpr auto rowsPerWarp = (ROWS_PER_BLOCK + numWarps - 1) / numWarps;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const dim_offset = blockIdx.z * ROWS_PER_BLOCK;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
  state += state_batch * params.state_stride_batch + head * DIM * DSTATE;

  __shared__ SharedStorageSimple<input_t, ROWS_PER_BLOCK, DSTATE> sram;

  static constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;
  using load_input_t = PackedAligned<input_t>;

  auto const A_value = toFloat(A[head]);

  auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
  if (dt_bias) dt_value += toFloat(dt_bias[head]);
  if (dt_softplus) {
    dt_value = thresholded_softplus(dt_value);
  }

  auto const dA = __expf(A_value * dt_value);

  auto d_value = D ? toFloat(D[head]) : 0.f;

  // Load x slice and B (warp 0), z slice and C (warp 1)
  if (warp == 0) {
    for (auto d = lane; d < ROWS_PER_BLOCK; d += warpSize) {
      if (dim_offset + d < DIM)
        sram.x[d] = x[batch * params.x_stride_batch + head * DIM + dim_offset + d];
    }
    for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.B[i]);
      *dst = *reinterpret_cast<load_input_t const*>(
          &B[batch * params.B_stride_batch + group * DSTATE + i]);
    }
  } else if (warp == 1) {
    for (auto d = lane; d < ROWS_PER_BLOCK; d += warpSize) {
      if (dim_offset + d < DIM)
        sram.z[d] = z ? z[batch * params.z_stride_batch + head * DIM + dim_offset + d] : input_t(0);
    }
    for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.C[i]);
      *dst = *reinterpret_cast<load_input_t const*>(
          &C[batch * params.C_stride_batch + group * DSTATE + i]);
    }
  }
  __syncthreads();

  for (auto _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++) {
    auto d = dim_offset + _d;
    if (d >= DIM) break;

    float x_value = toFloat(sram.x[_d]);
    float out_value = d_value * x_value * int(lane == 0);

    for (int i = lane * load_state_t::count; i < DSTATE; i += warpSize * load_state_t::count) {
      auto rState = make_zeros<load_state_t>();
      if (state_batch != params.pad_slot_id)
        rState = *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]);

      for (int ii = 0; ii < load_state_t::count; ii++) {
        auto state_value = toFloat(rState.val[ii]);
        auto B_value = toFloat(sram.B[i + ii]);
        auto C_value = toFloat(sram.C[i + ii]);

        auto const dB = B_value * dt_value;
        auto const new_state = state_value * dA + dB * x_value;

        convertAndStore(&rState.val[ii], new_state);

        out_value += new_state * C_value;
      }
      if (params.update_state && state_batch != params.pad_slot_id)
        *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]) = rState;
    }

    out_value = warpReduceSum(out_value);
    if (lane == 0) {
      sram.out[_d] = out_value;
    }
  }

  __syncthreads();

  for (int l = lane; l < rowsPerWarp; l += warpSize) {
    auto _d = warp * rowsPerWarp + l;
    auto d = dim_offset + _d;
    if (d < DIM) {
      auto out_value = sram.out[_d];
      if (z) {
        float z_value = toFloat(sram.z[_d]);
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
      }
      convertAndStore(&output[batch * params.out_stride_batch + head * DIM + d], out_value);
    }
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          int rowsPerStage, int dim, int dstate, uint8_t numStages>
struct SharedStorageVertical {
  alignas(128) state_t state[numStages][rowsPerStage * dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t x[dim];
  alignas(alignof(PackedAligned<input_t>)) input_t z[dim];
  alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];
  float out[dim];  // dt is special cause we're gonna store input in there as well

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  barrier_t bar_empty[numStages];
  barrier_t bar_full[numStages];
  barrier_t bar_consumers;
};

template <typename input_t, typename state_t, int DIM, int DSTATE, int rowsPerStage, int numStages,
          bool readState, bool writeState, bool hasZ, typename SramT>
__device__ __forceinline__ void producer_func_vertical(SramT& sram, CUtensorMap const& tensorState,
                                                       input_t const* x_global_ptr,
                                                       input_t const* B_global_ptr,
                                                       input_t const* C_global_ptr,
                                                       input_t const* z_global_ptr, int batch,
                                                       int head) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  namespace cde = cuda::device::experimental;

  auto constexpr stagesReadOnly = numStages;
  auto constexpr stagesBoth = DIM / rowsPerStage - numStages;
  auto constexpr stagesWriteOnly = numStages;

  auto constexpr bytesState = rowsPerStage * DSTATE * sizeof(state_t);
  auto constexpr bytesX = DIM * sizeof(input_t);
  auto constexpr bytesB = DSTATE * sizeof(input_t);
  auto constexpr bytesC = DSTATE * sizeof(input_t);
  auto constexpr bytesZ = hasZ ? DIM * sizeof(input_t) : 0;
  auto constexpr bytesInputs = bytesX + bytesB + bytesC + bytesZ;

  // Phase 1, iter 0: fire all input vector loads + state load (if readState)
  // All inputs piggyback onto bar_full[0] so consumers get them before stage 0
  {
    constexpr auto stage = 0;
    constexpr auto d = 0;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    cuda::device::memcpy_async_tx(&sram.x[0], x_global_ptr, cuda::aligned_size_t<16>(bytesX),
                                  sram.bar_full[stage]);
    cuda::device::memcpy_async_tx(&sram.B[0], B_global_ptr, cuda::aligned_size_t<16>(bytesB),
                                  sram.bar_full[stage]);
    cuda::device::memcpy_async_tx(&sram.C[0], C_global_ptr, cuda::aligned_size_t<16>(bytesC),
                                  sram.bar_full[stage]);
    if constexpr (hasZ) {
      cuda::device::memcpy_async_tx(&sram.z[0], z_global_ptr, cuda::aligned_size_t<16>(bytesZ),
                                    sram.bar_full[stage]);
    }

    if constexpr (readState) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, 0, d, head,
                                                    batch, sram.bar_full[stage]);
      auto const _ =
          cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesState + bytesInputs);
    } else {
      auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesInputs);
    }
  }

  // Phase 1, iter 1..stagesReadOnly-1: state only (x already in flight)
#pragma unroll
  for (int iter = 1; iter < stagesReadOnly; ++iter) {
    auto const stage = iter % numStages;
    auto const d = iter * rowsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (readState) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, 0, d, head,
                                                    batch, sram.bar_full[stage]);
      auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesState);
    } else {
      auto const _ = sram.bar_full[stage].arrive();
    }
  }

  // Phase 2: Both read and write (steady state)
#pragma unroll
  for (int iter = 0; iter < stagesBoth; ++iter) {
    auto const stage = (stagesReadOnly + iter) % numStages;
    auto const d_read = (stagesReadOnly + iter) * rowsPerStage;
    auto const d_write = iter * rowsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (readState || writeState) {
      // Unblock async proxy for writeback
      cde::fence_proxy_async_shared_cta();
      // Writeback
      if constexpr (writeState) {
        cde::cp_async_bulk_tensor_4d_shared_to_global(&tensorState, 0, d_write, head, batch,
                                                      &sram.state[stage][0]);

        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
      }

      // Read next
      if constexpr (readState) {
        cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, 0,
                                                      d_read, head, batch, sram.bar_full[stage]);
        auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesState);
      } else {
        auto const _ = sram.bar_full[stage].arrive();
      }
    } else {
      auto const _ = sram.bar_full[stage].arrive();
    }
  }

  // Phase 3: Write only (draining the pipeline)
#pragma unroll
  for (int iter = 0; iter < stagesWriteOnly; ++iter) {
    auto const stage = (stagesReadOnly + stagesBoth + iter) % numStages;
    auto const d_write = (stagesBoth + iter) * rowsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (writeState) {
      // Unblock async proxy for writeback
      cde::fence_proxy_async_shared_cta();
      cde::cp_async_bulk_tensor_4d_shared_to_global(&tensorState, 0, d_write, head, batch,
                                                    &sram.state[stage][0]);

      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
    }
  }
#endif
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM,
          int DSTATE, int consumerWarps, int rowsPerStage, int numStages, bool useStateCache>
__device__ __forceinline__ void consumer_func_vertical(
    int lane, int warp, float d_value, float dt_value, float dA,
    SharedStorageVertical<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE,
                          numStages>& sram) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  namespace cde = cuda::device::experimental;
  for (auto dBegin = 0, stage = 0; dBegin < DIM;
       dBegin += rowsPerStage, stage = (stage + 1) % numStages) {
    // wait for the producer
    sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

#pragma unroll
    for (auto dd = warp; dd < rowsPerStage; dd += consumerWarps) {
      auto d = dBegin + dd;
      float const x_value = toFloat(sram.x[d]);
      float out_value = d_value * x_value * int(lane == 0);  // first lane has the value

      constexpr auto bankSize = sizeof(uint32_t);
      constexpr auto stateValuesPerBank = bankSize / sizeof(state_t);

      if constexpr (sizeof(state_t) == sizeof(input_t)) {
        for (int i = lane * stateValuesPerBank; i < DSTATE; i += warpSize * stateValuesPerBank) {
          auto* sState_ptr = reinterpret_cast<uint32_t*>(&sram.state[stage][dd * DSTATE + i]);
          uint32_t rState = *sState_ptr;
          auto* rState_ptr = reinterpret_cast<state_t*>(&rState);

          uint32_t rB = *reinterpret_cast<uint32_t const*>(&sram.B[i]);
          auto* rB_ptr = reinterpret_cast<input_t const*>(&rB);

          uint32_t rC = *reinterpret_cast<uint32_t const*>(&sram.C[i]);
          auto* rC_ptr = reinterpret_cast<input_t const*>(&rC);

          for (int e = 0; e < stateValuesPerBank; e++) {
            float state_value;
            if constexpr (!useStateCache) {
              state_value = 0.f;
            } else {
              state_value = toFloat(rState_ptr[e]);
            }
            auto const B_value = toFloat(rB_ptr[e]);
            auto const C_value = toFloat(rC_ptr[e]);

            auto const dB = B_value * dt_value;
            auto const new_state = state_value * dA + dB * x_value;

            convertAndStore(&rState_ptr[e], new_state);
            out_value += new_state * C_value;
          }
          *sState_ptr = rState;
        }
      } else {
        for (int i = lane * stateValuesPerBank; i < DSTATE; i += warpSize * stateValuesPerBank) {
          auto* sState_ptr = reinterpret_cast<uint32_t*>(&sram.state[stage][dd * DSTATE + i]);
          uint32_t rState = *sState_ptr;
          auto* rState_ptr = reinterpret_cast<state_t*>(&rState);

          for (int e = 0; e < stateValuesPerBank; e++) {
            float state_value;
            if constexpr (!useStateCache) {
              state_value = 0.f;
            } else {
              state_value = toFloat(rState_ptr[e]);
            }
            auto const B_value = toFloat(sram.B[i + e]);
            auto const C_value = toFloat(sram.C[i + e]);
            auto const dB = B_value * dt_value;
            auto const new_state = state_value * dA + dB * x_value;

            convertAndStore(&rState_ptr[e], new_state);
            out_value += new_state * C_value;
          }
          *sState_ptr = rState;
        }
      }

      out_value = warpReduceSum(out_value);
      if (lane == 0) {
        sram.out[d] = out_value;
      }
    }

    // Unblock producer
    cde::fence_proxy_async_shared_cta();
    auto _ = sram.bar_empty[stage].arrive();
  }
#endif
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int DIM, int DSTATE, int consumerWarps, int rowsPerStage,
          int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer_vertical(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);

  auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  constexpr auto numWarps = 1 + consumerWarps;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? __ldg(&state_batch_indices[batch]) : batch;

  extern __shared__ uint8_t sbuffer[];
  using sram_t = SharedStorageVertical<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM,
                                       DSTATE, numStages>;
  auto& sram = *reinterpret_cast<sram_t*>(sbuffer);

  namespace cde = cuda::device::experimental;
  namespace cg = cooperative_groups;

  for (int stage = warp; stage < numStages; stage += numWarps) {
    if (lane > 0) continue;
    constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
    init(&sram.bar_empty[stage], num_arrivals);
    init(&sram.bar_full[stage], num_arrivals);
    // signal to async proxy that barriers are initilized
    cde::fence_proxy_async_shared_cta();
  }
  if (lane == 0 && warp == 0) {
    init(&sram.bar_consumers, warpSize * consumerWarps);
  }
  __syncthreads();

  if (warp == consumerWarps)  // producer
  {
    // auto const state_offset = (state_batch * nheads + head) * DIM;
    auto const read_state = (state_batch != params.pad_slot_id);
    auto const write_state = read_state && params.update_state;

    if (lane == 0) {
      auto const* x_global_ptr = &x[batch * params.x_stride_batch + head * DIM];
      auto const* B_global_ptr = &B[batch * params.B_stride_batch + group * DSTATE];
      auto const* C_global_ptr = &C[batch * params.C_stride_batch + group * DSTATE];
      auto const* z_global_ptr = z ? &z[batch * params.z_stride_batch + head * DIM] : nullptr;
      auto const call = [&]<bool readState, bool writeState, bool hasZ>() {
        producer_func_vertical<input_t, state_t, DIM, DSTATE, rowsPerStage, numStages, readState,
                               writeState, hasZ>(sram, tensorState, x_global_ptr, B_global_ptr,
                                                 C_global_ptr, hasZ ? z_global_ptr : nullptr,
                                                 state_batch, head);
      };
      auto const dispatch_state = [&]<bool hasZ>() {
        if (read_state && write_state)
          call.template operator()<true, true, hasZ>();
        else if (read_state)
          call.template operator()<true, false, hasZ>();
        else
          call.template operator()<false, false, hasZ>();
      };

      cg::invoke_one(cg::coalesced_threads(), [&]() {
        if (z_global_ptr)
          dispatch_state.template operator()<true>();
        else
          dispatch_state.template operator()<false>();
      });
    }
  } else {  // consumers

#pragma unroll
    // Unblock the producer
    for (uint8_t stage = 0; stage < numStages; ++stage) {
      auto const _ = sram.bar_empty[stage].arrive();
    }

    // Load A, D, dt, dt_bias via __ldg (read-only texture cache) —
    // these are broadcast scalars read once per block.
    auto const A_value = toFloat(__ldg(&A[head]));

    auto const d_value = D ? toFloat(__ldg(&D[head])) : 0.f;

    auto dt_value = toFloat(__ldg(&dt[batch * params.dt_stride_batch + head]));
    if (dt_bias) dt_value += toFloat(__ldg(&dt_bias[head]));
    if (params.dt_softplus) {
      dt_value = thresholded_softplus(dt_value);
    }
    auto const dA = __expf(A_value * dt_value);

    if (state_batch != params.pad_slot_id)
      consumer_func_vertical<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                             rowsPerStage, numStages, true>(lane, warp, d_value, dt_value, dA,
                                                            sram);
    else
      consumer_func_vertical<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                             rowsPerStage, numStages, false>(lane, warp, d_value, dt_value, dA,
                                                             sram);

    // Write output — wait for all consumer warps to finish writing sram.out
    sram.bar_consumers.wait(sram.bar_consumers.arrive());
    auto d = warp * warpSize + lane;
    if (d < DIM) {
      auto out_value = sram.out[d];
      if (z) {
        float z_value = toFloat(sram.z[d]);
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
      }
      convertAndStore(&output[batch * params.out_stride_batch + head * DIM + d], out_value);
    }
  }
#endif
}

#ifdef FLASHINFER_MAMBA_ENABLE_SM90

template <typename input_t, typename weight_t, typename matrixA_t,
          typename state_t,  //
          int dim, int dstate, int stageCols, uint8_t numStages>
struct SharedStorageHorizontal {
  alignas(128) state_t state[numStages][dim * stageCols];
  alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  barrier_t bar_empty[numStages];
  barrier_t bar_full[numStages];
  barrier_t bar_consumers;
};

template <typename state_t, int DIM, int DSTATE, int colsPerStage, int numStages, bool readState,
          bool writeState, typename SramT>
__device__ __forceinline__ void producer_func_horizontal(SramT& sram,
                                                         CUtensorMap const& tensorState, int batch,
                                                         int head) {
  namespace cde = cuda::device::experimental;

  auto constexpr stagesReadOnly = numStages;
  auto constexpr stagesBoth = DSTATE / colsPerStage - numStages;
  auto constexpr stagesWriteOnly = numStages;

  auto constexpr bytesState = DIM * colsPerStage * sizeof(state_t);
  auto constexpr bytesToArrive = bytesState;

  // Phase 1: Read only (filling the pipeline)
#pragma unroll
  for (int iter = 0; iter < stagesReadOnly; ++iter) {
    auto const stage = iter % numStages;
    auto const i = iter * colsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (readState) {
      cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, i, 0, head,
                                                    batch, sram.bar_full[stage]);
      auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
    } else {
      auto const _ = sram.bar_full[stage].arrive();
    }
  }

  // Phase 2: Both read and write (steady state)
#pragma unroll
  for (int iter = 0; iter < stagesBoth; ++iter) {
    auto const stage = (stagesReadOnly + iter) % numStages;
    auto const i_read = (stagesReadOnly + iter) * colsPerStage;
    auto const i_write = iter * colsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (readState || writeState) {
      // Unblock async proxy for writeback
      cde::fence_proxy_async_shared_cta();
      // Writeback
      if constexpr (writeState) {
        cde::cp_async_bulk_tensor_4d_shared_to_global(&tensorState, i_write, 0, head, batch,
                                                      &sram.state[stage][0]);
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
      }

      // Read next
      if constexpr (readState) {
        cde::cp_async_bulk_tensor_4d_global_to_shared(&sram.state[stage][0], &tensorState, i_read,
                                                      0, head, batch, sram.bar_full[stage]);
        auto const _ = cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
      } else {
        auto const _ = sram.bar_full[stage].arrive();
      }
    } else {
      auto const _ = sram.bar_full[stage].arrive();
    }
  }

  // Phase 3: Write only (draining the pipeline)
#pragma unroll
  for (int iter = 0; iter < stagesWriteOnly; ++iter) {
    auto const stage = (stagesReadOnly + stagesBoth + iter) % numStages;
    auto const i_write = (stagesBoth + iter) * colsPerStage;

    sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

    if constexpr (writeState) {
      // Unblock async proxy for writeback
      cde::fence_proxy_async_shared_cta();
      cde::cp_async_bulk_tensor_4d_shared_to_global(&tensorState, i_write, 0, head, batch,
                                                    &sram.state[stage][0]);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
    }
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM,
          int DSTATE, int consumerWarps, int colsPerStage, int numStages, bool useStateCache>
__device__ __forceinline__ void consumer_func_horizontal(
    int d, int member, float A_value, float dt_value, float x_value,
    SharedStorageHorizontal<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, colsPerStage,
                            numStages>& sram,
    float& out_value) {
  namespace cde = cuda::device::experimental;
  constexpr auto lanesPerRow = (consumerWarps * warpSize) / DIM;
  constexpr auto itemsPerThread = colsPerStage / lanesPerRow;
  auto const group = d % (warpSize / lanesPerRow);

  // #pragma unroll 1
  for (int iBegin = 0, stage = 0; iBegin < DSTATE;
       iBegin += colsPerStage, stage = (stage + 1) % numStages) {
    // wait for the producer
    sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

    constexpr auto bankSize = sizeof(uint32_t);
    constexpr auto stateValuesPerBank = bankSize / sizeof(state_t);
    constexpr auto numBanks = 32;
    if constexpr (sizeof(state_t) == sizeof(input_t)) {
#pragma unroll
      for (int item = 0; item < itemsPerThread; item += stateValuesPerBank) {
        auto const baseCol = item + member * itemsPerThread;
        // If I just use baseCol as the index, a lot of bank conflicts will arise.
        //
        auto const ii =
            conflict_free_column<colsPerStage, stateValuesPerBank, numBanks>(group, baseCol);

        auto const i = iBegin + ii;

        auto* sState_ptr = reinterpret_cast<uint*>(&sram.state[stage][d * colsPerStage + ii]);
        uint32_t rState = *sState_ptr;
        auto* rState_ptr = reinterpret_cast<state_t*>(&rState);

        uint32_t rB = *reinterpret_cast<uint32_t const*>(&sram.B[i]);
        auto* rB_ptr = reinterpret_cast<input_t const*>(&rB);

        uint32_t rC = *reinterpret_cast<uint32_t const*>(&sram.C[i]);
        auto* rC_ptr = reinterpret_cast<input_t const*>(&rC);

        for (int e = 0; e < stateValuesPerBank; e++) {
          float state_value;
          if constexpr (!useStateCache) {
            state_value = 0.f;
          } else {
            state_value = toFloat(rState_ptr[e]);
          }

          auto const B_value = toFloat(rB_ptr[e]);
          auto const C_value = toFloat(rC_ptr[e]);

          auto const dA = __expf(A_value * dt_value);
          auto const dB = B_value * dt_value;
          auto const new_state = state_value * dA + dB * x_value;

          convertAndStore(&rState_ptr[e], new_state);
          out_value += new_state * C_value;
        }
        *sState_ptr = rState;
      }
    } else {
      for (int item = 0; item < itemsPerThread; item += stateValuesPerBank) {
        auto const baseCol = item + member * itemsPerThread;
        auto const ii =
            conflict_free_column<colsPerStage, stateValuesPerBank, numBanks>(group, baseCol);
        auto const i = iBegin + ii;

        auto* sState_ptr = reinterpret_cast<uint*>(&sram.state[stage][d * colsPerStage + ii]);
        uint32_t rState = *sState_ptr;
        auto* rState_ptr = reinterpret_cast<state_t*>(&rState);

        for (int e = 0; e < stateValuesPerBank; e++) {
          float state_value;
          if constexpr (!useStateCache) {
            state_value = 0.f;
          } else {
            state_value = toFloat(rState_ptr[e]);
          }

          auto const B_value = toFloat(sram.B[i + e]);
          auto const C_value = toFloat(sram.C[i + e]);

          auto const dA = __expf(A_value * dt_value);
          auto const dB = B_value * dt_value;
          auto const new_state = state_value * dA + dB * x_value;

          convertAndStore(&rState_ptr[e], new_state);
          out_value += new_state * C_value;
        }
        *sState_ptr = rState;
      }
    }

    auto _ = sram.bar_empty[stage].arrive();
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int DIM, int DSTATE, int consumerWarps, int colsPerStage,
          int headsGroupsRatio, int numStages = 1>
__global__ void selective_state_update_kernel_producer_consumer_horizontal(
    SelectiveStateUpdateParams params, __grid_constant__ CUtensorMap const tensorState) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto const* __restrict__ x = reinterpret_cast<input_t const*>(params.x);
  auto const* __restrict__ dt = reinterpret_cast<weight_t const*>(params.dt);
  auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ B = reinterpret_cast<input_t const*>(params.B);
  auto const* __restrict__ C = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);

  int const nheads = params.nheads;

  constexpr auto numWarps = 1 + consumerWarps;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const group = head / headsGroupsRatio;
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;

  extern __shared__ uint8_t sbuffer[];
  using sram_t = SharedStorageHorizontal<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE,
                                         colsPerStage, numStages>;
  auto& sram = *reinterpret_cast<sram_t*>(sbuffer);

  namespace cde = cuda::device::experimental;
  namespace cg = cooperative_groups;

  for (int stage = warp; stage < numStages; stage += numWarps) {
    if (lane > 0) continue;
    constexpr auto num_arrivals = 1 + consumerWarps * warpSize;
    init(&sram.bar_empty[stage], num_arrivals);
    init(&sram.bar_full[stage], num_arrivals);
    // signal to async proxy that barriers are initilized
    cde::fence_proxy_async_shared_cta();
  }
  if (lane == 0 && warp == 0) {
    init(&sram.bar_consumers, warpSize * consumerWarps);
  }
  __syncthreads();

  if (warp == consumerWarps)  // producer
  {
    auto const read_state = (state_batch != params.pad_slot_id);
    auto const write_state = read_state && params.update_state;

    cg::invoke_one(cg::coalesced_threads(), [&]() {
      if (read_state && write_state)
        producer_func_horizontal<state_t, DIM, DSTATE, colsPerStage, numStages, true, true>(
            sram, tensorState, state_batch, head);
      else if (read_state && !write_state)
        producer_func_horizontal<state_t, DIM, DSTATE, colsPerStage, numStages, true, false>(
            sram, tensorState, state_batch, head);
      else
        producer_func_horizontal<state_t, DIM, DSTATE, colsPerStage, numStages, false, false>(
            sram, tensorState, state_batch, head);
    });
  } else {  // consumers

    using load_t = PackedAligned<input_t>;

    // Unblock the producer
#pragma unroll
    for (auto stage = 0; stage < numStages; ++stage) {
      auto const _ = sram.bar_empty[stage].arrive();
    }

    // Load A
    auto const A_value = toFloat(A[head]);

    // Load D
    auto const d_value = D ? toFloat(D[head]) : 0.f;

    // load dt_value
    auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
    if (dt_bias) dt_value += toFloat(dt_bias[head]);
    if (params.dt_softplus) {
      dt_value = thresholded_softplus(dt_value);
    }

    if (warp == 0) {  // Load B
      for (auto d = lane * load_t::count; d < DSTATE; d += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.B[d]);
        *dst = *reinterpret_cast<load_t const*>(
            &B[batch * params.B_stride_batch + group * DSTATE + d]);
      }
    } else if (warp == 1) {  // Load C
      for (auto i = lane * load_t::count; i < DSTATE; i += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
        *dst = *reinterpret_cast<load_t const*>(
            &C[batch * params.C_stride_batch + group * DSTATE + i]);
      }
    }

    constexpr auto lanesPerRow = (consumerWarps * warpSize) / DIM;
    static_assert(lanesPerRow >= 1);
    constexpr auto rowsPerWarp = warpSize / lanesPerRow;
    auto const group = lane % rowsPerWarp;
    auto const member = lane / rowsPerWarp;
    auto const d = warp * rowsPerWarp + group;
    auto const x_value = toFloat(x[batch * params.x_stride_batch + head * DIM + d]);
    auto const z_value = z ? toFloat(z[batch * params.z_stride_batch + head * DIM + d]) : 0.f;

    sram.bar_consumers.wait(sram.bar_consumers.arrive());

    // Thread
    float out_value = 0.f;
    if (state_batch != params.pad_slot_id)
      consumer_func_horizontal<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                               colsPerStage, numStages, true>(d, member, A_value, dt_value, x_value,
                                                              sram, out_value);
    else
      consumer_func_horizontal<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                               colsPerStage, numStages, false>(d, member, A_value, dt_value,
                                                               x_value, sram, out_value);

    out_value += __shfl_down_sync(UINT32_MAX, out_value, 16);
    if constexpr (lanesPerRow == 4) {
      out_value += __shfl_down_sync(UINT32_MAX, out_value, 8);
    }

    if (member == 0) {
      out_value += d_value * x_value;

      // Write output
      if (z) {
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
      }
      convertAndStore(&output[batch * params.out_stride_batch + head * DIM + d], out_value);
    }
  }
}

#endif  // FLASHINFER_MAMBA_ENABLE_SM90 (horizontal kernel)

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, SSUAlgorithm algorithm,
                                cudaStream_t stream) {
  auto [sm_major, sm_minor] = GetCudaComputeCapability();

  // Common alignment checks for all kernels
  check_ptr_alignment_input_vars<input_t>(params);

  // Resolve auto to a concrete algorithm based on GPU architecture and batch size
  SSUAlgorithm algo = algorithm;
  if (algo == SSUAlgorithm::kAuto) {
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
    if (sm_major < 9) {
      algo = SSUAlgorithm::kSimple;
    } else {
      // At small batch sizes, the tiled simple kernel outperforms producer-consumer
      // kernels because it has lower per-block overhead and can still saturate the GPU
      // via dim-tiling. Threshold: batch*nheads < 2*num_SMs (i.e. not enough blocks
      // for the non-tiled producer-consumer kernels to hide latency).
      int const total_blocks = params.batch * params.nheads;
      int const num_sms = GetCudaMultiProcessorCount();
      if (total_blocks < num_sms * 2)
        algo = SSUAlgorithm::kSimple;
      else if (sm_major < 10)
        algo = SSUAlgorithm::kVertical;
      else
        // On Blackwell+: vertical is slightly faster for fp32 state,
        // horizontal is faster for fp16/bf16 state.
        algo = (sizeof(state_t) == 4) ? SSUAlgorithm::kVertical : SSUAlgorithm::kHorizontal;
    }
#else
    algo = SSUAlgorithm::kSimple;
#endif
  }

  if (algo == SSUAlgorithm::kSimple) {
    constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
    using load_state_t = PackedAligned<state_t, stateLoadSize>;

    FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.state) % sizeof(load_state_t) == 0,
                     "state pointer must be aligned to ", sizeof(load_state_t), " bytes");
    FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                     "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

    constexpr int numWarps = 4;
    constexpr int ROWS_PER_BLOCK = 4;
    int const total_blocks = params.batch * params.nheads;
    int const num_sms = GetCudaMultiProcessorCount();

    dim3 block(warpSize, numWarps);
    if (total_blocks < num_sms * 2) {
      // Tiled: split dim across blocks for better GPU occupancy at small batch sizes
      int const dim_tiles = (DIM + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
      dim3 grid(params.batch, params.nheads, dim_tiles);
      selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM,
                                           DSTATE, ROWS_PER_BLOCK, numWarps>
          <<<grid, block, 0, stream>>>(params);
    } else {
      // Non-tiled: enough blocks already for full occupancy; ROWS_PER_BLOCK == DIM so blockIdx.z ==
      // 0
      dim3 grid(params.batch, params.nheads);
      selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM,
                                           DSTATE, DIM, numWarps>
          <<<grid, block, 0, stream>>>(params);
    }
  }
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  else if (algo == SSUAlgorithm::kVertical) {
    constexpr auto numConsumers = 4;
    constexpr auto numWarps = 1 + numConsumers;
    constexpr auto numStages = 3;
    constexpr auto rowsPerStage = 4 * numConsumers;
    FLASHINFER_CHECK(params.dim % rowsPerStage == 0, "dim must be divisible by ", rowsPerStage,
                     " for vertical kernel");
    auto scan_func = selective_state_update_kernel_producer_consumer_vertical<
        input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE, numConsumers,
        rowsPerStage, numStages>;

    dim3 block(warpSize, numWarps);
    dim3 grid(params.batch, params.nheads);

    auto state_tensor =
        tma::buildNdDescriptor(typeid(state_t),
                               /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
                               /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
                               /*tiles*/ {DSTATE, rowsPerStage, 1, 1}, params.state);

    using sram_t = SharedStorageVertical<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM,
                                         DSTATE, numStages>;
    constexpr size_t smem_size = sizeof(sram_t);
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(scan_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    scan_func<<<grid, block, smem_size, stream>>>(params, state_tensor);
  } else if (algo == SSUAlgorithm::kHorizontal) {
    constexpr auto numConsumers = (DIM / 64) * 4;
    constexpr auto numProducers = 1;
    constexpr auto numWarps = numProducers + numConsumers;

    constexpr auto sectorSize = 32;  // bytes
    constexpr auto stageCols = 2 * sectorSize / sizeof(state_t);

    constexpr auto totalStages = DSTATE / stageCols;
    constexpr auto numStages = (totalStages >= 4) ? 4 : totalStages;

    auto ratio_launcher = [&]<int RATIO>() {
      auto scan_func = selective_state_update_kernel_producer_consumer_horizontal<
          input_t, weight_t, matrixA_t, state_t, stateIndex_t, DIM, DSTATE, numConsumers, stageCols,
          RATIO, numStages>;

      dim3 block(warpSize, numWarps);
      dim3 grid(params.batch, params.nheads);

      auto state_tensor =
          tma::buildNdDescriptor(typeid(state_t),
                                 /*shapes*/ {DSTATE, DIM, params.nheads, params.state_cache_size},
                                 /*strides*/ {1, DSTATE, DSTATE * DIM, params.state_stride_batch},
                                 /*tiles*/ {stageCols, DIM, 1, 1}, params.state);
      static_assert(DSTATE % stageCols == 0 && DSTATE >= stageCols);

      using sram_t = SharedStorageHorizontal<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE,
                                             stageCols, numStages>;
      constexpr size_t smem_size = sizeof(sram_t);
      FLASHINFER_CUDA_CHECK(
          cudaFuncSetAttribute(scan_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      scan_func<<<grid, block, smem_size, stream>>>(params, state_tensor);
    };

    dispatchRatio(params, std::integer_sequence<int, 1, 8, 16>{}, ratio_launcher);
  }
#endif
  else {
    FLASHINFER_CHECK(false, "Unsupported SSU algorithm: ", SSUAlgorithmToString(algo),
                     ". Vertical/horizontal require FLASHINFER_MAMBA_ENABLE_SM90.");
  }
}

}  // namespace flashinfer::mamba
