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
#ifndef FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
#define FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cuda/barrier>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"

namespace flashinfer::mamba {

using namespace conversion;

struct SelectiveStateUpdateParams {
  uint32_t batch{}, nheads{}, dim{}, dstate{}, ngroups{}, state_cache_size{};
  int32_t pad_slot_id{-1};
  bool dt_softplus{false};

  int64_t x_stride_batch{}, dt_stride_batch{}, B_stride_batch{}, C_stride_batch{},
      out_stride_batch{}, z_stride_batch{};

  void* __restrict__ state{nullptr};  // state_t: (state_cache_size, nheads, dim, dstate)
  void* __restrict__ x{nullptr};      // input_t: (batch, nheads, dim)
  void* __restrict__ dt{
      nullptr};  // weight_t: (batch, nheads) but pretends to be (batch, nheads, dim)
  void* __restrict__ dt_bias{nullptr};  // weight_t (nheads) but pretends to be (nheads, dim)
  void* __restrict__ A{nullptr};  // matrixA_t: (nheads) but pretends to be (nheads, dim, dstate)
  void* __restrict__ B{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ C{nullptr};  // input_t: (batch, ngroups, dstate)
  void* __restrict__ D{nullptr};  // weight_t: (nheads) but pretends to be (nheads, dim)
  void* __restrict__ z{nullptr};  // input_t: (batch, nheads, dim)
  void* __restrict__ output{nullptr};               // input_t: (batch, nheads, dim)
  void* __restrict__ state_batch_indices{nullptr};  // state_batch_indices: (batch,)
};

__forceinline__ __device__ float softplus(float x) { return __logf(1.f + __expf(x)); }

__device__ __forceinline__ float thresholded_softplus(float dt_value) {
  constexpr float threshold = 20.f;
  return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

// Simple packed vector type for loading N elements of type T
template <typename T, int N = sizeof(float4) / sizeof(T)>
struct alignas(N * sizeof(T)) PackedAligned {
  T val[N];
  static constexpr int count = N;
  using dtype = T;
};

template <class load_t>
__device__ __forceinline__ auto make_zeros() -> load_t {
  load_t ret{};
#pragma unroll
  for (int i = 0; i < ret.count; i++)
    ret.val[i] = typename load_t::dtype{};  // default initialization
  return ret;
};

// Computes the vector load size that ensures full warp utilization.
// Avoids cases like: dstate=64, load_t = sizeof(float4)/sizeof(f16), warpsize=32 (32 * 8 > 64)
// in which case a part of the warp would be idle.
template <typename T, int DSTATE>
inline constexpr auto getVectorLoadSizeForFullUtilization() -> unsigned {
  static_assert(sizeof(float4) >= sizeof(T));
  constexpr unsigned maxHardwareLoadSize = sizeof(float4) / sizeof(T);
  constexpr unsigned warpSize = 32;
  constexpr unsigned maxLogicalLoadSize = (unsigned)DSTATE / warpSize;
  return maxHardwareLoadSize < maxLogicalLoadSize ? maxHardwareLoadSize : maxLogicalLoadSize;
}

__device__ __forceinline__ float warpReduceSum(float val) {
  constexpr auto warpSize = 32;
  for (int s = warpSize / 2; s > 0; s /= 2) {
    val += __shfl_down_sync(UINT32_MAX, val, s);
  }
  return val;
}

template <typename input_t, int dim, int dstate>
struct SharedStorageSimple {
  alignas(alignof(PackedAligned<input_t>)) input_t x[dim];
  float out[dim];
  alignas(alignof(PackedAligned<input_t>)) input_t z[dim];
  alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM,
          int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple(SelectiveStateUpdateParams params) {
  auto* __restrict__ output =
      reinterpret_cast<input_t*>(params.output);  // output: (batch, nheads, dim)
  auto* __restrict__ state =
      reinterpret_cast<state_t*>(params.state);  // state: (batch, nheads, dim, dstate)

  auto const* __restrict__ x =
      reinterpret_cast<input_t const*>(params.x);  // x: (batch, nheads, dim)
  auto const* __restrict__ dt =
      reinterpret_cast<weight_t const*>(params.dt);                           // dt: (batch, nheads)
  auto const* __restrict__ A = reinterpret_cast<matrixA_t const*>(params.A);  // A: (nheads)
  auto const* __restrict__ B =
      reinterpret_cast<input_t const*>(params.B);  // B: (batch, ngroups, dstate)
  auto const* __restrict__ C =
      reinterpret_cast<input_t const*>(params.C);  // C: (batch, ngroups, dstate)
  auto const* __restrict__ D = reinterpret_cast<weight_t const*>(params.D);  // D: (nheads, dim)
  auto const* __restrict__ dt_bias = reinterpret_cast<weight_t const*>(params.dt_bias);  // (nheads)
  auto const* __restrict__ z = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ state_batch_indices =
      reinterpret_cast<int const*>(params.state_batch_indices);
  bool const dt_softplus = params.dt_softplus;

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  constexpr auto warpSize = 32;
  constexpr auto rowsPerWarp = (DIM + numWarps - 1) / numWarps;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
  state += (state_batch * nheads + head) * DIM * DSTATE;

  __shared__ SharedStorageSimple<input_t, DIM, DSTATE> sram;

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

  if (warp == 0) {
    for (auto d = lane * load_input_t::count; d < DIM; d += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.x[d]);
      *dst = *reinterpret_cast<load_input_t const*>(
          &x[batch * params.x_stride_batch + head * DIM + d]);
    }
    for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.B[i]);
      *dst = *reinterpret_cast<load_input_t const*>(
          &B[batch * params.B_stride_batch + group * DSTATE + i]);
    }
  } else if (warp == 1) {  // Load z, C
    for (auto d = lane * load_input_t::count; d < DIM; d += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.z[d]);
      *dst = z ? *reinterpret_cast<load_input_t const*>(
                     &z[batch * params.z_stride_batch + head * DIM + d])
               : make_zeros<load_input_t>();
    }
    for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
      auto* dst = reinterpret_cast<load_input_t*>(&sram.C[i]);
      *dst = *reinterpret_cast<load_input_t const*>(
          &C[batch * params.C_stride_batch + group * DSTATE + i]);
    }
  }
  __syncthreads();

  for (auto _d = warp * rowsPerWarp; _d < (warp + 1) * rowsPerWarp; _d++) {
    auto d = _d;
    if (d >= DIM) break;

    float x_value = toFloat(sram.x[_d]);
    float out_value = d_value * x_value * int(lane == 0);  // first lane has the value

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
      if (state_batch != params.pad_slot_id)
        *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]) = rState;
    }

    // warpReduce the out_value
    out_value = warpReduceSum(out_value);
    if (lane == 0) {
      sram.out[_d] = out_value;
    }
  }

  __syncthreads();

  for (int l = lane; l < rowsPerWarp; l += warpSize) {
    auto d = warp * rowsPerWarp + l;
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
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          int rowsPerStage, int dim, int dstate, uint8_t numStages>
struct SharedStorage {
  alignas(128) state_t state[numStages][rowsPerStage * dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t x[dim];
  float out[dim];  // dt is special cause we're gonna store input in there as well
  alignas(alignof(PackedAligned<input_t>)) input_t z[dim];
  alignas(alignof(PackedAligned<input_t>)) input_t B[dstate];
  alignas(alignof(PackedAligned<input_t>)) input_t C[dstate];

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  barrier_t bar_empty[numStages];
  barrier_t bar_full[numStages];
  barrier_t bar_consumers;
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM,
          int DSTATE, int consumerWarps, int rowsPerStage, int numStages, bool useStateCache>
__device__ __forceinline__ void consumer_func_vertical(
    int lane, int warp, float d_value, float dt_value, float dA,
    SharedStorage<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE, numStages>&
        sram) {
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
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DIM,
          int DSTATE, int consumerWarps, int rowsPerStage, int numStages = 1>
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
      reinterpret_cast<int const*>(params.state_batch_indices);

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  constexpr auto warpSize = 32;
  constexpr auto numWarps = 1 + consumerWarps;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;

  using sram_t =
      SharedStorage<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE, numStages>;
  // Use dynamic shared memory to allow opting into extended shared memory on SM90+
  extern __shared__ __align__(128) char smem[];
  sram_t& sram = *reinterpret_cast<sram_t*>(smem);

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
    auto const state_offset = (state_batch * nheads + head) * DIM;

    for (int d = 0, stage = 0; d < DIM + rowsPerStage * numStages;
         d += rowsPerStage, stage = (stage + 1) % numStages) {
      if (lane == 0) {
        cg::invoke_one(cg::coalesced_threads(), [&]() {
          sram.bar_empty[stage].wait(sram.bar_empty[stage].arrive());

          if (state_batch != params.pad_slot_id) {
            // Writeback
            if (d >= rowsPerStage * numStages) {
              cde::cp_async_bulk_tensor_2d_shared_to_global(
                  &tensorState,
                  /*x*/ 0,
                  /*y*/ state_offset + d - rowsPerStage * numStages, &sram.state[stage][0]);
              cde::cp_async_bulk_commit_group();
              cde::cp_async_bulk_wait_group_read<0>();
            }

            if (d < DIM) {
              cde::cp_async_bulk_tensor_2d_global_to_shared(&sram.state[stage][0], &tensorState,
                                                            /*x*/ 0, /*y*/ state_offset + d,
                                                            sram.bar_full[stage]);

              // Unblock the consumers
              auto constexpr bytesState = rowsPerStage * DSTATE * sizeof(state_t);
              auto constexpr bytesToArrive = bytesState;
              auto const _ =
                  cuda::device::barrier_arrive_tx(sram.bar_full[stage], 1, bytesToArrive);
            }
          } else {
            auto const _ = sram.bar_full[stage].arrive();
          }
        });
      }
    }
  } else {  // consumers

    using load_t = PackedAligned<input_t>;

#pragma unroll
    // Unblock the producer
    for (uint8_t stage = 0; stage < numStages; ++stage) {
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
    auto const dA = __expf(A_value * dt_value);

    if (warp == 0) {  // Load x, B
      for (auto d = lane * load_t::count; d < DIM; d += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.x[d]);
        *dst = *reinterpret_cast<load_t const*>(&x[batch * params.x_stride_batch + head * DIM + d]);
      }
      for (auto i = lane * load_t::count; i < DSTATE; i += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
        *dst = *reinterpret_cast<load_t const*>(
            &B[batch * params.B_stride_batch + group * DSTATE + i]);
      }
    } else if (warp == 1) {  // Load z, C
      for (auto d = lane * load_t::count; d < DIM; d += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.z[d]);
        *dst =
            z ? *reinterpret_cast<load_t const*>(&z[batch * params.z_stride_batch + head * DIM + d])
              : make_zeros<load_t>();
      }
      for (auto i = lane * load_t::count; i < DSTATE; i += warpSize * load_t::count) {
        auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
        *dst = *reinterpret_cast<load_t const*>(
            &C[batch * params.C_stride_batch + group * DSTATE + i]);
      }
    }

    sram.bar_consumers.wait(sram.bar_consumers.arrive());

    if (state_batch != params.pad_slot_id)
      consumer_func_vertical<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                             rowsPerStage, numStages, true>(lane, warp, d_value, dt_value, dA,
                                                            sram);
    else
      consumer_func_vertical<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, consumerWarps,
                             rowsPerStage, numStages, false>(lane, warp, d_value, dt_value, dA,
                                                             sram);

    // Write output
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

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream) {
  auto [sm_major, sm_minor] = GetCudaComputeCapability();

  constexpr auto warpSize = 32;

#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  if (sm_major < 9)  // pre-Hopper
#endif
  {
    auto dispatch_dim_dstate = [&]<int DIM, int DSTATE>() {
      // Alignment checks for vectorized loads in simple kernel
      constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
      using load_state_t = PackedAligned<state_t, stateLoadSize>;
      using load_input_t = PackedAligned<input_t>;

      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.state) % sizeof(load_state_t) == 0,
                       "state pointer must be aligned to ", sizeof(load_state_t), " bytes");
      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.x) % sizeof(load_input_t) == 0,
                       "x pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.x_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "x batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      if (params.z) {
        FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.z) % sizeof(load_input_t) == 0,
                         "z pointer must be aligned to ", sizeof(load_input_t), " bytes");
        FLASHINFER_CHECK((params.z_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                         "z batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      }
      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.B) % sizeof(load_input_t) == 0,
                       "B pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.C) % sizeof(load_input_t) == 0,
                       "C pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.B_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "B batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.C_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "C batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                       "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

      constexpr int numWarps = 4;
      dim3 block(warpSize, numWarps);
      dim3 grid(params.batch, params.nheads);
      selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, DIM, DSTATE,
                                           numWarps><<<grid, block, 0, stream>>>(params);
    };

    auto dispatch_dstate = [&]<int DIM>() {
      switch (params.dstate) {
        case 64:
          dispatch_dim_dstate.template operator()<DIM, 64>();
          break;
        case 128:
          dispatch_dim_dstate.template operator()<DIM, 128>();
          break;
        case 256:
          dispatch_dim_dstate.template operator()<DIM, 256>();
          break;
        default:
          FLASHINFER_CHECK(false, "Unsupported dstate value. Supported values are: 64, 128, 256");
      }
    };

    switch (params.dim) {
      case 64:
        dispatch_dstate.template operator()<64>();
        break;
      case 128:
        dispatch_dstate.template operator()<128>();
        break;
      default:
        FLASHINFER_CHECK(false, "Unsupported dim value. Supported values are: 64, 128");
    }
  }
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  else {

    auto dispatch_dim_dstate = [&]<int DIM, int DSTATE>() {
      // Alignment checks for vectorized loads in Hopper kernel
      // Note: State uses TMA which requires 128B alignment (checked below)
      // x, z, B, and C use PackedAligned<input_t>
      using load_input_t = PackedAligned<input_t>;

      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.x) % sizeof(load_input_t) == 0,
                       "x pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.x_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "x batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      if (params.z) {
        FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.z) % sizeof(load_input_t) == 0,
                         "z pointer must be aligned to ", sizeof(load_input_t), " bytes");
        FLASHINFER_CHECK((params.z_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                         "z batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      }
      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.B) % sizeof(load_input_t) == 0,
                       "B pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.C) % sizeof(load_input_t) == 0,
                       "C pointer must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.B_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "B batch stride must be aligned to ", sizeof(load_input_t), " bytes");
      FLASHINFER_CHECK((params.C_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
                       "C batch stride must be aligned to ", sizeof(load_input_t), " bytes");

      constexpr auto numConsumers = 4;
      constexpr auto numWarps = 1 + numConsumers;
      constexpr auto numStages = 3;
      constexpr auto rowsPerStage = 4 * numConsumers;
      FLASHINFER_CHECK(params.dim % rowsPerStage == 0, "dim must be divisible by ", rowsPerStage,
                       " for SM90+ kernel");
      auto scan_func = selective_state_update_kernel_producer_consumer_vertical<
          input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, numConsumers, rowsPerStage,
          numStages>;

      dim3 block(32, numWarps);
      dim3 grid(params.batch, params.nheads);

      auto nh = params.nheads;
      auto dim = params.dim;

      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 ==
                       0);  // TMA requires 128B aligned
      auto tensorState = tma::createTensorMap<state_t>(
          params.state, params.state_cache_size * nh * dim, DSTATE, rowsPerStage, DSTATE);

      // Calculate shared memory size and opt-in to extended shared memory
      using sram_t = SharedStorage<input_t, weight_t, matrixA_t, state_t, rowsPerStage, DIM, DSTATE,
                                   numStages>;
      constexpr size_t smem_size = sizeof(sram_t);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(scan_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      scan_func<<<grid, block, smem_size, stream>>>(params, tensorState);
    };

    auto dispatch_dstate = [&]<int DIM>() {
      switch (params.dstate) {
        case 64:
          dispatch_dim_dstate.template operator()<DIM, 64>();
          break;
        case 128:
          dispatch_dim_dstate.template operator()<DIM, 128>();
          break;
        case 256:
          dispatch_dim_dstate.template operator()<DIM, 256>();
          break;
        default:
          FLASHINFER_CHECK(false, "Unsupported dstate value. Supported values are: 64, 128, 256");
      }
    };

    switch (params.dim) {
      case 64:
        dispatch_dstate.template operator()<64>();
        break;
      case 128:
        dispatch_dstate.template operator()<128>();
        break;
      default:
        FLASHINFER_CHECK(false, "Unsupported dim value. Supported values are: 64, 128");
    }
  }
#endif
}

}  // namespace flashinfer::mamba

#endif  // FLASHINFER_MAMBA_SELECTIVE_STATE_UPDATE_CUH_
