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

template <typename T>
__device__ inline auto make_zero() -> T;

template <>
__device__ inline auto make_zero<float2>() -> float2 {
  return make_float2(0.f, 0.f);
}

template <typename compute_t, typename load_t>
__device__ inline auto make_zeros() -> load_t {
  load_t rValue;
#pragma unroll
  for (int i = 0; i < sizeof(load_t) / sizeof(compute_t); i++) {
    auto* dst = reinterpret_cast<compute_t*>(&rValue) + i;
    convertAndStore(dst, 0.f);
  }
  return rValue;
}

__device__ __forceinline__ float warpReduceSum(float val) {
  constexpr auto warpSize = 32;
  for (int s = warpSize / 2; s > 0; s /= 2) {
    val += __shfl_down_sync(UINT32_MAX, val, s);
  }
  return val;
}

template <typename input_t, typename weight_t, typename state_t>
struct VectorizedLoadTraits {};

template <>
struct VectorizedLoadTraits<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> {
  using input = float2;
  using weight = float2;
  using state = float2;
  static constexpr auto chunk_size = sizeof(input) / sizeof(__nv_bfloat16);
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t, int DSTATE,
          int numWarps>
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
  int const dim = params.dim;

  constexpr auto warpSize = 32;
  auto const dim_offset = blockIdx.x * warpSize * numWarps;
  auto const batch = blockIdx.y;
  auto const head = blockIdx.z;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
  state += state_batch * nheads * dim * DSTATE + head * dim * DSTATE;

  __shared__ input_t sx[numWarps * warpSize];
  __shared__ float sdt[numWarps * warpSize];
  __shared__ weight_t sz[numWarps * warpSize];

  auto const A_value = toFloat(A[head]);

  auto dt_value = toFloat(dt[batch * params.dt_stride_batch + head]);
  if (dt_bias) dt_value += toFloat(dt_bias[head]);
  if (dt_softplus) {
    dt_value = thresholded_softplus(dt_value);
  }

  auto const dA = __expf(A_value * dt_value);

  auto d_value = D ? toFloat(D[head]) : 0.f;

  auto _d = warp * warpSize + lane;
  auto d = dim_offset + _d;
  if (d < dim) {
    sx[_d] = x[batch * params.x_stride_batch + head * dim + d];
    if (z) {
      sz[_d] = z[batch * params.z_stride_batch + head * dim + d];
    } else {
      convertAndStore(&sz[_d], 0.f);
    }
  } else {
    convertAndStore(&sx[_d], 0.f);
    convertAndStore(&sz[_d], 0.f);
  }

  using Load = VectorizedLoadTraits<input_t, weight_t, state_t>;

  for (auto _d = warp * warpSize; _d < (warp + 1) * warpSize; _d++) {
    auto d = dim_offset + _d;
    if (d >= dim) break;

    float x_value = toFloat(sx[_d]);
    float out_value = d_value * x_value * int(lane == 0);  // first lane has the value

    for (int i = threadIdx.x * Load::chunk_size; i < DSTATE; i += warpSize * Load::chunk_size) {
      auto rState = make_zeros<state_t, Load::state>();
      if (state_batch != params.pad_slot_id)
        rState = *reinterpret_cast<typename Load::state*>(&state[d * DSTATE + i]);
      auto rB = *reinterpret_cast<typename Load::input const*>(
          &B[batch * params.B_stride_batch + group * DSTATE + i]);
      auto rC = *reinterpret_cast<typename Load::input const*>(
          &C[batch * params.C_stride_batch + group * DSTATE + i]);

      auto* state_vals = reinterpret_cast<state_t*>(&rState);
      auto const* B_vals = reinterpret_cast<input_t const*>(&rB);
      auto const* C_vals = reinterpret_cast<input_t const*>(&rC);

      for (int ii = 0; ii < Load::chunk_size; ii++) {
        auto state_value = toFloat(state_vals[ii]);
        auto B_value = toFloat(B_vals[ii]);
        auto C_value = toFloat(C_vals[ii]);

        auto const dB = B_value * dt_value;
        auto const new_state = state_value * dA + dB * x_value;
        convertAndStore(&state_vals[ii], new_state);

        out_value += new_state * C_value;
      }
      if (state_batch != params.pad_slot_id)
        *reinterpret_cast<typename Load::state*>(&state[d * DSTATE + i]) = rState;
    }

    // warpReduce the out_value
    out_value = warpReduceSum(out_value);
    if (lane == 0) {
      sdt[_d] = out_value;
    }
  }

  if (d < dim) {
    auto out_value = sdt[_d];
    if (z) {
      float z_value = toFloat(sz[_d]);
      float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
      float silu_z = z_value * sig_z;
      out_value *= silu_z;
    }
    convertAndStore(&output[batch * params.out_stride_batch + head * dim + d], out_value);
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          int rowsPerStage, int dim, int dstate, uint8_t numStages>
struct SharedStorage {
  alignas(128) state_t state[numStages][rowsPerStage * dstate];
  input_t x[dim];
  float out[dim];  // dt is special cause we're gonna store input in there as well
  input_t z[dim];
  input_t B[dstate];
  input_t C[dstate];

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  barrier_t bar_empty[numStages];
  barrier_t bar_full[numStages];
  barrier_t bar_consumers;
};

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
  int const dim = params.dim;

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
#pragma nv_diag_suppress 20054
  __shared__ sram_t sram;
#pragma nv_diag_default 20054

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

  if (warp == consumerWarps) {
    auto const state_offset = (state_batch * nheads + head) * dim;

    for (int d = 0, stage = 0; d < dim + rowsPerStage * numStages;
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

            if (d < dim) {
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

    using load_t = float2;
    static constexpr auto vectorizedLoadSize = sizeof(load_t) / sizeof(weight_t);

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
      for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize) {
        auto* dst = reinterpret_cast<load_t*>(&sram.x[d]);
        *dst = *reinterpret_cast<load_t const*>(&x[batch * params.x_stride_batch + head * dim + d]);
      }
      for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize) {
        auto* dst = reinterpret_cast<load_t*>(&sram.B[i]);
        *dst = *reinterpret_cast<load_t const*>(
            &B[batch * params.B_stride_batch + group * DSTATE + i]);
      }
    } else if (warp == 1) {  // Load z, C
      for (auto d = lane * vectorizedLoadSize; d < dim; d += warpSize * vectorizedLoadSize) {
        auto* dst = reinterpret_cast<load_t*>(&sram.z[d]);
        *dst =
            z ? *reinterpret_cast<load_t const*>(&z[batch * params.z_stride_batch + head * dim + d])
              : make_zero<load_t>();
      }
      for (auto i = lane * vectorizedLoadSize; i < DSTATE; i += warpSize * vectorizedLoadSize) {
        auto* dst = reinterpret_cast<load_t*>(&sram.C[i]);
        *dst = *reinterpret_cast<load_t const*>(
            &C[batch * params.C_stride_batch + group * DSTATE + i]);
      }
    }

    sram.bar_consumers.wait(sram.bar_consumers.arrive());

    for (auto dBegin = 0, stage = 0; dBegin < dim;
         dBegin += rowsPerStage, stage = (stage + 1) % numStages) {
      // wait for the producer
      sram.bar_full[stage].wait(sram.bar_full[stage].arrive());

#pragma unroll
      for (auto dd = warp; dd < rowsPerStage; dd += consumerWarps) {
        auto d = dBegin + dd;
        float const x_value = toFloat(sram.x[d]);
        float out_value = toFloat(d_value) * x_value * int(lane == 0);  // first lane has the value

        for (int i = lane; i < DSTATE; i += warpSize) {
          auto const state_value = (state_batch != params.pad_slot_id)
                                       ? toFloat(sram.state[stage][dd * DSTATE + i])
                                       : 0.f;
          auto const B_value = toFloat(sram.B[i]);
          auto const C_value = toFloat(sram.C[i]);

          auto const dB = B_value * dt_value;
          auto const new_state = state_value * dA + dB * x_value;

          convertAndStore(&sram.state[stage][dd * DSTATE + i], new_state);
          out_value += new_state * C_value;
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

    // Write output
    sram.bar_consumers.wait(sram.bar_consumers.arrive());
    auto d = warp * warpSize + lane;
    if (d < dim) {
      auto out_value = sram.out[d];
      if (z) {
        float z_value = toFloat(sram.z[d]);
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
      }
      convertAndStore(&output[batch * params.out_stride_batch + head * dim + d], out_value);
    }
  }
#endif
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t>
void invokeSelectiveStateUpdate(SelectiveStateUpdateParams& params, cudaStream_t stream) {
  auto [sm_major, sm_minor] = GetCudaComputeCapability();

#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  if (sm_major < 9)  // pre-Hopper
#endif
  {
    auto dispatch_dstate = [&]<int DSTATE>() {
      constexpr int numWarps = 2;
      int const blocks_per_dim = (params.dim + 32 * numWarps - 1) / (32 * numWarps);
      dim3 block(32, numWarps);
      dim3 grid(blocks_per_dim, params.batch, params.nheads);
      selective_state_update_kernel_simple<input_t, weight_t, matrixA_t, state_t, DSTATE, numWarps>
          <<<grid, block, 0, stream>>>(params);
    };

    switch (params.dstate) {
      case 64:
        dispatch_dstate.template operator()<64>();
        break;
      case 128:
        dispatch_dstate.template operator()<128>();
        break;
      case 256:
        dispatch_dstate.template operator()<256>();
        break;
      default:
        FLASHINFER_CHECK(false, "Unsupported dstate value. Supported values are: 64, 128, 256");
    }
  }
#ifdef FLASHINFER_MAMBA_ENABLE_SM90
  else {

    auto dispatch_dim_dstate = [&]<int DIM, int DSTATE>() {
      constexpr auto numConsumers = 4;
      constexpr auto numWarps = 1 + numConsumers;
      constexpr auto numStages = 3;
      constexpr auto rowsPerStage = 4 * numConsumers;
      FLASHINFER_CHECK(params.dim % rowsPerStage == 0);
      auto scan_func = selective_state_update_kernel_producer_consumer_vertical<
          input_t, weight_t, matrixA_t, state_t, DIM, DSTATE, numConsumers, rowsPerStage,
          numStages>;

      dim3 block(32, numWarps);
      dim3 grid(params.batch, params.nheads);

      auto nh = params.nheads;
      auto dim = params.dim;
      auto B = params.state_cache_size;

      FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.state) % 128 ==
                       0);  // TMA requires 128B aligned
      auto tensorState =
          tma::createTensorMap<state_t>(params.state, B * nh * dim, DSTATE, rowsPerStage, DSTATE);

      scan_func<<<grid, block, 0, stream>>>(params, tensorState);
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
