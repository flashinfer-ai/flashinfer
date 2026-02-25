#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cuda/barrier>
#include <iostream>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"

namespace flashinfer::mamba::mtp {

using namespace conversion;

template <typename input_t, typename state_t, int TOKENS_MTP, int DIM, int DSTATE, int STATE_ROWS>
struct SharedStorageSimple {
  input_t x[TOKENS_MTP][DIM];
  float out[TOKENS_MTP][DIM];
  input_t z[TOKENS_MTP][DIM];
  input_t B[TOKENS_MTP][DSTATE];
  input_t C[TOKENS_MTP][DSTATE];
  state_t state[STATE_ROWS][DSTATE];
};

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int TOKENS_MTP, int DIM, int DSTATE, int numWarps>
__global__ void selective_state_update_kernel_simple_mtp(SelectiveStateMTPParams params) {
  auto* __restrict__ output = reinterpret_cast<input_t*>(params.output);
  auto* __restrict__ state = reinterpret_cast<state_t*>(params.state);
  auto* __restrict__ intermediate_states = reinterpret_cast<state_t*>(params.intermediate_states);

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
  auto const* __restrict__ intermediate_state_indices =
      reinterpret_cast<stateIndex_t const*>(params.intermediate_state_indices);
  bool const dt_softplus = params.dt_softplus;

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  auto const batch = blockIdx.x;
  auto const head = blockIdx.y;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
  auto const intermediate_cache_idx =
      intermediate_state_indices ? intermediate_state_indices[batch] : state_batch;
  state += state_batch * params.state_stride_batch + head * DIM * DSTATE;

  constexpr auto stateRowsPerWarpPerStage = 4;
  constexpr auto stageRows = stateRowsPerWarpPerStage * numWarps;

  extern __shared__ __align__(128) char smem[];
  auto& sram =
      *reinterpret_cast<SharedStorageSimple<input_t, state_t, TOKENS_MTP, DIM, DSTATE, stageRows>*>(
          smem);

  static constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;
  using load_input_t = PackedAligned<input_t>;
  using load_weight_t = PackedAligned<weight_t>;

  auto const A_value = toFloat(A[head]);
  auto const d_value = D ? toFloat(D[head]) : 0.f;
  auto const dt_bias_value = dt_bias ? toFloat(dt_bias[head]) : 0.f;

  // Loop over multiple tokens
  if (warp == 0) {  // Load x: gmem -> smem
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto d = lane * load_input_t::count; d < DIM; d += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.x[mtp_step][d]);
        *dst = *reinterpret_cast<load_input_t const*>(
            &x[batch * params.x_stride_batch + mtp_step * params.x_stride_mtp + head * DIM + d]);
      }
    }
  } else if (warp == 1) {  // Load B: gmem -> smem
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.B[mtp_step][i]);
        *dst = *reinterpret_cast<load_input_t const*>(
            &B[batch * params.B_stride_batch + mtp_step * params.B_stride_mtp + group * DSTATE +
               i]);
      }
    }
  } else if (warp == 2) {  // Load z: gmem -> smem
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto d = lane * load_input_t::count; d < DIM; d += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.z[mtp_step][d]);
        *dst = z ? *reinterpret_cast<load_input_t const*>(
                       &z[batch * params.z_stride_batch + mtp_step * params.z_stride_mtp +
                          head * DIM + d])
                 : make_zeros<load_input_t>();
      }
    }
  }
  // Load C: gmem -> smem
  else if (warp == 3) {
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.C[mtp_step][i]);
        *dst = *reinterpret_cast<load_input_t const*>(
            &C[batch * params.C_stride_batch + mtp_step * params.C_stride_mtp + group * DSTATE +
               i]);
      }
    }
  }

  float rdt[TOKENS_MTP];
  for (int step = 0; step < TOKENS_MTP; step++) {
    auto dt_value =
        dt_bias_value +
        toFloat(dt[batch * params.dt_stride_batch + step * params.dt_stride_mtp + head]);
    if (dt_softplus) {
      dt_value = thresholded_softplus(dt_value);
    }
    rdt[step] = dt_value;
  }

  __syncthreads();

  for (auto dBegin = 0; dBegin < DIM; dBegin += stageRows) {
    // Load state gmem -> smem
    for (int warpRow = 0; warpRow < stateRowsPerWarpPerStage; warpRow++) {
      auto dd = warp * stateRowsPerWarpPerStage + warpRow;
      auto d = dBegin + dd;
      if (d < DIM) {
        if (state_batch != params.pad_slot_id) {
          for (int i = lane * load_state_t::count; i < DSTATE;
               i += warpSize * load_state_t::count) {
            auto* dst = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
            *dst = *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]);
          }
        }
      }
    }

    // Compute how many input_t elements to pack per SRAM load based on DSTATE/warpSize ratio
    constexpr auto stateValuesPerThread = DSTATE / warpSize;
    // We will be loading two-banks worth of input_t at a time instead of 1 in order to reduce the
    // load on LSU.
    constexpr auto maxPackedElements = sizeof(uint64_t) / sizeof(input_t);
    constexpr auto packedSramLdInputElements =
        (stateValuesPerThread >= maxPackedElements) ? maxPackedElements : stateValuesPerThread;
    static_assert(stateValuesPerThread % packedSramLdInputElements == 0,
                  "stateValuesPerThread must be divisible by packedSramLdInputElements");
    using packed_input_t = PackedAligned<input_t, packedSramLdInputElements>;
    float rState[stateValuesPerThread];
    packed_input_t rB;
    packed_input_t rC;

    for (int warpRow = 0; warpRow < stateRowsPerWarpPerStage; warpRow++) {
      auto dd = warp * stateRowsPerWarpPerStage + warpRow;
      auto d = dBegin + dd;

      if (d >= DIM) break;

      // Load state smem -> rmem
      // There is a bank conflict here, but we are not in a hot loop and we must align the state
      // indices with the input indices
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        int i = lane * packed_input_t::count +
                (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                (ii % packed_input_t::count);
        rState[ii] =
            (state_batch != params.pad_slot_id && i < DSTATE) ? toFloat(sram.state[dd][i]) : 0.f;
      }

      for (int step = 0; step < TOKENS_MTP; step++) {
        float x_value = toFloat(sram.x[step][d]);
        float out_value = d_value * x_value * int(lane == 0);  // first lane has the value

        // Compute dt value for this token
        auto dt_value = rdt[step];
        auto const dA = __expf(A_value * dt_value);

        // Process state in groups of packed_input_t::count to match B/C bank-aligned loads
        for (int ii = 0; ii < stateValuesPerThread; ii += packed_input_t::count) {
          int base_i = lane * packed_input_t::count +
                       (ii / packed_input_t::count) * warpSize * packed_input_t::count;

          // Bank-aligned load for B and C
          rB = *reinterpret_cast<packed_input_t const*>(&sram.B[step][base_i]);
          rC = *reinterpret_cast<packed_input_t const*>(&sram.C[step][base_i]);

#pragma unroll
          for (int k = 0; k < packed_input_t::count; k++) {
            auto& state_value = rState[ii + k];
            auto B_value = toFloat(rB.val[k]);
            auto C_value = toFloat(rC.val[k]);

            auto const dB = B_value * dt_value;
            auto const new_state = state_value * dA + dB * x_value;
            state_value = new_state;

            out_value += new_state * C_value;
          }

          if constexpr (sizeof(state_t) == sizeof(input_t)) {
            if (intermediate_states) {
              using packed_state_t = PackedAligned<state_t, packed_input_t::count>;
              packed_state_t rStateOut;
#pragma unroll
              for (int k = 0; k < packed_input_t::count; k++) {
                convertAndStore(&rStateOut.val[k], rState[ii + k]);
              }
              *reinterpret_cast<packed_state_t*>(&sram.state[dd][base_i]) = rStateOut;
            }
          } else {
            if (intermediate_states) {
#pragma unroll
              for (int k = 0; k < packed_input_t::count; k++) {
                convertAndStore(&sram.state[dd][base_i + k], rState[ii + k]);
              }
            }
          }
        }

        out_value = warpReduceSum(out_value);
        if (lane == 0) {
          sram.out[step][d] = out_value;
        }

        if (intermediate_states && state_batch != params.pad_slot_id) {
          for (int i = lane * load_state_t::count; i < DSTATE;
               i += warpSize * load_state_t::count) {
            auto* src = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
            auto* dst = reinterpret_cast<load_state_t*>(
                &intermediate_states[intermediate_cache_idx *
                                         params.intermediate_state_stride_batch +
                                     step * nheads * DIM * DSTATE + head * DIM * DSTATE +
                                     d * DSTATE + i]);
            *dst = *src;
          }
        }
      }

      // Update state if enabled and not padded
      if (params.update_state && state_batch != params.pad_slot_id) {
        // Store to rmem -> smem
        for (int ii = 0; ii < stateValuesPerThread; ii++) {
          int i = lane * packed_input_t::count +
                  (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                  (ii % packed_input_t::count);
          if (i < DSTATE) {
            convertAndStore(&sram.state[dd][i], rState[ii]);
          }
        }
        // store smem -> gmem
        for (int i = lane * load_state_t::count; i < DSTATE; i += warpSize * load_state_t::count) {
          auto* src = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
          *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]) = *src;
        }
      }
    }
  }

  __syncthreads();

  for (auto step = warp; step < TOKENS_MTP; step += numWarps) {
    for (auto d = lane; d < DIM; d += warpSize) {
      auto out_value = sram.out[step][d];
      if (z) {
        float z_value = toFloat(sram.z[step][d]);
        float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
        float silu_z = z_value * sig_z;
        out_value *= silu_z;
      }
      auto* dst = reinterpret_cast<input_t*>(
          &output[batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM + d]);
      convertAndStore(dst, out_value);
    }
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t>
void invokeSelectiveStateUpdateMTP(SelectiveStateMTPParams& params, SSUAlgorithm algorithm,
                                   cudaStream_t stream) {
  // MTP only supports the simple kernel
  FLASHINFER_CHECK(algorithm == SSUAlgorithm::kAuto || algorithm == SSUAlgorithm::kSimple,
                   "MTP selective_state_update only supports 'auto' or 'simple' algorithm, got ",
                   static_cast<int32_t>(algorithm));
  // Common alignment checks for all kernels
  check_ptr_alignment_input_vars<input_t>(params);

  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(params.state) % sizeof(load_state_t) == 0,
                   "state pointer must be aligned to ", sizeof(load_state_t), " bytes");
  FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                   "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

  constexpr int numWarps = 4;
  constexpr int stateRowsPerWarpPerStage = 4;
  constexpr int stageRows = stateRowsPerWarpPerStage * numWarps;

  dim3 block(warpSize, numWarps);
  dim3 grid(params.batch, params.nheads);

  auto func =
      selective_state_update_kernel_simple_mtp<input_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                               NTOKENS_MTP, DIM, DSTATE, numWarps>;
  using sram_t = SharedStorageSimple<input_t, state_t, NTOKENS_MTP, DIM, DSTATE, stageRows>;
  constexpr size_t smem_size = sizeof(sram_t);

  FLASHINFER_CUDA_CHECK(
      cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  func<<<grid, block, smem_size, stream>>>(params);
}

}  // namespace flashinfer::mamba::mtp
