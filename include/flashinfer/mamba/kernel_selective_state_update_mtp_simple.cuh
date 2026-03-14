#ifndef FLASHINFER_MAMBA_KERNEL_SELECTIVE_STATE_UPDATE_MTP_SIMPLE_CUH_
#define FLASHINFER_MAMBA_KERNEL_SELECTIVE_STATE_UPDATE_MTP_SIMPLE_CUH_

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cmath>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"

namespace flashinfer::mamba::mtp {

using namespace conversion;

template <typename input_t, typename state_t, typename state_scale_t, int TOKENS_MTP,
          int ROWS_PER_BLOCK, int DSTATE, int STATE_ROWS>
struct SharedStorageSimple {
  static constexpr bool scaleState = !std::is_same_v<state_scale_t, void>;
  alignas(alignof(PackedAligned<input_t>)) input_t x[TOKENS_MTP][ROWS_PER_BLOCK];
  alignas(alignof(PackedAligned<float>)) float out[TOKENS_MTP][ROWS_PER_BLOCK];
  alignas(alignof(PackedAligned<input_t>)) input_t z[TOKENS_MTP][ROWS_PER_BLOCK];
  alignas(alignof(PackedAligned<input_t>)) input_t B[TOKENS_MTP][DSTATE];
  alignas(alignof(PackedAligned<input_t>)) input_t C[TOKENS_MTP][DSTATE];
  alignas(alignof(PackedAligned<state_t>)) state_t state[STATE_ROWS][DSTATE];
  alignas(alignof(PackedAligned<float>))
      std::conditional_t<scaleState, state_scale_t, char> state_scale[STATE_ROWS];
};

// Grid: (batch, nheads, cdiv(DIM, ROWS_PER_BLOCK))
// When ROWS_PER_BLOCK == DIM, degenerates to the non-tiled case (blockIdx.z == 0 always).
template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int TOKENS_MTP, int DIM, int DSTATE,
          int ROWS_PER_BLOCK, int PHILOX_ROUNDS, int numWarps>
__global__ void selective_state_update_kernel_simple_mtp(SelectiveStateMTPParams params) {
  constexpr bool scaleState = !std::is_same_v<state_scale_t, void>;
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
  auto const dim_offset = blockIdx.z * ROWS_PER_BLOCK;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  // State scale pointer (only used when scaleState == true)
  [[maybe_unused]] auto* __restrict__ state_scale =
      reinterpret_cast<state_scale_t*>(params.state_scale);

  // Load device-side Philox seed once into a register
  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  auto const state_batch = (state_batch_indices) ? state_batch_indices[batch] : batch;
  auto const intermediate_cache_idx =
      intermediate_state_indices ? intermediate_state_indices[batch] : state_batch;
  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;
  state += state_ptr_offset;
  if constexpr (scaleState) {
    state_scale += state_batch * params.state_scale_stride_batch + head * DIM;
  }

  constexpr auto stateRowsPerWarpPerStage = 4;
  constexpr auto stageRows = stateRowsPerWarpPerStage * numWarps;

  extern __shared__ __align__(128) char smem[];
  auto& sram = *reinterpret_cast<SharedStorageSimple<input_t, state_t, state_scale_t, TOKENS_MTP,
                                                     ROWS_PER_BLOCK, DSTATE, stageRows>*>(smem);

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
      for (auto d = lane * load_input_t::count; d < ROWS_PER_BLOCK;
           d += warpSize * load_input_t::count) {
        if (dim_offset + d < DIM) {
          auto* dst = reinterpret_cast<load_input_t*>(&sram.x[mtp_step][d]);
          *dst = *reinterpret_cast<load_input_t const*>(
              &x[batch * params.x_stride_batch + mtp_step * params.x_stride_mtp + head * DIM +
                 dim_offset + d]);
        }
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
      for (auto d = lane * load_input_t::count; d < ROWS_PER_BLOCK;
           d += warpSize * load_input_t::count) {
        if (dim_offset + d < DIM) {
          auto* dst = reinterpret_cast<load_input_t*>(&sram.z[mtp_step][d]);
          *dst = z ? *reinterpret_cast<load_input_t const*>(
                         &z[batch * params.z_stride_batch + mtp_step * params.z_stride_mtp +
                            head * DIM + dim_offset + d])
                   : make_zeros<load_input_t>();
        }
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

  for (auto dBegin = 0; dBegin < ROWS_PER_BLOCK; dBegin += stageRows) {
    // Load state gmem -> smem
    for (int warpRow = 0; warpRow < stateRowsPerWarpPerStage; warpRow++) {
      auto dd = warp * stateRowsPerWarpPerStage + warpRow;
      auto d = dBegin + dd;
      if (dim_offset + d < DIM) {
        if (state_batch != params.pad_slot_id) {
          for (int i = lane * load_state_t::count; i < DSTATE;
               i += warpSize * load_state_t::count) {
            auto* dst = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
            *dst = *reinterpret_cast<load_state_t*>(&state[(dim_offset + d) * DSTATE + i]);
          }
        }
      }
    }
    // Load state_scale gmem -> smem (contiguous across warpRows)
    if constexpr (scaleState) {
      for (int warpRow = lane; warpRow < stateRowsPerWarpPerStage; warpRow += warpSize) {
        auto dd = warp * stateRowsPerWarpPerStage + warpRow;
        auto d = dBegin + dd;
        if (dim_offset + d < DIM && state_batch != params.pad_slot_id) {
          sram.state_scale[dd] = state_scale[dim_offset + d];
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
      auto d = dim_offset + dBegin + dd;  // global DIM index

      if (d >= DIM) break;

      // Load state smem -> rmem
      // There is a bank conflict here, but we are not in a hot loop and we must align the state
      // indices with the input indices
      float state_decode_scale = 1.f;
      if constexpr (scaleState) {
        if (state_batch != params.pad_slot_id) state_decode_scale = toFloat(sram.state_scale[dd]);
      }
      for (int ii = 0; ii < stateValuesPerThread; ii++) {
        int i = lane * packed_input_t::count +
                (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                (ii % packed_input_t::count);
        rState[ii] = (state_batch != params.pad_slot_id && i < DSTATE)
                         ? toFloat(sram.state[dd][i]) * state_decode_scale
                         : 0.f;
      }

      for (int step = 0; step < TOKENS_MTP; step++) {
        float x_value = toFloat(sram.x[step][d - dim_offset]);
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

          // Store intermediate state to smem (non-scaleState path)
          if constexpr (!scaleState) {
            if constexpr (sizeof(state_t) == sizeof(input_t)) {
              if (intermediate_states) {
                using packed_state_t = PackedAligned<state_t, packed_input_t::count>;
                packed_state_t rStateOut;
                // Philox-4x32 produces 4 random ints per call; amortize across packed elements.
                [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
                for (int k = 0; k < packed_input_t::count; k++) {
                  if constexpr (PHILOX_ROUNDS > 0) {
                    // SR only applies to fp16 state, so packed count is always >= 2.
                    static_assert(packed_input_t::count >= 2,
                                  "Stochastic rounding requires fp16 state (packed count >= 2)");
                    if (k % 4 == 0)
                      philox_randint4x<PHILOX_ROUNDS>(
                          rand_seed, state_ptr_offset + d * DSTATE + base_i + k, rand_ints[0],
                          rand_ints[1], rand_ints[2], rand_ints[3]);
                    rStateOut.val[k] = cvt_rs_f16_f32(rState[ii + k], rand_ints[k % 4] & 0x1FFFu);
                  } else {
                    convertAndStore(&rStateOut.val[k], rState[ii + k]);
                  }
                }
                *reinterpret_cast<packed_state_t*>(&sram.state[dd][base_i]) = rStateOut;
              }
            } else {
              if (intermediate_states) {
                // Philox-4x32 produces 4 random ints per call; amortize across packed elements.
                [[maybe_unused]] uint32_t rand_ints[4];
#pragma unroll
                for (int k = 0; k < packed_input_t::count; k++) {
                  if constexpr (PHILOX_ROUNDS > 0) {
                    if (k % 4 == 0)
                      philox_randint4x<PHILOX_ROUNDS>(
                          rand_seed, state_ptr_offset + d * DSTATE + base_i + k, rand_ints[0],
                          rand_ints[1], rand_ints[2], rand_ints[3]);
                    sram.state[dd][base_i + k] =
                        cvt_rs_f16_f32(rState[ii + k], rand_ints[k % 4] & 0x1FFFu);
                  } else {
                    convertAndStore(&sram.state[dd][base_i + k], rState[ii + k]);
                  }
                }
              }
            }
          }
        }

        // For scaleState + intermediate_states: quantize rState → sram.state with block scaling
        if constexpr (scaleState) {
          if (intermediate_states && state_batch != params.pad_slot_id) {
            // 2-pass: compute max, then encode
            float istate_max = std::numeric_limits<float>::lowest();
            for (int ii = 0; ii < stateValuesPerThread; ii++) {
              istate_max = fmaxf(istate_max, fabsf(rState[ii]));
            }
            istate_max = warpReduceMax(istate_max);
            istate_max = __shfl_sync(UINT32_MAX, istate_max, 0);
            float const ie_scale =
                (istate_max == 0.f)
                    ? 1.f
                    : static_cast<float>(std::numeric_limits<state_t>::max()) / istate_max;
            float const id_scale = 1.f / ie_scale;

            // Encode rState → sram.state
            for (int ii = 0; ii < stateValuesPerThread; ii++) {
              int i = lane * packed_input_t::count +
                      (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                      (ii % packed_input_t::count);
              if (i < DSTATE) {
                convertAndStore(&sram.state[dd][i], rState[ii] * ie_scale);
              }
            }
            // Store decode scale to smem for later gmem write
            if (lane == 0) sram.state_scale[dd] = id_scale;
          }
        }

        out_value = warpReduceSum(out_value);
        if (lane == 0) {
          sram.out[step][d - dim_offset] = out_value;
        }

        if (intermediate_states && state_batch != params.pad_slot_id) {
          // Write intermediate state smem → gmem
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
          // Write intermediate state decode scale → gmem
          if constexpr (scaleState) {
            if (lane == 0) {
              auto* iscales = reinterpret_cast<float*>(params.intermediate_state_scales);
              iscales[intermediate_cache_idx * params.intermediate_state_scales_stride_batch +
                      step * nheads * DIM + head * DIM + d] = sram.state_scale[dd];
            }
          }
        }
      }

      // Update state if enabled and not padded
      if (params.update_state && state_batch != params.pad_slot_id) {
        // When intermediate_states is enabled, sram.state[dd] already holds the
        // stochastically-rounded (or scaled) state from the last token step's intermediate write.
        // Skip the redundant Philox PRNG / re-quantization and write directly to gmem.
        if (!intermediate_states) {
          if constexpr (scaleState) {
            // 2-pass quantization: compute max, then re-encode
            float new_state_max = std::numeric_limits<float>::lowest();
            for (int ii = 0; ii < stateValuesPerThread; ii++) {
              new_state_max = fmaxf(new_state_max, fabsf(rState[ii]));
            }
            new_state_max = warpReduceMax(new_state_max);
            new_state_max = __shfl_sync(UINT32_MAX, new_state_max, 0);
            float const new_encode_scale =
                (new_state_max == 0.f)
                    ? 1.f
                    : static_cast<float>(std::numeric_limits<state_t>::max()) / new_state_max;
            float const new_decode_scale = 1.f / new_encode_scale;

            // Re-encode state values and store to smem
            for (int ii = 0; ii < stateValuesPerThread; ii++) {
              int i = lane * packed_input_t::count +
                      (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                      (ii % packed_input_t::count);
              if (i < DSTATE) {
                convertAndStore(&sram.state[dd][i], rState[ii] * new_encode_scale);
              }
            }
            if (lane == 0) convertAndStore(&sram.state_scale[dd], new_decode_scale);
          } else {
            // Store to rmem -> smem
            // Philox-4x32 produces 4 random ints per call; amortize across consecutive elements.
            [[maybe_unused]] uint32_t rand_ints[4];
            for (int ii = 0; ii < stateValuesPerThread; ii++) {
              int i = lane * packed_input_t::count +
                      (ii / packed_input_t::count) * warpSize * packed_input_t::count +
                      (ii % packed_input_t::count);
              if (i < DSTATE) {
                if constexpr (PHILOX_ROUNDS > 0) {
                  if (ii % 4 == 0)
                    philox_randint4x<PHILOX_ROUNDS>(rand_seed, state_ptr_offset + d * DSTATE + i,
                                                    rand_ints[0], rand_ints[1], rand_ints[2],
                                                    rand_ints[3]);
                  sram.state[dd][i] = cvt_rs_f16_f32(rState[ii], rand_ints[ii % 4] & 0x1FFFu);
                } else {
                  convertAndStore(&sram.state[dd][i], rState[ii]);
                }
              }
            }
          }
        }
        // store smem -> gmem
        for (int i = lane * load_state_t::count; i < DSTATE; i += warpSize * load_state_t::count) {
          auto* src = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
          *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]) = *src;
        }
      }
    }
    // Store state_scale smem -> gmem (contiguous across warpRows)
    if constexpr (scaleState) {
      if (params.update_state && state_batch != params.pad_slot_id) {
        for (int warpRow = lane; warpRow < stateRowsPerWarpPerStage; warpRow += warpSize) {
          auto dd = warp * stateRowsPerWarpPerStage + warpRow;
          auto d = dim_offset + dBegin + dd;
          if (d < DIM) {
            state_scale[d] = sram.state_scale[dd];
          }
        }
      }
    }
  }

  __syncthreads();

  for (auto step = warp; step < TOKENS_MTP; step += numWarps) {
    for (auto d = lane; d < ROWS_PER_BLOCK; d += warpSize) {
      if (dim_offset + d < DIM) {
        auto out_value = sram.out[step][d];
        if (z) {
          float z_value = toFloat(sram.z[step][d]);
          float sig_z = __fdividef(1.f, (1.f + __expf(0.f - z_value)));
          float silu_z = z_value * sig_z;
          out_value *= silu_z;
        }
        auto* dst = reinterpret_cast<input_t*>(
            &output[batch * params.out_stride_batch + step * params.out_stride_mtp + head * DIM +
                    dim_offset + d]);
        convertAndStore(dst, out_value);
      }
    }
  }
}

}  // namespace flashinfer::mamba::mtp

#endif  // FLASHINFER_MAMBA_KERNEL_SELECTIVE_STATE_UPDATE_MTP_SIMPLE_CUH_
