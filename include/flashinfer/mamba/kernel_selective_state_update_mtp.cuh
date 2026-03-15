#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cuda/barrier>
#include <iostream>
#include <type_traits>

#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "common.cuh"
#include "conversion.cuh"
#include "create_tensor_map.cuh"

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

// Grid: (batch_or_n_sequences, nheads, cdiv(DIM, ROWS_PER_BLOCK))
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
  auto const* __restrict__ cu_seqlens =
      reinterpret_cast<cuSeqlensIndex_t const*>(params.cu_seqlens);
  auto const* __restrict__ num_accepted_tokens =
      reinterpret_cast<numAcceptedIndex_t const*>(params.num_accepted_tokens);
  auto const* __restrict__ dst_state_batch_indices =
      reinterpret_cast<stateIndex_t const*>(params.dst_state_batch_indices);
  bool const dt_softplus = params.dt_softplus;

  int const nheads = params.nheads;
  int const ngroups = params.ngroups;

  auto const seq_idx = blockIdx.x;
  auto const head = blockIdx.y;
  auto const dim_offset = blockIdx.z * ROWS_PER_BLOCK;
  auto const group = head / (nheads / ngroups);
  auto lane = threadIdx.x % warpSize;
  auto warp = threadIdx.y;

  int bos;
  int seq_len;
  bool const has_cu_seqlens = (cu_seqlens != nullptr);
  if (has_cu_seqlens) {
    bos = __ldg(&cu_seqlens[seq_idx]);
    int eos = __ldg(&cu_seqlens[seq_idx + 1]);
    seq_len = eos - bos;
    if (seq_len <= 0) return;
  } else {
    bos = 0;
    seq_len = TOKENS_MTP;
  }

  int init_token_idx = 0;
  if (num_accepted_tokens) {
    int num_accepted = __ldg(&num_accepted_tokens[seq_idx]);
    init_token_idx = max(num_accepted - 1, 0);
  }

  // State scale pointer (only used when scaleState == true)
  [[maybe_unused]] auto* __restrict__ state_scale =
      reinterpret_cast<state_scale_t*>(params.state_scale);

  // Load device-side Philox seed once into a register
  [[maybe_unused]] int64_t const rand_seed = params.rand_seed ? *params.rand_seed : 0;

  int64_t state_batch;
  if (state_batch_indices) {
    state_batch = static_cast<int64_t>(
        state_batch_indices[seq_idx * params.state_batch_indices_stride_batch +
                            init_token_idx * params.state_batch_indices_stride_T]);
  } else {
    state_batch = static_cast<int64_t>(seq_idx);
  }
  auto const intermediate_cache_idx =
      intermediate_state_indices ? intermediate_state_indices[seq_idx] : state_batch;
  auto const state_ptr_offset = state_batch * params.state_stride_batch + head * DIM * DSTATE;
  state += state_ptr_offset;
  if constexpr (scaleState) {
    state_scale += state_batch * params.state_scale_stride_batch + head * DIM;
  }

  int64_t const x_base = has_cu_seqlens ? (int64_t)bos * params.x_stride_batch
                                        : (int64_t)seq_idx * params.x_stride_batch;
  int64_t const x_tstride = has_cu_seqlens ? params.x_stride_batch : params.x_stride_mtp;

  int64_t const dt_base = has_cu_seqlens ? (int64_t)bos * params.dt_stride_batch
                                         : (int64_t)seq_idx * params.dt_stride_batch;
  int64_t const dt_tstride = has_cu_seqlens ? params.dt_stride_batch : params.dt_stride_mtp;

  int64_t const B_base = has_cu_seqlens ? (int64_t)bos * params.B_stride_batch
                                        : (int64_t)seq_idx * params.B_stride_batch;
  int64_t const B_tstride = has_cu_seqlens ? params.B_stride_batch : params.B_stride_mtp;

  int64_t const C_base = has_cu_seqlens ? (int64_t)bos * params.C_stride_batch
                                        : (int64_t)seq_idx * params.C_stride_batch;
  int64_t const C_tstride = has_cu_seqlens ? params.C_stride_batch : params.C_stride_mtp;

  int64_t const out_base = has_cu_seqlens ? (int64_t)bos * params.out_stride_batch
                                          : (int64_t)seq_idx * params.out_stride_batch;
  int64_t const out_tstride = has_cu_seqlens ? params.out_stride_batch : params.out_stride_mtp;

  int64_t const z_base = z ? (has_cu_seqlens ? (int64_t)bos * params.z_stride_batch
                                             : (int64_t)seq_idx * params.z_stride_batch)
                           : 0;
  int64_t const z_tstride = z ? (has_cu_seqlens ? params.z_stride_batch : params.z_stride_mtp) : 0;

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
          if (mtp_step < seq_len) {
            *dst = *reinterpret_cast<load_input_t const*>(
                &x[x_base + mtp_step * x_tstride + head * DIM + dim_offset + d]);
          } else {
            *dst = make_zeros<load_input_t>();
          }
        }
      }
    }
  } else if (warp == 1) {  // Load B: gmem -> smem
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.B[mtp_step][i]);
        if (mtp_step < seq_len) {
          *dst = *reinterpret_cast<load_input_t const*>(
              &B[B_base + mtp_step * B_tstride + group * DSTATE + i]);
        } else {
          *dst = make_zeros<load_input_t>();
        }
      }
    }
  } else if (warp == 2) {  // Load z: gmem -> smem
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto d = lane * load_input_t::count; d < ROWS_PER_BLOCK;
           d += warpSize * load_input_t::count) {
        if (dim_offset + d < DIM) {
          auto* dst = reinterpret_cast<load_input_t*>(&sram.z[mtp_step][d]);
          if (z && mtp_step < seq_len) {
            *dst = *reinterpret_cast<load_input_t const*>(
                &z[z_base + mtp_step * z_tstride + head * DIM + dim_offset + d]);
          } else {
            *dst = make_zeros<load_input_t>();
          }
        }
      }
    }
  }
  // Load C: gmem -> smem
  else if (warp == 3) {
    for (int mtp_step = 0; mtp_step < TOKENS_MTP; mtp_step++) {
      for (auto i = lane * load_input_t::count; i < DSTATE; i += warpSize * load_input_t::count) {
        auto* dst = reinterpret_cast<load_input_t*>(&sram.C[mtp_step][i]);
        if (mtp_step < seq_len) {
          *dst = *reinterpret_cast<load_input_t const*>(
              &C[C_base + mtp_step * C_tstride + group * DSTATE + i]);
        } else {
          *dst = make_zeros<load_input_t>();
        }
      }
    }
  }

  float rdt[TOKENS_MTP];
  for (int step = 0; step < TOKENS_MTP; step++) {
    if (step < seq_len) {
      auto dt_value = dt_bias_value + toFloat(dt[dt_base + step * dt_tstride + head]);
      if (dt_softplus) {
        dt_value = thresholded_softplus(dt_value);
      }
      rdt[step] = dt_value;
    } else {
      rdt[step] = 0.f;
    }
  }

  __syncthreads();

  bool const has_dst_indices = (dst_state_batch_indices != nullptr);
  bool const has_intermediate = (intermediate_states != nullptr);

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
        if (step >= seq_len) break;

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
              if (has_intermediate || has_dst_indices) {
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
              if (has_intermediate || has_dst_indices) {
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

        // For scaleState + per-step writes: quantize rState → sram.state with block scaling
        if constexpr (scaleState) {
          if ((has_intermediate || has_dst_indices) && state_batch != params.pad_slot_id) {
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

        if (state_batch != params.pad_slot_id) {
          if (has_dst_indices) {
            auto dst_idx = static_cast<int64_t>(
                dst_state_batch_indices[seq_idx * params.dst_state_batch_indices_stride_batch +
                                        step * params.dst_state_batch_indices_stride_T]);
            if (dst_idx != params.pad_slot_id) {
              auto* dst_state_ptr = reinterpret_cast<state_t*>(params.state);
              for (int i = lane * load_state_t::count; i < DSTATE;
                   i += warpSize * load_state_t::count) {
                auto* src = reinterpret_cast<load_state_t*>(&sram.state[dd][i]);
                *reinterpret_cast<load_state_t*>(
                    &dst_state_ptr[dst_idx * params.state_stride_batch + head * DIM * DSTATE +
                                   d * DSTATE + i]) = *src;
              }
              if constexpr (scaleState) {
                if (lane == 0) {
                  auto* dst_scale = reinterpret_cast<state_scale_t*>(params.state_scale);
                  dst_scale[dst_idx * params.state_scale_stride_batch + head * DIM + d] =
                      sram.state_scale[dd];
                }
              }
            }
          } else if (has_intermediate) {
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
      }

      // Update state if enabled and not padded
      if (params.update_state && state_batch != params.pad_slot_id && !has_dst_indices) {
        // When intermediate_states is enabled, sram.state[dd] already holds the
        // stochastically-rounded (or scaled) state from the last token step's intermediate write.
        // Skip the redundant Philox PRNG / re-quantization and write directly to gmem.
        if (!has_intermediate) {
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
      if (params.update_state && state_batch != params.pad_slot_id && !has_dst_indices) {
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
    if (step >= seq_len) continue;
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
            &output[out_base + step * out_tstride + head * DIM + dim_offset + d]);
        convertAndStore(dst, out_value);
      }
    }
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void invokeSelectiveStateUpdateMTP(SelectiveStateMTPParams& params, SSUAlgorithm algorithm,
                                   cudaStream_t stream) {
  constexpr bool scaleState = !std::is_same_v<state_scale_t, void>;
  // Stochastic rounding is only implemented for fp16 state
  if constexpr (PHILOX_ROUNDS > 0) {
    static_assert(std::is_same_v<state_t, half>,
                  "Stochastic rounding (PHILOX_ROUNDS > 0) only supports fp16 state");
  }
  // MTP only supports the simple kernel
  FLASHINFER_CHECK(algorithm == SSUAlgorithm::kAuto || algorithm == SSUAlgorithm::kSimple,
                   "MTP selective_state_update only supports 'auto' or 'simple' algorithm, got ",
                   static_cast<int32_t>(algorithm));
  // Common alignment checks for all kernels
  check_ptr_alignment_input_vars<input_t>(params);

  constexpr auto stateLoadSize = getVectorLoadSizeForFullUtilization<state_t, DSTATE>();
  using load_state_t = PackedAligned<state_t, stateLoadSize>;

  FLASHINFER_CHECK_ALIGNMENT(params.state, sizeof(load_state_t));
  FLASHINFER_CHECK((params.dim * params.dstate * sizeof(state_t)) % sizeof(load_state_t) == 0,
                   "state head stride must be aligned to ", sizeof(load_state_t), " bytes");

  constexpr int numWarps = 4;
  constexpr int stateRowsPerWarpPerStage = 4;
  constexpr int stateRowsPerBlockPerStage = stateRowsPerWarpPerStage * numWarps;
  int const total_tiles = params.batch * params.nheads;
  int const num_sms = GetCudaMultiProcessorCount();

  dim3 block(warpSize, numWarps);
  if (total_tiles < num_sms * 2) {
    // Small tile per CTA (stateRowsPerBlockPerStage * DSTATE): split dim across grid.z for GPU
    // occupancy
    int const dim_tiles = (DIM + stateRowsPerBlockPerStage - 1) / stateRowsPerBlockPerStage;
    dim3 grid(params.batch, params.nheads, dim_tiles);
    auto func = selective_state_update_kernel_simple_mtp<
        input_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t, NTOKENS_MTP, DIM,
        DSTATE, stateRowsPerBlockPerStage, PHILOX_ROUNDS, numWarps>;
    using sram_t =
        SharedStorageSimple<input_t, state_t, state_scale_t, NTOKENS_MTP, stateRowsPerBlockPerStage,
                            DSTATE, stateRowsPerBlockPerStage>;
    constexpr size_t smem_size = sizeof(sram_t);
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    func<<<grid, block, smem_size, stream>>>(params);
  } else {
    // Full tile per CTA (DIM * DSTATE): enough blocks for occupancy, no dim splitting needed
    dim3 grid(params.batch, params.nheads);
    auto func = selective_state_update_kernel_simple_mtp<input_t, weight_t, matrixA_t, state_t,
                                                         stateIndex_t, state_scale_t, NTOKENS_MTP,
                                                         DIM, DSTATE, DIM, PHILOX_ROUNDS, numWarps>;
    using sram_t = SharedStorageSimple<input_t, state_t, state_scale_t, NTOKENS_MTP, DIM, DSTATE,
                                       stateRowsPerBlockPerStage>;
    constexpr size_t smem_size = sizeof(sram_t);
    FLASHINFER_CUDA_CHECK(
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    func<<<grid, block, smem_size, stream>>>(params);
  }
}

}  // namespace flashinfer::mamba::mtp
