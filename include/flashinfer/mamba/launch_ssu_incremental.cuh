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
#ifndef FLASHINFER_MAMBA_LAUNCH_SSU_INCREMENTAL_CUH_
#define FLASHINFER_MAMBA_LAUNCH_SSU_INCREMENTAL_CUH_

// Launcher functions for the incremental SSU kernel.
// Includes both the bf16/fp16/fp32 and 8-bit kernel headers.

#include "kernel_ssu_incremental.cuh"
#include "kernel_ssu_incremental_8bit.cuh"

namespace flashinfer::mamba::incremental {

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` splits each head's DIM axis across `D_SPLIT` CTAs.
// `launchSsuIncrementalImpl` is the per-D_SPLIT specialization;
// `launchSsuIncremental` (below) is the runtime dispatcher.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int D_SPLIT>
void launchSsuIncrementalImpl(SsuIncrementalParams& params, cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;

  FLASHINFER_CHECK(params.nheads % params.ngroups == 0, "nheads (", params.nheads,
                   ") must be divisible by ngroups (", params.ngroups, ")");

  // cp.async.ca with .L2::128B requires 16B-aligned pointers (128-bit / sizeof element).
  // The .L2::128B hint further requires the base address to be 128B-aligned for full
  // cache line utilization, but the hardware only faults on < 16B alignment.
  // All cp.async-loaded operands need 16B alignment; output is also vectorized
  // (Pair<input_t> stores partitioned by m16n8k16 partition_C — base must be at
  // least 16B-aligned for the stride math to keep per-thread stores aligned).
  FLASHINFER_CHECK_ALIGNMENT(params.B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.C, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.x, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.state, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.old_x, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.old_B, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.output, 16);
  if (params.z != nullptr) {
    FLASHINFER_CHECK_ALIGNMENT(params.z, 16);
  }

  // Per-CTA D = DIM / D_SPLIT.  Smem footprint shrinks for D-owned
  // buffers (state, x, z, old_x); non-D buffers (B, C, old_B, scalars) unchanged.
  constexpr int D_PER_CTA = DIM / D_SPLIT;

  auto dispatch_hpg = [&]<int HEADS_PER_GROUP>() {
    if constexpr (sizeof(state_t) == 1) {
      // int8 chain rewrite — uses ssu_incremental_kernel_8bit +
      // SsuIncrementalStorage8bit.  Only D_SPLIT == 1 is valid (the wrapper
      // asserts this); D_SPLIT == 2 still gets template-instantiated by the
      // public dispatcher's switch but is unreachable at runtime — gate the
      // body with `if constexpr (D_SPLIT == 1)` so that path doesn't launch.
      if constexpr (D_SPLIT == 1) {
        auto func =
            ssu_incremental_kernel_8bit<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                        state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                        HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS>;
        constexpr size_t smem_size = sizeof(
            SsuIncrementalStorage8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);

        dim3 grid(D_SPLIT, params.batch, params.nheads);
        dim3 block(warpSize, NUM_WARPS);

        if constexpr (smem_size > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        func<<<grid, block, smem_size, stream>>>(params);
      }
    } else {
      // Generic kernel: bf16 / fp16 / fp32 state, supports D_SPLIT ∈ {1, 2}.
      auto func = ssu_incremental_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                         state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                         HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, D_SPLIT>;

      constexpr size_t smem_size = sizeof(
          SsuIncrementalStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);

      // Grid is (D_SPLIT, batch, nheads).  D-tile is the fastest axis so the
      // `D_SPLIT` CTAs of the same head land on adjacent SMs and share L2
      // lines for the redundantly-loaded inputs (C, B, dt, ...).
      dim3 grid(D_SPLIT, params.batch, params.nheads);
      dim3 block(warpSize, NUM_WARPS);

      if constexpr (smem_size > 0) {
        FLASHINFER_CUDA_CHECK(
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }
      func<<<grid, block, smem_size, stream>>>(params);
    }
  };

  dispatchRatio(params, std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64>{},
                [&]<int HPG>() { dispatch_hpg.template operator()<HPG>(); });
}

// Public dispatcher: routes on `params.d_split`.  Allowed values: {1, 2}.
// D_SPLIT=4 (D_PER_CTA=16) requires warp-count restructure for the output
// MMA's `_1×4` layout.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchSsuIncremental(SsuIncrementalParams& params, cudaStream_t stream) {
  auto launch = [&]<int D_SPLIT>() {
    launchSsuIncrementalImpl<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                             state_scale_t, D_SPLIT>(params, stream);
  };
  switch (params.d_split) {
    case 1:
      launch.template operator()<1>();
      break;
    case 2:
      launch.template operator()<2>();
      break;
    default:
      FLASHINFER_CHECK(false, "Unsupported d_split: ", params.d_split,
                       ".  Allowed values: {1, 2}.  d_split=4 needs "
                       "warp-count restructure.");
  }
}

}  // namespace flashinfer::mamba::incremental

#endif  // FLASHINFER_MAMBA_LAUNCH_SSU_INCREMENTAL_CUH_
