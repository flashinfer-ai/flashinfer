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
#ifndef FLASHINFER_MAMBA_LAUNCH_CHECKPOINTING_SSU_CUH_
#define FLASHINFER_MAMBA_LAUNCH_CHECKPOINTING_SSU_CUH_

// Launcher functions for the incremental SSU kernel.
// Includes both the bf16/fp16/fp32 and 8-bit kernel headers.

#include "kernel_checkpointing_ssu.cuh"
#include "kernel_checkpointing_ssu_8bit.cuh"
#include "kernel_checkpointing_ssu_main.cuh"
#include "kernel_checkpointing_ssu_precompute.cuh"

namespace flashinfer::mamba::checkpointing {

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` splits each head's DIM axis across `D_SPLIT` CTAs.
// `VARLEN` selects the packed-token gmem layout (cu_seqlens-driven).
// `launchCheckpointingSsuImpl` is the per-(D_SPLIT, VARLEN) specialization;
// `launchCheckpointingSsu` (below) is the runtime dispatcher.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int D_SPLIT, bool VARLEN>
void launchCheckpointingSsuImpl(CheckpointingSsuParams& params, cudaStream_t stream) {
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

  // HEADS_PER_GROUP is JIT-stamped via the customize_config jinja, so only
  // one (nheads / ngroups) specialization gets baked into this .so.  The
  // wrapper has already validated `nheads / ngroups == HEADS_PER_GROUP`
  // before reaching us — the kernel cross-checks with an assert below.
  FLASHINFER_CHECK(params.nheads / params.ngroups == HEADS_PER_GROUP,
                   "nheads/ngroups (=", params.nheads / params.ngroups,
                   ") must match JIT HEADS_PER_GROUP=", HEADS_PER_GROUP);
  // PDL launch attribute.  ENABLE_PDL is JIT-stamped (see
  // checkpointing_ssu_customize_config.jinja); the kernel's body has its
  // PDL PTX gated on the same constexpr via `if constexpr (ENABLE_PDL)`, so
  // the .so contains exactly one load path.  When ENABLE_PDL is false the
  // attribute is set to 0 (effectively no PDL) — cudaLaunchKernelEx is
  // used either way per FlashInfer convention (see norm.cuh:135).
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = ENABLE_PDL ? 1 : 0;

  // ── Two-kernel split: precompute → main, when the caller provides scratch ──
  // (cb_scaled/cb_old/cumAdt_vec).  bf16/fp16 only (precompute + main require
  // 2-byte input + state).  Both launched on `stream` with the same
  // programmatic-serialization attr: the conv1d→precompute→main PDL chain
  // overlaps when ENABLE_PDL is stamped; otherwise stream order serializes them
  // (still correct — the main reads cb_scaled the precompute wrote).  The
  // precompute does NOT yet trigger early-completion, so the main's gdc_wait
  // resolves at precompute completion — correct, no overlap; the early-trigger
  // overlap is the S9 refinement.
  if (params.cb_scaled != nullptr) {
    if constexpr (sizeof(state_t) == 2 && sizeof(input_t) == 2) {
      // Precompute: grid (batch, ngroups, 1) — one CTA/group (HEADS_PER_CTA ==
      // HEADS_PER_GROUP, first cut).  DT_SOFTPLUS is JIT-folded; dispatch on the
      // runtime flag.
      auto launch_precompute = [&]<bool DT_SOFTPLUS>() {
        auto pfunc =
            checkpointing_ssu_precompute_kernel<input_t, dt_t, weight_t, matrixA_t, state_t,
                                                stateIndex_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                                HEADS_PER_GROUP, NUM_WARPS, DT_SOFTPLUS, VARLEN>;
        constexpr size_t psmem = sizeof(
            CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE, NUM_WARPS>);
        if constexpr (psmem > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(pfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, psmem));
        }
        cudaLaunchConfig_t pcfg;
        pcfg.gridDim = dim3(params.batch, params.ngroups, 1);
        pcfg.blockDim = dim3(warpSize, NUM_WARPS);
        pcfg.dynamicSmemBytes = psmem;
        pcfg.stream = stream;
        pcfg.attrs = attrs;
        pcfg.numAttrs = 1;
        FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&pcfg, pfunc, params));
      };
      if (params.dt_softplus) {
        launch_precompute.template operator()<true>();
      } else {
        launch_precompute.template operator()<false>();
      }

      // Main: grid (D_SPLIT, batch, nheads), same smem as the monolithic.
      auto mfunc =
          checkpointing_ssu_main_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                        state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                        HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, D_SPLIT, VARLEN>;
      constexpr size_t msmem = sizeof(
          CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);
      if constexpr (msmem > 0) {
        FLASHINFER_CUDA_CHECK(
            cudaFuncSetAttribute(mfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, msmem));
      }
      cudaLaunchConfig_t mcfg;
      mcfg.gridDim = dim3(D_SPLIT, params.batch, params.nheads);
      mcfg.blockDim = dim3(warpSize, NUM_WARPS);
      mcfg.dynamicSmemBytes = msmem;
      mcfg.stream = stream;
      mcfg.attrs = attrs;
      mcfg.numAttrs = 1;
      FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&mcfg, mfunc, params));
      return;
    } else {
      FLASHINFER_CHECK(false,
                       "two-kernel SSU split (cb_scaled provided) requires 2-byte input + state "
                       "(bf16/fp16); got sizeof(input_t)=",
                       sizeof(input_t), ", sizeof(state_t)=", sizeof(state_t));
    }
  }

  auto launch_kernel = [&]() {
    if constexpr (sizeof(state_t) == 1) {
      // int8 chain rewrite — uses checkpointing_ssu_kernel_8bit +
      // CheckpointingSsuStorage8bit.  Only D_SPLIT == 1 is valid (the wrapper
      // asserts this); D_SPLIT == 2 still gets template-instantiated by the
      // public dispatcher's switch but is unreachable at runtime — gate the
      // body with `if constexpr (D_SPLIT == 1)` so that path doesn't launch.
      if constexpr (D_SPLIT == 1) {
        auto func =
            checkpointing_ssu_kernel_8bit<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                          state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                          HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, VARLEN>;
        constexpr size_t smem_size =
            sizeof(CheckpointingSsuStorage8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA,
                                               DSTATE>);

        if constexpr (smem_size > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }

        cudaLaunchConfig_t config;
        config.gridDim = dim3(D_SPLIT, params.batch, params.nheads);
        config.blockDim = dim3(warpSize, NUM_WARPS);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = 1;
        FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&config, func, params));
      } else {
        FLASHINFER_CHECK(false,
                         "checkpointing_ssu_kernel_8bit: unsupported D_SPLIT != 1 for 8-bit "
                         "state_t (got D_SPLIT=",
                         D_SPLIT, ")");
      }
    } else {
      // Generic kernel: bf16 / fp16 / fp32 state, supports D_SPLIT ∈ {1, 2}.
      auto func =
          checkpointing_ssu_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                   state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                   HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, D_SPLIT, VARLEN>;

      constexpr size_t smem_size = sizeof(
          CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);

      if constexpr (smem_size > 0) {
        FLASHINFER_CUDA_CHECK(
            cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }

      // Grid is (D_SPLIT, batch, nheads).  D-tile is the fastest axis so the
      // `D_SPLIT` CTAs of the same head land on adjacent SMs and share L2
      // lines for the redundantly-loaded inputs (C, B, dt, ...).
      cudaLaunchConfig_t config;
      config.gridDim = dim3(D_SPLIT, params.batch, params.nheads);
      config.blockDim = dim3(warpSize, NUM_WARPS);
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 1;
      FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&config, func, params));
    }
  };

  launch_kernel();
}

// Public dispatcher: routes on `params.d_split` ({1, 2}) and varlen
// (`params.cu_seqlens != nullptr` → VARLEN=true).  Each (D_SPLIT, VARLEN)
// pair gets its own template specialization — the JIT URI distinguishes them
// only via `d_split` today, so the same compiled `.so` will hold all four
// specializations after this commit.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t>
void launchCheckpointingSsu(CheckpointingSsuParams& params, cudaStream_t stream) {
  bool const is_varlen = (params.cu_seqlens != nullptr);
  auto launch = [&]<int D_SPLIT, bool VARLEN>() {
    launchCheckpointingSsuImpl<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                               state_scale_t, D_SPLIT, VARLEN>(params, stream);
  };
  auto launch_d_split = [&]<int D_SPLIT>() {
    if (is_varlen) {
      launch.template operator()<D_SPLIT, true>();
    } else {
      launch.template operator()<D_SPLIT, false>();
    }
  };
  switch (params.d_split) {
    case 1:
      launch_d_split.template operator()<1>();
      break;
    case 2:
      launch_d_split.template operator()<2>();
      break;
    default:
      FLASHINFER_CHECK(false, "Unsupported d_split: ", params.d_split,
                       ".  Allowed values: {1, 2}.  d_split=4 needs "
                       "warp-count restructure.");
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_LAUNCH_CHECKPOINTING_SSU_CUH_
