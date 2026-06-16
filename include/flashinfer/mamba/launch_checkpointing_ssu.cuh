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

// Map a runtime heads-per-CTA `hc` (a value on the HEADS_PER_GROUP>>k chain the
// tiling heuristic produces) to a compile-time template arg.  Recurses
// HC = HC_MAX, HC_MAX/2, ... 1 and calls fn<HC>() for the largest HC <= hc, so
// the per-head smem footprint and grid.z are both compile-time constants.
template <int HC_MAX, typename Fn>
void dispatch_heads_per_cta(int hc, Fn&& fn) {
  if constexpr (HC_MAX <= 1) {
    fn.template operator()<1>();
  } else if (hc >= HC_MAX) {
    fn.template operator()<HC_MAX>();
  } else {
    dispatch_heads_per_cta<HC_MAX / 2>(hc, fn);
  }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` splits each head's DIM axis across `D_SPLIT` CTAs.
// `VARLEN` selects the packed-token gmem layout (cu_seqlens-driven).
// `launchCheckpointingSsuImpl` is the per-(D_SPLIT, VARLEN) specialization;
// `launchCheckpointingSsu` (below) is the runtime dispatcher.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int D_SPLIT, bool VARLEN>
void launchCheckpointingSsuImpl(CheckpointingSsuParams& params, int main_heads_per_cta,
                                int precompute_heads_per_cta, cudaStream_t stream) {
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
  // PDL launch attribute (EXTERNAL chain: a programmatic upstream kernel — e.g.
  // conv1d — overlapping THIS kernel).  ENABLE_PDL is JIT-stamped (see
  // checkpointing_ssu_customize_config.jinja); the kernel body gates its
  // conv1d-facing PDL PTX on the same constexpr via `if constexpr (ENABLE_PDL)`.
  // This is the ONLY PDL knob a caller controls — "does my upstream conv1d
  // overlap the SSU?".  The two-kernel split's INTERNAL precompute→main PDL is
  // NOT gated by this (see below): it is intrinsic to the split, always on.
  // When ENABLE_PDL is false the attribute is 0 — cudaLaunchKernelEx is used
  // either way per FlashInfer convention (see norm.cuh:135).
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = ENABLE_PDL ? 1 : 0;

  // ── Two-kernel split: precompute → main, when the caller provides scratch ──
  // (cb_scaled/cb_old/cumAdt_vec).  bf16/fp16 only (precompute + main require
  // 2-byte input + state).
  //
  // INTERNAL PDL (precompute → main) is ALWAYS on — it is the split's mechanism,
  // not a user knob: the main co-launches with the precompute (its attr below is
  // hard-wired to 1) and runs its precompute-independent state replay while the
  // precompute computes CB / waits on conv1d; the main's own gdc_wait then blocks
  // for the precompute to finish before it reads cb_scaled.  This is what lets
  // the replay overlap conv1d on the cliff, so it must not be disable-able.
  //
  // EXTERNAL PDL (conv1d → precompute) rides on ENABLE_PDL (`attrs`): the
  // precompute co-launches with a programmatic conv1d and gdc_waits for B/C.
  // The precompute fires cudaTriggerProgrammaticLaunchCompletion() at its TOP
  // (unconditional) so the main becomes eligible to launch immediately.
  if (params.cb_scaled != nullptr) {
    if constexpr (sizeof(state_t) == 2 && sizeof(input_t) == 2) {
      // Precompute: grid (batch, ngroups, ceil(HEADS_PER_GROUP/HEADS_PER_CTA)).
      // Heads are tiled across grid.z to fill the GPU at small batch (heuristic
      // below).  HEADS_PER_CTA is picked at runtime, dispatched to a template arg;
      // DT_SOFTPLUS is JIT-folded on the runtime flag.
      auto launch_precompute = [&]<int HEADS_PER_CTA, bool DT_SOFTPLUS>() {
        auto pfunc = checkpointing_ssu_precompute_kernel<
            input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t, NPREDICTED, MAX_WINDOW, DIM,
            DSTATE, HEADS_PER_GROUP, HEADS_PER_CTA, NUM_WARPS, DT_SOFTPLUS, VARLEN>;
        constexpr size_t psmem =
            sizeof(CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE,
                                                     NUM_WARPS, HEADS_PER_CTA>);
        if constexpr (psmem > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(pfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, psmem));
        }
        constexpr int head_tiles = (HEADS_PER_GROUP + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
        cudaLaunchConfig_t pcfg;
        pcfg.gridDim = dim3(params.batch, params.ngroups, head_tiles);
        pcfg.blockDim = dim3(warpSize, NUM_WARPS);
        pcfg.dynamicSmemBytes = psmem;
        pcfg.stream = stream;
        pcfg.attrs = attrs;
        pcfg.numAttrs = 1;
        FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&pcfg, pfunc, params));
      };
      // ── Head-tiling heuristic: balance precompute parallelism vs MAIN co-residency ──
      // Tiling heads across grid.z sets #precompute CTAs = batch·ngroups·ceil(HPG/hc) =
      // batch·nheads/hc.  S (sm_count) and the main's per-SM occupancy are device/kernel
      // constants → cache statically; the hc choice is justified at hc_ideal below.
      static int const sm_count = [] {
        int dev = 0, s = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev);
        return s > 0 ? s : 1;
      }();
      // The MAIN co-launches with the precompute (internal PDL, always on) and the two
      // compete for SMs.  occ_main = main blocks resident per SM (it owns the same big
      // smem as the monolith, so typically 1).  Query once for the real D_SPLIT kernel.
      static int const occ_main = [] {
        int o = 0;
        constexpr size_t msmem = sizeof(
            CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);
        auto mf = checkpointing_ssu_main_kernel<
            input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t, state_scale_t, NPREDICTED,
            MAX_WINDOW, DIM, DSTATE, HEADS_PER_GROUP, PHILOX_ROUNDS, NUM_WARPS, D_SPLIT, VARLEN>;
        if constexpr (msmem > 48 * 1024) {
          cudaFuncSetAttribute(mf, cudaFuncAttributeMaxDynamicSharedMemorySize, msmem);
        }
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, mf, NUM_WARPS * (int)warpSize, msmem);
        return o > 0 ? o : 1;
      }();
      // The main launches main_ctas = batch·nheads·D_SPLIT.  Cliff sweeps (b∈{8,16,64},
      // B200): hc=2 is the robust optimum (hc=1 never beat it) ONCE the main is multi-wave
      // and pipelines over the precompute.  But when the main alone fits in ≤1 resident
      // wave it is FRAGILE — a big precompute footprint spills it into a 2nd wave
      // (b=8: hc=2 → main 13.4µs vs hc=4 → 9.7µs, +3µs).  So back off to hc=4 there.  The
      // earlier "fill one precompute wave" rule ignored the main and floored to hc=1 at
      // small batch — exactly the over-saturation that starves the main.
      // (Provisional: only D_SPLIT==1 / ngroups==1 / b∈{8,16,64} measured.)  Snap to HPG>>k.
      int const main_ctas = params.batch * params.nheads * D_SPLIT;
      int const hc_ideal = (main_ctas <= sm_count * occ_main) ? 4 : 2;
      int hc = HEADS_PER_GROUP;
      while (hc > 1 && hc > hc_ideal) hc >>= 1;
      // Overrides (both snap to the HEADS_PER_GROUP>>k chain in dispatch_heads_per_cta):
      //  • precompute_heads_per_cta — the exposed HOST handle (Python-tunable; 0 = heuristic);
      //  • FLASHINFER_SSU_HEADS_PER_CTA — an ad-hoc env for quick sweeps.
      // Precedence: explicit handle > env > heuristic.
      static int const hc_env = [] {
        char const* e = std::getenv("FLASHINFER_SSU_HEADS_PER_CTA");
        return e ? std::atoi(e) : 0;
      }();
      if (hc_env > 0) hc = hc_env;
      if (precompute_heads_per_cta > 0) hc = precompute_heads_per_cta;
      dispatch_heads_per_cta<HEADS_PER_GROUP>(hc, [&]<int HEADS_PER_CTA>() {
        if (params.dt_softplus) {
          launch_precompute.template operator()<HEADS_PER_CTA, true>();
        } else {
          launch_precompute.template operator()<HEADS_PER_CTA, false>();
        }
      });

      // Main: grid (D_SPLIT, batch, ngroups·ceil(HPG/MHC)).  Per-CTA smem is UNCHANGED
      // from the monolithic — head-tiling reuses the same single buffers across the
      // MHC heads, so occupancy is identical and only the CTA count drops (fewer waves).
      // main_heads_per_cta is a HOST knob (Python-tunable); dispatch_heads_per_cta snaps
      // it to the HPG>>k chain and binds it to the MAIN_HEADS_PER_CTA template arg — it
      // never reaches the kernel as a runtime value.
      dispatch_heads_per_cta<HEADS_PER_GROUP>(main_heads_per_cta, [&]<int MHC>() {
        auto mfunc = checkpointing_ssu_main_kernel<input_t, dt_t, weight_t, matrixA_t, state_t,
                                                   stateIndex_t, state_scale_t, NPREDICTED,
                                                   MAX_WINDOW, DIM, DSTATE, HEADS_PER_GROUP,
                                                   PHILOX_ROUNDS, NUM_WARPS, D_SPLIT, VARLEN, MHC>;
        constexpr size_t msmem = sizeof(
            CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>);
        if constexpr (msmem > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(mfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, msmem));
        }
        // INTERNAL PDL: the main ALWAYS co-launches with the precompute (attr
        // hard-wired to 1, independent of ENABLE_PDL) — the split's mechanism, not
        // a user knob.  Its gdc_wait (unconditional in the kernel body) blocks for
        // the precompute before reading cb_scaled, so this is always correct.
        cudaLaunchAttribute main_attrs[1];
        main_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        main_attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t mcfg;
        // grid.z = ngroups·(HPG/MHC); MHC divides HPG (kernel static_assert), so the
        // integer divide is the exact ceil.  Each CTA owns MHC consecutive heads in a group.
        mcfg.gridDim = dim3(D_SPLIT, params.batch, params.ngroups * (HEADS_PER_GROUP / MHC));
        mcfg.blockDim = dim3(warpSize, NUM_WARPS);
        mcfg.dynamicSmemBytes = msmem;
        mcfg.stream = stream;
        mcfg.attrs = main_attrs;
        mcfg.numAttrs = 1;
        FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&mcfg, mfunc, params));
      });
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
void launchCheckpointingSsu(CheckpointingSsuParams& params, int main_heads_per_cta,
                            int precompute_heads_per_cta, cudaStream_t stream) {
  bool const is_varlen = (params.cu_seqlens != nullptr);
  auto launch = [&]<int D_SPLIT, bool VARLEN>() {
    launchCheckpointingSsuImpl<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                               state_scale_t, D_SPLIT, VARLEN>(params, main_heads_per_cta,
                                                               precompute_heads_per_cta, stream);
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
