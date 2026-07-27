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

// Map a runtime precompute warp count to a compile-time template arg.
// Valid values: 4, 8, 16 (minimum 4 — the CB matmul uses 4 warps).
template <typename Fn>
void dispatch_precompute_num_warps(int nw, Fn&& fn) {
  if (nw >= 16) {
    fn.template operator()<16>();
  } else if (nw >= 8) {
    fn.template operator()<8>();
  } else {
    fn.template operator()<4>();
  }
}

// ── Dispatcher ─────────────────────────────────────────────────────────────
// `D_SPLIT` splits each head's DIM axis across `D_SPLIT` CTAs.
// `VARLEN` selects the packed-token gmem layout (cu_seqlens-driven).
// `launchCheckpointingSsuImpl` is the per-(D_SPLIT, VARLEN) specialization;
// `launchCheckpointingSsu` (below) is the runtime dispatcher.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int D_SPLIT, bool VARLEN>
void launchCheckpointingSsuImpl(CheckpointingSsuParams& params, int precompute_heads_per_cta,
                                cudaStream_t stream) {
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
  FLASHINFER_CHECK_ALIGNMENT(params.x_cache, 16);
  FLASHINFER_CHECK_ALIGNMENT(params.B_cache, 16);
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
  // NGROUPS is JIT-stamped too; the ratio check above doesn't pin the absolute
  // group count, and the kernel's unflatten bakes NHEADS = NGROUPS * HEADS_PER_GROUP.
  FLASHINFER_CHECK(params.ngroups == NGROUPS, "ngroups (=", params.ngroups,
                   ") must match JIT NGROUPS=", NGROUPS);
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
  // (cb_scaled/cb_old/cumAdt_vec).  Requires 2-byte input (bf16/fp16
  // activations); state may be 2-byte (LDSM path) or 4-byte f32 (LDS.64 +
  // in-register narrow — see make_a_s2r).  8-bit state has no two-kernel
  // split (the monolithic checkpointing_ssu_kernel_8bit handles it).
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
    if constexpr ((sizeof(state_t) == 2 || sizeof(state_t) == 4) && sizeof(input_t) == 2) {
      // Precompute: grid (batch, ngroups, ceil(HEADS_PER_GROUP/HEADS_PER_CTA)).
      // Heads are tiled across grid.z to fill the GPU at small batch (heuristic
      // below).  HEADS_PER_CTA is picked at runtime, dispatched to a template arg;
      // DT_SOFTPLUS is JIT-folded on the runtime flag.
      auto launch_precompute =
          [&]<int HEADS_PER_CTA, int PRECOMPUTE_NUM_WARPS, bool DT_SOFTPLUS>() {
            auto pfunc =
                checkpointing_ssu_precompute_kernel<input_t, dt_t, weight_t, matrixA_t, state_t,
                                                    stateIndex_t, NPREDICTED, MAX_WINDOW, DIM,
                                                    DSTATE, HEADS_PER_GROUP, HEADS_PER_CTA,
                                                    PRECOMPUTE_NUM_WARPS, DT_SOFTPLUS, VARLEN>;
            constexpr size_t psmem =
                sizeof(CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE,
                                                         PRECOMPUTE_NUM_WARPS, HEADS_PER_CTA>);
            if constexpr (psmem > 0) {
              FLASHINFER_CUDA_CHECK(
                  cudaFuncSetAttribute(pfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, psmem));
            }
            constexpr int head_tiles = (HEADS_PER_GROUP + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
            cudaLaunchConfig_t pcfg;
            pcfg.gridDim = dim3(params.batch, params.ngroups, head_tiles);
            pcfg.blockDim = dim3(warpSize, PRECOMPUTE_NUM_WARPS);
            pcfg.dynamicSmemBytes = psmem;
            pcfg.stream = stream;
            pcfg.attrs = attrs;
            pcfg.numAttrs = 1;
            FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&pcfg, pfunc, params));
          };
      static int const num_sms = [] {
        int dev = 0, n = 0;
        FLASHINFER_CUDA_CHECK(cudaGetDevice(&dev));
        FLASHINFER_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
        return n;
      }();

      // ── Head-tiling default: heads/CTA == warps/CTA (1 head/warp), coupled in
      // launch_precompute (PRECOMPUTE_NUM_WARPS = max(hc, 4)).  hc=8 (→ 8 warps) measured as
      // the robust optimum on B200, across BOTH the cliff and large batch:
      //   • small batch: the precompute is latency-bound + under-occupied — 8 warps double
      //     the resident warps (occ 6%→12%, matching triton's _dynamic_precompute) to hide
      //     the load latency (no-issue 93%→88%, b=16 11.3µs→10.7µs).
      //   • large batch: fewer head-tiles per group HALVE the redundant per-group C·B matmul
      //     (b=1024: hc=8 136µs vs the old hc=2 143µs — now beats triton-replay-pm).
      // At tiny grids the precompute is so under-occupied that spreading into MORE, smaller
      // CTAs (hc=4 → 4 tiles vs hc=8 → 2) lights up more SMs and edges ahead (+0.1–0.2µs),
      // so back off there.  The boundary — measured as b=8 at ng=1/HPG=16 on the 148-SM B200 —
      // is the hc=8 grid (batch·ngroups·HPG/8 CTAs) covering ≥ ~1/10 of the SMs; expressing it
      // that way (instead of raw `batch >= 8`) scales it with ngroups (TP) and GPU size.
      // hc=16 regressed small batch (too few CTAs, 512 thr) without helping large — not worth
      // a 3rd tier.  Tunable via precompute_heads_per_cta / the env below;
      // dispatch_heads_per_cta snaps to the HEADS_PER_GROUP>>k chain (clamps to HPG).
      int64_t const hc8_grid = static_cast<int64_t>(params.batch) * params.ngroups *
                               (HEADS_PER_GROUP >= 8 ? HEADS_PER_GROUP / 8 : 1);
      int const hc_ideal = (hc8_grid * 10 >= num_sms) ? 8 : 4;
      int hc = HEADS_PER_GROUP;
      while (hc > 1 && hc > hc_ideal) hc >>= 1;
      // Overrides (both snap to the HEADS_PER_GROUP>>k chain in dispatch_heads_per_cta):
      //  • precompute_heads_per_cta — the exposed HOST handle (Python-tunable; 0 = heuristic);
      //  • FLASHINFER_SSU_HEADS_PER_CTA — an ad-hoc env for quick sweeps.
      // Precedence: explicit handle > env > heuristic.  Envs are read PER LAUNCH
      // (getenv is ~ns against a launch; not static, so sweeps/tests see each change).
      int const hc_env = [] {
        char const* e = std::getenv("FLASHINFER_SSU_HEADS_PER_CTA");
        return e ? std::atoi(e) : 0;
      }();
      if (hc_env > 0) hc = hc_env;
      if (precompute_heads_per_cta > 0) hc = precompute_heads_per_cta;
      int const pnw_env = [] {
        char const* e = std::getenv("FLASHINFER_SSU_PRECOMPUTE_NUM_WARPS");
        return e ? std::atoi(e) : 0;
      }();
      dispatch_heads_per_cta<HEADS_PER_GROUP>(hc, [&]<int HEADS_PER_CTA>() {
        int pnw = (HEADS_PER_CTA > 4) ? HEADS_PER_CTA : 4;  // default: 1 head/warp, min 4
        if (pnw_env > 0) pnw = pnw_env;
        dispatch_precompute_num_warps(pnw, [&]<int PNW>() {
          if (params.dt_softplus) {
            launch_precompute.template operator()<HEADS_PER_CTA, PNW, true>();
          } else {
            launch_precompute.template operator()<HEADS_PER_CTA, PNW, false>();
          }
        });
      });

      // ── Persistent main: 1D grid-stride loop over single-head work-units ──
      // One work-unit = one head; the grid-stride loop is the head-tiling (no host knob).
      // Launch min(cta_per_sm·NUM_SMS, total_work) CTAs; each grid-strides over work-units.
      // Fewer CTAs leave SM room to co-reside with conv1d (closing the no-write gap to
      // triton-replay-pm), and consecutive work-units (head_tile-fastest, see kernel) keep
      // per-group C/old_B L2-hot.  Default cta_per_sm huge ⇒ grid == total_work ⇒ one
      // work-unit/CTA ⇒ bit-identical to the old (D_SPLIT, batch, nheads) launch.
      //
      // MAIN_NUM_WARPS=4: the main is NUM_WARPS-generic but feed-bound (long/short scoreboard),
      // not warp-hideable, so 8 warps buys nothing and costs finer MMA tiling + publish barriers.
      constexpr int MAIN_NUM_WARPS = 4;

      int const main_total_work =
          D_SPLIT * static_cast<int>(params.batch) * static_cast<int>(params.nheads);

      // ── Launch-regime defaults: {NUM_STAGES, cta_per_sm} per (state width, total work) ──
      // THE tuning table — the defaults here must stand on their own; the
      // FLASHINFER_SSU_MAIN_{PIPELINE_STAGES,CTA_PER_SM} envs are optional overrides for
      // experiments, never required.  Tuned on the production-shaped workload (mixed PNATs +
      // conv1d/PDL, bench_ssu_checkpoint_mixed.py):
      //   • 2-byte state → {1, 16} at every work level.  Occupancy regime: ~29 KB/block → 6
      //     blocks/SM at the derived reg cap; depth 2 loses b=1024 bf16 (94.0 vs 79.6 µs
      //     no-write).  cps=16 slightly oversubscribed beats exactly-resident (133.6 vs 139.6);
      //     below-occupancy grids are catastrophic.
      //   • 4-byte state → {1, 16} at every work level.  In mixed batches stg1/cps16 beats
      //     stg2/cps4 at EVERY batch (b=32: 14.05 vs 17.22; b=1024: 154.9 vs 163.3 µs e2e):
      //     the cps=4 grid's long-lived CTAs eat a binomial write-count straggler tail
      //     (~9.5 µs, slot-order experiment) and can't co-reside with conv1d.  The ONE regime
      //     preferring {2, 4} is UNIFORM-PNAT kernel isolation at work ≳ 8·SMs, where depth 2
      //     covers the doubled state prefetch (122.8 vs 150.5 µs b=1024 no-write) — that mix
      //     is undetectable at launch (per-slot PNATs live on device) and isn't production-
      //     shaped, so it is NOT a default; the envs can pin it for such experiments.
      // `total_work` parameterizes the table for future batch-dependent entries (none today:
      // small batches self-cap the grid at total_work, so cps needs no small-batch entry).
      struct MainRegime {
        int stages, cta_per_sm;
      };
      auto const regime = [](int /*total_work*/) -> MainRegime { return {1, 16}; }(main_total_work);

      // Envs are read PER LAUNCH (getenv is ~ns against a launch): tests monkeypatch them
      // per-case, and a static latch would pin the first-seen value for the whole process.
      int const main_cta_per_sm = [&] {
        char const* e = std::getenv("FLASHINFER_SSU_MAIN_CTA_PER_SM");
        int const v = e ? std::atoi(e) : 0;  // unset/<=0 → regime default
        return v > 0 ? v : regime.cta_per_sm;
      }();
      int64_t const main_grid_ll = static_cast<int64_t>(main_cta_per_sm) * num_sms;
      int const main_grid =
          static_cast<int>(main_grid_ll < main_total_work ? main_grid_ll : main_total_work);

      int const main_stages = [&] {
        char const* e = std::getenv("FLASHINFER_SSU_MAIN_PIPELINE_STAGES");
        int const v = e ? std::atoi(e) : 0;  // unset → regime default
        FLASHINFER_CHECK(v == 0 || v == 1 || v == 2,
                         "FLASHINFER_SSU_MAIN_PIPELINE_STAGES must be 1 or 2, got ", v);
        return v > 0 ? v : regime.stages;
      }();

      // INTERNAL PDL: the main ALWAYS co-launches with the precompute (attr hard-wired to 1,
      // independent of ENABLE_PDL).  The one-shot gdc_wait on a CTA's first work-unit blocks for
      // the precompute before reading cb_scaled / cumAdt_vec.
      cudaLaunchAttribute main_attrs[1];
      main_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      main_attrs[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t mcfg;
      mcfg.gridDim = dim3(main_grid);
      mcfg.blockDim = dim3(warpSize, MAIN_NUM_WARPS);
      mcfg.stream = stream;
      mcfg.attrs = main_attrs;
      mcfg.numAttrs = 1;

      // NGROUPS is the jinja-stamped compile-time group count → the kernel's unflatten divides only
      // by compile-time constants (no runtime div/mod).
      auto launch_main = [&](auto stages_tag) {
        constexpr int MAIN_STAGES = decltype(stages_tag)::value;
        auto mfunc =
            checkpointing_ssu_main_kernel<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                          state_scale_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE,
                                          HEADS_PER_GROUP, PHILOX_ROUNDS, MAIN_NUM_WARPS, D_SPLIT,
                                          VARLEN, NGROUPS, MAIN_STAGES>;
        constexpr size_t msmem =
            sizeof(CheckpointingSsuMainStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA,
                                               DSTATE, MAIN_STAGES, VARLEN>);
        if constexpr (msmem > 0) {
          FLASHINFER_CUDA_CHECK(
              cudaFuncSetAttribute(mfunc, cudaFuncAttributeMaxDynamicSharedMemorySize, msmem));
        }
        mcfg.dynamicSmemBytes = msmem;
        FLASHINFER_CUDA_CHECK(cudaLaunchKernelEx(&mcfg, mfunc, params));
      };
      if (main_stages == 2) {
        launch_main(std::integral_constant<int, 2>{});
      } else {
        launch_main(std::integral_constant<int, 1>{});
      }
      return;
    } else {
      FLASHINFER_CHECK(false,
                       "two-kernel SSU split (cb_scaled provided) requires 2-byte input "
                       "(bf16/fp16) and 2- or 4-byte state (bf16/fp16/fp32); got "
                       "sizeof(input_t)=",
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
void launchCheckpointingSsu(CheckpointingSsuParams& params, int precompute_heads_per_cta,
                            cudaStream_t stream) {
  bool const is_varlen = (params.cu_seqlens != nullptr);
  auto launch = [&]<int D_SPLIT, bool VARLEN>() {
    launchCheckpointingSsuImpl<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                               state_scale_t, D_SPLIT, VARLEN>(params, precompute_heads_per_cta,
                                                               stream);
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
