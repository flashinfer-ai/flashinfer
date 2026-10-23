<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MXFP8 × MXFP4 MegaMoE

This Blackwell CuTe DSL kernel combines expert dispatch, FC1, SwiGLU,
FC2, token return, and deterministic top-k reduction. Activations use E4M3
with E8M0 scales per 32 values; weights use packed E2M1 with E8M0 scales
per 32 values. The output is BF16. The native kernel supports both
single-CTA and paired-CTA MMA, and the lean FC12 path shares its compute
implementation with MegaMoE.

The MoE EP backend is `sm100_mxfp8_mxfp4_bf16_cutedsl`. Its configuration,
weight preprocessing, and offline tuner live in
[`backends/mega/kernel/sm100/mxfp8_mxfp4_bf16_cutedsl`](../../../backends/mega/kernel/sm100/mxfp8_mxfp4_bf16_cutedsl).
Use a supported Blackwell CUDA/CuTe DSL installation. Multi-rank MegaMoE
requires the MoE EP symmetric-memory setup and peer-accessible NVLink GPUs;
all participating ranks must use the same configuration and collective order.

## Input and lifecycle contract

`hidden` and the post-SwiGLU `intermediate_size` must be positive multiples
of 128. `top_k` is in `[1, 32]` and cannot exceed the number of experts.
Experts divide evenly among the EP ranks; native dispatch supports 1–16
ranks. Symbolically, BF16 source weights have shapes
`w13=[local_experts, 2 * intermediate, hidden]` and
`w2=[local_experts, hidden, intermediate]`. Use the backend's
`preprocess_mega_weights` to obtain its packed and swizzled representation.
FC1 gate/up rows are interleaved in groups of eight. Passing an arbitrary
packed FP4 tensor without the corresponding scale layout is unsupported.

The staging path accepts BF16 inputs for quantization, or prequantized E4M3
inputs with E8M0/K32 scales. Routing IDs are int64 and routing weights are
FP32. Each FC2 route is rounded to BF16, weighted in FP32, and reduced in
fixed top-k order before the final BF16 conversion.

The result is BF16 `[tokens, hidden]`. The low-level launcher returns a
workspace view when `output=None`; copy it to retain it across launches.

Allocate the workspace, preprocess weights, stage inputs, tune, and run an
eager warmup before CUDA graph capture. Keep workspace and tensor addresses
alive through replay. A tactic change can alter workspace offsets and the
compiled kernel, so apply it outside capture and warm up again. Destroy the
workspace only after outstanding GPU work completes, collectively for EP.

## Selecting a tactic

The defaults below cover the backend configuration, tactic fields,
kernel environment switches, the integration helpers, and every offline CLI
option. General FlashInfer build/logging controls are described in the
[repository environment reference](../../../../../CLAUDE.md#environment-variables).
Defaults assume no matching persistent cache entry.

`Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig` has these fields:

| Field | Default | Meaning |
|---|---|---|
| `intermediate_size` | Required; no default | Post-SwiGLU width, a positive multiple of 128. |
| `top_k` | Required; no default | Routes per token, in `[1, 32]`. |
| `kernel_name` | `"sm100_mxfp8_mxfp4_bf16_cutedsl"` | Fixed backend name. |
| `quant` | MXFP4 weights, MXFP8 activations, BF16 output; `swizzled_scale_factors=True`, `per_token_scale=False` | Block-scaled quantization contract described above. |
| `gate_up_clamp` | `None` | No clamp. A supplied value must be finite and positive. |
| `fast_math` | `True` | Use approximate math in the SwiGLU epilogue. |
| `knobs` | `None` | Resolve through the persistent cache, then the default tactic. |

The `knobs` choices are:

| Value | Behavior |
|---|---|
| `None` | Use the general autotuner cache, then the legacy knob cache/default; participate in `flashinfer.autotune()`. |
| `{}` | Bypass the knob cache and use the default tactic. |
| A partial tactic dictionary | Bypass the knob cache and merge the supplied fields over defaults. |
| `"auto"` | Run the collective candidate sweep on the first eager compute and retain the winner. |

Explicit tactic fields override the capacity-based defaults. Use a fresh
process and workspace when changing diagnostic environment switches.
`gate_up_clamp` optionally supplies a finite,
positive SwiGLU clamp; `fast_math` defaults to `True`. Keep these numerical
settings fixed while comparing tactics.

### Default policy

MMA M is 256 with cluster `(2, 1, 1)` when `hidden % 256 == 0` and
`(2 * intermediate) % 256 == 0`; otherwise M is 128 with cluster `(1, 1, 1)`.
The following table uses **workspace capacity in tokens per rank**, rather
than the live token count or the global batch size.

| Capacity | Token tile N | Group hint | Pair-tile claim mode | Pair-tile epilogue warps | In-kernel reduction |
|---|---|---|---|---|---|
| Below 256 | 64 | 4096 | `atomic_counter` | 4 | On |
| 256–511 | 128 | 4096 | `static` | 8 | Off |
| 512 and above | 192 for pairs, 128 otherwise | 512 | `alternating` | 8 | Off |

Single-CTA defaults always use static claims and four epilogue warps.
All defaults use K128, automatic operand stages, `flag_batch=1`,
`token_back_mode="epi_warps"`, and `cta_local_counts=True`. FC2 stores use
native TMEM loads and a shared transpose with 256-byte row segments.
`persistent_clusters=None` uses the full available grid; a positive integer
caps the cluster count at device occupancy. Tune with representative routing
and live token counts to replace these heuristics.

### Tactic dictionary reference

Here, capacity means allocated tokens per rank. Pair eligibility is defined
in the default policy above; single-CTA tiles are used otherwise.

| Knob | Default | Values and effect |
|---|---|---|
| `mma_tiler_mnk` | `(M, N, 128)`; M256 for pairs, M128 otherwise; N from the capacity table | M128 or M256, N64/N128/N192. M is output features, N is routed tokens. The host schema accepts positive K multiples of 32; use K128 for the current device implementation. N256 is unsupported. |
| `cluster_shape_mnk` | `(2, 1, 1)` for pairs; `(1, 1, 1)` otherwise | M256 requires `(2, 1, 1)`. The host schema allows cluster axes 1/2/4, a final axis of 1, and at most 16 CTAs; device assembly enforces additional layout constraints. Pinning M without a cluster automatically selects the matching cluster. |
| `num_stages` | `"auto"` | `"auto"` or 2–8. Auto fits the deepest operand ring into shared memory, accounting for dispatch and epilogue exchange storage. Auto can select up to 12 stages for pairs or 8 for single-CTA tiles; a pinned value still must fit. |
| `persistent_clusters` | `None` | Use the full available grid, or cap it with a positive integer clamped to device occupancy. |
| `group_hint` | 4096 below capacity 512; 512 otherwise | Positive integer or `None`. Controls expert grouping in the static work order; smaller groups can expose FC2 earlier. Alternating claims ignore it. |
| `flag_batch` | 1 | 1–32. Number of dispatched arrivals batched before publishing readiness. |
| `load_balance_mode` | Pairs: capacity table; single CTA: `"static"` | `static`: fixed tile assignment; `atomic_counter`: dynamic claims over the same linear order; `alternating`: independent FC1/FC2 counters with FC1 warmup and dependency-aware FC2 claims. |
| `claim_order` | `"expert"` | `"expert"` or `"longest_first"`. The latter prioritizes expert token counts; it is available for explicit experiments but excluded from the default tuning sweep. |
| `weight_l2_evict_first` | `True` for pairs through capacity 512; `False` otherwise | Boolean. Give streamed weight and scale TMA loads the EVICT_FIRST L2 policy to preserve token/intermediate reuse. |
| `token_back_mode` | `"epi_warps"` | Send FC2 rows from epilogue warps. This is the only supported token-back mode. |
| `epilogue_warps` | 8 for pairs at capacity 256 or above; 4 otherwise | 4 or 8. Eight splits token groups across two warpgroups, increasing threads and shared storage and potentially reducing operand stages. |
| `in_kernel_reduce` | `True` below capacity 256; `False` otherwise | Boolean. Reduce top-k in the persistent kernel tail instead of launching the standalone reducer. Mega only; reduction order is unchanged. |
| `cta_local_counts` | `True` | Boolean. Each CTA acquires startup signals and computes owner counts into reusable shared scratch, removing the closing startup grid rendezvous. CTA 0 retains a global diagnostic copy. |

Dispatch and startup options affect Mega execution, not lean FC12.
Use `validate_tactic` for host validation and `resolve_tactic` to inspect the
complete tactic. Passing host validation alone does not guarantee that a
configuration fits shared memory or is accepted by the device assembler.

### Diagnostic environment switches

Use tactic dictionary fields for scheduling A/B runs.
In the next table, append each diagnostic suffix to `FLASHINFER_MXFP8_MXFP4_`.

| Suffix | Default | Purpose |
|---|---|---|
| `WAIT_STATS` | `0` | `1` allocates and instruments per-CTA wait statistics and the startup ring. Set before workspace allocation. Read via `read_mega_wait_stats`, `read_mega_startup_ring`, `summarize`, and `decode_startup_ring` in `wait_stats_config.py`. |
| `IKET_FINE_CTAS` | `0` | Number of initial CTAs with detailed IKET ranges; requires `CUTE_DSL_COMPILER_OPT=iket`. |

Related controls use their full names below:

| Variable | Default | Purpose |
|---|---|---|
| `FLASHINFER_MEGA_FUSED_STAGE` | `1` | Use fused input quantization/staging when eligible; `0` disables it. Prequantized inputs are copied unchanged. |
| `FLASHINFER_MOE_EP_KNOB_CACHE` | Unset/empty: `~/.cache/flashinfer/moe_ep_knob_cache.json` | Override the tuning-cache file. `0`, `off`, `none`, `disable`, or `disabled` disable reads and writes (case-insensitive, whitespace trimmed for these disable values). |
| `MEGA_NO_DIST` | `0` | `1` selects single-process setup; keep `0` for distributed EP. |
| `MEGA_USE_NCU` | `0` | `1` selects the profiling tail signal path. |
| `FLASHINFER_MOE_EP_SKIP_DSL_CHECK` | Unset: check enabled | `1` bypasses the shared shim's CuTe DSL version check. |
| `CUTE_DSL_COMPILER_OPT` | Unset/empty: no IKET instrumentation | Include the `iket` token to enable this kernel's IKET markers; `IKET_FINE_CTAS` controls detailed CTA coverage. |

Dynamic-N MMA, native FC1 TMEM loads, and skipping empty FC1 epilogue
groups are always enabled. FC1 uses direct stores; FC2 uses the shared
transpose with 256-byte segments. In-kernel top-k reduction uses the
non-pipelined path, with placement controlled by `in_kernel_reduce`.

Instrumentation changes timing, so disable it when selecting performance winners.

## Autotuning

The native backend delegates selection, GPU profiling, and persistence to
[FlashInfer's general autotuner](../../../../autotuner/autotuner.py).
Use `with flashinfer.autotune():` with `knobs=None`, or retain the existing
CLI, `knobs="auto"`, and explicit tuning function. Explicit knob dictionaries,
including `{}`, remain pinned.

Tactic preparation switches the workspace layout and compiles/binds kernels
outside measurement and capture, even with zero warmup iterations. All EP ranks
agree on candidates, blocklists, cache state, and profiling settings before
profiling. A preparation failure is shared before peers launch the collective
kernel. The score is GPU latency reduced by the maximum across ranks.

Profiling uses the real staged inputs and preserves symmetric storage, including
under cold-L2 overrides. Unpinned backend workspaces allocate capacity for the
admitted candidate set. Selection completes eagerly per workspace; serving and
CUDA graphs reuse the selected tactic. Warm up before capture, and do not replay
a graph after changing the workspace's tactic.

### Offline CLI

Run outside the serving process, with the deployment's GPU type, EP world
size, capacities, and representative routing. Supply your own dimensions;
the following commands intentionally leave them as shell variables.

```bash
export FLASHINFER_MOE_EP_KNOB_CACHE=/tmp/mxfp8_mxfp4_knobs.json
torchrun --nproc_per_node="$EP_SIZE" -m flashinfer.moe_ep.tune \
    --dtype mxfp8_mxfp4 \
    --hidden "$HIDDEN" --intermediate "$INTERMEDIATE" \
    --num-experts "$EXPERTS" --topk "$TOP_K" \
    --max-tokens "$CAPACITY" --live-tokens "$LIVE_TOKENS" \
    --warmup-iters 3 --timed-iters 10
```

For one GPU, replace the `torchrun` prefix with
`MEGA_NO_DIST=1 python -m flashinfer.moe_ep.tune` and keep the arguments.
`--max-tokens` accepts several capacities, with a separate sweep for each.
Omit `--live-tokens` to use each capacity as its live count.

Every CLI option is listed below. Required geometry arguments have no built-in
values. Defaults belong to the shared CLI; in particular, always supply
`--dtype mxfp8_mxfp4` to select this backend.

| Option | Default | Behavior for this backend |
|---|---|---|
| `--dtype` | `nvfp4` | Set to `mxfp8_mxfp4`. |
| `--arch` | `auto` | Auto selects `sm107` on compute capability `(10, 7)`, `sm100` otherwise. This backend requires `sm100`; an unsupported family is rejected. |
| `--hidden` | Required | Activation/output width. |
| `--intermediate` | Required | Post-SwiGLU width. |
| `--num-experts` | Required | Total experts across EP ranks. |
| `--topk` | Required | Routes per token. |
| `--max-tokens` | Required | One or more workspace capacities per rank. |
| `--live-tokens` | `None` | Use each capacity as its live token count; a supplied count must be positive and no larger than any requested capacity. |
| `--gate-up-clamp` | `None` | No activation clamp. |
| `--combine-dtype` | `bf16` | Only `bf16` is accepted with `mxfp8_mxfp4`. |
| `--fp8-scale-mode` | `per_tensor` | Applies only to the Hopper FP8 backends; ignored here. Activations always use E8M0/K32 scales. |
| `--allow-nondeterministic` | `False` | Adds no candidates here; reduction remains deterministic. |
| `--max-candidates` | `None` | No truncation; a positive value keeps that many candidates from the front of the list. |
| `--skew` | `None` | Near-uniform random routing. A supplied ratio requests skewed routing. |
| `--sweep` | `default` | Full candidate grid; `schedule` sweeps only scheduling fields. |
| `--base-knobs` | `None` | For `schedule`, use the default tactic for this capacity and geometry. Supplied JSON overrides it. This backend does not use a cached winner as the schedule base. Ignored by the default sweep. |
| `--warmup-iters` | 3 | Warmup launches per candidate; compilation/binding happens before warmup. |
| `--timed-iters` | 10 | Timed launches per candidate. |
| `--seed` | 0 | Synthetic inputs/routing use `seed + rank`. |
| `-h`, `--help` | Not requested | Print CLI help and exit. |

The CLI uses `fast_math=True`; there is no CLI flag for it. Use the explicit
Python tuner to compare a different numerical setting.

The default pair-tile search has 128 candidates. Its initial 32 comprise
24 static candidates over N64/N128/N192, group hints 512/4096, four/eight epilogue warps, and both
reduction placements; four alternating N192 candidates; and four atomic N64
candidates. Each is also tried with a 64-cluster cap and both weight L2
eviction policies. The single-CTA search
has 32 candidates over N64/N128,
automatic or pinned 3–5 stages, group hints 512/4096, and both reduction
placements. CTA-local counts are not a separate search axis; pass explicit
candidates to compare them.

`--max-candidates` truncates the list for smoke testing. `--skew` changes
synthetic routing to a target maximum-to-mean expert load ratio. The
`--sweep schedule` option pins the default geometry (or `--base-knobs` JSON)
and sweeps static/atomic claims with group hints `None`, 128, 256, and 512;
it does not include alternating claims. `--base-knobs` applies only to that
schedule sweep. `--seed` controls synthetic inputs and routing;
`--gate-up-clamp` sets the activation clamp. This backend uses BF16 combine
and deterministic reduction, so `--allow-nondeterministic` adds no candidates.

### Explicit candidate sweep

The helpers in [`integration.py`](integration.py) use the following defaults.
Geometry, rank/world size, weights, workspace, and live-token arguments are
required wherever they appear in a signature; they have no implicit values.

| API | Optional argument | Default and behavior |
|---|---|---|
| `get_symm_buffer_for_mxfp8_mxfp4_mega_moe` | `knobs` | `None`: persistent cache lookup, then the default tactic. This low-level allocator accepts a mapping or `None`, not `"auto"`. |
| Same allocator | `capacity_tactics` | `None`: size only for the resolved tactic. Supply a sequence to reserve enough space for every listed candidate as well. |
| Same allocator | `gate_up_clamp` | `None`: accepted for API compatibility but unused during allocation. Set the clamp on compute/tuner calls. |
| `mxfp8_mxfp4_mega_moe` | `gate_up_clamp` | `None`: no clamp. |
| Same compute function | `fast_math` | `True`: approximate SwiGLU math. |
| `autotune_mxfp8_mxfp4_mega_moe` | `gate_up_clamp` | `None`: no clamp. |
| Same tuner | `fast_math` | `True`: approximate SwiGLU math, held fixed across candidates. |
| Same tuner | `candidates` | `None`: use the 128-candidate pair grid or 32-candidate single-CTA grid, then filter for workspace fit. |
| Same tuner | `warmup_iters` | 3 warmup launches per candidate; compilation/binding happens before warmup. |
| Same tuner | `timed_iters` | 10 timed launches per candidate. |
| `create_dummy_mxfp8_mxfp4_inputs` | `seed` | 0: seed the local GPU generator with `seed + rank`. |
| Same input helper | `capacity_tactics` | `None`: reserve for the full candidate grid. An explicit empty sequence reserves only for the default tactic. |
| Same input helper | `gate_up_clamp` | `None`: accepted but unused during input generation. Set it on compute/tuner calls. |

Both compute and tuner require the positional `output` argument. Passing
`None` to compute uses the weighted workspace output view; the tuner
ignores `output` and always times the workspace's stable output view. `num_tokens` is required on both
calls. The input helper starts from `knobs={}` (bypassing the cache), creates
near-uniform random expert assignments, and sets each routing weight to
`1 / topk`. Backend `knobs="auto"` reserves for the full candidate grid and
uses the tuner's default iteration counts with the config's numerical settings.

This single-GPU example uses deliberately small synthetic fixtures. Run it
with `MEGA_NO_DIST=1`. The helper stages random inputs, preprocesses weights,
and reserves enough capacity for every supplied candidate.

```python
import torch
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
    autotune_mxfp8_mxfp4_mega_moe,
    create_dummy_mxfp8_mxfp4_inputs,
    mxfp8_mxfp4_mega_moe,
)

candidates = [
    {"persistent_clusters": None},
    {"persistent_clusters": 64},
]
y, fc1, fc2, workspace = create_dummy_mxfp8_mxfp4_inputs(
    rank=0, world_size=1, num_experts=4, max_tokens=32,
    live_tokens=16, topk=2, hidden=256, intermediate=256,
    capacity_tactics=candidates,
)
try:
    winner = autotune_mxfp8_mxfp4_mega_moe(
        y, fc1, fc2, workspace, num_tokens=16, candidates=candidates,
    )
    result = mxfp8_mxfp4_mega_moe(None, fc1, fc2, workspace, num_tokens=16)
    torch.cuda.synchronize()
    print(winner)
finally:
    torch.cuda.synchronize()
    workspace.destroy()
```

For live inputs, allocate with
`get_symm_buffer_for_mxfp8_mxfp4_mega_moe(..., capacity_tactics=candidates)`,
stage the inputs, and call the same tuner. All EP ranks must participate
with identical candidate order. Candidates that do not fit the allocation
are dropped with a warning. The sweep overwrites workspace outputs and is
not allowed during graph capture.

### Reusing results and measuring performance

New tuning results use the general autotuner cache, including
`autotune(cache=...)` and the managed v2 store. Identity includes geometry,
EP size, allocated capacity, the collectively gathered live-token counts,
clamp, fast-math setting, and tactic schema. The selected tactic
is reapplied after profiling and on cache hits. Exhausting all candidates raises
an error instead of reporting an unmeasured fallback as a winner.

Rank 0 also exports winners to the legacy MoE EP knob cache for existing offline
workflows, without recording the old wall-clock `p50_us`. Explicit knobs take
precedence, followed by a matching general result, then the legacy result or
default. `FLASHINFER_MOE_EP_KNOB_CACHE=off` disables the legacy bridge without
disabling the general cache.

For the legacy compatibility workflow, start the application with the same
`FLASHINFER_MOE_EP_KNOB_CACHE` path and
`knobs=None`. Cache keys include GPU name, dtype, world size, hidden and
intermediate dimensions, expert count, top-k, combine format, and capacity.
Capacity lookup prefers exact matches, then the smallest larger bucket,
then the largest smaller bucket. Keep separate cache files for distinct
live-token/routing workloads sharing a capacity; the legacy cache does not
key live token counts or routing distribution. The general cache keys live
counts, but does not key routing values. Retune when code, compiler settings,
or hardware changes. The default path is
`~/.cache/flashinfer/moe_ep_knob_cache.json`.

For measurements, finish compilation and tuning first, preallocate outputs,
and keep the measured scope consistent: compute-only calls exclude input
quantization/staging, while a full layer call includes them. Validate against
the reference before timing; include sparse/empty experts and skewed routing.
For graph timing, warm up the exact buffers before capture and replay on all
ranks in the same order. Report the slowest rank's latency for collective
execution. Debug wait counters and IKET traces are diagnostic runs, separate
from uninstrumented performance measurements.
