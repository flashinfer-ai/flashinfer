# Batched GEMM configuration and resolution

This document describes the Prims-TS batched GEMM configuration model: how it
separates caller-supplied options from resolved scheduling decisions, and how
the original configuration schema maps to the current canonical form.

## Request versus resolved state

`BatchedGemmConfig` is frozen, and its constructor fields describe requests.
Warp indices, padding, task-only warp counts, launch width, and K-dependent
prefetch depth are derived rather than supplied as constructor options.

`cfg.warp_layout` is a cached, frozen `WarpLayout` containing task assignments
and effective register budgets. `compute_warp_layout(cfg)` returns that same
layout without modifying the request. Use `dataclasses.replace` to change a
request. A copied config resolves from its new inputs, without inheriting
previously resolved loader widths.

```python
from dataclasses import replace
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    BatchMode, RouteImpl, make_config, resolve_problem_config, validate_config,
)

request = make_config(
    batch_mode=int(BatchMode.BATCH_M),
    transpose_mma_output=0,
    route_act=int(RouteImpl.LDGSTS),
    num_load_a_warps=8,
)
cfg = resolve_problem_config(request, problem_k=2048)
validate_config(cfg)
threads = cfg.warp_layout.threads_per_cta
another_request = replace(request, num_load_a_warps=4)
```

K specialization returns another immutable config. It resolves short/odd-K MMA
unrolling and delayed SFB producer commits without changing a reusable request.
The MoE compilation hash uses effective assignments, budgets, transport, active
stage settings, and K-dependent constexpr decisions. It excludes the raw K and
private specialization state, while retaining any effective decisions K changes.

## Canonical options and migration

| Previous spelling or behavior | Current contract |
| --- | --- |
| `num_gather_warps`, `gather_warp_idx`, `gather_regs` | Use A loader settings in Batch-M and B loader settings in Batch-N, for both TMA and LDGSTS. |
| Loader warp counts overwritten by resolution | `num_load_a/b/sfa/sfb_warps=0` requests automatic selection. Explicit supported counts are honored or rejected. Ordinary non-routed TMA loaders require one warp because each warp issues a complete tile. Effective counts and indices live in `warp_layout`. |
| Routed-only per-operand register overrides | `load_a/b/sfa/sfb_regs` refer to their operand loader for every transport. Zero requests a default. Shared register defaults expand once into the resolved layout; fused tasks use one effective budget. |
| `uses_unfused_tmem_sf_copy`, `FUSED_UTCCP` | `fuse_sf_copy_to_mma` controls placement. UTCCP versus LDS+STTM describes the copy instruction independently. |
| Overlap implicitly enabled fusion | `use_max_tmem_overlap=1` requires explicit `fuse_sf_copy_to_mma=1` and one accumulator stage. Fused scale-factor columns directly follow the `2 * tile_n - epi_tile_n` accumulator window, and the combined footprint must fit the architecture's TMEM capacity (512 columns on SM100-family GPUs and 576 on SM107); config validation enforces this. Local JSON presets spell these out. |
| `cluster_m`, `has_cluster` | `umma_cta_count`, `uses_two_cta_mma`. Physical preferred/fallback CGA dimensions remain independent. |
| `route_sfs_act=NONE` sometimes gathered SF | Default `AUTO` selects LDGSTS for TMA-routed block-scaled activations, otherwise NONE. Explicit NONE always uses pre-expanded SF. `sf_route` exposes the effective transport. |
| `do_pdl_wait_for_num_non_exiting_ctas` | Derive the required pre-task wait from `use_pdl and use_early_exit`. The dependency wait remains in the kernel. `skip_early_exit_pdl=1` opts out of the kernel-entry wait; see [Early-exit PDL](#early-exit-pdl). |
| `sf_bits`, `use_two_tma_load_warps` | Removed unused options. |
| `dtype_acc`, `tile_m` | Accumulation is FP32 and CTA staging M is fixed at 128 in the implemented kernels. Neither is a constructor option. MMA instruction M remains independent. |
| `sf_block_size_a/b/c`, `sf_vec_size` | Derive supported per-operand/output block lengths from numerical formats. Unsupported block-size overrides are rejected as unknown options. |
| `num_load_sfab_warps` | Derive the single DeepSeek FP32-scale loader from the active schedule. |
| `has_scale_factors`, `has_epilogue_quant`, `does_scale_c`, dtype `*_kind`, `tmem_total_cols`, `has_deepseek_fp8` | Consolidated into input/output block-scaling predicates, canonical dtype fields, `tmem_required_cols`, and `use_deepseek_fp8`. |
| `LDG_PLUS_STS` | Removed from the supported transport enum and CLI. External generated-metadata adapters can still identify unsupported legacy values. |
| `trtllm_gemm_config_index` | Use `prims_ts_gemm_config_index` for local MoE configuration rows. |

The CLI, benchmark adapters, MoE mapper, packaged JSON, resources, and tests use
the canonical contract. Shared enums live in the host-safe `config_types.py`;
reading MoE configuration metadata does not import the CUDA/DSL kernel module.

## Early-exit PDL

With `use_pdl` and `use_early_exit`, the kernel reads `num_non_exiting_ctas` before
any task runs, so it issues `griddepcontrol.wait` at kernel entry and the task
manager is told the wait has completed (`assume_pdl_wait_completed`). No
`PdlWaitBarrier` resource is built in that case. A kernel that follows another
GEMM therefore cannot start until the predecessor has completed.

`skip_early_exit_pdl=1` drops that entry wait. The wait then comes only from the
`PdlWaitBarrier` resource on the GMEM A/B operand sourced from the predecessor,
which the task manager verifies as usual, so the kernel's CTAs can become
resident while the predecessor is still running. Validation requires
`use_pdl=1` and `use_early_exit=1`. The default is `0`.

The count is read without waiting for the predecessor, so the option is correct
only if the count was produced by a kernel that completed before the
predecessor passed its own grid wait and triggered `launch_dependents`. In the
MoE pipeline this is the intended FC2 arrangement: routing writes the count, FC1 waits on routing
at kernel entry and triggers only afterwards, so FC2 can only launch once routing
has completed. Do not set it on FC1 (whose predecessor is routing itself and
needs the entry wait), or on a kernel whose predecessor does not wait before
triggering. `launch_dependents` is emitted at the end of the load task, so the
overlap is with the predecessor's MMA and epilogue, not its loads.

The MoE mapper enables `skip_early_exit_pdl` for FC2 when the launch uses PDL
and the FC2 config uses early exit. FC1 always keeps the entry wait. A MoE JSON
row may set `skip_early_exit_pdl` to `false` to opt out for that FC2 tactic;
setting it to `true` is rejected because the mapper owns the safe default.

## Barrier and pipeline invariants

The CLC census is generated from `clc_task_participation`, which lists actual
tasks, warp widths, and CTA policies. A fused operand/SF task contributes once.
Operand task construction and census use the same `operand_cga_policy` decision.
Host and device counts evaluate the same task list against the physical CGA.
The static verifier retains an explicit projection onto its atom-only model;
that projection must not replace the runtime physical count.

Combined SFA/SFB copy has one pipeline cursor. Validation requires equal TMEM
SFA/SFB stage depths on that path, and the pipeline uses that common depth.
Separate SF pipelines retain independent depths. Fusing the producer tasks does
not merge their independent SMEM operand/SF queues.

Combined-SF barriers are atom-relative. LDS+STTM SF copies receive arrivals
from all participating CTAs, and routed loads synchronize through the proxy
barrier. Resource release and pipeline tails must preserve their established
ordering.
