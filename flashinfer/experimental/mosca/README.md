# MOSCA kernels (experimental)

MOSCA is an NVIDIA kernel-optimization project. It starts from a kernel that
already serves a model, measures it on that model's cells (operation, shape,
dtype, parallel layout, GPU), and changes the kernel or its launch
configuration only where a change is proven faster with unchanged outputs.

This package ships MOSCA-optimized kernels as **experimental backends behind
stable FlashInfer APIs**. The first kernel targets DeepSeek-family MoE at
specific TP/EP configurations and is validated there (below). It is the
starting point: further MOSCA kernels (other models, other operations, other
GPUs) enter as new entries of the same layout, each with its own validated
coverage.

| Kernel (`kernel_name`) | FlashInfer API | Derived from | MOSCA changes | Validated on |
|---|---|---|---|---|
| `sm100_fp8_fp4_bf16_mosca` | `MoEEpMegaLayer` (`MegaConfig`) | DeepGEMM `sm100_fp8_fp4_mega_moe`, `sgl-project/DeepGEMM@1ff3932` | small-T dispatch, sparse expert counts, in-kernel all-gather, tile rule | DeepSeek-V4-Pro MoE layer on B200, EP 1/4/8, TP-attention + EP and DP-attention + EP layouts |

`catalog.py` holds the same facts as data. The tests, this README and the
runtime coverage check all read from it.

## How a MOSCA kernel is selected

Only explicitly. Importing the config from `flashinfer.experimental` and
constructing it is the opt-in; no environment variable is involved:

```python
from flashinfer.experimental.mosca.megamoe import Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig
from flashinfer.moe_ep import MegaConfig, MoEEpLayer

layer = MoEEpLayer(bootstrap, fleet_params, weights, backend=MegaConfig(
    megakernel=Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(
        intermediate_size=3072, top_k=6, activation_clamp=10.0)))
```

The call path:

1. **Registration.** Importing `flashinfer.experimental.mosca.megamoe` imports
   its `backend` module, which registers the name with
   `@register_mega_kernel("sm100_fp8_fp4_bf16_mosca")`. The import is cheap: no
   JIT, tvm-ffi or symmetric-memory code loads at import time.
2. **Core stays out of it.** Core never imports this package, so
   `import flashinfer` and `import flashinfer.moe_ep` load none of it. This is
   the containment rule of `flashinfer/experimental/README.md`, and
   `tests/experimental/test_containment.py` checks it.
3. **Layer.** `MoEEpLayer` (`flashinfer/moe_ep/layer.py`) returns a
   `MoEEpMegaLayer` for any `MegaConfig`.
4. **Lookup.** `MoEEpMegaLayer` calls `create_mega_kernel(config)`
   (`flashinfer/moe_ep/core/kernel/registry.py`), which looks
   `config.kernel_name` up in the registry.
5. **Warnings.** The backend emits one `ExperimentalWarning`. If the
   configuration is outside the validated coverage below, it emits one more.

Nothing picks it automatically:
- The mega path has no automatic choice between kernels.
- The cutedsl backends' `knobs="auto"` tunes knobs within one kernel, and this
  kernel has no knobs (`knobs` must be `None`).
- The kernel is not reachable from `backend="auto"`, autotuning or
  trace-apply.

A framework that already uses `sm100_fp8_fp4_bf16_deepgemm` switches by
constructing this config instead: the weights, routing and output contract are
the same. The replicated token layout (below) also changes the framework's side:
it passes the full TP batch and drops its own post-MoE all-gather.

## What is compiled, and when: no prebuilt binaries

- **Source only.** This package carries no cubin and no prebuilt library. It
  holds:
  - the vendored upstream sources (`csrc/vendor/`), byte-identical to
    upstream;
  - the MOSCA kernel and FlashInfer-side host code (`csrc/megamoe/`,
    `csrc/common/`);
  - Python modules.
- **Compiled on first use.** FlashInfer's JIT compiles tvm-ffi modules and
  caches them in its JIT cache:
  - one host planner per process;
  - one kernel module per launch configuration.

  The flags are upstream DeepGEMM's JIT flags: `sm_100f`, C++20,
  `--register-usage-level=10`, and no `-use_fast_math`. Module names hash every
  compiled source and the configuration. Nothing is registered for AOT
  (`flashinfer/aot.py`).
- **Built at warmup, not in `forward`.** By default (`precompile=True`),
  preparing the workspace does three things:
  - it plans every per-rank token count from 0 to the capacity;
  - each EP rank builds its share of the distinct kernel modules (each module
    under its own lock), and every rank confirms the builds;
  - every rank then loads them all, and confirms again.

  A failed build therefore raises on every rank instead of leaving peers inside
  the kernel's NVLink barrier. The same step checks that every rank built the
  same backend (config and geometry).
- **Capture guard.** `precompile=False` is refused with more than one EP rank,
  where one rank compiling inside `forward` would leave its peers in the
  barrier. With one rank, CUDA-graph capture of a launch configuration that was
  never built raises before any work is staged.
- **Residency check.** Loading a module checks that the device can hold the
  whole persistent grid of 2-CTA clusters at once
  (`cudaOccupancyMaxActiveClusters`). If it cannot, the load raises instead of
  the kernel hanging.
- **Distribution.**
  - Nightly wheels and source installs carry `csrc/`.
  - PyPI release wheels drop `flashinfer/experimental/<package>/csrc/**`
    (`flashinfer/experimental/README.md`), so there the first kernel load raises
    with that explanation.
  - This is why `csrc/` sits directly under `mosca/`: the release-wheel rule
    matches only that depth.
  - `pyproject.toml` lists `csrc/**`, the manifests and the license notice as
    package data.

## Where MOSCA changes the code

**Device code.** MOSCA's device-side changes are in one file this package
owns:
- `csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh` is upstream's
  `deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` at `1ff3932` with the MOSCA
  changes: +163 / -13 lines, 40 of them its license and provenance
  header.
- The 17 headers under `csrc/vendor/deep_gemm/` are byte-identical to upstream
  and carry no MOSCA change.

`vendor/deep_gemm.json` records the upstream hashes.
`python -m flashinfer.experimental.mosca._vendor deep_gemm <upstream checkout> --diff`
re-checks them against a checkout of the upstream commit and prints the
kernel's diff from upstream.

Within the kernel file:

| Feature (catalog) | Switch | Macro | Lines in the kernel file | Host side |
|---|---|---|---|---|
| `small_t` | `small_t=True` (default) | `DGX_SMALL_T` | 481-494, 502-515, 540-544, 564-581 | none |
| `sparse_count` | `sparse_count=True` | `DGX_SPARSE_COUNT` | 483-491, 516-518, 572-576 | none |
| `in_kernel_all_gather` | `token_layout="replicated"` | `DGX_BCAST` | 135-143, 895-899, 1776-1782, 1970-1984, 2001-2011 | gathered region in the buffer layout (`csrc/megamoe/sm100_fp8_fp4_planner.cu`), row offset packed into the token-count argument (`csrc/megamoe/sm100_fp8_fp4_launcher.cu`), slice staging and output copy (`megamoe/backend.py`) |
| `tile_rule` | `tile_rule="..."` | none | none | parsed, validated and applied by the planner (`csrc/megamoe/sm100_fp8_fp4_planner.cu`) |

**Changes outside the macro blocks.** Every other added line is one of:
- the license and provenance header;
- a comment;
- one of two upstream code blocks wrapped in braces so the small-T branch can
  skip them;
- the parameter rename `num_tokens` -> `num_tokens_arg`. It is unpacked under
  `DGX_BCAST`; otherwise it is copied as is.

**Behavior with every switch off.** The kernel is upstream DeepGEMM's. The
multi-GPU test checks this: with every switch off, outputs are bitwise equal to
the stable `sm100_fp8_fp4_bf16_deepgemm` backend.

## Validated coverage

| Model / shape | GPU | EP | Deployment | `token_layout` | Features | Correctness | Performance |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro MoE layer: 384 experts, hidden 7168, intermediate 3072, top-6 | B200 | 1, 4, 8 | TP attention + EP MoE | `replicated` | small_t, in-kernel all-gather | bitwise == the partitioned kernel per `tensor_split` slice + NCCL all-gather; torch oracle; graph replay == eager | kernel time vs stable DeepGEMM + NCCL all-gather, EP8 |
| same | B200 | 1, 4, 8 | DP attention + EP MoE | `partitioned` | small_t, sparse_count | bitwise == the switches-off kernel and == stable DeepGEMM for every switch (the tile rule included), equal and uneven per-rank token counts; torch oracle for every switch; graph replay == eager; planner == `deep_gemm` for every per-rank count | kernel time vs stable DeepGEMM, EP8 |

The tests are in `tests/experimental/mosca/`. The benchmark is
`benchmarks/moe_ep/bench_mosca_megamoe.py`.

Outside these cells the backend still runs, because every switch preserves
results and the switches-off kernel is upstream's. It is not validated there,
though, and it warns once per configuration.

**Tile rules are outside every cell.** The tile rule preserves results, and the
tests check `DSV4_PRO_DEP8_TILE_RULE` bitwise in all three of its windows. Its
speed, though, belongs to the deployment it was tuned in: SGLang DEP8 with the
shared expert in kernel. On this backend the shared expert is separate, and the
rule measured slower at T=64. So the backend warns for any `tile_rule`.

**Speed is reported as time saved:** (baseline - MOSCA) / baseline, where "+"
means less time.

## Package layout

```
flashinfer/experimental/mosca/
  catalog.py             every MOSCA kernel: API, config, base kernel, features, validated cells
  _vendor.py             vendored trees: manifest loading, install check, upstream re-check
  vendor/<tree>.json     one manifest per vendored tree (upstream commit, file hashes,
                         files derived from upstream, ported host files)
  csrc/vendor/<tree>/    vendored upstream sources: verbatim, no MOSCA changes
  csrc/common/           shims for kernels built on a vendored tree
  csrc/<family>/         MOSCA kernels and FlashInfer-side host code of one kernel family
  <family>/              Python side of one kernel family: config, backend, JIT, weights
  megamoe/               family 1: the moe_ep mega path
  README.md, VENDOR.md, LICENSE.DeepGEMM
```

## Adding a MOSCA kernel

1. **Upstream source.** Reuse a tree under `csrc/vendor/` or add one.
   - Vendor only the include closure the kernel needs, verbatim and without
     the files MOSCA changes.
   - Copy each file MOSCA changes into `csrc/<family>/`. Give it FlashInfer's
     license header with the line "Portions derived from <upstream>,
     <copyright>, <license>; see LICENSE.<Upstream> at the root of this
     package", then the upstream file and commit and what changed. Keep
     upstream's layout, and exclude the file from formatters.
   - Write `vendor/<tree>.json` (`files`, `derived_sources`,
     `ported_host_sources`) and add the upstream license as
     `LICENSE.<Upstream>`.
   - `.pre-commit-config.yaml` already excludes `csrc/vendor/`.
2. **Family.** Put the kernel in the family subpackage of the FlashInfer API it
   serves: `megamoe/` for `MoEEpMegaLayer`, or a new family for another API.
   - **Selection.** Its config selects it.
   - **Registration** goes through that API's experimental hook, without a
     module-level import from core:
     - a registry API (like the moe_ep mega path) registers when the family
       package is imported;
     - an API with `backend_requirement` uses a lightweight `support.py` with an
       `@experimental_backend` checker (`flashinfer/experimental/README.md`).
   - **Host code** lives under `csrc/<family>/`.
   - **Switches.** Gate every MOSCA change on a switch, so that all switches off
     is the upstream kernel.
3. **Catalog.** Add an entry to `catalog.py`: features with their macros and
   files, and the validated cells.
4. **Tests.** Add tests under `tests/experimental/mosca/`. Where a stable
   backend runs the same upstream kernel, compare bitwise against it with every
   switch off and on.
5. **Benchmark.** Add one under `benchmarks/` whose numbers back the catalog's
   performance claims.

Packaging needs no change: `pyproject.toml` already lists `csrc/**` and
`vendor/*.json` for this package.

**Why one `csrc/` for every family.** The family subpackages share one `csrc/`
directly under `mosca/` on purpose: the release-wheel rule of `build_backend.py`
drops `flashinfer/experimental/<package>/csrc/**` only. Per-family `csrc/`
directories (`mosca/<family>/csrc/`) would not match that rule, so their
sources would ship in the PyPI wheel. Removing a family therefore touches
`<family>/`, `csrc/<family>/`, its catalog entry, and any vendored tree only it
uses.

## Status

Experimental (`flashinfer/experimental/README.md`). Owner: Hoonkyung Cho
(NVIDIA). The tracking issue and graduation plan are linked from the PR.
`megamoe/README.md` describes the MegaMoE kernel's contract, and `VENDOR.md`
gives provenance and the re-sync workflow.
