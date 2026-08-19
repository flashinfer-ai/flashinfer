# next_cutedsl_megamoe — vendored kernel drop

Rubin (SM107) MegaMoE kernels from the kernel team's `cutedsl_megamoe` repo,
**`next/` greenfield tree** (a different codebase generation from the
`kernel_src/sm100/cutedsl_megamoe` SM100 drop and the `kernel_src/sm90` fork — the
`next/` tree uses fully relative imports and a composable `api.py`
kernel-component model).

## Provenance

- Upstream repo: the NVIDIA kernel team's `cutedsl_megamoe` repository
  (internal; see `ACKNOWLEDGEMENT.md` for authors/contacts).
- Upstream commit: `92dd334` (2026-08-15; brings in `a5b4d33` "Rubin MegaMoE
  Perf Improvment" — the mixed-CGA preferred/fallback cluster launch,
  reworked FC12 scheduler, and the token-in size-copy reorder around the
  metadata-ready wait). Previous snapshot: `882c83e2` (2026-08-08). The
  `rubin/inference/mega` files at `92dd334` are identical to upstream
  commit `47881ad2` (2026-08-15).
- Copied subtree: `next/sources/` → `src/sources/`
- DSL requirement: the kernels need `cutlass.utils.rubin_helpers`, which is
  in NO public `nvidia-cutlass-dsl` release (<= 4.7.0; the 4.7.0 wheel
  contains no Rubin files, and 4.8 had not shipped as of 2026-08-18).
  Validated on an NVIDIA-internal CuTe DSL nightly build (2026-08-03, git
  `d88cc85`); Rubin support in the public wheels is expected in the 4.8
  line. When a Rubin-capable public release ships, re-run the
  `oracle_sm107` / `mega_sm107` test targets against it.

## Scope of this drop (inference only)

Only the **Rubin inference block-scaled swap-AB MegaMoE kernel** closure is
vendored — `sources/kernel_src/rubin/inference/mega/`
(`BlockScaledSwapAbMegaMoeKernel` + its FC12 mainloop / gated-act epilogue /
extension / dynamic mainloop / topk reduce) plus the shared runtime it imports
(`sources/{api,quant_def}.py`, `sources/helpers/`, `sources/communication/`,
`sources/kernel_src/{schedulers,function_mapping}`).

The kernel is generic over `QuantKind` (nvfp4, mxfp4, mxfp8_e4m3, mxfp8_e5m2,
mxfp4_mxfp8); the flashinfer backends currently wire up nvfp4 and mxfp8.

Deliberately NOT vendored (future migrations extend this same directory):
`rubin/training/` (the fwd_glu fprop / bwd_dglu dgrad / traditional wgrad
training kernels — an earlier revision of this drop vendored the fwd_glu
subtree; it was removed when the flashinfer backends moved to the inference
kernel), `rubin/inference/local_mega/` (the single-GPU fused-routing
`BlockScaledSwapAbLocalMegaMoeKernel` added upstream in `4e0498e` — no EP
token comm, out of scope for the moe_ep backends), and the whole
`kernel_src/blackwell/` tree.

Note: upstream `92dd334` moved `software_sync.py` from
`sources/communication/nvlink_domain/` to `sources/helpers/`; the vendored
tree mirrors the move.

## Pending local diffs vs upstream

Per `kernel_src/README.md`, `src/` is verbatim except for the following
recorded diffs (both a consequence of the inference-only scope — upstream
ships each file as a `<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` marker shim
importing from the un-vendored `blackwell/` tree, and upstream's
`kernel_export` script inlines the blackwell source at export time; the same
whole-file inline was performed at vendor time. If the blackwell tree is ever
vendored, these can revert to the marker-shim form):

1. `src/sources/kernel_src/rubin/inference/mega/topk_reduce.py` — inlines
   `blackwell/inference/mega/topk_reduce.py`.
2. `src/sources/kernel_src/rubin/custom_mix_cga_helpers.py` — inlines
   `blackwell/custom_mix_cga_helpers.py` (mixed-CGA TMA helpers + the
   TMA-to-UMMA mixed-cluster pipeline; no relative imports, so the copy is
   byte-identical to the blackwell source).

## How this drop was ported / how to re-vendor a future upstream update

The `92dd334` refresh followed this procedure; reuse it for the next sync.
Ground rule: **never hand-edit anything under `src/`** — files are either
verbatim upstream copies or recorded whole-file inlines (see the local-diffs
section above). All adaptation lives in `shim/` and the backends.

1. **Fetch + pick the upstream commit.** In the mirror repo, `git fetch`,
   then choose the sync point (normally `origin/main`; if a reference
   result quotes a dev-branch commit, check whether the `rubin/inference` files at
   that commit are identical to main — `git diff <main> <dev> -- '**/rubin/inference/**'`
   — before preferring one over the other).
2. **Diff the vendored closure file-by-file.** For every `.py` under
   `src/sources/`, `cmp` against `next/sources/<same path>` upstream. This
   yields three lists: CHANGED (copy over), GONE-UPSTREAM (moved/deleted —
   follow the move, e.g. `software_sync.py` → `helpers/`), and files only
   changed here (the recorded inlines — leave them unless their blackwell
   source changed; check with
   `git diff <old> <new> -- '**/blackwell/**/<file>'`).
3. **Copy changed files verbatim** (`cp`, no editing), `git rm` files that
   moved away, and add any NEW files the updated kernel imports.
4. **Handle `COPY_FROM_IMPORT` marker shims.** Files whose upstream body is
   `<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` + a relative import from the
   un-vendored `blackwell/` tree get the blackwell source inlined
   WHOLE-FILE at the same path (mirroring upstream's own `kernel_export`).
   Record each one in the local-diffs section above.
5. **Run the closure scan.** AST-walk every vendored file and verify all
   relative imports resolve inside `src/sources/` (the scan used for
   `92dd334` found 40 files, zero unresolved). Also purge `__pycache__`
   remnants of removed subtrees.
6. **Sync the shim.** Diff the kernel's `ProblemDesc`/`ImplDesc` requirement
   dicts (`block_scaled_swap_ab_mega_moe_kernel.py`) and the upstream
   `tester/solvers/inference_solver.py` validity rules against
   `shim/block_scaled.py`. New `OptionalRequirement` keys are
   backward-compatible (the old shim keeps working) but should be exposed as
   config knobs with the solver's validation mirrored — e.g. `92dd334`'s
   `fallback_cluster_shape_mn` / `preferred_cluster_count` /
   `fallback_cluster_count`, whose occupancy recipe the shim replicates from
   `launch_cluster_configuration()` in the solver + `max_active_clusters()`
   in `tester/host_utils.py`.
7. **Thread new knobs to the backends** (`backends/mega/kernel/sm107/*/
   config.py` + `backend.py`: dataclass field, allocator kwarg, workspace
   pool key) and extend the tests
   (`tests/moe_ep/test_sm107_block_scaled_*.py`: config defaults, shim
   validation cases, an oracle case exercising the new path).
8. **Update this file** (provenance commit, scope notes, local diffs) and
   validate on a Rubin node: the sm107 config tests plus the
   `oracle_sm107` and `mega_sm107` targets of `tests/moe_ep/run_tests.sh`
   must be green (the `92dd334` refresh was validated 2026-08-17). Perf
   tracking lives in `TUNING.md`
   (`benchmarks/bench_moe_ep_sm107_block_scaled_mega.py`).

## Layout

- `src/` — the drop (`sources` becomes a top-level module via
  `shim/_paths.py`, same mechanism as the sibling trees)
- `shim/` — flashinfer-owned adaptation layer; the only importer of `src/`
