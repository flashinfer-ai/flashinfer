# Vendoring record: sm120/swapab_cutedsl_megakernel

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Repo**: https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe.git
  (fork of the NVIDIA CuTeDSL MegaMoE kernel team's repo; see the mother
  tree's `ACKNOWLEDGEMENT.md` for authors/contacts)
- **Branch**: `run/sm120-mxfp8-perf`
- **Vendored commit**: `d19d30a748f9e402b8a2a33083fdf530231cf647`
  ("Adapt SM120 runner to latest MXFP8 host utils", 2026-07-27)
- **Last synced**: 2026-08-06 (copied from the local worktree
  `/home/scratch.mhoqueanik_gpu/cutedsl_megamoe_sm120/sm120_swapab_wt`)
- **Vendored subset**: the five kernel packages only (`common/`, `src/`,
  `moe_sm120_mxfp8_swapab/`, `moe_mxfp8_glu/`, `moe_nvfp4_swapab/`) under
  `src/` — no repo scaffolding (`ci/`, `tester/`, `tests/`, `scripts/`,
  `pyproject.toml`, …). `moe_mxfp8_glu/` is included because
  `moe_sm120_mxfp8_swapab/mega_reference.py` lazily imports its torch
  reference (`mega_reference_mxfp8`); `moe_nvfp4_swapab/` is included
  because that reference module in turn imports
  `moe_nvfp4_swapab.runner_common` / `.mega_reference` at module top.

## Policy

- `src/` is a **verbatim** copy of the upstream drop: no injected files, no
  local edits. `diff -r src/<pkg> <upstream>/<pkg>` must come back clean.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Known-broken upstream features at this snapshot

- **`in_kernel_fc2_reduce` (REDG in-flight combine)**: the drop's own
  `mega_runner` crashes with `cudaErrorIllegalAddress` under
  `--in_kernel_fc2_reduce` (verified 2026-08-06, RTX PRO 6000 / 4 ranks on
  1 GPU, DSL 4.6.1), and the flag appears in none of the drop's test
  scripts. The FI backend rejects the flag; the shim keeps the plumbing so a
  fixed drop only needs the backend guard and test skips removed.
- **`cluster_shape_mnk` with cluster_m > 1**: fails at `cute.compile` with
  "expects num_multicast to be 1 for non multicast G2S copies" (verified
  2026-08-06, same setup, reproduced with the drop's own `mega_runner` at
  `--cluster_shape_mnk 2,1,1`); the drop's test scripts always use `1,1,1`.
  The shim config rejects anything but `(1, 1, 1)`.
- **`gate_up_clamp`**: dead plumbing — `kernel_fc12.py` stores the ctor arg
  and never reads it; kernel output is bit-identical with and without the
  clamp while the torch reference applies it (verified 2026-08-06 by A/B on
  dense data; the drop's ±0.5-sparse test data never reaches any clamp, so
  its own runner cannot see this). The FI backend rejects a set clamp; the
  shim keeps passing it to the ctor for a fixed drop.
- **`mma_tiler` N=128 numerics (every world size)**: silently wrong outputs.
  world_size=1 (`MEGA_NO_DIST`): 5–20% of cells off (worst-hit tokens
  scattered per expert), reproduced with the drop's own `mega_runner` at
  `MEGA_NO_DIST=1 --mma_tiler_mnk 64,128,128` on both its standard geometry
  and ours. world_size=2 and 4 (verified 2026-08-07 on RTX PRO 6000 and RTX
  6000D, rank-sharing, dense data, deepseek_v3 geometry): rel-L2 vs the
  bf16 dense reference sits in the ~6.35% MXFP8 band at small tokens/rank
  and degrades once tokens fill past an N=64 tile — ws2: 10–28% across
  16..8192 tokens/rank; ws4: 8–25% across 8..4096 — with run-to-run
  magnitude variation (consistent with a race). N=64 stays in band at every
  point tested (ws1/ws2/ws4, up to 8192 tokens/rank), at ~23% lower
  large-batch throughput. NOTE: the earlier "bit-exact at world_size=4 with
  N=128" observation came from the drop's 1%-sparse ±0.5 test data, which
  cannot see this failure; dense activations expose it at every world size.
  The FI backend pins `mma_tiler_mnk=(64, 64, 128)` at all world sizes
  unless the caller passes an explicit tiler knob; avoid N=128 entirely
  until a fixed drop.

## Pending local diffs vs upstream

The snapshot was taken from a **dirty worktree**: two files carried
uncommitted changes on top of `d19d30a` and are vendored as found on disk
(they are still verbatim w.r.t. the worktree, not the commit):

- `moe_sm120_mxfp8_swapab/mega_runner.py` (+~25/−~19 lines)
- `src/bootstrap.py` (+~42/−~7 lines)

Fold these into a committed upstream state on the next re-sync.

## Related trees

- `kernel_src/cutedsl_megamoe/` is the mother-repo snapshot (SM100 NVFP4 +
  MXFP8 kernels). This tree is a **separate snapshot** of a fork carrying the
  SM120 MXFP8 swap-AB kernel; it is not merged into the mother tree on
  purpose: one kernel_src dir = one upstream commit. If upstream merges the
  SM120 kernel into the mother repo, fold this tree into that one on the next
  re-sync.
- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is the analogous separate
  snapshot for the Hopper FP8 pull kernel.

## Consumers

- `backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/`
