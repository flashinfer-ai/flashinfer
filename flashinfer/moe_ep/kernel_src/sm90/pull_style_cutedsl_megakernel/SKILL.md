# Updating the SM90 (Hopper) CuTeDSL MegaMoE kernel src

## Provenance

This tree vendors one atomic SM90 kernel-team drop supporting:

- Hopper FP8 fused MegaMoE;
- Humming MXFP4 × FP8 fused MegaMoE;
- Humming MXFP4 × FP8 Green Context split MegaMoE.

It is a fork of the same kernel repository that
`kernel_src/cutedsl_megamoe` vendors for SM100, but the two trees are separate
backends. Top-level module names collide (`common`, `src` and
`moe_nvfp4_swapab`), so `shim/_paths.bootstrap_paths` rejects loading both in
one process. Drops are updated independently; never sync individual shared
files across the SM90 and SM100 trees.

The authoritative source chain is:

- verified target baseline:
  `9e9b873013756d8c67f79afcae5dd21b8391e149`;
- fused Phase-A v2:
  `1766988168658dbf73f3378467b4224f2fee9875`;
- unified fused/split Phase B:
  `d0c99d67efb3a1600a9993377a849ff5f4ed14d8`.

See `VENDOR_PROVENANCE.md` for package trees, aggregate hashes, donor status
identity and the exact Phase-A-to-Phase-B delta. The previous instruction to
exclude Green Context is superseded: `green_context.py`, Green graph support
and split runtime now belong to the unified drop.

Local extension pending upstream: `moe_hopper_fp8/heuristic_config.py` carries
a `token_back_mode` field per bucket. Re-apply it when syncing a drop that has
not picked it up.

## Layout

```text
kernel_src/sm90/pull_style_cutedsl_megakernel/
├── src/                    ← VERBATIM kernel-team drop; NEVER edit or add files here
│   ├── common/             ← shared constants and host utilities
│   ├── src/                ← dispatch, sym_buffer, token_comm and common runtime
│   ├── moe_nvfp4_swapab/   ← scheduler, reduction and runner utilities
│   └── moe_hopper_fp8/     ← FP8 + Humming MXFP4 fused/split kernels and Green runtime
│       (benchmark_data/ is excluded from the copy)
├── __init__.py             ← public API; talks only to shim/
├── shim/                   ← all FlashInfer adaptation lives here
│   ├── _paths.py           ← path bootstrap and sibling-tree exclusivity guard
│   ├── comm.py             ← dist/NVSHMEM bootstrap and launch state
│   ├── hopper_fp8.py       ← SM90 FP8 frontend
│   ├── hopper_mxfp4.py     ← fused Humming MXFP4 frontend
│   ├── hopper_mxfp4_split.py ← Green Context split frontend
│   └── kernel_helpers.py   ← lazy raw-kernel helper/reference exports
├── SKILL.md
├── VENDOR_PROVENANCE.md
└── TUNING.md
```

The fused kernel classes are `Sm90MegaMoEFp8Kernel`,
`Sm90MegaMoESwapABFp8Kernel` and
`Sm90MegaMoESwapABMxfp4Fp8Kernel` in
`src/moe_hopper_fp8/megamoe_kernel_fp8.py`. The standalone MXFP4 FC12 class
is `Sm90SwapABSwigluMxfp4Fp8Fc12Kernel`.

`src/moe_hopper_fp8/mega_runner.py` imports the non-vendored `tester/` package
at module scope. It is a standalone driver and must not be imported by shim
code, though its construct/launch calls remain a useful signature reference.

## Updating from a new kernel-team drop

Use the same atomic workflow as `kernel_src/cutedsl_megamoe/SKILL.md`:

```bash
rm -rf flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/{common,src,moe_nvfp4_swapab,moe_hopper_fp8}
cp -r <new_drop>/{common,src,moe_nvfp4_swapab,moe_hopper_fp8} \
  flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/
rm -rf flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/*/__pycache__ \
  flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_hopper_fp8/benchmark_data
```

Do not copy repository scaffolding (`ci/`, `tester/`, `tests/`, `scripts/`,
`.git`, `pyproject.toml`, `dispatch_test.py`, `README.md` or
`moe_mxfp8_glu/`). Do not selectively re-exclude Green/split files.

Then:

1. compare all four packages recursively and refresh the package trees and
   aggregate hashes in `VENDOR_PROVENANCE.md`;
2. audit fused and split shim compile/launch/workspace signatures;
3. preserve explicit identity for format, execution mode, tactics, graph
   variant, counter banks and Green generation;
4. rerun permanent Humming/standalone tests, fused and split
   single/2/4-rank tests, replay/lifecycle tests and full Hopper FP8
   no-regression;
5. finish with the locked 4×H200 fused/split benchmark and Nsight Systems
   proof that K1 and K2 use distinct Green Contexts and overlap.

Both MXFP4 execution modes have dedicated offline/online tuning and persistent
cache paths. A complete explicit tactic bypasses lookup, `knobs="auto"` runs
the bounded mode-specific collective sweep, and `knobs=None` looks up that
mode's cache identity before using its own frozen per-token heuristic. Split
cache winners contain the complete immutable Green session tactic: both K1/K2
tiles, clusters, group hints and stage counts, the SM partition, counter-bank
count, graph variant, and IKET selection.

Keep fused MXFP4, split MXFP4, and ordinary FP8 identities isolated. Split must
never consume a fused/FP8 cache entry or heuristic, and its session/cache
identity must retain format/layout, execution mode, model shape, EP world size,
token bucket, clamp, every tactic axis, and fixed-pointer Green lifecycle
semantics. Cache adaptation belongs in `shim/`; do not modify the verbatim
`src/` packages for tuning integration.
