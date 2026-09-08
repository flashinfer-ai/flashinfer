# Updating the SM90 (Hopper) CuTeDSL MegaMoE kernel source

## Provenance

This tree vendors one atomic SM90 kernel snapshot supporting:

- Hopper FP8 fused MegaMoE;
- Humming MXFP4 x FP8 fused MegaMoE;
- Humming MXFP4 x FP8 Green Context split MegaMoE.

It is a fork of the kernel repository also vendored by
`kernel_src/cutedsl_megamoe` for SM100, but the two trees are independent.
Their top-level module names (`common`, `src`, and `moe_nvfp4_swapab`) collide,
so `shim/_paths.bootstrap_paths` rejects loading both trees in one process.
Never synchronize individual files between the SM90 and SM100 trees.

The current snapshot is a reviewed semantic merge formed in a dedicated clean
kernel staging repository:

- latest PR4688 FlashInfer base:
  `b1b6b399b7a9885cbe7d543459d0d9a6797b61b4`;
- matching staged kernel baseline:
  `9e9b873013756d8c67f79afcae5dd21b8391e149`;
- previous MXFP4 fused checkpoint:
  `1766988168658dbf73f3378467b4224f2fee9875`;
- previous unified MXFP4 fused/split checkpoint:
  `d0c99d67efb3a1600a9993377a849ff5f4ed14d8`;
- current joint source identity: the four package tree IDs and content hashes
  recorded in `VENDOR_PROVENANCE.md`.

`VENDOR_PROVENANCE.md` records the package trees, aggregate hashes, merge
contract, and recursive-copy evidence. The four packages under `src/` must
remain byte-for-byte identical to those recorded package snapshots; all
FlashInfer adaptation belongs outside them in `shim/` and the backend.

The merged source retains the latest PR4688 FP8 features as well as MXFP4:

- dispatch deduplication and grouped/quantized combine support;
- `active_dispatch_warps`;
- FC1 store offload and early-done publication;
- producer-warp folding and its effective-layout rules;
- invalid-route and stale-slot masking;
- Humming packed weights and hybrid activation scaling;
- Green Context K1/K2/K3 execution, counters, resets, and graph lifecycle.

`moe_hopper_fp8/heuristic_config.py` remains the latest PR4688 FP8 heuristic
source of truth. MXFP4 heuristic/candidate/cache policy is implemented in the
FlashInfer shim and must not modify these vendored packages.

## Layout

```text
kernel_src/sm90/pull_style_cutedsl_megakernel/
├── src/                       atomic staged kernel snapshot; do not edit here
│   ├── common/                shared constants and host helpers
│   ├── src/                   dispatch, token communication, symmetric buffers
│   ├── moe_nvfp4_swapab/      scheduler, reduction, and runner utilities
│   └── moe_hopper_fp8/        FP8 and Humming MXFP4 fused/split kernels
│       (benchmark_data/ is intentionally excluded)
├── __init__.py                public FlashInfer-facing API
├── shim/                      all local adapters, tuning, and cache integration
│   ├── _paths.py              import bootstrap and sibling-tree exclusion
│   ├── comm.py                dist/NVSHMEM bootstrap and launch state
│   ├── hopper_fp8.py          SM90 FP8 frontend
│   ├── hopper_mxfp4.py        fused Humming MXFP4 frontend
│   ├── hopper_mxfp4_split.py  Green Context split frontend
│   └── kernel_helpers.py      lazy raw-kernel helper exports
├── SKILL.md
├── VENDOR_PROVENANCE.md
└── TUNING.md
```

The fused kernel classes are `Sm90MegaMoEFp8Kernel`,
`Sm90MegaMoESwapABFp8Kernel`, and `Sm90MegaMoESwapABMxfp4Fp8Kernel` in
`src/moe_hopper_fp8/megamoe_kernel_fp8.py`. The standalone MXFP4 FC12 class is
`Sm90SwapABSwigluMxfp4Fp8Fc12Kernel`.

`src/moe_hopper_fp8/mega_runner.py` imports the non-vendored `tester` package
at module scope. Do not import it from shim code; use it only as a signature
reference.

## Updating the source snapshot

1. Start from the exact current package snapshot or a verified upstream
   baseline in a separate clean kernel repository.
2. Form and review the complete joint version there. Do not hand-merge files
   inside FlashInfer's `src/` directory.
3. Copy `common/`, `src/`, `moe_nvfp4_swapab/`, and `moe_hopper_fp8/` as one
   unit. Do not copy repository scaffolding, tests, scripts, generated Python
   bytecode, or `moe_hopper_fp8/benchmark_data`.
4. Recursively compare all four target packages to the staging tree and update
   every tree/hash in `VENDOR_PROVENANCE.md`.
5. Audit all fused and split constructor, launch, workspace, and reduction
   signatures in the shim.
6. Preserve complete identities for format, execution mode, requested and
   effective tactics, graph variant, counter banks, Green generation, hardware,
   world size, model shape, token bucket, clamp, and routing profile.
7. Run FP8 no-regression plus MXFP4 standalone, fused/split 1/2/4-rank,
   graph-replay, lifecycle, autotune, and cache tests before accepting the
   snapshot.

## Tuning and cache contract

Shared score, synchronization, and persistent-cache machinery must be reused
across FP8 and MXFP4. MXFP4-specific code owns only its packed-weight/layout
rules and fused/split execution differences.

- A complete explicit tactic bypasses cache and heuristic lookup.
- `knobs="auto"` collectively times only the compact mode-specific candidate
  union with rank-local medians followed by an all-rank maximum.
- `knobs=None` performs the mode/hardware/routing-scoped cache lookup and then
  falls back to that mode's per-token heuristic.
- Fused MXFP4, split MXFP4, and ordinary FP8 identities must never match.
- Split cache values contain the complete immutable Green session tactic:
  K1/K2 tiles, clusters, group hints, stage counts, SM partition, counter-bank
  count, graph variant, and IKET selection.
- BF16-combine MXFP4 candidates keep `grouped_token_back=false` and
  `combine_format="bf16"` until a separate numerical profile is certified.
- Effective-layout deduplication must account for producer folding, active
  dispatch warps, store offload, and early publication.

See `TUNING.md` for legal knob combinations, correctness gates, and benchmark
protocol. No external kernel-team approval is implied by this local provenance
record.
