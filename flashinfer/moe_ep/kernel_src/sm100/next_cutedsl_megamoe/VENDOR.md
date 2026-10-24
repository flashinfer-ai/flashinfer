# Blackwell next/ GenPhase source fork

This package owns the SM100/SM103 GenPhase implementation independently of
`../../sm107/next_cutedsl_megamoe/`. The Rubin snapshot, frontend and exports
remain unchanged. The legacy Blackwell inference drop in
`../cutedsl_megamoe/` also keeps its existing device implementation.

## Source and reproducibility

- Upstream: the kernel team's CuTeDSL MegaMoE repository; see the shared
  [acknowledgements](../cutedsl_megamoe/ACKNOWLEDGEMENT.md).
- Base source: upstream commit `1667b47a3c911ecade464ab524baf192a0bf5962`,
  exported by `next/export_src.py` from that commit.
- Exporter SHA-256:
  `a1deca5d171a4ba92ee56b5ee82f064d5dc8f44cdf52b9ce49c88d46f10fb523`.
- Integration baseline: FlashInfer commit
  `0a0cecb50b1250f0b609a9c51f1b25816ba073b8` (PR #6447). Its Rubin export
  supplies the dependency closure and sparse router/reset fixes carried by
  this fork. This is a Blackwell port of that baseline, not a new upstream drop.
- Local raw package: `src/blackwell_genphase_sources/`, 38 Python files.
  The private namespace allows the original Rubin `sources` package and
  legacy Blackwell packages to remain untouched and independently importable.

To reconstruct the starting tree, take `src/sources/` from that integration
baseline's `sm107/next_cutedsl_megamoe/`. Rename the root to
`blackwell_genphase_sources` and `kernel_src/rubin` to `kernel_src/blackwell`.
Remove the unused generic MegaMoE kernel, generic FC12 mainloop and
`dynamic_mainloop.py`. Replace the generated root API with the single
`BlackwellInferenceGenphaseMegaMoE` entry point. Reapply the explicit Blackwell
changes below. This fork intentionally differs from the upstream export;
compare unchanged dependency files byte-for-byte and review the listed port
files separately. Do not silently regenerate it from upstream main.

## Explicit local changes

The #6447 patches to counted CTA barriers, valid-row gather accounting,
masked BF16 reduction, sparse routing, proxy fences and reset quiescence are
inherited from the integration baseline. Preserve their original ordering.
The complete baseline inventory remains in
[the Rubin vendoring record](../../sm107/next_cutedsl_megamoe/VENDOR.md).
Reading that record does not create a runtime dependency on the Rubin package.

Paths below are relative to `src/blackwell_genphase_sources/`.

| File | Blackwell port |
| --- | --- |
| `__init__.py` | Export only `BlackwellInferenceGenphaseMegaMoE` from the Blackwell subtree. |
| `kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py` | Fix architecture to SM100, select native 1x MMA and the 512-column TMEM plan; admit MXFP8 E4M3, N128/CGA4 or N64/CGA2/CGA4, grouped scheduling and separate BF16 reduction. Include the architecture in the generated kernel identity. SM103 uses the same device plan with its native compiler target. Fall back to the `cutlass.utils` SMEM query on DSL 4.6/4.7, which lack `cutlass.memory`. |
| `kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py` | Use native K32 MMA (four instructions per K128 stage), two-dimensional Blackwell helper shapes, fixed N and static two-CTA B-row ownership. Remove Rubin helper imports and dynamic-N paths. Disable dynamic Gather for the fixed-N atom: load complete B tiles with static transaction counts. N64 uses seven disjoint accumulator stages and two gather warps at CGA4 (four at CGA2); N128 uses three stages and four gather warps. |
| `kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py` | Rename the independently owned workspace labels and architecture descriptions; preserve device computation. |
| `kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_extension.py` | Identify the Blackwell-owned adapter; preserve scheduling/tensor-view behavior. |
| `kernel_src/blackwell/inference/mega/tma_gather.py` | Rename the Gather4 helper and its call site to reflect Blackwell ownership; preserve the instruction implementation. |
| `kernel_src/blackwell/inference/mega/token_comm_gen_specialized.py` | Publish each refine-stage writer's generic shared-memory stores with an async-proxy fence before the group barrier and elected bulk store. |

All remaining dependency files retain the integration baseline's bytes.
Changes under `src/` must be inventoried here; never format them for lint.

## FlashInfer frontend and scope

`shim/` owns path setup, config validation, weight conversion, workspace,
compilation, launch and the numerical oracle. It imports only this fork's raw
namespace. Public consumers use this package's `__init__.py`.
The six existing GenPhase names are also lazily re-exported by
`../cutedsl_megamoe/__init__.py` for compatibility.

Supported: native SM100/SM103, MXFP8 E4M3, SwiGLU, BF16 combine with separate
reduction, grouped N128/CGA4 or N64/CGA2/CGA4, early/late top-k weights and
capacity at most 1024 per rank. The low-level config retains N128/CGA4 defaults.
The explicit GenPhase backend resolves performance knobs from an exact workload
cache or geometry estimate and supports collective warmup tuning. Blackwell's
default backend selection does not select this path. NVFP4, dynamic N and
in-kernel reduction are outside this port.

Native correctness is covered in the existing Blackwell workspace, oracle and
multirank graph test files; no new test file is required. Host construction
checks establish layout/admission only, not native device correctness.

The SM100 MXFP8 backend exposes this fork through the explicit
`kernel_variant="genphase"` config, with separate weight packing and workspace
pooling. Its default remains legacy inference. The GenPhase public API also
exposes prequantized weight conversion and external-weight validation; these
are host adapters and do not change the raw-source inventory above.
