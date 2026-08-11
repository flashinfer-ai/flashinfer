# Updating the SM90 (Hopper) CuTeDSL MegaMoE kernel src

## Provenance

This tree vendors the kernel team's **SM90 FP8 MegaMoE** drop — a fork of the
same kernel repo that `kernel_src/sm100/cutedsl_megamoe` vendors (Bangyu's
SM100 tree). The SM90 work (Vincent's `hopper_megamoe` branch) moved the
shared runtime forward, so this tree duplicates `common/`, `src/`, and
`moe_nvfp4_swapab/` at its own revision instead of sharing the SM100 copies.
The two trees are **separate backends**:

- top-level module names collide (`common`, `src`, `moe_nvfp4_swapab`), so
  only one tree can be active per process — `shim/_paths.bootstrap_paths`
  raises if the sibling tree's modules are already imported. A process runs on
  either Hopper or Blackwell, never both, so this is not a practical limit.
- drops are updated independently; never "sync" shared files across the trees.

Current drop: kernel repo commit `1275b8b` ("Merge branch
'vincent/hopper_megamoe' into 'main'", 2026-07).

## Layout

```
kernel_src/sm90/pull_style_cutedsl_megakernel/
├── src/                    ← VERBATIM kernel-team drop; NEVER edit or add files here
│   ├── common/             ← shared constants/host utils (SM90-drop revision)
│   ├── src/                ← CuTeDSL core src (bootstrap, dispatch, sym_buffer, token_comm, …)
│   ├── moe_nvfp4_swapab/   ← NVFP4 package (hopper_fp8 reuses its runner_common,
│   │                          fc1_fc2_fuse_sched, topk_reduce, custom_ext, moe_utils)
│   └── moe_hopper_fp8/     ← SM90 FP8 kernel implementation
│       (benchmark_data/ is excluded from the copy — data blobs only)
├── __init__.py             ← public API for moe_ep; talks ONLY to shim/ (our code)
├── shim/                   ← thin adapters over src/ (our code) — ALL adaptation lives here
│   ├── _paths.py           ← adds sibling src/ to sys.path + sibling-tree exclusivity guard
│   ├── comm.py             ← dist/NVSHMEM bootstrap, sym heap, launch-cache state
│   ├── hopper_fp8.py       ← SM90 FP8 frontend (config, symm buffer, compute entry)
│   └── kernel_helpers.py   ← lazy re-export point for raw-kernel helpers/reference
├── SKILL.md                ← this file (drop-update workflow)
└── TUNING.md               ← measured perf vs the kernel drop's reference sweep,
                              benchmark methodology, knob surface, next levers
```

The kernel classes are `Sm90MegaMoEFp8Kernel` and `Sm90MegaMoESwapABFp8Kernel`
in `src/moe_hopper_fp8/megamoe_kernel_fp8.py` (FP8 E4M3/E5M2, per-tensor or
blockwise scaling, native or swap-A/B layouts).

Note: `src/moe_hopper_fp8/mega_runner.py` imports the kernel repo's `tester/`
package (not vendored) at module scope — it is the drop's standalone test
driver and must not be imported by shim code. It IS the authoritative template
for kernel construct/launch kwargs (`run_kernel()`) when writing the shim.

## When the kernel team drops a new version of src/

Same workflow as `kernel_src/sm100/cutedsl_megamoe/SKILL.md`, with this tree's
package set:

```bash
rm -rf flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/{common,src,moe_nvfp4_swapab,moe_hopper_fp8}
cp -r <new_drop>/{common,src,moe_nvfp4_swapab,moe_hopper_fp8} \
    flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/
rm -rf flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/*/__pycache__ \
    flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_hopper_fp8/benchmark_data
```

Do NOT copy the drop's repo scaffolding (`ci/`, `tester/`, `tests/`,
`scripts/`, `.git`, `pyproject.toml`, `dispatch_test.py`, `README.md`,
`moe_mxfp8_glu/` — the SM90 backend doesn't use it).

Then audit the shim against the new drop: the highest-churn surface is the
kernel construct/launch signature (`Sm90MegaMoEFp8Kernel.__init__` /
`.__call__`) mirrored by the shim's compile/launch path, and the
`moe_hopper_fp8/mega_runner.py` `run_kernel()` driver it was modeled on.
