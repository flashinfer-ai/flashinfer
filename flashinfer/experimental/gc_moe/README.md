# GC MoE backend (experimental)

Green-Context megakernel MoE for SM90, SM100/SM103 and SM120, integrated from
[Bruce-Lee-LY/gc-moe](https://github.com/Bruce-Lee-LY/gc-moe). The public
entry points are `flashinfer.gc_moe`, `flashinfer.GCMoEWorkspace` and
`flashinfer.transform_gc_moe_weights` (all marked experimental); everything
here is JIT-only and never registered for ahead-of-time compilation.

## Layout

```
flashinfer/experimental/gc_moe/
  support.py     compute-capability -> backend selection, lightweight checks
  plan.py        GcMoePlan: per-backend overrides of the upstream env switches
  workspace.py   GCMoEWorkspace: owned workspace allocation + staging views
  weights.py     checkpoint-format -> packed-weight adapter
  dispatch.py    backend consistency, serialization, launch
  kernel_src/gc_moe/
    VENDOR.md    pinned upstream commit, exporter command, deviations
    SKILL.md     re-sync procedure
    __init__.py  the drop's surface (the only import consumers should use)
    shim/        environment, error translation, the single src importer
    src/         vendored upstream sources + documented local patches
```

## Rules

- `kernel_src/gc_moe/src/` vendors upstream gc-moe 0.6.2, commit
  `7902831ec4a6691b6268b308bcff5ec0cb15ec86`, plus the local patches listed in
  `VENDOR.md`; all other adaptation lives in `shim/`. Keep local patches
  minimal and re-check them when re-vendoring (see `SKILL.md`).
- `shim/launch.py` is the only module that imports `src/`: the vendored JIT
  resolves its PTX assembler at import time, so the environment must be
  prepared first.
- Consumers import the drop only through `kernel_src/gc_moe/__init__.py`.
- The 12 `src/` directories without `__init__.py` are PEP 420 namespace
  subpackages; `packages.find` ships their modules normally and package-data
  adds `VENDOR.md`/`SKILL.md` to the wheel.

## Verification

```
python -m pytest tests/experimental/gc_moe -q      # correctness + graphs
python examples/experimental/gc_moe.py             # runnable example
python benchmarks/bench_gc_moe.py                  # latency sweep
```
