# GC MoE vendor snapshot

- Upstream: <https://github.com/Bruce-Lee-LY/gc-moe>
- Commit: `7902831ec4a6691b6268b308bcff5ec0cb15ec86` (2026-11-15, package
  version 0.6.2)
- Exported tree: the upstream `gc_moe/` package only — 37 Python files,
  14,877 lines before the local patches below
- Exporter:

  ```bash
  git -C <gc-moe-checkout> archive 7902831 -- gc_moe \
    | tar -x -C src --strip-components=1
  ```

- Local patches: the three files listed under *Local patches on top of the
  pin*; every other file is byte-identical to the pinned commit.
- Version record: `__init__.py` mirrors this commit as `UPSTREAM_COMMIT` and
  the package version as `UPSTREAM_VERSION`;
  `tests/experimental/gc_moe/test_gc_moe_api.py` fails if either drifts from
  `src/__init__.py` or this file.

## Local patches on top of the pin

These are the only differences from the pinned commit. They carry PR review
fixes until upstream picks them up; re-apply them after every re-export and
drop this section when a later upstream release contains them.

1. **Drain the final combine store (SM100).** `src/impls/sm100_fp8_fp4_gc_moe.py`,
   end of `GCMoERoleKernel._role3_combine`: the chunk loop waits for the
   previous chunk's `bulk_s2g_1d` at its top, so the last committed store was
   never drained. A trailing `tma_store_wait_0()` on the issuing lane keeps the
   CTA from retiring while the copy still reads `store_smem`; other lanes have
   no pending group and the wait is a no-op there.
2. **Retire cached weights with the workspace (SM100).**
   `src/gc/__init__.py` and `src/jit_kernels/impls/sm100_fp8_fp4_gc_moe.py`:
   `GCSymmBuffer.destroy()` calls a new `release_workspace()`, which pops the
   `_CACHE` entries whose local symmetric-buffer pointer matches
   (`_capture_key` index 4). Those entries hold the TensorMaps and pin the
   routed weight payload/scale tensors through `_Prepared.keepalive`, so
   without this a replaced model's device weights are never freed.
   Address-independent cubins in `_BINARY_CACHE` stay cached.

## Verification

```bash
rm -rf /tmp/gc_moe_upstream && mkdir -p /tmp/gc_moe_upstream
git -C <gc-moe-checkout> archive 7902831 -- gc_moe \
  | tar -x -C /tmp/gc_moe_upstream --strip-components=1
diff -r --exclude=__pycache__ src /tmp/gc_moe_upstream
# must name only the three files under *Local patches on top of the pin*
```

## Deliberate deviations from the `moe_ep/kernel_src` layout

1. **One drop for three architectures.** Upstream keeps SM90, SM100 and SM120
   in one commit and one package; splitting it into
   `sm90/…`, `sm100/…`, `sm120/…` drops would copy the same commit three
   times. The architecture entry point is chosen in
   `flashinfer/experimental/gc_moe/dispatch.py`.
2. **No `__init__.py` injection.** Twelve upstream directories
   (`apis`, `comm`, `common`, `impls`, `jit_kernels`, `jit_kernels/heuristics`,
   `jit_kernels/impls`, `layout`, `mma`, `ptx`, `runtime`, `scheduler`) ship no
   `__init__.py`. They import as PEP 420 namespace packages nested inside this
   regular package, so no `__init__.py` files are added to `src/`; `[tool.setuptools.packages.find]`
   (namespaces on by default) discovers each one and ships its modules, and
   `[tool.setuptools.package-data]` only adds the two provenance notes
   (`VENDOR.md`, `SKILL.md`) to the wheel. Verify with
   `python -m build --wheel` + `unzip -l` before releasing.
3. **Environment variables.** Upstream reads eight `GC_MOE_*` variables:
   `GC_MOE_PTXAS` at import time and seven switches at call time. The shim
   resolves and validates the assembler before the first import (`shim/env.py`)
   and pins `GC_MOE_PRIMARY_CTX=1` there; the pinned mode also makes the two
   `GC_MOE_*_BALANCED_WORKQUEUES` switches inert, so `plan.py` exposes only
   dispatch, the NVLink barrier timeout and `ptxas` as `GcMoePlan` fields.
