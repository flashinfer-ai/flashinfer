---
name: custom-jit-cache
description: Build a minimal flashinfer-jit-cache wheel holding only the kernels one deployment uses, for only the GPU architectures it runs on
---

# Building a custom, minimal jit-cache wheel

Use this when someone wants a smaller `flashinfer-jit-cache` than the official
per-architecture providers: "only the kernels my model needs", "only H100",
"shrink the jit-cache wheel". The full guide is
[`docs/custom_jit_cache.rst`](../../../docs/custom_jit_cache.rst); the tool is
[`scripts/custom_jit_cache.py`](../../../scripts/custom_jit_cache.py).

## How it works

- Every jit-cache module is an nvcc `JitSpec` from `gen_all_modules()` in
  `flashinfer/aot.py`, identified by its `name` (a URI encoding dtypes, head
  dims, flags). Provider wheels ship `<name>/<name>.so` plus a manifest listing
  the names; at runtime `flashinfer/jit/env.py` looks modules up by name.
- `FLASHINFER_JIT_USAGE_LOG=<file>` makes `JitSpec.build_and_load()` append
  one JSON line per module a process loads (AOT hit or JIT build).
- A recipe (JSON, or TOML on Python 3.11+) combines usage logs, exact names,
  `fnmatch` include/exclude globs and `aot_config` overrides of
  `get_default_config()`. `flashinfer.aot.ModuleSelection` / `select_modules()`
  apply it; the provider build reads it from
  `FLASHINFER_JIT_CACHE_PROVIDER_SELECTION`.

## Steps

1. **Capture** on the target GPU (fastest with the full official jit-cache
   installed, so nothing compiles):
   ```bash
   python scripts/custom_jit_cache.py capture --log usage.jsonl -- python serve.py ...
   ```
   Make the run cover every path the deployment takes: prefill and decode, all
   CUDA-graph batch sizes, speculative decoding, autotuning. Multi-process jobs
   can share one log via `export FLASHINFER_JIT_USAGE_LOG=...`.
2. **Recipe**: `python scripts/custom_jit_cache.py init --name NAME --from-usage usage.jsonl`.
   Edit `cuda_archs` to the deployment GPUs; add `modules` / `include` for paths
   the capture missed. Example: `scripts/custom_jit_cache_recipes/`.
3. **Check** (CUDA toolkit needed, no GPU): `python scripts/custom_jit_cache.py resolve NAME.json -v`.
   Use `list --arch 9.0a --pattern REGEX` to find names.
4. **Build** from a checkout at the deployment's FlashInfer version, installed
   with `pip install --no-build-isolation -e .`:
   ```bash
   python scripts/custom_jit_cache.py build NAME.json --local-version cu130
   pip install dist/custom-jit-cache/NAME/*.whl
   ```
5. **Verify**: run the workload with `FLASHINFER_DISABLE_JIT=1`; a
   `MissingJITCacheError` names any module to add to the recipe.

## Pitfalls

- Module names change between releases (some embed source hashes). Capture
  and build with the same FlashInfer version; `init`/`resolve` warn otherwise.
- Only nvcc modules can be shipped. CuTe-DSL modules built through
  `JitSpecCuteDsl` appear in the log as runtime-only; kernels compiled by direct
  `cute.compile()`, Triton or cuTile (e.g. SM80 `rmsnorm`) bypass `JitSpec`, so
  they are neither logged nor blocked by `FLASHINFER_DISABLE_JIT`.
- `build` passes `--skip-dependency-check`: with a CUDA local version the
  provider's build requires pin isolated-build wheels the current env may lack.
- Custom providers reuse the official distribution names
  (`flashinfer-jit-cache-sm90a`), so they replace, not coexist with, the
  official jit-cache.
- When building by hand, delete `flashinfer-jit-cache-provider/build/` first;
  setuptools' staging directory otherwise leaks modules from an earlier, larger
  build of the same arch into the wheel (the script does this for you).
- Changing the recipe format or `ModuleSelection`? Update
  `docs/custom_jit_cache.rst`, this skill and
  `tests/jit/test_custom_jit_cache.py` together.
