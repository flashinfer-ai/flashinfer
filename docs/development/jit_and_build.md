# JIT and build reference

General workflow rules live in [AGENTS.md](../../AGENTS.md). Read the section
relevant to your change:

- [Compilation and workspace paths](#compilation-and-workspace-paths)
- [Module caches and source keys](#module-caching)
- [Compiler and CI caches](#compiler-caches)
- [New operations and tracing](#adding-a-new-operation)
- [Precompiled packages](#building-precompiled-packages)
- [Build implementation and dependencies](#build-implementation)

## Compilation and workspace paths

`JitSpec` in `flashinfer/jit/core.py` defines `try_load()`, `build()`, and
`load()`. Its shared `build_and_load()` handles caching, locking, and
`FLASHINFER_DISABLE_JIT`. Toolchains implement that lifecycle:

- `JitSpecNvcc`, returned by `gen_jit_spec()`, takes `name`, `sources`,
  `extra_cuda_cflags`, `extra_cflags`, and `extra_ldflags`. It generates Ninja
  rules, builds `.so` modules, and loads them through TVM-FFI.
- `JitSpecCuteDsl` in `flashinfer/jit/cute_dsl_core.py` persists CuTe-DSL kernels.
- The private TIRx spec in `flashinfer/kda_kernels/tirx/cache.py` fingerprints
  sources/compiler and checks artifacts. TVM/TIRx imports occur only for an
  explicit `recurrent_kda(..., backend="tirx")`; see [KDA](../api/kda.rst).

Installed source/AOT directories may be read-only. Write generated artifacts to
workspace paths:

| Directory | Writable | Use for |
|-----------|----------|---------|
| `FLASHINFER_GEN_SRC_DIR` | ✓ Yes | Generated source files (Jinja output, copied .cu files) |
| `FLASHINFER_JIT_DIR` | ✓ Yes | Compiled `.so` outputs |
| `FLASHINFER_CSRC_DIR` | ✗ No | Read-only source templates |
| `FLASHINFER_AOT_DIR` | ✗ No | Read-only pre-compiled binaries |

`CompilationContext` detects GPUs or reads `FLASHINFER_CUDA_ARCH_LIST`.
Generators constrain targets with `supported_major_versions`; no supported
architecture raises `RuntimeError`. FP4 CuTe-DSL worker parallelism uses
`FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS`.

A `gen_*_module()` computes a parameter-specific URI, writes under
`FLASHINFER_GEN_SRC_DIR / uri`, and returns `gen_jit_spec(...)`. Read templates
from `csrc/`; render Jinja only for needed specialization and use
`write_if_different()` to avoid unnecessary rebuilds. Follow existing
`DISPATCH_DTYPE` / `DISPATCH_BLOCK_SIZE` macros for runtime specialization.

## Module caching

`@functools.cache` retains loaded modules in Python; the file cache under
`~/.cache/flashinfer/` retains compiled artifacts. nvcc freshness relies on
Ninja dependency/timestamp checks, not source-content hashes. Replacing a source
with an older timestamp can therefore reuse a stale object. CuTe-DSL uses source
hashes as described below. Set `FLASHINFER_WORKSPACE_BASE` to change the cache
base; remove `~/.cache/flashinfer/` to clear the default cache.

### CuTe-DSL artifacts and invalidation

With caching enabled and hashable sources, `build_and_load_cute_dsl_kernel()`
wraps `JitSpecCuteDsl`, exporting kernels with
`export_to_c()` and reloading through
`cute.runtime.load_module(..., enable_tvm_ffi=True)`. Each
`cached_ops/<module>_<arch>_cute_dsl/` contains `meta.json` and one `.o` per
specialization. These are single-architecture artifacts: the target comes from
`CUTE_DSL_ARCH` or the current device, rather than an nvcc fatbin.

Changes to DSL/Torch/TVM-FFI distributions, Python ABI, or kernel-source SHA256
invalidate the module, which rebuilds lazily. See
`flashinfer/quantization/kernels/nvfp4_quantize.py` for usage. Set
`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` to disable persistence.

### Source-key coverage

By default, SHA256 covers the caller's `extra_key_files`, often just `__file__`.
`FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1` extends coverage to the in-package
source dependencies. Analysis/hashing lives in `flashinfer/jit/_cute_dsl_source.py`;
`cute_dsl_core.py` manages compilation and artifacts. Strict analysis follows:

- Plain modules and their imports, including literal `importlib.import_module`
  calls, keyed by content.
- Package names through their defining bindings: re-exports to their source,
  functions into their bodies, modules through accessed attributes, and the
  submodule of the same name.
- Initializer code executed at import, including evaluated function annotations,
  for every keyed file in that package. Initializer keys cover this code and
  import lines for used names; unused re-exports or `__all__` additions do not
  invalidate artifacts.
- Star imports when they can replace bindings, and lazy fallbacks when bindings
  may be absent.

Side effects of modules the keyed code neither imports nor uses, nonliteral
dynamic imports, and runtime file reads are outside this analysis; supply files explicitly
in `extra_key_files`. `version.py` and `_build_meta.py` are excluded so caches
can survive checkout changes.

Task-scheduled attention uses this cache through
`flashinfer/attention/prims_ts/_compile_cache.py`. Its keys cover compile specs,
options, embedded device-capacity constants, and the attention source tree,
including shared preparation, masks, and reductions. `_hardware.py` caches
occupancy queries per device and cluster size.

## Compiler caches

Devcontainers set `FLASHINFER_NVCC_LAUNCHER` / `FLASHINFER_CXX_LAUNCHER` to
sccache; `~/.cache/sccache` is a volume that survives rebuilds. Inspect it with
`sccache --show-stats`; clear a launcher variable to disable that path. Changing
launchers changes Ninja commands and can rebuild objects. Linking, cubin
embedding, and CuTe-DSL compilation bypass nvcc's sccache. Do not share one local
sccache directory between containers.

For persistent per-runner caches, S3 seeds, locking, pruning, and publication,
follow [the CI cache contract](../../ci/COMPILE_CACHE.md).
Runner cache budgets use `CI_COMPILE_CACHE_MAX_MB` (default 40960) and
`CI_COMPILE_CACHE_MIN_FREE_MB` (default 20480); see
[the implementation](../../ci/compile_cache.sh) for size/free-space handling.

### CI wheel builds and test wrappers

Release/nightly provider builds use server-side sccache and
`scripts/build_jit_cache_wheel.py`. The script uses the container's torch-pinned
interpreter: isolated-environment paths in TVM-FFI includes would change every
sccache key. It also fixes installed-header mtimes. GCC's `#pragma once` can
skip headers matching size, content, and mtime; byte-identical Torch
`ATen/ops/*_native.h` stubs otherwise make preprocessing depend on pip timing.

After 90 minutes without output, the watchdog records system, process, compiler,
and sccache state, sends TERM to the build process group, then KILL after two
minutes, and exits 124. Compiler discovery scans the container, including
sccache-owned processes outside the group. `docker run --rm` provides final
cleanup.

PR GPU lanes use `scripts/run_with_test_sccache.sh` with the release/nightly
AOT builds' S3 cache and prefix, publishing `sccache-stats-<kind>-<cuda>-<index>` artifacts. The
repository variable `FLASHINFER_GPU_TEST_SCCACHE=0` disables it for comparisons;
a failed sccache install falls back to an uncached lane. Server-owned compiles
survive a unit's process-group kill; per-unit memory telemetry excludes nvcc. Locally, run
`scripts/run_with_test_sccache.sh scripts/task_run_unit_tests.sh` (read-only
without AWS credentials).

### Provider architecture checks

PR `AOT Build Import` applies release architecture rules: a module emitting a
foreign provider's CUDA target or PTX fails validation. Before compilation,
`scripts/check_jit_spec_gencodes.py --provider <arch>` generates specs through
`flashinfer.aot.generate_jit_specs` and checks explicit `-gencode` flags with
`provider_flag_issues` in `flashinfer/jit/_cuda_architecture.py`. This needs Torch
and nvcc, not a GPU. Specs without explicit flags use the provider-filtered
compilation context.

`Provider Spec Lint (cuXXX)` runs `scripts/task_check_jit_spec_gencodes.sh` per
CUDA image before AOT builds and participates in `Test Results Summary`; AOT
lanes repeat the check. After wheel builds,
`scripts/verify_jit_cache_provider_architectures.py` performs the release
`cuobjdump` inspection. Register multi-architecture sources only for targets in
`current_compilation_context.TARGET_CUDA_ARCHS`; see `gen_ulysses_lowp_module`.

Before parallel provider lint, `scripts/prefetch_jit_spec_headers.py` fills the
shared artifact cache with the pinned TRT-LLM manifests and export headers.
A slow cold download must finish before another architecture starts waiting
for its lock; otherwise that wait can time out and the fail-fast pool kills
the downloader. Prefetch uses the generators' checksum validation and download
limits, fetches no cubins, and stops the lint if a required header is missing.

The PR AOT lane (`scripts/task_test_jit_cache_package_build_import.sh`) uses the
pinned rapidsai/sccache fork for nvcc (`FLASHINFER_SCCACHE_NVCC_IMPL=rapids`, key
prefix `<tag>-rapids`). Dependency-list hashing reduces recorded nvcc-hit host
CPU cost from 7.4 to about 0.3 CPU-s. Host C++ stays uncached because the fork
fails on `g++ -MMD` for files including `<vector>`. Use
`FLASHINFER_SCCACHE_NVCC_IMPL=upstream` for comparisons.

## Adding a new operation

For stable CUDA operations, use the
[kernel tutorial](../../.claude/skills/add-cuda-kernel/SKILL.md) and this checklist:

1. Write raw-pointer kernels in `include/flashinfer/`, a launcher in `csrc/`,
   and TVM-FFI exports in a `*_jit_binding.cu` file.
2. Add a generator in `flashinfer/jit/`; use Jinja only if specialization needs it.
3. Expose the Python API, cache its loaded modules with `@functools.cache`, and
   export it from `flashinfer/__init__.py`.
4. Add correctness tests, register in `flashinfer/aot.py`, and complete tracing
   below. Experimental features follow their separate JIT-only policy.

Examples: `flashinfer/norm/__init__.py` for a simple operation without Jinja,
`flashinfer/sampling.py` for templating, and `flashinfer/decode.py` for plan/run
and workspace management.

### Trace templates for new or updated APIs

Stable public APIs decorated with `@flashinfer_api` need a `trace=` template for
`fi_trace()` and benchmark-definition dumps:

1. Define/update `TraceTemplate` axes, inputs, outputs, and optional reference in
   `flashinfer/trace/templates/<category>.py`. Wire it with
   `@flashinfer_api(trace=my_trace)` on the function or class `run()` method.
   When a module gains its first traced stable API, add it to
   `_TRACE_REGISTRATION_MODULES` in `tests/trace/template_registry.py`; the
   inventory test fails otherwise, and the generic template tests skip it.
2. Add a realistic call to `tests/trace/example.py` and update its docstring's
   output-file list.
3. Regenerate into a scratch directory, for example
   `PYTHONPATH=$PWD FLASHINFER_TRACE_DUMP_DIR=/tmp/fi_trace_new python tests/trace/example.py`,
   then copy the new or intentionally changed JSON files into
   `tests/trace/fi_trace_out/`. Do not delete that directory first: examples the
   current GPU cannot run are skipped, and their committed files would be lost.
   Run `pytest tests/trace/ -v` and commit the JSON files alongside the code.

For an init shared by Const shapes, list keyword-only defaults in
`init._trace_init_const_axes` so each definition binds its resolved Const values.
Test exported snippets with only Var axes supplied; replay must not require
callers to override Const defaults.

## Building precompiled packages

```bash
# One binary provider for the target architecture.
export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
python -m build --no-isolation --wheel flashinfer-jit-cache-provider

# Matching shim with exact provider dependencies.
export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
python -m build --no-isolation --wheel flashinfer-jit-cache
```

The provider runs registered `gen_*_module()` functions through
`flashinfer/aot.py` for one architecture. The shim contains metadata and exact
dependencies, not binaries. Keep `FLASHINFER_LOCAL_VERSION` and
`FLASHINFER_DEV_RELEASE_SUFFIX` identical across matching provider/shim wheels.

## Build implementation

`build_backend.py` is the PEP 517 backend; `version.txt` produces
`flashinfer/_build_meta.py`. Editable installs create these data symlinks:
`3rdparty/cutlass` to `flashinfer/data/cutlass`, `csrc` to `flashinfer/data/csrc`,
and `include` to `flashinfer/data/include`.

`_flashinfer_filelock` uses Unix `fcntl.flock` or Portalocker's `Win32Locker`.
Callers own opening, timeout policy, and cleanup; nonblocking contention raises
`BlockingIOError`. Keeping the module top-level avoids GPU imports in standalone
tools. Existing `filelock.FileLock` users retain their behavior.

`build_backend.build_wheel` calls `_slim_wheel`: level-9 deflate, removal of
vendored `3rdparty` Python trees swept up by package discovery, and optional
experimental-source exclusion with `FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1`
in release builds, except the two HC JIT sources in `_EXPERIMENTAL_SOURCES_KEEP`.
It regenerates RECORD. See `tests/cpu/infra/test_build_backend_slim_wheel.py`.

## Bindings and dependencies

Export framework-independent functions with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, func)`. Torch custom operators provide
`torch.compile()` / CUDA graph integration; build rules use Ninja.

| Dependency | Reference |
| --- | --- |
| TVM-FFI | <https://tvm.apache.org/ffi/> |
| CUTLASS | Prefer local `3rdparty/cutlass/` source; <https://github.com/NVIDIA/cutlass> |
| CuTe DSL | <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html> |
| PTX ISA | <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.md> |

CuTe DSL comes from `nvidia-cutlass-dsl`, separately from the CUTLASS submodule.
Dependencies are declared in `requirements.txt` and `3rdparty/`; versioning
follows [CONTRIBUTING.md](../../CONTRIBUTING.md#release-versioning).
