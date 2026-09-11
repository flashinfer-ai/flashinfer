# FlashInfer JIT Cache and AOT Wheel Use Cases

*Working note - v0.1 - June 2026*

## 1. Purpose

The choice of JIT vs AOT deployment of GPU kernels for inference requires reasoning about tradeoffs across binary size, warmup overhead, and runtime performance. FlashInfer leaves these tradeoffs to the user, providing three related but distinct runtime artifact paths:

- `flashinfer-python`: the core package. It generates CUDA/C++ sources, compiles JIT shared libraries on demand, and can download cubin artifacts on demand.
- `flashinfer-jit-cache`: an optional package containing pre-built JIT shared libraries (`.so` files) for a specific FlashInfer version, CUDA build, and architecture set.
- `flashinfer-cubin`: an optional package containing pre-downloaded cubin/header artifacts used by some TensorRT-LLM, DeepGEMM, cuDNN, and CuTe DSL backed paths.

However, today there are several problems:

* It's not obvious what the JIT overhead implications are for some given subset of packages
* The binary size of the `flashinfer-jit-cache` package is growing rapidly, and it's not obvious what to do about that
* Users generally complain about a bucket of UX issues related to the above, and without an overall strategy, we're solving each one locally rather than moving in some coherent longer term direction

This document is a proposal to address these problems. It summarizes the current design, use cases, requirements, KPIs, and concrete measurement methods. The intent is to iterate on this with the community to set a long term vision for improving the JIT/AOT experience for FlashInfer.

## 2. Current Design Summary

### 2.1 Runtime JIT path

The core JIT abstraction is `JitSpec` in [`flashinfer/jit/core.py`](../../flashinfer/jit/core.py). Runtime workspace paths and installed AOT/cubin package discovery live in [`flashinfer/jit/env.py`](../../flashinfer/jit/env.py), Ninja generation and compiler flags live in [`flashinfer/jit/cpp_ext.py`](../../flashinfer/jit/cpp_ext.py), and target architecture selection lives in [`flashinfer/compilation_context.py`](../../flashinfer/compilation_context.py).

A generated module has:

- a stable `name`, also used as the on-disk cache key;
- source files, usually generated or copied under `FLASHINFER_GEN_SRC_DIR`;
- host, CUDA, linker, and include flags;
- an optional `needs_device_linking` mode.

At runtime, `JitSpec.build_and_load()` first checks for an AOT library. If an AOT `.so` exists at `FLASHINFER_AOT_DIR/<name>/<name>.so`, it loads that file through TVM FFI. Otherwise it writes a Ninja file under `FLASHINFER_JIT_DIR/<name>/build.ninja`, runs Ninja, and loads `FLASHINFER_JIT_DIR/<name>/<name>.so`.

Ninja is run on each JIT load path. Incrementality comes from Ninja depfiles and generated source timestamps, not from embedding source hashes in the URI. The URI primarily encodes the operation and specialization parameters such as dtype, head dimension, backend, index dtype, sliding-window mode, logits-soft-cap mode, and some architecture-specific choices.

The writable runtime workspace is:

```
<FLASHINFER_WORKSPACE_BASE or $HOME>/.cache/flashinfer/<flashinfer-version>/<sorted-target-arch-list>/
  generated/
  cached_ops/
  flashinfer_jit.log
```

This versioned and architecture-scoped layout avoids reusing incompatible compiled binaries across FlashInfer versions or architecture sets.

### 2.2 AOT JIT-cache wheel

`flashinfer-jit-cache` is built by [`flashinfer-jit-cache/build_backend.py`](../../flashinfer-jit-cache/build_backend.py). The backend imports the main project and calls `flashinfer.aot.compile_and_package_modules()` in [`flashinfer/aot.py`](../../flashinfer/aot.py). The packaged runtime lookup is exposed by [`flashinfer-jit-cache/flashinfer_jit_cache/__init__.py`](../../flashinfer-jit-cache/flashinfer_jit_cache/__init__.py).

The AOT build:

- requires `FLASHINFER_CUDA_ARCH_LIST`;
- redirects `jit_env` paths to the source tree and a build directory;
- calls the same `gen_*_module()` functions used by runtime JIT;
- enumerates a default coverage matrix in `flashinfer/aot.py`;
- builds the generated `JitSpec` list via a multi-module Ninja file;
- copies `cached_ops/<name>/<name>.so` into `flashinfer_jit_cache/jit_cache/<name>/<name>.so`;
- packages those `.so` files in a platform-specific wheel.

AOT coverage is therefore an explicit registry problem: if a runtime call generates a `JitSpec.name` absent from the wheel, `flashinfer-jit-cache` cannot satisfy that call, and `JitSpec.build_and_load()` falls back to runtime JIT compilation. If `FLASHINFER_DISABLE_JIT` is set, that fallback is refused: `JitSpec.build()` raises `MissingJITCacheError` instead of compiling, so a JIT cache miss under that setting is a hard failure, not a silent compile ([`flashinfer/jit/core.py`](../../flashinfer/jit/core.py)).

The current default AOT matrix includes common attention, GEMM, MoE, communication, activation, norm, sampling, page, rope, quantization, selective state update, XQA, and cuDNN FMHA modules. `flashinfer/aot.py` also documents some intentional exclusions, including `gen_pod_module` and `gen_deepgemm_sm100_module`.

### 2.3 Cubin package

`flashinfer-cubin` is separate from the JIT-cache wheel. It is built by [`flashinfer-cubin/build_backend.py`](../../flashinfer-cubin/build_backend.py), which downloads artifacts listed by [`flashinfer/artifacts.py`](../../flashinfer/artifacts.py) into `flashinfer_cubin/cubins/`. The packaged runtime lookup is exposed by [`flashinfer-cubin/flashinfer_cubin/__init__.py`](../../flashinfer-cubin/flashinfer_cubin/__init__.py).

Runtime cubin lookup is handled by [`flashinfer/jit/cubin_loader.py`](../../flashinfer/jit/cubin_loader.py):

- If `flashinfer-cubin` is installed, `FLASHINFER_CUBIN_DIR` points into that package.
- Otherwise `FLASHINFER_CUBIN_DIR` may be provided by environment variable.
- Otherwise the fallback is `~/.cache/flashinfer/cubins`.
- Artifacts are addressed by relative path plus expected SHA-256.
- Missing or mismatched files are downloaded from `FLASHINFER_CUBINS_REPOSITORY` unless `FLASHINFER_NO_DOWNLOAD` is set.
- Downloads use file locks, temporary files, atomic replace, retries, jitter, and checksum validation.

Some compiled JIT/AOT shared libraries register a callback with `setup_cubin_loader()`. The C++ side requests a cubin by artifact name and SHA, and Python provides the bytes from the package, local cache, or remote repository.

### 2.4 Version and compatibility checks

The loader in [`flashinfer/jit/env.py`](../../flashinfer/jit/env.py) gives installed packages priority:

- `flashinfer-cubin` must match the exact FlashInfer version, unless version checks are disabled or the main package is an editable/source install with version `0.0.0+unknown`.
- `flashinfer-jit-cache` must start with the FlashInfer version because it can carry a CUDA local version suffix, such as `+cu129`.

The architecture set is controlled by `CompilationContext` in [`flashinfer/compilation_context.py`](../../flashinfer/compilation_context.py):

- `FLASHINFER_CUDA_ARCH_LIST` wins when set.
- Otherwise local CUDA devices are detected through PyTorch.
- The arch list contributes to the runtime JIT workspace path.
- Some generators further restrict supported major versions and fail early when no supported target architecture remains.

## 3. Use Case Classes

### 3.1 General Python package users

Users install FlashInfer and call APIs directly in scripts, notebooks, or model-serving experiments.

What matters most:

- Simple install path.
- Correct fallback behavior.
- Clear first-use latency expectations.
- Helpful errors when CUDA, nvcc, or GPU architecture support is missing.
- Ability to inspect configuration.

Primary package mode:

- `flashinfer-python` alone works when runtime compilation and downloads are acceptable.
- Optional `flashinfer-cubin` and `flashinfer-jit-cache` reduce first-use overhead.

### 3.2 Production serving with warmup allowed

Serving systems can run a startup warmup phase before admitting traffic.

What matters most:

- Predictable warmup duration.
- No compile or download after warmup.
- Stable cache location and persistence across process restarts or container restarts.
- Controlled parallel compilation to avoid CPU or memory spikes.

Primary package mode:

- Either `flashinfer-python` with a persisted runtime cache, or optional prebuilt packages.
- `FLASHINFER_WORKSPACE_BASE` can point to a persistent local or shared cache.

### 3.3 Production serving with strict cold-start or offline constraints

Serving systems cannot compile at runtime, cannot download artifacts, or have tight first-token/cold-start SLAs.

What matters most:

- Zero runtime JIT.
- Zero runtime artifact download.
- Version-compatible optional packages.
- Complete coverage for the deployed model/workload mix.
- Fail-fast behavior during image validation rather than live traffic.

Primary package mode:

- `flashinfer-python` plus `flashinfer-jit-cache` and usually `flashinfer-cubin`.
- Validation should run with `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1`.

### 3.4 Framework integrators such as vLLM and SGLang

Frameworks exercise FlashInfer indirectly through many model, dtype, shape, backend, and architecture combinations.

What matters most:

- AOT coverage for the actual specialization names reached by representative workloads.
- Clear attribution when a miss comes from an unsupported or unregistered parameter combination.
- No hidden runtime compiler dependency in prebuilt serving images.
- Compatibility across the framework's supported CUDA and GPU architecture matrix.
- Package-size and build-time tradeoffs that fit their release process.

Primary package mode:

- Depends on deployment tier.
- For strict serving images, use the same validation as production offline mode.
- For developer images, runtime JIT may be acceptable.

### 3.5 Customers building pruned AOT packages

Some customers want to compile only the AOT kernels they need, for only the SM families they deploy, and at a time controlled by their own model/framework automation.

What matters most:

- Build a customer-owned `flashinfer-jit-cache` compatible package from a workload-specific manifest.
- Include only kernels needed for selected models, framework paths, and backend flags.
- Include only selected SM families, for example Hopper and Blackwell.
- Keep binary size bounded and insensitive to unrelated kernels added upstream.
- Reduce dependency on a hosted, broad FlashInfer AOT package.
- Allow an agent or framework workflow to discover required kernels and regenerate the package.
- Keep validation fail-fast when the pruned package misses a required kernel.

Primary package mode:

- `flashinfer-python` plus a customer-built, pruned `flashinfer-jit-cache` compatible wheel.
- `flashinfer-cubin` may still be needed for cubin-backed paths unless a matching pruned cubin/artifact story exists.

### 3.6 Release and CI maintainers

Maintainers build and publish `flashinfer-python`, `flashinfer-jit-cache`, and `flashinfer-cubin`.

What matters most:

- Reliable AOT builds across CUDA versions and CPU architectures.
- Bounded build time and memory.
- Reusable compiler cache.
- Wheel size visibility.
- Automated missing-coverage reports before publishing.
- Reproducible package index layout.

Primary package mode:

- Build all package artifacts in CI.
- Run package-level tests in an isolated install environment.

### 3.7 Kernel development

This is an internal/advanced use case: FlashInfer contributors, not end users of `flashinfer-python`, editing `include/`, `csrc/`, Jinja templates, or Python dispatch code and wanting changes picked up without reinstalling the package.

What matters most:

- Fast edit-test loop.
- Correct recompilation when sources or included headers change.
- Generated source visibility for debugging.
- Useful compiler output and line information when needed.
- No writes into installed package directories.

Primary package mode:

- `flashinfer-python` editable install.
- Usually no `flashinfer-jit-cache`.
- `flashinfer-cubin` optional only when working on cubin-backed paths.

## 4. Requirements and KPIs

### 4.1 High-level KPIs

These three are coarse enough to put on a slide or discuss with the community, and are what we should optimize over time:

| KPI | What "good" looks like | Measurement |
| --- | --- | --- |
| Package size | Wheels stay small enough for practical image builds/pulls, and grow predictably as kernels are added rather than unboundedly | Wheel size by package (`flashinfer-python`, `flashinfer-jit-cache`, `flashinfer-cubin`), CUDA index, and architecture set |
| Model/workload coverage with all packages installed | A representative model suite runs with zero runtime JIT and zero runtime download once `flashinfer-jit-cache` and `flashinfer-cubin` are both installed | Missing `JitSpec.name` count and missing cubin/header count for the representative workload, run with `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1`, target 0 |
| JIT time when packages are not installed | Cold-compile time for representative APIs is bounded and predictable when `flashinfer-jit-cache`/`flashinfer-cubin` are absent | Cold first-call latency per representative API/module, broken out into compile time vs. download time |

The rest of this section lists the supporting engineering requirements. Most of them are expected to "just work" rather than being independently remarkable on their own, but each is independently measurable and regressable, so we track it as a KPI too.

### 4.2 Detailed requirements and KPIs

| Use case | Requirement | KPI |
| --- | --- | --- |
| Kernel development | Source changes are picked up without reinstall | Time from source edit to passing targeted test; number of manual reinstall steps, target 0 |
| Kernel development | Debuggable compiler/runtime failures | Presence of generated source, `build.ninja`, JIT log, and optional `-lineinfo`/debug build artifacts |
| General users | First API call succeeds on supported systems | First-call success rate by API, CUDA version, and GPU arch |
| General users | First-use behavior is understandable | Presence of a clear signal (log line or equivalent) of which path was taken on first call: AOT `.so` hit, local JIT-cache hit, runtime compile, or cubin download — these are not parallel checks at one point in time; AOT-vs-local-JIT is resolved inside `JitSpec.build_and_load()` before any compile, while a cubin download is a separate, later event triggered by the already-loaded kernel at launch time via `setup_cubin_loader()` |
| General users | First-use overhead is acceptable | Cold first-call latency and warm repeated-call latency per representative API, against per-use-case targets to be defined with the community |
| Warmup serving | Warmup absorbs all compilation/download work | Runtime JIT count after warmup, target 0; runtime download count after warmup, target 0 |
| Offline serving | No compiler or network dependency | Test pass rate with `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1`, target 100% for supported workload |
| Offline serving | AOT package covers required modules | Missing `JitSpec.name` count, target 0 for the selected workload matrix |
| Framework integrators | Coverage matches real framework behavior | Missing module count per vLLM/SGLang model suite, grouped by API/backend/dtype/head-dim/arch |
| Framework integrators | No unacceptable cold-start regression | End-to-end engine startup and first-token latency across package combinations |
| Customer-pruned AOT | Package contains only selected workload coverage | Wheel size, `.so` count, and total binary bytes by manifest; unrelated-upstream-kernel size growth, target 0 |
| Customer-pruned AOT | Customer can rebuild at chosen time | Time and manual steps from workload manifest to installable wheel; reproducible build success rate |
| Customer-pruned AOT | SM family pruning is effective | Wheel size by `FLASHINFER_CUDA_ARCH_LIST`; unsupported-SM accidental inclusion count, target 0 |
| Customer-pruned AOT | Pruned package is complete for selected models | Disable-JIT/no-download pass rate for selected model suite, target 100%; missing `JitSpec.name` count |
| Cubin-backed paths | Artifacts are locally available and valid | Cubin/header local availability ratio; checksum mismatch count, target 0 |
| Release maintainers | AOT build finishes reliably | Build success rate, wall time, max RSS, OOM count, timeout count |
| Release maintainers | Build resources are controlled | `MAX_JOBS`, `FLASHINFER_NVCC_THREADS`, memory GB/job, and sccache hit rate |
| Release maintainers | Packages remain distributable | Wheel size by package/CUDA/arch and install time |
| Compatibility | Wrong package combinations fail early | Version mismatch detection rate in smoke tests |
| Concurrency | Shared caches are race-safe | Concurrent import/build/download stress test pass rate |

## 5. Engineering Measurement Details

### 5.1 Baseline local inspection

Use the CLI to capture installed package, CUDA, artifact, and module state:

```
python -m flashinfer show-config
python -m flashinfer module-status
python -m flashinfer module-status --detailed --filter not-compiled
python -m flashinfer list-cubins
```

The module status path registers the default AOT module set through `flashinfer.aot.register_default_modules()` when the registry is empty. This measures default registry coverage, not necessarily application workload coverage.

### 5.2 Default AOT registry coverage

Build and install a `flashinfer-jit-cache` wheel, then verify it with the same `FLASHINFER_CUDA_ARCH_LIST` used for the build:

```
export FLASHINFER_CUDA_ARCH_LIST="<same arch list used for the wheel build>"
python -m flashinfer show-config
python scripts/verify_all_modules_compiled.py
```

This matters because `verify_all_modules_compiled.py` registers the default AOT set through the current `CompilationContext`. If `FLASHINFER_CUDA_ARCH_LIST` is unset, the context falls back to visible local GPUs, and a broad multi-architecture wheel can be under-checked on a single-architecture host.

This reports:

- total registered default modules;
- compiled modules;
- not-compiled modules;
- names and sources for misses.

CI already exercises this in `scripts/task_test_jit_cache_package_build_import.sh`.

Use this KPI to answer: "Does the wheel contain everything `flashinfer/aot.py` says it should contain for this CUDA and architecture set?"

Do not use it as the only vLLM/SGLang KPI. It cannot prove coverage for framework-specific runtime parameter combinations that are not in the default AOT registry.

### 5.3 Workload coverage with runtime JIT forbidden

To measure real workload coverage:

1. Install `flashinfer-python`, `flashinfer-jit-cache`, and `flashinfer-cubin` into an isolated environment.
2. Run from outside the source checkout to avoid importing local files.
3. Set:

```
export FLASHINFER_DISABLE_JIT=1
export FLASHINFER_NO_DOWNLOAD=1
export FLASHINFER_JIT_CACHE_REPORT_FILE=/tmp/flashinfer_jit_cache_report.json
```

4. Run the representative test or workload suite.
5. Aggregate misses:

```
python scripts/print_jit_cache_summary.py /tmp/flashinfer_jit_cache_report.json
```

The pytest integration in `tests/conftest.py` records `MissingJITCacheError` as skipped tests with:

- test name;
- missing module name;
- source list;
- expected AOT path;
- device-linking flag.

For vLLM/SGLang, the same idea should be adapted to their integration tests or startup/warmup scripts. The output should be grouped by `JitSpec.name`, then mapped back to the operation and parameter combination encoded in the name.

Framework-specific caveats from OSS inspection:

- vLLM's FlashInfer availability check currently requires `flashinfer-python` and either `flashinfer-cubin` or `nvcc`. A no-nvcc test without `flashinfer-cubin` can make vLLM skip FlashInfer before it proves anything about `flashinfer-jit-cache` coverage.
- SGLang's FlashInfer availability check is lighter: it checks an enable env var, importability of `flashinfer`, and CUDA availability. Missing specialized APIs or artifacts tend to surface later when a selected backend is initialized.
- For both frameworks, use `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1` as the direct proof that the selected workload is covered by installed AOT `.so` files and local cubins.
- Run validation through framework backend selectors, not only direct FlashInfer API tests, because the frameworks may choose or reject FlashInfer based on model config, GPU architecture, CUDA graph mode, deterministic mode, and quantization mode.

### 5.4 Cold-start and warm-start latency

Measure at least these package/cache scenarios. The first four isolate which package or flag combination avoids compilation/download; the last one is narrower by design: it uses only `flashinfer-python` with a persisted `FLASHINFER_WORKSPACE_BASE` to measure whether a process/container restart reuses a prior run's compiled `.so` files instead of recompiling, which is the cache-persistence story called out in the "[Production serving with warmup allowed](#32-production-serving-with-warmup-allowed)" use case (§3.2). It is not meant to cover every package combination; if other combinations need their own warm-restart numbers, add separate rows rather than broadening this one.

| Scenario | Packages | Cache state | Environment |
| --- | --- | --- | --- |
| Core only, true cold | `flashinfer-python` | empty JIT cache and empty cubin cache | downloads/JIT allowed |
| Core plus cubin | `flashinfer-python`, `flashinfer-cubin` | empty JIT cache | downloads avoided for packaged cubins |
| Core plus JIT cache | `flashinfer-python`, `flashinfer-jit-cache` | empty cubin cache | JIT avoided for covered modules |
| Fully prebuilt/offline | all three packages | empty user cache | `FLASHINFER_DISABLE_JIT=1`, `FLASHINFER_NO_DOWNLOAD=1` |
| Warm cache, restart | `flashinfer-python` only | runtime cache already populated by a prior run at the same `FLASHINFER_WORKSPACE_BASE` | normal env, no disable flags |

For each scenario, measure:

- import time;
- model/engine initialization time;
- explicit FlashInfer warmup time;
- first token or first request latency;
- repeated steady-state latency;
- number of runtime JIT builds;
- number of cubin downloads;
- failure mode, if any.

Use an isolated `FLASHINFER_WORKSPACE_BASE` per run so cache state is explicit:

```
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-cache-$RUN_ID
```

Enable JIT diagnostics when measuring compile behavior:

```
export FLASHINFER_JIT_VERBOSE=1
export FLASHINFER_LOGGING_LEVEL=DEBUG
```

The JIT log is under:

```
$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>/flashinfer_jit.log
```

### 5.5 Cubin availability and integrity

For online prefetch and status inspection, use:

```
python -m flashinfer download-cubin
python -m flashinfer list-cubins
```

These commands may contact `FLASHINFER_CUBINS_REPOSITORY` to fetch checksum manifests or missing files. Run them before offline validation, not as the offline validation itself.

For the offline workload proof, install `flashinfer-cubin` or pre-populate `FLASHINFER_CUBIN_DIR`, then set:

```
export FLASHINFER_NO_DOWNLOAD=1
```

Then run the representative workload paths that call `get_artifact()` or `setup_cubin_loader()`. A missing runtime artifact then fails instead of downloading. Checksum failures are already detected by `load_cubin()` and `verify_cubin()`. Do not disable checksums for validation; `FLASHINFER_CUBIN_CHECKSUM_DISABLED=1` is a debug escape hatch.

Suggested cubin KPIs:

- `downloaded / total` from `list-cubins`;
- missing artifact failures under `FLASHINFER_NO_DOWNLOAD=1`;
- checksum mismatch count;
- download retry/failure count in logs;
- total downloaded artifact size;
- package wheel size for `flashinfer-cubin`.

### 5.6 AOT build resource measurement

The AOT build is resource intensive. The current scripts expose the knobs and diagnostics:

- `FLASHINFER_CUDA_ARCH_LIST`: architecture matrix to compile.
- `MAX_JOBS`: Ninja parallel jobs.
- `FLASHINFER_NVCC_THREADS`: per-nvcc `--threads=N`.
- `FLASHINFER_NVCC_LAUNCHER` and `FLASHINFER_CXX_LAUNCHER`: compiler cache launcher, commonly `sccache`.
- `scripts/jit_cache_build_common.sh`: computes memory-aware parallelism and configures sccache.
- `scripts/aot_memory_monitor.py`: records memory over time.

CI already reports:

- build wall time;
- wheel size via `du -h flashinfer-jit-cache/dist/*`;
- memory summaries in AOT test scripts;
- sccache stats.

Suggested build KPIs:

- AOT build success rate by CUDA version and CPU architecture;
- wall time p50/p95;
- max RSS and OOM count;
- timeout count;
- sccache compile requests, hit rate, and non-cacheable count;
- wheel size by CUDA index and architecture set;
- number of generated JIT specs.

### 5.7 Runtime performance validation

The JIT-cache wheel should change initialization behavior, not kernel semantics. For covered modules, the `.so` is produced through the same generator path as runtime JIT.

Still measure:

- correctness tests for representative APIs with and without AOT;
- steady-state kernel latency after warmup;
- benchmark parity between runtime-JIT-built `.so` and AOT-loaded `.so`;
- CUDA graph and `torch.compile` compatibility for APIs used by framework integrations.

Use existing FlashInfer benchmarks and framework-level benchmarks. Attribute regressions carefully: cache package choice should not affect a hot kernel unless the AOT architecture list, compiler version, flags, or selected backend differs from the runtime JIT environment.

### 5.8 Self-built pruned AOT package measurement

The customer-pruned AOT requirement is different from asking FlashInfer to publish smaller official wheels. The customer wants to own the AOT kernel selection and rebuild timing.

Current implementation state:

- `python -m flashinfer.aot` can compile a configured subset into an AOT directory. It supports coarse knobs such as FA2/FA3 head dimensions, dtypes, sliding-window/logits-soft-cap booleans, and inclusion flags for comm, Gemma, OAI OSS, MoE, activation, miscellaneous, and XQA kernels.
- `FLASHINFER_CUDA_ARCH_LIST` already controls the target architecture set and is the main lever for Hopper/Blackwell-only builds.
- `flashinfer-jit-cache/build_backend.py` currently calls `compile_and_package_modules(..., config=None)`, which means the normal wheel build uses the default AOT config unless the build backend or `flashinfer/aot.py` is changed.
- Runtime AOT lookup prefers an installed `flashinfer_jit_cache` package. There is no documented environment variable for pointing at an arbitrary AOT directory, so the clean customer deliverable is a compatible wheel package, not just a loose directory.

Current coarse-pruned directory build shape:

```
export FLASHINFER_CUDA_ARCH_LIST="<Hopper/Blackwell arch list, e.g. 9.0a 10.0a>"
python -m flashinfer.aot \
  --out-dir /tmp/flashinfer-aot-subset \
  --build-dir /tmp/flashinfer-aot-build \
  --fa2-head-dim 128,128 \
  --fa3-head-dim 128,128 \
  --use-sliding-window false \
  --use-logits-soft-cap false \
  --add-comm false \
  --add-gemma false \
  --add-oai-oss false \
  --add-moe true \
  --add-act true \
  --add-misc true \
  --add-xqa false
```

This produces an AOT directory. Additional packaging work is needed to make it an installable, version-compatible `flashinfer-jit-cache` wheel.

A practical measurement flow:

1. Pick a representative model/framework workload and run it once with runtime JIT allowed.
2. Collect the generated `JitSpec.name` values from the runtime cache and JIT logs, or run with `FLASHINFER_DISABLE_JIT=1` against a candidate package and collect misses through `FLASHINFER_JIT_CACHE_REPORT_FILE`.
3. Convert those names back to the generating API/backend/dtype/head-dim/feature combinations.
4. Build a pruned AOT package for only those combinations and the selected `FLASHINFER_CUDA_ARCH_LIST`.
5. Install the pruned package with `flashinfer-python` and any required cubin artifacts.
6. Re-run the workload with `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1`.
7. Compare wheel size, `.so` count, image size, image pull time, engine startup time, first-token latency, and missing module count against the official broad package.

Today this likely requires customer-specific scripting or a patch to the AOT registry/build backend. A productized answer should expose a stable manifest or trace-to-manifest workflow that can be driven by an agent or framework.

## 6. OSS Integration Snapshot

### 6.1 Scope and limitations

This snapshot is based on pinned files from vLLM commit [`405c7cf28352e45f68c338e643a4146550f3194e`](https://github.com/vllm-project/vllm/commit/405c7cf28352e45f68c338e643a4146550f3194e) and SGLang commit [`9b4432fe18be01d62fdc3aefe16c155cc9d96a95`](https://github.com/sgl-project/sglang/commit/9b4432fe18be01d62fdc3aefe16c155cc9d96a95), inspected on June 16, 2026. This should still be re-run against fresh pinned commits before publishing a final external-facing document.

### 6.2 vLLM

vLLM treats FlashInfer as a broad serving backend, not only an attention implementation.

Packaging and image path:

- `requirements/cuda.txt` pins `flashinfer-python==0.6.12`, `flashinfer-cubin==0.6.12`, and `apache-tvm-ffi==0.1.9`, with a note that FlashInfer should be updated together with the Dockerfile.
- The Dockerfile installs CUDA development tools because FlashInfer, DeepGEMM, and EP kernels can require runtime compilation.
- The Dockerfile installs `flashinfer-jit-cache==${FLASHINFER_VERSION}` from the CUDA-version-specific FlashInfer wheel index.
- After later pip installs, the Dockerfile runs `flashinfer show-config && flashinfer download-cubin`; the comments say doing this earlier caused about 2.5 GB of duplicated image layers when later pip installs touched FlashInfer files.

Availability and dispatch:

- `vllm/utils/flashinfer.py` checks for `flashinfer` without importing it to avoid CUDA side effects.
- If `flashinfer-cubin` is not installed and `nvcc` is not available, vLLM reports FlashInfer unavailable. `VLLM_HAS_FLASHINFER_CUBIN` can override cubin detection.
- TRTLLM attention support is gated on SM100, no batch-invariant mode, and either installed cubins or connectivity to `FLASHINFER_CUBINS_REPOSITORY`.
- FlashInfer API use is centralized behind lazy wrappers and feature probes, which helps vLLM tolerate FlashInfer API drift but also means missing optional submodules can disable individual paths.

Runtime surface:

- The `FLASHINFER` attention backend supports fp16/bf16 model dtypes, KV cache `auto`, fp16/bf16, fp8 variants, and `nvfp4`; block sizes 16/32/64; head sizes 64/128/256/512; and compute capability 7.5 through 12.1.
- Native paged prefill/decode wrappers and TRTLLM attention are both used. CUDA graph decode can require separate FlashInfer wrappers per batch size.
- NVFP4 KV cache is SM100-only and forces the FlashInfer wrapper backend to `trtllm-gen` because fa2/fa3 do not support NVFP4.
- The TRTLLM path is preferred for decode where supported; prefill selection depends on token count, DCP, cache dtype, sinks, speculative reorder, and explicit attention config.
- Additional vLLM FlashInfer surfaces include MLA and sparse MLA decode, top-k/top-p sampling, FP4/FP8 GEMM and quantization helpers, fused MoE variants, allreduce fusion, and NVLink MoE all-to-all.

What this implies:

- vLLM's official image path appears to value predictable startup enough to install both the JIT-cache wheel and cubins, while still carrying `nvcc` for other runtime-compiled systems.
- A vLLM KPI matrix must include attention, MLA, sampling, quantization/GEMM, MoE, allreduce, and all-to-all modes. Attention-only coverage will understate the real integration surface.
- For offline validation, do not remove `flashinfer-cubin` unless the test is explicitly about vLLM's fallback behavior. Otherwise vLLM may mark FlashInfer unavailable before exercising AOT coverage.

### 6.3 SGLang

SGLang exposes FlashInfer as a selectable backend family with many fallbacks and model-specific defaults.

Packaging and image path:

- `python/pyproject.toml` pins `flashinfer_python[cu13]==0.6.12`, `flashinfer_cubin==0.6.12`, and `apache-tvm-ffi==0.1.9`. The FlashInfer dependency comment says to keep it aligned with the JIT-cache version in the Dockerfile.
- The Dockerfile has `INSTALL_FLASHINFER_JIT_CACHE=0` by default and a separate `flashinfer_cache` stage that installs `flashinfer-jit-cache==${FLASHINFER_VERSION}` from the CUDA-specific FlashInfer wheel index when enabled.
- The Dockerfile copies the staged `flashinfer_jit_cache` package into `dist-packages` if that stage produced it.
- SGLang CI treats `flashinfer-cubin` as a 150+ MB artifact and `flashinfer-jit-cache` as a 1.2+ GB artifact worth preserving if versions match.
- In virtualenv CI, SGLang stabilizes FlashInfer JIT source/include paths by symlinking FlashInfer data and TVM FFI includes through a stable host cache path. This works around cached Ninja files that contain per-job virtualenv paths.

Availability and configuration:

- `is_flashinfer_available()` checks `SGLANG_IS_FLASHINFER_AVAILABLE`, importability of `flashinfer`, and CUDA availability.
- FlashInfer env configuration includes workspace size, paged attention mode, NVFP4 flags, deterministic split tile sizes, and autotune cache control.
- Server arguments expose FlashInfer for attention, prefill attention, decode attention, sampling, DSA top-k, FP8 GEMM, FP4 GEMM, MoE runner backends, allreduce fusion, MLA ragged behavior, Mamba, and linear attention.
- Several model and hardware cases automatically select FlashInfer backends, especially on SM100/SM120 and for selected MoE or multimodal models.

Runtime surface:

- The main FlashInfer attention backend imports `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `fast_decode_plan`, and cascade `merge_state`.
- It uses a reusable global workspace, defaulting to 384 MB and increasing for some models or deterministic mode.
- Deterministic inference changes FlashInfer decode tensor-core use, split tile sizes, CUDA graph KV splitting, and workspace size.
- The TRTLLM MHA backend is implemented on top of the FlashInfer attention backend and uses FlashInfer TRTLLM MHA kernels with a separate zero-initialized workspace defaulting to 512 MB.
- The sampler uses FlashInfer top-k/top-p/min-p sampling when selected, but disallows seeded sampling on that backend.
- Quantization and MoE paths import FlashInfer FP8/FP4, MXFP4, TRTLLM MoE, CUTLASS MoE, and CuTe DSL helpers.
- FlashInfer allreduce fusion probes available FlashInfer comm APIs and preflights workspace allocations to avoid distributed hangs.
- SGLang's kernel wrapper uses FlashInfer norm functions for supported dtypes, but avoids them under `torch.compile(..., fullgraph=True)` because the FlashInfer JIT module loading path had untraceable filesystem calls in the referenced version.

What this implies:

- SGLang's integration needs optionality. FlashInfer may be the preferred backend for some models and architectures, but SGLang keeps explicit fallbacks and many backend switches.
- JIT-cache KPIs should cover shared-cache stability in ephemeral virtualenv CI, not just static image validation.
- Graph capture, deterministic inference, autotune cache behavior, workspace allocation, and seeded sampling should be part of compatibility measurement.

### 6.4 KPI implications

The OSS integrations suggest these additions to the framework KPI matrix:

- Split AOT `.so` coverage from cubin/header coverage. Both matter, but they fail differently and are gated differently.
- Measure official framework image behavior separately from minimal offline image behavior.
- Track "FlashInfer selected" and "FlashInfer available" as KPIs in addition to "FlashInfer call succeeded"; framework gates can skip FlashInfer before an AOT miss is observable.
- Include backend flags and model-triggered defaults in the matrix, for example vLLM TRTLLM attention, sampler, allreduce/all-to-all, NVFP4 KV cache, MLA/sparse MLA, and SGLang attention, TRTLLM MHA, sampling, FP8/FP4 GEMM, MoE runner, allreduce fusion, Mamba, and linear attention paths.
- Include graph-capture and compile modes: CUDA graph decode, deterministic inference, batch-invariant mode, and `torch.compile`/Dynamo where the frameworks support them.
- For CI, report missing `JitSpec.name` values together with framework-level context: model, backend flag, dtype, KV cache dtype, page size, head dimension, architecture, CUDA version, and whether cubin lookup was local or attempted a download.

## 7. Customer Input: Workload-Pruned AOT Packages

A customer asked how they can compile AOT kernels themselves instead of consuming the hosted broad `flashinfer-jit-cache` package.

Their stated motivations:

- The hosted AOT package has grown, increasing Docker image size and image download time.
- They want to compile only the kernels they need.
- They want to compile only for the SM families they care about, specifically Hopper and Blackwell.
- They want to connect kernel generation to an agent or framework workflow that can run when they choose and for the models they care about.
- They want higher control over package size and fewer dependencies on components they cannot control.
- They want binary sizes to remain stable as FlashInfer adds unrelated kernels upstream.

What they did not provide yet:

- The exact kernel list or model suite they intend to cover.
- A target image size or package size budget.
- Whether a pruned `flashinfer-cubin` story is also required, or whether the concern is only the AOT `.so` package.

Initial interpretation:

- SM-architecture-specific official wheels help, but do not solve the core request. The ask is for workload pruning, not only architecture pruning.
- A customer-owned AOT package should be driven by a manifest of required `JitSpec.name` values or higher-level generator parameters.
- The manifest should be derivable from framework/model warmup traces so an agent can regenerate it without hand-editing `flashinfer/aot.py`.
- The output should be a normal `flashinfer-jit-cache` compatible wheel so FlashInfer's existing AOT lookup and version checks continue to work.
- Validation should be the same as strict offline serving: `FLASHINFER_DISABLE_JIT=1` and `FLASHINFER_NO_DOWNLOAD=1`.

Likely product/documentation deliverables:

- A documented "build a custom AOT wheel" recipe.
- A manifest format or config file for AOT module selection.
- A tool to collect required module names from a workload run and summarize misses.
- A build entrypoint that packages only the manifest-selected modules into a compatible `flashinfer_jit_cache` wheel.
- A size report that breaks down `.so` count and bytes by API family and target SM list.
- Guidance on cubin-backed paths so customers understand when `flashinfer-cubin` still contributes to image size.

Priority assessment:

- The request is not a production blocker today, but the customer is motivated and framed it as internal housekeeping.
- P1 seems reasonable if the goal is a documented, supported custom AOT build path with manifest-based pruning.
- A three-week estimate is plausible for a minimal productized path if scoped to JIT-cache `.so` packaging and validation. It is likely larger if it also includes automatic framework trace generation, arbitrary manifest-to-generator reverse mapping, or pruned cubin packaging.

## 8. Questions for vLLM, SGLang, and Customers

Ask each integration team for a concrete workload matrix, not just a package preference. The questions below are grouped by which §4.1 high-level KPI they help pin down, rather than listed flat, and they exclude anything FlashInfer should already know about its own primitive coverage (for example, which FlashInfer APIs frameworks reach is not asked here; that gap, if any, is FlashInfer's to close, not the integration team's to report).

Package size:

- What package size increase is acceptable? What is the target package size, image size, or image pull-time budget?
- Is the package size concern limited to `flashinfer-jit-cache`, or does it also include `flashinfer-cubin` and TensorRT-LLM artifacts?
- Do they need a single broad wheel or multiple CUDA/architecture-specific indexes?
- Do they need customer-owned AOT package generation, or are official architecture-specific wheels sufficient? If customer-owned, should the input be a hand-written manifest, framework trace, model list, or generated `JitSpec.name` list?

Model/workload coverage with all packages installed:

- Which models, dtypes, quantization modes, attention backends, head dimensions, page sizes, sliding-window modes, logits-soft-cap modes, index dtypes, and MLA/XQA/MoE variants must be covered?
- Which GPU architectures and CUDA versions are in their release/support matrix?
- Which framework gates should be treated as release-blocking when FlashInfer is unavailable or skipped?
- How should missing AOT coverage be reported in their CI: hard failure, skipped test report, or nightly advisory?

JIT time when packages are not installed:

- Is runtime `nvcc` available in production images? Are production images expected to work without runtime `nvcc`, or is `nvcc` intentionally part of the serving image contract?
- Is network egress available during startup?
- Is there an explicit cold-start, engine-start, or first-token latency SLA?
- In the "[Production serving with warmup allowed](#32-production-serving-with-warmup-allowed)" use case (§3.2), how much overhead is acceptable for kernel compilation, and how long may a startup warmup phase run?

Useful artifacts to request:

- a minimal startup script for each supported serving mode;
- a representative model suite;
- expected `FLASHINFER_CUDA_ARCH_LIST`;
- current cold-start and warm-start latency numbers;
- a log of generated/missing `JitSpec.name` values from a disable-JIT run.
- desired AOT manifest granularity: exact `JitSpec.name`, API-family plus parameter lists, model list, or framework warmup script.

## 9. Source Pointers

Implementation:

- `flashinfer/jit/core.py`: `JitSpec`, build/load path, `FLASHINFER_DISABLE_JIT`, registry, multi-spec build.
- `flashinfer/jit/env.py`: workspace paths, AOT/cubin package discovery, version checks.
- `flashinfer/jit/cpp_ext.py`: Ninja generation, compiler flags, CUDA version discovery, parallelism knobs.
- `flashinfer/compilation_context.py`: architecture detection and `FLASHINFER_CUDA_ARCH_LIST` parsing.
- `flashinfer/aot.py`: default AOT module registry, coarse CLI pruning knobs, and build configuration.
- `flashinfer-jit-cache/build_backend.py`: AOT wheel build backend; currently builds with the default AOT config.
- `flashinfer-jit-cache/flashinfer_jit_cache/__init__.py`: packaged AOT directory lookup.
- `flashinfer-cubin/build_backend.py`: cubin wheel build backend.
- `flashinfer-cubin/flashinfer_cubin/__init__.py`: packaged cubin directory lookup.
- `flashinfer/artifacts.py`: artifact path/checksum manifests and bulk download.
- `flashinfer/jit/cubin_loader.py`: runtime artifact lookup, download, checksum, callback registration.

Measurement and CI:

- `flashinfer/__main__.py`: `show-config`, `module-status`, `download-cubin`, `list-cubins`, `clear-cache`.
- `scripts/verify_all_modules_compiled.py`: default AOT registry coverage check.
- `scripts/task_test_jit_cache_package_build_import.sh`: build/install/check `flashinfer-jit-cache`.
- `scripts/task_test_nightly_build.sh`: package-level tests with `FLASHINFER_DISABLE_JIT=1`.
- `tests/conftest.py`: missing JIT-cache report collection.
- `scripts/print_jit_cache_summary.py`: aggregate missing module report.
- `.github/workflows/nightly-release.yml`: nightly package build, disable-JIT test shards, report aggregation.
- `.github/workflows/release.yml`: release package build and upload.

External OSS files inspected:

- vLLM: [`requirements/cuda.txt`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/requirements/cuda.txt), [`docker/Dockerfile`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/docker/Dockerfile), [`vllm/utils/flashinfer.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/utils/flashinfer.py), [`vllm/v1/attention/backends/flashinfer.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/v1/attention/backends/flashinfer.py), [`vllm/v1/attention/backends/mla/flashinfer_mla.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/v1/attention/backends/mla/flashinfer_mla.py), [`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py), [`vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/v1/sample/ops/topk_topp_sampler.py), [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/distributed/device_communicators/flashinfer_all_reduce.py), and [`vllm/distributed/device_communicators/all2all.py`](https://github.com/vllm-project/vllm/blob/405c7cf28352e45f68c338e643a4146550f3194e/vllm/distributed/device_communicators/all2all.py).
- SGLang: [`python/pyproject.toml`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/pyproject.toml), [`docker/Dockerfile`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/docker/Dockerfile), [`python/sglang/srt/utils/common.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/utils/common.py), [`python/sglang/srt/environ.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/environ.py), [`python/sglang/srt/server_args.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/server_args.py), [`python/sglang/srt/layers/attention/flashinfer_backend.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/layers/attention/flashinfer_backend.py), [`python/sglang/srt/layers/attention/trtllm_mha_backend.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/layers/attention/trtllm_mha_backend.py), [`python/sglang/srt/layers/sampler.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/layers/sampler.py), [`python/sglang/srt/layers/quantization/fp8_utils.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/layers/quantization/fp8_utils.py), [`python/sglang/srt/layers/quantization/mxfp4.py`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/python/sglang/srt/layers/quantization/mxfp4.py), and [`docs/advanced_features/server_arguments.md`](https://github.com/sgl-project/sglang/blob/9b4432fe18be01d62fdc3aefe16c155cc9d96a95/docs/advanced_features/server_arguments.md).

## 10. Initial Interpretation of Design Goals

The implementation and history point to these original design goals:

- Keep runtime JIT as the default because it supports fast development and broad specialization without exploding the base wheel.
- Move heavyweight prebuilt JIT modules into `flashinfer-jit-cache` so users who need faster/offline startup can opt in without forcing every user to download a large base package.
- Move cubins into `flashinfer-cubin` because those artifacts have a separate production pipeline, are addressed by checksums, and may be updated or mirrored independently from generated JIT `.so` modules.
- Prefer installed optional packages over writable user caches, while retaining environment overrides for CI, mirrors, and shared caches.
- Make mismatched package versions fail early.
- Scope runtime JIT cache paths by FlashInfer version and target architecture set to avoid binary incompatibility and cache fragmentation.
- Use file locks, atomic replacement, deterministic arch sorting, and isolated workdirs to reduce concurrency races in shared caches and CI.
- Treat AOT coverage as measurable and extensible: missing runtime specializations should become explicit `JitSpec.name` misses that can be added to `flashinfer/aot.py` or intentionally left to runtime JIT.
