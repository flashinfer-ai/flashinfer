# Working on FlashInfer

FlashInfer provides GPU kernels for LLM serving. Kernels compile on demand, so
a new process picks up source changes without reinstalling. A running process
keeps the modules it already loaded, and an installed `flashinfer-jit-cache`
artifact is loaded before changed source is considered; uninstall that package
when validating kernel edits. Read any nested `AGENTS.md` before changing that
subtree.

## Common commands

Run commands from the repository root. Use the existing Python/CUDA environment.

| Task | Command |
| --- | --- |
| Initialize submodules | `git submodule update --init --recursive` |
| Install for development | `pip install --no-build-isolation -e . -v` |
| Run a test file | `pytest tests/<path>.py` (append `::test_name` for one case) |
| Run regular / full test matrices | `pytest tests/` / `pytest tests/ --full` |
| Run multi-GPU tests | `mpirun -np 4 pytest tests/comm/test_allreduce_unified_api.py` |
| Install pre-commit hooks | `pre-commit install` |
| Check changed files | `pre-commit run --files <paths>` |
| Check all files | `pre-commit run -a` |
| Benchmark | `python benchmarks/flashinfer_benchmark.py --routine <name> <flags>` |
| Collect environment details | `python -m flashinfer.collect_env` |

`--no-build-isolation` avoids installing conflicting PyTorch/CUDA dependencies.
The minimum Python version lives in `.python-version`; when raising it, update
`project.requires-python` and `tool.mypy.python_version` in `pyproject.toml` too.

## Code layout and boundaries

| Path | Purpose |
| --- | --- |
| `include/flashinfer/` | Framework-independent CUDA kernels; raw pointers, no Torch headers |
| `csrc/` | Launchers and TVM-FFI bindings; optional Jinja templates |
| `flashinfer/` | Python APIs and backend implementations |
| `flashinfer/jit/` | Module generators, compilation, loading, and cache paths |
| `flashinfer/trace/templates/` | Public API trace definitions |
| `tests/`, `benchmarks/` | Correctness tests and performance measurements |
| `flashinfer/aot.py` | Registration for precompiled kernel packages |
| `build_backend.py` | PEP 517 build backend |

Match nearby code and reuse existing helpers. Keep comments focused on reasons
and constraints. Explain intentional departures from established patterns.

Generated sources belong in `FLASHINFER_GEN_SRC_DIR`, and compiled outputs in
`FLASHINFER_JIT_DIR`. Never write generated artifacts into installed package
source or AOT directories: they may be read-only.

For new CUDA operations, follow the
[kernel tutorial](.claude/skills/add-cuda-kernel/SKILL.md) and
[operation/trace checklist](docs/development/jit_and_build.md#adding-a-new-operation).
Public APIs use `@flashinfer_api` with a `TraceTemplate`; update trace examples,
regenerate their JSON files, and run `tests/trace/`. Register supported stable
operations for AOT and export them from the Python package.

Experimental backends live under `flashinfer/experimental/`; experimental public
APIs stay in core with `@flashinfer_experimental_api`. Explicit API/backend use
warns once; automatic selection requires
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. Experimental kernels are
JIT-only and tested under `tests/experimental/`. Follow the
[subtree instructions](flashinfer/experimental/AGENTS.md) and
[experimental policy](flashinfer/experimental/README.md).

## Validation and performance

Run the tests affected by a change and the changed-file pre-commit hooks.
Tests under `tests/cpu` must run without a GPU; public test runs have a separate
CPU lane. Use the regular/full suites when the scope warrants them. Guard tests with
`flashinfer.utils` architecture checks or the API's backend-support helpers;
see [testing and scheduling](docs/development/testing.md).

Check numerical behavior against a reference. For CUDA graphs, validate replay
as inputs, plans, and backend choices change; an eager run alone does not prove
captured buffers remain valid. Read the
[backend lifetime notes](docs/development/backend_notes.md) before changing
attention planning or provider selection.

Performance results need a reproducible benchmark, named GPU, problem sizes,
and absolute before/after times. State cache conditions and timing boundaries;
keep compilation/startup, host time, and GPU kernel time distinct. Use
`flashinfer.testing.bench_gpu_time(..., enable_cupti=True)` for kernel timing
(CUPTI with CUDA-event fallback; `pip install -U cupti-python` for CUDA 13+).
See the [benchmarking guide](.claude/skills/benchmark-kernel/SKILL.md).

For resource reuse or longer lifetimes, apply the
[resource lifetime and performance checklist](docs/code_review_guidance.md#resource-lifetime-and-performance)
during design and review. Measure retained memory and aggregate live resources,
test changing workloads and failure cleanup, and check application headroom.
An API warmup benchmark does not establish full vLLM startup or KV-cache fit.

## Reviews, pull requests, and CI

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and
[the review guidance](docs/code_review_guidance.md). Agent reviews include kernel
implementation details: read indexing, precision, synchronization, and lifetime
logic; label findings by confidence. Passing tests complement that review.
A [self-review](.claude/skills/self-review/SKILL.md) is recommended before a PR.

Keep the [default PR template](.github/pull_request_template.md), explain the
final behavior, and report relevant validation and performance results.
PR titles and descriptions often become squash-merge messages; make them useful
to someone investigating the change later. Mention AI assistance transparently.

Trigger targeted CI by default:

- Public: `@flashinfer-bot run tests/<dir-or-file> [tests/...]`
- Internal: `/bot run tests/<dir-or-file> [tests/...]`

Targets are paths under `tests/`, without `::` selectors. Bare `run` asks for a
scope recommendation: focused or uncertain results are only posted as advice,
but a full-suite recommendation launches the full suite. `run all` requests the
expensive full suite; declare and justify that scope in the PR template. Pushing a commit does not start a new
public CI run; request tests for the final head when needed. See
[CI command details](docs/design_docs/ci_bot_and_targeted_testing.md).

## Debugging and detailed references

API logging defaults to off. Both logging variables are read when FlashInfer is
imported, so set them before import. Set `FLASHINFER_LOGLEVEL=1` for names, `3`
for inputs/outputs, `5` for statistics, or `10` for tensor dumps.
`FLASHINFER_LOGDEST` selects stdout, stderr, or a file. Inputs are logged before
execution. During CUDA graph capture, level 5 captures a statistics kernel that
prints on each replay; statistics are skipped only when that kernel cannot run. For verbose optimized nvcc builds, set
`FLASHINFER_JIT_VERBOSE=1` and `FLASHINFER_JIT_DEBUG=0`: if debug is unset, verbose
also enables debug builds. Set debug to `1` for symbols, `-O0`, and device
debugging. See the
[CUDA debugging guide](.claude/skills/debug-cuda-crash/SKILL.md).

The default cache is `~/.cache/flashinfer/<version>/<arch>/`, with `generated/`
and `cached_ops/` subdirectories. `FLASHINFER_WORKSPACE_BASE` changes its base.
Inspect generated sources and `build.ninja` when diagnosing builds. Use a fresh
workspace for cold-cache comparisons.

| Working on | Read |
| --- | --- |
| JIT, cache keys, bindings, tracing, or wheel builds | [JIT and build reference](docs/development/jit_and_build.md) |
| Backend dispatch, graph lifetimes, or dtype contracts | [Backend implementation notes](docs/development/backend_notes.md) |
| Test scheduling, resource telemetry, or architecture guards | [Testing reference](docs/development/testing.md) |
| Build, cache, logging, or backend environment settings | [Environment variables](docs/development/environment_variables.md) |
| Persistent CI caches and artifact publication | [CI cache contract](ci/COMPILE_CACHE.md) |

Keep this guide and relevant skills/reference pages synchronized with code.
Put detailed backend contracts and new environment settings in the linked
references so this file remains a concise entry point.
