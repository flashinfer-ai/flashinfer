# GPU compile cache

`ci/bash.sh` enables persistent compilation caching only when a GPU caller sets
`CI_COMPILE_CACHE_DIR`. The PR Test workflow uses its work volume; unrelated
validation workflows continue to start cold.

## Isolation and lifecycle

The layout is `<root>/runner-<name hash>/images/<pulled image ID>/`. The runner
identity comes from `CI_COMPILE_CACHE_RUNNER`, then `RUNNER_NAME`, with `local`
as the fallback. Hashing avoids path collisions from sanitized runner names.
The local immutable Docker image ID distinguishes image rebuilds under a fixed
tag, and Docker executes that same image ID after pulling it.

A stable `<root>/.runner-<hash>.lock` protects cache maintenance, snapshots, and
execution. Eviction removes only the runner's image directories, never the lock
or another runner's cache. Multiple services with identical runner identities
serialize rather than corrupting or pruning an active shared cache.

`CI_COMPILE_CACHE_MODE` selects:

- `read-only` (default): mount an existing seed read-only, preserve timestamps
  while copying it into a job-private bind mount, and discard the private tree
  on completion or failure. A missing seed starts cold. A seed exceeding the
  size/free-space budget is skipped without mutating it.
- `read-write`: use the persistent image directory directly. The workflow
  selects this only for scheduled runs on `refs/heads/main` (main has no push
  trigger); PRs and manual runs consume private copies. These jobs prune Ninja modules, generated sources, autotuning
  results, and JIT logs before/after execution. Old image directories and cache
  pressure are handled while holding the runner lock.

Missing locking/image-identity support or invalid configuration disables cache
reuse. A failed image pull stops the job rather than running an older image.
The Docker exit status survives cache cleanup.

This prevents ordinary PR test execution from writing back its compilation
results. It is not a sandbox for malicious code on a self-hosted runner:
approved PR scripts can already execute host commands and invoke Docker.
Runner provisioning and authorization remain the host security boundary.

## Shared S3 layer

`ci/compile_cache_s3.sh` adds a layer shared by every runner, so new EC2 leases
and newly added runners do not start cold. The workflow's "Configure compile
cache" step selects it with `CI_COMPILE_CACHE_S3_URL` and
`CI_COMPILE_CACHE_S3_LANE` (GPU plus `part1`..`part5` for the A10G shards,
`all` otherwise). Objects live at
`s3://<SCCACHE_BUCKET>/compile-cache/main/<image ID>/<GPU>/<lane>.tar.gz` and
are read anonymously over HTTPS.

- `read-only` jobs keep `runner-<hash>/s3/<image ID>/<lane>/tree`, an extracted
  mirror refreshed under the runner lock by a conditional (ETag) download, and
  mount it at `/compile-cache-s3:ro`. The entrypoint copies the mirror into the
  private tree first, then merges the seed over it: a CuTe-DSL module absent
  from the private tree is copied whole, one present gets missing objects only
  when both `meta.json` files are identical, and other artifacts are copied
  only when missing. Mirrors of images unused for a week are evicted.
- Writers are `read-write` jobs with `CI_COMPILE_CACHE_WRITER_DIR` set: scheduled
  runs on main once the `COMPILE_CACHE_S3_ROLE_ARN` repository variable names the
  upload role. They skip the seed, which shares a host with pull request jobs,
  and compile in that fresh directory on top of the S3 tarball alone. After the
  tests, an `always()` step assumes the role through OIDC (its trust policy
  accepts only `repo:flashinfer-ai/flashinfer:ref:refs/heads/main`) and uploads
  the directory's content-keyed artifacts as one object. A single PUT is
  atomic, so concurrent writers never interleave a module's `meta.json` and
  objects. Nothing is uploaded when the job built nothing new; a tarball over
  `CI_COMPILE_CACHE_S3_MAX_MB` (default 4096) is cut down to the job's own
  builds.

Every S3 failure only warns. As above, this protects the shared layer from
ordinary pull request jobs, not from code that takes over a self-hosted host.

## Compatibility after dependency installation

An image ID alone is insufficient: the test wrapper can install branch-specific
requirements or dependency overrides. CuTe-DSL metadata therefore fingerprints
the distributions actually installed in the compiling Python process: all
`nvidia-cutlass-dsl*` packages, Torch, TVM-FFI, and Python's ABI tag. Wheel RECORD
metadata and direct-URL VCS commits distinguish same-version rebuilds/overrides.
As with the existing process-local compiler cache, dependencies must not be
replaced inside a live Python process. Unversioned edits to external editable
dependencies are not captured by wheel metadata.

Source dependency analysis and hashing are implemented in
`flashinfer/jit/_cute_dsl_source.py`, separate from compilation and artifact
management in `cute_dsl_core.py`. Its module docstring describes the intended
CI use and analysis limitations.

Strict source hashing follows static imports and resolvable literal dynamic
imports. Package initializers are resolved per name a kernel looks up, so a
re-export leads to its source rather than to everything the package exports;
their import-time code, including evaluated function annotations, is keyed,
while an added re-export or `__all__` entry is not. Star imports and lazy
fallbacks remain covered when an explicit binding can be overwritten or absent.
Import-time side effects of modules no keyed code imports or uses,
nonliteral imports, and runtime file reads are not covered: list the latter
through `extra_key_files`. This is not a general Python dependency analyzer.

## Validation and measurement

CPU lifecycle tests mock Docker but use real locks, copies, pruning, and shell
exit handling. They live outside `tests/` so broad GPU runs and packaged nightly
tests do not collect host-infrastructure checks. The pre-commit workflow runs
them once on a CPU runner for each PR/main push:

```bash
python -m unittest discover -s ci/tests -p 'test_compile_cache*.py' -v
python -m pytest -q --confcutdir=tests/jit \
  tests/jit/test_jit_cache_key_and_build_log.py \
  tests/jit/test_missing_jit_cache_reporting.py
bash -n ci/bash.sh ci/compile_cache.sh ci/compile_cache_entrypoint.sh ci/compile_cache_s3.sh
```

The JIT tests need FlashInfer's import dependencies but do not execute GPU
kernels. Mocked lifecycle tests establish policy, not GPU artifact compatibility.
For performance validation, compare equivalent cold/warm jobs on the same
persistent runner, source, image, and test scope. Include private-copy and
maintenance overhead in job time; JSONL build totals are task time across
processes, not hit counts or elapsed wall time. No GPU speedup is claimed by
these correctness changes.
