# Watcher CI execution evidence

Correlated `PR Test` dispatches retain execution facts without changing ordinary
human-triggered CI, its lane intersections, test exit codes, or the positive
`watcher-ci-receipt-v1` submission receipt. A submission receipt identifies roots;
it does not establish executed coverage.

The setup job runs `scripts/pr_checks/ci_execution_manifest.py` from the trusted
workflow revision and publishes:

- Artifact `watcher-ci-plan-<runId>-<runAttempt>`, containing `plan.json`.
- Schema version 1 and kind `watcher-ci-plan`.
- Repository, PR, requested head, frozen base, actual checkout and actual commit
  parents; provider, run, attempt and workflow revision; request key, canonical
  requested scope and scope digest.
- The exact computed `{include: [...]}` matrix, its canonical JSON SHA256, and
  whether the workflow's ordinary skip decision suppressed execution.

Each scheduled lane downloads the actual Setup artifact named by its job output, checks its local checkout
and parents, and records a result through the trusted writer staged outside the
workspace mounted into Docker. Its lane key is the SHA256 of the raw canonical
matrix entry. The wrapper observes the real `ci/bash.sh` entrypoint process and
preserves its exit code. `commandStarted` means the entrypoint started, **not**
that pytest reached test execution. Missing startup, setup failures and cancellation
must not be treated as passing tests.

After the lane, an `always()` step publishes
`watcher-ci-result-<laneKey>-<runId>-<runAttempt>`, containing `result.json` and
bounded copies under `evidence/`:

| Suffix | Evidence kind |
|---|---|
| `.xml` | `junit` |
| `manifest.json` | `execution-plan` |
| `.results.json` | `execution-results` |
| `.meta.json` | `execution-metadata` |
| `.telemetry.json` | `telemetry` |

These are existing pytest/sharding artifacts, not reconstructed test results.
Correlated commands forward the observed checkout as `SOURCE_GIT_SHA` so the
existing sharding manifest can retain it. A fresh `junit/` directory is required;
pre-existing output cannot prove a new attempt. Capture rejects linked files and
directories and is bounded to 256 files, 8 MiB per file, 32 MiB total, 2,048
inventory entries, and directory depth 16. Each copied artifact records relative
path, byte size and SHA256. Incomplete capture is an explicit `artifactErrors`
entry. The writer never evaluates XML or emits a coverage-passed verdict.

Results repeat plan/run identity, actual checkout/parents and matrix digest, and
add the raw lane, lane key, provider-visible job name, `evidenceKind` (`build-import`
or `test`), actual command start/exit,
and the normalized lane-specific requested targets. Empty lane targets on a bare
run do not mean no tests were selected: the original sharding plan/results remain
the evidence. Hardware and ranks are `not_recorded` unless actual observations
are available; runner labels, CUDA image labels and job configuration are never
converted to observed GPU usage or rank participation.

## Actual container inventory

Watcher lanes also mount a trusted stdlib inventory helper read-only and run it
inside the same container immediately before the unchanged lane script. Its
separate probe process has a 20-second deadline and 64 KiB output limit; it reads
the actual container checkout and uses CUDA driver enumeration for GPU name and
compute capability. CPU lanes record the actual platform architecture. Failures
emit an explicit unavailable reason and do not replace the original test exit.

The helper emits one context-bound pre-test frame, including a fresh nonce and
its trusted source digest. The host writer accepts only that first frame and
keeps the observation outside the PR-mounted workspace. Later test stdout is
opaque and cannot replace the accepted inventory. The retained artifact has kind
`hardware-observation`, schema version 1, kind field
`watcher-ci-hardware-observation`, and `{context, capturedAt, hardware}`. Context
pins all plan/request/checkout/run/attempt/lane identity, `planRunAttempt`,
`captureNonce`, and `probeSha256`. `result.hardwareContext` retains the expected
binding; the original JSON receives a normal path/hash/size descriptor.

`hardware` includes `status`, `source`, `devices`, `gpuCount`, `cpuArchitecture`,
`gpuUseProven: false`, and `rankParticipation: not_recorded`. Observed GPU entries
have `ordinal`, `name`, `computeCapability`, and normalized `hardwareTarget`.
GPU source is `container-cuda-driver` with `enumeration: cuda-visible`; CPU source
is `container-platform`. Missing/failed inventory is an explicit coverage gap.
Inventory describes visible devices before the lane, not devices used by kernels
or participating distributed ranks. Existing same-runner source trust boundaries
still apply; this is not isolation from arbitrary host-level PR scripts.

The consumer must independently verify the provider run/attempt/workflow/actor,
all planned job names and actual job states, merge ancestry, artifact identities,
and complete execution results against the requested scope. Zero executed cases,
skipped cases, missing lanes, cancellation, absent artifacts or failed cases
cannot establish complete test coverage. AOT build/import lanes have no JUnit by
default and remain build evidence only; they cannot satisfy GPU test coverage.

Artifacts preserve attempts rather than overwriting earlier attempts. A failed-job
rerun can inherit a successful Setup from an earlier attempt. Its result keeps
`runAttempt` equal to the actual executing attempt and `planRunAttempt` equal to
the original Setup attempt; it never invents a fresh Setup execution. The request,
checkout, workflow revision and matrix must remain identical, and a plan from a
future attempt is rejected. Consumers must verify any inherited successful lanes
rather than assuming they executed again. This is
source-reviewed CI evidence, not cryptographic isolation from arbitrary PR code
running on the same self-hosted runner. Existing CI-safe review and runner trust
boundaries still apply. The writer adds no provider write permissions or tokens
to requested-code jobs.

## Local verification

The local `ci-request-execution-contracts` pre-commit hook runs this suite in an
isolated Python environment with pinned PyYAML when its producer/workflow/test
inputs change. The existing required `pre-commit` workflow executes the hook;
no branch-protection rule, provider write permission, or GPU CI job is added.

CPU-only fixtures exercise real temporary Git merge objects (including shallow
checkout headers), actual workflow shell branches with fake Docker entrypoints,
exit-code preservation, missing/linked/oversized artifacts and attempt identity.
They do not call GitHub, allocate runners/GPUs or execute contributor tests:

```bash
python3 -m unittest discover -s tests/ci -p 'test_ci_execution*.py'
python3 -m unittest discover -s tests/ci -p 'test_ci_request*.py'
```
