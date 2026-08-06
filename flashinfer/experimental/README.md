# FlashInfer Experimental APIs and Backends

This document is the normative policy for experimental functionality in
FlashInfer. Code under `flashinfer/experimental/` and every API marked with
`@flashinfer_experimental_api` is governed by the rules below.

## Overview

FlashInfer applies uniform review and support expectations to stable
user-facing functionality. That works well for long-term-supported features,
but less well for fast-moving work — client-GPU (e.g. SM12x) kernels, new
operations from the latest models, or highly specialized kernels for specific
problem sizes — where users value functional availability over API or
implementation stability.

FlashInfer therefore distinguishes:

- an **experimental API** — a public interface that may change or disappear
  (e.g. a new op for a new model); and
- an **experimental backend** — an implementation that is not yet ready for
  stable support (e.g. a generated or architecture-specific kernel).

These are separate concerns:

|                  | Stable Backend        | Experimental Backend |
| ---------------- | --------------------- | -------------------- |
| Stable API       | Normal path           | Allowed with opt-in  |
| Experimental API | Not a target use case | Allowed with opt-in  |

In this README, **core** refers to the existing non-experimental codebase
(everything outside `flashinfer/experimental/`).

## Selected design

- **Experimental APIs live in core**, marked with
  `@flashinfer_experimental_api` (defined in `flashinfer/api_logging.py`).
  Graduation is then a tag removal with no import-path change for users.
- **Experimental backends and backend-specific logic live under
  `flashinfer/experimental/`** (importable as `flashinfer.experimental`).

Where core contains an experimental entry point, it may include only:

- the public API signature;
- general (shared) validation;
- the feature-gate check;
- backend selection;
- a direct handoff to `flashinfer.experimental`.

Backend-specific support checks, heuristics, routing, compilation, caching,
and kernels remain under `flashinfer/experimental/`. If the *API* is
experimental, its implementation lives here too, regardless of the
implementation's maturity — this keeps removal a one-directory deletion.

## Feature gating

All experimental behavior requires:

```bash
export FLASHINFER_ENABLE_EXPERIMENTAL_FEATURES=1
```

**Without** this opt-in:

- experimental APIs raise `RuntimeError` when called;
- stable APIs must not select experimental backends;
- automatic routing (backend dispatch, autotuning, trace-apply substitution)
  must consider only stable backends.

**The environment variable is the single opt-in.** With it set, stable APIs
may route to experimental backends automatically: backend dispatch,
autotuning, and trace-apply substitution may consider experimental backends
alongside stable ones. Setting the flag is consent to experimental behavior
wherever it applies; an explicit `backend=` argument remains available to
force a specific backend but is not required.

Importing `flashinfer.experimental` is always allowed so that tooling, docs,
and introspection work without the flag; the gate is enforced at call time.

### Gating primitives

Defined in `flashinfer/api_logging.py` and re-exported from
`flashinfer.experimental`:

- `@flashinfer_experimental_api(trace=..., feature=...)`
  — marks a public experimental API. Composes with `@flashinfer_api`
  (logging/dump/trace still work), enforces the gate, emits an
  `ExperimentalWarning` once per process, and sets `is_experimental = True`
  for mechanical identification.
- `require_experimental(feature)` — call in the thin core entry point of a
  stable API before handing off to an experimental backend.
- `is_experimental_enabled()` — raw gate check.

### Adding an experimental API (new public function)

Implement the backend under `flashinfer/experimental/<feature>/`, then add
the public function in the appropriate core module, decorated:

```python
from .api_logging import flashinfer_experimental_api

@flashinfer_experimental_api
def my_new_op(x, ...):
    """Docstring (the decorator prepends the experimental banner)."""
    # shared validation only
    from .experimental.my_feature import run  # deferred import
    return run(x, ...)
```

The decorator enforces the gate (`RuntimeError` when off), warns once per
process (`ExperimentalWarning`), and sets `is_experimental = True`.

Note that the function body is exactly the **thin core entry point** from the
selected design: shared validation, then a deferred (function-local) import
and a direct handoff to `flashinfer.experimental`. Everything
backend-specific stays under `flashinfer/experimental/`, and the deferred
import keeps `import flashinfer` from ever loading experimental code.

### Exposing an experimental backend from a stable API

In the stable API's dispatch — behind an explicit `backend=` value or an
automatic-routing branch — guard the handoff with `require_experimental`:

```python
from .api_logging import require_experimental

if backend == "experimental_xyz":
    require_experimental("op_name experimental_xyz backend")
    from .experimental.xyz import run  # deferred import
    return run(...)
```

## Ownership and lifecycle

Every experimental feature must have:

- a **named owner**;
- a **tracking issue** that documents the feature's use case, states the
  reason for entering the experimental path, and states a graduation plan
  with a target release (e.g. "finalize the API and graduate by the 0.6.xx
  release in four weeks").

Default intent for any experimental PR is graduation within **four weeks**.
Experimental features are reviewed on that cadence and either continue
incubating, graduate through the normal stable process, or are removed.

Continued incubation is not automatic. At each lifecycle review, the owner
must document why the feature is still experimental, provide evidence of
active use or progress, and identify the remaining graduation blockers. A
maintainer must approve the extension. Repeating the same justification
without meaningful progress is not sufficient.

Broken tests, loss of ownership, or lack of meaningful usage are sufficient
reasons for removal **without** stable deprecation guarantees.

## Admission criteria

The admitted use cases are the fast-moving work this policy exists for:

- **client-GPU kernels and backends** (e.g. SM12x);
- **new operations from the latest models**, where functional support can
  land ahead of a finalized stable API;
- **highly specialized kernels for specific problem sizes**, which enter as
  experimental backends behind an existing stable API rather than as new
  APIs.

Other categories may be admitted with maintainer approval.

A feature requires documented justification, such as:

- a new operator family or algorithm;
- a materially different API contract;
- a new compiler, backend, or architecture-specific implementation path;
- a use case FlashInfer has not committed to maintain.

A new experimental API should **not** be created for a parameter-space
extension of an existing stable API. An experimental backend may implement an
existing stable API when it introduces a materially new implementation path.

## Testing and review

Every experimental feature must include:

- correctness tests against a reference;
- at least one representative supported configuration;
- validation on the intended hardware;
- a runnable example.

Tests live under `tests/experimental/` and run in a separate CI lane; they
must pass for PRs that modify the feature.

Changes under `flashinfer/experimental/` receive narrower review focused on
eligibility, correctness, containment, and obvious safety or
maintainability risks. Changes to core (including thin entry points) follow
the normal core review process; reviewers verify that integration is
explicit, stable behavior is unchanged by default, and backend-specific logic
has not leaked into core.

The tracking issue is reviewed together with the PR itself, especially when
the PR introduces a new experimental feature (as opposed to small
improvements to an existing experimental backend): reviewers assess the
documented use case, the reason for taking the experimental path, and the
graduation plan as part of admission.

Broad portability, comprehensive performance coverage, and long-term
maintainability are **not** required at admission.

### Relaxations vs. the stable-API checklist

Relative to the "Adding a New Operation" checklist in the root `CLAUDE.md`:

- **Trace templates** (`flashinfer/trace/templates/`) are optional for
  experimental APIs (needed for graduation).
- **AOT registration** (`flashinfer/aot.py`) is **prohibited**: experimental
  features are JIT-only so they never ship in `flashinfer-jit-cache` /
  `flashinfer-cubin` pre-built packages. Exceptions require maintainer
  approval.
- Top-level export from `flashinfer/__init__.py` is optional and, if added,
  must be lazy (no eager import of `flashinfer.experimental`).

## Containment rules

- Core must not import from `flashinfer.experimental`, except inside a
  sanctioned thin entry point, and then only via a deferred (function-local)
  import.
- `flashinfer/__init__.py` must not eagerly import `flashinfer.experimental`
  (import cost; experimental code may depend on optional packages).
- Code under `flashinfer/experimental/` may import freely from core.
- Experimental backends for stable, autotunable APIs must ensure that
  experimental tactics captured while experimental features are enabled do
  not interfere with environments where experimental features are turned
  off: autotune caches and `trace_apply` configurations containing
  experimental tactics must be safely skippable by loaders — falling back to
  stable tactics — when the gate is off or the feature has been removed.

## User contract

Experimental APIs and backends provide no compatibility or long-term support
guarantees and may change or be removed without deprecation.

Documentation for each feature must identify:

- whether the API, the backend, or both are experimental;
- supported use cases and limitations;
- required feature gates;
- whether the feature participates in automatic routing (dispatch,
  autotuning, trace-apply) once the gate is set.

Experimental features are intended primarily for main-branch users, community
containers, and local framework integrations. They should not be enabled by
default in supported framework releases. A request for default or supported
release-path integration is a signal that the feature should be considered
for graduation.

## Graduation checklist

To graduate a feature to stable:

1. Finalize the API contract (naming, dtypes, keyword-only perf parameters).
2. Move backend code from `flashinfer/experimental/` to its stable home;
   leave a re-export shim for one release if the experimental path was
   user-visible.
3. Replace `@flashinfer_experimental_api` with `@flashinfer_api` and remove
   `require_experimental` calls from the entry point.
4. Complete the full stable-API checklist in the root `CLAUDE.md` (trace
   template, `tests/trace/example.py`, docs page, top-level export).
5. Register in `flashinfer/aot.py` if the feature should ship pre-compiled.
6. Move tests from `tests/experimental/` to the corresponding stable test
   directory.
7. Close the tracking issue, noting the graduating release.

Removal instead of graduation: delete the backend directory, the tagged API,
and the tests; note the removal in the tracking issue. No deprecation cycle
is required.
