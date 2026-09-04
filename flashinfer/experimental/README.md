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

Using experimental functionality is always an **explicit, visible opt-in**.
There are two explicit forms, and neither needs an environment variable:

- calling a function marked `@flashinfer_experimental_api`;
- naming an experimental backend in a stable API, e.g. `backend="sm12x_cute"`.

Both emit an `ExperimentalWarning` once (per API, or per API/backend pair).

What is gated is **automatic selection**. A stable API called with
`backend="auto"` — including the dispatch heuristics and autotuning behind
it — may pick an experimental backend only when the user opts in with:

```bash
export FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1
```

Without it, automatic selection considers only stable backends, and a call
whose only viable candidates are experimental fails with a
`BackendSupportedError` that names the variable and suggests passing
`backend=` explicitly. With it, experimental backends join the candidate list,
and an `ExperimentalWarning` is emitted once per API/backend pair when one is
picked.

Trace-apply is outside this gate: experimental backends are not required to
support trace, and a trace-apply solution is the deployer's explicit choice.

Importing `flashinfer.experimental` is always allowed so that tooling, docs,
and introspection work; nothing is enforced at import time.

### Gating primitives

Defined in `flashinfer/api_logging.py` (`experimental_backend` lives in
`flashinfer/utils.py` next to `backend_requirement`), all re-exported from
`flashinfer.experimental`:

- `@flashinfer_experimental_api(trace=..., feature=...)` — marks a public
  experimental API. Composes with `@flashinfer_api` (logging/dump/trace still
  work), emits an `ExperimentalWarning` once on first use, and sets
  `is_experimental = True` for mechanical identification. It does **not**
  consult the environment variable: calling the API is the opt-in.
- `@experimental_backend` — marks a `@backend_requirement` checker as an
  experimental backend. `backend="auto"` skips it unless the variable is set;
  explicit `backend="<name>"` always works and warns once; the name appears in
  `<api>.experimental_backends`. A checker defined under
  `flashinfer.experimental` without this marker makes `@backend_requirement`
  raise `ValueError` at import time.
- `require_experimental_auto_backends(feature)` — for stable APIs that route
  `backend="auto"` by hand (without `@backend_requirement`): call it in the
  automatic-routing branch before handing off to an experimental backend.
- `experimental_auto_backends_allowed()` — raw check of the variable.
- `warn_experimental_backend_once(api, backend)` — the once-per-pair warning,
  for hand-rolled dispatch that selects an experimental backend explicitly.

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

Calling `my_new_op` is the opt-in. The decorator warns once per process
(`ExperimentalWarning`) and sets `is_experimental = True`; no environment
variable is involved.

Note that the function body is exactly the **thin core entry point** from the
selected design: shared validation, then a deferred (function-local) import
and a direct handoff to `flashinfer.experimental`. Everything
backend-specific stays under `flashinfer/experimental/`, and the deferred
import keeps `import flashinfer` from ever loading experimental kernels.

### Exposing an experimental backend from a stable API

Worked example: a hypothetical SM12x CuTe GEMM backend behind the stable
`mm_bf16`. The experimental package declares its own support checker and
marks it. The support module must stay **lightweight** — dtype, shape, and
compute-capability logic only, no kernel or JIT imports — because core imports
it at module load so that `@backend_requirement` can register the backend:

```python
# flashinfer/experimental/sm12x_gemm/support.py
import torch
from flashinfer.utils import supported_compute_capability
from flashinfer.experimental import experimental_backend

@experimental_backend
@supported_compute_capability([120, 121])
def check_sm12x_cute(a, b, out=None, backend="auto"):
    return a.dtype == torch.bfloat16 and a.shape[-1] % 64 == 0
```

The core entry point registers it like any other backend. Nothing else in
core changes shape, and the kernel import stays deferred:

```python
# flashinfer/gemm/gemm_base.py
from ..experimental.sm12x_gemm.support import check_sm12x_cute  # checker only

@backend_requirement(
    backend_checks={
        "cutlass": _check_cutlass,
        "cudnn": _check_cudnn,
        "sm12x_cute": check_sm12x_cute,  # experimental: auto skips it unless opted in
    },
    heuristic_func=_heuristic_mm_bf16,
)
@flashinfer_api
def mm_bf16(a, b, out=None, backend="auto"):
    if backend == "auto":
        backend = mm_bf16.suitable_auto_backends[0]
    if backend == "sm12x_cute":
        from ..experimental.sm12x_gemm import run  # deferred import
        return run(a, b, out)
    ...
```

Resulting behavior:

| Call | Variable unset | `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1` |
|---|---|---|
| `mm_bf16(a, b)` | candidates: cutlass, cudnn | may include `sm12x_cute`; warns once if picked |
| `mm_bf16(a, b, backend="sm12x_cute")` | runs; warns once for (mm_bf16, sm12x_cute) | same |

Autotuned APIs tune over `mm_bf16.suitable_auto_backends`, so the same filter
keeps experimental backends out of autotuning while the variable is unset.

Stable APIs that route `"auto"` by hand (no `@backend_requirement`) apply the
same rule with the helper — only on the automatic branch, never on the
explicit one:

```python
from .api_logging import require_experimental_auto_backends

if backend == "auto" and candidate == "experimental_xyz":
    require_experimental_auto_backends("op_name -> experimental_xyz")
if candidate == "experimental_xyz":
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

The PR declares itself experimental by ticking the **Experimental Track**
checkbox in the PR template and linking the tracking issue there; maintainers
add the `experimental` label. The tracking issue is reviewed together with the
PR itself, especially when the PR introduces a new experimental feature (as
opposed to small improvements to an existing experimental backend): reviewers
assess the documented use case, the reason for taking the experimental path,
and the graduation plan as part of admission.

Broad portability, comprehensive performance coverage, and long-term
maintainability are **not** required at admission.

### Relaxations vs. the stable-API checklist

Relative to the "Adding a New Operation" checklist in the root `CLAUDE.md`:

- **Trace templates** (`flashinfer/trace/templates/`) are optional for
  experimental APIs (needed for graduation).
- **AOT registration** (`flashinfer/aot.py`) is **prohibited**: experimental
  features are JIT-only so they never ship in `flashinfer-jit-cache` /
  `flashinfer-cubin` pre-built packages. There is no exception -- shipping an
  experimental kernel in a pre-built package is what the rule exists to
  prevent, and a graduating feature registers for AOT as part of graduation.
- Top-level export from `flashinfer/__init__.py` is optional and, if added,
  must be lazy (no eager import of `flashinfer.experimental`).

## Containment rules

- Core must not import from `flashinfer.experimental` at module level, with
  one narrow exception: the lightweight *support module* that defines an
  `@experimental_backend` checker may be imported so `@backend_requirement`
  can register the backend. Kernels, JIT specs, and heavy dependencies are
  reached only through a deferred (function-local) import inside a sanctioned
  thin entry point.
- `flashinfer/__init__.py` must not eagerly import `flashinfer.experimental`
  (import cost; experimental code may depend on optional packages).
- Code under `flashinfer/experimental/` may import freely from core.
- Experimental backends for stable, autotunable APIs must ensure that
  experimental tactics captured while experimental features are enabled do
  not interfere with environments where experimental features are turned
  off: autotune caches and `trace_apply` configurations containing
  experimental tactics must be safely skippable by loaders — falling back to
  stable tactics — when `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` is unset or the feature has been removed.

### Isolation from stable tests and checks

Stable CI lanes and PR checks must never fail because of experimental code:

- `tests/experimental/` is excluded from `pytest tests/` via `norecursedirs`
  in `pytest.ini`; experimental tests run only in their own lane, invoked
  explicitly as `pytest tests/experimental/`.
- Trace-registry tooling (`tests/trace/`) and `scripts/pr_checks/` match
  `@flashinfer_api` by literal name on purpose; `@flashinfer_experimental_api`
  functions are excluded from the stable trace-consistency tests and from the
  docstring and API/RST coverage checks.
- Experimental APIs are not listed in `docs/api/*.rst` before graduation
  (the API/RST check would report them as stale).

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
