# CLAUDE.md — flashinfer/experimental/

Agent guidance for code under `flashinfer/experimental/`. The normative
policy is [README.md](./README.md) in this directory; this file is the
operational summary. The root `CLAUDE.md` still applies except where relaxed
below.

## What lives here

Experimental **backends** and backend-specific logic: support checks,
heuristics, routing, compilation, caching, and kernels not yet ready for
stable support. Public experimental **APIs** do NOT live here — they live in
core, marked with `@flashinfer_experimental_api`, with only a thin entry
point that validates, selects the backend, and hands off to this package.

## Hard rules (containment)

- **Never** import `flashinfer.experimental` from core at module level, with
  one exception: the lightweight support module that defines an
  `@experimental_backend` checker (dtype/shape/CC logic only, no kernel or
  JIT imports). Kernels are reached via a deferred (function-local) import in
  a sanctioned thin entry point.
- **Never** register experimental modules in `flashinfer/aot.py` —
  experimental features are JIT-only and must not ship in pre-built packages.
- **Never** let `backend="auto"` (dispatch or autotuning) pick an
  experimental backend unless `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`.
  `@backend_requirement` enforces this for checkers marked
  `@experimental_backend`; hand-rolled `"auto"` routing must call
  `require_experimental_auto_backends(...)` on its automatic branch.
- Explicit opt-ins need **no** environment variable: calling an
  `@flashinfer_experimental_api` function, or passing `backend="<name>"` to a
  stable API. Both warn once (`ExperimentalWarning`).
- **Never** eagerly import this package from `flashinfer/__init__.py`.
- Autotuning inherits the gate because it tunes over the filtered candidate
  list. Trace-apply is out of scope: experimental backends need not support
  trace.

## Adding an experimental API (new public function)

1. Implement the backend under `flashinfer/experimental/<feature>/`.
2. Add the public function in the appropriate core module, decorated:

   ```python
   from .api_logging import flashinfer_experimental_api

   @flashinfer_experimental_api
   def my_new_op(x, ...):
       """Docstring (the decorator prepends the experimental banner)."""
       # shared validation only
       from .experimental.my_feature import run  # deferred import
       return run(x, ...)
   ```

3. Calling the API is the opt-in. The decorator warns once per process
   (`ExperimentalWarning`) and sets `is_experimental = True`; it does not
   consult any environment variable.

## Exposing an experimental backend from a stable API

Mark the backend's support checker in the experimental package and register
it in the stable API's `@backend_requirement`; `backend="auto"` then skips it
unless `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`, while an explicit
`backend="sm12x_cute"` always works (and warns once):

```python
# flashinfer/experimental/sm12x_gemm/support.py  (lightweight: no kernel imports)
from flashinfer.utils import supported_compute_capability
from flashinfer.experimental import experimental_backend

@experimental_backend
@supported_compute_capability([120, 121])
def check_sm12x_cute(a, b, out=None, backend="auto"):
    return a.dtype == torch.bfloat16 and a.shape[-1] % 64 == 0
```

```python
# flashinfer/gemm/gemm_base.py
from ..experimental.sm12x_gemm.support import check_sm12x_cute  # checker only

@backend_requirement(
    backend_checks={"cutlass": _check_cutlass, "sm12x_cute": check_sm12x_cute},
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

For hand-rolled `"auto"` routing (no `@backend_requirement`), guard only the
automatic branch:

```python
from .api_logging import require_experimental_auto_backends

if backend == "auto" and candidate == "experimental_xyz":
    require_experimental_auto_backends("op_name -> experimental_xyz")
```

See README.md for the full worked example and the resulting behavior table.

## Requirements for every feature

- Named owner + tracking issue (use case, reason, graduation plan/target;
  default intent: graduate within four weeks).
- Declare the PR via the **Experimental Track** checkbox in the PR template,
  linking the tracking issue (maintainers add the `experimental` label).
- Correctness tests vs. a reference in `tests/experimental/`, validated on
  the intended hardware, plus a runnable example.

## Checklist deltas vs. root CLAUDE.md's "Adding a New Operation"

| Step | Experimental status |
|------|---------------------|
| Trace template + `tests/trace/example.py` | Optional (recommended before graduation) |
| Register in `flashinfer/aot.py` | Prohibited |
| Export in `flashinfer/__init__.py` | Optional; must be lazy if added |
| Tests | Required, under `tests/experimental/` |
| Docs | Must state experimental status, how to opt in, and limitations |

Everything else (JIT module structure, framework separation of
`include/`/`csrc/`, coding style) follows the root `CLAUDE.md`.

## Graduation / removal

Follow the graduation checklist in [README.md](./README.md). Removal needs no
deprecation cycle: delete the backend directory, the tagged API, and the
tests, and note it in the tracking issue.
