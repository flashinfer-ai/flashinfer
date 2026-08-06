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
point that validates, checks the feature gate, and hands off to this package.

## Hard rules (containment)

- **Never** import `flashinfer.experimental` from core at module level. A
  sanctioned thin entry point in core may do a deferred (function-local)
  import after calling `require_experimental(...)`.
- **Never** register experimental modules in `flashinfer/aot.py` —
  experimental features are JIT-only and must not ship in pre-built packages.
- **Never** let a stable API route here while the gate is off. Setting
  `FLASHINFER_ENABLE_EXPERIMENTAL_FEATURES=1` permits experimental behavior,
  including automatic routing from stable APIs to experimental backends
  (dispatch, autotuning, trace-apply); without it, routing must consider
  only stable backends.
- **Never** eagerly import this package from `flashinfer/__init__.py`.
- Autotuner tactic enumeration and trace_apply substitution must not pick
  experimental paths without the gate.

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

3. The decorator enforces the gate (RuntimeError when off), warns once per
   process (`ExperimentalWarning`), and sets `is_experimental = True`.

## Exposing an experimental backend from a stable API

In the stable API's dispatch — behind an explicit `backend=` value or an
automatic-routing branch — guard the handoff with `require_experimental`:

```python
from .api_logging import require_experimental

if backend == "experimental_xyz":
    require_experimental("op_name experimental_xyz backend")
    from .experimental.xyz import run  # deferred import
    return run(...)
```

## Requirements for every feature

- Named owner + tracking issue (use case, reason, graduation plan/target;
  default intent: graduate within four weeks).
- Correctness tests vs. a reference in `tests/experimental/`, validated on
  the intended hardware, plus a runnable example.

## Checklist deltas vs. root CLAUDE.md's "Adding a New Operation"

| Step | Experimental status |
|------|---------------------|
| Trace template + `tests/trace/example.py` | Optional (recommended before graduation) |
| Register in `flashinfer/aot.py` | Prohibited |
| Export in `flashinfer/__init__.py` | Optional; must be lazy if added |
| Tests | Required, under `tests/experimental/` |
| Docs | Must state experimental status, gate, and limitations |

Everything else (JIT module structure, framework separation of
`include/`/`csrc/`, coding style) follows the root `CLAUDE.md`.

## Graduation / removal

Follow the graduation checklist in [README.md](./README.md). Removal needs no
deprecation cycle: delete the backend directory, the tagged API, and the
tests, and note it in the tracking issue.
