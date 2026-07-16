# TODO: suppress nixl_ep everywhere until a proper backend is installed

Status: analysis only, nothing implemented yet (2026-07-14).

Goal: when the nixl_ep backend is not actually usable (no staged
`nixl_ep_cpp*.so`, no `nixl-cu13` runtime wheel), the nixl_ep path is
suppressed across **all of moe_ep** — nothing imports, registers,
advertises, or warns about it. That covers every consumer: mega/cutedsl
users, split-path users on nccl_ep, and plain `import flashinfer.moe_ep`.
Once the proper backend is installed, everything lights up unchanged.

The probe for "properly installed" already exists — `_probe_nixl_ep()` in
`moe_ep/__init__.py` (checks `backends/split/comm/nixl_ep/_libs/` for
`nixl_ep_cpp*.so`) — the work is gating the surfaces below on it.

## Current surfaces where nixl_ep leaks through when not installed

1. **Unconditional import at package import time** —
   `moe_ep/__init__.py` (bottom):

   ```python
   from .backends.split.comm.nixl_ep import fleet as _nixl_ep_fleet
   ```

   and `backends/split/comm/__init__.py` imports `NvepConfig` from
   `nixl_ep.config`. Importing the module currently succeeds by design
   (the `.so` loads are lazy, see the nixl_ep `__init__` docstring), but it
   still pays import cost and — more importantly — any future regression in
   that subtree breaks `import flashinfer.moe_ep` for **every** moe_ep
   consumer (mega, split/nccl_ep, or anything just importing the package),
   none of whom asked for nixl. Fix: gate the fleet import on
   `_probe_nixl_ep()` (mirror how `available_backends()` already filters).

2. **Registry side effect regardless of install state** —
   `backends/split/comm/nixl_ep/fleet.py:182-183`:

   ```python
   # Module-load side effect: register the backend.
   _BACKEND_REGISTRY["nixl_ep"] = NixlEpFleet
   ```

   So `create_fleet(backend="nixl_ep")` resolves the class and only fails
   inside `NixlEpFleet.__init__` via `_require_built("nixl_ep")`
   (fleet.py:79). The registry (and the "available:" list in
   `create_fleet`'s unknown-backend `KeyError`, `core/comm/fleet.py:69-71`)
   should reflect what is actually usable.

   **Error-UX decision needed** when suppressed: a bare
   `KeyError: unknown backend 'nixl_ep'` would be a regression vs today's
   `MoEEpNotBuiltError` with the rebuild hint. Recommended shape: keep the
   name *known* to `create_fleet` (a static set of recognised backends) but
   not *registered*; on a recognised-but-unbuilt name raise
   `MoEEpNotBuiltError` with `_REBUILD_HINT`, on a truly unknown name keep
   the `KeyError`.

3. **Build-time noise for users who never asked for nixl_ep** — the default
   `pip install .` attempts the nixl_ep build with skip-with-warning
   semantics (`BUILD_NIXL_EP=1` upgrades to hard error), and the
   `_REBUILD_HINT` / import-time `RuntimeWarning` in `moe_ep/__init__.py`
   mention nixl for every moe_ep install regardless of which path the user
   wants. Consider making the nixl_ep build opt-in (or at least silencing
   its probe-miss warnings unless `BUILD_NIXL_EP` is set), so an install
   without nixl deps is warning-clean.

4. **Runtime-requirements coupling** — `core/runtime/bootstrap.py:241`
   treats `("nccl_ep", "nixl_ep")` together when computing bootstrap
   requirements. Harmless today (only reachable when a split comm backend
   is explicitly selected), but double-check nothing on the mega path pulls
   this branch in when nixl_ep is suppressed.

## Non-goals / already correct

- `available_backends()` and `have_nixl_ep()` already probe correctly —
  keep them as the single source of truth.
- The lazy `.so` loading design in `backends/split/comm/nixl_ep/__init__.py`
  (`_preload_libnixl` / `_load_nixl_ep_cpp` are module-level but not invoked
  at import) is right; don't make loading eager, make *registration*
  conditional.
- Validation branches in `core/validation/common.py` (`backend ==
  "nixl_ep"` limits: LOW_LATENCY-only, EXPERT_MAJOR-only, hidden-size set,
  1024 max tokens/rank) only fire when nixl_ep is explicitly selected —
  no change needed.
- nccl_ep has the same import-time registration pattern
  (`_nccl_ep_fleet`); this TODO is scoped to nixl_ep per request, but if
  the gating shape works, apply it symmetrically in the same PR.

## Sketch

```python
# moe_ep/__init__.py (bottom)
from .backends.split.comm.nccl_ep import fleet as _nccl_ep_fleet  # noqa
if _probe_nixl_ep():
    from .backends.split.comm.nixl_ep import fleet as _nixl_ep_fleet  # noqa
```

plus the recognised-but-unbuilt error path in `create_fleet`, plus a test:
`import flashinfer.moe_ep` with no `_libs/` staged → no nixl imports in
`sys.modules`, `"nixl_ep" not in available_backends()`,
`create_fleet(..., backend="nixl_ep")` raises `MoEEpNotBuiltError` (not
`KeyError`), and zero warnings emitted.
