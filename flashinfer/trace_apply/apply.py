# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Trace Apply engine: enable/disable, monkey-patch, dispatch.

A single wrapper handles both stateless free functions and stateful plan/run
Wrapper APIs — the only difference is how the call *namespace* is built. From
the namespace everything is shared: extract the const axes, recompute the
definition name (``name_prefix`` + const-axis abbrevs, the same convention the
trace collector uses), look it up in the registered ``{definition_name:
solution}`` mapping (cached per name), and adapt its outputs to the API.

Error policy is strict: a *matched* solution that fails to load or run re-raises
(a broken registered solution must surface); a genuine miss falls back to the
original API.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

import torch

from flashinfer.trace.solution import Solution
from flashinfer.trace_apply import adapt, plan_capture
from flashinfer.trace_apply.config import ENABLE_ENV, PATH_ENV
from flashinfer.trace_apply.loaders import load as load_solution

_log = logging.getLogger("flashinfer.trace_apply")

_MISSING = object()


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def current_sm(device: Any = None) -> str | None:
    """SM arch string ("sm{major}{minor}") of the current/again device, or None.

    Uses ``flashinfer.utils.get_compute_capability`` — the device reports its own
    compute capability, so there is no GPU product-name table to maintain.
    """
    try:
        from flashinfer.utils import get_compute_capability  # noqa: PLC0415

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        major, minor = get_compute_capability(device)
        return f"sm{major}{minor}"
    except Exception:  # noqa: BLE001 — no CUDA device, etc.
        return None


# ---------------------------------------------------------------------------
# Signature binding + axis extraction
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _signature(fn: Callable) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def bind_namespace(
    original: Callable, args: tuple, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Bind args/kwargs to ``original``'s signature → flat ``{param: value}``."""
    sig = _signature(original)
    if sig is None:
        return dict(kwargs)
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return dict(kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def build_extractor_maps(templates: list) -> list[dict[str, Callable]]:
    """Pre-build each template's per-axis extractor callables (once, at install)."""
    maps: list[dict[str, Callable]] = []
    for tmpl in templates:
        try:
            maps.append(tmpl._build_axis_extractors())
        except Exception:  # noqa: BLE001 — a malformed template must not break others
            continue
    return maps


def extract_axes(
    extractor_maps: list[dict[str, Callable]], namespace: dict[str, Any]
) -> dict[str, int]:
    """Run extractors over a (param → value) namespace → full concrete axis vector.
    Multiple templates merge: each axis is filled by the first that resolves it."""
    axes: dict[str, int] = {}
    for emap in extractor_maps:
        for axis_name, fn in emap.items():
            if axis_name in axes:
                continue
            try:
                val = fn(namespace)
            except Exception:  # noqa: BLE001
                val = None
            if val is not None:
                axes[axis_name] = int(val)
    return axes


# ---------------------------------------------------------------------------
# Stats (per (fi_api, status))
# ---------------------------------------------------------------------------

_stats_lock = Lock()
_stats: Counter[tuple[str, str]] = Counter()
_logged_dispatch: set[str] = set()


def bump_stat(fi_api: str, status: str) -> None:
    with _stats_lock:
        _stats[(fi_api, status)] += 1


def stats_snapshot() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with _stats_lock:
        for (fi_api, status), count in _stats.items():
            out.setdefault(fi_api, {})[status] = count
    return out


def _reset_stats() -> None:
    with _stats_lock:
        _stats.clear()
        _logged_dispatch.clear()


# ---------------------------------------------------------------------------
# Loaded-callable cache (per solution)
# ---------------------------------------------------------------------------

_loaded_lock = Lock()
_loaded: dict[tuple[str, str], Callable] = {}
_loading_locks: dict[tuple[str, str], Lock] = {}


def _get_loaded_solution(solution: Solution) -> Callable:
    key = (solution.definition, solution.name)
    with _loaded_lock:
        cached = _loaded.get(key)
        if cached is not None:
            return cached
        lock = _loading_locks.get(key)
        if lock is None:
            lock = _loading_locks[key] = Lock()
    # Build under a per-solution lock so concurrent callers don't compile twice;
    # different solutions still build in parallel.
    with lock:
        with _loaded_lock:
            cached = _loaded.get(key)
            if cached is not None:
                return cached
        fn = load_solution(solution)
        with _loaded_lock:
            _loaded[key] = fn
        return fn


def reset_loaded() -> None:
    with _loaded_lock:
        _loaded.clear()


# A resolved dispatch entry: (callable, dps, positional).
_Entry = tuple


def _resolve_entry(value: Any) -> _Entry:
    """Resolve a ``solutions`` mapping value into ``(fn, dps, positional)``.

    A plain Python callable is used directly (value-returning, keyword-called); a
    first-class ``Solution`` is loaded via the loaders (its language family
    decides positional-vs-keyword and destination-passing-style).
    """
    if isinstance(value, Solution):
        fn = _get_loaded_solution(value)
        return (
            fn,
            bool(value.spec.destination_passing_style),
            not value.spec.is_python_family,
        )
    if callable(value):
        return (value, False, False)
    raise TypeError(
        f"solutions values must be a callable or a Solution, got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Wrapper construction (unified stateless + stateful)
# ---------------------------------------------------------------------------


def _make_wrapper(
    *,
    fi_api: str,
    original: Callable,
    build_namespace: Callable[[tuple, dict], dict[str, Any]],
    extractor_maps: list,
    template: Any,
    dests: dict,
    is_inplace: bool,
    solutions_by_name: dict[str, Any],
) -> Callable:
    """Wrapper that routes a call to a registered solution by *definition name*.

    Per call: build the namespace, extract the const axes, compute this
    template's definition name (``name_prefix`` + const-axis abbrevs — the same
    logic the trace collector uses), and look it up in ``solutions_by_name``. The
    per-name decision (resolved entry or miss) is cached, so later calls for the
    same shape are a dict lookup. Inside CUDA-graph capture only the cached path
    runs (eager warmup populates the cache before capture)."""
    cache_lock = Lock()
    by_name: dict[str, _Entry | None] = {}

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        try:
            namespace = build_namespace(args, kwargs)
            axes = extract_axes(extractor_maps, namespace)
            name = template.definition_name(axes)
        except Exception:  # noqa: BLE001 — never let extraction break the engine
            return original(*args, **kwargs)

        with cache_lock:
            cached = by_name.get(name, _MISSING)

        if cached is _MISSING:
            if torch.cuda.is_current_stream_capturing():
                bump_stat(fi_api, "fallback_unwarmed_in_capture")
                return original(*args, **kwargs)
            cached = _resolve_name(
                fi_api=fi_api,
                name=name,
                solutions_by_name=solutions_by_name,
                cache=by_name,
                cache_lock=cache_lock,
            )

        if cached is None:
            bump_stat(fi_api, "fallback_no_candidate")
            return original(*args, **kwargs)

        fn, dps, positional = cached
        # Strict: a *matched* solution that raises is NOT masked by falling back.
        try:
            result = adapt.adapt_and_call(
                template=template,
                fn=fn,
                namespace=namespace,
                dps=dps,
                is_inplace=is_inplace,
                dests=dests,
                positional=positional,
            )
        except Exception:
            bump_stat(fi_api, "error")
            _log.error(
                "Trace Apply: applied solution for %s raised (strict mode → re-raising; "
                "the registered solution is broken for these inputs).",
                fi_api,
            )
            raise

        bump_stat(fi_api, "hit")
        return result

    # idempotency marker (so install() won't double-wrap)
    wrapper._trace_apply = True  # type: ignore[attr-defined]
    return wrapper


def _resolve_name(
    *,
    fi_api: str,
    name: str,
    solutions_by_name: dict[str, Any],
    cache: dict,
    cache_lock: Lock,
) -> _Entry | None:
    value = solutions_by_name.get(name)
    if _log.isEnabledFor(logging.DEBUG):
        _log.debug(
            "Trace Apply: resolve %s name=%s -> %s",
            fi_api,
            name,
            "HIT" if value is not None else "miss",
        )
    if value is None:
        with cache_lock:
            cache[name] = None
        return None
    _mark_fired(name)  # this registered solution actually matched a live call
    # Strict: a matched solution that fails to load/build is an error, not a miss.
    try:
        entry = _resolve_entry(value)
    except Exception as e:
        bump_stat(fi_api, "error")
        raise RuntimeError(
            f"Trace Apply: failed to load solution for definition {name!r} on {fi_api}: {e}"
        ) from e
    with cache_lock:
        cache[name] = entry
    if fi_api not in _logged_dispatch:
        _logged_dispatch.add(fi_api)
        _log.info(
            "Trace Apply: applying solution for definition %r on %s (dps=%s).",
            name,
            fi_api,
            entry[1],
        )
    return entry


# ---------------------------------------------------------------------------
# Namespace builders (stateless vs stateful plan/run)
# ---------------------------------------------------------------------------


def _stateless_namespace_builder(
    bind_target: Callable,
) -> Callable[[tuple, dict], dict]:
    def build(args: tuple, kwargs: dict) -> dict:
        return bind_namespace(bind_target, args, kwargs)

    return build


def _stateful_namespace_builder(
    bind_target: Callable, template: Any, adapter
) -> Callable[[tuple, dict], dict]:
    def build(args: tuple, kwargs: dict) -> dict:
        ns = bind_namespace(bind_target, args, kwargs)
        self_obj = args[0] if args else None
        return plan_capture.augment_namespace(adapter, template, ns, self_obj)

    return build


def _make_plan_wrapper(plan_original: Callable) -> Callable:
    @functools.wraps(plan_original)
    def plan_wrapper(self, *args, **kwargs):
        try:
            bound = bind_namespace(plan_original, (self,) + args, kwargs)
            bound.pop("self", None)
            plan_capture.stash_plan_kwargs(self, bound)
        except Exception:  # noqa: BLE001 — stashing must never break plan()
            _log.debug("Trace Apply: failed to stash plan kwargs", exc_info=True)
        return plan_original(self, *args, **kwargs)

    plan_wrapper._trace_apply = True  # type: ignore[attr-defined]
    return plan_wrapper


def _is_inplace_api(original: Callable) -> bool:
    """True if the API mutates buffers and returns None (e.g. fused_add_rmsnorm)."""
    try:
        ann = inspect.signature(original).return_annotation
        # With ``from __future__ import annotations`` the annotation is the
        # string ``"None"`` rather than the ``None`` object, so match both.
        return ann is None or ann is type(None) or ann == "None"
    except (TypeError, ValueError):
        return False


def _derive_output_dests(template: Any, original: Callable) -> dict[str, str]:
    """Output ``{name: destination API param}``: the trace-declared bindings plus
    caller output buffers (``out=``/``lse=``) auto-derived from the live API
    signature.

    The trace only declares bindings that *can't* be derived (in-place writes,
    e.g. ``fused_add_rmsnorm`` → ``input``/``residual``). The ``out=``/``lse=``
    buffers, by contrast, are a uniform FlashInfer convention readable straight
    off ``original``'s signature, so they're derived here rather than stored in
    the trace. A trace ``param`` always wins (so a non-conventional buffer can be
    declared explicitly)."""
    dests = dict(adapt.output_dests(template))
    sig = _signature(original)
    if sig is None:
        return dests
    params = sig.parameters
    for json_key in adapt._tensor_output_keys(template):
        if json_key in dests:
            continue
        buf = "lse" if json_key == "lse" else "out"
        if buf in params:
            dests[json_key] = buf
    return dests


# ---------------------------------------------------------------------------
# Target / alias resolution
# ---------------------------------------------------------------------------


def _resolve_target(dotted: str) -> tuple[Any, str] | None:
    parts = dotted.split(".")
    if len(parts) < 2:
        return None
    module = None
    module_end = 0
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        try:
            module = importlib.import_module(candidate)
            module_end = i
            break
        except ImportError:
            continue
    if module is None:
        return None
    owner: Any = module
    for part in parts[module_end : len(parts) - 1]:
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    attr = parts[-1]
    if not hasattr(owner, attr):
        return None
    return owner, attr


def _build_alias_map() -> dict[int, list[tuple[Any, str]]]:
    """Reverse-index flashinfer module attributes by id(value) so re-exported
    free functions get patched everywhere they are bound as a module attribute.
    (Class methods have no module alias and are patched via the class.)"""
    amap: dict[int, list[tuple[Any, str]]] = {}
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("flashinfer"):
            continue
        if mod_name.startswith("flashinfer.trace_apply"):
            continue
        d = getattr(mod, "__dict__", None)
        if not d:
            continue
        for attr, val in list(d.items()):
            if callable(val):
                amap.setdefault(id(val), []).append((mod, attr))
    return amap


def _registry_by_fi_api() -> dict[str, tuple[Callable, list]]:
    """fi_api -> (undecorated original, [templates]) from the live trace registry."""
    out: dict[str, tuple[Callable, list]] = {}
    try:
        from flashinfer.api_logging import _TRACE_REGISTRY  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return out
    for original, template, _label in _TRACE_REGISTRY:
        module = getattr(original, "__module__", "") or ""
        qualname = getattr(original, "__qualname__", "") or ""
        fi_api = f"{module}.{qualname}" if module else qualname
        if not fi_api:
            continue
        entry = out.get(fi_api)
        if entry is None:
            out[fi_api] = (original, [template])
        else:
            entry[1].append(template)
    return out


# ---------------------------------------------------------------------------
# Install / state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Patch:
    owner: Any
    attr: str
    original: Callable


_patches: list[_Patch] = []
_install_lock = Lock()
_solutions: dict[str, Any] = {}
# Names registered in the current solutions map that have not yet matched a live
# call (each is cleared as it fires). Surfaced on disable to catch silent no-ops.
_unfired: set[str] = set()
_unfired_lock = Lock()


def _mark_fired(name: str) -> None:
    with _unfired_lock:
        _unfired.discard(name)


def _template_matches_any(template: Any, sol_map: dict[str, Any]) -> bool:
    """True if some registered definition name could be produced by this template
    (equals ``name_prefix``, or starts with ``name_prefix`` + ``'_'``). Used to
    skip-wrap APIs that no registered solution can ever target."""
    prefix = (
        template.name_prefix if template.name_prefix is not None else template.op_type
    )
    if not prefix:
        return False
    return any(k == prefix or k.startswith(prefix + "_") for k in sol_map)


def _solutions_from_env() -> dict[str, Any]:
    """Load a ``{definition_name: Solution}`` mapping from ``FLASHINFER_TRACE_APPLY_PATH``.

    Scans ``<path>/solutions/**/*.json`` (first-class Solution JSON) and keys each
    by the definition it targets — one solution per definition (duplicate →
    error). Returns ``{}`` if the path is unset or has no ``solutions/`` dir. This
    is the entry used by deployments (and spawned worker processes) that configure
    Trace Apply via the environment rather than an in-memory mapping."""
    import json as _json
    from pathlib import Path

    path = os.environ.get(PATH_ENV)
    if not path:
        return {}
    sols_dir = Path(path).expanduser() / "solutions"
    if not sols_dir.is_dir():
        return {}
    out: dict[str, Any] = {}
    for p in sorted(sols_dir.rglob("*.json")):
        sol = Solution.from_dict(_json.loads(p.read_text(encoding="utf-8")))
        if sol.definition in out:
            raise ValueError(
                f"Trace Apply: duplicate solution for definition {sol.definition!r} "
                f"(in {p}); the folder must hold one solution per definition."
            )
        out[sol.definition] = sol
    return out


def enable_apply(solutions: dict[str, Any] | None = None) -> int:
    """Substitute kernels at runtime, selected by definition name.

    ``solutions`` maps a definition name (e.g. ``"rmsnorm_h1536"``) to either a
    Python callable or a first-class :class:`~flashinfer.trace.Solution`. Every
    public FlashInfer API whose template could produce one of those names is
    wrapped; a call whose computed definition name is in the mapping dispatches
    to the registered solution, otherwise it falls back to the original kernel.
    Returns the number of attributes wrapped. Idempotent (re-enabling replaces
    the previous mapping).

    If ``solutions`` is ``None``, the environment configuration is used; if
    nothing is configured this is a no-op.
    """
    global _solutions, _unfired
    with _install_lock:
        if _patches:
            _disable_locked()
        if solutions is None:
            solutions = _solutions_from_env()
        sol_map = {str(k): v for k, v in dict(solutions).items()}
        if not sol_map:
            _log.info("Trace Apply: no solutions provided; nothing to apply.")
            return 0
        registry = _registry_by_fi_api()
        alias_map = _build_alias_map()

        wrapped = 0
        matched_apis = 0
        for fi_api, (original, templates) in registry.items():
            template0 = templates[0] if templates else None
            if template0 is None:
                continue
            if not _template_matches_any(template0, sol_map):
                continue  # skip-wrap: no registered solution can target this API
            resolved = _resolve_target(fi_api)
            if resolved is None:
                _log.debug("Trace Apply: cannot resolve target %s; skipping.", fi_api)
                continue
            owner, attr = resolved
            current = getattr(owner, attr)
            if getattr(current, "_trace_apply", False):
                continue

            matched_apis += 1
            extractor_maps = build_extractor_maps(templates)
            dests = _derive_output_dests(template0, original)
            is_inplace = _is_inplace_api(original)
            if is_inplace and not dests:
                _log.warning(
                    "Trace Apply: %s is in-place but has no output destination map; "
                    "skipping to avoid silent corruption.",
                    fi_api,
                )
                continue

            if plan_capture.is_stateful(fi_api):
                adapter = plan_capture.adapter_for(fi_api)
                build_ns = _stateful_namespace_builder(original, template0, adapter)
                if hasattr(owner, adapter.plan_attr):
                    plan_current = getattr(owner, adapter.plan_attr)
                    if not getattr(plan_current, "_trace_apply", False):
                        setattr(
                            owner, adapter.plan_attr, _make_plan_wrapper(plan_current)
                        )
                        _patches.append(
                            _Patch(
                                owner=owner,
                                attr=adapter.plan_attr,
                                original=plan_current,
                            )
                        )
            else:
                build_ns = _stateless_namespace_builder(original)

            wrapper = _make_wrapper(
                fi_api=fi_api,
                original=current,
                build_namespace=build_ns,
                extractor_maps=extractor_maps,
                template=template0,
                dests=dests,
                is_inplace=is_inplace,
                solutions_by_name=sol_map,
            )
            # Patch the canonical target + module-level aliases for the same
            # callable object (free functions only; methods have none).
            targets = [(owner, attr)]
            for o2, a2 in alias_map.get(id(current), []):
                if (o2, a2) != (owner, attr):
                    targets.append((o2, a2))
            for o, a in targets:
                if getattr(o, a, None) is not current:
                    continue
                setattr(o, a, wrapper)
                _patches.append(_Patch(owner=o, attr=a, original=current))
                wrapped += 1

        _solutions = sol_map
        with _unfired_lock:
            _unfired = set(sol_map)
        _log.info(
            "Trace Apply: wrapped %d attributes across %d APIs for %d registered solution(s).",
            wrapped,
            matched_apis,
            len(sol_map),
        )
        return wrapped


def disable_apply() -> None:
    with _install_lock:
        _disable_locked()


def _disable_locked() -> None:
    global _solutions, _unfired
    while _patches:
        p = _patches.pop()
        try:
            setattr(p.owner, p.attr, p.original)
        except Exception:  # noqa: BLE001
            _log.warning(
                "Trace Apply: failed to revert %s.%s", p.owner, p.attr, exc_info=True
            )
    with _unfired_lock:
        never = sorted(_unfired)
        _unfired = set()
    if never:
        _log.warning(
            "Trace Apply: %d registered solution(s) never matched a call: %s. "
            "Check the definition names/shapes — these were NOT applied.",
            len(never),
            never,
        )
    _solutions = {}
    reset_loaded()
    _reset_stats()


def is_enabled() -> bool:
    return bool(_patches)


def _enable_apply_from_env() -> None:
    """Import-time hook: enable apply if ``FLASHINFER_TRACE_APPLY=1``, loading the
    solutions from ``FLASHINFER_TRACE_APPLY_PATH``. Internal — called only by the
    ``flashinfer`` package import. Warning-only: a bad config never makes
    ``import flashinfer`` fail."""
    enabled = os.environ.get(ENABLE_ENV, "0")
    if enabled in ("", "0"):
        return
    if enabled != "1":
        warnings.warn(
            f"{ENABLE_ENV} must be '0' or '1', got {enabled!r}; trace apply disabled.",
            stacklevel=2,
        )
        return
    try:
        n = enable_apply()  # solutions=None → loaded from the environment
        if n == 0:
            warnings.warn(
                f"{ENABLE_ENV}=1 but no solutions were loaded from {PATH_ENV}; "
                "set it to a curated solutions folder.",
                stacklevel=2,
            )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to enable trace apply: {type(exc).__name__}: {exc}", stacklevel=2
        )


__all__ = [
    "current_sm",
    "disable_apply",
    "enable_apply",
    "is_enabled",
    "stats_snapshot",
]
