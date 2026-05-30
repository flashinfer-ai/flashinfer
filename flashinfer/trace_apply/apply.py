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
the namespace everything is shared: extract the axis vector + input dtypes, form
the routing key ``(fi_api, const-axes, input-dtypes)``, look up the one
registered solution (cached per shape), and adapt its outputs to the API.

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

from flashinfer.trace_apply import adapt, plan_state
from flashinfer.trace_apply.config import (
    ENABLE_ENV,
    ApplyPolicy,
    load_config,
    resolve_trace_path,
)
from flashinfer.trace_apply.loaders import load as load_solution
from flashinfer.trace_apply.routing import Candidate, Index, build_index, lookup

_log = logging.getLogger("flashinfer.trace_apply")

# Opt-in: log the full extracted axis vector + dtypes on every newly-resolved
# shape (hit or miss), to author/debug matching solutions.
_DEBUG = os.environ.get("FLASHINFER_TRACE_APPLY_DEBUG", "0") not in ("0", "", "false", "False")

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


def bind_namespace(original: Callable, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
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


def extract_axes(extractor_maps: list[dict[str, Callable]], namespace: dict[str, Any]) -> dict[str, int]:
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
# Stats (per (fi_api, status) and (author, status))
# ---------------------------------------------------------------------------

_stats_lock = Lock()
_stats: Counter[tuple[str, str]] = Counter()
_author_stats: Counter[tuple[str, str]] = Counter()
_logged_dispatch: set[str] = set()


def bump_stat(fi_api: str, status: str, author: str | None = None) -> None:
    with _stats_lock:
        _stats[(fi_api, status)] += 1
        if author is not None:
            _author_stats[(author, status)] += 1


def stats_snapshot() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with _stats_lock:
        for (fi_api, status), count in _stats.items():
            out.setdefault(fi_api, {})[status] = count
    return out


def author_stats_snapshot() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with _stats_lock:
        for (author, status), count in _author_stats.items():
            out.setdefault(author, {})[status] = count
    return out


def reset_stats() -> None:
    with _stats_lock:
        _stats.clear()
        _author_stats.clear()
        _logged_dispatch.clear()


# ---------------------------------------------------------------------------
# Loaded-callable cache (per solution)
# ---------------------------------------------------------------------------

_loaded_lock = Lock()
_loaded: dict[tuple[str, str], Callable] = {}
_loading_locks: dict[tuple[str, str], Lock] = {}


def _get_loaded(cand: Candidate) -> Callable:
    key = (cand.solution.definition, cand.solution.name)
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
        fn = load_solution(cand.solution)
        with _loaded_lock:
            _loaded[key] = fn
        return fn


def reset_loaded() -> None:
    with _loaded_lock:
        _loaded.clear()


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
    const_names: set[str],
    dests: dict,
    activation: dict,
    is_inplace: bool,
    index: Index,
    policy: ApplyPolicy,
) -> Callable:
    """Thin wrapper around ``original`` routing to the registered solution.

    Routing key = the call's const axes + input dtypes (the definition identity);
    var axes do not gate dispatch. Per-key decision is cached, so later calls for
    the same shape are a dict lookup. Inside CUDA-graph capture only the cached
    path runs."""
    cache_lock = Lock()
    # cache key = (const-axes frozenset, input-dtypes frozenset)
    by_key: dict[tuple, tuple[Callable, str, bool, bool] | None] = {}

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        try:
            namespace = build_namespace(args, kwargs)
            axes = extract_axes(extractor_maps, namespace)
            dtypes = adapt.extract_input_dtypes(template, namespace)
        except Exception:  # noqa: BLE001 — never let extraction break the engine
            return original(*args, **kwargs)

        const_axes = {k: v for k, v in axes.items() if k in const_names}
        key = (frozenset(const_axes.items()), dtypes)
        with cache_lock:
            cached = by_key.get(key, _MISSING)

        if cached is _MISSING:
            if torch.cuda.is_current_stream_capturing():
                # Cannot resolve/JIT inside capture; eager warmup populates the
                # cache before capture.
                bump_stat(fi_api, "fallback_unwarmed_in_capture")
                return original(*args, **kwargs)
            cached = _resolve_and_cache(
                fi_api=fi_api,
                const_axes=const_axes,
                input_dtypes=dtypes,
                index=index,
                policy=policy,
                cache=by_key,
                cache_lock=cache_lock,
                key=key,
            )

        if cached is None:
            bump_stat(fi_api, "fallback_no_candidate")
            return original(*args, **kwargs)

        fn, author, dps, positional = cached
        # Strict: a *matched* solution that raises is NOT masked by falling back.
        try:
            result = adapt.adapt_and_call(
                template=template,
                fn=fn,
                namespace=namespace,
                dps=dps,
                is_inplace=is_inplace,
                dests=dests,
                activation=activation,
                positional=positional,
            )
        except Exception:
            bump_stat(fi_api, "error", author)
            _log.error(
                "Trace Apply: applied solution for %s raised (strict mode → re-raising; "
                "the registered solution is broken for these inputs).",
                fi_api,
            )
            raise

        bump_stat(fi_api, "hit", author)
        return result

    wrapper._trace_apply = True  # type: ignore[attr-defined] — idempotency marker
    return wrapper


def _resolve_and_cache(
    *,
    fi_api: str,
    const_axes: dict[str, int],
    input_dtypes: frozenset,
    index: Index,
    policy: ApplyPolicy,
    cache: dict,
    cache_lock: Lock,
    key: tuple,
) -> tuple[Callable, str, bool, bool] | None:
    sm = current_sm()
    cand = lookup(index, fi_api, const_axes, input_dtypes, sm, policy)
    if _DEBUG:
        _log.info(
            "Trace Apply [debug]: resolve %s sm=%s const_axes=%s dtypes=%s -> %s",
            fi_api,
            sm,
            dict(sorted(const_axes.items())),
            dict(sorted(input_dtypes)),
            "HIT(" + cand.solution.name + ")" if cand else "miss",
        )
    if cand is None:
        with cache_lock:
            cache[key] = None
        return None
    # Strict: a matched solution that fails to load/build is an error, not a miss.
    try:
        fn = _get_loaded(cand)
    except Exception as e:
        bump_stat(fi_api, "error", cand.solution.author)
        raise RuntimeError(
            f"Trace Apply: failed to load solution {cand.solution.name!r} "
            f"(language={cand.solution.spec.language.value}) for {fi_api}: {e}"
        ) from e
    positional = not cand.solution.spec.is_python_family
    entry = (
        fn,
        cand.solution.author,
        bool(cand.solution.spec.destination_passing_style),
        positional,
    )
    with cache_lock:
        cache[key] = entry
    if fi_api not in _logged_dispatch:
        _logged_dispatch.add(fi_api)
        _log.info(
            "Trace Apply: applying solution %r (author=%s, lang=%s, dps=%s) for %s const_axes=%s",
            cand.solution.name,
            cand.solution.author,
            cand.solution.spec.language.value,
            entry[2],
            fi_api,
            dict(sorted(const_axes.items())),
        )
    return entry


# ---------------------------------------------------------------------------
# Namespace builders (stateless vs stateful plan/run)
# ---------------------------------------------------------------------------


def _stateless_namespace_builder(bind_target: Callable) -> Callable[[tuple, dict], dict]:
    def build(args: tuple, kwargs: dict) -> dict:
        return bind_namespace(bind_target, args, kwargs)

    return build


def _stateful_namespace_builder(bind_target: Callable, template: Any, adapter) -> Callable[[tuple, dict], dict]:
    def build(args: tuple, kwargs: dict) -> dict:
        ns = bind_namespace(bind_target, args, kwargs)
        self_obj = args[0] if args else None
        return plan_state.augment_namespace(adapter, template, ns, self_obj)

    return build


def _make_plan_wrapper(plan_original: Callable) -> Callable:
    @functools.wraps(plan_original)
    def plan_wrapper(self, *args, **kwargs):
        try:
            bound = bind_namespace(plan_original, (self,) + args, kwargs)
            bound.pop("self", None)
            plan_state.stash_plan_kwargs(self, bound)
        except Exception:  # noqa: BLE001 — stashing must never break plan()
            _log.debug("Trace Apply: failed to stash plan kwargs", exc_info=True)
        return plan_original(self, *args, **kwargs)

    plan_wrapper._trace_apply = True  # type: ignore[attr-defined]
    return plan_wrapper


def _is_inplace_api(original: Callable) -> bool:
    """True if the API mutates buffers and returns None (e.g. fused_add_rmsnorm)."""
    try:
        return inspect.signature(original).return_annotation is None
    except (TypeError, ValueError):
        return False


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
_index: Index | None = None
_policy: ApplyPolicy | None = None


def enable_apply(path: str | None = None, policy: ApplyPolicy | None = None) -> int:
    """Load the solution folder, build the routing table, and wrap every public
    flashinfer API that has a registered solution for the running SM. Returns the
    number of attributes wrapped. Idempotent (re-enabling re-reads the folder)."""
    global _index, _policy
    with _install_lock:
        if _patches:
            _disable_locked()
        config = load_config(path)
        index = build_index(config)
        pol = policy or ApplyPolicy()
        registry = _registry_by_fi_api()
        alias_map = _build_alias_map()

        wrapped = 0
        for fi_api, (original, templates) in registry.items():
            if not index.has_candidates_for(fi_api):
                continue  # skip-wrap: no registered solution for this API
            resolved = _resolve_target(fi_api)
            if resolved is None:
                _log.debug("Trace Apply: cannot resolve target %s; skipping.", fi_api)
                continue
            owner, attr = resolved
            current = getattr(owner, attr)
            if getattr(current, "_trace_apply", False):
                continue

            extractor_maps = build_extractor_maps(templates)
            template0 = templates[0] if templates else None

            dests = adapt.output_dests(fi_api)
            is_inplace = _is_inplace_api(original)
            if is_inplace and not dests:
                _log.warning(
                    "Trace Apply: %s is in-place but has no output destination map; "
                    "skipping to avoid silent corruption.",
                    fi_api,
                )
                continue

            if plan_state.is_stateful(fi_api):
                adapter = plan_state.adapter_for(fi_api)
                build_ns = _stateful_namespace_builder(original, template0, adapter)
                if hasattr(owner, adapter.plan_attr):
                    plan_current = getattr(owner, adapter.plan_attr)
                    if not getattr(plan_current, "_trace_apply", False):
                        setattr(owner, adapter.plan_attr, _make_plan_wrapper(plan_current))
                        _patches.append(_Patch(owner=owner, attr=adapter.plan_attr, original=plan_current))
            else:
                build_ns = _stateless_namespace_builder(original)

            wrapper = _make_wrapper(
                fi_api=fi_api,
                original=current,
                build_namespace=build_ns,
                extractor_maps=extractor_maps,
                template=template0,
                const_names=index.const_names(fi_api),
                dests=dests,
                activation=adapt.output_activation(fi_api),
                is_inplace=is_inplace,
                index=index,
                policy=pol,
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

        _index = index
        _policy = pol
        _log.info(
            "Trace Apply: wrapped %d attributes for %d routed APIs.",
            wrapped,
            sum(1 for fi in registry if index.has_candidates_for(fi)),
        )
        return wrapped


def disable_apply() -> None:
    with _install_lock:
        _disable_locked()


def _disable_locked() -> None:
    global _index, _policy
    while _patches:
        p = _patches.pop()
        try:
            setattr(p.owner, p.attr, p.original)
        except Exception:  # noqa: BLE001
            _log.warning("Trace Apply: failed to revert %s.%s", p.owner, p.attr, exc_info=True)
    _index = None
    _policy = None
    reset_loaded()
    reset_stats()


def is_enabled() -> bool:
    return bool(_patches)


def get_index() -> Index | None:
    return _index


def get_policy() -> ApplyPolicy | None:
    return _policy


def enable_apply_from_env() -> None:
    """Enable apply at import time if ``FLASHINFER_TRACE_APPLY=1``. Warning-only:
    a bad config never makes ``import flashinfer`` fail."""
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
        resolve_trace_path(None)  # validate FLASHINFER_TRACE_PATH is set
        enable_apply()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to enable trace apply: {type(exc).__name__}: {exc}", stacklevel=2
        )


__all__ = [
    "enable_apply",
    "disable_apply",
    "enable_apply_from_env",
    "is_enabled",
    "get_index",
    "get_policy",
    "current_sm",
    "stats_snapshot",
    "author_stats_snapshot",
    "reset_stats",
]
