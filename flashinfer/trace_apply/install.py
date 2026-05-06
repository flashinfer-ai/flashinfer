from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from flashinfer.trace_apply.axes import (
    bind_namespace,
    build_extractor_maps,
    fetch_plan_kwargs,
    stash_plan_kwargs,
)
from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import current_sm
from flashinfer.trace_apply.index import Index, build_index
from flashinfer.trace_apply.runtime import make_wrapper, reset_loaded, reset_stats
from flashinfer.trace_apply.source import load_trace
from flashinfer.trace_apply.stateful import adapter_for, is_stateful

_log = logging.getLogger("flashinfer.trace_apply")


@dataclass(slots=True)
class _Patch:
    owner: Any  # module or class
    attr: str
    original: Callable


_patches: list[_Patch] = []
_install_lock = Lock()
_index: Index | None = None
_config: TraceApplyConfig | None = None


# ---------------------------------------------------------------------------
# Live template registry  (fi_api -> (undecorated original, [TraceTemplate]))
# ---------------------------------------------------------------------------


def _fi_api_of(original: Callable) -> str:
    module = getattr(original, "__module__", "") or ""
    qualname = getattr(original, "__qualname__", "") or ""
    return f"{module}.{qualname}" if module else qualname


def _registry_by_fi_api() -> dict[str, tuple[Callable, list]]:
    out: dict[str, tuple[Callable, list]] = {}
    try:
        from flashinfer.api_logging import _TRACE_REGISTRY  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return out
    for original, template, _label in _TRACE_REGISTRY:
        fi_api = _fi_api_of(original)
        if not fi_api:
            continue
        entry = out.get(fi_api)
        if entry is None:
            out[fi_api] = (original, [template])
        else:
            entry[1].append(template)
    return out


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
    functions get patched everywhere they are bound as a module attribute.
    (Class methods have no module alias and are patched via the class.)
    """
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


# ---------------------------------------------------------------------------
# Namespace builders (stateless vs stateful plan/run)
# ---------------------------------------------------------------------------


def _make_stateless_namespace(original: Callable) -> Callable:
    def build(args: tuple, kwargs: dict) -> dict:
        return bind_namespace(original, args, kwargs)

    return build


def _make_stateful_namespace(run_original: Callable, template: Any, adapter) -> Callable:
    # template-param name to write each plan-sourced input under (so the
    # template extractors + build_candidate_kwargs find it).
    def _ns_key(json_key: str) -> str:
        desc = template.inputs.get(json_key) if template is not None else None
        return (getattr(desc, "param", None) or json_key) if desc else json_key

    plan_ns_key = {jk: _ns_key(jk) for jk in adapter.plan_inputs}
    self_ns_key = {jk: _ns_key(jk) for jk in adapter.self_attrs}

    def build(args: tuple, kwargs: dict) -> dict:
        ns = bind_namespace(run_original, args, kwargs)
        self_obj = args[0] if args else None
        if self_obj is not None:
            # Preferred: read plan-derived inputs straight off the wrapper
            # instance (robust to fast-path planners that bypass plan()).
            for json_key, attr in adapter.self_attrs.items():
                val = getattr(self_obj, attr, None)
                if val is not None:
                    ns[self_ns_key[json_key]] = val
            # Fallback: the public plan() we wrapped stashed its kwargs.
            plan_bound = fetch_plan_kwargs(id(self_obj))
            for json_key, plan_param in adapter.plan_inputs.items():
                ns_key = plan_ns_key[json_key]
                if ns.get(ns_key) is None:
                    val = plan_bound.get(plan_param)
                    if val is not None:
                        ns[ns_key] = val
        return ns

    return build


def _make_plan_wrapper(plan_original: Callable) -> Callable:
    def plan_wrapper(self, *args, **kwargs):
        try:
            bound = bind_namespace(plan_original, (self,) + args, kwargs)
            stash_plan_kwargs(id(self), bound)
        except Exception:  # noqa: BLE001 — stashing must never break plan()
            _log.debug("Trace Apply: failed to stash plan kwargs", exc_info=True)
        return plan_original(self, *args, **kwargs)

    plan_wrapper.__wrapped__ = plan_original  # type: ignore[attr-defined]
    plan_wrapper.__name__ = getattr(plan_original, "__name__", "plan")
    plan_wrapper.__qualname__ = getattr(plan_original, "__qualname__", "plan")
    plan_wrapper._trace_apply = True  # type: ignore[attr-defined]
    return plan_wrapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(
    path: str | None = None,
    config: TraceApplyConfig | None = None,
) -> int:
    """Activate Trace Apply: load the trace, build the index, and wrap the
    public flashinfer attributes that have a matching candidate on the running
    SM. Returns the number of attributes wrapped. Idempotent.
    """
    global _index, _config
    with _install_lock:
        if _patches:
            _disable_locked()
        trace = load_trace(path)
        index = build_index(trace)
        cfg = config or TraceApplyConfig()
        sm = _safe_current_sm()
        registry = _registry_by_fi_api()
        alias_map = _build_alias_map()

        wrapped = 0
        for fi_api, (original, templates) in registry.items():
            if sm is None or not index.has_candidates_for(fi_api, sm):
                continue  # skip-wrap: no candidate for this API on this SM
            resolved = _resolve_target(fi_api)
            if resolved is None:
                _log.debug("Trace Apply: cannot resolve target %s; skipping.", fi_api)
                continue
            owner, attr = resolved
            current = getattr(owner, attr)
            if getattr(current, "_trace_apply", False):
                continue  # already wrapped

            extractor_maps = build_extractor_maps(templates)
            template0 = templates[0] if templates else None

            # Stateful wrappers (plan/run): wrap plan() to stash, and build the
            # run namespace by merging plan-stashed inputs.
            if is_stateful(fi_api):
                adapter = adapter_for(fi_api)
                build_ns = _make_stateful_namespace(original, template0, adapter)
                plan_attr = adapter.plan_attr
                if hasattr(owner, plan_attr):
                    plan_current = getattr(owner, plan_attr)
                    if not getattr(plan_current, "_trace_apply", False):
                        setattr(owner, plan_attr, _make_plan_wrapper(plan_current))
                        _patches.append(_Patch(owner=owner, attr=plan_attr, original=plan_current))
            else:
                build_ns = _make_stateless_namespace(original)

            wrapper = make_wrapper(
                fi_api=fi_api,
                original=current,  # fall back to the decorated public callable
                build_namespace=build_ns,
                extractor_maps=extractor_maps,
                template=template0,
                index=index,
                config=cfg,
            )
            # Patch the canonical target + module-level aliases pointing to the
            # same callable object (free functions only; methods have none).
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
        _config = cfg
        _log.info(
            "Trace Apply: wrapped %d attributes for %d candidate APIs (sm=%s).",
            wrapped,
            sum(1 for fi in registry if sm and index.has_candidates_for(fi, sm)),
            sm,
        )
        return wrapped


def disable() -> None:
    with _install_lock:
        _disable_locked()


def _disable_locked() -> None:
    global _index, _config
    while _patches:
        p = _patches.pop()
        try:
            setattr(p.owner, p.attr, p.original)
        except Exception:  # noqa: BLE001
            _log.warning(
                "Trace Apply: failed to revert %s.%s", p.owner, p.attr, exc_info=True
            )
    _index = None
    _config = None
    reset_loaded()
    reset_stats()


def _safe_current_sm() -> str | None:
    try:
        return current_sm()
    except Exception:  # noqa: BLE001 — no CUDA device, etc.
        return None


def is_installed() -> bool:
    return bool(_patches)


def get_index() -> Index | None:
    return _index


def get_config() -> TraceApplyConfig | None:
    return _config


__all__ = ["install", "disable", "is_installed", "get_index", "get_config"]
