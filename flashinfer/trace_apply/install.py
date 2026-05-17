from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import current_sm
from flashinfer.trace_apply.index import Index, build_index
from flashinfer.trace_apply.runtime import make_wrapper, reset_loaded, reset_stats
from flashinfer.trace_apply.schema import Definition
from flashinfer.trace_apply.source import load_trace

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
# Target attribute resolution
# ---------------------------------------------------------------------------


def _resolve_target(dotted: str) -> tuple[Any, str] | None:
    """Resolve "flashinfer.norm.rmsnorm" or "flashinfer.x.Wrapper.run" to
    `(owner, attr)` such that `setattr(owner, attr, new)` does the right thing.
    Returns None if any part cannot be resolved.
    """
    parts = dotted.split(".")
    if len(parts) < 2:
        return None

    # Walk down: import the longest module prefix we can, then descend through
    # class attributes for the rest.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(
    path: str | None = None,
    config: TraceApplyConfig | None = None,
) -> int:
    """Activate Trace Apply: load the trace, build the index, and wrap the
    public flashinfer attributes that have a matching candidate on the
    running SM. Returns the number of attributes wrapped.

    Idempotent: a second call replaces the prior install.
    """
    global _index, _config
    with _install_lock:
        if _patches:
            _disable_locked()
        trace = load_trace(path)
        index = build_index(trace)
        cfg = config or TraceApplyConfig()
        sm = _safe_current_sm()
        wrapped = 0
        for defn in index.definitions.values():
            if not _wraps_anything(index, defn, sm):
                continue
            target_dotted = defn.fi_api()
            if target_dotted is None:
                continue
            resolved = _resolve_target(target_dotted)
            if resolved is None:
                _log.debug("Trace Apply: cannot resolve target %s for %s; skipping.", target_dotted, defn.name)
                continue
            owner, attr = resolved
            original = getattr(owner, attr)
            wrapper = make_wrapper(definition=defn, original=original, index=index, config=cfg)
            setattr(owner, attr, wrapper)
            _patches.append(_Patch(owner=owner, attr=attr, original=original))
            wrapped += 1
        _index = index
        _config = cfg
        _log.info("Trace Apply: wrapped %d flashinfer attributes (sm=%s).", wrapped, sm)
        return wrapped


def disable() -> None:
    """Revert every patch and clear caches."""
    with _install_lock:
        _disable_locked()


def _disable_locked() -> None:
    global _index, _config
    while _patches:
        p = _patches.pop()
        try:
            setattr(p.owner, p.attr, p.original)
        except Exception:  # noqa: BLE001
            _log.warning("Trace Apply: failed to revert %s.%s", p.owner, p.attr, exc_info=True)
    _index = None
    _config = None
    reset_loaded()
    reset_stats()


def _wraps_anything(index: Index, defn: Definition, sm: str | None) -> bool:
    """Skip-wrap test: True iff the trace has at least one PASSED candidate
    for this definition on the running SM.
    """
    if sm is None:
        return False
    return index.has_candidates_for(defn.name, sm)


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
