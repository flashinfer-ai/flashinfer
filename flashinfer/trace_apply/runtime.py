from __future__ import annotations

import logging
from collections import Counter
from threading import Lock
from typing import Any, Callable

import torch

from flashinfer.trace_apply.axes import extract_axes
from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import current_sm
from flashinfer.trace_apply.index import Candidate, Index
from flashinfer.trace_apply.loader import load as load_solution
from flashinfer.trace_apply.lookup import lookup
from flashinfer.trace_apply.schema import Definition

_log = logging.getLogger("flashinfer.trace_apply")


# ---------------------------------------------------------------------------
# Per-process stats
# ---------------------------------------------------------------------------

_stats_lock = Lock()
_stats: Counter[tuple[str, str]] = Counter()  # (definition, status) -> count


def bump_stat(definition: str, status: str) -> None:
    with _stats_lock:
        _stats[(definition, status)] += 1


def stats_snapshot() -> dict[str, dict[str, int]]:
    """Snapshot the in-process counter table by definition."""
    out: dict[str, dict[str, int]] = {}
    with _stats_lock:
        for (defn, status), count in _stats.items():
            out.setdefault(defn, {})[status] = count
    return out


def reset_stats() -> None:
    with _stats_lock:
        _stats.clear()


# ---------------------------------------------------------------------------
# Per-(solution) loaded-callable cache
# ---------------------------------------------------------------------------

_loaded_lock = Lock()
_loaded: dict[tuple[str, str], Callable] = {}  # (definition, solution) -> callable


def _get_loaded(cand: Candidate) -> Callable:
    key = (cand.solution.definition, cand.solution.name)
    with _loaded_lock:
        cached = _loaded.get(key)
        if cached is not None:
            return cached
    # Loading happens outside the lock so we don't serialize JIT compiles.
    fn = load_solution(cand.solution)
    with _loaded_lock:
        _loaded[key] = fn
        return fn


def reset_loaded() -> None:
    with _loaded_lock:
        _loaded.clear()


# ---------------------------------------------------------------------------
# Wrapper construction
# ---------------------------------------------------------------------------


def make_wrapper(
    *,
    definition: Definition,
    original: Callable,
    index: Index,
    config: TraceApplyConfig,
) -> Callable:
    """Return a thin wrapper around `original` that routes to a Trace Apply
    candidate when one is available, falling through to `original` otherwise.

    Per-shape decision is cached so every call after the first for a given
    shape is a dict lookup. Inside a CUDA graph capture, only the cached path
    is exercised — never JIT, never `lookup()` on a fresh shape.
    """

    cache_lock = Lock()
    by_axes_cache: dict[frozenset[tuple[str, int]], Callable | None] = {}

    def wrapper(*args, **kwargs):
        # Extract axes from the call. Cheap on every call (a few shape reads).
        try:
            axes = extract_axes(definition, original, args, kwargs)
        except Exception:  # noqa: BLE001 — never let axis extraction break the engine
            return original(*args, **kwargs)

        axes_key = frozenset(axes.items())
        with cache_lock:
            cached = by_axes_cache.get(axes_key, _MISSING)

        if cached is _MISSING:
            # Cache miss. Inside capture we cannot resolve safely (lookup may
            # touch the index / load a kernel via JIT). Fall through this call;
            # the engine's eager warmup is expected to populate the cache before
            # capture begins.
            if torch.cuda.is_current_stream_capturing():
                bump_stat(definition.name, "fallback_unwarmed_in_capture")
                return original(*args, **kwargs)
            cached = _resolve_and_cache(
                definition=definition,
                axes=axes,
                index=index,
                config=config,
                cache=by_axes_cache,
                cache_lock=cache_lock,
                axes_key=axes_key,
            )

        if cached is None:
            bump_stat(definition.name, "fallback_no_candidate")
            return original(*args, **kwargs)

        try:
            result = cached(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — fall back to the reference impl on any runtime error
            _log.warning(
                "Trace Apply candidate raised %s for %s; falling back to FlashInfer default.",
                type(e).__name__,
                definition.name,
            )
            bump_stat(definition.name, "fallback_runtime_error")
            return original(*args, **kwargs)

        bump_stat(definition.name, "hit")
        return result

    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(original, "__name__", "trace_apply_wrapper")
    wrapper.__qualname__ = getattr(original, "__qualname__", wrapper.__name__)
    return wrapper


_MISSING = object()


def _resolve_and_cache(
    *,
    definition: Definition,
    axes: dict[str, int],
    index: Index,
    config: TraceApplyConfig,
    cache: dict[frozenset[tuple[str, int]], Callable | None],
    cache_lock: Lock,
    axes_key: frozenset[tuple[str, int]],
) -> Callable | None:
    sm = current_sm()
    cand = lookup(index, definition.name, axes, sm, config)
    if cand is None:
        with cache_lock:
            cache[axes_key] = None
        return None
    try:
        fn = _get_loaded(cand)
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "Trace Apply failed to load candidate %s for %s: %s; falling back.",
            cand.solution.name,
            definition.name,
            e,
        )
        with cache_lock:
            cache[axes_key] = None
        return None
    with cache_lock:
        cache[axes_key] = fn
    return fn


__all__ = [
    "bump_stat",
    "make_wrapper",
    "reset_loaded",
    "reset_stats",
    "stats_snapshot",
]
