from __future__ import annotations

import logging
import os
from collections import Counter
from threading import Lock
from typing import Any, Callable

import torch

from flashinfer.trace_apply.axes import build_candidate_kwargs, extract_from_namespace
from flashinfer.trace_apply.config import TraceApplyConfig
from flashinfer.trace_apply.hardware import current_sm
from flashinfer.trace_apply.index import Candidate, Index
from flashinfer.trace_apply.loader import load as load_solution
from flashinfer.trace_apply.lookup import lookup

_log = logging.getLogger("flashinfer.trace_apply")

# Opt-in: log the full extracted axis vector on every newly-resolved shape
# (hit or miss). Used to author trace records that match an engine's real
# shapes. Enable with FLASHINFER_TRACE_APPLY_DEBUG=1.
_DEBUG_AXES = os.environ.get("FLASHINFER_TRACE_APPLY_DEBUG", "0") not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# Per-process stats — bucketed by (fi_api, status) and (author, status)
# ---------------------------------------------------------------------------

_stats_lock = Lock()
_stats: Counter[tuple[str, str]] = Counter()  # (fi_api, status) -> count
_author_stats: Counter[tuple[str, str]] = Counter()  # (author, status) -> count

# fi_apis we have already logged a first-dispatch INFO line for. Cross-process
# (per-worker) visibility of "what got applied" — engines run the model in a
# subprocess where stats_snapshot() lives, so a one-time log per API is the
# practical signal an operator sees.
_logged_dispatch: set[str] = set()


def bump_stat(fi_api: str, status: str, author: str | None = None) -> None:
    with _stats_lock:
        _stats[(fi_api, status)] += 1
        if author is not None:
            _author_stats[(author, status)] += 1


def stats_snapshot() -> dict[str, dict[str, int]]:
    """Snapshot the in-process counter table by fi_api.

    Status keys: hit, fallback_no_candidate, fallback_unwarmed_in_capture,
    fallback_runtime_error.
    """
    out: dict[str, dict[str, int]] = {}
    with _stats_lock:
        for (fi_api, status), count in _stats.items():
            out.setdefault(fi_api, {})[status] = count
    return out


def author_stats_snapshot() -> dict[str, dict[str, int]]:
    """Snapshot per-author attribution (hits/fallbacks by Solution.author)."""
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
# Per-(definition, solution) loaded-callable cache
# ---------------------------------------------------------------------------

_loaded_lock = Lock()
_loaded: dict[tuple[str, str], Callable] = {}


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
    fi_api: str,
    original: Callable,
    build_namespace: Callable,
    extractor_maps: list,
    template: Any,
    index: Index,
    config: TraceApplyConfig,
) -> Callable:
    """Return a thin wrapper around ``original`` that routes to a Trace Apply
    candidate when one is available, falling through to ``original`` otherwise.

    ``build_namespace(args, kwargs)`` returns a ``{param: value}`` namespace for
    the call (for stateful wrappers it merges plan-stashed kwargs). Axes are
    extracted from that namespace; the candidate is invoked with kwargs named by
    the template's inputs (``build_candidate_kwargs``), matching how
    flashinfer-bench solutions are written.

    Per-shape decision is cached so every call after the first for a given shape
    is a dict lookup. Inside a CUDA-graph capture only the cached path runs —
    never ``lookup()`` on a fresh shape, never JIT.
    """

    cache_lock = Lock()
    by_axes_cache: dict[frozenset[tuple[str, int]], tuple[Callable, str] | None] = {}

    def wrapper(*args, **kwargs):
        try:
            namespace = build_namespace(args, kwargs)
            axes = extract_from_namespace(extractor_maps, namespace)
        except Exception:  # noqa: BLE001 — never let extraction break the engine
            return original(*args, **kwargs)

        axes_key = frozenset(axes.items())
        with cache_lock:
            cached = by_axes_cache.get(axes_key, _MISSING)

        if cached is _MISSING:
            # Cache miss. Inside capture we cannot resolve safely (lookup/JIT).
            # Fall through; engine eager warmup is expected to populate the
            # cache before capture begins.
            if torch.cuda.is_current_stream_capturing():
                bump_stat(fi_api, "fallback_unwarmed_in_capture")
                return original(*args, **kwargs)
            cached = _resolve_and_cache(
                fi_api=fi_api,
                axes=axes,
                index=index,
                config=config,
                cache=by_axes_cache,
                cache_lock=cache_lock,
                axes_key=axes_key,
            )

        if cached is None:
            bump_stat(fi_api, "fallback_no_candidate")
            return original(*args, **kwargs)

        fn, author = cached
        try:
            call_kwargs = build_candidate_kwargs(template, namespace)
            result = fn(**call_kwargs)
        except Exception as e:  # noqa: BLE001 — any runtime error → reference impl
            _log.warning(
                "Trace Apply candidate raised %s for %s; falling back to FlashInfer default.",
                type(e).__name__,
                fi_api,
            )
            bump_stat(fi_api, "fallback_runtime_error", author)
            return original(*args, **kwargs)

        bump_stat(fi_api, "hit", author)
        return result

    wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(original, "__name__", "trace_apply_wrapper")
    wrapper.__qualname__ = getattr(original, "__qualname__", wrapper.__name__)
    wrapper._trace_apply = True  # type: ignore[attr-defined] — idempotency marker
    return wrapper


_MISSING = object()


def _resolve_and_cache(
    *,
    fi_api: str,
    axes: dict[str, int],
    index: Index,
    config: TraceApplyConfig,
    cache: dict[frozenset[tuple[str, int]], tuple[Callable, str] | None],
    cache_lock: Lock,
    axes_key: frozenset[tuple[str, int]],
) -> tuple[Callable, str] | None:
    sm = current_sm()
    cand = lookup(index, fi_api, axes, sm, config)
    if _DEBUG_AXES:
        _log.info(
            "Trace Apply [debug]: resolve %s sm=%s axes=%s -> %s",
            fi_api,
            sm,
            dict(sorted(axes.items())),
            "HIT(" + cand.solution.name + ")" if cand else "miss",
        )
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
            fi_api,
            e,
        )
        with cache_lock:
            cache[axes_key] = None
        return None
    entry = (fn, cand.solution.author)
    with cache_lock:
        cache[axes_key] = entry
    if fi_api not in _logged_dispatch:
        _logged_dispatch.add(fi_api)
        _log.info(
            "Trace Apply: applying solution '%s' (author=%s, %.4f ms) for %s axes=%s",
            cand.solution.name,
            cand.solution.author,
            cand.record.performance.latency_ms if cand.record.performance else float("nan"),
            fi_api,
            dict(sorted(axes.items())),
        )
    return entry


__all__ = [
    "bump_stat",
    "make_wrapper",
    "reset_loaded",
    "reset_stats",
    "stats_snapshot",
    "author_stats_snapshot",
]
