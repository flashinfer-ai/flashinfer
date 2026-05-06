from __future__ import annotations

import inspect
from threading import local
from typing import Any, Callable

# Imported lazily inside functions to avoid a hard dependency at module load.
# from flashinfer.trace.template import Tensor, Scalar


def bind_namespace(original: Any, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind args/kwargs to ``original``'s signature → flat ``{param: value}``.

    The template descriptors look up values by Python parameter name, so we
    turn positional args into named ones first. ``original`` is the
    *undecorated* function from the trace registry (real signature).
    """
    try:
        sig = inspect.signature(original)
    except (TypeError, ValueError):
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


def extract_from_namespace(
    extractor_maps: list[dict[str, Callable]], namespace: dict[str, Any]
) -> dict[str, int]:
    """Run the template extractors over a (param-name → value) namespace and
    return the full concrete axis vector. Multiple templates merge: each axis is
    filled by the first that can resolve it.
    """
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


def build_candidate_kwargs(template: Any, namespace: dict[str, Any]) -> dict[str, Any]:
    """Map a template's inputs to a ``{json_key: value}`` kwargs dict for the
    candidate, resolving each input from ``namespace`` via its ``param`` /
    ``tuple_idx`` (handles tuple params like ``paged_kv_cache=(k, v)``).

    flashinfer-bench solutions are written against the Definition's input names,
    so the candidate is invoked as ``solution(**build_candidate_kwargs(...))``.
    """
    from flashinfer.trace.template import Scalar, Tensor  # noqa: PLC0415

    out: dict[str, Any] = {}
    for json_key, desc in template.inputs.items():
        param = getattr(desc, "param", None) or json_key
        val = namespace.get(param)
        if val is None:
            continue
        if isinstance(desc, Tensor) and desc.tuple_idx is not None:
            if isinstance(val, (tuple, list)) and len(val) > desc.tuple_idx:
                val = val[desc.tuple_idx]
            else:
                continue
        out[json_key] = val
    return out


# ---------------------------------------------------------------------------
# Plan/run state for stateful wrappers (BatchDecode/Prefill/MLA/Cascade ...)
# ---------------------------------------------------------------------------

_plan_state = local()


def stash_plan_kwargs(wrapper_id: int, kwargs: dict[str, Any]) -> None:
    """Record the bound kwargs seen at ``plan()`` time for this wrapper instance,
    so the matching ``run()`` call can recover plan-derived inputs/axes.
    """
    if not hasattr(_plan_state, "by_wrapper"):
        _plan_state.by_wrapper = {}
    _plan_state.by_wrapper[wrapper_id] = dict(kwargs)


def fetch_plan_kwargs(wrapper_id: int) -> dict[str, Any]:
    if not hasattr(_plan_state, "by_wrapper"):
        return {}
    return dict(_plan_state.by_wrapper.get(wrapper_id, {}))


__all__ = [
    "bind_namespace",
    "build_extractor_maps",
    "extract_from_namespace",
    "build_candidate_kwargs",
    "stash_plan_kwargs",
    "fetch_plan_kwargs",
]
