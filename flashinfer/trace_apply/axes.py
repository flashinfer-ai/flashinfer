from __future__ import annotations

import inspect
from threading import local
from typing import Any

import torch

from flashinfer.trace_apply.schema import Definition


def _resolve_call_kwargs(target: Any, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind args/kwargs to the target's signature and return a flat dict
    keyed by parameter name. We use this to look up tensor inputs by name.
    """
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        # Built-in / C function without a signature — fall back to kwargs only.
        return dict(kwargs)
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError:
        return dict(kwargs)
    return dict(bound.arguments)


def extract_axes(
    definition: Definition,
    target: Any,
    args: tuple,
    kwargs: dict[str, Any],
) -> dict[str, int]:
    """Return a complete `{axis_name: int}` dict for this call.

    Const axes come from `Definition.axes[name].value`. Var axes are read from
    the first input whose shape mentions them.
    """
    axes: dict[str, int] = dict(definition.const_axes())

    var_names = set(definition.var_axis_names())
    if not var_names:
        return axes

    bound = _resolve_call_kwargs(target, args, kwargs)

    # For each var axis, find an input that mentions it and read the
    # corresponding shape entry. Definitions are well-formed when produced via
    # `fi_trace`, so the same axis name appears at a stable position across
    # inputs that reference it.
    for input_name, spec in definition.inputs.items():
        if not spec.shape:
            continue
        tensor = bound.get(input_name)
        if not isinstance(tensor, torch.Tensor):
            continue
        if len(tensor.shape) != len(spec.shape):
            continue
        for pos, axis_name in enumerate(spec.shape):
            if axis_name in var_names and axis_name not in axes:
                axes[axis_name] = int(tensor.shape[pos])

    return axes


# ---------------------------------------------------------------------------
# Plan/run state for stateful wrappers (BatchDecodeWithPagedKVCacheWrapper et al.)
# ---------------------------------------------------------------------------

_plan_state = local()


def stash_plan_axes(wrapper_id: int, axes: dict[str, int]) -> None:
    if not hasattr(_plan_state, "by_wrapper"):
        _plan_state.by_wrapper = {}
    _plan_state.by_wrapper[wrapper_id] = dict(axes)


def fetch_plan_axes(wrapper_id: int) -> dict[str, int]:
    if not hasattr(_plan_state, "by_wrapper"):
        return {}
    return dict(_plan_state.by_wrapper.get(wrapper_id, {}))


__all__ = ["extract_axes", "stash_plan_axes", "fetch_plan_axes"]
