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

"""
fi_trace: Generate `flashinfer-bench <https://github.com/flashinfer-ai/flashinfer-bench>`_
compatible definition JSON for FlashInfer APIs.

Every ``@flashinfer_api(trace=<template>)``-decorated function supports two
usage modes:

Auto-dump (recommended)
-----------------------
Set environment variables **before** importing flashinfer, then run your
workload normally.  No explicit ``fi_trace`` call is needed.

.. code-block:: bash

    FLASHINFER_TRACE_DUMP=1 \\
    FLASHINFER_TRACE_DUMP_DIR=./fi_trace_out \\
    python my_script.py

Every decorated function writes a ``<name>.json`` file on its **first** call
for each unique set of const-axis values (e.g. head dimensions, vocab size).
Subsequent calls with the same shape are deduplicated — the file is written
only once per process.  The output directory is created automatically.

Explicit call (for selective or programmatic use)
-------------------------------------------------
Each decorated function also has a ``.fi_trace(**kwargs)`` attribute.  Pass
the same tensor arguments you would pass to the real function; fi_trace
introspects their shapes / dtypes and returns the definition dict.

.. code-block:: python

    import flashinfer, torch

    hidden = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")

    defn = flashinfer.rmsnorm.fi_trace(input=hidden, weight=weight)

    import json
    print(json.dumps(defn, indent=2))

For class-method APIs use the unbound (class-level) form, or the module-level
helper:

.. code-block:: python

    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.fi_trace import fi_trace

    defn = BatchDecodeWithPagedKVCacheWrapper.run.fi_trace(
        q=q_tensor, paged_kv_cache=(k_cache, v_cache)
    )
    # or with a live instance:
    defn = fi_trace(wrapper.run, q=q_tensor, paged_kv_cache=(k, v))

Both modes support an optional ``save_dir`` argument / env-var to control
where the JSON file is written.  Explicit ``save_dir`` always writes; the
auto-dump path deduplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Legacy registry — kept for backwards compatibility.
# New code should use @flashinfer_api(trace=TraceTemplate(...)) instead.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Any] = {}


def register_fi_trace(qualname: str, spec: Any) -> None:
    """Register a legacy FiTraceSpec for the function with the given qualname.

    .. deprecated::
        Use ``@flashinfer_api(trace=TraceTemplate(...))`` instead.
    """
    _REGISTRY[qualname] = spec


def build_fi_trace_fn(spec: Any) -> Callable[..., Dict[str, Any]]:
    """Build a fi_trace callable from a legacy FiTraceSpec.

    .. deprecated::
        Use ``TraceTemplate.build_fi_trace_fn`` instead.
    """
    # Import the old implementation from the trace package for backwards compat.
    from .trace.template import (  # noqa: PLC0415,F401
        Const,
        Scalar,
        Tensor,
        TraceTemplate,
        Var,
    )
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    import torch  # noqa: PLC0415

    _DTYPE_MAP = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.int8: "int8",
        torch.uint8: "uint8",
    }
    try:
        _DTYPE_MAP[torch.float8_e4m3fn] = "float8_e4m3fn"
        _DTYPE_MAP[torch.float8_e5m2] = "float8_e5m2"
    except AttributeError:
        pass

    def _dtype_str(dtype):
        return _DTYPE_MAP.get(dtype, str(dtype).replace("torch.", ""))

    def _get_tensor(kwargs, param, tuple_idx=None):
        val = kwargs.get(param)
        if val is None:
            return None
        if tuple_idx is not None:
            if isinstance(val, (tuple, list)) and len(val) > tuple_idx:
                val = val[tuple_idx]
            else:
                return None
        return val if isinstance(val, torch.Tensor) else None

    def fi_trace(save_dir=None, **kwargs):
        axis_values: Dict[str, int] = {}
        for axis_name, axis_def in spec.axes.items():
            if axis_def.extract is not None:
                try:
                    val = axis_def.extract(kwargs)
                    if val is not None:
                        axis_values[axis_name] = int(val)
                except Exception:
                    pass

        axes_json: Dict[str, Any] = {}
        for axis_name, axis_def in spec.axes.items():
            entry: Dict[str, Any] = {"type": "var" if axis_def.is_var else "const"}
            if not axis_def.is_var and axis_name in axis_values:
                entry["value"] = axis_values[axis_name]
            if axis_def.description:
                entry["description"] = axis_def.description
            axes_json[axis_name] = entry

        inputs_json: Dict[str, Any] = {}
        for inp in spec.inputs:
            if inp.is_scalar:
                val = kwargs.get(inp.func_param)
                dtype = (
                    _dtype_str(val.dtype)
                    if isinstance(val, torch.Tensor)
                    else "float32"
                )
                entry = {"shape": None, "dtype": dtype}
            else:
                t = _get_tensor(kwargs, inp.func_param, inp.tuple_idx)
                entry = {
                    "shape": inp.dim_names,
                    "dtype": _dtype_str(t.dtype) if t is not None else "unknown",
                }
            if inp.optional:
                entry["optional"] = True
            if inp.description:
                entry["description"] = inp.description
            inputs_json[inp.json_name] = entry

        outputs_json: Dict[str, Any] = {}
        for out in spec.outputs:
            dtype = out.dtype
            if dtype.startswith("from_input:"):
                src_param = dtype[len("from_input:") :]
                t = _get_tensor(kwargs, src_param)
                dtype = _dtype_str(t.dtype) if t is not None else "unknown"
            entry = {"shape": out.dim_names, "dtype": dtype}
            if out.description:
                entry["description"] = out.description
            outputs_json[out.json_name] = entry

        const_parts = [
            f"{n}{v}"
            for n, a in spec.axes.items()
            if not a.is_var and n in axis_values
            for v in (axis_values[n],)
        ]
        name = spec.op_type + ("_" + "_".join(const_parts) if const_parts else "")

        tags = [f"fi_api:{spec.fi_api}"] + spec.extra_tags
        result: Dict[str, Any] = {
            "name": name,
            "description": spec.description,
            "op_type": spec.op_type,
            "tags": tags,
            "axes": axes_json,
        }
        if spec.constraints:
            result["constraints"] = spec.constraints
        result["inputs"] = inputs_json
        result["outputs"] = outputs_json

        _trace_dir = os.environ.get("FLASHINFER_TRACE_DUMP_DIR")
        effective_dir = save_dir if save_dir is not None else _trace_dir
        if effective_dir is not None:
            out_dir = Path(effective_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json.dumps(result, indent=2))

        return result

    return fi_trace


# ---------------------------------------------------------------------------
# Public helper: fi_trace(func_or_method, **kwargs)
# ---------------------------------------------------------------------------


def fi_trace(
    func_or_method: Callable,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a flashinfer-bench definition JSON for any FlashInfer API call.

    Parameters
    ----------
    func_or_method:
        A ``@flashinfer_api``-decorated function or (bound) method.
    save_dir:
        Directory where the JSON definition file should be written.
        Falls back to ``FLASHINFER_TRACE_DUMP_DIR`` env-var when *None*.
    **kwargs:
        The same tensor arguments you would pass to the real API.

    Returns
    -------
    dict
        A flashinfer-bench compatible definition dictionary.

    Examples
    --------
    Standalone function::

        defn = fi_trace(flashinfer.norm.rmsnorm, input=hidden, weight=weight)

    Bound method (instance.run)::

        defn = fi_trace(wrapper.run, q=q_tensor, paged_kv_cache=(k, v))

    Class-level (unbound)::

        defn = fi_trace(
            flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run,
            q=q_tensor, paged_kv_cache=(k, v),
        )
    """
    actual_func = getattr(func_or_method, "__func__", func_or_method)
    trace_fn = getattr(actual_func, "fi_trace", None)
    if trace_fn is None:
        qualname = getattr(actual_func, "__qualname__", repr(actual_func))
        raise ValueError(
            f"No fi_trace spec is registered for '{qualname}'. "
            "Only @flashinfer_api(trace=...)-decorated functions support fi_trace."
        )
    return trace_fn(save_dir=save_dir, **kwargs)
