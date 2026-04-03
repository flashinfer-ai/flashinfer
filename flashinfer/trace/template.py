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
TraceTemplate and associated classes for the fi_trace system.

Design
------
A :class:`TraceTemplate` describes the schema of a FlashInfer operation
independently from any specific Python function.  Templates live in
``flashinfer/trace/templates/`` and are referenced by the
``@flashinfer_api(trace=<template>)`` decorator.

Axis extraction is **automatic**: the extraction logic is derived from the
``dim_names`` of the ``Tensor`` inputs — no lambda functions required.

Example::

    from flashinfer.trace.template import TraceTemplate, Var, Const, Tensor, Scalar

    rmsnorm_trace = TraceTemplate(
        op_type="rmsnorm",
        axes={"num_tokens": Var(), "hidden_size": Const()},
        inputs={
            "input":  Tensor(["num_tokens", "hidden_size"]),
            "weight": Tensor(["hidden_size"]),
            "eps":    Scalar("float32"),
        },
        outputs={"output": Tensor(["num_tokens", "hidden_size"])},
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch

# These are read lazily at each call so that the caller can set them after
# importing flashinfer (e.g. in scripts run with ``python -m``).


def _get_trace_dump_dir() -> Optional[str]:
    """Return the current FLASHINFER_TRACE_DUMP_DIR value (may be None)."""
    return os.environ.get("FLASHINFER_TRACE_DUMP_DIR")


def _is_trace_dump_enabled() -> bool:
    """Return True if auto-dump is currently enabled via FLASHINFER_TRACE_DUMP."""
    return os.environ.get("FLASHINFER_TRACE_DUMP", "0") not in ("0", "")


# Keep these module-level names for backwards compatibility with any code that
# imports them directly; they reflect the value at module-load time and are
# NOT updated if the env var changes later.
_TRACE_DUMP_DIR: Optional[str] = os.environ.get("FLASHINFER_TRACE_DUMP_DIR")
_TRACE_DUMP_ENABLED: bool = _is_trace_dump_enabled()

# In-memory deduplication: names of traces already written this process.
_DUMPED_NAMES: set = set()

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: Dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
    torch.uint8: "uint8",
}


def _dtype_str(dtype: torch.dtype) -> str:
    return _DTYPE_MAP.get(dtype, str(dtype).replace("torch.", ""))


def _get_tensor(
    kwargs: Dict[str, Any],
    param: str,
    tuple_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    val = kwargs.get(param)
    if val is None:
        return None
    if tuple_idx is not None:
        if isinstance(val, (tuple, list)) and len(val) > tuple_idx:
            val = val[tuple_idx]
        else:
            return None
    return val if isinstance(val, torch.Tensor) else None


# ---------------------------------------------------------------------------
# Axis markers
# ---------------------------------------------------------------------------


class Var:
    """Runtime-variable axis (e.g., ``batch_size``, ``seq_len``)."""

    def __init__(self, description: str = "") -> None:
        self.description = description


class Const:
    """Compile-time-constant axis (e.g., ``hidden_size``, ``num_heads``).

    Parameters
    ----------
    description:
        Human-readable description included in the JSON.
    abbrev:
        Short prefix used in the auto-generated file name.

        * ``None`` (default) — use the axis name as-is (backwards compatible).
        * ``""`` — omit this axis from the file name entirely.
        * Any other string — use that as the prefix, e.g. ``"h"`` produces
          ``h32`` for ``num_qo_heads=32``.
    """

    def __init__(self, description: str = "", abbrev: Optional[str] = None) -> None:
        self.description = description
        self.abbrev = abbrev


# ---------------------------------------------------------------------------
# Input / Output descriptors
# ---------------------------------------------------------------------------


class Tensor:
    """Descriptor for a tensor input or output.

    Parameters
    ----------
    dim_names:
        Ordered list of axis names for each tensor dimension.
    param:
        Python parameter name to look up in ``kwargs``.  Defaults to the
        key name in the ``inputs``/``outputs`` dict.
    tuple_idx:
        When the parameter is a tuple (e.g. ``paged_kv_cache=(k, v)``),
        the index into that tuple.
    dtype:
        For *outputs*: explicit dtype string such as ``"float32"``.
        For *inputs*: ignored — dtype is read from the actual tensor.
    dtype_from:
        For *outputs*: name of an input ``param`` whose dtype to copy.
        Takes precedence over ``dtype`` when both are set.
    optional:
        Whether the tensor may be absent.
    description:
        Human-readable description (included in the JSON).
    """

    def __init__(
        self,
        dim_names: List[str],
        *,
        param: Optional[str] = None,
        tuple_idx: Optional[int] = None,
        dtype: Optional[str] = None,
        dtype_from: Optional[str] = None,
        optional: bool = False,
        description: str = "",
    ) -> None:
        self.dim_names = dim_names
        self.param = param
        self.tuple_idx = tuple_idx
        self.dtype = dtype
        self.dtype_from = dtype_from
        self.optional = optional
        self.description = description


class Scalar:
    """Descriptor for a scalar (non-tensor) input.

    Parameters
    ----------
    dtype:
        Fixed dtype string (e.g. ``"float32"``).
    param:
        Python parameter name. Defaults to the key name in the dict.
    optional:
        Whether the scalar may be absent.
    description:
        Human-readable description.
    """

    def __init__(
        self,
        dtype: str = "float32",
        *,
        param: Optional[str] = None,
        optional: bool = False,
        description: str = "",
    ) -> None:
        self.dtype = dtype
        self.param = param
        self.optional = optional
        self.description = description


# ---------------------------------------------------------------------------
# TraceTemplate
# ---------------------------------------------------------------------------


class TraceTemplate:
    """Complete schema for generating a flashinfer-bench definition JSON.

    Parameters
    ----------
    op_type:
        Operation type string (e.g. ``"rmsnorm"``, ``"gqa_paged"``).
    name_prefix:
        Short, human-readable prefix used in the generated file name and the
        ``name`` field of the JSON.  When *None* (default) the prefix falls
        back to ``op_type``.  Set this explicitly when two templates share the
        same ``op_type`` and would otherwise produce identical file names
        (e.g. ``"gqa_paged_decode"`` vs ``"gqa_paged_prefill"`` both have
        ``op_type="gqa_paged"``).
    axes:
        Ordered ``dict`` of ``axis_name → Var() | Const()``.
    inputs:
        Ordered ``dict`` of ``json_name → Tensor | Scalar``.
    outputs:
        Ordered ``dict`` of ``json_name → Tensor | Scalar``.
    reference:
        Optional Python callable that implements the reference computation.
    constraints:
        Optional list of Python-expression strings (flashinfer-bench schema).
    tags:
        Additional tags (beyond the mandatory ``fi_api:...`` tag).
    description:
        Description field for the output JSON.
    """

    def __init__(
        self,
        op_type: str,
        axes: Dict[str, Union[Var, Const]],
        inputs: Dict[str, Union[Tensor, Scalar]],
        outputs: Dict[str, Union[Tensor, Scalar]],
        *,
        name_prefix: Optional[str] = None,
        reference: Optional[Callable] = None,
        constraints: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        self.op_type = op_type
        self.name_prefix = name_prefix
        self.axes = axes
        self.inputs = inputs
        self.outputs = outputs
        self.reference = reference
        self.constraints = constraints or []
        self.tags = tags or []
        self.description = description

    # ------------------------------------------------------------------
    # Axis extraction (automatic)
    # ------------------------------------------------------------------

    def _build_axis_extractors(
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Optional[int]]]:
        """Build per-axis extraction callables from tensor dim_names.

        For each axis in ``self.axes``, scan all ``Tensor`` inputs to find
        which tensor contains that axis and at which dimension index.  The
        resulting callable reads ``kwargs[param][tuple_idx].shape[dim_idx]``
        at call time.
        """
        extractors: Dict[str, Callable[[Dict[str, Any]], Optional[int]]] = {}
        for axis_name in self.axes:
            # Strategy 1: find the first Tensor input whose dim_names mention
            # this axis and read the corresponding shape dimension.
            for json_key, descriptor in self.inputs.items():
                if not isinstance(descriptor, Tensor):
                    continue
                if axis_name not in descriptor.dim_names:
                    continue
                param = descriptor.param if descriptor.param is not None else json_key
                tidx = descriptor.tuple_idx
                dim_idx = descriptor.dim_names.index(axis_name)

                def _make_extractor(
                    p: str, ti: Optional[int], di: int
                ) -> Callable[[Dict[str, Any]], Optional[int]]:
                    def extractor(kw: Dict[str, Any]) -> Optional[int]:
                        t = _get_tensor(kw, p, ti)
                        if t is None or di >= t.ndim:
                            return None
                        return int(t.shape[di])

                    return extractor

                extractors[axis_name] = _make_extractor(param, tidx, dim_idx)
                break  # Use first match only.

            if axis_name in extractors:
                continue

            # Strategy 2: fall back to reading the axis value directly from a
            # scalar kwarg whose name matches the axis name.  This handles
            # integer arguments like ``top_k``, ``n_group``, ``topk_group``.
            def _make_scalar_extractor(
                name: str,
            ) -> Callable[[Dict[str, Any]], Optional[int]]:
                def extractor(kw: Dict[str, Any]) -> Optional[int]:
                    val = kw.get(name)
                    if val is None:
                        return None
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        return None

                return extractor

            extractors[axis_name] = _make_scalar_extractor(axis_name)

        return extractors

    # ------------------------------------------------------------------
    # fi_trace callable factory
    # ------------------------------------------------------------------

    def build_fi_trace_fn(self, fi_api: str) -> Callable[..., Dict[str, Any]]:
        """Return a ``fi_trace(save_dir=None, **kwargs)`` callable.

        Parameters
        ----------
        fi_api:
            Fully qualified Python name of the decorated function
            (e.g. ``"flashinfer.norm.rmsnorm"``).
        """
        axis_extractors = self._build_axis_extractors()
        template = self  # capture in closure

        def fi_trace(
            save_dir: Optional[Union[str, Path]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            # ── 1. Extract axis values ─────────────────────────────────────
            axis_values: Dict[str, int] = {}
            for axis_name, extractor in axis_extractors.items():
                try:
                    val = extractor(kwargs)
                    if val is not None:
                        axis_values[axis_name] = val
                except Exception:
                    pass

            # ── 3. Build "axes" section ────────────────────────────────────
            axes_json: Dict[str, Any] = {}
            for axis_name, marker in template.axes.items():
                is_var = isinstance(marker, Var)
                entry: Dict[str, Any] = {"type": "var" if is_var else "const"}
                if not is_var and axis_name in axis_values:
                    entry["value"] = axis_values[axis_name]
                if marker.description:
                    entry["description"] = marker.description
                axes_json[axis_name] = entry

            # ── 4. Build "inputs" section ──────────────────────────────────
            inputs_json: Dict[str, Any] = {}
            for json_key, descriptor in template.inputs.items():
                if isinstance(descriptor, Scalar):
                    entry = {"shape": None, "dtype": descriptor.dtype}
                else:
                    param = (
                        descriptor.param if descriptor.param is not None else json_key
                    )
                    t = _get_tensor(kwargs, param, descriptor.tuple_idx)
                    entry = {
                        "shape": descriptor.dim_names,
                        "dtype": _dtype_str(t.dtype) if t is not None else "unknown",
                    }
                if descriptor.optional:
                    entry["optional"] = True
                if descriptor.description:
                    entry["description"] = descriptor.description
                inputs_json[json_key] = entry

            # ── 5. Build "outputs" section ─────────────────────────────────
            outputs_json: Dict[str, Any] = {}
            for json_key, descriptor in template.outputs.items():
                if isinstance(descriptor, Scalar):
                    entry = {"shape": None, "dtype": descriptor.dtype}
                else:
                    # Resolve dtype for outputs
                    dtype: str
                    if descriptor.dtype_from is not None:
                        ref_param = descriptor.dtype_from
                        ref_t = _get_tensor(kwargs, ref_param)
                        dtype = (
                            _dtype_str(ref_t.dtype) if ref_t is not None else "unknown"
                        )
                    elif descriptor.dtype is not None:
                        dtype = descriptor.dtype
                    else:
                        # Auto-infer: find first input tensor with overlapping dims
                        dtype = "unknown"
                        for in_key, in_desc in template.inputs.items():
                            if not isinstance(in_desc, Tensor):
                                continue
                            if any(
                                d in in_desc.dim_names for d in descriptor.dim_names
                            ):
                                in_param = (
                                    in_desc.param
                                    if in_desc.param is not None
                                    else in_key
                                )
                                ref_t = _get_tensor(kwargs, in_param, in_desc.tuple_idx)
                                if ref_t is not None:
                                    dtype = _dtype_str(ref_t.dtype)
                                    break
                    entry = {"shape": descriptor.dim_names, "dtype": dtype}
                if descriptor.optional:
                    entry["optional"] = True
                if descriptor.description:
                    entry["description"] = descriptor.description
                outputs_json[json_key] = entry

            # ── 6. Resolve name (explicit override or auto-generate) ──────
            if name is None:
                # Use name_prefix from the template when set (preferred: short,
                # semantic names like "gqa_paged_decode", "gdn_mtp").
                # Fall back to op_type otherwise.
                prefix = (
                    template.name_prefix
                    if template.name_prefix is not None
                    else template.op_type
                )
                const_parts = []
                for n, marker in template.axes.items():
                    if not isinstance(marker, Const) or n not in axis_values:
                        continue
                    # abbrev="" → omit from name; abbrev=None → use axis name
                    pfx = marker.abbrev if marker.abbrev is not None else n
                    if pfx == "":
                        continue
                    const_parts.append(f"{pfx}{axis_values[n]}")
                name = prefix + ("_" + "_".join(const_parts) if const_parts else "")

            # ── 7. Assemble definition ─────────────────────────────────────
            all_tags = [f"fi_api:{fi_api}"] + template.tags
            result: Dict[str, Any] = {
                "name": name,
                "description": template.description,
                "op_type": template.op_type,
                "tags": all_tags,
                "axes": axes_json,
            }
            if template.constraints:
                result["constraints"] = template.constraints
            result["inputs"] = inputs_json
            result["outputs"] = outputs_json
            if template.reference is not None:
                try:
                    import inspect  # noqa: PLC0415

                    result["reference"] = inspect.getsource(template.reference)
                except (OSError, TypeError):
                    pass

            # ── 8. Write JSON file if requested ───────────────────────────
            # Deduplication only applies to auto-dump (save_dir=None): once a
            # named trace has been auto-dumped this process, skip re-writing it.
            # Explicit save_dir= calls always write (no dedup).
            effective_dir = save_dir if save_dir is not None else _get_trace_dump_dir()
            _is_auto_dump = save_dir is None
            if effective_dir is not None and (
                not _is_auto_dump or name not in _DUMPED_NAMES
            ):
                out_dir = Path(effective_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{name}.json"
                out_path.write_text(json.dumps(result, indent=2))
                if _is_auto_dump:
                    _DUMPED_NAMES.add(name)

            return result

        fi_trace.__doc__ = (
            f"Generate a flashinfer-bench definition JSON for op_type='{self.op_type}'.\n\n"
            f"FlashInfer API: {fi_api}\n"
        )
        return fi_trace
