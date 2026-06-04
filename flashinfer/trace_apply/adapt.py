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

"""Input resolution and output adaptation between a Solution and a real API.

A solution implements the Definition's inputs/outputs, but real flashinfer APIs
differ in how outputs are delivered: value-returning (optionally into an ``out=``
buffer) vs in-place / destination-passing (returns ``None``, mutates caller
buffers — e.g. ``fused_add_rmsnorm``). The helpers here reconcile the two so the
engine sees the original API's return/write convention.

Calling convention:
* Python-family solutions are called by keyword (Definition input names).
* C++/CUDA (TVM-FFI) solutions are called positionally.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

# Output binding is read straight from the TraceTemplate's output descriptors —
# no hardcoded per-API tables. ``param`` is the API parameter an output is
# written into (an ``out=`` buffer, or an input buffer for in-place APIs).
#
# Data-dependent behavior (e.g. an ``lse`` output produced only when
# ``return_lse=True``) is NOT modeled here: the runtime flag is passed to the
# solution as an ordinary input and the solution returns the matching arity,
# exactly like the real kernel. adapt_and_call is a transparent pass-through.


def output_dests(template: Any) -> dict[str, str]:
    """``{output_name: destination API param}`` from the template's outputs.
    Empty for pure value-returning APIs (no output declares a ``param``)."""
    out: dict[str, str] = {}
    for json_key, desc in template.outputs.items():
        param = getattr(desc, "param", None)
        if param:
            out[json_key] = param
    return out


# ---------------------------------------------------------------------------
# Input resolution (template inputs -> solution call args)
# ---------------------------------------------------------------------------


def _resolve_input(desc: Any, namespace: dict[str, Any]) -> Any:
    """Resolve one template input descriptor against the call namespace,
    handling tuple params (e.g. ``paged_kv_cache=(k, v)``)."""
    param = getattr(desc, "param", None) or None
    # param may be None when the json_key itself is the param; caller passes the
    # json_key fallback via the loop below.
    val = namespace.get(param) if param is not None else None
    tuple_idx = getattr(desc, "tuple_idx", None)
    if tuple_idx is not None and val is not None:
        if isinstance(val, (tuple, list)) and len(val) > tuple_idx:
            val = val[tuple_idx]
        else:
            val = None
    return val


def build_candidate_kwargs(template: Any, namespace: dict[str, Any]) -> dict[str, Any]:
    """Map template inputs to a ``{json_key: value}`` kwargs dict for the solution.

    flashinfer-bench solutions are written against the Definition's input names,
    so a Python-family candidate is invoked as ``solution(**build_candidate_kwargs)``.
    """
    out: dict[str, Any] = {}
    for json_key, desc in template.inputs.items():
        param = getattr(desc, "param", None) or json_key
        val = namespace.get(param)
        if val is None:
            continue
        tuple_idx = getattr(desc, "tuple_idx", None)
        if tuple_idx is not None:
            if isinstance(val, (tuple, list)) and len(val) > tuple_idx:
                val = val[tuple_idx]
            else:
                continue
        out[json_key] = val
    return out


def ordered_input_values(template: Any, namespace: dict[str, Any]) -> list[Any]:
    """Input values in the template's declared order, ``None`` for absent inputs
    (positions preserved). Used to call C++/CUDA (TVM-FFI) solutions, which take
    positional, fixed-arity arguments."""
    out: list[Any] = []
    for json_key, desc in template.inputs.items():
        param = getattr(desc, "param", None) or json_key
        val = namespace.get(param)
        tuple_idx = getattr(desc, "tuple_idx", None)
        if tuple_idx is not None and val is not None:
            val = (
                val[tuple_idx]
                if isinstance(val, (tuple, list)) and len(val) > tuple_idx
                else None
            )
        out.append(val)
    return out


def extract_input_dtypes(
    template: Any, namespace: dict[str, Any]
) -> frozenset[tuple[str, str]]:
    """``{(input_name, dtype_str)}`` over **required tensor** inputs present in
    the call. Mirrors ``Definition.input_dtypes`` so the routing key matches.

    Uses trace's own ``_dtype_str`` (imported) so the runtime dtype string is
    identical to the one stored in the definition by construction.
    """
    from flashinfer.trace.template import Scalar, _dtype_str  # noqa: PLC0415

    out: dict[str, str] = {}
    for json_key, desc in template.inputs.items():
        if isinstance(desc, Scalar):
            continue
        if getattr(desc, "optional", False):
            continue
        param = getattr(desc, "param", None) or json_key
        val = namespace.get(param)
        tuple_idx = getattr(desc, "tuple_idx", None)
        if tuple_idx is not None and val is not None:
            val = (
                val[tuple_idx]
                if isinstance(val, (tuple, list)) and len(val) > tuple_idx
                else None
            )
        if isinstance(val, torch.Tensor):
            out[json_key] = _dtype_str(val.dtype)
    return frozenset(out.items())


# ---------------------------------------------------------------------------
# Output adaptation
# ---------------------------------------------------------------------------


def _tensor_output_keys(template: Any) -> list[str]:
    from flashinfer.trace.template import Tensor  # noqa: PLC0415

    return [k for k, d in template.outputs.items() if isinstance(d, Tensor)]


def _alloc_like(
    template: Any, json_key: str, namespace: dict[str, Any]
) -> torch.Tensor:
    """Best-effort output buffer allocation for a DPS solution when the caller
    provided none. Uses the output's ``dtype_from`` input."""
    desc = template.outputs[json_key]
    src = getattr(desc, "dtype_from", None)
    if src and isinstance(namespace.get(src), torch.Tensor):
        return torch.empty_like(namespace[src])
    raise RuntimeError(
        f"Trace Apply: cannot allocate output buffer for {json_key!r} "
        f"(DPS solution, no caller buffer, no usable dtype_from)."
    )


def adapt_and_call(
    *,
    template: Any,
    fn: Callable,
    namespace: dict[str, Any],
    dps: bool,
    is_inplace: bool,
    dests: dict[str, str],
    positional: bool = False,
) -> Any:
    """Invoke the solution and reconcile its outputs with the API convention.

    Transparent pass-through: any data-dependent behavior (e.g. ``return_lse``)
    is the solution's responsibility — it receives the flag as an input and
    returns the matching arity, just like the real kernel. We only bind outputs
    to their destination buffers and return the same arity the solution produced.

    - value-returning: ``fn`` returns one value or a tuple; each is copied into
      its destination buffer (``out=`` / in-place input) when one exists, and the
      same arity is returned.
    - DPS: pass output buffers (caller-provided, else allocated) for every
      non-optional output plus any optional output the caller supplied a buffer
      for; the solution writes them; return them.
    - ``positional``: C++/CUDA (TVM-FFI) solutions take positional args; Python
      take keyword.
    Returns ``None`` for in-place APIs, else the (single or tuple) output value.
    """
    out_keys = _tensor_output_keys(template)

    def dest_buf(json_key: str):
        param = dests.get(json_key)
        if not param:
            return None
        b = namespace.get(param)
        return b if isinstance(b, torch.Tensor) else None

    if dps:
        # Materialize non-optional outputs always; optional outputs only when the
        # caller supplied a destination buffer for them.
        out_bufs: dict[str, Any] = {}
        for jk in out_keys:
            buf = dest_buf(jk)
            if buf is None:
                if getattr(template.outputs[jk], "optional", False):
                    continue
                buf = _alloc_like(template, jk, namespace)
            out_bufs[jk] = buf
        if positional:
            fn(*ordered_input_values(template, namespace), *out_bufs.values())
        else:
            fn(**build_candidate_kwargs(template, namespace), **out_bufs)
        produced = list(out_bufs.values())
    else:
        if positional:
            ret = fn(*ordered_input_values(template, namespace))
        else:
            ret = fn(**build_candidate_kwargs(template, namespace))
        values = list(ret) if isinstance(ret, (tuple, list)) else [ret]
        # Copy each returned value into its destination buffer (if any); keep the
        # solution's own arity as the returned arity.
        produced = []
        for i, val in enumerate(values):
            if i < len(out_keys) and isinstance(val, torch.Tensor):
                buf = dest_buf(out_keys[i])
                if buf is not None and val.data_ptr() != buf.data_ptr():
                    buf.copy_(val)
                    val = buf
            produced.append(val)

    if is_inplace:
        return None
    if not produced:
        return None
    if len(produced) == 1:
        return produced[0]
    return tuple(produced)


__all__ = [
    "adapt_and_call",
    "build_candidate_kwargs",
    "extract_input_dtypes",
    "ordered_input_values",
    "output_dests",
]
