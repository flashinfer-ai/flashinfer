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

# OUTPUT_DESTS maps, per fi_api, each Definition output -> the API parameter
# holding its destination buffer (an input tensor for in-place APIs; ``out=`` /
# ``lse`` for value-returning ones). OUTPUT_ACTIVATION gates optional outputs by
# a runtime flag (``lse`` is active only when ``return_lse=True``) so the
# returned arity matches the API.
OUTPUT_DESTS: dict[str, dict[str, str]] = {
    "flashinfer.norm.rmsnorm": {"output": "out"},
    "flashinfer.norm.fused_add_rmsnorm": {"output": "input", "residual": "residual"},
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run": {"output": "out", "lse": "lse"},
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run": {"output": "out", "lse": "lse"},
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run": {"output": "out", "lse": "lse"},
    "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run": {"output": "out", "lse": "lse"},
}

OUTPUT_ACTIVATION: dict[str, dict[str, str]] = {
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run": {"lse": "return_lse"},
    "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run": {"lse": "return_lse"},
    "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run": {"lse": "return_lse"},
    "flashinfer.mla._core.BatchMLAPagedAttentionWrapper.run": {"lse": "return_lse"},
}


def output_dests(fi_api: str) -> dict[str, str]:
    return OUTPUT_DESTS.get(fi_api, {})


def output_activation(fi_api: str) -> dict[str, str]:
    return OUTPUT_ACTIVATION.get(fi_api, {})


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


def extract_input_dtypes(template: Any, namespace: dict[str, Any]) -> frozenset[tuple[str, str]]:
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


def _alloc_like(template: Any, json_key: str, namespace: dict[str, Any]) -> torch.Tensor:
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
    activation: dict[str, str] | None = None,
    positional: bool = False,
) -> Any:
    """Invoke the solution and reconcile its outputs with the API convention.

    - value-returning: copy each returned output into its destination buffer
      when one exists (``out=`` / in-place input).
    - DPS: locate/allocate output buffers for the *active* outputs, pass them,
      let the solution write them.
    - ``activation`` gates optional outputs by a runtime flag (``lse`` active
      only when ``return_lse=True``) so the returned arity matches the API.
    - ``positional``: C++/CUDA (TVM-FFI) solutions take positional args
      ``(inputs..., active outputs...)``; Python-family take keyword.
    Returns ``None`` for in-place APIs, else the (single or tuple) output value.
    """
    activation = activation or {}
    out_keys = _tensor_output_keys(template)

    def is_active(json_key: str) -> bool:
        flag = activation.get(json_key)
        return True if flag is None else bool(namespace.get(flag))

    active_keys = [jk for jk in out_keys if is_active(jk)]

    def dest_buf(json_key: str):
        param = dests.get(json_key)
        if not param:
            return None
        b = namespace.get(param)
        return b if isinstance(b, torch.Tensor) else None

    if dps:
        out_bufs: dict[str, Any] = {}
        for jk in active_keys:
            buf = dest_buf(jk)
            out_bufs[jk] = buf if buf is not None else _alloc_like(template, jk, namespace)
        if positional:
            fn(*ordered_input_values(template, namespace), *[out_bufs[jk] for jk in active_keys])
        else:
            fn(**build_candidate_kwargs(template, namespace), **out_bufs)
        produced_active: list[Any] = [out_bufs[jk] for jk in active_keys]
    else:
        if positional:
            ret = fn(*ordered_input_values(template, namespace))
        else:
            ret = fn(**build_candidate_kwargs(template, namespace))
        produced = list(ret) if isinstance(ret, (tuple, list)) else [ret]
        # Map solution outputs to template outputs by position; copy into any
        # destination buffer; collect the active subset for the return value.
        produced_active = []
        for i, jk in enumerate(out_keys):
            val = produced[i] if i < len(produced) else None
            if val is not None:
                buf = dest_buf(jk)
                if buf is not None and isinstance(val, torch.Tensor) and val.data_ptr() != buf.data_ptr():
                    buf.copy_(val)
                    val = buf
            if jk in active_keys:
                produced_active.append(val)
        if any(v is None for v in produced_active):
            missing = [jk for jk, v in zip(active_keys, produced_active) if v is None]
            raise RuntimeError(
                f"Trace Apply: solution did not return required active output(s) {missing}."
            )

    if is_inplace:
        return None
    if not produced_active:
        return None
    if len(produced_active) == 1:
        return produced_active[0]
    return tuple(produced_active)


__all__ = [
    "OUTPUT_DESTS",
    "OUTPUT_ACTIVATION",
    "output_dests",
    "output_activation",
    "build_candidate_kwargs",
    "ordered_input_values",
    "extract_input_dtypes",
    "adapt_and_call",
]
