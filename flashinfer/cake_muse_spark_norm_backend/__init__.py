"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Backend of the Cake single-GPU norm kernels of the Muse Spark family as one
self-contained package (the host bodies, the ``cake_jit`` registry and loader
glue and the generated CUDA sources under ``csrc/``): the gated residual with
the next RMSNorm fused behind
``flashinfer.norm.gated_residual_next_rmsnorm(..., backend="cake")``
(:func:`cake_gated_residual_next_rmsnorm`; hidden size 8192), the gated
residual RMSNorm behind ``flashinfer.norm.gated_residual_rmsnorm(...,
backend="cake")`` (:func:`cake_gated_residual_rmsnorm`; hidden size 8192) and
the per-head RMSNorm times the sigmoid gate behind
``flashinfer.norm.rmsnorm_sigmoid_gate(..., backend="cake")``
(:func:`cake_rmsnorm_sigmoid_gate`; head dimension 64, in place); SM100 /
SM103.  The stable, ``@flashinfer_api``-decorated entry points and the
user-facing documentation live in :mod:`flashinfer.norm`; import it.

The two-norm kernels ship in three variants of their optional inputs:
``alpha`` a tensor or a scalar, ``post_weight`` present or not
(``alpha_tensor_pw``, ``alpha_scalar_nopw``, ``alpha_scalar_pw``).  A call with
a tensor ``alpha`` and no ``post_weight`` runs the ``alpha_tensor_pw`` programs
with a zero ``post_weight`` vector and ``post_gain_center = 1.0``: ``branch *
(0 + 1)`` is ``branch`` exactly, so the result is the one a dedicated program
would give.  The zero vector is allocated once per device on first use (never
inside CUDA graph capture: :func:`warm_up` prepares a device before a capture).

The gated residual RMSNorm ships per mode (``weight`` absent: ``mode0``;
``weight`` applied before the norm: ``mode1``) in a plain and a programmatic
dependent launch build (``_pdl``); the sigmoid gate ships one BF16 route whose
programs are selected on the head count and the token count together.  Each
route's windows may leave gaps where the kernel lane measured the FlashInfer
kernel ahead: the ``is_cake_*_supported`` predicates answer per call so that
the facade's ``"auto"`` keeps the CuTe DSL kernel there.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    HIDDEN_SIZE,
    OPS,
    VARIANTS,
    arch_for_capability,
    route_key,
)

GATED_RESIDUAL_NEXT_RMSNORM = "gated_residual_next_rmsnorm"
GATED_RESIDUAL_RMSNORM = "gated_residual_rmsnorm"
RMSNORM_SIGMOID_GATE = "rmsnorm_sigmoid_gate"
#: The two-norm kernels load every operand in 256-bit accesses: rows and vectors are 32-byte aligned.
ALIGNMENT_BYTES = 32
#: The gated residual RMSNorm and the sigmoid gate load in 128-bit accesses: 16-byte aligned storage.
ROW_ALIGNMENT_BYTES = 16
#: The head dimension of the sigmoid gate's rows.
HEAD_DIM = 64

_zero_post_weights: dict[int, torch.Tensor] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _arch(device: torch.device) -> Optional[str]:
    """The exported architecture of ``device``, or ``None`` for a device the package has no build for."""
    if device.type != "cuda":
        return None
    return arch_for_capability(tuple(torch.cuda.get_device_capability(device)))


def gated_residual_next_rmsnorm_variant(
    tensor_alpha: bool, has_post_weight: bool
) -> str:
    """The variant a call maps onto (``alpha_tensor_pw``, ``alpha_scalar_nopw``, ``alpha_scalar_pw``,
    ``alpha_tensor_nopw``)."""
    alpha = "alpha_tensor" if tensor_alpha else "alpha_scalar"
    return f"{alpha}_{'pw' if has_post_weight else 'nopw'}"


def gated_residual_next_rmsnorm_route(tensor_alpha: bool, has_post_weight: bool) -> str:
    """The route a call launches from: its variant's, except that a tensor ``alpha`` without ``post_weight``
    runs the ``alpha_tensor_pw`` route with a zero ``post_weight`` (module docstring)."""
    variant = gated_residual_next_rmsnorm_variant(tensor_alpha, has_post_weight)
    if variant == "alpha_tensor_nopw":
        variant = "alpha_tensor_pw"
    return route_key(GATED_RESIDUAL_NEXT_RMSNORM, variant)


def is_cake_gated_residual_next_rmsnorm_supported(
    device: torch.device,
    *,
    tensor_alpha: bool = True,
    has_post_weight: bool = True,
    rows: Optional[int] = None,
) -> bool:
    """Whether this package ships a program for ``device`` and the call's variant (tensor or scalar ``alpha``,
    with or without ``post_weight``) -- and, with ``rows``, one whose window admits that row count: a route's
    windows may leave gaps where the kernel lane measured the FlashInfer kernel ahead, and no program is shipped
    for those row counts.  The tensor contract is :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    route = gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    if not _loader.route_available(arch, route):
        return False
    if rows is None:
        return True
    return any(
        _loader.admits(_loader.MODULES[name].get("select_rule"), int(rows), arch)
        for name in _loader.route_modules(arch, route)
    )


def zero_post_weight(device: torch.device) -> torch.Tensor:
    """The zero BF16 ``post_weight`` vector of ``device`` (one per device, allocated on first use)."""
    index = _device_index(device)
    found = _zero_post_weights.get(index)
    if found is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cake_gated_residual_next_rmsnorm: the zero post_weight vector of device "
                f"cuda:{index} is allocated on first use, which cannot happen inside CUDA graph "
                "capture; call the function once (or warm_up) on this device before capturing"
            )
        found = torch.zeros(
            HIDDEN_SIZE, dtype=torch.bfloat16, device=torch.device("cuda", index)
        )
        _zero_post_weights[index] = found
    return found


def warm_up(
    device: torch.device, *, tensor_alpha: bool = True, has_post_weight: bool = True
) -> None:
    """Load the libraries of the call's route on ``device``, read their co-residency and allocate the zero
    ``post_weight`` vector where the call needs one, so that a later call can run inside CUDA graph capture."""
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_next_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    route = gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    index = _device_index(device)
    for name in _loader.route_modules(arch, route):
        _loader.load(name, arch)
        if _loader.MODULES[name]["grid_rule"]["kind"] == "persistent_rows":
            _loader.cluster_capacity(name, arch, index)
    if tensor_alpha and not has_post_weight:
        zero_post_weight(device)


def _check(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: {name} must be a contiguous {dtype} tensor of shape "
            f"{tuple(shape)} on {device} with {ALIGNMENT_BYTES}-byte aligned storage"
        )


def cake_gated_residual_next_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: Optional[torch.Tensor],
    beta: torch.Tensor,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    *,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    post_weight: Optional[torch.Tensor] = None,
    post_gain_center: float = 0.0,
    alpha_scalar: Optional[float] = None,
) -> None:
    """``residual_out = alpha * residual + beta * RMSNorm(branch * (post_weight + post_gain_center))`` in FP32
    and ``ffn_input = bf16(RMSNorm(residual_out) * (next_weight + next_gain_center))``, one launch of the
    exported program whose window admits the row count.

    ``alpha`` is the FP32 gate vector, or ``None`` with ``alpha_scalar`` the scalar gate (``alpha`` is then
    unread); ``post_weight`` ``None`` drops the first multiply.  The outputs may alias ``residual`` and
    ``branch``.  An empty batch launches nothing.
    """
    device = branch.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_next_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if branch.ndim != 2 or branch.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: branch must be [rows, {HIDDEN_SIZE}], got {tuple(branch.shape)}"
        )
    rows = int(branch.shape[0])
    matrix = (rows, HIDDEN_SIZE)
    vector = (HIDDEN_SIZE,)
    _check(branch, "branch", shape=matrix, dtype=torch.bfloat16, device=device)
    _check(residual, "residual", shape=matrix, dtype=torch.float32, device=device)
    _check(
        next_weight, "next_weight", shape=vector, dtype=torch.bfloat16, device=device
    )
    _check(beta, "beta", shape=vector, dtype=torch.float32, device=device)
    _check(
        residual_out, "residual_out", shape=matrix, dtype=torch.float32, device=device
    )
    _check(ffn_input, "ffn_input", shape=matrix, dtype=torch.bfloat16, device=device)
    tensor_alpha = alpha_scalar is None
    if tensor_alpha:
        if alpha is None:
            raise ValueError(
                "cake_gated_residual_next_rmsnorm: pass the alpha tensor or alpha_scalar"
            )
        _check(alpha, "alpha", shape=vector, dtype=torch.float32, device=device)
        alpha_value = 1.0
    else:
        # The scalar programs never read the alpha slot; beta keeps the tensor ABI.
        alpha = beta
        alpha_value = float(alpha_scalar)  # type: ignore[arg-type]
    has_post_weight = post_weight is not None
    if post_weight is not None:
        _check(
            post_weight,
            "post_weight",
            shape=vector,
            dtype=torch.bfloat16,
            device=device,
        )
        gain = float(post_gain_center)
    elif tensor_alpha:
        # No dedicated program: the post-weight route with a zero vector and gain 1.0 is exact.
        post_weight = zero_post_weight(device)
        gain = 1.0
    else:
        # The no-post-weight programs never read the slot; next_weight keeps the tensor ABI.
        post_weight = next_weight
        gain = 0.0
    route = gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: no exported program of route {route!r} on {arch}"
        )
    if rows == 0:
        return
    if not is_cake_gated_residual_next_rmsnorm_supported(
        device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight, rows=rows
    ):
        # a declared gap of the route: the facade's "auto" runs the CuTe DSL kernel there
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: no exported program of route {route!r} on {arch} admits "
            f"{rows} rows"
        )
    name = _loader.selected_module(arch, route, rows)
    values = {
        "branch": branch,
        "residual": residual,
        "alpha": alpha,
        "beta": beta,
        "next_weight": next_weight,
        "post_weight": post_weight,
        "residual_out": residual_out,
        "ffn_input": ffn_input,
        "rows": rows,
        "post_eps": float(post_eps),
        "next_eps": float(next_eps),
        "post_gain_center": gain,
        "next_gain_center": float(next_gain_center),
        "alpha_value": alpha_value,
    }
    # The loader binds the current torch stream of the *current* device; make that the
    # tensors' device so a call made while another device is current keeps the order
    # of the stream its inputs were produced on.
    with torch.cuda.device(device):
        _loader.run_module(name, values, rows=rows, device_index=_device_index(device))


def gated_residual_rmsnorm_variant(weight_before_norm: bool, enable_pdl: bool) -> str:
    """The variant a gated residual RMSNorm call maps onto: ``mode0`` (no ``weight``) or ``mode1`` (``weight +
    gain_center`` applied to ``branch`` before its norm), with ``_pdl`` for a programmatic dependent launch."""
    return f"mode{int(bool(weight_before_norm))}{'_pdl' if enable_pdl else ''}"


def gated_residual_rmsnorm_route(weight_before_norm: bool, enable_pdl: bool) -> str:
    """The route a gated residual RMSNorm call launches from (its variant's)."""
    return route_key(
        GATED_RESIDUAL_RMSNORM,
        gated_residual_rmsnorm_variant(weight_before_norm, enable_pdl),
    )


def is_cake_gated_residual_rmsnorm_supported(
    device: torch.device,
    *,
    weight_before_norm: bool = False,
    enable_pdl: bool = False,
    rows: Optional[int] = None,
) -> bool:
    """Whether this package ships a program for ``device`` and the call's variant (``weight`` applied before the norm
    or absent, programmatic dependent launch or not) -- and, with ``rows``, one whose window admits that row count
    (a route's windows may leave gaps where the kernel lane measured the FlashInfer kernel ahead).  The tensor
    contract is :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    route = gated_residual_rmsnorm_route(weight_before_norm, enable_pdl)
    if not _loader.route_available(arch, route):
        return False
    if rows is None:
        return True
    return any(
        _loader.admits(_loader.MODULES[name].get("select_rule"), int(rows), arch)
        for name in _loader.route_modules(arch, route)
    )


def _check_row(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % ROW_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_gated_residual_rmsnorm: {name} must be a contiguous {dtype} tensor of shape "
            f"{tuple(shape)} on {device} with {ROW_ALIGNMENT_BYTES}-byte aligned storage"
        )


def cake_gated_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
    gain_center: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """``out = alpha * residual + beta * RMSNorm(branch * (weight + gain_center))`` in FP32 (``weight`` ``None``
    drops the multiply), one launch of the exported program of the call's variant whose window admits the row
    count; ``enable_pdl`` selects the programmatic dependent launch build.  ``out`` may alias ``residual``.  An
    empty batch launches nothing.
    """
    device = branch.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if branch.ndim != 2 or branch.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"cake_gated_residual_rmsnorm: branch must be [rows, {HIDDEN_SIZE}], got {tuple(branch.shape)}"
        )
    rows = int(branch.shape[0])
    matrix = (rows, HIDDEN_SIZE)
    vector = (HIDDEN_SIZE,)
    _check_row(branch, "branch", shape=matrix, dtype=torch.bfloat16, device=device)
    _check_row(residual, "residual", shape=matrix, dtype=torch.float32, device=device)
    _check_row(alpha, "alpha", shape=vector, dtype=torch.float32, device=device)
    _check_row(beta, "beta", shape=vector, dtype=torch.float32, device=device)
    _check_row(out, "out", shape=matrix, dtype=torch.float32, device=device)
    weight_before_norm = weight is not None
    if weight is not None:
        _check_row(weight, "weight", shape=vector, dtype=torch.bfloat16, device=device)
    route = gated_residual_rmsnorm_route(weight_before_norm, enable_pdl)
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_gated_residual_rmsnorm: no exported program of route {route!r} on {arch}"
        )
    if rows == 0:
        return
    if weight is None:
        # The mode-0 programs never read the weight slot; a row of branch keeps the BF16 tensor ABI.
        weight = branch.reshape(-1)[:HIDDEN_SIZE]
    if not is_cake_gated_residual_rmsnorm_supported(
        device, weight_before_norm=weight_before_norm, enable_pdl=enable_pdl, rows=rows
    ):
        # a declared gap of the route: the facade's "auto" runs the CuTe DSL kernel there
        raise ValueError(
            f"cake_gated_residual_rmsnorm: no exported program of route {route!r} on {arch} admits "
            f"{rows} rows"
        )
    name = _loader.selected_module(arch, route, rows)
    values = {
        "branch": branch,
        "residual": residual,
        "weight": weight,
        "alpha": alpha,
        "beta": beta,
        "out": out,
        "rows": rows,
        "eps": float(eps),
        "gain_center": float(gain_center),
    }
    with torch.cuda.device(device):
        _loader.run_module(name, values, rows=rows, device_index=_device_index(device))


def rmsnorm_sigmoid_gate_route() -> str:
    """The one route of the sigmoid gate (BF16 rows of 64)."""
    return route_key(RMSNORM_SIGMOID_GATE, "bf16")


def is_cake_rmsnorm_sigmoid_gate_supported(
    device: torch.device,
    *,
    heads: Optional[int] = None,
    tokens: Optional[int] = None,
) -> bool:
    """Whether this package ships a program for ``device`` -- and, with ``heads``, one whose windows name that head
    count (a 2-D ``(rows, 64)`` call is one head), with ``tokens`` too, one whose window admits that token count.
    The tensor contract is :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    route = rmsnorm_sigmoid_gate_route()
    if not _loader.route_available(arch, route):
        return False
    if heads is None:
        return True
    for name in _loader.route_modules(arch, route):
        windows = _loader.MODULES[name]["select_rule"]["windows"].get(arch, ())
        for count, above, at_most in windows:
            if int(count) != int(heads):
                continue
            if tokens is None or (
                tokens > int(above) and (at_most is None or tokens <= int(at_most))
            ):
                return True
    return False


def _gate_view(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, int]:
    """``tensor`` as ``(tokens, heads, 64)`` with its token stride, after the layout checks of the contract: BF16,
    2-D ``(tokens, 64)`` or 3-D ``(tokens, heads, 64)``, unit stride along the row, packed heads, a token stride that
    is a multiple of eight elements and 16-byte aligned storage."""
    if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must be a CUDA bfloat16 tensor"
        )
    if tensor.ndim == 2:
        tensor = tensor.view(tensor.shape[0], 1, tensor.shape[1])
    if tensor.ndim != 3 or tensor.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must be [tokens, {HEAD_DIM}] or [tokens, heads, {HEAD_DIM}], "
            f"got {tuple(tensor.shape)}"
        )
    if (
        tensor.stride(-1) != 1
        or tensor.stride(1) != HEAD_DIM
        or tensor.stride(0) % 8 != 0
        or tensor.data_ptr() % ROW_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must hold contiguous rows of {HEAD_DIM} with packed heads, a token "
            f"stride that is a multiple of 8 elements and {ROW_ALIGNMENT_BYTES}-byte aligned storage"
        )
    return tensor, int(tensor.stride(0))


def cake_rmsnorm_sigmoid_gate(
    input: torch.Tensor, gate: torch.Tensor, *, eps: float
) -> None:
    """``input = bf16(RMSNorm(input) * sigmoid(gate))`` per row of 64, in place, one launch of the exported program
    whose window admits the head count and the token count.  ``input`` and ``gate`` are ``(tokens, 64)`` or
    ``(tokens, heads, 64)`` with the same shape; their token strides may differ (``gate`` is typically a view of a
    wider projection).  An empty batch launches nothing.
    """
    device = input.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_rmsnorm_sigmoid_gate supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if gate.shape != input.shape or gate.device != device:
        raise ValueError(
            "cake_rmsnorm_sigmoid_gate: gate must have the shape and device of input, got "
            f"{tuple(gate.shape)} on {gate.device} for {tuple(input.shape)} on {device}"
        )
    x, input_stride = _gate_view(input, "input")
    g, gate_stride = _gate_view(gate, "gate")
    tokens, heads = int(x.shape[0]), int(x.shape[1])
    route = rmsnorm_sigmoid_gate_route()
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: no exported program of route {route!r} on {arch}"
        )
    if tokens == 0:
        return
    if not is_cake_rmsnorm_sigmoid_gate_supported(device, heads=heads, tokens=tokens):
        # a head count or a token count outside the route's windows: the facade's "auto" runs the CuTe DSL kernel
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: no exported program of route {route!r} on {arch} admits "
            f"{tokens} tokens of {heads} heads"
        )
    name = _loader.selected_module(arch, route, tokens, heads)
    values = {
        "input": x,
        "gate": g,
        "tokens": tokens,
        "heads": heads,
        "input_stride": input_stride,
        "gate_stride": gate_stride,
        "eps": float(eps),
    }
    with torch.cuda.device(device):
        _loader.run_module(
            name, values, rows=tokens, device_index=_device_index(device)
        )


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GATED_RESIDUAL_NEXT_RMSNORM",
    "GATED_RESIDUAL_RMSNORM",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "OPS",
    "RMSNORM_SIGMOID_GATE",
    "ROW_ALIGNMENT_BYTES",
    "VARIANTS",
    "cake_gated_residual_next_rmsnorm",
    "cake_gated_residual_rmsnorm",
    "cake_rmsnorm_sigmoid_gate",
    "gated_residual_next_rmsnorm_route",
    "gated_residual_next_rmsnorm_variant",
    "gated_residual_rmsnorm_route",
    "gated_residual_rmsnorm_variant",
    "is_cake_gated_residual_next_rmsnorm_supported",
    "is_cake_gated_residual_rmsnorm_supported",
    "is_cake_rmsnorm_sigmoid_gate_supported",
    "rmsnorm_sigmoid_gate_route",
    "warm_up",
    "zero_post_weight",
]
