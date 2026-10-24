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
glue and the generated CUDA sources under ``csrc/``): today the gated residual
with the next RMSNorm fused behind
``flashinfer.norm.gated_residual_next_rmsnorm(..., backend="cake")``
(:func:`cake_gated_residual_next_rmsnorm`; hidden size 8192, SM100 / SM103).
The stable, ``@flashinfer_api``-decorated entry point and the user-facing
documentation live in :mod:`flashinfer.norm`; import it.

The kernels ship in three variants of their optional inputs: ``alpha`` a
tensor or a scalar, ``post_weight`` present or not (``alpha_tensor_pw``,
``alpha_scalar_nopw``, ``alpha_scalar_pw``).  A call with a tensor ``alpha``
and no ``post_weight`` runs the ``alpha_tensor_pw`` programs with a zero
``post_weight`` vector and ``post_gain_center = 1.0``: ``branch * (0 + 1)`` is
``branch`` exactly, so the result is the one a dedicated program would give.
The zero vector is allocated once per device on first use (never inside CUDA
graph capture: :func:`warm_up` prepares a device before a capture).
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
#: The kernels load every operand in 256-bit accesses: rows and vectors are 32-byte aligned.
ALIGNMENT_BYTES = 32

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
) -> bool:
    """Whether this package ships a program for ``device`` and the call's variant (tensor or scalar ``alpha``,
    with or without ``post_weight``); the tensor contract is :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    return _loader.route_available(
        arch, gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
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


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GATED_RESIDUAL_NEXT_RMSNORM",
    "HIDDEN_SIZE",
    "OPS",
    "VARIANTS",
    "cake_gated_residual_next_rmsnorm",
    "gated_residual_next_rmsnorm_route",
    "gated_residual_next_rmsnorm_variant",
    "is_cake_gated_residual_next_rmsnorm_supported",
    "warm_up",
    "zero_post_weight",
]
