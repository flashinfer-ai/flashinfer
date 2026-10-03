# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fused low-M BF16 GEMM and SwiGLU public API."""

from __future__ import annotations

from typing import Literal, Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import mm_bf16_swiglu_trace
from ..utils import (
    LibraryError,
    backend_requirement,
    is_sm100a_supported,
    supported_compute_capability,
)

_GATE_UP_PAIR_ROWS = 16
_OUTPUT_ALIGNMENT = 64
_K_ALIGNMENT = 128
_MAX_M = 64


def _dense_tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two validated dense tensors overlap in storage."""
    if lhs.device != rhs.device or lhs.numel() == 0 or rhs.numel() == 0:
        return False
    lhs_start = lhs.data_ptr()
    rhs_start = rhs.data_ptr()
    lhs_end = lhs_start + lhs.numel() * lhs.element_size()
    rhs_end = rhs_start + rhs.numel() * rhs.element_size()
    return lhs_start < rhs_end and rhs_start < lhs_end


@flashinfer_api
def prepare_bf16_swiglu_weight(
    weight: torch.Tensor,
    *,
    input_order: Literal["gate_up", "up_gate"] = "gate_up",
) -> torch.Tensor:
    r"""Prepare a canonical BF16 gate/up weight for :func:`mm_bf16_swiglu`.

    The input is a contiguous row-major ``(2 * N, K)`` linear weight.  The
    two projections may be stored as ``[gate, up]`` (the default, matching
    common checkpoint layouts) or ``[up, gate]``.  This function returns the
    column-major ``(K, 2 * N)`` view consumed by the fused kernel.  Its
    contiguous backing storage is interleaved in 16-row groups as::

        [up[0:16], gate[0:16], up[16:32], gate[16:32], ...]

    Weight preparation allocates and reorders the weight, and is intended to
    run once while loading a model, not in a model's forward pass.

    Parameters
    ----------
    weight:
        Contiguous row-major BF16 tensor with shape ``(2 * N, K)``.
    input_order:
        Projection order in ``weight``: ``"gate_up"`` or ``"up_gate"``.

    Returns
    -------
    torch.Tensor
        BF16 tensor with shape ``(K, 2 * N)`` whose transpose is contiguous.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"weight must be a torch.Tensor; got {type(weight).__name__}")
    if input_order not in ("gate_up", "up_gate"):
        raise ValueError(
            f"input_order must be 'gate_up' or 'up_gate'; got {input_order!r}"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2-D with shape (2 * N, K); got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight must be bfloat16; got {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous row-major")

    gate_up_width, k = (int(dim) for dim in weight.shape)
    if gate_up_width <= 0 or gate_up_width % 2:
        raise ValueError(
            "weight.shape[0] must be a positive even gate/up width; "
            f"got {gate_up_width}"
        )
    n = gate_up_width // 2
    if n % _OUTPUT_ALIGNMENT:
        raise ValueError(
            f"logical output width N must be divisible by {_OUTPUT_ALIGNMENT}; "
            f"got N={n}"
        )
    if k <= 0 or k % _K_ALIGNMENT:
        raise ValueError(f"K must be a positive multiple of {_K_ALIGNMENT}; got K={k}")

    first, second = weight[:n], weight[n:]
    gate, up = (first, second) if input_order == "gate_up" else (second, first)
    groups = n // _GATE_UP_PAIR_ROWS
    prepared_rows = torch.stack(
        (
            up.view(groups, _GATE_UP_PAIR_ROWS, k),
            gate.view(groups, _GATE_UP_PAIR_ROWS, k),
        ),
        dim=1,
    ).reshape(gate_up_width, k)
    return prepared_rows.transpose(0, 1)


def _validate_mm_bf16_swiglu_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
    pdl: bool,
) -> tuple[int, int, int]:
    """Validate the public tensor contract and return logical ``(M, N, K)``."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be torch.Tensor instances")
    if not isinstance(pdl, bool):
        raise TypeError(f"pdl must be bool; got {type(pdl).__name__}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            "a and b must be 2-D with shapes (M, K) and (K, 2 * N); "
            f"got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"a and b must both be bfloat16; got {a.dtype} and {b.dtype}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("a and b must be on the same CUDA device")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous row-major")
    if not b.T.is_contiguous():
        raise ValueError(
            "b must be column-major (b.T contiguous); pass the result of "
            "prepare_bf16_swiglu_weight"
        )

    m, k = (int(dim) for dim in a.shape)
    if not 1 <= m <= _MAX_M:
        raise ValueError(
            f"mm_bf16_swiglu requires 1 <= M <= {_MAX_M}; got M={m}. Above the "
            "cap the tile selector degrades badly, and this op cannot fall back "
            "internally because it only holds the interleaved weight. Use "
            "mm_bf16 followed by silu_and_mul on the canonical (2 * N, K) "
            "weight instead."
        )
    if int(b.shape[0]) != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    gate_up_width = int(b.shape[1])
    if gate_up_width <= 0 or gate_up_width % 2:
        raise ValueError(
            f"b.shape[1] must be a positive even gate/up width; got {gate_up_width}"
        )
    n = gate_up_width // 2
    if n % _OUTPUT_ALIGNMENT:
        raise ValueError(
            f"logical output width N must be divisible by {_OUTPUT_ALIGNMENT}; "
            f"got N={n}"
        )
    if k <= 0 or k % _K_ALIGNMENT:
        raise ValueError(f"K must be a positive multiple of {_K_ALIGNMENT}; got K={k}")

    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"out must be a torch.Tensor; got {type(out).__name__}")
        if out.device != a.device:
            raise ValueError(f"out must be on {a.device}; got {out.device}")
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be bfloat16; got {out.dtype}")
        if out.shape != (m, n):
            raise ValueError(f"out must have shape {(m, n)}; got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous row-major")
        if _dense_tensors_overlap(out, a):
            raise ValueError("out must not overlap a storage")
        if _dense_tensors_overlap(out, b):
            raise ValueError("out must not overlap b storage")

    tensors = (a, b) if out is None else (a, b, out)
    if any(tensor.data_ptr() % 32 for tensor in tensors):
        raise ValueError("a, b, and out must be 32-byte aligned")

    return m, n, k


@supported_compute_capability([100, 103])
def _mm_bf16_swiglu_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
) -> bool:
    _validate_mm_bf16_swiglu_tensors(a, b, out, pdl)
    if not is_sm100a_supported(a.device):
        raise ValueError("mm_bf16_swiglu requires SM100/SM103 with CUDA 12.8+")

    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        raise LibraryError("mm_bf16_swiglu requires nvidia-cutlass-dsl")
    return True


@backend_requirement({}, common_check=_mm_bf16_swiglu_requirement)
@flashinfer_api(trace=mm_bf16_swiglu_trace)
def mm_bf16_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute a low-M BF16 GEMM with a fused, unclamped SwiGLU epilogue.

    The operation computes two projections and returns::

        gate_bf16 = bf16(a.float() @ weight_gate.float().T)
        up_bf16 = bf16(a.float() @ weight_up.float().T)
        out = bf16(silu(gate_bf16.float()) * up_bf16.float())

    Here ``weight_gate`` and ``weight_up`` are the canonical projections
    recovered from the prepared ``b`` layout.

    The intermediate BF16 round trip deliberately matches an unfused BF16
    GEMM followed by a BF16 SwiGLU kernel.  This v1 API does not implement
    bias, alpha/beta, or gate/up clamping.

    The kernel tile is picked by a cost model bounded by the device's SM
    count and L2 capacity, not by profiling, so this op is not registered
    with the autotuner and :func:`flashinfer.autotuner.autotune` has no
    effect on it.

    ``M`` is capped at 64 rather than falling back to an unfused path,
    because ``b`` is interleaved and the unfused composition needs the
    canonical weight this function never receives.  **Retain the canonical
    ``(2 * N, K)`` weight** if the caller can exceed the cap; the fallback is
    :func:`~flashinfer.gemm.mm_bf16` on it followed by
    :func:`~flashinfer.activation.silu_and_mul`.  The cap itself is not a
    hardware limit: past it a wide ``N`` can leave no tile satisfying either
    the one-wave or the L2-residency bound, and the tile selector then
    degrades badly.

    TODO(@mattteochen): rank tiles by weight traffic once neither bound is
    satisfiable, then raise or drop the cap entirely.

    Parameters
    ----------
    a:
        Contiguous row-major BF16 tensor with shape ``(M, K)``.  ``M`` must
        be in ``[1, 64]``.
    b:
        Prepared column-major BF16 tensor with shape ``(K, 2 * N)``.  Obtain
        it from :func:`prepare_bf16_swiglu_weight`; ``N`` must be divisible
        by 64 and ``K`` by 128.
    pdl:
        Enable Programmatic Dependent Launch.
    out:
        Optional contiguous row-major BF16 output with shape ``(M, N)``. Its
        storage must not overlap ``a`` or ``b``.

    Returns
    -------
    torch.Tensor
        BF16 tensor with shape ``(M, N)``.
    """
    m, k = (int(dim) for dim in a.shape)
    n = int(b.shape[1]) // 2
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    from .kernels.dense_bf16_swiglu_sm100_splitk import (
        default_swiglu_tactic,
        run_splitk_swiglu,
    )

    return run_splitk_swiglu(a, b, out, pdl, default_swiglu_tactic(m, n, k))


__all__ = ["mm_bf16_swiglu", "prepare_bf16_swiglu_weight"]
