# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Cake ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107), deterministic training module.

The forward projection ``Y = X @ W[e].T``, the data gradient ``dX = G @ W[e]``
and the weight gradient ``dW[e] = G[e].T @ X[e]`` over groups of rows that an
int32 device tensor of cumulative end offsets describes (``offs`` ``[E]``, or
``m_indptr`` ``[E + 1]``; never read on the host).  Every launch reduces in a
fixed order: ``Y`` and both gradients are bitwise reproducible run to run, also
when the weight gradient is added into a caller's buffer (``accumulate=True``,
``fuse_wgrad_accumulation=True``).

The module is the mode.  No entry takes a ``deterministic`` argument (passing
one raises ``TypeError``), and ``torch.use_deterministic_algorithms(True)``
changes nothing here (no alert, no warning).  The non-deterministic module is
:mod:`flashinfer.cake_moe_grouped_gemm_nondet_train`.

Each function below is a stable FlashInfer API (``@flashinfer_api``; no
experimental marking): it carries the signature and the documentation of its
namesake in the backend package
:mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train` and hands
off to it with the same arguments.  The prepared launch and the
``torch.autograd.Function`` resolve from the package on first access.  Every
function imports the package inside its body (a deferred import), so importing
this module loads none of it.  :func:`is_cake_moe_grouped_gemm_det_train_supported`
tells whether a device has the programs of every operation.

Usage::

    from flashinfer.cake_moe_grouped_gemm_det_train import (
        cake_moe_grouped_gemm_det,
        cake_moe_grouped_gemm_det_train_wgrad,
    )

    y = cake_moe_grouped_gemm_det(x, w, offs)  # autograd
    dw = cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out_dtype=torch.float32)
    cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=main_grad, accumulate=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from .api_logging import flashinfer_api

if TYPE_CHECKING:
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_autograd import (
        CakeMoEGroupedGemmDetFunction,
    )
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import (
        CakeMoEGroupedGemmDetLaunch,
    )

__all__ = [
    "CakeMoEGroupedGemmDetFunction",
    "CakeMoEGroupedGemmDetLaunch",
    "cake_moe_grouped_gemm_det",
    "cake_moe_grouped_gemm_det_train_dgrad",
    "cake_moe_grouped_gemm_det_train_forward",
    "cake_moe_grouped_gemm_det_train_prepare_dgrad",
    "cake_moe_grouped_gemm_det_train_prepare_forward",
    "cake_moe_grouped_gemm_det_train_prepare_wgrad",
    "cake_moe_grouped_gemm_det_train_wgrad",
    "is_cake_moe_grouped_gemm_det_train_supported",
]

_PACKAGE = "flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train"


@flashinfer_api
def cake_moe_grouped_gemm_det_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights, one ``[N, K]`` matrix per group.
    offs : torch.Tensor
        ``[E]`` int32 cumulative end offsets on the device (``offs[-1] == sum_m``),
        or ``m_indptr`` ``[E + 1]`` with a leading 0; read by the kernel, never
        on the host.
    out : Optional[torch.Tensor]
        ``[sum_m, N]`` bfloat16 output (allocated when omitted); may be row-padded.

    Returns
    -------
    torch.Tensor
        ``out``.  ``N % 256 == 0`` and ``K % 64 == 0`` are required.

    Notes
    -----
    Prepares and launches in one call.  For repeated launches over the same
    tensors (or CUDA Graph capture) use ``cake_moe_grouped_gemm_det_train_prepare_forward`` once and
    call ``launch()`` on the returned object.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_moe_grouped_gemm_det_train_forward as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_det_train_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  See ``cake_moe_grouped_gemm_det_train_forward`` for the prepared form.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[sum_m, N]`` packed output-gradient rows.
    w : torch.Tensor
        BF16 ``[E, N, K]`` expert weights, read in place.
    offs : torch.Tensor
        int32 device cumulative end offsets ``[E]``, or leading-zero ``m_indptr [E + 1]``.
    out : torch.Tensor, optional
        Caller-owned BF16 ``[sum_m, K]`` output; allocated when omitted.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_moe_grouped_gemm_det_train_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_det_train_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    accumulate: bool = False,
) -> torch.Tensor:
    """``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` for every group ``e``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, bitwise deterministic, exact zeros for empty
    groups.  ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256,
    or 512 when chosen) are required.  See ``cake_moe_grouped_gemm_det_train_forward`` for the
    prepared form.

    ``accumulate=True`` (keyword-only, default ``False``) adds into ``out``
    instead, ``out[e] += G[e].T @ X[e]`` (Megatron's gradient-accumulation
    fusion into an fp32 or bf16 ``main_grad``; ``out`` required, bfloat16 or
    float32, ``out_dtype`` None or its dtype, no overlap of ``out`` with itself,
    ``g``, ``x`` or ``offs``: ``ValueError``), and returns ``out``, which holds
    AccumulateGrad's bits of the deterministic weight gradient in its dtype,
    bitwise reproducible run to run: fused into the 512-wide k tile's programs
    (the main stage and the ordered tail reduce add into ``out`` in place)
    where the route serves their accumulate builds for ``out``'s dtype, else
    the overwrite into a scratch of ``out``'s dtype (``E * N * K`` elements,
    ``4 * E * N * K`` bytes for fp32) added with ``out.add_`` on the same
    stream; both give the same bits
    (``cake_moe_grouped_gemm_det_train_prepare_wgrad`` and the package README
    give the routes; a prepared launch is CUDA-graph capturable and every
    launch and replay adds again).  ``sum_m == 0`` leaves ``out`` unchanged.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[sum_m, N]`` packed output-gradient rows.
    x : torch.Tensor
        BF16 ``[sum_m, K]`` packed activation rows.
    offs : torch.Tensor
        int32 device cumulative end offsets ``[E]``, or leading-zero ``m_indptr [E + 1]``.
    out : torch.Tensor, optional
        Caller-owned ``[E, N, K]`` output; required when accumulating.
    out_dtype : torch.dtype, optional
        ``torch.bfloat16`` or ``torch.float32``; defaults to ``out.dtype`` or bfloat16 without ``out``.
    num_groups : int, optional
        Expert count ``E``; must be supplied for ``m_indptr`` input.
    accumulate : bool, default=False
        Add the weight gradient into ``out`` instead of overwriting it.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_moe_grouped_gemm_det_train_wgrad as _impl,
    )

    return _impl(
        g,
        x,
        offs,
        out=out,
        out_dtype=out_dtype,
        num_groups=num_groups,
        accumulate=accumulate,
    )


@flashinfer_api
def cake_moe_grouped_gemm_det_train_prepare_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0`` (the
    512-column tile when ``N % 512 == 0`` on the architectures of
    ``wide_fwd_archs``, its L2 evict-first twin where ``fwd_lwh_ef`` or
    ``fwd_evict_first`` holds for ``K``, its 3 / 3 head / tail twin where
    ``fwd_wide_ts33`` holds).
    ``out`` may be row-padded (any row stride, unit column
    stride).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[sum_m, K]`` packed activation rows.
    w : torch.Tensor
        BF16 ``[E, N, K]`` expert weights.
    offs : torch.Tensor
        int32 device cumulative end offsets ``[E]``, or leading-zero ``m_indptr [E + 1]``.
    out : torch.Tensor, optional
        Caller-owned BF16 ``[sum_m, N]`` output; allocated when omitted.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import (
        cake_moe_grouped_gemm_det_train_prepare_forward as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_det_train_prepare_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``dX[sum_m, K] = grouped G @ W[e]`` (W read in place; bf16 in, bf16 out).

    ``g``: ``[sum_m, N]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0`` (the
    512-column tile when ``K % 512 == 0`` on the architectures of
    ``wide_dgrad_archs``, its L2 evict-first twin where ``dgrad_evict_first``
    holds for ``K``, ``sum_m`` and ``E``, its 3 / 3 head / tail twin where
    ``dgrad_wide_ts33`` holds).

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[sum_m, N]`` packed output-gradient rows.
    w : torch.Tensor
        BF16 ``[E, N, K]`` expert weights, read in place.
    offs : torch.Tensor
        int32 device cumulative end offsets ``[E]``, or leading-zero ``m_indptr [E + 1]``.
    out : torch.Tensor, optional
        Caller-owned BF16 ``[sum_m, K]`` output; allocated when omitted.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import (
        cake_moe_grouped_gemm_det_train_prepare_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_det_train_prepare_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    accumulate: bool = False,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).

    ``accumulate=True`` (keyword-only, default ``False``) adds instead: every
    launch computes ``out[e] <- out[e] + G[e].T @ X[e]`` into the caller's
    ``out`` (required, bf16 or fp32; ``out_dtype`` None or ``out.dtype``; rows
    and experts that do not overlap, no memory shared with ``g``, ``x`` or
    ``offs``; ``ValueError`` otherwise, before anything is allocated or
    bound), so ``out`` holds AccumulateGrad's bits of the deterministic weight
    gradient in its dtype, bitwise reproducible run to run (an empty group
    adds exact zeros: its values stay, ``-0.0`` becomes ``+0.0``).
    ``plan["accumulate"]`` is ``"fused"`` (the accumulate builds of the main
    stage and, for a split tail, of the tail reduce add into ``out`` in place:
    ``stages`` is the overwrite's with ``ACC_SUFFIX`` on each) or
    ``"composed"`` (the overwrite of this plan into a contiguous scratch of
    ``out.dtype``, ``E * N * K`` elements that this preparation allocates and
    the launch owns, then ``out.add_(scratch)`` on the same stream;
    ``plan["accumulate_reason"]`` says why); both give the same bits, and
    every other plan key is the overwrite's (module docstring, accumulate
    mode).  No row (``sum_m == 0``) launches nothing and leaves ``out``
    unchanged.  Each ``launch()`` and each replay of a graph that captured it
    adds again; keep the launch alive while such a graph may replay.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[sum_m, N]`` packed output-gradient rows.
    x : torch.Tensor
        BF16 ``[sum_m, K]`` packed activation rows.
    offs : torch.Tensor
        int32 device cumulative end offsets ``[E]``, or leading-zero ``m_indptr [E + 1]``.
    out : torch.Tensor, optional
        Caller-owned ``[E, N, K]`` output; required when accumulating.
    out_dtype : torch.dtype, optional
        ``torch.bfloat16`` or ``torch.float32``; defaults to ``out.dtype`` or bfloat16 without ``out``.
    num_groups : int, optional
        Expert count ``E``; must be supplied for ``m_indptr`` input.
    accumulate : bool, default=False
        Add the weight gradient into ``out`` instead of overwriting it.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_backend import (
        cake_moe_grouped_gemm_det_train_prepare_wgrad as _impl,
    )

    return _impl(
        g,
        x,
        offs,
        out=out,
        out_dtype=out_dtype,
        num_groups=num_groups,
        accumulate=accumulate,
    )


@flashinfer_api
def cake_moe_grouped_gemm_det(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    *,
    fuse_wgrad_accumulation: bool = False,
) -> torch.Tensor:
    """Differentiable ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``.

    The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
    For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

    Parameters
    ----------
    x : torch.Tensor
        ``[sum_m, K]`` bfloat16 rows of all groups, group after group.
    w : torch.Tensor
        ``[E, N, K]`` bfloat16 weights (``[N, K]`` per group).
    offs : torch.Tensor
        int32 device tensor: ``[E]`` cumulative end offsets or ``m_indptr``
        ``[E + 1]``.  Read by the kernels, never on the host.
    fuse_wgrad_accumulation : bool
        Keyword-only.  ``False`` (default): the backward returns ``w``'s
        gradient as below.  ``True`` (Megatron's
        ``gradient_accumulation_fusion``): ``w.main_grad`` must be an fp32 or
        bf16 tensor of ``w``'s shape on its device with unit stride along the
        last dimension and rows and experts that do not overlap (``ValueError``,
        checked at the forward call when autograd records it: ``w`` requires
        grad, grad mode is on and the call is not a checkpoint recompute inside
        a backward; and again at the backward, which reads ``w.main_grad`` then
        and also refuses one that shares memory with the operands).  Under
        ``torch.no_grad()`` or inference mode the forward needs no
        ``main_grad``.  The backward adds the deterministic ``dW`` into it in
        place (``main_grad[e] += G[e].T @ X[e]``, the package's accumulate
        mode: fused into the weight-gradient programs where the route serves
        their accumulate builds for ``main_grad``'s dtype, else ``dW`` in
        ``main_grad``'s dtype into a scratch, then ``main_grad.add_`` on the
        same stream), which gives AccumulateGrad's bits of that ``dW`` on
        either route, bitwise reproducible run to run, and returns for ``w`` a fresh
        ``torch.empty`` of ``w``'s shape and dtype when ``w`` has the attribute
        ``grad_added_to_main_grad`` (set to ``True``; ``torch.zeros`` when
        ``w.zero_out_wgrad`` is true), else ``None``: ``w.grad`` is then
        untouched and no gradient hook of ``w`` fires.  Every backward adds
        (``retain_graph`` twice adds twice), also a ``torch.autograd.grad``
        whose ``inputs`` exclude ``w`` (split backward-weight schedules are not
        supported); a ``w`` that does not require grad is never added to.  Not
        supported (``NotImplementedError``) under ``torch.jit.trace``,
        ``torch.compile`` and a ``create_graph`` backward; under a reentrant
        checkpoint ``w`` arrives detached, without ``main_grad``
        (``ValueError``).  Cost: on the composed route the weight-gradient
        launch holds a scratch of ``E * N * K`` elements of ``main_grad``'s
        dtype (``4 * E * N * K`` bytes for fp32, freed after the add), which
        the programs write and the add reads back besides its read and write
        of ``main_grad``; the fused route holds none.  The placeholder takes
        ``2 * E * N * K`` bytes.  Under
        ``torch.use_deterministic_algorithms(True)`` with
        ``torch.utils.deterministic.fill_uninitialized_memory`` on (the
        default) torch also fills the scratch and the ``torch.empty``
        placeholder with NaN when it allocates them.

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16.  ``backward`` produces ``x.grad`` ``[sum_m, K]``
        bfloat16 and ``w.grad`` ``[E, N, K]`` in ``w``'s dtype (bfloat16); use
        :func:`cake_moe_grouped_gemm_det_train_wgrad` directly for an fp32 weight gradient, or
        ``fuse_wgrad_accumulation=True`` to add it into an fp32 ``main_grad``.
        The forward GEMM runs after autograd saved ``x``, ``w`` and ``offs``:
        under a non-reentrant checkpoint with early stop on, the recomputation
        skips it when these are the last saves of the innermost checkpointed
        region, as for a built-in op.

    Notes
    -----
    Deterministic only: the forward and both gradients reduce in a fixed order
    and are bitwise reproducible run to run, with or without
    ``fuse_wgrad_accumulation``.  There is no ``deterministic`` parameter
    (passing one raises ``TypeError``) and
    ``torch.use_deterministic_algorithms(True)`` changes nothing (no alert, no
    warning).  The non-deterministic wrapper is
    ``flashinfer.cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet``.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_autograd import (
        cake_moe_grouped_gemm_det as _impl,
    )

    return _impl(x, w, offs, fuse_wgrad_accumulation=fuse_wgrad_accumulation)


@flashinfer_api
def is_cake_moe_grouped_gemm_det_train_supported(
    device: Optional[torch.device] = None,
) -> bool:
    """Whether the deterministic MoE grouped GEMM runs on ``device``.

    True when ``device`` (default: the current CUDA device) is a CUDA device whose
    architecture this checkout registers a program for, for every operation of the
    module: the forward, the data gradient and the weight gradient (either output
    dtype).  False without CUDA, for a device that is not CUDA or not visible, and
    for an architecture without the programs.  The query loads no program and
    launches nothing.

    Parameters
    ----------
    device : torch.device, optional
        Device to query; ``None`` selects the current CUDA device.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.type != "cuda":
        return False
    if device.index is not None and device.index >= torch.cuda.device_count():
        return False
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        cake_backend,
        cake_jit,
    )

    return all(
        cake_backend.generated_program_available(device, op) for op in cake_jit.OPS
    )


# The two types of the module, resolved from the package on first access (importing this module
# imports none of it): public name -> the submodule that defines it.
_LAZY_TYPES = {
    "CakeMoEGroupedGemmDetFunction": "cake_autograd",
    "CakeMoEGroupedGemmDetLaunch": "cake_backend",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_TYPES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{_PACKAGE}.{submodule}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))
