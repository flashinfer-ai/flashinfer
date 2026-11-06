# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental deterministic Cake ragged BF16 grouped GEMM entry points.

Thin core entry points, per ``flashinfer/experimental/README.md``: each
function below carries the public signature and the documentation of its
namesake in :mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train`, adds the
experimental-API marking, and hands off to it.
The kernels, the host planning, the autograd wrapper and the JIT
registration live in that package, which every function imports inside its
body (a deferred import, never at module level), so importing
:mod:`flashinfer.grouped_mm` loads none of it.
"""

from __future__ import annotations

from typing import Optional

import torch

from .api_logging import flashinfer_experimental_api

__all__ = [
    "grouped_det_gemm_fwd",
    "grouped_det_gemm_dgrad",
    "grouped_det_gemm_wgrad",
    "cake_grouped_det_mm",
]


@flashinfer_experimental_api(feature="cake_moe_grouped_det_gemm.grouped_det_gemm_fwd")
def grouped_det_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T`` for every group ``e``.

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
    tensors (or CUDA Graph capture) use ``prepare_grouped_det_gemm_fwd`` once and
    call ``launch()`` on the returned object.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        grouped_det_gemm_fwd as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_experimental_api(feature="cake_moe_grouped_det_gemm.grouped_det_gemm_dgrad")
def grouped_det_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  See ``grouped_det_gemm_fwd`` for the prepared form.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        grouped_det_gemm_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_experimental_api(feature="cake_moe_grouped_det_gemm.grouped_det_gemm_wgrad")
def grouped_det_gemm_wgrad(
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

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, bitwise deterministic, exact zeros for empty
    groups.  ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256,
    or 512 when chosen) are required.  See ``grouped_det_gemm_fwd`` for the
    prepared form.

    ``accumulate=True`` (keyword-only, default ``False``) adds into ``out``
    instead, ``out[e] += G[e].T @ X[e]`` (Megatron's gradient-accumulation
    fusion into an fp32 or bf16 ``main_grad``; ``out`` required, bfloat16 or
    float32, ``out_dtype`` None or its dtype, no overlap of ``out`` with itself,
    ``g``, ``x`` or ``offs``: ``ValueError``), and returns ``out``: the
    deterministic weight gradient in ``out``'s dtype is written into a scratch
    of that dtype and added with ``out.add_`` on the same stream, so ``out``
    holds AccumulateGrad's bits of it, bitwise reproducible run to run.  The
    scratch takes ``E * N * K`` elements (``4 * E * N * K`` bytes for fp32)
    for the call; ``prepare_grouped_det_gemm_wgrad(..., accumulate=True)``
    keeps it with a prepared launch (CUDA-graph capturable; every launch and
    replay adds again).  ``sum_m == 0`` leaves ``out`` unchanged.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train import (
        grouped_det_gemm_wgrad as _impl,
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


@flashinfer_experimental_api(feature="cake_moe_grouped_det_gemm.cake_grouped_det_mm")
def cake_grouped_det_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    *,
    fuse_wgrad_accumulation: bool = False,
) -> torch.Tensor:
    """Differentiable ragged grouped GEMM ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``.

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
        place (``main_grad[e] += G[e].T @ X[e]``: ``dW`` in ``main_grad``'s
        dtype into a scratch, then ``main_grad.add_`` on the same stream, the
        package's accumulate mode), which gives AccumulateGrad's bits of that
        ``dW``, bitwise reproducible run to run, and returns for ``w`` a fresh
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
        (``ValueError``).  Cost: the weight-gradient launch holds a scratch of
        ``E * N * K`` elements of ``main_grad``'s dtype (``4 * E * N * K``
        bytes for fp32, freed after the add), which the programs write and the
        add reads back besides its read and write of ``main_grad``; the
        placeholder takes ``2 * E * N * K`` bytes.  Under
        ``torch.use_deterministic_algorithms(True)`` with
        ``torch.utils.deterministic.fill_uninitialized_memory`` on (the
        default) torch also fills the scratch and the ``torch.empty``
        placeholder with NaN when it allocates them.

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16.  ``backward`` produces ``x.grad`` ``[sum_m, K]``
        bfloat16 and ``w.grad`` ``[E, N, K]`` in ``w``'s dtype (bfloat16); use
        :func:`grouped_det_gemm_wgrad` directly for an fp32 weight gradient, or
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
    ``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_grouped_mm(...,
    deterministic=False)``.
    """
    from .experimental.cake.cake_moe_grouped_gemm_det_train.cake_autograd import (
        cake_grouped_det_mm as _impl,
    )

    return _impl(x, w, offs, fuse_wgrad_accumulation=fuse_wgrad_accumulation)
