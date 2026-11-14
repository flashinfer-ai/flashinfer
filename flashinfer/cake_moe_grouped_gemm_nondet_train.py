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

Cake ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107), non-deterministic
training module.

The functions of this module are stable FlashInfer APIs (``@flashinfer_api``;
no experimental marking) and hand off to the backend package
:mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train` with the
same arguments: the weight gradient with the 512-wide k tile finishes its
split tail (the tiles of its last partial wave, cut into k-step units over the
clusters) in the same launch, where the deterministic package adds a second
tail-reduce launch: each unit stores its fp32 partial to a slot and the tile's
last unit sums the slots in slot order.

The module is the mode.  No entry takes a ``deterministic`` argument (passing
one raises ``TypeError``); every call runs the non-deterministic package, and
nothing here routes to the deterministic module
:mod:`flashinfer.cake_moe_grouped_gemm_det_train`, whose entries have the same
shapes::

    this module (non-deterministic)                  deterministic module
    cake_moe_grouped_gemm_nondet_train_forward       cake_moe_grouped_gemm_det_train_forward
    cake_moe_grouped_gemm_nondet_train_dgrad         cake_moe_grouped_gemm_det_train_dgrad
    cake_moe_grouped_gemm_nondet_train_wgrad         cake_moe_grouped_gemm_det_train_wgrad
    cake_moe_grouped_gemm_nondet_train_prepare_<op>  cake_moe_grouped_gemm_det_train_prepare_<op>
    CakeMoEGroupedGemmNondetLaunch                   CakeMoEGroupedGemmDetLaunch
    cake_moe_grouped_gemm_nondet                     cake_moe_grouped_gemm_det
    CakeMoEGroupedGemmNondetFunction                 CakeMoEGroupedGemmDetFunction

Every launch runs in a fixed order, so no launch alerts under
``torch.use_deterministic_algorithms(True)``.  The alert stays for an
order-varying launch (a weight gradient planned with ``plan["unordered_tail"]``
and a split tail), which only an architecture listed in the package's
``unordered_tail_archs`` (empty) plans: it would raise PyTorch's
non-determinism alert immediately before it runs (``RuntimeError``, nothing
launches; with ``warn_only=True`` a ``UserWarning``, then the launch runs);
there is never a deterministic fallback.

Numerical contract: every output (``Y``, ``dX``, ``dW`` in bf16 or fp32, and
the accumulated ``out`` / ``main_grad``) repeats bit for bit for the same
inputs, shapes, device SM count and configuration.  ``Y`` and ``dX`` are
bitwise the deterministic package's (copies of the same programs), and so
is ``dW`` wherever the 256-wide k tile runs or the 512-wide k tile does not
split its tail.  Every whole tile of ``dW`` is reduced by one CTA pair over the
group's rows in a fixed order, and the units of a split tail tile are summed
in slot order; the tail plan and group order differ from the deterministic
package's, so such a tile can differ from its bits, within its error budgets.
The operands (bf16), the accumulation (fp32), the one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the device-resident
``offs`` contract (no host synchronisation, no padding), the exact zeros of
empty groups, the error budgets and the fixes the deterministic package
carries are unchanged.  The split tail's fp32 slots and tickets are workspaces
kept per (device, CUDA stream); only the tickets rest at zero (every launch
leaves them zero), and the slots, ``tail_splits`` per tail tile, need no value
between launches.  A CUDA-graph capture takes buffers of its own.

Weight-gradient accumulation
(``cake_moe_grouped_gemm_nondet_train_wgrad(..., out=out, accumulate=True)``,
``out[e] += G[e].T @ X[e]``, and
``cake_moe_grouped_gemm_nondet(..., fuse_wgrad_accumulation=True)``, which adds
into ``w.main_grad`` in the backward; Megatron's
``gradient_accumulation_fusion``) is the package's accumulate mode (fused into
the 512-wide k tile's epilogue where served, else the overwrite into a scratch
and ``out.add_``), with the same numerical contract.

This is a local operator: on a context-parallel rank it runs on that rank's
local rows; it adds no collective, and the reduction of a parameter gradient
across ranks stays with the caller.

Usage::

    from flashinfer.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet,
        cake_moe_grouped_gemm_nondet_train_wgrad,
    )

    y = cake_moe_grouped_gemm_nondet(x, w, offs)  # dW split tail in one launch
    dw = cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out_dtype=torch.float32)

The kernels, the host planning, the autograd wrapper and the JIT registration
live in the package, which every function imports inside its body (a deferred
import), so importing this module loads none of it; the prepared launch and the
``torch.autograd.Function`` resolve from the package on first access.
:func:`is_cake_moe_grouped_gemm_nondet_train_supported` tells whether a device has
the programs of every operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from .api_logging import flashinfer_api

if TYPE_CHECKING:
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_autograd import (
        CakeMoEGroupedGemmNondetFunction,
    )
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        CakeMoEGroupedGemmNondetLaunch,
    )

__all__ = [
    "CakeMoEGroupedGemmNondetFunction",
    "CakeMoEGroupedGemmNondetLaunch",
    "cake_moe_grouped_gemm_nondet",
    "cake_moe_grouped_gemm_nondet_train_dgrad",
    "cake_moe_grouped_gemm_nondet_train_forward",
    "cake_moe_grouped_gemm_nondet_train_prepare_dgrad",
    "cake_moe_grouped_gemm_nondet_train_prepare_forward",
    "cake_moe_grouped_gemm_nondet_train_prepare_wgrad",
    "cake_moe_grouped_gemm_nondet_train_wgrad",
    "is_cake_moe_grouped_gemm_nondet_train_supported",
]

_PACKAGE = "flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train"


@flashinfer_api
def cake_moe_grouped_gemm_nondet_train_forward(
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
    The forward programs are copies of the deterministic package's and give its
    bits; they never alert under torch's deterministic mode.
    Prepares and launches in one call.  For repeated launches over the same
    tensors (or CUDA Graph capture) use ``cake_moe_grouped_gemm_nondet_train_prepare_forward`` once and
    call ``launch()`` on the returned object.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet_train_forward as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_nondet_train_dgrad(
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
    ``N % 64 == 0`` are required.  The programs are copies of the deterministic
    package's and give its bits (no alert under torch's deterministic mode).  See
    ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet_train_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_nondet_train_wgrad(
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
    dtype of ``out``, else bfloat16).  fp32 accumulation, one rounding at the
    store, exact zeros for empty groups.  With the 512-wide k tile a split tail
    is summed in slot order in the same launch, so ``dW`` repeats bit for bit
    for the same inputs, shapes, device SM count and configuration, and no
    launch alerts under ``torch.use_deterministic_algorithms(True)``.
    ``N % 256 == 0`` and ``K`` a
    multiple of the selected k tile (256, or 512 when chosen) are required.
    See ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.

    ``accumulate=True`` adds into ``out`` instead, ``out[e] += G[e].T @ X[e]``
    (Megatron's gradient-accumulation fusion; ``out`` required, bf16 or fp32,
    ``out_dtype`` None or its dtype), and returns ``out``: fused into the
    512-wide k tile's epilogue where the route serves that stage's accumulate
    build, else the overwrite into a scratch followed by ``out.add_``
    (``cake_moe_grouped_gemm_nondet_train_prepare_wgrad`` and the package README give the routes and
    the numerical contract).

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet_train_wgrad as _impl,
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
def cake_moe_grouped_gemm_nondet_train_prepare_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmNondetLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0`` (the
    512-column tile when ``N % 512 == 0`` on the architectures of
    ``wide_fwd_archs``, its L2 evict-first twin where ``fwd_lwh_ef`` or
    ``fwd_evict_first`` holds for ``K``, its 3 / 3 head / tail twin where
    ``fwd_wide_ts33`` holds).
    ``out`` may be row-padded (any row stride, unit column
    stride).  The programs are copies of the deterministic package's and give
    its bits.

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        cake_moe_grouped_gemm_nondet_train_prepare_forward as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_nondet_train_prepare_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmNondetLaunch:
    """Prepare ``dX[sum_m, K] = grouped G @ W[e]`` (W read in place; bf16 in, bf16 out).

    ``g``: ``[sum_m, N]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0`` (the
    512-column tile when ``K % 512 == 0`` on the architectures of
    ``wide_dgrad_archs``, its L2 evict-first twin where ``dgrad_evict_first``
    holds for ``K``, ``sum_m`` and ``E``, its 3 / 3 head / tail twin where
    ``dgrad_wide_ts33`` holds).  The programs are copies of the deterministic
    package's and give its bits.

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        cake_moe_grouped_gemm_nondet_train_prepare_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_api
def cake_moe_grouped_gemm_nondet_train_prepare_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    accumulate: bool = False,
) -> CakeMoEGroupedGemmNondetLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out) with this package's programs.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).

    With the 512-wide k tile a split tail is finished in the same launch by the
    ordered one-launch tail (``plan["ordered_tail"]`` and
    ``plan["tail_splits"] > 1``: each unit stores its fp32 partial to a slot of
    ``tail_sums`` and the tile's last unit sums the slots in slot order); with
    the 256-wide k tile the deterministic program and its ordered tail reduce
    run.  Either way ``dW`` repeats bit for bit for the same inputs, shapes,
    device SM count and configuration, and no launch alerts under
    ``torch.use_deterministic_algorithms(True)`` (the alert serves only an
    architecture listed in ``unordered_tail_archs``: module docstring).

    ``accumulate=True`` adds instead: every launch computes
    ``out[e] <- out[e] + G[e].T @ X[e]`` into the caller's ``out`` (required, bf16
    or fp32; ``out_dtype`` None or ``out.dtype``; rows and experts that do not
    overlap, no memory shared with ``g``, ``x`` or ``offs``; ``ValueError``
    otherwise).  Empty groups are not written by the fused route, and no row
    (``sum_m == 0``) launches nothing and leaves ``out`` unchanged.
    ``plan["accumulate"]`` is ``"fused"`` (one launch of the stage's accumulate
    build, ``stages == (stage + ACC_SUFFIX,)``) or ``"composed"`` (the overwrite
    into a scratch of ``out.dtype`` owned by the launch, then ``out.add_``;
    ``plan["accumulate_reason"]`` says why; keep the launch alive while a graph
    that captured it may replay); every other plan key is the overwrite's
    (module docstring, accumulate mode).  Each ``launch()`` and each replay of a
    graph that captured it adds again.

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        cake_moe_grouped_gemm_nondet_train_prepare_wgrad as _impl,
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
def cake_moe_grouped_gemm_nondet(
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
        ``False`` (default): the backward returns ``w``'s gradient as below.
        ``True`` (Megatron's ``gradient_accumulation_fusion``): ``w.main_grad``
        must be an fp32 or bf16 tensor of ``w``'s shape on its device with unit
        stride along the last dimension (``ValueError``, checked at the forward
        call when autograd records it: ``w`` requires grad, grad mode is on and
        the call is not a checkpoint recompute inside a backward; and again at
        the backward, which reads ``w.main_grad`` then).  Under
        ``torch.no_grad()`` or inference mode the forward needs no
        ``main_grad``.  The backward adds ``dW`` into it in place
        (``main_grad[e] += G[e].T @ X[e]``, the package's accumulate mode, fused
        into the weight-gradient kernel where the route serves it) and returns
        for ``w`` a fresh ``torch.empty`` of ``w``'s shape and dtype when ``w``
        has the attribute ``grad_added_to_main_grad`` (set to ``True``;
        ``torch.zeros`` when ``w.zero_out_wgrad`` is true), else ``None``:
        ``w.grad`` is then untouched and no gradient hook of ``w`` fires.  Every
        backward adds (``retain_graph`` twice adds twice), also a
        ``torch.autograd.grad`` whose ``inputs`` exclude ``w`` (split
        backward-weight schedules are not supported); a ``w`` that does not
        require grad is never added to.  Not supported (``NotImplementedError``)
        under ``torch.jit.trace``, ``torch.compile`` and a ``create_graph``
        backward; under a reentrant checkpoint ``w`` arrives detached, without
        ``main_grad`` (``ValueError``).

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16, the deterministic forward's bits.  ``backward``
        produces ``x.grad`` ``[sum_m, K]`` bfloat16 (the deterministic bits) and
        ``w.grad`` ``[E, N, K]`` in ``w``'s dtype (bfloat16), which repeats bit
        for bit for the same inputs, shapes, device SM count and configuration
        (a split tail of the 512-wide k tile is summed in slot order); use
        :func:`cake_moe_grouped_gemm_nondet_train_wgrad` directly for an fp32 weight gradient.  No
        launch of the forward or the backward alerts under
        ``torch.use_deterministic_algorithms(True)``; an order-varying weight-gradient
        launch, which only an architecture listed in ``unordered_tail_archs`` plans,
        would raise the non-determinism alert immediately before it runs
        (``RuntimeError``; under ``warn_only=True`` a ``UserWarning``, then it
        runs; no fallback).  The forward GEMM runs after
        autograd saved ``x``, ``w`` and ``offs``: under a non-reentrant
        checkpoint with early stop on, the recomputation skips it when these
        are the last saves of the innermost checkpointed region, as for a
        built-in op.

    Notes
    -----
    Non-deterministic only: there is no ``deterministic`` parameter (passing
    one raises ``TypeError``).  The deterministic wrapper is
    ``flashinfer.cake_moe_grouped_gemm_det_train.cake_moe_grouped_gemm_det``;
    nothing here routes to it.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_autograd import (
        cake_moe_grouped_gemm_nondet as _impl,
    )

    return _impl(x, w, offs, fuse_wgrad_accumulation=fuse_wgrad_accumulation)


@flashinfer_api
def is_cake_moe_grouped_gemm_nondet_train_supported(
    device: Optional[torch.device] = None,
) -> bool:
    """Whether the non-deterministic MoE grouped GEMM runs on ``device``.

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
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_backend,
        cake_jit,
    )

    return all(
        cake_backend.generated_program_available(device, op) for op in cake_jit.OPS
    )


# The two types of the module, resolved from the package on first access (importing this module
# imports none of it): public name -> the submodule that defines it.
_LAZY_TYPES = {
    "CakeMoEGroupedGemmNondetFunction": "cake_autograd",
    "CakeMoEGroupedGemmNondetLaunch": "cake_backend",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_TYPES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{_PACKAGE}.{submodule}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))
