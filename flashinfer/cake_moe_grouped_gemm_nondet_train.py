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

The functions of this module carry the experimental-API marking and hand off
to the backend package
:mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train` with the
same arguments: the weight gradient with the 512-wide k tile adds its split
tail (the tiles of its last partial wave, cut into k-step units over the
clusters) in place with unordered fp32 reductions, in one launch.

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

Under ``torch.use_deterministic_algorithms(True)`` an order-varying launch
raises PyTorch's non-determinism alert immediately before it runs
(``RuntimeError``, nothing launches; with ``warn_only=True`` a ``UserWarning``,
then the non-deterministic launch runs); there is never a deterministic
fallback.  An order-varying launch is the weight gradient whose 512-wide k
tile splits its tail (``plan["unordered_tail"]`` and
``plan["tail_splits"] > 1``), in overwrite or accumulate mode:
:func:`cake_moe_grouped_gemm_nondet_train_wgrad`, ``launch()`` of such a
prepared weight gradient, and the backward of
:func:`cake_moe_grouped_gemm_nondet` when it computes ``dW``.  The forward
pass, the data gradient, the 256-wide k tile's weight gradient, an unsplit
tail and every preparation run in a fixed order and never alert.

Numerical contract: only the run-to-run bitwise reproducibility of the weight
gradient ``dW`` is relaxed.  ``Y`` and ``dX`` are bitwise the deterministic
package's (copies of the same programs; on sm_100a the data gradient's L2
evict-first twin is the package's own build of the deterministic wide program,
whose cache policy changes no arithmetic), and so is ``dW`` wherever the
256-wide k tile runs or the 512-wide k tile does not split its tail.  Every
whole tile of ``dW`` is reduced by one CTA pair over the group's rows in a
fixed order; only the units of a split tail tile are added in an order the
launch does not fix.  The operands (bf16), the accumulation (fp32), the one
round-to-nearest rounding to bf16 (or the fp32 store), the layouts, the
device-resident ``offs`` contract (no host synchronisation, no padding), the
exact zeros of empty groups, the error budgets and the fixes the deterministic
package carries are unchanged.  The split tail's fp32 sums and tickets are
workspaces kept per (device, CUDA stream) at rest at zero (every launch leaves
them zero); a CUDA-graph capture takes zeroed buffers of its own.

Weight-gradient accumulation
(``cake_moe_grouped_gemm_nondet_train_wgrad(..., out=out, accumulate=True)``,
``out[e] += G[e].T @ X[e]``, and
``cake_moe_grouped_gemm_nondet(..., fuse_wgrad_accumulation=True)``, which adds
into ``w.main_grad`` in the backward; Megatron's
``gradient_accumulation_fusion``) is the package's accumulate mode (fused into
the 512-wide k tile's epilogue where served, else the overwrite into a scratch
and ``out.add_``), with the same alert rule.

This is a local operator: on a context-parallel rank it runs on that rank's
local rows; it adds no collective, and the reduction of a parameter gradient
across ranks stays with the caller.

Usage::

    from flashinfer.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet,
        cake_moe_grouped_gemm_nondet_train_wgrad,
    )

    y = cake_moe_grouped_gemm_nondet(x, w, offs)  # unordered dW split tail
    dw = cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out_dtype=torch.float32)

The kernels, the host planning, the autograd wrapper and the JIT registration
live in the package, which every function imports inside its body (a deferred
import), so importing this module loads none of it; the prepared launch and the
``torch.autograd.Function`` resolve from the package on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

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
]

_PACKAGE = "flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train"


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_forward"
)
def cake_moe_grouped_gemm_nondet_train_forward(
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


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_dgrad"
)
def cake_moe_grouped_gemm_nondet_train_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` for every group ``e``.

    ``g`` is ``[sum_m, N]`` bfloat16, ``w`` is ``[E, N, K]`` bfloat16 and is
    read in place (no transpose copy), ``offs`` is ``[E]`` int32 device end
    offsets (or ``m_indptr`` ``[E + 1]``) and ``out`` is ``[sum_m, K]`` bfloat16.  ``K % 256 == 0`` and
    ``N % 64 == 0`` are required.  The programs are copies of the deterministic
    package's, but for the sm_100a evict-first twin, this package's own build of
    the deterministic wide tile, and all give its bits (no alert under torch's
    deterministic mode).  See
    ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet_train_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_wgrad"
)
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

    ``g`` is ``[sum_m, N]`` bfloat16, ``x`` is ``[sum_m, K]`` bfloat16,
    ``offs`` is ``[E]`` int32 device end offsets (``E = offs.numel()``; pass
    ``num_groups=E`` to hand in ``m_indptr`` ``[E + 1]`` instead) and ``out``
    is ``[E, N, K]`` in ``out_dtype`` (bfloat16 or float32; defaults to the
    dtype of ``out``, else bfloat16).  fp32 accumulation, one rounding at the
    store, exact zeros for empty groups.  With the 512-wide k tile a split tail
    is added with unordered fp32 reductions, so ``dW`` is not bitwise
    reproducible run to run, and under
    ``torch.use_deterministic_algorithms(True)`` such a launch raises the
    non-determinism alert immediately before it runs (a fixed-order launch does
    not).  ``N % 256 == 0`` and ``K`` a
    multiple of the selected k tile (256, or 512 when chosen) are required.
    See ``cake_moe_grouped_gemm_nondet_train_forward`` for the prepared form.

    ``accumulate=True`` adds into ``out`` instead, ``out[e] += G[e].T @ X[e]``
    (Megatron's gradient-accumulation fusion; ``out`` required, bf16 or fp32,
    ``out_dtype`` None or its dtype), and returns ``out``: fused into the
    512-wide k tile's epilogue where the route serves that stage's accumulate
    build, else the overwrite into a scratch followed by ``out.add_``
    (``cake_moe_grouped_gemm_nondet_train_prepare_wgrad`` and the package README give the routes and
    the numerical contract).
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


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_prepare_forward"
)
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
    ``wide_fwd_archs``, its 3 / 3 head / tail twin where ``fwd_wide_ts33`` holds).
    ``out`` may be row-padded (any row stride, unit column
    stride).  The programs are copies of the deterministic package's and give
    its bits.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        cake_moe_grouped_gemm_nondet_train_prepare_forward as _impl,
    )

    return _impl(x, w, offs, out=out)


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_prepare_dgrad"
)
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
    package's, but for the sm_100a evict-first twin, this package's own build of
    the deterministic wide tile, and all give the deterministic package's bits.
    """
    from .experimental.cake.cake_moe_grouped_gemm_nondet_train.cake_backend import (
        cake_moe_grouped_gemm_nondet_train_prepare_dgrad as _impl,
    )

    return _impl(g, w, offs, out=out)


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet_train_prepare_wgrad"
)
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
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), non-deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).

    With the 512-wide k tile a split tail is added in place with unordered fp32
    reductions
    (``plan["unordered_tail"]`` and ``plan["tail_splits"] > 1``, the launch's
    ``order_varying``), so ``dW`` is not bitwise reproducible run to run; with
    the 256-wide k tile the deterministic program and its ordered tail reduce
    run.  The preparation never alerts; under
    ``torch.use_deterministic_algorithms(True)`` an order-varying launch raises
    the non-determinism alert at each ``launch()`` (module docstring).

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


@flashinfer_experimental_api(
    feature="cake_moe_grouped_gemm_nondet_train.cake_moe_grouped_gemm_nondet"
)
def cake_moe_grouped_gemm_nondet(
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
        ``w.grad`` ``[E, N, K]`` in ``w``'s dtype (bfloat16), not bitwise
        reproducible run to run where the 512-wide k tile splits its tail; use
        :func:`cake_moe_grouped_gemm_nondet_train_wgrad` directly for an fp32 weight gradient.  Under
        ``torch.use_deterministic_algorithms(True)`` the backward raises the
        non-determinism alert immediately before a weight-gradient launch that
        adds a split tail with unordered reductions (``RuntimeError``; under
        ``warn_only=True`` a ``UserWarning``, then it runs; no fallback); the
        forward, ``dX`` and a fixed-order ``dW`` do not alert.  The forward GEMM runs after
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
