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
"""

# torch.autograd wrapper of the non-deterministic ragged BF16 MoE grouped GEMM:
# the forward projection ``Y = X @ W[e].T`` with the activation gradient
# ``dX = G @ W[e]`` and the weight gradient ``dW[e] = G[e].T @ X[e]`` in
# ``backward``.  The forward and the activation gradient run this package's
# copies of the deterministic programs (the deterministic bits; a
# non-reentrant activation-checkpoint replay of the forward gives the same
# ``Y``); the weight gradient finishes a split tail in the same launch, in
# slot order, so it also repeats bit for bit for the same inputs, shapes,
# device SM count and configuration (``cake_backend``, numerical mode).
# ``CakeMoEGroupedGemmNondetFunction`` calls the
# three generated programs as plain functions; no torch custom op is
# registered.  ``cake_moe_grouped_gemm_nondet`` launches the forward after
# ``CakeMoEGroupedGemmNondetFunction.apply`` returns, so autograd saves ``x``, ``w`` and ``offs``
# before the GEMM runs, as it does for a built-in op: a non-reentrant
# checkpoint with early stop on (the torch default) then skips the recomputed
# GEMM when these are the last saves of the innermost checkpointed region.
#
# ``fuse_wgrad_accumulation=True`` follows the Megatron / Transformer Engine
# convention: the backward adds the weight gradient into ``w.main_grad`` (read
# at backward time) with the package's accumulate mode instead of returning it,
# and returns a fresh placeholder gradient for ``w`` when ``w`` carries the
# ``grad_added_to_main_grad`` attribute (``torch.zeros`` under
# ``w.zero_out_wgrad``), else none.

from __future__ import annotations

import torch

from .cake_backend import (
    _prepare_wgrad,
    _require_flag,
    cake_moe_grouped_gemm_nondet_train_prepare_dgrad,
    cake_moe_grouped_gemm_nondet_train_prepare_forward,
)

_MAIN_GRAD_DTYPES = (torch.float32, torch.bfloat16)

# -1 outside the autograd engine; inside a backward, a forward runs only as torch's
# checkpoint recompute, whose ``w`` may be an alias without ``w``'s Python attributes
# (an inner region of nested non-reentrant checkpoints).  The backward of the original
# forward reads its own ``ctx.w`` and checks ``main_grad`` there.
_GRAPH_TASK_ID = getattr(torch._C, "_current_graph_task_id", None)


def _in_backward() -> bool:
    return _GRAPH_TASK_ID is not None and _GRAPH_TASK_ID() != -1


def _fwd_op(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    cake_moe_grouped_gemm_nondet_train_prepare_forward(x, w, offs, out=out).launch()


def _dgrad_op(
    g: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    cake_moe_grouped_gemm_nondet_train_prepare_dgrad(g, w, offs, out=out).launch()


def _wgrad_op(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
) -> None:
    # An order-varying launch (none under the shipped constants) raises the deterministic-mode alert
    # right before it runs.
    _prepare_wgrad(g, x, offs, out, None, num_groups)._run(
        "CakeMoEGroupedGemmNondetFunction.backward"
    )


def _wgrad_acc_op(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    main_grad: torch.Tensor,
    num_groups: int,
) -> None:
    # ``main_grad[e] += G[e].T @ X[e]``; an order-varying launch (none under the shipped constants)
    # raises the deterministic-mode alert right before it runs (``main_grad`` is then untouched under
    # the strict mode).
    _prepare_wgrad(g, x, offs, main_grad, None, num_groups, accumulate=True)._run(
        "CakeMoEGroupedGemmNondetFunction.backward"
    )


def _main_grad(w: torch.Tensor, entry: str) -> torch.Tensor:
    """``w.main_grad``, checked: a tensor of ``w``'s shape, fp32 or bf16, on ``w``'s device, with unit stride
    along the last dimension (``ValueError`` otherwise)."""
    main_grad = getattr(w, "main_grad", None)
    if not isinstance(main_grad, torch.Tensor):
        raise ValueError(
            f"{entry}: fuse_wgrad_accumulation=True adds the weight gradient into w.main_grad, and w has "
            f"no main_grad tensor (got {type(main_grad).__name__}); a reentrant checkpoint "
            "(use_reentrant=True) passes detached copies of its inputs without it, use "
            "use_reentrant=False"
        )
    if (
        tuple(main_grad.shape) != tuple(w.shape)
        or main_grad.dtype not in _MAIN_GRAD_DTYPES
        or main_grad.device != w.device
        or main_grad.dim() != 3
        or main_grad.stride(2) != 1
    ):
        raise ValueError(
            f"{entry}: w.main_grad must be a float32 or bfloat16 tensor of w's shape "
            f"{list(w.shape)} on {w.device} with unit stride along the last dimension; got "
            f"{main_grad.dtype} {list(main_grad.shape)} on {main_grad.device}, strides "
            f"{list(main_grad.stride())}"
        )
    return main_grad


def _bf16_rows(t: torch.Tensor) -> torch.Tensor:
    """bf16 with unit stride along the last dimension, as the kernels read it."""
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


# Passed by ``cake_moe_grouped_gemm_nondet`` as the fourth argument of
# ``CakeMoEGroupedGemmNondetFunction.apply``:
# ``forward`` then allocates ``Y`` without launching, and the wrapper fills it
# after ``apply`` returns (after autograd saved the inputs).  ``forward`` accepts
# nothing else in that position but ``None``.
_DEFER_FWD = object()


def _fill_fwd(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, y: torch.Tensor
) -> None:
    """The forward GEMM into ``y`` as ``forward`` runs it: grad mode off, so no node and no saved tensor."""
    with torch.no_grad():
        _fwd_op(x, w, offs, y.detach())


class CakeMoEGroupedGemmNondetFunction(torch.autograd.Function):
    """``Y = grouped X @ W[e].T`` with ``dX`` and ``dW`` from the non-deterministic package's Cake programs.

    ``x`` ``[sum_m, K]`` bf16, ``w`` ``[E, N, K]`` bf16, ``offs`` int32 device
    end offsets ``[E]`` (or ``m_indptr`` ``[E + 1]``).  ``forward`` saves ``x``,
    ``w`` and ``offs`` only.  ``backward`` returns ``dX`` ``[sum_m, K]`` bf16 (the
    deterministic bits) and ``dW`` ``[E, N, K]`` in the dtype of ``w``, whose
    split tail (512-wide k tile) is summed in slot order in the same launch;
    empty groups give exact zeros.  Every output repeats bit for bit for the
    same inputs, shapes, device SM count and configuration, and no launch
    alerts under ``torch.use_deterministic_algorithms(True)`` (the alert serves
    only an architecture listed in ``unordered_tail_archs``: ``cake_backend``).
    Through
    :func:`cake_moe_grouped_gemm_nondet` the forward GEMM runs after autograd saved ``x``,
    ``w`` and ``offs``, so a checkpoint's early stop can skip it; a direct
    ``CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)`` launches inside ``forward``
    and a checkpoint always recomputes it, as does the wrapper under
    ``torch.jit.trace`` and ``torch.compile``.  ``apply`` takes no mode
    argument: a fourth argument other than ``None`` (such as a stale ``True`` /
    ``False`` numerical mode) raises ``TypeError``.

    The fifth argument, ``fuse_wgrad_accumulation`` (default ``False``; a bool,
    ``ValueError`` otherwise), makes the backward add the weight gradient into
    ``w.main_grad`` instead (:func:`cake_moe_grouped_gemm_nondet`; a direct call
    passes ``apply(x, w, offs, None, True)``); ``forward`` then keeps the Python object ``w``
    on ``ctx`` (a saved tensor unpacked under a checkpoint has no attributes).
    ``forward`` runs with grad mode off and cannot tell whether autograd
    records the call, so it does not check ``main_grad``:
    :func:`cake_moe_grouped_gemm_nondet` checks it before ``apply`` when autograd records
    the call, and the backward checks it again (the only check of a direct
    ``apply``).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        offs: torch.Tensor,
        _launch=None,
        fuse_wgrad_accumulation: bool = False,
    ):
        if _launch is not None and _launch is not _DEFER_FWD:
            raise TypeError(
                "CakeMoEGroupedGemmNondetFunction.apply(x, w, offs) takes no fourth argument (an internal "
                "slot: omit it, or pass None before fuse_wgrad_accumulation), got "
                f"{_launch!r}: the non-deterministic module has no deterministic argument "
                "(the deterministic autograd function is "
                "flashinfer.cake_moe_grouped_gemm_det_train.CakeMoEGroupedGemmDetFunction)"
            )
        _require_flag(
            fuse_wgrad_accumulation,
            "fuse_wgrad_accumulation",
            "CakeMoEGroupedGemmNondetFunction",
        )
        num_groups = int(w.shape[0])
        y = torch.empty(x.shape[0], w.shape[1], dtype=torch.bfloat16, device=x.device)
        if _launch is not _DEFER_FWD:
            _fwd_op(x, w, offs, y)
        ctx.save_for_backward(x, w, offs)
        ctx.num_groups = num_groups
        ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        # The parameter itself: its main_grad is read at backward time (it may be re-pointed).
        ctx.w = w if fuse_wgrad_accumulation else None
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, w, offs = ctx.saved_tensors
        fuse = ctx.fuse_wgrad_accumulation and ctx.needs_input_grad[1]
        if fuse and torch.is_grad_enabled():
            raise NotImplementedError(
                "CakeMoEGroupedGemmNondetFunction.backward: fuse_wgrad_accumulation=True adds into w.main_grad, which a "
                "double backward (create_graph=True) cannot differentiate; use "
                "fuse_wgrad_accumulation=False"
            )
        main_grad = (
            _main_grad(ctx.w, "CakeMoEGroupedGemmNondetFunction.backward")
            if fuse
            else None
        )
        # The autograd engine runs backward on a worker thread that has not
        # touched the CUDA runtime yet; the programs encode their TMA
        # descriptors through the driver API, which needs the primary context
        # current on the calling thread.  One runtime call binds it: the
        # capture-status query, which is legal while a CUDA graph captures the
        # backward, then a stream query outside a capture (a stream that is
        # being captured must not be queried: that would end the capture).
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(x.device).query()
        g = _bf16_rows(grad_y)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x, dtype=torch.bfloat16)
            _dgrad_op(g, w, offs, dx)
        if fuse:
            _wgrad_acc_op(g, x, offs, main_grad, ctx.num_groups)
            param = ctx.w
            if hasattr(param, "grad_added_to_main_grad"):
                # Megatron's convention: flag the accumulation and return a fresh placeholder, which
                # AccumulateGrad and the gradient hooks then see (never a cached tensor: AccumulateGrad
                # keeps a gradient with no other reference as w.grad).
                param.grad_added_to_main_grad = True
                fill = (
                    torch.zeros
                    if getattr(param, "zero_out_wgrad", False)
                    else torch.empty
                )
                dw = fill(param.shape, dtype=param.dtype, device=param.device)
        elif ctx.needs_input_grad[1]:
            dw = torch.empty(
                ctx.num_groups, w.shape[1], w.shape[2], dtype=w.dtype, device=w.device
            )
            _wgrad_op(g, x, offs, dw, ctx.num_groups)
        return (dx, dw) + (None,) * (len(ctx.needs_input_grad) - 2)


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
    _require_flag(
        fuse_wgrad_accumulation,
        "fuse_wgrad_accumulation",
        "cake_moe_grouped_gemm_nondet",
    )
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"x and w must be bfloat16 (got {x.dtype} and {w.dtype})")
    if torch.jit.is_tracing() or torch.compiler.is_compiling():
        if fuse_wgrad_accumulation:
            # A backward that writes a tensor the graph does not return is invisible to a trace and
            # to AOTAutograd.
            raise NotImplementedError(
                "cake_moe_grouped_gemm_nondet: fuse_wgrad_accumulation=True is not supported under torch.jit.trace "
                "or torch.compile"
            )
        # A trace records ``apply`` only, and Dynamo cannot lift the deferred
        # launch out of a checkpointed region: keep the launch inside ``forward``.
        return CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)
    # Only a call that autograd records needs ``main_grad`` before the backward, so it is
    # checked here, where grad mode is the caller's (``forward`` runs with it off): not
    # under ``no_grad`` or inference mode, and not in a checkpoint recompute inside a
    # backward, whose ``w`` may lack ``main_grad`` (the original forward's backward checks).
    if (
        fuse_wgrad_accumulation
        and w.requires_grad
        and torch.is_grad_enabled()
        and not _in_backward()
    ):
        _main_grad(w, "cake_moe_grouped_gemm_nondet")
    # ``apply`` saves x, w and offs; a checkpoint recomputation may stop inside it.
    if fuse_wgrad_accumulation:
        y = CakeMoEGroupedGemmNondetFunction.apply(x, w, offs, _DEFER_FWD, True)
    else:
        y = CakeMoEGroupedGemmNondetFunction.apply(x, w, offs, _DEFER_FWD)
    _fill_fwd(x, w, offs, y)
    return y
