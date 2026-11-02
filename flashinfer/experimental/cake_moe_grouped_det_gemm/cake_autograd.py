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

# torch.autograd wrapper of the ragged BF16 MoE grouped GEMM: the forward
# projection ``Y = X @ W[e].T`` with the activation gradient ``dX = G @ W[e]``
# and the weight gradient ``dW[e] = G[e].T @ X[e]`` in ``backward``.
# ``CakeGroupedDetMm`` calls the three generated programs as plain functions; no
# torch custom op is registered.  ``cake_grouped_det_mm`` launches the forward
# after ``CakeGroupedDetMm.apply`` returns, so autograd saves ``x``, ``w`` and
# ``offs`` before the GEMM runs, as it does for a built-in op: a non-reentrant
# checkpoint with early stop on (the torch default) then skips the recomputed
# GEMM when these are the last saves of the innermost checkpointed region.

from __future__ import annotations

from typing import Any

import torch

from .cake_backend import (
    prepare_grouped_det_gemm_dgrad,
    prepare_grouped_det_gemm_fwd,
    prepare_grouped_det_gemm_wgrad,
)


def _fwd_op(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    prepare_grouped_det_gemm_fwd(x, w, offs, out=out).launch()


def _dgrad_op(
    g: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    prepare_grouped_det_gemm_dgrad(g, w, offs, out=out).launch()


def _wgrad_op(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
) -> None:
    prepare_grouped_det_gemm_wgrad(g, x, offs, out=out, num_groups=num_groups).launch()


def _bf16_rows(t: torch.Tensor) -> torch.Tensor:
    """bf16 with unit stride along the last dimension, as the kernels read it."""
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


# Passed by ``cake_grouped_det_mm`` as the fifth argument of
# ``CakeGroupedDetMm.apply``: ``forward`` then allocates ``Y`` without launching,
# and the wrapper fills it after ``apply`` returns (after autograd saved the
# inputs).
_DEFER_FWD = object()


def _fill_fwd(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, y: torch.Tensor
) -> None:
    """The forward GEMM into ``y`` as ``forward`` runs it: grad mode off, so no node and no saved tensor."""
    with torch.no_grad():
        _fwd_op(x, w, offs, y.detach())


class CakeGroupedDetMm(torch.autograd.Function):
    """``Y = grouped X @ W[e].T`` with ``dX`` and ``dW`` from the Cake programs.

    ``x`` ``[sum_m, K]`` bf16, ``w`` ``[E, N, K]`` bf16, ``offs`` int32 device
    end offsets ``[E]`` (or ``m_indptr`` ``[E + 1]``).  ``backward`` returns
    ``dX`` ``[sum_m, K]`` bf16 and ``dW`` ``[E, N, K]`` in the dtype of ``w``;
    both reductions are bitwise deterministic and empty groups give exact
    zeros.  Through :func:`cake_grouped_det_mm` the forward GEMM runs after
    autograd saved ``x``, ``w`` and ``offs``, so a checkpoint's early stop can
    skip it; a direct ``CakeGroupedDetMm.apply(x, w, offs, deterministic)``
    launches inside ``forward`` and a checkpoint always recomputes it, as does
    the wrapper under ``torch.jit.trace`` and ``torch.compile``.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        offs: torch.Tensor,
        deterministic: bool,
        _launch=None,
    ):
        num_groups = int(w.shape[0])
        y = torch.empty(x.shape[0], w.shape[1], dtype=torch.bfloat16, device=x.device)
        if _launch is not _DEFER_FWD:
            _fwd_op(x, w, offs, y)
        ctx.save_for_backward(x, w, offs)
        ctx.num_groups = num_groups
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, w, offs = ctx.saved_tensors
        # The autograd engine runs backward on a worker thread that has not
        # touched the CUDA runtime yet; the programs encode their TMA
        # descriptors through the driver API, which needs the primary context
        # current on the calling thread.  One runtime call binds it.
        torch.cuda.current_stream(x.device).query()
        g = _bf16_rows(grad_y)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x, dtype=torch.bfloat16)
            _dgrad_op(g, w, offs, dx)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(
                ctx.num_groups, w.shape[1], w.shape[2], dtype=w.dtype, device=w.device
            )
            _wgrad_op(g, x, offs, dw, ctx.num_groups)
        return (dx, dw) + (None,) * (len(ctx.needs_input_grad) - 2)


def cake_grouped_det_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    deterministic: bool = True,
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
    deterministic : bool
        Reduction mode of the gradients.  The Cake programs always reduce in a
        fixed order (no atomics); ``False`` is accepted for API compatibility
        and currently runs the same deterministic programs.

    Returns
    -------
    torch.Tensor
        ``[sum_m, N]`` bfloat16.  ``backward`` produces ``x.grad`` ``[sum_m, K]``
        bfloat16 and ``w.grad`` ``[E, N, K]`` in ``w``'s dtype (bfloat16); use
        :func:`grouped_det_gemm_wgrad` directly for an fp32 weight gradient.
        The forward GEMM runs after autograd saved ``x``, ``w`` and ``offs``:
        under a non-reentrant checkpoint with early stop on, the recomputation
        skips it when these are the last saves of the innermost checkpointed
        region, as for a built-in op.
    """
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"x and w must be bfloat16 (got {x.dtype} and {w.dtype})")
    if torch.jit.is_tracing() or torch.compiler.is_compiling():
        # A trace records ``apply`` only, and Dynamo cannot lift the deferred
        # launch out of a checkpointed region: keep the launch inside ``forward``.
        return CakeGroupedDetMm.apply(x, w, offs, bool(deterministic))
    # ``apply`` saves x, w and offs; a checkpoint recomputation may stop inside it.
    y = CakeGroupedDetMm.apply(x, w, offs, bool(deterministic), _DEFER_FWD)
    _fill_fwd(x, w, offs, y)
    return y


# The names of the package before the deterministic-family rename (when it was
# ``flashinfer.experimental.cake_moe_grouped_gemm``), kept as deprecated
# compatibility aliases of the names above (resolved on access to the very same
# objects).
_RENAMED: dict[str, str] = {
    "CakeGroupedMm": "CakeGroupedDetMm",
    "cake_grouped_mm": "cake_grouped_det_mm",
}


def __getattr__(name: str) -> Any:
    if name in _RENAMED:
        return globals()[_RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
