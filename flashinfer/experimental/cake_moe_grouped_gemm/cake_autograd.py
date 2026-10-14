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
# and the weight gradient ``dW[e] = G[e].T @ X[e]`` in ``backward``.  The
# three generated programs are also registered as FlashInfer custom ops with
# fake (meta) implementations so tracing tools see their output shapes.

from __future__ import annotations

import torch

from ...api_logging import flashinfer_experimental_api
from ...utils import register_custom_op, register_fake_op
from .cake_backend import (
    prepare_grouped_gemm_dgrad,
    prepare_grouped_gemm_fwd,
    prepare_grouped_gemm_wgrad,
)

_OP_PREFIX = "flashinfer::cake_moe_grouped_gemm_"


@register_custom_op(_OP_PREFIX + "fwd", mutates_args=("out",))
def _fwd_op(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    prepare_grouped_gemm_fwd(x, w, offs, out=out).launch()


@register_fake_op(_OP_PREFIX + "fwd")
def _fwd_fake(
    x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    pass


@register_custom_op(_OP_PREFIX + "dgrad", mutates_args=("out",))
def _dgrad_op(
    g: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    prepare_grouped_gemm_dgrad(g, w, offs, out=out).launch()


@register_fake_op(_OP_PREFIX + "dgrad")
def _dgrad_fake(
    g: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, out: torch.Tensor
) -> None:
    pass


@register_custom_op(_OP_PREFIX + "wgrad", mutates_args=("out",))
def _wgrad_op(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
) -> None:
    prepare_grouped_gemm_wgrad(g, x, offs, out=out, num_groups=num_groups).launch()


@register_fake_op(_OP_PREFIX + "wgrad")
def _wgrad_fake(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
) -> None:
    pass


def _bf16_rows(t: torch.Tensor) -> torch.Tensor:
    """bf16 with unit stride along the last dimension, as the kernels read it."""
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t


class CakeGroupedMm(torch.autograd.Function):
    """``Y = grouped X @ W[e].T`` with ``dX`` and ``dW`` from the Cake programs.

    ``x`` ``[sum_m, K]`` bf16, ``w`` ``[E, N, K]`` bf16, ``offs`` int32 device
    end offsets ``[E]`` (or ``m_indptr`` ``[E + 1]``).  ``backward`` returns
    ``dX`` ``[sum_m, K]`` bf16 and ``dW`` ``[E, N, K]`` in the dtype of ``w``;
    both reductions are bitwise deterministic and empty groups give exact
    zeros.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, deterministic: bool
    ):
        num_groups = int(w.shape[0])
        y = torch.empty(x.shape[0], w.shape[1], dtype=torch.bfloat16, device=x.device)
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
        return dx, dw, None, None


@flashinfer_experimental_api(feature="cake_moe_grouped_gemm.cake_grouped_mm")
def cake_grouped_mm(
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
        :func:`grouped_gemm_wgrad` directly for an fp32 weight gradient.
    """
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"x and w must be bfloat16 (got {x.dtype} and {w.dtype})")
    return CakeGroupedMm.apply(x, w, offs, bool(deterministic))
