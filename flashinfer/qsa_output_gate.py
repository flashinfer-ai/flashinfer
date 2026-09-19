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

import functools
from typing import Optional

import torch

from .jit.qsa_output_gate import gen_qsa_output_gate_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_qsa_output_gate_module():
    return gen_qsa_output_gate_module().build_and_load()


@register_custom_op("flashinfer::qsa_output_gate", mutates_args=("out",))
def _qsa_output_gate(
    attention: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> None:
    get_qsa_output_gate_module().qsa_output_gate(attention, gate, out)


@register_fake_op("flashinfer::qsa_output_gate")
def _qsa_output_gate_fake(
    attention: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> None:
    pass


def qsa_output_gate(
    attention: torch.Tensor,
    gate: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Scale an attention output by ``sigmoid(gate)``, elementwise.

    A gated attention layer multiplies its output by the logistic of a second
    projection before the output projection sees it. A kernel that fuses this
    into its own epilogue rounds once; a wrapper that returns the attention
    output alone leaves the caller to do it, and a chain of elementwise ops
    rounds the logistic to the output dtype on the way through and, under CUDA
    graph capture, leaves every intermediate in the graph's private pool.

    This does it in one pass: the attention value is read at the output dtype,
    the gate is widened to float, the product is formed in float, and the result
    is stored once. **When ``out`` is supplied the call allocates nothing** --
    no tensor, no workspace, and nothing that lands in a graph's private pool
    under capture. Leaving ``out`` out allocates the one output, which is the
    convenience form and not the one a captured step should use.

    Unlike the rest of FlashInfer this module is built **without**
    ``-use_fast_math``: it stands in for an expression the caller could have
    written, so it is judged against one. That keeps the accurate logistic and,
    more to the point, keeps flush-to-zero off -- with it on, a gate below about
    ``-87.3`` drives the logistic subnormal and the product collapses to zero
    instead of to a small number, which at the top of the attention range is a
    difference of thousands of representable values rather than a rounding.

    What that buys is a result within one representable value of the expression
    above, in the same order. On an ``sm_80`` build the two were measured equal
    bit for bit over every value a ``bfloat16`` gate can hold, but that is a
    measurement, not a promise across architectures and toolkits.

    ``attention`` may be taller than ``out``. A plan whose row count is fixed
    pads its batch, and only the rows ``out`` has are read, so the padding never
    has to be copied away separately. Passing ``out=attention`` scales in place,
    which is what a caller with no padding wants.

    The gate usually arrives as a view of a fused projection, so only its head
    axis is contiguous; the row and head strides are taken from the tensor.

    Parameters
    ----------
    attention : torch.Tensor
        Attention output, shape ``[rows_or_more, num_heads, head_dim]``,
        ``float16`` or ``bfloat16``, contiguous along ``head_dim``.
    gate : torch.Tensor
        Gate to apply, shape ``[rows, num_heads, head_dim]``, same dtype,
        contiguous along ``head_dim``. Left unchanged.
    out : Optional[torch.Tensor]
        Where to write, shape ``[rows, num_heads, head_dim]``. Allocated when
        omitted; pass ``attention`` to scale in place.

    Returns
    -------
    torch.Tensor
        ``out``.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> attention = torch.full((1, 1, 2), 2.0, dtype=torch.bfloat16, device="cuda")
    >>> gate = torch.zeros(1, 1, 2, dtype=torch.bfloat16, device="cuda")
    >>> flashinfer.qsa_output_gate(attention, gate)
    tensor([[[1., 1.]]], device='cuda:0', dtype=torch.bfloat16)
    """
    if attention.ndim != 3:
        raise ValueError(
            f"attention must be 3D [rows, num_heads, head_dim], got {attention.ndim}D"
        )
    if gate.ndim != 3:
        raise ValueError(
            f"gate must be 3D [rows, num_heads, head_dim], got {gate.ndim}D"
        )
    if out is None:
        out = torch.empty_like(gate)
    if out.shape != gate.shape:
        raise ValueError(
            f"out must have the gate's shape {tuple(gate.shape)}, got {tuple(out.shape)}"
        )
    if out.shape[0]:
        _qsa_output_gate(attention, gate, out)
    return out
