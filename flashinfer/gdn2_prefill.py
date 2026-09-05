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

from typing import Literal, Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .trace.templates.gdn2 import gdn2_prefill_trace

_CU_SEQLENS_DTYPES = (torch.int32, torch.int64)


@flashinfer_api(trace=gdn2_prefill_trace)
def chunk_gated_delta_rule2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    *,
    backend: Literal["auto", "cudnn"] = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule 2 (GDN-2) attention for prefill.

    GDN-2 generalizes :func:`flashinfer.chunk_gated_delta_rule`'s per-head
    scalar gates to channel-wise ones: the forget gate ``g`` and the erase gate
    ``beta`` are per key channel, and a third gate ``w`` scales the incoming
    value per value channel.

    .. math::

        S_t &= \mathrm{diag}(g_t) S_{t-1} \\
        v^{new}_t &= w_t \odot v_t - (\beta_t \odot k_t)^\top S_t \\
        S_t &\mathrel{+}= k_t \otimes v^{new}_t \\
        o_t &= \mathrm{scale} \cdot q_t^\top S_t

    Setting ``g``, ``beta`` and ``w`` to per-channel constants recovers GDN:
    :math:`w \equiv \beta` and a channel-constant :math:`g` make the update the
    scalar-gated delta rule.

    Only varlen (packed / THD) input is accepted, matching
    :func:`flashinfer.chunk_gated_delta_rule`: pass ``cu_seqlens`` and lay the
    tokens of all sequences end to end.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``[total_seq_len, num_q_heads / num_k_heads / num_v_heads, head_size]``,
        float16 or bfloat16.  ``num_k_heads`` must equal ``num_q_heads`` or
        ``num_v_heads``, and the query and value head counts must be equal or
        one a multiple of the other.  Strides are honored, so only the innermost
        dimension has to be contiguous.
    g : torch.Tensor, optional
        Channel-wise forget gate in linear space (``alpha``, elementwise in
        ``(0, 1]``), shape ``[total_seq_len, num_sab_heads, head_size]``, at
        float32, bfloat16 or float16.  All-ones (no decay) when ``None``.
    beta : torch.Tensor, optional
        Channel-wise erase gate ``[total_seq_len, num_sab_heads, head_size]``,
        post-sigmoid, read at ``q.dtype``.  All-ones when ``None``.
    w : torch.Tensor, optional
        Channel-wise write gate ``[total_seq_len, num_sab_heads, head_size]``,
        read at ``q.dtype``.  All-ones when ``None``.
    scale : float, optional
        Query scale; ``1 / sqrt(head_size)`` when ``None`` or ``0.0``, matching
        :func:`flashinfer.chunk_gated_delta_rule`.
    initial_state : torch.Tensor, optional
        Incoming recurrent state ``[num_seqs, num_sab_heads, head_size,
        head_size]``, V-major (``[..., V, K]``), float32 or bfloat16.  A zero
        state when ``None``.
    output_final_state : bool
        Return the outgoing recurrent state alongside the output.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths ``[num_seqs + 1]``, int32 or int64.
        Required.
    use_qk_l2norm_in_kernel : bool
        Fuse the q/k L2 normalization into the kernel instead of expecting
        pre-normalized inputs.
    output : torch.Tensor, optional
        Pre-allocated output ``[total_seq_len, num_o_heads, head_size]``,
        written in place.  Allocated internally when ``None``.
    output_state : torch.Tensor, optional
        Pre-allocated final-state buffer, written in place.  Allocated
        internally when ``None`` and ``output_final_state`` is set.  It must
        not alias ``initial_state``; the kernel splits one sequence across
        CTAs, so the CTA reading the incoming state would race the one writing
        the outgoing state.
    backend : Literal["auto", "cudnn"], optional
        FlashInfer carries no GDN-2 kernel of its own, so ``"auto"`` (default)
        and ``"cudnn"`` both run cuDNN's fused SM100 linear-attention engine
        through :func:`flashinfer.cudnn.cudnn_chunk_gated_delta_rule2`.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``output``, or ``(output, final_state)`` when ``output_final_state``.

    Note
    ----
    Requires an SM100-family (Blackwell) device and cudnn-frontend 1.28+ with
    the ``cutedsl`` extra (``pip install 'nvidia-cudnn-frontend[cutedsl]'``).
    Everything finer -- head dims, input dtypes, head-count relations -- is the
    engine's call: a graph it cannot serve is declined by cuDNN (the per-engine
    reason lands in the frontend's log).
    """
    if backend not in ("auto", "cudnn"):
        raise ValueError(f'backend must be "auto" or "cudnn", got {backend!r}')
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for varlen mode")
    if cu_seqlens.dtype not in _CU_SEQLENS_DTYPES:
        raise ValueError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    from .cudnn import cudnn_chunk_gated_delta_rule2

    return cudnn_chunk_gated_delta_rule2(
        q,
        k,
        v,
        g,
        beta,
        w,
        scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
        output_state=output_state,
    )
