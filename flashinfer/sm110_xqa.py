# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental native SM110 XQA attention entry points."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

__all__ = ["prepare", "attention"]


@flashinfer_experimental_api
def prepare(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    q_cu_seq_lens: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    sm_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    partition_tokens: Optional[int] = None,
) -> Any:
    """Prepare exact-SM110a attention and return a backend PreparedAttention.

    D128 supports one FP16 decode query with contiguous KV. D512 supports
    uniform or packed tree queries with FP16/E4M3 contiguous or page128 KV.
    Preparation validates metadata, compiles and may allocate output/workspace;
    call it before timing or graph capture. The returned plan's ``run()`` uses
    the caller's current CUDA stream. See the SM110 XQA backend README for the
    tensor layouts, device-metadata preconditions and workspace ordering rules.
    """
    from .experimental.sm110_xqa.backend import prepare as backend_prepare

    return backend_prepare(
        q,
        kv,
        sequence_lengths,
        mask=mask,
        out=out,
        page_table=page_table,
        page_size=page_size,
        q_cu_seq_lens=q_cu_seq_lens,
        max_q_len=max_q_len,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        workspace=workspace,
        partition_tokens=partition_tokens,
    )


@flashinfer_experimental_api
def attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Prepare and execute SM110 attention once, returning the FP16 output.

    Keyword arguments match :func:`prepare`. Reuse ``prepare(...).run()`` for
    prepared replay without tensor allocation.
    """
    from .experimental.sm110_xqa.backend import attention as backend_attention

    return backend_attention(q, kv, sequence_lengths, **kwargs)
