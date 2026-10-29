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

Packed-varlen causal GQA attention training with head dimension 256
(SM100 / SM103 / SM107), the Cake backend.

Thin experimental entry points.  The checks, the launch plan, the workspace,
the autograd Function and the generated kernels live in
``flashinfer.experimental.cake_gqa_train``; this module forwards every
keyword.  Every switch is a keyword argument; no environment variable is read.

Operands: ``q`` BF16 ``[T_q, H_q, 256]``, ``k`` / ``v`` BF16
``[T_k, H_kv, 256]`` with ``H_q / H_kv`` even (unit head-dim stride, 16-byte
aligned token / head strides: strided views pass without a copy),
``cu_seqlens_q`` / ``cu_seqlens_k`` int32 ``[B + 1]`` on the device (never
read on the host).  Segment ``b`` has ``q_len`` query rows and ``k_len`` keys;
query row ``i`` sees keys ``j <= i + k_len - q_len`` (FlashAttention's
bottom-right alignment), so a row with ``i + k_len - q_len < 0`` sees no key
and gets ``out = 0``, ``lse = -inf`` and zero gradients.  ``T_q``, ``T_k``,
``B`` and every length are run-time values: no recompilation, no
device-to-host read, no device-wide synchronization, no cooperative launch.

Usage::

    from flashinfer.gqa_train import varlen_gqa_attention, gqa_train_workspace

    ws = gqa_train_workspace(device)                     # optional: reused across calls and shapes
    out = varlen_gqa_attention(q, k, v, cu_q, cu_k, softmax_scale=0.0625, workspace=ws)
    out.backward(dout)                                   # BF16 dq, dk, dv

    # FP32 dK / dV added into caller-owned destinations through a row map, produced at backward time:
    out = varlen_gqa_attention(q, k, v, cu_q, cu_k, dkv_acc=lambda: (dk_dst, dv_dst, row_map))
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake packed-varlen HD256 GQA training attention (SM100/SM103/SM107)"


def _pkg():
    from .experimental import cake_gqa_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gqa_train


@flashinfer_experimental_api(feature=_FEATURE)
def varlen_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dkv_acc: Optional[Callable[[], tuple]] = None,
    max_seqlen_k: Optional[int] = None,
    workspace: Optional[Any] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Packed-varlen causal GQA attention, head dimension 256, with autograd.

    Parameters
    ----------
    q, k, v : torch.Tensor
        BF16 ``[T_q, H_q, 256]`` and ``[T_k, H_kv, 256]``; query head ``h`` reads key head ``h / (H_q / H_kv)``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[B + 1]`` on the device, ``cu[0] = 0``, non-decreasing, ``cu_q[B] = T_q``, ``cu_k[B] = T_k``.
    softmax_scale : float, optional
        Default ``256 ** -0.5``.
    causal : bool
        Must be ``True`` (bottom-right alignment per segment).
    dkv_acc : callable, optional
        ``None``: ``k`` / ``v`` get BF16 gradients (FP32 sums over the query heads rounded once).  A zero-argument
        callable, called in backward, returning ``(dk_acc, dv_acc, row_map)`` adds the FP32 window gradients into
        the caller's FP32 ``[N_dst, H_kv, 256]`` destinations (window row ``j`` into row ``row_map[j]``, int32
        ``[T_k]``, repeated rows accumulate), or returning ``(dk_win, dv_win)`` stores the FP32 window gradients
        ``[T_k, H_kv, 256]`` into them; ``k`` / ``v`` then get no gradient.  Nothing is held for the backward.
    max_seqlen_k : int, optional
        A host int at least every segment's key count (the caller's contract, as FlashAttention's
        ``max_seqlen_k``; a smaller value leaves key blocks out); it sizes the backward grid.  ``None`` bounds it by
        ``T_k`` (always correct; more CTAs that retire at once).  Never read from the device.
    workspace : optional
        A workspace of :func:`gqa_train_workspace`, reused across calls and shapes (grown, never shrunk; one stream
        at a time); autograd holds it by weak reference.  ``None``: the device's per-stream default workspace.
    return_lse : bool
        Also return the FP32 natural-log LSE ``[H_q, T_q]`` (no gradient flows through it).

    Returns
    -------
    ``out`` BF16 ``[T_q, H_q, 256]``, or ``(out, lse)``.
    """
    return _pkg().varlen_gqa_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        dkv_acc=dkv_acc,
        max_seqlen_k=max_seqlen_k,
        workspace=workspace,
        return_lse=return_lse,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def varlen_gqa_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    workspace: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward without autograd: ``(out [T_q, H_q, 256] bf16, lse [H_q, T_q] fp32)``; ``out`` / ``lse`` may be
    given (contiguous)."""
    return _pkg().gqa_attention_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
        out=out,
        lse=lse,
        workspace=workspace,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def varlen_gqa_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dkv_dtype: torch.dtype = torch.bfloat16,
    dk_acc: Optional[torch.Tensor] = None,
    dv_acc: Optional[torch.Tensor] = None,
    row_map: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    workspace: Optional[Any] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Backward without autograd: ``(dq, dk, dv)`` from the forward's ``out`` and ``lse`` and a contiguous ``dout``.

    ``dq`` is BF16.  dK / dV are FP32 sums over the query heads of each key head, returned as BF16
    (``dkv_dtype=torch.bfloat16``), as FP32 window gradients (``dkv_dtype=torch.float32``) or, with ``row_map``,
    added into the FP32 destinations ``dk_acc`` / ``dv_acc`` ``[N_dst, H_kv, 256]`` (returned as ``dk`` / ``dv``).
    ``max_seqlen_k``: see :func:`varlen_gqa_attention`.
    """
    return _pkg().gqa_attention_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
        dq=dq,
        dk=dk,
        dv=dv,
        dkv_dtype=dkv_dtype,
        dk_acc=dk_acc,
        dv_acc=dv_acc,
        row_map=row_map,
        max_seqlen_k=max_seqlen_k,
        workspace=workspace,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def gqa_train_workspace(device: Union[torch.device, str, int, None] = None) -> Any:
    r"""A workspace for :func:`varlen_gqa_attention` on ``device``: the forward's tile counter and the backward
    scratch, grown to the largest shape seen (``.nbytes``; ``.release()`` drops the scratch)."""
    return _pkg().Workspace(device)


@flashinfer_experimental_api(feature=_FEATURE)
def gqa_train_workspace_bytes(total_q: int, num_heads_q: int, num_segments: int) -> int:
    r"""Bytes of backward scratch one call needs: FP32 ``[H_q / 2, rows_pad, 256]`` plus two ``[H_q / 2,
    rows_pad]``, ``rows_pad = round_up(2 T_q + 128 B, 128)``."""
    return _pkg().workspace_bytes(total_q, num_heads_q, num_segments)


__all__ = [
    "gqa_train_workspace",
    "gqa_train_workspace_bytes",
    "varlen_gqa_attention",
    "varlen_gqa_attention_bwd",
    "varlen_gqa_attention_fwd",
]
