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

Cake deterministic packed-varlen causal GQA attention training with head
dimension 256 (SM100 / SM103 / SM107).

Entries: :func:`cake_gqa_det` (with autograd),
:func:`cake_gqa_det_train_forward` / :func:`cake_gqa_det_train_backward`
(the two halves without autograd), :func:`cake_gqa_det_train_workspace` and
:func:`cake_gqa_det_train_workspace_bytes`.  The autograd class
``CakeGQADetFunction``, the workspace type ``CakeGQADetWorkspace`` and
the ``dkv_acc`` callable type ``CakeGQADetDkvAccSource`` resolve to the
backend's objects on first access.

The mode is part of the name: no entry takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  Every sum of the forward and the backward
has one fixed order: ``out``, ``lse``, dQ and every dK / dV form (BF16, FP32
window, FP32 row-map adds with repeated destination rows) repeat bit for bit
run to run, for one shape, one ``max_seqlen_k``, one workspace budget and the
same host boundaries (given or not).
The backward's main kernel sums dK and dV of each key block in tensor
memory and stores dS^T into a workspace; dQ is a GEMM over that workspace.
The backward runs in chunks that fit the workspace's ``budget_bytes`` (default
16 GiB; the smallest chunk, one KV head and two key blocks, runs even past
it); it raises no PyTorch determinism alert.  The interface matches the
non-deterministic counterpart ``flashinfer.cake_gqa_nondet_train``.

A stable API (``@flashinfer_api``; no ``ExperimentalWarning``) of thin
wrappers: the checks, the launch plan, the workspace, the autograd Function
and the generated kernels live in the backend package
``flashinfer.experimental.cake.cake_gqa_det_train``, imported on first use;
this module forwards every keyword.  Every switch is a keyword argument; no
environment variable is read.

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

    from flashinfer.cake_gqa_det_train import cake_gqa_det, cake_gqa_det_train_workspace

    ws = cake_gqa_det_train_workspace(device)         # optional: reused across calls and shapes
    out = cake_gqa_det(q, k, v, cu_q, cu_k, softmax_scale=0.0625, workspace=ws)
    out.backward(dout)                                   # BF16 dq, dk, dv

    # FP32 dK / dV added into caller-owned destinations through a row map, produced at backward time:
    out = cake_gqa_det(q, k, v, cu_q, cu_k, dkv_acc=lambda: (dk_dst, dv_dst, row_map))
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

from .api_logging import flashinfer_api

if (
    TYPE_CHECKING
):  # resolved at run time by ``__getattr__`` (no module-level backend import)
    from .experimental.cake.cake_gqa_det_train import (
        CakeGQADetDkvAccSource,
        CakeGQADetFunction,
        CakeGQADetWorkspace,
    )

__all__ = [
    "CakeGQADetDkvAccSource",
    "CakeGQADetFunction",
    "CakeGQADetWorkspace",
    "cake_gqa_det",
    "cake_gqa_det_train_backward",
    "cake_gqa_det_train_forward",
    "cake_gqa_det_train_workspace",
    "cake_gqa_det_train_workspace_bytes",
]

# Backend objects this module re-exports, resolved on first access (``__getattr__``).
_BACKEND_OBJECTS = frozenset(
    {"CakeGQADetDkvAccSource", "CakeGQADetFunction", "CakeGQADetWorkspace"}
)


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use (experimental containment: core reaches the kernels through a
    # deferred import).
    from .experimental.cake import cake_gqa_det_train  # noqa: PLC0415

    return cake_gqa_det_train


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_pkg(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BACKEND_OBJECTS)


@flashinfer_api
def cake_gqa_det(
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
    cu_seqlens_q_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
    cu_seqlens_k_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    r"""Packed-varlen causal GQA attention, head dimension 256, with autograd (``CakeGQADetFunction``).

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T_q, H_q, 256]`` queries.
    k, v : torch.Tensor
        BF16 ``[T_k, H_kv, 256]`` keys and values; query head ``h`` reads key head ``h / (H_q / H_kv)``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[B + 1]`` on the device, ``cu[0] = 0``, non-decreasing, ``cu_q[B] = T_q``, ``cu_k[B] = T_k``.
    softmax_scale : float, optional
        Default ``256 ** -0.5``.
    causal : bool
        Must be ``True`` (bottom-right alignment per segment).
    dkv_acc : callable, optional
        ``None``: ``k`` / ``v`` get BF16 gradients (FP32 sums over the query heads rounded once).  A zero-argument
        callable (``CakeGQADetDkvAccSource``), called in backward, returning ``(dk_acc, dv_acc, row_map)`` adds
        the FP32 window gradients into the caller's FP32 ``[N_dst, H_kv, 256]`` destinations (window row ``j`` into
        row ``row_map[j]``, int32 ``[T_k]``, repeated rows accumulate), or returning ``(dk_win, dv_win)`` stores the
        FP32 window gradients ``[T_k, H_kv, 256]`` into them; ``k`` / ``v`` then get no gradient.  Nothing is held
        for the backward.
    max_seqlen_k : int, optional
        A host int at least every segment's key count (the caller's contract, as FlashAttention's
        ``max_seqlen_k``; a smaller value leaves key blocks out); it sizes the backward grid.  ``None`` bounds it by
        ``T_k`` (always correct; more CTAs that retire at once).  Never read from the device.
    workspace : optional
        A workspace of :func:`cake_gqa_det_train_workspace`, reused across calls and shapes (grown, never
        shrunk; one stream at a time); autograd holds it by weak reference.  ``None``: the device's per-stream
        default workspace.
    return_lse : bool
        Also return the FP32 natural-log LSE ``[H_q, T_q]`` (no gradient flows through it).
    cu_seqlens_q_host, cu_seqlens_k_host : sequence of int or torch.Tensor, optional
        Host copies of ``cu_seqlens_q`` / ``cu_seqlens_k`` (lists or 1-D CPU tensors; both or neither; never copied
        to the device).  With them the backward plans its dS^T workspace chunks from each key block's exact tile
        count: fewer, fuller chunks on long layouts and a workspace of the exact size; without them it bounds the
        counts from ``T_q`` and ``B``.  The kernels read the device boundaries: host boundaries that disagree only
        move the chunks (and the dQ rounding) while every device key segment fits the host plan's key blocks and
        every chunk holds its device tiles; otherwise a device-side assert fires before the main kernel runs
        (reported at the next synchronization; the CUDA context is unusable after it).  The dQ bits depend on them.

    Returns
    -------
    ``out`` BF16 ``[T_q, H_q, 256]``, or ``(out, lse)``.

    Notes
    -----
    The forward and the backward are fixed-order: ``out``, ``lse`` and every gradient repeat bit for bit run to run
    (one shape, one ``max_seqlen_k``, one workspace budget, the same host boundaries); neither raises PyTorch's
    determinism alert.
    """
    return _pkg().cake_gqa_det(
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
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
    )


@flashinfer_api
def cake_gqa_det_train_forward(
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
    r"""Forward without autograd, causal with bottom-right alignment per segment.

    Parameters
    ----------
    q, k, v : torch.Tensor
        As in :func:`cake_gqa_det`.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        As in :func:`cake_gqa_det`.
    softmax_scale : float, optional
        Default ``256 ** -0.5``.
    out : torch.Tensor, optional
        A contiguous BF16 ``[T_q, H_q, 256]`` output to write; ``None`` allocates one.
    lse : torch.Tensor, optional
        A contiguous FP32 ``[H_q, T_q]`` natural-log LSE to write; ``None`` allocates one.
    workspace : optional
        As in :func:`cake_gqa_det`.

    Returns
    -------
    ``(out, lse)``.  A row that sees no key gets ``out = 0`` and ``lse = -inf``.

    Notes
    -----
    Fixed-order: it never raises the deterministic-algorithms alert.
    """
    return _pkg().cake_gqa_det_train_forward(
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


@flashinfer_api
def cake_gqa_det_train_backward(
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
    cu_seqlens_q_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
    cu_seqlens_k_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Backward without autograd: ``(dq, dk, dv)`` from the forward's ``out`` and ``lse``.

    dK / dV are FP32 sums over the query heads of each key head.

    Parameters
    ----------
    q, k, v : torch.Tensor
        The forward's operands (see :func:`cake_gqa_det`).
    out : torch.Tensor
        The forward's BF16 ``[T_q, H_q, 256]`` output.
    lse : torch.Tensor
        The forward's FP32 ``[H_q, T_q]`` LSE.
    dout : torch.Tensor
        Contiguous BF16 ``[T_q, H_q, 256]`` gradient of ``out``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        As in :func:`cake_gqa_det`.
    softmax_scale : float, optional
        The forward's scale; default ``256 ** -0.5``.
    dq : torch.Tensor, optional
        A contiguous BF16 ``[T_q, H_q, 256]`` to write; ``None`` allocates one.
    dk, dv : torch.Tensor, optional
        Contiguous ``[T_k, H_kv, 256]`` tensors of ``dkv_dtype`` to write (overwritten); ``None`` allocates them.
        Not used with ``row_map``.
    dkv_dtype : torch.dtype
        ``torch.bfloat16`` (default): dK / dV rounded once to BF16.  ``torch.float32``: the FP32 window
        gradients are stored.
    dk_acc, dv_acc : torch.Tensor, optional
        FP32 ``[N_dst, H_kv, 256]`` destinations that the window gradients are added into; they need ``row_map``.
    row_map : torch.Tensor, optional
        int32 ``[T_k]``: window row ``j`` adds into destination row ``row_map[j]`` (rows may repeat;
        out-of-range rows are not checked).
    max_seqlen_k : int, optional
        As in :func:`cake_gqa_det`.
    workspace : optional
        As in :func:`cake_gqa_det`.
    cu_seqlens_q_host, cu_seqlens_k_host : sequence of int or torch.Tensor, optional
        As in :func:`cake_gqa_det`.

    Returns
    -------
    ``(dq, dk, dv)``; with ``row_map``, ``dk`` / ``dv`` are ``dk_acc`` / ``dv_acc``.

    Notes
    -----
    Bitwise reproducible run to run for one shape, one ``max_seqlen_k``, one workspace ``budget_bytes`` and the same
    host boundaries (given or not): dK and dV of a key block are summed in tensor memory in one order, dS^T goes to
    the workspace and dQ is a GEMM over it with each output tile summed in one order, and the row-map destinations
    are folded in ascending window order.  A call whose dS^T exceeds the budget runs in chunks (KV heads first, then
    key-block ranges); the budget and the host boundaries move the dQ rounding, not its accuracy.  Raises no PyTorch
    determinism alert.
    """
    return _pkg().cake_gqa_det_train_backward(
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
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
    )


@flashinfer_api
def cake_gqa_det_train_workspace(
    device: Union[torch.device, str, int, None] = None,
    *,
    budget_bytes: Optional[int] = None,
) -> Any:
    r"""A workspace (``CakeGQADetWorkspace``) for :func:`cake_gqa_det` and its two halves.

    It holds the forward's tile counter and the backward scratch, grown to the largest shape seen and never shrunk
    (``.nbytes``; ``.release()`` drops the scratch).  The dS^T workspace and the row-map window gradients are
    allocated for each backward call and freed when it returns.  One workspace serves one stream at a time.

    Parameters
    ----------
    device : torch.device, str or int, optional
        The CUDA device; ``None`` is the current device.
    budget_bytes : int, optional
        The bound of one backward call's dS^T workspace; ``None`` is 16 GiB.  A call whose dS^T exceeds it runs in
        chunks (at least one KV head and two key blocks per chunk, even past the budget).  The chunks set the dQ
        rounding: one budget gives the same bits on every run.

    Returns
    -------
    A ``CakeGQADetWorkspace``.
    """
    if budget_bytes is None:
        return _pkg().CakeGQADetWorkspace(device)
    return _pkg().CakeGQADetWorkspace(device, budget_bytes=budget_bytes)


@flashinfer_api
def cake_gqa_det_train_workspace_bytes(
    total_q: int,
    num_heads_q: int,
    num_segments: int,
    *,
    total_k: Optional[int] = None,
    num_heads_kv: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    row_map: bool = False,
    budget_bytes: Optional[int] = None,
    cu_seqlens_q_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
    cu_seqlens_k_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> int:
    r"""Bytes of backward workspace one call needs.

    Parameters
    ----------
    total_q : int
        ``T_q``, the packed query rows.
    num_heads_q : int
        ``H_q``, the query heads.
    num_segments : int
        ``B``, the segments (``len(cu_seqlens_q) - 1``).
    total_k : int, optional
        ``T_k``, given together with ``num_heads_kv``; with both the dS^T workspace is included.  One without
        the other raises ``ValueError``.
    num_heads_kv : int, optional
        ``H_kv``, given together with ``total_k``.
    max_seqlen_k : int, optional
        As in :func:`cake_gqa_det`.
    row_map : bool
        Include the two FP32 ``[T_k, H_kv, 256]`` window gradients of a row-map call (needs ``total_k`` and
        ``num_heads_kv``, else ``ValueError``).
    budget_bytes : int, optional
        As in :func:`cake_gqa_det_train_workspace`.
    cu_seqlens_q_host, cu_seqlens_k_host : sequence of int or torch.Tensor, optional
        As in :func:`cake_gqa_det`: the exact dS^T chunk of the backward call that gets them.

    Returns
    -------
    The bytes of FP32 ``[H_q / 2, rows_pad, 256]`` plus two ``[H_q / 2, rows_pad]``, with
    ``rows_pad = round_up(2 T_q + 128 B, 128)``, plus the dS^T chunk and the window gradients when asked.
    """
    kwargs: dict[str, Any] = {
        "total_k": total_k,
        "num_heads_kv": num_heads_kv,
        "max_seqlen_k": max_seqlen_k,
        "row_map": row_map,
        "cu_seqlens_q_host": cu_seqlens_q_host,
        "cu_seqlens_k_host": cu_seqlens_k_host,
    }
    if budget_bytes is not None:
        kwargs["budget_bytes"] = budget_bytes
    return _pkg().cake_gqa_det_train_workspace_bytes(
        total_q, num_heads_q, num_segments, **kwargs
    )
