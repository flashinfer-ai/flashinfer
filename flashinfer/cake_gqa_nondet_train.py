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

Cake non-deterministic packed-varlen causal GQA attention training with head
dimension 256 (SM100 / SM103 / SM107).

Entries: :func:`cake_gqa_nondet` (with autograd),
:func:`cake_gqa_nondet_train_forward` / :func:`cake_gqa_nondet_train_backward`
(the two halves without autograd), :func:`cake_gqa_nondet_train_workspace` and
:func:`cake_gqa_nondet_train_workspace_bytes`.  The autograd class
``CakeGQANondetFunction``, the workspace type ``CakeGQANondetWorkspace`` and
the ``dkv_acc`` callable type ``CakeGQANondetDkvAccSource`` resolve to the
backend's objects on first access.

The mode is part of the name: no entry takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  Every sum of the forward and the backward
has one fixed order: ``out``, ``lse``, dQ and every dK / dV form repeat bit
for bit run to run for one shape, one ``max_seqlen_k``, one workspace budget
and the same host boundaries (given or not), and nothing raises PyTorch's
determinism alert.

A stable API (``@flashinfer_api``; no ``ExperimentalWarning``) of thin
wrappers: the checks, the launch plan, the workspace, the autograd Function
and the generated kernels live in the backend package
``flashinfer.experimental.cake.cake_gqa_nondet_train``, imported on first use;
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

    from flashinfer.cake_gqa_nondet_train import cake_gqa_nondet, cake_gqa_nondet_train_workspace

    ws = cake_gqa_nondet_train_workspace(device)         # optional: reused across calls and shapes
    out = cake_gqa_nondet(q, k, v, cu_q, cu_k, softmax_scale=0.0625, workspace=ws)
    out.backward(dout)                                   # BF16 dq, dk, dv

    # FP32 dK / dV added into caller-owned destinations through a row map, produced at backward time:
    out = cake_gqa_nondet(q, k, v, cu_q, cu_k, dkv_acc=lambda: (dk_dst, dv_dst, row_map))
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
    from .experimental.cake.cake_gqa_nondet_train import (
        CakeGQANondetDkvAccSource,
        CakeGQANondetFunction,
        CakeGQANondetWorkspace,
    )

__all__ = [
    "CakeGQANondetDkvAccSource",
    "CakeGQANondetFunction",
    "CakeGQANondetWorkspace",
    "cake_gqa_nondet",
    "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_workspace",
    "cake_gqa_nondet_train_workspace_bytes",
]

# Backend objects this module re-exports, resolved on first access (``__getattr__``).
_BACKEND_OBJECTS = frozenset(
    {"CakeGQANondetDkvAccSource", "CakeGQANondetFunction", "CakeGQANondetWorkspace"}
)


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use (experimental containment: core reaches the kernels through a
    # deferred import).
    from .experimental.cake import cake_gqa_nondet_train  # noqa: PLC0415

    return cake_gqa_nondet_train


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_pkg(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BACKEND_OBJECTS)


@flashinfer_api
def cake_gqa_nondet(
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
    r"""Packed-varlen causal GQA attention, head dimension 256, with autograd (``CakeGQANondetFunction``).

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
        callable (``CakeGQANondetDkvAccSource``), called in backward, returning ``(dk_acc, dv_acc, row_map)`` adds
        the FP32 window gradients into the caller's FP32 ``[N_dst, H_kv, 256]`` destinations (window row ``j`` into
        row ``row_map[j]``, int32 ``[T_k]``, repeated rows accumulate), or returning ``(dk_win, dv_win)`` stores the
        FP32 window gradients ``[T_k, H_kv, 256]`` into them; ``k`` / ``v`` then get no gradient.  Nothing is held
        for the backward.
    max_seqlen_k : int, optional
        A host int at least every segment's key count (the caller's contract, as FlashAttention's
        ``max_seqlen_k``; a smaller value leaves key blocks out); it sizes the backward grids and the dS^T workspace
        chunks.  ``None`` bounds it by ``T_k`` (always correct; more, mostly empty, chunks).  Never read from the
        device.
    workspace : optional
        A workspace of :func:`cake_gqa_nondet_train_workspace`, reused across calls and shapes (grown, never
        shrunk; one stream at a time); autograd holds it by weak reference.  Its ``budget_bytes`` bounds the
        backward's dS^T workspace.  ``None``: the device's per-stream default workspace (16 GiB budget).
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
    The forward and the backward are bitwise reproducible run to run (one fixed summation order; the bits depend on
    the shapes, ``max_seqlen_k``, the workspace budget and the host boundaries) and raise no PyTorch determinism
    alert.
    """
    return _pkg().cake_gqa_nondet(
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
def cake_gqa_nondet_train_forward(
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
        As in :func:`cake_gqa_nondet`.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        As in :func:`cake_gqa_nondet`.
    softmax_scale : float, optional
        Default ``256 ** -0.5``.
    out : torch.Tensor, optional
        A contiguous BF16 ``[T_q, H_q, 256]`` output to write; ``None`` allocates one.
    lse : torch.Tensor, optional
        A contiguous FP32 ``[H_q, T_q]`` natural-log LSE to write; ``None`` allocates one.
    workspace : optional
        As in :func:`cake_gqa_nondet`.

    Returns
    -------
    ``(out, lse)``.  A row that sees no key gets ``out = 0`` and ``lse = -inf``.

    Notes
    -----
    Fixed-order and bitwise reproducible run to run; it never raises the deterministic-algorithms alert.
    """
    return _pkg().cake_gqa_nondet_train_forward(
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
def cake_gqa_nondet_train_backward(
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
        The forward's operands (see :func:`cake_gqa_nondet`).
    out : torch.Tensor
        The forward's BF16 ``[T_q, H_q, 256]`` output.
    lse : torch.Tensor
        The forward's FP32 ``[H_q, T_q]`` LSE.
    dout : torch.Tensor
        Contiguous BF16 ``[T_q, H_q, 256]`` gradient of ``out``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        As in :func:`cake_gqa_nondet`.
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
        out-of-range rows are not checked); a destination row becomes ``((dst + w_j1) + w_j2) + ...`` over its
        window rows in ascending ``j``.
    max_seqlen_k : int, optional
        As in :func:`cake_gqa_nondet`.
    workspace : optional
        As in :func:`cake_gqa_nondet`.
    cu_seqlens_q_host, cu_seqlens_k_host : sequence of int or torch.Tensor, optional
        As in :func:`cake_gqa_nondet`.

    Returns
    -------
    ``(dq, dk, dv)``; with ``row_map``, ``dk`` / ``dv`` are ``dk_acc`` / ``dv_acc``.

    Notes
    -----
    Bitwise reproducible run to run: dS^T goes to a workspace chunk (allocated per call, at most the workspace's
    ``budget_bytes`` above the minimum chunk), dK / dV are summed by the CTA pair that owns their key block, dQ is
    a GEMM over the workspace, and the row-map destinations are folded in window order; nothing adds with
    atomics.  The bits depend on the shapes, ``max_seqlen_k``, the budget and the host boundaries only (another
    budget or other host boundaries may move the last bits of dQ).  No PyTorch determinism alert is raised.
    """
    return _pkg().cake_gqa_nondet_train_backward(
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
def cake_gqa_nondet_train_workspace(
    device: Union[torch.device, str, int, None] = None,
    *,
    budget_bytes: Optional[int] = None,
) -> Any:
    r"""A workspace (``CakeGQANondetWorkspace``) for :func:`cake_gqa_nondet` and its two halves.

    It holds the forward's tile counter and the backward scratch, grown to the largest shape seen and never
    shrunk (``.nbytes``; ``.release()`` drops the scratch).  The backward's dS^T workspace and row-map window
    gradients are allocated per call from PyTorch's caching allocator and freed when the call returns.  One
    workspace serves one stream at a time.

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
    A ``CakeGQANondetWorkspace``.
    """
    return _pkg().CakeGQANondetWorkspace(device, budget_bytes=budget_bytes)


@flashinfer_api
def cake_gqa_nondet_train_workspace_bytes(
    total_q: int,
    num_heads_q: int,
    num_segments: int,
    *,
    total_k: int = 0,
    num_heads_kv: int = 0,
    max_seqlen_k: Optional[int] = None,
    row_map: bool = False,
    budget_bytes: Optional[int] = None,
    cu_seqlens_q_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
    cu_seqlens_k_host: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> int:
    r"""Bytes of device memory one backward call needs beside its operands and outputs.

    Parameters
    ----------
    total_q : int
        ``T_q``, the packed query rows.
    num_heads_q : int
        ``H_q``, the query heads.
    num_segments : int
        ``B``, the segments (``len(cu_seqlens_q) - 1``).
    total_k : int, optional
        ``T_k``; with ``num_heads_kv`` also given the per-call dS^T workspace is included.
    num_heads_kv : int, optional
        ``H_kv``, given together with ``total_k``.
    max_seqlen_k : int, optional
        As in :func:`cake_gqa_nondet`.
    row_map : bool
        Include the two per-call FP32 ``[T_k, H_kv, 256]`` window gradients of a row-map call.
    budget_bytes : int, optional
        As in :func:`cake_gqa_nondet_train_workspace`.
    cu_seqlens_q_host, cu_seqlens_k_host : sequence of int or torch.Tensor, optional
        As in :func:`cake_gqa_nondet`: the exact dS^T chunk of the backward call that gets them.

    Returns
    -------
    The bytes of FP32 ``[H_q / 2, rows_pad, 256]`` plus two ``[H_q / 2, rows_pad]`` (the scratch a workspace
    keeps), with ``rows_pad = round_up(2 T_q + 128 B, 128)``, plus the dS^T chunk and the window gradients when
    asked.
    """
    return _pkg().cake_gqa_nondet_train_workspace_bytes(
        total_q,
        num_heads_q,
        num_segments,
        total_k=total_k,
        num_heads_kv=num_heads_kv,
        max_seqlen_k=max_seqlen_k,
        row_map=row_map,
        budget_bytes=budget_bytes,
        cu_seqlens_q_host=cu_seqlens_q_host,
        cu_seqlens_k_host=cu_seqlens_k_host,
    )
