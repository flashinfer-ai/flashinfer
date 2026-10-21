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

Packed KDA (Kimi Delta Attention) chunked training forward and backward
(SM100 / SM103 / SM107), the Cake backend.

Thin experimental entry points.  The contract, the host plan, the workspace,
the launch sequences, the autograd Function and the generated kernels live in
``flashinfer.experimental.cake_kda_train``; this module validates nothing
itself and forwards every keyword.  Every switch is a keyword argument; no
environment variable is read.

Operands (one packed batch of documents): ``q, k, v`` BF16 ``[1, S, H, 128]``
(``q`` / ``k`` L2-normalized by the caller), ``g`` FP32 ``[1, S, H, 128]`` (the
natural-log decay gate, preactivated), ``beta`` BF16 ``[1, S, H]`` (through
the sigmoid), ``cu_seqlens`` int32 ``[num_docs + 1]`` on the device.  The
batch axis of 1 is optional; outputs take the rank of the inputs.  Gradients:
``dq, dk, dv, dbeta`` BF16, ``dg`` FP32.  Per-document zero initial state, no
final state, ``scale = 128 ** -0.5`` by default, ``H`` in ``1..64``.

Usage::

    from flashinfer.kda_train import chunk_kda, chunk_kda_fwd, chunk_kda_bwd

    out = chunk_kda(q, k, v, g, beta, cu_seqlens)                  # autograd
    out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, save_for_backward=True)
    dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout, saved=saved)
    dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout)  # checkpoint replay

    ws = kda_train_workspace(S, num_docs, H, device)               # caller-owned, reused
    out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws)
    ws = kda_train_workspace(S, num_docs, H, device, save_mode="tape")   # saved set v2: the backward
    out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws)  # skips its recurrence recompute
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake packed KDA chunked training (SM100/SM103/SM107)"

# The default scale of the GLM contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    save_for_backward: bool = True,
    workspace: Optional[Any] = None,
    save_mode: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Packed KDA chunked training forward.

    Parameters
    ----------
    q, k, v : torch.Tensor
        BF16 ``[1, S, H, 128]`` (or ``[S, H, 128]``), contiguous.
    g : torch.Tensor
        FP32 ``[1, S, H, 128]`` natural-log decay gate.
    beta : torch.Tensor
        BF16 ``[1, S, H]`` delta-rule gate after the sigmoid.
    cu_seqlens : torch.Tensor
        int32 ``[num_docs + 1]`` document offsets on the device (first 0, last ``S``).
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Write the saved set (``Aqk`` / ``Akk`` on the token rows plus the device chunk tables).
    workspace : KdaWorkspace, optional
        A reused workspace from :func:`kda_train_workspace`; without one this call allocates its own.
    save_mode : str, optional
        ``"light"`` (saved set version 1: ``Aqk`` / ``Akk`` and the tables, 256 B per token-head) or ``"tape"``
        (version 2: also the forward's state tape and ``v_new``, 768 B more per token-head, so the backward skips
        its recurrence recompute); ``None`` takes the workspace's default (``"light"`` without one).

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v``; ``saved`` a ``KdaChunkSaved`` (versioned) or ``None``.
    """
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        save_mode=save_mode,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dout: torch.Tensor,
    *,
    saved: Optional[Any] = None,
    scale: Optional[float] = None,
    workspace: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Packed KDA chunked training backward.

    Parameters
    ----------
    q, k, v, g, beta, cu_seqlens
        The forward's operands.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : KdaChunkSaved, optional
        The forward's saved set (version 1 or 2); ``None`` recomputes it from the inputs (checkpoint replay).
    scale : float, optional
        The forward's scale; ``128 ** -0.5`` when ``None``.
    workspace : KdaWorkspace, optional
        A reused workspace with backward buffers; without one this call allocates its own.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v, beta`` and the FP32 gradient of ``g``, ranked
    like the inputs.  Bitwise reproducible for the same inputs.
    """
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        dout,
        saved=saved,
        scale=DEFAULT_SCALE if scale is None else scale,
        workspace=workspace,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    workspace: Optional[Any] = None,
    save_mode: Optional[str] = None,
) -> torch.Tensor:
    r"""Packed KDA chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; see :func:`chunk_kda_fwd` for the
    operands and ``save_mode`` and :func:`chunk_kda_bwd` for the gradients.
    """
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=DEFAULT_SCALE if scale is None else scale,
        workspace=workspace,
        save_mode=save_mode,
    )


def kda_train_workspace(
    S: int,
    num_docs: int,
    H: int,
    device: torch.device | str,
    *,
    backward: bool = True,
    save_mode: str = "light",
) -> Any:
    """A caller-owned ``KdaWorkspace`` (one arena of internal intermediates) for ``S`` tokens in ``num_docs``
    documents with ``H`` heads on ``device``, reused by every call that receives it: the forward layout only with
    ``backward=False``; ``save_mode`` is the default of the calls on it and sizes the arena for that mode."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.KdaWorkspace.for_problem(
        S, num_docs, H, device, backward=backward, save_mode=save_mode
    )


def kda_train_workspace_bytes(
    S: int,
    num_docs: int,
    H: int,
    *,
    backward: bool = True,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
) -> int:
    """Bytes :func:`kda_train_workspace` allocates for the problem (``sm_count``: the device's SM count, the
    current CUDA device's when ``None``; it sets the segment count of the segmented recurrences)."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.workspace_bytes(
        S, num_docs, H, backward=backward, save_mode=save_mode, sm_count=sm_count
    )


def kda_train_saved_bytes(
    S: int,
    H: int,
    num_docs: Optional[int] = None,
    *,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
) -> int:
    """Bytes of the saved set: ``Aqk`` and ``Akk`` (``256 * S * H``), plus the device tables when ``num_docs`` is
    given, plus the state tape, ``v_new`` and the segment transfers with ``save_mode="tape"``."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.saved_bytes(
        S, H, num_docs, save_mode=save_mode, sm_count=sm_count
    )


def is_kda_train_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is a target of the Cake kernels
    (compute capability 10.0, 10.3 or 10.7)."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.is_supported(device)


__all__ = [
    "DEFAULT_SCALE",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "is_kda_train_supported",
    "kda_train_saved_bytes",
    "kda_train_workspace",
    "kda_train_workspace_bytes",
]
