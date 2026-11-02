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
(SM100 / SM103 / SM107), the Cake backend.  Deterministic: the same inputs, hardware and
plan give bitwise the same outputs and gradients.

Thin experimental entry points.  The contract, the host plan, the workspace,
the launch sequences, the autograd Function and the generated kernels live in
``flashinfer.experimental.cake_kda_det_train``; this module validates nothing
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

    from flashinfer.cake_kda_det_train import chunk_kda_det, chunk_kda_det_fwd, chunk_kda_det_bwd

    out = chunk_kda_det(q, k, v, g, beta, cu_seqlens)                  # autograd
    out, saved = chunk_kda_det_fwd(q, k, v, g, beta, cu_seqlens, save_for_backward=True)
    dq, dk, dv, dbeta, dg = chunk_kda_det_bwd(q, k, v, g, beta, cu_seqlens, dout, saved=saved)
    dq, dk, dv, dbeta, dg = chunk_kda_det_bwd(q, k, v, g, beta, cu_seqlens, dout)  # checkpoint replay

    ws = kda_det_train_workspace(S, num_docs, H, device)               # caller-owned, reused
    out, saved = chunk_kda_det_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws)
    ws = kda_det_train_workspace(S, num_docs, H, device, save_mode="tape")   # saved set v2: the backward
    out, saved = chunk_kda_det_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws)  # skips its recurrence recompute
    arena = kda_det_train_arena(lean=True, tail_passes=4)                    # the lower-memory arena option
    ws = kda_det_train_workspace(S, num_docs, H, device, arena=arena)        # (off by default): a smaller backward
    out = chunk_kda_det(q, k, v, g, beta, cu_seqlens, workspace=ws)          # arena, the same gradients bitwise
    arena = kda_det_train_arena(lean=True, tail_passes=4, batch_tokens=B)    # and over groups of whole documents of
    ws = kda_det_train_workspace(S, num_docs, H, device, arena=arena, doc_lengths=lengths)   # at most B tokens: the
    out = chunk_kda_det(q, k, v, g, beta, cu_seqlens, workspace=ws, doc_lengths=lengths)     # arena of the largest group
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake packed KDA chunked training (SM100/SM103/SM107)"

# The default scale of the GLM contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_det_fwd(
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Packed KDA chunked training forward.

    Parameters
    ----------
    q, k, v : torch.Tensor
        BF16 ``[1, S, H, 128]`` (or ``[S, H, 128]``), contiguous.
    g : torch.Tensor
        FP32 ``[1, S, H, 128]`` natural-log decay gate (preactivated); with ``use_gate_in_kernel=True`` the raw
        gate, BF16 ``[1, S, H, 128]``, which may be a strided view (unit innermost stride, token and head strides
        multiples of 8 elements, rows that do not overlap, a 16-byte aligned address; never copied).
    beta : torch.Tensor
        BF16 ``[1, S, H]`` delta-rule gate after the sigmoid.
    cu_seqlens : torch.Tensor
        int32 ``[num_docs + 1]`` document offsets on the device (first 0, last ``S``).
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Write the saved set (``Aqk`` / ``Akk`` on the token rows plus the device chunk tables).
    workspace : KdaDetWorkspace, optional
        A reused workspace from :func:`kda_det_train_workspace`; without one this call allocates its own.
    save_mode : str, optional
        ``"light"`` (saved set version 4: ``Aqk`` / ``Akk`` and the tables, 256 B per token-head, the forward's
        gated queries ``qg``, 256 B per internal token-head, plus the segment-boundary states on the segmented
        route when ``S >= 384 * NSEG``, so the backward skips the forward half of its recurrence recompute, and
        the WY representation on the rows whose backward reads it from the set) or ``"tape"`` (version 2: also
        the forward's state tape and ``v_new``, 768 B more per token-head, so the backward skips the whole
        recompute); ``None`` takes the workspace's default (``"light"`` without one).
    arena : KdaDetArena, optional
        The lower-memory arena option (:func:`kda_det_train_arena`; off by default): the forward's layout is the
        same under every option, and a batched option (``batch_tokens``) runs the forward once per document
        group; ``None`` takes the workspace's option, a differing option on a given workspace is refused.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers; ``len == num_docs``, ``sum == S``),
        required under a batched option (the host reads no tensor) and checked against ``cu_seqlens`` on the device.
    use_gate_in_kernel : bool
        ``False`` (the default): ``g`` is the activated FP32 gate and ``A_log``, ``dt_bias``, ``lower_bound`` must
        be ``None``.  ``True``: ``g`` is the raw BF16 gate and the kernels activate it as
        ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``; the workspace must be allocated with
        ``use_gate_in_kernel=True`` and the saved set records the mode.
    A_log : torch.Tensor, optional
        FP32 ``[H]``, contiguous (raw gate mode).
    dt_bias : torch.Tensor, optional
        FP32 ``[H * 128]`` or ``[H, 128]``, contiguous (raw gate mode).
    lower_bound : float, optional
        The gate's lower bound, required in raw gate mode: finite, ``-5.0 <= lower_bound < 0.0``.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v``; ``saved`` a ``KdaDetChunkSaved`` (versioned) or ``None``.
    """
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.chunk_kda_det_fwd(
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
        arena=arena,
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_det_bwd(
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Packed KDA chunked training backward.

    Parameters
    ----------
    q, k, v, g, beta, cu_seqlens
        The forward's operands.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : KdaDetChunkSaved, optional
        The forward's saved set (version 4 or 2); ``None`` recomputes it from the inputs (checkpoint replay).
    scale : float, optional
        The forward's scale; ``128 ** -0.5`` when ``None``.
    workspace : KdaDetWorkspace, optional
        A reused workspace with backward buffers; without one this call allocates its own.
    arena : KdaDetArena, optional
        The lower-memory arena option (:func:`kda_det_train_arena`; off by default) of that per-call workspace:
        the lean backward layouts and the tail split (``tail_passes``), the same gradients bitwise; ``None``
        takes a given workspace's option, a differing one is refused.  A lean arena serves no checkpoint replay
        (``saved=None``); a batched option (``batch_tokens``) runs the backward once per document group.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers), required under a batched option and
        checked against ``cu_seqlens`` on the device.
    use_gate_in_kernel, A_log, dt_bias, lower_bound
        The forward's gate mode and raw gate parameters (see :func:`chunk_kda_det_fwd`); a saved set of the other mode,
        or with another ``lower_bound``, is refused.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v, beta`` and the FP32 gradient of ``g``, ranked
    like the inputs.  With ``use_gate_in_kernel=True``: ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, ``dg``
    the BF16 gradient of the raw gate (contiguous, ranked like ``g``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias``
    FP32 shaped like ``dt_bias``.  Bitwise reproducible for the same inputs.
    """
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.chunk_kda_det_bwd(
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
        arena=arena,
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_det(
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    r"""Packed KDA chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; see :func:`chunk_kda_det_fwd` for the
    operands, ``save_mode``, ``arena``, ``doc_lengths`` (the node records the option and the document lengths,
    so a backward that allocates its own workspace takes them too) and the raw gate mode
    (``use_gate_in_kernel=True``, ``A_log``, ``dt_bias``, ``lower_bound``: gradients to the raw gate ``g``, BF16,
    and to ``A_log`` / ``dt_bias``, FP32) and :func:`chunk_kda_det_bwd` for the gradients.
    """
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.chunk_kda_det(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=DEFAULT_SCALE if scale is None else scale,
        workspace=workspace,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


def kda_det_train_arena(
    *,
    lean: bool = False,
    tail_passes: int = 1,
    batch_tokens: Optional[int] = None,
    max_docs_per_batch: Optional[int] = None,
) -> Any:
    """The lower-memory arena option (a ``KdaDetArena``; the default, ``lean=False, tail_passes=1``, no batching, is
    the full layout).  ``lean=True`` selects the lean backward layouts (4108 B per internal token-head with one
    tail pass instead of the default's 4488, tape 3720; no checkpoint replay on such a workspace); ``tail_passes
    >= 2`` (implies ``lean``) runs the backward's tail, the fused gradient / GLM intra / gate epilogue, once per
    range of ``ceil(NC_pad / tail_passes)`` chunks over per-range buffers: 2820 B per internal token-head plus
    three state blocks on the floor, the same gradients bitwise, some speed cost (see the package README).
    ``batch_tokens`` runs the forward and the backward over groups of whole, contiguous documents of at most that
    many tokens (and at most ``max_docs_per_batch`` documents, which needs ``batch_tokens``) in an arena sized for
    the largest group (at ``tail_passes``; each group runs its tail in the fewest passes up to that which fit the
    arena); the calls then take ``doc_lengths``.  Bitwise the same results, except that an exactly-zero
    ``u`` / ``w`` entry of a chunk whose pair partner changed may change sign; refused on the rows with at most four
    (document, head) chains on the segmented route."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.KdaDetArena(
        lean=lean,
        tail_passes=tail_passes,
        batch_tokens=batch_tokens,
        max_docs_per_batch=max_docs_per_batch,
    )


def kda_det_train_workspace(
    S: int,
    num_docs: int,
    H: int,
    device: torch.device | str,
    *,
    backward: bool = True,
    save_mode: str = "light",
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
    use_gate_in_kernel: bool = False,
) -> Any:
    """A caller-owned ``KdaDetWorkspace`` (one arena of internal intermediates) for ``S`` tokens in ``num_docs``
    documents with ``H`` heads on ``device``, reused by every call that receives it: the forward layout only with
    ``backward=False``; ``save_mode`` is the default of the calls on it and sizes the arena for that mode;
    ``arena`` (:func:`kda_det_train_arena`) is the lower-memory arena option, off when ``None``; under a batched
    option ``doc_lengths`` (the documents' token counts) sizes the arena for the exact document groups instead of
    the bound of every group; ``use_gate_in_kernel`` is the gate mode of the calls it serves (a call in the other
    mode is refused)."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.KdaDetWorkspace.for_problem(
        S,
        num_docs,
        H,
        device,
        backward=backward,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
        gate=_gate(use_gate_in_kernel),
    )


def kda_det_train_workspace_bytes(
    S: int,
    num_docs: int,
    H: int,
    *,
    backward: bool = True,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
    use_gate_in_kernel: bool = False,
) -> int:
    """Bytes :func:`kda_det_train_workspace` allocates for the problem (``sm_count``: the device's SM count, the
    current CUDA device's when ``None``; it sets the segment count of the segmented recurrences; ``arena``: the
    lower-memory arena option, off when ``None``; ``doc_lengths``: the exact document groups of a batched option,
    else their bound; ``use_gate_in_kernel``: the gate mode, whose backward layouts carry no gate epilogue
    buffers in raw mode)."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.workspace_bytes(
        S,
        num_docs,
        H,
        backward=backward,
        save_mode=save_mode,
        sm_count=sm_count,
        arena=arena,
        doc_lengths=doc_lengths,
        gate=_gate(use_gate_in_kernel),
    )


def _gate(use_gate_in_kernel: bool) -> str:
    """The backend's gate mode of ``use_gate_in_kernel`` (``ValueError`` for a non-bool)."""
    if not isinstance(use_gate_in_kernel, bool):
        raise ValueError(
            f"use_gate_in_kernel must be a bool, got {use_gate_in_kernel!r}"
        )
    return "raw" if use_gate_in_kernel else "g"


def kda_det_train_saved_bytes(
    S: int,
    H: int,
    num_docs: Optional[int] = None,
    *,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
) -> int:
    """Bytes of the saved set: ``Aqk`` and ``Akk`` (``256 * S * H``), plus, when ``num_docs`` is given, the device
    tables and the version's per-call tensors: the forward's gated queries ``qg`` (and, on the rows whose backward
    reads it from the set, the WY representation ``w`` / ``u``) and the segment-boundary states under the size rule
    with ``save_mode="light"``; the state tape, ``v_new`` and the segment transfers with ``save_mode="tape"``."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.saved_bytes(
        S, H, num_docs, save_mode=save_mode, sm_count=sm_count
    )


def is_kda_det_train_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is a target of the Cake kernels
    (compute capability 10.0, 10.3 or 10.7)."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend.is_supported(device)


__all__ = [
    "DEFAULT_SCALE",
    "chunk_kda_det",
    "chunk_kda_det_bwd",
    "chunk_kda_det_fwd",
    "is_kda_det_train_supported",
    "kda_det_train_arena",
    "kda_det_train_saved_bytes",
    "kda_det_train_workspace",
    "kda_det_train_workspace_bytes",
]
