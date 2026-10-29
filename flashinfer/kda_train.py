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
    arena = kda_train_arena(lean=True, tail_passes=4)                    # the lower-memory arena option
    ws = kda_train_workspace(S, num_docs, H, device, arena=arena)        # (off by default): a smaller backward
    out = chunk_kda(q, k, v, g, beta, cu_seqlens, workspace=ws)          # arena, the same gradients bitwise
    arena = kda_train_arena(lean=True, tail_passes=4, batch_tokens=B)    # and over groups of whole documents of
    ws = kda_train_workspace(S, num_docs, H, device, arena=arena, doc_lengths=lengths)   # at most B tokens: the
    out = chunk_kda(q, k, v, g, beta, cu_seqlens, workspace=ws, doc_lengths=lengths)     # arena of the largest group

Context-parallel training (KCP): the packed batch is split over the ranks
of a process group (``kcp_partition``), every rank holds its fragments of the
documents, and the per-document state recurrences cross the rank boundaries
through the plan's collectives (``build_kcp_context``)::

    from flashinfer.kda_train import kcp_partition, build_kcp_context, chunk_kda_kcp

    part = kcp_partition(cu_seqlens_global_cpu, world_size)            # once per batch layout
    ctx = build_kcp_context(part, group, rank, device)                 # tables and collectives of this rank
    out_local = chunk_kda_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd
    out_local, saved = chunk_kda_kcp_fwd(q_local, k_local, v_local, g_local, beta_local, ctx)
    grads = chunk_kda_kcp_bwd(q_local, k_local, v_local, g_local, beta_local, ctx, dout_local, saved=saved)
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
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
        ``"light"`` (saved set version 4: ``Aqk`` / ``Akk`` and the tables, 256 B per token-head, the forward's
        gated queries ``qg``, 256 B per internal token-head, plus the segment-boundary states on the segmented
        route when ``S >= 384 * NSEG``, so the backward skips the forward half of its recurrence recompute, and
        the WY representation on the rows whose backward reads it from the set) or ``"tape"`` (version 2: also
        the forward's state tape and ``v_new``, 768 B more per token-head, so the backward skips the whole
        recompute); ``None`` takes the workspace's default (``"light"`` without one).
    arena : KdaArena, optional
        The lower-memory arena option (:func:`kda_train_arena`; off by default): the forward's layout is the
        same under every option, and a batched option (``batch_tokens``) runs the forward once per document
        group; ``None`` takes the workspace's option, a differing option on a given workspace is refused.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers; ``len == num_docs``, ``sum == S``),
        required under a batched option (the host reads no tensor) and checked against ``cu_seqlens`` on the device.

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
        arena=arena,
        doc_lengths=doc_lengths,
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Packed KDA chunked training backward.

    Parameters
    ----------
    q, k, v, g, beta, cu_seqlens
        The forward's operands.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : KdaChunkSaved, optional
        The forward's saved set (version 4 or 2); ``None`` recomputes it from the inputs (checkpoint replay).
    scale : float, optional
        The forward's scale; ``128 ** -0.5`` when ``None``.
    workspace : KdaWorkspace, optional
        A reused workspace with backward buffers; without one this call allocates its own.
    arena : KdaArena, optional
        The lower-memory arena option (:func:`kda_train_arena`; off by default) of that per-call workspace:
        the lean backward layouts and the tail split (``tail_passes``), the same gradients bitwise; ``None``
        takes a given workspace's option, a differing one is refused.  A lean arena serves no checkpoint replay
        (``saved=None``); a batched option (``batch_tokens``) runs the backward once per document group.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers), required under a batched option and
        checked against ``cu_seqlens`` on the device.

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
        arena=arena,
        doc_lengths=doc_lengths,
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
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    r"""Packed KDA chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; see :func:`chunk_kda_fwd` for the
    operands, ``save_mode``, ``arena`` and ``doc_lengths`` (the node records the option and the document lengths,
    so a backward that allocates its own workspace takes them too) and :func:`chunk_kda_bwd` for the gradients.
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
        arena=arena,
        doc_lengths=doc_lengths,
    )


def kda_train_arena(
    *,
    lean: bool = False,
    tail_passes: int = 1,
    batch_tokens: Optional[int] = None,
    max_docs_per_batch: Optional[int] = None,
) -> Any:
    """The lower-memory arena option (a ``KdaArena``; the default, ``lean=False, tail_passes=1``, no batching, is
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
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.KdaArena(
        lean=lean,
        tail_passes=tail_passes,
        batch_tokens=batch_tokens,
        max_docs_per_batch=max_docs_per_batch,
    )


def kda_train_workspace(
    S: int,
    num_docs: int,
    H: int,
    device: torch.device | str,
    *,
    backward: bool = True,
    save_mode: str = "light",
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
) -> Any:
    """A caller-owned ``KdaWorkspace`` (one arena of internal intermediates) for ``S`` tokens in ``num_docs``
    documents with ``H`` heads on ``device``, reused by every call that receives it: the forward layout only with
    ``backward=False``; ``save_mode`` is the default of the calls on it and sizes the arena for that mode;
    ``arena`` (:func:`kda_train_arena`) is the lower-memory arena option, off when ``None``; under a batched
    option ``doc_lengths`` (the documents' token counts) sizes the arena for the exact document groups instead of
    the bound of every group."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.KdaWorkspace.for_problem(
        S,
        num_docs,
        H,
        device,
        backward=backward,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
    )


def kda_train_workspace_bytes(
    S: int,
    num_docs: int,
    H: int,
    *,
    backward: bool = True,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
    arena: Optional[Any] = None,
    doc_lengths: Optional[Sequence[int]] = None,
) -> int:
    """Bytes :func:`kda_train_workspace` allocates for the problem (``sm_count``: the device's SM count, the
    current CUDA device's when ``None``; it sets the segment count of the segmented recurrences; ``arena``: the
    lower-memory arena option, off when ``None``; ``doc_lengths``: the exact document groups of a batched option,
    else their bound)."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.workspace_bytes(
        S,
        num_docs,
        H,
        backward=backward,
        save_mode=save_mode,
        sm_count=sm_count,
        arena=arena,
        doc_lengths=doc_lengths,
    )


def kda_train_saved_bytes(
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
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.saved_bytes(
        S, H, num_docs, save_mode=save_mode, sm_count=sm_count
    )


# --------------------------------------------------------------------------- context-parallel training (KCP)
#
# The KCP entry points forward to ``flashinfer.experimental.cake_kda_train.cake_kcp`` (the per-rank launch
# sequences over the kernels of the ``cake_kcp_jit`` registry) and ``cake_kcp_plan`` (the partition, the context's
# tables and collectives, the byte queries).  Every keyword is forwarded; nothing is validated here.


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_partition(
    cu_seqlens_global_cpu: Any,
    world_size: int,
    layout: str = "contiguous",
    fragments: Optional[Any] = None,
    *,
    cut_points: Optional[Any] = None,
) -> Any:
    r"""The context-parallel partition of one packed batch over ``world_size`` ranks.

    Parameters
    ----------
    cu_seqlens_global_cpu
        int32 ``[num_docs + 1]`` document offsets of the whole batch, on the host (a tensor or a sequence).
    world_size : int
        The number of ranks of the context-parallel group.
    layout : str
        ``"contiguous"`` (equal token shares in order), ``"zigzag"`` or ``"zigzag_per_document"`` (the load-balanced
        layouts); ``fragments`` gives an explicit ragged layout instead, and ``cut_points`` (per document the
        ``2 * world_size + 1`` cut offsets) applies to ``layout="zigzag_per_document"`` only.

    Returns
    -------
    ``KcpPartition``: per rank its fragments (document, token range) of the batch; the input of
    :func:`build_kcp_context` on every rank.
    """
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.kcp_partition(
        cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
    )


@flashinfer_experimental_api(feature=_FEATURE)
def build_kcp_context(
    part: Any,
    group: Any,
    rank: int,
    device: Any,
    conv_kernel_size: int = 4,
    *,
    sm_count: Optional[int] = None,
) -> Any:
    r"""This rank's context of a partition: its device tables, the collectives of the state and halo exchanges and
    the segment plan of its fragments.

    Parameters
    ----------
    part : KcpPartition
        From :func:`kcp_partition`, the same object on every rank.
    group
        The ``torch.distributed`` process group of the context-parallel ranks.
    rank : int
        This process's rank in ``group``.
    device
        The CUDA device of this rank's operands.
    conv_kernel_size : int
        The width of the causal convolution whose history rows the context exchanges (4).
    sm_count : int, optional
        The device's SM count when it should not be queried.

    Returns
    -------
    ``KcpContext``: the argument every KCP entry point takes; reusable across calls of the same batch layout.
    """
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.build_kcp_context(
        part, group, rank, device, conv_kernel_size, sm_count=sm_count
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_kcp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: Optional[float] = None,
    save_for_backward: bool = True,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: Optional[Any] = None,
    **options: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Context-parallel packed KDA chunked training forward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta : torch.Tensor
        This rank's shard of the operands of :func:`chunk_kda_fwd` (the tokens of its fragments, in the
        partition's order): BF16 ``[1, S_local, H, 128]`` ``q, k, v``, FP32 ``g``, BF16 ``[1, S_local, H]`` ``beta``.
    ctx : KcpContext
        From :func:`build_kcp_context`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Write the saved set (the local saved set of :func:`chunk_kda_fwd` plus the boundary states the backward
        starts from).
    save_mode : str
        ``"light"`` or ``"tape"`` as for :func:`chunk_kda_fwd`.
    boundary_mode : str
        ``"saved"`` keeps the received boundary states in the saved set; ``"replay"`` saves nothing and lets the
        backward replay the forward's exchanges.
    workspace : KcpWorkspace, optional
        A reused workspace of this rank; without one the call allocates its own.
    **options
        Further KCP options (``save_transfer``, ``exchange``, ``head_groups``, ``offset_class``, ``graph``),
        forwarded.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (this rank's tokens); ``saved`` a ``KcpSaved`` or ``None``.
    """
    from .experimental.cake_kda_train import cake_kcp

    return cake_kcp.chunk_kda_kcp_fwd(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_for_backward=save_for_backward,
        save_mode=save_mode,
        boundary_mode=boundary_mode,
        workspace=workspace,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_kcp_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: Optional[Any] = None,
    scale: Optional[float] = None,
    save_mode: str = "light",
    workspace: Optional[Any] = None,
    **options: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Context-parallel packed KDA chunked training backward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta, ctx
        The forward's inputs (same contracts as :func:`chunk_kda_kcp_fwd`).
    dout : torch.Tensor
        BF16 gradient of this rank's ``out``.
    saved : KcpSaved, optional
        The forward's saved set; without one the forward is replayed (checkpointing).
    scale, save_mode, workspace, **options
        As for :func:`chunk_kda_kcp_fwd`; the backward reads its own options, among them ``spec3_probe``: the
        debug probe of the first saved-mode backward, ``"off"`` by default, ``"record"``, ``"warn"`` or
        ``"assert"`` (the same value on every rank).

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)`` of this rank's tokens: BF16 ``dq, dk, dv, dbeta``, FP32 ``dg``.
    """
    from .experimental.cake_kda_train import cake_kcp

    return cake_kcp.chunk_kda_kcp_bwd(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        dout,
        saved=saved,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    scale: Optional[float] = None,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: Optional[Any] = None,
    **options: Any,
) -> torch.Tensor:
    r"""Autograd-enabled context-parallel packed KDA chunked training step of this rank's fragments (the forward of
    :func:`chunk_kda_kcp_fwd`; the backward of :func:`chunk_kda_kcp_bwd` from its saved set).

    Returns
    -------
    torch.Tensor
        BF16 ``out`` like ``v`` (this rank's tokens).
    """
    from .experimental.cake_kda_train import cake_kcp

    return cake_kcp.chunk_kda_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        boundary_mode=boundary_mode,
        workspace=workspace,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_workspace_bytes(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: bool = False,
    C: int = 8192,
) -> int:
    """Bytes of this rank's KCP workspace for ``H`` heads: the local arena of :func:`kda_train_workspace_bytes` plus
    the boundary pools of the context's collectives, as ``KcpWorkspace`` allocates them (the backward's send pools
    share the forward's storage; the ring exchange's hop pools, allocated by the first ``exchange="ring"`` call, are
    not counted).  The saved set and the convolution's halo rows are per-call tensors, so ``save_transfer`` and
    ``C`` do not change it."""
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.kcp_workspace_bytes(
        ctx,
        H=H,
        save_mode=save_mode,
        backward=backward,
        save_transfer=save_transfer,
        C=C,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_saved_bytes(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    save_transfer: bool = False,
    with_conv: bool = True,
    C: int = 8192,
) -> int:
    """Bytes of this rank's KCP saved set for ``H`` heads: the local saved set (:func:`kda_train_saved_bytes`), the
    boundary states, and (``with_conv``) the convolution's history rows of ``C`` channels."""
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.kcp_saved_bytes(
        ctx,
        H=H,
        save_mode=save_mode,
        save_transfer=save_transfer,
        with_conv=with_conv,
        C=C,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_comm_bytes(
    ctx: Any,
    *,
    H: int = 64,
    C: int = 8192,
    save_transfer: bool = False,
    boundary_mode: str = "saved",
) -> dict:
    """Bytes this rank sends and receives in one training step (forward and backward) for ``H`` heads and ``C``
    convolution channels: the state summaries, the boundary states and the halo rows of the context's collectives,
    as a table per collective and per boundary (``cake_kcp_plan.kcp_comm_bytes``)."""
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.kcp_comm_bytes(
        ctx, H=H, C=C, save_transfer=save_transfer, boundary_mode=boundary_mode
    )


def is_kda_train_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is a target of the Cake kernels
    (compute capability 10.0, 10.3 or 10.7)."""
    from .experimental.cake_kda_train import cake_backend

    return cake_backend.is_supported(device)


__all__ = [
    "DEFAULT_SCALE",
    "build_kcp_context",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "chunk_kda_kcp",
    "chunk_kda_kcp_bwd",
    "chunk_kda_kcp_fwd",
    "is_kda_train_supported",
    "kcp_comm_bytes",
    "kcp_partition",
    "kcp_saved_bytes",
    "kcp_workspace_bytes",
    "kda_train_arena",
    "kda_train_saved_bytes",
    "kda_train_workspace",
    "kda_train_workspace_bytes",
]
