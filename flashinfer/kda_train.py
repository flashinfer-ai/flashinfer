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

Packed KDA (Kimi Delta Attention) chunked training (SM100 / SM103 / SM107), the Cake backend: the
context-parallel (KCP) entry points, and the single-rank entry points under their previous names.

The single-rank entry points live in ``flashinfer.cake_kda_det_train`` (``chunk_kda_det_fwd``,
``chunk_kda_det_bwd``, ``chunk_kda_det``, ``kda_det_train_arena``, ``kda_det_train_workspace``,
``kda_det_train_workspace_bytes``, ``kda_det_train_saved_bytes``, ``is_kda_det_train_supported``,
``DEFAULT_SCALE``).  This module binds their previous names (``chunk_kda_fwd``, ``chunk_kda_bwd``,
``chunk_kda``, ``kda_train_arena``, ``kda_train_workspace``, ``kda_train_workspace_bytes``,
``kda_train_saved_bytes``, ``is_kda_train_supported``) to the same objects.  The host plan, the
workspace, the launch sequences, the autograd Functions and the generated kernels live in
``flashinfer.experimental.cake_kda_det_train``.

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

    ctx_next = build_kcp_context(part_next, group, rank, device, heads=H)   # prepare the next pack while this one
                                                                          # runs: one async copy, no host wait
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api
from .cake_kda_det_train import (
    _FEATURE,
    DEFAULT_SCALE,
    chunk_kda_det,
    chunk_kda_det_bwd,
    chunk_kda_det_fwd,
    is_kda_det_train_supported,
    kda_det_train_arena,
    kda_det_train_saved_bytes,
    kda_det_train_workspace,
    kda_det_train_workspace_bytes,
)

# The single-rank entry points under their previous names: the same objects as the det names.
chunk_kda_fwd = chunk_kda_det_fwd
chunk_kda_bwd = chunk_kda_det_bwd
chunk_kda = chunk_kda_det
kda_train_arena = kda_det_train_arena
kda_train_workspace = kda_det_train_workspace
kda_train_workspace_bytes = kda_det_train_workspace_bytes
kda_train_saved_bytes = kda_det_train_saved_bytes
is_kda_train_supported = is_kda_det_train_supported


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
    heads: Optional[Any] = None,
    stream: Optional[Any] = None,
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
    heads : int or sequence of int, optional
        Head counts whose head plans are derived now (host work only) instead of at the first call.
    stream : torch.cuda.Stream, optional
        The stream of the context's one asynchronous table copy (default: the current stream of ``device``).  The
        host never waits for device work: a pack can be prepared while the previous one runs.  A thread other than
        the one that runs the pack must pass the stream that will run it, or call ``ctx.wait_ready(stream)`` before
        that stream reads the tables (the KCP entry points call ``ctx.wait_ready()`` on their current stream).

    Returns
    -------
    ``KcpContext``: the argument every KCP entry point takes; reusable across calls of the same batch layout.
    """
    from .experimental.cake_kda_train import cake_kcp_plan

    return cake_kcp_plan.build_kcp_context(
        part,
        group,
        rank,
        device,
        conv_kernel_size,
        sm_count=sm_count,
        heads=heads,
        stream=stream,
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
        A reused workspace of this rank; without one the call allocates its own.  One workspace may serve every
        prepared pack of the rank (``KcpWorkspace.reserve(ctxs, H)`` sizes it for all of them up front).
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
    *,
    recompute_handle: Optional[Any] = None,
    checkpoint_keep: Optional[str] = None,
    **options: Any,
) -> torch.Tensor:
    r"""Autograd-enabled context-parallel packed KDA chunked training step of this rank's fragments (the forward of
    :func:`chunk_kda_kcp_fwd`; the backward of :func:`chunk_kda_kcp_bwd` from its saved set).

    Parameters
    ----------
    recompute_handle : KcpRecomputeHandle, optional
        Keyword-only, default ``None`` (unchanged behavior).  A handle (``flashinfer.kda_train.KcpRecomputeHandle``
        or any object with an ``entries`` dict, such as the convolution's handle of the same layer) makes the
        ``torch.utils.checkpoint`` recompute of this call run without the forward state exchange: the original
        forward exchanges as always and stores its incoming states (the saved set's own tensors, key ``"kda"``) in
        the handle; the recompute inside the backward reads them back, so ``out`` and the gradients are bitwise
        those of the same checkpoint without a handle.  The backward still exchanges.  Create one handle per layer
        and micro-step, outside the outermost checkpoint region, and pass it as an argument of the checkpointed
        function; a forward outside the backward that finds an entry raises.  Refused with ``graph="replay"``,
        inside a CUDA-graph capture, with ``debug_sequence``, ``spec3_probe`` and ``boundary_mode="replay"``.  The
        exchange is a collective: every rank of ``ctx.group`` passes a handle for the same call or none, created at
        the same checkpoint nesting level; a mismatch hangs or corrupts data (no check is made).
    checkpoint_keep : str, optional
        With a handle only.  ``"states"`` (default): the handle keeps ``S_in`` and the call runs with
        ``save_transfer=False`` (bitwise the default form; the backward re-sends the maps).  ``"transfer"``: it also
        keeps the owner's c1 receive tensor (``save_transfer=True``; not with ``exchange="ring"``).  An explicit
        ``save_transfer`` must agree with the tier (``"states"`` -> ``False``, ``"transfer"`` -> ``True``); a
        contradicting value raises ``ValueError``.  The retained bytes are ``kcp_saved_bytes(ctx, H=...,
        checkpoint_keep=..., with_conv=False)``.

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
        recompute_handle=recompute_handle,
        checkpoint_keep=checkpoint_keep,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_workspace_bytes(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: Optional[bool] = None,
    C: int = 8192,
) -> int:
    """Bytes of this rank's KCP workspace for ``H`` heads: the local arena (that of :func:`kda_train_workspace_bytes`
    for a backward workspace; a forward-only one takes the KCP forward layout, 2432 B per internal token-head) plus
    the boundary pools of the context's collectives, as ``KcpWorkspace`` allocates them (the backward's send pools
    share the forward's storage; the ring exchange's hop pools, allocated by the first ``exchange="ring"`` call, are
    not counted).  The saved set and the convolution's halo rows are per-call tensors, so ``C`` does not change it,
    and ``save_transfer`` does not change a backward workspace.  A forward-only workspace (``backward=False``) has
    two states.  After its forwards with a saved set it holds the bytes returned here for the forward's
    ``save_transfer`` (``None``, the default, resolves to ``True`` as the a2a forward does; pass ``False`` for the
    ring or the recompute form); with ``True`` it holds no c1 receive, transfer staging or carry pool, with ``False``
    it holds the first two.  Once a forward without a saved set has allocated the rest, it holds
    ``cake_kcp_plan.kcp_arena_layout(ctx, H=H, backward=False, save_transfer=save_transfer)["lazy"]`` bytes more.
    Under ``exchange="ring"`` it never holds the c1 receive pool.  ``ctx`` may be a list of contexts: the bytes of
    one workspace that serves all of them (``KcpWorkspace.reserve(ctxs, H)``): each pool at its largest need and
    the largest local arena, not their sum."""
    from .experimental.cake_kda_train import cake_kcp_plan

    if save_transfer is None:
        save_transfer = cake_kcp_plan.SAVE_TRANSFER_DEFAULT
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
    save_transfer: Optional[bool] = None,
    with_conv: bool = True,
    C: int = 8192,
    checkpoint_keep: Optional[str] = None,
) -> int:
    """Bytes of this rank's KCP saved set for ``H`` heads: the local saved set (:func:`kda_train_saved_bytes`), the
    boundary states, and (``with_conv``) the convolution's history rows of ``C`` channels.  ``checkpoint_keep``
    (``"states"`` | ``"transfer"``): instead the bytes a recompute handle of that tier retains for the layer
    (:func:`chunk_kda_kcp`; plus one convolution's history rows with ``with_conv``).  ``save_transfer``: ``None``
    is ``False`` without a tier; with one, an explicit value that contradicts it raises ``ValueError``."""
    from .experimental.cake_kda_train import cake_kcp_plan

    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    return cake_kcp_plan.kcp_saved_bytes(
        ctx,
        H=H,
        save_mode=save_mode,
        save_transfer=save_transfer,
        with_conv=with_conv,
        C=C,
        checkpoint_keep=checkpoint_keep,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kcp_comm_bytes(
    ctx: Any,
    *,
    H: int = 64,
    C: int = 8192,
    save_transfer: Optional[bool] = None,
    boundary_mode: str = "saved",
    checkpoint_keep: Optional[str] = None,
) -> dict:
    """Bytes this rank sends and receives in one training step (forward and backward) for ``H`` heads and ``C``
    convolution channels: the state summaries, the boundary states and the halo rows of the context's collectives,
    as a table per collective and per boundary (``cake_kcp_plan.kcp_comm_bytes``).  ``checkpoint_keep``: the
    traffic of a checkpointed step with a recompute handle of that tier (the tier fixes ``save_transfer``; ``None``
    is ``False`` without a tier; an explicit value that contradicts the tier raises ``ValueError``)."""
    from .experimental.cake_kda_train import cake_kcp_plan

    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    return cake_kcp_plan.kcp_comm_bytes(
        ctx,
        H=H,
        C=C,
        save_transfer=save_transfer,
        boundary_mode=boundary_mode,
        checkpoint_keep=checkpoint_keep,
    )


def __getattr__(name: str) -> Any:
    # ``KcpRecomputeHandle`` (the KCP route's checkpoint recompute handle) resolves on first access, so importing this
    # module does not import the KCP host module (torch.distributed, the KCP loader), like the entry points above.
    if name == "KcpRecomputeHandle":
        from .experimental.cake_kda_train.cake_kcp import KcpRecomputeHandle

        return KcpRecomputeHandle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_SCALE",
    "KcpRecomputeHandle",  # noqa: F822 - resolved by __getattr__
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
