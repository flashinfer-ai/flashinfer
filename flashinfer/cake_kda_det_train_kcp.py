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

Deterministic context-parallel (KCP) packed KDA (Kimi Delta Attention) chunked training
(SM100 / SM103 / SM107), the Cake backend.  ``flashinfer.kda_train`` keeps the names these
entry points had before (``chunk_kda_kcp``, ``kcp_partition``, ``KcpRecomputeHandle``, ...)
as compatibility aliases of the objects defined here.

The packed batch is split over the ranks of a process group (``partition_det_kcp``), every rank
holds its fragments of the documents, and the per-document state recurrences cross the rank
boundaries through the plan's collectives (``build_context_det_kcp``).  The per-rank launch
sequences, the autograd Function, the plan and the generated kernels live in
``flashinfer.experimental.cake_kda_det_train_kcp``; the single-rank entry points are in
``flashinfer.cake_kda_det_train``::

    from flashinfer.cake_kda_det_train_kcp import partition_det_kcp, build_context_det_kcp, chunk_kda_det_kcp

    part = partition_det_kcp(cu_seqlens_global_cpu, world_size)          # once per batch layout
    ctx = build_context_det_kcp(part, group, rank, device)               # tables and collectives of this rank
    out_local = chunk_kda_det_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd
    out_local, saved = chunk_kda_det_fwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)
    grads = chunk_kda_det_bwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx, dout_local, saved=saved)

    ctx_next = build_context_det_kcp(part_next, group, rank, device, heads=H)   # prepare the next pack while
                                                                               # this one runs: no host wait

The width-4 causal convolution with halo of the same partition takes this module's ``ctx``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api
from .cake_kda_det_train import _FEATURE, DEFAULT_SCALE

# The KCP entry points forward to ``flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp`` (the per-rank
# launch sequences over the kernels of the ``cake_kcp_jit`` registry) and ``cake_kcp_plan`` (the partition, the
# context's tables and collectives, the byte queries).  Every keyword is forwarded; nothing is validated here.


@flashinfer_experimental_api(feature=_FEATURE)
def partition_det_kcp(
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
    ``PartitionDetKcp``: per rank its fragments (document, token range) of the batch; the input of
    :func:`build_context_det_kcp` on every rank.
    """
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan

    return cake_kcp_plan.partition_det_kcp(
        cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
    )


@flashinfer_experimental_api(feature=_FEATURE)
def build_context_det_kcp(
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
    part : PartitionDetKcp
        From :func:`partition_det_kcp`, the same object on every rank.
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
    ``ContextDetKcp``: the argument every KCP entry point takes; reusable across calls of the same batch layout.
    """
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan

    return cake_kcp_plan.build_context_det_kcp(
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
def chunk_kda_det_fwd_kcp(
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
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Context-parallel packed KDA chunked training forward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta : torch.Tensor
        This rank's shard of the operands of :func:`~flashinfer.cake_kda_det_train.chunk_kda_det_fwd` (the tokens of its fragments, in the
        partition's order): BF16 ``[1, S_local, H, 128]`` ``q, k, v``, FP32 ``g``, BF16 ``[1, S_local, H]`` ``beta``.
    ctx : ContextDetKcp
        From :func:`build_context_det_kcp`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Write the saved set (the local saved set of :func:`~flashinfer.cake_kda_det_train.chunk_kda_det_fwd` plus the boundary states the backward
        starts from).
    save_mode : str
        ``"light"`` or ``"tape"`` as for :func:`~flashinfer.cake_kda_det_train.chunk_kda_det_fwd`.
    boundary_mode : str
        ``"saved"`` keeps the received boundary states in the saved set; ``"replay"`` saves nothing and lets the
        backward replay the forward's exchanges.
    workspace : WorkspaceDetKcp, optional
        A reused workspace of this rank; without one the call allocates its own.  One workspace may serve every
        prepared pack of the rank (``WorkspaceDetKcp.reserve(ctxs, H)`` sizes it for all of them up front).
    use_qk_l2norm_in_kernel : bool
        Keyword-only, default ``False`` (unchanged behavior).  ``True`` normalizes ``q`` and ``k`` inside the
        kernels: per token and per 128-channel head, ``q * rsqrt(sum(q * q) + 1e-6)`` in FP32, rounded once to BF16
        (likewise ``k``; ``v`` is not normalized), with the FP32 pullback fused into the BF16 ``dq`` / ``dk``.
        ``q`` / ``k`` are then this rank's raw rows and may be strided views (for example columns of a packed
        projection or convolution output: innermost stride 1, head stride 128 elements, token stride and base
        address 16-byte aligned, token stride >= H*128 (no overlapping rows); never copied).  The saved set keeps
        the raw ``q`` / ``k`` (the caller's tensors) and adds the FP32 per-token rstd ``[2, S_local, H]``;
        normalized ``q`` / ``k`` are never saved.  The workspace must be of the same mode
        (``WorkspaceDetKcp(use_qk_l2norm_in_kernel=True)``).  Until the single-rank in-kernel normalization
        programs are available on the device, ``True`` raises ``NotImplementedError`` on every rank alike.  With
        ``False`` every launch, saved tensor, key and byte is unchanged.
    **options
        Further KCP options (``save_transfer``, ``exchange``, ``head_groups``, ``offset_class``, ``graph``),
        forwarded.
        The raw gate mode of :func:`~flashinfer.cake_kda_det_train.chunk_kda_det_fwd` (keywords
        ``use_gate_in_kernel``, ``A_log``, ``dt_bias``, ``lower_bound``, forwarded like the options) applies to this
        rank's shard: with ``use_gate_in_kernel=True`` ``g`` is the rank's raw gate (BF16, any view under the
        single-rank stride rule, never copied), ``A_log`` FP32 ``[H]`` and ``dt_bias`` FP32 ``[H * 128]`` or
        ``[H, 128]`` are the full parameters (the same tensors on every rank) and ``lower_bound`` is required
        (``-5 <= lower_bound < 0``); the workspace must be of the same mode
        (``WorkspaceDetKcp(use_gate_in_kernel=True)``).

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (this rank's tokens); ``saved`` a ``SavedDetKcp`` or ``None``.
    """
    from .experimental.cake_kda_det_train_kcp import cake_kcp

    return cake_kcp.chunk_kda_det_fwd_kcp(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_det_bwd_kcp(
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
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, ...]:
    r"""Context-parallel packed KDA chunked training backward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta, ctx
        The forward's inputs (same contracts as :func:`chunk_kda_det_fwd_kcp`).
    dout : torch.Tensor
        BF16 gradient of this rank's ``out``.
    saved : SavedDetKcp, optional
        The forward's saved set; without one the forward is replayed (checkpointing).
    scale, save_mode, workspace, **options
        As for :func:`chunk_kda_det_fwd_kcp`; the backward reads its own options, among them ``spec3_probe``: the
        debug probe of the first saved-mode backward, ``"off"`` by default, ``"record"``, ``"warn"`` or
        ``"assert"`` (the same value on every rank); and the forward's raw gate keywords.
    use_qk_l2norm_in_kernel : bool
        Keyword-only, default ``False``: the forward's value.  With ``True`` ``dq`` / ``dk`` are the gradients of
        the raw ``q`` / ``k``; a saved set or a workspace of the other mode raises ``ValueError``.

    Returns
    -------
    Five tensors in the explicit gate mode (``use_gate_in_kernel=False``): ``(dq, dk, dv, dbeta, dg)`` of this
    rank's tokens, BF16 ``dq, dk, dv, dbeta``, FP32 ``dg``.  Seven tensors in raw gate mode
    (``use_gate_in_kernel=True``): ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, ``dg`` the BF16 gradient of the
    rank's raw gate, ``dA_log`` FP32 ``[H]`` and ``ddt_bias`` FP32 (``dt_bias``'s shape) this rank's partial sums --
    the gradients through its own tokens.  Sum them over the context-parallel group like the gradients of any
    replicated parameter, or with :func:`reduce_gate_param_grads_det_kcp` for a bitwise reproducible sum (an NCCL
    all-reduce does not fix its summation order).
    """
    from .experimental.cake_kda_det_train_kcp import cake_kcp

    return cake_kcp.chunk_kda_det_bwd_kcp(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_det_kcp(
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
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> torch.Tensor:
    r"""Autograd-enabled context-parallel packed KDA chunked training step of this rank's fragments (the forward of
    :func:`chunk_kda_det_fwd_kcp`; the backward of :func:`chunk_kda_det_bwd_kcp` from its saved set).

    Parameters
    ----------
    recompute_handle : RecomputeHandleDetKcp, optional
        Keyword-only, default ``None`` (unchanged behavior).  A handle (``flashinfer.cake_kda_det_train_kcp.RecomputeHandleDetKcp``
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
        contradicting value raises ``ValueError``.  The retained bytes are ``kda_det_saved_bytes_kcp(ctx, H=...,
        checkpoint_keep=..., with_conv=False)``.
    use_qk_l2norm_in_kernel : bool
        Keyword-only, default ``False``.  As for :func:`chunk_kda_det_fwd_kcp`; the gradients go to the raw ``q`` /
        ``k``, and a recompute handle's fingerprint records the mode (its entry does not change: the recompute
        writes its rstd afresh).
    **options
        As for :func:`chunk_kda_det_fwd_kcp`, the raw gate keywords included: in raw gate mode the gradients go to
        the raw gate (BF16), ``A_log`` and ``dt_bias`` (FP32; this rank's partial sums, see
        :func:`chunk_kda_det_bwd_kcp`), and a recompute handle's entry records the mode and ``lower_bound``.

    Returns
    -------
    torch.Tensor
        BF16 ``out`` like ``v`` (this rank's tokens).
    """
    from .experimental.cake_kda_det_train_kcp import cake_kcp

    return cake_kcp.chunk_kda_det_kcp(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kda_det_workspace_bytes_kcp(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: Optional[bool] = None,
    C: int = 8192,
    local_arena: Optional[str] = None,
    lean_passes: Optional[int] = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> int:
    """Bytes of this rank's KCP workspace for ``H`` heads: the local arena (that of :func:`~flashinfer.cake_kda_det_train.kda_det_train_workspace_bytes`
    for a backward workspace; a forward-only one takes the KCP forward layout, 2432 B per internal token-head) plus
    the boundary pools of the context's collectives, as ``WorkspaceDetKcp`` allocates them (the backward's send pools
    share the forward's storage; the ring exchange's hop pools, allocated by the first ``exchange="ring"`` call, are
    not counted).  The saved set and the convolution's halo rows are per-call tensors, so ``C`` does not change it,
    and ``save_transfer`` does not change a backward workspace.  A forward-only workspace (``backward=False``) has
    two states.  After its forwards with a saved set it holds the bytes returned here for the forward's
    ``save_transfer`` (``None``, the default, resolves to ``True`` as the a2a forward does; pass ``False`` for the
    ring or the recompute form); with ``True`` it holds no c1 receive, transfer staging or carry pool, with ``False``
    it holds the first two.  Once a forward without a saved set has allocated the rest, it holds
    ``cake_kcp_plan.kcp_arena_layout(ctx, H=H, backward=False, save_transfer=save_transfer)["lazy"]`` bytes more.
    Under ``exchange="ring"`` it never holds the c1 receive pool.  ``ctx`` may be a list of contexts: the bytes of
    one workspace that serves all of them (``WorkspaceDetKcp.reserve(ctxs, H)``): each pool at its largest need and
    the largest local arena, not their sum.  ``local_arena`` / ``lean_passes`` are those of the workspace
    (``WorkspaceDetKcp(local_arena="lean", lean_passes=...)``, opt-in: the backward's lean local arena, its tail in
    ``lean_passes`` passes, two by default, with its small tail-count table, on the head counts it serves);
    ``None`` keeps the arena above.  ``use_gate_in_kernel``: the workspace's gate mode
    (``WorkspaceDetKcp(use_gate_in_kernel=True)``: the raw mode's local arena, the replay arena; the pools do not
    change).  ``use_qk_l2norm_in_kernel``: the workspace's in-kernel Q / K L2 norm mode
    (``WorkspaceDetKcp(use_qk_l2norm_in_kernel=True)``: the local arena of that mode, the replay arena; the pools do
    not change)."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan

    if save_transfer is None:
        save_transfer = cake_kcp_plan.SAVE_TRANSFER_DEFAULT
    return cake_kcp_plan.kda_det_workspace_bytes_kcp(
        ctx,
        H=H,
        save_mode=save_mode,
        backward=backward,
        save_transfer=save_transfer,
        C=C,
        local_arena=local_arena,
        lean_passes=lean_passes,
        use_gate_in_kernel=use_gate_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def reduce_gate_param_grads_det_kcp(
    ctx: Any, dA_log: torch.Tensor, ddt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw gate mode's full ``(dA_log, ddt_bias)`` from every rank's partial sums (:func:`chunk_kda_det_bwd_kcp`),
    bitwise reproducible: one all-gather on ``ctx.group``, then an FP32 sum in rank order on every rank, so every
    rank gets the same bits under every schedule.  An NCCL all-reduce gives the same sum to FP32 rounding but does
    not fix its summation order.  A collective: every rank of ``ctx.group`` calls it.  With ``world_size == 1``
    copies of the inputs; with ``world_size > 1`` and no initialized process group it raises ``RuntimeError``."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp

    return cake_kcp.reduce_gate_param_grads_det_kcp(ctx, dA_log, ddt_bias)


@flashinfer_experimental_api(feature=_FEATURE)
def kda_det_saved_bytes_kcp(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    save_transfer: Optional[bool] = None,
    with_conv: bool = True,
    C: int = 8192,
    checkpoint_keep: Optional[str] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> int:
    """Bytes of this rank's KCP saved set for ``H`` heads: the local saved set (:func:`~flashinfer.cake_kda_det_train.kda_det_train_saved_bytes`), the
    boundary states, and (``with_conv``) the convolution's history rows of ``C`` channels.  ``checkpoint_keep``
    (``"states"`` | ``"transfer"``): instead the bytes a recompute handle of that tier retains for the layer
    (:func:`chunk_kda_det_kcp`; plus one convolution's history rows with ``with_conv``).  ``save_transfer``: ``None``
    is ``False`` without a tier; with one, an explicit value that contradicts it raises ``ValueError``.
    ``use_qk_l2norm_in_kernel``: the saved set of the in-kernel Q / K L2 norm adds its FP32 rstd, ``8 * S_local * H``
    bytes (a recompute handle's bytes do not change)."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan

    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    return cake_kcp_plan.kda_det_saved_bytes_kcp(
        ctx,
        H=H,
        save_mode=save_mode,
        save_transfer=save_transfer,
        with_conv=with_conv,
        C=C,
        checkpoint_keep=checkpoint_keep,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kda_det_comm_bytes_kcp(
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
    as a table per collective and per boundary (``cake_kcp_plan.kda_det_comm_bytes_kcp``).  ``checkpoint_keep``: the
    traffic of a checkpointed step with a recompute handle of that tier (the tier fixes ``save_transfer``; ``None``
    is ``False`` without a tier; an explicit value that contradicts the tier raises ``ValueError``)."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan

    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    return cake_kcp_plan.kda_det_comm_bytes_kcp(
        ctx,
        H=H,
        C=C,
        save_transfer=save_transfer,
        boundary_mode=boundary_mode,
        checkpoint_keep=checkpoint_keep,
    )


# The types of the route, resolved on first access so that importing this module imports neither the KCP host
# module (torch.distributed, the KCP loader) nor the plan: public name -> the submodule that defines it.
_LAZY_TYPES = {
    "ContextDetKcp": "cake_kcp_plan",
    "OptionsDetKcp": "cake_kcp",
    "PartitionDetKcp": "cake_kcp_plan",
    "RecomputeHandleDetKcp": "cake_kcp",
    "SavedDetKcp": "cake_kcp",
    "WorkspaceDetKcp": "cake_kcp",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_TYPES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415

    return getattr(
        import_module(f"flashinfer.experimental.cake_kda_det_train_kcp.{submodule}"),
        name,
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "DEFAULT_SCALE",
    "ContextDetKcp",  # noqa: F822 - resolved by __getattr__
    "OptionsDetKcp",  # noqa: F822
    "PartitionDetKcp",  # noqa: F822
    "RecomputeHandleDetKcp",  # noqa: F822
    "SavedDetKcp",  # noqa: F822
    "WorkspaceDetKcp",  # noqa: F822
    "build_context_det_kcp",
    "chunk_kda_det_bwd_kcp",
    "chunk_kda_det_fwd_kcp",
    "chunk_kda_det_kcp",
    "kda_det_comm_bytes_kcp",
    "kda_det_saved_bytes_kcp",
    "kda_det_workspace_bytes_kcp",
    "partition_det_kcp",
    "reduce_gate_param_grads_det_kcp",
]
