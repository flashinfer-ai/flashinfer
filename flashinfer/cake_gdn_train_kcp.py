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

Context-parallel (KCP) packed GDN (Gated DeltaNet) chunked training with
precise cross-rank state passing (SM100 / SM103 / SM107), the Cake backend,
with a choice of numerical mode.

Every entry takes ``deterministic: bool = True``:

* ``deterministic=True`` (the default) runs the deterministic route, the
  objects of ``flashinfer.cake_gdn_det_train_kcp``: bitwise reproducible at
  the same topology.
* ``deterministic=False`` runs the non-deterministic package
  ``flashinfer.experimental.cake_gdn_train_kcp``: the same contract and the
  same accuracy budgets, not bitwise equal to the deterministic route
  (run-to-run reproducible today for a fixed plan and partition; later
  versions may use unordered reductions).
* With ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  runs the deterministic route and warns once per process.

The mode is never stored in the context: a context, a partition and the byte
queries of either route describe the same plan.  A saved set is read only by
the backward of the route that produced it, and an options instance is
converted to the routed package's options type (mappings and keywords work for
both).  Thin experimental entry points: the partition / context plan, the host
orchestration, the saved set, the autograd Function and the generated kernels
live in the routed package; this module forwards every keyword.  Every switch
is a keyword argument; no environment variable is read.

Operands of one rank (its ``T_r`` local tokens in fragment order): ``q, k``
BF16 ``[1, T_r, H_qk, 128]`` (L2-normalized by the caller, or in the kernel with
``use_qk_l2norm_in_kernel=True`` on the deterministic route), ``v`` BF16
``[1, T_r, HV, 128]`` with ``HV % H_qk == 0`` (production ``(16, 48)``),
``g`` FP32 ``[1, T_r, HV]`` (the natural-log decay, preactivated, no clamp),
``beta`` FP32 ``[1, T_r, HV]`` (through the sigmoid).  Strided packed-QKV
views (token stride ``(2 H_qk + HV) * 128``) pass through without a copy.
The batch axis of 1 is optional; outputs take the rank of the inputs.
Gradients ``(dq, dk, dv, dbeta, dg)``: ``dq, dk, dv`` BF16, ``dbeta, dg``
FP32.  ``scale = 128 ** -0.5`` by default; ``T_r`` may be 0.

Usage::

    from flashinfer.cake_gdn_train_kcp import partition_kcp, build_context_kcp, chunk_gdn_kcp

    # deterministic=False selects the non-deterministic package; the default True is the deterministic route

    part = partition_kcp(cu_seqlens_global_cpu, world_size, layout="zigzag_per_document")   # once per batch layout
    ctx = build_context_kcp(part, group, rank, device, conv_kernel_size=4)                 # tables of this rank
    out = chunk_gdn_kcp(q, k, v, g, beta, ctx, save_mode="light", deterministic=False)     # autograd
    out = chunk_gdn_kcp(q, k, v, g, beta, ctx, boundary_mode="replay")                      # keep nothing; re-run the forward in backward
    out, saved = chunk_gdn_fwd_kcp(q, k, v, g, beta, ctx)
    dq, dk, dv, dbeta, dg = chunk_gdn_bwd_kcp(q, k, v, g, beta, ctx, dout, saved=saved)
    dq, dk, dv, dbeta, dg = chunk_gdn_bwd_kcp(q, k, v, g, beta, ctx, dout)                 # checkpoint replay
    gdn_comm_bytes_kcp(ctx, H_qk=16, HV=48, C=10240)                                       # bytes per collective

The width-4 causal convolution with halo of the same partition is
``flashinfer.cake_causal_conv1d_train.cake_causal_conv1d_kcp(x_local, weight, ctx)``
with this module's ``ctx``.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake context-parallel packed GDN chunked training (SM100/SM103/SM107)"

# The default scale of the contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


# The routed modules are imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from .experimental import cake_gdn_train_kcp  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gdn_train_kcp


@functools.cache
def _det():
    from . import cake_gdn_det_train_kcp  # noqa: PLC0415 - the deterministic route, imported on first use

    return cake_gdn_det_train_kcp


# One warning per process when torch's deterministic algorithms turn a ``deterministic=False`` call deterministic.
_DET_FALLBACK_WARNED = False


def _deterministic(deterministic: bool, entry: str) -> bool:
    """``True`` when ``entry`` runs the deterministic route: ``deterministic=True``, or ``deterministic=False`` while
    ``torch.are_deterministic_algorithms_enabled()`` (warned once per process)."""
    global _DET_FALLBACK_WARNED
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"{entry}: deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return True
    if torch.are_deterministic_algorithms_enabled():
        if not _DET_FALLBACK_WARNED:
            _DET_FALLBACK_WARNED = True
            warnings.warn(
                f"{entry}: deterministic=False runs the deterministic route because torch's deterministic "
                "algorithms are enabled (torch.use_deterministic_algorithms(True)); this warning is shown once",
                UserWarning,
                stacklevel=3,
            )
        return True
    return False


# A deterministic-route option the non-deterministic package does not have yet.
_DET_ONLY_L2N = "use_qk_l2norm_in_kernel"


def _nd_without_l2n(mapping: dict) -> dict:
    """``mapping`` for the non-deterministic route: ``use_qk_l2norm_in_kernel`` dropped when ``False``, refused when
    ``True`` (that route has no in-kernel Q / K L2 norm yet)."""
    if _DET_ONLY_L2N not in mapping:
        return mapping
    if mapping[_DET_ONLY_L2N]:
        raise ValueError(
            "the non-deterministic KCP route does not support use_qk_l2norm_in_kernel yet; "
            "pass deterministic=True for the in-kernel Q/K L2 norm"
        )
    return {k: v for k, v in mapping.items() if k != _DET_ONLY_L2N}


def _options_for(det: bool, options: Any) -> Any:
    """``options`` for the routed package: an options instance of the other route becomes the mapping of its fields
    (the routed package validates it); a mapping, ``None`` or the routed package's own instance pass through.  On the
    non-deterministic route ``use_qk_l2norm_in_kernel`` is dropped when ``False`` and refused when ``True``."""
    if options is None:
        return options
    if isinstance(options, dict):
        return options if det else _nd_without_l2n(options)
    own = _det().OptionsDetKcp if det else _pkg().OptionsKcp
    if isinstance(options, own) or not dataclasses.is_dataclass(options):
        return options
    fields = {f.name: getattr(options, f.name) for f in dataclasses.fields(options)}
    return fields if det else _nd_without_l2n(fields)


@flashinfer_experimental_api(feature=_FEATURE)
def partition_kcp(
    cu_seqlens_global_cpu: Any,
    world_size: int,
    layout: str = "contiguous",
    fragments: Optional[Any] = None,
    *,
    cut_points: Optional[Any] = None,
    deterministic: bool = True,
) -> Any:
    r"""Partition one packed batch of documents over the ranks of a context-parallel group.

    Parameters
    ----------
    cu_seqlens_global_cpu
        The packed documents' ``cu_seqlens`` (CPU int64 tensor or a sequence of ints; first 0).
    world_size : int
        Ranks of the context-parallel group.
    layout : str
        ``"contiguous"`` (near-equal token intervals), ``"zigzag"`` (``2 W`` parts, rank ``r`` holds parts ``r`` and
        ``2W-1-r``) or ``"zigzag_per_document"`` (every document cut into ``2 W`` pieces at ``cut_points``, default
        near-equal; rank ``r`` holds pieces ``r`` and ``2W-1-r`` of every document).  No ``total % W`` requirement;
        cut points need not be multiples of 64; zero-length fragments are legal.
    fragments : optional
        An explicit (ragged) table ``fragments[rank] = [(doc, offset_in_doc, length), ...]`` in local order,
        validated against ``cu_seqlens``.
    cut_points : optional
        Per document the ``2 W + 1`` cut offsets of ``"zigzag_per_document"``.

    deterministic : bool
        The route (see the module docstring); both routes build the same partition.

    Returns
    -------
    ``PartitionKcp`` (``PartitionDetKcp`` on the deterministic route): per rank the fragment table
    ``(doc, off, length, prev, next)`` in document order; reused across layers (metadata only).
    """
    if _deterministic(deterministic, "partition_kcp"):
        return _det().partition_det_kcp(
            cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
        )
    return _pkg().partition_kcp(
        cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
    )


@flashinfer_experimental_api(feature=_FEATURE)
def build_context_kcp(
    part: Any,
    group: Any,
    rank: int,
    device: Any,
    conv_kernel_size: int = 4,
    *,
    sm_count: Optional[int] = None,
    graph_replay: bool = False,
    deterministic: bool = True,
) -> Any:
    r"""This rank's view of a partition: local ``cu_seqlens`` (int32 device), the device fragment table, the crossing
    fragment lists (in / out) with their byte counts, the exchange plan, the convolution halo plan (at most
    ``conv_kernel_size - 1`` history rows per continuing fragment) and the segment plan.  ``group`` is the caller's
    ``torch.distributed`` process group (every collective of the route runs on it); CPU metadata plus small int32
    device tables built once; no tensor contents are read.  ``deterministic`` selects the route; the context holds no
    mode.  ``graph_replay=True`` builds the context for calls with ``graph="replay"``: the non-deterministic route then
    drops the one-chain (``P = 1``) rows of its faster plan, which save launches that graph replay removes; the
    deterministic route ignores the hint.
    """
    if not isinstance(graph_replay, bool):
        raise TypeError(
            f"build_context_kcp: graph_replay must be a bool, got {type(graph_replay).__name__}"
        )
    if _deterministic(deterministic, "build_context_kcp"):
        return _det().build_context_det_kcp(
            part, group, rank, device, conv_kernel_size, sm_count=sm_count
        )
    return _pkg().build_context_kcp(
        part,
        group,
        rank,
        device,
        conv_kernel_size,
        sm_count=sm_count,
        graph_replay=graph_replay,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_gdn_fwd_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: Optional[float] = None,
    save_for_backward: bool = True,
    workspace: Optional[Any] = None,
    options: Optional[Any] = None,
    deterministic: bool = True,
    **knobs: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Context-parallel packed GDN training forward of this rank's fragments.

    Parameters
    ----------
    q, k : torch.Tensor
        BF16 ``[1, T_r, H_qk, 128]`` (or ``[T_r, H_qk, 128]``); strided packed-QKV views accepted.
    v : torch.Tensor
        BF16 ``[1, T_r, HV, 128]``, ``HV % H_qk == 0``.
    g : torch.Tensor
        FP32 ``[1, T_r, HV]`` natural-log decay gate (preactivation form, no clamp).
    beta : torch.Tensor
        FP32 ``[1, T_r, HV]`` delta-rule gate after the sigmoid.
    ctx : ContextKcp
        This rank's context from :func:`build_context_kcp`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Keep the saved set (``boundary_mode="saved"``); ``False`` or ``boundary_mode="replay"`` returns ``None``.
    workspace : WorkspaceKcp, optional
        A reused caller-held workspace (``gdn_workspace_bytes_kcp``); without one the call allocates its own.
    options, **knobs
        Keyword-only options (``save_mode`` ``"light"`` | ``"full"``, ``boundary_mode`` ``"saved"`` | ``"replay"``,
        ``exchange`` ``"a2a"`` | ``"ring"``, ``save_transfer``, ``offset_class`` ``"fla"`` | ``"fp32"``, ``layout`` ``"tma"`` |
        ``"copy"``, ``group_reduce`` ``"fused"`` | ``"host"``, ``graph`` ``"off"`` | ``"replay"``,
        ``reuse_recompute_tapes`` ``"auto"`` (default) | ``True`` | ``False`` (``False`` with ``graph="replay"``; see
        :func:`chunk_gdn_kcp`),
        ``recompute_output`` ``"auto"`` | ``"always"`` (an autograd-node switch; this forward-only entry never skips a
        launch), ...).

    deterministic : bool
        The route (see the module docstring).

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (fresh); ``saved`` the saved set or ``None`` (readable by the same
    route's backward only).
    """
    if _deterministic(deterministic, "chunk_gdn_fwd_kcp"):
        return _det().chunk_gdn_det_fwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_for_backward=save_for_backward,
            workspace=workspace,
            options=_options_for(True, options),
            **knobs,
        )
    return _pkg().chunk_gdn_fwd_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        options=_options_for(False, options),
        deterministic=False,
        **_nd_without_l2n(knobs),
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_gdn_bwd_kcp(
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
    workspace: Optional[Any] = None,
    options: Optional[Any] = None,
    deterministic: bool = True,
    **knobs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Context-parallel packed GDN training backward.

    Parameters
    ----------
    q, k, v, g, beta, ctx
        The forward's operands and context.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : optional
        The forward's saved set; ``None`` recomputes it from the inputs (checkpoint replay: the summaries and the
        exchange are re-run).  A saved set of the other route is refused (``ValueError``).
    scale, workspace, options, **knobs
        As in :func:`chunk_gdn_fwd_kcp` (the same options as the forward's).
    deterministic : bool
        The route (see the module docstring); the forward's route.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v`` (the ``HV / H_qk`` value heads' contributions to
    ``dq`` / ``dk`` reduced in FP32 and rounded once), FP32 gradients of ``beta`` and ``g``; fresh tensors, ranked
    like the inputs.  Bitwise reproducible at the same topology on the deterministic route; within the same accuracy
    budgets, not bitwise equal to the deterministic route, on the non-deterministic one.
    """
    if _deterministic(deterministic, "chunk_gdn_bwd_kcp"):
        if saved is not None and not isinstance(saved, _det().SavedDetKcp):
            raise ValueError(
                "chunk_gdn_bwd_kcp: the saved set comes from the non-deterministic route "
                f"({type(saved).__name__}); the deterministic backward reads its own forward's saved set only"
            )
        return _det().chunk_gdn_det_bwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            scale=scale,
            workspace=workspace,
            options=_options_for(True, options),
            **knobs,
        )
    return _pkg().chunk_gdn_bwd_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        dout,
        saved=saved,
        scale=DEFAULT_SCALE if scale is None else scale,
        workspace=workspace,
        options=_options_for(False, options),
        deterministic=False,
        **_nd_without_l2n(knobs),
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_gdn_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: Optional[float] = None,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: Optional[Any] = None,
    graph: str = "off",
    options: Optional[Any] = None,
    recompute_handle: Optional[Any] = None,
    deterministic: bool = True,
    **knobs: Any,
) -> torch.Tensor:
    r"""Context-parallel packed GDN training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; every saved tensor travels through
    ``save_for_backward`` only, so a checkpointed forward retains nothing.  See :func:`chunk_gdn_fwd_kcp` for the
    operands and options and :func:`chunk_gdn_bwd_kcp` for the gradients.

    ``reuse_recompute_tapes`` (keyword-only, default ``"auto"``; ``True`` | ``False``): under
    ``torch.utils.checkpoint(use_reentrant=False)`` the ``save_mode="light"`` backward binds the forward tapes that
    torch's recompute of the forward (inside ``.backward()``) just wrote, instead of replaying the prep program and
    the state walk itself; the gradients are bitwise the same, the checkpointed forward still retains nothing, and
    the direct (non-checkpoint) autograd path, ``boundary_mode="replay"`` and ``save_mode="full"`` are unchanged.
    The kept tapes live from the recompute to this node's backward: when a later operation of the checkpointed
    region saves a tensor (an output projection, an MLP), they live through that operation's backward and raise the
    region's backward peak by the full tape set (one GB300, 261,987 tokens, ``(2, 6)``, GDN + a 768x768 projection
    + a SwiGLU MLP: 9250.54 MiB against 6927.31 MiB replayed, 21.29 against 22.70 ms per step of the region);
    ``"auto"`` (the default) keeps them only when this node is the region's last saver (the checkpoint-frame proof of
    ``recompute_output``) and takes the replay form otherwise -- a region this node closes keeps today's speed and
    memory; ``True`` always keeps them, ``False`` never (the replay form).  One kept set is
    live at a time: in a checkpoint segment holding several GDN layers only the last layer skips the replay (the
    earlier layers replay as before).  Not available with ``graph="replay"``: there the default resolves to
    ``False`` and an explicit ``True`` / ``"auto"`` is refused.

    ``recompute_output`` (keyword-only, ``"auto"`` default | ``"always"``): torch's recompute of a checkpointed
    forward returns nothing, so when this node is the last operation of the checkpointed region that saves a tensor,
    the recompute's output is read by nothing; ``"auto"`` leaves that output program out after reading the proof from
    torch's checkpoint frame (the node is executable, the remaining saved-tensor holders are exactly this node's,
    early stop is on, and the torch release -- ``torch.__version__`` without its local ``+cuNNN`` build tag -- is one
    of the package's verified releases, ``cake_kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS``) and launches it as before in every other state -- a later operation
    of the region that saves the output (another layer, a norm), a reentrant checkpoint, a forward outside a
    recompute, a non-executable node, another torch build (``cake_kcp.LAST_CALL["fwd"]["fwd_o_reason"]`` records the
    decision and its reason).  The gradients and the retained bytes are unchanged either way; ``"always"`` is the
    unconditional launch.

    ``recompute_handle`` (keyword-only, default ``None``; a :class:`RecomputeHandleKcp`): the checkpoint recompute
    of this forward runs without cross-rank communication.  The original forward exchanges as always and stores
    this rank's incoming states ``S_in`` (the saved set's own tensor) in the handle; torch's recompute of the forward
    inside ``.backward()`` reads them back and runs no exchange, no transfer-summary walk and no merge -- only the
    prep program, at ``P > 1`` the local segment walks and carry merges, and the state walk from ``S_in``, the
    launches of the light backward's replay -- so
    the gradients are bitwise those of the same checkpoint without a handle.  The backward still exchanges.  The
    handle retains ``S_in`` (``gdn_saved_bytes_kcp(ctx, ..., save_mode="light", save_transfer=False)``) from the
    original forward until the checkpoint frame holding it is freed, instead of retaining nothing (``with_conv=False``
    in that query: the convolution keeps its own rows).  Create one handle per layer and micro-step, outside the
    outermost checkpoint region, and pass it as an argument of the checkpointed function (``RecomputeHandleKcp`` of
    this module serves both routes)::

        def layer(x, handle):
            ...
            return chunk_gdn_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, RecomputeHandleKcp(), use_reentrant=False)

    A handle holds one entry per key: it serves this call and at most one context-parallel convolution call (whose
    halo rows go under the key ``"conv"``); a layer with separate Q / K / V convolutions uses one handle per
    convolution call, and this call shares one of them.  Needs ``boundary_mode="saved"`` and ``graph="off"``; implies ``save_transfer=False``.  A forward
    outside the backward that finds an entry raises (a handle reused across micro-steps).

    ``deterministic`` (default ``True``) selects the route (see the module docstring); the checkpoint recompute of a
    forward runs the forward's route as long as the torch deterministic-algorithms setting does not change in between.
    """
    if _deterministic(deterministic, "chunk_gdn_kcp"):
        return _det().chunk_gdn_det_kcp(
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
            graph=graph,
            options=_options_for(True, options),
            recompute_handle=recompute_handle,
            **knobs,
        )
    return _pkg().chunk_gdn_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_mode=save_mode,
        boundary_mode=boundary_mode,
        workspace=workspace,
        graph=graph,
        options=_options_for(False, options),
        recompute_handle=recompute_handle,
        deterministic=False,
        **_nd_without_l2n(knobs),
    )


@flashinfer_experimental_api(feature=_FEATURE)
def gdn_workspace_bytes_kcp(
    ctx: Any, *, H_qk: int, HV: int, deterministic: bool = True, **kw: Any
) -> int:
    r"""Caller-held workspace bytes of one rank for ``(ctx, H_qk, HV)`` (exchange pools, staging, tapes): the most
    the workspace binds in the forward or the backward of the given options (``reuse_recompute_tapes`` as a keyword;
    the per-phase terms are ``gdn_workspace_layout_kcp`` of the package) on the route ``deterministic`` selects."""
    if _deterministic(deterministic, "gdn_workspace_bytes_kcp"):
        return _det().gdn_det_workspace_bytes_kcp(ctx, H_qk=H_qk, HV=HV, **kw)
    return _pkg().gdn_workspace_bytes_kcp(ctx, H_qk=H_qk, HV=HV, **_nd_without_l2n(kw))


@flashinfer_experimental_api(feature=_FEATURE)
def gdn_saved_bytes_kcp(
    ctx: Any, *, H_qk: int, HV: int, deterministic: bool = True, **kw: Any
) -> int:
    r"""Saved-set bytes of one layer on this rank (``save_mode``, ``save_transfer`` as keywords), counted per
    crossing fragment, on the route ``deterministic`` selects."""
    if _deterministic(deterministic, "gdn_saved_bytes_kcp"):
        return _det().gdn_det_saved_bytes_kcp(ctx, H_qk=H_qk, HV=HV, **kw)
    return _pkg().gdn_saved_bytes_kcp(ctx, H_qk=H_qk, HV=HV, **_nd_without_l2n(kw))


@flashinfer_experimental_api(feature=_FEATURE)
def gdn_comm_bytes_kcp(
    ctx: Any,
    *,
    H_qk: int,
    HV: int,
    C: int = 10240,
    deterministic: bool = True,
    **kw: Any,
) -> dict:
    r"""Bytes this rank moves per collective (``c1``, ``c2``, ``b1``, ``b2``, ``halo_fwd``, ``halo_bwd``) and per
    boundary, counted per crossing fragment; ``C`` is the convolution channel count of the halo rows; on the route
    ``deterministic`` selects."""
    if _deterministic(deterministic, "gdn_comm_bytes_kcp"):
        return _det().gdn_det_comm_bytes_kcp(ctx, H_qk=H_qk, HV=HV, C=C, **kw)
    return _pkg().gdn_comm_bytes_kcp(ctx, H_qk=H_qk, HV=HV, C=C, **kw)


def is_gdn_train_kcp_supported(
    device: torch.device | str | int | None = None,
    *,
    deterministic: bool = True,
) -> bool:
    r"""``True`` when ``device`` (default: the current CUDA device) is an exported target of the package of the route
    ``deterministic`` selects."""
    if _deterministic(deterministic, "is_gdn_train_kcp_supported"):
        return _det().is_gdn_det_train_kcp_supported(device)
    if not torch.cuda.is_available():
        return False
    try:
        pkg = _pkg()
        from .experimental.cake_gdn_train_kcp import cake_jit  # noqa: PLC0415

        if device is None:
            index = torch.cuda.current_device()
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                return False
            index = dev.index if dev.index is not None else torch.cuda.current_device()
        return _device_arch(cake_jit, index) in cake_jit.ARCHES and pkg is not None
    except Exception:  # noqa: BLE001 - an unexported device or a package without its registry is "not supported"
        return False


def _device_arch(cake_jit: Any, index: int) -> str | None:
    """The exported ``sm_`` arch name of CUDA device ``index``: the loader's ``cake_backend.device_arch`` when the
    package ships it, else the registry's ``ARCH_BY_CAPABILITY`` map of ``torch.cuda.get_device_capability``."""
    try:
        from .experimental.cake_gdn_train_kcp import cake_backend  # noqa: PLC0415

        return cake_backend.device_arch(index)
    except ImportError:
        pass
    by_cap = getattr(cake_jit, "ARCH_BY_CAPABILITY", None) or {}
    return by_cap.get(tuple(torch.cuda.get_device_capability(index)))


# The public types of the package (the contract's exported names), resolved on first access so that importing the
# facade imports the package lazily: ``OptionsKcp``, ``ContextKcp``, ``PartitionKcp``.  ``SavedKcp`` (returned by
# ``chunk_gdn_fwd_kcp``) and ``WorkspaceKcp`` resolve the same way but are not public names of the contract.
_LAZY_TYPES = (
    "OptionsKcp",
    "ContextKcp",
    "PartitionKcp",
    "RecomputeHandleKcp",
    "SavedKcp",
    "WorkspaceKcp",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_pkg(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "ContextKcp",  # noqa: F822 - resolved by __getattr__
    "OptionsKcp",  # noqa: F822
    "PartitionKcp",  # noqa: F822
    "RecomputeHandleKcp",  # noqa: F822
    "build_context_kcp",
    "chunk_gdn_kcp",
    "chunk_gdn_bwd_kcp",
    "chunk_gdn_fwd_kcp",
    "gdn_comm_bytes_kcp",
    "gdn_saved_bytes_kcp",
    "gdn_workspace_bytes_kcp",
    "is_gdn_train_kcp_supported",
    "partition_kcp",
]
