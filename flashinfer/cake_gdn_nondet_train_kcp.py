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
precise cross-rank state passing (SM100 / SM103 / SM107), the Cake backend:
the entry points of the non-deterministic package
``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp``.

Mode
----

* The mode is in the name: this module runs the non-deterministic package
  only.  No entry takes a ``deterministic`` keyword (a stray one is a
  ``TypeError``), and no call is routed to, or falls back to, another package.
* Every launch of the package is fixed-order today, so
  ``torch.use_deterministic_algorithms(True)``, with or without
  ``warn_only=True``, runs it unchanged: no alert, no warning and no fallback.
  The contract does not promise run-to-run bits.
* The deterministic counterpart is ``flashinfer.cake_gdn_det_train_kcp``, a
  separate package with its own objects.  The two packages use the same
  partition algorithm, but their head plans differ (the per-rank ``P`` rule
  and the table layout of the context), so use each package's own context.
  This module refuses an options instance of another package (``TypeError``);
  the backward refuses a saved set of another package (``ValueError``).
* The package keeps the deterministic package's contract and accuracy
  budgets; its results are not bitwise equal to that package's.  It also takes
  ``save_mode="light_akk"``: the light saved set plus the forward's BF16
  ``Akk`` tape, which the backward's prep replay reads instead of solving the
  block inverse; it needs ``use_qk_l2norm_in_kernel=True`` and is the default
  there (a call that leaves ``save_mode`` unset), while calls without the norm
  default to ``"light"``.

A stable FlashInfer API: thin ``@flashinfer_api`` entry points that import the
package on first use.  The partition / context plan, the host orchestration,
the saved set, the autograd Function and the generated kernels live in the
package; this module forwards every keyword.  Every switch is a keyword
argument; no environment variable is read.

Operands of one rank (its ``T_r`` local tokens in fragment order): ``q, k``
BF16 ``[1, T_r, H_qk, 128]`` (L2-normalized by the caller, or raw with
``use_qk_l2norm_in_kernel=True``), ``v`` BF16
``[1, T_r, HV, 128]`` with ``HV % H_qk == 0`` (production ``(16, 48)``),
``g`` FP32 ``[1, T_r, HV]`` (the natural-log decay, preactivated, no clamp),
``beta`` FP32 ``[1, T_r, HV]`` (through the sigmoid).  Strided packed-QKV
views (token stride ``(2 H_qk + HV) * 128``) pass through without a copy.
The batch axis of 1 is optional; outputs take the rank of the inputs.
Gradients ``(dq, dk, dv, dbeta, dg)``: ``dq, dk, dv`` BF16, ``dbeta, dg``
FP32.  ``scale = 128 ** -0.5`` by default; ``T_r`` may be 0.

Usage::

    from flashinfer.cake_gdn_nondet_train_kcp import (
        cake_gdn_nondet_kcp,
        cake_gdn_nondet_train_kcp_backward,
        cake_gdn_nondet_train_kcp_build_context,
        cake_gdn_nondet_train_kcp_comm_bytes,
        cake_gdn_nondet_train_kcp_forward,
        cake_gdn_nondet_train_kcp_partition,
    )

    part = cake_gdn_nondet_train_kcp_partition(cu_seqlens_global_cpu, world_size, layout="zigzag_per_document")
    ctx = cake_gdn_nondet_train_kcp_build_context(part, group, rank, device, conv_kernel_size=4)
    out = cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, save_mode="light")          # autograd
    out = cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, boundary_mode="replay")     # keep nothing; re-run the forward in backward
    out, saved = cake_gdn_nondet_train_kcp_forward(q, k, v, g, beta, ctx)
    dq, dk, dv, dbeta, dg = cake_gdn_nondet_train_kcp_backward(q, k, v, g, beta, ctx, dout, saved=saved)
    dq, dk, dv, dbeta, dg = cake_gdn_nondet_train_kcp_backward(q, k, v, g, beta, ctx, dout)   # checkpoint replay
    cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=16, HV=48, C=10240)                         # bytes per collective

The partition is computed once per batch layout and the context once per rank;
both are reused across layers.

The width-4 causal convolution with halo of the same partition is
``flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp(x_local, weight, ctx)``
with this module's ``ctx``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api

# The default scale of the contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


# The package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from .experimental.cake import cake_gdn_nondet_train_kcp  # noqa: PLC0415 - the backend package, imported on first use

    return cake_gdn_nondet_train_kcp


class _PackageKeywordsSignature(inspect.Signature):
    """``__signature__`` of a facade function that hands its ``**`` keywords to the package function of its name: the
    facade function's own parameters without ``**``, then each keyword-only parameter of the package function that the
    facade function does not name (every option field, keyword-only).  The package parameters are read on first use of
    the signature, which imports the package, so importing this module still imports nothing; the facade function
    stays a plain function (its source and type hints are its own)."""

    __slots__ = ("_own", "_name", "_full")

    def __init__(
        self, fn: Callable[..., Any]
    ) -> None:  # no Signature.__init__: the parameters are read on first use
        self._own = inspect.signature(fn)
        self._name = fn.__name__
        self._full: Optional[inspect.Signature] = None

    def _resolved(self) -> inspect.Signature:
        if self._full is None:
            params = [
                p for p in self._own.parameters.values() if p.kind is not p.VAR_KEYWORD
            ]
            named = {p.name for p in params}
            theirs = inspect.signature(getattr(_pkg(), self._name)).parameters
            params += [
                p
                for p in theirs.values()
                if p.kind is p.KEYWORD_ONLY and p.name not in named
            ]
            self._full = self._own.replace(parameters=params)
        return self._full

    # every Signature method (parameters, bind, str, ==, hash) reads these two slots of the base class
    @property  # type: ignore[override]
    def _parameters(self) -> Any:
        return self._resolved().parameters

    @property  # type: ignore[override]
    def _return_annotation(self) -> Any:
        return self._own.return_annotation

    def replace(self, **changes: Any) -> inspect.Signature:  # type: ignore[override]
        return self._resolved().replace(**changes)

    def __reduce__(self) -> Any:
        return self._resolved().__reduce__()

    def __repr__(self) -> str:
        return repr(self._resolved())


def _forwards_keywords(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``fn`` as handing its ``**`` keywords to the package function of its name: its ``__signature__`` is a
    :class:`_PackageKeywordsSignature`."""
    fn.__signature__ = _PackageKeywordsSignature(fn)  # type: ignore[attr-defined]
    return fn


def _options_for(options: Any) -> Any:
    """``options`` for the package: ``None``, a dict (the package validates it) or this package's own
    ``CakeGDNNondetKcpOptions`` pass through.  An options instance of another package (the deterministic one's
    included) is a ``TypeError``: its fields are not converted."""
    if options is None or isinstance(options, dict):
        return options
    if isinstance(options, _pkg().CakeGDNNondetKcpOptions):
        return options
    raise TypeError(
        "options must be a CakeGDNNondetKcpOptions, a dict or None, got "
        f"{type(options).__module__}.{type(options).__qualname__}"
    )


@flashinfer_api
def cake_gdn_nondet_train_kcp_partition(
    cu_seqlens_global_cpu: Any,
    world_size: int,
    layout: str = "contiguous",
    fragments: Optional[Any] = None,
    *,
    cut_points: Optional[Any] = None,
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

    Returns
    -------
    ``CakeGDNNondetKcpPartition``: per rank the fragment table
    ``(doc, off, length, prev, next)`` in document order; reused across layers (metadata only).
    """
    return _pkg().cake_gdn_nondet_train_kcp_partition(
        cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
    )


@flashinfer_api
def cake_gdn_nondet_train_kcp_build_context(
    part: Any,
    group: Any,
    rank: int,
    device: Any,
    conv_kernel_size: int = 4,
    *,
    sm_count: Optional[int] = None,
    graph_replay: bool = False,
) -> Any:
    r"""This rank's view of a partition: local ``cu_seqlens`` (int32 device), the device fragment table, the crossing
    fragment lists (in / out) with their byte counts, the exchange plan, the convolution halo plan (at most
    ``conv_kernel_size - 1`` history rows per continuing fragment) and the segment plan.  ``group`` is the caller's
    ``torch.distributed`` process group (every collective of the package runs on it); CPU metadata plus small int32
    device tables built once; no tensor contents are read.  The context holds no mode; its head plan is this
    package's own.  ``graph_replay=True`` builds the context for calls with ``graph="replay"``: the package then drops the
    one-chain (``P = 1``) rows of its faster plan, which save launches that graph replay removes.

    Parameters
    ----------
    part
        Partition returned by :func:`cake_gdn_nondet_train_kcp_partition`.
    group
        Caller-owned ``torch.distributed`` process group for all collectives.
    rank : int
        This rank's index within ``group``.
    device
        CUDA device on which to build this rank's tables.
    conv_kernel_size : int, default=4
        Convolution width used to plan the history-row halo exchange.
    sm_count : int, optional
        SM count to use when building the segment plan.
    graph_replay : bool, default=False
        Build the context for calls with ``graph="replay"``.
    """
    if not isinstance(graph_replay, bool):
        raise TypeError(
            f"cake_gdn_nondet_train_kcp_build_context: graph_replay must be a bool, got {type(graph_replay).__name__}"
        )
    return _pkg().cake_gdn_nondet_train_kcp_build_context(
        part,
        group,
        rank,
        device,
        conv_kernel_size,
        sm_count=sm_count,
        graph_replay=graph_replay,
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_train_kcp_forward(
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
    ctx : CakeGDNNondetKcpContext
        This rank's context from :func:`cake_gdn_nondet_train_kcp_build_context`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Keep the saved set (``boundary_mode="saved"``); ``False`` or ``boundary_mode="replay"`` returns ``None``.
    workspace : CakeGDNNondetKcpWorkspace, optional
        A reused caller-held workspace (``cake_gdn_nondet_train_kcp_workspace_bytes``); without one the call allocates its own.
    options
        Keyword-only options (``save_mode`` ``"light"`` | ``"full"`` | ``"light_akk"`` (see the module docstring;
        unset: ``"light_akk"`` with ``use_qk_l2norm_in_kernel=True``, else ``"light"``),
        ``boundary_mode`` ``"saved"`` | ``"replay"``,
        ``exchange`` ``"a2a"`` | ``"ring"``, ``save_transfer``, ``offset_class`` ``"fla"`` | ``"fp32"``, ``layout`` ``"tma"`` |
        ``"copy"``, ``group_reduce`` ``"fused"`` | ``"host"``, ``graph`` ``"off"`` | ``"replay"``,
        ``reuse_recompute_tapes`` ``"auto"`` (default) | ``True`` | ``False`` (see :func:`cake_gdn_nondet_kcp`),
        ``recompute_output`` ``"auto"`` | ``"always"`` (an autograd-node switch; this forward-only entry never skips a
        launch), ...).
    **knobs
        Keyword overrides for the backend options described above.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (fresh); ``saved`` the saved set or ``None`` (readable by this
    package's backward only).
    """
    return _pkg().cake_gdn_nondet_train_kcp_forward(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        options=_options_for(options),
        **knobs,
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_train_kcp_backward(
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
        exchange are re-run).  A saved set of the deterministic package is refused (``ValueError``).
    scale, workspace, options
        As in :func:`cake_gdn_nondet_train_kcp_forward` (the same options as the forward's).
    **knobs
        Additional backend options, as in :func:`cake_gdn_nondet_train_kcp_forward`.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v`` (the ``HV / H_qk`` value heads' contributions to
    ``dq`` / ``dk`` reduced in FP32 and rounded once), FP32 gradients of ``beta`` and ``g``; fresh tensors, ranked
    like the inputs; within the deterministic package's accuracy budgets, not bitwise equal to its results.
    """
    return _pkg().cake_gdn_nondet_train_kcp_backward(
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
        options=_options_for(options),
        **knobs,
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: Optional[float] = None,
    save_mode: Optional[str] = None,
    boundary_mode: str = "saved",
    workspace: Optional[Any] = None,
    graph: str = "off",
    options: Optional[Any] = None,
    recompute_handle: Optional[Any] = None,
    **knobs: Any,
) -> torch.Tensor:
    r"""Context-parallel packed GDN training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; every saved tensor travels through
    ``save_for_backward`` only, so a checkpointed forward retains nothing.  See :func:`cake_gdn_nondet_train_kcp_forward` for the
    operands and options and :func:`cake_gdn_nondet_train_kcp_backward` for the gradients.

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
    earlier layers replay as before).  Under ``graph="replay"`` a caller-held workspace keeps its own tapes under
    ``graph="replay"`` and the backward binds them only while the workspace made no other call in between (else the
    replay form); the kept tapes add no memory there, while the recompute's own hand-off graph holds its static
    results like every graph key (one more saved set per checkpointed layer and, when that recompute launches the
    output program -- ``recompute_output="always"``, or ``True`` with a later saver -- a full static output); there
    ``"auto"`` decides only from ``cake_kcp.GRAPH_RECOMPUTE_MIN_ROWS`` token x value-head rows per rank (a smaller
    recompute replays the original forward's graph: its host-bound step gains nothing from the hand-off).

    ``recompute_output`` (keyword-only, ``"auto"`` default | ``"always"``): torch's recompute of a checkpointed
    forward returns nothing, so when this node is the last operation of the checkpointed region that saves a tensor,
    the recompute's output is read by nothing; ``"auto"`` leaves that output program out after reading the proof from
    torch's checkpoint frame (the node is executable, the remaining saved-tensor holders are exactly this node's,
    early stop is on, and the torch release -- ``torch.__version__`` without its local ``+cuNNN`` build tag -- is one
    of the package's verified releases, ``cake_kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS``) and launches it as before in every other state -- a later operation
    of the region that saves the output (another layer, a norm), a reentrant checkpoint, a forward outside a
    recompute, a non-executable node, another torch build, and under ``graph="replay"`` with the default
    ``reuse_recompute_tapes="auto"`` a rank below ``cake_kcp.GRAPH_RECOMPUTE_MIN_ROWS`` (reason ``"small"``; an
    explicit ``True`` / ``False`` reads the proof at every size) (``cake_kcp.LAST_CALL["fwd"]["fwd_o_reason"]``
    records the decision and its reason).  The gradients and the retained bytes are unchanged either way;
    ``"always"`` is the unconditional launch.

    ``recompute_handle`` (keyword-only, default ``None``; a :class:`CakeGDNNondetKcpRecomputeHandle`): the checkpoint recompute
    of this forward runs without cross-rank communication.  The original forward exchanges as always and stores
    this rank's incoming states ``S_in`` (the saved set's own tensor) in the handle; torch's recompute of the forward
    inside ``.backward()`` reads them back and runs no exchange, no transfer-summary walk and no merge -- only the
    prep program, at ``P > 1`` the local segment walks and carry merges, and the state walk from ``S_in``, the
    launches of the light backward's replay -- so
    the gradients are bitwise those of the same checkpoint without a handle.  The backward still exchanges.  The
    handle retains ``S_in`` (``cake_gdn_nondet_train_kcp_saved_bytes(ctx, ..., save_mode="light", save_transfer=False)``) from the
    original forward until the checkpoint frame holding it is freed, instead of retaining nothing (``with_conv=False``
    in that query: the convolution keeps its own rows).  Create one handle per layer and micro-step, outside the
    outermost checkpoint region, and pass it as an argument of the checkpointed function (``CakeGDNNondetKcpRecomputeHandle`` of
    this module)::

        def layer(x, handle):
            ...
            return cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, CakeGDNNondetKcpRecomputeHandle(), use_reentrant=False)

    A handle holds one entry per key: it serves this call and at most one context-parallel convolution call (whose
    halo rows go under the key ``"conv"``); a layer with separate Q / K / V convolutions uses one handle per
    convolution call, and this call shares one of them.  Needs ``boundary_mode="saved"`` and ``graph="off"``; implies ``save_transfer=False``.  A forward
    outside the backward that finds an entry raises (a handle reused across micro-steps).

    ``q`` / ``k`` / ``v`` that are head views of one packed ``[..., T, (2 H_qk + HV) * 128]`` row (the convolution's
    output) get the full backward speed as the views of :func:`cake_gdn_nondet_train_kcp_packed_qkv_views` (or of
    the deterministic package's ``cake_gdn_det_train_kcp_packed_qkv_views``): both helpers' split nodes pass a packed
    gradient row on without a copy.  The package writes that row only for the outputs of either helper.  Plain ``split`` /
    ``view`` views of the row give the same results, but torch's split backward concatenates ``dq`` / ``dk`` / ``dv``
    into a new row: about 0.2 ms (6-7 %) more backward per call at ``(16, 48)`` and 32,768
    tokens on B200 and GB300 and 0.17 ms (7-8 %) on R200, with the in-kernel norm off or on
    (``flashinfer/experimental/cake/cake_gdn_nondet_train_kcp/README.md``, "Packed Q/K/V").

    ``torch.use_deterministic_algorithms`` changes neither the forward nor its checkpoint recompute (see the module
    docstring).
    """
    return _pkg().cake_gdn_nondet_kcp(
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
        options=_options_for(options),
        recompute_handle=recompute_handle,
        **knobs,
    )


@flashinfer_api
def cake_gdn_nondet_train_kcp_packed_qkv_views(
    qkv: torch.Tensor, *, H_qk: int, HV: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``q`` / ``k`` / ``v`` as head views ``[..., T, H_qk | H_qk | HV, 128]`` of a packed
    ``[..., T, (2 H_qk + HV) * 128]`` row, with no copy.  :func:`cake_gdn_nondet_kcp` takes the views as they are and
    its backward writes ``dq`` / ``dk`` / ``dv`` into one packed row, which this split's backward returns as the
    gradient of ``qkv`` without a concatenation.  Plain ``split`` views of ``qkv`` give the same gradients through
    torch's split backward, which concatenates them (:func:`cake_gdn_nondet_kcp`).

    Needs a dense channel axis (``qkv.stride(-1) == 1``).  The views are the outputs of one custom autograd node:
    hooks and ``retain_grad`` on them fire as on any autograd output; its backward concatenates instead when the
    incoming gradients are not the head views of one packed row (a missing gradient counts as zeros, a hook that
    returns a new gradient); autograd refuses an in-place change to the views while ``qkv`` requires grad.

    Example::

        q, k, v = cake_gdn_nondet_train_kcp_packed_qkv_views(conv_out, H_qk=16, HV=48)   # conv_out: [T, 10240]
        out = cake_gdn_nondet_kcp(q, k, v, g, beta, ctx)

    Parameters
    ----------
    qkv : torch.Tensor
        Packed projection rows ``[..., T, (2 * H_qk + HV) * 128]`` with unit channel stride.
    H_qk : int
        Number of query and key heads, each with 128 channels.
    HV : int
        Number of value heads, each with 128 channels.
    """
    return _pkg().cake_gdn_nondet_train_kcp_packed_qkv_views(qkv, H_qk=H_qk, HV=HV)


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_train_kcp_workspace_bytes(
    ctx: Any, *, H_qk: int, HV: int, **kw: Any
) -> int:
    r"""Caller-held workspace bytes of one rank for ``(ctx, H_qk, HV)`` (exchange pools, staging, tapes): the most
    the workspace binds in the forward or the backward of the given options (``reuse_recompute_tapes`` as a keyword;
    the per-phase terms are ``cake_bytes.cake_gdn_nondet_train_kcp_workspace_layout`` of the package).
    ``save_mode="light_akk"`` gives the figure of a call outside a checkpoint: under
    ``torch.utils.checkpoint(use_reentrant=False)`` that call runs as ``"light"`` and binds the ``save_mode="light"``
    figure, so a checkpointed ``"light_akk"`` layer is sized with ``save_mode="light"``.

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_nondet_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    **kw
        Backend options for this query, including the call's save and boundary settings.
    """
    return _pkg().cake_gdn_nondet_train_kcp_workspace_bytes(ctx, H_qk=H_qk, HV=HV, **kw)


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_train_kcp_saved_bytes(
    ctx: Any, *, H_qk: int, HV: int, **kw: Any
) -> int:
    r"""Saved-set bytes of one layer on this rank (``save_mode``, ``save_transfer`` as keywords), counted per
    crossing fragment.

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_nondet_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    **kw
        Backend options for this query, including the call's save and boundary settings.
    """
    return _pkg().cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV, **kw)


@flashinfer_api
@_forwards_keywords
def cake_gdn_nondet_train_kcp_comm_bytes(
    ctx: Any,
    *,
    H_qk: int,
    HV: int,
    C: int = 10240,
    **kw: Any,
) -> dict:
    r"""Bytes this rank moves per collective (``c1``, ``c2``, ``b1``, ``b2``, ``halo_fwd``, ``halo_bwd``) and per
    boundary, counted per crossing fragment; ``C`` is the convolution channel count of the halo rows.

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_nondet_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    C : int, default=10240
        Convolution channel count used to size halo traffic.
    **kw
        Backend options for this query: ``save_transfer`` and ``boundary_mode``.
    """
    return _pkg().cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C, **kw)


@flashinfer_api
def is_cake_gdn_nondet_train_kcp_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    r"""``True`` when ``device`` (default: the current CUDA device) is an exported target of the package
    (``sm_100a``, ``sm_103a``, ``sm_107a``).  ``False`` without CUDA, for a device that is not a CUDA device and for
    a device that cannot be queried.  A package that fails to import raises: it is not reported as unsupported.

    Parameters
    ----------
    device : torch.device, str, int, optional
        Device to query; ``None`` selects the current CUDA device.
    """
    from .experimental.cake.cake_gdn_nondet_train_kcp import cake_backend  # noqa: PLC0415 - imported on first use

    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            index = torch.cuda.current_device()
        else:
            dev = torch.device(device)
            if dev.type != "cuda":
                return False
            index = dev.index if dev.index is not None else torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
    except Exception:  # noqa: BLE001 - a device that cannot be queried is "not supported"
        return False
    return cake_backend.arch_for_capability(capability) is not None


# The public types of the package, resolved on first access so that importing this module does not import the
# package: the options, the partition and the context, the recompute handle, the saved set (returned by
# ``cake_gdn_nondet_train_kcp_forward``), the workspace and the autograd Function.
_LAZY_TYPES = (
    "CakeGDNNondetKcpContext",
    "CakeGDNNondetKcpFunction",
    "CakeGDNNondetKcpOptions",
    "CakeGDNNondetKcpPartition",
    "CakeGDNNondetKcpRecomputeHandle",
    "CakeGDNNondetKcpSaved",
    "CakeGDNNondetKcpWorkspace",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_pkg(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "CakeGDNNondetKcpContext",  # noqa: F822 - resolved by __getattr__
    "CakeGDNNondetKcpFunction",  # noqa: F822
    "CakeGDNNondetKcpOptions",  # noqa: F822
    "CakeGDNNondetKcpPartition",  # noqa: F822
    "CakeGDNNondetKcpRecomputeHandle",  # noqa: F822
    "CakeGDNNondetKcpSaved",  # noqa: F822
    "CakeGDNNondetKcpWorkspace",  # noqa: F822
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_build_context",
    "cake_gdn_nondet_train_kcp_comm_bytes",
    "cake_gdn_nondet_train_kcp_forward",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
    "cake_gdn_nondet_train_kcp_partition",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
    "is_cake_gdn_nondet_train_kcp_supported",
]
