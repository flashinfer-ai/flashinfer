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

Deterministic context-parallel (KCP) packed GDN (Gated DeltaNet) chunked
training with precise cross-rank state passing (SM100 / SM103 / SM107), the
Cake backend.

Mode and dispatch rules: every name here carries the mode word ``det``
(FlashInfer #8928: ``cake_gdn_det_kcp``, ``CakeGDNDetKcp<Noun>``,
``cake_gdn_det_train_kcp_<verb>``); none takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  ``torch.use_deterministic_algorithms``
changes nothing here: every launch has a fixed order, so no alert is raised
and the results are bitwise reproducible run to run at the same topology.
``cake_gdn_det_train_kcp_backward`` refuses a saved set or a workspace of the
non-deterministic mode with ``ValueError``.  The non-deterministic mode is a
separate package with its own partition, context and plan rules (the ``b1``
layout differs; the head plan and the ``P`` rule are each package's own);
nothing here routes to it.

Thin entry points (a stable API: no ``ExperimentalWarning``).  The partition / context plan, the host
orchestration, the saved set, the autograd Function and the generated kernels
live in ``flashinfer.experimental.cake.cake_gdn_det_train_kcp``; this module validates
nothing itself and forwards every keyword.  Every switch is a keyword
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

    from flashinfer.cake_gdn_det_train_kcp import (
        cake_gdn_det_kcp,
        cake_gdn_det_train_kcp_backward,
        cake_gdn_det_train_kcp_build_context,
        cake_gdn_det_train_kcp_comm_bytes,
        cake_gdn_det_train_kcp_forward,
        cake_gdn_det_train_kcp_partition,
    )

    part = cake_gdn_det_train_kcp_partition(cu_seqlens_global_cpu, world_size, layout="zigzag_per_document")   # once per batch layout
    ctx = cake_gdn_det_train_kcp_build_context(part, group, rank, device, conv_kernel_size=4)                 # tables of this rank
    out = cake_gdn_det_kcp(q, k, v, g, beta, ctx, save_mode="light", boundary_mode="saved")   # autograd
    out = cake_gdn_det_kcp(q, k, v, g, beta, ctx, boundary_mode="replay")                      # keep nothing; re-run the forward in backward
    out, saved = cake_gdn_det_train_kcp_forward(q, k, v, g, beta, ctx)
    dq, dk, dv, dbeta, dg = cake_gdn_det_train_kcp_backward(q, k, v, g, beta, ctx, dout, saved=saved)
    dq, dk, dv, dbeta, dg = cake_gdn_det_train_kcp_backward(q, k, v, g, beta, ctx, dout)                 # checkpoint replay
    cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=16, HV=48, C=10240)                                       # bytes per collective

``use_qk_l2norm_in_kernel=True`` (keyword-only, default ``False``; the FLA
``chunk_gated_delta_rule`` keyword) normalizes ``q`` and ``k`` inside the
recurrence programs, per token and 128-channel head:
``q_hat = bf16(x * r)``, ``r = 1 / sqrt(sum x^2 + 1e-6)`` with ``x = fp32(q)``,
in a fixed FP32 operation order with FTZ off (one BF16 rounding); ``q`` / ``k``
are the raw rows (for example the convolution's output views).  The gradient is
the exact derivative of that FP32 form at the stored point: ``dq = r (dq_hat -
y (dq_hat . y))`` with ``y = x * r`` recomputed in FP32 from the raw rows and the
saved ``r``, rounded once to BF16.  The saved set grows by ``8 T_r H_qk`` bytes
(the FP32 ``r`` of q and k; 4 MiB at ``T_r = 32768``, ``H_qk = 16``); the
normalized rows (``2 x 256 T_r H_qk`` bytes) are transient workspace in the
forward and the backward and never retained.  Pass the same value to
``cake_gdn_det_train_kcp_backward`` as to ``cake_gdn_det_train_kcp_forward``; every mode (saved / replay,
light / full, checkpoint, recompute handle, CUDA-graph replay, ring) supports it.

The width-4 causal convolution with halo of the same partition is
``flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp(x_local, weight, ctx)``
with this module's ``ctx``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api

# The default scale of the contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


_PACKAGE: Any = None  # the backend package once imported (every facade call reads it)


def _pkg():
    global _PACKAGE
    if _PACKAGE is None:
        from .experimental.cake import cake_gdn_det_train_kcp  # noqa: PLC0415 - the backend package, imported on first use

        _PACKAGE = cake_gdn_det_train_kcp
    return _PACKAGE


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


@flashinfer_api
def cake_gdn_det_train_kcp_partition(
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
    ``CakeGDNDetKcpPartition``: per rank the fragment table ``(doc, off, length, prev, next)`` in document order; reused across
    layers (metadata only).
    """
    return _pkg().cake_gdn_det_train_kcp_partition(
        cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
    )


@flashinfer_api
def cake_gdn_det_train_kcp_build_context(
    part: Any,
    group: Any,
    rank: int,
    device: Any,
    conv_kernel_size: int = 4,
    *,
    sm_count: Optional[int] = None,
) -> Any:
    r"""This rank's view of a partition: local ``cu_seqlens`` (int32 device), the device fragment table, the crossing
    fragment lists (in / out) with their byte counts, the exchange plan, the convolution halo plan (at most
    ``conv_kernel_size - 1`` history rows per continuing fragment) and the segment plan.  ``group`` is the caller's
    ``torch.distributed`` process group (every collective of the route runs on it); CPU metadata plus small int32
    device tables built once; no tensor contents are read.

    Parameters
    ----------
    part
        Partition returned by :func:`cake_gdn_det_train_kcp_partition`.
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
    """
    return _pkg().cake_gdn_det_train_kcp_build_context(
        part, group, rank, device, conv_kernel_size, sm_count=sm_count
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_train_kcp_forward(
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
    use_qk_l2norm_in_kernel: bool = False,
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
    ctx : CakeGDNDetKcpContext
        This rank's context from :func:`cake_gdn_det_train_kcp_build_context`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward : bool
        Keep the saved set (``boundary_mode="saved"``); ``False`` or ``boundary_mode="replay"`` returns ``None``.
    workspace : CakeGDNDetKcpWorkspace, optional
        A reused caller-held workspace (``cake_gdn_det_train_kcp_workspace_bytes``); without one the call allocates its own.
    options
        Keyword-only options (``save_mode`` ``"light"`` | ``"full"``, ``boundary_mode`` ``"saved"`` | ``"replay"``,
        ``exchange`` ``"a2a"`` | ``"ring"``, ``save_transfer``, ``offset_class`` ``"fla"`` | ``"fp32"``, ``layout`` ``"tma"`` |
        ``"copy"``, ``group_reduce`` ``"fused"`` | ``"host"``, ``graph`` ``"off"`` | ``"replay"``,
        ``reuse_recompute_tapes`` ``"auto"`` (default) | ``True`` | ``False`` (``False`` with ``graph="replay"``; see
        :func:`cake_gdn_det_kcp`),
        ``recompute_output`` ``"auto"`` | ``"always"`` (an autograd-node switch; this forward-only entry never skips a
        launch), ...).
    **knobs
        Keyword overrides for the backend options described above.
    use_qk_l2norm_in_kernel : bool
        Normalize ``q`` / ``k`` per token and head inside the programs (module docstring); the saved set then
        carries the FP32 ``r`` of q and k.  With an ``options`` instance it overrides only when ``True``.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (fresh); ``saved`` the saved set or ``None``.
    """
    return _pkg().cake_gdn_det_train_kcp_forward(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=DEFAULT_SCALE if scale is None else scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        options=options,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **knobs,
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_train_kcp_backward(
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
    use_qk_l2norm_in_kernel: bool = False,
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
        exchange are re-run).
    scale, workspace, options, use_qk_l2norm_in_kernel
        As in :func:`cake_gdn_det_train_kcp_forward` (the same options as the forward's; a saved set of the other
        ``use_qk_l2norm_in_kernel`` value is refused).
    **knobs
        Additional backend options, as in :func:`cake_gdn_det_train_kcp_forward`.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v`` (the ``HV / H_qk`` value heads' contributions to
    ``dq`` / ``dk`` reduced in FP32 and rounded once), FP32 gradients of ``beta`` and ``g``; fresh tensors, ranked
    like the inputs.  Bitwise reproducible at the same topology; saved and replay modes agree bitwise.
    """
    return _pkg().cake_gdn_det_train_kcp_backward(
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
        options=options,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **knobs,
    )


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_kcp(
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
    use_qk_l2norm_in_kernel: bool = False,
    **knobs: Any,
) -> torch.Tensor:
    r"""Context-parallel packed GDN training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; every saved tensor travels through
    ``save_for_backward`` only, so a checkpointed forward retains nothing.  See :func:`cake_gdn_det_train_kcp_forward` for the
    operands and options and :func:`cake_gdn_det_train_kcp_backward` for the gradients.

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

    ``recompute_handle`` (keyword-only, default ``None``; a :class:`CakeGDNDetKcpRecomputeHandle`): the checkpoint recompute
    of this forward runs without cross-rank communication.  The original forward exchanges as always and stores
    this rank's incoming states ``S_in`` (the saved set's own tensor) in the handle; torch's recompute of the forward
    inside ``.backward()`` reads them back and runs no exchange, no transfer-summary walk and no merge -- only the
    prep program, at ``P > 1`` the local segment walks and carry merges, and the state walk from ``S_in``, the
    launches of the light backward's replay -- so
    the gradients are bitwise those of the same checkpoint without a handle.  The backward still exchanges.  The
    handle retains ``S_in`` (``cake_gdn_det_train_kcp_saved_bytes(ctx, ..., save_mode="light", save_transfer=False)``) from the
    original forward until the checkpoint frame holding it is freed, instead of retaining nothing (``with_conv=False``
    in that query: the convolution keeps its own rows).  Create one handle per layer and micro-step, outside the
    outermost checkpoint region, and pass it as an argument of the checkpointed function::

        def layer(x, handle):
            ...
            return cake_gdn_det_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, CakeGDNDetKcpRecomputeHandle(), use_reentrant=False)

    A handle holds one entry per key: it serves this call and at most one context-parallel convolution call (whose
    halo rows go under the key ``"conv"``); a layer with separate Q / K / V convolutions uses one handle per
    convolution call, and this call shares one of them.  Needs ``boundary_mode="saved"`` and ``graph="off"``; implies ``save_transfer=False``.  A forward
    outside the backward that finds an entry raises (a handle reused across micro-steps).

    ``use_qk_l2norm_in_kernel`` (keyword-only, default ``False``): the in-kernel per-head L2 norm of ``q`` / ``k``
    (module docstring); it is part of the saved set and of the recompute handle's entry, so a checkpoint, a handle
    and a CUDA-graph replay see it like any other option.

    ``q`` / ``k`` / ``v`` that are head views of one packed ``[..., T, (2 H_qk + HV) * 128]`` row (the convolution's
    output) get the full backward speed only as :func:`cake_gdn_det_train_kcp_packed_qkv_views` views.  Plain ``split`` / ``view``
    views of the row give bitwise the same results, but torch's split backward concatenates ``dq`` / ``dk`` /
    ``dv`` into a new row: about 1.1 ms (33-36 %) more backward per call at ``(16, 48)`` and 32,768 tokens on GB200
    and GB300, with the in-kernel norm off or on (package README, "Packed Q/K/V").
    """
    return _pkg().cake_gdn_det_kcp(
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
        options=options,
        recompute_handle=recompute_handle,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **knobs,
    )


@flashinfer_api
def cake_gdn_det_train_kcp_packed_qkv_views(
    qkv: torch.Tensor, *, H_qk: int, HV: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``q`` / ``k`` / ``v`` as head views ``[..., T, H_qk | H_qk | HV, 128]`` of a packed
    ``[..., T, (2 H_qk + HV) * 128]`` row, with no copy.  :func:`cake_gdn_det_kcp` takes the views as they are and its
    backward writes ``dq`` / ``dk`` / ``dv`` into one packed row, which this split's backward returns as the gradient
    of ``qkv`` without a concatenation.  Plain ``split`` views of ``qkv`` give the same gradients through torch's
    split backward, which concatenates them (:func:`cake_gdn_det_kcp`).

    Needs a dense channel axis (``qkv.stride(-1) == 1``).  The views are the outputs of one custom autograd node:
    hooks and ``retain_grad`` on them fire as on any autograd output; its backward concatenates instead when the
    incoming gradients are not the head views of one packed row (a missing gradient counts as zeros, a hook that
    returns a new gradient); autograd refuses an in-place change to the views while ``qkv`` requires grad.

    Example::

        q, k, v = cake_gdn_det_train_kcp_packed_qkv_views(conv_out, H_qk=16, HV=48)        # conv_out: [T, 10240]
        out = cake_gdn_det_kcp(q, k, v, g, beta, ctx)

    Parameters
    ----------
    qkv : torch.Tensor
        Packed projection rows ``[..., T, (2 * H_qk + HV) * 128]`` with unit channel stride.
    H_qk : int
        Number of query and key heads, each with 128 channels.
    HV : int
        Number of value heads, each with 128 channels.
    """
    return _pkg().cake_gdn_det_train_kcp_packed_qkv_views(qkv, H_qk=H_qk, HV=HV)


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_train_kcp_workspace_bytes(
    ctx: Any, *, H_qk: int, HV: int, **kw: Any
) -> int:
    r"""Caller-held workspace bytes of one rank for ``(ctx, H_qk, HV)`` (exchange pools, staging, tapes): the most
    the workspace binds in the forward or the backward of the given options (``reuse_recompute_tapes`` as a keyword;
    the per-phase terms are ``cake_gdn_det_train_kcp_workspace_layout`` of the package).  ``use_qk_l2norm_in_kernel=True`` adds the
    transient normalized rows ``qn`` / ``kn`` (``2 x 256 T_r H_qk`` bytes) to the forward and every backward phase,
    and the ``8 T_r H_qk``-byte FP32 ``r`` where the backward re-runs the prep program.

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_det_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    **kw
        Backend options for this query, including the call's save and boundary settings.
    """
    return _pkg().cake_gdn_det_train_kcp_workspace_bytes(ctx, H_qk=H_qk, HV=HV, **kw)


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_train_kcp_saved_bytes(
    ctx: Any, *, H_qk: int, HV: int, **kw: Any
) -> int:
    r"""Saved-set bytes of one layer on this rank (``save_mode``, ``save_transfer`` as keywords), counted per
    crossing fragment; ``use_qk_l2norm_in_kernel=True`` adds the FP32 ``r`` of q and k, ``8 T_r H_qk`` bytes
    (``boundary_mode="saved"``).

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_det_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    **kw
        Backend options for this query, including the call's save and boundary settings.
    """
    return _pkg().cake_gdn_det_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV, **kw)


@flashinfer_api
@_forwards_keywords
def cake_gdn_det_train_kcp_comm_bytes(
    ctx: Any, *, H_qk: int, HV: int, C: int = 10240, **kw: Any
) -> dict:
    r"""Bytes this rank moves per collective (``c1``, ``c2``, ``b1``, ``b2``, ``halo_fwd``, ``halo_bwd``) and per
    boundary, counted per crossing fragment; ``C`` is the convolution channel count of the halo rows.

    Parameters
    ----------
    ctx
        This rank's context from :func:`cake_gdn_det_train_kcp_build_context`.
    H_qk : int
        Number of query and key heads.
    HV : int
        Number of value heads.
    C : int, default=10240
        Convolution channel count used to size halo traffic.
    **kw
        Backend options for this query: ``save_transfer`` and ``boundary_mode``.
    """
    return _pkg().cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C, **kw)


@flashinfer_api
def is_cake_gdn_det_train_kcp_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    r"""``True`` when ``device`` (default: the current CUDA device) is an exported target of the package.

    Parameters
    ----------
    device : torch.device, str, int, optional
        Device to query; ``None`` selects the current CUDA device.
    """
    if not torch.cuda.is_available():
        return False
    try:
        pkg = _pkg()
        from .experimental.cake.cake_gdn_det_train_kcp import cake_jit  # noqa: PLC0415

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
        from .experimental.cake.cake_gdn_det_train_kcp import cake_backend  # noqa: PLC0415

        return cake_backend.device_arch(index)
    except ImportError:
        pass
    by_cap = getattr(cake_jit, "ARCH_BY_CAPABILITY", None) or {}
    return by_cap.get(tuple(torch.cuda.get_device_capability(index)))


# The public types of the package, resolved on first access so that importing the facade imports the package
# lazily: the options, partition, context and recompute-handle types the caller builds, the saved set and workspace
# types the forward returns and the backward reads, and the autograd Function of ``cake_gdn_det_kcp``.
_LAZY_TYPES = (
    "CakeGDNDetKcpContext",
    "CakeGDNDetKcpFunction",
    "CakeGDNDetKcpOptions",
    "CakeGDNDetKcpPartition",
    "CakeGDNDetKcpRecomputeHandle",
    "CakeGDNDetKcpSaved",
    "CakeGDNDetKcpWorkspace",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TYPES:
        return getattr(_pkg(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "CakeGDNDetKcpContext",  # noqa: F822 - resolved by __getattr__
    "CakeGDNDetKcpFunction",  # noqa: F822
    "CakeGDNDetKcpOptions",  # noqa: F822
    "CakeGDNDetKcpPartition",  # noqa: F822
    "CakeGDNDetKcpRecomputeHandle",  # noqa: F822
    "CakeGDNDetKcpSaved",  # noqa: F822
    "CakeGDNDetKcpWorkspace",  # noqa: F822
    "cake_gdn_det_kcp",
    "cake_gdn_det_train_kcp_backward",
    "cake_gdn_det_train_kcp_build_context",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_forward",
    "cake_gdn_det_train_kcp_packed_qkv_views",
    "cake_gdn_det_train_kcp_partition",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_workspace_bytes",
    "is_cake_gdn_det_train_kcp_supported",
]
