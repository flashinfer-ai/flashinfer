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

Context-parallel (KCP) packed KDA (Kimi Delta Attention) chunked training
(SM100 / SM103 / SM107), the Cake backend, with a choice of numerical mode.

Every entry takes ``deterministic: bool = True`` and rejects a value that is not
a ``bool`` (``TypeError``):

* ``deterministic=True`` (the default) runs the deterministic package
  ``flashinfer.experimental.cake_kda_det_train_kcp`` (the objects behind
  ``flashinfer.cake_kda_det_train_kcp``): the same inputs, hardware, plan and
  process group give bitwise the same outputs and gradients.
* ``deterministic=False`` runs the package
  ``flashinfer.experimental.cake_kda_train_kcp``.  Its KCP programs and
  exchanges are the deterministic package's; its rank-local chunk stages are
  those of ``flashinfer.experimental.cake.cake_kda_train``
  (``flashinfer.cake_kda_train`` with ``deterministic=False``), whose backward
  moves the ``q * dq`` term of the gate gradient ``dg`` from the fused
  dq / dk / dg program into the GLM intra program on the rows its registry
  lists.  The schedule has a fixed order: at the same topology repeated calls
  give bitwise the same results, ``out``, ``dq``, ``dk``, ``dv`` and ``dbeta``
  are bitwise the deterministic package's, and ``dg`` differs from it by FP32
  rounding only (the same accuracy budget).
* With ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  runs the deterministic package and warns once per process.

The mode is never stored in the context: a partition, a context and the byte
queries of either package describe the same plan, a partition of either
package builds the context of both (``build_context_kcp`` converts it to the
type of the package it routes to), and a context of either package serves the
calls of both (``partition_kcp`` / ``build_context_kcp`` build it with the
package ``deterministic`` selects).  The mode of
:func:`chunk_kda_kcp` is fixed at its forward call: the autograd node's backward
is the forward's package.  A saved set is read by the backward of the package
that wrote it (the other one raises ``ValueError``), and a workspace serves its
own package: allocate it with the ``deterministic`` value of the calls that use
it.  When torch's deterministic algorithms turn a ``deterministic=False`` call
with a non-deterministic workspace deterministic, the call runs on a workspace
of its own.

Thin experimental entry points: the partition / context plan, the launch
sequences, the saved set, the autograd Functions and the generated kernels live
in the two packages, which every function imports inside its body (importing
this module loads neither); this module forwards every keyword.  Every switch
is a keyword argument; no environment variable is read.

Name map (the deterministic names are unchanged and always deterministic)::

    flashinfer.cake_kda_det_train_kcp      this module (deterministic=...)
      partition_det_kcp                      partition_kcp
      build_context_det_kcp                  build_context_kcp
      chunk_kda_det_fwd_kcp                  chunk_kda_fwd_kcp
      chunk_kda_det_bwd_kcp                  chunk_kda_bwd_kcp
      chunk_kda_det_kcp                      chunk_kda_kcp
      kda_det_workspace_bytes_kcp            kda_workspace_bytes_kcp
      kda_det_saved_bytes_kcp                kda_saved_bytes_kcp
      kda_det_comm_bytes_kcp                 kda_comm_bytes_kcp
      ContextDetKcp, PartitionDetKcp, ...    ContextKcp, PartitionKcp, ... (the non-deterministic package's types)

``flashinfer.kda_train.chunk_kda_kcp`` stays the deterministic entry (the name
it had before the deterministic rename), without a ``deterministic`` keyword;
this module is the explicit namespace of the selectable mode.

Usage::

    from flashinfer.cake_kda_train_kcp import (partition_kcp, build_context_kcp, chunk_kda_kcp,
                                               chunk_kda_fwd_kcp, chunk_kda_bwd_kcp)

    part = partition_kcp(cu_seqlens_global_cpu, world_size, deterministic=False)       # once per batch layout
    ctx = build_context_kcp(part, group, rank, device, deterministic=False)            # this rank's tables
    out_local = chunk_kda_kcp(q_local, k_local, v_local, g_local, beta_local, ctx, deterministic=False)
    out_local, saved = chunk_kda_fwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx, deterministic=False)
    grads = chunk_kda_bwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx, dout_local, saved=saved,
                              deterministic=False)

The width-4 causal convolution with halo of the same partition takes this module's ``ctx``.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api
from .cake_kda_train import _FEATURE, DEFAULT_SCALE, _deterministic

# The module of the non-deterministic package's workspace and saved-set types (recognized without importing the
# package: none exists before the package is imported).
_ND_KCP = f"{__package__}.experimental.cake_kda_train_kcp.cake_kcp"
# The plan modules of the two packages, whose partition types are recognized the same way.
_ND_PLAN = f"{__package__}.experimental.cake_kda_train_kcp.cake_kcp_plan"
_DET_PLAN = f"{__package__}.experimental.cake_kda_det_train_kcp.cake_kcp_plan"


def _det_kcp() -> Any:
    """The deterministic package's host module, imported on first use."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp  # noqa: PLC0415

    return cake_kcp


def _det_plan() -> Any:
    """The deterministic package's plan module, imported on first use."""
    from .experimental.cake_kda_det_train_kcp import cake_kcp_plan  # noqa: PLC0415

    return cake_kcp_plan


def _nd_kcp() -> Any:
    """The non-deterministic package's host module, imported on first use."""
    from .experimental.cake_kda_train_kcp import cake_kcp  # noqa: PLC0415

    return cake_kcp


def _nd_plan() -> Any:
    """The non-deterministic package's plan module, imported on first use."""
    from .experimental.cake_kda_train_kcp import cake_kcp_plan  # noqa: PLC0415

    return cake_kcp_plan


def _det_workspace(workspace: Any, deterministic: bool, entry: str) -> Any:
    """``workspace`` of a call routed to the deterministic package.  A non-deterministic workspace launches that
    package's local records: it is refused under ``deterministic=True``, and under torch's deterministic algorithms
    the call runs on a workspace of its own (``None``)."""
    nd = sys.modules.get(_ND_KCP)
    if workspace is None or nd is None or not isinstance(workspace, nd.WorkspaceKcp):
        return workspace
    if deterministic:
        raise TypeError(
            f"{entry}: the workspace is a WorkspaceKcp of deterministic=False; a deterministic=True call needs a "
            "WorkspaceDetKcp (flashinfer.cake_kda_det_train_kcp)"
        )
    return None


def _partition_for(det: bool, part: Any) -> Any:
    """``part`` as the routed package's partition: a partition of the other package (the same frozen fields)
    becomes the same partition of this one, every field kept (``layout`` included) and its fragments rebuilt as this
    package's fragments; the routed package's own partition and any other object pass through (the plan refuses the
    latter).  The other package's type is recognized through ``sys.modules`` (none exists before that package is
    imported), so neither route imports the other package."""
    other = sys.modules.get(_ND_PLAN if det else _DET_PLAN)
    if other is None:
        return part
    if not isinstance(part, other.PartitionKcp if det else other.PartitionDetKcp):
        return part
    plan = _det_plan() if det else _nd_plan()
    own = plan.PartitionDetKcp if det else plan.PartitionKcp
    fragment = plan.FragmentDetKcp if det else plan.FragmentKcp

    def fields(obj: Any) -> dict[str, Any]:
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}

    rebuilt = fields(part)
    rebuilt["fragments"] = tuple(
        tuple(fragment(**fields(fr)) for fr in frs) for frs in part.fragments
    )
    return own(**rebuilt)


def _det_saved(saved: Any, entry: str) -> Any:
    """``saved`` of a backward routed to the deterministic package: a saved set of the non-deterministic package is
    refused (``ValueError``) before it reaches the deterministic backward."""
    nd = sys.modules.get(_ND_KCP)
    if saved is not None and nd is not None and isinstance(saved, nd.SavedKcp):
        raise ValueError(
            f"{entry}: the saved set comes from the non-deterministic package (deterministic=False); the "
            "deterministic backward reads its own forward's saved set only"
        )
    return saved


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
    deterministic : bool
        The package that builds the partition (see the module docstring); both build the same partition.

    Returns
    -------
    ``PartitionKcp`` (``PartitionDetKcp`` with ``deterministic=True``): per rank its fragments (document, token range)
    of the batch; the input of :func:`build_context_kcp` on every rank.
    """
    if _deterministic(deterministic, "partition_kcp"):
        return _det_plan().partition_det_kcp(
            cu_seqlens_global_cpu, world_size, layout, fragments, cut_points=cut_points
        )
    return _nd_plan().partition_kcp(
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
    heads: Optional[Any] = None,
    stream: Optional[Any] = None,
    deterministic: bool = True,
) -> Any:
    r"""This rank's context of a partition: its device tables, the collectives of the state and halo exchanges and
    the segment plan of its fragments.

    Parameters
    ----------
    part : PartitionKcp
        From :func:`partition_kcp`, the same object on every rank.  A partition of either package serves both: one of
        the other package than the call runs (another ``deterministic``, or torch's deterministic algorithms switched
        on between the two calls) is converted to the routed package's type with the same fields.
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
        The stream of the context's one asynchronous table copy (default: the current stream of ``device``); see
        :func:`flashinfer.cake_kda_det_train_kcp.build_context_det_kcp`.
    deterministic : bool
        The package that builds the context (see the module docstring).  The context holds no mode: a context of
        either package serves the calls of both.

    Returns
    -------
    ``ContextKcp`` (``ContextDetKcp`` with ``deterministic=True``): the argument every KCP entry point takes;
    reusable across calls of the same batch layout.
    """
    det = _deterministic(deterministic, "build_context_kcp")
    part = _partition_for(det, part)
    if det:
        return _det_plan().build_context_det_kcp(
            part,
            group,
            rank,
            device,
            conv_kernel_size,
            sm_count=sm_count,
            heads=heads,
            stream=stream,
        )
    return _nd_plan().build_context_kcp(
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
def chunk_kda_fwd_kcp(
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
    deterministic: bool = True,
    **options: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    r"""Context-parallel packed KDA chunked training forward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta : torch.Tensor
        This rank's shard of the operands of :func:`~flashinfer.cake_kda_train.chunk_kda_fwd` (the tokens of its
        fragments, in the partition's order): BF16 ``[1, S_local, H, 128]`` ``q, k, v``, FP32 ``g``, BF16
        ``[1, S_local, H]`` ``beta``.
    ctx : ContextKcp
        From :func:`build_context_kcp`.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    save_for_backward, save_mode, boundary_mode
        As for :func:`flashinfer.cake_kda_det_train_kcp.chunk_kda_det_fwd_kcp`.
    workspace : optional
        A reused workspace of this rank of the same ``deterministic`` (``WorkspaceKcp`` / ``WorkspaceDetKcp``);
        without one the call allocates its own.
    deterministic : bool
        ``True`` runs the deterministic package, ``False`` the non-deterministic one (see the module docstring).
        The forward programs of the two are the same: bitwise the same ``out``.
    **options
        Further KCP options (``save_transfer``, ``exchange``, ``head_groups``, ``offset_class``, ``graph``),
        forwarded.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v`` (this rank's tokens); ``saved`` the package's saved set or ``None``
    (readable by the same package's backward only).
    """
    if _deterministic(deterministic, "chunk_kda_fwd_kcp"):
        return _det_kcp().chunk_kda_det_fwd_kcp(
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
            workspace=_det_workspace(workspace, deterministic, "chunk_kda_fwd_kcp"),
            **options,
        )
    return _nd_kcp().chunk_kda_fwd_kcp(
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
        deterministic=False,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def chunk_kda_bwd_kcp(
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
    deterministic: bool = True,
    **options: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Context-parallel packed KDA chunked training backward of this rank's fragments.

    Parameters
    ----------
    q, k, v, g, beta, ctx
        The forward's inputs (same contracts as :func:`chunk_kda_fwd_kcp`).
    dout : torch.Tensor
        BF16 gradient of this rank's ``out``.
    saved : optional
        The forward's saved set, written by the package ``deterministic`` selects (the other package's set raises
        ``ValueError``); without one the forward is replayed (checkpointing).
    scale, save_mode, workspace, **options
        As for :func:`chunk_kda_fwd_kcp`; the backward reads its own options (``spec3_probe`` among them).
    deterministic : bool
        ``True`` runs the deterministic package, ``False`` the non-deterministic one (see the module docstring); the
        forward's value.

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)`` of this rank's tokens: BF16 ``dq, dk, dv, dbeta``, FP32 ``dg``.  Bitwise
    reproducible at the same topology in both modes.
    """
    if _deterministic(deterministic, "chunk_kda_bwd_kcp"):
        return _det_kcp().chunk_kda_det_bwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=_det_saved(saved, "chunk_kda_bwd_kcp"),
            scale=scale,
            save_mode=save_mode,
            workspace=_det_workspace(workspace, deterministic, "chunk_kda_bwd_kcp"),
            **options,
        )
    return _nd_kcp().chunk_kda_bwd_kcp(
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
        deterministic=False,
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
    deterministic: bool = True,
    **options: Any,
) -> torch.Tensor:
    r"""Autograd-enabled context-parallel packed KDA chunked training step of this rank's fragments (the forward of
    :func:`chunk_kda_fwd_kcp`; the backward of :func:`chunk_kda_bwd_kcp` from its saved set).

    ``recompute_handle`` / ``checkpoint_keep``: the checkpoint recompute without the forward state exchange, as for
    :func:`flashinfer.cake_kda_det_train_kcp.chunk_kda_det_kcp` (a ``RecomputeHandleKcp`` or any object with an
    ``entries`` dict; every rank passes a handle for the same call or none).  ``deterministic`` selects the package at
    the forward call; the node's backward runs the same package, and the checkpoint recompute of a forward runs the
    forward's package as long as torch's deterministic-algorithms setting does not change in between.

    Returns
    -------
    torch.Tensor
        BF16 ``out`` like ``v`` (this rank's tokens).
    """
    if _deterministic(deterministic, "chunk_kda_kcp"):
        return _det_kcp().chunk_kda_det_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_mode=save_mode,
            boundary_mode=boundary_mode,
            workspace=_det_workspace(workspace, deterministic, "chunk_kda_kcp"),
            recompute_handle=recompute_handle,
            checkpoint_keep=checkpoint_keep,
            **options,
        )
    return _nd_kcp().chunk_kda_kcp(
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
        deterministic=False,
        **options,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kda_workspace_bytes_kcp(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    backward: bool = True,
    save_transfer: Optional[bool] = None,
    C: int = 8192,
    local_arena: Optional[str] = None,
    lean_passes: Optional[int] = None,
    deterministic: bool = True,
) -> int:
    """Bytes of this rank's KCP workspace for ``H`` heads on the package ``deterministic`` selects (the same for both
    values); see :func:`flashinfer.cake_kda_det_train_kcp.kda_det_workspace_bytes_kcp`.  ``ctx`` may be a list of
    contexts (one workspace that serves all of them).  ``local_arena`` / ``lean_passes`` are those of the workspace
    (``WorkspaceKcp`` / ``WorkspaceDetKcp`` ``(local_arena="lean", lean_passes=...)``, opt-in: the backward's lean
    local arena, its tail in ``lean_passes`` passes, two by default); ``None`` keeps the replay arena."""
    det = _deterministic(deterministic, "kda_workspace_bytes_kcp")
    plan = _det_plan() if det else _nd_plan()
    if save_transfer is None:
        save_transfer = plan.SAVE_TRANSFER_DEFAULT
    query = plan.kda_det_workspace_bytes_kcp if det else plan.kda_workspace_bytes_kcp
    return query(
        ctx,
        H=H,
        save_mode=save_mode,
        backward=backward,
        save_transfer=save_transfer,
        C=C,
        local_arena=local_arena,
        lean_passes=lean_passes,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kda_saved_bytes_kcp(
    ctx: Any,
    *,
    H: int = 64,
    save_mode: str = "light",
    save_transfer: Optional[bool] = None,
    with_conv: bool = True,
    C: int = 8192,
    checkpoint_keep: Optional[str] = None,
    deterministic: bool = True,
) -> int:
    """Bytes of this rank's KCP saved set for ``H`` heads on the package ``deterministic`` selects (the same for both
    values); see :func:`flashinfer.cake_kda_det_train_kcp.kda_det_saved_bytes_kcp` for ``save_transfer``,
    ``with_conv``, ``C`` and ``checkpoint_keep``."""
    det = _deterministic(deterministic, "kda_saved_bytes_kcp")
    plan = _det_plan() if det else _nd_plan()
    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    query = plan.kda_det_saved_bytes_kcp if det else plan.kda_saved_bytes_kcp
    return query(
        ctx,
        H=H,
        save_mode=save_mode,
        save_transfer=save_transfer,
        with_conv=with_conv,
        C=C,
        checkpoint_keep=checkpoint_keep,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kda_comm_bytes_kcp(
    ctx: Any,
    *,
    H: int = 64,
    C: int = 8192,
    save_transfer: Optional[bool] = None,
    boundary_mode: str = "saved",
    checkpoint_keep: Optional[str] = None,
    deterministic: bool = True,
) -> dict:
    """Bytes this rank sends and receives in one training step for ``H`` heads and ``C`` convolution channels, per
    collective and per boundary, on the package ``deterministic`` selects (the same for both values); see
    :func:`flashinfer.cake_kda_det_train_kcp.kda_det_comm_bytes_kcp`."""
    det = _deterministic(deterministic, "kda_comm_bytes_kcp")
    plan = _det_plan() if det else _nd_plan()
    if save_transfer is None and checkpoint_keep is None:
        save_transfer = False
    query = plan.kda_det_comm_bytes_kcp if det else plan.kda_comm_bytes_kcp
    return query(
        ctx,
        H=H,
        C=C,
        save_transfer=save_transfer,
        boundary_mode=boundary_mode,
        checkpoint_keep=checkpoint_keep,
    )


# The types of the non-deterministic package, resolved on first access so that importing this module imports
# neither the KCP host module (torch.distributed, the KCP loader) nor the plan: public name -> the submodule that
# defines it.  The deterministic package's types are those of ``flashinfer.cake_kda_det_train_kcp``.
_LAZY_TYPES = {
    "ContextKcp": "cake_kcp_plan",
    "OptionsKcp": "cake_kcp",
    "PartitionKcp": "cake_kcp_plan",
    "RecomputeHandleKcp": "cake_kcp",
    "SavedKcp": "cake_kcp",
    "WorkspaceKcp": "cake_kcp",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_TYPES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415

    return getattr(
        import_module(f"flashinfer.experimental.cake_kda_train_kcp.{submodule}"),
        name,
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_TYPES))


__all__ = [
    "DEFAULT_SCALE",
    "ContextKcp",  # noqa: F822 - resolved by __getattr__
    "OptionsKcp",  # noqa: F822
    "PartitionKcp",  # noqa: F822
    "RecomputeHandleKcp",  # noqa: F822
    "SavedKcp",  # noqa: F822
    "WorkspaceKcp",  # noqa: F822
    "build_context_kcp",
    "chunk_kda_bwd_kcp",
    "chunk_kda_fwd_kcp",
    "chunk_kda_kcp",
    "kda_comm_bytes_kcp",
    "kda_saved_bytes_kcp",
    "kda_workspace_bytes_kcp",
    "partition_kcp",
]
