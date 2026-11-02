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
(SM100 / SM103 / SM107), the Cake backend, with a choice of numerical mode.

Every entry takes ``deterministic: bool = True`` and rejects a value that is not
a ``bool`` (``TypeError``):

* ``deterministic=True`` (the default) runs the deterministic package
  ``flashinfer.experimental.cake_kda_det_train`` (the objects behind
  ``flashinfer.cake_kda_det_train``): the same inputs, hardware and plan give
  bitwise the same outputs and gradients.
* ``deterministic=False`` runs the package
  ``flashinfer.experimental.cake.cake_kda_train``.  Its backward moves the
  ``q * dq`` term of the gate gradient ``dg`` from the fused dq / dk / dg
  program into the GLM intra program, so the fused program no longer reads
  ``q``.  The schedule has a fixed order: repeated calls give bitwise the same
  results, ``out``, ``dq``, ``dk``, ``dv`` and ``dbeta`` are bitwise the
  deterministic package's, and ``dg`` differs from it by FP32 rounding only
  (the same accuracy budget).  On some rows the checkpoint replay
  (``saved=None``) and a saved-set backward take different program pairs, and
  their ``dg`` then differ by FP32 rounding too.
* With ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  runs the deterministic package and warns once per process.

Raw gate mode (``use_gate_in_kernel=True`` with ``A_log``, ``dt_bias`` and
``lower_bound``; FlashInfer #7816) runs on the deterministic package only: the
entries forward the keywords to it, ``chunk_kda_bwd`` then returns seven tensors
(``dA_log`` / ``ddt_bias`` added), and the workspace and its byte query take
``use_gate_in_kernel``.  ``deterministic=False`` raises ``ValueError`` for it
(also for a saved set of that mode); the off values (``use_gate_in_kernel=False``,
``A_log`` / ``dt_bias`` / ``lower_bound`` ``None``) are accepted in both modes.

The mode of :func:`chunk_kda` is fixed at its forward call: the autograd node's
backward is the forward's package.  Both packages write the saved set in the
same format, so either backward reads it.  A workspace serves its own package:
allocate it with the ``deterministic`` value of the calls that use it (the
non-deterministic package refuses another package's workspace, and a
``deterministic=True`` call refuses a ``deterministic=False`` workspace).  When
torch's deterministic algorithms turn a ``deterministic=False`` call with a
``deterministic=False`` workspace deterministic, the call runs on a workspace of
its own with the same arena option.  An arena option
(:func:`kda_train_arena`) of either package serves both.

Thin experimental entry points: the contract, the host plan, the workspace,
the launch sequences, the autograd Functions and the generated kernels live in
the two packages, which every function imports inside its body (importing this
module loads neither); this module forwards every keyword.  Every switch is a
keyword argument; this module reads no environment variable (the
non-deterministic package's ``chunk_kda`` reads ``CAKE_KDA_CKPT_HANDOFF``, the
benchmarking override of ``checkpoint_handoff``).

Operands (one packed batch of documents): ``q, k, v`` BF16 ``[1, S, H, 128]``
(``q`` / ``k`` L2-normalized by the caller), ``g`` FP32 ``[1, S, H, 128]`` (the
natural-log decay gate, preactivated), ``beta`` BF16 ``[1, S, H]`` (through
the sigmoid), ``cu_seqlens`` int32 ``[num_docs + 1]`` on the device.  The
batch axis of 1 is optional; outputs take the rank of the inputs.  Gradients:
``dq, dk, dv, dbeta`` BF16, ``dg`` FP32.  Per-document zero initial state, no
final state, ``scale = 128 ** -0.5`` by default, ``H`` in ``1..64``.

Name map (the deterministic names are unchanged and always deterministic)::

    flashinfer.cake_kda_det_train      this module (deterministic=...)
      chunk_kda_det_fwd                  chunk_kda_fwd
      chunk_kda_det_bwd                  chunk_kda_bwd
      chunk_kda_det                      chunk_kda
      kda_det_train_arena                kda_train_arena
      kda_det_train_workspace            kda_train_workspace
      kda_det_train_workspace_bytes      kda_train_workspace_bytes
      kda_det_train_saved_bytes          kda_train_saved_bytes
      is_kda_det_train_supported         is_kda_train_supported

``flashinfer.kda_train`` binds the same spellings (``chunk_kda``,
``chunk_kda_fwd``, ...) to the deterministic objects, without a
``deterministic`` keyword; this module is the explicit namespace of the
selectable mode.

Usage::

    from flashinfer.cake_kda_train import chunk_kda, chunk_kda_fwd, chunk_kda_bwd, kda_train_workspace

    out = chunk_kda(q, k, v, g, beta, cu_seqlens)                        # deterministic (default)
    out = chunk_kda(q, k, v, g, beta, cu_seqlens, deterministic=False)   # the dg term in the GLM intra program
    ws = kda_train_workspace(S, num_docs, H, device, deterministic=False)
    out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws, deterministic=False)
    dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout, saved=saved, workspace=ws,
                                          deterministic=False)
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
import warnings
from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake packed KDA chunked training, selectable mode (SM100/SM103/SM107)"

# The default scale of the GLM contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5

# The module of the non-deterministic package's workspace type (a workspace of it is recognized without importing
# the package: none exists before the package is imported).
_ND_BACKEND = f"{__package__}.experimental.cake.cake_kda_train.cake_backend"

# The flashinfer package directory: frames below it are this module, its decorators and the packages.
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# One warning per process when torch's deterministic algorithms turn a ``deterministic=False`` call deterministic.
_DET_FALLBACK_WARNED = False


def _det() -> Any:
    """The deterministic package's backend, imported on first use."""
    from .experimental.cake_kda_det_train import cake_backend

    return cake_backend


def _nd() -> Any:
    """The non-deterministic package's backend, imported on first use."""
    from .experimental.cake.cake_kda_train import cake_backend

    return cake_backend


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside the flashinfer package, for a warning issued by the function
    that calls this one (the decorators between the caller and the entry vary)."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None and not os.path.abspath(
            parent.f_code.co_filename
        ).startswith(_PACKAGE_ROOT + os.sep):
            return level
        frame = parent
    return 3


def _deterministic(deterministic: bool, entry: str) -> bool:
    """``True`` when ``entry`` runs the deterministic package: ``deterministic=True``, or ``deterministic=False`` while
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
                f"{entry}: deterministic=False runs the deterministic package because torch's deterministic "
                "algorithms are enabled (torch.use_deterministic_algorithms(True)); this warning is shown once",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        return True
    return False


def _gate_mode(use_gate_in_kernel: bool) -> str:
    """The packages' gate mode of ``use_gate_in_kernel``: ``"raw"`` for ``True``, ``"g"`` for ``False``
    (``ValueError`` for a non-bool, as the deterministic facade)."""
    if not isinstance(use_gate_in_kernel, bool):
        raise ValueError(
            f"use_gate_in_kernel must be a bool, got {use_gate_in_kernel!r}"
        )
    return "raw" if use_gate_in_kernel else "g"


def _arena_for(det: bool, arena: Any) -> Any:
    """``arena`` as the routed package's arena type: an arena option of the other package becomes the same option of
    this one (the two types have the same fields); ``None`` and the routed package's own pass through."""
    if arena is None:
        return None
    own = _det().KdaDetArena if det else _nd().KdaArena
    if isinstance(arena, own) or not dataclasses.is_dataclass(arena):
        return arena
    return own(**{f.name: getattr(arena, f.name) for f in dataclasses.fields(arena)})


def _det_call(
    workspace: Any,
    arena: Any,
    save_mode: Optional[str],
    deterministic: bool,
    entry: str,
) -> tuple[Any, Any, Optional[str]]:
    """``(workspace, arena, save_mode)`` of a call routed to the deterministic package.  A non-deterministic
    workspace launches that package's programs: it is refused under ``deterministic=True``, and under torch's
    deterministic algorithms the call runs on a workspace of its own with that workspace's arena option and default
    ``save_mode`` (where the call gives none)."""
    nd = sys.modules.get(_ND_BACKEND)
    if workspace is None or nd is None or not isinstance(workspace, nd.KdaWorkspace):
        return workspace, _arena_for(True, arena), save_mode
    if deterministic:
        raise TypeError(
            f"{entry}: the workspace was allocated with deterministic=False; a deterministic=True call needs a "
            "workspace of kda_train_workspace(..., deterministic=True)"
        )
    return (
        None,
        _arena_for(True, workspace.arena_option if arena is None else arena),
        workspace.save_mode if save_mode is None else save_mode,
    )


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
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    deterministic: bool = True,
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
    workspace : optional
        A reused workspace from :func:`kda_train_workspace` of the same ``deterministic``; without one this call
        allocates its own.
    save_mode : str, optional
        ``"light"`` (saved set version 4) or ``"tape"`` (version 2); ``None`` takes the workspace's default
        (``"light"`` without one).  See :func:`flashinfer.cake_kda_det_train.chunk_kda_det_fwd`.
    arena : optional
        The lower-memory arena option (:func:`kda_train_arena`; off by default); ``None`` takes the workspace's
        option, a differing option on a given workspace is refused.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers), required under a batched option.
    use_gate_in_kernel, A_log, dt_bias, lower_bound
        The raw gate mode (``g`` the raw BF16 gate, activated in the kernels); see
        :func:`flashinfer.cake_kda_det_train.chunk_kda_det_fwd`.  ``deterministic=True`` only: ``deterministic=False``
        raises ``ValueError`` unless they are off (``False`` / ``None``).
    deterministic : bool
        ``True`` runs the deterministic package, ``False`` the non-deterministic one (see the module docstring).
        The forward programs of the two are the same: bitwise the same ``out`` and saved set.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v``; ``saved`` the package's saved set (versioned) or ``None``.
    """
    scale = DEFAULT_SCALE if scale is None else scale
    if _deterministic(deterministic, "chunk_kda_fwd"):
        workspace, arena, save_mode = _det_call(
            workspace, arena, save_mode, deterministic, "chunk_kda_fwd"
        )
        return _det().chunk_kda_det_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
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
    return _nd().chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        save_mode=save_mode,
        arena=_arena_for(False, arena),
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        deterministic=False,
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
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Packed KDA chunked training backward.

    Parameters
    ----------
    q, k, v, g, beta, cu_seqlens
        The forward's operands.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : optional
        The forward's saved set (version 4 or 2, of either package: the format is the same); ``None`` recomputes it
        from the inputs (checkpoint replay).
    scale : float, optional
        The forward's scale; ``128 ** -0.5`` when ``None``.
    workspace : optional
        A reused workspace with backward buffers, of the same ``deterministic``; without one this call allocates
        its own.
    arena : optional
        The lower-memory arena option of that per-call workspace (:func:`kda_train_arena`); ``None`` takes a given
        workspace's option, a differing one is refused.  A lean arena serves no checkpoint replay.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers), required under a batched option.
    use_gate_in_kernel, A_log, dt_bias, lower_bound
        The forward's raw gate mode (see :func:`chunk_kda_fwd`); ``deterministic=True`` only.  With
        ``deterministic=False`` they must be off, and a saved set of the raw gate mode is refused (``ValueError``).
    deterministic : bool
        ``True`` runs the deterministic package, ``False`` the non-deterministic one (see the module docstring).

    Returns
    -------
    ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v, beta`` and the FP32 gradient of ``g``, ranked
    like the inputs.  Bitwise reproducible for the same inputs and path in both modes.  In raw gate mode
    (``deterministic=True`` only) ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, as
    :func:`flashinfer.cake_kda_det_train.chunk_kda_det_bwd` returns them.
    """
    scale = DEFAULT_SCALE if scale is None else scale
    if _deterministic(deterministic, "chunk_kda_bwd"):
        workspace, arena, _save_mode = _det_call(
            workspace, arena, None, deterministic, "chunk_kda_bwd"
        )
        return _det().chunk_kda_det_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            dout,
            saved=saved,
            scale=scale,
            workspace=workspace,
            arena=arena,
            doc_lengths=doc_lengths,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
    return _nd().chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        dout,
        saved=saved,
        scale=scale,
        workspace=workspace,
        arena=_arena_for(False, arena),
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        deterministic=False,
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
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
    checkpoint_handoff: str = "auto",
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Packed KDA chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; see :func:`chunk_kda_fwd` for the
    operands, ``save_mode``, ``arena`` and ``doc_lengths`` and :func:`chunk_kda_bwd` for the gradients.
    ``deterministic`` selects the package at the forward call; the node's backward runs the same package.
    ``checkpoint_handoff`` (``"auto"``, ``"on"`` or ``"off"``): the non-deterministic package's checkpoint
    hand-off (its README); the deterministic package has no hand-off and ignores it.  ``use_gate_in_kernel``,
    ``A_log``, ``dt_bias``, ``lower_bound``: the raw gate mode (gradients to the raw gate ``g``, BF16, and to
    ``A_log`` / ``dt_bias``, FP32), ``deterministic=True`` only (see :func:`chunk_kda_fwd`).
    """
    scale = DEFAULT_SCALE if scale is None else scale
    if checkpoint_handoff not in ("auto", "on", "off"):
        raise ValueError(
            f"chunk_kda: checkpoint_handoff must be 'auto', 'on' or 'off', got {checkpoint_handoff!r}"
        )
    if _deterministic(deterministic, "chunk_kda"):
        workspace, arena, save_mode = _det_call(
            workspace, arena, save_mode, deterministic, "chunk_kda"
        )
        return _det().chunk_kda_det(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            workspace=workspace,
            save_mode=save_mode,
            arena=arena,
            doc_lengths=doc_lengths,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
    return _nd().chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale=scale,
        workspace=workspace,
        save_mode=save_mode,
        arena=_arena_for(False, arena),
        doc_lengths=doc_lengths,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        checkpoint_handoff=checkpoint_handoff,
        deterministic=False,
    )


def kda_train_arena(
    *,
    lean: bool = False,
    tail_passes: int = 1,
    batch_tokens: Optional[int] = None,
    max_docs_per_batch: Optional[int] = None,
    deterministic: bool = True,
) -> Any:
    """The lower-memory arena option of the package ``deterministic`` selects (the default, ``lean=False,
    tail_passes=1``, no batching, is the full layout); see
    :func:`flashinfer.cake_kda_det_train.kda_det_train_arena`.  The two packages' options have the same fields and
    either serves the calls of this module."""
    cls = (
        _det().KdaDetArena
        if _deterministic(deterministic, "kda_train_arena")
        else _nd().KdaArena
    )
    return cls(
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
    use_gate_in_kernel: bool = False,
    deterministic: bool = True,
) -> Any:
    """A caller-owned workspace of the package ``deterministic`` selects (one arena of internal intermediates) for
    ``S`` tokens in ``num_docs`` documents with ``H`` heads on ``device``, reused by every call of the same
    ``deterministic`` that receives it; see :func:`flashinfer.cake_kda_det_train.kda_det_train_workspace` for
    ``backward``, ``save_mode``, ``arena``, ``doc_lengths`` and ``use_gate_in_kernel`` (the gate mode of the calls
    it serves; ``True`` with ``deterministic=False`` raises ``ValueError``).  The two packages' workspaces have the
    same size."""
    det = _deterministic(deterministic, "kda_train_workspace")
    kwargs = dict(
        backward=backward,
        save_mode=save_mode,
        arena=_arena_for(det, arena),
        doc_lengths=doc_lengths,
        gate=_gate_mode(use_gate_in_kernel),
    )
    if det:
        return _det().KdaDetWorkspace.for_problem(S, num_docs, H, device, **kwargs)
    # the non-deterministic workspace takes the mode it requires
    return _nd().KdaWorkspace.for_problem(
        S, num_docs, H, device, deterministic=False, **kwargs
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
    use_gate_in_kernel: bool = False,
    deterministic: bool = True,
) -> int:
    """Bytes :func:`kda_train_workspace` allocates for the problem (the same for both values of
    ``deterministic``); see :func:`flashinfer.cake_kda_det_train.kda_det_train_workspace_bytes`
    (``use_gate_in_kernel``: the gate mode; ``True`` with ``deterministic=False`` raises ``ValueError``)."""
    det = _deterministic(deterministic, "kda_train_workspace_bytes")
    return (_det() if det else _nd()).workspace_bytes(
        S,
        num_docs,
        H,
        backward=backward,
        save_mode=save_mode,
        sm_count=sm_count,
        arena=_arena_for(det, arena),
        doc_lengths=doc_lengths,
        gate=_gate_mode(use_gate_in_kernel),
    )


def kda_train_saved_bytes(
    S: int,
    H: int,
    num_docs: Optional[int] = None,
    *,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
    deterministic: bool = True,
) -> int:
    """Bytes of the saved set (the same for both values of ``deterministic``); see
    :func:`flashinfer.cake_kda_det_train.kda_det_train_saved_bytes`."""
    det = _deterministic(deterministic, "kda_train_saved_bytes")
    return (_det() if det else _nd()).saved_bytes(
        S, H, num_docs, save_mode=save_mode, sm_count=sm_count
    )


def is_kda_train_supported(
    device: torch.device | str | int | None = None, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` (the current CUDA device by default) is a target of the package ``deterministic`` selects
    (compute capability 10.0, 10.3 or 10.7 for both)."""
    det = _deterministic(deterministic, "is_kda_train_supported")
    return (_det() if det else _nd()).is_supported(device)


__all__ = [
    "DEFAULT_SCALE",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "is_kda_train_supported",
    "kda_train_arena",
    "kda_train_saved_bytes",
    "kda_train_workspace",
    "kda_train_workspace_bytes",
]
