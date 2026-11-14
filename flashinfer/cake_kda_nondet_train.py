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
(SM100 / SM103 / SM107), the Cake backend, the non-deterministic package.  Its
backward moves the ``q * dq`` term of the gate gradient ``dg`` from the fused
dq / dk / dg program into the GLM intra program (the ``S1`` pairs), so the
fused program no longer reads ``q``.  The schedule has a fixed order: repeated
calls on the same inputs and path give bitwise the same results.  ``out``,
``dq``, ``dk``, ``dv`` and ``dbeta`` are bitwise the deterministic package's
(``flashinfer.cake_kda_det_train``), and ``dg`` differs from it by FP32
rounding only (the same accuracy budget).  On some rows the checkpoint replay
(``saved=None``) and a saved-set backward take different program pairs, and
their ``dg`` then differ by FP32 rounding too.  The raw gate mode
(``use_gate_in_kernel=True``) and the in-kernel Q/K L2 norm
(``use_qk_l2norm_in_kernel=True``) run copies of the deterministic package's
programs in the same order, so ``out`` and every gradient (``dA_log`` and
``ddt_bias`` included) are bitwise the deterministic package's in that mode.

No launch of this package is unordered, so ``torch.use_deterministic_algorithms(True)``
(also with ``warn_only=True``) raises no alert here, and the entries never run
the deterministic package.  They refuse a workspace, a saved set or an arena
option of the deterministic package with ``ValueError`` naming the mode.

Checkpoint hand-off: under ``torch.utils.checkpoint`` (reentrant or not),
:func:`cake_kda_nondet` lets the forward that torch re-runs inside the backward
hand its state tapes to the node's backward, which then skips its recurrence
recompute; the gradients do not change.  ``checkpoint_handoff="auto"`` (the
default) engages on the verified devices and problem sizes the package README
lists, ``"on"`` on every call it supports, ``"off"`` never; the keyword is the
only switch (the package reads no environment variable).  The raw gate mode and
the in-kernel Q/K L2 norm never engage it.

Thin entry points.  The contract, the host plan, the workspace,
the launch sequences, the autograd Function and the generated kernels live in
``flashinfer.experimental.cake.cake_kda_nondet_train``; this module validates nothing
itself and forwards every keyword.  Every switch is a keyword argument; this
module reads no environment variable.

Operands (one packed batch of documents): ``q, k, v`` BF16 ``[1, S, H, 128]``
(``q`` / ``k`` L2-normalized by the caller, or raw with
``use_qk_l2norm_in_kernel=True``), ``g`` FP32 ``[1, S, H, 128]`` (the
natural-log decay gate, preactivated), ``beta`` BF16 ``[1, S, H]`` (through
the sigmoid), ``cu_seqlens`` int32 ``[num_docs + 1]`` on the device.  The
batch axis of 1 is optional; outputs take the rank of the inputs.  Gradients:
``dq, dk, dv, dbeta`` BF16, ``dg`` FP32.  Per-document zero initial state, no
final state, ``scale = 128 ** -0.5`` by default, ``H`` in ``1..64``.

Usage::

    from flashinfer.cake_kda_nondet_train import (
        cake_kda_nondet, cake_kda_nondet_train_arena, cake_kda_nondet_train_backward, cake_kda_nondet_train_forward,
        cake_kda_nondet_train_workspace,
    )
    from torch.utils.checkpoint import checkpoint

    out = cake_kda_nondet(q, k, v, g, beta, cu_seqlens)                  # autograd
    out = checkpoint(cake_kda_nondet, q, k, v, g, beta, cu_seqlens, use_reentrant=False)  # checkpoint hand-off ("auto")
    out = cake_kda_nondet(q, k, v, g, beta, cu_seqlens, checkpoint_handoff="off")  # no hand-off
    out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu_seqlens, save_for_backward=True)
    dq, dk, dv, dbeta, dg = cake_kda_nondet_train_backward(q, k, v, g, beta, cu_seqlens, dout, saved=saved)
    dq, dk, dv, dbeta, dg = cake_kda_nondet_train_backward(q, k, v, g, beta, cu_seqlens, dout)  # checkpoint replay

    ws = cake_kda_nondet_train_workspace(S, num_docs, H, device)  # caller-owned, reused
    out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu_seqlens, workspace=ws)
    # saved set v2: the backward skips its recurrence recompute
    ws = cake_kda_nondet_train_workspace(S, num_docs, H, device, save_mode="tape")
    out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu_seqlens, workspace=ws)
    # the lower-memory arena option (off by default): a smaller backward arena, the same gradients bitwise
    arena = cake_kda_nondet_train_arena(lean=True, tail_passes=4)
    ws = cake_kda_nondet_train_workspace(S, num_docs, H, device, arena=arena)
    out = cake_kda_nondet(q, k, v, g, beta, cu_seqlens, workspace=ws)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The default scale of the GLM contract: ``head_dim ** -0.5`` with head dimension 128.
DEFAULT_SCALE = 128**-0.5


@flashinfer_api
def cake_kda_nondet_train_forward(
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
    use_qk_l2norm_in_kernel: bool = False,
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
    workspace : CakeKDANondetWorkspace, optional
        A reused workspace from :func:`cake_kda_nondet_train_workspace`; without one this call allocates its own.  A
        workspace of the deterministic package is refused (``ValueError``).
    save_mode : str, optional
        ``"light"`` (saved set version 4: ``Aqk`` / ``Akk`` and the tables, 256 B per token-head, the forward's
        gated queries ``qg``, 256 B per internal token-head, plus the segment-boundary states on the segmented
        route when ``S >= 384 * NSEG``, so the backward skips the forward half of its recurrence recompute, and
        the WY representation on the rows whose backward reads it from the set) or ``"tape"`` (version 2: also
        the forward's state tape and ``v_new``, 768 B more per token-head, so the backward skips the whole
        recompute); ``None`` takes the workspace's default (``"light"`` without one).
    arena : CakeKDANondetArena, optional
        The lower-memory arena option (:func:`cake_kda_nondet_train_arena`; off by default): the forward's layout is the
        same under every option, and a batched option (``batch_tokens``) runs the forward once per document
        group; ``None`` takes the workspace's option, a differing option on a given workspace is refused, as is an
        option of the deterministic package.
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
    use_qk_l2norm_in_kernel : bool
        ``False`` (the default): ``q`` / ``k`` are L2-normalized by the caller and contiguous.  ``True``: ``q`` /
        ``k`` are the raw rows, normalized per token and head inside the kernels (``bf16(x * r)``, ``r = 1 /
        sqrt(sum(x^2) + 1e-6)`` in FP32); they may be strided views (unit innermost stride, head stride 128, token
        stride a multiple of 8 elements and at least ``128 * H``, a 16-byte aligned address; never copied).  The
        saved set adds the FP32 reciprocal norms ``[2, S, H]`` (no normalized rows are kept) and records the mode;
        the workspace must be allocated with ``use_qk_l2norm_in_kernel=True``.

    Returns
    -------
    ``(out, saved)``: ``out`` BF16 like ``v``; ``saved`` a ``CakeKDANondetChunkSaved`` (versioned) or ``None``.
    ``out`` and the saved set are bitwise the deterministic package's (the same forward programs).
    """
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.cake_kda_nondet_train_forward(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@flashinfer_api
def cake_kda_nondet_train_backward(
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
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, ...]:
    r"""Packed KDA chunked training backward.

    Parameters
    ----------
    q, k, v, g, beta, cu_seqlens
        The forward's operands.
    dout : torch.Tensor
        BF16 gradient of ``out``, laid out like ``v``.
    saved : CakeKDANondetChunkSaved, optional
        The forward's saved set (version 4 or 2) of :func:`cake_kda_nondet_train_forward`; ``None`` recomputes it from
        the inputs (checkpoint replay).  A saved set of the deterministic package is refused (``ValueError``).
    scale : float, optional
        The forward's scale; ``128 ** -0.5`` when ``None``.
    workspace : CakeKDANondetWorkspace, optional
        A reused workspace with backward buffers; without one this call allocates its own.  A workspace of the
        deterministic package is refused (``ValueError``).
    arena : CakeKDANondetArena, optional
        The lower-memory arena option (:func:`cake_kda_nondet_train_arena`; off by default) of that per-call workspace:
        the lean backward layouts and the tail split (``tail_passes``), the same gradients bitwise; ``None``
        takes a given workspace's option, a differing one is refused, as is an option of the deterministic
        package.  A lean arena serves no checkpoint replay (``saved=None``); a batched option (``batch_tokens``)
        runs the backward once per document group.
    doc_lengths : sequence of int, optional
        The documents' token counts in ``cu_seqlens`` order (host integers), required under a batched option and
        checked against ``cu_seqlens`` on the device.
    use_gate_in_kernel, A_log, dt_bias, lower_bound
        The forward's gate mode and raw gate parameters (see :func:`cake_kda_nondet_train_forward`); a saved set of
        the other mode, or with another ``lower_bound``, is refused.
    use_qk_l2norm_in_kernel : bool
        The forward's in-kernel Q/K L2 norm mode (``q`` / ``k`` then the raw rows; ``dq`` / ``dk`` their
        gradients); a saved set of the other mode is refused.

    Returns
    -------
    tuple of torch.Tensor
        ``(dq, dk, dv, dbeta, dg)``: BF16 gradients of ``q, k, v, beta`` and the FP32 gradient of ``g``, ranked
        like the inputs.  With ``use_gate_in_kernel=True``: ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, ``dg``
        the BF16 gradient of the raw gate (contiguous, ranked like ``g``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias``
        FP32 shaped like ``dt_bias``.  Bitwise reproducible for the same inputs and path.  ``dq``, ``dk``, ``dv`` and
        ``dbeta`` are bitwise the deterministic package's and ``dg`` differs from it by FP32 rounding; in raw gate mode
        and with the in-kernel Q/K L2 norm every gradient is bitwise the deterministic package's.
    """
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.cake_kda_nondet_train_backward(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@flashinfer_api
def cake_kda_nondet(
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
    use_qk_l2norm_in_kernel: bool = False,
    checkpoint_handoff: str = "auto",
) -> torch.Tensor:
    r"""Packed KDA chunked training as an autograd node (``torch.utils.checkpoint`` compatible).

    Returns ``out`` BF16 like ``v`` with gradients to ``q, k, v, g, beta``; see :func:`cake_kda_nondet_train_forward` for the
    operands, ``save_mode``, ``arena``, ``doc_lengths`` (the node records the option and the document lengths,
    so a backward that allocates its own workspace takes them too) and the raw gate mode
    (``use_gate_in_kernel=True``, ``A_log``, ``dt_bias``, ``lower_bound``: gradients to the raw gate ``g``, BF16,
    and to ``A_log`` / ``dt_bias``, FP32), the in-kernel Q/K L2 norm (``use_qk_l2norm_in_kernel=True``: raw ``q`` /
    ``k`` in, their gradients out) and :func:`cake_kda_nondet_train_backward` for the gradients.  The node's
    backward is this package's.  ``checkpoint_handoff`` (``"auto"``, ``"on"`` or ``"off"``; another value raises
    ``ValueError``) is the checkpoint hand-off under ``torch.utils.checkpoint`` (the module docstring; the raw gate
    mode and the in-kernel Q/K L2 norm never engage it).

    Parameters
    ----------
    q, k, v : torch.Tensor
        BF16 packed rows ``[1, S, H, 128]`` (or ``[S, H, 128]``), as in
        :func:`cake_kda_nondet_train_forward`.
    g : torch.Tensor
        FP32 activated decay gate; raw BF16 gate when ``use_gate_in_kernel=True``.
    beta : torch.Tensor
        BF16 ``[1, S, H]`` delta-rule gate after the sigmoid.
    cu_seqlens : torch.Tensor
        int32 device document offsets, first 0 and last ``S``.
    scale : float, optional
        Output scale; ``128 ** -0.5`` when ``None``.
    workspace : optional
        Caller-owned workspace from :func:`cake_kda_nondet_train_workspace`.
    save_mode : str, optional
        ``"light"`` or ``"tape"``; ``None`` uses the workspace's default (``"light"`` without one).
    arena : optional
        Lower-memory arena option from :func:`cake_kda_nondet_train_arena`.
    doc_lengths : sequence of int, optional
        Host document lengths in offset order; required by a batched arena.
    use_gate_in_kernel : bool, default=False
        Activate the raw BF16 gate inside the kernels.
    A_log : torch.Tensor, optional
        Contiguous FP32 ``[H]`` raw-gate parameter.
    dt_bias : torch.Tensor, optional
        Contiguous FP32 ``[H * 128]`` or ``[H, 128]`` raw-gate bias.
    lower_bound : float, optional
        Required in raw-gate mode, with ``-5 <= lower_bound < 0``.
    use_qk_l2norm_in_kernel : bool, default=False
        Normalize raw query and key rows inside the kernels; input layout rules are
        those of :func:`cake_kda_nondet_train_forward`.
    checkpoint_handoff : str, default="auto"
        Checkpoint hand-off mode: ``"auto"``, ``"on"`` or ``"off"``.
    """
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.cake_kda_nondet(
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        checkpoint_handoff=checkpoint_handoff,
    )


def cake_kda_nondet_train_arena(
    *,
    lean: bool = False,
    tail_passes: int = 1,
    batch_tokens: Optional[int] = None,
    max_docs_per_batch: Optional[int] = None,
) -> Any:
    """The lower-memory arena option (a ``CakeKDANondetArena``; the default, ``lean=False, tail_passes=1``, no batching,
    is the full layout).  ``lean=True`` selects the lean backward layouts (4108 B per internal token-head with one tail
    pass instead of the default's 4488, tape 3720; no checkpoint replay on such a workspace); ``tail_passes >= 2``
    (implies ``lean``) runs the backward's tail, the fused gradient / GLM intra / gate epilogue, once per range of
    ``ceil(NC_pad / tail_passes)`` chunks over per-range buffers: 2820 B per internal token-head plus three state blocks
    on the floor, the same gradients bitwise, some speed cost (see the package README).  ``batch_tokens`` runs the
    forward and the backward over groups of whole, contiguous documents of at most that many tokens (and at most
    ``max_docs_per_batch`` documents, which needs ``batch_tokens``) in an arena sized for the largest group (at
    ``tail_passes``; each group runs its tail in the fewest passes up to that which fit the arena); the calls then take
    ``doc_lengths``.  Bitwise the same results, except that an exactly-zero ``u`` / ``w`` entry of a chunk whose pair
    partner changed may change sign; refused on the rows with at most four (document, head) chains on the segmented
    route."""
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.CakeKDANondetArena(
        lean=lean,
        tail_passes=tail_passes,
        batch_tokens=batch_tokens,
        max_docs_per_batch=max_docs_per_batch,
    )


def cake_kda_nondet_train_workspace(
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
    use_qk_l2norm_in_kernel: bool = False,
) -> Any:
    """A caller-owned ``CakeKDANondetWorkspace`` (one arena of internal intermediates) for ``S`` tokens in ``num_docs``
    documents with ``H`` heads on ``device``, reused by every call that receives it: the forward layout only with
    ``backward=False``; ``save_mode`` is the default of the calls on it and sizes the arena for that mode;
    ``arena`` (:func:`cake_kda_nondet_train_arena`) is the lower-memory arena option, off when ``None``; under a batched
    option ``doc_lengths`` (the documents' token counts) sizes the arena for the exact document groups instead of
    the bound of every group; ``use_gate_in_kernel`` / ``use_qk_l2norm_in_kernel`` are the gate and Q/K L2 norm
    modes of the calls it serves (a call in another mode is refused).  The deterministic package's entries refuse it,
    and this package's entries refuse the deterministic package's workspace."""
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.CakeKDANondetWorkspace.for_problem(
        S,
        num_docs,
        H,
        device,
        backward=backward,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
        gate=_gate(use_gate_in_kernel),
        l2n=_l2n(use_qk_l2norm_in_kernel),
    )


def cake_kda_nondet_train_workspace_bytes(
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
    use_qk_l2norm_in_kernel: bool = False,
) -> int:
    """Bytes :func:`cake_kda_nondet_train_workspace` allocates for the problem (``sm_count``: the device's SM count, the
    current CUDA device's when ``None``; it sets the segment count of the segmented recurrences; ``arena``: the
    lower-memory arena option, off when ``None``; ``doc_lengths``: the exact document groups of a batched option,
    else their bound; ``use_gate_in_kernel``: the gate mode, whose backward layouts carry no gate epilogue
    buffers in raw mode; ``use_qk_l2norm_in_kernel``: the Q/K L2 norm mode, likewise without those buffers)."""
    from .experimental.cake.cake_kda_nondet_train import cake_backend

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
        l2n=_l2n(use_qk_l2norm_in_kernel),
    )


def _gate(use_gate_in_kernel: bool) -> str:
    """The backend's gate mode of ``use_gate_in_kernel`` (``ValueError`` for a non-bool)."""
    if not isinstance(use_gate_in_kernel, bool):
        raise ValueError(
            f"use_gate_in_kernel must be a bool, got {use_gate_in_kernel!r}"
        )
    return "raw" if use_gate_in_kernel else "g"


def _l2n(use_qk_l2norm_in_kernel: bool) -> str:
    """The backend's Q/K L2 norm mode of ``use_qk_l2norm_in_kernel`` (``ValueError`` for a non-bool)."""
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise ValueError(
            f"use_qk_l2norm_in_kernel must be a bool, got {use_qk_l2norm_in_kernel!r}"
        )
    return "fused" if use_qk_l2norm_in_kernel else "off"


def cake_kda_nondet_train_saved_bytes(
    S: int,
    H: int,
    num_docs: Optional[int] = None,
    *,
    save_mode: str = "light",
    sm_count: Optional[int] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> int:
    """Bytes of the saved set: ``Aqk`` and ``Akk`` (``256 * S * H``), plus, when ``num_docs`` is given, the device
    tables and the version's per-call tensors: the forward's gated queries ``qg`` (and, on the rows whose backward
    reads it from the set, the WY representation ``w`` / ``u``) and the segment-boundary states under the size rule
    with ``save_mode="light"``; the state tape, ``v_new`` and the segment transfers with ``save_mode="tape"``;
    with ``use_qk_l2norm_in_kernel=True`` the FP32 reciprocal norms ``[2, S, H]`` (``8 * S * H``) in either."""
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.saved_bytes(
        S,
        H,
        num_docs,
        save_mode=save_mode,
        sm_count=sm_count,
        l2n=_l2n(use_qk_l2norm_in_kernel),
    )


def is_cake_kda_nondet_train_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    """Whether ``device`` (the current CUDA device by default) is a target of the Cake kernels
    (compute capability 10.0, 10.3 or 10.7)."""
    from .experimental.cake.cake_kda_nondet_train import cake_backend

    return cake_backend.is_supported(device)


__all__ = [
    "DEFAULT_SCALE",
    "cake_kda_nondet",
    "cake_kda_nondet_train_arena",
    "cake_kda_nondet_train_backward",
    "cake_kda_nondet_train_forward",
    "cake_kda_nondet_train_saved_bytes",
    "cake_kda_nondet_train_workspace",
    "cake_kda_nondet_train_workspace_bytes",
    "is_cake_kda_nondet_train_supported",
]
