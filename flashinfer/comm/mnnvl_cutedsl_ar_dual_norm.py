# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNNVL all-reduce of row-parallel partials fused into the residual boundary."""

from __future__ import annotations

import torch
from torch.distributed import ProcessGroup

from ..api_logging import flashinfer_api
from ..trace.templates.comm import all_reduce_gated_residual_next_rmsnorm_trace
from .mnnvl_cutedsl.kernel_ll.all_reduce_dual_norm import (
    LLAllReduceDualNorm,
    LLAllReduceDualNormTuning,
    _DualNormConstants,
    default_all_reduce_dual_norm_tuning,
)
from .mnnvl_cutedsl.kernel_ll.column_gather import PushPlane
from .mnnvl_cutedsl.runtime import collective_support, is_dense_on
from .mnnvl_cutedsl_ag import _AllGatherWorkspace, supports_push_plane

__all__ = [
    "LLAllReduceDualNormTuning",
    "MNNVLCuteDSLAllReduceDualNormWorkspace",
    "all_reduce_gated_residual_next_rmsnorm",
    "supports_all_reduce_gated_residual_next_rmsnorm",
]


class MNNVLCuteDSLAllReduceDualNormWorkspace(_AllGatherWorkspace[LLAllReduceDualNorm]):
    @flashinfer_api
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        *,
        group: ProcessGroup,
        enable_pdl: bool = True,
        push_plane: PushPlane | None = None,
    ) -> None:
        """Create the workspace on every rank of ``group`` (collective).

        One-shot calls fit ``M * H * 2 <= half_bytes / tp_size`` on the plane,
        and ``M <= min(max_token_num, 128 // tp_size)`` on the mailbox.

        Args:
            tp_size: Number of ranks in ``group``, at least 2.
            tp_rank: This rank's index in ``group``.
            max_token_num: Largest ``M`` a call may reduce.
            hidden_dim: Hidden size ``H``, a multiple of ``8 * tp_size`` and 1024.
            group: TP process group.
            enable_pdl: Launch the kernel with programmatic dependent launch.
            push_plane: Exchange through the caller's plane instead of an own
                mailbox; must pass :func:`supports_push_plane` on every rank.
                Two-shot calls also need its ``peer_addresses``.
        """
        peers = push_plane.peer_addresses if push_plane is not None else None
        device = torch.device("cuda", torch.cuda.current_device())
        # All ranks must agree: reject or construct together.
        if push_plane is not None and not collective_support(
            supports_push_plane(push_plane, max_token_num, hidden_dim)
            # Per-rank slots must be 16-byte aligned.
            and push_plane.half_bytes % (16 * tp_size) == 0
            and (
                peers is None
                or peers.shape == (tp_size,)
                and peers.dtype == torch.int64
                and is_dense_on(peers, device, 8)
            ),
            group,
            device,
        ):
            raise ValueError("unsupported push plane")
        super().__init__(
            tp_size,
            max_token_num,
            hidden_dim,
            group,
            lambda: LLAllReduceDualNorm(
                hidden_size=hidden_dim,
                tp_size=tp_size,
                rank=tp_rank,
                capacity_m=max_token_num,
                group=group,
                enable_pdl=enable_pdl,
                push_plane=push_plane,
            ),
        )

    def fits_rows(self, num_tokens: int, pushed: bool, one_shot: bool = True) -> bool:
        """Whether ``num_tokens`` rows of one exchange fit the mailbox or one plane slot."""
        plane = self.impl.push_plane
        # num_tokens must be in [1, max_token_num].
        if not 1 <= num_tokens <= self.max_token_num:
            return False
        # Push is only compatible with one-shot.
        if pushed and not one_shot:
            return False
        # No push plane: the workspace exchanges through its own mailbox.
        if plane is None:
            # The mailbox cannot be pushed into.
            if pushed:
                return False
            # Sized in rows at construction: two-shot holds max_token_num, one-shot one_shot_rows.
            return not one_shot or num_tokens <= self.impl.one_shot_rows
        # The plane is the caller's, sized in bytes: this rank's part must fit one slot.
        # One-shot: the whole [M, H] partial. Two-shot: [M, H / tp] scatter + gather.
        row_bytes = self.hidden_dim * 2
        need = (
            num_tokens * row_bytes
            if one_shot
            else 2 * num_tokens * row_bytes // self.tp_size
        )
        # Two-shot also needs peer_addresses.
        return need <= plane.half_bytes // self.tp_size and (
            one_shot or plane.peer_addresses is not None
        )

    @flashinfer_api
    def compile(
        self,
        *,
        num_tokens: int,
        post_eps: float,
        next_eps: float,
        gate: str = "channel",
        has_post_weight: bool = False,
        post_weight_is_input_scale: bool = False,
        has_next_weight: bool = True,
        want_router: bool = False,
        pushed: bool = False,
        use_oneshot: bool | None = None,
        tuning: LLAllReduceDualNormTuning | None = None,
    ) -> None:
        """Compile the kernel a call will use, e.g. before graph capture.

        Args:
            num_tokens: ``M`` of the call; picks the default ``tuning``.
            post_eps: Epsilon of the branch RMSNorm.
            next_eps: Epsilon of the next RMSNorm.
            gate: ``"none"``, ``"scalar"`` (``[1]`` alpha and beta) or ``"channel"`` (``[H]``).
            has_post_weight: Whether calls pass ``post_weight``.
            post_weight_is_input_scale: Whether ``post_weight`` scales the branch before its norm.
            has_next_weight: Whether calls pass ``next_weight``.
            want_router: Whether calls return the FP32 copy of the next norm.
            pushed: Whether calls pass ``partial=None`` (consumer only).
            use_oneshot: The calls' ``use_oneshot``.
            tuning: Launch shape; None uses the default for ``num_tokens``.
        """
        one_shot = _one_shot(pushed, use_oneshot)
        if pushed and not one_shot:
            raise ValueError("pushed calls are one-shot")
        self.impl.compiled(
            _constants(
                self.hidden_dim,
                num_tokens,
                tuning,
                post_eps=post_eps,
                next_eps=next_eps,
                gate=gate,
                has_post_weight=has_post_weight,
                post_weight_is_input_scale=post_weight_is_input_scale,
                has_next_weight=has_next_weight,
                has_router=want_router,
                pushed=pushed,
                one_shot=one_shot,
            )
        )


def _one_shot(pushed: bool, use_oneshot: bool | None) -> bool:
    """Whether a call runs one-shot: ``use_oneshot``, else one-shot only when pushed."""
    return pushed if use_oneshot is None else use_oneshot


def _constants(
    hidden: int,
    num_tokens: int,
    tuning: LLAllReduceDualNormTuning | None,
    *,
    post_eps: float,
    next_eps: float,
    gate: str,
    has_post_weight: bool,
    post_weight_is_input_scale: bool,
    has_next_weight: bool,
    has_router: bool,
    pushed: bool,
    one_shot: bool,
) -> _DualNormConstants:
    return _DualNormConstants(
        post_eps=float(post_eps),
        next_eps=float(next_eps),
        has_post_weight=has_post_weight,
        post_weight_is_input_scale=has_post_weight and post_weight_is_input_scale,
        has_next_weight=has_next_weight,
        gate=gate,
        has_router=has_router,
        pushed=pushed,
        one_shot=one_shot,
        tuning=tuning or default_all_reduce_dual_norm_tuning(hidden, num_tokens),
    )


@flashinfer_api
def supports_all_reduce_gated_residual_next_rmsnorm(
    workspace: MNNVLCuteDSLAllReduceDualNormWorkspace,
    partial: torch.Tensor | None,
    residual: torch.Tensor,
    *,
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    post_weight: torch.Tensor | None = None,
    next_weight: torch.Tensor | None = None,
    use_oneshot: bool | None = None,
    tuning: LLAllReduceDualNormTuning | None = None,
) -> bool:
    """Check only this rank. Use ``workspace.check_support`` to agree across ranks.

    Args:
        workspace: Workspace the call will use.
        partial, residual, alpha, beta, post_weight, next_weight, tuning: The
            arguments of :func:`all_reduce_gated_residual_next_rmsnorm`.
        use_oneshot: Its ``use_oneshot``.

    Returns:
        Whether this rank's arguments fit the fused path.
    """
    hidden, device = workspace.hidden_dim, workspace.device
    pushed = partial is None
    m = residual.shape[0] if residual.ndim == 2 else 0
    # Needs a launch shape (the caller's or the default for M).
    tuning = tuning or (default_all_reduce_dual_norm_tuning(hidden, m) if m else None)
    if tuning is None:
        return False
    one_shot = _one_shot(pushed, use_oneshot)
    # The rows must fit the workspace (also rejects pushed 2-shot).
    if not workspace.fits_rows(m, pushed, one_shot):
        return False
    # residual must be [M, H] FP32, dense, 32-byte aligned.
    if not (
        residual.shape == (m, hidden)
        and residual.dtype == torch.float32
        and is_dense_on(residual, device, 32)
    ):
        return False
    # partial must be [M, H] BF16, dense, 16-byte aligned.
    if not pushed and not (
        partial.shape == (m, hidden)
        and partial.dtype == torch.bfloat16
        and is_dense_on(partial, device, 16)
    ):
        return False
    # alpha and beta must be given together, as [1] or [H].
    gate_len = alpha.numel() if alpha is not None else 0
    if (alpha is None) != (beta is None) or gate_len not in (0, 1, hidden):
        return False
    # alpha/beta must be FP32 and post_weight/next_weight [H] BF16, all dense and aligned.
    vectors = [(v, torch.float32, 32 if gate_len > 1 else 4) for v in (alpha, beta)]
    vectors += [(v, torch.bfloat16, 16) for v in (post_weight, next_weight)]
    for v, dtype, align in vectors:
        shape = (gate_len,) if dtype == torch.float32 else (hidden,)
        if v is not None and not (
            v.dtype == dtype and is_dense_on(v, device, align) and v.shape == shape
        ):
            return False
    # H must split into 8-element vectors over the cluster's threads.
    return hidden % (8 * tuning.threads * tuning.cluster_size) == 0


@flashinfer_api(trace=all_reduce_gated_residual_next_rmsnorm_trace)
def all_reduce_gated_residual_next_rmsnorm(
    partial: torch.Tensor | None,
    residual: torch.Tensor,
    workspace: MNNVLCuteDSLAllReduceDualNormWorkspace,
    *,
    post_eps: float,
    next_eps: float,
    alpha: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    post_weight: torch.Tensor | None = None,
    post_weight_is_input_scale: bool = False,
    next_weight: torch.Tensor | None = None,
    want_router: bool = False,
    use_oneshot: bool | None = None,
    tuning: LLAllReduceDualNormTuning | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """All-reduce row-parallel partials across TP, then the gated residual boundary.

    With ``b = bf16(sum_r partial[r])`` (FP32 sum in rank order)::

        b = bf16(b * w_post)                        # post_weight_is_input_scale
        n = bf16(b * rsqrt(mean(b**2) + post_eps) * w_post)   # w_post if not input scale
        residual_out = alpha * residual + beta * n  # FP32; alpha = beta = 1 without a gate
        y = residual_out * rsqrt(mean(residual_out**2) + next_eps) * w_next
        norm_out = bf16(y); router_out = y

    With ``tuning.cluster_size > 1``, ``mean(residual_out**2)`` comes from the
    moments of ``alpha * residual`` and ``beta * b`` (one cluster reduction),
    without the BF16 rounding of ``n``.

    One-shot and two-shot give the same ``b`` up to the sign of a zero sum, and
    a pushed call the same ``b`` as a one-shot call on the producer's output.

    Args:
        partial: ``[M, H]`` BF16 row-parallel partial of this rank, or None when
            a producer already stored every rank's partial into the workspace's
            :class:`PushPlane`, just before on the same stream and without
            advancing the epoch. The producer must store with the plane's
            ``atom_bytes`` sentinel convention; FlashInfer's peer-store BF16
            GEMM (``layout="broadcast"``, ``slot_bytes = half_bytes / TP``)
            writes 32-bit atoms, so its plane needs ``atom_bytes=4``.
        residual: ``[M, H]`` FP32 stream, identical on every rank.
        workspace: Workspace created on every rank of the TP group.
        post_eps: Epsilon of the branch RMSNorm.
        next_eps: Epsilon of the next RMSNorm.
        alpha: ``[H]`` or ``[1]`` FP32 stream gate, or None (no gate).
        beta: ``[H]`` or ``[1]`` FP32 branch gate, or None (no gate).
        post_weight: Optional ``[H]`` BF16 branch norm weight ``w_post``.
        post_weight_is_input_scale: Apply ``post_weight`` before the branch norm.
        next_weight: Optional ``[H]`` BF16 weight ``w_next`` of the next RMSNorm.
        want_router: Also return ``y`` in FP32.
        use_oneshot: True: every rank multicasts its whole partial to every
            rank (one exchange round). False: two-shot (scatter slices, reduce,
            multicast the slice sums). None: two-shot, or one-shot for pushed
            calls. Pushed calls are one-shot; False rejects them.
        tuning: Launch shape; None uses the default for ``M``. Compile it with
            :meth:`MNNVLCuteDSLAllReduceDualNormWorkspace.compile` before graph capture.

    Returns:
        ``(residual_out, norm_out, router_out)``; ``router_out`` is None unless
        ``want_router``.
    """
    vectors = dict(
        alpha=alpha, beta=beta, post_weight=post_weight, next_weight=next_weight
    )
    if not supports_all_reduce_gated_residual_next_rmsnorm(
        workspace, partial, residual, use_oneshot=use_oneshot, tuning=tuning, **vectors
    ):
        raise ValueError("unsupported all-reduce dual norm arguments")
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual, dtype=torch.bfloat16)
    router_out = torch.empty_like(residual) if want_router else None
    gate = "none" if alpha is None else "scalar" if alpha.numel() == 1 else "channel"
    one_shot = _one_shot(partial is None, use_oneshot)
    constants = _constants(
        workspace.hidden_dim,
        residual.shape[0],
        tuning,
        post_eps=post_eps,
        next_eps=next_eps,
        gate=gate,
        has_post_weight=post_weight is not None,
        post_weight_is_input_scale=post_weight_is_input_scale,
        has_next_weight=next_weight is not None,
        has_router=want_router,
        pushed=partial is None,
        one_shot=one_shot,
    )
    workspace.impl(
        partial,
        residual,
        residual_out,
        norm_out,
        router_out,
        constants=constants,
        **vectors,
    )
    return residual_out, norm_out, router_out
