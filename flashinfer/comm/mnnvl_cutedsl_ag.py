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

"""MNNVL column all-gather fused with a gated residual and the next RMSNorm.

Each TP rank holds the same tokens and a contiguous ``[M, H/TP]`` block of
columns (for example a column-sharded output projection). The op gathers the
blocks in rank order and runs the residual boundary of
``flashinfer.norm.gated_residual_next_rmsnorm`` on the full rows, without
materializing the gathered ``[M, H]`` tensor. It is two kernels (LL publish,
then gather + dual norm) over a dedicated LL mailbox.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .mnnvl_cutedsl.runtime import require_mnnvl_multicast_device
from .mnnvl_cutedsl.kernel_ll.all_gather_dual_norm import (
    LLAllGatherDualNorm,
    LLAllGatherDualNormTuning,
    _NormConstants,
)

__all__ = [
    "LLAllGatherDualNormTuning",
    "MNNVLCuteDSLAllGatherDualNormWorkspace",
    "all_gather_gated_residual_next_rmsnorm",
    "supports_all_gather_gated_residual_next_rmsnorm",
]


class MNNVLCuteDSLAllGatherDualNormWorkspace:
    """Mailbox, stage state and compiled kernels for one TP group.

    Construction is collective over ``group``. Compile every constant set
    with :meth:`compile` before CUDA graph capture. Calls must not overlap.
    """

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        *,
        group: ProcessGroup,
        tuning: LLAllGatherDualNormTuning | None = None,
    ) -> None:
        if tp_size not in (2, 4, 8):
            raise ValueError("tp_size must be 2, 4, or 8")
        if max_token_num <= 0:
            raise ValueError("max_token_num must be positive")
        device = require_mnnvl_multicast_device()
        self.tp_size = tp_size
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.device = device
        self._group = group
        self._impl: LLAllGatherDualNorm | None = LLAllGatherDualNorm(
            hidden_size=hidden_dim,
            tp_size=tp_size,
            rank=tp_rank,
            capacity_m=max_token_num,
            group=group,
            tuning=tuning or LLAllGatherDualNormTuning(),
        )

    @property
    def impl(self) -> LLAllGatherDualNorm:
        if self._impl is None:
            raise RuntimeError("workspace has been destroyed")
        return self._impl

    def destroy(self) -> None:
        """Release the mailbox once in-flight work on every rank is done."""
        if self._impl is not None:
            torch.cuda.synchronize(self.device)
            dist.barrier(self._group)
            self._impl = None

    def compile(
        self,
        *,
        post_eps: float,
        next_eps: float,
        next_gain_center: float = 0.0,
        has_post_weight: bool = False,
        post_gain_center: float = 0.0,
    ) -> None:
        """Compile one constant set ahead of time, e.g. before graph capture."""
        self.impl.compiled(
            _norm_constants(
                post_eps, next_eps, next_gain_center, has_post_weight, post_gain_center
            )
        )


def _norm_constants(
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    has_post_weight: bool,
    post_gain_center: float,
) -> _NormConstants:
    return _NormConstants(
        post_eps=float(post_eps),
        next_eps=float(next_eps),
        post_gain_center=float(post_gain_center) if has_post_weight else 0.0,
        next_gain_center=float(next_gain_center),
        has_post_weight=has_post_weight,
    )


def _is_dense(t: torch.Tensor, device: torch.device, alignment: int) -> bool:
    return t.device == device and t.is_contiguous() and t.data_ptr() % alignment == 0


def supports_all_gather_gated_residual_next_rmsnorm(
    workspace: MNNVLCuteDSLAllGatherDualNormWorkspace,
    branch_shard: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    post_weight: torch.Tensor | None = None,
) -> bool:
    """Whether this call fits the workspace; identical on every rank for equal M."""
    hidden = workspace.hidden_dim
    device = workspace.device
    m = branch_shard.shape[0] if branch_shard.ndim == 2 else 0
    vectors = [(alpha, torch.float32, 32), (beta, torch.float32, 32)]
    vectors.append((next_weight, torch.bfloat16, 16))
    if post_weight is not None:
        vectors.append((post_weight, torch.bfloat16, 16))
    return (
        1 <= m <= workspace.max_token_num
        and branch_shard.shape[1] == hidden // workspace.tp_size
        and branch_shard.dtype == torch.bfloat16
        and _is_dense(branch_shard, device, 16)
        and residual.shape == (m, hidden)
        and residual.dtype == torch.float32
        and _is_dense(residual, device, 32)
        and all(
            v.shape == (hidden,) and v.dtype == dtype and _is_dense(v, device, align)
            for v, dtype, align in vectors
        )
    )


def all_gather_gated_residual_next_rmsnorm(
    branch_shard: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    workspace: MNNVLCuteDSLAllGatherDualNormWorkspace,
    *,
    post_eps: float,
    next_eps: float,
    next_gain_center: float = 0.0,
    post_weight: torch.Tensor | None = None,
    post_gain_center: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather column shards across TP, then the gated residual and next RMSNorm.

    With ``branch = concat_r(branch_shard[r], dim=1)`` and
    ``b = branch * (post_weight + post_gain_center)`` (or ``branch``)::

        residual_out = alpha * residual + (beta * b) * rsqrt(mean(b**2) + post_eps)
        ffn_input = (
            residual_out
            * rsqrt(mean(residual_out**2) + next_eps)
            * (next_weight + next_gain_center)
        )

    Args:
        branch_shard: ``[M, H/TP]`` BF16 columns ``[rank*H/TP, (rank+1)*H/TP)``.
        residual: ``[M, H]`` FP32 stream, identical on every rank.
        alpha: ``[H]`` FP32 stream gate.
        beta: ``[H]`` FP32 branch gate.
        next_weight: ``[H]`` BF16 weight applied after the next normalization.
        workspace: Workspace created on every rank of the TP group.
        post_eps: Epsilon of the branch RMSNorm.
        next_eps: Epsilon of the next RMSNorm.
        next_gain_center: Constant added to ``next_weight``.
        post_weight: Optional ``[H]`` BF16 weight applied before the branch norm.
        post_gain_center: Constant added to ``post_weight``.

    Returns:
        ``(residual_out, ffn_input)``.
    """
    if not supports_all_gather_gated_residual_next_rmsnorm(
        workspace, branch_shard, residual, alpha, beta, next_weight, post_weight
    ):
        raise ValueError("unsupported all-gather dual norm arguments")
    residual_out = torch.empty_like(residual)
    ffn_input = torch.empty_like(residual, dtype=torch.bfloat16)
    workspace.impl(
        branch_shard,
        residual,
        alpha,
        beta,
        next_weight,
        residual_out,
        ffn_input,
        constants=_norm_constants(
            post_eps,
            next_eps,
            next_gain_center,
            post_weight is not None,
            post_gain_center,
        ),
        post_weight=post_weight,
    )
    return residual_out, ffn_input
