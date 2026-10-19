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

"""MNNVL column all-gathers fused into the consumer of the gathered rows.

Each TP rank holds the same tokens and a contiguous ``[M, H/TP]`` block of
columns (for example a column-sharded output projection). Each op gathers the
blocks in rank order and runs its consumer on the full rows, without
materializing the gathered ``[M, H]`` tensor: either the residual boundary of
``flashinfer.norm.gated_residual_next_rmsnorm`` or an RMSNorm followed by
NVFP4 quantization. Each is two kernels (LL publish, then gather + consumer)
over a dedicated LL mailbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .mnnvl_cutedsl.runtime import (
    collective_support,
    is_dense_on,
    require_mnnvl_multicast_device,
)
from ..api_logging import flashinfer_api
from ..trace.templates.comm import (
    all_gather_gated_residual_next_rmsnorm_trace,
    all_gather_rmsnorm_fp4_quantize_trace,
)
from .mnnvl_cutedsl.kernel_ll.all_gather_dual_norm import (
    LLAllGatherDualNorm,
    LLAllGatherDualNormTuning,
    _NormConstants,
)
from .mnnvl_cutedsl.kernel_ll.all_gather_norm_quant import (
    LLAllGatherNormQuant,
    _NormQuantConstants,
)

__all__ = [
    "LLAllGatherDualNormTuning",
    "MNNVLCuteDSLAllGatherDualNormWorkspace",
    "MNNVLCuteDSLAllGatherNormQuantWorkspace",
    "all_gather_gated_residual_next_rmsnorm",
    "all_gather_rmsnorm_fp4_quantize",
    "supports_all_gather_gated_residual_next_rmsnorm",
    "supports_all_gather_rmsnorm_fp4_quantize",
]


_Impl = TypeVar("_Impl")


class _AllGatherWorkspace(Generic[_Impl]):
    """Mailbox, stage state and compiled kernels for one TP group.

    Construction is collective over ``group``. Compile every constant set
    with :meth:`compile` before CUDA graph capture. Calls must not overlap.
    Before selecting this path, pass each rank's local ``supports_*`` verdict
    to :meth:`check_support` on every rank. Local checks alone are not a
    rank-consistent fallback decision.
    """

    def __init__(
        self,
        tp_size: int,
        max_token_num: int,
        hidden_dim: int,
        group: ProcessGroup,
        create: Callable[[], _Impl],
    ) -> None:
        if tp_size < 2:
            raise ValueError("tp_size must be at least 2")
        if max_token_num <= 0:
            raise ValueError("max_token_num must be positive")
        self.device = require_mnnvl_multicast_device()
        self.tp_size = tp_size
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self._group = group
        self._impl: _Impl | None = create()

    def check_support(self, local_supported: bool) -> bool:
        """Collectively agree on a local ``supports_*`` result before capture.

        Call on every rank during warmup, even when its local result is False.
        All ranks must use the agreed path and keep valid layouts for replay.
        """
        return collective_support(local_supported, self._group, self.device)

    @property
    def impl(self) -> _Impl:
        if self._impl is None:
            raise RuntimeError("workspace has been destroyed")
        return self._impl

    def destroy(self) -> None:
        """Release the mailbox once in-flight work on every rank is done."""
        if self._impl is not None:
            torch.cuda.synchronize(self.device)
            dist.barrier(self._group)
            self._impl = None

    def fits(self, shard: torch.Tensor) -> bool:
        """Whether this rank's ``[M, H/TP]`` BF16 shard fits the workspace."""
        m = shard.shape[0] if shard.ndim == 2 else 0
        return (
            1 <= m <= self.max_token_num
            and shard.shape[1] == self.hidden_dim // self.tp_size
            and shard.dtype == torch.bfloat16
            and is_dense_on(shard, self.device, 16)
        )


class MNNVLCuteDSLAllGatherDualNormWorkspace(_AllGatherWorkspace[LLAllGatherDualNorm]):
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
        super().__init__(
            tp_size,
            max_token_num,
            hidden_dim,
            group,
            lambda: LLAllGatherDualNorm(
                hidden_size=hidden_dim,
                tp_size=tp_size,
                rank=tp_rank,
                capacity_m=max_token_num,
                group=group,
                tuning=tuning or LLAllGatherDualNormTuning(),
            ),
        )

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


def supports_all_gather_gated_residual_next_rmsnorm(
    workspace: MNNVLCuteDSLAllGatherDualNormWorkspace,
    branch_shard: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    post_weight: torch.Tensor | None = None,
) -> bool:
    """Check only this rank. Use ``workspace.check_support`` to agree across ranks."""
    hidden = workspace.hidden_dim
    device = workspace.device
    vectors = [(alpha, torch.float32, 32), (beta, torch.float32, 32)]
    vectors.append((next_weight, torch.bfloat16, 16))
    if post_weight is not None:
        vectors.append((post_weight, torch.bfloat16, 16))
    return (
        workspace.fits(branch_shard)
        and residual.shape == (branch_shard.shape[0], hidden)
        and residual.dtype == torch.float32
        and is_dense_on(residual, device, 32)
        and all(
            v.shape == (hidden,) and v.dtype == dtype and is_dense_on(v, device, align)
            for v, dtype, align in vectors
        )
    )


@flashinfer_api(trace=all_gather_gated_residual_next_rmsnorm_trace)
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


class MNNVLCuteDSLAllGatherNormQuantWorkspace(
    _AllGatherWorkspace[LLAllGatherNormQuant]
):
    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        *,
        group: ProcessGroup,
        enable_pdl: bool = True,
    ) -> None:
        super().__init__(
            tp_size,
            max_token_num,
            hidden_dim,
            group,
            lambda: LLAllGatherNormQuant(
                hidden_size=hidden_dim,
                tp_size=tp_size,
                rank=tp_rank,
                capacity_m=max_token_num,
                group=group,
                enable_pdl=enable_pdl,
            ),
        )

    def compile(self, *, eps: float, gain_center: float = 0.0) -> None:
        """Compile one constant set ahead of time, e.g. before graph capture."""
        self.impl.compiled(_NormQuantConstants(float(eps), float(gain_center)))


def supports_all_gather_rmsnorm_fp4_quantize(
    workspace: MNNVLCuteDSLAllGatherNormQuantWorkspace,
    shard: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
) -> bool:
    """Check only this rank. Use ``workspace.check_support`` to agree across ranks."""
    hidden = workspace.hidden_dim
    device = workspace.device
    return (
        workspace.fits(shard)
        and weight.shape == (hidden,)
        and weight.dtype == torch.bfloat16
        and is_dense_on(weight, device, 16)
        and global_scale.numel() == 1
        and global_scale.dtype == torch.float32
        and is_dense_on(global_scale, device, 4)
    )


@flashinfer_api(trace=all_gather_rmsnorm_fp4_quantize_trace)
def all_gather_rmsnorm_fp4_quantize(
    shard: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    workspace: MNNVLCuteDSLAllGatherNormQuantWorkspace,
    *,
    eps: float,
    gain_center: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather column shards across TP, then RMSNorm and NVFP4-quantize the rows.

    With ``x = concat_r(shard[r], dim=1)``::

        y = bf16(x * rsqrt(mean(x**2) + eps) * (weight + gain_center))
        packed, scales = fp4_quantize(
            y, global_scale, sf_vec_size=16, is_sf_swizzled_layout=False
        )

    Args:
        shard: ``[M, H/TP]`` BF16 columns ``[rank*H/TP, (rank+1)*H/TP)``.
        weight: ``[H]`` BF16 RMSNorm weight.
        global_scale: ``[1]`` FP32 NVFP4 global scale.
        workspace: Workspace created on every rank of the TP group.
        eps: RMSNorm epsilon.
        gain_center: Constant added to ``weight``.

    Returns:
        ``(packed, scales)``: ``[M, H/2]`` uint8 E2M1 pairs and ``[M, H/16]``
        uint8 E4M3 block scales in linear layout.
    """
    if not supports_all_gather_rmsnorm_fp4_quantize(
        workspace, shard, weight, global_scale
    ):
        raise ValueError("unsupported all-gather RMSNorm FP4 quantize arguments")
    m = shard.shape[0]
    hidden = workspace.hidden_dim
    packed = torch.empty((m, hidden // 2), dtype=torch.uint8, device=shard.device)
    scales = torch.empty((m, hidden // 16), dtype=torch.uint8, device=shard.device)
    workspace.impl(
        shard,
        weight,
        global_scale.reshape(1),
        packed,
        scales,
        constants=_NormQuantConstants(float(eps), float(gain_center)),
    )
    return packed, scales
