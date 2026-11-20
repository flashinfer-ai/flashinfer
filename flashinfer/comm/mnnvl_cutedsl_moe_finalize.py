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

"""MNNVL MoE finalize with a per-expert RMSNorm, fused into the all-reduce.

For expert parallelism with replicated tokens: every rank holds the GEMM2
rows of its local experts in permuted order. Each rank RMS-normalizes every
local route's row over the full hidden width, accumulates the route-weighted
rows of each token, and the per-rank partial sums are all-reduced. The result
is the replicated ``[M, hidden]`` BF16 MoE output. Two kernels (LL publish with
the finalize arithmetic, then the LL reduction without a post-reduce norm)
over a dedicated LL mailbox.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .mnnvl_cutedsl.kernel_ll.moe_finalize_norm_all_reduce import (
    LLMoEFinalizeNormAllReduce,
)
from .mnnvl_cutedsl.runtime import (
    collective_support,
    is_dense_on,
    require_mnnvl_multicast_device,
)
from ..api_logging import flashinfer_api
from ..trace.templates.comm import (
    moe_finalize_rmsnorm_allreduce_trace,
)

__all__ = [
    "MNNVLCuteDSLMoEFinalizeAllReduceWorkspace",
    "moe_finalize_rmsnorm_allreduce",
    "supports_moe_finalize_rmsnorm_allreduce",
]


class MNNVLCuteDSLMoEFinalizeAllReduceWorkspace:
    """Mailbox, stage state and compiled kernels for one TP group.

    Construction is collective over ``group``. Compile every configuration
    with :meth:`compile` before CUDA graph capture. Calls must not overlap.
    Before selecting this path, pass each rank's local ``supports_*`` verdict
    to :meth:`check_support` on every rank. Local checks alone are not a
    rank-consistent fallback decision.
    """

    def __init__(
        self,
        tp_size: int,
        tp_rank: int,
        max_token_num: int,
        hidden_dim: int,
        top_k: int,
        *,
        group: ProcessGroup,
        enable_pdl: bool = True,
    ) -> None:
        if tp_size < 2:
            raise ValueError("tp_size must be at least 2")
        if max_token_num <= 0:
            raise ValueError("max_token_num must be positive")
        if hidden_dim <= 0 or hidden_dim % 1024:
            raise ValueError("hidden_dim must be a positive multiple of 1024")
        self.device = require_mnnvl_multicast_device()
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self._group = group
        self._impl: LLMoEFinalizeNormAllReduce | None = LLMoEFinalizeNormAllReduce(
            hidden_size=hidden_dim,
            topk=top_k,
            tp_size=tp_size,
            rank=tp_rank,
            capacity_m=max_token_num,
            group=group,
            enable_pdl=enable_pdl,
        )

    def check_support(self, local_supported: bool) -> bool:
        """Collectively agree on a local ``supports_*`` result before capture.

        Call on every rank during warmup, even when its local result is False.
        All ranks must use the agreed path and keep valid layouts for replay.
        """
        return collective_support(local_supported, self._group, self.device)

    @property
    def impl(self) -> LLMoEFinalizeNormAllReduce:
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
        row_stride: int,
        weight_dtype: torch.dtype,
        eps: float,
        gain_center: float = 0.0,
        weight_before_norm: bool = True,
    ) -> None:
        """Compile one configuration ahead of time, e.g. before graph capture."""
        self.impl.publisher(
            row_stride=row_stride,
            weight_dtype=weight_dtype,
            weight_before_norm=weight_before_norm,
            eps=float(eps),
            gain_center=float(gain_center),
        )


def supports_moe_finalize_rmsnorm_allreduce(
    workspace: MNNVLCuteDSLMoEFinalizeAllReduceWorkspace,
    routed_rows: torch.Tensor,
    route_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Check only this rank. Use ``workspace.check_support`` to agree across ranks."""
    hidden = workspace.hidden_dim
    device = workspace.device
    m = output.shape[0] if output.ndim == 2 else 0
    routes = (m, workspace.top_k)
    return (
        1 <= m <= workspace.max_token_num
        and output.shape[1] == hidden
        and output.dtype == torch.bfloat16
        and is_dense_on(output, device, 16)
        and routed_rows.ndim == 2
        and routed_rows.shape[1] >= hidden
        and routed_rows.shape[1] % 8 == 0
        and routed_rows.dtype == torch.bfloat16
        and is_dense_on(routed_rows, device, 16)
        and route_weights.shape == routes
        and route_weights.dtype in (torch.bfloat16, torch.float32)
        and is_dense_on(route_weights, device, route_weights.element_size())
        and permuted_idx.shape == routes
        and permuted_idx.dtype == torch.int32
        and is_dense_on(permuted_idx, device, 4)
        and norm_weight.shape == (hidden,)
        and norm_weight.dtype == torch.bfloat16
        and is_dense_on(norm_weight, device, 16)
    )


@flashinfer_api(trace=moe_finalize_rmsnorm_allreduce_trace)
def moe_finalize_rmsnorm_allreduce(
    routed_rows: torch.Tensor,
    route_weights: torch.Tensor,
    permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    workspace: MNNVLCuteDSLMoEFinalizeAllReduceWorkspace,
    *,
    eps: float,
    gain_center: float = 0.0,
    weight_before_norm: bool = True,
) -> None:
    """Per-expert RMSNorm finalize of local routes, all-reduced into ``output``.

    For token ``t`` and route ``k`` with local row ``x = routed_rows[p]``
    (``p = permuted_idx[t, k]``; routes with ``p < 0`` or
    ``p >= routed_rows.shape[0]`` belong to other ranks)::

        s = x * (norm_weight + gain_center)    # if weight_before_norm
        partial[t] = sum_k route_weights[t, k] * s * rsqrt(mean(s**2) + eps)
        output[t] = sum over ranks of bf16(partial[t])

    Without ``weight_before_norm`` the weight multiplies after the norm.

    Args:
        routed_rows: ``[rows, >=hidden]`` BF16 GEMM2 output in permuted order.
        route_weights: ``[M, top_k]`` BF16 or FP32 routing weights.
        permuted_idx: ``[M, top_k]`` int32 row of each route on this rank.
        norm_weight: ``[hidden]`` BF16 RMSNorm weight.
        output: ``[M, hidden]`` BF16 destination, identical on every rank.
        workspace: Workspace created on every rank of the group.
        eps: RMSNorm epsilon.
        gain_center: Constant added to ``norm_weight``.
        weight_before_norm: Whether the weight scales the row before the norm.
    """
    if not supports_moe_finalize_rmsnorm_allreduce(
        workspace, routed_rows, route_weights, permuted_idx, norm_weight, output
    ):
        raise ValueError("unsupported MoE finalize all-reduce arguments")
    workspace.impl(
        routed_rows,
        route_weights,
        permuted_idx,
        norm_weight,
        output,
        weight_before_norm=weight_before_norm,
        eps=float(eps),
        gain_center=float(gain_center),
    )
