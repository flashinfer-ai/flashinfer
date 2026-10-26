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

Fused row-wise FP8 GEMM + all-reduce.

    Y = sum_r (A_r @ W_r.T) * (a_scale_r[:, None] * w_scale_r[None, :])

Every rank holds its K-sharded FP8 (e4m3) operands ``A_r`` (``[M, K_l]``) and
``W_r`` (``[N, K_l]``) with fp32 row scales ``a_scale_r`` (``[M]``) and column
scales ``w_scale_r`` (``[N]``); the scales are applied in fp32 before the
cross-rank sum and every rank receives the full bfloat16 result ``[M, N]``.

Backends:
    - ``backend="cake"``: exact source-built SM100 implementation for four-rank
      NCCL groups (one fused kernel per call; see
      :mod:`.cake_fp8_rowwise_gemm_allreduce`).
    - ``backend="auto"``: the Cake backend (the only implementation).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from flashinfer.api_logging import flashinfer_api


@flashinfer_api
def fp8_rowwise_gemm_allreduce(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused row-wise FP8 GEMM + all-reduce over ``group``.

    ``backend="cake"`` selects the exact source-built SM100 fused kernel for
    four-rank NCCL groups; ``backend="auto"`` selects the same route (no other
    implementation exists). Unsupported inputs raise instead of falling back.

    Parameters
    ----------
    a : torch.Tensor
        This rank's contiguous FP8 (e4m3) activation shard of shape
        ``[M, K_l]``. ``a``, ``a_scale``, ``w``, and ``w_scale`` must be CUDA
        tensors on one shared device.
    a_scale : torch.Tensor
        Contiguous FP32 row scales of shape ``[M]``.
    w : torch.Tensor
        This rank's contiguous FP8 (e4m3) weight shard of shape ``[N, K_l]``.
    w_scale : torch.Tensor
        Contiguous FP32 column scales of shape ``[N]``.
    group : torch.distributed.ProcessGroup
        Four-rank NCCL process group on one SM100 node.
    backend : str, optional
        ``"auto"`` and ``"cake"`` both select the Cake backend.
    out : torch.Tensor, optional
        Contiguous BF16 output buffer of shape ``[M, N]``. A new tensor is
        allocated when omitted.

    Returns
    -------
    torch.Tensor
        The reduced BF16 product of shape ``[M, N]`` on every rank.
    """

    if backend == "cake":
        from .cake_fp8_rowwise_gemm_allreduce import fp8_rowwise_gemm_allreduce_cake

        return fp8_rowwise_gemm_allreduce_cake(
            a, a_scale, w, w_scale, group, backend="cake", out=out
        )
    if backend != "auto":
        raise ValueError("backend must be exactly 'auto' or 'cake'")
    from .cake_fp8_rowwise_gemm_allreduce import fp8_rowwise_gemm_allreduce_cake

    return fp8_rowwise_gemm_allreduce_cake(
        a, a_scale, w, w_scale, group, backend="cake", out=out
    )


@flashinfer_api
def prepare_fp8_rowwise_gemm_allreduce(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = "auto",
    max_rows: Optional[int] = None,
):
    """Bind ``group``, ``N``, ``K_l`` and a row capacity once; return a capturable launcher.

    The sample operands fix ``N`` and ``K_l``; ``max_rows`` (default: the rows
    of ``a``) sizes the symmetric workspace in this single collective and must
    be identical on every rank: the preparation checks it in that collective
    and raises ``ValueError`` on every rank when it differs. The returned
    ``launcher(a, a_scale, w, w_scale, *, out=None)`` serves any
    operands of the bound ``N`` and ``K_l`` with at most ``max_rows`` rows
    without another collective and is CUDA-graph capturable. Both
    ``backend="auto"`` and ``backend="cake"`` select the Cake route (SM100,
    four-rank NCCL groups, torch's default CUDA or NVSHMEM symmetric-memory
    backend). Unsupported inputs raise during preparation instead of falling
    back to another implementation.

    Parameters
    ----------
    a : torch.Tensor
        Sample contiguous FP8 (e4m3) activation shard of shape ``[M, K_l]``.
        ``a``, ``a_scale``, ``w``, and ``w_scale`` must be CUDA tensors on one
        shared device.
    a_scale : torch.Tensor
        Sample contiguous FP32 row scales of shape ``[M]``.
    w : torch.Tensor
        Sample contiguous FP8 (e4m3) weight shard of shape ``[N, K_l]``.
    w_scale : torch.Tensor
        Sample contiguous FP32 column scales of shape ``[N]``.
    group : torch.distributed.ProcessGroup
        Four-rank NCCL process group on one SM100 node.
    backend : str, optional
        ``"auto"`` and ``"cake"`` both select the Cake backend.
    max_rows : int, optional
        Maximum row count accepted by the returned launcher. Must be at least
        the sample input's row count. Defaults to the sample input's row count
        and must match on every rank.

    Returns
    -------
    Callable
        A CUDA-graph-capturable launcher with signature
        ``launcher(a, a_scale, w, w_scale, *, out=None)``.
    """

    if backend not in {"auto", "cake"}:
        raise ValueError("backend must be exactly 'auto' or 'cake'")
    from .cake_fp8_rowwise_gemm_allreduce import (
        prepare_fp8_rowwise_gemm_allreduce_cake,
    )

    return prepare_fp8_rowwise_gemm_allreduce_cake(
        a, a_scale, w, w_scale, group, backend="cake", max_rows=max_rows
    )


__all__ = ["fp8_rowwise_gemm_allreduce", "prepare_fp8_rowwise_gemm_allreduce"]
