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
"""

from typing import Any, Optional

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; the workspace, route selection (one-shot /
# two-shot K1, fused K23 / K3-ESS / persistent tail, grouped / pinned poll
# schedule), launch binding and JIT registration live in
# flashinfer.experimental.kimi_k3_tp12_tail.

_FEATURE = "Kimi-K3 TP12 fused LatentMoE tail"


def _backend(backend: str):
    if backend != "cake":
        raise ValueError("the Kimi-K3 TP12 tail currently supports backend='cake'")
    from .experimental.kimi_k3_tp12_tail import cake_backend

    return cake_backend


@flashinfer_experimental_api(feature=_FEATURE)
def create_kimi_k3_tp12_tail_workspace(
    *,
    rank: int,
    max_tokens: int = 4096,
    group: Any = None,
    device: Optional[torch.device] = None,
    comm_backend: Any = None,
    backend: str = "cake",
) -> Any:
    r"""Create the twelve-rank workspace of the Kimi-K3 TP12 fused LatentMoE tail.

    Collective: every rank of the twelve-rank ``group`` must call it.  It
    allocates FlashInfer's MNNVL Lamport workspaces (CUDA fabric symmetric
    memory with a multicast mapping; three rotating buffers per stage) sized
    for up to ``max_tokens`` tokens plus the normalised-latent and
    up-projection slice buffers of this rank.  Reuse the workspace for every
    call of the layer and ``destroy()`` it before the process group.

    Parameters
    ----------
    rank : int
        This rank in ``[0, 12)``; must equal the process group rank.
    max_tokens : int
        Largest ``M`` the workspace serves (default 4096).
    group : torch.distributed.ProcessGroup, optional
        The twelve-rank group (default: the world group).  Used for the handle
        exchange and the creation barriers only; the kernels never call NCCL.
    device : torch.device, optional
        CUDA device of this rank (default: the current device).
    comm_backend : optional
        An explicit FlashInfer ``CommBackend`` for the handle exchange instead
        of ``TorchDistBackend(group)``.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3Tp12TailWorkspace
    """
    return _backend(backend).KimiK3Tp12TailWorkspace(
        rank=rank,
        max_tokens=max_tokens,
        group=group,
        device=device,
        comm_backend=comm_backend,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
    backend: str = "cake",
) -> Any:
    r"""Prepare the Kimi-K3 TP12 fused LatentMoE tail of one rank for repeated launches.

    The experimental generated-program backend computes, replicated on the
    twelve ranks of a GB200 / GB300 NVL72 multi-node NVLink domain::

        latent = KimiRMSNorm(sum_r routed_partial_r)                  [M, 3584]  (eps 1e-5)
        out    = BF16(latent @ up_weight.T + sum_r shared_partial_r)  [M, 7168]

    in two generated launches per rank, with cuBLAS between them for ``M > 4``:
    a Lamport all-reduce of ``routed_partial`` fused with the norm (one-shot
    for ``M <= 16``, two-shot above; below 256 tokens it also scatters this
    rank's columns of ``shared_partial`` to their owner ranks), the
    up-projection of this rank's contiguous 640- or 512-row slice of
    ``up_weight`` (``7168 = 8 x 640 + 4 x 512``; fused into the tail kernel for
    ``M <= 4``, cuBLAS above), and the tail: the owner reduce of the
    ``shared_partial`` columns fused with the add of the slice, one BF16
    rounding and the multicast all-gather of ``out`` (one CTA per eight output
    columns for ``M <= 4``, one CTA per token below 256 tokens, a persistent
    token pipeline above).  ``out`` is bitwise identical on every rank.
    Every rank must prepare and launch the same ``M``.

    Parameters
    ----------
    routed_partial : torch.Tensor
        Contiguous BF16 ``[M, 3584]`` routed-expert partial sum of this rank.
    shared_partial : torch.Tensor
        Contiguous BF16 ``[M, 7168]`` shared-expert partial of this rank.
    norm_weight : torch.Tensor
        Contiguous BF16 ``[3584]`` ``routed_expert_norm`` weight (replicated).
    up_weight : torch.Tensor
        Contiguous BF16 ``[7168, 3584]`` ``routed_expert_up_proj`` weight
        (replicated; the rank's slice is a row-slice view, no copy).
    out : torch.Tensor
        Caller-owned BF16 ``[M, 7168]`` output.
    workspace : KimiK3Tp12TailWorkspace
        From :func:`create_kimi_k3_tp12_tail_workspace`; ``M <= max_tokens``.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3Tp12TailRunner
        Calling it launches the two generated kernels (and cuBLAS for ``M > 4``)
        on the current stream with no CUDA allocation or host synchronization
        and returns ``out``.  CUDA Graph capture of the runner is supported;
        prepare outside capture.
    """
    return _backend(backend).prepare_kimi_k3_tp12_tail(
        routed_partial, shared_partial, norm_weight, up_weight, out, workspace=workspace
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
    backend: str = "cake",
) -> torch.Tensor:
    r"""Run the Kimi-K3 TP12 fused LatentMoE tail of one rank once.

    Equivalent to :func:`prepare_kimi_k3_tp12_tail` followed by one launch;
    returns ``out``.
    """
    return _backend(backend).prepare_kimi_k3_tp12_tail(
        routed_partial, shared_partial, norm_weight, up_weight, out, workspace=workspace
    )()


__all__ = [
    "create_kimi_k3_tp12_tail_workspace",
    "kimi_k3_tp12_tail",
    "prepare_kimi_k3_tp12_tail",
]
