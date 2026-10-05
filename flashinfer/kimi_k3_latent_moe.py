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

"""Experimental Kimi-K3 Stable LatentMoE front / tail projections on SM100 / SM103."""

from typing import Any

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; the host planner (decode / prefill route,
# ring depth, cluster pairs, stream-K plan, PDL trigger), launch binding and
# JIT registration live in flashinfer.experimental.kimi_k3_latent_moe.

_FEATURE = "Kimi-K3 LatentMoE front/tail"


def _backend(backend: str):
    if backend != "cake":
        raise ValueError(
            "the Kimi-K3 LatentMoE projections currently support backend='cake'"
        )
    from .experimental.kimi_k3_latent_moe import cake_backend

    return cake_backend


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
    *,
    backend: str = "cake",
) -> Any:
    r"""Prepare the Kimi-K3 LatentMoE front projections of one rank for repeated launches.

    The experimental generated-program backend computes, in one launch for
    ``T <= 128`` tokens (weight-streaming tcgen05 kernel) or one persistent
    2-CTA GEMM launch for ``T > 128``::

        logits     = FP32(x @ gate_weight.T)                          [T, 896]
        latent     = BF16(x @ down_weight.T)                          [T, 3584]
        shared_act = SiTU(x @ shared_gate.T, x @ shared_up.T)         [T, 6144 / TP]

    with ``SiTU(g, u) = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`` in FP32 and one
    BF16 rounding (``nvidia/Kimi-K3-NVFP4`` ``KimiSparseMoeBlock`` semantics).

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 ``[T, 7168]`` hidden states.
    gate_weight : torch.Tensor
        Contiguous BF16 ``[896, 7168]`` router gate weight (replicated).
    down_weight : torch.Tensor
        Contiguous BF16 ``[3584, 7168]`` ``routed_expert_down_proj`` weight (replicated).
    shared_gate_up_weight : torch.Tensor
        Contiguous BF16 ``[2 * 6144 / TP, 7168]``: this rank's shared-expert gate rows
        followed by its up rows (``MergedColumnParallel`` shard).  TP is 1 or 8.
    logits, latent, shared_act : torch.Tensor
        Caller-owned outputs: FP32 ``[T, 896]``, BF16 ``[T, 3584]``, BF16 ``[T, 6144 / TP]``.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3LatentMoeRunner
        Calling it launches the route on the current stream with no CUDA
        allocation or host synchronization and returns the three outputs.  CUDA
        Graph capture of the runner is supported; prepare outside capture.
    """
    return _backend(backend).prepare_kimi_k3_latent_moe_front(
        x, gate_weight, down_weight, shared_gate_up_weight, logits, latent, shared_act
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
    *,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Run the Kimi-K3 LatentMoE front projections of one rank once.

    Equivalent to :func:`prepare_kimi_k3_latent_moe_front` followed by one
    launch; returns ``(logits, latent, shared_act)``.
    """
    return _backend(backend).prepare_kimi_k3_latent_moe_front(
        x, gate_weight, down_weight, shared_gate_up_weight, logits, latent, shared_act
    )()


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
    backend: str = "cake",
) -> Any:
    r"""Prepare the Kimi-K3 LatentMoE tail of one rank for repeated launches.

    The experimental generated-program backend computes::

        y   = KimiRMSNorm(sum_p routed[p])                                  [T, 3584]  (eps 1e-5)
        out = BF16(y[:, cols] @ up_weight[:, cols].T + shared_act @ shared_down_weight.T)   [T, 7168]

    where ``cols`` is this rank's ``3584 / tp`` latent slice; with ``tp > 1`` the caller
    all-reduces ``out`` afterwards.  For ``T <= 128`` the norm is fused into one
    weight-streaming tcgen05 launch; for ``T > 128`` a one-pass RMSNorm kernel precedes a
    persistent 2-CTA GEMM launched programmatic-dependent.

    Parameters
    ----------
    routed : torch.Tensor
        Contiguous BF16 ``[P, T, 3584]``: ``P`` un-reduced routed-expert partials
        (summed on device in FP32 and rounded once).
    norm_weight : torch.Tensor
        Contiguous BF16 ``[3584]`` ``routed_expert_norm`` weight.
    up_weight : torch.Tensor
        Contiguous BF16 ``[7168, 3584]`` ``routed_expert_up_proj`` weight (replicated).
    shared_act : torch.Tensor
        Contiguous BF16 ``[T, 6144 / tp]`` shared-expert SiTU activation of this rank.
    shared_down_weight : torch.Tensor
        Contiguous BF16 ``[7168, 6144 / tp]`` ``RowParallel`` shard of the shared down weight.
    out : torch.Tensor
        Caller-owned BF16 ``[T, 7168]`` output (this rank's partial when ``tp > 1``).
    tp, rank : int
        Tensor-parallel degree (1 or 8) and this rank.
    y_workspace : torch.Tensor
        Caller-owned BF16 ``[T, 3584]`` buffer receiving the normalised latent.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3LatentMoeRunner
        Calling it launches the route on the current stream with no CUDA
        allocation or host synchronization and returns ``(y_workspace, out)``.
        CUDA Graph capture of the runner is supported; prepare outside capture.
    """
    return _backend(backend).prepare_kimi_k3_latent_moe_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
    backend: str = "cake",
) -> torch.Tensor:
    r"""Run the Kimi-K3 LatentMoE tail of one rank once.

    Equivalent to :func:`prepare_kimi_k3_latent_moe_tail` followed by one
    launch; returns ``out``.
    """
    _backend(backend).prepare_kimi_k3_latent_moe_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
    )()
    return out


__all__ = [
    "kimi_k3_latent_moe_front",
    "kimi_k3_latent_moe_tail",
    "prepare_kimi_k3_latent_moe_front",
    "prepare_kimi_k3_latent_moe_tail",
]
