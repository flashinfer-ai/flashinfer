# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental adapter for BF16 Mixture of Kittens training kernels."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_mok_bf16(*, ep_size=16, local_experts=16, topk=8):
    """Prepare the explicit Cake BF16 MoK toy training backend.

    Returns an adapter with ``build_schedule``, ``forward`` and ``backward``
    methods using the public Mixture of Kittens functional signatures.
    Use :func:`create_mok_bf16_workspace` for caller-owned workspace and
    metadata. All compute uses the standalone CUDA sources shipped here.

    Prepare and warm up outside graph capture. Supported scheduler layouts
    are (EP, local experts, top-k) = (1, 4, 2), (4, 4, 2), (16, 16, 8), (64, 4, 8).
    Tokens and hidden/intermediate dimensions remain runtime values subject
    to the documented alignment constraints. This is a synthetic training
    experiment, with no full-model or optimizer integration.
    """
    from .experimental.cake_mok_bf16.backend import MoKFunctional

    return MoKFunctional(ep_size, local_experts, topk)


@flashinfer_experimental_api
def create_mok_bf16_workspace(
    *,
    group,
    device,
    num_local_tokens,
    hidden_size,
    topk,
    source_capacity=None,
    fwd_num_comm_sms=40,
    bwd_num_comm_sms=40,
    minibatch_size=4096,
    macrobatch_size=32768,
    schedule_capacity_multiplier=3 / 16,
):
    """Collectively create a caller-owned ``(config, workspace)`` pair.

    The process group must be initialized and its CUDA device current.
    Source input counts may differ across ranks, including zero. Common
    physical capacity is negotiated once; optional ``source_capacity``
    reserves future growth and must agree across ranks. Returned workspace
    exposes ``source_capacity`` and physical ``storage``. Defaults describe
    a small toy ring; see the backend README for capacity and reuse rules.
    Call outside CUDA Graph capture; retain the workspace for every replay.
    """
    from .experimental.cake_mok_bf16.workspace import MoKConfig
    from .experimental.cake_mok_bf16.backend import create_source_workspace

    config = MoKConfig(
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=schedule_capacity_multiplier,
    )
    return config, create_source_workspace(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        source_capacity=source_capacity,
    )
