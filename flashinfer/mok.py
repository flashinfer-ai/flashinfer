# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental adapter for Mixture of Kittens (MoK) MoE training kernels."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_mok_bf16(
    *, ep_size=16, local_experts=16, topk=8, clamped_swiglu=False, fp32_wgrad=False
):
    """Prepare the explicit Cake MoK training backend.

    Returns an adapter with ``build_schedule``, ``forward``,
    ``recompute_forward_context`` and ``backward`` methods using the public
    Mixture of Kittens functional signatures. Use
    :func:`create_mok_bf16_workspace` for caller-owned workspace and metadata.
    All compute uses the standalone CUDA sources shipped here.

    Routed experts run in BF16, or natively in MXFP8 when the routed weights
    are caller-prequantized MXFP8 tuples (see
    :func:`quantize_mok_mxfp8_weights`); the shared expert stays BF16.
    ``swiglu_limit`` (a per-call runtime value) selects clamped SwiGLU,
    ``silu(min(gate, L)) * clamp(up, -L, L)``, for routed and shared experts.
    ``clamped_swiglu`` and ``fp32_wgrad`` choose the kernels compiled now;
    call the adapter's ``prepare``/``prepare_recompute`` for any other
    variant before CUDA Graph capture. With ``fp32_wgrad=True``, ``backward``
    adds every weight gradient into caller-owned FP32 accumulators passed as
    ``weight_grad_accumulators``.

    Prepare and warm up outside graph capture. Supported scheduler layouts
    (EP, local experts) are (1, 4) and (4, 4) for diagnostics, 256 routed
    experts at EP 4/8/16/32/64 and 288 routed experts at EP 8/32, with top-k
    2 or 8. Tokens, hidden/intermediate widths and per-rank source counts
    remain runtime values subject to the documented alignment constraints.
    This is a synthetic training experiment, with no full-model or optimizer
    integration.
    """
    from .experimental.cake_mok_bf16.backend import MoKFunctional

    return MoKFunctional(
        ep_size,
        local_experts,
        topk,
        clamped_swiglu=clamped_swiglu,
        fp32_wgrad=fp32_wgrad,
    )


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
    The same workspace serves BF16 and MXFP8 routed experts.
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


@flashinfer_experimental_api
def quantize_mok_mxfp8_weights(gate, up, down):
    """Prequantize BF16 routed expert weights for native MXFP8 MoK training.

    ``gate``/``up`` are ``[E_local, I, H]`` and ``down`` is ``[E_local, H, I]``
    BF16 tensors. Uses the MoK MXFP8 recipe (32-element UE8M0 blocks along
    the reduction dimension, E4M3 data, tensor-core scale tiles) in both
    operand orientations. Returns ``(forward_weights, backward_weights)``:
    pass ``forward_weights`` as the routed gate/up/down arguments of
    ``forward`` (and its first two entries to ``recompute_forward_context``)
    and ``backward_weights`` to ``backward``. Requantize after every weight
    update; weight gradients are returned in BF16 (or FP32 accumulators).
    """
    from .experimental.cake_mok_bf16.mxfp8 import quantize_routed_weights

    return quantize_routed_weights(gate, up, down)
