# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental adapter for BF16 Mixture of Kittens training kernels."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_mok_bf16(*, ep_size=16, local_experts=16, topk=8, mxfp8=False):
    """Prepare the explicit Cake BF16 MoK toy training backend.

    Returns an adapter with ``build_schedule``, ``forward``,
    ``recompute_forward_context`` and ``backward`` methods using the public
    Mixture of Kittens functional signatures (``recompute_forward_context``
    rebuilds the backward context from ``x`` without the down projections,
    combine or output; it is bitwise identical to the saved context).
    Use :func:`create_mok_bf16_workspace` for caller-owned workspace and
    metadata. All compute uses the standalone CUDA sources shipped here.

    Prepare and warm up outside graph capture. Supported scheduler layouts
    (EP, local experts, top-k) are the toy layouts (1, 4, 2), (4, 4, 2),
    (16, 16, 8), (64, 4, 8), the 256-expert GLM-5.2 layouts (4, 64, 8),
    (8, 32, 8), (32, 8, 8) and the 288-expert GLM-5.3-Flash layouts
    (8, 36, 8), (32, 9, 8). Per-rank source counts are runtime values (any
    count within the workspace capacity, including zero); hidden and
    intermediate sizes must be multiples of 256. ``forward``/``backward``
    accept ``swiglu_limit`` (``None`` = plain SwiGLU, a positive float =
    clamped ``silu(min(gate, L)) * clamp(up, -L, L)`` for both experts).
    Routed expert weights may be MXFP8 tuples from :func:`mxfp8_quantize`
    (MoK's conventions: ``(w_fp8, w_sc)`` pairs for ``forward`` and
    ``recompute_forward_context``, the gate/up 4-tuples and the down
    ``(w_t_fp8, w_t_sc)`` pair for ``backward``), which selects the native
    MXFP8 kernels (block-scaled tcgen05 MMA with FP32 accumulation, fused
    activation/gradient quantization, BF16 shared experts); ``backward(...,
    wgrad_f32=True)`` returns FP32 routed weight gradients. ``mxfp8=True``
    precompiles those kernels; otherwise they compile on first use, outside
    CUDA Graph capture. This is a synthetic training experiment, with no
    full-model or optimizer integration.
    """
    from .experimental.cake_mok_bf16.backend import MoKFunctional

    return MoKFunctional(ep_size, local_experts, topk, mxfp8=mxfp8)


@flashinfer_experimental_api
def mxfp8_quantize(x_bf16, return_normal=True, return_transposed=True):
    """Quantize BF16 expert weights with MoK's MXFP8 recipe.

    Returns ``(x_fp8, x_sc, x_fp8_t, x_sc_t)``: E4M3 data ``[E, M, N]`` with
    E8M0 scale tiles ``[E * M / 128, N / 128, 32, 16]`` (one scale per 32-element
    block along the last axis; ``amax / 448`` rounded up to a power of two, floor
    ``1e-12``), plus the transposed layout ``[E, N, M]`` / ``[E * N / 128, M / 128,
    32, 16]``. Layouts not requested are ``None``; a 2-D input returns 2-D data.
    Dimensions must be multiples of 128. Prequantize the routed expert weights
    once per optimizer step and pass the tuples to the adapter from
    :func:`prepare_mok_bf16`. Compiles on first use per device (outside CUDA
    Graph capture).
    """
    from .experimental.cake_mok_bf16.backend import mxfp8_quantize as quantize

    return quantize(x_bf16, return_normal, return_transposed)


# Communication SMs (forward, backward) per precision: the splits at which the complete
# training step is fastest on B200 (EP4 GLM-5.2 shape sweep; backend README).
COMM_SMS_DEFAULTS = {"bf16": (24, 28), "mxfp8": (40, 40)}


@flashinfer_experimental_api
def create_mok_bf16_workspace(
    *,
    group,
    device,
    num_local_tokens,
    hidden_size,
    topk,
    source_capacity=None,
    fwd_num_comm_sms=None,
    bwd_num_comm_sms=None,
    minibatch_size=4096,
    macrobatch_size=32768,
    schedule_capacity_multiplier=3 / 16,
    precision="bf16",
):
    """Collectively create a caller-owned ``(config, workspace)`` pair.

    The process group must be initialized and its CUDA device current.
    Source input counts may differ across ranks, including zero. Common
    physical capacity is negotiated once; optional ``source_capacity``
    reserves future growth and must agree across ranks. Returned workspace
    exposes ``source_capacity`` and physical ``storage``. Defaults describe
    a small toy ring; see the backend README for capacity and reuse rules.
    Call outside CUDA Graph capture; retain the workspace for every replay.

    ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` left ``None`` take the measured
    default of ``precision`` (``"bf16"`` or ``"mxfp8"``): the BF16 kernels run
    their communication clusters on 24 (forward) / 28 (backward) SMs, the
    MXFP8 kernels on 40 / 40, which is where their steps are fastest on B200
    (see the backend README). The precision only selects these defaults; the
    kernels a call runs are chosen by the weights passed to it.
    """
    from .experimental.cake_mok_bf16.workspace import MoKConfig
    from .experimental.cake_mok_bf16.backend import create_source_workspace

    if precision not in COMM_SMS_DEFAULTS:
        raise ValueError(
            f"precision must be one of {sorted(COMM_SMS_DEFAULTS)}, got {precision!r}"
        )
    default_fwd, default_bwd = COMM_SMS_DEFAULTS[precision]
    if fwd_num_comm_sms is None:
        fwd_num_comm_sms = default_fwd
    if bwd_num_comm_sms is None:
        bwd_num_comm_sms = default_bwd
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
def context_defined_rows(config, context):
    """Rows of a forward or recomputed context that the kernels define.

    Returns ``(routed_rows, shared_rows)``: the routed rings hold the retained
    macrobatch, i.e. their first ``min(macrobatch_size, routed rows)`` rows, and
    the shared activations cover the real source rows. Rows beyond these are
    never written (allocator contents), so a bitwise comparison of a saved and
    a recomputed context must stop there. Synchronizes on the schedule's routed
    row count; call outside CUDA Graph capture.
    """
    from .experimental.cake_mok_bf16.backend import context_defined_rows as rows

    return rows(config, context)
