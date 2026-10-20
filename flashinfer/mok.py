# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental adapter for Mixture of Kittens (MoK) MoE training kernels.

Thin public entry points of the experimental package
:mod:`flashinfer.experimental.cake_mok_bf16`: the three functions below carry
the public signatures, documentation and the experimental-API marking and
forward to :mod:`flashinfer.experimental.cake_mok_bf16.api`, which holds the
bodies, the measured communication-SM tables and their resolvers. Following the
experimental-track rules, this core module imports the package only inside the
functions (and lazily for the table and resolver names), never at import time.
"""

from .api_logging import flashinfer_experimental_api

# Tables and resolvers that live in the package but stay reachable here
# (``flashinfer.mok.COMM_SMS``, ``flashinfer.mok.resolve_comm_sms(...)``, ...).
_PACKAGE_NAMES = (
    "COMM_SMS",
    "IMBALANCE_COMM_SMS",
    "IMBALANCE_THRESHOLD",
    "IMBALANCE_THRESHOLDS",
    "MOK_PRECISIONS",
    "resolve_comm_sms",
    "resolve_imbalance_comm_sms",
    "resolve_imbalance_threshold",
)


def __getattr__(name):
    if name in _PACKAGE_NAMES:
        from .experimental.cake_mok_bf16 import api as _api

        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PACKAGE_NAMES))


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
    from .experimental.cake_mok_bf16 import api as _api

    return _api.prepare_mok_bf16(
        ep_size=ep_size,
        local_experts=local_experts,
        topk=topk,
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
    precision="bf16",
    imbalance_comm_sms="auto",
    imbalance_threshold="auto",
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

    ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` are the fixed (forward, backward)
    communication-SM counts of every launch (factory default 40 / 40). ``"auto"``
    for both takes the measured row of the device, ``precision`` and EP group
    size (:func:`resolve_comm_sms`, :data:`COMM_SMS`; GB300 MXFP8 at EP8: 44 / 36,
    no row elsewhere, so the factory split stays in force there).
    ``imbalance_comm_sms`` is the (forward, backward) communication-SM split the
    launches switch to while the hottest rank's routed-row load is at least
    ``imbalance_threshold`` times the mean per-rank load (observed on the
    gathered expert IDs of each step without a host synchronization; the counts
    are identical on every rank, each rank applies its own latest completed
    observation). ``"auto"`` (default) takes the measured row of the device
    for ``precision`` (:func:`resolve_imbalance_comm_sms`, keyed by compute
    capability and SM count; B200 BF16: 24 / 28, B200 MXFP8: 44 / 36, no row
    elsewhere - GB200 and GB300 included - so the fixed split stays in force
    there), ``None`` keeps the fixed split and a pair sets it. ``precision``
    only selects that row and the threshold. ``imbalance_threshold`` ``"auto"``
    (default) takes the measured ratio of ``precision``
    (:func:`resolve_imbalance_threshold`; :data:`IMBALANCE_THRESHOLDS`): BF16 4,
    between the 3x load ratio, where its split shows no gain, and 5.6x, where it
    gains 1-2 %; MXFP8 2.7, between 2.4x (no gain) and 3x (gains 1 %). A number
    sets the ratio. Balanced steps never switch.
    ``MoKConfig.comm_split_log`` (default ``False``; ``dataclasses.replace(config,
    comm_split_log=True)`` on the returned config) prints each split change per
    rank to stderr.
    """
    from .experimental.cake_mok_bf16 import api as _api

    return _api.create_mok_bf16_workspace(
        group=group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        source_capacity=source_capacity,
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=schedule_capacity_multiplier,
        precision=precision,
        imbalance_comm_sms=imbalance_comm_sms,
        imbalance_threshold=imbalance_threshold,
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
    from .experimental.cake_mok_bf16 import api as _api

    return _api.quantize_mok_mxfp8_weights(
        gate,
        up,
        down,
    )
