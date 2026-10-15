# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental adapter for Mixture of Kittens (MoK) MoE training kernels."""

from .api_logging import flashinfer_experimental_api

MOK_PRECISIONS = ("bf16", "mxfp8")

# Imbalance-aware communication split, keyed by device compute capability, SM
# count and routed precision. Only devices with a measured gain have a row;
# "auto" resolves to None (the fixed split) elsewhere. The SM count separates
# B200 (148 SMs) from GB200 (152 SMs, also compute capability (10, 0)), where
# the B200 BF16 row measured slower than the fixed split at 8x skew and above on
# EP16-64 (GB200 NVL72, FlashInfer PR #6439 review). B200 BF16 row 24 / 28: EP8
# complete training steps, two rounds on one node, step time against the fixed
# 20 / 24 on this package's backward programs. Hottest rank at 5.6x / 7.4x the
# mean routed load: -1.0 / -1.7 % and -1.2 / -1.3 %. Hottest rank at 3.0x (one
# rank's experts at twice the routing probability): +0.4 / +0.7 % (full step),
# +0.5 / +0.8 % (checkpoint step) and -0.1 ... +1.6 % with ragged source rows
# (8 cells, 7 slower) - no gain, within or just above the ~0.8 % spread of
# identical arms, so the fixed split is kept. The BF16 threshold sits between
# the measured 3.0x and 5.6x; the crossover in between is not measured, and
# with 4 ranks the ratio reaches 4 only when one rank carries every routed row.
# B200 MXFP8 row 44 / 36 against the fixed 40 / 32 (same protocol, EP8, two
# rounds on one node): hottest rank at 3.0x -1.0 % (full) / -1.0 % (checkpoint),
# at 5.6x -2.7 %; balanced steps +0.5 / +0.8 % (so they must not switch); with
# 4 ranks at 2.4x +0.8 % (one screening round). The MXFP8 threshold sits
# between the measured 2.4x (no gain) and 3.0x (gain). Balanced steps never
# switch, so the fixed split stays in force for them.
IMBALANCE_COMM_SMS = {
    (10, 0): {148: {"bf16": (24, 28), "mxfp8": (44, 36)}},
}
IMBALANCE_THRESHOLDS = {"bf16": 4.0, "mxfp8": 2.7}
# The BF16 threshold under the single-precision name of the first imbalance
# split revision; kept so callers written against that name keep working.
IMBALANCE_THRESHOLD = IMBALANCE_THRESHOLDS["bf16"]

# Fixed (forward, backward) communication-SM split for ``fwd_num_comm_sms="auto"``
# / ``bwd_num_comm_sms="auto"``, keyed by device compute capability, SM count,
# routed precision and EP group size. Only devices and group sizes with a
# measured gain have a row; "auto" resolves to the factory split (40 / 40)
# elsewhere, and the factory defaults themselves are unchanged (the row is
# opt-in). GB300 (compute capability (10, 3), 152 SMs) MXFP8 EP8 row 44 / 36
# against the fixed 40 / 32: EP8 complete training steps on two nodes (4 GPUs
# each), two rounds on the same nodes, hottest-rank step time. Hottest rank at
# 3.0x the mean routed load: -2.0 / -2.2 % (full step), -2.0 / -1.8 %
# (checkpoint step); at 5.6x: -3.1 / -3.2 %; at 7.4x: -3.2 / -3.3 %; balanced
# steps +0.7 / +0.5 % (full) and +0.1 / +0.2 % (checkpoint) - the balanced
# full step sits at the edge of the ~0.6 % spread of identical arms, which is
# why the row is opt-in rather than the default. GB300 EP4 (one node) measured
# no unconditional gain (balanced steps +0.7 ... +1.0 % slower) and has no row;
# B200 keeps its imbalance switch above.
COMM_SMS = {
    (10, 3): {152: {"mxfp8": {8: (44, 36)}}},
}
_FACTORY_COMM_SMS = (40, 40)


def resolve_imbalance_comm_sms(precision, device=None):
    """(forward, backward) communication-SM split used under imbalanced routing
    for ``precision`` (``"bf16"`` or ``"mxfp8"``) on ``device``, or ``None``
    when the device (compute capability and SM count) has no measured row (the
    fixed split is kept).
    """
    import torch

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return IMBALANCE_COMM_SMS.get(capability, {}).get(sm_count, {}).get(precision)


def resolve_comm_sms(precision, ep_size, device=None):
    """Fixed (forward, backward) communication-SM split measured for
    ``precision`` (``"bf16"`` or ``"mxfp8"``) at EP group size ``ep_size`` on
    ``device``, or ``None`` when the device (compute capability and SM count),
    precision or group size has no measured row (the factory split is kept).
    """
    import torch

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    ep_size = int(ep_size)
    if ep_size < 1:
        raise ValueError(f"ep_size must be a positive group size, got {ep_size}")
    if device is None:
        device = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(device))
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return (
        COMM_SMS.get(capability, {}).get(sm_count, {}).get(precision, {}).get(ep_size)
    )


def resolve_imbalance_threshold(precision):
    """Load ratio (hottest rank over the mean per-rank routed load) from which
    the imbalance split of ``precision`` (``"bf16"`` or ``"mxfp8"``) applies.
    """
    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    return IMBALANCE_THRESHOLDS[precision]


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
    ``CAKE_MOK_COMM_SPLIT_LOG=1`` prints each split change.
    """
    from .experimental.cake_mok_bf16.workspace import MoKConfig
    from .experimental.cake_mok_bf16.backend import create_source_workspace

    if precision not in MOK_PRECISIONS:
        raise ValueError(
            f"precision must be one of {MOK_PRECISIONS}, got {precision!r}"
        )
    auto = (fwd_num_comm_sms == "auto", bwd_num_comm_sms == "auto")
    if any(auto):
        if not all(auto):
            raise ValueError(
                'fwd_num_comm_sms and bwd_num_comm_sms must both be "auto" or both '
                "be communication-SM counts"
            )
        import torch.distributed as dist

        fwd_num_comm_sms, bwd_num_comm_sms = (
            resolve_comm_sms(precision, dist.get_world_size(group), device)
            or _FACTORY_COMM_SMS
        )
    if isinstance(imbalance_comm_sms, str):
        if imbalance_comm_sms != "auto":
            raise ValueError(
                'imbalance_comm_sms must be "auto", None or a (forward, backward) pair'
            )
        imbalance_comm_sms = resolve_imbalance_comm_sms(precision, device)
    elif imbalance_comm_sms is not None:
        imbalance_comm_sms = tuple(imbalance_comm_sms)
        if len(imbalance_comm_sms) != 2:
            raise ValueError(
                "imbalance_comm_sms must be a (forward, backward) pair of counts"
            )
    if isinstance(imbalance_threshold, str):
        if imbalance_threshold != "auto":
            raise ValueError('imbalance_threshold must be "auto" or a number')
        imbalance_threshold = resolve_imbalance_threshold(precision)
    config = MoKConfig(
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        imbalance_comm_sms=imbalance_comm_sms,
        imbalance_threshold=imbalance_threshold,
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
