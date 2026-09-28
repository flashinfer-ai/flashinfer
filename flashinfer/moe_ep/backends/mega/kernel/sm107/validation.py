"""Contracts shared by the Rubin block-scaled backends."""

from __future__ import annotations

import torch

from .....core.validation.common import MoEEpConfigError


def make_workspace_config(
    k, fleet_params, *, rank, world_size, quant_kind, gate_up_clamp, overrides
):
    from .....kernel_src.sm107.next_cutedsl_megamoe import (
        KNOB_KEYS,
        Sm107BlockScaledMoeConfig,
    )

    tuning = {name: getattr(k, name) for name in KNOB_KEYS if hasattr(k, name)}
    tuning["reduce_topk_in_kernel"] = k.in_kernel_fc2_reduce
    tuning.update(overrides)
    for name in (
        "mma_tiler_mnk",
        "cluster_shape_mn",
        "fallback_cluster_shape_mn",
        "fc2_tma_stages",
    ):
        if tuning.get(name) is None:
            tuning.pop(name, None)
    try:
        return Sm107BlockScaledMoeConfig(
            num_total_experts=fleet_params.num_experts,
            max_tokens_per_rank=fleet_params.max_tokens_per_rank,
            num_topk=k.top_k,
            hidden=fleet_params.token_hidden_size,
            intermediate=k.intermediate_size,
            rank=rank,
            world_size=world_size,
            quant_kind=quant_kind,
            gate_up_clamp=gate_up_clamp,
            apply_topk_at_fc1=k.apply_topk_in_fc1,
            max_sm_count=k.max_sm_count,
            **tuning,
        )
    except (TypeError, ValueError) as exc:
        raise MoEEpConfigError(str(exc)) from exc


def validate_unit_scalars(t) -> None:
    for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        if getattr(t, name) is not None:
            raise MoEEpConfigError(
                f"SM107 uses unit normalization and does not accept {name}; "
                "pre-quantize activations with norm_const=1 and omit this scalar."
            )


def validate_forward_metadata(
    hidden_states,
    topk_ids,
    topk_weights,
    fleet_params,
    *,
    top_k,
    quantize_input,
    quant_kind,
    scales=None,
) -> None:
    """Validate metadata without synchronizing the GPU or reading tensor values."""
    if hidden_states.ndim != 2:
        raise MoEEpConfigError("hidden_states must be 2D [num_tokens, hidden]")
    tokens, cols = hidden_states.shape
    hidden = fleet_params.token_hidden_size
    fp4 = quant_kind == "nvfp4"
    if quant_kind not in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
        raise MoEEpConfigError(f"unsupported quant_kind {quant_kind!r}")
    expected_cols = hidden // 2 if fp4 and not quantize_input else hidden
    if tokens > fleet_params.max_tokens_per_rank or cols != expected_cols:
        raise MoEEpConfigError(
            f"hidden_states must have at most {fleet_params.max_tokens_per_rank} "
            f"rows and {expected_cols} columns; got {tuple(hidden_states.shape)}"
        )
    data_dtype = (
        getattr(torch, "float4_e2m1fn_x2", torch.uint8)
        if fp4
        else torch.float8_e4m3fn
        if quant_kind == "mxfp8_e4m3"
        else torch.float8_e5m2
    )
    allowed = (
        (torch.bfloat16,)
        if quantize_input
        else ((data_dtype, torch.uint8) if fp4 else (data_dtype,))
    )
    if hidden_states.dtype not in allowed:
        raise MoEEpConfigError(f"hidden_states dtype must be one of {allowed}")
    if topk_ids.shape != (tokens, top_k) or topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError(
            f"topk_ids and topk_weights must have shape ({tokens}, {top_k})"
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise MoEEpConfigError("topk_ids must have dtype int32 or int64")
    if not topk_weights.is_floating_point():
        raise MoEEpConfigError("topk_weights must have a floating-point dtype")
    tensors = [hidden_states, topk_ids, topk_weights]
    if not quantize_input:
        if scales is None:
            raise MoEEpConfigError(
                "MoEEpTensors.scales is required when quantize_input=False"
            )
        sf_cols = hidden // (16 if fp4 else 32)
        padded_cols = (sf_cols + 3) // 4 * 4
        if (
            scales.ndim != 2
            or scales.shape[0] != tokens
            or not sf_cols <= scales.shape[1] <= padded_cols
        ):
            raise MoEEpConfigError(
                f"scales must have {tokens} rows and {sf_cols}..{padded_cols} columns"
            )
        expected_sf = torch.float8_e4m3fn if fp4 else torch.float8_e8m0fnu
        if scales.dtype not in (expected_sf, torch.uint8):
            raise MoEEpConfigError(
                f"scales must have dtype {expected_sf} or uint8 containing its raw bytes"
            )
        tensors.append(scales)
    if not hidden_states.is_cuda or any(
        t.device != hidden_states.device for t in tensors
    ):
        raise MoEEpConfigError(
            "all SM107 forward tensors must be on the same CUDA device"
        )
    if hidden_states.device.index != torch.cuda.current_device():
        raise MoEEpConfigError("SM107 inputs must be on the current CUDA device")


def validate_routing_values(topk_ids, topk_weights, num_experts: int) -> None:
    """Device assertions run before dispatch, including during graph replay.

    Duplicate valid routes violate the router's workspace capacity proof.
    -1 is a masked slot; all other IDs must be in range. A failed CUDA
    assertion invalidates that worker's CUDA context, as with invalid indices
    in other CUDA indexing operations.
    """
    ordered = topk_ids.sort(dim=1).values
    unique = (ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)
    valid = ((topk_ids >= -1) & (topk_ids < num_experts)).all()
    valid = valid & unique.all() & torch.isfinite(topk_weights).all()
    torch._assert_async(
        valid,
        "SM107 routes require unique expert IDs in [0, E), or -1, and finite scores",
    )


def validate_weight_layout(weight, scale, *, scale_dtype) -> None:
    if not weight.permute(0, 2, 1).is_contiguous():
        raise MoEEpConfigError(
            "SM107 weights must have contiguous physical [expert, output, K] storage (K stride 1)"
        )
    if scale.dtype != scale_dtype or not scale.is_contiguous():
        raise MoEEpConfigError(
            f"SM107 weight scales must be contiguous {scale_dtype}; reinterpret raw bytes explicitly"
        )
    if weight.device != scale.device:
        raise MoEEpConfigError("SM107 weights and scales must be on the same device")
    if weight.data_ptr() % 16 or scale.data_ptr() % 16:
        raise MoEEpConfigError("SM107 weights and scales must be 16-byte aligned")
