"""Host preprocessing for SM90 Humming MXFP4 x FP8 MegaMoE weights.

The caller supplies canonical packed E2M1 payloads and raw K32 E8M0 exponent
bytes in :class:`PrequantizedMoEWeights`.  This module performs the one-time
Humming payload rewrite/fold and returns the four-slot weight ABI used by the
Hopper MegaMoE frontend.  It never materializes a persistent E4M3 weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch


HUMMING_MAX_EXP_RANGE = 11
HUMMING_EPILOGUE_COMPENSATION = 64.0
HUMMING_GROUP_SIZE = 32
HUMMING_FOLD_M = 64
HUMMING_FOLD_K = 128
MXFP4_GATE_UP_INTERLEAVE = 8
MXFP4_SETUP_EXPERT_CHUNK = 4

# Kernel-ready leg:
#   (packed_weight_K_major, folded_offset, unit_activation_placeholder,
#    per_expert_humming_residual_times_64)
TransformedWeightLeg = Tuple[
    "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"
]
TransformedMegaWeights = Tuple[TransformedWeightLeg, TransformedWeightLeg]


def _require_cuda_tensor(tensor: "torch.Tensor", *, label: str) -> None:
    from ......core.validation.common import MoEEpConfigError

    if not tensor.is_cuda:
        raise MoEEpConfigError(f"{label} must be a CUDA tensor")


def _interleave_gate_up_8(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """Canonical ``gate || up`` rows -> ``gate8, up8`` row groups.

    The helper is deliberately dtype-agnostic: it must be applied identically
    to FC1's packed payload and raw E8M0 scale plane before Humming.
    """
    if tensor.ndim != 3:
        raise ValueError(
            "FC1 gate/up tensor must be 3D (experts, 2*intermediate, cols); "
            f"got {tuple(tensor.shape)}"
        )
    if intermediate_size <= 0 or intermediate_size % MXFP4_GATE_UP_INTERLEAVE:
        raise ValueError(
            "intermediate_size must be a positive multiple of "
            f"{MXFP4_GATE_UP_INTERLEAVE}; got {intermediate_size}"
        )
    full_width = 2 * intermediate_size
    if tensor.shape[1] != full_width:
        raise ValueError(
            f"expected FC1 tensor with {full_width} rows, got {tuple(tensor.shape)}"
        )

    block = MXFP4_GATE_UP_INTERLEAVE
    experts, _, cols = tensor.shape
    gate = tensor[:, :intermediate_size, :].contiguous()
    up = tensor[:, intermediate_size:, :].contiguous()
    pairs = intermediate_size // block
    out = tensor.new_empty(tensor.shape)
    out_view = out.view(experts, pairs, 2, block, cols)
    out_view[:, :, 0].copy_(gate.view(experts, pairs, block, cols))
    out_view[:, :, 1].copy_(up.view(experts, pairs, block, cols))
    return out.contiguous()


def _validate_raw_prequantized_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> int:
    """Validate the canonical raw packed MXFP4/E8M0 production ABI."""
    import torch

    from ......core.validation.common import MoEEpConfigError

    if not isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            "sm90_fp8_mxfp4_bf16_pull_cutedsl requires PrequantizedMoEWeights "
            "with raw packed E2M1 payloads and both raw E8M0 scale planes; "
            "BF16 re-quantization and format fallback are not supported"
        )
    if hidden_size <= 0 or hidden_size % HUMMING_FOLD_K:
        raise MoEEpConfigError(
            f"hidden_size must be a positive multiple of {HUMMING_FOLD_K}; "
            f"got {hidden_size}"
        )
    if intermediate_size <= 0 or intermediate_size % HUMMING_FOLD_K:
        raise MoEEpConfigError(
            "intermediate_size must be a positive multiple of "
            f"{HUMMING_FOLD_K}; got {intermediate_size}"
        )

    raw_tensors = (
        ("w13", weights.w13),
        ("w13_scale", weights.w13_scale),
        ("w2", weights.w2),
        ("w2_scale", weights.w2_scale),
    )
    for name, tensor in raw_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise MoEEpConfigError(
                f"PrequantizedMoEWeights.{name} must be a torch.Tensor"
            )
        if tensor.ndim != 3:
            raise MoEEpConfigError(
                f"PrequantizedMoEWeights.{name} must be 3D; "
                f"got shape {tuple(tensor.shape)}"
            )

    num_experts = int(weights.w13.shape[0])
    if num_experts <= 0:
        raise MoEEpConfigError("raw MXFP4 weights must contain at least one expert")
    expected = {
        "w13": (num_experts, 2 * intermediate_size, hidden_size // 2),
        "w13_scale": (
            num_experts,
            2 * intermediate_size,
            hidden_size // HUMMING_GROUP_SIZE,
        ),
        "w2": (num_experts, hidden_size, intermediate_size // 2),
        "w2_scale": (
            num_experts,
            hidden_size,
            intermediate_size // HUMMING_GROUP_SIZE,
        ),
    }
    tensors = {
        "w13": weights.w13,
        "w13_scale": weights.w13_scale,
        "w2": weights.w2,
        "w2_scale": weights.w2_scale,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != torch.uint8:
            raise MoEEpConfigError(
                f"PrequantizedMoEWeights.{name} must have dtype torch.uint8; "
                f"got {tensor.dtype}"
            )
        if tuple(tensor.shape) != expected[name]:
            raise MoEEpConfigError(
                f"PrequantizedMoEWeights.{name} must have shape "
                f"{expected[name]}; got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"PrequantizedMoEWeights.{name} must be contiguous")
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise MoEEpConfigError(
            "all raw MXFP4 payload and E8M0 scale tensors must share one device"
        )
    for name, tensor in tensors.items():
        _require_cuda_tensor(tensor, label=f"PrequantizedMoEWeights.{name}")
    return num_experts


def _humming_preprocess(
    weight: "torch.Tensor",
    raw_scale: "torch.Tensor",
    *,
    max_range: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Lazy boundary to FlashInfer's existing, authoritative Humming path."""
    from flashinfer.fused_moe.prepare import (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming,
    )

    return preprocess_moe_weights_for_sm90_mixed_gemm_humming(
        weight,
        raw_scale,
        max_range=max_range,
        interleave=True,
    )


def _allocate_expert_output(part: "torch.Tensor", experts: int) -> "torch.Tensor":
    import torch

    return torch.empty((experts, *part.shape[1:]), dtype=part.dtype, device=part.device)


def _preprocess_humming_leg_chunked(
    weight: "torch.Tensor",
    raw_scale: "torch.Tensor",
    *,
    max_range: int = HUMMING_MAX_EXP_RANGE,
    expert_chunk_size: int = MXFP4_SETUP_EXPERT_CHUNK,
    gate_up_intermediate_size: int | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Run expert-independent Humming preprocessing with bounded temporaries.

    Chunking is allowed only along the expert dimension.  Splitting rows or K
    would change the per-expert exponent base/residual and is therefore not
    equivalent.
    """
    if expert_chunk_size <= 0:
        raise ValueError(f"expert_chunk_size must be positive, got {expert_chunk_size}")
    if max_range != HUMMING_MAX_EXP_RANGE:
        raise ValueError(
            f"the production Humming ABI fixes max_range=11; got {max_range}"
        )
    if weight.ndim != 3 or raw_scale.ndim != 3:
        raise ValueError("weight and raw_scale must both be 3D")
    experts = int(weight.shape[0])
    if experts <= 0 or raw_scale.shape[0] != experts:
        raise ValueError("weight and raw_scale must have the same nonzero expert dim")

    outputs: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"] | None = None
    for begin in range(0, experts, expert_chunk_size):
        end = min(begin + expert_chunk_size, experts)
        payload_part = weight[begin:end]
        scale_part = raw_scale[begin:end]
        if gate_up_intermediate_size is not None:
            # The payload and scale rows MUST undergo the identical mapping.
            payload_part = _interleave_gate_up_8(
                payload_part, intermediate_size=gate_up_intermediate_size
            )
            scale_part = _interleave_gate_up_8(
                scale_part, intermediate_size=gate_up_intermediate_size
            )
        parts = _humming_preprocess(
            payload_part,
            scale_part,
            max_range=max_range,
        )
        if not isinstance(parts, tuple) or len(parts) != 3:
            raise RuntimeError("Humming preprocessor must return a 3-tuple")
        if any(part.shape[0] != end - begin for part in parts):
            raise RuntimeError("Humming preprocessor changed the expert dimension")
        if outputs is None:
            outputs = tuple(_allocate_expert_output(part, experts) for part in parts)  # type: ignore[assignment]
        for output, part in zip(outputs, parts, strict=True):
            output[begin:end].copy_(part)

    assert outputs is not None
    return outputs


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    humming_max_range: int = HUMMING_MAX_EXP_RANGE,
    expert_chunk_size: int = MXFP4_SETUP_EXPERT_CHUNK,
) -> TransformedMegaWeights:
    """Raw canonical MXFP4/E8M0 -> kernel-ready Humming four-slot ABI."""
    import torch

    num_experts = _validate_raw_prequantized_weights(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    assert isinstance(weights, PrequantizedMoEWeights)

    fc1_processed, fc1_offset, fc1_residual = _preprocess_humming_leg_chunked(
        weights.w13,
        weights.w13_scale,
        max_range=humming_max_range,
        expert_chunk_size=expert_chunk_size,
        gate_up_intermediate_size=intermediate_size,
    )
    fc2_processed, fc2_offset, fc2_residual = _preprocess_humming_leg_chunked(
        weights.w2,
        weights.w2_scale,
        max_range=humming_max_range,
        expert_chunk_size=expert_chunk_size,
    )

    # Preserve storage-K stride 1.  Calling contiguous() after these transposes
    # would silently break the TMA/WGMMA packed-K contract.
    fc1_weight = fc1_processed.transpose(1, 2)
    fc2_weight = fc2_processed.transpose(1, 2)
    device = weights.w13.device
    fc1_act_placeholder = torch.ones((1,), dtype=torch.float32, device=device)
    fc2_act_placeholder = torch.ones((1,), dtype=torch.float32, device=device)
    fc1_weight_scale = (
        fc1_residual.to(torch.float32) * HUMMING_EPILOGUE_COMPENSATION
    ).contiguous()
    fc2_weight_scale = (
        fc2_residual.to(torch.float32) * HUMMING_EPILOGUE_COMPENSATION
    ).contiguous()

    transformed: TransformedMegaWeights = (
        (fc1_weight, fc1_offset, fc1_act_placeholder, fc1_weight_scale),
        (fc2_weight, fc2_offset, fc2_act_placeholder, fc2_weight_scale),
    )
    # Keep preprocessing and user-supplied transformed-weight validation on
    # exactly the same host contract.
    validate_transformed_mega_weights(
        transformed,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        world_size=1,
        num_experts=num_experts,
    )
    return transformed


def _check_leg_structure(transformed: object) -> None:
    from ......core.validation.common import MoEEpConfigError

    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError("transformed_weights must be a 2-tuple (fc1, fc2)")
    for index, leg in enumerate(transformed):
        label = "fc1" if index == 0 else "fc2"
        if not isinstance(leg, tuple) or len(leg) != 4:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be "
                "(weight, folded_offset, activation_placeholder, weight_scale)"
            )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """Validate kernel-ready packed-K weights and Humming metadata."""
    import torch

    from ......core.validation.common import MoEEpConfigError

    if world_size <= 0 or num_experts <= 0 or num_experts % world_size:
        raise MoEEpConfigError(
            "num_experts must be positive and divisible by world_size"
        )
    if (
        hidden_size <= 0
        or intermediate_size <= 0
        or hidden_size % HUMMING_FOLD_K
        or intermediate_size % HUMMING_FOLD_K
    ):
        raise MoEEpConfigError(
            "hidden_size and intermediate_size must be positive multiples of "
            f"{HUMMING_FOLD_K}"
        )
    _check_leg_structure(transformed)
    local_experts = num_experts // world_size
    expected = (
        (
            "fc1",
            (local_experts, hidden_size // 2, 2 * intermediate_size),
            (
                local_experts,
                (2 * intermediate_size) // HUMMING_FOLD_M,
                hidden_size // HUMMING_FOLD_K,
                16,
                16,
            ),
        ),
        (
            "fc2",
            (local_experts, intermediate_size // 2, hidden_size),
            (
                local_experts,
                hidden_size // HUMMING_FOLD_M,
                intermediate_size // HUMMING_FOLD_K,
                16,
                16,
            ),
        ),
    )

    all_tensors: list[torch.Tensor] = []
    for leg, (label, weight_shape, offset_shape) in zip(
        transformed, expected, strict=True
    ):
        weight, offset, activation_placeholder, weight_scale = leg
        for name, tensor in (
            ("weight", weight),
            ("folded_offset", offset),
            ("activation_placeholder", activation_placeholder),
            ("weight_scale", weight_scale),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise MoEEpConfigError(
                    f"transformed_weights {label} {name} must be a torch.Tensor"
                )
            all_tensors.append(tensor)
        if tuple(weight.shape) != weight_shape or weight.dtype != torch.uint8:
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be uint8 with shape "
                f"{weight_shape}; got {weight.dtype} {tuple(weight.shape)}"
            )
        if weight.stride(1) != 1:
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must have storage-K "
                f"stride 1 on dim 1; got strides {tuple(weight.stride())}"
            )
        if tuple(offset.shape) != offset_shape or offset.dtype != torch.uint8:
            raise MoEEpConfigError(
                f"transformed_weights {label} folded_offset must be uint8 with "
                f"shape {offset_shape}; got {offset.dtype} {tuple(offset.shape)}"
            )
        if not offset.is_contiguous() or offset.data_ptr() % 16:
            raise MoEEpConfigError(
                f"transformed_weights {label} folded_offset must be contiguous "
                "and 16-byte aligned"
            )
        if (
            tuple(activation_placeholder.shape) != (1,)
            or activation_placeholder.dtype != torch.float32
            or not activation_placeholder.is_contiguous()
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} activation_placeholder must be "
                "contiguous float32 with shape (1,)"
            )
        if (
            tuple(weight_scale.shape) != (local_experts,)
            or weight_scale.dtype != torch.float32
            or not weight_scale.is_contiguous()
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} weight_scale must be contiguous "
                f"float32 with shape ({local_experts},)"
            )
    if len({tensor.device for tensor in all_tensors}) != 1:
        raise MoEEpConfigError(
            "all transformed MXFP4 weights and scales must share one device"
        )
    for tensor in all_tensors:
        _require_cuda_tensor(
            tensor,
            label="all transformed MXFP4 weights and scales",
        )


__all__ = [
    "HUMMING_EPILOGUE_COMPENSATION",
    "HUMMING_FOLD_K",
    "HUMMING_FOLD_M",
    "HUMMING_GROUP_SIZE",
    "HUMMING_MAX_EXP_RANGE",
    "MXFP4_GATE_UP_INTERLEAVE",
    "MXFP4_SETUP_EXPERT_CHUNK",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
