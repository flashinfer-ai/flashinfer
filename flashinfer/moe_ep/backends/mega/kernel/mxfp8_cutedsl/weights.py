"""Mega-path MXFP8 weight preprocessing for CuTeDSL MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    import torch

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]

Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2"]


def _mxfp8_data_dtype(kind: Mxfp8Kind) -> "torch.dtype":
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly); the
    # package import also bootstraps sys.path for the kernel packages.
    from .....kernel_src.cutedsl_megamoe import kind_data_dtype

    return kind_data_dtype(kind)


def _swizzle_expert_scales(raw_sf: "torch.Tensor") -> "torch.Tensor":
    from .....kernel_src.cutedsl_megamoe import to_blocked

    return to_blocked(raw_sf)


def _fc1_weight_from_w13(
    w13: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """(E, 2*I, H) gate||up bf16 -> (E, H, 2*I) kernel FC1 layout."""
    _, two_i, _ = w13.shape
    i = intermediate_size
    if two_i != 2 * i:
        raise ValueError(
            f"expected w13 with {2 * i} rows (gate||up), got shape {tuple(w13.shape)}"
        )
    return (
        _interleave_gate_up_32(w13, intermediate_size=2 * i)
        .transpose(1, 2)
        .contiguous()
    )


def _interleave_gate_up_32(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import Mxfp8BlockSize

    block = Mxfp8BlockSize
    if intermediate_size % (2 * block) != 0:
        raise ValueError(
            "MXFP8 MegaMOE requires full FC1 width to be divisible by "
            f"{2 * block}, got {intermediate_size}."
        )
    if tensor.shape[1] != intermediate_size:
        raise ValueError(
            f"expected FC1 tensor with shape (experts, {intermediate_size}, ...), "
            f"got {tuple(tensor.shape)}"
        )

    half = intermediate_size // 2
    gate = tensor[:, :half, :].contiguous()
    up = tensor[:, half:, :].contiguous()
    num_pairs = half // block
    out = tensor.new_empty(tensor.shape)
    out_view = out.view(tensor.shape[0], num_pairs, 2, block, tensor.shape[2])
    gate_view = gate.view(tensor.shape[0], num_pairs, block, tensor.shape[2])
    up_view = up.view(tensor.shape[0], num_pairs, block, tensor.shape[2])
    out_view[:, :, 0].copy_(gate_view)
    out_view[:, :, 1].copy_(up_view)
    return out.contiguous()


def _fc1_kernel_weight_from_canonical_mxfp8(
    w13: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    return (
        _interleave_gate_up_32(w13, intermediate_size=2 * intermediate_size)
        .transpose(1, 2)
        .contiguous()
    )


def _quantize_mxfp8_weight_k_major(
    weight_k_major: "torch.Tensor",
    *,
    kind: Mxfp8Kind,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize with K on the trailing dim; return K-major fp8 + plain E8M0 SF."""
    import torch

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import mxfp8_quantize_per_block_32

    data_dtype = _mxfp8_data_dtype(kind)
    return mxfp8_quantize_per_block_32(weight_k_major.to(torch.float32), data_dtype)


def _is_mxfp8_weight(weight: "torch.Tensor", *, kind: Mxfp8Kind) -> bool:
    return weight.dtype == _mxfp8_data_dtype(kind)


def _as_mxfp8_scale(scale: "torch.Tensor") -> "torch.Tensor":
    import torch

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import Mxfp8ScaleDtype

    if scale.dtype == Mxfp8ScaleDtype:
        return scale
    if scale.dtype == torch.uint8:
        return scale.view(Mxfp8ScaleDtype)
    raise ValueError(
        f"MXFP8 weight scales must have dtype {Mxfp8ScaleDtype} or torch.uint8, "
        f"got {scale.dtype}"
    )


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: float | None = None,
    activation_clamp: float | None = None,
) -> TransformedMegaWeights:
    """bf16 (or pre-quantized) weights → MXFP8 + swizzled-SF mega layout."""
    import torch

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly); the
    # shim exposes the cutlass-pulling stacking helper lazily via the boundary.
    from .....kernel_src.cutedsl_megamoe import (
        Mxfp8BlockSize,
        _stack_byte_reinterpretable_tensors,
        ceil_div,
    )

    del gate_up_clamp, activation_clamp  # MXFP8 weight quant uses a fixed 1.0 norm.

    fc1_out = 2 * intermediate_size
    num_experts = weights.w13.shape[0]
    data_dtype = _mxfp8_data_dtype(kind)

    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    kernel_fc1_shape = (num_experts, hidden_size, fc1_out)
    kernel_fc2_shape = (num_experts, intermediate_size, hidden_size)

    hidden_sf_cols = ceil_div(hidden_size, Mxfp8BlockSize)
    intermediate_sf_cols = ceil_div(intermediate_size, Mxfp8BlockSize)

    if weights.w13_scale is not None and weights.w2_scale is not None:
        w13_scale_in = _as_mxfp8_scale(weights.w13_scale)
        w2_scale_in = _as_mxfp8_scale(weights.w2_scale)
        expected_w13_scale_shape = (
            num_experts,
            fc1_out,
            hidden_sf_cols,
        )
        expected_w2_scale_shape = (
            num_experts,
            hidden_size,
            intermediate_sf_cols,
        )
        if w13_scale_in.shape != expected_w13_scale_shape:
            raise ValueError(
                f"w13_scale must have shape {expected_w13_scale_shape}, "
                f"got {tuple(w13_scale_in.shape)}"
            )
        if w2_scale_in.shape != expected_w2_scale_shape:
            raise ValueError(
                f"w2_scale must have shape {expected_w2_scale_shape}, "
                f"got {tuple(w2_scale_in.shape)}"
            )

        if (
            weights.w13.shape == kernel_fc1_shape
            and weights.w2.shape == kernel_fc2_shape
        ):
            if not _is_mxfp8_weight(weights.w13, kind=kind) or not _is_mxfp8_weight(
                weights.w2, kind=kind
            ):
                raise ValueError(
                    f"packed MXFP8 weights must have dtype {data_dtype}; got "
                    f"{weights.w13.dtype} / {weights.w2.dtype}"
                )
            fc1_weight = weights.w13
            fc2_weight = weights.w2
            w13_scale = w13_scale_in
        elif (
            weights.w13.shape == logical_w13_shape
            and weights.w2.shape == logical_w2_shape
        ):
            if not _is_mxfp8_weight(weights.w13, kind=kind) or not _is_mxfp8_weight(
                weights.w2, kind=kind
            ):
                raise ValueError(
                    f"packed MXFP8 weights must have dtype {data_dtype}; got "
                    f"{weights.w13.dtype} / {weights.w2.dtype}"
                )
            fc1_weight = _fc1_kernel_weight_from_canonical_mxfp8(
                weights.w13, intermediate_size=intermediate_size
            )
            fc2_weight = weights.w2.transpose(1, 2).contiguous()
            w13_scale = _interleave_gate_up_32(w13_scale_in, intermediate_size=fc1_out)
        else:
            raise ValueError(
                "pre-quantized MXFP8 weights must be in kernel layout "
                f"{kernel_fc1_shape} / {kernel_fc2_shape} or SGLang canonical "
                f"layout {logical_w13_shape} / {logical_w2_shape}; got "
                f"{tuple(weights.w13.shape)} / {tuple(weights.w2.shape)}"
            )
        fc1_sf_swizzled = [
            _swizzle_expert_scales(w13_scale[e]) for e in range(num_experts)
        ]
        fc2_sf_swizzled = [
            _swizzle_expert_scales(w2_scale_in[e]) for e in range(num_experts)
        ]
    else:
        if weights.w13.shape != logical_w13_shape:
            raise ValueError(
                f"w13 must have shape {logical_w13_shape}, "
                f"got {tuple(weights.w13.shape)}"
            )
        if weights.w2.shape != logical_w2_shape:
            raise ValueError(
                f"w2 must have shape {logical_w2_shape}, got {tuple(weights.w2.shape)}"
            )

        fc1_fp32 = _fc1_weight_from_w13(
            weights.w13, intermediate_size=intermediate_size
        )
        fc1_q_parts = []
        fc1_sf_parts = []
        fc2_q_parts = []
        fc2_sf_parts = []
        for expert in range(num_experts):
            fc1_q, fc1_sf = _quantize_mxfp8_weight_k_major(
                fc1_fp32[expert].transpose(0, 1),
                kind=kind,
            )
            fc2_hw = weights.w2[expert]
            fc2_q, fc2_sf = _quantize_mxfp8_weight_k_major(
                fc2_hw,
                kind=kind,
            )
            fc1_q_parts.append(fc1_q.transpose(0, 1))
            fc1_sf_parts.append(fc1_sf)
            fc2_q_parts.append(fc2_q.transpose(0, 1))
            fc2_sf_parts.append(fc2_sf)

        fc1_weight = torch.stack(fc1_q_parts, dim=0)
        fc2_weight = torch.stack(fc2_q_parts, dim=0)
        fc1_sf_swizzled = [_swizzle_expert_scales(sf) for sf in fc1_sf_parts]
        fc2_sf_swizzled = [_swizzle_expert_scales(sf) for sf in fc2_sf_parts]

    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_experts, fc1_flat_sf_size
    )
    fc2_weight_sf = _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0).view(
        num_experts, fc2_flat_sf_size
    )

    return (fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf)


def _mxfp8_swizzled_flat_sf_size(rows: int, cols: int) -> int:
    import torch

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import Mxfp8ScaleDtype, to_blocked

    plain = torch.zeros(rows, cols, dtype=Mxfp8ScaleDtype)
    return to_blocked(plain).numel()


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready MXFP8 weights (``preprocess_weights=False``)."""
    import torch

    from .....core.validation.common import MoEEpConfigError
    from ..weight_validation import (
        check_transformed_mega_weights_structure,
        check_transformed_weight_pair,
    )

    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )

    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    data_dtype = _mxfp8_data_dtype(kind)

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import Mxfp8BlockSize, ceil_div

    hidden_sf_cols = ceil_div(hidden_size, Mxfp8BlockSize)
    intermediate_sf_cols = ceil_div(intermediate_size, Mxfp8BlockSize)
    fc1_flat_sf = _mxfp8_swizzled_flat_sf_size(fc1_out, hidden_sf_cols)
    fc2_flat_sf = _mxfp8_swizzled_flat_sf_size(hidden_size, intermediate_sf_cols)

    check_transformed_mega_weights_structure(transformed)
    check_transformed_weight_pair(
        transformed[0],
        label="fc1",
        num_local_experts=local_experts,
        weight_dtype=data_dtype,
        expected_weight_shape=(local_experts, hidden_size, fc1_out),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc1_flat_sf),
    )
    check_transformed_weight_pair(
        transformed[1],
        label="fc2",
        num_local_experts=local_experts,
        weight_dtype=data_dtype,
        expected_weight_shape=(local_experts, intermediate_size, hidden_size),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc2_flat_sf),
    )


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
