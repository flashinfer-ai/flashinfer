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


def _require_cutedsl_paths() -> None:
    from ..cutedsl_backend_kernels import bootstrap_paths

    bootstrap_paths()


def _mxfp8_data_dtype(kind: Mxfp8Kind) -> "torch.dtype":
    _require_cutedsl_paths()
    from common.host_utils import kind_data_dtype

    return kind_data_dtype(kind)


def _swizzle_expert_scales(raw_sf: "torch.Tensor") -> "torch.Tensor":
    _require_cutedsl_paths()
    from moe_nvfp4_swapab.runner_common import to_blocked

    return to_blocked(raw_sf)


def _fc1_weight_from_w13(
    w13: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """(E, 2*I, H) gate||up bf16 → (E, H, I) with 32-wide gate/up column interleave."""
    _require_cutedsl_paths()
    from common.megamoe_constants import Mxfp8BlockSize

    block = Mxfp8BlockSize
    num_experts, two_i, hidden = w13.shape
    i = intermediate_size
    if two_i != 2 * i:
        raise ValueError(
            f"expected w13 with {2 * i} rows (gate||up), got shape {tuple(w13.shape)}"
        )
    if i % (2 * block) != 0:
        raise ValueError(
            "MXFP8 MegaMOE requires intermediate_size to be divisible by "
            f"{2 * block} (gate/up pairs at block={block}), got {intermediate_size}."
        )

    gate = w13[:, :i, :]
    up = w13[:, i:, :]
    num_pairs = i // (2 * block)
    out = w13.new_empty(num_experts, hidden, i)
    for pair in range(num_pairs):
        row_base = pair * block
        col_base = pair * (2 * block)
        g_block = gate[:, row_base : row_base + block, :].transpose(1, 2)
        u_block = up[:, row_base : row_base + block, :].transpose(1, 2)
        out[:, :, col_base : col_base + block] = g_block
        out[:, :, col_base + block : col_base + 2 * block] = u_block
    return out


def _quantize_mxfp8_weight_k_major(
    weight_k_major: "torch.Tensor",
    *,
    kind: Mxfp8Kind,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize with K on the trailing dim; return K-major fp8 + plain E8M0 SF."""
    import torch

    _require_cutedsl_paths()
    from common.host_utils import mxfp8_quantize_per_block_32

    data_dtype = _mxfp8_data_dtype(kind)
    return mxfp8_quantize_per_block_32(weight_k_major.to(torch.float32), data_dtype)


def _is_mxfp8_weight(weight: "torch.Tensor", *, kind: Mxfp8Kind) -> bool:
    return weight.dtype == _mxfp8_data_dtype(kind)


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

    _require_cutedsl_paths()
    from common.megamoe_constants import Mxfp8BlockSize
    from moe_nvfp4_swapab.mega_runner import _stack_byte_reinterpretable_tensors
    from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype, ceil_div

    del gate_up_clamp, activation_clamp  # MXFP8 weight quant uses a fixed 1.0 norm.

    fc1_out = 2 * intermediate_size
    i_down = intermediate_size // 2
    num_experts = weights.w13.shape[0]
    data_dtype = _mxfp8_data_dtype(kind)

    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    kernel_fc1_shape = (num_experts, hidden_size, intermediate_size)
    kernel_fc2_shape = (num_experts, i_down, hidden_size)

    hidden_sf_cols = ceil_div(hidden_size, Mxfp8BlockSize)
    i_down_sf_cols = ceil_div(i_down, Mxfp8BlockSize)

    if weights.w13_scale is not None and weights.w2_scale is not None:
        if weights.w13.shape != kernel_fc1_shape or weights.w2.shape != kernel_fc2_shape:
            raise ValueError(
                "pre-quantized MXFP8 weights must be in kernel layout "
                f"{kernel_fc1_shape} / {kernel_fc2_shape}; got "
                f"{tuple(weights.w13.shape)} / {tuple(weights.w2.shape)}"
            )
        expected_w13_scale_shape = (
            num_experts,
            intermediate_size,
            hidden_sf_cols,
        )
        expected_w2_scale_shape = (
            num_experts,
            hidden_size,
            i_down_sf_cols,
        )
        if weights.w13_scale.shape != expected_w13_scale_shape:
            raise ValueError(
                f"w13_scale must have shape {expected_w13_scale_shape}, "
                f"got {tuple(weights.w13_scale.shape)}"
            )
        if weights.w2_scale.shape != expected_w2_scale_shape:
            raise ValueError(
                f"w2_scale must have shape {expected_w2_scale_shape}, "
                f"got {tuple(weights.w2_scale.shape)}"
            )
        if not _is_mxfp8_weight(weights.w13, kind=kind) or not _is_mxfp8_weight(
            weights.w2, kind=kind
        ):
            raise ValueError(
                f"packed MXFP8 weights must have dtype {data_dtype}; got "
                f"{weights.w13.dtype} / {weights.w2.dtype}"
            )
        if weights.w13_scale.dtype != Mxfp8ScaleDtype or weights.w2_scale.dtype != Mxfp8ScaleDtype:
            raise ValueError(
                f"MXFP8 weight scales must have dtype {Mxfp8ScaleDtype}"
            )
        fc1_weight = weights.w13
        fc2_weight = weights.w2
        fc1_sf_swizzled = [
            _swizzle_expert_scales(weights.w13_scale[e]) for e in range(num_experts)
        ]
        fc2_sf_swizzled = [
            _swizzle_expert_scales(weights.w2_scale[e]) for e in range(num_experts)
        ]
    else:
        if weights.w13.shape != logical_w13_shape:
            raise ValueError(
                f"w13 must have shape {logical_w13_shape}, "
                f"got {tuple(weights.w13.shape)}"
            )
        if weights.w2.shape != logical_w2_shape:
            raise ValueError(
                f"w2 must have shape {logical_w2_shape}, "
                f"got {tuple(weights.w2.shape)}"
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
            fc2_hw = weights.w2[expert, :, :i_down]
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


__all__ = ["MoEWeightPack", "TransformedMegaWeights", "preprocess_mega_weights"]
