"""Mega-path FP8 weight preprocessing for the SM90 pull-style CuTeDSL kernel.

Weight quantization follows the kernel's dequant convention (verified against
``moe_hopper_fp8/mega_reference_fp8.py`` and the drop's
``MegaMoEFp8Tester.generate_inputs``): ``fp32 ~= fp8_payload * scale``.

* ``per_tensor``: one fp32 scalar per expert weight,
  ``scale[e] = absmax(w[e]) / fp8_max``; the kernel computes
  ``fc_out = raw_fp8_gemm * activation_scale * weight_scale[e]``.  The E8M0
  weight-SF planes are unit placeholders in the swizzled flat layout the TMA
  SFA descriptor expects (dispatched wire format; unused by dequantization).
* ``blockwise``: DeepGEMM-style fp32 scales per 128x128 ``(N, K)`` weight
  block (``quantize_fp8_weight_block_nk``); the per-tensor dequant-scale slots
  ride as ``None`` (the shim substitutes cached unit tensors — the kernel ABI
  still takes them but ignores their values).

FC1 gate/up interleave granularity is ``Fp8GateUpInterleave = 8`` (the SM90
kernel's PostSwigluHalf fold), NOT the SM100 MXFP8 tree's 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple

from .....weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch

# Kernel-ready weight leg: (weight, weight_sf, activation_dequant_scale,
# weight_dequant_scale).  The two dequant-scale slots are real (1,) / (E,)
# fp32 tensors in per_tensor mode and None in blockwise mode.
TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor | None", "torch.Tensor | None"],
    Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor | None", "torch.Tensor | None"],
]

Sm90Fp8Kind = Literal["fp8_e4m3", "fp8_e5m2"]
Sm90Fp8ScaleMode = Literal["per_tensor", "blockwise"]


def _fp8_data_dtype(kind: Sm90Fp8Kind) -> "torch.dtype":
    # Backend talks only to the pull_style_cutedsl_megakernel shim (never src/
    # directly); the package import also bootstraps sys.path for the kernel
    # packages.
    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import kind_data_dtype

    return kind_data_dtype(kind)


def _interleave_gate_up_8(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """(E, 2*I, ...) gate||up halves -> 8-row gate/up interleave.

    The SM90 FP8 kernel's SwiGLU epilogue folds FC1 output columns as
    ``[pair, {gate, up}, 8]`` (``Fp8GateUpInterleave``), so the FC1 weight's N
    axis must carry gate/up interleaved in blocks of 8 rows.
    """
    # Backend talks only to the pull_style_cutedsl_megakernel shim boundary.
    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import Fp8GateUpInterleave

    block = Fp8GateUpInterleave
    if intermediate_size % (2 * block) != 0:
        raise ValueError(
            "SM90 FP8 MegaMoE requires full FC1 width to be divisible by "
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


def _quantize_fp8_per_tensor_expert(
    weight_nk: "torch.Tensor",
    *,
    data_dtype: "torch.dtype",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One-expert per-tensor quant: ``fp32 ~= payload * scale`` convention.

    ``scale = absmax / fp8_max`` (no headroom margin: weights are quantized
    once, not re-quantized in a chain like the fc1-out activation).
    """
    import torch

    fp32 = weight_nk.to(torch.float32)
    fp8_max = float(torch.finfo(data_dtype).max)
    absmax = fp32.abs().amax()
    scale = (absmax / fp8_max).clamp_min(1.0e-12)
    payload = (fp32 / scale).to(data_dtype)
    return payload, scale.reshape(1)


def _swizzle_unit_e8m0_sf(
    num_experts: int, rows: int, cols: int, device
) -> "torch.Tensor":
    """Unit E8M0 SF planes in the swizzled flat (E, flat) placeholder layout.

    Matches the drop driver's per-tensor weight-SF assembly (unit values,
    atom-swizzled via ``to_blocked``): the plane is a TMA-descriptor
    placeholder the GEMM dequantization never reads.
    """
    import torch

    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
        _stack_byte_reinterpretable_tensors,
        to_blocked,
    )

    plane = torch.ones(rows, cols, dtype=torch.float8_e8m0fnu, device=device)
    swizzled = to_blocked(plane)
    return _stack_byte_reinterpretable_tensors(
        [swizzled] * num_experts, dim=0
    ).view(num_experts, swizzled.numel())


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Sm90Fp8Kind = "fp8_e4m3",
    fp8_scale_mode: Sm90Fp8ScaleMode = "per_tensor",
    fc1_activation_dequant_scale: float = 1.0,
    fc2_activation_dequant_scale: float = 1.0,
) -> TransformedMegaWeights:
    """bf16 canonical weights → kernel-ready SM90 FP8 mega layout.

    ``fc1_activation_dequant_scale`` / ``fc2_activation_dequant_scale`` are
    the per-tensor static activation calibration scalars (materialized here as
    the (1,) fp32 launch tensors the kernel ABI takes; identical on all EP
    ranks — see the config docstring).  Ignored in blockwise mode.
    """
    import torch

    # Backend talks only to the pull_style_cutedsl_megakernel shim boundary.
    from .....core.validation.common import MoEEpConfigError

    if isinstance(weights, PrequantizedMoEWeights):
        # PORT NOTE: pre-quantized FP8 packs (per-expert or blockwise scales)
        # are not wired yet for sm90_pull_fp8; kernel-ready weights can still
        # bypass preprocessing entirely via MegaConfig.preprocess_weights=False
        # + transformed_weights (validated by validate_transformed_mega_weights).
        raise MoEEpConfigError(
            "sm90_pull_fp8 does not support PrequantizedMoEWeights yet; pass "
            "canonical bf16 w13/w2, or supply kernel-ready transformed "
            "weights with MegaConfig.preprocess_weights=False"
        )

    blockwise = fp8_scale_mode == "blockwise"
    fc1_out = 2 * intermediate_size
    num_experts = weights.w13.shape[0]
    data_dtype = _fp8_data_dtype(kind)
    device = weights.w13.device

    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    if weights.w13.shape != logical_w13_shape:
        raise ValueError(
            f"w13 must have shape {logical_w13_shape}, got {tuple(weights.w13.shape)}"
        )
    if weights.w2.shape != logical_w2_shape:
        raise ValueError(
            f"w2 must have shape {logical_w2_shape}, got {tuple(weights.w2.shape)}"
        )
    if blockwise and (hidden_size % 128 != 0 or intermediate_size % 128 != 0):
        raise ValueError(
            "blockwise FP8 requires hidden_size and intermediate_size to be "
            f"multiples of 128; got hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}."
        )

    # FC1 N axis: gate/up interleaved in blocks of 8 rows BEFORE quantization,
    # so blockwise (N, K) scale blocks align with the layout the kernel sees.
    w13_interleaved = _interleave_gate_up_8(
        weights.w13, intermediate_size=fc1_out
    )

    fc1_q_parts: list["torch.Tensor"] = []
    fc2_q_parts: list["torch.Tensor"] = []
    fc1_sf_parts: list["torch.Tensor"] = []
    fc2_sf_parts: list["torch.Tensor"] = []
    for expert in range(num_experts):
        if blockwise:
            from .....kernel_src.sm90.pull_style_cutedsl_megakernel import (
                quantize_fp8_weight_block_nk,
            )

            fc1_q, fc1_sf = quantize_fp8_weight_block_nk(
                w13_interleaved[expert].to(torch.float32), data_dtype
            )
            fc2_q, fc2_sf = quantize_fp8_weight_block_nk(
                weights.w2[expert].to(torch.float32), data_dtype
            )
        else:
            fc1_q, fc1_sf = _quantize_fp8_per_tensor_expert(
                w13_interleaved[expert], data_dtype=data_dtype
            )
            fc2_q, fc2_sf = _quantize_fp8_per_tensor_expert(
                weights.w2[expert], data_dtype=data_dtype
            )
        fc1_q_parts.append(fc1_q)
        fc2_q_parts.append(fc2_q)
        fc1_sf_parts.append(fc1_sf)
        fc2_sf_parts.append(fc2_sf)

    # Kernel weight layout: (E, K, N) with K stride-1.  Quantization ran in
    # logical (N, K) row-major; the transpose keeps K stride-1 WITHOUT a
    # .contiguous() re-pack (which would silently break the K-major invariant
    # the SM90 GEMM's TMA descriptors depend on).
    fc1_weight = torch.stack(fc1_q_parts, dim=0).transpose(1, 2)
    fc2_weight = torch.stack(fc2_q_parts, dim=0).transpose(1, 2)

    if blockwise:
        # (E, 2I/128, H/128) / (E, H/128, I/128) fp32 — used by the GEMMs.
        fc1_weight_sf = torch.stack(fc1_sf_parts, dim=0)
        fc2_weight_sf = torch.stack(fc2_sf_parts, dim=0)
        return (
            (fc1_weight, fc1_weight_sf, None, None),
            (fc2_weight, fc2_weight_sf, None, None),
        )

    # Per-tensor: (E,) fp32 weight scales + (1,) fp32 static activation scales
    # (real dequant inputs), plus swizzled unit E8M0 placeholder SF planes.
    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import ceil_div

    fc1_weight_dequant_scale = torch.cat(fc1_sf_parts).to(device)
    fc2_weight_dequant_scale = torch.cat(fc2_sf_parts).to(device)
    fc1_act_scale = torch.tensor(
        (float(fc1_activation_dequant_scale),), dtype=torch.float32, device=device
    )
    fc2_act_scale = torch.tensor(
        (float(fc2_activation_dequant_scale),), dtype=torch.float32, device=device
    )

    hidden_sf_cols = ceil_div(hidden_size, 32)
    intermediate_sf_cols = ceil_div(intermediate_size, 32)
    fc1_weight_sf = _swizzle_unit_e8m0_sf(
        num_experts, fc1_out, hidden_sf_cols, device
    )
    fc2_weight_sf = _swizzle_unit_e8m0_sf(
        num_experts, hidden_size, intermediate_sf_cols, device
    )

    return (
        (fc1_weight, fc1_weight_sf, fc1_act_scale, fc1_weight_dequant_scale),
        (fc2_weight, fc2_weight_sf, fc2_act_scale, fc2_weight_dequant_scale),
    )


def _check_leg_structure(transformed: object) -> None:
    """4-tuple-per-leg layout (this kernel carries the fp8 dequant scales)."""
    from .....core.validation.common import MoEEpConfigError

    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError(
            "transformed_weights must be a 2-tuple (fc1, fc2), got "
            f"{type(transformed).__name__}"
        )
    for idx, leg in enumerate(transformed):
        label = "fc1" if idx == 0 else "fc2"
        if not isinstance(leg, tuple) or len(leg) != 4:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be (weight, weight_sf, "
                "activation_dequant_scale, weight_dequant_scale), got "
                f"{type(leg).__name__}"
            )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Sm90Fp8Kind = "fp8_e4m3",
    fp8_scale_mode: Sm90Fp8ScaleMode = "per_tensor",
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready FP8 weights (``preprocess_weights=False``).

    Shape/dtype checks only (the shim frontend re-validates strides and
    per-launch invariants); the dequant-scale slots must be present in
    per_tensor mode and may be None in blockwise mode.
    """
    import torch

    from .....core.validation.common import MoEEpConfigError
    from ..weight_validation import check_transformed_weight_pair

    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )

    _check_leg_structure(transformed)

    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    data_dtype = _fp8_data_dtype(kind)
    blockwise = fp8_scale_mode == "blockwise"

    if blockwise:
        fc1_sf_shape = (local_experts, fc1_out // 128, hidden_size // 128)
        fc2_sf_shape = (local_experts, hidden_size // 128, intermediate_size // 128)
        sf_dtype = torch.float32
    else:
        from .....kernel_src.sm90.pull_style_cutedsl_megakernel import ceil_div

        fc1_sf_shape = (
            local_experts,
            _swizzled_flat_e8m0_size(fc1_out, ceil_div(hidden_size, 32)),
        )
        fc2_sf_shape = (
            local_experts,
            _swizzled_flat_e8m0_size(hidden_size, ceil_div(intermediate_size, 32)),
        )
        sf_dtype = torch.uint8

    check_transformed_weight_pair(
        transformed[0][:2],
        label="fc1",
        num_local_experts=local_experts,
        weight_dtype=data_dtype,
        expected_weight_shape=(local_experts, hidden_size, fc1_out),
        scale_dtype=sf_dtype,
        expected_scale_shape=fc1_sf_shape,
    )
    check_transformed_weight_pair(
        transformed[1][:2],
        label="fc2",
        num_local_experts=local_experts,
        weight_dtype=data_dtype,
        expected_weight_shape=(local_experts, intermediate_size, hidden_size),
        scale_dtype=sf_dtype,
        expected_scale_shape=fc2_sf_shape,
    )

    for idx, (label, expert_dim) in enumerate((("fc1", local_experts), ("fc2", local_experts))):
        _weight, _sf, act_scale, weight_scale = transformed[idx]
        for name, scale, shape in (
            (f"{label} activation_dequant_scale", act_scale, (1,)),
            (f"{label} weight_dequant_scale", weight_scale, (expert_dim,)),
        ):
            if scale is None:
                if not blockwise:
                    raise MoEEpConfigError(
                        f"transformed_weights {name} is required in per_tensor "
                        "fp8_scale_mode (only blockwise may pass None)"
                    )
                continue
            if not isinstance(scale, torch.Tensor):
                raise MoEEpConfigError(
                    f"transformed_weights {name} must be a torch.Tensor, "
                    f"got {type(scale).__name__}"
                )
            if tuple(scale.shape) != shape:
                raise MoEEpConfigError(
                    f"transformed_weights {name} must have shape {shape}, "
                    f"got {tuple(scale.shape)}"
                )
            if scale.dtype != torch.float32:
                raise MoEEpConfigError(
                    f"transformed_weights {name} must be float32, got {scale.dtype}"
                )


def _swizzled_flat_e8m0_size(rows: int, cols: int) -> int:
    import torch

    # Backend talks only to the pull_style_cutedsl_megakernel shim boundary.
    from .....kernel_src.sm90.pull_style_cutedsl_megakernel import to_blocked

    plain = torch.zeros(rows, cols, dtype=torch.float8_e8m0fnu)
    return to_blocked(plain).numel()


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
