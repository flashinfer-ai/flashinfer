"""Typed weight bundle for the SM90 push NVFP4 runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ......fused_moe.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    load_modelopt_nvfp4_state_dict,
)
from ......fused_moe.sm90_nvfp4_repack import (
    NVFP4RSWeightView,
    NVFP4SM90WeightViewV3,
    build_nvfp4_rs_weight_view,
    repack_nvfp4_sm90_v3,
)
from .weights import Sm90PushWeights, _per_block_cast_128x128

NvFp4Mode = Literal["w4a8", "w4a16_rs"]

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_FOLDED_FP8_CHUNK_ROWS = 1024


@torch.no_grad()
def _fold_nvfp4_checkpoint_to_fp8_blockscale(
    checkpoint: NVFP4Checkpoint,
    *,
    interleave_gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(checkpoint, NVFP4Checkpoint):
        raise TypeError("checkpoint must be an NVFP4Checkpoint")
    experts, rows, columns = checkpoint.logical_shape
    if rows % 128 or columns % 128:
        raise ValueError(
            "folded FP8 weights require logical N and K dimensions divisible by 128"
        )
    expected_mapping = tuple(range(experts))
    if checkpoint.expert_mapping != expected_mapping:
        raise ValueError("folded FP8 weights require identity-ordered local experts")
    if interleave_gate_up and rows % 256:
        raise ValueError("gate/up interleave requires N divisible by 256")

    output = torch.empty(
        (experts, rows, columns),
        dtype=torch.float8_e4m3fn,
        device=checkpoint.device,
    )
    scales = torch.empty(
        (experts, rows // 128, columns // 128),
        dtype=torch.float32,
        device=checkpoint.device,
    )
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=checkpoint.device)
    alpha = checkpoint.global_alpha_per_expert.to(torch.float32)
    row_blocks = rows // 128
    output_bytes = output.view(torch.uint8).reshape(experts, row_blocks, 128, columns)
    block_mapping = None
    if interleave_gate_up:
        logical_blocks = torch.arange(row_blocks, device=checkpoint.device)
        blocks_per_half = row_blocks // 2
        block_mapping = logical_blocks.remainder(
            blocks_per_half
        ) * 2 + logical_blocks.div(blocks_per_half, rounding_mode="floor")
    chunk_rows = max(
        128,
        min(rows, (_FOLDED_FP8_CHUNK_ROWS // 128) * 128),
    )

    for expert in range(experts):
        expert_alpha = alpha[expert]
        for row_begin in range(0, rows, chunk_rows):
            row_end = min(row_begin + chunk_rows, rows)
            rows_in_chunk = row_end - row_begin
            packed = checkpoint.packed_e2m1[expert, row_begin:row_end, : columns // 2]
            low = packed.bitwise_and(0x0F)
            high = packed.bitwise_right_shift(4).bitwise_and(0x0F)
            codes = torch.stack((low, high), dim=-1).reshape(rows_in_chunk, columns)
            decoded = values[codes.to(torch.int64)]
            per16 = checkpoint.scale_e4m3_per16[
                expert, row_begin:row_end, : columns // 16
            ].to(torch.float32)
            decoded.reshape(rows_in_chunk, columns // 16, 16).mul_(per16.unsqueeze(-1))
            decoded.mul_(expert_alpha)
            decoded = torch.where(decoded == 0, torch.zeros_like(decoded), decoded)
            quantized, block_scales = _per_block_cast_128x128(decoded)
            block_begin = row_begin // 128
            block_end = row_end // 128
            quantized_bytes = quantized.view(torch.uint8).reshape(
                block_end - block_begin, 128, columns
            )
            if block_mapping is None:
                output_bytes[expert, block_begin:block_end].copy_(quantized_bytes)
                scales[expert, block_begin:block_end].copy_(block_scales)
            else:
                destination = block_mapping[block_begin:block_end]
                output_bytes[expert].index_copy_(0, destination, quantized_bytes)
                scales[expert].index_copy_(0, destination, block_scales)
    if not bool((torch.isfinite(scales) & (scales > 0)).all()):
        raise ValueError("folded FP8 block scales must be finite and positive")
    return output, scales


def fold_nvfp4_checkpoint_to_fp8_blockscale(
    checkpoint: NVFP4Checkpoint,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert canonical NVFP4 weights to E4M3 with 128x128 FP32 scales."""

    return _fold_nvfp4_checkpoint_to_fp8_blockscale(
        checkpoint,
        interleave_gate_up=False,
    )


def _validate_folded_checkpoint_pair(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    require_same_device: bool = True,
) -> None:
    if not isinstance(w13, NVFP4Checkpoint) or not isinstance(w2, NVFP4Checkpoint):
        raise TypeError("w13 and w2 must be NVFP4Checkpoint instances")
    if require_same_device and w13.device != w2.device:
        raise ValueError("w13 and w2 checkpoints must share a device")
    if w13.expert_mapping != w2.expert_mapping:
        raise ValueError("w13 and w2 checkpoints must share an expert mapping")
    experts, two_i, hidden = w13.logical_shape
    w2_experts, w2_hidden, intermediate = w2.logical_shape
    if w2_experts != experts or w2_hidden != hidden or two_i != 2 * intermediate:
        raise ValueError(
            "folded FP8 checkpoints must have shapes (E, 2I, H) and (E, H, I)"
        )
    if hidden % 128 or intermediate % 128:
        raise ValueError(
            "folded FP8 checkpoints require hidden and intermediate dimensions "
            "divisible by 128"
        )


def make_sm90_push_folded_fp8_weights_from_checkpoints(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    interleave_gate_up: bool = False,
) -> Sm90PushWeights:
    """Build weights consumed by ``Sm90PushFp8MegaMoeConfig`` from NVFP4."""

    _validate_folded_checkpoint_pair(
        w13,
        w2,
    )
    w13_fp8, w13_sf = _fold_nvfp4_checkpoint_to_fp8_blockscale(
        w13,
        interleave_gate_up=interleave_gate_up,
    )
    w2_fp8, w2_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(w2)
    return Sm90PushWeights(
        w13_fp8=w13_fp8,
        w13_sf=w13_sf,
        w2_fp8=w2_fp8,
        w2_sf=w2_sf,
        w13_interleaved=interleave_gate_up,
    )


@dataclass(frozen=True)
class Sm90PushNvFp4Weights:
    """FC1 and FC2 views tagged with their consuming kernel layout."""

    nvfp4_mode: NvFp4Mode
    w13: NVFP4SM90WeightViewV3 | NVFP4RSWeightView
    w2: NVFP4SM90WeightViewV3 | NVFP4RSWeightView

    def __post_init__(self) -> None:
        expected: type[NVFP4SM90WeightViewV3] | type[NVFP4RSWeightView]
        if self.nvfp4_mode == "w4a8":
            expected = NVFP4SM90WeightViewV3
        elif self.nvfp4_mode == "w4a16_rs":
            expected = NVFP4RSWeightView
        else:
            raise ValueError(
                f"nvfp4_mode must be 'w4a8' or 'w4a16_rs', got {self.nvfp4_mode!r}"
            )
        if not isinstance(self.w13, expected) or not isinstance(self.w2, expected):
            raise TypeError(
                f"{self.nvfp4_mode} weights must contain two {expected.__name__} views"
            )


def make_sm90_push_nvfp4_weights_from_checkpoints(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    nvfp4_mode: NvFp4Mode = "w4a8",
    group_size: int = 128,
    residual_scheme: str = "generic",
) -> Sm90PushNvFp4Weights:
    """Convert canonical NVFP4 checkpoints to one kernel-ready layout."""

    if not isinstance(w13, NVFP4Checkpoint) or not isinstance(w2, NVFP4Checkpoint):
        raise TypeError("w13 and w2 must be NVFP4Checkpoint instances")
    if w13.device != w2.device:
        raise ValueError("w13 and w2 checkpoints must share a device")
    if w13.expert_mapping != w2.expert_mapping:
        raise ValueError("w13 and w2 checkpoints must share an expert mapping")
    if nvfp4_mode == "w4a8":
        return Sm90PushNvFp4Weights(
            nvfp4_mode,
            repack_nvfp4_sm90_v3(
                w13,
                group_size=group_size,
                residual_scheme=residual_scheme,
            ),
            repack_nvfp4_sm90_v3(
                w2,
                group_size=group_size,
                residual_scheme=residual_scheme,
            ),
        )
    if nvfp4_mode == "w4a16_rs":
        expected_mapping = tuple(range(w13.logical_shape[0]))
        if w13.expert_mapping != expected_mapping:
            raise ValueError("w4a16_rs requires identity-ordered local experts")
        return Sm90PushNvFp4Weights(
            nvfp4_mode,
            build_nvfp4_rs_weight_view(
                w13.packed_e2m1,
                w13.scale_e4m3_per16,
                w13.global_alpha_per_expert.contiguous(),
            ),
            build_nvfp4_rs_weight_view(
                w2.packed_e2m1,
                w2.scale_e4m3_per16,
                w2.global_alpha_per_expert.contiguous(),
            ),
        )
    raise ValueError(f"nvfp4_mode must be 'w4a8' or 'w4a16_rs', got {nvfp4_mode!r}")


def _move_modelopt_checkpoint(
    checkpoint: NVFP4Checkpoint,
    device,
) -> NVFP4Checkpoint:
    if device is not None:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("SM90 push NVFP4 checkpoint weights require a CUDA device")
        if checkpoint.device != target:
            return NVFP4Checkpoint(
                checkpoint.packed_e2m1.to(target),
                checkpoint.scale_e4m3_per16.to(target),
                checkpoint.global_alpha.to(target),
                checkpoint.logical_shape,
                checkpoint.expert_mapping,
                checkpoint.source_format_version,
            )
        return checkpoint
    if checkpoint.device.type != "cuda":
        raise ValueError(
            "ModelOpt tensors are on CPU; pass device='cuda:<index>' before repacking"
        )
    return checkpoint


def _load_modelopt_checkpoint_pair(
    state_dict,
    *,
    w13_prefix: str,
    w2_prefix: str,
    device=None,
) -> tuple[NVFP4Checkpoint, NVFP4Checkpoint]:
    """Load two ModelOpt checkpoints onto one optional target device."""

    w13 = _move_modelopt_checkpoint(
        load_modelopt_nvfp4_state_dict(state_dict, prefix=w13_prefix),
        device,
    )
    w2 = _move_modelopt_checkpoint(
        load_modelopt_nvfp4_state_dict(state_dict, prefix=w2_prefix),
        device,
    )
    return w13, w2


def load_sm90_push_nvfp4_modelopt_weights(
    state_dict,
    *,
    w13_prefix: str,
    w2_prefix: str,
    nvfp4_mode: NvFp4Mode = "w4a8",
    group_size: int = 128,
    residual_scheme: str = "generic",
    device=None,
) -> Sm90PushNvFp4Weights:
    """Load two ModelOpt tensors and convert them to a kernel-ready bundle."""

    w13, w2 = _load_modelopt_checkpoint_pair(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        device=device,
    )
    return make_sm90_push_nvfp4_weights_from_checkpoints(
        w13,
        w2,
        nvfp4_mode=nvfp4_mode,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )


def load_sm90_push_nvfp4_modelopt_folded_fp8_weights(
    state_dict,
    *,
    w13_prefix: str,
    w2_prefix: str,
    interleave_gate_up: bool = False,
    device=None,
) -> Sm90PushWeights:
    """Load ModelOpt NVFP4 tensors for ``Sm90PushFp8MegaMoeConfig``.

    CPU tensors are moved and folded one layer at a time when ``device`` is set.
    CUDA source tensors remain owned by the caller.
    """

    source_w13 = load_modelopt_nvfp4_state_dict(state_dict, prefix=w13_prefix)
    source_w2 = load_modelopt_nvfp4_state_dict(state_dict, prefix=w2_prefix)
    _validate_folded_checkpoint_pair(
        source_w13,
        source_w2,
        require_same_device=device is None,
    )
    w13 = _move_modelopt_checkpoint(source_w13, device)
    w13_fp8, w13_sf = _fold_nvfp4_checkpoint_to_fp8_blockscale(
        w13,
        interleave_gate_up=interleave_gate_up,
    )
    del source_w13, w13
    w2 = _move_modelopt_checkpoint(source_w2, device)
    w2_fp8, w2_sf = fold_nvfp4_checkpoint_to_fp8_blockscale(w2)
    return Sm90PushWeights(
        w13_fp8=w13_fp8,
        w13_sf=w13_sf,
        w2_fp8=w2_fp8,
        w2_sf=w2_sf,
        w13_interleaved=interleave_gate_up,
    )


__all__ = [
    "NvFp4Mode",
    "Sm90PushNvFp4Weights",
    "fold_nvfp4_checkpoint_to_fp8_blockscale",
    "load_sm90_push_nvfp4_modelopt_folded_fp8_weights",
    "load_sm90_push_nvfp4_modelopt_weights",
    "make_sm90_push_folded_fp8_weights_from_checkpoints",
    "make_sm90_push_nvfp4_weights_from_checkpoints",
]
