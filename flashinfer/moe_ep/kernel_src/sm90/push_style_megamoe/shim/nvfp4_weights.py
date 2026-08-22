"""Typed weight bundle for the SM90 push NVFP4 runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

from .nvfp4_checkpoint import (
    NVFP4Checkpoint,
    load_modelopt_nvfp4_state_dict,
)
from .nvfp4_repack import (
    NVFP4SM90WeightViewV3,
    NVFP4SM90WeightViewV4,
    repack_nvfp4_sm90_w4a8,
)
from .weights import Sm90PushWeights, _per_block_cast_128x128

NvFp4WeightPolicy = Literal["packed", "folded", "hot_folded", "dual"]

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

    w13: NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4
    w2: NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4

    def __post_init__(self) -> None:
        valid_views = isinstance(
            self.w13, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)
        ) and isinstance(self.w2, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4))
        if not valid_views:
            raise TypeError("W4A8 weights contain incompatible view types")
        if self.w13.manifest.layout_version != self.w2.manifest.layout_version:
            raise ValueError("W4A8 FC1 and FC2 views must use the same layout version")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _fp8_weight_bytes(weights: Sm90PushWeights | None) -> int:
    if weights is None:
        return 0
    return sum(
        _tensor_bytes(tensor)
        for tensor in (
            weights.w13_fp8,
            weights.w13_sf,
            weights.w2_fp8,
            weights.w2_sf,
        )
    )


def _nvfp4_view_bytes(
    view: NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
) -> int:
    return sum(
        _tensor_bytes(tensor)
        for tensor in (
            view.packed_e2m1,
            view.scale_e4m3_per16,
            view.promotion_group_scale,
            view.promotion_residual,
            view.global_alpha,
        )
    )


@dataclass(frozen=True)
class Sm90PushNvFp4HotFoldedWeights:
    """Static hot-prefix FP8 and cold-suffix NVFP4 weight bundle."""

    hot_experts: int
    total_experts: int
    hot_fp8: Sm90PushWeights | None
    cold_nvfp4: Sm90PushNvFp4Weights | None

    def __post_init__(self) -> None:
        if type(self.hot_experts) is not int or type(self.total_experts) is not int:
            raise TypeError("hot_experts and total_experts must be integers")
        if self.total_experts <= 0:
            raise ValueError("total_experts must be positive")
        if not 0 <= self.hot_experts <= self.total_experts:
            raise ValueError("hot_experts must be in [0, total_experts]")
        if self.hot_experts == 0:
            if self.hot_fp8 is not None:
                raise ValueError("hot_fp8 must be None when hot_experts is zero")
        elif not isinstance(self.hot_fp8, Sm90PushWeights):
            raise TypeError("hot_fp8 must be Sm90PushWeights for a non-empty prefix")
        if self.hot_experts == self.total_experts:
            if self.cold_nvfp4 is not None:
                raise ValueError("cold_nvfp4 must be None when every expert is hot")
        else:
            if not isinstance(self.cold_nvfp4, Sm90PushNvFp4Weights):
                raise TypeError(
                    "cold_nvfp4 must be Sm90PushNvFp4Weights for a non-empty suffix"
                )
            expected_mapping = tuple(range(self.hot_experts, self.total_experts))
            for view in (self.cold_nvfp4.w13, self.cold_nvfp4.w2):
                if not isinstance(view, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
                    raise TypeError("hot-folded cold weights must contain W4A8 views")
                if view.manifest.expert_mapping != expected_mapping:
                    raise ValueError(
                        "hot-folded cold weights must map the frozen expert suffix"
                    )
        if self.hot_fp8 is not None:
            if self.hot_fp8.w13_interleaved:
                raise ValueError("hot-folded FP8 weights must use non-interleaved FC1")
            if int(self.hot_fp8.w13_fp8.shape[0]) != self.hot_experts:
                raise ValueError("hot FP8 expert count does not match hot_experts")
            if int(self.hot_fp8.w2_fp8.shape[0]) != self.hot_experts:
                raise ValueError("hot FP8 FC2 expert count does not match hot_experts")

    @property
    def execution_identity(self) -> tuple[str, int, int, int]:
        layout_version = 0
        if self.cold_nvfp4 is not None:
            cold_w13 = cast(
                NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
                self.cold_nvfp4.w13,
            )
            layout_version = cold_w13.manifest.layout_version
        return ("hot-prefix-v1", self.hot_experts, self.total_experts, layout_version)

    @property
    def folded_bytes(self) -> int:
        return _fp8_weight_bytes(self.hot_fp8)

    @property
    def packed_bytes(self) -> int:
        if self.cold_nvfp4 is None:
            return 0
        w13 = cast(NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4, self.cold_nvfp4.w13)
        w2 = cast(NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4, self.cold_nvfp4.w2)
        return _nvfp4_view_bytes(w13) + _nvfp4_view_bytes(w2)

    @property
    def resident_bytes(self) -> int:
        return self.folded_bytes + self.packed_bytes


@dataclass(frozen=True)
class Sm90PushNvFp4DualWeights:
    """Full packed NVFP4 and full folded FP8 views kept resident together."""

    packed_nvfp4: Sm90PushNvFp4Weights
    folded_fp8: Sm90PushWeights

    def __post_init__(self) -> None:
        if not isinstance(self.packed_nvfp4, Sm90PushNvFp4Weights):
            raise TypeError("packed_nvfp4 must be Sm90PushNvFp4Weights")
        if not isinstance(self.folded_fp8, Sm90PushWeights):
            raise TypeError("folded_fp8 must be Sm90PushWeights")
        w13 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w13,
        )
        w2 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w2,
        )
        experts, two_i, hidden = w13.manifest.logical_shape
        w2_experts, w2_hidden, intermediate = w2.manifest.logical_shape
        if (w2_experts, w2_hidden, two_i) != (experts, hidden, 2 * intermediate):
            raise ValueError("dual packed weights must have (E, 2I, H)/(E, H, I)")
        if tuple(self.folded_fp8.w13_fp8.shape) != (experts, two_i, hidden):
            raise ValueError("dual folded FC1 shape does not match packed FC1")
        if tuple(self.folded_fp8.w2_fp8.shape) != (
            experts,
            hidden,
            intermediate,
        ):
            raise ValueError("dual folded FC2 shape does not match packed FC2")
        if self.folded_fp8.w13_interleaved:
            raise ValueError("dual folded FC1 must use the non-interleaved layout")
        if self.folded_fp8.w13_fp8.device != w13.packed_e2m1.device:
            raise ValueError("dual packed and folded weights must share a device")

    @property
    def total_experts(self) -> int:
        w13 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w13,
        )
        return int(w13.manifest.logical_shape[0])

    @property
    def execution_identity(self) -> tuple[str, int, int]:
        w13 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w13,
        )
        return ("dual-folded-v1", self.total_experts, w13.manifest.layout_version)

    @property
    def folded_bytes(self) -> int:
        return _fp8_weight_bytes(self.folded_fp8)

    @property
    def packed_bytes(self) -> int:
        w13 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w13,
        )
        w2 = cast(
            NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4,
            self.packed_nvfp4.w2,
        )
        return _nvfp4_view_bytes(w13) + _nvfp4_view_bytes(w2)

    @property
    def resident_bytes(self) -> int:
        return self.packed_bytes + self.folded_bytes


@dataclass(frozen=True)
class NvFp4ResidencyEstimate:
    """Geometry-derived resident bytes for one NVFP4 weight policy."""

    policy: NvFp4WeightPolicy
    local_experts: int
    hot_experts: int
    packed_bytes: int
    folded_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.packed_bytes + self.folded_bytes

    @property
    def mib(self) -> float:
        return self.total_bytes / float(1 << 20)


def estimate_nvfp4_residency(
    local_experts: int,
    policy: NvFp4WeightPolicy,
    *,
    hidden_size: int,
    intermediate_size: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    hot_expert_count: int = 0,
) -> NvFp4ResidencyEstimate:
    """Estimate packed/folded weight residency without allocating tensors."""

    if type(local_experts) is not int or local_experts <= 0:
        raise ValueError("local_experts must be a positive integer")
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be divisible by 128")
    if group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    if residual_scheme not in ("generic", "pow2"):
        raise ValueError("residual_scheme must be generic or pow2")
    if policy not in ("packed", "folded", "hot_folded", "dual"):
        raise ValueError("unsupported NVFP4 weight policy")
    if policy == "packed":
        hot = 0
    elif policy in ("folded", "dual"):
        hot = local_experts
    else:
        hot = hot_expert_count
        if not 0 < hot < local_experts:
            raise ValueError("hot_folded requires a nonempty proper hot prefix")

    matrix_elements = 3 * hidden_size * intermediate_size
    residual_bytes_per_element = 4 / 32 if residual_scheme == "generic" else 2 / 32
    packed_per_expert = (
        int(
            matrix_elements
            * (0.5 + 1 / 16 + 4 / group_size + residual_bytes_per_element)
        )
        + 8
    )
    folded_per_expert = matrix_elements + (
        12 * hidden_size * intermediate_size // (128 * 128)
    )
    packed_experts = (
        local_experts if policy in ("packed", "dual") else local_experts - hot
    )
    folded_experts = hot
    return NvFp4ResidencyEstimate(
        policy=policy,
        local_experts=local_experts,
        hot_experts=hot,
        packed_bytes=packed_experts * packed_per_expert,
        folded_bytes=folded_experts * folded_per_expert,
    )


def _slice_checkpoint_experts(
    checkpoint: NVFP4Checkpoint,
    begin: int,
    end: int,
) -> NVFP4Checkpoint:
    alpha = checkpoint.global_alpha
    if checkpoint.alpha_scope == "per_expert":
        alpha = alpha[begin:end].clone(memory_format=torch.contiguous_format)
    return NVFP4Checkpoint(
        checkpoint.packed_e2m1[begin:end].contiguous(),
        checkpoint.scale_e4m3_per16[begin:end].contiguous(),
        alpha,
        (end - begin, checkpoint.logical_shape[1], checkpoint.logical_shape[2]),
        checkpoint.expert_mapping[begin:end],
        checkpoint.source_format_version,
    )


def _validate_hot_folded_request(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    hot_experts: int,
    *,
    require_same_device: bool = True,
) -> int:
    _validate_folded_checkpoint_pair(
        w13,
        w2,
        require_same_device=require_same_device,
    )
    experts = w13.logical_shape[0]
    if type(hot_experts) is not int:
        raise TypeError("hot_experts must be an integer")
    if not 0 <= hot_experts <= experts:
        raise ValueError(f"hot_experts must be in [0, {experts}]")
    if w13.expert_mapping != tuple(range(experts)):
        raise ValueError("hot-folded weights require identity-ordered local experts")
    return experts


def _partition_hot_folded_checkpoint(
    checkpoint: NVFP4Checkpoint,
    hot_experts: int,
    *,
    group_size: int,
    residual_scheme: str,
    payload_layout: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor] | None,
    NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4 | None,
]:
    experts = checkpoint.logical_shape[0]
    hot = None
    if hot_experts:
        hot = fold_nvfp4_checkpoint_to_fp8_blockscale(
            _slice_checkpoint_experts(checkpoint, 0, hot_experts)
        )
    cold = None
    if hot_experts != experts:
        cold = repack_nvfp4_sm90_w4a8(
            _slice_checkpoint_experts(checkpoint, hot_experts, experts),
            group_size=group_size,
            residual_scheme=residual_scheme,
            payload_layout=payload_layout,
            allow_legacy_layout=payload_layout == 3,
        )
    return hot, cold


def make_sm90_push_nvfp4_hot_folded_weights_from_checkpoints(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    hot_experts: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
) -> Sm90PushNvFp4HotFoldedWeights:
    """Fold a local expert prefix and retain the suffix in packed NVFP4 form."""

    experts = _validate_hot_folded_request(w13, w2, hot_experts)
    hot_w13, cold_w13 = _partition_hot_folded_checkpoint(
        w13,
        hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )
    hot_w2, cold_w2 = _partition_hot_folded_checkpoint(
        w2,
        hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )
    hot_fp8 = (
        None
        if hot_w13 is None or hot_w2 is None
        else Sm90PushWeights(
            w13_fp8=hot_w13[0],
            w13_sf=hot_w13[1],
            w2_fp8=hot_w2[0],
            w2_sf=hot_w2[1],
            w13_interleaved=False,
        )
    )
    cold_nvfp4 = (
        None
        if cold_w13 is None or cold_w2 is None
        else Sm90PushNvFp4Weights(cold_w13, cold_w2)
    )
    return Sm90PushNvFp4HotFoldedWeights(
        hot_experts=hot_experts,
        total_experts=experts,
        hot_fp8=hot_fp8,
        cold_nvfp4=cold_nvfp4,
    )


def make_sm90_push_nvfp4_weights_from_checkpoints(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
) -> Sm90PushNvFp4Weights:
    """Convert canonical NVFP4 checkpoints to one kernel-ready layout."""

    if not isinstance(w13, NVFP4Checkpoint) or not isinstance(w2, NVFP4Checkpoint):
        raise TypeError("w13 and w2 must be NVFP4Checkpoint instances")
    if w13.device != w2.device:
        raise ValueError("w13 and w2 checkpoints must share a device")
    if w13.expert_mapping != w2.expert_mapping:
        raise ValueError("w13 and w2 checkpoints must share an expert mapping")
    if payload_layout not in (3, 4):
        raise ValueError("payload_layout must be 3 or 4")
    w13_view = repack_nvfp4_sm90_w4a8(
        w13,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        allow_legacy_layout=payload_layout == 3,
    )
    w2_view = repack_nvfp4_sm90_w4a8(
        w2,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
        allow_legacy_layout=payload_layout == 3,
    )
    return Sm90PushNvFp4Weights(w13_view, w2_view)


def make_sm90_push_nvfp4_dual_weights_from_checkpoints(
    w13: NVFP4Checkpoint,
    w2: NVFP4Checkpoint,
    *,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
) -> Sm90PushNvFp4DualWeights:
    """Keep full packed W4A8 and full folded FP8 representations resident."""

    packed = make_sm90_push_nvfp4_weights_from_checkpoints(
        w13,
        w2,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )
    folded = make_sm90_push_folded_fp8_weights_from_checkpoints(w13, w2)
    return Sm90PushNvFp4DualWeights(packed, folded)


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
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
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
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )


def load_sm90_push_nvfp4_modelopt_dual_weights(
    state_dict,
    *,
    w13_prefix: str,
    w2_prefix: str,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
    device=None,
) -> Sm90PushNvFp4DualWeights:
    """Load ModelOpt weights and retain packed plus fully folded views."""

    w13, w2 = _load_modelopt_checkpoint_pair(
        state_dict,
        w13_prefix=w13_prefix,
        w2_prefix=w2_prefix,
        device=device,
    )
    return make_sm90_push_nvfp4_dual_weights_from_checkpoints(
        w13,
        w2,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
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


def load_sm90_push_nvfp4_modelopt_hot_folded_weights(
    state_dict,
    *,
    w13_prefix: str,
    w2_prefix: str,
    hot_experts: int,
    group_size: int = 128,
    residual_scheme: str = "generic",
    payload_layout: Literal[3, 4] = 4,
    device=None,
) -> Sm90PushNvFp4HotFoldedWeights:
    """Load a static hot-prefix bundle while moving one source tensor at a time."""

    source_w13 = load_modelopt_nvfp4_state_dict(state_dict, prefix=w13_prefix)
    source_w2 = load_modelopt_nvfp4_state_dict(state_dict, prefix=w2_prefix)
    experts = _validate_hot_folded_request(
        source_w13,
        source_w2,
        hot_experts=hot_experts,
        require_same_device=device is None,
    )
    w13 = _move_modelopt_checkpoint(source_w13, device)
    hot_w13, cold_w13 = _partition_hot_folded_checkpoint(
        w13,
        hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )
    del source_w13, w13
    w2 = _move_modelopt_checkpoint(source_w2, device)
    hot_w2, cold_w2 = _partition_hot_folded_checkpoint(
        w2,
        hot_experts,
        group_size=group_size,
        residual_scheme=residual_scheme,
        payload_layout=payload_layout,
    )
    hot_fp8 = (
        None
        if hot_w13 is None or hot_w2 is None
        else Sm90PushWeights(
            w13_fp8=hot_w13[0],
            w13_sf=hot_w13[1],
            w2_fp8=hot_w2[0],
            w2_sf=hot_w2[1],
            w13_interleaved=False,
        )
    )
    cold_nvfp4 = (
        None
        if cold_w13 is None or cold_w2 is None
        else Sm90PushNvFp4Weights(cold_w13, cold_w2)
    )
    return Sm90PushNvFp4HotFoldedWeights(
        hot_experts=hot_experts,
        total_experts=experts,
        hot_fp8=hot_fp8,
        cold_nvfp4=cold_nvfp4,
    )


__all__ = [
    "NvFp4ResidencyEstimate",
    "NvFp4WeightPolicy",
    "Sm90PushNvFp4DualWeights",
    "Sm90PushNvFp4HotFoldedWeights",
    "Sm90PushNvFp4Weights",
    "estimate_nvfp4_residency",
    "fold_nvfp4_checkpoint_to_fp8_blockscale",
    "load_sm90_push_nvfp4_modelopt_dual_weights",
    "load_sm90_push_nvfp4_modelopt_folded_fp8_weights",
    "load_sm90_push_nvfp4_modelopt_hot_folded_weights",
    "load_sm90_push_nvfp4_modelopt_weights",
    "make_sm90_push_folded_fp8_weights_from_checkpoints",
    "make_sm90_push_nvfp4_dual_weights_from_checkpoints",
    "make_sm90_push_nvfp4_hot_folded_weights_from_checkpoints",
    "make_sm90_push_nvfp4_weights_from_checkpoints",
]
