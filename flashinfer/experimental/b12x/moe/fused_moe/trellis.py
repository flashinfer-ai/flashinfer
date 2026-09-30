"""Preparation of canonical ``b12x_trellis`` MoE checkpoint tensors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import MixedTrellisRotations
    from b12x.moe._shared.kernels.w4a16.prepare import PreparedW4A16MoeWeights

from .config import (
    RateGranularity,
    ScaleGranularity,
    TrellisCodebook,
    TrellisConfig,
    TrellisScaleFactorsConfig,
)
from .weights import ScaleFactors, TrellisWeights
from .source import TrellisExtent, TrellisSource
from .trellis_layout import TrellisStaging, append_intermediate_signs, assemble_uniform_slots


_SLOT_CHANNELS = 32
_PROJECTIONS = 3
_TIERS = (3, 4, 5)


@dataclass(frozen=True)
class PreparedProjectionTrellisWeights:
    """Prepared MCG K3/K4/K5 tiers and their projection descriptor tables."""

    tiers: tuple[
        PreparedW4A16MoeWeights,
        PreparedW4A16MoeWeights,
        PreparedW4A16MoeWeights,
    ]
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    rotations: MixedTrellisRotations
    gate_counts: tuple[int, int, int]
    up_counts: tuple[int, int, int]
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    params_dtype: torch.dtype
    source_format: str = "b12x_trellis"
    w13_layout: str = "trellis_t256_proj"
    weight_layout: str = "trellis_mixed3"
    scale_format: str = "e4m3_k32"
    trellis_codebook: str = "mcg"


def _coalesce_payloads(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Place differently shaped tier payloads in one canonical allocation."""

    if not tensors:
        raise ValueError("at least one tier payload is required")
    combined = torch.cat(tuple(tensor.reshape(-1) for tensor in tensors)).contiguous()
    views: list[torch.Tensor] = []
    cursor = 0
    for tensor in tensors:
        count = tensor.numel()
        views.append(combined.narrow(0, cursor, count).view_as(tensor))
        cursor += count
    return combined, tuple(views)


def _require_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _selected_scale_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    return _require_cuda_tensor(
        tensor, name=name, dtype=torch.float16, device=device
    )


def _validate_factor_presence(
    factors: ScaleFactors,
    declaration: TrellisScaleFactorsConfig,
    *,
    name: str,
) -> None:
    if declaration.gains is ScaleGranularity.NONE:
        if factors.gains is not None:
            raise ValueError(f"{name}.gains is declared 'none' but a tensor was supplied")
    elif factors.gains is None:
        raise ValueError(f"{name}.gains requires a tensor")


def _expert_axis(granularity: ScaleGranularity) -> bool:
    return granularity is ScaleGranularity.PER_EXPERT


def _effective_input_scales(
    factors: ScaleFactors,
    declaration: TrellisScaleFactorsConfig,
    *,
    num_experts: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_factor_presence(factors, declaration, name="input_scales")
    vectors = _selected_scale_tensor(
        factors.vectors, name="input_scales.vectors", device=device
    )
    if not _expert_axis(declaration.vectors):
        if tuple(vectors.shape) == (hidden_size,):
            vectors = vectors.reshape(1, 1, hidden_size)
        elif tuple(vectors.shape) == (2, hidden_size):
            vectors = vectors.reshape(1, 2, hidden_size)
        else:
            raise ValueError(
                "selected input scale vectors must be fp16 [H] or [2,H]"
            )
    else:
        if tuple(vectors.shape) == (num_experts, hidden_size):
            vectors = vectors.reshape(num_experts, 1, hidden_size)
        elif tuple(vectors.shape) != (num_experts, 2, hidden_size):
            raise ValueError(
                "per-expert input scale vectors must be fp16 [E,H] or [E,2,H]"
            )

    gains = factors.gains
    if gains is None:
        effective = vectors
    else:
        gains = _selected_scale_tensor(
            gains, name="input_scales.gains", device=device
        )
        gain_experts = num_experts if _expert_axis(declaration.gains) else 1
        allowed = (
            {(), (1,), (2,)}
            if gain_experts == 1
            else {(num_experts,), (num_experts, 2)}
        )
        if tuple(gains.shape) not in allowed:
            raise ValueError(
                f"selected input scale gains must have one of {sorted(allowed)}"
            )
        if gains.ndim == 0:
            gains = gains.reshape(1, 1)
        elif gains.ndim == 1:
            gains = gains.reshape(gain_experts, 1)
        effective = vectors * gains.unsqueeze(-1)

    if int(effective.shape[0]) not in (1, num_experts):
        effective = effective.expand(num_experts, -1, -1)
    if int(effective.shape[1]) == 1:
        shared = effective[:, 0].contiguous()
        return shared, shared
    return effective[:, 0].contiguous(), effective[:, 1].contiguous()


def _effective_intermediate_scales(
    factors: ScaleFactors,
    declaration: TrellisScaleFactorsConfig,
    *,
    num_experts: int,
    intermediate_size: int,
    device: torch.device,
) -> torch.Tensor:
    _validate_factor_presence(
        factors, declaration, name="intermediate_scales"
    )
    vectors = _selected_scale_tensor(
        factors.vectors, name="intermediate_scales.vectors", device=device
    )
    if _expert_axis(declaration.vectors):
        expected = (num_experts, 3, intermediate_size)
        if tuple(vectors.shape) != expected:
            raise ValueError(
                f"per-expert intermediate scale vectors must be fp16 {expected}"
            )
    else:
        expected = (3, intermediate_size)
        if tuple(vectors.shape) != expected:
            raise ValueError(
                f"selected intermediate scale vectors must be fp16 {expected}"
            )
        vectors = vectors.unsqueeze(0)

    gains = factors.gains
    if gains is not None:
        gains = _selected_scale_tensor(
            gains, name="intermediate_scales.gains", device=device
        )
        if _expert_axis(declaration.gains):
            if tuple(gains.shape) == (num_experts,):
                gains = gains.reshape(num_experts, 1)
            elif tuple(gains.shape) != (num_experts, 3):
                raise ValueError(
                    "per-expert intermediate gains must be fp16 [E] or [E,3]"
                )
        else:
            if gains.ndim == 0 or tuple(gains.shape) == (1,):
                gains = gains.reshape(1, 1)
            elif tuple(gains.shape) == (3,):
                gains = gains.reshape(1, 3)
            else:
                raise ValueError(
                    "selected intermediate gains must be fp16 [1] or [3]"
                )
        vectors = vectors * gains.unsqueeze(-1)
    return vectors.expand(num_experts, -1, -1).contiguous()


def _effective_output_scales(
    factors: ScaleFactors,
    declaration: TrellisScaleFactorsConfig,
    *,
    num_experts: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    _validate_factor_presence(factors, declaration, name="output_scales")
    vectors = _selected_scale_tensor(
        factors.vectors, name="output_scales.vectors", device=device
    )
    if _expert_axis(declaration.vectors):
        expected = (num_experts, hidden_size)
        if tuple(vectors.shape) != expected:
            raise ValueError(f"per-expert output scale vectors must be fp16 {expected}")
    else:
        expected = (hidden_size,)
        if tuple(vectors.shape) != expected:
            raise ValueError(f"selected output scale vectors must be fp16 {expected}")
        vectors = vectors.unsqueeze(0)

    gains = factors.gains
    if gains is not None:
        gains = _selected_scale_tensor(
            gains, name="output_scales.gains", device=device
        )
        if _expert_axis(declaration.gains):
            if tuple(gains.shape) != (num_experts,):
                raise ValueError("per-expert output gains must be fp16 [E]")
            gains = gains.reshape(num_experts, 1)
        else:
            if gains.numel() != 1:
                raise ValueError("selected output gains must contain one fp16 value")
            gains = gains.reshape(1, 1)
        vectors = vectors * gains
    return vectors.contiguous()


def _local_rate_matrix(
    config: TrellisConfig,
    rate: torch.Tensor,
    *,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    _require_cuda_tensor(rate, name="rate", dtype=torch.uint8, device=device)
    if config.rate.group_size is not None:
        raise NotImplementedError(
            "grouped trellis rates are represented by the format but are not "
            "implemented by the fused MoE runtime"
        )
    granularity = config.rate.granularity
    if granularity in (RateGranularity.UNIFORM, RateGranularity.PER_LAYER):
        if rate.numel() != 1:
            raise ValueError("selected uniform/per-layer rate must contain one byte")
        return rate.reshape(1, 1).expand(num_experts, _PROJECTIONS)
    if granularity is RateGranularity.PER_EXPERT:
        if tuple(rate.shape) != (num_experts,):
            raise ValueError(f"selected per-expert rate must be uint8[{num_experts}]")
        return rate.reshape(num_experts, 1).expand(-1, _PROJECTIONS)
    expected = (num_experts, _PROJECTIONS)
    if tuple(rate.shape) != expected:
        raise ValueError(
            f"selected per-expert-projection rate must be uint8{expected}"
        )
    return rate


def _symmetric_bits(
    config: TrellisConfig, rates: torch.Tensor, *, uniform_bits: int | None = None
) -> torch.Tensor:
    host = rates.detach().cpu().to(torch.int64)
    low = host & 0x0F
    high = host >> 4
    if not torch.equal(low, high):
        raise ValueError(
            "fused MoE trellis rates must encode symmetric K values (0xKK)"
        )
    if uniform_bits is not None:
        allowed = (uniform_bits,)
    elif config.codebook is TrellisCodebook.MCG:
        allowed = (3, 4, 5)
    elif config.codebook is TrellisCodebook.LUT_E4M3:
        allowed = (3,)
    else:
        raise NotImplementedError(
            "lut_fp16 is represented by the v2 format but is not implemented "
            "by the config-only fused MoE planner"
        )
    if any(int(value) not in allowed for value in low.reshape(-1).tolist()):
        raise ValueError(
            f"{config.codebook.value} fused MoE rates must use K{allowed}; "
            f"observed {sorted(set(low.reshape(-1).tolist()))}"
        )
    return low


def _matrix_section_bytes(hidden_size: int, bits: int) -> int:
    return (hidden_size // 16) * 64 * int(bits)


def _bundle_offsets(bits: torch.Tensor, hidden_size: int) -> list[list[int]]:
    offsets: list[list[int]] = []
    cursor = 0
    for expert in range(int(bits.shape[0])):
        expert_offsets = []
        for projection in range(_PROJECTIONS):
            expert_offsets.append(cursor)
            cursor += _matrix_section_bytes(
                hidden_size, int(bits[expert, projection])
            )
        offsets.append(expert_offsets)
    return offsets


def _projection_native(
    codes: torch.Tensor,
    *,
    experts: list[int],
    projection: int,
    bits: int,
    offsets: list[list[int]],
    hidden_size: int,
    fc1: bool,
) -> torch.Tensor:
    slots = int(codes.shape[0])
    hidden_tiles = hidden_size // 16
    section = _matrix_section_bytes(hidden_size, bits)
    if not experts:
        if fc1:
            return torch.zeros(
                (1, hidden_tiles, 2 * slots, 16 * bits),
                dtype=torch.int16,
                device=codes.device,
            )
        return torch.zeros(
            (1, 2 * slots, hidden_tiles, 16 * bits),
            dtype=torch.int16,
            device=codes.device,
        )
    sections = torch.stack(
        tuple(
            codes[:, offsets[expert][projection] : offsets[expert][projection] + section]
            for expert in experts
        ),
        dim=1,
    )
    words = sections.contiguous().view(torch.int16).reshape(
        slots, len(experts), 2, hidden_tiles, 16 * bits
    )
    if fc1:
        return words.permute(1, 3, 0, 2, 4).reshape(
            len(experts), hidden_tiles, 2 * slots, 16 * bits
        )
    return words.permute(1, 0, 2, 3, 4).reshape(
        len(experts), 2 * slots, hidden_tiles, 16 * bits
    )


def _intermediate_hadamard_rows(
    values: torch.Tensor,
    sign_patterns: torch.Tensor,
    *,
    intermediate_size: int,
    device: torch.device,
) -> torch.Tensor:
    return append_intermediate_signs(
        values.to(device=device), sign_patterns,
        extent=TrellisExtent(
            global_intermediate_size=intermediate_size,
            first_slot=0, slot_count=intermediate_size // 32,
        ),
    )


def _uniform_prepared(
    config: TrellisConfig,
    weights: TrellisWeights,
    bits: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate: torch.Tensor,
    down_svh: torch.Tensor,
    extent: TrellisExtent | None = None,
    staging: TrellisStaging = TrellisStaging(),
    tile_config: tuple[int, int, int, int] = (64, 256, 64, 256),
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    unique = sorted(set(bits.reshape(-1).tolist()))
    if len(unique) != 1:
        raise ValueError("uniform trellis preparation requires one K value")
    bit = int(unique[0])
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

    w13, down = assemble_uniform_slots(
        weights.codes, num_experts=num_experts, hidden_size=hidden_size,
        bits=bit, device=gate_suh.device, staging=staging,
    )
    prepared = prepare_trellis256_moe_weights(
        w13=w13,
        w2=down,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation=activation,
        fc1_tile_n=tile_config[1],
        fc2_tile_n=tile_config[3],
        params_dtype=params_dtype,
        w13_layout="trellis_t256_proj",
        trellis_bits=bit,
        codebook=config.codebook.value,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate.reshape(num_experts, -1),
        down_svh=down_svh,
        tile_config=tile_config,
        dummy_scale=dummy_scale,
        workspace=workspace,
    )
    if config.transform.expert.kind == "none":
        if weights.expert_sign_patterns is not None:
            raise ValueError(
                "expert_sign_patterns is invalid when expert transform is 'none'"
            )
        return prepared
    sign_patterns = weights.expert_sign_patterns
    if sign_patterns is None:
        raise ValueError("intermediate_hadamard preparation requires expert_sign_patterns")
    _require_cuda_tensor(
        sign_patterns,
        name="expert_sign_patterns",
        dtype=torch.uint8,
        device=gate_suh.device,
    )
    if tuple(sign_patterns.shape) != (num_experts,):
        raise ValueError(f"expert_sign_patterns must be uint8[{num_experts}]")
    if extent is None and torch.count_nonzero(sign_patterns).item():
        raise NotImplementedError(
            "rank-local intermediate_hadamard preparation currently requires "
            "all-zero expert_sign_patterns"
        )
    assert prepared.trellis is not None
    return replace(
        prepared,
        trellis=replace(
            prepared.trellis,
            intermediate_hadamard=True,
            intermediate_rotations=append_intermediate_signs(
                intermediate.reshape(num_experts, -1),
                sign_patterns,
                extent=extent or TrellisExtent(
                    global_intermediate_size=intermediate_size,
                    first_slot=0, slot_count=intermediate_size // 32,
                ),
            ),
        ),
    )


def _projection_prepared(
    config: TrellisConfig,
    weights: TrellisWeights,
    bits: torch.Tensor,
    *,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate: torch.Tensor,
    down_svh: torch.Tensor,
) -> PreparedProjectionTrellisWeights:
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        MixedTrellisRotations, build_projection_tiered_maps,
    )
    from b12x.moe._shared.kernels.w4a16.prepare import (
        _finalize_prepared_trellis_weights,
    )

    if config.codebook is not TrellisCodebook.MCG:
        raise ValueError("projection-tiered trellis execution requires codebook 'mcg'")
    if config.transform.expert.kind != "none":
        raise ValueError("projection-tiered trellis execution has no expert transform")
    if weights.expert_sign_patterns is not None:
        raise ValueError("expert_sign_patterns is invalid without an expert transform")

    offsets = _bundle_offsets(bits, hidden_size)
    memberships = tuple(
        tuple(
            [
                expert
                for expert in range(num_experts)
                if int(bits[expert, projection]) == bit
            ]
            for projection in range(_PROJECTIONS)
        )
        for bit in _TIERS
    )
    dummy_scale = torch.zeros(4, dtype=torch.uint8, device=weights.codes.device)
    shared_workspace: torch.Tensor | None = None
    tiers: list[PreparedW4A16MoeWeights] = []
    for tier_index, bit in enumerate(_TIERS):
        gate_ids, up_ids, down_ids = memberships[tier_index]
        gate = _projection_native(
            weights.codes,
            experts=gate_ids,
            projection=0,
            bits=bit,
            offsets=offsets,
            hidden_size=hidden_size,
            fc1=True,
        )
        up = _projection_native(
            weights.codes,
            experts=up_ids,
            projection=1,
            bits=bit,
            offsets=offsets,
            hidden_size=hidden_size,
            fc1=True,
        )
        down = _projection_native(
            weights.codes,
            experts=down_ids,
            projection=2,
            bits=bit,
            offsets=offsets,
            hidden_size=hidden_size,
            fc1=False,
        )
        if gate_ids or up_ids:
            w13 = torch.cat(
                tuple(
                    value
                    for value, ids in ((gate, gate_ids), (up, up_ids))
                    if ids
                )
            ).contiguous()
        else:
            w13 = gate
        prepared = _finalize_prepared_trellis_weights(
            context=f"MCG K{bit} projection-tier preparation",
            device=weights.codes.device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            params_dtype=params_dtype,
            w13=w13,
            w2=down,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate.reshape(num_experts, -1),
            down_svh=down_svh,
            rotation_columns=3 * intermediate_size,
            tile_config=(128, 128, 128, 128),
            required_fc1_tile_n=128,
            dummy_scale=dummy_scale,
            workspace=shared_workspace,
            codebook="mcg",
            trellis_bits=bit,
            fc1_pair_kind=None,
            fc2_pair_kind=None,
            fc1_pair_modes=None,
            fc2_pair_modes=None,
        )
        if shared_workspace is None:
            shared_workspace = prepared.workspace
        prepared = replace(
            prepared,
            w2_global_scale=torch.ones(
                (max(len(down_ids), 1),),
                dtype=torch.float32,
                device=weights.codes.device,
            ),
        )
        tiers.append(prepared)

    combined_w13, tier_w13 = _coalesce_payloads(
        tuple(tier.w13 for tier in tiers)
    )
    combined_w2, tier_w2 = _coalesce_payloads(tuple(tier.w2 for tier in tiers))
    rebound = [
        replace(tier, w13=w13, w2=w2)
        for tier, w13, w2 in zip(tiers, tier_w13, tier_w2, strict=True)
    ]
    tier_tuple = tuple(rebound)
    assert len(tier_tuple) == 3

    tier_ids = {bit: index for index, bit in enumerate(_TIERS)}
    gate_tiers = [tier_ids[int(value)] for value in bits[:, 0]]
    up_tiers = [tier_ids[int(value)] for value in bits[:, 1]]
    down_tiers = [tier_ids[int(value)] for value in bits[:, 2]]
    route, descriptor = build_projection_tiered_maps(
        gate_tiers,
        up_tiers,
        down_tiers,
        tier_slots=(num_experts, num_experts, num_experts),
        device=weights.codes.device,
    )
    broadcast_input = int(gate_suh.shape[0]) == 1
    broadcast_output = int(down_svh.shape[0]) == 1
    rotations = MixedTrellisRotations(
        intermediate=torch.cat((intermediate,) * 3, dim=0)
        .reshape(3 * num_experts, -1)
        .contiguous(),
        gate_suh=(
            gate_suh
            if broadcast_input
            else torch.cat((gate_suh,) * 3, dim=0).contiguous()
        ),
        up_suh=(
            up_suh
            if broadcast_input
            else torch.cat((up_suh,) * 3, dim=0).contiguous()
        ),
        down_svh=(
            down_svh
            if broadcast_output
            else torch.cat((down_svh,) * 3, dim=0).contiguous()
        ),
    )
    gate_counts = tuple(len(tier[0]) for tier in memberships)
    up_counts = tuple(len(tier[1]) for tier in memberships)
    return PreparedProjectionTrellisWeights(
        tiers=tier_tuple,  # type: ignore[arg-type]
        global_to_combined=route,
        descriptor_map=descriptor,
        rotations=rotations,
        gate_counts=gate_counts,
        up_counts=up_counts,
        w13=combined_w13,
        w2=combined_w2,
        w13_scale=dummy_scale,
        w2_scale=dummy_scale,
        w13_global_scale=tier_tuple[0].w13_global_scale,
        w2_global_scale=torch.cat(
            tuple(tier.w2_global_scale for tier in tier_tuple)
        ).contiguous(),
        workspace=tier_tuple[0].workspace,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        params_dtype=params_dtype,
    )


def prepare_trellis_weights(
    source: TrellisSource | TrellisConfig,
    weights: TrellisWeights,
    *,
    activation: str,
    params_dtype: torch.dtype,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device | str | None = None,
    staging: TrellisStaging | None = None,
    tile_config: tuple[int, int, int, int] | None = None,
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights | PreparedProjectionTrellisWeights:
    """Prepare one rank-local canonical Trellis MoE layer."""

    if isinstance(source, TrellisConfig):
        source = TrellisSource(config=source)
    config = source.config
    codes = weights.codes
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be torch.uint8, got {codes.dtype}")
    device = torch.device(device) if device is not None else codes.device
    if device.type != "cuda":
        raise ValueError("CPU trellis payloads require an explicit CUDA destination")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if codes.device.type != "cpu" and codes.device != device:
        raise ValueError("trellis codes must reside on CPU or the destination CUDA device")
    if codes.ndim != 2 or codes.stride(1) != 1:
        raise ValueError("codes must have contiguous uint8 logical rows")
    uniform = source.uniform_bits is not None or config.codebook is not TrellisCodebook.MCG
    if not uniform and (codes.device != device or not codes.is_contiguous()):
        raise ValueError("projection-tiered MCG preparation requires contiguous CUDA codes")
    if not uniform and staging is not None:
        raise ValueError("bounded host staging is supported only for uniform rates")
    staging = staging or TrellisStaging()
    if not isinstance(staging, TrellisStaging):
        raise TypeError("staging must be a TrellisStaging")
    if source.extent is not None and source.extent.intermediate_size != intermediate_size:
        raise ValueError("trellis extent width differs from the local intermediate size")
    if int(codes.shape[0]) * _SLOT_CHANNELS != intermediate_size:
        raise ValueError(
            "codes first dimension must equal intermediate_size/32: "
            f"got {int(codes.shape[0])} rows for I={intermediate_size}"
        )
    if config.transform.projection.kind != "scaled_hadamard" or (
        config.transform.projection.block_size != 128
    ):
        raise NotImplementedError(
            "fused MoE trellis preparation implements scaled_hadamard(128)"
        )

    rates = _local_rate_matrix(
        config, weights.rate, num_experts=num_experts, device=weights.rate.device
    )
    bits = _symmetric_bits(config, rates, uniform_bits=source.uniform_bits)
    offsets = _bundle_offsets(bits, hidden_size)
    required_row_bytes = max(
        offset
        + _matrix_section_bytes(hidden_size, int(bits[expert, projection]))
        for expert, expert_offsets in enumerate(offsets)
        for projection, offset in enumerate(expert_offsets)
    )
    if int(codes.shape[1]) != required_row_bytes:
        raise ValueError(
            f"codes row_stride={int(codes.shape[1])} does not match the "
            f"canonical projection payload ({required_row_bytes} bytes)"
        )

    def move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device.type != "cpu" and tensor.device != device:
            raise ValueError("trellis metadata must reside on CPU or the destination device")
        return tensor.to(device=device)

    def move_scales(factors: ScaleFactors) -> ScaleFactors:
        return ScaleFactors(move_tensor(factors.vectors), move_tensor(factors.gains))

    weights = replace(
        weights,
        input_scales=move_scales(weights.input_scales),
        intermediate_scales=move_scales(weights.intermediate_scales),
        output_scales=move_scales(weights.output_scales),
        expert_sign_patterns=move_tensor(weights.expert_sign_patterns),
    )
    scale = config.scale
    gate_suh, up_suh = _effective_input_scales(
        weights.input_scales,
        scale.input_scales,
        num_experts=num_experts,
        hidden_size=hidden_size,
        device=device,
    )
    intermediate = _effective_intermediate_scales(
        weights.intermediate_scales,
        scale.intermediate_scales,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        device=device,
    )
    down_svh = _effective_output_scales(
        weights.output_scales,
        scale.output_scales,
        num_experts=num_experts,
        hidden_size=hidden_size,
        device=device,
    )

    if uniform:
        return _uniform_prepared(
            config,
            weights,
            bits,
            activation=activation,
            params_dtype=params_dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate=intermediate,
            down_svh=down_svh,
            extent=source.extent,
            staging=staging,
            tile_config=tile_config or (64, 256, 64, 256),
            dummy_scale=dummy_scale,
            workspace=workspace,
        )
    return _projection_prepared(
        config,
        weights,
        bits,
        params_dtype=params_dtype,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate=intermediate,
        down_svh=down_svh,
    )


__all__ = ["PreparedProjectionTrellisWeights", "prepare_trellis_weights"]
