"""Low-level EXL3 compatibility and asymmetric-pair preparation.

Uniform extents forward through the public checkpoint adapter and common
trellis preparation. Pair extents retain their distinct record addressing;
they are not coerced into the canonical uniform-rate API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from b12x.moe._shared.exl3_schema import (
    SLOTS_PER_PAIR,
    RATE_CODE_PAIR_KINDS,
    RATE_STRUCTURE_UNIFORM,
    matrix_slot_bytes,
    rate_code,
    rate_code_bits,
)

if TYPE_CHECKING:
    from b12x.moe._shared.kernels.w4a16.prepare import PreparedW4A16MoeWeights

from b12x.moe.checkpoints.exl3 import (
    Exl3Layer,
    read_exl3_layer as read_exl3_layer,
    read_exl3_manifest as read_exl3_manifest,
    trellis_from_exl3,
)


def _extent_rotation_tables(
    layer: Exl3Layer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """suh/svh device tensors plus [E, 3*I_local] boundary values."""

    experts = layer.manifest.geometry.num_experts
    local = layer.local_intermediate_size
    values = layer.rotations.to(device=device)
    columns = [
        values[:, :, matrix, :].permute(1, 0, 2).reshape(experts, local)
        for matrix in range(3)
    ]
    intermediate = torch.cat(columns, dim=1).contiguous()

    def _side(tensor: torch.Tensor) -> torch.Tensor:
        moved = tensor.to(device=device)
        if moved.dim() == 1:
            moved = moved.reshape(1, -1)
        return moved.contiguous()

    return (
        _side(layer.gate_suh),
        _side(layer.up_suh),
        _side(layer.down_svh),
        intermediate,
    )


def prepare_exl3_moe_weights(
    layer: Exl3Layer,
    *,
    activation: str,
    device: torch.device | str,
    params_dtype: torch.dtype = torch.float16,
    tile_config: tuple[int, int, int, int] | None = None,
    dummy_scale: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> PreparedW4A16MoeWeights:
    """Prepare an EXL3 extent; uniform payloads use common trellis preparation."""

    manifest = layer.manifest
    device = torch.device(device)
    if manifest.rates.structure == RATE_STRUCTURE_UNIFORM:
        from b12x.moe.fused_moe.trellis import prepare_trellis_weights

        source, weights = trellis_from_exl3(layer)
        return prepare_trellis_weights(
            source, weights,
            activation=activation, params_dtype=params_dtype,
            num_experts=manifest.geometry.num_experts,
            hidden_size=manifest.geometry.hidden_size,
            intermediate_size=layer.local_intermediate_size,
            device=device,
            tile_config=tile_config or (
                (128, 128, 128, 128)
                if manifest.hadamard.intermediate_hadamard and manifest.rates.bits == 2
                else (64, 256, 64, 256)
            ),
            dummy_scale=dummy_scale, workspace=workspace,
        )
    gate, up, down, rotations = _extent_rotation_tables(layer, device)
    return _prepare_exl3_pair_extent(
        layer, device=device, gate_suh=gate, up_suh=up, down_svh=down,
        rotations=rotations, params_dtype=params_dtype,
        tile_config=tile_config or (64, 256, 64, 256),
        dummy_scale=dummy_scale, workspace=workspace,
    )


def _prepare_exl3_pair_extent(
    layer: Exl3Layer,
    *,
    device: torch.device,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    rotations: torch.Tensor,
    params_dtype: torch.dtype,
    tile_config: tuple[int, int, int, int],
    dummy_scale: torch.Tensor | None,
    workspace: torch.Tensor | None,
) -> PreparedW4A16MoeWeights:
    """Prepare a per-expert-pair extent through the pair kernel machinery.

    The fused kernel's pair decode operates on one 256-channel pair per
    rank (FC2 pairs lie on the local K axis), so a per-expert-rate extent
    is exactly one pair of slots.
    """

    from b12x.moe._shared.kernels.w4a16.prepare import (
        _finalize_prepared_trellis_weights,
        _restore_plane_words,
    )

    manifest = layer.manifest
    geometry = manifest.geometry
    if layer.slot_count != SLOTS_PER_PAIR:
        raise ValueError(
            "per-expert-pair EXL3 extents must cover exactly one "
            f"256-channel pair ({SLOTS_PER_PAIR} slots); got "
            f"{layer.slot_count}"
        )
    if manifest.hadamard.intermediate_hadamard:
        raise ValueError(
            "intermediate-Hadamard execution of per-expert-pair EXL3 extents has "
            "no qualified kernel path"
        )
    # The pair runtime orders each 256-channel pair record-major: every
    # slot contributes its first 16 channels to the low record and its
    # last 16 to the high record. Rotation rows must match that order.
    experts_count = geometry.num_experts
    per_matrix = []
    values = layer.rotations.to(rotations.device)
    for matrix in range(3):
        planes = values[:, :, matrix, :].reshape(
            SLOTS_PER_PAIR, experts_count, 2, 16
        )
        low = planes[:, :, 0, :].permute(1, 0, 2).reshape(experts_count, -1)
        high = planes[:, :, 1, :].permute(1, 0, 2).reshape(experts_count, -1)
        per_matrix.append(torch.cat((low, high), dim=1))
    rotations = torch.cat(per_matrix, dim=1).contiguous()
    assert layer.rates_fc1 is not None and layer.rates_fc2 is not None
    fc1_codes = layer.rates_fc1[0].to(torch.int64)
    fc2_codes = layer.rates_fc2[0].to(torch.int64)
    experts = geometry.num_experts
    hidden_tiles = geometry.hidden_size // 16

    kinds = {
        RATE_CODE_PAIR_KINDS[int(code)]
        for code in torch.cat((fc1_codes, fc2_codes)).unique().tolist()
    }
    if kinds == {"P33"} or kinds == {"P33", "P24"}:
        pair_kind = "PDYNAMIC"
        high_code = rate_code(2, 4)
    elif kinds == {"P33", "P43"}:
        pair_kind = "P33_P43"
        high_code = rate_code(4, 3)
    else:
        raise ValueError(
            f"EXL3 per-expert extents with pair kinds {sorted(kinds)} have "
            "no fused execution arm; whole-expert K4 tiers run through "
            "mixed-tier or multi-launch execution"
        )

    def _restore(codes: torch.Tensor, matrix: int, *, fc1: bool):
        sections = []
        for expert in range(experts):
            low_bits, high_bits = rate_code_bits(int(codes[expert]))
            begin = 0
            for m in range(matrix):
                m_codes = fc1_codes if m < 2 else fc2_codes
                lo, hi = rate_code_bits(int(m_codes[expert]))
                begin += matrix_slot_bytes(geometry.hidden_size, lo, hi)
            section = matrix_slot_bytes(geometry.hidden_size, low_bits, high_bits)
            raw = layer.codes[:, _bundle_offset(layer, expert) + begin :][
                :, :section
            ]
            words = (
                raw.contiguous()
                .to(device=device)
                .view(torch.int16)
                .reshape(SLOTS_PER_PAIR, 1, -1)
                .permute(1, 0, 2)
            )
            low_words = hidden_tiles * 16 * low_bits
            low = words[..., :low_words].reshape(
                1, SLOTS_PER_PAIR, hidden_tiles, 16 * low_bits
            )
            high = words[..., low_words:].reshape(
                1, SLOTS_PER_PAIR, hidden_tiles, 16 * high_bits
            )
            sections.append(_restore_plane_words(low, high, fc1=fc1))
        return sections

    if pair_kind == "PDYNAMIC":
        fc1_modes = (fc1_codes == high_code).to(torch.int32).to(device)
        fc2_modes = (fc2_codes == high_code).to(torch.int32).to(device)
        gate = _restore(fc1_codes, 0, fc1=True)
        up = _restore(fc1_codes, 1, fc1=True)
        down = _restore(fc2_codes, 2, fc1=False)
        w13 = torch.cat(
            [torch.cat(gate, dim=0), torch.cat(up, dim=0)]
        ).reshape(-1)
        w2 = torch.cat(down, dim=0).reshape(-1)
        fc1_pair_modes: torch.Tensor = fc1_modes.contiguous()
        fc2_pair_modes: torch.Tensor = fc2_modes.contiguous()
    else:
        # Compact gap-free pools with per-expert descriptors, matching the
        # fused kernel's P33_P43 addressing.
        def _compact(codes, sections):
            lengths = torch.tensor(
                [section.numel() // 2 for section in sections],
                dtype=torch.int64,
            )
            offsets = torch.zeros_like(lengths)
            offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
            modes = (codes == high_code).to(torch.int64)
            descriptors = ((offsets << 1) | modes).to(device)
            return offsets, descriptors

        gate = _restore(fc1_codes, 0, fc1=True)
        up = _restore(fc1_codes, 1, fc1=True)
        down = _restore(fc2_codes, 2, fc1=False)
        _, fc1_descriptors = _compact(fc1_codes, gate)
        _, fc2_descriptors = _compact(fc2_codes, down)
        w13 = torch.cat(
            [torch.cat(gate, dim=1), torch.cat(up, dim=1)]
        ).reshape(-1)
        w2 = torch.cat(down, dim=1).reshape(-1)
        fc1_pair_modes = fc1_descriptors.contiguous()
        fc2_pair_modes = fc2_descriptors.contiguous()

    return _finalize_prepared_trellis_weights(
        context="EXL3 per-expert-pair preparation",
        device=device,
        hidden_size=geometry.hidden_size,
        intermediate_size=layer.local_intermediate_size,
        num_experts=experts,
        params_dtype=params_dtype,
        w13=w13,
        w2=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=rotations,
        down_svh=down_svh,
        rotation_columns=rotations.shape[1],
        tile_config=tile_config,
        required_fc1_tile_n=256,
        dummy_scale=dummy_scale,
        workspace=workspace,
        codebook=manifest.codebook,
        trellis_bits=3,
        fc1_pair_kind=pair_kind,
        fc2_pair_kind=pair_kind,
        fc1_pair_modes=fc1_pair_modes,
        fc2_pair_modes=fc2_pair_modes,
        intermediate_hadamard=manifest.hadamard.intermediate_hadamard,
    )


def _bundle_offset(layer: Exl3Layer, expert: int) -> int:
    """Byte offset of one expert's bundle within this extent's rows."""

    manifest = layer.manifest
    if manifest.rates.structure == RATE_STRUCTURE_UNIFORM:
        assert manifest.rates.bits is not None
        section = matrix_slot_bytes(
            manifest.geometry.hidden_size,
            manifest.rates.bits,
            manifest.rates.bits,
        )
        return expert * 3 * section
    assert layer.rates_fc1 is not None and layer.rates_fc2 is not None
    offset = 0
    for e in range(expert):
        fc1_lo, fc1_hi = rate_code_bits(int(layer.rates_fc1[0, e]))
        fc2_lo, fc2_hi = rate_code_bits(int(layer.rates_fc2[0, e]))
        offset += 2 * matrix_slot_bytes(
            manifest.geometry.hidden_size, fc1_lo, fc1_hi
        ) + matrix_slot_bytes(manifest.geometry.hidden_size, fc2_lo, fc2_hi)
    return offset
