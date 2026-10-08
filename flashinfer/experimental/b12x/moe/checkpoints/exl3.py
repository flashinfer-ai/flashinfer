"""Normalize uniform ``exl3-v1`` slot extents without requantization.

Only this adapter interprets manifest row padding and interleaved FC1 side
tables. Preparation and execution consume ordinary TrellisSource/Weights.
Per-expert-pair containers remain available through the low-level reader but
are not accepted by this uniform-rate adapter.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import torch

from b12x.moe._shared.exl3_schema import (
    SLOTS_PER_PAIR,
    EXL3_MANIFEST_FILENAME,
    EXL3_SCHEMA,
    Exl3Manifest,
    RATE_CODE_PAIR_KINDS,
    RATE_STRUCTURE_PER_EXPERT_PAIR,
)
from b12x.moe.fused_moe.config import TrellisConfig
from b12x.moe.fused_moe.source import TrellisExtent, TrellisSource
from b12x.moe.fused_moe.weights import ScaleFactors, TrellisWeights


@dataclass(frozen=True)
class Exl3Layer:
    """One layer's extent-sliced content, validated against the manifest."""

    manifest: Exl3Manifest
    layer_index: int
    first_slot: int
    slot_count: int
    codes: torch.Tensor
    rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    rates_fc1: torch.Tensor | None
    rates_fc2: torch.Tensor | None
    sign_pattern: torch.Tensor | None

    @property
    def local_intermediate_size(self) -> int:
        return self.slot_count * self.manifest.geometry.slot_channels


def read_exl3_manifest(root: str | pathlib.Path) -> Exl3Manifest:
    root = pathlib.Path(root)
    data = json.loads((root / EXL3_MANIFEST_FILENAME).read_text())
    return Exl3Manifest.from_dict(data)


def read_exl3_layer(
    root: str | pathlib.Path,
    manifest: Exl3Manifest,
    layer_index: int,
    *,
    first_slot: int,
    slot_count: int,
    verify_sha: bool = False,
) -> Exl3Layer:
    """Load one rank extent of one layer as CPU tensors.

    The extent is validated against the manifest's layout declarations and
    the safetensors metadata is cross-checked against the manifest before
    any tensor is interpreted.
    """

    from safetensors import safe_open

    root = pathlib.Path(root)
    manifest.validate_extent(first_slot, slot_count)
    if layer_index not in manifest.layers:
        raise ValueError(f"EXL3 manifest does not declare layer {layer_index}")
    ref = manifest.layers[layer_index]
    path = root / ref.file
    if verify_sha:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ref.sha256:
            raise ValueError(
                f"EXL3 layer {layer_index} sha256 mismatch: manifest "
                f"{ref.sha256}, file {digest}"
            )

    geometry = manifest.geometry
    per_expert = manifest.rates.structure == RATE_STRUCTURE_PER_EXPERT_PAIR
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        expected = {
            "schema": EXL3_SCHEMA,
            "codebook": manifest.codebook,
            "layer": str(int(layer_index)),
            "num_experts": str(geometry.num_experts),
            "hidden_size": str(geometry.hidden_size),
            "intermediate_size": str(geometry.intermediate_size),
            "slot_channels": str(geometry.slot_channels),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"EXL3 layer {layer_index} metadata {key!r} is "
                    f"{metadata.get(key)!r}; the manifest declares {value!r}"
                )
        names = set(handle.keys())
        required = {"codes", "rotations", "gate_suh", "up_suh", "down_svh"}
        if per_expert:
            required |= {"rates_fc1", "rates_fc2"}
        if manifest.hadamard.intermediate_hadamard:
            required |= {"sign_pattern"}
        if names != required:
            raise ValueError(
                f"EXL3 layer {layer_index} tensors {sorted(names)} do not "
                f"match the declared set {sorted(required)}"
            )

        codes_slice = handle.get_slice("codes")
        codes_shape = codes_slice.get_shape()
        if (
            len(codes_shape) != 2
            or codes_shape[0] != geometry.num_slots
            or codes_shape[1] % manifest.layout.row_alignment
        ):
            raise ValueError(
                f"EXL3 layer {layer_index} codes shape {codes_shape} violates "
                "the declared geometry or row alignment"
            )
        codes = codes_slice[first_slot : first_slot + slot_count]

        rotations_slice = handle.get_slice("rotations")
        if tuple(rotations_slice.get_shape()) != (
            geometry.num_slots,
            geometry.num_experts,
            3,
            geometry.slot_channels,
        ):
            raise ValueError(
                f"EXL3 layer {layer_index} rotations shape is not "
                "[num_slots, num_experts, 3, slot_channels]"
            )
        rotations = rotations_slice[first_slot : first_slot + slot_count]

        h_shapes = (
            (geometry.num_experts, geometry.hidden_size)
            if manifest.hadamard.per_expert_input_rotations
            else (geometry.hidden_size,)
        )
        sides = {}
        for name in ("gate_suh", "up_suh", "down_svh"):
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != h_shapes or tensor.dtype != torch.float16:
                raise ValueError(
                    f"EXL3 layer {layer_index} {name} must be fp16 {h_shapes}"
                )
            sides[name] = tensor

        rates_fc1 = rates_fc2 = None
        if per_expert:
            pairs = geometry.num_slots // SLOTS_PER_PAIR
            first_pair = first_slot // SLOTS_PER_PAIR
            pair_count = slot_count // SLOTS_PER_PAIR
            declared = manifest.rates.pair_kinds or frozenset()
            observed: set[str] = set()
            tables = {}
            for name in ("rates_fc1", "rates_fc2"):
                table_slice = handle.get_slice(name)
                if tuple(table_slice.get_shape()) != (
                    pairs,
                    geometry.num_experts,
                ):
                    raise ValueError(
                        f"EXL3 layer {layer_index} {name} must be "
                        "[num_slots/8, num_experts]"
                    )
                table = table_slice[first_pair : first_pair + pair_count]
                for code in table.unique().tolist():
                    kind = RATE_CODE_PAIR_KINDS.get(int(code))
                    if kind is None:
                        raise ValueError(
                            f"EXL3 layer {layer_index} {name} contains "
                            f"unknown rate code {int(code):#x}"
                        )
                    observed.add(kind)
                tables[name] = table
            if not observed <= set(declared):
                raise ValueError(
                    f"EXL3 layer {layer_index} rate tables use kinds "
                    f"{sorted(observed)} outside the declared "
                    f"{sorted(declared)}"
                )
            rates_fc1, rates_fc2 = tables["rates_fc1"], tables["rates_fc2"]

        sign_pattern = None
        if manifest.hadamard.intermediate_hadamard:
            sign_pattern = handle.get_tensor("sign_pattern")
            if (
                tuple(sign_pattern.shape) != (geometry.num_experts,)
                or sign_pattern.dtype != torch.uint8
                or bool(torch.any(sign_pattern > 7))
            ):
                raise ValueError(
                    f"EXL3 layer {layer_index} sign_pattern must be "
                    "uint8[num_experts] in 0..7"
                )

    return Exl3Layer(
        manifest=manifest,
        layer_index=layer_index,
        first_slot=first_slot,
        slot_count=slot_count,
        codes=codes,
        rotations=rotations,
        gate_suh=sides["gate_suh"],
        up_suh=sides["up_suh"],
        down_svh=sides["down_svh"],
        rates_fc1=rates_fc1,
        rates_fc2=rates_fc2,
        sign_pattern=sign_pattern,
    )


def trellis_from_exl3(layer: Exl3Layer) -> tuple[TrellisSource, TrellisWeights]:
    """Validate a CPU extent and expose codeword views plus FP16 scale tables.

    The payload is not moved to CUDA here. ``prepare_weights(device=...)``
    controls destination ownership and bounded staging. No codeword or scale
    is numerically converted.
    """

    if not isinstance(layer, Exl3Layer):
        raise TypeError("layer must be an Exl3Layer")
    manifest = layer.manifest
    manifest.validate_extent(layer.first_slot, layer.slot_count)
    if manifest.rates.structure != "uniform":
        raise NotImplementedError("the trellis adapter requires uniform EXL3 rates")
    bits = manifest.rates.bits
    assert bits is not None
    geometry = manifest.geometry
    experts, hidden = geometry.num_experts, geometry.hidden_size
    local = layer.local_intermediate_size
    payload = experts * 3 * (hidden // 16) * 64 * bits
    codes = layer.codes
    if (
        codes.device.type != "cpu"
        or codes.dtype != torch.uint8
        or codes.ndim != 2
        or codes.shape[0] != layer.slot_count
        or codes.shape[1] < payload
        or codes.stride(1) != 1
    ):
        raise ValueError(
            "EXL3 codes must be CPU uint8 slot rows containing every expert"
        )
    if bool(torch.any(codes[:, payload:] != 0)):
        raise ValueError("EXL3 codes row padding must be zero")
    if layer.rotations.shape != (layer.slot_count, experts, 3, 32):
        raise ValueError("EXL3 rotations must have shape [slots,experts,3,32]")
    for name in ("rotations", "gate_suh", "up_suh", "down_svh"):
        tensor = getattr(layer, name)
        if tensor.dtype != torch.float16 or tensor.device.type != "cpu":
            raise ValueError(f"EXL3 {name} must be an FP16 CPU tensor")
    per_expert = manifest.hadamard.per_expert_input_rotations
    side_shape = (experts, hidden) if per_expert else (hidden,)
    if any(
        getattr(layer, name).shape != side_shape
        for name in ("gate_suh", "up_suh", "down_svh")
    ):
        raise ValueError(f"EXL3 hidden-axis tables must have shape {side_shape}")
    gate, up = layer.gate_suh, layer.up_suh
    expert_transform: dict[str, object] = {"kind": "none"}
    if manifest.hadamard.intermediate_hadamard:
        half = geometry.num_slots // 2
        if layer.first_slot < half < layer.first_slot + layer.slot_count:
            raise ValueError(
                "EXL3 transformed extent cannot cross the FC1 side-table boundary"
            )
        if (
            layer.sign_pattern is None
            or layer.sign_pattern.dtype != torch.uint8
            or layer.sign_pattern.device.type != "cpu"
            or layer.sign_pattern.shape != (experts,)
            or bool(torch.any(layer.sign_pattern > 7))
        ):
            raise ValueError("EXL3 sign_pattern must be CPU uint8 [experts] in 0..7")
        # The stored length-2I transform interleaves gate/up within each half.
        # Both physical FC1 projections therefore use the same source side table.
        gate = up = gate if layer.first_slot < half else up
        expert_transform = {
            "kind": "intermediate_hadamard",
            "pre_block_size": manifest.hadamard.pre_block,
            "post_block_size": manifest.hadamard.post_block,
            "sign_pattern_granularity": "per_expert",
        }
    elif layer.sign_pattern is not None:
        raise ValueError("EXL3 sign_pattern is invalid without an expert transform")
    side_granularity = "per_expert" if per_expert else "per_layer"
    config = TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": manifest.codebook,
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {"vectors": side_granularity, "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": side_granularity, "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": expert_transform,
            },
        }
    )
    return (
        TrellisSource(
            config=config,
            uniform_bits=bits,
            extent=TrellisExtent(
                global_intermediate_size=geometry.intermediate_size,
                first_slot=layer.first_slot,
                slot_count=layer.slot_count,
            ),
        ),
        TrellisWeights(
            codes=codes[:, :payload],
            rate=torch.tensor([bits * 17], dtype=torch.uint8),
            input_scales=ScaleFactors(
                gate.contiguous() if gate is up else torch.stack((gate, up), dim=-2)
            ),
            intermediate_scales=ScaleFactors(
                layer.rotations.permute(1, 2, 0, 3)
                .reshape(experts, 3, local)
                .contiguous()
            ),
            output_scales=ScaleFactors(layer.down_svh.contiguous()),
            expert_sign_patterns=layer.sign_pattern,
        ),
    )


__all__ = ["read_exl3_manifest", "read_exl3_layer", "trellis_from_exl3"]
