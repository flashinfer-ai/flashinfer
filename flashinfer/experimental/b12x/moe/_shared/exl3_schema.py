"""EXL3 checkpoint container schema.

EXL3 is the TP-shard-independent checkpoint container for trellis-coded MoE
expert weights. Storage is organized around 32-channel *slots* on the
intermediate axis: every slot row holds all experts' code words for those
channels, so a tensor-parallel rank loads a contiguous slot range and
nothing else. All behavior is declared in a manifest — codebook, rate
structure, intermediate Hadamard transform, and geometry — and the reader
derives byte addressing purely from those declarations.

Storage schema id: ``exl3-v1``. A checkpoint directory contains
``exl3-manifest.json`` plus one ``exl3-layer-<NNNNN>.safetensors`` file per
MoE layer. Per-layer tensors:

- ``codes``: u8 ``[num_slots, row_stride]`` — trellis code words only,
  expert-id-major bundles per row, zero padding to the row stride.
- ``rotations``: fp16 ``[num_slots, num_experts, 3, slot_channels]`` —
  per-channel intermediate-boundary values for gate/up/down in physical
  channel order.
- ``rates_fc1``/``rates_fc2``: u8 ``[num_slots/8, num_experts]`` — one
  rate byte per (256-channel pair, expert); present iff the rate structure
  is ``per_expert_pair``.
- ``gate_suh``/``up_suh``/``down_svh``: fp16 ``[hidden_size]`` or
  ``[num_experts, hidden_size]`` — hidden-axis incoherence values.
- ``sign_pattern``: u8 ``[num_experts]`` in ``0..7`` — present iff the
  intermediate Hadamard transform is declared.

A rate byte is ``(low_bits << 4) | high_bits`` and is exactly the fused
kernel's pair-kind vocabulary expressed as data. Uniform checkpoints carry
no rate tables; their single bitrate is declared in the manifest.

Within one ``codes`` row, expert bundles are concatenated in expert-id
order; each bundle is gate ‖ up ‖ down and each matrix section stores its
low-record plane followed by its high-record plane (``[H/16][16*low]i16``
‖ ``[H/16][16*high]i16``). Under a uniform rate structure the two planes
are the slot's two consecutive N16 (FC1) or K16 (FC2) tiles.

This module is torch-free: manifest parsing, fail-closed validation, extent
legality, and byte arithmetic. Tensor I/O and preparation live with the
W4A16 kernel host code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trellis_codebooks import (
    CODEBOOKS,
    MCG,
    MCG_MULTIPLIER,
    validate_codebook_bits,
)

EXL3_SCHEMA = "exl3-v1"
EXL3_MANIFEST_KIND = "exl3-manifest"
EXL3_MANIFEST_FILENAME = "exl3-manifest.json"

SLOT_CHANNELS = 32
SLOTS_PER_PAIR = 8

RATE_STRUCTURE_UNIFORM = "uniform"
RATE_STRUCTURE_PER_EXPERT_PAIR = "per_expert_pair"

# The fused kernel's pair-kind vocabulary as rate bytes.
RATE_CODE_PAIR_KINDS: dict[int, str] = {
    0x22: "P22",
    0x33: "P33",
    0x24: "P24",
    0x43: "P43",
    0x44: "P44",
}
PAIR_KIND_RATE_CODES: dict[str, int] = {
    kind: code for code, kind in RATE_CODE_PAIR_KINDS.items()
}


def layer_filename(layer_index: int) -> str:
    return f"exl3-layer-{int(layer_index):05d}.safetensors"


def rate_code(low_bits: int, high_bits: int) -> int:
    return (int(low_bits) << 4) | int(high_bits)


def rate_code_bits(code: int) -> tuple[int, int]:
    return (int(code) >> 4) & 0xF, int(code) & 0xF


def matrix_slot_bytes(hidden_size: int, low_bits: int, high_bits: int) -> int:
    """Trellis bytes one slot contributes to one expert matrix.

    A slot holds two 16-channel record planes; each plane stores
    ``hidden_size/16`` tiles of ``16*bits`` int16 words.
    """

    return (int(hidden_size) // 16) * 32 * (int(low_bits) + int(high_bits))


def bundle_bytes(
    hidden_size: int, fc1_code: int, fc2_code: int
) -> int:
    """Per-(expert, slot) bundle size: gate ‖ up ‖ down trellis words."""

    fc1_low, fc1_high = rate_code_bits(fc1_code)
    fc2_low, fc2_high = rate_code_bits(fc2_code)
    return 2 * matrix_slot_bytes(hidden_size, fc1_low, fc1_high) + (
        matrix_slot_bytes(hidden_size, fc2_low, fc2_high)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_keys(
    mapping: dict, *, required: set[str], optional: set[str], where: str
) -> None:
    _require(isinstance(mapping, dict), f"{where} must be a JSON object")
    keys = set(mapping.keys())
    unknown = keys - required - optional
    _require(not unknown, f"{where} has unknown keys {sorted(unknown)}")
    missing = required - keys
    _require(not missing, f"{where} is missing keys {sorted(missing)}")


@dataclass(frozen=True)
class Exl3Geometry:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    slot_channels: int
    num_slots: int
    moe_layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class Exl3Rates:
    structure: str
    bits: int | None
    pair_kinds: frozenset[str] | None

    def uniform_code(self) -> int | None:
        if self.structure != RATE_STRUCTURE_UNIFORM:
            return None
        assert self.bits is not None
        return rate_code(self.bits, self.bits)


@dataclass(frozen=True)
class Exl3Hadamard:
    intermediate_hadamard: bool
    pre_block: int | None
    post_block: int | None
    per_expert_input_rotations: bool


@dataclass(frozen=True)
class Exl3Layout:
    row_alignment: int
    extent_alignment_slots: int
    extent_barriers: tuple[int, ...]


@dataclass(frozen=True)
class Exl3LayerRef:
    file: str
    sha256: str


@dataclass(frozen=True)
class Exl3Manifest:
    codebook: str
    codebook_seed: int | None
    geometry: Exl3Geometry
    rates: Exl3Rates
    hadamard: Exl3Hadamard
    layout: Exl3Layout
    layers: dict[int, Exl3LayerRef]

    @staticmethod
    def from_dict(data: dict) -> "Exl3Manifest":
        _require_keys(
            data,
            required={
                "kind",
                "schema",
                "codebook",
                "geometry",
                "rates",
                "hadamard",
                "layout",
                "layers",
            },
            optional={"codebook_seed"},
            where="EXL3 manifest",
        )
        _require(
            data["kind"] == EXL3_MANIFEST_KIND,
            f"EXL3 manifest kind must be {EXL3_MANIFEST_KIND!r}, "
            f"got {data['kind']!r}",
        )
        _require(
            data["schema"] == EXL3_SCHEMA,
            f"EXL3 manifest schema must be {EXL3_SCHEMA!r}, got {data['schema']!r}",
        )

        codebook = data["codebook"]
        _require(
            codebook in CODEBOOKS,
            f"EXL3 codebook must be one of {sorted(CODEBOOKS)}, got {codebook!r}",
        )
        seed = data.get("codebook_seed")
        if codebook == MCG:
            _require(
                isinstance(seed, int) and seed == MCG_MULTIPLIER,
                "EXL3 mcg checkpoints must declare codebook_seed "
                f"{MCG_MULTIPLIER:#010x}",
            )
        else:
            _require(
                seed is None,
                f"EXL3 codebook_seed is valid only for mcg, not {codebook!r}",
            )

        geometry = _parse_geometry(data["geometry"])
        rates = _parse_rates(data["rates"], codebook=codebook)
        hadamard = _parse_hadamard(data["hadamard"], geometry=geometry)
        layout = _parse_layout(data["layout"], geometry=geometry)
        layers = _parse_layers(data["layers"], geometry=geometry)
        return Exl3Manifest(
            codebook=codebook,
            codebook_seed=seed,
            geometry=geometry,
            rates=rates,
            hadamard=hadamard,
            layout=layout,
            layers=layers,
        )

    def validate_extent(self, first_slot: int, slot_count: int) -> None:
        """Reject rank extents the layout declarations make illegal."""

        alignment = self.layout.extent_alignment_slots
        slots = self.geometry.num_slots
        _require(
            slot_count > 0 and first_slot >= 0,
            f"EXL3 extent [{first_slot}, {first_slot + slot_count}) is empty "
            "or negative",
        )
        _require(
            first_slot + slot_count <= slots,
            f"EXL3 extent [{first_slot}, {first_slot + slot_count}) exceeds "
            f"{slots} slots",
        )
        _require(
            first_slot % alignment == 0 and slot_count % alignment == 0,
            f"EXL3 extent [{first_slot}, {first_slot + slot_count}) must "
            f"align to {alignment} slots",
        )
        for barrier in self.layout.extent_barriers:
            _require(
                not (first_slot < barrier < first_slot + slot_count),
                f"EXL3 extent [{first_slot}, {first_slot + slot_count}) "
                f"crosses the declared barrier at slot {barrier}",
            )


def _parse_geometry(data: dict) -> Exl3Geometry:
    _require_keys(
        data,
        required={
            "num_experts",
            "hidden_size",
            "intermediate_size",
            "slot_channels",
            "num_slots",
            "moe_layer_indices",
        },
        optional=set(),
        where="EXL3 geometry",
    )
    for name in (
        "num_experts",
        "hidden_size",
        "intermediate_size",
        "slot_channels",
        "num_slots",
    ):
        _require(
            isinstance(data[name], int) and data[name] > 0,
            f"EXL3 geometry {name} must be a positive integer",
        )
    _require(
        data["slot_channels"] == SLOT_CHANNELS,
        f"EXL3 slots hold {SLOT_CHANNELS} channels; got {data['slot_channels']}",
    )
    _require(
        data["intermediate_size"] % data["slot_channels"] == 0,
        "EXL3 intermediate_size must be a multiple of slot_channels",
    )
    _require(
        data["num_slots"] * data["slot_channels"] == data["intermediate_size"],
        "EXL3 num_slots must equal intermediate_size / slot_channels",
    )
    _require(
        data["hidden_size"] % 16 == 0,
        "EXL3 hidden_size must be a multiple of 16",
    )
    indices = data["moe_layer_indices"]
    _require(
        isinstance(indices, list)
        and len(indices) > 0
        and all(isinstance(i, int) and i >= 0 for i in indices)
        and len(set(indices)) == len(indices),
        "EXL3 moe_layer_indices must be distinct non-negative integers",
    )
    return Exl3Geometry(
        num_experts=data["num_experts"],
        hidden_size=data["hidden_size"],
        intermediate_size=data["intermediate_size"],
        slot_channels=data["slot_channels"],
        num_slots=data["num_slots"],
        moe_layer_indices=tuple(sorted(indices)),
    )


def _parse_rates(data: dict, *, codebook: str) -> Exl3Rates:
    _require_keys(
        data,
        required={"structure"},
        optional={"bits", "pair_kinds"},
        where="EXL3 rates",
    )
    structure = data["structure"]
    if structure == RATE_STRUCTURE_UNIFORM:
        _require(
            "bits" in data and "pair_kinds" not in data,
            "uniform EXL3 rates declare bits and no pair_kinds",
        )
        bits = data["bits"]
        _require(
            isinstance(bits, int) and bits in (2, 3, 4, 5, 6),
            f"EXL3 uniform bits must be one of 2..6, got {bits!r}",
        )
        validate_codebook_bits(codebook, bits)
        return Exl3Rates(structure=structure, bits=bits, pair_kinds=None)
    if structure == RATE_STRUCTURE_PER_EXPERT_PAIR:
        _require(
            "pair_kinds" in data and "bits" not in data,
            "per_expert_pair EXL3 rates declare pair_kinds and no bits",
        )
        kinds = data["pair_kinds"]
        _require(
            isinstance(kinds, list)
            and len(kinds) > 0
            and all(kind in PAIR_KIND_RATE_CODES for kind in kinds)
            and len(set(kinds)) == len(kinds),
            "EXL3 pair_kinds must be distinct members of "
            f"{sorted(PAIR_KIND_RATE_CODES)}",
        )
        for kind in kinds:
            for bits in rate_code_bits(PAIR_KIND_RATE_CODES[kind]):
                validate_codebook_bits(codebook, bits)
        return Exl3Rates(
            structure=structure, bits=None, pair_kinds=frozenset(kinds)
        )
    raise ValueError(
        "EXL3 rates structure must be 'uniform' or 'per_expert_pair', "
        f"got {structure!r}"
    )


def _parse_hadamard(data: dict, *, geometry: Exl3Geometry) -> Exl3Hadamard:
    _require_keys(
        data,
        required={"intermediate_hadamard", "per_expert_input_rotations"},
        optional={"pre_block", "post_block"},
        where="EXL3 hadamard",
    )
    intermediate_hadamard = data["intermediate_hadamard"]
    _require(
        isinstance(intermediate_hadamard, bool), "EXL3 hadamard intermediate_hadamard must be a boolean"
    )
    per_expert = data["per_expert_input_rotations"]
    _require(
        isinstance(per_expert, bool),
        "EXL3 per_expert_input_rotations must be a boolean",
    )
    if not intermediate_hadamard:
        _require(
            "pre_block" not in data and "post_block" not in data,
            "EXL3 hadamard blocks are valid only when intermediate_hadamard is true",
        )
        return Exl3Hadamard(
            intermediate_hadamard=False,
            pre_block=None,
            post_block=None,
            per_expert_input_rotations=per_expert,
        )
    _require(
        "pre_block" in data and "post_block" in data,
        "EXL3 checkpoints with an intermediate Hadamard must declare pre_block and post_block",
    )
    pre_block, post_block = data["pre_block"], data["post_block"]
    for name, value in (("pre_block", pre_block), ("post_block", post_block)):
        _require(
            isinstance(value, int) and value > 0 and value % SLOT_CHANNELS == 0,
            f"EXL3 hadamard {name} must be a positive multiple of "
            f"{SLOT_CHANNELS}",
        )
    _require(
        geometry.intermediate_size % post_block == 0,
        "EXL3 intermediate_size must be a multiple of post_block",
    )
    _require(
        geometry.hidden_size % pre_block == 0,
        "EXL3 hidden_size must be a multiple of pre_block",
    )
    return Exl3Hadamard(
        intermediate_hadamard=True,
        pre_block=pre_block,
        post_block=post_block,
        per_expert_input_rotations=per_expert,
    )


def _parse_layout(data: dict, *, geometry: Exl3Geometry) -> Exl3Layout:
    _require_keys(
        data,
        required={"row_alignment", "extent_alignment_slots"},
        optional={"extent_barriers"},
        where="EXL3 layout",
    )
    alignment = data["row_alignment"]
    _require(
        isinstance(alignment, int) and alignment > 0,
        "EXL3 row_alignment must be a positive integer",
    )
    extent_alignment = data["extent_alignment_slots"]
    _require(
        isinstance(extent_alignment, int)
        and extent_alignment > 0
        and geometry.num_slots % extent_alignment == 0,
        "EXL3 extent_alignment_slots must be a positive divisor of num_slots",
    )
    barriers = data.get("extent_barriers", [])
    _require(
        isinstance(barriers, list)
        and all(
            isinstance(b, int) and 0 < b < geometry.num_slots
            for b in barriers
        )
        and len(set(barriers)) == len(barriers),
        "EXL3 extent_barriers must be distinct interior slot indices",
    )
    return Exl3Layout(
        row_alignment=alignment,
        extent_alignment_slots=extent_alignment,
        extent_barriers=tuple(sorted(barriers)),
    )


def _parse_layers(data: dict, *, geometry: Exl3Geometry) -> dict[int, Exl3LayerRef]:
    _require(isinstance(data, dict) and data, "EXL3 layers must be non-empty")
    layers: dict[int, Exl3LayerRef] = {}
    for key, value in data.items():
        _require(
            isinstance(key, str) and key.isdigit(),
            f"EXL3 layer keys must be decimal strings, got {key!r}",
        )
        index = int(key)
        _require_keys(
            value,
            required={"file", "sha256"},
            optional=set(),
            where=f"EXL3 layer {index}",
        )
        _require(
            isinstance(value["file"], str)
            and isinstance(value["sha256"], str)
            and len(value["sha256"]) == 64,
            f"EXL3 layer {index} must declare file and hex sha256",
        )
        layers[index] = Exl3LayerRef(file=value["file"], sha256=value["sha256"])
    _require(
        set(layers.keys()) == set(geometry.moe_layer_indices),
        "EXL3 layers must cover exactly geometry.moe_layer_indices",
    )
    return layers
