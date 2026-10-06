"""Synthetic EXL3 checkpoint writer for tests and benchmarks.

Generates deterministic, schema-complete `exl3-v1` checkpoints
(manifest plus per-layer safetensors) from random trellis words. This is a
fixture generator, not a converter: it exists so the reader, planner, and
serving paths can be exercised without a quantizer, and so one packer
implementation is shared by every test and benchmark.

The per-(expert, slot) plane words are exposed as an intermediate
representation (`Exl3LayerPayloads`) so equivalence tests can feed the same
logical weights to other packers and compare prepared tensors byte for
byte.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field

import torch

from b12x.moe._shared.exl3_schema import (
    SLOT_CHANNELS,
    SLOTS_PER_PAIR,
    EXL3_MANIFEST_FILENAME,
    EXL3_MANIFEST_KIND,
    EXL3_SCHEMA,
    Exl3Manifest,
    RATE_CODE_PAIR_KINDS,
    RATE_STRUCTURE_PER_EXPERT_PAIR,
    RATE_STRUCTURE_UNIFORM,
    layer_filename,
    matrix_slot_bytes,
    rate_code,
    rate_code_bits,
)
from b12x.moe._shared.trellis_codebooks import MCG, MCG_MULTIPLIER


@dataclass(frozen=True)
class Exl3SynthConfig:
    """Declarations for one synthetic checkpoint."""

    codebook: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    moe_layer_indices: tuple[int, ...]
    # Uniform structure declares bits; per-expert structure declares tables
    # via ``rate_tables`` below.
    bits: int | None = None
    # Optional per-layer {layer: (rates_fc1, rates_fc2)} u8 tables of shape
    # [num_slots/8, num_experts]. Present iff bits is None.
    rate_tables: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    intermediate_hadamard: bool = False
    pre_block: int | None = None
    post_block: int | None = None
    per_expert_input_rotations: bool = False
    # Unit hidden-axis tables for routes that fold no input-side rotation.
    unit_hidden_rotations: bool = False
    row_alignment: int = 4096
    extent_alignment_slots: int = 4
    extent_barriers: tuple[int, ...] = ()
    seed: int = 0

    @property
    def num_slots(self) -> int:
        return self.intermediate_size // SLOT_CHANNELS


@dataclass
class Exl3LayerPayloads:
    """One layer's logical content before row assembly.

    ``planes[(expert, slot, matrix)]`` is ``(low_plane, high_plane)``,
    each an int16 tensor ``[hidden_size/16, 16*bits_plane]``. Matrices are
    indexed 0=gate, 1=up, 2=down.
    """

    planes: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]
    rotations: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    down_svh: torch.Tensor
    sign_pattern: torch.Tensor | None
    rates_fc1: torch.Tensor | None
    rates_fc2: torch.Tensor | None
    metadata: dict[str, str] = field(default_factory=dict)


def _layer_rate_codes(
    config: Exl3SynthConfig, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(pair, expert) rate-code tables, synthesized for uniform rates."""

    pairs = config.num_slots // SLOTS_PER_PAIR
    if config.bits is not None:
        code = rate_code(config.bits, config.bits)
        table = torch.full(
            (pairs, config.num_experts), code, dtype=torch.uint8
        )
        return table, table
    assert config.rate_tables is not None
    rates_fc1, rates_fc2 = config.rate_tables[layer]
    expected = (pairs, config.num_experts)
    for name, table in (("rates_fc1", rates_fc1), ("rates_fc2", rates_fc2)):
        if tuple(table.shape) != expected or table.dtype != torch.uint8:
            raise ValueError(
                f"{name} must be uint8 {expected}, got "
                f"{table.dtype} {tuple(table.shape)}"
            )
        for code in table.unique().tolist():
            if int(code) not in RATE_CODE_PAIR_KINDS:
                raise ValueError(f"{name} contains unknown rate code {code:#x}")
    return rates_fc1, rates_fc2


def synth_layer_payloads(
    config: Exl3SynthConfig, layer: int
) -> Exl3LayerPayloads:
    """Deterministic random content for one layer."""

    generator = torch.Generator().manual_seed(
        (config.seed << 20) ^ (layer * 2654435761 % (1 << 31))
    )
    hidden_tiles = config.hidden_size // 16
    experts = config.num_experts
    slots = config.num_slots
    rates_fc1, rates_fc2 = _layer_rate_codes(config, layer)

    def _plane(bits: int) -> torch.Tensor:
        return torch.randint(
            -(1 << 15),
            1 << 15,
            (hidden_tiles, 16 * bits),
            dtype=torch.int16,
            generator=generator,
        )

    planes: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for expert in range(experts):
        for slot in range(slots):
            pair = slot // SLOTS_PER_PAIR
            fc1_low, fc1_high = rate_code_bits(int(rates_fc1[pair, expert]))
            fc2_low, fc2_high = rate_code_bits(int(rates_fc2[pair, expert]))
            planes[(expert, slot, 0)] = (_plane(fc1_low), _plane(fc1_high))
            planes[(expert, slot, 1)] = (_plane(fc1_low), _plane(fc1_high))
            planes[(expert, slot, 2)] = (_plane(fc2_low), _plane(fc2_high))

    def _values(shape: tuple[int, ...]) -> torch.Tensor:
        raw = torch.rand(shape, generator=generator, dtype=torch.float32)
        return (0.5 + raw).to(torch.float16)

    rotations = _values((slots, experts, 3, SLOT_CHANNELS))
    h_shape = (
        (experts, config.hidden_size)
        if config.per_expert_input_rotations
        else (config.hidden_size,)
    )

    def _h_values() -> torch.Tensor:
        if config.unit_hidden_rotations:
            return torch.ones(h_shape, dtype=torch.float16)
        return _values(h_shape)
    sign_patterns = None
    if config.intermediate_hadamard:
        sign_patterns = torch.randint(
            0, 8, (experts,), dtype=torch.uint8, generator=generator
        )
    uniform = config.bits is not None
    return Exl3LayerPayloads(
        planes=planes,
        rotations=rotations,
        gate_suh=_h_values(),
        up_suh=_h_values(),
        down_svh=_h_values(),
        sign_pattern=sign_patterns,
        rates_fc1=None if uniform else rates_fc1,
        rates_fc2=None if uniform else rates_fc2,
    )


def assemble_code_rows(
    config: Exl3SynthConfig, layer: int, payloads: Exl3LayerPayloads
) -> torch.Tensor:
    """Pack plane words into the expert-major, zero-padded ``codes`` tensor."""

    rates_fc1, rates_fc2 = _layer_rate_codes(config, layer)
    slots = config.num_slots
    row_bytes = []
    for slot in range(slots):
        pair = slot // SLOTS_PER_PAIR
        total = 0
        for expert in range(config.num_experts):
            total += 2 * matrix_slot_bytes(
                config.hidden_size, *rate_code_bits(int(rates_fc1[pair, expert]))
            ) + matrix_slot_bytes(
                config.hidden_size, *rate_code_bits(int(rates_fc2[pair, expert]))
            )
        row_bytes.append(total)
    alignment = config.row_alignment
    stride = (max(row_bytes) + alignment - 1) // alignment * alignment
    codes = torch.zeros((slots, stride), dtype=torch.uint8)
    for slot in range(slots):
        cursor = 0
        for expert in range(config.num_experts):
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                for plane in (low, high):
                    raw = plane.contiguous().view(torch.uint8).reshape(-1)
                    codes[slot, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
    return codes


def _manifest_dict(config: Exl3SynthConfig) -> dict:
    rates: dict[str, object] = {}
    if config.bits is not None:
        rates = {"structure": RATE_STRUCTURE_UNIFORM, "bits": config.bits}
    else:
        kinds: set[str] = set()
        assert config.rate_tables is not None
        for rates_fc1, rates_fc2 in config.rate_tables.values():
            for table in (rates_fc1, rates_fc2):
                kinds.update(
                    RATE_CODE_PAIR_KINDS[int(code)]
                    for code in table.unique().tolist()
                )
        rates = {
            "structure": RATE_STRUCTURE_PER_EXPERT_PAIR,
            "pair_kinds": sorted(kinds),
        }
    hadamard: dict[str, object] = {
        "intermediate_hadamard": config.intermediate_hadamard,
        "per_expert_input_rotations": config.per_expert_input_rotations,
    }
    if config.intermediate_hadamard:
        hadamard["pre_block"] = config.pre_block
        hadamard["post_block"] = config.post_block
    manifest: dict[str, object] = {
        "kind": EXL3_MANIFEST_KIND,
        "schema": EXL3_SCHEMA,
        "codebook": config.codebook,
        "geometry": {
            "num_experts": config.num_experts,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "slot_channels": SLOT_CHANNELS,
            "num_slots": config.num_slots,
            "moe_layer_indices": list(config.moe_layer_indices),
        },
        "rates": rates,
        "hadamard": hadamard,
        "layout": {
            "row_alignment": config.row_alignment,
            "extent_alignment_slots": config.extent_alignment_slots,
            "extent_barriers": list(config.extent_barriers),
        },
        "layers": {},
    }
    if config.codebook == MCG:
        manifest["codebook_seed"] = MCG_MULTIPLIER
    return manifest


def layer_metadata(config: Exl3SynthConfig, layer: int) -> dict[str, str]:
    return {
        "schema": EXL3_SCHEMA,
        "codebook": config.codebook,
        "layer": str(int(layer)),
        "num_experts": str(config.num_experts),
        "hidden_size": str(config.hidden_size),
        "intermediate_size": str(config.intermediate_size),
        "slot_channels": str(SLOT_CHANNELS),
    }


def write_exl3_checkpoint(
    root: str | pathlib.Path, config: Exl3SynthConfig
) -> Exl3Manifest:
    """Write a complete synthetic checkpoint and return its parsed manifest."""

    from safetensors.torch import save_file

    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_dict(config)
    for layer in config.moe_layer_indices:
        payloads = synth_layer_payloads(config, layer)
        tensors: dict[str, torch.Tensor] = {
            "codes": assemble_code_rows(config, layer, payloads),
            "rotations": payloads.rotations,
            "gate_suh": payloads.gate_suh,
            "up_suh": payloads.up_suh,
            "down_svh": payloads.down_svh,
        }
        if payloads.rates_fc1 is not None:
            assert payloads.rates_fc2 is not None
            tensors["rates_fc1"] = payloads.rates_fc1
            tensors["rates_fc2"] = payloads.rates_fc2
        if payloads.sign_pattern is not None:
            tensors["sign_pattern"] = payloads.sign_pattern
        filename = layer_filename(layer)
        save_file(
            tensors, str(root / filename), metadata=layer_metadata(config, layer)
        )
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        manifest["layers"][str(int(layer))] = {
            "file": filename,
            "sha256": digest,
        }
    (root / EXL3_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return Exl3Manifest.from_dict(manifest)
