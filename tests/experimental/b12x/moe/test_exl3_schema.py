"""Fail-closed validation tests for the exl3-v1 container schema."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from b12x.moe._shared.exl3_schema import (
    SLOT_CHANNELS,
    EXL3_MANIFEST_FILENAME,
    Exl3Manifest,
    bundle_bytes,
    matrix_slot_bytes,
    rate_code,
)
from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig,
    assemble_code_rows,
    synth_layer_payloads,
    write_exl3_checkpoint,
)


def _uniform_manifest_dict(
    *,
    codebook: str = "lut_e4m3",
    bits: int = 2,
    intermediate_hadamard: bool = False,
) -> dict:
    hadamard: dict = {
        "intermediate_hadamard": intermediate_hadamard,
        "per_expert_input_rotations": False,
    }
    if intermediate_hadamard:
        hadamard["pre_block"] = 512
        hadamard["post_block"] = 128
    data: dict = {
        "kind": "exl3-manifest",
        "schema": "exl3-v1",
        "codebook": codebook,
        "geometry": {
            "num_experts": 8,
            "hidden_size": 512,
            "intermediate_size": 512,
            "slot_channels": 32,
            "num_slots": 16,
            "moe_layer_indices": [1, 2],
        },
        "rates": {"structure": "uniform", "bits": bits},
        "hadamard": hadamard,
        "layout": {
            "row_alignment": 4096,
            "extent_alignment_slots": 4,
            "extent_barriers": [8],
        },
        "layers": {
            "1": {"file": "exl3-layer-00001.safetensors", "sha256": "0" * 64},
            "2": {"file": "exl3-layer-00002.safetensors", "sha256": "0" * 64},
        },
    }
    if codebook == "mcg":
        data["codebook_seed"] = 0xCBAC1FED
    return data


def _per_expert_manifest_dict() -> dict:
    data = _uniform_manifest_dict(bits=3)
    data["rates"] = {
        "structure": "per_expert_pair",
        "pair_kinds": ["P33", "P43"],
    }
    return data


def test_uniform_manifest_parses() -> None:
    manifest = Exl3Manifest.from_dict(_uniform_manifest_dict())
    assert manifest.codebook == "lut_e4m3"
    assert manifest.rates.uniform_code() == rate_code(2, 2)
    assert manifest.geometry.num_slots == 16
    assert manifest.layout.extent_barriers == (8,)


@pytest.mark.parametrize("bits", [2, 3])
def test_canonical_weight_plan_preserves_exl3_transform_and_rate(bits: int) -> None:
    from b12x.moe import fused_moe
    from tests.experimental.b12x.moe.test_trellis_adapter import _layer
    from b12x.moe.checkpoints.exl3 import trellis_from_exl3

    source, _ = trellis_from_exl3(_layer(first=8, slots=4, bits=bits))
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=3, hidden_size=512, intermediate_size=128
        ),
    )
    assert plan.prepared_format.weights is fused_moe.WeightEncoding.TRELLIS
    assert plan.prepared_format.packing is fused_moe.WeightPacking.TRELLIS_NATIVE
    assert plan._impl.trellis_bits == bits
    assert plan._impl.intermediate_hadamard_blocks == (512, 128)
    assert plan.geometry.intermediate_size == 128
    assert plan.source.extent.global_intermediate_size == 3072
    assert plan._impl.source_format == "b12x_trellis"


@pytest.mark.parametrize(
    ("mode", "experts", "intermediate", "message"),
    [
        ("a8", 8, 256, "A16 activations"),
        ("a16", 8, 1024, "extent width differs"),
    ],
)
def test_canonical_exl3_plan_rejects_incompatible_contract(
    mode: str, experts: int, intermediate: int, message: str
) -> None:
    from b12x.moe import fused_moe
    from tests.experimental.b12x.moe.test_trellis_adapter import _layer
    from b12x.moe.checkpoints.exl3 import trellis_from_exl3

    source, _ = trellis_from_exl3(_layer(first=0, slots=8, experts=8))
    with pytest.raises(ValueError, match=message):
        fused_moe.plan_weights(
            source=source,
            activation=fused_moe.ActivationSpec(
                mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=experts, hidden_size=512, intermediate_size=intermediate
            ),
        )


def test_per_expert_manifest_parses() -> None:
    manifest = Exl3Manifest.from_dict(_per_expert_manifest_dict())
    assert manifest.rates.pair_kinds == frozenset({"P33", "P43"})
    assert manifest.rates.uniform_code() is None


def test_mcg_manifest_requires_seed() -> None:
    data = _uniform_manifest_dict(codebook="mcg", bits=3)
    Exl3Manifest.from_dict(data)
    missing = copy.deepcopy(data)
    del missing["codebook_seed"]
    with pytest.raises(ValueError, match="codebook_seed"):
        Exl3Manifest.from_dict(missing)
    wrong = copy.deepcopy(data)
    wrong["codebook_seed"] = 1
    with pytest.raises(ValueError, match="codebook_seed"):
        Exl3Manifest.from_dict(wrong)


def test_lut_manifest_forbids_seed() -> None:
    data = _uniform_manifest_dict()
    data["codebook_seed"] = 0xCBAC1FED
    with pytest.raises(ValueError, match="only for mcg"):
        Exl3Manifest.from_dict(data)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(kind="unknown-manifest"), "kind"),
        (lambda d: d.update(schema="exl3-v2"), "schema"),
        (lambda d: d.update(codebook="unregistered_codebook"), "codebook"),
        (lambda d: d.update(profile="k2"), "unknown keys"),
        (lambda d: d["geometry"].update(num_slots=15), "num_slots"),
        (lambda d: d["geometry"].update(slot_channels=16), "32 channels"),
        (lambda d: d["geometry"].update(hidden_size=250), "multiple of 16"),
        (lambda d: d["geometry"].update(rotation_multiplier=5), "unknown keys"),
        (lambda d: d["geometry"].update(moe_layer_indices=[1, 1]), "distinct"),
        (lambda d: d["rates"].update(bits=7), "2..6"),
        (lambda d: d["rates"].update(structure="profile"), "structure"),
        (
            lambda d: d["rates"].update(pair_kinds=["P33"]),
            "uniform EXL3 rates declare bits and no pair_kinds",
        ),
        (lambda d: d["hadamard"].update(pre_block=512), "intermediate_hadamard"),
        (lambda d: d["layout"].update(extent_alignment_slots=5), "divisor"),
        (lambda d: d["layout"].update(extent_barriers=[0]), "interior"),
        (lambda d: d["layers"].pop("2"), "moe_layer_indices"),
        (
            lambda d: d["layers"]["1"].update(sha256="ff"),
            "hex sha256",
        ),
    ],
)
def test_manifest_fails_closed(mutate, match) -> None:
    data = _uniform_manifest_dict()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        Exl3Manifest.from_dict(data)


def test_lut_bit_ranges_fail_closed() -> None:
    with pytest.raises(ValueError, match="lut_e4m3"):
        Exl3Manifest.from_dict(_uniform_manifest_dict(bits=5))
    with pytest.raises(ValueError, match="lut_fp16"):
        Exl3Manifest.from_dict(_uniform_manifest_dict(codebook="lut_fp16", bits=3))
    Exl3Manifest.from_dict(_uniform_manifest_dict(codebook="lut_fp16", bits=6))


def test_per_expert_rates_fail_closed() -> None:
    data = _per_expert_manifest_dict()
    data["rates"]["pair_kinds"] = ["P33", "P35"]
    with pytest.raises(ValueError, match="pair_kinds"):
        Exl3Manifest.from_dict(data)
    data = _per_expert_manifest_dict()
    data["rates"]["bits"] = 3
    with pytest.raises(ValueError, match="no bits"):
        Exl3Manifest.from_dict(data)


def test_intermediate_hadamard_manifest_block_rules() -> None:
    manifest = Exl3Manifest.from_dict(
        _uniform_manifest_dict(intermediate_hadamard=True)
    )
    assert manifest.hadamard.pre_block == 512
    assert manifest.hadamard.post_block == 128
    data = _uniform_manifest_dict(intermediate_hadamard=True)
    del data["hadamard"]["post_block"]
    with pytest.raises(ValueError, match="pre_block and post_block"):
        Exl3Manifest.from_dict(data)
    data = _uniform_manifest_dict(intermediate_hadamard=True)
    data["hadamard"]["post_block"] = 48
    with pytest.raises(ValueError, match="multiple of 32"):
        Exl3Manifest.from_dict(data)


def test_extent_validation() -> None:
    manifest = Exl3Manifest.from_dict(_uniform_manifest_dict())
    manifest.validate_extent(0, 8)
    manifest.validate_extent(8, 8)
    manifest.validate_extent(12, 4)
    with pytest.raises(ValueError, match="align"):
        manifest.validate_extent(2, 4)
    with pytest.raises(ValueError, match="align"):
        manifest.validate_extent(0, 6)
    with pytest.raises(ValueError, match="exceeds"):
        manifest.validate_extent(12, 8)
    with pytest.raises(ValueError, match="barrier"):
        manifest.validate_extent(4, 8)


def test_bundle_byte_arithmetic_matches_frozen_sizes() -> None:
    # Kimi-K3 geometry: H=3584 reproduces the frozen per-slot matrix sizes.
    assert matrix_slot_bytes(3584, 2, 2) == 28_672
    assert matrix_slot_bytes(3584, 3, 3) == 43_008
    assert matrix_slot_bytes(3584, 4, 3) == 50_176
    assert matrix_slot_bytes(3584, 4, 4) == 57_344
    assert bundle_bytes(3584, rate_code(3, 3), rate_code(3, 3)) == 129_024


def _small_config(**overrides) -> Exl3SynthConfig:
    defaults = dict(
        codebook="lut_e4m3",
        num_experts=4,
        hidden_size=128,
        intermediate_size=512,
        moe_layer_indices=(1,),
        bits=2,
        row_alignment=4096,
        extent_alignment_slots=4,
        seed=7,
    )
    defaults.update(overrides)
    return Exl3SynthConfig(**defaults)


def test_synth_rows_are_expert_major_and_zero_padded() -> None:
    config = _small_config()
    payloads = synth_layer_payloads(config, 1)
    codes = assemble_code_rows(config, 1, payloads)
    assert codes.dtype == torch.uint8
    assert codes.shape[0] == config.num_slots
    assert codes.shape[1] % config.row_alignment == 0
    per_bundle = bundle_bytes(config.hidden_size, rate_code(2, 2), rate_code(2, 2))
    payload = per_bundle * config.num_experts
    assert bool(torch.all(codes[:, payload:] == 0))
    # Expert 1's gate low plane sits after expert 0's complete bundle.
    low, _high = payloads.planes[(1, 0, 0)]
    raw = low.contiguous().view(torch.uint8).reshape(-1)
    assert torch.equal(codes[0, per_bundle : per_bundle + raw.numel()], raw)


def test_write_exl3_checkpoint_round_trips(tmp_path) -> None:
    config = _small_config()
    manifest = write_exl3_checkpoint(tmp_path, config)
    assert manifest.rates.uniform_code() == rate_code(2, 2)
    on_disk = json.loads((tmp_path / EXL3_MANIFEST_FILENAME).read_text())
    reparsed = Exl3Manifest.from_dict(on_disk)
    assert reparsed == manifest

    from safetensors import safe_open

    ref = manifest.layers[1]
    with safe_open(str(tmp_path / ref.file), framework="pt") as handle:
        metadata = handle.metadata()
        names = set(handle.keys())
        rotations = handle.get_tensor("rotations")
    assert metadata["schema"] == "exl3-v1"
    assert metadata["codebook"] == "lut_e4m3"
    assert metadata["layer"] == "1"
    assert names == {"codes", "rotations", "gate_suh", "up_suh", "down_svh"}
    assert rotations.shape == (
        config.num_slots,
        config.num_experts,
        3,
        SLOT_CHANNELS,
    )


def test_write_exl3_checkpoint_intermediate_hadamard_and_per_expert(tmp_path) -> None:
    pairs = 512 // SLOT_CHANNELS // 8
    fc1 = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    fc1[0, 1] = rate_code(4, 3)
    fc2 = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    config = _small_config(
        bits=None,
        rate_tables={1: (fc1, fc2)},
        extent_alignment_slots=8,
    )
    manifest = write_exl3_checkpoint(tmp_path / "per-expert", config)
    assert manifest.rates.pair_kinds == frozenset({"P33", "P43"})

    intermediate_hadamard = _small_config(
        intermediate_hadamard=True,
        pre_block=512,
        post_block=128,
        seed=9,
        hidden_size=512,
    )
    manifest = write_exl3_checkpoint(
        tmp_path / "intermediate_hadamard", intermediate_hadamard
    )
    assert manifest.hadamard.intermediate_hadamard

    from safetensors import safe_open

    with safe_open(
        str(tmp_path / "intermediate_hadamard" / manifest.layers[1].file),
        framework="pt",
    ) as handle:
        assert "sign_pattern" in set(handle.keys())


def test_synth_rejects_unknown_rate_codes() -> None:
    pairs = 2
    bad = torch.full((pairs, 4), 0x53, dtype=torch.uint8)
    good = torch.full((pairs, 4), rate_code(3, 3), dtype=torch.uint8)
    config = _small_config(bits=None, rate_tables={1: (bad, good)})
    with pytest.raises(ValueError, match="unknown rate code"):
        synth_layer_payloads(config, 1)
