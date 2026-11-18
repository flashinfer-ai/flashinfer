"""CPU qualification of lossless slot assembly and EXL3 normalization.

These tests validate physical layouts, not CUDA kernel execution. The CUDA
preparation/execution/graph comparisons are in test_exl3_prepare.py.
"""

from dataclasses import replace

import pytest
import torch

from b12x.moe._shared.exl3_schema import Exl3Manifest
from b12x.moe.checkpoints.exl3 import Exl3Layer, read_exl3_layer, trellis_from_exl3
from b12x.moe.fused_moe.source import TrellisExtent, TrellisSource
from b12x.moe.fused_moe.trellis import (
    _effective_input_scales,
    _effective_intermediate_scales,
    _effective_output_scales,
    _local_rate_matrix,
    _symmetric_bits,
)
from b12x.moe.fused_moe.trellis_layout import (
    TrellisStaging,
    append_intermediate_signs,
    assemble_uniform_slots,
)


def _layer(*, first=48, slots=12, experts=3, bits=2, per_expert=True, hadamard=True):
    hidden, global_i = 512, 3072
    had = {"intermediate_hadamard": hadamard, "per_expert_input_rotations": per_expert}
    if hadamard:
        had.update(pre_block=512, post_block=128)
    manifest = Exl3Manifest.from_dict(
        {
            "kind": "exl3-manifest",
            "schema": "exl3-v1",
            "codebook": "lut_e4m3",
            "geometry": {
                "num_experts": experts,
                "hidden_size": hidden,
                "intermediate_size": global_i,
                "slot_channels": 32,
                "num_slots": global_i // 32,
                "moe_layer_indices": [1],
            },
            "rates": {"structure": "uniform", "bits": bits},
            "hadamard": had,
            "layout": {
                "row_alignment": 65536,
                "extent_alignment_slots": 4,
                "extent_barriers": [48],
            },
            "layers": {
                "1": {"file": "exl3-layer-00001.safetensors", "sha256": "0" * 64}
            },
        }
    )
    payload = experts * 3 * (hidden // 16) * 64 * bits
    row_bytes = (payload + 65535) // 65536 * 65536
    codes = torch.zeros((slots, row_bytes), dtype=torch.uint8)
    codes[:, :payload] = torch.randint(0, 256, (slots, payload), dtype=torch.uint8)
    side_shape = (experts, hidden) if per_expert else (hidden,)
    return Exl3Layer(
        manifest=manifest,
        layer_index=1,
        first_slot=first,
        slot_count=slots,
        codes=codes,
        rotations=torch.randn(slots, experts, 3, 32).half(),
        gate_suh=torch.randn(side_shape).half(),
        up_suh=torch.randn(side_shape).half(),
        down_svh=torch.randn(side_shape).half(),
        rates_fc1=None,
        rates_fc2=None,
        sign_pattern=(torch.arange(experts) % 8).byte() if hadamard else None,
    )


def _independent_planes(layer):
    """Explicit slot/expert/matrix addressing, independent of batched assembly."""
    experts, hidden = (
        layer.manifest.geometry.num_experts,
        layer.manifest.geometry.hidden_size,
    )
    bits, slots = layer.manifest.rates.bits, layer.slot_count
    tiles = hidden // 16
    w13 = torch.empty(2, experts, tiles, 2 * slots, 16 * bits, dtype=torch.int16)
    w2 = torch.empty(experts, 2 * slots, tiles, 16 * bits, dtype=torch.int16)
    section = tiles * 64 * bits
    for slot in range(slots):
        for expert in range(experts):
            for matrix in range(3):
                start = (expert * 3 + matrix) * section
                words = (
                    layer.codes[slot, start : start + section]
                    .contiguous()
                    .view(torch.int16)
                )
                for half in range(2):
                    plane = words[
                        half * tiles * 16 * bits : (half + 1) * tiles * 16 * bits
                    ]
                    if matrix < 2:
                        w13[matrix, expert, :, 2 * slot + half] = plane.view(
                            tiles, 16 * bits
                        )
                    else:
                        w2[expert, 2 * slot + half] = plane.view(tiles, 16 * bits)
    return w13, w2


@pytest.mark.parametrize("first,slots", [(0, 8), (8, 4), (48, 12), (60, 8)])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("batch", [1, 2, 64])
def test_uniform_assembly_is_bit_exact(first, slots, bits, batch):
    layer = _layer(first=first, slots=slots, bits=bits)
    source, weights = trellis_from_exl3(layer)
    assert weights.codes.data_ptr() == layer.codes.data_ptr()
    before = layer.codes.clone()
    actual = assemble_uniform_slots(
        weights.codes,
        num_experts=3,
        hidden_size=512,
        bits=bits,
        device="cpu",
        staging=TrellisStaging(max_experts=batch),
    )
    for result, expected in zip(actual, _independent_planes(layer)):
        assert torch.equal(result, expected)
    assert torch.equal(before, layer.codes)
    assert source.extent.first_slot == first
    assert source.uniform_bits == bits
    assert source.config.to_dict()["version"] == 2


@pytest.mark.parametrize("first", [0, 8, 48, 60])
@pytest.mark.parametrize("per_expert", [False, True])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("hadamard", [False, True])
def test_side_and_intermediate_tables_are_bit_exact(
    first, per_expert, experts, hadamard
):
    layer = _layer(
        first=first, slots=8, experts=experts, per_expert=per_expert, hadamard=hadamard
    )
    source, weights = trellis_from_exl3(layer)
    scales = source.config.scale
    gate, up = _effective_input_scales(
        weights.input_scales,
        scales.input_scales,
        num_experts=experts,
        hidden_size=512,
        device=torch.device("cpu"),
    )
    expected_gate, expected_up = layer.gate_suh, layer.up_suh
    if hadamard:
        expected_gate = expected_up = layer.gate_suh if first < 48 else layer.up_suh
    assert torch.equal(gate, expected_gate.reshape(-1, 512))
    assert torch.equal(up, expected_up.reshape(-1, 512))
    if hadamard:
        assert gate.data_ptr() == up.data_ptr()
        assert weights.input_scales.vectors.numel() == expected_gate.numel()
    intermediate = _effective_intermediate_scales(
        weights.intermediate_scales,
        scales.intermediate_scales,
        num_experts=experts,
        intermediate_size=256,
        device=torch.device("cpu"),
    )
    expected = torch.stack(
        [
            torch.cat([layer.rotations[s, e] for s in range(8)], dim=1)
            for e in range(experts)
        ]
    )
    assert torch.equal(intermediate, expected)
    down = _effective_output_scales(
        weights.output_scales,
        scales.output_scales,
        num_experts=experts,
        hidden_size=512,
        device=torch.device("cpu"),
    )
    assert torch.equal(down, layer.down_svh.reshape(-1, 512))


@pytest.mark.parametrize("first,slots", [(0, 8), (8, 4), (48, 12), (60, 8)])
def test_signs_are_sliced_from_global_sequence(first, slots):
    layer = _layer(first=first, slots=slots, experts=8)
    source, weights = trellis_from_exl3(layer)
    local = slots * 32
    values = weights.intermediate_scales.vectors.reshape(8, 3 * local)
    actual = append_intermediate_signs(
        values, weights.expert_sign_patterns, extent=source.extent
    )
    assert torch.equal(actual[:, : 3 * local], values)
    for pattern in range(8):
        signs = []
        for axis, width, offset, size in (
            (1, 6144, first * 64, local * 2),
            (2, 3072, first * 32, local),
        ):
            if pattern == 0:
                full = torch.ones(width)
            else:
                gen = torch.Generator().manual_seed(
                    (0x6A09E667F3BCC909 * pattern + 0xBB67AE8584CAA73B * axis) % (2**63)
                )
                full = torch.randint(2, (width,), generator=gen) * 2 - 1
            signs.append(full[offset : offset + size])
        assert torch.equal(actual[pattern, 3 * local :], torch.cat(signs).half())


def test_staging_limit_selects_complete_experts_and_rejects_insufficient_budget():
    layer = _layer()
    _, weights = trellis_from_exl3(layer)
    one = layer.slot_count * 3 * 32 * 64 * 2
    staging = TrellisStaging(max_bytes=one * 2 + 7)
    assert staging.batch_size(slots=12, hidden_size=512, bits=2) == 2
    actual = assemble_uniform_slots(
        weights.codes,
        num_experts=3,
        hidden_size=512,
        bits=2,
        device="cpu",
        staging=staging,
    )
    for result, expected in zip(actual, _independent_planes(layer)):
        assert torch.equal(result, expected)
    with pytest.raises(ValueError, match="insufficient"):
        assemble_uniform_slots(
            weights.codes,
            num_experts=3,
            hidden_size=512,
            bits=2,
            device="cpu",
            staging=TrellisStaging(max_bytes=one - 1),
        )


@pytest.mark.parametrize(
    "change,message",
    [
        ("padding", "padding"),
        ("rotation_dtype", "FP16"),
        ("sign_pattern", "sign_pattern"),
        ("short_codes", "containing every expert"),
    ],
)
def test_adapter_rejects_corrupt_payload(change, message):
    layer = _layer()
    if change == "padding":
        layer.codes[0, -1] = 1
    elif change == "rotation_dtype":
        layer = replace(layer, rotations=layer.rotations.bfloat16())
    elif change == "sign_pattern":
        layer.sign_pattern[0] = 8
    else:
        layer = replace(layer, codes=layer.codes[:, :100])
    with pytest.raises(ValueError, match=message):
        trellis_from_exl3(layer)


def test_preparation_verifies_declared_uniform_rate():
    source, weights = trellis_from_exl3(_layer())
    rates = _local_rate_matrix(
        source.config, weights.rate, num_experts=3, device=torch.device("cpu")
    )
    assert torch.equal(
        _symmetric_bits(source.config, rates, uniform_bits=2), torch.full((3, 3), 2)
    )
    with pytest.raises(ValueError, match="observed"):
        _symmetric_bits(source.config, rates, uniform_bits=3)
    with pytest.raises(ValueError, match="symmetric"):
        _symmetric_bits(
            source.config, torch.tensor([[0x23]], dtype=torch.uint8), uniform_bits=2
        )


@pytest.mark.parametrize("first,count", [(-1, 4), (96, 4), (4, 0)])
def test_extent_rejects_invalid_bounds(first, count):
    with pytest.raises(ValueError):
        TrellisExtent(global_intermediate_size=3072, first_slot=first, slot_count=count)


def test_uniform_rate_rejects_undefined_codebook_range():
    source, _ = trellis_from_exl3(_layer())
    with pytest.raises(ValueError, match="K2/K3/K4"):
        TrellisSource(config=source.config, uniform_bits=5)


@pytest.mark.parametrize("first,slots", [(0, 8), (8, 4), (48, 12), (60, 8)])
def test_safetensors_reader_and_adapter_keep_extent_bytes(tmp_path, first, slots):
    from safetensors.torch import save_file

    full = _layer(first=0, slots=96)
    tensors = {
        name: getattr(full, name)
        for name in (
            "codes",
            "rotations",
            "gate_suh",
            "up_suh",
            "down_svh",
            "sign_pattern",
        )
    }
    path = tmp_path / full.manifest.layers[1].file
    save_file(
        tensors,
        path,
        metadata={
            "schema": "exl3-v1",
            "codebook": "lut_e4m3",
            "layer": "1",
            "num_experts": "3",
            "hidden_size": "512",
            "intermediate_size": "3072",
            "slot_channels": "32",
        },
    )
    loaded = read_exl3_layer(
        tmp_path, full.manifest, 1, first_slot=first, slot_count=slots
    )
    assert torch.equal(loaded.codes, full.codes[first : first + slots])
    assert torch.equal(loaded.rotations, full.rotations[first : first + slots])
    _, weights = trellis_from_exl3(loaded)
    for actual, expected in zip(
        assemble_uniform_slots(
            weights.codes, num_experts=3, hidden_size=512, bits=2, device="cpu"
        ),
        _independent_planes(loaded),
    ):
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_exl3_layer(
            tmp_path,
            full.manifest,
            1,
            first_slot=first,
            slot_count=slots,
            verify_sha=True,
        )
