"""EXL3 preparation equivalence and fail-closed reader tests.

Independent-assembly checks reconstruct the runtime word order with
naive loops so the shared restore helper is never compared against
itself; frozen-container equivalence lives in test_exl3_compat.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from b12x.moe._shared.exl3_schema import (
    EXL3_MANIFEST_FILENAME,
    rate_code,
)
from b12x.moe._shared.kernels.w4a16.exl3 import (
    prepare_exl3_moe_weights,
    read_exl3_layer,
)
from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig,
    synth_layer_payloads,
    write_exl3_checkpoint,
)
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _plane_pair(payloads, expert: int, slot: int, matrix: int):
    return payloads.planes[(expert, slot, matrix)]


def _naive_fc1_words(low_planes, high_planes) -> torch.Tensor:
    """FC1 runtime order via explicit loops: per K16 tile, all low-plane
    windows slot-major, then all high-plane windows."""

    hidden_tiles = low_planes[0].shape[0]
    rows = []
    for kt in range(hidden_tiles):
        low = torch.cat([plane[kt] for plane in low_planes])
        high = torch.cat([plane[kt] for plane in high_planes])
        rows.append(torch.cat((low, high)))
    return torch.cat(rows)


def _naive_fc2_words(low_planes, high_planes) -> torch.Tensor:
    """FC2 runtime order: the K-major low planes, then the high planes."""

    low = torch.cat([plane.reshape(-1) for plane in low_planes])
    high = torch.cat([plane.reshape(-1) for plane in high_planes])
    return torch.cat((low, high))


@requires_cuda
def test_exl3_uniform_mcg_matches_direct_binder(tmp_path) -> None:
    hidden, global_i, experts, bits = 256, 512, 3, 3
    slots = global_i // 32
    config = Exl3SynthConfig(
        codebook="mcg",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(0,),
        bits=bits,
        per_expert_input_rotations=True,
        extent_alignment_slots=4,
        seed=5,
    )
    manifest = write_exl3_checkpoint(tmp_path, config)
    layer = read_exl3_layer(
        tmp_path, manifest, 0, first_slot=0, slot_count=slots
    )
    device = _device()
    exl3_prepared = prepare_exl3_moe_weights(
        layer, activation="silu", device=device
    )

    payloads = synth_layer_payloads(config, 0)
    hidden_tiles = hidden // 16
    w13 = torch.empty(
        (2, experts, hidden_tiles, 2 * slots, 16 * bits), dtype=torch.int16
    )
    w2 = torch.empty(
        (experts, 2 * slots, hidden_tiles, 16 * bits), dtype=torch.int16
    )
    for expert in range(experts):
        for slot in range(slots):
            for matrix in range(2):
                low, high = _plane_pair(payloads, expert, slot, matrix)
                w13[matrix, expert, :, 2 * slot] = low
                w13[matrix, expert, :, 2 * slot + 1] = high
            low, high = _plane_pair(payloads, expert, slot, 2)
            w2[expert, 2 * slot] = low
            w2[expert, 2 * slot + 1] = high

    rotations = torch.cat(
        [
            payloads.rotations[:, :, matrix, :]
            .permute(1, 0, 2)
            .reshape(experts, global_i)
            for matrix in range(3)
        ],
        dim=1,
    ).to(device)
    binder_prepared = prepare_trellis256_moe_weights(
        w13.to(device),
        w2.to(device),
        hidden_size=hidden,
        intermediate_size=global_i,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=256,
        fc2_tile_n=256,
        params_dtype=torch.float16,
        w13_layout="trellis_t256_proj",
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=payloads.gate_suh.to(device),
        up_suh=payloads.up_suh.to(device),
        intermediate_rotations=rotations,
        down_svh=payloads.down_svh.to(device),
        tile_config=(64, 256, 64, 256),
    )

    assert torch.equal(exl3_prepared.w13, binder_prepared.w13)
    assert torch.equal(exl3_prepared.w2, binder_prepared.w2)
    assert torch.equal(
        exl3_prepared.intermediate_rotations,
        binder_prepared.intermediate_rotations,
    )
    assert exl3_prepared.trellis_codebook == "mcg"
    assert exl3_prepared.source_format == "exl3"


def _per_expert_config(kinds: dict[int, tuple[int, int]]) -> Exl3SynthConfig:
    experts = len(kinds)
    fc1 = torch.tensor(
        [[rate_code(*kinds[e]) for e in range(experts)]], dtype=torch.uint8
    )
    fc2 = torch.full_like(fc1, rate_code(3, 3))
    return Exl3SynthConfig(
        codebook="lut_e4m3",
        num_experts=experts,
        hidden_size=256,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=None,
        rate_tables={0: (fc1, fc2)},
        extent_alignment_slots=8,
        seed=3,
    )


@requires_cuda
@pytest.mark.parametrize(
    "high_rates, expected_kind",
    [((2, 4), "PDYNAMIC"), ((4, 3), "P33_P43")],
)
def test_exl3_per_expert_pair_matches_naive_assembly(
    tmp_path, high_rates, expected_kind
) -> None:
    kinds = {0: (3, 3), 1: high_rates, 2: (3, 3)}
    config = _per_expert_config(kinds)
    manifest = write_exl3_checkpoint(tmp_path, config)
    layer = read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    device = _device()
    prepared = prepare_exl3_moe_weights(
        layer, activation="situ", device=device
    )
    assert prepared.fc1_trellis_pair_kind == expected_kind
    assert prepared.fc2_trellis_pair_kind == expected_kind

    payloads = synth_layer_payloads(config, 0)
    experts = config.num_experts

    def _matrix_planes(expert: int, matrix: int):
        lows, highs = [], []
        for slot in range(8):
            low, high = _plane_pair(payloads, expert, slot, matrix)
            lows.append(low)
            highs.append(high)
        return lows, highs

    expected_gate = [
        _naive_fc1_words(*_matrix_planes(e, 0)) for e in range(experts)
    ]
    expected_up = [
        _naive_fc1_words(*_matrix_planes(e, 1)) for e in range(experts)
    ]
    expected_down = [
        _naive_fc2_words(*_matrix_planes(e, 2)) for e in range(experts)
    ]
    expected_w13 = torch.cat(expected_gate + expected_up).to(device)
    expected_w2 = torch.cat(expected_down).to(device)
    assert torch.equal(prepared.w13.view(torch.int16), expected_w13)
    assert torch.equal(prepared.w2.view(torch.int16), expected_w2)

    # Rotation rows follow the pair runtime's record-major channel order:
    # each slot's first 16 channels belong to the low record, its last 16
    # to the high record.
    expected_rotations = []
    for matrix in range(3):
        planes = payloads.rotations[:, :, matrix, :].reshape(8, experts, 2, 16)
        low = planes[:, :, 0, :].permute(1, 0, 2).reshape(experts, -1)
        high = planes[:, :, 1, :].permute(1, 0, 2).reshape(experts, -1)
        expected_rotations.append(torch.cat((low, high), dim=1))
    assert torch.equal(
        prepared.intermediate_rotations,
        torch.cat(expected_rotations, dim=1).to(device),
    )

    modes = prepared.fc1_trellis_pair_modes
    assert modes is not None
    if expected_kind == "PDYNAMIC":
        assert modes.tolist() == [0, 1, 0]
        fc2_modes = prepared.fc2_trellis_pair_modes
        assert fc2_modes is not None and fc2_modes.tolist() == [0, 0, 0]
    else:
        # Descriptors: gap-free u32 offsets over the expert sections with
        # bit zero selecting the high-rate record pair.
        lengths = [section.numel() // 2 for section in expected_gate]
        offsets = [0, lengths[0], lengths[0] + lengths[1]]
        assert modes.tolist() == [
            (offsets[0] << 1) | 0,
            (offsets[1] << 1) | 1,
            (offsets[2] << 1) | 0,
        ]


def test_exl3_reader_fails_closed(tmp_path) -> None:
    config = Exl3SynthConfig(
        codebook="lut_e4m3",
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=2,
        extent_alignment_slots=4,
        seed=1,
    )
    manifest = write_exl3_checkpoint(tmp_path, config)
    read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)

    with pytest.raises(ValueError, match="align"):
        read_exl3_layer(tmp_path, manifest, 0, first_slot=1, slot_count=4)
    with pytest.raises(ValueError, match="does not declare layer"):
        read_exl3_layer(tmp_path, manifest, 7, first_slot=0, slot_count=4)

    # Manifest/metadata disagreement: tamper the manifest codebook (the
    # tampered value must still parse) and require the metadata cross-check
    # to reject the layer.
    data = json.loads((tmp_path / EXL3_MANIFEST_FILENAME).read_text())
    data["codebook"] = "mcg"
    data["codebook_seed"] = 0xCBAC1FED
    data["rates"]["bits"] = 3
    from b12x.moe._shared.exl3_schema import Exl3Manifest

    tampered = Exl3Manifest.from_dict(data)
    with pytest.raises(ValueError, match="metadata 'codebook'"):
        read_exl3_layer(tmp_path, tampered, 0, first_slot=0, slot_count=4)

    # sha verification.
    ref = manifest.layers[0]
    path = tmp_path / ref.file
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        read_exl3_layer(
            tmp_path, manifest, 0, first_slot=0, slot_count=4, verify_sha=True
        )


@requires_cuda
def test_exl3_uniform_padding_must_be_zero(tmp_path) -> None:
    config = Exl3SynthConfig(
        codebook="lut_e4m3",
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        moe_layer_indices=(0,),
        bits=2,
        extent_alignment_slots=4,
        seed=2,
    )
    manifest = write_exl3_checkpoint(tmp_path, config)
    layer = read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    corrupted = layer.codes.clone()
    corrupted[0, -1] = 1
    from dataclasses import replace

    with pytest.raises(ValueError, match="padding must be zero"):
        prepare_exl3_moe_weights(
            replace(layer, codes=corrupted),
            activation="situ",
            device=_device(),
        )


def _uniform_reference(layer, *, activation, device):
    """Independent word/rotation assembly, bound to the shared compute kernel.

    Neither the checkpoint adapter nor common trellis preparation participates
    in this oracle. Global sign coordinates and side-table selection are
    reconstructed from the source extent explicitly.
    """
    from tests.experimental.b12x.moe.test_trellis_adapter import _independent_planes

    geometry = layer.manifest.geometry
    experts, local = geometry.num_experts, layer.local_intermediate_size
    w13, w2 = _independent_planes(layer)
    rotations = torch.stack([
        torch.cat([layer.rotations[s, e] for s in range(layer.slot_count)], dim=1)
        for e in range(experts)
    ]).reshape(experts, 3 * local).to(device)
    gate, up = layer.gate_suh.to(device), layer.up_suh.to(device)
    hadamard = layer.manifest.hadamard.intermediate_hadamard
    if hadamard:
        gate = up = gate if layer.first_slot < geometry.num_slots // 2 else up
    tiles = (128, 128, 128, 128) if hadamard and layer.manifest.rates.bits == 2 else (64, 256, 64, 256)
    result = prepare_trellis256_moe_weights(
        w13.to(device), w2.to(device), hidden_size=geometry.hidden_size,
        intermediate_size=local, num_experts=experts, activation=activation,
        fc1_tile_n=tiles[1], fc2_tile_n=tiles[3], params_dtype=torch.float16,
        w13_layout="trellis_t256_proj", trellis_bits=layer.manifest.rates.bits,
        codebook=layer.manifest.codebook, gate_suh=gate, up_suh=up,
        intermediate_rotations=rotations, down_svh=layer.down_svh.to(device),
        tile_config=tiles,
    )
    if not hadamard:
        return result
    signs = []
    for pattern in layer.sign_pattern.tolist():
        parts = []
        for axis, multiplier in ((1, 2), (2, 1)):
            length = multiplier * geometry.intermediate_size
            begin = multiplier * 32 * layer.first_slot
            if pattern == 0:
                full = torch.ones(length)
            else:
                gen = torch.Generator().manual_seed(
                    (0x6A09E667F3BCC909 * pattern + 0xBB67AE8584CAA73B * axis) % (2**63)
                )
                full = torch.randint(2, (length,), generator=gen) * 2 - 1
            parts.append(full[begin:begin + multiplier * local])
        signs.append(torch.cat(parts))
    return replace(result, trellis=replace(
        result.trellis, intermediate_hadamard=True,
        intermediate_rotations=torch.cat((rotations, torch.stack(signs).half().to(device)), dim=1),
    ))


@requires_cuda
@pytest.mark.parametrize("first_slot, slots", [(0, 8), (8, 4), (48, 12), (60, 8)])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("capacity", [1, 8, 129])
def test_canonical_exl3_preparation_preserves_source_extent(
    tmp_path, first_slot, slots, activation, capacity
):
    """Uneven TP extents preserve rotations in preparation, execution and graphs."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.moe import fused_moe
    from b12x.moe.checkpoints.exl3 import trellis_from_exl3
    from b12x.moe._shared.kernels.w4a16.exl3 import read_exl3_layer
    from b12x.moe._shared.kernels.w4a16.exl3_synth import write_exl3_checkpoint
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared
    from tests.experimental.b12x.moe.test_w4a16_mixed_trellis import _serial_tier

    config = Exl3SynthConfig(
        codebook="lut_e4m3",
        num_experts=2,
        hidden_size=512,
        intermediate_size=3072,
        moe_layer_indices=(1,),
        bits=2,
        intermediate_hadamard=True,
        pre_block=512,
        post_block=128,
        per_expert_input_rotations=True,
        extent_alignment_slots=4,
        extent_barriers=(48,),
        seed=53,
    )
    manifest = write_exl3_checkpoint(tmp_path, config)
    layer = read_exl3_layer(
        tmp_path, manifest, 1, first_slot=first_slot, slot_count=slots
    )
    layer.sign_pattern.copy_(torch.tensor([0, 6], dtype=torch.uint8))
    expected = _uniform_reference(layer, activation=activation, device=_device())
    source_metadata, source_weights = trellis_from_exl3(layer)
    plan = fused_moe.plan_weights(
        source=source_metadata,
        activation=fused_moe.ActivationSpec(
            mode="a16", nonlinearity=activation, io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=2, hidden_size=512, intermediate_size=slots * 32
        ),
    )
    prepared = fused_moe.prepare_weights(
        plan=plan, weights=source_weights, device=_device(),
    )
    actual = prepared._impl.representation.value
    assert actual.params_dtype == expected.params_dtype == torch.float16
    assert actual.tile_config == expected.tile_config
    for name in (
        "w13",
        "w2",
        "intermediate_rotations",
        "gate_suh",
        "up_suh",
        "down_svh",
    ):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0, atol=0
        )
    assert actual.intermediate_hadamard
    assert actual.gate_suh.data_ptr() == actual.up_suh.data_ptr()

    topk = 2
    torch.manual_seed(617)
    source = (torch.randn(capacity, 512, device=_device()) * 0.01).bfloat16()
    route_ids = torch.tensor([0, 1], device=_device(), dtype=torch.int32)
    route_ids = route_ids.repeat(capacity, 1)
    route_weights = torch.softmax(torch.randn(capacity, topk, device=_device()), -1)
    expert_map = torch.arange(2, device=_device(), dtype=torch.int32)
    output = torch.empty_like(source)
    execution = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
        routing=fused_moe.RoutingSpec(),
    )

    def allocate(state):
        return tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                     for s in state.scratch.scratch_specs())

    def bind(state, scratch, rows):
        return state.bind(
            scratch=scratch, a=source[:rows], experts=prepared,
            topk_weights=route_weights[:rows], topk_ids=route_ids[:rows],
            output=output[:rows],
        )

    def primer(state):
        scratch = allocate(state)
        binding = bind(state, scratch, capacity)
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    def oracle(rows):
        return _serial_tier(
            source[:rows], expected, route_weights[:rows], route_ids[:rows],
            expert_map, block_size_m=8, activation=activation,
        ).to(output.dtype)

    with PreparationSession(device=source.device, autotune=False) as session:
        request = execution.request(name="exl3-source-extent", prepare_call=primer)
        session.prepare((request,))
        state = require_prepared(request.plan, "moe.decode")
        scratch = allocate(state)
        route_pack_launches = state.scratch._prewarmed_route_pack_launches
        assert route_pack_launches is not None
        pointers = tuple(t.data_ptr() for t in (*scratch, output))
        for rows in sorted({1, min(8, capacity), capacity, min(17, capacity)}):
            reference = oracle(rows)
            with kernel_resolution_guard("EXL3 source extent"):
                binding = bind(state, scratch, rows)
                assert binding.route_pack_launches is route_pack_launches
                result = state.run(binding)
            assert result.data_ptr() == output.data_ptr()
            assert torch.isfinite(result).all() and torch.count_nonzero(result)
            torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-3)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                # Bind initializes synchronization scalars in caller scratch;
                # capture those writes together with the consuming kernels.
                captured = state.run(bind(state, scratch, min(17, capacity)))
            for factor in (-0.5, 2.0):
                source.mul_(factor)
                route_weights.copy_(route_weights.flip(-1))
                reference = oracle(min(17, capacity))
                for tensor in scratch:
                    tensor.fill_(255)
                before = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == before
                assert pointers == tuple(t.data_ptr() for t in (*scratch, output))
                assert torch.isfinite(captured).all() and torch.count_nonzero(captured)
                torch.testing.assert_close(captured, reference, rtol=2e-3, atol=2e-3)
        finally:
            graph.reset()
