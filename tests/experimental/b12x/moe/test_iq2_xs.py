"""Independent codec, compact-storage and public planning contracts."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut_cpu
from b12x.moe import fused_moe as moe
from b12x.moe._shared.kernels.w4a16.iq2_xs import pack_iq2_xs_matrix
from b12x.testing.iq2_xs_reference import dequantize_blocks, descriptor_vectors


def blocks(e=2, n=544, k=768, codec="iq2_xs"):
    block_size = 32 if codec == "q8_0" else 256
    g = torch.Generator().manual_seed(712)
    raw = torch.randint(0, 256, (e, n, k // block_size, 34 if codec == "q8_0" else 66 if codec == "iq2_xxs" else 74), dtype=torch.uint8, generator=g)
    bases = (torch.randn(e, n, k // block_size, generator=g) * 0.01).half()
    raw[..., :2] = bases[..., None].view(torch.uint8)
    return raw


def test_relu2_direct_route_workspace_covers_capacity():
    from b12x.moe._shared.kernels.w4a16.host import (
        packed_gemm_scratch_elements,
        plan_w4a16_buffers,
    )

    prepared = SimpleNamespace(
        num_experts=8, intermediate_size=1280, hidden_size=1024,
        is_gated=False, weight_layout="iq2_xs",
    )
    plan = plan_w4a16_buffers(prepared, m=8, topk=2, sms=48, block_size_m=8)
    for rows in (1, 2, 4, 8):
        for size_n, allocated in (
            (1280, plan.fc1_c_tmp_elements), (1024, plan.fc2_c_tmp_elements),
        ):
            assert allocated >= packed_gemm_scratch_elements(
                size_n=size_n, route_slots=rows * 2 * 8, moe_block_size=8,
                sms=48, weight_layout="iq2_xs",
            )


def test_iq2_xs_rejects_eight_way_warp_reduction():
    from b12x.moe._shared.kernels.w4a16.kernel import _candidate_tile_fits

    assert not _candidate_tile_fits(
        problem_n=1280, problem_k=1024, cta_m_blocks=1, tile_n=64,
        tile_k=256, cta_threads=256, max_shared_mem=101376,
        scale_format="iq2_xs", weight_layout="iq2_xs",
    )


def test_byte_selectors_reconstruct_the_independent_magnitude_grid():
    from b12x._lib.quant.iq2_xs import iq2_xs_pair_selectors_cpu

    selectors = iq2_xs_pair_selectors_cpu()
    assert selectors.shape == (512, 4) and selectors.dtype == torch.int16
    source = torch.tensor([8, 25, 43, 43], dtype=torch.bfloat16).view(torch.uint8)
    indices = (selectors.int()[..., None] >> (4 * torch.arange(4))) & 15
    assert torch.all(indices < 8)
    pairs = source[indices.long()].contiguous().view(torch.bfloat16).reshape(512, 8)
    expected = descriptor_vectors()[:512].abs().bfloat16()
    assert torch.equal(pairs, expected)


def unpack_planes(words, metadata, shape):
    e, n, kb, _ = shape
    raw = torch.empty(shape, dtype=torch.uint8)
    q8 = shape[-1] == 34
    k16 = kb * (2 if q8 else 16)
    row_words = 4 if q8 else 1
    payload_bytes = 32 if q8 else 64
    q = (
        words.reshape(e, k16, n // 16, 8, 2, row_words)
        .transpose(3, 4).reshape(e, k16, n, row_words).view(torch.uint8)
    )
    raw[..., 2:2 + payload_bytes] = q.permute(0, 2, 1, 3).reshape(e, n, kb, payload_bytes)
    raw[..., :2] = (
        metadata[: e * kb * n * 2].view(torch.int16)
        .reshape(e, kb, n // 16, 8, 2).transpose(-2, -1)
        .reshape(e, kb, n, 1).view(torch.uint8).permute(0, 2, 1, 3)
    )
    if shape[-1] == 74:
        raw[..., 66:] = (
            metadata[e * kb * n * 2 :]
            .reshape(e, kb, n // 16, 8, 8, 2).transpose(-2, -1)
            .reshape(e, kb, n // 16, 8, 16)
            .permute(0, 2, 4, 1, 3)
            .reshape(e, n, kb, 8)
        )
    return raw


def test_every_descriptor_matches_independent_table():
    actual, reference = iq2_xs_execution_lut_cpu(), descriptor_vectors()
    assert (
        hashlib.sha256(reference.numpy().tobytes()).hexdigest()
        == "516a77f2ff3be1f22250b0ef65df740495343ec90feb2e21cb7a04a22041deeb"
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
def test_compact_planes_preserve_source_bytes_and_projection_order(swap, codec):
    source = blocks(codec=codec)
    original = source.clone()
    words, metadata = pack_iq2_xs_matrix(source, codec=codec, swap_halves=swap)
    assert words.numel() * words.element_size() + metadata.numel() == source.numel()
    assert (
        words.untyped_storage().nbytes() + metadata.untyped_storage().nbytes()
        == source.numel()
    )
    expected = torch.cat(source.chunk(2, 1)[::-1], 1) if swap else source
    reconstructed = unpack_planes(words, metadata, source.shape)
    assert torch.equal(reconstructed, expected)
    assert torch.equal(source, original)


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("n,k", [(128, 256), (256, 256), (384, 768), (768, 768)])
@pytest.mark.parametrize("tile_scales", [False, True])
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
def test_tiled_descriptors_preserve_compact_bytes(swap, n, k, tile_scales, codec):
    source = blocks(e=2, n=n, k=k, codec=codec)
    original = source.clone()
    words, metadata = pack_iq2_xs_matrix(
        source, codec=codec, swap_halves=swap, tile_descriptors=True, tile_scales=tile_scales
    )
    canonical_words, canonical_metadata = pack_iq2_xs_matrix(source, codec=codec, swap_halves=swap)
    untiled = (
        words.reshape(2, n // 64, k // 128, 8, 4, 8, 2, 4 if codec == "q8_0" else 1)
        .permute(0, 2, 3, 1, 4, 5, 6, 7).contiguous().reshape(-1)
    )
    assert torch.equal(untiled, canonical_words)
    base_bytes = 2 * (k // 256) * n * 2
    if tile_scales and codec == "iq2_xs":
        untiled_scales = (
            metadata[base_bytes:].reshape(2, n // 64, k // 256, 2, 4, 4, 8, 2)
            .permute(0, 2, 1, 5, 3, 4, 6, 7).contiguous().reshape(-1)
        )
        restored_metadata = torch.cat((metadata[:base_bytes], untiled_scales))
    else:
        restored_metadata = metadata
    assert torch.equal(restored_metadata, canonical_metadata)
    expected = torch.cat(source.chunk(2, 1)[::-1], 1) if swap else source
    assert torch.equal(unpack_planes(untiled, restored_metadata, source.shape), expected)
    assert words.untyped_storage().nbytes() + metadata.untyped_storage().nbytes() == source.numel()
    assert torch.equal(source, original)


@pytest.mark.parametrize(
    "base_bits", [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x3C01, 0x7BFF, 0xFBFF]
)
def test_scale_nibbles_and_fp16_edges_survive_packing(base_bits):
    source = blocks(e=1, n=32, k=256)
    source[..., 0], source[..., 1] = base_bits & 255, base_bits >> 8
    source[..., 66:] = (
        torch.arange(8, dtype=torch.uint8) * 2
        + (torch.arange(8, dtype=torch.uint8) * 2 + 1) * 16
    )
    words, metadata = pack_iq2_xs_matrix(source)
    decoded = dequantize_blocks(unpack_planes(words, metadata, source.shape)).bfloat16()
    reference = dequantize_blocks(source).bfloat16()
    assert torch.equal(decoded.view(torch.int16), reference.view(torch.int16))
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize("bits", [0x7C00, 0xFC00, 0x7E00])
def test_nonfinite_bases_rejected(bits):
    source = blocks(e=1, n=16, k=256)
    source[0, 0, 0, 0], source[0, 0, 0, 1] = bits & 255, bits >> 8
    with pytest.raises(ValueError, match="finite"):
        pack_iq2_xs_matrix(source)


def weight_plan(
    *, mode="a16", activation="silu", dtype=torch.bfloat16, h=2048, i=512, packing=None, codec="iq2_xs"
):
    return moe.plan_weights(
        source=moe.PackedSource(
            format=codec, w13_layout=moe.W13Layout.W31
        ),
        activation=moe.ActivationSpec(
            mode=mode, nonlinearity=activation, io_dtype=dtype
        ),
        geometry=moe.MoEGeometry(num_experts=256, hidden_size=h, intermediate_size=i),
        constraints=moe.WeightPlanConstraints(required_packing=packing),
    )


@pytest.mark.parametrize("activation", ["silu", "relu2"])
@pytest.mark.parametrize("i", [256, 512])
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs"])
def test_public_weight_plan(activation, i, codec):
    plan = weight_plan(activation=activation, i=i, codec=codec)
    assert plan.prepared_format.weights is moe.WeightEncoding(codec)
    assert plan.prepared_format.scales is moe.ScaleEncoding(codec)
    assert plan.prepared_format.packing is moe.WeightPacking(codec + "_compact")
    assert plan._impl.w4a16_weight_layout == codec
    assert plan._impl.w4a16_scale_format == codec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "a8"},
        {"mode": "a4"},
        {"dtype": torch.float16},
        {"activation": "swigluoai"},
        {"h": 384},
        {"i": 128},
        {"packing": "mma_packed"},
    ],
)
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs"])
def test_unsupported_weight_contracts_fail_at_planning(kwargs, codec):
    with pytest.raises((ValueError, NotImplementedError)):
        weight_plan(**kwargs, codec=codec)


@pytest.mark.parametrize(
    "source", [torch.zeros(2, 16, 1, 73, dtype=torch.uint8), torch.zeros(2, 16, 1, 74)]
)
def test_invalid_payload_rejected(source):
    with pytest.raises((TypeError, ValueError)):
        moe.IQ2XSWeights(source, source)
    with pytest.raises((TypeError, ValueError)):
        pack_iq2_xs_matrix(source)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"num_tokens": 1}, True),
        ({"num_tokens": 8}, True),
        ({"num_tokens": 9}, False),
        ({"activation": "relu2"}, True),
        ({"activation": "relu2", "num_tokens": 1}, True),
        ({"activation": "relu2", "num_tokens": 6}, True),
        ({"activation": "relu2", "num_tokens": 7}, True),
        ({"activation": "relu2", "num_tokens": 9}, False),
        ({"activation": "relu2", "num_tokens": 6, "deterministic_output": True}, False),
        ({"activation": "relu2", "num_tokens": 6, "collect_activation_amax": True}, False),
        ({"activation": "relu2", "num_tokens": 6, "apply_router_weight_on_input": True}, False),
        ({"deterministic_output": True}, False),
        ({"collect_activation_amax": True}, False),
        ({"apply_router_weight_on_input": True}, False),
    ],
)
def test_direct_route_eligibility(overrides, expected):
    from b12x.moe.fused_moe._impl import _w4a16_direct_routing_supported

    query = SimpleNamespace(
        **{
            "source_format": "iq2_xs",
            "quant_mode": "w4a16",
            "w4a16_weight_layout": None,
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_tokens": 8,
            "io_dtype": "bfloat16",
            "activation": "silu",
            "deterministic_output": False,
            "collect_activation_amax": False,
            "apply_router_weight_on_input": False,
            **overrides,
        }
    )
    assert _w4a16_direct_routing_supported(query) is expected


@pytest.mark.parametrize("capacity", [1, 2, 4, 6, 7, 8, 9])
def test_relu2_tuning_races_supported_route_modes(capacity):
    from b12x.moe.fused_moe._tuning import MoeDecodeQuery, TUNING
    from b12x.preparation import DeviceIdentity, FrozenMapping

    query = MoeDecodeQuery(
        quant_mode="w4a16", quant_modes=("w4a16",), source_format="iq2_xs",
        activation="relu2", io_dtype="bfloat16", num_experts=512,
        hidden_size=1024, intermediate_size=1280, top_k=4,
        num_tokens=capacity, routed_rows=capacity * 4, route_num_experts=512,
        route_logits_dtype=None, apply_router_weight_on_input=False,
        collect_activation_amax=False, deterministic_output=False,
        swiglu_limit=None, swiglu_alpha=1.0, swiglu_beta=0.0,
        w13_layout="w31", weight_layouts=("iq2_xs",),
        w4a16_weight_layout="iq2_xs", w4a16_scale_format="iq2_xs",
        w4a16_block_size_m=None, fast_math=True, numerical_recipe=None,
        controls=FrozenMapping(),
    )
    device = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
    configs = [config for _, config in TUNING.eligible_plan(query, device).candidates]
    expected = {"direct", "packed"} if capacity <= 8 else {"packed"}
    assert {config.w4a16_route_mode for config in configs} == expected
    for config in configs:
        assert TUNING.configure(query, device=device, override=config).default == config


@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs"])
def test_weight_codec_must_match_plan(codec):
    other = "iq2_xxs" if codec == "iq2_xs" else "iq2_xs"
    plan = weight_plan(codec=codec)
    weights = moe.BlockQuantWeights(blocks(e=1, n=16, k=256, codec=other),
                                   blocks(e=1, n=16, k=256, codec=other), codec=other)
    with pytest.raises(TypeError, match="matching"):
        moe.prepare_weights(plan=plan, weights=weights)


def test_codec_table_identity_and_sizes():
    from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
    from b12x._lib.quant.block_codec import block_codec
    for selectors in (False, True):
        xs = iq2_xs_execution_lut("cpu", prepare=True, selectors=selectors)
        xxs = iq2_xs_execution_lut("cpu", prepare=True, selectors=selectors, codec="iq2_xxs")
        assert xs.data_ptr() != xxs.data_ptr()
        assert xxs.numel() * xxs.element_size() == block_codec("iq2_xxs").lut_bytes(selectors=selectors)
        assert iq2_xs_execution_lut("cpu", selectors=selectors, codec="iq2_xxs") is xxs
