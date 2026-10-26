"""Real SM100 integration for the clamped routed NVFP4 geometry."""

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe import SwiGLU
from flashinfer.fused_moe.prepare import _activation_param_view
from tests.moe.cake_warp_decode_reference import (
    CLAMPED_GEOMETRY,
    check_cross_stream_parity,
    check_graph_replay_after_mutation,
    check_layer_graph_replay,
    check_reprepare_and_release_without_stream_sync,
    check_same_address_receipt_generations,
    prepare_fixture,
)


def test_clamped_swiglu_parameter_view_keeps_zero_beta():
    view = _activation_param_view(SwiGLU(limit=10.0), 256, torch.device("cpu"))
    assert set(view) == {"gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"}
    torch.testing.assert_close(view["gemm1_alpha"], torch.ones(256), atol=0, rtol=0)
    torch.testing.assert_close(view["gemm1_beta"], torch.zeros(256), atol=0, rtol=0)
    torch.testing.assert_close(
        view["gemm1_clamp_limit"], torch.full((256,), 10.0), atol=0, rtol=0
    )


@pytest.fixture(scope="module")
def clamped_fixture():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("clamped warp-decode integration requires exact SM100")
    fixture = prepare_fixture(CLAMPED_GEOMETRY, seed=20260924)
    view = fixture.weight_view
    torch.testing.assert_close(
        view["gemm1_clamp_limit"] * view["output1_scale_gate_scalar"],
        torch.full((256,), 10.0, device="cuda"),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.unique(view["gemm1_clamp_limit"]).numel() > 1
    yield fixture
    torch.cuda.synchronize()
    del fixture


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", CLAMPED_GEOMETRY.token_counts)
def test_clamped_warp_decode_all_shapes(clamped_fixture, num_tokens):
    # Actual compiled candidate and official TRT-LLM baseline, same physical
    # weights/activations/routes/scales; no test double substitutes GPU work.
    check_cross_stream_parity(clamped_fixture, num_tokens)


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", (1, 7, 9, 11, 12, 23, 32))
def test_clamped_layer_graph_routes(clamped_fixture, num_tokens):
    # Exercise the public MoELayer/weight-pack API and graph replay, including
    # every route family and the two count-rank selections.
    check_layer_graph_replay(clamped_fixture, num_tokens)


@pytest.mark.gpu
def test_clamped_same_address_receipt_generations(clamped_fixture):
    check_same_address_receipt_generations(clamped_fixture)


@pytest.mark.gpu
def test_clamped_reprepare_and_release_without_caller_stream_sync(clamped_fixture):
    check_reprepare_and_release_without_stream_sync(clamped_fixture)


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", (1, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 32))
def test_clamped_graph_replay_follows_mutations_and_in_flight_release(
    clamped_fixture, num_tokens
):
    # Every schedule / route-packer boundary of the clamped geometry.
    check_graph_replay_after_mutation(clamped_fixture, num_tokens)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_clamped_sources_keep_target_and_compile_options(target):
    from flashinfer.jit import cake_fused_moe_warp_decode as jit

    csrc = jit._get_cake_fused_moe_warp_decode_csrc_dir()
    clamped = jit._load_clamped_e256_sources(csrc, target)
    spec = jit.gen_cake_fused_moe_warp_decode_module(target)
    assert len(spec.sources) == len(set(spec.sources))
    if target == "sm100a":
        assert [source.name for source in clamped] == list(jit._CLAMPED_E256_SOURCES)
        assert len(clamped) == len(set(clamped)) == 15
        assert all(source.parent.name == "dsv4_clamped_e256" for source in clamped)
        assert set(clamped) <= set(spec.sources)
        assert (
            "-DFLASHINFER_CAKE_WARP_DECODE_HAS_CLAMPED_E256=1" in spec.extra_cuda_cflags
        )
        for source in clamped:
            assert "--use_fast_math" in spec.extra_cuda_cflags_by_source[source]
        with pytest.raises(FileNotFoundError, match="clamped E256 sources not found"):
            jit._load_clamped_e256_sources(csrc / "missing", target)
    else:
        assert not clamped
        assert not any(
            source.parent.name == "dsv4_clamped_e256" for source in spec.sources
        )
        assert (
            "-DFLASHINFER_CAKE_WARP_DECODE_HAS_CLAMPED_E256=0" in spec.extra_cuda_cflags
        )
    for source in spec.sources:
        if source.name in jit._NO_FAST_MATH_SOURCES:
            assert "--use_fast_math" not in spec.extra_cuda_cflags_by_source[source]
