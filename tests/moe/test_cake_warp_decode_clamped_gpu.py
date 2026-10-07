"""Real SM100 integration for the clamped routed NVFP4 geometry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from flashinfer.fused_moe import SwiGLU
from flashinfer.fused_moe.prepare import _activation_param_view


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
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "cake_warp_decode.py"
    name = "_cake_warp_decode_clamped_gpu_harness"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    sys.modules[name] = harness
    spec.loader.exec_module(harness)
    geometry = next(g for g in harness.GEOMETRIES if g.name == "e256_i2048_k6_clamp10")
    fixture = harness._prepare_fixture(geometry, seed=20260924)
    view = fixture.weight_view
    torch.testing.assert_close(
        view["gemm1_clamp_limit"] * view["output1_scale_gate_scalar"],
        torch.full((256,), 10.0, device="cuda"),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.unique(view["gemm1_clamp_limit"]).numel() > 1
    yield harness, fixture
    torch.cuda.synchronize()
    del fixture
    sys.modules.pop(name, None)


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", range(1, 33))
def test_clamped_warp_decode_all_shapes(clamped_fixture, num_tokens):
    harness, fixture = clamped_fixture
    # Actual compiled candidate and official TRT-LLM baseline, same physical
    # weights/activations/routes/scales; no test double substitutes GPU work.
    row = harness._correctness_case(fixture, num_tokens)
    assert row["official_baseline"] == "available"


@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", (1, 7, 9, 11, 12, 23, 32))
def test_clamped_layer_graph_routes(clamped_fixture, num_tokens):
    harness, fixture = clamped_fixture
    # Exercise the public MoELayer/weight-pack API and graph replay, including
    # every route family and the two count-rank selections.
    harness._layer_graph_case(fixture, num_tokens=num_tokens)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_clamped_sources_keep_target_and_compile_options(target):
    from flashinfer.jit import cake_fused_moe_warp_decode as jit

    csrc = jit._get_cake_fused_moe_warp_decode_csrc_dir()
    clamped = jit._load_clamped_e256_sources(csrc, target)
    spec = jit.gen_cake_fused_moe_warp_decode_module(target)
    assert len(spec.sources) == len(set(spec.sources))
    if target == "sm100a":
        assert len(clamped) == 15
        assert set(clamped) <= set(spec.sources)
        assert (
            "-DFLASHINFER_CAKE_WARP_DECODE_HAS_CLAMPED_E256=1" in spec.extra_cuda_cflags
        )
        for source in clamped:
            assert "--use_fast_math" in spec.extra_cuda_cflags_by_source[source]
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
