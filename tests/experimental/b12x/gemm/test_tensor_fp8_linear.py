from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import cutlass.cute as cute
import pytest
import torch

from b12x.gemm import tensor_fp8_linear
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared
from b12x.gemm.tensor_fp8_linear import api as tensor_fp8_api
from b12x.gemm.tensor_fp8_linear import _kernel as tensor_fp8_kernel

from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


@contextmanager
def _prepared(source, packed, *, expected_m=None):
    plan = tensor_fp8_linear.plan(tensor_fp8_linear.query_from_call(
        source, packed, expected_m=expected_m,
    ))
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(
            name="tensor_fp8", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run_tensor_fp8(
                    source, packed.values, packed.scale_mma, packed.block_scale,
                    packed.output_scale, out_dtype=torch.bfloat16, stream=None,
                ),
            ),
        ),))
        session.freeze()
        yield plan


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    output_scale = torch.tensor([0.0002], dtype=torch.float32, device="cuda")
    packed = tensor_fp8_linear.pack_weight(weight, output_scale)
    return source, weight, output_scale, packed


def test_mm_matches_static_tensor_fp8_reference() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260729)

    source, weight, output_scale, packed = _make_inputs(7, 128, 64)
    with _prepared(source, packed) as plan:
        actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        expected = (source.float() @ weight.float().T) * output_scale
        torch.cuda.synchronize()

        assert actual.shape == (7, 64)
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )


def test_mm_writes_all_rows_for_unaligned_output_stride() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814)

    source, weight, output_scale, packed = _make_inputs(3, 128, 132)
    with _prepared(source, packed) as plan:
        actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        expected = (source.float() @ weight.float().T) * output_scale
        torch.cuda.synchronize()

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )


def test_wide_unaligned_output_uses_expected_m_under_cuda_graph() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260816)

    source, weight, output_scale, packed = _make_inputs(2, 1024, 16386)
    with _prepared(source, packed, expected_m=2048) as plan:
        tensor_fp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        graph.replay()
        torch.cuda.synchronize()
        expected = (source.float() @ weight.float().T) * output_scale

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )
        graph.reset()


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260730)

    source, weight, output_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.values.shape == (40, 256)
    assert torch.count_nonzero(packed.values[:, 160:]) == 0

    with _prepared(source, packed) as plan:
        actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        expected = (source.float() @ weight.float().T) * output_scale
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )


def test_mm_uses_plain_fp8_mma_not_scale_storage() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260731)

    source, weight, output_scale, packed = _make_inputs(4, 128, 64)
    packed = replace(packed, scale_mma=torch.zeros_like(packed.scale_mma))

    with _prepared(source, packed) as plan:
        actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        expected = (source.float() @ weight.float().T) * output_scale
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )


def test_aligned_decode_uses_prepared_fp8_recipe() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260823)

    source, weight, output_scale, packed = _make_inputs(8, 128, 128)
    with _prepared(source, packed, expected_m=8) as plan:
        actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        expected = (source.float() @ weight.float().T) * output_scale

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = tensor_fp8_linear.mm(source, packed, plan=plan)
        output_ptr = graph_output.data_ptr()
        for _ in range(3):
            graph.replay()

        zero_scale_weight = replace(
            packed,
            block_scale=torch.zeros_like(packed.block_scale),
        )
        zero = tensor_fp8_linear.mm(source, zero_scale_weight, plan=plan)
        torch.cuda.synchronize()

        assert packed.block_scale.shape == (1, 1)
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-3,
        )
        assert graph_output.data_ptr() == output_ptr
        torch.testing.assert_close(graph_output, actual, rtol=0, atol=0)
        state = require_prepared(plan, "gemm.blockscaled.fixed")
        if state.use_block:
            assert torch.count_nonzero(zero) == 0
        else:
            torch.testing.assert_close(zero, actual, rtol=0, atol=0)
        graph.reset()


def test_block_fp8_recipe_is_bounded_by_regime_alignment_and_sm_count() -> None:
    common = {
        "live_m": 8,
        "expected_m": 8,
        "out_features": 5120,
        "padded_in_features": 6144,
    }

    assert tensor_fp8_kernel._use_block_fp8_recipe(**common, sm_count=48)
    assert not tensor_fp8_kernel._use_block_fp8_recipe(
        **(common | {"expected_m": 16}), sm_count=48
    )
    assert not tensor_fp8_kernel._use_block_fp8_recipe(
        **(common | {"out_features": 5119}), sm_count=48
    )
    assert not tensor_fp8_kernel._use_block_fp8_recipe(**common, sm_count=188)


def test_is_supported_honors_kernel_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        tensor_fp8_api, "default_is_supported", lambda *args, **kw: True
    )
    monkeypatch.setattr(
        tensor_fp8_api,
        "_kernel_is_supported",
        lambda: (False, "plain FP8 MMA unavailable"),
    )

    assert not tensor_fp8_api.is_supported()


def test_mm_default_path_captures() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260801)

    source, _, _, packed = _make_inputs(4, 128, 64)
    with _prepared(source, packed) as plan:
        eager = tensor_fp8_linear.mm(source, packed, plan=plan).clone()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = tensor_fp8_linear.mm(source, packed, plan=plan)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        graph.reset()


def test_prewarm_primes_dynamic_capacity_before_frozen_resolution() -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard

    require_b12x()
    source, weight, scale, packed = _make_inputs(130, 256, 144)
    with (
        kernel_resolution_guard("cold implicit plan"),
        pytest.raises(RuntimeError, match="not prepared.*frozen"),
    ):
        tensor_fp8_linear.mm(source, packed)
    assert tensor_fp8_linear.prewarm(packed, [129, 130, 0, 129]) == 2
    with kernel_resolution_guard("prewarmed implicit plan"):
        for rows in (129, 130):
            actual = tensor_fp8_linear.mm(source[:rows], packed)
            reference = source[:rows].float() @ weight.float().T * scale
            torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=1e-2)
