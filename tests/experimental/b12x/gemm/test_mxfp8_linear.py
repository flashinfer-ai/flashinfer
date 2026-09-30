"""gemm.mxfp8_linear: ModelOpt MXFP8 linear vs quantized reference, K-padding
semantics, and CUDA-graph capture of the default fused path.

Curated from b12x tests/test_gemm_mxfp8_linear.py (kept whole — it was
already tight).
"""

from __future__ import annotations

import cutlass.cute as cute
import pytest
import torch

from b12x._lib.utils import convert_sf_from_mma_layout
from b12x.preparation import PreparationSession, PreparedCall
from b12x.quantization import mxfp8 as mxfp8_quant
from b12x.gemm import blockscaled, mxfp8_linear
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    empty_mxfp8_rows_bases,
    mxfp8_rows_from_bases,
)

from ._blockscaled import prepared
from ..conftest import require_b12x


def require_mxf8_mma() -> None:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        pytest.skip("CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op")


def _quantize_modelopt_mxfp8_rows(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = map(int, source.shape)
    chunks = width // 32
    blocked = source.to(torch.float32).reshape(rows, chunks, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_exp = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    scale_u8 = (scale_exp + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (
        (blocked / scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, width)
        .contiguous()
    )
    return values, scale_u8.contiguous()


def _quantize_source(source):
    rows, width = source.shape
    bases = empty_mxfp8_rows_bases(rows, width, num_groups=1, device=source.device)
    packed = mxfp8_rows_from_bases(*bases, rows, width, num_groups=1)
    args = (source, packed.values, packed.scale_rows, packed.scale_mma)
    plan = mxfp8_quant.plan(mxfp8_quant.query_from_call(*args))
    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(
            name="reference_quantization", prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(*args),
            ),
        ),))
        session.freeze()
        mxfp8_quant.quantize_rows(*args, plan=plan)
    return packed


def _reference_from_packed(source: torch.Tensor, packed_weight) -> torch.Tensor:
    rows, width = map(int, source.shape)
    padded_width = int(packed_weight.padded_in_features)
    if width != padded_width:
        padded = source.new_zeros((rows, padded_width))
        padded[:, :width] = source
        source = padded.contiguous()
    x_q = _quantize_source(source)
    x_deq = dequantize_mxfp8_rows_torch(x_q.values, x_q.scale_rows)
    w_deq = dequantize_mxfp8_rows_torch(
        packed_weight.weight.values, packed_weight.weight.scale_rows
    )
    return x_deq @ w_deq.T


def _make_inputs(tokens: int, in_features: int, out_features: int):
    source = (
        torch.randn((tokens, in_features), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()
    weight_bf16 = (
        torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        / 8
    ).contiguous()
    weight, weight_scale = _quantize_modelopt_mxfp8_rows(weight_bf16)
    packed = mxfp8_linear.pack_weight(weight, weight_scale)
    return source, weight_scale, packed


def test_mm_matches_quantized_reference_small_n() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260614)

    source, _, packed = _make_inputs(7, 128, 32)
    expected = _reference_from_packed(source, packed)
    with prepared(source, packed, activation_mode="quantized") as plan:
        actual = mxfp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        assert actual.shape == (7, 32)
        torch.testing.assert_close(
            actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
        )


def test_mm_persistent_ctas_complete_single_stage_epilogue_stores() -> None:
    require_b12x()
    require_mxf8_mma()

    tokens, in_features, out_features = 1372, 128, 4096
    source_values = torch.ones(
        (tokens, in_features), device="cuda", dtype=torch.float8_e4m3fn
    )
    source_scale = torch.full(
        (tokens, in_features // 32), 127, device="cuda", dtype=torch.uint8
    )
    weight = torch.ones(
        (out_features, in_features), device="cuda", dtype=torch.float8_e4m3fn
    )
    weight_scale = torch.full(
        (out_features, in_features // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    packed = mxfp8_linear.pack_weight(weight, weight_scale)

    with prepared((source_values, source_scale), packed, expected_m=tokens) as plan:
        for _ in range(4):
            actual = mxfp8_linear.mm(
                (source_values, source_scale), packed, plan=plan
            )
            torch.cuda.synchronize()
            assert torch.all(actual == in_features)


@pytest.mark.parametrize("out_features", (12448, 12544, 14336))
@pytest.mark.parametrize("capture", (False, True), ids=("eager", "graph"))
def test_mm_prefill_swizzle_bounds_weight_scales(
    out_features: int, capture: bool
) -> None:
    """BK64 swizzle padding must not read beyond the packed weight scales.

    N=12448 is a TP2 GLM KDA projection; N=12544 has full scale atoms but
    a partial 16-tile raster; N=14336 has a complete raster. Run under
    compute-sanitizer with PYTORCH_NO_CUDA_MEMORY_CACHING=1 and select the
    eager cases for physical allocation bounds. Graph cases require PyTorch's
    caching allocator for capture-time output and workspace allocations.
    """
    require_b12x()
    require_mxf8_mma()

    capacity, in_features = 4096, 4096
    source = torch.ones((capacity, in_features), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(
        (out_features, in_features), dtype=torch.float8_e4m3fn, device="cuda"
    )
    exponents = torch.arange(out_features, device="cuda") % 4 - 2
    scales = (exponents + 127).to(torch.uint8)[:, None].expand(
        out_features, in_features // 32
    ).contiguous()
    packed = blockscaled.pack_weight(weight, scales)
    expected = (in_features * 2.0 ** exponents).to(torch.bfloat16)

    with prepared(
        source, packed, activation_mode="quantized"
    ) as plan:
        for tokens in (137, 2048, capacity):
            actual = blockscaled.mm(source[:tokens], packed, plan=plan)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected.expand(tokens, -1), rtol=0, atol=0)

        if not capture:
            return

        for tokens in (127, 257):
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    actual = blockscaled.mm(source[:tokens], packed, plan=plan)
                address = actual.data_ptr()
                for multiplier in (0.5, 2.0, 1.0):
                    source.fill_(multiplier)
                    actual.fill_(float("nan"))
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert actual.data_ptr() == address
                    assert (
                        torch.cuda.memory_stats()["allocation.all.allocated"]
                        == allocations
                    )
                    torch.testing.assert_close(
                        actual, (expected * multiplier).expand(tokens, -1), rtol=0, atol=0
                    )
            finally:
                graph.reset()


@pytest.mark.parametrize("tokens", (2, 3, 8, 15, 16, 17, 32, 99))
def test_mm_writes_all_rows_for_unaligned_output_width(tokens: int) -> None:
    """A small-batch GEMM must store every live row when N spans multiple tiles."""
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260814 + tokens)

    source, _, packed = _make_inputs(tokens, 7168, 132)
    expected = _reference_from_packed(source, packed)
    with prepared(source, packed, activation_mode="quantized", expected_m=tokens) as plan:
        actual = mxfp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        assert actual.shape == (tokens, 132)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-2,
        )


def test_mm_unaligned_output_stride_captures_and_replays() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260822)

    source, _, packed = _make_inputs(8, 7168, 132)
    replacement = torch.randn_like(source).div_(4)
    expected = _reference_from_packed(replacement, packed)
    with prepared(source, packed, activation_mode="quantized") as plan:
        mxfp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = mxfp8_linear.mm(source, packed, plan=plan)
        source.copy_(replacement)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(),
            expected.to(actual.dtype).float(),
            rtol=1e-2,
            atol=2e-2,
        )
        graph.reset()


def test_mm_pads_k32_to_dense_tile() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260615)

    source, weight_scale, packed = _make_inputs(3, 160, 40)

    assert packed.in_features == 160
    assert packed.padded_in_features == 256
    assert packed.weight.values.shape == (40, 256)
    assert packed.weight.scale_rows.shape == (1, 40, 8)
    torch.testing.assert_close(
        packed.weight.scale_rows.view(torch.uint8)[0, :, :5], weight_scale
    )
    assert torch.all(packed.weight.scale_rows.view(torch.uint8)[0, :, 5:] == 127)

    expected = _reference_from_packed(source, packed)
    with prepared(source, packed, activation_mode="quantized") as plan:
        actual = mxfp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        assert actual.shape == (3, 40)
        torch.testing.assert_close(
            actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
        )


def test_mm_default_fused_path_captures_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260616)

    source, _, packed = _make_inputs(1, 160, 40)

    with prepared(source, packed) as plan:
        eager = mxfp8_linear.mm(source, packed, plan=plan).clone()
        torch.cuda.synchronize()

        mxfp8_linear.mm(source, packed, plan=plan)  # warm before capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = mxfp8_linear.mm(source, packed, plan=plan)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("tokens,in_features", [(1, 160), (8, 256), (9, 256), (17, 160)])
def test_mm_uses_quantizer_without_scale_padding_initialization(
    monkeypatch, tokens: int, in_features: int,
) -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260903)

    from b12x.gemm._shared import wo_mxfp8

    source, _, packed = _make_inputs(tokens, in_features, 384)
    expected = _reference_from_packed(source, packed)
    allocations = []
    original = wo_mxfp8.empty_mxfp8_rows_bases

    def poison_storage(*args, **kwargs):
        allocations.append(kwargs.get("initialize_scales", True))
        bases = original(*args, **kwargs)
        # UE8M0 byte 255 is NaN: any unwritten scale consumed by GEMM is visible.
        bases[1].fill_(255)
        bases[2].fill_(255)
        return bases

    monkeypatch.setattr(wo_mxfp8, "empty_mxfp8_rows_bases", poison_storage)
    with prepared(source, packed, activation_mode="quantized", expected_m=32) as plan:
        actual = mxfp8_linear.mm(source, packed, plan=plan)
        torch.cuda.synchronize()

        assert allocations == [False]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.float(), expected.to(actual.dtype).float(), rtol=0, atol=0
        )


def test_mm_quantizer_reuses_planned_capacity_under_frozen_resolution() -> None:
    require_b12x()
    require_mxf8_mma()

    source, _, packed = _make_inputs(16, 384, 256)
    expected = _reference_from_packed(source, packed).to(source.dtype)
    with prepared(source, packed, activation_mode="quantized") as plan:
        for tokens in (1, 8, 9, 16):
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    actual = mxfp8_linear.mm(source[:tokens], packed, plan=plan)
                address = actual.data_ptr()
                allocated = torch.cuda.memory_allocated()
                for _ in range(3):
                    actual.fill_(float("nan"))
                    graph.replay()
                torch.cuda.synchronize()
                assert actual.data_ptr() == address
                assert torch.cuda.memory_allocated() == allocated
                torch.testing.assert_close(actual, expected[:tokens], rtol=0, atol=0)
            finally:
                graph.reset()


def test_blockscaled_mm_accepts_prequantized_mxfp8_and_replays() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260823)

    tokens, in_features, out_features = 6, 128, 64
    source, _, packed = _make_inputs(tokens, in_features, out_features)
    source_q = _quantize_source(source)
    source_scale_storage = convert_sf_from_mma_layout(
        source_q.scale_mma,
        m=tokens,
        k=in_features,
        num_groups=1,
        sf_vec_size=32,
    )
    with prepared(source, packed, activation_mode="quantized") as plan:
        expected = blockscaled.mm(source, packed, plan=plan)
    for scales in (source_q.scale_mma, source_scale_storage):
        operands = (source_q.values, scales)
        with prepared(operands, packed) as plan:
            actual = blockscaled.mm(operands, packed, plan=plan, out_dtype=torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    graph_output = blockscaled.mm(
                        operands, packed, plan=plan, out_dtype=torch.bfloat16,
                    )
                output_ptr = graph_output.data_ptr()
                allocated = torch.cuda.memory_allocated()
                for _ in range(3):
                    graph_output.fill_(float("nan"))
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                assert graph_output.data_ptr() == output_ptr
                torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            finally:
                graph.reset()


def test_blockscaled_mm_accepts_compact_mxfp8_scales_with_k_padding() -> None:
    require_b12x()
    require_mxf8_mma()
    torch.manual_seed(20260824)

    source, _, packed = _make_inputs(6, 160, 40)
    source_values, source_scale_rows = _quantize_modelopt_mxfp8_rows(source)
    operands = (source_values, source_scale_rows)
    with prepared(operands, packed) as plan:
        actual = blockscaled.mm(operands, packed, plan=plan, out_dtype=torch.bfloat16)
    with prepared(source, packed, activation_mode="quantized") as plan:
        expected = blockscaled.mm(source, packed, plan=plan)

    assert actual.shape == (6, 40)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
