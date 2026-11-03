from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch
from b12x._lib.dense_gemm import (
    dense_gemm,
    dense_gemm_fused_quant_a,
    dense_gemm_fused_quant_a_grouped,
)
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm._shared.wo_mxfp8 import (
    dequantize_mxfp8_rows_torch,
    empty_mxfp8_rows_for_dense_gemm,
    quantize_mxfp8_rows_torch,
)

from b12x.testing.reference.helpers import dequantize_grouped_nvfp4, require_b12x


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _quantize_nvfp4_operand(
    source_gmk: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    groups, rows, _ = map(int, source_gmk.shape)
    row_counts = torch.full(
        (groups,), rows, dtype=torch.int32, device=source_gmk.device
    )
    amax = source_gmk.abs().amax().to(torch.float32)
    global_scale = (
        torch.tensor(
            [torch.finfo(torch.float8_e4m3fn).max * 6.0],
            dtype=torch.float32,
            device=source_gmk.device,
        )
        / amax
    )
    return quantize_grouped_nvfp4_torch(
        source_gmk, row_counts, global_scale
    ), global_scale


def _dequantize_nvfp4_dense_operand(
    operand: tuple[torch.Tensor, torch.Tensor],
    *,
    k: int,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    packed_mkl, scale_mma = operand
    packed_gmk = packed_mkl.permute(2, 0, 1).contiguous()
    return dequantize_grouped_nvfp4(
        packed_gmk,
        scale_mma,
        k,
        global_scale,
    )


def _mxfp8_gemm_reference(
    source_mkl: torch.Tensor,
    b_values: torch.Tensor,
    b_scale_rows: torch.Tensor,
) -> torch.Tensor:
    a_quant = quantize_mxfp8_rows_torch(source_mkl)
    a_dequant = dequantize_mxfp8_rows_torch(a_quant.values, a_quant.scale_rows).to(
        torch.bfloat16
    )
    b_dequant = dequantize_mxfp8_rows_torch(b_values, b_scale_rows).to(torch.bfloat16)
    return torch.einsum("mkl,nkl->mnl", a_dequant, b_dequant).to(torch.bfloat16)


def test_cute_migration_dense_nvfp4_gpu_oracle_and_graph() -> None:
    require_b12x()
    generator = torch.Generator(device="cuda").manual_seed(46_001)
    m, n, k = 32, 128, 128
    a_source = (
        torch.randn(
            (1, m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    b_source = (
        torch.randn(
            (1, n, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    )
    a, a_global_scale = _quantize_nvfp4_operand(a_source)
    b, b_global_scale = _quantize_nvfp4_operand(b_source)
    alpha = (1.0 / (a_global_scale[0] * b_global_scale[0])).reshape(1)
    out = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def run() -> torch.Tensor:
        return dense_gemm(
            a,
            b,
            out=out,
            alpha=alpha,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=False,
        )

    run()
    torch.cuda.synchronize()
    a_dequant = _dequantize_nvfp4_dense_operand(a, k=k, global_scale=a_global_scale)
    b_dequant = _dequantize_nvfp4_dense_operand(b, k=k, global_scale=b_global_scale)
    expected = torch.einsum("gmk,gnk->mng", a_dequant, b_dequant).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,expected_m,a_inner_span", ((2, None, 0), (8, 8, 32), (1, 8, 32), (1, 1, 32))
)
def test_cute_migration_dense_fused_quant_gpu_oracle_and_graph(
    m: int,
    expected_m: int | None,
    a_inner_span: int,
) -> None:
    require_b12x()
    generator = torch.Generator(device="cuda").manual_seed(46_002)
    n, k = 128, 128
    source = (
        torch.randn(
            (m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    if a_inner_span:
        physical = torch.empty(
            (k // a_inner_span, m, a_inner_span),
            dtype=source.dtype,
            device=source.device,
        )
        physical.copy_(source.view(m, k // a_inner_span, a_inner_span).permute(1, 0, 2))
        source = physical.permute(1, 2, 0)

    def logical_source():
        return source if not a_inner_span else source.permute(0, 2, 1).reshape(m, k)

    b_source = (
        torch.randn(
            (n, k, 1),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 32
    ).contiguous()
    b_quant = quantize_mxfp8_rows_torch(b_source)
    out = torch.empty((m, n, 1), dtype=torch.bfloat16, device="cuda")

    def run() -> torch.Tensor:
        return dense_gemm_fused_quant_a(
            source,
            b_quant.values,
            b_quant.scale_mma,
            out=out,
            mma_tiler_mn=(64, 64),
            expected_m=expected_m,
            a_inner_span=a_inner_span,
        )

    run()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        logical_source().unsqueeze(-1), b_quant.values, b_quant.scale_rows
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        logical_source().unsqueeze(-1), b_quant.values, b_quant.scale_rows
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m,expected_m", ((2, None), (1, 1), (8, 8), (1, 8)))
def test_cute_migration_dense_grouped_fused_quant_gpu_oracle_and_graph(
    m: int,
    expected_m: int | None,
) -> None:
    require_b12x()
    generator = torch.Generator(device="cuda").manual_seed(46_003)
    n, k, groups = 128, 128, 2
    source = (
        torch.randn(
            (m, groups, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    b_source = (
        torch.randn(
            (n, k, groups),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 32
    ).contiguous()
    b_quant = quantize_mxfp8_rows_torch(b_source)
    # The dense kernel writes C as physical [L,M,N]; retain that storage
    # contract while exposing the public logical [M,N,L] view.
    out = torch.empty((groups, m, n), dtype=torch.bfloat16, device="cuda").as_strided(
        (m, n, groups), (n, 1, m * n)
    )

    def run() -> torch.Tensor:
        return dense_gemm_fused_quant_a_grouped(
            source,
            b_quant.values,
            b_quant.scale_mma,
            groups=groups,
            out=out,
            mma_tiler_mn=(64, 64),
            expected_m=expected_m,
        )

    run()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.permute(0, 2, 1),
        b_quant.values,
        b_quant.scale_rows,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = _mxfp8_gemm_reference(
        source.permute(0, 2, 1),
        b_quant.values,
        b_quant.scale_rows,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 4, 8, 9, 16])
@pytest.mark.parametrize("k", [128, 640, 2560, 6144])
def test_cute_migration_mxfp8_quant_gpu_oracle_and_graph(m: int, k: int) -> None:
    require_b12x()
    generator = torch.Generator(device="cuda").manual_seed(46_100 + m)
    source = (
        torch.randn(
            (m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    expected = quantize_mxfp8_rows_torch(source)
    actual = empty_mxfp8_rows_for_dense_gemm(m, k, device="cuda")
    values = actual.values
    scale_rows = actual.scale_rows
    scale_mma = actual.scale_mma

    def run() -> None:
        quantize_mxfp8_rows_cute(source, values, scale_rows, scale_mma)

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        values.view(torch.uint8), expected.values.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = quantize_mxfp8_rows_torch(source)
    torch.testing.assert_close(
        values.view(torch.uint8), expected.values.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("capacity", [8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_quant_planned_capacity_reuses_callable(
    capacity, dtype, monkeypatch
) -> None:
    import b12x._lib.quant.mxfp8_rows as quant_module
    from b12x._lib.runtime_control import kernel_resolution_guard

    require_b12x()
    source = torch.randn((capacity, 640), dtype=dtype, device="cuda")
    actual = empty_mxfp8_rows_for_dense_gemm(capacity, 640, device="cuda")
    resolve = quant_module._get_compiled_mxfp8_rows_quant
    calls = []

    def track(*args, **kwargs):
        compiled = resolve(*args, **kwargs)
        calls.append(compiled)
        return compiled

    monkeypatch.setattr(quant_module, "_get_compiled_mxfp8_rows_quant", track)

    def run(rows):
        quantize_mxfp8_rows_cute(
            source[:rows],
            actual.values,
            actual.scale_rows,
            actual.scale_mma,
            expected_m=capacity,
        )

    run(capacity)
    warmed = calls[-1]
    scalar = resolve(640, dtype, 0, 256, "linear")
    pointers = tuple(
        t.data_ptr()
        for t in (source, actual.values, actual.scale_rows, actual.scale_mma)
    )
    with kernel_resolution_guard("MXFP8 quantization within one planned row capacity"):
        for rows in (1, 8, 9, 16):
            if rows > capacity:
                continue
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(rows)
            assert calls[-1] is warmed
            source.normal_().mul_(0.25)
            expected = empty_mxfp8_rows_for_dense_gemm(capacity, 640, device="cuda")
            for storage in (actual, expected):
                for tensor in (storage.values, storage.scale_rows, storage.scale_mma):
                    tensor.view(torch.uint8).fill_(0xA5)
            scalar(
                source[:rows], expected.values, expected.scale_rows, expected.scale_mma
            )
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert pointers == tuple(
                t.data_ptr()
                for t in (source, actual.values, actual.scale_rows, actual.scale_mma)
            )
            for name in ("values", "scale_rows", "scale_mma"):
                torch.testing.assert_close(
                    getattr(actual, name).view(torch.uint8),
                    getattr(expected, name).view(torch.uint8),
                    rtol=0,
                    atol=0,
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_decode_quant_finite_bit_patterns_match_scalar(dtype) -> None:
    from b12x._lib.quant.mxfp8_rows import _get_compiled_mxfp8_rows_quant

    source = torch.arange(65536, device="cuda", dtype=torch.int32)
    source = source.to(torch.int16).view(dtype).view(8, 8192)
    source.masked_fill_(~torch.isfinite(source), 0)
    reference = empty_mxfp8_rows_for_dense_gemm(8, 8192, device="cuda")
    actual = empty_mxfp8_rows_for_dense_gemm(8, 8192, device="cuda")
    for storage in (reference, actual):
        for tensor in (storage.values, storage.scale_rows, storage.scale_mma):
            tensor.view(torch.uint8).fill_(0xA5)
    scalar = _get_compiled_mxfp8_rows_quant(8192, dtype, 0, 256, "linear")
    scalar(source, reference.values, reference.scale_rows, reference.scale_mma)
    quantize_mxfp8_rows_cute(source, actual.values, actual.scale_rows, actual.scale_mma)
    for name in ("values", "scale_rows", "scale_mma"):
        torch.testing.assert_close(
            getattr(actual, name).view(torch.uint8),
            getattr(reference, name).view(torch.uint8),
            atol=0,
            rtol=0,
        )


def test_cute_migration_mxfp8_quant_trellis_native_mma_order() -> None:
    """The direct-trellis order permutes bytes only within each K32 group."""
    require_b12x()
    m, k = 16, 128
    generator = torch.Generator(device="cuda").manual_seed(46_832)
    source = (
        torch.randn(
            (m, k),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 4
    ).contiguous()
    expected = quantize_mxfp8_rows_torch(source)
    actual = empty_mxfp8_rows_for_dense_gemm(m, k, device="cuda")

    def run() -> None:
        quantize_mxfp8_rows_cute(
            source,
            actual.values,
            actual.scale_rows,
            actual.scale_mma,
            value_order="trellis_native_mma",
        )

    run()
    torch.cuda.synchronize()
    perm = torch.tensor(
        (
            0,
            1,
            8,
            9,
            4,
            5,
            12,
            13,
            2,
            3,
            10,
            11,
            6,
            7,
            14,
            15,
            20,
            21,
            28,
            29,
            16,
            17,
            24,
            25,
            22,
            23,
            30,
            31,
            18,
            19,
            26,
            27,
        ),
        device="cuda",
    )
    expected_values = (
        expected.values.view(torch.uint8)
        .reshape(m, k // 32, 32)[:, :, perm]
        .reshape(m, k)
    )
    torch.testing.assert_close(
        actual.values.view(torch.uint8), expected_values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.scale_rows.view(torch.uint8),
        expected.scale_rows.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scale_mma.view(torch.uint8),
        expected.scale_mma.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    source.copy_(
        torch.randn(
            source.shape,
            generator=generator,
            dtype=source.dtype,
            device=source.device,
        )
        / 4
    )
    graph.replay()
    torch.cuda.synchronize()
    expected = quantize_mxfp8_rows_torch(source)
    expected_values = (
        expected.values.view(torch.uint8)
        .reshape(m, k // 32, 32)[:, :, perm]
        .reshape(m, k)
    )
    torch.testing.assert_close(
        actual.values.view(torch.uint8), expected_values, rtol=0, atol=0
    )
