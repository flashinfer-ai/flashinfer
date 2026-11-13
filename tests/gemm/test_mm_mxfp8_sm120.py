"""Tests for SM120 MXFP8 GEMM (issue #2728)."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_mxfp8, SfLayout
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


def _exact_mxfp8_operands(m, n, k):
    a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16)
    aq, bq, sa, sb = _prepare_mxfp8(a, b, swizzled=True)
    aq.copy_(a.to(torch.float8_e4m3fn))
    bq.copy_(b.to(torch.float8_e4m3fn))
    sa.fill_(127)  # E8M0 exponent for scale 1.
    sb.fill_(127)
    return aq, bq.T, sa, sb, a.float() @ b.float().T


def _skip_if_b12x_unavailable():
    _skip_if_not_sm120()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_mxfp8_dsl_supported
    from flashinfer.jit.cpp_ext import get_cuda_version

    if get_cuda_version().major < 13 or not _b12x_mxfp8_dsl_supported():
        pytest.skip("b12x MXFP8 requires CUDA 13 and CuTe DSL >= 4.6")


@pytest.mark.parametrize("m", [1, 2, 129])
@pytest.mark.parametrize("n", [128, 129, 130, 136, 255])
@pytest.mark.parametrize("k", [128, 384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["b12x", "auto"])
def test_mm_mxfp8_sm120_output_row_alignment(m, n, k, out_dtype, backend):
    """Tail-N output rows must use their actual stride without touching guards."""
    _skip_if_b12x_unavailable()
    a, b, sa, sb, reference = _exact_mxfp8_operands(m, n, k)
    # Keep the pointer aligned and guard both ends of the supplied output.
    storage = torch.full((m * n + 256,), -123, device="cuda", dtype=out_dtype)
    out = storage[128:-128].view(m, n)
    mm_mxfp8(a, b, sa, sb, out=out, out_dtype=out_dtype, backend=backend)
    torch.testing.assert_close(out, reference.to(out_dtype), rtol=0, atol=0)
    assert torch.all(storage[:128] == -123)
    assert torch.all(storage[-128:] == -123)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_tail_n_tactics_and_graph(out_dtype):
    """Every b12x tactic handles both epilogues when its cache is reused."""
    _skip_if_b12x_unavailable()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_runner

    runner = _b12x_gemm_mxfp8_runner(12, 0, True, out_dtype)
    cases = [(17, n, 384) for n in (128, 129, 130, 131, 132, 133, 134, 135, 136, 129)]
    # Every advertised tile requires more CTAs than the SM count here.
    cases.append((513, 8193, 128))
    cases.append((513, 8193, 384))
    for m, n, k in cases:
        a, b, sa, sb, reference = _exact_mxfp8_operands(m, n, k)
        out = torch.empty((m, n), device="cuda", dtype=out_dtype)
        inputs = [a, b, sa, sb, None, out, None]
        tactics = runner.get_valid_tactics(inputs, None)
        assert tactics
        for tactic in tactics:
            runner.forward(inputs, tactic=tactic)
            torch.testing.assert_close(out, reference.to(out_dtype), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner.forward(inputs)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            runner.forward(inputs)
        a.zero_()
        graph.replay()
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["b12x", "auto"])
def test_mm_mxfp8_sm120_tail_n_autotune(backend):
    """Tuned tactics remain correct across M buckets and output alignments."""
    _skip_if_b12x_unavailable()
    from flashinfer import autotune

    for m, n in ((1, 129), (3, 129), (6, 136), (17, 129), (129, 257)):
        a, b, sa, sb, reference = _exact_mxfp8_operands(m, n, 384)
        with autotune(True):
            result = mm_mxfp8(a, b, sa, sb, backend=backend)
        torch.testing.assert_close(result, reference.to(result.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("m,n", [(17, 129), (513, 8193)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_tail_n_block_scales(m, n, out_dtype):
    """Predicated stores preserve independently scaled K blocks and output rows."""
    _skip_if_b12x_unavailable()
    k = 384

    def operand(rows):
        values = torch.randint(-2, 3, (rows, k), device="cuda").float()
        exponents = torch.randint(-3, 4, (rows, k // 32), device="cuda")
        scales = torch.pow(2.0, exponents).repeat_interleave(32, dim=1)
        return (values * scales).to(torch.bfloat16)

    a, b = operand(m), operand(n)
    aq, bq, sa, sb = _prepare_mxfp8(a, b, swizzled=True)
    result = mm_mxfp8(aq, bq.T, sa, sb, out_dtype=out_dtype, backend="b12x")
    reference = (a.float() @ b.float().T).to(out_dtype)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def _is_sm120_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] == 12
    except RuntimeError:
        return False


def _skip_if_not_sm120():
    if not _is_sm120_available():
        pytest.skip("Requires SM12x GPU")


def _prepare_mxfp8(a_bf16, b_bf16, swizzled: bool):
    sflayout = SfLayout.layout_128x4 if swizzled else SfLayout.layout_linear
    a_fp8, a_sf = mxfp8_quantize(a_bf16, sf_swizzle_layout=sflayout)
    b_fp8, b_sf = mxfp8_quantize(b_bf16, sf_swizzle_layout=sflayout)
    if not swizzled:
        m, k = a_bf16.shape
        n = b_bf16.shape[0]
        a_sf = a_sf.view(m, k // 32)
        b_sf = b_sf.view(n, k // 32).t()
    return a_fp8, b_fp8, a_sf, b_sf


# Swizzled (layout_128x4) scale: mxfp8_quantize pads the scale buffer to pad_up(M, 128)
# rows internally, so arbitrary M is supported.
@pytest.mark.parametrize("m", [1, 17, 100, 128, 256, 512, 1024])
@pytest.mark.parametrize("n", [128, 256, 512, 1024])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_swizzled(m, n, k, out_dtype):
    _skip_if_not_sm120()

    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    result = mm_mxfp8(
        a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=out_dtype, backend="cutlass"
    )

    assert result.shape == (m, n)
    assert result.dtype == out_dtype
    assert torch.isfinite(result).all(), "Output contains NaN/Inf"

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99 for M={m},N={n},K={k}"


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_all_tactics(out_dtype):
    """Every advertised tactic must run and match the reference.

    No hardcoded tactic count or index list: whatever the module advertises
    has to work on the device the test runs on.
    """
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    m, n, k = 512, 512, 512
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    assert num_tactics > 0
    workspace = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for tactic in range(num_tactics):
        out = torch.empty(m, n, dtype=out_dtype, device="cuda")
        module.mxfp8_gemm(a_fp8, b_fp8, a_sf, b_sf, out, workspace, tactic)
        assert torch.isfinite(out).all(), f"tactic {tactic}: output contains NaN/Inf"
        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
        ).item()
        assert cos_sim > 0.99, f"tactic {tactic}: cos_sim={cos_sim:.4f}"


def test_mm_mxfp8_sm120_auto_tactic():
    """Verify SM120 MXFP8 produces correct results (tactic auto-selected)."""
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    m, n, k = 256, 256, 256
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    assert num_tactics > 0

    result = mm_mxfp8(
        a_fp8,
        b_fp8.T,
        a_sf,
        b_sf,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    assert result.shape == (m, n)
    assert torch.isfinite(result).all()
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.98, f"cos_sim={cos_sim:.4f}"


def test_mm_mxfp8_sm120_rejects_linear_scales():
    """SM120 CUTLASS MXFP8 must reject non-swizzled (2D) scale tensors."""
    _skip_if_not_sm120()

    m, n, k = 128, 128, 128
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=False)

    with pytest.raises((RuntimeError, ValueError)):
        mm_mxfp8(
            a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="cutlass"
        )
