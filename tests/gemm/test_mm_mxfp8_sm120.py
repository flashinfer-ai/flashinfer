"""Tests for SM120 MXFP8 GEMM (issue #2728)."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_mxfp8, SfLayout
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


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


def _skip_if_not_b12x():
    _skip_if_not_sm120()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_mxfp8_dsl_supported
    from flashinfer.jit.cpp_ext import get_cuda_version

    if get_cuda_version().major < 13 or not _b12x_mxfp8_dsl_supported():
        pytest.skip("b12x MXFP8 requires CUDA 13+ and nvidia-cutlass-dsl >= 4.6.0")


def _mxfp8_tail_operand(rows, k, *, guard=False):
    x, sf = mxfp8_quantize(
        torch.randn(rows, k, device="cuda", dtype=torch.bfloat16),
        sf_swizzle_layout=SfLayout.layout_128x4,
    )
    padded_k = (k + 127) // 128 * 128
    padded = torch.zeros(rows, padded_k, device="cuda", dtype=x.dtype)
    padded.view(torch.uint8)[:, :k].copy_(x.view(torch.uint8))
    padded_sf = sf.clone()
    if k % 128:
        # [row block, K tile, row % 32, row // 32 % 4, scale group].
        # Poison nonexistent groups, including code 255 (UE8M0 NaN).
        sf.view(-1, padded_k // 128, 32, 4, 4)[:, -1, :, :, k % 128 // 32 :] = 255
        padded_sf.view(-1, padded_k // 128, 32, 4, 4)[:, -1, :, :, k % 128 // 32 :] = (
            127
        )
    if guard:
        # A contiguous logical tensor followed by FP8 NaNs; row strides and
        # the TMA descriptor must still describe K, not padded_k.
        storage = torch.full((rows * k + 128,), 127, device="cuda", dtype=torch.uint8)
        storage[: rows * k].copy_(x.view(torch.uint8).flatten())
        x = storage[: rows * k].view(x.dtype).view(rows, k)
    return x, sf, padded, padded_sf


@pytest.mark.parametrize("m", [1, 6, 512])
@pytest.mark.parametrize("k", [128, 160, 192, 224, 256, 544, 576, 608, 640])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_b12x_tail_padding_parity(m, k, out_dtype):
    _skip_if_not_b12x()
    torch.manual_seed(5175)
    a, sa, ap, sap = _mxfp8_tail_operand(m, k, guard=m == 6)
    b, sb, bp, sbp = _mxfp8_tail_operand(5120, k, guard=m == 6)
    reference = mm_mxfp8(ap, bp.T, sap, sbp, out_dtype=out_dtype, backend="b12x")
    out = torch.full_like(reference, float("nan"))
    result = mm_mxfp8(a, b.T, sa, sb, out=out, out_dtype=out_dtype, backend="b12x")
    assert result.data_ptr() == out.data_ptr()
    assert torch.isfinite(result).all()
    # Both paths use the same quantized operands and group order. The padded
    # reference adds only exact zeros; no quantization tolerance is needed.
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("k", [544, 576, 608])
def test_mm_mxfp8_b12x_tail_auto_graph(k, monkeypatch):
    _skip_if_not_b12x()
    from flashinfer.gemm import gemm_base

    torch.manual_seed(5175)
    a, sa, ap, sap = _mxfp8_tail_operand(6, k)
    b, sb, bp, sbp = _mxfp8_tail_operand(5120, k)
    reference = mm_mxfp8(ap, bp.T, sap, sbp, backend="b12x")
    factory = gemm_base._b12x_gemm_mxfp8_runner
    calls = []

    def traced_factory(*args, **kwargs):
        runner = factory(*args, **kwargs)
        forward = runner.forward

        def traced_forward(*args, **kwargs):
            calls.append(True)
            return forward(*args, **kwargs)

        runner.forward = traced_forward
        return runner

    monkeypatch.setattr(gemm_base, "_b12x_gemm_mxfp8_runner", traced_factory)
    out = torch.empty_like(reference)
    # Warm both the compiled kernel and alpha cache before capture.
    mm_mxfp8(a, b.T, sa, sb, out=out, backend="auto")
    assert calls, "auto did not execute the b12x runner"
    assert mm_mxfp8.suitable_auto_backends[0] == "b12x"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mm_mxfp8(a, b.T, sa, sb, out=out, backend="auto")
    for replay in range(6):
        if replay == 3:
            # Replay must read current input, not capture-time intermediates.
            a.view(torch.uint8).bitwise_xor_(128)
        out.fill_(float("nan"))
        graph.replay()
        expected = reference if replay < 3 else -reference
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k", [544, 576, 608])
@pytest.mark.parametrize("m,n", [(6, 5120), (129, 160)])
def test_mm_mxfp8_b12x_tail_all_tactics(m, n, k):
    _skip_if_not_b12x()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_runner

    torch.manual_seed(5175)
    a, sa, ap, sap = _mxfp8_tail_operand(m, k)
    b, sb, bp, sbp = _mxfp8_tail_operand(n, k)
    reference = mm_mxfp8(ap, bp.T, sap, sbp, backend="b12x")
    out = torch.empty_like(reference)
    major, minor = get_compute_capability(a.device)
    runner = _b12x_gemm_mxfp8_runner(major, minor, True, out.dtype)
    inputs = [a, b.T, sa, sb, out.dtype, out, None]
    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics
    for tactic in tactics:
        out.fill_(float("nan"))
        runner.forward(inputs, tactic=tactic)
        torch.testing.assert_close(out, reference, rtol=0, atol=0, msg=str(tactic))


@pytest.mark.parametrize("k", [528, 560, 592])
def test_mm_mxfp8_b12x_tail_rejects_incomplete_scale_group(k):
    _skip_if_not_b12x()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_requirement

    a = torch.empty(6, k, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty(5120, k, device="cuda", dtype=a.dtype)
    sa = torch.empty(128 * ((k + 127) // 128 * 4), device="cuda", dtype=torch.uint8)
    sb = torch.empty(5120 * ((k + 127) // 128 * 4), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiple of 32"):
        mm_mxfp8(a, b.T, sa, sb, backend="b12x")
    assert not _b12x_gemm_mxfp8_requirement(
        a, b.T, sa, sb, use_8x4_sf_layout=False, backend="auto"
    )
