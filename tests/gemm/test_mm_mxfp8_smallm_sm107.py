"""Tests for the "specialized" backend of mm_mxfp8 (SM107 small-M kernel).

Dispatch contract under test: backend="auto" routes to the specialized kernel
only on its strict (M, N, K) allowlist; an explicitly selected backend is
never silently swapped and never falls back (explicit "cute-dsl" never gets
the specialized kernel, explicit "specialized" that cannot serve a call
raises NotImplementedError with the refusal reason);
FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 disables the kernel (auto: stock
path, explicit "specialized": raise). GPU tests are skipped unless the device
is SM107 (compute capability 10.7).
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_mxfp8, SfLayout
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.gemm import mxfp8_smallm_sm107 as specialized
from flashinfer.gemm.gemm_base import MM_MXFP8_BACKENDS
from flashinfer.gemm.mxfp8_smallm_sm107 import (
    _DISABLE_ENV,
    _load_allowlist,
    _mm_mxfp8_specialized_stats,
    _swizzled_sf_numel,
)
from flashinfer.utils import get_compute_capability

_MIN_COS_SIM = 0.98

# Every allowlisted shape: each has its own compile-time route, so each must
# run numerically. Derived from the shipped dispatch surface so the test can
# never drift from the allowlist.
_ALLOWLIST = _load_allowlist()
assert _ALLOWLIST is not None, "specialized allowlist must ship with the package"
_ON_LIST_SHAPES = sorted(_ALLOWLIST["shapes"])
# A neighbor of the measured families that is NOT allowlisted.
_OFF_LIST_SHAPE = (20, 7168, 5120)


def _is_sm107() -> bool:
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda")) == (10, 7)


requires_sm100plus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="mxfp8_quantize backend requires SM100 or newer",
)

requires_sm107 = pytest.mark.skipif(
    not _is_sm107(),
    reason="mm_mxfp8 specialized backend requires an SM107 (compute 10.7) GPU",
)


def _isolate_autotuner_cache(monkeypatch):
    """Swap in a fresh AutoTuner singleton for the test (restored by
    monkeypatch): routing assertions that count specialized-kernel launches
    must exercise the default (heuristic) runner, not a stock winner some
    earlier tuned run left in the process-wide profiling cache."""
    from flashinfer.autotuner.autotuner import AutoTuner

    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())


def test_allowlist_is_well_formed():
    """CPU-side invariants of the packaged dispatch allowlist."""
    allowlist = _load_allowlist()
    assert allowlist is not None
    assert allowlist["cc"] == (10, 7)
    shapes = allowlist["shapes"]
    assert _OFF_LIST_SHAPE not in shapes
    for m, n, k in shapes:
        assert 1 <= m <= 32, (m, n, k)
        assert k % 32 == 0, (m, n, k)
        # the kernel's vector stores rely on 8-byte-aligned bf16 output rows
        assert n % 4 == 0, (m, n, k)
    assert len(shapes) == 34
    for shape in _ON_LIST_SHAPES:
        assert shape in shapes, shape
    # measured rows that did not beat the strongest stock configuration on
    # both stock arms (latest release and current main) must stay excluded
    for excluded in [
        (16, 8192, 1280),
        (24, 8192, 1280),
        (16, 5120, 576),
        (16, 1792, 5120),
        (24, 6144, 384),
        (24, 5120, 576),
        (24, 6144, 1024),
        (32, 6144, 1024),
        (24, 5120, 2048),
    ]:
        assert excluded not in shapes, excluded


def _fp8_cpu_view(shape, offset):
    """A contiguous fp8 CPU tensor whose data pointer sits `offset` bytes
    into its storage (fresh allocations are 256-byte aligned, so `offset`
    controls the pointer's alignment class)."""
    n = shape[0] * shape[1]
    base = torch.zeros(n + 256, dtype=torch.uint8)
    return base[offset : offset + n].view(torch.float8_e4m3fn).view(shape)


def test_offset_operand_view_declines_with_pointer_alignment(monkeypatch):
    """The kernel's 32-byte ldg256 operand loads require aligned pointers;
    a contiguous view into offset storage must be declined CPU-side."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = 4, 6144, 384
    a = _fp8_cpu_view((m, k), 16)
    b = _fp8_cpu_view((n, k), 0).t()
    out = torch.empty(m, n, dtype=torch.bfloat16)
    sf = torch.zeros(_swizzled_sf_numel(max(m, n), k), dtype=torch.uint8)
    assert (
        specialized._common_supported(a, b, sf, sf, out, torch.bfloat16, False)
        == "pointer_alignment"
    )
    # aligned control: the same metadata with offset 0 passes this guard
    a0 = _fp8_cpu_view((m, k), 0)
    assert (
        specialized._common_supported(a0, b, sf, sf, out, torch.bfloat16, False)
        != "pointer_alignment"
    )


def test_strided_scale_declines_with_scale_layout():
    """A strided 1D scale view passes the ndim/dtype checks but interleaves
    foreign bytes; the layout guard must decline it CPU-side."""
    m, n, k = 4, 6144, 384
    a = _fp8_cpu_view((m, k), 0)
    b = _fp8_cpu_view((n, k), 0).t()
    sf = torch.zeros(2 * _swizzled_sf_numel(n, k), dtype=torch.uint8)
    strided = sf[::2]
    assert (
        specialized._layout_supported(a, b, strided, strided, False) == "scale_layout"
    )


@requires_sm107
@pytest.mark.parametrize("index,offset", [(0, 16), (1, 16), (2, 1), (3, 1), (4, 1)])
def test_specialized_misaligned_refusal(index, offset, monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = 4, 6144, 384
    a, sa = _const_operand(m, k, 1, 127)
    b, sb = _const_operand(n, k, 1, 127)
    args = [a, b, sa, sb, torch.empty((m, n), device="cuda", dtype=torch.bfloat16)]
    t = args[index]
    args[index] = torch.empty(t.numel() + offset, device=t.device, dtype=t.dtype)[
        offset:
    ].view(t.shape)
    a, b, sa, sb, out = args
    assert args[index].is_contiguous()
    assert (
        specialized.specialized_mm_mxfp8_eligible(a, b.T, sa, sb, out, out.dtype, False)
        == "pointer_alignment"
    )
    with pytest.raises(NotImplementedError, match="pointer_alignment"):
        mm_mxfp8(a, b.T, sa, sb, out=out, backend="specialized")
    module = specialized._get_specialized_module()
    assert module is not None
    with pytest.raises(RuntimeError, match="pointer_alignment"):
        module.mm_mxfp8_smallm_sm107_run(*args)


@requires_sm107
@pytest.mark.parametrize("other_gpu", [False, True], ids=["cpu", "other_gpu"])
@pytest.mark.parametrize("index", range(5))
def test_direct_ffi_rejects_misplaced_tensor(other_gpu, index):
    if other_gpu and torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    module = specialized._get_specialized_module()
    assert module is not None
    m, n, k = 4, 6144, 384
    a, sa = _const_operand(m, k, 1, 127)
    b, sb = _const_operand(n, k, 1, 127)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    args = [a, b, sa, sb, out]
    args[index] = (
        args[index].to(
            torch.device("cuda", (a.device.index + 1) % torch.cuda.device_count())
        )
        if other_gpu
        else args[index].cpu()
    )
    names = ["aq", "bq", "a_sf", "b_sf", "result"]
    expected = (
        "aq must be a CUDA tensor"
        if index == 0 and not other_gpu
        else (
            "bq must be on the same device as aq"
            if index == 0
            else names[index] + " must be on the same device as aq"
        )
    )
    with pytest.raises(ValueError, match=expected):
        module.mm_mxfp8_smallm_sm107_run(*args)

    # The same buffers with the original scale must still launch correctly.
    module.mm_mxfp8_smallm_sm107_run(a, b, sa, sb, out)
    assert torch.equal(out, torch.full_like(out, k))


@requires_sm107
def test_generic_launch_error_crosses_ffi():
    # M=1 enters gen_m<1>; K=128 gives one valid K group. N=0 makes
    # grid.x zero at the reachable, non-clustered launch_gen site.
    module = specialized._get_specialized_module()
    assert module is not None
    a, sa = _const_operand(1, 128, 1, 127)
    b, sb = _const_operand(1, 128, 1, 127)
    out = torch.empty((1, 0), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="fallback kernel launch failed"):
        module.mm_mxfp8_smallm_sm107_run(a, b[:0], sa, sb, out)


@requires_sm107
@pytest.mark.parametrize("m,n,k", [(1, 129, 128), (4, 6144, 384), (8, 1152, 5120)])
def test_direct_epilogue_lanes_and_canaries(m, n, k):
    module = specialized._get_specialized_module()
    assert module is not None
    a, sa = _const_operand(m, k, 1, 127)
    b, sb = _const_operand(n, k, 1, 127)
    values = (1 + torch.arange(n, device="cuda") % 4).float()
    b.copy_(values[:, None].expand(n, k))
    storage = torch.full((m * n + 16,), -1, dtype=torch.bfloat16, device="cuda")
    out = storage[8:-8].view(m, n)
    module.mm_mxfp8_smallm_sm107_run(a, b, sa, sb, out)
    assert torch.equal(out, (values * k).bfloat16().expand(m, n))
    assert (storage[:8] == -1).all() and (storage[-8:] == -1).all()


def test_backends_declaration():
    assert MM_MXFP8_BACKENDS[0] == "auto"
    assert "specialized" in MM_MXFP8_BACKENDS
    assert all(isinstance(b, str) for b in MM_MXFP8_BACKENDS)
    stats = _mm_mxfp8_specialized_stats()
    assert (
        stats["kill_switch_env"]
        == _DISABLE_ENV
        == ("FLASHINFER_SPECIALIZED_KERNEL_DISABLE")
    )
    assert "fallback_reasons" in stats and "backend_dispatches" in stats


def test_swizzled_sf_numel_matches_quantizer_padding():
    for rows, k in [
        (1, 384),
        (24, 6144),
        (32, 1024),
        (6144, 384),
        (768, 6144),
        (4, 576),  # 18 scale columns pad to 20
    ]:
        expected = ((rows + 127) // 128 * 128) * ((k // 32 + 3) // 4 * 4)
        assert _swizzled_sf_numel(rows, k) == expected


@requires_sm100plus
def test_swizzled_sf_numel_matches_quantizer_buffer():
    """The helper must agree with the scale buffer mxfp8_quantize actually
    returns: a padded class (K=576, 18 scale columns pad to 20) and an
    unpadded control (K=384)."""
    for rows, k in [(4, 576), (4, 384)]:
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        _, sf = mxfp8_quantize(x, sf_swizzle_layout=SfLayout.layout_128x4)
        assert sf.numel() == _swizzled_sf_numel(rows, k), (rows, k, sf.numel())


def _reference_swizzle(lin, rows, k):
    """Reference 128x4 swizzler (the layout contract of mxfp8_quantize's
    swizzled output): pad rows to a multiple of 128 and the per-row scale
    count (k // 32) to a multiple of 4, then interleave
    (row_tile, 4, 32, col_tile, 4) -> (row_tile, col_tile, 32, 4, 4)."""
    k4 = (k // 32 + 3) // 4 * 4
    r_pad = (rows + 127) // 128 * 128
    full = torch.full((r_pad, k4), 127, dtype=torch.uint8)
    full[:rows, : k // 32] = lin
    t = full.view(r_pad // 128, 4, 32, k4 // 4, 4).permute(0, 3, 2, 1, 4)
    return t.contiguous().view(-1)


@requires_sm100plus
def test_unswizzle_dequant_roundtrip_padded_k():
    """quantize(swizzle) -> _unswizzle_sf -> _dequant roundtrip on CPU,
    focused on the padded classes: k // 32 % 4 != 0 (scale-column padding)
    and rows not a multiple of 128 (row padding)."""
    torch.manual_seed(0)
    for rows, k in [(4, 160), (20, 416), (7, 96), (24, 5152), (130, 224), (1, 32)]:
        lin = torch.randint(100, 140, (rows, k // 32), dtype=torch.uint8)
        swizzled = _reference_swizzle(lin, rows, k)
        assert swizzled.numel() == _swizzled_sf_numel(rows, k), (rows, k)
        unswizzled = specialized._unswizzle_sf(swizzled, rows, k)
        assert unswizzled.shape == (rows, k // 32), (rows, k, unswizzled.shape)
        assert torch.equal(unswizzled, lin), (rows, k)
        # dequant consumes the unswizzled scales at the right geometry and
        # a scale byte of 127 (2^0) reproduces the fp8 values exactly.
        q = torch.randn(rows, k).to(torch.float8_e4m3fn)
        ones = torch.full((rows, k // 32), 127, dtype=torch.uint8)
        assert torch.equal(specialized._dequant(q, ones), q.float()), (rows, k)


def _quantized_problem(m, n, k):
    torch.manual_seed(m * 1000003 + n * 101 + k)
    a_bf16 = torch.randn([m, k], device="cuda", dtype=torch.bfloat16) * 0.25
    b_bf16 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16) * 0.25
    a_q, a_sf = mxfp8_quantize(a_bf16, sf_swizzle_layout=SfLayout.layout_128x4)
    b_q, b_sf = mxfp8_quantize(b_bf16, sf_swizzle_layout=SfLayout.layout_128x4)
    reference = torch.mm(a_bf16, b_bf16.T)
    return a_q, b_q, a_sf, b_sf, reference


def _count_kernel_launches(monkeypatch):
    """Count actual specialized-kernel launches, wherever dispatch enters."""
    calls = []
    real_launch = specialized._launch_specialized

    def counting_launch(*args, **kwargs):
        served = real_launch(*args, **kwargs)
        if served:
            calls.append(1)
        return served

    monkeypatch.setattr(specialized, "_launch_specialized", counting_launch)
    return calls


def _assert_close_to_reference(reference, result, context):
    assert torch.isfinite(result).all(), f"{context}: output contains NaN/Inf"
    cos_sim = F.cosine_similarity(
        reference.float().reshape(-1), result.float().reshape(-1), dim=0
    ).item()
    assert cos_sim > _MIN_COS_SIM, f"{context}: cosine similarity {cos_sim:.4f}"
    # Cosine similarity is scale-blind: also pin the magnitude, so a result
    # that is a correctly-directed multiple of the reference fails.
    norm_ratio = (
        result.float().norm() / reference.float().norm().clamp_min(1e-20)
    ).item()
    assert 0.9 < norm_ratio < 1.1, f"{context}: norm ratio {norm_ratio:.4f}"


def test_reference_gate_rejects_scaled_output():
    """The acceptance gate itself must fail a magnitude error that cosine
    similarity alone would accept."""
    reference = torch.randn(64, 64)
    with pytest.raises(AssertionError, match="norm ratio"):
        _assert_close_to_reference(reference, reference * 2.0, "doubled")


@requires_sm107
@pytest.mark.parametrize("m,n,k", _ON_LIST_SHAPES)
def test_auto_routes_to_specialized_on_allowlist(m, n, k, monkeypatch):
    """backend="auto" on allowlisted shapes runs the specialized kernel and
    matches both the bf16 reference and the stock path."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    _isolate_autotuner_cache(monkeypatch)
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)

    fast = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 1, "allowlisted auto call must run the specialized kernel"
    assert fast.shape == (m, n) and fast.dtype == torch.bfloat16
    _assert_close_to_reference(reference, fast, f"specialized ({m},{n},{k})")

    # The stock path on the same quantized inputs (kill-switch engaged).
    monkeypatch.setenv(_DISABLE_ENV, "1")
    stock = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 1, "kill-switch run must not run the specialized kernel"
    _assert_close_to_reference(reference, stock, f"stock path ({m},{n},{k})")

    # Both consume the identical quantized operands, so they must agree
    # tightly with each other (fp32 accumulation in both kernels).
    cos_sim = F.cosine_similarity(
        stock.float().reshape(-1), fast.float().reshape(-1), dim=0
    ).item()
    assert cos_sim > 0.999, f"specialized vs stock cosine similarity {cos_sim:.5f}"


@requires_sm107
def test_explicit_backend_is_never_swapped(monkeypatch):
    """Explicit backend="cute-dsl" on an allowlisted shape must get cute-dsl,
    never the specialized kernel."""
    from flashinfer.cute_dsl.availability import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip(
            "explicit backend='cute-dsl' requires the optional CuTe DSL "
            "package; the specialized-backend tests do not"
        )
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _ON_LIST_SHAPES[0]
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="cute-dsl")
    assert len(calls) == 0, (
        "explicit cute-dsl must never receive the specialized kernel"
    )
    _assert_close_to_reference(reference, res, "explicit cute-dsl")


@requires_sm107
def test_explicit_specialized_serves_allowlist(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _ON_LIST_SHAPES[1]
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 1
    _assert_close_to_reference(reference, res, "explicit specialized")


def _exact_dequant_reference(a_q, b_q, a_sf, b_sf):
    """Exact fp32 dequant + torch.mm on the quantized operands: the composable
    reference the kernel must reproduce up to fp32 accumulation order and the
    final bf16 rounding of the output."""
    m, k = a_q.shape
    n = b_q.shape[0]
    a_lin = specialized._unswizzle_sf(a_sf, m, k)
    b_lin = specialized._unswizzle_sf(b_sf, n, k)
    a_f = specialized._dequant(a_q, a_lin)
    b_f = specialized._dequant(b_q, b_lin)
    return torch.mm(a_f, b_f.T)


def _assert_matches_exact_reference(reference, result, context):
    """Tight acceptance against the exact-dequant reference: both sides
    consume identical quantized operands, so the only admissible differences
    are fp32 accumulation order and the bf16 output rounding."""
    assert torch.isfinite(result).all(), f"{context}: output contains NaN/Inf"
    ref = reference.float()
    res = result.float()
    denom = ref.abs().max().item()
    if denom == 0.0:
        # An exactly-zero reference admits only an exactly-zero result; a
        # floor on the denominator would instead accept anything small.
        assert res.abs().max().item() == 0.0, (
            f"{context}: nonzero result for an all-zero reference"
        )
        return
    rel_err = ((res - ref).abs().max() / denom).item()
    assert rel_err < 1e-2, f"{context}: max relative error {rel_err:.2e}"


@requires_sm107
def test_specialized_matches_exact_reference_on_every_allowlisted_shape(
    monkeypatch,
):
    """Every shipped allowlist shape runs the kernel numerically (each exact-M
    route, including the partial-tile M=2 / M=12 routes, is its own
    compile-time instantiation) and matches the exact fp32 dequant reference
    with a tight tolerance."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    allowlist = _load_allowlist()
    assert allowlist is not None
    calls = _count_kernel_launches(monkeypatch)
    for served, (m, n, k) in enumerate(sorted(allowlist["shapes"]), start=1):
        a_q, b_q, a_sf, b_sf, _ = _quantized_problem(m, n, k)
        reference = _exact_dequant_reference(a_q, b_q, a_sf, b_sf)
        res = mm_mxfp8(
            a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
        )
        assert len(calls) == served, (
            f"({m},{n},{k}): allowlisted explicit call must run the kernel"
        )
        _assert_matches_exact_reference(reference, res, f"allowlist ({m},{n},{k})")


@requires_sm107
def test_specialized_small_scale_precise_path(monkeypatch):
    """A dequantizes to 2^-136, below half the smallest BF16 subnormal.
    Ordinary BF16 MMA would return zero. Neither scale triggers rebalancing.
    """
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = (4, 5120, 576)
    a_q, a_sf = _const_operand(m, k, 2.0**-9, 0)
    b_q, b_sf = _const_operand(n, k, 256, 127)
    reference = torch.full((m, n), k * 2.0**-128, dtype=torch.bfloat16, device="cuda")
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 1
    assert torch.equal(reference, res)
    # Keep the separate zero-output negative control for the relative-error helper.
    with pytest.raises(AssertionError):
        _assert_matches_exact_reference(
            reference, torch.zeros_like(res), "small-scale all-zero guard"
        )


def _const_operand(rows, k, value, scale_byte, device="cuda"):
    """E4M3 operand filled with ``value`` plus a valid padded/swizzled scale
    buffer filled with the UE8M0 byte ``scale_byte``."""
    q = torch.full((rows, k), float(value), dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    sf = torch.full(
        (specialized._swizzled_sf_numel(rows, k),),
        scale_byte,
        dtype=torch.uint8,
        device=device,
    )
    return q, sf


@requires_sm107
@pytest.mark.parametrize("m,n,k", [(4, 6144, 384), (4, 5120, 576), (8, 1152, 5120)])
def test_extreme_scale_exponents_match_exact_value(m, n, k, monkeypatch):
    """Reviewer-reported extreme MXFP8 exponent range: UE8M0 scale byte 254 on
    an operand of 4 dequantizes to 2^129 — past BF16 (and FP32) range — while
    every product term is an exact finite 4 (the B blocks carry scale byte 0).
    The kernel must return the exact K*4, matching the tensor-core reference,
    not inf: such k groups take the rebalanced precise path, which applies the
    combined block-scale exponent after the raw-value product."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    a_q, a_sf = _const_operand(m, k, 4.0, 254)
    b_q, b_sf = _const_operand(n, k, 1.0, 0)
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 1
    expected = torch.full((m, n), float(4 * k), dtype=torch.bfloat16, device="cuda")
    assert torch.equal(res, expected), (
        f"extreme-exponent result != exact {4 * k} "
        f"(max={res.float().max().item()}, min={res.float().min().item()})"
    )


@requires_sm107
def test_bf16_max_operand_times_zero_is_zero(monkeypatch):
    """Finite BF16-max A straight out of mxfp8_quantize (E4M3 code 256 with
    scale byte 247 — the scaled up-conversion alone would overflow to inf)
    paired with an all-zero quantized B must give exactly zero, matching the
    tensor-core reference, not NaN."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = (4, 6144, 384)
    a_bf16 = torch.full(
        (m, k), torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16, device="cuda"
    )
    b_bf16 = torch.zeros((n, k), dtype=torch.bfloat16, device="cuda")
    a_q, a_sf = mxfp8_quantize(a_bf16, sf_swizzle_layout=SfLayout.layout_128x4)
    b_q, b_sf = mxfp8_quantize(b_bf16, sf_swizzle_layout=SfLayout.layout_128x4)
    # The scenario must actually exercise the overflow-range class.
    a_lin = specialized._unswizzle_sf(a_sf, m, k)
    assert (a_lin >= 240).any(), "test setup must produce UE8M0 scales >= 240"
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 1
    assert torch.equal(res, torch.zeros_like(res)), (
        "BF16-max x zero must be exactly zero, not NaN/inf"
    )


@requires_sm107
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two GPUs",
)
def test_operands_on_non_current_device(monkeypatch):
    """A valid call with every operand on GPU 1 while GPU 0 is current must
    run on GPU 1's stream/device (the FFI binding establishes the operand
    device; the capture check is scoped to a.device) and must restore the
    caller's current device."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = (4, 6144, 384)
    dev1 = torch.device("cuda:1")
    if get_compute_capability(dev1) != (10, 7):
        pytest.skip("second GPU is not SM107")
    a_q, a_sf = _const_operand(m, k, 1.0, 127, device=dev1)
    b_q, b_sf = _const_operand(n, k, 1.0, 127, device=dev1)
    torch.cuda.set_device(0)
    calls = _count_kernel_launches(monkeypatch)
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 1
    assert torch.cuda.current_device() == 0, "caller's current device not restored"
    assert res.device == a_q.device
    expected = torch.full((m, n), float(k), dtype=torch.bfloat16, device=dev1)
    assert torch.equal(res, expected)


@requires_sm107
def test_explicit_specialized_off_list_raises(monkeypatch):
    """Explicit "specialized" off the allowlist raises NotImplementedError
    with the refusal reason (an explicitly selected backend never falls
    back), and the reason is recorded in the dispatch stats."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _OFF_LIST_SHAPE
    a_q, b_q, a_sf, b_sf, _ = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)
    before = _mm_mxfp8_specialized_stats()

    with pytest.raises(NotImplementedError, match="shape_not_allowlisted"):
        mm_mxfp8(
            a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
        )
    assert len(calls) == 0, "off-list explicit call must not run the kernel"
    after = _mm_mxfp8_specialized_stats()
    assert after["explicit_refusals"] == before["explicit_refusals"] + 1
    assert after["fallback_reasons"].get("shape_not_allowlisted", 0) > (
        before["fallback_reasons"].get("shape_not_allowlisted", 0)
    )


@requires_sm107
def test_auto_off_list_takes_stock_path(monkeypatch):
    """backend="auto" off the allowlist never touches the specialized kernel
    and produces the stock path's exact results."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _OFF_LIST_SHAPE
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)

    res = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 0, "off-list auto call must not run the specialized kernel"
    _assert_close_to_reference(reference, res, f"off-list auto ({m},{n},{k})")

    monkeypatch.setenv(_DISABLE_ENV, "1")
    stock = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 0
    assert torch.equal(res, stock), (
        "off-list auto dispatch must produce the stock path's exact results"
    )


@requires_sm107
def test_kill_switch_disables_specialized(monkeypatch):
    _isolate_autotuner_cache(monkeypatch)
    m, n, k = _ON_LIST_SHAPES[0]
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)

    monkeypatch.setenv(_DISABLE_ENV, "1")
    # auto: the stock path serves the call.
    res = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 0, "kill switch must disable the specialized kernel"
    _assert_close_to_reference(reference, res, "kill-switch auto")
    # explicit specialized: raises with the kill_switch reason (an explicitly
    # selected backend never falls back).
    before = _mm_mxfp8_specialized_stats()
    with pytest.raises(NotImplementedError, match="kill_switch"):
        mm_mxfp8(
            a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
        )
    assert len(calls) == 0
    after = _mm_mxfp8_specialized_stats()
    assert after["explicit_refusals"] == before["explicit_refusals"] + 1
    assert after["fallback_reasons"].get("kill_switch", 0) > (
        before["fallback_reasons"].get("kill_switch", 0)
    )

    monkeypatch.delenv(_DISABLE_ENV)
    mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 1, "the kernel must come back once the switch is lifted"


@requires_sm107
def test_auto_non_bf16_out_keeps_stock_path(monkeypatch):
    """A non-bf16 out_dtype on an allowlisted shape must keep the stock path."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _ON_LIST_SHAPES[0]
    a_q, b_q, a_sf, b_sf, _ = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)
    mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.float16)
    assert len(calls) == 0


# ---------------------------------------------------------------------------
# Default-candidate / autotune-availability surface (GPU-free)
# ---------------------------------------------------------------------------


def _cpu_mxfp8_args(m=8, n=768, k=384):
    a = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
    b = torch.zeros((k, n), dtype=torch.float8_e4m3fn)
    a_sf = torch.zeros(specialized._swizzled_sf_numel(m, k), dtype=torch.uint8)
    b_sf = torch.zeros(specialized._swizzled_sf_numel(n, k), dtype=torch.uint8)
    return a, b, a_sf, b_sf


def test_heuristic_ranks_specialized_first_when_eligible(monkeypatch):
    """On its allowlist the specialized kernel is a DEFAULT auto candidate,
    ranked first (untuned auto runs it; the autotuner races it as itself)."""
    from flashinfer.gemm import gemm_base

    a, b, a_sf, b_sf = _cpu_mxfp8_args()
    monkeypatch.setattr(
        gemm_base, "specialized_mm_mxfp8_eligible", lambda *args, **kw: None
    )
    order = gemm_base._heuristic_func_mm_mxfp8(
        ["specialized", "cutedsl_low_latency"], a, b, a_sf, b_sf
    )
    assert order and order[0] == "specialized", order


def test_heuristic_excludes_specialized_when_ineligible(monkeypatch):
    from flashinfer.gemm import gemm_base

    a, b, a_sf, b_sf = _cpu_mxfp8_args()
    monkeypatch.setattr(
        gemm_base,
        "specialized_mm_mxfp8_eligible",
        lambda *args, **kw: "shape_not_allowlisted",
    )
    order = gemm_base._heuristic_func_mm_mxfp8(
        ["specialized", "cutedsl_low_latency"], a, b, a_sf, b_sf
    )
    assert "specialized" not in order, order


def test_runner_declines_ineligible_inputs_cpu():
    """The TunableRunner candidate advertises no tactic off its guards, so
    the autotuner skips it instead of timing a kernel it cannot launch."""
    a, b, a_sf, b_sf = _cpu_mxfp8_args()
    out = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.bfloat16)
    runner = specialized.SpecializedMxfp8SmallMRunner(False)
    inputs = [a, b, a_sf, b_sf, torch.bfloat16, out, None]
    assert runner.get_valid_tactics(inputs, None) == []
    assert runner.supports_inputs(inputs) is False
    assert runner.is_tactic_compatible(inputs, -1) is False
    # The v2 autotuner default-races (runners[0], tactic=-1) even when
    # get_valid_tactics returned [], with bucket-synthesized shapes that can
    # be off the allowlist: forward itself must refuse rather than launch the
    # kernel outside its contract (the profiler records the refusal as a
    # failed tactic).
    with pytest.raises(NotImplementedError, match="cannot serve this profile"):
        runner.forward(inputs, tactic=-1)


def test_eligibility_probe_tolerates_out_none(monkeypatch):
    # An out=None probe whose arguments satisfy every GPU-free guard
    # (allowlisted shape, column-major b, kill switch cleared) must reach
    # the device guard — proving the None tolerance itself, not an earlier
    # refusal — and must never raise.
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _ON_LIST_SHAPES[0]
    a = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
    b = torch.zeros((n, k), dtype=torch.float8_e4m3fn).t()
    a_sf = torch.zeros(_swizzled_sf_numel(m, k), dtype=torch.uint8)
    b_sf = torch.zeros(_swizzled_sf_numel(n, k), dtype=torch.uint8)
    reason = specialized.specialized_mm_mxfp8_eligible(
        a, b, a_sf, b_sf, None, torch.bfloat16, False
    )
    assert reason == "device"
    # The out_dtype guard still answers before the probe substitutes out.
    a, b, a_sf, b_sf = _cpu_mxfp8_args()
    assert (
        specialized.specialized_mm_mxfp8_eligible(
            a, b, a_sf, b_sf, None, torch.float16, False
        )
        == "out_dtype"
    )
