"""Tests for the "specialized" backend of mm_mxfp8 (SM107 small-M kernel).

Dispatch contract under test: backend="auto" routes to the specialized kernel
only on its strict (M, N, K) allowlist; an explicitly selected backend is
never silently swapped (explicit "cute-dsl" never gets the specialized kernel,
explicit "specialized" that cannot serve a call takes the composable path,
never another fused backend); FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 forces
the composable/stock path. GPU tests are skipped unless the device is SM107
(compute capability 10.7).
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

# A spread of allowlisted shapes across the measured families.
_ON_LIST_SHAPES = [
    (4, 6144, 384),
    (1, 5120, 2048),
    (8, 1152, 5120),
    (24, 768, 6144),
    (32, 1536, 6144),
]
# A neighbor of the measured families that is NOT allowlisted.
_OFF_LIST_SHAPE = (20, 7168, 5120)


def _is_sm107() -> bool:
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda")) == (10, 7)


requires_sm107 = pytest.mark.skipif(
    not _is_sm107(),
    reason="mm_mxfp8 specialized backend requires an SM107 (compute 10.7) GPU",
)


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
    for rows, k in [(1, 384), (24, 6144), (32, 1024), (6144, 384), (768, 6144)]:
        expected = ((rows + 127) // 128 * 128) * ((k // 32 + 3) // 4 * 4)
        assert _swizzled_sf_numel(rows, k) == expected


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


@requires_sm107
def test_explicit_specialized_off_list_takes_composable_path(monkeypatch):
    """Explicit "specialized" off the allowlist falls back to the composable
    dequant + torch.mm path (never another fused backend) with a recorded
    reason, and the result matches the quantized-exact reference."""
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    m, n, k = _OFF_LIST_SHAPE
    a_q, b_q, a_sf, b_sf, _ = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)
    before = _mm_mxfp8_specialized_stats()

    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 0, "off-list explicit call must not run the kernel"
    after = _mm_mxfp8_specialized_stats()
    assert after["composable_dispatches"] == before["composable_dispatches"] + 1
    assert after["fallback_reasons"].get("shape_not_allowlisted", 0) > (
        before["fallback_reasons"].get("shape_not_allowlisted", 0)
    )

    # The composable path is dequant-exact for the quantized operands.
    # b_q from mxfp8_quantize is already (N, K) row-major — the layout
    # _unswizzle_sf(b_sf, n, k) scales — so it is dequantized as-is.
    exact = (
        specialized._dequant(a_q, specialized._unswizzle_sf(a_sf, m, k))
        @ specialized._dequant(b_q, specialized._unswizzle_sf(b_sf, n, k)).t()
    )
    assert torch.equal(res, exact.to(torch.bfloat16))


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
def test_kill_switch_forces_composable_or_stock_path(monkeypatch):
    m, n, k = _ON_LIST_SHAPES[0]
    a_q, b_q, a_sf, b_sf, reference = _quantized_problem(m, n, k)
    calls = _count_kernel_launches(monkeypatch)

    monkeypatch.setenv(_DISABLE_ENV, "1")
    # auto: the stock path serves the call.
    res = mm_mxfp8(a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16)
    assert len(calls) == 0, "kill switch must disable the specialized kernel"
    _assert_close_to_reference(reference, res, "kill-switch auto")
    # explicit specialized: the composable path serves the call.
    before = _mm_mxfp8_specialized_stats()
    res = mm_mxfp8(
        a_q, b_q.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="specialized"
    )
    assert len(calls) == 0
    after = _mm_mxfp8_specialized_stats()
    assert after["composable_dispatches"] == before["composable_dispatches"] + 1
    assert after["fallback_reasons"].get("kill_switch", 0) > (
        before["fallback_reasons"].get("kill_switch", 0)
    )
    _assert_close_to_reference(reference, res, "kill-switch explicit specialized")

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


def test_eligibility_probe_tolerates_out_none():
    a, b, a_sf, b_sf = _cpu_mxfp8_args()
    # CPU tensors fail the device guard, never raise.
    reason = specialized.specialized_mm_mxfp8_eligible(
        a, b, a_sf, b_sf, None, torch.bfloat16, False
    )
    assert isinstance(reason, str) and reason
    assert (
        specialized.specialized_mm_mxfp8_eligible(
            a, b, a_sf, b_sf, None, torch.float16, False
        )
        == "out_dtype"
    )
