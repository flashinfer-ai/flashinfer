import pytest
import torch

from flashinfer import mm_bf16, plan_mm_bf16
from flashinfer.gemm.gemm_base import _MM_BF16_PLAN_TABLES
from flashinfer.utils import get_compute_capability


def _skip_unless_supported():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    cc = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_compute_capability_supported(cc):
        pytest.skip(f"mm_bf16 not supported on sm{cc}.")


def _make_weight(n: int, k: int):
    # (k, n) column-major, the layout mm_bf16 / plan_mm_bf16 expect.
    return torch.randn(n, k, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("n,k", [(128, 256), (2048, 512)])
def test_plan_numerics_across_m(has_bias, n, k):
    """run() must match F.linear for bucket and non-bucket M values."""
    _skip_unless_supported()
    torch.manual_seed(0)
    b = _make_weight(n, k)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if has_bias else None
    plan = plan_mm_bf16(b, bias=bias, max_m=256)
    with torch.inference_mode():
        for m in [1, 3, 24, 40, 100, 233, 256]:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            got = plan.run(a)
            ref = torch.nn.functional.linear(a, b.transpose(-2, -1), bias)
            torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_plan_out_param_and_zero_m():
    _skip_unless_supported()
    torch.manual_seed(0)
    n, k = 128, 256
    b = _make_weight(n, k)
    plan = plan_mm_bf16(b, max_m=64)
    with torch.inference_mode():
        a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(16, n, device="cuda", dtype=torch.bfloat16)
        got = plan.run(a, out=out)
        assert got is out
        ref = torch.nn.functional.linear(a, b.transpose(-2, -1))
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

        empty = plan.run(torch.empty(0, k, device="cuda", dtype=torch.bfloat16))
        assert empty.shape == (0, n)


def test_plan_m_above_max_falls_back():
    """M beyond the tuned range must still be correct (auto-dispatch path)."""
    _skip_unless_supported()
    torch.manual_seed(0)
    n, k = 128, 256
    b = _make_weight(n, k)
    plan = plan_mm_bf16(b, max_m=64)
    with torch.inference_mode():
        a = torch.randn(100, k, device="cuda", dtype=torch.bfloat16)
        got = plan.run(a)
        ref = torch.nn.functional.linear(a, b.transpose(-2, -1))
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_plan_table_shared_across_same_shape_weights():
    """Weights with identical problem signature share one tuned table."""
    _skip_unless_supported()
    n, k = 128, 256
    before = len(_MM_BF16_PLAN_TABLES)
    plan_a = plan_mm_bf16(_make_weight(n, k), max_m=64)
    after_first = len(_MM_BF16_PLAN_TABLES)
    plan_b = plan_mm_bf16(_make_weight(n, k), max_m=64)
    assert len(_MM_BF16_PLAN_TABLES) == after_first >= before
    assert plan_a._buckets == plan_b._buckets
