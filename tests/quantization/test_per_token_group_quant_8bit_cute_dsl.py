# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the specialized CuTe-DSL sm90 backend of per_token_group_quant_8bit."""

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.quantization import per_token_group_quant_8bit
from flashinfer.quantization.fp8_quantization import (
    PER_TOKEN_GROUP_QUANT_BACKENDS,
    PER_TOKEN_GROUP_QUANT_CUTE_DSL_DISABLE_ENV,
    SPECIALIZED_KERNEL_DISABLE_ENV,
    _ptgq_specialized_dispatch_table,
    _ptgq_stats,
    _reset_per_token_group_quant_stats,
    use_per_token_group_quant_specialized,
)

requires_fast_path = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9
    or not is_cute_dsl_available(),
    reason="needs an sm90 device and the nvidia-cutlass-dsl runtime",
)


def _reference(x: torch.Tensor, eps: float = 1e-10):
    # Scale convention of the kernels: multiply by the reciprocal of 448
    # (what PyTorch itself does for a CUDA tensor divided by a Python
    # scalar), not the IEEE quotient.
    grouped = x.reshape(-1, 128).float()
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / 448.0
    q = (grouped / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (
        q.reshape(x.shape),
        scale.reshape(x.shape[:-1] + (x.shape[-1] // 128,)),
    )


def _eligible(x, group_size=128, dst_dtype=torch.float8_e4m3fn, cms=False):
    ok, _reason = use_per_token_group_quant_specialized(
        x, group_size, dst_dtype, cms, False, False
    )
    return ok


@requires_fast_path
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),  # single group
        (3, 2560),  # small-T decode shape, 16-lane variant
        (333, 6144),  # non-power-of-two T
        (1024, 6144),  # 8-lane variant (num_groups >= 16384)
    ],
)
@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_specialized_matches_reference(shape, backend):
    torch.manual_seed(42)
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16).uniform_(-2.0, 2.0)
    assert _eligible(x)

    _reset_per_token_group_quant_stats()
    actual_q, actual_s = per_token_group_quant_8bit(x, 128, backend=backend)
    assert _ptgq_stats["backend_dispatches"]["specialized"] == 1
    expected_q, expected_s = _reference(x)

    assert actual_q.dtype == torch.float8_e4m3fn
    assert actual_s.shape == shape[:-1] + (shape[-1] // 128,)
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    # The kernel multiplies by rcp.approx of the scale instead of dividing;
    # allow one-ulp fp8 differences on a vanishing fraction of elements.
    exact = (actual_q.view(torch.uint8) == expected_q.view(torch.uint8)).float()
    assert exact.mean().item() >= 0.999
    ng = shape[-1] // 128
    deq = actual_q.float().view(-1, ng, 128) * actual_s.view(-1, ng, 1)
    deq_ref = expected_q.float().view(-1, ng, 128) * expected_s.view(-1, ng, 1)
    torch.testing.assert_close(deq, deq_ref, rtol=0.07, atol=1e-3)


@requires_fast_path
def test_explicit_stock_backends_never_swapped():
    """backend='cutile' / 'cuda' never reach the specialized kernel, even
    on a fully on-contract call."""
    torch.manual_seed(11)
    x = torch.empty((16, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-2, 2)
    assert _eligible(x)
    for stock in ("cutile", "cuda"):
        _reset_per_token_group_quant_stats()
        q, s = per_token_group_quant_8bit(x, 128, backend=stock)
        assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
        assert _ptgq_stats["fallback_reasons"] == {}
        assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32


@requires_fast_path
def test_unknown_backend_raises():
    x = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    with pytest.raises(ValueError, match="Unknown backend"):
        per_token_group_quant_8bit(x, 128, backend="fastest")


@requires_fast_path
@pytest.mark.parametrize(
    "env", [SPECIALIZED_KERNEL_DISABLE_ENV, PER_TOKEN_GROUP_QUANT_CUTE_DSL_DISABLE_ENV]
)
def test_kill_switches_match_stock(monkeypatch, env):
    torch.manual_seed(7)
    x = torch.empty((64, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-3, 3)

    fast_q, fast_s = per_token_group_quant_8bit(x, 128, backend="specialized")

    monkeypatch.setenv(env, "1")
    assert not _eligible(x)
    _reset_per_token_group_quant_stats()
    stock_q, stock_s = per_token_group_quant_8bit(x, 128)  # auto
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {"kill_switch": 1}

    # Scales: bit-identical on the cutile build the measurement used; across
    # cuda-tile versions the stock scale can differ by one fp32 ulp (the
    # kernels agree on absmax/448 up to reciprocal-vs-division rounding).
    torch.testing.assert_close(fast_s, stock_s, rtol=2e-7, atol=0)
    exact = (fast_q.view(torch.uint8) == stock_q.view(torch.uint8)).float()
    assert exact.mean().item() >= 0.999


@requires_fast_path
@pytest.mark.parametrize(
    "kwargs",
    [
        {"column_major_scales": True},
        {"dst_dtype": torch.int8},
        {"group_size": 64},
    ],
)
def test_option_forms_stay_on_stock_path(kwargs):
    x = torch.empty((8, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    group_size = kwargs.pop("group_size", 128)
    dst_dtype = kwargs.pop("dst_dtype", torch.float8_e4m3fn)
    cms = kwargs.get("column_major_scales", False)
    ok, reason = use_per_token_group_quant_specialized(
        x, group_size, dst_dtype, cms, False, False
    )
    assert not ok and reason in ("option_form", "group_or_dtype")
    # Explicit "specialized" off-contract falls back to the composable
    # cutile path (recorded), it never errors and never reroutes elsewhere.
    _reset_per_token_group_quant_stats()
    q, s = per_token_group_quant_8bit(
        x,
        group_size,
        dst_dtype=dst_dtype,
        column_major_scales=cms,
        backend="specialized",
    )
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {reason: 1}
    assert q.shape == x.shape


def test_dispatch_table_present_and_well_formed():
    table = _ptgq_specialized_dispatch_table()
    assert table is not None, "packaged allowlist must load"
    assert table["backends"] == list(PER_TOKEN_GROUP_QUANT_BACKENDS)
    assert table["kill_switch"] == SPECIALIZED_KERNEL_DISABLE_ENV
    op = table["ops"]["per_token_group_quant_8bit"]
    assert op["group_size"] == 128
    assert op["compute_capability"] == [9, 0]


@requires_fast_path
def test_specialized_cuda_graph_matches_eager():
    torch.manual_seed(3)
    x = torch.empty((256, 6144), device="cuda", dtype=torch.bfloat16).uniform_(-2, 2)

    # Warm up (JIT compile) before capture, as documented.
    eager_q, eager_s = per_token_group_quant_8bit(x, 128, backend="specialized")
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_q, graph_s = per_token_group_quant_8bit(x, 128, backend="specialized")
    g.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_s, eager_s, rtol=0, atol=0)
    torch.testing.assert_close(
        graph_q.view(torch.uint8), eager_q.view(torch.uint8), rtol=0, atol=0
    )
