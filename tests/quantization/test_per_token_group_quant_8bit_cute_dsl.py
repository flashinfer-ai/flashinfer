# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the specialized CuTe-DSL sm90 backend of per_token_group_quant_8bit."""

import logging

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_available
from flashinfer.cutile import is_cuda_tile_available
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


def _skip_unless_stock_cutile_usable() -> None:
    if not is_cuda_tile_available():
        pytest.skip("needs cuda.tile")


def _reference(x: torch.Tensor, eps: float = 1e-10):
    # Evaluate the IEEE quotient in FP64, then round once to FP32.
    # A Python scalar division of an FP32 CUDA tensor can use reciprocal multiply.
    grouped = x.reshape(-1, 128).float()
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = (absmax.double() / 448.0).float()
    inv_scale = scale.double().reciprocal().float()
    q = (grouped * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
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
    # bf16-exact absmax where the reciprocal-multiply scale differs from the
    # IEEE quotient by one fp32 ulp (fl32(255 * fl32(1/448)) != fl32(255/448)).
    x.view(-1)[0] = 255.0
    assert _eligible(x)

    _reset_per_token_group_quant_stats()
    actual_q, actual_s = per_token_group_quant_8bit(x, 128, backend=backend)
    assert _ptgq_stats["backend_dispatches"]["specialized"] == 1
    expected_q, expected_s = _reference(x)

    assert actual_q.dtype == torch.float8_e4m3fn
    assert actual_s.shape == shape[:-1] + (shape[-1] // 128,)
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    ng = shape[-1] // 128
    deq = actual_q.float().view(-1, ng, 128) * actual_s.view(-1, ng, 1)
    deq_ref = expected_q.float().view(-1, ng, 128) * expected_s.view(-1, ng, 1)
    torch.testing.assert_close(deq, deq_ref, rtol=0.07, atol=1e-3)


@requires_fast_path
@pytest.mark.parametrize("groups", [1, 16384], ids=["16-lane", "8-lane"])
@pytest.mark.parametrize("case", ["zero_scale", "mixed_nan"])
def test_specialized_exceptional_products_match_cutile(groups, case):
    _skip_unless_stock_cutile_usable()
    x = torch.zeros((groups, 128), device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if case == "zero_scale":
        kwargs["eps"] = 0.0
    else:
        # One NaN among finite values; scale = 448 / 448 exactly.
        x.fill_(448.0)
        x[0, 7] = float("nan")

    stock_q, stock_s = per_token_group_quant_8bit(x, 128, backend="cutile", **kwargs)
    stock_deq = stock_q.float() * stock_s  # one group per row
    if case == "zero_scale":
        assert (stock_s == 0).all() and (stock_deq == 0).all()
        # Check the reported dequantization on the other stock backend;
        # cuTile, not CUDA, is the compatibility oracle for auto.
        cuda_q, cuda_s = per_token_group_quant_8bit(x, 128, backend="cuda", **kwargs)
        assert (cuda_s == 0).all() and (cuda_q.float() * cuda_s == 0).all()
    else:
        assert (stock_s == 1).all()
        assert stock_q.float()[0, 7].item() == -448.0

    for backend in ("auto", "specialized"):
        _reset_per_token_group_quant_stats()
        q, s = per_token_group_quant_8bit(x, 128, backend=backend, **kwargs)
        assert _ptgq_stats["backend_dispatches"] == {"specialized": 1, "stock": 0}
        assert torch.equal(s.view(torch.int32), stock_s.view(torch.int32))
        assert torch.equal(q.view(torch.uint8), stock_q.view(torch.uint8))
        # Numeric equality alone cannot distinguish the signs of zero.
        assert torch.equal(
            (q.float() * s).view(torch.int32), stock_deq.view(torch.int32)
        )


@requires_fast_path
@pytest.mark.parametrize("stock", ["cutile", "cuda"])
def test_explicit_stock_backends_never_swapped(stock):
    """backend='cutile' / 'cuda' never reach the specialized kernel, even
    on a fully on-contract call."""
    if stock == "cutile":
        _skip_unless_stock_cutile_usable()
    torch.manual_seed(11)
    x = torch.empty((16, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-2, 2)
    assert _eligible(x)
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
    # Under the kill switch, auto serves the call on the stock cutile route;
    # explicit "specialized" raises (never falls back).
    _skip_unless_stock_cutile_usable()
    torch.manual_seed(7)
    x = torch.empty((64, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-3, 3)
    # A bf16-exact absmax of 255 distinguishes IEEE division from
    # reciprocal multiplication. Both routes must produce the exact scale.
    x.view(-1)[0] = 255.0

    fast_q, fast_s = per_token_group_quant_8bit(x, 128, backend="specialized")

    monkeypatch.setenv(env, "1")
    assert not _eligible(x)
    _reset_per_token_group_quant_stats()
    with pytest.raises(NotImplementedError, match="kill_switch"):
        per_token_group_quant_8bit(x, 128, backend="specialized")
    _reset_per_token_group_quant_stats()
    stock_q, stock_s = per_token_group_quant_8bit(x, 128)  # auto
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {"kill_switch": 1}

    torch.testing.assert_close(fast_s, stock_s, rtol=0, atol=0)
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
    # Off-contract forms are served by the composable stock cutile route.
    _skip_unless_stock_cutile_usable()
    x = torch.empty((8, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    group_size = kwargs.pop("group_size", 128)
    dst_dtype = kwargs.pop("dst_dtype", torch.float8_e4m3fn)
    cms = kwargs.get("column_major_scales", False)
    ok, reason = use_per_token_group_quant_specialized(
        x, group_size, dst_dtype, cms, False, False
    )
    assert not ok and reason in ("option_form", "group_or_dtype")
    # Explicit "specialized" off-contract raises with the reason (an
    # explicitly selected backend never falls back); auto serves the same
    # call on the stock cutile path with the reason recorded.
    _reset_per_token_group_quant_stats()
    with pytest.raises(NotImplementedError, match=reason):
        per_token_group_quant_8bit(
            x,
            group_size,
            dst_dtype=dst_dtype,
            column_major_scales=cms,
            backend="specialized",
        )
    assert _ptgq_stats["fallback_reasons"] == {reason: 1}
    _reset_per_token_group_quant_stats()
    q, s = per_token_group_quant_8bit(
        x,
        group_size,
        dst_dtype=dst_dtype,
        column_major_scales=cms,
        backend="auto",
    )
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {reason: 1}
    assert q.shape == x.shape


@requires_fast_path
def test_misaligned_input_stays_on_stock_path():
    """A contiguous offset view with a non-16-byte-aligned data pointer is
    off-contract (the kernel's 128-bit vectorized loads require 16-byte
    alignment) and is declined like any other off-contract form."""
    torch.manual_seed(5)
    buf = torch.empty(16 * 2560 + 1, device="cuda", dtype=torch.bfloat16)
    buf.uniform_(-2, 2)
    x = buf[1:].view(16, 2560)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    ok, reason = use_per_token_group_quant_specialized(
        x, 128, torch.float8_e4m3fn, False, False, False
    )
    assert not ok and reason == "alignment"

    _skip_unless_stock_cutile_usable()
    # explicit "specialized" raises on the misaligned form
    _reset_per_token_group_quant_stats()
    with pytest.raises(NotImplementedError, match="alignment"):
        per_token_group_quant_8bit(x, 128, backend="specialized")
    assert _ptgq_stats["fallback_reasons"] == {"alignment": 1}
    # auto serves it on the stock path with the reason recorded
    _reset_per_token_group_quant_stats()
    q, s = per_token_group_quant_8bit(x, 128, backend="auto")
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {"alignment": 1}
    expected_q, expected_s = _reference(x)
    torch.testing.assert_close(s, expected_s, rtol=2e-7, atol=0)
    exact = (q.view(torch.uint8) == expected_q.view(torch.uint8)).float()
    assert exact.mean().item() >= 0.999


@requires_fast_path
def test_eager_failure_memo_is_auto_only(monkeypatch, caplog):
    from flashinfer.quantization import fp8_quantization as dispatch
    from flashinfer.quantization.kernels.cute_dsl import (
        per_token_group_quant_8bit_cute_dsl as kernel,
    )

    _skip_unless_stock_cutile_usable()
    x = torch.randn((8, 256), device="cuda", dtype=torch.bfloat16)
    stock_q, stock_s = per_token_group_quant_8bit(x, 128, backend="cutile")
    attempts = []

    def fail(*args):
        attempts.append(1)
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(kernel, "per_token_group_quant_fp8_g128_cute_dsl", fail)
    monkeypatch.setattr(dispatch, "_ptgq_specialized_failed", False)
    _reset_per_token_group_quant_stats()
    with pytest.raises(NotImplementedError, match="jit_or_launch_failure"):
        per_token_group_quant_8bit(x, 128, backend="specialized")
    assert len(attempts) == 1
    assert not dispatch._ptgq_specialized_failed
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 0}

    with caplog.at_level(logging.WARNING, logger=dispatch.__name__):
        for _ in range(2):
            q, s = per_token_group_quant_8bit(x, 128)
            assert torch.equal(q.view(torch.uint8), stock_q.view(torch.uint8))
            assert torch.equal(s, stock_s)
        with pytest.raises(NotImplementedError, match="jit_or_launch_failure"):
            per_token_group_quant_8bit(x, 128, backend="specialized")
    assert len(attempts) == 2  # first explicit call, then first auto only
    assert dispatch._ptgq_specialized_failed
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 2}
    assert _ptgq_stats["fallback_reasons"] == {"jit_or_launch_failure": 4}
    assert (
        sum(
            r.name == dispatch.__name__ and "kernel failed" in r.getMessage()
            for r in caplog.records
        )
        == 1
    )


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

    # Warm up (JIT compile) before capture, as documented, and assert the
    # warmup itself ran on the specialized kernel.
    _reset_per_token_group_quant_stats()
    per_token_group_quant_8bit(x, 128, backend="specialized")
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 1, "stock": 0}
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_q, graph_s = per_token_group_quant_8bit(x, 128, backend="specialized")
    # The captured call must also have taken the specialized route.
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 2, "stock": 0}

    # Mutate the input in place after capture; the replay must read the new
    # values, not bytes baked in at capture time.
    x.uniform_(-5.0, 5.0)
    g.replay()
    torch.cuda.synchronize()

    eager_q, eager_s = per_token_group_quant_8bit(x, 128, backend="specialized")
    torch.testing.assert_close(graph_s, eager_s, rtol=0, atol=0)
    torch.testing.assert_close(
        graph_q.view(torch.uint8), eager_q.view(torch.uint8), rtol=0, atol=0
    )


@requires_fast_path
@pytest.mark.parametrize("different_device", [False, True])
def test_cold_graph_capture_auto_falls_back_with_dedicated_reason(
    monkeypatch, different_device
):
    """A not-yet-compiled specialized variant cannot JIT under CUDA graph
    capture; backend='auto' records the stock cutile kernel into the graph
    under the dedicated reason 'capturing_before_jit' (not a generic
    jit_or_launch_failure), and the failure memo is not set, so later eager
    calls still take the specialized path."""
    _skip_unless_stock_cutile_usable()
    from flashinfer.quantization.kernels.cute_dsl import (
        per_token_group_quant_8bit_cute_dsl as _mod,
    )

    caller = torch.cuda.current_device()
    if different_device:
        if torch.cuda.device_count() < 2:
            pytest.skip("needs a second CUDA device for the calling context")
        caller = (caller + 1) % torch.cuda.device_count()
        with torch.cuda.device(caller):
            torch.cuda.current_stream()  # initialize the caller context before capture
    torch.manual_seed(9)
    x = torch.empty((8, 2560), device="cuda", dtype=torch.bfloat16).uniform_(-2, 2)
    # Warm the stock path so the capture below replays a compiled kernel.
    per_token_group_quant_8bit(x, 128, backend="cutile")
    torch.cuda.synchronize()

    # Force a cold specialized variant for this capture.
    monkeypatch.setattr(_mod, "_CACHE", {})
    _reset_per_token_group_quant_stats()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.cuda.device(caller):
        if different_device:
            assert caller != x.device.index
            assert not torch.cuda.is_current_stream_capturing()
        with pytest.raises(NotImplementedError, match="capturing_before_jit"):
            per_token_group_quant_8bit(x, 128, backend="specialized")
        graph_q, graph_s = per_token_group_quant_8bit(x, 128, backend="auto")
    assert _ptgq_stats["backend_dispatches"] == {"specialized": 0, "stock": 1}
    assert _ptgq_stats["fallback_reasons"] == {"capturing_before_jit": 2}

    x.uniform_(-3, 3)
    g.replay()
    torch.cuda.synchronize()
    eager_q, eager_s = per_token_group_quant_8bit(x, 128, backend="cutile")
    torch.testing.assert_close(graph_s, eager_s, rtol=0, atol=0)
    torch.testing.assert_close(
        graph_q.view(torch.uint8), eager_q.view(torch.uint8), rtol=0, atol=0
    )

    # The capture-time refusal must not trip the process-level failure memo.
    # Assert the memo BEFORE any reset (a reset would mask a wrongly-set memo),
    # then make the eager auto call WITHOUT clearing it and check the counter
    # delta, so a capture handler that sets the memo fails this test.
    from flashinfer.quantization import fp8_quantization as _ptgq_mod

    assert _ptgq_mod._ptgq_specialized_failed is False
    before = dict(_ptgq_stats["backend_dispatches"])
    per_token_group_quant_8bit(x, 128, backend="auto")
    after = _ptgq_stats["backend_dispatches"]
    assert after["specialized"] == before.get("specialized", 0) + 1
    assert after["stock"] == before.get("stock", 0)
