"""BF16 cuDNN tuning with plans built by FLASHINFER_CUDNN_PLAN_BUILD_WORKERS."""

import warnings

import pytest
import torch

import flashinfer
import flashinfer.gemm._cudnn_plan_pool as pool_mod
import flashinfer.gemm.gemm_base as gb
from flashinfer.autotuner import autotune
from flashinfer.autotuner.autotuner import _profile_measurement_scope

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not gb._is_cudnn_override_shape_available(),
    reason="needs CUDA and cuDNN override-shape support",
)

N, K = 512, 256


def _set_workers(monkeypatch, workers):
    current = pool_mod._pool
    if current is not None:
        current.shutdown()
    monkeypatch.setenv(pool_mod.ENV_VAR, str(workers))
    monkeypatch.setattr(pool_mod, "_pool", None)
    monkeypatch.setattr(pool_mod, "_pool_resolved", False)
    gb.clear_cudnn_graph_cache()


@pytest.fixture
def pool_off(monkeypatch):
    _set_workers(monkeypatch, 0)
    yield
    gb.clear_cudnn_graph_cache()


@pytest.fixture
def pool_on(monkeypatch):
    def enable():
        _set_workers(monkeypatch, 2)
        pool = pool_mod.get_cudnn_plan_build_pool()
        assert pool is not None
        return pool

    yield enable
    if pool_mod._pool is not None:
        pool_mod._pool.shutdown()
    gb.clear_cudnn_graph_cache()


def _inputs(m):
    a = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    out = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    return [a, b, None, False, out, workspace]


def _runner():
    return gb._cudnn_gemm_bf16_runner(is_a_k_major=True, is_b_k_major=True)


@pytest.mark.parametrize("m", [1, 64, 128])
def test_pool_tactics_match_in_process(pool_off, pool_on, monkeypatch, m):
    inputs = _inputs(m)
    expected = _runner().get_valid_tactics(inputs, None)
    pool = pool_on()
    got = _runner().get_valid_tactics(inputs, None)
    assert pool.last_responses, "the pool was not used"
    assert not pool.broken
    assert got == expected


@pytest.mark.parametrize("m", [37, 128])
def test_pool_plans_bitwise_equal(pool_off, pool_on, m):
    torch.manual_seed(0)
    inputs = _inputs(m)
    runner = _runner()
    tactics = runner.get_valid_tactics(inputs, None)
    expected = {}
    for tactic in tactics:
        runner.forward(inputs, tactic=tactic)
        expected[tactic] = inputs[4].clone()

    pool = pool_on()
    runner = _runner()
    assert runner.get_valid_tactics(inputs, None) == tactics
    assert pool.last_responses, "the pool was not used"
    for tactic in tactics:
        inputs[4].zero_()
        runner.forward(inputs, tactic=tactic)
        assert torch.equal(inputs[4], expected[tactic]), tactic


def test_pool_plans_profile_without_mismatch(pool_on):
    pool = pool_on()
    inputs = _inputs(64)
    runner = _runner()
    tactics = runner.get_valid_tactics(inputs, None)
    assert pool.last_responses and tactics
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _profile_measurement_scope():
            for tactic in tactics:
                runner.forward(inputs, tactic=tactic)
    assert not [w for w in caught if "did not match" in str(w.message)]


def test_mm_bf16_autotune_with_pool_matches_reference(pool_on):
    pool = pool_on()
    a = torch.randn(48, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").T
    with autotune(True):
        out = flashinfer.mm_bf16(a, b, backend="cudnn")
    assert pool.last_responses, "the pool was not used"
    assert not pool.broken
    torch.testing.assert_close(out, a @ b, rtol=2e-2, atol=5e-2)
