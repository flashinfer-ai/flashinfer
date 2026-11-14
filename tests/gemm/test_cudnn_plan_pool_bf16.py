"""BF16 cuDNN tuning with plans built by FLASHINFER_CUDNN_PLAN_BUILD_WORKERS."""

import json
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


def _spec_and_handle(m):
    import cudnn

    from flashinfer.gemm import _cudnn_plan_build as pb

    runner = _runner()
    inputs = _inputs(m)
    a, b, bias, _, out, _ = inputs
    kwargs = runner._override_graph_kwargs(a, b, bias, out)
    spec = gb._bf16_override_graph_spec(
        kwargs["batch"],
        kwargs["n"],
        kwargs["k"],
        kwargs["o_type"],
        kwargs["bias_is_not_none"],
        kwargs["cache_m"],
        kwargs["is_a_k_major"],
        kwargs["is_b_k_major"],
    )
    handle = gb._get_cudnn_handle(a.device, torch.cuda.current_stream(a.device))
    heur = [int(cudnn.heur_mode.A), int(cudnn.heur_mode.FALLBACK)]
    return pb, spec, handle, heur


def _engine_config(blob: bytes) -> dict:
    # Serialized plans embed host addresses and a graph id, so equal plans
    # differ in a few bytes; the embedded engine JSON (engine id plus every
    # knob choice) identifies the plan.
    text = blob.decode("latin-1")
    start = text.index('"engine":{"engineId"') + len('"engine":')
    config, _ = json.JSONDecoder().raw_decode(text, start)
    kernels = (config.get("payload") or {}).get("kernels") or []
    return {
        "engineId": config["engineId"],
        "knobChoices": config.get("knobChoices"),
        "kernels": [kernel.get("kernelName") for kernel in kernels],
    }


@pytest.mark.parametrize("shards", [1, 2])
def test_shard_exports_each_tactics_own_plan(pool_off, shards):
    # Bitwise outputs can coincide across plans; compare the exported plans.
    pb, spec, handle, heur = _spec_and_handle(64)
    exported = {}
    for shard in range(shards):
        candidates, plans = pb.build_shard(spec, handle, heur, (shard, shards))
        exported.update({tactic: blob for _, tactic, blob in plans})
    assert sorted(exported) == sorted(candidates)
    configs = set()
    for tactic in candidates:
        graph = pb.build_bf16_override_graph(spec, handle)
        pb.build_single_plan(graph, tactic)
        expected = _engine_config(bytes(graph.serialize(as_bytes=True)))
        assert _engine_config(exported[tactic]) == expected, tactic
        configs.add(json.dumps(expected, sort_keys=True))
    assert len(configs) > 1, "every candidate has the same engine config"


def test_pool_returning_none_builds_in_process(pool_off, pool_on, monkeypatch):
    inputs = _inputs(64)
    expected = _runner().get_valid_tactics(inputs, None)
    pool = pool_on()
    monkeypatch.setattr(pool, "build", lambda *a, **k: None)
    assert _runner().get_valid_tactics(inputs, None) == expected
    assert not gb._cudnn_override_prebuilt


def test_second_call_reuses_pool_plans(pool_on, monkeypatch):
    pool = pool_on()
    calls = []
    real_build = pool.build
    monkeypatch.setattr(
        pool, "build", lambda *a, **k: calls.append(1) or real_build(*a, **k)
    )
    inputs = _inputs(64)
    first = _runner().get_valid_tactics(inputs, None)
    second = _runner().get_valid_tactics(inputs, None)
    assert first == second and len(calls) == 1
