"""CPU tests for the registry of pool-built single-plan cuDNN graphs."""

import types

import pytest
import torch

import flashinfer.gemm.gemm_base as gb


class FakeGraph:
    def __init__(self, handle=None):
        self.blob = None

    def deserialize(self, blob):
        if blob == b"bad":
            raise RuntimeError("cannot deserialize")
        self.blob = blob


@pytest.fixture(autouse=True)
def fake_cudnn(monkeypatch):
    # gemm_base binds ``cudnn`` only when the package imports; the CPU lane
    # may not have it.
    monkeypatch.setattr(
        gb, "cudnn", types.SimpleNamespace(pygraph=FakeGraph), raising=False
    )
    gb._cudnn_override_prebuilt.clear()
    gb._cudnn_single_plan_graphs.clear()
    yield
    gb._cudnn_override_prebuilt.clear()
    gb._cudnn_single_plan_graphs.clear()


def _built(tag, count=3):
    return [((7, ((1, i),)), f"{tag}{i}".encode()) for i in range(count)]


def _graphs(key):
    return [g for g in gb._cudnn_override_prebuilt[key].values() if g is not None]


def _store(key, built):
    return gb._store_cudnn_prebuilt(
        key, [t for t, _ in built], built, handle=None, build_here=()
    )


def test_storing_a_key_again_forgets_its_old_graphs():
    _store("k", _built("old"))
    old = _graphs("k")
    _store("k", _built("new"))
    assert not any(gb._is_cudnn_single_plan_graph(g) for g in old)
    assert all(gb._is_cudnn_single_plan_graph(g) for g in _graphs("k"))
    assert len(gb._cudnn_single_plan_graphs) == 3


def test_eviction_removes_whole_keys(monkeypatch):
    monkeypatch.setattr(gb, "_CUDNN_PREBUILT_MAX_KEYS", 2)
    for key in ("a", "b", "c"):
        _store(key, _built(key))
    assert gb._cudnn_override_prebuilt.get("a") is None
    assert list(gb._cudnn_override_prebuilt) == ["b", "c"]
    assert gb._cudnn_prebuilt_tactics("b") == [t for t, _ in _built("b")]
    assert len(gb._cudnn_single_plan_graphs) == 6


def test_registry_bound_counts_keys_not_graphs():
    assert gb._CUDNN_PREBUILT_MAX_KEYS == 2048


def test_lookup_requires_the_registered_object():
    _store("k", _built("x", 1))
    (graph,) = _graphs("k")
    assert gb._is_cudnn_single_plan_graph(graph)
    assert not gb._is_cudnn_single_plan_graph(FakeGraph())


class FakePool:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.disabled = result, error, None

    def build(self, spec, device_index, heur_modes):
        if self.error:
            raise self.error
        return self.result

    def disable(self, reason):
        self.disabled = reason


@pytest.mark.parametrize(
    "pool",
    [
        FakePool(error=KeyError("candidates")),
        FakePool(
            result=(
                [t for t, _ in _built("ok", 2)] + [(0, ())],
                _built("ok", 2) + [((0, ()), b"bad")],
                [],
            )
        ),
    ],
    ids=["build-raises", "deserialize-raises"],
)
def test_pool_failures_fall_back_and_disable_the_pool(pool):
    tactics = gb._pool_plan_tactics(
        pool, "k", spec=None, device_index=0, heur_modes=[0], handle=None
    )
    assert tactics is None
    assert pool.disabled
    assert "k" not in gb._cudnn_override_prebuilt
    assert not gb._cudnn_single_plan_graphs


def test_pool_result_is_stored_and_returned():
    pool = FakePool(result=([t for t, _ in _built("ok")], _built("ok"), []))
    tactics = gb._pool_plan_tactics(
        pool, "k", spec=None, device_index=0, heur_modes=[0], handle=None
    )
    assert tactics == [t for t, _ in _built("ok")]
    assert pool.disabled is None


def test_unbuilt_candidate_is_listed_and_fails_only_when_profiling(monkeypatch):
    built = _built("ok", 3)
    candidates = [t for t, _ in built] + [(9, ())]
    pool = FakePool(result=(candidates, built, []))
    tactics = gb._pool_plan_tactics(
        pool, "k", spec=None, device_index=0, heur_modes=[0], handle=None
    )
    assert tactics == candidates
    assert pool.disabled is None
    assert len(gb._cudnn_single_plan_graphs) == 3

    monkeypatch.setattr(gb, "_cudnn_override_key", lambda builder, kwargs: "k")
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return "in-process"

    monkeypatch.setattr(gb, "is_in_profile_measurement", lambda: True)
    with pytest.raises(RuntimeError, match="plan-build workers"):
        gb._get_cudnn_override_graph(builder, (9, ()))
    assert calls == []
    # Outside profiling (e.g. a tactic from a saved tuning cache) build it here.
    monkeypatch.setattr(gb, "is_in_profile_measurement", lambda: False)
    assert gb._get_cudnn_override_graph(builder, (9, ())) == "in-process"
    assert calls == [{"tactic": (9, ())}]


def test_plan_the_workers_could_not_export_is_built_here_even_when_profiling(
    monkeypatch,
):
    built = _built("ok", 2)
    candidates = [t for t, _ in built] + [(9, ())]
    pool = FakePool(result=(candidates, built, [(9, ())]))
    tactics = gb._pool_plan_tactics(
        pool, "k", spec=None, device_index=0, heur_modes=[0], handle=None
    )
    assert tactics == candidates
    assert len(gb._cudnn_single_plan_graphs) == 2

    monkeypatch.setattr(gb, "_cudnn_override_key", lambda builder, kwargs: "k")
    monkeypatch.setattr(gb, "is_in_profile_measurement", lambda: True)
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return "in-process"

    assert gb._get_cudnn_override_graph(builder, (9, ())) == "in-process"
    assert calls == [{"tactic": (9, ())}]
    gb._discard_cudnn_prebuilt("k")
    assert not gb._cudnn_single_plan_graphs


def test_worker_plan_bypassed_when_disallowed(monkeypatch):
    built = _built("ok", 2)
    _store("k", built)
    tactic = built[0][0]
    monkeypatch.setattr(gb, "_cudnn_override_key", lambda builder, kwargs: "k")
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return "in-process"

    assert gb._is_cudnn_single_plan_graph(gb._get_cudnn_override_graph(builder, tactic))
    assert calls == []
    assert (
        gb._get_cudnn_override_graph(builder, tactic, use_worker_plan=False)
        == "in-process"
    )
    assert calls == [{"tactic": tactic}]


M, N, K = 64, 32, 16


def _bf16(*shape):
    return torch.zeros(*shape, dtype=torch.bfloat16)


def _a():
    return _bf16(M, K)


def _b():
    return _bf16(N, K).T


def _out():
    return _bf16(M, N)


@pytest.mark.parametrize(
    "a, b, bias, out, a_k_major, b_k_major, ok",
    [
        (_a(), _b(), None, _out(), True, True, True),
        (_bf16(37, K), _b(), None, _bf16(37, N), True, True, False),
        (_a(), _b(), None, _bf16(M, N + 8)[:, :N], True, True, False),
        (_a(), _b(), _bf16(N), _out(), True, True, False),
        (_bf16(1, M, K), _b(), None, _out(), True, True, False),
        (_bf16(K, M).T, _b(), None, _out(), False, True, True),
        (_bf16(K, M).T, _b(), None, _out(), True, True, False),
        (_a(), _bf16(K, N), None, _out(), True, False, True),
        (_a(), _bf16(K, N), None, _out(), True, True, False),
    ],
    ids=[
        "built-geometry",
        "m-below-bucket",
        "padded-output",
        "bias",
        "batched",
        "a-mn-major",
        "a-layout-mismatch",
        "b-n-major",
        "b-layout-mismatch",
    ],
)
def test_worker_plan_geometry(a, b, bias, out, a_k_major, b_k_major, ok):
    assert (
        gb._bf16_worker_plan_geometry_ok(
            a,
            b,
            bias,
            out,
            m=M,
            n=N,
            k=K,
            is_a_k_major=a_k_major,
            is_b_k_major=b_k_major,
        )
        is ok
    )
