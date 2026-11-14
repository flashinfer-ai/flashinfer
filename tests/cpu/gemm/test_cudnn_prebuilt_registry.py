"""CPU tests for the registry of pool-built single-plan cuDNN graphs."""

import types

import pytest

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
    return list(gb._cudnn_override_prebuilt[key].values())


def test_storing_a_key_again_forgets_its_old_graphs():
    gb._store_cudnn_prebuilt("k", _built("old"), handle=None)
    old = _graphs("k")
    gb._store_cudnn_prebuilt("k", _built("new"), handle=None)
    assert not any(gb._is_cudnn_single_plan_graph(g) for g in old)
    assert all(gb._is_cudnn_single_plan_graph(g) for g in _graphs("k"))
    assert len(gb._cudnn_single_plan_graphs) == 3


def test_eviction_removes_whole_keys(monkeypatch):
    monkeypatch.setattr(gb, "_CUDNN_PREBUILT_MAX_KEYS", 2)
    for key in ("a", "b", "c"):
        gb._store_cudnn_prebuilt(key, _built(key), handle=None)
    assert gb._cudnn_override_prebuilt.get("a") is None
    assert list(gb._cudnn_override_prebuilt) == ["b", "c"]
    assert gb._cudnn_prebuilt_tactics("b") == [t for t, _ in _built("b")]
    assert len(gb._cudnn_single_plan_graphs) == 6


def test_registry_bound_counts_keys_not_graphs():
    assert gb._CUDNN_PREBUILT_MAX_KEYS == 2048


def test_lookup_requires_the_registered_object():
    gb._store_cudnn_prebuilt("k", _built("x", 1), handle=None)
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
        FakePool(result=_built("ok", 2) + [((0, ()), b"bad")]),
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
    pool = FakePool(result=_built("ok"))
    tactics = gb._pool_plan_tactics(
        pool, "k", spec=None, device_index=0, heur_modes=[0], handle=None
    )
    assert tactics == [t for t, _ in _built("ok")]
    assert pool.disabled is None
