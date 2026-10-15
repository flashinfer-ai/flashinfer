import pytest

from flashinfer import autotune
from flashinfer.autotuner import AutoTuner, TuningResult

from .utils import reset_autotuner


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    tuner = reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    yield tuner
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner.clear_cache()


class RecordingProtocol:
    def __init__(self):
        self.contexts = []

    def run(self, context):
        self.contexts.append(context)
        if not context.force_tune and context.lookup("shape") is not None:
            return TuningResult(context.lookup("shape"), cache_hit=True)
        if context.tuning_enabled:
            context.publish("shape", {"tile": [128, 256], "external_factor": 3})
            return TuningResult("tuned", candidates=4, successful_candidates=3)
        return TuningResult("deferred")


def test_protocol_uses_shared_controls_and_statistics():
    tuner = reset_autotuner()
    protocol = RecordingProtocol()
    assert tuner.choose_with_protocol("test.protocol", protocol) == "deferred"
    with autotune(True), autotune(False):
        assert tuner.choose_with_protocol("test.protocol", protocol) == "tuned"
    assert protocol.contexts[-1].tuner is AutoTuner.get()
    assert tuner.stats.tuned_op_total_configs["test.protocol"] == 4
    assert tuner.stats.tuned_op_successful_configs["test.protocol"] == 3
    assert tuner.choose_with_protocol("test.protocol", protocol)["tile"] == [128, 256]
    records = protocol.contexts[-1].records()
    records["shape"]["tile"][0] = 0
    assert protocol.contexts[-1].lookup("shape")["tile"] == [128, 256]
    assert not tuner.is_tuning_mode


def test_explicit_search_overrides_cache_and_disabled_mode():
    tuner = reset_autotuner()
    protocol = RecordingProtocol()
    for _ in range(2):
        assert (
            tuner.choose_with_protocol("test.protocol", protocol, force_tune=True)
            == "tuned"
        )
    assert tuner.stats.tuned_op_total_configs["test.protocol"] == 8
    assert not tuner.is_tuning_mode


def test_protocol_scope_restores_after_nested_context_exception():
    tuner = reset_autotuner()
    protocol = RecordingProtocol()
    with autotune(False):
        tuner.choose_with_protocol("test.protocol", protocol)
        outer = protocol.contexts[-1].scope
        with pytest.raises(RuntimeError, match="test"), autotune(False):
            tuner.choose_with_protocol("test.protocol", protocol)
            assert protocol.contexts[-1].scope is not outer
            raise RuntimeError("test")
        tuner.choose_with_protocol("test.protocol", protocol)
        assert protocol.contexts[-1].scope is outer
    tuner.choose_with_protocol("test.protocol", protocol)
    assert protocol.contexts[-1].scope is None


def test_skip_ops_delegates_but_quarantines_existing_results():
    from flashinfer import autotune_v2

    tuner = reset_autotuner()
    protocol = RecordingProtocol()
    with autotune_v2(persistent_cache=False):
        assert tuner.choose_with_protocol("test.protocol", protocol) == "tuned"
        identity = protocol.contexts[-1].cache_identity
        misses = tuner.stats.cache_misses
    with autotune_v2(persistent_cache=False, skip_ops={"test.protocol"}):
        assert tuner.choose_with_protocol("test.protocol", protocol) == "deferred"
        assert protocol.contexts[-1].skipped
        assert protocol.contexts[-1].cache_identity != identity
        assert tuner.stats.cache_misses == misses
    with autotune_v2(mode="replay", persistent_cache=False):
        assert tuner.choose_with_protocol("test.protocol", protocol)["tile"] == [
            128,
            256,
        ]
    assert len(protocol.contexts) == 3


def test_protocol_memory_is_separate_from_v1_json(tmp_path):
    tuner = reset_autotuner()
    path = str(tmp_path / "cache.json")
    with autotune(True, cache=path):
        tuner.choose_with_protocol("test.protocol", RecordingProtocol())
    assert not tuner.get_namespaced_records("test.protocol")
    assert tuner.choose_with_protocol("test.other", RecordingProtocol()) == "deferred"
    tuner.clear_cache()
    with autotune(False, cache=path):
        assert (
            tuner.choose_with_protocol("test.protocol", RecordingProtocol())
            == "deferred"
        )
    tuner.clear_cache()


def test_protocol_managed_persistence_replay_and_root_isolation(managed, tmp_path):
    from flashinfer import autotune_v2

    with autotune_v2():
        assert (
            managed.choose_with_protocol("test.protocol", RecordingProtocol())
            == "tuned"
        )
        assert len(list(tmp_path.glob("v2/*/protocols/*/*.json"))) == 1
    managed._managed_cache = None
    managed._managed_stores.clear()
    managed.clear_cache()
    with autotune_v2(mode="replay"):
        assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
            "tile"
        ] == [128, 256]
        assert (
            managed.choose_with_protocol("test.other", RecordingProtocol())
            == "deferred"
        )
    assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
        "tile"
    ] == [128, 256]
    with autotune_v2(mode="replay", cache_root=tmp_path / "other"):
        assert (
            managed.choose_with_protocol("test.protocol", RecordingProtocol())
            == "deferred"
        )


def test_protocol_memory_only_and_reload(managed, tmp_path):
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import autotune_v2_reload
    import json

    with autotune_v2():
        managed.choose_with_protocol("test.protocol", RecordingProtocol())
    path = next(tmp_path.glob("v2/*/protocols/*/*.json"))
    with autotune_v2(persistent_cache=False):
        protocol = RecordingProtocol()
        assert managed.choose_with_protocol("test.protocol", protocol) == "tuned"
        assert not protocol.contexts[-1].disk_cache_enabled
    record = json.loads(path.read_text())
    record["value"]["tile"] = [64, 64]
    path.write_text(json.dumps(record))
    assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
        "tile"
    ] == [128, 256]
    identity = managed.protocol_context("test.protocol").cache_identity
    autotune_v2_reload()
    assert managed.protocol_context("test.protocol").cache_identity != identity
    assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
        "tile"
    ] == [64, 64]
    with autotune_v2(mode="replay", cache_root=tmp_path / "other"):
        pass
    record["value"]["tile"] = [32, 32]
    path.write_text(json.dumps(record))
    autotune_v2_reload()
    with autotune_v2(mode="replay"):
        assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
            "tile"
        ] == [32, 32]
    path.write_text("broken")
    autotune_v2_reload()
    assert (
        managed.choose_with_protocol("test.protocol", RecordingProtocol()) == "deferred"
    )


def test_protocol_policy_partition_and_inherited_controls(managed):
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import MeasurementPolicy

    graph_policy = MeasurementPolicy(execution_mode="cuda_graph")
    with (
        autotune_v2(persistent_cache=False, measurement_policy=graph_policy),
        autotune(cuda_graph_profile_replays=3),
    ):
        context = managed.protocol_context("test.protocol")
        assert context.cuda_graph_profile_replays == 3
        assert context.measurement_policy is graph_policy
        managed.choose_with_protocol("test.protocol", RecordingProtocol())
    with autotune_v2(persistent_cache=False, mode="replay"):
        assert (
            managed.choose_with_protocol("test.protocol", RecordingProtocol())
            == "deferred"
        )
    with autotune_v2(
        persistent_cache=False, mode="replay", measurement_policy=graph_policy
    ):
        assert managed.choose_with_protocol("test.protocol", RecordingProtocol())[
            "tile"
        ] == [128, 256]


def test_protocol_policy_is_preserved_after_persistent_context(managed):
    from flashinfer import autotune_v2
    from flashinfer.autotune_cache import MeasurementPolicy

    policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True)
    with autotune_v2(measurement_policy=policy):
        identity = managed.protocol_context("test.protocol").cache_identity
    context = managed.protocol_context("test.protocol")
    assert context.cache_identity == identity
    assert context.measurement_policy == policy
    with autotune_v2(persistent_cache=False):
        assert managed.protocol_context("test.protocol").measurement_policy is None


@pytest.mark.parametrize("local_failure", [False, True])
def test_collective_cupti_initialization_precedes_launch(monkeypatch, local_failure):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    module = importlib.import_module("flashinfer.autotuner.autotuner")
    cupti = SimpleNamespace(
        ActivityKind=SimpleNamespace(RUNTIME=1, DRIVER=2, CONCURRENT_KERNEL=3),
        activity_enable=Mock(
            side_effect=RuntimeError("unavailable") if local_failure else None
        ),
        activity_register_callbacks=Mock(),
        activity_disable=Mock(),
        activity_flush_all=Mock(),
    )
    monkeypatch.setattr(module, "_load_cupti", lambda: cupti)
    run = Mock()

    def check_ready(error):
        assert (error is not None) == local_failure
        raise module._CuptiInfraError("an EP rank could not initialize tracing")

    with pytest.raises(module._CuptiInfraError, match="EP rank"):
        module._cupti_measure_spans(run, 1, check_ready=check_ready)
    run.assert_not_called()
    if not local_failure:
        assert cupti.activity_disable.call_count == 3
