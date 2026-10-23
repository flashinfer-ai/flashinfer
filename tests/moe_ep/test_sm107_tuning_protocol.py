"""Rubin cache compatibility and deferred automatic tuning without a GPU."""

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer import autotune, autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    autotune as native,
    block_scaled,
    knob_cache,
    tuning_protocol,
)
from tests.moe_ep.test_sm107_block_scaled_config import _BACKENDS
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_state import (
    AutoTuningState,
)


@pytest.mark.parametrize("config_cls,backend_cls,name", _BACKENDS)
def test_auto_initial_config_does_not_depend_on_rank_local_cache(
    config_cls, backend_cls, name, monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    backend = create_mega_kernel(config_cls(256, 3, knobs="auto"))
    backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=2))
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=384)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    initial = backend._resolved_config(fleet)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "managed"))
    knobs = {"fc2_use_bulk": False, "fc2_tma_stages": None}
    knob_cache.record_knobs(knobs, **backend._tuning_key(fleet))
    assert knob_cache.lookup_knobs(**backend._tuning_key(fleet)) == knobs
    assert backend._resolved_config(fleet) == initial
    assert backend._workspace_pool_key(fleet) is None


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "managed"))
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(
        block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", lambda cfg: None
    )
    tuner = AutoTuner.get()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner.clear_cache()
    tuner.reset_statistics()
    cfg = block_scaled.Sm107BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=128,
        num_topk=2,
        hidden=128,
        intermediate=64,
        rank=0,
        world_size=1,
    )
    workspace = SimpleNamespace(
        config=cfg,
        staged_tokens=lambda: 4,
        install_tuned_knobs=Mock(),
        _tuning_state=AutoTuningState(),
    )
    sweep = Mock(return_value=({"fc2_use_bulk": False}, 1.0, 1))
    monkeypatch.setattr(native, "_sweep_sm107_block_scaled_mega_moe", sweep)

    def run(**kwargs):
        return native.autotune_sm107_block_scaled_mega_moe(
            None,
            None,
            None,
            workspace,
            candidates=[{"fc2_use_bulk": False}],
            install_winner=True,
            force_tune=False,
            **kwargs,
        )

    yield SimpleNamespace(
        workspace=workspace,
        sweep=sweep,
        run=run,
        tuner=tuner,
        path=tmp_path / "knobs.json",
    )
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()


@pytest.mark.parametrize("routing_before_fc2", [False, True])
def test_deferred_search_persistence_and_shared_hit_file_output(
    state, tmp_path, routing_before_fc2
):
    state.workspace.config = dataclasses.replace(
        state.workspace.config, apply_routing_weights_before_fc2=routing_before_fc2
    )
    other = {"backend_revision": "other-backend", "knobs": {}, "max_tokens": 64}
    contents = json.dumps({"version": KNOB_CACHE_VERSION, "entries": [other]})
    state.path.write_text(contents)
    assert state.run() is None
    state.sweep.assert_not_called()
    with autotune(True, skip_ops={tuning_protocol.NAMESPACE}):
        assert state.run() is None
    state.sweep.assert_not_called()
    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
        assert list((tmp_path / "managed").glob("v2/*/entries/*.json"))
    state.sweep.assert_called_once()
    state.workspace.install_tuned_knobs.assert_called_once()
    cache = json.loads(state.path.read_text())
    assert cache["version"] == KNOB_CACHE_VERSION
    assert other in cache["entries"]
    record = next(e for e in cache["entries"] if e != other)
    assert record["apply_routing_weights_before_fc2"] is routing_before_fc2
    assert "apply_topk_at_fc1" not in record
    state.path.write_text(contents)
    state.tuner.clear_cache()
    state.tuner._managed_cache = None
    state.tuner._managed_stores.clear()
    with autotune_v2(mode="replay"):
        assert state.run() == {"fc2_use_bulk": False}
    assert json.loads(state.path.read_text()) == cache
    state.sweep.assert_called_once()
    assert state.workspace.install_tuned_knobs.call_count == 2


@pytest.mark.parametrize(
    "controls", [{"tuning_buckets": (64, 256)}, {"round_up": True}]
)
@pytest.mark.parametrize("mode", ["tune", "replay"])
def test_bucket_overrides_are_rejected_before_selection(state, controls, mode):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.automatic import tuning_context

    with autotune_v2(mode=mode, **controls):
        for select in (state.run, tuning_context):
            with pytest.raises(
                NotImplementedError, match="tuning_buckets and round_up"
            ):
                select()
    state.sweep.assert_not_called()
    state.workspace.install_tuned_knobs.assert_not_called()
    assert not state.path.exists()
    with autotune_v2(skip_ops={tuning_protocol.NAMESPACE}, **controls):
        assert state.run() is None
        assert tuning_context().skipped


def test_record_fallback_and_numerical_contract_are_shared(state):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    query = protocol._query(4)
    entries = []
    for capacity in (64, 256):
        entry = knob_cache.make_record(
            {"fc2_use_bulk": capacity == 256},
            **{**query, "max_tokens": capacity, "live_tokens": capacity},
        )
        entries.append(entry)
    for i, entry in enumerate(entries):
        state.tuner.publish_protocol_record(
            tuning_protocol.NAMESPACE, str(i), entry, persist=False
        )
    assert state.run() == {"fc2_use_bulk": True}
    state.sweep.assert_not_called()
    state.workspace.config = dataclasses.replace(
        state.workspace.config, apply_routing_weights_before_fc2=False
    )
    with autotune_v2(mode="replay"):
        assert state.run() is None
    state.workspace.config = dataclasses.replace(
        state.workspace.config, apply_routing_weights_before_fc2=True
    )
    # Identical geometry with a different activation must not reuse a record.
    state.workspace.config = dataclasses.replace(
        state.workspace.config, activation="situ", situ_beta=1.0, situ_linear_beta=1.0
    )
    with autotune(False):
        assert state.run() is None
    state.workspace.config = dataclasses.replace(
        state.workspace.config,
        activation="swiglu",
        situ_beta=None,
        situ_linear_beta=None,
        max_sm_count=16,
    )
    # Records carrying a different implementation/resource contract are rejected.
    entry = {**entries[0], "protocol_contract": {"different": True}}
    state.tuner.clear_cache()
    state.tuner.publish_protocol_record(
        tuning_protocol.NAMESPACE, "entry", entry, persist=False
    )
    state.path.unlink()
    with autotune(False):
        assert state.run() is None

    # An incompatible bucket must not hide a compatible larger file bucket.
    knob_cache.write_record(entry)
    knob_cache.write_record(entries[1])
    with autotune(False):
        assert state.run() == {"fc2_use_bulk": True}


@pytest.mark.parametrize("routing_before_fc2", [False, True])
def test_native_cache_keyword_aliases(state, routing_before_fc2):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    query = {
        **protocol._query(4),
        "apply_routing_weights_before_fc2": routing_before_fc2,
    }
    legacy_query = dict(query)
    legacy_query["apply_topk_at_fc1"] = legacy_query.pop(
        "apply_routing_weights_before_fc2"
    )
    knobs = {"fc2_use_bulk": False}
    with pytest.warns(DeprecationWarning, match="apply_topk_at_fc1"):
        knob_cache.record_knobs(knobs, **legacy_query)
    assert knob_cache.lookup_knobs(**query) == knobs
    with pytest.warns(DeprecationWarning, match="apply_topk_at_fc1"):
        assert knob_cache.lookup_knobs(**legacy_query) == knobs
    cache = json.loads(state.path.read_text())
    assert cache["version"] == KNOB_CACHE_VERSION
    assert cache["entries"][0]["apply_routing_weights_before_fc2"] is routing_before_fc2
    assert "apply_topk_at_fc1" not in cache["entries"][0]


@pytest.mark.parametrize("change", ["backend_revision", "environment", "resources"])
def test_cache_compatibility_after_runtime_change(state, monkeypatch, change):
    import importlib

    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
    if change == "backend_revision":
        monkeypatch.setattr(knob_cache, "BACKEND_REVISION", "new-revision")
    elif change == "environment":
        module = importlib.import_module("flashinfer.autotuner.autotuner")
        metadata = {**module._collect_metadata(), "cutlass_dsl_version": "new-version"}
        monkeypatch.setattr(module, "_collect_metadata", lambda: metadata)
    else:
        state.workspace.config = dataclasses.replace(
            state.workspace.config, max_sm_count=16
        )
    with autotune_v2(mode="replay"):
        assert state.run() is None
    state.sweep.assert_called_once()
    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
    assert state.sweep.call_count == 2


def test_native_import_and_explicit_search(state):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    knob_cache.record_knobs({"fc2_use_bulk": True}, **protocol._query(4))
    with autotune(False):
        assert state.run() == {"fc2_use_bulk": True}
    state.sweep.assert_not_called()
    assert state.tuner.get_protocol_records(tuning_protocol.NAMESPACE)
    # A search that leaves the source workspace unchanged can run after capture.
    state.workspace._tuning_state.graph_captured = True
    for _ in range(2):
        assert native.autotune_sm107_block_scaled_mega_moe(
            None, None, None, state.workspace, candidates=[{"fc2_use_bulk": False}]
        ) == {"fc2_use_bulk": False}
    assert state.sweep.call_count == 2
    # Only the online import installs a winner; offline searches leave it alone.
    state.workspace.install_tuned_knobs.assert_called_once()


def test_pending_capture_freezes_workspace(state, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert state.run() is None
    state.workspace.install_tuned_knobs.assert_not_called()
    with autotune(True), pytest.raises(RuntimeError, match="before CUDA graph capture"):
        state.run()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with autotune(True), pytest.raises(RuntimeError, match="captured workspace"):
        state.run()
    state.sweep.assert_not_called()


def test_explicit_search_preserves_attached_measurement_policy(state):
    from flashinfer.autotune_cache import MeasurementPolicy

    policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True)
    with autotune_v2(measurement_policy=policy):
        state.run()
    store = state.tuner._managed_cache
    assert native.autotune_sm107_block_scaled_mega_moe(
        None, None, None, state.workspace, candidates=[{"fc2_use_bulk": False}]
    ) == {"fc2_use_bulk": False}
    assert state.tuner._managed_cache is store
    assert state.sweep.call_count == 2
    measurement = state.sweep.call_args.kwargs["measurement"]
    assert measurement["cold_l2"] is True
    assert measurement["cuda_graph_profile_replays"] == 1
    with autotune_v2(mode="replay", measurement_policy=policy):
        assert state.run() == {"fc2_use_bulk": False}
    assert state.sweep.call_count == 2


def test_foreign_records_are_filtered_before_kernel_validation(state, monkeypatch):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    query = protocol._query(4)
    matching = knob_cache.make_record(
        {"fc2_use_bulk": False}, **{**query, "max_tokens": 256, "live_tokens": 256}
    )
    foreign = {**matching, "hidden": query["hidden"] * 2}
    build = Mock()
    monkeypatch.setattr(block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", build)
    with autotune_v2(mode="replay", persistent_cache=False):
        for key, record in (("foreign", foreign), ("matching", matching)):
            state.tuner.publish_protocol_record(
                tuning_protocol.NAMESPACE, key, record, persist=False
            )
        assert state.run() == {"fc2_use_bulk": False}
    build.assert_called_once()


def test_execution_storage_allocation_preserves_staged_state(state, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import comm

    workspace = object.__new__(block_scaled.Sm107BlockScaledSymmBuffer)
    workspace.config = dataclasses.replace(
        state.workspace.config, reduce_topk_in_kernel=True
    )
    workspace.kernel = SimpleNamespace(
        _device_workspace=SimpleNamespace(total_bytes=lambda kind: 128)
    )
    workspace._routing_checks = {"routing": object()}
    workspace.note_staged_tokens(4)
    scalar = torch.tensor([2.0])
    workspace.bind_scalars({"fc1_alpha": scalar})
    storage = Mock()
    storage.data_ptr.return_value = 128
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: storage)
    monkeypatch.setattr(comm, "sym_zeros", lambda *a, **k: storage)
    monkeypatch.setattr(comm, "compute_peer_offsets", lambda *a: (0, [0]))
    workspace._allocate_execution_state()
    assert workspace._scalar_tensor("fc1_alpha") is scalar
    assert workspace._routing_checks
    assert workspace.staged_tokens() == 4


def test_cli_persists_managed_cache_by_default(state, monkeypatch, tmp_path):
    from flashinfer.moe_ep import tune
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import tuning
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setattr(package, "require_sm107_dsl", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def tune_one(*args):
        return native.autotune_sm107_block_scaled_mega_moe(
            None, None, None, state.workspace, candidates=[{"fc2_use_bulk": False}]
        )

    monkeypatch.setattr(tuning, "tune_one", tune_one)
    options = [
        "--arch",
        "sm107",
        "--dtype",
        "nvfp4",
        "--hidden",
        "128",
        "--intermediate",
        "64",
        "--num-experts",
        "4",
        "--topk",
        "2",
        "--max-tokens",
        "128",
    ]
    for _ in range(2):
        assert tune.main(options) == 0
        state.tuner.clear_cache()
        state.tuner._managed_cache = None
        state.tuner._managed_stores.clear()
        with autotune_v2(mode="replay"):
            assert state.tuner.get_protocol_records(tuning_protocol.NAMESPACE)
        assert state.path.is_file()
    assert state.sweep.call_count == 2
    with pytest.raises(SystemExit):
        tune.main(options + ["--autotune-cache", str(tmp_path / "unused.json")])


def test_memory_only_skips_both_disk_caches_and_policy_is_explicit(state, tmp_path):
    from flashinfer.autotune_cache import MeasurementPolicy

    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    knob_cache.record_knobs({"fc2_use_bulk": True}, **protocol._query(4))
    original = state.path.read_bytes()
    with autotune_v2(mode="replay", persistent_cache=False):
        assert state.run() is None
    with autotune_v2(persistent_cache=False):
        assert state.run() == {"fc2_use_bulk": False}
    assert state.path.read_bytes() == original
    assert not (tmp_path / "managed").exists()
    with autotune_v2(measurement_policy=MeasurementPolicy(execution_mode="eager")):
        assert state.run() == {"fc2_use_bulk": False}
    assert state.sweep.call_args.kwargs["measurement"]["timer"] == "events_no_delay"


def test_replay_import_requires_provenance_before_tuning_reuse(state):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    knob_cache.record_knobs({"fc2_use_bulk": True}, **protocol._query(4))
    with autotune_v2(mode="replay"):
        assert state.run() == {"fc2_use_bulk": True}
    state.sweep.assert_not_called()
    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
    state.sweep.assert_called_once()
    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
    state.sweep.assert_called_once()


def test_compatibility_file_failure_does_not_break_selection(state, monkeypatch):
    def unavailable(*args):
        raise OSError("cache storage is unavailable")

    monkeypatch.setattr(knob_cache, "_load_entries", unavailable)
    monkeypatch.setattr(knob_cache, "write_record", unavailable)
    with autotune_v2():
        assert state.run() == {"fc2_use_bulk": False}
    state.sweep.assert_called_once()
    with autotune_v2(mode="replay"):
        assert state.run() == {"fc2_use_bulk": False}
    state.sweep.assert_called_once()


@pytest.mark.parametrize("fallback", ["empty_store", "skip_ops"])
def test_automatic_workspace_returns_to_initial_knobs(
    state, monkeypatch, tmp_path, fallback
):
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import automatic
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setattr(package, "sm107_candidates", lambda *a, **k: [{}])
    backend = SimpleNamespace(
        _kernel_config=SimpleNamespace(knobs="auto", in_kernel_fc2_reduce=False),
        ep_world_size=1,
        ep_comm_group=None,
    )
    workspace = state.workspace
    workspace.config = dataclasses.replace(workspace.config, fc2_use_bulk=True)

    def select():
        automatic.prepare_auto_configuration(backend, workspace, (None, None), None, 4)

    with autotune_v2():
        select()
    assert workspace._tuning_state.winner == {"fc2_use_bulk": False}
    assert automatic.get_tuned_knobs(workspace) == workspace._tuning_state.winner
    options = (
        {"cache_root": str(tmp_path / "empty")}
        if fallback == "empty_store"
        else {"skip_ops": {tuning_protocol.NAMESPACE}}
    )
    with autotune_v2(mode="replay", **options):
        select()
        assert (
            workspace._tuning_state.cache_identity
            == automatic.tuning_context().cache_identity
        )
    assert workspace._tuning_state.winner is None
    with pytest.raises(RuntimeError, match="no completed automatic tuning result"):
        automatic.get_tuned_knobs(workspace)
    assert workspace._tuning_state.tuned_identity is None
    initial = workspace._tuning_state.initial_knobs
    assert initial["fc2_use_bulk"] is True
    workspace.install_tuned_knobs.assert_called_with(initial, None, None)
    with autotune_v2(mode="replay"):
        select()
    assert workspace._tuning_state.winner == {"fc2_use_bulk": False}
    assert automatic.get_tuned_knobs(workspace) == workspace._tuning_state.winner
    state.sweep.assert_called_once()
    if fallback == "skip_ops":
        workspace._tuning_state.graph_captured = True
        with (
            autotune_v2(skip_ops={tuning_protocol.NAMESPACE}),
            pytest.raises(RuntimeError, match="skip_ops requires fallback"),
        ):
            select()


def test_capture_cannot_accept_a_cache_change_without_selection(state, monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import automatic
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

    monkeypatch.setattr(package, "sm107_candidates", lambda *a, **k: [{}])
    backend = SimpleNamespace(
        _kernel_config=SimpleNamespace(knobs="auto", in_kernel_fc2_reduce=False),
        ep_world_size=1,
        ep_comm_group=None,
    )
    workspace = state.workspace
    automatic.prepare_auto_configuration(backend, workspace, (None, None), None, 4)
    previous = workspace._tuning_state.cache_identity
    state.tuner.clear_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Warm up Rubin automatic selection"):
        automatic.coordinate_workspace_replacement(backend, workspace)
    # Explicit handles stay pinned and cannot claim a lookup from the new store.
    workspace._tuning_state.graph_captured = True
    automatic.prepare_auto_configuration(backend, workspace, (None, None), None, 4)
    assert workspace._tuning_state.cache_identity == previous
    state.sweep.assert_not_called()


@pytest.mark.parametrize("change", ["store", "policy"])
def test_replay_winner_invalidates_previous_tuning_identity(
    state, monkeypatch, tmp_path, change
):
    from flashinfer.autotune_cache import MeasurementPolicy
    from flashinfer.moe_ep.backends.mega.kernel.sm107 import automatic
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as package

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setattr(
        package,
        "sm107_candidates",
        lambda *a, **k: [{"fc2_use_bulk": False}, {"fc2_use_bulk": True}],
    )
    backend = SimpleNamespace(
        _kernel_config=SimpleNamespace(knobs="auto", in_kernel_fc2_reduce=False),
        ep_world_size=1,
        ep_comm_group=None,
    )
    first = state.workspace
    second = SimpleNamespace(
        config=first.config,
        staged_tokens=lambda: 4,
        _tuning_state=AutoTuningState(),
    )
    for workspace in (first, second):

        def install(knobs, *weights, workspace=workspace):
            workspace.config = dataclasses.replace(workspace.config, **knobs)

        workspace.install_tuned_knobs = Mock(side_effect=install)

    def select(workspace):
        automatic.prepare_auto_configuration(backend, workspace, (None, None), None, 4)

    options = (
        {"cache_root": tmp_path / "other"}
        if change == "store"
        else {"measurement_policy": MeasurementPolicy(execution_mode="eager")}
    )
    state.sweep.side_effect = [
        ({"fc2_use_bulk": False}, 1.0, 1),
        ({"fc2_use_bulk": True}, 1.0, 1),
    ]
    with autotune_v2():
        select(first)
    with autotune_v2(**options):
        select(second)
    with autotune_v2(mode="replay", **options):
        select(first)
        assert first.config.fc2_use_bulk is True
    with autotune_v2():
        # A captured workspace would need replacement for this same transition.
        first._tuning_state.graph_captured = True
        assert automatic.coordinate_workspace_replacement(backend, first)
        first._tuning_state.graph_captured = False
        select(first)
        assert first.config.fc2_use_bulk is False
        assert automatic.get_tuned_knobs(first) == {"fc2_use_bulk": False}
        assert not automatic.needs_selection(first, automatic.tuning_context())
    assert state.sweep.call_count == 2


def test_invalid_cached_resources_are_a_miss_before_bucket_selection(
    state, monkeypatch
):
    protocol = tuning_protocol.RubinTuningProtocol(None, None, None, state.workspace)
    for capacity, bulk in ((128, False), (256, True)):
        record = knob_cache.make_record(
            {"fc2_use_bulk": bulk},
            **{**protocol._query(4), "max_tokens": capacity, "live_tokens": capacity},
        )
        state.tuner.publish_protocol_record(
            tuning_protocol.NAMESPACE, str(capacity), record, persist=False
        )

    def build(config):
        if not config.fc2_use_bulk:
            raise ValueError("configuration exceeds available resources")

    monkeypatch.setattr(block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", build)
    assert state.run() == {"fc2_use_bulk": True}
    state.sweep.assert_not_called()


@pytest.mark.parametrize("mode", ["auto", "eager", "cuda_graph"])
@pytest.mark.parametrize("cold", [False, True])
def test_measurement_policy_reuse_is_isolated(state, mode, cold):
    from flashinfer.autotune_cache import MeasurementPolicy

    policy = MeasurementPolicy(execution_mode=mode, cold_l2=cold)
    with autotune_v2(measurement_policy=policy):
        state.run()
        settings = state.sweep.call_args.kwargs["measurement"]
        assert settings["cold_l2"] is cold
        assert settings["cuda_graph_profile_replays"] == (
            1 if mode == "cuda_graph" else None
        )
        assert (
            settings["timer"]
            == {
                "auto": "native_events",
                "eager": "events_no_delay",
                "cuda_graph": "events",
            }[mode]
        )
    with autotune_v2(measurement_policy=policy):
        state.run()
    state.sweep.assert_called_once()
    # A file winner for one policy must not satisfy another policy's replay.
    with autotune_v2(
        mode="replay",
        measurement_policy=MeasurementPolicy(execution_mode=mode, cold_l2=not cold),
    ):
        assert state.run() is None
    state.sweep.assert_called_once()


def _subgroup_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        patch.setattr(
            block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", lambda cfg: None
        )
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=3,
            timeout=timedelta(seconds=30),
        )
        group = dist.new_group([1, 2])
        try:
            if rank != 0:
                cfg = block_scaled.Sm107BlockScaledMoeConfig(
                    num_total_experts=4,
                    max_tokens_per_rank=128,
                    num_topk=2,
                    hidden=128,
                    intermediate=64,
                    rank=rank - 1,
                    world_size=2,
                )
                workspace = SimpleNamespace(
                    config=cfg,
                    staged_tokens=lambda: rank * 4,
                    install_tuned_knobs=Mock(),
                    _tuning_state=AutoTuningState(),
                )
                protocol = tuning_protocol.RubinTuningProtocol(
                    None,
                    None,
                    None,
                    workspace,
                    candidates=[{"fc2_use_bulk": False}],
                    process_group=group,
                    install_winner=True,
                )
                tuner = AutoTuner.get()
                tuner.clear_cache()
                if rank == 1:
                    record = knob_cache.make_record(
                        {"fc2_use_bulk": False}, **protocol._query(16)
                    )
                    tuner.publish_protocol_record(
                        tuning_protocol.NAMESPACE, "winner", record, persist=False
                    )
                with autotune(False):
                    assert tuner.choose_with_protocol(
                        tuning_protocol.NAMESPACE, protocol
                    ) == {"fc2_use_bulk": False}
                workspace.install_tuned_knobs.assert_called_once()

                # Both ranks have a hit, but only collective selection gives
                # them the same closest-capacity winner.
                tuner.clear_cache()
                record = knob_cache.make_record(
                    {"fc2_use_bulk": rank == 2},
                    **{**protocol._query(16), "max_tokens": 128 * rank},
                )
                tuner.publish_protocol_record(
                    tuning_protocol.NAMESPACE, "local-winner", record, persist=False
                )
                with autotune(False):
                    assert tuner.choose_with_protocol(
                        tuning_protocol.NAMESPACE, protocol
                    ) == {"fc2_use_bulk": False}
                assert workspace.install_tuned_knobs.call_count == 2
                with autotune(rank == 1), pytest.raises(ValueError, match="controls"):
                    tuner.choose_with_protocol(tuning_protocol.NAMESPACE, protocol)

                def prepare():
                    if rank == 1:
                        raise ValueError("candidate compilation failed")

                with pytest.raises(RuntimeError, match="candidate compilation failed"):
                    native._prepare_collectively(prepare, 2, group)

                from flashinfer.moe_ep.backends.mega.kernel.sm107 import automatic

                backend = SimpleNamespace(
                    _kernel_config=SimpleNamespace(
                        knobs="auto", in_kernel_fc2_reduce=False
                    ),
                    ep_world_size=2,
                    ep_comm_group=group,
                )
                from flashinfer.moe_ep.kernel_src.sm107 import (
                    next_cutedsl_megamoe as package,
                )

                patch.setattr(package, "sm107_candidates", lambda *a, **k: [{}])
                for explicit in (False, True):
                    workspace._tuning_state = AutoTuningState()
                    workspace.install_tuned_knobs.reset_mock()
                    with autotune_v2(mode="replay", persistent_cache=False):
                        tuner.clear_cache()

                        def select():
                            if not explicit:
                                assert not automatic.coordinate_workspace_replacement(
                                    backend, workspace
                                )
                            automatic.prepare_auto_configuration(
                                backend, workspace, (None, None), None, rank * 4
                            )

                        # Both ranks remember a miss. Only one sees a new record.
                        select()
                        select()
                        workspace.install_tuned_knobs.assert_not_called()
                        if rank == 1:
                            tuner.clear_cache()
                            tuner.publish_protocol_record(
                                tuning_protocol.NAMESPACE,
                                "winner",
                                knob_cache.make_record(
                                    {"fc2_use_bulk": False}, **protocol._query(16)
                                ),
                                persist=False,
                            )
                        select()
                        assert automatic.get_tuned_knobs(workspace) == {
                            "fc2_use_bulk": False
                        }
                        workspace.install_tuned_knobs.assert_called_once()
                        select()
                        workspace.install_tuned_knobs.assert_called_once()
                        # A settled peer must join selection after local invalidation.
                        if rank == 2:
                            tuner.clear_cache()
                        select()
                        assert workspace.install_tuned_knobs.call_count == 2

                with autotune_v2(mode="replay", persistent_cache=False):
                    workspace._tuning_state.graph_captured = True
                    workspace._tuning_state.winner = {"fc2_use_bulk": False}
                    workspace._tuning_state.cache_identity = (
                        automatic.tuning_context().cache_identity
                    )
                    assert not automatic.coordinate_workspace_replacement(
                        backend, workspace
                    )
                    # The settled peer must still participate after another
                    # rank invalidates its process-local cache.
                    if rank == 1:
                        tuner.clear_cache()
                    assert automatic.coordinate_workspace_replacement(
                        backend, workspace
                    )
            dist.barrier()
        finally:
            dist.destroy_process_group()


def test_ep_subgroup_cache_agreement_and_mode_mismatch(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _subgroup_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=3,
        join=True,
    )
