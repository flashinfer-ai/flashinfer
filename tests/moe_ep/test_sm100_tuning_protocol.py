"""Blackwell managed tuning, exact native-cache lookup and workspace state."""

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer import autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2_reload
from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION
from flashinfer.moe_ep.core.tuning_state import AutoTuningState
from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe import (
    Sm100BlockScaledMoeConfig,
    Sm100GenphaseMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
    autotune as native,
    knob_cache,
    tuning,
    tuning_protocol,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100 import automatic


class Workspace(SimpleNamespace):
    _build_kernel = staticmethod(lambda cfg: None)


@pytest.fixture(params=["standard", "genphase"])
def state(request, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    monkeypatch.setattr(knob_cache, "device_identity", lambda: "test-device")
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "managed"))
    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    tuner = AutoTuner.get()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner.clear_cache()
    tuner.reset_statistics()
    cls = (
        Sm100GenphaseMoeConfig
        if request.param == "genphase"
        else Sm100BlockScaledMoeConfig
    )
    cfg = cls(
        num_total_experts=8,
        max_tokens_per_rank=128,
        num_topk=2,
        hidden=512,
        intermediate=256,
        rank=0,
        world_size=1,
    )
    knobs = (
        tuning.sm100_genphase_candidates(cfg, sm_count=148, live_tokens=4)
        if request.param == "genphase"
        else tuning.sm100_standard_candidates(cfg)
    )[0]
    workspace = Workspace(
        config=cfg,
        staged_tokens=lambda: 4,
        install_tuned_knobs=Mock(),
        _tuning_state=AutoTuningState(),
    )
    sweep = Mock(return_value=(knobs, 1.0, 1))
    monkeypatch.setattr(native, "_sweep_sm100_mega_moe", sweep)

    def run(**kwargs):
        return native.autotune_sm100_block_scaled_mega_moe(
            None,
            None,
            None,
            workspace,
            candidates=[knobs],
            **{"force_tune": False, "install_winner": True, **kwargs},
        )

    yield SimpleNamespace(
        workspace=workspace,
        knobs=knobs,
        sweep=sweep,
        run=run,
        tuner=tuner,
        path=path,
        root=tmp_path / "managed",
    )
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()


def test_deferred_search_and_fresh_store_replay_writes_native_file(state):
    other = {"backend_revision": "unrelated"}
    state.path.write_text(
        json.dumps({"version": KNOB_CACHE_VERSION, "entries": [other]})
    )
    with autotune_v2(mode="replay"):
        assert state.run() is None
    state.sweep.assert_not_called()
    with autotune_v2():
        assert state.run() == state.knobs
    state.sweep.assert_called_once()
    contents = json.loads(state.path.read_text())
    assert other in contents["entries"]
    records = list(state.root.glob("v2/*/entries/*.json"))
    assert records
    state.path.unlink()
    state.tuner.clear_cache()
    state.tuner._managed_cache = None
    state.tuner._managed_stores.clear()
    with autotune_v2(mode="replay"):
        assert state.run() == state.knobs
    entry = json.loads(state.path.read_text())["entries"][0]
    assert entry == next(e for e in contents["entries"] if e != other)
    state.sweep.assert_called_once()


def test_native_import_exact_workload_and_configuration(state):
    cfg = state.workspace.config
    knob_cache.record_knobs(state.knobs, cfg, live_tokens=4, p50_us=1.0)
    with autotune_v2(mode="replay"):
        assert state.run() == state.knobs
    assert state.tuner.get_protocol_records(tuning_protocol.NAMESPACE)
    assert not list(state.root.glob("v2/*/entries/*.json"))
    for changes, count in [
        ({"max_tokens_per_rank": 256}, 4),
        ({}, 5),
        ({"gate_up_clamp": 7.0}, 4),
        ({"max_sm_count": 16}, 4),
    ]:
        state.workspace.config = dataclasses.replace(cfg, **changes)
        with autotune_v2(mode="replay"):
            assert state.run(num_tokens=count) is None
    state.sweep.assert_not_called()


def test_memory_partition_skip_and_forced_offline_search(state):
    knob_cache.record_knobs(
        state.knobs, state.workspace.config, live_tokens=4, p50_us=1.0
    )
    before = state.path.read_bytes()
    with autotune_v2(mode="replay", persistent_cache=False):
        assert state.run() is None
    with autotune_v2(persistent_cache=False):
        assert state.run() == state.knobs
    assert state.path.read_bytes() == before
    assert not list(state.root.glob("v2/*/entries/*.json"))
    with autotune_v2(mode="replay", persistent_cache=False):
        assert state.run() == state.knobs
    with autotune_v2(skip_ops={tuning_protocol.NAMESPACE}):
        assert state.run() is None
        assert state.run(force_tune=True, install_winner=False) == state.knobs
    assert state.sweep.call_count == 2
    # Offline search leaves the caller's workspace and compiled winner untouched.
    assert state.workspace.install_tuned_knobs.call_count == 2
    assert not list(state.root.glob("v2/*/entries/*.json"))
    with autotune_v2():
        state.run(force_tune=True, install_winner=False)
    assert list(state.root.glob("v2/*/entries/*.json"))


@pytest.mark.parametrize("options", [{"round_up": True}, {"tuning_buckets": (64, 128)}])
def test_unsupported_bucket_controls_fail_before_launch(state, options):
    with (
        autotune_v2(**options),
        pytest.raises(NotImplementedError, match="tuning_buckets"),
    ):
        state.run()
    state.sweep.assert_not_called()
    state.workspace.install_tuned_knobs.assert_not_called()


def test_measurement_defaults_and_explicit_policy(state):
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.measurement import (
        measurement_settings,
    )

    with autotune_v2(persistent_cache=False):
        context = state.tuner.protocol_context(tuning_protocol.NAMESPACE)
        assert measurement_settings(context) == {
            "timer": "native_events",
            "cold_l2": False,
            "cuda_graph_profile_replays": 1,
            "calls_per_sample": 8,
            "warmup_iters": 8,
        }
        state.run()
    with autotune_v2(
        persistent_cache=False,
        measurement_policy=MeasurementPolicy(execution_mode="eager", cold_l2=True),
    ):
        context = state.tuner.protocol_context(tuning_protocol.NAMESPACE)
        settings = measurement_settings(context)
        assert settings["cuda_graph_profile_replays"] is None
        assert settings["cold_l2"] and "calls_per_sample" not in settings
        state.run()
    assert state.sweep.call_count == 2


def test_captured_explicit_handle_and_default_replacement(state):
    state.workspace._tuning_state.graph_captured = True
    context = None
    with autotune_v2():
        context = automatic.tuning_context()
        backend = SimpleNamespace(
            _config=SimpleNamespace(knobs="auto"), ep_world_size=1
        )
        assert automatic.coordinate_workspace_replacement(backend, state.workspace)
        with pytest.raises(RuntimeError, match="captured workspace"):
            state.run()
    state.workspace._tuning_state.record_selection(
        state.knobs,
        context.cache_identity,
        automatic.tuning_identity(context, state.workspace),
    )
    with autotune_v2():
        assert not automatic.coordinate_workspace_replacement(backend, state.workspace)
        autotune_v2_reload()
        assert automatic.coordinate_workspace_replacement(backend, state.workspace)
    state.sweep.assert_not_called()


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
def test_standard_grid_is_complete_unique_and_fixed_numerics(kind):
    cfg = Sm100BlockScaledMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=128,
        num_topk=2,
        hidden=512,
        intermediate=256,
        rank=0,
        world_size=1,
        quant_kind=kind,
        activation="situ",
        situ_beta=7.0,
    )
    if kind == "nvfp4":
        # The legacy heuristic starts at K256, which cannot be paired with N256.
        cfg = dataclasses.replace(cfg, mma_tiler_mnk=(256, 128, 256))
    candidates = tuning.sm100_standard_candidates(cfg, allow_nondeterministic=True)
    assert len({json.dumps(c, sort_keys=True) for c in candidates}) == len(candidates)
    assert {c["mma_tiler_mnk"][1] for c in candidates} == {128, 256}
    assert {c["reduce_topk_in_kernel"] for c in candidates} == {False, True}
    assert {c["fc2_use_bulk"] for c in candidates} == {False, True}
    for knobs in candidates:
        assert set(knobs) == set(tuning.STANDARD_KNOB_KEYS)
        changed = dataclasses.replace(cfg, **knobs)
        assert (
            changed.activation == cfg.activation and changed.situ_beta == cfg.situ_beta
        )
    for changes in (
        {"use_custom_finalize": True},
        {"use_persistent_finalize_kernel": True},
        {"apply_routing_weights_before_fc2": False},
        {"combine_dtype": "mxfp8"},
    ):
        assert not any(
            c["reduce_topk_in_kernel"]
            for c in tuning.sm100_standard_candidates(
                dataclasses.replace(cfg, **changes), allow_nondeterministic=True
            )
        )


def test_genphase_grid_retains_geometries_hints_and_constraints():
    """Keep each admitted geometry and scheduling hint under knob constraints."""
    cfg = Sm100GenphaseMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=128,
        num_topk=2,
        hidden=512,
        intermediate=256,
        rank=0,
        world_size=1,
    )
    candidates = tuning.sm100_genphase_candidates(cfg, sm_count=148, live_tokens=4)
    assert len(candidates) == 12
    for tile, cluster in cfg.admitted_geometries:
        hints = {
            c["schedule_policy"][1]
            for c in candidates
            if c["mma_tiler_mnk"] == tile and c["cluster_shape_mn"] == cluster
        }
        assert hints == {
            None,
            2 * (148 // cluster[0]),
            4 * (148 // cluster[0]),
            8 * (148 // cluster[0]),
        }
    constraints = {"mma_tiler_mnk": (256, 64, 128), "schedule_policy": ("grouped", 3)}
    constrained = tuning.sm100_genphase_candidates(
        cfg, sm_count=148, constraints=constraints
    )
    assert len(constrained) == 2
    assert all(all(c[k] == v for k, v in constraints.items()) for c in constrained)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("finalize_mode", ["DEFAULT", "NONE"])
def test_standard_getter_knobs_install_into_fixed_workspace_config(kind, finalize_mode):
    from flashinfer import moe_ep

    cls = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    options = {} if kind == "nvfp4" else {"kind": kind}
    cfg = cls(
        256,
        2,
        kernel_name=f"sm100_{'nvfp4_nvfp4' if kind == 'nvfp4' else 'mxfp8_mxfp8'}_bf16_cutedsl_next",
        knobs="auto",
        finalize_mode=moe_ep.FinalizeMode[finalize_mode],
        **options,
    )
    backend = create_mega_kernel(cfg)
    backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=512)
    initial = backend._workspace_config(fleet)
    knobs = tuning.sm100_standard_candidates(initial)[-1]
    fixed = create_mega_kernel(dataclasses.replace(cfg, knobs=knobs))
    fixed.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
    assert fixed._workspace_config(fleet) == dataclasses.replace(initial, **knobs)


def test_genphase_auto_ignores_fixed_profile_count(monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    cfg = moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        256, 3, kernel_variant="genphase", knobs="auto"
    )
    backend = create_mega_kernel(cfg)
    backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=512)
    resolve = Mock(wraps=pkg.resolve_genphase_knobs)
    monkeypatch.setattr(pkg, "resolve_genphase_knobs", resolve)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    initial = backend._genphase_config(fleet)
    assert (
        backend._genphase_config(dataclasses.replace(fleet, tuning_live_tokens=16))
        == initial
    )
    assert resolve.call_count == 1
    assert resolve.call_args.kwargs["live_tokens"] is None
    fixed = create_mega_kernel(dataclasses.replace(cfg, knobs=None))
    fixed.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
    fixed._genphase_config(dataclasses.replace(fleet, tuning_live_tokens=16))
    assert resolve.call_args.kwargs["live_tokens"] == 16


def _subgroup_worker(rank, rendezvous):
    from datetime import timedelta
    import torch.distributed as dist

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(torch.cuda, "current_device", lambda: 0)
        patch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _: SimpleNamespace(multi_processor_count=148),
        )
        patch.setattr(knob_cache, "device_identity", lambda: "test-device")
        patch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        dist.init_process_group(
            "gloo",
            init_method=rendezvous,
            rank=rank,
            world_size=3,
            timeout=timedelta(seconds=30),
        )
        group = dist.new_group([1, 2])
        try:
            if rank:
                cfg = Sm100GenphaseMoeConfig(
                    num_total_experts=8,
                    max_tokens_per_rank=128,
                    num_topk=2,
                    hidden=512,
                    intermediate=256,
                    rank=rank - 1,
                    world_size=2,
                )
                candidates = tuning.sm100_genphase_candidates(
                    cfg, sm_count=148, live_tokens=8
                )
                workspace = Workspace(
                    config=cfg,
                    staged_tokens=lambda: rank * 4,
                    install_tuned_knobs=Mock(),
                    _tuning_state=AutoTuningState(),
                )
                backend = SimpleNamespace(
                    _config=SimpleNamespace(
                        knobs="auto", enable_in_kernel_fc2_reduce=False
                    ),
                    _genphase_constraints=lambda: {},
                    ep_world_size=2,
                    ep_comm_group=group,
                )
                tuner = AutoTuner.get()
                with autotune_v2(mode="replay", persistent_cache=False):
                    tuner.clear_cache()
                    # Each rank has a different winner. Both must install the same one.
                    record = knob_cache.make_record(
                        candidates[rank], cfg, live_tokens=8, p50_us=1.0
                    )
                    tuner.publish_protocol_record(
                        tuning_protocol.NAMESPACE, "rank-local", record, persist=False
                    )
                    automatic.prepare_auto_configuration(
                        backend, workspace, (None, None), None, rank * 4
                    )
                    assert automatic.get_tuned_knobs(workspace) == candidates[2]
                    # A settled rank must still participate in its peer's invalidation.
                    if rank == 1:
                        tuner.clear_cache()
                    automatic.prepare_auto_configuration(
                        backend, workspace, (None, None), None, rank * 4
                    )
                    assert workspace.install_tuned_knobs.call_count == 2
                    workspace._tuning_state.graph_captured = True
                    if rank == 2:
                        tuner.clear_cache()
                    assert automatic.coordinate_workspace_replacement(
                        backend, workspace
                    )

                workspace._tuning_state = AutoTuningState()
                protocol = tuning_protocol.BlackwellTuningProtocol(
                    None,
                    None,
                    None,
                    workspace,
                    candidates=candidates,
                    process_group=group,
                    install_winner=True,
                )
                with (
                    autotune_v2(
                        mode="tune" if rank == 1 else "replay", persistent_cache=False
                    ),
                    pytest.raises(ValueError, match="controls"),
                ):
                    tuner.choose_with_protocol(tuning_protocol.NAMESPACE, protocol)

                def prepare():
                    if rank == 1:
                        raise ValueError("candidate preparation failed")

                with pytest.raises(RuntimeError, match="candidate preparation failed"):
                    native._prepare_collectively(prepare, 2, group)
            dist.barrier()
        finally:
            dist.destroy_process_group()


def test_ep_subgroup_agreement_and_preparation_failure(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _subgroup_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=3,
        join=True,
    )


def test_cli_forces_search_and_persists_by_default(state, monkeypatch):
    from flashinfer.moe_ep import tune
    from flashinfer.moe_ep.backends.mega.kernel.sm100 import next_tuning

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        next_tuning,
        "tune_one",
        lambda *a: state.run(force_tune=True, install_winner=False, warmup_iters=3),
    )
    options = [
        "--arch",
        "sm100",
        "--implementation",
        "next",
        "--dtype",
        "mxfp8_e4m3",
        "--hidden",
        "512",
        "--intermediate",
        "256",
        "--num-experts",
        "8",
        "--topk",
        "2",
        "--max-tokens",
        "128",
    ]
    if state.workspace.config.kernel_variant == "genphase":
        options.extend(["--kernel-variant", "genphase"])
    for _ in range(2):
        assert tune.main(options) == 0
        assert state.path.is_file()
        assert list(state.root.glob("v2/*/entries/*.json"))
    assert state.sweep.call_count == 2
    state.workspace.install_tuned_knobs.assert_not_called()


def test_cli_measurement_reuses_online_warmup_defaults(state):
    with autotune_v2():
        state.run(force_tune=True, install_winner=False, warmup_iters=3, timed_iters=10)
        assert state.run() == state.knobs
    state.sweep.assert_called_once()


def test_native_cold_policy_has_one_call_per_flush(state):
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.measurement import (
        measurement_settings,
    )

    with autotune_v2(
        persistent_cache=False, measurement_policy=MeasurementPolicy(cold_l2=True)
    ):
        settings = measurement_settings(automatic.tuning_context())
        assert settings["calls_per_sample"] == 1
        assert settings["cold_l2"] and settings["cuda_graph_profile_replays"] == 1
