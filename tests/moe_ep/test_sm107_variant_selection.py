"""Rubin variant admission, cache selection and fixed-config handoff."""

import dataclasses
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer import autotune_v2, moe_ep
from flashinfer.autotuner import AutoTuner
from flashinfer.moe_ep.backends.mega.kernel.sm107 import variant_selection as selection
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.tuning_state import AutoTuningState
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    block_scaled,
    knob_cache,
)


CONFIGS = [
    (moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig, {}),
    (moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig, {"kind": "mxfp8_e4m3"}),
    (moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig, {"kind": "mxfp8_e5m2"}),
    (moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig, {}),
]


@pytest.fixture(autouse=True)
def host(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", lambda cfg: None
    )
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "managed"))
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    tuner = AutoTuner.get()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner.clear_cache()
    yield
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()


@pytest.mark.parametrize("config_type,options", CONFIGS)
@pytest.mark.parametrize(
    "capacity,combine,expected",
    [
        (128, "bf16", ["inference", "genphase"]),
        (1025, "bf16", ["inference"]),
        (128, "mxfp8", ["inference"]),
    ],
)
def test_variant_admission_and_fixed_config(
    config_type, options, capacity, combine, expected
):
    config = config_type(
        256, 3, kernel_variant="auto", knobs="auto", combine_dtype=combine, **options
    )
    backend = create_mega_kernel(config)
    assert isinstance(backend, selection.RubinVariantBackend)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=capacity, token_hidden_size=512
    )
    assert backend._eligible(fleet) == expected
    variant = expected[-1]
    native = backend._backend(variant)._resolved_config(fleet)
    knobs = {
        "mma_tiler_mnk": native.mma_tiler_mnk,
        "cluster_shape_mn": native.cluster_shape_mn,
        "fc2_use_bulk": native.fc2_use_bulk,
    }
    workspace = SimpleNamespace(
        _tuning_state=AutoTuningState(winner={"variant": variant, "knobs": knobs})
    )
    fixed = backend.get_tuned_config(workspace)
    assert type(fixed) is config_type
    assert fixed.kernel_variant == variant
    assert not isinstance(create_mega_kernel(fixed), selection.RubinVariantBackend)
    assert backend.get_tuned_knobs(workspace) == knobs
    fixed.knobs["fc2_use_bulk"] = not native.fc2_use_bulk
    assert (
        workspace._tuning_state.winner["knobs"]["fc2_use_bulk"] == native.fc2_use_bulk
    )


def test_explicit_variants_keep_existing_construction():
    for variant in ("inference", "genphase"):
        config = moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            256, 3, kernel_variant=variant
        )
        assert not isinstance(create_mega_kernel(config), selection.RubinVariantBackend)
    with pytest.raises(ValueError, match="requires knobs"):
        create_mega_kernel(dataclasses.replace(config, kernel_variant="auto"))


@pytest.fixture
def protocol(monkeypatch):
    config = moe_ep.Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        256, 3, kernel_variant="auto", knobs="auto"
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(
        num_experts=8, max_tokens_per_rank=128, token_hidden_size=512
    )
    raw = SimpleNamespace(config=backend._fallback._resolved_config(fleet))
    owner = selection.VariantWorkspace(
        fleet, Mock(), raw, tensors=SimpleNamespace(num_tokens=4)
    )
    protocol = selection._VariantProtocol(backend, owner, (None, None), None)
    contract = {"variants": ["inference", "genphase"]}
    monkeypatch.setattr(backend, "_eligible", lambda fleet: contract["variants"])
    monkeypatch.setattr(protocol, "_contract", lambda variants, context: contract)
    monkeypatch.setattr(protocol, "_install", Mock())
    return protocol, contract


def test_cache_hit_reconstructs_tuples_and_republishes_file(protocol, monkeypatch):
    protocol, contract = protocol
    key = hashlib.sha256(selection._json(contract).encode()).hexdigest()
    knobs = {"cluster_shape_mn": [4, 1], "fc2_use_bulk": True}
    native = {"knobs": knobs}
    record = {
        "contract": contract,
        "variant": "genphase",
        "knobs": knobs,
        "knob_record": native,
    }
    write = Mock()
    monkeypatch.setattr(knob_cache, "write_record", write)
    with autotune_v2(persistent_cache=False):
        context = AutoTuner.get().protocol_context(selection.NAMESPACE)
        record["measurement"] = selection._identity(context)[1]
        context.publish(key, record)
    with autotune_v2(mode="replay", persistent_cache=False):
        result = AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol)
        assert result["variant"] == "genphase"
        assert result["knobs"]["cluster_shape_mn"] == (4, 1)
        write.assert_not_called()
    protocol.owner._tuning_state = AutoTuningState()
    with autotune_v2():
        context = AutoTuner.get().protocol_context(selection.NAMESPACE)
        context.publish(key, record)
    with autotune_v2(mode="replay"):
        result = AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol)
        assert result["variant"] == "genphase"
        write.assert_called_once_with(native)
    assert protocol._install.call_count == 2


def test_miss_remains_pending_and_skip_uses_fallback(protocol, monkeypatch):
    protocol, _ = protocol
    allocation = Mock(return_value=SimpleNamespace(kernel=Mock(), workspace=Mock()))
    monkeypatch.setattr(protocol.backend, "_allocate_workspace", allocation)
    with autotune_v2(mode="replay", persistent_cache=False):
        assert (
            AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol) is None
        )
        allocation.assert_not_called()
        state = protocol.owner._tuning_state
        assert state.winner is None
        assert not selection._pending(state, selection._context())
    with autotune_v2(persistent_cache=False):
        assert selection._pending(state, selection._context())
    state.winner = {"variant": "genphase", "knobs": {}}
    with autotune_v2(skip_ops={selection.NAMESPACE}, persistent_cache=False):
        assert (
            AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol) is None
        )
        allocation.assert_called_once()
        assert state.winner is None
    protocol._install.assert_not_called()


@pytest.mark.parametrize("failure", ["resource", "knob_record"])
def test_invalid_cached_variant_is_a_collective_miss(protocol, monkeypatch, failure):
    protocol, contract = protocol
    record = {"contract": contract, "variant": "inference", "knobs": {}}
    if failure == "knob_record":
        record["knob_record"] = {"knobs": {"fc2_use_bulk": True}}
    else:

        def invalid_resource(config):
            raise AssertionError("cached tile no longer satisfies a kernel constraint")

        monkeypatch.setattr(
            block_scaled.Sm107BlockScaledSymmBuffer, "_build_kernel", invalid_resource
        )
    key = hashlib.sha256(selection._json(contract).encode()).hexdigest()
    with autotune_v2(mode="replay", persistent_cache=False):
        selection._context().publish(key, record)
        assert (
            AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol) is None
        )
    protocol._install.assert_not_called()


def test_captured_default_replacement_requires_pending_selection(protocol):
    protocol, _ = protocol
    owner = protocol.owner
    with autotune_v2(mode="replay", persistent_cache=False):
        context = selection._context()
        owner._tuning_state.record_selection(
            None, selection._cache_identity(context), None
        )
        owner._tuning_state.lookup_scope = (
            context.scope,
            selection._cache_identity(context),
        )
        owner._tuning_state.graph_captured = True
        assert not protocol.backend.coordinate_workspace_replacement(owner)
    with autotune_v2(persistent_cache=False):
        assert protocol.backend.coordinate_workspace_replacement(owner)
        owner._tuning_state.record_selection(
            {"variant": "inference", "knobs": {}},
            selection._cache_identity(selection._context()),
            selection._identity(selection._context()),
        )
        assert not protocol.backend.coordinate_workspace_replacement(owner)


def test_inner_skip_changes_selection_identity(protocol):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE as NATIVE_NAMESPACE,
    )

    protocol, _ = protocol
    state = protocol.owner._tuning_state
    with autotune_v2(persistent_cache=False):
        context = selection._context()
        state.record_selection(
            {"variant": "inference", "knobs": {}},
            selection._cache_identity(context),
            selection._identity(context),
        )
        state.graph_captured = True
        assert not selection._pending(state, context)
    with autotune_v2(skip_ops={NATIVE_NAMESPACE}, persistent_cache=False):
        assert selection._pending(state, selection._context())
        assert protocol.backend.coordinate_workspace_replacement(protocol.owner)


@pytest.mark.parametrize("local_captured", [False, True])
def test_captured_fallback_survives_another_replay_context(
    protocol, monkeypatch, local_captured
):
    protocol, _ = protocol
    owner, backend = protocol.owner, protocol.backend
    with autotune_v2(mode="replay", persistent_cache=False):
        context = selection._context()
        owner._tuning_state.record_selection(
            None, selection._cache_identity(context), None
        )
        owner._tuning_state.lookup_scope = (
            context.scope,
            selection._cache_identity(context),
        )
        owner._tuning_state.graph_captured = local_captured

    # A peer's captured storage also prevents a rank-local cache retry.
    monkeypatch.setattr(
        backend, "_gather", lambda value: [value, (*value[:2], True, value[3])]
    )
    choose = Mock(side_effect=AssertionError("captured fallback must not select"))
    monkeypatch.setattr(AutoTuner.get(), "choose_with_protocol", choose)
    with autotune_v2(mode="replay", persistent_cache=False):
        assert selection._pending(owner._tuning_state, selection._context())
        backend.compute(owner, (None, None), output=None)
    owner.kernel.compute.assert_called_once()
    with (
        autotune_v2(persistent_cache=False),
        pytest.raises(RuntimeError, match="another explicit workspace"),
    ):
        backend.compute(owner, (None, None), output=None)
    choose.assert_not_called()


def test_collective_cache_hits_reconcile_different_winners(protocol, monkeypatch):
    protocol, contract = protocol
    first = {
        "contract": contract,
        "variant": "genphase",
        "knobs": {"cluster_shape_mn": [4, 1], "fc2_use_bulk": True},
    }
    other = {"contract": contract, "variant": "inference", "knobs": {}}
    with autotune_v2(mode="replay", persistent_cache=False):
        context = selection._context()
        controls = (False, False, False, False, False, selection._identity(context)[1])
        gather = Mock(
            side_effect=[
                [controls, controls],
                [contract, contract],
                [first, other],
                [[True, True], [True, True]],
                [None, None],
            ]
        )
        monkeypatch.setattr(protocol.backend, "_gather", gather)
        selected = AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol)
    assert selected["variant"] == "genphase"
    assert selected["knobs"]["cluster_shape_mn"] == (4, 1)
    assert gather.call_count == 5


def test_install_prepares_independent_storage_before_releasing_previous(
    protocol, monkeypatch
):
    protocol, _ = protocol
    # Exercise the real installer; the cache-only fixture normally replaces it.
    monkeypatch.delattr(protocol, "_install")
    events = []
    previous = protocol.owner.kernel
    previous.destroy.side_effect = lambda _: events.append("release")
    raw = SimpleNamespace(prepare=lambda *weights: events.append("prepare"))
    selected = Mock()
    selected._allocate_workspace.return_value = raw
    monkeypatch.setattr(protocol.backend, "_backend", lambda *args: selected)
    protocol._install({"variant": "genphase", "knobs": {}})
    assert events == ["prepare", "release"]
    selected.prepare_workspace.assert_not_called()
    assert protocol.owner.workspace is raw


@pytest.mark.parametrize("output_view", [False, True])
def test_full_forward_sweep_selects_across_all_knobs(
    protocol, monkeypatch, output_view
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        autotune as native,
        correctness,
        tuning_measurement,
        tuning_protocol,
    )

    protocol, contract = protocol
    contract["native_skipped"] = False
    backend = protocol.backend
    configs = {
        variant: backend._backend(variant)._resolved_config(protocol.owner.fleet)
        for variant in contract["variants"]
    }
    grid = [{"epi_flag_batches": (flag, 4)} for flag in (1, 2)]
    measured = []
    events = []
    buffers = []
    output = torch.empty(4, 1024)[:, ::2]
    protocol.output = output if output_view else None

    class Buffer:
        _build_kernel = staticmethod(lambda cfg: None)

        def __init__(self, cfg):
            self.config, self.device = cfg, torch.device("cpu")
            self.x = torch.zeros(4, 512)
            self.x_sf = torch.zeros(4, 16)
            self.topk_idx = torch.zeros(4, 3, dtype=torch.int32)
            self.topk_weights = torch.ones(4, 3)
            self.output_activation = torch.zeros(4, 512)
            self.prepared = False
            self.destroyed = False
            buffers.append(self)

        def note_staged_tokens(self, count):
            self.count = count

        def prepare(self, *weights):
            self.prepared = True

        def destroy(self):
            self.destroyed = True

    class Kernel:
        def __init__(self, variant, knobs="auto"):
            self.cfg = dataclasses.replace(
                configs[variant], **({} if knobs == "auto" else knobs)
            )

        def _allocate_workspace(self, fleet):
            return Buffer(self.cfg)

        def stage_inputs(self, tensors, workspace, *, quantize_input):
            events.append("stage")
            workspace.note_staged_tokens(tensors.num_tokens)

        def compute(self, workspace, weights, *, output):
            assert self.cfg == workspace.config
            assert workspace.prepared
            assert events[-1] == "stage"
            events.append("compute")
            workspace.output_activation.fill_(1)
            if output is not None:
                output.copy_(workspace.output_activation)
                events.append("copy")

        def destroy(self, workspace):
            workspace.destroy()

    def measure(trial, weights, repeats, settings, group):
        trial.launch(*weights)
        expected_events = ["stage", "compute"] + (["copy"] if output_view else [])
        assert events[-len(expected_events) :] == expected_events
        key = (trial.config.kernel_variant, trial.config.epi_flag_batches[0])
        measured.append(key)
        # GenPhase's second candidate wins the full-forward objective. A cached
        # kernel-only winner for its first candidate must not narrow the sweep.
        return [
            {
                ("inference", 1): 9,
                ("inference", 2): 6,
                ("genphase", 1): 8,
                ("genphase", 2): 3,
            }[key]
        ] * repeats

    monkeypatch.setattr(backend, "_backend", Kernel)
    monkeypatch.setattr(block_scaled, "Sm107BlockScaledSymmBuffer", Buffer)
    monkeypatch.setattr(native, "sm107_candidates", lambda *args, **kwargs: grid)
    monkeypatch.setattr(correctness, "sampled_reference", lambda *a, **kw: (None, None))
    monkeypatch.setattr(
        correctness,
        "output_error",
        lambda actual, *a: 0 if torch.all(actual == 1) else 1,
    )
    monkeypatch.setattr(tuning_measurement, "measure_samples", measure)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **kw: tensor(*a, **{**kw, "device": "cpu"})
    )
    with autotune_v2(persistent_cache=False):
        native_context = AutoTuner.get().protocol_context(tuning_protocol.NAMESPACE)
        cached = {"knobs": grid[0], "p50_us": 1}
        native_context.publish("kernel-only", cached)
        previous = dict(native_context.records())
        result = AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol)
        assert dict(native_context.records()) == previous
    assert measured == [(v, flag) for v in ("inference", "genphase") for flag in (1, 2)]
    assert result["variant"] == "genphase"
    assert result["knobs"] == grid[1]
    assert result["knob_record"]["timing_boundary"] == "complete_forward"
    assert all(buffer.destroyed for buffer in buffers)
