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
        fleet,
        Mock(),
        raw,
        tensors=moe_ep.MoEEpTensors(
            torch.empty(4, 512), torch.empty(4, 3, dtype=torch.int32), torch.empty(4, 3)
        ),
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


@pytest.mark.parametrize("phase", ["selection", "settled", "capture"])
def test_compute_releases_staged_tensors(protocol, monkeypatch, phase):
    protocol, _ = protocol
    backend, owner = protocol.backend, protocol.owner
    tensors = owner.tensors
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: phase == "capture"
    )

    def choose(namespace, pending):
        assert pending.tensors is tensors
        assert owner.tensors is tensors

    choose = Mock(side_effect=choose)
    monkeypatch.setattr(AutoTuner.get(), "choose_with_protocol", choose)
    with autotune_v2(mode="replay", persistent_cache=False):
        if phase == "settled":
            owner._tuning_state.record_selection(
                {"variant": "inference", "knobs": {}},
                selection._cache_identity(selection._context()),
                None,
            )
        backend.compute(owner, (None, None), output=None)
    assert owner.tensors is None
    assert choose.call_count == (phase == "selection")
    if phase == "selection":
        owner.kernel.stage_inputs.assert_called_once_with(
            tensors, owner.workspace, quantize_input=True
        )
    owner.kernel.compute.assert_called_once()


@pytest.mark.parametrize("winner", [None, {"variant": "inference", "knobs": {}}])
def test_prepared_eager_execution_does_not_poll_cache_or_peers(
    protocol, monkeypatch, winner
):
    from tests.moe_ep.sm107_test_utils import unrelated_autotune_contexts

    protocol, _ = protocol
    backend, owner = protocol.backend, protocol.owner
    state = owner._tuning_state
    context = selection._context()
    state.record_selection(winner, selection._cache_identity(context), None)
    state.lookup_scope = (context.scope, selection._cache_identity(context))
    backend.compute(owner, (None, None), output=None)

    def unexpected(*args, **kwargs):
        raise AssertionError("prepared execution must not poll tuning state")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "_gather", unexpected)
        patch.setattr(selection, "_cache_identity", unexpected)
        # Rank-local invalidation is reconciled in an explicit context.
        AutoTuner.get().clear_cache()
        with unrelated_autotune_contexts():
            for _ in range(3):
                assert not backend.coordinate_workspace_replacement(owner)
                backend.compute(owner, (None, None), output=None)
                assert owner.tensors is None
        backend.compute(owner, (None, None), output=None)
    with autotune_v2(mode="replay", persistent_cache=False):
        assert not selection.prepared_for_serving(state, selection._context())
        assert selection._pending(state, selection._context())
    assert selection.prepared_for_serving(state, selection._context())
    # An actual scoped forward requires reconciliation after leaving that scope.
    owner.tensors = protocol.tensors
    monkeypatch.setattr(AutoTuner.get(), "choose_with_protocol", Mock())
    with autotune_v2(mode="replay", persistent_cache=False):
        backend.compute(owner, (None, None), output=None)
    assert not selection.prepared_for_serving(state, selection._context())


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


@pytest.fixture
def forward_sweep(protocol, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        autotune as native,
        correctness,
        tuning_measurement,
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
    state = SimpleNamespace(
        protocol=protocol,
        contract=contract,
        grid=grid,
        measured=measured,
        buffers=buffers,
        failure=None,
        rejected=set(),
        scores={
            ("inference", 1): 9,
            ("inference", 2): 6,
            ("genphase", 1): 8,
            ("genphase", 2): 3,
            ("genphase", 4): 7,
        },
    )

    class Buffer:
        _build_kernel = staticmethod(lambda cfg: None)

        def __init__(self, cfg):
            self.config, self.device = cfg, torch.device("cpu")
            live = protocol.tensors.num_tokens
            self.x = torch.zeros(live, 512)
            self.x_sf = torch.zeros(live, 16)
            self.topk_idx = torch.zeros(live, 3, dtype=torch.int32)
            self.topk_weights = torch.ones(live, 3)
            self.output_activation = torch.zeros(live, 512)
            self.prepared = False
            self.destroyed = False
            self._tuning_state = AutoTuningState()
            buffers.append(self)

        def note_staged_tokens(self, count):
            self.count = count

        def staged_tokens(self):
            return self.count

        def prepare(self, *weights):
            self.prepared = True

        def destroy(self):
            self.destroyed = True

    class Kernel:
        def __init__(self, variant, knobs="auto"):
            if not isinstance(variant, str):
                knobs, variant = variant.knobs, variant.kernel_variant
            self._kernel_config = dataclasses.replace(
                backend._config, kernel_variant=variant, knobs=knobs
            )
            self._ep_bootstrap = backend._ep_bootstrap
            self.cfg = dataclasses.replace(
                configs[variant], **({} if knobs == "auto" else knobs)
            )

        def bind_ep_bootstrap(self, bootstrap):
            self._ep_bootstrap = bootstrap

        def _allocate_workspace(self, fleet):
            return Buffer(self._resolved_config(fleet))

        def _resolved_config(self, fleet):
            return dataclasses.replace(
                self.cfg, max_tokens_per_rank=fleet.max_tokens_per_rank
            )

        def stage_inputs(self, tensors, workspace, *, quantize_input):
            events.append("stage")
            workspace.note_staged_tokens(tensors.num_tokens)

        def compute(self, workspace, weights, *, output):
            assert (
                dataclasses.replace(
                    self.cfg, max_tokens_per_rank=workspace.config.max_tokens_per_rank
                )
                == workspace.config
            )
            assert workspace.prepared
            assert events[-1] == "stage"
            if state.failure == "launch":
                raise RuntimeError("candidate launch failed")
            events.append("compute")
            rejected = (
                state.failure in (self.cfg.kernel_variant, "all")
                or (self.cfg.kernel_variant, self.cfg.epi_flag_batches[0])
                in state.rejected
            )
            workspace.output_activation.fill_(0 if rejected else 1)
            if output is not None:
                output.copy_(workspace.output_activation)
                events.append("copy")

        def destroy(self, workspace):
            workspace.destroy()

    def measure(trial, weights, repeats, settings, group):
        trial.launch(*weights)
        expected_events = ["stage", "compute"] + (
            ["copy"] if protocol.output is not None else []
        )
        assert events[-len(expected_events) :] == expected_events
        key = (trial.config.kernel_variant, trial.config.epi_flag_batches[0])
        measured.append(key)
        return [state.scores[key]] * repeats

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
    protocol.owner.kernel = Kernel("inference")
    return state


@pytest.mark.parametrize(
    "variant,output_view,failure",
    [
        (variant, view, None)
        for variant in (None, "inference", "genphase")
        for view in (False, True)
    ]
    + [
        (None, False, "inference"),
        (None, False, "genphase"),
        (None, False, "all"),
        (None, False, "launch"),
        ("inference", False, "inference"),
    ],
)
def test_full_forward_sweep_selects_across_all_knobs(
    forward_sweep, output_view, variant, failure
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        autotune as native,
        tuning_protocol,
    )

    protocol, contract = forward_sweep.protocol, forward_sweep.contract
    backend, grid = protocol.backend, forward_sweep.grid
    measured, buffers = forward_sweep.measured, forward_sweep.buffers
    forward_sweep.failure = failure
    protocol.output = torch.empty(4, 1024)[:, ::2] if output_view else None
    with autotune_v2(persistent_cache=False):
        native_context = AutoTuner.get().protocol_context(tuning_protocol.NAMESPACE)
        cached = {"knobs": grid[0], "p50_us": 1}
        native_context.publish("kernel-only", cached)
        previous = dict(native_context.records())
        if variant is None:
            if failure in ("all", "launch"):
                error, message = (
                    (selection.MoEEpConfigError, "no Rubin MegaMoE variant")
                    if failure == "all"
                    else (RuntimeError, "candidate launch failed")
                )
                with pytest.raises(error, match=message):
                    AutoTuner.get().choose_with_protocol(selection.NAMESPACE, protocol)
                protocol._install.assert_not_called()
            else:
                result = AutoTuner.get().choose_with_protocol(
                    selection.NAMESPACE, protocol
                )
                assert result["variant"] == (
                    "inference" if failure == "genphase" else "genphase"
                )
                assert result["knobs"] == grid[1]
                assert result["knob_record"]["timing_boundary"] == "complete_forward"
            records = native_context.records()
            assert records["kernel-only"] == previous["kernel-only"]
            winners = [v for k, v in records.items() if k != "kernel-only"]
            passing = [
                v for v in contract["variants"] if failure not in (v, "all", "launch")
            ]
            assert {v["kernel_variant"] for v in winners} == set(passing)
            assert all(tuple(v["knobs"]["epi_flag_batches"]) == (2, 4) for v in winners)
        else:
            from flashinfer.moe_ep.backends.mega.kernel.sm107.forward_tuning import (
                ForwardExecution,
            )

            kernel = backend._backend(variant)
            workspace = kernel._allocate_workspace(protocol.owner.fleet)
            kernel.stage_inputs(protocol.tensors, workspace, quantize_input=True)
            execution = ForwardExecution(
                kernel, protocol.tensors, quantize_input=True, output=protocol.output
            )
            if failure is not None:
                with pytest.raises(
                    native.NoPassingCandidate, match="no SM107 candidate"
                ):
                    execution.tune(
                        None, None, workspace, candidates=grid, force_tune=True
                    )
            else:
                assert (
                    execution.tune(
                        None, None, workspace, candidates=grid, force_tune=True
                    )
                    == grid[1]
                )
            workspace.destroy()
    variants = ("inference", "genphase") if variant is None else (variant,)
    variants = [v for v in variants if failure not in (v, "all", "launch")]
    assert measured == [(v, flag) for v in variants for flag in (1, 2)]
    if failure != "launch":
        assert all(buffer.destroyed for buffer in buffers)
    else:
        assert not any(buffer.destroyed for buffer in buffers)


def _fixed_forward_tune(protocol, variant, candidates):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.forward_tuning import (
        ForwardExecution,
    )

    kernel = protocol.backend._backend(variant)
    workspace = kernel._allocate_workspace(protocol.owner.fleet)
    kernel.stage_inputs(protocol.tensors, workspace, quantize_input=True)
    execution = ForwardExecution(
        kernel,
        protocol.tensors,
        quantize_input=protocol.owner.quantize_input,
        output=protocol.output,
    )
    knobs = execution.tune(None, None, workspace, candidates=candidates)
    workspace.destroy()
    return knobs


@pytest.mark.parametrize("seed,live", [("outer", 128), ("fixed", 128), ("outer", 4)])
def test_persisted_inner_winners_reused_at_new_capacity(
    forward_sweep, monkeypatch, seed, live
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.tuning_protocol import (
        NAMESPACE,
    )

    state, tuner = forward_sweep, AutoTuner.get()
    protocol = state.protocol
    monkeypatch.delattr(protocol, "_contract")
    protocol.tensors = moe_ep.MoEEpTensors(
        torch.empty(live, 512),
        torch.empty(live, 3, dtype=torch.int32),
        torch.empty(live, 3),
    )
    with autotune_v2():
        if seed == "outer":
            result = tuner.choose_with_protocol(selection.NAMESPACE, protocol)
            assert result["variant"] == "genphase"
        else:
            for variant in ("inference", "genphase"):
                assert (
                    _fixed_forward_tune(protocol, variant, state.grid) == state.grid[1]
                )
        records = tuner.get_protocol_records(NAMESPACE)
        assert len(records) == 2
        assert {r["kernel_variant"] for r in records.values()} == {
            "inference",
            "genphase",
        }
        assert len(knob_cache._load_entries(knob_cache.knob_cache_path())) == 2

    # Reload only the managed files. The compatibility file cannot hide a miss.
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    tuner.clear_cache()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    state.measured.clear()
    with autotune_v2(mode="replay"):
        for variant in ("inference", "genphase"):
            assert _fixed_forward_tune(protocol, variant, state.grid) == state.grid[1]
    assert not state.measured

    protocol.owner.fleet = dataclasses.replace(
        protocol.owner.fleet, max_tokens_per_rank=256
    )
    protocol.owner._tuning_state = AutoTuningState()
    protocol.owner.workspace.config = dataclasses.replace(
        protocol.owner.workspace.config, max_tokens_per_rank=256
    )
    with autotune_v2(mode="replay"):
        assert tuner.choose_with_protocol(selection.NAMESPACE, protocol) is None
    assert not state.measured

    # Old scores favor GenPhase; the current complete forwards favor inference.
    state.scores[("inference", 2)] = 1
    with autotune_v2():
        result = tuner.choose_with_protocol(selection.NAMESPACE, protocol)
        assert result["variant"] == "inference"
        assert result["contract"]["capacity"] == 256
        assert result["p50_us"] == 1
        if live == 128:
            assert result["knob_record"]["max_tokens"] == 128
            assert result["knob_record"]["p50_us"] == 6
            assert tuner.get_protocol_records(NAMESPACE) == records
        else:
            assert result["knob_record"]["max_tokens"] == 256
            assert len(tuner.get_protocol_records(NAMESPACE)) == 4
    flags = (2,) if live == 128 else (1, 2)
    assert state.measured == [(v, f) for v in ("inference", "genphase") for f in flags]
    state.measured.clear()
    with autotune_v2(mode="replay"):
        assert (
            tuner.choose_with_protocol(selection.NAMESPACE, protocol)["variant"]
            == "inference"
        )
    assert not state.measured
    # A separately updated inner winner must not be overwritten by outer replay.
    with autotune_v2():
        context = tuner.protocol_context(NAMESPACE)
        key, updated = next(
            (k, r) for k, r in context.records().items() if r == result["knob_record"]
        )
        updated["knobs"] = {"epi_flag_batches": [1, 4]}
        context.publish(key, updated)
    with autotune_v2(mode="replay"):
        tuner.choose_with_protocol(selection.NAMESPACE, protocol)
        assert tuner.get_protocol_record(NAMESPACE, key) == updated
    assert not state.measured
    assert all(buffer.destroyed for buffer in state.buffers)


@pytest.mark.parametrize(
    "mode", ["rejected", "force", "force-skipped", "kernel-only", "skip"]
)
def test_inner_reuse_respects_rejection_and_tuning_controls(
    forward_sweep, monkeypatch, mode
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        autotune as native,
        tuning_protocol,
    )

    state, tuner = forward_sweep, AutoTuner.get()
    protocol = state.protocol
    monkeypatch.delattr(protocol, "_contract")
    protocol.tensors = moe_ep.MoEEpTensors(
        torch.empty(128, 512),
        torch.empty(128, 3, dtype=torch.int32),
        torch.empty(128, 3),
    )
    with autotune_v2(persistent_cache=False):
        tuner.choose_with_protocol(selection.NAMESPACE, protocol)
        records = tuner.get_protocol_records(tuning_protocol.NAMESPACE)
    protocol.owner.fleet = dataclasses.replace(
        protocol.owner.fleet, max_tokens_per_rank=256
    )
    protocol.owner._tuning_state = AutoTuningState()
    protocol.owner.workspace.config = dataclasses.replace(
        protocol.owner.workspace.config, max_tokens_per_rank=256
    )
    if mode == "rejected":
        state.rejected.add(("inference", 2))
    elif mode == "kernel-only":
        with autotune_v2(persistent_cache=False):
            context = tuner.protocol_context(tuning_protocol.NAMESPACE)
            for key, record in records.items():
                record["timing_boundary"] = "kernel"
                context.publish(key, record)
    calls = []
    sweep = native._sweep_sm107_block_scaled_mega_moe

    def counted(*args, **kwargs):
        calls.append((args[3].config.kernel_variant, len(kwargs["candidates"])))
        return sweep(*args, **kwargs)

    monkeypatch.setattr(native, "_sweep_sm107_block_scaled_mega_moe", counted)
    with autotune_v2(
        persistent_cache=False,
        skip_ops={tuning_protocol.NAMESPACE}
        if mode in ("skip", "force-skipped")
        else None,
    ):
        tuner.choose_with_protocol(
            selection.NAMESPACE, protocol, force_tune=mode.startswith("force")
        )
        current = tuner.get_protocol_records(tuning_protocol.NAMESPACE)
    if mode == "rejected":
        assert calls == [("inference", 1), ("inference", 2), ("genphase", 1)]
        new_records = [r for r in current.values() if r["max_tokens"] == 256]
        assert len(new_records) == 1
        assert new_records[0]["kernel_variant"] == "inference"
        assert tuple(new_records[0]["knobs"]["epi_flag_batches"]) == (1, 4)
    elif mode == "skip":
        assert calls == [("inference", 1), ("genphase", 1)]
        assert current == records
    else:
        assert calls == [("inference", 2), ("genphase", 2)]
    assert all(buffer.destroyed for buffer in state.buffers)
