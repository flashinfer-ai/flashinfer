"""Complete-path selection, cache agreement and prepared-plan ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer import autotune_v2
from flashinfer.autotuner import AutoTuner
from flashinfer.moe_ep import AutoConfig, MegaConfig, SplitConfig
from flashinfer.moe_ep.core.selection import (
    BACKEND_NAMESPACE,
    Collective,
    plain,
    record_key,
)
from flashinfer.moe_ep.core.tuning_state import AutoTuningState, prepared_for_serving
from flashinfer.moe_ep.modes.auto_layer import MoEEpAutoWorkspace
from flashinfer.moe_ep.modes.selection import SplitProtocol


class LocalCollective(Collective):
    def __init__(self):
        self.rank, self.world = 0, 1
        self.measured = []

    def measure(self, fn, device, settings):
        result = fn()
        self.measured.append(result[0])
        return result[1]


class SplitTrial:
    def __init__(self):
        self.selected = None
        self.prepared = []
        self.scores = {(0, 10): 8.0, (0, 11): 3.0, (2, 20): 5.0}

    def enumerate(self):
        return [
            dict(
                runner=0,
                backend="first",
                implementation="FirstRunner",
                extras=(),
                tactics=[10, 11],
            ),
            dict(
                runner=2,
                backend="second",
                implementation="SecondRunner",
                extras=(),
                tactics=[20],
            ),
        ]

    def backend_config(self, index):
        return {"backend": index}

    def prepare(self, selection):
        self.selected = tuple(selection)
        self.prepared.append(self.selected)

    def forward(self, tensors):
        return self.selected, self.scores[self.selected]


@pytest.mark.parametrize("forced", [None, 0, 1])
def test_explicit_candidates(forced):
    config = AutoConfig((MegaConfig(object()), SplitConfig()), candidate=forced)
    assert config.candidate == forced
    with pytest.raises(ValueError):
        AutoConfig(config.candidates, candidate=2)
    with pytest.raises(ValueError):
        AutoConfig(())


def test_full_path_tactics_and_partial_cache_reuse(monkeypatch):
    tuner = AutoTuner.get()
    monkeypatch.setattr(
        tuner, "choose_one", Mock(side_effect=AssertionError("compute-only selection"))
    )
    c, plan = LocalCollective(), SplitTrial()
    t = SimpleNamespace(hidden_states=torch.empty(1))
    protocol = SplitProtocol(plan, t, c, {"capacity": 32}, {})
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        first = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert first["selection"] == [0, 11]
        assert c.measured == [(0, 10), (0, 11), (2, 20)]
        assert len(tuner.get_protocol_records(BACKEND_NAMESPACE)) == 2
        c.measured.clear()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert c.measured == [(0, 11), (2, 20)]
        # A new backend must search its own tactics even with a cached peer.
        old = plan.enumerate
        plan.enumerate = lambda: old() + [
            dict(
                runner=3,
                backend="third",
                implementation="ThirdRunner",
                extras=(),
                tactics=[30, 31],
            )
        ]
        plan.scores.update({(3, 30): 9.0, (3, 31): 2.0})
        c.measured.clear()
        third = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert c.measured == [(0, 11), (2, 20), (3, 30), (3, 31)]
        assert third["selection"] == [3, 31]
        c.measured.clear()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol, force_tune=True)
        assert len(c.measured) == 5
        # Capacity and objective changes cannot reuse the previous backend records.
        protocol.contract = {"capacity": 64}
        c.measured.clear()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert len(c.measured) == 5


def test_disabled_and_skipped_do_not_measure():
    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    t = SimpleNamespace(hidden_states=torch.empty(1))
    protocol = SplitProtocol(plan, t, c, {"capacity": 32}, {})
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
    c.measured.clear()
    with autotune_v2(mode="replay", persistent_cache=False):
        result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert result["selection"] == [0, 11]
        assert not c.measured
    with autotune_v2(persistent_cache=False, skip_ops={BACKEND_NAMESPACE}):
        result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        assert result["selection"] == [0, -1]
        assert not c.measured


def test_collective_cache_hit_requires_one_common_valid_winner():
    c = LocalCollective()
    context = SimpleNamespace(lookup=lambda key: "rank0")
    responses = iter([["rank0", "rank1"], [[False, True], [True, True]]])
    c.gather = lambda value: next(responses)
    assert c.lookup(context, "key", lambda r: r == "rank1") == "rank1"
    responses = iter([["rank0", "rank1"], [[True, False], [False, True]]])
    assert c.lookup(context, "key", lambda r: True) is None


def test_captured_plans_survive_replacement_until_handle_destruction():
    owner = Mock()
    ws = MoEEpAutoWorkspace(owner, None, [])
    first, second, third = Mock(), Mock(), Mock()
    ws._install(first, {})
    ws._state.graph_captured = True
    ws._install(second, {})
    first.destroy.assert_not_called()
    ws._install(third, {})
    second.destroy.assert_called_once()
    ws.destroy()
    first.destroy.assert_called_once()
    third.destroy.assert_called_once()
    ws.destroy()
    first.destroy.assert_called_once()


def test_bare_execution_generation_does_not_consult_cache_identity():
    class Context:
        scope = None
        force_tune = False
        execution_generation = 5

        @property
        def cache_identity(self):
            raise AssertionError("bare serving must not inspect cache entries")

    assert prepared_for_serving(AutoTuningState(serving_generation=5), Context())
    assert not prepared_for_serving(AutoTuningState(serving_generation=4), Context())


def test_record_contract_has_no_object_addresses():
    config = AutoConfig((SplitConfig(),))
    assert record_key(config) == record_key(AutoConfig((SplitConfig(),)))
    assert "0x" not in str(plain(config))


def test_rank_local_expert_offset_does_not_change_common_configuration():
    import dataclasses
    from flashinfer.moe_ep import FusedMoeKernelConfig
    from flashinfer.moe_ep.modes.auto_layer import _common_candidate
    from tests.moe_ep.test_fused_moe_weights import _nvfp4_moe_config

    cfg = _nvfp4_moe_config(
        num_experts=8, local_num_experts=4, offset=0, intermediate=256
    )
    first = SplitConfig(kernel=FusedMoeKernelConfig(cfg))
    second = dataclasses.replace(
        first,
        kernel=dataclasses.replace(
            first.kernel,
            moe_config=dataclasses.replace(
                cfg, experts=dataclasses.replace(cfg.experts, local_expert_offset=4)
            ),
        ),
    )
    assert _common_candidate(first) == _common_candidate(second)
    second.kernel.moe_config = dataclasses.replace(
        second.kernel.moe_config,
        activation=dataclasses.replace(cfg.activation, alpha=2.0),
    )
    assert _common_candidate(first) != _common_candidate(second)


@pytest.mark.parametrize(
    "local_rank,peer_tactics,measured",
    [(0, [11], [(0, 11)]), (1, [11, 10, 99], [(0, 11), (0, 10)])],
)
def test_split_intersects_tactics_and_backend_eligibility_across_ranks(
    local_rank, peer_tactics, measured
):
    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    c.world, c.rank = 2, local_rank
    peer = [
        dict(
            runner=0,
            backend="first",
            implementation="FirstRunner",
            extras=(),
            tactics=peer_tactics,
        )
    ]

    def gather(value):
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "tactics" in value[0]
        ):
            return [peer, value] if local_rank else [value, peer]
        return [value, value]

    c.gather = gather
    protocol = SplitProtocol(
        plan,
        SimpleNamespace(hidden_states=torch.empty(1)),
        c,
        {"eligibility": "intersection"},
        {},
    )
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
    assert result["selection"] == [0, 11]
    assert c.measured == measured


def test_outer_selects_full_plan_and_replays_without_measurement():
    from flashinfer.moe_ep import FleetParams
    from flashinfer.moe_ep.core.selection import NAMESPACE
    from flashinfer.moe_ep.modes.auto_layer import _OuterProtocol
    from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer

    class Plan:
        def __init__(self, index):
            self.index = index
            self.destroy = Mock()

        def selection(self):
            return {"kernel_variant": "inference", "knobs": {"tile": self.index}}

        def forward(self, tensors):
            return self.index, [9.0, 4.0][self.index]

    tuner, c = AutoTuner.get(), LocalCollective()
    owner = Mock()
    ws = MoEEpAutoWorkspace(owner, FleetParams(8, 32, 256), [])
    layer = SimpleNamespace(
        _collective=c,
        _config_contract=[[{}, {}]],
        _config=SimpleNamespace(candidate=None),
        _prototypes={0: Mock(spec=MoEEpMegaLayer), 1: Mock(spec=MoEEpMegaLayer)},
        _fleet_knobs=[[]],
        _environment=[{}],
        _mega_revisions=[{}],
        _failed_plans=[],
    )
    for prototype in layer._prototypes.values():
        prototype._kernel = Mock()
    protocol = _OuterProtocol(layer, ws, SimpleNamespace(hidden_states=torch.empty(1)))
    calls = []
    force_flags = []

    def prepare(index, *, selected=None, fallback=False, force_tune=False):
        calls.append((index, selected))
        force_flags.append(force_tune)
        return Plan(index)

    protocol._new_plan = prepare
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        result = tuner.choose_with_protocol(NAMESPACE, protocol)
        assert result["candidate"] == 1
        assert c.measured == [0, 1]
        c.measured.clear()
        calls.clear()
        result = tuner.choose_with_protocol(NAMESPACE, protocol)
        assert result["candidate"] == 1
        assert not c.measured
        assert calls == [(1, {"kernel_variant": "inference", "knobs": {"tile": 1}})]
        # An invalid record must not be installed as a fixed plan.
        key, record = next(iter(tuner.get_protocol_records(NAMESPACE).items()))
        record["selected"] = {"wrong": True}
        tuner.publish_protocol_record(NAMESPACE, key, record, persist=False)
        tuner.choose_with_protocol(NAMESPACE, protocol)
        assert c.measured == [0, 1]
        c.measured.clear()
        force_flags.clear()
        tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
        assert c.measured == [0, 1]
        assert force_flags == [True, True, False]
        c.measure = Mock(side_effect=RuntimeError("launch failure"))
        with pytest.raises(RuntimeError, match="launch failure"):
            tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
        assert len(layer._failed_plans) == 1
        layer._failed_plans[0].destroy.assert_not_called()
    ws.destroy()


def test_split_enumeration_normalizes_foreign_tactic_arrays():
    from flashinfer.fused_moe.layer import MoELayer

    class Array:
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    runner = Mock(
        backend_key="ffi",
        pack_inputs=Mock(return_value=[torch.empty(1)]),
        get_valid_tactics=Mock(return_value=Array([Array([8, 124])])),
        get_cache_key_extras=Mock(return_value=()),
    )
    layer = object.__new__(MoELayer)
    layer.runners = [runner]
    layer._tuning_runner_ids = [3]
    layer._eligible_runners = Mock(return_value=(0,))
    layer.tuner = Mock()
    layer.tuner._blocklist.filter.side_effect = lambda op, r, tactics: tactics
    candidates = layer._tuning_candidates(None, None)
    assert candidates[0]["runner"] == 3
    assert candidates[0]["tactics"] == [(8, 124)]
    assert plain(candidates)[0]["tactics"] == [[8, 124]]


def test_compute_op_skip_only_times_its_fallback():
    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    plan.scores[(0, -1)] = 1.0
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"skip": "compute"}, {}
    )
    with autotune_v2(persistent_cache=False, skip_ops={"moe_first"}):
        tuner.clear_cache()
        result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
    assert result["selection"] == [0, -1]
    assert c.measured == [(0, -1), (2, 20)]


def test_optional_preparation_does_not_swallow_runtime_failures():
    c = LocalCollective()

    def unsupported():
        raise ValueError("unsupported capacity")

    def failed():
        raise RuntimeError("device failure")

    assert c.prepare(unsupported, optional=True) is None
    with pytest.raises(RuntimeError, match="device failure"):
        c.prepare(failed, optional=True)


def test_failed_split_initialization_releases_its_runtime(monkeypatch):
    from flashinfer.moe_ep import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.modes import split_layer

    kernel, runtime = Mock(), Mock()
    monkeypatch.setattr(
        split_layer, "ensure_moe_ep_cuda_device", lambda bootstrap: None
    )
    monkeypatch.setattr(split_layer, "create_split_kernel", lambda config: kernel)
    monkeypatch.setattr(split_layer, "bootstrap_moe_ep_runtime", lambda *args: runtime)
    release = Mock()
    monkeypatch.setattr(split_layer, "finalize_moe_ep_runtime", release)

    def invalid(self, **kwargs):
        raise ValueError("invalid transport geometry")

    monkeypatch.setattr(split_layer.MoEEpSplitLayer, "_validate_at_init", invalid)
    with pytest.raises(ValueError, match="invalid transport geometry"):
        split_layer.MoEEpSplitLayer(BootstrapConfig(1, 0), FleetParams(2, 4, 8), Mock())
    release.assert_called_once_with(runtime)
    kernel.destroy.assert_called_once()


def test_workspace_rejects_inputs_on_another_device():
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.modes.auto_layer import MoEEpAutoLayer

    layer = SimpleNamespace(_device=torch.device("cuda", 0))
    t = MoEEpTensors(
        torch.empty(1, 8), torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1)
    )
    with pytest.raises(ValueError, match="layer's CUDA device"):
        MoEEpAutoLayer._validate_device(layer, t)


def test_rubin_weight_preparation_converts_only_the_fc1_order():
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm107.automatic import ep_weight_pack

    backend = SimpleNamespace(_kernel_config=SimpleNamespace(intermediate_size=2))
    weights = MoEWeightPack(
        torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4),
        torch.empty(1, 4, 2, dtype=torch.bfloat16),
    )
    native = ep_weight_pack(backend, weights)
    torch.testing.assert_close(native.w13[:, :2], weights.w13[:, 2:])
    torch.testing.assert_close(native.w13[:, 2:], weights.w13[:, :2])
    assert native.w2 is weights.w2


@pytest.mark.parametrize("peer_failure", [False, True])
def test_fixed_mega_preparation_is_coordinated_before_launch(monkeypatch, peer_failure):
    from flashinfer.moe_ep.modes import selection

    events = []
    collective = LocalCollective()
    collective.gather = lambda value: [
        value,
        ("RuntimeError: peer compilation failed", False) if peer_failure else value,
    ]
    monkeypatch.setattr(selection, "Collective", lambda bootstrap: collective)
    plan = object.__new__(selection.MegaPlan)
    plan.output = object()
    plan.layer = SimpleNamespace(
        _bootstrap=object(),
        _workspace=object(),
        _megakernel_config=SimpleNamespace(knobs={}),
        _get_transformed_weights=lambda: (),
        _kernel=SimpleNamespace(ep_prepare_fixed=lambda *a: events.append("compile")),
        stage_inputs=lambda t: events.append("stage"),
        compute_staged=lambda **kwargs: events.append("launch"),
    )
    tensors = SimpleNamespace(output=None)
    if peer_failure:
        with pytest.raises(RuntimeError, match="peer compilation failed"):
            plan.prepare(tensors)
        assert events == ["stage", "compile"]
    else:
        plan.prepare(tensors)
        assert events == ["stage", "compile", "launch"]


def test_skipping_outer_tuning_preserves_explicit_mega_knobs(monkeypatch):
    from flashinfer.moe_ep import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.modes import selection

    kernel = SimpleNamespace(knobs={"explicit": 1})
    prototype = SimpleNamespace(
        _mega_config=MegaConfig(kernel),
        _kernel=Mock(),
        _bootstrap=BootstrapConfig(1, 0),
        _get_transformed_weights=lambda: (),
    )
    construct = Mock()
    monkeypatch.setattr(selection, "MoEEpMegaLayer", construct)
    selection.MegaPlan(prototype, FleetParams(2, 4, 8), None, fallback=True)
    assert construct.call_args.args[3].megakernel is kernel
    prototype._kernel.ep_tuning_fallback.assert_not_called()


def test_split_releases_received_views_before_transport_teardown():
    import weakref
    from flashinfer.moe_ep.modes.selection import SplitPlan

    class Context:
        pass

    plan = object.__new__(SplitPlan)
    plan.context = Context()
    borrowed = weakref.ref(plan.context)
    plan.choices = []
    plan.state = object()

    def teardown():
        assert borrowed() is None

    plan.layer = SimpleNamespace(destroy=teardown)
    plan.destroy()
    assert plan.state is None
