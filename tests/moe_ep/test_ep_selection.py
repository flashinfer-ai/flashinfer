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
from flashinfer.moe_ep.core.tuning_measurement import CaptureUnsupported
from flashinfer.moe_ep.core.tuning_state import AutoTuningState, prepared_for_serving
from flashinfer.moe_ep.modes.auto_layer import MoEEpAutoWorkspace
from flashinfer.moe_ep.modes.selection import SplitProtocol


class LocalCollective(Collective):
    def __init__(self):
        self.rank, self.world = 0, 1
        self.measured = []

    def measure(self, fn, device, settings, *, warmup=None, warmups=3, repeats=10):
        for _ in range(warmups):
            (warmup or fn)()
        result = fn()
        self.measured.append(result[0])
        return result[1]


class SplitTrial:
    def __init__(self):
        self.selected = None
        self.prepared = []
        self.log = []
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

    def warmup(self, tensors):
        self.log.append(("warmup", self.selected))

    def forward(self, tensors):
        self.log.append(("sample", self.selected))
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
        assert plan.log == [
            event
            for selection in c.measured
            for event in [("warmup", selection)] * 3 + [("sample", selection)]
        ]
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


@pytest.mark.parametrize("failure", [CaptureUnsupported, RuntimeError])
def test_cached_split_tactic_capture_failure_retries_other_tactics(
    monkeypatch, failure
):
    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        c.measured.clear()
        attempted = []
        measure = c.measure

        def measure_with_failure(*args, **kwargs):
            attempted.append(plan.selected)
            if plan.selected == (0, 11):
                raise failure("cached tactic failed")
            return measure(*args, **kwargs)

        monkeypatch.setattr(c, "measure", measure_with_failure)
        if failure is RuntimeError:
            with pytest.raises(RuntimeError, match="cached tactic failed"):
                tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
            assert attempted == [(0, 11)]
        else:
            result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
            assert attempted == [(0, 11), (0, 10), (2, 20)]
            assert result["selection"] == [2, 20]
            records = tuner.get_protocol_records(BACKEND_NAMESPACE).values()
            assert sorted(r["selection"] for r in records) == [[0, 10], [2, 20]]


@pytest.mark.parametrize("all_runners", [False, True])
def test_capture_rejections_survive_replay_and_allow_fresh_search(
    monkeypatch, all_runners
):
    from flashinfer.moe_ep.core.selection import NoEligibleCandidate

    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    measure = c.measure

    def reject_runner(*args, **kwargs):
        if all_runners or plan.selected[0] == 0:
            raise CaptureUnsupported("runner cannot capture")
        return measure(*args, **kwargs)

    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        stale = tuner.get_protocol_records(BACKEND_NAMESPACE)
        monkeypatch.setattr(c, "measure", reject_runner)
        if all_runners:
            with pytest.raises(CaptureUnsupported):
                tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        else:
            assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)[
                "selection"
            ] == [2, 20]

    c.measured.clear()
    plan.prepared.clear()
    gather = c.gather

    def gather_stale_peer(value):
        if isinstance(value, dict) and "capture_rejected" in value:
            return [stale[record_key(value["contract"])], value]
        return gather(value)

    monkeypatch.setattr(c, "gather", gather_stale_peer)
    with autotune_v2(mode="replay", persistent_cache=False):
        if all_runners:
            with pytest.raises(CaptureUnsupported):
                tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
            assert not plan.prepared
        else:
            assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)[
                "selection"
            ] == [2, 20]
            assert plan.prepared == [(2, 20)]
        assert not c.measured
        # An existing outer winner must also consult the backend rejection.
        with pytest.raises(NoEligibleCandidate, match="failed capture"):
            protocol.check_replay(
                tuner.protocol_context(BACKEND_NAMESPACE), [0, 11], plan.enumerate()
            )
        # Capture failures do not exclude other workloads.
        protocol.contract = {"capacity": 64}
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            -1,
        ]
        protocol.contract = {"capacity": 32}

    monkeypatch.setattr(c, "gather", gather)
    monkeypatch.setattr(c, "measure", measure)
    with autotune_v2(persistent_cache=False):
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            11,
        ]
        assert c.measured == [(0, 10), (0, 11), (2, 20)]
        protocol.check_replay(
            tuner.protocol_context(BACKEND_NAMESPACE), [0, 11], plan.enumerate()
        )


@pytest.mark.parametrize("reject_first", [False, True])
def test_split_records_follow_backend_configuration_after_reordering(
    monkeypatch, reject_first
):
    from flashinfer.moe_ep.core.selection import NoEligibleCandidate

    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    measure = c.measure

    def reject(*args, **kwargs):
        if reject_first and plan.selected[0] == 0:
            raise CaptureUnsupported("first backend cannot capture")
        return measure(*args, **kwargs)

    monkeypatch.setattr(c, "measure", reject)
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
    choices = [dict(v, runner=2 - v["runner"]) for v in plan.enumerate()][::-1]
    monkeypatch.setattr(plan, "enumerate", lambda: choices)
    monkeypatch.setattr(plan, "backend_config", lambda index: {"backend": 2 - index})
    c.measured.clear()
    plan.prepared.clear()
    with autotune_v2(mode="replay", persistent_cache=False):
        result = tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        expected = [0, 20] if reject_first else [2, 11]
        assert result["selection"] == expected
        assert plan.prepared == [tuple(expected)]
        assert not c.measured
        if reject_first:
            context = tuner.protocol_context(BACKEND_NAMESPACE)
            with pytest.raises(NoEligibleCandidate, match="failed capture"):
                protocol.check_replay(context, [2, 11], choices)
            choices.pop(0)
            with pytest.raises(CaptureUnsupported):
                tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)


@pytest.mark.parametrize("malformed", [None, 42, "invalid", {"unexpected": []}])
def test_malformed_capture_rejection_record_is_ignored(malformed):
    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        for key, record in tuner.get_protocol_records(BACKEND_NAMESPACE).items():
            record["capture_rejected"] = malformed
            tuner.publish_protocol_record(BACKEND_NAMESPACE, key, record, persist=False)
    c.measured.clear()
    with autotune_v2(mode="replay", persistent_cache=False):
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            -1,
        ]
        protocol.check_replay(
            tuner.protocol_context(BACKEND_NAMESPACE), [0, 11], plan.enumerate()
        )
        assert not c.measured
    with autotune_v2(persistent_cache=False):
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            11,
        ]
        assert c.measured == [(0, 10), (0, 11), (2, 20)]


@pytest.mark.parametrize("error_stage", ["prepare", "measure"])
def test_capture_rejection_survives_a_later_trial_error(monkeypatch, error_stage):
    from flashinfer.moe_ep.core.selection import NoEligibleCandidate

    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        measure, prepare = c.measure, plan.prepare

        def fail_measure(*args, **kwargs):
            if plan.selected == (0, 11):
                raise CaptureUnsupported("cached tactic cannot capture")
            if plan.selected == (0, 10) and error_stage == "measure":
                raise RuntimeError("next trial failed")
            return measure(*args, **kwargs)

        def fail_prepare(selection):
            if tuple(selection) == (0, 10) and error_stage == "prepare":
                raise RuntimeError("next trial failed")
            return prepare(selection)

        monkeypatch.setattr(c, "measure", fail_measure)
        monkeypatch.setattr(plan, "prepare", fail_prepare)
        with pytest.raises(RuntimeError, match="next trial failed"):
            tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
    c.measured.clear()
    with autotune_v2(mode="replay", persistent_cache=False):
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            2,
            20,
        ]
        assert not c.measured
        with pytest.raises(NoEligibleCandidate, match="failed capture"):
            protocol.check_replay(
                tuner.protocol_context(BACKEND_NAMESPACE), [0, 11], plan.enumerate()
            )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("persistent", [False, True])
def test_capture_recovery_updates_each_ranks_store(tmp_path, rank, persistent):
    from flashinfer.autotune_cache import autotune_v2_reload

    tuner, c, plan = AutoTuner.get(), LocalCollective(), SplitTrial()
    protocol = SplitProtocol(
        plan, SimpleNamespace(hidden_states=torch.empty(1)), c, {"capacity": 32}, {}
    )
    options = dict(
        cache_root=str(tmp_path), cache_scope="context", persistent_cache=persistent
    )
    with autotune_v2(**options):
        tuner.clear_cache()
        tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)
        c.rank = rank
        context = tuner.protocol_context(BACKEND_NAMESPACE)
        for key, record in context.records().items():
            context.publish(
                key, {"contract": record["contract"], "capture_rejected": [10, 11, 20]}
            )
        c.measured.clear()
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            11,
        ]
        assert c.measured == [(0, 10), (0, 11), (2, 20)]
    if persistent:
        autotune_v2_reload()
    else:
        assert not list(tmp_path.rglob("*.json"))
    c.measured.clear()
    with autotune_v2(mode="replay", **options):
        assert tuner.choose_with_protocol(BACKEND_NAMESPACE, protocol)["selection"] == [
            0,
            11,
        ]
        assert not c.measured
        for record in tuner.get_protocol_records(BACKEND_NAMESPACE).values():
            assert record["selection"][1] not in record["capture_rejected"]


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


def test_teardown_releases_other_plans_and_runtime_after_a_failure(monkeypatch):
    from flashinfer.moe_ep.modes import auto_layer

    owner = Mock()
    workspace = MoEEpAutoWorkspace(owner, None, [])
    first = Mock(destroy=Mock(side_effect=RuntimeError("plan teardown failed")))
    retained = Mock()
    workspace._plan, workspace._retained = first, [retained]
    failed, prototype, runtime = Mock(), Mock(), object()
    layer = SimpleNamespace(
        _destroyed=False,
        _workspaces=[workspace],
        _failed_plans=[failed],
        _prototypes={0: prototype},
        _runtime=runtime,
    )
    release = Mock()
    order = Mock()
    for name, item in (
        ("first", first),
        ("retained", retained),
        ("failed", failed),
        ("prototype", prototype),
    ):
        order.attach_mock(item.destroy, name)
    order.attach_mock(release, "release")
    monkeypatch.setattr(auto_layer, "finalize_moe_ep_runtime", release)
    with pytest.raises(RuntimeError, match="plan teardown failed"):
        auto_layer.MoEEpAutoLayer.destroy(layer)
    for item in (first, retained, failed, prototype):
        item.destroy.assert_called_once()
    release.assert_called_once_with(runtime)
    assert [call[0] for call in order.mock_calls] == [
        "first",
        "retained",
        "failed",
        "prototype",
        "release",
    ]
    assert workspace._destroyed and layer._destroyed
    workspace.destroy()
    auto_layer.MoEEpAutoLayer.destroy(layer)
    release.assert_called_once()


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
    from flashinfer.moe_ep.algo_knobs import (
        FleetAlgoKnobAllocator,
        HandleAlgoKnobUserStream,
    )

    config = AutoConfig((SplitConfig(),))
    assert record_key(config) == record_key(AutoConfig((SplitConfig(),)))
    assert "0x" not in str(plain(config))
    assert record_key(FleetAlgoKnobAllocator(torch_caching=True)) == record_key(
        FleetAlgoKnobAllocator(torch_caching=True)
    )
    for knob in (
        FleetAlgoKnobAllocator(alloc_fn=1, free_fn=2),
        FleetAlgoKnobAllocator(context=3),
        HandleAlgoKnobUserStream(4),
    ):
        with pytest.raises(ValueError, match="process-local"):
            record_key(knob)


def test_numerical_contract_covers_activation_recipe_and_combine(monkeypatch):
    import dataclasses
    from flashinfer.moe_ep import (
        FusedMoeKernelConfig,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.modes.selection import numerical_contract
    from flashinfer.moe_ep.backends.mega.kernel.sm107.automatic import (
        ep_tuning_contract,
    )
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.block_scaled import (
        Sm107BlockScaledMoeConfig,
    )
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config
    from tests.moe_ep.test_fused_moe_weights import _nvfp4_moe_config

    cfg = _nvfp4_moe_config(
        num_experts=8, local_num_experts=4, offset=0, intermediate=256
    )
    split = SplitConfig(kernel=FusedMoeKernelConfig(cfg))
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    contract = numerical_contract(split, None, None)
    native = Sm107BlockScaledMoeConfig(
        num_total_experts=8,
        max_tokens_per_rank=32,
        num_topk=4,
        hidden=256,
        intermediate=256,
        rank=0,
        world_size=2,
        quant_kind="nvfp4",
        gate_up_clamp=contract["gate_up_clamp"],
        apply_routing_weights_before_fc2=False,
    )
    backend = SimpleNamespace(_resolved_config=lambda _: native)
    prototype = SimpleNamespace(
        _kernel=SimpleNamespace(
            ep_tuning_contract=lambda fleet: ep_tuning_contract(backend, fleet)
        )
    )
    mega = MegaConfig(Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(256, 4))
    assert numerical_contract(mega, prototype, None) == contract
    for overrides in (
        {"apply_routing_weights_before_fc2": True},
        {"combine_dtype": "nvfp4"},
    ):
        native = dataclasses.replace(native, **overrides)
        assert numerical_contract(mega, prototype, None) != contract
        native = dataclasses.replace(
            native, apply_routing_weights_before_fc2=False, combine_dtype="bf16"
        )
    split.kernel.moe_config = dataclasses.replace(
        cfg, quant=dataclasses.replace(cfg.quant, per_token_scale=True)
    )
    assert numerical_contract(split, None, None) != contract
    per_token = numerical_contract(split, None, None)
    split.kernel.moe_config = dataclasses.replace(
        split.kernel.moe_config,
        quant=dataclasses.replace(
            split.kernel.moe_config.quant, nvfp4_4over6=NVFP44Over6Config()
        ),
    )
    assert numerical_contract(split, None, None) != per_token


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


@pytest.mark.parametrize("split_capture_failure", [False, True])
def test_outer_selects_full_plan_and_replays_without_measurement(split_capture_failure):
    from flashinfer.moe_ep import FleetParams, MoEEpTensors
    from flashinfer.moe_ep.core.selection import NAMESPACE
    from flashinfer.moe_ep.modes.auto_layer import _OuterProtocol
    from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer

    from flashinfer.moe_ep.modes.selection import SplitPlan

    split_trials = []

    class UncapturableSplit(SplitTrial, SplitPlan):
        def __init__(self):
            super().__init__()
            self.destroy = Mock()
            split_trials.append(self)

        def forward(self, tensors):
            raise CaptureUnsupported("split capture failed")

    class Plan:
        def __init__(self, index):
            self.index = index
            self.destroy = Mock()

        def selection(self):
            return {"kernel_variant": "inference", "knobs": {"tile": self.index}}

        def forward(self, tensors):
            return self.index, [9.0, 4.0][self.index]

        warmup = forward

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
        rejected_candidates={},
        _numerical_contract={},
    )
    for prototype in layer._prototypes.values():
        prototype._kernel = Mock()
        prototype._megakernel_config = SimpleNamespace(kernel_variant="inference")
    tensors = MoEEpTensors(
        torch.empty(2, 8), torch.zeros(2, 1, dtype=torch.int64), torch.ones(2, 1)
    )
    protocol = _OuterProtocol(layer, ws, tensors)
    calls = []
    force_flags = []

    def prepare(index, *, selected=None, fallback=False, force_tune=False):
        calls.append((index, selected))
        force_flags.append(force_tune)
        return (
            UncapturableSplit() if split_capture_failure and index == 1 else Plan(index)
        )

    protocol._new_plan = prepare
    chosen = 0 if split_capture_failure else 1
    measured = [0] if split_capture_failure else [0, 1]
    with autotune_v2(persistent_cache=False):
        tuner.clear_cache()
        result = tuner.choose_with_protocol(NAMESPACE, protocol)
        assert result["candidate"] == chosen
        assert c.measured == measured
        c.measured.clear()
        calls.clear()
        result = tuner.choose_with_protocol(NAMESPACE, protocol)
        assert result["candidate"] == chosen
        assert not c.measured
        assert calls == [
            (chosen, {"kernel_variant": "inference", "knobs": {"tile": chosen}})
        ]
        key, record = next(iter(tuner.get_protocol_records(NAMESPACE).items()))
        # The workspace bound stays fixed, but each live workload needs its
        # own search. Neither workload may suppress the other's tuning.
        protocol.tensors = MoEEpTensors(
            tensors.hidden_states[:1], tensors.topk_ids[:1], tensors.topk_weights[:1]
        )
        tuner.choose_with_protocol(NAMESPACE, protocol)
        assert c.measured == measured
        assert len(tuner.get_protocol_records(NAMESPACE)) == 2
        c.measured.clear()
        protocol.tensors = tensors
        # An invalid record must not be installed as a fixed plan.
        record["selected"] = {"wrong": True}
        tuner.publish_protocol_record(NAMESPACE, key, record, persist=False)
        tuner.choose_with_protocol(NAMESPACE, protocol)
        assert c.measured == measured
        c.measured.clear()
        force_flags.clear()
        tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
        assert c.measured == measured
        assert force_flags == [True, True, False]
        if split_capture_failure:
            assert "No split tactic completed" in layer.rejected_candidates[1]
            assert not layer._failed_plans
            for trial in split_trials:
                trial.destroy.assert_called_once()
            saved = layer._prototypes.pop(0)
            with pytest.raises(
                RuntimeError, match="No EP candidate completed measurement"
            ):
                tuner.choose_with_protocol(NAMESPACE, protocol, force_tune=True)
            layer._prototypes[0] = saved
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
    layer._eligible_runners.side_effect = ValueError(
        "num_tokens=1024 exceeds tune_max_num_tokens=512"
    )
    with pytest.raises(ValueError, match="tune_max_num_tokens=512"):
        layer._tuning_candidates(None, None)


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
    from flashinfer.moe_ep.core.selection import NoEligibleCandidate

    c = LocalCollective()

    def unsupported():
        raise ValueError("unsupported capacity")

    def failed():
        raise RuntimeError("device failure")

    with pytest.raises(NoEligibleCandidate, match="unsupported capacity"):
        c.prepare(unsupported, optional=True)
    with pytest.raises(RuntimeError, match="device failure"):
        c.prepare(failed, optional=True)


@pytest.mark.parametrize("peer_failure", [False, True])
@pytest.mark.parametrize("stage", ["quantize", "compute"])
def test_split_warmup_coordinates_local_stages(monkeypatch, peer_failure, stage):
    from flashinfer.moe_ep.modes import selection

    c = LocalCollective()
    c.gather = lambda value: [
        value,
        ("ValueError: peer setup failed", True) if peer_failure else value,
    ]
    monkeypatch.setattr(selection, "Collective", lambda _: c)
    plan = object.__new__(selection.SplitPlan)
    plan.layer = SimpleNamespace(_bootstrap=object())
    if peer_failure:
        with (
            pytest.raises(RuntimeError, match="peer setup failed"),
            plan._prepare_stage(stage),
        ):
            pass
    else:
        with (
            pytest.raises(RuntimeError, match="local setup failed"),
            plan._prepare_stage(stage),
        ):
            raise ValueError("local setup failed")


def test_ineligible_split_plan_is_skipped_before_discovery_forward(monkeypatch):
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.modes import auto_layer, selection
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    c = LocalCollective()
    monkeypatch.setattr(selection, "Collective", lambda _: c)
    plan = object.__new__(selection.SplitPlan)
    plan.output = torch.empty(1)
    plan.context = None
    plan.layer = SimpleNamespace(
        _bootstrap=object(),
        _uses_communication=lambda: True,
        stage_hook=None,
        create_graph_state=Mock(side_effect=ValueError("cannot be captured")),
        _forward_prepared=Mock(),
    )
    monkeypatch.setattr(auto_layer, "SplitPlan", Mock(return_value=plan))
    plan.destroy = Mock()
    layer = SimpleNamespace(
        _prototypes={0: Mock(spec=MoEEpSplitLayer)},
        _failed_plans=[],
        _collective=c,
        rejected_candidates={},
    )
    tensors = MoEEpTensors(torch.empty(1, 8), torch.zeros(1, 1), torch.ones(1, 1))
    protocol = auto_layer._OuterProtocol(
        layer, SimpleNamespace(fleet=None, row_bound=1), tensors
    )
    assert protocol._new_plan(0) is None
    assert "cannot be captured" in layer.rejected_candidates[0]
    plan.layer._forward_prepared.assert_not_called()
    plan.destroy.assert_called_once()
    assert not layer._failed_plans
    # A fresh plan can recover from an ineligible cached tactic/profile.
    plan.layer.create_graph_state = Mock()
    assert protocol._new_plan(0) is plan
    assert not layer.rejected_candidates


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


def test_workspace_bound_preserves_live_workload_and_other_layouts():
    import dataclasses
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.modes.auto_layer import MoEEpAutoLayer, _signature

    t = MoEEpTensors(
        torch.empty(8, 16, dtype=torch.bfloat16),
        torch.empty(8, 4, dtype=torch.int64),
        torch.empty(8, 4),
        runtime_max_tokens_per_rank=8,
    )
    short = dataclasses.replace(
        t,
        hidden_states=t.hidden_states[:1],
        topk_ids=t.topk_ids[:1],
        topk_weights=t.topk_weights[:1],
    )
    assert _signature(t, 8) == _signature(short, 8)
    assert _signature(t) != _signature(short)
    assert _signature(t, 8) != _signature(
        dataclasses.replace(t, topk_ids=t.topk_ids.int()), 8
    )
    assert _signature(t, 8) != _signature(
        dataclasses.replace(t, runtime_max_tokens_per_rank=None), 8
    )

    class Layer:
        _destroyed = False
        _device = torch.device("cpu")
        _validate_device = MoEEpAutoLayer._validate_device

    layer = Layer()
    workspace = MoEEpAutoWorkspace(layer, None, _signature(t, 8), row_bound=8)
    MoEEpAutoLayer._check(layer, short, workspace)
    with pytest.raises(ValueError, match="profile changed"):
        MoEEpAutoLayer._check(
            layer, dataclasses.replace(short, topk_ids=t.topk_ids), workspace
        )


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dsl_available", [False, True])
def test_split_fixed_rows_stage_and_mask_each_tail(
    monkeypatch, ids_dtype, dsl_available
):
    from flashinfer import cute_dsl
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.modes.selection import SplitPlan
    from flashinfer.moe_ep.core import bf16_stage

    monkeypatch.setattr(bf16_stage, "fused_bf16_stage", lambda *a: False)
    monkeypatch.setattr(cute_dsl, "is_cute_dsl_available", lambda: dsl_available)
    plan = object.__new__(SplitPlan)
    plan.output = torch.empty(8, 16, dtype=torch.bfloat16)
    plan.layer = SimpleNamespace(_uses_communication=lambda: False)
    plan._staging = None
    full = MoEEpTensors(
        torch.arange(128).reshape(8, 16).to(torch.bfloat16),
        torch.arange(32).reshape(8, 4).to(ids_dtype),
        torch.ones(8, 4),
    )
    assert plan._stage(full) is full
    staged = plan._stage(full, force=True)
    for live in (1, 7, 0, 8, 2):
        short = MoEEpTensors(
            full.hidden_states[:live], full.topk_ids[:live], full.topk_weights[:live]
        )
        result = plan._stage(short)
        assert result.num_tokens == 8
        assert result.topk_ids.dtype == ids_dtype
        torch.testing.assert_close(result.hidden_states[:live], short.hidden_states)
        torch.testing.assert_close(result.topk_ids[:live], short.topk_ids)
        torch.testing.assert_close(result.topk_weights[:live], short.topk_weights)
        if live < 8:
            assert result is staged
            assert (result.topk_ids[live:] == -1).all()
    plan.layer._uses_communication = lambda: True
    assert plan._stage(short) is short


def test_shared_stager_does_not_bootstrap_a_kernel_package():
    import subprocess
    import sys
    import textwrap

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                import types
                import torch
                from flashinfer import cute_dsl, moe_ep

                other = types.ModuleType("common")
                other.__file__ = "/another_kernel/common.py"
                sys.modules["common"] = other
                from flashinfer.moe_ep.modes.selection import SplitPlan

                cute_dsl.is_cute_dsl_available = lambda: True
                plan = object.__new__(SplitPlan)
                plan._staging = None
                plan.output = torch.empty(4, 8, dtype=torch.bfloat16)
                plan.layer = types.SimpleNamespace(_uses_communication=lambda: False)
                tensors = moe_ep.MoEEpTensors(
                    torch.ones(1, 8, dtype=torch.bfloat16),
                    torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1),
                )
                staged = plan._stage(tensors)
                assert (staged.topk_ids[1:] == -1).all()
                assert "flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe" not in sys.modules
                assert sys.modules["common"] is other
            """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_direct_split_rejects_zero_runtime_capacity_for_empty_batch(monkeypatch):
    from flashinfer.moe_ep import FleetParams, MoEEpTensors
    from flashinfer.moe_ep.modes import split_layer

    monkeypatch.setattr(split_layer, "ensure_bootstrap_dist_validated", lambda _: None)
    layer = SimpleNamespace(
        _destroyed=False, _bootstrap=object(), _fleet_params=FleetParams(8, 64, 8)
    )
    tensors = MoEEpTensors(
        torch.empty(0, 8),
        torch.empty(0, 4, dtype=torch.int64),
        torch.empty(0, 4),
        runtime_max_tokens_per_rank=0,
    )
    with pytest.raises(ValueError, match=r"max\(num_tokens, 1\)"):
        split_layer.MoEEpSplitLayer.forward(layer, tensors)


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
    plan.output = torch.empty(8, 4)
    plan.layer = SimpleNamespace(
        _bootstrap=object(),
        _workspace=object(),
        _megakernel_config=SimpleNamespace(knobs={}),
        _get_transformed_weights=lambda: (),
        _kernel=SimpleNamespace(ep_prepare_fixed=lambda *a: events.append("compile")),
        stage_inputs=lambda t: events.append("stage"),
        compute_staged=lambda **kwargs: events.append("launch"),
    )
    tensors = SimpleNamespace(output=None, num_tokens=3)
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
