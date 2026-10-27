"""GDN arithmetic constraints survive tuning, serialization, and preparation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.device import DetectedDevice
from b12x.preparation.session import PreparationJob, _declaration_key
from b12x.sequence import gdn_prefill, kda_prefill
from b12x.sequence._shared.delta_prefill.preparation import _caps_from_query
from b12x.sequence.gdn_prefill import _impl, _tuning
from b12x.sequence.kda_prefill import _impl as kda_impl, _tuning as kda_tuning


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
CAPACITIES = (1, 2, 4, 16, 64, 128, 256, 512)


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    def reject_cuda():
        pytest.fail("recurrence policy metadata checks must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", reject_cuda)


def _caps(tokens=512, **kwargs):
    return gdn_prefill.Caps(
        device="cuda:0",
        max_tokens=tokens,
        max_seqs=1,
        max_state_slots=46,
        key_heads=16,
        value_heads=48,
        checkpoint_export=True,
        null_state_index=0,
        **kwargs,
    )


def _query(tokens=512, *, policy="auto", exhaustive=False):
    plan = gdn_prefill.plan(_caps(tokens, recurrence_policy=policy))
    return replace(plan.query, exhaustive=exhaustive)


@pytest.mark.parametrize("tokens", CAPACITIES)
@pytest.mark.parametrize("exhaustive", (False, True))
def test_sequential_policy_keeps_every_legal_performance_choice(tokens, exhaustive):
    auto = _query(tokens, exhaustive=exhaustive)
    restricted = replace(auto, recurrence_policy="sequential_k1")
    full = {config for _, config in _tuning.TUNING.choices(auto, DEVICE)}
    selected = {config for _, config in _tuning.TUNING.choices(restricted, DEVICE)}
    assert selected == {
        config
        for config in full
        if config.algorithm == "sequential" and config.k_split == 1
    }
    assert len({config.v_split for config in selected}) > 1
    assert len({config.stages for config in selected}) > 1
    assert len({config.window_tiles for config in selected}) > 1
    assert any(config.k_split > 1 for config in full)
    assert any(config.algorithm == "chunk_parallel" for config in full) == (
        exhaustive or tokens > 128
    )
    for config in selected:
        layout = _impl._materialize_layout(
            _caps(tokens, recurrence_policy="sequential_k1"), config
        )
        assert layout.parallel is None
        assert layout.k_split == 1
        assert (layout.v_split, layout.stages, layout.window_tiles) == (
            config.v_split,
            config.stages,
            config.window_tiles,
        )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("tokens", CAPACITIES)
def test_default_policy_keeps_default_configuration(tokens):
    implicit = gdn_prefill.plan(_caps(tokens))
    explicit = gdn_prefill.plan(_caps(tokens, recurrence_policy="auto"))
    restricted = gdn_prefill.plan(_caps(tokens, recurrence_policy="sequential_k1"))
    assert implicit.query == explicit.query
    defaults = [
        _tuning.TUNING.configure(plan.query, device=DEVICE).default
        for plan in (implicit, explicit, restricted)
    ]
    assert defaults[0] == defaults[1] == defaults[2]
    assert defaults[0] == _tuning.GdnPrefillConfig(
        v_split=64,
        k_split=1,
        stages=3,
        window_tiles=_tuning.default_window_tiles(48, tokens, 1),
    )


@pytest.mark.parametrize("exhaustive", (False, True))
def test_incompatible_overrides_and_assignments_rejected(exhaustive):
    query = _query(policy="sequential_k1", exhaustive=exhaustive)
    auto = replace(query, recurrence_policy="auto")
    candidates = tuple(_tuning.TUNING.choices(auto, DEVICE))
    incompatible = [
        next(pair for pair in candidates if pair[1].k_split == 2),
        next(pair for pair in candidates if pair[1].algorithm == "chunk_parallel"),
    ]
    for assignment, config in incompatible:
        assert (
            _tuning.TUNING.configure(auto, device=DEVICE, override=config).pinned
            == config
        )
        with pytest.raises(ValueError, match="recurrence_policy"):
            _tuning.TUNING.configure(query, device=DEVICE, override=config)
        with pytest.raises(ValueError, match="correctness predicates"):
            _tuning.TUNING.lower(query, DEVICE, assignment)


@pytest.mark.parametrize("policy", ("auto", "sequential_k1"))
def test_policy_reaches_serialized_query_and_compiler_caps(policy):
    caps = _caps(recurrence_policy=policy)
    plan = gdn_prefill.plan(caps)
    payload = _tuning.TUNING.encode_query(plan.query)
    assert payload["recurrence_policy"] == policy
    restored = _tuning.GdnPrefillQuery(**plan.query.to_dict())
    assert restored == plan.query
    assert _caps_from_query(_impl, restored, 0, is_gdn=True) == caps
    config = _tuning.TUNING.configure(restored, device=DEVICE).default
    jobs = plan._compile_jobs(config, DetectedDevice(0, DEVICE))
    assert len(jobs) == 1
    assert jobs[0].args[1]["recurrence_policy"] == policy
    assert _tuning.TUNING.decode_config(FrozenMapping(jobs[0].args[2])) == config


def test_policy_separates_coalescing_and_persisted_selection_keys():
    plans = [
        gdn_prefill.plan(_caps(recurrence_policy=policy))
        for policy in ("auto", "sequential_k1")
    ]
    assert _declaration_key(plans[0]) != _declaration_key(plans[1])
    job = object.__new__(PreparationJob)
    keys = []
    for plan in plans:
        configuration = _tuning.TUNING.configure(plan.query, device=DEVICE)
        request = plan.request(name="gdn", prepare_call=lambda state: None)
        obligation = SimpleNamespace(
            request=request,
            configuration=configuration,
            candidate_program_identity="shared-programs",
        )
        keys.append(job._choice_key(obligation, {}))
    assert keys[0] != keys[1]


@pytest.mark.parametrize("policy", (None, False, 1, "sequential", "invalid"))
def test_invalid_policy_rejected_at_caps_and_query(policy):
    with pytest.raises(ValueError, match="recurrence_policy"):
        _caps(recurrence_policy=policy)
    with pytest.raises(ValueError, match="recurrence_policy"):
        _tuning.TUNING.configure(
            replace(_query(), recurrence_policy=policy), device=DEVICE
        )


def test_kda_caps_and_query_contract_do_not_gain_gdn_policy():
    caps = kda_prefill.Caps(
        device="cuda:0",
        max_tokens=512,
        max_seqs=1,
        max_state_slots=46,
        heads=48,
        checkpoint_export=True,
        null_state_index=0,
    )
    plan = kda_prefill.plan(caps)
    assert _caps_from_query(kda_impl, plan.query, 0, is_gdn=False) == caps
    assert "recurrence_policy" not in plan.query.to_dict()
    assert "recurrence_policy" not in kda_tuning.TUNING.encode_query(plan.query)
    assert (
        kda_tuning.TUNING.query_schema_version,
        kda_tuning.TUNING.config_schema_version,
        kda_tuning.TUNING.candidate_contract_version,
    ) == (22, 22, 3)


@pytest.mark.parametrize("policy", ("auto", "sequential_k1"))
def test_fp32_precision_and_recurrence_policy_survive_compiler_roundtrip(policy):
    caps = _caps(intermediate_dtype=torch.float32, recurrence_policy=policy)
    plan = gdn_prefill.plan(caps)
    query = plan.query
    assert query.intermediate_dtype == "float32"
    assert query.recurrence_policy == policy
    assert _caps_from_query(_impl, query, 0, is_gdn=True) == caps
    choices = tuple(_tuning.TUNING.choices(query, DEVICE))
    assert len(choices) == 1
    config = choices[0][1]
    assert config == _tuning.TUNING.configure(query, device=DEVICE).default
    assert config.algorithm == "sequential" and config.k_split == 1
    assert _impl._materialize_layout(caps, config).workspace_windows == 0


def test_precision_and_policy_have_independent_preparation_identities():
    plans = [
        gdn_prefill.plan(_caps(intermediate_dtype=precision, recurrence_policy=policy))
        for precision in (torch.bfloat16, torch.float32)
        for policy in ("auto", "sequential_k1")
    ]
    assert len({_declaration_key(plan) for plan in plans}) == 4
    assert _tuning.TUNING.query_schema_version == 24
