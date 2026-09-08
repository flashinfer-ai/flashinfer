"""CPU-only contracts for MXFP4 split collective session rebuilding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    hopper_mxfp4_split,
    knob_cache,
    mxfp4_split_autotune as split_autotune,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    hopper_mxfp4_cache_provenance_sha256,
    hopper_mxfp4_candidates,
    hopper_mxfp4_default_tactic,
    hopper_mxfp4_ordered_candidates,
    hopper_mxfp4_tuning_provenance,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


@pytest.fixture(autouse=True)
def _allow_unit_test_device(monkeypatch):
    monkeypatch.setattr(
        split_autotune,
        "require_hopper_mxfp4_tuning_device",
        lambda: None,
    )


class _Tensor:
    def __init__(self, name):
        self.name = name
        self.copied_from = None

    def copy_(self, other):
        self.copied_from = other
        return self


class _Session:
    def __init__(self, config):
        self.config = config
        self.captured = False
        self._process_group = object()
        self.destroy_calls = 0
        self.prepare_calls = []

    def prepare_compile_only(self, inputs):
        self.prepare_calls.append(inputs)

    def destroy(self):
        self.destroy_calls += 1


class _Buffer:
    def __init__(
        self,
        *,
        rank=0,
        token_bucket=64,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    ):
        cfg = SimpleNamespace(
            rank=rank,
            world_size=4,
            num_tokens_per_rank=token_bucket,
            num_topk=6,
            num_total_experts=384,
            hidden=7168,
            intermediate=3072,
            clc_bundle_size=None,
            flag_batch=1,
            epi_flag_batch=(2, 4),
            gate_up_clamp=10.0,
            routing_profile=routing_profile,
        )
        self.num_total_experts = cfg.num_total_experts
        self.num_max_tokens = cfg.num_tokens_per_rank
        self.num_topk = cfg.num_topk
        self.hidden = cfg.hidden
        self.intermediate = cfg.intermediate
        self.rank = rank
        self.world_size = cfg.world_size
        self.x = _Tensor("x")
        self.x_sf = _Tensor("x_sf")
        self.topk_idx = _Tensor("topk_idx")
        self.topk_idx._sm90_mxfp4_staged_tokens = 37
        self.topk_weights = _Tensor("topk_weights")
        self.output_activation = _Tensor("output")
        self._session = _Session(cfg)
        self.session = self._session
        self._sym_roots = [object()]
        self._destroyed = False
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        self._destroyed = True
        self._sym_roots = []


def test_split_adapter_allocates_every_candidate_fresh_and_commits(monkeypatch):
    source = _Buffer()
    made = []

    def allocate(actual_source, tactic):
        candidate = _Buffer()
        candidate.x = actual_source.x
        candidate.x_sf = actual_source.x_sf
        candidate.topk_idx = actual_source.topk_idx
        candidate.topk_weights = actual_source.topk_weights
        candidate.output_activation = actual_source.output_activation
        candidate._sym_roots = []
        candidate.tactic = tactic
        made.append(candidate)
        return candidate

    monkeypatch.setattr(
        split_autotune._SplitTacticAdapter,
        "_make_candidate",
        staticmethod(allocate),
    )
    monkeypatch.setattr(split_autotune.torch.cuda, "synchronize", mock.Mock())
    free_sym_tensor = mock.Mock()
    monkeypatch.setattr(hopper_mxfp4_split, "free_sym_tensor", free_sym_tensor)

    candidates = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    adapter = split_autotune._SplitTacticAdapter(source)
    adapter.apply_knobs(candidates[0])
    first = adapter.current
    assert first.x is source.x
    assert first.x_sf is source.x_sf
    assert first.topk_idx is source.topk_idx
    assert first.topk_weights is source.topk_weights
    assert first.topk_idx._sm90_mxfp4_staged_tokens == 37
    assert first.tactic == candidates[0]

    adapter.discard()
    adapter.apply_knobs(candidates[1])
    second = adapter.current
    assert first.destroy_calls == 1
    assert second is not first

    free_sym_tensor.reset_mock()
    winner_session = second._session
    source_roots = source._sym_roots
    retired_session = source._session
    retired_roots = source._sym_roots
    adapter.commit()
    # Commit only swaps ownership. The old source stays alive until the
    # collective commit-success gate has completed on every rank.
    assert source.destroy_calls == 0
    assert retired_session.destroy_calls == 0
    assert retired_roots
    assert source._session is winner_session
    assert source._sym_roots is source_roots
    assert not source._destroyed
    assert second._destroyed
    assert second._sym_roots == []
    adapter.finalize_commit()
    assert retired_session.destroy_calls == 1
    free_sym_tensor.assert_not_called()
    adapter.close()
    assert second.destroy_calls == 0


def test_split_adapter_partial_prepare_waits_for_collective_discard(monkeypatch):
    source = _Buffer()
    candidate = _Buffer()
    candidate._sym_roots = []
    candidate._session.prepare_compile_only = mock.Mock(
        side_effect=ValueError("prepare rejected")
    )
    monkeypatch.setattr(
        split_autotune._SplitTacticAdapter,
        "_make_candidate",
        staticmethod(lambda source, tactic: candidate),
    )
    monkeypatch.setattr(
        hopper_mxfp4_split, "_split_inputs", mock.Mock(return_value=object())
    )
    tactic = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )[0]
    adapter = split_autotune._SplitTacticAdapter(source)

    adapter.apply_knobs(tactic)
    with pytest.raises(ValueError, match="prepare rejected"):
        adapter.prepare(object(), object())

    assert adapter.current is candidate
    assert candidate.destroy_calls == 0
    adapter.discard()
    assert candidate.destroy_calls == 1


def test_split_full_union_uses_bucket_winner_first_and_split_cache(monkeypatch):
    source = _Buffer(
        rank=0,
        token_bucket=64,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    inputs = SimpleNamespace(
        activation=SimpleNamespace(shape=(source.num_max_tokens, source.hidden))
    )
    source.session._input_validator = mock.Mock()
    split_inputs = mock.Mock(return_value=inputs)
    monkeypatch.setattr(hopper_mxfp4_split, "_split_inputs", split_inputs)
    y = SimpleNamespace(
        shape=(37, source.hidden),
        dtype=split_autotune.torch.bfloat16,
    )
    transformed_l1 = object()
    transformed_l2 = object()
    adapter = mock.Mock()
    adapter.current = object()
    monkeypatch.setattr(
        split_autotune, "_SplitTacticAdapter", mock.Mock(return_value=adapter)
    )
    record = mock.Mock(return_value="/tmp/cache.json")
    monkeypatch.setattr(knob_cache, "record_knobs", record)
    captured = {}
    full_candidates = hopper_mxfp4_ordered_candidates(
        source.session.config.num_tokens_per_rank,
        execution_mode="split",
        hidden=source.session.config.hidden,
        intermediate=source.session.config.intermediate,
        routing_profile=source.session.config.routing_profile,
    )

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["frontend"] = frontend
        captured["candidates"] = candidates
        captured["kwargs"] = kwargs
        winner = candidates[1]
        kwargs["preflight"]()
        kwargs["prepare_candidate"]()
        kwargs["commit_winner"]()
        kwargs["finalize_winner"]()
        kwargs["on_winner"](winner, 0.00125)
        return winner

    monkeypatch.setattr(split_autotune, "autotune_knobs", fake_autotune)
    union = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    winner = split_autotune.autotune_hopper_mxfp4_split_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        source,
        num_tokens=37,
        candidates=full_candidates,
    )

    default = hopper_mxfp4_default_tactic(
        64,
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    assert captured["candidates"][0] == default
    assert sorted(captured["candidates"], key=repr) == sorted(union, key=repr)
    assert captured["kwargs"]["process_group"] is source.session._process_group
    assert captured["kwargs"]["expected_world_size"] == 4
    assert winner == captured["candidates"][1]
    split_inputs.assert_called_once_with(source, transformed_l1, transformed_l2)
    source.session._input_validator._validate_inputs.assert_called_once_with(
        inputs,
        num_tokens=source.num_max_tokens,
    )
    adapter.prepare.assert_called_once_with(
        transformed_l1,
        transformed_l2,
        inputs=inputs,
    )
    adapter.commit.assert_called_once_with()
    adapter.finalize_commit.assert_called_once_with()
    adapter.close.assert_called_once_with()
    kwargs = record.call_args.kwargs
    assert "green_split" in kwargs["dtype"]
    assert kwargs["fp8_scale_mode"] == "mxfp4_hybrid"
    assert kwargs["world_size"] == 4
    assert kwargs["max_tokens"] == 64
    assert kwargs["gate_up_clamp"] == 10.0
    assert kwargs["routing_profile"] == source.session.config.routing_profile
    assert kwargs["tuning_provenance_sha256"] == (
        hopper_mxfp4_cache_provenance_sha256(
            execution_mode="split",
            routing_profile=source.session.config.routing_profile,
        )
    )
    assert kwargs["p50_us"] == pytest.approx(1250.0)
    provenance = hopper_mxfp4_tuning_provenance(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    assert provenance["runtime_manifest_sha256"] in kwargs["source"]


def test_split_autotune_failure_closes_candidate_without_committing(monkeypatch):
    source = _Buffer()
    adapter = mock.Mock()
    monkeypatch.setattr(
        split_autotune, "_SplitTacticAdapter", mock.Mock(return_value=adapter)
    )

    def fail(*args, **kwargs):
        raise RuntimeError("candidate sweep failed")

    monkeypatch.setattr(split_autotune, "autotune_knobs", fail)
    with pytest.raises(RuntimeError, match="candidate sweep failed"):
        split_autotune.autotune_hopper_mxfp4_split_mega_moe(
            object(),
            object(),
            object(),
            source,
        )
    adapter.commit.assert_not_called()
    adapter.close.assert_called_once_with()


def test_split_adapter_rollback_restores_source_before_discard(monkeypatch):
    source = _Buffer()
    candidate = _Buffer()
    monkeypatch.setattr(
        split_autotune._SplitTacticAdapter,
        "_make_candidate",
        staticmethod(lambda source, tactic: candidate),
    )
    monkeypatch.setattr(split_autotune.torch.cuda, "synchronize", mock.Mock())
    monkeypatch.setattr(hopper_mxfp4_split, "free_sym_tensor", mock.Mock())
    old_session = source._session
    old_roots = source._sym_roots

    tactic = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )[0]
    adapter = split_autotune._SplitTacticAdapter(source)
    adapter.apply_knobs(tactic)
    adapter.commit()
    adapter.rollback()

    assert source._session is old_session
    assert source._sym_roots is old_roots
    assert old_session.destroy_calls == 0
    assert candidate.destroy_calls == 0
    assert candidate._session.destroy_calls == 1
    adapter.close()


def test_split_supplied_candidates_must_be_frozen_union_subset(monkeypatch):
    source = _Buffer(rank=0, token_bucket=64)
    adapter = mock.Mock()
    monkeypatch.setattr(
        split_autotune, "_SplitTacticAdapter", mock.Mock(return_value=adapter)
    )
    union = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    subset = [union[2], union[0]]
    supplied = [
        {
            **subset[0],
            "k1_mma_tiler_mnk": list(subset[0]["k1_mma_tiler_mnk"]),
            "k1_cluster_shape_mnk": list(subset[0]["k1_cluster_shape_mnk"]),
            "k2_mma_tiler_mnk": list(subset[0]["k2_mma_tiler_mnk"]),
            "k2_cluster_shape_mnk": list(subset[0]["k2_cluster_shape_mnk"]),
        },
        subset[1],
    ]
    captured = {}
    record = mock.Mock()
    monkeypatch.setattr(knob_cache, "record_knobs", record)

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["candidates"] = candidates
        kwargs["on_winner"](candidates[0], 0.0005)
        return candidates[0]

    monkeypatch.setattr(split_autotune, "autotune_knobs", fake_autotune)
    assert (
        split_autotune.autotune_hopper_mxfp4_split_mega_moe(
            object(), object(), object(), source, candidates=supplied
        )
        == subset[0]
    )
    assert captured["candidates"] == subset
    record.assert_not_called()
    adapter.close.assert_called_once_with()

    outside = {**union[0], "k1_group_hint": 999999}
    with pytest.raises(ValueError, match="outside the frozen manifest candidate union"):
        split_autotune.autotune_hopper_mxfp4_split_mega_moe(
            object(), object(), object(), source, candidates=[outside]
        )

    with pytest.raises(ValueError, match="candidates must be unique"):
        split_autotune.autotune_hopper_mxfp4_split_mega_moe(
            object(),
            object(),
            object(),
            source,
            candidates=[union[0], union[0]],
        )
