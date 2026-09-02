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
    hopper_mxfp4_candidates,
    hopper_mxfp4_default_tactic,
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
        self._session = SimpleNamespace(
            config=cfg,
            captured=False,
            _process_group=object(),
        )
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

    def allocate(*args, **kwargs):
        candidate = _Buffer()
        candidate.allocator_args = args
        candidate.allocator_kwargs = kwargs
        made.append(candidate)
        return candidate

    monkeypatch.setattr(
        hopper_mxfp4_split,
        "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
        allocate,
    )
    monkeypatch.setattr(split_autotune.torch.cuda, "synchronize", mock.Mock())

    candidates = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    adapter = split_autotune._SplitTacticAdapter(source)
    adapter.apply_knobs(candidates[0])
    first = adapter.current
    assert first.x.copied_from is source.x
    assert first.x_sf.copied_from is source.x_sf
    assert first.topk_idx.copied_from is source.topk_idx
    assert first.topk_weights.copied_from is source.topk_weights
    assert first.topk_idx._sm90_mxfp4_staged_tokens == 37
    assert first.allocator_kwargs["split_k1_sm_count"] == candidates[0]["k1_sm_count"]
    assert first.allocator_kwargs["split_enable_iket"] is False
    assert (
        first.allocator_kwargs["routing_profile"]
        == source.session.config.routing_profile
    )

    adapter.apply_knobs(candidates[1])
    second = adapter.current
    assert first.destroy_calls == 1
    assert second is not first

    winner_session = second._session
    winner_roots = second._sym_roots
    adapter.commit()
    assert source.destroy_calls == 1
    assert source._session is winner_session
    assert source._sym_roots is winner_roots
    assert not source._destroyed
    assert second._destroyed
    assert second._sym_roots == []
    adapter.close()
    assert second.destroy_calls == 0


def test_split_autotune_uses_bucket_winner_first_and_split_cache(monkeypatch):
    source = _Buffer(
        rank=0,
        token_bucket=64,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    adapter = mock.Mock()
    adapter.current = object()
    monkeypatch.setattr(
        split_autotune, "_SplitTacticAdapter", mock.Mock(return_value=adapter)
    )
    record = mock.Mock(return_value="/tmp/cache.json")
    monkeypatch.setattr(knob_cache, "record_knobs", record)
    captured = {}

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["frontend"] = frontend
        captured["candidates"] = candidates
        captured["kwargs"] = kwargs
        winner = candidates[1]
        kwargs["on_winner"](winner, 0.00125)
        return winner

    monkeypatch.setattr(split_autotune, "autotune_knobs", fake_autotune)
    union = hopper_mxfp4_candidates(
        execution_mode="split",
        routing_profile=source.session.config.routing_profile,
    )
    winner = split_autotune.autotune_hopper_mxfp4_split_mega_moe(
        object(),
        object(),
        object(),
        source,
        num_tokens=37,
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
    adapter.commit.assert_called_once_with()
    adapter.close.assert_called_once_with()
    kwargs = record.call_args.kwargs
    assert "green_split" in kwargs["dtype"]
    assert kwargs["fp8_scale_mode"] == "mxfp4_hybrid"
    assert kwargs["world_size"] == 4
    assert kwargs["max_tokens"] == 64
    assert kwargs["gate_up_clamp"] == 10.0
    assert kwargs["routing_profile"] == source.session.config.routing_profile
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


def test_split_supplied_candidates_must_be_frozen_union_subset(monkeypatch):
    source = _Buffer(rank=1, token_bucket=64)
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

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["candidates"] = candidates
        return candidates[0]

    monkeypatch.setattr(split_autotune, "autotune_knobs", fake_autotune)
    assert (
        split_autotune.autotune_hopper_mxfp4_split_mega_moe(
            object(), object(), object(), source, candidates=supplied
        )
        == subset[0]
    )
    assert captured["candidates"] == subset
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
