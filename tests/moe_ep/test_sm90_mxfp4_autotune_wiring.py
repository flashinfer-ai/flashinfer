"""CPU contracts for fused/split MXFP4 cache and backend autotune wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90PullMxfp4MegaKernelBackend,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    autotune as autotune_module,
    hopper_mxfp4,
    hopper_mxfp4_split,
    knob_cache,
    mxfp4_tuner,
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
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        lambda: None,
    )


def _fleet(tokens=64):
    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=tokens,
        token_hidden_size=128,
    )


def _bound(config):
    backend = Sm90PullMxfp4MegaKernelBackend(config)
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=2, rank=0, device=0))
    backend._ep_comm_group = object()
    return backend


def test_cache_dtype_identities_spell_both_operand_formats():
    fused = hopper_mxfp4._MXFP4_TUNING_DTYPE_ID
    split = hopper_mxfp4_split._SPLIT_TUNING_IDENTITY
    assert fused == (
        "sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_humming_v1_"
        "fold_m64_k128_gateup8_packedk2_residual64_swapab_fused"
    )
    assert split == fused.removesuffix("fused") + "green_split_v1"
    assert fused != split


def test_split_none_is_cache_then_manifest_heuristic_selector():
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
    )
    assert config.knobs is None
    assert config.split_k1_sm_count is None
    assert config.split_k2_sm_count is None


def test_split_complete_knobs_and_auto_are_mode_specific():
    tactic = hopper_mxfp4_default_tactic(64, execution_mode="split")
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
        knobs=tactic,
    )
    assert config.knobs == tactic

    with pytest.raises(ValueError, match="complete split session tactic"):
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            execution_mode="split",
            knobs={"k1_sm_count": 80, "k2_sm_count": 52},
        )
    with pytest.raises(ValueError, match="complete split session tactic"):
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            execution_mode="split",
            knobs=hopper_mxfp4_default_tactic(64, execution_mode="fused"),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            execution_mode="split",
            knobs="auto",
            split_k1_sm_count=80,
            split_k2_sm_count=52,
        )


def test_split_resolver_uses_separate_identity_and_strict_cache(monkeypatch):
    lookup = mock.Mock(return_value=None)
    monkeypatch.setattr(knob_cache, "lookup_knobs", lookup)
    kwargs = dict(
        world_size=4,
        hidden=7168,
        intermediate=3072,
        num_total_experts=384,
        num_topk=6,
        num_max_tokens=64,
    )
    got = hopper_mxfp4_split._resolve_mxfp4_split_tactic(None, **kwargs)
    assert got == hopper_mxfp4_default_tactic(64, execution_mode="split")
    identity = lookup.call_args.kwargs["dtype"]
    assert identity == hopper_mxfp4_split._SPLIT_TUNING_IDENTITY
    assert identity != hopper_mxfp4._MXFP4_TUNING_DTYPE_ID
    assert "green_split" in identity
    assert (
        lookup.call_args.kwargs["routing_profile"]
        == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )

    lookup.return_value = hopper_mxfp4_default_tactic(64, execution_mode="fused")
    with pytest.raises(ValueError, match="mxfp4_split"):
        hopper_mxfp4_split._resolve_mxfp4_split_tactic(None, **kwargs)


def test_persistent_cache_never_crosses_fused_and_split(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FLASHINFER_MOE_EP_KNOB_CACHE",
        str(tmp_path / "cache.json"),
    )
    key = dict(
        fp8_scale_mode="mxfp4_hybrid",
        world_size=4,
        hidden=7168,
        intermediate=3072,
        num_experts=384,
        topk=6,
        max_tokens=64,
        device="NVIDIA H200",
    )
    split = hopper_mxfp4_default_tactic(64, execution_mode="split")
    assert knob_cache.record_knobs(
        split,
        dtype=hopper_mxfp4_split._SPLIT_TUNING_IDENTITY,
        **key,
    )
    assert (
        knob_cache.lookup_knobs(
            dtype=hopper_mxfp4_split._SPLIT_TUNING_IDENTITY,
            **key,
        )
        == split
    )
    assert (
        knob_cache.lookup_knobs(
            dtype=hopper_mxfp4._MXFP4_TUNING_DTYPE_ID,
            **key,
        )
        is None
    )
    # Ordinary SM90 FP8 shares the physical JSON file, but neither its dtype
    # nor its scale ABI may consume an MXFP4 split entry.
    fp8_key = {**key, "fp8_scale_mode": "per_tensor"}
    assert (
        knob_cache.lookup_knobs(
            dtype="fp8_e4m3",
            **fp8_key,
        )
        is None
    )


def test_split_backend_allocation_uses_resolved_complete_tactic(monkeypatch):
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
    )
    backend = _bound(config)
    tactic = hopper_mxfp4_default_tactic(64, execution_mode="split")
    resolver = mock.Mock(return_value=tactic)
    allocator = mock.Mock(return_value=object())
    monkeypatch.setattr(
        hopper_mxfp4_split,
        "_resolve_mxfp4_split_tactic",
        resolver,
    )
    monkeypatch.setattr(
        sm90_mega,
        "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
        allocator,
    )

    backend._allocate_workspace(_fleet())
    resolver.assert_called_once()
    assert (
        resolver.call_args.kwargs["routing_profile"]
        == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )
    kwargs = allocator.call_args.kwargs
    assert kwargs["split_k1_mma_tiler_mnk"] == tactic["k1_mma_tiler_mnk"]
    assert kwargs["split_k2_mma_tiler_mnk"] == tactic["k2_mma_tiler_mnk"]
    assert kwargs["split_k1_sm_count"] == tactic["k1_sm_count"]
    assert kwargs["split_k2_sm_count"] == tactic["k2_sm_count"]
    assert kwargs["split_counter_epoch_banks"] == tactic["counter_epoch_banks"]
    assert kwargs["split_graph_variant"] == tactic["graph_variant"]
    assert kwargs["split_enable_iket"] == tactic["enable_iket"]
    assert kwargs["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION


def test_split_backend_auto_calls_only_split_tuner_once(monkeypatch):
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
        knobs="auto",
    )
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = _bound(config)
    split_tune = mock.Mock(
        return_value=hopper_mxfp4_candidates(execution_mode="split")[0]
    )
    split_launch = mock.Mock(return_value=None)
    monkeypatch.setattr(
        sm90_mega,
        "autotune_hopper_mxfp4_split_mega_moe",
        split_tune,
    )
    monkeypatch.setattr(
        sm90_mega,
        "hopper_mxfp4_split_mega_moe",
        split_launch,
    )
    monkeypatch.setattr(
        sm90_mega,
        "autotune_hopper_mxfp4_mega_moe",
        mock.Mock(side_effect=AssertionError("fused autotune fallback")),
    )
    output = torch.empty((3, 128), dtype=torch.bfloat16)
    transformed = ("l1", "l2")

    assert backend.compute(object(), transformed, output=output) is output
    split_tune.assert_called_once()
    split_launch.assert_called_once()
    assert not backend._autotune_pending
    assert backend.compute(object(), transformed, output=output) is output
    split_tune.assert_called_once()
    assert split_launch.call_count == 2


def test_fused_autotune_records_only_fused_identity_and_manifest(monkeypatch):
    cfg = SimpleNamespace(
        rank=0,
        world_size=4,
        num_tokens_per_rank=64,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    buffer = SimpleNamespace(_frontend=SimpleNamespace(config=cfg))
    candidate = hopper_mxfp4_default_tactic(
        64,
        execution_mode="fused",
        routing_profile=cfg.routing_profile,
    )
    ep_group = object()
    record = mock.Mock(return_value="/tmp/cache.json")
    monkeypatch.setattr(knob_cache, "record_knobs", record)

    def fake_autotune(frontend, launch, candidates, **kwargs):
        assert kwargs["process_group"] is ep_group
        assert kwargs["expected_world_size"] == 4
        assert candidates == [candidate]
        kwargs["on_winner"](candidate, 0.00075)
        return candidate

    monkeypatch.setattr(autotune_module, "autotune_knobs", fake_autotune)
    winner = autotune_module.autotune_hopper_mxfp4_mega_moe(
        object(),
        object(),
        object(),
        buffer,
        num_tokens=37,
        process_group=ep_group,
        candidates=[candidate],
    )

    assert winner == candidate
    kwargs = record.call_args.kwargs
    assert kwargs["dtype"] == hopper_mxfp4._MXFP4_TUNING_DTYPE_ID
    assert kwargs["dtype"] != hopper_mxfp4_split._SPLIT_TUNING_IDENTITY
    assert "fused" in kwargs["dtype"]
    assert kwargs["p50_us"] == pytest.approx(750.0)
    assert kwargs["gate_up_clamp"] == 10.0
    assert kwargs["routing_profile"] == cfg.routing_profile
    provenance = hopper_mxfp4_tuning_provenance(
        execution_mode="fused",
        routing_profile=cfg.routing_profile,
    )
    assert provenance["runtime_manifest_sha256"] in kwargs["source"]


def test_fused_supplied_candidates_must_be_frozen_union_subset(monkeypatch):
    cfg = SimpleNamespace(
        rank=1,
        world_size=4,
        num_tokens_per_rank=64,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    buffer = SimpleNamespace(_frontend=SimpleNamespace(config=cfg))
    union = hopper_mxfp4_candidates(
        execution_mode="fused",
        routing_profile=cfg.routing_profile,
    )
    subset = [union[2], union[0]]
    supplied = [
        {
            **subset[0],
            "mma_tiler_mnk": list(subset[0]["mma_tiler_mnk"]),
            "cluster_shape_mnk": list(subset[0]["cluster_shape_mnk"]),
        },
        subset[1],
    ]
    captured = {}

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["candidates"] = candidates
        return candidates[0]

    monkeypatch.setattr(autotune_module, "autotune_knobs", fake_autotune)
    assert (
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(), object(), object(), buffer, candidates=supplied
        )
        == subset[0]
    )
    assert captured["candidates"] == subset

    outside = {**union[0], "group_hint": 999999}
    with pytest.raises(ValueError, match="outside the frozen manifest candidate union"):
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(), object(), object(), buffer, candidates=[outside]
        )

    with pytest.raises(ValueError, match="candidates must be unique"):
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(),
            object(),
            object(),
            buffer,
            candidates=[union[0], union[0]],
        )
